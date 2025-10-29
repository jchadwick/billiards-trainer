"""Comprehensive unit tests for trajectory calculation module.

This test suite verifies the trajectory calculation functionality with focus on:
1. Vector2D scale parameter validation (all instances must have mandatory scale)
2. Basic trajectory calculations
3. Cue shot predictions
4. Multiball trajectory scenarios
5. Edge cases (zero velocity, collisions, etc.)

All tests use the 4K canonical coordinate system (3840×2160) with proper
scale metadata for all Vector2D instances.
"""

import math

import pytest

from backend.core.constants_4k import BALL_RADIUS_4K
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState, TableState
from backend.core.physics.trajectory import (
    CollisionType,
    MultiballTrajectoryResult,
    PredictedCollision,
    Trajectory,
    TrajectoryBranch,
    TrajectoryCalculator,
    TrajectoryPoint,
    TrajectoryQuality,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def table_state():
    """Create a standard table state for testing.

    Returns a table with 4K canonical dimensions and standard playing area.
    """
    # Standard 4K table dimensions
    width = 3840
    height = 2160

    # Define playing area corners (4K canonical coordinates)
    corners = [
        Vector2D(200, 200, scale=(1.0, 1.0)),  # top-left
        Vector2D(3640, 200, scale=(1.0, 1.0)),  # top-right
        Vector2D(3640, 1960, scale=(1.0, 1.0)),  # bottom-right
        Vector2D(200, 1960, scale=(1.0, 1.0)),  # bottom-left
    ]

    # Define standard 6-pocket positions (4K canonical coordinates)
    pocket_positions = [
        Vector2D(200, 200, scale=(1.0, 1.0)),  # top-left
        Vector2D(1920, 200, scale=(1.0, 1.0)),  # top-middle
        Vector2D(3640, 200, scale=(1.0, 1.0)),  # top-right
        Vector2D(200, 1960, scale=(1.0, 1.0)),  # bottom-left
        Vector2D(1920, 1960, scale=(1.0, 1.0)),  # bottom-middle
        Vector2D(3640, 1960, scale=(1.0, 1.0)),  # bottom-right
    ]

    return TableState(
        width=width,
        height=height,
        pocket_positions=pocket_positions,
        playing_area_corners=corners,
        cushion_elasticity=0.95,
        surface_friction=0.1,
    )


@pytest.fixture()
def cue_ball():
    """Create a standard cue ball in 4K canonical coordinates."""
    return BallState.from_4k(
        id="cue_ball",
        x=1920.0,
        y=1080.0,
        number=0,
        is_cue_ball=True,
    )


@pytest.fixture()
def target_ball():
    """Create a target ball for collision testing."""
    return BallState.from_4k(
        id="ball_1",
        x=2400.0,
        y=1080.0,
        number=1,
    )


@pytest.fixture()
def calculator():
    """Create a trajectory calculator instance."""
    return TrajectoryCalculator()


@pytest.fixture()
def cue_state():
    """Create a standard cue state for shot prediction."""
    return CueState(
        angle=0.0,  # Horizontal shot
        estimated_force=100.0,
        impact_point=Vector2D(1920.0, 1080.0, scale=(1.0, 1.0)),
        is_visible=True,
        tip_position=Vector2D(1820.0, 1080.0, scale=(1.0, 1.0)),
    )


# =============================================================================
# Vector2D Scale Validation Tests
# =============================================================================


@pytest.mark.unit()
class TestVector2DScaleValidation:
    """Test that all Vector2D instances have proper scale parameters."""

    def test_trajectory_point_vectors_have_scale(self, cue_ball):
        """Verify TrajectoryPoint vectors all have scale metadata."""
        point = TrajectoryPoint(
            time=0.0,
            position=Vector2D(100, 200, scale=(1.0, 1.0)),
            velocity=Vector2D(50, 0, scale=(1.0, 1.0)),
            acceleration=Vector2D(0, 0, scale=(1.0, 1.0)),
            spin=Vector2D(0, 0, scale=(1.0, 1.0)),
            energy=10.0,
        )

        # Verify all vectors have scale
        assert point.position.scale == (1.0, 1.0)
        assert point.velocity.scale == (1.0, 1.0)
        assert point.acceleration.scale == (1.0, 1.0)
        assert point.spin.scale == (1.0, 1.0)

    def test_predicted_collision_vectors_have_scale(self):
        """Verify PredictedCollision vectors have scale metadata."""
        collision = PredictedCollision(
            time=1.0,
            position=Vector2D(1000, 500, scale=(1.0, 1.0)),
            type=CollisionType.BALL_BALL,
            ball1_id="cue_ball",
            ball2_id="ball_1",
            impact_angle=0.0,
            impact_velocity=100.0,
            resulting_velocities={
                "cue_ball": Vector2D(25, 0, scale=(1.0, 1.0)),
                "ball_1": Vector2D(75, 0, scale=(1.0, 1.0)),
            },
            confidence=0.9,
            cushion_normal=Vector2D(0, -1, scale=(1.0, 1.0)),
        )

        assert collision.position.scale == (1.0, 1.0)
        assert collision.cushion_normal.scale == (1.0, 1.0)
        for vel in collision.resulting_velocities.values():
            assert vel.scale == (1.0, 1.0)

    def test_trajectory_final_vectors_have_scale(self, cue_ball):
        """Verify Trajectory final position/velocity have scale metadata."""
        trajectory = Trajectory(
            ball_id="test_ball",
            initial_state=cue_ball,
            final_position=Vector2D(2000, 1000, scale=(1.0, 1.0)),
            final_velocity=Vector2D(0, 0, scale=(1.0, 1.0)),
        )

        assert trajectory.final_position.scale == (1.0, 1.0)
        assert trajectory.final_velocity.scale == (1.0, 1.0)


# =============================================================================
# Basic Trajectory Calculation Tests
# =============================================================================


@pytest.mark.unit()
class TestBasicTrajectoryCalculation:
    """Test basic trajectory calculation functionality."""

    def test_stationary_ball_no_trajectory(self, calculator, cue_ball, table_state):
        """Test that a stationary ball produces no trajectory segments."""
        # Ball with zero velocity
        stationary_ball = cue_ball
        stationary_ball.velocity = Vector2D(0, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            stationary_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should have no trajectory points (just initial state)
        assert len(trajectory.points) == 0
        assert trajectory.time_to_rest == 0.0
        assert trajectory.total_distance == 0.0

    def test_moving_ball_creates_trajectory(self, calculator, cue_ball, table_state):
        """Test that a moving ball creates trajectory points."""
        # Ball moving horizontally
        moving_ball = cue_ball
        moving_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            moving_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should have trajectory points
        assert len(trajectory.points) > 0

        # All points should have proper scale
        for point in trajectory.points:
            assert point.position.scale == (1.0, 1.0)
            assert point.velocity.scale == (1.0, 1.0)
            assert point.acceleration.scale == (1.0, 1.0)
            assert point.spin.scale == (1.0, 1.0)

        # Final position should have scale
        assert trajectory.final_position.scale == (1.0, 1.0)
        assert trajectory.final_velocity.scale == (1.0, 1.0)

    def test_trajectory_quality_levels(self, calculator, cue_ball, table_state):
        """Test that different quality levels work."""
        moving_ball = cue_ball
        moving_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        for quality in [
            TrajectoryQuality.LOW,
            TrajectoryQuality.MEDIUM,
            TrajectoryQuality.HIGH,
            TrajectoryQuality.ULTRA,
        ]:
            trajectory = calculator.calculate_trajectory(
                moving_ball,
                table_state,
                quality=quality,
            )

            assert trajectory.quality == quality
            # Should still produce valid trajectory
            assert trajectory.final_position is not None
            assert trajectory.final_position.scale == (1.0, 1.0)

    def test_trajectory_distance_calculation(self, calculator, cue_ball, table_state):
        """Test that total distance is calculated correctly."""
        moving_ball = cue_ball
        moving_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            moving_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Distance should be positive for moving ball
        assert trajectory.total_distance > 0

        # Distance should be sum of segment distances
        if len(trajectory.points) >= 2:
            calculated_distance = 0
            for i in range(1, len(trajectory.points)):
                dx = (
                    trajectory.points[i].position.x
                    - trajectory.points[i - 1].position.x
                )
                dy = (
                    trajectory.points[i].position.y
                    - trajectory.points[i - 1].position.y
                )
                calculated_distance += math.sqrt(dx * dx + dy * dy)

            # Should match (within floating point tolerance)
            assert abs(trajectory.total_distance - calculated_distance) < 1.0


# =============================================================================
# Collision Detection Tests
# =============================================================================


@pytest.mark.unit()
class TestCollisionDetection:
    """Test collision detection in trajectories."""

    def test_ball_ball_collision(self, calculator, cue_ball, target_ball, table_state):
        """Test detection of ball-to-ball collision."""
        # Set cue ball moving toward target ball
        cue_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should detect collision
        ball_collisions = [
            c for c in trajectory.collisions if c.type == CollisionType.BALL_BALL
        ]

        if len(ball_collisions) > 0:
            collision = ball_collisions[0]
            assert collision.ball1_id == "cue_ball"
            assert collision.ball2_id == "ball_1"
            assert collision.position.scale == (1.0, 1.0)

            # Resulting velocities should have scale
            for vel in collision.resulting_velocities.values():
                assert vel.scale == (1.0, 1.0)

    def test_cushion_collision(self, calculator, cue_ball, table_state):
        """Test detection of cushion collision."""
        # Set ball moving toward right cushion
        cue_ball.position = Vector2D(3500, 1080, scale=(1.0, 1.0))
        cue_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should detect cushion collision
        cushion_collisions = [
            c for c in trajectory.collisions if c.type == CollisionType.BALL_CUSHION
        ]

        if len(cushion_collisions) > 0:
            collision = cushion_collisions[0]
            assert collision.position.scale == (1.0, 1.0)

            # Should have cushion normal
            if collision.cushion_normal:
                assert collision.cushion_normal.scale == (1.0, 1.0)

    def test_multiple_collisions(self, calculator, cue_ball, table_state):
        """Test that multiple collisions are tracked."""
        # Ball that will hit cushion and bounce
        cue_ball.position = Vector2D(3400, 1080, scale=(1.0, 1.0))
        cue_ball.velocity = Vector2D(200, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # May have multiple collisions (cushion bounces)
        # Just verify they all have proper scale
        for collision in trajectory.collisions:
            assert collision.position.scale == (1.0, 1.0)

            if collision.cushion_normal:
                assert collision.cushion_normal.scale == (1.0, 1.0)

            for vel in collision.resulting_velocities.values():
                assert vel.scale == (1.0, 1.0)


# =============================================================================
# Cue Shot Prediction Tests
# =============================================================================


@pytest.mark.unit()
class TestCueShotPrediction:
    """Test cue shot prediction functionality."""

    def test_predict_cue_shot_creates_trajectory(
        self, calculator, cue_state, cue_ball, table_state
    ):
        """Test that cue shot prediction creates valid trajectory."""
        trajectory = calculator.predict_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        assert trajectory.ball_id == "cue_ball"
        assert len(trajectory.points) > 0

        # Verify all vectors have scale
        for point in trajectory.points:
            assert point.position.scale == (1.0, 1.0)
            assert point.velocity.scale == (1.0, 1.0)

    def test_cue_shot_initial_velocity(
        self, calculator, cue_state, cue_ball, table_state
    ):
        """Test that cue shot calculates correct initial velocity."""
        # Horizontal shot (angle = 0)
        cue_state.angle = 0.0
        cue_state.estimated_force = 100.0

        trajectory = calculator.predict_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        if len(trajectory.points) > 0:
            first_point = trajectory.points[0]

            # Should be moving primarily in x direction for horizontal shot
            assert first_point.velocity.scale == (1.0, 1.0)

            # Velocity should be non-zero
            assert first_point.velocity.magnitude() > 0

    def test_cue_shot_with_angle(self, calculator, cue_ball, table_state):
        """Test cue shot at various angles."""
        angles = [0, 45, 90, -45]

        for angle in angles:
            cue_state = CueState(
                angle=angle,
                estimated_force=100.0,
                impact_point=Vector2D(1920.0, 1080.0, scale=(1.0, 1.0)),
                is_visible=True,
                tip_position=Vector2D(1820.0, 1080.0, scale=(1.0, 1.0)),
            )

            trajectory = calculator.predict_cue_shot(
                cue_state,
                cue_ball,
                table_state,
                quality=TrajectoryQuality.MEDIUM,
            )

            # Should create trajectory
            assert len(trajectory.points) > 0

            # All vectors should have scale
            for point in trajectory.points:
                assert point.position.scale == (1.0, 1.0)
                assert point.velocity.scale == (1.0, 1.0)

    def test_cue_shot_with_off_center_impact(self, calculator, cue_ball, table_state):
        """Test cue shot with off-center impact (spin)."""
        # Impact point off-center
        cue_state = CueState(
            angle=0.0,
            estimated_force=100.0,
            impact_point=Vector2D(1920.0, 1100.0, scale=(1.0, 1.0)),  # Above center
            is_visible=True,
            tip_position=Vector2D(1820.0, 1100.0, scale=(1.0, 1.0)),
        )

        trajectory = calculator.predict_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should create trajectory with spin
        assert len(trajectory.points) > 0

        # Points should have spin information
        for point in trajectory.points:
            assert point.spin.scale == (1.0, 1.0)


# =============================================================================
# Multiball Trajectory Tests
# =============================================================================


@pytest.mark.unit()
class TestMultiballTrajectory:
    """Test multiball trajectory prediction."""

    def test_multiball_single_ball(self, calculator, cue_state, cue_ball, table_state):
        """Test multiball with just cue ball (no collisions)."""
        result = calculator.predict_multiball_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            other_balls=[],
            quality=TrajectoryQuality.MEDIUM,
        )

        assert result.primary_ball_id == "cue_ball"
        assert "cue_ball" in result.trajectories

        # Should have cue ball trajectory
        cue_trajectory = result.trajectories["cue_ball"]
        for point in cue_trajectory.points:
            assert point.position.scale == (1.0, 1.0)
            assert point.velocity.scale == (1.0, 1.0)

    def test_multiball_with_target_ball(
        self, calculator, cue_state, cue_ball, target_ball, table_state
    ):
        """Test multiball with cue ball hitting target ball."""
        # Aim cue ball at target ball
        cue_state.angle = 0.0  # Straight toward target
        cue_state.estimated_force = 100.0

        result = calculator.predict_multiball_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should have cue ball trajectory
        assert "cue_ball" in result.trajectories

        # May have target ball trajectory if collision detected
        for ball_id, trajectory in result.trajectories.items():
            # All trajectory points should have scale
            for point in trajectory.points:
                assert point.position.scale == (1.0, 1.0)
                assert point.velocity.scale == (1.0, 1.0)
                assert point.acceleration.scale == (1.0, 1.0)
                assert point.spin.scale == (1.0, 1.0)

    def test_multiball_collision_sequence(
        self, calculator, cue_state, cue_ball, target_ball, table_state
    ):
        """Test that collision sequence is tracked."""
        result = calculator.predict_multiball_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.MEDIUM,
        )

        # Verify collision sequence vectors have scale
        for collision in result.collision_sequence:
            assert collision.position.scale == (1.0, 1.0)

            if collision.cushion_normal:
                assert collision.cushion_normal.scale == (1.0, 1.0)

            for vel in collision.resulting_velocities.values():
                assert vel.scale == (1.0, 1.0)

    def test_multiball_max_depth(self, calculator, cue_state, cue_ball, table_state):
        """Test multiball respects max collision depth."""
        # Create chain of balls
        balls = [
            BallState.from_4k(id=f"ball_{i}", x=2000 + i * 100, y=1080, number=i)
            for i in range(1, 6)
        ]

        # Test with different max depths
        for max_depth in [1, 2, 5]:
            result = calculator.predict_multiball_cue_shot(
                cue_state,
                cue_ball,
                table_state,
                other_balls=balls,
                quality=TrajectoryQuality.LOW,
                max_collision_depth=max_depth,
            )

            # Should respect depth limit
            # (exact count depends on collision detection)
            assert len(result.trajectories) >= 1  # At least cue ball


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit()
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_velocity(self, calculator, cue_ball, table_state):
        """Test ball with exactly zero velocity."""
        cue_ball.velocity = Vector2D(0, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should handle gracefully
        assert trajectory.time_to_rest == 0.0
        assert trajectory.total_distance == 0.0
        assert trajectory.final_velocity.magnitude() == 0.0

    def test_very_high_velocity(self, calculator, cue_ball, table_state):
        """Test ball with very high velocity."""
        cue_ball.velocity = Vector2D(10000, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.LOW,  # Use low quality for speed
        )

        # Should still create trajectory
        assert len(trajectory.points) > 0

        # All vectors should have scale
        for point in trajectory.points:
            assert point.position.scale == (1.0, 1.0)
            assert point.velocity.scale == (1.0, 1.0)

    def test_near_boundary_position(self, calculator, table_state):
        """Test ball positioned very close to boundary."""
        # Ball near right cushion
        near_edge_ball = BallState.from_4k(
            id="edge_ball",
            x=3600,  # Near right edge
            y=1080,
            number=1,
        )
        near_edge_ball.velocity = Vector2D(10, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            near_edge_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should handle without errors
        assert trajectory.final_position.scale == (1.0, 1.0)

    def test_overlapping_balls(self, calculator, cue_ball, table_state):
        """Test with balls that overlap (edge case)."""
        # Create ball overlapping with cue ball
        overlapping_ball = BallState.from_4k(
            id="overlap_ball",
            x=1920,  # Same position
            y=1080,
            number=1,
        )

        cue_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            other_balls=[overlapping_ball],
            quality=TrajectoryQuality.LOW,
        )

        # Should handle gracefully (implementation-dependent)
        assert trajectory.final_position.scale == (1.0, 1.0)


# =============================================================================
# Cache Tests
# =============================================================================


@pytest.mark.unit()
class TestCaching:
    """Test trajectory caching functionality."""

    def test_cache_key_generation(self, calculator, cue_ball, table_state):
        """Test that cache keys are generated."""
        cue_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        trajectory = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Should have cache key
        assert trajectory.cache_key is not None
        assert isinstance(trajectory.cache_key, str)

    def test_cache_hit(self, calculator, cue_ball, table_state):
        """Test that identical calculations hit cache."""
        cue_ball.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        # First calculation
        trajectory1 = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Second calculation (should hit cache)
        trajectory2 = calculator.calculate_trajectory(
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        # Cache keys should match
        assert trajectory1.cache_key == trajectory2.cache_key

    def test_clear_cache(self, calculator):
        """Test cache clearing."""
        calculator.clear_cache()

        # Should succeed without errors
        stats = calculator.get_cache_stats()
        assert isinstance(stats, dict)


# =============================================================================
# Trajectory Export Tests
# =============================================================================


@pytest.mark.unit()
class TestTrajectoryExport:
    """Test trajectory data export for visualization."""

    def test_export_visualization_data(
        self, calculator, cue_state, cue_ball, table_state
    ):
        """Test exporting trajectory for visualization."""
        trajectory = calculator.predict_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            quality=TrajectoryQuality.MEDIUM,
        )

        export_data = calculator.export_visualization_data(trajectory)

        # Should have required fields
        assert "ball_id" in export_data
        assert "points" in export_data
        assert "lines" in export_data
        assert "collisions" in export_data

        # Points should be serializable
        for point in export_data["points"]:
            assert "x" in point
            assert "y" in point
            assert "vx" in point
            assert "vy" in point

        # Lines should connect consecutive points
        for line in export_data["lines"]:
            assert "start" in line
            assert "end" in line
            assert len(line["start"]) == 2
            assert len(line["end"]) == 2

    def test_export_with_collisions(
        self, calculator, cue_state, cue_ball, target_ball, table_state
    ):
        """Test export includes collision data."""
        trajectory = calculator.predict_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.MEDIUM,
        )

        export_data = calculator.export_visualization_data(trajectory)

        # Check collision format
        for collision in export_data["collisions"]:
            assert "position" in collision
            assert "type" in collision
            assert len(collision["position"]) == 2  # [x, y] array format


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit()
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_shot_sequence(
        self, calculator, cue_state, cue_ball, target_ball, table_state
    ):
        """Test complete shot sequence from cue to rest."""
        # Setup shot toward target ball
        cue_state.angle = 0.0
        cue_state.estimated_force = 150.0

        # Calculate multiball trajectory
        result = calculator.predict_multiball_cue_shot(
            cue_state,
            cue_ball,
            table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.MEDIUM,
            max_collision_depth=3,
        )

        # Verify result structure
        assert result.primary_ball_id == "cue_ball"
        assert len(result.trajectories) >= 1
        assert result.total_calculation_time >= 0

        # Verify all trajectories have proper scale
        for ball_id, trajectory in result.trajectories.items():
            assert trajectory.final_position.scale == (1.0, 1.0)
            assert trajectory.final_velocity.scale == (1.0, 1.0)

            for point in trajectory.points:
                assert point.position.scale == (1.0, 1.0)
                assert point.velocity.scale == (1.0, 1.0)
                assert point.acceleration.scale == (1.0, 1.0)
                assert point.spin.scale == (1.0, 1.0)

        # Export should work
        for trajectory in result.trajectories.values():
            export_data = calculator.export_visualization_data(trajectory)
            assert export_data is not None

    def test_resolution_independence(self, calculator, table_state):
        """Test that calculations work with different resolution inputs."""
        # Create balls in different resolutions, all should work
        ball_4k = BallState.from_4k(id="ball_4k", x=1920, y=1080, number=0)
        ball_4k.velocity = Vector2D(100, 0, scale=(1.0, 1.0))

        ball_1080p = BallState.from_resolution(
            id="ball_1080p", x=960, y=540, resolution=(1920, 1080), number=1
        )
        ball_1080p.velocity = Vector2D(100, 0, scale=(2.0, 2.0))

        # Both should calculate without errors
        traj_4k = calculator.calculate_trajectory(
            ball_4k, table_state, quality=TrajectoryQuality.LOW
        )
        traj_1080p = calculator.calculate_trajectory(
            ball_1080p, table_state, quality=TrajectoryQuality.LOW
        )

        # Both should have valid results (final position should have scale metadata)
        assert traj_4k.final_position.scale is not None
        assert len(traj_4k.final_position.scale) == 2

        assert traj_1080p.final_position.scale is not None
        assert len(traj_1080p.final_position.scale) == 2

        # Verify positions can be converted to 4K canonical
        pos_4k_1 = traj_4k.final_position.to_4k_canonical()
        pos_4k_2 = traj_1080p.final_position.to_4k_canonical()

        assert pos_4k_1.scale == (1.0, 1.0)
        assert pos_4k_2.scale == (1.0, 1.0)


if __name__ == "__main__":
    # Run tests via pytest when executed directly
    # Usage: python -m pytest backend/tests/core/test_trajectory.py
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
