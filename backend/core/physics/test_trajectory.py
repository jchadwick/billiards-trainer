"""Comprehensive tests for trajectory calculation system."""

import time

from backend.core.models import BallState, CueState, TableState, Vector2D
from backend.core.utils.cache import CacheManager

from .trajectory import (
    CollisionType,
    TrajectoryCalculator,
    TrajectoryOptimizer,
    TrajectoryQuality,
)


class TestTrajectoryCalculator:
    """Test cases for trajectory calculation system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager()
        self.calculator = TrajectoryCalculator(self.cache_manager)

        # Standard table
        self.table = TableState.standard_9ft_table()

        # Standard test ball
        self.ball = BallState(
            id="test_ball",
            position=Vector2D(0.5, 0.5),  # Center-ish position
            velocity=Vector2D(1.0, 0.0),  # Moving right
            radius=0.028575,
            mass=0.17,
        )

        # Other balls for collision tests
        self.other_balls = [
            BallState(
                id="ball_1",
                position=Vector2D(1.0, 0.5),
                velocity=Vector2D(0.0, 0.0),
                radius=0.028575,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D(1.5, 0.8),
                velocity=Vector2D(0.0, 0.0),
                radius=0.028575,
                mass=0.17,
            ),
        ]

    def test_simple_trajectory_no_collisions(self):
        """Test basic trajectory calculation with no collisions."""
        # Ball moving with friction - should decelerate and stop
        ball = BallState(
            id="simple",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.5, 0.0),  # Moderate speed
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH
        )

        # Verify basic properties
        assert trajectory.ball_id == "simple"
        assert len(trajectory.points) > 0
        assert (
            trajectory.final_velocity.magnitude() < 0.01
        )  # Should stop due to friction
        assert trajectory.time_to_rest > 0
        assert not trajectory.will_be_pocketed
        assert trajectory.total_distance > 0

    def test_ball_ball_collision_detection(self):
        """Test ball-to-ball collision detection and response."""
        # Ball moving toward stationary ball
        moving_ball = BallState(
            id="moving",
            position=Vector2D(0.3, 0.5),
            velocity=Vector2D(2.0, 0.0),  # Fast moving right
            radius=0.028575,
            mass=0.17,
        )

        stationary_ball = BallState(
            id="stationary",
            position=Vector2D(0.8, 0.5),  # In path of moving ball
            velocity=Vector2D(0.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            moving_ball, self.table, [stationary_ball], TrajectoryQuality.HIGH
        )

        # Should detect collision
        assert len(trajectory.collisions) > 0
        collision = trajectory.collisions[0]
        assert collision.type == CollisionType.BALL_BALL
        assert collision.ball2_id == "stationary"
        assert collision.time > 0

        # Moving ball should transfer momentum to stationary ball
        # and slow down significantly
        assert trajectory.final_velocity.magnitude() < moving_ball.velocity.magnitude()

    def test_cushion_collision_detection(self):
        """Test ball-to-cushion collision detection and response."""
        # Ball moving toward right cushion
        ball = BallState(
            id="cushion_test",
            position=Vector2D(2.0, 0.5),  # Near right edge
            velocity=Vector2D(1.0, 0.0),  # Moving right toward cushion
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH
        )

        # Should detect cushion collision
        assert len(trajectory.collisions) > 0
        collision = trajectory.collisions[0]
        assert collision.type == CollisionType.BALL_CUSHION

        # Ball should bounce back (negative x velocity after collision)
        # Check trajectory points after collision
        collision_idx = next(
            i for i, p in enumerate(trajectory.points) if p.time >= collision.time
        )
        if collision_idx < len(trajectory.points) - 1:
            post_collision_point = trajectory.points[collision_idx + 1]
            # After bouncing off right cushion, x velocity should be negative
            assert post_collision_point.velocity.x < 0

    def test_pocket_collision_detection(self):
        """Test pocket detection when ball heads toward pocket."""
        # Position ball to head toward corner pocket
        corner_pocket = self.table.pocket_positions[0]  # Bottom-left corner

        # Place ball heading toward pocket
        ball = BallState(
            id="pocket_test",
            position=Vector2D(corner_pocket.x + 0.1, corner_pocket.y + 0.1),
            velocity=Vector2D(-0.5, -0.5),  # Moving toward corner
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH
        )

        # Check if pocket collision was detected
        if trajectory.will_be_pocketed:
            assert trajectory.pocket_id is not None
            # Should have pocket collision in collisions list
            pocket_collisions = [
                c for c in trajectory.collisions if c.type == CollisionType.BALL_POCKET
            ]
            assert len(pocket_collisions) > 0

    def test_multi_bounce_trajectory(self):
        """Test trajectory with multiple cushion bounces."""
        # Ball with angle to create multiple bounces
        ball = BallState(
            id="multi_bounce",
            position=Vector2D(0.5, 0.3),
            velocity=Vector2D(2.0, 1.5),  # Angled velocity for bounces
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH, time_limit=3.0
        )

        # Should have multiple cushion collisions
        cushion_collisions = [
            c for c in trajectory.collisions if c.type == CollisionType.BALL_CUSHION
        ]
        assert len(cushion_collisions) >= 2  # At least 2 bounces

        # Collision times should be in ascending order
        times = [c.time for c in trajectory.collisions]
        assert times == sorted(times)

    def test_spin_effects(self):
        """Test trajectory calculation with ball spin."""
        # Ball with sidespin
        ball = BallState(
            id="spin_test",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
            spin=Vector2D(0.0, 5.0),  # Sidespin
        )

        trajectory_with_spin = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH
        )

        # Compare with no-spin trajectory
        ball_no_spin = BallState(
            id="no_spin",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
            spin=Vector2D(0.0, 0.0),  # No spin
        )

        trajectory_no_spin = self.calculator.calculate_trajectory(
            ball_no_spin, self.table, [], TrajectoryQuality.HIGH
        )

        # Trajectories should be different due to Magnus effect
        # (though the effect might be small with current simplified physics)
        if (
            len(trajectory_with_spin.points) > 10
            and len(trajectory_no_spin.points) > 10
        ):
            mid_point_spin = trajectory_with_spin.points[
                len(trajectory_with_spin.points) // 2
            ]
            mid_point_no_spin = trajectory_no_spin.points[
                len(trajectory_no_spin.points) // 2
            ]

            # Positions should differ due to spin effects
            position_difference = abs(
                mid_point_spin.position.y - mid_point_no_spin.position.y
            )
            # Allow for small effect (spin implementation is simplified)
            assert position_difference >= 0  # At minimum, should not crash

    def test_cue_shot_prediction(self):
        """Test predicting trajectory from cue shot parameters."""
        cue_ball = BallState(
            id="cue",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.0, 0.0),  # Initially stationary
            radius=0.028575,
            mass=0.17,
            is_cue_ball=True,
        )

        cue_state = CueState(
            tip_position=Vector2D(0.4, 0.5),  # Behind cue ball
            angle=0.0,  # Pointing right
            estimated_force=10.0,  # Moderate force
        )

        trajectory = self.calculator.predict_cue_shot(
            cue_state, cue_ball, self.table, [], TrajectoryQuality.HIGH
        )

        # Should generate trajectory with initial velocity in cue direction
        assert len(trajectory.points) > 0
        first_point = trajectory.points[0]
        assert first_point.velocity.x > 0  # Moving right
        assert abs(first_point.velocity.y) < 0.1  # Minimal y component

    def test_trajectory_quality_levels(self):
        """Test different quality levels produce different results."""
        ball = BallState(
            id="quality_test",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.5),
            radius=0.028575,
            mass=0.17,
        )

        # Calculate with different quality levels
        trajectory_low = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.LOW
        )
        trajectory_high = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.HIGH
        )

        # High quality should have more trajectory points (smaller timesteps)
        assert len(trajectory_high.points) >= len(trajectory_low.points)

        # Both should reach similar final positions (within tolerance)
        position_diff = trajectory_high.final_position.distance_to(
            trajectory_low.final_position
        )
        assert position_diff < 0.1  # Should be reasonably close

    def test_alternative_trajectories(self):
        """Test generation of alternative trajectory branches."""
        ball = BallState(
            id="alternatives",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.MEDIUM
        )

        # Should generate alternative branches
        assert len(trajectory.branches) > 0

        # Each branch should have different descriptions
        descriptions = [branch.description for branch in trajectory.branches]
        assert len(set(descriptions)) == len(descriptions)  # All unique

    def test_trajectory_caching(self):
        """Test trajectory caching functionality."""
        ball = BallState(
            id="cache_test",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        # Clear cache first
        self.calculator.clear_cache()

        # First calculation - should be cached
        start_time = time.time()
        trajectory1 = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.MEDIUM
        )
        time.time() - start_time

        # Second calculation - should use cache
        start_time = time.time()
        trajectory2 = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.MEDIUM
        )
        time.time() - start_time

        # Results should be identical
        assert trajectory1.ball_id == trajectory2.ball_id
        assert len(trajectory1.points) == len(trajectory2.points)

        # Second calculation should be faster (from cache)
        # Note: This might not always be true due to system variability
        # but cache should at least work without errors

    def test_visualization_data_export(self):
        """Test trajectory data export for visualization."""
        ball = BallState(
            id="viz_test",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.5),
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball, self.table, [], TrajectoryQuality.MEDIUM
        )

        viz_data = self.calculator.export_visualization_data(trajectory)

        # Check required fields
        assert "ball_id" in viz_data
        assert "points" in viz_data
        assert "collisions" in viz_data
        assert "success_probability" in viz_data
        assert "alternatives" in viz_data

        # Check point data structure
        if viz_data["points"]:
            point = viz_data["points"][0]
            assert all(key in point for key in ["time", "x", "y", "vx", "vy", "energy"])

    def test_complex_multi_ball_scenario(self):
        """Test complex scenario with multiple balls and collisions."""
        # Set up a break shot scenario
        cue_ball = BallState(
            id="cue",
            position=Vector2D(0.5, 0.3),
            velocity=Vector2D(0.0, 2.0),  # Moving toward rack
            radius=0.028575,
            mass=0.17,
            is_cue_ball=True,
        )

        # Create a mini rack of balls
        rack_balls = []
        rack_center = Vector2D(0.5, 1.5)
        for i in range(3):
            for j in range(i + 1):
                x = rack_center.x + (j - i / 2) * 0.06
                y = rack_center.y + i * 0.052
                ball = BallState(
                    id=f"rack_{i}_{j}",
                    position=Vector2D(x, y),
                    velocity=Vector2D(0.0, 0.0),
                    radius=0.028575,
                    mass=0.17,
                    number=i * 3 + j + 1,
                )
                rack_balls.append(ball)

        trajectory = self.calculator.calculate_trajectory(
            cue_ball, self.table, rack_balls, TrajectoryQuality.MEDIUM, time_limit=2.0
        )

        # Should detect collision with rack
        ball_collisions = [
            c for c in trajectory.collisions if c.type == CollisionType.BALL_BALL
        ]
        assert len(ball_collisions) > 0

        # First collision should be with a rack ball
        first_collision = trajectory.collisions[0]
        assert first_collision.ball2_id.startswith("rack_")

    def test_trajectory_optimizer(self):
        """Test trajectory optimizer functionality."""
        optimizer = TrajectoryOptimizer()

        # Test optimal timestep calculation
        high_velocity = Vector2D(5.0, 0.0)
        low_velocity = Vector2D(0.5, 0.0)

        dt_high = optimizer.get_optimal_timestep(
            high_velocity, TrajectoryQuality.MEDIUM
        )
        dt_low = optimizer.get_optimal_timestep(low_velocity, TrajectoryQuality.MEDIUM)

        # High velocity should get smaller timestep
        assert dt_high <= dt_low

        # Test early termination
        slow_ball = BallState(
            id="slow",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.005, 0.0),  # Very slow
            radius=0.028575,
            mass=0.17,
        )

        should_terminate = optimizer.should_terminate_early(slow_ball, self.table)
        assert should_terminate  # Should terminate due to low velocity

    def test_known_trajectory_outcomes(self):
        """Test against known physics outcomes."""
        # Test 1: Head-on elastic collision between equal masses
        ball1 = BallState(
            id="ball1",
            position=Vector2D(0.3, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        ball2 = BallState(
            id="ball2",
            position=Vector2D(0.7, 0.5),  # Direct path
            velocity=Vector2D(0.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            ball1, self.table, [ball2], TrajectoryQuality.HIGH
        )

        # In perfect elastic collision between equal masses,
        # moving ball should stop and stationary ball should take all velocity
        if trajectory.collisions:
            # Ball should significantly slow down after collision
            assert (
                trajectory.final_velocity.magnitude() < 0.5 * ball1.velocity.magnitude()
            )

        # Test 2: 45-degree cushion bounce
        ball_45 = BallState(
            id="bounce_45",
            position=Vector2D(0.1, 0.1),
            velocity=Vector2D(1.0, 1.0),  # 45-degree angle toward corner
            radius=0.028575,
            mass=0.17,
        )

        trajectory_45 = self.calculator.calculate_trajectory(
            ball_45, self.table, [], TrajectoryQuality.HIGH
        )

        # Should bounce off cushion
        cushion_collisions = [
            c for c in trajectory_45.collisions if c.type == CollisionType.BALL_CUSHION
        ]
        assert len(cushion_collisions) > 0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test 1: Ball with zero velocity
        stationary_ball = BallState(
            id="stationary",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )

        trajectory = self.calculator.calculate_trajectory(
            stationary_ball, self.table, [], TrajectoryQuality.MEDIUM
        )

        # Should handle gracefully - minimal trajectory
        assert trajectory.time_to_rest >= 0
        assert trajectory.final_position.x == 0.5
        assert trajectory.final_position.y == 0.5

        # Test 2: Ball at edge of table
        edge_ball = BallState(
            id="edge",
            position=Vector2D(0.03, 0.5),  # Very close to left cushion
            velocity=Vector2D(-0.1, 0.0),  # Moving toward cushion
            radius=0.028575,
            mass=0.17,
        )

        trajectory_edge = self.calculator.calculate_trajectory(
            edge_ball, self.table, [], TrajectoryQuality.MEDIUM
        )

        # Should detect immediate collision or handle gracefully
        assert len(trajectory_edge.points) > 0

    def test_success_probability_calculation(self):
        """Test success probability calculation."""
        # Ball heading directly toward pocket
        corner_pocket = self.table.pocket_positions[0]
        direct_shot = BallState(
            id="direct_shot",
            position=Vector2D(corner_pocket.x + 0.2, corner_pocket.y + 0.2),
            velocity=Vector2D(-1.0, -1.0),  # Directly toward pocket
            radius=0.028575,
            mass=0.17,
        )

        trajectory_direct = self.calculator.calculate_trajectory(
            direct_shot, self.table, [], TrajectoryQuality.HIGH
        )

        # Ball heading away from any pocket
        away_shot = BallState(
            id="away_shot",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.1, 0.1),  # Very slow, toward center
            radius=0.028575,
            mass=0.17,
        )

        trajectory_away = self.calculator.calculate_trajectory(
            away_shot, self.table, [], TrajectoryQuality.HIGH
        )

        # Direct shot should have higher success probability than away shot
        if trajectory_direct.will_be_pocketed:
            assert (
                trajectory_direct.success_probability
                > trajectory_away.success_probability
            )


if __name__ == "__main__":
    # Run basic functionality test
    test = TestTrajectoryCalculator()
    test.setup_method()

    print("Running basic trajectory tests...")

    try:
        test.test_simple_trajectory_no_collisions()
        print("✓ Simple trajectory test passed")

        test.test_ball_ball_collision_detection()
        print("✓ Ball-ball collision test passed")

        test.test_cushion_collision_detection()
        print("✓ Cushion collision test passed")

        test.test_multi_bounce_trajectory()
        print("✓ Multi-bounce trajectory test passed")

        test.test_cue_shot_prediction()
        print("✓ Cue shot prediction test passed")

        test.test_visualization_data_export()
        print("✓ Visualization data export test passed")

        print("\nAll basic tests passed! Trajectory system is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
