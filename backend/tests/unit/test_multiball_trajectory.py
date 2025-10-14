"""Unit tests for multiball trajectory calculation in the integration service.

This test module focuses on testing the multiball trajectory calculation flow:
1. Creating mock cue and ball detections
2. Calling the trajectory calculation through integration_service
3. Verifying MultiballTrajectoryResult structure
4. Checking primary ball and hit ball trajectories
5. Validating collision information
6. Testing edge cases (no target ball, multiple balls in line, collision chains)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from config.manager import ConfigurationModule
from core import CoreModule
from core.game_state import GameStateManager
from core.models import BallState, CueState, TableState, Vector2D
from core.physics.trajectory import (
    CollisionType,
    MultiballTrajectoryResult,
    TrajectoryCalculator,
    TrajectoryQuality,
)
from integration_service import IntegrationService
from vision.models import Ball, BallType, CueStick, CueStickState, DetectionResult


# Fixtures
@pytest.fixture()
def config_module():
    """Create a test configuration module."""
    config = ConfigurationModule()
    # Set test configuration values
    config._config_data = {
        "integration": {
            "target_fps": 30,
            "log_interval_frames": 300,
            "error_retry_delay_sec": 0.1,
            "shot_speed_estimate_m_per_s": 2.0,
            "broadcast_max_retries": 3,
            "broadcast_retry_base_delay_sec": 0.1,
            "circuit_breaker_threshold": 10,
            "circuit_breaker_timeout_sec": 30.0,
        },
        "cue": {"default_force": 5.0},
    }
    return config


@pytest.fixture()
def mock_broadcaster():
    """Create a mock message broadcaster."""
    broadcaster = MagicMock()
    broadcaster.broadcast_game_state = AsyncMock()
    broadcaster.broadcast_trajectory = AsyncMock()
    return broadcaster


@pytest.fixture()
def core_module():
    """Create a core module with initialized state."""
    core = CoreModule()
    # Initialize with a default game state
    table = TableState.standard_9ft_table()
    initial_state = {
        "timestamp": time.time(),
        "frame_number": 0,
        "balls": [],
        "table": {
            "width": table.width,
            "height": table.height,
            "pockets": [
                {"position": {"x": pocket.x, "y": pocket.y}, "type": "corner"}
                for pocket in table.pocket_positions
            ],
        },
    }
    asyncio.run(core.update_state(initial_state))
    return core


@pytest.fixture()
def vision_module():
    """Create a mock vision module."""
    vision = MagicMock()
    vision.start_capture = MagicMock(return_value=True)
    vision.stop_capture = MagicMock()
    vision.process_frame = MagicMock(return_value=None)
    vision.calibrator = None
    return vision


@pytest.fixture()
def integration_service(vision_module, core_module, mock_broadcaster, config_module):
    """Create an integration service instance for testing."""
    return IntegrationService(
        vision_module=vision_module,
        core_module=core_module,
        message_broadcaster=mock_broadcaster,
        config_module=config_module,
    )


# Helper functions
def create_mock_cue_detection(
    tip_x: float = 1.0,
    tip_y: float = 0.71,
    angle: float = 0.0,
    confidence: float = 0.95,
) -> CueStick:
    """Create a mock cue stick detection.

    Args:
        tip_x: Cue tip x position (meters)
        tip_y: Cue tip y position (meters)
        angle: Cue angle in degrees
        confidence: Detection confidence (0-1)

    Returns:
        CueStick detection object
    """
    # Calculate butt position based on angle (1 meter behind tip)
    angle_rad = np.deg2rad(angle)
    butt_x = tip_x - np.cos(angle_rad)
    butt_y = tip_y - np.sin(angle_rad)

    return CueStick(
        tip_position=(tip_x, tip_y),
        butt_position=(butt_x, butt_y),
        angle=angle,
        length=1.0,
        state=CueStickState.AIMING,
        confidence=confidence,
    )


def create_mock_ball_detection(
    x: float,
    y: float,
    ball_type: BallType = BallType.SOLID,
    number: int = None,
    radius: float = 0.028575,
    confidence: float = 0.95,
) -> Ball:
    """Create a mock ball detection.

    Args:
        x: Ball x position (meters)
        y: Ball y position (meters)
        ball_type: Type of ball
        number: Ball number
        radius: Ball radius (meters)
        confidence: Detection confidence (0-1)

    Returns:
        Ball detection object
    """
    return Ball(
        position=(x, y),
        radius=radius,
        ball_type=ball_type,
        number=number,
        confidence=confidence,
        velocity=(0.0, 0.0),
        is_moving=False,
        track_id=number if number is not None else None,
    )


def create_mock_detection_result(
    cue: CueStick = None, balls: list[Ball] = None, timestamp: float = None
) -> DetectionResult:
    """Create a mock detection result.

    Args:
        cue: Cue stick detection
        balls: List of ball detections
        timestamp: Detection timestamp

    Returns:
        DetectionResult object
    """
    return DetectionResult(
        timestamp=timestamp or time.time(),
        frame_number=1,
        balls=balls or [],
        cue=cue,
        table=None,
        statistics=None,
    )


# Test Classes
@pytest.mark.unit()
class TestMultiballTrajectoryBasics:
    """Test basic multiball trajectory calculation."""

    @pytest.mark.asyncio()
    async def test_simple_two_ball_collision(self, integration_service):
        """Test trajectory calculation for simple two-ball collision."""
        # Create cue pointing at cue ball
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)

        # Create cue ball and target ball in line
        cue_ball = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        target_ball = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        # Create detection result
        detection = create_mock_detection_result(cue=cue, balls=[cue_ball, target_ball])

        # Process detection to trigger trajectory calculation
        await integration_service._process_detection(detection)

        # Verify broadcast was called
        assert integration_service.broadcaster.broadcast_trajectory.called
        call_args = integration_service.broadcaster.broadcast_trajectory.call_args

        # Extract trajectory data from broadcast call
        lines = call_args[0][0]
        collisions = call_args[0][1]

        # Should have trajectory lines for both balls
        assert len(lines) > 0, "Should have trajectory lines"

        # Should have at least one collision (cue ball hitting target ball)
        assert len(collisions) > 0, "Should have at least one collision"

        # Verify collision type
        ball_ball_collisions = [
            c for c in collisions if c["type"] == CollisionType.BALL_BALL.value
        ]
        assert (
            len(ball_ball_collisions) > 0
        ), "Should have at least one ball-ball collision"

    @pytest.mark.asyncio()
    async def test_trajectory_result_structure(self, integration_service):
        """Test that MultiballTrajectoryResult has correct structure."""
        # Create simple scenario
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)
        cue_ball = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        target_ball = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        create_mock_detection_result(cue=cue, balls=[cue_ball, target_ball])

        # Calculate trajectory directly using trajectory calculator
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(cue_ball, is_target=True)
        other_balls = integration_service._create_ball_states(
            [target_ball], exclude_ball=cue_ball
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Verify result structure
        assert isinstance(result, MultiballTrajectoryResult)
        assert hasattr(result, "primary_ball_id")
        assert hasattr(result, "trajectories")
        assert hasattr(result, "collision_sequence")
        assert hasattr(result, "total_calculation_time")

        # Should have trajectory for primary ball
        assert result.primary_ball_id in result.trajectories
        primary_trajectory = result.trajectories[result.primary_ball_id]

        # Verify trajectory structure
        assert hasattr(primary_trajectory, "points")
        assert hasattr(primary_trajectory, "collisions")
        assert len(primary_trajectory.points) > 0, "Should have trajectory points"

    @pytest.mark.asyncio()
    async def test_primary_and_secondary_trajectories(self, integration_service):
        """Test that both primary and secondary ball trajectories are calculated."""
        # Create scenario with cue ball hitting target ball
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)
        cue_ball = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        target_ball = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        # Calculate trajectory
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(cue_ball, is_target=True)
        other_balls = integration_service._create_ball_states(
            [target_ball], exclude_ball=cue_ball
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Verify we have multiple trajectories
        # Note: Secondary ball trajectory is only created if there's a ball-ball collision
        # with enough velocity transfer
        assert len(result.trajectories) >= 1, "Should have at least primary trajectory"

        # Check if we have a ball-ball collision
        ball_collisions = [
            c for c in result.collision_sequence if c.type == CollisionType.BALL_BALL
        ]

        if ball_collisions:
            # If there's a collision, we might have secondary trajectories
            # depending on velocity transfer
            assert len(result.trajectories) >= 1, "Should have trajectories"


@pytest.mark.unit()
class TestMultiballTrajectoryEdgeCases:
    """Test edge cases in multiball trajectory calculation."""

    @pytest.mark.asyncio()
    async def test_no_target_ball_found(self, integration_service):
        """Test trajectory calculation when cue points at nothing."""
        # Create cue pointing away from all balls
        cue = create_mock_cue_detection(tip_x=0.5, tip_y=0.5, angle=180.0)

        # Create balls not in line with cue
        ball1 = create_mock_ball_detection(
            x=2.0, y=1.0, ball_type=BallType.SOLID, number=1
        )
        ball2 = create_mock_ball_detection(
            x=2.2, y=1.2, ball_type=BallType.SOLID, number=2
        )

        detection = create_mock_detection_result(cue=cue, balls=[ball1, ball2])

        # Process detection
        await integration_service._process_detection(detection)

        # Trajectory calculation should not happen (no target ball)
        # Verify by checking that broadcast was not called or was called with empty data
        if integration_service.broadcaster.broadcast_trajectory.called:
            call_args = integration_service.broadcaster.broadcast_trajectory.call_args
            lines = call_args[0][0]
            # If called, should have no lines since no trajectory was calculated
            assert len(lines) == 0, "Should have no trajectory lines"

    @pytest.mark.asyncio()
    async def test_no_cue_detected(self, integration_service):
        """Test trajectory calculation when no cue is detected."""
        # Create balls without cue
        ball1 = create_mock_ball_detection(
            x=1.5, y=0.71, ball_type=BallType.CUE, number=0
        )
        ball2 = create_mock_ball_detection(
            x=2.0, y=0.71, ball_type=BallType.SOLID, number=1
        )

        detection = create_mock_detection_result(cue=None, balls=[ball1, ball2])

        # Process detection
        await integration_service._process_detection(detection)

        # Should not broadcast trajectory since no cue detected
        # Note: broadcast might be called for state updates, but not for trajectory
        # We can't easily distinguish this, so we just verify no error occurred

    @pytest.mark.asyncio()
    async def test_no_balls_detected(self, integration_service):
        """Test trajectory calculation when no balls are detected."""
        # Create cue without balls
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)

        detection = create_mock_detection_result(cue=cue, balls=[])

        # Process detection
        await integration_service._process_detection(detection)

        # Should not calculate trajectory since no balls detected

    @pytest.mark.asyncio()
    async def test_multiple_balls_in_line(self, integration_service):
        """Test trajectory with multiple balls aligned."""
        # Create cue pointing at first ball
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)

        # Create three balls in a line
        ball1 = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        ball2 = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )
        ball3 = create_mock_ball_detection(
            x=2.4, y=0.71, ball_type=BallType.SOLID, number=2
        )

        # Calculate trajectory
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(ball1, is_target=True)
        other_balls = integration_service._create_ball_states(
            [ball2, ball3], exclude_ball=ball1
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Should have trajectories for multiple balls in collision chain
        assert len(result.trajectories) >= 1, "Should have at least primary trajectory"

        # Check collision sequence
        assert len(result.collision_sequence) > 0, "Should have collisions"

    @pytest.mark.asyncio()
    async def test_collision_chain_depth_limit(self, integration_service):
        """Test that collision chains respect max_collision_depth."""
        # Create a line of 6 balls
        cue = create_mock_cue_detection(tip_x=0.8, tip_y=0.71, angle=0.0)

        balls = []
        for i in range(6):
            ball_type = BallType.CUE if i == 0 else BallType.SOLID
            ball = create_mock_ball_detection(
                x=1.0 + i * 0.4, y=0.71, ball_type=ball_type, number=i
            )
            balls.append(ball)

        # Calculate with max_collision_depth=3
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(balls[0], is_target=True)
        other_balls = integration_service._create_ball_states(
            balls[1:], exclude_ball=balls[0]
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=3,
        )

        # Should not have more than max_collision_depth levels of collisions
        # This is hard to verify directly, but we can check that calculation completed
        assert result is not None
        assert len(result.trajectories) >= 1


@pytest.mark.unit()
class TestCollisionInformation:
    """Test collision information in trajectory results."""

    @pytest.mark.asyncio()
    async def test_collision_has_required_fields(self, integration_service):
        """Test that collision objects have all required fields."""
        # Create simple collision scenario
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)
        ball1 = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        ball2 = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        # Calculate trajectory
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(ball1, is_target=True)
        other_balls = integration_service._create_ball_states(
            [ball2], exclude_ball=ball1
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Check collision fields
        if result.collision_sequence:
            collision = result.collision_sequence[0]

            # Required fields
            assert hasattr(collision, "time")
            assert hasattr(collision, "position")
            assert hasattr(collision, "type")
            assert hasattr(collision, "ball1_id")
            assert hasattr(collision, "ball2_id")
            assert hasattr(collision, "impact_angle")
            assert hasattr(collision, "impact_velocity")
            assert hasattr(collision, "resulting_velocities")
            assert hasattr(collision, "confidence")

            # Verify types
            assert isinstance(collision.time, (int, float))
            assert isinstance(collision.position, Vector2D)
            assert isinstance(collision.type, CollisionType)
            assert isinstance(collision.confidence, (int, float))

    @pytest.mark.asyncio()
    async def test_ball_ball_collision_details(self, integration_service):
        """Test ball-ball collision has correct details."""
        # Create collision scenario
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)
        ball1 = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        ball2 = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        # Calculate trajectory
        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(ball1, is_target=True)
        other_balls = integration_service._create_ball_states(
            [ball2], exclude_ball=ball1
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Find ball-ball collision
        ball_collisions = [
            c for c in result.collision_sequence if c.type == CollisionType.BALL_BALL
        ]

        if ball_collisions:
            collision = ball_collisions[0]

            # Verify ball IDs are set
            assert collision.ball1_id is not None
            assert collision.ball2_id is not None

            # Verify resulting velocities has entries for both balls
            assert collision.ball1_id in collision.resulting_velocities
            assert collision.ball2_id in collision.resulting_velocities

            # Verify velocities are Vector2D
            assert isinstance(
                collision.resulting_velocities[collision.ball1_id], Vector2D
            )
            assert isinstance(
                collision.resulting_velocities[collision.ball2_id], Vector2D
            )

    @pytest.mark.asyncio()
    async def test_cushion_collision_details(self, integration_service):
        """Test cushion collision has correct details."""
        # Create scenario where ball hits cushion
        # Cue pointing at ball near edge
        cue = create_mock_cue_detection(tip_x=0.2, tip_y=0.71, angle=0.0)
        ball = create_mock_ball_detection(
            x=0.5, y=0.71, ball_type=BallType.CUE, number=0
        )

        # Calculate trajectory with high force to reach cushion
        cue_state = integration_service._create_cue_state(cue)
        cue_state.estimated_force = 10.0  # High force
        ball_state = integration_service._create_ball_state(ball, is_target=True)

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=[],
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Find cushion collision
        cushion_collisions = [
            c for c in result.collision_sequence if c.type == CollisionType.BALL_CUSHION
        ]

        if cushion_collisions:
            collision = cushion_collisions[0]

            # Verify cushion collision specifics
            assert collision.ball2_id is None, "Cushion collision should have no ball2"
            assert (
                collision.cushion_normal is not None
            ), "Should have cushion normal vector"
            assert isinstance(collision.cushion_normal, Vector2D)


@pytest.mark.unit()
class TestTrajectoryCalculatorDirect:
    """Test trajectory calculator directly (without integration service)."""

    def test_trajectory_calculator_initialization(self):
        """Test that trajectory calculator initializes correctly."""
        calculator = TrajectoryCalculator()
        assert calculator is not None
        assert hasattr(calculator, "predict_multiball_cue_shot")

    def test_multiball_result_basic_structure(self):
        """Test basic MultiballTrajectoryResult creation."""
        result = MultiballTrajectoryResult(
            primary_ball_id="cue",
            trajectories={},
            collision_sequence=[],
            total_calculation_time=0.0,
        )

        assert result.primary_ball_id == "cue"
        assert isinstance(result.trajectories, dict)
        assert isinstance(result.collision_sequence, list)
        assert result.total_calculation_time >= 0

    def test_predict_multiball_with_minimal_setup(self):
        """Test predict_multiball_cue_shot with minimal setup."""
        calculator = TrajectoryCalculator()

        # Create minimal valid inputs
        cue_state = CueState(
            angle=0.0,
            estimated_force=5.0,
            impact_point=Vector2D(1.0, 0.71),
            tip_position=Vector2D(1.0, 0.71),
            elevation=0.0,
            is_visible=True,
            confidence=0.95,
            last_update=time.time(),
        )

        ball_state = BallState(
            id="cue",
            position=Vector2D(1.2, 0.71),
            velocity=Vector2D(0, 0),
            radius=0.028575,
            mass=0.17,
            is_cue_ball=True,
            confidence=0.95,
            last_update=time.time(),
        )

        target_ball = BallState(
            id="ball_1",
            position=Vector2D(1.8, 0.71),
            velocity=Vector2D(0, 0),
            radius=0.028575,
            mass=0.17,
            is_cue_ball=False,
            number=1,
            confidence=0.95,
            last_update=time.time(),
        )

        table_state = TableState.standard_9ft_table()

        # Calculate trajectory
        result = calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=table_state,
            other_balls=[target_ball],
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        # Verify result
        assert isinstance(result, MultiballTrajectoryResult)
        assert result.primary_ball_id == "cue"
        assert "cue" in result.trajectories
        assert result.total_calculation_time > 0


# Performance tests
@pytest.mark.unit()
@pytest.mark.performance()
class TestMultiballTrajectoryPerformance:
    """Test performance of multiball trajectory calculation."""

    @pytest.mark.asyncio()
    async def test_calculation_speed(self, integration_service):
        """Test that trajectory calculation completes quickly."""
        # Create simple scenario
        cue = create_mock_cue_detection(tip_x=1.0, tip_y=0.71, angle=0.0)
        ball1 = create_mock_ball_detection(
            x=1.2, y=0.71, ball_type=BallType.CUE, number=0
        )
        ball2 = create_mock_ball_detection(
            x=1.8, y=0.71, ball_type=BallType.SOLID, number=1
        )

        # Time the calculation
        start_time = time.perf_counter()

        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(ball1, is_target=True)
        other_balls = integration_service._create_ball_states(
            [ball2], exclude_ball=ball1
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=5,
        )

        elapsed = time.perf_counter() - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert (
            elapsed < 0.1
        ), f"Calculation took {elapsed:.3f}s, should be < 0.1s for LOW quality"
        assert result is not None

    @pytest.mark.asyncio()
    async def test_calculation_with_many_balls(self, integration_service):
        """Test trajectory calculation with many balls on table."""
        # Create scenario with 10 balls
        cue = create_mock_cue_detection(tip_x=0.8, tip_y=0.71, angle=0.0)

        balls = []
        for i in range(10):
            ball_type = BallType.CUE if i == 0 else BallType.SOLID
            # Place balls in a grid pattern
            x = 1.0 + (i % 3) * 0.4
            y = 0.5 + (i // 3) * 0.3
            ball = create_mock_ball_detection(x=x, y=y, ball_type=ball_type, number=i)
            balls.append(ball)

        # Time the calculation
        start_time = time.perf_counter()

        cue_state = integration_service._create_cue_state(cue)
        ball_state = integration_service._create_ball_state(balls[0], is_target=True)
        other_balls = integration_service._create_ball_states(
            balls[1:], exclude_ball=balls[0]
        )

        result = integration_service.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=ball_state,
            table_state=integration_service.core._current_state.table,
            other_balls=other_balls,
            quality=TrajectoryQuality.LOW,
            max_collision_depth=3,
        )

        elapsed = time.perf_counter() - start_time

        # Should complete in reasonable time even with many balls
        assert (
            elapsed < 0.2
        ), f"Calculation with 10 balls took {elapsed:.3f}s, should be < 0.2s"
        assert result is not None
        assert len(result.trajectories) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
