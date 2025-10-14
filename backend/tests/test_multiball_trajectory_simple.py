"""Simplified unit tests for multiball trajectory calculation.

This is a standalone test file that doesn't depend on conftest.py
to avoid import issues during test execution.
"""

import sys
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import asyncio

# Now import backend modules (import directly from module files to avoid circular dependencies)
# Importing from core.__init__ triggers a chain that hits backend.config which isn't available
# So we import directly from the module files
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models directly from the .py file, not through core/__init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "core_models", backend_path / "core" / "models.py"
)
core_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_models)

BallState = core_models.BallState
CueState = core_models.CueState
TableState = core_models.TableState
Vector2D = core_models.Vector2D

spec = importlib.util.spec_from_file_location(
    "core_trajectory", backend_path / "core" / "physics" / "trajectory.py"
)
core_trajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_trajectory)

CollisionType = core_trajectory.CollisionType
MultiballTrajectoryResult = core_trajectory.MultiballTrajectoryResult
TrajectoryCalculator = core_trajectory.TrajectoryCalculator
TrajectoryQuality = core_trajectory.TrajectoryQuality


# Test the trajectory calculator directly
def test_trajectory_calculator_initialization():
    """Test that trajectory calculator initializes correctly."""
    calculator = TrajectoryCalculator()
    assert calculator is not None
    assert hasattr(calculator, "predict_multiball_cue_shot")
    print("✓ Trajectory calculator initialized successfully")


def test_multiball_result_structure():
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
    print("✓ MultiballTrajectoryResult structure validated")


def test_simple_two_ball_collision():
    """Test trajectory calculation for simple two-ball collision."""
    calculator = TrajectoryCalculator()

    # Create cue state pointing right at 0 degrees
    cue_state = CueState(
        angle=0.0,
        estimated_force=5.0,
        impact_point=Vector2D(1.2, 0.71),
        tip_position=Vector2D(1.0, 0.71),
        elevation=0.0,
        is_visible=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(1.2, 0.71),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create target ball directly in line
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
    start_time = time.perf_counter()
    result = calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=cue_ball,
        table_state=table_state,
        other_balls=[target_ball],
        quality=TrajectoryQuality.LOW,
        max_collision_depth=5,
    )
    elapsed = time.perf_counter() - start_time

    # Verify result
    assert isinstance(
        result, MultiballTrajectoryResult
    ), "Result should be MultiballTrajectoryResult"
    assert result.primary_ball_id == "cue", "Primary ball should be cue"
    assert "cue" in result.trajectories, "Should have trajectory for cue ball"
    assert (
        len(result.trajectories["cue"].points) > 0
    ), "Cue ball should have trajectory points"

    # Check for collisions
    assert len(result.collision_sequence) > 0, "Should have at least one collision"

    # Verify collision types
    ball_collisions = [
        c for c in result.collision_sequence if c.type == CollisionType.BALL_BALL
    ]

    print(f"✓ Simple two-ball collision calculated in {elapsed*1000:.2f}ms")
    print(f"  - Trajectories: {len(result.trajectories)} ball(s)")
    print(f"  - Total collisions: {len(result.collision_sequence)}")
    print(f"  - Ball-ball collisions: {len(ball_collisions)}")
    print(f"  - Cue ball trajectory points: {len(result.trajectories['cue'].points)}")

    # If we have a ball-ball collision, check for secondary trajectory
    if ball_collisions:
        collision = ball_collisions[0]
        print(f"  - First collision: {collision.ball1_id} -> {collision.ball2_id}")
        print(
            f"  - Collision position: ({collision.position.x:.2f}, {collision.position.y:.2f})"
        )

        # Check if target ball has a trajectory
        if "ball_1" in result.trajectories:
            print(
                f"  - Target ball trajectory: {len(result.trajectories['ball_1'].points)} points"
            )


def test_no_target_ball():
    """Test trajectory when cue points at nothing."""
    calculator = TrajectoryCalculator()

    # Create cue state pointing away from balls (angle 180 degrees = pointing left)
    cue_state = CueState(
        angle=180.0,
        estimated_force=5.0,
        impact_point=Vector2D(0.5, 0.5),
        tip_position=Vector2D(0.5, 0.5),
        elevation=0.0,
        is_visible=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(0.7, 0.5),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create target ball far away to the right (cue is pointing left)
    target_ball = BallState(
        id="ball_1",
        position=Vector2D(2.0, 1.0),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=False,
        number=1,
        confidence=0.95,
        last_update=time.time(),
    )

    table_state = TableState.standard_9ft_table()

    # Calculate trajectory - should not hit the ball
    result = calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=cue_ball,
        table_state=table_state,
        other_balls=[target_ball],
        quality=TrajectoryQuality.LOW,
        max_collision_depth=5,
    )

    # Should have cue ball trajectory but probably no ball-ball collision
    assert isinstance(result, MultiballTrajectoryResult)
    assert "cue" in result.trajectories

    ball_collisions = [
        c for c in result.collision_sequence if c.type == CollisionType.BALL_BALL
    ]
    print(
        f"✓ No target ball test: {len(ball_collisions)} ball-ball collisions (expected 0)"
    )


def test_three_ball_chain():
    """Test collision chain with three balls in a line."""
    calculator = TrajectoryCalculator()

    # Create cue pointing at first ball
    cue_state = CueState(
        angle=0.0,
        estimated_force=8.0,  # Higher force for chain reaction
        impact_point=Vector2D(1.0, 0.71),
        tip_position=Vector2D(1.0, 0.71),
        elevation=0.0,
        is_visible=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create three balls in a line
    ball1 = BallState(
        id="cue",
        position=Vector2D(1.2, 0.71),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=True,
        number=0,
        confidence=0.95,
        last_update=time.time(),
    )

    ball2 = BallState(
        id="ball_1",
        position=Vector2D(1.6, 0.71),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=False,
        number=1,
        confidence=0.95,
        last_update=time.time(),
    )

    ball3 = BallState(
        id="ball_2",
        position=Vector2D(2.0, 0.71),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=False,
        number=2,
        confidence=0.95,
        last_update=time.time(),
    )

    table_state = TableState.standard_9ft_table()

    # Calculate trajectory
    result = calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=ball1,
        table_state=table_state,
        other_balls=[ball2, ball3],
        quality=TrajectoryQuality.LOW,
        max_collision_depth=5,
    )

    # Should have trajectories for multiple balls
    assert isinstance(result, MultiballTrajectoryResult)
    assert (
        len(result.trajectories) >= 1
    ), "Should have at least the primary ball trajectory"

    ball_collisions = [
        c for c in result.collision_sequence if c.type == CollisionType.BALL_BALL
    ]

    print("✓ Three-ball chain test:")
    print(f"  - Trajectories: {len(result.trajectories)} ball(s)")
    print(f"  - Ball IDs with trajectories: {list(result.trajectories.keys())}")
    print(f"  - Total collisions: {len(result.collision_sequence)}")
    print(f"  - Ball-ball collisions: {len(ball_collisions)}")


def test_performance_with_many_balls():
    """Test performance with many balls on table."""
    calculator = TrajectoryCalculator()

    # Create cue
    cue_state = CueState(
        angle=45.0,  # Diagonal shot
        estimated_force=6.0,
        impact_point=Vector2D(0.8, 0.5),
        tip_position=Vector2D(0.8, 0.5),
        elevation=0.0,
        is_visible=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(1.0, 0.5),
        velocity=Vector2D(0, 0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=True,
        confidence=0.95,
        last_update=time.time(),
    )

    # Create 10 balls in various positions
    other_balls = []
    for i in range(10):
        x = 1.5 + (i % 3) * 0.3
        y = 0.5 + (i // 3) * 0.25
        ball = BallState(
            id=f"ball_{i+1}",
            position=Vector2D(x, y),
            velocity=Vector2D(0, 0),
            radius=0.028575,
            mass=0.17,
            is_cue_ball=False,
            number=i + 1,
            confidence=0.95,
            last_update=time.time(),
        )
        other_balls.append(ball)

    table_state = TableState.standard_9ft_table()

    # Time the calculation
    start_time = time.perf_counter()
    result = calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=cue_ball,
        table_state=table_state,
        other_balls=other_balls,
        quality=TrajectoryQuality.LOW,
        max_collision_depth=3,
    )
    elapsed = time.perf_counter() - start_time

    assert isinstance(result, MultiballTrajectoryResult)
    assert elapsed < 0.5, f"Calculation took too long: {elapsed:.3f}s"

    print(f"✓ Performance test with 10 balls: {elapsed*1000:.2f}ms")
    print(f"  - Trajectories calculated: {len(result.trajectories)}")
    print(f"  - Total collisions: {len(result.collision_sequence)}")


def test_collision_info_structure():
    """Test that collision objects have required fields."""
    calculator = TrajectoryCalculator()

    # Create simple collision scenario
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

    cue_ball = BallState(
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

    result = calculator.predict_multiball_cue_shot(
        cue_state=cue_state,
        ball_state=cue_ball,
        table_state=table_state,
        other_balls=[target_ball],
        quality=TrajectoryQuality.LOW,
        max_collision_depth=5,
    )

    # Check collision fields
    if result.collision_sequence:
        collision = result.collision_sequence[0]

        # Required fields
        assert hasattr(collision, "time"), "Collision missing 'time' field"
        assert hasattr(collision, "position"), "Collision missing 'position' field"
        assert hasattr(collision, "type"), "Collision missing 'type' field"
        assert hasattr(collision, "ball1_id"), "Collision missing 'ball1_id' field"
        assert hasattr(collision, "ball2_id"), "Collision missing 'ball2_id' field"
        assert hasattr(
            collision, "impact_angle"
        ), "Collision missing 'impact_angle' field"
        assert hasattr(
            collision, "impact_velocity"
        ), "Collision missing 'impact_velocity' field"
        assert hasattr(
            collision, "resulting_velocities"
        ), "Collision missing 'resulting_velocities' field"
        assert hasattr(collision, "confidence"), "Collision missing 'confidence' field"

        # Verify types
        assert isinstance(
            collision.time, (int, float)
        ), f"Collision time should be numeric, got {type(collision.time)}"
        assert isinstance(
            collision.position, Vector2D
        ), f"Collision position should be Vector2D, got {type(collision.position)}"
        assert isinstance(
            collision.type, CollisionType
        ), f"Collision type should be CollisionType, got {type(collision.type)}"
        assert isinstance(
            collision.confidence, (int, float)
        ), f"Collision confidence should be numeric, got {type(collision.confidence)}"

        print("✓ Collision information structure validated")
        print(f"  - Collision type: {collision.type.value}")
        print(f"  - Ball1: {collision.ball1_id}, Ball2: {collision.ball2_id}")
        print(f"  - Position: ({collision.position.x:.2f}, {collision.position.y:.2f})")
        print(f"  - Confidence: {collision.confidence:.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Running multiball trajectory calculation tests")
    print("=" * 70)
    print()

    tests = [
        (
            "Trajectory Calculator Initialization",
            test_trajectory_calculator_initialization,
        ),
        ("MultiballTrajectoryResult Structure", test_multiball_result_structure),
        ("Simple Two-Ball Collision", test_simple_two_ball_collision),
        ("No Target Ball", test_no_target_ball),
        ("Three-Ball Collision Chain", test_three_ball_chain),
        ("Performance with Many Balls", test_performance_with_many_balls),
        ("Collision Information Structure", test_collision_info_structure),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            print(f"[TEST] {name}")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            print()

    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
