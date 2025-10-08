#!/usr/bin/env python
"""Test the new geometric trajectory calculator."""

from backend.core.models import BallState, TableState, CueState, Vector2D
from backend.core.physics.trajectory import TrajectoryCalculator, TrajectoryQuality

def test_simple_straight_line():
    """Test a simple straight trajectory with no collisions."""
    print("Test 1: Simple straight line trajectory")

    # Create table
    table = TableState.standard_9ft_table()

    # Create ball in center moving right
    ball = BallState(
        id="test_ball",
        position=Vector2D(1.0, 0.635),  # Center of table
        velocity=Vector2D(1.0, 0.0),  # Moving right
        is_cue_ball=True,
    )

    # Calculate trajectory
    calc = TrajectoryCalculator()
    trajectory = calc.calculate_trajectory(ball, table, [], TrajectoryQuality.MEDIUM)

    print(f"  - Points generated: {len(trajectory.points)}")
    print(f"  - Collisions: {len(trajectory.collisions)}")
    print(f"  - Final position: ({trajectory.final_position.x:.3f}, {trajectory.final_position.y:.3f})")

    # Should hit right cushion
    assert len(trajectory.collisions) > 0, "Should have at least one collision"
    assert trajectory.collisions[0].type.value == "ball_cushion", "First collision should be cushion"
    print("  ✓ Test passed!")


def test_cushion_bounce():
    """Test trajectory with cushion bounce."""
    print("\nTest 2: Cushion bounce")

    table = TableState.standard_9ft_table()

    # Ball moving diagonally toward bottom-right corner
    ball = BallState(
        id="test_ball",
        position=Vector2D(1.0, 0.5),
        velocity=Vector2D(1.0, -0.5),  # Moving right and down
        is_cue_ball=True,
    )

    calc = TrajectoryCalculator()
    trajectory = calc.calculate_trajectory(ball, table, [], TrajectoryQuality.MEDIUM)

    print(f"  - Points generated: {len(trajectory.points)}")
    print(f"  - Collisions: {len(trajectory.collisions)}")
    for i, coll in enumerate(trajectory.collisions):
        print(f"    {i+1}. {coll.type.value} at ({coll.position.x:.3f}, {coll.position.y:.3f})")

    # Should have multiple bounces
    assert len(trajectory.collisions) >= 1, "Should have at least one cushion collision"
    print("  ✓ Test passed!")


def test_ball_collision():
    """Test trajectory with ball-ball collision."""
    print("\nTest 3: Ball-ball collision")

    table = TableState.standard_9ft_table()

    # Cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(0.5, 0.635),
        velocity=Vector2D(1.0, 0.0),  # Moving right
        is_cue_ball=True,
    )

    # Target ball directly in path
    target_ball = BallState(
        id="ball_1",
        position=Vector2D(1.0, 0.635),
        velocity=Vector2D(0, 0),
        number=1,
    )

    calc = TrajectoryCalculator()
    trajectory = calc.calculate_trajectory(cue_ball, table, [target_ball], TrajectoryQuality.MEDIUM)

    print(f"  - Points generated: {len(trajectory.points)}")
    print(f"  - Collisions: {len(trajectory.collisions)}")
    for i, coll in enumerate(trajectory.collisions):
        print(f"    {i+1}. {coll.type.value} at ({coll.position.x:.3f}, {coll.position.y:.3f})")
        if coll.ball2_id:
            print(f"       with {coll.ball2_id}")

    # Should hit the target ball
    assert len(trajectory.collisions) >= 1, "Should have at least one collision"
    has_ball_collision = any(c.type.value == "ball_ball" for c in trajectory.collisions)
    assert has_ball_collision, "Should have a ball-ball collision"
    print("  ✓ Test passed!")


def test_multiball_shot():
    """Test multiball trajectory prediction."""
    print("\nTest 4: Multiball cue shot")

    table = TableState.standard_9ft_table()

    # Cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(0.5, 0.635),
        is_cue_ball=True,
    )

    # Target ball
    target_ball = BallState(
        id="ball_1",
        position=Vector2D(1.2, 0.635),
        number=1,
    )

    # Cue stick aiming at target
    cue = CueState(
        tip_position=Vector2D(0.4, 0.635),
        angle=0.0,  # Pointing right
        estimated_force=5.0,
    )

    calc = TrajectoryCalculator()
    result = calc.predict_multiball_cue_shot(cue, cue_ball, table, [target_ball], TrajectoryQuality.MEDIUM)

    print(f"  - Balls with trajectories: {len(result.trajectories)}")
    print(f"  - Total collisions: {len(result.collision_sequence)}")
    for ball_id, traj in result.trajectories.items():
        print(f"    {ball_id}: {len(traj.points)} points, {len(traj.collisions)} collisions")

    assert "cue" in result.trajectories, "Should have cue ball trajectory"
    assert len(result.trajectories) >= 1, "Should have at least one trajectory"
    print("  ✓ Test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Geometric Trajectory Calculator")
    print("=" * 60)

    test_simple_straight_line()
    test_cushion_bounce()
    test_ball_collision()
    test_multiball_shot()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
