#!/usr/bin/env python3
"""Basic tests for core models to validate the system."""

import time

from .core.models import (
    BallState,
    CueState,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
    Vector2D,
)


def test_vector2d():
    """Test Vector2D operations."""
    v1 = Vector2D(3.0, 4.0)
    v2 = Vector2D(1.0, 2.0)

    # Test magnitude
    assert abs(v1.magnitude() - 5.0) < 0.001

    # Test addition
    v3 = v1 + v2
    assert v3.x == 4.0
    assert v3.y == 6.0

    # Test distance
    assert abs(v1.distance_to(v2) - 2.828) < 0.01

    print("✓ Vector2D tests passed")


def test_ball_state():
    """Test BallState creation and methods."""
    ball = BallState(
        id="test_ball",
        position=Vector2D(1.0, 2.0),
        velocity=Vector2D(0.5, 0.3),
        number=1,
    )

    assert ball.id == "test_ball"
    assert ball.position.x == 1.0
    assert ball.position.y == 2.0
    assert ball.number == 1
    assert not ball.is_cue_ball
    assert not ball.is_pocketed

    # Test movement detection
    assert ball.is_moving(threshold=0.1)

    # Test kinetic energy calculation
    ke = ball.kinetic_energy()
    assert ke > 0

    print("✓ BallState tests passed")


def test_table_state():
    """Test TableState creation."""
    table = TableState.standard_9ft_table()

    assert table.width > 0
    assert table.height > 0
    assert len(table.pocket_positions) == 6

    # Test point validation
    center_point = Vector2D(table.width / 2, table.height / 2)
    assert table.is_point_on_table(center_point)

    outside_point = Vector2D(-1.0, -1.0)
    assert not table.is_point_on_table(outside_point)

    print("✓ TableState tests passed")


def test_game_state():
    """Test GameState creation."""
    table = TableState.standard_9ft_table()
    balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(id="1", position=Vector2D(2.0, 1.0), number=1),
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    assert len(game_state.balls) == 2
    assert game_state.get_cue_ball() is not None
    assert game_state.get_cue_ball().id == "cue"
    assert len(game_state.get_numbered_balls()) == 1

    # Test validation
    errors = game_state.validate_consistency()
    assert len(errors) == 0  # Should be valid

    print("✓ GameState tests passed")


def test_shot_analysis():
    """Test ShotAnalysis creation."""
    shot = ShotAnalysis(
        shot_type=ShotType.DIRECT,
        difficulty=0.5,
        success_probability=0.8,
        recommended_force=50.0,
        recommended_angle=45.0,
        target_ball_id="1",
    )

    assert shot.shot_type == ShotType.DIRECT
    assert shot.difficulty == 0.5
    assert shot.success_probability == 0.8
    assert shot.is_safe_shot(safety_threshold=0.7)
    assert not shot.is_high_risk()

    print("✓ ShotAnalysis tests passed")


def test_cue_state():
    """Test CueState creation."""
    cue = CueState(tip_position=Vector2D(1.0, 1.0), angle=45.0, estimated_force=30.0)

    assert cue.angle == 45.0
    direction = cue.get_direction_vector()
    assert abs(direction.x - 0.707) < 0.01  # cos(45°)
    assert abs(direction.y - 0.707) < 0.01  # sin(45°)

    print("✓ CueState tests passed")


def test_create_initial_state():
    """Test creating initial game state."""
    initial_state = GameState.create_initial_state(GameType.PRACTICE)

    assert len(initial_state.balls) == 16  # 15 numbered + 1 cue
    assert initial_state.get_cue_ball() is not None
    assert len(initial_state.get_numbered_balls()) == 15
    assert initial_state.is_break

    print("✓ Initial state creation tests passed")


if __name__ == "__main__":
    """Run all tests."""
    print("Running basic model tests...")

    test_vector2d()
    test_ball_state()
    test_table_state()
    test_game_state()
    test_shot_analysis()
    test_cue_state()
    test_create_initial_state()

    print("\n🎉 All basic model tests passed!")
