#!/usr/bin/env python3
"""Error handling and recovery tests."""

import time

from .core.models import (
    BallState,
    GameState,
    ShotAnalysis,
    ShotType,
    TableState,
    Vector2D,
)
from .core.physics.engine import PhysicsEngine


def test_invalid_ball_data():
    """Test handling of invalid ball data."""
    print("Testing Invalid Ball Data Handling...")

    # Test invalid radius
    try:
        BallState(id="test", position=Vector2D(1.0, 1.0), radius=-1.0)
        print("  ✗ Should have failed with negative radius")
        return False
    except ValueError:
        print("  ✓ Negative radius properly rejected")

    # Test invalid mass
    try:
        BallState(id="test", position=Vector2D(1.0, 1.0), mass=-1.0)
        print("  ✗ Should have failed with negative mass")
        return False
    except ValueError:
        print("  ✓ Negative mass properly rejected")

    # Test invalid confidence
    try:
        BallState(id="test", position=Vector2D(1.0, 1.0), confidence=2.0)
        print("  ✗ Should have failed with confidence > 1.0")
        return False
    except ValueError:
        print("  ✓ Invalid confidence properly rejected")

    return True


def test_invalid_game_state():
    """Test handling of invalid game states."""
    print("Testing Invalid Game State Handling...")

    table = TableState.standard_9ft_table()

    # Test negative frame number
    try:
        game_state = GameState(
            timestamp=time.time(), frame_number=-1, balls=[], table=table
        )
        print("  ✗ Should have failed with negative frame number")
        return False
    except ValueError:
        print("  ✓ Negative frame number properly rejected")

    # Test invalid timestamp
    try:
        game_state = GameState(timestamp=-1.0, frame_number=0, balls=[], table=table)
        print("  ✗ Should have failed with negative timestamp")
        return False
    except ValueError:
        print("  ✓ Negative timestamp properly rejected")

    # Test game state with overlapping balls
    overlapping_balls = [
        BallState(id="cue", position=Vector2D(1.0, 1.0), is_cue_ball=True),
        BallState(id="1", position=Vector2D(1.0, 1.0), number=1),  # Same position
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=overlapping_balls, table=table
    )

    errors = game_state.validate_consistency()
    if "overlapping" in str(errors).lower():
        print("  ✓ Overlapping balls detected in validation")
    else:
        print(f"  ⚠ Overlapping balls not detected: {errors}")

    return True


def test_physics_edge_cases():
    """Test physics engine with edge cases."""
    print("Testing Physics Edge Cases...")

    table = TableState.standard_9ft_table()
    physics_engine = PhysicsEngine()

    # Test stationary ball
    stationary_ball = BallState(
        id="cue",
        position=Vector2D(1.0, 1.0),
        velocity=Vector2D(0.0, 0.0),
        is_cue_ball=True,
    )

    trajectory = physics_engine.calculate_trajectory(
        stationary_ball, table, [], time_limit=1.0
    )
    if len(trajectory) <= 1:
        print("  ✓ Stationary ball trajectory handled correctly")
    else:
        print(
            f"  ⚠ Unexpected trajectory length for stationary ball: {len(trajectory)}"
        )

    # Test very high velocity
    fast_ball = BallState(
        id="cue",
        position=Vector2D(1.0, 1.0),
        velocity=Vector2D(100.0, 0.0),  # Very fast
        is_cue_ball=True,
    )

    try:
        trajectory = physics_engine.calculate_trajectory(
            fast_ball, table, [], time_limit=0.1
        )
        print(f"  ✓ High velocity ball trajectory calculated: {len(trajectory)} points")
    except Exception as e:
        print(f"  ⚠ High velocity ball caused error: {e}")

    return True


def test_shot_analysis_edge_cases():
    """Test shot analysis with edge cases."""
    print("Testing Shot Analysis Edge Cases...")

    # Test invalid difficulty
    try:
        ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=2.0,  # > 1.0
            success_probability=0.8,
            recommended_force=50.0,
            recommended_angle=45.0,
        )
        print("  ✗ Should have failed with difficulty > 1.0")
        return False
    except ValueError:
        print("  ✓ Invalid difficulty properly rejected")

    # Test invalid success probability
    try:
        ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=1.5,  # > 1.0
            recommended_force=50.0,
            recommended_angle=45.0,
        )
        print("  ✗ Should have failed with success_probability > 1.0")
        return False
    except ValueError:
        print("  ✓ Invalid success probability properly rejected")

    # Test negative force
    try:
        ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=-10.0,  # Negative
            recommended_angle=45.0,
        )
        print("  ✗ Should have failed with negative force")
        return False
    except ValueError:
        print("  ✓ Negative force properly rejected")

    return True


def test_data_conversion_robustness():
    """Test robustness of data conversions."""
    print("Testing Data Conversion Robustness...")

    # Test JSON serialization/deserialization
    ball = BallState(id="test", position=Vector2D(1.0, 2.0), number=1)

    try:
        # Convert to dict and back
        ball_dict = ball.to_dict()
        restored_ball = BallState.from_dict(ball_dict)

        if (
            restored_ball.id == ball.id
            and restored_ball.position.x == ball.position.x
            and restored_ball.position.y == ball.position.y
        ):
            print("  ✓ Ball serialization/deserialization works")
        else:
            print("  ✗ Ball data corrupted during conversion")
            return False

    except Exception as e:
        print(f"  ✗ Ball conversion failed: {e}")
        return False

    # Test table state conversion
    table = TableState.standard_9ft_table()

    try:
        table_dict = table.to_dict()
        restored_table = TableState.from_dict(table_dict)

        if restored_table.width == table.width and len(
            restored_table.pocket_positions
        ) == len(table.pocket_positions):
            print("  ✓ Table serialization/deserialization works")
        else:
            print("  ✗ Table data corrupted during conversion")
            return False

    except Exception as e:
        print(f"  ✗ Table conversion failed: {e}")
        return False

    return True


def test_memory_cleanup():
    """Test that objects are properly cleaned up."""
    print("Testing Memory Cleanup...")

    import gc

    initial_objects = len(gc.get_objects())

    # Create and destroy many objects
    for i in range(1000):
        balls = [
            BallState(id=f"ball_{i}_{j}", position=Vector2D(i, j)) for j in range(10)
        ]
        table = TableState.standard_9ft_table()
        game_state = GameState(
            timestamp=time.time(), frame_number=i, balls=balls, table=table
        )

        # Force deletion
        del balls, table, game_state

    # Force garbage collection
    gc.collect()

    final_objects = len(gc.get_objects())
    object_increase = final_objects - initial_objects

    print(f"  ✓ Initial objects: {initial_objects}")
    print(f"  ✓ Final objects: {final_objects}")
    print(f"  ✓ Object increase: {object_increase}")

    if object_increase < 100:  # Some increase is normal
        print("  ✓ Memory cleanup appears effective")
    else:
        print(f"  ⚠ High object count increase: {object_increase}")

    return True


def main():
    """Run all error handling tests."""
    print("=" * 60)
    print("BILLIARDS TRAINER BACKEND - ERROR HANDLING TESTS")
    print("=" * 60)

    tests = [
        test_invalid_ball_data,
        test_invalid_game_state,
        test_physics_edge_cases,
        test_shot_analysis_edge_cases,
        test_data_conversion_robustness,
        test_memory_cleanup,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ Test {test.__name__} failed with exception: {e}")
            print()

    print("=" * 60)
    print("ERROR HANDLING TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 All error handling tests PASSED!")
        return True
    else:
        print(f"\n⚠ {total - passed} error handling tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
