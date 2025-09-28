#!/usr/bin/env python3
"""Comprehensive test script for the physics validation system.

This script tests various physics scenarios to ensure the validation system
properly catches invalid states and maintains system reliability.
"""

import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from core.models import BallState, Collision, TableState, Vector2D
from core.physics.engine import TrajectoryPoint
from core.validation.physics import PhysicsValidator


def create_test_ball(
    ball_id: str = "test_ball",
    position: Vector2D = Vector2D(1.0, 1.0),
    velocity: Vector2D = Vector2D(0.0, 0.0),
    radius: float = 0.028575,
    mass: float = 0.17,
) -> BallState:
    """Create a test ball with default parameters."""
    return BallState(
        id=ball_id,
        position=position,
        velocity=velocity,
        radius=radius,
        mass=mass,
        spin=Vector2D(0.0, 0.0),
    )


def create_test_table() -> TableState:
    """Create a standard test table."""
    return TableState.standard_9ft_table()


def test_ball_state_validation():
    """Test ball state validation with various scenarios."""
    print("Testing ball state validation...")
    validator = PhysicsValidator()

    # Test valid ball
    valid_ball = create_test_ball()
    result = validator.validate_ball_state(valid_ball)
    assert result.is_valid, f"Valid ball failed validation: {result.errors}"
    print("✓ Valid ball state passed")

    # Test invalid radius - create ball with valid constructor then modify
    invalid_radius_ball = create_test_ball()
    invalid_radius_ball.radius = -0.1  # Modify after creation
    result = validator.validate_ball_state(invalid_radius_ball)
    assert not result.is_valid, "Invalid radius ball should fail validation"
    assert any("invalid_radius" in error.error_type for error in result.errors)
    print("✓ Invalid radius ball properly rejected")

    # Test invalid mass - create ball with valid constructor then modify
    invalid_mass_ball = create_test_ball()
    invalid_mass_ball.mass = -0.1  # Modify after creation
    result = validator.validate_ball_state(invalid_mass_ball)
    assert not result.is_valid, "Invalid mass ball should fail validation"
    assert any("invalid_mass" in error.error_type for error in result.errors)
    print("✓ Invalid mass ball properly rejected")

    # Test excessive velocity
    high_velocity_ball = create_test_ball(
        velocity=Vector2D(25.0, 0.0)
    )  # Above 20 m/s limit
    result = validator.validate_ball_state(high_velocity_ball)
    assert not result.is_valid, "High velocity ball should fail validation"
    assert any("velocity_limit" in error.error_type for error in result.errors)
    print("✓ High velocity ball properly rejected")

    # Test ball out of bounds
    table = create_test_table()
    out_of_bounds_ball = create_test_ball(position=Vector2D(-0.1, 0.5))
    result = validator.validate_ball_state(out_of_bounds_ball, table)
    assert not result.is_valid, "Out of bounds ball should fail validation"
    assert any("position_out_of_bounds" in error.error_type for error in result.errors)
    print("✓ Out of bounds ball properly rejected")

    print("Ball state validation tests passed!\n")


def test_trajectory_validation():
    """Test trajectory validation with various scenarios."""
    print("Testing trajectory validation...")
    validator = PhysicsValidator()

    # Test valid trajectory
    valid_trajectory = [
        TrajectoryPoint(
            time=0.0, position=Vector2D(1.0, 1.0), velocity=Vector2D(2.0, 0.0)
        ),
        TrajectoryPoint(
            time=0.1, position=Vector2D(1.2, 1.0), velocity=Vector2D(1.8, 0.0)
        ),
        TrajectoryPoint(
            time=0.2, position=Vector2D(1.4, 1.0), velocity=Vector2D(1.6, 0.0)
        ),
    ]
    result = validator.validate_trajectory(valid_trajectory)
    assert result.is_valid, f"Valid trajectory failed validation: {result.errors}"
    print("✓ Valid trajectory passed")

    # Test empty trajectory
    empty_trajectory = []
    result = validator.validate_trajectory(empty_trajectory)
    assert not result.is_valid, "Empty trajectory should fail validation"
    assert any("empty_trajectory" in error.error_type for error in result.errors)
    print("✓ Empty trajectory properly rejected")

    # Test invalid time sequence
    invalid_time_trajectory = [
        TrajectoryPoint(
            time=0.0, position=Vector2D(1.0, 1.0), velocity=Vector2D(2.0, 0.0)
        ),
        TrajectoryPoint(
            time=0.1, position=Vector2D(1.2, 1.0), velocity=Vector2D(1.8, 0.0)
        ),
        TrajectoryPoint(
            time=0.05, position=Vector2D(1.1, 1.0), velocity=Vector2D(1.9, 0.0)
        ),  # Time goes backward
    ]
    result = validator.validate_trajectory(invalid_time_trajectory)
    assert not result.is_valid, "Invalid time sequence should fail validation"
    assert any("time_sequence" in error.error_type for error in result.errors)
    print("✓ Invalid time sequence properly rejected")

    # Test excessive velocity in trajectory
    high_velocity_trajectory = [
        TrajectoryPoint(
            time=0.0, position=Vector2D(1.0, 1.0), velocity=Vector2D(25.0, 0.0)
        ),  # Above limit
    ]
    result = validator.validate_trajectory(high_velocity_trajectory)
    assert not result.is_valid, "High velocity trajectory should fail validation"
    assert any("velocity_limit" in error.error_type for error in result.errors)
    print("✓ High velocity trajectory properly rejected")

    print("Trajectory validation tests passed!\n")


def test_collision_validation():
    """Test collision validation with various scenarios."""
    print("Testing collision validation...")
    validator = PhysicsValidator()

    ball1 = create_test_ball("ball1", Vector2D(1.0, 1.0), Vector2D(1.0, 0.0))
    ball2 = create_test_ball("ball2", Vector2D(1.1, 1.0), Vector2D(-1.0, 0.0))

    # Test valid ball collision
    valid_collision = Collision(
        time=0.1,
        position=Vector2D(1.05, 1.0),
        ball1_id="ball1",
        ball2_id="ball2",
        type="ball",
        impact_force=5.0,
    )
    result = validator.validate_collision(valid_collision, ball1, ball2)
    assert result.is_valid, f"Valid collision failed validation: {result.errors}"
    print("✓ Valid ball collision passed")

    # Test collision type mismatch - create valid collision then modify
    invalid_type_collision = Collision(
        time=0.1,
        position=Vector2D(1.05, 1.0),
        ball1_id="ball1",
        ball2_id="ball2",  # Create valid first
        type="ball",
    )
    # Now modify to create invalid state
    invalid_type_collision.ball2_id = None
    result = validator.validate_collision(invalid_type_collision, ball1, None)
    assert not result.is_valid, "Type mismatch collision should fail validation"
    assert any("collision_type_mismatch" in error.error_type for error in result.errors)
    print("✓ Collision type mismatch properly rejected")

    # Test negative collision time
    past_collision = Collision(
        time=-0.1,  # Negative time
        position=Vector2D(1.05, 1.0),
        ball1_id="ball1",
        ball2_id="ball2",
        type="ball",
    )
    result = validator.validate_collision(past_collision, ball1, ball2)
    # This should generate a warning, not an error
    assert len(result.warnings) > 0, "Negative time should generate warning"
    assert any(
        "negative_collision_time" in warning.error_type for warning in result.warnings
    )
    print("✓ Negative collision time properly warned")

    # Test excessive impact force
    high_force_collision = Collision(
        time=0.1,
        position=Vector2D(1.05, 1.0),
        ball1_id="ball1",
        ball2_id="ball2",
        type="ball",
        impact_force=100.0,  # Very high force
    )
    result = validator.validate_collision(high_force_collision, ball1, ball2)
    # This should generate a warning
    assert len(result.warnings) > 0, "High impact force should generate warning"
    assert any("impact_force" in warning.error_type for warning in result.warnings)
    print("✓ High impact force properly warned")

    print("Collision validation tests passed!\n")


def test_force_validation():
    """Test force validation with various scenarios."""
    print("Testing force validation...")
    validator = PhysicsValidator()

    # Test valid force
    valid_force = Vector2D(5.0, 3.0)
    mass = 0.17  # Standard ball mass
    result = validator.validate_forces(valid_force, mass)
    assert result.is_valid, f"Valid force failed validation: {result.errors}"
    print("✓ Valid force passed")

    # Test excessive force causing high acceleration
    excessive_force = Vector2D(50.0, 0.0)  # Very high force
    result = validator.validate_forces(excessive_force, mass)
    # Should generate acceleration limit error
    assert not result.is_valid, "Excessive force should fail validation"
    assert any("acceleration_limit" in error.error_type for error in result.errors)
    print("✓ Excessive force properly rejected")

    # Test invalid mass
    valid_force = Vector2D(5.0, 3.0)
    invalid_mass = -0.1
    result = validator.validate_forces(valid_force, invalid_mass)
    assert not result.is_valid, "Invalid mass should fail validation"
    assert any("invalid_mass" in error.error_type for error in result.errors)
    print("✓ Invalid mass properly rejected")

    # Test NaN forces
    nan_force = Vector2D(float("nan"), 5.0)
    result = validator.validate_forces(nan_force, mass)
    assert not result.is_valid, "NaN force should fail validation"
    assert any("invalid_force_values" in error.error_type for error in result.errors)
    print("✓ NaN force properly rejected")

    # Test infinite forces
    inf_force = Vector2D(float("inf"), 5.0)
    result = validator.validate_forces(inf_force, mass)
    assert not result.is_valid, "Infinite force should fail validation"
    assert any("invalid_force_values" in error.error_type for error in result.errors)
    print("✓ Infinite force properly rejected")

    print("Force validation tests passed!\n")


def test_energy_conservation():
    """Test energy conservation validation."""
    print("Testing energy conservation validation...")
    # Use more lenient tolerance for test scenarios
    validator = PhysicsValidator({"energy_tolerance": 0.15})  # 15% tolerance

    # Test acceptable energy conservation (use momentum conserving collision)
    ball1_before = create_test_ball("ball1", velocity=Vector2D(2.0, 0.0))
    ball2_before = create_test_ball("ball2", velocity=Vector2D(0.0, 0.0))

    # After elastic collision: momentum conserved, energy mostly conserved
    # For equal mass elastic collision: v1_final = 0, v2_final = v1_initial
    ball1_after = create_test_ball("ball1", velocity=Vector2D(0.0, 0.0))
    ball2_after = create_test_ball(
        "ball2", velocity=Vector2D(1.9, 0.0)
    )  # Slight energy loss

    result = validator.validate_energy_conservation(
        [ball1_before, ball2_before], [ball1_after, ball2_after]
    )
    assert result.is_valid, f"Energy conservation failed: {result.errors}"
    print("✓ Acceptable energy conservation passed")

    # Test energy violation
    ball1_after_high = create_test_ball(
        "ball1", velocity=Vector2D(3.0, 0.0)
    )  # Too much energy
    ball2_after_high = create_test_ball("ball2", velocity=Vector2D(3.0, 0.0))

    result = validator.validate_energy_conservation(
        [ball1_before, ball2_before], [ball1_after_high, ball2_after_high]
    )
    assert not result.is_valid, "Energy violation should fail validation"
    assert any("energy_conservation" in error.error_type for error in result.errors)
    print("✓ Energy violation properly detected")

    # Test energy increase (should be impossible)
    result = validator.validate_energy_conservation(
        [ball1_before, ball2_before], [ball1_after_high, ball2_after_high]
    )
    assert not result.is_valid, "Energy increase should fail validation"
    print("✓ Energy increase properly detected")

    print("Energy conservation tests passed!\n")


def test_momentum_conservation():
    """Test momentum conservation validation."""
    print("Testing momentum conservation validation...")
    validator = PhysicsValidator()

    # Test perfect momentum conservation
    ball1_before = create_test_ball("ball1", velocity=Vector2D(2.0, 0.0))
    ball2_before = create_test_ball("ball2", velocity=Vector2D(0.0, 0.0))

    # After collision: momentum should be conserved
    ball1_after = create_test_ball("ball1", velocity=Vector2D(1.0, 0.0))
    ball2_after = create_test_ball("ball2", velocity=Vector2D(1.0, 0.0))

    result = validator.validate_momentum_conservation(
        [ball1_before, ball2_before], [ball1_after, ball2_after]
    )
    assert result.is_valid, f"Momentum conservation failed: {result.errors}"
    print("✓ Perfect momentum conservation passed")

    # Test momentum violation
    ball1_after_wrong = create_test_ball("ball1", velocity=Vector2D(0.0, 0.0))
    ball2_after_wrong = create_test_ball(
        "ball2", velocity=Vector2D(0.0, 0.0)
    )  # No momentum

    result = validator.validate_momentum_conservation(
        [ball1_before, ball2_before], [ball1_after_wrong, ball2_after_wrong]
    )
    assert not result.is_valid, "Momentum violation should fail validation"
    assert any("momentum_conservation" in error.error_type for error in result.errors)
    print("✓ Momentum violation properly detected")

    print("Momentum conservation tests passed!\n")


def test_system_state_validation():
    """Test comprehensive system state validation."""
    print("Testing system state validation...")
    validator = PhysicsValidator()
    table = create_test_table()

    # Test valid system
    ball1 = create_test_ball("ball1", Vector2D(1.0, 1.0), Vector2D(1.0, 0.0))
    ball1.is_cue_ball = True
    ball2 = create_test_ball("ball2", Vector2D(2.0, 1.0), Vector2D(0.0, 0.0))

    result = validator.validate_system_state([ball1, ball2], table)
    assert result.is_valid, f"Valid system failed validation: {result.errors}"
    print("✓ Valid system state passed")

    # Test overlapping balls
    ball2_overlap = create_test_ball(
        "ball2", Vector2D(1.01, 1.0), Vector2D(0.0, 0.0)
    )  # Very close to ball1

    result = validator.validate_system_state([ball1, ball2_overlap], table)
    assert not result.is_valid, "Overlapping balls should fail validation"
    assert any("ball_overlap" in error.error_type for error in result.errors)
    print("✓ Overlapping balls properly detected")

    # Test multiple cue balls
    ball2_cue = create_test_ball("ball2", Vector2D(2.0, 1.0), Vector2D(0.0, 0.0))
    ball2_cue.is_cue_ball = True  # Second cue ball

    result = validator.validate_system_state([ball1, ball2_cue], table)
    assert not result.is_valid, "Multiple cue balls should fail validation"
    assert any("cue_ball_count" in error.error_type for error in result.errors)
    print("✓ Multiple cue balls properly detected")

    # Test no cue ball
    ball1.is_cue_ball = False

    result = validator.validate_system_state([ball1, ball2], table)
    assert not result.is_valid, "No cue ball should fail validation"
    assert any("cue_ball_count" in error.error_type for error in result.errors)
    print("✓ No cue ball properly detected")

    print("System state validation tests passed!\n")


def test_validation_summary():
    """Test validation summary functionality."""
    print("Testing validation summary...")
    validator = PhysicsValidator()

    # Create some validation results
    results = []

    # Valid result
    valid_ball = create_test_ball()
    valid_result = validator.validate_ball_state(valid_ball)
    results.append(valid_result)

    # Invalid result
    invalid_ball = create_test_ball()
    invalid_ball.radius = -0.1  # Modify after creation
    invalid_result = validator.validate_ball_state(invalid_ball)
    results.append(invalid_result)

    # Get summary
    summary = validator.get_validation_summary(results)

    assert summary["total_validations"] == 2
    assert summary["passed_validations"] == 1
    assert summary["failed_validations"] == 1
    assert summary["total_errors"] > 0
    assert not summary["overall_valid"]
    assert "invalid_radius" in summary["error_types"]

    print("✓ Validation summary properly generated")
    print(f"  - Total validations: {summary['total_validations']}")
    print(f"  - Passed: {summary['passed_validations']}")
    print(f"  - Failed: {summary['failed_validations']}")
    print(f"  - Error types: {list(summary['error_types'].keys())}")
    print(f"  - Average confidence: {summary['average_confidence']:.2f}")

    print("Validation summary tests passed!\n")


def test_performance():
    """Test validation performance with larger datasets."""
    print("Testing validation performance...")
    validator = PhysicsValidator()
    table = create_test_table()

    import time

    # Test with many balls
    balls = []
    for i in range(50):
        x = 0.5 + (i % 10) * 0.2
        y = 0.5 + (i // 10) * 0.2
        ball = create_test_ball(f"ball_{i}", Vector2D(x, y))
        if i == 0:
            ball.is_cue_ball = True
        balls.append(ball)

    start_time = time.time()
    result = validator.validate_system_state(balls, table)
    end_time = time.time()

    validation_time = end_time - start_time
    print(f"✓ Validated {len(balls)} balls in {validation_time:.4f} seconds")
    print(f"  - Average time per ball: {validation_time/len(balls)*1000:.2f} ms")
    print(f"  - Validation successful: {result.is_valid}")

    # Performance should be reasonable (less than 100ms for 50 balls)
    assert validation_time < 0.1, f"Validation too slow: {validation_time:.4f}s"

    print("Performance tests passed!\n")


def run_all_tests():
    """Run all physics validation tests."""
    print("=" * 60)
    print("PHYSICS VALIDATION SYSTEM COMPREHENSIVE TESTS")
    print("=" * 60)
    print()

    try:
        test_ball_state_validation()
        test_trajectory_validation()
        test_collision_validation()
        test_force_validation()
        test_energy_conservation()
        test_momentum_conservation()
        test_system_state_validation()
        test_validation_summary()
        test_performance()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("The physics validation system is working correctly.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
