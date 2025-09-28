# Physics Validation System

This module provides comprehensive physics validation to ensure system reliability and catch invalid physics states in the billiards trainer system.

## Overview

The physics validation system validates physics calculations, conservation laws, and system constraints to ensure reliable physics simulation and catch invalid states. It provides detailed error reporting with suggested fixes and confidence metrics.

## Core Components

### PhysicsValidator

The main validation class that provides comprehensive physics validation functionality.

**Features:**
- Ball state validation (position, velocity, mass, radius constraints)
- Trajectory validation (continuity, time sequences, physical limits)
- Collision validation (type consistency, position accuracy, force limits)
- Force validation (magnitude limits, acceleration constraints, NaN/infinity checks)
- Conservation law validation (energy and momentum conservation)
- System state validation (ball overlaps, cue ball count, total energy)
- Performance-conscious validation (configurable tolerances)

### ValidationError

Detailed error information with:
- Error type classification
- Severity levels (warning, error, critical)
- Human-readable messages
- Detailed error data
- Suggested fixes

### ValidationResult

Comprehensive validation results containing:
- Validity status
- List of errors and warnings
- Confidence score (0.0 to 1.0)

## Configuration

The validator accepts configuration parameters:

```python
config = {
    "energy_tolerance": 0.05,      # 5% energy conservation tolerance
    "momentum_tolerance": 0.01,    # 1% momentum conservation tolerance
    "velocity_tolerance": 0.001,   # m/s velocity precision
    "position_tolerance": 0.001,   # m position precision
    "max_velocity": 20.0,          # m/s maximum allowed velocity
    "max_acceleration": 100.0,     # m/s² maximum acceleration
    "max_spin": 100.0,             # rad/s maximum spin rate
    "max_force": 50.0,             # N maximum force magnitude
    "detailed_validation": True,    # Enable detailed checks
    "conservation_checks": True,    # Enable conservation law validation
}

validator = PhysicsValidator(config)
```

## Usage Examples

### Ball State Validation

```python
from core.validation.physics import PhysicsValidator
from core.models import BallState, Vector2D, TableState

validator = PhysicsValidator()
table = TableState.standard_9ft_table()

ball = BallState(
    id="cue_ball",
    position=Vector2D(1.0, 1.0),
    velocity=Vector2D(2.0, 0.0),
    is_cue_ball=True
)

result = validator.validate_ball_state(ball, table)
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.message}")
        if error.suggested_fix:
            print(f"Fix: {error.suggested_fix}")
```

### Trajectory Validation

```python
from core.physics.engine import TrajectoryPoint

trajectory = [
    TrajectoryPoint(time=0.0, position=Vector2D(1.0, 1.0), velocity=Vector2D(2.0, 0.0)),
    TrajectoryPoint(time=0.1, position=Vector2D(1.2, 1.0), velocity=Vector2D(1.8, 0.0)),
    TrajectoryPoint(time=0.2, position=Vector2D(1.4, 1.0), velocity=Vector2D(1.6, 0.0)),
]

result = validator.validate_trajectory(trajectory)
print(f"Trajectory valid: {result.is_valid}")
print(f"Confidence: {result.confidence:.2f}")
```

### Conservation Law Validation

```python
# Before collision
ball1_before = BallState(id="1", position=Vector2D(1,1), velocity=Vector2D(2,0))
ball2_before = BallState(id="2", position=Vector2D(2,1), velocity=Vector2D(0,0))

# After collision
ball1_after = BallState(id="1", position=Vector2D(1,1), velocity=Vector2D(0,0))
ball2_after = BallState(id="2", position=Vector2D(2,1), velocity=Vector2D(1.9,0))

# Validate energy conservation
energy_result = validator.validate_energy_conservation(
    [ball1_before, ball2_before],
    [ball1_after, ball2_after]
)

# Validate momentum conservation
momentum_result = validator.validate_momentum_conservation(
    [ball1_before, ball2_before],
    [ball1_after, ball2_after]
)
```

### System State Validation

```python
balls = [ball1, ball2, ball3]  # List of all balls
table = TableState.standard_9ft_table()

result = validator.validate_system_state(balls, table)
if not result.is_valid:
    print("System validation failed:")
    for error in result.errors:
        print(f"  - {error.error_type}: {error.message}")
```

## Integration with Core Module

The physics validator is automatically integrated into the Core Module when physics validation is enabled:

```python
from core import CoreModule, CoreModuleConfig

config = CoreModuleConfig(
    physics_validation_enabled=True,
    detailed_validation=True,
    conservation_checks=True,
    energy_tolerance=0.05,
    momentum_tolerance=0.01
)

core = CoreModule(config)

# Validation is now automatically performed on state updates
# and can be explicitly triggered:

# Validate current state
validation_result = await core.validate_state()

# Validate trajectory
trajectory_result = await core.validate_trajectory("cue_ball", Vector2D(2,0))

# Validate collision
collision_result = await core.validate_collision(collision, "ball1", "ball2")

# Get validation system status
summary = core.get_validation_summary()
```

## Error Types

The validation system identifies various types of physics violations:

### Ball State Errors
- `invalid_radius`: Ball radius is negative or zero
- `invalid_mass`: Ball mass is negative or zero
- `velocity_limit`: Ball velocity exceeds physical limits
- `spin_limit`: Ball spin rate is unreasonably high
- `position_out_of_bounds`: Ball position is outside table bounds
- `kinetic_energy`: Ball kinetic energy exceeds limits

### Trajectory Errors
- `empty_trajectory`: Trajectory contains no points
- `time_sequence`: Time values are not monotonically increasing
- `position_discontinuity`: Large jumps in position between points
- `velocity_limit`: Velocity exceeds limits at trajectory points

### Collision Errors
- `collision_type_mismatch`: Collision type doesn't match involved objects
- `collision_position`: Collision position inconsistent with object positions
- `post_collision_velocity`: Resulting velocities exceed limits
- `negative_collision_time`: Collision time is negative (past event)
- `impact_force`: Impact force is unusually high

### Force Errors
- `force_magnitude`: Force magnitude is unusually high
- `acceleration_limit`: Resulting acceleration exceeds physical limits
- `invalid_mass`: Mass value is invalid for force calculations
- `invalid_force_values`: Force contains NaN or infinite values

### Conservation Errors
- `energy_conservation`: Energy conservation violation detected
- `energy_increase`: Energy increased without external input
- `momentum_conservation`: Momentum conservation violation detected
- `momentum_creation`: Momentum created from zero initial momentum
- `zero_initial_energy`: Cannot validate with zero initial energy

### System Errors
- `ball_overlap`: Balls are overlapping
- `cue_ball_count`: Wrong number of cue balls (should be exactly 1)
- `system_energy`: Total system energy is unreasonably high

## Performance

The validation system is designed for real-time use:

- **Ball state validation**: ~0.01ms per ball
- **Trajectory validation**: ~0.1ms for typical trajectory
- **System validation**: ~0.5ms for 15 balls
- **Conservation checks**: ~0.05ms per check

Performance can be tuned by:
- Disabling detailed validation for production
- Adjusting tolerances for specific use cases
- Disabling conservation checks for non-critical scenarios

## Testing

Comprehensive test suite validates all aspects of the physics validation system:

```bash
cd backend
python core/validation/test_physics_validator.py
```

The test suite covers:
- Valid and invalid ball states
- Trajectory continuity and constraints
- Collision validation scenarios
- Force limit checking
- Conservation law validation
- System state validation
- Performance benchmarks
- Error reporting accuracy

## Error Handling

The validation system provides multiple levels of error handling:

1. **Critical Errors**: System cannot continue (NaN values, infinite forces)
2. **Errors**: Physics violations that should be corrected (overlapping balls, conservation violations)
3. **Warnings**: Suspicious but potentially valid states (high forces, negative collision times)

Each error includes:
- Clear description of the problem
- Detailed data about the violation
- Suggested fix when applicable
- Confidence metric for the validation

## Integration Points

The physics validator integrates with:

- **PhysicsEngine**: Validates trajectories and collision calculations
- **GameStateManager**: Validates state updates and transitions
- **CoreModule**: Provides validation API for external clients
- **CollisionDetector/Resolver**: Validates collision physics
- **ForceCalculator**: Validates force calculations

This ensures comprehensive validation coverage across the entire physics simulation pipeline.
