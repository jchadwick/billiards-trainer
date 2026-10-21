# Physics Validation System

`PhysicsValidator` checks ball states, trajectories, collisions, forces, conservation laws, and system state. Its coordinate units follow the core physics model: spatial positions and table geometry are canonical 3840 × 2160 pixels, velocities are pixels per second, accelerations are pixels per second squared, and time remains seconds. Mass and force remain SI quantities. This is separate from physical table-plane positions in meters.

Read [`docs/COORDINATE_SYSTEMS.md`](../../../docs/COORDINATE_SYSTEMS.md) for the repository-wide rules on scale, bounds, radii, extents, and calibration.

## Configuration

`position_tolerance` is in canonical pixels and `velocity_tolerance` is in pixels per second. Explicit `max_velocity` and `max_acceleration` values use those corresponding pixel units. The validator's defaults convert reference physical limits using a documented pixels-per-meter value; use calibrated dimensions when physical-to-image conversion is needed.

```python
from backend.core.validation.physics import PhysicsValidator

validator = PhysicsValidator(
    {
        "position_tolerance": 1.0,  # canonical pixels
        "velocity_tolerance": 1.0,  # canonical pixels / second
        "energy_tolerance": 0.05,  # relative fraction
        "momentum_tolerance": 0.01,  # relative fraction
        "max_spin": 100.0,  # radians / second
        "max_force": 50.0,  # newtons
        "detailed_validation": True,
        "conservation_checks": True,
    }
)
```

## Validate ball and table state

```python
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, TableState
from backend.core.validation.physics import PhysicsValidator

# Calibration corners are pixels in the declared 4K source frame.
table = TableState.from_calibration(
    table_corners_pixel=[(320, 280), (3520, 280), (3520, 1880), (320, 1880)],
    camera_resolution=(3840, 2160),
    table_dimensions_real=(2.54, 1.27),  # meters, stored separately
)
ball = BallState(
    id="cue",
    position=Vector2D.from_4k(1920, 1080),
    velocity=Vector2D.from_4k(0, 0),  # pixel-space velocity
    is_cue_ball=True,
)

result = PhysicsValidator().validate_ball_state(ball, table)
if not result.is_valid:
    for error in result.errors:
        print(error.error_type, error.message)
```

`TableState.from_calibration` scales the supplied image corners into canonical pixels and stores optional real dimensions as meters. Ball coordinates and table pixel geometry must be in the same canonical image plane before validation.

## Validate a trajectory

```python
from backend.core.coordinates import Vector2D
from backend.core.physics.engine import TrajectoryPoint
from backend.core.validation.physics import PhysicsValidator

trajectory = [
    TrajectoryPoint(0.0, Vector2D.from_4k(1000, 900), Vector2D.from_4k(10, 0)),
    TrajectoryPoint(0.1, Vector2D.from_4k(1001, 900), Vector2D.from_4k(10, 0)),
    TrajectoryPoint(0.2, Vector2D.from_4k(1002, 900), Vector2D.from_4k(10, 0)),
]
result = PhysicsValidator().validate_trajectory(trajectory)
print(result.is_valid, result.confidence)
```

## Validate conservation across a collision

```python
from backend.core.coordinates import Vector2D
from backend.core.models import BallState
from backend.core.validation.physics import PhysicsValidator

before = [
    BallState("cue", Vector2D.from_4k(1000, 900), Vector2D.from_4k(100, 0), is_cue_ball=True),
    BallState("object", Vector2D.from_4k(1100, 900), Vector2D.from_4k(0, 0)),
]
after = [
    BallState("cue", Vector2D.from_4k(1000, 900), Vector2D.from_4k(0, 0), is_cue_ball=True),
    BallState("object", Vector2D.from_4k(1100, 900), Vector2D.from_4k(95, 0)),
]
validator = PhysicsValidator()
energy = validator.validate_energy_conservation(before, after)
momentum = validator.validate_momentum_conservation(before, after)
```

## Validate all balls on a table

```python
result = PhysicsValidator().validate_system_state([ball], table)
if not result.is_valid:
    for error in result.errors:
        print(f"{error.error_type}: {error.message}")
```

## Core module integration

`CoreModule` accepts `CoreModuleConfig(physics_validation_enabled=True)`. Its `validate_state`, `validate_trajectory`, and `validate_collision` methods are asynchronous and operate on the current state. Use `Vector2D.from_4k` for canonical pixel velocities passed to `validate_trajectory`.

## Result types and errors

`ValidationResult` contains `is_valid`, `errors`, `warnings`, and a confidence score. Each `ValidationError` contains an error type, severity, message, details, and an optional suggested fix. Detected conditions include invalid ball dimensions or motion, trajectory discontinuities, collision inconsistencies, force limits, conservation violations, overlaps, and invalid cue-ball counts.

Performance varies with the host, state size, and enabled checks. Use measured benchmarks for performance decisions; this guide does not promise fixed per-call latency.
