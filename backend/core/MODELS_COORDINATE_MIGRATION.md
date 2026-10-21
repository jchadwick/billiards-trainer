# Historical BallState Coordinate Migration Note (Superseded)

> **Historical migration record, not current usage guidance.** The earlier text described meter-based core storage, coordinate-space enum metadata, factory methods, and compatibility behavior that do not match the current model API. Follow [`docs/COORDINATE_SYSTEMS.md`](../../docs/COORDINATE_SYSTEMS.md) and [`COORDINATE_CONVERTER_EXAMPLES.md`](COORDINATE_CONVERTER_EXAMPLES.md).

## Current model contract

- `backend.core.coordinates.Vector2D` represents image-plane geometry with explicit `(scale_x, scale_y)` metadata. Scale multiplies supplied x/y into the 3840 × 2160 canonical plane.
- `Vector2D.from_4k(x, y)` creates canonical image coordinates. `Vector2D.from_resolution(x, y, (width, height))` creates a source-resolution point with its canonical scale.
- `BallState.from_4k(...)` and `BallState.from_resolution(...)` create ball positions. `BallState` normalizes its position and pixel radius to canonical image pixels.
- `TableState` stores pixel dimensions, pocket geometry, and optional playing-area corners in canonical image space. Optional real-world table dimensions and cushion height remain explicitly in meters.
- Physical world points are separate from core image vectors. Calibration relates the two; do not mix units in arithmetic.

## Minimal current example

```python
from backend.core.coordinates import Vector2D
from backend.core.models import BallState

ball = BallState(
    id="cue",
    position=Vector2D.from_resolution(320, 180, (640, 360)),
    velocity=Vector2D.from_4k(0, 0),
)
assert ball.position.scale == (1.0, 1.0)
```

For complete conversion, calibration, serialization, radius, dimensions, and bounds examples, see [`COORDINATE_CONVERTER_EXAMPLES.md`](COORDINATE_CONVERTER_EXAMPLES.md).
