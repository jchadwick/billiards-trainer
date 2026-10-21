# Current Coordinate Conversion Examples

> **Supersedes the former `CoordinateConverter` examples.** That class and its world-meter storage workflow are not the current core API. Use the shared contract in [`docs/COORDINATE_SYSTEMS.md`](../../docs/COORDINATE_SYSTEMS.md) and the APIs below.

The core image plane is 3840 × 2160 canonical pixels. Every source point carries an explicit scale to that plane; the scale multiplies x and y independently. A canonical point still declares identity scale `(1.0, 1.0)`. Core physics and game state use canonical pixel geometry. Physical meter points remain separate and map to image points only through calibration.

## Convert a camera point

```python
from backend.core.coordinates import Vector2D

# A point from a 1920 × 1080 frame. The factory records scale=(2, 2).
source_point = Vector2D.from_resolution(960, 540, (1920, 1080))
canonical_point = source_point.to_4k_canonical()
assert (canonical_point.x, canonical_point.y) == (1920, 1080)
assert canonical_point.scale == (1.0, 1.0)
```

For callers that already have scale metadata, construct the vector directly and provide it:

```python
source_point = Vector2D(320, 180, scale=(6.0, 6.0))
canonical_point = source_point.to_4k_canonical()
```

## Convert canonical points for a display frame

```python
from backend.core.coordinates import Vector2D

canonical_point = Vector2D.from_4k(1920, 1080)
hd_point = canonical_point.to_resolution((1920, 1080))
assert (hd_point.x, hd_point.y) == (960, 540)
```

The scale on `hd_point` records how its source values map into canonical pixels. Use its x/y values for the 1920 × 1080 display frame.

## Convert raw coordinates at a boundary

```python
from backend.core.resolution_converter import ResolutionConverter

canonical_xy = ResolutionConverter.scale_to_4k(320, 180, (640, 360))
frame_xy = ResolutionConverter.scale_from_4k(*canonical_xy, (1920, 1080))
assert canonical_xy == (1920.0, 1080.0)
assert frame_xy == (960.0, 540.0)
```

Use `scale_distance_to_4k` or `scale_distance_from_4k` for a circular pixel radius. Give each radius or extent its own scale metadata; do not assume a point's scale applies to an unrelated geometry field.

## Build core game state from a source frame

```python
from backend.core.models import BallState

ball = BallState.from_resolution(
    id="cue", x=320, y=180, resolution=(640, 360), radius=6
)
# BallState normalizes its position and radius to canonical pixels.
assert ball.position.scale == (1.0, 1.0)
```

For a canonical point, use `BallState.from_4k(...)` or pass a `Vector2D.from_4k(...)` position. Core table dimensions, bounds, and pocket radii each have their own scale metadata.

## Calibrate pixel geometry to physical table dimensions

```python
from backend.core.models import TableState

table = TableState.from_calibration(
    table_corners_pixel=[(50, 40), (590, 40), (590, 320), (50, 320)],
    camera_resolution=(640, 360),
    table_dimensions_real=(2.54, 1.27),  # meters: width, height
)
assert table.width > 0  # canonical pixels
assert table.real_width_meters == 2.54
```

The calibration pairing connects image pixels and physical coordinates. Do not construct a `Vector2D` in meters or use an uncalibrated pixels-per-meter constant for production geometry.

## Scale table bounds and extents independently

Core table geometry accepts `dimensions_scale`; individual circular radii use their own scale fields. `CoordinateMetadata` bounds use `table_bounds_scale`. Bounds from a 640 × 360 source use a `(6.0, 6.0)` scale when passed to that model. Scale all four edges consistently, and retain the bounds' unit and source context.

See [`models.py`](models.py), [`coordinates.py`](coordinates.py), and [`resolution_converter.py`](resolution_converter.py) for implementations. API physical-point models are defined in [`backend/api/models/common.py`](../api/models/common.py).
