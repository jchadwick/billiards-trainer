# Resolution Configuration Utilities

`backend/core/resolution_config.py` is a small lookup and bounds-validation utility. It does not provide table-size registries, `CoordinateSpace`, a `ResolutionConfig` class, or coordinate conversion. For conversion to and from canonical image coordinates, use `Vector2D` and `ResolutionConverter` as described in [`docs/COORDINATE_SYSTEMS.md`](../../docs/COORDINATE_SYSTEMS.md).

## Current API

```python
from backend.core.resolution_config import (
    StandardResolution,
    get_standard_resolution,
    validate_point_in_bounds,
)

frame_size = get_standard_resolution("1080p")
assert frame_size == (1920, 1080)
assert StandardResolution.UHD_4K.value == (3840, 2160)
assert validate_point_in_bounds(960, 540, *frame_size)
```

`get_standard_resolution` returns a `(width, height)` tuple or `None` for an unknown name. `validate_point_in_bounds` checks a point against a rectangle from `(0, 0)` to `(width, height)`, with an optional pixel margin. Neither helper attaches coordinate metadata or converts units.

## Create a scale-aware point

```python
from backend.core.coordinates import Vector2D
from backend.core.resolution_converter import ResolutionConverter

point = Vector2D.from_resolution(960, 540, (1920, 1080))
assert point.to_4k_canonical().x == 1920
assert ResolutionConverter.calculate_scale_to_4k((640, 360)) == (6.0, 6.0)
```

`StandardResolution` includes VGA, SVGA, 720p, 900p, 1080p, QHD, UHD 4K, DCI 4K, and UHD 8K. The canonical image space is UHD 4K (3840 × 2160); DCI 4K is a distinct source resolution and must be scaled independently.
