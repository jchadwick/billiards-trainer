# BallState Coordinate Space Migration Guide

## Overview

The `core/models.py` module has been updated to support the enhanced Vector2D from `coordinates.py` that includes coordinate space metadata. This update maintains **full backward compatibility** while enabling new code to use proper coordinate space tracking.

## What Changed

### 1. Enhanced Vector2D Support

**Before:**
```python
ball = BallState(
    id="ball_1",
    position=Vector2D(1.27, 0.635),
    number=1
)
# Position has no coordinate space metadata
```

**After (Recommended):**
```python
ball = BallState.create(
    id="ball_1",
    x=1.27, y=0.635,
    number=1
)
# Position is automatically tagged with WORLD_METERS coordinate space
assert ball.position.coordinate_space == CoordinateSpace.WORLD_METERS
```

### 2. New Factory Methods

Four factory methods have been added to BallState:

#### `BallState.create()` - Create in World Meters (Recommended)
```python
ball = BallState.create("ball_1", x=1.27, y=0.635, number=1)
# Position is in WORLD_METERS coordinate space
```

#### `BallState.from_camera_pixels()` - Create from Camera Coordinates
```python
from core.coordinates import Resolution

camera_res = Resolution(1920, 1080)
ball = BallState.from_camera_pixels(
    "ball_1",
    x=960, y=540,
    camera_resolution=camera_res,
    converter=coordinate_converter,  # Optional: auto-convert to world meters
    number=1
)
# Position is in CAMERA_PIXELS (or WORLD_METERS if converter provided)
```

#### `BallState.from_table_pixels()` - Create from Table Coordinates
```python
table_res = Resolution(640, 360)
ball = BallState.from_table_pixels(
    "ball_1",
    x=320, y=180,
    table_resolution=table_res,
    converter=coordinate_converter,  # Optional: auto-convert to world meters
    number=1
)
```

#### `BallState.from_normalized()` - Create from Normalized Coordinates
```python
ball = BallState.from_normalized(
    "ball_1",
    x=0.5, y=0.5,  # Normalized [0,1]
    converter=coordinate_converter,  # Optional: auto-convert to world meters
    number=1
)
```

### 3. Coordinate Conversion Helpers

#### `get_position_in_space()` - Convert to Any Coordinate Space
```python
camera_res = Resolution(1920, 1080)
camera_pos = ball.get_position_in_space(
    CoordinateSpace.CAMERA_PIXELS,
    converter,
    camera_res
)
print(f"Ball at ({camera_pos.x}, {camera_pos.y}) in camera pixels")
```

#### `to_camera_pixels()` - Convert to Camera Pixels
```python
camera_res = Resolution(1920, 1080)
camera_pos = ball.to_camera_pixels(converter, camera_res)
```

#### `to_table_pixels()` - Convert to Table Pixels
```python
table_res = Resolution(640, 360)
table_pos = ball.to_table_pixels(converter, table_res)
```

#### `has_coordinate_metadata()` - Check for Metadata
```python
if ball.has_coordinate_metadata():
    print(f"Ball is in {ball.position.coordinate_space}")
else:
    print("Ball uses legacy Vector2D without metadata")
```

## Backward Compatibility

### Legacy Code Continues to Work

All existing code using the old Vector2D class continues to work without changes:

```python
# This still works exactly as before
legacy_ball = BallState(
    id="ball_1",
    position=Vector2D(1.27, 0.635),
    number=1
)
```

### Serialization Compatibility

**Legacy Format (still supported):**
```json
{
  "id": "ball_1",
  "position": {"x": 1.27, "y": 0.635},
  "number": 1
}
```

**Enhanced Format (with metadata):**
```json
{
  "id": "ball_1",
  "position": {
    "x": 1.27,
    "y": 0.635,
    "coordinate_space": "world_meters"
  },
  "number": 1
}
```

### Automatic Detection

The `from_dict()` method automatically detects whether the serialized data contains coordinate metadata and creates the appropriate Vector2D type:

```python
# Automatically uses enhanced Vector2D if metadata present
ball1 = BallState.from_dict({
    "id": "ball_1",
    "position": {
        "x": 1.27,
        "y": 0.635,
        "coordinate_space": "world_meters"
    },
    "number": 1
})
assert ball1.has_coordinate_metadata()

# Automatically uses legacy Vector2D if no metadata
ball2 = BallState.from_dict({
    "id": "ball_2",
    "position": {"x": 1.0, "y": 2.0},
    "number": 2
})
assert not ball2.has_coordinate_metadata()
```

## Migration Path

### Phase 1: Awareness (Current)
- Legacy Vector2D class remains with deprecation notice in docstring
- New code should use factory methods (BallState.create(), etc.)
- Existing code continues to work unchanged

### Phase 2: Gradual Migration
- Update ball creation code to use factory methods
- Add coordinate converters to enable automatic conversion
- Update vision/detection code to use from_camera_pixels()

### Phase 3: Full Adoption
- All new BallState instances use enhanced Vector2D
- Coordinate conversions happen automatically via converters
- Legacy Vector2D only exists in old serialized data

## Examples

### Example 1: Vision Detection Integration
```python
from core.coordinates import Resolution, CoordinateSpace
from core.coordinate_converter import get_coordinate_converter
from core.models import BallState

# Get coordinate converter (has calibration data)
converter = get_coordinate_converter()

# Vision system detects ball in camera pixels
camera_res = Resolution(1920, 1080)
detected_x, detected_y = 960, 540

# Create ball with automatic conversion to world meters
ball = BallState.from_camera_pixels(
    id="ball_1",
    x=detected_x,
    y=detected_y,
    camera_resolution=camera_res,
    converter=converter,  # Automatically converts to WORLD_METERS
    number=1
)

# Ball position is now in world meters
assert ball.position.coordinate_space == CoordinateSpace.WORLD_METERS
print(f"Ball at ({ball.position.x:.3f}m, {ball.position.y:.3f}m)")
```

### Example 2: WebSocket/API Response
```python
# Backend stores balls in world meters
ball = BallState.create("ball_1", x=1.27, y=0.635, number=1)

# Convert to camera pixels for frontend display
camera_res = Resolution(1920, 1080)
camera_pos = ball.to_camera_pixels(converter, camera_res)

# Send to client
response = {
    "ball_id": ball.id,
    "position_camera": {
        "x": camera_pos.x,
        "y": camera_pos.y
    },
    "position_world": {
        "x": ball.position.x,
        "y": ball.position.y
    }
}
```

### Example 3: Multi-Resolution Support
```python
# Ball in world meters
ball = BallState.create("ball_1", x=1.27, y=0.635, number=1)

# Convert to different resolutions as needed
camera_1080p = ball.to_camera_pixels(converter, Resolution(1920, 1080))
camera_720p = ball.to_camera_pixels(converter, Resolution(1280, 720))
table_640x360 = ball.to_table_pixels(converter, Resolution(640, 360))
```

## Type Hints

The BallState now accepts `Union[Vector2D, EnhancedVector2D]` for position, velocity, and spin:

```python
@dataclass
class BallState:
    id: str
    position: Union[Vector2D, EnhancedVector2D]
    velocity: Union[Vector2D, EnhancedVector2D] = field(default_factory=Vector2D.zero)
    spin: Union[Vector2D, EnhancedVector2D] = field(default_factory=Vector2D.zero)
    # ... other fields
```

This allows both legacy and enhanced vectors to coexist.

## Testing

Run the verification test:
```python
from core.coordinates import CoordinateSpace, Resolution
from core.models import BallState

# Test factory method
ball = BallState.create("test", 1.27, 0.635, number=1)
assert ball.has_coordinate_metadata()
assert ball.position.coordinate_space == CoordinateSpace.WORLD_METERS

# Test serialization round-trip
ball_dict = ball.to_dict()
ball2 = BallState.from_dict(ball_dict)
assert ball2.has_coordinate_metadata()

# Test copy
ball3 = ball.copy()
assert ball3.has_coordinate_metadata()

# Test legacy compatibility
from core.models import Vector2D
legacy_ball = BallState(id="legacy", position=Vector2D(1.0, 2.0))
assert not legacy_ball.has_coordinate_metadata()
```

## Summary

This update provides a smooth migration path from untyped coordinate data to fully-typed coordinate spaces while maintaining 100% backward compatibility. New code should use the factory methods and conversion helpers to ensure proper coordinate space tracking throughout the system.
