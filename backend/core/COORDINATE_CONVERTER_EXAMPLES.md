# CoordinateConverter Usage Examples

This document provides practical examples for using the `CoordinateConverter` utility to transform coordinates between different coordinate spaces in the billiards trainer system.

## Overview

The `CoordinateConverter` provides a centralized, type-safe way to convert between:
- **Camera Pixels**: Native camera resolution coordinates (e.g., 1920x1080)
- **World Meters**: Real-world metric coordinates (canonical storage format)
- **Table Pixels**: Table-specific pixel coordinates
- **Normalized Coordinates**: Resolution-independent [0,1] coordinates
- **Detection Pixels**: Model-specific resolutions (e.g., YOLO input size)

**Key Principle**: All conversions go through **World Meters** as the canonical intermediate format, ensuring consistency and preventing error accumulation.

## Quick Start

### Basic Setup

```python
from backend.core.coordinate_converter import (
    CoordinateConverter,
    CoordinateSpace,
    Resolution,
)
from backend.core.models import Vector2D

# Create converter with calibration data
converter = CoordinateConverter(
    table_width_meters=2.54,      # 9ft table width
    table_height_meters=1.27,     # 4.5ft table height
    pixels_per_meter=754.0,       # From calibration
    camera_resolution=Resolution(1920, 1080),
)
```

### Using Default 9ft Table

```python
# Create converter with defaults (standard 9ft table)
converter = CoordinateConverter()
```

## Common Use Cases

### 1. Converting Camera Detection to World Coordinates

When you detect a ball in camera pixels and need to convert it to world meters for physics calculations:

```python
# Ball detected at pixel position
detected_position = Vector2D(960, 540)

# Convert to world meters (canonical format)
world_position = converter.camera_pixels_to_world_meters(
    detected_position,
    Resolution(1920, 1080)
)

# Now use in physics calculations
print(f"Ball at ({world_position.x:.2f}m, {world_position.y:.2f}m)")
```

### 2. Converting World Coordinates for Display

When you need to render world coordinates on a canvas or camera view:

```python
# Ball position from physics engine (in meters)
ball_world = Vector2D(1.27, 0.635)  # Table center

# Convert to camera pixels for display
display_position = converter.world_meters_to_camera_pixels(
    ball_world,
    Resolution(1920, 1080)
)

# Render at pixel position
print(f"Render at ({display_position.x:.0f}px, {display_position.y:.0f}px)")
```

### 3. Batch Converting Multiple Positions

When converting many positions at once (e.g., trajectory points):

```python
# Multiple ball positions in camera pixels
ball_positions = [
    Vector2D(100, 100),
    Vector2D(500, 500),
    Vector2D(900, 800),
]

# Batch convert to world meters (more efficient)
world_positions = converter.camera_pixels_to_world_meters_batch(
    ball_positions,
    Resolution(1920, 1080)
)

# Use in physics
for i, world_pos in enumerate(world_positions):
    print(f"Ball {i}: ({world_pos.x:.2f}m, {world_pos.y:.2f}m)")
```

### 4. Working with Different Camera Resolutions

When dealing with different resolutions (e.g., YOLO detection at 640x640):

```python
# Ball detected in YOLO's 640x640 input
yolo_position = Vector2D(320, 320)  # Center of YOLO frame

# Convert from YOLO resolution to world meters
world_position = converter.camera_pixels_to_world_meters(
    yolo_position,
    Resolution(640, 640)  # YOLO's input resolution
)

# Convert to native camera resolution for display
camera_position = converter.world_meters_to_camera_pixels(
    world_position,
    Resolution(1920, 1080)  # Native camera resolution
)

print(f"YOLO (320, 320) -> Camera ({camera_position.x:.0f}, {camera_position.y:.0f})")
```

### 5. Generic Conversion Between Any Spaces

When you need maximum flexibility:

```python
# Convert from detection pixels to world meters
position = Vector2D(320, 320)
world_pos = converter.convert(
    position,
    from_space=CoordinateSpace.DETECTION_PIXELS,
    to_space=CoordinateSpace.WORLD_METERS,
    from_resolution=Resolution(640, 640),
)

# Convert from world meters to normalized table coordinates
normalized = converter.convert(
    world_pos,
    from_space=CoordinateSpace.WORLD_METERS,
    to_space=CoordinateSpace.TABLE_NORMALIZED,
)

print(f"Normalized table position: ({normalized.x:.2f}, {normalized.y:.2f})")
```

### 6. Working with Normalized Coordinates

When you need resolution-independent positioning:

```python
# Define position as fraction of screen (center)
normalized = Vector2D(0.5, 0.5)

# Convert to any pixel resolution
hd_pixels = converter.normalized_to_camera_pixels(
    normalized, Resolution(1920, 1080)
)
sd_pixels = converter.normalized_to_camera_pixels(
    normalized, Resolution(640, 480)
)

print(f"HD center: ({hd_pixels.x:.0f}, {hd_pixels.y:.0f})")
print(f"SD center: ({sd_pixels.x:.0f}, {sd_pixels.y:.0f})")
```

### 7. Table Coordinate Conversion

When working with table-specific coordinate systems:

```python
# Position in table pixel space (640x360)
table_position = Vector2D(320, 180)  # Center of table

# Convert to world meters
world_pos = converter.table_pixels_to_world_meters(
    table_position,
    Resolution(640, 360)
)

print(f"Table center in world: ({world_pos.x:.2f}m, {world_pos.y:.2f}m)")
# Should be approximately (1.27m, 0.635m) - half of 2.54m x 1.27m
```

### 8. Validating Conversions

When debugging or testing coordinate transformations:

```python
# Validate that conversion is reversible
original = Vector2D(960, 540)

is_valid = converter.validate_conversion(
    original,
    CoordinateSpace.CAMERA_PIXELS,
    CoordinateSpace.WORLD_METERS,
    tolerance=1e-6,
    from_resolution=Resolution(1920, 1080),
)

if not is_valid:
    print("⚠️  Warning: Conversion is not reversible!")
else:
    print("✓ Conversion validated successfully")
```

## Advanced Use Cases

### 9. Perspective Correction

When dealing with camera perspective distortion:

```python
import numpy as np
from backend.core.coordinate_converter import PerspectiveTransform

# Define homography matrix from calibration
homography = np.array([
    [1.1, 0.05, -10],
    [0.02, 1.05, -5],
    [0.0001, 0.0002, 1.0],
], dtype=np.float64)

transform = PerspectiveTransform(matrix=homography)

# Create converter with perspective correction
converter = CoordinateConverter(
    perspective_transform=transform,
)

# Conversions will automatically apply perspective correction
corrected_world = converter.camera_pixels_to_world_meters(
    Vector2D(960, 540),
    Resolution(1920, 1080),
)
```

### 10. Integration with Vision Detection

Typical workflow when processing camera frames:

```python
# In your vision detection code
def process_detection_result(detection_result):
    """Convert vision detections to world coordinates."""
    converter = get_global_converter()

    # Extract ball positions from detection
    camera_positions = [
        Vector2D(ball.position[0], ball.position[1])
        for ball in detection_result.balls
    ]

    # Batch convert to world coordinates
    world_positions = converter.camera_pixels_to_world_meters_batch(
        camera_positions,
        detection_result.frame_size,
    )

    # Create BallState objects in world coordinates
    ball_states = [
        BallState(
            id=f"ball_{i}",
            position=world_pos,
            # ... other properties
        )
        for i, world_pos in enumerate(world_positions)
    ]

    return ball_states
```

### 11. Getting Calibration Information

When you need to inspect or serialize calibration parameters:

```python
# Get calibration info as dictionary
info = converter.get_calibration_info()

print(f"Table: {info['table_width_meters']}m x {info['table_height_meters']}m")
print(f"Pixels per meter: {info['pixels_per_meter']}")
print(f"Camera: {info['camera_resolution']['width']}x{info['camera_resolution']['height']}")
print(f"Has perspective: {info['has_perspective_transform']}")

# Serialize for storage
import json
calibration_json = json.dumps(info, indent=2)
```

## Best Practices

### 1. Always Use World Meters for Storage

```python
# ✅ CORRECT: Store in world meters
ball_state = BallState(
    id="cue",
    position=world_position,  # In meters!
)

# ❌ WRONG: Don't store in pixels
ball_state = BallState(
    id="cue",
    position=pixel_position,  # Resolution-dependent!
)
```

### 2. Convert at Boundaries Only

```python
# ✅ CORRECT: Convert once at input/output boundaries
camera_pos = detect_ball()
world_pos = converter.camera_pixels_to_world_meters(camera_pos, cam_res)
trajectory = physics.calculate(world_pos)  # Work in meters
display_pos = converter.world_meters_to_camera_pixels(trajectory.final, cam_res)

# ❌ WRONG: Don't convert back and forth
camera_pos = detect_ball()
world_pos = converter.camera_pixels_to_world_meters(camera_pos, cam_res)
camera_again = converter.world_meters_to_camera_pixels(world_pos, cam_res)  # Why?
```

### 3. Use Batch Conversion When Possible

```python
# ✅ CORRECT: Batch convert for better performance
positions = [detect_ball(frame) for frame in frames]
world_positions = converter.camera_pixels_to_world_meters_batch(positions, cam_res)

# ❌ SUBOPTIMAL: Individual conversions
world_positions = [
    converter.camera_pixels_to_world_meters(pos, cam_res)
    for pos in positions
]
```

### 4. Validate in Tests

```python
# Always validate conversions in your tests
def test_my_coordinate_conversion():
    converter = CoordinateConverter()
    original = Vector2D(500, 500)

    # Convert and validate
    assert converter.validate_conversion(
        original,
        CoordinateSpace.CAMERA_PIXELS,
        CoordinateSpace.WORLD_METERS,
        from_resolution=Resolution(1920, 1080),
    )
```

## Common Pitfalls

### 1. Forgetting Resolution

```python
# ❌ WRONG: Missing resolution can cause incorrect scaling
world_pos = converter.camera_pixels_to_world_meters(camera_pos)  # Uses default!

# ✅ CORRECT: Always specify resolution
world_pos = converter.camera_pixels_to_world_meters(
    camera_pos,
    Resolution(1920, 1080)
)
```

### 2. Mixing Coordinate Spaces

```python
# ❌ WRONG: Mixing pixels and meters without conversion
pixel_pos = Vector2D(100, 100)  # pixels
meter_pos = Vector2D(1.0, 0.5)  # meters
distance = pixel_pos.distance_to(meter_pos)  # NONSENSE!

# ✅ CORRECT: Convert to same space first
meter_pos1 = converter.camera_pixels_to_world_meters(pixel_pos, cam_res)
meter_pos2 = Vector2D(1.0, 0.5)
distance = meter_pos1.distance_to(meter_pos2)  # Correct!
```

### 3. Assuming Same Resolution

```python
# ❌ WRONG: Assuming all pixels are equal
yolo_pos = Vector2D(320, 320)  # From 640x640 YOLO
camera_pos = Vector2D(320, 320)  # From 1920x1080 camera
# These are NOT the same location!

# ✅ CORRECT: Account for resolution
yolo_world = converter.camera_pixels_to_world_meters(
    yolo_pos, Resolution(640, 640)
)
camera_world = converter.camera_pixels_to_world_meters(
    camera_pos, Resolution(1920, 1080)
)
```

## Performance Considerations

- **Batch Operations**: Use `*_batch()` methods when converting multiple points
- **Avoid Redundant Conversions**: Cache converted values when possible
- **Perspective Transforms**: Add ~200ns overhead per conversion
- **Validation**: Only use `validate_conversion()` in tests, not production

## Further Reading

- See `/thoughts/resolution_standardization_plan.md` for complete design documentation
- See `backend/tests/unit/test_coordinate_converter.py` for comprehensive test examples
- See `backend/core/coordinate_converter.py` source code for implementation details
