# Resolution Configuration Utilities

A lightweight utility module for managing standard resolutions, table dimensions, and coordinate spaces in the billiards trainer system.

## Overview

This module provides:
- **Standard video resolutions** (HD, 4K, 8K, etc.)
- **Standard billiards table dimensions** (6ft to 12ft, snooker tables)
- **Coordinate space management** with validation and transformation
- **Scaling utilities** for converting between pixel and world coordinates

## Quick Start

```python
from backend.core.resolution_config import (
    ResolutionConfig,
    get_standard_resolution,
    get_table_dimensions,
)

# Get standard resolution
resolution = get_standard_resolution("1080p")  # (1920, 1080)

# Get table dimensions
table_dims = get_table_dimensions("9ft")  # (2.54, 1.27) meters

# Create coordinate spaces
pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
table_space = ResolutionConfig.create_table_space((2.54, 1.27))

# Scale coordinates from pixels to table
from backend.core.models import Vector2D
pixel_pos = Vector2D(960, 540)
table_pos = ResolutionConfig.scale_vector(pixel_pos, pixel_space, table_space)
```

## Standard Resolutions

### Available Resolutions

| Name | Resolution | Aspect Ratio | Common Name |
|------|-----------|--------------|-------------|
| `VGA` | 640×480 | 4:3 | VGA |
| `SVGA` | 800×600 | 4:3 | SVGA |
| `HD_720` | 1280×720 | 16:9 | 720p/HD |
| `HD_900` | 1600×900 | 16:9 | 900p |
| `HD_1080` | 1920×1080 | 16:9 | 1080p/Full HD |
| `QHD` | 2560×1440 | 16:9 | 1440p/2K |
| `UHD_4K` | 3840×2160 | 16:9 | 4K UHD |
| `DCI_4K` | 4096×2160 | ~1.9:1 | DCI 4K |
| `UHD_8K` | 7680×4320 | 16:9 | 8K UHD |

### Aliases Supported

The module supports common aliases for easier access:
- `"1080"`, `"1080p"`, `"FHD"`, `"FULL_HD"` → HD_1080
- `"720"`, `"720p"` → HD_720
- `"4K"` → UHD_4K
- `"2K"` → QHD
- `"8K"` → UHD_8K

## Standard Table Sizes

### Available Table Sizes

| Name | Dimensions (m) | Common Size |
|------|---------------|-------------|
| `SIX_FOOT` | 1.83 × 0.91 | 6-foot table |
| `SEVEN_FOOT` | 2.13 × 1.07 | 7-foot table |
| `EIGHT_FOOT` | 2.44 × 1.22 | 8-foot table |
| `NINE_FOOT` | 2.54 × 1.27 | 9-foot table (tournament) |
| `SNOOKER_SMALL` | 3.05 × 1.52 | 10-foot snooker |
| `SNOOKER_FULL` | 3.57 × 1.78 | 12-foot snooker |

### Aliases Supported

- `"6"`, `"6ft"`, `"6-foot"` → SIX_FOOT
- `"7"`, `"7ft"`, `"7-foot"` → SEVEN_FOOT
- `"8"`, `"8ft"`, `"8-foot"` → EIGHT_FOOT
- `"9"`, `"9ft"`, `"9-foot"` → NINE_FOOT

## Core Classes

### StandardResolution

Enum of standard video resolutions.

```python
from backend.core.resolution_config import StandardResolution

hd = StandardResolution.HD_1080
print(hd.width)         # 1920
print(hd.height)        # 1080
print(hd.aspect_ratio)  # ~1.778
print(hd.total_pixels)  # 2,073,600
```

### TableSize

Enum of standard billiards table sizes.

```python
from backend.core.resolution_config import TableSize

table = TableSize.NINE_FOOT
print(table.width)        # 2.54 (meters)
print(table.height)       # 1.27 (meters)
print(table.aspect_ratio) # ~2.0
print(table.area)         # 3.23 (square meters)
print(table.width_feet)   # 9
```

### CoordinateSpace

Represents a rectangular coordinate space with utilities for validation and transformation.

```python
from backend.core.resolution_config import CoordinateSpace

space = CoordinateSpace(width=1920, height=1080, unit="pixels")

# Properties
print(space.center)       # Vector2D(960, 540)
print(space.aspect_ratio) # 1.778

# Validation
is_inside = space.contains_point(500, 300)  # True
is_outside = space.contains_point(-10, 300) # False

# Clamping
clamped_x, clamped_y = space.clamp_point(-10, 2000)  # (0, 1080)

# Normalization
norm_x, norm_y = space.normalize_point(960, 540)  # (0.5, 0.5)
```

### ResolutionConfig

Static utility class for resolution and coordinate operations.

```python
from backend.core.resolution_config import ResolutionConfig

# Get resolutions and dimensions
res = ResolutionConfig.get_resolution("1080p")
dims = ResolutionConfig.get_table_dimensions("9ft")

# Create coordinate spaces
pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
table_space = ResolutionConfig.create_table_space((2.54, 1.27))
centered_space = ResolutionConfig.create_table_space((2.54, 1.27), centered=True)

# Validate coordinates
is_valid, error = ResolutionConfig.validate_coordinates(100, 200, pixel_space)

# Scale coordinates between spaces
scaled_x, scaled_y = ResolutionConfig.scale_coordinates(
    960, 540, pixel_space, table_space
)
```

## Common Use Cases

### 1. Ball Position Validation

```python
from backend.core.resolution_config import ResolutionConfig, get_table_dimensions
from backend.core.models import Vector2D

# Setup table space
table_dims = get_table_dimensions("9ft")
table_space = ResolutionConfig.create_table_space(table_dims)

# Validate ball position
ball_pos = Vector2D(1.27, 0.64)  # Center of table
is_valid, error = ResolutionConfig.validate_vector(ball_pos, table_space)

# Validate with margin for ball radius
ball_radius = 0.028575  # Standard pool ball radius in meters
is_valid, error = ResolutionConfig.validate_vector(
    ball_pos, table_space, margin=ball_radius
)
```

### 2. Pixel to World Coordinate Conversion

```python
from backend.core.resolution_config import ResolutionConfig
from backend.core.models import Vector2D

# Setup spaces
pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
table_space = ResolutionConfig.create_table_space((2.54, 1.27))

# Ball detected at pixel position
detected_pixel = Vector2D(960, 540)

# Convert to table coordinates
table_position = ResolutionConfig.scale_vector(
    detected_pixel, pixel_space, table_space
)
print(f"Ball at table position: ({table_position.x:.2f}m, {table_position.y:.2f}m)")
```

### 3. Coordinate Clamping

```python
from backend.core.resolution_config import ResolutionConfig, get_table_dimensions

# Setup table space
table_space = ResolutionConfig.create_table_space(get_table_dimensions("9ft"))

# Ball position might be slightly out of bounds due to detection noise
detected_pos = Vector2D(2.6, 1.3)  # Slightly outside table

# Clamp to valid table bounds
valid_pos = table_space.clamp_vector(detected_pos)
print(f"Clamped position: ({valid_pos.x:.2f}m, {valid_pos.y:.2f}m)")
```

### 4. Multi-Resolution Support

```python
from backend.core.resolution_config import get_standard_resolution, ResolutionConfig

# Support different camera resolutions
resolutions = ["720p", "1080p", "4K"]

for res_name in resolutions:
    res = get_standard_resolution(res_name)
    space = ResolutionConfig.create_pixel_space(res)
    print(f"{res_name}: {space.width}x{space.height}")
```

## API Reference

### Convenience Functions

- `get_standard_resolution(name: str) -> Optional[Tuple[int, int]]`
  - Get resolution by name or alias
  - Returns `(width, height)` or `None`

- `get_table_dimensions(size: str) -> Optional[Tuple[float, float]]`
  - Get table dimensions by size name or alias
  - Returns `(width, height)` in meters or `None`

- `validate_point_in_space(x, y, width, height, margin=0.0) -> bool`
  - Quick validation for rectangular bounds
  - Returns `True` if point is within bounds

### ResolutionConfig Methods

- `get_resolution(name: str)` - Get resolution with alias support
- `get_table_dimensions(size: str)` - Get table dimensions with alias support
- `create_pixel_space(resolution: Tuple[int, int])` - Create pixel coordinate space
- `create_table_space(table_size: Tuple[float, float], centered: bool = False)` - Create table coordinate space
- `validate_coordinates(x, y, space, margin=0.0)` - Validate coordinates in space
- `validate_vector(position: Vector2D, space, margin=0.0)` - Validate Vector2D in space
- `scale_coordinates(x, y, from_space, to_space)` - Scale coordinates between spaces
- `scale_vector(position: Vector2D, from_space, to_space)` - Scale Vector2D between spaces

## Examples

See `resolution_config_examples.py` for comprehensive usage examples covering:
1. Getting standard resolutions
2. Getting table dimensions
3. Creating coordinate spaces
4. Validating coordinates
5. Scaling coordinates
6. Clamping coordinates
7. Normalizing coordinates
8. Complete workflow from detection to validation

Run the examples:
```bash
python backend/core/resolution_config_examples.py
```

## Design Philosophy

This module follows these principles:

1. **Lightweight**: No external dependencies beyond stdlib and core models
2. **Type-safe**: Uses dataclasses and enums for type safety
3. **Flexible**: Supports multiple naming conventions and aliases
4. **Validated**: All coordinate operations include validation options
5. **Well-documented**: Comprehensive docstrings and examples
6. **Testable**: Full test coverage with integration tests

## Testing

Run the test suite:
```bash
pytest backend/tests/core/test_resolution_config.py -v
```

## Integration

This module integrates with:
- **Vision Module**: For pixel coordinate validation and transformation
- **Core Models**: Uses `Vector2D` from core models
- **Physics Module**: Table dimensions for physics calculations
- **Calibration Module**: Resolution and dimension reference values

## Future Enhancements

Potential additions:
- Support for custom/non-standard resolutions
- Aspect ratio validation and correction
- Projection/perspective transformation utilities
- Distance/scale calculation helpers
- More coordinate system types (polar, etc.)
