# Coordinate Conversion Test Suite

Comprehensive test suite for coordinate space conversions in the billiards trainer system.

## Overview

This test suite validates the coordinate conversion system that transforms between different coordinate spaces:

- **Pixel coordinates**: Camera/vision space (e.g., 1920x1080 pixels)
- **Normalized coordinates**: Resolution-independent 0-1 range
- **Table/World coordinates**: Physics space in meters (e.g., 2.54m x 1.27m)

## Test Coverage

### 1. Vector2D Creation (`TestVector2DCreation`)
- Basic vector creation
- Zero vector
- Equality comparison
- Copy operations

### 2. Pixel ↔ Normalized Conversions
- **`TestPixelToNormalizedConversion`**:
  - Center point (960, 540) → (0.5, 0.5)
  - Origin (0, 0) → (0, 0)
  - Max point (1920, 1080) → (1, 1)
  - Invalid resolution error handling

- **`TestNormalizedToPixelConversion`**:
  - Center (0.5, 0.5) → (960, 540)
  - Origin (0, 0) → (0, 0)
  - Max (1, 1) → (1920, 1080)

### 3. Pixel ↔ Table Conversions
- **`TestPixelToTableConversion`**:
  - Linear scaling (no perspective)
  - Origin, center, and max points
  - Homography matrix support (identity matrix test)

- **`TestTableToPixelConversion`**:
  - Reverse transformations
  - Origin, center, and max points

### 4. Round-Trip Conversions (`TestRoundTripConversions`)
Tests that `A → B → A = A` for all conversion pairs:
- Pixel → Normalized → Pixel
- Normalized → Pixel → Normalized
- Pixel → Table → Pixel
- Table → Pixel → Table
- Normalized → Table → Normalized

### 5. Edge Cases (`TestEdgeCases`)
- Zero coordinates (0, 0)
- Negative pixel coordinates (off-screen points)
- Out-of-range normalized values (> 1.0)
- Very small values (near floating-point precision)
- Very large values (1e6+)

### 6. Resolution Scaling (`TestResolutionScaling`)
- Different resolutions (VGA, 720p, 1080p, 4K)
- Resolution independence through normalized space
- Aspect ratio handling (16:9, 1:1, 9:16)

### 7. Batch Conversions (`TestBatchConversions`)
- Batch pixel to normalized
- Empty list handling
- Single point handling

### 8. Generic Conversion API (`TestGenericConversion`)
- Same-space conversion (returns copy)
- Generic pixel → normalized
- Missing metadata error handling
- Missing resolution error handling

### 9. Error Handling (`TestErrorHandling`)
- Zero resolution dimensions
- Negative resolution dimensions
- Zero table dimensions
- Negative table dimensions
- Invalid homography w component

### 10. Coordinate Metadata (`TestCoordinateMetadata`)
- Metadata creation
- Equality comparison
- String representation

### 11. Property-Based Tests (`TestPropertyBased`)
*Requires hypothesis library*
- Pixel → normalized → pixel roundtrip property
- Normalized values in [0,1] bounds for valid pixels
- Center always maps to (0.5, 0.5)

### 12. Real-World Scenarios (`TestRealWorldScenarios`)
- Ball detection → physics pipeline
- Trajectory visualization pipeline
- Multi-resolution tracking

## Running the Tests

### With pytest (recommended)

```bash
cd backend
pytest tests/unit/test_coordinate_conversion.py -v
```

### Run specific test class

```bash
pytest tests/unit/test_coordinate_conversion.py::TestPixelToNormalizedConversion -v
```

### Run with coverage

```bash
pytest tests/unit/test_coordinate_conversion.py --cov=core --cov-report=html
```

### Standalone validation

If pytest has issues with the conftest, you can run a quick validation:

```bash
cd backend
python -c "
import sys
import importlib.util

spec = importlib.util.spec_from_file_location('test_coord', 'tests/unit/test_coordinate_conversion.py')
test_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_mod)

Vector2D = test_mod.Vector2D
converter = test_mod.CoordinateConverter

# Quick validation
resolution = (1920, 1080)
center = Vector2D(960, 540)
normalized = converter.pixel_to_normalized(center, resolution)
assert abs(normalized.x - 0.5) < 1e-10
assert abs(normalized.y - 0.5) < 1e-10
print('✓ Coordinate conversions working correctly')
"
```

## Test Statistics

- **Test file**: `backend/tests/unit/test_coordinate_conversion.py`
- **Test classes**: 14
- **Test methods**: 60+
- **Lines of code**: 1300+
- **Mock classes**: 3 (CoordinateSpace, CoordinateMetadata, CoordinateConverter)

## Implementation Notes

### Mock Classes

The test file includes mock implementations of the coordinate conversion system:

1. **`CoordinateSpace`**: Enum of coordinate space types
2. **`CoordinateMetadata`**: Metadata container for conversions
3. **`CoordinateConverter`**: Conversion utilities

These can serve as a specification for the actual implementation.

### Vector2D Compatibility

The tests work with both:
- The real `Vector2D` from `core.models` (when available)
- A minimal mock `Vector2D` for standalone testing

### Hypothesis Support

Property-based tests using `hypothesis` are included but will be skipped if hypothesis is not installed. Install with:

```bash
pip install hypothesis
```

## Examples

### Basic Conversion

```python
from core.models import Vector2D
from tests.unit.test_coordinate_conversion import CoordinateConverter

resolution = (1920, 1080)

# Pixel to normalized
pixel = Vector2D(960, 540)
normalized = CoordinateConverter.pixel_to_normalized(pixel, resolution)
# Result: Vector2D(0.5, 0.5)

# Normalized to pixel
normalized = Vector2D(0.75, 0.25)
pixel = CoordinateConverter.normalized_to_pixel(normalized, resolution)
# Result: Vector2D(1440, 270)
```

### Table Conversion

```python
resolution = (1920, 1080)
table_dimensions = (2.54, 1.27)  # 9-foot table in meters

# Pixel to table
pixel = Vector2D(960, 540)
table = CoordinateConverter.pixel_to_table(pixel, resolution, table_dimensions)
# Result: Vector2D(1.27, 0.635) - center of table
```

### Batch Conversion

```python
from tests.unit.test_coordinate_conversion import (
    CoordinateConverter,
    CoordinateSpace,
    CoordinateMetadata,
)

points = [Vector2D(0, 0), Vector2D(960, 540), Vector2D(1920, 1080)]
metadata = CoordinateMetadata(
    space=CoordinateSpace.PIXEL,
    resolution=(1920, 1080),
)

normalized_points = CoordinateConverter.convert_batch(
    points,
    CoordinateSpace.PIXEL,
    CoordinateSpace.NORMALIZED,
    from_metadata=metadata,
)
# Result: [Vector2D(0, 0), Vector2D(0.5, 0.5), Vector2D(1, 1)]
```

## Integration with Codebase

### Current Usage

The coordinate conversion system is used throughout the codebase:

1. **Vision Module** (`backend/vision/`):
   - Detects balls in pixel coordinates
   - Stores `coordinate_space` and `source_resolution` metadata

2. **Core Module** (`backend/core/`):
   - Works in table/world coordinates (meters)
   - Physics calculations require metric units

3. **Integration Service** (`backend/integration_service.py`):
   - Converts between vision (pixel) and core (table) coordinates
   - Maintains calibration data for transformations

### Planned Implementation

The test suite serves as a specification for implementing the actual coordinate conversion system:

1. Create `backend/core/utils/coordinates.py` with:
   - `CoordinateSpace` enum
   - `CoordinateMetadata` class
   - `CoordinateConverter` class

2. Update `Vector2D` in `backend/core/models.py`:
   - Add `coordinate_space` metadata field
   - Add `source_resolution` metadata field

3. Update vision models in `backend/vision/models.py`:
   - Use standardized coordinate metadata
   - Add conversion helper methods

## Future Enhancements

1. **Homography Support**:
   - Implement perspective transformation
   - Add calibration matrix support
   - Test with real calibration data

2. **Additional Coordinate Spaces**:
   - Screen/projector coordinates
   - Camera coordinates
   - Normalized device coordinates (NDC)

3. **Performance Optimization**:
   - Batch numpy operations
   - Cache frequently used conversions
   - Vectorized transformations

4. **Validation**:
   - Coordinate bounds checking
   - Automatic space detection
   - Conversion chain validation

## Related Files

- **Test file**: `backend/tests/unit/test_coordinate_conversion.py`
- **Core models**: `backend/core/models.py` (Vector2D)
- **Vision models**: `backend/vision/models.py` (Ball, CueStick, Table)
- **Integration service**: `backend/integration_service.py`
- **Geometry utils**: `backend/core/utils/geometry.py`

## Contributing

When adding new coordinate conversions:

1. Add test cases to the appropriate test class
2. Test all round-trip conversions
3. Test edge cases (zero, max, negative values)
4. Add error handling tests
5. Update this README with new functionality

## License

Part of the billiards-trainer project.
