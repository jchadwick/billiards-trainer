# Plan: Remove Hardcoded Table Dimensions, Use Calibration

## Current Problem

The system has hardcoded table dimensions in multiple places:
- `TABLE_WIDTH_4K = 3200` and `TABLE_HEIGHT_4K = 1600` in constants_4k.py
- Factory methods like `TableState.standard_9ft_table()` (2.54m × 1.27m)
- Factory methods like `TableState.standard_table_4k()` (3200px × 1600px)
- Validation that expects tables to match these hardcoded values

**This violates the fundamental principle that table boundaries must come from calibration.**

## How Calibration Works

The calibration system (`vision/calibration/geometry.py`) captures:

```python
@dataclass
class GeometricCalibration:
    table_dimensions_real: tuple[float, float]  # Real world (meters), e.g., (2.54, 1.27)
    table_corners_pixel: list[tuple[float, float]]  # Corners in camera pixels
    perspective_correction: PerspectiveCorrection  # Transform matrix
    calibration_error: float
```

This provides:
1. **Real-world dimensions** - The actual table size in meters
2. **Pixel corners** - Where the table appears in the camera image
3. **Transform matrix** - How to correct perspective distortion

## Correct Data Flow

### Current (Wrong):
```
Vision Detection → Hardcoded Constants → Game State
                    ↓
                  3200×1600 (ignored reality)
```

### Correct:
```
Calibration → Table Corners (camera pixels) → Resolution Transform → Table Bounds (4K pixels) → Game State
                                                    ↓
                                   Actual table size in canonical coordinates
```

## Key Insight: 4K Resolution vs Table Dimensions

**4K Resolution (3840×2160)** is the **coordinate system canvas**, not the table size.

The table size in 4K coordinates should be calculated from:
1. Detected/calibrated table corners in camera space
2. Transformed to 4K canonical space using resolution conversion
3. Width/height calculated from transformed corners

Example:
- Camera resolution: 1920×1080
- Detected table corners: [(100,50), (1800,50), (1800,1000), (100,1000)]
- Camera table width: ~1700 pixels
- Transform to 4K (scale 2x): ~3400 pixels in 4K space
- This is the **actual** table width in 4K coordinates (NOT 3200!)

## What TABLE_WIDTH_4K and TABLE_HEIGHT_4K Really Mean

These constants were **incorrectly conceptualized**. They represent:
- A **theoretical** table size assuming specific camera positioning
- **NOT** a canonical table size that all tables must match

The truth:
- **4K Resolution (3840×2160)**: The coordinate system canvas (KEEP THIS)
- **Table dimensions**: Variable, depends on calibration (REMOVE HARDCODED VALUES)

## Implementation Plan

### Phase 1: Understand Current State ✓
- [x] Audit all hardcoded table dimensions
- [x] Understand calibration system
- [x] Identify correct data flow

### Phase 2: Remove Hardcoded Table Dimensions

#### 2.1 Update constants_4k.py
```python
# REMOVE (these assume specific camera positioning):
# TABLE_WIDTH_4K = 3200
# TABLE_HEIGHT_4K = 1600
# TABLE_LEFT_4K = 320
# TABLE_TOP_4K = 280
# TABLE_RIGHT_4K = 3520
# TABLE_BOTTOM_4K = 1880
# POCKET_POSITIONS_4K = [...]

# KEEP (these are coordinate system bounds):
CANONICAL_WIDTH = 3840  # 4K frame width
CANONICAL_HEIGHT = 2160  # 4K frame height
CANONICAL_RESOLUTION = (3840, 2160)
```

**Rationale:** Table dimensions depend on camera position and calibration, not fixed constants.

#### 2.2 Remove Factory Methods
```python
# REMOVE from models.py:
# - TableState.standard_9ft_table()
# - TableState.standard_table_4k()
```

**Replacement:** Require table to be created from calibration data

#### 2.3 Update TableState Creation
```python
# NEW factory method:
@classmethod
def from_calibration(
    cls,
    calibration: GeometricCalibration,
    camera_resolution: tuple[int, int]
) -> "TableState":
    """Create TableState from calibration data.

    Args:
        calibration: Geometric calibration with table dimensions and corners
        camera_resolution: Camera resolution for coordinate transformation

    Returns:
        TableState with dimensions in 4K canonical coordinates
    """
    # Transform corners from camera pixels to 4K canonical
    corners_4k = [
        ResolutionConverter.convert_coordinates(
            corner[0], corner[1],
            camera_resolution,
            CANONICAL_RESOLUTION
        )
        for corner in calibration.table_corners_pixel
    ]

    # Calculate table dimensions from transformed corners
    width_4k = calculate_width_from_corners(corners_4k)
    height_4k = calculate_height_from_corners(corners_4k)

    # Calculate pocket positions from corners
    pocket_positions = calculate_pocket_positions(corners_4k)

    return cls(
        width=width_4k,
        height=height_4k,
        pocket_positions=pocket_positions,
        playing_area_corners=[Vector2D.from_4k(x, y) for x, y in corners_4k],
        # Real-world dimensions stored as metadata
        physical_width=calibration.table_dimensions_real[0],
        physical_height=calibration.table_dimensions_real[1],
    )
```

#### 2.4 Update GameStateManager
```python
# REMOVE:
def _create_default_table(self) -> TableState:
    return TableState.standard_9ft_table()

# REPLACE with:
def __init__(self, calibration: GeometricCalibration, camera_resolution: tuple[int, int], ...):
    """Initialize game state manager.

    Args:
        calibration: Geometric calibration data (REQUIRED)
        camera_resolution: Camera resolution (REQUIRED)
        ...
    """
    self._table = TableState.from_calibration(calibration, camera_resolution)
    # No default table - calibration is mandatory
```

**Rationale:** System cannot operate without calibration

#### 2.5 Update Validation
```python
# REMOVE validation against hardcoded dimensions:
if abs(table.width - TABLE_WIDTH_4K) > dimension_tolerance:
    result.add_warning(...)

# REPLACE with reasonable bounds checking:
# Table should fit within 4K frame
if table.width <= 0 or table.width > CANONICAL_WIDTH:
    result.add_error(f"Table width {table.width} outside valid range (0, {CANONICAL_WIDTH})")
if table.height <= 0 or table.height > CANONICAL_HEIGHT:
    result.add_error(f"Table height {table.height} outside valid range (0, {CANONICAL_HEIGHT})")

# Validate aspect ratio is reasonable for pool tables (1.5:1 to 2.5:1)
aspect_ratio = table.width / table.height
if not 1.5 <= aspect_ratio <= 2.5:
    result.add_warning(f"Table aspect ratio {aspect_ratio:.2f} unusual for pool table")
```

#### 2.6 Update Integration Service
```python
# Use actual detected table dimensions from vision
# No clamping to hardcoded values
detection_data["table"] = {
    "corners": table.corners,  # From vision detection
    "width": calculate_width(table.corners),  # Actual width
    "height": calculate_height(table.corners),  # Actual height
    "pockets": table.pockets,
}
```

### Phase 3: Update Tests

All tests must provide calibration data instead of using factory methods:

```python
# Create test calibration
calibration = GeometricCalibration(
    table_dimensions_real=(2.54, 1.27),  # 9-foot table
    table_corners_pixel=[(100, 50), (1820, 50), (1820, 1030), (100, 1030)],
    perspective_correction=None,
    coordinate_mapping=None,
    calibration_error=0.5,
    calibration_date=datetime.now().isoformat(),
)

# Create table from calibration
table = TableState.from_calibration(calibration, camera_resolution=(1920, 1080))
```

### Phase 4: Migration Path

For existing deployments without calibration:

1. **Add calibration requirement check on startup**
2. **Provide calibration wizard in UI**
3. **Generate default calibration from camera resolution** (temporary fallback):
   ```python
   def generate_default_calibration(camera_resolution: tuple[int, int]) -> GeometricCalibration:
       """Generate default calibration assuming table fills 85% of frame."""
       width, height = camera_resolution
       margin_x = int(width * 0.075)
       margin_y = int(height * 0.075)

       return GeometricCalibration(
           table_dimensions_real=(2.54, 1.27),  # Standard 9ft
           table_corners_pixel=[
               (margin_x, margin_y),
               (width - margin_x, margin_y),
               (width - margin_x, height - margin_y),
               (margin_x, height - margin_y),
           ],
           perspective_correction=None,
           coordinate_mapping=None,
           calibration_error=999.0,  # High error = uncalibrated
           calibration_date=datetime.now().isoformat(),
       )
   ```

## Benefits

1. **Accurate to reality** - Table dimensions match actual setup
2. **Flexible** - Works with any table size, camera position
3. **Calibration-driven** - Forces proper system setup
4. **No magic numbers** - All dimensions come from measurement
5. **Better validation** - Validates against physical constraints, not arbitrary constants

## Files to Modify

1. `backend/core/constants_4k.py` - Remove table dimension constants
2. `backend/core/models.py` - Remove factory methods, add from_calibration()
3. `backend/core/game_state.py` - Require calibration in __init__
4. `backend/core/validation/state.py` - Update validation logic
5. `backend/core/validation/physics.py` - Update PIXELS_PER_METER calculation
6. `backend/integration_service.py` - Use actual detected dimensions
7. `backend/integration_service_conversion_helpers.py` - Remove hardcoded validation bounds
8. All test files - Use calibration-based table creation

## Success Criteria

- [ ] No hardcoded table dimensions anywhere (except test fixtures)
- [ ] System requires calibration to start
- [ ] Table dimensions reflect actual camera view
- [ ] Validation accepts reasonable table dimensions
- [ ] All tests pass with calibration-based tables
- [ ] Physics calculations use actual table dimensions
