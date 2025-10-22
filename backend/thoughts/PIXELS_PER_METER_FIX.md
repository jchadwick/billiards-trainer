# Pixels Per Meter Calculation Fix

## Problem

The `pixels_per_meter` calculation was using the full camera resolution width instead of the actual table region width, resulting in incorrect world coordinate conversions.

### Before (Wrong):
```
pixels_per_meter = camera_width / table_width_meters
                 = 1920px / 2.54m
                 = 756 px/m
```

### After (Correct):
```
pixels_per_meter = table_region_width / table_width_meters
                 = 569px / 2.54m
                 = 224 px/m
```

## Analysis

From the config file, we have:
- **Camera resolution**: 1920×1080 pixels
- **Table region corners** (from calibration):
  - Top-left: (37, 45)
  - Top-right: (606, 39)
  - Bottom-right: (604, 326)
  - Bottom-left: (40, 322)
- **Table region dimensions**: ~569×287 pixels
- **Real table size**: 2.54m × 1.27m (9ft table)

### Calculation Details

**Width calculation:**
```
Top edge width:  sqrt((606-37)^2 + (39-45)^2)  = 569.0px
Bottom edge width: sqrt((604-40)^2 + (326-322)^2) = 564.0px
Average width: (569.0 + 564.0) / 2 = 566.5px

pixels_per_meter = 566.5px / 2.54m = 223.0 px/m
```

**Height calculation (for verification):**
```
Left edge height:  sqrt((40-37)^2 + (322-45)^2)  = 277.0px
Right edge height: sqrt((604-606)^2 + (326-39)^2) = 287.0px
Average height: (277.0 + 287.0) / 2 = 282.0px

pixels_per_meter = 282.0px / 1.27m = 222.0 px/m
```

The width and height calculations are very close (223 vs 222), confirming the accuracy.

### Impact on Coordinates

**Example: Ball at pixel position (300, 150)**

Old (wrong) calculation:
- X: 300px / 756 px/m = 0.397m
- Y: 150px / 756 px/m = 0.198m

New (correct) calculation:
- X: 300px / 224 px/m = 1.339m
- Y: 150px / 224 px/m = 0.670m

The new values are **3.375x larger** than the old values.

## Implementation

### Changes Made

1. **`integration_service_conversion_helpers.py`**:
   - Modified `_create_coordinate_converter()` to:
     - Calculate pixels_per_meter from table corners when available
     - Update `self.pixels_per_meter` instance variable for legacy methods
     - Save calculated value to config for persistence
   - Added logging to show calculated values

2. **`integration_service.py`**:
   - Added `_initialize_table_corners_from_config()` method to load corners on startup
   - Modified `_process_detection()` to update table corners when table is detected
   - Call initialization in `__init__()` to set up corners from config

### Verification

Test script results:
```
Expected pixels_per_meter:
  From width: 224.0 px/m
  From height: 226.0 px/m

Old WRONG calculation (full camera width):
  755.9 px/m
  Correction factor: 0.296x

Calculated pixels_per_meter: 223.0 px/m

✓ SUCCESS: pixels_per_meter matches expected value!
✓ SUCCESS: Coordinate conversion is working correctly!
```

## Benefits

1. **Accurate world coordinates**: Ball and cue positions are now correctly converted from pixel space to meter space
2. **Proper trajectory calculations**: Physics engine receives accurate position data
3. **Correct distance measurements**: All distance-based calculations (collisions, velocities) are now accurate
4. **Persistent calibration**: Calculated value is saved to config for reuse across sessions
5. **Dynamic updates**: System recalculates when table corners change (e.g., camera moved)

## Configuration

The calculated `pixels_per_meter` is saved to:
```json
{
  "vision": {
    "calibration": {
      "pixels_per_meter": 223.0
    }
  }
}
```

And uses table corners from:
```json
{
  "table": {
    "playing_area_corners": [
      {"x": 37, "y": 45},
      {"x": 606, "y": 39},
      {"x": 604, "y": 326},
      {"x": 40, "y": 322}
    ]
  }
}
```

## Future Enhancements

1. Add perspective transform to handle camera angle distortion
2. Support non-rectangular tables or warped camera views
3. Add validation to detect when camera has moved significantly
4. Provide calibration UI to adjust table corners interactively
