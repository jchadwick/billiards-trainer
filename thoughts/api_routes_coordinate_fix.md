# API Routes Coordinate Fix Summary

## Overview
Fixed API routes in `backend/api/routes/` to preserve scale metadata when converting from core models to API responses.

## Problem
The game.py route was manually extracting x,y coordinates using array notation `[x, y]`, which stripped the scale metadata that is mandatory in all Vector2D instances. This caused downstream issues where coordinate conversions couldn't determine the proper scale.

## Solution
Updated conversion functions to use `vector2d_to_dict()` from `backend/api/models/converters.py`, which preserves scale metadata in the format:
```python
{
    "x": float,
    "y": float,
    "scale": [scale_x, scale_y]
}
```

## Files Modified

### 1. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/game.py`

#### Changes Made:

**Import Addition (Line 30):**
```python
from api.models.converters import vector2d_to_dict
```

**Function: `convert_ball_state_to_info()` (Lines 49-63):**
- **Before:**
  ```python
  position=[ball.position.x, ball.position.y],
  velocity=[ball.velocity.x, ball.velocity.y],
  ```
- **After:**
  ```python
  position=vector2d_to_dict(ball.position),  # Preserves scale metadata
  velocity=vector2d_to_dict(ball.velocity),  # Preserves scale metadata
  ```
- **Impact:** All ball position and velocity data now includes scale metadata

**Function: `convert_cue_state_to_info()` (Lines 66-78):**
- **Before:**
  ```python
  tip_position=[cue.tip_position.x, cue.tip_position.y],
  ```
- **After:**
  ```python
  tip_position=vector2d_to_dict(cue.tip_position),  # Preserves scale metadata
  ```
- **Impact:** Cue tip position now includes scale metadata

**Function: `convert_table_state_to_info()` (Lines 81-92):**
- **Before:**
  ```python
  pocket_positions=[[p.x, p.y] for p in table.pocket_positions],
  ```
- **After:**
  ```python
  pocket_positions=[vector2d_to_dict(p) for p in table.pocket_positions],  # Preserves scale metadata
  ```
- **Impact:** All pocket positions now include scale metadata

### 2. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/calibration.py`
- **Status:** ✅ No changes needed
- **Reason:** This file does not perform any Vector2D to coordinate array conversions
- **Verification:** Searched for patterns like `[x, y]` extraction - none found

### 3. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/vision.py`
- **Status:** ✅ No changes needed
- **Reason:** This file does not perform any Vector2D to coordinate array conversions
- **Verification:** Searched for patterns like `[x, y]` extraction - none found

### 4. Other Route Files Checked:
- `/Users/jchadwick/code/billiards-trainer/backend/api/routes/stream.py` - ✅ No changes needed
- `/Users/jchadwick/code/billiards-trainer/backend/api/routes/config.py` - ✅ No changes needed
- `/Users/jchadwick/code/billiards-trainer/backend/api/routes/diagnostics.py` - ✅ No changes needed
- `/Users/jchadwick/code/billiards-trainer/backend/api/routes/health.py` - ✅ No changes needed
- `/Users/jchadwick/code/billiards-trainer/backend/api/routes/modules.py` - ✅ No changes needed

## API Response Model Compatibility

The API response models in `/Users/jchadwick/code/billiards-trainer/backend/api/models/responses.py` already support dict format with scale metadata:

### BallInfo (Line 328)
```python
position: PositionWithScale = Field(..., description="Ball position with scale metadata")
velocity: PositionWithScale = Field(..., description="Ball velocity with scale metadata")
```

### CueInfo (Line 345)
```python
tip_position: PositionWithScale = Field(..., description="Cue tip position with scale metadata")
```

### TableInfo (Line 360)
```python
pocket_positions: list[PositionWithScale] = Field(..., description="Pocket positions with scale metadata")
```

The `PositionWithScale` model accepts the dict format returned by `vector2d_to_dict()`.

## Converter Implementation

The `vector2d_to_dict()` function in `/Users/jchadwick/code/billiards-trainer/backend/api/models/converters.py` (Lines 70-96):

```python
def vector2d_to_dict(vector: Vector2D) -> dict[str, Any]:
    """Convert core Vector2D to dict with mandatory scale.

    Returns:
        Dictionary with x, y, and scale fields:
        {
            "x": float,
            "y": float,
            "scale": [scale_x, scale_y]
        }

    Raises:
        ValueError: If vector does not have scale metadata
    """
    # Validates scale metadata is present and valid
    if not hasattr(vector, 'scale') or vector.scale is None:
        raise ValueError("Vector2D must have scale metadata")
    if not isinstance(vector.scale, (tuple, list)) or len(vector.scale) != 2:
        raise ValueError(f"Scale must be a 2-element tuple or list...")
    if vector.scale[0] <= 0 or vector.scale[1] <= 0:
        raise ValueError(f"Scale factors must be positive...")

    return {
        "x": vector.x,
        "y": vector.y,
        "scale": list(vector.scale)  # Convert tuple to list for JSON
    }
```

## Impact Analysis

### Endpoints Affected

All endpoints in `/api/game/` that return game state data:

1. **GET `/api/game/state`** (Line 120)
   - Returns current game state with balls, cue, and table
   - All coordinates now include scale metadata

2. **GET `/api/game/history`** (Line 189)
   - Returns historical game states
   - All coordinates in historical data now include scale metadata

3. **POST `/api/game/reset`** (Line 277)
   - Returns new game state after reset
   - All coordinates now include scale metadata

4. **POST `/api/game/export`** (Line 365)
   - Exports session data
   - All coordinates in exported data now include scale metadata

### Data Flow

```
Core Models (Vector2D with scale)
    ↓
convert_*_to_info() [using vector2d_to_dict()]
    ↓
API Response Models (PositionWithScale)
    ↓
JSON Response (dict with x, y, scale)
    ↓
Client receives complete coordinate data
```

## Testing Recommendations

1. **Unit Tests:**
   - Test `convert_ball_state_to_info()` ensures scale metadata is present
   - Test `convert_cue_state_to_info()` ensures scale metadata is present
   - Test `convert_table_state_to_info()` ensures scale metadata is present

2. **Integration Tests:**
   - Call GET `/api/game/state` and verify response includes scale in all coordinates
   - Call GET `/api/game/history` and verify historical data includes scale
   - Call POST `/api/game/reset` and verify new state includes scale

3. **Validation Tests:**
   - Verify ValueError is raised if Vector2D lacks scale metadata
   - Verify scale values are positive
   - Verify scale is a 2-element tuple/list

## Backward Compatibility

⚠️ **Breaking Change:** This changes the response format from:
```json
{
  "position": [100.0, 200.0]
}
```

To:
```json
{
  "position": {
    "x": 100.0,
    "y": 200.0,
    "scale": [1920.0, 1080.0]
  }
}
```

**Impact:**
- Clients expecting array format will need to update to dict format
- The API models already support this format via `PositionWithScale`
- This is consistent with the rest of the 4K migration effort

## Verification Commands

```bash
# Search for any remaining array-style coordinate conversions
cd /Users/jchadwick/code/billiards-trainer/backend
grep -r "\[.*\.x.*,.*\.y.*\]" api/routes/

# Verify imports are correct
grep "from api.models.converters import" api/routes/game.py

# Run tests to ensure everything still works
pytest backend/tests/unit/test_coordinate_conversion.py -v
```

## Related Documentation

- `/Users/jchadwick/code/billiards-trainer/backend/core/MODELS_COORDINATE_MIGRATION.md` - Core models migration
- `/Users/jchadwick/code/billiards-trainer/backend/core/COORDINATE_CONVERTER_EXAMPLES.md` - Conversion examples
- `/Users/jchadwick/code/billiards-trainer/backend/api/models/converters.py` - Converter implementation

## Conclusion

✅ **All API routes now preserve scale metadata when converting from core models to API responses**

- Modified 3 conversion functions in `game.py`
- Added import for `vector2d_to_dict()`
- Verified other route files don't need changes
- Confirmed API response models support the new format
- Documented the change for future reference

The fix ensures that coordinate scale information flows correctly from the vision system through the core logic and out to API clients, maintaining consistency across the entire 4K migration effort.
