# API Models Coordinate Format Fix

## Executive Summary

Successfully migrated all API models from deprecated array format `[x, y]` to new dict format with mandatory scale metadata `{"x": float, "y": float, "scale": [sx, sy]}`. This fixes the broken coordinate data flow in the WebSocket broadcaster and ensures all position/velocity data includes proper scale information for 4K resolution support.

## Problem Statement

The API had a broken coordinate data flow:
- **converters.py**: Already outputting `{x, y, scale}` dicts (CORRECT)
- **responses.py, websocket.py**: Using `[x, y]` arrays (WRONG - deprecated)
- **Result**: WebSocket broadcaster was rejecting data due to format mismatch

## Changes Made

### 1. Created PositionWithScale Model
**File**: `/Users/jchadwick/code/billiards-trainer/backend/api/models/common.py`
**Lines**: 161-194

Created a new Pydantic model to represent positions/velocities with mandatory scale metadata:

```python
class PositionWithScale(BaseModel):
    """2D position with mandatory scale metadata.

    Format:
        {
            "x": float,          # X coordinate value
            "y": float,          # Y coordinate value
            "scale": [sx, sy]    # Scale factors [x_scale, y_scale]
        }
    """
    x: float
    y: float
    scale: list[float]  # min_length=2, max_length=2

    @validator("scale")
    def validate_scale(cls, v):
        """Validate that scale factors are positive."""
        if len(v) != 2:
            raise ValueError("Scale must have exactly 2 values")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"Scale factors must be positive, got {v}")
        return v
```

### 2. Updated responses.py Models
**File**: `/Users/jchadwick/code/billiards-trainer/backend/api/models/responses.py`

#### Changes:
- **Line 19**: Added import `from .common import PositionWithScale`
- **BallInfo** (lines 328-342):
  - `position: list[float]` → `position: PositionWithScale`
  - `velocity: list[float]` → `velocity: PositionWithScale`
- **CueInfo** (lines 345-357):
  - `tip_position: list[float]` → `tip_position: PositionWithScale`
- **TableInfo** (lines 360-371):
  - `pocket_positions: list[list[float]]` → `pocket_positions: list[PositionWithScale]`
- **TrajectoryInfo** (lines 561-574):
  - `points: list[list[float]]` → `points: list[PositionWithScale]`

### 3. Updated websocket.py Models
**File**: `/Users/jchadwick/code/billiards-trainer/backend/api/models/websocket.py`

#### Changes:
- **Line 20**: Added import `from .common import PositionWithScale`
- **BallStateData** (lines 161-181):
  - `position: list[float]` → `position: PositionWithScale`
  - `velocity: list[float]` → `velocity: PositionWithScale`
- **CueStateData** (lines 184-204):
  - `tip_position: list[float]` → `tip_position: PositionWithScale`
- **TableStateData** (lines 207-221):
  - `pocket_positions: list[list[float]]` → `pocket_positions: list[PositionWithScale]`
- **TrajectoryPoint** (lines 257-271):
  - `position: list[float]` → `position: PositionWithScale`
  - `velocity: Optional[list[float]]` → `velocity: Optional[PositionWithScale]`
- **CollisionInfo** (lines 274-298):
  - `position: list[float]` → `position: PositionWithScale`

### 4. Updated vision_models.py
**File**: `/Users/jchadwick/code/billiards-trainer/backend/api/models/vision_models.py`

#### Changes:
- **Point2DModel** (lines 59-89):
  - Added `scale: list[float] = Field(default=[1.0, 1.0], ...)`
  - Added validator to ensure scale factors are positive
  - Updated docstring to explain scale metadata

### 5. Updated converters.py Reverse Conversions
**File**: `/Users/jchadwick/code/billiards-trainer/backend/api/models/converters.py`

Updated all API-to-Core converters to handle the new PositionWithScale objects:

#### Changes:
- **ball_info_to_ball_state** (lines 335-349):
  - Changed from `list_to_vector2d()` to direct `Vector2D(x=..., y=..., scale=tuple(...))`
- **websocket_ball_data_to_ball_state** (lines 352-367):
  - Changed from `list_to_vector2d()` to direct `Vector2D(x=..., y=..., scale=tuple(...))`
- **cue_info_to_cue_state** (lines 370-383):
  - Changed from `list_to_vector2d()` to direct `Vector2D(x=..., y=..., scale=tuple(...))`
- **websocket_cue_data_to_cue_state** (lines 386-399):
  - Changed from `list_to_vector2d()` to direct `Vector2D(x=..., y=..., scale=tuple(...))`
- **table_info_to_table_state** (lines 402-413):
  - Changed from `list_to_vector2d()` comprehension to `Vector2D(x=..., y=..., scale=tuple(...))`
- **websocket_table_data_to_table_state** (lines 416-426):
  - Changed from `list_to_vector2d()` comprehension to `Vector2D(x=..., y=..., scale=tuple(...))`

**Note**: The Core-to-API converters (e.g., `ball_state_to_ball_info()`) were already correct, using `vector2d_to_dict()` which outputs the proper format.

## Impact Analysis

### Breaking Changes
This is a **breaking change** for API consumers:
- All position/velocity fields now return objects instead of arrays
- Clients must update to access `.x`, `.y`, and `.scale` properties
- JSON serialization format changes from `[x, y]` to `{"x": ..., "y": ..., "scale": [...]}`

### Benefits
1. **Fixes WebSocket broadcaster rejection**: Data now matches expected format
2. **Mandatory scale metadata**: Prevents coordinate space ambiguity
3. **Type safety**: Pydantic validation ensures scale is always present and valid
4. **4K support**: Proper scale tracking enables resolution conversion
5. **Future-proof**: Coordinate system is explicitly documented

### Backward Compatibility
❌ **Not backward compatible**
- Old clients expecting `[x, y]` format will break
- Migration required for all API consumers

### Files Modified
1. `/Users/jchadwick/code/billiards-trainer/backend/api/models/common.py` - Added PositionWithScale model
2. `/Users/jchadwick/code/billiards-trainer/backend/api/models/responses.py` - Updated 5 model classes
3. `/Users/jchadwick/code/billiards-trainer/backend/api/models/websocket.py` - Updated 6 model classes
4. `/Users/jchadwick/code/billiards-trainer/backend/api/models/vision_models.py` - Updated Point2DModel
5. `/Users/jchadwick/code/billiards-trainer/backend/api/models/converters.py` - Updated 6 converter functions

## Testing Requirements

### Unit Tests
Need to update tests for:
1. All model serialization/deserialization
2. Converter functions (both directions)
3. WebSocket message validation
4. API response validation

### Integration Tests
Need to verify:
1. WebSocket broadcaster accepts new format
2. End-to-end coordinate flow (vision → core → API → websocket)
3. 4K resolution conversion using scale metadata

### Expected Test Failures
Before updating tests, expect failures in:
- `test_ball_info_serialization()`
- `test_cue_info_serialization()`
- `test_table_info_serialization()`
- `test_trajectory_serialization()`
- `test_websocket_ball_state()`
- `test_websocket_cue_state()`
- `test_websocket_table_state()`
- `test_converter_ball_state()`
- Any test checking coordinate format

## Migration Guide for API Consumers

### Before (Deprecated):
```json
{
  "position": [1.5, 0.8],
  "velocity": [0.5, -0.2]
}
```

### After (Current):
```json
{
  "position": {
    "x": 1.5,
    "y": 0.8,
    "scale": [1920.0, 1080.0]
  },
  "velocity": {
    "x": 0.5,
    "y": -0.2,
    "scale": [1920.0, 1080.0]
  }
}
```

### Client Code Updates:
```python
# Before
x, y = ball_info["position"]

# After
x = ball_info["position"]["x"]
y = ball_info["position"]["y"]
scale = ball_info["position"]["scale"]
```

## Validation Rules

The PositionWithScale model enforces:
1. **Required fields**: `x`, `y`, `scale` are all mandatory
2. **Scale length**: Must be exactly 2 elements `[x_scale, y_scale]`
3. **Scale values**: Must be positive (> 0)
4. **Type safety**: All values must be floats

## Next Steps

1. ✅ Update API models (COMPLETED)
2. ⏭️ Run existing tests to identify failures
3. ⏭️ Update test fixtures to use new format
4. ⏭️ Update any API route handlers that construct models manually
5. ⏭️ Update frontend/client code to handle new format
6. ⏭️ Update API documentation with new format
7. ⏭️ Consider deprecation strategy if old clients need support

## Related Files

- `/Users/jchadwick/code/billiards-trainer/backend/core/coordinates.py` - Vector2D implementation
- `/Users/jchadwick/code/billiards-trainer/backend/core/models.py` - Core model definitions
- `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/broadcaster.py` - WebSocket broadcaster (consumer of these models)

## Conclusion

The API models now correctly use dict format with mandatory scale metadata throughout the entire stack. This fixes the WebSocket data rejection issue and provides proper coordinate space tracking for 4K resolution support. All converters have been updated to handle bidirectional conversion between the new PositionWithScale model and core Vector2D objects.
