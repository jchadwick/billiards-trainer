# 4K Coordinate System Comprehensive Audit
**Date**: 2025-10-21
**Scope**: Full codebase scan for vector/coordinate references
**Method**: 5 parallel subagent searches

---

## EXECUTIVE SUMMARY

### Migration Status by Module

| Module | Status | Summary |
|--------|--------|---------|
| **backend/core/** | ✅ **COMPLETE** | Fully migrated to 4K with Vector2D + scale metadata |
| **backend/tests/** | ✅ **COMPLETE** | Comprehensive 4K validation tests |
| **backend/video/** | ✅ **CORRECT** | No coordinate handling (frame delivery only) |
| **backend/streaming/** | ✅ **CORRECT** | Pixel-only processing (no conversion needed) |
| **backend/integration_service*.py** | ✅ **MIGRATED** | Proper 4K conversion implementation |
| **backend/api/** | ❌ **CRITICAL ISSUES** | Mixed array/dict formats, scale metadata stripped |
| **backend/vision/** | ⚠️ **NOT MIGRATED** | Uses tuples in native camera resolution |

---

## CRITICAL ISSUES REQUIRING IMMEDIATE FIXES

### 1. API Module - Format Inconsistency Crisis

**Problem**: The API layer has a **broken coordinate data flow** where different parts use incompatible formats:

```
Core Models (Vector2D with scale)
    ↓
converters.py (outputs {x, y, scale} dicts) ✅ CORRECT
    ↓
routes/game.py (strips to [x, y] arrays) ❌ BREAKS HERE
    ↓
responses.py/websocket.py (expects [x, y] arrays) ❌ WRONG FORMAT
    ↓
WebSocket broadcaster (expects {x, y, scale} dicts) ❌ VALIDATION FAILS
    ↓
❌ DATA DROPPED / CLIENTS BROKEN
```

**Affected Files**:
- `backend/api/routes/game.py` - strips scale metadata when converting
- `backend/api/models/responses.py` - defines `position: list[float]` (deprecated)
- `backend/api/models/websocket.py` - defines `position: list[float]` (deprecated)
- `backend/api/websocket/schemas.py` - enforces array format in Pydantic schemas
- `backend/api/routes/debug.py` - JavaScript expects `{x, y, scale}` but may receive arrays

**Impact**:
- WebSocket broadcaster **rejects** game state data with array positions (warns and drops data)
- Debug page **fails to render** when receiving array positions
- Clients have **no coordinate space metadata** to know what resolution coordinates represent
- **Data loss** - scale information discarded during conversion

### 2. Vision Module - No 4K Integration

**Problem**: Vision module operates in **native camera resolution** (typically 1920×1080) with no 4K standardization:

```python
# vision/models.py uses tuples in camera pixels
Ball.position: tuple[float, float]  # (960, 540) in 1080p
Ball.radius: float  # 20 pixels in 1080p

# vision/detection/balls.py has hardcoded pixel values
min_radius: int = 15  # pixels - NOT resolution-aware
max_radius: int = 26  # pixels - NOT resolution-aware
expected_radius: int = 20  # pixels - NOT resolution-aware
```

**Issues**:
- Uses `tuple[float, float]` instead of `Vector2D`
- No integration with `resolution_converter.py`
- Hardcoded pixel thresholds break at different resolutions
- No coordinate space metadata (can't tell if it's 4K vs 1080p vs 720p)

**Affected Files**:
- `backend/vision/models.py` - uses tuples instead of Vector2D
- `backend/vision/detection/balls.py` - hardcoded pixel radius values
- `backend/vision/detection/cue.py` - hardcoded pixel distance thresholds
- `backend/vision/detection/detector_adapter.py` - outputs in camera pixels

### 3. Integration Service - Coordinate Format Mismatch

**Problem**: Core's `asdict(GameState)` serialization produces dict format but broadcaster expects array format:

```python
# integration_service.py line ~200
state_dict = asdict(game_state)  # Produces {"position": {"x": ..., "y": ...}}

# But broadcaster.py expects:
# {"position": [x, y]}  # Array format

# Requires manual conversion in integration_service_conversion_helpers.py
```

**Impact**: Requires manual conversion layer to bridge incompatible formats

---

## DETAILED MODULE ANALYSES

### backend/core/ - ✅ FULLY MIGRATED

**Status**: Perfect 4K implementation with comprehensive Vector2D + scale metadata

**Key Achievements**:
1. **Vector2D class** (`core/coordinates.py`):
   - Mandatory `scale` metadata tuple `(scale_x, scale_y)`
   - Factory methods: `from_4k(x, y)` creates with `scale=(1.0, 1.0)`
   - Factory methods: `from_resolution(x, y, resolution)` auto-calculates scale
   - Conversion methods: `to_4k_canonical()`, `to_resolution(target_resolution)`
   - All arithmetic operations preserve scale metadata

2. **4K Constants** (`core/constants_4k.py`):
   - `CANONICAL_RESOLUTION = (3840, 2160)`
   - `TABLE_WIDTH_4K = 3200`, `TABLE_HEIGHT_4K = 1600` pixels
   - `BALL_RADIUS_4K = 36` pixels
   - `POCKET_RADIUS_4K = 72` pixels
   - All pocket positions in 4K coordinates

3. **Resolution Converter** (`core/resolution_converter.py`):
   - `calculate_scale_to_4k(source_resolution)` - returns (scale_x, scale_y)
   - `scale_to_4k(x, y, source_resolution)` - converts coords to 4K
   - `scale_from_4k(x_4k, y_4k, target_resolution)` - converts from 4K
   - Distance/radius scaling methods

4. **Models** (`core/models.py`):
   - `BallState.position: Vector2D` with scale
   - `BallState.velocity: Vector2D` with scale
   - `CueState.tip_position: Vector2D` with scale
   - `TableState.pocket_positions: list[Vector2D]` with scale
   - Factory methods: `from_4k()`, `from_resolution()`

5. **Physics Engine** (`core/physics/`):
   - All calculations in 4K pixels
   - `TrajectoryCalculator` uses 4K coordinates throughout
   - `CollisionDetector` operates in 4K pixels
   - Constants in pixels: `BALL_RADIUS = 36.0`, `GRAVITY = 9.81 * PIXELS_PER_METER`

**No Issues Found** - Complete, correct, and comprehensive 4K implementation

---

### backend/tests/ - ✅ COMPREHENSIVE 4K VALIDATION

**Status**: Excellent test coverage validating the 4K coordinate system

**Test Coverage**:

1. **Vector2D & Scale Tests** (`test_vector2d_4k.py`):
   - Factory methods (`from_4k`, `from_resolution`)
   - Scale preservation (1080p→2.0x, 720p→3.0x)
   - Conversions (`to_4k_canonical`, `to_resolution`)
   - Round-trip accuracy
   - Geometric operations
   - Serialization with scale

2. **Coordinate Conversion Tests** (3 files):
   - Multiple coordinate spaces (pixel, normalized, meters, 4K)
   - Perspective transformations with homography
   - Batch conversions
   - Round-trip accuracy
   - Edge cases

3. **Resolution Scaling Tests** (`test_resolution_converter.py`):
   - Scale calculations (1080p→2.0x, 720p→3.0x)
   - Coordinate scaling
   - Distance/radius scaling
   - Round-trip accuracy

4. **4K Constants Tests** (`test_constants_4k.py`):
   - Validates all 4K constants
   - Table bounds checking
   - Pocket positions
   - Coordinate validation helpers

5. **Vision Integration Tests** (`test_vision_integration_4k_conversion.py`):
   - 1080p→4K conversion (960,540 → 1920,1080)
   - 720p→4K conversion (640,360 → 1920,1080)
   - Ball/cue/table conversions
   - Velocity scaling
   - Position clamping

**Key Finding**: No deprecated patterns found in tests - all use Vector2D or proper tuple formats

---

### backend/video/ - ✅ CORRECTLY ISOLATED

**Status**: No coordinate handling (intentional and correct)

**Functionality**:
- Raw frame delivery via shared memory
- Only stores frame dimensions as metadata
- No coordinate conversions or transformations
- Correctly isolated from coordinate system concerns

**Files**:
- `video/process.py` - frame capture and buffering
- `video/__main__.py` - video service entry point

**No Issues** - Proper separation of concerns

---

### backend/streaming/ - ✅ CORRECTLY ISOLATED

**Status**: Pixel-only image processing (no conversion needed)

**Functionality**:
- `enhanced_camera_module.py` - frame preprocessing
- Auto-detects resolution
- Table cropping in camera pixel space
- No coordinate conversion (not its responsibility)

**No Issues** - Operates correctly in camera pixel space

---

### backend/integration_service*.py - ✅ PROPERLY MIGRATED

**Status**: Correct 4K conversion implementation with minor issues

**Key Implementation** (`integration_service_conversion_helpers.py`):
```python
# Proper 4K conversion pattern
def convert_ball_position(ball_data, source_resolution):
    # Create Vector2D from source resolution
    position = Vector2D.from_resolution(
        ball_data["x"],
        ball_data["y"],
        source_resolution
    )
    # Convert to 4K canonical
    position_4k = position.to_4k_canonical()
    return {"x": position_4k.x, "y": position_4k.y, "scale": position_4k.scale}
```

**Issues**:
1. ⚠️ Core's `asdict()` produces dicts but broadcaster expects arrays
2. ⚠️ Legacy deprecated methods not removed
3. ⚠️ Validation uses hardcoded 4K pixel bounds while config uses meters

**Overall**: Good implementation with cleanup needed

---

### backend/api/ - ❌ CRITICAL ISSUES

**Status**: Broken coordinate data flow with format inconsistencies

#### Files Analysis:

**✅ CORRECT FILES**:

1. **api/models/converters.py** - Properly implements dict format with scale:
   ```python
   def vector2d_to_dict(vector: Vector2D) -> dict:
       if not hasattr(vector, 'scale') or vector.scale is None:
           raise ValueError("Vector2D must have scale metadata")
       return {
           "x": vector.x,
           "y": vector.y,
           "scale": [vector.scale[0], vector.scale[1]]
       }
   ```

2. **api/websocket/broadcaster.py** - Validates scale metadata:
   ```python
   # Lines 325-359: Strict validation
   if "scale" not in position:
       logger.warning("position missing mandatory 'scale' metadata")
       return
   if not isinstance(position["scale"], (list, tuple)):
       logger.warning("invalid scale format")
       return
   ```

3. **api/routes/debug.py** - JavaScript expects scale metadata:
   ```javascript
   if (firstBall && firstBall.position && firstBall.position.scale) {
       const scaleX = firstBall.position.scale[0];
       const scaleY = firstBall.position.scale[1];
       // Calculate source resolution from scale
   }
   ```

**❌ BROKEN FILES**:

1. **api/routes/game.py** - Strips scale metadata:
   ```python
   # Lines 48-59
   def convert_ball_state_to_info(ball: BallState) -> BallInfo:
       return BallInfo(
           position=[ball.position.x, ball.position.y],  # ❌ LOSES SCALE
           velocity=[ball.velocity.x, ball.velocity.y],  # ❌ LOSES SCALE
           ...
       )
   ```

2. **api/models/responses.py** - Uses deprecated array format:
   ```python
   class BallInfo(BaseModel):
       position: list[float]  # ❌ Should be dict with scale
       velocity: list[float]  # ❌ Should be dict with scale
   ```

3. **api/models/websocket.py** - Uses deprecated array format:
   ```python
   class BallStateData(BaseModel):
       position: list[float]  # ❌ Should be dict with scale
       velocity: list[float]  # ❌ Should be dict with scale
   ```

4. **api/websocket/schemas.py** - Enforces array format:
   ```python
   class BallData(BaseModel):
       position: list[float] = Field(..., min_length=2, max_length=2)  # ❌
   ```

5. **api/models/vision_models.py** - No scale metadata:
   ```python
   class Point2DModel(BaseModel):
       x: float
       y: float
       # ❌ No scale field
   ```

**Impact**: Creates a broken data pipeline where broadcaster rejects game state data

---

### backend/vision/ - ⚠️ NOT MIGRATED TO 4K

**Status**: Uses tuples in native camera resolution (no 4K standardization)

**Current Implementation**:
```python
# vision/models.py
class Ball:
    position: tuple[float, float]  # (x, y) in camera pixels
    radius: float  # in camera pixels
    coordinate_space: str = "pixel"
    source_resolution: Optional[tuple[int, int]]  # Metadata only
```

**Hardcoded Pixel Values** (resolution-dependent):
```python
# vision/detection/balls.py
min_radius: int = 15  # pixels - breaks at different resolutions
max_radius: int = 26  # pixels
expected_radius: int = 20  # pixels

# vision/detection/cue.py
max_cue_length = 800  # pixels
max_distance_to_cue_ball = 40  # pixels
max_tip_distance = 300  # pixels
```

**What Works**:
- ✅ Consistent tuple format throughout
- ✅ Coordinate space metadata tracking
- ✅ Homography-based pixel↔world transformations
- ✅ Resolution info stored in calibration

**What's Missing**:
- ❌ No Vector2D usage
- ❌ No resolution_converter integration
- ❌ No 4K standardization
- ❌ Hardcoded values not resolution-aware

**Migration Needed**:
1. Add `resolution_converter` integration in `detector_adapter.py`
2. Convert hardcoded pixel values to resolution-relative
3. Output positions in 4K canonical instead of camera pixels
4. Optionally: Convert tuples to Vector2D for consistency

---

## MIGRATION PRIORITY PLAN

### PHASE 1: CRITICAL - Fix API Module (IMMEDIATE)

**Goal**: Standardize API on dict format with mandatory scale metadata

**Tasks**:
1. Update `api/models/responses.py`:
   ```python
   class PositionWithScale(BaseModel):
       x: float
       y: float
       scale: list[float] = Field(..., min_length=2, max_length=2)

   class BallInfo(BaseModel):
       position: PositionWithScale  # Changed from list[float]
       velocity: PositionWithScale  # Changed from list[float]
   ```

2. Update `api/models/websocket.py`:
   ```python
   class BallStateData(BaseModel):
       position: PositionWithScale  # Changed from list[float]
       velocity: PositionWithScale  # Changed from list[float]
   ```

3. Update `api/websocket/schemas.py`:
   ```python
   class BallData(BaseModel):
       position: PositionWithScale  # Changed from list[float]
   ```

4. Update `api/routes/game.py`:
   ```python
   from api.models.converters import vector2d_to_dict

   def convert_ball_state_to_info(ball: BallState) -> BallInfo:
       return BallInfo(
           position=vector2d_to_dict(ball.position),  # Preserves scale
           velocity=vector2d_to_dict(ball.velocity),  # Preserves scale
           ...
       )
   ```

5. Update `api/models/vision_models.py`:
   ```python
   class Point2DModel(BaseModel):
       x: float
       y: float
       scale: list[float]  # Add scale metadata
   ```

6. Update frontend/client code to expect dict format

**Impact**: BREAKING CHANGE - requires client updates

**Validation**:
- Run integration tests
- Test WebSocket data flow
- Verify debug page renders correctly

---

### PHASE 2: HIGH PRIORITY - Vision Module 4K Integration

**Goal**: Convert vision module to output 4K canonical coordinates

**Tasks**:

1. **Add resolution conversion to detector_adapter.py**:
   ```python
   from core.resolution_converter import ResolutionConverter
   from core.coordinates import Vector2D

   def yolo_to_ball(...):
       # Existing YOLO→pixel conversion
       x_px, y_px = bbox_to_center(bbox, image_shape)

       # NEW: Convert to 4K canonical
       source_resolution = (image_shape[1], image_shape[0])
       x_4k, y_4k = ResolutionConverter.scale_to_4k(
           x_px, y_px, source_resolution
       )

       # Create Ball with 4K position
       return Ball(
           position=(x_4k, y_4k),
           coordinate_space="4k_canonical",
           source_resolution=source_resolution,
           ...
       )
   ```

2. **Create resolution-aware detection config**:
   ```python
   # vision/detection/balls.py
   from core.constants_4k import BALL_RADIUS_4K
   from core.resolution_converter import ResolutionConverter

   class BallDetectionConfig:
       def __init__(self, camera_resolution: tuple[int, int]):
           # Scale 4K values to camera resolution
           scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(
               camera_resolution
           )
           self.min_radius = int(BALL_RADIUS_4K / scale_x * 0.7)
           self.max_radius = int(BALL_RADIUS_4K / scale_x * 1.3)
           self.expected_radius = int(BALL_RADIUS_4K / scale_x)
   ```

3. **Update vision/models.py** (optional - could keep tuples):
   ```python
   from core.coordinates import Vector2D

   @dataclass
   class Ball:
       position: Vector2D  # Changed from tuple[float, float]
       radius: float
       coordinate_space: str = "4k_canonical"  # Changed from "pixel"
       source_resolution: Optional[tuple[int, int]] = None
   ```

**Impact**: BREAKING CHANGE - affects vision→core integration

**Validation**:
- Run `test_vision_integration_4k_conversion.py`
- Verify ball detection accuracy maintained
- Test with multiple camera resolutions (1080p, 720p, 4K)

---

### PHASE 3: MEDIUM PRIORITY - Cleanup & Documentation

**Tasks**:
1. Remove deprecated methods from `integration_service_conversion_helpers.py`
2. Fix validation threshold unit mismatches
3. Update API documentation with coordinate format requirements
4. Add migration guide for clients
5. Create coordinate system architecture diagram
6. Document coordinate space metadata requirements

---

### PHASE 4: LOW PRIORITY - Frontend Updates

**Tasks**:
1. Update frontend to handle dict position format
2. Add coordinate space awareness to rendering
3. Support multi-resolution display
4. Add coordinate system debugging tools

---

## VERIFICATION CHECKLIST

After implementing fixes, verify:

- [ ] WebSocket broadcaster accepts game state data without warnings
- [ ] Debug page renders correctly with scale metadata
- [ ] Ball positions consistent across all API endpoints
- [ ] Vision detection outputs 4K canonical coordinates
- [ ] All tests pass (especially `test_vision_integration_4k_conversion.py`)
- [ ] No coordinate space validation warnings in logs
- [ ] Frontend renders correctly at multiple display resolutions
- [ ] Trajectory calculations work with 4K coordinates
- [ ] Collision detection accurate with 4K ball positions

---

## FILES REQUIRING CHANGES

### PHASE 1 (Critical - API Module):
1. `/Users/jchadwick/code/billiards-trainer/backend/api/models/responses.py`
2. `/Users/jchadwick/code/billiards-trainer/backend/api/models/websocket.py`
3. `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/schemas.py`
4. `/Users/jchadwick/code/billiards-trainer/backend/api/routes/game.py`
5. `/Users/jchadwick/code/billiards-trainer/backend/api/models/vision_models.py`

### PHASE 2 (High Priority - Vision Module):
6. `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/detector_adapter.py`
7. `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/balls.py`
8. `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/cue.py`
9. `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/table.py`
10. `/Users/jchadwick/code/billiards-trainer/backend/vision/models.py` (optional)

### PHASE 3 (Cleanup):
11. `/Users/jchadwick/code/billiards-trainer/backend/integration_service_conversion_helpers.py`
12. Documentation files

---

## CONCLUSION

The billiards trainer codebase has made **significant progress** toward 4K coordinate standardization:

**✅ COMPLETE**:
- Core module (100% migrated)
- Tests (comprehensive validation)
- Video/streaming (correctly isolated)
- Integration service (proper 4K conversion)

**❌ CRITICAL ISSUES**:
- API module has broken coordinate data flow
- Format inconsistency between converters, routes, and broadcaster
- Scale metadata stripped in route conversions

**⚠️ NEEDS MIGRATION**:
- Vision module operates in camera pixels (not 4K)
- Hardcoded pixel values not resolution-aware
- No Vector2D usage in vision

**Next Steps**: Execute Phase 1 fixes immediately to restore API coordinate data flow, then proceed with Phase 2 vision migration.
