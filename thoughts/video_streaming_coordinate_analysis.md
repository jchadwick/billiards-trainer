# Video/Streaming/Integration Service Coordinate Analysis

**Date:** 2025-10-21
**Scope:** backend/video/, backend/streaming/, backend/integration_service*.py
**Status:** Complete and thorough analysis

## Executive Summary

All three modules (video, streaming, integration_service) are **NOT** using the new 4K pixel-based coordinate system directly. However:

1. **backend/video/** - Pure frame delivery, NO coordinate handling (CORRECT)
2. **backend/streaming/** - Uses pixel coordinates, NO coordinate conversion (CORRECT for module purpose)
3. **backend/integration_service*.py** - **CORRECTLY IMPLEMENTS** 4K conversion in integration_service_conversion_helpers.py

## Detailed Analysis by Module

---

## 1. backend/video/ Module

**Purpose:** Standalone video capture process that writes frames to shared memory for Vision and API consumption.

### Files Analyzed:
- `/Users/jchadwick/code/billiards-trainer/backend/video/process.py`
- `/Users/jchadwick/code/billiards-trainer/backend/video/ipc/shared_memory.py`
- `/Users/jchadwick/code/billiards-trainer/backend/video/__main__.py`
- `/Users/jchadwick/code/billiards-trainer/backend/video/__init__.py`

### Coordinate/Vector References: NONE

**Status: ✅ CORRECT - No coordinate handling needed**

### What This Module Does:
- Captures video frames from camera/file
- Applies distortion correction (undistortion maps)
- Writes raw frames to shared memory (triple-buffered)
- **Does NOT process coordinates** - only pixel data

### Frame Dimensions Referenced:
1. **Line 261** - Resolution config: `[1920, 1080]` (default camera resolution)
2. **Lines 146-198** - `_initialize_undistortion_maps(width: int, height: int)` - uses actual frame dimensions
3. **Lines 307-340** - `_initialize_ipc_writer(width: int, height: int)` - creates shared memory with actual frame size
4. **Lines 531-533** - Extracts frame dimensions from numpy array: `height, width = frame.shape[:2]`

### Shared Memory Layout (shared_memory.py):
```
Header Block (4KB):
- Frame width (4 bytes): Frame width in pixels
- Frame height (4 bytes): Frame height in pixels
- Bytes per frame (8 bytes): Size of each frame buffer in bytes
```

### Key Observations:
- **No Vector2D usage** - Correctly! This module only handles raw pixel data
- **No coordinate conversions** - Correctly! Not responsible for coordinate systems
- **Frame dimensions are metadata** - Used for buffer sizing and undistortion map creation
- **Resolution is source-dependent** - Actual camera/video resolution detected at runtime

### Architecture:
```
Camera/Video → Undistortion → Shared Memory (BGR24 format)
                                    ↓
                        Vision Module reads frames
                        API streams frames
```

**Verdict:** This module is **correctly isolated** from coordinate system concerns. It should remain pixel-agnostic and only handle raw frame delivery.

---

## 2. backend/streaming/ Module

**Purpose:** Enhanced camera module with fisheye correction and preprocessing for web streaming.

### Files Analyzed:
- `/Users/jchadwick/code/billiards-trainer/backend/streaming/enhanced_camera_module.py`
- `/Users/jchadwick/code/billiards-trainer/backend/streaming/__init__.py`

### Coordinate/Vector References: LIMITED (Resolution and cropping only)

**Status: ✅ CORRECT for streaming purpose - Not responsible for coordinate conversion**

### What This Module Does:
- Captures camera frames (similar to video module)
- Applies fisheye correction (undistortion)
- Applies image preprocessing (CLAHE, brightness, contrast)
- Crops to table boundaries (optional)
- Encodes frames as JPEG for web streaming
- **Does NOT process game coordinates** - only image processing

### Resolution/Dimension References:

1. **Lines 41, 102-104** - Resolution configuration
   ```python
   resolution: Optional[tuple[int, int]]  # Auto-detect if None
   resolution = config.get("vision.camera.resolution")
   if resolution and isinstance(resolution, list) and len(resolution) == 2:
       resolution = tuple(resolution)
   ```

2. **Lines 226-272** - `_detect_actual_resolution()` - Auto-detects camera capabilities
   ```python
   actual_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   actual_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   ```

3. **Line 326** - Undistortion map creation
   ```python
   h, w = self.config.resolution[1], self.config.resolution[0]
   ```

4. **Lines 520-571** - `_crop_to_table()` - Table felt detection and cropping
   ```python
   x, y, w, h = cv2.boundingRect(largest_contour)
   return frame[y : y + h, x : x + w]
   ```

5. **Lines 592-627** - `get_frame_for_streaming()` - Resize for bandwidth optimization
   ```python
   max_width: Optional[int] = None,
   max_height: Optional[int] = None,
   ```

### HSV Color Ranges for Table Detection:
```python
table_crop_hsv_lower: tuple[int, int, int]  # Default: (35, 40, 40) - green felt
table_crop_hsv_upper: tuple[int, int, int]  # Default: (85, 255, 255)
```

### Key Observations:
- **No Vector2D usage** - Correctly! This is image processing, not game state
- **No coordinate conversions** - Correctly! Not responsible for physics coordinates
- **Resolution handling** - For camera configuration and image processing only
- **Table crop** - Pixel-based bounding box, not game coordinates
- **Frame callback** - Broadcasts processed frames but doesn't handle coordinate metadata

### Architecture:
```
Camera → Fisheye Correction → Table Crop → Preprocessing → JPEG Encode
                                                                ↓
                                                    WebSocket Stream
```

**Verdict:** This module is **correctly isolated** from the 4K coordinate system. It handles image processing in whatever resolution the camera provides. Coordinate conversion is the responsibility of Vision/Integration modules.

---

## 3. backend/integration_service*.py Module

**Purpose:** Connects Vision → Core → Broadcast data flow, converting Vision detections to Core game state.

### Files Analyzed:
- `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`
- `/Users/jchadwick/code/billiards-trainer/backend/integration_service_conversion_helpers.py`

### Coordinate/Vector References: EXTENSIVE (4K conversion implemented)

**Status: ✅ CORRECT - Properly implements 4K pixel coordinate system**

---

### 3A. integration_service.py

**Coordinate System Used:** Converts Vision pixel detections → Dictionary format → Core 4K via conversion helpers

### Position Dictionary References (Vision → Broadcast Format):

1. **Lines 559-560** - Ball positions converted to dict format:
   ```python
   "position": {"x": ball.position[0], "y": ball.position[1]},
   "velocity": {"x": ball.velocity[0], "y": ball.velocity[1]},
   ```

2. **Line 576** - Cue tip position:
   ```python
   "tip_position": {"x": cue.tip_position[0], "y": cue.tip_position[1]},
   ```

3. **Lines 588, 595** - Table corners and pockets:
   ```python
   {"x": float(corner[0]), "y": float(corner[1])}
   "position": {"x": pocket.position[0], "y": pocket.position[1]},
   ```

4. **Lines 1106-1117** - **CRITICAL: Coordinate format conversion for WebSocket broadcast**
   ```python
   # The broadcaster expects positions as [x, y] but asdict() converts Vector2D to {'x': ..., 'y': ...}
   balls_converted = []
   for ball in balls:
       ball_copy = ball.copy()
       position = ball_copy.get("position")
       if isinstance(position, dict) and "x" in position and "y" in position:
           ball_copy["position"] = [position["x"], position["y"]]
       # Also convert velocity if present
       velocity = ball_copy.get("velocity")
       if isinstance(velocity, dict) and "x" in velocity and "y" in velocity:
           ball_copy["velocity"] = [velocity["x"], velocity["y"]]
       balls_converted.append(ball_copy)
   ```
   **Issue:** This converts Core's Vector2D (dict format from asdict()) back to array format for legacy broadcaster compatibility.

5. **Lines 1229-1230, 1242** - Trajectory broadcast format:
   ```python
   "start": [start_point.position.x, start_point.position.y],
   "end": [end_point.position.x, end_point.position.y],
   "position": [collision.position.x, collision.position.y],
   ```

### Key Methods That Use Conversion Helpers:

1. **Lines 427-441** - `vision_cue_to_cue_state()` - Delegates to StateConversionHelpers
2. **Lines 443-463** - `vision_ball_to_ball_state()` - Delegates to StateConversionHelpers
3. **Lines 732-761** - Trajectory calculation uses conversion helpers:
   ```python
   cue_state = self.state_converter.vision_cue_to_cue_state(detection.cue, timestamp=detection.timestamp)
   ball_state = self.state_converter.vision_ball_to_ball_state(target_ball, is_target=True, ...)
   ```

### Deprecated Methods (Still Present):
- **Lines 465-479** - `_create_cue_state()` - DEPRECATED, delegates to conversion helper
- **Lines 481-496** - `_create_ball_state()` - DEPRECATED, delegates to conversion helper

**Verdict:** This module correctly delegates coordinate conversion to StateConversionHelpers.

---

### 3B. integration_service_conversion_helpers.py

**Coordinate System Used:** 4K pixel-based canonical system (3840×2160)

**Status: ✅ FULLY MIGRATED TO 4K PIXEL SYSTEM**

### Documentation Header (Lines 11-27):
```python
"""
COORDINATE SYSTEM UPDATE (2025-10-21):
--------------------------------------
This module has been updated to use the new 4K pixel-based coordinate system.

OLD APPROACH (DEPRECATED):
- Vision detections in pixels → meters conversion using pixels_per_meter calibration
- Required table corner detection and homography transforms
- Complex CoordinateConverter with perspective transforms

NEW APPROACH (CURRENT):
- Vision detections in pixels → 4K canonical pixels using resolution scale
- Automatic scale calculation: scale = 4K_resolution / source_resolution
- Simple Vector2D.from_resolution() → to_4k_canonical() conversion
- No calibration required - pure pixel-based coordinates throughout

All positions, velocities, and sizes are now in 4K canonical pixel space (3840×2160).
The scale metadata is preserved in Vector2D for future coordinate transformations.
"""
```

### Core Conversion Implementation:

#### Ball Conversion (Lines 274-454) - `vision_ball_to_ball_state()`:

**Coordinate Space Tracking:**
```python
# Line 313: Track source coordinate space
source_space = ball.coordinate_space if hasattr(ball, "coordinate_space") else "pixel"

if source_space == "pixel":
    # Lines 320-322: Get source resolution from ball metadata
    source_resolution = ball.source_resolution if hasattr(ball, "source_resolution") and ball.source_resolution else self.camera_resolution

    # Lines 325-329: Use Vector2D.from_resolution() with automatic scale calculation
    position_with_scale = Vector2D.from_resolution(
        ball.position[0],
        ball.position[1],
        source_resolution
    )

    # Lines 331-332: Convert to 4K canonical (scale=[1.0, 1.0])
    position_4k = position_with_scale.to_4k_canonical()

    # Lines 335-341: Convert velocity using the same scale
    velocity_with_scale = Vector2D.from_resolution(
        ball.velocity[0],
        ball.velocity[1],
        source_resolution
    )
    velocity_4k = velocity_with_scale.to_4k_canonical()

    # Lines 343-345: Convert radius using the scale factor
    scale_x = 3840 / source_resolution[0]
    radius_4k = ball.radius * scale_x if ball.radius > 0 else 36.0  # Default ball radius in 4K pixels
```

**Validation (4K pixel bounds):**
```python
# Lines 371-381: Validate position is within 4K frame (0-3840, 0-2160)
if position_4k.x < 0 or position_4k.x > 3840 or position_4k.y < 0 or position_4k.y > 2160:
    logger.warning(f"Position ({position_4k.x:.1f}, {position_4k.y:.1f}) outside 4K frame, clamping")
    position_4k = Vector2D(
        max(0, min(3840, position_4k.x)),
        max(0, min(2160, position_4k.y)),
        scale=(1.0, 1.0)
    )

# Lines 383-395: Validate velocity (max ~12600 px/s ≈ 10 m/s at 1260 px/m)
velocity_mag = (velocity_4k.x ** 2 + velocity_4k.y ** 2) ** 0.5
max_velocity_px = 12600.0  # Roughly 10 m/s in 4K pixels/second
```

**BallState Creation:**
```python
# Lines 397-412: Create BallState using from_4k() factory method
ball_state = BallState.from_4k(
    id=ball_id,
    x=position_4k.x,
    y=position_4k.y,
    vx=velocity_4k.x,
    vy=velocity_4k.y,
    radius=radius_4k,
    mass=0.17,  # Standard pool ball mass in kg
    is_cue_ball=(ball.ball_type.value == "cue" if ball.ball_type else False),
    is_pocketed=False,
    number=ball.number,
    confidence=ball.confidence,
    last_update=timestamp if timestamp is not None else time.time(),
)
```

#### Cue Conversion (Lines 456-595) - `vision_cue_to_cue_state()`:

**Same pattern as ball conversion:**
```python
# Lines 509-524: Convert tip position from camera pixels to 4K
if source_space == "pixel":
    source_resolution = detected_cue.source_resolution if hasattr(detected_cue, "source_resolution") and detected_cue.source_resolution else self.camera_resolution

    tip_position_with_scale = Vector2D.from_resolution(
        detected_cue.tip_position[0],
        detected_cue.tip_position[1],
        source_resolution
    )
    tip_position_4k = tip_position_with_scale.to_4k_canonical()

    # Lines 527-528: Convert cue length
    scale_x = 3840 / source_resolution[0]
    length_4k = detected_cue.length * scale_x if hasattr(detected_cue, "length") and detected_cue.length > 0 else 1851.0
    # Default cue length in 4K pixels (~1.47m * 1260px/m)
```

**CueState Creation:**
```python
# Lines 565-577: Create cue state using from_4k() factory method
cue_state = CueState.from_4k(
    angle=detected_cue.angle,
    estimated_force=estimated_force,
    tip_x=tip_position_4k.x,
    tip_y=tip_position_4k.y,
    elevation=0.0,
    length=length_4k,
    is_visible=True,
    confidence=detected_cue.confidence,
    last_update=timestamp if timestamp is not None else time.time(),
)
```

### Deprecated Legacy Methods (Still Present):

1. **Lines 238-254** - `_pixels_to_meters()` - DEPRECATED
   ```python
   """DEPRECATED: This method is obsolete with the new 4K pixel-based coordinate system.
   Use Vector2D.from_resolution() and to_4k_canonical() instead."""
   ```

2. **Lines 256-272** - `_pixels_per_second_to_meters_per_second()` - DEPRECATED

3. **Lines 140-204** - `_create_coordinate_converter()` - DEPRECATED
   ```python
   # TODO: Remove CoordinateConverter instantiation - no longer needed
   ```

4. **Lines 104-111** - Coordinate converter attributes
   ```python
   self.camera_resolution: Resolution = (camera_width, camera_height)
   # TODO: Remove coordinate_converter - no longer needed with 4K pixel system
   # self.coordinate_converter: Optional[CoordinateConverter] = None
   ```

### Configuration and Validation:

**Table Dimensions (Lines 62-78):**
```python
# Default dimensions in meters for legacy support
# TODO: Migrate to 4K pixel dimensions (3200×1600) once meter system is fully deprecated
self.table_width_meters = 2.54  # 9 feet in meters
self.table_height_meters = 1.27  # 4.5 feet in meters
```

**Validation Thresholds (Lines 74-99):**
```python
self.max_ball_velocity = config.get("integration.max_ball_velocity_m_per_s", 10.0)
self.max_position_x = config.get("integration.max_position_x", self.table_width_meters + 0.5)
self.max_position_y = config.get("integration.max_position_y", self.table_height_meters + 0.5)
self.min_ball_confidence = config.get("integration.min_ball_confidence", 0.1)
self.min_cue_confidence = config.get("integration.min_cue_confidence", 0.05)
```

**NOTE:** Validation thresholds are still in meters but conversion logic uses 4K pixels. This is a minor inconsistency.

### Conversion Statistics (Lines 675-693):
```python
def get_conversion_stats(self) -> dict:
    return {
        "ball_conversions": self._ball_conversion_count,
        "cue_conversions": self._cue_conversion_count,
        "coordinate_conversions": self._coordinate_conversions,
        "validation_warnings": self._validation_warnings,
        "ball_warning_rate": ...,
        "coordinate_system": "4K_canonical",
        "resolution_based_conversion": True,
    }
```

**Verdict:** This module is **fully migrated to 4K pixel system** with proper Vector2D usage and scale tracking.

---

## Summary of Coordinate Systems by Module

| Module | Coordinate System | Vector2D Usage | Status |
|--------|------------------|----------------|---------|
| **video/** | None (raw frames only) | No | ✅ Correct |
| **streaming/** | Camera pixels (processing) | No | ✅ Correct |
| **integration_service.py** | Vision pixels → Core via helpers | Indirect | ✅ Correct |
| **integration_service_conversion_helpers.py** | Camera pixels → 4K canonical | Yes (proper usage) | ✅ Migrated |

---

## Issues Found

### 1. ⚠️ Coordinate Format Inconsistency (integration_service.py)

**Location:** Lines 1106-1117

**Issue:** Core's `asdict(GameState)` converts Vector2D to `{"x": ..., "y": ...}` but broadcaster expects `[x, y]` arrays.

**Code:**
```python
# The broadcaster expects positions as [x, y] but asdict() converts Vector2D to {'x': ..., 'y': ...}
balls_converted = []
for ball in balls:
    ball_copy = ball.copy()
    position = ball_copy.get("position")
    if isinstance(position, dict) and "x" in position and "y" in position:
        ball_copy["position"] = [position["x"], position["y"]]
    velocity = ball_copy.get("velocity")
    if isinstance(velocity, dict) and "x" in velocity and "y" in velocity:
        ball_copy["velocity"] = [velocity["x"], velocity["y"]]
    balls_converted.append(ball_copy)
```

**Impact:** Manual conversion required before WebSocket broadcast. Should standardize on one format.

**Recommendation:** Either:
- Update broadcaster to accept dict format `{"x": ..., "y": ...}`
- Or create a standardized serialization method for Vector2D

---

### 2. ⚠️ Legacy Code Not Removed (integration_service_conversion_helpers.py)

**Locations:**
- Lines 104-111: Commented-out coordinate_converter attributes
- Lines 140-204: `_create_coordinate_converter()` method (empty stub)
- Lines 238-254: `_pixels_to_meters()` - DEPRECATED
- Lines 256-272: `_pixels_per_second_to_meters_per_second()` - DEPRECATED

**Issue:** Dead code and deprecated methods still present.

**Impact:** Code clutter, potential confusion for future developers.

**Recommendation:** Remove all deprecated methods and commented-out code in next cleanup pass.

---

### 3. ⚠️ Validation Threshold Units Mismatch (integration_service_conversion_helpers.py)

**Location:** Lines 74-78, 91-93

**Issue:** Validation thresholds configured in meters but conversion logic uses 4K pixels.

**Code:**
```python
self.max_position_x = config.get("integration.max_position_x", self.table_width_meters + 0.5)  # meters
self.max_position_y = config.get("integration.max_position_y", self.table_height_meters + 0.5)  # meters

# But validation is done in 4K pixels (lines 371-381):
if position_4k.x < 0 or position_4k.x > 3840 or position_4k.y < 0 or position_4k.y > 2160:
```

**Impact:** `self.max_position_x/y` are not actually used for 4K validation (hardcoded 3840/2160 instead). Dead configuration.

**Recommendation:** Either remove unused meter-based thresholds or convert them to 4K pixel values.

---

### 4. ℹ️ No Vector2D Usage in video/streaming (Expected)

**Status:** This is actually CORRECT behavior.

**Rationale:**
- **video/** should remain coordinate-agnostic (only handles raw pixel data)
- **streaming/** is for image processing, not game coordinates
- Coordinate conversion is the responsibility of Vision/Integration modules

**No action needed.**

---

## Deprecated Patterns Found

### None! (Array positions are not used here)

These modules do NOT use the deprecated `[x, y]` array format for physics/game coordinates:
- `backend/video/` - No coordinates at all ✅
- `backend/streaming/` - No game coordinates ✅
- `backend/integration_service_conversion_helpers.py` - Uses Vector2D properly ✅

**However:** `integration_service.py` DOES convert Vector2D dicts to arrays for WebSocket broadcast (see Issue #1).

---

## Coordinate Space Flow

```
Camera/Video File (e.g., 1920×1080)
          ↓
    [video module - no coordinate handling]
          ↓
  Shared Memory (raw BGR24 frames)
          ↓
    Vision Module (detections in camera pixels)
          ↓
    [integration_service_conversion_helpers.py]
          ↓
  Vector2D.from_resolution(x, y, source_resolution)
          ↓
  Vector2D.to_4k_canonical() → 4K pixels (3840×2160, scale=[1.0, 1.0])
          ↓
  BallState/CueState.from_4k(...) → Core module
          ↓
  Physics calculations (4K pixel space)
          ↓
  [integration_service.py - format conversion]
          ↓
  WebSocket broadcast (array format [x, y])
```

---

## Hardcoded Values

### 4K Resolution Constants (integration_service_conversion_helpers.py):
- **Line 26, 344, 354, 371, 377, 527, 559**: `3840` (4K width)
- **Line 26, 371, 378, 560**: `2160` (4K height)
- **Line 64**: `3200px × 1600px` (table dimensions in 4K - commented)
- **Line 345**: `36.0` (default ball radius in 4K pixels)
- **Line 386**: `12600.0` (max velocity in 4K pixels/second)
- **Line 528**: `1851.0` (default cue length in 4K pixels)

### Default Resolutions (video/streaming):
- **video/process.py Line 261**: `[1920, 1080]` (default camera resolution)
- **streaming/enhanced_camera_module.py Line 98**: `1920 × 1080` (default camera resolution)

### Table Dimensions (integration_service_conversion_helpers.py):
- **Line 69**: `2.54` (9-foot table width in meters)
- **Line 70**: `1.27` (9-foot table height in meters)

**Note:** These are configuration defaults and can be overridden.

---

## Recommendations

### High Priority:

1. **Standardize coordinate serialization format** (Issue #1)
   - Choose either dict `{"x": ..., "y": ...}` or array `[x, y]` format
   - Update broadcaster or create Vector2D serialization helper
   - Eliminate manual conversion in `_on_state_updated_async()`

2. **Remove deprecated code** (Issue #2)
   - Delete `_create_coordinate_converter()` stub
   - Remove `_pixels_to_meters()` and `_pixels_per_second_to_meters_per_second()`
   - Clean up commented-out coordinate_converter references

### Medium Priority:

3. **Update validation thresholds to 4K pixels** (Issue #3)
   - Convert meter-based max_position_x/y to 4K pixel values
   - Or remove if not used

4. **Document coordinate space metadata**
   - Add `coordinate_space` and `source_resolution` to Vision Ball/CueStick models
   - Ensure metadata is preserved through conversion pipeline

### Low Priority:

5. **Consider making 4K constants configurable**
   - Allow different canonical resolutions (e.g., 1080p, 8K)
   - Move hardcoded 3840×2160 to configuration

---

## Test Coverage Needed

1. **Coordinate conversion accuracy**
   - Test conversion from various source resolutions (720p, 1080p, 4K) → 4K canonical
   - Verify scale calculations
   - Test boundary clamping

2. **Broadcast format compatibility**
   - Test Vector2D dict → array conversion
   - Verify WebSocket clients can parse coordinates

3. **Edge cases**
   - Positions outside frame bounds
   - Extreme velocities
   - Missing source_resolution metadata

---

## File Summary Table

| File | Lines | Vector2D | Position Arrays | Coordinates | Status |
|------|-------|----------|-----------------|-------------|--------|
| video/process.py | 565 | No | No | None (frame dimensions only) | ✅ |
| video/ipc/shared_memory.py | 810 | No | No | None (frame metadata) | ✅ |
| video/__main__.py | 151 | No | No | None | ✅ |
| streaming/enhanced_camera_module.py | 754 | No | No | Image processing only | ✅ |
| integration_service.py | 1290 | Indirect | Yes (broadcast) | Vision pixels → helpers | ✅ |
| integration_service_conversion_helpers.py | 694 | Yes | No | Camera px → 4K canonical | ✅ |

**Legend:**
- ✅ = Correct for module's purpose
- ⚠️ = Minor issues (see Issues section)

---

## Conclusion

All three module groups are functioning correctly for their respective purposes:

1. **video/** - ✅ Correctly isolated from coordinate systems (raw frame delivery)
2. **streaming/** - ✅ Correctly handles image processing in camera pixel space
3. **integration_service*.py** - ✅ Properly implements 4K pixel coordinate conversion

The main issues are legacy code cleanup and broadcast format standardization, not fundamental coordinate system problems.

**Overall Assessment:** 🟢 **GOOD** - System is properly migrated to 4K pixel coordinates where needed, with clean separation of concerns.

---

## Additional Notes

- These modules do NOT directly interact with the deprecated array position format `[x, y]` for physics calculations
- Coordinate conversion is centralized in `integration_service_conversion_helpers.py`
- Vision module detections → Integration helpers → Core 4K physics → Broadcast (format conversion)
- No hardcoded table dimensions in video/streaming (correctly!)

---

**Generated:** 2025-10-21 by Claude Code
**Analysis Scope:** Very thorough search with multiple pattern matching strategies
**Files Analyzed:** 8 files across 3 module groups
