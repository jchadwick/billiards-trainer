# Coordinate Space Bug Analysis

## Problem Statement

The user reports seeing ball coordinates like `Ball #1: (2974, 1256)` while the coordinate space shows `1920x1080`. This is physically impossible - the ball position is outside the camera frame bounds.

## Root Cause Analysis

### The Data Flow

1. **Vision Module** (`backend/vision/models.py`)
   - Outputs `Ball` objects with positions in **raw camera pixels**
   - Ball coordinates are in the full camera resolution (e.g., 3840x2160 for a 4K camera)
   - Has a `coordinate_space: str = "pixel"` field that should indicate this

2. **Integration Service** (`backend/integration_service_conversion_helpers.py`)
   - Converts vision `Ball` to core `BallState`
   - **INCORRECTLY assumes** vision outputs are in pixel space at a specific resolution
   - Simply divides pixel values by `pixels_per_meter` (default: 756.0)
   - Does NOT apply perspective transform
   - Does NOT check what resolution the pixels are in

3. **Core Module** (`backend/core/models.py`)
   - `BallState.position` is documented as being in **meters**
   - Comment on line 142: "position: Vector2D" (should be in meters)
   - Comment on line 405: "coordinate_space='world_meters'" (canonical)

4. **Broadcaster** (`backend/api/websocket/broadcaster.py`)
   - Receives ball positions from GameState
   - Assumes they are in **world meters** (line 291-294 comments)
   - Sends them to frontend as-is with metadata

### The Bug

The conversion in `integration_service_conversion_helpers.py` lines 125-136:

```python
# Convert coordinates from pixels to meters
# Vision Ball positions are in pixel coordinates, Core BallState expects meters
if source_space == "pixel":
    position_m = Vector2D(
        self._pixels_to_meters(ball.position[0]),  # ← WRONG!
        self._pixels_to_meters(ball.position[1])   # ← WRONG!
    )
```

This does: `meters = pixels / 756.0`

**What it SHOULD do:**
1. Apply perspective transform (if calibrated) to convert camera pixels to table-relative coordinates
2. THEN convert table pixels to meters using the calibrated `pixels_per_meter` ratio

### Example Calculation

Given ball at camera pixels `(2974, 1256)`:

**Current (broken) conversion:**
```
x_meters = 2974 / 756.0 = 3.934m
y_meters = 1256 / 756.0 = 1.661m
```

**Problem:** A standard 9-foot pool table is only `2.54m × 1.27m`!
The ball appears to be **1.4m beyond the table edge** and **0.4m past the far end**.

**What's actually happening:**
- The ball is at pixel position `(2974, 1256)` in the **raw camera view**
- This could be a 4K camera (3840×2160) where the table only occupies part of the frame
- Without perspective transform, we're treating the entire camera view as if it's the table
- The `pixels_per_meter` calibration ONLY applies to table-relative coordinates, not raw camera pixels

## The Missing Piece: Perspective Transform

The `CoordinateConverter` class (`backend/core/coordinate_converter.py`) line 339-340 shows the correct flow:

```python
# Apply perspective transform if available
if self.perspective_transform is not None:
    scaled_vector = self.perspective_transform.apply(scaled_vector)

# Convert pixels to meters
world_x = scaled_vector.x / self.pixels_per_meter
world_y = scaled_vector.y / self.pixels_per_meter
```

**The perspective transform:**
1. Takes raw camera pixels (e.g., 2974, 1256 in 3840×2160 frame)
2. Warps them to table-relative coordinates using the calibrated playing area corners
3. Outputs table-aligned pixels (e.g., 450, 230 in a ~600×300 table view)
4. THEN you can divide by `pixels_per_meter` to get meters

## Where the Conversion Should Happen

Looking at `integration_service_conversion_helpers.py`, the code has access to:
- `self.config` - which should contain calibration data
- Camera resolution information
- Table playing area corners (from config)

But it's NOT using the `CoordinateConverter` class that was specifically designed for this!

## Evidence from Config

Actual config values from `/Users/jchadwick/code/billiards-trainer/config.json`:

```json
{
  "vision": {
    "camera": {
      "resolution": [1920, 1080]
    },
    "calibration": {
      "pixels_per_meter": NOT SET  // Defaults to 756.0 in code
    }
  },
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

**Critical Finding:**
- Playing area corners define a region of **569px × 287px** in a 1920×1080 camera frame
- Using default `pixels_per_meter = 756.0`, this implies a table size of **0.75m × 0.38m**
- Actual 9-foot pool table should be **2.54m × 1.27m**

**Root Cause #1: Wrong pixels_per_meter calibration**
The `pixels_per_meter` value should be calculated from the TABLE REGION, not the full camera:
- Correct calculation: `pixels_per_meter = 569px / 2.54m = 224 px/m` (for width)
- Or: `pixels_per_meter = 287px / 1.27m = 226 px/m` (for height)
- Current (wrong): `pixels_per_meter = 756 px/m` (probably from full camera width: 1920/2.54)

**Root Cause #2: No perspective transform applied**
The playing area corners show the table is slightly trapezoid-shaped due to camera angle:
- Top edge: 569px wide (y≈40)
- Bottom edge: 564px wide (y≈324)
- This requires perspective transformation to properly convert to meters

## The Fix

### Immediate Fix

`integration_service_conversion_helpers.py` should:

1. **Use CoordinateConverter** instead of simple division:
```python
from backend.core.coordinate_converter import CoordinateConverter, Resolution

# In __init__:
self.coordinate_converter = CoordinateConverter(
    table_width_meters=config.get("table.width_meters", 2.54),
    table_height_meters=config.get("table.height_meters", 1.27),
    pixels_per_meter=self.pixels_per_meter,
    camera_resolution=Resolution(
        config.get("vision.camera.width", 1920),
        config.get("vision.camera.height", 1080)
    ),
    # TODO: Add perspective transform from calibration
)

# In vision_ball_to_ball_state:
position_m = self.coordinate_converter.camera_pixels_to_world_meters(
    Vector2D(ball.position[0], ball.position[1]),
    camera_resolution=Resolution(...)  # From vision detection metadata
)
```

2. **Load perspective transform from calibration:**
   - Table playing area corners need to be used to build perspective transform
   - This should come from camera calibration data
   - CoordinateConverter supports this via `perspective_transform` parameter

### Long-term Fix

1. **Vision module should include resolution metadata** in DetectionResult
   - Add `frame_size: tuple[int, int]` to Ball class (already exists!)
   - Ensure it's populated from the actual camera frame

2. **Calibration should output perspective transform matrix**
   - Not just corner points
   - Save homography matrix from calibration
   - Load it into CoordinateConverter

3. **Add validation**
   - Check that converted ball positions are within table bounds
   - Log warnings when positions are outside expected ranges
   - Already partially implemented in `_validate_position()` but limits are wrong

## What Coordinate Space are Balls Actually In?

Based on the evidence:

**Vision Output:** Camera pixels (e.g., 2974, 1256 in a 3840×2160 frame)

**After conversion_helpers:** INCORRECTLY "meters" (3.934m, 1.661m) - physically impossible

**What it should be:** World meters (e.g., 1.8m, 0.7m) after proper perspective transform + scaling

**Frontend receives:** Whatever the conversion_helpers outputs, labeled as "world_meters"

**Frontend displays:** Garbage coordinates because they're in the wrong space

## Testing the Fix

After fixing, verify:
1. Ball positions are within table bounds (0 to 2.54m in X, 0 to 1.27m in Y)
2. Ball #1 at camera pixel (2974, 1256) should convert to something like (1.8m, 0.65m)
3. Perspective transform is actually being applied (check debug logs)
4. `pixels_per_meter` calibration is correct for the TABLE view, not camera view

## Summary

### Exact Coordinate Space the Balls Are In

**When Vision Detects:** Raw camera pixels (e.g., 2974, 1256 in 1920×1080 frame, but could be higher if camera is 4K)

**After integration_service conversion:** INCORRECTLY "meters" (e.g., 3.934m, 1.661m) - calculated by simply dividing pixels by 756

**What reaches WebSocket/Frontend:** Bogus "world_meters" coordinates that are actually just `camera_pixels / 756`

**What it SHOULD be:** True world meters (e.g., 1.2m, 0.6m) after:
1. Applying perspective transform using playing area corners
2. Converting table-relative pixels to meters using CORRECT pixels_per_meter ratio (~224 px/m)

### Where the Conversion is Failing

**File:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service_conversion_helpers.py`
**Lines:** 125-136 in `vision_ball_to_ball_state()`

The code does:
```python
position_m = Vector2D(
    self._pixels_to_meters(ball.position[0]),  # ball.position[0] / 756.0
    self._pixels_to_meters(ball.position[1])   # ball.position[1] / 756.0
)
```

This is wrong because:
1. It operates on raw camera pixels without perspective transform
2. Uses wrong pixels_per_meter ratio (756 instead of ~224)
3. Ignores the playing area corners that define the table region

### What Needs to be Fixed

1. **Immediate fix:** Update `pixels_per_meter` in config to correct value:
   ```json
   "vision": {
     "calibration": {
       "pixels_per_meter": 224.0
     }
   }
   ```

2. **Proper fix:** Use `CoordinateConverter` with perspective transform:
   - Load playing area corners from config
   - Build perspective transformation matrix
   - Apply transform BEFORE dividing by pixels_per_meter
   - Use the existing `camera_pixels_to_world_meters()` method

3. **Validation fix:** Enable position validation to catch out-of-bounds balls:
   - Check x is in [0, 2.54]
   - Check y is in [0, 1.27]
   - Log warnings when balls appear outside table

### Expected Results After Fix

Ball at camera pixels `(2974, 1256)` should convert to approximately:
- **With only pixels_per_meter fix:** (13.3m, 5.6m) - still wrong, outside table
- **With perspective transform:** Should first transform to table-relative pixels like (450, 180)
- **Then to meters:** (450/224 = 2.0m, 180/224 = 0.8m) - valid table position

The frontend should then receive coordinates that make sense on a 2.54m × 1.27m table.

## References

- `/Users/jchadwick/code/billiards-trainer/backend/integration_service_conversion_helpers.py` lines 125-136
- `/Users/jchadwick/code/billiards-trainer/backend/core/coordinate_converter.py` lines 299-346
- `/Users/jchadwick/code/billiards-trainer/backend/vision/models.py` lines 63-103
- `/Users/jchadwick/code/billiards-trainer/backend/core/models.py` lines 134-238
- `/Users/jchadwick/code/billiards-trainer/config.json` lines 1350-1367
