# Coordinate Space Tracking Implementation

## Overview
Updated the integration service to properly track and convert coordinate spaces when transforming Vision module detections to Core module state objects.

## Changes Made

### 1. Added Coordinate Space Conversion Infrastructure

#### File: `/Users/jchadwick/code/billiards-trainer/backend/integration_service_conversion_helpers.py`

**New Configuration Parameters:**
- `vision.calibration.pixels_per_meter`: Scaling factor for pixel-to-meter conversion
  - Default: 756.0 px/m (calculated as 1920px / 2.54m for standard 9ft table)
  - Configurable via config.json

**New Helper Methods:**
```python
def _pixels_to_meters(self, pixel_value: float) -> float
    """Convert pixel coordinates/distances to meters."""

def _pixels_per_second_to_meters_per_second(self, pixels_per_sec: float) -> float
    """Convert velocity from pixels/second to meters/second."""
```

**New Tracking Counter:**
- `self._coordinate_conversions`: Tracks total coordinate space conversions for debugging

### 2. Ball State Conversion with Coordinate Tracking

#### Updated Method: `vision_ball_to_ball_state()`

**Key Changes:**
1. **Coordinate Space Detection:**
   - Checks `ball.coordinate_space` attribute (defaults to "pixel")
   - Logs source coordinate space for debugging

2. **Automatic Conversion:**
   - If source is "pixel": Converts position, velocity, and radius to meters
   - If source is "world"/"meters": Uses values directly (no conversion)

3. **Conversion Process:**
   ```python
   # Position conversion
   position_m = Vector2D(
       self._pixels_to_meters(ball.position[0]),
       self._pixels_to_meters(ball.position[1])
   )

   # Velocity conversion
   velocity_m = Vector2D(
       self._pixels_per_second_to_meters_per_second(ball.velocity[0]),
       self._pixels_per_second_to_meters_per_second(ball.velocity[1])
   )

   # Radius conversion
   radius_m = self._pixels_to_meters(ball.radius)
   ```

4. **Validation Order (IMPORTANT):**
   - Conversion happens BEFORE validation
   - Position/velocity validation uses meter values, not pixel values
   - This ensures validation thresholds are applied correctly

5. **Enhanced Logging:**
   - Periodic logging (every 100 conversions) shows:
     - Source coordinate space
     - Position in meters (with 3 decimal precision)
     - Velocity in m/s (with 3 decimal precision)
     - Scaling factor used

### 3. Cue State Conversion with Coordinate Tracking

#### Updated Method: `vision_cue_to_cue_state()`

**Key Changes:**
1. **Coordinate Space Detection:**
   - Checks `detected_cue.coordinate_space` attribute (defaults to "pixel")
   - Logs source coordinate space for debugging

2. **Automatic Conversion:**
   - If source is "pixel": Converts tip position and length to meters
   - If source is "world"/"meters": Uses values directly

3. **Conversion Process:**
   ```python
   # Tip position conversion
   tip_position_m = Vector2D(
       self._pixels_to_meters(detected_cue.tip_position[0]),
       self._pixels_to_meters(detected_cue.tip_position[1])
   )

   # Length conversion
   length_m = self._pixels_to_meters(detected_cue.length)
   ```

4. **Enhanced Logging:**
   - Periodic logging (every 100 conversions) shows:
     - Source coordinate space
     - Tip position in meters
     - Length in meters
     - Scaling factor used

### 4. Enhanced Statistics Tracking

#### Updated Method: `get_conversion_stats()`

**New Statistics:**
```python
{
    "ball_conversions": int,
    "cue_conversions": int,
    "coordinate_conversions": int,  # NEW
    "validation_warnings": int,
    "ball_warning_rate": float,
    "pixels_per_meter": float,      # NEW
    "coordinate_conversion_enabled": True  # NEW
}
```

## Coordinate Space Metadata

### Vision Module (Source)
- **Coordinate Space:** Pixel coordinates
- **Origin:** Top-left corner of camera frame
- **Units:** Pixels
- **Metadata Fields:**
  - `Ball.coordinate_space`: "pixel" (default)
  - `CueStick.coordinate_space`: "pixel" (default)
  - `Ball.source_resolution`: (width, height) of source frame

### Core Module (Target)
- **Coordinate Space:** World/meter coordinates
- **Origin:** Table coordinate system
- **Units:** Meters (SI)
- **Expected Values:**
  - Position: Typically 0-2.54m (x) × 0-1.27m (y) for 9ft table
  - Velocity: Meters per second (m/s)
  - Radius: ~0.028575m (standard pool ball)

## Configuration

### Required Config Values

Add to `config.json`:
```json
{
  "vision": {
    "calibration": {
      "pixels_per_meter": 756.0
    }
  }
}
```

### Calculating pixels_per_meter

For accurate conversions, calculate based on your camera setup:

```
pixels_per_meter = frame_width_pixels / table_width_meters
```

Example for 9ft table (2.54m wide) at 1920px resolution:
```
pixels_per_meter = 1920 / 2.54 ≈ 756 px/m
```

## Logging and Debugging

### Coordinate Conversion Logging

**Every 100 conversions:**
```
DEBUG: Coordinate conversion #100:
  pixel(960.0,540.0) -> meters(1.270,0.714),
  scaling=756.0px/m
```

**Ball Conversion Logging (every 100):**
```
DEBUG: Ball conversion #100:
  id=ball_123,
  source_space=pixel,
  pos_meters=(1.270m,0.714m),
  vel_m/s=(0.132,0.066),
  confidence=0.95,
  is_cue=False
```

**Cue Conversion Logging (every 100):**
```
DEBUG: Cue conversion #100:
  source_space=pixel,
  angle=45.00deg,
  force=5.00N,
  tip_meters=(1.270m,0.714m),
  length_meters=1.470m,
  confidence=0.92
```

**Statistics Summary (every 100 conversions):**
```
INFO: Ball conversion stats:
  100 conversions,
  100 coordinate conversions,
  5 warnings,
  scaling=756.0px/m
```

## Benefits

1. **Proper Physics Calculations:**
   - Core physics engine receives correct SI units (meters, m/s)
   - Trajectory calculations work with real-world distances
   - Forces and velocities use proper scaling

2. **Debugging Support:**
   - Clear logging shows coordinate space at each step
   - Conversion statistics help identify issues
   - Easy to verify scaling factors are correct

3. **Flexibility:**
   - Supports both pixel and world coordinate inputs
   - Automatic detection and conversion
   - Configurable scaling factors

4. **Validation:**
   - Position/velocity validation happens in meters
   - Proper bounds checking for table dimensions
   - Physics validator works with correct units

## Testing Recommendations

1. **Verify Scaling Factor:**
   - Measure actual table width
   - Calculate expected pixels_per_meter
   - Check conversion logs match expectations

2. **Visual Verification:**
   - Place ball at known position (e.g., table center)
   - Check converted position matches expected meters
   - Verify trajectory calculations align with visual feedback

3. **Monitor Statistics:**
   - Check `get_conversion_stats()` output
   - Verify coordinate_conversions counter increments
   - Look for validation warnings

## Future Enhancements

1. **Dynamic Calibration:**
   - Calculate pixels_per_meter from table calibration data
   - Use homography matrix for more accurate conversions
   - Support perspective corrections

2. **Multiple Coordinate Systems:**
   - Support table-relative coordinates
   - Add camera-relative coordinates
   - Implement coordinate system transformations

3. **Validation Enhancements:**
   - Add coordinate space type checking
   - Validate conversion consistency
   - Track conversion accuracy metrics
