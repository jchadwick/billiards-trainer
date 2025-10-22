# Vision Detection Resolution-Aware Threshold Fix

## Summary

Successfully migrated all vision detection modules to use resolution-aware pixel thresholds instead of hardcoded values. All pixel-based parameters are now automatically scaled based on camera resolution relative to the 4K canonical resolution (3840×2160).

## Problem Statement

Vision detection modules had hardcoded pixel values that were calibrated for specific resolutions (typically 1080p). These values would break or perform poorly at different resolutions:

```python
# Before: Hardcoded for 1080p
min_radius: int = 15  # pixels - assumes 1920×1080
max_radius: int = 26  # pixels
max_cue_length = 800  # pixels
max_distance_to_cue_ball = 40  # pixels
```

At 4K (3840×2160), these values would be too small. At 720p (1280×720), they would be too large.

## Solution Overview

All detection modules now:
1. Accept `camera_resolution` parameter in their constructors
2. Calculate scale factor from 4K canonical to camera resolution
3. Scale all pixel-based parameters automatically
4. Use 4K constants as the baseline for all measurements

## Files Modified

### 1. Ball Detection (`backend/vision/detection/balls.py`)

**Changes:**
- Added imports for `BALL_RADIUS_4K` and `ResolutionConverter`
- Updated `BallDetectionConfig` to accept `camera_resolution` parameter
- Modified `from_config()` classmethod to scale all pixel-based parameters:
  - `min_radius`, `max_radius`, `expected_radius` - Based on BALL_RADIUS_4K (36px in 4K)
  - `roi_margin` - Scaled from 4K default
- Updated `BallDetector.__init__()` to accept and store `camera_resolution`
- Updated `_detect_pockets_from_background()` to scale pocket detection parameters

**Scaled Parameters:**
- Ball radii: Scaled from BALL_RADIUS_4K = 36 pixels
  - min_radius = BALL_RADIUS_4K * scale * 0.7 (70% of expected)
  - max_radius = BALL_RADIUS_4K * scale * 1.3 (130% of expected)
  - expected_radius = BALL_RADIUS_4K * scale
- ROI margin: Scaled from 50px in 4K
- Pocket dimensions: Scaled from 2× ball radius in 4K

**Example:**
For 1080p (1920×1080):
- Scale factor: 0.5 (half of 4K)
- Ball radius: 36 * 0.5 = 18 pixels
- Min radius: 18 * 0.7 = 12.6 → 12 pixels
- Max radius: 18 * 1.3 = 23.4 → 23 pixels

### 2. Cue Detection (`backend/vision/detection/cue.py`)

**Changes:**
- Added imports for `BALL_RADIUS_4K`, `TABLE_WIDTH_4K`, and `ResolutionConverter`
- Updated `CueDetector.__init__()` to accept `camera_resolution` parameter
- Calculated scale factor and stored as `self.scale`
- Scaled all pixel-based parameters throughout the detector

**Scaled Parameters:**
- **Geometry:**
  - min_cue_length: 300px in 4K → scaled
  - max_cue_length: 1800px in 4K → scaled (based on physical cue length)
  - min_line_thickness: 6px in 4K → scaled
  - max_line_thickness: 50px in 4K → scaled
  - ball_radius: BALL_RADIUS_4K → scaled

- **Hough Transform:**
  - hough_min_line_length: 200px in 4K → scaled
  - hough_max_line_gap: 40px in 4K → scaled

- **Motion Analysis:**
  - velocity_threshold: 10px/frame → scaled
  - acceleration_threshold: 4px/frame² → scaled
  - striking_velocity_threshold: 30px/frame → scaled
  - position_movement_threshold: 20px → scaled
  - min_ball_speed: 4px/frame → scaled

- **Tracking:**
  - max_tracking_distance: 100px → scaled

- **Validation:**
  - max_thickness_search: 60px → scaled
  - edge_margin: 40px → scaled
  - position_edge_margin: 20px → scaled

- **Proximity:**
  - max_distance_to_cue_ball: 80px → scaled (~2× ball radius)
  - max_tip_distance: 600px → scaled
  - max_reasonable_distance: 400px → scaled
  - contact_threshold: 60px → scaled (~ball diameter)
  - overlap_threshold: 100px → scaled

- **Shot Detection:**
  - max_velocity: 100px/frame → scaled
  - follow_draw_threshold: 10px → scaled
  - center_offset_threshold: 10px → scaled

**Example:**
For 1080p (1920×1080):
- Scale factor: 0.5
- max_cue_length: 1800 * 0.5 = 900 pixels
- max_distance_to_cue_ball: 80 * 0.5 = 40 pixels
- contact_threshold: 60 * 0.5 = 30 pixels

### 3. Table Detection (`backend/vision/detection/table.py`)

**Changes:**
- Added imports for `POCKET_RADIUS_4K` and `ResolutionConverter`
- Updated `TableDetector.__init__()` to accept `camera_resolution` parameter
- Calculated scale factor and stored as `self.scale`
- Scaled pocket detection parameters

**Scaled Parameters:**
- **Pocket Detection:**
  - min_pocket_area: Based on (POCKET_RADIUS_4K * 0.3)² → scaled
  - max_pocket_area: Based on (POCKET_RADIUS_4K * 1.5)² → scaled
  - max_expected_pocket_distance: 200px → scaled

**Example:**
For 1080p (1920×1080):
- Scale factor: 0.5
- Pocket radius: 72 * 0.5 = 36 pixels
- min_pocket_area: π * (36 * 0.3)² = π * 11.6² ≈ 407 pixels²
- max_pocket_area: π * (36 * 1.5)² = π * 54² ≈ 9161 pixels²

### 4. YOLO Detector (`backend/vision/detection/yolo_detector.py`)

**Changes:**
- Updated `YOLODetector.__init__()` to accept `camera_resolution` parameter
- Stored camera_resolution as instance variable
- Passed camera_resolution to BallDetector when initializing OpenCV classifier

**Impact:**
- OpenCV ball classification used for refinement now uses resolution-aware parameters

### 5. Vision Module (`backend/vision/__init__.py`)

**Changes:**
- Updated YOLODetector instantiation to pass `self.config.camera_resolution`
- Updated CueDetector instantiation to pass `self.config.camera_resolution`
- Updated TableDetector instantiation to pass `self.config.camera_resolution`

**Integration:**
All detectors now receive the actual camera resolution from the vision module configuration, ensuring consistent scaling across the entire detection pipeline.

## Resolution Scaling Logic

### Scale Factor Calculation

```python
from core.resolution_converter import ResolutionConverter

# Calculate scale from 4K to camera resolution
scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(camera_resolution)
scale = scale_x  # Use X scale for consistency (assumes square pixels)
```

### Scaling Examples

| Camera Resolution | Scale Factor | Ball Radius | Max Cue Length | Contact Threshold |
|-------------------|--------------|-------------|----------------|-------------------|
| 4K (3840×2160)    | 1.0          | 36px        | 1800px         | 60px              |
| 1080p (1920×1080) | 0.5          | 18px        | 900px          | 30px              |
| 720p (1280×720)   | 0.333        | 12px        | 600px          | 20px              |
| 480p (640×480)    | 0.167        | 6px         | 300px          | 10px              |

## Parameters That Are NOT Scaled

The following parameters are resolution-independent and remain unscaled:

1. **Ratios and Proportions:**
   - `radius_tolerance` (0.30 = 30%)
   - `min_circularity` (0.75)
   - `max_overlap_ratio` (0.30)
   - `hough_min_dist_ratio` (0.8)

2. **Angles:**
   - `angle_change_threshold` (5 degrees)
   - `max_angle_change` (45 degrees)
   - `english_deviation_angle` (30 degrees)

3. **Color Values:**
   - HSV color ranges
   - Thresholds based on color intensity

4. **Confidence Scores:**
   - `min_confidence` (0.4)
   - `min_detection_confidence` (0.6)

5. **Iteration Counts:**
   - `tracking_history_size` (10)
   - `corner_max_iterations` (30)

## 4K Constants Used

From `backend/core/constants_4k.py`:

```python
CANONICAL_RESOLUTION = (3840, 2160)  # 4K UHD
BALL_RADIUS_4K = 36  # pixels
BALL_DIAMETER_4K = 72  # pixels
POCKET_RADIUS_4K = 72  # pixels
TABLE_WIDTH_4K = 3200  # pixels
TABLE_HEIGHT_4K = 1600  # pixels
```

## Benefits

1. **Resolution Independence:** Detection works consistently across all resolutions
2. **Automatic Scaling:** No manual calibration needed when changing resolution
3. **Predictable Behavior:** All measurements scale proportionally
4. **Maintainability:** Single source of truth for physical dimensions (4K constants)
5. **Backward Compatibility:** Existing configs still work, scaled automatically

## Testing Recommendations

1. **Multi-Resolution Testing:**
   - Test at 4K (3840×2160) - baseline
   - Test at 1080p (1920×1080) - most common
   - Test at 720p (1280×720) - lower quality cameras
   - Verify detection quality is consistent

2. **Verification:**
   - Check ball detection accuracy at different resolutions
   - Verify cue stick detection at different scales
   - Confirm pocket and table detection work across resolutions

3. **Edge Cases:**
   - Very low resolution (640×480) - ensure parameters don't scale too small
   - Very high resolution (8K) - ensure parameters don't become too large
   - Non-standard aspect ratios

## Migration Notes

### For Configuration Files

No changes needed! Existing configuration files work as-is. If pixel values are specified in config, they are used directly. If not specified, defaults are calculated based on 4K constants and scaled to camera resolution.

### For New Detectors

When creating new detection modules:
1. Accept `camera_resolution` in `__init__`
2. Calculate scale factor using `ResolutionConverter.calculate_scale_from_4k()`
3. Use 4K constants as base values
4. Scale pixel-based parameters by multiplying by scale factor
5. Cast to `int()` for pixel counts
6. Document which parameters are scaled

### Default Values Pattern

```python
# Good: Resolution-aware default
scale_x, _ = ResolutionConverter.calculate_scale_from_4k(camera_resolution)
default_max_distance = int(100 * scale_x)  # 100px in 4K
max_distance = config.get("max_distance", default_max_distance)

# Bad: Hardcoded default
max_distance = config.get("max_distance", 50)  # Assumes specific resolution
```

## Future Enhancements

1. **Aspect Ratio Handling:**
   - Currently uses scale_x for both dimensions (assumes square pixels)
   - Could use separate scale_x and scale_y for non-square pixels
   - Add aspect ratio validation

2. **DPI/PPI Awareness:**
   - Consider physical sensor size
   - Account for lens focal length
   - True physical-to-pixel mapping

3. **Adaptive Scaling:**
   - Analyze actual ball size in frame
   - Adjust scale factor dynamically
   - Compensate for perspective distortion

4. **Configuration Validation:**
   - Warn if config values seem incorrect for resolution
   - Suggest optimal values based on detected conditions
   - Auto-calibration mode

## Conclusion

All vision detection modules now use resolution-aware pixel thresholds based on 4K canonical constants. This ensures consistent detection quality across all camera resolutions without manual calibration. The implementation is backward compatible and requires no configuration changes.

The system now properly scales:
- Ball detection radii
- Cue stick length and thickness thresholds
- Motion analysis distances and velocities
- Tracking distances
- Pocket detection parameters
- Contact and proximity thresholds

All scaling is automatic based on the camera_resolution parameter passed to each detector during initialization.
