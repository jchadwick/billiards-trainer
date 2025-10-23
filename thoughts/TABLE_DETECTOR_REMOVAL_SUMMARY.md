# TableDetector Module Removal Summary

## Overview
The TableDetector module has been completely removed from the billiards-trainer backend to eliminate 21ms per-frame processing overhead. Table boundaries are now determined exclusively through static calibration.

## Files Deleted
- `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/table.py` (1058 lines)

## Import Statements Removed

### backend/vision/__init__.py
- **Line 24**: `from .detection.table import TableDetector`

### backend/api/routes/calibration.py
- **Line 1602-1604**: Conditional import of TableDetector (replaced with HTTP 501 error)
- **Line 1767**: `from backend.vision.detection.table import TableDetector` (unused import removed)

### backend/vision/calibrate_from_grid.py
- **Lines 69-72**: Dynamic module loading of TableDetector (replaced with comment)

## Code Blocks Removed

### backend/vision/__init__.py

#### Initialization (lines 397-405)
```python
# Table detection (separate from ball/cue detection)
if self.config.enable_table_detection:
    base_detection_config = {"debug_mode": self.config.debug_mode}
    self.table_detector = TableDetector(
        base_detection_config,
        camera_resolution=self.config.camera_resolution,
    )
else:
    self.table_detector = None
```
**Replaced with**: `self.table_detector = None` with comment

#### Processing Logic (lines 914-950)
```python
# Table detection
if self.table_detector and self.config.enable_table_detection:
    if self.profiler:
        self.profiler.start_stage("table_detection")

    try:
        table_confidence_threshold = _get_config_value(
            "vision.detection.table_detection_confidence_threshold", 0.5
        )
        table_result = self.table_detector.detect_complete_table(
            processed_frame
        )
        if (
            table_result
            and table_result.confidence > table_confidence_threshold
        ):
            # Convert to our Table model format
            detected_table = Table(
                corners=table_result.corners.to_list(),
                pockets=[
                    pocket.position for pocket in table_result.pockets
                ],
                width=table_result.width,
                height=table_result.height,
                surface_color=table_result.surface_color,
            )

            self.stats.detection_accuracy["table"] = table_result.confidence
        else:
            self.stats.detection_accuracy["table"] = 0.0

    except Exception as e:
        logger.warning(f"Table detection failed: {e}")
        self.stats.detection_accuracy["table"] = 0.0

    if self.profiler:
        self.profiler.end_stage("table_detection")
```
**Replaced with**: `self.stats.detection_accuracy["table"] = 0.0` with comment

### backend/api/routes/calibration.py

#### Fisheye Auto-Calibration Endpoint (lines 1600-1624)
```python
# Import table detector
try:
    from ...vision.detection.table import TableDetector
except ImportError:
    from vision.detection.table import TableDetector

# Detect table corners with very relaxed constraints for fisheye-distorted images
# Since we're calibrating FOR fisheye correction, the image will be heavily distorted
table_config = {
    "expected_aspect_ratio": table_width / table_height,
    "aspect_ratio_tolerance": 10.0,  # Effectively disable aspect ratio check
    "min_table_area_ratio": 0.05,  # Allow smaller table detection
    "side_length_tolerance": 0.5,  # Allow 50% difference in opposite sides (fisheye!)
}
table_detector = TableDetector(table_config)
table_corners_result = table_detector.detect_table_boundaries(frame)

if table_corners_result is None:
    raise HTTPException(
        status_code=400,
        detail="Could not detect table boundaries in the frame. Ensure the table is clearly visible and well-lit.",
    )

# Convert TableCorners to list format
table_corners = table_corners_result.to_list()
```
**Replaced with**: HTTP 501 error indicating automatic fisheye calibration is no longer supported

### backend/vision/calibrate_from_grid.py

#### Auto-Detection Function (lines 161-214)
```python
def detect_table_corners_auto(
    image: NDArray[np.uint8],
) -> Optional[list[tuple[float, float]]]:
    """Automatically detect table corners using TableDetector.
    ...
    """
    logger.info("Attempting automatic table corner detection...")

    # Create minimal config for TableDetector
    table_detector_config = {
        "color_ranges": {
            "green": {
                "lower_hsv": [35, 40, 40],
                "upper_hsv": [85, 255, 255],
            }
        },
        "geometry": {
            "expected_aspect_ratio": 2.0,
            "aspect_ratio_tolerance": 0.5,
            "side_length_tolerance": 0.15,
            "min_table_area_ratio": 0.1,
            "max_table_area_ratio": 0.95,
            "playing_surface_inset_ratio": 0.02,
        },
        ...
    }

    detector = TableDetector(table_detector_config)
    corners_obj = detector.detect_table_boundaries(image, use_pocket_detection=False)

    if corners_obj is None:
        logger.warning("Automatic detection failed")
        return None

    corners_list = corners_obj.to_list()
    logger.info(f"Auto-detected corners: {corners_list}")

    return corners_list
```
**Replaced with**: Function stub that always returns None with warning

### backend/tools/performance_diagnostics.py

#### Table Detection Block (lines 87-106)
```python
# Table detection
if self.table_detector and self.config.enable_table_detection:
    self.profiler.start_stage("table_detection")
    try:
        table_confidence_threshold = config.get(
            "vision.detection.table_detection_confidence_threshold", 0.5
        )
        table_result = self.table_detector.detect_complete_table(processed_frame)
        if table_result and table_result.confidence > table_confidence_threshold:
            from vision.models import Table
            detected_table = Table(
                corners=table_result.corners.to_list(),
                pockets=[pocket.position for pocket in table_result.pockets],
                width=table_result.width,
                height=table_result.height,
                surface_color=table_result.surface_color,
            )
    except Exception as e:
        logger.warning(f"Table detection failed: {e}")
    self.profiler.end_stage("table_detection")
```
**Replaced with**: Comment only

## Configuration Keys That Need Cleanup

These configuration keys in `config.json` are now obsolete and can be removed in a future cleanup:

### Primary Config Keys
- `vision.detection.enable_table_detection` (line 72) - **Currently set to `false`** ✓
- `vision.detection.table_detection_confidence_threshold` (line 77)

### Table Detection Config Section
Entire `vision.detection.table_detection` object (lines 100-201):
- `vision.detection.table_detection.color_ranges`
  - `green.lower_hsv`, `green.upper_hsv`
  - `turquoise.lower_hsv`, `turquoise.upper_hsv`
  - `blue.lower_hsv`, `blue.upper_hsv`
  - `red.lower_hsv`, `red.upper_hsv`
- `vision.detection.table_detection.geometry`
  - `expected_aspect_ratio`
  - `aspect_ratio_tolerance`
  - `side_length_tolerance`
  - `min_table_area_ratio`
  - `max_table_area_ratio`
  - `playing_surface_inset_ratio`
- `vision.detection.table_detection.edge_detection`
  - `canny_low_threshold`
  - `canny_high_threshold`
  - `contour_epsilon_factor`
  - `contour_epsilon_multipliers`
- `vision.detection.table_detection.corner_refinement`
  - `window_size`
  - `max_iterations`
  - `epsilon`
- `vision.detection.table_detection.pocket_detection`
  - `color_threshold`
  - `min_area`
  - `max_area`
  - `min_confidence`
  - `max_expected_distance`
- `vision.detection.table_detection.morphology`
  - `large_kernel_size`
  - `small_kernel_size`
- `vision.detection.table_detection.temporal_stability`
  - `blending_alpha`
  - `min_previous_confidence`
- `vision.detection.table_detection.confidence_weights`
  - `geometry`
  - `pockets`
  - `surface`

## What Was Preserved

### Static Table Calibration
- `table.playing_area_corners` - Static table boundaries from calibration
- `table.marker_dots` - Marker dot positions for masking
- `table.calibration_resolution_width/height` - Resolution used during calibration

### Table Coordinate Transformation
All code that uses pre-configured table boundaries for coordinate transformation remains intact.

## Issues and Concerns

1. **Fisheye Auto-Calibration Endpoint Disabled**: The `/api/v1/vision/calibration/camera/auto-calibrate` endpoint now returns HTTP 501. Users must use manual corner selection or pre-calibrate cameras.

2. **Auto-Detection in calibrate_from_grid.py Disabled**: The automatic corner detection function now always returns None. Users must use manual corner selection mode.

3. **No Runtime Table Boundary Adjustment**: The system now relies entirely on static calibration. If the camera moves or table position changes, manual recalibration is required.

## Performance Impact

**Expected improvement**: ~21ms per frame removed from vision processing pipeline
- Table detection was taking 20-100ms per frame
- With table detection disabled, this overhead is completely eliminated
- System now uses cached static calibration data only

## Migration Path

For systems that were using dynamic table detection:
1. Perform manual table calibration using the grid-based calibration tool
2. Save calibration data to `config.json` under `table.playing_area_corners`
3. Ensure `vision.detection.enable_table_detection` is set to `false`
4. Use manual recalibration if camera or table position changes
