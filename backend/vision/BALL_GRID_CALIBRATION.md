# Ball Grid Camera Calibration

## Overview

The `calibrate_distortion_from_balls.py` script provides an alternative camera calibration method that uses billiard balls arranged in a regular grid pattern instead of traditional chessboard patterns. This is particularly useful for pool table cameras where placing a chessboard pattern may be impractical.

## How It Works

### 1. Ball Detection
- Uses OpenCV's HoughCircles to detect circular objects
- Filters out false positives (pockets, reflections) by:
  - Excluding detections near image borders (5% margin)
  - Rejecting dark/low-brightness circles (pockets, shadows)
- Visualizes detections: green = valid balls, red = filtered out

### 2. Grid Structure Inference
- Uses DBSCAN clustering on y-coordinates to group balls into rows
- Sorts balls within each row by x-coordinate
- Handles incomplete rows (missing balls)
- Determines grid dimensions from most common row length

### 3. Ideal Grid Calculation
- Calculates average horizontal and vertical spacing from detected balls
- Generates an ideal regular grid with uniform spacing
- Centers the ideal grid at the detected grid's center

### 4. Distortion Calibration
- Uses detected ball positions as image points
- Uses ideal grid positions as object points (z=0 plane)
- Fixes camera intrinsics (focal length, principal point)
- Optimizes only radial distortion coefficients (k1, k2)
- Clamps extreme distortion values to [-0.3, 0.3] for stability

### 5. Validation
- Applies undistortion to the image
- Re-detects balls and measures grid straightness
- Reports improvement in grid regularity

## Usage

### Basic Usage
```bash
python backend/vision/calibrate_distortion_from_balls.py
```

### With Custom Parameters
```bash
python backend/vision/calibrate_distortion_from_balls.py \
    --image path/to/grid.jpg \
    --output path/to/camera_params.yaml \
    --debug-dir path/to/debug \
    --min-radius 20 \
    --max-radius 80
```

### Parameters
- `--image`: Path to grid image (default: `backend/vision/test_data/grid.jpg`)
- `--output`: Path to save calibration YAML (default: `backend/calibration_data/camera/camera_params.yaml`)
- `--debug-dir`: Directory for debug images (default: `backend/calibration_data/camera/debug`)
- `--min-radius`: Minimum ball radius in pixels (default: 20)
- `--max-radius`: Maximum ball radius in pixels (default: 80)

## Output Files

### Calibration File (YAML)
```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 3072., 0., 1920., 0., 3072., 1080., 0., 0., 1. ]
dist_coeffs: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ 0.3, -0.3, 0., 0. ]
image_width: 3840
image_height: 2160
calibration_error: 360.37
calibration_date: "2025-10-21T22:41:42.993595"
calibration_method: ball_grid
notes: Camera calibration computed from billiard ball grid pattern
```

### Debug Images
- `detected_balls.jpg`: Shows all detected circles (green = valid, red = filtered)
- `grid_analysis.jpg`: Shows detected grid (green) vs ideal grid (blue)
- `undistorted.jpg`: Undistorted version of the original image

## Example Output

```
================================================================================
CALIBRATION SUMMARY
================================================================================
Grid structure: 4 rows × 4 columns
Calibration points: 16
RMS error: 360.37 pixels

Camera matrix:
[[3.072e+03 0.000e+00 1.920e+03]
 [0.000e+00 3.072e+03 1.080e+03]
 [0.000e+00 0.000e+00 1.000e+00]]

Distortion coefficients: [ 0.3 -0.3  0.   0. ]
  k1 (radial): 0.300000
  k2 (radial): -0.300000

Grid straightness improvement: 15.2%
  Before: 294.81px mean deviation
  After:  250.15px mean deviation

Calibration saved to: backend/calibration_data/camera/camera_params.yaml
Debug images saved to: backend/calibration_data/camera/debug/
================================================================================
```

## Integration with Existing Code

The generated YAML file is fully compatible with `CameraCalibrator.load_calibration()`:

```python
from backend.vision.calibration.camera import CameraCalibrator

calibrator = CameraCalibrator()
calibrator.load_calibration("backend/calibration_data/camera/camera_params_ball_grid.yaml")

# Use the calibration
undistorted = calibrator.undistort_image(image)
```

## Requirements

The script requires:
- OpenCV (`cv2`)
- NumPy
- scikit-learn (for DBSCAN clustering)

All dependencies are already in the project's requirements.

## Limitations and Considerations

### Current Limitations
1. **Grid Detection**: The clustering algorithm may not perfectly detect irregular grids
   - Works best with complete 4×4 or similar regular grids
   - May misclassify if too many balls are missing or misaligned

2. **Single Image Calibration**: Using only one image provides less robust calibration than multi-image chessboard calibration
   - Fixed focal length and principal point assumptions
   - Only radial distortion (k1, k2) is optimized
   - Higher calibration error is expected

3. **Distortion Limits**: Extreme distortion values are clamped to [-0.3, 0.3]
   - Prevents unstable calibration from limited data
   - May underestimate very strong barrel/pincushion distortion

### Best Practices
1. **Image Quality**: Use a high-resolution image with clear ball positions
2. **Lighting**: Ensure even lighting to help with ball detection
3. **Grid Layout**: Arrange balls in as perfect a grid as possible
4. **Ball Count**: More balls = better calibration (minimum 4, recommended 16+)
5. **Coverage**: Spread balls across the entire image area for better distortion estimation

### When to Use This Method
- Quick calibration when a chessboard pattern is unavailable
- Initial distortion estimate for table-mounted cameras
- Validation/comparison with other calibration methods

### When NOT to Use This Method
- High-precision applications requiring <1px calibration error
- Cameras with extreme fisheye distortion
- When a proper chessboard calibration is feasible

## Technical Details

### Camera Model
The script uses the standard pinhole camera model with radial distortion:

```
x_distorted = x * (1 + k1*r² + k2*r⁴)
y_distorted = y * (1 + k1*r² + k2*r⁴)
```

Where:
- `r² = x² + y²` (normalized distance from principal point)
- `k1`, `k2` are radial distortion coefficients
- Tangential distortion is assumed to be zero

### Calibration Flags
The script uses these OpenCV calibration flags:
- `CALIB_USE_INTRINSIC_GUESS`: Start with estimated focal length
- `CALIB_FIX_PRINCIPAL_POINT`: Assume center at image center
- `CALIB_FIX_ASPECT_RATIO`: Assume square pixels (fx = fy)
- `CALIB_FIX_FOCAL_LENGTH`: Only optimize distortion
- `CALIB_ZERO_TANGENT_DIST`: Assume no tangential distortion
- `CALIB_FIX_K3`: Only use k1 and k2 (not k3)

These constraints improve stability when calibrating from limited data.

## Future Improvements

Potential enhancements:
1. **Better Grid Detection**: Use 2D DBSCAN or RANSAC for more robust grid inference
2. **Multi-Image Support**: Accept multiple grid images for better calibration
3. **Automatic Grid Size Detection**: Detect expected grid size from ball density
4. **Interactive Mode**: Allow manual correction of grid structure
5. **Table Constraint**: Use known table dimensions to constrain calibration
6. **Focal Length Optimization**: Optionally optimize focal length for better accuracy

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Camera Calibration Method](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)
- Project specification: FR-VIS-039 to FR-VIS-043 (camera calibration requirements)
