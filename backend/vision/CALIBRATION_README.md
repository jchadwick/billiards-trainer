# Camera Distortion Calibration

This directory contains tools for calibrating camera barrel/fisheye distortion using the billiards table geometry.

## Overview

The calibration script (`calibrate_from_grid.py`) detects barrel distortion in camera images by analyzing the table boundaries, which should be perfectly rectangular. Any deviation from straight edges indicates lens distortion that needs to be corrected.

## Quick Start

### Running the Calibration

From the `backend` directory:

```bash
python vision/calibrate_from_grid.py --save-debug-images
```

This will:
1. Load the test grid image (`vision/test_data/grid.jpg`)
2. Attempt automatic table corner detection
3. Fall back to manual corner selection if automatic fails
4. Calibrate fisheye distortion parameters
5. Save calibration to `calibration_data/camera/camera_params.yaml`
6. Save debug visualization images

### Manual Corner Selection

If automatic detection fails, you'll see an interactive window:

1. Click the 4 table corners in **CLOCKWISE** order starting from **TOP-LEFT**:
   - Top-left corner
   - Top-right corner
   - Bottom-right corner
   - Bottom-left corner

2. Press **'r'** to reset if you make a mistake
3. Press **Enter** to confirm when all 4 corners are selected
4. Press **'q'** to quit

## Command Line Options

```bash
python vision/calibrate_from_grid.py [options]
```

### Options

- `--input PATH` - Path to input image (default: `vision/test_data/grid.jpg`)
- `--output PATH` - Path to output YAML file (default: `calibration_data/camera/camera_params.yaml`)
- `--manual-corners` - Force manual corner selection (skip automatic detection)
- `--show-visualization` - Display before/after comparison window
- `--save-debug-images` - Save intermediate debug images to `calibration_data/camera/debug/`
- `--table-width METERS` - Real table width in meters (default: 2.54 for 9ft table)
- `--table-height METERS` - Real table height in meters (default: 1.27 for 9ft table)

### Examples

Use a different image:
```bash
python vision/calibrate_from_grid.py --input path/to/image.jpg
```

Force manual corner selection:
```bash
python vision/calibrate_from_grid.py --manual-corners
```

Show interactive visualization:
```bash
python vision/calibrate_from_grid.py --show-visualization
```

## Output Files

### Calibration File

The main output is `calibration_data/camera/camera_params.yaml` in OpenCV YAML format:

```yaml
camera_matrix: !!opencv-matrix
  rows: 3
  cols: 3
  data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
dist_coeffs: !!opencv-matrix
  rows: 4
  cols: 1
  data: [ k1, k2, 0, 0 ]  # Radial distortion coefficients
image_width: 3840
image_height: 2160
calibration_error: 7.43  # RMS error in pixels
calibration_date: "2025-10-21T22:35:23"
calibration_method: table_rectangle
```

### Debug Images (with `--save-debug-images`)

Saved to `calibration_data/camera/debug/`:

1. `original_with_corners.jpg` - Original image with detected corners highlighted
2. `undistorted.jpg` - Image after distortion correction applied

## Understanding the Results

### Distortion Coefficients

The calibration computes two radial distortion coefficients:

- **k1** (primary radial distortion)
  - Negative values: barrel distortion (fisheye effect)
  - Positive values: pincushion distortion
  - Typical range: -0.3 to +0.3

- **k2** (secondary radial distortion)
  - Fine-tunes the correction for stronger distortion
  - Usually opposite sign to k1

### Calibration Quality

- **RMS Error** < 5 pixels: Excellent calibration
- **RMS Error** 5-10 pixels: Good calibration (typical for table-based method)
- **RMS Error** > 10 pixels: May need recalibration or better corner detection

### Example Output

```
CALIBRATION RESULTS
============================================================
Resolution: 3840x2160
Calibration Date: 2025-10-21T22:35:23
RMS Error: 7.4328 pixels

Camera Matrix:
  fx: 3072.00
  fy: 3072.00
  cx: 1920.00
  cy: 1080.00

Distortion Coefficients:
  k1 (radial): 0.168009
  k2 (radial): -0.297938

Distortion Type: barrel distortion (fisheye)
Distortion Strength: moderate
============================================================
```

## Using the Calibration in the Video Module

Once calibrated, the Video Module should:

1. Load the calibration file on startup:
   ```python
   from vision.calibration.camera import CameraCalibrator

   calibrator = CameraCalibrator()
   calibrator.load_calibration("calibration_data/camera/camera_params.yaml")
   ```

2. Apply undistortion to each frame **before** writing to shared memory:
   ```python
   # In video processing loop
   frame = camera.read()
   undistorted_frame = calibrator.undistort_image(frame)
   shared_memory.write(undistorted_frame)
   ```

This ensures all downstream modules (vision processing, physics, etc.) work with distortion-corrected images.

## Troubleshooting

### Automatic Detection Fails

**Symptoms**: Script immediately falls back to manual selection

**Solutions**:
- Ensure table occupies most of the frame
- Check that table felt contrasts with surroundings
- Try adjusting lighting in the test image
- Use manual corner selection (`--manual-corners`)

### High Calibration Error

**Symptoms**: RMS error > 15 pixels

**Solutions**:
- Ensure corners are accurately selected
- Check that table is actually rectangular (not keystone distorted)
- Verify table dimensions are correct
- Try different epsilon values in automatic detection

### Distortion Looks Wrong

**Symptoms**: Undistorted image appears more distorted

**Solutions**:
- Check corner order (must be clockwise from top-left)
- Verify you selected table **edges**, not outer rail edges
- Ensure all 4 corners are on the same plane (playing surface)

### Can't See Difference

**Symptoms**: Original and undistorted images look identical

**Solutions**:
- Your camera may have minimal distortion (k1 ≈ 0)
- This is fine! The calibration will be a no-op
- Check debug images to verify the straightness of table edges

## Advanced Usage

### Custom Table Dimensions

For an 8-foot table:
```bash
python vision/calibrate_from_grid.py \
    --table-width 2.34 \
    --table-height 1.17
```

For a 7-foot table:
```bash
python vision/calibrate_from_grid.py \
    --table-width 1.98 \
    --table-height 0.99
```

### Batch Processing Multiple Images

```bash
for img in images/*.jpg; do
    python vision/calibrate_from_grid.py \
        --input "$img" \
        --output "calibrations/$(basename $img .jpg).yaml"
done
```

## Technical Details

### Calibration Method

The script uses the `CameraCalibrator.calibrate_fisheye_from_table()` method which:

1. Takes the 4 table corners as input
2. Generates additional sample points along table edges (default: 5 per edge = 20 total)
3. Uses these points as 3D-to-2D correspondences
4. Fixes camera intrinsics (focal length, principal point) based on image size
5. Optimizes only radial distortion coefficients (k1, k2) for stability
6. Returns calibrated camera parameters

This single-image approach is less accurate than traditional chessboard calibration but sufficient for correcting moderate barrel distortion.

### Coordinate System

- **Image coordinates**: Top-left origin, x-right, y-down (pixels)
- **Table coordinates**: Center origin, x along width, y along length (meters)
- **World Z-axis**: Perpendicular to table (height = 0 for table surface)

## See Also

- `vision/calibration/camera.py` - CameraCalibrator implementation
- `vision/detection/table.py` - TableDetector for automatic corner detection
- `calibration_data/camera/camera_params.yaml` - Current calibration file
