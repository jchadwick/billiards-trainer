# Camera Calibration Data

This directory contains calibration files for camera distortion correction.

## Files

- `camera_params.yaml` - Current camera calibration parameters (OpenCV YAML format)
- `debug/` - Debug visualization images (if generated with `--save-debug-images`)

## Current Calibration

Last calibrated: **2025-10-21**

### Parameters

- **Resolution**: 3840x2160 (4K)
- **RMS Error**: 7.43 pixels
- **Distortion Type**: Barrel distortion (fisheye)
- **k1**: 0.168009 (primary radial)
- **k2**: -0.297938 (secondary radial)

### Calibration Method

Computed from billiards table rectangular geometry using the test grid image.

## Usage

To recalibrate, run from the `backend` directory:

```bash
python vision/calibrate_from_grid.py --save-debug-images
```

See `vision/CALIBRATION_README.md` for detailed instructions.

## Loading in Code

```python
from vision.calibration.camera import CameraCalibrator

calibrator = CameraCalibrator()
if calibrator.load_calibration("calibration_data/camera/camera_params.yaml"):
    # Apply to frames
    undistorted = calibrator.undistort_image(frame)
```

## Notes

- This calibration is specific to the camera and resolution used
- Recalibrate if you change cameras, lenses, or resolution
- The calibration corrects for lens distortion only, not perspective distortion
