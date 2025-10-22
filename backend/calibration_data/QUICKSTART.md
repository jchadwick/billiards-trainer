# Camera Calibration - Quick Start

## TL;DR

```bash
cd backend
python vision/calibrate_from_grid.py --save-debug-images
```

Click 4 table corners (clockwise from top-left), press Enter.

Output: `calibration_data/camera/camera_params.yaml`

## What This Does

Calibrates camera barrel/fisheye distortion by analyzing the pool table's rectangular geometry.

## Current Calibration

✅ **Already calibrated** on 2025-10-21 using `vision/test_data/grid.jpg`

- Resolution: 3840x2160
- RMS Error: 7.43 pixels (good)
- Distortion: Moderate barrel (k1=0.168)

## When to Recalibrate

- Changed camera or lens
- Changed resolution
- Moved camera position significantly
- Distortion seems incorrect

## Full Documentation

See `vision/CALIBRATION_README.md` for complete instructions.
