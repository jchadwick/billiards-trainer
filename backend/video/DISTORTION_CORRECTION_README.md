# Video Module Distortion Correction

## Overview

The Video Module now includes built-in camera distortion correction that automatically undistorts frames before writing them to shared memory. This ensures all downstream consumers (Vision Module, API, etc.) receive corrected frames without needing to implement their own correction.

## Features

- **Automatic calibration loading** from YAML file on startup
- **Performance-optimized** using pre-computed remap maps
- **Graceful fallback** to raw frames if calibration unavailable
- **Configurable** via config.json (can be disabled)
- **Zero downstream changes** - transparent to consumers
- **Performance monitoring** - tracks undistortion time per frame

## Quick Start

### 1. Enable Distortion Correction (Default)

Edit `config.json`:
```json
{
  "video": {
    "enable_distortion_correction": true,
    "calibration_file_path": "backend/calibration_data/camera/camera_params.yaml"
  }
}
```

### 2. Run Video Module

```bash
python -m backend.video
```

Expected output:
```
Loading camera calibration from: backend/calibration_data/camera/camera_params.yaml
Camera calibration loaded successfully. Resolution: (3840, 2160), Error: 7.4328
Initializing undistortion maps for 1920x1080...
Undistortion maps initialized successfully
...
Stats: ... undistortion_enabled=true, frames_undistorted=300, avg_undistortion_time=2.15ms
```

## Configuration

### Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `video.enable_distortion_correction` | boolean | `true` | Enable/disable distortion correction |
| `video.calibration_file_path` | string | `"backend/calibration_data/camera/camera_params.yaml"` | Path to calibration file (relative to project root) |

### Disable Distortion Correction

To use raw uncorrected frames:

```json
{
  "video": {
    "enable_distortion_correction": false
  }
}
```

## Calibration File Format

The calibration file must be in OpenCV YAML format:

```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
dist_coeffs: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ k1, k2, p1, p2 ]
image_width: 3840
image_height: 2160
calibration_error: 7.4328
calibration_date: "2025-10-21T22:35:23.542343"
```

### Calibration Parameters

- **camera_matrix**: 3x3 intrinsic camera matrix
  - `fx`, `fy`: Focal lengths in pixels
  - `cx`, `cy`: Principal point (optical center)
- **dist_coeffs**: Distortion coefficients
  - `k1`, `k2`: Radial distortion
  - `p1`, `p2`: Tangential distortion (usually 0)
- **image_width**, **image_height**: Calibration resolution

## Error Handling

The system handles errors gracefully and continues with raw frames:

### Calibration File Not Found
```
WARNING - Calibration file not found: backend/calibration_data/camera/camera_params.yaml.
          Continuing with raw frames (no distortion correction).
```
→ **Action**: Check file path, create calibration if needed

### Calibration Load Failed
```
WARNING - Failed to load calibration from camera_params.yaml.
          Continuing with raw frames (no distortion correction).
```
→ **Action**: Verify YAML format, check file permissions

### Undistortion Map Initialization Failed
```
ERROR - Failed to initialize undistortion maps: [error details].
        Disabling distortion correction.
```
→ **Action**: Check OpenCV installation, verify calibration parameters

### Per-Frame Undistortion Error
```
ERROR - Undistortion failed: [error details]
```
→ **Action**: Check system resources, frame returns raw (uncorrected)

## Performance

### Expected Performance

| Resolution | Initialization | Per-Frame | Memory Overhead |
|------------|----------------|-----------|-----------------|
| 1920×1080 | ~200ms | 1-3ms | ~50MB |
| 3840×2160 (4K) | ~300ms | 4-8ms | ~200MB |

### Optimization Details

1. **Pre-computed Maps**: Uses `cv2.initUndistortRectifyMap()` + `cv2.remap()`
   - 3-5x faster than `cv2.undistort()` per frame
   - Maps computed once at startup, reused for all frames

2. **Hardware Acceleration**: OpenCV uses SIMD and multi-threading automatically

3. **Memory Efficiency**: Maps stored as 32-bit float for optimal performance

### Performance Monitoring

Statistics are logged every 10 seconds:

```
Stats: uptime=10.0s, frames_captured=300, frames_written=300, fps=30.0,
       camera_fps=30.0, camera_dropped=0, write_counter=300,
       undistortion_enabled=true, frames_undistorted=300,
       avg_undistortion_time=2.15ms
```

**Key Metrics**:
- `undistortion_enabled`: `true` if correction active
- `frames_undistorted`: Total frames corrected
- `avg_undistortion_time`: Average time per frame (milliseconds)

### Performance Troubleshooting

**If avg_undistortion_time > 10ms**:
1. Check CPU usage - close unnecessary processes
2. Verify OpenCV optimizations: `python -c "import cv2; print(cv2.getBuildInformation())"`
3. Consider reducing resolution or disabling correction
4. Check for thermal throttling on Raspberry Pi/embedded devices

## Technical Implementation

### Distortion Correction Pipeline

```
1. Startup:
   ┌─────────────────────────┐
   │ Load Calibration        │
   │ (camera_matrix, dist)   │
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │ Initialize Undist Maps  │
   │ (cv2.initUndistortRect) │
   └───────────┬─────────────┘
               │
2. Per Frame:  │
   ┌───────────▼─────────────┐
   │ Capture Raw Frame       │
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │ Apply cv2.remap()       │
   │ (fast, uses precomp)    │
   └───────────┬─────────────┘
               │
   ┌───────────▼─────────────┐
   │ Write to Shared Memory  │
   └─────────────────────────┘
```

### Code Flow

```python
# In VideoProcess.__init__()
self._enable_distortion_correction = config.get("video.enable_distortion_correction", True)
self._calibration_file_path = config.get("video.calibration_file_path", ...)

# In VideoProcess.start()
self._load_calibration()  # Loads YAML file
# ... get first frame to determine dimensions ...
self._initialize_undistortion_maps(width, height)  # Pre-compute remap maps

# In VideoProcess._main_loop()
frame, frame_info = self.camera.get_latest_frame()
processed_frame = self._undistort_frame(frame)  # Apply correction
self.ipc_writer.write_frame(processed_frame, ...)
```

## Testing

### Unit Test
```bash
cd /Users/jchadwick/code/billiards-trainer
python -c "
import sys
from pathlib import Path

backend_dir = Path('backend')
sys.path.insert(0, str(backend_dir.parent))

from backend.video.process import VideoProcess
from backend.config import config

print('✓ VideoProcess available')
print(f'✓ Distortion correction: {config.get(\"video.enable_distortion_correction\")}')
print(f'✓ Calibration path: {config.get(\"video.calibration_file_path\")}')
"
```

### Integration Test
```bash
# Start video module with calibration
python -m backend.video

# In another terminal, check logs
tail -f logs/video.log | grep -E "(calibration|undistortion)"
```

### Visual Verification
1. Start Video Module with correction enabled
2. Open web interface showing video feed
3. Look for straight lines (table edges) - should appear straight
4. Compare with correction disabled - edges may curve (barrel/pincushion distortion)

## Troubleshooting

### Issue: "Calibration file not found"

**Check**:
```bash
ls -la backend/calibration_data/camera/camera_params.yaml
```

**Solution**:
- Verify path in config.json matches actual file location
- Use absolute path if needed: `/full/path/to/camera_params.yaml`
- Run calibration wizard to generate file

### Issue: "No distortion correction applied"

**Verify**:
1. Config setting: `"enable_distortion_correction": true`
2. Logs show: `"Camera calibration loaded successfully"`
3. Logs show: `"Undistortion maps initialized successfully"`
4. Stats show: `"undistortion_enabled=true"`

**If all true but still no correction**:
- Check distortion coefficients - very small values (< 0.01) have minimal effect
- Verify camera resolution matches calibration resolution
- Compare side-by-side with raw feed

### Issue: High CPU usage / slow frame rate

**Solutions**:
1. **Disable correction**:
   ```json
   {"video": {"enable_distortion_correction": false}}
   ```

2. **Reduce resolution**: Use lower camera resolution if possible

3. **Check OpenCV build**:
   ```python
   import cv2
   info = cv2.getBuildInformation()
   # Look for: SIMD, TBB, CUDA, OpenCL
   ```

4. **Monitor performance**:
   ```bash
   # Watch undistortion time
   tail -f logs/video.log | grep "avg_undistortion_time"
   ```

### Issue: Distortion makes image worse

**Possible causes**:
1. **Wrong calibration**: Calibration from different camera/resolution
2. **Over-correction**: Distortion coefficients too large
3. **Resolution mismatch**: Calibration was for different resolution

**Solutions**:
- Re-calibrate camera at current resolution
- Verify calibration quality (reprojection error < 1.0)
- Try disabling correction and observe difference

## Creating Calibration Files

### Using Chessboard Pattern

1. **Print chessboard**: 9×6 internal corners, 25mm squares
2. **Capture images**: 10-20 images at different angles/positions
3. **Run calibration**:
   ```python
   from backend.vision.calibration.camera import CameraCalibrator

   calibrator = CameraCalibrator()
   success, params = calibrator.calibrate_intrinsics(images)

   if success:
       calibrator.save_fisheye_calibration_yaml("camera_params.yaml")
   ```

### Using Table Rectangle (Automatic)

The system can auto-calibrate from the table's rectangular geometry:

```python
from backend.vision.calibration.camera import CameraCalibrator
import cv2

calibrator = CameraCalibrator()
image = cv2.imread("table_image.jpg")
table_corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # 4 corners

success, params = calibrator.calibrate_fisheye_from_table(
    image, table_corners, table_dimensions=(2.54, 1.27)
)

if success:
    calibrator.save_fisheye_calibration_yaml(
        "backend/calibration_data/camera/camera_params.yaml"
    )
```

## Future Enhancements

### Planned Features
- [ ] Runtime calibration reload (without restart)
- [ ] Multiple calibration profiles (per-camera)
- [ ] API endpoint to toggle correction on/off
- [ ] ROI-based undistortion (only table area)
- [ ] GPU acceleration (CUDA/OpenCL)

### Alternative Approaches
- **Dual buffers**: Separate raw/corrected shared memory segments
- **Partial correction**: Only undistort detection regions
- **Adaptive quality**: Reduce correction under high CPU load
- **Multi-resolution**: Different correction levels for different consumers

## References

- **OpenCV Calibration Tutorial**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **Camera Calibration Theory**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Remap Performance**: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- **Project Calibration Docs**: `backend/calibration_data/camera/README.md`

## Support

For issues or questions:
1. Check logs: `logs/video.log`
2. Verify configuration: `config.json`
3. Test calibration loading (see Testing section)
4. Review performance stats in logs
