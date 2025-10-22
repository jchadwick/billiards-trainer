# Video Module Distortion Correction Integration

## Summary

Integrated camera distortion correction into the Video Module to undistort frames before writing them to shared memory. The implementation includes dual feed support (can be toggled on/off), graceful error handling, and performance monitoring.

## Changes Made

### 1. Modified Files

#### backend/video/process.py
- **Added imports**: `Path`, `np` (numpy), `CameraCalibrator`
- **New instance variables**:
  - `_enable_distortion_correction`: Config flag to enable/disable correction
  - `_calibration_file_path`: Path to calibration YAML file
  - `calibrator`: CameraCalibrator instance
  - `_undistortion_maps`: Pre-computed remap maps for fast undistortion
  - `_frames_undistorted`: Counter for undistorted frames
  - `_total_undistortion_time`: Performance tracking

- **New methods**:
  - `_load_calibration()`: Loads camera calibration from YAML file
  - `_initialize_undistortion_maps()`: Pre-computes cv2.remap maps for fast processing
  - `_undistort_frame()`: Applies undistortion to a single frame

- **Modified methods**:
  - `__init__()`: Initialize distortion correction variables
  - `_main_loop()`: Apply undistortion before writing to shared memory
  - `_log_statistics()`: Include undistortion performance metrics
  - `start()`: Load calibration and initialize undistortion maps

#### config.json
- Added `video.enable_distortion_correction`: `true` (default)
- Added `video.calibration_file_path`: `"backend/calibration_data/camera/camera_params.yaml"`

## Features

### 1. Automatic Calibration Loading
- Loads calibration from configured path on startup
- Validates calibration file exists and is readable
- Graceful fallback to raw frames if calibration unavailable

### 2. Performance Optimization
- Pre-computes undistortion maps using `cv2.initUndistortRectifyMap()`
- Uses fast `cv2.remap()` instead of `cv2.undistort()` per frame
- Maps computed once at initialization based on actual frame dimensions

### 3. Error Handling
- **File not found**: Logs warning, continues with raw frames
- **Load failure**: Logs warning, disables correction, continues
- **Map initialization failure**: Logs error, disables correction
- **Per-frame undistortion failure**: Logs error, returns raw frame

### 4. Performance Monitoring
- Tracks number of frames undistorted
- Measures total undistortion time
- Logs average undistortion time per frame (in milliseconds)
- Statistics logged every 10 seconds with other metrics

### 5. Dual Feed Support (Option B)
- Configuration flag `enable_distortion_correction` toggles correction on/off
- When `false`: raw frames pass through unchanged
- When `true`: frames are undistorted before writing to shared memory
- Same shared memory segment, no protocol changes

## Configuration

```json
{
  "video": {
    "enable_distortion_correction": true,
    "calibration_file_path": "backend/calibration_data/camera/camera_params.yaml"
  }
}
```

### Options
- `enable_distortion_correction`: `true` to enable, `false` for raw frames
- `calibration_file_path`: Relative or absolute path to OpenCV YAML calibration file

## Usage

### Normal Operation (Distortion Correction Enabled)
```bash
# config.json has enable_distortion_correction: true
python -m backend.video
```

Output:
```
Loading camera calibration from: backend/calibration_data/camera/camera_params.yaml
Camera calibration loaded successfully. Resolution: (3840, 2160), Error: 7.4328
Initializing undistortion maps for 1920x1080...
Undistortion maps initialized successfully
Stats: ... undistortion_enabled=true, frames_undistorted=300, avg_undistortion_time=2.15ms
```

### Raw Feed (Distortion Correction Disabled)
```json
{
  "video": {
    "enable_distortion_correction": false
  }
}
```

Output:
```
Distortion correction is disabled
Stats: ... undistortion_enabled=false
```

### Missing Calibration File (Automatic Fallback)
```
Calibration file not found: backend/calibration_data/camera/camera_params.yaml.
Continuing with raw frames (no distortion correction).
Stats: ... undistortion_enabled=false
```

## Performance

### Expected Performance
- **Initialization**: ~100-500ms (one-time, loads calibration + computes maps)
- **Per-frame overhead**: ~1-3ms @ 1920x1080 (using pre-computed maps)
- **Memory overhead**: ~50MB for undistortion maps (32-bit float maps)
- **CPU impact**: Minimal (<5% additional with hardware acceleration)

### Optimization Details
1. **Pre-computed maps**: Using `cv2.initUndistortRectifyMap()` + `cv2.remap()` is 3-5x faster than `cv2.undistort()`
2. **Map caching**: Maps computed once at startup, reused for all frames
3. **OpenCV optimizations**: Uses SIMD, multi-threading internally
4. **Memory layout**: Maps use CV_32FC1 format for optimal performance

## Technical Details

### Undistortion Pipeline
1. **Calibration load**: Read camera matrix and distortion coefficients from YAML
2. **Optimal camera matrix**: Compute with `cv2.getOptimalNewCameraMatrix()`
3. **Map generation**: Create remap maps with `cv2.initUndistortRectifyMap()`
4. **Frame processing**: Apply with `cv2.remap(frame, map1, map2, INTER_LINEAR)`

### Calibration File Format
OpenCV YAML format (compatible with `cv2.FileStorage`):
```yaml
%YAML:1.0
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
```

## Backward Compatibility

- **No breaking changes**: Existing code continues to work
- **Default behavior**: Distortion correction enabled by default
- **Shared memory protocol**: Unchanged (no new fields/buffers)
- **Vision Module**: No changes required (receives corrected frames transparently)

## Future Enhancements

### Possible Improvements
1. **Dual buffer support**: Maintain separate raw/corrected buffers
2. **Runtime toggle**: Add API endpoint to enable/disable correction
3. **Calibration hot-reload**: Reload calibration without restarting process
4. **Adaptive quality**: Reduce undistortion quality under high CPU load
5. **ROI optimization**: Only undistort table region, not full frame

### Alternative Approaches
- **GPU acceleration**: Use CUDA/OpenCL for undistortion (cupy, opencv-cuda)
- **Partial undistortion**: Only undistort detection regions
- **Multi-resolution**: Generate multiple undistortion levels for different consumers

## Testing

### Manual Testing
1. Start Video Module with calibration enabled
2. Verify calibration loads successfully in logs
3. Check statistics show `undistortion_enabled=true`
4. Verify average undistortion time is reasonable (<5ms)

### Verification Steps
```bash
# Check logs for calibration loading
tail -f logs/video.log | grep -i calibration

# Monitor statistics
tail -f logs/video.log | grep "Stats:"

# Test with calibration disabled
# Edit config.json: "enable_distortion_correction": false
python -m backend.video
```

### Expected Log Output
```
VideoProcess initialized
Loading camera calibration from: backend/calibration_data/camera/camera_params.yaml
Camera calibration loaded successfully. Resolution: (3840, 2160), Error: 7.4328
Initializing camera...
First frame received: 1920x1080, channels=3
Initializing undistortion maps for 1920x1080...
Undistortion maps initialized successfully
Video Module process started successfully
Stats: uptime=10.0s, frames_captured=300, frames_written=300, fps=30.0,
       camera_fps=30.0, camera_dropped=0, write_counter=300,
       undistortion_enabled=true, frames_undistorted=300, avg_undistortion_time=2.15ms
```

## Troubleshooting

### Issue: Calibration file not found
**Solution**: Check path in config.json, ensure file exists
```bash
ls -la backend/calibration_data/camera/camera_params.yaml
```

### Issue: High undistortion time (>10ms)
**Solutions**:
- Check CPU usage, close other processes
- Verify OpenCV built with optimizations (SIMD, threading)
- Consider disabling correction for lower-end hardware

### Issue: Distortion correction not applied
**Check**:
1. `enable_distortion_correction: true` in config.json
2. Calibration file exists and loads successfully
3. Logs show "Undistortion maps initialized successfully"
4. Statistics show `undistortion_enabled=true`

## References

- **OpenCV Calibration**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **Camera Distortion**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Remap Performance**: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
