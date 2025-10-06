# GPU Acceleration Implementation for Billiards Trainer

## Overview

This document describes the implementation of GPU hardware acceleration using Intel VAAPI for the billiards trainer system.

## Implementation Summary

### 1. GPU Acceleration Utilities Module (`backend/vision/gpu_utils.py`)

Created a comprehensive GPU acceleration module that provides:

- **VAAPI Configuration**: Automatic setup of environment variables for Intel hardware acceleration
- **OpenCL Support**: GPU-accelerated image processing using OpenCV's OpenCL backend
- **Hardware Detection**: Automatic detection and fallback when GPU is unavailable
- **GPU-Accelerated Operations**:
  - Image resizing (`resize`)
  - Color space conversion (`cvt_color`)
  - Gaussian blur (`gaussian_blur`)
  - Bilateral filtering (`bilateral_filter`)
  - Median blur (`median_blur`)
  - Morphological operations (`morphology_ex`)
  - 2D convolution filters (`filter_2d`)
  - Perspective transformations (`warp_perspective`)

**Key Features**:
- Uses `cv2.UMat` (Unified Memory) for efficient GPU memory management
- Automatic CPU fallback if GPU operations fail
- Detailed logging of GPU capabilities
- Thread-safe singleton pattern

### 2. Enhanced Preprocessing Module (`backend/vision/preprocessing.py`)

Updated the image preprocessing pipeline to leverage GPU acceleration:

- **GPU Initialization**: Automatically initializes GPU accelerator when `enable_gpu` config is true
- **GPU-Accelerated Operations**: All major preprocessing operations now use GPU when available:
  - Frame resizing
  - Color space conversions (BGR↔RGB, HSV, LAB, YUV, GRAY)
  - Noise reduction (Gaussian, bilateral, median)
  - Morphological operations (opening, closing)
  - Sharpening filters
- **Statistics Tracking**: Added `gpu_enabled` flag to preprocessing statistics

### 3. Configuration Updates (`backend/config/default.json`)

- Changed `vision.processing.use_gpu` from `false` to `true` by default
- System will now attempt to use GPU acceleration automatically

### 4. Environment Configuration (`dist/run.sh`)

Added VAAPI environment variables to startup script:

```bash
export LIBVA_DRIVER_NAME=iHD
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;vaapi|hwaccel_device;/dev/dri/renderD128"
export LIBVA_MESSAGING_LEVEL=1
```

**Environment Variables**:
- `LIBVA_DRIVER_NAME=iHD`: Uses Intel's iHD VAAPI driver for GPU acceleration
- `OPENCV_FFMPEG_CAPTURE_OPTIONS`: Enables hardware-accelerated video decoding via FFmpeg
- `LIBVA_MESSAGING_LEVEL=1`: Reduces verbose VAAPI logging

### 5. Application Startup (`backend/main.py`)

Added early VAAPI configuration before any video operations:
- Calls `configure_vaapi_env()` at application startup
- Ensures environment is configured before OpenCV initialization
- Graceful fallback if configuration fails

## How It Works

### GPU Detection and Initialization

1. On application startup, `configure_vaapi_env()` sets required environment variables
2. When vision preprocessing is initialized, `GPUAccelerator` checks for:
   - OpenCL support in OpenCV (`cv2.ocl.haveOpenCL()`)
   - VAAPI driver availability (via `LIBVA_DRIVER_NAME` environment variable)
3. If GPU is available, OpenCL is enabled globally (`cv2.ocl.setUseOpenCL(True)`)

### GPU-Accelerated Operations

When GPU acceleration is enabled:

1. **Upload**: Image data is uploaded to GPU memory using `cv2.UMat`
2. **Process**: OpenCV operations are performed on GPU
3. **Download**: Results are downloaded back to CPU memory using `.get()`

Example flow for image resize:
```python
umat = cv2.UMat(frame)              # Upload to GPU
resized = cv2.resize(umat, size)     # Process on GPU
result = resized.get()               # Download from GPU
```

### Automatic Fallback

If any GPU operation fails:
- Exception is caught and logged
- Operation automatically falls back to CPU implementation
- Processing continues without interruption

## Performance Benefits

GPU acceleration provides significant performance improvements for:

1. **High-Resolution Video**: 1920x1080 frame processing
2. **Real-Time Processing**: 30+ FPS video streams
3. **Image Filtering**: Gaussian blur, bilateral filtering
4. **Color Conversions**: BGR↔HSV, LAB conversions for ball detection
5. **Morphological Operations**: Opening/closing for noise reduction

Expected speedup: 2-5x faster for most operations when GPU is available.

## Verification

### Check GPU Status

The system logs GPU availability at startup:
```
INFO: GPU acceleration is AVAILABLE
INFO: OpenCL device: Intel(R) UHD Graphics
INFO: GPU acceleration ENABLED for preprocessing
```

### Monitor GPU Usage

On the target system, monitor GPU usage:
```bash
# Check VAAPI driver
vainfo

# Monitor GPU usage
intel_gpu_top

# Check OpenCL devices
clinfo
```

### Processing Statistics

GPU status is included in preprocessing statistics:
```json
{
  "frames_processed": 1000,
  "avg_processing_time": 8.5,
  "gpu_enabled": true
}
```

## Hardware Requirements

### Minimum Requirements
- Intel GPU with VAAPI support (HD Graphics 4000 or newer)
- Linux system with `/dev/dri/renderD128` device
- libva and libva-intel-driver packages installed

### Recommended
- Intel UHD Graphics or newer
- At least 256MB GPU memory (shared)
- Ubuntu 20.04+ or similar with recent kernel

## Troubleshooting

### GPU Not Detected

If GPU acceleration is not working:

1. **Check VAAPI driver**:
   ```bash
   vainfo
   ```
   Should show Intel driver and supported profiles.

2. **Check device permissions**:
   ```bash
   ls -l /dev/dri/renderD128
   ```
   User should have read/write access.

3. **Verify OpenCL**:
   ```bash
   clinfo | grep "Device Name"
   ```
   Should list Intel GPU.

4. **Check logs**:
   ```bash
   grep -i "gpu\|vaapi\|opencl" logs/*.log
   ```

### Fallback to CPU

If you see warnings like "GPU acceleration requested but not available":
- System will continue working with CPU processing
- Performance will be slower but functionality is preserved
- Check GPU requirements and drivers

### Permission Issues

If you see "Permission denied" for `/dev/dri/renderD128`:
```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
# Log out and back in
```

## Future Enhancements

Potential improvements:
1. **CUDA Support**: Add NVIDIA GPU support using CUDA backend
2. **Batch Processing**: Process multiple frames simultaneously on GPU
3. **Advanced Filters**: GPU implementation of advanced detection algorithms
4. **Memory Optimization**: Keep data on GPU across operations
5. **Performance Profiling**: Detailed GPU vs CPU timing comparisons

## References

- [OpenCV OpenCL Support](https://docs.opencv.org/4.x/d0/d1e/tutorial_js_intro.html)
- [Intel VAAPI Documentation](https://github.com/intel/libva)
- [FFmpeg VAAPI](https://trac.ffmpeg.org/wiki/Hardware/VAAPI)
- [OpenCL Programming Guide](https://www.khronos.org/opencl/)

## Testing

To verify GPU acceleration is working on the target environment:

1. Start the application: `./run.sh`
2. Check startup logs for GPU messages
3. Monitor GPU usage: `intel_gpu_top`
4. Compare processing times with GPU enabled vs disabled

---

**Status**: ✅ Implemented and deployed to target environment (192.168.1.31)
**Date**: 2025-10-02
**Author**: Claude Code
