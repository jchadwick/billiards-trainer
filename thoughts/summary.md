# ONNX Runtime Opset 22 Compatibility Fix

## Problem
`make run` was failing with:
```
ONNXRuntimeError: ONNX Runtime only *guarantees* support for models stamped with
official released onnx opset versions. Opset 22 is under development and support
for this is limited.
```

The YOLO model (`models/yolov8n-billiards.onnx`) was exported with opset 22, but
the installed ONNX Runtime version (1.18.1) only supported up to opset 21.

## Solution

### 1. Updated ONNX Runtime
Upgraded from 1.18.1 to 1.23.1:
- `onnxruntime`: 1.18.1 → 1.23.1 (supports opset 22)
- `onnx`: 1.16.1 → 1.17.0

### 2. Added Auto-Detection for Hardware Acceleration
Implemented automatic device detection in `backend/vision/detection/yolo_detector.py`:

**Priority Order:**
1. CUDA GPU (if available)
2. CoreML (Apple Silicon with ONNX model) ← **Active on this Mac**
3. MPS (Apple Silicon with PyTorch model)
4. CPU (fallback)

**Key Features:**
- No configuration required - automatically detects best device
- Platform-aware (checks for Apple Silicon + ONNX model combo)
- Verifies CoreML provider availability before using
- Logs detection decisions for transparency

### 3. Updated Configuration
Changed `yolo_device` from specific values to `"auto"` in `config.json`:
```json
{
  "vision": {
    "detection": {
      "yolo_device": "auto"  // Auto-detects best device
    }
  }
}
```

### 4. Updated Requirements
Added ONNX dependencies to `backend/requirements.txt`:
```
# YOLO/ONNX for object detection
onnxruntime>=1.23.0  # Has CoreML support for Apple Silicon
onnx>=1.17.0
ultralytics>=8.0.0
# Note: Use device="auto" in config for hardware acceleration on Mac
```

## Results

### Hardware Acceleration on Apple Silicon
- **CoreML Execution Provider**: Active
- **Nodes on CoreML**: 230/233 (98.7%)
- **Nodes on CPU**: 3/233 (1.3%)
- **Startup time**: 0.68s

### Device Detection
```
INFO:backend.vision.detection.yolo_detector:Apple Silicon detected with ONNX model - using CoreML
INFO:backend.vision.detection.yolo_detector:Auto-detected device: coreml
INFO:backend.vision.detection.yolo_detector:CoreML execution provider is available
```

## Benefits

1. **Cross-platform**: Works on Mac (CoreML), Linux/Windows (CUDA), and CPU-only systems
2. **Zero configuration**: Automatically selects best device
3. **Future-proof**: ONNX Runtime 1.23.1 supports latest opsets
4. **Performance**: ~99% hardware acceleration on Apple Silicon

## Testing

```bash
# Verify ONNX Runtime version and providers
python3 -c "import onnxruntime as ort; print('Version:', ort.__version__); print('Providers:', ort.get_available_providers())"

# Test CoreML with opset 22 model
python3 -c "import onnxruntime as ort; sess = ort.InferenceSession('models/yolov8n-billiards.onnx', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']); print('Providers:', sess.get_providers())"

# Run application
make run
```

## Files Modified

1. `backend/requirements.txt` - Added ONNX dependencies with versions
2. `backend/vision/detection/yolo_detector.py` - Added `_auto_detect_device()` method
3. `config.json` - Changed `yolo_device` to `"auto"`

## Backward Compatibility

The auto-detection still respects explicit device settings:
- `"auto"` or `null` → Auto-detect
- `"cpu"` → Force CPU
- `"cuda"` → Force CUDA
- `"coreml"` → Force CoreML (Mac only)
- `"mps"` → Force MPS (Mac only)
- `"tpu"` → Use Google Coral Edge TPU
