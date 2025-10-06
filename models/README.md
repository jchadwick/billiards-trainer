# Models Directory

This directory contains trained models for billiards ball detection and tracking.

## Model Format Requirements

**Preferred Format:** ONNX (`.onnx`)
- Cross-platform compatibility
- Optimized inference with ONNX Runtime
- No framework-specific dependencies

**Also Supported:** PyTorch (`.pt`, `.pth`)
- Native PyTorch inference
- Requires PyTorch installation

**Edge TPU Format:** TensorFlow Lite (`.tflite`)
- Optimized for Google Coral Edge TPU accelerators
- Ultra-low latency inference (typically 5-15ms per frame)
- Lower power consumption
- Requires Edge TPU compiled model

## Performance Targets

Models should meet the following requirements:
- **Frame Rate:** 30+ FPS on target hardware
- **File Size:** < 10MB (preferably < 5MB)
- **Latency:** < 33ms per frame inference time
- **Accuracy:** 95%+ ball detection accuracy on test set

## Model Naming Convention

Use descriptive names that indicate the model type and version:

```
{model_type}_{architecture}_{version}.{ext}

Examples:
- ball_detector_yolov8n_v1.onnx
- ball_detector_efficientdet_v2.pt
- cue_tracker_mobilenet_v1.onnx
```

## Expected Input/Output Format

### Ball Detection Model

**Input:**
- Shape: `(1, 3, H, W)` or `(1, H, W, 3)` depending on format
- Data type: `float32`
- Value range: [0.0, 1.0] normalized RGB
- Recommended resolution: 640x640 or 416x416

**Output:**
- Bounding boxes: `(N, 4)` - [x1, y1, x2, y2] or [cx, cy, w, h]
- Confidence scores: `(N, 1)` - [0.0, 1.0]
- Class labels: `(N, 1)` - integer class IDs
- Expected classes: 0=cue ball, 1-15=numbered balls

### Cue Detection Model

**Input:**
- Shape: `(1, 3, H, W)` or `(1, H, W, 3)` depending on format
- Data type: `float32`
- Value range: [0.0, 1.0] normalized RGB

**Output:**
- Line parameters: `(1, 4)` - [x1, y1, x2, y2] or [rho, theta]
- Confidence: `(1, 1)` - [0.0, 1.0]

## Adding Custom Models

### Method 1: API Upload (Recommended)

Upload custom ONNX models via the REST API:

```bash
# Upload a custom ONNX model
curl -X POST "http://localhost:8000/api/v1/vision/model/upload" \
  -F "file=@ball_detector_custom_v1.onnx" \
  -F "run_inference_test=true"

# Response includes:
# - Validation results (input/output shapes, opset version, etc.)
# - Inference test results (success, timing)
# - Model path and versioned filename

# List available models
curl "http://localhost:8000/api/v1/vision/model/info"

# Load a specific model
curl -X POST "http://localhost:8000/api/v1/vision/model/load?model_name=ball_detector_custom_v1_20250601_120000.onnx"
```

**Requirements for uploaded models:**
- **Format:** ONNX (.onnx) only
- **Architecture:** YOLOv8-compatible detection model
- **Input shape:** `[batch, 3, height, width]` (typically 640x640)
- **Output format:** YOLO detection format with bounding boxes, scores, and class IDs
- **Classes:** 19 classes (0-15: balls, 16: cue stick, 17: table, 18: pocket)

**Validation checks:**
1. Valid ONNX format (using `onnx.checker` if available)
2. Model metadata extraction (input/output shapes, opset version)
3. Successful test inference on a blank frame
4. Reasonable file size (< 100MB recommended)

**Automatic versioning:**
- Uploaded models are saved with timestamp: `{original_name}_{YYYYMMDD_HHMMSS}.onnx`
- Prevents overwriting existing models
- Maintains history of uploaded models

### Method 2: Manual Installation

1. **Train your model** using the training pipeline (see `training_runs/`)
2. **Export to ONNX** format:
   ```python
   import torch

   model.eval()
   dummy_input = torch.randn(1, 3, 640, 640)
   torch.onnx.export(
       model,
       dummy_input,
       "ball_detector_custom_v1.onnx",
       opset_version=11,
       input_names=['input'],
       output_names=['boxes', 'scores', 'labels']
   )
   ```
3. **Validate the model** meets performance targets
4. **Place the model file** in this directory
5. **Update configuration** in `config/vision.json` or environment variables:
   ```json
   {
     "ball_detection": {
       "model_path": "models/ball_detector_custom_v1.onnx",
       "model_type": "onnx"
     }
   }
   ```

## Training Instructions

See `training_runs/` directory for:
- Training scripts and notebooks
- Dataset preparation utilities
- Model evaluation metrics
- Hyperparameter tuning results

For detailed training documentation, refer to:
- `docs/training.md` (if available)
- `backend/vision/detection/README.md` (detector-specific docs)

## Model Optimization Tips

1. **Quantization:** Reduce model size and improve inference speed
   ```python
   # Example: INT8 quantization for ONNX
   from onnxruntime.quantization import quantize_dynamic

   quantize_dynamic(
       "model.onnx",
       "model_quantized.onnx",
       weight_type=QuantType.QInt8
   )
   ```

2. **Pruning:** Remove unnecessary weights
3. **Knowledge Distillation:** Train smaller model from larger teacher
4. **Architecture Selection:** Use mobile-optimized architectures
   - YOLOv8n (nano)
   - EfficientDet-Lite
   - MobileNet-based detectors

## Benchmarking

Test model performance before deployment:

```bash
# Run benchmark script
python scripts/benchmark_model.py models/ball_detector_v1.onnx

# Expected output:
# - Average FPS
# - Memory usage
# - Latency statistics
# - mAP on test set
```

## Current Models

| Model | Type | Size | FPS | mAP | Notes |
|-------|------|------|-----|-----|-------|
| (none yet) | - | - | - | - | Add models here |

## Troubleshooting

**Model not loading:**
- Check file path in configuration
- Verify ONNX Runtime or PyTorch is installed
- Validate model format with `onnx.checker.check_model()`

**Poor performance:**
- Check input preprocessing matches training
- Verify model was trained on similar data
- Consider model quantization or optimization

**Low FPS:**
- Profile inference time
- Check hardware acceleration (CUDA, CoreML)
- Consider lighter architecture
- Reduce input resolution

## Google Coral Edge TPU Support

### Overview

The billiards-trainer system supports hardware acceleration using Google Coral Edge TPU devices. Edge TPUs provide:
- **Ultra-low latency**: 5-15ms per frame (vs 50-100ms on CPU)
- **Power efficiency**: <2W power consumption
- **Consistent performance**: Dedicated ML accelerator without competing workloads

### Supported Devices

- **USB Accelerator**: Plug-and-play USB device for any Linux system
- **M.2 Accelerator**: PCIe/M.2 module for embedded systems
- **Dev Board**: Complete development platform with integrated TPU

### Model Conversion to Edge TPU

To use a YOLO model on Edge TPU, you must convert it to TensorFlow Lite format and compile it for Edge TPU:

#### Step 1: Export YOLO to TensorFlow Lite

```bash
# Using ultralytics (recommended)
yolo export model=yolov8n-pool.pt format=tflite imgsz=320

# This creates: yolov8n-pool_saved_model/yolov8n-pool_float32.tflite
```

**Important Notes:**
- Use smaller input sizes (320x320 or 416x416) for better TPU performance
- YOLOv8n (nano) is recommended for Edge TPU
- Larger models may not fit or may be slower on TPU

#### Step 2: Compile for Edge TPU

Install the Edge TPU Compiler:

```bash
# Install Edge TPU Compiler (Linux only)
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

Compile the model:

```bash
# Compile TFLite model for Edge TPU
edgetpu_compiler yolov8n-pool_float32.tflite

# This creates: yolov8n-pool_float32_edgetpu.tflite
# Move to models directory
mv yolov8n-pool_float32_edgetpu.tflite models/yolov8n-pool_edgetpu.tflite
```

**Compiler Output:**
The compiler will show:
- Model compatibility with Edge TPU
- Percentage of operations that will run on TPU (aim for >90%)
- Any unsupported operations (will fall back to CPU)
- Estimated inference time

#### Step 3: Verify Compilation

Check compilation results:

```bash
# View compilation log
cat yolov8n-pool_float32_edgetpu.log

# Look for:
# - "Number of operations that will run on Edge TPU: XX"
# - "Number of operations that will run on CPU: YY"
# Ideally, >90% of operations should be on Edge TPU
```

### Configuration for Edge TPU

#### Option 1: Using Configuration File

Create or use `config/vision_tpu_example.json`:

```json
{
  "vision": {
    "detection": {
      "detection_backend": "yolo",
      "yolo_device": "tpu",
      "yolo_model_path": "models/yolov8n-pool_edgetpu.tflite",
      "tpu_device_path": null,
      "yolo_confidence": 0.4,
      "yolo_nms_threshold": 0.45
    }
  }
}
```

**TPU Device Path Options:**
- `null` or omit: Auto-detect TPU device (recommended)
- `"usb"`: Force USB accelerator
- `"pcie"`: Force PCIe/M.2 accelerator
- `"/dev/bus/usb/001/002"`: Specific USB device path

#### Option 2: Programmatic Configuration

```python
from backend.vision.detection.yolo_detector import YOLODetector

# Create TPU-accelerated detector
detector = YOLODetector(
    model_path="models/yolov8n-pool_edgetpu.tflite",
    device="tpu",
    tpu_device_path=None,  # Auto-detect
    confidence=0.4,
    nms_threshold=0.45,
    auto_fallback=True  # Fall back to CPU if TPU unavailable
)

# Check if TPU is being used
info = detector.get_model_info()
print(f"Using TPU: {info['using_tpu']}")
print(f"TPU available: {info.get('tpu_available', False)}")
```

### Installing PyCoral (TPU Runtime)

The PyCoral library is required for TPU inference:

```bash
# For Linux (x86-64 or ARM64)
pip install pycoral

# Or install with specific Python version
pip3.9 install pycoral

# Verify installation
python -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

**Installation Notes:**
- PyCoral is only available for Linux (x86-64, ARM64)
- Requires Python 3.7-3.10 (check compatibility)
- Not available on macOS or Windows
- For development on non-Linux systems, use CPU/CUDA and deploy to Linux device with TPU

### Troubleshooting TPU

#### TPU Not Detected

```bash
# Check if TPU is connected and recognized
lsusb | grep "Global Unichip"

# Expected output (USB Accelerator):
# Bus 001 Device 005: ID 1a6e:089a Global Unichip Corp.

# List Edge TPU devices
python3 -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

#### Permission Issues

```bash
# Add udev rules for TPU access (USB Accelerator)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-edgetpu-accelerator.rules

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to plugdev group
sudo usermod -aG plugdev $USER

# Log out and log back in for group changes to take effect
```

#### Model Not Compatible

If the compiler reports low Edge TPU operation percentage:

1. **Use smaller model**: YOLOv8n instead of YOLOv8s/m/l
2. **Reduce input size**: Use 320x320 instead of 640x640
3. **Check for unsupported ops**: Review compiler log for unsupported operations
4. **Simplify model architecture**: Some custom layers may not be supported

#### Performance Issues

```python
# Check inference time
import time
detector = YOLODetector(model_path="models/yolov8n-pool_edgetpu.tflite", device="tpu")

start = time.time()
detections = detector.detect_balls(frame)
print(f"Inference time: {(time.time() - start) * 1000:.1f}ms")

# Expected: 5-15ms on Edge TPU, 50-100ms on CPU
```

### Performance Comparison

| Device | Model | Resolution | Inference Time | Power | Cost |
|--------|-------|------------|----------------|-------|------|
| Coral USB | YOLOv8n | 320x320 | ~8ms | <2W | $60 |
| Coral USB | YOLOv8n | 416x416 | ~12ms | <2W | $60 |
| CPU (x86) | YOLOv8n | 640x640 | ~80ms | 10-20W | - |
| CUDA (GTX 1660) | YOLOv8n | 640x640 | ~15ms | 120W | $200+ |

### Best Practices

1. **Model Size**: Use YOLOv8n (nano) for best TPU performance
2. **Input Resolution**: 320x320 or 416x416 for optimal speed/accuracy tradeoff
3. **Quantization**: TPU models are automatically quantized during compilation
4. **Batch Size**: Always use batch_size=1 on Edge TPU
5. **Preprocessing**: Minimize preprocessing overhead (resize, normalize) in Python
6. **Multiple TPUs**: Can use multiple USB accelerators for parallel processing

### Additional Resources

- **Coral Documentation**: https://coral.ai/docs/
- **Edge TPU Compiler**: https://coral.ai/docs/edgetpu/compiler/
- **PyCoral API**: https://coral.ai/docs/reference/py/
- **Model Optimization**: https://coral.ai/docs/edgetpu/models-intro/
- **Supported Operations**: https://coral.ai/docs/edgetpu/compiler/#supported-operations

## References

- ONNX: https://onnx.ai/
- YOLOv8: https://github.com/ultralytics/ultralytics
- ONNX Runtime: https://onnxruntime.ai/
- Google Coral: https://coral.ai/
- Edge TPU Compiler: https://coral.ai/docs/edgetpu/compiler/
