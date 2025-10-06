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

## References

- ONNX: https://onnx.ai/
- YOLOv8: https://github.com/ultralytics/ultralytics
- ONNX Runtime: https://onnxruntime.ai/
