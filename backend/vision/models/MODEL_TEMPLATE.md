# Model Documentation Template

## Overview

This template provides a comprehensive structure for documenting machine learning models in the billiards trainer system.

---

## Model Architecture

### Base Architecture
- **Model**: YOLOv8-nano
- **Framework**: Ultralytics YOLOv8
- **Export Format**: ONNX
- **Input Size**: 640x640 (default)
- **Parameters**: ~3.2M
- **Model Size**: ~6MB (ONNX format)

### Architecture Details
```
YOLOv8n Architecture:
- Backbone: CSPDarknet with C2f modules
- Neck: PANet-style feature pyramid
- Head: Decoupled detection head
- Anchor-free detection
```

### Model Specifications
- **Input Shape**: (1, 3, 640, 640) - NCHW format
- **Output Shape**: Varies by number of classes
- **Normalization**: Values scaled to [0, 1]
- **Color Format**: RGB

---

## Training Dataset

### Dataset Composition
- **Total Images**: [FILL IN]
- **Training Set**: [FILL IN] images
- **Validation Set**: [FILL IN] images
- **Test Set**: [FILL IN] images

### Data Sources
- [List data sources, e.g., manual capture, synthetic generation, public datasets]

### Class Distribution
| Class ID | Class Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | [Class] | [Count] | [%] |
| 1 | [Class] | [Count] | [%] |
| ... | ... | ... | ... |

### Augmentation Strategy
- Random horizontal flip
- Random scaling (0.8-1.2x)
- Color jittering
- Mosaic augmentation
- [Add other augmentations]

### Data Quality
- **Annotation Format**: YOLO format (class x_center y_center width height)
- **Annotation Tool**: [e.g., LabelImg, Roboflow]
- **Quality Checks**: [Describe validation process]

---

## Performance Metrics

### Detection Performance
- **mAP@50**: [FILL IN] (mean Average Precision at IoU=0.5)
- **mAP@50-95**: [FILL IN] (mean Average Precision at IoU=0.5:0.95)
- **Precision**: [FILL IN]
- **Recall**: [FILL IN]

### Per-Class Performance
| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|---------|
| [Class] | [Value] | [Value] | [Value] |
| ... | ... | ... | ... |

### Runtime Performance
- **CPU Inference** (Intel i5/AMD Ryzen 5 equivalent):
  - FPS: [FILL IN]
  - Latency: [FILL IN] ms

- **GPU Inference** (NVIDIA RTX 3060 equivalent):
  - FPS: [FILL IN]
  - Latency: [FILL IN] ms

- **Edge Device** (Raspberry Pi 4, if applicable):
  - FPS: [FILL IN]
  - Latency: [FILL IN] ms

### Benchmark Conditions
- **Test Hardware**: [Specify hardware used for benchmarks]
- **Batch Size**: 1
- **Image Size**: 640x640
- **Runtime**: ONNX Runtime
- **Optimization**: [e.g., FP32, FP16, INT8]

---

## Usage Examples

### Python - Basic Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolov8n-pool.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Class: {class_id}, Conf: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
```

### Python - ONNX Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession('models/yolov8n-pool.onnx')

# Preprocess image
image = Image.open('path/to/image.jpg').resize((640, 640))
image_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: image_array})

# Process outputs
predictions = outputs[0]
# [Further processing based on model output format]
```

### Integration with Vision System

```python
from backend.vision.detection.balls import BallDetector

# Initialize detector
detector = BallDetector(model_path='models/yolov8n-pool.onnx')

# Detect balls in frame
frame = cv2.imread('path/to/image.jpg')
detections = detector.detect(frame)

# Process detections
for detection in detections:
    ball_type = detection['class']
    confidence = detection['confidence']
    bbox = detection['bbox']
    print(f"Detected {ball_type} with confidence {confidence:.2f}")
```

---

## Class Names and IDs Mapping

### Standard Pool Ball Classes

| Class ID | Class Name | Description | Color/Pattern |
|----------|------------|-------------|---------------|
| 0 | cue_ball | White cue ball | White |
| 1 | ball_1 | Solid yellow | Yellow |
| 2 | ball_2 | Solid blue | Blue |
| 3 | ball_3 | Solid red | Red |
| 4 | ball_4 | Solid purple | Purple |
| 5 | ball_5 | Solid orange | Orange |
| 6 | ball_6 | Solid green | Green |
| 7 | ball_7 | Solid maroon | Maroon |
| 8 | ball_8 | Black eight ball | Black |
| 9 | ball_9 | Striped yellow | Yellow stripe |
| 10 | ball_10 | Striped blue | Blue stripe |
| 11 | ball_11 | Striped red | Red stripe |
| 12 | ball_12 | Striped purple | Purple stripe |
| 13 | ball_13 | Striped orange | Orange stripe |
| 14 | ball_14 | Striped green | Green stripe |
| 15 | ball_15 | Striped maroon | Maroon stripe |
| 16 | cue_tip | Cue stick tip | N/A |

### Class Groupings

**Solids**: Class IDs 1-7
**Stripes**: Class IDs 9-15
**Special**: Class IDs 0 (cue ball), 8 (eight ball)
**Cue**: Class ID 16

---

## Hardware Requirements

### Minimum Requirements
- **CPU**: Dual-core 2.0 GHz (x86_64)
- **RAM**: 2GB
- **Storage**: 50MB for model files
- **OS**: Linux, Windows, macOS

### Recommended Requirements
- **CPU**: Quad-core 2.5 GHz or better
- **RAM**: 4GB or more
- **GPU**: Optional but recommended for real-time performance
- **Storage**: 100MB for model files and cache

### Supported Hardware
- **Desktop/Laptop**: Intel/AMD x86_64 processors
- **GPU**: NVIDIA GPUs with CUDA support (optional)
- **Edge Devices**: Raspberry Pi 4 (4GB+ RAM), Jetson Nano

### Software Dependencies
```
Python >= 3.8
numpy >= 1.20.0
opencv-python >= 4.5.0
onnxruntime >= 1.12.0
ultralytics >= 8.0.0 (for training/export)
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Low detection accuracy
**Symptoms**: Missing balls, false positives, incorrect classifications

**Solutions**:
- Verify lighting conditions match training data
- Check image quality and resolution
- Adjust confidence threshold (default: 0.25)
- Ensure proper camera calibration
- Verify model file integrity

#### Issue: Slow inference speed
**Symptoms**: Low FPS, high latency

**Solutions**:
- Use GPU acceleration if available
- Reduce input image size (e.g., 320x320 instead of 640x640)
- Enable model quantization (INT8)
- Use ONNX Runtime optimizations
- Check CPU usage and close unnecessary processes

#### Issue: Model fails to load
**Symptoms**: File not found, corrupted model errors

**Solutions**:
- Verify model file path is correct
- Check file permissions
- Re-download or re-export model
- Verify ONNX runtime version compatibility

#### Issue: Inconsistent results
**Symptoms**: Detection varies significantly between similar frames

**Solutions**:
- Implement temporal smoothing
- Use tracking to maintain consistency
- Increase confidence threshold
- Check for motion blur or camera shake

### Performance Optimization

1. **Batch Processing**: Process multiple frames in batches when possible
2. **ROI Detection**: Only process relevant regions of interest
3. **Frame Skipping**: Skip frames during periods of no motion
4. **Model Quantization**: Use INT8 quantization for faster inference
5. **Hardware Acceleration**: Enable GPU or specialized accelerators

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Visualize detections
import cv2
for detection in detections:
    bbox = detection['bbox']
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
cv2.imshow('Detections', frame)
cv2.waitKey(0)

# Check model outputs
print(f"Output shape: {outputs[0].shape}")
print(f"Output range: [{outputs[0].min()}, {outputs[0].max()}]")
```

---

## Version History

### Version 1.0.0 (YYYY-MM-DD)
- Initial release
- [Number] classes
- mAP@50: [Value]
- Trained on [Dataset size] images

### Version 0.9.0 (YYYY-MM-DD)
- Beta release
- Known issues: [List issues]

### Version 0.5.0 (YYYY-MM-DD)
- Alpha release
- Initial training results

---

## Training Instructions

### Prerequisites
```bash
pip install ultralytics
```

### Training Command
```bash
yolo detect train \
    data=path/to/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=yolov8n-pool
```

### Export to ONNX
```bash
yolo export \
    model=runs/detect/yolov8n-pool/weights/best.pt \
    format=onnx \
    imgsz=640
```

### Validation
```bash
yolo detect val \
    model=runs/detect/yolov8n-pool/weights/best.pt \
    data=path/to/dataset.yaml
```

---

## License and Attribution

- **Model License**: [Specify license, e.g., MIT, Apache 2.0]
- **Base Architecture**: YOLOv8 by Ultralytics (AGPL-3.0)
- **Dataset License**: [Specify if applicable]

---

## Contact and Support

For issues, questions, or contributions:
- **Repository**: [GitHub URL]
- **Issues**: [Issue tracker URL]
- **Documentation**: [Docs URL]

---

## References

1. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
2. ONNX: https://onnx.ai/
3. [Additional references]
