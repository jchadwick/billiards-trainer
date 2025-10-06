# Detector Benchmark Tool - Implementation Summary

## Overview

A comprehensive performance benchmarking tool has been created at `tools/benchmark_detectors.py` to compare YOLO vs OpenCV ball detection with detailed metrics, accuracy analysis, and visualization capabilities.

## What Was Implemented

### Core Features

1. **Dual Detector Benchmarking**
   - OpenCV traditional computer vision detection (always tested)
   - YOLO deep learning detection (optional, if model provided)
   - Fair comparison on identical frames

2. **Performance Metrics**
   - FPS (frames per second) - min, max, average, median
   - Latency (milliseconds) - min, max, average
   - Detection counts per frame
   - Detection variance (consistency metric)
   - Average confidence scores

3. **Accuracy Metrics** (when ground truth provided)
   - True Positives, False Positives, False Negatives
   - Precision, Recall, F1 Score
   - IoU-based matching (configurable threshold)
   - Per-frame accuracy tracking

4. **Comparison Analysis**
   - YOLO speedup factor (YOLO FPS / OpenCV FPS)
   - YOLO accuracy gain (F1 score difference)
   - Winner determination (based on accuracy or speed)

5. **Visualization**
   - Side-by-side comparison frames
   - OpenCV detections (green circles)
   - YOLO detections (blue circles)
   - Detection counts and frame numbers
   - Saved to output directory

6. **Output Formats**
   - JSON: Complete detailed metrics (`models/benchmark_results.json`)
   - Text: Human-readable report (`models/benchmark_report.txt`)
   - Console: Real-time progress and summary table
   - Images: Visualization frames (optional)

## File Structure

```
tools/
├── benchmark_detectors.py          # Main benchmark script (NEW)
├── README_BENCHMARK.md             # Comprehensive documentation (NEW)
├── example_ground_truth.json       # Example ground truth format (NEW)
├── compare_detectors.py            # Visual comparison tool (existing)
└── ...other tools...

models/
├── benchmark_results.json          # JSON output (generated)
└── benchmark_report.txt            # Text report (generated)

benchmark_output/                    # Visualization frames (optional)
└── frame_*.jpg
```

## Key Implementation Details

### Classes

1. **`DetectionMetrics`**: Tracks metrics for a single detector
   - Frame timing arrays
   - Detection count arrays
   - Accuracy counters (TP/FP/FN)
   - Calculated statistics (FPS, latency, precision, recall, F1)

2. **`BenchmarkResults`**: Complete comparison results
   - OpenCV metrics
   - YOLO metrics (optional)
   - Video metadata
   - Comparative analysis
   - Winner determination

3. **`DetectorBenchmark`**: Main benchmark framework
   - Video loading and processing
   - Detector initialization
   - Ground truth loading
   - Frame-by-frame benchmarking
   - Visualization generation
   - Results compilation

### Algorithms

1. **IoU Calculation**: Accurate Intersection over Union for circular objects
   - Handles partial overlap
   - Handles one circle inside another
   - Geometric formula for intersection area

2. **Detection Matching**: Hungarian-like matching algorithm
   - Greedy matching of detections to ground truth
   - IoU threshold-based matching (default 0.5)
   - Tracks matched/unmatched detections

3. **Statistical Analysis**: Comprehensive metrics calculation
   - FPS from frame times
   - Latency statistics (min/max/avg)
   - Detection variance for consistency
   - Precision/Recall/F1 from TP/FP/FN

## Usage Examples

### Basic Benchmark (OpenCV only)
```bash
python tools/benchmark_detectors.py --video test_video.mp4
```

### With YOLO Model
```bash
python tools/benchmark_detectors.py \
  --video test_video.mp4 \
  --yolo-model models/billiards_yolo.pt
```

### With Ground Truth + Visualization
```bash
python tools/benchmark_detectors.py \
  --video test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --ground-truth annotations.json \
  --visualize \
  --frames 100
```

### Full Featured Benchmark
```bash
python tools/benchmark_detectors.py \
  --video test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --ground-truth annotations.json \
  --visualize \
  --output-dir benchmark_results \
  --results-json results/benchmark.json \
  --results-text results/report.txt \
  --frames 500 \
  --verbose
```

## Ground Truth Format

JSON file with frame-indexed ball annotations:

```json
{
  "0": [
    {"x": 320.5, "y": 240.2, "radius": 20.1, "type": "cue"},
    {"x": 450.3, "y": 280.7, "radius": 19.8, "type": "solid"}
  ],
  "1": [
    {"x": 321.1, "y": 241.5, "radius": 20.0, "type": "cue"}
  ]
}
```

## Output Example

### Console Table
```
================================================================================
========================= DETECTOR BENCHMARK RESULTS ==========================
================================================================================

Metric                         OpenCV                 YOLO
------------------------------------------------------------------------
Total Frames Processed           1000                 1000
Total Detections                14523                14891
Avg Detections/Frame            14.52                14.89

Average FPS                     28.45                 15.23
Average Latency (ms)            35.15                 65.67

Precision                        0.912                 0.947
Recall                           0.883                 0.940
F1 Score                         0.897                 0.943

Winner: YOLO
```

### JSON Output
```json
{
  "opencv_metrics": {
    "detector_name": "OpenCV",
    "total_frames": 1000,
    "avg_fps": 28.45,
    "avg_latency_ms": 35.15,
    "precision": 0.912,
    "recall": 0.883,
    "f1_score": 0.897
  },
  "yolo_metrics": { ... },
  "yolo_speedup_factor": 0.54,
  "yolo_accuracy_gain": 0.046,
  "winner": "yolo"
}
```

## Comparison with Existing `compare_detectors.py`

| Feature | benchmark_detectors.py | compare_detectors.py |
|---------|------------------------|----------------------|
| Purpose | Performance benchmarking | Visual comparison |
| Output | Metrics, reports, JSON | Comparison video |
| Ground Truth | Yes (optional) | No |
| Accuracy Metrics | Yes (P/R/F1) | No |
| Performance Metrics | Detailed (FPS/latency) | Basic timing |
| Visualization | Optional frames | Real-time/video |
| Use Case | Quantitative analysis | Qualitative review |

**Recommendation**: Use both tools together:
1. Use `compare_detectors.py` for initial visual inspection
2. Use `benchmark_detectors.py` for detailed performance analysis

## Testing

The tool has been validated with:
- ✅ Syntax check (py_compile)
- ✅ Import verification (all dependencies available)
- ✅ Help output validation
- ✅ Error handling (missing video file)
- ✅ Example ground truth file created

## Dependencies

All dependencies already in the project:
- `opencv-python` (cv2)
- `numpy`
- `ultralytics` (optional, for YOLO)
- Backend vision modules (BallDetector, YOLODetector, models)

## Future Enhancements

Possible improvements:
1. Support for image sequences (not just videos)
2. Multi-model comparison (more than 2 detectors)
3. Real-time plotting with matplotlib
4. Export to CSV for spreadsheet analysis
5. Automated report generation with charts
6. Confidence threshold tuning recommendations
7. ROC curve analysis
8. Precision-recall curves

## Conclusion

The benchmark tool provides comprehensive, quantitative analysis of detector performance with:
- Complete performance metrics (FPS, latency, throughput)
- Accuracy metrics when ground truth available
- Visual comparison capabilities
- Multiple output formats (JSON, text, images)
- Fair, reproducible comparison methodology

This enables data-driven decisions about which detector to use in production based on specific requirements (accuracy vs speed, resource constraints, etc.).
