# Detector Benchmark Tool

Performance benchmarking tool for comparing YOLO vs OpenCV ball detection.

## Overview

`benchmark_detectors.py` provides comprehensive performance analysis comparing:
- **YOLO** (YOLOv8) deep learning detection
- **OpenCV** traditional computer vision detection

The tool measures FPS, latency, accuracy, and detection quality to help you choose the best detector for your use case.

## Quick Start

### Basic Benchmark (OpenCV only)
```bash
python tools/benchmark_detectors.py --video path/to/test_video.mp4
```

### With YOLO Model
```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt
```

### Limited Frames + Visualization
```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --frames 100 \
  --visualize \
  --output-dir benchmark_output
```

### With Ground Truth Annotations
```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --ground-truth annotations.json \
  --visualize
```

## Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--video` | Yes | Path to test video file or image sequence |
| `--yolo-model` | No | Path to YOLO model file (.pt or .onnx) |
| `--frames` | No | Number of frames to test (default: all frames) |
| `--ground-truth` | No | Path to ground truth annotations JSON |
| `--visualize` | No | Generate side-by-side comparison visualizations |
| `--output-dir` | No | Directory to save visualization frames (default: benchmark_output) |
| `--results-json` | No | Path to save JSON results (default: models/benchmark_results.json) |
| `--results-text` | No | Path to save text report (default: models/benchmark_report.txt) |
| `--verbose` / `-v` | No | Enable verbose logging |

## Ground Truth Format

Ground truth annotations should be in JSON format:

```json
{
  "0": [
    {"x": 320.5, "y": 240.2, "radius": 20.1, "type": "cue"},
    {"x": 450.3, "y": 280.7, "radius": 19.8, "type": "solid"}
  ],
  "1": [
    {"x": 321.1, "y": 241.5, "radius": 20.0, "type": "cue"},
    {"x": 455.2, "y": 282.3, "radius": 19.9, "type": "solid"}
  ]
}
```

Where:
- Keys are frame numbers (as strings)
- Values are arrays of ball annotations
- Each annotation has: `x`, `y`, `radius`, and optionally `type`

## Output Files

### JSON Results (`models/benchmark_results.json`)
Complete benchmark data including:
- Frame-by-frame timing measurements
- Detection counts per frame
- Accuracy metrics (if ground truth provided)
- Statistical summaries

### Text Report (`models/benchmark_report.txt`)
Human-readable summary table with:
- Performance metrics (FPS, latency)
- Detection statistics
- Accuracy metrics (precision, recall, F1)
- Comparative analysis

### Visualization Frames
Side-by-side comparison images showing:
- Left: OpenCV detections (green circles)
- Right: YOLO detections (blue circles)
- Detection counts and frame numbers

## Metrics Explained

### Performance Metrics
- **FPS**: Frames processed per second (higher is better)
- **Latency**: Processing time per frame in milliseconds (lower is better)
- **Avg Detections/Frame**: Average number of balls detected per frame
- **Detection Variance**: Consistency of detection count (lower is better)

### Accuracy Metrics (requires ground truth)
- **True Positives (TP)**: Correct detections matching ground truth
- **False Positives (FP)**: Incorrect detections (ghosts)
- **False Negatives (FN)**: Missed balls that should have been detected
- **Precision**: TP / (TP + FP) - What fraction of detections are correct?
- **Recall**: TP / (TP + FN) - What fraction of balls are detected?
- **F1 Score**: Harmonic mean of precision and recall (overall accuracy)

### Comparative Metrics
- **YOLO Speedup Factor**: YOLO FPS / OpenCV FPS (>1.0 means YOLO is faster)
- **YOLO Accuracy Gain**: YOLO F1 - OpenCV F1 (positive means YOLO is more accurate)
- **Winner**: Overall best detector based on accuracy (or speed if no ground truth)

## Example Output

```
================================================================================
========================= DETECTOR BENCHMARK RESULTS ==========================
================================================================================

Test Video: test_footage.mp4
Video Properties: 1000 frames @ 30.00 FPS
Benchmark Date: 2025-10-06T16:30:00
Ground Truth: Available

--------------------------------------------------------------------------------
-------------------------- PERFORMANCE METRICS ---------------------------------
--------------------------------------------------------------------------------

Metric                         OpenCV                 YOLO
------------------------------------------------------------------------
Total Frames Processed           1000                 1000
Total Detections                14523                14891
Avg Detections/Frame            14.52                14.89
Detection Variance               2.34                 1.87

Average FPS                     28.45                 15.23
Min FPS                         21.12                 12.34
Max FPS                         35.67                 18.90
Median FPS                      28.90                 15.45

Average Latency (ms)            35.15                 65.67
Min Latency (ms)                28.05                 52.91
Max Latency (ms)                47.35                 81.05

Average Confidence               0.823                 0.912

--------------------------------------------------------------------------------
------------------ ACCURACY METRICS (vs Ground Truth) -------------------------
--------------------------------------------------------------------------------

Metric                         OpenCV                 YOLO
------------------------------------------------------------------------
True Positives                  13245                14102
False Positives                  1278                  789
False Negatives                  1755                  898

Precision                        0.912                 0.947
Recall                           0.883                 0.940
F1 Score                         0.897                 0.943

--------------------------------------------------------------------------------
------------------------- COMPARATIVE ANALYSIS ---------------------------------
--------------------------------------------------------------------------------

YOLO Speedup Factor: 0.54x
  → OpenCV is 1.87x FASTER than YOLO

YOLO Accuracy Gain: +0.046 (F1 score difference)
  → YOLO is MORE ACCURATE than OpenCV

Winner: YOLO

================================================================================
```

## Interpreting Results

### When to use YOLO:
- Accuracy is critical
- You have a trained model
- Processing time is not critical
- GPU is available for faster inference

### When to use OpenCV:
- Real-time performance is critical
- No trained model available
- Running on CPU-only systems
- Simpler deployment requirements

### Tie scenarios:
- Similar accuracy (F1 difference < 0.05)
- Trade-off between speed and accuracy acceptable
- May want to use hybrid approach

## Creating Ground Truth

To create ground truth annotations for accuracy testing:

1. Use the annotation tool (if available) or manually annotate
2. Export to JSON format (see format above)
3. Ensure frame numbers match video frames (0-indexed)
4. Include all visible balls in each frame

## Tips for Best Results

1. **Use representative test videos**: Include various lighting, angles, and game states
2. **Test sufficient frames**: At least 100 frames for statistical significance
3. **Compare like-for-like**: Use same video, same settings
4. **Test multiple videos**: Different tables, lighting conditions
5. **Monitor resource usage**: Check CPU/GPU utilization during benchmark

## Troubleshooting

### "Model file not found"
- Check that YOLO model path is correct
- Ensure model file has .pt or .onnx extension

### "Video file not found"
- Verify video path is correct
- Check video file is readable (not corrupted)

### Low FPS / High latency
- This is expected for accurate measurement
- Benchmark runs single-threaded for fairness
- Real-world performance may be higher with optimizations

### High variance in detections
- May indicate inconsistent lighting in video
- Could suggest detector needs tuning
- Check visualization frames to understand why

### Memory errors
- Reduce `--frames` to process fewer frames
- Disable `--visualize` to save memory
- Close other applications

## Advanced Usage

### Batch Testing Multiple Videos
```bash
for video in videos/*.mp4; do
  python tools/benchmark_detectors.py \
    --video "$video" \
    --yolo-model models/billiards_yolo.pt \
    --frames 100 \
    --results-json "results/$(basename $video .mp4)_benchmark.json"
done
```

### Comparing Multiple Models
```bash
# Test model v1
python tools/benchmark_detectors.py \
  --video test.mp4 \
  --yolo-model models/yolo_v1.pt \
  --results-json results/v1_benchmark.json

# Test model v2
python tools/benchmark_detectors.py \
  --video test.mp4 \
  --yolo-model models/yolo_v2.pt \
  --results-json results/v2_benchmark.json

# Compare results
python -c "
import json
v1 = json.load(open('results/v1_benchmark.json'))
v2 = json.load(open('results/v2_benchmark.json'))
print(f'V1 F1: {v1[\"yolo_metrics\"][\"f1_score\"]:.3f}')
print(f'V2 F1: {v2[\"yolo_metrics\"][\"f1_score\"]:.3f}')
"
```

## See Also

- `train_yolo.py` - Train custom YOLO models
- `dataset_creator.py` - Create training datasets
- `video_debugger.py` - Debug detection issues
- `annotation_validator.py` - Validate annotation quality
