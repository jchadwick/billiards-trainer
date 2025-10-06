# Benchmark Tool - Quick Start Guide

## Installation

No additional dependencies needed - all requirements already in project.

```bash
cd /Users/jchadwick/code/billiards-trainer
```

## Basic Usage

### 1. Test OpenCV Detector Only

```bash
python tools/benchmark_detectors.py --video path/to/test_video.mp4
```

This will:
- Process all frames in the video
- Benchmark OpenCV detector performance
- Print results to console
- Save detailed metrics to `models/benchmark_results.json`
- Save text report to `models/benchmark_report.txt`

### 2. Compare YOLO vs OpenCV

```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt
```

This will:
- Benchmark both detectors on the same frames
- Compare performance (FPS, latency)
- Determine winner based on speed and consistency
- Show detailed comparison table

### 3. With Visualization

```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --visualize \
  --frames 100
```

This will:
- Process first 100 frames only
- Generate side-by-side comparison images
- Save visualization frames to `benchmark_output/`
- Show OpenCV (green) vs YOLO (blue) detections

### 4. With Ground Truth Accuracy

```bash
python tools/benchmark_detectors.py \
  --video path/to/test_video.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --ground-truth tools/example_ground_truth.json \
  --frames 100
```

This will:
- Compare detections against ground truth
- Calculate Precision, Recall, F1 Score
- Report True Positives, False Positives, False Negatives
- Determine winner based on accuracy (not just speed)

## Reading Results

### Console Output

Look for the summary table:

```
Metric                         OpenCV                 YOLO
------------------------------------------------------------------------
Average FPS                     28.45                 15.23
Average Latency (ms)            35.15                 65.67
F1 Score                         0.897                 0.943
Winner: YOLO
```

Key takeaways:
- **Higher FPS is better** (faster processing)
- **Lower latency is better** (faster per-frame)
- **Higher F1 score is better** (more accurate)
- **Winner** shows which detector is recommended

### JSON Results

Open `models/benchmark_results.json` for detailed data:

```json
{
  "opencv_metrics": {
    "avg_fps": 28.45,
    "f1_score": 0.897,
    ...
  },
  "yolo_metrics": {
    "avg_fps": 15.23,
    "f1_score": 0.943,
    ...
  },
  "winner": "yolo"
}
```

### Text Report

Open `models/benchmark_report.txt` for formatted report you can share.

## Interpreting Results

### Performance vs Accuracy Trade-off

**OpenCV is faster but less accurate:**
- Use when: Real-time performance critical, CPU-only
- Example: Live game tracking, resource-constrained systems

**YOLO is slower but more accurate:**
- Use when: Accuracy critical, GPU available
- Example: Training analysis, shot replay analysis

### Speedup Factor

```
YOLO Speedup Factor: 0.54x
→ OpenCV is 1.87x FASTER than YOLO
```

- `> 1.0`: YOLO is faster
- `< 1.0`: OpenCV is faster
- Factor shows how many times faster

### Accuracy Gain

```
YOLO Accuracy Gain: +0.046 (F1 score difference)
→ YOLO is MORE ACCURATE than OpenCV
```

- `> 0.05`: YOLO significantly more accurate
- `< -0.05`: OpenCV significantly more accurate
- `-0.05 to 0.05`: Similar accuracy (tie)

## Common Use Cases

### 1. Model Selection

**Question**: Should I use YOLO or OpenCV for my application?

**Answer**: Run benchmark with representative video:
```bash
python tools/benchmark_detectors.py \
  --video typical_game_footage.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --frames 200
```

Check winner and trade-offs in results.

### 2. Model Comparison

**Question**: Which YOLO model version is better?

**Answer**: Run benchmark for each model:
```bash
# Test model v1
python tools/benchmark_detectors.py \
  --video test.mp4 \
  --yolo-model models/yolo_v1.pt \
  --results-json results/v1.json

# Test model v2
python tools/benchmark_detectors.py \
  --video test.mp4 \
  --yolo-model models/yolo_v2.pt \
  --results-json results/v2.json

# Compare F1 scores
python -c "
import json
v1 = json.load(open('results/v1.json'))
v2 = json.load(open('results/v2.json'))
print(f'V1: {v1[\"yolo_metrics\"][\"avg_fps\"]:.1f} FPS')
print(f'V2: {v2[\"yolo_metrics\"][\"avg_fps\"]:.1f} FPS')
"
```

### 3. Performance Optimization

**Question**: Did my code optimization improve performance?

**Answer**: Benchmark before and after:
```bash
# Before optimization
python tools/benchmark_detectors.py --video test.mp4 --results-json before.json

# Make code changes...

# After optimization
python tools/benchmark_detectors.py --video test.mp4 --results-json after.json

# Compare
python -c "
import json
before = json.load(open('before.json'))
after = json.load(open('after.json'))
speedup = after['opencv_metrics']['avg_fps'] / before['opencv_metrics']['avg_fps']
print(f'Speedup: {speedup:.2f}x')
"
```

## Tips

1. **Use at least 100 frames** for statistical significance
2. **Test on diverse videos** (different lighting, angles)
3. **Include ground truth** for accuracy metrics when possible
4. **Enable visualization** on small sample to verify correctness
5. **Save results** with descriptive names for comparison

## Troubleshooting

### "Video file not found"
- Check path is correct
- Use absolute path if relative path fails

### "YOLO model not found"
- Ensure model file exists
- Check file extension is .pt or .onnx

### Low FPS / High latency
- This is expected during benchmarking
- Tool runs single-threaded for fair comparison
- Real-world performance may be higher

### Memory issues
- Use `--frames 100` to limit frames processed
- Disable `--visualize` to save memory

## Next Steps

- See `README_BENCHMARK.md` for comprehensive documentation
- See `BENCHMARK_SUMMARY.md` for implementation details
- Use `compare_detectors.py` for visual comparison
- Create ground truth with annotation tools

## Example Workflow

```bash
# 1. Quick visual check
python tools/compare_detectors.py --video test.mp4 --detector both --display

# 2. Detailed benchmark
python tools/benchmark_detectors.py \
  --video test.mp4 \
  --yolo-model models/billiards_yolo.pt \
  --frames 200 \
  --visualize

# 3. Review results
cat models/benchmark_report.txt

# 4. Check visualization
open benchmark_output/frame_000050.jpg
```

This workflow gives you both qualitative (visual) and quantitative (metrics) comparison!
