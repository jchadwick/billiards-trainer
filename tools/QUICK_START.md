# Quick Start Guide - Detector Comparison Tool

## 5-Minute Quick Start

### 1. Basic Comparison (No YOLO model)

If you don't have a YOLO model, you can still test OpenCV detector:

```bash
python tools/compare_detectors.py \
  --video path/to/your/video.mp4 \
  --detector opencv \
  --display
```

### 2. With YOLO Model

If you have a trained YOLO model:

```bash
python tools/compare_detectors.py \
  --video path/to/your/video.mp4 \
  --detector both \
  --yolo-model path/to/model.pt \
  --display
```

### 3. Save Comparison Video

```bash
python tools/compare_detectors.py \
  --video path/to/your/video.mp4 \
  --detector both \
  --yolo-model path/to/model.pt \
  --output comparison.mp4
```

### 4. With Background Subtraction

For better OpenCV detection, provide an empty table image:

```bash
python tools/compare_detectors.py \
  --video path/to/your/video.mp4 \
  --detector opencv \
  --background path/to/empty_table.jpg \
  --display
```

## Common Scenarios

### Scenario 1: Testing Detection Quality

You want to see how well detectors work:

```bash
python tools/compare_detectors.py \
  --video test_shot.mp4 \
  --detector both \
  --yolo-model models/billiards.pt \
  --background empty_table.jpg \
  --display
```

**What to look for:**
- Green boxes = OpenCV detections
- Blue boxes = YOLO detections
- Bottom stats show matched/unmatched balls
- Position differences in pixels

### Scenario 2: Performance Testing

You want to measure processing speed:

```bash
python tools/compare_detectors.py \
  --video test_shot.mp4 \
  --detector both \
  --yolo-model models/billiards.pt \
  --max-frames 100 \
  --stats-json performance.json
```

**Output:**
- Console shows avg processing time and FPS
- JSON file has detailed statistics
- Compare ms/frame between detectors

### Scenario 3: Debugging Specific Frames

Something goes wrong at frame 250:

```bash
python tools/compare_detectors.py \
  --video test_shot.mp4 \
  --detector both \
  --start-frame 240 \
  --max-frames 20 \
  --display \
  --verbose
```

**Controls:**
- Space bar = pause/resume
- Q = quit

### Scenario 4: Creating Demo Video

You want to show detector comparison:

```bash
python tools/compare_detectors.py \
  --video demo_shot.mp4 \
  --detector both \
  --yolo-model models/billiards.pt \
  --background empty_table.jpg \
  --output demo_comparison.mp4
```

**Result:**
- Side-by-side video
- Annotated with ball positions
- Performance metrics displayed

## Configuration Tuning

### If OpenCV Detects Too Many False Positives

Edit `tools/example_opencv_config.json`:

```json
{
  "min_confidence": 0.6,          // Increase from 0.4
  "hough_param2": 35,             // Increase from 30
  "max_overlap_ratio": 0.2,       // Decrease from 0.3
  "use_background_subtraction": true
}
```

Then use:
```bash
python tools/compare_detectors.py \
  --video video.mp4 \
  --detector opencv \
  --config tools/example_opencv_config.json \
  --display
```

### If YOLO is Too Sensitive

Edit `tools/example_yolo_config.json`:

```json
{
  "confidence": 0.6,              // Increase from 0.4
  "nms_threshold": 0.5,           // Increase from 0.45
  "device": "cpu"
}
```

Then use:
```bash
python tools/compare_detectors.py \
  --video video.mp4 \
  --detector yolo \
  --yolo-model model.pt \
  --yolo-config tools/example_yolo_config.json \
  --display
```

## Understanding the Output

### Visual Display

```
┌───────────────────────┬───────────────────────┐
│ OpenCV (Green)        │ YOLO (Blue)           │
│ ●  ●  ●  ●  ●         │ ●  ●  ●  ●            │
│ Balls: 5              │ Balls: 4              │
│ Time: 45ms            │ Time: 12ms            │
└───────────────────────┴───────────────────────┘
Frame 42 | Matched: 4 | OpenCV only: 1 | YOLO only: 0
```

**Interpretation:**
- **Matched: 4** = Both detectors found 4 balls in same positions
- **OpenCV only: 1** = OpenCV found 1 extra ball (false positive or YOLO miss)
- **YOLO only: 0** = YOLO didn't find any unique balls
- **Time difference** = YOLO is ~3.75x faster here

### Statistics Output

```
OpenCV Detector:
  Average processing time: 45.23ms
  Theoretical max FPS: 22.1
  Average balls detected: 8.3
```

**What this means:**
- Each frame takes 45ms to process
- Could theoretically handle 22 FPS if video was only detection
- Consistently finds ~8 balls per frame

### Position Differences

```
Average position difference: 2.43 pixels
```

**What this means:**
- When both detectors find the same ball, they disagree by ~2.4 pixels on average
- < 3 pixels = Excellent agreement
- 3-5 pixels = Good agreement
- 5-10 pixels = Moderate agreement
- > 10 pixels = Poor agreement (check calibration)

## Troubleshooting One-Liners

### Video won't play
```bash
# Check video info
ffprobe your_video.mp4

# Convert to compatible format if needed
ffmpeg -i your_video.mp4 -c:v libx264 -c:a aac converted.mp4
```

### YOLO model won't load
```bash
# Check if ultralytics is installed
python -c "import ultralytics; print('OK')"

# Install if missing
pip install ultralytics
```

### Display window doesn't show
```bash
# Use output file instead
python tools/compare_detectors.py \
  --video video.mp4 \
  --detector opencv \
  --output result.mp4

# Then play with default player
open result.mp4  # macOS
xdg-open result.mp4  # Linux
```

### Too slow/fast
```bash
# Process fewer frames
python tools/compare_detectors.py \
  --video video.mp4 \
  --detector both \
  --max-frames 50 \
  --display

# Or skip frames
python tools/compare_detectors.py \
  --video video.mp4 \
  --detector both \
  --start-frame 100 \
  --max-frames 100 \
  --display
```

## Next Steps

1. **Read full documentation**: See `README_compare_detectors.md`
2. **Tune configurations**: Edit JSON config files
3. **Test on your videos**: Use your own test footage
4. **Compare different models**: Try multiple YOLO models
5. **Export statistics**: Use `--stats-json` for analysis

## Getting Help

If something doesn't work:

1. Try with `--verbose` flag for detailed logs
2. Check the full README: `tools/README_compare_detectors.md`
3. Verify video file with `ffprobe`
4. Test with a shorter video first
5. Check OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

## Example Workflow

Complete workflow for testing a new YOLO model:

```bash
# 1. Test OpenCV baseline
python tools/compare_detectors.py \
  --video test.mp4 \
  --detector opencv \
  --background empty_table.jpg \
  --max-frames 50 \
  --stats-json opencv_baseline.json

# 2. Test new YOLO model
python tools/compare_detectors.py \
  --video test.mp4 \
  --detector yolo \
  --yolo-model models/new_model.pt \
  --max-frames 50 \
  --stats-json yolo_new.json

# 3. Compare both
python tools/compare_detectors.py \
  --video test.mp4 \
  --detector both \
  --yolo-model models/new_model.pt \
  --background empty_table.jpg \
  --output comparison.mp4 \
  --stats-json comparison.json \
  --display

# 4. Review statistics
cat comparison.json
```

This gives you baseline, new model performance, and direct comparison!
