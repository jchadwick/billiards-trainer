# Detector Comparison Tool

Visual comparison tool for YOLO and OpenCV ball detectors in the billiards vision system.

## Features

- **Side-by-Side Comparison**: View YOLO and OpenCV detector results simultaneously
- **Performance Metrics**: Real-time processing time and FPS statistics
- **Visual Annotations**: Color-coded bounding boxes and labels (Green for OpenCV, Blue for YOLO)
- **Detection Analysis**: Highlights matched balls, unique detections, and position differences
- **Video Output**: Save comparison videos with annotations
- **Statistics Export**: JSON statistics for further analysis
- **Flexible Configuration**: Support for custom detector configurations
- **Background Subtraction**: Optional background frame for improved OpenCV detection

## Installation

The tool requires the following dependencies (already included in the project):

```bash
# Core dependencies
pip install opencv-python numpy

# For YOLO detector (optional)
pip install ultralytics

# For ONNX models (optional)
pip install onnx onnxruntime
```

## Usage

### Basic Usage

```bash
# Compare both detectors on a video
python tools/compare_detectors.py --video test_video.mp4 --detector both --output comparison.mp4

# Display results in real-time
python tools/compare_detectors.py --video test_video.mp4 --detector both --display

# Test only OpenCV detector
python tools/compare_detectors.py --video test_video.mp4 --detector opencv --display

# Test only YOLO detector (requires model)
python tools/compare_detectors.py --video test_video.mp4 --detector yolo --yolo-model models/billiards.pt --display
```

### Advanced Usage

```bash
# With background subtraction for OpenCV
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector both \
  --background empty_table.jpg \
  --output comparison.mp4

# Process specific frame range
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector both \
  --start-frame 100 \
  --max-frames 300 \
  --display

# With custom configurations
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector both \
  --config opencv_config.json \
  --yolo-model models/billiards.pt \
  --yolo-config yolo_config.json \
  --output comparison.mp4

# Export statistics to JSON
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector both \
  --output comparison.mp4 \
  --stats-json statistics.json
```

## Command-Line Arguments

### Required Arguments

- `--video VIDEO`: Path to input video file

### Optional Arguments

- `--detector {opencv,yolo,both}`: Which detector(s) to use (default: both)
- `--output OUTPUT`: Path to save output video
- `--display`: Display results in real-time window
- `--config CONFIG`: JSON config file for OpenCV detector
- `--yolo-model YOLO_MODEL`: Path to YOLO model file (.pt or .onnx)
- `--yolo-config YOLO_CONFIG`: JSON config file for YOLO detector
- `--background BACKGROUND`: Path to background frame image (empty table)
- `--start-frame START_FRAME`: Frame to start processing from
- `--max-frames MAX_FRAMES`: Maximum number of frames to process
- `--stats-json STATS_JSON`: Path to save statistics JSON file
- `--verbose`: Enable verbose logging

## Configuration Files

### OpenCV Configuration (opencv_config.json)

```json
{
  "detection_method": "combined",
  "hough_dp": 1.0,
  "hough_min_dist_ratio": 0.8,
  "hough_param1": 50,
  "hough_param2": 30,
  "min_radius": 15,
  "max_radius": 26,
  "expected_radius": 20,
  "radius_tolerance": 0.30,
  "min_circularity": 0.75,
  "min_confidence": 0.4,
  "max_overlap_ratio": 0.30,
  "use_background_subtraction": true,
  "background_threshold": 30
}
```

### YOLO Configuration (yolo_config.json)

```json
{
  "device": "cpu",
  "confidence": 0.4,
  "nms_threshold": 0.45,
  "auto_fallback": true
}
```

## Visual Output

### Display Format

When using `--detector both`, the output shows:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│   OpenCV Detector (Green)       │   YOLO Detector (Blue)          │
│   ┌───────────┐                 │   ┌───────────┐                 │
│   │ Ball 1    │                 │   │ Ball 1    │                 │
│   │  ●  0.95  │                 │   │  ●  0.92  │                 │
│   └───────────┘                 │   └───────────┘                 │
│                                 │                                 │
│   Balls: 8                      │   Balls: 7                      │
│   Time: 45.2ms                  │   Time: 12.3ms                  │
└─────────────────────────────────┴─────────────────────────────────┘
Frame 123 | Matched: 7 | OpenCV only: 1 | YOLO only: 0 | Avg diff: 2.3px
```

### Color Coding

- **Green circles/text**: OpenCV detector results
- **Blue circles/text**: YOLO detector results
- **Yellow text**: Comparison metrics at bottom

### Labels

Each detected ball shows:
- Ball type (cue, solid, stripe, eight)
- Ball number (1-15, if detected)
- Confidence score (0.0-1.0)

## Output Statistics

When using `--stats-json`, the tool generates a JSON file with:

```json
{
  "total_frames": 500,
  "opencv": {
    "avg_time_ms": 45.2,
    "max_fps": 22.1,
    "avg_balls_detected": 8.3
  },
  "yolo": {
    "avg_time_ms": 12.8,
    "max_fps": 78.1,
    "avg_balls_detected": 7.9
  },
  "comparison": {
    "avg_matched_balls": 7.5,
    "avg_position_diff_pixels": 2.4
  }
}
```

### Console Statistics

The tool also prints comprehensive statistics to the console:

```
======================================================================
DETECTOR COMPARISON STATISTICS
======================================================================
Total frames processed: 500

OpenCV Detector:
  Average processing time: 45.23ms
  Theoretical max FPS: 22.1
  Average balls detected: 8.3

YOLO Detector:
  Average processing time: 12.78ms
  Theoretical max FPS: 78.2
  Average balls detected: 7.9

Comparison:
  Average matched balls: 7.5
  Average position difference: 2.43 pixels

  YOLO is 3.54x faster
======================================================================
```

## Interactive Controls (Display Mode)

When using `--display`, the following keyboard controls are available:

- **Q**: Quit processing
- **Space**: Pause/resume (press any key to continue)

## Examples

### Example 1: Quick Comparison

```bash
python tools/compare_detectors.py \
  --video recordings/shot_001.mp4 \
  --detector both \
  --display
```

### Example 2: Full Analysis with Output

```bash
python tools/compare_detectors.py \
  --video recordings/full_game.mp4 \
  --detector both \
  --background recordings/empty_table.jpg \
  --yolo-model models/billiards_v1.pt \
  --output analysis/comparison_full_game.mp4 \
  --stats-json analysis/statistics.json
```

### Example 3: Performance Testing

```bash
# Test OpenCV only
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector opencv \
  --max-frames 100 \
  --stats-json opencv_perf.json

# Test YOLO only
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector yolo \
  --yolo-model models/billiards.pt \
  --max-frames 100 \
  --stats-json yolo_perf.json
```

### Example 4: Debugging Specific Frames

```bash
python tools/compare_detectors.py \
  --video test_video.mp4 \
  --detector both \
  --start-frame 250 \
  --max-frames 50 \
  --display \
  --verbose
```

## Interpreting Results

### Matched Balls

Balls are considered "matched" if detected by both methods and their positions are within 30 pixels of each other. This threshold can be adjusted in the code by modifying `match_threshold` in `DetectorComparator.__init__()`.

### Position Differences

The average position difference indicates how closely the two detectors agree on ball positions. Values under 5 pixels indicate excellent agreement.

### Detection Count Differences

- **OpenCV only**: Balls detected by OpenCV but not YOLO (potential false positives or YOLO misses)
- **YOLO only**: Balls detected by YOLO but not OpenCV (potential false positives or OpenCV misses)

### Performance Metrics

- Processing time includes full detection pipeline
- FPS represents theoretical maximum (actual video FPS may be limited by video file)
- Consider GPU acceleration for YOLO to improve performance

## Troubleshooting

### YOLO Model Not Loading

```
Error: Failed to load YOLO model
```

**Solutions:**
- Verify model file exists and path is correct
- Check ultralytics package is installed: `pip install ultralytics`
- For ONNX models, ensure onnxruntime is installed: `pip install onnxruntime`
- Try CPU mode if GPU fails: `--yolo-config '{"device": "cpu"}'`

### Video Won't Open

```
Error: Failed to open video
```

**Solutions:**
- Verify video file exists and path is correct
- Check video codec is supported by OpenCV
- Try converting video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Poor OpenCV Detection

**Solutions:**
- Provide background frame: `--background empty_table.jpg`
- Adjust detection parameters in config file
- Ensure good lighting and contrast in video

### Display Window Issues

**Solutions:**
- On macOS, ensure you're not using SSH without X11 forwarding
- Use `--output` to save video instead of displaying
- Check OpenCV GUI support: `python -c "import cv2; print(cv2.getBuildInformation())"`

## Performance Tips

1. **Use GPU for YOLO**: Set `device: "cuda"` in YOLO config (requires CUDA)
2. **Process subset of frames**: Use `--start-frame` and `--max-frames` for testing
3. **Optimize OpenCV**: Adjust detection parameters to reduce processing time
4. **Video encoding**: Use H.264 codec for faster writing (requires ffmpeg)
5. **Background subtraction**: Improves OpenCV accuracy but adds processing time

## Technical Details

### Architecture

The tool uses the following components:

- **DetectorComparator**: Main comparison engine
  - Manages both detectors
  - Performs ball matching algorithm
  - Calculates statistics

- **DetectionComparison**: Per-frame comparison data
  - Stores detection results
  - Calculates match metrics
  - Tracks position differences

- **ComparisonStatistics**: Aggregate statistics
  - Performance metrics
  - Detection accuracy
  - Cross-detector comparison

### Ball Matching Algorithm

The tool uses a greedy nearest-neighbor matching algorithm:

1. For each OpenCV detection, find nearest YOLO detection
2. If distance < threshold (30px), mark as matched
3. Track unmatched detections from both sides
4. Calculate average position difference for matched pairs

This provides a good balance between accuracy and performance for real-time comparison.

## Contributing

To extend the tool:

1. **Add new detectors**: Implement detector interface in `compare_frame()`
2. **Custom visualizations**: Modify `visualize_comparison()`
3. **Additional metrics**: Extend `DetectionComparison` dataclass
4. **Export formats**: Add new export functions alongside `save_statistics_json()`

## License

This tool is part of the billiards-trainer project and follows the same license.
