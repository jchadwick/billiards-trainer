# Dataset Creator - Frame Extraction Tool for YOLO Training

A sophisticated frame extraction tool designed to create diverse, high-quality training datasets for YOLO object detection models from billiards video footage.

## Features

### Smart Sampling Strategies

1. **Uniform Sampling** - Extract frames at regular intervals
2. **Scene Change Detection** - Extract frames when significant changes occur
3. **Random Sampling** - Extract frames at random positions
4. **Hybrid** - Combine uniform + scene change for maximum diversity

### Quality Analysis

- **Sharpness Detection** - Using Laplacian variance
- **Brightness Analysis** - HSV-based brightness scoring
- **Motion Blur Detection** - Reject blurry frames
- **Scene Complexity** - Edge density analysis
- **Color Variety** - Histogram entropy calculation

### Edge Case Detection

- **Shadow Detection** - Identify frames with ball shadows
- **Cluster Detection** - Find balls close together
- **Occlusion Detection** - Detect hands/cue overlapping balls
- **Motion Blur** - Flag blurry frames from ball movement

### Output

- High-quality JPEG frames (configurable quality 0-100)
- Comprehensive metadata.json with:
  - Frame timestamps and positions
  - Quality metrics
  - Ball count estimates
  - Edge case flags
  - Scene characteristics
  - Extraction statistics

## Installation

The tool uses standard dependencies already in the project:

```bash
# Already installed with the project
pip install opencv-python numpy
```

## Usage

### Basic Usage

Extract 1000 frames with default settings:

```bash
python tools/dataset_creator.py video.mp4
```

### Specify Output Directory

```bash
python tools/dataset_creator.py video.mp4 --output dataset/raw
```

### Control Frame Count

```bash
python tools/dataset_creator.py video.mp4 --count 1500
```

### Choose Sampling Strategy

```bash
# Uniform intervals (fast, predictable)
python tools/dataset_creator.py video.mp4 --strategy uniform

# Scene changes (captures variety)
python tools/dataset_creator.py video.mp4 --strategy scene_change

# Random sampling (good distribution)
python tools/dataset_creator.py video.mp4 --strategy random

# Hybrid (best of both worlds) - DEFAULT
python tools/dataset_creator.py video.mp4 --strategy hybrid
```

### Quality Control

```bash
# Only extract high-quality frames
python tools/dataset_creator.py video.mp4 --min-quality 0.7

# Maximum JPEG quality
python tools/dataset_creator.py video.mp4 --jpg-quality 98

# Adjust scene change sensitivity
python tools/dataset_creator.py video.mp4 --scene-threshold 25.0
```

### Edge Case Analysis

```bash
# Enable comprehensive edge case detection
python tools/dataset_creator.py video.mp4 --analyze-edge-cases
```

### Advanced Examples

```bash
# High-quality dataset with edge case analysis
python tools/dataset_creator.py video.mp4 \
  --output dataset/raw \
  --count 1500 \
  --strategy hybrid \
  --min-quality 0.7 \
  --jpg-quality 98 \
  --analyze-edge-cases

# Scene-based extraction for diverse configurations
python tools/dataset_creator.py video.mp4 \
  --strategy scene_change \
  --scene-threshold 25.0 \
  --min-quality 0.6

# Quick extraction without metadata
python tools/dataset_creator.py video.mp4 \
  --count 500 \
  --strategy uniform \
  --no-metadata

# Verbose output for debugging
python tools/dataset_creator.py video.mp4 --verbose
```

## Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `video` | - | Input video file path | Required |
| `--output` | `-o` | Output directory | `dataset/raw` |
| `--count` | `-c` | Target frame count | `1000` |
| `--strategy` | `-s` | Sampling strategy | `hybrid` |
| `--min-quality` | `-q` | Min quality threshold (0.0-1.0) | `0.5` |
| `--jpg-quality` | - | JPEG quality (0-100) | `95` |
| `--scene-threshold` | - | Scene change threshold | `30.0` |
| `--analyze-edge-cases` | - | Enable edge case detection | `False` |
| `--no-metadata` | - | Skip metadata.json | `False` |
| `--verbose` | `-v` | Verbose logging | `False` |

## Output Structure

```
dataset/raw/
├── frame_000000_0.000.jpg
├── frame_000039_1.300.jpg
├── frame_000078_2.600.jpg
├── ...
└── metadata.json
```

### Frame Naming Convention

Format: `frame_{frame_number:06d}_{timestamp:.3f}.jpg`

- `frame_number`: Original frame number in video (6 digits, zero-padded)
- `timestamp`: Time in seconds from video start (3 decimal places)

Example: `frame_001234_41.133.jpg` = Frame #1234 at 41.133 seconds

### Metadata JSON Structure

```json
{
  "extraction_date": "2025-10-06T16:18:22.131912",
  "video_source": "demo.mkv",
  "configuration": {
    "target_count": 1000,
    "strategy": "hybrid",
    "min_quality": 0.5,
    "jpg_quality": 95,
    "scene_change_threshold": 30.0
  },
  "statistics": {
    "total_video_frames": 1951,
    "frames_extracted": 1000,
    "frames_rejected": 50,
    "extraction_time": 45.2,
    "video_duration": 65.0,
    "average_quality": 0.657,
    "edge_case_count": 125
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "filename": "frame_000000_0.000.jpg",
      "video_source": "demo.mkv",
      "resolution": [1280, 720],
      "quality_score": 0.715,
      "sharpness_score": 0.770,
      "brightness": 0.660,
      "estimated_ball_count": 12,
      "has_shadows": false,
      "has_clusters": true,
      "has_occlusions": false,
      "has_motion_blur": false,
      "edge_case_flags": ["clusters"],
      "scene_complexity": 0.024,
      "color_variety": 0.462
    }
  ]
}
```

## Sampling Strategies Explained

### Uniform Sampling

- Extracts frames at regular intervals
- Fast and predictable
- Good for consistent coverage
- Best for: Videos with gradual changes

**Example**: For 1000 frames from 2000-frame video, extracts every 2nd frame

### Scene Change Detection

- Detects significant visual changes
- Captures different game states
- May miss subtle variations
- Best for: Videos with distinct shots/setups

**How it works**: Compares histogram correlation between consecutive frames

### Random Sampling

- Purely random frame selection
- Good statistical distribution
- May miss important scenes
- Best for: Large datasets, avoiding bias

### Hybrid (Recommended)

- Combines uniform + scene change
- 50% uniform for coverage
- 50% scene changes for variety
- Best for: Most use cases

## Quality Metrics Explained

### Quality Score (0.0-1.0)

Average of sharpness and brightness scores. Higher is better.

- **< 0.4**: Poor quality (blurry, dark)
- **0.4-0.6**: Acceptable
- **0.6-0.8**: Good quality
- **> 0.8**: Excellent quality

### Sharpness Score (0.0-1.0)

Laplacian variance normalized. Measures focus quality.

- Higher = sharper edges
- Lower = motion blur or out of focus

### Brightness (0.0-1.0)

Average V channel in HSV color space.

- **< 0.3**: Too dark
- **0.3-0.7**: Good lighting
- **> 0.7**: Very bright

## Edge Cases Explained

### Shadows

Circular dark regions near balls, typically 5-20% of frame.

**Use case**: Train model to distinguish shadows from balls

### Clusters

Balls within 2.5 radii of each other.

**Use case**: Train model to separate touching/nearby balls

### Occlusions

Hands, cue stick, or other objects partially covering balls.

**Use case**: Train model to detect partially visible balls

### Motion Blur

Sharpness below threshold due to ball movement.

**Use case**: Decide whether to include or filter these frames

## Best Practices

### For Training Dataset

1. Use **hybrid strategy** for diversity
2. Set **min-quality 0.6** to avoid bad frames
3. Enable **--analyze-edge-cases** to balance dataset
4. Target **1000-1500 frames** per video

```bash
python tools/dataset_creator.py video.mp4 \
  --count 1500 \
  --strategy hybrid \
  --min-quality 0.6 \
  --analyze-edge-cases
```

### For Validation Dataset

1. Use **random strategy** to avoid bias
2. Set **min-quality 0.7** for high quality
3. Target **200-300 frames** per video

```bash
python tools/dataset_creator.py video.mp4 \
  --output dataset/validation \
  --count 250 \
  --strategy random \
  --min-quality 0.7
```

### For Edge Case Dataset

1. Use **scene_change strategy** for variety
2. Set **min-quality 0.5** to allow challenging frames
3. Enable **--analyze-edge-cases**
4. Filter metadata later for specific edge cases

```bash
python tools/dataset_creator.py video.mp4 \
  --output dataset/edge_cases \
  --strategy scene_change \
  --min-quality 0.5 \
  --analyze-edge-cases
```

## Workflow Integration

### Step 1: Extract Frames

```bash
python tools/dataset_creator.py video.mp4 \
  --output dataset/raw \
  --count 1500 \
  --strategy hybrid \
  --min-quality 0.6 \
  --analyze-edge-cases
```

### Step 2: Review Metadata

```bash
# Count frames by edge case
jq '.frames | group_by(.edge_case_flags[]) | map({case: .[0].edge_case_flags[0], count: length})' dataset/raw/metadata.json

# Find high-quality frames
jq '.frames[] | select(.quality_score > 0.7) | .filename' dataset/raw/metadata.json

# Find frames with many balls
jq '.frames[] | select(.estimated_ball_count > 10) | .filename' dataset/raw/metadata.json
```

### Step 3: Annotate with YOLO

Use a YOLO annotation tool (e.g., LabelImg, Roboflow) on extracted frames.

### Step 4: Train YOLO Model

```bash
python tools/train_yolo.py \
  --data dataset/raw \
  --labels dataset/labels \
  --epochs 100
```

## Troubleshooting

### No frames extracted

- **Cause**: Quality threshold too high
- **Solution**: Lower `--min-quality` (try 0.3-0.4)

### Too few frames

- **Cause**: Video too short or quality threshold too high
- **Solution**: Increase `--count` or lower `--min-quality`

### All frames from same scene

- **Cause**: Using uniform with slow-changing video
- **Solution**: Switch to `scene_change` or `hybrid` strategy

### Metadata JSON error

- **Cause**: Special characters in video path
- **Solution**: Use relative paths or rename video

### Out of memory

- **Cause**: Very high resolution video
- **Solution**: Process in batches or reduce frame count

## Performance Notes

- **Speed**: ~100-300 frames/second (depends on resolution)
- **Storage**: ~100-200KB per frame at quality 95
- **Memory**: ~50MB peak usage for HD video

### Optimization Tips

1. Use `--no-metadata` for faster processing
2. Disable `--analyze-edge-cases` if not needed
3. Use `uniform` strategy for maximum speed
4. Lower `--jpg-quality` for smaller files

## Examples with Real Data

### Extract from multiple videos

```bash
for video in videos/*.mp4; do
  python tools/dataset_creator.py "$video" \
    --output "dataset/raw/$(basename "$video" .mp4)" \
    --count 1000 \
    --analyze-edge-cases
done
```

### Create balanced dataset

```bash
# Extract diverse frames
python tools/dataset_creator.py video.mp4 \
  --output dataset/diverse \
  --strategy hybrid \
  --count 1000

# Extract edge cases
python tools/dataset_creator.py video.mp4 \
  --output dataset/edge_cases \
  --strategy scene_change \
  --count 500 \
  --min-quality 0.4 \
  --analyze-edge-cases

# Combine later after annotation
```

## Next Steps

After extracting frames:

1. **Annotate** frames with YOLO annotation tool
2. **Split** into train/validation/test sets
3. **Augment** data if needed (rotations, brightness, etc.)
4. **Train** YOLO model with `tools/train_yolo.py`
5. **Evaluate** model performance
6. **Iterate** - extract more frames from underrepresented cases

## See Also

- `tools/train_yolo.py` - YOLO training script
- `tools/video_debugger.py` - Video analysis tool
- `backend/vision/detection/balls.py` - Ball detection module

## License

Part of the Billiards Trainer System. See main project LICENSE file.
