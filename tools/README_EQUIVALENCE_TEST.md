# Backend Equivalence Test

This test script verifies that the integrated backend system (`IntegrationService` + `VisionModule` + `CoreModule`) produces the same results as the standalone `video_debugger.py` tool.

## Purpose

The equivalence test ensures that:
1. Ball detections are identical between both systems
2. Cue stick detections match (position, angle)
3. Trajectory calculations produce the same results (lines, collisions)
4. Both systems process frames consistently

## Usage

### Basic Usage

```bash
# Test a video file (all frames)
python tools/test_backend_equivalence.py demo.mkv

# Test first 100 frames
python tools/test_backend_equivalence.py demo.mkv --frames 100

# Test with custom position tolerance (in pixels)
python tools/test_backend_equivalence.py demo.mkv --tolerance 5.0

# Save detailed reports to custom directory
python tools/test_backend_equivalence.py demo.mkv --output-dir ./my-test-results
```

### Advanced Options

```bash
python tools/test_backend_equivalence.py <video_file> \
  --frames N \              # Max frames to test (default: all)
  --tolerance T \           # Position tolerance in pixels (default: 2.0)
  --angle-tolerance A \     # Angle tolerance in degrees (default: 1.0)
  --output-dir DIR \        # Output directory (default: ./equivalence-test-results)
  --log-level LEVEL         # Logging level: DEBUG, INFO, WARNING, ERROR
```

## What Gets Tested

### 1. Ball Detections
- Number of balls detected
- Ball positions (within tolerance)
- Ball types/IDs
- Ball tracking consistency

### 2. Cue Stick Detection
- Cue tip position
- Cue angle
- Cue stick presence/absence
- Detection confidence

### 3. Trajectory Calculations
- Number of trajectory line segments
- Line segment endpoints (within tolerance)
- Number of collisions
- Collision positions and types
- Multi-ball trajectory consistency

## Output

The test generates two reports:

### 1. JSON Report (`equivalence_report_TIMESTAMP.json`)
Complete test results in JSON format with:
- Per-frame comparison results
- All ball positions and errors
- Complete trajectory data
- Detailed difference analysis

### 2. Summary Report (`equivalence_summary_TIMESTAMP.txt`)
Human-readable summary with:
- Overall equivalence percentage
- Statistics (avg/max errors)
- Count of mismatches by type
- First 50 differences listed
- Pass/fail verdict

## Exit Codes

- `0`: Systems are equivalent (≥99% match) or mostly equivalent (≥95% match)
- `1`: Systems have significant differences (<95% match) or test failed

## Example Output

```
================================================================================
EQUIVALENCE TEST RESULTS
================================================================================

Video: demo.mkv
Total Frames: 300
Frames with Differences: 3
Equivalence: 99.00%

Ball Position Error (avg): 0.87px
Ball Position Error (max): 2.14px
Ball Count Mismatches: 0
Cue Detection Mismatches: 1
Trajectory Mismatches: 2

================================================================================
✓ SYSTEMS ARE EQUIVALENT
```

## Interpreting Results

### Equivalence Levels

- **99-100%**: Systems are equivalent (expected)
- **95-99%**: Mostly equivalent with minor differences (acceptable)
- **<95%**: Significant differences (investigation needed)

### Common Differences

1. **Position Errors < 2px**: Normal due to floating-point precision
2. **Occasional Cue Mismatches**: Different frame timing can cause slight differences
3. **Trajectory Line Count**: Small differences expected due to collision depth limits

### When to Investigate

- Ball count mismatches > 5% of frames
- Average position error > 5px
- Trajectory mismatches > 10% of frames
- Consistent pattern of differences

## Requirements

- Python 3.10+
- Backend modules (VisionModule, CoreModule, IntegrationService)
- OpenCV, NumPy
- YOLO model (for detection)
- Video file in supported format (MKV, MP4)

## Troubleshooting

### "Failed to start video_debugger"
- Check video file path is correct
- Ensure video file is readable
- Verify YOLO model exists

### "No frames available"
- Video file may be corrupted
- Check video codec is supported
- Try different video file

### High error rates
- Check position tolerance is reasonable (2-5px)
- Verify both systems use same configuration
- Ensure table calibration is consistent

## Implementation Details

The test works by:

1. **Video Debugger Runner**: Processes frames using standalone `video_debugger.py` logic
2. **Backend Runner**: Processes same frames through integrated backend
3. **Frame Comparator**: Compares results frame-by-frame
4. **Report Generator**: Analyzes differences and generates reports

Both systems process the same video frames sequentially, allowing direct comparison of outputs.

## Configuration

The test uses configuration from `config/current.json` for:
- Table playing area corners
- Calibration resolution
- Detection parameters

Ensure configuration is consistent between test runs for reproducible results.
