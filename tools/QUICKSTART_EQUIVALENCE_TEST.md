# Quick Start - Backend Equivalence Test

## 1. Run Your First Test

```bash
# Using the convenience script (recommended)
./tools/run_equivalence_test.sh --video demo.mkv --frames 100

# Or directly with Python
python tools/test_backend_equivalence.py demo.mkv --frames 100
```

## 2. Check the Results

The test will display a summary like this:

```
================================================================================
EQUIVALENCE TEST RESULTS
================================================================================

Video: demo.mkv
Total Frames: 100
Frames with Differences: 2
Equivalence: 98.00%

Ball Position Error (avg): 0.87px
Ball Position Error (max): 2.14px
Ball Count Mismatches: 0
Cue Detection Mismatches: 1
Trajectory Mismatches: 1

================================================================================
✓ SYSTEMS ARE EQUIVALENT
```

## 3. Review Detailed Reports

Reports are saved in `./equivalence-test-results/`:

```bash
# View the summary
cat equivalence-test-results/equivalence_summary_*.txt

# View detailed JSON (use jq for pretty formatting)
jq . equivalence-test-results/equivalence_report_*.json | less
```

## 4. Understanding the Results

### Equivalence Levels

- **99-100%**: Perfect equivalence ✓
- **95-99%**: Mostly equivalent (minor differences) ⚠
- **<95%**: Significant differences - investigate ✗

### Key Metrics

1. **Ball Position Error (avg)**: Should be < 2px
2. **Ball Count Mismatches**: Should be 0 or very low
3. **Cue Detection Mismatches**: A few are OK (timing differences)
4. **Trajectory Mismatches**: Should be < 10%

## 5. Common Use Cases

### Test Specific Number of Frames
```bash
./tools/run_equivalence_test.sh --video demo.mkv --frames 200
```

### Increase Position Tolerance
```bash
./tools/run_equivalence_test.sh --video demo.mkv --tolerance 5.0
```

### Debug Mode (verbose logging)
```bash
python tools/test_backend_equivalence.py demo.mkv --log-level DEBUG
```

### Test Entire Video
```bash
python tools/test_backend_equivalence.py demo.mkv
```

## 6. Troubleshooting

### "Video file not found"
- Check the file path is correct
- Use absolute or relative path from project root

### "Failed to start video_debugger"
- Ensure YOLO model exists: `models/yolov8n-pool-1280.onnx`
- Check video file is valid and readable

### High Error Rates
- Try increasing tolerance: `--tolerance 5.0`
- Check that both systems use same config
- Verify video file quality

## 7. Example Session

```bash
# 1. Run test
./tools/run_equivalence_test.sh --video demo.mkv --frames 100

# 2. Check results
cat equivalence-test-results/equivalence_summary_*.txt

# 3. If all good, test full video
python tools/test_backend_equivalence.py demo.mkv

# 4. Review detailed differences if needed
jq '.differences_summary' equivalence-test-results/equivalence_report_*.json
```

## 8. What to Expect

### First Run (with calibration)
- May take 1-2 minutes for 100 frames
- Systems load models and initialize

### Subsequent Runs
- Faster due to cached models
- Consistent results expected

### Typical Results
- Equivalence: 98-100%
- Ball position error: 0.5-1.5px
- Minor cue/trajectory differences due to timing

## Need More Help?

See the full documentation:
- `tools/README_EQUIVALENCE_TEST.md` - Complete guide
- `tools/test_backend_equivalence.py --help` - Command-line options
- `tools/run_equivalence_test.sh --help` - Wrapper script options
