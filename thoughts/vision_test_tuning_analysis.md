# Vision Test Parameter Tuning Analysis

## Progress Made

### Fixed Critical Bugs
1. **Config parsing bug**: Test used `"quality"` key but code looked for `"quality_filters"` - FIXED in balls.py:107
2. **Resolution scaling**: Added adaptive param2 boost based on image resolution to handle 4K test images
3. **Test config corrections**:
   - expected_radius: 20 → 50 (matches ~50-65px balls in test images)
   - radius_tolerance: 0.30 → 0.40
   - min_confidence: 0.4 → 0.25

## Current Status

**Test Results** (param2_boost=12, expected_radius=50, tolerance=0.40, min_confidence=0.25):
- ✅ test_calibration_straight_on_view: PASSED
- ✅ test_motion_blur_handling: PASSED
- ❌ test_empty_table_no_false_positives: FAILED (has false positives)
- ❌ test_multiple_balls_detection_accuracy: ~17% recall (need 98%)
- ❌ test_clustered_balls_separation: ~6% recall (need 95%)
- ❌ test_full_table_all_balls: 0% recall (need 98%)
- ❌ test_cue_detection_aiming: Cue not detected
- ❌ test_cue_detection_frame_with_cue: Cue not detected

## Core Challenge

Parameter tuning tradeoff:
- **Higher param2** → Fewer false positives BUT miss real balls
- **Lower param2** → Detect more balls BUT get false positives on empty table

Current param2: 30 (config) + 12 (4K boost) = 42

### Debugging Insights (from debug script)
With param2=40:
- Hough found: 9 candidates
- After filtering: 3 candidates
- After classification: 2 balls
- Expected: 5 balls

Issues:
1. Hough not detecting all balls (param2 still too high OR balls don't have strong enough edges)
2. Filtering removing valid candidates (radius/brightness checks too strict)
3. Classification filtering some candidates (confidence too low)

## Recommended Next Steps

### 1. Ball Detection Parameter Sweep
Test these param2_boost values systematically:
- 8, 9, 10, 11, 12, 13, 14, 15
- For each, measure empty_table false positives AND multiple_balls recall
- Find sweet spot

### 2. Alternative: Use Detection Method "hough" Instead of "combined"
The "combined" method runs Hough+Contour+Blob and merges, which may create duplicates.
Try using just "hough" in test config.

### 3. Brightness Filter Review
The `_is_bright_enough` method might be rejecting valid balls. Consider:
- Lowering min_avg_brightness threshold
- Adjusting based on image characteristics

### 4. Cue Detection
Separate issue - cue detector failing entirely. Need to:
- Check cue detection parameters
- Verify line detection is working
- Fix NumPy deprecation warnings in cue.py:1081-1082

## Files Modified
- backend/vision/detection/balls.py (config parsing, param2 scaling)
- backend/vision/test_vision_with_test_data.py (test config)
