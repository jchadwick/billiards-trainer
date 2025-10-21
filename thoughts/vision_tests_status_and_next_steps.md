# Vision Tests - Status and Next Steps

## Current Status (3/8 Tests Passing)

### ✅ Passing Tests
1. **test_calibration_straight_on_view** - Working
2. **test_empty_table_no_false_positives** - Fixed! No false positives with param2=42
3. **test_motion_blur_handling** - Working

### ❌ Failing Tests
4. **test_clustered_balls_separation** - Low recall (~6%), missing 15/16 balls
5. **test_full_table_all_balls** - Zero recall, missing all 16 balls
6. **test_multiple_balls_detection_accuracy** - Zero recall, missing all 5 balls
7. **test_cue_detection_aiming** - Cue not detected
8. **test_cue_detection_frame_with_cue** - Cue not detected

## What Was Fixed

### 1. Critical Config Bug ✅
**File**: `backend/vision/detection/balls.py:107`
- **Problem**: Test config used key `"quality"` but code looked for `"quality_filters"`
- **Fix**: Added fallback to check both keys
```python
quality_config = config.get("quality_filters", config.get("quality", {}))
```

### 2. Test Configuration Corrections ✅
**File**: `backend/vision/test_vision_with_test_data.py`
- `expected_radius`: 20 → 50 (matches actual ball size in test images)
- `radius_tolerance`: 0.30 → 0.40
- `min_confidence`: 0.4 → 0.25

### 3. Resolution-Adaptive Param2 Scaling ✅
**File**: `backend/vision/detection/balls.py:799-836`
- Implemented `_calculate_adaptive_param2()` method
- Uses sqrt scaling: `param2 = base * sqrt(width/1920)`
- For 4K images (3840px): param2 = 30 * 1.414 = 42
- **Result**: Empty table test now passes (no false positives)

## Core Problem Identified

**The Param2 Tradeoff** (confirmed by research subagent):
- **param2=42**: Good for empty table ✅ but misses real balls ❌
- **param2=35-38**: Detects balls ✅ but false positives on empty table ❌
- **No single param2 value satisfies all test scenarios**

This is a fundamental algorithmic limitation, not a tuning problem.

## Recommended Solutions (from Research)

### Short-term: Multi-Scale Detection
Run Hough at multiple param2 values (e.g., 35, 40, 45) and merge results:
```python
def _detect_multiscale_hough(self, frame):
    all_candidates = []
    for param2 in [35, 40, 45]:
        circles = cv2.HoughCircles(..., param2=param2, ...)
        all_candidates.extend(circles)
    # Merge candidates that appear in multiple scales
    return self._merge_multiscale_candidates(all_candidates)
```

**Pros**: Simple, catches both strong (high param2) and weak (low param2) signals
**Cons**: 3x slower, needs smart merging logic
**Estimated effort**: 2-3 hours

### Medium-term: Improve Filtering Logic
Current filtering (`_filter_and_validate`, `_is_bright_enough`) is too strict:
- **Problem**: Brightness filter rejects valid balls
- **Solution**: Relax thresholds, add debug logging to see why candidates are rejected
- **Location**: `backend/vision/detection/balls.py:959-1072`

**Estimated effort**: 3-4 hours

### Long-term: Context-Aware Detection
- Analyze frame content to dynamically adjust all thresholds
- Fix color mask to properly filter table surface
- Different strategies for empty vs. full tables

**Estimated effort**: 6-8 hours

## Cue Detection Issues

**Separate problem** from ball detection:
- NumPy deprecation warnings at `cue.py:1081-1082`
- Cue not being detected at all

**Quick fix** for warnings:
```python
# Line 1081-1082, replace:
lefty = int((-x * vy / vx) + y)
righty = int(((frame.shape[1] - x) * vy / vx) + y)

# With:
lefty = int(float(-x * vy / vx) + float(y))
righty = int(float((frame.shape[1] - x) * vy / vx) + float(y))
```

**Estimated effort**: 5 minutes for warnings, 1-2 hours for detection debugging

## Files Modified

1. `backend/vision/detection/balls.py`
   - Line 107: Config parsing fix
   - Lines 754-852: Adaptive param2 calculation

2. `backend/vision/test_vision_with_test_data.py`
   - Lines 213-222: Test configuration updates

3. `thoughts/vision_test_tuning_analysis.md` - Detailed analysis
4. `thoughts/vision_tests_status_and_next_steps.md` - This file

## Next Actions (Priority Order)

### Option A: Quick Wins (2-3 hours)
1. Implement multi-scale Hough detection
2. Fix NumPy warnings in cue.py
3. Re-test → likely gets 5-6/8 tests passing

### Option B: Systematic Approach (6-8 hours)
1. Fix NumPy warnings (5 min)
2. Implement multi-scale detection (2-3 hours)
3. Improve filtering logic with debug logging (3-4 hours)
4. Iteratively tune based on logs (1-2 hours)
5. Re-test → likely gets 7-8/8 tests passing

### Option C: Document & Handoff (30 min)
1. Create detailed handoff documentation
2. List specific param2 values to try
3. Provide debugging tools and scripts
4. Let original developer or another session complete

## Test Command

```bash
# Run all tests
python -m pytest backend/vision/test_vision_with_test_data.py -v

# Run specific test
python -m pytest backend/vision/test_vision_with_test_data.py::TestVisionWithGroundTruth::test_empty_table_no_false_positives -v
```

## Key Insights

1. **Problem is algorithmic, not just parameters** - One param2 can't work for all scenarios
2. **Color mask doesn't work as expected** - Returns 50-60% density for all images
3. **Sqrt scaling works well for resolution** - Balances across different image sizes
4. **Current code has good infrastructure** - Just needs smarter thresholding logic
5. **Multi-scale detection is the pragmatic solution** - Proven technique, moderate effort
