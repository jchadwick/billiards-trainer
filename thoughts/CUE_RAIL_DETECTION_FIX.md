# Cue Detection Rail False Positive Fix

## Problem
Cue detection was sometimes detecting perfectly horizontal or vertical lines that followed the rail instead of the actual cue stick. This resulted in the detected cue line snapping to rail edges rather than tracking the player's cue.

## Root Cause
The `_estimate_cue_angle()` method in `yolo_detector.py` uses Hough Line Transform to detect lines in the YOLO-detected cue region. The algorithm:

1. Extracts the bounding box region where YOLO detected the cue
2. Applies Canny edge detection to find edges
3. Uses Hough Line Transform to find all straight lines
4. Selects the **longest line** as the cue stick

**The Issue:** When the YOLO bounding box includes part of a table rail (the raised edges around the playing surface), the rail edge creates a strong, long, perfectly straight line. Since rails are:
- Perfectly horizontal (long sides) or vertical (short sides)
- Very straight and well-defined edges
- Often longer than the visible portion of the cue stick

The Hough transform would detect the rail as the "longest line" and select it instead of the cue stick.

## Solution
Added filtering logic to distinguish between rail lines and cue lines based on angle:

1. **Calculate angle for each detected line**
2. **Classify lines as "rail" or "valid cue" based on angle:**
   - Rail lines: Within 15° of horizontal (0°/180°) or vertical (90°/270°)
   - Valid cue lines: All other angles
3. **Prefer valid cue lines** when selecting the longest line
4. **Fallback to rail lines** only if no valid cue lines are found (edge case)

### Key Parameters
- `RAIL_ANGLE_THRESHOLD = 15.0°` - Lines within this tolerance of 0/90/180/270 degrees are considered potential rails

### Rationale
Pool/billiards cue sticks are **rarely perfectly horizontal or vertical** when:
- Aiming at balls (cue is at an angle to the table)
- Positioned for a shot (angled toward the cue ball)
- In any normal playing position

Rails, on the other hand, are **always** perfectly aligned with the table edges (0°, 90°, 180°, 270°).

By filtering out near-horizontal/vertical lines, we eliminate rail false positives while preserving actual cue detections.

## Implementation Details

### Changes Made
- `backend/vision/detection/yolo_detector.py:1154-1207` - Added rail filtering logic to `_estimate_cue_angle()`

### Algorithm Flow
```
1. Detect all lines with Hough Transform
2. For each line:
   - Calculate angle
   - Check if angle is within 15° of 0/90/180/270
   - Classify as "rail" or "valid cue"
3. Select longest line from "valid cue" lines
4. If no valid cue lines, fall back to longest rail line
5. Return angle of selected line
```

## Testing Recommendations
1. **Test with cue near rails** - Verify cue is tracked correctly even when close to table edge
2. **Test various cue angles** - Ensure detection works across full range of playing angles
3. **Test edge cases:**
   - Cue perfectly horizontal/vertical (rare but possible) - should fall back to rail lines
   - Multiple long lines in frame - should select the actual cue
   - Rail clearly visible in YOLO bbox - should filter out rail line

## Potential Adjustments
If issues persist, consider:
- **Adjusting RAIL_ANGLE_THRESHOLD** (currently 15°)
  - Increase if legitimate angled cues are being filtered
  - Decrease if rails at slight angles are still getting through
- **Adding position-based filtering** - Rails are typically at frame edges
- **Using YOLO confidence** - Weight line selection by proximity to YOLO detection center
- **Color/intensity filtering** - Rails and cues often have different colors/intensities

## Related Files
- `backend/vision/detection/yolo_detector.py` - Core detection logic
- `backend/vision/models.py` - Data models (CueStick, Table with rails)
- `thoughts/CUE_DETECTION_FIX.md` - Previous fix for frame caching issue
