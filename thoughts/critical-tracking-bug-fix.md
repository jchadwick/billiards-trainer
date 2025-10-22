# Critical Tracking Bug Fix - Index Mapping Issue

## Problem

Tracks were not updating when balls moved. Ghost balls would remain at old positions even after the real ball moved to a new location.

**Symptom**: After hitting a ball, the track would stay at the old position instead of following the ball to its new location.

## Root Cause

**Critical index mapping bug** in `_associate_detections_to_tracks()` method (lines 396-442).

The bug occurred because:
1. `_build_cost_matrix()` creates a cost matrix with only **valid tracks** (line 453)
2. The cost matrix rows are indexed 0 to `len(valid_tracks)-1`
3. The Hungarian algorithm returns row indices into this **cost matrix** (line 416)
4. These row indices were being used directly as indices into `self.tracks` (old lines 425, 427, 430, 432)
5. **But `self.tracks` might have invalid/deleted tracks!** The indices don't match!

### Example of the Bug

```
self.tracks = [
    Track(id=1, DELETED),    # index 0 - invalid
    Track(id=2, CONFIRMED),  # index 1 - valid
    Track(id=3, CONFIRMED),  # index 2 - valid
    Track(id=4, LOST),       # index 3 - invalid
]

valid_tracks = [1, 2]  # Indices of valid tracks in self.tracks

cost_matrix shape: (2, num_detections)  # Only 2 rows for 2 valid tracks
  Row 0 -> Track at self.tracks[1] (id=2)
  Row 1 -> Track at self.tracks[2] (id=3)

Hungarian algorithm returns: track_idx=0, detection_idx=5
  This means: cost_matrix[0, 5] is the best match

OLD CODE BUG:
  Uses track_idx=0 directly -> self.tracks[0] = DELETED track!  ❌

CORRECT:
  Should map: cost_matrix row 0 -> valid_tracks[0] -> self.tracks[1]  ✅
```

### Impact

This bug caused:
- Tracks not updating when balls moved
- Ghost balls at old positions
- New detections creating duplicate tracks instead of associating with existing ones
- Complete failure of motion tracking

## Fix

### Changes to `_build_cost_matrix()` (lines 444-511)

**Return valid_track_indices mapping**:

```python
# BEFORE
def _build_cost_matrix(self, detections: list[Ball]) -> tuple[NDArray[np.float64], list[float]]:
    valid_tracks = [i for i, track in enumerate(self.tracks) if track.is_valid()]
    ...
    return cost_matrix, distance_thresholds

# AFTER
def _build_cost_matrix(self, detections: list[Ball]) -> tuple[NDArray[np.float64], list[float], list[int]]:
    """Build cost matrix for Hungarian algorithm.

    Returns:
        Tuple of (cost_matrix, distance_thresholds, valid_track_indices) where:
        - cost_matrix: NxM matrix of association costs
        - distance_thresholds: per-track maximum association distances
        - valid_track_indices: mapping from cost_matrix row index to actual track index
    """
    valid_tracks = [i for i, track in enumerate(self.tracks) if track.is_valid()]
    ...
    return cost_matrix, distance_thresholds, valid_tracks
```

### Changes to `_associate_detections_to_tracks()` (lines 396-442)

**Use the index mapping**:

```python
# BEFORE (BUGGY)
cost_matrix, distance_thresholds = self._build_cost_matrix(detections)
track_indices, detection_indices = linear_sum_assignment(cost_matrix)

for track_idx, detection_idx in zip(track_indices, detection_indices):
    cost = cost_matrix[track_idx, detection_idx]
    threshold = distance_thresholds[track_idx]

    if cost < threshold and self.tracks[track_idx].is_valid():  # ❌ Wrong index!
        matched_tracks.append((track_idx, detection_idx))       # ❌ Wrong index!

# AFTER (FIXED)
cost_matrix, distance_thresholds, valid_track_indices = self._build_cost_matrix(detections)

if len(valid_track_indices) == 0:
    return [], list(range(len(detections))), list(range(len(self.tracks)))

cost_matrix_row_indices, detection_indices = linear_sum_assignment(cost_matrix)

for cost_matrix_row_idx, detection_idx in zip(cost_matrix_row_indices, detection_indices):
    # Map from cost matrix row index to actual track index
    actual_track_idx = valid_track_indices[cost_matrix_row_idx]  # ✅ Correct mapping!

    cost = cost_matrix[cost_matrix_row_idx, detection_idx]
    threshold = distance_thresholds[cost_matrix_row_idx]

    if cost < threshold and self.tracks[actual_track_idx].is_valid():  # ✅ Correct index!
        matched_tracks.append((actual_track_idx, detection_idx))       # ✅ Correct index!
```

## Testing

### Quality Checks
- ✅ Syntax validation passed (py_compile)
- ✅ Linter passed (ruff check)

### Expected Behavior After Fix

1. **Ball Movement**: When a ball moves, the track should follow it to the new position
2. **No Ghost Balls**: Old positions should not persist as ghost balls
3. **Correct Association**: Moved balls should be associated with their existing tracks, not create new ones
4. **Track Continuity**: Track IDs should remain stable across ball movement

## Files Modified

- `/Users/jchadwick/code/billiards-trainer/backend/vision/tracking/tracker.py`
  - Lines 407-442: Fixed association logic with proper index mapping
  - Lines 444-511: Updated cost matrix to return valid_track_indices

## Severity

**CRITICAL** - This bug completely broke motion tracking. Without this fix, the system could not track moving balls at all.

## Additional Fix: UNKNOWN Ball Type Compatibility

### Second Bug Found

After fixing the index mapping bug, tracking still didn't work because of **overly strict type matching**.

**Problem**:
- YOLO often detects balls as generic "ball" class → `BallType.UNKNOWN`
- Tracks have specific types like `BallType.SOLID`, `BallType.STRIPE` from initial classification
- Compatibility check rejected UNKNOWN detections from matching typed tracks
- Cost matrix applied 2x penalty for "type mismatch" even when detection was UNKNOWN

**Impact**: Balls that moved would get new UNKNOWN detections, but couldn't associate with their existing track because the track had a specific type.

### Fix Applied

**Updated compatibility check** (lines 513-524):

```python
# BEFORE
if (
    track.ball_type != detection.ball_type
    and detection.ball_type != BallType.CUE
    and track.ball_type != BallType.CUE
):
    return False

# AFTER
if (
    track.ball_type != detection.ball_type
    and detection.ball_type not in [BallType.CUE, BallType.UNKNOWN]  # ✅ Allow UNKNOWN
    and track.ball_type not in [BallType.CUE, BallType.UNKNOWN]
):
    return False
```

**Updated type mismatch penalty** (lines 490-497):

```python
# BEFORE
if (
    track.ball_type != detection.ball_type
    and detection.ball_type != BallType.CUE
):
    distance *= type_mismatch_penalty

# AFTER
# Skip penalty if either is UNKNOWN (generic ball detection)
if (
    track.ball_type != detection.ball_type
    and detection.ball_type not in [BallType.CUE, BallType.UNKNOWN]  # ✅ No penalty for UNKNOWN
    and track.ball_type not in [BallType.CUE, BallType.UNKNOWN]
):
    distance *= type_mismatch_penalty
```

**Result**: UNKNOWN detections can now freely associate with any track, maintaining track continuity even when classification isn't available.

## Related Issues

This bug was independent of the other improvements made for:
- Ball detection speed
- Edge ball detection
- Velocity-based search radius

However, it was **masked** by those issues - the system couldn't track moving balls for THREE critical reasons:
1. Index mapping bug (tracks accessing wrong indices)
2. Type compatibility bug (UNKNOWN detections rejected)
3. Small search radius (fixed earlier with velocity-adaptive search)

All three needed to be fixed for moving ball tracking to work.
