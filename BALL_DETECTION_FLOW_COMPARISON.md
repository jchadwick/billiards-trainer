# Ball Detection Flow Comparison: video_debugger.py vs Backend

## Executive Summary

The video_debugger.py has a **working ball detection pipeline** that successfully detects and tracks balls, while the backend has a similar structure but with **critical configuration bugs** that allow ghost balls to appear.

### Root Cause (TL;DR)
**The backend's config file is missing the `vision.tracking` section entirely**, causing the tracker to use default `min_hits=3` instead of the required `min_hits=10`. This low threshold confirms tracks too quickly, allowing transient false detections (ghost balls) to appear as confirmed balls.

### The Fix
Add this to `/config/current.json`:
```json
{
  "vision": {
    "tracking": {
      "min_hits": 10,
      "return_tentative_tracks": false,
      "max_age": 30,
      "max_distance": 100.0
    },
    "detection": {
      "detection_backend": "yolo"
    }
  }
}
```

**Impact:** This single config change will filter out ghost balls by requiring 10 consecutive detections before confirming a track.

---

---

## Side-by-Side Flow Comparison

### video_debugger.py (WORKING) - process_frame() method

```
Frame Input
    ↓
[1] YOLO Detection (hybrid with OpenCV validation)
    - YOLODetector.detect_balls_with_classification()
    - min_confidence=0.25
    - enable_opencv_classification=True
    - min_ball_size=20
    ↓
[2] Tracking (with ghost filtering)
    - ObjectTracker.update_tracking()
    - min_hits=10 (requires 10 consecutive hits to confirm)
    - max_age=30
    - return_tentative_tracks=False (CRITICAL: filters ghosts)
    ↓
[3] Result: Confirmed tracks only
    - Ghost balls filtered out
    - Stable ball positions returned
```

### Backend - VisionModule._process_single_frame() method

```
Frame Input
    ↓
[1] Detection (multiple backends possible)
    Option A: YOLO (if configured)
        - detector.detect_balls(processed_frame)
        - Falls back to OpenCV on failure
    Option B: OpenCV (default)
        - BallDetector.detect_balls(processed_frame)
    ↓
[2] Tracking (IF enabled and tracker exists)
    - tracker.update_tracking(detected_balls, timestamp)
    - Configuration from vision.tracking config
    - May have different min_hits, max_age settings
    ↓
[3] Result: Returns detected_balls
    - May include tentative tracks depending on config
    - May return raw detections if tracking disabled
```

---

## Critical Differences

### 1. **Detector Configuration**

| Aspect | video_debugger.py | Backend |
|--------|-------------------|---------|
| **YOLO Confidence** | 0.15 (very low, catches more) | Not explicitly set in VisionModule |
| **OpenCV Classification** | Enabled (`enable_opencv_classification=True`) | Not mentioned in backend flow |
| **Min Ball Size** | 20 pixels | Configured per BallDetectionConfig (15-26 range) |
| **Fallback Strategy** | Auto-fallback enabled | Configurable via `fallback_to_opencv` |

**Impact:** video_debugger uses hybrid YOLO+OpenCV detection with very permissive settings to catch all balls, then relies on tracking to filter ghosts. Backend may be using different detection settings.

---

### 2. **Tracker Configuration (MOST CRITICAL)**

| Parameter | video_debugger.py | Backend (default) | Impact |
|-----------|-------------------|-------------------|--------|
| **min_hits** | 10 | 3 | Backend confirms tracks 3x faster → more false positives |
| **return_tentative_tracks** | False | Not set (likely True?) | Backend may return unconfirmed detections |
| **max_age** | 30 | 30 | Same |
| **collision filtering** | Yes (min_hits_during_collision=30) | Yes (if configured) | Similar |
| **max_distance** | 100.0 | 50.0 | Backend more strict on association |

**Impact:** The **min_hits=10** and **return_tentative_tracks=False** settings in video_debugger are CRITICAL for ghost ball filtering. Backend's lower min_hits (3) means tracks are confirmed much faster, allowing transient detections (ghosts) to appear as confirmed balls.

---

### 3. **Detection Method Flow**

#### video_debugger.py:
```python
# Line 776-784
if self.yolo_detector is not None:
    # Use hybrid YOLO+OpenCV detection (backend handles classification)
    detected_balls = self.yolo_detector.detect_balls_with_classification(
        frame,
        min_confidence=0.25
    )
else:
    # Use OpenCV detector
    detected_balls = self.detector.detect_balls(frame)
```

**Key:** Uses `detect_balls_with_classification()` which applies OpenCV validation on top of YOLO results.

#### Backend VisionModule:
```python
# Line 1018-1027
if self.config.enable_ball_detection:
    try:
        detected_balls = self.detector.detect_balls(processed_frame)

        # Update tracking if available
        if self.tracker and self.config.enable_tracking:
            detected_balls = self.tracker.update_tracking(
                detected_balls, timestamp
            )
```

**Key:** Uses generic `detect_balls()` method which may not include OpenCV classification step. Tracking is conditional on configuration.

---

### 4. **Frame Processing Order**

| Step | video_debugger.py | Backend |
|------|-------------------|---------|
| **Preprocessing** | Not used (works on raw frame) | Applied if `preprocessing_enabled=True` |
| **Background Subtraction** | Not used in main loop | Can be enabled via config |
| **Detection** | Direct YOLO call | Through detector factory/adapter |
| **Tracking** | Always applied (unconditional) | Only if `enable_tracking=True` |
| **Result Format** | List[Ball] with track_ids | DetectionResult dataclass |

**Impact:** video_debugger has a simpler, more direct pipeline. Backend has more abstraction layers which could introduce bugs or configuration mismatches.

---

## Missing Steps in Backend

### 1. **No Explicit Hybrid Detection**
Backend's detector factory may not be calling the hybrid YOLO+OpenCV classification method that video_debugger uses. The backend calls generic `detect_balls()` which might skip the OpenCV validation step.

**Location:** `/backend/vision/__init__.py` line 1020
**Fix Required:** Ensure detector uses `detect_balls_with_classification()` when using YOLO backend.

### 2. **Tracker May Not Be Initialized**
Backend's tracker initialization depends on config:
```python
# Line 488-496
if self.config.enable_tracking:
    tracking_config = config_mgr.get("vision.tracking", {})
    self.tracker = ObjectTracker(tracking_config)
else:
    self.tracker = None
```

If `vision.tracking` config is missing or `enable_tracking` is False, no tracking happens!

**Impact:** Without tracking, all raw detections (including ghosts) are returned directly.

### 3. **Tracker Config Not Loaded from Constants**
video_debugger hard-codes tracker config with proven values:
```python
self.tracker_config = {
    "max_age": 30,
    "min_hits": 10,  # CRITICAL
    "return_tentative_tracks": False,  # CRITICAL
    # ... other settings
}
```

Backend loads from config file which may have different values or be missing entirely.

**Impact:** Wrong tracker settings = ghost balls appear.

---

## Specific Bugs in Backend

### Bug #1: Tracking May Be Disabled
**File:** `/backend/vision/__init__.py`
**Line:** 488-496
**Issue:** If config `vision.tracking` is not set or `enable_tracking=False`, tracker is None and tracking is skipped entirely.

**Evidence:**
```python
if self.config.enable_tracking:
    tracking_config = config_mgr.get("vision.tracking", {})
    self.tracker = ObjectTracker(tracking_config)
else:
    self.tracker = None
```

Then at line 1023:
```python
if self.tracker and self.config.enable_tracking:
    detected_balls = self.tracker.update_tracking(detected_balls, timestamp)
```

If tracker is None, **raw detections are returned without filtering**.

### Bug #2: Wrong Tracker Parameters
**File:** Config file (likely `/config/current.json` or default config)
**Issue:** The `vision.tracking` section may have:
- `min_hits: 3` (too low, should be 10+)
- `return_tentative_tracks: true` (should be false)
- Missing collision filtering parameters

**Impact:** Even if tracking runs, it confirms tracks too quickly and returns tentative tracks, allowing ghosts to appear.

### Bug #3: Detector May Not Use Hybrid Mode
**File:** `/backend/vision/__init__.py`
**Line:** 1020
**Issue:** The code calls `detector.detect_balls()` instead of `detector.detect_balls_with_classification()`.

**Evidence:**
```python
detected_balls = self.detector.detect_balls(processed_frame)
```

Should be:
```python
if self.yolo_detector:
    detected_balls = self.yolo_detector.detect_balls_with_classification(
        processed_frame, min_confidence=0.25
    )
```

### Bug #4: Detection Backend May Be OpenCV
**File:** Config file
**Issue:** The config may have `vision.detection.detection_backend: "opencv"` instead of `"yolo"`.

**Impact:** OpenCV detector is less accurate and more prone to false positives than YOLO+OpenCV hybrid.

---

## Configuration Comparison

### video_debugger.py (Effective Config)
```python
# Detector
detection_backend = "yolo"
yolo_model_path = "models/yolov8n-pool-1280.onnx"
yolo_device = "cpu"
confidence = 0.15  # Very permissive
enable_opencv_classification = True
min_ball_size = 20

# Tracker
max_age = 30
min_hits = 10  # CRITICAL: High threshold
return_tentative_tracks = False  # CRITICAL: No tentative
max_distance = 100.0
collision_threshold = 60.0
min_hits_during_collision = 30
```

### Backend (Actual Config File - CONFIRMED ISSUES)
```json
{
  "vision": {
    "detection": {
      "detection_backend": "opencv",  // BUG CONFIRMED: Should be "yolo"
      "enable_ball_detection": true,
      "use_opencv_validation": true   // OK
    },
    // BUG CONFIRMED: NO "tracking" section at all!
    // This means tracker gets empty config {} and uses hard-coded defaults
    "processing": {
      "enable_tracking": true,  // OK: Tracking is enabled
      "tracking_max_distance": 50  // Only one tracking param
    }
  }
}
```

**CRITICAL:** The `vision.tracking` config section is **completely missing**! This means the tracker is initialized with an **empty config dictionary**, causing it to use hard-coded defaults in `ObjectTracker.__init__()`:

```python
# From backend/vision/tracking/tracker.py lines 237-247
self.max_age = config.get("max_age", 30)           # Default: 30 (same as video_debugger)
self.min_hits = config.get("min_hits", 3)          # Default: 3 (video_debugger uses 10!)
self.return_tentative_tracks = config.get("return_tentative_tracks", False)  # Default: False (OK)
```

**ROOT CAUSE IDENTIFIED:** The default `min_hits=3` is **WAY TOO LOW** for ghost filtering. video_debugger uses `min_hits=10` which requires 10 consecutive detections before confirming a track, effectively filtering out transient ghost detections.

---

## Root Cause Summary

### Confirmed Bugs in Backend Configuration

1. ✅ **CRITICAL BUG:** Missing `vision.tracking` config section
   - **Impact:** Tracker uses default `min_hits=3` instead of required `min_hits=10`
   - **Result:** Ghost balls confirmed too quickly, appear in output
   - **Fix:** Add tracking config section with proper parameters

2. ✅ **BUG:** Wrong detection backend
   - **Current:** `"detection_backend": "opencv"`
   - **Should be:** `"detection_backend": "yolo"`
   - **Impact:** Less accurate detection, more false positives
   - **Fix:** Change backend to YOLO

3. ✅ **ISSUE:** OpenCV validation is enabled but YOLO backend not used
   - **Current:** `"use_opencv_validation": true` but backend is OpenCV
   - **Impact:** Hybrid detection not working (no YOLO to validate)
   - **Fix:** Must use YOLO backend for hybrid mode to work

## Recommendations

### Immediate Fixes (Priority 1)
1. **Add Tracking Config Section:** Add complete `vision.tracking` section to config with:
   - `min_hits: 10` (CRITICAL for ghost filtering)
   - `return_tentative_tracks: false`
   - `max_age: 30`
   - `max_distance: 100.0`
   - `collision_threshold: 60.0`
   - `min_hits_during_collision: 30`
2. **Switch to YOLO Backend:** Change `vision.detection.detection_backend: "yolo"`
3. **Verify Tracking Enabled:** Ensure `vision.processing.enable_tracking: true` (already OK)

### Code Changes (Priority 2)
5. **Use Hybrid Detection Method:** Change line 1020 in `/backend/vision/__init__.py` to call `detect_balls_with_classification()`
6. **Add Fallback Validation:** Ensure tracker is always created even if config is missing
7. **Add Detection Logging:** Log detection counts and tracker stats every N frames (like video_debugger does)

### Configuration Template (Priority 3)
8. **Create Reference Config:** Export video_debugger's working config to a JSON template file
9. **Config Validation:** Add startup validation to check critical tracking parameters
10. **Default Overrides:** Hard-code safe defaults if config values are missing

---

## Fixed Configuration Template

Add this to `/config/current.json` under the `vision` section:

```json
{
  "vision": {
    "detection": {
      "detection_backend": "yolo",
      "yolo_model_path": "models/yolov8n-pool-1280.onnx",
      "yolo_confidence": 0.15,
      "yolo_nms_threshold": 0.45,
      "yolo_device": "cpu",
      "use_opencv_validation": true,
      "fallback_to_opencv": true,
      "enable_ball_detection": true,
      "enable_cue_detection": true,
      "enable_table_detection": true,
      "min_ball_size": 20
    },
    "tracking": {
      "max_age": 30,
      "min_hits": 10,
      "max_distance": 100.0,
      "process_noise": 5.0,
      "measurement_noise": 20.0,
      "collision_threshold": 60.0,
      "min_hits_during_collision": 30,
      "motion_speed_threshold": 10.0,
      "return_tentative_tracks": false
    },
    "processing": {
      "enable_tracking": true,
      "enable_preprocessing": false,
      "use_gpu": false
    }
  }
}
```

**Key Changes:**
1. Added complete `vision.tracking` section (was missing)
2. Changed `detection_backend` from `"opencv"` to `"yolo"`
3. Set `min_hits: 10` (was defaulting to 3)
4. Set `return_tentative_tracks: false` (critical)
5. Added YOLO-specific parameters matching video_debugger

---

## Testing Recommendations

### Unit Tests Needed
1. Test that `enable_tracking=True` in config results in tracker being initialized
2. Test that tracker config is loaded correctly from config file
3. Test that YOLO+OpenCV hybrid detection is used when backend is "yolo"
4. Test that tentative tracks are NOT returned when `return_tentative_tracks=False`

### Integration Tests Needed
5. Run backend with same video file as video_debugger, compare ball counts
6. Log detection vs tracked ball counts frame-by-frame
7. Measure ghost ball appearance rate with different min_hits values
8. Compare processing time between backends

### Configuration Tests Needed
9. Test with missing tracking config (should use safe defaults)
10. Test with tracking disabled (should log warning but not crash)
11. Test with OpenCV backend vs YOLO backend (compare accuracy)

---

## Summary of Differences

| Category | video_debugger.py | Backend | Result |
|----------|-------------------|---------|--------|
| **Detection Method** | YOLO+OpenCV hybrid | Configurable (likely OpenCV only) | Backend less accurate |
| **Tracking Enabled** | Always | Conditional (may be disabled) | Backend may skip tracking |
| **Ghost Filtering** | Strong (min_hits=10, no tentative) | Weak (min_hits=3, may return tentative) | Backend shows ghosts |
| **Configuration** | Hard-coded proven values | Loaded from file (may be wrong) | Backend unreliable |
| **Fallback Strategy** | Auto-fallback to OpenCV | Configurable | Both can work |
| **Pipeline Complexity** | Simple, direct | Abstracted, multi-layer | Backend harder to debug |

**Root Cause:** The backend's **tracker configuration** is wrong or tracking is disabled entirely. This causes raw detections (including ghosts) to be returned instead of confirmed tracks only.

**Primary Fix:** Update config file to match video_debugger's proven tracker settings:
- `min_hits: 10`
- `return_tentative_tracks: false`
- `enable_tracking: true`
- `detection_backend: "yolo"`
- `use_opencv_validation: true`
