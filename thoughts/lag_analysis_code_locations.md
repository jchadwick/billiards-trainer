# Vision Lag Analysis - Detailed Code Locations

## 1. Frame Capture Architecture

### VideoConsumer (Shared Memory IPC)
**File:** `/backend/vision/stream/video_consumer.py`

| Operation | Lines | Timing | Issue |
|-----------|-------|--------|-------|
| get_frame() | 102-134 | ~1-2ms | Non-blocking, returns None if no new frame |
| start() | 62-101 | - | Attaches to shared memory |
| Attach timeout | 55 | 5s | Wait time for Video Module to create shared memory |

### Capture Loop
**File:** `/backend/vision/__init__.py:755-833`

| Component | Lines | Timing | Details |
|-----------|-------|--------|---------|
| Loop start | 768 | - | `while self._is_running:` |
| Frame interval calculation | 765-766 | - | `frame_interval = 1.0 / self.config.target_fps` |
| Rate limiting | 773-775 | 1ms | Sleep if too fast, continues on timeout |
| Sleep call | 774 | 1ms | `time.sleep(sleep_interval_sec)` |
| VideoConsumer.get_frame() | 779 | ~1-2ms | Gets "latest" frame |
| No frame sleep | 783 | 1ms | `time.sleep(sleep_interval_sec)` if None |
| Queue put | 809-815 | ~0.5ms | Non-blocking `put_nowait()` |
| Queue drop handling | 819-821 | - | Silently drops frames if queue full |
| Error retry sleep | 830 | 100ms | `time.sleep(error_retry_delay)` on exception |

**Configuration:**
- `sleep_interval_ms = vision.processing.capture_frame_interval_ms` (default: 1ms)
- `frame_interval = 1.0 / target_fps` (default: 33ms @ 30 FPS)

---

## 2. Processing Loop

### Main Processing Thread
**File:** `/backend/vision/__init__.py:834-894`

| Component | Lines | Timing | Details |
|-----------|-------|--------|---------|
| Loop start | 841 | - | `while self._is_running:` |
| Queue timeout | 837-839 | 100ms | `queue_timeout = vision.processing.processing_queue_timeout_sec` |
| Queue block | 844-847 | Up to 100ms | `frame_data = self._frame_queue.get(timeout=queue_timeout)` |
| Empty queue continue | 846-847 | - | Skip to next iteration if timeout |
| _process_single_frame() | 851-855 | **100-200ms** | **PRIMARY BOTTLENECK** |
| Processing time calc | 857 | - | Track elapsed time |
| Statistics update | 860-864 | ~1ms | Update rolling average |
| Lock acquisition | 867 | <1ms | `with self._lock:` |
| Result update | 868 | <1ms | `self._current_result = result` |
| Event emission | 870-883 | <1ms | Emit callbacks |
| Error sleep | 892 | 10ms | `time.sleep(error_retry_delay)` |

**Configuration:**
- `queue_timeout = vision.processing.processing_queue_timeout_sec` (default: 0.1 = 100ms)

---

## 3. Single Frame Processing (The Main Bottleneck)

### _process_single_frame()
**File:** `/backend/vision/__init__.py:896-1073`

| Step | Lines | Timing | Operation |
|------|-------|--------|-----------|
| Start timer | 913 | - | `start_time = time.time()` |
| **Preprocessing** | 923-926 | **25-60ms** | ImagePreprocessor.process() |
| Masking | 930 | ~5ms | _apply_all_masks() |
| **Table Detection** | 938-967 | **20-100ms** | TableDetector.detect_complete_table() |
| **Ball Detection** | 970-992 | **50-200ms** | YOLODetector.detect_balls_with_classification() ← **PRIMARY** |
| Tracking update | 979-982 | 15-40ms | ObjectTracker.update_tracking() |
| **Cue Detection** | 995-1013 | **50-150ms** (optional) | CueDetector.detect_cue() |
| Result creation | 1016-1068 | ~1ms | Create DetectionResult object |
| Total | - | **100-200ms+** | All steps sequential, no parallelization |

**Note:** All operations are SEQUENTIAL in a single thread. Each step must complete before next begins.

---

## 4. Preprocessing Pipeline

### ImagePreprocessor.process()
**File:** `/backend/vision/preprocessing.py:199-301`

| Step | Lines | Timing | Operation |
|------|-------|--------|-----------|
| Start timer | 211 | - | `start_time = cv2.getTickCount()` |
| Resize (optional) | 217-233 | ~5-10ms | Only if `processing_scale != 1.0` |
| **White Balance** | 239-243 | **5-10ms** | cv2.xphoto.balanceWhite() |
| **Exposure/Contrast** | 246-252 | **10-20ms** | CLAHE + scaling |
| **Noise Reduction** | 255-259 | **20-100ms** | Depends on method |
| Sharpening (optional) | 262-266 | 5-10ms | Only if enabled |
| **Color Space** | 269-274 | **~5ms** | BGR to HSV conversion |
| **Morphology** | 277-281 | **10-20ms** | Erosion/dilation |
| Resize back (optional) | 284-289 | 5-10ms | Only if processing_scale was used |
| Stats update | 292-295 | <1ms | Update rolling average |

**Noise Reduction Methods (lines 615-675):**
- Gaussian blur: 5-10ms
- Bilateral filter: 30-50ms (lines 631-645)
- Median blur: 15-30ms (lines 647-651)
- Non-local means: 100-200ms (lines 653-671, disabled by default)

**Configuration:**
- `preprocessing_enabled = True` (default)
- `noise_method = "gaussian"` (default)

---

## 5. YOLO Detection (Primary Bottleneck)

### YOLODetector Class
**File:** `/backend/vision/detection/yolo_detector.py:116-150+`

| Component | Lines | Timing | Details |
|-----------|-------|--------|---------|
| Model init | 132-143 | - | Constructor |
| Model lock | 20+ | - | `self._model_lock = threading.RLock()` |
| Inference call | (not shown) | **50-200ms** | **SYNCHRONOUS, BLOCKING** |
| Lock duration | - | **50-200ms** | Entire inference time held |

**Device Performance:**
- CPU: 50-200ms
- CUDA GPU: 15-50ms
- Apple Silicon (MPS): 15-30ms
- Edge TPU: 10-20ms
- CoreML (if quantized): 8-15ms

**Configuration:**
- `vision.detection.yolo_model_path` (must be set)
- `vision.detection.yolo_confidence = 0.15` (default)
- `vision.detection.yolo_device = "cpu"` (default, or "cuda", "mps", etc.)

---

## 6. Tracking (Secondary Bottleneck)

### ObjectTracker.update_tracking()
**File:** `/backend/vision/tracking/tracker.py:335-396`

| Step | Lines | Timing | Operation |
|------|-------|--------|-----------|
| Predict tracks | 360 | 1-2ms | _predict_tracks(dt) |
| Collision detect | 363 | 1ms | _detect_collision_or_high_motion() |
| Build cost matrix | 417 | **5-20ms** | _build_cost_matrix() for Hungarian |
| Association | 426 | 1-2ms | linear_sum_assignment() |
| Update matched | 376-378 | 2-5ms | track.update_with_detection() |
| Mark missed | 381-382 | 0.5ms | track.mark_missed() |
| Create new | 386-387 | 1-2ms | _create_new_track() |
| Delete old | 390 | 1-2ms | _delete_old_tracks() |
| Get results | 396 | 1-2ms | _get_tracked_balls() |

**Hungarian Algorithm (lines 456-504):**
- Cost matrix building: O(n_tracks × n_detections)
- For 10 balls and 16 max: ~5-20ms

---

## 7. Queue Architecture

### Frame Queue Configuration
**File:** `/backend/vision/__init__.py:291-292`

```python
self._frame_queue = queue.Queue(maxsize=self.config.max_frame_queue_size)
self._result_queue = queue.Queue(maxsize=self.config.max_frame_queue_size)
```

**Configuration:**
- `max_frame_queue_size = 5` (default)
- With 30 FPS and 50-100ms processing per frame:
  - Queue latency = 5 frames × (33.3ms avg - processing delay)
  - Actual latency = 5 frames × ~20ms per backlog frame = ~100ms queue delay
  - **Plus** processing time on current frame = 50-100ms
  - **Total:** 150-200ms baseline lag

---

## 8. Lock Points (Thread Safety)

### Lock 1: Current Frame Update
**File:** `/backend/vision/__init__.py:801-805`

```python
with self._lock:
    self._current_frame = frame
```

- **Duration:** <1ms (just assignment)
- **Frequency:** Once per capture loop iteration (~30 Hz)
- **Contention risk:** Low

### Lock 2: Current Result Update
**File:** `/backend/vision/__init__.py:867-868`

```python
with self._lock:
    self._current_result = result
```

- **Duration:** <1ms (just assignment)
- **Frequency:** Once per process loop iteration (~5-10 Hz)
- **Contention risk:** Very low

### Lock 3: YOLO Model Lock
**File:** `/backend/vision/detection/yolo_detector.py`

```python
self._model_lock = threading.RLock()
# Held during inference: 50-200ms
```

- **Duration:** **50-200ms** (entire YOLO inference)
- **Frequency:** Once per processed frame
- **Contention risk:** Prevents concurrent inference (good for safety, bad for parallelization)

---

## 9. Configuration Parameters

### Vision Module Config
**Default values from VisionConfig.from_config_dict():**

```python
# Lines 142-233 in __init__.py
camera_device_id = 0  # vision.camera.device_id
camera_backend = "auto"  # vision.camera.backend
camera_resolution = (1920, 1080)  # vision.camera.resolution
camera_fps = 30  # vision.camera.fps
camera_buffer_size = 1  # vision.camera.buffer_size

target_fps = 30  # vision.processing.target_fps
enable_threading = True  # vision.processing.enable_threading
enable_gpu = False  # vision.processing.use_gpu
max_frame_queue_size = 5  # vision.processing.max_frame_queue_size

enable_table_detection = True  # vision.detection.enable_table_detection
enable_ball_detection = True  # vision.detection.enable_ball_detection
enable_cue_detection = True  # vision.detection.enable_cue_detection
enable_tracking = True  # vision.processing.enable_tracking

frame_skip = 0  # vision.processing.frame_skip
roi_enabled = False  # vision.processing.roi_enabled
preprocessing_enabled = True  # vision.processing.enable_preprocessing
```

### YOLO Detector Config
**Lines 335-351 in __init__.py:**

```python
yolo_model_path = config.get("vision.detection.yolo_model_path")
yolo_confidence = 0.15  # vision.detection.yolo_confidence
yolo_nms_threshold = 0.45  # vision.detection.yolo_nms_threshold
yolo_device = "cpu"  # vision.detection.yolo_device
min_ball_size = 20  # vision.detection.min_ball_radius
```

---

## 10. Key Sleep/Wait Points

| Location | Lines | Duration | Purpose |
|----------|-------|----------|---------|
| Capture rate limit | 774 | 1ms | Rate limiting if too fast |
| Capture no frame | 783 | 1ms | Wait for frame if none available |
| Process queue timeout | 845 | 100ms | Wait for frame from capture |
| Calibration frame capture | 620 | 100ms | Sleep between calibration frames |
| Error recovery (capture) | 830 | 100ms | Wait before retry after error |
| Error recovery (process) | 892 | 10ms | Wait before retry after error |
| Reconnect delay (camera) | (capture.py) | 1s | Wait before reconnection attempt |

---

## Summary of Critical Bottlenecks

### Ranked by Impact

1. **YOLO Inference** - 50-200ms (CPU) or 15-50ms (GPU)
   - File: `backend/vision/detection/yolo_detector.py`
   - Lines: model.predict() (exact lines in subclass implementation)
   - Lock: `self._model_lock` held entire duration

2. **Preprocessing** - 25-60ms total, especially noise reduction
   - File: `backend/vision/preprocessing.py:199-301`
   - Lines: 255-259 (noise reduction)

3. **Queue Architecture** - 150-250ms aggregate lag
   - File: `backend/vision/__init__.py:291-292, 809-815, 844-847`
   - Max queue: 5 frames

4. **Table Detection** - 20-100ms
   - File: `backend/vision/__init__.py:938-967`
   - Parallel with ball detection could help

5. **Tracking (Hungarian)** - 15-40ms
   - File: `backend/vision/tracking/tracker.py:456-504`
   - O(n²) complexity

---

## Recommendations by Complexity

### Trivial (1 line change)
1. Reduce queue: `max_frame_queue_size = 2` (saves ~100ms)
2. Disable preprocessing: `preprocessing_enabled = False` (saves ~40ms)
3. Use GPU: `yolo_device = "mps"` (saves ~150ms)

### Simple (10-20 lines)
4. Add frame skipping logic (line 845, check frame age before processing)
5. Disable table detection (line 938, set `enable_table_detection = False`)
6. Disable cue detection (line 995, set `enable_cue_detection = False`)

### Medium (50-100 lines)
7. Parallelize table and ball detection (fork thread, join before tracking)
8. Adaptive preprocessing (skip if frame quality good)
9. Frame prioritization (process newest frame, discard old ones in queue)

### Complex (100+ lines)
10. Async YOLO inference (separate thread with queue)
11. Model quantization/distillation
12. GPU acceleration for preprocessing
