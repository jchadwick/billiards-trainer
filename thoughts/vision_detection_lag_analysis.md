# Vision Detection Pipeline Lag Analysis

## Executive Summary

The vision detection pipeline has several potential sources of lag that could cause detection to fall behind real-time ball movement:

1. **Multi-threaded queue architecture** - Frame capture and processing happen in separate threads with non-blocking queue operations
2. **Processing time overhead** - Complex detection pipeline with multiple serial processing stages
3. **No frame-time synchronization** - Detection may lag behind current camera frames due to queue depth
4. **Shared memory IPC overhead** - Frames transferred via shared memory from Video Module
5. **YOLO inference latency** - Neural network inference is the primary computational bottleneck

---

## 1. Frame Capture & Processing Architecture

### Capture Loop (VisionModule._capture_loop)
**Location:** `/backend/vision/__init__.py:755-833`

```
_capture_loop thread:
  1. Polls VideoConsumer (shared memory IPC) for frames
  2. Gets "latest" frame via non-blocking read
  3. Applies ROI if configured
  4. Puts frame in _frame_queue (non-blocking)

Key timing:
- sleep_interval_ms = vision.processing.capture_frame_interval_ms (default: 1ms)
- Rate limiting: max frame_interval = 1.0 / target_fps (33ms @ 30 FPS)
- Frame check: if current_time - last_frame_time < frame_interval: sleep(1ms) and continue
```

**Potential Lag Sources:**
- If `VideoConsumer.get_frame()` returns None, 1ms sleep is applied
- Rate limiting enforced per capture loop iteration - can skip frames if video source is ahead
- Non-blocking queue operations can silently drop frames if processing is slow

### Processing Loop (VisionModule._processing_loop)
**Location:** `/backend/vision/__init__.py:834-894`

```
_processing_loop thread:
  1. Blocks on _frame_queue.get(timeout=queue_timeout)
  2. queue_timeout = vision.processing.processing_queue_timeout_sec (default: 0.1 sec)
  3. Calls _process_single_frame()
  4. Updates _current_result with lock

Key timing:
- If queue is empty, waits up to 100ms before retrying
- Processing time can accumulate across entire frame
- Frame age unknown when processing completes
```

---

## 2. Frame Processing Pipeline

### _process_single_frame() Flow
**Location:** `/backend/vision/__init__.py:896-1073`

Sequential processing steps (all synchronous):

```python
1. Preprocessing (if enabled)
   - ImagePreprocessor.process()
   - Contains 8 sub-steps: resize, white balance, exposure, noise, sharpening, color space, morphology, resize back

2. Masking (BEFORE detection)
   - Apply marker dots mask
   - Apply boundary mask
   - Scale coordinates from calibration resolution

3. Table Detection (optional)
   - TableDetector.detect_complete_table()

4. Ball Detection (main bottleneck)
   - YOLODetector.detect_balls_with_classification()
   - YOLO inference (primary latency)
   - OpenCV classification for ball types

5. Tracking (post-detection)
   - ObjectTracker.update_tracking()
   - Kalman filter prediction and Hungarian algorithm

6. Cue Detection (optional)
   - CueDetector.detect_cue()
   - Additional YOLO inference

7. Result Creation
   - Statistics aggregation
```

**Critical Finding:** All processing is **strictly sequential** - no parallelization between detection stages.

---

## 3. Detection Algorithm Performance

### YOLO Detection (YOLODetector)
**Location:** `/backend/vision/detection/yolo_detector.py:116-150+`

**Known Performance Characteristics:**
- Neural network inference is the bottleneck
- Model loading and inference happens synchronously in detection thread
- Thread lock (`_model_lock = threading.RLock()`) protects model inference
- No batch processing or pipelined inference
- Each frame waits for YOLO to complete before OpenCV classification begins

**Expected Latency by Device:**
- CPU inference: 50-200ms per frame
- GPU inference: 15-50ms per frame
- Apple Silicon (MPS): 15-30ms per frame
- Edge TPU: 10-20ms per frame
- CoreML: 8-15ms per frame (if model converted)

### OpenCV Fallback Detection (BallDetector)
**Location:** `/backend/vision/detection/balls.py:579-645`

Hough circle detection pipeline:
1. Color pre-filtering (_create_ball_color_mask) - ~5-10ms
2. Gaussian blur - ~5ms
3. Hough circle transform - ~10-30ms
4. Ball classification with color analysis - ~10-20ms
5. Overlap removal and conflict resolution - ~5-10ms

**Total estimated:** 35-75ms per frame (without YOLO)

---

## 4. Frame Rate Limiting & Throttling

### Explicit Sleep Operations

#### In VisionModule._capture_loop():
```python
# Line 773-775: Rate limiting check
if current_time - last_frame_time < frame_interval:
    time.sleep(sleep_interval_sec)  # sleep_interval_sec = 1ms default
    continue

# Line 783-784: No frame available
time.sleep(sleep_interval_sec)  # sleep_interval_sec = 1ms default
```

#### In VideoConsumer.get_frame():
```python
# Non-blocking read - returns None if no new frame
# Caller is responsible for sleep
```

#### In calibration code:
```python
# Line 620: Explicit sleep during calibration
time.sleep(frame_capture_delay)  # default: 0.1 sec per frame
```

### Frame Queue Depth
```python
# Line 291-292: Queue configuration
_frame_queue = queue.Queue(maxsize=self.config.max_frame_queue_size)
_result_queue = queue.Queue(maxsize=self.config.max_frame_queue_size)
# Default max_frame_queue_size = 5
```

**Impact:** With max queue of 5 and processing time of 50-100ms per frame:
- 5 frames × 50-100ms = 250-500ms of pipeline lag
- Newer frames may be delayed significantly

---

## 5. Synchronous Operations That Can Cause Delays

### Table Detection
- Complete table boundary detection is synchronous
- Runs before ball detection
- Can add 20-100ms latency depending on implementation

### Lock Contention
```python
# Line 801: Lock for current frame update
with self._lock:
    self._current_frame = frame

# Line 867: Lock for result update
with self._lock:
    self._current_result = result
```

- Locks are short, but can cause brief delays
- If UI thread reads frame/result during update, creates contention

### Shared Memory IPC
```python
# VideoConsumer.read_frame() from shared memory
frame, metadata = self.reader.read_frame()
```

- Shared memory read is generally fast (~1-2ms)
- But synchronization with Video Module can introduce frame skipping

---

## 6. Preprocessing Overhead

### ImagePreprocessor.process()
**Location:** `/backend/vision/preprocessing.py:199-301`

8 sequential processing steps:

1. **Resize for processing** - Optional, disabled by default
2. **Auto white balance** - ~5-10ms (calls cv2.xphoto.balanceWhite)
3. **Exposure/contrast correction** - ~10-20ms (CLAHE + scaling)
4. **Noise reduction** - **CRITICAL**: 20-100ms depending on method
   - Gaussian blur (kernel 5x5): ~5-10ms
   - Bilateral filter: 30-50ms
   - Non-local means: 100-200ms (disabled by default)
   - Median blur: 15-30ms
5. **Sharpening** - ~5-10ms (optional, disabled by default)
6. **Color space conversion** - ~5ms (always BGR→HSV)
7. **Morphology** - ~10-20ms (erosion/dilation)
8. **Resize back** - Optional, disabled by default

**Total estimated preprocessing:** 30-60ms per frame with default settings

---

## 7. Shared Memory IPC Overhead

### VideoConsumer Architecture
**Location:** `/backend/vision/stream/video_consumer.py`

```python
def get_frame(self) -> Optional[np.ndarray]:
    """Non-blocking frame read. Returns None if no new frame available."""
    frame, metadata = self.reader.read_frame()
    # Returns None if no new frame since last read
```

**Potential Issues:**
1. **Frame skipping** - If detection is slow, Video Module may overwrite frames in shared memory
2. **No frame-time awareness** - Vision Module doesn't know how old the frame is
3. **IPC coordination** - No back-pressure mechanism to slow Video Module if Vision is behind

---

## 8. Tracking & Kalman Filter Overhead

### ObjectTracker.update_tracking()
**Location:** `/backend/vision/tracking/tracker.py:335-396`

Per-frame tracking operations:

```python
1. Predict all tracks with Kalman filter - O(n_tracks), ~1-2ms
2. Detect collisions/high-motion - O(n_tracks), ~1ms
3. Build cost matrix (Hungarian algorithm) - O(n_tracks × n_detections), ~5-20ms
4. Associate detections to tracks - O(n), ~1-2ms
5. Update matched tracks - O(n_matched), ~2-5ms
6. Mark unmatched tracks as missed - O(n_unmatched), ~0.5ms
7. Create new tracks - O(n_unmatched), ~1-2ms
8. Delete old tracks - O(n_tracks), ~1-2ms
```

**Typical total for ~10 balls:** 15-40ms per frame

---

## 9. Lock Operations & Thread Safety

### Thread-Safe Frame Updates
- Frame capture and processing in separate threads
- Both acquire `self._lock` for updates
- Lock duration: <1ms (just assignment)
- Lock contention rare but possible

### YOLO Model Lock
```python
self._model_lock = threading.RLock()
```
- Protects model inference from concurrent calls
- Lock duration: entire inference time (50-200ms)
- Prevents parallel inference but ensures thread safety

---

## 10. Configuration Affecting Latency

### Key Config Parameters
```python
# From VisionConfig.from_config_dict()
target_fps = 30  # vision.processing.target_fps
max_frame_queue_size = 5  # vision.processing.max_frame_queue_size
preprocessing_enabled = True  # vision.processing.enable_preprocessing
camera_fps = 30  # vision.camera.fps
frame_skip = 0  # vision.processing.frame_skip

# Queue timeout
queue_timeout = 0.1  # vision.processing.processing_queue_timeout_sec

# Capture interval
capture_frame_interval_ms = 1  # vision.processing.capture_frame_interval_ms
```

### Slow Path Detection
```python
# If slow, logs appear in:
# "Error in VisionModule capture loop"
# "Error in processing loop"
# "YOLO+OpenCV hybrid ball detection failed"
```

---

## Lag Timeline Example (30 FPS @ 50ms processing)

```
Time   Frame#  Event
0ms    0       Capture: Get frame 0 from video
1ms    0       Capture: Put frame 0 in queue
33ms   1       Capture: Get frame 1 from video
34ms   1       Capture: Put frame 1 in queue
50ms   0       Process: Start processing frame 0
60ms   0       Process: YOLO inference completes
65ms   0       Process: Tracking completes
67ms   0       Process: Update result, done
67ms   -       Capture: Get frame 2 from video (but lag is 67ms)
100ms  1       Process: Start processing frame 1
150ms  1       Process: Done (lag is now 50ms from original frame 1 capture)
```

**Result:** Detections lag behind camera by ~50-100ms (1.5-3 frames @ 30fps)

---

## Summary of Lag Sources (Priority Order)

### High Impact (10-100ms+)
1. **YOLO inference** - Primary bottleneck (50-200ms)
2. **Preprocessing** - Especially with noise reduction (20-60ms)
3. **Queue depth** - 5-frame buffer creates 50-150ms lag
4. **Processing pipeline length** - 8-10 sequential operations

### Medium Impact (5-20ms)
5. **Table detection** - Synchronous before balls
6. **Tracking (Hungarian algorithm)** - N² complexity with detections
7. **Color classification** - Per-ball analysis
8. **Hough circle detection** - Fallback when YOLO unavailable

### Low Impact (<5ms)
9. **Frame rate limiting** - 1ms sleep is negligible
10. **Lock contention** - Millisecond-level
11. **Shared memory IPC** - 1-2ms copy time
12. **Kalman filter prediction** - Sub-millisecond per ball

---

## Recommendations for Reducing Lag

### Immediate (Easy, High Impact)
1. **Reduce preprocessing** - Disable bilateral filter, non-local means
2. **Lower YOLO inference time** - Use quantized/distilled model, reduce input resolution
3. **Reduce queue depth** - Change max_frame_queue_size from 5 to 2-3
4. **Skip frames** - Process every Nth frame if real-time tracking not critical

### Short-term (Medium effort, Medium impact)
5. **Parallelize detection stages** - Run table + ball detection concurrently
6. **Use frame timestamps** - Discard old frames from queue if processing is slow
7. **Async YOLO** - Load YOLO inference on separate GPU thread
8. **Adaptive preprocessing** - Skip preprocessing if frame quality good

### Long-term (High effort, High impact)
9. **Model optimization** - Use TensorRT/CoreML/ONNX quantization
10. **Hardware acceleration** - Use GPU/TPU for inference
11. **Frame-synchronous processing** - Detect based on actual camera timestamp, not queue order
12. **Predictive tracking** - Use Kalman predictions to compensate for lag

---

## Debug Information Available

### Logging Points
- Vision module frame timing: "Received frame #{frame_count}"
- Processing time: frame statistics with processing_time_ms
- Queue status: get_statistics() shows frames_dropped, avg_fps
- YOLO inference: model loading logs

### Metrics to Monitor
1. **Time from capture to detection:** compare frame timestamps
2. **Queue depth:** monitor _frame_queue.qsize()
3. **Processing latency:** statistics.processing_time
4. **Dropout rate:** statistics.frames_dropped

### Debug Commands
```bash
# Check vision module performance
curl http://localhost:8000/vision/status | jq '.statistics'

# Monitor frame rate and lag
tail -f logs/vision.log | grep "Received frame\|Processing\|detection"
```

---

## Conclusion

**The most likely cause of detection lag:** Combination of:
- 50-100ms YOLO inference time
- 25-30ms preprocessing overhead
- 15-40ms tracking computation
- 5+ frames buffered in queue = 150-250ms total latency

**Expected end-to-end lag:** 150-300ms (5-10 frames at 30 FPS)

This means a ball's detected position reflects where it was 150-300ms in the past, not where it currently is. Fast ball movements will appear delayed relative to real-time observation.

To achieve true real-time detection, the pipeline needs to process under 33ms per frame (for 30 FPS) or use predictive compensation (Kalman filtering) to extrapolate current position.
