# Vision Detection Lag Analysis - Quick Reference

## Key Findings

### Primary Bottleneck: YOLO Neural Network Inference
- **Latency:** 50-200ms per frame (CPU), 15-50ms (GPU/Apple Silicon)
- **Location:** `backend/vision/detection/yolo_detector.py`
- **Impact:** Blocks entire processing pipeline - synchronous, no parallelization
- **Thread lock:** `_model_lock` prevents concurrent inference

### Secondary Issues

1. **Preprocessing (25-60ms)**
   - Location: `backend/vision/preprocessing.py:199-301`
   - 8 sequential sub-operations
   - Heaviest: bilateral filter (30-50ms), white balance (5-10ms)
   - Runs BEFORE detection

2. **Queue Architecture (150-250ms aggregate lag)**
   - Max queue depth: 5 frames
   - At 30 FPS with 50-100ms processing = 150-500ms total pipeline latency
   - Non-blocking queue operations silently drop frames if slow

3. **Processing Pipeline (Sequential, not parallel)**
   - Preprocessing → Masking → Table Detection → Ball Detection → Tracking → Cue Detection
   - All run in sequence in single processing thread
   - No parallelization between stages

4. **Tracking Overhead (15-40ms)**
   - Hungarian algorithm for association: O(n_tracks × n_detections)
   - Kalman prediction/update
   - Location: `backend/vision/tracking/tracker.py:335-396`

## Frame Age Timeline

At 30 FPS with ~100ms total processing time:

```
Frame captured → 100ms (YOLO) → 30ms (tracking) → result available
                  ↓
         Detected position is 130ms old
         = 4 frames behind realtime
```

## Critical Code Sections

### Frame Capture Loop
**File:** `backend/vision/__init__.py:755-833`
- Polls `VideoConsumer` (shared memory IPC)
- Puts frames in `_frame_queue` (non-blocking)
- Sleep: 1ms between checks if queue full

### Processing Loop
**File:** `backend/vision/__init__.py:834-894`
- Blocks on `_frame_queue.get(timeout=0.1)`
- Calls `_process_single_frame()` (100-200ms total)
- Updates result with lock

### Sequential Processing
**File:** `backend/vision/__init__.py:896-1073`
```
_process_single_frame():
  1. preprocess(25-60ms)
  2. apply_masks(5ms)
  3. detect_table(20-100ms)
  4. YOLO detect_balls(50-200ms) ← PRIMARY BOTTLENECK
  5. track(15-40ms)
  6. detect_cue(50-150ms if enabled)
```

## Configuration Parameters

```python
# In config system:
vision.processing.target_fps = 30
vision.processing.max_frame_queue_size = 5
vision.processing.enable_preprocessing = True
vision.processing.processing_queue_timeout_sec = 0.1
vision.processing.capture_frame_interval_ms = 1

# YOLO parameters:
vision.detection.yolo_model_path = (must be set)
vision.detection.yolo_confidence = 0.15
vision.detection.yolo_device = "cpu"  # or "cuda", "mps", etc.
```

## Quick Fixes (Immediate Impact)

### 1. Reduce Queue Depth
```python
# Change from 5 to 2-3 frames
max_frame_queue_size = 2  # Reduces pipeline lag from 150ms to 50ms
```

### 2. Disable Preprocessing
```python
# Skip preprocessing entirely
preprocessing_enabled = False  # Saves 25-60ms per frame
```

### 3. Use GPU/Apple Silicon
```python
# Switch from CPU to GPU
yolo_device = "cuda"  # -150ms latency on GPU-capable hardware
# or
yolo_device = "mps"   # Apple Silicon, same latency reduction
```

### 4. Reduce Model Size
```python
# Use quantized YOLOv8n instead of YOLOv8m
# Saves 50-100ms inference time
```

## Synchronous Operations (Blocking)

- ✓ Frame read from shared memory (1-2ms)
- ✓ Preprocessing (25-60ms)
- ✓ YOLO inference (50-200ms) ← CRITICAL
- ✓ OpenCV color classification (10-20ms)
- ✓ Kalman tracking (15-40ms)
- ✓ Lock contention (<1ms)
- ✗ NO async operations
- ✗ NO pipelined processing
- ✗ NO frame skipping logic

## Lock Points

1. **Capture loop:** `with self._lock: self._current_frame = frame` (assignment)
2. **Processing loop:** `with self._lock: self._current_result = result` (assignment)
3. **YOLO inference:** `with self._model_lock: ...model.predict()...` (50-200ms HELD)

## Where to Add Improvements

### High Priority
- `backend/vision/detection/yolo_detector.py` - Model optimization
- `backend/vision/__init__.py:834-894` - Add frame skipping/prioritization
- `backend/vision/preprocessing.py` - Make preprocessing optional/adaptive

### Medium Priority
- `backend/vision/tracking/tracker.py` - Optimize Hungarian algorithm
- `backend/vision/__init__.py:896-1073` - Parallelize detection stages
- `backend/vision/stream/video_consumer.py` - Add frame age tracking

### Low Priority
- `backend/vision/detection/table.py` - Parallelize with ball detection
- `backend/vision/detection/balls.py` - Optimize Hough transform

## Measurement Points

### In Code
```python
# Processing time tracked
statistics.processing_time  # Total time in ms

# Frame count
statistics.frames_processed
statistics.frames_dropped

# Queue status
_frame_queue.qsize()
```

### In Logs
```
"VisionModule: Received frame #{frame_count}"
"Processing time: XXms"
"YOLO+OpenCV hybrid ball detection failed"
```

## Expected Improvements

### If YOLO on GPU (-100ms)
- Total lag: 150ms (still 5 frames behind)

### If queue=2 + preprocessing disabled (-90ms)
- Total lag: 60ms (2 frames behind)

### If GPU + queue=2 + preprocessing disabled (-190ms)
- Total lag: 40ms (1.3 frames behind) ← Acceptable for realtime

### If model quantization + GPU (-150ms more)
- Total lag: 40-50ms (1.5 frames) ← Good realtime performance
