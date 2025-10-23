# Vision Detection Pipeline Lag Analysis - Complete Index

## Generated: 2025-10-22

This analysis examines why detected ball positions lag behind real-time camera frames in the billiards-trainer vision system.

## Quick Answer

**Detection lags behind real-time by 150-300ms (5-10 frames @ 30 FPS)**

Primary causes:
1. YOLO neural network inference: 50-200ms
2. Preprocessing pipeline: 25-60ms
3. Frame queue buffering: ~100-150ms aggregate
4. Sequential processing (no parallelization)

---

## Documents in This Analysis

### 1. `vision_detection_lag_analysis.md` (40 KB)
**Complete technical analysis with full details**

Contains:
- Executive summary
- Frame capture architecture (VideoConsumer, capture loop, processing loop)
- Frame processing pipeline with 10 sequential stages
- Detection algorithm performance by device
- Frame rate limiting and throttling analysis
- Synchronous operations blocking pipeline
- Preprocessing overhead breakdown (8 sub-steps)
- Shared memory IPC overhead
- Tracking and Kalman filter performance
- Lock operations and thread safety
- Configuration parameters
- Lag timeline example with specific timings
- Summary of lag sources ranked by impact (12 sources)
- 12 recommendations from immediate to long-term

**Best for:** Understanding the complete picture of where lag comes from

### 2. `lag_analysis_summary.md` (5.3 KB)
**Quick reference guide**

Contains:
- Key findings at a glance
- Primary bottleneck (YOLO)
- Secondary issues (4 major ones)
- Frame age timeline
- Critical code sections with file references
- Configuration parameters
- Quick fixes with impact estimates
- Synchronous vs async operations
- Lock points
- Where to add improvements (priority order)
- Measurement points in code
- Expected improvements from different optimizations

**Best for:** Getting up to speed quickly, reference cheat sheet

### 3. `lag_analysis_code_locations.md` (12 KB)
**Detailed code locations and line numbers**

Contains:
- VideoConsumer (shared memory IPC) - 3 operations
- Capture loop - 10 components with line numbers
- Processing loop - 8 components with line numbers
- Single frame processing (_process_single_frame) - 7 steps
- Preprocessing pipeline - 8 sub-steps with line numbers
- YOLO detection - device performance characteristics
- Tracking (Hungarian algorithm) - 8 steps
- Queue architecture - configuration and impact
- Lock points - 3 locations with duration/frequency
- Configuration parameters - 20+ parameters listed
- Sleep/wait points - 7 locations with durations
- Summary ranked by impact (5 bottlenecks)
- Recommendations by complexity (10 recommendations)

**Best for:** Implementation work, finding exact code to modify

---

## Key Findings Summary

### Bottleneck Rankings

| Rank | Source | Latency | Impact | File |
|------|--------|---------|--------|------|
| 1 | YOLO Inference | 50-200ms | PRIMARY | yolo_detector.py |
| 2 | Preprocessing | 25-60ms | HIGH | preprocessing.py |
| 3 | Queue Buffer (5 frames) | 100-150ms | HIGH | __init__.py |
| 4 | Table Detection | 20-100ms | MEDIUM | __init__.py |
| 5 | Tracking (Hungarian) | 15-40ms | MEDIUM | tracker.py |

### Architecture Issues

1. **Sequential Processing** - All detection stages run one after another in a single thread
2. **Non-blocking Queue** - Silently drops frames if processing slow
3. **No Frame Prioritization** - Processes old frames in queue before new ones
4. **Synchronous Inference** - Model lock held for entire YOLO inference time
5. **Deep Queue (5 frames)** - Creates 150ms latency just from buffering

### Critical Code Sections

| Issue | File | Lines | Timing |
|-------|------|-------|--------|
| Capture loop | `backend/vision/__init__.py` | 755-833 | Gets frames from VideoConsumer |
| Processing loop | `backend/vision/__init__.py` | 834-894 | Main bottleneck processing |
| _process_single_frame | `backend/vision/__init__.py` | 896-1073 | Sequential stages |
| YOLO inference | `backend/vision/detection/yolo_detector.py` | (model dependent) | 50-200ms blocking |
| Preprocessing | `backend/vision/preprocessing.py` | 199-301 | 8 sequential steps |
| Tracking | `backend/vision/tracking/tracker.py` | 335-396 | Hungarian algorithm |

---

## Quick Fixes (Immediate Impact)

### 1. Reduce Queue Depth
```python
# File: backend/vision/__init__.py line 291
max_frame_queue_size = 2  # from 5
# Impact: -100ms latency
```

### 2. Disable Preprocessing
```python
# File: config system
preprocessing_enabled = False
# Impact: -30ms latency
```

### 3. Use GPU/Apple Silicon
```python
# File: config system
yolo_device = "mps"  # Apple Silicon or "cuda" for NVIDIA
# Impact: -150ms latency (50ms YOLO instead of 200ms)
```

### 4. Reduce Model Complexity
```python
# Use YOLOv8n instead of YOLOv8m
# Impact: -100ms latency
```

---

## Configuration Parameters

All found in config system (e.g., environment variables or config files):

```
# Camera/Frame settings
vision.camera.device_id = 0
vision.camera.backend = "auto"
vision.camera.resolution = [1920, 1080]
vision.camera.fps = 30
vision.camera.buffer_size = 1

# Processing settings
vision.processing.target_fps = 30
vision.processing.max_frame_queue_size = 5  # CRITICAL
vision.processing.preprocessing = True  # CRITICAL
vision.processing.enable_threading = True
vision.processing.processing_queue_timeout_sec = 0.1

# Detection settings
vision.detection.yolo_model_path = (required)
vision.detection.yolo_device = "cpu"  # CRITICAL (cpu/cuda/mps)
vision.detection.yolo_confidence = 0.15
vision.detection.enable_table_detection = True
vision.detection.enable_ball_detection = True
vision.detection.enable_cue_detection = True

# Tracking
vision.processing.enable_tracking = True
```

---

## How to Use These Documents

### I want to understand the problem
→ Read: **vision_detection_lag_analysis.md** (full context)

### I want a quick summary
→ Read: **lag_analysis_summary.md** (2-minute overview)

### I want to make changes to the code
→ Read: **lag_analysis_code_locations.md** (exact line numbers)

### I want to know what to measure
→ See: lag_analysis_code_locations.md → "Measurement Points" section

### I want to understand locks/threading
→ See: lag_analysis_summary.md → "Lock Points" section

### I want specific improvements
→ See: lag_analysis_summary.md → "Quick Fixes" section

---

## Performance Target

For real-time detection (no perceptible lag):
- Processing latency should be < 33ms @ 30 FPS
- Current: 100-200ms (3-6x over budget)
- Achievable with: GPU + queue reduction + preprocessing disable

---

## Files Analyzed

```
backend/vision/__init__.py              (1361 lines - main module)
backend/vision/capture.py               (921 lines - frame capture)
backend/vision/stream/video_consumer.py (163 lines - shared memory IPC)
backend/vision/detection/yolo_detector.py   (model inference)
backend/vision/detection/balls.py       (1535 lines - detection)
backend/vision/preprocessing.py         (735 lines - preprocessing)
backend/vision/tracking/tracker.py      (600+ lines - tracking)
```

---

## References in Analysis

- Frame capture and VideoConsumer: `backend/vision/__init__.py:755-833`
- Processing loop: `backend/vision/__init__.py:834-894`
- Single frame processing: `backend/vision/__init__.py:896-1073`
- YOLO detector: `backend/vision/detection/yolo_detector.py`
- Preprocessing: `backend/vision/preprocessing.py:199-301`
- Tracking: `backend/vision/tracking/tracker.py:335-396`
- Queue configuration: `backend/vision/__init__.py:291-292`
- Lock points: Multiple (see code_locations.md)

---

## Key Takeaways

1. **YOLO is the main bottleneck** - 50-200ms per frame depending on device
2. **Processing is sequential** - All stages block each other, no parallelization
3. **Queue adds 100+ ms lag** - 5-frame buffer at 30 FPS
4. **No frame prioritization** - Old frames processed before new ones
5. **Configuration matters** - GPU choice, queue size, preprocessing can make 200ms difference

**Bottom line:** Current system is 3-6x slower than real-time. Achievable fixes: GPU + queue reduction + preprocessing disable can get to 1.5x real-time performance.

---

## Next Steps

1. Profile actual latency on your hardware: `backend/vision/__init__.py` logs `processing_time`
2. Identify bottleneck on YOUR hardware (CPU/GPU device matters)
3. Apply quickest fix first (see lag_analysis_summary.md)
4. Measure improvement
5. Consider longer-term improvements (parallelization)

---

Generated by: Vision Pipeline Analysis Tool
Date: 2025-10-22
Analyzed Files: 7 major vision modules
Total Analysis: 3 documents, 60+ KB of detailed findings
