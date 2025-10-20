# Phase 2 Implementation Plan: Shared Memory IPC Architecture

**Estimated Effort**: 55-70 hours (7-9 days)
**Risk Level**: MEDIUM
**Dependencies**: None (can start immediately)
**Parallelizable with**: Phase 3 (Enhanced Detection)

---

## Objectives

1. Extract video capture into separate OS process
2. Implement triple-buffered shared memory for frame distribution
3. Update Vision Module to consume from shared memory
4. Update API streaming to consume from shared memory
5. Achieve <5ms frame latency (down from 20-50ms)
6. Support 10+ concurrent consumers without degradation

---

## Architecture Overview

```
Before (Current):
┌─────────────────────────────────────┐
│   Single Backend Process            │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ EnhancedCameraModule        │   │
│  │ (threading.Lock + copy())   │   │
│  └───────┬─────────────────────┘   │
│          │ frame.copy()            │
│          ├──────────┬──────────┐   │
│          ↓          ↓          ↓   │
│    VisionModule  Stream1   Stream2 │
│                                     │
└─────────────────────────────────────┘

After (Phase 2):
┌──────────────────┐         ┌─────────────────────────────┐
│ Video Process    │         │   Backend Process           │
│                  │         │                             │
│ ┌──────────────┐ │  IPC    │  ┌────────────────────┐    │
│ │VideoProcess  │ │◄────────┤  │ VisionModule       │    │
│ │   camera     │ │ Shared  │  │ (VideoConsumer)    │    │
│ │   capture    │ │ Memory  │  └────────────────────┘    │
│ └──────┬───────┘ │         │  ┌────────────────────┐    │
│        │         │         │  │ Stream Endpoint    │    │
│        ↓         │         │  │ (VideoConsumer)    │    │
│ ┌──────────────┐ │         │  └────────────────────┘    │
│ │SharedMemory  │◄┼─────────┼──┘                         │
│ │Writer        │ │         │                             │
│ │(3 buffers)   │ │         └─────────────────────────────┘
│ └──────────────┘ │
└──────────────────┘
```

---

## Task Breakdown

### Task 2.1: Extract and Adapt Shared Memory Module (15-20h)

**Objective**: Create reusable shared memory infrastructure

#### Subtasks:

**2.1.1 Copy Core Shared Memory Module (2-3h)**
- Copy from v2: `/Users/jchadwick/code/billiards-trainer-v2/backend/video/ipc/shared_memory.py`
- Place in: `/Users/jchadwick/code/billiards-trainer/backend/video/ipc/shared_memory.py`
- Create `__init__.py` with exports

**Files to Create:**
```
backend/video/
├── __init__.py
└── ipc/
    ├── __init__.py
    └── shared_memory.py
```

**2.1.2 Adapt to Current Config System (3-4h)**
- Replace v2's ConfigManager calls with current Config singleton
- Update imports: `from backend.config import config`
- Test configuration loading

**Configuration Keys to Add** (`config.json`):
```json
{
  "video": {
    "shared_memory_name": "billiards_video",
    "shared_memory_attach_timeout_sec": 5.0,
    "process": {
      "shutdown_timeout": 10.0,
      "main_loop_sleep": 0.001
    }
  }
}
```

**2.1.3 Add Unit Tests (5-6h)**
- Test writer initialization
- Test reader attachment with timeout
- Test triple-buffer write/read cycle
- Test fallback to file-backed mmap
- Test cleanup and resource management

**Test File**: `backend/tests/unit/test_shared_memory.py`

**2.1.4 Integration Testing (5-7h)**
- Test with actual camera frames
- Measure latency (target <5ms)
- Test multiple readers
- Test reader attach/detach during operation
- Stress test with 10+ concurrent readers

**Test File**: `backend/tests/integration/test_video_ipc.py`

**Deliverables**:
- ✅ Working shared memory module
- ✅ All tests passing
- ✅ Documentation in docstrings
- ✅ Measured latency <5ms

---

### Task 2.2: Create Video Module Process (20-25h)

**Objective**: Separate camera capture into independent process

#### Subtasks:

**2.2.1 Create VideoProcess Class (8-10h)**

**File**: `backend/video/process.py`

**Core Components**:
```python
class VideoProcess:
    """Main orchestrator for Video Module process"""

    def __init__(self, config: Config):
        self.config = config
        self.camera = None  # OpenCVBackend or similar
        self.ipc_writer = None  # SharedMemoryFrameWriter
        self.shutdown_event = threading.Event()

    def _initialize_camera(self):
        """Initialize camera from config"""
        # Use existing CameraCapture or create new OpenCVBackend

    def _initialize_ipc_writer(self, width, height):
        """Create shared memory writer with actual dimensions"""

    def _main_loop(self):
        """Capture frames and write to shared memory"""
        for frame in self.camera.capture():
            if self.shutdown_event.is_set():
                break
            self.ipc_writer.write_frame(frame.data, frame.sequence)

    def start(self):
        """Initialize subsystems and start main loop"""

    def stop(self):
        """Graceful shutdown"""
```

**Integration Points**:
- Reuse existing `backend/vision/capture.py` (CameraCapture class)
- Or adapt v2's `backend/video/camera/opencv_backend.py`
- Decision: Reuse existing to minimize changes

**2.2.2 Create Process Entry Point (3-4h)**

**File**: `backend/video/__main__.py`

```python
"""Video Module - Standalone Process Entry Point"""
import sys
from backend.config import config
from backend.video.process import VideoProcess

def main() -> int:
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.json"
    # Note: current config is singleton, may need adjustment

    # Environment overrides (for testing)
    import os
    video_file = os.environ.get("VIDEO_FILE")
    if video_file:
        config.set("vision.camera.device_id", video_file)
        config.set("vision.camera.loop_video", True)

    # Create and start process
    process = VideoProcess(config)
    return process.start()

if __name__ == "__main__":
    sys.exit(main())
```

**2.2.3 Signal Handling (3-4h)**
- SIGTERM handler for graceful shutdown
- SIGINT handler for Ctrl+C
- Cleanup on exit

**2.2.4 Resource Cleanup (3-4h)**
- Stop camera capture
- Close shared memory (writer owns unlink)
- Log final statistics

**2.2.5 Testing (3-4h)**
- Test process startup/shutdown
- Test with video file (loop mode)
- Test with camera device
- Test signal handling
- Test resource cleanup

**Deliverables**:
- ✅ Standalone Video Module process
- ✅ Clean startup/shutdown
- ✅ Signal handling
- ✅ Executable via `python -m backend.video`

---

### Task 2.3: Update Vision Module to Use SHM (12-15h)

**Objective**: Replace current frame access with shared memory reads

#### Subtasks:

**2.3.1 Create VideoConsumer Wrapper (5-6h)**

**File**: `backend/vision/stream/video_consumer.py`

```python
"""VideoConsumer - Simplified wrapper around SharedMemoryFrameReader"""
from typing import Optional
import numpy as np
from backend.video.ipc.shared_memory import SharedMemoryFrameReader
from backend.config import config

class VideoConsumer:
    """Wraps SharedMemoryFrameReader with error handling"""

    def __init__(self):
        self.reader: Optional[SharedMemoryFrameReader] = None
        self.is_running = False

        # Load config
        self._shm_name = config.get("video.shared_memory_name", "billiards_video")
        self._attach_timeout = config.get("video.shared_memory_attach_timeout_sec", 5.0)

    def start(self):
        """Attach to Video Module's shared memory"""
        self.reader = SharedMemoryFrameReader(name=self._shm_name)
        try:
            self.reader.attach(timeout=self._attach_timeout)
            self.is_running = True
        except TimeoutError as e:
            raise VideoModuleNotAvailableError(
                f"Video Module not available. Is it running? "
                f"Start with: python -m backend.video"
            ) from e

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking frame read. Returns None if no new frame."""
        if not self.is_running:
            return None

        frame, metadata = self.reader.read_frame()
        return frame

    def stop(self):
        """Detach from shared memory"""
        if self.reader:
            self.reader.detach()
        self.is_running = False
```

**2.3.2 Integrate into VisionModule (4-5h)**

**File**: `backend/vision/__init__.py`

**Changes**:
```python
# OLD: Use CameraCapture directly
self.camera = CameraCapture(camera_config)

# NEW: Use VideoConsumer
from backend.vision.stream.video_consumer import VideoConsumer
self._video_consumer = VideoConsumer()

# In start():
self._video_consumer.start()  # Attach to SHM

# In _capture_loop():
# OLD:
frame_data = self.camera.get_latest_frame()

# NEW:
frame = self._video_consumer.get_frame()
if frame is None:
    await asyncio.sleep(0.001)
    continue
```

**2.3.3 Update Error Handling (2-3h)**
- Handle VideoModuleNotAvailableError
- Graceful degradation if Video Module crashes
- Logging for debugging

**2.3.4 Testing (1-1h)**
- Test with Video Module running
- Test Video Module not running (timeout)
- Test Video Module crash during operation
- Measure frame latency

**Deliverables**:
- ✅ VisionModule uses shared memory
- ✅ Graceful error handling
- ✅ <5ms frame access latency

---

### Task 2.4: Update API Streaming to Use SHM (8-10h)

**Objective**: Enable multiple concurrent streaming clients

#### Subtasks:

**2.4.1 Add Video Streaming Endpoint (4-5h)**

**File**: `backend/api/routes/stream.py`

**New Implementation**:
```python
from backend.video.ipc.shared_memory import SharedMemoryFrameReader
import cv2
import asyncio

@router.get("/stream/video")
async def stream_video(
    request: Request,
    quality: int = Query(85, ge=1, le=100),
    fps: int = Query(30, ge=1, le=60)
):
    async def generate_frames():
        reader = None
        try:
            # Create independent reader
            shm_name = config.get("video.shared_memory_name", "billiards_video")
            reader = SharedMemoryFrameReader(name=shm_name)
            reader.attach(timeout=5.0)

            frame_delay = 1.0 / fps
            last_frame_number = -1

            while True:
                # Check disconnect
                if await request.is_disconnected():
                    break

                # Read frame
                frame, metadata = reader.read_frame()

                if frame is not None and metadata.frame_number != last_frame_number:
                    # Encode JPEG
                    success, buffer = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )

                    if success:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes() + b"\r\n"
                        )
                        last_frame_number = metadata.frame_number

                await asyncio.sleep(frame_delay)

        finally:
            if reader:
                reader.detach()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

**2.4.2 Remove EnhancedCameraModule Dependency (2-3h)**
- Keep EnhancedCameraModule for now (backward compatibility)
- Add feature flag to switch between old/new
- Default to new (shared memory) implementation

**2.4.3 Testing (2-2h)**
- Test single client streaming
- Test 10+ concurrent clients
- Measure CPU usage per client
- Test client disconnect cleanup

**Deliverables**:
- ✅ API streams from shared memory
- ✅ Support 10+ concurrent clients
- ✅ Clean disconnect handling

---

## Testing Strategy

### Unit Tests
- `test_shared_memory.py` - All shared memory operations
- `test_video_process.py` - VideoProcess initialization/shutdown
- `test_video_consumer.py` - VideoConsumer attach/read/detach

### Integration Tests
- `test_video_to_vision.py` - End-to-end frame flow
- `test_video_to_api.py` - Streaming endpoint
- `test_multiple_readers.py` - Concurrent consumer stress test

### Performance Tests
- Measure frame latency (target <5ms)
- Measure memory usage (target ~20MB constant)
- Measure CPU usage (target <10% for Video Module)
- Stress test with 20 concurrent readers

### Acceptance Criteria
- ✅ Frame latency <5ms (90% reduction from current)
- ✅ Support 10+ streaming clients without degradation
- ✅ Video Module runs as independent process
- ✅ All existing functionality preserved
- ✅ Graceful error handling
- ✅ Clean startup/shutdown

---

## Migration Strategy

### Step 1: Parallel Implementation (Week 1-2)
- Implement alongside existing code
- Feature flag: `USE_SHARED_MEMORY=true/false`
- Default to `false` initially

### Step 2: Testing & Validation (Week 2)
- Run both implementations in parallel
- Compare results, latency, stability
- Fix any issues

### Step 3: Gradual Rollout (Week 3)
- Enable for Vision Module first
- Monitor for 2-3 days
- Enable for API streaming
- Monitor for 2-3 days

### Step 4: Cleanup (Week 4)
- Remove feature flag
- Remove old EnhancedCameraModule (if not needed)
- Update documentation

---

## Rollback Plan

If critical issues arise:

1. **Immediate**: Set `USE_SHARED_MEMORY=false` in config
2. **Short-term**: Revert to previous commit
3. **Investigation**: Analyze logs, reproduce issue
4. **Fix Forward**: Address root cause, re-enable

---

## Dependencies and Prerequisites

### Technical Dependencies
- Python 3.8+ (for `multiprocessing.shared_memory`)
- NumPy (already installed)
- OpenCV (already installed)

### Process Dependencies
- None - can start immediately
- Can run in parallel with Phase 3 (Enhanced Detection)

### Configuration Prerequisites
- Add video module config sections (shown above)

---

## Success Metrics

### Performance Metrics
- Frame latency: <5ms (currently 20-50ms)
- Memory usage: ~20MB constant (currently 55-60MB for 5 consumers)
- CPU usage: <10% for Video Module
- Concurrent clients: 10+ without degradation (currently 5 max)

### Stability Metrics
- Video Module uptime: 99%+
- Reader attach success rate: 99%+
- Frame drop rate: <1%
- Clean shutdown success: 100%

### Quality Metrics
- All unit tests pass
- All integration tests pass
- No regressions in existing functionality
- Code review approval

---

## Documentation Deliverables

1. **README.md** for video module
2. **Architecture diagram** (before/after)
3. **API documentation** for VideoConsumer
4. **Deployment guide** (how to run Video Module)
5. **Troubleshooting guide** (common issues)

---

## Timeline

**Week 1**: Tasks 2.1 + 2.2 (Shared Memory + Video Process)
**Week 2**: Task 2.3 (Vision Module integration)
**Week 3**: Task 2.4 + Testing (API integration + validation)
**Week 4**: Gradual rollout + monitoring

**Total**: 4 weeks (55-70 hours)

---

## Risk Mitigation

### Risk: Video Module crashes frequently
**Mitigation**: Add watchdog monitoring (Phase 4)

### Risk: Shared memory permission issues
**Mitigation**: File-backed fallback already implemented

### Risk: Integration breaks existing functionality
**Mitigation**: Feature flag, parallel implementation, gradual rollout

### Risk: Latency doesn't improve as expected
**Mitigation**: Benchmark at each stage, identify bottlenecks early

---

## Next Steps

1. Review this plan with team
2. Create feature branch: `feature/shared-memory-ipc`
3. Begin Task 2.1 (Extract Shared Memory Module)
4. Set up CI/CD for Video Module process
5. Schedule daily standups for progress tracking
