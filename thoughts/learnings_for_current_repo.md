# Learnings from billiards-trainer-v2 for Current Repository

**Generated:** 2025-10-20
**Purpose:** Identify actionable improvements from v2 architecture and implementation
**Priority:** Organized by ROI (Return on Investment)

---

## Executive Summary

Based on comprehensive analysis of billiards-trainer (original) and billiards-trainer-v2 codebases, this document outlines specific, actionable learnings prioritized by ROI. The v2 repository represents a complete architectural redesign with several production-ready improvements that can be selectively adopted.

**Key Insight:** The current repository has already completed the massive "hardcoded values elimination" effort (500+ values moved to config), which was a major pain point. The v2 architecture offers different patterns that can be adopted incrementally without requiring a complete rewrite.

---

## High-Impact, Low-Effort Changes (Weeks 1-2)

### 1. Simple Configuration System ✅ ALREADY IMPLEMENTED

**Status:** Current repo already uses simplified config (backend/config.py)

**What it is:** Single-file, dot-notation configuration loader (~150 lines)

**Benefits:**
- No complex machinery (no encryption, validation, hot-reload overhead)
- Easy to understand and debug
- Fast initialization
- Sufficient for 95% of use cases

**Current Implementation:** `/Users/jchadwick/code/billiards-trainer/backend/config.py`

**Recommendation:** Keep current simple implementation. The old complex config system (backend/config/ directory with 40+ files) was over-engineered for actual needs.

**Effort:** 0 hours (already done correctly)

---

### 2. Circuit Breaker Pattern for Broadcast Failures

**What it is:** Implement circuit breaker pattern to prevent cascade failures in WebSocket broadcasting

**Why beneficial:**
- Prevents CPU waste from attempting broadcasts when all clients disconnected
- Stops log spam from repeated broadcast failures
- Automatically recovers when clients reconnect
- Protects system resources during client issues

**Current problem:**
- When clients disconnect, integration service continues attempting broadcasts
- Each failed broadcast creates error logs and wastes CPU cycles
- No automatic backoff or recovery mechanism

**Implementation:**
```python
# backend/api/websocket/circuit_breaker.py
class CircuitBreaker:
    """Circuit breaker for broadcast operations.

    States:
    - CLOSED: Normal operation (broadcasts attempted)
    - OPEN: Too many failures (broadcasts blocked)
    - HALF-OPEN: Testing recovery (one attempt allowed)

    After N failures (default: 10), circuit opens for timeout period (default: 30s)
    """

    def __init__(self, failure_threshold: int = 10, timeout_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.consecutive_failures = 0
        self.circuit_open_time: Optional[float] = None
        self.is_open = False

    def can_attempt(self) -> bool:
        """Check if broadcast should be attempted."""
        if not self.is_open:
            return True

        # Check if timeout elapsed (enter half-open state)
        if time.time() - self.circuit_open_time >= self.timeout_seconds:
            logger.info("Circuit breaker entering half-open state")
            return True

        return False

    def record_success(self) -> None:
        """Record successful broadcast - close circuit."""
        if self.is_open:
            logger.info("Circuit breaker closing after successful operation")
        self.consecutive_failures = 0
        self.circuit_open_time = None
        self.is_open = False

    def record_failure(self) -> None:
        """Record failed broadcast - increment counter."""
        self.consecutive_failures += 1

        if not self.is_open and self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            self.circuit_open_time = time.time()
            logger.error(
                f"Circuit breaker OPENED after {self.consecutive_failures} failures. "
                f"Will retry after {self.timeout_seconds}s"
            )
```

**Integration:**
```python
# backend/api/websocket/broadcaster.py
class WebSocketBroadcaster:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("api.websocket.circuit_breaker.threshold", 10),
            timeout_seconds=config.get("api.websocket.circuit_breaker.timeout", 30.0)
        )

    async def broadcast(self, event_type: str, data: dict):
        """Broadcast with circuit breaker protection."""
        if not self.circuit_breaker.can_attempt():
            logger.debug("Circuit breaker OPEN - skipping broadcast")
            return

        try:
            await self._do_broadcast(event_type, data)
            self.circuit_breaker.record_success()
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            self.circuit_breaker.record_failure()
```

**Configuration additions:**
```json
{
  "api": {
    "websocket": {
      "circuit_breaker": {
        "enabled": true,
        "failure_threshold": 10,
        "timeout_seconds": 30.0
      }
    }
  }
}
```

**Files to modify:**
- `backend/api/websocket/broadcaster.py` - Add circuit breaker
- `backend/api/websocket/manager.py` - Integrate with WebSocket manager
- `backend/integration_service.py` - Use circuit breaker for broadcasts
- `config.json` - Add circuit breaker configuration

**Effort:** 8-12 hours
**ROI:** Very High - Prevents resource waste, improves stability
**Dependencies:** None
**Testing:** Unit tests + manual disconnect testing

---

### 3. Broadcast Metrics and Monitoring

**What it is:** Track broadcast success/failure rates and expose via API

**Why beneficial:**
- Visibility into system health
- Early warning for client connectivity issues
- Performance optimization insights
- Debugging aid

**Implementation:**
```python
# backend/api/websocket/metrics.py
@dataclass
class BroadcastMetrics:
    """Metrics for broadcast operations."""
    successful_broadcasts: int = 0
    failed_broadcasts: int = 0
    total_retries: int = 0
    circuit_breaker_opens: int = 0

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.successful_broadcasts + self.failed_broadcasts
        return (self.successful_broadcasts / total * 100) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "successful_broadcasts": self.successful_broadcasts,
            "failed_broadcasts": self.failed_broadcasts,
            "total_retries": self.total_retries,
            "circuit_breaker_opens": self.circuit_breaker_opens,
            "success_rate_percent": round(self.success_rate(), 2)
        }
```

**API endpoint:**
```python
# backend/api/routes/metrics.py
@router.get("/api/metrics/websocket")
async def get_websocket_metrics():
    """Get WebSocket broadcast metrics."""
    return broadcaster.metrics.to_dict()
```

**Files to create/modify:**
- `backend/api/websocket/metrics.py` - New metrics dataclass
- `backend/api/routes/metrics.py` - New metrics endpoint
- `backend/api/websocket/broadcaster.py` - Track metrics
- `backend/api/main.py` - Register metrics router

**Effort:** 6-8 hours
**ROI:** High - Visibility improves operational confidence
**Dependencies:** None
**Testing:** Unit tests + API integration tests

---

### 4. Structured Error Types for Retry Logic

**What it is:** Classify errors into categories to determine retry behavior

**Why beneficial:**
- Smart retry decisions (don't retry validation errors)
- Faster failure detection
- Cleaner error handling
- Better logging

**Implementation:**
```python
# backend/api/websocket/errors.py
from enum import Enum

class BroadcastErrorType(Enum):
    """Types of broadcast errors for retry logic."""
    TRANSIENT = "transient"     # Network errors - RETRY
    VALIDATION = "validation"   # Data errors - DON'T RETRY
    UNKNOWN = "unknown"         # Unknown - retry with caution

def classify_error(error: Exception) -> BroadcastErrorType:
    """Classify error to determine retry behavior."""
    # Network/connection errors - transient
    if isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return BroadcastErrorType.TRANSIENT

    # Data validation errors - don't retry
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return BroadcastErrorType.VALIDATION

    # Unknown - treat as transient but log warning
    return BroadcastErrorType.UNKNOWN
```

**Integration:**
```python
# backend/api/websocket/broadcaster.py
async def broadcast(self, event_type: str, data: dict, max_retries: int = 3):
    """Broadcast with smart retry logic."""
    for attempt in range(max_retries + 1):
        try:
            await self._do_broadcast(event_type, data)
            return
        except Exception as e:
            error_type = classify_error(e)

            # Don't retry validation errors
            if error_type == BroadcastErrorType.VALIDATION:
                logger.error(f"Validation error in broadcast: {e}")
                return

            # Retry transient errors
            if attempt < max_retries:
                logger.warning(f"Broadcast attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Broadcast failed after {max_retries} retries: {e}")
```

**Files to create/modify:**
- `backend/api/websocket/errors.py` - New error classification
- `backend/api/websocket/broadcaster.py` - Use error classification
- `backend/integration_service.py` - Apply to integration broadcasts

**Effort:** 4-6 hours
**ROI:** Medium-High - Faster failures, less wasted retries
**Dependencies:** None
**Testing:** Unit tests with different error types

---

## Medium-Effort Improvements (Weeks 3-5)

### 5. Shared Memory IPC for Frame Sharing

**What it is:** Use shared memory (triple-buffered) for zero-copy frame delivery between processes

**Why beneficial:**
- **Massive latency reduction:** 1-2ms vs 20-50ms+ current serialization
- Zero-copy frame sharing (no pickling/unpickling)
- Enables true multi-process architecture
- Multiple consumers can read same frame simultaneously
- Lock-free reading

**Current problem:**
- Frame sharing between vision and API likely uses slow serialization
- High latency for video streaming
- Cannot efficiently support multiple consumers (web UI, projector, recording)

**Architecture:**
```
Video Module (Writer)                    Vision Module (Reader 1)
       |                                        |
       v                                        v
[Capture Frame]                          [Read from Buffer M]
       |                                        |
       v                                        v
[Write to Buffer N]          <-- Shared Memory --> [Process Frame]
       |
       v                                   API Module (Reader 2)
[Update Header]                                 |
                                                v
                                          [Read from Buffer M]
                                                |
                                                v
                                          [Stream to Client]
```

**Memory layout:**
```
[Header Block - 4KB]
  - magic_number: 0x424954414C4C5344
  - frame_width, frame_height, format
  - current_read_index, current_write_index
  - write_counter, frame_number, timestamp

[Frame Buffer 0 - width * height * bytes_per_pixel]
[Frame Buffer 1 - width * height * bytes_per_pixel]
[Frame Buffer 2 - width * height * bytes_per_pixel]
```

**Implementation:**
```python
# backend/video/ipc/shared_memory_writer.py
from multiprocessing import shared_memory
import numpy as np

class SharedMemoryFrameWriter:
    """Write frames to shared memory with triple buffering."""

    def __init__(self, name: str, width: int, height: int, format: str = "bgr24"):
        self.name = name
        self.width = width
        self.height = height
        self.bytes_per_pixel = 3  # BGR24
        self.frame_size = width * height * self.bytes_per_pixel

        # Create shared memory: header + 3 frame buffers
        total_size = 4096 + (3 * self.frame_size)
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)

        # Initialize header
        self._init_header()

        self.current_buffer = 0
        self.frame_number = 0

    def write_frame(self, frame: np.ndarray, frame_number: Optional[int] = None):
        """Write frame to shared memory."""
        if frame_number is None:
            frame_number = self.frame_number
            self.frame_number += 1

        # Write to current buffer
        buffer_offset = 4096 + (self.current_buffer * self.frame_size)
        frame_bytes = frame.tobytes()
        self.shm.buf[buffer_offset:buffer_offset + self.frame_size] = frame_bytes

        # Update header atomically
        import struct
        import time
        timestamp = time.time()
        header = struct.pack(
            '<QIIIIIQQQQI',  # Format string
            0x424954414C4C5344,  # magic
            1,  # version
            3,  # buffer_count
            self.width,
            self.height,
            2,  # BGR24 format
            self.frame_size,
            frame_number,
            int(timestamp),
            int((timestamp % 1) * 1e9),
            self.current_buffer
        )
        self.shm.buf[0:len(header)] = header

        # Move to next buffer (0 → 1 → 2 → 0)
        self.current_buffer = (self.current_buffer + 1) % 3

# backend/video/ipc/shared_memory_reader.py
class SharedMemoryFrameReader:
    """Read frames from shared memory."""

    def __init__(self, name: str):
        self.name = name
        self.shm = None
        self.last_read_frame = -1

    def attach(self, timeout: float = 5.0):
        """Attach to existing shared memory."""
        import time
        start = time.time()

        while time.time() - start < timeout:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name)
                return
            except FileNotFoundError:
                time.sleep(0.1)

        raise TimeoutError(f"Failed to attach to shared memory '{self.name}'")

    def read_frame(self) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """Read latest frame (non-blocking)."""
        # Read header
        header_data = bytes(self.shm.buf[0:4096])
        frame_number, read_index, width, height, format_code = self._parse_header(header_data)

        # Check if new frame available
        if frame_number <= self.last_read_frame:
            return None, None

        # Read frame from buffer
        frame_size = width * height * 3  # BGR24
        buffer_offset = 4096 + (read_index * frame_size)
        frame_bytes = bytes(self.shm.buf[buffer_offset:buffer_offset + frame_size])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))

        self.last_read_frame = frame_number

        metadata = {
            'frame_number': frame_number,
            'width': width,
            'height': height,
            'format': format_code
        }

        return frame.copy(), metadata
```

**Configuration:**
```json
{
  "video": {
    "shared_memory": {
      "enabled": true,
      "segment_name": "billiards_video",
      "width": 1920,
      "height": 1080,
      "format": "bgr24"
    }
  }
}
```

**Files to create:**
- `backend/video/ipc/__init__.py`
- `backend/video/ipc/shared_memory_writer.py` - Writer implementation
- `backend/video/ipc/shared_memory_reader.py` - Reader implementation
- `backend/video/ipc/frame_formats.py` - Frame format definitions
- `backend/tests/unit/test_shared_memory_ipc.py` - Tests

**Files to modify:**
- `backend/streaming/enhanced_camera_module.py` - Use shared memory writer
- `backend/vision/` - Use shared memory reader
- `backend/api/routes/stream.py` - Use shared memory reader for streaming

**Effort:** 25-35 hours
**ROI:** Very High for video-heavy applications
**Dependencies:** Python 3.8+ (multiprocessing.shared_memory)
**Testing:** Integration tests with mock video + multiple readers

**Performance impact:**
- Current: ~20-50ms frame latency (serialization overhead)
- With shared memory: ~1-2ms frame latency (measured in v2)
- Memory: ~20MB for 1920x1080 BGR24 triple buffer (negligible)

---

### 6. Hybrid YOLO + Classical CV Detection

**What it is:** Run YOLO and OpenCV detectors in parallel, merge results with Weighted Boxes Fusion

**Why beneficial:**
- **Major accuracy improvement:** 70-95% detection vs current ~40% (user-reported)
- Complementary strengths: YOLO learns from data, CV provides fast baseline
- Graceful degradation: CV fallback if YOLO fails
- Better generalization across lighting/table variations

**Current problem (from v2 research):**
- User reports ~40% actual detection rate
- Classical CV struggles with varied conditions
- YOLOv8 Nano standalone failed (too small)

**Architecture:**
```
Input Frame
    │
    ├──> Classical CV (CPU) ──┐
    │    - HSV masks          │
    │    - HoughCircles       │
    │    - 20-30ms            │
    │                         │
    └──> YOLO v11 Medium ─────┤
         - Coral TPU          │
         - TFLite INT8        │
         - 7-10ms             │
                              ▼
                  Weighted Boxes Fusion
                        (WBF)
                              │
                              ▼
                   Merged Detections
                   (Best of both)
```

**Why YOLOv11 Medium (not Nano):**
- User's YOLOv8 Nano standalone failed
- Medium: 25M params vs Nano: 3.2M (8x capacity)
- Medium: 49-52% mAP vs Nano: 37-40% mAP
- Coral TPU can handle Medium at 100-150 FPS (plenty of headroom)
- Better generalization to real-world conditions

**Implementation:**
```python
# backend/vision/detection/hybrid_detector.py
from concurrent.futures import ThreadPoolExecutor
from .opencv_detector import OpenCVDetector
from .yolo_detector import YOLODetector
from .fusion import weighted_boxes_fusion

class HybridDetector:
    """Hybrid detector combining Classical CV and YOLO."""

    def __init__(self, config):
        self.cv_detector = OpenCVDetector(config)
        self.yolo_detector = YOLODetector(config)

        # Fusion configuration
        self.iou_threshold = config.get("vision.detection.hybrid.iou_threshold", 0.5)
        self.cv_weight = config.get("vision.detection.hybrid.cv_weight", 0.6)
        self.yolo_weight = config.get("vision.detection.hybrid.yolo_weight", 0.4)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """Detect balls using hybrid approach."""
        # Run both detectors in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            cv_future = executor.submit(self.cv_detector.detect, frame)
            yolo_future = executor.submit(self.yolo_detector.detect, frame)

            cv_detections = cv_future.result()
            yolo_detections = yolo_future.result()

        # Merge detections using WBF
        merged = weighted_boxes_fusion(
            cv_detections,
            yolo_detections,
            iou_threshold=self.iou_threshold,
            cv_weight=self.cv_weight,
            yolo_weight=self.yolo_weight
        )

        return merged

# backend/vision/detection/fusion.py
def weighted_boxes_fusion(
    detections_cv: List[Tuple[int, int, int, float]],
    detections_yolo: List[Tuple[int, int, int, float]],
    iou_threshold: float = 0.5,
    cv_weight: float = 0.6,
    yolo_weight: float = 0.4
) -> List[Tuple[int, int, int, float]]:
    """Merge detections using Weighted Boxes Fusion.

    Unlike NMS (which discards overlapping detections), WBF:
    - Groups overlapping detections by IoU
    - Averages positions weighted by confidence
    - Combines confidence scores
    - Preserves information from both detectors

    Provides 3-5% mAP improvement over NMS in ensemble systems.
    """
    # Weight confidence scores
    cv_weighted = [(x, y, r, conf * cv_weight) for x, y, r, conf in detections_cv]
    yolo_weighted = [(x, y, r, conf * yolo_weight) for x, y, r, conf in detections_yolo]

    all_detections = cv_weighted + yolo_weighted
    if not all_detections:
        return []

    # Cluster overlapping detections by IoU
    clusters = []
    used = set()

    for i, det_i in enumerate(all_detections):
        if i in used:
            continue

        cluster = [det_i]
        used.add(i)

        for j, det_j in enumerate(all_detections):
            if j in used:
                continue

            iou = calculate_circle_iou(det_i, det_j)
            if iou >= iou_threshold:
                cluster.append(det_j)
                used.add(j)

        clusters.append(cluster)

    # Merge each cluster using weighted averaging
    merged = []
    for cluster in clusters:
        total_conf = sum(conf for _, _, _, conf in cluster)

        merged_x = sum(x * conf for x, _, _, conf in cluster) / total_conf
        merged_y = sum(y * conf for _, y, _, conf in cluster) / total_conf
        merged_r = sum(r * conf for _, _, r, conf in cluster) / total_conf
        merged_conf = total_conf / len(cluster)

        merged.append((int(merged_x), int(merged_y), int(merged_r), merged_conf))

    return merged
```

**Configuration:**
```json
{
  "vision": {
    "detection": {
      "mode": "hybrid",

      "opencv": {
        "enabled": true
      },

      "yolo": {
        "enabled": true,
        "model_path": "backend/vision/models/yolov11m_billiards_edgetpu.tflite",
        "device": "coral",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45
      },

      "hybrid": {
        "enabled": true,
        "fusion_method": "wbf",
        "parallel_execution": true,
        "iou_threshold": 0.5,
        "cv_weight": 0.6,
        "yolo_weight": 0.4,
        "fallback_to_cv_on_failure": true
      }
    }
  }
}
```

**Files to create:**
- `backend/vision/detection/yolo_detector.py` - YOLO wrapper for Coral TPU
- `backend/vision/detection/hybrid_detector.py` - Hybrid ensemble
- `backend/vision/detection/fusion.py` - WBF algorithm
- `backend/vision/tests/test_yolo_detector.py`
- `backend/vision/tests/test_hybrid_detector.py`
- `backend/vision/tests/test_fusion.py`

**Files to modify:**
- `backend/vision/__init__.py` - Add hybrid detector option
- `backend/vision/detection/detector_adapter.py` - Route to hybrid detector

**Training requirements:**
- Dataset: 500-700 annotated images from YOUR actual setup
- Annotation: ~50-70 hours (use CVAT or Label Studio)
- Training: 8-16 hours on GPU (100 epochs)
- Export: ONNX → TFLite → Coral compilation

**Effort:** 115-165 hours total (can parallelize)
- Phase 1: YOLO Integration (25-35h)
- Phase 2: Dataset & Training (83-136h, can overlap)
- Phase 3: Hybrid Architecture (20-30h)
- Phase 4: Testing & Tuning (20-30h)

**ROI:** Extremely High IF detection accuracy is critical issue
**Dependencies:**
- Coral Edge TPU hardware (~$75)
- pycoral library
- TensorFlow Lite
- Training dataset from real usage

**Expected improvement:**
- Current: ~40% detection rate (user-reported)
- Target: 70-95% detection rate
- Precision: 85-95%
- Recall: 85-95%

**Testing:**
- Benchmark: hybrid vs CV-only vs YOLO-only
- Real-world validation with user's actual setup
- Parameter tuning (weights, thresholds)

**Note:** This is HIGH effort but provides TRANSFORMATIONAL improvement if detection is currently poor. Should be prioritized if vision accuracy is blocking production use.

---

### 7. Pure Event Bus Architecture (Optional)

**What it is:** Simplify core to be "dumb plumbing" - pure event router without state management

**Why beneficial:**
- Simpler debugging (clear data flow)
- Easier testing (no hidden state)
- Better separation of concerns
- Modules become more independent

**Current pattern:**
- Core has game state management
- Integration service bridges vision → core → broadcast
- Some state duplication/confusion

**V2 pattern:**
```
Vision Module                    API Module
     |                               |
     ├─ DetectionUpdate event        ├─ Subscribes to events
     │                               │
     └─> Core EventBus (router) ────┤
         - No inspection             └─> WebSocket broadcast
         - No state                       - Clients interpret
         - Fire-and-forget                - Clients build state
```

**Key principle:** "Intelligence at the edges"
- Core: Dumb router (just forwards events)
- Vision: Emits raw detections
- Physics: Emits predictions
- API: Forwards to clients
- Clients: Build game state from events

**Recommendation:** Consider for v3 or major refactor. Current architecture is working; this is a cleaner pattern but requires significant rework.

**Effort:** 40-60 hours (architectural change)
**ROI:** Medium (cleaner but not urgent)
**Dependencies:** Requires rethinking state management
**Testing:** Full integration test suite rewrite

---

## High-Effort Architectural Changes (Months 2-3)

### 8. Separate Video Module Process

**What it is:** Run video capture as separate system process (not Python subprocess)

**Why beneficial:**
- Exclusive camera access (no conflicts with vision)
- Can restart vision without losing video
- Better resource isolation
- Enables systemd/Docker management
- Multiple consumers at different resolutions

**Architecture:**
```
┌─────────────────────────────┐
│  Video Module (systemd)     │
│  - Exclusive camera access  │
│  - Shared memory writer     │
│  - Health monitoring        │
└──────────────┬──────────────┘
               │ Shared Memory
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌──────────┐    ┌──────────┐
│  Vision  │    │   API    │
│  Module  │    │  Module  │
│ (Reader) │    │ (Reader) │
└──────────┘    └──────────┘
```

**Current pattern:**
- Camera likely managed by vision or streaming module
- Tight coupling
- Cannot restart vision without stopping video

**Implementation:**
```python
# backend/video/__main__.py
"""Video Module entry point - runs as separate process."""
import sys
from video.camera import CameraManager
from video.ipc import SharedMemoryFrameWriter
from config import Config

def main():
    config = Config()

    # Initialize camera
    camera = CameraManager(config)
    camera.start()

    # Initialize shared memory writer
    writer = SharedMemoryFrameWriter(
        name=config.get("video.shared_memory.segment_name", "billiards_video"),
        width=config.get("video.camera.width", 1920),
        height=config.get("video.camera.height", 1080)
    )

    try:
        while True:
            frame = camera.capture()
            if frame is not None:
                writer.write_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        writer.cleanup()
        camera.stop()

if __name__ == "__main__":
    main()
```

**Systemd service:**
```ini
# /etc/systemd/system/billiards-video.service
[Unit]
Description=Billiards Trainer Video Module
After=network.target

[Service]
Type=simple
User=billiards
WorkingDirectory=/opt/billiards-trainer
ExecStart=/opt/billiards-trainer/venv/bin/python -m backend.video
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Files to create:**
- `backend/video/__main__.py` - Entry point
- `backend/video/camera/` - Camera management
- `backend/video/health.py` - Health monitoring
- `backend/video/watchdog.py` - Process watchdog
- `scripts/install-video-service.sh` - Systemd installation

**Files to modify:**
- `backend/vision/` - Use shared memory reader instead of direct camera
- `backend/api/routes/stream.py` - Use shared memory reader
- `backend/streaming/enhanced_camera_module.py` - Remove or refactor

**Effort:** 35-50 hours
**ROI:** High for production deployments, lower for development
**Dependencies:** Shared memory IPC (item #5)
**Testing:** Multi-process integration tests

**Benefits over current:**
- Video survives vision crashes
- Can have multiple vision consumers (A/B testing)
- Better resource management
- Production-grade architecture

---

### 9. Type-Safe Event Definitions

**What it is:** Define all events as Pydantic models for type safety and validation

**Why beneficial:**
- Compile-time type checking
- Runtime validation
- Self-documenting API
- IDE autocomplete
- Prevents event schema drift

**Implementation:**
```python
# backend/core/events.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class EventType(str, Enum):
    """All possible event types."""
    DETECTION_UPDATE = "detection.update"
    BALL_DETECTED = "ball.detected"
    CUE_DETECTED = "cue.detected"
    TRAJECTORY_PREDICTED = "trajectory.predicted"
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"

class BallDetection(BaseModel):
    """Ball detection data."""
    id: int = Field(..., description="Persistent ball ID")
    x: float = Field(..., ge=0, description="X coordinate in pixels")
    y: float = Field(..., ge=0, description="Y coordinate in pixels")
    radius: float = Field(..., gt=0, description="Radius in pixels")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    color: Optional[str] = Field(None, description="Ball color if known")
    is_moving: bool = Field(False, description="Whether ball is in motion")

class DetectionUpdateEvent(BaseModel):
    """Detection update event."""
    event_type: EventType = EventType.DETECTION_UPDATE
    timestamp: float = Field(..., description="Unix timestamp")
    frame_number: int = Field(..., ge=0, description="Frame sequence number")
    balls: List[BallDetection] = Field(default_factory=list)
    cue: Optional[CueDetection] = None

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "detection.update",
                "timestamp": 1729435200.123,
                "frame_number": 1234,
                "balls": [
                    {"id": 1, "x": 100, "y": 200, "radius": 15, "confidence": 0.95, "is_moving": False}
                ]
            }
        }
```

**Usage:**
```python
# Publishing events
event = DetectionUpdateEvent(
    timestamp=time.time(),
    frame_number=1234,
    balls=[
        BallDetection(id=1, x=100, y=200, radius=15, confidence=0.95)
    ]
)
event_bus.publish(event)

# Subscribing to events
def handle_detection(event: DetectionUpdateEvent):
    for ball in event.balls:
        print(f"Ball {ball.id} at ({ball.x}, {ball.y})")

event_bus.subscribe(EventType.DETECTION_UPDATE, handle_detection)
```

**Benefits:**
```python
# Type checking catches errors at development time
event = DetectionUpdateEvent(
    timestamp="invalid",  # ❌ Type checker error: Expected float
    balls=[
        BallDetection(confidence=1.5)  # ❌ Validation error: Must be ≤ 1
    ]
)

# IDE autocomplete
event.balls[0].  # ← IDE shows: id, x, y, radius, confidence, color, is_moving

# Runtime validation
try:
    event = DetectionUpdateEvent(**raw_data)
except ValidationError as e:
    logger.error(f"Invalid event: {e}")
```

**Files to create:**
- `backend/core/events.py` - Event definitions
- `backend/core/event_bus.py` - Type-safe event bus
- `backend/tests/unit/test_events.py` - Event validation tests

**Files to modify:**
- `backend/vision/` - Emit typed events
- `backend/core/physics/` - Emit typed events
- `backend/api/websocket/` - Subscribe with type hints
- `backend/integration_service.py` - Use typed events

**Effort:** 20-30 hours
**ROI:** High for long-term maintainability
**Dependencies:** Pydantic (already in requirements)
**Testing:** Unit tests for each event type + validation tests

---

## Documentation & Process Improvements

### 10. Comprehensive Module Specifications

**What v2 has:** Detailed SPECS.md for each module with:
- Module purpose and boundaries
- Functional requirements (numbered FR-XXX-000)
- Data models
- API contracts
- Testing requirements
- Non-requirements (what module does NOT do)

**Current state:** Some docs exist but inconsistent

**Recommendation:** Create/update SPECS.md for each module
- `backend/vision/SPECS.md`
- `backend/core/SPECS.md`
- `backend/api/SPECS.md`
- `backend/streaming/SPECS.md`

**Effort:** 15-20 hours
**ROI:** Medium-High for team collaboration
**Benefits:**
- Clear module boundaries
- Prevents scope creep
- Easier onboarding
- Refactoring guide

---

### 11. Separation of Test Types

**What v2 has:** Clear separation:
- `tests/unit/` - Fast, isolated, no I/O
- `tests/integration/` - Module interactions
- `tests/system/` - End-to-end tests

**Current state:** Tests mixed together

**Recommendation:**
```
backend/tests/
├── unit/              # Fast, isolated (run on every commit)
│   ├── test_config.py
│   ├── test_geometry.py
│   └── ...
├── integration/       # Module interactions (run before merge)
│   ├── test_vision_api.py
│   ├── test_websocket_broadcast.py
│   └── ...
└── system/           # End-to-end (run before release)
    ├── test_full_workflow.py
    └── ...
```

**pytest configuration:**
```ini
# pytest.ini
[pytest]
markers =
    unit: Fast unit tests (no I/O)
    integration: Integration tests (module interactions)
    system: System tests (end-to-end)
    slow: Slow tests (skip in quick runs)

# Run only unit tests
python -m pytest -m unit

# Run integration + unit
python -m pytest -m "unit or integration"
```

**Effort:** 10-15 hours (reorganization + markers)
**ROI:** Medium - Faster CI, clearer test purposes

---

## Summary Table: ROI by Effort

| # | Learning | Effort | ROI | Priority | Dependencies |
|---|----------|--------|-----|----------|--------------|
| 1 | Simple Config | 0h | ✅ Done | - | None |
| 2 | Circuit Breaker | 8-12h | Very High | **1** | None |
| 3 | Broadcast Metrics | 6-8h | High | **2** | None |
| 4 | Error Classification | 4-6h | Medium-High | **3** | None |
| 5 | Shared Memory IPC | 25-35h | Very High | **4** | Python 3.8+ |
| 6 | Hybrid YOLO+CV | 115-165h | Extreme | **5** | Coral TPU, Dataset |
| 7 | Pure Event Bus | 40-60h | Medium | Later | Major refactor |
| 8 | Separate Video | 35-50h | High | **6** | Item #5 |
| 9 | Type-Safe Events | 20-30h | High | **7** | Pydantic |
| 10 | Module Specs | 15-20h | Medium-High | Ongoing | None |
| 11 | Test Separation | 10-15h | Medium | Ongoing | None |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Weeks 1-2)
1. Circuit Breaker Pattern (8-12h)
2. Broadcast Metrics (6-8h)
3. Error Classification (4-6h)

**Total:** ~18-26 hours
**Impact:** Immediate stability and observability improvements

### Phase 2: Performance (Weeks 3-4)
4. Shared Memory IPC (25-35h)
5. Type-Safe Events (20-30h)

**Total:** ~45-65 hours
**Impact:** Major latency reduction, better type safety

### Phase 3: Detection Quality (Weeks 5-8)
6. Hybrid YOLO+CV Detection (115-165h)
   - Can parallelize dataset creation with implementation

**Total:** ~115-165 hours
**Impact:** Transformational detection accuracy (IF needed)

### Phase 4: Production Architecture (Weeks 9-10)
7. Separate Video Module (35-50h)

**Total:** ~35-50 hours
**Impact:** Production-grade reliability

### Ongoing
- Module Specifications (15-20h total, incrementally)
- Test Reorganization (10-15h)

---

## Critical Decision Points

### 1. Is Detection Accuracy a Problem?

**If YES (user reports ~40% detection):**
- Prioritize Hybrid YOLO+CV (item #6) immediately after Phase 1
- This is HIGH effort but TRANSFORMATIONAL if detection is blocking production

**If NO (current detection acceptable):**
- Skip item #6 or defer to later
- Focus on stability and performance improvements

### 2. Do You Have Coral TPU Hardware?

**If YES:**
- Hybrid YOLO approach is feasible
- Shared memory IPC enables better architecture

**If NO:**
- Budget $75 for Coral Edge TPU USB
- Or consider CPU-based YOLO (slower, may not hit real-time)

### 3. Is Video Latency Critical?

**If YES (real-time augmented projection):**
- Prioritize Shared Memory IPC (item #5)
- This enables 1-2ms frame delivery vs current 20-50ms+

**If NO (web UI only):**
- Current architecture may be sufficient
- Shared memory still valuable but less urgent

---

## Files Changed Summary

### High Priority (Phase 1-2)
```
backend/api/websocket/
├── circuit_breaker.py        [NEW]
├── errors.py                 [NEW]
├── metrics.py                [NEW]
├── broadcaster.py            [MODIFY]
└── manager.py                [MODIFY]

backend/api/routes/
└── metrics.py                [NEW]

backend/video/ipc/
├── __init__.py               [NEW]
├── shared_memory_writer.py   [NEW]
├── shared_memory_reader.py   [NEW]
└── frame_formats.py          [NEW]

backend/core/
└── events.py                 [NEW]

backend/tests/unit/
├── test_circuit_breaker.py   [NEW]
├── test_shared_memory_ipc.py [NEW]
└── test_events.py            [NEW]

config.json                    [MODIFY - add new settings]
```

### Medium Priority (Phase 3)
```
backend/vision/detection/
├── yolo_detector.py          [NEW]
├── hybrid_detector.py        [NEW]
├── fusion.py                 [NEW]
└── detector_adapter.py       [MODIFY]

backend/vision/models/
└── yolov11m_billiards_edgetpu.tflite [NEW - trained model]

backend/vision/tests/
├── test_yolo_detector.py     [NEW]
├── test_hybrid_detector.py   [NEW]
└── test_fusion.py            [NEW]
```

---

## Configuration Additions

```json
{
  "api": {
    "websocket": {
      "circuit_breaker": {
        "enabled": true,
        "failure_threshold": 10,
        "timeout_seconds": 30.0
      },
      "metrics": {
        "enabled": true
      }
    }
  },

  "video": {
    "shared_memory": {
      "enabled": true,
      "segment_name": "billiards_video",
      "width": 1920,
      "height": 1080,
      "format": "bgr24"
    }
  },

  "vision": {
    "detection": {
      "mode": "hybrid",

      "opencv": {
        "enabled": true
      },

      "yolo": {
        "enabled": true,
        "model_path": "backend/vision/models/yolov11m_billiards_edgetpu.tflite",
        "device": "coral",
        "confidence_threshold": 0.25
      },

      "hybrid": {
        "enabled": true,
        "fusion_method": "wbf",
        "iou_threshold": 0.5,
        "cv_weight": 0.6,
        "yolo_weight": 0.4
      }
    }
  }
}
```

---

## Conclusion

The v2 repository provides several valuable patterns that can be selectively adopted:

**Immediate (Weeks 1-2):** Circuit breaker, metrics, error classification
- Low effort, high impact
- Improves stability and observability
- No architectural changes required

**Medium-term (Weeks 3-8):** Shared memory IPC, hybrid detection
- Higher effort but significant improvements
- Shared memory: ~10-20x latency reduction
- Hybrid detection: ~2-3x accuracy improvement (if needed)

**Long-term (Months 2-3):** Separate video process, pure event bus
- Architectural improvements
- Better for production but not urgent
- Consider for v3 or major refactor

**Key insight:** The current repo already made the right choice with simple configuration. Focus on stability and performance improvements rather than complex abstractions.

**Next step:** Review this document, decide on detection accuracy priority, then start with Phase 1 (circuit breaker + metrics) for quick wins.
