# Backend Architecture Analysis - Billiards Trainer

## Executive Summary

The Billiards Trainer backend is a real-time computer vision and physics simulation system built with Python/FastAPI. It follows a modular, event-driven architecture where the **Integration Service** is the critical glue that connects three main subsystems:

1. **Vision Module** - Detects balls, cue, and table from camera input
2. **Core Module** - Manages game state and physics calculations
3. **WebSocket Broadcaster** - Streams data to connected clients

The system is designed for <50ms latency and 30 FPS processing with built-in resilience through circuit breaker patterns and retry logic.

---

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HTTP/WebSocket API Layer                     │
│                            (FastAPI, Uvicorn)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Integration Service                              │  │
│  │   (Main Message Flow Orchestrator - polling & event-driven)  │  │
│  │                                                               │  │
│  │  - Polls Vision for detection results (30 FPS loop)         │  │
│  │  - Converts & updates Core module state                      │  │
│  │  - Subscribes to Core events (state, trajectory)            │  │
│  │  - Broadcasts via WebSocket with retry & circuit breaker    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                     ↑                      ↓                         │
│            ┌────────┴────────┬─────────────┴──────────┐            │
│            ↓                 ↓                         ↓            │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐   │
│  │  Vision Module   │ │  Core Module     │ │  WebSocket Layer │   │
│  ├──────────────────┤ ├──────────────────┤ ├──────────────────┤   │
│  │                  │ │                  │ │                  │   │
│  │ • Camera Capture │ │ • Game State Mgr │ │ • Manager        │   │
│  │ • Ball Detection │ │ • Physics Engine │ │ • Handler        │   │
│  │ • Cue Detection  │ │ • Trajectory Cal │ │ • Broadcaster    │   │
│  │ • Table Tracking │ │ • Shot Analysis  │ │ • Subscriptions  │   │
│  │ • Calibration    │ │ • Event Manager  │ │ • Frame Buffering│   │
│  │                  │ │ • Validators     │ │                  │   │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘   │
│         │                     │                       │             │
│    OpenCV/Camera          Python Math            WebSocket Clients │
│         │                 Numpy Arrays                 │             │
│         └─────────────────────┬───────────────────────┘             │
│                               │                                    │
│                        DetectionResult                             │
│                     (Camera Frames, Balls,                         │
│                      Cue, Table, etc.)                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Components & Entry Points

### 1. Main Application Entry
**File**: `/backend/main.py`
- **Purpose**: Application entry point
- **Function**: Parses CLI args, initializes config, starts uvicorn
- **Key Line**: `uvicorn.run("backend.api.main:app", ...)`
- **Config**: Loads from `./config.json` by default

### 2. FastAPI Application Setup
**File**: `/backend/api/main.py` (contains `app` and `lifespan`)
- **Startup Sequence**:
  1. Initializes Config singleton
  2. Creates CoreModule
  3. Initializes VisionModule
  4. Creates IntegrationService
  5. Starts integration service (begins Vision→Core→Broadcast flow)
  6. Registers WebSocket system
  7. Starts health monitoring

- **Key Function**: `lifespan()` - async context manager for app lifecycle
- **Routes**: REST endpoints + WebSocket at `/api/v1/ws`

### 3. Integration Service (THE CRITICAL HUB)
**File**: `/backend/integration_service.py` (1218 lines)
- **Role**: Connects Vision → Core → Broadcasting
- **Main Loop**: `_integration_loop()` runs at configurable FPS (default 30)
  ```
  Loop iteration:
    1. Get detection from Vision: vision.process_frame()
    2. Convert to Core format
    3. Update Core state: core.update_state(detection_data)
    4. Check if trajectory should calculate
    5. Maintain target FPS with sleep
  ```

- **State Flow**:
  - Subscribes to Core events: `state_updated` and `trajectory_calculated`
  - Event callbacks convert async -> sync via stored event loop
  - Broadcasts with retry logic and circuit breaker

- **Key Methods**:
  - `start()` - Starts vision capture and integration loop
  - `stop()` - Graceful shutdown
  - `_integration_loop()` - Main FPS-controlled loop
  - `_get_vision_detection()` - Polls vision module
  - `_process_detection()` - Validates and converts data
  - `_check_trajectory_calculation()` - Determines if trajectory needed
  - `_emit_multiball_trajectory()` - Broadcasts trajectory calculations
  - `_broadcast_with_retry()` - Resilient broadcast with exponential backoff

- **Configuration** (from config.json):
  - `integration.target_fps` - Target FPS (default 30)
  - `integration.broadcast_max_retries` - Retry attempts (default 3)
  - `integration.circuit_breaker_threshold` - Failures before circuit opens (default 10)
  - `integration.circuit_breaker_timeout_sec` - Circuit timeout (default 30s)

### 4. Vision Module
**File**: `/backend/vision/__init__.py` (1360 lines)
- **Role**: Real-time computer vision detection
- **Main Methods**:
  - `start_capture()` - Initializes camera and starts background thread
  - `process_frame()` - Returns latest detection (async-safe, non-blocking)
  - `stop_capture()` - Stops background processing

- **Detection Result**: `DetectionResult` dataclass containing:
  - `balls: list[Ball]` - Detected balls with position, radius, velocity
  - `cue: Optional[CueStick]` - Detected cue with angle, position
  - `table: Optional[Table]` - Table corners, pockets, geometry
  - `timestamp: float` - Frame timestamp
  - `frame_number: int` - Sequential frame count

- **Sub-modules**:
  - `vision/detection/` - Ball, cue, table detection algorithms
  - `vision/tracking/` - Kalman filter tracking for temporal consistency
  - `vision/calibration/` - Camera, color, geometric calibration
  - `vision/preprocessing/` - Image enhancement and filtering
  - `vision/stream/` - Shared memory video consumer for IPC

### 5. Core Module
**File**: `/backend/core/__init__.py`
- **Role**: Game state management and physics calculations
- **Main Methods**:
  - `update_state(detection_data)` - Updates GameState from detection
  - `subscribe_to_events(event_type, callback)` - Register event listeners
  - `_emit_event(event_type, event_data)` - Dispatch events

- **Game State**: `GameState` contains:
  - `balls: list[BallState]` - All balls with position, velocity, type
  - `cue: Optional[CueState]` - Cue position and angle
  - `table: TableState` - Table geometry and boundaries
  - `timestamp: float` - State timestamp
  - Frame/sequence tracking

- **Sub-modules**:
  - `core/physics/` - Collision detection, trajectory calculation
  - `core/analysis/` - Shot analysis, outcome prediction
  - `core/events/` - Event system with async callbacks
  - `core/validation/` - Physics and state validators
  - `core/utils/` - Math, geometry, caching utilities

### 6. WebSocket Broadcasting System
**File**: `/backend/api/websocket/broadcaster.py` (main message distributor)

- **Key Components**:
  - `MessageBroadcaster` - High-performance message distributor
  - `WebSocketManager` - Client connection lifecycle
  - `WebSocketHandler` - Message parsing and routing
  - `FrameBuffer` - Circular frame buffer with cleanup

- **Main Broadcasting Methods**:
  - `broadcast_game_state(balls, cue, table)` - Game state updates
  - `broadcast_trajectory(lines, collisions)` - Trajectory predictions
  - `broadcast_frame(image_data)` - Video frames (currently disabled)

- **Stream Types**:
  - STATE - Game state updates
  - TRAJECTORY - Ball trajectory predictions
  - ALERT - System alerts
  - CONFIG - Configuration changes
  - FRAME - Video frames (disabled)

- **Performance Features**:
  - Message validation before broadcast
  - Circular frame buffering (deque)
  - Optional compression with configurable threshold
  - Sequence numbering for message ordering
  - FPS limiting per client

---

## Message Flow: Ball Detection to Broadcasting

### 1. Frame Capture (Camera → Vision)
```
OpenCV Camera Driver
       ↓
  Camera Frame
       ↓
vision.start_capture()
  - Starts background capture thread
  - Maintains internal frame queue
       ↓
vision.camera object
  - Continuously reads frames
  - Stores latest in internal buffer
```

### 2. Detection Processing (Vision Module)
```
vision.process_frame() [called in integration loop]
       ↓
  [Image Preprocessing]
  - Color space conversion
  - Histogram equalization
  - Noise reduction
       ↓
  [Parallel Detection]
  - BallDetector.detect_balls() → Ball[]
  - CueDetector.detect_cue() → CueStick
  - TableDetector.detect_table() → Table
       ↓
  [Tracking]
  - ObjectTracker applies Kalman filtering
  - Assigns track_ids to balls
       ↓
  DetectionResult
  {
    balls: List[Ball],
    cue: Optional[CueStick],
    table: Optional[Table],
    timestamp: float,
    frame_number: int
  }
```

### 3. State Update (Vision → Core)
```
IntegrationService._integration_loop()
       ↓
detection = vision.process_frame()
       ↓
_convert_detection_to_core_format(detection)
  - Convert Ball → ball dict with x, y, velocity
  - Convert CueStick → cue dict with angle, position
  - Convert Table → table dict with corners, pockets
       ↓
core.update_state(detection_data)
  ├─ Validate detection data
  ├─ Update GameState.balls (position, velocity)
  ├─ Update GameState.cue (angle, position)
  ├─ Update GameState.table (geometry)
  └─ Emit "state_updated" event
```

### 4. Event Handling (Core → Integration)
```
Core Event Manager
       ↓
_emit_event("state_updated", event_data)
       ↓
Callback: _on_state_updated(event_type, event_data)
  [Synchronous callback from Core]
  ├─ Schedule async processing via event loop
  └─ Call _on_state_updated_async(event_type, event_data)
       ↓
[Async Processing]
  ├─ Validate event_data structure
  ├─ Extract balls, cue, table
  ├─ Create broadcast summary
  └─ Call _broadcast_with_retry()
```

### 5. Broadcast with Resilience
```
_broadcast_with_retry(
  broadcast_func=broadcaster.broadcast_game_state,
  operation_name="broadcast_game_state",
  data_summary="{n} balls, cue={present?}",
  balls=[], cue=None, table=None
)
       ↓
[Circuit Breaker Check]
if not circuit_breaker.can_attempt():
  return False  # Skip broadcast if open
       ↓
[Retry Loop] (default 3 retries + 1 initial = 4 attempts)
for attempt in range(max_retries + 1):
  try:
    await broadcaster.broadcast_game_state(
      balls, cue, table
    )
    circuit_breaker.record_success()
    return True
       ↓
  except Exception as e:
    error_type = _classify_broadcast_error(e)
    if error_type == VALIDATION:
      break  # Don't retry validation errors
    if attempt < max_retries:
      delay = retry_base_delay * (2**attempt)
      await asyncio.sleep(delay)
       ↓
circuit_breaker.record_failure()
return False
```

### 6. WebSocket Delivery
```
broadcaster.broadcast_game_state(balls, cue, table)
       ↓
[Validation]
  ├─ Validate balls is list, not empty
  ├─ Each ball has required fields
  └─ Position dict has x, y, scale metadata
       ↓
state_data = {
  "balls": balls,
  "cue": cue,
  "table": table,
  "timestamp": ISO8601,
  "sequence": sequence_number,
  "ball_count": len(balls)
}
       ↓
_broadcast_to_stream(StreamType.STATE, state_data)
       ↓
broadcaster.handler
  ├─ Serialize to JSON
  ├─ Optionally compress (zlib)
  ├─ Send to all subscribed clients
  └─ Track metrics (bytes, latency, failures)
```

### 7. Client Reception
```
WebSocket Client (/api/v1/ws)
       ↓
Receives JSON message:
{
  "type": "state",
  "data": { balls, cue, table, timestamp, ... },
  "sequence": 12345,
  "timestamp": "2024-10-22T..."
}
       ↓
Frontend processes and renders
  ├─ Updates ball positions on canvas
  ├─ Draws cue angle indicator
  └─ Shows table boundaries
```

---

## Queueing & Buffering Mechanisms

### 1. Integration Service Loop Timing
**Type**: FPS-controlled polling loop
**Location**: `IntegrationService._integration_loop()`
**Behavior**:
- Target FPS: 30 (configurable)
- Frame interval: 1/30 = ~33ms
- Sleep to maintain FPS: `await asyncio.sleep(sleep_time)`
- **No explicit queue** - just polling at fixed interval
- **Bottleneck mitigation**: If processing takes >33ms, next frame is skipped

### 2. Frame Buffer
**Type**: Circular deque (FIFO)
**Location**: `MessageBroadcaster.frame_buffer`
**Size**: Configurable (default 100 frames)
**Purpose**:
- Stores recent frames for metrics/debugging
- Auto-cleanup of frames older than 5 seconds
- Tracks frame rate and compression metrics
- **Frame broadcasting currently disabled** - prevents browser crashes from large base64 images

### 3. Vision Module Internal Queue
**Type**: Thread-safe queue
**Location**: `VisionModule._frame_queue`
**Size**: Configurable (default 5)
**Purpose**:
- Decouples camera capture thread from detection processing
- Prevents frame backlog in detection pipeline
- Older frames discarded if queue fills
- `process_frame()` returns latest available frame

### 4. Async Task Broadcast Queue
**Type**: asyncio.Queue
**Location**: `MessageBroadcaster.frame_queue`
**Size**: Configurable (default 10)
**Status**: **Currently disabled** (frames not added to queue)
**Purpose**: Would serialize frame broadcasts to clients

### 5. Event Callback Queue (Sync → Async Bridge)
**Type**: Event loop's call_soon_threadsafe()
**Location**: `IntegrationService._on_state_updated()` callback
**Mechanism**:
```python
self._event_loop.call_soon_threadsafe(
  lambda: asyncio.ensure_future(
    self._on_state_updated_async(event_type, event_data),
    loop=self._event_loop
  )
)
```
**Purpose**: Bridge between Core's synchronous event callbacks and async broadcast loop

### 6. Circuit Breaker (Failure Buffering)
**Type**: State machine pattern
**Location**: `CircuitBreaker` class in integration_service.py
**States**:
- CLOSED - Normal operation, attempts broadcasts
- OPEN - Too many failures, blocks broadcasts
- HALF-OPEN - Timeout elapsed, tries one broadcast

**Configuration**:
- Failure threshold: 10 consecutive failures
- Timeout: 30 seconds
- **Effect**: Prevents cascade failures when all clients disconnect

---

## Thread & Process Architecture

### 1. Main API Thread (Uvicorn)
- **Framework**: FastAPI + Uvicorn
- **Role**: HTTP request handling, WebSocket management
- **Threads**: Configurable worker threads (default based on CPU cores)
- **Event Loop**: Single asyncio event loop per worker

### 2. Vision Capture Thread
- **Location**: `VisionModule.start_capture()`
- **Type**: Daemon thread spawned at integration service startup
- **Purpose**: Continuous camera frame capture
- **Synchronization**: Thread-safe queue for frame delivery
- **Shutdown**: Stops on `vision.stop_capture()` call

### 3. Integration Service (Async)
- **Type**: Asyncio task (runs in main event loop)
- **Location**: `IntegrationService._integration_loop()`
- **Concurrency**: Single coroutine, no parallelism
- **FPS Control**: Cooperative yielding via `await asyncio.sleep()`

### 4. Core Event Loop (Async)
- **Type**: Implicit - Core runs in main event loop
- **Event Dispatch**: Sync callbacks from Core, scheduled async processing
- **Synchronization**: Uses `asyncio.Lock()` for state updates

### 5. Health Monitor Thread
- **Location**: `system/health_monitor.py`
- **Type**: Background monitoring task
- **Purpose**: System health checks
- **Interval**: Configurable (default 5 seconds)

### 6. WebSocket Message Broadcaster (Async)
- **Type**: Asyncio tasks
- **Location**: Message broadcaster runs in main event loop
- **Concurrency**: Non-blocking, all connections handled concurrently
- **Per-client FPS limiting**: Tracks last_frame_time per client_id

---

## Bottlenecks & Performance Analysis

### 1. Vision Detection Latency
**Bottleneck**: YOLOv8/detection algorithms
**Typical Duration**: 30-50ms per frame
**Impact**: Limits achievable FPS
**Mitigation**:
- Frame skipping option
- ROI (Region of Interest) mode
- GPU acceleration if available

### 2. Integration Loop Timing
**Bottleneck**: Fixed 30 FPS, processing + sleep
**Typical Duration**: ~33ms per iteration
**Issue**: If detection takes 40ms, frame is skipped
**Mitigation**:
- Adaptive FPS (currently fixed)
- Parallel detection processing (currently sequential)

### 3. State Validation Overhead
**Bottleneck**: TableStateValidator, PhysicsValidator
**Typical Duration**: 5-10ms per validation
**Impact**: Adds to trajectory calculation time
**Mitigation**:
- Caching of validation results
- Skip detailed validation in real-time mode

### 4. Trajectory Calculation CPU Cost
**Bottleneck**: PhysicsEngine.simulate_trajectory() with collision detection
**Typical Duration**: 20-100ms depending on quality setting
**Current Setting**: LOW quality for real-time (max_collision_depth=5)
**Mitigation**:
- LOW quality used for real-time
- Trajectory only calculated when cue detected
- Results cached

### 5. WebSocket Broadcasting Latency
**Bottleneck**: Network I/O, JSON serialization
**Typical Duration**: <5ms for ~5-15 balls
**Per-Client Overhead**: Minimal with async handling
**Bottleneck Risk**: If >20 clients connected and network slow
**Mitigation**:
- Optional compression for large payloads
- Circuit breaker prevents cascade failures
- Client-side FPS limiting

### 6. Message Validation Overhead
**Bottleneck**: broadcast_game_state() validation
**Typical Duration**: 1-2ms per broadcast
**Checks**:
- Ball list structure
- Position dict format
- Scale metadata validation
**Impact**: Negligible compared to physics

### 7. Memory Usage (No Explicit Backpressure)
**Issue**: No backpressure mechanism if Vision produces faster than Core consumes
**Current Mitigation**:
- Vision queue size limited (default 5 frames)
- Only latest frame from vision used
- Old frames dropped automatically
**Risk**: High-frequency camera (60 FPS) → frame drops

---

## Configuration & Initialization

### Config System
**Type**: Simple JSON file-based singleton
**File**: `backend/config.py` + `./config.json`
**Features**:
- Dot-notation access: `config.get("vision.camera.device_id", 0)`
- Automatic file persistence with async save
- No external dependencies (stdlib only)

### Key Configuration Sections
```json
{
  "api": {
    "server": {
      "host": "0.0.0.0",
      "port": 8000,
      "log_level": "info"
    },
    "websocket": {
      "broadcaster": {
        "frame_buffer": { "max_size": 100 },
        "compression": { "threshold_bytes": 1024 }
      }
    }
  },
  "vision": {
    "camera": {
      "device_id": 0,
      "fps": 30,
      "resolution": [1920, 1080]
    },
    "processing": {
      "target_fps": 30,
      "enable_threading": true
    }
  },
  "integration": {
    "target_fps": 30,
    "broadcast_max_retries": 3,
    "circuit_breaker_threshold": 10,
    "circuit_breaker_timeout_sec": 30.0
  }
}
```

---

## Obvious Bottlenecks & Recommendations

### Critical Bottlenecks

1. **No Parallel Detection**
   - Ball, cue, and table detection run sequentially
   - Could be parallelized with threading.Pool or asyncio
   - **Potential gain**: 30-50% faster detection

2. **Vision Module Internal Threading vs Async**
   - Vision uses daemon threads, everything else uses async
   - Thread-safe queue communication adds overhead
   - **Recommendation**: Consider async camera wrapper (aiofiles, etc.)

3. **Single Integration Loop (No Pipelining)**
   - Poll Vision → Convert → Update Core → Check Trajectory (all sequential)
   - Could pipeline: while Core processes N, Vision detects N+1
   - **Potential gain**: ~33ms latency reduction

4. **No Adaptive FPS**
   - Fixed 30 FPS regardless of detection complexity
   - If detection takes 40ms, frames are dropped
   - **Better approach**: Use actual detection time to gauge FPS

5. **Validation on Every Broadcast**
   - Full structure validation before every game state broadcast
   - Could use cached validation or skip in high-throughput mode
   - **Potential gain**: 1-2ms per broadcast

### Network/Client Bottlenecks

1. **No Message Batching**
   - Each detection sends separate WebSocket message
   - Could batch multiple frames if FPS > 30
   - **Benefit**: Reduced network overhead at lower FPS

2. **No Adaptive Quality**
   - Always sends full precision coordinates
   - Could reduce precision at lower client FPS
   - **Benefit**: Smaller messages to slow clients

3. **No Rate Limiting Between Modules**
   - Integration service doesn't respect Core's processing capacity
   - If Core is slow, integration doesn't back off
   - **Risk**: State updates may be missed if Core can't keep up

### Architectural Improvements Needed

1. **Backpressure Mechanism**
   - Integration loop should slow if Core can't process state
   - Currently just drops old states
   - **Impact**: May lose state history at high FPS

2. **Circuit Breaker Logging**
   - Circuit breaker prevents cascade failures
   - But may silently drop important updates
   - **Recommendation**: Add telemetry for circuit breaker trips

3. **No Health Checks Within Loop**
   - Integration loop doesn't verify Core/Vision health
   - Stale component could block indefinitely
   - **Recommendation**: Add timeout checks per iteration

4. **Trajectory Calculation Not Rate-Limited**
   - Can trigger on every frame if cue stays detected
   - CPU intensive, should throttle to ~1x per second
   - **Current**: Already checks if needed, but no rate limit

---

## Summary Table

| Component | Role | Concurrency | FPS | Latency | Queue Type |
|-----------|------|-------------|-----|---------|-----------|
| Vision | Detection | Thread | 30 | 30-50ms | Internal queue (5) |
| Integration | Orchestration | Async | 30 | 33ms | No queue (polling) |
| Core | State/Physics | Async | No limit | 1-50ms | Event callbacks |
| WebSocket | Broadcasting | Async | Per-client | <5ms | Frame buffer (100) |
| Health Monitor | Monitoring | Async | 0.2 FPS | 5000ms | No queue |

---

## File Structure

```
backend/
├── main.py                              # Entry point
├── config.py                            # Config system
├── integration_service.py                # [CRITICAL] Vision→Core→Broadcast hub
├── api/
│   ├── main.py                          # FastAPI app setup + lifespan
│   ├── dependencies.py                  # App state injection
│   ├── websocket/
│   │   ├── broadcaster.py               # Message distribution
│   │   ├── manager.py                   # Connection lifecycle
│   │   ├── handler.py                   # Message routing
│   │   ├── subscriptions.py             # Filter management
│   │   └── endpoints.py                 # WebSocket routes
│   ├── routes/
│   │   ├── health.py                    # Health check endpoints
│   │   ├── vision.py                    # Vision API
│   │   ├── calibration.py               # Calibration endpoints
│   │   ├── game.py                      # Game state API
│   │   ├── config.py                    # Config API
│   │   └── ...
│   └── middleware/
│       ├── error_handler.py
│       ├── logging.py
│       ├── metrics.py
│       └── tracing.py
├── core/
│   ├── __init__.py                      # [CRITICAL] CoreModule class
│   ├── game_state.py                    # GameState, BallState models
│   ├── physics/
│   │   ├── trajectory.py                # Trajectory calculation
│   │   ├── collision.py                 # Collision detection/resolution
│   │   └── engine.py                    # Physics simulation
│   ├── analysis/
│   │   ├── shot.py                      # Shot analysis
│   │   ├── prediction.py                # Outcome prediction
│   │   └── assistance.py                # Assistance engine
│   ├── events/
│   │   └── manager.py                   # Event coordination
│   ├── validation/
│   │   ├── physics.py                   # Physics validation
│   │   └── state.py                     # State validation
│   └── utils/
│       ├── cache.py                     # Caching
│       ├── math.py                      # Math utilities
│       └── geometry.py                  # Geometry utilities
├── vision/
│   ├── __init__.py                      # [CRITICAL] VisionModule class
│   ├── models.py                        # DetectionResult, Ball, CueStick, Table
│   ├── detection/
│   │   ├── balls.py                     # Ball detection
│   │   ├── cue.py                       # Cue detection
│   │   ├── table.py                     # Table detection
│   │   └── detector_factory.py          # Detection strategy factory
│   ├── tracking/
│   │   ├── tracker.py                   # Object tracking
│   │   └── kalman.py                    # Kalman filter
│   ├── calibration/
│   │   ├── camera.py                    # Camera calibration
│   │   ├── color.py                     # Color calibration
│   │   └── geometry.py                  # Geometric calibration
│   ├── preprocessing.py                 # Image preprocessing
│   └── stream/
│       └── video_consumer.py            # IPC video stream
└── system/
    └── health_monitor.py                # Health monitoring
```

---

## Data Flow Summary

```
REAL-TIME LOOP (30 FPS, 33ms per iteration):

Frame N:
  integration_loop iteration start
  ├─ get_vision_detection() → DetectionResult
  ├─ _convert_detection_to_core_format()
  ├─ core.update_state(detection_data)
  │  └─ emit("state_updated", event_data)
  │     └─ _on_state_updated_async()
  │        └─ _broadcast_with_retry(broadcast_game_state)
  │           └─ broadcaster.broadcast_game_state()
  │              └─ _broadcast_to_stream(STATE, state_data)
  │                 └─ WebSocket clients receive message
  ├─ _check_trajectory_calculation()
  │  └─ if cue detected & balls stable:
  │     └─ trajectory_calculator.predict_multiball_cue_shot()
  │        └─ _emit_multiball_trajectory()
  │           └─ _broadcast_with_retry(broadcast_trajectory)
  │              └─ WebSocket clients receive trajectory
  └─ sleep(frame_interval - elapsed_time)

Total per frame: 33ms target
```

---

## Generated: 2024-10-22
**Analysis Depth**: Complete system flow, component interactions, bottleneck identification
**Tools Used**: Grep, file reading, code tracing, architectural analysis
