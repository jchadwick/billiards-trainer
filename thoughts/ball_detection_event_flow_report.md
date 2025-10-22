# Ball Detection Event Flow Analysis - Complete Backend Report

## Executive Summary

The billiards-trainer backend has a comprehensive event system for ball detection and broadcasting. Ball detection events flow through a well-architected pipeline:

**Vision Module** → **Detection** → **Core Module** → **Event System** → **Integration Service** → **WebSocket Broadcaster** → **Clients**

The system is functional but has some critical gaps that should be understood for effective debugging and development.

---

## 1. Ball Detection Flow - Complete Pipeline

### 1.1 Ball Detection (Vision Module)

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/balls.py`

The `BallDetector` class handles all ball detection:

- **Method:** `detect_balls()` (lines 522-588)
- **Process:**
  1. Applies background subtraction if configured
  2. Runs combined detection (Hough circles, contour-based, blob detection)
  3. Filters and validates candidates
  4. Classifies ball types by color
  5. Removes overlapping detections
  6. Resolves conflicts (multiple cue/8-balls)
  7. Returns list of `Ball` objects

- **Output:** List of `Ball` objects with:
  - `position` (x, y in pixels)
  - `radius` (pixels)
  - `ball_type` (CUE, EIGHT, OTHER)
  - `number` (None for simplified classification)
  - `confidence` (0-1 score)
  - `velocity` (0, 0 for static detection)
  - `is_moving` (False for initial detection)

**Key Configuration:** `BallDetectionConfig` (lines 42-200)
- Detection method: COMBINED (Hough + Contour + Blob)
- Quality filters: circularity, confidence, convexity
- Size constraints: 15-26px radius
- Pocket detection and exclusion
- Playing area masking

---

### 1.2 Detection Result Collection

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/vision/detection/detector_adapter.py` or similar

The vision module collects detection results into a `DetectionResult` object with:
- `balls` - List of detected Ball objects
- `cue` - Detected CueStick or None
- `table` - Detected TableState or None
- `timestamp` - Frame timestamp
- `frame_number` - Frame count
- `confidence` - Overall detection confidence

---

### 1.3 Vision Module Frame Processing

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/vision/__init__.py`

The `VisionModule.process_frame()` method:
1. Gets frame from camera
2. Runs ball detection via `BallDetector.detect_balls()`
3. Detects cue stick position
4. Detects table
5. Returns complete `DetectionResult`

---

## 2. Event Creation and Flow - The Integration Service

### 2.1 Integration Service Overview

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/integration_service.py`

The `IntegrationService` is the critical bridge:
- Polls Vision for detections at configured FPS (default 30Hz)
- Converts Vision format to Core format
- Updates Core game state
- Subscribes to Core events
- Broadcasts state changes via WebSocket

**Key Methods:**
- `start()` (line 266) - Starts integration loop
- `_integration_loop()` (line 355) - Main polling loop
- `_get_vision_detection()` (line 395) - Polls vision
- `_process_detection()` (line 409) - Updates core state
- `_on_state_updated()` (line 1006) - Handles core state events
- `_on_state_updated_async()` (line 1034) - Broadcasts to WebSocket

### 2.2 Detection Processing Pipeline

**Flow in `_process_detection()` (line 409-425):**

```
1. Get DetectionResult from Vision via _get_vision_detection()
   ↓
2. Convert to Core format via _convert_detection_to_core_format()
   - Converts Ball → detection_data dict
   - Converts CueStick → detection_data dict
   - Converts Table → detection_data dict
   ↓
3. Update Core state via core.update_state(detection_data)
   - Core validates and processes the data
   - Core updates GameState
   - Core emits "state_updated" event
   ↓
4. Check if trajectory calculation needed via _check_trajectory_calculation()
   ↓
5. Integration service subscribes to "state_updated" event
   - Triggers _on_state_updated() callback
```

### 2.3 State Update Event Handler

**Location:** `_on_state_updated_async()` (lines 1034-1140)

**Process:**
1. Validates event_data structure (dict with balls, table, timestamp)
2. Extracts balls list, cue object, table object
3. Converts position dicts from `{x, y}` to `[x, y]` format for broadcaster
4. Calls `_broadcast_with_retry()` with:
   - `broadcaster.broadcast_game_state` method
   - balls_converted list
   - cue object
   - table object

**Critical Lines:**
- Line 1064: Extracts balls from event_data
- Line 1065: Extracts cue from event_data
- Line 1066: Extracts table from event_data
- Line 1112: Converts position format for broadcaster
- Line 1133-1140: Broadcasts with retry logic

---

## 3. WebSocket Broadcasting System

### 3.1 Message Broadcaster

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/broadcaster.py`

The `MessageBroadcaster` class handles WebSocket streaming:

#### Key Method: `broadcast_game_state()` (lines 265-336)

```python
async def broadcast_game_state(
    self,
    balls: list[dict[str, Any]],
    cue: Optional[dict[str, Any]] = None,
    table: Optional[dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
)
```

**Process:**
1. Validates balls parameter (must be list)
2. Validates each ball has required fields
3. Validates position field format [x, y]
4. Constructs state_data dict with:
   - balls list
   - cue object
   - table object
   - timestamp
   - sequence number
   - ball_count
5. Broadcasts via `_broadcast_to_stream(StreamType.STATE, state_data)`

**Validation Logic (lines 283-324):**
- Checks balls is a list (returns early if not)
- Validates each ball is a dict
- Validates each ball has 'position' field
- Validates position is [x, y] format

**Internal Broadcasting (lines 563-590):**
- Calls `websocket_manager.broadcast_to_stream()`
- Tracks latency and message stats
- Handles JSON serialization with NumpyEncoder
- Updates broadcast statistics

### 3.2 WebSocket Manager and Handler

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/manager.py`

The `websocket_manager` handles:
- Client subscriptions to streams
- Stream type routing
- Connection management
- Filtering and targeting

**Key Stream Types:**
```python
class StreamType(Enum):
    STATE = "state"
    TRAJECTORY = "trajectory"
    FRAME = "frame"
    ALERT = "alert"
    CONFIG = "config"
```

### 3.3 WebSocket Endpoints

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/api/websocket/endpoints.py`

Main WebSocket endpoint:
- Path: `/ws`
- Handles connections and message routing
- Subscriptions managed per client
- Rate limiting: 10 connections per IP per minute

---

## 4. Event System Architecture

### 4.1 Core Event Manager

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/core/events/manager.py`

The `EventManager` class provides:
- Legacy subscription system via `subscribe_to_events()`
- Enhanced event system with `emit_enhanced_event()`
- Event history tracking
- Event filtering
- Module coordination

**Event Types (lines 22-46):**
- STATE_UPDATED = "state_updated"
- VISION_DATA_RECEIVED = "vision_data_received"
- TRAJECTORY_CALCULATED = "trajectory_calculated"
- BALL_MOVED = "ball_moved"
- COLLISION_DETECTED = "collision_detected"
- BALL_POCKETED = "ball_pocketed"
- And others...

### 4.2 Core Event Handlers

**Location:** `/Users/jchadwick/code/billiards-trainer/backend/core/events/handlers.py`

The `CoreEventHandlers` class provides handlers for:
- `handle_state_change()` - Processes state updates
- `handle_vision_data()` - Handles vision input
- `handle_ball_movement()` - Ball motion tracking
- `handle_collision_event()` - Collision detection
- `handle_shot_event()` - Shot detection
- And others...

---

## 5. Integration Service Architecture

### 5.1 Initialization

**Location:** `__init__()` (lines 187-264)

Initializes with:
- Vision module reference
- Core module reference
- Message broadcaster reference
- Configuration module
- Trajectory calculator
- Physics validators
- Circuit breaker for fault tolerance
- Broadcast metrics tracking

### 5.2 Main Integration Loop

**Location:** `_integration_loop()` (lines 355-393)

```
while self.running:
  1. Get detection from vision via _get_vision_detection()
  2. If detection available:
     a. Process via _process_detection()
     b. Increment frame count
     c. Log progress periodically
  3. Maintain target FPS (default 30Hz)
  4. Sleep to avoid busy-waiting
  5. Handle errors with retry delay
```

### 5.3 Vision Data Subscription

**Location:** `_subscribe_to_core_events()` (lines 343-353)

```python
self.core.subscribe_to_events("state_updated", self._on_state_updated)
self.core.subscribe_to_events("trajectory_calculated", self._on_trajectory_calculated)
```

This creates the critical link between:
- **Core Module** (emits state updates)
- **Integration Service** (receives and broadcasts)
- **WebSocket Clients** (receive broadcasts)

---

## 6. The Complete Data Flow Diagram

```
VISION MODULE (detection/balls.py)
    ↓
Ball Detection (hough + contour + blob)
    ↓ returns Ball[], CueStick, Table
    ↓
VISION MODULE (process_frame())
    ↓
DetectionResult {balls[], cue, table, timestamp, frame_number}
    ↓
INTEGRATION SERVICE (integration_service.py)
    ↓
_integration_loop() polls vision at 30Hz
    ↓ gets DetectionResult
    ↓
_process_detection()
    ↓ converts to Core format
    ↓
CORE MODULE (core/game_state.py)
    ↓
update_state(detection_data)
    ↓ validates and updates GameState
    ↓ emits "state_updated" event
    ↓
EVENT MANAGER (core/events/manager.py)
    ↓
emit_event("state_updated", event_data)
    ↓ calls subscribers
    ↓
INTEGRATION SERVICE (_on_state_updated)
    ↓ schedules async handler
    ↓
_on_state_updated_async()
    ↓ validates and converts data
    ↓
BROADCASTER (broadcast_game_state())
    ↓ validates balls/cue/table
    ↓
WEBSOCKET MANAGER
    ↓ _broadcast_to_stream(StreamType.STATE, data)
    ↓
WEBSOCKET HANDLER
    ↓
Connected WebSocket Clients
    ↓
Browser/Client Applications
```

---

## 7. Identified Gaps and Potential Issues

### 7.1 Frame Broadcasting Disabled

**Issue:** MJPEG frame broadcasting via WebSocket is disabled
**Location:** `broadcaster.py` lines 245-253

```python
# DISABLED: Frame broadcasting via WebSocket to prevent browser crashes from large base64 images
# Frames are still buffered for internal metrics and potential future use
# if self.is_streaming and not self.frame_queue.full():
#     try:
#         self.frame_queue.put_nowait(frame_data)
```

**Impact:** WebSocket clients do NOT receive video frames directly
**Solution:** Use MJPEG HTTP endpoint at `/api/v1/stream/video`

### 7.2 Circuit Breaker for Broadcast Failures

**Issue:** When all WebSocket clients disconnect, broadcasts fail repeatedly
**Location:** `integration_service.py` lines 59-181

**Solution:** CircuitBreaker pattern prevents cascade failures:
- Opens after 10 consecutive failures
- Blocks broadcasts for 30 seconds
- Allows testing to recover
- Reopens automatically

### 7.3 Event Loop Requirement

**Issue:** Integration service requires running event loop
**Location:** `integration_service.py` lines 275-276

```python
# Capture the event loop for use in event callbacks
self._event_loop = asyncio.get_running_loop()
```

**Impact:** Cannot be used in non-async contexts
**Status:** Expected behavior - part of async architecture

### 7.4 Data Format Conversions

**Issue:** Multiple format conversions may cause data loss
**Locations:**
1. Vision Ball → Core BallState (line 1443-463)
2. Core GameState → dict (line 1107-1117)
3. position Vector2D → [x, y] (line 1112)

**Validation:** All conversions have validation and error handling

---

## 8. Configuration and Performance

### 8.1 Integration Service Configuration

**File:** `default.json` or similar

```json
{
  "integration": {
    "target_fps": 30,
    "log_interval_frames": 300,
    "error_retry_delay_sec": 0.1,
    "broadcast_max_retries": 3,
    "broadcast_retry_base_delay_sec": 0.1,
    "circuit_breaker_threshold": 10,
    "circuit_breaker_timeout_sec": 30.0,
    "max_ball_velocity_m_per_s": 10.0
  }
}
```

### 8.2 WebSocket Configuration

```json
{
  "api": {
    "websocket": {
      "broadcaster": {
        "frame_buffer": {
          "max_size": 100,
          "max_age_seconds": 5.0
        },
        "compression": {
          "threshold_bytes": 1024,
          "level": 6,
          "ratio_threshold": 0.9
        },
        "fps": {
          "default_target_fps": 30.0,
          "limiter_cleanup_age_seconds": 300
        }
      }
    }
  }
}
```

---

## 9. Debugging Ball Detection Events

### 9.1 Check Ball Detection

```bash
# See if balls are being detected
curl http://localhost:8000/api/v1/vision/detect
```

### 9.2 Check Core State Updates

Add logging to `_on_state_updated_async()` in integration_service.py:
```python
logger.info(f"State updated: {len(balls)} balls detected")
```

### 9.3 Check WebSocket Broadcasting

Monitor WebSocket messages:
1. Connect to `ws://localhost:8000/ws`
2. Send subscription: `{"type": "subscribe", "streams": ["state"]}`
3. Receive state messages

### 9.4 Check Event System

```python
# In integration_service.py
print(f"Event loop: {self._event_loop}")
print(f"Circuit breaker open: {self.circuit_breaker.is_open}")
print(f"Broadcast stats: {self.broadcast_metrics}")
```

---

## 10. Thread Safety and Async Considerations

### 10.1 Thread-Safe Event Subscription

**Location:** `core/events/manager.py` lines 249-265

```python
def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
    """Subscribe to state change events with thread-safe ID generation."""
    subscription_id = str(uuid.uuid4())
    with self._lock:
        self._subscribers[event_type][subscription_id] = callback
    return subscription_id
```

Uses `threading.RLock()` for thread safety.

### 10.2 Cross-Thread Event Scheduling

**Location:** `integration_service.py` lines 1023-1030

```python
self._event_loop.call_soon_threadsafe(
    lambda: asyncio.ensure_future(
        self._on_state_updated_async(event_type, event_data),
        loop=self._event_loop,
    )
)
```

Schedules async work from sync callback in different thread.

---

## 11. Summary of Key Files

| File | Purpose | Key Methods |
|------|---------|-------------|
| `vision/detection/balls.py` | Ball detection | `detect_balls()`, `classify_ball_type()` |
| `integration_service.py` | Vision→Core→Broadcast | `_integration_loop()`, `_on_state_updated_async()` |
| `core/game_state.py` | Game state management | `update_state()` |
| `core/events/manager.py` | Event system | `emit_event()`, `subscribe_to_events()` |
| `api/websocket/broadcaster.py` | WebSocket broadcasting | `broadcast_game_state()` |
| `api/websocket/manager.py` | WebSocket management | `broadcast_to_stream()` |
| `api/websocket/endpoints.py` | WebSocket endpoints | `websocket_endpoint()` |

---

## 12. Recommendations

1. **Add Observability:** Log all state transitions and broadcasts
2. **Add Metrics:** Track detection latency, broadcast latency, error rates
3. **Add Health Checks:** Monitor circuit breaker status, event loop status
4. **Add Tests:** Test ball detection with various lighting conditions
5. **Document Configuration:** Provide clear docs on tuning FPS, thresholds
6. **Monitor Performance:** Track CPU usage, memory, frame drops
