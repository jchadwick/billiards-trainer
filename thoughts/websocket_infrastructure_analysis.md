# WebSocket Infrastructure Analysis Report

## Executive Summary

The backend implements a comprehensive WebSocket infrastructure with the following key components:

1. **WebSocket Server Setup** - FastAPI with Starlette WebSocket support
2. **Connection Management** - Advanced handler with tracking, monitoring, and health checks
3. **Event Publishing Architecture** - Multi-layered: Handler → Manager → Broadcaster
4. **Subscription System** - Stream-based with filtering and quality management
5. **Data Flow Integration** - Vision → Core → Broadcast pipeline via IntegrationService

### Critical Finding: Missing Connection Between Broadcaster and WebSocket Manager

There is a **significant architectural gap** where events are being published to the broadcaster but the subscribers are not being properly maintained between the handler, manager, and broadcaster layers.

---

## 1. WebSocket Server Initialization

### File: `/backend/api/main.py` (lines 260-533)

**Startup Flow:**
1. FastAPI application created with lifecycle management (lifespan context manager)
2. WebSocket system initialized during startup:
   - `initialize_websocket_system()` called (line 294)
   - Message broadcaster started
   - Handler monitoring started
   - All components initialized before integration service starts

**Endpoint Registration:**
- REST management endpoints: `/api/v1/websocket/*` (prefix from websocket_router)
- Main WebSocket endpoint: `/api/v1/ws` (line 533)
- Health endpoints: `/api/v1/websocket/health`
- Broadcast endpoints: `/api/v1/websocket/broadcast/*`

**Initialization Code (lines 287-314):**
```python
# Initialize WebSocket components
app_state.websocket_manager = websocket_manager
app_state.websocket_handler = websocket_handler
app_state.message_broadcaster = message_broadcaster

# Start WebSocket system services
await initialize_websocket_system()

# Initialize and start integration service
app_state.integration_service = IntegrationService(
    vision_module=app_state.vision_module,
    core_module=app_state.core_module,
    message_broadcaster=app_state.message_broadcaster,
    config_module=app_state.config_module,
)
await app_state.integration_service.start()
```

### File: `/backend/api/websocket/__init__.py` (lines 141-165)

**Initialization Functions:**
```python
async def initialize_websocket_system() -> None:
    """Initialize the complete WebSocket system."""
    await message_broadcaster.start_streaming()
    await websocket_handler.start_monitoring()
```

**Shutdown Functions:**
```python
async def shutdown_websocket_system() -> None:
    """Gracefully shutdown the WebSocket system."""
    await message_broadcaster.stop_streaming()
    await websocket_handler.stop_monitoring()
    # Disconnect all active connections
```

---

## 2. WebSocket Connection Handler

### File: `/backend/api/websocket/handler.py` (lines 17-567)

**Class: WebSocketConnection (lines 17-88)**
- Represents individual WebSocket connection with metadata
- Tracks client_id, user_id, subscriptions, connection time
- Maintains quality metrics (latency, jitter)
- Stores message statistics (count, bytes sent/received)

**Class: WebSocketHandler (lines 90-567)**

**Key Methods:**
- `async connect(websocket, token)` (lines 125-180)
  - Accepts new WebSocket connection
  - Generates unique client_id (UUID)
  - Stores connection in `self.connections` dict
  - Starts monitoring if not already running
  - Sends welcome message to client

- `async disconnect(client_id)` (lines 182-209)
  - Cleans up connection state
  - Removes from user_connections mapping
  - Cleans up rate limiting windows
  - Stops monitoring if no connections remain

- `async handle_message(client_id, message)` (lines 211-282)
  - Processes incoming JSON messages
  - Handles message types: ping, subscribe, unsubscribe, get_status
  - Rate limiting enforcement (100 msg/min per connection)
  - Error handling and validation

- `async broadcast_to_subscribers(stream_type, message)` (lines 312-338)
  - **KEY METHOD**: Broadcasts to clients subscribed to a specific stream
  - Filters connections by subscription status and alive status
  - Sends concurrently to all subscribers
  - Logs failures

- `async broadcast_message(message)` (lines 340-359)
  - Broadcasts message to all connected clients
  - Concurrent sending with error handling

**Global Instance:**
```python
websocket_handler = WebSocketHandler()  # Line 567
```

---

## 3. WebSocket Manager (Client Lifecycle)

### File: `/backend/api/websocket/manager.py` (lines 78-562)

**Enums:**
- `ConnectionState` - CONNECTING, CONNECTED, RECONNECTING, DISCONNECTING, DISCONNECTED, ERROR
- `StreamType` - FRAME, STATE, TRAJECTORY, ALERT, CONFIG

**Key Data Structures:**

**ClientSession (lines 51-76):**
```python
client_id: str
user_id: Optional[str]
connection_state: ConnectionState
subscription_filters: dict[StreamType, SubscriptionFilter]
permissions: set[str]
metadata: dict[str, Any]
```

**SubscriptionFilter (lines 38-49):**
```python
stream_type: StreamType
min_fps: Optional[float]
max_fps: Optional[float]
quality_level: str
include_fields: Optional[list[str]]
exclude_fields: Optional[list[str]]
conditions: dict[str, Any]
```

**Class: WebSocketManager (lines 78-562)**

**Key Methods:**
- `async register_client(client_id, user_id, permissions, metadata)` (lines 93-123)
  - Creates ClientSession and stores in `self.sessions`
  - Tracks user sessions for multi-connection support
  - Emits "client_registered" event

- `async subscribe_to_stream(client_id, stream_type, filter_config)` (lines 157-212)
  - **CRITICAL**: Subscribes client to stream
  - Validates permissions
  - Stores filter configuration
  - **ALSO subscribes in websocket_handler** (line 198):
    ```python
    if client_id in websocket_handler.connections:
        connection = websocket_handler.connections[client_id]
        connection.add_subscription(stream_type.value)
    ```
  - Emits "stream_subscribed" event

- `async broadcast_to_stream(stream_type, data, apply_filters)` (lines 249-305)
  - **CRITICAL METHOD**: Broadcasts to all subscribers of a stream
  - Gets subscribers from `self.stream_subscribers[stream_type]`
  - Applies message filtering if requested
  - Sends via `websocket_handler.send_to_client()`

- `async send_alert(level, message, code, ...)` (lines 329-365)
  - Sends targeted alerts to specific clients/users or all subscribers

- `async get_session_info(client_id)` (lines 367-393)
  - Returns detailed session information

**Global Instance:**
```python
websocket_manager = WebSocketManager()  # Line 562
```

---

## 4. Message Broadcaster

### File: `/backend/api/websocket/broadcaster.py` (lines 1-708)

**Key Classes:**

**FrameMetrics (lines 40-51):**
- Tracks frame streaming performance
- Stores: frames_sent, bytes_sent, compression_ratio, average_latency, dropped_frames

**BroadcastStats (lines 53-70):**
- Overall broadcasting statistics
- Stores: messages_sent, bytes_sent, failed_sends, average_latency, peak_latency

**FrameBuffer (lines 72-124):**
- Circular buffer for frame data
- Automatic cleanup of old frames
- Calculates frame rate

**Class: MessageBroadcaster (lines 127-708)**

**Key Methods:**
- `async broadcast_frame(image_data, width, height, format, quality, fps)` (lines 185-263)
  - Converts image to base64
  - Applies compression if beneficial
  - **DISABLED**: Frame broadcasting commented out (lines 245-253)
  - Only updates metrics, doesn't broadcast to clients
  - **BOTTLENECK**: Video streaming to WebSocket disabled to prevent browser crashes

- `async broadcast_game_state(balls, cue, table, timestamp)` (lines 265-336)
  - **KEY METHOD**: Broadcasts game state with validation
  - Validates balls and cue data
  - Calls `_broadcast_to_stream(StreamType.STATE, state_data)` (line 336)

- `async broadcast_trajectory(lines, collisions, confidence, calculation_time)` (lines 338-421)
  - **KEY METHOD**: Broadcasts trajectory data with validation
  - Validates trajectory lines and collision data
  - Calls `_broadcast_to_stream(StreamType.TRAJECTORY, trajectory_data)` (line 421)

- `async broadcast_alert(level, message, code, details, target_clients, target_users)` (lines 423-449)
  - Broadcasts alert messages with optional targeting
  - Can send to all subscribers or specific clients/users

- `async broadcast_config_update(config_section, config_data, change_summary)` (lines 451-466)
  - Broadcasts configuration changes
  - Calls `_broadcast_to_stream(StreamType.CONFIG, config_update_data)` (line 466)

- `async _broadcast_to_stream(stream_type, data)` (lines 563-590)
  - **INTERNAL METHOD**: Core broadcasting mechanism
  - Calls `websocket_manager.broadcast_to_stream(stream_type, data, apply_filters=True)`
  - Tracks latency and statistics
  - Catches and logs broadcast failures

- `async start_streaming()` (lines 152-167)
  - Starts background frame processor and cleanup tasks

- `async stop_streaming()` (lines 169-183)
  - Stops all background tasks

**Global Instance:**
```python
message_broadcaster = MessageBroadcaster()  # Line 708
```

**Configuration from config.py:**
- `api.websocket.broadcaster.frame_buffer.max_size` (default 100)
- `api.websocket.broadcaster.compression.threshold_bytes` (default 1024)
- `api.websocket.broadcaster.compression.level` (default 6)
- `api.websocket.broadcaster.fps.default_target_fps` (default 30.0)

---

## 5. Event Routing & Subscription System

### File: `/backend/api/websocket/manager.py` (lines 84-86)

**Stream Subscribers Tracking:**
```python
self.stream_subscribers: dict[StreamType, set[str]] = {
    stream_type: set() for stream_type in StreamType
}
```

**Subscription Flow:**
1. Client connects → `handler.connect()` called
2. Client sends subscribe message → `handler._handle_subscribe()`
3. Subscribe handler calls `websocket_manager.subscribe_to_stream()`
4. Manager adds client_id to `stream_subscribers[stream_type]`
5. Manager also updates handler subscription: `connection.add_subscription(stream_type.value)`
6. Emits "stream_subscribed" event

**Broadcast Flow:**
1. Event triggered (e.g., trajectory calculated)
2. Calls `broadcaster.broadcast_trajectory()`
3. Broadcaster calls `websocket_manager.broadcast_to_stream()`
4. Manager looks up subscribers in `stream_subscribers[StreamType]`
5. Sends to each subscriber via `websocket_handler.send_to_client()`

### File: `/backend/api/websocket/subscriptions.py` (lines 87-100)

**Advanced Subscription Manager** - Provides additional filtering and aggregation capabilities on top of basic subscriptions.

---

## 6. Integration Service (Vision → Core → Broadcast)

### File: `/backend/integration_service.py` (lines 1-1100+)

**Purpose:** Connects Vision detection → Core state update → WebSocket broadcast

**Initialization (lines 187-264):**
- Stores references to vision_module, core_module, message_broadcaster
- Initializes trajectory calculator, validators
- Configures broadcast retry logic and circuit breaker

**Core Loop (lines 355-393):**
```python
async def _integration_loop(self) -> None:
    while self.running:
        # Get detection from Vision
        detection_result = await self._get_vision_detection()

        if detection_result:
            # Process detection (update Core state)
            await self._process_detection(detection_result)
```

**Event Subscriptions (lines 343-353):**
```python
def _subscribe_to_core_events(self) -> None:
    self.core.subscribe_to_events("state_updated", self._on_state_updated)
    self.core.subscribe_to_events(
        "trajectory_calculated", self._on_trajectory_calculated
    )
```

**State Update Handler (lines 1006-1099):**
```python
async def _on_state_updated_async(self, event_type: str, event_data: dict) -> None:
    # Extract balls, cue, table from event_data
    balls = event_data.get("balls", [])
    cue = event_data.get("cue")
    table = event_data.get("table")

    # Broadcast via _broadcast_with_retry()
    await self._broadcast_with_retry(
        self.broadcaster.broadcast_game_state,
        "broadcast_game_state",
        f"{len(balls)} balls",
        balls=balls,
        cue=cue,
        table=table,
    )
```

**Trajectory Calculation (lines 663-853):**
- Detects when cue is present and balls are stationary
- Calculates multiball trajectory via trajectory_calculator
- Calls `_emit_multiball_trajectory()` to broadcast

**Broadcast with Retry (lines 914-1004):**
```python
async def _broadcast_with_retry(
    self,
    broadcast_func: Callable,
    operation_name: str,
    data_summary: str,
    *args, **kwargs
) -> bool:
    # Circuit breaker check
    if not self.circuit_breaker.can_attempt():
        return False

    # Retry loop with exponential backoff
    for attempt in range(self.max_retries + 1):
        try:
            await broadcast_func(*args, **kwargs)
            self.circuit_breaker.record_success()
            return True
        except Exception as e:
            # Handle retries
```

---

## 7. WebSocket Endpoints (REST API)

### File: `/backend/api/websocket/endpoints.py` (lines 1-478)

**Main WebSocket Endpoint (lines 78-175):**
- `/ws` - Accepts WebSocket connections
- Rate limiting per IP address (10 connections/min)
- Connection lifecycle management
- Message handling loop

**REST Management Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/connections` | GET | Get all active connections |
| `/connections/{client_id}` | GET | Get specific connection info |
| `/connections/{client_id}/subscribe` | POST | Subscribe to streams |
| `/connections/{client_id}/unsubscribe` | POST | Unsubscribe from streams |
| `/connections/{client_id}/disconnect` | POST | Force disconnect |
| `/health` | GET | System health status |
| `/health/summary` | GET | Health summary |
| `/broadcast/frame` | POST | Broadcast test frame |
| `/broadcast/alert` | POST | Broadcast test alert |
| `/system/start` | POST | Start WebSocket system |
| `/system/stop` | POST | Stop WebSocket system |
| `/metrics` | GET | Get detailed metrics |

---

## 8. Health Monitoring & Metrics

### File: `/backend/api/websocket/monitoring.py` (lines 1-100+)

**HealthStatus Enum:**
- EXCELLENT (< 20ms latency)
- GOOD (< 50ms latency)
- FAIR (< 100ms latency)
- POOR (< 200ms latency)
- CRITICAL (> 200ms latency or high packet loss)

**QualityMetrics:**
- Latency measurements (min, max, avg, jitter)
- Throughput metrics (messages, bytes)
- Error metrics (failed sends, timeouts, disconnections)
- Connection stability score
- Active issues tracking

---

## 9. Configuration Management

### File: `/backend/config.py` (via config_manager)

**WebSocket Configuration Keys:**
```python
api.websocket.broadcaster.frame_buffer.max_size
api.websocket.broadcaster.compression.threshold_bytes
api.websocket.broadcaster.compression.level
api.websocket.broadcaster.compression.ratio_threshold
api.websocket.broadcaster.frame_queue.max_size
api.websocket.broadcaster.fps.default_target_fps
api.websocket.broadcaster.cleanup.interval_seconds
api.websocket.manager.max_reconnect_attempts
api.websocket.manager.reconnect_delay
api.websocket.manager.auto_reconnect_enabled
api.websocket.manager.max_reconnect_delay
integration.target_fps
integration.broadcast_max_retries
integration.broadcast_retry_base_delay_sec
integration.circuit_breaker_threshold
integration.circuit_breaker_timeout_sec
```

---

## 10. Critical Issues & Bottlenecks

### Issue 1: Subscription Management Desynchronization

**Problem:** There are two separate subscription tracking systems that can get out of sync:

1. **WebSocketHandler.connections[client_id].subscriptions** (set of strings)
2. **WebSocketManager.stream_subscribers[StreamType]** (set of client_ids)

**Code References:**
- Handler stores subscriptions: `/backend/api/websocket/handler.py:36` and `:61-67`
- Manager stores subscriptions: `/backend/api/websocket/manager.py:84-86`

**Symptom:** When clients subscribe/unsubscribe, both systems should be updated. Line 461 in `handler.py` attempts to update the manager:
```python
await websocket_manager.subscribe_to_stream(client_id, stream_type)
```

But this creates a circular dependency and potential race conditions.

**Impact:** Broadcasts may fail to reach some subscribed clients because the stream_subscribers dict is incomplete or out of sync.

### Issue 2: Frame Broadcasting is Disabled

**Problem:** Video frame broadcasting to WebSocket is completely disabled.

**File:** `/backend/api/websocket/broadcaster.py:245-253`
```python
# DISABLED: Frame broadcasting via WebSocket to prevent browser crashes from large base64 images
# Frames are still buffered for internal metrics and potential future use
# if self.broadcast_frame_queue.put_nowait(frame_data):
#     ...
```

**Reason:** Large base64-encoded images crash browsers

**Impact:**
- No video stream to clients
- Clients cannot see live camera feed
- Can still see state/trajectory data if subscribed

**Solution Needed:** Implement MJPEG stream endpoint as alternative

### Issue 3: Circuit Breaker Pattern Implementation

**Problem:** The integration service uses a circuit breaker to prevent cascade failures, but the circuit breaker state is not synchronized across multiple integration service instances (if running in parallel).

**File:** `/backend/integration_service.py:59-181`

**Implementation Details:**
- Opens after 10 consecutive failures (configurable)
- Stays open for 30 seconds (configurable)
- Half-open state allows test broadcast

**Impact:** In multi-worker deployments, each worker has its own circuit breaker instance, potentially leading to inconsistent broadcast behavior.

### Issue 4: Missing Broadcast Calls

**Problem:** Not all state changes are being broadcast. Events must be explicitly handled in integration service.

**File:** `/backend/integration_service.py:343-351`
- Only "state_updated" and "trajectory_calculated" are subscribed
- Other game events (ball pocketed, game started, etc.) are not broadcast

**Impact:** Frontend doesn't receive updates for all significant game events

### Issue 5: Synchronous to Asynchronous Event Callback Bridge

**Problem:** Core module events are synchronous callbacks, but WebSocket broadcasting requires async operations.

**File:** `/backend/integration_service.py:1006-1030`
```python
def _on_state_updated(self, event_type: str, event_data: dict[str, Any]) -> None:
    # Synchronous callback
    # Must schedule async work
    self._event_loop.call_soon_threadsafe(
        lambda: asyncio.ensure_future(...)
    )
```

**Risk:** If event loop is not available or dies, state updates won't be broadcast.

---

## 11. Message Flow Diagrams

### Normal Publish Flow

```
Vision Module
    ↓
Integration Service._get_vision_detection()
    ↓
Core Module.update_state()
    ↓
Core Module emits "state_updated" event
    ↓
IntegrationService._on_state_updated() [SYNC]
    ↓
call_soon_threadsafe() schedules async callback
    ↓
IntegrationService._on_state_updated_async() [ASYNC]
    ↓
_broadcast_with_retry()
    ↓
message_broadcaster.broadcast_game_state()
    ↓
_broadcast_to_stream(StreamType.STATE)
    ↓
websocket_manager.broadcast_to_stream()
    ↓
Get subscribers: stream_subscribers[StreamType.STATE]
    ↓
For each subscriber:
  websocket_handler.send_to_client(client_id, message)
    ↓
WebSocket.send_text()
    ↓
Client receives message
```

### Client Subscribe Flow

```
Client sends: {"type": "subscribe", "data": {"streams": ["state"]}}
    ↓
FastAPI endpoint receives message
    ↓
websocket_handler.handle_message()
    ↓
_handle_subscribe()
    ↓
websocket_manager.subscribe_to_stream()
    ↓
stream_subscribers[StreamType.STATE].add(client_id)
websocket_handler.connections[client_id].add_subscription("state")
    ↓
Send subscription confirmation
    ↓
Client now receives "state" broadcasts
```

---

## 12. Missing Connection Analysis

### Gap 1: Handler Subscriptions Not Used in Broadcasts

**Issue:** When broadcasting, the manager directly accesses `stream_subscribers[stream_type]`, but the handler's `connections[client_id].subscriptions` is also maintained.

**File References:**
- Maintained in: `/backend/api/websocket/handler.py:61-67` (add/remove_subscription)
- Used in: `/backend/api/websocket/handler.py:312-321` (broadcast_to_subscribers)
- But ignored in: `/backend/api/websocket/manager.py:249-305` (broadcast_to_stream)

**Result:** Two subscription lists that should be synchronized but might diverge

**Fix Needed:** Use single source of truth OR ensure synchronization

### Gap 2: Frame Broadcasting Disabled But Pipeline Intact

**Issue:** Frame broadcasting is disabled (line 245-253 in broadcaster.py), but the pipeline is still set up.

**Impact:**
- Clients that subscribe to "frame" stream won't receive anything
- No error message - just silent non-delivery
- Frontend may hang waiting for first frame

### Gap 3: Subscription Filters Not Applied Consistently

**Issue:** Advanced subscription filters exist but are only applied during broadcast_to_stream(), not during handler's broadcast_to_subscribers()

**File:** `/backend/api/websocket/manager.py:286-289` (applies filters)
vs. `/backend/api/websocket/handler.py:317-320` (no filter application)

**Impact:**
- Inconsistent behavior if both paths are used
- Could lead to duplicate messages or missed filtering

---

## 13. Relevant File Paths & Line Numbers

### Core Infrastructure Files

| File | Key Lines | Purpose |
|------|-----------|---------|
| `/backend/api/websocket/handler.py` | 17-567 | WebSocket connection management |
| `/backend/api/websocket/manager.py` | 78-562 | Client lifecycle & subscription |
| `/backend/api/websocket/broadcaster.py` | 127-708 | Event publication & broadcasting |
| `/backend/api/websocket/endpoints.py` | 1-478 | REST API endpoints |
| `/backend/api/websocket/__init__.py` | 141-165 | System initialization |
| `/backend/api/websocket/monitoring.py` | 1-100+ | Health monitoring |
| `/backend/api/websocket/subscriptions.py` | 87-100+ | Advanced subscriptions |
| `/backend/api/main.py` | 259-532 | Application lifecycle |
| `/backend/integration_service.py` | 1-1100+ | Vision→Core→Broadcast pipeline |

### Related Files

| File | Purpose |
|------|---------|
| `/backend/config.py` | Configuration management |
| `/backend/core/events/manager.py` | Core event system |
| `/backend/vision/__init__.py` | Vision module |

---

## 14. Architecture Strengths

1. **Modular Design:** Clear separation of concerns (handler, manager, broadcaster)
2. **Resilience:** Circuit breaker pattern prevents cascade failures
3. **Monitoring:** Health monitoring and quality metrics
4. **Flexible Subscriptions:** Advanced filtering and quality levels support
5. **Error Handling:** Comprehensive error handling with retries
6. **Configuration:** Externalized configuration via config_manager

---

## 15. Architecture Weaknesses

1. **Dual Subscription Tracking:** Handler and manager maintain separate subscription lists
2. **Disabled Features:** Frame broadcasting disabled with no alternative
3. **Synchronous-to-Async Bridge:** Fragile event loop bridge for Core callbacks
4. **Limited Event Coverage:** Only state_updated and trajectory_calculated events handled
5. **No Message Queuing:** No persistent message queue for offline clients
6. **No Message History:** No way for clients to catch up on missed messages
7. **Single Circuit Breaker:** Per-integration-service, not per-broadcast-path
8. **No Broadcast Acknowledgment:** No way to verify clients received messages

---

## 16. Recommendations

### Priority 1: Fix Subscription Desynchronization
- Use only `WebSocketManager.stream_subscribers` as source of truth
- Remove subscription tracking from handler or keep in sync
- Add validation to ensure consistency

### Priority 2: Implement Alternative Video Streaming
- Add MJPEG HTTP endpoint
- Add WebRTC peer connection support
- Document why frame broadcast is disabled

### Priority 3: Expand Event Coverage
- Subscribe to all Core module events
- Create broadcast handlers for each event type
- Document which events trigger which broadcasts

### Priority 4: Add Message Persistence
- Implement message buffer per client
- Allow clients to request missed messages
- Add sequence numbers to all messages

### Priority 5: Add Broadcast Acknowledgment
- Require client acknowledgment of important messages
- Implement delivery confirmation for critical events
- Add timeout handling for unacknowledged messages

---

## 17. Testing Endpoints

**Start System:**
```bash
curl -X POST http://localhost:8000/api/v1/websocket/system/start
```

**Get Health:**
```bash
curl http://localhost:8000/api/v1/websocket/health
```

**Get Metrics:**
```bash
curl http://localhost:8000/api/v1/websocket/metrics
```

**Broadcast Test Alert:**
```bash
curl -X POST "http://localhost:8000/api/v1/websocket/broadcast/alert?level=warning&message=Test&code=TEST_001"
```

**Get Connections:**
```bash
curl http://localhost:8000/api/v1/websocket/connections
```

---

## Conclusion

The WebSocket infrastructure is well-architected with proper separation of concerns, resilience patterns, and monitoring. However, there are critical gaps:

1. **Subscription desynchronization** between handler and manager
2. **Disabled frame broadcasting** leaving a gap in the feature set
3. **Limited event coverage** missing many state change events
4. **Fragile async bridge** for Core events to WebSocket broadcasts

These should be addressed to ensure reliable real-time data delivery to frontend clients.
