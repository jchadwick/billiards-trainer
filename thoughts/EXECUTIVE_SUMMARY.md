# Ball Detection Events - Executive Summary

## Complete Flow Overview

Ball detection events flow through this architecture:

```
Vision Module (detects balls)
    ↓
Integration Service (polls & converts)
    ↓
Core Module (validates & updates state)
    ↓
Event Manager (emits state_updated event)
    ↓
Integration Service (receives event)
    ↓
Message Broadcaster (formats for WebSocket)
    ↓
WebSocket Manager (routes to clients)
    ↓
Browser Clients (receives state update)
```

---

## Critical File Locations

| Component | File Path | Key Method | Line |
|-----------|-----------|-----------|------|
| **Ball Detection** | `backend/vision/detection/balls.py` | `detect_balls()` | 522-588 |
| **Detection Results** | `backend/vision/__init__.py` | `process_frame()` | N/A |
| **Data Flow Bridge** | `backend/integration_service.py` | `_integration_loop()` | 355-393 |
| **State Update Handler** | `backend/integration_service.py` | `_on_state_updated_async()` | 1034-1140 |
| **WebSocket Broadcast** | `backend/api/websocket/broadcaster.py` | `broadcast_game_state()` | 265-336 |
| **Event System** | `backend/core/events/manager.py` | `emit_event()` | 286-336 |
| **Event Handlers** | `backend/core/events/handlers.py` | `handle_state_change()` | 88-131 |
| **WebSocket Endpoint** | `backend/api/websocket/endpoints.py` | `websocket_endpoint()` | 78-175 |

---

## Key Data Structures

### Ball Detection Output
```python
Ball {
    position: (x: float, y: float)           # pixels
    radius: float                             # pixels
    ball_type: BallType                      # CUE, EIGHT, OTHER
    confidence: float                        # 0-1
    velocity: (x: float, y: float)          # pixels/frame
    is_moving: bool
    number: Optional[int]                   # None (simplified)
}
```

### State Update Event Data
```python
event_data: {
    "balls": [                              # List of ball dicts
        {
            "position": [x, y],             # MUST be [x, y] list format
            "velocity": [vx, vy],           # Optional
            "radius": r,                    # Float
            "confidence": c,                # 0-1
            "is_cue_ball": bool,           # Boolean
            # ... other fields
        }
    ],
    "cue": dict or None,                   # Cue stick data
    "table": dict or None,                 # Table data
    "timestamp": float,                    # Unix timestamp
    "frame_number": int
}
```

### WebSocket Message to Clients
```python
{
    "type": "state",
    "data": {
        "balls": [...],                    # Validated list of ball dicts
        "cue": {...},
        "table": {...},
        "timestamp": ISO8601,
        "sequence": int,                   # Message sequence number
        "ball_count": int
    }
}
```

---

## Entry Points for Ball Events

### 1. Ball Detection Triggered
- **When:** Every frame at ~30 FPS
- **File:** `backend/vision/detection/balls.py`
- **Method:** `BallDetector.detect_balls()`
- **Output:** List of `Ball` objects

### 2. Detection Processed
- **When:** Every frame (if detection available)
- **File:** `backend/integration_service.py`
- **Method:** `_process_detection()`
- **Flow:** Vision format → Core format → Core state update

### 3. Core State Updated
- **When:** After Core validates detection data
- **File:** `backend/core/game_state.py`
- **Method:** `update_state()`
- **Result:** Emits "state_updated" event

### 4. Event Emitted
- **When:** Core state changes
- **File:** `backend/core/events/manager.py`
- **Method:** `emit_event("state_updated", event_data)`
- **Subscribers:** Integration service callbacks

### 5. WebSocket Broadcast
- **When:** Integration service receives state event
- **File:** `backend/integration_service.py`
- **Method:** `_on_state_updated_async()`
- **Result:** Calls `broadcaster.broadcast_game_state()`

### 6. Client Receives Update
- **When:** WebSocket connection established + subscribed to "state"
- **Protocol:** WebSocket message with type="state"
- **Connection:** `ws://localhost:8000/ws`

---

## Configuration Values

### Integration Service (FPS and Timing)
```json
{
  "integration": {
    "target_fps": 30,                    # Processing FPS
    "log_interval_frames": 300,          # Log every 300 frames
    "error_retry_delay_sec": 0.1,        # Retry delay on error
    "broadcast_max_retries": 3,          # Retry count
    "circuit_breaker_threshold": 10,     # Failures before open
    "circuit_breaker_timeout_sec": 30.0  # Recovery timeout
  }
}
```

### Ball Detection (Quality)
```json
{
  "vision": {
    "ball_detection": {
      "detection_method": "combined",     # Hough + Contour + Blob
      "min_radius": 15,                   # Min ball size (pixels)
      "max_radius": 26,                   # Max ball size (pixels)
      "min_confidence": 0.4               # Confidence threshold
    }
  }
}
```

---

## Validation Checks in the Pipeline

### 1. Vision Module (`balls.py`)
- Circle geometry validation
- Radius constraints (15-26px)
- Brightness and shadow filtering
- Pocket proximity check
- Overlap detection and resolution

### 2. Integration Service (`integration_service.py`)
- Format conversion validation
- Data type checking
- Physics validation (velocity bounds)
- Position bounds checking

### 3. WebSocket Broadcaster (`broadcaster.py`)
- Balls must be list (line 284)
- Each ball must be dict (line 298)
- Position required and must be [x,y] (lines 305-324)
- Cue must be dict or None (line 1088)
- Table must be dict or None (line 1097)

### 4. WebSocket Handler (`handler.py`)
- Connection validation
- Message format validation
- Client subscription management

---

## Known Issues and Gaps

### 1. Frame Broadcasting Disabled
- **Issue:** Video frames NOT sent over WebSocket
- **Reason:** Prevents browser crashes from large base64 images
- **Workaround:** Use HTTP MJPEG endpoint `/api/v1/stream/video`

### 2. Thread Safety
- Integration service runs in async loop
- Core module may run in different thread
- Uses `call_soon_threadsafe()` to schedule work
- **Status:** Properly implemented

### 3. Data Format Conversions
- Ball position converted from (x,y) to [x,y] at line 1112
- Vector2D -> dict -> list conversion chain
- **Status:** Working with validation

### 4. Circuit Breaker
- Activates after 10 consecutive broadcast failures
- Blocks broadcasts for 30 seconds
- Prevents cascade failures when clients disconnect
- **Status:** Properly implemented

---

## How to Debug Ball Events Not Reaching WebSocket

### Trace Points

1. **Check Detection** (vision/detection/balls.py)
   ```python
   logger.info(f"Detected {len(final_balls)} balls")
   ```

2. **Check Core Update** (core/game_state.py)
   ```python
   logger.info(f"State updated with {len(balls)} balls")
   ```

3. **Check Event Emission** (core/events/manager.py)
   ```python
   logger.debug(f"Emitting state_updated with {len(balls)} balls")
   ```

4. **Check Integration Callback** (integration_service.py:1064)
   ```python
   logger.info(f"Received state update: {len(balls)} balls")
   ```

5. **Check Broadcast** (api/websocket/broadcaster.py:283)
   ```python
   logger.info(f"Broadcasting {len(balls)} balls to WebSocket")
   ```

6. **Check WebSocket Delivery** (api/websocket/handler.py)
   ```python
   logger.info(f"Sending to {len(active_clients)} clients")
   ```

### Testing WebSocket Connection
```bash
# Terminal 1: Connect to WebSocket
wscat -c ws://localhost:8000/ws

# Terminal 2: Subscribe to state stream
{"type": "subscribe", "streams": ["state"]}

# Should receive messages like:
{"type": "state", "data": {"balls": [...], ...}}
```

---

## Performance Considerations

- **Detection Rate:** 30 FPS (configurable)
- **Broadcast Rate:** Limited per client via FPS limiter
- **Compression:** DEFLATE compression for large messages
- **Frame Buffering:** 100 frames max (configurable)
- **Memory:** ~5 MB per 100 frames of state history

---

## Quick Fix Checklist

If ball events aren't reaching WebSocket:

- [ ] Check vision module is running and detecting balls
- [ ] Check integration service is started
- [ ] Check WebSocket endpoint is accessible
- [ ] Check client is subscribed to "state" stream
- [ ] Check browser WebSocket connection is open
- [ ] Check no JavaScript errors in browser console
- [ ] Check circuit breaker isn't open (check logs)
- [ ] Check position format is [x, y] not {x, y}
- [ ] Check validation isn't failing (check logs for "Event will be skipped")
- [ ] Monitor integration service logs for state update events

---

## File Reference Map

```
BACKEND ARCHITECTURE
├── vision/
│   ├── detection/
│   │   └── balls.py              <- Ball detection algorithm
│   └── __init__.py               <- Vision module interface
├── core/
│   ├── game_state.py             <- Game state management
│   ├── events/
│   │   ├── manager.py            <- Event system core
│   │   └── handlers.py           <- Event handlers
│   └── models.py                 <- Data models (Ball, GameState, etc)
├── api/
│   ├── websocket/
│   │   ├── broadcaster.py        <- WebSocket broadcast logic
│   │   ├── handler.py            <- WebSocket message handling
│   │   ├── manager.py            <- Connection management
│   │   └── endpoints.py          <- /ws endpoint
│   └── main.py                   <- FastAPI setup
└── integration_service.py        <- Vision→Core→WebSocket bridge
```

---

## Related Documentation

- Event system: `backend/core/events/manager.py` (docstrings)
- WebSocket protocol: `backend/api/websocket/schemas.py`
- Configuration: `backend/config.py` or `config/default.json`
- Models: `backend/core/models.py` and `backend/vision/models.py`
