# Ball Detection Events - Complete Analysis Index

This directory contains a comprehensive analysis of the ball detection event flow in the billiards-trainer backend.

## Quick Start

**Start here:** `/thoughts/EXECUTIVE_SUMMARY.md`
- Overview of complete flow
- Critical file locations
- Quick fix checklist for debugging

## Detailed Analysis

**Full technical details:** `/thoughts/ball_detection_event_flow_report.md`
- Complete 12-section analysis
- File-by-file breakdown
- Configuration reference
- Thread safety analysis

## Understanding the Flow

### The Complete Data Path

```
Vision Module (detects balls)
    ↓ detection/balls.py:522-588 detect_balls()
Ball Detection → Ball[] objects
    ↓
Integration Service polling
    ↓ integration_service.py:355-393 _integration_loop()
Format Conversion → Core format
    ↓
Core Module validation
    ↓ core/game_state.py update_state()
State Update → emits event
    ↓
Event Manager routing
    ↓ core/events/manager.py:286-336 emit_event()
Event Distribution → subscribers
    ↓
Integration Service handler
    ↓ integration_service.py:1034-1140 _on_state_updated_async()
WebSocket Broadcast → formatted data
    ↓
WebSocket Broadcaster
    ↓ api/websocket/broadcaster.py:265-336 broadcast_game_state()
Client Distribution → validated message
    ↓
Connected WebSocket Clients
    ↓ ws://localhost:8000/ws
Browser Application → receives update
```

## Critical Components by Purpose

### Detection
- **File:** `backend/vision/detection/balls.py`
- **Key Method:** `detect_balls()` (lines 522-588)
- **Output:** List of Ball objects with position, radius, confidence
- **Configuration:** BallDetectionConfig (lines 42-200)

### Data Bridge
- **File:** `backend/integration_service.py`
- **Key Methods:**
  - `_integration_loop()` (lines 355-393) - polling at 30 FPS
  - `_process_detection()` (lines 409-425) - format conversion
  - `_on_state_updated_async()` (lines 1034-1140) - event handling
- **Responsibility:** Vision → Core → WebSocket

### Event System
- **Manager:** `backend/core/events/manager.py` (lines 201-724)
- **Handlers:** `backend/core/events/handlers.py` (lines 16-461)
- **Event:** "state_updated" - fired when game state changes
- **Subscribers:** Integration service callbacks

### WebSocket Broadcasting
- **Broadcaster:** `backend/api/websocket/broadcaster.py` (lines 127-709)
- **Key Method:** `broadcast_game_state()` (lines 265-336)
- **Validation:** Lines 283-324 (balls, cue, table validation)
- **Routing:** Via websocket_manager to connected clients

## Key Data Structures

### Ball Detection Output
```python
Ball {
    position: Tuple[float, float]      # (x, y) in pixels
    radius: float                       # pixels
    ball_type: BallType                # CUE, EIGHT, OTHER
    confidence: float                  # 0-1 score
    velocity: Tuple[float, float]     # (vx, vy)
    is_moving: bool
    number: Optional[int]              # None (simplified)
}
```

### WebSocket Message Format
```json
{
  "type": "state",
  "data": {
    "balls": [
      {
        "position": [x, y],
        "radius": r,
        "confidence": c,
        "is_cue_ball": bool,
        "velocity": [vx, vy]
      }
    ],
    "cue": {...},
    "table": {...},
    "timestamp": "ISO8601",
    "sequence": number,
    "ball_count": number
  }
}
```

## File Reference

### Vision Module (Detection)
- `backend/vision/detection/balls.py` - Ball detection algorithm
- `backend/vision/__init__.py` - Vision module interface
- `backend/vision/models.py` - Ball model definition

### Core Module (State Management)
- `backend/core/game_state.py` - Game state management
- `backend/core/events/manager.py` - Event system core
- `backend/core/events/handlers.py` - Event handlers
- `backend/core/models.py` - GameState model

### Integration Layer
- `backend/integration_service.py` - Vision→Core→WebSocket bridge

### WebSocket Layer
- `backend/api/websocket/broadcaster.py` - Broadcasting logic
- `backend/api/websocket/manager.py` - Connection management
- `backend/api/websocket/handler.py` - Message handling
- `backend/api/websocket/endpoints.py` - /ws endpoint
- `backend/api/websocket/schemas.py` - Protocol schemas

## Configuration Reference

### Integration Service
Located in `config/default.json` or environment:

```json
{
  "integration": {
    "target_fps": 30,                      # Processing frame rate
    "log_interval_frames": 300,            # Logging frequency
    "error_retry_delay_sec": 0.1,         # Error retry delay
    "broadcast_max_retries": 3,            # Broadcast retry count
    "circuit_breaker_threshold": 10,       # Failure threshold
    "circuit_breaker_timeout_sec": 30.0   # Recovery timeout
  }
}
```

### Ball Detection
```json
{
  "vision": {
    "ball_detection": {
      "detection_method": "combined",      # Method: combined/hough/contour/blob
      "min_radius": 15,                    # Minimum size (pixels)
      "max_radius": 26,                    # Maximum size (pixels)
      "min_confidence": 0.4,               # Confidence threshold
      "hough_circles": {...},              # Hough parameters
      "quality_filters": {...}             # Quality parameters
    }
  }
}
```

## Validation Pipeline

Data passes through multiple validation layers:

### 1. Vision Module (`balls.py`)
- Circle geometry validation
- Radius bounds (15-26px)
- Confidence threshold (>0.4)
- Shadow and brightness filtering
- Pocket proximity exclusion
- Overlap detection

### 2. Integration Service (`integration_service.py`)
- Format conversion validation
- Data type checking
- Physics validation
- Position bounds

### 3. WebSocket Broadcaster (`broadcaster.py`)
- Balls must be list ✓
- Each ball must be dict ✓
- Position required as [x, y] ✓
- Cue must be dict or None ✓
- Table must be dict or None ✓

### 4. WebSocket Handler
- Connection validation
- Message format validation
- Subscription management

## Debugging Guide

### Check Detection
```bash
# See current detections
curl http://localhost:8000/api/v1/vision/detect
```

### Monitor Events
Add logging to `integration_service.py`:
```python
# Line 1064 (in _on_state_updated_async)
logger.info(f"State update: {len(balls)} balls")
```

### Test WebSocket
```bash
# Terminal 1
wscat -c ws://localhost:8000/ws

# Terminal 2 (in wscat)
{"type": "subscribe", "streams": ["state"]}
```

### Check Circuit Breaker
```python
# In integration_service logs
circuit_breaker.get_status()
```

## Known Gaps and Issues

### 1. Frame Broadcasting Disabled
- **Issue:** Video frames NOT sent over WebSocket
- **Reason:** Prevent browser crashes from large base64
- **Workaround:** Use `/api/v1/stream/video` HTTP endpoint

### 2. Thread Safety
- Integration runs in async loop
- Core may run in different thread
- Uses `call_soon_threadsafe()` properly ✓

### 3. Data Format Conversions
- Multiple format conversions occur
- Position: Vector2D → dict → [x, y] list
- All conversions validated ✓

### 4. Circuit Breaker
- Opens after 10 failures
- Blocks for 30 seconds
- Prevents cascade failures ✓

## Performance Metrics

- **Detection Rate:** 30 FPS (configurable)
- **Broadcast Rate:** Limited per client
- **Compression:** DEFLATE for messages >1KB
- **Frame Buffer:** Max 100 frames
- **Memory:** ~5MB per 100 frames

## Quick Fix Checklist

If ball events not reaching WebSocket:

- [ ] Vision module detecting balls (check logs)
- [ ] Integration service started (check main.py)
- [ ] WebSocket endpoint accessible (check /api/health)
- [ ] Client subscribed to "state" stream
- [ ] WebSocket connection open (check browser dev tools)
- [ ] No validation failures (check logs for "skipped")
- [ ] Circuit breaker not open (check logs)
- [ ] Position format [x,y] not {x,y}
- [ ] Event loop running (check integration service)
- [ ] No connectivity issues (test with wscat)

## Additional Documentation

- Event system deep dive: See `EXECUTIVE_SUMMARY.md` sections 4-5
- WebSocket details: See `websocket_infrastructure_analysis.md`
- Vision architecture: See `vision_architecture_research.md`
- Configuration: See `config/default.json`

---

**Generated:** October 21, 2024
**Status:** Comprehensive analysis of all ball detection event flows
**Coverage:** Vision detection → Event system → WebSocket broadcasting
