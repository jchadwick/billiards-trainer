# WebSocket Infrastructure - Quick Reference

## System Components Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│  (/backend/api/main.py - lines 260-533)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        v                v                v
  ┌──────────┐    ┌────────────┐   ┌──────────────┐
  │ Handler  │    │  Manager   │   │ Broadcaster  │
  │ (h.py)   │◄───┤ (mgr.py)   │───┤ (bcast.py)   │
  │ 567 LOC  │    │ 562 LOC    │   │ 708 LOC      │
  └──────────┘    └────────────┘   └──────────────┘
        │
        │ Manages WebSocket connections
        │ Stores subscriptions (DUAL TRACKING!)
        │ Sends messages to clients
        │
  ┌─────┴───────────────────────────────────────────┐
  │                                                 │
  │  Dual Subscription Tracking (ISSUE #1)         │
  │  - Handler: connections[id].subscriptions      │
  │  - Manager: stream_subscribers[type]           │
  │                                                 │
  └─────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│        Integration Service (integration_service.py)  │
│         Vision → Core → Broadcast Pipeline           │
│                                                      │
│  1. Poll Vision.process_frame()                     │
│  2. Update Core.update_state()                      │
│  3. Core emits "state_updated" event [SYNC]        │
│  4. Schedule async broadcast via event_loop        │
│  5. Call broadcaster.broadcast_game_state()        │
│  6. Manager broadcasts to subscribers              │
│  7. Clients receive messages                       │
└──────────────────────────────────────────────────────┘
```

## Data Flow

### Publish Path (Vision → Clients)

```
Vision Detection
    │
    ├─→ DetectionResult {
    │   balls: [Ball],
    │   cue: CueStick,
    │   table: Table
    │ }
    │
    ├─→ Integration._process_detection()
    │
    ├─→ Core.update_state(detection_data)
    │
    ├─→ Core emits "state_updated" [SYNC CALLBACK]
    │
    ├─→ IntegrationService._on_state_updated()
    │   [Schedules async work on event loop]
    │
    ├─→ IntegrationService._on_state_updated_async()
    │
    ├─→ _broadcast_with_retry()
    │   [Circuit breaker + exponential backoff]
    │
    ├─→ Broadcaster.broadcast_game_state()
    │   [Validate data]
    │
    ├─→ Broadcaster._broadcast_to_stream(StreamType.STATE)
    │
    ├─→ Manager.broadcast_to_stream()
    │   [Get subscribers from stream_subscribers[STATE]]
    │
    ├─→ For each subscriber:
    │   Handler.send_to_client(client_id, message)
    │
    ├─→ WebSocket.send_text()
    │
    └─→ Client receives message
```

### Subscribe Path (Client → Subscriptions)

```
Client → WebSocket: {"type": "subscribe", "data": {"streams": ["state"]}}
    │
    ├─→ FastAPI endpoint receives
    │
    ├─→ Handler.handle_message()
    │
    ├─→ Handler._handle_subscribe()
    │
    ├─→ Manager.subscribe_to_stream(client_id, "state")
    │   └─→ stream_subscribers[STATE].add(client_id)
    │   └─→ connections[client_id].add_subscription("state")
    │
    ├─→ Send confirmation
    │
    └─→ Client now subscribed to "state" stream
```

## Key Files & Line Numbers

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Handler | handler.py | 17-567 | WebSocket connections, message routing |
| Manager | manager.py | 78-562 | Subscriptions, broadcast coordination |
| Broadcaster | broadcaster.py | 127-708 | Event publishing, data validation |
| Endpoints | endpoints.py | 1-478 | REST API + WebSocket endpoint |
| Integration | integration_service.py | 1-1100+ | Vision→Core→Broadcast pipeline |
| Init | __init__.py | 141-165 | System startup/shutdown |
| Main | main.py | 259-533 | FastAPI app lifecycle |

## Critical Issues

### Issue 1: Subscription Desynchronization [HIGH PRIORITY]
**Location:** Handler (line 36, 61-67) vs Manager (line 84-86)

Two subscription tracking systems:
- `WebSocketHandler.connections[id].subscriptions` (set of strings)
- `WebSocketManager.stream_subscribers[StreamType]` (set of client_ids)

**Risk:** Out-of-sync subscriptions → missed broadcasts

**Fix:** Use single source of truth (manager) or ensure strict synchronization

### Issue 2: Frame Broadcasting Disabled [MEDIUM PRIORITY]
**Location:** broadcaster.py lines 245-253

Frame (video) broadcasting commented out to prevent browser crashes

**Risk:** No video stream to clients. State/trajectory still work.

**Fix:** Implement MJPEG endpoint or WebRTC instead

### Issue 3: Limited Event Coverage [MEDIUM PRIORITY]
**Location:** integration_service.py lines 343-351

Only 2 events subscribed:
- "state_updated"
- "trajectory_calculated"

Other game events not broadcast (ball pocketed, game started, etc.)

**Risk:** Frontend doesn't get all relevant updates

**Fix:** Subscribe to all Core module events

### Issue 4: Fragile Async Bridge [MEDIUM PRIORITY]
**Location:** integration_service.py lines 1006-1030

Core emits sync callbacks → Must schedule async work on event loop

```python
self._event_loop.call_soon_threadsafe(
    lambda: asyncio.ensure_future(...)
)
```

**Risk:** If event loop dies or unavailable, broadcasts fail silently

**Fix:** Use async event system in Core module

### Issue 5: Single Circuit Breaker [LOW PRIORITY]
**Location:** integration_service.py lines 59-181

One circuit breaker per integration service instance

**Risk:** Multi-worker deployments have inconsistent state

**Fix:** Implement shared circuit breaker (Redis/shared memory)

## Stream Types

```python
StreamType.FRAME       # Video frames (DISABLED)
StreamType.STATE       # Game state (balls, cue, table)
StreamType.TRAJECTORY  # Predicted shot trajectory
StreamType.ALERT       # System alerts and notifications
StreamType.CONFIG      # Configuration changes
```

## Configuration Keys

```
api.websocket.broadcaster.frame_buffer.max_size = 100
api.websocket.broadcaster.compression.threshold_bytes = 1024
api.websocket.broadcaster.compression.level = 6
api.websocket.broadcaster.fps.default_target_fps = 30
api.websocket.manager.max_reconnect_attempts = 5
api.websocket.manager.reconnect_delay = 1.0
integration.target_fps = 30
integration.broadcast_max_retries = 3
integration.circuit_breaker_threshold = 10
integration.circuit_breaker_timeout_sec = 30
```

## Valid Stream Types (from handler.py line 451)

```python
["frame", "state", "trajectory", "alert", "config"]
```

## Rate Limiting

- **Connection rate:** 10 connections per IP per minute (endpoints.py:30)
- **Message rate:** 100 messages per client per minute (handler.py:99)

## Monitoring & Health

**Health Status Levels:**
- EXCELLENT: < 20ms latency
- GOOD: < 50ms latency
- FAIR: < 100ms latency
- POOR: < 200ms latency
- CRITICAL: > 200ms latency

**Metrics Tracked:**
- Latency (min, max, avg, jitter)
- Throughput (messages, bytes)
- Errors (failed sends, timeouts, disconnections)
- Connection stability score

## Testing

```bash
# Start WebSocket system
curl -X POST http://localhost:8000/api/v1/websocket/system/start

# Get health status
curl http://localhost:8000/api/v1/websocket/health

# Get metrics
curl http://localhost:8000/api/v1/websocket/metrics

# Get active connections
curl http://localhost:8000/api/v1/websocket/connections

# Broadcast test alert
curl -X POST "http://localhost:8000/api/v1/websocket/broadcast/alert?level=warning&message=Test&code=TEST"
```

## Missing Functionality

1. **Message History** - Clients can't catch up on missed messages
2. **Message Queuing** - No persistent queue for offline clients
3. **Broadcast Acknowledgment** - No delivery confirmation
4. **Per-Client Throttling** - No per-client rate limiting config
5. **Message Prioritization** - All messages treated equally
6. **Client Grouping** - Can't broadcast to specific client groups (except by user)

## Architecture Patterns Used

1. **Handler Pattern** - WebSocketHandler for connection management
2. **Manager Pattern** - WebSocketManager for lifecycle management
3. **Broadcaster Pattern** - MessageBroadcaster for event publishing
4. **Circuit Breaker Pattern** - Prevents cascade failures
5. **Subscription Pattern** - Stream-based pub/sub
6. **Retry Pattern** - Exponential backoff with max retries
7. **Filter Pattern** - Advanced subscription filtering

## Next Steps

1. **Immediate:** Fix subscription desynchronization
2. **Short-term:** Implement MJPEG video streaming
3. **Medium-term:** Expand event coverage
4. **Long-term:** Add message persistence and acknowledgment
