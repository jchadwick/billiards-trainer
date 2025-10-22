# WebSocket Infrastructure Analysis

This directory contains detailed analysis of the backend WebSocket infrastructure.

## Documents

### 1. websocket_infrastructure_analysis.md (25 KB)
Comprehensive 17-section technical analysis covering:
- WebSocket server initialization and startup flow
- Connection handler architecture and methods
- WebSocket manager for client lifecycle
- Message broadcaster and event publication
- Event routing and subscription system
- Integration service (Vision→Core→Broadcast pipeline)
- REST API endpoints for WebSocket management
- Health monitoring and metrics collection
- Configuration management
- **Critical Issues & Bottlenecks** (5 major issues identified)
- Message flow diagrams
- Missing connection analysis with specific code references
- Relevant file paths and line numbers
- Architecture strengths and weaknesses
- 16 recommendations prioritized by impact

### 2. websocket_summary.md (9.9 KB)
Quick reference guide including:
- System component overview with ASCII diagrams
- Data flow documentation
- Key files and line numbers table
- 5 critical issues prioritized by severity
- Stream types and configuration keys
- Rate limiting specifications
- Health monitoring levels
- Testing endpoints
- Missing functionality checklist
- Architecture patterns used
- Next steps roadmap

## Key Findings

### Critical Issues Identified

#### Issue 1: Subscription Desynchronization [HIGH]
Two separate subscription tracking systems that can get out of sync:
- `WebSocketHandler.connections[client_id].subscriptions`
- `WebSocketManager.stream_subscribers[StreamType]`

**Impact:** Broadcasts may fail to reach subscribed clients

#### Issue 2: Frame Broadcasting Disabled [MEDIUM]
Video frame broadcasting to WebSocket is completely disabled (broadcaster.py:245-253)

**Reason:** Large base64 images crash browsers
**Impact:** No video stream to clients

#### Issue 3: Limited Event Coverage [MEDIUM]
Only 2 Core events subscribed (state_updated, trajectory_calculated)

**Impact:** Frontend doesn't receive all game state changes

#### Issue 4: Fragile Async Bridge [MEDIUM]
Core events are synchronous but WebSocket broadcasting requires async

**Risk:** Silent failures if event loop dies

#### Issue 5: Single Circuit Breaker [LOW]
Not synchronized across multi-worker deployments

## Architecture Overview

```
Vision → Integration Service → Core Module
                                    ↓
                            Event System [SYNC]
                                    ↓
                        Async Event Handler [EVENT LOOP]
                                    ↓
                        Message Broadcaster
                                    ↓
                        WebSocket Manager
                                    ↓
                        WebSocket Handler
                                    ↓
                        WebSocket Connections
                                    ↓
                        Connected Clients
```

## File Reference

### Core WebSocket Files (9 files, ~3000 LOC)
- `/backend/api/websocket/handler.py` (567 lines) - Connection management
- `/backend/api/websocket/manager.py` (562 lines) - Subscription management
- `/backend/api/websocket/broadcaster.py` (708 lines) - Event publishing
- `/backend/api/websocket/endpoints.py` (478 lines) - REST API endpoints
- `/backend/api/websocket/subscriptions.py` - Advanced filtering
- `/backend/api/websocket/monitoring.py` - Health monitoring
- `/backend/api/websocket/schemas.py` - Message schemas
- `/backend/api/websocket/__init__.py` (165 lines) - System init/shutdown

### Integration Files
- `/backend/api/main.py` (lines 260-533) - FastAPI app lifecycle
- `/backend/integration_service.py` (1100+ lines) - Vision→Core→Broadcast

### Supporting Files
- `/backend/config.py` - Configuration management
- `/backend/core/events/manager.py` - Core event system
- `/backend/core/game_state.py` - Game state model

## Stream Types

1. **FRAME** - Video frames (CURRENTLY DISABLED)
2. **STATE** - Game state (balls, cue, table positions)
3. **TRAJECTORY** - Predicted shot trajectories
4. **ALERT** - System alerts and notifications
5. **CONFIG** - Configuration changes

## How to Use This Analysis

1. **For quick overview:** Start with `websocket_summary.md`
2. **For implementation:** Reference specific line numbers in the full analysis
3. **For debugging:** Check "Missing Connection Analysis" section for gaps
4. **For improvements:** Review "Recommendations" section prioritized by impact

## Next Steps

### Immediate (Priority 1)
- Fix subscription desynchronization between handler and manager

### Short-term (Priority 2)
- Implement MJPEG video streaming endpoint
- Add missing event subscriptions for all Core events

### Medium-term (Priority 3)
- Fix async event loop bridge fragility
- Add message persistence for offline clients

### Long-term (Priority 4)
- Add broadcast acknowledgment system
- Implement shared circuit breaker for multi-worker deployments

## Testing

```bash
# Start WebSocket system
curl -X POST http://localhost:8000/api/v1/websocket/system/start

# Get health
curl http://localhost:8000/api/v1/websocket/health

# Get metrics
curl http://localhost:8000/api/v1/websocket/metrics

# Get connections
curl http://localhost:8000/api/v1/websocket/connections

# Broadcast test alert
curl -X POST "http://localhost:8000/api/v1/websocket/broadcast/alert?level=warning&message=Test&code=TEST"
```

## Strengths of Current Implementation

1. Modular three-layer architecture (Handler, Manager, Broadcaster)
2. Comprehensive health monitoring and metrics
3. Advanced subscription filtering and quality management
4. Circuit breaker pattern for resilience
5. Retry logic with exponential backoff
6. Concurrent message delivery with error handling
7. Externalized configuration management

## Weaknesses Requiring Fixes

1. Dual subscription tracking (sync issue)
2. Disabled video streaming (no alternative)
3. Limited event coverage (missing game events)
4. Fragile async/sync bridge
5. No message persistence
6. No broadcast acknowledgment
7. Single non-shared circuit breaker

---

Analysis generated: 2025-10-21
Framework: FastAPI + Starlette WebSocket
Python Version: 3.9+
