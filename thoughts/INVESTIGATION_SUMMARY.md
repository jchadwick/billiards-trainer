# Backend Architecture Investigation - Summary

## Investigation Completed: 2024-10-22

### Deliverables Created

Two comprehensive documentation files have been created in `/thoughts/`:

1. **BACKEND_ARCHITECTURE_ANALYSIS.md** (761 lines, 29KB)
   - Executive summary
   - Architecture overview with ASCII diagram
   - Complete component descriptions
   - Detailed message flow (Ball Detection → Broadcasting)
   - Queue and buffering mechanisms
   - Thread and process architecture
   - Performance bottleneck analysis
   - Configuration details
   - File structure and data flow summary

2. **ARCHITECTURE_DIAGRAMS.md** (548 lines, 36KB)
   - 6 detailed ASCII diagrams with annotations:
     1. System component diagram
     2. Message flow (Detection → Broadcasting)
     3. Queueing and buffering flow
     4. Performance timeline (33ms per frame)
     5. Threading and concurrency model
     6. Error handling and resilience flow

---

## Key Findings

### Architecture Summary

**Core Pattern**: 3-tier modular architecture with event-driven messaging

```
Vision Module (Thread-based)
       ↓ [DetectionResult]
Integration Service (Async polling orchestrator - THE CRITICAL HUB)
       ↓ [Core format data]
Core Module (Async event-driven)
       ↓ [Event callbacks]
WebSocket Broadcaster (Async)
       ↓ [JSON messages]
Connected Clients
```

### The Integration Service is the Critical Hub

**File**: `/backend/integration_service.py` (1218 lines)

**Responsibilities**:
- Polls Vision module at 30 FPS (~33ms per frame)
- Converts DetectionResult to Core format
- Updates Core game state
- Subscribes to Core events (state_updated, trajectory_calculated)
- Broadcasts updates to WebSocket clients with retry logic
- Implements circuit breaker pattern for failure resilience

**Main Loop**:
```
while running:
  detection = vision.process_frame()
  core.update_state(converted_detection)
  check_trajectory_calculation()
  sleep(remaining_time_to_maintain_fps)
```

### Message Flow (from detection to client)

1. **Camera Frame**: OpenCV captures at 30 FPS (native)
2. **Vision Processing**: Ball, cue, table detection (30-50ms)
3. **Integration Polling**: Gets latest DetectionResult (2ms)
4. **State Update**: Core.update_state() with validation (5ms)
5. **Event Broadcast**: Async callback triggered (synced via event loop)
6. **WebSocket Send**: MessageBroadcaster.broadcast_game_state() (<5ms)
7. **Client Reception**: Browser receives JSON message
8. **Rendering**: Frontend draws balls, cue, trajectories

**Total Latency**: ~40-60ms (detection bottleneck)

### Queuing Mechanisms

| Queue | Type | Size | Location | Purpose |
|-------|------|------|----------|---------|
| Vision Internal | Deque | 5 | vision.py | Decouple camera thread from detection |
| Integration Loop | Poll | - | integration_service.py | Fixed 30 FPS with sleep |
| Frame Buffer | Deque | 100 | broadcaster.py | Metrics/debugging (not for streaming) |
| Event Bridge | call_soon_threadsafe | - | integration_service.py | Sync→Async callback scheduling |
| Circuit Breaker | State Machine | - | integration_service.py | Prevent cascade failures |
| Async Broadcaster | Queue | 10 | broadcaster.py | Frame queue (currently disabled) |

### Thread & Process Architecture

**Single Uvicorn Worker**:
- Main Event Loop (Async): FastAPI, WebSocket, Integration, Health Monitor
- Vision Daemon Thread: Camera capture (I/O bound, non-blocking to main loop)
- Synchronization: Thread-safe queue for Vision ↔ Integration data passing

**Concurrency Model**:
- Non-blocking async for orchestration
- Separate daemon thread for camera (I/O bound)
- Minimal lock contention (uses thread-safe queue only)

---

## Critical Bottlenecks

### Performance Bottlenecks (Ranked by Impact)

1. **Vision Detection Latency** (30-50ms per frame)
   - GPU acceleration not utilized
   - Ball, cue, table detection runs sequentially
   - Potential 30-50% improvement with parallelization

2. **Fixed 30 FPS Integration Loop**
   - If detection takes 40ms, frame is skipped
   - No adaptive FPS based on actual processing time
   - Potential 10-20% improvement with dynamic FPS

3. **No Integration Loop Pipelining**
   - Poll Vision → Convert → Update Core → Check Trajectory (all sequential)
   - Could pipeline: Vision(N+1) while Core processes (N)
   - Potential 30ms latency reduction

4. **Validation Overhead**
   - Every broadcast validates full ball structure (1-2ms)
   - Could be cached or skipped in high-throughput mode
   - Potential 1-2ms improvement

5. **Trajectory Calculation CPU Cost**
   - 20-100ms per trajectory (currently uses LOW quality)
   - Only triggered when cue detected (good)
   - Potential 30-50% improvement with parallel physics engine

### Architectural Risks

1. **No Backpressure Between Modules**
   - If Core is slow, Integration doesn't slow down
   - State updates may be missed
   - Risk: Undetected state coherency issues

2. **Vision Thread vs Async Mismatch**
   - Vision uses daemon threads, everything else async
   - Thread-safe queue adds overhead
   - Risk: Scaling issues with multiple cameras

3. **Circuit Breaker Without Observability**
   - Opens silently after 10 failures (30s timeout)
   - May drop important updates without alerting
   - Risk: Silent state loss

4. **No Health Checks in Integration Loop**
   - Loop doesn't verify Vision/Core health
   - Stale component could block indefinitely
   - Risk: Hung integration without alerting

5. **Rate Limiting Missing**
   - No rate limiting between modules
   - Trajectory calculation can trigger every frame if cue stays detected
   - Risk: CPU spike when cue is stable

---

## Key Files & Entry Points

### Critical Entry Points

| File | Role | Lines | Key Method |
|------|------|-------|------------|
| `/backend/main.py` | App entry | 89 | `main()` |
| `/backend/api/main.py` | FastAPI app | 631 | `lifespan()`, `create_app()` |
| `/backend/integration_service.py` | **HUB** | 1218 | `_integration_loop()`, `start()` |
| `/backend/vision/__init__.py` | Detection | 1360 | `process_frame()`, `start_capture()` |
| `/backend/core/__init__.py` | State/Physics | ~600 | `update_state()`, `subscribe_to_events()` |
| `/backend/api/websocket/broadcaster.py` | Broadcasting | ~700 | `broadcast_game_state()` |
| `/backend/api/websocket/manager.py` | Connections | ~400 | `register_client()`, `stream_subscribers` |

### Startup Sequence

```
main.py
  └─ uvicorn.run()
     └─ api/main.py::create_app()
        └─ api/main.py::lifespan() startup:
           1. Config.load()
           2. CoreModule()
           3. VisionModule()
           4. IntegrationService()
           5. await integration_service.start()
              └─ vision.start_capture() [daemon thread starts]
              └─ await _integration_loop() [main polling loop]
           6. Register components with health_monitor
```

---

## Configuration System

**Type**: JSON file-based singleton (no external dependencies)

**Key Config Sections**:
```json
{
  "api.server": { "host", "port", "log_level" },
  "vision.camera": { "device_id", "fps", "resolution" },
  "vision.processing": { "target_fps", "enable_threading" },
  "integration": {
    "target_fps": 30,
    "broadcast_max_retries": 3,
    "circuit_breaker_threshold": 10,
    "circuit_breaker_timeout_sec": 30
  }
}
```

**Persistence**: Async background save on every config change (non-blocking)

---

## Resilience Patterns Implemented

1. **Circuit Breaker Pattern**
   - Prevents cascade failures when clients disconnect
   - States: CLOSED (normal) → OPEN (blocked) → HALF-OPEN (testing)
   - Default: 10 consecutive failures trigger, 30s timeout

2. **Exponential Backoff Retry**
   - 3 retries with delays: 0.1s, 0.2s, 0.4s, 0.8s
   - Error classification: VALIDATION (no retry), TRANSIENT (retry), UNKNOWN (retry cautious)

3. **Validation Before Broadcast**
   - Check ball structure, position format, scale metadata
   - Log warnings on invalid data, drop message

4. **Integration Loop Error Handling**
   - Catches exceptions, increments error counter
   - Continues running (resilient to individual frame failures)

5. **Graceful Shutdown**
   - Cancels integration loop task
   - Stops vision capture
   - Shuts down WebSocket system
   - Clean resource cleanup

---

## Performance Characteristics

### Typical Latency Breakdown (per frame)

```
Vision Detection:           30-50ms  (dominant bottleneck)
Integration Polling:         2ms     (negligible)
Core State Update:           5ms     (with validation)
WebSocket Broadcast:        <5ms     (async, non-blocking)
─────────────────────────────────
Total Frame Latency:       40-62ms   (at 30 FPS)
Target Frame Time:         33ms      (30 FPS)
Status:                    EXCEEDS target at 30 FPS
```

### Trajectory Calculation

- **Duration**: 20-100ms depending on quality
- **Triggered**: Only when cue detected and balls stationary
- **Quality**: LOW (max_collision_depth=5) for real-time
- **Frequency**: Not rate-limited (can trigger every frame if cue stable)

### WebSocket Broadcasting

- **Messages/Frame**: 1-2 (game_state + optional trajectory)
- **JSON Size**: ~0.5-2KB per message (depending on ball count)
- **Compression**: Optional zlib if >1024 bytes
- **Per-Client Overhead**: Minimal with async handling
- **Bottleneck Risk**: >20 clients on slow network

---

## Obvious Improvements (Quick Wins)

1. **Parallel Detection** (30-50% gain)
   - Use asyncio.gather() or threading.Pool for ball/cue/table detection
   - Estimated gain: 15-25ms per frame

2. **Adaptive FPS** (10-20% gain)
   - Track actual detection time
   - Dynamically adjust FPS to 20-30 based on load
   - Estimated gain: 5-10ms per frame

3. **Integration Pipelining** (30ms gain)
   - Process Vision(N+1) while Core handles (N)
   - Requires queue-based communication instead of polling
   - Estimated gain: 30ms latency reduction

4. **Trajectory Rate Limiting** (CPU savings)
   - Throttle trajectory calculation to 1x per second (not every frame)
   - Already partially done with state checks

5. **Broadcast Validation Caching** (1-2ms gain)
   - Cache validation results if data structure hasn't changed
   - Negligible impact but cleaner code

---

## Recommended Reading Order

For understanding the architecture:

1. **Start Here**: `INVESTIGATION_SUMMARY.md` (this file)
2. **Visual Overview**: `ARCHITECTURE_DIAGRAMS.md` (diagrams 1-2)
3. **Deep Dive**: `BACKEND_ARCHITECTURE_ANALYSIS.md` (full details)
4. **Code Review**:
   - `/backend/integration_service.py` - Main orchestrator
   - `/backend/api/main.py` - App startup
   - `/backend/vision/__init__.py` - Vision module
   - `/backend/core/__init__.py` - Core module
   - `/backend/api/websocket/broadcaster.py` - Broadcasting

---

## Questions Answered

### How is the vision system structured?

Multi-layered with entry points:
- `VisionModule.start_capture()` - Starts daemon thread for camera
- `VisionModule.process_frame()` - Non-blocking detection result getter
- Sub-modules: detection/, tracking/, calibration/, preprocessing/

### What is the message flow from detection to publishing?

1. Vision (thread) → DetectionResult
2. Integration (polling) → Convert to Core format
3. Core (async) → Update state, emit event
4. Integration (callback) → Call broadcaster
5. Broadcaster (async) → JSON to WebSocket clients

### What queuing mechanisms exist?

6 queuing mechanisms identified:
1. Vision internal queue (5 frames, deque)
2. Integration polling loop (fixed 30 FPS)
3. Frame buffer (100 frames, metrics only)
4. Event loop call_soon_threadsafe (sync→async bridge)
5. Circuit breaker state machine (failure buffering)
6. Async broadcaster queue (disabled)

### What is the thread/process architecture?

Single Uvicorn worker with:
- 1 main asyncio event loop (HTTP, WebSocket, Integration)
- 1 daemon thread for Vision capture
- Thread-safe queue for Vision ↔ Integration communication
- Minimal lock contention

### What are the bottlenecks?

Top 3:
1. Vision detection latency (30-50ms) - GPU not utilized
2. Fixed 30 FPS integration loop - No adaptive FPS
3. No integration pipelining - Sequential processing

---

Generated: 2024-10-22
Investigation depth: Complete system architecture, message flows, bottleneck analysis
