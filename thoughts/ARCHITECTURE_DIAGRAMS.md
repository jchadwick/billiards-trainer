# Billiards Trainer - Architecture Diagrams

## 1. System Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BILLIARDS TRAINER BACKEND                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         FastAPI Server (Uvicorn)                        │ │
│  │                         Port 8000, Multi-worker                         │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │                  Integration Service (Main Hub)                   │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │ _integration_loop() - 30 FPS, ~33ms per iteration           │ │ │ │
│  │  │  │  1. Poll Vision.process_frame()                            │ │ │ │
│  │  │  │  2. Convert DetectionResult to Core format                 │ │ │ │
│  │  │  │  3. Call Core.update_state()                               │ │ │ │
│  │  │  │  4. Check if trajectory calculation needed                 │ │ │ │
│  │  │  │  5. Sleep to maintain FPS                                  │ │ │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │ │ │
│  │  │                                                                   │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │ Event Subscriptions & Broadcasting                           │ │ │ │
│  │  │  │  • "state_updated" → _on_state_updated_async()            │ │ │ │
│  │  │  │    → broadcast_game_state() via WebSocket                 │ │ │ │
│  │  │  │  • "trajectory_calculated" → _on_trajectory_calculated()  │ │ │ │
│  │  │  │    → broadcast_trajectory() via WebSocket                 │ │ │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │ │ │
│  │  │                                                                   │ │ │
│  │  │  ┌─────────────────────────────────────────────────────────────┐ │ │ │
│  │  │  │ Resilience Patterns                                          │ │ │ │
│  │  │  │  • Retry logic with exponential backoff (3 retries)        │ │ │ │
│  │  │  │  • Circuit breaker (10 failures = open, 30s timeout)      │ │ │ │
│  │  │  └─────────────────────────────────────────────────────────────┘ │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │           ↑                      ↓                   ↓                 │ │
│  │    ┌──────┴────────┐     ┌──────┴───────┐    ┌────┴───────┐          │ │
│  │    ↓               ↓     ↓              ↓    ↓            ↓           │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │           Vision Module           │ Core Module │  WebSocket    │ │ │
│  │  │  ┌─────────────────────┐         ├──────────────┤  System      │ │ │
│  │  │  │ start_capture()     │ ┌──────┐│update_state()│ ┌──────────┐ │ │
│  │  │  │ - Daemon thread     │ │      │├──────────────┤ │Broadcast │ │ │
│  │  │  │ - OpenCV camera     │ │      ││emit_event()  │ │game_state│ │ │
│  │  │  │ - Frame queue (5)   │ │      │└──────────────┘ └──────────┘ │ │
│  │  │  └─────────────────────┘ │      │ ┌──────────────┐ ┌──────────┐ │ │
│  │  │  ┌─────────────────────┐ │      │ │Physics Engine │ │Broadcast │ │ │
│  │  │  │ process_frame()     │ │      │ │Trajectory Cal │ │trajectory│ │ │
│  │  │  │ - Ball detection    │ │      │ │Collision Det  │ └──────────┘ │ │
│  │  │  │ - Cue detection     │ │      │ └──────────────┘ ┌──────────┐ │ │
│  │  │  │ - Table tracking    │ │      │ ┌──────────────┐ │Manager   │ │ │
│  │  │  │ - Kalman tracking   │ │      │ │Event Manager │ │Handler   │ │ │
│  │  │  └─────────────────────┘ │      │ │Shot Analyzer │ │Frame Buf │ │ │
│  │  │  ┌─────────────────────┐ │      │ │Outcome Pred  │ └──────────┘ │ │
│  │  │  │ DetectionResult:    │ │      │ └──────────────┘              │ │
│  │  │  │ {                   │ │      │                               │ │
│  │  │  │   balls: Ball[]     │ │      │                               │ │
│  │  │  │   cue: CueStick     │─┘      │                               │ │
│  │  │  │   table: Table      │        │                               │ │
│  │  │  │   timestamp         │        │                               │ │
│  │  │  │ }                   │        │                               │ │
│  │  │  └─────────────────────┘        │                               │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ REST API Routes (/api/v1/*)                                       │ │ │
│  │  │ • health, vision, calibration, game, config, debug, diagnostics  │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │ │
│  │  │ WebSocket Endpoint (/api/v1/ws)                                   │ │ │
│  │  │ Connected clients receive:                                        │ │ │
│  │  │ • STATE messages (game state updates)                            │ │ │
│  │  │ • TRAJECTORY messages (prediction lines & collisions)            │ │ │
│  │  │ • ALERT messages (system events)                                 │ │ │
│  │  │ • CONFIG messages (settings changes)                             │ │ │
│  │  └──────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                    ↓                                ↓                        │
│           OpenCV Camera              WebSocket Clients (Browser)            │
│           Hardware Access            TCP/IP Network                         │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Message Flow: Detection → Broadcasting

```
┌─────────────────┐
│  OpenCV Camera  │ (30 FPS native)
└────────┬────────┘
         │
         │ Frame
         ↓
┌──────────────────────┐
│ Vision Module        │
│ (Daemon Thread)      │
├──────────────────────┤
│ • Capture frames     │
│ • Detect balls/cue   │ (30-50ms per frame)
│ • Track objects      │
│ • Internal queue(5)  │
└────────┬─────────────┘
         │
         │ DetectionResult
         │ {balls, cue, table, timestamp}
         ↓
   ┌─────────────────────────────────────────────────────────┐
   │ Integration Service (Main Loop - 30 FPS)               │
   │ Frame N iteration: ~33ms target                         │
   ├─────────────────────────────────────────────────────────┤
   │                                                         │
   │ 1. get_vision_detection()                              │
   │    └─→ vision.process_frame()                          │
   │        └─→ returns latest DetectionResult              │
   │                                                         │
   │ 2. convert to Core format                              │
   │    └─→ _convert_detection_to_core_format()             │
   │                                                         │
   │ 3. Update Core state (5ms)                             │
   │    └─→ core.update_state(detection_data)               │
   │        ├─→ Validate input                              │
   │        ├─→ Update GameState fields                     │
   │        └─→ Emit "state_updated" event                  │
   │                                                         │
   │ 4. Event callback (Sync → Async bridge)                │
   │    └─→ _on_state_updated_async()                       │
   │        └─→ _broadcast_with_retry()                     │
   │            ├─→ Check circuit breaker                   │
   │            ├─→ Retry up to 3 times (exponential delay) │
   │            └─→ broadcast_game_state()                  │
   │                                                         │
   │ 5. Check if trajectory needed                          │
   │    └─→ _check_trajectory_calculation()                 │
   │        └─→ if cue detected:                            │
   │            └─→ trajectory_calculator.predict() (20-100ms)
   │                └─→ _emit_multiball_trajectory()        │
   │                    └─→ broadcast_trajectory()          │
   │                                                         │
   │ 6. Sleep to maintain FPS                               │
   │    └─→ await asyncio.sleep(remaining_time)             │
   │                                                         │
   └────────┬──────────────────────────────────────────────┘
            │
            │ broadcast_game_state(balls, cue, table)
            ↓
   ┌──────────────────────────────────────────┐
   │ Message Broadcaster                      │
   ├──────────────────────────────────────────┤
   │ 1. Validate balls structure              │
   │    • Check list type                     │
   │    • Verify position dict format         │
   │    • Confirm scale metadata present      │
   │                                          │
   │ 2. Build state message                   │
   │    {                                     │
   │      "type": "state",                    │
   │      "balls": [...],                     │
   │      "cue": {...},                       │
   │      "table": {...},                     │
   │      "timestamp": "ISO8601",             │
   │      "sequence": 12345                   │
   │    }                                     │
   │                                          │
   │ 3. JSON serialize                        │
   │                                          │
   │ 4. Optional compress (zlib)              │
   │    if size > 1024 bytes                  │
   │                                          │
   │ 5. Broadcast to all subscribers          │
   │    for each connected WebSocket client:  │
   │      └─→ await client.send(message)      │
   │                                          │
   │ 6. Track metrics                         │
   │    • bytes_sent                          │
   │    • messages_sent                       │
   │    • failed_sends                        │
   │    • average_latency                     │
   │                                          │
   └────────┬───────────────────────────────┘
            │
            │ JSON/WebSocket
            ↓
   ┌──────────────────────────────┐
   │ WebSocket Clients (Browser)  │
   ├──────────────────────────────┤
   │ Receive JSON message         │
   │ Parse and render on canvas:  │
   │ • Draw balls                 │
   │ • Show cue angle             │
   │ • Display trajectory lines   │
   │ • Show collision points      │
   └──────────────────────────────┘
```

## 3. Queueing & Buffering Flow

```
CAMERA FRAMES:
OpenCV Camera (30 FPS)
    ↓
Vision._capture_thread (daemon)
    ↓ [continuous loop]
Internal frame queue (deque, size=5)
    ↓ [non-blocking, drops old frames if full]
vision.process_frame() [called from integration loop]
    ↓ [returns latest available frame]

─────────────────────────────────────────

DETECTION RESULTS:
vision.process_frame()
    ↓ [returns DetectionResult or None]
IntegrationService._integration_loop() (30 FPS)
    ↓
core.update_state()
    ├─→ emit "state_updated" event
    │   ↓
    │   (Core → Integration callback)
    │   _on_state_updated()
    │   └→ (Sync → Async bridge)
    │      call_soon_threadsafe()
    │      └→ _on_state_updated_async()
    │         └→ _broadcast_with_retry()
    │            ├→ Check circuit_breaker.can_attempt()
    │            │  [OPEN: blocks all broadcasts]
    │            │  [CLOSED/HALF-OPEN: allows attempt]
    │            │
    │            └→ Retry loop (0-3 retries):
    │               ├→ try: broadcaster.broadcast_game_state()
    │               │  └→ SUCCESS: circuit_breaker.record_success()
    │               │
    │               └→ except:
    │                  ├→ Classify error type
    │                  │  (VALIDATION: don't retry)
    │                  │  (TRANSIENT: retry)
    │                  │  (UNKNOWN: retry with caution)
    │                  │
    │                  ├→ await asyncio.sleep(2^attempt * base_delay)
    │                  └→ RETRY or FAIL
    │
    └─→ emit "trajectory_calculated" event
        └→ (similar path as state)
           broadcast_trajectory()

─────────────────────────────────────────

MESSAGE BROADCAST:
broadcaster.broadcast_game_state(balls, cue, table)
    ↓
Validation (1-2ms):
    ├→ Check balls is list, not empty
    ├→ Check each ball has position dict
    ├→ Check position has x, y, scale
    └→ Check scale values are positive

State construction:
    └→ {balls, cue, table, timestamp, sequence}

JSON serialization:
    ├→ Convert Python objects to JSON
    └→ Handle numpy types

Optional compression:
    └→ If size > 1024 bytes:
       └→ zlib.compress() with level 6
          └→ Only use if compression_ratio < 0.9

Broadcast to subscribers:
    ├→ Get all clients subscribed to STATE stream
    │   (from websocket_manager.stream_subscribers[STATE])
    │
    └→ For each subscribed client:
       ├→ FPS limiting check
       │  (don't exceed client's requested FPS)
       │
       └→ async send(message)
          ├→ Queue in client's send buffer
          └→ Track latency & failures

Frame buffer (circular deque, size=100):
    └→ Store frame_data for metrics/debugging
       └→ Auto-cleanup frames older than 5s

─────────────────────────────────────────

CIRCUIT BREAKER STATES:
[CLOSED]
    Normal operation, all broadcasts attempted
         ↓ [10 consecutive failures]
[OPEN]
    Broadcast attempts blocked for 30s
         ↓ [30s timeout elapsed]
[HALF-OPEN] (temporary)
    One broadcast attempt allowed
         ├→ SUCCESS: → [CLOSED]
         └→ FAILURE: → [OPEN]
```

## 4. Performance Timeline (33ms per frame at 30 FPS)

```
Time (ms)  Event                                    Duration
─────────────────────────────────────────────────────────────
0          Frame N iteration begins
           ├─ get_vision_detection()               2ms
           │  └─ vision.process_frame()
           │
2          ├─ _convert_detection_to_core_format()  0.5ms
           │
2.5        ├─ core.update_state()                  2ms
           │  └─ Validation & state update
           │
4.5        ├─ Emit "state_updated"                 (async)
           │  └─ Schedule _on_state_updated_async()
           │
4.5        ├─ _check_trajectory_calculation()      0.5ms (if checking)
           │
5          ├─ Sleep to maintain FPS                ~28ms
           │  └─ await asyncio.sleep(28)
           │
33         ◄─ Frame N+1 begins (next iteration)


PARALLEL (in background, doesn't block main loop):
─────────────────────────────────────────────────────
During sleep(28ms):

  _on_state_updated_async() runs:
  ├─ Validate event_data           1ms
  ├─ Extract balls/cue/table       0.5ms
  ├─ Call _broadcast_with_retry()
  │  ├─ Check circuit_breaker      0.1ms
  │  ├─ Call broadcast_game_state()
  │  │  ├─ Validate structure      1-2ms
  │  │  ├─ Build message           0.5ms
  │  │  ├─ JSON serialize          1ms
  │  │  ├─ Optional compress       1-5ms (if enabled)
  │  │  └─ Send to clients         <5ms (async, non-blocking)
  │  │
  │  └─ Track metrics              0.1ms
  │
  └─ Total broadcast latency: ~5-10ms

OR if trajectory calculation:

  _check_trajectory_calculation() runs:
  ├─ Validate table state          2-3ms
  ├─ Call trajectory_calculator    20-100ms (CPU intensive)
  ├─ Convert to broadcast format   1ms
  └─ Call _broadcast_with_retry()  5-10ms

Note: Trajectory calculation (20-100ms) takes longer than frame
      interval (33ms), so only calculated when explicitly needed,
      not on every frame.
```

## 5. Threading & Concurrency Model

```
┌────────────────────────────────────────────────────────────┐
│ Uvicorn Worker Process (Single)                            │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Asyncio Event Loop (Main)                           │  │
│  │                                                     │  │
│  │  Tasks:                                            │  │
│  │  ├─ FastAPI HTTP handler                          │  │
│  │  ├─ WebSocket connection handler (per client)     │  │
│  │  ├─ Integration service loop                      │  │
│  │  ├─ Message broadcaster (sending)                 │  │
│  │  ├─ Health monitor                                │  │
│  │  └─ Other background tasks                        │  │
│  │                                                     │  │
│  │  Synchronization:                                  │  │
│  │  ├─ asyncio.Lock() for Core state updates        │  │
│  │  ├─ asyncio.Queue() for frame buffering          │  │
│  │  └─ call_soon_threadsafe() for sync→async bridge │  │
│  │                                                     │  │
│  └────────┬──────────────────────────────────────────┘  │
│           ↑  ↓                                          │
│           │  │ (non-blocking, cooperative)             │
│           │  │                                         │
│  ┌────────┴──┴───────────────────────────────────────┐  │
│  │ Vision Capture Thread (Daemon)                    │  │
│  │                                                   │  │
│  │  • Continuous loop: while running:               │  │
│  │    ├─ camera.read()    [blocking, 30-50ms]      │  │
│  │    └─ queue.put(frame) [thread-safe, non-block] │  │
│  │                                                   │  │
│  │  • Started by: IntegrationService.start()        │  │
│  │  • Stopped by: IntegrationService.stop()         │  │
│  │  • Queue size: 5 (old frames dropped)            │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Synchronization between threads:                       │
│  ├─ queue.Queue (thread-safe)                          │
│  │  └─ Used for frame passing                         │
│  │                                                     │
│  └─ No locks needed                                    │
│     └─ Vision writes only to queue                    │
│     └─ Integration reads only from queue              │
│                                                         │
└─────────────────────────────────────────────────────────┘

Concurrency summary:
  • Event loop: Handles HTTP, WebSocket, Integration, Health Monitor
  • Vision thread: Handles camera capture in parallel
  • Multiple WebSocket clients: Handled concurrently by event loop

Threading model:
  ✓ Daemon thread for camera (I/O bound)
  ✓ Async for everything else (CPU bound coordination)
  ✓ Minimal lock contention
  ✓ Thread-safe queue for Vision ↔ Integration
```

## 6. Error Handling & Resilience Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Error Handling & Resilience Mechanisms                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. BROADCAST RETRY LOGIC                                    │
│    ─────────────────────────                               │
│    _broadcast_with_retry()                                  │
│    ├─→ Check circuit_breaker.can_attempt()                 │
│    │   └─ If OPEN: return False immediately               │
│    │                                                       │
│    └─→ For attempt in range(max_retries + 1):             │
│        ├─ Try: broadcast_func(*args, **kwargs)            │
│        │  └─ SUCCESS: record_success(), return True       │
│        │                                                  │
│        └─ Except: error_type = classify_error(e)          │
│           ├─ VALIDATION: break (don't retry)              │
│           ├─ TRANSIENT: sleep(2^attempt * delay), retry   │
│           └─ UNKNOWN: sleep(2^attempt * delay), retry     │
│                                                            │
│    Default config:                                         │
│    ├─ max_retries: 3                                       │
│    ├─ retry_base_delay: 0.1s                              │
│    └─ Backoff: 0.1s, 0.2s, 0.4s, 0.8s                    │
│                                                            │
├──────────────────────────────────────────────────────────────┤
│ 2. CIRCUIT BREAKER PATTERN                                   │
│    ───────────────────────                                  │
│    CircuitBreaker class                                     │
│                                                            │
│    State machine:                                          │
│    [CLOSED] ──(10 failures)──→ [OPEN] ──(30s timeout)──→  │
│    [HALF-OPEN] ──(success)──→ [CLOSED]                    │
│              └─(failure)─→ [OPEN]                         │
│                                                            │
│    Logic:                                                  │
│    can_attempt():                                          │
│    ├─ If closed: return True                              │
│    ├─ If open and timeout elapsed:                        │
│    │  ├─ Enter half-open state                           │
│    │  └─ return True (one attempt allowed)               │
│    └─ If open and timeout not elapsed: return False      │
│                                                            │
│    on_success():                                           │
│    └─ Reset: consecutive_failures=0, is_open=False       │
│                                                            │
│    on_failure():                                           │
│    ├─ consecutive_failures++                              │
│    └─ If consecutive_failures >= threshold:              │
│       ├─ is_open = True                                  │
│       └─ circuit_open_time = now()                       │
│                                                            │
│    Default config:                                        │
│    ├─ failure_threshold: 10                              │
│    └─ timeout_seconds: 30                                │
│                                                            │
├──────────────────────────────────────────────────────────────┤
│ 3. VALIDATION ERRORS                                         │
│    ────────────────                                          │
│    _classify_broadcast_error(error):                        │
│    ├─ If error contains: "validation", "invalid", "schema"  │
│    │  └─ BroadcastErrorType.VALIDATION (don't retry)       │
│    │                                                        │
│    ├─ If error contains: "connection", "timeout", "network" │
│    │  └─ BroadcastErrorType.TRANSIENT (retry)             │
│    │                                                        │
│    └─ Else: BroadcastErrorType.UNKNOWN (retry cautiously)  │
│                                                            │
│    Impact:                                                │
│    ├─ VALIDATION: Logged error, no retries              │
│    ├─ TRANSIENT: Logged, retries with backoff           │
│    └─ UNKNOWN: Logged, retries with backoff             │
│                                                            │
├──────────────────────────────────────────────────────────────┤
│ 4. DATA VALIDATION                                            │
│    ──────────────                                            │
│    broadcast_game_state(balls, cue, table):                 │
│    ├─ Validate balls is list                              │
│    ├─ Validate balls not empty                            │
│    ├─ For each ball:                                      │
│    │  ├─ Validate is dict                                │
│    │  ├─ Validate has position                           │
│    │  ├─ Validate position is dict with x, y            │
│    │  └─ Validate position has scale (mandatory)         │
│    ├─ Validate cue is dict or None                       │
│    ├─ Validate table is dict or None                     │
│    └─ On validation failure: log warning, return early   │
│                                                            │
│    Impact:                                                │
│    └─ Invalid data: counted as validation_failures       │
│       but doesn't trigger retry (immediate drop)         │
│                                                            │
├──────────────────────────────────────────────────────────────┤
│ 5. INTEGRATION LOOP ERROR HANDLING                            │
│    ─────────────────────────────────────                     │
│    _integration_loop():                                      │
│    while self.running:                                     │
│    ├─ try:                                               │
│    │  ├─ detection = get_vision_detection()             │
│    │  ├─ process_detection(detection)                   │
│    │  ├─ check_trajectory_calculation()                 │
│    │  └─ sleep(remaining_time)                          │
│    │                                                     │
│    ├─ except asyncio.CancelledError:                    │
│    │  └─ break  (clean shutdown)                        │
│    │                                                     │
│    └─ except Exception:                                 │
│       ├─ error_count++                                 │
│       ├─ logger.error(...)                             │
│       └─ await asyncio.sleep(error_retry_delay)        │
│          └─ Continue loop (don't crash)               │
│                                                     │
│    Impact:                                          │
│    └─ Any error in processing: logged, counted,  │
│       but loop continues (resilient)              │
│                                                     │
└───────────────────────────────────────────────────┘
```

---

Generated: 2024-10-22
For architecture questions, see: BACKEND_ARCHITECTURE_ANALYSIS.md
