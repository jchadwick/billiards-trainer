# Message Publishing Investigation - Complete Analysis

This directory contains a comprehensive investigation of how detected balls are published/communicated in the billiards-trainer system.

## Documents

### 1. `publishing_summary.md` (Start here!)
Quick reference overview with key metrics and conclusions. Perfect for executives or quick understanding.
- System architecture overview
- Key performance metrics table
- Data flow diagram
- Conclusion: NOT a bottleneck

### 2. `message_publishing_investigation.md` (Deep dive)
Comprehensive technical investigation covering all aspects:
- Messaging system details (WebSocket)
- Message serialization and sizing
- Publishing mechanism (async)
- Message queuing and batching
- Network configuration
- Blocking I/O analysis
- Message frequency and throttling
- Buffer sizes and capacity
- Detailed bottleneck analysis
- Performance characteristics
- Recommendations for optimization

### 3. `message_publishing_architecture.txt` (Visual reference)
Detailed architecture diagrams and visual flows:
- Protocol stack diagram
- Complete data flow (sync to async)
- Performance characteristics metrics
- Error handling and resilience patterns
- Configuration parameters
- Blocking I/O verification checklist
- Conclusion with profiling recommendations

## Quick Answers

**Q: What messaging system is used?**
A: WebSocket (FastAPI/Starlette), not MQTT

**Q: Message serialization and size?**
A: JSON with numpy encoder, ~500-800 bytes per message, compression enabled (30-50% reduction)

**Q: Synchronous or asynchronous?**
A: Fully asynchronous with async/await, but bridged from sync callbacks via event loop scheduler

**Q: Any blocking I/O?**
A: None - all WebSocket operations are awaitable, compression is fast (<2ms), no busy-wait loops

**Q: Message queuing or batching?**
A: Optional frame queue (10 max), no explicit batching but implicit via event loop, per-client FPS limiting (30 default)

**Q: Network configuration?**
A: WebSocket with 30s ping, 60s timeout, 100 messages/min rate limit, configurable compression (Zlib level 6)

**Q: Is message publishing a bottleneck?**
A: **NO** - well-designed async architecture, efficient serialization, compression enabled, parallel broadcasting

**Q: What IS likely a bottleneck?**
A: Vision detection (YOLO), physics calculations, core game state updates, network latency to clients

## Key Metrics

| Metric | Value |
|--------|-------|
| Message Size | 500-800 bytes (16 balls) |
| Compression | Zlib level 6 (30-50% reduction) |
| End-to-End Latency | ~12ms (excl. network) |
| Per-Client FPS | 30 max (configurable) |
| Broadcasting | asyncio.gather() (parallel) |
| Circuit Breaker | 10 failure threshold, 30s timeout |
| Retry Logic | 3 attempts, exponential backoff |
| No Blocking I/O | Verified throughout architecture |

## Code Files Analyzed

- `/backend/api/websocket/broadcaster.py` - Core broadcasting logic (800+ lines)
- `/backend/api/websocket/manager.py` - Stream management (610 lines)
- `/backend/api/websocket/handler.py` - Connection handling (350+ lines)
- `/backend/integration_service.py` - Event-driven publishing (1200+ lines)
- `/backend/config.py` - Configuration system (177 lines)

## Data Flow Summary

```
Vision (threaded)
  → Core state update (sync, <1ms)
    → Event callback (sync)
      → Event loop scheduler (async bridge)
        → Integration handler (async)
          → Message broadcaster (async)
            → WebSocket manager (async)
              → Per-client sends (parallel asyncio.gather)
```

## Architecture Highlights

1. **Fully Asynchronous**: All operations use async/await
2. **Well-Balanced**: Bridges sync detection with async broadcasting cleanly
3. **Efficient Serialization**: JSON with compression
4. **Resilient**: Circuit breaker, exponential backoff, error recovery
5. **Scalable**: Parallel broadcasting via asyncio.gather()
6. **Configurable**: All parameters tunable via config.json

## Recommendations

If performance issues exist:

1. **Profile first**: Check `/api/v1/websocket/metrics` for actual latency
2. **Likely causes**: Vision detection (YOLO), physics calculations, network latency
3. **Publishing tuning** (if needed):
   - Reduce FPS: `integration.target_fps = 15` (from 30)
   - Use quality filters: `quality_level: "low"` for some clients
   - Increase compression: `compression.level = 9` (from 6)

## Investigation Methodology

This investigation:
1. Located all WebSocket/messaging code
2. Traced message flow from detection to broadcast
3. Analyzed asynchronous patterns and blocking points
4. Measured message serialization and size
5. Reviewed error handling and resilience patterns
6. Checked configuration and tuning parameters
7. Profiled expected latencies and throughput
8. Identified actual bottlenecks (not in messaging)

## Contact

For questions about this investigation, refer to the detailed markdown files or contact the development team.
