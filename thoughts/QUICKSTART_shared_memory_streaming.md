# Quick Start: Shared Memory Streaming

This guide shows you how to use the new shared memory-based video streaming endpoints.

## Prerequisites

1. **Video Module** must be running (separate process that captures video and writes to shared memory)
2. **API Server** must be running
3. **Config** must have shared memory settings

## Starting the System

### Step 1: Start Video Module

```bash
# In terminal 1
cd /Users/jchadwick/code/billiards-trainer
python -m backend.video
```

Expected output:
```
INFO - Initializing SharedMemoryFrameWriter: billiards_video, 1920x1080 BGR24, ...
INFO - Shared memory initialized: billiards_video (...)
INFO - Video Module started successfully
```

### Step 2: Start API Server

```bash
# In terminal 2
cd /Users/jchadwick/code/billiards-trainer
python -m backend.api.main
```

Expected output:
```
INFO - Application startup complete.
INFO - Uvicorn running on http://0.0.0.0:8000
```

## Testing the Endpoints

### Test 1: Direct Shared Memory Endpoint

```bash
# Stream from shared memory (new endpoint)
curl http://localhost:8000/stream/video/shm?quality=85&fps=30
```

Or open in browser:
```
http://localhost:8000/stream/video/shm?quality=85&fps=30
```

### Test 2: Feature Flag Routing

To use the main `/stream/video` endpoint with shared memory:

1. Edit `config.json`:
```json
{
  "video": {
    "use_shared_memory": true
  }
}
```

2. Restart API server

3. Test:
```bash
curl http://localhost:8000/stream/video?quality=85&fps=30
```

### Test 3: Multiple Concurrent Clients

```bash
# Run test script with 10 clients
python backend/tests/manual/test_streaming_endpoints.py --mode shm --clients 10

# Or 15 clients for stress test
python backend/tests/manual/test_streaming_endpoints.py --mode shm --clients 15
```

Expected output:
```
INFO - TEST: Concurrent Streaming (10 clients)
INFO - Starting 10 concurrent clients...
INFO - Client 1 connecting to http://localhost:8000/stream/video/shm...
...
INFO - Concurrent Client Results:
INFO -   Total Clients: 10
INFO -   Average FPS per Client: 29.5
INFO -   Total Errors: 0
INFO - TEST RESULT: PASS
```

## Configuration

### Required Settings (config.json)

```json
{
  "video": {
    "use_shared_memory": false,  // Set to true to enable feature flag routing
    "shared_memory_name": "billiards_video",
    "shared_memory_attach_timeout_sec": 5.0,
    "process": {
      "shutdown_timeout": 10.0,
      "main_loop_sleep": 0.001
    }
  }
}
```

### Optional Stream Settings

```json
{
  "api": {
    "stream": {
      "quality": {
        "default_jpeg_quality": 85,  // Default JPEG quality
        "min_jpeg_quality": 1,
        "max_jpeg_quality": 100
      },
      "framerate": {
        "default_fps": 30,  // Default target FPS
        "min_fps": 1,
        "max_fps": 60
      },
      "performance": {
        "frame_log_interval": 30,  // Log every N frames
        "sleep_interval_ms": 10     // Sleep interval between frame checks
      }
    }
  }
}
```

## Endpoints

### GET /stream/video/shm

**New endpoint** - Always uses shared memory (regardless of feature flag)

**Query Parameters**:
- `quality` (optional): JPEG quality 1-100, default: 85
- `fps` (optional): Target FPS 1-60, default: 30

**Example**:
```bash
curl "http://localhost:8000/stream/video/shm?quality=90&fps=25"
```

**Response**:
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- Format: MJPEG stream

**Error Responses**:
- `503`: Video Module not running
- `500`: Stream generation error

### GET /stream/video

**Enhanced endpoint** - Supports both modes via feature flag

**Behavior**:
- If `video.use_shared_memory=true`: Uses shared memory
- If `video.use_shared_memory=false`: Uses legacy camera module

**Query Parameters**:
- `quality` (optional): JPEG quality 1-100
- `fps` (optional): Target FPS 1-60
- `width` (optional): Max width for scaling (legacy mode only)
- `height` (optional): Max height for scaling (legacy mode only)

**Example**:
```bash
# With shared memory enabled
curl "http://localhost:8000/stream/video?quality=85&fps=30"
```

## Troubleshooting

### Error: "Video Module is not running"

**Problem**: API can't connect to shared memory

**Solution**:
1. Start Video Module: `python -m backend.video`
2. Check Video Module logs for errors
3. Verify config has correct `video.shared_memory_name`

### Error: "Permission denied" on shared memory

**Problem**: Can't create/access shared memory segment

**Solution**: Shared memory module has automatic fallback to file-backed mmap
- Check logs for "Falling back to mmap file" message
- Fallback location: `/tmp/billiards_shm/`
- No action needed, but performance may be slightly lower

### No frames received

**Problem**: Streaming but not receiving frames

**Possible causes**:
1. Video Module not capturing frames (check Video Module logs)
2. Camera not connected or video file not found
3. Network issue (firewall, timeout)

**Solutions**:
1. Check Video Module logs: `tail -f logs/video_module.log`
2. Verify camera/video file in config
3. Test with low FPS: `fps=5`

### High CPU usage

**Problem**: CPU usage too high with many clients

**Expected**: ~5-10ms per client for JPEG encoding

**If higher**:
1. Lower JPEG quality: `quality=70`
2. Reduce FPS: `fps=15`
3. Check if multiple processes competing for CPU
4. Consider hardware acceleration (future enhancement)

## Performance Tips

### For Low Bandwidth Networks

```bash
# Use lower quality and FPS
curl "http://localhost:8000/stream/video/shm?quality=60&fps=15"
```

### For High Quality Streaming

```bash
# Use higher quality (more CPU, more bandwidth)
curl "http://localhost:8000/stream/video/shm?quality=95&fps=30"
```

### For Many Concurrent Clients

- Use moderate quality: `quality=75-85`
- Use standard FPS: `fps=30`
- Monitor CPU usage
- Consider rate limiting if needed

## Monitoring

### Check Streaming Status

```bash
curl http://localhost:8000/stream/video/status
```

Response:
```json
{
  "streaming": {
    "active_streams": 5,
    "total_frames_served": 15000,
    "avg_fps": 30.0,
    "uptime": 500.0
  }
}
```

### Check Video Module Status

Video Module logs show:
- Frame capture rate
- Frame write rate
- Shared memory stats
- Error counts

## Migration from Legacy Mode

### Step 1: Test in Parallel

Keep legacy mode enabled, test new endpoint separately:

```bash
# Test shared memory endpoint (Video Module required)
curl http://localhost:8000/stream/video/shm

# Test legacy endpoint (uses EnhancedCameraModule)
curl http://localhost:8000/stream/video
```

### Step 2: Enable Feature Flag

Once confident shared memory works:

1. Edit `config.json`: `"video.use_shared_memory": true`
2. Restart API server
3. All traffic to `/stream/video` now uses shared memory

### Step 3: Monitor and Validate

- Watch logs for "Using shared memory streaming mode" messages
- Monitor CPU/memory usage
- Test with multiple clients
- Verify no errors

### Step 4: Rollback if Needed

1. Edit `config.json`: `"video.use_shared_memory": false`
2. Restart API server
3. Stop Video Module (not needed in legacy mode)

## Example Use Cases

### Use Case 1: Single Client Viewer

```bash
# Browser-based viewer
open http://localhost:8000/stream/video/shm?quality=90&fps=30
```

### Use Case 2: Multiple Dashboard Clients

```bash
# Start 10 dashboard clients
for i in {1..10}; do
  curl http://localhost:8000/stream/video/shm?quality=80&fps=30 > /dev/null &
done

# Check active streams
curl http://localhost:8000/stream/video/status | jq '.streaming.active_streams'
```

### Use Case 3: Low-Latency Monitoring

```bash
# High FPS, moderate quality
curl "http://localhost:8000/stream/video/shm?quality=75&fps=60"
```

### Use Case 4: Bandwidth-Constrained

```bash
# Low quality, low FPS
curl "http://localhost:8000/stream/video/shm?quality=50&fps=10"
```

## Next Steps

1. **Load Testing**: Run test script with 15+ clients
2. **Integration**: Integrate with frontend application
3. **Monitoring**: Set up metrics collection
4. **Optimization**: Tune quality/FPS based on network conditions
5. **Documentation**: Update API docs with new endpoints

## Support

For issues or questions:
- Check logs: `logs/api.log`, `logs/video_module.log`
- Review implementation doc: `thoughts/IMPLEMENTATION_streaming_endpoints_phase2_task4.md`
- Run test script: `python backend/tests/manual/test_streaming_endpoints.py --mode all`
