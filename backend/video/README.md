# Video Module - Standalone Process

The Video Module is a standalone process that captures video frames from a camera or video file and writes them to shared memory for consumption by the Vision Module and API streaming endpoints.

## Architecture

```
┌──────────────────────────────────┐
│   Video Module Process           │
│   (python -m backend.video)      │
│                                  │
│  ┌────────────────────────────┐ │
│  │  VideoProcess              │ │
│  │  - Camera capture          │ │
│  │  - Frame writing           │ │
│  │  - Signal handling         │ │
│  └────────────────────────────┘ │
│           │                      │
│           ↓                      │
│  ┌────────────────────────────┐ │
│  │  SharedMemoryFrameWriter   │ │
│  │  - Triple buffering        │ │
│  │  - Lock-free reads         │ │
│  └────────────────────────────┘ │
└──────────────────────────────────┘
           │
           │ Shared Memory (IPC)
           │
           ↓
┌──────────────────────────────────┐
│   Consumer Processes             │
│   - Vision Module                │
│   - API Streaming                │
│   - (Other readers)              │
└──────────────────────────────────┘
```

## Features

- **Standalone Process**: Runs independently from the main backend
- **Shared Memory IPC**: Low-latency frame delivery (<5ms)
- **Triple Buffering**: Lock-free reading for consumers
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT
- **Resource Cleanup**: Automatic cleanup of camera and shared memory
- **Flexible Configuration**: Support for camera devices and video files
- **Loop Mode**: Video files can loop continuously for testing

## Usage

### Starting the Video Module

```bash
# Run with default config.json
python -m backend.video

# Run with custom config file
python -m backend.video /path/to/config.json
```

### Environment Overrides

For testing and development, you can override configuration via environment variables:

```bash
# Use video file instead of camera
VIDEO_FILE=/path/to/video.mp4 python -m backend.video

# Override log level
LOG_LEVEL=DEBUG python -m backend.video

# Combine overrides
VIDEO_FILE=assets/demo3.mp4 LOG_LEVEL=INFO python -m backend.video
```

### Stopping the Video Module

The process handles signals gracefully:

```bash
# Send SIGTERM (preferred)
kill -TERM <pid>

# Send SIGINT (Ctrl+C)
kill -INT <pid>
```

## Configuration

The Video Module reads configuration from `config.json`. Relevant sections:

### Video Configuration

```json
{
  "video": {
    "shared_memory_name": "billiards_video",
    "shared_memory_attach_timeout_sec": 5.0,
    "process": {
      "shutdown_timeout": 10.0,
      "main_loop_sleep": 0.001
    }
  }
}
```

### Camera Configuration

The Video Module uses the same camera configuration as the Vision Module:

```json
{
  "vision": {
    "camera": {
      "device_id": 0,
      "backend": "auto",
      "resolution": [1920, 1080],
      "fps": 30,
      "exposure_mode": "auto",
      "buffer_size": 2,
      "loop_video": true,
      "auto_reconnect": true
    }
  }
}
```

## Components

### VideoProcess (`process.py`)

Main orchestrator class that manages the video capture lifecycle:

1. **Initialization**
   - Loads configuration
   - Sets up signal handlers
   - Initializes camera
   - Initializes shared memory writer

2. **Main Loop**
   - Captures frames from camera
   - Writes frames to shared memory
   - Logs statistics periodically
   - Runs until shutdown signal

3. **Cleanup**
   - Stops camera capture
   - Closes shared memory
   - Logs final statistics

### Entry Point (`__main__.py`)

Process entry point that handles:
- Configuration loading
- Environment variable overrides
- Logging setup
- Process initialization
- Exit code handling

## Shared Memory

The Video Module uses triple-buffered shared memory for frame delivery:

- **Format**: BGR24 (OpenCV default)
- **Buffers**: 3 (writer rotates, readers always get latest)
- **Size**: ~20MB for 1920x1080 frames
- **Latency**: <5ms from capture to consumer read

### Memory Layout

```
[Header Block (4KB)]
[Frame Buffer 0 (width × height × 3 bytes)]
[Frame Buffer 1 (width × height × 3 bytes)]
[Frame Buffer 2 (width × height × 3 bytes)]
```

### Header Contents

- Magic number and version
- Frame dimensions and format
- Current read/write indices
- Frame metadata (number, timestamp)
- Writer PID and reader count

## Performance

### Targets

- **Frame Rate**: 30 FPS sustained
- **Latency**: <5ms write to shared memory
- **CPU Usage**: <10%
- **Memory**: ~20MB stable

### Actual Performance (Demo Video)

- Frame Rate: 22-24 FPS (limited by video file FPS)
- Resolution: 3840x2160 (4K)
- Memory: ~96MB shared memory (3 × 4K buffers)
- CPU: <5%
- Shutdown Time: <100ms

## Testing

### Basic Test

```bash
python test_video_process.py
```

Tests:
- Process startup
- Shared memory initialization
- Frame reading
- Graceful shutdown

### Loop Mode Test

```bash
python test_video_loop.py
```

Tests:
- Video file loop mode
- Extended reading
- Frame number monitoring

### Manual Test

```bash
# Terminal 1: Start Video Module
VIDEO_FILE=assets/demo3.mp4 python -m backend.video

# Terminal 2: Read from shared memory
python -c "
from backend.video.ipc.shared_memory import SharedMemoryFrameReader
import time

reader = SharedMemoryFrameReader('billiards_video')
reader.attach()
print(f'Connected: {reader.width}x{reader.height} {reader.frame_format.name}')

for i in range(100):
    frame, metadata = reader.read_frame()
    if frame is not None:
        print(f'Frame {metadata.frame_number}: {frame.shape}')
    time.sleep(0.033)

reader.detach()
"
```

## Troubleshooting

### Process Won't Start

**Problem**: Process exits immediately or fails to start

**Solutions**:
1. Check config file exists and is valid JSON
2. Check camera device is available (for hardware cameras)
3. Check video file exists (for video file sources)
4. Enable DEBUG logging: `LOG_LEVEL=DEBUG python -m backend.video`

### Shared Memory Errors

**Problem**: Consumers can't attach to shared memory

**Solutions**:
1. Ensure Video Module process is running
2. Check shared memory name matches in config
3. Check permissions (shared memory uses /dev/shm on Linux)
4. Look for fallback to file-backed memory in logs

### Low Frame Rate

**Problem**: FPS lower than expected

**Solutions**:
1. Check source FPS (video files have fixed FPS)
2. Check CPU usage (may be throttled)
3. Check camera configuration (wrong resolution/FPS)
4. Enable frame logging: set `frame_log_interval: 1` in config

### Memory Leaks

**Problem**: Memory usage grows over time

**Solutions**:
1. Check for BufferError warnings (Python multiprocessing issue)
2. Restart process periodically
3. Monitor with: `ps aux | grep "backend.video"`
4. Check shared memory: `ls -lh /dev/shm/` (Linux)

### Camera Reconnection

**Problem**: Camera disconnects and doesn't reconnect

**Solutions**:
1. Check `auto_reconnect: true` in config
2. Check `max_reconnect_attempts` is sufficient
3. Check camera logs for errors
4. Verify camera is properly connected

## Integration with Vision Module

The Vision Module will consume frames from the Video Module via shared memory:

```python
from backend.video.ipc.shared_memory import SharedMemoryFrameReader

# In Vision Module initialization
reader = SharedMemoryFrameReader(name="billiards_video")
reader.attach(timeout=5.0)

# In Vision Module main loop
frame, metadata = reader.read_frame()
if frame is not None:
    # Process frame...
    pass
```

See [Phase 2 Implementation Plan](../../thoughts/PLAN_phase2_shared_memory_ipc.md) for full integration details.

## Future Enhancements

- Health monitoring and metrics
- Watchdog for automatic restart
- Multiple video sources
- Frame preprocessing options
- Dynamic resolution switching
- Performance profiling tools

## References

- [Phase 2 Implementation Plan](../../thoughts/PLAN_phase2_shared_memory_ipc.md)
- [Shared Memory IPC Module](ipc/shared_memory.py)
- [Camera Capture Module](../vision/capture.py)
- [Configuration System](../config.py)
