# Streaming Service Specification

## Module Purpose

The Streaming Service is a separate system-level process that manages camera capture and provides multiple video streams at different resolutions and frame rates. It acts as the single source of truth for camera access, eliminating conflicts and enabling true multi-client support. The Vision module and all other consumers become clients of this streaming service.

## Architecture Overview

### Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│                     Camera Hardware                          │
│                      /dev/video0                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ (exclusive access)
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Streaming Service                           │
│                 (System-level process)                       │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │           Camera Capture Thread                 │         │
│  │         (Single producer, 60fps native)         │         │
│  └──────────────────┬──────────────────────────────┘         │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────┐         │
│  │            Frame Buffer Ring                     │         │
│  │         (Shared memory, zero-copy)               │         │
│  └──────────────────┬──────────────────────────────┘         │
│                     │                                        │
│  ┌──────────────────┴───────────────────────────────┐       │
│  │                Stream Publishers                   │       │
│  ├────────────────────────────────────────────────────┤       │
│  │                                                    │       │
│  │  High Quality:  1920x1080 @ 30fps  (H.264/RTSP)  │       │
│  │  Medium Quality: 1280x720 @ 30fps  (MJPEG/HTTP)  │       │
│  │  Low Quality:    640x480 @ 15fps   (MJPEG/HTTP)  │       │
│  │  Analysis:       1920x1080 @ 30fps (Raw frames)   │       │
│  │                                                    │       │
│  └────────────────────────────────────────────────────┘       │
│                                                              │
└───────┬──────────────┬──────────────┬──────────────┬───────┘
        │              │              │              │
        │ RTSP/H.264   │ HTTP/MJPEG   │ HTTP/MJPEG   │ Shared Memory
        │ rtsp://      │ :8001        │ :8002        │ /dev/shm/
        │              │              │              │
┌───────▼────────┐ ┌───▼────┐ ┌──────▼──────┐ ┌─────▼────────┐
│  Web Frontend  │ │ Mobile │ │  Recording  │ │Vision Module │
│   (High/Med)   │ │  App   │ │   Service   │ │  (Analysis)  │
└────────────────┘ └────────┘ └─────────────┘ └──────────────┘
```

## Functional Requirements

### 1. Camera Management
- **FR-STR-001**: Exclusive camera access with no conflicts
- **FR-STR-002**: Automatic camera detection and configuration
- **FR-STR-003**: Support for multiple camera backends (V4L2, GStreamer)
- **FR-STR-004**: Hardware-accelerated capture when available
- **FR-STR-005**: Graceful camera reconnection on disconnect

### 2. Stream Generation
- **FR-STR-006**: Generate multiple simultaneous streams from single capture
- **FR-STR-007**: Support different resolutions per stream
- **FR-STR-008**: Support different frame rates per stream
- **FR-STR-009**: Support different codecs (H.264, MJPEG, raw)
- **FR-STR-010**: Zero-copy frame distribution via shared memory

### 3. Stream Protocols
- **FR-STR-011**: RTSP server for high-quality H.264 streaming
- **FR-STR-012**: HTTP server for MJPEG streaming
- **FR-STR-013**: Shared memory interface for local processes
- **FR-STR-014**: WebRTC support for low-latency browser streaming
- **FR-STR-015**: HLS/DASH for adaptive bitrate streaming

### 4. Performance
- **FR-STR-016**: Hardware acceleration via V4L2 M2M or VA-API
- **FR-STR-017**: Multi-threaded stream encoding
- **FR-STR-018**: Configurable buffer sizes and latency modes
- **FR-STR-019**: Dynamic quality adjustment based on load
- **FR-STR-020**: Memory-mapped frame buffers for zero-copy

## Non-Functional Requirements

### Performance Requirements
- **NFR-STR-001**: Support 60fps capture at 1080p
- **NFR-STR-002**: < 10ms latency for local shared memory access
- **NFR-STR-003**: < 50ms latency for network streaming
- **NFR-STR-004**: Support 10+ simultaneous clients per stream
- **NFR-STR-005**: < 30% CPU usage with hardware acceleration

### Reliability Requirements
- **NFR-STR-006**: Automatic restart on crash
- **NFR-STR-007**: Watchdog monitoring for hung capture
- **NFR-STR-008**: Graceful degradation under high load
- **NFR-STR-009**: No memory leaks over 24-hour operation
- **NFR-STR-010**: Clean shutdown without corrupting streams

## Recommended Implementation: GStreamer Pipeline

**Why GStreamer for Multiple Consumers:**
- Single camera capture shared across all consumers (backend + 2-10 frontends)
- Zero-copy shared memory for local backend vision processing
- Multiple network streams (RTSP, MJPEG) for frontend clients
- Each client can connect/disconnect independently without affecting others
- Hardware-accelerated encoding support
- Built-in multi-client handling with proper buffering

## Implementation Options

### Option 1: GStreamer Pipeline (Recommended for Multi-Consumer)

**Advantages:**
- Mature, production-ready framework
- Built-in hardware acceleration support
- Extensive codec and protocol support
- Plugin architecture for extensibility
- Zero-copy pipeline optimization

**Implementation:**
```bash
# Camera capture with multiple output streams
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  video/x-raw,width=1920,height=1080,framerate=60/1 ! \
  tee name=t \
  \
  t. ! queue ! \
    videoscale ! video/x-raw,width=1920,height=1080 ! \
    videorate ! video/x-raw,framerate=30/1 ! \
    x264enc speed-preset=ultrafast tune=zerolatency ! \
    rtph264pay ! \
    rtspsink location=rtsp://localhost:8554/high \
  \
  t. ! queue ! \
    videoscale ! video/x-raw,width=1280,height=720 ! \
    videorate ! video/x-raw,framerate=30/1 ! \
    jpegenc ! \
    multipartmux ! \
    tcpserversink host=0.0.0.0 port=8001 \
  \
  t. ! queue ! \
    videoscale ! video/x-raw,width=640,height=480 ! \
    videorate ! video/x-raw,framerate=15/1 ! \
    jpegenc ! \
    multipartmux ! \
    tcpserversink host=0.0.0.0 port=8002 \
  \
  t. ! queue ! \
    shmsink socket-path=/tmp/camera-raw shm-size=100000000
```

### Option 2: FFmpeg with Multiple Outputs

**Advantages:**
- Widely available and well-documented
- Good codec support
- Simple command-line interface
- Hardware acceleration via VA-API/NVENC

**Implementation:**
```bash
# FFmpeg with multiple output streams
ffmpeg -f v4l2 -framerate 60 -video_size 1920x1080 -i /dev/video0 \
  -map 0:v -c:v libx264 -preset ultrafast -tune zerolatency \
    -s 1920x1080 -r 30 -f rtsp rtsp://localhost:8554/high \
  -map 0:v -c:v mjpeg -s 1280x720 -r 30 \
    -f mpjpeg tcp://0.0.0.0:8001?listen=1 \
  -map 0:v -c:v mjpeg -s 640x480 -r 15 \
    -f mpjpeg tcp://0.0.0.0:8002?listen=1
```

### Option 3: Custom Python Service with OpenCV

**Advantages:**
- Full control over implementation
- Easy integration with existing Python codebase
- Can use shared memory directly with multiprocessing

**Implementation:**
```python
import cv2
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory
import threading
import time

class StreamingService:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Shared memory for zero-copy frame sharing
        self.frame_shm = shared_memory.SharedMemory(
            create=True,
            size=1920*1080*3,  # RGB frame
            name='camera_frames'
        )
        self.frame_array = np.ndarray(
            (1080, 1920, 3),
            dtype=np.uint8,
            buffer=self.frame_shm.buf
        )

        # Stream threads
        self.streams = {
            'high': StreamPublisher(1920, 1080, 30, 'rtsp', 8554),
            'medium': StreamPublisher(1280, 720, 30, 'mjpeg', 8001),
            'low': StreamPublisher(640, 480, 15, 'mjpeg', 8002),
        }

    def capture_loop(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                # Write to shared memory (zero-copy)
                np.copyto(self.frame_array, frame)

                # Notify all streams
                for stream in self.streams.values():
                    stream.new_frame_available(self.frame_array)
```

## Integration with Vision Module

### Vision Module as Consumer

The Vision module becomes a consumer of the streaming service:

```python
class VisionModule:
    def __init__(self):
        # Connect to shared memory stream
        self.frame_shm = shared_memory.SharedMemory(name='camera_frames')
        self.frame_array = np.ndarray(
            (1080, 1920, 3),
            dtype=np.uint8,
            buffer=self.frame_shm.buf
        )

        # Or connect via network stream
        self.stream = cv2.VideoCapture('rtsp://localhost:8554/high')

    def process_frame(self):
        # Get frame from shared memory (zero-copy)
        frame = self.frame_array.copy()  # Copy for processing

        # Or from network stream
        ret, frame = self.stream.read()

        # Process frame for detection
        return self.detect_objects(frame)
```

## System Integration

### Service Management

```bash
# systemd service file: /etc/systemd/system/camera-streaming.service
[Unit]
Description=Camera Streaming Service
After=network.target

[Service]
Type=simple
User=billiards
Group=video
ExecStart=/usr/bin/gst-launch-1.0 [pipeline]
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker Compose Integration

```yaml
version: '3.8'
services:
  streaming:
    image: billiards/streaming-service
    devices:
      - /dev/video0:/dev/video0
    ports:
      - "8554:8554"  # RTSP
      - "8001:8001"  # MJPEG High
      - "8002:8002"  # MJPEG Low
    volumes:
      - /dev/shm:/dev/shm  # Shared memory
    restart: always

  api:
    image: billiards/api
    depends_on:
      - streaming
    environment:
      STREAM_URL: rtsp://streaming:8554/high
    volumes:
      - /dev/shm:/dev/shm  # Shared memory access
```

## Benefits of This Architecture

### 1. True Multi-Client Support
- Each client gets independent stream connection
- No interference between clients
- Different clients can use different quality streams

### 2. Performance Optimization
- Single camera capture shared across all streams
- Hardware acceleration for encoding
- Zero-copy frame sharing via shared memory
- Parallel encoding for different streams

### 3. Reliability
- Service can restart independently of API
- Watchdog monitoring prevents hangs
- Automatic reconnection on camera issues

### 4. Flexibility
- Easy to add new stream formats
- Simple to adjust quality/resolution
- Can add recording, motion detection, etc.

### 5. Debugging
- Streams can be tested independently
- Standard tools (VLC, ffplay) can verify streams
- Clear separation of concerns

## Migration Path

### Phase 1: Standalone Streaming Service
1. Deploy GStreamer/FFmpeg streaming service
2. Test with VLC/ffplay clients
3. Verify multi-client support

### Phase 2: API Integration
1. Update API to proxy streams from service
2. Remove direct camera access from API
3. Add stream status monitoring

### Phase 3: Vision Integration
1. Update Vision module to consume from stream
2. Use shared memory for zero-latency access
3. Remove camera capture from Vision module

### Phase 4: Production Hardening
1. Add systemd service management
2. Implement health checks and monitoring
3. Add automatic restart and recovery

## Success Criteria

1. **Multi-Client Support**: 5+ simultaneous clients without degradation
2. **Low Latency**: < 50ms end-to-end for local clients
3. **Reliability**: 99.9% uptime over 24 hours
4. **Performance**: < 30% CPU with hardware acceleration
5. **Quality**: Consistent frame rate and resolution per stream