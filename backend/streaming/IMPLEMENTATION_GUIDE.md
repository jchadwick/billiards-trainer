# GStreamer Multi-Consumer Streaming Implementation Guide

## Architecture Overview

For a billiards trainer system with **1 backend vision processor + 2-10 frontend clients**, GStreamer provides the optimal solution:

```
┌──────────────────────────────────────────────────────────────┐
│                    Camera (/dev/video0)                       │
└────────────────────────┬─────────────────────────────────────┘
                         │ Exclusive access
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              GStreamer Pipeline (System Service)              │
│                                                               │
│  v4l2src → videoconvert                                       │
│         → opencv-fisheye (fisheye correction)                 │
│         → videobalance (brightness/contrast)                  │
│         → tee (split into 4 outputs)                          │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │ Output 1: Backend Vision (Shared Memory)       │           │
│  │  - 1920x1080 @30fps                            │           │
│  │  - Raw BGR frames                              │           │
│  │  - Zero-copy via shmsink                       │           │
│  │  - < 10ms latency                              │           │
│  └────────────────────────────────────────────────┘           │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │ Output 2: Frontend High (RTSP H.264)           │           │
│  │  - 1920x1080 @30fps                            │           │
│  │  - H.264 encoding                              │           │
│  │  - rtsp://IP:8554/high                         │           │
│  │  - For desktop/tablet clients                  │           │
│  └────────────────────────────────────────────────┘           │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │ Output 3: Frontend Medium (HTTP MJPEG)         │           │
│  │  - 1280x720 @30fps                             │           │
│  │  - MJPEG encoding                              │           │
│  │  - http://IP:8001                              │           │
│  │  - Up to 10 concurrent clients                 │           │
│  └────────────────────────────────────────────────┘           │
│                                                               │
│  ┌────────────────────────────────────────────────┐           │
│  │ Output 4: Frontend Low (HTTP MJPEG)            │           │
│  │  - 640x480 @15fps                              │           │
│  │  - MJPEG encoding                              │           │
│  │  - http://IP:8002                              │           │
│  │  - For mobile/slow connections                 │           │
│  └────────────────────────────────────────────────┘           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
         │                 │              │              │
         │                 │              │              │
    ┌────▼──────┐   ┌──────▼────┐   ┌────▼─────┐   ┌───▼──────┐
    │  Vision   │   │  Desktop  │   │  Tablet  │   │  Mobile  │
    │  Backend  │   │  Browser  │   │  Browser │   │  Browser │
    └───────────┘   └───────────┘   └──────────┘   └──────────┘
    (shmsrc)        (RTSP)          (MJPEG)        (MJPEG)
```

## Why GStreamer Wins for Multi-Consumer

### ✅ Advantages

1. **True Multi-Client Support**
   - Each `tee` branch is independent
   - Clients can connect/disconnect without affecting others
   - Built-in buffering prevents one slow client from blocking others

2. **Zero-Copy for Backend**
   - Shared memory (`shmsink`/`shmsrc`) eliminates network overhead
   - < 10ms latency for vision processing
   - No encoding/decoding overhead for local consumer

3. **Fisheye Correction Integration**
   - Custom GStreamer element uses OpenCV directly
   - Correction happens once at source
   - All consumers get corrected frames

4. **Automatic Client Management**
   - `tcpserversink` handles multiple clients natively
   - Automatic buffering and flow control
   - No custom multi-client code needed

### ⚠️ Considerations

1. **Setup Complexity**: More complex than Python-only solution
2. **Dependencies**: Requires GStreamer + Python GI bindings
3. **Debugging**: GStreamer debugging is less straightforward
4. **Development Time**: ~1 week vs 2-4 days for Python-only

### ❌ When NOT to Use GStreamer

- If you only had 1 consumer (backend only) → Use Python/OpenCV
- If network latency doesn't matter → Use Python/OpenCV
- If you need rapid prototyping → Use Python/OpenCV first

## Implementation Steps

### Phase 1: Install Dependencies (1-2 hours)

```bash
# On Raspberry Pi / Debian
sudo apt-get update
sudo apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3-gst-1.0 \
    python3-gi \
    libgstreamer1.0-dev

# Install OpenCV if not already installed
pip install opencv-python pyyaml
```

### Phase 2: Create Fisheye Calibration (2-4 hours)

```bash
# Use existing calibration script or create new one
cd /opt/billiards-trainer

# Capture calibration images (checkerboard pattern)
python scripts/capture-calibration-images.py --count 20

# Run fisheye calibration
python scripts/calibrate-fisheye.py \
    --images calibration/images/*.jpg \
    --output calibration/camera.yaml
```

### Phase 3: Setup GStreamer OpenCV Plugin (2-3 hours)

```bash
# Copy plugin to GStreamer plugin directory
sudo cp scripts/gst-opencv-fisheye.py /usr/lib/python3/dist-packages/gst/

# Set plugin path
export GST_PLUGIN_PATH=/usr/lib/python3/dist-packages/gst

# Test plugin
gst-inspect-1.0 opencv-fisheye
```

### Phase 4: Start Streaming Service (1 hour)

```bash
# Make script executable
chmod +x scripts/gstreamer-multistream.sh

# Start streaming service
./scripts/gstreamer-multistream.sh

# Or install as systemd service
sudo cp scripts/camera-streaming.service /etc/systemd/system/
sudo systemctl enable camera-streaming
sudo systemctl start camera-streaming
```

### Phase 5: Update Backend Vision (2-3 hours)

```python
# In backend/vision/__init__.py or similar

from ..streaming.gstreamer_consumer import GStreamerFrameConsumer

class VisionModule:
    def __init__(self, config):
        # Use GStreamer consumer instead of direct camera
        self.frame_source = GStreamerFrameConsumer("/tmp/camera-backend")
        self.frame_source.start()

        # Rest of vision module initialization
        self.table_detector = TableDetector()
        self.ball_detector = BallDetector()
        # ...

    def process_frame(self):
        # Get frame from GStreamer (already corrected and preprocessed)
        frame = self.frame_source.get_frame()

        if frame is None:
            return None

        # Process frame
        result = self.detect_objects(frame)
        return result
```

### Phase 6: Update API to Proxy Streams (1-2 hours)

```python
# In backend/api/routes/stream.py

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import requests

router = APIRouter()

@router.get("/stream/video")
async def video_stream(quality: str = "medium"):
    """Proxy video stream from GStreamer service."""

    # Map quality to GStreamer port
    stream_urls = {
        "high": "http://localhost:8554/high",    # RTSP
        "medium": "http://localhost:8001",        # MJPEG 720p
        "low": "http://localhost:8002"            # MJPEG 480p
    }

    url = stream_urls.get(quality, stream_urls["medium"])

    # Proxy stream
    def stream_generator():
        with requests.get(url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=8192):
                yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

### Phase 7: Test Multi-Client Support (1-2 hours)

```bash
# Terminal 1: Check backend is consuming
python -c "
from backend.streaming.gstreamer_consumer import GStreamerFrameConsumer
import time

consumer = GStreamerFrameConsumer()
consumer.start()

for i in range(100):
    frame = consumer.get_frame()
    print(f'Frame {i}: {frame.shape if frame is not None else None}')
    time.sleep(0.1)
"

# Terminal 2: Test frontend stream with VLC
vlc http://192.168.1.31:8001

# Terminal 3: Test another frontend client
curl http://192.168.1.31:8002 > /dev/null

# Terminal 4: Check GStreamer stats
gst-inspect-1.0 --print-all
```

## Performance Tuning

### Optimize for Low Latency (Backend Vision)

```bash
# In gstreamer-multistream.sh, for shmsink output:
shmsink socket-path=/tmp/camera-backend \
    sync=false \           # Don't wait for downstream
    wait-for-connection=false \  # Don't block on consumer
    shm-size=20000000 \    # Large enough for frame
    buffer-time=0          # Minimal buffering
```

### Optimize for Bandwidth (Frontend Streams)

```bash
# For MJPEG streams, adjust quality and resolution
jpegenc quality=70 ! \    # Lower quality = less bandwidth
videoscale ! video/x-raw,width=640,height=480 ! \  # Smaller size
videorate ! video/x-raw,framerate=15/1  # Lower FPS
```

### Enable Hardware Acceleration

```bash
# On Raspberry Pi with V4L2 M2M
x264enc → v4l2h264enc

# On systems with VA-API
x264enc → vaapih264enc

# Check available encoders
gst-inspect-1.0 | grep h264enc
```

## Troubleshooting

### Problem: Backend not receiving frames

```bash
# Check shmsink is creating socket
ls -la /tmp/camera-backend

# Check permissions
sudo chmod 666 /tmp/camera-backend

# Test with gst-launch
gst-launch-1.0 shmsrc socket-path=/tmp/camera-backend ! \
    video/x-raw,format=BGR,width=1920,height=1080 ! \
    videoconvert ! autovideosink
```

### Problem: Frontend clients getting errors

```bash
# Check if ports are listening
netstat -tlnp | grep -E '8001|8002|8554'

# Check GStreamer logs
GST_DEBUG=3 ./scripts/gstreamer-multistream.sh

# Test stream directly
ffplay http://localhost:8001
```

### Problem: High CPU usage

```bash
# Enable hardware encoding
# Check available hardware encoders
gst-inspect-1.0 | grep -i 'h264\|265\|vaapi'

# Reduce resolution/FPS for frontend streams
# Use lower JPEG quality (quality=60 instead of 85)
```

## Migration from DirectCameraModule

### Before (Direct Camera Access)
```python
camera = DirectCameraModule(config)
camera.start_capture()
frame = camera.get_frame()
```

### After (GStreamer Consumer)
```python
camera = GStreamerFrameConsumer("/tmp/camera-backend")
camera.start()
frame = camera.get_frame()
```

### Benefits of Migration
- ✅ Frames already have fisheye correction applied
- ✅ Frames already have brightness/contrast adjusted
- ✅ Multiple frontends can view simultaneously
- ✅ No camera access conflicts
- ✅ Lower latency (no encoding for backend)

## Summary

**Implementation Timeline:**
- Setup: 1 day
- Testing: 1 day
- Integration: 2-3 days
- **Total: ~5 days**

**Final Architecture:**
- ✅ 1 camera capture point (GStreamer service)
- ✅ 1 backend consumer (shared memory, zero-copy)
- ✅ 2-10 frontend consumers (HTTP/RTSP streams)
- ✅ Fisheye correction at source
- ✅ Image preprocessing at source
- ✅ Each consumer independent

This is the optimal solution for your multi-consumer requirement with centralized preprocessing.