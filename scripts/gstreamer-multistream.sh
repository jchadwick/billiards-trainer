#!/bin/bash

# GStreamer Multi-Stream Camera Service with Fisheye Correction
# Serves: 1 backend (shared memory) + 2-10 frontend clients (HTTP/RTSP)

set -e

# Configuration
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
CALIBRATION_FILE="${CALIBRATION_FILE:-/opt/billiards-trainer/calibration/camera.yaml}"

# Stream ports
SHM_SOCKET="/tmp/camera-backend"     # Backend vision (shared memory)
RTSP_PORT="8554"                      # High quality RTSP
MJPEG_HIGH_PORT="8001"                # High quality MJPEG (1280x720)
MJPEG_LOW_PORT="8002"                 # Low quality MJPEG (640x480)

# Image adjustments
BRIGHTNESS="${BRIGHTNESS:-0.1}"       # -1 to 1
CONTRAST="${CONTRAST:-1.2}"           # 0 to 2
SATURATION="${SATURATION:-1.0}"       # 0 to 2

echo "Starting multi-consumer camera streaming with fisheye correction..."

# Build GStreamer pipeline with OpenCV fisheye correction
gst-launch-1.0 -v \
  v4l2src device=$CAMERA_DEVICE ! \
  video/x-raw,width=1920,height=1080,framerate=30/1 ! \
  \
  videoconvert ! \
  \
  `# Fisheye correction using OpenCV (if calibration exists)` \
  $(if [ -f "$CALIBRATION_FILE" ]; then
    echo "opencv-undistort calibration-file=$CALIBRATION_FILE !"
  fi) \
  \
  `# Image adjustments` \
  videobalance \
    brightness=$BRIGHTNESS \
    contrast=$CONTRAST \
    saturation=$SATURATION ! \
  \
  `# Split into multiple outputs` \
  tee name=t \
  \
  `# Output 1: Backend Vision (shared memory, full resolution, raw frames)` \
  t. ! queue max-size-buffers=2 leaky=downstream ! \
    video/x-raw,format=BGR ! \
    shmsink \
      socket-path=$SHM_SOCKET \
      sync=false \
      wait-for-connection=false \
      shm-size=20000000 \
  \
  `# Output 2: Frontend High Quality (RTSP H.264, 1920x1080 @30fps)` \
  t. ! queue max-size-buffers=2 ! \
    videoconvert ! \
    x264enc \
      speed-preset=ultrafast \
      tune=zerolatency \
      bitrate=4000 \
      key-int-max=30 ! \
    rtph264pay config-interval=1 pt=96 ! \
    rtspsink location=rtsp://0.0.0.0:$RTSP_PORT/high \
  \
  `# Output 3: Frontend Medium (HTTP MJPEG, 1280x720 @30fps)` \
  t. ! queue max-size-buffers=2 ! \
    videoscale ! video/x-raw,width=1280,height=720 ! \
    jpegenc quality=85 ! \
    multipartmux boundary=frame ! \
    tcpserversink \
      host=0.0.0.0 \
      port=$MJPEG_HIGH_PORT \
      sync-method=latest-keyframe \
      recover-policy=keyframe \
      max-clients=10 \
  \
  `# Output 4: Frontend Low/Mobile (HTTP MJPEG, 640x480 @15fps)` \
  t. ! queue max-size-buffers=2 ! \
    videoscale ! video/x-raw,width=640,height=480 ! \
    videorate ! video/x-raw,framerate=15/1 ! \
    jpegenc quality=70 ! \
    multipartmux boundary=frame ! \
    tcpserversink \
      host=0.0.0.0 \
      port=$MJPEG_LOW_PORT \
      sync-method=latest-keyframe \
      recover-policy=keyframe \
      max-clients=10
