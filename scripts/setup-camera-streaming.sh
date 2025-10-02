#!/bin/bash

# Camera Streaming Service Setup Script
# Configures and starts a multi-stream camera service using GStreamer
# Author: Billiards Trainer Team
# Version: 1.0.0

set -e  # Exit on error

# Configuration
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
RTSP_PORT="${RTSP_PORT:-8554}"
MJPEG_HIGH_PORT="${MJPEG_HIGH_PORT:-8001}"
MJPEG_LOW_PORT="${MJPEG_LOW_PORT:-8002}"
SHM_PATH="${SHM_PATH:-/tmp/camera-raw}"
LOG_FILE="${LOG_FILE:-/var/log/camera-streaming.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1" >> "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $1" >> "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if running as root or with video group
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root. Consider running as a user in the 'video' group."
    elif ! groups | grep -q video; then
        log_error "Current user is not in the 'video' group. Run: sudo usermod -a -G video $USER"
        exit 1
    fi

    # Check for GStreamer
    if ! command -v gst-launch-1.0 &> /dev/null; then
        log_error "GStreamer not found. Install with: sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
        exit 1
    fi

    # Check for camera device
    if [[ ! -e "$CAMERA_DEVICE" ]]; then
        log_error "Camera device $CAMERA_DEVICE not found"
        log_info "Available devices:"
        ls -la /dev/video* 2>/dev/null || echo "No video devices found"
        exit 1
    fi

    # Check camera permissions
    if [[ ! -r "$CAMERA_DEVICE" ]] || [[ ! -w "$CAMERA_DEVICE" ]]; then
        log_error "No read/write permissions for $CAMERA_DEVICE"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Test camera with v4l2
test_camera() {
    log_info "Testing camera device $CAMERA_DEVICE..."

    if command -v v4l2-ctl &> /dev/null; then
        # Get camera capabilities
        log_info "Camera capabilities:"
        v4l2-ctl -d "$CAMERA_DEVICE" --list-formats-ext | head -20

        # Test capture
        log_info "Testing capture..."
        if timeout 2 v4l2-ctl -d "$CAMERA_DEVICE" --stream-mmap --stream-count=1 &> /dev/null; then
            log_info "Camera test successful"
        else
            log_warn "Camera test with v4l2-ctl failed, but may still work with GStreamer"
        fi
    else
        log_warn "v4l2-ctl not installed, skipping camera test"
    fi
}

# Kill existing streaming processes
cleanup_existing() {
    log_info "Cleaning up existing streaming processes..."

    # Kill existing GStreamer processes
    pkill -f "gst-launch.*$CAMERA_DEVICE" || true

    # Kill processes using the ports
    fuser -k "$RTSP_PORT/tcp" 2>/dev/null || true
    fuser -k "$MJPEG_HIGH_PORT/tcp" 2>/dev/null || true
    fuser -k "$MJPEG_LOW_PORT/tcp" 2>/dev/null || true

    sleep 1
    log_info "Cleanup complete"
}

# Start GStreamer pipeline
start_gstreamer_pipeline() {
    log_info "Starting GStreamer multi-stream pipeline..."

    # Build the pipeline command
    PIPELINE="gst-launch-1.0 -e \
        v4l2src device=$CAMERA_DEVICE ! \
        video/x-raw,width=1920,height=1080,framerate=30/1 ! \
        tee name=t \
        \
        t. ! queue max-size-buffers=2 ! \
            videoscale ! video/x-raw,width=1920,height=1080 ! \
            videorate ! video/x-raw,framerate=30/1 ! \
            videoconvert ! \
            x264enc speed-preset=ultrafast tune=zerolatency key-int-max=30 ! \
            rtph264pay config-interval=1 pt=96 ! \
            udpsink host=127.0.0.1 port=$RTSP_PORT \
        \
        t. ! queue max-size-buffers=2 ! \
            videoscale ! video/x-raw,width=1280,height=720 ! \
            videorate ! video/x-raw,framerate=30/1 ! \
            jpegenc quality=85 ! \
            multipartmux boundary=frame ! \
            tcpserversink host=0.0.0.0 port=$MJPEG_HIGH_PORT \
        \
        t. ! queue max-size-buffers=2 ! \
            videoscale ! video/x-raw,width=640,height=480 ! \
            videorate ! video/x-raw,framerate=15/1 ! \
            jpegenc quality=75 ! \
            multipartmux boundary=frame ! \
            tcpserversink host=0.0.0.0 port=$MJPEG_LOW_PORT \
        \
        t. ! queue ! \
            shmsink socket-path=$SHM_PATH shm-size=10000000 wait-for-connection=false"

    # Start the pipeline in background
    log_info "Executing pipeline..."
    nohup bash -c "$PIPELINE" >> "$LOG_FILE" 2>&1 &
    PIPELINE_PID=$!

    # Wait a moment for pipeline to start
    sleep 3

    # Check if pipeline is running
    if kill -0 $PIPELINE_PID 2>/dev/null; then
        log_info "GStreamer pipeline started successfully (PID: $PIPELINE_PID)"
        echo $PIPELINE_PID > /tmp/camera-streaming.pid
        return 0
    else
        log_error "GStreamer pipeline failed to start"
        tail -n 20 "$LOG_FILE"
        return 1
    fi
}

# Start simple FFmpeg fallback
start_ffmpeg_fallback() {
    log_info "Starting FFmpeg fallback streaming..."

    # Simple MJPEG stream with FFmpeg
    FFMPEG_CMD="ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i $CAMERA_DEVICE \
        -c:v mjpeg -q:v 5 \
        -f mpjpeg -boundary_tag frame \
        tcp://0.0.0.0:$MJPEG_HIGH_PORT?listen=1"

    if command -v ffmpeg &> /dev/null; then
        nohup bash -c "$FFMPEG_CMD" >> "$LOG_FILE" 2>&1 &
        FFMPEG_PID=$!
        sleep 2

        if kill -0 $FFMPEG_PID 2>/dev/null; then
            log_info "FFmpeg streaming started (PID: $FFMPEG_PID)"
            echo $FFMPEG_PID > /tmp/camera-streaming.pid
            return 0
        else
            log_error "FFmpeg streaming failed to start"
            return 1
        fi
    else
        log_error "FFmpeg not installed"
        return 1
    fi
}

# Test streams
test_streams() {
    log_info "Testing streams..."

    # Test MJPEG high quality stream
    if timeout 2 curl -s "http://localhost:$MJPEG_HIGH_PORT" > /dev/null; then
        log_info "MJPEG high quality stream (port $MJPEG_HIGH_PORT): OK"
    else
        log_warn "MJPEG high quality stream not responding"
    fi

    # Test MJPEG low quality stream
    if timeout 2 curl -s "http://localhost:$MJPEG_LOW_PORT" > /dev/null; then
        log_info "MJPEG low quality stream (port $MJPEG_LOW_PORT): OK"
    else
        log_warn "MJPEG low quality stream not responding"
    fi

    # Test shared memory
    if [[ -e "$SHM_PATH" ]]; then
        log_info "Shared memory socket created at $SHM_PATH"
    else
        log_warn "Shared memory socket not found"
    fi
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."

    SERVICE_FILE="/etc/systemd/system/camera-streaming.service"

    if [[ $EUID -ne 0 ]]; then
        log_warn "Not running as root, cannot create systemd service"
        log_info "To create service, run: sudo $0 --install-service"
        return
    fi

    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Camera Streaming Service
After=network.target

[Service]
Type=simple
User=$SUDO_USER
Group=video
Environment="CAMERA_DEVICE=$CAMERA_DEVICE"
Environment="RTSP_PORT=$RTSP_PORT"
Environment="MJPEG_HIGH_PORT=$MJPEG_HIGH_PORT"
Environment="MJPEG_LOW_PORT=$MJPEG_LOW_PORT"
ExecStart=$0 --daemon
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable camera-streaming.service
    log_info "Systemd service created and enabled"
    log_info "Start with: sudo systemctl start camera-streaming"
}

# Print stream URLs
print_urls() {
    echo ""
    log_info "=== Stream URLs ==="
    echo -e "${GREEN}High Quality MJPEG:${NC} http://$(hostname -I | awk '{print $1}'):$MJPEG_HIGH_PORT"
    echo -e "${GREEN}Low Quality MJPEG:${NC} http://$(hostname -I | awk '{print $1}'):$MJPEG_LOW_PORT"
    echo -e "${GREEN}RTSP Stream:${NC} rtsp://$(hostname -I | awk '{print $1}'):$RTSP_PORT/stream"
    echo -e "${GREEN}Shared Memory:${NC} $SHM_PATH"
    echo ""
    log_info "Test with: vlc http://$(hostname -I | awk '{print $1}'):$MJPEG_HIGH_PORT"
}

# Main execution
main() {
    case "${1:-}" in
        --install-service)
            create_systemd_service
            exit 0
            ;;
        --daemon)
            # Running as daemon, skip interactive parts
            ;;
        --stop)
            log_info "Stopping camera streaming..."
            cleanup_existing
            exit 0
            ;;
        --test)
            check_prerequisites
            test_camera
            exit 0
            ;;
        --help)
            echo "Usage: $0 [--install-service|--daemon|--stop|--test|--help]"
            echo ""
            echo "Options:"
            echo "  --install-service  Create and enable systemd service (requires root)"
            echo "  --daemon          Run as daemon (used by systemd)"
            echo "  --stop            Stop all streaming processes"
            echo "  --test            Test camera without starting streams"
            echo "  --help            Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  CAMERA_DEVICE     Camera device (default: /dev/video0)"
            echo "  RTSP_PORT         RTSP stream port (default: 8554)"
            echo "  MJPEG_HIGH_PORT   High quality MJPEG port (default: 8001)"
            echo "  MJPEG_LOW_PORT    Low quality MJPEG port (default: 8002)"
            exit 0
            ;;
    esac

    log_info "Starting Camera Streaming Service"

    # Run checks
    check_prerequisites
    test_camera
    cleanup_existing

    # Try GStreamer first
    if start_gstreamer_pipeline; then
        log_info "GStreamer pipeline running"
    else
        log_warn "GStreamer failed, trying FFmpeg fallback"
        if ! start_ffmpeg_fallback; then
            log_error "All streaming methods failed"
            exit 1
        fi
    fi

    # Test the streams
    sleep 2
    test_streams

    # Print URLs
    print_urls

    # Keep running if in daemon mode
    if [[ "${1:-}" == "--daemon" ]]; then
        log_info "Running in daemon mode"
        # Wait for the streaming process
        wait $(cat /tmp/camera-streaming.pid 2>/dev/null)
    else
        log_info "Streaming service started in background"
        log_info "Stop with: $0 --stop"
    fi
}

# Run main function
main "$@"
