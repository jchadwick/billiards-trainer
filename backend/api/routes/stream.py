"""Video streaming endpoints for real-time camera access.

Provides video streaming capabilities including:
- MJPEG streaming over HTTP for real-time video
- Frame rate control and quality settings
- Camera status and health monitoring
- Rate limiting and access control
"""

import asyncio
import logging
import time
from typing import Any, Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import vision module
try:
    from backend.vision import CameraStatus, VisionConfig, VisionModule
except ImportError:
    # Fallback for development/testing
    try:
        from backend.vision import CameraStatus, VisionConfig, VisionModule
    except ImportError:
        # Another fallback for direct execution
        import os
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from vision import CameraStatus, VisionModule

from backend.api.dependencies import ApplicationState, get_app_state
from backend.api.middleware.rate_limit import rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Video Streaming"])

# Global streaming state
_streaming_clients = set()
_stream_stats = {
    "active_streams": 0,
    "total_frames_served": 0,
    "start_time": time.time(),
    "last_frame_time": 0.0,
}


class StreamingError(Exception):
    """Base exception for streaming errors."""

    pass


def get_vision_module(
    app_state: ApplicationState = Depends(get_app_state),
) -> VisionModule:
    """Get or create vision module instance."""
    if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
        # Initialize vision module with default config
        vision_config = {
            "camera_device_id": 0,
            "camera_backend": "auto",
            "camera_resolution": (1920, 1080),
            "camera_fps": 30,
            "target_fps": 30,
            "enable_threading": True,
            "enable_table_detection": False,  # Disable for streaming to reduce load
            "enable_ball_detection": False,
            "enable_cue_detection": False,
            "enable_tracking": False,
            "debug_mode": False,
        }

        try:
            app_state.vision_module = VisionModule(vision_config)
            logger.info("Vision module initialized for streaming")
        except Exception as e:
            logger.error(f"Failed to initialize vision module: {e}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Vision Module Unavailable",
                    "message": "Failed to initialize camera system",
                    "code": "STREAM_001",
                    "details": {"error": str(e)},
                },
            )

    return app_state.vision_module


async def generate_mjpeg_stream(
    vision_module: VisionModule,
    quality: int = 80,
    max_fps: int = 30,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> bytes:
    """Generate MJPEG stream from camera frames.

    Args:
        vision_module: Vision module instance
        quality: JPEG compression quality (1-100)
        max_fps: Maximum frame rate
        max_width: Maximum frame width for scaling
        max_height: Maximum frame height for scaling

    Yields:
        MJPEG frame data
    """
    client_id = id(asyncio.current_task())
    _streaming_clients.add(client_id)
    _stream_stats["active_streams"] = len(_streaming_clients)

    logger.info(f"Starting MJPEG stream for client {client_id}")

    try:
        # Start camera capture if not already running
        if not vision_module.camera.is_connected():
            if not vision_module.start_capture():
                raise StreamingError("Failed to start camera capture")

        frame_interval = 1.0 / max_fps if max_fps > 0 else 0
        last_frame_time = 0

        while client_id in _streaming_clients:
            try:
                current_time = time.time()

                # Rate limiting
                if (
                    frame_interval > 0
                    and (current_time - last_frame_time) < frame_interval
                ):
                    await asyncio.sleep(0.01)
                    continue

                # Get latest frame from vision module
                frame = vision_module.get_current_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Resize frame if requested
                if max_width or max_height:
                    h, w = frame.shape[:2]

                    # Calculate scaling factor
                    scale_x = max_width / w if max_width else 1.0
                    scale_y = max_height / h if max_height else 1.0
                    scale = min(scale_x, scale_y)

                    if scale < 1.0:
                        new_width = int(w * scale)
                        new_height = int(h * scale)
                        frame = cv2.resize(frame, (new_width, new_height))

                # Encode frame as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, buffer = cv2.imencode(".jpg", frame, encode_params)

                if not success:
                    logger.warning("Failed to encode frame as JPEG")
                    await asyncio.sleep(0.01)
                    continue

                # Format as MJPEG
                frame_data = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
                    + frame_data
                    + b"\r\n"
                )

                # Update statistics
                _stream_stats["total_frames_served"] += 1
                _stream_stats["last_frame_time"] = current_time
                last_frame_time = current_time

            except Exception as e:
                logger.error(f"Error generating frame for client {client_id}: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Stream error for client {client_id}: {e}")
        raise StreamingError(f"Stream generation failed: {e}")

    finally:
        # Cleanup
        _streaming_clients.discard(client_id)
        _stream_stats["active_streams"] = len(_streaming_clients)
        logger.info(f"MJPEG stream ended for client {client_id}")


@router.get("/video")
@rate_limit(requests_per_minute=60, burst_size=10)  # Rate limiting for streaming
async def video_stream(
    request: Request,
    quality: int = Query(80, ge=1, le=100, description="JPEG quality (1-100)"),
    fps: int = Query(30, ge=1, le=60, description="Maximum frame rate"),
    width: Optional[int] = Query(
        None, ge=160, le=3840, description="Maximum frame width"
    ),
    height: Optional[int] = Query(
        None, ge=120, le=2160, description="Maximum frame height"
    ),
    vision_module: VisionModule = Depends(get_vision_module),
) -> StreamingResponse:
    """Live video streaming endpoint using MJPEG over HTTP.

    Provides real-time video stream from the camera with configurable
    quality and frame rate settings. Supports automatic scaling and
    efficient MJPEG compression.

    Query Parameters:
        quality: JPEG compression quality (1-100, default: 80)
        fps: Maximum frame rate (1-60, default: 30)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        Streaming response with MJPEG video data

    Headers:
        Content-Type: multipart/x-mixed-replace; boundary=frame
        Cache-Control: no-cache
        Connection: close
    """
    try:
        # Check camera status
        camera_status = vision_module.camera.get_status()
        if camera_status == CameraStatus.ERROR:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Camera Error",
                    "message": "Camera is in error state",
                    "code": "STREAM_002",
                    "details": {"status": camera_status.value},
                },
            )

        if camera_status == CameraStatus.DISCONNECTED:
            # Try to start capture
            if not vision_module.start_capture():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Camera Unavailable",
                        "message": "Unable to connect to camera",
                        "code": "STREAM_003",
                        "details": {"status": camera_status.value},
                    },
                )

        logger.info(
            f"Starting video stream: quality={quality}, fps={fps}, size={width}x{height}"
        )

        # Create streaming response
        return StreamingResponse(
            generate_mjpeg_stream(
                vision_module=vision_module,
                quality=quality,
                max_fps=fps,
                max_width=width,
                max_height=height,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video stream setup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Stream Setup Failed",
                "message": "Unable to initialize video stream",
                "code": "STREAM_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/video/status")
async def stream_status(
    vision_module: VisionModule = Depends(get_vision_module),
) -> dict[str, Any]:
    """Get video streaming status and statistics.

    Returns comprehensive information about the streaming system including
    camera status, active streams, performance metrics, and health data.

    Returns:
        Dictionary containing streaming status and statistics
    """
    try:
        # Get camera health
        camera_health = vision_module.camera.get_health()
        camera_info = vision_module.camera.get_camera_info()
        vision_stats = vision_module.get_statistics()

        # Calculate streaming statistics
        uptime = time.time() - _stream_stats["start_time"]
        avg_fps = 0.0
        if uptime > 0 and _stream_stats["total_frames_served"] > 0:
            avg_fps = _stream_stats["total_frames_served"] / uptime

        return {
            "camera": {
                "status": camera_health.status.value,
                "connected": vision_module.camera.is_connected(),
                "info": camera_info,
                "health": {
                    "frames_captured": camera_health.frames_captured,
                    "frames_dropped": camera_health.frames_dropped,
                    "fps": camera_health.fps,
                    "error_count": camera_health.error_count,
                    "last_error": camera_health.last_error,
                    "uptime": camera_health.uptime,
                },
            },
            "streaming": {
                "active_streams": _stream_stats["active_streams"],
                "total_frames_served": _stream_stats["total_frames_served"],
                "avg_fps": avg_fps,
                "uptime": uptime,
                "last_frame_time": _stream_stats["last_frame_time"],
            },
            "vision": {
                "processing_fps": vision_stats.get("avg_fps", 0.0),
                "frames_processed": vision_stats.get("frames_processed", 0),
                "frames_dropped": vision_stats.get("frames_dropped", 0),
                "avg_processing_time_ms": vision_stats.get(
                    "avg_processing_time_ms", 0.0
                ),
                "is_running": vision_stats.get("is_running", False),
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Stream status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Status Check Failed",
                "message": "Unable to retrieve streaming status",
                "code": "STREAM_004",
                "details": {"error": str(e)},
            },
        )


@router.post("/video/start")
async def start_video_capture(
    vision_module: VisionModule = Depends(get_vision_module),
) -> dict[str, Any]:
    """Start video capture and processing.

    Initializes the camera and begins frame capture. This endpoint
    should be called before attempting to stream video.

    Returns:
        Dictionary containing operation status and camera information
    """
    try:
        if vision_module.camera.is_connected():
            return {
                "status": "already_running",
                "message": "Video capture is already active",
                "camera_info": vision_module.camera.get_camera_info(),
            }

        logger.info("Starting video capture via API request")

        if vision_module.start_capture():
            # Wait a moment for camera to stabilize
            await asyncio.sleep(0.5)

            return {
                "status": "started",
                "message": "Video capture started successfully",
                "camera_info": vision_module.camera.get_camera_info(),
                "camera_health": vision_module.camera.get_health().__dict__,
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Capture Start Failed",
                    "message": "Unable to start video capture",
                    "code": "STREAM_005",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start video capture failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Capture Start Error",
                "message": "Error starting video capture",
                "code": "STREAM_001",
                "details": {"error": str(e)},
            },
        )


@router.post("/video/stop")
async def stop_video_capture(
    vision_module: VisionModule = Depends(get_vision_module),
) -> dict[str, Any]:
    """Stop video capture and processing.

    Stops the camera capture and disconnects from the camera device.
    All active streaming connections will be terminated.

    Returns:
        Dictionary containing operation status
    """
    try:
        logger.info("Stopping video capture via API request")

        # Stop any active streams
        _streaming_clients.clear()
        _stream_stats["active_streams"] = 0

        # Stop vision module capture
        vision_module.stop_capture()

        return {
            "status": "stopped",
            "message": "Video capture stopped successfully",
            "active_streams_terminated": len(_streaming_clients),
        }

    except Exception as e:
        logger.error(f"Stop video capture failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Capture Stop Error",
                "message": "Error stopping video capture",
                "code": "STREAM_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/video/frame")
@rate_limit(
    requests_per_minute=120, burst_size=20
)  # Higher rate limit for single frames
async def get_single_frame(
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
    width: Optional[int] = Query(
        None, ge=160, le=3840, description="Maximum frame width"
    ),
    height: Optional[int] = Query(
        None, ge=120, le=2160, description="Maximum frame height"
    ),
    vision_module: VisionModule = Depends(get_vision_module),
):
    """Get a single frame from the camera as JPEG.

    Captures and returns a single frame from the camera, useful for
    snapshots or testing without setting up a continuous stream.

    Query Parameters:
        quality: JPEG compression quality (1-100, default: 90)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        JPEG image data
    """
    try:
        # Ensure camera is running
        if not vision_module.camera.is_connected():
            if not vision_module.start_capture():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Camera Unavailable",
                        "message": "Unable to connect to camera",
                        "code": "STREAM_003",
                    },
                )

            # Wait for camera to stabilize
            await asyncio.sleep(0.5)

        # Get current frame
        frame = vision_module.get_current_frame()
        if frame is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "No Frame Available",
                    "message": "Camera is not providing frames",
                    "code": "STREAM_006",
                },
            )

        # Resize frame if requested
        if width or height:
            h, w = frame.shape[:2]

            # Calculate scaling factor
            scale_x = width / w if width else 1.0
            scale_y = height / h if height else 1.0
            scale = min(scale_x, scale_y)

            if scale < 1.0:
                new_width = int(w * scale)
                new_height = int(h * scale)
                frame = cv2.resize(frame, (new_width, new_height))

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if not success:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Frame Encoding Failed",
                    "message": "Unable to encode frame as JPEG",
                    "code": "STREAM_007",
                },
            )

        # Return JPEG data
        from fastapi.responses import Response

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single frame capture failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Frame Capture Error",
                "message": "Error capturing single frame",
                "code": "STREAM_001",
                "details": {"error": str(e)},
            },
        )
