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
    from ...vision import CameraStatus, VisionConfig, VisionModule
except ImportError:
    # Fallback for development/testing
    try:
        from ...vision import CameraStatus, VisionConfig, VisionModule
    except ImportError:
        # Another fallback for direct execution
        import os
        import sys

        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from vision import CameraStatus, VisionModule

from ..dependencies import ApplicationState, get_app_state

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


async def get_vision_module(
    app_state: ApplicationState = Depends(get_app_state),
) -> VisionModule:
    """Get the shared vision module instance.

    The vision module is initialized at application startup with full resolution
    and OpenCV processing enabled. This ensures:
    1. Only ONE camera instance is created (no conflicts)
    2. OpenCV processing has priority (runs continuously)
    3. Web streaming reads frames from the same instance via get_current_frame()
    """
    logger.debug("get_vision_module called")

    if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
        logger.error("Vision module not initialized - should be created at startup")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Vision Module Unavailable",
                "message": "Camera system not initialized. Camera may not be available on this system.",
                "code": "STREAM_001",
            },
        )

    logger.debug("Using shared vision module instance")
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

    logger.info(
        f"Starting MJPEG stream for client {client_id} (quality={quality}, max_fps={max_fps})"
    )

    try:
        # Verify camera is running (it should be started at application startup)
        logger.debug(f"Checking camera connection status for client {client_id}")
        if not vision_module.camera.is_connected():
            logger.error(
                f"Camera not connected for client {client_id}. Camera should be running at startup."
            )
            raise StreamingError(
                "Camera not available. The camera should be initialized at application startup."
            )
        logger.debug(f"Camera is connected for client {client_id}")

        frame_interval = 1.0 / max_fps if max_fps > 0 else 0
        last_frame_time = 0
        frame_count = 0

        logger.info(f"Entering stream loop for client {client_id}")

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
                logger.debug(f"Getting frame for client {client_id}")
                frame = vision_module.get_current_frame()
                if frame is None:
                    logger.debug(f"No frame available for client {client_id}")
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.debug(f"Client {client_id} processed {frame_count} frames")

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
            # Camera should be started at application startup
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Camera Unavailable",
                    "message": "Camera is not running. Camera should be initialized at application startup.",
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
    app_state: ApplicationState = Depends(get_app_state),
) -> dict[str, Any]:
    """Get video streaming status and statistics.

    Returns comprehensive information about the streaming system including
    camera status, active streams, performance metrics, and health data.

    Returns:
        Dictionary containing streaming status and statistics
    """
    try:
        # Check if vision module exists, but don't create it
        vision_module = (
            app_state.vision_module if hasattr(app_state, "vision_module") else None
        )

        if not vision_module:
            # Return minimal status when vision module not initialized
            return {
                "camera": {
                    "status": "not_initialized",
                    "connected": False,
                    "info": {},
                    "health": {
                        "frames_captured": 0,
                        "frames_dropped": 0,
                        "fps": 0.0,
                        "error_count": 0,
                        "last_error": None,
                        "uptime": 0.0,
                    },
                },
                "streaming": {
                    "active_streams": _stream_stats["active_streams"],
                    "total_frames_served": _stream_stats["total_frames_served"],
                    "avg_fps": 0.0,
                    "uptime": time.time() - _stream_stats["start_time"],
                    "last_frame_time": _stream_stats["last_frame_time"],
                },
                "vision": {
                    "processing_fps": 0.0,
                    "frames_processed": 0,
                    "frames_dropped": 0,
                    "avg_processing_time_ms": 0.0,
                    "is_running": False,
                },
                "timestamp": time.time(),
            }

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
    """Check video capture status.

    The camera is automatically started at application startup for continuous
    OpenCV processing. This endpoint returns the current status.

    Returns:
        Dictionary containing camera status and information
    """
    try:
        if vision_module.camera.is_connected():
            return {
                "status": "running",
                "message": "Video capture is active (started at application startup)",
                "camera_info": vision_module.camera.get_camera_info(),
                "camera_health": vision_module.camera.get_health().__dict__,
            }
        else:
            return {
                "status": "not_running",
                "message": "Camera is not connected. Check system logs for camera initialization errors.",
                "camera_info": {},
            }

    except Exception as e:
        logger.error(f"Check video capture status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Status Check Error",
                "message": "Error checking video capture status",
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
        # Verify camera is running (should be started at application startup)
        if not vision_module.camera.is_connected():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Camera Unavailable",
                    "message": "Camera is not running. Camera should be initialized at application startup.",
                    "code": "STREAM_003",
                },
            )

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
