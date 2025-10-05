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
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import WebSocket broadcaster for real-time frame streaming
from ..websocket import message_broadcaster

# Import vision module
try:
    from ...streaming.enhanced_camera_module import (
        EnhancedCameraConfig,
        EnhancedCameraModule,
    )
    from ...vision.capture import CameraHealth, CameraStatus
except ImportError:
    # Fallback for when relative imports don't work
    from backend.streaming.enhanced_camera_module import (
        EnhancedCameraConfig,
        EnhancedCameraModule,
    )
    from backend.vision.capture import CameraHealth, CameraStatus

from ..dependencies import ApplicationState, get_app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Video Streaming"])


# Compatibility wrapper to make EnhancedCameraModule work with existing API
class CameraModuleAdapter:
    """Adapter to make EnhancedCameraModule compatible with DirectCameraModule interface."""

    def __init__(self, enhanced_module: EnhancedCameraModule):
        """Initialize adapter with enhanced camera module.

        Args:
            enhanced_module: The EnhancedCameraModule instance to wrap
        """
        self._module = enhanced_module
        self.camera = self  # Self-reference for compatibility

    def start_capture(self) -> bool:
        """Start camera capture."""
        return self._module.start_capture()

    def stop_capture(self):
        """Stop camera capture."""
        self._module.stop_capture()

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._module.running

    def get_status(self) -> CameraStatus:
        """Get camera status."""
        if self._module.running:
            return CameraStatus.CONNECTED
        return CameraStatus.DISCONNECTED

    def get_camera_info(self) -> dict:
        """Get camera information."""
        stats = self._module.get_statistics()
        return {
            "device_id": self._module.config.device_id,
            "resolution": stats.get("resolution", (0, 0)),
            "fps": stats.get("fps", 0),
            "backend": "enhanced",
            "fisheye_correction": stats.get("fisheye_correction_enabled", False),
            "preprocessing": stats.get("preprocessing_enabled", False),
        }

    def get_health(self) -> CameraHealth:
        """Get camera health information."""
        stats = self._module.get_statistics()
        return CameraHealth(
            status=self.get_status(),
            frames_captured=0,  # EnhancedCameraModule doesn't track this
            frames_dropped=0,
            fps=stats.get("fps", 0.0),
            last_frame_time=time.time(),
            error_count=0,
            last_error=None,
            connection_attempts=0,
            uptime=0.0,
        )

    def get_frame(self, processed: bool = True) -> Optional[np.ndarray]:
        """Get current frame for vision processing.

        Args:
            processed: If True, return processed frame. If False, return raw frame.

        Returns:
            Current frame or None if not available.
        """
        return self._module.get_frame(processed=processed)

    def get_frame_for_streaming(
        self, scale: float = 0.5, raw: bool = False
    ) -> Optional[np.ndarray]:
        """Get frame for streaming with optional downsampling.

        Args:
            scale: Scaling factor for downsampling
            raw: If True, get raw frame without fisheye correction or processing
        """
        frame = self._module.get_frame(processed=not raw)

        if frame is None:
            return None

        # Apply scaling if requested
        if scale != 1.0:
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        return frame

    def get_current_frame(self, raw: bool = False) -> Optional[np.ndarray]:
        """Get current frame.

        Args:
            raw: If True, get raw frame without fisheye correction or processing
        """
        return self._module.get_frame(processed=not raw)

    def get_statistics(self) -> dict:
        """Get module statistics."""
        stats = self._module.get_statistics()
        return {
            "is_running": stats.get("running", False),
            "frames_processed": 0,
            "frames_dropped": 0,
            "avg_fps": stats.get("fps", 0.0),
            "avg_processing_time_ms": 0.0,
        }


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
) -> CameraModuleAdapter:
    """Get or lazily create the shared camera module instance.

    Creates camera module on first access and reuses it for all subsequent requests.
    This ensures:
    1. Only ONE camera instance is created (no conflicts)
    2. Server startup is fast (camera init happens on first access)
    3. Enhanced streaming with fisheye correction and preprocessing

    Note: Using EnhancedCameraModule with fisheye correction and preprocessing
    capabilities for improved image quality.
    """
    logger.debug("get_vision_module called")

    if not hasattr(app_state, "vision_module") or app_state.vision_module is None:
        logger.info(
            "Creating shared EnhancedCameraModule instance (lazy initialization)"
        )

        # Check if calibration file exists
        import os

        calibration_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "calibration/camera_fisheye_default.yaml",
        )
        enable_fisheye = os.path.exists(calibration_path)

        if enable_fisheye:
            logger.info(f"Found calibration file at {calibration_path}")
        else:
            logger.warning(
                f"Calibration file not found at {calibration_path}, disabling fisheye correction"
            )

        # Configure enhanced camera with fisheye correction and preprocessing
        camera_config = EnhancedCameraConfig(
            device_id=1,
            resolution=(3840, 2160),  # Request 4K resolution
            fps=30,
            enable_fisheye_correction=enable_fisheye,  # Enable if calibration file exists
            calibration_file=calibration_path if enable_fisheye else None,
            enable_table_crop=False,  # Disabled for testing - was cropping to 5x5!
            enable_preprocessing=False,  # Disable preprocessing to preserve natural colors
            brightness=0,
            contrast=1.0,
            enable_clahe=False,  # CLAHE was washing out colors
            clahe_clip_limit=2.0,
            clahe_grid_size=8,
            buffer_size=1,
        )

        try:
            logger.info("[get_vision_module] Initializing EnhancedCameraModule...")

            # Define async callback for WebSocket frame broadcasting
            async def on_frame_captured(frame_data, width, height):
                """Broadcast captured frames to WebSocket clients."""
                try:
                    await message_broadcaster.broadcast_frame(
                        image_data=frame_data,
                        width=width,
                        height=height,
                        quality=75,  # JPEG quality for WebSocket streaming
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting frame to WebSocket: {e}")

            # Run the blocking EnhancedCameraModule initialization in a thread pool
            logger.info(
                "[get_vision_module] Creating EnhancedCameraModule in executor..."
            )
            loop = asyncio.get_event_loop()
            enhanced_module = await loop.run_in_executor(
                None,
                lambda: EnhancedCameraModule(
                    camera_config,
                    event_loop=loop,
                    frame_callback=on_frame_captured,
                ),
            )
            logger.info(
                "[get_vision_module] EnhancedCameraModule created with WebSocket integration!"
            )
            logger.info(
                f"[get_vision_module] Camera resolution: {enhanced_module.config.resolution}"
            )

            # Wrap in compatibility adapter
            app_state.vision_module = CameraModuleAdapter(enhanced_module)
            logger.info("[get_vision_module] Camera module wrapped in adapter")

            logger.info("[get_vision_module] Starting camera capture...")
            # Start camera capture
            success = await loop.run_in_executor(
                None, app_state.vision_module.start_capture
            )
            logger.info(f"[get_vision_module] start_capture returned: {success}")

            if success:
                logger.info(
                    "Shared camera module created and camera started successfully"
                )
            else:
                logger.error("Camera capture failed to start")
                app_state.vision_module = None
                raise HTTPException(
                    status_code=503,
                    detail="Camera capture failed to start",
                )
        except Exception as e:
            logger.error(f"Failed to initialize camera module: {e}", exc_info=True)
            app_state.vision_module = None
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize camera system: {str(e)}",
            )

    logger.debug("Using shared camera module instance")
    return app_state.vision_module


async def generate_mjpeg_stream(
    vision_module: CameraModuleAdapter,
    quality: int = 80,
    max_fps: int = 30,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    raw: bool = False,
) -> bytes:
    """Generate MJPEG stream from camera frames.

    Args:
        vision_module: Vision module instance
        quality: JPEG compression quality (1-100)
        max_fps: Maximum frame rate
        max_width: Maximum frame width for scaling
        max_height: Maximum frame height for scaling
        raw: If True, stream raw frames without fisheye correction

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
        # Camera should already be started by get_vision_module() lazy init
        logger.debug(f"Checking camera connection status for client {client_id}")
        if not vision_module.camera.is_connected():
            logger.warning(
                f"Camera not connected for client {client_id}, this shouldn't happen after lazy init"
            )
            raise StreamingError("Camera not connected")
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

                # Get latest frame from vision module at full resolution
                logger.debug(f"Getting frame for client {client_id}")
                frame = vision_module.get_frame_for_streaming(scale=1.0, raw=raw)
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
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
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
                detail=f"Camera is in error state (status: {camera_status.value})",
            )

        if camera_status == CameraStatus.DISCONNECTED:
            raise HTTPException(
                status_code=503,
                detail=f"Camera is not connected (status: {camera_status.value})",
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
            detail=f"Unable to initialize video stream: {str(e)}",
        )


@router.get("/video/raw")
async def video_stream_raw(
    request: Request,
    quality: int = Query(80, ge=1, le=100, description="JPEG quality (1-100)"),
    fps: int = Query(30, ge=1, le=60, description="Maximum frame rate"),
    width: Optional[int] = Query(
        None, ge=160, le=3840, description="Maximum frame width"
    ),
    height: Optional[int] = Query(
        None, ge=120, le=2160, description="Maximum frame height"
    ),
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
) -> StreamingResponse:
    """Live RAW video streaming endpoint (no fisheye correction or processing).

    Provides real-time video stream from the camera WITHOUT any fisheye correction
    or image processing. Useful for debugging and comparing raw vs corrected streams.

    Query Parameters:
        quality: JPEG compression quality (1-100, default: 80)
        fps: Maximum frame rate (1-60, default: 30)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        Streaming response with MJPEG video data (raw, uncorrected)

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
                detail=f"Camera is in error state (status: {camera_status.value})",
            )

        if camera_status == CameraStatus.DISCONNECTED:
            raise HTTPException(
                status_code=503,
                detail=f"Camera is not connected (status: {camera_status.value})",
            )

        logger.info(
            f"Starting RAW video stream: quality={quality}, fps={fps}, size={width}x{height}"
        )

        # Create streaming response with raw=True
        return StreamingResponse(
            generate_mjpeg_stream(
                vision_module=vision_module,
                quality=quality,
                max_fps=fps,
                max_width=width,
                max_height=height,
                raw=True,  # Get raw frames without fisheye correction
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
        logger.error(f"RAW video stream setup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unable to initialize RAW video stream: {str(e)}",
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
            detail=f"Unable to retrieve streaming status: {str(e)}",
        )


@router.post("/video/start")
async def start_video_capture(
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
) -> dict[str, Any]:
    """Get video capture status.

    The camera is started automatically when vision module is first accessed (lazy initialization).
    This endpoint returns the current status.

    Returns:
        Dictionary containing camera status and information
    """
    try:
        if vision_module.camera.is_connected():
            return {
                "status": "running",
                "message": "Video capture is active",
                "camera_info": vision_module.camera.get_camera_info(),
                "camera_health": vision_module.camera.get_health().__dict__,
            }
        else:
            return {
                "status": "not_running",
                "message": "Camera is not connected",
                "camera_info": {},
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Check video capture status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking video capture status: {str(e)}",
        )


@router.post("/video/stop")
async def stop_video_capture(
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
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
            detail=f"Error stopping video capture: {str(e)}",
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
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
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
        # Camera should already be started by lazy init
        if not vision_module.camera.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Camera is not connected",
            )

        # Get current frame
        frame = vision_module.get_current_frame()
        if frame is None:
            raise HTTPException(
                status_code=503,
                detail="Camera is not providing frames",
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
                detail="Unable to encode frame as JPEG",
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
            detail=f"Error capturing single frame: {str(e)}",
        )


@router.get("/video/frame/raw")
async def get_single_frame_raw(
    quality: int = Query(90, ge=1, le=100, description="JPEG quality (1-100)"),
    width: Optional[int] = Query(
        None, ge=160, le=3840, description="Maximum frame width"
    ),
    height: Optional[int] = Query(
        None, ge=120, le=2160, description="Maximum frame height"
    ),
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
):
    """Get a single RAW frame from the camera as JPEG (no fisheye correction).

    Captures and returns a single frame from the camera WITHOUT any fisheye
    correction or processing. Useful for debugging and comparing raw vs corrected frames.

    Query Parameters:
        quality: JPEG compression quality (1-100, default: 90)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        JPEG image data (raw, uncorrected)
    """
    try:
        # Camera should already be started by lazy init
        if not vision_module.camera.is_connected():
            raise HTTPException(
                status_code=503,
                detail="Camera is not connected",
            )

        # Get current RAW frame (no fisheye correction)
        frame = vision_module.get_current_frame(raw=True)
        if frame is None:
            raise HTTPException(
                status_code=503,
                detail="Camera is not providing frames",
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
                detail="Unable to encode frame as JPEG",
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
        logger.error(f"Single RAW frame capture failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error capturing single RAW frame: {str(e)}",
        )
