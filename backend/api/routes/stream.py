"""Video streaming endpoints for real-time camera access.

Provides video streaming capabilities including:
- MJPEG streaming over HTTP for real-time video
- Frame rate control and quality settings
- Camera status and health monitoring
- Rate limiting and access control
- Shared memory based streaming for high concurrency support
"""

import asyncio
import logging

# Import vision module - ensure backend dir is in path
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

# Import WebSocket broadcaster for real-time frame streaming
from ..websocket import message_broadcaster

backend_dir = Path(__file__).parent.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.streaming.enhanced_camera_module import (
    EnhancedCameraConfig,
    EnhancedCameraModule,
)
from backend.video.ipc.shared_memory import SharedMemoryFrameReader
from backend.vision.capture import CameraHealth, CameraStatus

from ..dependencies import ApplicationState, get_app_state, get_config_module

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Video Streaming"])


def _get_stream_config(app_state: ApplicationState) -> dict:
    """Get stream configuration from config module.

    Args:
        app_state: Application state containing config module

    Returns:
        Stream configuration dictionary with defaults if config not available
    """
    if app_state.config_module:
        config = app_state.config_module.get("api.stream", {})
        logger.debug(
            f"_get_stream_config: got {type(config)} with keys: {list(config.keys()) if isinstance(config, dict) else 'not a dict'}"
        )
        return config
    # Return empty dict if config not available - will use defaults
    logger.warning("_get_stream_config: No config module available")
    return {}


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
    logger.info("=== get_vision_module called ===")
    print("=== get_vision_module called ===", flush=True)

    # Use a separate attribute for streaming camera module to avoid conflicts with VisionModule
    streaming_module_attr = "streaming_camera_module"

    logger.info(
        f"=== Checking {streaming_module_attr}: hasattr={hasattr(app_state, streaming_module_attr)}, is_none={getattr(app_state, streaming_module_attr, None) is None} ==="
    )
    print(
        f"=== Checking {streaming_module_attr}: hasattr={hasattr(app_state, streaming_module_attr)}, is_none={getattr(app_state, streaming_module_attr, None) is None} ===",
        flush=True,
    )

    if (
        not hasattr(app_state, streaming_module_attr)
        or getattr(app_state, streaming_module_attr, None) is None
    ):
        logger.info(
            "Creating shared EnhancedCameraModule instance (lazy initialization)"
        )
        print(
            "Creating shared EnhancedCameraModule instance (lazy initialization)",
            flush=True,
        )

        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        logger.info(
            f"[get_vision_module] stream_config keys: {list(stream_config.keys()) if stream_config else 'None'}"
        )
        # Use vision.camera.* as fallback source of truth for camera settings
        camera_cfg = stream_config.get("camera", {})
        logger.info(
            f"[get_vision_module] camera_cfg from api.stream (fallback only): {camera_cfg}"
        )
        processing_cfg = stream_config.get("processing", {})

        # Check if calibration file exists
        import os

        calibration_file = processing_cfg.get(
            "calibration_file", "calibration/camera_fisheye_default.yaml"
        )
        calibration_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            calibration_file,
        )
        enable_fisheye = processing_cfg.get(
            "enable_fisheye_correction", True
        ) and os.path.exists(calibration_path)

        if enable_fisheye:
            logger.info(f"Found calibration file at {calibration_path}")
        else:
            logger.warning(
                f"Calibration file not found at {calibration_path}, disabling fisheye correction"
            )

        # Configure enhanced camera from config using the new from_config method
        # Note: This uses api.stream config section which has different structure
        # than streaming section. We'll use the streaming config if available,
        # otherwise fall back to api.stream config for backwards compatibility.

        # Try to load from streaming config first (new structure)
        if hasattr(app_state, "config_module") and app_state.config_module:
            try:
                # First check if streaming config exists and has the right structure
                streaming_test = app_state.config_module.get("streaming", {})
                logger.info(f"Streaming config check: {streaming_test}")

                camera_config = EnhancedCameraConfig.from_config(
                    app_state.config_module
                )
                logger.info(
                    f"Loaded EnhancedCameraConfig from streaming configuration: device_id={camera_config.device_id}, resolution={camera_config.resolution}"
                )

                # Validate that we got the video file path, not the default
                if camera_config.device_id == 1 and camera_cfg.get("device_id") != 1:
                    logger.warning(
                        f"from_config returned default device_id=1, but api.stream has device_id={camera_cfg.get('device_id')}. Using fallback."
                    )
                    raise ValueError("Config mismatch detected, using fallback")

            except Exception as e:
                logger.warning(
                    f"Failed to load from streaming config: {e}, falling back to manual config"
                )
                logger.exception("Full exception:")
                # Fall back to manual configuration using vision.camera.* as source of truth
                # Get values from vision.camera.* hierarchy, with api.stream as last resort fallback
                vision_device_id = app_state.config_module.get(
                    "vision.camera.device_id", None
                )
                vision_resolution = app_state.config_module.get(
                    "vision.camera.resolution", None
                )
                vision_fps = app_state.config_module.get("vision.camera.fps", None)
                vision_buffer_size = app_state.config_module.get(
                    "vision.camera.buffer_size", None
                )

                # Use vision.camera.* values if available, otherwise fall back to api.stream
                device_id = (
                    vision_device_id
                    if vision_device_id is not None
                    else camera_cfg.get("device_id", 0)
                )
                resolution = (
                    vision_resolution
                    if vision_resolution is not None
                    else camera_cfg.get("resolution", [1920, 1080])
                )
                fps = (
                    vision_fps if vision_fps is not None else camera_cfg.get("fps", 30)
                )
                buffer_size = (
                    vision_buffer_size
                    if vision_buffer_size is not None
                    else camera_cfg.get("buffer_size", 1)
                )

                logger.info(
                    f"Fallback config: device_id={device_id}, type={type(device_id)}, resolution={resolution}, fps={fps}"
                )
                camera_config = EnhancedCameraConfig(
                    device_id=device_id,
                    resolution=tuple(resolution) if resolution else None,
                    fps=fps,
                    enable_fisheye_correction=enable_fisheye,
                    calibration_file=calibration_path if enable_fisheye else None,
                    fisheye_alpha=1.0,
                    enable_table_crop=processing_cfg.get("enable_table_crop", False),
                    table_crop_hsv_lower=(35, 40, 40),
                    table_crop_hsv_upper=(85, 255, 255),
                    table_crop_morphology_kernel_size=5,
                    table_crop_padding_ratio=0.05,
                    enable_preprocessing=processing_cfg.get(
                        "enable_preprocessing", False
                    ),
                    brightness=camera_cfg.get("brightness", 0),
                    contrast=camera_cfg.get("contrast", 1.0),
                    enable_clahe=processing_cfg.get("enable_clahe", False),
                    clahe_clip_limit=processing_cfg.get("clahe_clip_limit", 2.0),
                    clahe_grid_size=processing_cfg.get("clahe_grid_size", 8),
                    default_jpeg_quality=80,
                    enable_gpu=False,
                    buffer_size=buffer_size,
                    startup_timeout_seconds=5.0,
                    thread_join_timeout_seconds=2.0,
                )
        else:
            # No config module, use hardcoded defaults matching vision.camera.* defaults
            logger.warning("No config module available, using hardcoded defaults")
            # Use vision.camera.* default values (matching default.json)
            device_id = camera_cfg.get(
                "device_id", 0
            )  # Match vision.camera.device_id default
            resolution = camera_cfg.get(
                "resolution", [1920, 1080]
            )  # Match vision.camera.resolution default
            fps = camera_cfg.get("fps", 30)
            buffer_size = camera_cfg.get("buffer_size", 1)
            logger.info(
                f"No config module - device_id={device_id}, type={type(device_id)}, resolution={resolution}, fps={fps}"
            )
            camera_config = EnhancedCameraConfig(
                device_id=device_id,
                resolution=tuple(resolution) if resolution else None,
                fps=fps,
                enable_fisheye_correction=enable_fisheye,
                calibration_file=calibration_path if enable_fisheye else None,
                fisheye_alpha=1.0,
                enable_table_crop=processing_cfg.get("enable_table_crop", False),
                table_crop_hsv_lower=(35, 40, 40),
                table_crop_hsv_upper=(85, 255, 255),
                table_crop_morphology_kernel_size=5,
                table_crop_padding_ratio=0.05,
                enable_preprocessing=processing_cfg.get("enable_preprocessing", False),
                brightness=camera_cfg.get("brightness", 0),
                contrast=camera_cfg.get("contrast", 1.0),
                enable_clahe=processing_cfg.get("enable_clahe", False),
                clahe_clip_limit=processing_cfg.get("clahe_clip_limit", 2.0),
                clahe_grid_size=processing_cfg.get("clahe_grid_size", 8),
                default_jpeg_quality=80,
                enable_gpu=False,
                buffer_size=buffer_size,
                startup_timeout_seconds=5.0,
                thread_join_timeout_seconds=2.0,
            )

        try:
            logger.info("[get_vision_module] Initializing EnhancedCameraModule...")

            # Define async callback for WebSocket frame broadcasting
            async def on_frame_captured(frame_data, width, height):
                """Broadcast captured frames to WebSocket clients."""
                try:
                    quality_cfg = stream_config.get("quality", {})
                    await message_broadcaster.broadcast_frame(
                        image_data=frame_data,
                        width=width,
                        height=height,
                        quality=quality_cfg.get("websocket_jpeg_quality", 75),
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
            setattr(
                app_state, streaming_module_attr, CameraModuleAdapter(enhanced_module)
            )
            logger.info("[get_vision_module] Camera module wrapped in adapter")

            logger.info("[get_vision_module] Starting camera capture...")
            # Start camera capture
            streaming_module = getattr(app_state, streaming_module_attr)
            success = await loop.run_in_executor(None, streaming_module.start_capture)
            logger.info(f"[get_vision_module] start_capture returned: {success}")

            if success:
                logger.info(
                    "Shared camera module created and camera started successfully"
                )
            else:
                logger.error("Camera capture failed to start")
                setattr(app_state, streaming_module_attr, None)
                raise HTTPException(
                    status_code=503,
                    detail="Camera capture failed to start",
                )
        except Exception as e:
            logger.error(f"Failed to initialize camera module: {e}", exc_info=True)
            setattr(app_state, streaming_module_attr, None)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize camera system: {str(e)}",
            )

    logger.debug("Using shared camera module instance")
    return getattr(app_state, streaming_module_attr)


async def generate_mjpeg_stream(
    vision_module: CameraModuleAdapter,
    quality: int,
    max_fps: int,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    raw: bool = False,
    stream_config: Optional[dict] = None,
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

        # Get performance config
        perf_cfg = (stream_config or {}).get("performance", {})
        frame_log_interval = perf_cfg.get("frame_log_interval", 30)
        sleep_interval = perf_cfg.get("sleep_interval_ms", 10) / 1000.0
        resolution_cfg = (stream_config or {}).get("resolution", {})
        default_scale = resolution_cfg.get("default_scale", 1.0)

        logger.info(f"Entering stream loop for client {client_id}")

        while client_id in _streaming_clients:
            try:
                current_time = time.time()

                # Rate limiting
                if (
                    frame_interval > 0
                    and (current_time - last_frame_time) < frame_interval
                ):
                    await asyncio.sleep(sleep_interval)
                    continue

                # Get latest frame from vision module
                logger.debug(f"Getting frame for client {client_id}")
                frame = vision_module.get_frame_for_streaming(
                    scale=default_scale, raw=raw
                )
                if frame is None:
                    logger.debug(f"No frame available for client {client_id}")
                    await asyncio.sleep(sleep_interval)
                    continue

                frame_count += 1
                if frame_count % frame_log_interval == 0:
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
                    await asyncio.sleep(sleep_interval)
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
                await asyncio.sleep(sleep_interval * 10)  # Longer sleep on error

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
    app_state: ApplicationState = Depends(get_app_state),
    quality: Optional[int] = Query(None, description="JPEG quality (1-100)"),
    fps: Optional[int] = Query(None, description="Maximum frame rate"),
    width: Optional[int] = Query(None, description="Maximum frame width"),
    height: Optional[int] = Query(None, description="Maximum frame height"),
) -> StreamingResponse:
    """Live video streaming endpoint using shared memory IPC.

    Provides real-time video stream from the Video Module via shared memory IPC.
    Supports high concurrency (10+ clients) with minimal CPU overhead per client.

    Prerequisites:
        - Video Module must be running: python -m backend.video

    Query Parameters:
        quality: JPEG compression quality (uses config default if not specified)
        fps: Maximum frame rate (uses config default if not specified)
        width: Maximum frame width for scaling (optional, not yet supported)
        height: Maximum frame height for scaling (optional, not yet supported)

    Returns:
        Streaming response with MJPEG video data

    Headers:
        Content-Type: multipart/x-mixed-replace; boundary=frame
        Cache-Control: no-cache
        Connection: close
    """
    try:
        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        quality_cfg = stream_config.get("quality", {})
        framerate_cfg = stream_config.get("framerate", {})

        # Apply defaults
        quality = (
            quality
            if quality is not None
            else quality_cfg.get("default_jpeg_quality", 85)
        )
        fps = fps if fps is not None else framerate_cfg.get("default_fps", 30)

        # Note: width/height scaling not supported in shared memory mode yet
        if width or height:
            logger.warning(
                "Width/height scaling not supported in shared memory mode, "
                "parameters will be ignored"
            )

        # Validate quality
        min_quality = quality_cfg.get("min_jpeg_quality", 1)
        max_quality = quality_cfg.get("max_jpeg_quality", 100)
        if quality < min_quality or quality > max_quality:
            raise HTTPException(
                status_code=400,
                detail=f"Quality must be between {min_quality} and {max_quality}",
            )

        # Validate fps
        min_fps = framerate_cfg.get("min_fps", 1)
        max_fps_limit = framerate_cfg.get("max_fps", 60)
        if fps < min_fps or fps > max_fps_limit:
            raise HTTPException(
                status_code=400,
                detail=f"FPS must be between {min_fps} and {max_fps_limit}",
            )

        logger.info(
            f"Starting shared memory video stream: quality={quality}, fps={fps}"
        )

        # Create streaming response using shared memory
        return StreamingResponse(
            generate_mjpeg_stream_from_shm(
                request=request,
                app_state=app_state,
                quality=quality,
                fps=fps,
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
    app_state: ApplicationState = Depends(get_app_state),
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
    quality: Optional[int] = Query(None, description="JPEG quality (1-100)"),
    fps: Optional[int] = Query(None, description="Maximum frame rate"),
    width: Optional[int] = Query(None, description="Maximum frame width"),
    height: Optional[int] = Query(None, description="Maximum frame height"),
) -> StreamingResponse:
    """Live RAW video streaming endpoint (no fisheye correction or processing).

    Provides real-time video stream from the camera WITHOUT any fisheye correction
    or image processing. Useful for debugging and comparing raw vs corrected streams.

    Query Parameters:
        quality: JPEG compression quality (uses config default if not specified)
        fps: Maximum frame rate (uses config default if not specified)
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
        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        quality_cfg = stream_config.get("quality", {})
        framerate_cfg = stream_config.get("framerate", {})
        resolution_cfg = stream_config.get("resolution", {})

        # Apply defaults and validate ranges (same validation as video_stream)
        quality = (
            quality
            if quality is not None
            else quality_cfg.get("default_jpeg_quality", 80)
        )
        fps = fps if fps is not None else framerate_cfg.get("default_fps", 30)

        # Validate quality
        min_quality = quality_cfg.get("min_jpeg_quality", 1)
        max_quality = quality_cfg.get("max_jpeg_quality", 100)
        if quality < min_quality or quality > max_quality:
            raise HTTPException(
                status_code=400,
                detail=f"Quality must be between {min_quality} and {max_quality}",
            )

        # Validate fps
        min_fps = framerate_cfg.get("min_fps", 1)
        max_fps_limit = framerate_cfg.get("max_fps", 60)
        if fps < min_fps or fps > max_fps_limit:
            raise HTTPException(
                status_code=400,
                detail=f"FPS must be between {min_fps} and {max_fps_limit}",
            )

        # Validate resolution if provided
        if width is not None:
            min_width = resolution_cfg.get("min_width", 160)
            max_width = resolution_cfg.get("max_width", 3840)
            if width < min_width or width > max_width:
                raise HTTPException(
                    status_code=400,
                    detail=f"Width must be between {min_width} and {max_width}",
                )

        if height is not None:
            min_height = resolution_cfg.get("min_height", 120)
            max_height = resolution_cfg.get("max_height", 2160)
            if height < min_height or height > max_height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Height must be between {min_height} and {max_height}",
                )

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
                raw=True,
                stream_config=stream_config,
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


async def generate_mjpeg_stream_from_shm(
    request: Request,
    app_state: ApplicationState,
    quality: int,
    fps: int,
) -> bytes:
    """Generate MJPEG stream from shared memory frames.

    Each client gets an independent SharedMemoryFrameReader for zero-copy
    frame access. This enables 10+ concurrent clients without degradation.

    Args:
        request: FastAPI request for disconnect detection
        app_state: Application state for config access
        quality: JPEG compression quality (1-100)
        fps: Target frame rate

    Yields:
        MJPEG frame data
    """
    client_id = id(asyncio.current_task())
    reader: Optional[SharedMemoryFrameReader] = None

    logger.info(
        f"Starting shared memory stream for client {client_id} (quality={quality}, fps={fps})"
    )

    try:
        # Get shared memory configuration
        shm_name = app_state.config_module.get(
            "video.shared_memory_name", "billiards_video"
        )
        attach_timeout = app_state.config_module.get(
            "video.shared_memory_attach_timeout_sec", 5.0
        )

        # Create independent reader for this client
        reader = SharedMemoryFrameReader(name=shm_name)

        # Attach with timeout (raises TimeoutError if Video Module not running)
        try:
            reader.attach(timeout=attach_timeout)
            logger.info(f"Client {client_id} attached to shared memory: {shm_name}")
        except TimeoutError:
            logger.error(
                f"Client {client_id} failed to attach: Video Module not running"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Video Module is not running. "
                    "Start it with: python -m backend.video"
                ),
            )

        # Frame rate limiting
        frame_delay = 1.0 / fps if fps > 0 else 0
        last_frame_number = -1
        last_frame_time = 0
        frame_count = 0

        # Performance config
        stream_config = _get_stream_config(app_state)
        perf_cfg = stream_config.get("performance", {})
        frame_log_interval = perf_cfg.get("frame_log_interval", 30)
        sleep_interval = perf_cfg.get("sleep_interval_ms", 10) / 1000.0

        # Track this client
        _streaming_clients.add(client_id)
        _stream_stats["active_streams"] = len(_streaming_clients)

        while True:
            try:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"Client {client_id} disconnected")
                    break

                current_time = time.time()

                # Rate limiting
                if frame_delay > 0 and (current_time - last_frame_time) < frame_delay:
                    await asyncio.sleep(sleep_interval)
                    continue

                # Read frame from shared memory (non-blocking, returns None if no new frame)
                frame, metadata = reader.read_frame()

                if frame is None or (
                    metadata and metadata.frame_number == last_frame_number
                ):
                    # No new frame available
                    await asyncio.sleep(sleep_interval)
                    continue

                frame_count += 1
                if frame_count % frame_log_interval == 0:
                    logger.debug(f"Client {client_id} processed {frame_count} frames")

                # Encode frame as JPEG
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, buffer = cv2.imencode(".jpg", frame, encode_params)

                if not success:
                    logger.warning(f"Client {client_id} failed to encode frame as JPEG")
                    await asyncio.sleep(sleep_interval)
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
                if metadata:
                    last_frame_number = metadata.frame_number

            except Exception as e:
                logger.error(
                    f"Error generating frame for client {client_id}: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(sleep_interval * 10)  # Longer sleep on error

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream error for client {client_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Stream generation failed: {str(e)}",
        )

    finally:
        # Cleanup: detach from shared memory
        if reader is not None:
            try:
                reader.detach()
                logger.info(f"Client {client_id} detached from shared memory")
            except Exception as e:
                logger.error(f"Error detaching reader for client {client_id}: {e}")

        # Remove from tracking
        _streaming_clients.discard(client_id)
        _stream_stats["active_streams"] = len(_streaming_clients)
        logger.info(f"Shared memory stream ended for client {client_id}")


@router.get("/video/shm")
async def video_stream_shm(
    request: Request,
    app_state: ApplicationState = Depends(get_app_state),
    quality: Optional[int] = Query(
        None, description="JPEG quality (1-100, default: 85)"
    ),
    fps: Optional[int] = Query(
        None, description="Target frame rate (1-60, default: 30)"
    ),
) -> StreamingResponse:
    """Live video streaming endpoint using shared memory IPC.

    This endpoint reads frames from the Video Module's shared memory region,
    enabling high-concurrency streaming (10+ clients) with minimal CPU overhead
    per client. Each client gets an independent SharedMemoryFrameReader.

    Prerequisites:
        - Video Module must be running: python -m backend.video

    Query Parameters:
        quality: JPEG compression quality (1-100, default: 85)
        fps: Target frame rate (1-60, default: 30)

    Returns:
        Streaming response with MJPEG video data from shared memory

    Headers:
        Content-Type: multipart/x-mixed-replace; boundary=frame
        Cache-Control: no-cache
        Connection: close

    Error Responses:
        503: Video Module not running or not available
        500: Stream generation error
    """
    try:
        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        quality_cfg = stream_config.get("quality", {})
        framerate_cfg = stream_config.get("framerate", {})

        # Apply defaults
        quality = (
            quality
            if quality is not None
            else quality_cfg.get("default_jpeg_quality", 85)
        )
        fps = fps if fps is not None else framerate_cfg.get("default_fps", 30)

        # Validate quality
        min_quality = quality_cfg.get("min_jpeg_quality", 1)
        max_quality = quality_cfg.get("max_jpeg_quality", 100)
        if quality < min_quality or quality > max_quality:
            raise HTTPException(
                status_code=400,
                detail=f"Quality must be between {min_quality} and {max_quality}",
            )

        # Validate fps
        min_fps = framerate_cfg.get("min_fps", 1)
        max_fps_limit = framerate_cfg.get("max_fps", 60)
        if fps < min_fps or fps > max_fps_limit:
            raise HTTPException(
                status_code=400,
                detail=f"FPS must be between {min_fps} and {max_fps_limit}",
            )

        logger.info(
            f"Starting shared memory video stream: quality={quality}, fps={fps}"
        )

        # Create streaming response
        return StreamingResponse(
            generate_mjpeg_stream_from_shm(
                request=request,
                app_state=app_state,
                quality=quality,
                fps=fps,
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
        logger.error(f"Shared memory video stream setup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unable to initialize shared memory video stream: {str(e)}",
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
        # Check if streaming camera module exists, but don't create it
        streaming_module_attr = "streaming_camera_module"
        vision_module = getattr(app_state, streaming_module_attr, None)

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
    app_state: ApplicationState = Depends(get_app_state),
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
    quality: Optional[int] = Query(None, description="JPEG quality (1-100)"),
    width: Optional[int] = Query(None, description="Maximum frame width"),
    height: Optional[int] = Query(None, description="Maximum frame height"),
):
    """Get a single frame from the camera as JPEG.

    Captures and returns a single frame from the camera, useful for
    snapshots or testing without setting up a continuous stream.

    Query Parameters:
        quality: JPEG compression quality (uses single_frame config default if not specified)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        JPEG image data
    """
    try:
        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        quality_cfg = stream_config.get("quality", {})
        resolution_cfg = stream_config.get("resolution", {})

        # Apply defaults and validate
        quality = (
            quality
            if quality is not None
            else quality_cfg.get("single_frame_jpeg_quality", 90)
        )

        # Validate quality
        min_quality = quality_cfg.get("min_jpeg_quality", 1)
        max_quality = quality_cfg.get("max_jpeg_quality", 100)
        if quality < min_quality or quality > max_quality:
            raise HTTPException(
                status_code=400,
                detail=f"Quality must be between {min_quality} and {max_quality}",
            )

        # Validate resolution if provided
        if width is not None:
            min_width = resolution_cfg.get("min_width", 160)
            max_width = resolution_cfg.get("max_width", 3840)
            if width < min_width or width > max_width:
                raise HTTPException(
                    status_code=400,
                    detail=f"Width must be between {min_width} and {max_width}",
                )

        if height is not None:
            min_height = resolution_cfg.get("min_height", 120)
            max_height = resolution_cfg.get("max_height", 2160)
            if height < min_height or height > max_height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Height must be between {min_height} and {max_height}",
                )

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
    app_state: ApplicationState = Depends(get_app_state),
    vision_module: CameraModuleAdapter = Depends(get_vision_module),
    quality: Optional[int] = Query(None, description="JPEG quality (1-100)"),
    width: Optional[int] = Query(None, description="Maximum frame width"),
    height: Optional[int] = Query(None, description="Maximum frame height"),
):
    """Get a single RAW frame from the camera as JPEG (no fisheye correction).

    Captures and returns a single frame from the camera WITHOUT any fisheye
    correction or processing. Useful for debugging and comparing raw vs corrected frames.

    Query Parameters:
        quality: JPEG compression quality (uses single_frame config default if not specified)
        width: Maximum frame width for scaling (optional)
        height: Maximum frame height for scaling (optional)

    Returns:
        JPEG image data (raw, uncorrected)
    """
    try:
        # Get stream configuration
        stream_config = _get_stream_config(app_state)
        quality_cfg = stream_config.get("quality", {})
        resolution_cfg = stream_config.get("resolution", {})

        # Apply defaults and validate (same as get_single_frame)
        quality = (
            quality
            if quality is not None
            else quality_cfg.get("single_frame_jpeg_quality", 90)
        )

        # Validate quality
        min_quality = quality_cfg.get("min_jpeg_quality", 1)
        max_quality = quality_cfg.get("max_jpeg_quality", 100)
        if quality < min_quality or quality > max_quality:
            raise HTTPException(
                status_code=400,
                detail=f"Quality must be between {min_quality} and {max_quality}",
            )

        # Validate resolution if provided
        if width is not None:
            min_width = resolution_cfg.get("min_width", 160)
            max_width = resolution_cfg.get("max_width", 3840)
            if width < min_width or width > max_width:
                raise HTTPException(
                    status_code=400,
                    detail=f"Width must be between {min_width} and {max_width}",
                )

        if height is not None:
            min_height = resolution_cfg.get("min_height", 120)
            max_height = resolution_cfg.get("max_height", 2160)
            if height < min_height or height > max_height:
                raise HTTPException(
                    status_code=400,
                    detail=f"Height must be between {min_height} and {max_height}",
                )

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
