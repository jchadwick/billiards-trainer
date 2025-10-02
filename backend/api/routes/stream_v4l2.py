"""Minimal video streaming using V4L2/ffmpeg subprocess.

This module provides the simplest possible camera streaming by using
ffmpeg subprocess, avoiding all OpenCV initialization issues.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

# Use absolute imports
from backend.vision.v4l2_camera import V4L2CameraModule
from backend.api.dependencies import ApplicationState, get_app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["Video Streaming"])


async def get_camera_module(
    app_state: ApplicationState = Depends(get_app_state),
) -> V4L2CameraModule:
    """Get or create the shared V4L2 camera module.

    Lazy initialization - creates camera on first access.
    """
    if not hasattr(app_state, "camera_module") or app_state.camera_module is None:
        logger.info("Creating V4L2 camera module (lazy init)")

        config = {
            "camera_device_id": 0,
            "camera_resolution": (1920, 1080),
            "camera_fps": 30,
        }

        try:
            # Create module (doesn't start camera yet)
            camera_module = V4L2CameraModule(config)

            # Start camera in executor to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                lambda: camera_module.start_capture(timeout=10.0)
            )

            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to start camera"
                )

            app_state.camera_module = camera_module
            logger.info("V4L2 camera module created and started")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}", exc_info=True)
            app_state.camera_module = None
            raise HTTPException(
                status_code=503,
                detail=f"Camera initialization failed: {e}"
            )

    return app_state.camera_module


@router.get("/video")
async def video_stream(
    camera_module: V4L2CameraModule = Depends(get_camera_module),
) -> StreamingResponse:
    """Live MJPEG video stream from camera.

    Uses ffmpeg subprocess to stream v4l2 camera directly.
    """
    if not camera_module.is_running():
        raise HTTPException(
            status_code=503,
            detail="Camera is not running"
        )

    logger.info("Starting video stream")

    return StreamingResponse(
        camera_module.generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/video/status")
async def stream_status(
    app_state: ApplicationState = Depends(get_app_state),
) -> dict[str, Any]:
    """Get camera and streaming status."""
    camera_module = getattr(app_state, "camera_module", None)

    if not camera_module:
        return {
            "camera": {
                "status": "not_initialized",
                "is_running": False,
            },
            "timestamp": time.time(),
        }

    stats = camera_module.get_statistics()

    return {
        "camera": {
            "status": "running" if stats["is_running"] else "stopped",
            "is_running": stats["is_running"],
            "device_path": stats["device_path"],
            "resolution": stats["resolution"],
            "target_fps": stats["target_fps"],
        },
        "timestamp": time.time(),
    }


@router.post("/video/start")
async def start_video(
    camera_module: V4L2CameraModule = Depends(get_camera_module),
) -> dict[str, Any]:
    """Start video capture (camera is auto-started on first access)."""
    is_running = camera_module.is_running()

    return {
        "status": "running" if is_running else "error",
        "message": "Camera is running" if is_running else "Camera failed to start",
    }


@router.post("/video/stop")
async def stop_video(
    camera_module: V4L2CameraModule = Depends(get_camera_module),
) -> dict[str, Any]:
    """Stop video capture."""
    logger.info("Stopping camera via API")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, camera_module.stop_capture)

    return {
        "status": "stopped",
        "message": "Camera stopped successfully",
    }
