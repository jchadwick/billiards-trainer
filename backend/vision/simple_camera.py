"""Simplified camera module for basic MJPEG streaming without vision processing.

This module provides camera capture functionality without the overhead of
detection components (TableDetector, BallDetector, etc.) that may hang during
initialization. Use this for basic video streaming when full vision processing
is not needed or is causing initialization issues.
"""

import logging
import threading
import time
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .capture import CameraCapture, CameraHealth, CameraStatus

logger = logging.getLogger(__name__)


class SimpleCameraModule:
    """Simplified camera module for basic streaming without vision processing.

    This module provides the same camera interface as VisionModule but without
    the detection components, making it suitable for basic MJPEG streaming
    when full vision processing is not needed.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize simplified camera module.

        Args:
            config: Camera configuration dictionary with keys:
                - camera_device_id: Camera device index (default: 0)
                - camera_backend: Camera backend ("auto", "v4l2", "dshow", etc.)
                - camera_resolution: Tuple of (width, height)
                - camera_fps: Target frame rate
                - camera_buffer_size: Frame buffer size
        """
        logger.info("[SimpleCameraModule] Starting __init__")

        if config is None:
            config = {}

        # Extract camera-specific config
        camera_config = {
            "device_id": config.get("camera_device_id", 0),
            "backend": config.get("camera_backend", "auto"),
            "resolution": config.get("camera_resolution", (1920, 1080)),
            "fps": config.get("camera_fps", 30),
            "buffer_size": config.get("camera_buffer_size", 1),
        }

        logger.info(f"[SimpleCameraModule] Extracted camera config: {camera_config}")

        # Create camera capture
        logger.info("[SimpleCameraModule] About to create CameraCapture...")
        self.camera = CameraCapture(camera_config)
        logger.info("[SimpleCameraModule] CameraCapture created successfully")

        # Threading and state
        self._lock = threading.Lock()
        self._current_frame: Optional[NDArray[np.uint8]] = None
        self._frame_update_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Statistics
        self._start_time = time.time()
        self._frames_processed = 0

        logger.info("SimpleCameraModule initialized successfully")

    def start_capture(self) -> bool:
        """Start camera capture.

        Returns:
            True if capture started successfully
        """
        try:
            logger.info("[SimpleCameraModule] start_capture called")

            if self._is_running:
                logger.warning("[SimpleCameraModule] Capture already running")
                return True

            # Start camera
            logger.info("[SimpleCameraModule] About to call camera.start_capture()...")
            if not self.camera.start_capture():
                logger.error("[SimpleCameraModule] camera.start_capture() returned False")
                return False
            logger.info("[SimpleCameraModule] camera.start_capture() succeeded")

            # Start frame update thread
            self._is_running = True
            logger.info("[SimpleCameraModule] Starting frame update thread...")
            self._frame_update_thread = threading.Thread(
                target=self._frame_update_loop,
                name="SimpleCameraFrameUpdate",
                daemon=True
            )
            self._frame_update_thread.start()
            logger.info("[SimpleCameraModule] Frame update thread started")

            logger.info("[SimpleCameraModule] Capture started successfully")
            return True

        except Exception as e:
            logger.error(f"[SimpleCameraModule] Failed to start capture: {e}", exc_info=True)
            return False

    def stop_capture(self) -> None:
        """Stop camera capture."""
        try:
            logger.info("Stopping SimpleCameraModule capture")

            if not self._is_running:
                logger.warning("Capture not running")
                return

            # Stop frame update thread
            self._is_running = False
            if self._frame_update_thread and self._frame_update_thread.is_alive():
                self._frame_update_thread.join(timeout=5.0)

            # Stop camera
            self.camera.stop_capture()

            logger.info("SimpleCameraModule capture stopped")

        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)

    def _frame_update_loop(self) -> None:
        """Background thread to continuously update current frame."""
        logger.info("SimpleCameraModule frame update loop started")

        while self._is_running:
            try:
                # Get latest frame from camera
                frame_data = self.camera.get_latest_frame()
                if frame_data is not None:
                    frame, frame_info = frame_data

                    # Update current frame
                    with self._lock:
                        self._current_frame = frame
                        self._frames_processed += 1

                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
                else:
                    # No frame available, wait a bit
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in frame update loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("SimpleCameraModule frame update loop ended")

    def get_current_frame(self) -> Optional[NDArray[np.uint8]]:
        """Get the latest captured frame.

        Returns:
            Latest frame or None if no frame available
        """
        with self._lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
            return None

    def get_statistics(self) -> dict[str, Any]:
        """Get camera statistics.

        Returns:
            Dictionary containing camera statistics
        """
        uptime = time.time() - self._start_time
        avg_fps = self._frames_processed / uptime if uptime > 0 else 0.0

        return {
            "frames_processed": self._frames_processed,
            "frames_dropped": 0,  # Not tracked in simple mode
            "avg_processing_time_ms": 0.0,  # Not applicable
            "avg_fps": avg_fps,
            "detection_accuracy": {},  # Not applicable
            "uptime_seconds": uptime,
            "last_error": None,
            "is_running": self._is_running,
            "camera_connected": self.camera.is_connected(),
        }

    def __del__(self) -> None:
        """Cleanup on destruction."""
        try:
            self.stop_capture()
        except Exception:
            pass
