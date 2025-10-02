"""Direct OpenCV camera module with minimal dependencies.

This module provides a simple, thread-safe camera interface using OpenCV's VideoCapture.
It avoids complex wrappers and imports that can cause initialization hangs.
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStatus:
    """Simple camera status wrapper for compatibility with existing code."""

    def __init__(self, camera_module: "DirectCameraModule"):
        self._module = camera_module

    def is_active(self) -> bool:
        """Check if camera is actively capturing."""
        return self._module.is_capturing()

    def get_last_frame_time(self) -> Optional[float]:
        """Get timestamp of last captured frame."""
        return self._module._last_frame_time


class DirectCameraModule:
    """Minimal, direct OpenCV camera implementation.

    Features:
    - Single producer thread continuously capturing frames
    - Thread-safe frame buffer with lock
    - Rate limiting per consumer type
    - No complex imports or dependencies
    - Simple start/stop interface
    """

    # Consumer types for rate limiting
    CONSUMER_PROCESSING = "processing"
    CONSUMER_STREAMING = "streaming"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize camera module.

        Args:
            config: Camera configuration dict with keys:
                - device_id: Camera device index (default: 0)
                - resolution: Tuple of (width, height) (default: (1920, 1080))
                - fps: Target FPS (default: 30)
                - buffer_size: Capture buffer size (default: 1)
        """
        config = config or {}

        # Extract configuration
        self.device_id = config.get("device_id", 0)
        self.resolution = config.get("resolution", (1920, 1080))
        self.fps = config.get("fps", 30)
        self.buffer_size = config.get("buffer_size", 1)

        # Rate limiting configuration (seconds between frames per consumer)
        self.rate_limits = {
            self.CONSUMER_PROCESSING: 1.0 / 30.0,  # Max 30 FPS for processing
            self.CONSUMER_STREAMING: 1.0 / 15.0,  # Max 15 FPS for streaming
        }

        # Thread-safe state
        self._capture: Optional[cv2.VideoCapture] = None
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # Signals camera is initialized
        self._init_error: Optional[str] = None  # Store initialization errors

        # Statistics
        self._frame_count = 0
        self._last_frame_time: Optional[float] = None
        self._start_time: Optional[float] = None
        self._last_consumer_access = {
            self.CONSUMER_PROCESSING: 0.0,
            self.CONSUMER_STREAMING: 0.0,
        }
        self._dropped_frames = {
            self.CONSUMER_PROCESSING: 0,
            self.CONSUMER_STREAMING: 0,
        }

        # Create status wrapper for compatibility
        self.camera = CameraStatus(self)

        logger.info(
            f"DirectCameraModule initialized with device={self.device_id}, "
            f"resolution={self.resolution}, fps={self.fps}"
        )

    def start_capture(self, timeout: float = 10.0) -> bool:
        """Start camera capture.

        Args:
            timeout: Maximum seconds to wait for camera initialization

        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if self._capture_thread and self._capture_thread.is_alive():
            logger.warning("Capture already running")
            return True

        logger.info("Starting camera capture thread...")

        # Clear events and error
        self._stop_event.clear()
        self._ready_event.clear()
        self._init_error = None
        self._start_time = time.time()

        # Start capture thread (camera init happens in thread)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="DirectCameraCapture", daemon=True
        )
        self._capture_thread.start()

        # Wait for camera to initialize (with timeout)
        logger.info(f"Waiting up to {timeout}s for camera initialization...")
        if self._ready_event.wait(timeout=timeout):
            # Check if initialization succeeded
            if self._init_error:
                logger.error(f"Camera initialization failed: {self._init_error}")
                self.stop_capture()
                return False

            logger.info("Camera capture started successfully")
            return True
        else:
            # Timeout waiting for camera
            logger.error(f"Camera initialization timed out after {timeout}s")
            self.stop_capture()
            return False

    def stop_capture(self):
        """Stop camera capture and release resources."""
        logger.info("Stopping camera capture...")

        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
            self._capture_thread = None

        # Release camera
        if self._capture:
            self._capture.release()
            self._capture = None

        # Clear current frame
        with self._frame_lock:
            self._current_frame = None

        logger.info("Camera capture stopped")

    def is_capturing(self) -> bool:
        """Check if camera is actively capturing."""
        return (
            self._capture_thread is not None
            and self._capture_thread.is_alive()
            and not self._stop_event.is_set()
        )

    def _capture_loop(self):
        """Main capture loop running in separate thread.

        Initializes camera, then continuously captures frames and updates the shared frame buffer.
        """
        logger.info("Capture loop started - initializing camera...")

        # Initialize camera in this thread (non-blocking for async event loop)
        try:
            self._capture = cv2.VideoCapture(self.device_id)
            if not self._capture.isOpened():
                error_msg = f"Failed to open camera device {self.device_id}"
                logger.error(error_msg)
                self._init_error = error_msg
                self._ready_event.set()  # Signal failure
                return

            # Configure camera
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Log actual settings
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera opened successfully: resolution={actual_width}x{actual_height}, "
                f"fps={actual_fps}"
            )

            # Signal successful initialization
            self._ready_event.set()

        except Exception as e:
            error_msg = f"Exception during camera initialization: {e}"
            logger.error(error_msg, exc_info=True)
            self._init_error = error_msg
            self._ready_event.set()  # Signal failure
            return

        # Main capture loop
        logger.info("Starting frame capture loop...")
        frame_interval = 1.0 / self.fps

        while not self._stop_event.is_set():
            try:
                # Capture frame
                ret, frame = self._capture.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Update shared frame buffer
                with self._frame_lock:
                    self._current_frame = frame
                    self._frame_count += 1
                    self._last_frame_time = time.time()

                # Rate limiting
                time.sleep(frame_interval)

            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info(f"Capture loop stopped after {self._frame_count} frames")

    def _should_provide_frame(self, consumer_type: str) -> bool:
        """Check if enough time has passed to provide a frame to this consumer.

        Args:
            consumer_type: Type of consumer requesting frame

        Returns:
            bool: True if frame should be provided
        """
        now = time.time()
        last_access = self._last_consumer_access.get(consumer_type, 0.0)
        rate_limit = self.rate_limits.get(consumer_type, 0.0)

        return (now - last_access) >= rate_limit

    def _mark_consumer_access(self, consumer_type: str):
        """Mark that a consumer accessed a frame.

        Args:
            consumer_type: Type of consumer that accessed frame
        """
        self._last_consumer_access[consumer_type] = time.time()

    def get_frame_for_processing(self) -> Optional[np.ndarray]:
        """Get full resolution frame for processing.

        Returns rate-limited copy of current frame for processing consumers.

        Returns:
            Optional[np.ndarray]: Frame copy or None if not available or rate limited
        """
        if not self._should_provide_frame(self.CONSUMER_PROCESSING):
            self._dropped_frames[self.CONSUMER_PROCESSING] += 1
            return None

        with self._frame_lock:
            if self._current_frame is None:
                return None
            frame = self._current_frame.copy()

        self._mark_consumer_access(self.CONSUMER_PROCESSING)
        return frame

    def get_frame_for_streaming(self, scale: float = 0.5) -> Optional[np.ndarray]:
        """Get downsampled frame for streaming.

        Returns rate-limited, downsampled copy of current frame for streaming consumers.

        Args:
            scale: Scaling factor for downsampling (default: 0.5)

        Returns:
            Optional[np.ndarray]: Downsampled frame copy or None if not available or rate limited
        """
        if not self._should_provide_frame(self.CONSUMER_STREAMING):
            self._dropped_frames[self.CONSUMER_STREAMING] += 1
            return None

        with self._frame_lock:
            if self._current_frame is None:
                return None
            frame = self._current_frame.copy()

        # Downsample
        if scale != 1.0:
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        self._mark_consumer_access(self.CONSUMER_STREAMING)
        return frame

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame without rate limiting.

        Compatibility method for existing code that expects immediate frame access.

        Returns:
            Optional[np.ndarray]: Frame copy or None if not available
        """
        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get camera statistics.

        Returns:
            dict: Statistics including frame count, FPS, dropped frames, etc.
        """
        now = time.time()
        uptime = now - self._start_time if self._start_time else 0.0
        actual_fps = self._frame_count / uptime if uptime > 0 else 0.0

        return {
            "is_capturing": self.is_capturing(),
            "frame_count": self._frame_count,
            "uptime_seconds": uptime,
            "actual_fps": actual_fps,
            "target_fps": self.fps,
            "last_frame_time": self._last_frame_time,
            "resolution": self.resolution,
            "device_id": self.device_id,
            "dropped_frames": dict(self._dropped_frames),
            "rate_limits": dict(self.rate_limits),
        }

    def get_frame_shape(self) -> Optional[tuple]:
        """Get shape of current frame.

        Returns:
            Optional[tuple]: Frame shape (height, width, channels) or None
        """
        with self._frame_lock:
            if self._current_frame is None:
                return None
            return self._current_frame.shape

    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_capture()
