"""Camera capture interface for the vision module.

Provides robust camera capture with support for multiple backends,
error handling, reconnection, and health monitoring.

Implements requirements:
- FR-VIS-001: Initialize and configure camera device with specified parameters
- FR-VIS-002: Capture continuous video stream at configurable frame rate (15-60 FPS)
- FR-VIS-003: Support multiple camera backends (V4L2, DirectShow, GStreamer)
- FR-VIS-004: Handle camera disconnection and reconnection gracefully
- FR-VIS-005: Provide camera status and health monitoring
"""

import contextlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Import Kinect v2 support
from .kinect2_capture import KINECT2_AVAILABLE, Kinect2Capture, Kinect2Status

# Import configuration types
try:
    from ..config.models.schemas import CameraBackend, CameraSettings, ExposureMode
except ImportError:
    # Fallback for development/testing
    from enum import Enum

    class CameraBackend(str, Enum):
        AUTO = "auto"
        V4L2 = "v4l2"
        DSHOW = "dshow"
        GSTREAMER = "gstreamer"
        OPENCV = "opencv"
        KINECT2 = "kinect2"

    class ExposureMode(str, Enum):
        AUTO = "auto"
        MANUAL = "manual"


logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class CameraHealth:
    """Camera health monitoring data."""

    status: CameraStatus
    frames_captured: int
    frames_dropped: int
    fps: float
    last_frame_time: float
    error_count: int
    last_error: Optional[str]
    connection_attempts: int
    uptime: float


@dataclass
class FrameInfo:
    """Frame metadata."""

    frame_number: int
    timestamp: float
    size: tuple[int, int]
    channels: int


class CameraCapture:
    """Camera interface and capture management with robust error handling.

    Features:
    - Multiple backend support (V4L2, DirectShow, GStreamer, OpenCV)
    - Automatic reconnection on failure
    - Health monitoring and statistics
    - Thread-safe frame buffering
    - Configurable frame rates and resolutions
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize camera with configuration.

        Args:
            config: Camera configuration dictionary with keys:
                - device_id: Camera device index (default: 0)
                - backend: Camera backend ("auto", "v4l2", "dshow", "gstreamer")
                - resolution: Tuple of (width, height) (default: (1920, 1080))
                - fps: Target frame rate (default: 30)
                - exposure_mode: "auto" or "manual"
                - exposure_value: Manual exposure value (0.0-1.0)
                - gain: Camera gain (default: 1.0)
                - buffer_size: Frame buffer size (default: 1)
                - auto_reconnect: Enable automatic reconnection (default: True)
                - reconnect_delay: Delay between reconnection attempts (default: 1.0)
                - max_reconnect_attempts: Max reconnection attempts (default: 5)
        """
        self._config = config
        self._device_id = config.get("device_id", 0)
        self._backend = config.get("backend", "auto")
        self._resolution = config.get("resolution", (1920, 1080))
        self._fps = config.get("fps", 30)
        self._exposure_mode = config.get("exposure_mode", "auto")
        self._exposure_value = config.get("exposure_value")
        self._gain = config.get("gain", 1.0)
        self._buffer_size = config.get("buffer_size", 1)
        self._auto_reconnect = config.get("auto_reconnect", True)
        self._reconnect_delay = config.get("reconnect_delay", 1.0)
        self._max_reconnect_attempts = config.get("max_reconnect_attempts", 5)

        # Camera state
        self._cap: Optional[cv2.VideoCapture] = None
        self._kinect2: Optional[Kinect2Capture] = None
        self._status = CameraStatus.DISCONNECTED
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=self._buffer_size)
        self._use_kinect2 = self._backend == "kinect2"

        # Health monitoring
        self._start_time = time.time()
        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_frame_time = 0.0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._connection_attempts = 0
        self._fps_tracker = []

        # Thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._status_callback: Optional[Callable[[CameraStatus], None]] = None

        logger.info(
            f"Camera capture initialized with device_id={self._device_id}, "
            f"backend={self._backend}, resolution={self._resolution}"
        )

    def set_status_callback(self, callback: Callable[[CameraStatus], None]) -> None:
        """Set callback function for status changes."""
        self._status_callback = callback

    def _update_status(self, status: CameraStatus) -> None:
        """Update camera status and notify callback."""
        with self._lock:
            if self._status != status:
                old_status = self._status
                self._status = status
                logger.info(f"Camera status changed: {old_status} -> {status}")

                if self._status_callback:
                    try:
                        self._status_callback(status)
                    except Exception as e:
                        logger.error(f"Status callback error: {e}")

    def _get_opencv_backend(self) -> int:
        """Convert backend string to OpenCV constant."""
        backend_map = {
            "auto": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "gstreamer": cv2.CAP_GSTREAMER,
            "opencv": cv2.CAP_OPENCV_MJPEG,
            "kinect2": -1,  # Special case handled separately
        }
        return backend_map.get(self._backend.lower(), cv2.CAP_ANY)

    def _configure_camera(self) -> bool:
        """Configure camera parameters."""
        if not self._cap or not self._cap.isOpened():
            logger.warning("Cannot configure camera - capture not opened")
            return False

        try:
            # Set resolution
            width, height = self._resolution
            logger.debug(f"Setting camera resolution to {width}x{height}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Set frame rate
            logger.debug(f"Setting camera FPS to {self._fps}")
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

            # Set buffer size
            logger.debug(f"Setting camera buffer size to {self._buffer_size}")
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

            # Configure exposure
            if self._exposure_mode == "auto":
                logger.debug("Setting auto exposure mode")
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            elif self._exposure_mode == "manual" and self._exposure_value is not None:
                logger.debug(f"Setting manual exposure to {self._exposure_value}")
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_value)

            # Set gain
            logger.debug(f"Setting camera gain to {self._gain}")
            self._cap.set(cv2.CAP_PROP_GAIN, self._gain)

            # Verify settings
            actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS "
                f"(requested: {width}x{height} @ {self._fps} FPS)"
            )

            return True

        except Exception as e:
            logger.error(f"Camera configuration failed: {e}", exc_info=True)
            self._last_error = str(e)
            self._error_count += 1
            return False

    def _connect_kinect2(self) -> bool:
        """Connect to Kinect v2 device."""
        if not KINECT2_AVAILABLE:
            logger.error("Kinect v2 support not available - install pylibfreenect2")
            return False

        try:
            # Create Kinect v2 configuration from main config
            kinect_config = {
                "enable_color": True,
                "enable_depth": True,
                "enable_infrared": False,
                "min_depth": 500,
                "max_depth": 4000,
                "depth_smoothing": True,
                "auto_reconnect": self._auto_reconnect,
            }

            self._kinect2 = Kinect2Capture(kinect_config)

            # Set up status callback to mirror to main status
            def kinect_status_callback(kinect_status: Kinect2Status):
                if kinect_status == Kinect2Status.CONNECTED:
                    self._update_status(CameraStatus.CONNECTED)
                elif kinect_status == Kinect2Status.DISCONNECTED:
                    self._update_status(CameraStatus.DISCONNECTED)
                elif kinect_status == Kinect2Status.ERROR:
                    self._update_status(CameraStatus.ERROR)
                elif kinect_status == Kinect2Status.RECONNECTING:
                    self._update_status(CameraStatus.RECONNECTING)

            self._kinect2.set_status_callback(kinect_status_callback)

            if self._kinect2.start_capture():
                logger.info("Kinect v2 connected successfully")
                return True
            else:
                logger.error("Failed to start Kinect v2 capture")
                return False

        except Exception as e:
            logger.error(f"Kinect v2 connection failed: {e}")
            self._last_error = str(e)
            self._error_count += 1
            self._kinect2 = None
            return False

    def _connect_camera(self) -> bool:
        """Establish camera connection."""
        try:
            self._connection_attempts += 1
            logger.info(f"Camera connection attempt #{self._connection_attempts}")

            # Handle Kinect v2 connection
            if self._use_kinect2:
                logger.info("Using Kinect v2 backend")
                return self._connect_kinect2()

            # Handle regular camera connection
            backend = self._get_opencv_backend()

            logger.info(
                f"Connecting to camera device_id={self._device_id} with backend={self._backend} (OpenCV backend code: {backend})"
            )

            logger.debug("Creating VideoCapture instance...")
            self._cap = cv2.VideoCapture(self._device_id, backend)

            if not self._cap.isOpened():
                logger.error(f"VideoCapture failed to open camera {self._device_id}")
                raise RuntimeError(f"Failed to open camera {self._device_id}")

            logger.info(
                f"VideoCapture opened successfully for camera {self._device_id}"
            )

            logger.debug("Configuring camera parameters...")
            if not self._configure_camera():
                raise RuntimeError("Camera configuration failed")

            logger.info("Camera configured, testing frame capture...")
            # Test frame capture
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.error(
                    f"Test frame capture failed: ret={ret}, frame={'None' if frame is None else 'available'}"
                )
                raise RuntimeError("Failed to capture test frame")

            logger.info(
                f"Test frame captured successfully: shape={frame.shape}, dtype={frame.dtype}"
            )
            logger.info(
                f"Camera connected successfully (attempt {self._connection_attempts})"
            )
            return True

        except Exception as e:
            logger.error(f"Camera connection failed: {e}", exc_info=True)
            self._last_error = str(e)
            self._error_count += 1

            if self._cap:
                logger.debug("Releasing failed camera capture")
                self._cap.release()
                self._cap = None

            return False

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        logger.info("Camera capture loop started")
        frame_count = 0

        while not self._stop_event.is_set():
            try:
                # Check connection status
                if self._use_kinect2:
                    if not self._kinect2 or not self._kinect2.is_connected():
                        logger.warning("Kinect v2 disconnected in capture loop")
                        if self._auto_reconnect:
                            self._update_status(CameraStatus.RECONNECTING)
                            if self._connect_camera():
                                self._update_status(CameraStatus.CONNECTED)
                            else:
                                logger.debug(
                                    f"Reconnection failed, sleeping {self._reconnect_delay}s"
                                )
                                time.sleep(self._reconnect_delay)
                                continue
                        else:
                            self._update_status(CameraStatus.ERROR)
                            break
                else:
                    if not self._cap or not self._cap.isOpened():
                        logger.warning("Camera disconnected in capture loop")
                        if self._auto_reconnect:
                            self._update_status(CameraStatus.RECONNECTING)
                            if self._connect_camera():
                                self._update_status(CameraStatus.CONNECTED)
                            else:
                                logger.debug(
                                    f"Reconnection failed, sleeping {self._reconnect_delay}s"
                                )
                                time.sleep(self._reconnect_delay)
                                continue
                        else:
                            self._update_status(CameraStatus.ERROR)
                            break

                # Capture frame
                current_time = time.time()
                frame = None

                if self._use_kinect2:
                    # Get frame from Kinect v2
                    logger.debug("Attempting to get Kinect v2 frame")
                    kinect_frame = self._kinect2.get_latest_frame()
                    if kinect_frame and kinect_frame.color is not None:
                        frame = kinect_frame.color
                        # Convert RGB to BGR for OpenCV compatibility
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        logger.debug(f"Kinect v2 frame captured: shape={frame.shape}")
                    else:
                        logger.warning("Failed to capture Kinect v2 frame")
                        self._error_count += 1
                        self._last_error = "Kinect v2 frame capture failed"

                        if self._auto_reconnect:
                            self._update_status(CameraStatus.RECONNECTING)
                            time.sleep(self._reconnect_delay)
                            continue
                        else:
                            break
                else:
                    # Get frame from regular camera
                    logger.debug("Attempting to read frame from camera")
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        logger.warning(
                            f"Failed to capture frame: ret={ret}, frame={'None' if frame is None else 'available'}"
                        )
                        self._error_count += 1
                        self._last_error = "Frame capture failed"

                        if self._auto_reconnect:
                            self._update_status(CameraStatus.RECONNECTING)
                            time.sleep(self._reconnect_delay)
                            continue
                        else:
                            break

                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.debug(
                            f"Frame captured successfully: shape={frame.shape}, count={frame_count}"
                        )

                # Update statistics
                self._frames_captured += 1
                self._last_frame_time = current_time

                # Calculate FPS
                self._fps_tracker.append(current_time)
                if len(self._fps_tracker) > 10:
                    self._fps_tracker.pop(0)

                # Add frame to queue
                frame_info = FrameInfo(
                    frame_number=self._frames_captured,
                    timestamp=current_time,
                    size=(frame.shape[1], frame.shape[0]),
                    channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                )

                try:
                    self._frame_queue.put_nowait((frame.copy(), frame_info))
                except queue.Full:
                    # Drop oldest frame if queue is full
                    try:
                        self._frame_queue.get_nowait()
                        self._frames_dropped += 1
                        self._frame_queue.put_nowait((frame.copy(), frame_info))
                    except queue.Empty:
                        pass

                # Maintain target frame rate
                if self._fps > 0:
                    time.sleep(max(0, 1.0 / self._fps - 0.001))

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self._last_error = str(e)
                self._error_count += 1

                if self._auto_reconnect:
                    self._update_status(CameraStatus.RECONNECTING)
                    time.sleep(self._reconnect_delay)
                else:
                    self._update_status(CameraStatus.ERROR)
                    break

        logger.info("Camera capture loop ended")

    def start_capture(self) -> bool:
        """Start camera capture.

        Returns:
            True if capture started successfully, False otherwise
        """
        logger.info("start_capture called")
        with self._lock:
            if self._capture_thread and self._capture_thread.is_alive():
                logger.warning("Capture already running")
                return True

            logger.info("Clearing stop event and updating status to CONNECTING")
            self._stop_event.clear()
            self._update_status(CameraStatus.CONNECTING)

            # Reset statistics
            logger.debug("Resetting capture statistics")
            self._start_time = time.time()
            self._frames_captured = 0
            self._frames_dropped = 0
            self._fps_tracker.clear()

            # Clear frame queue
            logger.debug("Clearing frame queue")
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            # Connect camera
            logger.info("Attempting to connect to camera")
            if not self._connect_camera():
                logger.error("Camera connection failed")
                self._update_status(CameraStatus.ERROR)
                return False

            logger.info("Camera connected, starting capture thread")
            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="CameraCapture", daemon=True
            )
            self._capture_thread.start()

            self._update_status(CameraStatus.CONNECTED)
            logger.info("Camera capture started successfully")
            return True

    def stop_capture(self) -> None:
        """Stop camera capture."""
        with self._lock:
            if not self._capture_thread or not self._capture_thread.is_alive():
                logger.info("Capture not running")
                return

            logger.info("Stopping camera capture...")
            self._stop_event.set()

            # Wait for capture thread to finish
            if self._capture_thread:
                self._capture_thread.join(timeout=5.0)
                if self._capture_thread.is_alive():
                    logger.warning("Capture thread did not stop gracefully")

            # Release camera resources
            if self._use_kinect2 and self._kinect2:
                self._kinect2.stop_capture()
                self._kinect2 = None
            elif self._cap:
                self._cap.release()
                self._cap = None

            self._update_status(CameraStatus.DISCONNECTED)
            logger.info("Camera capture stopped")

    def get_frame(self) -> Optional[tuple[np.ndarray, FrameInfo]]:
        """Get latest captured frame.

        Returns:
            Tuple of (frame, frame_info) if available, None otherwise
        """
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[tuple[np.ndarray, FrameInfo]]:
        """Get the most recent frame, discarding older ones.

        Returns:
            Tuple of (frame, frame_info) if available, None otherwise
        """
        latest = None
        try:
            while True:
                latest = self._frame_queue.get_nowait()
        except queue.Empty:
            return latest

    def is_connected(self) -> bool:
        """Check if camera is connected and capturing."""
        with self._lock:
            if self._use_kinect2:
                return (
                    self._status == CameraStatus.CONNECTED
                    and self._kinect2 is not None
                    and self._kinect2.is_connected()
                )
            else:
                return (
                    self._status == CameraStatus.CONNECTED
                    and self._cap is not None
                    and self._cap.isOpened()
                )

    def get_status(self) -> CameraStatus:
        """Get current camera status."""
        with self._lock:
            return self._status

    def get_health(self) -> CameraHealth:
        """Get comprehensive camera health information."""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self._start_time

            # Calculate current FPS
            current_fps = 0.0
            if len(self._fps_tracker) > 1:
                time_span = self._fps_tracker[-1] - self._fps_tracker[0]
                if time_span > 0:
                    current_fps = (len(self._fps_tracker) - 1) / time_span

            return CameraHealth(
                status=self._status,
                frames_captured=self._frames_captured,
                frames_dropped=self._frames_dropped,
                fps=current_fps,
                last_frame_time=self._last_frame_time,
                error_count=self._error_count,
                last_error=self._last_error,
                connection_attempts=self._connection_attempts,
                uptime=uptime,
            )

    def get_camera_info(self) -> dict[str, Any]:
        """Get camera device information."""
        if self._use_kinect2:
            if not self._kinect2 or not self._kinect2.is_connected():
                return {}

            try:
                kinect_info = self._kinect2.get_device_info()
                return {
                    "device_id": "kinect2",
                    "backend": self._backend,
                    "device_type": "Microsoft Kinect v2",
                    "has_depth": True,
                    "has_color": True,
                    "depth_range": kinect_info.get("depth_range", "500-4000mm"),
                    "color_resolution": kinect_info.get("color_resolution", "1080p"),
                    "depth_enabled": kinect_info.get("depth_enabled", True),
                    "color_enabled": kinect_info.get("color_enabled", True),
                    "infrared_enabled": kinect_info.get("infrared_enabled", False),
                }
            except Exception as e:
                logger.error(f"Error getting Kinect v2 info: {e}")
                return {}
        else:
            if not self._cap or not self._cap.isOpened():
                return {}

            try:
                return {
                    "device_id": self._device_id,
                    "backend": self._backend,
                    "device_type": "Standard Camera",
                    "has_depth": False,
                    "has_color": True,
                    "resolution": (
                        int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ),
                    "fps": self._cap.get(cv2.CAP_PROP_FPS),
                    "exposure": self._cap.get(cv2.CAP_PROP_EXPOSURE),
                    "gain": self._cap.get(cv2.CAP_PROP_GAIN),
                    "brightness": self._cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    "contrast": self._cap.get(cv2.CAP_PROP_CONTRAST),
                    "saturation": self._cap.get(cv2.CAP_PROP_SATURATION),
                    "buffer_size": int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                }
            except Exception as e:
                logger.error(f"Error getting camera info: {e}")
                return {}

    def update_config(self, config: dict[str, Any]) -> bool:
        """Update camera configuration at runtime.

        Args:
            config: New configuration parameters

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update internal config
            self._config.update(config)

            # Apply changes that can be updated without reconnection
            if self._cap and self._cap.isOpened():
                if "fps" in config:
                    self._fps = config["fps"]
                    self._cap.set(cv2.CAP_PROP_FPS, self._fps)

                if (
                    "exposure_value" in config
                    and config.get("exposure_mode") == "manual"
                ):
                    self._exposure_value = config["exposure_value"]
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                    self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_value)

                if "gain" in config:
                    self._gain = config["gain"]
                    self._cap.set(cv2.CAP_PROP_GAIN, self._gain)

            logger.info(f"Camera configuration updated: {config}")
            return True

        except Exception as e:
            logger.error(f"Camera config update failed: {e}")
            self._last_error = str(e)
            self._error_count += 1
            return False

    def get_depth_frame(self) -> Optional[NDArray[np.float64]]:
        """Get latest depth frame from Kinect v2.

        Returns:
            Depth frame as numpy array in millimeters, or None if not available
        """
        if not self._use_kinect2 or not self._kinect2:
            logger.warning("Depth frame requested but Kinect v2 not available")
            return None

        kinect_frame = self._kinect2.get_latest_frame()
        if kinect_frame and kinect_frame.depth is not None:
            return kinect_frame.depth
        return None

    def get_3d_point(
        self, u: int, v: int, depth: float
    ) -> Optional[tuple[float, float, float]]:
        """Convert depth pixel to 3D world coordinates using Kinect v2.

        Args:
            u, v: Pixel coordinates in depth image
            depth: Depth value in millimeters

        Returns:
            (x, y, z) coordinates in millimeters, or None if not available
        """
        if not self._use_kinect2 or not self._kinect2:
            logger.warning("3D point conversion requested but Kinect v2 not available")
            return None

        return self._kinect2.depth_to_3d(u, v, depth)

    def get_kinect_calibration(self) -> Optional[dict]:
        """Get Kinect v2 calibration parameters for 3D reconstruction.

        Returns:
            Dictionary with camera parameters, or None if not available
        """
        if not self._use_kinect2 or not self._kinect2:
            return None

        return self._kinect2.get_calibration_data()

    def __enter__(self) -> "CameraCapture":
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop_capture()

    def __del__(self) -> None:
        """Destructor - ensure resources are cleaned up."""
        with contextlib.suppress(Exception):
            self.stop_capture()
