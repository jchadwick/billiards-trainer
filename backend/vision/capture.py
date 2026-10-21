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
from types import TracebackType
from typing import Any, Callable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.perception.camera_source import is_http_camera_source

# Import Kinect v2 support
from .kinect2_capture import KINECT2_AVAILABLE, Kinect2Capture, Kinect2Status

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
    """Frame metadata.

    ``timestamp`` is sampled by the software after the frame is retrieved from
    its backend. OpenCV and the current Kinect adapter do not provide a common
    sensor-exposure clock, so this value must not be interpreted as exposure
    time. ``camera_read_seconds`` is a monotonic duration around the backend
    retrieval operation when the performance audit is enabled.
    """

    frame_number: int
    timestamp: float
    size: tuple[int, int]
    channels: int
    camera_read_seconds: float | None = None
    capture_queue_frames_skipped: int = 0
    capture_queue_depth_after_drain: int | None = None


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
                - device_id: Camera device index (default: 0) or video file path (str)
                  An HTTP(S) URI selects an MJPEG network source instead of a device.
                - backend: Camera backend ("auto", "v4l2", "dshow", "gstreamer", "kinect2")
                - resolution: Tuple of (width, height) (default: [1920, 1080])
                - fps: Target frame rate (default: 30)
                - control_policy: "source_defaults" or "require"; ordinary webcams
                  keep OS-selected controls unless explicit controls are required
                - exposure_mode: "auto" or "manual"
                - exposure_value: Manual exposure value (0.0-1.0)
                - gain: Camera gain (default: 1.0)
                - buffer_size: Frame buffer size (default: 1)
                - auto_reconnect: Enable automatic reconnection (default: True)
                - reconnect_delay: Delay between reconnection attempts in seconds (default: 1.0)
                - max_reconnect_attempts: Max reconnection attempts (default: 5)
                - stop_timeout: Maximum time to wait for the capture worker during stop (default: 5.0)
                - frame_log_interval: Log every Nth frame (default: 30)
                - fps_tracker_size: FPS calculation window size (default: 10)
                - frame_sleep_compensation: Sleep compensation for frame rate in seconds (default: 0.001)
                - loop_video: Loop video playback for file sources (default: False)
                - video_start_frame: Starting frame number for video files (default: 0)
                - video_end_frame: Ending frame number for video files (default: None = end of video)
                - kinect2: Kinect v2 specific configuration (dict):
                    - enable_color: Enable color stream (default: True)
                    - enable_depth: Enable depth stream (default: True)
                    - enable_infrared: Enable infrared stream (default: False)
                    - min_depth: Minimum depth in millimeters (default: 500)
                    - max_depth: Maximum depth in millimeters (default: 4000)
                    - depth_smoothing: Enable depth smoothing (default: True)
                    - auto_reconnect: Enable auto-reconnect for Kinect (default: True)

        Raises:
            ValueError: If video_start_frame >= video_end_frame when both are specified
        """
        self._config = config
        self._device_id = config.get("device_id", 0)
        self._backend = config.get("backend", "auto")
        self._loop_video = config.get("loop_video", False)
        self._is_stream_uri = is_http_camera_source(self._device_id)
        self._is_video_file = (
            isinstance(self._device_id, str) and not self._is_stream_uri
        )
        if isinstance(self._device_id, str) and self._device_id.startswith(
            "gstreamer-device-v1:"
        ):
            raise ValueError(
                "stable GStreamer selectors require the Rust GStreamer capture adapter"
            )

        # Camera parameters - get from config with fallback to defaults
        self._resolution = tuple(config.get("resolution", [1920, 1080]))
        self._fps = config.get("fps", 30)
        self._control_policy = config.get("control_policy", "source_defaults")
        if not isinstance(self._control_policy, str) or self._control_policy not in {
            "source_defaults",
            "require",
        }:
            raise ValueError("control_policy must be 'source_defaults' or 'require'")
        self._exposure_mode = config.get("exposure_mode", "auto")
        self._exposure_value = config.get("exposure_value")
        self._gain = config.get("gain", 1.0)
        self._buffer_size = config.get("buffer_size", 1)
        if (
            isinstance(self._buffer_size, bool)
            or not isinstance(self._buffer_size, int)
            or self._buffer_size <= 0
        ):
            raise ValueError("buffer_size must be a positive integer")

        # Connection parameters
        self._auto_reconnect = config.get("auto_reconnect", True)
        self._reconnect_delay = config.get("reconnect_delay", 1.0)
        self._max_reconnect_attempts = config.get("max_reconnect_attempts", 5)
        self._stop_timeout = config.get("stop_timeout", 5.0)

        # Performance parameters
        self._frame_log_interval = config.get("frame_log_interval", 30)
        self._fps_tracker_size = config.get("fps_tracker_size", 10)
        self._frame_sleep_compensation = config.get("frame_sleep_compensation", 0.001)
        self._audit_timing_enabled = bool(
            config.get("performance_audit_enabled", False)
        )

        # Video frame range parameters
        self._video_start_frame = config.get("video_start_frame", 0)
        self._video_end_frame = config.get("video_end_frame")

        # Kinect2 parameters - store for later use
        kinect2_config = config.get("kinect2", {})
        self._kinect2_min_depth = kinect2_config.get("min_depth", 500)
        self._kinect2_max_depth = kinect2_config.get("max_depth", 4000)

        # Validate frame range parameters
        if (
            self._video_end_frame is not None
            and self._video_start_frame >= self._video_end_frame
        ):
            raise ValueError(
                f"video_start_frame ({self._video_start_frame}) must be less than "
                f"video_end_frame ({self._video_end_frame})"
            )

        # Camera state
        self._cap: Optional[cv2.VideoCapture] = None
        self._kinect2: Optional[Kinect2Capture] = None
        self._status = CameraStatus.DISCONNECTED
        self._capture_thread: Optional[threading.Thread] = None
        # A timed-out stop cannot cancel a native OpenCV/Kinect read. In that
        # case the capture worker remains the sole owner and releases the
        # source itself after the read returns and the worker exits.
        self._deferred_source_release = False
        self._stop_event = threading.Event()
        # Network streams are freshness-first: retain only the newest frame if
        # downstream inference falls behind, regardless of the local-camera
        # queue setting.
        self._frame_queue: queue.Queue[tuple[NDArray[Any], FrameInfo]] = queue.Queue(
            maxsize=1 if self._is_stream_uri else self._buffer_size
        )
        self._use_kinect2 = self._backend == "kinect2"

        # Health monitoring
        self._start_time = time.time()
        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_frame_time = 0.0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._connection_attempts = 0
        self._fps_tracker: list[float] = []

        # Thread safety
        self._lock = threading.Lock()
        # Serialize external start/stop calls without holding the state lock
        # across backend startup, worker joins, or native resource release.
        self._lifecycle_lock = threading.Lock()

        # Callbacks
        self._status_callback: Optional[Callable[[CameraStatus], None]] = None

        source_type = (
            "uStreamer MJPEG source"
            if self._is_stream_uri
            else "video file" if self._is_video_file else "camera device"
        )
        log_msg = (
            f"Camera capture initialized with {source_type}={self._device_id}, "
            f"backend={self._backend}, resolution={self._resolution}"
        )
        if self._is_video_file and (
            self._video_start_frame != 0 or self._video_end_frame is not None
        ):
            log_msg += (
                f", frame_range=[{self._video_start_frame}, "
                f"{self._video_end_frame if self._video_end_frame is not None else 'end'}]"
            )
        logger.info(log_msg)

    def set_status_callback(self, callback: Callable[[CameraStatus], None]) -> None:
        """Set callback function for status changes."""
        self._status_callback = callback

    def _update_status_unsafe(self, status: CameraStatus) -> None:
        """Update camera status without acquiring lock (must be called with lock held)."""
        if self._status != status:
            old_status = self._status
            self._status = status
            logger.info(f"Camera status changed: {old_status} -> {status}")

            if self._status_callback:
                try:
                    self._status_callback(status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

    def _update_status(self, status: CameraStatus) -> None:
        """Update camera status and notify callback."""
        with self._lock:
            self._update_status_unsafe(status)

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

        if self._is_stream_uri:
            # Resolution, frame rate and camera controls belong to the process
            # that owns the physical camera (uStreamer). Buffer reduction is
            # best-effort because several OpenCV backends ignore this property.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                logger.debug("Stream backend does not expose a buffer-size control")
            return True

        # Video files don't need configuration - use their native settings
        if self._is_video_file:
            logger.info("Skipping configuration for video file - using native settings")
            return True

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

            # Driver defaults keep ordinary webcams usable across platforms
            # where OpenCV may not expose portable control readbacks. Explicit
            # controls remain an opt-in policy and are still written here.
            if self._control_policy == "require":
                if self._exposure_mode == "auto":
                    logger.debug("Setting auto exposure mode")
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                elif (
                    self._exposure_mode == "manual" and self._exposure_value is not None
                ):
                    logger.debug(f"Setting manual exposure to {self._exposure_value}")
                    self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                    self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_value)

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
            # Check for kinect2-specific config section first, fallback to defaults
            kinect2_config = self._config.get("kinect2", {})
            kinect_config = {
                "enable_color": kinect2_config.get("enable_color", True),
                "enable_depth": kinect2_config.get("enable_depth", True),
                "enable_infrared": kinect2_config.get("enable_infrared", False),
                "min_depth": kinect2_config.get("min_depth", 500),
                "max_depth": kinect2_config.get("max_depth", 4000),
                "depth_smoothing": kinect2_config.get("depth_smoothing", True),
                "auto_reconnect": kinect2_config.get(
                    "auto_reconnect", self._auto_reconnect
                ),
                "stop_timeout": self._stop_timeout,
            }

            self._kinect2 = Kinect2Capture(kinect_config)

            # Set up status callback to mirror to main status
            def kinect_status_callback(kinect_status: Kinect2Status) -> None:
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
            # A URI is a network source, not a V4L2/DirectShow device. Let
            # OpenCV choose its available HTTP/MJPEG backend independently of
            # the local camera backend setting.
            backend = cv2.CAP_ANY if self._is_stream_uri else self._get_opencv_backend()

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

            # Seek to start frame for video files
            if self._is_video_file and self._video_start_frame > 0:
                logger.info(f"Seeking to start frame {self._video_start_frame}")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._video_start_frame)
                actual_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                logger.info(f"Video positioned at frame {actual_frame}")

            # Do not probe with a separate read thread here. Native camera
            # reads are not cancellable by Python futures, so a timed-out
            # startup probe can outlive its owner and race both release and
            # restart. The capture worker owns every read and measures the
            # first frame on the same path as all later frames.
            logger.info(
                f"Camera opened and configured successfully (attempt {self._connection_attempts})"
            )
            return True

        except Exception as e:
            logger.error(f"Camera connection failed: {e}", exc_info=True)
            self._last_error = str(e)
            self._error_count += 1
            # Startup no longer launches a separate probe reader, so ordinary
            # connection failures have no outstanding native reader.
            if self._cap:
                logger.debug("Releasing failed camera capture")
                self._cap.release()
                self._cap = None

            return False

    def _capture_worker(self) -> None:
        """Run acquisition and honor deferred cleanup after a stop timeout.

        OpenCV and Kinect native reads cannot be cancelled safely from another
        thread. If the caller's join deadline expires, it leaves the source
        attached to this worker. Cleanup here happens only after the blocking
        read has returned and the loop no longer touches the device. The
        ``stop_timeout`` setting bounds the caller's wait; it cannot guarantee
        cancellation of a backend-native read.
        """
        try:
            self._capture_loop()
        finally:
            with self._lock:
                if self._deferred_source_release:
                    source = self._take_source_unsafe()
                else:
                    source = (None, None)
            if source[0] is not None or source[1] is not None:
                self._release_source(source)
                with self._lock:
                    self._deferred_source_release = False
                    self._update_status_unsafe(CameraStatus.DISCONNECTED)

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
                frame: NDArray[Any] | None = None
                frame_timestamp: float | None = None
                camera_read_seconds: float | None = None
                camera_read_started_ns = 0
                camera_read_finished_ns = 0

                if self._use_kinect2:
                    # Get frame from Kinect v2
                    logger.debug("Attempting to get Kinect v2 frame")
                    kinect_capture = self._kinect2
                    if kinect_capture is None:
                        self._update_status(CameraStatus.ERROR)
                        continue
                    if self._audit_timing_enabled:
                        camera_read_started_ns = time.perf_counter_ns()
                    kinect_frame = kinect_capture.get_latest_frame()
                    if self._audit_timing_enabled:
                        camera_read_finished_ns = time.perf_counter_ns()
                        camera_read_seconds = (
                            max(0, camera_read_finished_ns - camera_read_started_ns)
                            / 1e9
                        )
                    if kinect_frame and kinect_frame.color is not None:
                        frame = kinect_frame.color
                        frame_timestamp = time.time()
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
                    # Get frame from regular camera or video file
                    logger.debug("Attempting to read frame from camera/video")
                    capture = self._cap
                    if capture is None:
                        self._update_status(CameraStatus.ERROR)
                        continue

                    # Check if we've reached the end frame for video files
                    if self._is_video_file and self._video_end_frame is not None:
                        current_frame_pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                        if current_frame_pos >= self._video_end_frame:
                            if self._loop_video:
                                logger.info(
                                    f"Reached end frame {self._video_end_frame}, looping to frame {self._video_start_frame}..."
                                )
                                capture.set(
                                    cv2.CAP_PROP_POS_FRAMES, self._video_start_frame
                                )
                                continue
                            else:
                                logger.info(
                                    f"Reached end frame {self._video_end_frame}"
                                )
                                self._update_status(CameraStatus.DISCONNECTED)
                                break

                    if self._audit_timing_enabled:
                        camera_read_started_ns = time.perf_counter_ns()
                    ret, frame = capture.read()
                    if self._audit_timing_enabled:
                        camera_read_finished_ns = time.perf_counter_ns()
                        camera_read_seconds = (
                            max(
                                0,
                                camera_read_finished_ns - camera_read_started_ns,
                            )
                            / 1e9
                        )
                    if ret and frame is not None:
                        frame_timestamp = time.time()

                    # A native read may only return after stop has timed out.
                    # Do not publish that frame or transition back to CONNECTED.
                    if self._stop_event.is_set():
                        break

                    if not ret or frame is None:
                        # Check if this is end of video file
                        if self._is_video_file:
                            if self._loop_video:
                                logger.info(
                                    f"End of video reached, looping to frame {self._video_start_frame}..."
                                )
                                capture.set(
                                    cv2.CAP_PROP_POS_FRAMES, self._video_start_frame
                                )
                                continue
                            else:
                                logger.info("End of video reached")
                                self._update_status(CameraStatus.DISCONNECTED)
                                break

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
                            self._update_status(CameraStatus.ERROR)
                            break

                    frame_count += 1
                    if (
                        frame_count % self._frame_log_interval == 0
                    ):  # Log at configured interval
                        logger.debug(
                            f"Frame captured successfully: shape={frame.shape}, count={frame_count}"
                        )

                # The timestamp is software retrieval time, not exposure time.
                # Take it after the possibly blocking backend read completes.
                current_time = (
                    frame_timestamp if frame_timestamp is not None else time.time()
                )

                # A transient read failure may leave the device open. Restore
                # connected health when a later real frame proves acquisition
                # has resumed instead of leaving consumers stuck in reconnecting.
                if self._status != CameraStatus.CONNECTED:
                    self._update_status(CameraStatus.CONNECTED)

                # Update statistics
                self._frames_captured += 1
                self._last_frame_time = current_time

                # Calculate FPS
                self._fps_tracker.append(current_time)
                if len(self._fps_tracker) > self._fps_tracker_size:
                    self._fps_tracker.pop(0)

                # Add frame to queue
                frame_info = FrameInfo(
                    frame_number=self._frames_captured,
                    timestamp=current_time,
                    size=(frame.shape[1], frame.shape[0]),
                    channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                    camera_read_seconds=camera_read_seconds,
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
                    time.sleep(max(0, 1.0 / self._fps - self._frame_sleep_compensation))

            except Exception as e:
                if self._stop_event.is_set():
                    break
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

    def _take_source_unsafe(
        self,
    ) -> tuple[Optional[cv2.VideoCapture], Optional[Kinect2Capture]]:
        """Detach the source references while holding the camera state lock."""
        camera, kinect = self._cap, self._kinect2
        self._cap = None
        self._kinect2 = None
        return camera, kinect

    @staticmethod
    def _release_source(
        source: tuple[Optional[cv2.VideoCapture], Optional[Kinect2Capture]],
    ) -> None:
        """Release detached backend resources without holding the state lock.

        Callers must first establish that no capture worker is using the
        source. Native cleanup can invoke callbacks or block, so it must stay
        outside the state lock.
        """
        camera, kinect = source
        if kinect is not None:
            kinect.stop_capture()
        if camera is not None:
            camera.release()

    def start_capture(self) -> bool:
        """Start camera capture.

        Returns:
            True if capture started successfully, False otherwise
        """
        logger.info("start_capture called")
        with self._lifecycle_lock:
            with self._lock:
                if self._capture_thread and self._capture_thread.is_alive():
                    if self._stop_event.is_set():
                        logger.warning(
                            "Capture worker is still stopping; refusing overlapping restart"
                        )
                        return False
                    logger.warning("Capture already running")
                    return True

                # A worker can end naturally at replay EOF or after an
                # unrecoverable source error. Detach its source here and
                # release it below, outside the status lock.
                if self._capture_thread is not None:
                    self._capture_thread = None
                previous_source = self._take_source_unsafe()

                logger.info("Clearing stop event and updating status to CONNECTING")
                self._stop_event.clear()
                self._update_status_unsafe(CameraStatus.CONNECTING)

                # Reset statistics and the bounded frame queue for the new
                # capture session.
                self._start_time = time.time()
                self._frames_captured = 0
                self._frames_dropped = 0
                self._fps_tracker.clear()
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break

            self._release_source(previous_source)

            # Backend startup can block or call status callbacks. Do not hold
            # the state lock while opening/configuring the source.
            logger.info("Attempting to connect to camera")
            if not self._connect_camera():
                logger.error("Camera connection failed")
                self._update_status(CameraStatus.ERROR)
                return False

            logger.info("Camera connected, starting capture thread")
            with self._lock:
                self._capture_thread = threading.Thread(
                    target=self._capture_worker, name="CameraCapture", daemon=True
                )
                self._capture_thread.start()
                self._update_status_unsafe(CameraStatus.CONNECTED)
            logger.info("Camera capture started successfully")
            return True

    def stop_capture(self) -> None:
        """Stop camera capture."""
        # Never hold the state lock while joining: the worker can need it to
        # publish a final status transition before it exits. Also release the
        # source if the worker ended naturally (for example, at replay EOF).
        with self._lifecycle_lock:
            with self._lock:
                capture_thread = self._capture_thread
                if capture_thread and capture_thread.is_alive():
                    logger.info("Stopping camera capture...")
                    self._stop_event.set()
                else:
                    logger.info("Capture not running")

            # Join outside the state lock so worker status callbacks can finish.
            if capture_thread and capture_thread.is_alive():
                capture_thread.join(timeout=self._stop_timeout)
                if capture_thread.is_alive():
                    logger.warning(
                        f"Capture thread remains the source owner after {self._stop_timeout}s stop timeout"
                    )
                    with self._lock:
                        # Recheck after acquiring the lock: the worker may have
                        # completed between the first liveness check and here.
                        if capture_thread.is_alive():
                            self._deferred_source_release = True
                            self._last_error = (
                                "Capture worker did not stop before the configured timeout; "
                                "the source remains owned until its native read returns"
                            )
                            self._update_status_unsafe(CameraStatus.ERROR)
                            return

            with self._lock:
                source = self._take_source_unsafe()
                if self._capture_thread is capture_thread:
                    self._capture_thread = None
                self._deferred_source_release = False

            self._release_source(source)
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
        skipped_frames = 0
        try:
            while True:
                next_frame = self._frame_queue.get_nowait()
                if latest is not None:
                    skipped_frames += 1
                latest = next_frame
        except queue.Empty:
            if latest is not None:
                latest[1].capture_queue_frames_skipped = skipped_frames
                latest[1].capture_queue_depth_after_drain = self._frame_queue.qsize()
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
                depth_range_str = (
                    f"{self._kinect2_min_depth}-{self._kinect2_max_depth}mm"
                )
                return {
                    "device_id": "kinect2",
                    "backend": self._backend,
                    "device_type": "Microsoft Kinect v2",
                    "has_depth": True,
                    "has_color": True,
                    "depth_range": kinect_info.get("depth_range", depth_range_str),
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

    def get_kinect_calibration(self) -> Optional[dict[str, Any]]:
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.stop_capture()

    def __del__(self) -> None:
        """Destructor - ensure resources are cleaned up."""
        with contextlib.suppress(Exception):
            self.stop_capture()
