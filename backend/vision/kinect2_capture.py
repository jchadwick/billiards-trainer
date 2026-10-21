"""Kinect v2 capture interface for enhanced depth-based vision.

Provides enhanced billiards table detection using depth information from Kinect v2,
improving accuracy in challenging lighting conditions and enabling 3D ball tracking.

Requirements:
- libfreenect2 with Python bindings
- Kinect v2 sensor connected via USB 3.0
"""

import contextlib
import importlib
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from types import ModuleType, TracebackType
from typing import Any, Callable, Optional, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray


class _CameraParameters(Protocol):
    """Camera intrinsics exposed by pylibfreenect2."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float


class _RawFrame(Protocol):
    def asarray(self) -> NDArray[Any]: ...


class _FrameSet(Protocol):
    def __bool__(self) -> bool: ...

    def __contains__(self, frame_type: int) -> bool: ...

    def getFrame(self, frame_type: int) -> _RawFrame: ...


class _FrameListener(Protocol):
    def waitForNewFrame(self, timeout_ms: int) -> _FrameSet | None: ...

    def release(self, frames: _FrameSet) -> None: ...


class _KinectDevice(Protocol):
    def setColorFrameListener(self, listener: _FrameListener) -> None: ...

    def setIrAndDepthFrameListener(self, listener: _FrameListener) -> None: ...

    def start(self) -> None: ...

    def startStreams(self, rgb: bool, depth: bool) -> None: ...

    def getIrCameraParams(self) -> _CameraParameters: ...

    def getColorCameraParams(self) -> _CameraParameters: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _Freenect2(Protocol):
    def enumerateDevices(self) -> int: ...

    def getDeviceSerialNumber(self, index: int) -> str: ...

    def openDevice(self, serial: str) -> _KinectDevice | None: ...


class _FrameTypeCodes(Protocol):
    Color: int
    Depth: int
    Ir: int


class _Kinect2SDK(Protocol):
    FrameType: _FrameTypeCodes
    Freenect2: Callable[[], _Freenect2]
    SyncMultiFrameListener: Callable[[int], _FrameListener]
    Registration: Callable[[_CameraParameters, _CameraParameters], object]


# Kinect v2 dependencies are optional because this SDK is hardware-specific
# and unavailable on many development and CI machines. Importing dynamically
# avoids pretending the optional extension has installed type stubs; the
# protocol above describes the exact subset consumed by this module.
try:
    _kinect2_module: ModuleType | None = importlib.import_module("pylibfreenect2")
except ImportError:
    _kinect2_module = None

KINECT2_AVAILABLE = _kinect2_module is not None
logger = logging.getLogger(__name__)


def _load_kinect2_sdk() -> _Kinect2SDK:
    if _kinect2_module is None:
        raise RuntimeError("pylibfreenect2 is not installed")
    return cast("_Kinect2SDK", _kinect2_module)


class Kinect2Status(Enum):
    """Kinect v2 connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    NOT_AVAILABLE = "not_available"


@dataclass
class Kinect2FrameInfo:
    """Kinect v2 frame metadata with depth information."""

    frame_number: int
    timestamp: float
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    has_depth: bool
    has_color: bool
    depth_range: tuple[float, float]  # min, max depth in mm


@dataclass
class Kinect2Frame:
    """Combined color and depth frame from Kinect v2."""

    color: Optional[NDArray[np.float64]]  # RGB color image
    depth: Optional[NDArray[np.float64]]  # Depth map in millimeters
    infrared: Optional[NDArray[np.float64]]  # Infrared image
    frame_info: Kinect2FrameInfo


class Kinect2Capture:
    """Kinect v2 capture interface with depth and color streams.

    Features:
    - Synchronized color and depth capture
    - 3D ball position tracking
    - Enhanced table detection using depth
    - Improved lighting tolerance
    - Automatic calibration assistance
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Kinect v2 capture.

        Args:
            config: Configuration dictionary with keys:
                - enable_color: Enable color stream (default: True)
                - enable_depth: Enable depth stream (default: True)
                - enable_infrared: Enable infrared stream (default: False)
                - color_resolution: Color resolution (default: "1080p")
                - depth_mode: Depth processing mode (default: "default")
                - min_depth: Minimum depth in mm (default: 500)
                - max_depth: Maximum depth in mm (default: 4000)
                - depth_smoothing: Apply depth smoothing (default: True)
                - auto_reconnect: Enable automatic reconnection (default: True)
                - stop_timeout: Maximum time to wait for capture shutdown (default: 5.0)
        """
        if not KINECT2_AVAILABLE:
            raise RuntimeError(
                "Kinect v2 support not available. Please install libfreenect2 "
                "and pylibfreenect2: pip install pylibfreenect2"
            )

        self._sdk = _load_kinect2_sdk()
        self._config = config
        self._enable_color = config.get("enable_color", True)
        self._enable_depth = config.get("enable_depth", True)
        self._enable_infrared = config.get("enable_infrared", False)
        self._color_resolution = config.get("color_resolution", "1080p")
        self._depth_mode = config.get("depth_mode", "default")
        self._min_depth = config.get("min_depth", 500)  # mm
        self._max_depth = config.get("max_depth", 4000)  # mm
        self._depth_smoothing = config.get("depth_smoothing", True)
        self._auto_reconnect = config.get("auto_reconnect", True)
        self._stop_timeout = config.get("stop_timeout", 5.0)
        if (
            isinstance(self._stop_timeout, bool)
            or not isinstance(self._stop_timeout, (int, float))
            or not math.isfinite(self._stop_timeout)
            or self._stop_timeout <= 0
        ):
            raise ValueError("stop_timeout must be finite and positive")

        # Kinect v2 objects
        self._freenect2: _Freenect2 | None = None
        self._device: _KinectDevice | None = None
        self._listener: _FrameListener | None = None
        self._registration: object | None = None

        # Capture state
        self._status = Kinect2Status.DISCONNECTED
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[Kinect2Frame] = queue.Queue(maxsize=5)

        # Statistics
        self._start_time = time.time()
        self._frames_captured = 0
        self._frames_dropped = 0
        self._last_frame_time = 0.0
        self._error_count = 0
        self._last_error: str | None = None
        self._connection_attempts = 0

        # Thread safety
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        # Callbacks
        self._status_callback: Callable[[Kinect2Status], None] | None = None

        logger.info(
            f"Kinect v2 capture initialized - "
            f"color: {self._enable_color}, depth: {self._enable_depth}, "
            f"infrared: {self._enable_infrared}"
        )

    def set_status_callback(self, callback: Callable[[Kinect2Status], None]) -> None:
        """Set callback function for status changes."""
        self._status_callback = callback

    def _update_status(self, status: Kinect2Status) -> None:
        """Update Kinect v2 status and notify callback."""
        with self._lock:
            self._update_status_unsafe(status)

    def _update_status_unsafe(self, status: Kinect2Status) -> None:
        """Update status when the caller already holds ``_lock``."""
        if self._status != status:
            old_status = self._status
            self._status = status
            logger.info(f"Kinect v2 status changed: {old_status} -> {status}")

            if self._status_callback:
                try:
                    self._status_callback(status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

    def _connect_kinect(self) -> bool:
        """Establish Kinect v2 connection."""
        try:
            self._connection_attempts += 1
            logger.info(
                f"Connecting to Kinect v2 (attempt {self._connection_attempts})"
            )

            self._freenect2 = self._sdk.Freenect2()
            if self._freenect2.enumerateDevices() == 0:
                raise RuntimeError("No Kinect v2 devices found")

            serial = self._freenect2.getDeviceSerialNumber(0)
            logger.info(f"Found Kinect v2 device: {serial}")

            self._device = self._freenect2.openDevice(serial)
            if self._device is None:
                raise RuntimeError("Unable to open Kinect v2 device")

            frame_types = 0
            if self._enable_color:
                frame_types |= self._sdk.FrameType.Color
            if self._enable_depth:
                frame_types |= self._sdk.FrameType.Depth
            if self._enable_infrared:
                frame_types |= self._sdk.FrameType.Ir

            self._listener = self._sdk.SyncMultiFrameListener(frame_types)
            self._device.setColorFrameListener(self._listener)
            self._device.setIrAndDepthFrameListener(self._listener)

            if self._enable_color:
                self._device.start()
            else:
                self._device.startStreams(rgb=False, depth=True)

            if self._enable_color and self._enable_depth:
                self._registration = self._sdk.Registration(
                    self._device.getIrCameraParams(),
                    self._device.getColorCameraParams(),
                )

            frames = self._listener.waitForNewFrame(timeout_ms=5000)
            if not frames:
                raise RuntimeError("Failed to capture test frames from Kinect v2")
            self._listener.release(frames)

            logger.info(
                f"Kinect v2 connected successfully (attempt {self._connection_attempts})"
            )
            return True

        except Exception as e:
            logger.error(f"Kinect v2 connection failed: {e}")
            self._last_error = str(e)
            self._error_count += 1
            self._cleanup_device()
            return False

    def _cleanup_device(self) -> None:
        """Clean up Kinect v2 resources."""
        try:
            if self._device is not None:
                self._device.stop()
                self._device.close()
                self._device = None

            self._listener = None
            self._registration = None
            self._freenect2 = None
        except Exception as e:
            logger.error(f"Error during Kinect v2 cleanup: {e}")

    def _process_frames(self, frames: _FrameSet) -> Optional[Kinect2Frame]:
        """Process raw Kinect v2 frames into usable format."""
        try:
            color_frame: NDArray[Any] | None = None
            depth_frame: NDArray[Any] | None = None
            infrared_frame: NDArray[Any] | None = None
            depth_range: tuple[float, float] = (0.0, 0.0)

            if self._enable_color and self._sdk.FrameType.Color in frames:
                color_raw = frames.getFrame(self._sdk.FrameType.Color)
                color_frame = cv2.cvtColor(color_raw.asarray(), cv2.COLOR_BGRA2RGB)

            if self._enable_depth and self._sdk.FrameType.Depth in frames:
                depth_raw = frames.getFrame(self._sdk.FrameType.Depth)
                depth_frame = depth_raw.asarray()
                depth_frame = np.where(
                    (depth_frame >= self._min_depth) & (depth_frame <= self._max_depth),
                    depth_frame,
                    0,
                )

                if self._depth_smoothing:
                    depth_frame = cv2.medianBlur(depth_frame.astype(np.float32), 5)

                valid_depth = depth_frame[depth_frame > 0]
                if len(valid_depth) > 0:
                    depth_range = (float(valid_depth.min()), float(valid_depth.max()))

            if self._enable_infrared and self._sdk.FrameType.Ir in frames:
                ir_raw = frames.getFrame(self._sdk.FrameType.Ir)
                infrared_frame = ir_raw.asarray()
                infrared_frame = (infrared_frame / 65535.0 * 255).astype(np.uint8)

            self._frames_captured += 1
            frame_info = Kinect2FrameInfo(
                frame_number=self._frames_captured,
                timestamp=time.time(),
                color_size=color_frame.shape[:2] if color_frame is not None else (0, 0),
                depth_size=depth_frame.shape[:2] if depth_frame is not None else (0, 0),
                has_depth=depth_frame is not None,
                has_color=color_frame is not None,
                depth_range=depth_range,
            )

            return Kinect2Frame(
                color=color_frame,
                depth=depth_frame,
                infrared=infrared_frame,
                frame_info=frame_info,
            )

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._last_error = str(e)
            self._error_count += 1
            return None

    def _capture_loop(self) -> None:
        """Main Kinect v2 capture loop."""
        logger.info("Kinect v2 capture loop started")

        while not self._stop_event.is_set():
            try:
                if self._device is None:
                    if self._auto_reconnect:
                        self._update_status(Kinect2Status.RECONNECTING)
                        if self._connect_kinect():
                            self._update_status(Kinect2Status.CONNECTED)
                        else:
                            time.sleep(2.0)
                            continue
                    else:
                        self._update_status(Kinect2Status.ERROR)
                        break

                listener = self._listener
                if listener is None:
                    raise RuntimeError("Kinect v2 listener is unavailable")
                frames = listener.waitForNewFrame(timeout_ms=1000)
                if not frames:
                    logger.warning("Kinect v2 frame timeout")
                    continue

                kinect_frame = self._process_frames(frames)
                listener.release(frames)
                if kinect_frame is None:
                    continue

                self._last_frame_time = time.time()
                try:
                    self._frame_queue.put_nowait(kinect_frame)
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                        self._frames_dropped += 1
                        self._frame_queue.put_nowait(kinect_frame)
                    except queue.Empty:
                        pass

            except Exception as e:
                logger.error(f"Kinect v2 capture loop error: {e}")
                self._last_error = str(e)
                self._error_count += 1

                if self._auto_reconnect:
                    self._update_status(Kinect2Status.RECONNECTING)
                    time.sleep(2.0)
                else:
                    self._update_status(Kinect2Status.ERROR)
                    break

        logger.info("Kinect v2 capture loop ended")

    def start_capture(self) -> bool:
        """Start Kinect v2 capture."""
        with self._lifecycle_lock:
            with self._lock:
                if self._capture_thread and self._capture_thread.is_alive():
                    if self._stop_event.is_set():
                        logger.warning(
                            "Kinect v2 worker is still stopping; refusing restart"
                        )
                        return False
                    logger.warning("Kinect v2 capture already running")
                    return True

                # A naturally ended worker runs its own cleanup before the
                # thread is considered stopped. Retire its reference before a
                # new device session is opened.
                self._capture_thread = None
                self._stop_event.clear()
                self._update_status_unsafe(Kinect2Status.CONNECTING)
                self._start_time = time.time()
                self._frames_captured = 0
                self._frames_dropped = 0
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        break

            if not self._connect_kinect():
                self._update_status(Kinect2Status.ERROR)
                return False

            with self._lock:
                self._capture_thread = threading.Thread(
                    target=self._capture_worker, name="Kinect2Capture", daemon=True
                )
                self._capture_thread.start()
                self._update_status_unsafe(Kinect2Status.CONNECTED)
            logger.info("Kinect v2 capture started successfully")
            return True

    def _capture_worker(self) -> None:
        """Own Kinect SDK resources until acquisition has completely stopped."""
        try:
            self._capture_loop()
        finally:
            self._cleanup_device()
            self._update_status(Kinect2Status.DISCONNECTED)

    def stop_capture(self) -> None:
        """Stop Kinect v2 capture."""
        with self._lifecycle_lock:
            with self._lock:
                capture_thread = self._capture_thread
                if capture_thread and capture_thread.is_alive():
                    logger.info("Stopping Kinect v2 capture...")
                    self._stop_event.set()
                else:
                    logger.info("Kinect v2 capture not running")

            if capture_thread and capture_thread.is_alive():
                capture_thread.join(timeout=self._stop_timeout)
                if capture_thread.is_alive():
                    with self._lock:
                        if capture_thread.is_alive():
                            self._last_error = (
                                "Kinect worker did not stop before the configured timeout; "
                                "device resources remain owned until acquisition returns"
                            )
                            self._update_status_unsafe(Kinect2Status.ERROR)
                            logger.warning(self._last_error)
                            return

            with self._lock:
                if self._capture_thread is capture_thread:
                    self._capture_thread = None
            # A start failure can leave resources without a worker. Once no
            # worker owns them, cleanup is safe and callbacks run lock-free.
            self._cleanup_device()
            self._update_status(Kinect2Status.DISCONNECTED)
            logger.info("Kinect v2 capture stopped")

    def get_frame(self) -> Optional[Kinect2Frame]:
        """Get latest captured Kinect v2 frame."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_frame(self) -> Optional[Kinect2Frame]:
        """Get most recent Kinect v2 frame, discarding older ones."""
        latest: Kinect2Frame | None = None
        try:
            while True:
                latest = self._frame_queue.get_nowait()
        except queue.Empty:
            return latest

    def is_connected(self) -> bool:
        """Check if Kinect v2 is connected and capturing."""
        with self._lock:
            return self._status == Kinect2Status.CONNECTED and self._device is not None

    def get_status(self) -> Kinect2Status:
        """Get current Kinect v2 status."""
        with self._lock:
            return self._status

    def get_device_info(self) -> dict[str, Any]:
        """Get Kinect v2 device information."""
        if self._device is None:
            return {}

        return {
            "device_type": "Microsoft Kinect v2",
            "color_enabled": self._enable_color,
            "depth_enabled": self._enable_depth,
            "infrared_enabled": self._enable_infrared,
            "depth_range": f"{self._min_depth}-{self._max_depth}mm",
            "color_resolution": self._color_resolution,
            "depth_mode": self._depth_mode,
            "has_registration": self._registration is not None,
        }

    def get_calibration_data(self) -> Optional[dict[str, Any]]:
        """Get Kinect v2 calibration parameters for 3D reconstruction."""
        if self._device is None:
            return None

        try:
            ir_params = self._device.getIrCameraParams()
            color_params = self._device.getColorCameraParams()
            return {
                "ir_camera": {
                    "fx": ir_params.fx,
                    "fy": ir_params.fy,
                    "cx": ir_params.cx,
                    "cy": ir_params.cy,
                    "k1": ir_params.k1,
                    "k2": ir_params.k2,
                    "k3": ir_params.k3,
                    "p1": ir_params.p1,
                    "p2": ir_params.p2,
                },
                "color_camera": {
                    "fx": color_params.fx,
                    "fy": color_params.fy,
                    "cx": color_params.cx,
                    "cy": color_params.cy,
                    "k1": color_params.k1,
                    "k2": color_params.k2,
                    "k3": color_params.k3,
                    "p1": color_params.p1,
                    "p2": color_params.p2,
                },
            }
        except Exception as e:
            logger.error(f"Error getting calibration data: {e}")
            return None

    def depth_to_3d(
        self, u: int, v: int, depth: float
    ) -> Optional[tuple[float, float, float]]:
        """Convert depth pixel to 3D world coordinates."""
        if self._device is None or depth <= 0:
            return None

        try:
            ir_params = self._device.getIrCameraParams()
            x = (u - ir_params.cx) * depth / ir_params.fx
            y = (v - ir_params.cy) * depth / ir_params.fy
            return (x, y, depth)
        except Exception as e:
            logger.error(f"Depth to 3D conversion error: {e}")
            return None

    def __enter__(self) -> "Kinect2Capture":
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
