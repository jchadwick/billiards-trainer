"""Vision Module - Computer vision processing for billiards analysis.

This module provides comprehensive computer vision capabilities for detecting and tracking
pool table elements including table boundaries, balls, and cue sticks in real-time.
"""

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .calibration.camera import CameraCalibrator
from .calibration.color import ColorCalibrator
from .calibration.geometry import GeometricCalibrator
from .capture import CameraCapture, CameraHealth, CameraStatus, FrameInfo
from .config_manager import VisionConfigurationManager, create_vision_config_manager
from .detection.balls import BallDetector
from .detection.cue import CueDetector
from .detection.detector_factory import BaseDetector, create_detector
from .detection.table import TableDetector

# Optional camera modules
try:
    from .direct_camera import DirectCameraModule
except ImportError:
    DirectCameraModule = None

# Import vision components
from .models import (
    Ball,
    BallType,
    CalibrationData,
    CameraCalibration,
    CameraFrame,
    ColorCalibration,
    CueState,
    CueStick,
    DetectionResult,
    DetectionSession,
    FrameStatistics,
    GeometricCalibration,
    Pocket,
    PocketType,
    ShotEvent,
    Table,
)
from .preprocessing import ImagePreprocessor

try:
    from .simple_camera import SimpleCameraModule
except ImportError:
    SimpleCameraModule = None

from .tracking.tracker import ObjectTracker

# Set up logging
logger = logging.getLogger(__name__)


def _get_config_value(key_path: str, default: Any) -> Any:
    """Get configuration value from the config system.

    Args:
        key_path: Dot-separated path to config value (e.g., "vision.processing.target_fps")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    try:
        from ..config.manager import ConfigurationModule

        config_mgr = ConfigurationModule()
        return config_mgr.get(key_path, default)
    except Exception:
        return default


@dataclass
class VisionConfig:
    """Configuration for Vision Module.

    All default values are loaded from the configuration system.
    This class provides a structured interface to vision configuration parameters.
    """

    # Camera settings (from vision.camera.*)
    camera_device_id: int
    camera_backend: str
    camera_resolution: tuple[int, int]
    camera_fps: int
    camera_buffer_size: int

    # Processing settings (from vision.processing.*)
    target_fps: int
    enable_threading: bool
    enable_gpu: bool
    max_frame_queue_size: int

    # Detection settings (from vision.detection.*)
    enable_table_detection: bool
    enable_ball_detection: bool
    enable_cue_detection: bool
    enable_tracking: bool

    # Detector backend configuration (from vision.detection.*)
    detection_backend: str
    use_opencv_validation: bool
    fallback_to_opencv: bool

    # Performance settings (from vision.processing.*)
    frame_skip: int
    roi_enabled: bool
    preprocessing_enabled: bool

    # Background subtraction (from vision.detection.background_subtraction.*)
    background_image_path: Optional[str]
    use_background_subtraction: bool
    background_threshold: int

    # Debug settings (from vision.debug*)
    debug_mode: bool
    save_debug_images: bool
    debug_output_path: str

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> "VisionConfig":
        """Create VisionConfig from configuration dictionary.

        Args:
            config_dict: Configuration dictionary (can be empty, will use defaults)

        Returns:
            VisionConfig instance with values from config or defaults
        """
        # Import here to avoid circular dependencies
        from ..config.manager import ConfigurationModule

        # Get configuration manager (will load from default.json)
        config_mgr = ConfigurationModule()

        # Extract values from config dict or use config manager defaults
        return cls(
            # Camera settings
            camera_device_id=config_dict.get(
                "camera_device_id", config_mgr.get("vision.camera.device_id", 0)
            ),
            camera_backend=config_dict.get(
                "camera_backend", config_mgr.get("vision.camera.backend", "auto")
            ),
            camera_resolution=tuple(
                config_dict.get(
                    "camera_resolution",
                    config_mgr.get("vision.camera.resolution", [1920, 1080]),
                )
            ),
            camera_fps=config_dict.get(
                "camera_fps", config_mgr.get("vision.camera.fps", 30)
            ),
            camera_buffer_size=config_dict.get(
                "camera_buffer_size", config_mgr.get("vision.camera.buffer_size", 1)
            ),
            # Processing settings
            target_fps=config_dict.get(
                "target_fps", config_mgr.get("vision.processing.target_fps", 30)
            ),
            enable_threading=config_dict.get(
                "enable_threading",
                config_mgr.get("vision.processing.enable_threading", True),
            ),
            enable_gpu=config_dict.get(
                "enable_gpu", config_mgr.get("vision.processing.use_gpu", False)
            ),
            max_frame_queue_size=config_dict.get(
                "max_frame_queue_size",
                config_mgr.get("vision.processing.max_frame_queue_size", 5),
            ),
            # Detection settings
            enable_table_detection=config_dict.get(
                "enable_table_detection",
                config_mgr.get("vision.detection.enable_table_detection", True),
            ),
            enable_ball_detection=config_dict.get(
                "enable_ball_detection",
                config_mgr.get("vision.detection.enable_ball_detection", True),
            ),
            enable_cue_detection=config_dict.get(
                "enable_cue_detection",
                config_mgr.get("vision.detection.enable_cue_detection", True),
            ),
            enable_tracking=config_dict.get(
                "enable_tracking",
                config_mgr.get("vision.processing.enable_tracking", True),
            ),
            # Detector backend configuration
            detection_backend=config_dict.get(
                "detection_backend",
                config_mgr.get("vision.detection.detection_backend", "opencv"),
            ),
            use_opencv_validation=config_dict.get(
                "use_opencv_validation",
                config_mgr.get("vision.detection.use_opencv_validation", False),
            ),
            fallback_to_opencv=config_dict.get(
                "fallback_to_opencv",
                config_mgr.get("vision.detection.fallback_to_opencv", True),
            ),
            # Performance settings
            frame_skip=config_dict.get(
                "frame_skip", config_mgr.get("vision.processing.frame_skip", 0)
            ),
            roi_enabled=config_dict.get(
                "roi_enabled", config_mgr.get("vision.processing.roi_enabled", False)
            ),
            preprocessing_enabled=config_dict.get(
                "preprocessing_enabled",
                config_mgr.get("vision.processing.enable_preprocessing", True),
            ),
            # Background subtraction
            background_image_path=config_dict.get(
                "background_image_path",
                config_mgr.get(
                    "vision.detection.background_subtraction.background_image_path",
                    None,
                ),
            ),
            use_background_subtraction=config_dict.get(
                "use_background_subtraction",
                config_mgr.get(
                    "vision.detection.background_subtraction.enabled", False
                ),
            ),
            background_threshold=config_dict.get(
                "background_threshold",
                config_mgr.get("vision.detection.background_subtraction.threshold", 30),
            ),
            # Debug settings
            debug_mode=config_dict.get(
                "debug_mode", config_mgr.get("vision.debug", False)
            ),
            save_debug_images=config_dict.get(
                "save_debug_images", config_mgr.get("vision.save_debug_images", False)
            ),
            debug_output_path=config_dict.get(
                "debug_output_path",
                config_mgr.get("vision.debug_output_path", "/tmp/vision_debug"),
            ),
        )


@dataclass
class VisionStatistics:
    """Vision processing statistics."""

    frames_processed: int = 0
    frames_dropped: int = 0
    avg_processing_time: float = 0.0
    avg_fps: float = 0.0
    detection_accuracy: dict[str, float] = None
    last_error: Optional[str] = None
    uptime: float = 0.0

    def __post_init__(self):
        if self.detection_accuracy is None:
            self.detection_accuracy = {"table": 0.0, "balls": 0.0, "cue": 0.0}


class VisionModuleError(Exception):
    """Base exception for Vision Module errors."""

    pass


class VisionModule:
    """Main Vision Module - Computer vision processing interface.

    Provides real-time computer vision processing for billiards table analysis
    including table detection, ball tracking, and cue stick detection.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the Vision Module.

        Args:
            config: Configuration dictionary for the vision module
        """
        # Parse configuration - use from_config_dict to load defaults from config system
        if config is None:
            config = {}
        self.config = VisionConfig.from_config_dict(config)

        # Initialize statistics
        self.stats = VisionStatistics()
        self._start_time = time.time()

        # Initialize components
        self._initialize_components()

        # Threading and state management
        self._capture_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._frame_queue = asyncio.Queue(maxsize=self.config.max_frame_queue_size)
        self._result_queue = asyncio.Queue(maxsize=self.config.max_frame_queue_size)
        self._lock = threading.Lock()

        # Current state
        self._current_frame: Optional[NDArray[np.float64]] = None
        self._current_result: Optional[DetectionResult] = None
        self._frame_number = 0

        # Event callbacks
        self._callbacks: dict[str, list[Callable[..., Any]]] = {
            "frame_processed": [],
            "detection_complete": [],
            "error_occurred": [],
        }

        # Region of interest
        self._roi_corners: Optional[list[tuple[int, int]]] = None

        logger.info("Vision Module initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all vision processing components."""
        try:
            logger.info("Initializing vision components")

            # Camera capture
            camera_config = {
                "device_id": self.config.camera_device_id,
                "backend": self.config.camera_backend,
                "resolution": self.config.camera_resolution,
                "fps": self.config.camera_fps,
                "buffer_size": self.config.camera_buffer_size,
            }
            logger.info(f"Creating CameraCapture with config: {camera_config}")
            self.camera = CameraCapture(camera_config)
            logger.info("CameraCapture created successfully")

            # Image preprocessing - create config based on vision module settings
            preprocessing_config = {}
            if hasattr(self.config, "enable_gpu"):
                # Note: PreprocessingConfig doesn't have a use_gpu field, so we skip it
                pass
            logger.debug("Creating ImagePreprocessor")
            self.preprocessor = ImagePreprocessor(preprocessing_config)

            # Detection components
            # Note: use_background_subtraction and background_threshold are passed to detector
            # instances after creation, not in the config
            base_detection_config = {
                "debug_mode": self.config.debug_mode,
            }

            # Initialize unified detector using factory
            # The detector handles both ball and cue detection
            if self.config.enable_ball_detection or self.config.enable_cue_detection:
                # Get detection backend from config (default to 'opencv')
                backend = getattr(self.config, "detection_backend", "opencv")

                # Create detector configuration
                detector_config = {
                    "ball_detection": (
                        base_detection_config.copy()
                        if self.config.enable_ball_detection
                        else {}
                    ),
                    "cue_detection": (
                        base_detection_config.copy()
                        if self.config.enable_cue_detection
                        else {}
                    ),
                }

                logger.info(f"Creating detector with backend: {backend}")
                self.detector: Optional[BaseDetector] = create_detector(
                    backend, detector_config
                )

                # Set background subtraction parameters on detector instances
                # (these are instance variables, not config)
                if (
                    hasattr(self.detector, "ball_detector")
                    and self.detector.ball_detector
                ):
                    self.detector.ball_detector.use_background_subtraction = (
                        self.config.use_background_subtraction
                    )
                    self.detector.ball_detector.background_threshold = (
                        self.config.background_threshold
                    )
                if (
                    hasattr(self.detector, "cue_detector")
                    and self.detector.cue_detector
                ):
                    self.detector.cue_detector.use_background_subtraction = (
                        self.config.use_background_subtraction
                    )
                    self.detector.cue_detector.background_threshold = (
                        self.config.background_threshold
                    )

                # Check if OpenCV validation is enabled for hybrid approach
                self.use_opencv_validation = getattr(
                    self.config, "use_opencv_validation", False
                )
                self.fallback_to_opencv = getattr(
                    self.config, "fallback_to_opencv", True
                )

                # Keep legacy ball_detector and cue_detector references for compatibility
                # These point to the underlying detectors if using OpenCV backend
                if backend.lower() == "opencv":
                    from .detection.detector_factory import OpenCVDetector

                    if isinstance(self.detector, OpenCVDetector):
                        self.ball_detector = (
                            self.detector.ball_detector
                            if self.config.enable_ball_detection
                            else None
                        )
                        self.cue_detector = (
                            self.detector.cue_detector
                            if self.config.enable_cue_detection
                            else None
                        )
                    else:
                        self.ball_detector = None
                        self.cue_detector = None
                else:
                    # For YOLO or other backends, keep legacy references as None
                    # or create OpenCV fallback detectors
                    if self.fallback_to_opencv:
                        logger.info("Creating OpenCV fallback detectors")
                        if self.config.enable_ball_detection:
                            self.ball_detector = BallDetector(base_detection_config)
                            # Set background subtraction parameters
                            self.ball_detector.use_background_subtraction = (
                                self.config.use_background_subtraction
                            )
                            self.ball_detector.background_threshold = (
                                self.config.background_threshold
                            )
                        else:
                            self.ball_detector = None

                        if self.config.enable_cue_detection:
                            self.cue_detector = CueDetector(base_detection_config)
                            # Set background subtraction parameters
                            self.cue_detector.use_background_subtraction = (
                                self.config.use_background_subtraction
                            )
                            self.cue_detector.background_threshold = (
                                self.config.background_threshold
                            )
                        else:
                            self.cue_detector = None
                    else:
                        self.ball_detector = None
                        self.cue_detector = None
            else:
                self.detector = None
                self.ball_detector = None
                self.cue_detector = None
                self.use_opencv_validation = False
                self.fallback_to_opencv = False

            # Table detection (separate from ball/cue detection)
            if self.config.enable_table_detection:
                self.table_detector = TableDetector(base_detection_config)
            else:
                self.table_detector = None

            # Load background frame if configured
            if self.config.background_image_path:
                self._load_background_frame(self.config.background_image_path)

            # Tracking
            if self.config.enable_tracking:
                # Import here to avoid circular dependencies
                from ..config.manager import ConfigurationModule

                config_mgr = ConfigurationModule()
                tracking_config = config_mgr.get("vision.tracking", {})
                self.tracker = ObjectTracker(tracking_config)
            else:
                self.tracker = None

            # Calibration components
            self.camera_calibrator = CameraCalibrator()
            self.color_calibrator = ColorCalibrator()
            self.geometry_calibrator = GeometricCalibrator()

            logger.info("All vision components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vision components: {e}")
            raise VisionModuleError(f"Component initialization failed: {e}")

    def start_capture(self) -> bool:
        """Start camera capture and processing.

        Returns:
            True if capture started successfully
        """
        try:
            logger.info("VisionModule.start_capture called")

            if self._is_running:
                logger.warning("Capture is already running")
                return True

            # Start camera
            logger.info("Starting camera capture...")
            if not self.camera.start_capture():
                logger.error("Camera.start_capture returned False")
                raise VisionModuleError("Failed to start camera capture")

            logger.info("Camera capture started, initializing vision module state")

            # Reset statistics
            self.stats = VisionStatistics()
            self._start_time = time.time()
            self._frame_number = 0

            # Start processing threads
            self._is_running = True

            if self.config.enable_threading:
                logger.info("Starting vision processing threads")
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, name="VisionCapture", daemon=True
                )
                self._processing_thread = threading.Thread(
                    target=self._processing_loop, name="VisionProcessing", daemon=True
                )

                self._capture_thread.start()
                logger.debug("Capture thread started")
                self._processing_thread.start()
                logger.debug("Processing thread started")
            else:
                logger.info("Threading disabled, running in single-threaded mode")

            logger.info("Vision capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start capture: {e}", exc_info=True)
            self.stats.last_error = str(e)
            self._emit_event("error_occurred", {"error": str(e)})
            return False

    def stop_capture(self) -> None:
        """Stop camera capture and processing."""
        try:
            if not self._is_running:
                logger.warning("Capture is not running")
                return

            # Signal shutdown
            self._is_running = False

            # Stop camera
            self.camera.stop_capture()

            # Wait for threads to finish
            thread_timeout = _get_config_value(
                "vision.processing.thread_shutdown_timeout_sec", 5.0
            )
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=thread_timeout)

            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=thread_timeout)

            # Clear queues
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            logger.info("Vision capture stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping capture: {e}")
            self.stats.last_error = str(e)

    def process_frame(self) -> Optional[DetectionResult]:
        """Process single frame and return detections.

        Returns:
            Detection results or None if no frame available
        """
        try:
            if not self._is_running:
                # Single frame processing mode
                frame_data = self.camera.get_latest_frame()
                if frame_data is None:
                    return None

                frame, frame_info = frame_data
                return self._process_single_frame(
                    frame, frame_info.frame_number, frame_info.timestamp
                )
            else:
                # Return latest result from processing thread
                with self._lock:
                    return self._current_result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.stats.last_error = str(e)
            self._emit_event("error_occurred", {"error": str(e)})
            return None

    def get_current_frame(self) -> Optional[NDArray[np.float64]]:
        """Get latest captured frame.

        Returns:
            Latest frame or None if no frame available
        """
        logger.debug("get_current_frame called")
        with self._lock:
            if self._current_frame is not None:
                logger.debug(
                    f"Returning current frame: shape={self._current_frame.shape}"
                )
                return self._current_frame.copy()
            else:
                logger.debug("No current frame available")
                return None

    def calibrate_camera(self) -> bool:
        """Perform camera calibration.

        Returns:
            True if calibration successful
        """
        try:
            logger.info("Starting camera calibration")

            # Capture calibration images
            num_frames = _get_config_value("vision.calibration.camera.num_frames", 20)
            min_frames_required = _get_config_value(
                "vision.calibration.camera.min_frames_required", 10
            )
            frame_capture_delay = _get_config_value(
                "vision.calibration.camera.frame_capture_delay_sec", 0.1
            )

            calibration_frames = []
            for _i in range(num_frames):
                frame_data = self.camera.get_latest_frame()
                if frame_data is not None:
                    frame, frame_info = frame_data
                    calibration_frames.append(frame)
                time.sleep(frame_capture_delay)

            if len(calibration_frames) < min_frames_required:
                raise VisionModuleError("Insufficient frames for calibration")

            # Perform calibration
            success = self.camera_calibrator.calibrate(calibration_frames)

            if success:
                logger.info("Camera calibration completed successfully")
            else:
                logger.error("Camera calibration failed")

            return success

        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            self.stats.last_error = str(e)
            return False

    def calibrate_colors(
        self, sample_image: Optional[NDArray[np.float64]] = None
    ) -> dict:
        """Auto-calibrate color thresholds.

        Args:
            sample_image: Image to use for calibration (uses current frame if None)

        Returns:
            Calibration results
        """
        try:
            if sample_image is None:
                sample_image = self.get_current_frame()

            if sample_image is None:
                raise VisionModuleError("No image available for color calibration")

            logger.info("Starting color calibration")

            # Perform color calibration
            calibration_results = self.color_calibrator.auto_calibrate(sample_image)

            logger.info("Color calibration completed successfully")
            return calibration_results

        except Exception as e:
            logger.error(f"Color calibration failed: {e}")
            self.stats.last_error = str(e)
            return {}

    def set_roi(self, corners: list[tuple[int, int]]) -> None:
        """Set region of interest for processing.

        Args:
            corners: List of (x, y) corner points defining ROI
        """
        try:
            expected_corner_count = _get_config_value(
                "vision.defaults.roi_corner_count", 4
            )
            expected_corner_dims = _get_config_value(
                "vision.defaults.roi_corner_dimensions", 2
            )

            if len(corners) != expected_corner_count:
                raise ValueError(
                    f"ROI must have exactly {expected_corner_count} corners"
                )

            # Validate corners
            for corner in corners:
                if (
                    not isinstance(corner, (tuple, list))
                    or len(corner) != expected_corner_dims
                ):
                    raise ValueError(
                        f"Each corner must be a ({', '.join(['x'] * expected_corner_dims)}) tuple"
                    )
                if not all(isinstance(coord, (int, float)) for coord in corner):
                    raise ValueError("Corner coordinates must be numeric")

            self._roi_corners = corners
            self.config.roi_enabled = True

            logger.info(f"ROI set to corners: {corners}")

        except Exception as e:
            logger.error(f"Failed to set ROI: {e}")
            self.stats.last_error = str(e)

    def get_statistics(self) -> dict:
        """Get processing statistics.

        Returns:
            Dictionary containing performance and accuracy statistics
        """
        # Update uptime
        self.stats.uptime = time.time() - self._start_time

        # Calculate current FPS
        if self.stats.uptime > 0:
            self.stats.avg_fps = self.stats.frames_processed / self.stats.uptime

        return {
            "frames_processed": self.stats.frames_processed,
            "frames_dropped": self.stats.frames_dropped,
            "avg_processing_time_ms": self.stats.avg_processing_time * 1000,
            "avg_fps": self.stats.avg_fps,
            "detection_accuracy": self.stats.detection_accuracy.copy(),
            "uptime_seconds": self.stats.uptime,
            "last_error": self.stats.last_error,
            "is_running": self._is_running,
            "camera_connected": self.camera.is_connected() if self.camera else False,
        }

    def subscribe_to_events(
        self, event_type: str, callback: Callable[..., Any]
    ) -> bool:
        """Subscribe to vision events.

        Args:
            event_type: Type of event ('frame_processed', 'detection_complete', 'error_occurred')
            callback: Function to call when event occurs

        Returns:
            True if subscription successful
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
            return True
        return False

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        logger.info("VisionModule capture loop started")
        frame_count = 0
        log_interval = _get_config_value("vision.frame_logging.log_interval_frames", 30)
        sleep_interval_ms = _get_config_value(
            "vision.processing.capture_frame_interval_ms", 1
        )
        sleep_interval_sec = sleep_interval_ms / 1000.0

        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = 0

        while self._is_running:
            try:
                current_time = time.time()

                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(sleep_interval_sec)
                    continue

                # Get frame from camera (updated interface)
                logger.debug("VisionModule: Getting latest frame from camera")
                frame_data = self.camera.get_latest_frame()
                if frame_data is None:
                    logger.debug("VisionModule: No frame data available from camera")
                    self.stats.frames_dropped += 1
                    continue

                frame, frame_info = frame_data
                frame_count += 1

                if frame_count % log_interval == 0:
                    logger.debug(
                        f"VisionModule: Received frame #{frame_count}: shape={frame.shape}, timestamp={frame_info.timestamp}"
                    )

                # Apply ROI if enabled
                if self.config.roi_enabled and self._roi_corners:
                    logger.debug("Applying ROI to frame")
                    frame = self._apply_roi(frame)

                # Update current frame
                with self._lock:
                    self._current_frame = frame
                    logger.debug(
                        f"VisionModule: Updated current frame (frame #{frame_count})"
                    )

                # Add to processing queue (non-blocking)
                try:
                    self._frame_queue.put_nowait(
                        {
                            "frame": frame,
                            "timestamp": frame_info.timestamp,
                            "frame_number": frame_info.frame_number,
                        }
                    )
                    self._frame_number = frame_info.frame_number + 1
                    last_frame_time = current_time

                except asyncio.QueueFull:
                    logger.debug("Processing queue full, dropping frame")
                    self.stats.frames_dropped += 1

            except Exception as e:
                logger.error(f"Error in VisionModule capture loop: {e}", exc_info=True)
                self.stats.last_error = str(e)
                self._emit_event("error_occurred", {"error": str(e)})
                error_retry_delay = _get_config_value(
                    "vision.processing.capture_error_retry_delay_sec", 0.1
                )
                time.sleep(error_retry_delay)

        logger.info("VisionModule capture loop ended")

    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        logger.info("Starting processing loop")
        queue_timeout = _get_config_value(
            "vision.processing.processing_queue_timeout_sec", 0.1
        )

        while self._is_running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_data = self._frame_queue.get(timeout=queue_timeout)
                except asyncio.QueueEmpty:
                    continue

                # Process frame
                start_time = time.time()
                result = self._process_single_frame(
                    frame_data["frame"],
                    frame_data["frame_number"],
                    frame_data["timestamp"],
                )

                processing_time = time.time() - start_time

                # Update statistics
                self.stats.frames_processed += 1
                self.stats.avg_processing_time = (
                    self.stats.avg_processing_time * (self.stats.frames_processed - 1)
                    + processing_time
                ) / self.stats.frames_processed

                # Update current result
                with self._lock:
                    self._current_result = result

                # Emit events
                self._emit_event(
                    "frame_processed",
                    {
                        "frame_number": frame_data["frame_number"],
                        "processing_time": processing_time,
                    },
                )

                if result:
                    self._emit_event(
                        "detection_complete",
                        {"result": result, "frame_number": frame_data["frame_number"]},
                    )

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats.last_error = str(e)
                self._emit_event("error_occurred", {"error": str(e)})
                error_retry_delay = _get_config_value(
                    "vision.processing.processing_error_retry_delay_sec", 0.01
                )
                time.sleep(error_retry_delay)

        logger.info("Processing loop ended")

    def _process_single_frame(
        self,
        frame: NDArray[np.uint8],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[DetectionResult]:
        """Process a single frame through the complete pipeline.

        Args:
            frame: Input frame to process
            frame_number: Frame sequence number
            timestamp: Frame timestamp

        Returns:
            Detection results or None if processing failed
        """
        try:
            start_time = time.time()

            if frame_number is None:
                frame_number = self._frame_number
                self._frame_number += 1

            if timestamp is None:
                timestamp = time.time()

            # Preprocessing
            if self.config.preprocessing_enabled:
                processed_frame = self.preprocessor.process(frame)
            else:
                processed_frame = frame

            # Detection
            detected_table = None
            detected_balls = []
            detected_cue = None

            # Table detection
            if self.table_detector and self.config.enable_table_detection:
                try:
                    table_confidence_threshold = _get_config_value(
                        "vision.detection.table_detection_confidence_threshold", 0.5
                    )
                    table_result = self.table_detector.detect_complete_table(
                        processed_frame
                    )
                    if (
                        table_result
                        and table_result.confidence > table_confidence_threshold
                    ):
                        # Convert to our Table model format
                        detected_table = Table(
                            corners=table_result.corners.to_list(),
                            pockets=[
                                pocket.position for pocket in table_result.pockets
                            ],
                            width=table_result.width,
                            height=table_result.height,
                            surface_color=table_result.surface_color,
                        )

                        self.stats.detection_accuracy["table"] = table_result.confidence
                    else:
                        self.stats.detection_accuracy["table"] = 0.0

                except Exception as e:
                    logger.warning(f"Table detection failed: {e}")
                    self.stats.detection_accuracy["table"] = 0.0

            # Ball and Cue detection using unified detector with fallback
            if self.detector:
                # Ball detection
                if self.config.enable_ball_detection:
                    try:
                        detected_balls = self.detector.detect_balls(processed_frame)

                        # Update tracking if available
                        if self.tracker and self.config.enable_tracking:
                            detected_balls = self.tracker.update_tracking(
                                detected_balls, timestamp
                            )

                        max_balls = _get_config_value(
                            "vision.detection.max_balls_on_table", 16
                        )
                        detection_rate = len(detected_balls) / float(max_balls)
                        self.stats.detection_accuracy["balls"] = min(
                            detection_rate, 1.0
                        )

                    except NotImplementedError as e:
                        # YOLO not implemented yet, fall back to OpenCV
                        logger.warning(
                            f"Primary detector not implemented: {e}, falling back to OpenCV"
                        )
                        if self.fallback_to_opencv and self.ball_detector:
                            try:
                                detected_balls = self.ball_detector.detect_balls(
                                    processed_frame
                                )

                                # Update tracking if available
                                if self.tracker and self.config.enable_tracking:
                                    detected_balls = self.tracker.update_tracking(
                                        detected_balls, timestamp
                                    )

                                detection_rate = len(detected_balls) / 16.0
                                self.stats.detection_accuracy["balls"] = min(
                                    detection_rate, 1.0
                                )
                            except Exception as fallback_e:
                                logger.warning(
                                    f"Fallback ball detection also failed: {fallback_e}"
                                )
                                self.stats.detection_accuracy["balls"] = 0.0
                        else:
                            self.stats.detection_accuracy["balls"] = 0.0

                    except Exception as e:
                        logger.warning(f"Ball detection failed: {e}")
                        # Try fallback if available
                        if self.fallback_to_opencv and self.ball_detector:
                            try:
                                logger.info(
                                    "Attempting fallback to OpenCV ball detector"
                                )
                                detected_balls = self.ball_detector.detect_balls(
                                    processed_frame
                                )

                                # Update tracking if available
                                if self.tracker and self.config.enable_tracking:
                                    detected_balls = self.tracker.update_tracking(
                                        detected_balls, timestamp
                                    )

                                detection_rate = len(detected_balls) / 16.0
                                self.stats.detection_accuracy["balls"] = min(
                                    detection_rate, 1.0
                                )
                            except Exception as fallback_e:
                                logger.warning(
                                    f"Fallback ball detection also failed: {fallback_e}"
                                )
                                self.stats.detection_accuracy["balls"] = 0.0
                        else:
                            self.stats.detection_accuracy["balls"] = 0.0

                # Cue detection
                if self.config.enable_cue_detection:
                    try:
                        # Get cue ball position for improved cue detection
                        cue_ball_pos = None
                        for ball in detected_balls:
                            if ball.ball_type == BallType.CUE:
                                cue_ball_pos = ball.position
                                break

                        detected_cue = self.detector.detect_cue(
                            processed_frame, cue_ball_pos
                        )
                        self.stats.detection_accuracy["cue"] = (
                            1.0 if detected_cue else 0.0
                        )

                    except NotImplementedError as e:
                        # YOLO not implemented yet, fall back to OpenCV
                        logger.warning(
                            f"Primary cue detector not implemented: {e}, falling back to OpenCV"
                        )
                        if self.fallback_to_opencv and self.cue_detector:
                            try:
                                detected_cue = self.cue_detector.detect_cue(
                                    processed_frame, cue_ball_pos
                                )
                                self.stats.detection_accuracy["cue"] = (
                                    1.0 if detected_cue else 0.0
                                )
                            except Exception as fallback_e:
                                logger.warning(
                                    f"Fallback cue detection also failed: {fallback_e}"
                                )
                                self.stats.detection_accuracy["cue"] = 0.0
                        else:
                            self.stats.detection_accuracy["cue"] = 0.0

                    except Exception as e:
                        logger.warning(f"Cue detection failed: {e}")
                        # Try fallback if available
                        if self.fallback_to_opencv and self.cue_detector:
                            try:
                                logger.info(
                                    "Attempting fallback to OpenCV cue detector"
                                )
                                detected_cue = self.cue_detector.detect_cue(
                                    processed_frame, cue_ball_pos
                                )
                                self.stats.detection_accuracy["cue"] = (
                                    1.0 if detected_cue else 0.0
                                )
                            except Exception as fallback_e:
                                logger.warning(
                                    f"Fallback cue detection also failed: {fallback_e}"
                                )
                                self.stats.detection_accuracy["cue"] = 0.0
                        else:
                            self.stats.detection_accuracy["cue"] = 0.0
            else:
                # No detector available, use legacy detectors if they exist
                # Ball detection
                if self.ball_detector and self.config.enable_ball_detection:
                    try:
                        detected_balls = self.ball_detector.detect_balls(
                            processed_frame
                        )

                        # Update tracking if available
                        if self.tracker and self.config.enable_tracking:
                            detected_balls = self.tracker.update_tracking(
                                detected_balls, timestamp
                            )

                        max_balls = _get_config_value(
                            "vision.detection.max_balls_on_table", 16
                        )
                        detection_rate = len(detected_balls) / float(max_balls)
                        self.stats.detection_accuracy["balls"] = min(
                            detection_rate, 1.0
                        )

                    except Exception as e:
                        logger.warning(f"Ball detection failed: {e}")
                        self.stats.detection_accuracy["balls"] = 0.0

                # Cue detection
                if self.cue_detector and self.config.enable_cue_detection:
                    try:
                        detected_cue = self.cue_detector.detect_cue(processed_frame)
                        self.stats.detection_accuracy["cue"] = (
                            1.0 if detected_cue else 0.0
                        )

                    except Exception as e:
                        logger.warning(f"Cue detection failed: {e}")
                        self.stats.detection_accuracy["cue"] = 0.0

            # Create result
            processing_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds

            if detected_table is None:
                # Create a default table if detection failed
                h, w = frame.shape[:2]
                default_surface_color = _get_config_value(
                    "vision.defaults.table_surface_color_rgb", [60, 200, 100]
                )
                detected_table = Table(
                    corners=[(0, 0), (w, 0), (w, h), (0, h)],
                    pockets=[],
                    width=w,
                    height=h,
                    surface_color=tuple(default_surface_color),
                )

            # Create frame statistics
            default_quality = _get_config_value(
                "vision.defaults.default_frame_quality", 0.5
            )
            detected_quality = _get_config_value(
                "vision.defaults.detected_frame_quality", 1.0
            )

            statistics = FrameStatistics(
                frame_number=frame_number,
                timestamp=timestamp,
                processing_time=processing_time,
                balls_detected=len(detected_balls),
                balls_tracked=len(
                    [b for b in detected_balls if b.track_id is not None]
                ),
                cue_detected=detected_cue is not None,
                table_detected=detected_table is not None,
                detection_confidence=sum(self.stats.detection_accuracy.values())
                / max(len(self.stats.detection_accuracy), 1),
                frame_quality=(
                    detected_quality if detected_table is not None else default_quality
                ),
            )

            result = DetectionResult(
                frame_number=frame_number,
                timestamp=timestamp,
                balls=detected_balls,
                cue=detected_cue,
                table=detected_table,
                statistics=statistics,
            )

            return result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.stats.last_error = str(e)
            return None

    def _apply_roi(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply region of interest cropping to frame."""
        expected_corner_count = _get_config_value("vision.defaults.roi_corner_count", 4)
        if not self._roi_corners or len(self._roi_corners) != expected_corner_count:
            return frame

        try:
            # Create mask for ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi_points = np.array(self._roi_corners, dtype=np.int32)
            cv2.fillPoly(mask, [roi_points], 255)

            # Apply mask
            result = cv2.bitwise_and(frame, frame, mask=mask)
            return result

        except Exception as e:
            logger.warning(f"Failed to apply ROI: {e}")
            return frame

    def _load_background_frame(self, image_path: str) -> None:
        """Load background frame from file and set it for detectors.

        Args:
            image_path: Path to background image file
        """
        try:
            import os

            if not os.path.exists(image_path):
                logger.warning(f"Background image not found: {image_path}")
                return

            # Load image
            background_frame = cv2.imread(image_path)
            if background_frame is None:
                logger.error(f"Failed to load background image: {image_path}")
                return

            logger.info(f"Loaded background frame from: {image_path}")

            # Set background for unified detector
            if self.detector:
                self.detector.set_background_frame(background_frame)
                logger.info("Background frame set for unified detector")

            # Set background for fallback detectors if they exist
            if self.ball_detector and self.ball_detector != getattr(
                getattr(self.detector, "ball_detector", None), None, None
            ):
                self.ball_detector.set_background_frame(background_frame)
                logger.info("Background frame set for fallback ball detector")

            if self.cue_detector and self.cue_detector != getattr(
                getattr(self.detector, "cue_detector", None), None, None
            ):
                self.cue_detector.set_background_frame(background_frame)
                logger.info("Background frame set for fallback cue detector")

        except Exception as e:
            logger.error(f"Failed to load background frame: {e}")
            self.stats.last_error = str(e)

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit event to registered callbacks."""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.warning(f"Error in event callback for {event_type}: {e}")

    def __del__(self) -> None:
        """Cleanup on destruction."""
        with contextlib.suppress(Exception):
            self.stop_capture()


# Export main classes
__all__ = [
    # Core vision module
    "VisionModule",
    "VisionConfig",
    "VisionStatistics",
    "VisionModuleError",
    "SimpleCameraModule",
    "DirectCameraModule",
    # Data models
    "Ball",
    "BallType",
    "CameraFrame",
    "CueStick",
    "CueState",
    "Table",
    "Pocket",
    "PocketType",
    "DetectionResult",
    "FrameStatistics",
    # Camera capture
    "CameraCapture",
    "CameraStatus",
    "CameraHealth",
    "FrameInfo",
    # Configuration
    "VisionConfigurationManager",
    "create_vision_config_manager",
    # Calibration
    "CameraCalibration",
    "ColorCalibration",
    "GeometricCalibration",
    "CalibrationData",
    # Session tracking
    "DetectionSession",
    "ShotEvent",
]
