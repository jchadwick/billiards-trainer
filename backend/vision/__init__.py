"""Vision Module - Computer vision processing for billiards analysis.

This module provides comprehensive computer vision capabilities for detecting and tracking
pool table elements including table boundaries, balls, and cue sticks in real-time.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import cv2
import numpy as np

from .calibration.color import ColorCalibrator
from .calibration.geometry import GeometricCalibrator
from .coordinates import (
    playing_area_to_frame,
    point_is_within_scaled_radius,
    point_to_frame,
    radius_to_frame_ellipse,
)
from .models import (
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
from .motion_monitor import MotionAssessment, MotionMonitor
from .phase_scheduler import (
    BallMapObservation,
    BallPosition,
    CueObservation,
    PhaseDecision,
    VisionPhaseScheduler,
)

# Shared memory video consumer (for IPC with Video Module)
from .stream.video_consumer import VideoConsumer, VideoModuleNotAvailableError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .detection.yolo_detector import YOLODetector
    from .tracking.tracker import ObjectTracker

# Ball is now in backend.common.models - import from there instead
# Not imported here to avoid circular dependency


# Set up logging
logger = logging.getLogger(__name__)

_BALL_DETECTION_SUCCESS_STATUSES = frozenset({"detections", "no_detections"})
_BALL_DETECTION_FAILURE_STAGES = frozenset(
    {"inference", "decoding", "classification", "unknown"}
)


def _get_config_value(key_path: str, default: Any) -> Any:
    """Get configuration value from the config system.

    Args:
        key_path: Dot-separated path to config value (e.g., "vision.processing.target_fps")
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    try:
        from ..config import config

        return config.get(key_path, default)
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
    max_frame_queue_size: int

    # Detection settings (from vision.detection.*)
    enable_ball_detection: bool
    enable_tracking: bool

    # Performance settings (from vision.processing.*)
    frame_skip: int
    roi_enabled: bool

    # Background subtraction (from vision.detection.background_subtraction.*)
    background_image_path: str | None
    use_background_subtraction: bool
    background_threshold: int

    # Debug settings (from vision.debug*)
    debug_mode: bool
    save_debug_images: bool
    debug_output_path: str

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> VisionConfig:
        """Create VisionConfig from configuration dictionary.

        Args:
            config_dict: Configuration dictionary (can be empty, will use defaults)

        Returns:
            VisionConfig instance with values from config or defaults
        """
        # Import here to avoid circular dependencies
        from ..config import config

        # Use configuration singleton
        config_mgr = config

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
            max_frame_queue_size=config_dict.get(
                "max_frame_queue_size",
                config_mgr.get("vision.processing.max_frame_queue_size", 5),
            ),
            # Detection settings
            enable_ball_detection=config_dict.get(
                "enable_ball_detection",
                config_mgr.get("vision.detection.enable_ball_detection", True),
            ),
            enable_tracking=config_dict.get(
                "enable_tracking",
                config_mgr.get("vision.processing.enable_tracking", True),
            ),
            # Performance settings
            frame_skip=config_dict.get(
                "frame_skip", config_mgr.get("vision.processing.frame_skip", 0)
            ),
            roi_enabled=config_dict.get(
                "roi_enabled", config_mgr.get("vision.processing.roi_enabled", False)
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
    detection_accuracy: dict[str, float] = field(
        default_factory=lambda: {"table": 0.0, "balls": 0.0, "cue": 0.0}
    )
    last_error: str | None = None
    uptime: float = 0.0


class VisionModuleError(Exception):
    """Base exception for Vision Module errors."""

    pass


class VisionModule:
    """Main Vision Module - Computer vision processing interface.

    Provides real-time computer vision processing for billiards table analysis
    including table detection, ball tracking, and cue stick detection.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the Vision Module.

        Args:
            config: Configuration dictionary for the vision module
        """
        # Parse configuration - use from_config_dict to load defaults from config system
        if config is None:
            config = {}
        module_config = config
        # Phase transitions and tracking accept an injectable monotonic clock
        # for deterministic replay. Wall-clock publication timestamps remain
        # separate source metadata and performance timers use the real clock.
        self._clock = clock or time.monotonic
        self.config = VisionConfig.from_config_dict(config)

        # Initialize statistics
        self.stats = VisionStatistics()
        self._start_time = time.monotonic()

        # Video source - shared memory IPC with Video Module
        self._video_consumer = VideoConsumer()

        # Component references have explicit nullable contracts because the
        # corresponding configuration switches can disable detection/tracking.
        self.detector: YOLODetector | None = None
        self.tracker: ObjectTracker | None = None

        # Initialize components
        self._initialize_components()

        # Threading and state management
        self._capture_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._is_running = False
        # Live localization keeps exactly one pending frame. If inference is
        # busy, capture replaces that slot so the next operation starts from
        # the newest available image instead of replaying historical frames.
        self._frame_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[Any] = queue.Queue(
            maxsize=self.config.max_frame_queue_size
        )
        self._lock = threading.Lock()

        # Current state
        self._current_frame: NDArray[np.uint8] | None = None
        self._current_result: DetectionResult | None = None
        self._frame_number = 0

        # The phase-aware pipeline remains opt-in until camera recordings have
        # validated its cadence profiles. Existing users therefore retain the
        # continuous full-resolution path by default.
        phase_settings = module_config.get(
            "phase_scheduler",
            module_config.get(
                "vision.processing.phase_scheduler",
                _get_config_value("vision.processing.phase_scheduler", {}),
            ),
        )
        motion_settings = module_config.get(
            "motion_monitor",
            module_config.get(
                "vision.processing.motion_monitor",
                _get_config_value("vision.processing.motion_monitor", {}),
            ),
        )
        self.phase_scheduler = VisionPhaseScheduler(phase_settings)
        self.motion_monitor = MotionMonitor(motion_settings)
        self._aiming_request_override: bool | None = None
        self._phase_geometry_signature: tuple[Any, ...] | None = None
        self._phase_calibration_revision: str | None = None
        self._phase_last_motion_assessment: MotionAssessment | None = None
        self._phase_last_motion_timestamp: float | None = None
        self._phase_last_ball_timestamp: float | None = None
        self._phase_last_ball_source_timestamp: float | None = None
        self._phase_last_ball_source_frame_number: int | None = None
        self._phase_last_ball_source_timestamp_kind: str | None = None
        self._phase_last_source_frame_number: int | None = None
        self._phase_last_source_process_id: int | None = None
        self._phase_last_source_timestamp: float | None = None
        self._phase_last_cue_timestamp: float | None = None
        self._phase_last_cue: CueStick | None = None
        self._phase_last_ball_map: list[Any] = []

        # Event callbacks
        self._callbacks: dict[str, list[Callable[..., Any]]] = {
            "frame_processed": [],
            "detection_complete": [],
            "error_occurred": [],
        }

        # Region of interest
        self._roi_corners: list[tuple[int, int]] | None = None

        logger.info("Vision Module initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all vision processing components."""
        try:
            logger.info("Initializing vision components")

            # Use VideoConsumer to read frames from Video Module via shared memory
            logger.info("Using shared memory IPC (VideoConsumer)")

            # VideoConsumer was constructed before component setup; attach is
            # deferred until start_capture.
            logger.info("VideoConsumer initialized (will attach on start_capture)")

            # YOLO detection with optional experimental OpenCV classification.
            if self.config.enable_ball_detection:
                # Load YOLO configuration parameters - use shared config manager
                from ..config import config as config_mgr

                # Get YOLO parameters from config
                yolo_model_path = config_mgr.get("vision.detection.yolo_model_path")
                yolo_confidence = config_mgr.get(
                    "vision.detection.yolo_confidence", 0.15
                )
                yolo_nms_threshold = config_mgr.get(
                    "vision.detection.yolo_nms_threshold", 0.45
                )
                yolo_device = config_mgr.get("vision.detection.yolo_device", "cpu")
                min_ball_size = config_mgr.get("vision.detection.min_ball_radius", 20)

                enable_opencv_classification = bool(
                    config_mgr.get(
                        "vision.detection.enable_opencv_classification", False
                    )
                )

                logger.info(
                    f"Initializing YOLO detector: model={yolo_model_path}, "
                    f"confidence={yolo_confidence}, device={yolo_device}, "
                    f"min_ball_size={min_ball_size}, "
                    f"opencv_classification={enable_opencv_classification}"
                )

                # Import YOLO detector
                from .detection.yolo_detector import YOLODetector

                try:
                    # Exact ball numbers are not supported by the configured
                    # binary YOLO model. Keep the color/stripe heuristic opt-in
                    # until it is validated against representative table footage.
                    self.detector = YOLODetector(
                        model_path=yolo_model_path,
                        device=yolo_device,
                        confidence=yolo_confidence,
                        nms_threshold=yolo_nms_threshold,
                        auto_fallback=False,  # Fail loudly if YOLO not available
                        enable_opencv_classification=enable_opencv_classification,
                        min_ball_size=min_ball_size,  # Filter out markers and noise
                        camera_resolution=self.config.camera_resolution,
                    )
                    logger.info("YOLO detector initialized successfully")

                except Exception as e:
                    logger.error(
                        f"Failed to initialize YOLO detector: {e}. "
                        "YOLO detection is required - cannot fall back to OpenCV-only mode."
                    )
                    raise VisionModuleError(
                        f"YOLO detector initialization failed: {e}. "
                        "Ensure YOLO model is available at the configured path."
                    )
            else:
                self.detector = None

            # Load background frame if configured
            if self.config.background_image_path:
                self._load_background_frame(self.config.background_image_path)

            # Tracking
            if self.config.enable_tracking:
                # Import here to avoid circular dependencies
                from ..config import config
                from .tracking.tracker import ObjectTracker

                tracking_config = config.get("vision.tracking", {})
                self.tracker = ObjectTracker(tracking_config, clock=self._clock)
            else:
                self.tracker = None

            # Camera calibration is owned by the live calibration manager and
            # camera service, which share the active camera lifecycle.
            self.color_calibrator = ColorCalibrator()
            self.geometry_calibrator = GeometricCalibrator()

            logger.info("All vision components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vision components: {e}")
            raise VisionModuleError(f"Component initialization failed: {e}")

    def start_capture(self) -> bool:
        """Start video capture and processing via shared memory IPC.

        Returns:
            True if capture started successfully
        """
        try:
            logger.info("VisionModule.start_capture called")

            if self._is_running:
                logger.warning("Capture is already running")
                return True

            # Attach to Video Module via shared memory
            logger.info("Starting VideoConsumer (shared memory IPC)...")
            try:
                self._video_consumer.start()
                logger.info("VideoConsumer attached to shared memory successfully")
            except VideoModuleNotAvailableError as e:
                logger.error(f"Failed to attach to Video Module: {e}")
                raise VisionModuleError(
                    f"Failed to attach to Video Module via shared memory: {e}\n"
                    "The legacy Python capture publisher is retired; use the "
                    "Rust server with a configured live source."
                ) from e

            logger.info("Video source started, initializing vision module state")

            # Reset statistics
            self.stats = VisionStatistics()
            self._start_time = time.monotonic()
            self._frame_number = 0
            # A newly attached stream starts a fresh source epoch. Reuse the
            # same reset contract used when an in-process camera restart is
            # detected so old frame IDs, calibration, and tracker state cannot
            # make the first current observations look stale or duplicated.
            self._reset_after_source_restart()

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
        """Stop video capture and processing."""
        try:
            if not self._is_running:
                logger.warning("Capture is not running")
                return

            # Signal shutdown
            self._is_running = False

            # Stop VideoConsumer
            logger.info("Stopping VideoConsumer...")
            self._video_consumer.stop()

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
                except queue.Empty:
                    break

            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break

            logger.info("Vision capture stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping capture: {e}")
            self.stats.last_error = str(e)

    def close(self) -> None:
        """Stop capture and release vision resources during shutdown."""
        self.stop_capture()

    def process_frame(self) -> DetectionResult | None:
        """Get latest detection results from processing thread.

        Returns:
            Detection results or None if no frame available
        """
        try:
            # Return latest result from processing thread
            with self._lock:
                return self._current_result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.stats.last_error = str(e)
            self._emit_event("error_occurred", {"error": str(e)})
            return None

    def get_current_frame(self) -> NDArray[np.uint8] | None:
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

    def calibrate_colors(
        self, sample_image: NDArray[np.uint8] | None = None
    ) -> dict[str, Any]:
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
            calibration_results = self.color_calibrator.create_default_profile(
                sample_image
            ).to_dict()

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

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary containing performance and accuracy statistics
        """
        # Update uptime
        self.stats.uptime = time.monotonic() - self._start_time

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
            "video_consumer_running": (self._video_consumer.is_running),
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
        last_frame_time = 0.0

        while self._is_running:
            try:
                current_time = time.monotonic()

                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(sleep_interval_sec)
                    continue

                # Get frame from VideoConsumer (shared memory IPC)
                logger.debug("VisionModule: Getting frame from VideoConsumer")
                frame, source_metadata = self._video_consumer.get_frame_with_metadata()

                if frame is None:
                    # No new frame available, sleep briefly and continue
                    time.sleep(sleep_interval_sec)
                    continue

                # Preserve source identity and publication time. Local monotonic
                # times are reserved for elapsed intervals and queue age.
                frame_count += 1
                published_at = (
                    source_metadata.timestamp
                    if source_metadata is not None
                    else time.time()
                )
                source_frame_number = (
                    source_metadata.frame_number
                    if source_metadata is not None
                    else None
                )
                frame_number = (
                    source_frame_number
                    if source_frame_number is not None
                    else frame_count
                )

                if frame_count % log_interval == 0:
                    logger.debug(
                        f"VisionModule: Received frame #{frame_count}: shape={frame.shape}, "
                        f"publication_timestamp={published_at:.3f}s"
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

                # Rate limiting follows consumed source frames, independent of
                # whether a frame is accepted into the processing queue.
                last_frame_time = current_time

                # Add to processing queue. Latest-frame mode evicts stale work.
                try:
                    frame_data = {
                        "frame": frame,
                        "timestamp": published_at,
                        "frame_number": frame_number,
                        "source_frame_number": source_frame_number,
                        "source_timestamp": (
                            published_at if source_metadata is not None else None
                        ),
                        "source_timestamp_kind": (
                            "shared_memory_publication"
                            if source_metadata is not None
                            else "vision_capture_fallback"
                        ),
                        "source_process_id": (
                            source_metadata.writer_pid
                            if source_metadata is not None
                            else None
                        ),
                        "source_publication_monotonic_ns": (
                            source_metadata.publication_monotonic_ns
                            if source_metadata is not None
                            else None
                        ),
                        "enqueued_monotonic": time.monotonic(),
                    }
                    self._enqueue_frame(frame_data)
                except queue.Full:
                    logger.debug("Processing queue full, dropping incoming frame")
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

    def _enqueue_frame(self, frame_data: dict[str, Any]) -> None:
        """Retain only the newest pending frame without blocking capture."""
        while True:
            try:
                self._frame_queue.put_nowait(frame_data)
                return
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    # The processing thread won the race for the queued item;
                    # retry insertion without losing the incoming frame.
                    continue
                self.stats.frames_dropped += 1

    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        logger.info("Starting processing loop")
        queue_timeout = _get_config_value(
            "vision.processing.processing_queue_timeout_sec", 0.1
        )

        while self._is_running:
            try:
                # Get frame (blocking with timeout), then discard any backlog
                # accumulated while inference was running.
                try:
                    frame_data = self._frame_queue.get(timeout=queue_timeout)
                except queue.Empty:
                    continue

                while True:
                    try:
                        newer_frame_data = self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
                    frame_data = newer_frame_data
                    self.stats.frames_dropped += 1

                # Process frame
                queue_delay_seconds = max(
                    0.0,
                    time.monotonic()
                    - frame_data.get("enqueued_monotonic", time.monotonic()),
                )
                start_time = time.monotonic()
                result = self._process_single_frame(
                    frame_data["frame"],
                    frame_data["frame_number"],
                    frame_data["timestamp"],
                    source_frame_number=frame_data.get("source_frame_number"),
                    source_timestamp=frame_data.get("source_timestamp"),
                    source_timestamp_kind=frame_data.get("source_timestamp_kind"),
                    source_process_id=frame_data.get("source_process_id"),
                    source_publication_monotonic_ns=frame_data.get(
                        "source_publication_monotonic_ns"
                    ),
                )

                processing_time = time.monotonic() - start_time

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
                        "source_frame_number": frame_data.get("source_frame_number"),
                        "source_timestamp": frame_data.get("source_timestamp"),
                        "source_timestamp_kind": frame_data.get(
                            "source_timestamp_kind"
                        ),
                        "queue_delay_seconds": queue_delay_seconds,
                        "publication_age_seconds": (
                            max(0.0, time.time() - frame_data["source_timestamp"])
                            if frame_data.get("source_timestamp") is not None
                            else None
                        ),
                        "processing_time": processing_time,
                    },
                )

                if result:
                    self._emit_event(
                        "detection_complete",
                        {
                            "result": result,
                            "frame_number": frame_data["frame_number"],
                            "source_frame_number": frame_data.get(
                                "source_frame_number"
                            ),
                            "source_timestamp": frame_data.get("source_timestamp"),
                            "source_timestamp_kind": frame_data.get(
                                "source_timestamp_kind"
                            ),
                        },
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

    def set_aiming_requested(self, requested: bool | None) -> None:
        """Set an explicit aiming request, or restore cue-based inference.

        Passing ``None`` leaves aiming intent to fresh cue evidence when the
        phase scheduler's ``infer_aiming_from_cue`` option is enabled. An
        explicit ``False`` disables that inference until changed or reset.
        """
        self._aiming_request_override = None if requested is None else bool(requested)

    @staticmethod
    def _phase_table_geometry(
        frame_shape: tuple[int, ...],
    ) -> tuple[list[tuple[float, float]] | None, str]:
        """Return configured corners scaled into current source-frame pixels."""
        height, width = frame_shape[:2]
        try:
            from ..config import config as config_manager

            raw_corners = config_manager.get("table.playing_area_corners", [])
        except Exception:
            raw_corners = []

        polygon: list[tuple[float, float]] | None = None
        try:
            if len(raw_corners) == 4:
                polygon = playing_area_to_frame(raw_corners, (width, height))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            polygon = None

        # Include each point's scale metadata so scale-only edits invalidate
        # phase stability and prediction based on the previous map.
        revision = f"{width}x{height}:{raw_corners!r}"
        return polygon, revision

    def _resolve_aiming_request(self) -> bool | None:
        """Return explicit intent or leave cue-based inference to the scheduler."""
        override = getattr(self, "_aiming_request_override", None)
        return None if override is None else bool(override)

    def _monotonic_now(self) -> float:
        """Read the injected processing clock, falling back for test adapters."""
        return getattr(self, "_clock", time.monotonic)()

    def _reset_after_source_restart(self) -> None:
        """Forget temporal state tied to the previous shared-memory stream."""
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        self.motion_monitor.reset()
        self.phase_scheduler = VisionPhaseScheduler(self.phase_scheduler.settings)
        self._phase_geometry_signature = None
        self._phase_calibration_revision = None
        self._phase_last_ball_map = []
        self._phase_last_ball_timestamp = None
        self._phase_last_ball_source_timestamp = None
        self._phase_last_ball_source_frame_number = None
        self._phase_last_ball_source_timestamp_kind = None
        self._phase_last_motion_assessment = None
        self._phase_last_motion_timestamp = None
        self._phase_last_cue = None
        self._phase_last_cue_timestamp = None
        self._phase_last_source_frame_number = None
        self._phase_last_source_process_id = None
        self._phase_last_source_timestamp = None

    @staticmethod
    def _resize_for_phase(
        frame: NDArray[np.uint8], max_dimension: int | None
    ) -> NDArray[np.uint8]:
        """Downscale for scheduled analysis without upscaling or changing display."""
        if max_dimension is None or max_dimension <= 0:
            return frame
        height, width = frame.shape[:2]
        largest = max(width, height)
        if largest <= max_dimension:
            return frame
        scale = max_dimension / float(largest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return cast(
            "NDArray[np.uint8]",
            cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA),
        )

    @staticmethod
    def _crop_to_calibrated_table(
        frame: NDArray[np.uint8],
        table_polygon: list[tuple[float, float]] | None,
        source_resolution: tuple[int, int],
        margin_fraction: float,
    ) -> tuple[NDArray[np.uint8], tuple[int, int, int, int] | None]:
        """Crop an analysis view to calibrated table bounds in source pixels."""
        if table_polygon is None or len(table_polygon) != 4:
            return frame, None
        frame_height, frame_width = frame.shape[:2]
        source_width, source_height = source_resolution
        scale_x = frame_width / float(source_width)
        scale_y = frame_height / float(source_height)
        points = [(float(x) * scale_x, float(y) * scale_y) for x, y in table_polygon]
        left, right = min(x for x, _ in points), max(x for x, _ in points)
        top, bottom = min(y for _, y in points), max(y for _, y in points)
        margin_x = max(0.0, float(margin_fraction)) * (right - left)
        margin_y = max(0.0, float(margin_fraction)) * (bottom - top)
        x0 = max(0, int(np.floor(left - margin_x)))
        y0 = max(0, int(np.floor(top - margin_y)))
        x1 = min(frame_width, int(np.ceil(right + margin_x)))
        y1 = min(frame_height, int(np.ceil(bottom + margin_y)))
        if x1 <= x0 or y1 <= y0:
            return frame, None
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    @staticmethod
    def _normalise_cropped_balls_to_source(
        balls: list[Any],
        source_resolution: tuple[int, int],
        processing_resolution: tuple[int, int],
        crop_rect: tuple[int, int, int, int],
    ) -> list[Any]:
        """Restore crop offsets, resize factors, and canonical coordinates."""
        from ..core.resolution_converter import ResolutionConverter
        from .detection.detector_adapter import canonical_point_to_source_pixels

        crop_x, crop_y, crop_right, crop_bottom = crop_rect
        crop_resolution = (crop_right - crop_x, crop_bottom - crop_y)
        source_width, source_height = source_resolution
        processing_width, processing_height = processing_resolution
        if min(*crop_resolution, *source_resolution, *processing_resolution) <= 0:
            raise ValueError("Frame and crop resolutions must be positive")
        scale_x = source_width / float(processing_width)
        scale_y = source_height / float(processing_height)

        def point_to_source(
            point: tuple[float, float], coordinate_space: str
        ) -> tuple[float, float]:
            if coordinate_space == "4k_canonical":
                crop_point = canonical_point_to_source_pixels(point, crop_resolution)
            else:
                crop_point = (float(point[0]), float(point[1]))
            return (
                (crop_point[0] + crop_x) * scale_x,
                (crop_point[1] + crop_y) * scale_y,
            )

        def vector_to_source(
            vector: tuple[float, float], coordinate_space: str
        ) -> tuple[float, float]:
            if coordinate_space == "4k_canonical":
                crop_vector = ResolutionConverter.scale_from_4k(
                    float(vector[0]), float(vector[1]), crop_resolution
                )
            else:
                crop_vector = (float(vector[0]), float(vector[1]))
            return (crop_vector[0] * scale_x, crop_vector[1] * scale_y)

        normalized: list[Any] = []
        for ball in balls:
            coordinate_space = getattr(ball, "coordinate_space", "pixel")
            if coordinate_space not in {"4k_canonical", "pixel"}:
                normalized.append(ball)
                continue

            ball.position = point_to_source(ball.position, coordinate_space)
            radius_in_crop_pixels = (
                ResolutionConverter.scale_distance_from_4k(
                    float(ball.radius), crop_resolution
                )
                if coordinate_space == "4k_canonical"
                else float(ball.radius)
            )
            source_radius = radius_in_crop_pixels * scale_x

            # Motion vectors and history are part of the same coordinate
            # contract as the current position. Apply the crop/resize mapping
            # to them too, while only points receive the crop origin offset.
            for attribute in ("velocity", "acceleration"):
                vector = getattr(ball, attribute, None)
                if vector is not None:
                    setattr(ball, attribute, vector_to_source(vector, coordinate_space))
            predicted_position = getattr(ball, "predicted_position", None)
            if predicted_position is not None:
                ball.predicted_position = point_to_source(
                    predicted_position, coordinate_space
                )
            history = getattr(ball, "position_history", None)
            if history is not None:
                ball.position_history = [
                    point_to_source(point, coordinate_space) for point in history
                ]

            if coordinate_space == "4k_canonical":
                ball.position = ResolutionConverter.scale_to_4k(
                    ball.position[0], ball.position[1], source_resolution
                )
                ball.radius = ResolutionConverter.scale_distance_to_4k(
                    source_radius, source_resolution
                )
                for attribute in ("velocity", "acceleration"):
                    vector = getattr(ball, attribute, None)
                    if vector is not None:
                        setattr(
                            ball,
                            attribute,
                            ResolutionConverter.scale_to_4k(
                                vector[0], vector[1], source_resolution
                            ),
                        )
                if ball.predicted_position is not None:
                    ball.predicted_position = ResolutionConverter.scale_to_4k(
                        ball.predicted_position[0],
                        ball.predicted_position[1],
                        source_resolution,
                    )
                if ball.position_history is not None:
                    ball.position_history = [
                        ResolutionConverter.scale_to_4k(
                            point[0], point[1], source_resolution
                        )
                        for point in ball.position_history
                    ]
            else:
                ball.radius = source_radius
            ball.source_resolution = source_resolution
            normalized.append(ball)
        return normalized

    @staticmethod
    def _normalise_balls_to_source(
        balls: list[Any],
        source_resolution: tuple[int, int],
        processing_resolution: tuple[int, int],
    ) -> list[Any]:
        """Restore detector outputs to source-frame coordinates and metadata."""
        from ..core.resolution_converter import ResolutionConverter
        from .detection.detector_adapter import canonical_point_to_source_pixels

        source_width, source_height = source_resolution
        if min(*source_resolution, *processing_resolution) <= 0:
            raise ValueError("Frame resolutions must be positive")
        default_resolution = processing_resolution
        normalized: list[Any] = []
        for ball in balls:
            # Vision detector outputs are image pixels unless the producer
            # explicitly marks them canonical. Never infer canonical space
            # from a missing metadata field.
            coordinate_space = getattr(ball, "coordinate_space", "pixel")
            described_resolution = getattr(ball, "source_resolution", None)
            if not described_resolution or min(described_resolution) <= 0:
                described_resolution = default_resolution
            described_width, described_height = described_resolution
            if coordinate_space == "4k_canonical":
                x, y = canonical_point_to_source_pixels(
                    ball.position, (described_width, described_height)
                )
                radius = ResolutionConverter.scale_distance_from_4k(
                    ball.radius, (described_width, described_height)
                )
            elif coordinate_space == "pixel":
                x, y = float(ball.position[0]), float(ball.position[1])
                radius = float(ball.radius)
            else:
                normalized.append(ball)
                continue

            scale_x = source_width / float(described_width)
            scale_y = source_height / float(described_height)
            source_x = float(x) * scale_x
            source_y = float(y) * scale_y
            source_radius = float(radius) * scale_x
            if coordinate_space == "4k_canonical":
                ball.position = ResolutionConverter.scale_to_4k(
                    source_x, source_y, source_resolution
                )
                ball.radius = ResolutionConverter.scale_distance_to_4k(
                    source_radius, source_resolution
                )
            else:
                ball.position = (source_x, source_y)
                ball.radius = source_radius
                for attribute in ("velocity", "acceleration"):
                    vector = getattr(ball, attribute, None)
                    if vector is not None:
                        setattr(
                            ball,
                            attribute,
                            (float(vector[0]) * scale_x, float(vector[1]) * scale_y),
                        )
                predicted_position = getattr(ball, "predicted_position", None)
                if predicted_position is not None:
                    ball.predicted_position = (
                        float(predicted_position[0]) * scale_x,
                        float(predicted_position[1]) * scale_y,
                    )
                history = getattr(ball, "position_history", None)
                if history is not None:
                    ball.position_history = [
                        (float(point[0]) * scale_x, float(point[1]) * scale_y)
                        for point in history
                    ]
            ball.source_resolution = source_resolution
            normalized.append(ball)
        return normalized

    @staticmethod
    def _normalise_cue_to_source(
        cue: CueStick | None,
        source_resolution: tuple[int, int],
        processing_resolution: tuple[int, int],
    ) -> CueStick | None:
        """Restore cue source metadata after a phase-specific resize."""
        if cue is None or cue.coordinate_space == "world":
            return cue
        from ..core.resolution_converter import ResolutionConverter

        if min(*source_resolution, *processing_resolution) <= 0:
            raise ValueError("Frame resolutions must be positive")
        described_resolution = cue.source_resolution or processing_resolution
        described_width, described_height = described_resolution
        source_width, source_height = source_resolution
        if min(described_resolution) <= 0:
            raise ValueError("Cue source resolution must be positive")
        scale_x = source_width / float(described_width)
        scale_y = source_height / float(described_height)

        if cue.coordinate_space == "pixel":

            def point_to_source(
                point: tuple[float, float] | None,
            ) -> tuple[float, float] | None:
                if point is None:
                    return None
                return (float(point[0]) * scale_x, float(point[1]) * scale_y)

            def vector_to_source(
                vector: tuple[float, float],
            ) -> tuple[float, float]:
                return (float(vector[0]) * scale_x, float(vector[1]) * scale_y)

            cue.tip_position = point_to_source(cue.tip_position)  # type: ignore[assignment]
            cue.butt_position = point_to_source(cue.butt_position)  # type: ignore[assignment]
            cue.tip_velocity = vector_to_source(cue.tip_velocity)
            cue.shaft_points = [
                point
                for point in (point_to_source(point) for point in cue.shaft_points)
                if point is not None
            ]
            cue.predicted_contact_point = point_to_source(cue.predicted_contact_point)
            cue.aiming_line = (
                [
                    point
                    for point in (point_to_source(point) for point in cue.aiming_line)
                    if point is not None
                ]
                if cue.aiming_line is not None
                else None
            )
            cue.length *= scale_x
            cue.width *= scale_x
        elif cue.coordinate_space != "4k_canonical":
            return cue
        cue.source_resolution = source_resolution
        return cue

    @staticmethod
    def _normalise_table_to_source(
        table: Table,
        source_resolution: tuple[int, int],
        processing_resolution: tuple[int, int],
    ) -> Table:
        """Restore table boundaries and extents to the captured frame space.

        Pockets inherit their coordinate space from the containing table. This
        keeps one source-resolution declaration authoritative for corners,
        rails, playing-area boundaries, pocket geometry, and table extents.
        """
        if table.coordinate_space == "world":
            return table
        if min(*source_resolution, *processing_resolution) <= 0:
            raise ValueError("Frame resolutions must be positive")

        described_resolution = table.source_resolution or processing_resolution
        if min(described_resolution) <= 0:
            raise ValueError("Table source resolution must be positive")
        if table.coordinate_space == "pixel":
            scale_x = source_resolution[0] / float(described_resolution[0])
            scale_y = source_resolution[1] / float(described_resolution[1])

            def scale_point(point: tuple[float, float]) -> tuple[float, float]:
                return (float(point[0]) * scale_x, float(point[1]) * scale_y)

            table.corners = [scale_point(point) for point in table.corners]
            table.rails = [
                [scale_point(point) for point in rail] for rail in table.rails
            ]
            table.playing_area = [scale_point(point) for point in table.playing_area]
            table.width *= scale_x
            table.height *= scale_y
            for pocket in table.pockets:
                pocket.position = scale_point(pocket.position)
                pocket.corners = [scale_point(point) for point in pocket.corners]
                pocket.radius *= scale_x
        elif table.coordinate_space != "4k_canonical":
            return table

        table.source_resolution = source_resolution
        return table

    @staticmethod
    def _ball_positions_in_source(
        balls: list[Any], source_resolution: tuple[int, int]
    ) -> tuple[BallPosition, ...]:
        """Build scheduler observations in native source pixels, never 4K space."""
        from .detection.detector_adapter import canonical_point_to_source_pixels

        positions: list[BallPosition] = []
        for ball in balls:
            coordinate_space = getattr(ball, "coordinate_space", "pixel")
            if coordinate_space == "4k_canonical":
                point = canonical_point_to_source_pixels(
                    ball.position,
                    getattr(ball, "source_resolution", None) or source_resolution,
                )
            elif coordinate_space == "pixel":
                point = ball.position
            else:
                continue
            positions.append(
                BallPosition(
                    x=float(point[0]),
                    y=float(point[1]),
                    confidence=float(getattr(ball, "confidence", 1.0)),
                )
            )
        return tuple(positions)

    def _assess_phase_motion(
        self,
        frame: NDArray[np.uint8],
        polygon: list[tuple[float, float]] | None,
        timestamp: float,
    ) -> tuple[MotionAssessment | None, str]:
        """Run one calibrated monitor observation and normalize bootstrap noise."""
        candidates = []
        for ball in self._phase_last_ball_map:
            coordinate_space = getattr(ball, "coordinate_space", "pixel")
            if coordinate_space == "4k_canonical":
                from ..core.resolution_converter import ResolutionConverter
                from .detection.detector_adapter import canonical_point_to_source_pixels

                position = canonical_point_to_source_pixels(
                    ball.position,
                    ball.source_resolution or (frame.shape[1], frame.shape[0]),
                )
                radius = ResolutionConverter.scale_distance_from_4k(
                    ball.radius,
                    ball.source_resolution or (frame.shape[1], frame.shape[0]),
                )
            elif coordinate_space == "pixel":
                position, radius = ball.position, ball.radius
            else:
                continue
            candidates.append((float(position[0]), float(position[1]), float(radius)))

        assessment = self.motion_monitor.assess(
            frame,
            polygon,
            candidate_balls=candidates,
            timestamp=timestamp,
        )
        return assessment, "observed" if assessment.valid else "failed"

    def _process_single_frame(
        self,
        frame: NDArray[np.uint8],
        frame_number: int | None = None,
        timestamp: float | None = None,
        source_frame_number: int | None = None,
        source_timestamp: float | None = None,
        source_timestamp_kind: str | None = None,
        source_process_id: int | None = None,
        source_publication_monotonic_ns: int | None = None,
    ) -> DetectionResult | None:
        """Process a single frame through the complete pipeline.

        Args:
            frame: Input frame to process
            frame_number: Frame sequence number
            timestamp: Frame timestamp

        Returns:
            Detection results or None if processing failed
        """
        try:
            start_time = time.monotonic()

            if frame_number is None:
                frame_number = self._frame_number
                self._frame_number += 1

            if timestamp is None:
                timestamp = time.time()

            source_height, source_width = frame.shape[:2]
            source_resolution = (source_width, source_height)
            if source_frame_number is not None:
                previous_source_process_id = getattr(
                    self, "_phase_last_source_process_id", None
                )
                source_process_restarted = (
                    source_process_id is not None
                    and previous_source_process_id is not None
                    and source_process_id != previous_source_process_id
                )
                if source_process_restarted:
                    self._reset_after_source_restart()
                previous_source_frame = getattr(
                    self, "_phase_last_source_frame_number", None
                )
                previous_source_timestamp = getattr(
                    self, "_phase_last_source_timestamp", None
                )
                if (
                    not source_process_restarted
                    and previous_source_frame is not None
                    and source_frame_number <= previous_source_frame
                ):
                    reset_at = int(
                        _get_config_value(
                            "vision.processing.source_restart_sequence_reset_threshold",
                            5,
                        )
                    )
                    reset_drop = int(
                        _get_config_value(
                            "vision.processing.source_restart_sequence_drop_threshold",
                            60,
                        )
                    )
                    reset_sequence = (
                        source_frame_number < previous_source_frame
                        and previous_source_frame > reset_at
                        and source_frame_number <= reset_at
                    )
                    reset_large_drop = (
                        source_frame_number < previous_source_frame
                        and previous_source_frame - source_frame_number >= reset_drop
                        and source_timestamp is not None
                        and previous_source_timestamp is not None
                        and source_timestamp > previous_source_timestamp
                    )
                    if reset_sequence or reset_large_drop:
                        self._reset_after_source_restart()
                    else:
                        self.stats.frames_dropped += 1
                        return None
                elif (
                    not source_process_restarted
                    and source_timestamp is not None
                    and previous_source_timestamp is not None
                    and source_timestamp <= previous_source_timestamp
                ):
                    self.stats.frames_dropped += 1
                    return None
                self._phase_last_source_frame_number = source_frame_number
                self._phase_last_source_process_id = source_process_id
                if source_timestamp is not None:
                    self._phase_last_source_timestamp = source_timestamp
            phase_aware = bool(self.phase_scheduler.enabled)
            phase_decision: PhaseDecision | None = None
            table_polygon: list[tuple[float, float]] | None = None
            calibration_revision: str | None = None
            frame_monotonic_time = self._monotonic_now()

            if phase_aware:
                table_polygon, calibration_revision = self._phase_table_geometry(
                    frame.shape
                )
                geometry_signature = (
                    source_width,
                    source_height,
                    calibration_revision,
                )
                if geometry_signature != self._phase_geometry_signature:
                    # Reference pixels and source-coordinate map entries are
                    # tied to both camera resolution and table calibration.
                    self.motion_monitor.reset()
                    self._phase_last_ball_map = []
                    self._phase_last_ball_timestamp = None
                    self._phase_last_ball_source_timestamp = None
                    self._phase_last_ball_source_frame_number = None
                    self._phase_last_ball_source_timestamp_kind = None
                    self._phase_last_motion_assessment = None
                    self._phase_last_motion_timestamp = None
                    self._phase_last_cue = None
                    self._phase_last_cue_timestamp = None
                    self._phase_geometry_signature = geometry_signature

                phase_decision = self.phase_scheduler.update(
                    now=frame_monotonic_time,
                    calibration_revision=calibration_revision,
                    aiming_requested=self._resolve_aiming_request(),
                )
                self._phase_calibration_revision = calibration_revision

            # Start profiling this frame

            # Preserve the original frame for display. Scheduled inference can
            # use a smaller view when a validated phase profile requests one.
            processed_frame = frame

            # Apply masking ONCE - marker dots and table boundaries
            # This happens BEFORE any detection so YOLO/OpenCV never see masked regions

            if phase_aware and phase_decision and phase_decision.run_motion_monitor:
                motion_timestamp = self._monotonic_now()
                motion_assessment, motion_status = self._assess_phase_motion(
                    frame, table_polygon, motion_timestamp
                )
                if motion_assessment is not None:
                    self._phase_last_motion_assessment = motion_assessment
                    self._phase_last_motion_timestamp = motion_timestamp
            else:
                motion_assessment = None
                motion_status = "skipped"

            processed_frame = self._apply_all_masks(processed_frame)
            processing_max_dimension = (
                phase_decision.processing_max_dimension
                if phase_aware and phase_decision
                else None
            )
            processing_input = self._resize_for_phase(
                processed_frame, processing_max_dimension
            )
            processing_height, processing_width = processing_input.shape[:2]
            processing_resolution = (processing_width, processing_height)
            ball_processing_input = processing_input
            ball_crop_rect: tuple[int, int, int, int] | None = None
            ball_detection_view = "full_frame"
            crop_profile = (
                self.phase_scheduler.settings.get("ball_detection_crop", {})
                if phase_aware
                else {}
            )
            crop_balls_to_table = bool(
                phase_decision and crop_profile.get(phase_decision.phase.value, False)
            )
            if crop_balls_to_table:
                ball_processing_input, ball_crop_rect = self._crop_to_calibrated_table(
                    processing_input,
                    table_polygon,
                    source_resolution,
                    self.phase_scheduler.settings.get(
                        "ball_crop_margin_fraction", 0.08
                    ),
                )
                ball_detection_view = (
                    "calibrated_table_crop"
                    if ball_crop_rect is not None
                    else "full_frame_calibration_unavailable"
                )

            # A unique per-processing-pass token prevents reuse when sequence
            # metadata repeats. Source metadata and view name make the cache
            # contract explicit to detector implementations.
            frame_cache_key = (
                "vision-frame",
                frame_number,
                source_frame_number,
                source_timestamp,
                source_timestamp_kind,
                f"masked-{processing_width}x{processing_height}",
                object(),
            )
            ball_frame_cache_key = (
                frame_cache_key
                if ball_crop_rect is None
                else (*frame_cache_key, "table-crop", ball_crop_rect)
            )

            # Ball and cue inference have independent schedules. Skipped jobs
            # preserve their latest successful result and never call tracking.
            run_ball_detection = bool(
                self.detector
                and self.config.enable_ball_detection
                and (
                    not phase_aware
                    or (
                        phase_decision
                        and (
                            phase_decision.verify_ball_map
                            or phase_decision.track_moving_balls
                        )
                    )
                )
            )
            run_cue_detection = bool(
                self.detector
                and hasattr(self.detector, "detect_cue")
                and (not phase_aware or (phase_decision and phase_decision.detect_cue))
            )
            ball_detection_status = "skipped"
            ball_detection_failure_stage: str | None = None
            cue_detection_status = "skipped"
            detected_balls = list(self._phase_last_ball_map) if phase_aware else []
            map_candidate_balls: list[Any] = []
            ball_observation_timestamp: float | None = None
            cue_observation_timestamp: float | None = None
            ball_map_observation: BallMapObservation | None = None
            cue_observation: CueObservation | None = None
            detected_cue = self._phase_last_cue if phase_aware else None

            if phase_aware and phase_decision and phase_decision.verify_ball_map:
                if not run_ball_detection:
                    ball_detection_status = "failed"
            if phase_aware and phase_decision and phase_decision.detect_cue:
                if not run_cue_detection:
                    cue_detection_status = "failed"

            if run_ball_detection and self.detector is not None:
                try:
                    # Keep color classification aligned with the detector's
                    # resized/cropped view while preserving the original color
                    # evidence. Masks are useful for localization inference,
                    # but can erase pixels needed to classify a ball.
                    classification_input = self._resize_for_phase(
                        frame, processing_max_dimension
                    )
                    if ball_crop_rect is not None:
                        crop_x, crop_y, crop_right, crop_bottom = ball_crop_rect
                        classification_input = classification_input[
                            crop_y:crop_bottom, crop_x:crop_right
                        ]
                    outcome_method = getattr(
                        self.detector,
                        "detect_balls_with_classification_outcome",
                        None,
                    )
                    if callable(outcome_method):
                        outcome = outcome_method(
                            ball_processing_input,
                            frame_key=ball_frame_cache_key,
                            classification_frame=classification_input,
                        )
                        raw_status = getattr(outcome, "status", None)
                        status_name = getattr(raw_status, "value", raw_status)
                        status_name = (
                            status_name.lower() if isinstance(status_name, str) else ""
                        )
                        if status_name in _BALL_DETECTION_SUCCESS_STATUSES:
                            # Both detections and a completed empty inference
                            # are valid observations and may update the map.
                            raw_balls = list(getattr(outcome, "detections", ()))
                            ball_detection_status = "observed"
                        elif status_name == "model_unavailable":
                            raw_balls = []
                            ball_detection_status = "unavailable"
                        elif status_name == "failed":
                            raw_balls = []
                            ball_detection_status = "failed"
                            raw_failure_stage = getattr(outcome, "failure_stage", None)
                            failure_stage_name = getattr(
                                raw_failure_stage, "value", raw_failure_stage
                            )
                            ball_detection_failure_stage = (
                                failure_stage_name
                                if isinstance(failure_stage_name, str)
                                and failure_stage_name in _BALL_DETECTION_FAILURE_STAGES
                                else "unknown"
                            )
                        else:
                            # Unknown status values fail closed rather than
                            # treating an empty detection collection as success.
                            raw_balls = []
                            ball_detection_status = "failed"
                            ball_detection_failure_stage = "unknown"
                    else:
                        # Legacy detector plugins have no status-bearing API;
                        # their returned list, including [], is a completed
                        # observation under the legacy contract.
                        raw_balls = self.detector.detect_balls_with_classification(
                            ball_processing_input,
                            frame_key=ball_frame_cache_key,
                            classification_frame=classification_input,
                        )
                        ball_detection_status = "observed"
                    if ball_detection_status == "observed":
                        if ball_crop_rect is None:
                            raw_balls = self._normalise_balls_to_source(
                                list(raw_balls),
                                source_resolution,
                                processing_resolution,
                            )
                        else:
                            raw_balls = self._normalise_cropped_balls_to_source(
                                list(raw_balls),
                                source_resolution,
                                processing_resolution,
                                ball_crop_rect,
                            )

                        filtered_balls: list[Any] = []
                        excluded_count = 0
                        for ball in raw_balls:
                            coordinate_space = getattr(
                                ball, "coordinate_space", "pixel"
                            )
                            if coordinate_space == "4k_canonical":
                                from .detection.detector_adapter import (
                                    canonical_point_to_source_pixels,
                                )

                                ball_x, ball_y = canonical_point_to_source_pixels(
                                    ball.position, source_resolution
                                )
                            elif coordinate_space == "pixel":
                                ball_x, ball_y = ball.position
                            else:
                                filtered_balls.append(ball)
                                continue

                            if self._is_ball_in_invalid_region(
                                ball_x, ball_y, source_width, source_height
                            ):
                                excluded_count += 1
                                logger.debug(
                                    "Excluded ball at (%.1f, %.1f) in an invalid region",
                                    ball_x,
                                    ball_y,
                                )
                            else:
                                filtered_balls.append(ball)

                        if excluded_count:
                            logger.debug(
                                "Filtered out %s ball(s) in invalid regions",
                                excluded_count,
                            )

                        map_candidate_balls = list(filtered_balls)
                        # Detection construction happens after frame capture and can
                        # run long after a frame waited in the processing queue.
                        # Use the frame's wall-clock publication/arrival time for
                        # per-ball freshness; it is not a camera exposure time.
                        for ball in filtered_balls:
                            if hasattr(ball, "last_seen"):
                                ball.last_seen = timestamp
                        detected_balls = list(filtered_balls)
                        if (
                            self.tracker
                            and self.config.enable_tracking
                            and (
                                not phase_aware
                                or (
                                    phase_decision and phase_decision.track_moving_balls
                                )
                            )
                        ):
                            detected_balls = self.tracker.update_tracking(
                                detected_balls,
                                frame_number,
                                # Tracker intervals must use the same monotonic
                                # domain as its expiry and prediction settings.
                                # `timestamp` is wall-clock publication time and
                                # must remain source metadata only.
                                frame_monotonic_time,
                            )
                        ball_detection_status = "observed"
                        ball_observation_timestamp = self._monotonic_now()
                        if phase_aware:
                            self._phase_last_ball_map = list(detected_balls)
                            self._phase_last_ball_timestamp = ball_observation_timestamp
                            self._phase_last_ball_source_timestamp = source_timestamp
                            self._phase_last_ball_source_frame_number = (
                                source_frame_number
                                if source_frame_number is not None
                                else frame_number
                            )
                            self._phase_last_ball_source_timestamp_kind = (
                                source_timestamp_kind
                            )

                        max_balls = _get_config_value(
                            "vision.detection.max_balls_on_table", 16
                        )
                        detection_rate = len(detected_balls) / float(max_balls)
                        self.stats.detection_accuracy["balls"] = min(
                            detection_rate, 1.0
                        )
                    else:
                        # Keep prior phase-aware state for display, but never
                        # normalize or track unavailable/failed model output.
                        self.stats.detection_accuracy["balls"] = 0.0
                        detected_balls = (
                            list(self._phase_last_ball_map) if phase_aware else []
                        )
                except Exception as e:
                    # Backend exceptions may contain local paths or runtime
                    # details; retain only the safe failure category.
                    logger.error("Ball detection failed (%s)", type(e).__name__)
                    self.stats.detection_accuracy["balls"] = 0.0
                    ball_detection_status = "failed"
                    ball_detection_failure_stage = "unknown"
                    detected_balls = (
                        list(self._phase_last_ball_map) if phase_aware else []
                    )
            if phase_aware and phase_decision and phase_decision.verify_ball_map:
                if ball_detection_status == "observed":
                    ball_map_observation = BallMapObservation(
                        timestamp=ball_observation_timestamp,
                        balls=self._ball_positions_in_source(
                            map_candidate_balls, source_resolution
                        ),
                        full=True,
                    )
                elif ball_detection_status in {"failed", "unavailable"}:
                    ball_map_observation = BallMapObservation(
                        timestamp=self._monotonic_now(),
                        balls=(),
                        full=True,
                        successful=False,
                        uncertain=True,
                    )

            if run_cue_detection and self.detector is not None:
                try:
                    yolo_detection = self.detector.detect_cue(
                        processing_input,
                        frame_key=frame_cache_key,
                    )
                    # A scheduled detector call replaces the cached result.
                    # A successful inference with no cue must clear the prior
                    # measurement instead of renewing its age as if it were seen.
                    detected_cue = None
                    if yolo_detection:
                        from .detection.detector_adapter import yolo_cue_to_cue_stick

                        detected_cue = yolo_cue_to_cue_stick(
                            yolo_detection,
                            image_shape=(processing_height, processing_width),
                            min_confidence=0.3,
                        )
                        detected_cue = self._normalise_cue_to_source(
                            detected_cue, source_resolution, processing_resolution
                        )

                    cue_detection_status = "observed"
                    cue_observation_timestamp = self._monotonic_now()
                    self.stats.detection_accuracy["cue"] = (
                        detected_cue.confidence if detected_cue else 0.0
                    )
                    if phase_aware:
                        self._phase_last_cue = detected_cue
                        self._phase_last_cue_timestamp = cue_observation_timestamp
                        minimum_cue_confidence = self.phase_scheduler.settings.get(
                            "minimum_cue_confidence", 0.3
                        )
                        cue_reliable = bool(
                            detected_cue
                            and detected_cue.confidence >= minimum_cue_confidence
                        )
                        cue_observation = CueObservation(
                            timestamp=cue_observation_timestamp,
                            detected=detected_cue is not None,
                            reliable=cue_reliable,
                        )
                except Exception as e:
                    logger.error("Cue detection failed: %s", e)
                    self.stats.detection_accuracy["cue"] = 0.0
                    cue_detection_status = "failed"
                    detected_cue = None
                    if phase_aware:
                        cue_observation_timestamp = self._monotonic_now()
                        self._phase_last_cue = None
                        self._phase_last_cue_timestamp = cue_observation_timestamp
                        cue_observation = CueObservation(
                            timestamp=cue_observation_timestamp,
                            detected=False,
                            reliable=False,
                        )
            if phase_aware and detected_cue is not None:
                cue_age_seconds = (
                    self._monotonic_now() - self._phase_last_cue_timestamp
                    if self._phase_last_cue_timestamp is not None
                    else float("inf")
                )
                maximum_cue_age_seconds = self.phase_scheduler.settings[
                    "maximum_cue_age_seconds"
                ]
                if cue_age_seconds > maximum_cue_age_seconds:
                    # Keep the last observation timestamp for diagnostics, but
                    # stop publishing its geometry once it is no longer fresh.
                    self._phase_last_cue = None
                    detected_cue = None
                    self.stats.detection_accuracy["cue"] = 0.0
                    if cue_detection_status == "skipped":
                        cue_detection_status = "stale"

            if phase_aware and phase_decision and phase_decision.detect_cue:
                if cue_detection_status == "failed" and cue_observation is None:
                    cue_observation_timestamp = self._monotonic_now()
                    cue_observation = CueObservation(
                        timestamp=cue_observation_timestamp,
                        detected=False,
                        reliable=False,
                    )
                elif cue_detection_status == "observed" and cue_observation is None:
                    cue_observation_timestamp = self._monotonic_now()
                    cue_observation = CueObservation(
                        timestamp=cue_observation_timestamp,
                        detected=False,
                        reliable=True,
                    )

            post_phase_decision: PhaseDecision | None = None
            if phase_aware:
                moving_ball_count: int | None = (
                    sum(
                        1
                        for ball in detected_balls
                        if getattr(ball, "is_moving", False)
                    )
                    if ball_detection_status == "observed"
                    else None
                )
                if (
                    moving_ball_count is None
                    and phase_decision is not None
                    and phase_decision.phase.value == "MANY_MOVING"
                    and motion_assessment is not None
                    and motion_assessment.valid
                    and not motion_assessment.motion_detected
                    and not motion_assessment.likely_occlusion
                    and not motion_assessment.illumination_change
                ):
                    # In MANY_MOVING detailed detections are intentionally
                    # suspended. Fresh quiet image evidence still contributes
                    # a zero moving-ball estimate so hysteresis can recover.
                    moving_ball_count = 0
                post_update_time = self._monotonic_now()
                post_phase_decision = self.phase_scheduler.update(
                    now=post_update_time,
                    motion_assessment=motion_assessment,
                    ball_map_observation=ball_map_observation,
                    cue_observation=cue_observation,
                    moving_ball_count=moving_ball_count,
                    aiming_requested=self._resolve_aiming_request(),
                    calibration_revision=calibration_revision,
                )
                if (
                    motion_assessment is not None
                    and motion_assessment.valid
                    and not motion_assessment.motion_detected
                    and not motion_assessment.likely_occlusion
                    and not motion_assessment.illumination_change
                    and ball_map_observation is not None
                    and ball_map_observation.successful
                    and post_phase_decision.map_trusted
                ):
                    self.motion_monitor.commit_verified_reference(
                        frame,
                        table_polygon,
                        settled=True,
                        map_reconciled=True,
                        timestamp=post_update_time,
                    )

            # Create result
            processing_time = (
                time.monotonic() - start_time
            ) * 1000  # Convert to milliseconds
            frame_publication_age_seconds = self._calculate_publication_age_seconds(
                source_timestamp=source_timestamp,
                source_timestamp_kind=source_timestamp_kind,
                source_publication_monotonic_ns=source_publication_monotonic_ns,
                fallback_timestamp=timestamp,
            )

            # Create a default table (no detection needed)
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
                coordinate_space="pixel",
                source_resolution=source_resolution,
            )
            detected_table = self._normalise_table_to_source(
                detected_table, source_resolution, processing_resolution
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
                table_detected=True,
                detection_confidence=sum(self.stats.detection_accuracy.values())
                / max(len(self.stats.detection_accuracy), 1),
                frame_quality=(
                    detected_quality if detected_table is not None else default_quality
                ),
            )

            ball_observation_source_frame_number: int | None
            if ball_detection_status == "observed":
                ball_positions_status = "observed"
                ball_observation_source_timestamp = source_timestamp
                ball_observation_source_frame_number = (
                    source_frame_number
                    if source_frame_number is not None
                    else frame_number
                )
                ball_observation_source_timestamp_kind = source_timestamp_kind
            elif phase_aware and self._phase_last_ball_timestamp is not None:
                # The ball list is intentionally retained through scheduler
                # skips and inference failures, but its source identity must
                # remain the successful observation that produced it.
                ball_positions_status = "cached"
                ball_observation_source_timestamp = (
                    self._phase_last_ball_source_timestamp
                )
                ball_observation_source_frame_number = (
                    self._phase_last_ball_source_frame_number
                )
                ball_observation_source_timestamp_kind = (
                    self._phase_last_ball_source_timestamp_kind
                )
            else:
                ball_positions_status = "unavailable"
                ball_observation_source_timestamp = None
                ball_observation_source_frame_number = None
                ball_observation_source_timestamp_kind = None

            component_statuses = {
                "balls": ball_detection_status,
                "cue": cue_detection_status,
                "motion": motion_status,
            }
            failed_components = {
                name: status
                for name, status in component_statuses.items()
                if status == "failed"
            }
            incomplete_components = {
                name: status
                for name, status in component_statuses.items()
                if status in {"failed", "unavailable"}
            }
            safe_error_messages = [
                (
                    f"balls:failed:{ball_detection_failure_stage or 'unknown'}"
                    if name == "balls"
                    else f"{name}:failed"
                )
                for name in failed_components
            ]

            result = DetectionResult(
                frame_number=frame_number,
                timestamp=timestamp,
                balls=detected_balls,
                cue=detected_cue,
                table=detected_table,
                statistics=statistics,
                source_frame_number=source_frame_number,
                source_timestamp=source_timestamp,
                source_timestamp_kind=source_timestamp_kind,
                source_process_id=source_process_id,
                source_publication_monotonic_ns=source_publication_monotonic_ns,
                frame_publication_age_seconds=frame_publication_age_seconds,
                phase_aware=phase_aware,
                vision_phase=(
                    post_phase_decision.phase.value.lower()
                    if post_phase_decision
                    else None
                ),
                phase=(
                    post_phase_decision.phase.value.lower()
                    if post_phase_decision
                    else None
                ),
                phase_reason=(
                    post_phase_decision.reason if post_phase_decision else None
                ),
                map_revision=(
                    post_phase_decision.map_revision if post_phase_decision else None
                ),
                map_trusted=(
                    post_phase_decision.map_trusted if post_phase_decision else None
                ),
                moving_ball_count=(
                    post_phase_decision.moving_ball_count
                    if post_phase_decision
                    else None
                ),
                aim_calculation_eligible=(
                    post_phase_decision.can_calculate_aim
                    if post_phase_decision
                    else None
                ),
                calibration_revision=calibration_revision,
                is_complete=not incomplete_components,
                has_errors=bool(failed_components),
                error_messages=safe_error_messages,
                ball_detection_status=ball_detection_status,
                ball_detection_failure_stage=ball_detection_failure_stage,
                cue_detection_status=cue_detection_status,
                motion_monitor_status=motion_status,
                ball_positions_status=ball_positions_status,
                ball_observation_source_timestamp=(ball_observation_source_timestamp),
                ball_observation_source_frame_number=(
                    ball_observation_source_frame_number
                ),
                ball_observation_source_timestamp_kind=(
                    ball_observation_source_timestamp_kind
                ),
                ball_observation_timestamp=(
                    self._phase_last_ball_timestamp if phase_aware else None
                ),
                maximum_ball_observation_gap_seconds=(
                    self.phase_scheduler.settings.get(
                        "maximum_ball_observation_gap_seconds"
                    )
                    if phase_aware
                    else None
                ),
                actual_path_history_max_points=(
                    self.phase_scheduler.settings.get("actual_path_history_max_points")
                    if phase_aware
                    else None
                ),
                ball_detection_view=ball_detection_view,
                cue_observation_timestamp=(
                    self._phase_last_cue_timestamp if phase_aware else None
                ),
                motion_observation_timestamp=(
                    self._phase_last_motion_timestamp if phase_aware else None
                ),
                component_observation_status=(
                    component_statuses if phase_aware else {}
                ),
            )

            # End profiling for this frame

            return result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.stats.last_error = str(e)

            return None

    @staticmethod
    def _calculate_publication_age_seconds(
        *,
        source_timestamp: float | None,
        source_timestamp_kind: str | None,
        source_publication_monotonic_ns: int | None,
        fallback_timestamp: float,
    ) -> float | None:
        """Measure age from shared-memory publication using elapsed-time clocks.

        Shared-memory publication timestamps and ``perf_counter_ns`` are
        produced on the same machine, so their difference is not affected by
        wall-clock corrections. Older records may not contain the reserved
        monotonic header value; those retain the legacy wall-clock estimate.
        A negative or otherwise invalid delta is unavailable, never clamped to
        zero, because doing so could make a future or incomparable timestamp
        appear fresh.
        """
        if source_publication_monotonic_ns is not None:
            if (
                isinstance(source_publication_monotonic_ns, bool)
                or not isinstance(source_publication_monotonic_ns, int)
                or source_publication_monotonic_ns <= 0
            ):
                return None

            try:
                age_ns = time.perf_counter_ns() - source_publication_monotonic_ns
            except (OSError, RuntimeError):
                return None
            if age_ns < 0:
                return None
            return age_ns / 1_000_000_000

        # Records predating the monotonic shared-memory field retain their
        # wall-clock age estimate. The fallback timestamp is only meaningful
        # for frames captured without shared-memory metadata.
        age_reference_timestamp = source_timestamp
        if (
            age_reference_timestamp is None
            and source_timestamp_kind == "vision_capture_fallback"
        ):
            age_reference_timestamp = fallback_timestamp
        if age_reference_timestamp is None or not math.isfinite(
            age_reference_timestamp
        ):
            return None

        wall_age_seconds = time.time() - age_reference_timestamp
        if not math.isfinite(wall_age_seconds) or wall_age_seconds < 0:
            return None
        return wall_age_seconds

    def _is_ball_in_invalid_region(
        self, x: float, y: float, frame_width: int, frame_height: int
    ) -> bool:
        """Check if a ball detection is in an invalid region.

        Invalid regions for BALLS only:
        1. Marker dots - can't have balls on marker stickers
        2. Outside playing area - balls can't be outside the table boundary

        Note: Cue stick CAN be on rails (outside playing area), so this check
        is ONLY for ball detections, not cue detections.

        Args:
            x, y: Ball center coordinates
            frame_width, frame_height: Current frame dimensions

        Returns:
            True if ball is in an invalid region and should be excluded
        """
        try:
            from ..config import config

            frame_size = (frame_width, frame_height)

            # Check 1: Marker dots (balls can't be on marker stickers)
            marker_config = config.get("vision.detection.marker_filtering", {})
            if marker_config.get("enabled", True):
                marker_dots = config.get("table.marker_dots", [])
                mask_radius = marker_config.get("mask_radius_px")

                for dot in marker_dots:
                    try:
                        center = point_to_frame(dot, frame_size)
                        if point_is_within_scaled_radius(
                            x,
                            y,
                            center,
                            mask_radius,
                            dot["scale"],
                            frame_size,
                        ):
                            return True  # Ball is on a marker dot
                    except (KeyError, TypeError, ValueError):
                        logger.warning(
                            "Skipping marker exclusion with invalid scaled geometry"
                        )

            # Check 2: Outside playing area (balls can't be outside the table)
            playing_area_corners = config.get("table.playing_area_corners", [])
            if len(playing_area_corners) == 4:
                try:
                    corners = np.asarray(
                        playing_area_to_frame(playing_area_corners, frame_size),
                        dtype=np.float32,
                    )
                    result = cv2.pointPolygonTest(corners, (float(x), float(y)), False)
                    if result < 0:
                        return True  # Ball is outside the playing area
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Skipping playing-area exclusion with invalid scaled geometry"
                    )

            return False  # Ball is in a valid location

        except Exception as e:
            logger.error(f"Failed to check ball region: {e}")
            return False  # Don't filter on error

    def _apply_all_masks(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply marker dot masking before detection.

        Masks applied:
        1. Marker dots - painted over with dark green (post-filtering excludes ball detections)

        Note: Boundary masking is DISABLED because cue stick can be on rails.
        Post-filtering in detect_balls ensures ball detections in invalid regions
        (marker dots or outside playing area) are excluded.

        Configured marker coordinates and their radii are scaled through their
        explicit canonical scale metadata into the current frame.

        Args:
            frame: Preprocessed frame

        Returns:
            Frame with marker dots masked
        """
        try:
            from ..config import config

            # Create copy to avoid modifying original
            masked_frame = frame.copy()

            # Get current frame dimensions
            frame_height, frame_width = frame.shape[:2]

            frame_size = (frame_width, frame_height)

            # MASK 1: Marker dots (spots/stickers on table)
            marker_config = config.get("vision.detection.marker_filtering", {})
            marker_filtering_enabled = marker_config.get("enabled", True)

            if marker_filtering_enabled:
                marker_dots = config.get("table.marker_dots", [])
                mask_radius = marker_config.get("mask_radius_px")

                if marker_dots:
                    # Log masking config only on first application (avoid spam)
                    if not hasattr(self, "_logged_marker_masking"):
                        logger.info(
                            f"Applying marker dot masking: {len(marker_dots)} dots, "
                            f"reference radius={mask_radius}px "
                            f"(post-filtering will exclude any detections in these regions)"
                        )
                        self._logged_marker_masking = True

                    # Paint dots with any color - doesn't matter since post-filtering
                    # will exclude detections in these regions anyway
                    # Using (0, 128, 0) dark green as a visual indicator
                    mask_color = (0, 128, 0)  # Dark green - easy to spot if needed

                    for dot in marker_dots:
                        try:
                            scaled_x, scaled_y = point_to_frame(dot, frame_size)
                            radius_x, radius_y = radius_to_frame_ellipse(
                                mask_radius, dot["scale"], frame_size
                            )
                            cv2.ellipse(
                                masked_frame,
                                (round(scaled_x), round(scaled_y)),
                                (max(1, round(radius_x)), max(1, round(radius_y))),
                                0,
                                0,
                                360,
                                mask_color,
                                -1,
                            )
                        except (KeyError, TypeError, ValueError):
                            logger.warning(
                                "Skipping marker mask with invalid scaled geometry"
                            )
                else:
                    if not hasattr(self, "_logged_no_marker_dots"):
                        logger.warning(
                            "Marker filtering enabled but no marker dots configured!"
                        )
                        self._logged_no_marker_dots = True
            else:
                if not hasattr(self, "_logged_marker_disabled"):
                    logger.info("Marker dot filtering is DISABLED")
                    self._logged_marker_disabled = True

            # MASK 2: Boundary masking DISABLED
            # We DON'T mask the outer rail area because:
            # - Cue stick often rests on the rails and needs to be detected
            # - Post-filtering will reject BALLS outside playing area instead
            # This keeps the full frame visible to YOLO while still preventing
            # false ball detections outside the playing area

            return masked_frame

        except Exception as e:
            logger.error(f"Failed to apply masks: {e}", exc_info=True)
            return frame  # Return unmasked on error

    def _apply_roi(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
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
            return cast("NDArray[np.uint8]", result)

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

            # Set background for the detector's integrated OpenCV classifier.
            if self.detector:
                self.detector.set_background_frame(
                    cast("NDArray[np.uint8]", background_frame)
                )
                logger.info("Background frame set for integrated ball classifier")

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
    # Data models
    # Note: Ball has been moved to backend.common.models
    "BallType",
    "CameraFrame",
    "CueStick",
    "CueState",
    "Table",
    "Pocket",
    "PocketType",
    "DetectionResult",
    "FrameStatistics",
    # Shared memory IPC
    "VideoConsumer",
    "VideoModuleNotAvailableError",
    # Calibration
    "CameraCalibration",
    "ColorCalibration",
    "GeometricCalibration",
    "CalibrationData",
    # Session tracking
    "DetectionSession",
    "ShotEvent",
]
