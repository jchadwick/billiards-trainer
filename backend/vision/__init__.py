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
from .detection.table import TableDetector

# Import vision components
from .models import (
    Ball,
    BallType,
    CalibrationData,
    CameraCalibration,
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
from .tracking.tracker import ObjectTracker

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for Vision Module."""

    # Camera settings
    camera_device_id: int = 0
    camera_backend: str = "auto"
    camera_resolution: tuple[int, int] = (1920, 1080)
    camera_fps: int = 30
    camera_buffer_size: int = 1

    # Processing settings
    target_fps: int = 30
    enable_threading: bool = True
    enable_gpu: bool = False
    max_frame_queue_size: int = 5

    # Detection settings
    enable_table_detection: bool = True
    enable_ball_detection: bool = True
    enable_cue_detection: bool = True
    enable_tracking: bool = True

    # Performance settings
    frame_skip: int = 0
    roi_enabled: bool = False
    preprocessing_enabled: bool = True

    # Debug settings
    debug_mode: bool = False
    save_debug_images: bool = False
    debug_output_path: str = "/tmp/vision_debug"


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
        # Parse configuration
        if config is None:
            config = {}
        self.config = VisionConfig(**config)

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
            # Camera capture
            camera_config = {
                "device_id": self.config.camera_device_id,
                "backend": self.config.camera_backend,
                "resolution": self.config.camera_resolution,
                "fps": self.config.camera_fps,
                "buffer_size": self.config.camera_buffer_size,
            }
            self.camera = CameraCapture(camera_config)

            # Image preprocessing - create config based on vision module settings
            preprocessing_config = {}
            if hasattr(self.config, "enable_gpu"):
                # Note: PreprocessingConfig doesn't have a use_gpu field, so we skip it
                pass
            self.preprocessor = ImagePreprocessor(preprocessing_config)

            # Detection components
            detection_config = {"debug_mode": self.config.debug_mode}

            if self.config.enable_table_detection:
                self.table_detector = TableDetector(detection_config)
            else:
                self.table_detector = None

            if self.config.enable_ball_detection:
                self.ball_detector = BallDetector(detection_config)
            else:
                self.ball_detector = None

            if self.config.enable_cue_detection:
                self.cue_detector = CueDetector(detection_config)
            else:
                self.cue_detector = None

            # Tracking
            if self.config.enable_tracking:
                self.tracker = ObjectTracker({})
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
            if self._is_running:
                logger.warning("Capture is already running")
                return True

            # Start camera
            if not self.camera.start_capture():
                raise VisionModuleError("Failed to start camera capture")

            # Reset statistics
            self.stats = VisionStatistics()
            self._start_time = time.time()
            self._frame_number = 0

            # Start processing threads
            self._is_running = True

            if self.config.enable_threading:
                self._capture_thread = threading.Thread(
                    target=self._capture_loop, name="VisionCapture", daemon=True
                )
                self._processing_thread = threading.Thread(
                    target=self._processing_loop, name="VisionProcessing", daemon=True
                )

                self._capture_thread.start()
                self._processing_thread.start()

            logger.info("Vision capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
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
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=5.0)

            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)

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
        with self._lock:
            return (
                self._current_frame.copy() if self._current_frame is not None else None
            )

    def calibrate_camera(self) -> bool:
        """Perform camera calibration.

        Returns:
            True if calibration successful
        """
        try:
            logger.info("Starting camera calibration")

            # Capture calibration images
            calibration_frames = []
            for _i in range(20):  # Capture 20 frames for calibration
                frame_data = self.camera.get_latest_frame()
                if frame_data is not None:
                    frame, frame_info = frame_data
                    calibration_frames.append(frame)
                time.sleep(0.1)

            if len(calibration_frames) < 10:
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
            if len(corners) != 4:
                raise ValueError("ROI must have exactly 4 corners")

            # Validate corners
            for corner in corners:
                if not isinstance(corner, (tuple, list)) or len(corner) != 2:
                    raise ValueError("Each corner must be a (x, y) tuple")
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
        logger.info("Starting capture loop")

        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = 0

        while self._is_running:
            try:
                current_time = time.time()

                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue

                # Get frame from camera (updated interface)
                frame_data = self.camera.get_latest_frame()
                if frame_data is None:
                    self.stats.frames_dropped += 1
                    continue

                frame, frame_info = frame_data

                # Apply ROI if enabled
                if self.config.roi_enabled and self._roi_corners:
                    frame = self._apply_roi(frame)

                # Update current frame
                with self._lock:
                    self._current_frame = frame

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
                    self.stats.frames_dropped += 1

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.stats.last_error = str(e)
                self._emit_event("error_occurred", {"error": str(e)})
                time.sleep(0.1)  # Brief pause before retry

        logger.info("Capture loop ended")

    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread."""
        logger.info("Starting processing loop")

        while self._is_running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_data = self._frame_queue.get(timeout=0.1)
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
                time.sleep(0.01)  # Brief pause before retry

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
                    table_result = self.table_detector.detect_complete_table(
                        processed_frame
                    )
                    if table_result and table_result.confidence > 0.5:
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

            # Ball detection
            if self.ball_detector and self.config.enable_ball_detection:
                try:
                    detected_balls = self.ball_detector.detect_balls(processed_frame)

                    # Update tracking if available
                    if self.tracker and self.config.enable_tracking:
                        detected_balls = self.tracker.update_tracking(
                            detected_balls, timestamp
                        )

                    detection_rate = len(detected_balls) / 16.0  # Assume max 16 balls
                    self.stats.detection_accuracy["balls"] = min(detection_rate, 1.0)

                except Exception as e:
                    logger.warning(f"Ball detection failed: {e}")
                    self.stats.detection_accuracy["balls"] = 0.0

            # Cue detection
            if self.cue_detector and self.config.enable_cue_detection:
                try:
                    detected_cue = self.cue_detector.detect_cue(processed_frame)
                    self.stats.detection_accuracy["cue"] = 1.0 if detected_cue else 0.0

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
                detected_table = Table(
                    corners=[(0, 0), (w, 0), (w, h), (0, h)],
                    pockets=[],
                    width=w,
                    height=h,
                    surface_color=(60, 200, 100),
                )

            # Create frame statistics
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
                frame_quality=1.0 if detected_table is not None else 0.5,
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
        if not self._roi_corners or len(self._roi_corners) != 4:
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
    "Ball",
    "BallType",
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
