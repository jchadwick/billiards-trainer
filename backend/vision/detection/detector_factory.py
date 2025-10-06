"""Detector factory for creating ball and cue detectors with pluggable backends.

Provides a unified interface for creating detectors with different implementations:
- YOLO-based detector (future ML backend)
- OpenCV-based detector (current computer vision backend)

The factory pattern allows easy swapping of detection backends and simplifies
testing and configuration management.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..models import Ball, CueStick
from .balls import BallDetector
from .cue import CueDetector


class BaseDetector(ABC):
    """Abstract base interface for all detector implementations.

    All detector backends (YOLO, OpenCV, etc.) must implement this interface
    to ensure consistent behavior and enable drop-in replacement.
    """

    @abstractmethod
    def detect_balls(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> list[Ball]:
        """Detect all balls in the frame.

        Args:
            frame: Input frame in BGR format
            table_mask: Optional mask for table region

        Returns:
            List of detected Ball objects with positions, types, and metadata
        """
        pass

    @abstractmethod
    def detect_cue(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[CueStick]:
        """Detect cue stick in the frame.

        Args:
            frame: Input frame in BGR format
            cue_ball_pos: Optional cue ball position for improved detection

        Returns:
            CueStick object if detected, None otherwise
        """
        pass

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set background reference frame for background subtraction.

        Optional method that detectors can implement to support background subtraction.
        Default implementation does nothing. Subclasses should override if needed.

        Args:
            frame: Background reference frame (empty table)
        """
        # Default implementation - subclasses can override
        _ = frame  # Prevent unused parameter warning


class OpenCVDetector(BaseDetector):
    """OpenCV-based detector implementation using traditional computer vision.

    This detector uses classical computer vision algorithms:
    - Hough circles for ball detection
    - Color-based ball classification
    - Line detection for cue stick tracking
    - Background subtraction for improved accuracy
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize OpenCV detector with configuration.

        Args:
            config: Configuration dictionary with detector parameters
        """
        # Extract ball detector config
        ball_config = config.get("ball_detection", {})
        self.ball_detector = BallDetector(ball_config)

        # Extract cue detector config
        cue_config = config.get("cue_detection", {})
        self.cue_detector = CueDetector(cue_config)

    def detect_balls(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> list[Ball]:
        """Detect balls using OpenCV algorithms.

        Uses Hough circle detection combined with color classification
        and background subtraction for robust ball detection.

        Args:
            frame: Input frame in BGR format
            table_mask: Optional mask for table region

        Returns:
            List of detected Ball objects
        """
        return self.ball_detector.detect_balls(frame, table_mask)

    def detect_cue(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[CueStick]:
        """Detect cue stick using line detection algorithms.

        Uses multiple line detection methods (Hough, LSD, morphological)
        and temporal tracking for robust cue detection.

        Args:
            frame: Input frame in BGR format
            cue_ball_pos: Optional cue ball position for improved detection

        Returns:
            CueStick object if detected, None otherwise
        """
        return self.cue_detector.detect_cue(frame, cue_ball_pos)

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set background reference frame for both ball and cue detectors.

        Args:
            frame: Background reference frame (empty table)
        """
        self.ball_detector.set_background_frame(frame)
        self.cue_detector.set_background_frame(frame)


class YOLODetector(BaseDetector):
    """YOLO-based detector implementation (placeholder for future ML backend).

    This detector will use YOLO (You Only Look Once) neural network for
    real-time object detection. Currently raises NotImplementedError.

    Future implementation will:
    - Load trained YOLO model for pool balls and cue sticks
    - Perform inference on input frames
    - Post-process detections into Ball and CueStick objects
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize YOLO detector with configuration.

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        # Future: Load YOLO model, weights, and configuration
        # self.model = load_yolo_model(config.get("model_path"))
        # self.confidence_threshold = config.get("confidence_threshold", 0.5)
        # self.nms_threshold = config.get("nms_threshold", 0.4)

    def detect_balls(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> list[Ball]:
        """Detect balls using YOLO neural network.

        Args:
            frame: Input frame in BGR format
            table_mask: Optional mask for table region

        Returns:
            List of detected Ball objects

        Raises:
            NotImplementedError: YOLO backend not yet implemented
        """
        raise NotImplementedError(
            "YOLO detector backend is not yet implemented. "
            "Use 'opencv' backend for current functionality."
        )

    def detect_cue(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[CueStick]:
        """Detect cue stick using YOLO neural network.

        Args:
            frame: Input frame in BGR format
            cue_ball_pos: Optional cue ball position hint

        Returns:
            CueStick object if detected, None otherwise

        Raises:
            NotImplementedError: YOLO backend not yet implemented
        """
        raise NotImplementedError(
            "YOLO detector backend is not yet implemented. "
            "Use 'opencv' backend for current functionality."
        )


def create_detector(
    backend: str = "opencv", config: Optional[dict[str, Any]] = None
) -> BaseDetector:
    """Factory function to create detector with specified backend.

    Creates a detector instance using the specified backend implementation.
    Validates the backend name and raises an error for unknown backends.

    Args:
        backend: Detector backend to use ('opencv' or 'yolo')
        config: Optional configuration dictionary for the detector

    Returns:
        BaseDetector instance of the requested type

    Raises:
        ValueError: If backend is not recognized

    Examples:
        >>> # Create OpenCV detector with default config
        >>> detector = create_detector('opencv')

        >>> # Create OpenCV detector with custom config
        >>> config = {'ball_detection': {'min_radius': 15}}
        >>> detector = create_detector('opencv', config)

        >>> # Create YOLO detector (when implemented)
        >>> detector = create_detector('yolo', {'model_path': 'model.pt'})
    """
    if config is None:
        config = {}

    backend_lower = backend.lower()

    if backend_lower == "opencv":
        return OpenCVDetector(config)
    elif backend_lower == "yolo":
        return YOLODetector(config)
    else:
        raise ValueError(
            f"Unknown detector backend: '{backend}'. "
            f"Supported backends are: 'opencv', 'yolo'"
        )
