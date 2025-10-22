"""Detector factory for creating ball and cue detectors.

Provides a unified interface for creating detectors using YOLO+OpenCV hybrid detection.
YOLO is used for object localization with mandatory OpenCV refinement for classification.

The factory pattern simplifies testing and configuration management.

Note: Pure OpenCV-only detection has been removed. YOLO is now the only supported
backend, and OpenCV classification is always enabled for accurate ball type detection.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ..models import Ball, CueStick

# Keep BallDetector import for reference/testing only
# It is no longer used in production paths
from .balls import BallDetector  # noqa: F401 - kept for testing


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


# OpenCVDetector class has been removed - YOLO+OpenCV hybrid is now the only supported approach
# Pure OpenCV detection (without YOLO localization) is no longer used in production paths


class YOLODetector(BaseDetector):
    """YOLO+OpenCV hybrid detector implementation.

    This detector uses YOLO (You Only Look Once) neural network for object localization
    with mandatory OpenCV refinement for accurate ball type classification.

    The hybrid approach combines:
    - YOLO: Fast, accurate object detection and localization
    - OpenCV: Precise color-based classification for ball types (solids vs stripes)

    This is the only supported detection backend. Pure OpenCV-only detection has been
    removed as it was less accurate for object localization.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize YOLO+OpenCV hybrid detector with configuration.

        OpenCV classification is always enabled for accurate ball type detection.

        Args:
            config: Configuration dictionary with YOLO model parameters:
                - model_path: Path to YOLO model weights
                - confidence_threshold: Detection confidence threshold (default: 0.5)
                - nms_threshold: Non-maximum suppression threshold (default: 0.4)
                - Additional OpenCV classification parameters
        """
        self.config = config
        # Future: Load YOLO model, weights, and configuration
        # self.model = load_yolo_model(config.get("model_path"))
        # self.confidence_threshold = config.get("confidence_threshold", 0.5)
        # self.nms_threshold = config.get("nms_threshold", 0.4)
        #
        # OpenCV classification is always enabled (mandatory in hybrid mode)
        # self.opencv_classifier = OpenCVClassifier(config)

    def detect_balls(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> list[Ball]:
        """Detect balls using YOLO localization + OpenCV classification.

        Args:
            frame: Input frame in BGR format
            table_mask: Optional mask for table region

        Returns:
            List of detected Ball objects with accurate positions and types

        Raises:
            NotImplementedError: YOLO backend not yet implemented
        """
        raise NotImplementedError(
            "YOLO+OpenCV hybrid detector backend is not yet implemented. "
            "This is the planned implementation that will replace pure OpenCV detection."
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
            "YOLO+OpenCV hybrid detector backend is not yet implemented. "
            "This is the planned implementation that will replace pure OpenCV detection."
        )


def create_detector(
    backend: str = "yolo", config: Optional[dict[str, Any]] = None
) -> BaseDetector:
    """Factory function to create YOLO+OpenCV hybrid detector.

    Creates a detector instance using YOLO for object localization with mandatory
    OpenCV refinement for ball type classification. This is the only supported
    detection approach.

    Args:
        backend: Detector backend to use (must be 'yolo', default: 'yolo')
        config: Optional configuration dictionary for the detector with YOLO parameters:
                - model_path: Path to YOLO model weights
                - confidence_threshold: Detection confidence threshold
                - nms_threshold: Non-maximum suppression threshold
                - OpenCV classification parameters

    Returns:
        YOLODetector instance configured for hybrid detection

    Raises:
        ValueError: If backend is not 'yolo' (e.g., if 'opencv' is requested)

    Examples:
        >>> # Create YOLO+OpenCV hybrid detector with defaults
        >>> detector = create_detector()  # 'yolo' is default

        >>> # Create detector with custom YOLO configuration
        >>> config = {
        ...     'model_path': 'models/yolov8n-pool.onnx',
        ...     'confidence_threshold': 0.6,
        ...     'nms_threshold': 0.4
        ... }
        >>> detector = create_detector('yolo', config)

    Note:
        The 'opencv' backend option has been permanently removed. All detection
        now uses YOLO for localization with mandatory OpenCV classification.
        Pure OpenCV detection (without YOLO) has been removed due to lower
        localization accuracy.
    """
    if config is None:
        config = {}

    backend_lower = backend.lower()

    if backend_lower == "yolo":
        # Always use YOLO+OpenCV hybrid (OpenCV classification is mandatory)
        return YOLODetector(config)
    elif backend_lower == "opencv":
        raise ValueError(
            "OpenCV-only backend is no longer supported. "
            "The system now exclusively uses YOLO+OpenCV hybrid detection. "
            "Please use backend='yolo' (default) which includes mandatory OpenCV classification. "
            "Pure OpenCV detection has been removed due to inferior localization accuracy."
        )
    else:
        raise ValueError(
            f"Unknown detector backend: '{backend}'. "
            f"Only 'yolo' is supported (with mandatory OpenCV classification for ball types)."
        )
