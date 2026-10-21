"""Detection algorithms used by the billiards vision system."""

from .balls import BallDetectionConfig, BallDetector, DetectionMethod
from .hybrid_validator import (
    BallFeatures,
    BallPositionRefiner,
    extract_ball_features,
    refine_ball_position,
)
from .yolo_detector import (
    BallClass,
    Detection,
    ModelFormat,
    TableElements,
    YOLODetector,
    ball_class_to_type,
    create_detector,
)

__all__ = [
    # OpenCV ball classification and detection helpers.
    "BallDetector",
    "BallDetectionConfig",
    "DetectionMethod",
    # YOLO detection (primary implementation)
    "YOLODetector",
    "Detection",
    "TableElements",
    "BallClass",
    "ModelFormat",
    "create_detector",
    "ball_class_to_type",
    # Hybrid validation and refinement
    "BallPositionRefiner",
    "refine_ball_position",
    "BallFeatures",
    "extract_ball_features",
]
