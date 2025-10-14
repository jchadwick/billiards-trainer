"""Detection module for billiards vision system.

Provides object detection algorithms for:
- Balls (YOLO + OpenCV fallback)
- Cue stick (YOLO + line detection fallback)
- Table elements (YOLO + geometric detection)

Implements hybrid detection approach combining deep learning (YOLOv8)
with traditional computer vision (OpenCV) for robust performance.
"""

from .balls import BallDetectionConfig, BallDetector, DetectionMethod
from .cue import CueDetector
from .detector_factory import BaseDetector
from .detector_factory import YOLODetector as FactoryYOLODetector
from .detector_factory import create_detector as create_unified_detector
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
    # Ball detection (kept for reference/testing, not used in production)
    "BallDetector",
    "BallDetectionConfig",
    "DetectionMethod",
    # Cue detection (kept for reference/testing, not used in production)
    "CueDetector",
    # YOLO detection (primary implementation)
    "YOLODetector",
    "Detection",
    "TableElements",
    "BallClass",
    "ModelFormat",
    "create_detector",
    "ball_class_to_type",
    # Detector factory (unified interface - YOLO+OpenCV hybrid only)
    "BaseDetector",
    "FactoryYOLODetector",
    "create_unified_detector",
    # Hybrid validation and refinement
    "BallPositionRefiner",
    "refine_ball_position",
    "BallFeatures",
    "extract_ball_features",
]
