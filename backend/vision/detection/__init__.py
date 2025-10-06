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
from .detector_factory import BaseDetector, OpenCVDetector
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
    # Ball detection
    "BallDetector",
    "BallDetectionConfig",
    "DetectionMethod",
    # Cue detection
    "CueDetector",
    # YOLO detection (legacy)
    "YOLODetector",
    "Detection",
    "TableElements",
    "BallClass",
    "ModelFormat",
    "create_detector",
    "ball_class_to_type",
    # Detector factory (new unified interface)
    "BaseDetector",
    "OpenCVDetector",
    "FactoryYOLODetector",
    "create_unified_detector",
    # Hybrid validation and refinement
    "BallPositionRefiner",
    "refine_ball_position",
    "BallFeatures",
    "extract_ball_features",
]
