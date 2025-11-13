"""Analysis module exports."""

from .prediction import ShotPredictor, TrajectoryPredictor
from .shot import IllegalShotReason, ShotAnalysis, ShotAnalyzer

__all__ = [
    "TrajectoryPredictor",
    "ShotPredictor",  # Backward compatibility alias
    "ShotAnalyzer",
    "ShotAnalysis",
    "IllegalShotReason",
]
