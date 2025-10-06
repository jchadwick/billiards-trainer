"""Tracking module for multi-object tracking and prediction.

This module provides comprehensive tracking capabilities including:
- Kalman filter-based position and velocity estimation
- Multi-object tracking with Hungarian algorithm association
- Performance optimization and adaptive parameter tuning
- Smooth trajectory generation and missing detection prediction
- Integration utilities for various detection formats
"""

# Optional integration utilities
try:
    from .integration import (
        IntegratedTracker,
        MissingDetectionPredictor,
        TrackingConfig,
        TrackingResult,
        TrajectorySmoothing,
        convert_detection_format,
        create_integrated_tracker,
    )
except ImportError:
    IntegratedTracker = None
    MissingDetectionPredictor = None
    TrackingConfig = None
    TrackingResult = None
    TrajectorySmoothing = None
    convert_detection_format = None
    create_integrated_tracker = None

from .kalman import KalmanFilter, KalmanState

# Optional optimization components
try:
    from .optimization import (
        AdaptiveParameterTuning,
        MemoryPool,
        PerformanceMetrics,
        TrackingOptimizer,
    )
except ImportError:
    AdaptiveParameterTuning = None
    MemoryPool = None
    PerformanceMetrics = None
    TrackingOptimizer = None

from .tracker import ObjectTracker, Track, TrackState

__all__ = [
    # Core tracking components
    "KalmanFilter",
    "KalmanState",
    "ObjectTracker",
    "Track",
    "TrackState",
    # Optimization components
    "TrackingOptimizer",
    "PerformanceMetrics",
    "AdaptiveParameterTuning",
    "MemoryPool",
    # Integration components
    "IntegratedTracker",
    "TrackingResult",
    "TrackingConfig",
    "TrajectorySmoothing",
    "MissingDetectionPredictor",
    # Utility functions
    "create_integrated_tracker",
    "convert_detection_format",
]
