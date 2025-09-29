"""Vision utilities package.

This package provides utility functions and classes for the vision system,
including performance metrics, coordinate transformations, and visualization tools.
"""

from .metrics import (
    AggregationType,
    DetectionAccuracyMetrics,
    MetricType,
    MetricValue,
    PerformanceProfile,
    PerformanceTimer,
    VisionMetricsCollector,
    get_current_fps,
    get_metrics_collector,
    get_system_stats,
    record_detection,
    record_fps,
    time_function,
)
from .transforms import (
    CameraCalibration,
    CoordinateSystem,
    CoordinateTransformer,
    Point2D,
    Point3D,
    TransformationMatrix,
    apply_perspective_correction,
    create_perspective_matrix,
    denormalize_table_coordinates,
    normalize_table_coordinates,
)
from .visualization import TableVisualization

__all__ = [
    # Metrics
    "VisionMetricsCollector",
    "MetricValue",
    "DetectionAccuracyMetrics",
    "PerformanceProfile",
    "PerformanceTimer",
    "MetricType",
    "AggregationType",
    "get_metrics_collector",
    "time_function",
    "record_fps",
    "record_detection",
    "get_current_fps",
    "get_system_stats",
    # Transforms
    "CoordinateTransformer",
    "Point2D",
    "Point3D",
    "TransformationMatrix",
    "CameraCalibration",
    "CoordinateSystem",
    "create_perspective_matrix",
    "apply_perspective_correction",
    "normalize_table_coordinates",
    "denormalize_table_coordinates",
    # Visualization
    "TableVisualization",
]
