"""Vision utilities package.

This package provides utility functions and classes for the vision system,
including performance metrics, coordinate transformations, and visualization tools.
"""

from .metrics import (
    VisionMetricsCollector,
    MetricValue,
    DetectionAccuracyMetrics,
    PerformanceProfile,
    PerformanceTimer,
    MetricType,
    AggregationType,
    get_metrics_collector,
    time_function,
    record_fps,
    record_detection,
    get_current_fps,
    get_system_stats,
)

from .transforms import (
    CoordinateTransformer,
    Point2D,
    Point3D,
    TransformationMatrix,
    CameraCalibration,
    CoordinateSystem,
    create_perspective_matrix,
    apply_perspective_correction,
    normalize_table_coordinates,
    denormalize_table_coordinates,
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