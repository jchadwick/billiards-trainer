"""Performance metrics utilities for vision system.

This module provides comprehensive performance monitoring and metrics collection
for the vision processing pipeline, including FPS tracking, detection accuracy,
memory usage monitoring, and latency measurements.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    DETECTION_COUNT = "detection_count"
    FPS = "fps"
    ERROR_RATE = "error_rate"


class AggregationType(Enum):
    """Types of metric aggregation."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    SUM = "sum"
    COUNT = "count"


@dataclass
class MetricValue:
    """Individual metric measurement."""

    timestamp: float
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metric value."""
        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got {type(self.value)}")


@dataclass
class PerformanceProfile:
    """Performance profiling data for vision components."""

    component_name: str
    total_execution_time: float = 0.0
    call_count: int = 0
    average_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    memory_peak: float = 0.0
    last_update: float = field(default_factory=time.time)

    def update(self, execution_time: float, memory_usage: float = 0.0) -> None:
        """Update profile with new measurement."""
        self.total_execution_time += execution_time
        self.call_count += 1
        self.average_time = self.total_execution_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.memory_peak = max(self.memory_peak, memory_usage)
        self.last_update = time.time()


@dataclass
class DetectionAccuracyMetrics:
    """Accuracy metrics for detection algorithms."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    confidence_scores: list[float] = field(default_factory=list)
    detection_times: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Calculate precision score."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall score."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        return float(np.mean(self.confidence_scores)) if self.confidence_scores else 0.0

    @property
    def average_detection_time(self) -> float:
        """Calculate average detection time."""
        return float(np.mean(self.detection_times)) if self.detection_times else 0.0


class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(
        self,
        metrics_collector: "VisionMetricsCollector",
        metric_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize performance timer.

        Args:
            metrics_collector: Metrics collector instance
            metric_name: Name of the metric to record
            metadata: Optional metadata to attach to the measurement
        """
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.metadata = metadata or {}
        self.start_time = 0.0
        self.start_memory = 0.0

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop timing and record metric."""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory

        # Record execution time
        self.metrics_collector.record_metric(
            MetricType.LATENCY,
            self.metric_name,
            execution_time * 1000,  # Convert to milliseconds
            metadata={**self.metadata, "memory_delta_mb": memory_delta},
        )

        # Record memory usage if significant change
        if abs(memory_delta) > 1.0:  # > 1MB change
            self.metrics_collector.record_metric(
                MetricType.MEMORY,
                f"{self.metric_name}_memory",
                memory_delta,
                metadata=self.metadata,
            )


class VisionMetricsCollector:
    """Comprehensive metrics collection system for vision processing.

    Features:
    - Real-time performance monitoring
    - Detection accuracy tracking
    - Memory and CPU usage monitoring
    - FPS calculation and tracking
    - Latency measurement for pipeline components
    - Aggregated statistics and reporting
    - Thread-safe metric collection
    """

    def __init__(self, max_history_size: int = 10000) -> None:
        """Initialize metrics collector.

        Args:
            max_history_size: Maximum number of historical values to keep per metric
        """
        self.max_history_size = max_history_size

        # Metric storage
        self.metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history_size)
        )
        self.performance_profiles: dict[str, PerformanceProfile] = {}
        self.detection_metrics: dict[str, DetectionAccuracyMetrics] = defaultdict(
            DetectionAccuracyMetrics
        )

        # FPS tracking
        self.frame_timestamps: deque = deque(
            maxlen=100
        )  # Last 100 frames for FPS calculation
        self.processing_times: deque = deque(maxlen=1000)  # Processing time history

        # System monitoring
        self.system_metrics_enabled = True
        self.system_monitor_interval = 5.0  # seconds
        self._system_monitor_thread: Optional[threading.Thread] = None
        self._monitoring_active = False

        # Thread safety
        self._lock = threading.RLock()

        logger.info("VisionMetricsCollector initialized")

    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop, daemon=True
        )
        self._system_monitor_thread.start()
        logger.info("Started system monitoring")

    def stop_monitoring(self) -> None:
        """Stop background system monitoring."""
        self._monitoring_active = False
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=1.0)
        logger.info("Stopped system monitoring")

    def record_metric(
        self,
        metric_type: MetricType,
        name: str,
        value: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a metric value.

        Args:
            metric_type: Type of metric
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        with self._lock:
            metric_key = f"{metric_type.value}_{name}"
            metric_value = MetricValue(
                timestamp=time.time(), value=value, metadata=metadata or {}
            )
            self.metrics[metric_key].append(metric_value)

    def record_frame_processed(self) -> None:
        """Record that a frame has been processed (for FPS calculation)."""
        with self._lock:
            current_time = time.time()
            self.frame_timestamps.append(current_time)

            # Calculate FPS if we have enough samples
            if len(self.frame_timestamps) >= 2:
                time_window = self.frame_timestamps[-1] - self.frame_timestamps[0]
                if time_window > 0:
                    fps = (len(self.frame_timestamps) - 1) / time_window
                    self.record_metric(MetricType.FPS, "vision_pipeline", fps)

    def record_processing_time(self, component: str, processing_time: float) -> None:
        """Record processing time for a component.

        Args:
            component: Component name
            processing_time: Processing time in seconds
        """
        with self._lock:
            # Update performance profile
            if component not in self.performance_profiles:
                self.performance_profiles[component] = PerformanceProfile(component)

            self.performance_profiles[component].update(processing_time)

            # Record as metric
            self.record_metric(
                MetricType.LATENCY,
                component,
                processing_time * 1000,  # Convert to milliseconds
                metadata={"component": component},
            )

    def record_detection_result(
        self,
        detector_name: str,
        true_positive: bool = False,
        false_positive: bool = False,
        true_negative: bool = False,
        false_negative: bool = False,
        confidence: Optional[float] = None,
        detection_time: Optional[float] = None,
    ) -> None:
        """Record detection accuracy result.

        Args:
            detector_name: Name of the detector
            true_positive: True if this was a true positive detection
            false_positive: True if this was a false positive detection
            true_negative: True if this was a true negative detection
            false_negative: True if this was a false negative detection
            confidence: Detection confidence score (0-1)
            detection_time: Time taken for detection in seconds
        """
        with self._lock:
            metrics = self.detection_metrics[detector_name]

            if true_positive:
                metrics.true_positives += 1
            elif false_positive:
                metrics.false_positives += 1
            elif true_negative:
                metrics.true_negatives += 1
            elif false_negative:
                metrics.false_negatives += 1

            if confidence is not None:
                metrics.confidence_scores.append(confidence)
                # Keep only recent confidence scores
                if len(metrics.confidence_scores) > 1000:
                    metrics.confidence_scores = metrics.confidence_scores[-1000:]

            if detection_time is not None:
                metrics.detection_times.append(detection_time)
                # Keep only recent detection times
                if len(metrics.detection_times) > 1000:
                    metrics.detection_times = metrics.detection_times[-1000:]

    def time_component(
        self, component_name: str, metadata: Optional[dict[str, Any]] = None
    ) -> PerformanceTimer:
        """Create a timer context manager for a component.

        Args:
            component_name: Name of the component to time
            metadata: Optional metadata to attach

        Returns:
            PerformanceTimer context manager
        """
        return PerformanceTimer(self, component_name, metadata)

    def get_metric_stats(
        self,
        metric_type: MetricType,
        name: str,
        aggregation: AggregationType = AggregationType.MEAN,
        time_window: Optional[float] = None,
    ) -> Optional[float]:
        """Get aggregated metric statistics.

        Args:
            metric_type: Type of metric
            name: Metric name
            aggregation: Type of aggregation to perform
            time_window: Time window in seconds (None for all data)

        Returns:
            Aggregated metric value or None if no data
        """
        with self._lock:
            metric_key = f"{metric_type.value}_{name}"
            if metric_key not in self.metrics:
                return None

            values = self.metrics[metric_key]
            if not values:
                return None

            # Filter by time window if specified
            if time_window is not None:
                current_time = time.time()
                cutoff_time = current_time - time_window
                values = [v for v in values if v.timestamp >= cutoff_time]

            if not values:
                return None

            # Extract numeric values
            numeric_values = [v.value for v in values]

            # Calculate aggregation
            if aggregation == AggregationType.MEAN:
                return float(np.mean(numeric_values))
            elif aggregation == AggregationType.MEDIAN:
                return float(np.median(numeric_values))
            elif aggregation == AggregationType.MIN:
                return float(np.min(numeric_values))
            elif aggregation == AggregationType.MAX:
                return float(np.max(numeric_values))
            elif aggregation == AggregationType.STD:
                return float(np.std(numeric_values))
            elif aggregation == AggregationType.PERCENTILE_95:
                return float(np.percentile(numeric_values, 95))
            elif aggregation == AggregationType.PERCENTILE_99:
                return float(np.percentile(numeric_values, 99))
            elif aggregation == AggregationType.SUM:
                return float(np.sum(numeric_values))
            elif aggregation == AggregationType.COUNT:
                return len(numeric_values)
            else:
                return None

    def get_current_fps(self, time_window: float = 5.0) -> float:
        """Get current FPS over specified time window.

        Args:
            time_window: Time window in seconds

        Returns:
            Current FPS
        """
        with self._lock:
            if len(self.frame_timestamps) < 2:
                return 0.0

            current_time = time.time()
            cutoff_time = current_time - time_window

            # Count frames in time window
            recent_frames = [t for t in self.frame_timestamps if t >= cutoff_time]

            if len(recent_frames) < 2:
                return 0.0

            time_span = recent_frames[-1] - recent_frames[0]
            if time_span <= 0:
                return 0.0

            return (len(recent_frames) - 1) / time_span

    def get_detection_metrics(
        self, detector_name: str
    ) -> Optional[DetectionAccuracyMetrics]:
        """Get detection accuracy metrics for a detector.

        Args:
            detector_name: Name of the detector

        Returns:
            Detection metrics or None if not found
        """
        with self._lock:
            return self.detection_metrics.get(detector_name)

    def get_performance_profile(
        self, component_name: str
    ) -> Optional[PerformanceProfile]:
        """Get performance profile for a component.

        Args:
            component_name: Name of the component

        Returns:
            Performance profile or None if not found
        """
        with self._lock:
            return self.performance_profiles.get(component_name)

    def get_system_metrics(self) -> dict[str, float]:
        """Get current system metrics.

        Returns:
            Dictionary of system metrics
        """
        try:
            process = psutil.Process()
            system = psutil

            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "system_cpu_percent": system.cpu_percent(),
                "system_memory_percent": system.virtual_memory().percent,
                "thread_count": process.num_threads(),
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}

    def get_comprehensive_report(
        self, time_window: Optional[float] = None
    ) -> dict[str, Any]:
        """Generate comprehensive metrics report.

        Args:
            time_window: Time window in seconds (None for all data)

        Returns:
            Comprehensive metrics report
        """
        with self._lock:
            report = {
                "timestamp": time.time(),
                "time_window": time_window,
                "fps": {
                    "current": self.get_current_fps(),
                    "average": self.get_metric_stats(
                        MetricType.FPS, "vision_pipeline", time_window=time_window
                    ),
                },
                "system": self.get_system_metrics(),
                "performance_profiles": {},
                "detection_metrics": {},
                "latency_stats": {},
                "memory_stats": {},
            }

            # Add performance profiles
            for name, profile in self.performance_profiles.items():
                report["performance_profiles"][name] = {
                    "average_time_ms": profile.average_time * 1000,
                    "min_time_ms": profile.min_time * 1000,
                    "max_time_ms": profile.max_time * 1000,
                    "call_count": profile.call_count,
                    "memory_peak_mb": profile.memory_peak,
                    "last_update": profile.last_update,
                }

            # Add detection metrics
            for name, metrics in self.detection_metrics.items():
                report["detection_metrics"][name] = {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "accuracy": metrics.accuracy,
                    "average_confidence": metrics.average_confidence,
                    "average_detection_time_ms": metrics.average_detection_time * 1000,
                    "total_detections": metrics.true_positives
                    + metrics.false_positives,
                }

            # Add latency statistics
            for metric_key in self.metrics:
                if metric_key.startswith("latency_"):
                    component = metric_key.replace("latency_", "")
                    report["latency_stats"][component] = {
                        "mean_ms": self.get_metric_stats(
                            MetricType.LATENCY,
                            component,
                            AggregationType.MEAN,
                            time_window,
                        ),
                        "median_ms": self.get_metric_stats(
                            MetricType.LATENCY,
                            component,
                            AggregationType.MEDIAN,
                            time_window,
                        ),
                        "p95_ms": self.get_metric_stats(
                            MetricType.LATENCY,
                            component,
                            AggregationType.PERCENTILE_95,
                            time_window,
                        ),
                        "p99_ms": self.get_metric_stats(
                            MetricType.LATENCY,
                            component,
                            AggregationType.PERCENTILE_99,
                            time_window,
                        ),
                        "max_ms": self.get_metric_stats(
                            MetricType.LATENCY,
                            component,
                            AggregationType.MAX,
                            time_window,
                        ),
                    }

            # Add memory statistics
            for metric_key in self.metrics:
                if metric_key.startswith("memory_"):
                    component = metric_key.replace("memory_", "")
                    report["memory_stats"][component] = {
                        "mean_mb": self.get_metric_stats(
                            MetricType.MEMORY,
                            component,
                            AggregationType.MEAN,
                            time_window,
                        ),
                        "max_mb": self.get_metric_stats(
                            MetricType.MEMORY,
                            component,
                            AggregationType.MAX,
                            time_window,
                        ),
                        "min_mb": self.get_metric_stats(
                            MetricType.MEMORY,
                            component,
                            AggregationType.MIN,
                            time_window,
                        ),
                    }

            return report

    def reset_metrics(
        self, metric_type: Optional[MetricType] = None, name: Optional[str] = None
    ) -> None:
        """Reset metrics data.

        Args:
            metric_type: Type of metrics to reset (None for all)
            name: Specific metric name to reset (None for all)
        """
        with self._lock:
            if metric_type is None and name is None:
                # Reset everything
                self.metrics.clear()
                self.performance_profiles.clear()
                self.detection_metrics.clear()
                self.frame_timestamps.clear()
                self.processing_times.clear()
            elif metric_type is not None and name is not None:
                # Reset specific metric
                metric_key = f"{metric_type.value}_{name}"
                if metric_key in self.metrics:
                    self.metrics[metric_key].clear()
            elif metric_type is not None:
                # Reset all metrics of specific type
                keys_to_remove = [
                    k for k in self.metrics if k.startswith(f"{metric_type.value}_")
                ]
                for key in keys_to_remove:
                    self.metrics[key].clear()

        logger.info(f"Reset metrics: type={metric_type}, name={name}")

    def _system_monitor_loop(self) -> None:
        """Background system monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.get_system_metrics()

                # Record each metric
                for metric_name, value in system_metrics.items():
                    if "cpu" in metric_name:
                        self.record_metric(MetricType.CPU, metric_name, value)
                    elif "memory" in metric_name:
                        self.record_metric(MetricType.MEMORY, metric_name, value)

                # Sleep until next collection
                time.sleep(self.system_monitor_interval)

            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                time.sleep(1.0)

    def __enter__(self) -> "PerformanceTimer":
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.stop_monitoring()


# Global metrics collector instance
_global_metrics_collector: Optional[VisionMetricsCollector] = None


def get_metrics_collector() -> VisionMetricsCollector:
    """Get or create global metrics collector instance.

    Returns:
        Global VisionMetricsCollector instance
    """
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = VisionMetricsCollector()
    return _global_metrics_collector


def reset_global_metrics() -> None:
    """Reset the global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is not None:
        _global_metrics_collector.reset_metrics()


# Convenience functions for common operations
def time_function(
    func: Callable[..., Any], component_name: Optional[str] = None
) -> Callable[..., Any]:
    """Decorator to time function execution.

    Args:
        func: Function to time
        component_name: Component name (uses function name if None)

    Returns:
        Decorated function
    """

    def wrapper(*args, **kwargs):
        name = component_name or func.__name__
        with get_metrics_collector().time_component(name):
            return func(*args, **kwargs)

    return wrapper


def record_fps() -> None:
    """Record frame processing for FPS calculation."""
    get_metrics_collector().record_frame_processed()


def record_detection(
    detector_name: str,
    is_correct: bool,
    confidence: Optional[float] = None,
    detection_time: Optional[float] = None,
) -> None:
    """Record detection result.

    Args:
        detector_name: Name of the detector
        is_correct: Whether the detection was correct
        confidence: Detection confidence score
        detection_time: Time taken for detection
    """
    get_metrics_collector().record_detection_result(
        detector_name=detector_name,
        true_positive=is_correct,
        false_positive=not is_correct,
        confidence=confidence,
        detection_time=detection_time,
    )


def get_current_fps() -> float:
    """Get current FPS.

    Returns:
        Current frames per second
    """
    return get_metrics_collector().get_current_fps()


def get_system_stats() -> dict[str, float]:
    """Get current system statistics.

    Returns:
        Dictionary of system metrics
    """
    return get_metrics_collector().get_system_metrics()
