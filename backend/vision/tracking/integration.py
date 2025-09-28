"""Integration module for tracking with detection systems."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .optimization import AdaptiveParameterTuning, PerformanceMetrics, TrackingOptimizer
from .tracker import ObjectTracker, TrackState

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Complete tracking system configuration."""

    # Core tracking parameters
    max_age: int = 30
    min_hits: int = 3
    max_distance: float = 50.0
    process_noise: float = 1.0
    measurement_noise: float = 10.0

    # Performance optimization
    enable_optimization: bool = True
    parallel_processing: bool = True
    max_threads: int = 4
    memory_limit_mb: int = 512

    # Adaptive tuning
    enable_adaptive_tuning: bool = True
    tuning_window: int = 50

    # Integration settings
    smooth_trajectories: bool = True
    predict_missing_detections: bool = True
    interpolate_positions: bool = True
    confidence_threshold: float = 0.3

    # Debug and monitoring
    debug_mode: bool = False
    performance_monitoring: bool = True
    save_trajectories: bool = False


class IntegratedTracker:
    """Integrated tracking system that combines detection and tracking modules.

    Features:
    - Seamless integration with detection modules
    - Performance optimization
    - Adaptive parameter tuning
    - Smooth trajectory generation
    - Missing detection prediction
    - Comprehensive monitoring
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize integrated tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = TrackingConfig(**config)

        # Core tracker
        tracker_config = {
            "max_age": self.config.max_age,
            "min_hits": self.config.min_hits,
            "max_distance": self.config.max_distance,
            "process_noise": self.config.process_noise,
            "measurement_noise": self.config.measurement_noise,
        }
        self.tracker = ObjectTracker(tracker_config)

        # Performance optimization
        if self.config.enable_optimization:
            optimizer_config = {
                "parallel_processing": self.config.parallel_processing,
                "max_threads": self.config.max_threads,
                "memory_limit_mb": self.config.memory_limit_mb,
                "adaptive_algorithms": True,
            }
            self.optimizer = TrackingOptimizer(optimizer_config)
        else:
            self.optimizer = None

        # Adaptive parameter tuning
        if self.config.enable_adaptive_tuning:
            self.adaptive_tuner = AdaptiveParameterTuning()
        else:
            self.adaptive_tuner = None

        # State tracking
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.processing_times = []
        self.track_quality_history = {}

        # Trajectory smoothing
        self.trajectory_smoother = (
            TrajectorySmoothing() if self.config.smooth_trajectories else None
        )

        # Missing detection predictor
        self.missing_predictor = (
            MissingDetectionPredictor()
            if self.config.predict_missing_detections
            else None
        )

        logger.info("Integrated tracker initialized successfully")

    def process_frame(
        self,
        detections: list[Any],
        frame_number: int,
        timestamp: Optional[float] = None,
    ) -> "TrackingResult":
        """Process a frame with detections and return tracking results.

        Args:
            detections: List of detections from detection module
            frame_number: Frame sequence number
            timestamp: Frame timestamp

        Returns:
            Comprehensive tracking results
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = time.time()

        # Convert detection format if needed
        converted_detections = self._convert_detections(detections)

        # Predict missing detections if enabled
        if self.missing_predictor and self.config.predict_missing_detections:
            predicted_detections = self.missing_predictor.predict_missing(
                self.tracker.tracks, converted_detections
            )
            converted_detections.extend(predicted_detections)

        # Apply performance optimizations
        if self.optimizer:
            # Optimize prediction phase
            self.optimizer.optimize_prediction_phase(
                self.tracker.tracks, timestamp - self.last_frame_time
            )

            # Optimize memory usage
            self.optimizer.optimize_memory_usage(self.tracker.tracks)

        # Perform tracking update
        tracked_objects = self.tracker.update_tracking(
            converted_detections, frame_number, timestamp
        )

        # Apply trajectory smoothing
        if self.trajectory_smoother:
            smoothed_objects = self.trajectory_smoother.smooth_trajectories(
                tracked_objects, self.tracker.tracks
            )
        else:
            smoothed_objects = tracked_objects

        # Interpolate positions if enabled
        if self.config.interpolate_positions:
            interpolated_objects = self._interpolate_positions(smoothed_objects)
        else:
            interpolated_objects = smoothed_objects

        # Calculate performance metrics
        processing_time = time.time() - start_time
        metrics = self._calculate_metrics(processing_time, len(converted_detections))

        # Update adaptive parameters
        if self.adaptive_tuner:
            self.adaptive_tuner.update_performance(metrics)
            updated_params = self.adaptive_tuner.get_current_parameters()
            self._update_tracker_parameters(updated_params)

        # Monitor performance
        if self.optimizer and self.config.performance_monitoring:
            self.optimizer.monitor_performance(
                len(self.tracker.tracks), len(converted_detections), processing_time
            )

        # Create comprehensive result
        result = TrackingResult(
            frame_number=frame_number,
            timestamp=timestamp,
            tracked_objects=interpolated_objects,
            tracking_statistics=self.tracker.get_tracking_statistics(),
            performance_metrics=metrics,
            track_trajectories=self.tracker.get_track_trajectories(),
            predictions=self.tracker.predict_positions(
                1 / 30.0
            ),  # Predict 1 frame ahead
            velocities=self.tracker.get_object_velocities(),
            accelerations=self.tracker.get_object_accelerations(),
        )

        # Update state
        self.frame_count += 1
        self.last_frame_time = timestamp
        self.processing_times.append(processing_time)

        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]

        return result

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "frames_processed": self.frame_count,
            "average_processing_time": np.mean(self.processing_times)
            if self.processing_times
            else 0,
            "current_fps": 1.0 / self.processing_times[-1]
            if self.processing_times
            else 0,
            "tracking_statistics": self.tracker.get_tracking_statistics(),
        }

        if self.optimizer:
            summary.update(self.optimizer.get_performance_summary())

        if self.adaptive_tuner:
            summary[
                "adaptive_parameters"
            ] = self.adaptive_tuner.get_current_parameters()

        return summary

    def reset(self) -> None:
        """Reset tracking system state."""
        self.tracker.reset()
        self.frame_count = 0
        self.processing_times.clear()
        self.track_quality_history.clear()

        if self.trajectory_smoother:
            self.trajectory_smoother.reset()

        if self.missing_predictor:
            self.missing_predictor.reset()

        logger.info("Integrated tracker reset")

    def _convert_detections(self, detections: list[Any]) -> list[Any]:
        """Convert detection format to internal format."""
        # This would handle different detection module formats
        # For now, assume detections are already in correct format
        return detections

    def _interpolate_positions(self, tracked_objects: list[Any]) -> list[Any]:
        """Interpolate positions for smooth motion."""
        # Simple interpolation based on velocity
        for obj in tracked_objects:
            if hasattr(obj, "velocity") and hasattr(obj, "position"):
                # Apply velocity-based smoothing
                vx, vy = obj.velocity
                if abs(vx) > 0.1 or abs(vy) > 0.1:  # Only if moving
                    # Slight position adjustment for smoother motion
                    dt = 1 / 60.0  # Assume 60 FPS for interpolation
                    smooth_x = obj.position[0] + vx * dt * 0.1  # 10% interpolation
                    smooth_y = obj.position[1] + vy * dt * 0.1
                    obj.position = (smooth_x, smooth_y)

        return tracked_objects

    def _calculate_metrics(
        self, processing_time: float, detection_count: int
    ) -> PerformanceMetrics:
        """Calculate performance metrics for current frame."""
        metrics = PerformanceMetrics()
        metrics.total_time = processing_time
        metrics.tracks_processed = len(self.tracker.tracks)
        metrics.detections_processed = detection_count

        if processing_time > 0:
            metrics.fps = 1.0 / processing_time

        return metrics

    def _update_tracker_parameters(self, params: dict[str, Any]) -> None:
        """Update tracker parameters from adaptive tuning."""
        if "max_distance" in params:
            self.tracker.max_distance = params["max_distance"]

        # Update Kalman filter parameters for existing tracks
        for track in self.tracker.tracks:
            if hasattr(track, "kalman_filter"):
                if "process_noise" in params:
                    track.kalman_filter.Q *= (
                        params["process_noise"] / self.config.process_noise
                    )
                if "measurement_noise" in params:
                    track.kalman_filter.R *= (
                        params["measurement_noise"] / self.config.measurement_noise
                    )

        # Update config for new tracks
        if "process_noise" in params:
            self.config.process_noise = params["process_noise"]
        if "measurement_noise" in params:
            self.config.measurement_noise = params["measurement_noise"]


@dataclass
class TrackingResult:
    """Comprehensive tracking result."""

    frame_number: int
    timestamp: float
    tracked_objects: list[Any]
    tracking_statistics: dict[str, Any]
    performance_metrics: PerformanceMetrics
    track_trajectories: dict[int, list[tuple[float, float]]]
    predictions: dict[int, tuple[float, float]]
    velocities: dict[int, tuple[float, float]]
    accelerations: dict[int, tuple[float, float]]


class TrajectorySmoothing:
    """Trajectory smoothing utilities."""

    def __init__(self, window_size: int = 5):
        """Initialize trajectory smoother.

        Args:
            window_size: Size of smoothing window
        """
        self.window_size = window_size
        self.position_history = {}

    def smooth_trajectories(
        self, tracked_objects: list[Any], tracks: list[Any]
    ) -> list[Any]:
        """Apply trajectory smoothing to tracked objects.

        Args:
            tracked_objects: List of tracked objects
            tracks: List of track objects

        Returns:
            List of objects with smoothed trajectories
        """
        smoothed_objects = []

        for obj in tracked_objects:
            if hasattr(obj, "track_id"):
                track_id = obj.track_id

                # Update position history
                if track_id not in self.position_history:
                    self.position_history[track_id] = []

                self.position_history[track_id].append(obj.position)

                # Keep only recent positions
                if len(self.position_history[track_id]) > self.window_size:
                    self.position_history[track_id] = self.position_history[track_id][
                        -self.window_size :
                    ]

                # Apply smoothing
                if len(self.position_history[track_id]) >= 3:
                    smoothed_pos = self._apply_smoothing(
                        self.position_history[track_id]
                    )
                    obj.position = smoothed_pos

            smoothed_objects.append(obj)

        return smoothed_objects

    def _apply_smoothing(
        self, positions: list[tuple[float, float]]
    ) -> tuple[float, float]:
        """Apply smoothing to position sequence."""
        if len(positions) < 3:
            return positions[-1]

        # Simple moving average
        weights = np.array([0.1, 0.3, 0.6])  # Give more weight to recent positions
        positions_array = np.array(positions[-3:])
        smoothed = np.average(positions_array, axis=0, weights=weights)

        return tuple(smoothed)

    def reset(self):
        """Reset smoothing history."""
        self.position_history.clear()


class MissingDetectionPredictor:
    """Predict missing detections based on track history."""

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize missing detection predictor.

        Args:
            confidence_threshold: Minimum confidence for predicted detections
        """
        self.confidence_threshold = confidence_threshold

    def predict_missing(self, tracks: list[Any], detections: list[Any]) -> list[Any]:
        """Predict missing detections for lost tracks.

        Args:
            tracks: List of current tracks
            detections: List of current detections

        Returns:
            List of predicted detections
        """
        predicted_detections = []

        # Get positions of current detections
        detection_positions = [
            det.position for det in detections if hasattr(det, "position")
        ]

        for track in tracks:
            if (
                hasattr(track, "state")
                and track.state == TrackState.LOST
                and hasattr(track, "kalman_filter")
            ):
                predicted_pos = track.kalman_filter.get_position()
                confidence = track.kalman_filter.confidence

                # Check if prediction is far from existing detections
                min_dist = float("inf")
                for det_pos in detection_positions:
                    dist = np.sqrt(
                        (predicted_pos[0] - det_pos[0]) ** 2
                        + (predicted_pos[1] - det_pos[1]) ** 2
                    )
                    min_dist = min(min_dist, dist)

                # Create predicted detection if confident and not too close to existing
                if confidence > self.confidence_threshold and (
                    min_dist > 30 or len(detection_positions) == 0
                ):
                    # Create a pseudo-detection for the lost track
                    predicted_detection = self._create_predicted_detection(track)
                    if predicted_detection:
                        predicted_detections.append(predicted_detection)

        return predicted_detections

    def _create_predicted_detection(self, track: Any) -> Optional[Any]:
        """Create a predicted detection from track."""
        try:
            # This would create a detection object compatible with the tracking system
            # For now, return None as the exact format depends on the detection module
            return None
        except Exception as e:
            logger.warning(f"Failed to create predicted detection: {e}")
            return None

    def reset(self):
        """Reset predictor state."""
        pass


# Factory function for easy initialization
def create_integrated_tracker(
    config: Optional[dict[str, Any]] = None
) -> IntegratedTracker:
    """Create an integrated tracker with default or custom configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured IntegratedTracker instance
    """
    if config is None:
        config = {}

    # Set sensible defaults
    default_config = {
        "max_age": 30,
        "min_hits": 3,
        "max_distance": 50.0,
        "process_noise": 1.0,
        "measurement_noise": 10.0,
        "enable_optimization": True,
        "parallel_processing": True,
        "max_threads": 4,
        "enable_adaptive_tuning": True,
        "smooth_trajectories": True,
        "predict_missing_detections": True,
        "confidence_threshold": 0.3,
        "performance_monitoring": True,
    }

    # Merge with user config
    final_config = {**default_config, **config}

    return IntegratedTracker(final_config)


# Integration utilities
def convert_detection_format(detections: list[Any], source_format: str) -> list[Any]:
    """Convert detections from various formats to internal format.

    Args:
        detections: List of detections
        source_format: Source format ('opencv', 'yolo', 'custom', etc.)

    Returns:
        List of converted detections
    """
    if source_format == "opencv":
        return _convert_from_opencv(detections)
    elif source_format == "yolo":
        return _convert_from_yolo(detections)
    elif source_format == "custom":
        return _convert_from_custom(detections)
    else:
        return detections  # Assume already in correct format


def _convert_from_opencv(detections: list[Any]) -> list[Any]:
    """Convert from OpenCV detection format."""
    # Placeholder for OpenCV format conversion
    return detections


def _convert_from_yolo(detections: list[Any]) -> list[Any]:
    """Convert from YOLO detection format."""
    # Placeholder for YOLO format conversion
    return detections


def _convert_from_custom(detections: list[Any]) -> list[Any]:
    """Convert from custom detection format."""
    # Placeholder for custom format conversion
    return detections
