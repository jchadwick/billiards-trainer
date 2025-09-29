"""Performance optimization utilities for tracking system."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking system."""

    frame_processing_time: float = 0.0
    prediction_time: float = 0.0
    association_time: float = 0.0
    update_time: float = 0.0
    total_time: float = 0.0
    tracks_processed: int = 0
    detections_processed: int = 0
    fps: float = 0.0
    memory_usage_mb: float = 0.0


class TrackingOptimizer:
    """Performance optimization utilities for the tracking system.

    Features:
    - Parallel processing of track predictions
    - Efficient cost matrix computation
    - Memory optimization
    - Performance monitoring
    - Adaptive algorithm selection
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize tracking optimizer.

        Args:
            config: Optimization configuration
        """
        # Performance settings
        self.enable_parallel_processing = config.get("parallel_processing", True)
        self.max_threads = config.get("max_threads", 4)
        self.memory_limit_mb = config.get("memory_limit_mb", 512)
        self.adaptive_algorithms = config.get("adaptive_algorithms", True)

        # Monitoring
        self.performance_history = deque(maxlen=100)
        self.metrics = PerformanceMetrics()

        # Threading
        self.thread_pool = (
            ThreadPoolExecutor(max_workers=self.max_threads)
            if self.enable_parallel_processing
            else None
        )
        self._lock = threading.Lock()

        # Caching
        self._cost_matrix_cache = {}
        self._cache_size_limit = 50

    def optimize_prediction_phase(self, tracks: list, dt: float) -> None:
        """Optimize track prediction phase using parallel processing.

        Args:
            tracks: List of tracks to predict
            dt: Time delta for prediction
        """
        start_time = time.time()

        if self.enable_parallel_processing and len(tracks) > 10:
            # Use parallel processing for large number of tracks
            self._parallel_prediction(tracks, dt)
        else:
            # Sequential processing for small number of tracks
            self._sequential_prediction(tracks, dt)

        self.metrics.prediction_time = time.time() - start_time

    def optimize_association_phase(
        self, tracks: list, detections: list
    ) -> NDArray[np.float64]:
        """Optimize detection-track association using efficient algorithms.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            Optimized cost matrix
        """
        start_time = time.time()

        # Check cache first
        cache_key = (len(tracks), len(detections))
        if (
            cache_key in self._cost_matrix_cache
            and len(self._cost_matrix_cache) < self._cache_size_limit
        ):
            base_matrix = self._cost_matrix_cache[cache_key]
        else:
            base_matrix = np.zeros((len(tracks), len(detections)))
            self._cost_matrix_cache[cache_key] = base_matrix

        # Use vectorized operations for distance calculation
        cost_matrix = self._vectorized_cost_computation(tracks, detections)

        self.metrics.association_time = time.time() - start_time
        return cost_matrix

    def optimize_memory_usage(self, tracks: list) -> None:
        """Optimize memory usage by cleaning up old data.

        Args:
            tracks: List of tracks to optimize
        """
        # Clean up track histories that are too long
        max_history_length = 50

        for track in tracks:
            if (
                hasattr(track, "position_history")
                and len(track.position_history) > max_history_length
            ):
                # Keep only recent history
                track.position_history = deque(
                    list(track.position_history)[-max_history_length:],
                    maxlen=max_history_length,
                )

            if (
                hasattr(track, "confidence_history")
                and len(track.confidence_history) > 20
            ):
                track.confidence_history = deque(
                    list(track.confidence_history)[-20:], maxlen=20
                )

        # Clean up cache if it gets too large
        if len(self._cost_matrix_cache) > self._cache_size_limit:
            # Remove oldest entries
            keys_to_remove = list(self._cost_matrix_cache.keys())[
                : -self._cache_size_limit // 2
            ]
            for key in keys_to_remove:
                del self._cost_matrix_cache[key]

    def adaptive_algorithm_selection(self, num_tracks: int, num_detections: int) -> str:
        """Select optimal algorithm based on problem size.

        Args:
            num_tracks: Number of tracks
            num_detections: Number of detections

        Returns:
            Recommended algorithm ('hungarian', 'greedy', 'auction')
        """
        total_associations = num_tracks * num_detections

        if total_associations < 50:
            return "hungarian"  # Optimal for small problems
        elif total_associations < 500:
            return "greedy"  # Good balance for medium problems
        else:
            return "auction"  # Efficient for large problems

    def monitor_performance(
        self, tracks_count: int, detections_count: int, frame_time: float
    ) -> PerformanceMetrics:
        """Monitor and update performance metrics.

        Args:
            tracks_count: Number of tracks processed
            detections_count: Number of detections processed
            frame_time: Total frame processing time

        Returns:
            Updated performance metrics
        """
        with self._lock:
            self.metrics.tracks_processed = tracks_count
            self.metrics.detections_processed = detections_count
            self.metrics.total_time = frame_time

            if frame_time > 0:
                self.metrics.fps = 1.0 / frame_time

            # Estimate memory usage
            self.metrics.memory_usage_mb = self._estimate_memory_usage()

            # Add to history
            self.performance_history.append(
                {
                    "timestamp": time.time(),
                    "fps": self.metrics.fps,
                    "total_time": frame_time,
                    "tracks": tracks_count,
                    "detections": detections_count,
                }
            )

            return self.metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Performance statistics
        """
        if not self.performance_history:
            return {}

        history_data = list(self.performance_history)

        fps_values = [entry["fps"] for entry in history_data if entry["fps"] > 0]
        time_values = [entry["total_time"] for entry in history_data]

        return {
            "current_fps": self.metrics.fps,
            "average_fps": np.mean(fps_values) if fps_values else 0,
            "min_fps": np.min(fps_values) if fps_values else 0,
            "max_fps": np.max(fps_values) if fps_values else 0,
            "average_processing_time": np.mean(time_values) if time_values else 0,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "parallel_processing_enabled": self.enable_parallel_processing,
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _parallel_prediction(self, tracks: list, dt: float) -> None:
        """Perform track predictions in parallel."""

        def predict_track(track):
            if hasattr(track, "predict") and track.is_valid():
                track.predict(dt)

        # Submit prediction tasks to thread pool
        futures = [self.thread_pool.submit(predict_track, track) for track in tracks]

        # Wait for all predictions to complete
        for future in futures:
            future.result()

    def _sequential_prediction(self, tracks: list, dt: float) -> None:
        """Perform track predictions sequentially."""
        for track in tracks:
            if hasattr(track, "predict") and track.is_valid():
                track.predict(dt)

    def _vectorized_cost_computation(
        self, tracks: list, detections: list
    ) -> NDArray[np.float64]:
        """Compute cost matrix using vectorized operations.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            Cost matrix
        """
        if not tracks or not detections:
            return np.array([])

        # Extract positions
        track_positions = np.array(
            [
                (
                    track.kalman_filter.get_position()
                    if hasattr(track, "kalman_filter")
                    else (0, 0)
                )
                for track in tracks
            ]
        )

        detection_positions = np.array(
            [det.position if hasattr(det, "position") else (0, 0) for det in detections]
        )

        # Vectorized distance computation
        # track_positions: (n_tracks, 2)
        # detection_positions: (n_detections, 2)

        # Expand dimensions for broadcasting
        track_pos_expanded = track_positions[:, np.newaxis, :]  # (n_tracks, 1, 2)
        det_pos_expanded = detection_positions[np.newaxis, :, :]  # (1, n_detections, 2)

        # Compute squared distances
        diff = track_pos_expanded - det_pos_expanded  # (n_tracks, n_detections, 2)
        distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_tracks, n_detections)

        return distances

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simple estimation based on cache size and metrics
        cache_size = len(self._cost_matrix_cache) * 8  # Rough estimate
        history_size = len(self.performance_history) * 0.1

        return (cache_size + history_size) / 1024.0  # Convert to MB

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # This would need more sophisticated tracking in a real implementation
        # For now, return a placeholder
        return 0.8 if self._cost_matrix_cache else 0.0

    def _get_optimization_recommendations(self) -> list[str]:
        """Get performance optimization recommendations."""
        recommendations = []

        if self.metrics.fps < 15:
            recommendations.append("Consider reducing number of tracks or detections")

        if self.metrics.memory_usage_mb > self.memory_limit_mb * 0.8:
            recommendations.append(
                "Memory usage is high, consider reducing history lengths"
            )

        if not self.enable_parallel_processing and self.metrics.tracks_processed > 20:
            recommendations.append("Enable parallel processing for better performance")

        if self.metrics.association_time > self.metrics.total_time * 0.5:
            recommendations.append(
                "Association phase is bottleneck, consider greedy algorithm"
            )

        return recommendations

    def __del__(self) -> None:
        """Cleanup thread pool on deletion."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class MemoryPool:
    """Memory pool for reusing numpy arrays to reduce allocation overhead."""

    def __init__(self, max_arrays: int = 100) -> None:
        """Initialize memory pool.

        Args:
            max_arrays: Maximum number of arrays to keep in pool
        """
        self.max_arrays = max_arrays
        self.arrays_2d = {}  # size -> list of arrays
        self.arrays_1d = {}  # size -> list of arrays
        self._lock = threading.Lock()

    def get_array_2d(
        self, shape: tuple[int, int], dtype=np.float64
    ) -> NDArray[np.float64]:
        """Get a 2D array from pool or create new one.

        Args:
            shape: Array shape
            dtype: Array data type

        Returns:
            Numpy array
        """
        with self._lock:
            size_key = (shape, dtype)

            if size_key in self.arrays_2d and self.arrays_2d[size_key]:
                array = self.arrays_2d[size_key].pop()
                array.fill(0)  # Clear previous values
                return array
            else:
                return np.zeros(shape, dtype=dtype)

    def return_array_2d(self, array: NDArray[np.float64]) -> None:
        """Return array to pool for reuse.

        Args:
            array: Array to return to pool
        """
        with self._lock:
            size_key = (array.shape, array.dtype)

            if size_key not in self.arrays_2d:
                self.arrays_2d[size_key] = []

            if len(self.arrays_2d[size_key]) < self.max_arrays:
                self.arrays_2d[size_key].append(array)

    def get_array_1d(self, size: int, dtype=np.float64) -> NDArray[np.float64]:
        """Get a 1D array from pool or create new one."""
        with self._lock:
            size_key = (size, dtype)

            if size_key in self.arrays_1d and self.arrays_1d[size_key]:
                array = self.arrays_1d[size_key].pop()
                array.fill(0)
                return array
            else:
                return np.zeros(size, dtype=dtype)

    def return_array_1d(self, array: NDArray[np.float64]) -> None:
        """Return 1D array to pool for reuse."""
        with self._lock:
            size_key = (array.shape[0], array.dtype)

            if size_key not in self.arrays_1d:
                self.arrays_1d[size_key] = []

            if len(self.arrays_1d[size_key]) < self.max_arrays:
                self.arrays_1d[size_key].append(array)

    def clear(self) -> None:
        """Clear all arrays from pool."""
        with self._lock:
            self.arrays_2d.clear()
            self.arrays_1d.clear()


class AdaptiveParameterTuning:
    """Adaptive parameter tuning for tracking algorithms based on performance."""

    def __init__(self) -> None:
        """Initialize adaptive parameter tuning."""
        self.performance_history = deque(maxlen=50)
        self.parameter_history = deque(maxlen=50)

        # Default parameters
        self.current_params = {
            "max_distance": 50.0,
            "process_noise": 1.0,
            "measurement_noise": 10.0,
            "min_hits": 3,
            "max_age": 30,
        }

        # Adaptation ranges
        self.param_ranges = {
            "max_distance": (20.0, 100.0),
            "process_noise": (0.1, 10.0),
            "measurement_noise": (1.0, 50.0),
            "min_hits": (1, 10),
            "max_age": (10, 100),
        }

    def update_performance(self, metrics: PerformanceMetrics) -> None:
        """Update performance history and adapt parameters.

        Args:
            metrics: Current performance metrics
        """
        self.performance_history.append(
            {
                "fps": metrics.fps,
                "tracks": metrics.tracks_processed,
                "detections": metrics.detections_processed,
                "accuracy": self._estimate_accuracy(metrics),
            }
        )

        self.parameter_history.append(self.current_params.copy())

        # Adapt parameters if we have enough history
        if len(self.performance_history) >= 10:
            self._adapt_parameters()

    def get_current_parameters(self) -> dict[str, Any]:
        """Get current optimized parameters."""
        return self.current_params.copy()

    def _estimate_accuracy(self, metrics: PerformanceMetrics) -> float:
        """Estimate tracking accuracy from metrics."""
        # Simple heuristic based on processing consistency
        if metrics.tracks_processed == 0:
            return 0.0

        # Assume higher FPS with stable track count indicates better performance
        base_accuracy = min(metrics.fps / 30.0, 1.0)  # Normalize to 30 FPS

        return base_accuracy

    def _adapt_parameters(self) -> None:
        """Adapt parameters based on performance history."""
        recent_performance = list(self.performance_history)[-5:]
        avg_fps = np.mean([p["fps"] for p in recent_performance])
        avg_accuracy = np.mean([p["accuracy"] for p in recent_performance])

        # Simple adaptation rules
        if avg_fps < 15:  # Low FPS
            # Reduce computational load
            self.current_params["max_distance"] *= 0.9
            self.current_params["max_age"] = max(
                10, int(self.current_params["max_age"] * 0.9)
            )

        elif avg_fps > 25 and avg_accuracy < 0.8:  # Good FPS but poor accuracy
            # Increase precision
            self.current_params["max_distance"] *= 1.1
            self.current_params["process_noise"] *= 0.9

        # Clamp parameters to valid ranges
        self._clamp_parameters()

    def _clamp_parameters(self) -> None:
        """Clamp parameters to valid ranges."""
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in self.current_params:
                self.current_params[param] = np.clip(
                    self.current_params[param], min_val, max_val
                )
