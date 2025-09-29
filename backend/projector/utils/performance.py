"""Performance optimization utilities for trajectory rendering.

This module provides advanced performance monitoring and optimization
capabilities for the trajectory rendering system, ensuring smooth
real-time operation at target frame rates.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ...core.game_state import Vector2D
from ...core.physics.trajectory import Trajectory, TrajectoryPoint

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance optimization levels."""

    MAXIMUM_QUALITY = "maximum_quality"
    HIGH_QUALITY = "high_quality"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MAXIMUM_PERFORMANCE = "maximum_performance"


@dataclass
class PerformanceMetrics:
    """Performance metrics for trajectory rendering."""

    # Frame rate metrics
    current_fps: float = 0.0
    average_fps: float = 0.0
    min_fps: float = float("inf")
    max_fps: float = 0.0
    target_fps: float = 60.0

    # Timing metrics (in milliseconds)
    frame_time: float = 0.0
    trajectory_render_time: float = 0.0
    effects_render_time: float = 0.0
    total_render_time: float = 0.0

    # Resource usage
    active_trajectories: int = 0
    total_trajectory_points: int = 0
    active_effects: int = 0
    active_particles: int = 0
    memory_usage_mb: float = 0.0

    # Quality metrics
    rendered_segments: int = 0
    culled_segments: int = 0
    simplified_points: int = 0
    level_of_detail_active: bool = False

    # Performance state
    quality_level: PerformanceLevel = PerformanceLevel.BALANCED
    auto_optimization_active: bool = False
    frame_drops: int = 0

    def fps_percentage(self) -> float:
        """Get current FPS as percentage of target."""
        if self.target_fps <= 0:
            return 100.0
        return (self.current_fps / self.target_fps) * 100.0

    def is_performance_adequate(self, threshold: float = 90.0) -> bool:
        """Check if performance meets threshold percentage of target FPS."""
        return self.fps_percentage() >= threshold


@dataclass
class OptimizationSettings:
    """Settings for performance optimization."""

    # Auto-optimization thresholds
    enable_auto_optimization: bool = True
    fps_drop_threshold: float = 0.8  # Below 80% of target FPS
    fps_recovery_threshold: float = 0.95  # Above 95% of target FPS

    # Level of detail settings
    enable_lod: bool = True
    lod_distance_threshold: float = 100.0  # pixels
    lod_point_reduction_factor: float = 0.5

    # Trajectory simplification
    enable_trajectory_simplification: bool = True
    simplification_tolerance: float = 2.0  # pixels
    min_points_per_trajectory: int = 10

    # Effects optimization
    reduce_effects_under_load: bool = True
    max_particles_under_load: int = 50
    disable_effects_threshold: float = 0.7  # Below 70% of target FPS

    # Memory management
    enable_memory_optimization: bool = True
    max_cached_trajectories: int = 10
    cache_cleanup_interval: float = 30.0  # seconds


class TrajectorySimplifier:
    """Utility for simplifying trajectory paths to improve performance."""

    @staticmethod
    def simplify_points(
        points: list[TrajectoryPoint], tolerance: float = 2.0
    ) -> list[TrajectoryPoint]:
        """Simplify trajectory points using Douglas-Peucker algorithm.

        Args:
            points: List of trajectory points to simplify
            tolerance: Simplification tolerance in pixels

        Returns:
            Simplified list of trajectory points
        """
        if len(points) <= 2:
            return points

        # Convert to 2D points for simplification
        point_coords = [(p.position.x, p.position.y) for p in points]
        simplified_indices = TrajectorySimplifier._douglas_peucker(
            point_coords, tolerance
        )

        # Return original points at simplified indices
        return [points[i] for i in simplified_indices]

    @staticmethod
    def _douglas_peucker(
        points: list[tuple[float, float]], tolerance: float
    ) -> list[int]:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) <= 2:
            return list(range(len(points)))

        # Find the point with maximum distance from line between first and last
        max_distance = 0.0
        max_index = 0

        for i in range(1, len(points) - 1):
            distance = TrajectorySimplifier._point_to_line_distance(
                points[i], points[0], points[-1]
            )
            if distance > max_distance:
                max_distance = distance
                max_index = i

        result_indices = []

        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursively simplify left and right segments
            left_indices = TrajectorySimplifier._douglas_peucker(
                points[: max_index + 1], tolerance
            )
            right_indices = TrajectorySimplifier._douglas_peucker(
                points[max_index:], tolerance
            )

            # Adjust right indices to global coordinates
            right_indices = [i + max_index for i in right_indices]

            # Combine results (avoid duplicate at junction)
            result_indices = left_indices + right_indices[1:]
        else:
            # Keep only endpoints
            result_indices = [0, len(points) - 1]

        return result_indices

    @staticmethod
    def _point_to_line_distance(
        point: tuple[float, float],
        line_start: tuple[float, float],
        line_end: tuple[float, float],
    ) -> float:
        """Calculate perpendicular distance from point to line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate line length
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if line_length_sq == 0:
            # Line is actually a point
            return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

        # Calculate perpendicular distance
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = line_length_sq**0.5

        return numerator / denominator


class LevelOfDetailManager:
    """Manages level of detail for trajectory rendering based on distance and performance."""

    def __init__(self, viewport_width: float = 1920.0, viewport_height: float = 1080.0):
        """Initialize LOD manager.

        Args:
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.viewport_center = Vector2D(viewport_width / 2, viewport_height / 2)

    def get_lod_level(
        self, trajectory: Trajectory, camera_position: Optional[Vector2D] = None
    ) -> int:
        """Get level of detail for trajectory based on distance and importance.

        Args:
            trajectory: Trajectory to evaluate
            camera_position: Camera position (uses viewport center if None)

        Returns:
            LOD level (0 = highest detail, higher = lower detail)
        """
        if not trajectory.points:
            return 0

        camera_pos = camera_position or self.viewport_center

        # Calculate average distance from camera to trajectory
        avg_distance = self._calculate_average_distance(trajectory, camera_pos)

        # Determine LOD based on distance and trajectory importance
        if trajectory.success_probability > 0.8:
            # High success probability - keep high detail
            importance_modifier = -1
        elif trajectory.success_probability < 0.3:
            # Low success probability - can reduce detail
            importance_modifier = 1
        else:
            importance_modifier = 0

        # Distance-based LOD
        if avg_distance < 100:
            base_lod = 0
        elif avg_distance < 300:
            base_lod = 1
        elif avg_distance < 600:
            base_lod = 2
        else:
            base_lod = 3

        return max(0, base_lod + importance_modifier)

    def apply_lod(self, trajectory: Trajectory, lod_level: int) -> Trajectory:
        """Apply level of detail to trajectory.

        Args:
            trajectory: Original trajectory
            lod_level: LOD level to apply

        Returns:
            LOD-optimized trajectory
        """
        if lod_level == 0 or not trajectory.points:
            return trajectory  # No optimization needed

        # Create copy for modification
        import copy

        optimized_trajectory = copy.deepcopy(trajectory)

        # Apply point reduction based on LOD level
        reduction_factors = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
        reduction_factor = reduction_factors.get(lod_level, 0.2)

        target_points = max(10, int(len(trajectory.points) * reduction_factor))

        if len(trajectory.points) > target_points:
            # Use uniform sampling for simplicity
            step = len(trajectory.points) / target_points
            selected_indices = [int(i * step) for i in range(target_points)]

            # Always include the last point
            if selected_indices[-1] != len(trajectory.points) - 1:
                selected_indices[-1] = len(trajectory.points) - 1

            optimized_trajectory.points = [
                trajectory.points[i] for i in selected_indices
            ]

        return optimized_trajectory

    def _calculate_average_distance(
        self, trajectory: Trajectory, camera_position: Vector2D
    ) -> float:
        """Calculate average distance from camera to trajectory points."""
        if not trajectory.points:
            return 0.0

        total_distance = 0.0
        for point in trajectory.points:
            dx = point.position.x - camera_position.x
            dy = point.position.y - camera_position.y
            distance = (dx**2 + dy**2) ** 0.5
            total_distance += distance

        return total_distance / len(trajectory.points)


class PerformanceMonitor:
    """Monitors and manages trajectory rendering performance."""

    def __init__(self, target_fps: float = 60.0, history_length: int = 60):
        """Initialize performance monitor.

        Args:
            target_fps: Target frame rate
            history_length: Number of frames to keep in performance history
        """
        self.target_fps = target_fps
        self.history_length = history_length

        # Performance tracking
        self.metrics = PerformanceMetrics(target_fps=target_fps)
        self.fps_history: deque[float] = deque(maxlen=history_length)
        self.frame_time_history: deque[float] = deque(maxlen=history_length)

        # Timing
        self._frame_start_time = 0.0
        self._last_update_time = time.time()
        self._frame_count = 0

        # Optimization components
        self.settings = OptimizationSettings()
        self.simplifier = TrajectorySimplifier()
        self.lod_manager = LevelOfDetailManager()

        # Auto-optimization state
        self._optimization_level = PerformanceLevel.BALANCED
        self._consecutive_slow_frames = 0
        self._consecutive_fast_frames = 0

    def begin_frame(self) -> None:
        """Mark the beginning of a frame for timing."""
        self._frame_start_time = time.time()

    def end_frame(self) -> None:
        """Mark the end of a frame and update metrics."""
        frame_end_time = time.time()
        frame_time = frame_end_time - self._frame_start_time

        # Update frame timing
        self.metrics.frame_time = frame_time * 1000  # Convert to milliseconds
        self.frame_time_history.append(frame_time)

        # Calculate FPS
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.fps_history.append(current_fps)
            self.metrics.current_fps = current_fps

            # Update FPS statistics
            self.metrics.min_fps = min(self.metrics.min_fps, current_fps)
            self.metrics.max_fps = max(self.metrics.max_fps, current_fps)

            if self.fps_history:
                self.metrics.average_fps = sum(self.fps_history) / len(self.fps_history)

        self._frame_count += 1

        # Check for performance issues
        self._check_performance()

    def update_rendering_metrics(
        self,
        trajectory_time: float,
        effects_time: float,
        trajectories_count: int,
        effects_count: int,
        particles_count: int,
    ) -> None:
        """Update rendering-specific metrics.

        Args:
            trajectory_time: Time spent rendering trajectories (ms)
            effects_time: Time spent rendering effects (ms)
            trajectories_count: Number of active trajectories
            effects_count: Number of active effects
            particles_count: Number of active particles
        """
        self.metrics.trajectory_render_time = trajectory_time
        self.metrics.effects_render_time = effects_time
        self.metrics.total_render_time = trajectory_time + effects_time
        self.metrics.active_trajectories = trajectories_count
        self.metrics.active_effects = effects_count
        self.metrics.active_particles = particles_count

    def optimize_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Optimize trajectory based on current performance level.

        Args:
            trajectory: Original trajectory

        Returns:
            Optimized trajectory
        """
        if not self.settings.enable_auto_optimization:
            return trajectory

        optimized = trajectory

        # Apply LOD if enabled
        if self.settings.enable_lod:
            lod_level = self.lod_manager.get_lod_level(trajectory)
            optimized = self.lod_manager.apply_lod(optimized, lod_level)

        # Apply simplification if needed
        if (
            self.settings.enable_trajectory_simplification
            and self._optimization_level
            in [PerformanceLevel.PERFORMANCE, PerformanceLevel.MAXIMUM_PERFORMANCE]
        ):
            optimized.points = self.simplifier.simplify_points(
                optimized.points, self.settings.simplification_tolerance
            )

        return optimized

    def should_reduce_effects(self) -> bool:
        """Check if effects should be reduced for performance."""
        if not self.settings.reduce_effects_under_load:
            return False

        fps_percentage = self.metrics.fps_percentage()
        return fps_percentage < (self.settings.disable_effects_threshold * 100)

    def get_max_particles(self) -> int:
        """Get maximum allowed particles based on performance."""
        if self.should_reduce_effects():
            return self.settings.max_particles_under_load

        # Scale based on performance level
        base_max = 200
        multipliers = {
            PerformanceLevel.MAXIMUM_QUALITY: 2.0,
            PerformanceLevel.HIGH_QUALITY: 1.5,
            PerformanceLevel.BALANCED: 1.0,
            PerformanceLevel.PERFORMANCE: 0.5,
            PerformanceLevel.MAXIMUM_PERFORMANCE: 0.25,
        }

        return int(base_max * multipliers.get(self._optimization_level, 1.0))

    def _check_performance(self) -> None:
        """Check performance and adjust optimization level if needed."""
        if not self.settings.enable_auto_optimization:
            return

        fps_percentage = self.metrics.fps_percentage()

        # Check for performance issues
        if fps_percentage < (self.settings.fps_drop_threshold * 100):
            self._consecutive_slow_frames += 1
            self._consecutive_fast_frames = 0

            # Increase optimization after several slow frames
            if self._consecutive_slow_frames >= 10:
                self._increase_optimization()
                self._consecutive_slow_frames = 0

        elif fps_percentage > (self.settings.fps_recovery_threshold * 100):
            self._consecutive_fast_frames += 1
            self._consecutive_slow_frames = 0

            # Decrease optimization after sustained good performance
            if self._consecutive_fast_frames >= 60:  # 1 second at 60 FPS
                self._decrease_optimization()
                self._consecutive_fast_frames = 0

        # Update frame drop counter
        if fps_percentage < 90:
            self.metrics.frame_drops += 1

    def _increase_optimization(self) -> None:
        """Increase optimization level for better performance."""
        current_levels = list(PerformanceLevel)
        current_index = current_levels.index(self._optimization_level)

        if current_index < len(current_levels) - 1:
            self._optimization_level = current_levels[current_index + 1]
            self.metrics.quality_level = self._optimization_level
            self.metrics.auto_optimization_active = True

            logger.info(
                f"Performance optimization increased to {self._optimization_level.value}"
            )

    def _decrease_optimization(self) -> None:
        """Decrease optimization level for better quality."""
        current_levels = list(PerformanceLevel)
        current_index = current_levels.index(self._optimization_level)

        if current_index > 0:
            self._optimization_level = current_levels[current_index - 1]
            self.metrics.quality_level = self._optimization_level

            logger.info(
                f"Performance optimization decreased to {self._optimization_level.value}"
            )

            # Turn off auto-optimization flag if back to balanced or better
            if self._optimization_level in [
                PerformanceLevel.BALANCED,
                PerformanceLevel.HIGH_QUALITY,
                PerformanceLevel.MAXIMUM_QUALITY,
            ]:
                self.metrics.auto_optimization_active = False

    def get_performance_report(self) -> dict:
        """Get comprehensive performance report."""
        return {
            "fps": {
                "current": self.metrics.current_fps,
                "average": self.metrics.average_fps,
                "min": self.metrics.min_fps,
                "max": self.metrics.max_fps,
                "target": self.metrics.target_fps,
                "percentage": self.metrics.fps_percentage(),
            },
            "timing": {
                "frame_time_ms": self.metrics.frame_time,
                "trajectory_render_ms": self.metrics.trajectory_render_time,
                "effects_render_ms": self.metrics.effects_render_time,
                "total_render_ms": self.metrics.total_render_time,
            },
            "resources": {
                "active_trajectories": self.metrics.active_trajectories,
                "active_effects": self.metrics.active_effects,
                "active_particles": self.metrics.active_particles,
                "memory_mb": self.metrics.memory_usage_mb,
            },
            "optimization": {
                "level": self._optimization_level.value,
                "auto_active": self.metrics.auto_optimization_active,
                "frame_drops": self.metrics.frame_drops,
                "lod_active": self.metrics.level_of_detail_active,
            },
            "quality": {
                "rendered_segments": self.metrics.rendered_segments,
                "culled_segments": self.metrics.culled_segments,
                "simplified_points": self.metrics.simplified_points,
            },
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.metrics = PerformanceMetrics(target_fps=self.target_fps)
        self.fps_history.clear()
        self.frame_time_history.clear()
        self._frame_count = 0
        self._consecutive_slow_frames = 0
        self._consecutive_fast_frames = 0
