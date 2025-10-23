"""Performance profiler for vision pipeline.

Provides detailed timing instrumentation for every stage of the vision processing pipeline
to identify bottlenecks and optimization opportunities.
"""

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class FrameTimings:
    """Detailed timing breakdown for a single frame."""

    frame_number: int
    total_time: float = 0.0

    # Stage timings
    preprocessing_time: float = 0.0
    masking_time: float = 0.0
    table_detection_time: float = 0.0
    ball_detection_time: float = 0.0
    cue_detection_time: float = 0.0
    tracking_time: float = 0.0
    result_building_time: float = 0.0

    # YOLO-specific
    yolo_inference_time: float = 0.0
    yolo_preprocessing_time: float = 0.0
    yolo_postprocessing_time: float = 0.0

    # Queue/threading overhead
    queue_wait_time: float = 0.0
    lock_wait_time: float = 0.0

    def get_breakdown(self) -> dict[str, float]:
        """Get timing breakdown as dictionary."""
        return {
            "total": self.total_time,
            "preprocessing": self.preprocessing_time,
            "masking": self.masking_time,
            "table_detection": self.table_detection_time,
            "ball_detection": self.ball_detection_time,
            "yolo_inference": self.yolo_inference_time,
            "yolo_preprocessing": self.yolo_preprocessing_time,
            "yolo_postprocessing": self.yolo_postprocessing_time,
            "cue_detection": self.cue_detection_time,
            "tracking": self.tracking_time,
            "result_building": self.result_building_time,
            "queue_wait": self.queue_wait_time,
            "lock_wait": self.lock_wait_time,
        }

    def get_fps(self) -> float:
        """Calculate FPS from total time."""
        return 1.0 / self.total_time if self.total_time > 0 else 0.0


@dataclass
class AggregateStats:
    """Aggregate statistics across multiple frames."""

    frame_count: int = 0

    # Average timings
    avg_total: float = 0.0
    avg_preprocessing: float = 0.0
    avg_masking: float = 0.0
    avg_table_detection: float = 0.0
    avg_ball_detection: float = 0.0
    avg_yolo_inference: float = 0.0
    avg_cue_detection: float = 0.0
    avg_tracking: float = 0.0

    # Min/max
    min_total: float = float("inf")
    max_total: float = 0.0

    # FPS
    avg_fps: float = 0.0

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics."""
        return {
            "frame_count": self.frame_count,
            "avg_fps": self.avg_fps,
            "avg_total_ms": self.avg_total * 1000,
            "min_total_ms": (
                self.min_total * 1000 if self.min_total != float("inf") else 0
            ),
            "max_total_ms": self.max_total * 1000,
            "avg_preprocessing_ms": self.avg_preprocessing * 1000,
            "avg_masking_ms": self.avg_masking * 1000,
            "avg_table_detection_ms": self.avg_table_detection * 1000,
            "avg_ball_detection_ms": self.avg_ball_detection * 1000,
            "avg_yolo_inference_ms": self.avg_yolo_inference * 1000,
            "avg_cue_detection_ms": self.avg_cue_detection * 1000,
            "avg_tracking_ms": self.avg_tracking * 1000,
        }


class PerformanceProfiler:
    """Performance profiler for vision pipeline.

    Tracks detailed timing for each stage of frame processing and provides
    aggregate statistics and bottleneck analysis.
    """

    def __init__(
        self,
        history_size: int = 100,
        log_interval: int = 30,
        enable_console_logging: bool = False,
    ):
        """Initialize profiler.

        Args:
            history_size: Number of recent frames to keep for statistics
            log_interval: Log summary every N frames
            enable_console_logging: Whether to log summaries to console (default: False, use API only)
        """
        self.history_size = history_size
        self.log_interval = log_interval
        self.enable_console_logging = enable_console_logging

        # Ring buffer for recent timings
        self.recent_timings: deque[FrameTimings] = deque(maxlen=history_size)

        # Current frame being profiled
        self.current_frame: Optional[FrameTimings] = None
        self.stage_start_time: float = 0.0

        # Aggregate stats
        self.total_frames = 0
        self.start_time = time.time()

    def start_frame(self, frame_number: int) -> None:
        """Start profiling a new frame."""
        self.current_frame = FrameTimings(frame_number=frame_number)
        self.stage_start_time = time.time()

    def start_stage(self, stage: str) -> None:
        """Start timing a stage."""
        self.stage_start_time = time.time()

    def end_stage(self, stage: str) -> None:
        """End timing a stage and record duration."""
        if not self.current_frame:
            return

        elapsed = time.time() - self.stage_start_time

        # Map stage name to attribute
        stage_attr_map = {
            "preprocessing": "preprocessing_time",
            "masking": "masking_time",
            "table_detection": "table_detection_time",
            "ball_detection": "ball_detection_time",
            "yolo_inference": "yolo_inference_time",
            "yolo_preprocessing": "yolo_preprocessing_time",
            "yolo_postprocessing": "yolo_postprocessing_time",
            "cue_detection": "cue_detection_time",
            "tracking": "tracking_time",
            "result_building": "result_building_time",
            "queue_wait": "queue_wait_time",
            "lock_wait": "lock_wait_time",
        }

        if stage in stage_attr_map:
            setattr(self.current_frame, stage_attr_map[stage], elapsed)

    def end_frame(self) -> None:
        """Finish profiling current frame and update statistics."""
        if not self.current_frame:
            return

        # Calculate total time
        self.current_frame.total_time = sum(
            [
                self.current_frame.preprocessing_time,
                self.current_frame.masking_time,
                self.current_frame.table_detection_time,
                self.current_frame.ball_detection_time,
                self.current_frame.cue_detection_time,
                self.current_frame.tracking_time,
                self.current_frame.result_building_time,
            ]
        )

        # Add to history
        self.recent_timings.append(self.current_frame)
        self.total_frames += 1

        # Log periodically (only if console logging enabled)
        if self.enable_console_logging and self.total_frames % self.log_interval == 0:
            self._log_summary()

        self.current_frame = None

    def get_current_stats(self) -> AggregateStats:
        """Get aggregate statistics from recent frames."""
        if not self.recent_timings:
            return AggregateStats()

        stats = AggregateStats()
        stats.frame_count = len(self.recent_timings)

        # Calculate averages
        total_times = [t.total_time for t in self.recent_timings]
        stats.avg_total = statistics.mean(total_times)
        stats.min_total = min(total_times)
        stats.max_total = max(total_times)
        stats.avg_fps = 1.0 / stats.avg_total if stats.avg_total > 0 else 0.0

        stats.avg_preprocessing = statistics.mean(
            [t.preprocessing_time for t in self.recent_timings]
        )
        stats.avg_masking = statistics.mean(
            [t.masking_time for t in self.recent_timings]
        )
        stats.avg_table_detection = statistics.mean(
            [t.table_detection_time for t in self.recent_timings]
        )
        stats.avg_ball_detection = statistics.mean(
            [t.ball_detection_time for t in self.recent_timings]
        )
        stats.avg_yolo_inference = statistics.mean(
            [t.yolo_inference_time for t in self.recent_timings]
        )
        stats.avg_cue_detection = statistics.mean(
            [t.cue_detection_time for t in self.recent_timings]
        )
        stats.avg_tracking = statistics.mean(
            [t.tracking_time for t in self.recent_timings]
        )

        return stats

    def get_bottlenecks(self, top_n: int = 3) -> list[tuple[str, float]]:
        """Identify the top N bottleneck stages.

        Args:
            top_n: Number of top bottlenecks to return

        Returns:
            List of (stage_name, avg_time_ms) tuples, sorted by time descending
        """
        stats = self.get_current_stats()

        stage_times = [
            ("preprocessing", stats.avg_preprocessing * 1000),
            ("masking", stats.avg_masking * 1000),
            ("table_detection", stats.avg_table_detection * 1000),
            ("ball_detection", stats.avg_ball_detection * 1000),
            ("yolo_inference", stats.avg_yolo_inference * 1000),
            ("cue_detection", stats.avg_cue_detection * 1000),
            ("tracking", stats.avg_tracking * 1000),
        ]

        # Sort by time descending
        stage_times.sort(key=lambda x: x[1], reverse=True)

        return stage_times[:top_n]

    def _log_summary(self) -> None:
        """Log performance summary."""
        stats = self.get_current_stats()
        summary = stats.get_summary()

        logger.info(
            f"\n"
            f"=== Performance Summary (last {stats.frame_count} frames) ===\n"
            f"FPS: {summary['avg_fps']:.1f} (total: {summary['avg_total_ms']:.1f}ms, "
            f"min: {summary['min_total_ms']:.1f}ms, max: {summary['max_total_ms']:.1f}ms)\n"
            f"Breakdown:\n"
            f"  Preprocessing:    {summary['avg_preprocessing_ms']:6.1f}ms\n"
            f"  Masking:          {summary['avg_masking_ms']:6.1f}ms\n"
            f"  Table Detection:  {summary['avg_table_detection_ms']:6.1f}ms\n"
            f"  Ball Detection:   {summary['avg_ball_detection_ms']:6.1f}ms\n"
            f"    (YOLO only:     {summary['avg_yolo_inference_ms']:6.1f}ms)\n"
            f"  Cue Detection:    {summary['avg_cue_detection_ms']:6.1f}ms\n"
            f"  Tracking:         {summary['avg_tracking_ms']:6.1f}ms\n"
        )

        # Show top bottlenecks
        bottlenecks = self.get_bottlenecks(top_n=3)
        logger.info("Top bottlenecks:")
        for i, (stage, time_ms) in enumerate(bottlenecks, 1):
            pct = (
                (time_ms / summary["avg_total_ms"] * 100)
                if summary["avg_total_ms"] > 0
                else 0
            )
            logger.info(f"  {i}. {stage:20s}: {time_ms:6.1f}ms ({pct:5.1f}%)")

    def get_realtime_status(self) -> dict[str, any]:
        """Get real-time performance status for monitoring.

        Returns:
            Dictionary with current performance metrics including CPU and memory
        """
        stats = self.get_current_stats()
        bottlenecks = self.get_bottlenecks(top_n=5)

        target_fps = 15.0
        target_frame_time = 1000.0 / target_fps  # ms

        # Get CPU and memory usage
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.0)  # Non-blocking
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

        # Determine performance status
        status = "good"
        if stats.avg_fps < 10 or cpu_percent > 80 or memory_mb > 2048:
            status = "poor"
        elif stats.avg_fps < 15 or cpu_percent > 60 or memory_mb > 1024:
            status = "degraded"

        return {
            "fps": stats.avg_fps,
            "target_fps": target_fps,
            "meeting_target": stats.avg_fps >= target_fps,
            "frame_time_ms": stats.avg_total * 1000,
            "target_frame_time_ms": target_frame_time,
            "overhead_ms": max(0, stats.avg_total * 1000 - target_frame_time),
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "status": status,
            "bottlenecks": [
                {"stage": stage, "avg_time_ms": time_ms}
                for stage, time_ms in bottlenecks
            ],
            "frame_count": stats.frame_count,
            "total_frames": self.total_frames,
            "uptime_seconds": time.time() - self.start_time,
        }


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_profiler(
    enabled: bool = True,
    history_size: int = 100,
    log_interval: int = 30,
    enable_console_logging: bool = False,
) -> Optional[PerformanceProfiler]:
    """Get or create global profiler instance.

    Args:
        enabled: Whether profiling is enabled
        history_size: Number of frames to keep in history
        log_interval: Log summary every N frames
        enable_console_logging: Whether to log summaries to console

    Returns:
        Profiler instance or None if disabled
    """
    global _profiler

    if not enabled:
        return None

    if _profiler is None:
        _profiler = PerformanceProfiler(
            history_size=history_size,
            log_interval=log_interval,
            enable_console_logging=enable_console_logging,
        )
        if enable_console_logging:
            logger.info("Performance profiler initialized with console logging")
        else:
            logger.debug("Performance profiler initialized (API-only mode)")

    return _profiler


def reset_profiler() -> None:
    """Reset global profiler instance."""
    global _profiler
    _profiler = None
