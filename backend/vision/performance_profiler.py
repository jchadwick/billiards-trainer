"""Opt-in, bounded timing instrumentation for the vision pipeline.

Frame and stage timings use monotonic clocks. Stage spans are thread-local and
can be nested, while completed frames are retained in a bounded ring buffer.
Inclusive stage time is useful for understanding a subsystem; exclusive time
avoids double-counting nested work when ranking bottlenecks.
"""

from __future__ import annotations

import importlib
import logging
import math
import statistics
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, cast


class _MemoryInfo(Protocol):
    rss: int


class _Process(Protocol):
    def cpu_percent(self, interval: float | None = None) -> float: ...

    def memory_info(self) -> _MemoryInfo: ...


class _Psutil(Protocol):
    Process: Callable[[], _Process]


psutil = cast("_Psutil", importlib.import_module("psutil"))
logger = logging.getLogger(__name__)

_STAGE_ATTRIBUTES = {
    "masking": "masking_time",
    "ball_detection": "ball_detection_time",
    "yolo_inference": "yolo_inference_time",
    "detector_inference": "yolo_inference_time",
    "yolo_preprocessing": "yolo_preprocessing_time",
    "preprocessing": "yolo_preprocessing_time",
    "yolo_postprocessing": "yolo_postprocessing_time",
    "postprocessing": "yolo_postprocessing_time",
    "cue_detection": "cue_detection_time",
    "tracking": "tracking_time",
    "result_building": "result_building_time",
    "queue_wait": "queue_wait_time",
    "lock_wait": "lock_wait_time",
}


@dataclass
class _OpenSpan:
    name: str
    started_ns: int
    child_ns: int = 0


@dataclass
class FrameTimings:
    """Timing data for one frame, with wall time and exclusive stage costs."""

    frame_number: int
    started_ns: int = field(default_factory=time.perf_counter_ns)
    total_time: float = 0.0
    inclusive_stages: dict[str, float] = field(default_factory=dict)
    exclusive_stages: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    _open_spans: list[_OpenSpan] = field(default_factory=list, repr=False)

    # Compatibility attributes used by the existing status API.
    masking_time: float = 0.0
    ball_detection_time: float = 0.0
    cue_detection_time: float = 0.0
    tracking_time: float = 0.0
    result_building_time: float = 0.0
    yolo_inference_time: float = 0.0
    yolo_preprocessing_time: float = 0.0
    yolo_postprocessing_time: float = 0.0
    queue_wait_time: float = 0.0
    lock_wait_time: float = 0.0

    def get_breakdown(self) -> dict[str, float]:
        """Return wall time and inclusive stage timings in seconds."""
        return {"total": self.total_time, **self.inclusive_stages}

    def get_fps(self) -> float:
        """Return reciprocal service time for legacy callers."""
        return 1.0 / self.total_time if self.total_time > 0 else 0.0


@dataclass
class AggregateStats:
    """Summary of the recent bounded frame window."""

    frame_count: int = 0
    avg_total: float = 0.0
    avg_masking: float | None = None
    avg_ball_detection: float | None = None
    avg_yolo_inference: float | None = None
    avg_cue_detection: float | None = None
    avg_tracking: float | None = None
    min_total: float = float("inf")
    max_total: float = 0.0
    avg_fps: float = 0.0
    percentiles_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    stage_exclusive_ms: dict[str, float] = field(default_factory=dict)
    observed_fps: float | None = None

    def get_summary(self) -> dict[str, Any]:
        """Return compatible aggregate keys plus defensible distribution data."""

        def optional_ms(value: float | None) -> float | None:
            return value * 1000 if value is not None else None

        return {
            "frame_count": self.frame_count,
            "avg_fps": self.avg_fps,
            "observed_processing_fps": self.observed_fps,
            "avg_total_ms": self.avg_total * 1000,
            "min_total_ms": self.min_total * 1000 if self.frame_count else 0.0,
            "max_total_ms": self.max_total * 1000,
            "avg_masking_ms": optional_ms(self.avg_masking),
            "avg_ball_detection_ms": optional_ms(self.avg_ball_detection),
            "avg_yolo_inference_ms": optional_ms(self.avg_yolo_inference),
            "avg_cue_detection_ms": optional_ms(self.avg_cue_detection),
            "avg_tracking_ms": optional_ms(self.avg_tracking),
            "latency_percentiles_ms": self.percentiles_ms,
            "exclusive_stage_mean_ms": self.stage_exclusive_ms,
        }


class PerformanceProfiler:
    """Thread-safe, opt-in vision timing profiler with bounded storage."""

    def __init__(
        self,
        history_size: int = 100,
        log_interval: int = 30,
        enable_console_logging: bool = False,
        resource_sample_interval_seconds: float = 1.0,
        target_fps: float | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        event_sink: Callable[[dict[str, Any]], object] | None = None,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if log_interval < 1:
            raise ValueError("log_interval must be positive")
        if resource_sample_interval_seconds <= 0:
            raise ValueError("resource sample interval must be positive")
        self.history_size = history_size
        self.log_interval = log_interval
        self.enable_console_logging = enable_console_logging
        self.target_fps = target_fps
        self._clock_ns = clock_ns
        self._event_sink = event_sink
        self.recent_timings: deque[FrameTimings] = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._frames_by_thread: dict[int, FrameTimings] = {}
        self.total_frames = 0
        self.start_time = time.monotonic()
        self._process = psutil.Process()
        self._cpu_percent: float | None = None
        self._memory_mb: float | None = None
        self._resource_interval = resource_sample_interval_seconds
        self._sampler_stop = threading.Event()
        self._sampler = threading.Thread(
            target=self._sample_resources,
            name="vision-performance-sampler",
            daemon=True,
        )
        self._process.cpu_percent(interval=None)  # Prime psutil's interval sample.
        self._sampler.start()

    @property
    def current_frame(self) -> FrameTimings | None:
        """Return the current thread's frame for backward compatibility."""
        return getattr(self._thread_state, "current_frame", None)

    @current_frame.setter
    def current_frame(self, frame: FrameTimings | None) -> None:
        self._thread_state.current_frame = frame

    def start_frame(
        self, frame_number: int, metadata: dict[str, Any] | None = None
    ) -> None:
        """Start a frame context isolated from frames on other threads."""
        frame = FrameTimings(
            frame_number=frame_number,
            started_ns=self._clock_ns(),
            metadata=dict(metadata or {}),
        )
        self.current_frame = frame
        with self._lock:
            self._frames_by_thread[threading.get_ident()] = frame

    def start_stage(self, stage: str) -> None:
        """Start a stage span, nesting it beneath the currently open span."""
        frame = self.current_frame
        if frame is not None:
            frame._open_spans.append(_OpenSpan(stage, self._clock_ns()))

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        """Measure a nested stage and close it even when the operation fails."""
        self.start_stage(stage)
        try:
            yield
        finally:
            self.end_stage(stage)

    def end_stage(self, stage: str) -> None:
        """Close the most recent matching stage and account inclusive/exclusive time."""
        frame = self.current_frame
        if frame is None:
            return
        index = next(
            (
                i
                for i in range(len(frame._open_spans) - 1, -1, -1)
                if frame._open_spans[i].name == stage
            ),
            None,
        )
        if index is None:
            return
        # If an exception skips a nested ``finally``, close children at the
        # parent boundary so their elapsed time remains accounted once.
        while len(frame._open_spans) > index + 1:
            self.end_stage(frame._open_spans[-1].name)
        span = frame._open_spans.pop(index)
        elapsed_ns = max(0, self._clock_ns() - span.started_ns)
        inclusive = elapsed_ns / 1_000_000_000
        exclusive = max(0, elapsed_ns - span.child_ns) / 1_000_000_000
        frame.inclusive_stages[stage] = (
            frame.inclusive_stages.get(stage, 0.0) + inclusive
        )
        frame.exclusive_stages[stage] = (
            frame.exclusive_stages.get(stage, 0.0) + exclusive
        )
        if stage in _STAGE_ATTRIBUTES:
            attribute = _STAGE_ATTRIBUTES[stage]
            setattr(frame, attribute, getattr(frame, attribute) + inclusive)
        if index > 0:
            frame._open_spans[index - 1].child_ns += elapsed_ns

    def end_frame(self) -> None:
        """Finish the current frame and retain its wall-clock service time."""
        frame = self.current_frame
        if frame is None:
            return
        # Close any unbalanced span at the frame boundary. This preserves the
        # frame record while making an instrumentation bug visible in timings.
        while frame._open_spans:
            self.end_stage(frame._open_spans[-1].name)
        now_ns = self._clock_ns()
        frame.total_time = max(0, now_ns - frame.started_ns) / 1_000_000_000
        self.current_frame = None
        with self._lock:
            self._frames_by_thread.pop(threading.get_ident(), None)
            self.recent_timings.append(frame)
            self.total_frames += 1
            count = self.total_frames
        if self._event_sink is not None:
            try:
                self._event_sink(
                    {
                        "event_type": "vision_frame_complete",
                        "frame_number": frame.frame_number,
                        "source_frame_number": frame.metadata.get(
                            "source_frame_number"
                        ),
                        "source_timestamp": frame.metadata.get("source_timestamp"),
                        "source_timestamp_kind": frame.metadata.get(
                            "source_timestamp_kind"
                        ),
                        "source_process_id": frame.metadata.get("source_process_id"),
                        "calibration_revision": frame.metadata.get(
                            "calibration_revision"
                        ),
                        "map_revision": frame.metadata.get("map_revision"),
                        "phase": frame.metadata.get("phase"),
                        "frame_status": frame.metadata.get("frame_status", "complete"),
                        "component_observation_status": frame.metadata.get(
                            "component_observation_status"
                        ),
                        "ball_positions_status": frame.metadata.get(
                            "ball_positions_status"
                        ),
                        "ball_count": frame.metadata.get("ball_count"),
                        "cue_present": frame.metadata.get("cue_present"),
                        "detector_cache_hits": frame.metadata.get(
                            "detector_cache_hits", 0
                        ),
                        "frame_publication_age_seconds": frame.metadata.get(
                            "frame_publication_age_seconds"
                        ),
                        "shared_memory_read_copy_seconds": frame.metadata.get(
                            "shared_memory_read_copy_seconds"
                        ),
                        "queue_delay_seconds": frame.metadata.get(
                            "queue_delay_seconds"
                        ),
                        "processing_queue_depth": frame.metadata.get(
                            "processing_queue_depth"
                        ),
                        "service_time_seconds": frame.total_time,
                        "inclusive_stages_seconds": frame.inclusive_stages,
                        "exclusive_stages_seconds": frame.exclusive_stages,
                    }
                )
            except Exception:
                logger.debug("Performance audit event sink failed", exc_info=True)
        if self.enable_console_logging and count % self.log_interval == 0:
            self._log_summary()

    def get_current_stats(self) -> AggregateStats:
        """Aggregate recent frames without adding work to frame processing."""
        with self._lock:
            frames = list(self.recent_timings)
        if not frames:
            return AggregateStats()
        totals = [frame.total_time for frame in frames]
        names = {name for frame in frames for name in frame.inclusive_stages}
        exclusive = {
            name: statistics.fmean(
                frame.exclusive_stages.get(name, 0.0) for frame in frames
            )
            * 1000
            for name in names
        }
        percentiles = {
            "total": _percentiles(totals),
            **{
                name: _percentiles(
                    [frame.inclusive_stages.get(name, 0.0) for frame in frames]
                )
                for name in names
            },
        }
        observed_fps = None
        if len(frames) > 1:
            frames.sort(key=lambda frame: frame.started_ns)
            elapsed = (frames[-1].started_ns - frames[0].started_ns) / 1e9
            if elapsed > 0:
                observed_fps = (len(frames) - 1) / elapsed
        means = {
            name: statistics.fmean(
                frame.inclusive_stages.get(name, 0.0) for frame in frames
            )
            for name in names
        }
        return AggregateStats(
            frame_count=len(frames),
            avg_total=statistics.fmean(totals),
            avg_masking=means.get("masking"),
            avg_ball_detection=means.get("ball_detection"),
            avg_yolo_inference=means.get(
                "detector_inference", means.get("yolo_inference")
            ),
            avg_cue_detection=means.get("cue_detection"),
            avg_tracking=means.get("tracking"),
            min_total=min(totals),
            max_total=max(totals),
            avg_fps=1.0 / statistics.fmean(totals) if statistics.fmean(totals) else 0.0,
            percentiles_ms=percentiles,
            stage_exclusive_ms=exclusive,
            observed_fps=observed_fps,
        )

    def get_bottlenecks(self, top_n: int = 3) -> list[tuple[str, float]]:
        """Rank exclusive stage costs so nested stages are not double-counted."""
        if top_n < 0:
            raise ValueError("top_n cannot be negative")
        stats = self.get_current_stats()
        stages = [(name, value) for name, value in stats.stage_exclusive_ms.items()]
        return sorted(stages, key=lambda item: item[1], reverse=True)[:top_n]

    def get_realtime_status(self) -> dict[str, Any]:
        """Return recent timing and asynchronously sampled resource values."""
        stats = self.get_current_stats()
        cpu_percent, memory_mb = self._cpu_percent, self._memory_mb
        target_fps = self.target_fps
        status = "collecting" if stats.frame_count else "unavailable"
        return {
            "fps": stats.avg_fps,
            "observed_processing_fps": stats.observed_fps,
            "target_fps": target_fps,
            "meeting_target": (
                stats.avg_fps >= target_fps if target_fps is not None else None
            ),
            "frame_time_ms": stats.avg_total * 1000,
            "target_frame_time_ms": 1000.0 / target_fps if target_fps else None,
            "overhead_ms": (
                max(0.0, stats.avg_total * 1000 - 1000.0 / target_fps)
                if target_fps
                else None
            ),
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "status": status,
            "bottlenecks": [
                {"stage": name, "avg_time_ms": time_ms}
                for name, time_ms in self.get_bottlenecks(5)
            ],
            "frame_count": stats.frame_count,
            "total_frames": self.total_frames,
            "uptime_seconds": time.monotonic() - self.start_time,
            **stats.get_summary(),
        }

    def close(self) -> None:
        """Stop the resource sampler during vision shutdown."""
        self._sampler_stop.set()
        if self._sampler.is_alive():
            self._sampler.join(timeout=max(0.1, self._resource_interval * 2))

    def _sample_resources(self) -> None:
        """Sample process resources away from the frame-processing thread."""
        while not self._sampler_stop.wait(self._resource_interval):
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
                memory_mb = self._process.memory_info().rss / (1024 * 1024)
                with self._lock:
                    self._cpu_percent = cpu_percent
                    self._memory_mb = memory_mb
                if self._event_sink is not None:
                    cpu_time_seconds = None
                    cpu_times = getattr(self._process, "cpu_times", None)
                    if callable(cpu_times):
                        cpu_time = cpu_times()
                        cpu_time_seconds = float(cpu_time.user + cpu_time.system)
                    self._event_sink(
                        {
                            "event_type": "process_resource_sample",
                            "resource_measurement_status": "available",
                            "sampled_monotonic_ns": time.perf_counter_ns(),
                            "cpu_percent": cpu_percent,
                            "cpu_time_seconds": cpu_time_seconds,
                            "resident_memory_bytes": int(memory_mb * 1024 * 1024),
                        }
                    )
            except Exception:
                logger.debug("Performance resource sample unavailable", exc_info=True)

    def _log_summary(self) -> None:
        """Log a periodic summary only when explicitly configured."""
        stats = self.get_current_stats()
        logger.info(
            "Vision performance: frames=%s mean=%.2fms p95=%.2fms observed_fps=%s",
            stats.frame_count,
            stats.avg_total * 1000,
            stats.percentiles_ms.get("total", {}).get("p95", 0.0),
            (
                f"{stats.observed_fps:.1f}"
                if stats.observed_fps is not None
                else "unknown"
            ),
        )


def _percentiles(values: list[float]) -> dict[str, float]:
    """Calculate stable nearest-rank percentiles without a NumPy dependency."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)

    def nearest_rank(percent: float) -> float:
        index = max(
            0, min(len(ordered) - 1, math.ceil(percent * len(ordered) / 100) - 1)
        )
        return ordered[index] * 1000

    return {f"p{percent}": nearest_rank(percent) for percent in (50, 95, 99)}


_profiler: PerformanceProfiler | None = None


def get_profiler(
    enabled: bool = True,
    history_size: int = 100,
    log_interval: int = 30,
    enable_console_logging: bool = False,
) -> PerformanceProfiler | None:
    """Get or create the optional process-level profiler."""
    global _profiler
    if not enabled:
        return None
    if _profiler is None:
        _profiler = PerformanceProfiler(
            history_size=history_size,
            log_interval=log_interval,
            enable_console_logging=enable_console_logging,
        )
    return _profiler


def reset_profiler() -> None:
    """Stop and discard the optional process-level profiler."""
    global _profiler
    if _profiler is not None:
        _profiler.close()
    _profiler = None
