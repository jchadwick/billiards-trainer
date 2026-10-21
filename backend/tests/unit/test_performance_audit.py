"""Correctness tests for opt-in performance timing and event export."""

import threading
import time
from types import SimpleNamespace

import pytest

import backend.performance_audit as performance_audit
from backend.performance_audit import (
    AsyncAuditRecorder,
    audit_domains_from_config,
    audit_enabled_from_config,
    monotonic_clock_domain_id,
)
from backend.vision.performance_profiler import PerformanceProfiler


class ManualClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value


def test_nested_spans_use_wall_time_and_exclusive_stage_costs() -> None:
    clock = ManualClock()
    profiler = PerformanceProfiler(
        history_size=4,
        resource_sample_interval_seconds=10,
        clock_ns=clock,
    )
    try:
        profiler.start_frame(7)
        profiler.start_stage("ball_detection")
        clock.value = 10_000_000
        profiler.start_stage("tracking")
        clock.value = 30_000_000
        profiler.end_stage("tracking")
        clock.value = 50_000_000
        profiler.end_stage("ball_detection")
        clock.value = 60_000_000
        profiler.end_frame()

        frame = profiler.recent_timings[-1]
        assert frame.total_time == 0.06
        assert frame.inclusive_stages == {
            "tracking": 0.02,
            "ball_detection": 0.05,
        }
        assert frame.exclusive_stages["tracking"] == 0.02
        assert abs(frame.exclusive_stages["ball_detection"] - 0.03) < 1e-12
        bottlenecks = profiler.get_bottlenecks()
        assert [stage for stage, _duration in bottlenecks] == [
            "ball_detection",
            "tracking",
        ]
        assert [duration for _stage, duration in bottlenecks] == pytest.approx(
            [30.0, 20.0]
        )
    finally:
        profiler.close()


def test_frame_event_exports_queue_depth_with_stage_timings() -> None:
    clock = ManualClock()
    events: list[dict[str, object]] = []
    profiler = PerformanceProfiler(
        resource_sample_interval_seconds=10,
        clock_ns=clock,
        event_sink=events.append,
    )
    try:
        profiler.start_frame(
            9,
            metadata={
                "source_frame_number": 27,
                "processing_queue_depth": 2,
                "shared_memory_read_copy_seconds": 0.003,
            },
        )
        with profiler.span("model_execute"):
            clock.value += 12_000_000
        clock.value += 3_000_000
        profiler.end_frame()

        assert len(events) == 1
        assert events[0]["source_frame_number"] == 27
        assert events[0]["processing_queue_depth"] == 2
        assert events[0]["shared_memory_read_copy_seconds"] == 0.003
        assert events[0]["service_time_seconds"] == 0.015
        assert events[0]["exclusive_stages_seconds"] == {"model_execute": 0.012}
    finally:
        profiler.close()


def test_incomplete_stage_closes_at_frame_boundary_and_history_is_bounded() -> None:
    clock = ManualClock()
    profiler = PerformanceProfiler(
        history_size=2,
        resource_sample_interval_seconds=10,
        clock_ns=clock,
    )
    try:
        for frame_number in range(3):
            profiler.start_frame(frame_number)
            profiler.start_stage("masking")
            clock.value += 5_000_000
            profiler.end_frame()

        assert [frame.frame_number for frame in profiler.recent_timings] == [1, 2]
        assert profiler.get_current_stats().frame_count == 2
        assert profiler.get_current_stats().percentiles_ms["total"]["p95"] == 5.0
        assert profiler.total_frames == 3
    finally:
        profiler.close()


def test_exception_closes_nested_spans_and_preserves_frame_timing() -> None:
    clock = ManualClock()
    profiler = PerformanceProfiler(
        history_size=2,
        resource_sample_interval_seconds=10,
        clock_ns=clock,
    )
    try:
        profiler.start_frame(8)
        with pytest.raises(RuntimeError, match="detector failed"):
            with profiler.span("preprocessing"):
                clock.value += 2_000_000
                with profiler.span("inference"):
                    clock.value += 3_000_000
                    raise RuntimeError("detector failed")

        clock.value += 1_000_000
        profiler.end_frame()
        frame = profiler.recent_timings[-1]
        assert frame.total_time == pytest.approx(0.006)
        assert frame.inclusive_stages["preprocessing"] == pytest.approx(0.005)
        assert frame.inclusive_stages["inference"] == pytest.approx(0.003)
        assert frame.exclusive_stages["preprocessing"] == pytest.approx(0.002)
    finally:
        profiler.close()


def test_simultaneous_frames_are_isolated_by_processing_thread() -> None:
    profiler = PerformanceProfiler(history_size=4, resource_sample_interval_seconds=10)
    barrier = threading.Barrier(2)

    def process(frame_number: int) -> None:
        profiler.start_frame(frame_number)
        profiler.start_stage("work")
        barrier.wait(timeout=2)
        profiler.end_stage("work")
        profiler.end_frame()

    threads = [threading.Thread(target=process, args=(number,)) for number in (1, 2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        assert all(not thread.is_alive() for thread in threads)
        assert {frame.frame_number for frame in profiler.recent_timings} == {1, 2}
    finally:
        profiler.close()


def test_async_recorder_drops_when_full_and_reports_export_errors() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_writer(_event: dict[str, object]) -> None:
        entered.set()
        release.wait(timeout=2)

    recorder = AsyncAuditRecorder(
        "unused.jsonl", queue_capacity=1, writer=blocked_writer
    )
    try:
        assert recorder.record({"event_type": "first"})
        assert entered.wait(timeout=1)
        assert recorder.record({"event_type": "second"})
        assert recorder.record({"event_type": "third"}) is False
        assert recorder.status()["dropped_events"] == 1
    finally:
        release.set()
        recorder.close(timeout_seconds=1)

    failures = AsyncAuditRecorder(
        "unused.jsonl", writer=lambda _event: (_ for _ in ()).throw(OSError("disk"))
    )
    failures.record({"event_type": "failed"})
    deadline = time.monotonic() + 1
    while failures.status()["export_errors"] == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert failures.status()["export_errors"] == 1
    failures.close(timeout_seconds=1)


def test_close_is_bounded_and_exporter_exits_after_slow_writer_recovers() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_writer(_event: dict[str, object]) -> None:
        entered.set()
        release.wait(timeout=2)

    recorder = AsyncAuditRecorder(
        "unused.jsonl", queue_capacity=1, writer=blocked_writer
    )
    assert recorder.record({"event_type": "in-flight"})
    assert entered.wait(timeout=1)
    assert recorder.record({"event_type": "queued"})

    started = time.monotonic()
    recorder.close(timeout_seconds=0.01)
    assert time.monotonic() - started < 0.1
    assert recorder.status()["worker_alive"] is True

    release.set()
    recorder.close(timeout_seconds=1)
    assert recorder.status()["worker_alive"] is False
    assert recorder.status()["written_events"] == 2


def test_audit_activation_defaults_off_and_accepts_launch_override() -> None:
    assert audit_enabled_from_config(False, None) is False
    assert audit_enabled_from_config(False, "0") is False
    assert audit_enabled_from_config(False, "1") is True
    assert audit_enabled_from_config(True, None) is True


def test_audit_domains_are_independent_but_globally_opt_in() -> None:
    assert audit_domains_from_config(False, None, trajectory_enabled=True) == {
        "state": False,
        "trajectory": False,
        "viewers": False,
    }
    assert audit_domains_from_config(
        False,
        "1",
        state_enabled=False,
        trajectory_enabled=True,
        viewers_enabled=True,
    ) == {"state": False, "trajectory": True, "viewers": True}


def test_macos_clock_domain_uses_boot_session_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance_audit.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        performance_audit,
        "_run_subprocess",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="boot-session-123\n"),
    )

    domain_id = monotonic_clock_domain_id()
    assert domain_id is not None
    assert domain_id.startswith("Darwin:boot-session-123:")


def test_clock_domain_is_unavailable_on_unsupported_operating_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(performance_audit.platform, "system", lambda: "Windows")

    assert monotonic_clock_domain_id() is None


def test_recorder_assigns_a_new_run_identity_per_process_session() -> None:
    first = AsyncAuditRecorder("unused.jsonl", writer=lambda _event: None)
    second = AsyncAuditRecorder("unused.jsonl", writer=lambda _event: None)
    try:
        assert first.run_id != second.run_id
        assert first.process_id > 0
    finally:
        first.close(timeout_seconds=1)
        second.close(timeout_seconds=1)


def test_recorder_events_include_verified_clock_identity() -> None:
    events: list[dict[str, object]] = []
    recorder = AsyncAuditRecorder("unused.jsonl", writer=events.append)
    assert recorder.record({"event_type": "clock_identity"})
    recorder.close(timeout_seconds=1)

    assert len(events) == 1
    assert events[0]["clock_domain_id"] == recorder.clock_domain_id
    assert events[0]["clock_implementation"] == recorder.clock_implementation


def test_export_limit_bounds_written_events_and_reports_intentional_loss() -> None:
    written: list[dict[str, object]] = []
    recorder = AsyncAuditRecorder(
        "unused.jsonl",
        queue_capacity=8,
        max_exported_events=2,
        writer=written.append,
    )
    for index in range(5):
        assert recorder.record({"event_type": "bounded", "index": index})
    recorder.close(timeout_seconds=1)

    status = recorder.status()
    assert status["written_events"] == 2
    assert status["export_limit_dropped_events"] == 3
    assert len(written) == 2
