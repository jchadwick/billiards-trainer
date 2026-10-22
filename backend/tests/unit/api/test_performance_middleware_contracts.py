"""Deterministic behavioral contracts for API performance monitoring."""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response

import backend.api.middleware.performance as performance


def _request(path: str = "/balls", method: str = "GET") -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
        }
    )


def _monitor(
    monkeypatch: pytest.MonkeyPatch,
    **config_overrides: Any,
) -> performance.PerformanceMonitor:
    """Create a monitor without starting its background resource-polling thread."""
    monkeypatch.setattr(
        performance.PerformanceMonitor, "_start_system_monitoring", lambda _self: None
    )
    monkeypatch.setattr(performance.psutil, "Process", lambda: _FakeProcess())
    config = performance.PerformanceConfig(
        enable_system_metrics=False, **config_overrides
    )
    return performance.PerformanceMonitor(config)


class _FakeProcess:
    def cpu_percent(self) -> float:
        return 12.5

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=16 * 1024 * 1024)

    def num_threads(self) -> int:
        return 4

    def num_fds(self) -> int:
        return 7


def test_performance_config_defaults_and_independent_limits() -> None:
    first = performance.PerformanceConfig()
    second = performance.PerformanceConfig()

    assert first.enable_monitoring is True
    assert first.enable_system_metrics is True
    assert first.enable_endpoint_stats is True
    assert first.enable_slow_query_detection is True
    assert first.slow_request_threshold_ms == 1000.0
    assert first.metrics_retention_minutes == 60
    assert first.stats_window_size == 1000
    assert first.system_metrics_interval_seconds == 30
    assert first.endpoint_stats_recent_durations_maxlen == 100
    assert first.cpu_percent_history_maxlen == 60
    assert first.memory_usage_history_maxlen == 60
    first.excluded_paths.append("/private")
    assert "/private" not in second.excluded_paths


def test_system_metrics_collects_resources_and_bounds_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(performance.psutil, "Process", lambda: process)
    monkeypatch.setattr(performance.psutil, "cpu_percent", lambda: 23.0)
    monkeypatch.setattr(
        performance.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=61.0, available=512 * 1024 * 1024),
    )
    monkeypatch.setattr(
        performance.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(used=25, total=100, free=75),
    )
    monkeypatch.setattr(
        performance.psutil,
        "net_io_counters",
        lambda: SimpleNamespace(
            bytes_sent=11, bytes_recv=22, packets_sent=3, packets_recv=4
        ),
    )
    metrics = performance.SystemMetrics(cpu_history_maxlen=2, memory_history_maxlen=1)

    first = metrics.get_current_metrics()
    second = metrics.get_current_metrics()
    third = metrics.get_current_metrics()

    assert first["process"] == {
        "cpu_percent": 12.5,
        "memory_mb": 16.0,
        "threads": 4,
        "file_descriptors": 7,
    }
    assert first["system"] == {
        "cpu_percent": 23.0,
        "memory_percent": 61.0,
        "memory_available_mb": 512.0,
        "disk_usage_percent": 25.0,
        "disk_free_mb": 75 / 1024 / 1024,
    }
    assert first["network"] == {
        "bytes_sent": 11,
        "bytes_recv": 22,
        "packets_sent": 3,
        "packets_recv": 4,
    }
    assert second["averages"] == {"cpu_avg_1min": 12.5, "memory_avg_1min": 16.0}
    assert third["averages"] == {"cpu_avg_1min": 12.5, "memory_avg_1min": 16.0}
    assert list(metrics.cpu_percent_history) == [12.5, 12.5]
    assert list(metrics.memory_usage_history) == [16.0]


def test_system_metrics_uses_zero_file_descriptors_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProcessWithoutFds:
        def cpu_percent(self) -> float:
            return 1.0

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=0)

        def num_threads(self) -> int:
            return 1

    monkeypatch.setattr(performance.psutil, "Process", ProcessWithoutFds)
    monkeypatch.setattr(performance.psutil, "cpu_percent", lambda: 0.0)
    monkeypatch.setattr(
        performance.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0, available=0),
    )
    monkeypatch.setattr(
        performance.psutil,
        "disk_usage",
        lambda _path: SimpleNamespace(used=0, total=1, free=1),
    )
    monkeypatch.setattr(
        performance.psutil,
        "net_io_counters",
        lambda: SimpleNamespace(
            bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0
        ),
    )

    assert (
        performance.SystemMetrics().get_current_metrics()["process"]["file_descriptors"]
        == 0
    )


def test_system_metrics_returns_empty_mapping_and_logs_on_collector_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(performance.psutil, "Process", lambda: _FakeProcess())
    metrics = performance.SystemMetrics()
    monkeypatch.setattr(
        metrics.process, "cpu_percent", Mock(side_effect=OSError("gone"))
    )

    with caplog.at_level(logging.ERROR):
        assert metrics.get_current_metrics() == {}

    assert "Error collecting system metrics: gone" in caplog.text
    assert not metrics.cpu_percent_history
    assert not metrics.memory_usage_history


def test_monitor_starts_daemon_poller_only_when_system_metrics_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: list[SimpleNamespace] = []

    def thread_factory(*, target: Any, daemon: bool) -> SimpleNamespace:
        thread = SimpleNamespace(target=target, daemon=daemon, started=False)
        thread.start = lambda: setattr(thread, "started", True)
        threads.append(thread)
        return thread

    monkeypatch.setattr(
        performance.PerformanceMonitor,
        "_start_system_monitoring",
        performance.PerformanceMonitor._start_system_monitoring,
    )
    monkeypatch.setattr(performance.threading, "Thread", thread_factory)
    monkeypatch.setattr(performance.psutil, "Process", lambda: _FakeProcess())
    event_loop = SimpleNamespace(run_until_complete=lambda coroutine: coroutine.close())
    installed_loops: list[Any] = []
    monkeypatch.setattr(performance.asyncio, "new_event_loop", lambda: event_loop)
    monkeypatch.setattr(performance.asyncio, "set_event_loop", installed_loops.append)

    enabled = performance.PerformanceMonitor(
        performance.PerformanceConfig(enable_system_metrics=True)
    )
    disabled = performance.PerformanceMonitor(
        performance.PerformanceConfig(enable_system_metrics=False)
    )

    assert len(threads) == 1
    assert threads[0].daemon is True
    assert threads[0].started is True
    assert callable(threads[0].target)
    threads[0].target()
    assert installed_loops == [event_loop]
    assert enabled.config.enable_system_metrics is True
    assert disabled.config.enable_system_metrics is False


def test_system_monitoring_loop_polls_updates_cache_alerts_and_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(
        monkeypatch,
        memory_alert_threshold_mb=10,
        cpu_alert_threshold_percent=20,
        system_metrics_interval_seconds=7,
    )
    metrics = {"process": {"memory_mb": 12, "cpu_percent": 21}}
    monitor.system_metrics.get_current_metrics = Mock(return_value=metrics)  # type: ignore[method-assign]
    sleep_intervals: list[float] = []

    async def fake_sleep(interval: float) -> None:
        sleep_intervals.append(interval)
        if len(sleep_intervals) > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(performance.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(performance, "time", SimpleNamespace(time=lambda: 123.0))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(monitor._system_monitoring_loop())

    assert sleep_intervals == [7, 7]
    assert monitor.system_metrics.get_current_metrics.call_count == 1
    assert monitor.system_metrics_cache == metrics
    assert monitor.last_system_metrics_update == 123.0
    assert monitor.memory_alerts_count == monitor.cpu_alerts_count == 1


def test_system_monitoring_loop_recovers_after_poll_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monitor = _monitor(monkeypatch)
    monitor.system_metrics.get_current_metrics = Mock(  # type: ignore[method-assign]
        side_effect=[
            RuntimeError("temporary"),
            {"process": {"memory_mb": 0, "cpu_percent": 0}},
        ]
    )
    sleep_count = 0

    async def fake_sleep(_interval: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(performance.asyncio, "sleep", fake_sleep)
    with caplog.at_level(logging.ERROR, logger="api.performance"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(monitor._system_monitoring_loop())

    assert "Error in system monitoring loop: temporary" in caplog.text
    assert monitor.system_metrics_cache == {
        "process": {"memory_mb": 0, "cpu_percent": 0}
    }
    assert monitor.system_metrics.get_current_metrics.call_count == 2


def test_alert_check_ignores_missing_metrics_and_counts_each_exceeded_threshold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monitor = _monitor(
        monkeypatch,
        memory_alert_threshold_mb=50,
        cpu_alert_threshold_percent=60,
    )
    monitor._check_system_alerts({})
    monitor._check_system_alerts({"process": {"memory_mb": 51, "cpu_percent": 61}})

    assert monitor.memory_alerts_count == 1
    assert monitor.cpu_alerts_count == 1
    assert "High memory usage detected" in caplog.text
    assert "High CPU usage detected" in caplog.text


def test_record_request_obeys_monitoring_and_excluded_path_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = _monitor(monkeypatch, enable_monitoring=False)
    disabled.record_request(_request(), Response("ok"), 3)
    assert not disabled.request_metrics

    excluded = _monitor(monkeypatch)
    excluded.record_request(_request("/health"), Response("ok"), 3)
    assert not excluded.request_metrics
    assert excluded.should_monitor_path("/balls") is True
    assert excluded.should_monitor_path("/health") is False


def test_record_request_tracks_sizes_errors_slow_requests_and_bounded_endpoint_data(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monitor = _monitor(
        monkeypatch,
        stats_window_size=2,
        endpoint_stats_recent_durations_maxlen=2,
        slow_request_threshold_ms=10,
    )
    monitor.system_metrics_cache = {"process": {"memory_mb": 12.0, "cpu_percent": 4.5}}
    times = iter(
        [
            datetime(2025, 1, 1, 0, 0, 1),
            datetime(2025, 1, 1, 0, 0, 2),
            datetime(2025, 1, 1, 0, 0, 3),
        ]
    )
    monkeypatch.setattr(
        performance, "datetime", SimpleNamespace(utcnow=lambda: next(times))
    )

    monitor.record_request(
        _request("/balls", "GET"), Response("ok", status_code=200), 5, 10, 2
    )
    monitor.record_request(
        _request("/balls", "GET"), Response("bad", status_code=500), 20, 0, 3
    )
    monitor.record_request(
        _request("/balls", "GET"), Response("slow", status_code=302), 30, 1, 4
    )

    assert len(monitor.request_metrics) == 2
    assert monitor.request_metrics[-1].method == "GET"
    assert monitor.request_metrics[-1].status_code == 302
    assert monitor.request_metrics[-1].request_size == 1
    assert monitor.request_metrics[-1].response_size == 4
    assert monitor.request_metrics[-1].memory_usage_mb == 12.0
    assert monitor.request_metrics[-1].cpu_usage_percent == 4.5
    assert monitor.request_metrics[-1].error is None
    endpoint = monitor.endpoint_stats["GET:/balls"]
    assert endpoint.total_requests == 3
    assert endpoint.total_errors == 1
    assert endpoint.error_rate == pytest.approx(100 / 3)
    assert endpoint.total_duration_ms == 55
    assert endpoint.min_duration_ms == 5
    assert endpoint.max_duration_ms == 30
    assert list(endpoint.recent_durations) == [20, 30]
    assert monitor.slow_requests_count == 2
    assert "Slow request detected" in caplog.text


def test_record_request_can_disable_endpoint_and_slow_request_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(
        monkeypatch,
        enable_endpoint_stats=False,
        enable_slow_query_detection=False,
        slow_request_threshold_ms=0,
    )
    monitor.record_request(_request(), Response("ok"), 100)

    assert len(monitor.request_metrics) == 1
    assert not monitor.endpoint_stats
    assert monitor.slow_requests_count == 0


def test_summary_handles_empty_retained_window_and_computes_sorted_percentiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(
        monkeypatch,
        metrics_retention_minutes=1,
        response_time_percentiles=[0, 50, 100],
    )
    fixed_now = datetime(2025, 1, 1, 12, 0, 0)
    monkeypatch.setattr(
        performance, "datetime", SimpleNamespace(utcnow=lambda: fixed_now)
    )
    monitor.request_metrics.extend(
        [
            performance.RequestMetrics(
                timestamp=fixed_now - timedelta(minutes=2),
                method="GET",
                path="/expired",
                status_code=500,
                duration_ms=999,
                request_size=0,
                response_size=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error="old",
            )
        ]
    )
    assert monitor.get_performance_summary() == {
        "message": "No recent metrics available"
    }

    for duration, status in [(30, 200), (10, 503), (20, 200), (40, 200)]:
        monitor.request_metrics.append(
            performance.RequestMetrics(
                timestamp=fixed_now,
                method="GET",
                path="/current",
                status_code=status,
                duration_ms=duration,
                request_size=0,
                response_size=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                error="failure" if status >= 400 else None,
            )
        )
    monitor.slow_requests_count = 3
    monitor.memory_alerts_count = 2
    monitor.cpu_alerts_count = 1
    monitor.system_metrics_cache = {"sample": True}

    summary = monitor.get_performance_summary()

    assert summary["overview"] == {
        "total_requests": 4,
        "error_count": 1,
        "error_rate_percent": 25.0,
        "slow_requests_count": 3,
        "monitoring_period_minutes": 1,
    }
    assert summary["response_times"] == {
        "average_ms": 25,
        "median_ms": 25,
        "min_ms": 10,
        "max_ms": 40,
        "percentiles": {"p0": 10, "p50": 20, "p100": 40},
    }
    assert summary["system_metrics"] == {"sample": True}
    assert summary["alerts"] == {
        "memory_alerts": 2,
        "cpu_alerts": 1,
        "slow_requests": 3,
    }


def test_endpoint_statistics_support_disabled_empty_and_recent_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = _monitor(monkeypatch, enable_endpoint_stats=False)
    assert disabled.get_endpoint_statistics() == {
        "message": "Endpoint statistics disabled"
    }

    monitor = _monitor(monkeypatch, endpoint_stats_recent_durations_maxlen=2)
    assert monitor.get_endpoint_statistics()["total_endpoints"] == 0
    now = datetime(2025, 1, 1)
    stats = performance.EndpointStats(
        path="/balls",
        method="POST",
        total_requests=3,
        total_errors=1,
        total_duration_ms=60,
        min_duration_ms=10,
        max_duration_ms=30,
        recent_durations=deque([20, 30], maxlen=2),
        last_request=now,
        error_rate=100 / 3,
    )
    monitor.endpoint_stats["POST:/balls"] = stats
    monitor.endpoint_stats["GET:/empty"] = performance.EndpointStats(
        path="/empty", method="GET", total_requests=0
    )

    endpoint_data = monitor.get_endpoint_statistics()["endpoints"]["POST:/balls"]
    assert endpoint_data == {
        "method": "POST",
        "path": "/balls",
        "total_requests": 3,
        "total_errors": 1,
        "error_rate_percent": pytest.approx(100 / 3),
        "average_duration_ms": 20,
        "min_duration_ms": 10,
        "max_duration_ms": 30,
        "last_request": now.isoformat(),
        "recent_average_ms": 25,
        "recent_median_ms": 25,
        "recent_requests_count": 2,
    }
    stats.min_duration_ms = float("inf")
    stats.last_request = None
    stats.recent_durations.clear()
    endpoint_data = monitor.get_endpoint_statistics()["endpoints"]["POST:/balls"]
    assert endpoint_data["min_duration_ms"] == 0
    assert endpoint_data["last_request"] is None
    assert "recent_average_ms" not in endpoint_data


def test_reset_clears_retained_request_endpoint_and_alert_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(monkeypatch)
    monitor.record_request(_request(), Response("error", status_code=500), 1)
    monitor.memory_alerts_count = monitor.cpu_alerts_count = 2
    monitor.slow_requests_count = 3

    monitor.reset_statistics()

    assert not monitor.request_metrics
    assert not monitor.endpoint_stats
    assert (
        monitor.slow_requests_count,
        monitor.memory_alerts_count,
        monitor.cpu_alerts_count,
    ) == (0, 0, 0)


def test_force_gc_reports_collection_count_and_process_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _monitor(monkeypatch)
    monkeypatch.setattr(performance.gc, "collect", lambda: 7)
    result = monitor.force_garbage_collection()

    assert result["objects_collected"] == 7
    assert result["memory_before_mb"] == 16
    assert datetime.fromisoformat(result["timestamp"])


def test_middleware_disabled_bypasses_timing_metrics_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = performance.PerformanceMiddleware(
        lambda scope, receive, send: None,
        config=performance.PerformanceConfig(
            enable_monitoring=False, enable_system_metrics=False
        ),
    )
    middleware.monitor.record_request = Mock()  # type: ignore[method-assign]
    monkeypatch.setattr(
        performance,
        "time",
        SimpleNamespace(
            time=Mock(side_effect=AssertionError("timed disabled request"))
        ),
    )
    response = Response("ordinary")
    calls: list[Request] = []

    async def call_next(request: Request) -> Response:
        calls.append(request)
        return response

    result = asyncio.run(middleware.dispatch(_request(), call_next))

    assert result is response
    assert len(calls) == 1
    middleware.monitor.record_request.assert_not_called()
    assert "X-Response-Time" not in response.headers


def test_middleware_times_response_records_sizes_and_adds_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = performance.PerformanceMiddleware(
        lambda scope, receive, send: None,
        config=performance.PerformanceConfig(enable_system_metrics=False),
    )
    recorded: list[dict[str, Any]] = []
    middleware.monitor.record_request = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign]
    clock = iter([100.0, 100.01234])
    monkeypatch.setattr(performance, "time", SimpleNamespace(time=lambda: next(clock)))
    response = Response("six!!")

    async def call_next(_request: Request) -> Response:
        return response

    result = asyncio.run(middleware.dispatch(_request("/observed", "POST"), call_next))

    assert result is response
    assert result.headers["X-Response-Time"] == "12.34ms"
    assert len(recorded) == 1
    assert recorded[0]["request"].url.path == "/observed"
    assert recorded[0]["response"] is response
    assert recorded[0]["duration_ms"] == pytest.approx(12.34, abs=0.001)
    assert recorded[0]["request_size"] == 0
    assert recorded[0]["response_size"] == 5


def test_middleware_suppresses_unreadable_response_body_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = performance.PerformanceMiddleware(
        lambda scope, receive, send: None,
        config=performance.PerformanceConfig(enable_system_metrics=False),
    )
    recorded: list[dict[str, Any]] = []
    middleware.monitor.record_request = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign]
    clock = iter([1.0, 1.001])
    monkeypatch.setattr(performance, "time", SimpleNamespace(time=lambda: next(clock)))

    class UnreadableBody:
        def __len__(self) -> int:
            raise RuntimeError("streaming response has no body")

    response = SimpleNamespace(body=UnreadableBody(), headers={}, status_code=200)

    async def call_next(_request: Request) -> Response:
        return response

    asyncio.run(middleware.dispatch(_request(), call_next))

    assert recorded[0]["response_size"] == 0
    assert response.headers["X-Response-Time"] == "1.00ms"


def test_middleware_accepts_response_without_body_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = performance.PerformanceMiddleware(
        lambda scope, receive, send: None,
        config=performance.PerformanceConfig(enable_system_metrics=False),
    )
    recorded: list[dict[str, Any]] = []
    middleware.monitor.record_request = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign]
    clock = iter([1.0, 1.001])
    monkeypatch.setattr(performance, "time", SimpleNamespace(time=lambda: next(clock)))
    response = SimpleNamespace(headers={}, status_code=204)

    async def call_next(_request: Request) -> Any:
        return response

    assert asyncio.run(middleware.dispatch(_request(), call_next)) is response
    assert recorded[0]["response_size"] == 0
    assert response.headers["X-Response-Time"] == "1.00ms"


def test_middleware_logs_and_reraises_request_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    middleware = performance.PerformanceMiddleware(
        lambda scope, receive, send: None,
        config=performance.PerformanceConfig(enable_system_metrics=False),
    )
    record_request = Mock()
    middleware.monitor.record_request = record_request  # type: ignore[method-assign]
    clock = iter([2.0, 2.25])
    monkeypatch.setattr(performance, "time", SimpleNamespace(time=lambda: next(clock)))

    async def fail(_request: Request) -> Response:
        raise ValueError("handler failed")

    with caplog.at_level(logging.ERROR, logger="api.performance"):
        with pytest.raises(ValueError, match="handler failed"):
            asyncio.run(middleware.dispatch(_request(), fail))

    assert "Request failed after 250.00ms: handler failed" in caplog.text
    record_request.assert_not_called()


def test_setup_and_global_monitor_accessors() -> None:
    app = FastAPI()
    config = performance.PerformanceConfig(enable_system_metrics=False)
    performance.setup_performance_monitoring(app, config)
    assert app.user_middleware
    assert app.user_middleware[0].cls is performance.PerformanceMiddleware
    assert app.user_middleware[0].kwargs["config"] is config

    monitor = performance.PerformanceMonitor(config)
    performance.set_performance_monitor(monitor)
    assert performance.get_performance_monitor() is monitor


def test_setup_uses_default_configuration_when_omitted() -> None:
    app = FastAPI()
    performance.setup_performance_monitoring(app)

    configured = app.user_middleware[0].kwargs["config"]
    assert isinstance(configured, performance.PerformanceConfig)
    assert configured.enable_monitoring is True
