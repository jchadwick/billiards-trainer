"""Deterministic tests for health-monitor resource and API measurements."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.system import health_monitor as health_monitor_module
from backend.system.health_monitor import ComponentHealth, HealthMonitor


class FakeConfig:
    """Provide explicit, stable values for health-monitor thresholds."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class FakePsutil:
    """Return deterministic host-resource readings without touching the host."""

    def __init__(
        self,
        cpu: float = 12.5,
        memory_percent: float = 40.0,
        available: int = 4 * 1024**3,
    ) -> None:
        self.cpu = cpu
        self.memory_percent = memory_percent
        self.available = available

    def cpu_percent(self, interval: float | None = None) -> float:
        assert interval == 0.1
        return self.cpu

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(percent=self.memory_percent, available=self.available)

    def disk_usage(self, path: str) -> SimpleNamespace:
        assert path == "/"
        return SimpleNamespace(used=50, total=100)

    def net_io_counters(self) -> SimpleNamespace:
        return SimpleNamespace(bytes_sent=1234, bytes_recv=5678)


def set_config(
    monkeypatch: pytest.MonkeyPatch, values: dict[str, float] | None = None
) -> None:
    """Keep threshold behavior independent of the developer's local config."""
    monkeypatch.setattr(health_monitor_module, "config", FakeConfig(values))


@pytest.mark.system
@pytest.mark.asyncio
async def test_resource_sample_records_all_metrics_and_stays_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal host sample is complete and uses byte and GiB conversions."""
    set_config(monkeypatch)
    monkeypatch.setattr(health_monitor_module, "psutil", FakePsutil())
    monitor = HealthMonitor()

    await monitor._check_system_resources()

    health = monitor.get_component_health("system")
    assert health is not None
    assert health.status == "healthy"
    assert health.metrics["cpu_usage"].value == 12.5
    assert health.metrics["memory_usage"].value == 40.0
    assert health.metrics["memory_available"].value == 4.0
    assert health.metrics["disk_usage"].value == 50.0
    assert health.metrics["network_bytes_sent"].value == 1234.0
    assert health.metrics["network_bytes_recv"].value == 5678.0


@pytest.mark.system
@pytest.mark.asyncio
async def test_resource_thresholds_mark_degraded_and_critical_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warning thresholds degrade health and critical thresholds take priority."""
    set_config(
        monkeypatch,
        {
            "api.health_monitor.system_resources.cpu_percent_warning": 20.0,
            "api.health_monitor.system_resources.cpu_percent_critical": 90.0,
        },
    )
    monitor = HealthMonitor()
    monkeypatch.setattr(health_monitor_module, "psutil", FakePsutil(cpu=50.0))

    await monitor._check_system_resources()

    degraded = monitor.get_component_health("system")
    assert degraded is not None
    assert degraded.status == "degraded"
    assert degraded.warnings == ["High resource usage"]
    assert not degraded.metrics["cpu_usage"].healthy

    monkeypatch.setattr(health_monitor_module, "psutil", FakePsutil(cpu=95.0))
    await monitor._check_system_resources()

    critical = monitor.get_component_health("system")
    assert critical is not None
    assert critical.status == "unhealthy"
    assert critical.warnings == ["Critical resource usage"]


@pytest.mark.system
@pytest.mark.asyncio
async def test_memory_and_disk_thresholds_affect_health_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory warnings degrade health while critical disk use is unhealthy."""
    set_config(
        monkeypatch,
        {
            "api.health_monitor.system_resources.memory_percent_warning": 80.0,
            "api.health_monitor.system_resources.memory_percent_critical": 95.0,
            "api.health_monitor.system_resources.disk_percent_warning": 40.0,
            "api.health_monitor.system_resources.disk_percent_critical": 90.0,
        },
    )
    monitor = HealthMonitor()
    monkeypatch.setattr(
        health_monitor_module, "psutil", FakePsutil(memory_percent=85.0)
    )

    await monitor._check_system_resources()

    memory_health = monitor.get_component_health("system")
    assert memory_health is not None
    assert memory_health.status == "degraded"
    assert memory_health.warnings == ["High resource usage"]
    assert not memory_health.metrics["memory_usage"].healthy

    class FullDiskPsutil(FakePsutil):
        def disk_usage(self, path: str) -> SimpleNamespace:
            assert path == "/"
            return SimpleNamespace(used=95, total=100)

    monkeypatch.setattr(health_monitor_module, "psutil", FullDiskPsutil())
    await monitor._check_system_resources()

    disk_health = monitor.get_component_health("system")
    assert disk_health is not None
    assert disk_health.status == "unhealthy"
    assert disk_health.warnings == ["Critical resource usage"]
    assert not disk_health.metrics["disk_usage"].healthy


@pytest.mark.system
@pytest.mark.asyncio
async def test_resource_sampling_is_unavailable_without_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional resource support leaves no misleading synthetic component."""
    set_config(monkeypatch)
    monkeypatch.setattr(health_monitor_module, "psutil", None)
    monitor = HealthMonitor()

    await monitor._check_system_resources()

    assert monitor.get_component_health("system") is None


@pytest.mark.system
@pytest.mark.asyncio
async def test_resource_sampling_failure_is_reported_and_later_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient host API error becomes a degraded sample, then recovers."""
    set_config(monkeypatch)
    monitor = HealthMonitor()

    class BrokenPsutil(FakePsutil):
        def cpu_percent(self, interval: float | None = None) -> float:
            raise OSError("resource provider unavailable")

    monkeypatch.setattr(health_monitor_module, "psutil", BrokenPsutil())
    await monitor._check_system_resources()
    failed = monitor.get_component_health("system")
    assert failed is not None
    assert failed.status == "degraded"
    assert failed.error_count == 1
    assert "resource provider unavailable" in failed.warnings[0]

    monkeypatch.setattr(health_monitor_module, "psutil", FakePsutil())
    await monitor._check_system_resources()
    recovered = monitor.get_component_health("system")
    assert recovered is not None
    assert recovered.status == "healthy"
    assert recovered.error_count == 0


@pytest.mark.system
def test_api_metrics_track_counts_error_rate_and_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API metrics convert units and evaluate configured response/error limits."""
    set_config(
        monkeypatch,
        {
            "api.health_monitor.api.max_response_time_ms": 100.0,
            "api.health_monitor.api.max_error_rate": 0.25,
        },
    )
    monkeypatch.setattr(health_monitor_module.time, "time", lambda: 110.0)
    monitor = HealthMonitor()
    monitor._startup_time = 100.0
    monitor.record_api_request(0.05, success=True)
    monitor.record_api_request(0.15, success=False)

    metrics = monitor.get_api_metrics()

    assert metrics["api_requests_total"].value == 2
    assert metrics["api_requests_per_second"].value == 0.2
    assert metrics["api_avg_response_time"].value == 100.0
    assert metrics["api_avg_response_time"].healthy is False
    assert metrics["api_error_rate"].value == 50.0
    assert metrics["api_error_rate"].healthy is False


@pytest.mark.system
def test_api_metrics_are_empty_until_a_request_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No samples means there are no fabricated latency or error metrics."""
    set_config(monkeypatch)

    assert HealthMonitor().get_api_metrics() == {}


@pytest.mark.system
def test_api_request_buffer_is_bounded_but_lifetime_counts_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evicting old timings does not lose request or error totals."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    monitor._max_request_times = 2
    for duration, success in [(0.01, True), (0.03, False), (0.05, True)]:
        monitor.record_api_request(duration, success=success)

    metrics = monitor.get_api_metrics()
    assert len(monitor._api_request_times) == 2
    assert monitor._api_request_times == [0.03, 0.05]
    assert metrics["api_requests_total"].value == 3
    assert metrics["api_avg_response_time"].value == 40.0
    assert metrics["api_error_rate"].value == pytest.approx(100 / 3)


@pytest.mark.system
def test_concurrent_api_recording_preserves_counts_and_buffer_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recording lock protects totals and bounded samples under concurrency."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    monitor._max_request_times = 37

    def record_batch(_worker: int) -> None:
        for index in range(100):
            monitor.record_api_request(0.001 * (index + 1), success=index % 2 == 0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record_batch, range(8)))

    assert monitor._api_request_count == 800
    assert monitor._api_error_count == 400
    assert len(monitor._api_request_times) == 37
    assert all(value > 0 for value in monitor._api_request_times)


@pytest.mark.system
def test_health_history_retains_only_the_most_recent_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History eviction is oldest-first and does not alter the current snapshot."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    monitor._max_history = 2
    snapshots = []
    for index in range(3):
        monitor._system_health = health_monitor_module.SystemHealth(
            overall_status=f"sample-{index}"
        )
        snapshots.append(monitor._system_health)
        monitor._add_to_history()

    assert monitor.get_health_history(10) == snapshots[-2:]
    assert monitor.get_system_health() is snapshots[-1]


@pytest.mark.system
def test_system_health_aggregates_component_statuses_alerts_and_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A component failure is reflected in overall health and alert details."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    monitor._component_health = {
        "core": ComponentHealth(
            component="core", status="degraded", warnings=["stale state"]
        ),
        "vision": ComponentHealth(
            component="vision", status="unhealthy", warnings=["camera offline"]
        ),
    }

    monitor._update_system_health()
    monitor._add_to_history()

    health = monitor.get_system_health()
    assert health.overall_status == "unhealthy"
    assert health.active_alerts == ["stale state", "camera offline"]
    assert monitor.get_health_history(1) == [health]


@pytest.mark.system
@pytest.mark.asyncio
async def test_monitoring_start_is_idempotent_and_stop_cancels_the_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated starts do not create duplicate loops and stop joins cancellation."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    entered = asyncio.Event()

    async def blocking_loop() -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(monitor, "_monitoring_loop", blocking_loop)
    await monitor.start_monitoring(check_interval=0.25)
    task = monitor._monitor_task
    await entered.wait()
    await monitor.start_monitoring(check_interval=1.0)

    assert monitor._monitor_task is task
    assert monitor._check_interval == 0.25

    await monitor.stop_monitoring()
    assert task is not None
    assert task.cancelled()
    assert monitor._is_running is False
    await monitor.stop_monitoring()


@pytest.mark.system
@pytest.mark.asyncio
async def test_monitoring_loop_retries_after_transient_check_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient exception waits the configured delay then resumes checking."""
    sleeps = []
    checks = 0
    set_config(
        monkeypatch,
        {"api.health_monitor.error_retry_delay": 0.125},
    )
    monitor = HealthMonitor()
    monitor._is_running = True
    monitor._check_interval = 0.0

    async def check_components() -> None:
        nonlocal checks
        checks += 1
        if checks == 1:
            raise RuntimeError("temporary sampling failure")
        monitor._is_running = False

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(monitor, "_check_all_components", check_components)
    monkeypatch.setattr(monitor, "_update_system_health", lambda: None)
    monkeypatch.setattr(monitor, "_add_to_history", lambda: None)
    monkeypatch.setattr(monitor, "_notify_health_callbacks", AsyncMock())
    monkeypatch.setattr(monitor, "_check_alerts", AsyncMock())
    monkeypatch.setattr(health_monitor_module.asyncio, "sleep", fake_sleep)

    await monitor._monitoring_loop()

    assert checks == 2
    assert sleeps == [0.125]


@pytest.mark.system
@pytest.mark.asyncio
async def test_callback_failures_do_not_block_other_health_or_alert_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One consumer's exception is isolated from later health and alert clients."""
    set_config(monkeypatch)
    monitor = HealthMonitor()
    received_health: list[str] = []
    received_alerts: list[str] = []

    def broken_health_callback(_health: Any) -> None:
        raise RuntimeError("health consumer failed")

    def good_health_callback(_health: Any) -> None:
        received_health.append("called")

    def broken_alert_callback(
        _level: str, _message: str, _details: dict[str, Any]
    ) -> None:
        raise RuntimeError("alert consumer failed")

    async def good_alert_callback(
        _level: str, message: str, _details: dict[str, Any]
    ) -> None:
        received_alerts.append(message)

    monitor.add_health_callback(broken_health_callback)
    monitor.add_health_callback(good_health_callback)
    monitor.add_alert_callback(broken_alert_callback)
    monitor.add_alert_callback(good_alert_callback)
    monitor._component_health["camera"] = ComponentHealth(
        component="camera", status="unhealthy"
    )

    await monitor._notify_health_callbacks()
    await monitor._check_alerts()

    assert received_health == ["called"]
    assert received_alerts == ["Component camera is unhealthy"]
