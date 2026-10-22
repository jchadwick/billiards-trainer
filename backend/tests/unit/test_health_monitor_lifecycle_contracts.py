"""Unit contracts for health-monitor sampling, cadence, and shutdown."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.system import health_monitor as health_monitor_module
from backend.system.health_monitor import ComponentHealth, HealthMonitor


class ConfigStub:
    """Supply only the explicit health-monitor settings a test needs."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def install_config(monkeypatch: pytest.MonkeyPatch, **settings: Any) -> None:
    """Avoid coupling health-monitor tests to workstation configuration."""
    monkeypatch.setattr(health_monitor_module, "config", ConfigStub(settings))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resource_sampling_uses_configured_cpu_interval_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU polling interval and resource warning limits come from configuration."""
    install_config(
        monkeypatch,
        **{
            "api.health_monitor.system_resources.cpu_check_interval": 0.375,
            "api.health_monitor.system_resources.cpu_percent_warning": 10.0,
        },
    )
    observed_intervals: list[float | None] = []

    class PsutilStub:
        def cpu_percent(self, interval: float | None = None) -> float:
            observed_intervals.append(interval)
            return 25.0

        def virtual_memory(self) -> SimpleNamespace:
            return SimpleNamespace(percent=35.0, available=4 * 1024**3)

        def disk_usage(self, path: str) -> SimpleNamespace:
            assert path == "/"
            return SimpleNamespace(used=20, total=100)

        def net_io_counters(self) -> SimpleNamespace:
            return SimpleNamespace(bytes_sent=10, bytes_recv=20)

    monkeypatch.setattr(health_monitor_module, "psutil", PsutilStub())
    monitor = HealthMonitor()

    await monitor._check_system_resources()

    component = monitor.get_component_health("system")
    assert component is not None
    assert observed_intervals == [0.375]
    assert component.status == "degraded"
    assert component.metrics["cpu_usage"].value == 25.0
    assert component.warnings == ["High resource usage"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_degraded_components_do_not_emit_unhealthy_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alert callbacks distinguish a degraded component from a failed one."""
    install_config(monkeypatch)
    monitor = HealthMonitor()
    alerts: list[tuple[str, str, dict[str, Any]]] = []

    def record_alert(level: str, message: str, details: dict[str, Any]) -> None:
        alerts.append((level, message, details))

    monitor.add_alert_callback(record_alert)
    monitor._component_health = {
        "vision": ComponentHealth(
            component="vision", status="degraded", warnings=["low frame rate"]
        ),
        "camera": ComponentHealth(
            component="camera", status="unhealthy", warnings=["camera stopped"]
        ),
    }

    await monitor._check_alerts()

    assert alerts == [
        (
            "error",
            "Component camera is unhealthy",
            {"component": "camera", "warnings": ["camera stopped"]},
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_monitor_can_restart_after_shutdown_without_leaking_old_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped monitor creates a fresh task on restart and can stop it again."""
    install_config(monkeypatch)
    monitor = HealthMonitor()
    started = asyncio.Event()
    invocation_count = 0

    async def waiting_loop() -> None:
        nonlocal invocation_count
        invocation_count += 1
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(monitor, "_monitoring_loop", waiting_loop)
    await monitor.start_monitoring(check_interval=0.25)
    first_task = monitor._monitor_task
    await started.wait()
    await monitor.stop_monitoring()

    assert first_task is not None
    assert first_task.cancelled()
    assert monitor._is_running is False

    started.clear()
    await monitor.start_monitoring(check_interval=0.5)
    second_task = monitor._monitor_task
    await started.wait()

    assert invocation_count == 2
    assert second_task is not None
    assert second_task is not first_task
    assert monitor._check_interval == 0.5

    await monitor.stop_monitoring()
    assert second_task.cancelled()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_component_checks_run_in_order_and_system_is_always_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered checks are awaited in stable order, including system resources."""
    install_config(monkeypatch)
    monitor = HealthMonitor()
    monitor.register_components(
        core_module=object(),
        vision_module=object(),
        config_module=object(),
        websocket_manager=object(),
    )
    order: list[str] = []

    def check(name: str) -> AsyncMock:
        async def record() -> None:
            order.append(name)

        return AsyncMock(side_effect=record)

    monkeypatch.setattr(monitor, "_check_core_module", check("core"))
    monkeypatch.setattr(monitor, "_check_vision_module", check("vision"))
    monkeypatch.setattr(monitor, "_check_config_module", check("config"))
    monkeypatch.setattr(monitor, "_check_websocket_manager", check("websocket"))
    monkeypatch.setattr(monitor, "_check_system_resources", check("system"))

    await monitor._check_all_components()

    assert order == ["core", "vision", "config", "websocket", "system"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("elapsed", "expected_sleep"),
    [(0.05, 0.15), (0.25, None)],
)
async def test_monitoring_cadence_subtracts_check_time_from_interval(
    monkeypatch: pytest.MonkeyPatch,
    elapsed: float,
    expected_sleep: float | None,
) -> None:
    """The loop waits only the remaining interval and skips waits when over budget."""
    install_config(monkeypatch)
    monitor = HealthMonitor()
    monitor._check_interval = 0.2
    monitor._is_running = True
    clock_values = iter((10.0, 10.0 + elapsed))
    monkeypatch.setattr(
        health_monitor_module,
        "time",
        SimpleNamespace(time=lambda: next(clock_values)),
    )
    requested_sleeps: list[float] = []

    async def complete_check() -> None:
        monitor._is_running = False

    async def record_sleep(delay: float) -> None:
        requested_sleeps.append(delay)

    monkeypatch.setattr(monitor, "_check_all_components", complete_check)
    monkeypatch.setattr(monitor, "_update_system_health", lambda: None)
    monkeypatch.setattr(monitor, "_add_to_history", lambda: None)
    monkeypatch.setattr(monitor, "_notify_health_callbacks", AsyncMock())
    monkeypatch.setattr(monitor, "_check_alerts", AsyncMock())
    monkeypatch.setattr(health_monitor_module.asyncio, "sleep", record_sleep)

    await monitor._monitoring_loop()

    if expected_sleep is None:
        assert requested_sleeps == []
    else:
        assert requested_sleeps == pytest.approx([expected_sleep])
