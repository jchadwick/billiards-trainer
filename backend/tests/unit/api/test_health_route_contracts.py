"""Deterministic response and failure contracts for the health routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.dependencies import ApplicationState, get_app_state
from backend.api.routes import health


def metric(value: Any) -> SimpleNamespace:
    """Shape a health-monitor metric without depending on host sampling."""
    return SimpleNamespace(value=value)


def component(
    status: str = "healthy",
    *,
    warnings: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Build the monitor record consumed by the HTTP route."""
    return SimpleNamespace(
        status=status,
        warnings=warnings or [],
        metrics=metrics or {},
        last_check=1_700_000_000.0,
        uptime=42.0,
    )


def install_monitor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    components: dict[str, Any] | None = None,
    api_metrics: dict[str, Any] | None = None,
    websocket_connections: int = 0,
    error: Exception | None = None,
) -> SimpleNamespace:
    """Replace the singleton monitor with a deterministic monitor boundary."""

    def get_system_health() -> SimpleNamespace:
        if error is not None:
            raise error
        return SimpleNamespace(components=components or {})

    monitor = SimpleNamespace(
        get_system_health=get_system_health,
        get_api_metrics=lambda: api_metrics or {},
        get_websocket_connection_count=lambda: websocket_connections,
    )
    monkeypatch.setattr(health, "health_monitor", monitor)
    return monitor


def client_for(state: ApplicationState | None = None) -> TestClient:
    """Build only the health router and inject application state."""
    application = FastAPI()
    application.include_router(health.router)
    application.dependency_overrides[get_app_state] = (
        lambda: state or ApplicationState()
    )
    return TestClient(application)


@pytest.fixture(autouse=True)
def reset_shutdown_route_state() -> Any:
    """Keep shutdown scheduling state isolated between route tests."""
    health._shutdown_scheduled = False
    health._shutdown_time = None
    yield
    health._shutdown_scheduled = False
    health._shutdown_time = None


def test_health_defaults_report_status_and_uptime_without_optional_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ApplicationState()
    state.startup_time = 100.0
    state.is_healthy = True
    install_monitor(monkeypatch, components={"must-not-be-read": component()})
    monkeypatch.setattr(health.time, "time", lambda: 112.5)

    response = client_for(state).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["uptime"] == 12.5
    assert payload["components"] == {}
    assert payload["metrics"] is None


def test_health_details_translate_monitor_status_and_add_unmonitored_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ApplicationState()
    cast("Any", state).core_module = object()
    cast("Any", state).config_module = object()
    cast("Any", state).websocket_manager = object()
    install_monitor(
        monkeypatch,
        components={
            "camera": component(
                "degraded", warnings=["camera frame is old", "retrying"]
            ),
            "tracker": component("unhealthy", warnings=["tracking lost"]),
            "future-component": component("not-a-known-status"),
            "vision": component(),
        },
    )

    response = client_for(state).get("/health?include_details=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["camera"]["status"] == "degraded"
    assert payload["components"]["camera"]["message"] == "camera frame is old; retrying"
    assert payload["components"]["camera"]["errors"] == [
        "camera frame is old",
        "retrying",
    ]
    assert payload["components"]["tracker"]["status"] == "unhealthy"
    assert payload["components"]["future-component"]["status"] == "unhealthy"
    assert payload["components"]["vision"]["message"] == "Operating normally"
    assert payload["components"]["core"]["status"] == "healthy"
    assert payload["components"]["config"]["status"] == "healthy"
    assert payload["components"]["websocket"]["status"] == "healthy"


def test_health_details_mark_missing_fallback_components_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ApplicationState()
    cast("Any", state).core_module = None
    cast("Any", state).config_module = None
    cast("Any", state).websocket_manager = None
    install_monitor(monkeypatch)

    payload = client_for(state).get("/health?include_details=true").json()

    for name in ("core", "config", "websocket"):
        assert payload["components"][name]["status"] == "unhealthy"
        assert "unavailable" in payload["components"][name]["message"].lower()


def test_health_details_monitor_failure_returns_structured_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(monkeypatch, error=RuntimeError("monitor snapshot failed"))

    response = client_for().get("/health?include_details=true")

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error": "Health Check Failed",
        "message": "Unable to determine system health",
        "code": "SYS_001",
        "details": {"error": "monitor snapshot failed"},
    }


def test_monitored_system_metrics_include_api_websocket_and_network_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(
        monkeypatch,
        components={
            "system": component(
                metrics={
                    "cpu_usage": metric(24.5),
                    "memory_usage": metric(60),
                    "disk_usage": metric(71),
                    "network_bytes_sent": metric(1200),
                    "network_bytes_recv": metric(3400),
                }
            )
        },
        api_metrics={
            "api_requests_per_second": metric(3.25),
            "api_avg_response_time": metric(12.0),
        },
        websocket_connections=7,
    )

    metrics = health.get_system_metrics()

    assert metrics is not None
    assert metrics.cpu_usage == 24.5
    assert metrics.memory_usage == 60
    assert metrics.disk_usage == 71
    assert metrics.network_io == {
        "bytes_sent": 1200,
        "bytes_recv": 3400,
        "packets_sent": 0,
        "packets_recv": 0,
    }
    assert metrics.api_requests_per_second == 3.25
    assert metrics.average_response_time == 12
    assert metrics.websocket_connections == 7


def test_monitored_system_metrics_sanitize_missing_and_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(
        monkeypatch,
        components={
            "system": component(
                metrics={
                    "cpu_usage": metric(True),
                    "memory_usage": metric("unknown"),
                    "network_bytes_sent": metric(None),
                }
            )
        },
        api_metrics={"api_avg_response_time": metric(False)},
    )

    metrics = health.get_system_metrics()

    assert metrics is not None
    assert metrics.cpu_usage == 0
    assert metrics.memory_usage == 0
    assert metrics.disk_usage == 0
    assert metrics.network_io["bytes_sent"] == 0
    assert metrics.api_requests_per_second == 0
    assert metrics.average_response_time == 0


def test_metrics_fall_back_to_psutil_when_monitor_has_no_system_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(
        monkeypatch,
        api_metrics={
            "api_requests_per_second": metric(2),
            "api_avg_response_time": metric(8.5),
        },
        websocket_connections=4,
    )

    class PsutilStub:
        def cpu_percent(self, interval: float | None = None) -> float:
            assert interval == 0.1
            return 11.0

        def virtual_memory(self) -> SimpleNamespace:
            return SimpleNamespace(percent=35.0)

        def disk_usage(self, path: str) -> SimpleNamespace:
            assert path == "/"
            return SimpleNamespace(used=25, total=100)

        def net_io_counters(self) -> SimpleNamespace:
            return SimpleNamespace(
                bytes_sent=100,
                bytes_recv=200,
                packets_sent=3,
                packets_recv=4,
            )

    monkeypatch.setattr(health, "_psutil", PsutilStub())

    metrics = health.get_system_metrics()

    assert metrics is not None
    assert metrics.cpu_usage == 11
    assert metrics.memory_usage == 35
    assert metrics.disk_usage == 25
    assert metrics.network_io == {
        "bytes_sent": 100,
        "bytes_recv": 200,
        "packets_sent": 3,
        "packets_recv": 4,
    }
    assert metrics.api_requests_per_second == 2
    assert metrics.average_response_time == 8.5
    assert metrics.websocket_connections == 4


def test_metrics_are_unavailable_when_monitor_and_optional_psutil_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(monkeypatch)
    monkeypatch.setattr(health, "_psutil", None)

    assert health.get_system_metrics() is None


def test_metrics_collection_failure_is_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(monkeypatch, error=RuntimeError("monitor unavailable"))

    assert health.get_system_metrics() is None


def test_metrics_with_zero_disk_capacity_are_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(monkeypatch)

    class ZeroCapacityPsutil:
        def cpu_percent(self, interval: float | None = None) -> float:
            return 1.0

        def virtual_memory(self) -> SimpleNamespace:
            return SimpleNamespace(percent=2.0)

        def disk_usage(self, path: str) -> SimpleNamespace:
            return SimpleNamespace(used=0, total=0)

        def net_io_counters(self) -> SimpleNamespace:
            return SimpleNamespace(
                bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0
            )

    monkeypatch.setattr(health, "_psutil", ZeroCapacityPsutil())

    assert health.get_system_metrics() is None


def test_health_metrics_endpoint_returns_monitor_metrics_and_validates_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(
        monkeypatch,
        components={"system": component(metrics={"cpu_usage": metric(3)})},
    )

    response = client_for().get("/health/metrics?time_range=1h")
    invalid = client_for().get("/health/metrics?time_range=2h")

    assert response.status_code == 200
    assert response.json()["cpu_usage"] == 3
    assert invalid.status_code == 422


def test_metrics_endpoint_uses_safe_defaults_when_metrics_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_monitor(monkeypatch)
    monkeypatch.setattr(health, "_psutil", None)

    response = client_for().get("/health/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "cpu_usage": 0,
        "memory_usage": 0,
        "disk_usage": 0,
        "network_io": {},
        "api_requests_per_second": 0,
        "websocket_connections": 0,
        "average_response_time": 0,
    }


def test_metrics_endpoint_wraps_unexpected_collection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        health,
        "get_system_metrics",
        lambda: (_ for _ in ()).throw(RuntimeError("metrics exploded")),
    )

    response = client_for().get("/health/metrics")

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Metrics Collection Failed"
    assert response.json()["detail"]["details"] == {"error": "metrics exploded"}


@pytest.mark.parametrize("commit", [None, "revision-abc"])
def test_version_endpoint_reports_capabilities_and_build_identity(
    monkeypatch: pytest.MonkeyPatch, commit: str | None
) -> None:
    if commit is None:
        monkeypatch.delenv("GIT_COMMIT", raising=False)
    else:
        monkeypatch.setenv("GIT_COMMIT", commit)

    response = client_for().get("/health/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["git_commit"] == (commit or "dev")
    assert payload["api_version"] == "v1"
    assert payload["capabilities"]["vision_processing"] is True
    assert "8ball" in payload["capabilities"]["game_types"]
    assert payload["build_date"] == "2024-01-15T00:00:00Z"


def test_version_endpoint_wraps_capability_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_capabilities(**_kwargs: Any) -> None:
        raise RuntimeError("capability registry unavailable")

    monkeypatch.setattr(health, "CapabilityInfo", fail_capabilities)

    response = client_for().get("/health/version")

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Version Info Failed"
    assert response.json()["detail"]["details"] == {
        "error": "capability registry unavailable"
    }


def test_readiness_reports_each_dependency_when_healthy_or_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ApplicationState()
    cast("Any", state).core_module = object()
    cast("Any", state).config_module = object()
    cast("Any", state).websocket_manager = object()
    readiness = AsyncMock(return_value=True)
    monkeypatch.setattr(health.rate_limiter, "check_readiness", readiness)

    ready = client_for(state).get("/health/ready")

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert set(ready.json()["components"].values()) == {"healthy"}
    readiness.assert_awaited_once()

    cast("Any", state).websocket_manager = None
    readiness.return_value = False
    unavailable = client_for(state).get("/health/ready")

    assert unavailable.status_code == 503
    detail = unavailable.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["components"] == {
        "core": "healthy",
        "config": "healthy",
        "websocket": "unavailable",
        "rate_limit_store": "unavailable",
    }


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (
            "python",
            {
                "provider": "python",
                "mode": "live",
                "source_health": "python_provider",
                "live_camera": True,
                "preview": "python_provider",
                "predictions": "python_core",
            },
        ),
        (
            "rust",
            {
                "provider": "rust",
                "mode": "unavailable",
                "source_health": "unavailable",
                "live_camera": False,
                "preview": "unavailable",
                "predictions": "unavailable",
            },
        ),
    ],
)
def test_perception_status_without_runtime_distinguishes_provider_modes(
    provider: str, expected: dict[str, Any]
) -> None:
    state = ApplicationState()
    state.perception_provider = provider

    response = client_for(state).get("/health/perception")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == expected["provider"]
    assert payload["mode"] == expected["mode"]
    assert payload["source_health"] == expected["source_health"]
    assert payload["live_camera"] is expected["live_camera"]
    assert payload["preview"] == expected["preview"]
    assert payload["predictions"] == expected["predictions"]
    assert payload["process_started"] is None
    assert payload["last_observation_age_seconds"] is None


def test_perception_status_with_runtime_reports_live_rust_counters() -> None:
    state = ApplicationState()
    state.perception_runtime = SimpleNamespace(
        status=SimpleNamespace(
            mode="live",
            process_started=True,
            process_running=True,
            process_return_code=None,
            process_status="running",
            first_observation_received=True,
            source_health="fresh",
            last_observation_age_seconds=0.02,
            source_silence_timeout_seconds=1.0,
            live_camera=True,
            preview="enabled",
            predictions="enabled",
            ingress_accepted_snapshots=20,
            ingress_queue_replacements=3,
            state_snapshots_published=17,
        )
    )

    payload = client_for(state).get("/health/perception").json()

    assert payload["provider"] == "rust"
    assert payload["process_status"] == "running"
    assert payload["first_observation_received"] is True
    assert payload["last_observation_age_seconds"] == 0.02
    assert payload["ingress_accepted_snapshots"] == 20
    assert payload["ingress_queue_replacements"] == 3
    assert payload["state_snapshots_published"] == 17


def test_liveness_endpoint_is_available_without_credentials() -> None:
    response = client_for().get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert response.json()["message"] == "Service is alive"


@pytest.mark.asyncio
async def test_shutdown_route_queues_background_task_and_reports_current_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = {"active_operations": 5}
    monkeypatch.setattr(health, "get_shutdown_progress", lambda: progress)
    tasks = health.BackgroundTasks()

    response = await health.graceful_shutdown(
        tasks, delay=3, force=True, save_state=False
    )

    assert response.acknowledged is True
    assert response.estimated_delay == 3
    assert response.active_operations == 5
    assert response.will_save_state is False
    assert health._shutdown_scheduled is True
    assert health._shutdown_time is not None
    assert len(tasks.tasks) == 1
    assert tasks.tasks[0].func is health.perform_graceful_shutdown
    assert tasks.tasks[0].args == (3, True, False)


@pytest.mark.asyncio
async def test_shutdown_route_rejects_duplicate_schedule_with_existing_time() -> None:
    scheduled_at = datetime(2026, 10, 21, 18, tzinfo=timezone.utc)
    health._shutdown_scheduled = True
    health._shutdown_time = scheduled_at

    with pytest.raises(HTTPException) as raised:
        await health.graceful_shutdown(health.BackgroundTasks())

    assert raised.value.status_code == 409
    detail = cast("dict[str, Any]", raised.value.detail)
    assert detail["error"] == "Shutdown Already Scheduled"
    assert detail["details"]["scheduled_time"] == scheduled_at.isoformat()


@pytest.mark.asyncio
async def test_shutdown_route_clears_schedule_when_progress_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_progress() -> dict[str, Any]:
        raise RuntimeError("coordinator status failed")

    monkeypatch.setattr(health, "get_shutdown_progress", fail_progress)

    with pytest.raises(HTTPException) as raised:
        await health.graceful_shutdown(health.BackgroundTasks(), delay=2)

    assert raised.value.status_code == 500
    detail = cast("dict[str, Any]", raised.value.detail)
    assert detail["error"] == "Shutdown Request Failed"
    assert detail["details"] == {"error": "coordinator status failed"}
    assert health._shutdown_scheduled is False
    assert health._shutdown_time is None


def test_shutdown_status_reports_progress_and_wraps_monitor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        health,
        "get_shutdown_progress",
        lambda: {"status": "draining", "active_operations": 2},
    )
    assert asyncio.run(health.shutdown_status()) == {
        "status": "draining",
        "active_operations": 2,
    }

    def fail_progress() -> dict[str, Any]:
        raise RuntimeError("progress unavailable")

    monkeypatch.setattr(health, "get_shutdown_progress", fail_progress)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(health.shutdown_status())
    assert raised.value.status_code == 500
    detail = cast("dict[str, Any]", raised.value.detail)
    assert detail["error"] == "Shutdown Status Failed"
    assert detail["details"] == {"error": "progress unavailable"}


@pytest.mark.asyncio
async def test_graceful_shutdown_failure_resets_scheduled_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = SimpleNamespace(initiate_shutdown=AsyncMock(return_value=False))
    monkeypatch.setattr(health, "shutdown_coordinator", coordinator)
    health._shutdown_scheduled = True
    health._shutdown_time = datetime.now(timezone.utc)

    await health.perform_graceful_shutdown(force=True, save_state=False)

    coordinator.initiate_shutdown.assert_awaited_once_with(force=True, save_state=False)
    assert health._shutdown_scheduled is False
    assert health._shutdown_time is None


@pytest.mark.asyncio
async def test_graceful_shutdown_success_waits_then_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = SimpleNamespace(initiate_shutdown=AsyncMock(return_value=True))
    sleep = AsyncMock()
    exit_calls: list[int] = []

    def capture_exit(code: int) -> None:
        exit_calls.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(health, "shutdown_coordinator", coordinator)
    monkeypatch.setattr(health.asyncio, "sleep", sleep)
    monkeypatch.setattr(health.sys, "exit", capture_exit)

    with pytest.raises(SystemExit) as raised:
        await health.perform_graceful_shutdown(delay=4)

    assert raised.value.code == 0
    assert exit_calls == [0]
    assert [call.args[0] for call in sleep.await_args_list] == [4, 0.1]
    coordinator.initiate_shutdown.assert_awaited_once_with(force=False, save_state=True)
    assert health._shutdown_scheduled is False
    assert health._shutdown_time is None


@pytest.mark.asyncio
async def test_graceful_shutdown_exception_exits_one_and_resets_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = SimpleNamespace(
        initiate_shutdown=AsyncMock(side_effect=RuntimeError("module failed to stop"))
    )
    exits: list[int] = []

    def capture_exit(code: int) -> None:
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(health, "shutdown_coordinator", coordinator)
    monkeypatch.setattr(health.sys, "exit", capture_exit)
    health._shutdown_scheduled = True

    with pytest.raises(SystemExit) as raised:
        await health.perform_graceful_shutdown()

    assert raised.value.code == 1
    assert exits == [1]
    assert health._shutdown_scheduled is False
    assert health._shutdown_time is None
