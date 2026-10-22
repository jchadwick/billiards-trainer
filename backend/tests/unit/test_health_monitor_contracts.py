"""Contract tests for health checks against deterministic component fakes."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.system import health_monitor as health_monitor_module
from backend.system.health_monitor import HealthMonitor


class ConfigStub:
    """Return explicit settings so tests do not depend on workstation config."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


def install_config(monkeypatch: pytest.MonkeyPatch, **values: Any) -> None:
    monkeypatch.setattr(health_monitor_module, "config", ConfigStub(values))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_core_check_reports_performance_and_validation_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(
        monkeypatch,
        **{
            "api.health_monitor.core_module.avg_update_time_threshold_ms": 5.0,
            "api.health_monitor.core_module.avg_physics_time_threshold_ms": 4.0,
            "api.health_monitor.core_module.avg_analysis_time_threshold_ms": 6.0,
            "api.health_monitor.core_module.max_error_count": 3,
            "api.health_monitor.max_warnings": 2,
        },
    )
    monitor = HealthMonitor()

    class CoreFake:
        def get_performance_metrics(self) -> SimpleNamespace:
            return SimpleNamespace(
                total_updates=37,
                avg_update_time=0.007,
                avg_physics_time=0.003,
                avg_analysis_time=0.008,
                errors_count=1,
            )

        def get_current_state(self) -> object:
            return object()

        async def validate_state(self) -> dict[str, Any]:
            return {
                "valid": False,
                "issues": ["stale position", "missing cue", "extra"],
            }

    monitor.register_components(core_module=CoreFake())
    await monitor._check_core_module()

    health = monitor.get_component_health("core")
    assert health is not None
    assert health.status == "degraded"
    assert health.metrics["total_updates"].value == 37
    assert health.metrics["avg_update_time"].value == 7.0
    assert health.metrics["avg_update_time"].healthy is False
    assert health.metrics["avg_physics_time"].healthy is True
    assert health.metrics["has_current_state"].value is True
    assert health.metrics["state_valid"].details == {
        "valid": False,
        "issues": ["stale position", "missing cue", "extra"],
    }
    assert health.warnings == [
        "Performance degradation detected",
        "stale position",
        "missing cue",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_core_check_isolates_optional_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(monkeypatch)
    monitor = HealthMonitor()

    class PartiallyBrokenCore:
        def get_performance_metrics(self) -> None:
            raise RuntimeError("metrics unavailable")

        def get_current_state(self) -> None:
            raise RuntimeError("state unavailable")

        async def validate_state(self) -> dict[str, bool]:
            return {"valid": True}

    monitor.register_components(core_module=PartiallyBrokenCore())
    await monitor._check_core_module()

    health = monitor.get_component_health("core")
    assert health is not None
    assert health.status == "degraded"
    assert health.warnings == ["Unable to retrieve performance metrics"]
    assert "total_updates" not in health.metrics
    assert health.metrics["state_valid"].healthy is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_vision_check_combines_capture_processing_and_camera_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(
        monkeypatch,
        **{
            "api.health_monitor.vision_module.max_frames_dropped": 2,
            "api.health_monitor.vision_module.max_processing_time_ms": 40.0,
            "api.health_monitor.vision_module.min_fps": 10.0,
            "api.health_monitor.vision_module.min_detection_accuracy": 0.7,
        },
    )
    monitor = HealthMonitor()
    camera_health = SimpleNamespace(
        status=SimpleNamespace(value="streaming"), device="fake-camera"
    )
    vision = SimpleNamespace(
        get_statistics=lambda: {
            "frames_processed": 123,
            "frames_dropped": 3,
            "avg_processing_time_ms": 25.0,
            "avg_fps": 30.0,
            "is_running": True,
            "camera_connected": True,
            "detection_accuracy": {"balls": 0.82, "cue": 0.61},
            "last_error": "one recoverable read error",
        },
        camera=SimpleNamespace(get_health=lambda: camera_health),
    )
    monitor.register_components(vision_module=vision)

    await monitor._check_vision_module()

    health = monitor.get_component_health("vision")
    assert health is not None
    assert health.status == "degraded"
    assert health.metrics["frames_processed"].value == 123
    assert health.metrics["frames_dropped"].healthy is False
    assert health.metrics["balls_accuracy"].healthy is True
    assert health.metrics["cue_accuracy"].healthy is False
    assert health.metrics["camera_health"].value == "streaming"
    assert health.metrics["camera_health"].healthy is True
    assert health.warnings == [
        "Performance issues detected",
        "Recent error: one recoverable read error",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_config_check_reports_validation_and_missing_watched_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(monkeypatch, **{"api.health_monitor.max_warnings": 1})
    monitor = HealthMonitor()

    class FileStub:
        def __init__(self, accessible: bool) -> None:
            self.accessible = accessible

        def exists(self) -> bool:
            return self.accessible

    class ConfigFake:
        def get(self, key: str, default: str) -> str:
            assert key == "app.name"
            assert default == "test"
            return "billiards"

        def validate(self) -> tuple[bool, list[str]]:
            return False, ["invalid camera mode", "invalid model path"]

        def is_hot_reload_enabled(self) -> bool:
            return True

        def get_watched_files(self) -> list[Path]:
            return [FileStub(True), FileStub(False)]  # type: ignore[list-item]

    monitor.register_components(config_module=ConfigFake())
    await monitor._check_config_module()

    health = monitor.get_component_health("config")
    assert health is not None
    assert health.status == "degraded"
    assert health.metrics["config_accessible"].healthy is True
    assert health.metrics["config_valid"].healthy is False
    assert health.metrics["hot_reload_enabled"].value is True
    assert health.metrics["config_files_accessible"].value == "1/2"
    assert health.metrics["config_files_accessible"].details == {
        "total": 2,
        "accessible": 1,
    }
    assert health.warnings == [
        "invalid camera mode",
        "Some configuration files not accessible",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_check_counts_connected_and_error_sessions() -> None:
    monitor = HealthMonitor()

    class WebSocketFake:
        sessions = {
            "connected": SimpleNamespace(
                connection_state=SimpleNamespace(value="connected")
            ),
            "failed": SimpleNamespace(connection_state=SimpleNamespace(value="error")),
            "closed": SimpleNamespace(connection_state=SimpleNamespace(value="closed")),
        }

        async def get_all_sessions(self) -> dict[str, Any]:
            return {
                "total_sessions": 3,
                "total_users": 2,
                "stream_subscribers": {"state": 2, "video": 1},
                "sessions": [
                    {"connection_state": "connected"},
                    {"connection_state": "closed"},
                ],
            }

    monitor.register_components(websocket_manager=WebSocketFake())
    await monitor._check_websocket_manager()

    health = monitor.get_component_health("websocket")
    assert health is not None
    assert health.status == "degraded"
    assert health.metrics["total_sessions"].value == 3
    assert health.metrics["active_sessions"].value == 1
    assert health.metrics["active_sessions"].details == {"total": 2, "active": 1}
    assert health.metrics["state_subscribers"].value == 2
    assert health.metrics["error_sessions"].value == 1
    assert health.warnings == ["1 sessions in error state"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_callback_failure_does_not_skip_later_health_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_config(monkeypatch)
    monitor = HealthMonitor()
    delivered: list[str] = []

    async def broken_callback(_health: Any) -> None:
        raise RuntimeError("async consumer failed")

    def succeeding_callback(_health: Any) -> None:
        delivered.append("received")

    monitor.add_health_callback(broken_callback)
    monitor.add_health_callback(succeeding_callback)
    await monitor._notify_health_callbacks()

    assert delivered == ["received"]
