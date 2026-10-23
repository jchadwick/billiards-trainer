"""Focused contracts for API assembly and startup/shutdown lifecycle behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

from backend.api import main as api_main
from backend.api.dependencies import ApplicationState
from backend.perception.provider import PerceptionProvider, PerceptionProviderError
from backend.tests.support.compat_api import create_compatibility_app


class _HealthMonitorStub:
    def __init__(self) -> None:
        self.registered: dict[str, Any] | None = None
        self.started_with: float | None = None
        self.stop_monitoring = AsyncMock()

    def register_components(self, **components: Any) -> None:
        self.registered = components

    async def start_monitoring(self, check_interval: float) -> None:
        self.started_with = check_interval


class _CacheStub:
    def __init__(self) -> None:
        self.clear = Mock()


class _DottedConfig:
    """Small config reader matching the production singleton's dotted lookup."""

    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.values
        for component in key.split("."):
            if not isinstance(value, dict) or component not in value:
                return default
            value = value[component]
        return value


def _install_lifecycle_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ApplicationState, _HealthMonitorStub, list[str]]:
    """Replace shared lifecycle dependencies so lifespan tests need no hardware."""
    state = ApplicationState()
    health = _HealthMonitorStub()
    registered_shutdowns: list[str] = []

    async def register_shutdown(name: str, _callback: Any) -> None:
        registered_shutdowns.append(name)

    monkeypatch.setattr(api_main, "app_state", state)
    monkeypatch.setattr(api_main, "health_monitor", health)
    monkeypatch.setattr(
        api_main,
        "rate_limiter",
        SimpleNamespace(initialize=AsyncMock(return_value=True), close=AsyncMock()),
    )
    monkeypatch.setattr(api_main, "initialize_websocket_system", AsyncMock())
    monkeypatch.setattr(api_main, "shutdown_websocket_system", AsyncMock())
    monkeypatch.setattr(api_main, "register_module_for_shutdown", register_shutdown)
    monkeypatch.setattr(api_main, "setup_signal_handlers", lambda: None)
    monkeypatch.setattr(
        api_main,
        "_create_rust_game_session_authority",
        lambda _configuration, _settings: object(),
    )
    return state, health, registered_shutdowns


def test_runtime_configuration_prefers_overrides_and_returns_isolated_values() -> None:
    base = _DottedConfig({"service": {"items": ["saved"], "timeout": 8}})
    overrides = {"service": {"items": ["override"]}}
    runtime = api_main._RuntimeConfiguration(base, overrides)

    result = runtime.get("service.items")
    result.append("caller mutation")

    assert runtime.get("service.items") == ["override"]
    assert runtime.get("service.timeout") == 8
    assert runtime.get("service.missing", 42) == 42
    assert base.values == {"service": {"items": ["saved"], "timeout": 8}}
    assert overrides == {"service": {"items": ["override"]}}


def test_runtime_configuration_overlays_nested_values_in_full_snapshot() -> None:
    class SnapshotConfig(_DottedConfig):
        def get_all(self) -> dict[str, Any]:
            return {"service": {"items": ["saved"], "timeout": 8}}

    runtime = api_main._RuntimeConfiguration(
        SnapshotConfig({}), {"service": {"items": ["override"]}}
    )

    assert runtime.get_all() == {"service": {"items": ["override"], "timeout": 8}}


def test_missing_session_authority_does_not_block_scene_runtime_startup(
    tmp_path: Any,
) -> None:
    configuration = _DottedConfig(
        {
            "game_session": {
                "store_path": "game_session.json",
                "maximum_store_file_bytes": 4096,
                "maximum_request_bytes": 4096,
                "maximum_response_bytes": 8192,
                "lock_timeout_ms": 2000,
                "operation_timeout_seconds": 5.0,
            }
        }
    )
    settings = SimpleNamespace(
        executable_path=tmp_path / "billiards-perception-live",
        project_root=tmp_path,
    )

    assert api_main._create_rust_game_session_authority(configuration, settings) is None


def test_session_authority_symlink_loop_does_not_block_scene_startup(
    tmp_path: Any,
) -> None:
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    perception_executable = binary_directory / "billiards-perception-live"
    perception_executable.write_text("", encoding="utf-8")
    authority_executable = binary_directory / "billiards-perception-game-session"
    authority_executable.symlink_to(authority_executable.name)
    configuration = _DottedConfig(
        {
            "game_session": {
                "store_path": "game_session.json",
                "maximum_store_file_bytes": 4096,
                "maximum_request_bytes": 4096,
                "maximum_response_bytes": 8192,
                "lock_timeout_ms": 2000,
                "operation_timeout_seconds": 5.0,
            }
        }
    )
    settings = SimpleNamespace(
        executable_path=perception_executable,
        project_root=tmp_path,
    )

    assert api_main._create_rust_game_session_authority(configuration, settings) is None


def test_middleware_configuration_uses_environment_specific_and_nested_values() -> None:
    middleware = api_main.get_middleware_config(
        development_mode=True,
        config={
            "api": {
                "middleware": {
                    "logging": {"excluded_paths": ["/private"], "log_body": False},
                    "tracing": {"sample_rate_development": 0.37},
                    "performance": {
                        "enable_system_metrics": False,
                        "slow_request_threshold_ms": 12,
                    },
                }
            }
        },
    )

    assert middleware["logging"].excluded_paths == {"/private"}
    assert middleware["logging"].log_body is True
    assert middleware["tracing"].sample_rate == 0.37
    assert middleware["performance"].enable_system_metrics is False
    assert middleware["performance"].slow_request_threshold_ms == 12


@pytest.mark.asyncio
async def test_rust_replay_lifespan_uses_one_publisher_and_closes_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, health, shutdown_names = _install_lifecycle_stubs(monkeypatch)
    runtime_events: list[str] = []
    publisher = object()

    class ReplaySupervisorStub:
        def __init__(self, _settings: Any, actual_publisher: Any) -> None:
            assert actual_publisher is publisher

        async def start(self) -> Any:
            runtime_events.append("started")
            return SimpleNamespace(first_observation_received=True)

        async def close(self) -> None:
            runtime_events.append("closed")

    monkeypatch.setattr(
        api_main,
        "require_api_provider",
        lambda _configuration: PerceptionProvider.RUST,
    )
    monkeypatch.setattr(
        api_main,
        "RustReplaySettings",
        SimpleNamespace(from_configuration=lambda _c: object()),
    )
    monkeypatch.setattr(api_main, "RustReplaySupervisor", ReplaySupervisorStub)
    monkeypatch.setattr(
        api_main,
        "SceneSnapshotStatePublisher",
        lambda broadcaster: publisher,
    )
    app = create_compatibility_app(
        {
            "development_mode": True,
            "perception": {"mode": "recorded_replay"},
        }
    )

    async with api_main.lifespan(app):
        assert state.is_healthy
        assert state.perception_runtime is not None
        assert state.scene_state_publisher is publisher
        assert state.core_module is None
        assert state.vision_module is None
        assert state.integration_service is None
        assert health.registered is not None
        assert health.registered["vision_module"] is None

    assert runtime_events == ["started", "closed"]
    assert state.perception_runtime is None
    assert state.scene_state_publisher is None
    assert shutdown_names == ["websocket", "config"]


@pytest.mark.asyncio
async def test_rust_live_lifespan_starts_only_rust_perception_and_closes_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, health, shutdown_names = _install_lifecycle_stubs(monkeypatch)
    publisher = object()
    settings = object()
    runtime_events: list[str] = []

    class LiveSupervisorStub:
        def __init__(self, actual_settings: Any, actual_publisher: Any) -> None:
            assert actual_settings is settings
            assert actual_publisher is publisher

        async def start(self) -> Any:
            runtime_events.append("started")
            return SimpleNamespace(ready=True)

        async def close(self) -> None:
            runtime_events.append("closed")

    monkeypatch.setattr(
        api_main,
        "RustLiveSettings",
        SimpleNamespace(from_configuration=Mock(return_value=settings)),
    )
    monkeypatch.setattr(api_main, "RustLiveSupervisor", LiveSupervisorStub)
    monkeypatch.setattr(
        api_main, "SceneSnapshotStatePublisher", lambda _broadcaster: publisher
    )
    monkeypatch.setattr("backend.vision.VisionModule", Mock())
    app = create_compatibility_app({"perception": {"provider": "rust", "mode": "live"}})

    async with api_main.lifespan(app):
        assert state.is_healthy
        assert state.perception_provider == PerceptionProvider.RUST.value
        assert state.scene_state_publisher is publisher
        assert state.perception_runtime is not None
        assert state.core_module is None
        assert state.vision_module is None
        assert state.integration_service is None
        assert health.registered is not None
        assert health.registered["vision_module"] is None
        assert health.registered["integration_service"] is None

    assert runtime_events == ["started", "closed"]
    assert state.perception_runtime is None
    assert state.scene_state_publisher is None
    assert shutdown_names == ["websocket", "config"]


@pytest.mark.asyncio
async def test_replay_without_first_observation_fails_startup_and_runs_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, health, _shutdown_names = _install_lifecycle_stubs(monkeypatch)
    closed: list[bool] = []

    class ReplaySupervisorStub:
        def __init__(self, _settings: Any, _publisher: Any) -> None:
            pass

        async def start(self) -> Any:
            return SimpleNamespace(first_observation_received=False)

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        api_main,
        "require_api_provider",
        lambda _configuration: PerceptionProvider.RUST,
    )
    monkeypatch.setattr(
        api_main,
        "RustReplaySettings",
        SimpleNamespace(from_configuration=lambda _c: object()),
    )
    monkeypatch.setattr(api_main, "RustReplaySupervisor", ReplaySupervisorStub)
    monkeypatch.setattr(api_main, "SceneSnapshotStatePublisher", lambda _b: object())
    app = create_compatibility_app(
        {
            "development_mode": True,
            "perception": {"mode": "recorded_replay"},
        }
    )

    with pytest.raises(RuntimeError, match="without a usable observation"):
        async with api_main.lifespan(app):
            pytest.fail("startup must fail before yielding")

    assert closed == [True]
    assert not state.is_healthy
    assert state.perception_runtime is None
    assert health.stop_monitoring.await_count == 1


@pytest.mark.asyncio
async def test_provider_preflight_failure_does_not_start_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _health, _shutdown_names = _install_lifecycle_stubs(monkeypatch)
    initialize_websockets = api_main.initialize_websocket_system

    monkeypatch.setattr(
        api_main,
        "require_api_provider",
        lambda _configuration: (_ for _ in ()).throw(
            PerceptionProviderError("invalid provider selection")
        ),
    )
    app = create_compatibility_app()

    with pytest.raises(PerceptionProviderError, match="invalid provider selection"):
        async with api_main.lifespan(app):
            pytest.fail("provider preflight must fail before startup")

    initialize_websockets.assert_not_awaited()
    assert not state.is_healthy
    assert state.core_module is None


@pytest.mark.asyncio
async def test_shutdown_registration_clears_core_caches_and_stops_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ApplicationState()
    registered: dict[str, Any] = {}

    async def capture_registration(name: str, callback: Any) -> None:
        registered[name] = callback

    monkeypatch.setattr(api_main, "register_module_for_shutdown", capture_registration)
    caches = [_CacheStub(), _CacheStub(), _CacheStub()]
    vision_events: list[str] = []
    vision = SimpleNamespace(
        stop_capture=lambda: vision_events.append("vision-capture"),
        camera=SimpleNamespace(stop_capture=lambda: vision_events.append("camera")),
    )
    state.core_module = SimpleNamespace(
        trajectory_cache=caches[0],
        analysis_cache=caches[1],
        collision_cache=caches[2],
        _state_lock=asyncio.Lock(),
    )
    state.vision_module = vision
    state.config_module = object()

    await api_main._register_shutdown_functions(state)

    assert set(registered) == {"websocket", "core", "config", "vision"}
    await registered["core"]()
    await registered["vision"]()

    for cache in caches:
        cache.clear.assert_called_once()
    assert vision_events == ["vision-capture", "camera"]


@pytest.mark.asyncio
async def test_shutdown_registration_always_includes_websocket_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names: list[str] = []

    async def record_registration(name: str, _callback: Any) -> None:
        names.append(name)

    monkeypatch.setattr(api_main, "register_module_for_shutdown", record_registration)
    state = ApplicationState()
    state.config_module = None

    await api_main._register_shutdown_functions(state)

    assert names == ["websocket"]
