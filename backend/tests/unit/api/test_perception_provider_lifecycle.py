"""The API must reject unavailable providers before starting dependencies."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.api import main
from backend.perception.provider import PerceptionProviderError


def test_rust_selection_fails_before_starting_live_python_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main.config,
        "get",
        lambda key, default=None: "rust" if key == "perception.provider" else default,
    )
    initialize_rate_limiter = AsyncMock(return_value=True)
    monkeypatch.setattr(main.rate_limiter, "initialize", initialize_rate_limiter)

    async def enter_lifespan() -> None:
        async with main.lifespan(main.app):
            pytest.fail("unsupported Rust provider unexpectedly entered lifespan")

    with pytest.raises(PerceptionProviderError, match="recorded-input replay"):
        asyncio.run(enter_lifespan())

    assert initialize_rate_limiter.await_count == 0
    assert main.app_state.is_healthy is False


def test_recorded_rust_lifespan_uses_one_publisher_and_closes_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay mode starts its own publisher without constructing Python vision."""
    app = main.create_app(
        {
            "perception": {"provider": "rust", "mode": "recorded_replay"},
        }
    )
    supervisor = SimpleNamespace(
        start=AsyncMock(return_value=SimpleNamespace(first_observation_received=True)),
        close=AsyncMock(),
    )
    publisher = object()
    settings = object()
    monkeypatch.setattr(
        main.RustReplaySettings,
        "from_configuration",
        Mock(return_value=settings),
    )
    supervisor_factory = Mock(return_value=supervisor)
    publisher_factory = Mock(return_value=publisher)
    monkeypatch.setattr(main, "RustReplaySupervisor", supervisor_factory)
    monkeypatch.setattr(main, "SceneSnapshotStatePublisher", publisher_factory)

    vision_factory = Mock(side_effect=AssertionError("Python vision must stay off"))
    integration_factory = Mock(
        side_effect=AssertionError("Python integration must stay off")
    )
    monkeypatch.setattr("backend.vision.VisionModule", vision_factory)
    monkeypatch.setattr(main, "IntegrationService", integration_factory)

    initialize_rate_limiter = AsyncMock(return_value=True)
    close_rate_limiter = AsyncMock()
    monkeypatch.setattr(main.rate_limiter, "initialize", initialize_rate_limiter)
    monkeypatch.setattr(main.rate_limiter, "close", close_rate_limiter)
    monkeypatch.setattr(main, "initialize_websocket_system", AsyncMock())
    monkeypatch.setattr(main, "shutdown_websocket_system", AsyncMock())
    monkeypatch.setattr(main, "_register_shutdown_functions", AsyncMock())
    monkeypatch.setattr(main, "setup_signal_handlers", Mock())
    monkeypatch.setattr(main.websocket_handler, "configure_performance_audit", Mock())
    health_monitor = SimpleNamespace(
        register_components=Mock(),
        start_monitoring=AsyncMock(),
        stop_monitoring=AsyncMock(),
    )
    monkeypatch.setattr(main, "health_monitor", health_monitor)
    monkeypatch.setattr(main, "CoreModule", Mock(return_value=object()))

    async def run_lifespan() -> None:
        async with main.lifespan(app):
            assert main.app_state.perception_provider == "rust"
            assert main.app_state.scene_state_publisher is publisher
            assert main.app_state.perception_runtime is supervisor
            assert main.app_state.vision_module is None
            assert main.app_state.integration_service is None
            assert supervisor.start.await_count == 1
            assert main.app_state.is_healthy is True

    try:
        asyncio.run(run_lifespan())
    finally:
        main.app_state.core_module = None
        main.app_state.perception_provider = None

    publisher_factory.assert_called_once_with(main.message_broadcaster)
    supervisor_factory.assert_called_once_with(settings, publisher)
    assert supervisor.close.await_count == 1
    assert initialize_rate_limiter.await_count == 1
    assert close_rate_limiter.await_count == 1
    vision_factory.assert_not_called()
    integration_factory.assert_not_called()
    health_monitor.register_components.assert_called_once()
