"""Behavioral tests for the Core integration adapters' live paths.

The companion integration contract tests cover coordinator event wiring,
vision admission, lifecycle, and trajectory/shot routing. This module focuses
on the remaining API, configuration, and projector adapter boundaries where a
failure can silently discard or corrupt the current state delivered to clients.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, cast

import pytest

from backend.core.integration import (
    APIInterfaceImpl,
    ConfigInterfaceImpl,
    ProjectorInterfaceImpl,
)


class _EventSink:
    """Capture config exchanges and projector commands at module boundaries."""

    def __init__(self) -> None:
        self.config_updates: list[tuple[str, dict[str, Any]]] = []
        self.projector_commands: list[dict[str, Any]] = []

    def exchange_config_with_module(
        self, module_name: str, config: dict[str, Any]
    ) -> None:
        self.config_updates.append((module_name, config))

    def send_projector_command(self, command: dict[str, Any]) -> None:
        self.projector_commands.append(command)


class _WebsocketManager:
    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def broadcast_game_state(self, state: dict[str, Any]) -> None:
        self.states.append(state)

    def broadcast_alert(self, alert: dict[str, Any]) -> None:
        self.alerts.append(alert)

    def broadcast(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


def _valid_state() -> dict[str, Any]:
    return {
        "timestamp": 10.0,
        "frame_number": 12,
        "source_frame_id": "camera-run:12",
        "balls": [
            {
                "id": "cue",
                "position": {"x": 100.123456, "y": 200.654321},
                "velocity": {"x": 4.567891, "y": -2.345678},
            }
        ],
        "cue": {"id": "cue"},
        "table": {"width": 1200},
        "validation_errors": ["internal-only"],
    }


def test_api_adapter_broadcasts_current_state_with_identity_and_public_fields() -> None:
    websocket = _WebsocketManager()
    api = APIInterfaceImpl(cast("Any", object()), websocket_manager=websocket)
    state = _valid_state()
    original_state = deepcopy(state)

    api.send_state_update(state)

    assert len(websocket.states) == 1
    delivered = websocket.states[0]
    assert delivered["frame_number"] == 12
    assert delivered["source_frame_id"] == "camera-run:12"
    assert delivered["balls"][0]["position"] == {"x": 100.1235, "y": 200.6543}
    assert delivered["balls"][0]["velocity"] == {"x": 4.5679, "y": -2.3457}
    assert "validation_errors" not in delivered
    assert state == original_state
    status = api.get_connection_status()
    assert status["message_count"] == 1
    assert status["handler_count"] == 0


def test_api_adapter_drops_invalid_state_without_advancing_client_sequence() -> None:
    websocket = _WebsocketManager()
    api = APIInterfaceImpl(cast("Any", object()), websocket_manager=websocket)

    api.send_state_update({"timestamp": 10.0, "balls": []})

    assert websocket.states == []
    assert api.connection_status["message_count"] == 0
    # Invalid input consumes the configured admission attempt, preventing a
    # malformed producer from bypassing the same bounded send rate as valid data.
    assert api.rate_limiter["state_updates"]["count"] == 1


def test_api_adapter_event_priority_and_invalid_event_handling() -> None:
    class GenericWebsocketManager:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        def broadcast(self, message: dict[str, Any]) -> None:
            self.messages.append(message)

    websocket = GenericWebsocketManager()
    api = APIInterfaceImpl(cast("Any", object()), websocket_manager=websocket)

    api.send_event_notification(
        {"event_type": "collision", "timestamp": 11.0, "message": "contact"}
    )
    api.send_event_notification(
        {"event_type": "state_changed", "timestamp": 12.0, "message": "settled"}
    )
    api.send_event_notification({"event_type": "missing_timestamp"})

    assert [message["priority"] for message in websocket.messages] == [
        "high",
        "medium",
    ]
    assert [message["data"]["event_type"] for message in websocket.messages] == [
        "collision",
        "state_changed",
    ]
    assert api.connection_status["message_count"] == 2


def test_api_adapter_rate_limit_resets_on_new_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1000.0
    monkeypatch.setattr("backend.core.integration.time.time", lambda: now)
    websocket = _WebsocketManager()
    api = APIInterfaceImpl(cast("Any", object()), websocket_manager=websocket)
    api.rate_limiter["state_updates"]["limit"] = 1

    api.send_state_update(_valid_state())
    api.send_state_update(_valid_state())
    assert len(websocket.states) == 1

    now += 61.0
    api.send_state_update(_valid_state())
    assert len(websocket.states) == 2
    assert api.connection_status["message_count"] == 2


def test_api_adapter_uses_async_broadcaster_for_latest_state_and_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AsyncBroadcaster:
        def __init__(self) -> None:
            self.states: list[dict[str, Any]] = []
            self.alerts: list[dict[str, Any]] = []

        async def broadcast_game_state(self, **state: Any) -> None:
            self.states.append(state)

        async def broadcast_alert(self, **alert: Any) -> None:
            self.alerts.append(alert)

    broadcaster = AsyncBroadcaster()
    submissions: list[tuple[Any, Any]] = []

    def run_coroutine_threadsafe(coroutine: Any, loop: Any) -> object:
        submissions.append((loop, coroutine))
        return asyncio.run(coroutine)

    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", run_coroutine_threadsafe)
    api = APIInterfaceImpl(
        cast("Any", object()),
        message_broadcaster=broadcaster,
        event_loop=object(),
    )

    api.send_state_update(_valid_state())
    api.send_event_notification(
        {"event_type": "shot_started", "timestamp": 13.0, "message": "break"}
    )

    assert len(submissions) == 2
    assert len(broadcaster.states) == 1
    assert broadcaster.states[0]["balls"][0]["id"] == "cue"
    assert broadcaster.states[0]["cue"] == {"id": "cue"}
    assert len(broadcaster.alerts) == 1
    assert broadcaster.alerts[0]["message"] == "break"


def test_api_websocket_handler_tracks_usage_and_isolates_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.core.integration.time.time", lambda: 1234.0)
    api = APIInterfaceImpl(cast("Any", object()))
    observed: list[str] = []

    def handler(message: str) -> str:
        observed.append(message)
        return "handled"

    handler_id = api.register_websocket_handler(handler)

    assert api.websocket_handlers[handler_id]["handler"]("newest") == "handled"
    handler_info = api.websocket_handlers[handler_id]
    assert observed == ["newest"]
    assert handler_info["message_count"] == 1
    assert handler_info["last_used"] == 1234.0
    assert api.unregister_websocket_handler(handler_id)
    assert not api.unregister_websocket_handler(handler_id)

    failing_id = api.register_websocket_handler(
        lambda _message: (_ for _ in ()).throw(RuntimeError("consumer failed"))
    )
    assert api.websocket_handlers[failing_id]["handler"]({"frame": 2}) is None
    assert api.websocket_handlers[failing_id]["message_count"] == 1


def test_config_validation_reports_required_ranges_types_and_unknown_keys() -> None:
    config = ConfigInterfaceImpl(cast("Any", object()))

    missing = config.validate_config("core", {"event_history_size": 50})
    assert not missing["valid"]
    assert "Missing required configuration: validation_enabled" in missing["errors"]
    assert "Value event_history_size=50 out of range [100, 10000]" in missing["errors"]

    result = config.validate_config(
        "vision", {"detection_frequency": "fast", "extra": True}
    )
    assert not result["valid"]
    assert any(
        "Invalid type for detection_frequency" in error for error in result["errors"]
    )
    assert result["warnings"] == ["Unknown configuration key: extra"]
    assert config.validate_config("not-a-module", {})["errors"] == [
        "Unknown module: not-a-module"
    ]


def test_config_update_keeps_cache_history_and_notifies_when_persistence_fails() -> (
    None
):
    class ConfigStore:
        def update_module_config(
            self, _module_name: str, _config: dict[str, Any]
        ) -> None:
            raise OSError("disk temporarily unavailable")

    events = _EventSink()
    config = ConfigInterfaceImpl(cast("Any", events), ConfigStore())
    notifications: list[dict[str, Any]] = []
    failing_subscription = config.subscribe_config_changes(
        lambda _notice: (_ for _ in ()).throw(RuntimeError("subscriber offline"))
    )
    active_subscription = config.subscribe_config_changes(notifications.append)
    update = {"detection_frequency": 48, "tracking_enabled": False}

    config.update_module_config("vision", update)

    assert config.module_configs["vision"]["detection_frequency"] == 48
    assert config.module_configs["vision"]["tracking_enabled"] is False
    assert len(events.config_updates) == 1
    assert events.config_updates[0][0] == "vision"
    history = config.get_config_history("vision")
    assert history["version"] == "1.0.1"
    assert sorted(history["changes"]["modified"], key=lambda change: change["key"]) == [
        {"key": "detection_frequency", "old_value": 60, "new_value": 48},
        {"key": "tracking_enabled", "old_value": True, "new_value": False},
    ]
    assert notifications[0]["module_name"] == "vision"
    assert config.config_subscribers[failing_subscription]["notification_count"] == 1
    assert config.config_subscribers[active_subscription]["notification_count"] == 1
    assert config.unsubscribe_config_changes(active_subscription)
    assert not config.unsubscribe_config_changes(active_subscription)


def test_config_update_rejects_invalid_changes_without_mutating_current_config() -> (
    None
):
    events = _EventSink()
    config = ConfigInterfaceImpl(cast("Any", events))
    original = dict(config.module_configs["vision"])

    config.update_module_config("vision", {"detection_frequency": 0})

    assert config.module_configs["vision"] == original
    assert config.get_config_history("vision") == {}
    assert events.config_updates == []


def test_config_reads_live_data_then_falls_back_to_metadata_enriched_cache() -> None:
    class ConfigStore:
        fail = False

        def get_module_config(self, module_name: str) -> dict[str, Any]:
            if self.fail:
                raise OSError("store unavailable")
            return {"module": module_name, "source": "live"}

    store = ConfigStore()
    config = ConfigInterfaceImpl(cast("Any", _EventSink()), store)

    assert config.get_module_config("vision") == {
        "module": "vision",
        "source": "live",
    }
    store.fail = True
    cached = config.get_module_config("vision")
    assert cached["detection_frequency"] == 60
    assert cached["_metadata"] == {
        "module_name": "vision",
        "last_updated": 0.0,
        "version": "1.0.0",
        "source": "cached_config",
    }
    assert config.get_module_config("unknown") == {}


def test_config_reset_replaces_module_with_defaults_and_increments_history() -> None:
    events = _EventSink()
    config = ConfigInterfaceImpl(cast("Any", events))
    config.update_module_config("api", {"update_frequency": 30})

    assert config.reset_module_config("api")
    assert config.module_configs["api"]["update_frequency"] == 60
    assert config.get_config_history("api")["version"] == "1.0.2"
    assert not config.reset_module_config("unknown")


def test_projector_settings_filter_invalid_values_and_mark_calibration_stale() -> None:
    events = _EventSink()
    projector = ProjectorInterfaceImpl(cast("Any", events))
    projector.calibration_status["is_calibrated"] = True

    projector.update_projection_settings(
        {"brightness": 0.4, "refresh_rate": 45, "resolution": [1280, 720]}
    )

    assert projector.projection_settings["brightness"] == 0.4
    assert projector.projection_settings["refresh_rate"] == 60
    assert projector.projection_settings["resolution"] == [1280, 720]
    assert projector.calibration_status["is_calibrated"] is False
    assert events.projector_commands[-1]["requires_restart"] is True
    assert events.projector_commands[-1]["settings"]["resolution_unit"] == "pixels"

    command_count = len(events.projector_commands)
    projector.update_projection_settings({"brightness": 3.0, "contrast": 0.0})
    assert len(events.projector_commands) == command_count


def test_projector_overlay_expiry_and_priority_are_applied_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _EventSink()
    projector = ProjectorInterfaceImpl(cast("Any", events))
    now = 100.0
    monkeypatch.setattr("backend.core.integration.time.time", lambda: now)

    projector.send_overlay_data(
        {
            "type": "shot_analysis",
            "timestamp": now,
            "priority": 1,
            "duration_ms": 10_000,
            "recommended_angle": 25,
        }
    )
    now += 1.0
    projector.send_overlay_data(
        {
            "type": "aim_assistance",
            "timestamp": now,
            "priority": 9,
            "duration_ms": 100,
            "aim_line": {"start": [10, 10], "end": [20, 20]},
        }
    )

    elements = events.projector_commands[-1]["overlay"]["elements"]
    assert [element["color"] for element in elements] == ["cyan", "green"]

    now += 0.2
    projector.send_overlay_data(
        {
            "type": "ball_tracking",
            "timestamp": now,
            "duration_ms": 10_000,
        }
    )
    assert set(projector.get_projection_status()["active_overlays"]) == {
        "shot_analysis",
        "ball_tracking",
    }


def test_projector_rejects_incomplete_overlay_without_replacing_active_state() -> None:
    events = _EventSink()
    projector = ProjectorInterfaceImpl(cast("Any", events))
    projector.send_overlay_data({"type": "valid", "timestamp": 10.0, "value": 1})
    original = dict(projector.active_overlays)

    projector.send_overlay_data({"type": "missing-time", "value": 2})

    assert projector.active_overlays == original
    assert len(events.projector_commands) == 1


def test_projector_module_failure_does_not_prevent_event_fanout() -> None:
    class BrokenProjector:
        def render_overlay(self, _overlay: dict[str, Any]) -> None:
            raise RuntimeError("display disconnected")

    events = _EventSink()
    projector = ProjectorInterfaceImpl(cast("Any", events), BrokenProjector())

    projector.send_overlay_data({"type": "generic", "timestamp": 10.0})

    assert len(events.projector_commands) == 1
    assert events.projector_commands[0]["type"] == "overlay_update"


@pytest.mark.parametrize("module_name", ["core", "vision", "api", "projector"])
def test_config_default_snapshots_are_valid_for_required_schema_fields(
    module_name: str,
) -> None:
    config = ConfigInterfaceImpl(cast("Any", _EventSink()))
    defaults = config._get_default_configs()[module_name]

    assert config.validate_config(module_name, defaults)["valid"]
