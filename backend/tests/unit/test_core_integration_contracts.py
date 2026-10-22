"""Behavioral contracts for Core module integration adapters.

These tests exercise module-to-module handoff and failure isolation without
starting camera, network, or event-worker threads. The contracts matter to the
live path because integration must forward the newest complete state quickly
and must not let an optional consumer failure interrupt state production.
"""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest

from backend.core.coordinates import Vector2D
from backend.core.events.manager import EventType
from backend.core.integration import (
    CoreModuleIntegrator,
    ModuleCapabilities,
    VisionInterfaceImpl,
)
from backend.core.models import (
    BallState,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
)


def _table() -> TableState:
    return TableState(
        width=3840,
        height=2160,
        pocket_positions=[
            Vector2D(x, y, scale=(1.0, 1.0))
            for x, y in (
                (0, 0),
                (1920, 0),
                (3840, 0),
                (0, 2160),
                (1920, 2160),
                (3840, 2160),
            )
        ],
    )


def _ball(ball_id: str, *, moving: bool = False) -> BallState:
    return BallState(
        id=ball_id,
        position=Vector2D(400, 300, scale=(1.0, 1.0)),
        velocity=Vector2D(5 if moving else 0, 0, scale=(1.0, 1.0)),
        number=8,
    )


def _state(*balls: BallState) -> GameState:
    return GameState(
        timestamp=12.5,
        frame_number=41,
        balls=list(balls),
        table=_table(),
        game_type=GameType.EIGHT_BALL,
        scores={1: 2},
    )


class _Coordinator:
    def __init__(self) -> None:
        self.registered: list[tuple[str, Any, list[str]]] = []
        self.statuses = {"vision": {"status": "ready"}}

    def register_module(self, name: str, interface: Any, features: list[str]) -> None:
        self.registered.append((name, interface, features))

    def get_module_status(self, name: str) -> dict[str, Any] | None:
        return self.statuses.get(name)

    def get_all_module_statuses(self) -> dict[str, dict[str, Any]]:
        return dict(self.statuses)


class _EventManager:
    def __init__(self) -> None:
        self.coordinator = _Coordinator()
        self.subscriptions: list[tuple[str, Any]] = []
        self.configs: list[tuple[str, dict[str, Any]]] = []
        self.state_updates: list[dict[str, Any]] = []
        self.trajectories: list[dict[str, Any]] = []
        self.lifecycle: list[tuple[str, str]] = []
        self.event_history = deque(["older-event"])
        self._lock = threading.RLock()

    def subscribe_to_events(self, event_type: str, callback: Any) -> None:
        self.subscriptions.append((event_type, callback))

    def exchange_config_with_module(
        self, module_name: str, config: dict[str, Any]
    ) -> None:
        self.configs.append((module_name, config))

    def send_state_update_to_api(self, data: dict[str, Any]) -> None:
        self.state_updates.append(data)

    def provide_trajectory_to_projector(self, data: dict[str, Any]) -> None:
        self.trajectories.append(data)

    def coordinate_module_lifecycle(self, action: str, module_name: str) -> None:
        self.lifecycle.append((action, module_name))

    def get_statistics(self) -> dict[str, Any]:
        return {"history_count": len(self.event_history)}


class _GameStateManager:
    def __init__(self, state: GameState | None = None) -> None:
        self.state = state
        self.updates: list[dict[str, Any]] = []
        self.validation_config: list[tuple[bool, bool]] = []

    def update_state(self, detection_data: dict[str, Any]) -> GameState:
        self.updates.append(detection_data)
        assert self.state is not None
        return self.state

    def set_validation_config(self, *, enabled: bool, auto_correct: bool) -> None:
        self.validation_config.append((enabled, auto_correct))

    def get_current_state(self) -> GameState | None:
        return self.state


class _Vision:
    def __init__(self, calibration: dict[str, Any] | None = None) -> None:
        self.calibration = calibration or {"ready": True}
        self.parameter_updates: list[dict[str, Any]] = []
        self.raises = False

    def request_calibration(self, calibration_type: str) -> dict[str, Any]:
        if self.raises:
            raise RuntimeError("calibration unavailable")
        return {**self.calibration, "type": calibration_type}

    def set_detection_parameters(self, parameters: dict[str, Any]) -> bool:
        self.parameter_updates.append(parameters)
        return True


class _API:
    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.raises = False

    def send_state_update(self, data: dict[str, Any]) -> None:
        if self.raises:
            raise RuntimeError("client disconnected")
        self.states.append(data)

    def send_event_notification(self, data: dict[str, Any]) -> None:
        if self.raises:
            raise RuntimeError("client disconnected")
        self.events.append(data)


class _Projector:
    def __init__(self) -> None:
        self.trajectories: list[dict[str, Any]] = []
        self.overlays: list[dict[str, Any]] = []
        self.raises = False

    def send_trajectory_data(self, data: dict[str, Any]) -> None:
        if self.raises:
            raise RuntimeError("projector unavailable")
        self.trajectories.append(data)

    def send_overlay_data(self, data: dict[str, Any]) -> None:
        if self.raises:
            raise RuntimeError("projector unavailable")
        self.overlays.append(data)

    def update_projection_settings(self, settings: dict[str, Any]) -> None:
        self.overlays.append(settings)


def _integrator(
    *, state: GameState | None = None
) -> tuple[CoreModuleIntegrator, _EventManager, _GameStateManager]:
    event_manager = _EventManager()
    manager = _GameStateManager(state)
    return (
        CoreModuleIntegrator(cast("Any", event_manager), cast("Any", manager)),
        event_manager,
        manager,
    )


def test_integrator_subscribes_to_all_module_boundary_events() -> None:
    integrator, event_manager, _ = _integrator()

    callbacks = dict(event_manager.subscriptions)
    assert set(callbacks) == {
        EventType.VISION_DATA_RECEIVED.value,
        EventType.STATE_UPDATED.value,
        EventType.TRAJECTORY_CALCULATED.value,
        EventType.CONFIG_CHANGED.value,
        EventType.MODULE_INITIALIZED.value,
        EventType.MODULE_SHUTDOWN.value,
    }
    assert callbacks[EventType.VISION_DATA_RECEIVED.value].__self__ is integrator


def test_vision_observation_is_converted_once_and_latest_state_is_broadcast() -> None:
    moving = _ball("cue", moving=True)
    still = _ball("8")
    integrator, events, manager = _integrator(state=_state(moving, still))
    observation = {"frame_number": 41, "source_frame_id": "cam-41", "balls": []}

    integrator._handle_vision_data(
        EventType.VISION_DATA_RECEIVED.value, {"detection_data": observation}
    )

    assert manager.updates == [observation]
    assert len(events.state_updates) == 1
    published = events.state_updates[0]
    assert published["frame_number"] == 41
    assert [ball["id"] for ball in published["balls"]] == ["cue", "8"]
    assert published["balls"][0]["position_unit"] == "canonical_pixels"
    assert len(events.trajectories) == 1
    assert [ball["id"] for ball in events.trajectories[0]["moving_balls"]] == ["cue"]
    assert integrator.integration_stats["messages_received"] == 1
    assert integrator.integration_stats["errors"] == 0


def test_vision_failure_does_not_publish_partial_state_and_is_counted() -> None:
    event_manager = _EventManager()
    manager = SimpleNamespace(
        update_state=lambda _data: (_ for _ in ()).throw(ValueError("bad observation"))
    )
    integrator = CoreModuleIntegrator(cast("Any", event_manager), cast("Any", manager))

    integrator._handle_vision_data(
        EventType.VISION_DATA_RECEIVED.value, {"detection_data": {"frame_number": 3}}
    )

    assert event_manager.state_updates == []
    assert event_manager.trajectories == []
    assert integrator.integration_stats["messages_received"] == 0
    assert integrator.integration_stats["errors"] == 1


def test_state_and_trajectory_consumer_failures_are_isolated() -> None:
    integrator, _, _ = _integrator()
    api = _API()
    projector = _Projector()
    api.raises = True
    projector.raises = True
    integrator.api_interface = api  # type: ignore[assignment]
    integrator.projector_interface = projector  # type: ignore[assignment]

    integrator._handle_state_update(EventType.STATE_UPDATED.value, {"frame_number": 4})
    integrator._handle_trajectory_data(
        EventType.TRAJECTORY_CALCULATED.value, {"trajectory_data": {"frame": 4}}
    )

    assert integrator.integration_stats["errors"] == 2
    assert integrator.integration_stats["messages_sent"] == 0


def test_config_event_updates_core_and_forwards_vision_parameters() -> None:
    integrator, event_manager, manager = _integrator()
    vision = _Vision()
    integrator.vision_interface = vision  # type: ignore[assignment]
    event_manager.event_history = deque(["old-1", "old-2", "old-3"])

    integrator._handle_config_change(
        EventType.CONFIG_CHANGED.value,
        {
            "module_name": "core",
            "config": {
                "validation_enabled": False,
                "auto_correct_enabled": False,
                "event_history_size": 2,
            },
        },
    )
    integrator._handle_config_change(
        EventType.CONFIG_CHANGED.value,
        {
            "module_name": "vision",
            "config": {"detection_parameters": {"detection_frequency": 30}},
        },
    )

    assert manager.validation_config == [(False, False)]
    assert event_manager.event_history.maxlen == 2
    assert list(event_manager.event_history) == ["old-2", "old-3"]
    assert vision.parameter_updates == [{"detection_frequency": 30}]


def test_invalid_event_history_configuration_preserves_existing_history() -> None:
    integrator, event_manager, _ = _integrator()
    event_manager.event_history = deque(["keep"], maxlen=3)

    integrator._apply_core_configuration({"event_history_size": True})

    assert event_manager.event_history.maxlen == 3
    assert list(event_manager.event_history) == ["keep"]


def test_module_lifecycle_registers_metadata_sends_welcome_and_cleans_up() -> None:
    integrator, events, _ = _integrator()
    vision = _Vision()
    integrator.register_vision_interface(vision)  # type: ignore[arg-type]
    integrator._handle_module_initialization(
        EventType.MODULE_INITIALIZED.value,
        {
            "module_name": "vision",
            "version": "2.1",
            "capabilities": ["ball-detection"],
            "requirements": ["camera"],
            "interfaces": ["detection"],
        },
    )

    assert events.coordinator.registered[0][0] == "vision"
    assert events.configs[0][0] == "vision"
    assert events.configs[0][1]["detection_frequency"] == 60
    assert integrator.connected_modules["vision"] == ModuleCapabilities(
        name="vision",
        version="2.1",
        features=["ball-detection"],
        requirements=["camera"],
        interfaces=["detection"],
    )

    integrator._handle_module_shutdown(
        EventType.MODULE_SHUTDOWN.value, {"module_name": "vision"}
    )

    assert integrator.vision_interface is None
    assert "vision" not in integrator.connected_modules


def test_trajectory_and_shot_analysis_are_routed_with_optional_consumers() -> None:
    integrator, events, _ = _integrator(state=_state())
    api = _API()
    projector = _Projector()
    integrator.api_interface = api  # type: ignore[assignment]
    integrator.projector_interface = projector  # type: ignore[assignment]
    trajectory = SimpleNamespace(to_dict=lambda: {"ball_id": "cue", "points": []})

    integrator.send_trajectory_update([trajectory])  # type: ignore[list-item]
    integrator.send_shot_analysis(
        ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.2,
            success_probability=0.8,
            recommended_force=2.0,
            recommended_angle=30.0,
        )
    )

    assert events.trajectories[0]["trajectories"] == [{"ball_id": "cue", "points": []}]
    assert api.events[0]["event_type"] == "shot_analysis"
    assert projector.overlays[0]["type"] == "shot_analysis"
    assert projector.overlays[0]["recommended_angle"] == 30.0


def test_calibration_request_and_status_methods_cover_connected_and_missing_paths() -> (
    None
):
    integrator, _, _ = _integrator()
    assert integrator.request_vision_calibration() is None

    vision = _Vision({"revision": 9})
    integrator.vision_interface = vision  # type: ignore[assignment]
    assert integrator.request_vision_calibration("table") == {
        "revision": 9,
        "type": "table",
    }
    vision.raises = True
    assert integrator.request_vision_calibration("table") is None
    assert integrator.get_module_status("vision") == {"status": "ready"}
    assert integrator.get_all_module_statuses() == {"vision": {"status": "ready"}}

    stats = integrator.get_integration_statistics()
    assert stats["module_count"] == 0
    assert stats["event_manager_stats"] == {"history_count": 1}


def test_shutdown_notifies_connected_modules_then_releases_interfaces() -> None:
    integrator, events, _ = _integrator()
    integrator.connected_modules = {
        "vision": ModuleCapabilities("vision", "1", [], [], []),
        "api": ModuleCapabilities("api", "1", [], [], []),
    }
    integrator.vision_interface = _Vision()  # type: ignore[assignment]
    integrator.api_interface = _API()  # type: ignore[assignment]

    integrator.shutdown_integration()

    assert events.lifecycle == [("shutdown", "vision"), ("shutdown", "api")]
    assert integrator.vision_interface is None
    assert integrator.api_interface is None


def test_vision_adapter_preserves_observation_identity_and_filters_bad_ball() -> None:
    event_manager = SimpleNamespace(
        receive_vision_data=lambda data: received.append(data)
    )
    received: list[dict[str, Any]] = []
    vision = VisionInterfaceImpl(cast("Any", event_manager))
    source = {
        "timestamp": 100.0,
        "frame_number": 17,
        "source_frame_id": "camera:17",
        "balls": [
            {
                "id": "ball-8",
                "position": {"x": 10.0, "y": 20.0, "scale": [1.0, 1.0]},
                "confidence": 0.95,
            },
            {"id": "malformed", "position": {"x": 1, "y": 2}, "confidence": 1.0},
        ],
    }

    vision.receive_detection_data(source)

    assert len(received) == 1
    assert received[0]["frame_number"] == 17
    assert received[0]["source_frame_id"] == "camera:17"
    assert [ball["id"] for ball in received[0]["processed_balls"]] == ["ball-8"]
    assert received[0]["processed_balls"][0]["passes_confidence_filter"] is True
    assert received[0]["detection_quality"]["ball_count"] == 1
    assert "processed_balls" not in source


def test_vision_adapter_ignores_incomplete_frame_and_reports_empty_detection() -> None:
    received: list[dict[str, Any]] = []
    vision = VisionInterfaceImpl(
        cast(
            "Any",
            SimpleNamespace(receive_vision_data=lambda data: received.append(data)),
        )
    )

    vision.receive_detection_data({"frame_number": 1, "balls": []})
    vision.receive_detection_data({"timestamp": 5.0, "frame_number": 2, "balls": []})

    assert len(received) == 1
    assert received[0]["frame_number"] == 2
    assert received[0]["detection_quality"] == {
        "score": 0.0,
        "issues": ["No balls detected"],
    }


def test_vision_adapter_parameter_validation_and_cached_calibration(
    monkeypatch: Any,
) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    event_manager = SimpleNamespace(
        send_config_update=lambda module, config: sent.append((module, config))
    )
    vision = VisionInterfaceImpl(cast("Any", event_manager))
    monkeypatch.setattr("backend.core.integration.time.time", lambda: 5000.0)
    vision.calibration_data["table"]["timestamp"] = 4999.0

    assert vision.set_detection_parameters(
        {"confidence_threshold": 0.5, "unsupported_setting": True}
    )
    assert vision.detection_parameters["confidence_threshold"] == 0.5
    assert sent == [("vision", {"detection_parameters": {"confidence_threshold": 0.5}})]
    old_parameters = dict(vision.detection_parameters)
    assert not vision.set_detection_parameters({"confidence_threshold": 1.5})
    assert vision.detection_parameters == old_parameters

    cached = vision.request_calibration("table")
    assert cached.get("is_stale") is None
    assert vision.request_calibration("unknown") == {}

    vision.calibration_data["table"]["timestamp"] = 0.0
    assert vision.request_calibration("table")["is_stale"] is True


def test_vision_adapter_prefers_live_calibration_then_falls_back_to_cache() -> None:
    class VisionModule:
        fail = False

        def get_calibration_data(self, kind: str) -> dict[str, Any]:
            if self.fail:
                raise RuntimeError("device is restarting")
            return {"source": "live", "kind": kind}

    module = VisionModule()
    vision = VisionInterfaceImpl(cast("Any", SimpleNamespace()), module)
    assert vision.request_calibration("table") == {"source": "live", "kind": "table"}

    module.fail = True
    vision.calibration_data["table"] = {"source": "cache", "timestamp": 2000.0}
    assert vision.request_calibration("table")["source"] == "cache"


@pytest.mark.parametrize(
    "position",
    [
        {"x": True, "y": 2.0, "scale": [1.0, 1.0]},
        {"x": float("nan"), "y": 2.0, "scale": [1.0, 1.0]},
        {"x": 1.0, "y": 2.0, "scale": [0.0, 1.0]},
    ],
)
def test_vision_adapter_rejects_non_finite_or_invalid_scaled_geometry(
    position: dict[str, Any],
) -> None:
    vision = VisionInterfaceImpl(cast("Any", SimpleNamespace()))

    assert not vision._validate_ball_detection(
        {"id": "ball", "position": position, "confidence": 1.0}
    )
