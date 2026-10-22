"""Edge contracts for Core integration coordination and geometry conversion.

These tests pin the safe boundary behavior used when optional modules connect,
fail, or provide incomplete calibration data. They use local test doubles so
they do not start worker threads or require projector/camera hardware.
"""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest

from backend.core.coordinates import Vector2D
from backend.core.events.manager import EventType
from backend.core.integration import (
    CoreModuleIntegrator,
    ProjectorInterfaceImpl,
    VisionInterfaceImpl,
    _table_coordinate_context,
    _validated_homography,
)
from backend.core.models import GameState, GameType, TableState


def _table(
    corners: list[Vector2D] | None = None,
    *,
    width: float = 2.0,
    height: float = 1.0,
) -> TableState:
    if corners is None:
        corners = [
            Vector2D.from_4k(x, y)
            for x, y in ((320, 280), (3520, 280), (3520, 1880), (320, 1880))
        ]
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D.from_4k(x, y)
            for x, y in (
                (320, 280),
                (1920, 280),
                (3520, 280),
                (320, 1880),
                (1920, 1880),
                (3520, 1880),
            )
        ],
        playing_area_corners=corners,
        real_width_meters=width,
        real_height_meters=height,
    )


class _Coordinator:
    def __init__(self) -> None:
        self.registered: list[tuple[str, Any, list[str]]] = []

    def register_module(self, name: str, interface: Any, features: list[str]) -> None:
        self.registered.append((name, interface, features))

    def get_module_status(self, _name: str) -> None:
        return None

    def get_all_module_statuses(self) -> dict[str, Any]:
        return {}


class _Events:
    def __init__(self) -> None:
        self.coordinator = _Coordinator()
        self.subscriptions: list[tuple[str, Any]] = []
        self.configs: list[tuple[str, dict[str, Any]]] = []
        self.states: list[dict[str, Any]] = []
        self.trajectories: list[dict[str, Any]] = []
        self.projector_commands: list[dict[str, Any]] = []
        self.lifecycle: list[tuple[str, str]] = []
        self.event_history = deque(["older"], maxlen=4)
        self._lock = threading.RLock()

    def subscribe_to_events(self, event_type: str, callback: Any) -> None:
        self.subscriptions.append((event_type, callback))

    def exchange_config_with_module(
        self, module_name: str, config: dict[str, Any]
    ) -> None:
        self.configs.append((module_name, config))

    def send_state_update_to_api(self, data: dict[str, Any]) -> None:
        self.states.append(data)

    def provide_trajectory_to_projector(self, data: dict[str, Any]) -> None:
        self.trajectories.append(data)

    def send_projector_command(self, command: dict[str, Any]) -> None:
        self.projector_commands.append(command)

    def coordinate_module_lifecycle(self, action: str, module_name: str) -> None:
        self.lifecycle.append((action, module_name))

    def get_statistics(self) -> dict[str, int]:
        return {"history_count": len(self.event_history)}


class _Manager:
    def __init__(self, state: GameState | None = None) -> None:
        self.state = state
        self.validation_updates: list[tuple[bool, bool]] = []

    def update_state(self, _data: dict[str, Any]) -> GameState:
        assert self.state is not None
        return self.state

    def set_validation_config(self, *, enabled: bool, auto_correct: bool) -> None:
        self.validation_updates.append((enabled, auto_correct))

    def get_current_state(self) -> GameState | None:
        return self.state


def _integrator(
    state: GameState | None = None,
) -> tuple[CoreModuleIntegrator, _Events, _Manager]:
    events = _Events()
    manager = _Manager(state)
    return (
        CoreModuleIntegrator(cast("Any", events), cast("Any", manager)),
        events,
        manager,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        "not a matrix",
        [[1.0, 0.0], [0.0, 1.0]],
        np.full((3, 3), np.nan),
        np.zeros((3, 3)),
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        np.eye(3),
        np.eye(3) * -4.0,
    ],
)
def test_homography_validation_rejects_unusable_and_uncalibrated_matrices(
    matrix: Any,
) -> None:
    assert _validated_homography(matrix) is None


def test_homography_validation_preserves_valid_projective_mapping() -> None:
    matrix = np.array([[2.0, 0.1, 15.0], [0.2, 3.0, 20.0], [0.0002, 0.0004, 1.0]])

    result = _validated_homography(matrix)

    assert result is not None
    np.testing.assert_array_equal(result, matrix)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (True, 1.0),
        (2.0, False),
        (0.0, 1.0),
        (2.0, -1.0),
        (float("inf"), 1.0),
        (2.0, float("nan")),
        ("2", 1.0),
    ],
)
def test_table_context_rejects_invalid_physical_dimensions(
    width: Any, height: Any
) -> None:
    table = SimpleNamespace(
        playing_area_corners=_table().playing_area_corners,
        real_width_meters=width,
        real_height_meters=height,
    )

    assert _table_coordinate_context(table) is None


def test_table_context_returns_detached_canonical_geometry_and_metric_transform() -> (
    None
):
    table = _table()

    context = _table_coordinate_context(table)

    assert context is not None
    assert context["coordinate_space"] == "4k_canonical"
    assert context["coordinate_unit"] == "canonical_pixels"
    assert context["table_meter_origin"] == "playing_area_top_left"
    assert context["playing_area_corners"] == (
        (320.0, 280.0),
        (3520.0, 280.0),
        (3520.0, 1880.0),
        (320.0, 1880.0),
    )
    assert isinstance(context["camera_to_table_meters"], tuple)
    transform = np.asarray(context["camera_to_table_meters"])
    mapped = cv2.perspectiveTransform(
        np.asarray([[[320.0, 280.0], [3520.0, 1880.0]]], dtype=np.float32),
        transform.astype(np.float32),
    )
    np.testing.assert_allclose(mapped[0], [[0.0, 0.0], [2.0, 1.0]], atol=1e-5)


def test_table_context_fails_closed_on_degenerate_or_unconvertible_corners() -> None:
    repeated = [Vector2D.from_4k(400, 400)] * 4
    assert _table_coordinate_context(_table(repeated)) is None

    class BadCorner:
        def to_4k_canonical(self) -> Any:
            raise ValueError("corner conversion failed")

    malformed = SimpleNamespace(
        playing_area_corners=[BadCorner()] * 4,
        real_width_meters=2.0,
        real_height_meters=1.0,
    )
    assert _table_coordinate_context(malformed) is None
    assert _table_coordinate_context(None) is None


def test_registration_welcome_and_shutdown_cover_each_module_boundary() -> None:
    integrator, events, _manager = _integrator()
    interfaces = {name: object() for name in ("vision", "api", "projector", "config")}

    integrator.register_vision_interface(cast("Any", interfaces["vision"]))
    integrator.register_api_interface(cast("Any", interfaces["api"]))
    integrator.register_projector_interface(cast("Any", interfaces["projector"]))
    integrator.register_config_interface(cast("Any", interfaces["config"]))

    assert [entry[0] for entry in events.coordinator.registered] == [
        "vision",
        "api",
        "projector",
        "config",
    ]
    assert {entry[0]: entry[2] for entry in events.coordinator.registered} == {
        "vision": ["detection", "tracking", "calibration"],
        "api": ["websocket", "rest", "state_broadcast"],
        "projector": ["visualization", "overlay", "calibration"],
        "config": ["configuration", "persistence", "validation"],
    }

    for name in interfaces:
        integrator._handle_module_initialization(
            EventType.MODULE_INITIALIZED.value,
            {"module_name": name, "capabilities": ["ready"]},
        )
    assert [name for name, _ in events.configs] == ["vision", "api", "projector"]
    projector_welcome = events.configs[-1][1]["projection_settings"]
    assert projector_welcome["resolution_scale"] == [2.0, 2.0]
    assert projector_welcome["resolution_unit"] == "pixels"

    for name in interfaces:
        integrator._handle_module_shutdown(
            EventType.MODULE_SHUTDOWN.value, {"module_name": name}
        )
    assert integrator.connected_modules == {}
    assert (
        integrator.vision_interface,
        integrator.api_interface,
        integrator.projector_interface,
        integrator.config_interface,
    ) == (None, None, None, None)


def test_config_forwarding_isolated_between_vision_and_projector_failures() -> None:
    integrator, _events, _manager = _integrator()
    forwarded: list[dict[str, Any]] = []

    class FailingVision:
        def set_detection_parameters(self, _parameters: dict[str, Any]) -> bool:
            raise RuntimeError("vision restarting")

    class WorkingProjector:
        def update_projection_settings(self, settings: dict[str, Any]) -> None:
            forwarded.append(settings)

    class FailingProjector:
        def update_projection_settings(self, _settings: dict[str, Any]) -> None:
            raise RuntimeError("projector disconnected")

    integrator.vision_interface = cast("Any", FailingVision())
    integrator.projector_interface = cast("Any", WorkingProjector())
    integrator._forward_config_change(
        "vision", {"detection_parameters": {"confidence_threshold": 0.8}}
    )
    integrator._forward_config_change(
        "projector", {"projection_settings": {"brightness": 0.6}}
    )
    assert forwarded == [{"brightness": 0.6}]

    integrator.projector_interface = cast("Any", FailingProjector())
    integrator._forward_config_change(
        "projector", {"projection_settings": {"brightness": 0.4}}
    )
    # Failed optional modules are handled locally; the coordinator remains usable.
    assert integrator.connected_modules == {}


def test_state_broadcast_skips_prediction_fanout_for_settled_state() -> None:
    integrator, events, _manager = _integrator()
    state = GameState(
        timestamp=1.0,
        frame_number=9,
        balls=[],
        table=_table(),
        game_type=GameType.EIGHT_BALL,
    )

    integrator._broadcast_state_update(state)

    assert len(events.states) == 1
    assert events.states[0]["frame_number"] == 9
    assert events.states[0]["balls"] == []
    assert events.trajectories == []


def test_projector_drops_uncalibrated_trajectory_geometry() -> None:
    events = _Events()
    projector = ProjectorInterfaceImpl(cast("Any", events))

    projector.send_trajectory_data(
        {
            "trajectories": [
                {
                    "coordinate_space": "4k_canonical",
                    "coordinate_unit": "canonical_pixels",
                    "lines": [
                        {
                            "start": {"x": 20, "y": 30, "scale": [1.0, 1.0]},
                            "end": {"x": 40, "y": 50, "scale": [1.0, 1.0]},
                        }
                    ],
                }
            ],
            "timestamp": 2.0,
        }
    )
    projector.send_trajectory_data({"timestamp": 4.0})

    assert events.trajectories == []
    assert len(events.projector_commands) == 1
    assert events.projector_commands[0]["overlay"]["elements"] == []


def test_vision_module_parameter_failure_keeps_local_and_event_config_coherent() -> (
    None
):
    notifications: list[tuple[str, dict[str, Any]]] = []

    class ConfigSink:
        def send_config_update(self, module_name: str, config: dict[str, Any]) -> None:
            notifications.append((module_name, config))

    class ReconnectingVision:
        def update_detection_parameters(self, _parameters: dict[str, Any]) -> None:
            raise RuntimeError("device reconnecting")

    adapter = VisionInterfaceImpl(cast("Any", ConfigSink()), ReconnectingVision())
    update = {"confidence_threshold": 0.45}

    assert adapter.set_detection_parameters(update)
    assert adapter.detection_parameters["confidence_threshold"] == 0.45
    assert notifications == [("vision", {"detection_parameters": update})]

    previous = dict(adapter.detection_parameters)
    assert not adapter.set_detection_parameters({"detection_frequency": 0})
    assert adapter.detection_parameters == previous
    assert len(notifications) == 1


def test_projector_calibration_accepts_redundant_points_and_reports_metric_error() -> (
    None
):
    projector = ProjectorInterfaceImpl(cast("Any", _Events()))
    expected = ((0, 0), (1000, 0), (1000, 500), (0, 500), (500, 250))
    points = [
        {
            "screen_position": {"x": x, "y": y, "scale": [1.0, 1.0]},
            "world_position": {
                "x": x / 1000.0,
                "y": y / 500.0,
                "unit": "meters",
                "origin": "playing_area_top_left",
            },
        }
        for x, y in expected
    ]

    assert projector.calibrate_projector(points)
    status = projector.get_projection_status()["calibration"]
    assert status["is_calibrated"] is True
    assert status["calibration_error_unit"] == "meters"
    assert status["calibration_error"] == pytest.approx(0.0, abs=1e-6)


def test_projector_calibration_rejects_insufficient_and_degenerate_points() -> None:
    projector = ProjectorInterfaceImpl(cast("Any", _Events()))
    assert not projector.calibrate_projector([])

    line_points = [
        {
            "screen_position": {"x": x, "y": 100, "scale": [1.0, 1.0]},
            "world_position": {
                "x": x / 1000.0,
                "y": 0.5,
                "unit": "meters",
                "origin": "playing_area_top_left",
            },
        }
        for x in (100, 100, 100, 100)
    ]
    assert not projector.calibrate_projector(line_points)
    assert projector.calibration_status["is_calibrated"] is False


def test_calibration_conversion_logs_and_drops_malformed_source_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector = ProjectorInterfaceImpl(cast("Any", _Events()))
    projector.calibration_status["is_calibrated"] = True
    projector.projection_settings["transformation_matrix"] = [
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0],
        [0.0, 0.0, 1.0],
    ]
    table_context = {
        "coordinate_space": "4k_canonical",
        "coordinate_unit": "canonical_pixels",
        "table_meter_origin": "playing_area_top_left",
        "camera_to_table_meters": [
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        projector.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    result = projector._convert_to_screen_coordinates(
        [{"x": float("nan"), "y": 20, "scale": [1.0, 1.0]}],
        coordinate_space="4k_canonical",
        coordinate_unit="canonical_pixels",
        table_geometry=table_context,
    )

    assert result == []
    assert warnings
    assert "rejected" in warnings[0][0]


def test_projector_calibration_returns_failure_when_opencv_cannot_fit_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector = ProjectorInterfaceImpl(cast("Any", _Events()))
    points = [
        {
            "screen_position": {"x": x, "y": y, "scale": [1.0, 1.0]},
            "world_position": {
                "x": x / 1000.0,
                "y": y / 500.0,
                "unit": "meters",
                "origin": "playing_area_top_left",
            },
        }
        for x, y in ((0, 0), (1000, 0), (1000, 500), (0, 500))
    ]
    monkeypatch.setattr(
        cv2,
        "getPerspectiveTransform",
        lambda *_args: (_ for _ in ()).throw(cv2.error("fit failed")),
    )

    assert not projector.calibrate_projector(points)
    assert projector.calibration_status["is_calibrated"] is False
