"""Regression tests for scale-aware Core serialization boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest

from backend.core import CoreModule, CoreModuleConfig, PerformanceMetrics
from backend.core.coordinates import Vector2D
from backend.core.events.handlers import CoreEventHandlers
from backend.core.events.manager import EventType
from backend.core.game_state import GameStateManager
from backend.core.integration import (
    CoreModuleIntegrator,
    ProjectorInterfaceImpl,
    VisionInterfaceImpl,
)
from backend.core.models import BallState, CueState, GameState, GameType, TableState
from backend.core.physics.trajectory import (
    CollisionType,
    PredictedCollision,
    Trajectory,
    TrajectoryBranch,
    TrajectoryCalculator,
    TrajectoryPoint,
)


def _table() -> TableState:
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D(x, y, scale=(1.0, 1.0))
            for x, y in (
                (100, 100),
                (1700, 100),
                (3300, 100),
                (100, 1700),
                (1700, 1700),
                (3300, 1700),
            )
        ],
        playing_area_corners=[
            Vector2D(x, y, scale=(1.0, 1.0))
            for x, y in ((100, 100), (3300, 100), (3300, 1700), (100, 1700))
        ],
        real_width_meters=2.54,
        real_height_meters=1.27,
    )


def _ball(ball_id: str, x: float, y: float, *, scale: float = 1.0) -> BallState:
    return BallState(
        id=ball_id,
        position=Vector2D(x, y, scale=(scale, scale)),
        velocity=Vector2D(3, 4, scale=(scale, scale)),
        spin=Vector2D(1, 2, scale=(scale, scale)),
        radius=18,
        radius_scale=(scale, scale),
        number=1,
    )


def _assert_scaled_point(point: dict[str, Any]) -> None:
    assert {"x", "y", "scale"}.issubset(point)
    assert point["scale"] == [1.0, 1.0]


def test_module_integrator_serializes_ball_scale_and_units() -> None:
    integrator = object.__new__(CoreModuleIntegrator)

    payload = integrator._serialize_ball(_ball("ball-1", 480, 270, scale=2))

    assert payload["position"] == {"x": 960.0, "y": 540.0, "scale": [1.0, 1.0]}
    assert payload["velocity"] == {"x": 6.0, "y": 8.0, "scale": [1.0, 1.0]}
    assert payload["spin"]["scale"] == [2.0, 2.0]
    assert payload["radius_scale"] == [1.0, 1.0]
    assert payload["radius_unit"] == "canonical_pixels"
    assert payload["mass_unit"] == "kg"
    assert payload["position_unit"] == "canonical_pixels"
    assert payload["velocity_unit"] == "canonical_pixels_per_second"


class _RecordingEventManager:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def subscribe_to_events(self, event_type: str, callback: Any) -> None:
        self.subscriptions.append((event_type, callback))

    def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.emitted.append((event_type, data))


def test_event_handler_ball_cue_and_collision_serialization_are_scaled() -> None:
    event_manager = _RecordingEventManager()
    handlers = cast("Any", CoreEventHandlers)(event_manager)
    first = _ball("first", 400, 400)
    second = _ball("second", 430, 400)
    cue = CueState(
        tip_position=Vector2D(300, 400, scale=(1.0, 1.0)),
        impact_point=Vector2D(390, 400, scale=(1.0, 1.0)),
        angle=0,
    )

    assert handlers._serialize_ball(first)["position"]["scale"] == [1.0, 1.0]
    cue_payload = handlers._serialize_cue(cue)
    assert cue_payload["tip_position"]["scale"] == [1.0, 1.0]
    assert cue_payload["impact_point"]["scale"] == [1.0, 1.0]
    assert cue_payload["length_unit"] == "meters"
    assert cue_payload["tip_position_unit"] == "canonical_pixels"

    state = GameState(
        timestamp=1.0,
        frame_number=1,
        balls=[first, second],
        table=_table(),
        game_type=GameType.PRACTICE,
    )
    state_payload = handlers._serialize_state(state)
    serialized_table = state_payload["table"]
    assert serialized_table["dimensions_scale"] == [1.0, 1.0]
    assert serialized_table["dimensions_unit"] == "canonical_pixels"
    assert serialized_table["real_world_dimensions"]["unit"] == "meters"
    assert all(
        corner["scale"] == [1.0, 1.0]
        for corner in serialized_table["playing_area_corners"]
    )
    assert state_payload["balls"][0]["radius_unit"] == "canonical_pixels"
    handlers._analyze_state_for_events(state)

    collision = next(
        data
        for event_type, data in event_manager.emitted
        if event_type == EventType.COLLISION_DETECTED.value
    )
    _assert_scaled_point(collision["position"])
    assert collision["position"]["x"] == pytest.approx(415)
    assert collision["position_unit"] == "canonical_pixels"

    handlers.collision_history = []
    handlers.handle_collision_event(
        EventType.COLLISION_DETECTED.value,
        {
            "ball1_id": "first",
            "ball2_id": "second",
            "position": {"x": 415, "y": 400, "scale": [1.0, 1.0]},
        },
    )
    assert handlers.collision_history[0]["position"] == {
        "x": 415,
        "y": 400,
        "scale": [1.0, 1.0],
    }
    assert handlers.collision_history[0]["position_unit"] == "canonical_pixels"
    with pytest.raises(ValueError, match="Scale is required"):
        handlers.handle_collision_event(
            EventType.COLLISION_DETECTED.value,
            {"position": {"x": 415, "y": 400}},
        )


def test_vision_adapter_requires_scaled_coordinates_and_scales_velocity() -> None:
    vision = object.__new__(VisionInterfaceImpl)
    vision.detection_parameters = {
        "tracking_enabled": True,
        "confidence_threshold": 0.0,
    }
    valid_detection = {
        "id": "ball",
        "position": {"x": 100, "y": 200, "scale": [2.0, 2.0]},
        "confidence": 1.0,
    }

    assert vision._validate_ball_detection(valid_detection)
    processed = vision._process_ball_detection(valid_detection)
    assert processed["velocity"] == {
        "x": 0.0,
        "y": 0.0,
        "scale": [2.0, 2.0],
        "unit": "pixels_per_second",
    }
    assert not vision._validate_ball_detection(
        {**valid_detection, "position": {"x": 100, "y": 200}}
    )


def test_projector_overlays_keep_points_scaled_and_mark_pixel_extents() -> None:
    projector = object.__new__(ProjectorInterfaceImpl)
    projector.projection_settings = {
        "resolution": [1920, 1080],
        "show_trajectories": True,
        "show_assistance": True,
        "show_aim_lines": True,
        "show_ball_ids": True,
    }
    trajectory = {
        "ball_id": "cue",
        "coordinate_space": "4k_canonical",
        "coordinate_unit": "canonical_pixels",
        "points": [
            {"time": 0, "x": 100, "y": 200, "scale": [1.0, 1.0]},
            {"time": 1, "x": 110, "y": 200, "scale": [1.0, 1.0]},
        ],
        "collisions": [
            {
                "position": {"x": 110, "y": 200, "scale": [1.0, 1.0]},
                "position_unit": "canonical_pixels",
            }
        ],
        "confidence": 1.0,
    }

    # Without camera/table and projector calibration transforms, drawing is
    # intentionally suppressed instead of forwarding camera pixels as though
    # they were in projector space.
    assert projector._process_trajectories({"trajectories": [trajectory]}) == []
    assert projector._create_trajectory_overlay([])["elements"] == []

    analysis = projector._create_shot_analysis_overlay({"success_probability": 0.5})
    default_position = analysis["elements"][0]["position"]
    assert default_position == {
        "x": 960.0,
        "y": 540.0,
        "scale": [2.0, 2.0],
        "coordinate_space": "projector_pixels",
    }

    aim = projector._create_aim_assistance_overlay(
        {"aim_line": {"start": [0, 0], "end": [1920, 1080]}}
    )
    assert aim["elements"][0]["start"]["scale"] == [2.0, 2.0]
    assert aim["elements"][0]["end"]["scale"] == [2.0, 2.0]

    settings = projector._projection_settings_payload()
    assert settings["resolution_unit"] == "pixels"
    assert settings["resolution_scale"] == [2.0, 2.0]
    calibration = projector._process_calibration_points(
        [
            {
                "screen_x": 100,
                "screen_y": 200,
                "screen_scale": [2.0, 2.0],
                "world_x": 0.1,
                "world_y": 0.2,
                "world_unit": "meters",
                "world_origin": "playing_area_top_left",
            }
        ]
    )[0]
    assert calibration["screen_x"] == 200
    assert calibration["screen_y"] == 400
    assert calibration["screen_scale"] == [1.0, 1.0]
    assert calibration["source_screen_position"]["scale"] == [2.0, 2.0]
    assert calibration["world_unit"] == "meters"

    assert projector._convert_to_screen_coordinates(cast("Any", [[10, 20]])) == []


def _projector_with_calibration() -> ProjectorInterfaceImpl:
    projector = ProjectorInterfaceImpl(event_manager=cast("Any", object()))
    projector.projection_settings["resolution"] = [1920, 1080]
    raw_points = (
        ((50, 40), (0.0, 0.0)),
        ((910, 60), (2.54, 0.0)),
        ((880, 495), (2.54, 1.27)),
        ((75, 465), (0.0, 1.27)),
    )
    points = [
        {
            "screen_position": {
                "x": screen[0],
                "y": screen[1],
                "scale": [2.0, 2.0],
            },
            "world_position": {
                "x": world[0],
                "y": world[1],
                "unit": "meters",
                "origin": "playing_area_top_left",
            },
        }
        for screen, world in raw_points
    ]
    assert projector.calibrate_projector(points)
    return projector


def test_projector_camera_transform_is_resolution_invariant_and_projective() -> None:
    projector = _projector_with_calibration()
    table_context = {
        "coordinate_space": "4k_canonical",
        "coordinate_unit": "canonical_pixels",
        "table_meter_origin": "playing_area_top_left",
        "camera_to_table_meters": _table_pixel_to_meter_matrix(),
    }
    canonical = projector._convert_to_screen_coordinates(
        [{"x": 1700, "y": 900, "scale": [1.0, 1.0]}],
        coordinate_space="4k_canonical",
        coordinate_unit="canonical_pixels",
        table_geometry=table_context,
    )
    source_1080 = projector._convert_to_screen_coordinates(
        [{"x": 850, "y": 450, "scale": [2.0, 2.0]}],
        coordinate_space="4k_canonical",
        coordinate_unit="canonical_pixels",
        table_geometry=table_context,
    )

    assert len(canonical) == len(source_1080) == 1
    assert source_1080[0]["x"] == pytest.approx(canonical[0]["x"])
    assert source_1080[0]["y"] == pytest.approx(canonical[0]["y"])
    assert canonical[0]["scale"] == [2.0, 2.0]
    assert canonical[0]["coordinate_space"] == "projector_pixels"
    # Check the computed screen coordinates against the projective calibration
    # using the known physical table center.
    projector_to_meters = np.asarray(
        projector.projection_settings["transformation_matrix"], dtype=np.float32
    )
    expected_canonical = cv2.perspectiveTransform(
        np.asarray([[[1.27, 0.635]]], dtype=np.float32),
        np.linalg.inv(projector_to_meters),
    )[0, 0]
    assert canonical[0]["x"] == pytest.approx(float(expected_canonical[0]) / 2.0)
    assert canonical[0]["y"] == pytest.approx(float(expected_canonical[1]) / 2.0)
    # The transform is perspective and output is native projector pixels with
    # its scale tag, so downstream multiplication returns canonical once.
    assert 0 < canonical[0]["x"] < 1920
    assert 0 < canonical[0]["y"] < 1080

    direct_meters = projector._convert_to_screen_coordinates(
        [{"x": 1.27, "y": 0.635, "scale": [1.0, 1.0]}],
        coordinate_space="table_meters",
        coordinate_unit="meters",
        table_geometry={"table_meter_origin": "playing_area_top_left"},
    )
    assert direct_meters[0]["x"] == pytest.approx(canonical[0]["x"])
    assert direct_meters[0]["y"] == pytest.approx(canonical[0]["y"])
    assert (
        projector._convert_to_screen_coordinates(
            [{"x": 1.27, "y": 0.635, "scale": [1.0, 1.0]}],
            coordinate_space="table_meters",
            coordinate_unit=None,
            table_geometry={"table_meter_origin": "playing_area_top_left"},
        )
        == []
    )


def _table_pixel_to_meter_matrix() -> tuple[tuple[float, ...], ...]:
    from backend.core.integration import _table_pixel_to_meter_homography

    matrix = _table_pixel_to_meter_homography(_table())
    assert matrix is not None
    return tuple(tuple(float(value) for value in row) for row in matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
    ],
)
def test_projector_transform_fails_closed_for_missing_identity_or_singular_matrix(
    matrix: Any,
) -> None:
    projector = object.__new__(ProjectorInterfaceImpl)
    projector.projection_settings = {
        "resolution": [1920, 1080],
        "transformation_matrix": matrix,
    }
    projector.calibration_status = {"is_calibrated": matrix is not None}
    points = [{"x": 100, "y": 200, "scale": [1.0, 1.0]}]

    assert (
        projector._convert_to_screen_coordinates(
            points,
            coordinate_space="4k_canonical",
            coordinate_unit="canonical_pixels",
            table_geometry={
                "coordinate_space": "4k_canonical",
                "coordinate_unit": "canonical_pixels",
                "table_meter_origin": "playing_area_top_left",
                "camera_to_table_meters": _table_pixel_to_meter_matrix(),
            },
        )
        == []
    )


def test_projector_calibration_requires_scaled_pixels_and_explicit_world_contract() -> (
    None
):
    projector = _projector_with_calibration()

    with pytest.raises(ValueError, match="explicit scale"):
        projector._process_calibration_points(
            [
                {
                    "screen_x": 100,
                    "screen_y": 200,
                    "world_x": 0.1,
                    "world_y": 0.2,
                    "world_unit": "meters",
                    "world_origin": "playing_area_top_left",
                }
            ]
        )
    with pytest.raises(ValueError, match="unit meters"):
        projector._process_calibration_points(
            [
                {
                    "screen_x": 100,
                    "screen_y": 200,
                    "screen_scale": [2.0, 2.0],
                    "world_x": 0.1,
                    "world_y": 0.2,
                    "world_origin": "playing_area_top_left",
                }
            ]
        )


def test_projector_calibration_rejects_identity_and_missing_table_geometry() -> None:
    projector = ProjectorInterfaceImpl(event_manager=cast("Any", object()))
    identity_points = [
        {
            "screen_position": {"x": x, "y": y, "scale": [1.0, 1.0]},
            "world_position": {
                "x": x,
                "y": y,
                "unit": "meters",
                "origin": "playing_area_top_left",
            },
        }
        for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]

    assert not projector.calibrate_projector(identity_points)
    assert "transformation_matrix" not in projector.projection_settings
    assert (
        projector._convert_to_screen_coordinates(
            [{"x": 10, "y": 20, "scale": [1.0, 1.0]}],
            coordinate_space="4k_canonical",
            coordinate_unit="canonical_pixels",
            table_geometry=None,
        )
        == []
    )


def test_projector_does_not_use_stale_matrix_after_calibration_is_invalidated() -> None:
    projector = _projector_with_calibration()
    projector.calibration_status["is_calibrated"] = False

    assert (
        projector._convert_to_screen_coordinates(
            [{"x": 1700, "y": 900, "scale": [1.0, 1.0]}],
            coordinate_space="4k_canonical",
            coordinate_unit="canonical_pixels",
            table_geometry={
                "coordinate_space": "4k_canonical",
                "coordinate_unit": "canonical_pixels",
                "table_meter_origin": "playing_area_top_left",
                "camera_to_table_meters": _table_pixel_to_meter_matrix(),
            },
        )
        == []
    )


def test_send_trajectory_update_attaches_detached_table_geometry_context() -> None:
    table = _table()
    captured: list[dict[str, Any]] = []
    integrator = object.__new__(CoreModuleIntegrator)
    integrator.game_state_manager = cast(
        "Any", SimpleNamespace(get_current_state=lambda: SimpleNamespace(table=table))
    )
    integrator.event_manager = cast(
        "Any", SimpleNamespace(provide_trajectory_to_projector=captured.append)
    )
    trajectory = cast("Any", SimpleNamespace(to_dict=lambda: {"points": []}))

    integrator.send_trajectory_update([trajectory])
    context = captured[0]["table_coordinate_context"]

    assert context["coordinate_space"] == "4k_canonical"
    assert context["coordinate_unit"] == "canonical_pixels"
    assert context["table_meter_origin"] == "playing_area_top_left"
    assert context["real_width_meters"] == 2.54
    assert isinstance(context["playing_area_corners"], tuple)
    assert isinstance(context["camera_to_table_meters"], tuple)


def test_table_transform_is_unavailable_without_dimensions_or_corners() -> None:
    from backend.core.integration import _table_pixel_to_meter_homography

    assert _table_pixel_to_meter_homography(SimpleNamespace()) is None
    assert (
        _table_pixel_to_meter_homography(
            SimpleNamespace(
                playing_area_corners=_table().playing_area_corners,
                real_width_meters=2.54,
                real_height_meters=None,
            )
        )
        is None
    )


@pytest.mark.parametrize(
    ("context", "tip_position", "expected_position"),
    [
        (
            {"source_resolution": [1920, 1080]},
            {"x": 480, "y": 270},
            (960, 540),
        ),
        (
            {},
            {"x": 480, "y": 270, "scale": [2.0, 2.0]},
            (960, 540),
        ),
    ],
)
def test_nested_cue_tip_position_uses_explicit_coordinate_context(
    context: dict[str, Any],
    tip_position: dict[str, Any],
    expected_position: tuple[int, int],
) -> None:
    manager = object.__new__(GameStateManager)

    cue = manager._extract_cue_state(
        {
            **context,
            "cue": {"tip_position": tip_position, "angle": 0},
        }
    )

    assert cue is not None
    assert cue.tip_position.x == pytest.approx(expected_position[0])
    assert cue.tip_position.y == pytest.approx(expected_position[1])


def test_nested_cue_tip_position_without_scale_or_resolution_is_rejected() -> None:
    manager = object.__new__(GameStateManager)
    with pytest.raises(ValueError, match="missing coordinate scale metadata"):
        manager._extract_cue_state(
            {"cue": {"tip_position": {"x": 480, "y": 270}, "angle": 0}}
        )


def test_trajectory_export_scales_every_point_endpoint_and_collision() -> None:
    calculator = TrajectoryCalculator()
    initial_ball = _ball("cue", 960, 1080)
    point_a = TrajectoryPoint(
        time=0.0,
        position=Vector2D(960, 1080, scale=(1.0, 1.0)),
        velocity=Vector2D(20, 0, scale=(1.0, 1.0)),
        acceleration=Vector2D(0, 0, scale=(1.0, 1.0)),
        spin=Vector2D(0, 0, scale=(1.0, 1.0)),
        energy=1.0,
    )
    point_b = TrajectoryPoint(
        time=0.1,
        position=Vector2D(962, 1080, scale=(1.0, 1.0)),
        velocity=Vector2D(19, 0, scale=(1.0, 1.0)),
        acceleration=Vector2D(-1, 0, scale=(1.0, 1.0)),
        spin=Vector2D(0, 0, scale=(1.0, 1.0)),
        energy=0.9,
    )
    collision = PredictedCollision(
        time=0.1,
        position=Vector2D(962, 1080, scale=(1.0, 1.0)),
        type=CollisionType.BALL_CUSHION,
        ball1_id="cue",
        ball2_id=None,
        impact_angle=0.0,
        impact_velocity=19.0,
        resulting_velocities={},
        confidence=0.9,
    )
    trajectory = Trajectory(
        ball_id="cue",
        initial_state=initial_ball,
        points=[point_a, point_b],
        collisions=[collision],
    )

    payload = calculator.export_visualization_data(trajectory)

    assert payload["coordinate_space"] == "4k_canonical"
    assert payload["coordinate_unit"] == "canonical_pixels"
    assert payload["coordinate_resolution"] == [3840, 2160]
    assert "points" not in payload
    assert len(payload["lines"]) == 1
    for line in payload["lines"]:
        _assert_scaled_point(line["start"])
        _assert_scaled_point(line["end"])
        assert line["coordinate_unit"] == "canonical_pixels"
    assert not any(
        key in payload["lines"][0] for key in ("x", "y", "vx", "vy", "scale")
    )
    _assert_scaled_point(payload["collisions"][0]["position"])
    collision_payload = payload["collisions"][0]
    assert collision_payload["position_unit"] == "canonical_pixels"
    assert collision_payload["ball1_id"] == "cue"
    assert collision_payload["ball2_id"] is None
    assert not any(key in collision_payload for key in ("x", "y", "scale", "ball_id"))


def test_projector_consumes_trajectory_lines_through_calibrated_transform() -> None:
    calculator = TrajectoryCalculator()
    initial_ball = _ball("cue", 100, 100)
    points = [
        TrajectoryPoint(
            time=time,
            position=Vector2D(x, y, scale=(1.0, 1.0)),
            velocity=Vector2D(20, 0, scale=(1.0, 1.0)),
            acceleration=Vector2D(0, 0, scale=(1.0, 1.0)),
            spin=Vector2D(0, 0, scale=(1.0, 1.0)),
            energy=1.0,
        )
        for time, x, y in ((0.0, 100, 100), (0.1, 500, 300), (0.2, 900, 600))
    ]
    branch = TrajectoryBranch(
        probability=0.2,
        description="alternate line",
        points=points[:2],
        collisions=[],
        final_state=initial_ball,
        success_metrics={"ball_id": "target"},
    )
    exported = calculator.export_visualization_data(
        Trajectory(
            ball_id="cue",
            initial_state=initial_ball,
            points=points,
            branches=[branch],
        )
    )
    assert "points" not in exported
    assert "points" not in exported["alternatives"][0]
    assert exported["alternatives"][0]["ball_id"] == "target"
    assert exported["alternatives"][0]["lines"][0]["start"]["scale"] == [1.0, 1.0]
    assert exported["alternatives"][0]["lines"][0]["end"]["scale"] == [1.0, 1.0]

    table_context = {
        "coordinate_space": "4k_canonical",
        "coordinate_unit": "canonical_pixels",
        "table_meter_origin": "playing_area_top_left",
        "camera_to_table_meters": _table_pixel_to_meter_matrix(),
    }
    projector = _projector_with_calibration()
    processed = projector._process_trajectories(
        {
            "trajectories": [exported],
            "table_coordinate_context": table_context,
        }
    )

    assert len(processed) == 1
    assert len(processed[0]["points"]) == len(points)
    expected = projector._convert_to_screen_coordinates(
        [
            exported["lines"][0]["start"],
            *(line["end"] for line in exported["lines"]),
        ],
        coordinate_space="4k_canonical",
        coordinate_unit="canonical_pixels",
        table_geometry=table_context,
    )
    assert processed[0]["points"] == expected
    assert all(point["coordinate_space"] == "projector_pixels" for point in expected)


@dataclass
class _Cache:
    value: Any = None

    def get(self, _key: str) -> Any:
        return self.value

    def set(self, _key: str, value: Any) -> None:
        self.value = value


class _FakePhysicsEngine:
    def __init__(self, points: list[Any]) -> None:
        self.points = points

    def calculate_trajectory(self, *_args: Any) -> list[Any]:
        return self.points


@pytest.mark.asyncio
async def test_core_trajectory_event_serializes_scaled_geometry() -> None:
    core = cast("Any", object.__new__(CoreModule))
    ball = _ball("cue", 960, 1080)
    core.config = CoreModuleConfig(async_processing=False, max_trajectory_time=0.1)
    core.metrics = PerformanceMetrics()
    core._current_state = GameState(
        timestamp=1.0,
        frame_number=1,
        balls=[ball],
        table=_table(),
        game_type=GameType.PRACTICE,
    )
    core.trajectory_cache = _Cache()
    core.physics_engine = _FakePhysicsEngine(
        [
            SimpleNamespace(
                position=Vector2D(960, 1080, scale=(1.0, 1.0)),
                time=0.0,
                collision=False,
            ),
            SimpleNamespace(
                position=Vector2D(962, 1080, scale=(1.0, 1.0)),
                time=0.1,
                collision=True,
            ),
        ]
    )
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def capture_event(event_type: str, data: dict[str, Any]) -> None:
        emitted.append((event_type, data))

    core._emit_event = capture_event

    await core.calculate_trajectory(
        "cue", Vector2D(10, 0, scale=(1.0, 1.0)), time_limit=0.1
    )

    event = next(data for name, data in emitted if name == "trajectory_calculated")
    trajectory = event["trajectory"]
    assert trajectory["coordinate_space"] == "4k_canonical"
    for point in trajectory["points"]:
        _assert_scaled_point(point)
    _assert_scaled_point(trajectory["lines"][0]["start"])
    _assert_scaled_point(trajectory["lines"][0]["end"])
    _assert_scaled_point(trajectory["collisions"][0]["position"])
