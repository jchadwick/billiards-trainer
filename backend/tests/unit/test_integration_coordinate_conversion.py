"""Cross-resolution contracts for the Vision-to-Core integration boundary."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.common.models import Ball
from backend.core.coordinates import Vector2D
from backend.core.game_state import GameStateManager
from backend.integration_service import IntegrationService
from backend.integration_service_conversion_helpers import (
    CoordinateConversionError,
    StateConversionHelpers,
)
from backend.vision.models import (
    BallType,
    CueState,
    CueStick,
    DetectionResult,
    FrameStatistics,
    Pocket,
    PocketType,
    Table,
)


def _fixture_detection(width: int, height: int, *, canonical: bool) -> DetectionResult:
    factor_x, factor_y = 3840 / width, 2160 / height
    coordinate_space = "4k_canonical" if canonical else "pixel"

    def point(x: float, y: float) -> tuple[float, float]:
        return (x / factor_x, y / factor_y)

    ball = Ball(
        position=point(1500, 1080),
        velocity=point(24, -8),
        acceleration=point(4, 2),
        radius=36 / factor_x,
        ball_type=BallType.CUE,
        confidence=0.98,
        track_id=7,
        predicted_position=point(1512, 1076),
        position_history=[point(1490, 1084)],
        coordinate_space=coordinate_space,
        source_resolution=(width, height),
    )
    cue = CueStick(
        tip_position=point(1300, 1080),
        angle=30,
        length=300 / factor_x,
        confidence=0.95,
        state=CueState.AIMING,
        tip_velocity=point(6, 2),
        butt_position=point(1000, 900),
        shaft_points=[point(1150, 990)],
        width=8 / factor_x,
        predicted_contact_point=point(1500, 1080),
        aiming_line=[point(1300, 1080), point(1500, 1080)],
        coordinate_space=coordinate_space,
        source_resolution=(width, height),
    )
    table = Table(
        corners=[
            point(200, 200),
            point(3640, 200),
            point(3640, 1960),
            point(200, 1960),
        ],
        pockets=[
            Pocket(
                position=point(200, 200),
                pocket_type=PocketType.CORNER,
                radius=72 / factor_x,
                corners=[point(164, 164), point(236, 164), point(200, 236)],
            )
        ],
        width=3440 / factor_x,
        height=1760 / factor_y,
        surface_color=(90, 140, 80),
        rails=[[point(200, 200), point(3640, 200)]],
        playing_area=[
            point(260, 260),
            point(3580, 260),
            point(3580, 1900),
            point(260, 1900),
        ],
        real_world_dimensions=(2.54, 1.27),
        pixels_per_meter=3440 / 2.54 / factor_x,
        coordinate_space=coordinate_space,
        source_resolution=(width, height),
    )
    return DetectionResult(
        frame_number=1,
        timestamp=123.0,
        balls=[ball],
        cue=cue,
        table=table,
        statistics=FrameStatistics(
            frame_number=1, timestamp=123.0, processing_time=1.0
        ),
        frame_size=(width, height),
    )


def _converted(detection: DetectionResult) -> dict:
    service = IntegrationService.__new__(IntegrationService)
    return service._convert_detection_to_core_format(detection)


def _assert_scale_on_every_point(value) -> None:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            assert value.get("scale") == [1.0, 1.0]
        for child in value.values():
            _assert_scale_on_every_point(child)
    elif isinstance(value, list):
        for child in value:
            _assert_scale_on_every_point(child)


def _assert_equivalent(left, right) -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key == "source_to_canonical_scale":
                continue
            _assert_equivalent(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_equivalent(left_item, right_item)
    elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
        assert left == pytest.approx(right)
    else:
        assert left == right


def test_integration_geometry_is_equal_at_1080p_and_4k_and_scaled_once():
    converted_1080 = _converted(_fixture_detection(1920, 1080, canonical=False))
    converted_4k = _converted(_fixture_detection(3840, 2160, canonical=True))

    for key in ("position", "velocity", "acceleration", "predicted_position"):
        assert converted_1080["balls"][0][key] == converted_4k["balls"][0][key]
    assert (
        converted_1080["balls"][0]["position_history"]
        == converted_4k["balls"][0]["position_history"]
    )
    assert converted_1080["balls"][0]["radius"] == pytest.approx(36)
    assert converted_1080["balls"][0]["radius_extent"] == {
        "x": 36,
        "y": 36,
        "scale": [1.0, 1.0],
    }

    cue_1080, cue_4k = converted_1080["cue"], converted_4k["cue"]
    for key in (
        "tip_position",
        "butt_position",
        "tip_velocity",
        "shaft_points",
        "aiming_line",
        "predicted_contact_point",
        "angle",
        "length",
        "width",
    ):
        assert cue_1080[key] == pytest.approx(cue_4k[key])
    assert cue_1080["length"] == pytest.approx(300)
    assert cue_1080["width"] == pytest.approx(8)

    table_1080, table_4k = converted_1080["table"], converted_4k["table"]
    for key in (
        "corners",
        "rails",
        "playing_area",
        "playing_area_corners",
        "pockets",
        "width",
        "height",
        "pocket_radius",
    ):
        _assert_equivalent(table_1080[key], table_4k[key])
    assert table_1080["width"] == pytest.approx(3440)
    assert table_1080["height"] == pytest.approx(1760)
    assert table_1080["pockets"][0]["radius_extent"] == {
        "x": 72,
        "y": 72,
        "scale": [1.0, 1.0],
    }
    assert table_1080["real_world_dimensions"] == (2.54, 1.27)
    assert table_1080["real_world_dimensions_unit"] == "meters"
    assert table_1080["source_to_canonical_scale"] == [2.0, 2.0]
    assert table_4k["source_to_canonical_scale"] == [1.0, 1.0]
    _assert_scale_on_every_point(converted_1080)


def test_ball_observation_times_survive_serialization_and_core_conversion(
    tmp_path,
):
    detection = _fixture_detection(3840, 2160, canonical=True)
    first, second = detection.balls[0], deepcopy(detection.balls[0])
    first.last_seen = 122.5
    second.track_id = 8
    second.position = (2000.0, 1200.0)
    second.last_seen = 121.25
    detection.balls = [first, second]

    serialized = _converted(detection)

    assert [ball["timestamp"] for ball in serialized["balls"]] == [122.5, 121.25]
    manager = GameStateManager(persistence_path=tmp_path)
    core_balls = manager._extract_ball_states(serialized)
    assert [ball.last_update for ball in core_balls] == [122.5, 121.25]

    # Older payloads remain compatible: the enclosing detection time is the
    # fallback only when no per-ball observation time was serialized.
    legacy_payload = deepcopy(serialized)
    for ball in legacy_payload["balls"]:
        ball.pop("timestamp")
    legacy_balls = manager._extract_ball_states(legacy_payload)
    assert [ball.last_update for ball in legacy_balls] == [123.0, 123.0]


def test_detection_frame_resolution_is_the_only_pixel_resolution_fallback():
    detection = _fixture_detection(1920, 1080, canonical=False)
    detection.balls[0].source_resolution = None
    result = _converted(detection)
    assert result["balls"][0]["position"]["x"] == pytest.approx(1500)

    detection.frame_size = (0, 0)
    with pytest.raises(CoordinateConversionError, match="source resolution"):
        _converted(detection)


@pytest.mark.parametrize("coordinate_space", ["world", "camera_pixels", ""])
def test_unsupported_coordinate_space_is_rejected(coordinate_space: str):
    detection = _fixture_detection(1920, 1080, canonical=False)
    detection.balls[0].coordinate_space = coordinate_space
    with pytest.raises(CoordinateConversionError, match="coordinate space"):
        _converted(detection)


def test_state_conversion_uses_object_resolution_and_rejects_world_space():
    helpers = StateConversionHelpers()
    ball = _fixture_detection(1920, 1080, canonical=False).balls[0]
    state = helpers.vision_ball_to_ball_state(ball, validate=False)
    assert state.position.x == pytest.approx(1500)
    assert state.position.y == pytest.approx(1080)
    assert state.velocity.x == pytest.approx(24)
    assert state.velocity.y == pytest.approx(-8)
    assert state.radius == pytest.approx(36)

    invalid = deepcopy(ball)
    invalid.coordinate_space = "world"
    with pytest.raises(CoordinateConversionError, match="coordinate space"):
        helpers.vision_ball_to_ball_state(invalid, validate=False)


@pytest.mark.asyncio
async def test_multiball_trajectory_serializer_canonicalizes_scaled_vectors():
    service = IntegrationService.__new__(IntegrationService)
    service.last_live_prediction = None
    service.frozen_comparison_prediction = {"preserve": True}
    service._broadcast_with_retry = AsyncMock()
    service.broadcaster = SimpleNamespace(broadcast_trajectory=AsyncMock())
    points = [
        SimpleNamespace(position=Vector2D(10, 20, scale=(2, 2))),
        SimpleNamespace(position=Vector2D(20, 30, scale=(2, 2))),
        SimpleNamespace(position=Vector2D(30, 40, scale=(2, 2))),
    ]
    collision = SimpleNamespace(
        time=0.2,
        position=Vector2D(15, 25, scale=(2, 2)),
        type=SimpleNamespace(value="ball_ball"),
        ball1_id="cue",
        ball2_id="target",
        impact_angle=0.0,
        confidence=0.9,
    )
    trajectory = SimpleNamespace(points=points, collisions=[collision])
    result = SimpleNamespace(
        trajectories={"cue": trajectory},
        primary_ball_id="cue",
        total_calculation_time=0.01,
    )

    await service._emit_multiball_trajectory(result)

    lines = service._broadcast_with_retry.await_args.args[3]
    collisions = service._broadcast_with_retry.await_args.args[4]
    assert lines[0]["start"] == {"x": 20, "y": 40, "scale": [1.0, 1.0]}
    assert lines[0]["end"] == {"x": 40, "y": 60, "scale": [1.0, 1.0]}
    assert collisions[0]["position"] == {
        "x": 30,
        "y": 50,
        "scale": [1.0, 1.0],
    }
    assert set(collisions[0]) == {
        "time",
        "position",
        "type",
        "ball1_id",
        "ball2_id",
        "angle",
        "confidence",
    }
    assert collisions[0]["ball1_id"] == "cue"
    assert collisions[0]["ball2_id"] == "target"
    assert "is_reference" not in service._broadcast_with_retry.await_args.kwargs
    assert service.last_live_prediction["lines"] == lines
    assert "is_reference" not in service.last_live_prediction
    assert service.frozen_comparison_prediction is None
