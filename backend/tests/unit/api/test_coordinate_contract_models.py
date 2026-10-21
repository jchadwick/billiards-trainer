"""Regression tests for explicit coordinate scale and physical-unit contracts."""

import pytest
from pydantic import ValidationError

from backend.api.models.common import BoundingBox, Coordinate2D, PositionWithScale
from backend.api.models.requests import (
    BallPositionUpdateRequest,
    CalibrationPointRequest,
    CalibrationValidateRequest,
)
from backend.api.models.responses import TableInfo
from backend.api.models.vision_models import (
    BoundingBoxModel,
    ContourModel,
    Point2DModel,
    VisionBallModel,
    VisionCueModel,
    VisionTableModel,
    WorldPoint2DModel,
)
from backend.api.models.websocket import BallStateData, TableStateData
from backend.api.websocket.schemas import (
    BallData,
    CollisionData,
    CueData,
    TableData,
    TrajectoryData,
    TrajectoryLine,
)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (PositionWithScale, {"x": 3.0, "y": 4.0}),
        (Coordinate2D, {"x": 3.0, "y": 4.0}),
        (Point2DModel, {"x": 3.0, "y": 4.0}),
        (BoundingBox, {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}),
        (
            BoundingBoxModel,
            {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
        ),
    ],
)
def test_image_coordinate_models_require_scale(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("scale", [[0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0]])
def test_image_coordinate_models_reject_invalid_scale(scale):
    with pytest.raises(ValidationError):
        Point2DModel(x=1.0, y=2.0, scale=scale)


def test_pixel_bounds_scale_origin_and_extents():
    box = BoundingBoxModel(
        x=10,
        y=20,
        width=30,
        height=40,
        scale=[2.0, 3.0],
    )

    assert box.model_dump() == {
        "x": 10.0,
        "y": 20.0,
        "width": 30.0,
        "height": 40.0,
        "scale": [2.0, 3.0],
        "confidence": 1.0,
    }


def test_world_point_uses_meter_units_instead_of_pixel_scale():
    point = WorldPoint2DModel(x=1.25, y=0.75, unit="meters")

    assert point.model_dump() == {"x": 1.25, "y": 0.75, "unit": "meters"}
    with pytest.raises(ValidationError):
        WorldPoint2DModel(x=1.25, y=0.75)


def test_vision_pixel_measurements_require_their_own_scale_metadata():
    point = {"x": 1.0, "y": 2.0, "scale": [2.0, 2.0]}
    with pytest.raises(ValidationError, match="scale"):
        ContourModel(
            points=[point],
            area=4.0,
            perimeter=8.0,
            center=point,
        )

    with pytest.raises(ValidationError, match="dimensions_scale"):
        VisionTableModel(
            corners=[point] * 4,
            surface_color=(0, 0, 0),
            width_pixels=100.0,
            height_pixels=50.0,
            confidence=1.0,
        )

    with pytest.raises(ValidationError, match="length_scale"):
        VisionCueModel(
            tip_position=point,
            end_position=point,
            angle=0.0,
            length_pixels=100.0,
            confidence=1.0,
            bounding_box={
                "x": 0.0,
                "y": 0.0,
                "width": 100.0,
                "height": 5.0,
                "scale": [2.0, 2.0],
            },
        )


def test_vision_ball_requires_scale_for_pixel_radius():
    point = {"x": 10.0, "y": 20.0, "scale": [2.0, 2.0]}
    box = {
        "x": 5.0,
        "y": 15.0,
        "width": 10.0,
        "height": 10.0,
        "scale": [2.0, 2.0],
    }
    payload = {
        "id": "cue",
        "center": point,
        "radius": 5.0,
        "color": (255, 255, 255),
        "confidence": 1.0,
        "bounding_box": box,
    }

    with pytest.raises(ValidationError, match="radius_scale"):
        VisionBallModel.model_validate(payload)

    payload["radius_scale"] = [2.0, 2.0]
    ball = VisionBallModel.model_validate(payload)
    assert ball.radius_scale == [2.0, 2.0]


def test_websocket_geometry_requires_scale_on_boundaries_and_trajectory():
    bare_point = {"x": 1.0, "y": 2.0}
    with pytest.raises(ValidationError):
        TableData(corners=[bare_point] * 4, pockets=[])
    with pytest.raises(ValidationError):
        TrajectoryLine(start=bare_point, end=bare_point, type="primary")
    with pytest.raises(ValidationError):
        CollisionData(position=bare_point, ball1_id="cue", angle=0.0)

    scaled_point = {"x": 1.0, "y": 2.0, "scale": [1.0, 1.0]}
    cushion_collision = CollisionData(
        position=scaled_point,
        ball1_id="cue",
        ball2_id=None,
        type="cushion",
        angle=0.0,
    )
    assert cushion_collision.ball1_id == "cue"
    assert cushion_collision.ball2_id is None
    assert "ball_id" not in cushion_collision.model_dump()

    with pytest.raises(ValidationError, match="ball_id"):
        CollisionData(
            position=scaled_point,
            ball1_id="cue",
            ball_id="2",
            angle=0.0,
        )

    trajectory = TrajectoryData(
        lines=[
            {
                "start": scaled_point,
                "end": scaled_point,
                "type": "secondary",
                "ball_id": "2",
            }
        ],
        collisions=[
            {
                "position": scaled_point,
                "ball1_id": "cue",
                "ball2_id": None,
                "type": "ball_cushion",
                "angle": 0.0,
                "time": 0.1,
                "confidence": 0.9,
            }
        ],
        timestamp="2026-01-01T00:00:00+00:00",
        sequence=1,
        line_count=1,
        collision_count=1,
        prediction_status="frozen_reference",
    )
    assert trajectory.prediction_status == "frozen_reference"
    assert "is_reference" not in trajectory.model_dump()
    with pytest.raises(ValidationError, match="is_reference"):
        TrajectoryData(
            lines=[{"start": scaled_point, "end": scaled_point, "type": "primary"}],
            line_count=1,
            collision_count=0,
            is_reference=True,
        )

    with pytest.raises(ValidationError, match="x"):
        CollisionData(
            position=scaled_point,
            ball1_id="cue",
            angle=0.0,
            x=scaled_point["x"],
        )

    with pytest.raises(ValidationError, match="radius_scale"):
        BallData(
            id="cue",
            position=scaled_point,
            radius=5.0,
            color="white",
        )
    with pytest.raises(ValidationError, match="length_scale"):
        CueData(
            angle=0.0,
            position=scaled_point,
            length=100.0,
        )


def test_api_state_extent_models_require_pixel_units_and_valid_scales():
    point = {"x": 1.0, "y": 2.0, "scale": [1.0, 1.0]}
    ball = {
        "id": "cue",
        "position": point,
        "velocity": point,
        "radius": 36.0,
        "radius_scale": [1.0, 1.0],
        "radius_unit": "canonical_pixels",
    }
    table = {
        "width": 3200.0,
        "height": 1600.0,
        "dimensions_scale": [1.0, 1.0],
        "dimensions_unit": "canonical_pixels",
        "pocket_positions": [point],
        "pocket_radius": 72.0,
        "pocket_radius_scale": [1.0, 1.0],
        "pocket_radius_unit": "canonical_pixels",
        "real_world_dimensions": {"width": 2.54, "height": 1.27, "unit": "meters"},
    }

    parsed_ball = BallStateData.model_validate(ball)
    parsed_table = TableStateData.model_validate(table)
    assert parsed_ball.radius_unit == "canonical_pixels"
    assert parsed_table.dimensions_unit == "canonical_pixels"
    assert parsed_table.real_world_dimensions is not None
    assert parsed_table.real_world_dimensions.unit == "meters"

    for invalid_scale in ([0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [True, 1.0]):
        with pytest.raises(ValidationError):
            BallStateData.model_validate({**ball, "radius_scale": invalid_scale})
        with pytest.raises(ValidationError):
            TableStateData.model_validate({**table, "dimensions_scale": invalid_scale})

    with pytest.raises(ValidationError, match="radius_unit"):
        BallStateData.model_validate({**ball, "radius_unit": "meters"})
    with pytest.raises(ValidationError, match="dimensions_unit"):
        TableStateData.model_validate({**table, "dimensions_unit": "meters"})
    with pytest.raises(ValidationError, match="pocket_radius_scale"):
        TableStateData.model_validate({**table, "pocket_radius_scale": [1.0, 2.0]})
    with pytest.raises(ValidationError, match="real_world_dimensions"):
        TableStateData.model_validate(
            {
                **table,
                "real_world_dimensions": {
                    "width": 2.54,
                    "height": 1.27,
                    "unit": "millimeters",
                },
            }
        )

    response_fields = {
        key: value for key, value in table.items() if key != "pocket_positions"
    }
    with pytest.raises(ValidationError, match="dimensions_scale"):
        TableInfo.model_validate(
            {
                **response_fields,
                "pocket_positions": [point],
                "dimensions_scale": [1.0, 0.0],
            }
        )


def test_api_requests_keep_pixel_and_world_coordinates_explicit():
    with pytest.raises(ValidationError):
        CalibrationPointRequest(
            session_id="session_123",
            point_id="corner_1",
            screen_position={"x": 10.0, "y": 20.0},
            world_position={"x": 0.1, "y": 0.2, "unit": "meters"},
        )
    with pytest.raises(ValidationError):
        CalibrationPointRequest(
            session_id="session_123",
            point_id="corner_1",
            screen_position={"x": 10.0, "y": 20.0, "scale": [2.0, 2.0]},
            world_position={"x": 0.1, "y": 0.2},
        )
    with pytest.raises(ValidationError):
        CalibrationValidateRequest(
            session_id="session_123",
            test_points=[{"screen": [10.0, 20.0], "world": [0.1, 0.2]}],
        )
    with pytest.raises(ValidationError):
        BallPositionUpdateRequest(
            ball_updates=[
                {
                    "id": "cue",
                    "position": [10.0, 20.0],
                    "velocity": [0.0, 0.0],
                }
            ]
        )

    request = CalibrationPointRequest(
        session_id="session_123",
        point_id="corner_1",
        screen_position={"x": 10.0, "y": 20.0, "scale": [2.0, 2.0]},
        world_position={"x": 0.1, "y": 0.2, "unit": "meters"},
    )
    assert request.screen_position.scale == [2.0, 2.0]
    assert request.world_position.unit == "meters"
