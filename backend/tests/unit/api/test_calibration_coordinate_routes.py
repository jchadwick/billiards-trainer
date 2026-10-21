"""Coordinate contracts for calibration API routes."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

import backend.api.routes.calibration as calibration_routes
import backend.config as backend_config
from backend.api.models.requests import (
    CalibrationPointRequest,
    CalibrationValidateRequest,
)

if TYPE_CHECKING:
    from backend.core import CoreModule


def _point_request(
    session_id: str,
    point_id: str,
    x: float,
    y: float,
    world_x: float,
    world_y: float,
) -> CalibrationPointRequest:
    return CalibrationPointRequest.model_validate(
        {
            "session_id": session_id,
            "point_id": point_id,
            "screen_position": {"x": x, "y": y, "scale": [2.0, 3.0]},
            "world_position": {
                "x": world_x,
                "y": world_y,
                "unit": "meters",
            },
            "confidence": 0.9,
        }
    )


@pytest.mark.parametrize(
    "screen_position",
    [
        {"x": 100, "y": 100},
        {"x": 100, "y": 100, "scale": [2]},
        {"x": 100, "y": 100, "scale": [0, 3]},
        {"x": 100, "y": 100, "scale": [float("nan"), 3]},
    ],
    ids=[
        "missing-scale",
        "wrong-scale-length",
        "non-positive-scale",
        "non-finite-scale",
    ],
)
def test_calibration_point_request_requires_valid_screen_scale(
    screen_position: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        CalibrationPointRequest.model_validate(
            {
                "session_id": "session-1",
                "point_id": "corner-1",
                "screen_position": screen_position,
                "world_position": {"x": 0.1, "y": 0.2, "unit": "meters"},
            }
        )


def test_calibration_capture_uses_canonical_pixels_for_homography(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-1"
    session: dict[str, Any] = {
        "status": "in_progress",
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
        "points": [],
        "points_captured": 0,
        "points_required": 4,
    }
    monkeypatch.setattr(
        calibration_routes,
        "_calibration_sessions",
        {session_id: session},
    )
    homography_inputs: list[
        tuple[list[tuple[float, float]], list[tuple[float, float]]]
    ] = []

    def capture_inputs(
        source: list[tuple[float, float]], target: list[tuple[float, float]]
    ) -> None:
        homography_inputs.append((source, target))
        return None

    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(capture_inputs),
    )
    source_points = [(100, 100), (500, 100), (500, 300), (100, 300)]
    target_points = [(0.1, 0.1), (2.1, 0.1), (2.1, 1.1), (0.1, 1.1)]

    for index, ((x, y), (world_x, world_y)) in enumerate(
        zip(source_points, target_points)
    ):
        result = asyncio.run(
            calibration_routes.capture_calibration_point(
                session_id,
                _point_request(
                    session_id,
                    f"corner-{index}",
                    x,
                    y,
                    world_x,
                    world_y,
                ),
                cast("CoreModule", None),
            )
        )

    assert result.can_proceed is True
    assert homography_inputs[0][0][0] == (200.0, 300.0)
    assert homography_inputs[0][1][0] == (0.1, 0.1)
    assert session["points"][0]["screen_position"] == {
        "x": 100.0,
        "y": 100.0,
        "scale": [2.0, 3.0],
    }
    assert session["points"][0]["screen_x"] == 200.0
    assert session["points"][0]["screen_y"] == 300.0
    assert session["points"][0]["world_position"] == {
        "x": 0.1,
        "y": 0.1,
        "unit": "meters",
    }


def test_current_calibration_returns_scale_aware_session_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-applied"
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        backend_config,
        "config",
        type(
            "ConfigStub",
            (),
            {"get": lambda self, key, default=None: default},
        )(),
    )
    monkeypatch.setattr(
        calibration_routes,
        "_calibration_sessions",
        {
            session_id: {
                "status": "applied",
                "applied_at": now,
                "points": [
                    {
                        "screen_position": {
                            "x": 100.0,
                            "y": 120.0,
                            "scale": [2.0, 3.0],
                        }
                    }
                ],
            }
        },
    )

    result = asyncio.run(
        calibration_routes.get_current_calibration(cast("CoreModule", None))
    )

    assert result["corners"] == [{"x": 100.0, "y": 120.0, "scale": [2.0, 3.0]}]


def test_calibration_validation_returns_scaled_image_and_meter_world_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-validate"
    screen_points = [
        (200.0, 300.0),
        (1000.0, 300.0),
        (1000.0, 900.0),
        (200.0, 900.0),
    ]
    world_points = [(0.1, 0.1), (2.1, 0.1), (2.1, 1.1), (0.1, 1.1)]
    session_points = [
        {
            "point_id": f"corner-{index}",
            "screen_x": screen_x,
            "screen_y": screen_y,
            "screen_position": {
                "x": screen_x / 2,
                "y": screen_y / 3,
                "scale": [2.0, 3.0],
            },
            "world_x": world_x,
            "world_y": world_y,
            "world_position": {
                "x": world_x,
                "y": world_y,
                "unit": "meters",
            },
            "confidence": 0.9,
        }
        for index, ((screen_x, screen_y), (world_x, world_y)) in enumerate(
            zip(screen_points, world_points)
        )
    ]
    monkeypatch.setattr(
        calibration_routes,
        "_calibration_sessions",
        {
            session_id: {
                "status": "ready_to_apply",
                "points": session_points,
            }
        },
    )
    request = CalibrationValidateRequest.model_validate(
        {
            "session_id": session_id,
            "test_points": [
                {
                    "screen": {"x": 50.0, "y": 50.0, "scale": [2.0, 3.0]},
                    "world": {"x": 1.1, "y": 0.6, "unit": "meters"},
                }
            ],
        }
    )

    result = asyncio.run(
        calibration_routes.validate_calibration_accuracy(
            session_id,
            request,
            cast("CoreModule", None),
        )
    )

    assert result.test_results[0]["screen_position"] == {
        "x": 100.0,
        "y": 100.0,
        "scale": [2.0, 3.0],
    }
    assert result.test_results[0]["world_position"] == {
        "x": 0.1,
        "y": 0.1,
        "unit": "meters",
    }
    assert result.test_results[-1]["screen_position"] == {
        "x": 50.0,
        "y": 50.0,
        "scale": [2.0, 3.0],
    }
    assert result.test_results[-1]["world_position"] == {
        "x": 1.1,
        "y": 0.6,
        "unit": "meters",
    }
