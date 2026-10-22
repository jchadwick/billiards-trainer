"""Edge contracts for calibration routes and their OpenCV boundaries.

Tests call route functions directly so their state transitions and HTTP error
mapping remain deterministic without requiring a camera, table, or filesystem
calibration artifact.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.api.routes.calibration as calibration_routes
from backend.api.middleware.error_handler import CustomHTTPException
from backend.api.models.requests import (
    CalibrationPointRequest,
    CalibrationValidateRequest,
)


def _session(
    *,
    status: str = "in_progress",
    expires_at: datetime | None = None,
    points: list[dict[str, Any]] | None = None,
    points_required: int = 4,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    current_points = points or []
    return {
        "session_id": "session-edge",
        "calibration_type": "quick",
        "status": status,
        "created_at": now,
        "expires_at": expires_at or now + timedelta(minutes=5),
        "points_captured": len(current_points),
        "points_required": points_required,
        "points": current_points,
    }


def _point_request(
    *,
    session_id: str = "session-edge",
    point_id: str = "corner-a",
    x: float = 100.0,
    y: float = 100.0,
    scale: tuple[float, float] = (1.0, 1.0),
    world_x: float = 0.2,
    world_y: float = 0.3,
) -> CalibrationPointRequest:
    return CalibrationPointRequest.model_validate(
        {
            "session_id": session_id,
            "point_id": point_id,
            "screen_position": {"x": x, "y": y, "scale": scale},
            "world_position": {
                "x": world_x,
                "y": world_y,
                "unit": "meters",
            },
            "confidence": 0.9,
        }
    )


def _four_points() -> list[dict[str, Any]]:
    return [
        {
            "point_id": f"corner-{index}",
            "screen_x": float(x),
            "screen_y": float(y),
            "world_x": float(world_x),
            "world_y": float(world_y),
            "world_position": {
                "x": float(world_x),
                "y": float(world_y),
                "unit": "meters",
            },
            "screen_position": {
                "x": float(x),
                "y": float(y),
                "scale": [1.0, 1.0],
            },
            "confidence": 0.9,
        }
        for index, (x, y, world_x, world_y) in enumerate(
            [
                (0, 0, 0.0, 0.0),
                (100, 0, 1.0, 0.0),
                (100, 100, 1.0, 1.0),
                (0, 100, 0.0, 1.0),
            ]
        )
    ]


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(calibration_routes, "_calibration_sessions", current)
    return current


@pytest.fixture
def camera_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    current: dict[str, Any] = {
        "active": True,
        "images": [],
        "chessboard_size": (9, 6),
        "square_size": 0.025,
        "min_images": 5,
    }
    monkeypatch.setattr(calibration_routes, "_camera_calibration_session", current)
    return current


def test_session_validation_distinguishes_missing_expired_and_wrong_state(
    sessions: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(HTTPException) as missing:
        calibration_routes.validate_calibration_session("absent")
    assert missing.value.status_code == 404

    expired = _session(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    sessions["expired"] = expired
    with pytest.raises(HTTPException) as wrong_state:
        calibration_routes.validate_calibration_session("expired", "in_progress")

    assert expired["status"] == "expired"
    assert wrong_state.value.status_code == 400
    assert "expired" in str(wrong_state.value.detail)


@pytest.mark.parametrize(
    ("path_id", "request_id", "x", "world_x", "expected_status", "message"),
    [
        ("session-edge", "another-session", 10.0, 0.2, 422, "must match"),
        ("session-edge", "session-edge", 4000.0, 0.2, 400, "Screen coordinates"),
        ("session-edge", "session-edge", 10.0, 10.1, 400, "World coordinates"),
    ],
)
def test_point_capture_rejects_mismatched_identity_and_out_of_range_coordinates(
    sessions: dict[str, dict[str, Any]],
    path_id: str,
    request_id: str,
    x: float,
    world_x: float,
    expected_status: int,
    message: str,
) -> None:
    sessions[path_id] = _session()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            calibration_routes.capture_calibration_point(
                path_id,
                _point_request(session_id=request_id, x=x, world_x=world_x),
                cast("Any", None),
            )
        )

    assert raised.value.status_code == expected_status
    assert message in str(raised.value.detail)
    assert sessions[path_id]["points"] == []


def test_recapturing_point_id_replaces_existing_point_without_inflating_count(
    sessions: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        "point_id": "corner-a",
        "screen_x": 10.0,
        "screen_y": 20.0,
        "world_x": 0.1,
        "world_y": 0.2,
    }
    session = _session(points=[original], points_required=4)
    sessions["session-edge"] = session
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(lambda *_args: None),
    )

    response = asyncio.run(
        calibration_routes.capture_calibration_point(
            "session-edge",
            _point_request(x=45.0, y=55.0, world_x=0.4, world_y=0.5),
            cast("Any", None),
        )
    )

    assert response.total_points == 1
    assert response.remaining_points == 3
    assert response.can_proceed is False
    assert len(session["points"]) == 1
    assert session["points"][0]["screen_x"] == 45.0
    assert session["points"][0]["world_x"] == 0.4


def test_calibration_math_handles_degenerate_inputs_and_opencv_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        calibration_routes.CalibrationMath.calculate_homography(
            [(0.0, 0.0)], [(1.0, 1.0)]
        )
        is None
    )
    assert calibration_routes.CalibrationMath.calculate_accuracy([]) == {
        "accuracy": 0.0,
        "max_error": float("inf"),
        "mean_error": float("inf"),
    }

    monkeypatch.setattr(
        calibration_routes.cv2,
        "findHomography",
        lambda *_args: (None, None),
    )
    many_sources = [(float(index), float(index % 2)) for index in range(5)]
    many_targets = [(x + 1.0, y + 2.0) for x, y in many_sources]
    assert (
        calibration_routes.CalibrationMath.calculate_homography(
            many_sources, many_targets
        )
        is None
    )

    assert not calibration_routes.CalibrationMath._validate_homography(
        np.eye(2, dtype=np.float64)
    )
    assert not calibration_routes.CalibrationMath._validate_homography(
        np.zeros((3, 3), dtype=np.float64)
    )


def test_apply_force_override_continues_after_backup_failure(
    sessions: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(status="ready_to_apply", points=_four_points())
    session["accuracy"] = 0.2
    sessions["session-edge"] = session
    matrix = np.eye(3, dtype=np.float64)
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(lambda *_args: matrix),
    )

    def fail_open(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("read-only backup directory")

    monkeypatch.setattr("builtins.open", fail_open)
    applied: list[dict[str, Any]] = []
    core = SimpleNamespace(apply_calibration=lambda **kwargs: applied.append(kwargs))

    response = asyncio.run(
        calibration_routes.apply_calibration_transformations(
            "session-edge",
            backup_previous=True,
            force_apply=True,
            core_module=cast("Any", core),
        )
    )

    assert response.success is True
    assert response.backup_created is False
    assert applied[0]["transformation_matrix"] is matrix
    assert session["status"] == "applied"


@pytest.mark.parametrize(
    ("matrix", "core", "expected_status", "message"),
    [
        (None, SimpleNamespace(), 500, "Failed to calculate transformation matrix"),
        (
            np.eye(3),
            SimpleNamespace(
                apply_calibration=lambda **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("core rejected calibration")
                )
            ),
            500,
            "Failed to apply calibration to system",
        ),
    ],
)
def test_apply_maps_homography_and_core_failures_to_server_errors(
    sessions: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    matrix: np.ndarray | None,
    core: Any,
    expected_status: int,
    message: str,
) -> None:
    session = _session(status="ready_to_apply", points=_four_points())
    session["accuracy"] = 0.95
    sessions["session-edge"] = session
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(lambda *_args: matrix),
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            calibration_routes.apply_calibration_transformations(
                "session-edge",
                backup_previous=False,
                core_module=cast("Any", core),
            )
        )

    assert raised.value.status_code == expected_status
    assert message in str(raised.value.detail)
    assert session["status"] == "ready_to_apply"


def test_validation_rejects_path_mismatch_and_test_points_outside_canonical_plane(
    sessions: dict[str, dict[str, Any]],
) -> None:
    sessions["session-edge"] = _session(status="ready_to_apply", points=_four_points())
    mismatched = CalibrationValidateRequest.model_validate(
        {"session_id": "different-session"}
    )
    with pytest.raises(HTTPException) as mismatch:
        asyncio.run(
            calibration_routes.validate_calibration_accuracy(
                "session-edge", mismatched, cast("Any", None)
            )
        )
    assert mismatch.value.status_code == 422

    outside = CalibrationValidateRequest.model_validate(
        {
            "session_id": "session-edge",
            "test_points": [
                {
                    "screen": {"x": 4000, "y": 10, "scale": [1, 1]},
                    "world": {"x": 1.0, "y": 1.0, "unit": "meters"},
                }
            ],
        }
    )
    with pytest.raises(HTTPException) as bounds:
        asyncio.run(
            calibration_routes.validate_calibration_accuracy(
                "session-edge", outside, cast("Any", None)
            )
        )
    assert bounds.value.status_code == 422
    assert "outside canonical image bounds" in str(bounds.value.detail)


def test_validation_fallback_reports_low_confidence_and_poor_accuracy(
    sessions: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _four_points()
    points[0]["confidence"] = 0.4
    sessions["session-edge"] = _session(status="completed", points=points)
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(lambda *_args: None),
    )
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_accuracy",
        staticmethod(
            lambda *_args: {
                "accuracy": 0.6,
                "max_error": 11.0,
                "mean_error": 6.0,
            }
        ),
    )
    request = CalibrationValidateRequest.model_validate(
        {"session_id": "session-edge", "accuracy_threshold": 0.9}
    )

    response = asyncio.run(
        calibration_routes.validate_calibration_accuracy(
            "session-edge", request, cast("Any", None)
        )
    )

    assert response.is_valid is False
    assert response.errors == [
        "Calibration accuracy is critically low",
        "Maximum error exceeds acceptable threshold",
    ]
    assert any("Recalibrate" in item for item in response.recommendations)
    assert any("outlier" in item for item in response.recommendations)
    assert any("confidence" in item for item in response.recommendations)
    assert response.test_results[0]["transformed_position"] == {
        "x": 0.0,
        "y": 0.0,
        "unit": "meters",
    }
    assert sessions["session-edge"]["validation_results"]["threshold_used"] == 0.9


def test_current_calibration_uses_configured_corners_and_maps_bad_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        def __init__(self, corners: list[dict[str, float]]) -> None:
            self.corners = corners

        def get(self, key: str, default: Any = None) -> Any:
            return self.corners if key == "table.playing_area_corners" else default

    corners = [
        {"x": float(index), "y": float(index + 1), "scale": [1.0, 1.0]}
        for index in range(4)
    ]
    monkeypatch.setattr("backend.config.config", Config(corners))
    monkeypatch.setattr(calibration_routes, "_calibration_sessions", {})

    response = asyncio.run(
        calibration_routes.get_current_calibration(cast("Any", None))
    )
    assert response["is_valid"] is True
    assert response["corners"][0]["x"] == 0.0
    assert response["transformation_matrix"] is None

    monkeypatch.setattr(
        "backend.config.config",
        Config([{"x": 1.0, "y": 2.0, "scale": [0.0, 1.0]}]),
    )
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(calibration_routes.get_current_calibration(cast("Any", None)))
    assert invalid.value.status_code == 500
    assert "Unable to retrieve current calibration" in str(invalid.value.detail)


def test_camera_capture_maps_camera_and_opencv_errors_separately(
    camera_session: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(CustomHTTPException) as unavailable:
        asyncio.run(
            calibration_routes.capture_camera_calibration_image(
                vision_module=SimpleNamespace(
                    get_frame=lambda **_kwargs: (_ for _ in ()).throw(
                        OSError("camera read failed")
                    )
                )
            )
        )
    assert unavailable.value.status_code == 503
    assert unavailable.value.error_code == "CAM_001"

    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    monkeypatch.setattr(
        calibration_routes.cv2,
        "findChessboardCorners",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("opencv failed")),
    )
    with pytest.raises(HTTPException) as processing_error:
        asyncio.run(
            calibration_routes.capture_camera_calibration_image(
                vision_module=SimpleNamespace(get_frame=lambda **_kwargs: frame)
            )
        )
    assert processing_error.value.status_code == 500
    assert "Unable to capture calibration image" in str(processing_error.value.detail)


def test_camera_processing_maps_calibrator_failure_to_processing_service_error(
    camera_session: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_session.update(
        images=[
            {
                "image": np.zeros((24, 32, 3), dtype=np.uint8),
                "corners": np.zeros((54, 1, 2), dtype=np.float32),
            }
        ],
        min_images=1,
    )
    monkeypatch.setattr(
        calibration_routes.cv2.fisheye,
        "calibrate",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("solver failed")),
    )

    with pytest.raises(CustomHTTPException) as raised:
        asyncio.run(calibration_routes.process_camera_calibration())

    assert raised.value.status_code == 503
    assert raised.value.error_code == "PROC_001"


def test_table_fisheye_detection_failure_is_mapped_to_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    monkeypatch.setattr(
        calibration_routes.cv2,
        "findContours",
        lambda *_args: ([], None),
    )
    # The no-table diagnostic images are deliberately redirected so this
    # route contract cannot leave artifacts in the repository.
    writes: list[str] = []
    monkeypatch.setattr(
        calibration_routes.cv2,
        "imwrite",
        lambda path, *_args: writes.append(str(path)) or True,
    )
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)

    with pytest.raises(CustomHTTPException) as raised:
        asyncio.run(
            calibration_routes.calibrate_fisheye_from_table(
                vision_module=SimpleNamespace(get_frame=lambda **_kwargs: frame)
            )
        )

    assert raised.value.status_code == 503
    assert raised.value.error_code == "PROC_001"
    assert len(writes) == 2
    assert writes[0].endswith("fisheye_debug_no_table.jpg")
    assert writes[1].endswith("fisheye_debug_no_table_mask.jpg")


def test_list_session_mapping_errors_are_returned_as_http_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration_routes,
        "_calibration_sessions",
        {"malformed": {"created_at": datetime.now(timezone.utc)}},
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(calibration_routes.list_calibration_sessions(status=None, limit=50))

    assert raised.value.status_code == 500
    assert "Unable to list calibration sessions" in str(raised.value.detail)


def test_point_request_model_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        CalibrationPointRequest.model_validate(
            {
                "session_id": "session-edge",
                "point_id": "corner-a",
                "screen_position": {"x": 1, "y": 2, "scale": [1, 1]},
                "world_position": {"x": 0.1, "y": 0.2, "unit": "meters"},
                "confidence": 1.1,
            }
        )
