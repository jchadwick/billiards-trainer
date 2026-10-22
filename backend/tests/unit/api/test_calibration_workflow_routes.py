"""Behavioral coverage for calibration session and camera workflows.

The fakes in this module keep route contracts deterministic and hardware-free:
OpenCV's fitting boundary is replaced only where the test is about persistence
or response behavior, while request validation and route state transitions are
exercised directly.
"""

import asyncio
import base64
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException

import backend.api.routes.calibration as calibration_routes
from backend.api.middleware.error_handler import CustomHTTPException


def _session(
    *,
    status: str = "in_progress",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
    points: list[dict[str, Any]] | None = None,
    points_required: int = 4,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "session_id": "session-a",
        "calibration_type": "quick",
        "status": status,
        "created_at": created_at or now,
        "expires_at": expires_at or now + timedelta(minutes=5),
        "points_captured": len(points or []),
        "points_required": points_required,
        "points": points or [],
    }


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(calibration_routes, "_calibration_sessions", current)
    return current


@pytest.fixture
def camera_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    current: dict[str, Any] = {
        "active": False,
        "images": [],
        "chessboard_size": (9, 6),
        "square_size": 0.025,
        "min_images": 10,
    }
    monkeypatch.setattr(calibration_routes, "_camera_calibration_session", current)
    return current


def test_start_calibration_returns_mode_contract_and_schedules_cleanup(
    sessions: dict[str, dict[str, Any]],
) -> None:
    tasks = BackgroundTasks()

    response = asyncio.run(
        calibration_routes.start_calibration_sequence(
            tasks,
            calibration_type="quick",
            timeout_seconds=120,
            core_module=cast("Any", None),
        )
    )

    assert response.session.status == "in_progress"
    assert response.session.calibration_type == "quick"
    assert response.expected_points == 4
    assert response.timeout == 120
    assert len(response.instructions) == 4
    assert len(sessions) == 1
    assert next(iter(sessions.values()))["metadata"]["timeout_seconds"] == 120
    assert len(tasks.tasks) == 1


def test_start_rejects_concurrent_session_unless_force_restart(
    sessions: dict[str, dict[str, Any]],
) -> None:
    previous = _session()
    sessions["session-a"] = previous

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            calibration_routes.start_calibration_sequence(
                BackgroundTasks(),
                force_restart=False,
                timeout_seconds=300,
                core_module=cast("Any", None),
            )
        )
    assert raised.value.status_code == 409
    assert "session-a" in str(raised.value.detail)

    response = asyncio.run(
        calibration_routes.start_calibration_sequence(
            BackgroundTasks(),
            timeout_seconds=300,
            calibration_type="advanced",
            force_restart=True,
            core_module=cast("Any", None),
        )
    )
    assert previous["status"] == "cancelled"
    assert previous["cancelled_at"].tzinfo is not None
    assert response.expected_points == 16


def test_get_session_marks_expired_in_progress_session(
    sessions: dict[str, dict[str, Any]],
) -> None:
    expired = _session(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    sessions["session-a"] = expired

    result = asyncio.run(calibration_routes.get_calibration_session("session-a"))

    assert result.status == "expired"
    assert expired["status"] == "expired"


def test_list_sessions_sorts_newest_first_and_filters_status(
    sessions: dict[str, dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc)
    sessions.update(
        {
            "older": _session(
                status="applied",
                created_at=now - timedelta(minutes=1),
            ),
            "newer": _session(
                status="in_progress",
                created_at=now,
            ),
        }
    )

    all_sessions = asyncio.run(
        calibration_routes.list_calibration_sessions(status=None, limit=50)
    )
    applied = asyncio.run(
        calibration_routes.list_calibration_sessions(status="applied", limit=50)
    )

    assert [item.session_id for item in all_sessions] == ["newer", "older"]
    assert [item.session_id for item in applied] == ["older"]


def test_delete_session_removes_data_and_returns_deleted_identity(
    sessions: dict[str, dict[str, Any]],
) -> None:
    sessions["session-a"] = _session(status="cancelled")

    response = asyncio.run(calibration_routes.delete_calibration_session("session-a"))

    assert "session-a" not in sessions
    assert response["success"] is True
    assert response["data"]["session_id"] == "session-a"


def test_cleanup_expires_active_sessions_and_removes_sessions_older_than_retention(
    sessions: dict[str, dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc)
    sessions.update(
        {
            "expired": _session(
                expires_at=now - timedelta(seconds=1),
                created_at=now - timedelta(days=1),
            ),
            "old": _session(
                status="applied",
                created_at=now - timedelta(days=31),
            ),
            "current": _session(),
        }
    )

    asyncio.run(calibration_routes.cleanup_expired_sessions())

    assert sessions["expired"]["status"] == "expired"
    assert "old" not in sessions
    assert sessions["current"]["status"] == "in_progress"


def test_apply_calibration_passes_homography_to_core_and_updates_session(
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, dict[str, Any]],
) -> None:
    matrix = np.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    points = [
        {
            "screen_x": float(i),
            "screen_y": float(i + 1),
            "world_x": float(i),
            "world_y": float(i + 1),
        }
        for i in range(4)
    ]
    session = _session(status="ready_to_apply", points=points)
    session["accuracy"] = 0.99
    sessions["session-a"] = session
    monkeypatch.setattr(
        calibration_routes.CalibrationMath,
        "calculate_homography",
        staticmethod(lambda _src, _dst: matrix),
    )
    calls: list[dict[str, Any]] = []
    core = SimpleNamespace(apply_calibration=lambda **kwargs: calls.append(kwargs))

    response = asyncio.run(
        calibration_routes.apply_calibration_transformations(
            "session-a",
            save_as_default=False,
            backup_previous=False,
            core_module=cast("Any", core),
        )
    )

    assert response.success is True
    assert response.backup_created is False
    assert response.transformation_matrix == matrix.tolist()
    assert calls[0]["transformation_matrix"] is matrix
    assert calls[0]["save_as_default"] is False
    assert session["status"] == "applied"
    assert session["transformation_matrix"] == matrix.tolist()


@pytest.mark.parametrize(
    ("session_id", "status", "point_count", "accuracy", "force_apply", "expected"),
    [
        ("missing", "ready_to_apply", 4, 1.0, False, 404),
        ("session-a", "in_progress", 4, 1.0, False, 400),
        ("session-a", "ready_to_apply", 3, 1.0, False, 400),
        ("session-a", "ready_to_apply", 4, 0.2, False, 400),
    ],
)
def test_apply_rejects_missing_incomplete_or_low_accuracy_sessions(
    sessions: dict[str, dict[str, Any]],
    session_id: str,
    status: str,
    point_count: int,
    accuracy: float,
    force_apply: bool,
    expected: int,
) -> None:
    if session_id != "missing":
        session = _session(
            status=status,
            points=[{"screen_x": 1, "screen_y": 1, "world_x": 0.1, "world_y": 0.1}]
            * point_count,
        )
        session["accuracy"] = accuracy
        sessions[session_id] = session

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            calibration_routes.apply_calibration_transformations(
                session_id,
                backup_previous=False,
                force_apply=force_apply,
                core_module=cast("Any", None),
            )
        )
    assert raised.value.status_code == expected


def test_start_camera_mode_records_selected_chessboard_contract(
    camera_session: dict[str, Any],
) -> None:
    response = asyncio.run(
        calibration_routes.start_camera_calibration_mode(
            chessboard_cols=8,
            chessboard_rows=5,
            square_size=0.03,
            min_images=12,
        )
    )

    assert response.mode_active is True
    assert response.chessboard_size == (8, 5)
    assert response.square_size == pytest.approx(0.03)
    assert response.min_images == 12
    assert camera_session["chessboard_size"] == (8, 5)
    assert camera_session["images"] == []


def test_start_camera_mode_rejects_second_active_session(
    camera_session: dict[str, Any],
) -> None:
    camera_session["active"] = True

    with pytest.raises(HTTPException) as raised:
        asyncio.run(calibration_routes.start_camera_calibration_mode())

    assert raised.value.status_code == 409


def test_camera_capture_requires_active_mode(camera_session: dict[str, Any]) -> None:
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            calibration_routes.capture_camera_calibration_image(
                vision_module=SimpleNamespace()
            )
        )
    assert raised.value.status_code == 400


def test_camera_capture_reports_missing_frame_as_camera_unavailable(
    camera_session: dict[str, Any],
) -> None:
    camera_session["active"] = True

    with pytest.raises(CustomHTTPException) as raised:
        asyncio.run(
            calibration_routes.capture_camera_calibration_image(
                vision_module=SimpleNamespace(get_frame=lambda **_kwargs: None)
            )
        )
    assert raised.value.status_code == 503


def test_camera_capture_without_chessboard_does_not_store_uncalibrated_image(
    monkeypatch: pytest.MonkeyPatch,
    camera_session: dict[str, Any],
) -> None:
    camera_session.update(active=True, min_images=2)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        calibration_routes.cv2,
        "findChessboardCorners",
        lambda _gray, _size, _flags: (False, None),
    )

    response = asyncio.run(
        calibration_routes.capture_camera_calibration_image(
            vision_module=SimpleNamespace(get_frame=lambda **_kwargs: frame)
        )
    )

    assert response.success is True
    assert response.chessboard_found is False
    assert response.corners_detected == 0
    assert response.total_images == 0
    assert response.can_process is False
    assert camera_session["images"] == []


def test_camera_capture_stores_detected_corners_and_encodes_optional_preview(
    monkeypatch: pytest.MonkeyPatch,
    camera_session: dict[str, Any],
) -> None:
    camera_session.update(active=True, min_images=1)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    corners = np.zeros((54, 1, 2), dtype=np.float32)
    monkeypatch.setattr(
        calibration_routes.cv2,
        "findChessboardCorners",
        lambda _gray, _size, _flags: (True, corners),
    )
    monkeypatch.setattr(
        calibration_routes.cv2, "cornerSubPix", lambda _gray, found, *_args: found
    )
    monkeypatch.setattr(
        calibration_routes.cv2, "drawChessboardCorners", lambda *_args: None
    )
    monkeypatch.setattr(
        calibration_routes.cv2,
        "imencode",
        lambda *_args: (True, np.frombuffer(b"preview", dtype=np.uint8)),
    )

    response = asyncio.run(
        calibration_routes.capture_camera_calibration_image(
            include_preview=True,
            vision_module=SimpleNamespace(get_frame=lambda **_kwargs: frame),
        )
    )

    assert response.chessboard_found is True
    assert response.corners_detected == 54
    assert response.total_images == 1
    assert response.can_process is True
    assert response.image_preview == base64.b64encode(b"preview").decode("ascii")
    assert camera_session["images"][0]["image"].shape == frame.shape


def test_process_camera_calibration_requires_active_mode_and_minimum_images(
    camera_session: dict[str, Any],
) -> None:
    with pytest.raises(HTTPException) as inactive:
        asyncio.run(calibration_routes.process_camera_calibration())
    assert inactive.value.status_code == 400

    camera_session.update(
        active=True, min_images=2, images=[{"image": np.zeros((4, 4, 3))}]
    )
    with pytest.raises(HTTPException) as insufficient:
        asyncio.run(calibration_routes.process_camera_calibration())
    assert insufficient.value.status_code == 400
    assert "captured 1" in str(insufficient.value.detail)


def test_camera_process_persists_fisheye_parameters_and_reports_quality(
    monkeypatch: pytest.MonkeyPatch,
    camera_session: dict[str, Any],
) -> None:
    camera_session.update(
        active=True,
        min_images=1,
        chessboard_size=(2, 2),
        square_size=0.04,
        images=[
            {
                "image": np.zeros((32, 48, 3), dtype=np.uint8),
                "corners": np.zeros((4, 1, 2), dtype=np.float32),
            }
        ],
    )
    persisted: dict[str, Any] = {}

    class FileStorageFake:
        def __init__(self, path: str, _mode: int) -> None:
            persisted["path"] = path

        def write(self, name: str, value: Any) -> None:
            persisted[name] = value

        def release(self) -> None:
            persisted["released"] = True

    def calibrate(*args: Any) -> tuple[float, np.ndarray, np.ndarray, Any, Any]:
        return 0.3, args[3], args[4], args[5], args[6]

    monkeypatch.setattr(calibration_routes.cv2, "FileStorage", FileStorageFake)
    monkeypatch.setattr(calibration_routes.cv2.fisheye, "calibrate", calibrate)
    monkeypatch.setattr(
        calibration_routes.cv2.fisheye,
        "CALIB_RECOMPUTE_EXTRINSIC",
        1,
        raising=False,
    )
    monkeypatch.setattr(
        calibration_routes.cv2.fisheye,
        "CALIB_CHECK_COND",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        calibration_routes.cv2.fisheye,
        "CALIB_FIX_SKEW",
        4,
        raising=False,
    )
    monkeypatch.setattr("os.makedirs", lambda *_args, **_kwargs: None)

    response = asyncio.run(
        calibration_routes.process_camera_calibration(
            save_to="calibration/test-camera.yaml"
        )
    )

    assert response.success is True
    assert response.quality_rating == "excellent"
    assert response.resolution == (48, 32)
    assert response.images_used == 1
    assert response.camera_matrix == np.zeros((3, 3)).tolist()
    assert response.distortion_coefficients == [0.0, 0.0, 0.0, 0.0]
    assert persisted["image_width"] == 48
    assert persisted["image_height"] == 32
    assert persisted["num_images_used"] == 1
    assert persisted["released"] is True


def test_stop_camera_mode_clears_captured_images(
    camera_session: dict[str, Any],
) -> None:
    camera_session.update(active=True, images=[{"id": "one"}, {"id": "two"}])

    response = asyncio.run(calibration_routes.stop_camera_calibration_mode())

    assert response.success is True
    assert response.mode_active is False
    assert "Captured 2 images" in response.message
    assert camera_session["active"] is False
    assert camera_session["images"] == []


def test_fisheye_table_calibration_reports_missing_camera_frame() -> None:
    with pytest.raises(CustomHTTPException) as raised:
        asyncio.run(
            calibration_routes.calibrate_fisheye_from_table(
                vision_module=SimpleNamespace(get_frame=lambda **_kwargs: None)
            )
        )

    assert raised.value.status_code == 503
