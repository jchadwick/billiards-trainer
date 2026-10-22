"""Deterministic lifecycle and persistence tests for calibration management."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from backend.vision.calibration.camera import CameraParameters
from backend.vision.calibration.color import ColorProfile, ColorThresholds
from backend.vision.calibration.geometry import (
    GeometricCalibration,
    PerspectiveCorrection,
)
from backend.vision.calibration.manager import CalibrationManager
from backend.vision.calibration.validation import CalibrationReport, ValidationResult

if TYPE_CHECKING:
    from pathlib import Path


def _camera_parameters() -> CameraParameters:
    return CameraParameters(
        camera_matrix=np.eye(3, dtype=np.float64),
        distortion_coefficients=np.zeros((5, 1), dtype=np.float64),
        resolution=(640, 480),
        calibration_error=0.2,
        calibration_date="2026-10-21T10:00:00",
    )


def _thresholds(hue_min: int = 40) -> ColorThresholds:
    return ColorThresholds(
        hue_min=hue_min,
        hue_max=80,
        saturation_min=30,
        saturation_max=255,
        value_min=20,
        value_max=230,
    )


def _color_profile(name: str = "test") -> ColorProfile:
    return ColorProfile(
        name=name,
        table_color=_thresholds(),
        ball_colors={"cue": _thresholds(0)},
        lighting_condition="test lighting",
        creation_date="2026-10-21T10:01:00",
        ambient_light_level=100.0,
    )


def _geometry_calibration() -> GeometricCalibration:
    return GeometricCalibration(
        perspective_correction=None,
        coordinate_mapping=None,
        table_dimensions_real=(2.54, 1.27),
        table_corners_pixel=[(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)],
        calibration_error=0.4,
        calibration_date="2026-10-21T10:02:00",
    )


def _report(score: float = 0.9) -> CalibrationReport:
    result = ValidationResult(
        test_name="manager-test",
        timestamp=123.0,
        passed=True,
        accuracy_score=score,
        error_metrics={},
        details={},
    )
    return CalibrationReport(
        session_id="validation-session",
        test_date="2026-10-21T10:03:00",
        camera_validation=result,
        color_validation=None,
        geometry_validation=None,
        integration_validation=None,
        overall_score=score,
        recommendations=[],
    )


@pytest.fixture
def manager(tmp_path: Path) -> CalibrationManager:
    """Constrain every manager cache and session artifact to the test directory."""
    return CalibrationManager(str(tmp_path / "calibration"))


def test_new_session_names_and_initial_status_are_consistent(
    manager: CalibrationManager,
) -> None:
    session_id = manager.start_new_session("table-1")

    assert re.fullmatch(r"table-1_[0-9a-f]{8}", session_id)
    assert manager.current_session is not None
    assert manager.current_session.session_id == session_id
    assert manager.current_session.system_ready is False
    assert manager.get_calibration_status() == {
        "session_active": True,
        "camera_calibrated": False,
        "color_calibrated": False,
        "geometry_calibrated": False,
        "system_ready": False,
        "session_id": session_id,
        "session_date": manager.current_session.created_date,
    }
    assert not manager.is_system_ready()
    assert manager.save_current_session()
    assert manager.list_sessions() == [session_id]


def test_calibration_flow_persists_components_validation_and_status(
    manager: CalibrationManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.start_new_session()
    camera = _camera_parameters()
    profile = _color_profile("practice")
    geometry = _geometry_calibration()
    observed: dict[str, Any] = {}

    def calibrate_intrinsics(images: list[np.ndarray]) -> tuple[bool, CameraParameters]:
        observed["images"] = images
        manager.camera_calibrator.camera_params = camera
        return True, camera

    monkeypatch.setattr(
        manager.camera_calibrator, "calibrate_intrinsics", calibrate_intrinsics
    )
    monkeypatch.setattr(
        manager.color_calibrator,
        "auto_calibrate_table_color",
        lambda frame: _thresholds(int(frame[0, 0, 0])),
    )
    monkeypatch.setattr(
        manager.color_calibrator,
        "calibrate_ball_colors",
        lambda _frame, samples: {"cue": _thresholds(len(samples))},
    )
    monkeypatch.setattr(
        manager.color_calibrator, "_estimate_ambient_light", lambda _f: 120.0
    )
    monkeypatch.setattr(
        manager.camera_calibrator, "undistort_image", lambda frame: frame + 1
    )

    def calibrate_geometry(
        frame: np.ndarray,
        corners: list[tuple[float, float]] | None,
        dimensions: tuple[float, float] | None,
    ) -> GeometricCalibration:
        observed["geometry_args"] = (frame.copy(), corners, dimensions)
        manager.geometry_calibrator.current_calibration = geometry
        return geometry

    monkeypatch.setattr(
        manager.geometry_calibrator, "calibrate_table_geometry", calibrate_geometry
    )
    monkeypatch.setattr(
        manager.validator, "generate_comprehensive_report", lambda *_args: _report()
    )

    images = [np.zeros((12, 16), dtype=np.uint8)]
    assert manager.calibrate_camera(images) == (True, camera)
    frame = np.full((4, 6, 3), 45, dtype=np.uint8)
    returned_profile = manager.calibrate_colors(
        frame, {"cue": [(0, 0, 1, 1)]}, profile_name=profile.name
    )
    assert returned_profile is not None
    assert returned_profile.name == "practice"
    assert returned_profile.table_color.hue_min == 45
    assert returned_profile.ball_colors["cue"].hue_min == 1
    assert returned_profile.ambient_light_level == 120.0
    assert (
        manager.calibrate_geometry(
            frame,
            [(0.0, 0.0)] * 4,
            (2.54, 1.27),
        )
        is geometry
    )
    report = manager.validate_system({"sentinel": 7})

    assert report is not None
    assert report.overall_score == 0.9
    assert manager.current_session is not None
    assert manager.current_session.system_ready
    assert manager.is_system_ready()
    assert observed["images"] is images
    geometry_frame, corners, dimensions = observed["geometry_args"]
    np.testing.assert_array_equal(geometry_frame, frame + 1)
    assert corners == [(0.0, 0.0)] * 4
    assert dimensions == (2.54, 1.27)

    saved_path = manager.sessions_dir / f"{manager.current_session.session_id}.json"
    saved_data = json.loads(saved_path.read_text())
    assert saved_data["camera_calibration"]["resolution"] == [640, 480]
    assert saved_data["color_profile"]["name"] == "practice"
    assert saved_data["geometric_calibration"]["table_dimensions_real"] == [2.54, 1.27]
    assert saved_data["validation_report"]["overall_score"] == 0.9

    restored = CalibrationManager(str(manager.base_dir))
    assert restored.load_session(manager.current_session.session_id)
    assert restored.current_session is not None
    assert json.loads(json.dumps(restored.current_session.to_dict())) == json.loads(
        json.dumps(manager.current_session.to_dict())
    )
    assert restored.camera_calibrator.camera_params is not None
    assert restored.color_calibrator.current_profile is not None
    assert restored.geometry_calibrator.current_calibration is not None
    assert restored.get_calibration_status()["camera_error"] == 0.2
    assert restored.get_calibration_status()["color_profile"] == "practice"
    assert restored.get_calibration_status()["geometry_error"] == 0.4


def test_validation_readiness_uses_strictly_greater_than_point_eight(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.start_new_session()
    reports = iter([_report(0.8), _report(0.80001)])
    monkeypatch.setattr(
        manager.validator, "generate_comprehensive_report", lambda *_args: next(reports)
    )

    first = manager.validate_system()
    assert first is not None
    assert first.overall_score == 0.8
    assert manager.current_session is not None
    assert manager.current_session.system_ready is False
    second = manager.validate_system({"case": "above threshold"})
    assert second is not None
    assert second.overall_score > 0.8
    assert manager.current_session.system_ready is True


def test_load_export_and_import_failures_do_not_claim_success(
    manager: CalibrationManager, tmp_path: Path
) -> None:
    manager.start_new_session()
    current_id = manager.current_session.session_id if manager.current_session else ""

    assert not manager.load_session("missing-session")
    assert manager.current_session is not None
    assert manager.current_session.session_id == current_id
    active_session = manager.current_session
    manager.current_session = None
    assert not manager.export_calibration(str(tmp_path / "missing-export.json"))
    manager.current_session = active_session
    assert not manager.import_calibration(str(tmp_path / "missing-import.json"))

    malformed_session = manager.sessions_dir / "malformed.json"
    malformed_session.write_text("not json")
    assert not manager.load_session("malformed")
    assert not manager.import_calibration(str(malformed_session))

    invalid_shape = tmp_path / "invalid-shape.json"
    invalid_shape.write_text(json.dumps({"export_version": "1.0"}))
    assert not manager.import_calibration(str(invalid_shape))


def test_export_import_refreshes_identity_and_persists_full_session(
    manager: CalibrationManager, tmp_path: Path
) -> None:
    source_id = manager.start_new_session("source")
    session = manager.current_session
    assert session is not None
    session.camera_calibration = _camera_parameters()
    session.color_profile = _color_profile()
    session.geometric_calibration = _geometry_calibration()
    session.validation_report = _report()
    session.system_ready = True
    export_path = tmp_path / "portable-calibration.json"

    assert manager.export_calibration(str(export_path))
    exported = json.loads(export_path.read_text())
    assert exported["export_version"] == "1.0"
    assert exported["session"]["session_id"] == source_id

    imported = CalibrationManager(str(tmp_path / "import-target"))
    assert imported.import_calibration(str(export_path))
    assert imported.current_session is not None
    assert imported.current_session.session_id.startswith("imported_")
    assert imported.current_session.session_id != source_id
    assert imported.current_session.system_ready
    assert imported.camera_calibrator.camera_params is not None
    assert imported.color_calibrator.current_profile is not None
    assert imported.geometry_calibrator.current_calibration is not None
    assert imported.current_session.session_id in imported.list_sessions()


def test_reset_specific_component_and_all_updates_session_and_readiness(
    manager: CalibrationManager,
) -> None:
    manager.start_new_session()
    manager.camera_calibrator.camera_params = _camera_parameters()
    manager.color_calibrator.current_profile = _color_profile()
    manager.geometry_calibrator.current_calibration = _geometry_calibration()
    session = manager.current_session
    assert session is not None
    session.camera_calibration = _camera_parameters()
    session.color_profile = _color_profile()
    session.geometric_calibration = _geometry_calibration()
    session.validation_report = _report()
    session.system_ready = True

    assert manager.reset_calibration("camera")
    status = manager.get_calibration_status()
    assert status["camera_calibrated"] is False
    reset_snapshot = json.loads(json.dumps(session.to_dict()))
    assert reset_snapshot["camera_calibration"] is None
    assert reset_snapshot["color_profile"] is not None
    assert reset_snapshot["system_ready"] is True

    assert manager.reset_calibration("color")
    status = manager.get_calibration_status()
    assert status["color_calibrated"] is False
    reset_snapshot = json.loads(json.dumps(session.to_dict()))
    assert reset_snapshot["color_profile"] is None
    assert manager.reset_calibration("geometry")
    status = manager.get_calibration_status()
    assert status["geometry_calibrated"] is False
    reset_snapshot = json.loads(json.dumps(session.to_dict()))
    assert reset_snapshot["geometric_calibration"] is None

    assert manager.reset_calibration()
    reset_snapshot = json.loads(json.dumps(session.to_dict()))
    assert reset_snapshot["validation_report"] is None
    assert reset_snapshot["system_ready"] is False


def test_fallback_readiness_without_session_requires_all_components(
    manager: CalibrationManager,
) -> None:
    assert not manager.is_system_ready()
    manager.camera_calibrator.camera_params = _camera_parameters()
    manager.color_calibrator.current_profile = _color_profile()
    assert not manager.is_system_ready()
    manager.geometry_calibrator.current_calibration = _geometry_calibration()

    assert manager.is_system_ready()
    status = manager.get_calibration_status()
    assert status["session_active"] is False
    assert status["camera_calibrated"] is True
    assert status["color_calibrated"] is True
    assert status["geometry_calibrated"] is True
    assert status["system_ready"] is True


def test_process_frame_applies_available_transforms_without_mutating_input(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    manager.camera_calibrator.camera_params = _camera_parameters()
    manager.geometry_calibrator.current_calibration = _geometry_calibration()
    correction = PerspectiveCorrection(
        source_points=[(0.0, 0.0)] * 4,
        target_points=[(0.0, 0.0)] * 4,
        transform_matrix=np.eye(3),
        inverse_matrix=np.eye(3),
        correction_quality=1.0,
    )
    geometry = manager.geometry_calibrator.current_calibration
    assert geometry is not None
    geometry.perspective_correction = correction
    observed: list[tuple[str, Any]] = []

    def undistort(value: np.ndarray) -> np.ndarray:
        observed.append(("camera", value.copy()))
        return value + 1

    def correct(value: np.ndarray, transform: PerspectiveCorrection) -> np.ndarray:
        observed.append(("geometry", transform))
        return value + 1

    monkeypatch.setattr(manager.camera_calibrator, "undistort_image", undistort)
    monkeypatch.setattr(
        manager.geometry_calibrator, "correct_keystone_distortion", correct
    )

    result = manager.process_frame(frame)

    np.testing.assert_array_equal(result, np.full_like(frame, 2))
    np.testing.assert_array_equal(frame, np.zeros_like(frame))
    assert observed[0][0] == "camera"
    assert observed[1] == ("geometry", correction)


def test_process_frame_returns_copied_input_when_transform_fails(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((2, 2), dtype=np.uint8)
    manager.camera_calibrator.camera_params = _camera_parameters()

    def fail(_frame: np.ndarray) -> np.ndarray:
        raise RuntimeError("camera transform failed")

    monkeypatch.setattr(manager.camera_calibrator, "undistort_image", fail)

    result = manager.process_frame(frame)

    np.testing.assert_array_equal(result, frame)
    assert result is not frame


def test_coordinate_conversion_delegates_to_geometry_calibrator(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    pixel = (12.5, 28.0)
    world = (1.2, 0.4)
    monkeypatch.setattr(
        manager.geometry_calibrator,
        "pixel_to_world_coordinates",
        lambda point: (point[0] / 10, point[1] / 10),
    )
    monkeypatch.setattr(
        manager.geometry_calibrator,
        "world_to_pixel_coordinates",
        lambda point: (point[0] * 10, point[1] * 10),
    )

    assert manager.pixel_to_world(pixel) == (1.25, 2.8)
    assert manager.world_to_pixel(world) == (12.0, 4.0)


def test_adaptive_recalibration_respects_threshold_and_persists_adaptation(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    assert not manager.adaptive_recalibration(frame)

    profile = _color_profile()
    manager.color_calibrator.current_profile = profile
    manager.start_new_session()
    monkeypatch.setattr(
        manager.color_calibrator, "_estimate_ambient_light", lambda _f: 120.0
    )
    monkeypatch.setattr(
        manager.color_calibrator,
        "adapt_to_lighting",
        lambda _f, _profile: _color_profile("adapted"),
    )
    assert not manager.adaptive_recalibration(frame, lighting_threshold=0.2)
    assert manager.adaptive_recalibration(frame, lighting_threshold=0.1)
    assert manager.color_calibrator.current_profile is not None
    assert manager.color_calibrator.current_profile.name == "adapted"
    assert manager.current_session is not None
    assert (
        manager.current_session.color_profile
        is manager.color_calibrator.current_profile
    )
    saved = json.loads(
        (
            manager.sessions_dir / f"{manager.current_session.session_id}.json"
        ).read_text()
    )
    assert saved["color_profile"]["name"] == "adapted"


def test_calibration_and_validation_exceptions_return_failure_values(
    manager: CalibrationManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        manager.camera_calibrator,
        "calibrate_intrinsics",
        lambda _images: (_ for _ in ()).throw(RuntimeError("camera failed")),
    )
    monkeypatch.setattr(
        manager.color_calibrator,
        "auto_calibrate_table_color",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("color failed")),
    )
    monkeypatch.setattr(
        manager.geometry_calibrator,
        "calibrate_table_geometry",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("geometry failed")),
    )
    monkeypatch.setattr(
        manager.validator,
        "generate_comprehensive_report",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("validation failed")),
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert manager.calibrate_camera([]) == (False, None)
    assert manager.calibrate_colors(frame) is None
    assert manager.calibrate_geometry(frame) is None
    assert manager.validate_system() is None


def test_context_manager_saves_active_session_and_does_not_suppress_errors(
    manager: CalibrationManager,
) -> None:
    session_id = manager.start_new_session("context")

    with manager as entered:
        assert entered is manager

    session_path = manager.sessions_dir / f"{session_id}.json"
    assert session_path.exists()
    assert json.loads(session_path.read_text())["session_id"] == session_id

    exception = RuntimeError("caller error")
    assert manager.__exit__(RuntimeError, exception, None) is False
