"""Behavioral tests for calibration validation and its report contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from backend.vision.calibration.color import ColorThresholds
from backend.vision.calibration.validation import (
    CalibrationReport,
    CalibrationValidator,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def validator(tmp_path: Path) -> CalibrationValidator:
    """Keep report files isolated from the repository and other tests."""
    return CalibrationValidator(cache_dir=str(tmp_path / "validation"))


def _camera_calibrator(*, camera_params: Any = object()) -> SimpleNamespace:
    return SimpleNamespace(
        camera_params=camera_params,
        chessboard_size=(2, 2),
        generate_chessboard_points=lambda: np.zeros((4, 3), dtype=np.float32),
        undistort_image=lambda frame: frame,
    )


def _color_thresholds(
    hue_min: int, hue_max: int, saturation_min: int = 100, value_min: int = 100
) -> ColorThresholds:
    return ColorThresholds(
        hue_min=hue_min,
        hue_max=hue_max,
        saturation_min=saturation_min,
        saturation_max=255,
        value_min=value_min,
        value_max=255,
    )


def _geometry_calibrator(
    *,
    mean_error: float = 0.01,
    max_error: float = 0.02,
    reverse_pixel: tuple[float, float] = (0.0, 0.0),
    perspective_quality: float = 0.9,
) -> SimpleNamespace:
    perspective = SimpleNamespace(correction_quality=perspective_quality)
    calibration = SimpleNamespace(
        perspective_correction=perspective,
        calibration_error=0.01,
        table_dimensions_real=(2.54, 1.27),
        table_corners_pixel=[(0.0, 0.0)] * 4,
    )
    return SimpleNamespace(
        current_calibration=calibration,
        validate_geometry=lambda _pixels, _world: {
            "mean_error": mean_error,
            "max_error": max_error,
        },
        world_to_pixel_coordinates=lambda _world: reverse_pixel,
        correct_keystone_distortion=lambda frame, _correction: frame,
    )


def test_validation_result_and_report_round_trip_with_optional_sections() -> None:
    result = ValidationResult(
        test_name="camera_calibration",
        timestamp=123.5,
        passed=True,
        accuracy_score=0.92,
        error_metrics={"mean_reprojection_error": 0.16},
        details={"images": 3},
    )
    report = CalibrationReport(
        session_id="session-12345678",
        test_date="2026-10-21T12:30:00",
        camera_validation=result,
        color_validation=None,
        geometry_validation=None,
        integration_validation=None,
        overall_score=0.92,
        recommendations=["ready"],
    )

    serialized = report.to_dict()

    assert serialized["camera_validation"] == result.to_dict()
    assert serialized["color_validation"] is None
    assert CalibrationReport.from_dict(serialized) == report
    assert ValidationResult.from_dict(result.to_dict()) == result


def test_camera_validation_reports_missing_parameters(
    validator: CalibrationValidator,
) -> None:
    result = validator.validate_camera_calibration(
        _camera_calibrator(camera_params=None), [np.zeros((8, 8), dtype=np.uint8)]
    )

    assert not result.passed
    assert result.accuracy_score == 0.0
    assert result.error_metrics == {"error": "No camera parameters available"}
    assert result.details["reason"] == "missing_parameters"


def test_camera_validation_scores_detected_and_missed_chessboards(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    projected = np.ones((4, 1, 2), dtype=np.float32)
    calibrator = _camera_calibrator(
        camera_params=SimpleNamespace(
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
            calibration_error=0.25,
        )
    )
    calls = iter([(True, corners), (False, None)])
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args: next(calls))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda _gray, found, *_args: found)
    monkeypatch.setattr(
        cv2, "solvePnP", lambda *_args: (True, np.zeros((3, 1)), np.ones((3, 1)))
    )
    monkeypatch.setattr(cv2, "projectPoints", lambda *_args: (projected, None))
    monkeypatch.setattr(cv2, "norm", lambda *_args: 2.0)
    monkeypatch.setattr(validator, "_test_distortion_correction", lambda *_args: 0.9)
    monkeypatch.setattr(
        "backend.vision.calibration.validation.config_manager.get",
        lambda key, default=None: (
            0.4 if key.endswith("min_detection_rate") else default
        ),
    )

    result = validator.validate_camera_calibration(
        calibrator,
        [np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)],
    )

    assert result.passed
    assert result.error_metrics["mean_reprojection_error"] == pytest.approx(0.5)
    assert result.error_metrics["max_reprojection_error"] == pytest.approx(0.5)
    assert result.error_metrics["detection_rate"] == pytest.approx(0.5)
    assert result.details == {
        "num_test_images": 2,
        "successful_detections": 1,
        "calibration_error": 0.25,
    }


def test_camera_validation_converts_detector_exception_to_failed_result(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cv2,
        "findChessboardCorners",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("opencv unavailable")),
    )

    result = validator.validate_camera_calibration(
        _camera_calibrator(), [np.zeros((8, 8), dtype=np.uint8)]
    )

    assert not result.passed
    assert result.error_metrics == {"exception": "opencv unavailable"}
    assert result.details == {
        "status": "exception",
        "error": "opencv unavailable",
    }


def test_camera_validation_rejects_pose_when_pnp_cannot_solve(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args: (True, corners))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda _gray, found, *_args: found)
    monkeypatch.setattr(
        cv2, "solvePnP", lambda *_args: (False, np.zeros((3, 1)), np.ones((3, 1)))
    )
    monkeypatch.setattr(validator, "_test_distortion_correction", lambda *_args: 0.9)

    result = validator.validate_camera_calibration(
        _camera_calibrator(
            camera_params=SimpleNamespace(
                camera_matrix=np.eye(3),
                distortion_coefficients=np.zeros(5),
                calibration_error=0.25,
            )
        ),
        [np.zeros((8, 8), dtype=np.uint8)],
    )

    assert not result.passed
    assert result.error_metrics["detection_rate"] == 1.0
    assert result.error_metrics["mean_reprojection_error"] == float("inf")
    assert result.details["successful_detections"] == 0


def test_distortion_quality_handles_missing_lines_and_detected_straight_lines(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibrator = _camera_calibrator()
    frame = np.zeros((16, 16), dtype=np.uint8)
    assert validator._test_distortion_correction(calibrator, None) == 0.0
    assert (
        validator._test_distortion_correction(
            _camera_calibrator(camera_params=None), frame
        )
        == 0.0
    )

    monkeypatch.setattr(cv2, "Canny", lambda *_args: np.zeros((16, 16), np.uint8))
    monkeypatch.setattr(cv2, "HoughLines", lambda *_args, **_kwargs: None)
    assert validator._test_distortion_correction(calibrator, frame) == 0.5

    straight_lines = np.zeros((6, 1, 2), dtype=np.float32)
    monkeypatch.setattr(cv2, "HoughLines", lambda *_args, **_kwargs: straight_lines)
    assert validator._test_distortion_correction(calibrator, frame) == pytest.approx(
        1.0
    )


def test_distortion_quality_contains_opencv_failures(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cv2,
        "Canny",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("edge failure")),
    )

    score = validator._test_distortion_correction(
        _camera_calibrator(), np.zeros((8, 8), dtype=np.uint8)
    )

    assert score == 0.0


def test_color_validation_uses_labeled_regions_and_profiles_threshold(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    hsv = np.zeros((4, 4, 3), dtype=np.uint8)
    hsv[:2, :2] = (5, 200, 200)
    hsv[2:, 2:] = (60, 200, 200)
    profile = SimpleNamespace(
        ball_colors={
            "red": _color_thresholds(0, 10),
            # Unlabeled profile classes should not affect this sample's FP rate.
            "blue": _color_thresholds(90, 110),
        },
        table_color=_color_thresholds(50, 70),
        ambient_light_level=0.7,
        name="test-profile",
    )
    calibrator = SimpleNamespace(current_profile=profile)
    monkeypatch.setattr(cv2, "cvtColor", lambda _frame, _conversion: hsv)
    validator.accuracy_thresholds["color_false_positive_rate"] = 0.3

    result = validator.validate_color_calibration(
        calibrator,
        np.zeros((4, 4, 3), dtype=np.uint8),
        {"red": [(0, 0, 2, 2)], "table": [(2, 2, 2, 2)], "unknown": [(0, 0, 1, 1)]},
    )

    assert result.passed
    assert result.accuracy_score == pytest.approx(1.0)
    assert result.error_metrics["false_positive_rate"] == pytest.approx(0.25)
    assert set(result.error_metrics["detection_results"]) == {"red", "table"}
    assert result.error_metrics["detection_results"]["red"]["f1_score"] == 1.0
    assert result.details == {
        "num_object_types": 2,
        "lighting_level": 0.7,
        "profile_name": "test-profile",
    }


def test_color_validation_missing_profile_and_exception(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    missing = validator.validate_color_calibration(
        SimpleNamespace(current_profile=None), frame, {}
    )
    assert not missing.passed
    assert missing.details["reason"] == "missing_profile"

    monkeypatch.setattr(
        cv2,
        "cvtColor",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bad frame")),
    )
    failed = validator.validate_color_calibration(
        SimpleNamespace(
            current_profile=SimpleNamespace(
                ball_colors={}, table_color=_color_thresholds(0, 10)
            )
        ),
        frame,
        {},
    )
    assert not failed.passed
    assert failed.error_metrics == {"exception": "bad frame"}


def test_color_validation_fails_when_false_positive_threshold_is_exceeded(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    hsv = np.full((2, 2, 3), (5, 200, 200), dtype=np.uint8)
    profile = SimpleNamespace(
        ball_colors={"red": _color_thresholds(0, 10)},
        table_color=_color_thresholds(50, 60),
        ambient_light_level=0.5,
        name="red-profile",
    )
    monkeypatch.setattr(cv2, "cvtColor", lambda *_args: hsv)

    result = validator.validate_color_calibration(
        SimpleNamespace(current_profile=profile),
        np.zeros((2, 2, 3), dtype=np.uint8),
        {"red": [(0, 0, 2, 2)]},
    )

    assert result.accuracy_score == 1.0
    assert result.error_metrics["false_positive_rate"] == 1.0
    assert not result.passed


def test_geometry_validation_scores_forward_reverse_and_perspective(
    validator: CalibrationValidator,
) -> None:
    result = validator.validate_geometric_calibration(
        _geometry_calibrator(), [(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]
    )

    assert result.passed
    assert result.error_metrics == {
        "forward_mean_error": 0.01,
        "forward_max_error": 0.02,
        "reverse_mean_error": 0.5,
        "perspective_quality": 0.9,
    }
    assert result.details["num_test_points"] == 2
    assert result.details["table_dimensions"] == (2.54, 1.27)


def test_geometry_validation_uses_strict_configured_thresholds(
    validator: CalibrationValidator,
) -> None:
    exact_thresholds = _geometry_calibrator(
        mean_error=0.02,
        reverse_pixel=(5.0, 0.0),
        perspective_quality=0.7,
    )

    result = validator.validate_geometric_calibration(
        exact_thresholds, [(0.0, 0.0)], [(0.0, 0.0)]
    )

    assert not result.passed
    assert result.error_metrics["reverse_mean_error"] == pytest.approx(5.0)
    assert result.accuracy_score == pytest.approx(0.7 / 3.0)


def test_geometry_validation_reports_missing_calibration_and_failures(
    validator: CalibrationValidator,
) -> None:
    missing = validator.validate_geometric_calibration(
        SimpleNamespace(current_calibration=None), [], []
    )
    assert not missing.passed
    assert missing.details["reason"] == "missing_calibration"

    broken = SimpleNamespace(
        current_calibration=SimpleNamespace(
            perspective_correction=None,
            calibration_error=0.0,
            table_dimensions_real=(1.0, 1.0),
        ),
        validate_geometry=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("geometry failure")
        ),
    )
    failed = validator.validate_geometric_calibration(broken, [], [])
    assert not failed.passed
    assert failed.error_metrics == {"exception": "geometry failure"}


def test_geometry_validation_handles_unavailable_reverse_mapping_and_perspective(
    validator: CalibrationValidator,
) -> None:
    calibrator = _geometry_calibrator()
    calibrator.world_to_pixel_coordinates = lambda _world: None
    calibrator.current_calibration.perspective_correction = None

    result = validator.validate_geometric_calibration(
        calibrator, [(0.0, 0.0)], [(0.0, 0.0)]
    )

    assert not result.passed
    assert result.error_metrics["reverse_mean_error"] == float("inf")
    assert result.error_metrics["perspective_quality"] == 0.0


def test_integration_validation_applies_calibrations_and_scores_components(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    camera = _camera_calibrator()
    camera_calls: list[np.ndarray] = []
    camera.undistort_image = lambda image: camera_calls.append(image) or image
    color = SimpleNamespace(current_profile=object())
    geometry = _geometry_calibrator()
    geometry.current_calibration.table_corners_pixel = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    detection_frames: list[np.ndarray] = []
    monkeypatch.setattr(
        validator,
        "_detect_balls_with_color_calibration",
        lambda image, _calibrator: detection_frames.append(image) or [(2.0, 2.0)],
    )

    result = validator.validate_integration(
        camera,
        color,
        geometry,
        frame,
        {
            "balls": [(2.0, 2.0)],
            "table_corners": geometry.current_calibration.table_corners_pixel,
        },
    )

    assert result.passed
    assert result.error_metrics == {
        "ball_detection_accuracy": 1.0,
        "table_detection_accuracy": 1.0,
    }
    assert camera_calls
    assert camera_calls[0] is not frame
    assert detection_frames
    assert detection_frames[0] is not frame
    assert result.details == {
        "camera_calibrated": True,
        "color_calibrated": True,
        "geometry_calibrated": True,
    }


def test_integration_validation_handles_missing_evidence_and_exceptions(
    validator: CalibrationValidator,
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    no_calibrations = validator.validate_integration(
        _camera_calibrator(camera_params=None),
        SimpleNamespace(current_profile=None),
        SimpleNamespace(current_calibration=None),
        frame,
        {"balls": [(1.0, 1.0)], "table_corners": [(0.0, 0.0)] * 4},
    )
    assert not no_calibrations.passed
    assert no_calibrations.accuracy_score == 0.0

    invalid_frame = validator.validate_integration(
        _camera_calibrator(),
        SimpleNamespace(current_profile=None),
        SimpleNamespace(current_calibration=None),
        None,  # type: ignore[arg-type]
        {},
    )
    assert not invalid_frame.passed
    assert "exception" in invalid_frame.error_metrics


def test_ball_detection_helper_finds_a_round_thresholded_object(
    validator: CalibrationValidator,
) -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(frame, (32, 44), 10, (0, 0, 255), -1)
    color = SimpleNamespace(
        current_profile=SimpleNamespace(ball_colors={"red": _color_thresholds(0, 10)})
    )

    centers = validator._detect_balls_with_color_calibration(frame, color)

    assert len(centers) == 1
    assert centers[0] == pytest.approx((32.0, 44.0), abs=1.0)


def test_ball_detection_helper_filters_small_degenerate_and_irregular_contours(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    contours = [object() for _ in range(5)]
    areas = {contours[0]: 50.0, **dict.fromkeys(contours[1:], 200.0)}
    perimeters = {
        contours[1]: 0.0,
        contours[2]: 100.0,
        contours[3]: 40.0,
        contours[4]: 40.0,
    }
    moments = {
        contours[3]: {"m00": 0.0, "m10": 0.0, "m01": 0.0},
        contours[4]: {"m00": 2.0, "m10": 20.0, "m01": 30.0},
    }
    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _conversion: frame)
    monkeypatch.setattr(cv2, "findContours", lambda *_args: (contours, None))
    monkeypatch.setattr(cv2, "contourArea", lambda contour: areas[contour])
    monkeypatch.setattr(cv2, "arcLength", lambda contour, _closed: perimeters[contour])
    monkeypatch.setattr(cv2, "moments", lambda contour: moments[contour])
    color = SimpleNamespace(
        current_profile=SimpleNamespace(ball_colors={"red": _color_thresholds(0, 10)})
    )

    centers = validator._detect_balls_with_color_calibration(
        np.zeros((8, 8, 3), dtype=np.uint8), color
    )

    assert centers == [(10.0, 15.0)]


def test_ball_detection_helper_returns_empty_without_profile(
    validator: CalibrationValidator,
) -> None:
    assert (
        validator._detect_balls_with_color_calibration(
            np.zeros((4, 4, 3), dtype=np.uint8),
            SimpleNamespace(current_profile=None),
        )
        == []
    )


@pytest.mark.parametrize(
    ("detected", "expected", "score"),
    [
        ([], [], 1.0),
        ([(1.0, 1.0)], [], 0.0),
        ([], [(1.0, 1.0)], 0.0),
        ([(1.0, 1.0)], [(2.0, 2.0), (90.0, 90.0)], 0.5),
    ],
)
def test_ball_accuracy_empty_and_partial_match_cases(
    validator: CalibrationValidator,
    detected: list[tuple[float, float]],
    expected: list[tuple[float, float]],
    score: float,
) -> None:
    assert validator._calculate_ball_detection_accuracy(detected, expected) == score


def test_ball_accuracy_uses_strict_configured_matching_radius(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.vision.calibration.validation.config_manager.get",
        lambda _key, default=None: 5.0 if default == 50.0 else default,
    )

    assert (
        validator._calculate_ball_detection_accuracy([(3.0, 4.0)], [(0.0, 0.0)]) == 0.0
    )


def test_table_accuracy_requires_four_ordered_corners_and_normalizes_error(
    validator: CalibrationValidator,
) -> None:
    expected = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    assert validator._calculate_table_detection_accuracy(expected, expected) == 1.0
    assert validator._calculate_table_detection_accuracy(expected[:3], expected) == 0.0

    shifted = [(10.0, y) for _x, y in expected]
    assert validator._calculate_table_detection_accuracy(
        shifted, expected
    ) == pytest.approx(0.5)


def test_comprehensive_report_includes_only_supplied_tests_and_persists(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera_result = ValidationResult(
        test_name="camera_calibration",
        timestamp=10.0,
        passed=False,
        accuracy_score=0.6,
        error_metrics={"mean_reprojection_error": 4.0, "detection_rate": 0.5},
        details={},
    )
    monkeypatch.setattr(
        validator,
        "validate_camera_calibration",
        lambda *_args: camera_result,
    )

    report = validator.generate_comprehensive_report(
        _camera_calibrator(),
        SimpleNamespace(),
        SimpleNamespace(),
        {"camera_test_images": [np.zeros((4, 4), dtype=np.uint8)]},
    )

    assert report.camera_validation == camera_result
    assert report.color_validation is None
    assert report.geometry_validation is None
    assert report.integration_validation is None
    assert report.overall_score == pytest.approx(0.6)
    assert any("Camera calibration" in item for item in report.recommendations)
    assert validator.load_report(report.session_id) == report


def test_comprehensive_report_runs_each_supplied_validation_and_averages_scores(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = {
        "camera": ValidationResult("camera", 1.0, True, 0.4, {}, {}),
        "color": ValidationResult("color", 2.0, True, 0.6, {}, {}),
        "geometry": ValidationResult("geometry", 3.0, True, 0.8, {}, {}),
        "integration": ValidationResult("integration", 4.0, True, 1.0, {}, {}),
    }
    monkeypatch.setattr(
        validator, "validate_camera_calibration", lambda *_args: results["camera"]
    )
    monkeypatch.setattr(
        validator, "validate_color_calibration", lambda *_args: results["color"]
    )
    monkeypatch.setattr(
        validator,
        "validate_geometric_calibration",
        lambda *_args: results["geometry"],
    )
    monkeypatch.setattr(
        validator, "validate_integration", lambda *_args: results["integration"]
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    report = validator.generate_comprehensive_report(
        _camera_calibrator(),
        SimpleNamespace(),
        SimpleNamespace(),
        {
            "camera_test_images": [frame],
            "color_test_frame": frame,
            "color_ground_truth": {"red": [(0, 0, 1, 1)]},
            "geometry_test_points": [(0.0, 0.0)],
            "geometry_expected_points": [(0.0, 0.0)],
            "integration_test_frame": frame,
            "integration_ground_truth": {"balls": []},
        },
    )

    assert report.camera_validation == results["camera"]
    assert report.color_validation == results["color"]
    assert report.geometry_validation == results["geometry"]
    assert report.integration_validation == results["integration"]
    assert report.overall_score == pytest.approx(0.7)


def test_recommendations_cover_each_failed_calibration_domain(
    validator: CalibrationValidator,
) -> None:
    camera = ValidationResult(
        "camera",
        0.0,
        False,
        0.0,
        {"mean_reprojection_error": 4.0, "detection_rate": 0.5},
        {},
    )
    color = ValidationResult(
        "color",
        0.0,
        False,
        0.0,
        {"overall_accuracy": 0.5, "false_positive_rate": 0.2},
        {},
    )
    geometry = ValidationResult(
        "geometry",
        0.0,
        False,
        0.0,
        {"forward_mean_error": 0.04, "perspective_quality": 0.5},
        {},
    )
    integration = ValidationResult("integration", 0.0, False, 0.0, {}, {})

    recommendations = validator._generate_recommendations(
        camera, color, geometry, integration
    )

    assert len(recommendations) == 7
    assert any("chessboard visibility" in item for item in recommendations)
    assert any("lighting and focus" in item for item in recommendations)
    assert any("color thresholds" in item for item in recommendations)
    assert any("Tighten color thresholds" in item for item in recommendations)
    assert any("Redetect table corners" in item for item in recommendations)
    assert any(
        "Perspective correction quality is low" in item for item in recommendations
    )
    assert any("integration needs improvement" in item for item in recommendations)


def test_recommendations_skip_metrics_that_remain_within_thresholds(
    validator: CalibrationValidator,
) -> None:
    camera = ValidationResult(
        "camera",
        0.0,
        False,
        0.0,
        {"mean_reprojection_error": 2.0, "detection_rate": 0.9},
        {},
    )
    color = ValidationResult(
        "color",
        0.0,
        False,
        0.0,
        {"overall_accuracy": 0.5, "false_positive_rate": 0.1},
        {},
    )
    geometry = ValidationResult(
        "geometry",
        0.0,
        False,
        0.0,
        {"forward_mean_error": 0.02, "perspective_quality": 0.5},
        {},
    )
    integration = ValidationResult("integration", 0.0, False, 0.0, {}, {})

    recommendations = validator._generate_recommendations(
        camera, color, geometry, integration
    )

    assert len(recommendations) == 3
    assert any("Color calibration accuracy is low" in item for item in recommendations)
    assert any(
        "Perspective correction quality is low" in item for item in recommendations
    )
    assert any("integration needs improvement" in item for item in recommendations)


def test_recommendations_and_report_loading_cover_success_and_invalid_json(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert validator._generate_recommendations(None, None, None, None) == [
        "All calibrations are performing well. System is ready for operation."
    ]
    assert validator.load_report("does-not-exist") is None

    report = CalibrationReport(
        session_id="bad-json-12345678",
        test_date="today",
        camera_validation=None,
        color_validation=None,
        geometry_validation=None,
        integration_validation=None,
        overall_score=0.0,
        recommendations=[],
    )
    report_path = validator.cache_dir / "validation_report_bad-json.json"
    report_path.write_text("{broken", encoding="utf-8")
    assert validator.load_report(report.session_id) is None

    monkeypatch.setattr(
        "builtins.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    validator._save_report(report)  # persistence failure is intentionally non-fatal


def test_stability_test_reports_scalar_and_position_variation(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 0.0}

    def advancing_time() -> float:
        clock["now"] += 0.1
        return clock["now"]

    monkeypatch.setattr(
        "backend.vision.calibration.validation.time.time", advancing_time
    )
    monkeypatch.setattr(
        "backend.vision.calibration.validation.time.sleep", lambda _s: None
    )
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    camera = _camera_calibrator()
    color = SimpleNamespace(
        current_profile=object(), _estimate_ambient_light=lambda _frame: 2.0
    )
    geometry = SimpleNamespace(
        current_calibration=object(),
        pixel_to_world_coordinates=lambda _point: (1.0, 2.0),
    )

    result = validator.stability_test(
        {"camera": camera, "color": color, "geometry": geometry},
        [frame],
        duration_minutes=1,
    )

    assert result["total_frames_tested"] > 0
    assert result["test_duration"] == 1
    assert result["camera"]["stability_score"] == pytest.approx(1.0)
    assert result["color"]["stability_score"] == pytest.approx(1.0)
    assert result["geometry"]["position_stability"] == pytest.approx(1.0)


def test_stability_test_rejects_empty_frames_and_allows_no_calibrators(
    validator: CalibrationValidator, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="At least one frame"):
        validator.stability_test({}, [], duration_minutes=0)

    monkeypatch.setattr("backend.vision.calibration.validation.time.time", lambda: 10.0)
    result = validator.stability_test({}, [np.zeros((2, 2), dtype=np.uint8)], 0)
    assert result == {"test_duration": 0, "total_frames_tested": 0}
