"""Deterministic contracts for table geometry calibration operations."""

import json

import pytest

from backend.vision.calibration import geometry as geometry_module
from backend.vision.calibration.geometry import (
    CoordinateMapping,
    GeometricCalibration,
    GeometricCalibrator,
    PerspectiveCorrection,
)

cv2 = geometry_module.cv2
np = geometry_module.np


@pytest.fixture
def calibrator(tmp_path) -> GeometricCalibrator:
    """Keep calibration cache writes isolated from the developer workspace."""
    return GeometricCalibrator(cache_dir=str(tmp_path / "geometry-cache"))


@pytest.fixture
def rectangle_corners() -> list[tuple[float, float]]:
    return [(40.0, 30.0), (440.0, 30.0), (440.0, 230.0), (40.0, 230.0)]


def test_manual_corners_are_used_without_touching_frame(
    calibrator: GeometricCalibrator,
) -> None:
    corners = [(1.0, 2.0), (8.0, 2.0), (8.0, 7.0), (1.0, 7.0)]

    assert (
        calibrator.detect_table_corners(np.zeros((2, 2, 3), dtype=np.uint8), corners)
        == corners
    )


def test_automatic_detection_falls_back_to_configured_frame_margin(
    calibrator: GeometricCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([], None))

    assert calibrator.detect_table_corners(frame) == [
        (50, 50),
        (590, 50),
        (590, 430),
        (50, 430),
    ]


def test_automatic_detection_finds_and_orders_rectangular_table(
    calibrator: GeometricCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    contour = np.array(
        [[60, 40], [440, 40], [440, 260], [60, 260]], dtype=np.int32
    ).reshape(-1, 1, 2)
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([contour], None))

    corners = calibrator.detect_table_corners(frame)

    assert len(corners) == 4
    assert corners == [(60, 40), (440, 40), (440, 260), (60, 260)]


def test_automatic_detection_filters_small_nonquadrilateral_and_lesser_contours(
    calibrator: GeometricCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    tiny = np.array([[1, 1], [11, 1], [11, 11], [1, 11]], dtype=np.int32).reshape(
        -1, 1, 2
    )
    triangle = np.array([[15, 15], [180, 15], [100, 150]], dtype=np.int32).reshape(
        -1, 1, 2
    )
    largest = np.array(
        [[40, 30], [440, 30], [440, 230], [40, 230]], dtype=np.int32
    ).reshape(-1, 1, 2)
    smaller = np.array(
        [[60, 50], [160, 50], [160, 150], [60, 150]], dtype=np.int32
    ).reshape(-1, 1, 2)
    monkeypatch.setattr(
        cv2, "findContours", lambda *_args: ([tiny, triangle, largest, smaller], None)
    )

    assert calibrator.detect_table_corners(frame) == [
        (40, 30),
        (440, 30),
        (440, 230),
        (40, 230),
    ]


def test_corner_sort_is_clockwise_from_top_left(
    calibrator: GeometricCalibrator,
) -> None:
    scrambled = [(440, 230), (40, 30), (40, 230), (440, 30)]

    assert calibrator._sort_corners_clockwise(scrambled) == [
        (40.0, 30.0),
        (440.0, 30.0),
        (440.0, 230.0),
        (40.0, 230.0),
    ]


def test_corner_sort_handles_points_without_two_top_corners(
    calibrator: GeometricCalibrator,
) -> None:
    points = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    assert len(calibrator._sort_corners_clockwise(points)) == 3


def test_perspective_transform_maps_source_corners_and_round_trips(
    calibrator: GeometricCalibrator, rectangle_corners: list[tuple[float, float]]
) -> None:
    target = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]

    correction = calibrator.calculate_perspective_transform(
        rectangle_corners, target_points=target
    )
    source_points = np.array(rectangle_corners, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(source_points, correction.transform_matrix)
    restored = cv2.perspectiveTransform(mapped, correction.inverse_matrix)

    np.testing.assert_allclose(mapped.reshape(-1, 2), target, atol=1e-4)
    np.testing.assert_allclose(restored.reshape(-1, 2), rectangle_corners, atol=1e-4)
    assert correction.correction_quality == pytest.approx(1.0)


@pytest.mark.parametrize("points", [[], [(0, 0)], [(0, 0), (1, 0), (1, 1)]])
def test_perspective_transform_rejects_non_quadrilaterals(
    calibrator: GeometricCalibrator, points: list[tuple[int, int]]
) -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        calibrator.calculate_perspective_transform(points)


def test_perspective_transform_uses_requested_target_size(
    calibrator: GeometricCalibrator, rectangle_corners: list[tuple[float, float]]
) -> None:
    correction = calibrator.calculate_perspective_transform(
        rectangle_corners, target_size=(320, 180)
    )

    assert correction.target_points == [(0, 0), (320, 0), (320, 180), (0, 180)]


def test_correct_keystone_returns_input_when_disabled_and_warps_when_configured(
    calibrator: GeometricCalibrator, rectangle_corners: list[tuple[float, float]]
) -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    frame[100, 200] = (20, 100, 240)
    correction = calibrator.calculate_perspective_transform(
        rectangle_corners,
        target_points=[(0, 0), (400, 0), (400, 200), (0, 200)],
    )

    assert calibrator.correct_keystone_distortion(frame, None) is frame
    corrected = calibrator.correct_keystone_distortion(frame, correction)
    assert corrected.shape == (200, 400, 3)
    assert corrected.dtype == np.uint8


def test_coordinate_mapping_centers_table_and_round_trips(
    calibrator: GeometricCalibrator, rectangle_corners: list[tuple[float, float]]
) -> None:
    mapping = calibrator.create_coordinate_mapping(rectangle_corners, (2.0, 1.0))

    assert mapping.pixel_bounds == ((40.0, 30.0), (440.0, 230.0))
    assert mapping.world_bounds == ((-1.0, -0.5), (1.0, 0.5))
    assert mapping.scale_factor == pytest.approx(200.0)
    assert calibrator.pixel_to_world_coordinates(
        (240.0, 130.0), mapping
    ) == pytest.approx((0.0, 0.0))
    pixel = calibrator.world_to_pixel_coordinates((0.3, -0.2), mapping)
    assert pixel == pytest.approx((300.0, 90.0))
    assert calibrator.pixel_to_world_coordinates(pixel, mapping) == pytest.approx(
        (0.3, -0.2)
    )


def test_coordinate_conversion_applies_rotation_in_both_directions(
    calibrator: GeometricCalibrator,
) -> None:
    mapping = CoordinateMapping(
        pixel_bounds=((0.0, 0.0), (200.0, 200.0)),
        world_bounds=((-1.0, -1.0), (1.0, 1.0)),
        scale_factor=100.0,
        rotation_angle=37.0,
        translation=(100.0, 100.0),
    )

    pixel = calibrator.world_to_pixel_coordinates((0.4, -0.25), mapping)

    assert calibrator.pixel_to_world_coordinates(pixel, mapping) == pytest.approx(
        (0.4, -0.25)
    )


def test_coordinate_conversion_without_current_mapping_preserves_input(
    calibrator: GeometricCalibrator,
) -> None:
    pixel = (123.5, 92.0)
    world = (-0.4, 0.8)

    assert calibrator.pixel_to_world_coordinates(pixel) == pixel
    assert calibrator.world_to_pixel_coordinates(world) == world


def test_coordinate_conversion_uses_current_calibration_mapping(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
) -> None:
    calibrator.current_calibration = GeometricCalibration(
        perspective_correction=None,
        coordinate_mapping=calibrator.create_coordinate_mapping(
            rectangle_corners, (2.0, 1.0)
        ),
        table_dimensions_real=(2.0, 1.0),
        table_corners_pixel=rectangle_corners,
        calibration_error=0.0,
        calibration_date="2026-01-01T00:00:00",
    )

    assert calibrator.world_to_pixel_coordinates((0.0, 0.0)) == pytest.approx(
        (240.0, 130.0)
    )


def test_table_calibration_persists_and_loads_cache(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
) -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)

    calibration = calibrator.calibrate_table_geometry(
        frame, rectangle_corners, (2.0, 1.0)
    )

    assert calibration is calibrator.current_calibration
    assert calibration.table_corners_pixel == rectangle_corners
    assert calibration.coordinate_mapping is not None
    assert calibration.perspective_correction is not None
    assert calibration.calibration_error == pytest.approx(0.0, abs=1e-4)
    cache_data = json.loads(
        (calibrator.cache_dir / "geometric_calibration.json").read_text()
    )
    assert cache_data["table_dimensions_real"] == [2.0, 1.0]

    restored = GeometricCalibrator(cache_dir=str(calibrator.cache_dir / "restored"))
    assert restored.load_calibration(
        str(calibrator.cache_dir / "geometric_calibration.json")
    )
    assert restored.current_calibration is not None
    assert restored.current_calibration.table_corners_pixel == [
        list(point) for point in rectangle_corners
    ]


def test_table_calibration_uses_standard_dimensions_when_omitted(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
) -> None:
    calibration = calibrator.calibrate_table_geometry(
        np.zeros((300, 500, 3), dtype=np.uint8), rectangle_corners
    )

    assert calibration.table_dimensions_real == (
        calibrator.standard_table_width,
        calibrator.standard_table_height,
    )


def test_validation_reports_error_statistics_and_rejects_invalid_inputs(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
) -> None:
    calibration = GeometricCalibration(
        perspective_correction=None,
        coordinate_mapping=calibrator.create_coordinate_mapping(
            rectangle_corners, (2.0, 1.0)
        ),
        table_dimensions_real=(2.0, 1.0),
        table_corners_pixel=rectangle_corners,
        calibration_error=0.0,
        calibration_date="2026-01-01T00:00:00",
    )
    calibrator.current_calibration = calibration

    result = calibrator.validate_geometry(
        [(240.0, 130.0), (340.0, 130.0)], [(0.0, 0.0), (0.5, 0.0)]
    )

    assert result["mean_error"] == pytest.approx(0.0)
    assert result["max_error"] == pytest.approx(0.0)
    assert result["std_error"] == pytest.approx(0.0)
    assert result["num_test_points"] == 2
    assert result["errors"] == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        calibrator.validate_geometry([(1.0, 1.0)], [])


def test_validation_requires_calibration(calibrator: GeometricCalibrator) -> None:
    with pytest.raises(ValueError, match="No calibration"):
        calibrator.validate_geometry([(1.0, 1.0)], [(0.0, 0.0)])


def test_barrel_correction_is_noop_for_zero_coefficients_and_preserves_shape(
    calibrator: GeometricCalibrator,
) -> None:
    frame = np.full((64, 96, 3), 127, dtype=np.uint8)

    assert calibrator.correct_barrel_distortion(frame) is frame
    corrected = calibrator.correct_barrel_distortion(frame, k1=0.1, k2=-0.02)
    assert corrected.shape == frame.shape
    assert corrected.dtype == frame.dtype


def test_rectification_grid_uses_width_height_and_draws_regular_axes(
    calibrator: GeometricCalibrator,
) -> None:
    grid = calibrator.create_rectification_grid((61, 45), grid_size=20)

    assert grid.shape == (45, 61, 3)
    assert grid.dtype == np.uint8
    assert tuple(grid[10, 20]) == (0, 255, 0)
    assert tuple(grid[20, 10]) == (0, 255, 0)
    assert tuple(grid[11, 11]) == (0, 0, 0)


def test_rectification_grid_uses_configured_default_spacing(
    calibrator: GeometricCalibrator,
) -> None:
    grid = calibrator.create_rectification_grid((42, 42))

    assert tuple(grid[20, 20]) == (0, 255, 0)
    assert tuple(grid[21, 21]) == (0, 0, 0)


def test_calibration_json_round_trips_perspective_and_coordinate_data(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
) -> None:
    correction = calibrator.calculate_perspective_transform(rectangle_corners)
    mapping = calibrator.create_coordinate_mapping(rectangle_corners, (2.0, 1.0))
    calibration = GeometricCalibration(
        correction,
        mapping,
        (2.0, 1.0),
        rectangle_corners,
        0.25,
        "2026-01-01T00:00:00",
    )

    restored = GeometricCalibration.from_dict(
        json.loads(json.dumps(calibration.to_dict()))
    )

    assert restored.table_dimensions_real == (2.0, 1.0)
    assert restored.calibration_error == pytest.approx(0.25)
    assert restored.perspective_correction is not None
    np.testing.assert_allclose(
        restored.perspective_correction.transform_matrix,
        correction.transform_matrix,
    )
    correction_round_trip = PerspectiveCorrection.from_dict(correction.to_dict())
    np.testing.assert_allclose(
        correction_round_trip.inverse_matrix, correction.inverse_matrix
    )
    assert restored.coordinate_mapping is not None
    assert restored.coordinate_mapping.scale_factor == pytest.approx(
        mapping.scale_factor
    )


def test_load_import_and_export_report_missing_or_malformed_files(
    calibrator: GeometricCalibrator, tmp_path
) -> None:
    assert not calibrator.load_calibration(str(tmp_path / "missing.json"))
    assert not calibrator.import_calibration(str(tmp_path / "missing.json"))
    assert not calibrator.export_calibration(str(tmp_path / "empty.json"))
    calibrator._save_calibration()

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json")
    assert not calibrator.import_calibration(str(malformed))


def test_export_reports_filesystem_failure(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
    tmp_path,
) -> None:
    calibrator.calibrate_table_geometry(
        np.zeros((300, 500, 3), dtype=np.uint8), rectangle_corners, (2.0, 1.0)
    )

    assert not calibrator.export_calibration(str(tmp_path / "missing" / "out.json"))


def test_import_and_export_round_trip_calibration_file(
    calibrator: GeometricCalibrator,
    rectangle_corners: list[tuple[float, float]],
    tmp_path,
) -> None:
    calibrator.calibrate_table_geometry(
        np.zeros((300, 500, 3), dtype=np.uint8), rectangle_corners, (2.0, 1.0)
    )
    export_path = tmp_path / "exported.json"

    assert calibrator.export_calibration(str(export_path))
    imported = GeometricCalibrator(cache_dir=str(tmp_path / "imported"))
    assert imported.import_calibration(str(export_path))
    assert imported.current_calibration is not None
    assert imported.current_calibration.calibration_error == pytest.approx(
        0.0, abs=1e-4
    )
