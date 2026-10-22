"""Deterministic behavioral coverage for camera and table calibration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pytest

from backend.vision.calibration.camera import (
    CameraCalibrator,
    CameraParameters,
    TableTransform,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def calibrator(tmp_path: Path) -> CameraCalibrator:
    """Keep all calibration cache writes inside pytest's temporary directory."""
    return CameraCalibrator(str(tmp_path / "camera-cache"))


def _camera_parameters() -> CameraParameters:
    return CameraParameters(
        camera_matrix=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0, 0, 1]]),
        distortion_coefficients=np.array([[0.1], [-0.02], [0.0], [0.0]]),
        resolution=(640, 480),
        calibration_error=0.25,
        calibration_date="2026-01-02T03:04:05",
    )


def _table_transform() -> TableTransform:
    return TableTransform(
        homography_matrix=np.array(
            [[0.002, 0.0, -1.0], [0.0, 0.002, -0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        table_corners_pixel=[(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)],
        table_corners_world=[(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)],
        table_dimensions=(2.0, 1.0),
        transformation_error=0.0,
    )


def test_chessboard_points_use_configured_pattern_and_square_size(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def get_config(key: str, default: Any = None) -> Any:
        if key.endswith("pattern_size"):
            return [3, 2]
        if key.endswith("square_size_meters"):
            return 0.04
        return default

    monkeypatch.setattr(
        "backend.vision.calibration.camera.config_manager.get", get_config
    )
    calibrator.chessboard_size = (3, 2)
    calibrator.square_size = 0.04

    points = calibrator.generate_chessboard_points()

    assert points.shape == (6, 3)
    np.testing.assert_allclose(
        points,
        [
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.08, 0.00, 0.0],
            [0.00, 0.04, 0.0],
            [0.04, 0.04, 0.0],
            [0.08, 0.04, 0.0],
        ],
    )


def test_intrinsic_calibration_rejects_too_few_images(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.vision.calibration.camera.config_manager.get",
        lambda key, default=None: (
            3 if key.endswith("min_calibration_images") else default
        ),
    )

    success, parameters = calibrator.calibrate_intrinsics(
        [np.zeros((30, 40), dtype=np.uint8)] * 2
    )

    assert not success
    assert parameters is None
    assert calibrator.camera_params is None


def test_intrinsic_calibration_rejects_too_few_detected_patterns(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.vision.calibration.camera.config_manager.get",
        lambda key, default=None: (
            2 if key.endswith("min_valid_detections") else default
        ),
    )
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args: (False, None))
    image = np.zeros((30, 40), dtype=np.uint8)

    success, parameters = calibrator.calibrate_intrinsics([image] * 5)

    assert not success
    assert parameters is None
    assert calibrator.camera_params is None


def test_intrinsic_calibration_refines_color_corners_saves_and_returns_parameters(
    calibrator: CameraCalibrator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corners = np.zeros((54, 1, 2), dtype=np.float32)
    refined = corners + 1
    camera_matrix = np.eye(3, dtype=np.float64)
    distortion = np.zeros((5, 1), dtype=np.float64)
    calibrate_call: dict[str, object] = {}
    monkeypatch.setattr(cv2, "cvtColor", lambda image, _code: image[:, :, 0])
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args: (True, corners))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda _gray, *_args: refined)

    def calibrate_camera(
        *args: Any,
    ) -> tuple[float, np.ndarray, np.ndarray, list[Any], list[Any]]:
        calibrate_call["image_size"] = args[2]
        calibrate_call["object_count"] = len(args[0])
        return (
            0.2,
            camera_matrix,
            distortion,
            [np.zeros(3) for _ in args[0]],
            [np.zeros(3) for _ in args[0]],
        )

    monkeypatch.setattr(cv2, "calibrateCamera", calibrate_camera)
    monkeypatch.setattr(cv2, "projectPoints", lambda *_args: (refined, None))
    monkeypatch.setattr(cv2, "norm", lambda *_args: 1.08)
    images = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(10)]

    success, parameters = calibrator.calibrate_intrinsics(images)

    assert success
    assert parameters is calibrator.camera_params
    assert parameters is not None
    assert parameters.resolution == (40, 30)
    assert parameters.calibration_error == pytest.approx(0.02)
    assert calibrate_call == {"image_size": (40, 30), "object_count": 10}
    saved = json.loads((tmp_path / "camera-cache" / "camera_params.json").read_text())
    assert saved["resolution"] == [40, 30]
    assert saved["calibration_error"] == pytest.approx(0.02)


def test_intrinsic_calibration_rejects_nonfinite_opencv_result(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    corners = np.zeros((54, 1, 2), dtype=np.float32)
    monkeypatch.setattr(cv2, "findChessboardCorners", lambda *_args: (True, corners))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda _gray, found, *_args: found)
    monkeypatch.setattr(
        cv2,
        "calibrateCamera",
        lambda *_args: (float("nan"), np.eye(3), np.zeros(5), [], []),
    )

    success, parameters = calibrator.calibrate_intrinsics(
        [np.zeros((30, 40), dtype=np.uint8) for _ in range(5)]
    )

    assert not success
    assert parameters is None
    assert calibrator.camera_params is None


def test_table_transform_maps_ordered_corners_and_uses_configured_dimensions(
    calibrator: CameraCalibrator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corners = [(10.0, 20.0), (810.0, 20.0), (810.0, 420.0), (10.0, 420.0)]
    observed: dict[str, Any] = {}

    def find_homography(
        source: np.ndarray, destination: np.ndarray, method: int, threshold: float
    ) -> tuple[np.ndarray, None]:
        observed.update(
            source=source, destination=destination, method=method, threshold=threshold
        )
        return (
            np.array([[0.0025, 0.0, -1.025], [0.0, 0.0025, -0.55], [0.0, 0.0, 1.0]]),
            None,
        )

    monkeypatch.setattr(cv2, "findHomography", find_homography)
    image = np.zeros((500, 900, 3), dtype=np.uint8)

    transform = calibrator.calibrate_table_transform(image, corners)

    assert transform is not None
    assert transform.table_dimensions == (2.54, 1.27)
    assert transform.table_corners_world == [
        (-1.27, -0.635),
        (1.27, -0.635),
        (1.27, 0.635),
        (-1.27, 0.635),
    ]
    assert observed["source"].shape == (4, 2)
    assert observed["method"] == cv2.RANSAC
    assert observed["threshold"] == 5.0
    saved = json.loads((tmp_path / "camera-cache" / "table_transform.json").read_text())
    assert saved["table_dimensions"] == [2.54, 1.27]


@pytest.mark.parametrize("corners", [[], [(0.0, 0.0)] * 3, [(0.0, 0.0)] * 5])
def test_table_transform_rejects_any_corner_count_other_than_four(
    calibrator: CameraCalibrator,
    monkeypatch: pytest.MonkeyPatch,
    corners: list[tuple[float, float]],
) -> None:
    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *_args: pytest.fail("homography must not run for invalid corners"),
    )

    assert (
        calibrator.calibrate_table_transform(
            np.zeros((20, 20), dtype=np.uint8), corners
        )
        is None
    )


def test_table_transform_reports_opencv_homography_failure(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "findHomography", lambda *_args: (None, None))

    transform = calibrator.calibrate_table_transform(
        np.zeros((20, 20), dtype=np.uint8),
        [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
        (2.0, 1.0),
    )

    assert transform is None
    assert calibrator.table_transform is None


def test_undistort_returns_original_when_parameters_are_unavailable(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2,
        "undistort",
        lambda *_args: pytest.fail("undistort must not run without camera parameters"),
    )

    assert calibrator.undistort_image(image) is image


def test_undistort_uses_cached_parameters_and_crops_valid_roi(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibrator.camera_params = _camera_parameters()
    source = np.arange(12 * 16, dtype=np.uint8).reshape((12, 16))
    expected = np.full((12, 16), 9, dtype=np.uint8)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        cv2,
        "getOptimalNewCameraMatrix",
        lambda matrix, distortion, size, alpha, new_size: (
            seen.update(matrix=matrix, distortion=distortion, size=size, alpha=alpha),
            (np.eye(3), (2, 3, 8, 6)),
        )[1],
    )
    monkeypatch.setattr(cv2, "undistort", lambda *_args: expected)

    undistorted = calibrator.undistort_image(source)

    assert undistorted.shape == (6, 8)
    assert np.all(undistorted == 9)
    assert seen["size"] == (16, 12)
    assert seen["alpha"] == 1
    assert seen["matrix"] is calibrator.camera_params.camera_matrix
    assert seen["distortion"] is calibrator.camera_params.distortion_coefficients


def test_undistort_accepts_explicit_parameters_and_preserves_image_for_empty_roi(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = np.zeros((12, 16, 3), dtype=np.uint8)
    output = np.full_like(source, 7)
    explicit_matrix = np.eye(3)
    explicit_distortion = np.zeros(4)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        cv2,
        "getOptimalNewCameraMatrix",
        lambda matrix, distortion, size, alpha, new_size: (
            seen.update(matrix=matrix, distortion=distortion, size=size),
            (matrix, (0, 0, 0, 0)),
        )[1],
    )
    monkeypatch.setattr(cv2, "undistort", lambda *_args: output)

    result = calibrator.undistort_image(source, explicit_matrix, explicit_distortion)

    assert result is output
    assert seen == {
        "matrix": explicit_matrix,
        "distortion": explicit_distortion,
        "size": (16, 12),
    }


def test_pixel_world_conversion_round_trips_and_missing_transform_returns_none(
    calibrator: CameraCalibrator,
) -> None:
    assert calibrator.pixel_to_world((10.0, 20.0)) is None
    assert calibrator.world_to_pixel((0.0, 0.0)) is None
    calibrator.table_transform = _table_transform()

    world = calibrator.pixel_to_world((250.0, 125.0))
    pixel = calibrator.world_to_pixel((0.25, -0.25))

    assert world == pytest.approx((-0.5, -0.25))
    assert pixel == pytest.approx((625.0, 125.0))


def test_manual_corner_adjustment_applies_offsets_before_recalibration(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    initial = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
    adjustments = [(1.0, 2.0), (-1.0, 2.0), (-1.0, -2.0), (1.0, -2.0)]
    captured: dict[str, Any] = {}
    expected = _table_transform()

    def calibrate(
        _image: np.ndarray, corners: list[tuple[float, float]]
    ) -> TableTransform:
        captured["image"] = _image
        captured["corners"] = corners
        return expected

    monkeypatch.setattr(calibrator, "calibrate_table_transform", calibrate)

    result = calibrator.manual_adjust_corners(image, initial, adjustments)

    assert result is expected
    assert captured == {
        "image": image,
        "corners": [(1.0, 2.0), (7.0, 2.0), (7.0, 6.0), (1.0, 6.0)],
    }


def test_calibration_validation_reports_round_trip_pixel_and_world_errors(
    calibrator: CameraCalibrator,
) -> None:
    calibrator.table_transform = _table_transform()

    report = calibrator.validate_calibration(
        np.zeros((10, 10), dtype=np.float64),
        [(0.1, 0.2), (0.8, -0.1)],
        [(550.0, 350.0), (900.0, 200.0)],
    )

    assert report["num_test_points"] == 2
    assert report["mean_pixel_error"] == pytest.approx(0.0, abs=1e-3)
    assert report["max_pixel_error"] == pytest.approx(0.0, abs=1e-3)
    assert report["mean_world_error"] == pytest.approx(0.0, abs=1e-6)
    assert report["max_world_error"] == pytest.approx(0.0, abs=1e-6)


def test_calibration_validation_rejects_mismatched_points_and_reports_unavailable_transform(
    calibrator: CameraCalibrator,
) -> None:
    with pytest.raises(ValueError, match="same length"):
        calibrator.validate_calibration(np.zeros((10, 10)), [(0.0, 0.0)], [])

    report = calibrator.validate_calibration(
        np.zeros((10, 10)), [(0.0, 0.0)], [(1.0, 1.0)]
    )

    assert report == {
        "mean_pixel_error": float("inf"),
        "max_pixel_error": float("inf"),
        "mean_world_error": float("inf"),
        "max_world_error": float("inf"),
        "num_test_points": 1,
    }


def test_json_cache_load_save_and_complete_export_import_round_trip(
    calibrator: CameraCalibrator, tmp_path: Path
) -> None:
    calibrator.camera_params = _camera_parameters()
    calibrator.table_transform = _table_transform()
    calibrator._save_camera_params()
    calibrator._save_table_transform()
    export_path = tmp_path / "calibration.json"
    assert calibrator.export_calibration(str(export_path))

    restored = CameraCalibrator(str(tmp_path / "camera-cache"))
    assert restored.load_camera_params()
    assert restored.load_table_transform()
    assert restored.camera_params is not None
    assert restored.table_transform is not None
    np.testing.assert_allclose(
        restored.camera_params.camera_matrix, calibrator.camera_params.camera_matrix
    )
    assert restored.table_transform.table_dimensions == (2.0, 1.0)

    imported = CameraCalibrator(str(tmp_path / "imported-cache"))
    assert imported.import_calibration(str(export_path))
    assert imported.camera_params is not None
    assert imported.table_transform is not None
    np.testing.assert_allclose(
        imported.camera_params.distortion_coefficients,
        calibrator.camera_params.distortion_coefficients,
    )
    np.testing.assert_allclose(
        imported.table_transform.homography_matrix,
        calibrator.table_transform.homography_matrix,
    )
    export_data = json.loads(export_path.read_text())
    assert export_data["calibrator_version"] == "1.0"


@pytest.mark.parametrize("loader", ["load_camera_params", "load_table_transform"])
def test_json_cache_loaders_recover_from_missing_or_malformed_files(
    calibrator: CameraCalibrator, loader: str, tmp_path: Path
) -> None:
    missing = getattr(calibrator, loader)(str(tmp_path / "missing.json"))
    assert not missing

    broken_file = tmp_path / f"broken-{loader}.json"
    broken_file.write_text("{not valid JSON", encoding="utf-8")
    malformed = getattr(calibrator, loader)(str(broken_file))

    assert not malformed
    if loader == "load_camera_params":
        assert calibrator.camera_params is None
    else:
        assert calibrator.table_transform is None


def test_import_accepts_partial_calibration_and_preserves_existing_other_half(
    calibrator: CameraCalibrator, tmp_path: Path
) -> None:
    camera = _camera_parameters()
    calibrator.table_transform = _table_transform()
    import_path = tmp_path / "camera-only.json"
    import_path.write_text(
        json.dumps({"camera_params": camera.to_dict()}), encoding="utf-8"
    )

    assert calibrator.import_calibration(str(import_path))
    assert calibrator.camera_params is not None
    assert calibrator.table_transform is not None
    np.testing.assert_allclose(
        calibrator.camera_params.camera_matrix, camera.camera_matrix
    )


def test_yaml_loader_reads_parameters_and_applies_resolution_fallback(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = np.eye(3)
    distortion = np.zeros((4, 1))

    class Node:
        def __init__(self, value: Any = None, empty: bool = False) -> None:
            self.value = value
            self.empty_value = empty

        def mat(self) -> Any:
            return self.value

        def real(self) -> float:
            return float(self.value)

        def string(self) -> str:
            return str(self.value)

        def empty(self) -> bool:
            return self.empty_value

    nodes = {
        "camera_matrix": Node(matrix),
        "dist_coeffs": Node(distortion),
        "image_width": Node(0),
        "image_height": Node(0),
        "calibration_error": Node(empty=True),
        "calibration_date": Node(empty=True),
    }

    class FileStorage:
        released = False

        def __init__(self, path: str, mode: int) -> None:
            assert path == "camera.yml"
            assert mode == cv2.FILE_STORAGE_READ

        def isOpened(self) -> bool:
            return True

        def getNode(self, name: str) -> Node:
            return nodes[name]

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(cv2, "FileStorage", FileStorage)

    assert calibrator.load_calibration("camera.yml")
    assert calibrator.camera_params is not None
    assert calibrator.camera_params.resolution == (1920, 1080)
    assert calibrator.camera_params.calibration_error == 0.0
    assert calibrator.camera_params.calibration_date == "unknown"


@pytest.mark.parametrize(
    ("camera_node", "distortion_node", "opened", "expected_release"),
    [
        (None, np.zeros((4, 1)), True, True),
        (np.eye(3), None, True, True),
        (np.eye(3), np.zeros((4, 1)), False, False),
    ],
)
def test_yaml_loader_fails_cleanly_for_unreadable_files_or_missing_nodes(
    calibrator: CameraCalibrator,
    monkeypatch: pytest.MonkeyPatch,
    camera_node: Any,
    distortion_node: Any,
    opened: bool,
    expected_release: bool,
) -> None:
    class Node:
        def __init__(self, value: Any) -> None:
            self.value = value

        def mat(self) -> Any:
            return self.value

    class FileStorage:
        released = False

        def __init__(self, *_args: Any) -> None:
            pass

        def isOpened(self) -> bool:
            return opened

        def getNode(self, name: str) -> Node:
            return Node(camera_node if name == "camera_matrix" else distortion_node)

        def release(self) -> None:
            self.released = True

    storage = FileStorage()
    monkeypatch.setattr(cv2, "FileStorage", lambda *_args: storage)

    assert not calibrator.load_calibration("missing.yml")
    assert storage.released is expected_release
    assert calibrator.camera_params is None


def test_yaml_save_creates_parent_and_writes_all_calibration_metadata(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parameters = _camera_parameters()
    writes: dict[str, Any] = {}

    class FileStorage:
        released = False

        def __init__(self, path: str, mode: int) -> None:
            assert path.endswith("nested/camera.yml")
            assert mode == cv2.FILE_STORAGE_WRITE

        def write(self, name: str, value: Any) -> None:
            writes[name] = value

        def release(self) -> None:
            self.released = True

    output_path = tmp_path / "nested" / "camera.yml"
    storage = FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    monkeypatch.setattr(cv2, "FileStorage", lambda *_args: storage)

    assert calibrator.save_fisheye_calibration_yaml(str(output_path), parameters)

    assert output_path.parent.is_dir()
    assert storage.released
    assert writes["camera_matrix"] is parameters.camera_matrix
    assert writes["dist_coeffs"] is parameters.distortion_coefficients
    assert writes["image_width"] == 640
    assert writes["image_height"] == 480
    assert writes["calibration_error"] == 0.25
    assert writes["calibration_date"] == "2026-01-02T03:04:05"
    assert writes["calibration_method"] == "table_rectangle"
    assert "rectangular geometry" in writes["notes"]


def test_yaml_save_reports_missing_parameters_and_opencv_exceptions(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert not calibrator.save_fisheye_calibration_yaml(str(tmp_path / "none.yml"))

    calibrator.camera_params = _camera_parameters()

    def fail_storage(*_args: Any) -> None:
        raise OSError("disk is unavailable")

    monkeypatch.setattr(cv2, "FileStorage", fail_storage)

    assert not calibrator.save_fisheye_calibration_yaml(str(tmp_path / "fail.yml"))


def test_fisheye_table_calibration_clamps_unreasonable_distortion_and_saves(
    calibrator: CameraCalibrator,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}

    def fake_calibrate(
        objects: list[np.ndarray],
        images: list[np.ndarray],
        size: tuple[int, int],
        matrix: np.ndarray,
        distortion: np.ndarray,
        **kwargs: Any,
    ) -> tuple[float, np.ndarray, np.ndarray, list[Any], list[Any]]:
        seen.update(
            objects=objects,
            images=images,
            size=size,
            matrix=matrix.copy(),
            kwargs=kwargs,
        )
        return (
            0.75,
            matrix,
            # OpenCV returns this standard column-vector shape in practice.
            np.array([0.8, -0.6, 0.0, 0.0, 0.0]).reshape(5, 1),
            [],
            [],
        )

    monkeypatch.setattr(cv2, "calibrateCamera", fake_calibrate)
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    corners = [(100.0, 100.0), (900.0, 100.0), (900.0, 500.0), (100.0, 500.0)]

    success, parameters = calibrator.calibrate_fisheye_from_table(
        image, corners, (2.0, 1.0)
    )

    assert success
    assert parameters is calibrator.camera_params
    assert parameters is not None
    assert parameters.resolution == (1000, 600)
    assert parameters.calibration_error == 0.75
    assert parameters.distortion_coefficients[:, 0] == pytest.approx(
        [0.25, -0.25, 0.0, 0.0]
    )
    assert seen["size"] == (1000, 600)
    assert seen["matrix"][0, 0] == pytest.approx(800.0)
    assert len(seen["objects"][0]) == 20
    assert len(seen["images"][0]) == 20
    assert "flags" in seen["kwargs"]
    assert (tmp_path / "camera-cache" / "camera_params.json").exists()


def test_fisheye_calibration_rejects_wrong_corner_count_and_recovers_from_opencv_error(
    calibrator: CameraCalibrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    assert calibrator.calibrate_fisheye_from_table(image, [(0.0, 0.0)] * 3) == (
        False,
        None,
    )

    monkeypatch.setattr(
        cv2,
        "calibrateCamera",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("solver failed")),
    )
    corners = [(100.0, 100.0), (900.0, 100.0), (900.0, 500.0), (100.0, 500.0)]

    success, parameters = calibrator.calibrate_fisheye_from_table(image, corners)

    assert not success
    assert parameters is None
    assert calibrator.camera_params is None
