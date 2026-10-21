"""Verify export of retained Python calibration into the Rust bundle contract."""

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from backend.vision.calibration.camera import CameraCalibrator, TableTransform
from backend.vision.calibration.perception_export import build_table_calibration_v1

CONTRACT_ROOT = (
    Path(__file__).resolve().parents[3] / "perception/contracts/calibration/v1"
)


def _calibrated_transform(tmp_path: Path) -> TableTransform:
    calibrator = CameraCalibrator(str(tmp_path / "calibration-cache"))
    image = np.zeros((500, 1000, 3), dtype=np.uint8)
    transform = calibrator.calibrate_table_transform(
        image,
        [(0.0, 0.0), (1000.0, 0.0), (1000.0, 500.0), (0.0, 500.0)],
        (2.0, 1.0),
    )
    assert transform is not None
    return transform


def _export(transform: TableTransform) -> dict[str, object]:
    return build_table_calibration_v1(
        transform,
        calibration_revision=3,
        camera_profile_id="test-camera-profile",
        rectification_revision=2,
        image_space="rectified",
        source_width_px=1000,
        source_height_px=500,
    )


def test_python_meter_calibration_exports_schema_valid_millimeter_homography(
    tmp_path: Path,
) -> None:
    transform = _calibrated_transform(tmp_path)
    bundle = _export(transform)
    schema = json.loads((CONTRACT_ROOT / "table-calibration.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(bundle)

    assert bundle["table_width_mm"] == pytest.approx(2000.0)
    assert bundle["table_height_mm"] == pytest.approx(1000.0)
    assert bundle["coordinate_frame"] == "center_origin_x_corner0_to1_y_corner0_to3"
    matrix = np.asarray(bundle["pixel_to_table_mm"])
    center = matrix @ np.array([500.0, 250.0, 1.0])
    assert center[0] / center[2] == pytest.approx(0.0, abs=1e-6)
    assert center[1] / center[2] == pytest.approx(0.0, abs=1e-6)


def test_python_export_matches_shared_perspective_bundle_fixture(
    tmp_path: Path,
) -> None:
    calibrator = CameraCalibrator(str(tmp_path / "calibration-cache"))
    image = np.zeros((500, 1000, 3), dtype=np.uint8)
    transform = calibrator.calibrate_table_transform(
        image,
        [(30.0, 20.0), (970.0, 45.0), (930.0, 480.0), (55.0, 455.0)],
        (2.0, 1.0),
    )
    assert transform is not None

    exported = _export(transform)
    rust_fixture = json.loads(
        (CONTRACT_ROOT / "table-calibration-python-export.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        key: value for key, value in exported.items() if key != "pixel_to_table_mm"
    } == {
        key: value for key, value in rust_fixture.items() if key != "pixel_to_table_mm"
    }
    np.testing.assert_allclose(
        exported["pixel_to_table_mm"],
        rust_fixture["pixel_to_table_mm"],
        rtol=1e-7,
        atol=1e-7,
    )


def test_export_requires_the_legacy_transform_center_origin_convention(
    tmp_path: Path,
) -> None:
    transform = _calibrated_transform(tmp_path)
    transform.table_corners_world[0] = (0.0, 0.0)

    with pytest.raises(ValueError, match="center-origin frame"):
        _export(transform)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"camera_profile_id": ""}, "camera_profile_id"),
        ({"calibration_revision": True}, "calibration_revision"),
        ({"source_width_px": 0}, "source image dimensions"),
        ({"source_width_px": 4_294_967_296}, "source image dimensions"),
        ({"image_space": "cropped"}, "image_space"),
    ],
)
def test_export_requires_explicit_valid_source_provenance(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    transform = _calibrated_transform(tmp_path)
    arguments: dict[str, object] = {
        "calibration_revision": 3,
        "camera_profile_id": "test-camera-profile",
        "rectification_revision": 2,
        "image_space": "rectified",
        "source_width_px": 1000,
        "source_height_px": 500,
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        build_table_calibration_v1(transform, **arguments)


def test_export_rejects_transform_that_does_not_map_its_declared_corners(
    tmp_path: Path,
) -> None:
    transform = _calibrated_transform(tmp_path)
    transform.homography_matrix[0, 2] += 0.25

    with pytest.raises(ValueError, match="does not map its ordered corners"):
        _export(transform)
