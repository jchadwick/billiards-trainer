"""Validate the shared table-calibration v1 contract fixture."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONTRACT_ROOT = (
    Path(__file__).resolve().parents[3] / "perception/contracts/calibration/v1"
)
SCHEMA_PATH = CONTRACT_ROOT / "table-calibration.schema.json"
FIXTURE_PATH = CONTRACT_ROOT / "table-calibration-center-origin.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = _read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_shared_center_origin_calibration_fixture_is_valid(
    validator: Draft202012Validator,
):
    fixture = _read_json(FIXTURE_PATH)

    validator.validate(fixture)

    assert fixture["coordinate_frame"] == "center_origin_x_corner0_to1_y_corner0_to3"
    assert fixture["table_width_mm"] == 2000.0
    assert fixture["table_height_mm"] == 1000.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("calibration_revision", -1),
        ("camera_profile_id", ""),
        ("image_space", "cropped"),
        ("source_width_px", 0),
        ("source_height_px", 4_294_967_296),
        ("coordinate_frame", "top_left_origin"),
        ("table_width_mm", 0.0),
        ("pixel_to_table_mm", [[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_calibration_schema_rejects_invalid_bundle_fields(
    validator: Draft202012Validator, field: str, value: object
):
    fixture = _read_json(FIXTURE_PATH)
    fixture[field] = value

    with pytest.raises(ValidationError):
        validator.validate(fixture)


def test_calibration_schema_rejects_unknown_bundle_fields(
    validator: Draft202012Validator,
):
    fixture = copy.deepcopy(_read_json(FIXTURE_PATH))
    fixture["unversioned_scale_override"] = 10.0

    with pytest.raises(ValidationError):
        validator.validate(fixture)
