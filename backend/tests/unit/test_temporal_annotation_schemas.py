"""Validate the versioned Rust temporal annotation interchange contracts."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "perception/contracts"


def _read_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def _v1_sample() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": {
            "video_sha256": "a" * 64,
            "width": 1280,
            "height": 720,
            "coordinate_space": "source_pixels",
        },
        "frames": [
            {
                "source_frame_index": 0,
                "presentation_time_seconds": 0.0,
                "expected_calibration_revision": None,
                "expected_source_session_epoch": None,
                "balls": [
                    {
                        "ball_id": "ball-1",
                        "status": "visible",
                        "center": {"x": 400.0, "y": 300.0},
                        "radius_px": 12.0,
                    }
                ],
                "cue": {"status": "absent"},
            }
        ],
    }


def _v2_sample() -> dict[str, Any]:
    return json.loads(
        (CONTRACT_ROOT / "v2/temporal-localization-annotations-v2.example.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def validators() -> dict[int, Draft202012Validator]:
    return {
        1: Draft202012Validator(
            _read_schema(
                CONTRACT_ROOT / "v1/temporal-localization-annotations-v1.schema.json"
            )
        ),
        2: Draft202012Validator(
            _read_schema(
                CONTRACT_ROOT / "v2/temporal-localization-annotations-v2.schema.json"
            )
        ),
    }


def test_legacy_pixel_only_annotation_schema_remains_valid(
    validators: dict[int, Draft202012Validator],
):
    validators[1].validate(_v1_sample())


def test_physical_annotation_schema_accepts_complete_and_unavailable_frames(
    validators: dict[int, Draft202012Validator],
):
    validators[2].validate(_v2_sample())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sample: sample["frames"][0]["balls"][0].pop("center_table_mm"),
        lambda sample: sample["frames"][0].update(
            physical_ground_truth_status="unavailable"
        ),
        lambda sample: sample["sequence"]["physical_ground_truth"].update(
            coordinate_frame="center_origin_x_right_y_up"
        ),
        lambda sample: sample["frames"][0].update(
            physical_ground_truth_status="unavailable", balls=[]
        ),
    ],
)
def test_physical_annotation_schema_rejects_incomplete_or_ambiguous_truth(
    validators: dict[int, Draft202012Validator], mutate,
):
    sample = copy.deepcopy(_v2_sample())
    mutate(sample)

    with pytest.raises(ValidationError):
        validators[2].validate(sample)


def test_v1_rejects_unversioned_physical_fields(
    validators: dict[int, Draft202012Validator],
):
    sample = _v1_sample()
    sample["sequence"]["physical_ground_truth"] = _v2_sample()["sequence"][
        "physical_ground_truth"
    ]

    with pytest.raises(ValidationError):
        validators[1].validate(sample)
