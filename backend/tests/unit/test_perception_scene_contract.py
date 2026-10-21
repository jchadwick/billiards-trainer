"""Validate language-neutral perception scene fixtures and their semantics."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "perception/contracts/v1"
SCHEMA_PATH = CONTRACT_ROOT / "scene-snapshot.schema.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _read_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "scene-success.json",
        "scene-failure.json",
        "scene-identity-update.json",
        "scene-restarted-session.json",
        "scene-rust-tracked-success.json",
    ],
)
def test_versioned_scene_fixtures_validate(
    validator: Draft202012Validator, fixture_name: str
):
    validator.validate(_read_json(CONTRACT_ROOT / fixture_name))


def test_successful_empty_observation_is_distinct_from_failure(
    validator: Draft202012Validator,
):
    success = _read_json(CONTRACT_ROOT / "scene-success.json")
    success["balls"] = []
    failure = _read_json(CONTRACT_ROOT / "scene-failure.json")

    validator.validate(success)
    validator.validate(failure)
    assert success["outcome"] == "success"
    assert success["balls"] == []
    assert failure["outcome"] == "failed"
    assert failure["reason"] == "inference_timeout"


@pytest.mark.parametrize("status", ["observed", "not_observed"])
def test_cue_evidence_has_independent_observed_state(
    validator: Draft202012Validator, status: str
):
    scene = _read_json(CONTRACT_ROOT / "scene-success.json")
    if status == "observed":
        cue = {
            "status": status,
            "last_observation": {
                "session_id": scene["source"]["session_id"],
                "frame_id": scene["source"]["frame_id"],
                "timestamp": scene["source"]["timestamp"],
            },
            "image_center_px": {"x": 1700.0, "y": 1000.0},
            "image_space": "raw",
            "image_width_px": 3840,
            "image_height_px": 2160,
            "confidence": 0.88,
        }
    else:
        cue = {
            "status": status,
            "last_observation": {
                "session_id": scene["source"]["session_id"],
                "frame_id": scene["source"]["frame_id"],
                "timestamp": scene["source"]["timestamp"],
            },
            "image_center_px": None,
            "image_space": "raw",
            "image_width_px": 3840,
            "image_height_px": 2160,
            "confidence": None,
        }
    scene["cue"] = cue

    validator.validate(scene)


def test_observed_cue_cannot_omit_its_location(validator: Draft202012Validator):
    scene = _read_json(CONTRACT_ROOT / "scene-success.json")
    scene["cue"].update({"status": "observed"})

    with pytest.raises(ValidationError):
        validator.validate(scene)


def test_invalid_calibration_cannot_publish_table_coordinates(
    validator: Draft202012Validator,
):
    scene = _read_json(CONTRACT_ROOT / "scene-success.json")
    scene["revisions"]["calibration_state"] = "invalid"

    with pytest.raises(ValidationError):
        validator.validate(scene)


def test_numbered_identity_requires_confidence_and_evidence(
    validator: Draft202012Validator,
):
    scene = _read_json(CONTRACT_ROOT / "scene-success.json")
    scene["balls"][0]["identity"].update({"number": 8, "confidence": 0.91})

    with pytest.raises(ValidationError):
        validator.validate(scene)


def test_spatial_uncertainty_is_not_conflated_with_detector_confidence(
    validator: Draft202012Validator,
):
    scene = _read_json(CONTRACT_ROOT / "scene-success.json")
    measurement = scene["balls"][0]["measurement"]
    measurement["detector_confidence"] = 0.99
    measurement["table_position_uncertainty"] = {
        "status": "available",
        "sigma_x_mm": 1.2,
        "sigma_y_mm": 0.8,
    }

    validator.validate(scene)


def test_identity_annotation_does_not_refresh_position_observation(
    validator: Draft202012Validator,
):
    observed = _read_json(CONTRACT_ROOT / "scene-success.json")
    annotated = _read_json(CONTRACT_ROOT / "scene-identity-update.json")

    validator.validate(observed)
    validator.validate(annotated)
    assert annotated["update_kind"] == "annotation"
    assert annotated["snapshot_revision"] > observed["snapshot_revision"]
    assert annotated["transport_sequence"] > observed["transport_sequence"]
    assert annotated["source"] == observed["source"]
    assert int(annotated["published_at"]["ticks"]) > int(
        observed["published_at"]["ticks"]
    )
    assert annotated["balls"][0]["track_id"] == observed["balls"][0]["track_id"]
    assert (
        annotated["balls"][0]["last_observation"]
        == observed["balls"][0]["last_observation"]
    )
    assert annotated["balls"][0]["measurement"] == observed["balls"][0]["measurement"]
    assert annotated["balls"][0]["identity"]["number"] == 8


def test_restarted_session_can_reuse_frame_counter_without_identity_collision(
    validator: Draft202012Validator,
):
    previous = _read_json(CONTRACT_ROOT / "scene-success.json")
    restarted = _read_json(CONTRACT_ROOT / "scene-restarted-session.json")
    previous["source"]["frame_id"] = "frame-1"
    previous["balls"][0]["last_observation"]["frame_id"] = "frame-1"

    validator.validate(previous)
    validator.validate(restarted)
    assert previous["source"]["frame_id"] == restarted["source"]["frame_id"]
    assert previous["source"]["session_id"] != restarted["source"]["session_id"]
    assert previous["engine_instance_id"] != restarted["engine_instance_id"]


@pytest.mark.parametrize(
    ("mutate", "validator_name", "field_path"),
    [
        (
            lambda scene: scene.update({"unknown_field": True}),
            "additionalProperties",
            "unknown_field",
        ),
        (
            lambda scene: scene["balls"][0]["identity"].update({"confidence": 1.2}),
            "oneOf",
            "confidence",
        ),
        (
            lambda scene: scene["balls"][0].update({"track_id": ""}),
            "minLength",
            "track_id",
        ),
        (
            lambda scene: scene["source"]["timestamp"].update({"ticks": 912345670000}),
            "type",
            "ticks",
        ),
        (
            lambda scene: scene["balls"][0].update({"measurement": None}),
            "type",
            "measurement",
        ),
        (
            lambda scene: scene.update({"outcome": "failed", "reason": None}),
            "type",
            "reason",
        ),
    ],
)
def test_schema_rejects_invalid_or_ambiguous_scene_data(
    validator: Draft202012Validator,
    mutate: Any,
    validator_name: str,
    field_path: str,
):
    scene = copy.deepcopy(_read_json(CONTRACT_ROOT / "scene-success.json"))
    mutate(scene)

    with pytest.raises(ValidationError) as error:
        validator.validate(scene)
    assert error.value.validator == validator_name
    assert field_path in error.value.message or field_path in {
        str(path_part) for path_part in error.value.absolute_path
    }
