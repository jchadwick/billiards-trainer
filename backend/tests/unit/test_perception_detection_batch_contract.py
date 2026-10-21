"""Validate the language-neutral frame-scoped detection batch contract."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

CONTRACT_ROOT = (
    Path(__file__).resolve().parents[3] / "perception/contracts/detection-batch/v1"
)
SCHEMA_PATH = CONTRACT_ROOT / "detection-batch.schema.json"
FIXTURE_NAMES = (
    "detection-batch-success.json",
    "detection-batch-empty-success.json",
    "detection-batch-failure.json",
    "detection-batch-restarted-session.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _read_json(SCHEMA_PATH)


@pytest.fixture(scope="module")
def validator(schema: dict[str, Any]) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_shared_detection_batch_fixtures_validate(
    validator: Draft202012Validator, fixture_name: str
) -> None:
    validator.validate(_read_json(CONTRACT_ROOT / fixture_name))


def test_empty_success_and_failure_are_distinct(
    validator: Draft202012Validator,
) -> None:
    empty = _read_json(CONTRACT_ROOT / "detection-batch-empty-success.json")
    failed = _read_json(CONTRACT_ROOT / "detection-batch-failure.json")

    validator.validate(empty)
    validator.validate(failed)
    assert empty["outcome"] == "success"
    assert empty["detections"] == []
    assert failed["outcome"] == "failed"
    assert failed["reason"] == "inference_timeout"


def test_detection_identity_is_scoped_to_the_full_source_observation(
    validator: Draft202012Validator,
) -> None:
    first = _read_json(CONTRACT_ROOT / "detection-batch-success.json")
    restarted = _read_json(CONTRACT_ROOT / "detection-batch-restarted-session.json")

    validator.validate(first)
    validator.validate(restarted)
    assert first["source"]["frame_id"] == restarted["source"]["frame_id"]
    assert (
        first["detections"][0]["detection_index"]
        == restarted["detections"][0]["detection_index"]
    )
    assert first["source"]["session_id"] != restarted["source"]["session_id"]
    assert first["engine_instance_id"] != restarted["engine_instance_id"]


@pytest.mark.parametrize(
    ("mutate", "validator_name", "path"),
    [
        (
            lambda batch: batch.update({"unknown": True}),
            "additionalProperties",
            (),
        ),
        (
            lambda batch: batch["detections"][0].update({"confidence": 1.5}),
            "maximum",
            ("detections", 0, "confidence"),
        ),
        (
            lambda batch: batch.update({"schema_version": 2}),
            "const",
            ("schema_version",),
        ),
        (
            lambda batch: batch.update({"outcome": "failed", "reason": "failed"}),
            "maxItems",
            ("detections",),
        ),
    ],
)
def test_schema_rejects_unknown_or_invalid_batch_data(
    validator: Draft202012Validator,
    mutate: Any,
    validator_name: str,
    path: tuple[str | int, ...],
) -> None:
    batch = copy.deepcopy(_read_json(CONTRACT_ROOT / "detection-batch-success.json"))
    mutate(batch)

    with pytest.raises(ValidationError) as error:
        validator.validate(batch)
    assert error.value.validator == validator_name
    assert tuple(error.value.absolute_path) == path


def test_still_replay_does_not_claim_a_source_timestamp(
    validator: Draft202012Validator,
) -> None:
    batch = _read_json(CONTRACT_ROOT / "detection-batch-empty-success.json")
    validator.validate(batch)
    assert batch["source"]["timestamp"] is None
