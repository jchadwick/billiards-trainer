"""Request-bound Rust capture acknowledgement validation tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]
import pytest

from backend.perception.capture_acknowledgement import (
    CaptureAcknowledgementError,
    validate_capture_session_acknowledgement,
)
from backend.perception.capture_configuration import (
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
    resolve_capture_settings,
)

CONTRACT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "perception"
    / "contracts"
    / "capture-session-ack"
    / "v1"
)
CAPTURE_REQUEST_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "perception"
    / "contracts"
    / "capture-settings"
    / "v1"
    / "capture-settings-default.json"
)
V2_CONTRACT_ROOT = CONTRACT_ROOT.parent / "v2"


class DefaultConfiguration:
    """Select the numeric camera source represented by the request fixtures."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}
        vision = self.values.setdefault("vision", {})
        assert isinstance(vision, dict)
        camera = vision.setdefault("camera", {})
        assert isinstance(camera, dict)
        camera.setdefault("video_file_path", "")

    def get_all(self) -> dict[str, Any]:
        return self.values


def _settings() -> CaptureSettingsSnapshotV1:
    snapshot = resolve_capture_settings(DefaultConfiguration(), schema_version=1)
    assert isinstance(snapshot, CaptureSettingsSnapshotV1)
    return snapshot


def _settings_v2() -> CaptureSettingsSnapshotV2:
    snapshot = resolve_capture_settings(
        DefaultConfiguration(
            {"perception": {"capture": {"output_resolution": [1280, 720]}}}
        ),
        schema_version=2,
    )
    assert isinstance(snapshot, CaptureSettingsSnapshotV2)
    return snapshot


def _settings_v3(values: dict[str, object] | None = None) -> CaptureSettingsSnapshotV3:
    snapshot = resolve_capture_settings(DefaultConfiguration(values))
    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    return snapshot


def _acknowledgement(name: str = "capture-session-ack-default.json") -> dict[str, Any]:
    value = json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("capture acknowledgement fixture root must be an object")
    return value


def _acknowledgement_v2() -> dict[str, Any]:
    settings = _settings_v2()
    value = json.loads(
        (V2_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise AssertionError("v2 acknowledgement fixture root must be an object")
    value["request_sha256"] = hashlib.sha256(
        settings.to_json().encode("utf-8")
    ).hexdigest()
    report = value["source_report"]
    report["requested_profile"] = settings.requested["requested_profile"]
    report["negotiated_profile"] = settings.requested["expected_negotiated_profile"]
    report["applied_frame_geometry"]["camera_profile_id"] = settings.requested[
        "camera_profile_id"
    ]
    report["output_frame_geometry"] = settings.requested["output_frame_geometry"]
    return value


def _acknowledgement_v3(
    settings: CaptureSettingsSnapshotV3,
) -> dict[str, Any]:
    value = json.loads(
        (V2_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise AssertionError("v3 acknowledgement fixture must be an object")
    value["request_sha256"] = hashlib.sha256(
        settings.to_json().encode("utf-8")
    ).hexdigest()
    report = value["source_report"]
    report["requested_profile"] = settings.requested["requested_profile"]
    report["negotiated_profile"] = settings.requested["expected_negotiated_profile"]
    report["applied_frame_geometry"]["camera_profile_id"] = settings.requested[
        "camera_profile_id"
    ]
    report["output_frame_geometry"] = settings.requested["output_frame_geometry"]
    return value


def test_default_request_and_compatible_ack_fixture_validate_together() -> None:
    settings = _settings()
    expected_request = json.loads(CAPTURE_REQUEST_FIXTURE.read_text(encoding="utf-8"))
    acknowledgement = _acknowledgement()

    assert json.loads(settings.to_json()) == expected_request
    validated = validate_capture_session_acknowledgement(
        settings,
        acknowledgement,
        expected_session_id="capture-session-20261020-1",
    )

    assert validated.configuration_revision == settings.configuration_revision
    assert validated.request_sha256 == acknowledgement["request_sha256"]
    assert validated.session_id == "capture-session-20261020-1"
    assert validated.source_report["applied_controls"]["contrast"] == {
        "status": "unsupported"
    }


def test_v2_ack_fixture_validates_against_complete_v2_schema() -> None:
    schema = json.loads(
        (V2_CONTRACT_ROOT / "capture-session-ack.schema.json").read_text(
            encoding="utf-8"
        )
    )
    fixture = json.loads(
        (V2_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(fixture)


def test_default_v2_ack_fixture_matches_effective_request() -> None:
    settings = resolve_capture_settings(DefaultConfiguration(), schema_version=2)
    fixture = json.loads(
        (V2_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )

    validated = validate_capture_session_acknowledgement(
        settings,
        fixture,
        expected_session_id=fixture["source_report"]["session_id"],
    )

    assert validated.request_sha256 == fixture["request_sha256"]
    assert validated.source_report["output_frame_geometry"] == {
        "width_px": 1920,
        "height_px": 1080,
    }


def test_v3_source_defaults_accept_unknown_controls_and_preserve_report() -> None:
    settings = _settings_v3()
    acknowledgement = _acknowledgement_v3(settings)
    report = acknowledgement["source_report"]
    report["capabilities"]["automatic_exposure"] = "unknown"
    report["capabilities"]["gain_control"] = "unsupported"
    report["applied_controls"]["exposure_mode"] = {"status": "unknown"}
    report["applied_controls"]["gain"] = {"status": "unsupported"}

    validated = validate_capture_session_acknowledgement(
        settings,
        acknowledgement,
        expected_session_id=report["session_id"],
    )

    assert validated.source_report["capabilities"]["automatic_exposure"] == "unknown"
    assert validated.source_report["capabilities"]["gain_control"] == "unsupported"
    assert validated.source_report["applied_controls"]["gain"] == {
        "status": "unsupported"
    }


def test_v3_require_policy_still_rejects_missing_control_readback() -> None:
    settings = _settings_v3({"vision": {"camera": {"control_policy": "require"}}})
    acknowledgement = _acknowledgement_v3(settings)
    acknowledgement["source_report"]["applied_controls"]["gain"] = {"status": "unknown"}

    with pytest.raises(CaptureAcknowledgementError, match="readback is unavailable"):
        validate_capture_session_acknowledgement(
            settings,
            acknowledgement,
            expected_session_id=acknowledgement["source_report"]["session_id"],
        )


def test_v2_ack_matches_request_digest_profiles_controls_and_emitted_geometry() -> None:
    settings = _settings_v2()
    acknowledgement = _acknowledgement_v2()

    validated = validate_capture_session_acknowledgement(
        settings,
        acknowledgement,
        expected_session_id="capture-session-20261020-1",
    )

    assert validated.configuration_revision == settings.configuration_revision
    assert validated.request_sha256 == acknowledgement["request_sha256"]
    assert validated.source_report["output_frame_geometry"] == {
        "width_px": 1280,
        "height_px": 720,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["source_report"].pop("output_frame_geometry"),
            "schema v2",
        ),
        (
            lambda value: value["source_report"].update(
                output_frame_geometry={"width_px": 640, "height_px": 480}
            ),
            "output geometry differs",
        ),
        (
            lambda value: value.update(unexpected_field=True),
            "schema v2",
        ),
        (
            lambda value: value["source_report"]["requested_profile"].update(
                width_px=1280
            ),
            "requested profile differs",
        ),
        (
            lambda value: value["source_report"]["applied_controls"]["gain"].update(
                value=2.0
            ),
            "readback differs: gain",
        ),
        (
            lambda value: value.update(configuration_revision=2),
            "revision is stale",
        ),
        (lambda value: value.update(request_sha256="0" * 64), "digest differs"),
        (
            lambda value: value["source_report"].update(device_id="device-index:1"),
            "device differs",
        ),
    ],
)
def test_v2_ack_rejects_missing_or_mismatched_request_evidence(
    mutate: Any, message: str
) -> None:
    acknowledgement = _acknowledgement_v2()
    mutate(acknowledgement)

    with pytest.raises(CaptureAcknowledgementError, match=message):
        validate_capture_session_acknowledgement(
            _settings_v2(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_acknowledgement_report_is_returned_as_a_detached_copy() -> None:
    acknowledgement = _acknowledgement()
    validated = validate_capture_session_acknowledgement(
        _settings(),
        acknowledgement,
        expected_session_id="capture-session-20261020-1",
    )

    report = validated.source_report
    report["device_id"] = "mutated"
    assert validated.source_report["device_id"] == "device-index:0"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("configuration_revision", 2),
        ("request_sha256", "0" * 64),
    ],
)
def test_stale_revision_or_different_request_digest_cannot_establish_readiness(
    field: str, replacement: object
) -> None:
    acknowledgement = _acknowledgement()
    acknowledgement[field] = replacement

    with pytest.raises(CaptureAcknowledgementError):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_wrong_source_session_identity_is_rejected() -> None:
    with pytest.raises(CaptureAcknowledgementError, match="session ID differs"):
        validate_capture_session_acknowledgement(
            _settings(),
            _acknowledgement(),
            expected_session_id="another-session",
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("device_id",), "device-index:1", "device differs"),
        (("requested_profile", "width_px"), 1280, "requested profile differs"),
        (("negotiated_profile", "height_px"), 720, "negotiated profile differs"),
        (("applied_frame_geometry", "configuration_revision"), 2, "geometry differs"),
        (
            ("applied_frame_geometry", "camera_profile_id"),
            "other-profile",
            "geometry differs",
        ),
        (("applied_frame_geometry", "image_space"), "rectified", "geometry differs"),
        (("applied_frame_geometry", "rectification_revision"), 1, "geometry differs"),
    ],
)
def test_source_report_must_match_device_profiles_and_applied_geometry(
    path: tuple[str, ...], replacement: object, message: str
) -> None:
    acknowledgement = _acknowledgement()
    target = acknowledgement["source_report"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(CaptureAcknowledgementError, match=message):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_explicit_backend_request_rejects_a_different_opened_backend() -> None:
    settings = resolve_capture_settings(
        DefaultConfiguration({"vision": {"camera": {"backend": "v4l2"}}}),
        schema_version=1,
    )
    acknowledgement = _acknowledgement()
    acknowledgement["request_sha256"] = hashlib.sha256(
        settings.to_json().encode("utf-8")
    ).hexdigest()
    acknowledgement["source_report"]["applied_frame_geometry"]["camera_profile_id"] = (
        settings.requested["camera_profile_id"]
    )

    with pytest.raises(CaptureAcknowledgementError, match="backend differs"):
        validate_capture_session_acknowledgement(
            settings,
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_timestamp_semantics_and_required_capabilities_must_match() -> None:
    acknowledgement = _acknowledgement()
    acknowledgement["source_report"]["timestamp_semantics"] = {
        "kind": "source_timestamp",
        "unit": "nanoseconds",
        "clock": "monotonic",
        "domain_id": "camera-clock",
        "stage": "exposure",
    }

    with pytest.raises(CaptureAcknowledgementError, match="timestamp semantics differ"):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )

    acknowledgement = _acknowledgement()
    acknowledgement["source_report"]["capabilities"][
        "automatic_exposure"
    ] = "unsupported"
    with pytest.raises(CaptureAcknowledgementError, match="capability is unavailable"):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("exposure_mode", {"status": "unknown"}),
        ("gain", {"status": "unknown"}),
        ("gain", {"status": "unsupported"}),
        ("gain", {"status": "applied", "value": 0.5}),
    ],
)
def test_required_control_readbacks_must_be_applied_and_match(
    field: str, replacement: dict[str, object]
) -> None:
    acknowledgement = _acknowledgement()
    acknowledgement["source_report"]["applied_controls"][field] = replacement

    with pytest.raises(CaptureAcknowledgementError):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_optional_unsupported_or_unknown_controls_do_not_block_readiness() -> None:
    acknowledgement = _acknowledgement()
    validated = validate_capture_session_acknowledgement(
        _settings(),
        acknowledgement,
        expected_session_id="capture-session-20261020-1",
    )

    controls = validated.source_report["applied_controls"]
    assert controls["brightness"] == {"status": "unknown"}
    assert controls["contrast"] == {"status": "unsupported"}


@pytest.mark.parametrize(
    ("filename", "outcome"),
    [
        ("capture-session-ack-rejected.json", "rejected"),
        ("capture-session-ack-default.json", "failed"),
        ("capture-session-ack-default.json", "stopped"),
    ],
)
def test_noncompatible_outcomes_never_return_a_ready_acknowledgement(
    filename: str, outcome: str
) -> None:
    acknowledgement = _acknowledgement(filename)
    if outcome != "rejected":
        acknowledgement["outcome"] = outcome
        acknowledgement["failure"] = {
            "category": "runtime_failed" if outcome == "failed" else "stopped"
        }

    with pytest.raises(CaptureAcknowledgementError) as error:
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id=acknowledgement["source_report"]["session_id"],
        )
    assert error.value.outcome == outcome


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(request_sha256="not-a-sha256"),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(outcome="ready"),
        lambda value: value.update(outcome="compatible", source_report=None),
    ],
)
def test_acknowledgement_schema_rejects_malformed_or_unknown_fields(
    mutate: Any,
) -> None:
    acknowledgement = _acknowledgement()
    mutate(acknowledgement)

    with pytest.raises(CaptureAcknowledgementError, match="schema v1"):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_acknowledgement_rejects_non_integer_snapshot_schema_version() -> None:
    malformed_settings = replace(cast("Any", _settings()), schema_version=1.0)

    with pytest.raises(
        CaptureAcknowledgementError,
        match="schema version does not match its snapshot type",
    ):
        validate_capture_session_acknowledgement(
            malformed_settings,
            _acknowledgement(),
            expected_session_id="capture-session-20261020-1",
        )


def test_nonfinite_readbacks_are_rejected_before_schema_semantics() -> None:
    acknowledgement = _acknowledgement()
    acknowledgement["source_report"]["applied_controls"]["gain"]["value"] = float("inf")

    with pytest.raises(CaptureAcknowledgementError, match="finite JSON"):
        validate_capture_session_acknowledgement(
            _settings(),
            acknowledgement,
            expected_session_id="capture-session-20261020-1",
        )


def test_json_schema_accepts_shared_compatible_and_rejected_fixtures() -> None:
    schema = json.loads(
        (CONTRACT_ROOT / "capture-session-ack.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    validator.validate(_acknowledgement())
    validator.validate(_acknowledgement("capture-session-ack-rejected.json"))
