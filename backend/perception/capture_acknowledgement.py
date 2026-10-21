"""Validate Rust's request-bound capture-session acknowledgement.

The parser is a pure control-boundary function. It does not open hardware or
make a producer ready; callers may do so only after this function returns a
validated compatible report. Source-reported values remain claims from the
adapter and are not physical-device proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from backend.perception.capture_configuration import (
    CaptureSettingsSnapshot,
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
)

_CONTRACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "perception"
    / "contracts"
    / "capture-session-ack"
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class CaptureAcknowledgementError(ValueError):
    """A capture acknowledgement cannot establish runtime readiness."""

    def __init__(self, message: str, *, outcome: str | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True)
class ValidatedCaptureSessionAcknowledgement:
    """Detached acknowledgement evidence accepted for one exact request."""

    configuration_revision: int
    request_sha256: str
    session_id: str
    _source_report: dict[str, Any]

    @property
    def source_report(self) -> dict[str, Any]:
        """Return a detached source-report copy for diagnostics."""
        return cast("dict[str, Any]", json.loads(json.dumps(self._source_report)))


@lru_cache(maxsize=2)
def _acknowledgement_validator(schema_version: int) -> Draft202012Validator:
    schema_path = (
        _CONTRACT_ROOT / f"v{schema_version}" / "capture-session-ack.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_capture_session_acknowledgement(
    settings: CaptureSettingsSnapshot,
    acknowledgement: dict[str, Any],
    *,
    expected_session_id: str,
) -> ValidatedCaptureSessionAcknowledgement:
    """Validate a compatible source report against the exact supplied request.

    The request digest covers the exact versioned snapshot's compact JSON bytes.
    An opened-session report must match the announced source session and every
    request field required for localization readiness. Optional control
    readbacks remain visible but may be unknown or unsupported.
    """
    if not isinstance(
        settings,
        (
            CaptureSettingsSnapshotV1,
            CaptureSettingsSnapshotV2,
            CaptureSettingsSnapshotV3,
        ),
    ):
        raise TypeError("settings must be a validated capture settings snapshot")
    expected_schema_version = {
        CaptureSettingsSnapshotV1: 1,
        CaptureSettingsSnapshotV2: 2,
        CaptureSettingsSnapshotV3: 3,
    }[type(settings)]
    if (
        type(settings.schema_version) is not int
        or settings.schema_version != expected_schema_version
    ):
        raise CaptureAcknowledgementError(
            "capture settings schema version does not match its snapshot type"
        )
    if not isinstance(expected_session_id, str) or not _IDENTIFIER_PATTERN.fullmatch(
        expected_session_id
    ):
        raise CaptureAcknowledgementError("expected source-session ID is invalid")

    try:
        encoded_acknowledgement = json.dumps(
            acknowledgement,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        value = json.loads(encoded_acknowledgement)
    except (TypeError, ValueError) as error:
        raise CaptureAcknowledgementError(
            "capture acknowledgement is not finite JSON"
        ) from error
    if not isinstance(value, dict):
        raise CaptureAcknowledgementError("capture acknowledgement must be an object")
    try:
        acknowledgement_schema_version = 1 if settings.schema_version == 1 else 2
        _acknowledgement_validator(acknowledgement_schema_version).validate(value)
    except ValidationError as error:
        raise CaptureAcknowledgementError(
            "capture acknowledgement does not match schema "
            f"v{acknowledgement_schema_version}"
        ) from error

    expected_revision = settings.configuration_revision
    if value["configuration_revision"] != expected_revision:
        raise CaptureAcknowledgementError("capture acknowledgement revision is stale")

    request_sha256 = hashlib.sha256(settings.to_json().encode("utf-8")).hexdigest()
    if value["request_sha256"] != request_sha256:
        raise CaptureAcknowledgementError(
            "capture acknowledgement request digest differs"
        )

    outcome = value["outcome"]
    if outcome != "compatible":
        raise CaptureAcknowledgementError(
            f"capture session outcome is {outcome}", outcome=outcome
        )

    report = value["source_report"]
    if report["session_id"] != expected_session_id:
        raise CaptureAcknowledgementError("capture acknowledgement session ID differs")
    _validate_report_against_request(settings, report)

    return ValidatedCaptureSessionAcknowledgement(
        configuration_revision=expected_revision,
        request_sha256=request_sha256,
        session_id=expected_session_id,
        _source_report=report,
    )


def _validate_report_against_request(
    settings: CaptureSettingsSnapshot, report: dict[str, Any]
) -> None:
    requested = cast("dict[str, Any]", settings.requested)
    if report["device_id"] != requested["device_id"]:
        raise CaptureAcknowledgementError("capture source device differs from request")
    if requested["backend"] != "auto" and report["backend_id"] != requested["backend"]:
        raise CaptureAcknowledgementError("capture source backend differs from request")
    if report["requested_profile"] != requested["requested_profile"]:
        raise CaptureAcknowledgementError("capture requested profile differs")
    if report["negotiated_profile"] != requested["expected_negotiated_profile"]:
        raise CaptureAcknowledgementError("capture negotiated profile differs")
    if report["timestamp_semantics"] != requested["timestamp_semantics"]:
        raise CaptureAcknowledgementError("capture timestamp semantics differ")

    revision = settings.configuration_revision
    geometry = report["applied_frame_geometry"]
    for field, expected in (
        ("configuration_revision", revision),
        ("camera_profile_id", requested["camera_profile_id"]),
        ("image_space", requested["image_space"]),
        ("rectification_revision", requested["rectification_revision"]),
    ):
        if geometry[field] != expected:
            raise CaptureAcknowledgementError(
                f"capture applied geometry differs: {field}"
            )

    if settings.schema_version >= 2:
        expected_output_geometry = requested.get("output_frame_geometry")
        if not isinstance(expected_output_geometry, dict):
            raise CaptureAcknowledgementError(
                "v2 capture request output frame geometry is missing"
            )
        if report.get("output_frame_geometry") != expected_output_geometry:
            raise CaptureAcknowledgementError(
                "capture emitted RGB8 output geometry differs from request"
            )

    capabilities = report["capabilities"]
    if settings.schema_version == 3:
        camera_controls = requested["camera_controls"]
        if camera_controls["mode"] == "source_defaults":
            # The adapter owns these controls. Its actual capability and
            # readback statuses remain visible in the report but do not block
            # localization readiness or imply a requested value.
            requirements = {
                "manual_exposure": False,
                "automatic_exposure": False,
                "gain_control": False,
            }
            requested_controls: dict[str, Any] = {}
        else:
            requirements = camera_controls["requirements"]
            requested_controls = camera_controls["requested"]
    else:
        requirements = requested["control_requirements"]
        requested_controls = requested["requested_controls"]
    for required, field in (
        (requirements["manual_exposure"], "manual_exposure"),
        (requirements["automatic_exposure"], "automatic_exposure"),
        (requirements["gain_control"], "gain_control"),
    ):
        if required and capabilities[field] != "supported":
            raise CaptureAcknowledgementError(
                f"required capture capability is unavailable: {field}"
            )
    if (
        requested["timestamp_semantics"]["kind"] == "source_timestamp"
        and capabilities["source_timestamps"] != "supported"
    ):
        raise CaptureAcknowledgementError("source timestamps are not supported")

    supported_formats = capabilities["supported_pixel_formats"]
    if (
        supported_formats is not None
        and report["negotiated_profile"]["pixel_format"] not in supported_formats
    ):
        raise CaptureAcknowledgementError(
            "negotiated pixel format is absent from reported capabilities"
        )

    controls = report["applied_controls"]
    if requirements["manual_exposure"] or requirements["automatic_exposure"]:
        expected_mode = (
            "manual" if requested_controls["exposure_mode"] == "manual" else "automatic"
        )
        _require_applied_value(controls, "exposure_mode", expected_mode)
    if requirements["manual_exposure"]:
        exposure_value = requested_controls["exposure_value"]
        if exposure_value is not None:
            _require_applied_value(controls, "exposure_value", exposure_value)
    if requirements["gain_control"]:
        _require_applied_value(controls, "gain", requested_controls["gain"])


def _require_applied_value(
    controls: dict[str, Any], field: str, expected: object
) -> None:
    readback = controls[field]
    if readback["status"] != "applied":
        raise CaptureAcknowledgementError(
            f"required camera control readback is unavailable: {field}"
        )
    actual = readback["value"]
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise CaptureAcknowledgementError(
                f"required camera control readback differs: {field}"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or actual != expected
        ):
            raise CaptureAcknowledgementError(
                f"required camera control readback differs: {field}"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise CaptureAcknowledgementError(
            f"required camera control readback differs: {field}"
        )
