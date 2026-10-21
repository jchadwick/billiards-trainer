"""Resolve Python's effective capture configuration into the Rust request.

This module serializes requested settings only. A generated profile identifier
is a stable fingerprint of the raw camera mode and, for v2, emitted RGB8
geometry. It is not a device identity or an application acknowledgment. Rust
must compare the request with the opened source report before reading frames;
settings absent from that report remain unverified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import jsonschema  # type: ignore[import-untyped]
from referencing import Registry, Resource

from backend.config import load_schema_reference

_MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
_MAX_U32 = 4_294_967_295
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_CAPTURE_DEVICE_SELECTOR_PATTERN = re.compile(r"^gstreamer-device-v1:[0-9a-f]{64}$")
_PIXEL_FORMAT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_CALIBRATION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "perception"
    / "contracts"
    / "calibration"
    / "v1"
    / "table-calibration.schema.json"
)
_CAPTURE_SETTINGS_SCHEMA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "perception"
    / "contracts"
    / "capture-settings"
)


class ConfigurationReader(Protocol):
    """Configuration surface needed to resolve effective capture settings."""

    def get_all(self) -> dict[str, Any]: ...


class CaptureConfigurationError(ValueError):
    """The effective camera configuration cannot form a safe Rust request."""


@dataclass(frozen=True)
class _CaptureSettingsSnapshot:
    """Shared immutable representation for versioned capture requests."""

    schema_version: int
    configuration_revision: int
    _requested_json: str

    @property
    def requested(self) -> dict[str, object]:
        """Return a detached view so callers cannot mutate the stored request."""
        value = json.loads(self._requested_json)
        return cast("dict[str, object]", value)

    def to_dict(self) -> dict[str, object]:
        """Return an isolated JSON-compatible representation of this request."""
        value = json.loads(json.dumps(self.as_dict(), allow_nan=False))
        return cast("dict[str, object]", value)

    def as_dict(self) -> dict[str, object]:
        """Return the wire contract without implying source application."""
        return {
            "schema_version": self.schema_version,
            "configuration_revision": self.configuration_revision,
            "requested": self.requested,
        }

    def to_json(self) -> str:
        """Encode deterministic compact JSON for the Rust control boundary."""
        return json.dumps(
            self.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class CaptureSettingsSnapshotV1(_CaptureSettingsSnapshot):
    """Strict v1 request snapshot retained for explicit legacy interop."""


@dataclass(frozen=True)
class CaptureSettingsSnapshotV2(_CaptureSettingsSnapshot):
    """Strict request snapshot with distinct raw and emitted-frame geometry."""


@dataclass(frozen=True)
class CaptureSettingsSnapshotV3(_CaptureSettingsSnapshot):
    """Request snapshot with explicit source-default or strict control policy."""


CaptureSettingsSnapshot = (
    CaptureSettingsSnapshotV1 | CaptureSettingsSnapshotV2 | CaptureSettingsSnapshotV3
)


def resolve_capture_settings(
    configuration: ConfigurationReader,
    *,
    calibration_bundle: dict[str, object] | None = None,
    schema_version: int = 3,
) -> CaptureSettingsSnapshot:
    """Resolve supported live-camera values and preserve them as a request.

    Camera mode, emitted RGB8 geometry, image-space interpretation, freshness,
    and memory limits are all explicit. Video-file paths are rejected because
    recorded replay has its own provider and must not masquerade as live input.
    The optional calibration must already have been exported and validated
    against this exact requested profile by ``perception_export``.
    """
    defaults = load_schema_reference()
    perception = _object_at(defaults, "perception")
    capture_defaults = _object_at(perception, "capture")
    camera_defaults = _object_at(_object_at(defaults, "vision"), "camera")
    metadata_defaults = _object_at(defaults, "metadata")
    configured_values = configuration.get_all()

    if type(schema_version) is not int or schema_version not in {1, 2, 3}:
        raise CaptureConfigurationError(
            "capture settings schema_version must be 1, 2, or 3"
        )

    def value(path: str, fallback: object) -> object:
        current: object = configured_values
        for component in path.split("."):
            if not isinstance(current, dict) or component not in current:
                return fallback
            current = current[component]
        return current

    raw_video_file_path = value(
        "vision.camera.video_file_path", camera_defaults.get("video_file_path")
    )
    if raw_video_file_path is not None and not isinstance(raw_video_file_path, str):
        raise CaptureConfigurationError(
            "vision.camera.video_file_path must be a string or null"
        )
    if raw_video_file_path:
        raise CaptureConfigurationError(
            "live Rust capture cannot use vision.camera.video_file_path because the "
            "Python camera path gives video files precedence over device_id; clear "
            "the video path explicitly or select recorded replay"
        )

    configuration_revision = _safe_integer(
        value(
            "metadata.configuration_revision",
            metadata_defaults.get("configuration_revision"),
        ),
        "metadata.configuration_revision",
        minimum=1,
    )
    raw_device_id = value("vision.camera.device_id", camera_defaults.get("device_id"))
    raw_device_selector = value(
        "perception.capture.device_selector", capture_defaults.get("device_selector")
    )
    device_selector = _capture_device_selector(raw_device_selector)
    if device_selector is not None and schema_version != 3:
        raise CaptureConfigurationError(
            "perception.capture.device_selector requires capture-settings schema v3"
        )
    # The selector is Rust-specific and is carried in the same request field
    # used by the exact physical helper. Empty configuration preserves the
    # legacy numeric-index request, which remains the Python camera default.
    device_id = device_selector or _device_id(raw_device_id)
    backend = _identifier(
        value("vision.camera.backend", camera_defaults.get("backend")),
        "vision.camera.backend",
    )
    if backend not in {"auto", "v4l2", "dshow", "gstreamer", "opencv", "kinect2"}:
        raise CaptureConfigurationError("vision.camera.backend is unsupported")
    if device_selector is not None and backend not in {"auto", "gstreamer"}:
        raise CaptureConfigurationError(
            "perception.capture.device_selector requires the auto or gstreamer backend"
        )

    resolution = value("vision.camera.resolution", camera_defaults.get("resolution"))
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(part, bool) or not isinstance(part, int) for part in resolution
        )
    ):
        raise CaptureConfigurationError(
            "vision.camera.resolution must contain two positive integers"
        )
    width_px = _safe_integer(
        resolution[0], "vision.camera.resolution[0]", minimum=1, maximum=_MAX_U32
    )
    height_px = _safe_integer(
        resolution[1], "vision.camera.resolution[1]", minimum=1, maximum=_MAX_U32
    )
    if schema_version >= 2:
        output_resolution = value(
            "perception.capture.output_resolution",
            capture_defaults.get("output_resolution"),
        )
        if output_resolution is None:
            output_resolution = [width_px, height_px]
        if (
            not isinstance(output_resolution, list)
            or len(output_resolution) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int)
                for part in output_resolution
            )
        ):
            raise CaptureConfigurationError(
                "perception.capture.output_resolution must be null or contain two positive integers"
            )
        output_width_px = _safe_integer(
            output_resolution[0],
            "perception.capture.output_resolution[0]",
            minimum=1,
            maximum=_MAX_U32,
        )
        output_height_px = _safe_integer(
            output_resolution[1],
            "perception.capture.output_resolution[1]",
            minimum=1,
            maximum=_MAX_U32,
        )
    else:
        # V1 has no emitted-frame geometry field. Do not let a newer config
        # value change its historical bytes, profile identity, or semantics.
        output_width_px, output_height_px = width_px, height_px
    fps = _safe_integer(
        value("vision.camera.fps", camera_defaults.get("fps")),
        "vision.camera.fps",
        minimum=1,
        maximum=120,
    )
    pixel_format_value = value(
        "perception.capture.pixel_format", capture_defaults.get("pixel_format")
    )
    if not isinstance(pixel_format_value, str) or not _PIXEL_FORMAT_PATTERN.fullmatch(
        pixel_format_value
    ):
        raise CaptureConfigurationError(
            "perception.capture.pixel_format must be a bounded adapter-neutral identifier"
        )
    pixel_format = pixel_format_value
    image_space = value(
        "perception.capture.image_space", capture_defaults.get("image_space")
    )
    if not isinstance(image_space, str) or image_space not in {"raw", "rectified"}:
        raise CaptureConfigurationError(
            "perception.capture.image_space must be 'raw' or 'rectified'"
        )
    rectification_revision = _safe_integer(
        value(
            "perception.capture.rectification_revision",
            capture_defaults.get("rectification_revision"),
        ),
        "perception.capture.rectification_revision",
    )
    calibration_revision = _safe_integer(
        value(
            "perception.capture.calibration_revision",
            capture_defaults.get("calibration_revision"),
        ),
        "perception.capture.calibration_revision",
    )
    timestamp_semantics = value(
        "perception.capture.timestamp_semantics",
        capture_defaults.get("timestamp_semantics"),
    )
    if (
        not isinstance(timestamp_semantics, str)
        or timestamp_semantics != "read_completion_only"
    ):
        raise CaptureConfigurationError(
            "perception.capture.timestamp_semantics must be read_completion_only "
            "until a source clock is explicitly qualified"
        )

    controls_policy = "require"
    if schema_version == 3:
        configured_controls_policy = value(
            "vision.camera.control_policy", camera_defaults.get("control_policy")
        )
        if not isinstance(
            configured_controls_policy, str
        ) or configured_controls_policy not in {
            "source_defaults",
            "require",
        }:
            raise CaptureConfigurationError(
                "vision.camera.control_policy must be 'source_defaults' or 'require'"
            )
        controls_policy = configured_controls_policy

    requested_controls: dict[str, object] | None = None
    control_requirements: dict[str, bool] | None = None
    if schema_version < 3 or controls_policy == "require":
        exposure_mode = value(
            "vision.camera.exposure_mode", camera_defaults.get("exposure_mode")
        )
        if not isinstance(exposure_mode, str) or exposure_mode not in {
            "auto",
            "manual",
        }:
            raise CaptureConfigurationError(
                "vision.camera.exposure_mode must be 'auto' or 'manual'"
            )
        exposure_value = value(
            "vision.camera.exposure_value", camera_defaults.get("exposure_value")
        )
        if exposure_value is not None:
            exposure_value = _finite_number(
                exposure_value, "vision.camera.exposure_value"
            )
        if exposure_mode == "manual" and exposure_value is None:
            raise CaptureConfigurationError(
                "manual exposure requires vision.camera.exposure_value"
            )
        gain = _finite_number(
            value("vision.camera.gain", camera_defaults.get("gain")),
            "vision.camera.gain",
        )
        if gain < 0:
            raise CaptureConfigurationError("vision.camera.gain must be nonnegative")
        brightness = _bounded_control(
            value("vision.camera.brightness", camera_defaults.get("brightness")),
            "vision.camera.brightness",
        )
        contrast = _bounded_control(
            value("vision.camera.contrast", camera_defaults.get("contrast")),
            "vision.camera.contrast",
        )
        saturation = _bounded_control(
            value("vision.camera.saturation", camera_defaults.get("saturation")),
            "vision.camera.saturation",
        )
        auto_focus = value(
            "vision.camera.auto_focus", camera_defaults.get("auto_focus")
        )
        if not isinstance(auto_focus, bool):
            raise CaptureConfigurationError(
                "vision.camera.auto_focus must be a boolean"
            )
        buffer_size = _safe_integer(
            value("vision.camera.buffer_size", camera_defaults.get("buffer_size")),
            "vision.camera.buffer_size",
            minimum=1,
            maximum=_MAX_U32,
        )
        requested_controls = {
            "exposure_mode": exposure_mode,
            "exposure_value": exposure_value,
            "gain": gain,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "auto_focus": auto_focus,
            "buffer_size": buffer_size,
        }
        control_requirements = {
            "manual_exposure": exposure_mode == "manual",
            "automatic_exposure": exposure_mode == "auto",
            "gain_control": True,
        }
    auto_reconnect = value(
        "vision.camera.auto_reconnect", camera_defaults.get("auto_reconnect")
    )
    if not isinstance(auto_reconnect, bool):
        raise CaptureConfigurationError(
            "vision.camera.auto_reconnect must be a boolean"
        )
    reconnect_delay_seconds = _finite_number(
        value("vision.camera.reconnect_delay", camera_defaults.get("reconnect_delay")),
        "vision.camera.reconnect_delay",
    )
    if reconnect_delay_seconds <= 0:
        raise CaptureConfigurationError(
            "vision.camera.reconnect_delay must be positive"
        )
    max_reconnect_attempts = _safe_integer(
        value(
            "vision.camera.max_reconnect_attempts",
            camera_defaults.get("max_reconnect_attempts"),
        ),
        "vision.camera.max_reconnect_attempts",
        maximum=_MAX_U32,
    )
    stop_timeout_seconds = _finite_number(
        value("vision.camera.stop_timeout", camera_defaults.get("stop_timeout")),
        "vision.camera.stop_timeout",
    )
    if stop_timeout_seconds <= 0:
        raise CaptureConfigurationError("vision.camera.stop_timeout must be positive")

    maximum_frame_age_ms = _safe_integer(
        value(
            "perception.capture.maximum_frame_age_ms",
            capture_defaults.get("maximum_frame_age_ms"),
        ),
        "perception.capture.maximum_frame_age_ms",
        minimum=1,
    )
    maximum_rgb_frame_bytes = _safe_integer(
        value(
            "perception.capture.maximum_rgb_frame_bytes",
            capture_defaults.get("maximum_rgb_frame_bytes"),
        ),
        "perception.capture.maximum_rgb_frame_bytes",
        minimum=1,
        maximum=1_073_741_824,
    )
    maximum_aggregate_rgb_frame_bytes = _safe_integer(
        value(
            "perception.capture.maximum_aggregate_rgb_frame_bytes",
            capture_defaults.get("maximum_aggregate_rgb_frame_bytes"),
        ),
        "perception.capture.maximum_aggregate_rgb_frame_bytes",
        minimum=1,
        maximum=1_073_741_824,
    )
    if schema_version >= 2:
        # Python integers are unbounded, so this exact multiplication cannot
        # wrap even at the u32 width/height limits. Compare it before forming
        # the snapshot or allowing a caller to spawn the Rust process.
        minimum_rgb8_frame_bytes = output_width_px * output_height_px * 3
        if minimum_rgb8_frame_bytes > maximum_aggregate_rgb_frame_bytes:
            raise CaptureConfigurationError(
                "emitted RGB8 frame exceeds maximum_aggregate_rgb_frame_bytes"
            )
        if minimum_rgb8_frame_bytes > maximum_rgb_frame_bytes:
            raise CaptureConfigurationError(
                "emitted RGB8 frame exceeds maximum_rgb_frame_bytes"
            )
    if maximum_aggregate_rgb_frame_bytes < maximum_rgb_frame_bytes:
        raise CaptureConfigurationError(
            "aggregate RGB frame limit must be at least the per-frame limit"
        )

    profile = {
        "width_px": width_px,
        "height_px": height_px,
        "frame_rate": {
            "frames_numerator": fps,
            "seconds_denominator": 1,
        },
        "pixel_format": pixel_format,
    }
    reconnect_policy = {
        "enabled": auto_reconnect,
        "delay_seconds": reconnect_delay_seconds,
        "maximum_attempts": max_reconnect_attempts,
        "stop_timeout_seconds": stop_timeout_seconds,
    }
    profile_identity_source = {
        "device_id": device_id,
        "backend": backend,
        "profile": profile,
        "image_space": image_space,
        "rectification_revision": rectification_revision,
    }
    if schema_version >= 2:
        profile_identity_source["output_frame_geometry"] = {
            "width_px": output_width_px,
            "height_px": output_height_px,
        }
        profile_identity_source["output_pixel_format"] = "RGB8"
    profile_hash = hashlib.sha256(
        json.dumps(
            profile_identity_source,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    camera_profile_id = f"camera-profile-v{schema_version}:{profile_hash}"

    if calibration_bundle is not None:
        _validate_calibration_bundle(
            calibration_bundle,
            calibration_revision=calibration_revision,
            camera_profile_id=camera_profile_id,
            rectification_revision=rectification_revision,
            image_space=image_space,
            schema_version=schema_version,
            width_px=(output_width_px if schema_version >= 2 else width_px),
            height_px=(output_height_px if schema_version >= 2 else height_px),
        )
        calibration: dict[str, object] | None = json.loads(
            json.dumps(calibration_bundle, allow_nan=False)
        )
    else:
        calibration = None

    requested: dict[str, object] = {
        "device_id": device_id,
        "backend": backend,
        "requested_profile": profile,
        "expected_negotiated_profile": profile,
        "camera_profile_id": camera_profile_id,
        "image_space": image_space,
        "rectification_revision": rectification_revision,
        "calibration_revision": calibration_revision,
        "calibration": calibration,
        "timestamp_semantics": {"kind": "read_completion_only"},
        "reconnect_policy": reconnect_policy,
        "maximum_frame_age_ms": maximum_frame_age_ms,
        "maximum_rgb_frame_bytes": maximum_rgb_frame_bytes,
        "maximum_aggregate_rgb_frame_bytes": maximum_aggregate_rgb_frame_bytes,
    }
    if schema_version < 3:
        requested["control_requirements"] = control_requirements
        requested["requested_controls"] = requested_controls
    else:
        camera_controls: dict[str, object] = {"mode": controls_policy}
        if controls_policy == "require":
            camera_controls["requirements"] = control_requirements
            camera_controls["requested"] = requested_controls
        requested["camera_controls"] = camera_controls
    if schema_version >= 2:
        requested["output_frame_geometry"] = {
            "width_px": output_width_px,
            "height_px": output_height_px,
        }
    encoded_requested = json.dumps(
        requested,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if schema_version == 1:
        snapshot: CaptureSettingsSnapshot = CaptureSettingsSnapshotV1(
            schema_version=schema_version,
            configuration_revision=configuration_revision,
            _requested_json=encoded_requested,
        )
    elif schema_version == 2:
        snapshot = CaptureSettingsSnapshotV2(
            schema_version=schema_version,
            configuration_revision=configuration_revision,
            _requested_json=encoded_requested,
        )
    else:
        snapshot = CaptureSettingsSnapshotV3(
            schema_version=schema_version,
            configuration_revision=configuration_revision,
            _requested_json=encoded_requested,
        )
    try:
        _capture_settings_schema_validator(schema_version).validate(snapshot.as_dict())
    except jsonschema.ValidationError as exc:
        raise CaptureConfigurationError(
            f"capture settings violate schema v{schema_version}: {exc.message}"
        ) from exc
    return snapshot


def _object_at(value: object, key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get(key), dict):
        raise CaptureConfigurationError(f"bundled defaults must define object {key!r}")
    return cast("dict[str, Any]", value[key])


def _device_id(value: object) -> str:
    if isinstance(value, bool):
        raise CaptureConfigurationError("vision.camera.device_id must be an index")
    if isinstance(value, int) and value >= 0:
        return f"device-index:{value}"
    if isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        raise CaptureConfigurationError(
            "Rust live capture does not yet support an HTTP MJPEG source; "
            "use the Python capture provider for vision.camera.device_id URIs "
            "until the supervised Rust stream adapter is implemented"
        )
    raise CaptureConfigurationError(
        "Rust live capture requires a nonnegative integer camera device_id; "
        "HTTP MJPEG URIs belong to the Python capture path until the Rust "
        "stream adapter is implemented, and video-file paths belong to recorded replay"
    )


def _capture_device_selector(value: object) -> str | None:
    """Return an exact discovery selector, or ``None`` for numeric camera use."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CaptureConfigurationError(
            "perception.capture.device_selector must be null or a stable GStreamer selector"
        )
    if not _CAPTURE_DEVICE_SELECTOR_PATTERN.fullmatch(value):
        raise CaptureConfigurationError(
            "perception.capture.device_selector must match gstreamer-device-v1:<64 lowercase hexadecimal characters>"
        )
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise CaptureConfigurationError(f"{field} must be a bounded ASCII identifier")
    return value


def _safe_integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_SAFE_JSON_INTEGER,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CaptureConfigurationError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureConfigurationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CaptureConfigurationError(f"{field} must be a finite number")
    return result


def _bounded_control(value: object, field: str) -> float:
    """Validate camera image controls against the supported normalized range."""
    result = _finite_number(value, field)
    if not 0 <= result <= 1:
        raise CaptureConfigurationError(f"{field} must be in [0, 1]")
    return result


def _validate_calibration_bundle(
    value: dict[str, object],
    *,
    calibration_revision: int,
    camera_profile_id: str,
    rectification_revision: int,
    image_space: object,
    schema_version: int,
    width_px: int,
    height_px: int,
) -> None:
    if not isinstance(value, dict):
        raise CaptureConfigurationError("calibration bundle must be an object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CaptureConfigurationError(
            "calibration bundle must contain finite JSON values"
        ) from exc
    try:
        _calibration_schema_validator().validate(value)
    except jsonschema.ValidationError as exc:
        raise CaptureConfigurationError(
            f"calibration bundle violates table-calibration v1: {exc.message}"
        ) from exc
    if _safe_integer(value.get("schema_version"), "calibration.schema_version") != 1:
        raise CaptureConfigurationError("calibration bundle schema_version must be 1")
    if (
        _safe_integer(
            value.get("calibration_revision"), "calibration.calibration_revision"
        )
        != calibration_revision
    ):
        raise CaptureConfigurationError("calibration revision does not match request")
    if value.get("camera_profile_id") != camera_profile_id:
        raise CaptureConfigurationError(
            "calibration camera profile does not match request"
        )
    if (
        _safe_integer(
            value.get("rectification_revision"),
            "calibration.rectification_revision",
        )
        != rectification_revision
    ):
        raise CaptureConfigurationError(
            "calibration rectification revision does not match request"
        )
    if value.get("image_space") != image_space:
        raise CaptureConfigurationError(
            "calibration image space does not match request"
        )
    source_dimensions = (
        _safe_integer(
            value.get("source_width_px"),
            "calibration.source_width_px",
            minimum=1,
            maximum=_MAX_U32,
        ),
        _safe_integer(
            value.get("source_height_px"),
            "calibration.source_height_px",
            minimum=1,
            maximum=_MAX_U32,
        ),
    )
    if (schema_version >= 2 or image_space == "raw") and source_dimensions != (
        width_px,
        height_px,
    ):
        raise CaptureConfigurationError("calibration dimensions do not match request")
    if not isinstance(value.get("pixel_to_table_mm"), list):
        raise CaptureConfigurationError("calibration homography is missing")


@lru_cache(maxsize=1)
def _calibration_schema_validator() -> jsonschema.Draft202012Validator:
    """Load the strict cross-language calibration schema once per process."""
    try:
        schema = json.loads(_CALIBRATION_SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        return jsonschema.Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        raise CaptureConfigurationError(
            "bundled table-calibration v1 schema cannot be loaded"
        ) from exc


@lru_cache(maxsize=3)
def _capture_settings_schema_validator(
    schema_version: int,
) -> jsonschema.Draft202012Validator:
    """Load the matching closed request schema and calibration reference."""
    schema_path = (
        _CAPTURE_SETTINGS_SCHEMA_ROOT
        / f"v{schema_version}"
        / "capture-settings.schema.json"
    )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        calibration_schema = json.loads(
            _CALIBRATION_SCHEMA_PATH.read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator.check_schema(calibration_schema)
        registry = Registry().with_resource(
            calibration_schema["$id"], Resource.from_contents(calibration_schema)
        )
        return jsonschema.Draft202012Validator(schema, registry=registry)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        jsonschema.SchemaError,
    ) as exc:
        raise CaptureConfigurationError(
            f"bundled capture settings v{schema_version} schema cannot be loaded"
        ) from exc
