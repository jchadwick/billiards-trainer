"""Supervise the unqualified Rust live-capture process boundary.

This module deliberately does not make Rust live selectable by the API. It
proves the Python process contract only: one immutable capture request is
written to child stdin, the same request configures v3 scene ingress, and
readiness waits for a usable localization to reach the retained publisher.
The camera adapter and physical platform qualification remain separate gates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.perception.capture_configuration import (
    CaptureSettingsSnapshot,
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
)
from backend.perception.scene_ingress import SceneIngressConfig, SceneSnapshotIngress
from backend.perception.scene_state_worker import (
    SceneSnapshotStateWorker,
    SceneStateWorkerStats,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from backend.perception.capture_acknowledgement import (
        ValidatedCaptureSessionAcknowledgement,
    )
    from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

_RUST_CHILD_ENVIRONMENT_KEYS = ("TMPDIR",)
_OPENVINO_CHILD_ENVIRONMENT_KEYS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
_MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
_MAX_U64 = (1 << 64) - 1
_MAX_FRAME_RECORD_HEADER_BYTES = (1 << 32) - 1
_MAX_LIVE_AUDIT_RECORD_BYTES = 1_048_576
_MIN_LIVE_AUDIT_RECORD_BYTES = 1_024
_MAX_LIVE_AUDIT_QUEUE_CAPACITY = 4_096
_LIVE_AUDIT_SCHEMA_VERSION = 2
_GSTREAMER_CAPTURE_SELECTOR_PREFIX = "gstreamer-device-v1:"
_GSTREAMER_CAPTURE_SELECTOR_PATTERN = re.compile(r"^gstreamer-device-v1:[0-9a-f]{64}$")


@dataclass(frozen=True)
class _CaptureRuntimeIdentity:
    """Tracking identity taken only from the capture request snapshot."""

    configuration_revision: int
    camera_profile_id: str
    rectification_revision: int
    image_space: str


def _capture_runtime_identity(
    settings: CaptureSettingsSnapshot,
) -> _CaptureRuntimeIdentity:
    """Validate and extract tracking identity from the exact request object."""
    expected_version = {
        CaptureSettingsSnapshotV1: 1,
        CaptureSettingsSnapshotV2: 2,
        CaptureSettingsSnapshotV3: 3,
    }.get(type(settings))
    if expected_version is None:
        raise RustLiveConfigurationError(
            "capture settings must be a validated v1, v2, or v3 request snapshot"
        )
    if (
        type(settings.schema_version) is not int
        or settings.schema_version != expected_version
    ):
        raise RustLiveConfigurationError(
            "capture settings schema version does not match its snapshot type"
        )
    try:
        request = settings.to_dict()
    except (TypeError, ValueError) as error:
        raise RustLiveConfigurationError(
            "capture settings must contain finite JSON values"
        ) from error
    if (
        not isinstance(request.get("schema_version"), int)
        or isinstance(request.get("schema_version"), bool)
        or request["schema_version"] not in {1, 2, 3}
        or request["schema_version"] != settings.schema_version
    ):
        raise RustLiveConfigurationError(
            "capture settings schema version must match its v1, v2, or v3 snapshot"
        )
    revision = request.get("configuration_revision")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not 1 <= revision <= _MAX_SAFE_JSON_INTEGER
        or revision != settings.configuration_revision
    ):
        raise RustLiveConfigurationError(
            "capture configuration revision must be a positive safe JSON integer"
        )
    requested = request.get("requested")
    if not isinstance(requested, dict):
        raise RustLiveConfigurationError("capture request fields are missing")
    camera_profile_id = requested.get("camera_profile_id")
    rectification_revision = requested.get("rectification_revision")
    image_space = requested.get("image_space")
    calibration_revision = requested.get("calibration_revision")
    if not isinstance(camera_profile_id, str) or not camera_profile_id.strip():
        raise RustLiveConfigurationError(
            "capture camera_profile_id must be a non-empty string"
        )
    if (
        not isinstance(rectification_revision, int)
        or isinstance(rectification_revision, bool)
        or not 0 <= rectification_revision <= _MAX_SAFE_JSON_INTEGER
    ):
        raise RustLiveConfigurationError(
            "capture rectification_revision must be a non-negative safe integer"
        )
    if not isinstance(image_space, str) or image_space not in {"raw", "rectified"}:
        raise RustLiveConfigurationError("capture image_space is unsupported")
    if (
        not isinstance(calibration_revision, int)
        or isinstance(calibration_revision, bool)
        or not 0 <= calibration_revision <= _MAX_SAFE_JSON_INTEGER
    ):
        raise RustLiveConfigurationError(
            "capture calibration_revision must be a non-negative safe integer"
        )
    calibration = requested.get("calibration")
    output_geometry = requested.get("output_frame_geometry")
    if settings.schema_version >= 2:
        if not isinstance(output_geometry, dict):
            raise RustLiveConfigurationError(
                "v2 and v3 capture settings must specify output_frame_geometry"
            )
        for dimension in ("width_px", "height_px"):
            dimension_value = output_geometry.get(dimension)
            if (
                not isinstance(dimension_value, int)
                or isinstance(dimension_value, bool)
                or not 1 <= dimension_value <= 4_294_967_295
            ):
                raise RustLiveConfigurationError(
                    f"capture output_frame_geometry.{dimension} must be a positive u32"
                )
    elif output_geometry is not None:
        raise RustLiveConfigurationError(
            "v1 capture settings cannot specify output_frame_geometry"
        )
    if calibration is not None:
        profile = requested.get("requested_profile")
        if not isinstance(calibration, dict) or not isinstance(profile, dict):
            raise RustLiveConfigurationError(
                "capture calibration must remain embedded in a valid request"
            )
        if settings.schema_version >= 2:
            # Re-narrow the value in this block for static checking and keep
            # the calibration comparison adjacent to its expected dimensions.
            if not isinstance(output_geometry, dict):
                raise RustLiveConfigurationError(
                    "v2 capture calibration requires output frame geometry"
                )
            expected_calibration_width = output_geometry.get("width_px")
            expected_calibration_height = output_geometry.get("height_px")
        else:
            expected_calibration_width = profile.get("width_px")
            expected_calibration_height = profile.get("height_px")
        calibration_mismatches = [
            calibration.get("calibration_revision") != calibration_revision,
            calibration.get("camera_profile_id") != camera_profile_id,
            calibration.get("rectification_revision") != rectification_revision,
            calibration.get("image_space") != image_space,
        ]
        # V1 rectified calibration has historically recorded the dimensions
        # of the rectified image, which can differ from the negotiated camera
        # mode. Keep that legacy rule here; v2 makes emitted geometry explicit
        # and therefore requires an exact dimension match in either image
        # space.
        if settings.schema_version >= 2 or image_space == "raw":
            calibration_mismatches.extend(
                (
                    calibration.get("source_width_px") != expected_calibration_width,
                    calibration.get("source_height_px") != expected_calibration_height,
                )
            )
        if any(calibration_mismatches):
            raise RustLiveConfigurationError(
                "capture calibration differs from its request profile or revisions"
            )
    elif "calibration" not in requested:
        raise RustLiveConfigurationError(
            "capture request must explicitly include its calibration field"
        )
    return _CaptureRuntimeIdentity(
        configuration_revision=revision,
        camera_profile_id=camera_profile_id,
        rectification_revision=rectification_revision,
        image_space=image_space,
    )


def _seconds_to_milliseconds(name: str, seconds: float) -> int:
    """Convert a validated process timeout to the child's integer CLI unit."""
    try:
        milliseconds = math.ceil(seconds * 1000)
    except (OverflowError, ValueError) as error:
        raise RustLiveConfigurationError(
            f"{name} cannot be represented as a positive u64 millisecond timeout"
        ) from error
    if not 1 <= milliseconds <= _MAX_U64:
        raise RustLiveConfigurationError(
            f"{name} cannot be represented as a positive u64 millisecond timeout"
        )
    return milliseconds


def _sha256_file(path: Path) -> str:
    """Hash an artifact incrementally so model verification stays memory-bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


class RustLiveConfigurationError(ValueError):
    """A Rust live process cannot be launched with the supplied settings."""


class RustLiveStartupError(RuntimeError):
    """The child did not publish a usable scene before the readiness deadline."""


@dataclass(frozen=True)
class RustLiveInferenceSettings:
    """Explicit detector and tracking values for one Rust child process.

    The model export must remain tied to its source checkpoint by the sibling
    manifest. Camera identity, calibration and capture revisions intentionally
    do not appear here; they are owned by the exact capture request instead.
    """

    onnx_path: Path
    source_checkpoint_path: Path
    provider: str
    image_size: int
    confidence: float
    iou: float
    max_detections: int
    intra_op_threads: int | None
    warmup_runs: int
    ball_class_id: int
    cue_class_id: int | None
    association_gate_px: float
    association_prediction_horizon_ms: int
    lost_after_ms: int
    maximum_track_age_ms: int
    maximum_active_tracks: int
    recent_observation_capacity: int

    def validate(self) -> None:
        """Validate paths, model lineage, class IDs and runtime numeric bounds."""
        for name, path in (
            ("onnx_path", self.onnx_path),
            ("source_checkpoint_path", self.source_checkpoint_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise RustLiveConfigurationError(f"{name} must be an absolute Path")
            if not path.is_file():
                raise RustLiveConfigurationError(f"configured {name} does not exist")
        if not isinstance(self.provider, str) or self.provider not in {
            "cpu",
            "auto",
            "openvino",
        }:
            raise RustLiveConfigurationError(
                "provider must be 'cpu', 'auto', or 'openvino'"
            )
        if (
            not isinstance(self.image_size, int)
            or isinstance(self.image_size, bool)
            or self.image_size <= 0
            or self.image_size > 0xFFFFFFFF
            or self.image_size % 32 != 0
        ):
            raise RustLiveConfigurationError(
                "image_size must be a positive u32 multiple of 32"
            )
        for name, number in (("confidence", self.confidence), ("iou", self.iou)):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not 0 <= number <= 1
            ):
                raise RustLiveConfigurationError(
                    f"{name} must be finite and between zero and one"
                )
        for name, number, minimum, maximum in (
            ("warmup_runs", self.warmup_runs, 0, sys.maxsize),
            ("ball_class_id", self.ball_class_id, 0, 0xFFFFFFFF),
            ("max_detections", self.max_detections, 1, 0xFFFFFFFF),
            (
                "association_prediction_horizon_ms",
                self.association_prediction_horizon_ms,
                1,
                0xFFFFFFFFFFFFFFFF,
            ),
            ("lost_after_ms", self.lost_after_ms, 1, 0xFFFFFFFFFFFFFFFF),
            ("maximum_track_age_ms", self.maximum_track_age_ms, 1, 0xFFFFFFFFFFFFFFFF),
            ("maximum_active_tracks", self.maximum_active_tracks, 1, sys.maxsize),
            (
                "recent_observation_capacity",
                self.recent_observation_capacity,
                1,
                sys.maxsize,
            ),
        ):
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or not minimum <= number <= maximum
            ):
                raise RustLiveConfigurationError(
                    f"{name} must be an integer in [{minimum}, {maximum}]"
                )
        if self.intra_op_threads is not None and (
            not isinstance(self.intra_op_threads, int)
            or isinstance(self.intra_op_threads, bool)
            or not 1 <= self.intra_op_threads <= sys.maxsize
        ):
            raise RustLiveConfigurationError(
                "intra_op_threads must be a positive process-sized integer or None"
            )
        if self.cue_class_id is not None and (
            not isinstance(self.cue_class_id, int)
            or isinstance(self.cue_class_id, bool)
            or not 0 <= self.cue_class_id <= 0xFFFFFFFF
        ):
            raise RustLiveConfigurationError("cue_class_id must be a u32 or None")
        if self.cue_class_id == self.ball_class_id:
            raise RustLiveConfigurationError("ball and cue class IDs must differ")
        if self.maximum_track_age_ms <= self.lost_after_ms:
            raise RustLiveConfigurationError(
                "maximum_track_age_ms must be greater than lost_after_ms"
            )
        if (
            not isinstance(self.association_gate_px, (int, float))
            or isinstance(self.association_gate_px, bool)
            or not math.isfinite(self.association_gate_px)
            or self.association_gate_px <= 0
        ):
            raise RustLiveConfigurationError(
                "association_gate_px must be finite and positive"
            )
        try:
            unmatched_cost = (
                self.association_gate_px * (self.maximum_active_tracks + 1) + 1
            )
        except OverflowError as error:
            raise RustLiveConfigurationError(
                "association gate and track capacity exceed the runtime range"
            ) from error
        if not math.isfinite(unmatched_cost):
            raise RustLiveConfigurationError(
                "association gate and track capacity exceed the runtime range"
            )

        self._validate_model_manifest()

    def _validate_model_manifest(self) -> None:
        """Check both artifact digests and the two-class ONNX contract."""
        manifest_path = Path(f"{self.onnx_path}.manifest.json")
        if not manifest_path.is_file():
            raise RustLiveConfigurationError(
                "the ONNX export manifest is required beside the model"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise TypeError("manifest root is not an object")
            source_model = manifest["source_model"]
            exported_model = manifest["exported_model"]
            model = manifest["model"]
            export = manifest["export"]
            if not all(
                isinstance(section, dict)
                for section in (source_model, exported_model, model, export)
            ):
                raise TypeError("manifest sections are not objects")
            class_names = model["class_names"]
            class_count = model["class_count"]
            input_size = export["input_size"]
            source_digest = source_model["sha256"]
            exported_digest = exported_model["sha256"]
            exported_filename = exported_model["path"]
            exported_format = exported_model["format"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RustLiveConfigurationError(
                "the ONNX export manifest is missing required model lineage fields"
            ) from error

        if manifest.get("manifest_schema_version") != 1:
            raise RustLiveConfigurationError(
                "the ONNX export manifest version is unsupported"
            )
        if exported_format != "onnx" or not isinstance(exported_filename, str):
            raise RustLiveConfigurationError(
                "the export manifest must describe an ONNX artifact"
            )
        if Path(exported_filename).name != self.onnx_path.name:
            raise RustLiveConfigurationError(
                "the export manifest artifact filename differs from the ONNX path"
            )
        if not isinstance(class_names, (list, dict)) or not class_names:
            raise RustLiveConfigurationError(
                "the export manifest class map must be a non-empty list or object"
            )
        if (
            not isinstance(class_count, int)
            or isinstance(class_count, bool)
            or class_count != len(class_names)
        ):
            raise RustLiveConfigurationError(
                "the export manifest class count differs from its class map"
            )
        if not isinstance(input_size, list) or input_size != [
            self.image_size,
            self.image_size,
        ]:
            raise RustLiveConfigurationError(
                "image_size must match the fixed square size in the export manifest"
            )
        class_ids: set[int] = set()
        entries: list[Any] = []
        if isinstance(class_names, list):
            entries = class_names
        else:
            for raw_class_id, mapped_class_name in class_names.items():
                try:
                    mapped_class_id = int(raw_class_id)
                except (TypeError, ValueError) as error:
                    raise RustLiveConfigurationError(
                        "the export manifest class map is invalid"
                    ) from error
                if isinstance(raw_class_id, bool) or str(mapped_class_id) != str(
                    raw_class_id
                ):
                    raise RustLiveConfigurationError(
                        "the export manifest class map is invalid"
                    )
                entries.append({"id": mapped_class_id, "name": mapped_class_name})
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                class_id_value = entry.get("id")
                class_name_value = entry.get("name")
            elif isinstance(entry, str) and isinstance(class_names, list):
                class_id_value = index
                class_name_value = entry
            else:
                raise RustLiveConfigurationError(
                    "the export manifest class map is invalid"
                )
            if (
                not isinstance(class_id_value, int)
                or isinstance(class_id_value, bool)
                or class_id_value < 0
                or not isinstance(class_name_value, str)
                or not class_name_value.strip()
                or class_id_value in class_ids
            ):
                raise RustLiveConfigurationError(
                    "the export manifest class map is invalid"
                )
            class_ids.add(class_id_value)
        selected_ids = [self.ball_class_id]
        if self.cue_class_id is not None:
            selected_ids.append(self.cue_class_id)
        if any(class_id not in class_ids for class_id in selected_ids):
            raise RustLiveConfigurationError(
                "configured ball/cue class IDs must exist in the ONNX manifest"
            )

        try:
            actual_source_digest = _sha256_file(self.source_checkpoint_path)
            actual_export_digest = _sha256_file(self.onnx_path)
        except OSError as error:
            raise RustLiveConfigurationError(
                "model artifacts could not be read for lineage verification"
            ) from error
        if source_digest != actual_source_digest:
            raise RustLiveConfigurationError(
                "source checkpoint hash does not match the ONNX export manifest"
            )
        if exported_digest != actual_export_digest:
            raise RustLiveConfigurationError(
                "ONNX artifact hash does not match its export manifest"
            )


@dataclass(frozen=True)
class RustLiveSettings:
    """All process and transport settings for one immutable capture session.

    Values are required at construction so this boundary cannot silently
    invent a camera mode, port, timeout, or output-retention limit. Callers
    must resolve configuration first and provide the validated request
    snapshot that is also used by :class:`SceneSnapshotIngress`.
    """

    capture_settings: CaptureSettingsSnapshot
    inference: RustLiveInferenceSettings
    executable_path: Path
    project_root: Path
    bind_address: str
    port: int
    maximum_message_bytes: int
    capture_settings_maximum_bytes: int
    read_timeout_seconds: float
    source_silence_timeout_seconds: float
    startup_timeout_seconds: float
    stop_timeout_seconds: float
    graceful_stop_timeout_seconds: float
    poll_interval_ms: int
    connect_timeout_ms: int
    write_timeout_ms: int
    reconnect_initial_delay_ms: int
    reconnect_maximum_delay_ms: int
    output_tail_capacity_bytes: int
    capture_helper_executable_path: Path | None = None
    capture_helper_ready_record_max_bytes: int | None = None
    capture_helper_frame_header_max_bytes: int | None = None
    capture_helper_slow_reader_delay_ms: int | None = None
    capture_helper_pull_timeout_ms: int | None = None
    capture_helper_readiness_timeout_ms: int | None = None
    capture_helper_child_exit_timeout_ms: int | None = None
    capture_helper_child_exit_poll_interval_ms: int | None = None

    def validate(self) -> None:
        """Reject invalid settings before creating a listener or child process."""
        if not isinstance(
            self.capture_settings,
            (
                CaptureSettingsSnapshotV1,
                CaptureSettingsSnapshotV2,
                CaptureSettingsSnapshotV3,
            ),
        ):
            raise RustLiveConfigurationError(
                "capture_settings must be a validated v1, v2, or v3 request snapshot"
            )
        if not isinstance(self.inference, RustLiveInferenceSettings):
            raise RustLiveConfigurationError(
                "inference must be a RustLiveInferenceSettings"
            )
        self.inference.validate()
        _capture_runtime_identity(self.capture_settings)
        _validate_capture_helper_settings(self)
        if (
            not isinstance(self.executable_path, Path)
            or not self.executable_path.is_absolute()
        ):
            raise RustLiveConfigurationError("executable_path must be an absolute Path")
        if not self.executable_path.is_file() or not os.access(
            self.executable_path, os.X_OK
        ):
            raise RustLiveConfigurationError(
                "configured Rust live executable must exist and be executable"
            )
        if (
            not isinstance(self.project_root, Path)
            or not self.project_root.is_absolute()
            or not self.project_root.is_dir()
        ):
            raise RustLiveConfigurationError(
                "project_root must be an existing directory"
            )
        for name, value in (
            ("port", self.port),
            ("maximum_message_bytes", self.maximum_message_bytes),
            ("capture_settings_maximum_bytes", self.capture_settings_maximum_bytes),
            ("connect_timeout_ms", self.connect_timeout_ms),
            ("write_timeout_ms", self.write_timeout_ms),
            ("reconnect_initial_delay_ms", self.reconnect_initial_delay_ms),
            ("reconnect_maximum_delay_ms", self.reconnect_maximum_delay_ms),
            ("output_tail_capacity_bytes", self.output_tail_capacity_bytes),
            ("poll_interval_ms", self.poll_interval_ms),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise RustLiveConfigurationError(f"{name} must be an integer")
        if not 0 <= self.port <= 65535:
            raise RustLiveConfigurationError("port must be between 0 and 65535")
        for name, value in (
            ("maximum_message_bytes", self.maximum_message_bytes),
            ("capture_settings_maximum_bytes", self.capture_settings_maximum_bytes),
            ("connect_timeout_ms", self.connect_timeout_ms),
            ("write_timeout_ms", self.write_timeout_ms),
            ("output_tail_capacity_bytes", self.output_tail_capacity_bytes),
            ("poll_interval_ms", self.poll_interval_ms),
        ):
            if value <= 0:
                raise RustLiveConfigurationError(f"{name} must be positive")
        if self.reconnect_initial_delay_ms < 0:
            raise RustLiveConfigurationError(
                "reconnect_initial_delay_ms must be nonnegative"
            )
        if self.reconnect_maximum_delay_ms < self.reconnect_initial_delay_ms:
            raise RustLiveConfigurationError(
                "reconnect_maximum_delay_ms must be at least the initial delay"
            )
        if self.capture_settings_maximum_bytes > sys.maxsize:
            raise RustLiveConfigurationError(
                "capture_settings_maximum_bytes exceeds the current process size range"
            )
        if self.poll_interval_ms > _MAX_U64:
            raise RustLiveConfigurationError(
                "poll_interval_ms exceeds the Rust u64 range"
            )
        for name, numeric_value in (
            ("read_timeout_seconds", self.read_timeout_seconds),
            ("source_silence_timeout_seconds", self.source_silence_timeout_seconds),
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("stop_timeout_seconds", self.stop_timeout_seconds),
            ("graceful_stop_timeout_seconds", self.graceful_stop_timeout_seconds),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(numeric_value)
                or numeric_value <= 0
            ):
                raise RustLiveConfigurationError(f"{name} must be finite and positive")
        _seconds_to_milliseconds(
            "startup_timeout_seconds", self.startup_timeout_seconds
        )
        graceful_stop_ms = _seconds_to_milliseconds(
            "graceful_stop_timeout_seconds", self.graceful_stop_timeout_seconds
        )
        parent_stop_ms = _seconds_to_milliseconds(
            "stop_timeout_seconds", self.stop_timeout_seconds
        )
        if graceful_stop_ms >= parent_stop_ms:
            raise RustLiveConfigurationError(
                "graceful_stop_timeout_seconds must leave time before the parent stop timeout"
            )
        # Reuse ingress validation here, before the actual listener is started.
        # The identical snapshot object will be used again by the supervisor.
        try:
            SceneIngressConfig(
                bind_address=self.bind_address,
                port=self.port,
                maximum_message_bytes=self.maximum_message_bytes,
                read_timeout_seconds=self.read_timeout_seconds,
                capture_settings=self.capture_settings,
            )
        except ValueError as error:
            raise RustLiveConfigurationError(str(error)) from error


def _gstreamer_capture_selector(
    capture_settings: CaptureSettingsSnapshot,
) -> str | None:
    """Return a valid physical-camera selector from the exact v3 request."""
    requested = capture_settings.requested
    device_id = requested.get("device_id")
    if not isinstance(device_id, str) or not device_id.casefold().startswith(
        "gstreamer-device-"
    ):
        return None
    if not device_id.startswith(_GSTREAMER_CAPTURE_SELECTOR_PREFIX):
        raise RustLiveConfigurationError(
            "physical capture requires the supported gstreamer-device-v1 selector"
        )
    if not isinstance(capture_settings, CaptureSettingsSnapshotV3):
        raise RustLiveConfigurationError(
            "a stable GStreamer camera selector requires capture-settings v3"
        )
    if not _GSTREAMER_CAPTURE_SELECTOR_PATTERN.fullmatch(device_id):
        raise RustLiveConfigurationError(
            "physical capture requires a versioned lowercase GStreamer selector"
        )
    requested_backend = requested.get("backend")
    if requested_backend not in {"auto", "gstreamer"}:
        raise RustLiveConfigurationError(
            "physical GStreamer capture requires the auto or gstreamer backend"
        )
    if (
        requested.get("image_space") != "raw"
        or requested.get("rectification_revision") != 0
    ):
        raise RustLiveConfigurationError(
            "physical GStreamer capture requires raw, unrectified camera frames"
        )
    camera_controls = requested.get("camera_controls")
    if (
        not isinstance(camera_controls, dict)
        or camera_controls.get("mode") != "source_defaults"
    ):
        raise RustLiveConfigurationError(
            "physical GStreamer capture uses webcam defaults without writing controls"
        )
    return device_id


def _validate_capture_helper_settings(settings: RustLiveSettings) -> None:
    """Bind optional helper launch settings to the exact configured selector."""
    selector = _gstreamer_capture_selector(settings.capture_settings)
    helper_values = (
        settings.capture_helper_executable_path,
        settings.capture_helper_ready_record_max_bytes,
        settings.capture_helper_frame_header_max_bytes,
        settings.capture_helper_slow_reader_delay_ms,
        settings.capture_helper_pull_timeout_ms,
        settings.capture_helper_readiness_timeout_ms,
        settings.capture_helper_child_exit_timeout_ms,
        settings.capture_helper_child_exit_poll_interval_ms,
    )
    if selector is None:
        if any(value is not None for value in helper_values):
            raise RustLiveConfigurationError(
                "GStreamer helper settings require a stable selector in the exact capture request"
            )
        return

    helper_path = settings.capture_helper_executable_path
    if not isinstance(helper_path, Path) or not helper_path.is_absolute():
        raise RustLiveConfigurationError(
            "capture_helper_executable_path must be an absolute Path"
        )
    if not helper_path.is_file() or not os.access(helper_path, os.X_OK):
        raise RustLiveConfigurationError(
            "configured GStreamer capture helper must exist and be executable"
        )

    for name, value in (
        (
            "capture_helper_ready_record_max_bytes",
            settings.capture_helper_ready_record_max_bytes,
        ),
        (
            "capture_helper_frame_header_max_bytes",
            settings.capture_helper_frame_header_max_bytes,
        ),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 < value <= _MAX_FRAME_RECORD_HEADER_BYTES
        ):
            raise RustLiveConfigurationError(
                f"{name} must be a positive 32-bit frame-record limit"
            )

    delay_ms = settings.capture_helper_slow_reader_delay_ms
    if (
        not isinstance(delay_ms, int)
        or isinstance(delay_ms, bool)
        or not 0 <= delay_ms <= _MAX_U64
    ):
        raise RustLiveConfigurationError(
            "capture_helper_slow_reader_delay_ms must be a nonnegative integer"
        )
    timeout_values = (
        ("capture_helper_pull_timeout_ms", settings.capture_helper_pull_timeout_ms),
        (
            "capture_helper_readiness_timeout_ms",
            settings.capture_helper_readiness_timeout_ms,
        ),
        (
            "capture_helper_child_exit_timeout_ms",
            settings.capture_helper_child_exit_timeout_ms,
        ),
        (
            "capture_helper_child_exit_poll_interval_ms",
            settings.capture_helper_child_exit_poll_interval_ms,
        ),
    )
    for name, value in timeout_values:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 < value <= _MAX_U64
        ):
            raise RustLiveConfigurationError(
                f"{name} must be a positive integer in the Rust u64 range"
            )
    pull_timeout_ms = settings.capture_helper_pull_timeout_ms
    readiness_timeout_ms = settings.capture_helper_readiness_timeout_ms
    assert pull_timeout_ms is not None
    assert readiness_timeout_ms is not None
    if delay_ms >= pull_timeout_ms:
        raise RustLiveConfigurationError(
            "capture helper slow-reader delay must be shorter than its sample-pull timeout"
        )
    startup_timeout_ms = _seconds_to_milliseconds(
        "startup_timeout_seconds", settings.startup_timeout_seconds
    )
    if readiness_timeout_ms >= startup_timeout_ms:
        raise RustLiveConfigurationError(
            "capture helper readiness timeout must leave time in the overall live startup budget"
        )


@dataclass(frozen=True)
class RustLiveStatus:
    """Bounded live-process and state-publication status for diagnostics."""

    process_started: bool
    process_running: bool
    ready: bool
    process_return_code: int | None
    process_status: str
    failure: str | None
    ingress_accepted_snapshots: int
    ingress_rejected_handshakes: int
    ingress_queue_replacements: int
    state_snapshots_published: int
    source_health: str
    last_observation_age_seconds: float | None
    source_silence_timeout_seconds: float
    stderr_tail: str


class RustLiveAuditIngress:
    """Receive bounded scalar timing records from one supervised Rust child.

    The listener binds to IPv4 loopback on an ephemeral port and has no auth
    handshake. Its records are provenance/diagnostics only. A bounded retained
    tail supports correlation without turning the receiver into an unbounded
    event buffer; the caller's recorder callback is expected to enqueue to its
    own bounded asynchronous exporter.
    """

    def __init__(
        self,
        *,
        queue_capacity: int,
        maximum_record_bytes: int,
        connect_timeout_ms: int,
        write_timeout_ms: int,
        startup_timeout_seconds: float,
        retained_record_capacity: int,
        on_record: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> None:
        for name, value in (
            ("queue_capacity", queue_capacity),
            ("maximum_record_bytes", maximum_record_bytes),
            ("connect_timeout_ms", connect_timeout_ms),
            ("write_timeout_ms", write_timeout_ms),
            ("retained_record_capacity", retained_record_capacity),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise RustLiveConfigurationError(f"{name} must be an integer")
        if not 1 <= queue_capacity <= _MAX_LIVE_AUDIT_QUEUE_CAPACITY:
            raise RustLiveConfigurationError(
                f"queue_capacity must be between 1 and {_MAX_LIVE_AUDIT_QUEUE_CAPACITY}"
            )
        if (
            not _MIN_LIVE_AUDIT_RECORD_BYTES
            <= maximum_record_bytes
            <= _MAX_LIVE_AUDIT_RECORD_BYTES
        ):
            raise RustLiveConfigurationError(
                "maximum_record_bytes must be between "
                f"{_MIN_LIVE_AUDIT_RECORD_BYTES} and {_MAX_LIVE_AUDIT_RECORD_BYTES}"
            )
        if connect_timeout_ms <= 0 or write_timeout_ms <= 0:
            raise RustLiveConfigurationError("audit socket deadlines must be positive")
        if (
            isinstance(startup_timeout_seconds, bool)
            or not isinstance(startup_timeout_seconds, (int, float))
            or not math.isfinite(startup_timeout_seconds)
            or startup_timeout_seconds <= 0
        ):
            raise RustLiveConfigurationError(
                "startup_timeout_seconds must be finite and positive"
            )
        if retained_record_capacity < 1:
            raise RustLiveConfigurationError(
                "retained_record_capacity must be positive"
            )

        self.queue_capacity = queue_capacity
        self.maximum_record_bytes = maximum_record_bytes
        self.connect_timeout_ms = connect_timeout_ms
        self.write_timeout_ms = write_timeout_ms
        self._read_timeout_seconds = write_timeout_ms / 1000.0
        self._initial_record_timeout_seconds = (
            startup_timeout_seconds + self._read_timeout_seconds
        )
        self._records: deque[dict[str, Any]] = deque(maxlen=retained_record_capacity)
        self._on_record = on_record
        self._server: asyncio.AbstractServer | None = None
        self._handler_tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._address: tuple[str, int] | None = None
        self._connection_count = 0
        self._record_count = 0
        self._record_counts: Counter[str] = Counter()
        self._malformed_records = 0
        self._oversized_records = 0
        self._consumer_dropped_records = 0
        self._protocol_error: str | None = None
        self._footer: dict[str, Any] | None = None
        self._footer_received_monotonic_ns: int | None = None
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        """Return the started listener's local loopback address."""
        if self._address is None:
            raise RuntimeError("Rust live audit ingress has not started")
        return self._address

    @property
    def retained_records(self) -> list[dict[str, Any]]:
        """Return a copy of the bounded record tail."""
        return list(self._records)

    async def start(self) -> tuple[str, int]:
        """Start a local-only listener before launching the Rust child."""
        if self._server is not None:
            return self.address
        if self._closed:
            raise RuntimeError("a closed Rust live audit ingress cannot restart")
        self._server = await asyncio.start_server(
            self._handle_connection, host="127.0.0.1", port=0
        )
        sockets: list[Any] = list(self._server.sockets or [])
        if len(sockets) != 1:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            raise RuntimeError("Rust live audit ingress did not create one listener")
        host, port = sockets[0].getsockname()[:2]
        self._address = (str(host), int(port))
        return self._address

    async def close(self) -> None:
        """Close admission and wait briefly for the child stream to finish."""
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._handler_tasks:
            done, pending = await asyncio.wait(
                tuple(self._handler_tasks), timeout=self._read_timeout_seconds
            )
            for writer in tuple(self._writers):
                writer.close()
            for task in pending:
                task.cancel()
            if done or pending:
                await asyncio.gather(*done, *pending, return_exceptions=True)

    def summary(self) -> dict[str, Any]:
        """Return bounded receipt counts and explicit completeness status."""
        footer = self._footer
        footer_record_count_matches = (
            footer is not None
            and footer.get("records_written_before_footer") == self._record_count - 1
        )
        complete = (
            footer is not None
            and self._protocol_error is None
            and self._consumer_dropped_records == 0
            and footer_record_count_matches
        )
        return {
            "schema_version": _LIVE_AUDIT_SCHEMA_VERSION,
            "status": "complete" if complete else "partial",
            "connection_count": self._connection_count,
            "records_received": self._record_count,
            "record_counts": dict(self._record_counts),
            "malformed_records": self._malformed_records,
            "oversized_records": self._oversized_records,
            "consumer_dropped_records": self._consumer_dropped_records,
            "protocol_error": self._protocol_error,
            "footer_received": footer is not None,
            "footer_record_count_matches": footer_record_count_matches,
            "footer_received_monotonic_ns": self._footer_received_monotonic_ns,
            "footer": footer,
            "retained_record_count": len(self._records),
            "retained_record_limit": self._records.maxlen,
            "retained_records_truncated": self._record_count > len(self._records),
        }

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handler_tasks.add(task)
        self._writers.add(writer)
        self._connection_count += 1
        if self._connection_count > 1:
            self._set_protocol_error("received more than one Rust audit connection")
            writer.close()
            await writer.wait_closed()
            self._writers.discard(writer)
            if task is not None:
                self._handler_tasks.discard(task)
            return

        try:
            while True:
                try:
                    header = await asyncio.wait_for(
                        reader.readexactly(4),
                        timeout=(
                            self._initial_record_timeout_seconds
                            if self._record_count == 0
                            else self._read_timeout_seconds
                        ),
                    )
                except asyncio.IncompleteReadError as error:
                    if error.partial:
                        self._malformed_records += 1
                        self._set_protocol_error("truncated audit record length")
                    break
                except asyncio.TimeoutError:
                    self._set_protocol_error(
                        "timed out waiting for audit record length"
                    )
                    break

                record_length = struct.unpack(">I", header)[0]
                if record_length == 0:
                    self._malformed_records += 1
                    self._set_protocol_error("audit record length must be positive")
                    break
                if record_length > self.maximum_record_bytes:
                    self._oversized_records += 1
                    self._set_protocol_error(
                        "audit record exceeds configured maximum record size"
                    )
                    break
                try:
                    payload = await asyncio.wait_for(
                        reader.readexactly(record_length),
                        timeout=self._read_timeout_seconds,
                    )
                except asyncio.IncompleteReadError:
                    self._malformed_records += 1
                    self._set_protocol_error("truncated audit record payload")
                    break
                except asyncio.TimeoutError:
                    self._set_protocol_error(
                        "timed out waiting for audit record payload"
                    )
                    break

                try:
                    value = json.loads(
                        payload.decode("utf-8"),
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                    record = _validate_live_audit_record(value)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as error:
                    self._malformed_records += 1
                    self._set_protocol_error(f"invalid audit record: {error}")
                    break

                received_record = {
                    **record,
                    "audit_receiver_monotonic_ns": time.perf_counter_ns(),
                }
                self._record_count += 1
                self._record_counts[record["record_type"]] += 1
                self._records.append(received_record)
                if self._on_record is not None:
                    try:
                        accepted = self._on_record(received_record)
                    except Exception as error:
                        self._consumer_dropped_records += 1
                        self._set_protocol_error(
                            f"audit record consumer failed: {type(error).__name__}"
                        )
                        break
                    if accepted is False:
                        self._consumer_dropped_records += 1
                if record["record_type"] == "footer":
                    self._footer = record
                    self._footer_received_monotonic_ns = received_record[
                        "audit_receiver_monotonic_ns"
                    ]
                    break
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self._writers.discard(writer)
            if task is not None:
                self._handler_tasks.discard(task)

    def _set_protocol_error(self, message: str) -> None:
        if self._protocol_error is None:
            self._protocol_error = message


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is not allowed")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects before validating the versioned schema."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _audit_integer(
    record: dict[str, Any], name: str, *, minimum: int = 0, maximum: int = _MAX_U64
) -> int:
    value = record.get(name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer in {minimum}..={maximum}")
    return value


def _audit_identity(record: dict[str, Any]) -> None:
    for name in ("engine_instance_id", "source_session_id", "frame_id"):
        value = record.get(name)
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(
                f"{name} must be a non-empty string of at most 256 characters"
            )


def _audit_exact_fields(
    record: dict[str, Any], *, required: set[str], optional: set[str] | None = None
) -> None:
    optional_fields = optional or set()
    fields = set(record)
    missing = required - fields
    unknown = fields - required - optional_fields
    if missing or unknown:
        raise ValueError("audit record fields do not match schema version 1")


def _audit_source_timestamp(record: dict[str, Any]) -> None:
    value = record.get("source_timestamp")
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("source_timestamp must be an object or null")
    if set(value) != {"ticks", "unit", "clock", "domain_id", "stage"}:
        raise ValueError("source_timestamp fields do not match schema version 1")
    if not isinstance(value["ticks"], str) or not value["ticks"].isdecimal():
        raise ValueError("source_timestamp ticks must be decimal text")
    if value["unit"] != "nanoseconds":
        raise ValueError("source_timestamp unit must be nanoseconds")
    if value["clock"] not in {"monotonic", "unix_epoch", "sensor"}:
        raise ValueError("source_timestamp clock is not supported")
    if not isinstance(value["domain_id"], str) or len(value["domain_id"]) > 256:
        raise ValueError("source_timestamp domain_id is invalid")
    if value["stage"] not in {
        "exposure",
        "camera_read_complete",
        "shared_memory_publication",
        "frame_arrival",
        "inference_started",
        "inference_complete",
        "batch_publication",
    }:
        raise ValueError("source_timestamp stage is not supported")


def _audit_counters(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object or null")
    for key, count in value.items():
        if not isinstance(key, str) or (
            count is not None
            and (not isinstance(count, int) or isinstance(count, bool) or count < 0)
        ):
            raise ValueError(f"{name} contains an invalid counter")


def _validate_live_audit_record(value: Any) -> dict[str, Any]:
    """Validate supported scalar-only Rust audit protocol versions."""
    if not isinstance(value, dict):
        raise ValueError("audit record must be an object")
    record_type = value.get("record_type")
    if record_type not in {
        "frame",
        "scene_socket_write",
        "observation_outcome",
        "footer",
    }:
        raise ValueError("record_type is unknown")
    schema_version = _audit_integer(value, "schema_version", minimum=1, maximum=2)

    if record_type == "footer":
        _audit_exact_fields(
            value,
            required={
                "record_type",
                "schema_version",
                "export_status",
                "records_written_before_footer",
                "dropped_records",
                "scene_socket_write_successes",
                "scene_socket_write_failures",
                "skipped_observations",
                "rejected_observations",
                "failed_observations",
                "clean_shutdown",
            },
            optional={"pipeline_counters", "scene_transport_counters"},
        )
        for name in (
            "records_written_before_footer",
            "dropped_records",
            "scene_socket_write_successes",
            "scene_socket_write_failures",
            "skipped_observations",
            "rejected_observations",
            "failed_observations",
        ):
            _audit_integer(value, name)
        if value.get("export_status") != "complete":
            raise ValueError("footer export_status must be complete")
        if not isinstance(value.get("clean_shutdown"), bool):
            raise ValueError("footer clean_shutdown must be boolean")
        _audit_counters(value.get("pipeline_counters"), "pipeline_counters")
        _audit_counters(
            value.get("scene_transport_counters"), "scene_transport_counters"
        )
        return value

    _audit_integer(value, "process_id", minimum=1, maximum=(1 << 32) - 1)
    _audit_identity(value)
    if record_type == "scene_socket_write":
        _audit_exact_fields(
            value,
            required={
                "record_type",
                "schema_version",
                "process_id",
                "engine_instance_id",
                "source_session_id",
                "frame_id",
                "transport_sequence",
                "write_attempt",
                "outcome",
                "duration_ms",
            },
        )
        _audit_integer(value, "transport_sequence", minimum=1)
        _audit_integer(value, "write_attempt", minimum=1)
        if value.get("outcome") not in {"written", "failed"}:
            raise ValueError("scene write outcome must be written or failed")
        _audit_duration(value.get("duration_ms"), "duration_ms", allow_none=False)
        return value

    for name in (
        "configuration_revision",
        "tracking_configuration_revision",
        "rectification_revision",
    ):
        _audit_integer(value, name)
    if "calibration_revision" not in value or "source_timestamp" not in value:
        raise ValueError("audit record is missing required nullable fields")
    calibration_revision = value.get("calibration_revision")
    if calibration_revision is not None:
        _audit_integer(
            {"calibration_revision": calibration_revision}, "calibration_revision"
        )
    _audit_source_timestamp(value)
    if record_type == "observation_outcome":
        _audit_exact_fields(
            value,
            required={
                "record_type",
                "schema_version",
                "process_id",
                "engine_instance_id",
                "source_session_id",
                "frame_id",
                "transport_sequence",
                "configuration_revision",
                "tracking_configuration_revision",
                "rectification_revision",
                "calibration_revision",
                "source_timestamp",
                "outcome",
                "stage",
                "reason",
            },
        )
        transport_sequence = value.get("transport_sequence")
        if transport_sequence is not None:
            _audit_integer(
                {"transport_sequence": transport_sequence},
                "transport_sequence",
                minimum=1,
            )
        for name in ("outcome", "stage", "reason"):
            item = value.get(name)
            if not isinstance(item, str) or not item or len(item) > 128:
                raise ValueError(
                    f"{name} must be a non-empty string of at most 128 characters"
                )
        if value["outcome"] not in {"skipped", "rejected", "failed", "degraded"}:
            raise ValueError("observation outcome is not supported")
        return value

    _audit_exact_fields(
        value,
        required={
            "record_type",
            "schema_version",
            "process_id",
            "engine_instance_id",
            "source_session_id",
            "frame_id",
            "transport_sequence",
            "configuration_revision",
            "tracking_configuration_revision",
            "rectification_revision",
            "calibration_revision",
            "source_timestamp",
            "detection_count",
            "track_count",
            "durations_ms",
        },
    )
    _audit_integer(value, "transport_sequence", minimum=1)
    _audit_integer(value, "detection_count")
    _audit_integer(value, "track_count")
    durations = value.get("durations_ms")
    if not isinstance(durations, dict):
        raise ValueError("durations_ms must be an object")
    if schema_version == 1:
        required_durations = {
            "host_read_complete_to_inference_start",
            "preprocess",
            "inference",
            "decode_and_nms",
            "calibration",
            "tracking_and_state_assembly",
            "scene_json_serialization",
            "scene_queue_submit",
            "host_read_complete_to_scene_queue_submit",
        }
        optional_durations = {
            "calibration",
            "tracking_and_state_assembly",
            "scene_json_serialization",
            "scene_queue_submit",
        }
    else:
        required_durations = {
            "capture_helper_appsink_pull_wait",
            "capture_helper_rgb_buffer_allocation",
            "capture_helper_rgb_mapping_copy",
            "parent_framed_record_read",
            "parent_record_read_complete_to_inference_start",
            "preprocess",
            "inference",
            "decode_and_nms",
            "calibration",
            "tracking_and_state_assembly",
            "scene_json_serialization",
            "scene_queue_submit",
            "parent_record_read_complete_to_scene_queue_submit",
        }
        optional_durations = {
            "capture_helper_appsink_pull_wait",
            "capture_helper_rgb_buffer_allocation",
            "capture_helper_rgb_mapping_copy",
            "parent_framed_record_read",
            "calibration",
            "tracking_and_state_assembly",
            "scene_json_serialization",
            "scene_queue_submit",
        }
    if set(durations) != required_durations:
        raise ValueError(
            f"durations_ms fields do not match schema version {schema_version}"
        )
    for name, duration in durations.items():
        _audit_duration(
            duration,
            f"durations_ms.{name}",
            allow_none=name in optional_durations,
        )
    return value


def _audit_duration(value: Any, name: str, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative duration")


class RustLiveSupervisor:
    """Own one Rust child and the scene-state publication path it feeds."""

    def __init__(
        self,
        settings: RustLiveSettings,
        publisher: SceneSnapshotStatePublisher,
        audit_ingress: RustLiveAuditIngress | None = None,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._publisher = publisher
        self._audit_ingress = audit_ingress
        self._ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address=settings.bind_address,
                port=settings.port,
                maximum_message_bytes=settings.maximum_message_bytes,
                read_timeout_seconds=settings.read_timeout_seconds,
                capture_settings=settings.capture_settings,
            )
        )
        self._worker = SceneSnapshotStateWorker(
            self._ingress,
            publisher,
            source_silence_timeout_seconds=settings.source_silence_timeout_seconds,
        )
        self._process: asyncio.subprocess.Process | None = None
        self._process_wait_task: asyncio.Task[int] | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stdout_tail = bytearray()
        self._stderr_tail = bytearray()
        self._engine_instance_id: str | None = None
        self._source_session_id: str | None = None
        self._failure: str | None = None
        self._started = False
        self._closed = False
        self._ready = False

    @property
    def status(self) -> RustLiveStatus:
        """Return current process, handshake, ingress and publisher state."""
        process = self._process
        return_code = process.returncode if process is not None else None
        if self._closed:
            process_status = "stopped"
        elif process is None:
            process_status = "not_started"
        elif self._failure is not None or (
            return_code is not None and return_code != 0
        ):
            process_status = "failed"
        elif return_code is None:
            process_status = "running"
        else:
            process_status = "completed"
        worker_stats = self._worker.stats
        source_health = (
            "process_failed"
            if process_status == "failed"
            else (
                "process_exited"
                if return_code is not None and self._ready and not self._closed
                else worker_stats.source_health
            )
        )
        return RustLiveStatus(
            process_started=process is not None,
            process_running=process is not None and return_code is None,
            ready=(
                self._ready
                and self._failure is None
                and not self._closed
                and process is not None
                and return_code is None
                and worker_stats.source_health == "healthy"
            ),
            process_return_code=return_code,
            process_status=process_status,
            failure=self._failure,
            ingress_accepted_snapshots=self._ingress.stats.accepted_snapshots,
            ingress_rejected_handshakes=self._ingress.stats.rejected_handshakes,
            ingress_queue_replacements=self._ingress.stats.queue_replacements,
            state_snapshots_published=worker_stats.published,
            source_health=source_health,
            last_observation_age_seconds=worker_stats.last_observation_age_seconds,
            source_silence_timeout_seconds=(
                worker_stats.source_silence_timeout_seconds
            ),
            stderr_tail=self._stderr_tail.decode("utf-8", errors="replace"),
        )

    @property
    def capture_session_acknowledgement(
        self,
    ) -> ValidatedCaptureSessionAcknowledgement | None:
        """Return the v3 acknowledgement accepted for this request, if any."""
        return self._ingress.capture_session_acknowledgement

    @property
    def worker_stats(self) -> SceneStateWorkerStats:
        """Expose bounded scene-worker counters without event history."""
        return self._worker.stats

    async def start(self) -> RustLiveStatus:
        """Start the child once and wait for a usable published localization."""
        if self._started:
            if self._ready:
                return self.status
            raise RuntimeError("Rust live supervisor startup is already in progress")
        if self._closed:
            raise RuntimeError("a closed Rust live supervisor cannot restart")
        self._started = True
        startup_deadline = (
            asyncio.get_running_loop().time() + self._settings.startup_timeout_seconds
        )

        try:
            await self._worker.start()
            if self._audit_ingress is not None:
                await self._audit_ingress.start()
            address = self._ingress.address
            self._engine_instance_id = str(uuid.uuid4())
            self._source_session_id = str(uuid.uuid4())
            environment = {
                key: os.environ[key]
                for key in _RUST_CHILD_ENVIRONMENT_KEYS
                if key in os.environ
            }
            if self._settings.inference.provider == "openvino":
                environment.update(
                    {
                        key: os.environ[key]
                        for key in _OPENVINO_CHILD_ENVIRONMENT_KEYS
                        if key in os.environ
                    }
                )
            self._process = await asyncio.create_subprocess_exec(
                str(self._settings.executable_path),
                *self._build_arguments(address),
                cwd=self._settings.project_root,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._process_wait_task = asyncio.create_task(
                self._process.wait(), name="rust-live-process-wait"
            )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            assert self._process.stderr is not None
            self._stdout_task = asyncio.create_task(
                self._drain_output(self._process.stdout, self._stdout_tail),
                name="rust-live-stdout-drain",
            )
            self._stderr_task = asyncio.create_task(
                self._drain_output(self._process.stderr, self._stderr_tail),
                name="rust-live-stderr-drain",
            )

            await self._send_capture_request(startup_deadline)
            await self._await_first_observation(startup_deadline)
            self._ready = True
            status = self.status
            if not status.ready:
                raise RustLiveStartupError(
                    "Rust live process became unhealthy before readiness completed"
                )
            return status
        except asyncio.CancelledError as error:
            self._failure = str(error) or type(error).__name__
            await asyncio.shield(self._stop())
            raise
        except Exception as error:
            self._failure = str(error) or type(error).__name__
            await asyncio.shield(self._stop())
            if isinstance(error, RustLiveStartupError):
                raise
            raise RustLiveStartupError(self._failure) from error

    async def close(self) -> None:
        """Terminate/reap the child before releasing its worker and ingress."""
        if self._closed:
            return
        self._closed = True
        await self._stop()

    def _build_arguments(self, address: tuple[str, int]) -> list[str]:
        """Pass detector, tracker and transport settings explicitly to the child."""
        assert self._engine_instance_id is not None
        assert self._source_session_id is not None
        settings = self._settings
        inference = settings.inference
        capture = _capture_runtime_identity(settings.capture_settings)
        arguments = [
            "--onnx",
            str(inference.onnx_path),
            "--source-checkpoint",
            str(inference.source_checkpoint_path),
            "--provider",
            inference.provider,
            "--image-size",
            str(inference.image_size),
            "--confidence",
            str(inference.confidence),
            "--iou",
            str(inference.iou),
            "--max-detections",
            str(inference.max_detections),
            "--warmup-runs",
            str(inference.warmup_runs),
            "--source-image-space",
            capture.image_space,
            "--tracking-ball-class-id",
            str(inference.ball_class_id),
            "--tracking-camera-profile-id",
            capture.camera_profile_id,
            "--tracking-association-gate-px",
            str(inference.association_gate_px),
            "--tracking-association-prediction-horizon-ms",
            str(inference.association_prediction_horizon_ms),
            "--tracking-lost-after-ms",
            str(inference.lost_after_ms),
            "--tracking-maximum-track-age-ms",
            str(inference.maximum_track_age_ms),
            "--tracking-maximum-active-tracks",
            str(inference.maximum_active_tracks),
            "--tracking-recent-observation-capacity",
            str(inference.recent_observation_capacity),
            "--tracking-configuration-revision",
            str(capture.configuration_revision),
            "--tracking-rectification-revision",
            str(capture.rectification_revision),
            "--capture-settings-stdin",
            "--capture-settings-max-bytes",
            str(settings.capture_settings_maximum_bytes),
            "--engine-instance-id",
            self._engine_instance_id,
            "--source-session-id",
            self._source_session_id,
            "--scene-snapshots-tcp",
            f"{address[0]}:{address[1]}",
            "--scene-snapshots-max-message-bytes",
            str(settings.maximum_message_bytes),
            "--scene-snapshots-connect-timeout-ms",
            str(settings.connect_timeout_ms),
            "--scene-snapshots-write-timeout-ms",
            str(settings.write_timeout_ms),
            "--scene-snapshots-reconnect-initial-delay-ms",
            str(settings.reconnect_initial_delay_ms),
            "--scene-snapshots-reconnect-maximum-delay-ms",
            str(settings.reconnect_maximum_delay_ms),
            "--startup-timeout-ms",
            str(
                _seconds_to_milliseconds(
                    "startup_timeout_seconds", settings.startup_timeout_seconds
                )
            ),
            "--poll-interval-ms",
            str(settings.poll_interval_ms),
            "--graceful-stop-timeout-ms",
            str(
                _seconds_to_milliseconds(
                    "graceful_stop_timeout_seconds",
                    settings.graceful_stop_timeout_seconds,
                )
            ),
        ]
        if inference.intra_op_threads is not None:
            arguments.extend(("--intra-op-threads", str(inference.intra_op_threads)))
        if inference.cue_class_id is not None:
            arguments.extend(("--tracking-cue-class-id", str(inference.cue_class_id)))
        if _gstreamer_capture_selector(settings.capture_settings) is not None:
            helper_path = settings.capture_helper_executable_path
            assert helper_path is not None
            assert settings.capture_helper_ready_record_max_bytes is not None
            assert settings.capture_helper_frame_header_max_bytes is not None
            assert settings.capture_helper_slow_reader_delay_ms is not None
            assert settings.capture_helper_pull_timeout_ms is not None
            assert settings.capture_helper_readiness_timeout_ms is not None
            assert settings.capture_helper_child_exit_timeout_ms is not None
            assert settings.capture_helper_child_exit_poll_interval_ms is not None
            arguments.extend(
                (
                    "--capture-source",
                    "gstreamer-physical",
                    "--capture-helper-executable",
                    str(helper_path),
                    "--capture-helper-ready-record-max-bytes",
                    str(settings.capture_helper_ready_record_max_bytes),
                    "--capture-helper-frame-header-max-bytes",
                    str(settings.capture_helper_frame_header_max_bytes),
                    "--capture-helper-slow-reader-delay-ms",
                    str(settings.capture_helper_slow_reader_delay_ms),
                    "--capture-helper-pull-timeout-ms",
                    str(settings.capture_helper_pull_timeout_ms),
                    "--capture-helper-readiness-timeout-ms",
                    str(settings.capture_helper_readiness_timeout_ms),
                    "--capture-helper-child-exit-timeout-ms",
                    str(settings.capture_helper_child_exit_timeout_ms),
                    "--capture-helper-child-exit-poll-interval-ms",
                    str(settings.capture_helper_child_exit_poll_interval_ms),
                )
            )
        if self._audit_ingress is not None:
            audit_address = self._audit_ingress.address
            arguments.extend(
                (
                    "--live-audit-tcp",
                    f"{audit_address[0]}:{audit_address[1]}",
                    "--live-audit-queue-capacity",
                    str(self._audit_ingress.queue_capacity),
                    "--live-audit-max-record-bytes",
                    str(self._audit_ingress.maximum_record_bytes),
                    "--live-audit-connect-timeout-ms",
                    str(self._audit_ingress.connect_timeout_ms),
                    "--live-audit-write-timeout-ms",
                    str(self._audit_ingress.write_timeout_ms),
                )
            )
        # The complete calibration bundle remains inside the exact capture
        # request on stdin. Do not pass a separately sourced calibration path
        # or revision set that could drift from the request-bound v3 ACK.
        return arguments

    async def _send_capture_request(self, startup_deadline: float) -> None:
        """Write the exact request once and enforce the shared startup deadline."""
        process = self._process
        assert process is not None
        stdin = process.stdin
        assert stdin is not None

        async def write_and_close() -> None:
            # The versioned capture snapshot encoding is deterministic. Write its
            # exact UTF-8 representation once, without a newline or wrapper,
            # then close stdin to make the child's one-shot request boundary
            # unambiguous.
            request_bytes = self._settings.capture_settings.to_json().encode("utf-8")
            stdin.write(request_bytes)
            await stdin.drain()
            stdin.close()
            await stdin.wait_closed()

        remaining = max(0.0, startup_deadline - asyncio.get_running_loop().time())
        try:
            await asyncio.wait_for(write_and_close(), timeout=remaining)
        except asyncio.TimeoutError as error:
            raise RustLiveStartupError(
                "Rust live process did not accept the capture request before "
                "the startup deadline"
            ) from error

    async def _await_first_observation(self, startup_deadline: float) -> None:
        """Require successful publication and fail promptly on child exit."""
        observation_task = asyncio.create_task(
            self._worker.wait_for_first_observation(),
            name="rust-live-first-observation",
        )
        process_wait = self._process_wait_task
        assert process_wait is not None
        try:
            remaining = max(0.0, startup_deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait(
                {observation_task, process_wait},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if observation_task in done:
                if process_wait.done():
                    raise self._child_exit_error(process_wait.result())
                return
            if process_wait in done:
                return_code = process_wait.result()
                raise self._child_exit_error(return_code)
            raise RustLiveStartupError(
                "Rust live process did not publish a usable localization before "
                "the startup deadline"
            )
        finally:
            if not observation_task.done():
                observation_task.cancel()
            await asyncio.gather(observation_task, return_exceptions=True)

    def _child_exit_error(self, return_code: int) -> RustLiveStartupError:
        details = self._stderr_tail.decode("utf-8", errors="replace").strip()
        suffix = f": {details}" if details else ""
        return RustLiveStartupError(
            f"Rust live process exited with status {return_code} before a usable "
            f"localization{suffix}"
        )

    async def _drain_output(
        self, stream: asyncio.StreamReader, destination: bytearray
    ) -> None:
        """Drain pipes while retaining only the configured diagnostic tail."""
        capacity = self._settings.output_tail_capacity_bytes
        while chunk := await stream.read(capacity):
            destination.extend(chunk)
            excess = len(destination) - capacity
            if excess > 0:
                del destination[:excess]

    async def _stop(self) -> None:
        """Idempotently terminate/reap the process and close consumer resources."""
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self._settings.stop_timeout_seconds
                )
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

        tasks = [
            task
            for task in (
                self._process_wait_task,
                self._stdout_task,
                self._stderr_task,
            )
            if task is not None
        ]
        for task in tasks:
            if task is not self._process_wait_task and not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await self._worker.close()
        finally:
            if self._audit_ingress is not None:
                await self._audit_ingress.close()
