"""Python-to-Rust live capture request resolution and serialization tests."""

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import numpy as np
import pytest
from referencing import Registry, Resource

from backend.perception.capture_configuration import (
    CaptureConfigurationError,
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
    ConfigurationReader,
    resolve_capture_settings,
)
from backend.vision.calibration.camera import CameraCalibrator
from backend.vision.calibration.perception_export import build_table_calibration_v1

_CROSS_RUNTIME_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "perception"
    / "contracts"
    / "capture-settings"
    / "v1"
    / "capture-settings-default.json"
)
_V2_CAPTURE_SETTINGS_FIXTURE = (
    _CROSS_RUNTIME_FIXTURE.parents[1] / "v2" / "capture-settings-default.json"
)
_V3_CAPTURE_SETTINGS_FIXTURE = (
    _CROSS_RUNTIME_FIXTURE.parents[1] / "v3" / "capture-settings-default.json"
)
_CAPTURE_SETTINGS_SCHEMA = _CROSS_RUNTIME_FIXTURE.with_name(
    "capture-settings.schema.json"
)
_CALIBRATED_CAPTURE_SETTINGS_FIXTURE = _CROSS_RUNTIME_FIXTURE.with_name(
    "capture-settings-with-calibration.json"
)
_PYTHON_CALIBRATION_FIXTURE = (
    _CROSS_RUNTIME_FIXTURE.parents[2]
    / "calibration"
    / "v1"
    / "table-calibration-python-export.json"
)


def _resolve_v1(
    configuration: ConfigurationReader,
    *,
    calibration_bundle: dict[str, object] | None = None,
) -> CaptureSettingsSnapshotV1:
    """Exercise the exact legacy request when a test is about v1 behavior."""
    snapshot = resolve_capture_settings(
        configuration,
        calibration_bundle=calibration_bundle,
        schema_version=1,
    )
    assert isinstance(snapshot, CaptureSettingsSnapshotV1)
    return snapshot


class MappingConfiguration:
    """Small dot-path config reader with the same default semantics as Config."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}

    def get_all(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)


def live_camera_configuration(
    values: dict[str, object] | None = None,
) -> MappingConfiguration:
    """Build a test configuration that explicitly selects the live camera.

    The application defaults currently select a demo video. Tests that exercise
    live-camera request construction must therefore clear that source rather
    than accidentally inheriting a file input.
    """
    selected_values = copy.deepcopy(values) if values is not None else {}
    vision = selected_values.setdefault("vision", {})
    assert isinstance(vision, dict)
    camera = vision.setdefault("camera", {})
    assert isinstance(camera, dict)
    camera.setdefault("video_file_path", "")
    return MappingConfiguration(selected_values)


class ChangingSnapshotConfiguration:
    """Swap source values after one atomic snapshot is returned."""

    def __init__(self, initial: dict[str, object], replacement: dict[str, object]):
        self.values = initial
        self.replacement = replacement
        self.snapshot_count = 0

    def get_all(self) -> dict[str, Any]:
        self.snapshot_count += 1
        snapshot = copy.deepcopy(self.values)
        self.values = self.replacement
        return snapshot


def test_explicit_live_request_matches_the_rust_shared_fixture() -> None:
    snapshot = _resolve_v1(live_camera_configuration())
    expected = json.loads(_CROSS_RUNTIME_FIXTURE.read_text(encoding="utf-8"))

    assert snapshot.as_dict() == expected
    assert json.loads(snapshot.to_json()) == expected
    assert "applied" not in expected
    assert "verified" not in expected
    schema = json.loads(_CAPTURE_SETTINGS_SCHEMA.read_text(encoding="utf-8"))
    calibration_schema_path = (
        _CROSS_RUNTIME_FIXTURE.parents[2]
        / "calibration"
        / "v1"
        / "table-calibration.schema.json"
    )
    calibration_schema = json.loads(calibration_schema_path.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        calibration_schema["$id"], Resource.from_contents(calibration_schema)
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema, registry=registry).validate(expected)
    invalid_identifier = json.loads(json.dumps(expected))
    invalid_identifier["requested"]["device_id"] = "camera-☃"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema, registry=registry).validate(
            invalid_identifier
        )
    assert snapshot.requested["reconnect_policy"] == {
        "enabled": True,
        "delay_seconds": 1.0,
        "maximum_attempts": 5,
        "stop_timeout_seconds": 5.0,
    }


def test_live_request_is_v2_and_uses_camera_dimensions_for_rgb_output() -> None:
    snapshot = resolve_capture_settings(live_camera_configuration(), schema_version=2)
    schema = json.loads(
        (
            _CROSS_RUNTIME_FIXTURE.parents[1] / "v2" / "capture-settings.schema.json"
        ).read_text(encoding="utf-8")
    )
    calibration_schema_path = (
        _CROSS_RUNTIME_FIXTURE.parents[2]
        / "calibration"
        / "v1"
        / "table-calibration.schema.json"
    )
    calibration_schema = json.loads(calibration_schema_path.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        calibration_schema["$id"], Resource.from_contents(calibration_schema)
    )

    assert snapshot.schema_version == 2
    assert isinstance(snapshot, CaptureSettingsSnapshotV2)
    requested = snapshot.requested
    assert requested["output_frame_geometry"] == {
        "width_px": 1920,
        "height_px": 1080,
    }
    profile_id = requested["camera_profile_id"]
    profile = requested["requested_profile"]
    assert isinstance(profile_id, str)
    assert profile_id.startswith("camera-profile-v2:")
    assert isinstance(profile, dict)
    assert profile["width_px"] == 1920
    assert profile["height_px"] == 1080
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema, registry=registry).validate(
        snapshot.as_dict()
    )


def test_v2_live_request_matches_the_cross_runtime_fixture() -> None:
    snapshot = resolve_capture_settings(live_camera_configuration(), schema_version=2)
    fixture = json.loads(_V2_CAPTURE_SETTINGS_FIXTURE.read_text(encoding="utf-8"))

    assert snapshot.to_dict() == fixture


def test_default_live_request_uses_v3_driver_controls_and_shared_fixture() -> None:
    snapshot = resolve_capture_settings(live_camera_configuration())
    fixture = json.loads(_V3_CAPTURE_SETTINGS_FIXTURE.read_text(encoding="utf-8"))

    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    assert snapshot.to_dict() == fixture
    assert snapshot.requested["camera_controls"] == {"mode": "source_defaults"}
    assert "requested_controls" not in snapshot.requested
    assert "control_requirements" not in snapshot.requested


def test_v3_require_policy_keeps_controls_and_requirements_explicit() -> None:
    snapshot = resolve_capture_settings(
        live_camera_configuration(
            {
                "vision": {
                    "camera": {
                        "control_policy": "require",
                        "exposure_mode": "manual",
                        "exposure_value": -6,
                    }
                }
            }
        )
    )

    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    assert snapshot.requested["camera_controls"] == {
        "mode": "require",
        "requirements": {
            "manual_exposure": True,
            "automatic_exposure": False,
            "gain_control": True,
        },
        "requested": {
            "exposure_mode": "manual",
            "exposure_value": -6,
            "gain": 1.0,
            "brightness": 0.5,
            "contrast": 0.5,
            "saturation": 0.5,
            "auto_focus": True,
            "buffer_size": 2,
        },
    }


def test_v3_source_defaults_does_not_validate_ignored_control_values() -> None:
    snapshot = resolve_capture_settings(
        live_camera_configuration(
            {
                "vision": {
                    "camera": {
                        "control_policy": "source_defaults",
                        "exposure_mode": "manual",
                        "exposure_value": None,
                        "gain": "ignored with driver defaults",
                    }
                }
            }
        )
    )

    assert snapshot.requested["camera_controls"] == {"mode": "source_defaults"}


@pytest.mark.parametrize("policy", [None, "automatic", "require_controls", []])
def test_v3_rejects_unknown_camera_control_policy(policy: object) -> None:
    with pytest.raises(CaptureConfigurationError, match="control_policy"):
        resolve_capture_settings(
            live_camera_configuration(
                {"vision": {"camera": {"control_policy": policy}}}
            )
        )


def test_bundled_video_default_is_rejected_for_live_capture() -> None:
    with pytest.raises(CaptureConfigurationError, match="recorded replay"):
        resolve_capture_settings(MappingConfiguration())


def test_explicitly_cleared_video_path_allows_numeric_live_camera() -> None:
    snapshot = resolve_capture_settings(
        MappingConfiguration(
            {"vision": {"camera": {"video_file_path": "", "device_id": 1}}}
        )
    )

    assert snapshot.requested["device_id"] == "device-index:1"


def test_v3_stable_capture_selector_overrides_numeric_id_and_changes_profile() -> None:
    selector = "gstreamer-device-v1:" + ("a" * 64)
    other_selector = "gstreamer-device-v1:" + ("b" * 64)

    snapshot = resolve_capture_settings(
        live_camera_configuration(
            {
                "vision": {"camera": {"device_id": 3}},
                "perception": {"capture": {"device_selector": selector}},
            }
        )
    )
    other_snapshot = resolve_capture_settings(
        live_camera_configuration(
            {"perception": {"capture": {"device_selector": other_selector}}}
        )
    )

    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    assert snapshot.requested["device_id"] == selector
    assert (
        snapshot.requested["camera_profile_id"]
        != other_snapshot.requested["camera_profile_id"]
    )


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "/dev/v4l/by-id/webcam",
        "gstreamer-device-v1:" + ("A" * 64),
        "gstreamer-device-v1:" + ("a" * 63),
        42,
    ],
)
def test_invalid_stable_camera_selectors_fail_closed(selector: object) -> None:
    with pytest.raises(CaptureConfigurationError, match="device_selector"):
        resolve_capture_settings(
            live_camera_configuration(
                {"perception": {"capture": {"device_selector": selector}}}
            )
        )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_stable_camera_selector_requires_capture_settings_v3(
    schema_version: int,
) -> None:
    selector = "gstreamer-device-v1:" + ("a" * 64)
    with pytest.raises(CaptureConfigurationError, match="schema v3"):
        resolve_capture_settings(
            live_camera_configuration(
                {"perception": {"capture": {"device_selector": selector}}}
            ),
            schema_version=schema_version,
        )


def test_stable_camera_selector_rejects_non_gstreamer_backend() -> None:
    selector = "gstreamer-device-v1:" + ("a" * 64)
    with pytest.raises(CaptureConfigurationError, match="auto or gstreamer"):
        resolve_capture_settings(
            live_camera_configuration(
                {
                    "vision": {"camera": {"backend": "opencv"}},
                    "perception": {"capture": {"device_selector": selector}},
                }
            )
        )


def test_configured_video_file_path_is_rejected_for_live_capture() -> None:
    with pytest.raises(CaptureConfigurationError, match="video_file_path"):
        resolve_capture_settings(
            MappingConfiguration(
                {"vision": {"camera": {"video_file_path": "assets/demo3.mp4"}}}
            )
        )


def test_non_string_video_file_path_is_rejected() -> None:
    with pytest.raises(CaptureConfigurationError, match="must be a string or null"):
        resolve_capture_settings(
            MappingConfiguration({"vision": {"camera": {"video_file_path": 3}}})
        )


def test_v2_profile_fingerprint_includes_explicit_output_geometry() -> None:
    camera_mode: dict[str, object] = {
        "vision": {"camera": {"resolution": [1920, 1080]}}
    }
    native_size = resolve_capture_settings(live_camera_configuration(camera_mode))
    scaled = resolve_capture_settings(
        live_camera_configuration(
            {
                **camera_mode,
                "perception": {"capture": {"output_resolution": [1280, 720]}},
            }
        )
    )

    assert (
        native_size.requested["requested_profile"]
        == scaled.requested["requested_profile"]
    )
    assert (
        native_size.requested["output_frame_geometry"]
        != scaled.requested["output_frame_geometry"]
    )
    assert (
        native_size.requested["camera_profile_id"]
        != scaled.requested["camera_profile_id"]
    )


@pytest.mark.parametrize(
    "output_resolution",
    [[], [640], [640, 480, 3], [True, 480], [0, 480], [4_294_967_296, 480]],
)
def test_v2_output_resolution_must_be_a_valid_dimension_pair(
    output_resolution: list[object],
) -> None:
    with pytest.raises(CaptureConfigurationError, match="output_resolution"):
        resolve_capture_settings(
            live_camera_configuration(
                {"perception": {"capture": {"output_resolution": output_resolution}}}
            )
        )


def test_v2_rgb8_frame_limits_accept_exact_boundary_and_reject_under_limits() -> None:
    required_bytes = 1920 * 1080 * 3
    exact_boundary = resolve_capture_settings(
        live_camera_configuration(
            {
                "perception": {
                    "capture": {
                        "maximum_rgb_frame_bytes": required_bytes,
                        "maximum_aggregate_rgb_frame_bytes": required_bytes,
                    }
                }
            }
        )
    )
    assert exact_boundary.requested["output_frame_geometry"] == {
        "width_px": 1920,
        "height_px": 1080,
    }

    under_per_frame = live_camera_configuration(
        {
            "perception": {
                "capture": {
                    "maximum_rgb_frame_bytes": required_bytes - 1,
                    "maximum_aggregate_rgb_frame_bytes": required_bytes,
                }
            }
        }
    )
    with pytest.raises(CaptureConfigurationError, match="maximum_rgb_frame_bytes"):
        resolve_capture_settings(under_per_frame)

    under_aggregate = live_camera_configuration(
        {
            "perception": {
                "capture": {
                    "maximum_rgb_frame_bytes": required_bytes,
                    "maximum_aggregate_rgb_frame_bytes": required_bytes - 1,
                }
            }
        }
    )
    with pytest.raises(
        CaptureConfigurationError, match="maximum_aggregate_rgb_frame_bytes"
    ):
        resolve_capture_settings(under_aggregate)


def test_rectified_v2_calibration_dimensions_match_emitted_geometry() -> None:
    configuration = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1920, 1080]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                    "output_resolution": [1280, 720],
                }
            },
        }
    )
    settings = resolve_capture_settings(configuration)
    calibration = json.loads(_PYTHON_CALIBRATION_FIXTURE.read_text(encoding="utf-8"))
    calibration["camera_profile_id"] = settings.requested["camera_profile_id"]

    with pytest.raises(CaptureConfigurationError, match="calibration dimensions"):
        resolve_capture_settings(configuration, calibration_bundle=calibration)


def test_request_profile_identity_tracks_capture_settings_not_global_revision() -> None:
    first = _resolve_v1(
        live_camera_configuration({"metadata": {"configuration_revision": 12}})
    )
    unrelated_change = _resolve_v1(
        live_camera_configuration({"metadata": {"configuration_revision": 13}})
    )
    changed_camera = _resolve_v1(
        live_camera_configuration(
            {
                "metadata": {"configuration_revision": 13},
                "vision": {"camera": {"resolution": [1280, 720]}},
            }
        )
    )

    assert first.configuration_revision == 12
    assert unrelated_change.configuration_revision == 13
    assert (
        first.requested["camera_profile_id"]
        == unrelated_change.requested["camera_profile_id"]
    )
    assert (
        first.requested["camera_profile_id"]
        != changed_camera.requested["camera_profile_id"]
    )
    profile = changed_camera.requested["requested_profile"]
    assert isinstance(profile, dict)
    assert profile["width_px"] == 1280


def test_explicit_v1_request_ignores_new_output_resolution_configuration() -> None:
    baseline = _resolve_v1(live_camera_configuration())
    with_new_v2_option = _resolve_v1(
        live_camera_configuration(
            {"perception": {"capture": {"output_resolution": [1280, 720]}}}
        )
    )

    assert baseline.to_json() == with_new_v2_option.to_json()
    assert (
        baseline.requested["camera_profile_id"]
        == with_new_v2_option.requested["camera_profile_id"]
    )


def test_request_profile_identity_tracks_geometry_but_not_photometric_controls() -> (
    None
):
    baseline = _resolve_v1(live_camera_configuration())
    changed_controls = _resolve_v1(
        live_camera_configuration(
            {
                "vision": {
                    "camera": {
                        "gain": 2.0,
                        "brightness": 0.2,
                        "contrast": 0.8,
                        "saturation": 0.3,
                        "auto_focus": False,
                    }
                }
            }
        )
    )
    assert (
        baseline.requested["camera_profile_id"]
        == changed_controls.requested["camera_profile_id"]
    )

    geometry_changes: list[dict[str, object]] = [
        {"vision": {"camera": {"device_id": 1}}},
        {"vision": {"camera": {"backend": "opencv"}}},
        {"vision": {"camera": {"resolution": [1280, 720]}}},
        {"vision": {"camera": {"fps": 60}}},
        {"perception": {"capture": {"pixel_format": "BGR8"}}},
        {"perception": {"capture": {"image_space": "rectified"}}},
        {"perception": {"capture": {"rectification_revision": 1}}},
    ]
    for change in geometry_changes:
        changed = _resolve_v1(live_camera_configuration(change))
        assert (
            changed.requested["camera_profile_id"]
            != baseline.requested["camera_profile_id"]
        )


def test_snapshot_requested_view_cannot_mutate_the_canonical_request() -> None:
    snapshot = _resolve_v1(live_camera_configuration())
    original_json = snapshot.to_json()
    requested = snapshot.requested
    requested_profile = requested["requested_profile"]
    assert isinstance(requested_profile, dict)
    requested_profile["width_px"] = 640

    assert snapshot.to_json() == original_json
    stored_profile = snapshot.requested["requested_profile"]
    assert isinstance(stored_profile, dict)
    assert stored_profile["width_px"] == 1920


def test_resolver_uses_one_atomic_configuration_snapshot() -> None:
    settings = ChangingSnapshotConfiguration(
        initial={
            "metadata": {"configuration_revision": 7},
            "vision": {
                "camera": {
                    "resolution": [1280, 720],
                    "video_file_path": "",
                }
            },
        },
        replacement={
            "metadata": {"configuration_revision": 8},
            "vision": {
                "camera": {
                    "resolution": [640, 480],
                    "video_file_path": "",
                }
            },
        },
    )

    snapshot = _resolve_v1(settings)

    assert settings.snapshot_count == 1
    assert snapshot.configuration_revision == 7
    profile = snapshot.requested["requested_profile"]
    assert isinstance(profile, dict)
    assert profile["width_px"] == 1280


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {"vision": {"camera": {"device_id": "assets/demo3.mp4"}}},
            "recorded replay",
        ),
        (
            {"vision": {"camera": {"exposure_mode": "manual", "exposure_value": None}}},
            "manual exposure requires",
        ),
        (
            {"metadata": {"configuration_revision": True}},
            "configuration_revision",
        ),
        (
            {"perception": {"capture": {"image_space": "invented"}}},
            "image_space",
        ),
        (
            {"vision": {"camera": {"backend": "invented"}}},
            "backend is unsupported",
        ),
        (
            {"vision": {"camera": {"backend": "cäméra"}}},
            "bounded ASCII identifier",
        ),
        (
            {"vision": {"camera": {"brightness": 1.1}}},
            "brightness must be in",
        ),
        (
            {"vision": {"camera": {"auto_reconnect": 1}}},
            "auto_reconnect must be a boolean",
        ),
    ],
)
def test_invalid_or_unavailable_values_fail_closed(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(CaptureConfigurationError, match=message):
        _resolve_v1(live_camera_configuration(values))


def test_calibration_bundle_must_match_the_effective_capture_profile() -> None:
    settings = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1000, 500]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                }
            },
        }
    )
    profile = _resolve_v1(settings)
    calibration = json.loads(_PYTHON_CALIBRATION_FIXTURE.read_text(encoding="utf-8"))
    calibration["camera_profile_id"] = profile.requested["camera_profile_id"]
    calibration["calibration_revision"] = 2

    with pytest.raises(CaptureConfigurationError, match="calibration revision"):
        _resolve_v1(
            settings,
            calibration_bundle=calibration,
        )


def test_calibration_bundle_rejects_unknown_schema_properties() -> None:
    settings = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1000, 500]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                }
            },
        }
    )
    profile = _resolve_v1(settings)
    calibration = json.loads(_PYTHON_CALIBRATION_FIXTURE.read_text(encoding="utf-8"))
    calibration["camera_profile_id"] = profile.requested["camera_profile_id"]
    calibration["unexpected"] = True

    with pytest.raises(CaptureConfigurationError, match="violates table-calibration"):
        _resolve_v1(settings, calibration_bundle=calibration)


def test_calibration_bundle_rejects_non_finite_matrix_values() -> None:
    settings = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1000, 500]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                }
            },
        }
    )
    profile = _resolve_v1(settings)
    calibration = json.loads(_PYTHON_CALIBRATION_FIXTURE.read_text(encoding="utf-8"))
    calibration["camera_profile_id"] = profile.requested["camera_profile_id"]
    calibration["pixel_to_table_mm"][0][0] = float("nan")

    with pytest.raises(CaptureConfigurationError, match="finite JSON values"):
        _resolve_v1(settings, calibration_bundle=calibration)


def test_exported_calibration_is_included_with_matching_profile_and_revisions(
    tmp_path: Path,
) -> None:
    settings = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1000, 500]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                }
            },
        }
    )
    request = _resolve_v1(settings)
    profile_id = request.requested["camera_profile_id"]
    assert isinstance(profile_id, str)
    calibrator = CameraCalibrator(str(tmp_path / "calibration-cache"))
    transform = calibrator.calibrate_table_transform(
        np.zeros((400, 800, 3), dtype=np.uint8),
        [(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)],
        (2.0, 1.0),
    )
    assert transform is not None
    calibration = build_table_calibration_v1(
        transform,
        calibration_revision=3,
        camera_profile_id=profile_id,
        rectification_revision=2,
        image_space="rectified",
        source_width_px=800,
        source_height_px=400,
    )

    resolved = _resolve_v1(settings, calibration_bundle=calibration)

    assert resolved.requested["calibration"] == calibration
    assert resolved.requested["camera_profile_id"] == profile_id


def test_calibration_bearing_cross_runtime_fixture_is_schema_valid() -> None:
    fixture = json.loads(
        _CALIBRATED_CAPTURE_SETTINGS_FIXTURE.read_text(encoding="utf-8")
    )
    calibration = json.loads(_PYTHON_CALIBRATION_FIXTURE.read_text(encoding="utf-8"))
    configured = live_camera_configuration(
        {
            "vision": {"camera": {"resolution": [1000, 500]}},
            "perception": {
                "capture": {
                    "image_space": "rectified",
                    "rectification_revision": 2,
                    "calibration_revision": 3,
                }
            },
        }
    )
    profile_request = _resolve_v1(configured)
    calibration["camera_profile_id"] = profile_request.requested["camera_profile_id"]
    resolved_fixture = _resolve_v1(configured, calibration_bundle=calibration)
    settings_schema = json.loads(_CAPTURE_SETTINGS_SCHEMA.read_text(encoding="utf-8"))
    calibration_schema_path = (
        _CROSS_RUNTIME_FIXTURE.parents[2]
        / "calibration"
        / "v1"
        / "table-calibration.schema.json"
    )
    calibration_schema = json.loads(calibration_schema_path.read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        calibration_schema["$id"], Resource.from_contents(calibration_schema)
    )

    jsonschema.Draft202012Validator(settings_schema, registry=registry).validate(
        fixture
    )
    assert resolved_fixture.as_dict() == fixture
    assert fixture["requested"]["calibration"] is not None
    assert (
        fixture["requested"]["calibration"]["camera_profile_id"]
        == fixture["requested"]["camera_profile_id"]
    )


def test_snapshot_dict_is_detached_from_the_stored_request() -> None:
    snapshot = _resolve_v1(live_camera_configuration())
    first = snapshot.to_dict()
    requested = first["requested"]
    assert isinstance(requested, dict)
    requested["device_id"] = "changed"

    assert snapshot.requested["device_id"] == "device-index:0"
