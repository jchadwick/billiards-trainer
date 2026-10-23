"""Tests for the shared configuration validator and atomic store updates."""

import asyncio
import importlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from backend.api.routes.config import get_configuration_schema as get_schema_route
from backend.api.routes.config import (
    get_playing_area_corners,
    import_configuration,
    set_playing_area_corners,
    update_configuration,
)
from backend.config import (
    Config,
    ConfigValidationError,
    get_configuration_schema,
    load_schema_reference,
    validate_configuration,
)


@pytest.fixture
def isolated_config(tmp_path: Path) -> Iterator[Config]:
    """Use a temporary file while retaining the shared singleton identity."""
    singleton = Config._instance
    assert singleton is not None
    singleton_state = singleton.__dict__.copy()
    class_data = Config._config_data
    class_file = Config._config_file
    config_path = tmp_path / "settings.json"

    singleton.__dict__.clear()
    Config._instance = singleton
    Config._config_data = {}
    Config._config_file = None
    Config.set_config_file(config_path)
    try:
        yield singleton
    finally:
        singleton.__dict__.clear()
        singleton.__dict__.update(singleton_state)
        Config._instance = singleton
        Config._config_data = class_data
        Config._config_file = class_file


def test_validator_rejects_unknown_keys_wrong_types_ranges_and_non_finite_values() -> (
    None
):
    errors, _ = validate_configuration(
        {
            "api": {
                "server": {"port": 70000, "reload": 1, "unknown": "x"},
                "rate_limiting": {
                    "store": "filesystem",
                    "trusted_proxies": ["not-a-cidr"],
                },
            },
            "vision": {"camera": {"gain": float("inf")}},
        }
    )

    fields = {error["field"] for error in errors}
    assert "api.server.port" in fields
    assert "api.server.reload" in fields
    assert "api.server.unknown" in fields
    assert "api.rate_limiting.trusted_proxies[]" in fields
    assert "api.rate_limiting.store" in fields
    assert "vision.camera.gain" in fields


def test_validator_allows_documented_dynamic_logger_names() -> None:
    errors, _ = validate_configuration(
        {"system": {"logging": {"log_modules": {"custom.detector": "DEBUG"}}}}
    )

    assert errors == []


def test_retired_python_performance_audit_config_is_not_supported() -> None:
    errors, _ = validate_configuration({"performance_audit": {"enabled": True}})

    assert "performance_audit" in {error["field"] for error in errors}
    assert "performance_audit" not in load_schema_reference()


def test_rust_calibration_control_configuration_is_bounded_and_all_or_none() -> None:
    errors, _ = validate_configuration(
        {
            "perception": {
                "rust_live": {
                    "frame_preview_calibration_control_enabled": True,
                    "frame_preview_enabled": False,
                }
            }
        }
    )
    assert "perception.rust_live.frame_preview_enabled" in {
        error["field"] for error in errors
    }

    errors, _ = validate_configuration(
        {
            "perception": {
                "rust_live": {
                    "frame_preview_calibration_control_enabled": True,
                    "frame_preview_max_header_bytes": 2048,
                    "frame_preview_calibration_max_request_bytes": 2_000_000,
                }
            }
        }
    )
    assert {error["field"] for error in errors} >= {
        "perception.rust_live.frame_preview_max_header_bytes",
        "perception.rust_live.frame_preview_calibration_max_request_bytes",
    }

    errors, _ = validate_configuration(
        {
            "perception": {
                "rust_live": {
                    "frame_preview_calibration_control_enabled": False,
                    "frame_preview_enabled": False,
                }
            }
        }
    )
    assert errors == []


def test_replaceable_websocket_delivery_defaults_are_operational_settings() -> None:
    delivery = load_schema_reference()["api"]["websocket"]["delivery"]
    stream_performance = load_schema_reference()["api"]["stream"]["performance"]
    perception_freshness = load_schema_reference()["perception"]["freshness"]
    active_config = json.loads(
        (Path(__file__).resolve().parents[3] / "config.json").read_text(
            encoding="utf-8"
        )
    )

    assert perception_freshness["maximum_observation_age_seconds"] == 0.5
    assert delivery["max_replaceable_message_age_seconds"] == 0.25
    assert delivery["replaceable_send_timeout_seconds"] == 0.5
    assert stream_performance["maximum_send_backpressure_seconds"] == 0.5
    assert (
        active_config["perception"]["freshness"]["maximum_observation_age_seconds"]
        == 0.5
    )
    active_delivery = active_config["api"]["websocket"]["delivery"]
    assert active_delivery["max_replaceable_message_age_seconds"] == 0.25
    assert active_delivery["replaceable_send_timeout_seconds"] == 0.5
    assert (
        active_config["api"]["stream"]["performance"][
            "maximum_send_backpressure_seconds"
        ]
        == 0.5
    )

    errors, _ = validate_configuration(
        {
            "perception": {"freshness": {"maximum_observation_age_seconds": 0.75}},
            "api": {
                "websocket": {
                    "delivery": {
                        "max_replaceable_message_age_seconds": 0.75,
                        "replaceable_send_timeout_seconds": 1.25,
                    }
                },
                "stream": {"performance": {"maximum_send_backpressure_seconds": 1.25}},
            },
        }
    )
    assert errors == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_replaceable_message_age_seconds", 0),
        ("max_replaceable_message_age_seconds", -0.01),
        ("max_replaceable_message_age_seconds", float("inf")),
        ("max_replaceable_message_age_seconds", float("nan")),
        ("max_replaceable_message_age_seconds", "0.25"),
        ("max_replaceable_message_age_seconds", True),
        ("replaceable_send_timeout_seconds", 0),
        ("replaceable_send_timeout_seconds", -0.01),
        ("replaceable_send_timeout_seconds", float("inf")),
        ("replaceable_send_timeout_seconds", float("nan")),
        ("replaceable_send_timeout_seconds", "0.5"),
        ("replaceable_send_timeout_seconds", False),
    ],
)
def test_replaceable_websocket_delivery_rejects_invalid_limits(
    field: str, value: Any
) -> None:
    errors, _ = validate_configuration(
        {"api": {"websocket": {"delivery": {field: value}}}}
    )

    assert f"api.websocket.delivery.{field}" in {error["field"] for error in errors}


@pytest.mark.parametrize(
    "value",
    [0, -0.01, float("inf"), float("nan"), "0.5", True],
)
def test_perception_observation_age_rejects_invalid_limits(value: Any) -> None:
    errors, _ = validate_configuration(
        {"perception": {"freshness": {"maximum_observation_age_seconds": value}}}
    )
    assert "perception.freshness.maximum_observation_age_seconds" in {
        error["field"] for error in errors
    }


@pytest.mark.parametrize(
    "value",
    [0, -0.01, float("inf"), float("nan"), "0.5", True],
)
def test_video_stream_backpressure_limit_rejects_invalid_values(value: Any) -> None:
    errors, _ = validate_configuration(
        {
            "api": {
                "stream": {"performance": {"maximum_send_backpressure_seconds": value}}
            }
        }
    )

    assert "api.stream.performance.maximum_send_backpressure_seconds" in {
        error["field"] for error in errors
    }


def test_perception_provider_configuration_is_explicit_and_validated() -> None:
    assert load_schema_reference()["perception"]["provider"] == "rust"
    assert load_schema_reference()["perception"]["mode"] == "live"
    assert (
        load_schema_reference()["vision"]["camera"]["control_policy"]
        == "source_defaults"
    )

    errors, _ = validate_configuration({"perception": {"provider": "rust"}})
    assert errors == []

    errors, _ = validate_configuration({"perception": {"provider": "python"}})
    assert {error["field"] for error in errors} == {"perception.provider"}

    for mode in ("live", "recorded_replay"):
        errors, _ = validate_configuration({"perception": {"mode": mode}})
        assert errors == []

    errors, _ = validate_configuration({"perception": {"provider": "automatic"}})
    assert {error["field"] for error in errors} == {"perception.provider"}

    errors, _ = validate_configuration({"perception": {"mode": "automatic"}})
    assert {error["field"] for error in errors} == {"perception.mode"}

    for policy in ("source_defaults", "require"):
        errors, _ = validate_configuration(
            {"vision": {"camera": {"control_policy": policy}}}
        )
        assert errors == []

    errors, _ = validate_configuration(
        {"vision": {"camera": {"control_policy": "automatic"}}}
    )
    assert {error["field"] for error in errors} == {"vision.camera.control_policy"}


def test_capture_device_selector_is_nullable_and_strictly_validated() -> None:
    assert load_schema_reference()["perception"]["capture"]["device_selector"] is None
    selector_schema = get_configuration_schema()["properties"]["perception"][
        "properties"
    ]["capture"]["properties"]["device_selector"]
    assert selector_schema == {
        "anyOf": [
            {"type": "null"},
            {
                "type": "string",
                "pattern": r"^gstreamer-device-v1:[0-9a-f]{64}$",
            },
        ]
    }
    valid_selector = "gstreamer-device-v1:" + ("a" * 64)

    for selector in (None, valid_selector):
        errors, _ = validate_configuration(
            {"perception": {"capture": {"device_selector": selector}}}
        )
        assert errors == []

    for invalid_selector in (
        "",
        "/dev/v4l/by-id/webcam",
        "gstreamer-device-v1:" + ("A" * 64),
        "gstreamer-device-v1:" + ("a" * 63),
        3,
        True,
    ):
        errors, _ = validate_configuration(
            {"perception": {"capture": {"device_selector": invalid_selector}}}
        )
        assert "perception.capture.device_selector" in {
            error["field"] for error in errors
        }


def test_camera_device_id_infers_direct_device_or_mjpeg_uri() -> None:
    assert validate_configuration({"vision": {"camera": {"device_id": 0}}})[0] == []
    assert (
        validate_configuration(
            {"vision": {"camera": {"device_id": "http://127.0.0.1:8080/stream"}}}
        )[0]
        == []
    )
    assert validate_configuration(
        {"vision": {"camera": {"device_id": "https://user:secret@camera/stream"}}}
    )[0]
    assert validate_configuration({"vision": {"camera": {"device_id": "file.mp4"}}})[0]


def test_browser_viewer_url_is_optional_and_must_be_an_anonymous_http_uri() -> None:
    assert load_schema_reference()["api"]["stream"]["viewer_url"] is None
    assert validate_configuration({"api": {"stream": {"viewer_url": None}}})[0] == []
    assert (
        validate_configuration(
            {"api": {"stream": {"viewer_url": "http://192.0.2.10:8080/stream"}}}
        )[0]
        == []
    )
    assert validate_configuration(
        {"api": {"stream": {"viewer_url": "http://user:secret@camera/stream"}}}
    )[0]


def test_capture_frame_memory_limit_is_configured_and_validated() -> None:
    defaults = load_schema_reference()
    assert defaults["perception"]["capture"]["maximum_rgb_frame_bytes"] == 25_165_824
    # Three 24 MiB frame capacities cover callback-held, mailbox-pending, and
    # source-in-progress buffers in the bounded capture pipeline.
    assert (
        defaults["perception"]["capture"]["maximum_aggregate_rgb_frame_bytes"]
        == 75_497_472
    )

    for maximum in (1, 25_165_824, 1_073_741_824):
        errors, _ = validate_configuration(
            {
                "perception": {
                    "capture": {
                        "maximum_rgb_frame_bytes": maximum,
                        "maximum_aggregate_rgb_frame_bytes": max(maximum, 75_497_472),
                    }
                }
            }
        )
        assert errors == []

    for invalid_maximum in (0, -1, True, 1.5, "25165824", 1_073_741_825):
        errors, _ = validate_configuration(
            {"perception": {"capture": {"maximum_rgb_frame_bytes": invalid_maximum}}}
        )
        assert "perception.capture.maximum_rgb_frame_bytes" in {
            error["field"] for error in errors
        }

    for aggregate in (25_165_824, 75_497_472, 1_073_741_824):
        errors, _ = validate_configuration(
            {
                "perception": {
                    "capture": {"maximum_aggregate_rgb_frame_bytes": aggregate}
                }
            }
        )
        assert errors == []

    for invalid_aggregate in (
        0,
        -1,
        True,
        1.5,
        "75497472",
        1_073_741_825,
        25_165_823,
    ):
        errors, _ = validate_configuration(
            {
                "perception": {
                    "capture": {"maximum_aggregate_rgb_frame_bytes": invalid_aggregate}
                }
            }
        )
        assert "perception.capture.maximum_aggregate_rgb_frame_bytes" in {
            error["field"] for error in errors
        }

    # The aggregate limit follows the configured per-frame limit, rather than
    # staying tied to its 72 MiB default when the frame limit is customized.
    errors, _ = validate_configuration(
        {
            "perception": {
                "capture": {
                    "maximum_rgb_frame_bytes": 80_000_000,
                }
            }
        }
    )
    assert "perception.capture.maximum_aggregate_rgb_frame_bytes" in {
        error["field"] for error in errors
    }

    errors, _ = validate_configuration(
        {
            "perception": {
                "capture": {
                    "maximum_rgb_frame_bytes": 80_000_000,
                    "maximum_aggregate_rgb_frame_bytes": 80_000_000,
                }
            }
        }
    )
    assert errors == []

    errors, _ = validate_configuration(
        {
            "perception": {
                "capture": {
                    "maximum_rgb_frame_bytes": 1024,
                    "maximum_aggregate_rgb_frame_bytes": 1024,
                }
            }
        }
    )
    assert errors == []


def test_capture_output_resolution_is_nullable_two_dimensional_u32() -> None:
    defaults = load_schema_reference()
    assert defaults["perception"]["capture"]["output_resolution"] is None

    for accepted_resolution in (None, [640, 480], [1, 1], [4_294_967_295] * 2):
        errors, _ = validate_configuration(
            {"perception": {"capture": {"output_resolution": accepted_resolution}}}
        )
        assert errors == []

    invalid_resolutions: tuple[object, ...] = (
        [],
        [640],
        [640, 480, 3],
        [0, 480],
        [-1, 480],
        [True, 480],
        [640, 480.0],
        [4_294_967_296, 480],
        "640x480",
        {"width": 640, "height": 480},
    )
    for invalid_resolution in invalid_resolutions:
        errors, _ = validate_configuration(
            {"perception": {"capture": {"output_resolution": invalid_resolution}}}
        )
        assert "perception.capture.output_resolution" in {
            error["field"] for error in errors
        }

    schema = get_configuration_schema("perception")["properties"]["capture"]
    assert schema["properties"]["output_resolution"] == {
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4_294_967_295,
                },
                "minItems": 2,
                "maxItems": 2,
            },
        ]
    }


def test_validator_rejects_duplicate_camera_sections_and_accepts_canonical_settings() -> (
    None
):
    errors, warnings = validate_configuration(
        {
            "camera": {"resolution": {"width": 1280, "height": 720}},
            "api": {
                "stream": {
                    "camera": {"fps": 45},
                    "processing": {"enable_table_crop": True},
                }
            },
            "streaming": {"camera": {"fps": 45}},
            "vision": {"camera": {"fps": 45, "resolution": [1280, 720]}},
        }
    )

    fields = {error["field"] for error in errors}
    assert {
        "camera",
        "api.stream.camera",
        "api.stream.processing",
        "streaming.camera",
    } <= fields
    assert not any(field.startswith("vision.camera.") for field in fields)
    assert warnings == []


@pytest.mark.parametrize(
    ("point", "expected_field"),
    [
        ({"x": 100, "y": 100}, "table.marker_dots[].scale"),
        (
            {"x": 100, "y": 100, "scale": [6]},
            "table.marker_dots[].scale",
        ),
        (
            {"x": 100, "y": 100, "scale": [0, 6]},
            "table.marker_dots[].scale[]",
        ),
        (
            {"x": 100, "y": 100, "scale": [-1, 6]},
            "table.marker_dots[].scale[]",
        ),
        (
            {"x": 100, "y": 100, "scale": [True, 6]},
            "table.marker_dots[].scale",
        ),
        (
            {"x": 100, "y": 100, "scale": [float("nan"), 6]},
            "table.marker_dots[].scale[]",
        ),
        (
            {"x": 100, "y": 100, "scale": [6, 6], "unexpected": 1},
            "table.marker_dots[].unexpected",
        ),
    ],
    ids=[
        "missing-scale",
        "scale-length",
        "zero-factor",
        "negative-factor",
        "boolean-factor",
        "non-finite-factor",
        "unknown-point-key",
    ],
)
def test_validator_requires_explicit_positive_finite_coordinate_scale(
    point: dict[str, Any], expected_field: str
) -> None:
    errors, _ = validate_configuration({"table": {"marker_dots": [point]}})

    assert expected_field in {error["field"] for error in errors}


def test_geometry_defaults_scale_640_by_360_calibration_to_canonical_4k() -> None:
    reference = load_schema_reference()
    project_config = json.loads(
        (Path(__file__).resolve().parents[3] / "config.json").read_text(
            encoding="utf-8"
        )
    )

    for table in (reference["table"], project_config["table"]):
        assert table["calibration_resolution_width"] == 640
        assert table["calibration_resolution_height"] == 360
        for field in ("playing_area_corners", "marker_dots"):
            for point in table[field]:
                assert point["scale"] == [6.0, 6.0]


def test_config_store_rejects_unscaled_geometry_without_mutating_or_persisting(
    isolated_config: Config,
) -> None:
    before = isolated_config.get_all()

    with pytest.raises(ConfigValidationError) as error:
        isolated_config.apply_updates(
            {"table": {"marker_dots": [{"x": 100, "y": 100}]}}
        )

    assert "table.marker_dots[].scale" in {item["field"] for item in error.value.errors}
    assert isolated_config.get_all() == before
    assert isolated_config._config_file is not None
    assert not isolated_config._config_file.exists()


@pytest.mark.parametrize("timeout", [0, -0.01, float("inf"), float("nan")])
def test_rate_limit_store_timeout_must_be_positive_and_finite(timeout: float) -> None:
    errors, _ = validate_configuration(
        {"api": {"rate_limiting": {"store_timeout_seconds": timeout}}}
    )

    assert "api.rate_limiting.store_timeout_seconds" in {
        error["field"] for error in errors
    }


@pytest.mark.parametrize(
    ("candidate", "expected_field"),
    [
        (
            {"vision": {"camera": {"resolution": [640]}}},
            "vision.camera.resolution",
        ),
        (
            {"projector": {"display": {"resolution": [0, 1080]}}},
            "projector.display.resolution",
        ),
        (
            {"table": {"playing_area_corners": [{"x": 1} for _ in range(4)]}},
            "table.playing_area_corners[].y",
        ),
        (
            {
                "vision": {
                    "processing": {
                        "phase_scheduler": {"processing_max_dimension": {"AIMING": -1}}
                    }
                }
            },
            "vision.processing.phase_scheduler.processing_max_dimension.AIMING",
        ),
        ({"core": {"physics": {"time_step": 0}}}, "core.physics.time_step"),
    ],
    ids=[
        "resolution-length",
        "projector-resolution-positive-dimensions",
        "corner-required-coordinate",
        "dynamic-map-bound",
        "positive-time-step",
    ],
)
def test_validator_rejects_malformed_or_out_of_bounds_values(
    candidate: dict[str, Any], expected_field: str
) -> None:
    errors, _ = validate_configuration(candidate)

    assert expected_field in {error["field"] for error in errors}


def test_partial_update_checks_cross_field_constraints_against_merged_config(
    isolated_config: Config,
) -> None:
    isolated_config.apply_updates(
        {"vision": {"camera": {"video_start_frame": 30, "video_end_frame": 60}}}
    )

    with pytest.raises(ConfigValidationError) as error:
        isolated_config.apply_updates({"vision": {"camera": {"video_end_frame": 20}}})

    assert error.value.errors[0]["field"] == "vision.camera.video_end_frame"
    assert isolated_config.get("vision.camera.video_end_frame") == 60


def test_config_updates_persist_and_reload_with_leaf_field_names(
    isolated_config: Config, tmp_path: Path
) -> None:
    changed = isolated_config.apply_updates({"vision": {"camera": {"fps": 45}}})

    assert changed == ["vision.camera.fps"]
    persisted = json.loads((tmp_path / "settings.json").read_text())
    assert persisted["vision"] == {"camera": {"fps": 45}}
    assert persisted["metadata"]["configuration_revision"] == 2
    isolated_config.reload()
    assert isolated_config.get("vision.camera.fps") == 45
    assert isolated_config.get("metadata.configuration_revision") == 2


def test_configuration_revision_is_store_managed_and_advances_on_replace(
    isolated_config: Config,
) -> None:
    with pytest.raises(ValueError, match="managed by the configuration store"):
        isolated_config.apply_updates({"metadata": {"configuration_revision": 900}})

    isolated_config.replace_configuration(
        {
            "metadata": {"configuration_revision": 900},
            "vision": {"camera": {"fps": 45}},
        }
    )
    assert isolated_config.get("metadata.configuration_revision") == 2

    isolated_config.replace_configuration(
        {
            "metadata": {"configuration_revision": 2},
            "vision": {"camera": {"fps": 60}},
        }
    )
    assert isolated_config.get("metadata.configuration_revision") == 3

    changed = isolated_config.replace_configuration(
        {
            "metadata": {"configuration_revision": 1},
            "vision": {"camera": {"fps": 60}},
        }
    )
    assert changed == []
    assert isolated_config.get("metadata.configuration_revision") == 3


def test_set_uses_shared_validation_and_atomic_persistence(
    isolated_config: Config, tmp_path: Path
) -> None:
    config_path = tmp_path / "settings.json"

    isolated_config.set("integration.target_fps", 25)

    before = isolated_config.get_all()
    persisted_before = config_path.read_text(encoding="utf-8")
    assert before["integration"] == {"target_fps": 25}
    assert before["metadata"]["configuration_revision"] == 2

    with pytest.raises(ConfigValidationError) as error:
        isolated_config.set("integration.target_fps", 0)

    assert error.value.errors[0]["field"] == "integration.target_fps"
    assert isolated_config.get_all() == before
    assert config_path.read_text(encoding="utf-8") == persisted_before


def test_persistence_failure_keeps_memory_and_file_unchanged(
    isolated_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_config.apply_updates({"vision": {"camera": {"fps": 40}}})
    before = (tmp_path / "settings.json").read_text()
    snapshot = isolated_config.get_all()

    def fail_persistence(data: dict[str, Any]) -> None:
        raise OSError("disk is read-only")

    monkeypatch.setattr(isolated_config, "_save_config", fail_persistence)
    with pytest.raises(OSError, match="read-only"):
        isolated_config.apply_updates({"vision": {"camera": {"fps": 41}}})

    assert isolated_config.get_all() == snapshot
    assert (tmp_path / "settings.json").read_text() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_section_validation_only_and_update_target_the_requested_section(
    isolated_config: Config, tmp_path: Path
) -> None:
    path = tmp_path / "settings.json"
    validated = asyncio.run(
        update_configuration(
            {"camera": {"fps": 45}},
            section="vision",
            validate_only=True,
            force_update=False,
            config_module=isolated_config,
        )
    )
    assert validated.success is True
    assert validated.updated_fields == []
    assert not path.exists()

    applied = asyncio.run(
        update_configuration(
            {"camera": {"fps": 45}},
            section="vision",
            validate_only=False,
            force_update=False,
            config_module=isolated_config,
        )
    )
    assert applied.updated_fields == ["vision.camera.fps"]
    assert applied.restart_required is True
    assert isolated_config.get("vision.camera.fps") == 45


def test_rate_limit_updates_require_restart(isolated_config: Config) -> None:
    response = asyncio.run(
        update_configuration(
            {"api": {"rate_limiting": {"websocket_connections_per_ip": 12}}},
            section=None,
            validate_only=False,
            force_update=False,
            config_module=isolated_config,
        )
    )

    assert response.updated_fields == ["api.rate_limiting.websocket_connections_per_ip"]
    assert response.restart_required is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_fps", 0), ("log_interval_frames", 0)],
    ids=["zero-target-fps", "zero-log-modulo"],
)
def test_invalid_integration_scheduling_values_leave_memory_and_disk_unchanged(
    isolated_config: Config, tmp_path: Path, field: str, value: int
) -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            update_configuration(
                {"integration": {field: value}},
                section=None,
                validate_only=False,
                force_update=True,
                config_module=isolated_config,
            )
        )

    assert error.value.status_code == 422
    assert isolated_config.get_all() == {}
    assert not (tmp_path / "settings.json").exists()


def test_supported_integration_settings_can_be_updated(
    isolated_config: Config,
) -> None:
    changes = {
        "trajectory_quality": "high",
        "max_collision_depth": 4,
        "max_ball_velocity_m_per_s": 12.0,
        "max_position_x": 3.5,
        "max_position_y": 2.5,
        "broadcast_max_retries": 5,
        "broadcast_retry_base_delay_sec": 0.2,
        "circuit_breaker_threshold": 8,
        "circuit_breaker_timeout_sec": 45.0,
        "max_ball_velocity_4k_px_per_s": 13000.0,
        "max_position_x_4k": 4000,
        "max_position_y_4k": 2200,
        "min_ball_confidence": 0.2,
        "min_cue_confidence": 0.1,
    }
    payload = {
        "integration": changes,
        "cue": {"default_force": 8.0},
    }
    response = asyncio.run(
        update_configuration(
            payload,
            section=None,
            validate_only=False,
            force_update=False,
            config_module=isolated_config,
        )
    )

    assert response.success is True
    assert response.restart_required is True
    for key, value in changes.items():
        assert isolated_config.get(f"integration.{key}") == value
    assert isolated_config.get("cue.default_force") == 8.0


@pytest.mark.parametrize(
    "updates",
    [
        {"integration": {"target_fps": 25}},
        {"vision": {"detection": {"yolo_confidence": 0.5}}},
        {"vision": {"processing": {"target_fps": 25}}},
        {"core": {"physics": {"time_step": 0.002}}},
    ],
    ids=[
        "integration",
        "vision-detection",
        "vision-processing",
        "core-physics",
    ],
)
def test_construction_time_settings_report_restart_required(
    isolated_config: Config, updates: dict[str, Any]
) -> None:
    response = asyncio.run(
        update_configuration(
            updates,
            section=None,
            validate_only=False,
            force_update=False,
            config_module=isolated_config,
        )
    )

    assert response.success is True
    assert response.restart_required is True


def test_force_update_does_not_bypass_validation_errors(
    isolated_config: Config,
) -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            update_configuration(
                {"vision": {"camera": {"fps": 121}}},
                section=None,
                validate_only=False,
                force_update=True,
                config_module=isolated_config,
            )
        )

    assert error.value.status_code == 422
    assert isolated_config.get_all() == {}


def test_force_update_can_bypass_warnings_but_does_not_skip_validation(
    isolated_config: Config,
) -> None:
    with pytest.raises(HTTPException) as warning_error:
        asyncio.run(
            update_configuration(
                {"vision": {"camera": {"fps": 90}}},
                section=None,
                validate_only=False,
                force_update=False,
                config_module=isolated_config,
            )
        )
    assert warning_error.value.status_code == 400
    assert isolated_config.get_all() == {}

    response = asyncio.run(
        update_configuration(
            {"vision": {"camera": {"fps": 90}}},
            section=None,
            validate_only=False,
            force_update=True,
            config_module=isolated_config,
        )
    )
    assert response.success is True
    assert response.warnings
    assert isolated_config.get("vision.camera.fps") == 90


def test_import_validation_only_uses_shared_rules_and_does_not_write(
    isolated_config: Config, tmp_path: Path
) -> None:
    upload = UploadFile(
        filename="candidate.json",
        file=io.BytesIO(b'{"vision":{"camera":{"fps":45}}}'),
    )
    response = asyncio.run(
        import_configuration(
            file=upload,
            merge_strategy="merge",
            validate_only=True,
            force_update=False,
            config_module=isolated_config,
        )
    )

    assert response["data"]["is_valid"] is True
    assert isolated_config.get_all() == {}
    assert not (tmp_path / "settings.json").exists()


def test_import_rejects_invalid_configuration_without_persisting(
    isolated_config: Config, tmp_path: Path
) -> None:
    upload = UploadFile(
        filename="invalid.json",
        file=io.BytesIO(b'{"api":{"rate_limiting":{"max_tracked_ips":0}}}'),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            import_configuration(
                file=upload,
                merge_strategy="merge",
                validate_only=False,
                force_update=True,
                config_module=isolated_config,
            )
        )

    assert error.value.status_code == 422
    assert isolated_config.get_all() == {}
    assert not (tmp_path / "settings.json").exists()


def test_partial_replace_import_preserves_schema_and_supported_fields(
    isolated_config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_store = importlib.import_module("backend.config")
    runtime_path = tmp_path / "settings.json"
    schema_defaults = load_schema_reference()
    runtime_path.write_text(json.dumps(schema_defaults), encoding="utf-8")
    isolated_config._config_file = runtime_path
    isolated_config.reload()

    # Reproduce the former arrangement where the active runtime file was also
    # used as the schema template. The bundled schema source must stay separate.
    monkeypatch.setattr(
        config_store, "_DEFAULT_CONFIG_PATH", runtime_path, raising=False
    )
    schema_before = asyncio.run(get_schema_route(section=None))
    upload = UploadFile(
        filename="partial.json",
        file=io.BytesIO(
            b'{"api":{"rate_limiting":{"websocket_connections_per_ip":12}}}'
        ),
    )

    response = asyncio.run(
        import_configuration(
            file=upload,
            merge_strategy="replace",
            validate_only=False,
            force_update=False,
            config_module=isolated_config,
        )
    )

    assert (
        "api.rate_limiting.websocket_connections_per_ip"
        in response["data"]["updated_fields"]
    )
    assert json.loads(runtime_path.read_text(encoding="utf-8")) == {
        "api": {"rate_limiting": {"websocket_connections_per_ip": 12}},
        "metadata": {"configuration_revision": 2},
    }
    assert asyncio.run(get_schema_route(section=None)) == schema_before
    errors, _ = validate_configuration(
        {"vision": {"camera": {"fps": 45}}}, baseline=isolated_config.get_all()
    )
    assert errors == []
    assert "vision" in load_schema_reference()


def test_schema_route_matches_runtime_defaults_and_section_schema() -> None:
    full_schema = asyncio.run(get_schema_route(section=None))
    api_schema = asyncio.run(get_schema_route(section="api"))

    full_api_schema = full_schema["properties"]["api"]
    assert {
        key: value
        for key, value in full_api_schema.items()
        if key != "x-cross-field-rules"
    } == {
        key: value for key, value in api_schema.items() if key != "x-cross-field-rules"
    }
    assert "video" in full_schema["properties"]
    assert (
        full_schema["properties"]["metadata"]["properties"]["configuration_revision"][
            "readOnly"
        ]
        is True
    )
    frame_limit_schema = full_schema["properties"]["perception"]["properties"][
        "capture"
    ]["properties"]["maximum_rgb_frame_bytes"]
    assert frame_limit_schema == {
        "type": "integer",
        "minimum": 1,
        "maximum": 1_073_741_824,
    }
    aggregate_limit_schema = full_schema["properties"]["perception"]["properties"][
        "capture"
    ]["properties"]["maximum_aggregate_rgb_frame_bytes"]
    assert aggregate_limit_schema == {
        "type": "integer",
        "minimum": 1,
        "maximum": 1_073_741_824,
    }
    perception_schema = asyncio.run(get_schema_route(section="perception"))
    assert {
        "fields": [
            "perception.capture.maximum_rgb_frame_bytes",
            "perception.capture.maximum_aggregate_rgb_frame_bytes",
        ],
        "rule": "Aggregate capture frame memory must be at least the per-frame limit",
    } in perception_schema["x-cross-field-rules"]
    assert api_schema["properties"]["websocket"]["properties"]["delivery"][
        "properties"
    ]["send_timeout_seconds"] == {"type": "number", "minimum": 0.001}
    delivery_schema = api_schema["properties"]["websocket"]["properties"]["delivery"][
        "properties"
    ]
    assert delivery_schema["max_replaceable_message_age_seconds"] == {
        "type": "number",
        "minimum": 0.001,
    }
    assert delivery_schema["replaceable_send_timeout_seconds"] == {
        "type": "number",
        "minimum": 0.001,
    }
    assert perception_schema["properties"]["freshness"]["properties"][
        "maximum_observation_age_seconds"
    ] == {"type": "number", "minimum": 0.001}
    assert api_schema["properties"]["stream"]["properties"]["performance"][
        "properties"
    ]["maximum_send_backpressure_seconds"] == {"type": "number", "minimum": 0.001}
    assert api_schema["x-cross-field-rules"]
    rate_schema = api_schema["properties"]["rate_limiting"]["properties"]
    assert rate_schema["store"]["enum"] == ["memory", "redis"]
    assert rate_schema["store_timeout_seconds"] == {
        "type": "number",
        "exclusiveMinimum": 0,
    }
    config_defaults = json.loads(
        (Path(__file__).resolve().parents[3] / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config_defaults["api"]["rate_limiting"]["store_timeout_seconds"] == 2.0
    assert (
        load_schema_reference()["api"]["rate_limiting"]["store_timeout_seconds"] == 2.0
    )
    assert rate_schema["trusted_proxies"]["items"]["format"] == "ipv4-or-ipv6-cidr"
    integration_schema = full_schema["properties"]["integration"]["properties"]
    assert integration_schema["trajectory_quality"]["enum"] == [
        "low",
        "medium",
        "high",
        "ultra",
    ]
    assert integration_schema["max_collision_depth"]["minimum"] == 0
    assert integration_schema["broadcast_max_retries"]["minimum"] == 0
    assert "max_ball_velocity_4k_px_per_s" in integration_schema
    assert "camera" not in full_schema["properties"]
    stream_schema = api_schema["properties"]["stream"]["properties"]
    assert "camera" not in stream_schema
    assert "processing" not in stream_schema
    assert {"quality", "framerate", "resolution", "performance"} <= set(stream_schema)
    assert "camera" not in full_schema["properties"]["streaming"]["properties"]
    assert {"encoding", "performance"} <= set(
        full_schema["properties"]["streaming"]["properties"]
    )
    assert (
        full_schema["properties"]["cue"]["properties"]["default_force"]["maximum"] == 50
    )
    vision_camera_schema = full_schema["properties"]["vision"]["properties"]["camera"][
        "properties"
    ]
    assert vision_camera_schema["resolution"]["minItems"] == 2
    assert vision_camera_schema["resolution"]["maxItems"] == 2
    projector_resolution_schema = full_schema["properties"]["projector"]["properties"][
        "display"
    ]["properties"]["resolution"]
    assert projector_resolution_schema["minItems"] == 2
    assert projector_resolution_schema["maxItems"] == 2
    assert projector_resolution_schema["items"]["minimum"] == 1
    table_corner_schema = full_schema["properties"]["table"]["properties"][
        "playing_area_corners"
    ]
    assert table_corner_schema["items"]["required"] == ["x", "y", "scale"]
    assert table_corner_schema["items"]["properties"]["x"]["type"] == "number"
    assert table_corner_schema["items"]["properties"]["scale"]["minItems"] == 2
    assert table_corner_schema["items"]["properties"]["scale"]["maxItems"] == 2
    assert (
        table_corner_schema["items"]["properties"]["scale"]["items"]["exclusiveMinimum"]
        == 0
    )
    marker_schema = full_schema["properties"]["table"]["properties"]["marker_dots"][
        "items"
    ]
    assert marker_schema["required"] == ["x", "y", "scale"]
    physics_schema = full_schema["properties"]["core"]["properties"]["physics"]
    assert physics_schema["properties"]["time_step"]["exclusiveMinimum"] == 0
    assert get_configuration_schema("api") == api_schema


def test_project_configuration_persists_and_reloads_with_supported_schema(
    isolated_config: Config,
) -> None:
    project_config = json.loads(
        (Path(__file__).resolve().parents[3] / "config.json").read_text(
            encoding="utf-8"
        )
    )
    errors, _ = validate_configuration(project_config)
    assert errors == []

    isolated_config.replace_configuration(project_config)
    isolated_config.reload()

    project_config["metadata"]["configuration_revision"] = 2
    assert isolated_config.get_all() == project_config


def test_table_calibration_endpoint_uses_validated_atomic_store(
    isolated_config: Config,
) -> None:
    payload = json.dumps(
        {
            "corners": [
                {"x": 10.5, "y": 20.25, "scale": [6.0, 6.0]},
                {"x": 600, "y": 20, "scale": [6.0, 6.0]},
                {"x": 600, "y": 300, "scale": [6.0, 6.0]},
                {"x": 10, "y": 300, "scale": [6.0, 6.0]},
            ],
            "marker_dots": [{"x": 100.5, "y": 90.25, "scale": [6.0, 6.0]}],
            "calibration_resolution_width": 640,
            "calibration_resolution_height": 360,
        }
    ).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
    response = asyncio.run(
        set_playing_area_corners(request=request, config_module=isolated_config)
    )

    assert response["success"] is True
    assert isolated_config.get("table.playing_area_corners")[0] == {
        "x": 10.5,
        "y": 20.25,
        "scale": [6.0, 6.0],
    }
    assert isolated_config.get("table.marker_dots") == [
        {"x": 100.5, "y": 90.25, "scale": [6.0, 6.0]}
    ]
    isolated_config.reload()
    loaded = asyncio.run(get_playing_area_corners(config_module=isolated_config))
    assert loaded["corners"][0] == {
        "x": 10.5,
        "y": 20.25,
        "scale": [6.0, 6.0],
    }
    assert loaded["marker_dots"][0]["scale"] == [6.0, 6.0]
    assert loaded["calibration_resolution_width"] == 640
    assert loaded["calibration_resolution_height"] == 360


@pytest.mark.parametrize(
    "point",
    [
        {"x": 10, "y": 20},
        {"x": 10, "y": 20, "scale": [6]},
        {"x": 10, "y": 20, "scale": [0, 6]},
        {"x": 10, "y": 20, "scale": [float("inf"), 6]},
    ],
    ids=[
        "missing-scale",
        "wrong-scale-length",
        "non-positive-scale",
        "non-finite-scale",
    ],
)
def test_playing_area_route_rejects_invalid_scales_without_mutation(
    isolated_config: Config, point: dict[str, Any]
) -> None:
    before = isolated_config.get_all()
    payload = json.dumps(
        {
            "corners": [point]
            + [
                {"x": x, "y": y, "scale": [6.0, 6.0]}
                for x, y in ((600, 20), (600, 300), (10, 300))
            ],
            "calibration_resolution_width": 640,
            "calibration_resolution_height": 360,
        }
    ).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            set_playing_area_corners(request=request, config_module=isolated_config)
        )

    assert error.value.status_code == 422
    assert isolated_config.get_all() == before
    assert isolated_config._config_file is not None
    assert not isolated_config._config_file.exists()


def test_playing_area_route_rejects_scale_inconsistent_with_resolution(
    isolated_config: Config,
) -> None:
    payload = json.dumps(
        {
            "corners": [
                {"x": 10, "y": 20, "scale": [1.0, 1.0]},
                {"x": 600, "y": 20, "scale": [1.0, 1.0]},
                {"x": 600, "y": 300, "scale": [1.0, 1.0]},
                {"x": 10, "y": 300, "scale": [1.0, 1.0]},
            ],
            "calibration_resolution_width": 640,
            "calibration_resolution_height": 360,
        }
    ).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            set_playing_area_corners(request=request, config_module=isolated_config)
        )

    assert error.value.status_code == 422
    assert isolated_config._config_file is not None
    assert not isolated_config._config_file.exists()


def test_playing_area_route_checks_scale_against_existing_resolution(
    isolated_config: Config,
) -> None:
    isolated_config.apply_updates(
        {
            "table": {
                "calibration_resolution_width": 640,
                "calibration_resolution_height": 360,
            }
        }
    )
    before = isolated_config.get_all()
    assert isolated_config._config_file is not None
    before_file = isolated_config._config_file.read_text(encoding="utf-8")
    payload = json.dumps(
        {
            "corners": [
                {"x": 10, "y": 20, "scale": [1.0, 1.0]},
                {"x": 600, "y": 20, "scale": [1.0, 1.0]},
                {"x": 600, "y": 300, "scale": [1.0, 1.0]},
                {"x": 10, "y": 300, "scale": [1.0, 1.0]},
            ]
        }
    ).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            set_playing_area_corners(request=request, config_module=isolated_config)
        )

    assert error.value.status_code == 422
    assert isolated_config._config_file is not None
    assert isolated_config.get_all() == before
    assert isolated_config._config_file.read_text(encoding="utf-8") == before_file


def test_resolution_change_requires_existing_marker_scales_to_match(
    isolated_config: Config,
) -> None:
    isolated_config.apply_updates(
        {
            "table": {
                "calibration_resolution_width": 640,
                "calibration_resolution_height": 360,
                "marker_dots": [{"x": 100, "y": 100, "scale": [6.0, 6.0]}],
            }
        }
    )
    before = isolated_config.get_all()
    assert isolated_config._config_file is not None
    before_file = isolated_config._config_file.read_text(encoding="utf-8")
    payload = json.dumps(
        {
            "corners": [
                {"x": 20, "y": 20, "scale": [2.0, 2.0]},
                {"x": 1800, "y": 20, "scale": [2.0, 2.0]},
                {"x": 1800, "y": 1000, "scale": [2.0, 2.0]},
                {"x": 20, "y": 1000, "scale": [2.0, 2.0]},
            ],
            "calibration_resolution_width": 1920,
            "calibration_resolution_height": 1080,
        }
    ).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            set_playing_area_corners(request=request, config_module=isolated_config)
        )

    assert error.value.status_code == 422
    assert isolated_config.get_all() == before
    assert isolated_config._config_file.read_text(encoding="utf-8") == before_file
