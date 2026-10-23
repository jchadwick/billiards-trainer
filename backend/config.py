"""Thread-safe configuration loading, validation, and persistence.

The bundled ``config_schema_defaults.json`` is an immutable template for
supported runtime keys, types, and defaults. The active ``config.json`` may be
edited or replaced at runtime without changing validation rules or API schema.
"""

import copy
import ipaddress
import json
import logging
import math
import os
import re
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from backend.perception.camera_source import classify_camera_source
from backend.perception.rust_config_authority import (
    RustConfigurationAuthorityClient,
    RustConfigurationAuthorityError,
    RustConfigurationAuthorityProtocolError,
    RustConfigurationAuthorityRejected,
)

logger = logging.getLogger(__name__)

_CONFIG_LOCK = threading.RLock()
_MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
_MAX_U32_PIXEL_DIMENSION = (1 << 32) - 1
_CAPTURE_DEVICE_SELECTOR_PATTERN = re.compile(r"^gstreamer-device-v1:[0-9a-f]{64}$")
_SCHEMA_DEFAULTS_PATH = (
    Path(__file__).resolve().with_name("config_schema_defaults.json")
)

# Logging module names are intentionally user-extensible. Other object keys
# come from the bundled defaults snapshot and are checked strictly.
_DYNAMIC_MAP_PATHS = {"system.logging.log_modules"}
_CROSS_FIELD_RULES: list[dict[str, Any]] = [
    {
        "fields": [
            "vision.camera.video_start_frame",
            "vision.camera.video_end_frame",
        ],
        "rule": "video_start_frame must be less than video_end_frame when both are set",
        "kind": "strictly_increasing",
    },
    {
        "fields": [
            "api.stream.framerate.min_fps",
            "api.stream.framerate.default_fps",
            "api.stream.framerate.max_fps",
        ],
        "rule": "min_fps <= default_fps <= max_fps",
        "kind": "nondecreasing",
    },
    {
        "fields": [
            "api.stream.quality.min_jpeg_quality",
            "api.stream.quality.default_jpeg_quality",
            "api.stream.quality.max_jpeg_quality",
        ],
        "rule": "min_jpeg_quality <= default_jpeg_quality <= max_jpeg_quality",
        "kind": "nondecreasing",
    },
    {
        "fields": [
            "streaming.encoding.min_jpeg_quality",
            "streaming.encoding.default_jpeg_quality",
            "streaming.encoding.max_jpeg_quality",
        ],
        "rule": "min_jpeg_quality <= default_jpeg_quality <= max_jpeg_quality",
        "kind": "nondecreasing",
    },
    {
        "fields": [
            "perception.rust_live.reconnect_initial_delay_ms",
            "perception.rust_live.reconnect_maximum_delay_ms",
        ],
        "rule": "Rust live maximum reconnect delay must be at least the initial delay",
        "kind": "nondecreasing",
    },
    {
        "fields": [
            "perception.capture.maximum_rgb_frame_bytes",
            "perception.capture.maximum_aggregate_rgb_frame_bytes",
        ],
        "rule": "Aggregate capture frame memory must be at least the per-frame limit",
        "kind": "nondecreasing",
    },
]

# These ranges correspond to concrete runtime consumers and public settings.
_NUMERIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "api.server.port": (1, 65535),
    "api.network.port": (1, 65535),
    "development.server.port": (1, 65535),
    "vision.camera.fps": (1, 120),
    "vision.camera.buffer_size": (1, None),
    "vision.camera.gain": (0, None),
    "vision.camera.brightness": (0, 1),
    "vision.camera.contrast": (0, 1),
    "vision.camera.saturation": (0, 1),
    "vision.camera.max_reconnect_attempts": (0, None),
    "vision.camera.reconnect_delay": (0.001, None),
    "vision.camera.stop_timeout": (0.001, None),
    "api.rate_limiting.window_seconds": (1, None),
    "api.rate_limiting.max_tracked_ips": (1, None),
    "api.rate_limiting.websocket_connections_per_ip": (1, None),
    "api.rate_limiting.requests_per_minute": (1, None),
    "api.rate_limiting.burst_size": (1, None),
    "integration.log_interval_frames": (1, None),
    "integration.max_collision_depth": (0, None),
    "integration.broadcast_max_retries": (0, None),
    "integration.circuit_breaker_threshold": (1, None),
    "integration.max_position_x_4k": (1, None),
    "integration.max_position_y_4k": (1, None),
    "integration.min_ball_confidence": (0, 1),
    "integration.min_cue_confidence": (0, 1),
    "cue.default_force": (0, 50),
    "vision.camera.resolution[]": (1, None),
    "projector.display.resolution[]": (1, None),
    "vision.detection.yolo_confidence": (0, 1),
    "vision.detection.yolo_nms_threshold": (0, 1),
    "perception.rust_live.image_size": (32, None),
    "perception.capture.maximum_rgb_frame_bytes": (1, 1_073_741_824),
    "perception.capture.maximum_aggregate_rgb_frame_bytes": (1, 1_073_741_824),
    "perception.capture.maximum_frame_age_ms": (1, _MAX_SAFE_JSON_INTEGER),
    "perception.capture.rectification_revision": (0, _MAX_SAFE_JSON_INTEGER),
    "perception.capture.calibration_revision": (0, _MAX_SAFE_JSON_INTEGER),
    "metadata.configuration_revision": (1, _MAX_SAFE_JSON_INTEGER),
    "perception.rust_live.confidence": (0, 1),
    "perception.rust_live.iou": (0, 1),
    "game_session.maximum_request_bytes": (128, 65_536),
    "perception.rust_live.warmup_runs": (0, None),
    "perception.rust_live.ball_class_id": (0, None),
    "perception.rust_live.cue_class_id": (0, None),
    "perception.rust_live.max_detections": (1, 4294967295),
    "perception.rust_live.association_prediction_horizon_ms": (1, None),
    "perception.rust_live.lost_after_ms": (1, None),
    "perception.rust_live.maximum_track_age_ms": (1, None),
    "perception.rust_live.maximum_active_tracks": (1, None),
    "perception.rust_live.recent_observation_capacity": (1, None),
    "perception.rust_live.port": (0, 65535),
    "perception.rust_live.maximum_message_bytes": (1, 4294967295),
    "perception.rust_live.capture_settings_maximum_bytes": (1, None),
    "perception.rust_live.poll_interval_ms": (1, None),
    "perception.rust_live.connect_timeout_ms": (1, None),
    "perception.rust_live.write_timeout_ms": (1, None),
    "perception.rust_live.reconnect_initial_delay_ms": (0, None),
    "perception.rust_live.reconnect_maximum_delay_ms": (0, None),
    "perception.rust_live.output_tail_capacity_bytes": (1, None),
    "perception.rust_live.public_state_delivery_schema_version": (1, 1),
    "perception.rust_live.frame_preview_port": (0, 65535),
    "perception.rust_live.frame_preview_max_header_bytes": (128, 1048576),
    "perception.rust_live.frame_preview_max_jpeg_bytes": (1, 16777216),
    "perception.rust_live.frame_preview_max_concurrent_encodes": (1, 4),
    "perception.rust_live.frame_preview_jpeg_quality": (1, 100),
    "perception.rust_live.frame_preview_request_timeout_ms": (1, 60000),
    "perception.rust_live.frame_preview_write_timeout_ms": (1, 60000),
    "perception.rust_live.frame_preview_calibration_store_lock_timeout_ms": (
        1,
        60000,
    ),
    "perception.rust_live.frame_preview_calibration_store_max_file_bytes": (
        1,
        16777216,
    ),
    "perception.rust_live.frame_preview_calibration_max_request_bytes": (
        1024,
        1048576,
    ),
    "perception.rust_live.capture_helper_ready_record_max_bytes": (1, 4294967295),
    "perception.rust_live.capture_helper_frame_header_max_bytes": (1, 4294967295),
    "perception.rust_live.capture_helper_slow_reader_delay_ms": (0, None),
    "perception.rust_live.capture_helper_pull_timeout_ms": (1, None),
    "perception.rust_live.capture_helper_readiness_timeout_ms": (1, None),
    "perception.rust_live.capture_helper_child_exit_timeout_ms": (1, None),
    "perception.rust_live.capture_helper_child_exit_poll_interval_ms": (1, None),
    "api.stream.quality.default_jpeg_quality": (1, 100),
    "api.stream.quality.websocket_jpeg_quality": (1, 100),
    "api.stream.quality.single_frame_jpeg_quality": (1, 100),
    "api.stream.quality.min_jpeg_quality": (1, 100),
    "api.stream.quality.max_jpeg_quality": (1, 100),
    "api.stream.performance.maximum_send_backpressure_seconds": (0.001, None),
    "streaming.encoding.default_jpeg_quality": (1, 100),
    "streaming.encoding.min_jpeg_quality": (1, 100),
    "streaming.encoding.max_jpeg_quality": (1, 100),
    "vision.camera.video_start_frame": (0, None),
    "vision.camera.video_end_frame": (0, None),
    "table.calibration_resolution_width": (1, None),
    "table.calibration_resolution_height": (1, None),
    "table.playing_area_corners[].x": (0, None),
    "table.playing_area_corners[].y": (0, None),
    "table.marker_dots[].x": (0, None),
    "table.marker_dots[].y": (0, None),
    "api.websocket.delivery.replaceable_queue_capacity": (1, None),
    "api.websocket.delivery.control_queue_capacity": (1, None),
    "api.websocket.delivery.send_timeout_seconds": (0.001, None),
    # These are operational defaults for best-effort freshness. They are not
    # qualified end-to-end latency targets; deployments may tune them using
    # workload measurements.
    "api.websocket.delivery.max_replaceable_message_age_seconds": (0.001, None),
    "api.websocket.delivery.replaceable_send_timeout_seconds": (0.001, None),
    "api.websocket.delivery.aggregation_buffer_capacity": (1, None),
    "perception.freshness.maximum_observation_age_seconds": (0.001, None),
}
_EXCLUSIVE_MINIMUMS: dict[str, float] = {
    "core.physics.time_step": 0,
    "core.prediction.outcome_predictor.simulation.time_step": 0,
    "api.rate_limiting.store_timeout_seconds": 0,
    "integration.target_fps": 0,
    "integration.error_retry_delay_sec": 0,
    "integration.shot_speed_estimate_m_per_s": 0,
    "integration.max_ball_velocity_m_per_s": 0,
    "integration.max_position_x": 0,
    "integration.max_position_y": 0,
    "integration.broadcast_retry_base_delay_sec": 0,
    "integration.circuit_breaker_timeout_sec": 0,
    "integration.max_ball_velocity_4k_px_per_s": 0,
    "perception.rust_live.association_gate_px": 0,
    "perception.rust_live.read_timeout_seconds": 0,
    "perception.rust_live.source_silence_timeout_seconds": 0,
    "perception.rust_live.startup_timeout_seconds": 0,
    "perception.rust_live.stop_timeout_seconds": 0,
    "perception.rust_live.graceful_stop_timeout_seconds": 0,
}
_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "vision.camera.backend": (
        "auto",
        "v4l2",
        "dshow",
        "gstreamer",
        "opencv",
        "kinect2",
    ),
    "vision.camera.exposure_mode": ("auto", "manual"),
    "vision.camera.control_policy": ("source_defaults", "require"),
    "api.rate_limiting.store": ("memory", "redis"),
    "integration.trajectory_quality": ("low", "medium", "high", "ultra"),
    "perception.provider": ("rust",),
    "perception.rust_live.provider": ("cpu", "auto", "openvino"),
    "perception.capture.image_space": ("raw", "rectified"),
    "perception.capture.timestamp_semantics": ("read_completion_only",),
}
_STRICTLY_POSITIVE_DYNAMIC_PREFIXES = (
    "vision.processing.phase_scheduler.processing_max_dimension.",
)
_TWO_DIMENSIONAL_RESOLUTION_PATHS = {
    "vision.camera.resolution",
    "projector.display.resolution",
    "perception.capture.output_resolution",
}
_NULLABLE_RESOLUTION_PATHS = {"perception.capture.output_resolution"}
_TWO_DIMENSIONAL_SCALE_PATHS = {
    "table.playing_area_corners[].scale",
    "table.marker_dots[].scale",
}
_SCALE_FACTOR_PATHS = {f"{path}[]" for path in _TWO_DIMENSIONAL_SCALE_PATHS}
_EXCLUSIVE_MINIMUMS.update(dict.fromkeys(_SCALE_FACTOR_PATHS, 0))


class ConfigValidationError(ValueError):
    """Raised when a proposed configuration fails runtime validation."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__("Configuration validation failed")


class ConfigurationAuthorityConfigurationError(ValueError):
    """The explicitly selected configuration authority is not configured."""


class ConfigurationRevisionConflict(RuntimeError):
    """A Rust mutation was based on a revision superseded by another writer."""

    def __init__(self, message: str, *, current_revision: int | None) -> None:
        self.current_revision = current_revision
        super().__init__(message)


def _json_copy(value: Any) -> Any:
    """Return an isolated JSON-compatible copy and reject non-finite numbers."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Configuration must contain finite JSON values: {exc}"
        ) from exc


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge object updates recursively while replacing scalar and array values."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _error(path: str, message: str, value: Any, expected: str) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value):
        value = repr(value)
    return {
        "field": path or "config_data",
        "message": message,
        "current_value": value,
        "expected_type": expected,
    }


def _path_with_item(path: str) -> str:
    return path + "[]"


def _matches_expected_type(value: Any, example: Any, path: str) -> bool:
    if path.endswith(("[].x", "[].y")) and path.startswith(
        ("table.playing_area_corners", "table.marker_dots")
    ):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if path in {
        "vision.camera.exposure_value",
        "vision.camera.video_end_frame",
        "api.stream.viewer_url",
        *_NULLABLE_RESOLUTION_PATHS,
    } or path.startswith("vision.processing.phase_scheduler.processing_max_dimension."):
        return value is None or _matches_non_null_type(value, path)
    return _matches_non_null_type(value, path, example)


def _matches_non_null_type(value: Any, path: str, example: Any = None) -> bool:
    if path == "vision.camera.device_id":
        try:
            classify_camera_source(value)
        except ValueError:
            return False
        return True
    if path == "api.stream.viewer_url":
        if not isinstance(value, str):
            return False
        try:
            source = classify_camera_source(value)
        except ValueError:
            return False
        return source.kind == "ustreamer_mjpeg"
    if path in _NULLABLE_RESOLUTION_PATHS:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 1 <= item <= _MAX_U32_PIXEL_DIMENSION
                for item in value
            )
        )
    if path in _TWO_DIMENSIONAL_SCALE_PATHS:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
        )
    if path in _TWO_DIMENSIONAL_RESOLUTION_PATHS:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in value
            )
        )
    if example is None:
        if path == "perception.capture.device_selector":
            return value is None or isinstance(value, str)
        if path.endswith(".exposure_value"):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if path.endswith(".video_end_frame") or path.endswith(
            "processing_max_dimension"
        ):
            return isinstance(value, int) and not isinstance(value, bool)
        if path.startswith(
            "vision.processing.phase_scheduler.processing_max_dimension."
        ):
            return isinstance(value, int) and not isinstance(value, bool)
        return True
    if isinstance(example, bool):
        return isinstance(value, bool)
    if isinstance(example, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(example, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(example, str):
        return isinstance(value, str)
    if isinstance(example, list):
        return isinstance(value, list)
    if isinstance(example, dict):
        return isinstance(value, dict)
    return value is None


def _validate_value(
    value: Any,
    example: Any,
    path: str,
    errors: list[dict[str, Any]],
) -> None:
    if not _matches_expected_type(value, example, path):
        expected = (
            "array of two positive u32 pixel dimensions or null"
            if path in _NULLABLE_RESOLUTION_PATHS
            else (
                "array of two positive integers or null"
                if path.endswith(".resolution")
                else type(example).__name__ if example is not None else "number or null"
            )
        )
        errors.append(_error(path, f"Expected {expected}", value, expected))
        return

    if value is None:
        return

    if isinstance(value, dict):
        if not isinstance(example, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                _validate_value(child, None, child_path, errors)
            return
        dynamic = path in _DYNAMIC_MAP_PATHS or (not example and path)
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key not in example and not dynamic:
                errors.append(
                    _error(child_path, "Unknown configuration key", child, "known key")
                )
                continue
            child_example = (
                example.get(key) if key in example else _dynamic_example(example)
            )
            _validate_value(child, child_example, child_path, errors)
        return

    if isinstance(value, list):
        if path in _TWO_DIMENSIONAL_SCALE_PATHS and len(value) != 2:
            errors.append(
                _error(
                    path,
                    "Coordinate scale must contain exactly two factors",
                    value,
                    "array[2] of positive finite numbers",
                )
            )
            return
        if path == "table.playing_area_corners" and len(value) != 4:
            errors.append(
                _error(
                    path, "Exactly four table corners are required", value, "array[4]"
                )
            )
        if not value:
            return
        if path == "api.rate_limiting.trusted_proxies":
            for proxy in value:
                if not isinstance(proxy, str):
                    errors.append(
                        _error(
                            path + "[]",
                            "Trusted proxy entries must be CIDR strings",
                            proxy,
                            "CIDR string",
                        )
                    )
                    continue
                try:
                    ipaddress.ip_network(proxy, strict=False)
                except ValueError:
                    errors.append(
                        _error(
                            path + "[]",
                            "Trusted proxy entry must be a valid IPv4 or IPv6 CIDR",
                            proxy,
                            "CIDR string",
                        )
                    )
            return
        item_example = (
            next((item for item in example if item is not None), None)
            if isinstance(example, list)
            else None
        )
        for item in value:
            item_path = _path_with_item(path)
            if isinstance(item, dict) and isinstance(item_example, dict):
                for key, expected_value in item_example.items():
                    if key not in item:
                        errors.append(
                            _error(
                                f"{item_path}.{key}",
                                "Missing required property",
                                None,
                                type(expected_value).__name__,
                            )
                        )
                _validate_value(item, item_example, item_path, errors)
            elif item_example is not None:
                _validate_value(item, item_example, item_path, errors)
            elif not _is_json_value(item):
                errors.append(
                    _error(
                        item_path, "Value is not JSON-compatible", item, "JSON value"
                    )
                )
        return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            errors.append(_error(path, "Number must be finite", value, "finite number"))
            return
        range_spec = _numeric_range_for_path(path)
        if range_spec:
            minimum, maximum = range_spec
            if minimum is not None and value < minimum:
                errors.append(
                    _error(
                        path,
                        f"Value must be at least {minimum}",
                        value,
                        "number in range",
                    )
                )
            if maximum is not None and value > maximum:
                errors.append(
                    _error(
                        path,
                        f"Value must be at most {maximum}",
                        value,
                        "number in range",
                    )
                )
        exclusive_minimum = _EXCLUSIVE_MINIMUMS.get(path)
        if exclusive_minimum is not None and value <= exclusive_minimum:
            errors.append(
                _error(
                    path,
                    f"Value must be greater than {exclusive_minimum}",
                    value,
                    "number greater than exclusive minimum",
                )
            )

    if (
        path == "perception.capture.device_selector"
        and value is not None
        and (
            not isinstance(value, str)
            or not _CAPTURE_DEVICE_SELECTOR_PATTERN.fullmatch(value)
        )
    ):
        errors.append(
            _error(
                path,
                "Capture device selector must be empty or a versioned GStreamer selector",
                value,
                "empty string or gstreamer-device-v1 SHA-256 selector",
            )
        )

    allowed_values = _ENUM_VALUES.get(path)
    if allowed_values is not None and value not in allowed_values:
        errors.append(
            _error(
                path,
                f"Value must be one of {', '.join(allowed_values)}",
                value,
                "|".join(allowed_values),
            )
        )


def _dynamic_example(example: dict[str, Any]) -> Any:
    """Use the first configured dynamic-map value as its type template."""
    return next(iter(example.values()), None)


def _numeric_range_for_path(
    path: str,
) -> tuple[float | None, float | None] | None:
    """Return explicit bounds, including bounds shared by dynamic enum maps."""
    if path.startswith(_STRICTLY_POSITIVE_DYNAMIC_PREFIXES):
        return (1, None)
    return _NUMERIC_RANGES.get(path)


def _is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def validate_configuration(
    candidate: dict[str, Any],
    *,
    baseline: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate known keys, types, numeric bounds, and related settings.

    ``candidate`` may contain only a partial update. Cross-field validation is
    applied to the complete merged configuration passed by the store, and the
    merged snapshot is checked against the same supported-key schema.
    """
    rust_config = globals().get("config")
    if isinstance(rust_config, Config) and rust_config.authority_mode == "rust":
        result = rust_config.validate_configuration(
            candidate, merge=baseline is not None
        )
        return result["validation_errors"], result["warnings"]

    errors: list[dict[str, Any]] = []
    reference = load_schema_reference()
    baseline_with_defaults = _deep_merge(reference, baseline or {})
    combined = _deep_merge(baseline_with_defaults, candidate)
    schema_candidate = combined if baseline is not None else candidate
    _validate_value(schema_candidate, reference, "", errors)

    for rule in _CROSS_FIELD_RULES:
        values = [_dig(combined, path) for path in rule["fields"]]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            continue
        if rule["kind"] == "strictly_increasing":
            valid = all(left < right for left, right in zip(values, values[1:]))
        else:
            valid = all(left <= right for left, right in zip(values, values[1:]))
        if not valid:
            errors.append(
                _error(
                    rule["fields"][-1],
                    rule["rule"],
                    values[-1],
                    "values satisfying cross-field rule",
                )
            )

    live_image_size = _dig(combined, "perception.rust_live.image_size")
    if (
        isinstance(live_image_size, int)
        and not isinstance(live_image_size, bool)
        and live_image_size % 32 != 0
    ):
        errors.append(
            _error(
                "perception.rust_live.image_size",
                "Rust live image_size must be divisible by 32",
                live_image_size,
                "integer multiple of 32",
            )
        )

    rust_live = _dig(combined, "perception.rust_live")
    if isinstance(rust_live, dict):
        control_enabled = rust_live.get("frame_preview_calibration_control_enabled")
        if (
            control_enabled is True
            and rust_live.get("frame_preview_enabled") is not True
        ):
            errors.append(
                _error(
                    "perception.rust_live.frame_preview_enabled",
                    "Calibration control requires the Rust frame-preview listener",
                    rust_live.get("frame_preview_enabled"),
                    "true when calibration control is enabled",
                )
            )
        header_limit = rust_live.get("frame_preview_max_header_bytes")
        if (
            control_enabled is True
            and isinstance(header_limit, int)
            and not isinstance(header_limit, bool)
            and header_limit < 4096
        ):
            errors.append(
                _error(
                    "perception.rust_live.frame_preview_max_header_bytes",
                    "Calibration control requires a response limit of at least 4096 bytes",
                    header_limit,
                    "integer >= 4096",
                )
            )
        store_path = rust_live.get("frame_preview_calibration_store_path")
        if (
            control_enabled is True
            and isinstance(store_path, str)
            and (not store_path.strip() or len(store_path) > 1024)
        ):
            errors.append(
                _error(
                    "perception.rust_live.frame_preview_calibration_store_path",
                    "Calibration store path must be a non-empty path of at most 1024 characters",
                    store_path,
                    "non-empty path <= 1024 characters",
                )
            )

    warnings: list[str] = []
    for path in ("vision.camera.fps",):
        fps = _dig(combined, path)
        if isinstance(fps, (int, float)) and not isinstance(fps, bool) and fps > 60:
            warnings.append(f"{path}={fps} may increase CPU usage")
    return errors, warnings


def _dig(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for component in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(component)
    return value


def _schema_from_example(example: Any, path: str = "") -> dict[str, Any]:
    if isinstance(example, dict):
        properties = {
            key: _schema_from_example(value, f"{path}.{key}" if path else key)
            for key, value in example.items()
        }
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if path.endswith("[]"):
            schema["required"] = list(example)
        if path in _DYNAMIC_MAP_PATHS:
            dynamic_example = next(iter(example.values()), None)
            schema["additionalProperties"] = _schema_from_example(
                dynamic_example, f"{path}.*"
            )
        elif not example:
            schema["additionalProperties"] = True
        else:
            schema["additionalProperties"] = False
        return schema
    if isinstance(example, list):
        if path == "api.rate_limiting.trusted_proxies":
            return {
                "type": "array",
                "items": {"type": "string", "format": "ipv4-or-ipv6-cidr"},
            }
        item_example = next((item for item in example if item is not None), None)
        schema = {
            "type": "array",
            "items": (
                _schema_from_example(item_example, _path_with_item(path))
                if item_example is not None
                else {}
            ),
        }
        if path == "table.playing_area_corners":
            schema.update({"minItems": 4, "maxItems": 4})
        if path.endswith(".resolution"):
            schema["minItems"] = schema["maxItems"] = 2
        if path in _TWO_DIMENSIONAL_SCALE_PATHS:
            schema["minItems"] = schema["maxItems"] = 2
        return schema
    if path == "vision.camera.device_id":
        return {
            "oneOf": [
                {"type": "integer", "minimum": 0},
                {
                    "type": "string",
                    "maxLength": 2048,
                    "format": "uri",
                    "description": "HTTP(S) MJPEG URI or stable GStreamer camera selector",
                },
            ]
        }
    if example is None:
        if path == "api.stream.viewer_url":
            return {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "string",
                        "maxLength": 2048,
                        "format": "uri",
                    },
                ]
            }
        if path == "perception.capture.device_selector":
            return {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "string",
                        "pattern": r"^gstreamer-device-v1:[0-9a-f]{64}$",
                    },
                ]
            }
        if path in _NULLABLE_RESOLUTION_PATHS:
            return {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_U32_PIXEL_DIMENSION,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                    },
                ]
            }
        if path in {"vision.camera.exposure_value"}:
            return {"type": ["number", "null"]}
        if path == "vision.camera.video_end_frame":
            return {"type": ["integer", "null"], "minimum": 0}
        if path.endswith(".resolution"):
            return {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                ]
            }
        if path.startswith(
            "vision.processing.phase_scheduler.processing_max_dimension."
        ):
            return {"type": ["integer", "null"], "minimum": 1}
        return {}
    if isinstance(example, bool):
        return {"type": "boolean"}
    if path.endswith(("[].x", "[].y")) and path.startswith(
        ("table.playing_area_corners", "table.marker_dots")
    ):
        schema = {"type": "number"}
        minimum, maximum = _NUMERIC_RANGES[path]
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum
        return schema
    if isinstance(example, int):
        schema = {"type": "integer"}
    elif isinstance(example, float):
        schema = {"type": "number"}
    elif isinstance(example, str):
        schema = {"type": "string"}
        if path in _ENUM_VALUES:
            schema["enum"] = list(_ENUM_VALUES[path])
    else:
        return {}
    if path in _NUMERIC_RANGES:
        minimum, maximum = _NUMERIC_RANGES[path]
        if minimum is not None:
            schema["minimum"] = minimum
        if maximum is not None:
            schema["maximum"] = maximum
    elif path.startswith(_STRICTLY_POSITIVE_DYNAMIC_PREFIXES):
        schema["minimum"] = 1
    if path in _EXCLUSIVE_MINIMUMS:
        schema["exclusiveMinimum"] = _EXCLUSIVE_MINIMUMS[path]
    if path == "metadata.configuration_revision":
        schema["readOnly"] = True
    return schema


@lru_cache(maxsize=1)
def _load_schema_reference() -> dict[str, Any]:
    """Load the immutable schema/default snapshot once per process."""
    try:
        with _SCHEMA_DEFAULTS_PATH.open(encoding="utf-8") as file:
            reference = json.load(file)
        if isinstance(reference, dict):
            return reference
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Unable to load configuration schema defaults: %s", exc)
    return {}


def load_schema_reference() -> dict[str, Any]:
    """Return an isolated copy of the bundled supported-configuration defaults."""
    rust_config = globals().get("config")
    if isinstance(rust_config, Config) and rust_config.authority_mode == "rust":
        return rust_config.get_defaults()
    return copy.deepcopy(_load_schema_reference())


def get_configuration_schema(section: Optional[str] = None) -> dict[str, Any]:
    """Build the JSON Schema exposed by the configuration API."""
    rust_config = globals().get("config")
    if isinstance(rust_config, Config) and rust_config.authority_mode == "rust":
        return rust_config.get_schema(section)
    schema = _schema_from_example(load_schema_reference())
    if section is None:
        schema["x-cross-field-rules"] = copy.deepcopy(_CROSS_FIELD_RULES)
        return schema
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    if section not in properties:
        raise KeyError(section)
    section_schema = properties[section]
    relevant_rules = [
        {key: value for key, value in rule.items() if key in {"fields", "rule"}}
        for rule in _CROSS_FIELD_RULES
        if any(field.startswith(section + ".") for field in rule["fields"])
    ]
    if relevant_rules:
        section_schema["x-cross-field-rules"] = copy.deepcopy(relevant_rules)
    return section_schema


class Config:
    """Configuration facade with a transitional Python or Rust authority.

    Python remains the default authority until every configuration route is
    qualified against the Rust protocol. Selecting Rust is explicit and fails
    closed; Rust mode never reads, validates, or writes the Python config file.
    """

    _instance: Optional["Config"] = None
    _config_data: dict[str, Any] = {}
    _config_file: Optional[Path] = None
    _authority_mode: str | None = None
    _rust_client: RustConfigurationAuthorityClient | None = None
    _rust_revision: int | None = None
    _rust_defaults: dict[str, Any] | None = None
    _rust_schema: dict[str, Any] | None = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def authority_mode(self) -> str:
        """Return the active authority, resolving the explicit selector once."""
        if self._authority_mode is None:
            self._authority_mode = self._selected_authority_mode()
        return self._authority_mode

    @property
    def revision(self) -> int | None:
        """Return the cached Rust revision, or ``None`` for Python mode."""
        return self._rust_revision if self.authority_mode == "rust" else None

    @staticmethod
    def _selected_authority_mode() -> str:
        configured = os.getenv("BILLIARDS_CONFIG_AUTHORITY", "").strip().lower()
        if not configured or configured == "python":
            return "python"
        if configured != "rust":
            raise ConfigurationAuthorityConfigurationError(
                "BILLIARDS_CONFIG_AUTHORITY must be 'python' or 'rust'"
            )
        return configured

    @classmethod
    def bootstrap_from_environment(
        cls,
        config_path: str | Path | None = None,
        *,
        default_config_path: str | Path | None = None,
    ) -> "Config":
        """Prepare the selected authority before importing config consumers.

        In Python compatibility mode, no path preserves the legacy lazy load
        unless a default is supplied by the executable entry point. Rust mode
        requires a caller-provided path or ``BILLIARDS_CONFIG_PATH``.
        """
        authority_mode = cls._selected_authority_mode()
        instance = cls()
        if authority_mode == "rust":
            # Mark Rust as selected before validating its bootstrap inputs. If
            # initialization fails, later reads must not expose a prior Python
            # snapshot from this process.
            with _CONFIG_LOCK:
                instance._authority_mode = "rust"
                instance._config_data = {}
                instance._config_file = None
                instance._rust_client = None
                instance._rust_revision = None
                instance._rust_defaults = None
                instance._rust_schema = None
        selected_path = config_path or os.getenv("BILLIARDS_CONFIG_PATH")
        if selected_path is None:
            selected_path = default_config_path
        if authority_mode == "rust" and selected_path is None:
            raise ConfigurationAuthorityConfigurationError(
                "Rust configuration authority requires an explicit config path "
                "through --config or BILLIARDS_CONFIG_PATH"
            )
        if selected_path is not None:
            cls.set_config_file(selected_path)
        else:
            instance._authority_mode = authority_mode
        return cls()

    @classmethod
    def set_config_file(cls, config_path: str | Path) -> None:
        if cls._instance is None:
            cls._instance = cls()
        with _CONFIG_LOCK:
            instance = cls._instance
            selected_path = Path(config_path).expanduser().resolve()
            authority_mode = cls._selected_authority_mode()
            if authority_mode == "rust":
                # Clear the prior cache before validating the executable so a
                # bad Rust selection cannot fall through to Python data.
                instance._authority_mode = "rust"
                instance._config_file = selected_path
                instance._config_data = {}
                instance._rust_client = None
                instance._rust_revision = None
                instance._rust_defaults = None
                instance._rust_schema = None
                executable_value = os.getenv(
                    "BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE", ""
                ).strip()
                if not executable_value:
                    raise ConfigurationAuthorityConfigurationError(
                        "Rust configuration authority requires "
                        "BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE"
                    )
                executable = Path(executable_value).expanduser()
                if not executable.is_absolute():
                    raise ConfigurationAuthorityConfigurationError(
                        "BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE must be an absolute path"
                    )
                if not executable.is_file() or not os.access(executable, os.X_OK):
                    raise ConfigurationAuthorityConfigurationError(
                        "BILLIARDS_CONFIG_AUTHORITY_EXECUTABLE must name an executable file"
                    )
                instance._rust_client = RustConfigurationAuthorityClient(
                    executable, selected_path
                )
                instance._load_rust_snapshot()
                return

            same_file = (
                instance._config_file is not None
                and instance._config_file.resolve() == selected_path
            )
            minimum_revision = (
                _configuration_revision(
                    cls._config_data, default=_schema_configuration_revision()
                )
                if same_file
                else None
            )
            instance._config_file = selected_path
            instance._authority_mode = "python"
            instance._rust_client = None
            instance._rust_revision = None
            instance._rust_defaults = None
            instance._rust_schema = None
            instance._load_config(
                minimum_revision=minimum_revision,
                preserve_on_error=same_file,
            )

    def _load_rust_snapshot(self) -> None:
        """Load one complete Rust snapshot before publishing it to readers."""
        response = self._rust_call("snapshot")
        document = response.get("configuration")
        revision = response.get("revision")
        if not isinstance(document, dict) or type(revision) is not int or revision < 1:
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration snapshot is missing a valid revision or document",
                response=response,
            )
        metadata = document.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("configuration_revision") != revision
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration snapshot revision does not match its metadata",
                response=response,
            )
        self._config_data = copy.deepcopy(document)
        self._rust_revision = revision

    def _rust_call(self, operation: str, **parameters: Any) -> dict[str, Any]:
        """Call the selected authority and return its validated response body."""
        if self.authority_mode != "rust" or self._rust_client is None:
            raise ConfigurationAuthorityConfigurationError(
                "Rust configuration authority has not been initialized"
            )
        request = {"protocol_version": 1, "operation": operation, **parameters}
        try:
            return self._rust_client.call(request).body
        except RustConfigurationAuthorityRejected as error:
            if error.code == "revision_conflict":
                response = error.response or {}
                current_revision = response.get("current_revision")
                raise ConfigurationRevisionConflict(
                    error.message,
                    current_revision=(
                        current_revision if type(current_revision) is int else None
                    ),
                ) from error
            raise

    def _accept_rust_mutation(self, response: dict[str, Any]) -> list[str]:
        """Publish only a complete successful Rust mutation response to readers."""
        document = response.get("configuration")
        revision = response.get("revision")
        updated_fields = response.get("updated_fields")
        if (
            not isinstance(document, dict)
            or type(revision) is not int
            or revision < 1
            or not isinstance(updated_fields, list)
            or any(not isinstance(field, str) for field in updated_fields)
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration mutation response is incomplete",
                response=response,
            )
        metadata = document.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("configuration_revision") != revision
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration mutation revision does not match its document",
                response=response,
            )
        self._config_data = copy.deepcopy(document)
        self._rust_revision = revision
        return list(updated_fields)

    def _mutate_rust(self, operation: str, **parameters: Any) -> list[str]:
        """Serialize a Rust compare-and-swap and its local cache publication."""
        with _CONFIG_LOCK:
            expected_revision = self._rust_revision
            if expected_revision is None:
                raise ConfigurationAuthorityConfigurationError(
                    "Rust configuration snapshot is not initialized"
                )
            response = self._rust_call(
                operation,
                expected_revision=expected_revision,
                **parameters,
            )
            return self._accept_rust_mutation(response)

    def _load_config(
        self,
        *,
        minimum_revision: int | None = None,
        preserve_on_error: bool = False,
    ) -> None:
        if self.authority_mode == "rust":
            raise ConfigurationAuthorityConfigurationError(
                "Python configuration reads are disabled in Rust authority mode"
            )
        if self._config_file is None:
            self._config_file = Path(os.getcwd()) / "config.json"
        previous = self._config_data
        try:
            with self._config_file.open(encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("Configuration root must be an object")
            revision = _configuration_revision(
                data, default=_schema_configuration_revision()
            )
            if minimum_revision is not None and revision < minimum_revision:
                raise ValueError(
                    "metadata.configuration_revision cannot decrease on reload"
                )
            persisted_metadata = data.get("metadata")
            if (
                minimum_revision is not None
                and revision == minimum_revision
                and isinstance(persisted_metadata, dict)
                and "configuration_revision" in persisted_metadata
                and _configuration_without_revision(previous)
                != _configuration_without_revision(data)
            ):
                raise ValueError(
                    "configuration settings changed without increasing their revision"
                )
            metadata = data.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            metadata["configuration_revision"] = revision
            self._config_data = data
            logger.info("Loaded configuration from %s", self._config_file)
        except FileNotFoundError:
            logger.warning(
                "Config file not found: %s, using empty config", self._config_file
            )
            self._config_data = previous if preserve_on_error else {}
        except Exception as exc:
            logger.error("Error loading config from %s: %s", self._config_file, exc)
            self._config_data = previous if preserve_on_error else {}

    def get(self, key: str, default: Any = None) -> Any:
        with _CONFIG_LOCK:
            if self.authority_mode == "rust" and self._rust_revision is None:
                raise ConfigurationAuthorityConfigurationError(
                    "Rust configuration snapshot is not initialized"
                )
            if (
                self.authority_mode == "python"
                and not self._config_data
                and self._config_file is None
            ):
                self._load_config()
            value: Any = self._config_data
            for component in key.split("."):
                if isinstance(value, dict) and component in value:
                    value = value[component]
                else:
                    return default
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """Validate and atomically persist one dotted-path configuration update."""
        if not key or any(not component for component in key.split(".")):
            raise ValueError("Configuration key must be a non-empty dotted path")
        updates: dict[str, Any] = {}
        cursor = updates
        components = key.split(".")
        for component in components[:-1]:
            child: dict[str, Any] = {}
            cursor[component] = child
            cursor = child
        cursor[components[-1]] = value
        self.apply_updates(updates)

    def apply_updates(
        self, updates: dict[str, Any], *, force_update: bool = False
    ) -> list[str]:
        """Validate, merge, and persist a partial configuration update."""
        if self.authority_mode == "rust":
            return self._mutate_rust(
                "update",
                data=updates,
                force_update=force_update,
            )

        if not isinstance(updates, dict) or not updates:
            raise ValueError("Configuration updates must be a non-empty object")
        if _contains_path(updates, ("metadata", "configuration_revision")):
            raise ValueError(
                "metadata.configuration_revision is managed by the configuration store"
            )
        with _CONFIG_LOCK:
            updates = _json_copy(updates)
            candidate = _deep_merge(self._config_data, updates)
            errors, _ = validate_configuration(updates, baseline=self._config_data)
            if errors:
                raise ConfigValidationError(errors)
            changed_fields = _changed_leaf_paths(self._config_data, candidate)
            if changed_fields:
                self._commit(candidate)
            return changed_fields

    def replace_configuration(
        self, replacement: dict[str, Any], *, force_update: bool = False
    ) -> list[str]:
        """Validate and replace the complete stored configuration atomically."""
        if self.authority_mode == "rust":
            return self._mutate_rust(
                "replace",
                data=replacement,
                force_update=force_update,
            )

        if not isinstance(replacement, dict) or not replacement:
            raise ValueError("Configuration replacement must be a non-empty object")
        with _CONFIG_LOCK:
            replacement = _json_copy(replacement)
            errors, _ = validate_configuration(replacement)
            if errors:
                raise ConfigValidationError(errors)
            metadata = replacement.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            metadata["configuration_revision"] = _configuration_revision(
                self._config_data, default=_schema_configuration_revision()
            )
            changed_fields = _changed_leaf_paths(self._config_data, replacement)
            if changed_fields:
                self._commit(replacement)
            return changed_fields

    def _commit(self, data: dict[str, Any]) -> None:
        """Persist before publishing new in-memory state, with atomic replacement."""
        if self.authority_mode == "rust":
            raise ConfigurationAuthorityConfigurationError(
                "Python configuration writes are disabled in Rust authority mode"
            )
        if self._config_file is None:
            self._config_file = Path(os.getcwd()) / "config.json"
        previous = self._config_data
        candidate = copy.deepcopy(data)
        metadata = candidate.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        previous_revision = _configuration_revision(
            previous, default=_schema_configuration_revision()
        )
        if previous_revision >= _MAX_SAFE_JSON_INTEGER:
            raise ValueError("metadata.configuration_revision is exhausted")
        metadata["configuration_revision"] = previous_revision + 1
        try:
            self._save_config(candidate)
        except Exception:
            self._config_data = previous
            raise
        self._config_data = candidate

    def _save_config(self, data: dict[str, Any]) -> None:
        """Write complete configuration through fsync + same-directory rename."""
        if self.authority_mode == "rust":
            raise ConfigurationAuthorityConfigurationError(
                "Python configuration writes are disabled in Rust authority mode"
            )
        target = self._config_file
        if target is None:
            target = Path(os.getcwd()) / "config.json"
            self._config_file = target
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                json.dump(
                    data, temporary, indent=2, ensure_ascii=False, allow_nan=False
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temp_path, target)
            temp_path = None
            logger.debug("Saved configuration to %s", target)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Unable to remove temporary config file %s", temp_path
                    )

    def reload(self) -> None:
        with _CONFIG_LOCK:
            if self.authority_mode == "rust":
                self._load_rust_snapshot()
                return
            minimum_revision = _configuration_revision(
                self._config_data, default=_schema_configuration_revision()
            )
            self._load_config(
                minimum_revision=minimum_revision,
                preserve_on_error=True,
            )

    def get_all(self) -> dict[str, Any]:
        with _CONFIG_LOCK:
            if self.authority_mode == "rust" and self._rust_revision is None:
                raise ConfigurationAuthorityConfigurationError(
                    "Rust configuration snapshot is not initialized"
                )
            return copy.deepcopy(self._config_data)

    def resolve_capture_settings(self, *, schema_version: int = 3) -> dict[str, Any]:
        """Build a live capture request through the Rust configuration authority.

        The capture request determines which physical input Rust opens, so a
        Rust perception provider must not resolve it from Python-side defaults
        or a potentially stale local configuration cache. This operation reads
        the active document and its embedded Rust defaults together.
        """
        if self.authority_mode != "rust":
            raise ConfigurationAuthorityConfigurationError(
                "Rust live capture requires BILLIARDS_CONFIG_AUTHORITY=rust so "
                "capture settings are resolved by Rust"
            )
        if type(schema_version) is not int or schema_version != 3:
            raise ValueError("Rust live capture currently requires schema version 3")
        response = self._rust_call(
            "capture_settings", data={"schema_version": schema_version}
        )
        capture_settings = response.get("capture_settings")
        revision = response.get("revision")
        if (
            not isinstance(capture_settings, dict)
            or type(revision) is not int
            or revision < 1
            or type(capture_settings.get("configuration_revision")) is not int
            or capture_settings.get("configuration_revision") != revision
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust capture-settings response is missing a consistent request revision",
                response=response,
            )
        return copy.deepcopy(capture_settings)

    def validate_configuration(
        self, candidate: dict[str, Any], *, merge: bool = True
    ) -> dict[str, Any]:
        """Validate a candidate through Rust without mutating cached state."""
        if self.authority_mode != "rust":
            errors, warnings = validate_configuration(
                candidate,
                baseline=self.get_all() if merge else None,
            )
            return {
                "valid": not errors,
                "validation_errors": errors,
                "warnings": warnings,
                "revision": None,
            }
        response = self._rust_call("validate", data=candidate, merge=merge)
        if (
            type(response.get("valid")) is not bool
            or not isinstance(response.get("validation_errors"), list)
            or not isinstance(response.get("warnings"), list)
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration validation response is incomplete",
                response=response,
            )
        return response

    def get_defaults(self) -> dict[str, Any]:
        """Read and cache the embedded Rust defaults document."""
        if self.authority_mode != "rust":
            return load_schema_reference()
        if self._rust_defaults is None:
            response = self._rust_call("defaults")
            configuration = response.get("configuration")
            if not isinstance(configuration, dict):
                raise RustConfigurationAuthorityProtocolError(
                    "Rust configuration defaults response is incomplete",
                    response=response,
                )
            self._rust_defaults = copy.deepcopy(configuration)
        return copy.deepcopy(self._rust_defaults)

    def get_schema(self, section: str | None = None) -> dict[str, Any]:
        """Read and cache the Rust schema, optionally returning one section."""
        if self.authority_mode != "rust":
            return get_configuration_schema(section)
        if self._rust_schema is None:
            response = self._rust_call("schema")
            schema = response.get("schema")
            if not isinstance(schema, dict):
                raise RustConfigurationAuthorityProtocolError(
                    "Rust configuration schema response is incomplete",
                    response=response,
                )
            self._rust_schema = copy.deepcopy(schema)
        if section is None:
            return copy.deepcopy(self._rust_schema)
        properties = self._rust_schema.get("properties")
        schema = properties.get(section) if isinstance(properties, dict) else None
        if not isinstance(schema, dict):
            raise KeyError(section)
        return copy.deepcopy(schema)

    def export_configuration(
        self,
        *,
        sections: list[str] | None = None,
        include_defaults: bool = False,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Export the current configuration with optional section filtering."""
        if self.authority_mode != "rust":
            data = self.get_all()
            if sections is not None:
                data = {key: data[key] for key in sections if key in data}
            if not include_metadata:
                data.pop("metadata", None)
            if include_defaults:
                data = _deep_merge(self.get_defaults(), data)
            return data
        response = self._rust_call(
            "export",
            sections=sections,
            include_defaults=include_defaults,
            include_metadata=include_metadata,
        )
        configuration = response.get("configuration")
        if not isinstance(configuration, dict):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration export response is incomplete",
                response=response,
            )
        return copy.deepcopy(configuration)

    def reset_sections(
        self, sections: list[str], *, force_update: bool = False
    ) -> list[str]:
        """Reset selected top-level sections through the configured authority."""
        if self.authority_mode != "rust":
            if not sections:
                raise ValueError("At least one section is required")
            defaults = self.get_defaults()
            current = self.get_all()
            for section in sections:
                if section not in defaults:
                    raise KeyError(section)
                current[section] = defaults[section]
            return self.replace_configuration(current)
        return self._mutate_rust(
            "reset_section",
            sections=sections,
            force_update=force_update,
        )


def _changed_leaf_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, dict) or isinstance(after, dict):
        before_dict = before if isinstance(before, dict) else {}
        after_dict = after if isinstance(after, dict) else {}
        keys = sorted(set(before_dict) | set(after_dict))
        result: list[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            result.extend(
                _changed_leaf_paths(before_dict.get(key), after_dict.get(key), path)
            )
        return result
    if before == after:
        return []
    return [prefix]


def _contains_path(value: Any, path: tuple[str, ...]) -> bool:
    """Return whether an update object explicitly supplies a protected path."""
    if not path or not isinstance(value, dict):
        return False
    if path[0] not in value:
        return False
    return True if len(path) == 1 else _contains_path(value[path[0]], path[1:])


def _schema_configuration_revision() -> int:
    """Read the required initial revision from the immutable schema defaults."""
    defaults = load_schema_reference()
    metadata = defaults.get("metadata")
    revision = (
        metadata.get("configuration_revision") if isinstance(metadata, dict) else None
    )
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or not 1 <= revision <= _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError("schema default metadata.configuration_revision is invalid")
    return revision


def _configuration_revision(data: dict[str, Any], *, default: int) -> int:
    """Return a persisted monotonic revision and reject malformed disk state."""
    metadata = data.get("metadata")
    if "metadata" in data and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    revision = metadata.get("configuration_revision", default) if metadata else default
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or not 1 <= revision <= _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(
            "metadata.configuration_revision must be a positive safe integer"
        )
    return revision


def _configuration_without_revision(data: dict[str, Any]) -> dict[str, Any]:
    """Copy config content without its store-owned revision for reload checks."""
    result = copy.deepcopy(data)
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("configuration_revision", None)
        if not metadata:
            result.pop("metadata", None)
    return result


# Singleton instance for direct use by runtime modules.
config = Config()
