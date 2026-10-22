"""Deterministic child-process tests for the Rust live startup boundary."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from backend.config import load_schema_reference
from backend.perception.capture_configuration import (
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
    resolve_capture_settings,
)
from backend.perception.rust_frame_preview import (
    FramePreviewCalibrationControlSettings,
    FramePreviewProcessSettings,
)
from backend.perception.rust_live_supervisor import (
    RustLiveAuditIngress,
    RustLiveConfigurationError,
    RustLiveInferenceSettings,
    RustLiveSettings,
    RustLiveStartupError,
    RustLiveStatus,
    RustLiveSupervisor,
    _capture_runtime_identity,
    _validate_live_audit_record,
)
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENE_FIXTURE = PROJECT_ROOT / "perception/contracts/v1/scene-rust-tracked-success.json"
ACK_FIXTURE_ROOT = PROJECT_ROOT / "perception/contracts/capture-session-ack"


def _audit_frame_record(
    *, frame_id: str = "frame-1", sequence: int = 1
) -> dict[str, Any]:
    return {
        "record_type": "frame",
        "schema_version": 1,
        "process_id": 123,
        "engine_instance_id": "engine-a",
        "source_session_id": "session-a",
        "frame_id": frame_id,
        "transport_sequence": sequence,
        "configuration_revision": 1,
        "tracking_configuration_revision": 1,
        "rectification_revision": 1,
        "calibration_revision": None,
        "detection_count": 16,
        "track_count": 16,
        "source_timestamp": {
            "ticks": "123456789",
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": "source-a",
            "stage": "camera_read_complete",
        },
        "durations_ms": {
            "host_read_complete_to_inference_start": 0.2,
            "preprocess": 0.3,
            "inference": 2.4,
            "decode_and_nms": 0.4,
            "calibration": None,
            "tracking_and_state_assembly": 0.1,
            "scene_json_serialization": 0.2,
            "scene_queue_submit": 0.01,
            "host_read_complete_to_scene_queue_submit": 3.6,
        },
    }


def _audit_footer(*, records_written: int, dropped_records: int = 0) -> dict[str, Any]:
    return {
        "record_type": "footer",
        "schema_version": 1,
        "export_status": "complete",
        "records_written_before_footer": records_written,
        "dropped_records": dropped_records,
        "scene_socket_write_successes": 1,
        "scene_socket_write_failures": 0,
        "skipped_observations": 0,
        "rejected_observations": 0,
        "failed_observations": 0,
        "clean_shutdown": True,
        "pipeline_counters": {"frames_received": 1, "duplicate_frames": 0},
        "scene_transport_counters": {
            "written_snapshots": 1,
            "last_written_sequence": 1,
        },
    }


def test_rust_live_audit_schema_v2_accepts_nullable_segmented_capture_durations() -> (
    None
):
    value = _audit_frame_record()
    value["schema_version"] = 2
    value["durations_ms"] = {
        "capture_helper_appsink_pull_wait": None,
        "capture_helper_rgb_buffer_allocation": None,
        "capture_helper_rgb_mapping_copy": None,
        "parent_framed_record_read": None,
        "parent_record_read_complete_to_inference_start": 0.2,
        "preprocess": 0.3,
        "inference": 2.4,
        "decode_and_nms": 0.4,
        "calibration": None,
        "tracking_and_state_assembly": 0.1,
        "scene_json_serialization": 0.2,
        "scene_queue_submit": 0.01,
        "parent_record_read_complete_to_scene_queue_submit": 3.6,
    }

    assert _validate_live_audit_record(value) == value
    value["durations_ms"].pop("parent_framed_record_read")
    with pytest.raises(ValueError, match="schema version 2"):
        _validate_live_audit_record(value)


def test_rust_live_audit_schema_v3_requires_engine_total_on_frame_records() -> None:
    value = _audit_frame_record()
    value["schema_version"] = 3
    value["durations_ms"] = {
        "capture_helper_appsink_pull_wait": None,
        "capture_helper_rgb_buffer_allocation": None,
        "capture_helper_rgb_mapping_copy": None,
        "parent_framed_record_read": None,
        "parent_record_read_complete_to_inference_start": 0.2,
        "preprocess": 0.3,
        "inference": 2.4,
        "decode_and_nms": 0.4,
        "engine_total": 3.1,
        "calibration": None,
        "tracking_and_state_assembly": 0.1,
        "scene_json_serialization": 0.2,
        "scene_queue_submit": 0.01,
        "parent_record_read_complete_to_scene_queue_submit": 3.6,
    }

    assert _validate_live_audit_record(value) == value
    value["durations_ms"].pop("engine_total")
    with pytest.raises(ValueError, match="schema version 3"):
        _validate_live_audit_record(value)


def test_rust_live_audit_schema_v3_validates_stale_inference_timing_evidence() -> None:
    value = {
        "record_type": "observation_outcome",
        "schema_version": 3,
        "process_id": 123,
        "engine_instance_id": "engine-a",
        "source_session_id": "session-a",
        "frame_id": "stale-frame-1",
        "transport_sequence": None,
        "configuration_revision": 1,
        "tracking_configuration_revision": 2,
        "rectification_revision": 3,
        "calibration_revision": None,
        "source_timestamp": None,
        "outcome": "skipped",
        "stage": "freshness_check",
        "reason": "stale_after_inference",
        "inference_timings_ms": {
            "capture_helper_appsink_pull_wait": 6.0,
            "capture_helper_rgb_buffer_allocation": 3.0,
            "capture_helper_rgb_mapping_copy": 12.0,
            "parent_framed_record_read": 18.0,
            "parent_record_read_complete_to_inference_start": 2.0,
            "preprocess": 28.0,
            "inference": 144.0,
            "decode_and_nms": 12.0,
            "engine_total": 184.0,
            "parent_record_read_complete_to_inference_complete": 186.0,
            "maximum_frame_age": 100.0,
        },
    }

    assert _validate_live_audit_record(value) == value
    value["inference_timings_ms"] = None
    with pytest.raises(ValueError, match="must be an object"):
        _validate_live_audit_record(value)
    value["inference_timings_ms"] = {
        "capture_helper_appsink_pull_wait": 6.0,
        "capture_helper_rgb_buffer_allocation": 3.0,
        "capture_helper_rgb_mapping_copy": 12.0,
        "parent_framed_record_read": 18.0,
        "parent_record_read_complete_to_inference_start": 2.0,
        "preprocess": 28.0,
        "inference": 144.0,
        "decode_and_nms": 12.0,
        "engine_total": 184.0,
        "parent_record_read_complete_to_inference_complete": 186.0,
        "maximum_frame_age": 100.0,
    }
    value["inference_timings_ms"]["inference"] = float("inf")
    with pytest.raises(ValueError, match="finite non-negative"):
        _validate_live_audit_record(value)
    value["inference_timings_ms"]["inference"] = 144.0
    value["reason"] = "duplicate_source_frame"
    with pytest.raises(ValueError, match="only valid for stale-after-inference"):
        _validate_live_audit_record(value)


async def _send_audit_records(
    address: tuple[str, int], records: list[dict[str, Any]]
) -> None:
    reader, writer = await asyncio.open_connection(*address)
    del reader
    for record in records:
        payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _wait_for_audit_ingress(
    ingress: RustLiveAuditIngress,
    *,
    record_count: int | None = None,
    protocol_error: bool = False,
) -> None:
    """Wait for the server callback to consume the submitted test records."""
    deadline = asyncio.get_running_loop().time() + 1.0
    while asyncio.get_running_loop().time() < deadline:
        summary = ingress.summary()
        if protocol_error and summary["protocol_error"] is not None:
            return
        if record_count is not None and summary["records_received"] >= record_count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Rust audit ingress did not consume the expected records")


class EmptyConfiguration:
    """Explicitly select the numeric camera represented by live test fixtures."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = copy.deepcopy(values) if values is not None else {}
        vision = self.values.setdefault("vision", {})
        assert isinstance(vision, dict)
        camera = vision.setdefault("camera", {})
        assert isinstance(camera, dict)
        camera.setdefault("video_file_path", "")

    def get_all(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)


class RecordingBroadcaster:
    """Capture completed state publications without a WebSocket server."""

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []

    async def broadcast_game_state(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.states.append(kwargs)


def _write_child(
    tmp_path: Path,
    mode: str,
    *,
    request_record: Path | None = None,
) -> Path:
    executable = tmp_path / f"rust-live-{mode}"
    request_path = request_record or (tmp_path / f"request-{mode}.json")
    arguments_path = tmp_path / f"arguments-{mode}.json"
    launch_record_path = request_path.with_suffix(".launch.json")
    executable.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os, socket, struct, sys, time\n"
        "from pathlib import Path\n"
        "def arg(name): return sys.argv[sys.argv.index(name) + 1]\n"
        "if '--frame-preview-tcp' in sys.argv:\n"
        "  preview_bind = arg('--frame-preview-tcp').rsplit(':', 1)[0]\n"
        "  print('frame_preview_ready=' + json.dumps({'schema_version':1, 'address':preview_bind + ':49123', 'engine_instance_id':arg('--engine-instance-id'), 'capture_session_id':arg('--source-session-id')}, separators=(',', ':')), file=sys.stderr, flush=True)\n"
        "request = sys.stdin.buffer.read()\n"
        f"Path({str(request_path)!r}).write_bytes(request)\n"
        f"Path({str(arguments_path)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"Path({str(launch_record_path)!r}).write_text(json.dumps({{'argv':sys.argv[1:], 'env':dict(os.environ)}}))\n"
        "target = arg('--scene-snapshots-tcp').rsplit(':', 1)\n"
        f"mode = {mode!r}\n"
        "if mode == 'exit': sys.exit(9)\n"
        "if mode == 'timeout': time.sleep(30); sys.exit(0)\n"
        "client = socket.create_connection((target[0], int(target[1])))\n"
        "def send_frame(value):\n"
        "  payload = json.dumps(value, separators=(',', ':')).encode('utf-8')\n"
        "  client.sendall(struct.pack('>I', len(payload)) + payload)\n"
        "def read_exactly(count):\n"
        "  parts = bytearray()\n"
        "  while len(parts) < count:\n"
        "    chunk = client.recv(count - len(parts))\n"
        "    if not chunk: return None\n"
        "    parts.extend(chunk)\n"
        "  return bytes(parts)\n"
        "def read_frame():\n"
        "  header = read_exactly(4)\n"
        "  if header is None: return None\n"
        "  size = struct.unpack('>I', header)[0]\n"
        "  payload = read_exactly(size)\n"
        "  return json.loads(payload) if payload is not None else None\n"
        "request_value = json.loads(request)\n"
        f"ack_version = 1 if request_value['schema_version'] == 1 else 2\n"
        f"ack_path = Path({str(ACK_FIXTURE_ROOT)!r}) / ('v' + str(ack_version)) / 'capture-session-ack-default.json'\n"
        "ack = json.loads(ack_path.read_text())\n"
        "ack['request_sha256'] = hashlib.sha256(request).hexdigest()\n"
        "ack['source_report']['session_id'] = arg('--source-session-id')\n"
        "ack['source_report']['requested_profile'] = request_value['requested']['requested_profile']\n"
        "ack['source_report']['negotiated_profile'] = request_value['requested']['expected_negotiated_profile']\n"
        "ack['source_report']['applied_frame_geometry']['camera_profile_id'] = request_value['requested']['camera_profile_id']\n"
        "if request_value['schema_version'] >= 2: ack['source_report']['output_frame_geometry'] = request_value['requested']['output_frame_geometry']\n"
        "if mode == 'wrong_ack': ack['configuration_revision'] += 1\n"
        "hello = {'type':'hello', 'protocol_version':3, "
        "'engine_instance_id':arg('--engine-instance-id'), "
        "'source_session_id':arg('--source-session-id'), "
        "'capture_session_ack':ack}\n"
        "send_frame(hello)\n"
        "ready = read_frame()\n"
        "if ready is None: sys.exit(17)\n"
        "if mode == 'no_scene': sys.exit(0)\n"
        "scene = json.loads(Path(" + repr(str(SCENE_FIXTURE)) + ").read_text())\n"
        "scene['engine_instance_id'] = arg('--engine-instance-id')\n"
        "scene['source']['session_id'] = arg('--source-session-id')\n"
        "geometry = request_value['requested'].get('output_frame_geometry', request_value['requested']['expected_negotiated_profile'])\n"
        "scene['source']['width_px'] = geometry['width_px']\n"
        "scene['source']['height_px'] = geometry['height_px']\n"
        "for ball in scene['balls']:\n"
        "  ball['measurement']['image_width_px'] = geometry['width_px']\n"
        "  ball['measurement']['image_height_px'] = geometry['height_px']\n"
        "scene['cue']['image_width_px'] = geometry['width_px']\n"
        "scene['cue']['image_height_px'] = geometry['height_px']\n"
        "scene['transport_sequence'] = 1\n"
        "scene['snapshot_revision'] = 1\n"
        "send_frame(scene)\n"
        "if mode == 'one_scene': sys.exit(0)\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _capture_settings() -> CaptureSettingsSnapshotV1:
    snapshot = resolve_capture_settings(EmptyConfiguration(), schema_version=1)
    assert isinstance(snapshot, CaptureSettingsSnapshotV1)
    return snapshot


def _capture_settings_v2() -> CaptureSettingsSnapshotV2:
    snapshot = resolve_capture_settings(EmptyConfiguration(), schema_version=2)
    assert isinstance(snapshot, CaptureSettingsSnapshotV2)
    return snapshot


def _capture_settings_v3() -> CaptureSettingsSnapshotV3:
    snapshot = resolve_capture_settings(EmptyConfiguration())
    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    return snapshot


def _capture_settings_v3_with_selector(
    selector: str = f"gstreamer-device-v1:{'a' * 64}",
) -> CaptureSettingsSnapshotV3:
    snapshot = resolve_capture_settings(
        EmptyConfiguration({"perception": {"capture": {"device_selector": selector}}})
    )
    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    return snapshot


def _capture_settings_with_requested_values(
    snapshot: CaptureSettingsSnapshotV3,
    **updates: object,
) -> CaptureSettingsSnapshotV3:
    requested = snapshot.requested
    requested.update(updates)
    return CaptureSettingsSnapshotV3(
        schema_version=snapshot.schema_version,
        configuration_revision=snapshot.configuration_revision,
        _requested_json=json.dumps(requested, allow_nan=False, sort_keys=True),
    )


def _capture_helper_launch_settings(tmp_path: Path) -> dict[str, Any]:
    helper_path = tmp_path / "billiards-perception-gstreamer-probe"
    helper_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper_path.chmod(0o755)
    return {
        "capture_helper_executable_path": helper_path,
        "capture_helper_ready_record_max_bytes": 8192,
        "capture_helper_frame_header_max_bytes": 4096,
        "capture_helper_slow_reader_delay_ms": 0,
        "capture_helper_pull_timeout_ms": 1500,
        "capture_helper_readiness_timeout_ms": 500,
        "capture_helper_child_exit_timeout_ms": 2500,
        "capture_helper_child_exit_poll_interval_ms": 25,
    }


def test_supervisor_preserves_v1_rectified_calibration_dimensions() -> None:
    class RectifiedConfiguration:
        def get_all(self) -> dict[str, Any]:
            return {
                "vision": {
                    "camera": {
                        "resolution": [1000, 500],
                        "video_file_path": "",
                    }
                },
                "perception": {
                    "capture": {
                        "image_space": "rectified",
                        "rectification_revision": 2,
                        "calibration_revision": 3,
                    }
                },
            }

    configuration = RectifiedConfiguration()
    profile = resolve_capture_settings(configuration, schema_version=1)
    calibration_path = (
        PROJECT_ROOT
        / "perception/contracts/calibration/v1/table-calibration-python-export.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["camera_profile_id"] = profile.requested["camera_profile_id"]
    calibration["source_width_px"] = 800
    calibration["source_height_px"] = 400

    settings = resolve_capture_settings(
        configuration,
        calibration_bundle=calibration,
        schema_version=1,
    )

    identity = _capture_runtime_identity(settings)

    assert identity.image_space == "rectified"


def _inference_settings(tmp_path: Path, **updates: Any) -> RustLiveInferenceSettings:
    onnx_path = tmp_path / "model.onnx"
    checkpoint_path = tmp_path / "model.pt"
    onnx_path.write_bytes(b"test-onnx-export")
    checkpoint_path.write_bytes(b"test-source-checkpoint")
    manifest = {
        "manifest_schema_version": 1,
        "source_model": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "exported_model": {
            "path": onnx_path.name,
            "format": "onnx",
            "size_bytes": onnx_path.stat().st_size,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        },
        "model": {
            "class_count": 2,
            "class_names": [
                {"id": 0, "name": "cue"},
                {"id": 1, "name": "ball"},
            ],
        },
        "export": {"input_size": [640, 640]},
    }
    Path(f"{onnx_path}.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return RustLiveInferenceSettings(
        onnx_path=onnx_path,
        source_checkpoint_path=checkpoint_path,
        provider="cpu",
        image_size=640,
        confidence=0.31,
        iou=0.47,
        max_detections=300,
        intra_op_threads=2,
        warmup_runs=2,
        ball_class_id=1,
        cue_class_id=0,
        association_gate_px=17.25,
        association_prediction_horizon_ms=42,
        lost_after_ms=125,
        maximum_track_age_ms=575,
        maximum_active_tracks=24,
        recent_observation_capacity=12,
        **updates,
    )


def _settings(
    executable_path: Path,
    tmp_path: Path,
    *,
    startup_timeout_seconds: float = 2.0,
    capture_settings: (
        CaptureSettingsSnapshotV1
        | CaptureSettingsSnapshotV2
        | CaptureSettingsSnapshotV3
        | None
    ) = None,
    **updates: Any,
) -> RustLiveSettings:
    return RustLiveSettings(
        capture_settings=capture_settings or _capture_settings(),
        inference=_inference_settings(tmp_path),
        executable_path=executable_path,
        project_root=tmp_path,
        bind_address="127.0.0.1",
        port=0,
        maximum_message_bytes=262144,
        capture_settings_maximum_bytes=32768,
        read_timeout_seconds=2.0,
        source_silence_timeout_seconds=1.0,
        startup_timeout_seconds=startup_timeout_seconds,
        stop_timeout_seconds=1.0,
        graceful_stop_timeout_seconds=0.5,
        poll_interval_ms=25,
        connect_timeout_ms=1000,
        write_timeout_ms=1000,
        reconnect_initial_delay_ms=10,
        reconnect_maximum_delay_ms=100,
        output_tail_capacity_bytes=1024,
        **updates,
    )


def _runtime_configuration_with_defaults(overrides: dict[str, Any]) -> Any:
    """Create the same schema-backed dotted config view used by the API."""
    values = load_schema_reference()

    def merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(values, overrides)

    class RuntimeConfiguration:
        def get(self, key: str, default: Any = None) -> Any:
            current: Any = values
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return copy.deepcopy(current)

        def get_all(self) -> dict[str, Any]:
            return copy.deepcopy(values)

        def resolve_capture_settings(
            self, *, schema_version: int = 3
        ) -> dict[str, Any]:
            """Use the Python resolver only as a deterministic test oracle."""
            return resolve_capture_settings(
                self, schema_version=schema_version
            ).as_dict()

    return RuntimeConfiguration()


def test_rust_live_settings_load_exact_camera_model_and_process_configuration(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "billiards-perception"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    helper = tmp_path / "billiards-perception-gstreamer-probe"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper.chmod(0o755)
    inference = _inference_settings(tmp_path)
    configuration = _runtime_configuration_with_defaults(
        {
            "vision": {"camera": {"video_file_path": ""}},
            "perception": {
                "capture": {"device_selector": f"gstreamer-device-v1:{'b' * 64}"},
                "rust_live": {
                    "executable_path": str(executable),
                    "capture_helper_executable_path": str(helper),
                    "onnx_path": str(inference.onnx_path),
                    "source_checkpoint_path": str(inference.source_checkpoint_path),
                    "image_size": 640,
                },
            },
        }
    )

    settings = RustLiveSettings.from_configuration(configuration, project_root=tmp_path)

    assert settings.capture_settings.schema_version == 3
    assert settings.capture_settings.requested["device_id"].startswith(
        "gstreamer-device-v1:"
    )
    assert settings.inference.image_size == 640
    assert settings.inference.onnx_path == inference.onnx_path
    assert settings.executable_path == executable
    assert settings.capture_helper_executable_path == helper
    assert settings.public_state_delivery_schema_version == 1
    assert settings.frame_preview == FramePreviewProcessSettings(
        bind_address="127.0.0.1",
        port=0,
        maximum_header_bytes=8192,
        maximum_jpeg_bytes=4_194_304,
        maximum_concurrent_encodes=1,
        jpeg_quality=85,
        request_timeout_ms=1000,
        write_timeout_ms=2000,
        calibration_control=FramePreviewCalibrationControlSettings(
            store_path=tmp_path
            / "backend"
            / "calibration_data"
            / "calibration_table.json",
            store_lock_timeout_ms=100,
            maximum_store_file_bytes=65_536,
            maximum_request_bytes=65_536,
        ),
    )
    settings.validate()

    disabled = _runtime_configuration_with_defaults(
        {
            "vision": {"camera": {"video_file_path": ""}},
            "perception": {
                "capture": {"device_selector": f"gstreamer-device-v1:{'b' * 64}"},
                "rust_live": {
                    "executable_path": str(executable),
                    "capture_helper_executable_path": str(helper),
                    "onnx_path": str(inference.onnx_path),
                    "source_checkpoint_path": str(inference.source_checkpoint_path),
                    "image_size": 640,
                    "frame_preview_enabled": False,
                    "frame_preview_calibration_control_enabled": False,
                },
            },
        }
    )
    assert (
        RustLiveSettings.from_configuration(
            disabled, project_root=tmp_path
        ).frame_preview
        is None
    )


def test_rust_live_calibration_control_requires_the_complete_preview_settings_group(
    tmp_path: Path,
) -> None:
    executable = _write_child(tmp_path, "ready")
    inference = _inference_settings(tmp_path)
    shared_values = {
        "vision": {"camera": {"video_file_path": ""}},
        "perception": {
            "capture": {"device_selector": f"gstreamer-device-v1:{'b' * 64}"},
            "rust_live": {
                "executable_path": str(executable),
                "capture_helper_executable_path": str(executable),
                "onnx_path": str(inference.onnx_path),
                "source_checkpoint_path": str(inference.source_checkpoint_path),
                "frame_preview_enabled": False,
                "frame_preview_calibration_control_enabled": True,
            },
        },
    }
    configuration = _runtime_configuration_with_defaults(shared_values)

    with pytest.raises(RustLiveConfigurationError, match="requires frame preview"):
        RustLiveSettings.from_configuration(configuration, project_root=tmp_path)

    settings_with_path_escape = _runtime_configuration_with_defaults(
        {
            "vision": {"camera": {"video_file_path": ""}},
            "perception": {
                "capture": {"device_selector": f"gstreamer-device-v1:{'b' * 64}"},
                "rust_live": {
                    "executable_path": str(executable),
                    "capture_helper_executable_path": str(executable),
                    "onnx_path": str(inference.onnx_path),
                    "source_checkpoint_path": str(inference.source_checkpoint_path),
                    "frame_preview_calibration_store_path": "../outside/store.json",
                },
            },
        }
    )
    with pytest.raises(RustLiveConfigurationError, match="inside the project root"):
        RustLiveSettings.from_configuration(
            settings_with_path_escape, project_root=tmp_path
        )


def test_rust_live_settings_require_stable_physical_camera_selector(
    tmp_path: Path,
) -> None:
    configuration = _runtime_configuration_with_defaults(
        {"vision": {"camera": {"video_file_path": ""}}}
    )

    with pytest.raises(RustLiveConfigurationError, match="stable physical webcam"):
        RustLiveSettings.from_configuration(configuration, project_root=tmp_path)


def _supervisor(
    settings: RustLiveSettings,
) -> tuple[RustLiveSupervisor, RecordingBroadcaster]:
    broadcaster = RecordingBroadcaster()
    return (
        RustLiveSupervisor(settings, SceneSnapshotStatePublisher(broadcaster)),
        broadcaster,
    )


def test_exact_settings_bytes_acknowledgement_and_published_scene_establish_readiness(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[
        bytes,
        bytes,
        list[str],
        Any,
        Any,
        RustLiveStatus,
        RustLiveStatus,
        RecordingBroadcaster,
    ]:
        request_record = tmp_path / "stdin-request.bin"
        executable = _write_child(tmp_path, "ready", request_record=request_record)
        settings = _settings(executable, tmp_path)
        supervisor, broadcaster = _supervisor(settings)
        status = await supervisor.start()
        assert status.ready
        assert supervisor.process_id is not None
        encoded = request_record.read_bytes()
        expected = settings.capture_settings.to_json().encode("utf-8")
        acknowledgement = supervisor.capture_session_acknowledgement
        worker_stats = supervisor.worker_stats
        arguments = json.loads(
            (tmp_path / "arguments-ready.json").read_text(encoding="utf-8")
        )
        await supervisor.close()
        return (
            encoded,
            expected,
            arguments,
            acknowledgement,
            worker_stats,
            status,
            supervisor.status,
            broadcaster,
        )

    (
        encoded_request,
        expected_request,
        arguments,
        acknowledgement,
        worker_stats,
        ready_status,
        closed_status,
        broadcaster,
    ) = asyncio.run(run())

    assert encoded_request == expected_request
    assert not encoded_request.endswith(b"\n")
    launch_record = json.loads(
        (tmp_path / "stdin-request.launch.json").read_text(encoding="utf-8")
    )
    for argument in (
        "--capture-settings-stdin",
        "--engine-instance-id",
        "--source-session-id",
        "--scene-snapshots-tcp",
        "--scene-snapshots-max-message-bytes",
        "--scene-snapshots-connect-timeout-ms",
        "--scene-snapshots-write-timeout-ms",
        "--scene-snapshots-reconnect-initial-delay-ms",
        "--scene-snapshots-reconnect-maximum-delay-ms",
        "--capture-settings-max-bytes",
        "--max-detections",
        "--intra-op-threads",
        "--startup-timeout-ms",
        "--poll-interval-ms",
        "--graceful-stop-timeout-ms",
    ):
        assert argument in launch_record["argv"]
    assert expected_request.decode("utf-8") not in launch_record["argv"]
    assert expected_request.decode("utf-8") not in launch_record["env"].values()
    request = json.loads(encoded_request)
    assert request["schema_version"] == 1
    requested = request["requested"]
    expected_arguments = {
        "--onnx": str(tmp_path / "model.onnx"),
        "--source-checkpoint": str(tmp_path / "model.pt"),
        "--provider": "cpu",
        "--image-size": "640",
        "--confidence": "0.31",
        "--iou": "0.47",
        "--max-detections": "300",
        "--intra-op-threads": "2",
        "--warmup-runs": "2",
        "--capture-settings-max-bytes": "32768",
        "--source-image-space": requested["image_space"],
        "--tracking-ball-class-id": "1",
        "--tracking-cue-class-id": "0",
        "--tracking-camera-profile-id": requested["camera_profile_id"],
        "--tracking-association-gate-px": "17.25",
        "--tracking-association-prediction-horizon-ms": "42",
        "--tracking-lost-after-ms": "125",
        "--tracking-maximum-track-age-ms": "575",
        "--tracking-maximum-active-tracks": "24",
        "--tracking-recent-observation-capacity": "12",
        "--tracking-configuration-revision": str(request["configuration_revision"]),
        "--tracking-rectification-revision": str(requested["rectification_revision"]),
        "--startup-timeout-ms": "2000",
        "--poll-interval-ms": "25",
        "--graceful-stop-timeout-ms": "500",
    }
    for option, expected_value in expected_arguments.items():
        assert arguments[arguments.index(option) + 1] == expected_value
    assert "--capture-source" not in arguments
    assert "--calibration-bundle" not in arguments
    assert acknowledgement is not None
    assert (
        acknowledgement.configuration_revision
        == json.loads(encoded_request)["configuration_revision"]
    )
    assert (
        acknowledgement.request_sha256 == hashlib.sha256(expected_request).hexdigest()
    )
    assert ready_status.ready
    assert closed_status.process_status == "stopped"
    assert worker_stats.published == 1
    assert broadcaster.states[0]["observation_status"]["source_frame_id"]


@pytest.mark.parametrize("provider", ["cpu", "openvino"])
def test_live_supervisor_forwards_openvino_library_paths_only_when_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/openvino/linux-libs")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/test/openvino/macos-libs")
    request_record = tmp_path / f"request-{provider}.json"
    executable = _write_child(tmp_path, "ready", request_record=request_record)
    settings = _settings(executable, tmp_path)
    settings = replace(
        settings,
        inference=replace(settings.inference, provider=provider),
    )

    async def run() -> None:
        supervisor, _ = _supervisor(settings)
        assert (await supervisor.start()).ready
        await supervisor.close()

    asyncio.run(run())
    launch_record = json.loads(
        request_record.with_suffix(".launch.json").read_text(encoding="utf-8")
    )
    child_environment = launch_record["env"]
    for key, expected in (
        ("LD_LIBRARY_PATH", "/test/openvino/linux-libs"),
        ("DYLD_LIBRARY_PATH", "/test/openvino/macos-libs"),
    ):
        assert (child_environment.get(key) == expected) is (provider == "openvino")
    assert "HOME" not in child_environment


def test_transport_audit_is_exposed_only_as_an_explicit_supervisor_operation(
    tmp_path: Path,
) -> None:
    supervisor, _ = _supervisor(_settings(_write_child(tmp_path, "ready"), tmp_path))

    assert supervisor.finish_transport_audit().status == "disabled"

    supervisor.enable_transport_audit()
    summary = supervisor.finish_transport_audit()

    assert summary.status == "unavailable"
    assert summary.accepted_snapshots == 0
    assert summary.queue_capacity == 1


def test_stable_selector_passes_explicit_helper_settings_without_changing_request(
    tmp_path: Path,
) -> None:
    capture_settings = _capture_settings_v3_with_selector()
    helper_settings = _capture_helper_launch_settings(tmp_path)
    settings = _settings(
        _write_child(tmp_path, "ready"),
        tmp_path,
        capture_settings=capture_settings,
        **helper_settings,
    )
    supervisor, _ = _supervisor(settings)
    supervisor._engine_instance_id = "engine-selector-test"
    supervisor._source_session_id = "session-selector-test"

    arguments = supervisor._build_arguments(("127.0.0.1", 9000))
    option_values = {
        arguments[index]: arguments[index + 1]
        for index in range(len(arguments) - 1)
        if arguments[index].startswith("--capture-helper-")
    }

    assert option_values == {
        "--capture-helper-executable": str(
            helper_settings["capture_helper_executable_path"]
        ),
        "--capture-helper-ready-record-max-bytes": "8192",
        "--capture-helper-frame-header-max-bytes": "4096",
        "--capture-helper-slow-reader-delay-ms": "0",
        "--capture-helper-pull-timeout-ms": "1500",
        "--capture-helper-readiness-timeout-ms": "500",
        "--capture-helper-child-exit-timeout-ms": "2500",
        "--capture-helper-child-exit-poll-interval-ms": "25",
    }
    assert arguments[arguments.index("--capture-source") + 1] == "gstreamer-physical"
    assert capture_settings.requested["device_id"] == (
        f"gstreamer-device-v1:{'a' * 64}"
    )


def test_physical_selector_and_helper_settings_are_validated_before_startup(
    tmp_path: Path,
) -> None:
    valid_settings = _settings(
        _write_child(tmp_path, "ready"),
        tmp_path,
        capture_settings=_capture_settings_v3_with_selector(),
        **_capture_helper_launch_settings(tmp_path),
    )
    valid_settings.validate()

    bad_timeout = replace(valid_settings, capture_helper_pull_timeout_ms=0)
    with pytest.raises(RustLiveConfigurationError, match="pull_timeout_ms"):
        bad_timeout.validate()

    uppercase_selector = _capture_settings_with_requested_values(
        _capture_settings_v3_with_selector(),
        device_id=f"GSTREAMER-DEVICE-V1:{'a' * 64}",
    )
    bad_selector = replace(valid_settings, capture_settings=uppercase_selector)
    with pytest.raises(
        RustLiveConfigurationError, match="supported gstreamer-device-v1"
    ):
        bad_selector.validate()

    missing_helper = replace(valid_settings, capture_helper_executable_path=None)
    with pytest.raises(RustLiveConfigurationError, match="absolute Path"):
        missing_helper.validate()


@pytest.mark.parametrize("schema_version", [2, 3])
def test_v2_or_v3_capture_request_acknowledgement_and_scene_establish_readiness(
    tmp_path: Path, schema_version: int
) -> None:
    async def run() -> tuple[dict[str, Any], Any, RustLiveStatus]:
        request_record = tmp_path / f"stdin-request-v{schema_version}.bin"
        executable = _write_child(tmp_path, "ready", request_record=request_record)
        settings = _settings(
            executable,
            tmp_path,
            capture_settings=(
                _capture_settings_v2()
                if schema_version == 2
                else _capture_settings_v3()
            ),
        )
        supervisor, _ = _supervisor(settings)
        status = await supervisor.start()
        request = json.loads(request_record.read_bytes())
        acknowledgement = supervisor.capture_session_acknowledgement
        await supervisor.close()
        return request, acknowledgement, status

    request, acknowledgement, status = asyncio.run(run())

    assert status.ready
    assert request["schema_version"] == schema_version
    assert request["requested"]["output_frame_geometry"] == {
        "width_px": 1920,
        "height_px": 1080,
    }
    assert acknowledgement is not None
    assert (
        acknowledgement.request_sha256
        == hashlib.sha256(
            json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        acknowledgement.source_report["output_frame_geometry"]
        == request["requested"]["output_frame_geometry"]
    )


def test_inference_settings_require_explicit_validated_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="required positional arguments"):
        RustLiveInferenceSettings()  # type: ignore[call-arg]

    settings = _inference_settings(tmp_path)
    settings.validate()
    invalid_settings = (
        (replace(settings, provider="mps"), "provider"),
        (replace(settings, image_size=672), "fixed square size"),
        (replace(settings, max_detections=0), "max_detections"),
        (replace(settings, intra_op_threads=0), "intra_op_threads"),
        (replace(settings, confidence=float("nan")), "confidence"),
        (replace(settings, ball_class_id=2), "class IDs"),
        (
            replace(settings, maximum_track_age_ms=settings.lost_after_ms),
            "greater than lost_after_ms",
        ),
    )
    for candidate, message in invalid_settings:
        with pytest.raises(RustLiveConfigurationError, match=message):
            candidate.validate()


def test_inference_settings_reject_checkpoint_without_matching_lineage(
    tmp_path: Path,
) -> None:
    settings = _inference_settings(tmp_path)
    settings.source_checkpoint_path.write_bytes(b"different checkpoint")

    with pytest.raises(RustLiveConfigurationError, match="checkpoint hash"):
        settings.validate()


def test_live_settings_bound_request_input_and_order_shutdown_deadlines(
    tmp_path: Path,
) -> None:
    settings = _settings(_write_child(tmp_path, "ready"), tmp_path)
    RustLiveSupervisor(settings, SceneSnapshotStatePublisher(RecordingBroadcaster()))

    for candidate, message in (
        (
            replace(settings, capture_settings_maximum_bytes=0),
            "capture_settings_maximum_bytes",
        ),
        (replace(settings, poll_interval_ms=0), "poll_interval_ms"),
        (
            replace(
                settings, graceful_stop_timeout_seconds=settings.stop_timeout_seconds
            ),
            "must leave time before the parent stop timeout",
        ),
        (
            replace(settings, startup_timeout_seconds=1e308),
            "cannot be represented as a positive u64 millisecond timeout",
        ),
    ):
        with pytest.raises(RustLiveConfigurationError, match=message):
            RustLiveSupervisor(
                candidate, SceneSnapshotStatePublisher(RecordingBroadcaster())
            )


def test_omitted_intra_op_threads_preserve_onnx_runtime_default_policy(
    tmp_path: Path,
) -> None:
    settings = _settings(_write_child(tmp_path, "ready"), tmp_path)
    inference = replace(settings.inference, intra_op_threads=None)
    supervisor = RustLiveSupervisor(
        replace(settings, inference=inference),
        SceneSnapshotStatePublisher(RecordingBroadcaster()),
    )
    supervisor._engine_instance_id = "engine-test"
    supervisor._source_session_id = "session-test"

    assert "--intra-op-threads" not in supervisor._build_arguments(("127.0.0.1", 9000))


def test_rust_live_audit_ingress_validates_and_bounds_correlated_records() -> None:
    async def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        received: list[dict[str, Any]] = []
        ingress = RustLiveAuditIngress(
            queue_capacity=4,
            maximum_record_bytes=1024,
            connect_timeout_ms=500,
            write_timeout_ms=500,
            startup_timeout_seconds=1.0,
            retained_record_capacity=3,
            on_record=lambda record: received.append(record) is None,
        )
        address = await ingress.start()
        frame = _audit_frame_record()
        socket_write = {
            "record_type": "scene_socket_write",
            "schema_version": 1,
            "process_id": 123,
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": "frame-1",
            "transport_sequence": 1,
            "write_attempt": 1,
            "outcome": "written",
            "duration_ms": 0.4,
        }
        outcome = {
            "record_type": "observation_outcome",
            "schema_version": 1,
            "process_id": 123,
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": "frame-2",
            "transport_sequence": None,
            "configuration_revision": 1,
            "tracking_configuration_revision": 1,
            "rectification_revision": 1,
            "calibration_revision": None,
            "source_timestamp": None,
            "outcome": "skipped",
            "stage": "capture",
            "reason": "duplicate_source_frame",
        }
        await _send_audit_records(
            address,
            [frame, socket_write, outcome, _audit_footer(records_written=3)],
        )
        await _wait_for_audit_ingress(ingress, record_count=4)
        summary = ingress.summary()
        await ingress.close()
        return summary, received

    summary, received = asyncio.run(run())

    assert summary["status"] == "complete"
    assert summary["records_received"] == 4
    assert summary["record_counts"] == {
        "frame": 1,
        "scene_socket_write": 1,
        "observation_outcome": 1,
        "footer": 1,
    }
    assert summary["footer_record_count_matches"] is True
    assert summary["retained_record_count"] == 3
    assert summary["retained_records_truncated"] is True
    assert len(received) == 4
    assert received[0]["audit_receiver_monotonic_ns"] > 0


@pytest.mark.parametrize(
    ("payload", "expected_counter"),
    [
        (b'{"record_type":"frame","record_type":"footer"}', "malformed_records"),
        (b"x" * 1025, "oversized_records"),
    ],
)
def test_rust_live_audit_ingress_rejects_ambiguous_and_oversized_records(
    payload: bytes, expected_counter: str
) -> None:
    async def run() -> dict[str, Any]:
        ingress = RustLiveAuditIngress(
            queue_capacity=1,
            maximum_record_bytes=1024,
            connect_timeout_ms=500,
            write_timeout_ms=500,
            startup_timeout_seconds=1.0,
            retained_record_capacity=2,
        )
        address = await ingress.start()
        _reader, writer = await asyncio.open_connection(*address)
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await _wait_for_audit_ingress(ingress, protocol_error=True)
        summary = ingress.summary()
        await ingress.close()
        return summary

    summary = asyncio.run(run())
    assert summary["status"] == "partial"
    assert summary[expected_counter] == 1
    assert summary["protocol_error"] is not None


def test_rust_live_audit_ingress_marks_consumer_and_footer_loss_partial() -> None:
    async def run(*, consumer_accepts: bool) -> dict[str, Any]:
        ingress = RustLiveAuditIngress(
            queue_capacity=2,
            maximum_record_bytes=1024,
            connect_timeout_ms=500,
            write_timeout_ms=500,
            startup_timeout_seconds=1.0,
            retained_record_capacity=4,
            on_record=lambda _record: consumer_accepts,
        )
        address = await ingress.start()
        await _send_audit_records(
            address,
            [_audit_frame_record(), _audit_footer(records_written=1)],
        )
        await _wait_for_audit_ingress(ingress, record_count=2)
        summary = ingress.summary()
        await ingress.close()
        return summary

    summary = asyncio.run(run(consumer_accepts=False))
    assert summary["status"] == "partial"
    assert summary["consumer_dropped_records"] == 2
    assert summary["producer_dropped_records"] == 0
    assert summary["footer_record_count_matches"] is True


def test_rust_live_audit_ingress_marks_producer_loss_partial() -> None:
    async def run() -> dict[str, Any]:
        ingress = RustLiveAuditIngress(
            queue_capacity=2,
            maximum_record_bytes=1024,
            connect_timeout_ms=500,
            write_timeout_ms=500,
            startup_timeout_seconds=1.0,
            retained_record_capacity=4,
        )
        address = await ingress.start()
        await _send_audit_records(
            address,
            [
                _audit_frame_record(),
                _audit_footer(records_written=1, dropped_records=1),
            ],
        )
        await _wait_for_audit_ingress(ingress, record_count=2)
        summary = ingress.summary()
        await ingress.close()
        return summary

    summary = asyncio.run(run())
    assert summary["status"] == "partial"
    assert summary["producer_dropped_records"] == 1
    assert summary["footer_record_count_matches"] is True


def test_rust_live_audit_ingress_allows_the_child_startup_window_before_first_record() -> (
    None
):
    async def run() -> dict[str, Any]:
        ingress = RustLiveAuditIngress(
            queue_capacity=2,
            maximum_record_bytes=1024,
            connect_timeout_ms=10,
            write_timeout_ms=10,
            startup_timeout_seconds=0.2,
            retained_record_capacity=2,
        )
        address = await ingress.start()
        await asyncio.sleep(0.03)
        await _send_audit_records(
            address,
            [_audit_frame_record(), _audit_footer(records_written=1)],
        )
        await _wait_for_audit_ingress(ingress, record_count=2)
        summary = ingress.summary()
        await ingress.close()
        return summary

    summary = asyncio.run(run())
    assert summary["status"] == "complete"


def test_rust_live_supervisor_adds_audit_flags_only_when_receiver_is_enabled(
    tmp_path: Path,
) -> None:
    settings = _settings(_write_child(tmp_path, "ready"), tmp_path)
    supervisor = RustLiveSupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )
    supervisor._engine_instance_id = "engine-test"
    supervisor._source_session_id = "session-test"
    assert "--live-audit-tcp" not in supervisor._build_arguments(("127.0.0.1", 9000))

    async def enabled_arguments() -> list[str]:
        ingress = RustLiveAuditIngress(
            queue_capacity=8,
            maximum_record_bytes=4096,
            connect_timeout_ms=1500,
            write_timeout_ms=2000,
            startup_timeout_seconds=5.0,
            retained_record_capacity=32,
        )
        await ingress.start()
        enabled = RustLiveSupervisor(
            settings,
            SceneSnapshotStatePublisher(RecordingBroadcaster()),
            audit_ingress=ingress,
        )
        enabled._engine_instance_id = "engine-test"
        enabled._source_session_id = "session-test"
        arguments = enabled._build_arguments(("127.0.0.1", 9000))
        await ingress.close()
        return arguments

    arguments = asyncio.run(enabled_arguments())
    assert arguments[arguments.index("--live-audit-queue-capacity") + 1] == "8"
    assert arguments[arguments.index("--live-audit-max-record-bytes") + 1] == "4096"
    assert arguments[arguments.index("--live-audit-connect-timeout-ms") + 1] == "1500"
    assert arguments[arguments.index("--live-audit-write-timeout-ms") + 1] == "2000"


def test_rust_live_supervisor_adds_frame_preview_flags_only_when_configured(
    tmp_path: Path,
) -> None:
    settings = _settings(_write_child(tmp_path, "ready"), tmp_path)
    disabled = RustLiveSupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )
    disabled._engine_instance_id = "engine-test"
    disabled._source_session_id = "session-test"
    assert "--frame-preview-tcp" not in disabled._build_arguments(("127.0.0.1", 9000))
    assert not any(
        argument.startswith("--frame-preview-calibration-")
        for argument in disabled._build_arguments(("127.0.0.1", 9000))
    )

    preview_settings = FramePreviewProcessSettings(
        bind_address="127.0.0.1",
        port=0,
        maximum_header_bytes=8192,
        maximum_jpeg_bytes=4_194_304,
        maximum_concurrent_encodes=1,
        jpeg_quality=85,
        request_timeout_ms=1000,
        write_timeout_ms=2000,
        calibration_control=FramePreviewCalibrationControlSettings(
            store_path=tmp_path / "data" / "calibration_table.json",
            store_lock_timeout_ms=125,
            maximum_store_file_bytes=65_536,
            maximum_request_bytes=16_384,
        ),
    )
    enabled = RustLiveSupervisor(
        replace(settings, frame_preview=preview_settings),
        SceneSnapshotStatePublisher(RecordingBroadcaster()),
    )
    enabled._engine_instance_id = "engine-test"
    enabled._source_session_id = "session-test"
    arguments = enabled._build_arguments(("127.0.0.1", 9000))
    assert arguments[arguments.index("--frame-preview-tcp") + 1] == "127.0.0.1:0"
    assert arguments[arguments.index("--frame-preview-max-header-bytes") + 1] == "8192"
    assert arguments[arguments.index("--frame-preview-max-jpeg-bytes") + 1] == "4194304"
    assert (
        arguments[arguments.index("--frame-preview-max-concurrent-encodes") + 1] == "1"
    )
    assert arguments[
        arguments.index("--frame-preview-calibration-store-path") + 1
    ] == str(tmp_path / "data" / "calibration_table.json")
    assert (
        arguments[
            arguments.index("--frame-preview-calibration-store-lock-timeout-ms") + 1
        ]
        == "125"
    )
    assert (
        arguments[
            arguments.index("--frame-preview-calibration-store-max-file-bytes") + 1
        ]
        == "65536"
    )
    assert (
        arguments[arguments.index("--frame-preview-calibration-max-request-bytes") + 1]
        == "16384"
    )

    preview_only = RustLiveSupervisor(
        replace(
            settings, frame_preview=replace(preview_settings, calibration_control=None)
        ),
        SceneSnapshotStatePublisher(RecordingBroadcaster()),
    )
    preview_only._engine_instance_id = "engine-test"
    preview_only._source_session_id = "session-test"
    preview_only_arguments = preview_only._build_arguments(("127.0.0.1", 9000))
    assert "--frame-preview-tcp" in preview_only_arguments
    assert not any(
        argument.startswith("--frame-preview-calibration-")
        for argument in preview_only_arguments
    )


def test_rust_live_supervisor_captures_split_preview_readiness_records(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(_write_child(tmp_path, "ready"), tmp_path),
        frame_preview=FramePreviewProcessSettings(
            bind_address="127.0.0.1",
            port=0,
            maximum_header_bytes=8192,
            maximum_jpeg_bytes=4_194_304,
            maximum_concurrent_encodes=1,
            jpeg_quality=85,
            request_timeout_ms=1000,
            write_timeout_ms=2000,
        ),
    )
    supervisor = RustLiveSupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )
    supervisor._engine_instance_id = "engine-test"
    supervisor._source_session_id = "session-test"

    async def run() -> Any:
        loop = asyncio.get_running_loop()
        supervisor._frame_preview_ready_future = loop.create_future()

        class SplitStream:
            chunks = [
                b'frame_preview_ready={"schema_version":1,"address":"127.0.0.1:',
                b'40123","engine_instance_id":"engine-test",',
                b'"capture_session_id":"session-test"}\n',
            ]

            async def read(self, _size: int) -> bytes:
                return self.chunks.pop(0) if self.chunks else b""

        await supervisor._drain_output(
            cast("asyncio.StreamReader", SplitStream()),
            bytearray(),
            capture_frame_preview_ready=True,
        )
        return supervisor._frame_preview_ready_future.result()

    endpoint = asyncio.run(run())
    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 40123


def test_rust_live_start_waits_for_the_preview_listener_and_first_scene(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(_write_child(tmp_path, "ready"), tmp_path),
        frame_preview=FramePreviewProcessSettings(
            bind_address="127.0.0.1",
            port=0,
            maximum_header_bytes=8192,
            maximum_jpeg_bytes=4_194_304,
            maximum_concurrent_encodes=1,
            jpeg_quality=85,
            request_timeout_ms=1000,
            write_timeout_ms=2000,
        ),
    )

    async def run() -> tuple[RustLiveStatus, Any]:
        supervisor, _broadcaster = _supervisor(settings)
        try:
            status = await supervisor.start()
            assert supervisor._frame_preview_ready_future is not None
            endpoint = supervisor._frame_preview_ready_future.result()
            assert status.mode == "live"
            assert status.first_observation_received
            assert status.live_camera
            assert status.preview == "rust_current_frame"
            assert status.predictions == "unavailable"
            return status, endpoint
        finally:
            await supervisor.close()

    status, endpoint = asyncio.run(run())
    assert status.ready
    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 49123


@pytest.mark.parametrize("mode", ["wrong_ack", "no_scene", "exit"])
def test_mismatched_ack_missing_scene_and_child_exit_never_establish_readiness(
    tmp_path: Path, mode: str
) -> None:
    async def run() -> tuple[RustLiveSupervisor, RustLiveStartupError]:
        supervisor, _ = _supervisor(_settings(_write_child(tmp_path, mode), tmp_path))
        with pytest.raises(RustLiveStartupError) as error:
            await supervisor.start()
        return supervisor, error.value

    supervisor, error = asyncio.run(run())

    assert not supervisor.status.ready
    assert supervisor.status.process_return_code is not None
    if mode == "exit":
        assert "status 9" in str(error)
    else:
        assert "usable localization" in str(error)


def test_readiness_timeout_terminates_and_reaps_child(tmp_path: Path) -> None:
    async def run() -> RustLiveSupervisor:
        supervisor, _ = _supervisor(
            _settings(
                _write_child(tmp_path, "timeout"),
                tmp_path,
                startup_timeout_seconds=0.15,
            )
        )
        with pytest.raises(RustLiveStartupError, match="startup deadline"):
            await supervisor.start()
        return supervisor

    supervisor = asyncio.run(run())

    assert supervisor.status.process_return_code is not None
    assert supervisor.status.process_status == "failed"
    assert not supervisor.status.ready


def test_close_reaps_ready_child_and_closes_owned_publication_path(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[RustLiveSupervisor, RecordingBroadcaster]:
        supervisor, broadcaster = _supervisor(
            _settings(_write_child(tmp_path, "ready"), tmp_path)
        )
        status = await supervisor.start()
        assert status.ready
        await supervisor.close()
        return supervisor, broadcaster

    supervisor, broadcaster = asyncio.run(run())

    assert supervisor.status.process_return_code is not None
    assert supervisor.status.process_status == "stopped"
    assert not supervisor.status.ready
    assert supervisor.status.state_snapshots_published == 1
    assert broadcaster.states
