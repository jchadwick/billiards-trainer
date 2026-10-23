"""Tests for state stream identity and loss accounting."""

import asyncio
import copy
import hashlib
import json
import os
import struct
import subprocess
import time
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from backend.config import config
from backend.performance_audit import AsyncAuditRecorder
from backend.tools import collect_performance_audit as collector
from backend.tools.collect_performance_audit import (
    RustLiveStateAccumulator,
    StateAccumulator,
    TrajectoryAccumulator,
    _capture_transport_settings,
    _compare_overhead,
    _delivery_latency_summary,
    _event_export_status_by_process,
    _merge_final_export_status,
    _respond_to_server_ping,
    _run_capture_transport_benchmark,
    _send_client_heartbeat,
    _server_measurement_summary,
    build_parser,
    collect_run_summary,
    main,
    observation_age_ms,
    observation_from_message,
    percentile,
    run_audit,
    trajectory_from_message,
)


def _capture_transport_document(
    *,
    repetitions: int = 2,
    measured_frames: int = 11,
    build_profile: str = "release",
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for index in range(1, repetitions + 1):
        digest = f"{index:064x}"
        arm: dict[str, Any] = {
            "measured_frames": measured_frames,
            "frame_payload_sha256": digest,
            "frames_per_second": 123.5,
            "logical_payload_mib_per_second": 456.7,
            "data_movement_model": "frame Vec ownership moves through a bounded in-process sync_channel",
            "sampled_frame_completion": {
                "sampled_frames": 4,
                "p95_ns": 3_250_000,
            },
        }
        pipe_arm = dict(arm)
        pipe_arm["data_movement_model"] = (
            "helper writes payload through an OS stdout pipe"
        )
        runs.append(
            {
                "repetition": index,
                "in_process": dict(arm),
                "supervised_stdout_pipe": pipe_arm,
            }
        )
    return {
        "schema_version": 3,
        "experiment": "prepared_rgb8_in_process_vs_supervised_stdout_pipe",
        "scope": "isolated transport microbenchmark",
        "metadata": {"build_profile": build_profile},
        "workload": {
            "pixel_format": "RGB8",
            "width_px": 8,
            "height_px": 6,
            "warmup_frames_per_repetition": 2,
            "measured_frames_per_repetition": measured_frames,
            "repetitions": repetitions,
            "seed": 772,
            "max_payload_bytes": 4096,
            "max_total_frames_per_arm": 100,
            "max_total_samples": 16,
            "sampling_limit_per_arm_per_repetition": 4,
            "payload_bytes_per_frame": 144,
            "application_frame_buffer_count_per_arm": 2,
            "application_frame_buffer_bytes_per_arm": 288,
            "frame_records_per_arm_over_all_repetitions": (2 + measured_frames)
            * repetitions,
            "payload_content_model": "one deterministic RGB8 payload template repeated for every frame record in each arm",
            "payload_preparation_and_sha256": "generated and SHA-256 checked during arm setup/readiness; excluded from measured frame loops",
            "timed_operations": "in-process Vec ownership handoff versus helper frame serialization and parent pipe reads into a reusable buffer; payload generation and SHA-256 are excluded",
            "producer_consumer_model": "in_process_producer_thread_to_capacity_one_sync_channel; helper_process_writer_to_parent_consumer_over_OS_stdout_pipe",
        },
        "measurements_not_taken": {
            "camera_source_to_parent_latency": {"status": "unavailable"},
            "detector_accuracy_and_latency": {"status": "unavailable"},
            "process_cpu_time": {"status": "unavailable"},
            "resident_memory": {"status": "unavailable"},
            "operating_system_pipe_capacity": {"status": "unavailable"},
            "capture_mailbox_and_client_delivery": {"status": "unavailable"},
        },
        "repetitions": runs,
    }


@lru_cache(maxsize=1)
def _rust_timing_summary_executable() -> Path:
    """Build and return this checkout's timing CLI, never a stale PATH binary."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(collector.ROOT / "perception" / "Cargo.toml"),
            "--bin",
            "billiards-performance-audit-timing-summary",
        ],
        cwd=collector.ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
    target_directory = (
        Path(configured_target_dir).expanduser()
        if configured_target_dir
        else collector.ROOT / "perception" / "target"
    )
    if not target_directory.is_absolute():
        target_directory = collector.ROOT / target_directory
    executable_name = "billiards-performance-audit-timing-summary"
    if os.name == "nt":
        executable_name += ".exe"
    executable = target_directory / "debug" / executable_name
    assert executable.is_file()
    assert os.name == "nt" or os.access(executable, os.X_OK)
    return executable


@lru_cache(maxsize=1)
def _rust_trajectory_summary_executable() -> Path:
    """Build and return this checkout's trajectory summary CLI."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(collector.ROOT / "perception" / "Cargo.toml"),
            "--bin",
            "billiards-performance-audit-trajectory-summary",
        ],
        cwd=collector.ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
    target_directory = (
        Path(configured_target_dir).expanduser()
        if configured_target_dir
        else collector.ROOT / "perception" / "target"
    )
    if not target_directory.is_absolute():
        target_directory = collector.ROOT / target_directory
    executable_name = "billiards-performance-audit-trajectory-summary"
    if os.name == "nt":
        executable_name += ".exe"
    executable = target_directory / "debug" / executable_name
    assert executable.is_file()
    assert os.name == "nt" or os.access(executable, os.X_OK)
    return executable


@lru_cache(maxsize=1)
def _rust_resource_sampler_executable() -> Path:
    """Build and return this checkout's native resource helper."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--manifest-path",
            str(collector.ROOT / "perception" / "Cargo.toml"),
            "--bin",
            "billiards-performance-audit-resources",
        ],
        cwd=collector.ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    configured_target_dir = os.environ.get("CARGO_TARGET_DIR")
    target_directory = (
        Path(configured_target_dir).expanduser()
        if configured_target_dir
        else collector.ROOT / "perception" / "target"
    )
    if not target_directory.is_absolute():
        target_directory = collector.ROOT / target_directory
    executable_name = "billiards-performance-audit-resources"
    if os.name == "nt":
        executable_name += ".exe"
    executable = target_directory / "debug" / executable_name
    assert executable.is_file()
    assert os.name == "nt" or os.access(executable, os.X_OK)
    return executable


def _capture_transport_args(
    output_root: Path, *, include_executable: bool = True
) -> list[str]:
    args = [
        "--output-root",
        str(output_root),
        "--warmup",
        "0",
        "--duration",
        "1",
        "--repetitions",
        "1",
        "--quiet",
    ]
    if include_executable:
        args.extend(
            [
                "--capture-transport-benchmark-executable",
                "/tmp/billiards-capture-transport-bench",
                "--capture-transport-width-px",
                "8",
                "--capture-transport-height-px",
                "6",
                "--capture-transport-warmup-frames",
                "2",
                "--capture-transport-measured-frames",
                "11",
                "--capture-transport-repetitions",
                "2",
                "--capture-transport-sampling-limit",
                "4",
                "--capture-transport-seed",
                "772",
                "--capture-transport-max-payload-bytes",
                "4096",
                "--capture-transport-max-total-frames",
                "100",
                "--capture-transport-max-total-samples",
                "16",
                "--capture-transport-timeout-seconds",
                "15",
            ]
        )
    return args


def _rust_live_settings_file(tmp_path: Path) -> Path:
    executable = tmp_path / "rust-live"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    helper = tmp_path / "gstreamer-capture-helper"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper.chmod(0o755)
    onnx_path = tmp_path / "model.onnx"
    checkpoint_path = tmp_path / "model.pt"
    onnx_path.write_bytes(b"test-onnx-export")
    checkpoint_path.write_bytes(b"test-source-checkpoint")
    manifest = {
        "manifest_schema_version": 1,
        "source_model": {
            "path": checkpoint_path.name,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "exported_model": {
            "path": onnx_path.name,
            "format": "onnx",
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
    document = {
        "executable_path": str(executable),
        "project_root": str(tmp_path),
        "inference": {
            "onnx_path": str(onnx_path),
            "source_checkpoint_path": str(checkpoint_path),
            "provider": "cpu",
            "image_size": 640,
            "confidence": 0.3,
            "iou": 0.45,
            "max_detections": 16,
            "intra_op_threads": None,
            "warmup_runs": 0,
            "ball_class_id": 1,
            "cue_class_id": 0,
            "association_gate_px": 18.0,
            "association_prediction_horizon_ms": 50,
            "lost_after_ms": 200,
            "maximum_track_age_ms": 1000,
            "maximum_active_tracks": 16,
            "recent_observation_capacity": 32,
        },
        "supervisor": {
            "bind_address": "127.0.0.1",
            "port": 0,
            "maximum_message_bytes": 262144,
            "capture_settings_maximum_bytes": 32768,
            "read_timeout_seconds": 2.0,
            "source_silence_timeout_seconds": 1.0,
            "startup_timeout_seconds": 2.0,
            "stop_timeout_seconds": 1.0,
            "graceful_stop_timeout_seconds": 0.5,
            "poll_interval_ms": 25,
            "connect_timeout_ms": 1000,
            "write_timeout_ms": 1000,
            "reconnect_initial_delay_ms": 10,
            "reconnect_maximum_delay_ms": 100,
            "output_tail_capacity_bytes": 1024,
            "capture_helper_executable_path": str(helper),
            "capture_helper_ready_record_max_bytes": 8192,
            "capture_helper_frame_header_max_bytes": 4096,
            "capture_helper_slow_reader_delay_ms": 0,
            "capture_helper_pull_timeout_ms": 1500,
            "capture_helper_readiness_timeout_ms": 500,
            "capture_helper_child_exit_timeout_ms": 2500,
            "capture_helper_child_exit_poll_interval_ms": 25,
            "public_state_delivery_schema_version": 1,
            "frame_preview": None,
        },
    }
    settings_path = tmp_path / "rust-live-settings.json"
    settings_path.write_text(json.dumps(document), encoding="utf-8")
    return settings_path


def _rust_live_profile_args(settings_path: Path) -> list[str]:
    return [
        "--rust-live-qualification",
        "--rust-live-settings",
        str(settings_path),
        "--rust-live-device-selector",
        "gstreamer-device-v1:" + "a" * 64,
        "--rust-live-camera-backend",
        "gstreamer",
        "--rust-live-camera-width-px",
        "640",
        "--rust-live-camera-height-px",
        "360",
        "--rust-live-camera-fps",
        "30",
        "--rust-live-camera-pixel-format",
        "NV12",
        "--rust-live-image-space",
        "raw",
        "--rust-live-output-width-px",
        "640",
        "--rust-live-output-height-px",
        "360",
    ]


def _qualification_scene(frame_number: int) -> dict[str, Any]:
    scene_path = (
        Path(__file__).resolve().parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    frame_id = f"frame-{frame_number:04d}"
    timestamp = str(1_000_000_000 + frame_number)
    scene["source"].update(width_px=640, height_px=360, frame_id=frame_id)
    scene["source"]["session_id"] = "qualification-session"
    scene["source"]["timestamp"].update(
        ticks=timestamp,
        clock="monotonic",
        domain_id="source-clock",
        stage="camera_read_complete",
    )
    scene["published_at"]["ticks"] = str(2_000_000_000 + frame_number)
    scene["snapshot_revision"] = frame_number
    scene["transport_sequence"] = frame_number
    for ball in scene["balls"]:
        ball["last_observation"].update(
            session_id="qualification-session", frame_id=frame_id
        )
        ball["last_observation"]["timestamp"].update(
            ticks=timestamp,
            clock="monotonic",
            domain_id="source-clock",
            stage="camera_read_complete",
        )
        ball["measurement"].update(image_width_px=640, image_height_px=360)
        ball["measurement"]["image_center_px"] = {"x": 320.0, "y": 180.0}
        ball["measurement"]["table_center_mm"] = {"x": 4.0, "y": 5.0}
        ball["identity"].update(number=8, confidence=0.9, evidence=["fixture"])
    scene["cue"]["last_observation"].update(
        session_id="qualification-session", frame_id=frame_id
    )
    scene["cue"]["last_observation"]["timestamp"].update(
        ticks=timestamp,
        clock="monotonic",
        domain_id="source-clock",
        stage="camera_read_complete",
    )
    scene["cue"].update(image_width_px=640, image_height_px=360)
    scene["cue"]["image_center_px"] = {"x": 320.0, "y": 80.0}
    return cast("dict[str, Any]", scene)


def test_state_message_provenance_is_extracted_from_observation_status() -> None:
    observed = observation_from_message(
        {
            "sequence": 4,
            "data": {
                "sequence": 10,
                "ball_count": 2,
                "balls": [{"id": "ball-1", "position": {"x": 1, "y": 2}}],
                "cue": {"confidence": 0.9},
                "observation_status": {
                    "source_frame_number": 37,
                    "source_timestamp": 10.5,
                    "source_timestamp_kind": "shared_memory_publication",
                    "source_process_id": 101,
                    "source_publication_monotonic_ns": 9_500_000,
                    "vision_instance_id": "vision-a",
                    "map_revision": 3,
                    "calibration_revision": "cal-1",
                    "component_observation_status": {"balls": "observed"},
                },
            },
        }
    )

    assert observed == {
        "stream_sequence": 10,
        "source_frame_number": 37,
        "source_frame_id": None,
        "source_session_id": None,
        "source_observation_key": None,
        "state_complete": True,
        "source_timestamp": 10.5,
        "source_timestamp_stage": None,
        "source_timestamp_kind": "shared_memory_publication",
        "source_process_id": 101,
        "source_publication_monotonic_ns": 9_500_000,
        "vision_instance_id": "vision-a",
        "vision_process_id": None,
        "map_revision": 3,
        "calibration_revision": "cal-1",
        "ball_count": 2,
        "cue_present": True,
        "component_status": {"balls": "observed"},
    }


def test_empty_component_status_uses_published_live_component_statuses() -> None:
    observed = observation_from_message(
        {
            "type": "state",
            "data": {
                "sequence": 21,
                "ball_count": 1,
                "balls": [{"id": "ball-1", "position": {"x": 1, "y": 2}}],
                "cue": None,
                "observation_status": {
                    "source_frame_number": 104,
                    "component_observation_status": {},
                    "ball_positions_status": "observed",
                    "cue_detection_status": "failed",
                    "motion_monitor_status": "skipped",
                },
            },
        }
    )

    assert observed["component_status"] == {
        "balls": "observed",
        "cue": "failed",
        "motion": "skipped",
    }
    accumulator = StateAccumulator()
    accumulator.add(observed)
    summary = accumulator.summary(1.0)
    assert summary["unique_source_observations"] == 1
    assert summary["unknown_freshness_source_updates"] == 0
    assert summary["component_observation_status_counts"] == {
        "balls": {"observed": 1},
        "cue": {"failed": 1},
        "motion": {"skipped": 1},
    }


def test_rust_scene_state_provenance_is_correlated_and_complete() -> None:
    observed = observation_from_message(
        {
            "type": "state",
            "data": {
                "sequence": 12,
                "ball_count": 2,
                "balls": [{"id": "ball-1"}, {"id": "ball-2"}],
                "cue": None,
                "observation_status": {
                    "phase": "rust_scene_observation",
                    "source_session_id": "session-3",
                    "source_frame_id": "frame-000042",
                    "source_timestamp": "19000000042",
                    "source_timestamp_kind": "monotonic:clock-a:camera_read_complete",
                    "engine_instance_id": "engine-5",
                    "snapshot_revision": 42,
                    "calibration_revision": 7,
                    "ball_positions_status": "observed",
                    "observation_identity": {
                        "engine_instance_id": "engine-5",
                        "session_id": "session-3",
                        "frame_id": "frame-000042",
                    },
                },
            },
        }
    )

    assert observed["stream_sequence"] == 12
    assert observed["source_frame_number"] == 42
    assert observed["source_observation_key"] == (
        "engine-5",
        "session-3",
        "frame-000042",
    )
    assert observed["source_timestamp"] == "19000000042"
    assert observed["source_timestamp_stage"] == "camera_read_complete"
    assert observed["source_publication_monotonic_ns"] is None
    assert observed["state_complete"] is True
    assert observed["component_status"] == {"balls": "observed"}

    accumulator = StateAccumulator()
    accumulator.add(observed)
    accumulator.add(dict(observed, stream_sequence=13))
    summary = accumulator.summary(1)
    assert summary["unique_source_observations"] == 1
    assert summary["repeated_source_observations"] == 1
    assert summary["uncorrelated_state_messages"] == 0
    assert summary["complete_state_messages"] == 2
    assert summary["source_publication_wall_age_estimate_ms"]["count"] == 0


def test_project_metadata_hashes_rust_replay_assets_and_effective_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replay_frames = tmp_path / "frames"
    replay_frames.mkdir()
    (replay_frames / "frame-0001.png").write_bytes(b"frame one")
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx model")
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"source model")
    executable = tmp_path / "billiards-perception-infer"
    executable.write_bytes(b"native executable")
    monkeypatch.setattr(
        collector.config,
        "_config_data",
        {
            "perception": {
                "provider": "rust",
                "mode": "recorded_replay",
                "rust_replay": {
                    "onnx_path": str(onnx_path),
                    "source_checkpoint_path": str(checkpoint_path),
                    "input_path": str(replay_frames),
                    "executable_path": str(executable),
                    "provider": "openvino",
                },
            },
            "performance_audit": {},
        },
    )

    metadata = collector._project_metadata()

    assert metadata["perception_provider"] == "rust"
    assert metadata["model"]["sha256"] == collector._hash_file(onnx_path)
    assert metadata["source_model"]["sha256"] == collector._hash_file(checkpoint_path)
    assert metadata["inference_executable"]["sha256"] == collector._hash_file(
        executable
    )
    assert metadata["configured_video_input"]["kind"] == "recorded_frame_directory"
    assert metadata["configured_video_input"]["sha256"] == collector._hash_path(
        replay_frames
    )
    assert (
        metadata["effective_configuration"]["perception"]["rust_replay"]["provider"]
        == "openvino"
    )


def test_project_metadata_marks_missing_git_identity_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_git_metadata(*args: Any, **kwargs: Any) -> str:
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(collector.subprocess, "check_output", no_git_metadata)

    metadata = collector._project_metadata()

    assert metadata["revision"] is None
    assert metadata["revision_status"] == "unavailable"
    assert metadata["dirty_worktree_sha256"] is None
    assert metadata["dirty_worktree_files"] is None


def test_summary_separates_repeated_publications_from_unique_observations() -> None:
    summary = collect_run_summary(
        [
            {
                "source_frame_number": 1,
                "stream_sequence": 1,
                "component_status": {"balls": "observed"},
                "source_publication_age_ms": 4,
            },
            {
                "source_frame_number": 1,
                "stream_sequence": 2,
                "component_status": {"balls": "observed"},
                "source_publication_age_ms": 5,
            },
            {
                "source_frame_number": 3,
                "stream_sequence": 4,
                "component_status": {"balls": "observed"},
                "source_publication_age_ms": 8,
            },
            {"source_frame_number": None, "stream_sequence": 5},
        ],
        duration=2,
    )

    assert summary["state_messages_received"] == 4
    assert summary["state_messages_per_second"] == 2
    assert summary["unique_source_observations"] == 2
    assert summary["unique_source_frame_updates"] == 2
    assert summary["repeated_source_observations"] == 1
    assert summary["uncorrelated_state_messages"] == 1
    assert summary["source_sequence_gaps"] == 1
    assert summary["source_publication_wall_age_estimate_ms"] == {
        "count": 3,
        "retained_count": 3,
        "sample_limit": 10000,
        "truncated": False,
        "p50": 5,
        "p95": 8,
        "p99": 8,
    }


def test_cached_source_frames_do_not_inflate_fresh_observation_rate() -> None:
    summary = collect_run_summary(
        [
            {
                "source_frame_number": 1,
                "stream_sequence": 1,
                "component_status": {"balls": "observed"},
            },
            {
                "source_frame_number": 2,
                "stream_sequence": 2,
                "component_status": {"balls": "cached"},
            },
            {
                "source_frame_number": 2,
                "stream_sequence": 3,
                "component_status": {"balls": "cached"},
            },
        ],
        duration=2,
    )

    assert summary["unique_source_observations"] == 1
    assert summary["unique_observations_per_second"] == 0.5
    assert summary["unique_source_frame_updates"] == 2
    assert summary["cached_or_failed_source_updates"] == 1
    assert summary["repeated_source_observations"] == 1


def test_state_accumulator_labels_last_n_latency_retention() -> None:
    accumulator = StateAccumulator(maximum_samples=2)
    for sequence, age_ms in enumerate((5.0, 10.0, 20.0), start=1):
        accumulator.add(
            {
                "source_frame_number": sequence,
                "stream_sequence": sequence,
                "source_publication_age_ms": age_ms,
                "source_timestamp_kind": "shared_memory_publication",
                "source_publication_monotonic_ns": sequence * 1_000_000,
                "receive_monotonic_ns": sequence * 1_000_000 + 100_000,
                "clock_comparable": True,
            }
        )

    summary = accumulator.summary(3.0)
    assert summary["source_publication_wall_age_estimate_ms"] == {
        "count": 3,
        "retained_count": 2,
        "sample_limit": 2,
        "truncated": True,
        "p50": 10.0,
        "p95": 20.0,
        "p99": 20.0,
    }
    assert summary["source_publication_to_client_receipt_ms"] == {
        "count": 3,
        "retained_count": 2,
        "sample_limit": 2,
        "truncated": True,
        "p50": 0.1,
        "p95": 0.1,
        "p99": 0.1,
    }
    assert summary["client_latency_sample_count"] == 3
    assert summary["retained_client_latency_sample_count"] == 2
    assert summary["client_latency_samples_truncated"] is True


def test_state_and_trajectory_identity_windows_are_bounded_and_report_evictions() -> (
    None
):
    state = StateAccumulator(maximum_samples=2)
    for process_id in (10, 20, 30):
        state.add(
            {
                "source_frame_number": 1,
                "source_process_id": process_id,
                "source_timestamp_kind": "shared_memory_publication",
                "source_timestamp": float(process_id),
                "component_status": {"balls": "observed"},
            }
        )

    state_summary = state.summary(1.0)
    assert state_summary["source_identity_deduplication_window"] == {
        "policy": "least-recently-seen source identities",
        "capacity": 2,
        "retained": 2,
        "evictions": 1,
    }
    # An identity outside the window is counted as new because its high-water
    # mark has been intentionally discarded; that loss of deduplication memory
    # is disclosed rather than hidden.
    state.add(
        {
            "source_frame_number": 2,
            "source_process_id": 10,
            "source_timestamp_kind": "shared_memory_publication",
            "source_timestamp": 11.0,
            "component_status": {"balls": "observed"},
        }
    )
    assert state.summary(1.0)["source_identity_deduplication_window"]["evictions"] == 2

    trajectory = TrajectoryAccumulator(maximum_samples=2)
    for process_id in (10, 20, 30):
        trajectory.add(
            {
                "source_frame_number": 1,
                "vision_instance_id": "vision",
                "source_process_id": process_id,
                "stream_sequence": process_id,
            }
        )
    trajectory_summary = trajectory.summary(1.0)
    assert trajectory_summary["trajectory_source_identity_deduplication_window"] == {
        "policy": "least-recently-seen source identities",
        "capacity": 2,
        "retained": 2,
        "evictions": 1,
    }


def test_percentiles_and_publication_age_reject_invalid_inputs() -> None:
    assert percentile([], 0.95) is None
    assert percentile([3.0, 1.0, 2.0], 0.95) == 3.0
    assert observation_age_ms(98.0, 100.0) == 2000.0
    assert observation_age_ms("not-a-timestamp", 100.0) is None
    assert observation_age_ms(float("nan"), 100.0) is None


def test_state_accumulator_reports_completeness_gaps_duplicates_and_order() -> None:
    summary = collect_run_summary(
        [
            {
                "stream_sequence": sequence,
                "source_frame_number": sequence,
                "component_status": {"balls": "observed"},
            }
            for sequence in (1, 3, 3, 2, 4)
        ],
        duration=1,
    )

    assert summary["complete_state_messages"] == 5
    assert summary["incomplete_state_messages"] == 0
    assert summary["stream_sequence_gaps"] == 1
    assert summary["stream_sequence_duplicates"] == 1
    assert summary["stream_sequence_out_of_order_messages"] == 1


def test_state_accumulator_detects_same_process_source_restart_by_publication_time() -> (
    None
):
    source = {
        "vision_instance_id": "vision-a",
        "source_process_id": 101,
        "source_timestamp_kind": "shared_memory_publication",
        "component_status": {"balls": "observed"},
    }
    accumulator = StateAccumulator()
    for stream_sequence, frame_number, timestamp in (
        (1, 900, 100.0),
        (2, 901, 100.1),
        (3, 2, 101.0),  # same process, camera reopened and frame counter reset
        (4, 3, 101.1),
        (5, 902, 100.2),  # delayed message from before the restart
        (6, 3, 101.1),  # repeated publication of the active source frame
    ):
        accumulator.add(
            {
                **source,
                "stream_sequence": stream_sequence,
                "source_frame_number": frame_number,
                "source_timestamp": timestamp,
            }
        )

    summary = accumulator.summary(1.0)
    assert summary["unique_source_observations"] == 4
    assert summary["source_sequence_restarts"] == 1
    assert summary["source_sequence_gaps"] == 0
    assert summary["source_out_of_order_messages"] == 1
    assert summary["repeated_source_observations"] == 1


def test_server_summary_ranks_retained_vision_stage_spans() -> None:
    summary = _server_measurement_summary(
        [
            {
                "event_type": "vision_frame_complete",
                "source_frame_number": 1,
                "service_time_seconds": 0.2,
                "queue_delay_seconds": 0.01,
                "processing_queue_depth": 0,
                "shared_memory_read_copy_seconds": 0.002,
                "inclusive_stages_seconds": {
                    "ball_detection": 0.16,
                    "yolo_inference": 0.13,
                },
                "exclusive_stages_seconds": {
                    "ball_detection": 0.03,
                    "yolo_inference": 0.13,
                },
            },
            {
                "event_type": "vision_frame_complete",
                "source_frame_number": 2,
                "service_time_seconds": 0.3,
                "queue_delay_seconds": 0.02,
                "processing_queue_depth": 3,
                "shared_memory_read_copy_seconds": 0.004,
                "inclusive_stages_seconds": {
                    "ball_detection": 0.22,
                    "yolo_inference": 0.18,
                },
                "exclusive_stages_seconds": {
                    "ball_detection": 0.04,
                    "yolo_inference": 0.18,
                    "invalid": float("nan"),
                },
            },
            {
                "event_type": "integration_observation_processed",
                "coordinate_conversion_seconds": 0.001,
                "core_update_seconds": 0.003,
                "integration_lock_wait_seconds": 0.0005,
            },
        ]
    )

    vision = summary["vision"]
    assert vision["service_time_seconds"]["count"] == 2
    assert vision["service_time_seconds"]["p95"] == 0.3
    assert vision["queue_wait_seconds"]["mean"] == 0.015
    assert vision["sampled_processing_queue_depth"]["p95"] == 3
    assert vision["sampled_processing_queue_depth_max"] == 3
    assert vision["shared_memory_read_copy_seconds"]["p95"] == 0.004
    assert vision["stage_timings_seconds"]["yolo_inference"]["exclusive"] == {
        "count": 2,
        "p50": 0.13,
        "p95": 0.18,
        "p99": 0.18,
        "mean": 0.155,
    }
    assert [stage["stage"] for stage in vision["exclusive_stage_work_ranked"]] == [
        "yolo_inference",
        "ball_detection",
    ]
    assert vision["exclusive_stage_work_ranked"][0][
        "share_of_observed_exclusive_stage_seconds"
    ] == pytest.approx(0.8157894736842105)
    assert summary["integration"]["integration_lock_wait_seconds"]["p95"] == 0.0005


def test_python_collector_uses_rust_reducer_and_matches_shared_fixture() -> None:
    fixture_path = (
        collector.ROOT
        / "perception"
        / "tests"
        / "fixtures"
        / "server-audit-summary.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert (
        collector._server_measurement_summary(fixture["events"]) == fixture["expected"]
    )


def test_rust_server_summary_rejects_boolean_metric_values() -> None:
    summary = _server_measurement_summary(
        [
            {
                "event_type": "vision_frame_complete",
                "service_time_seconds": True,
                "queue_delay_seconds": True,
                "processing_queue_depth": True,
            },
            {"event_type": "process_resource_sample", "cpu_percent": True},
        ]
    )
    assert summary["vision"]["service_time_seconds"]["count"] == 0
    assert summary["vision"]["queue_wait_seconds"]["count"] == 0
    assert summary["vision"]["sampled_processing_queue_depth"]["count"] == 0
    assert summary["resources"]["cpu_percent"]["count"] == 0


def test_trajectory_work_timings_include_failed_stale_and_publish_failed_outcomes() -> (
    None
):
    outcomes = [
        ("trajectory_prediction_published", 0.1, 0.01, 0.02, 0.2, 0.03),
        ("trajectory_prediction_failed", 0.2, 0.03, 0.04, 0.4, None),
        ("trajectory_prediction_discarded_stale", 0.3, 0.04, 0.05, 0.5, None),
        ("trajectory_publication_failed", 0.4, 0.05, 0.06, 0.6, 0.07),
    ]
    events = [
        {
            "event_type": event_type,
            "prediction_queue_wait_seconds": queue_wait,
            "calculator_lock_wait_seconds": lock_wait,
            "executor_wait_seconds": executor_wait,
            "physics_compute_seconds": compute,
            "trajectory_publication_seconds": publication,
        }
        for event_type, queue_wait, lock_wait, executor_wait, compute, publication in outcomes
    ]

    trajectory = _server_measurement_summary(events)["trajectory"]

    assert trajectory["prediction_outcome_event_count"] == 4
    assert trajectory["prediction_outcome_counts"] == {
        "trajectory_prediction_discarded_stale": 1,
        "trajectory_prediction_failed": 1,
        "trajectory_prediction_published": 1,
        "trajectory_publication_failed": 1,
    }
    assert trajectory["prediction_queue_wait_seconds"]["count"] == 4
    assert trajectory["prediction_queue_wait_seconds"]["p95"] == 0.4
    assert trajectory["calculator_lock_wait_seconds"]["count"] == 4
    assert trajectory["executor_wait_seconds"]["count"] == 4
    assert trajectory["physics_compute_seconds"]["count"] == 4
    assert trajectory["physics_compute_seconds"]["p95"] == 0.6
    assert trajectory["trajectory_publication_seconds"]["count"] == 2
    assert trajectory["trajectory_publication_seconds"]["p95"] == 0.07


def test_server_summary_labels_sampled_detail_counts_as_recorded_events() -> None:
    summary = _server_measurement_summary(
        [
            {
                "event_type": "vision_frame_complete",
                "vision_instance_id": "vision-a",
                "vision_process_id": 12,
                "source_process_id": 9,
                "source_frame_number": frame_number,
            }
            for frame_number in (7, 8)
        ]
        + [
            {
                "event_type": "integration_observation_processed",
                "vision_instance_id": "vision-a",
                "vision_process_id": 12,
                "source_process_id": 9,
                "source_frame_number": 7,
            },
            {
                "event_type": "state_published",
                "vision_instance_id": "vision-a",
                "vision_process_id": 12,
                "source_process_id": 9,
                "source_frame_number": 7,
            },
        ]
    )

    assert "sampled" in summary["record_count_semantics"]
    assert summary["vision"]["recorded_frame_detail_events"] == 2
    assert (
        summary["vision"]["recorded_unique_source_observations_in_frame_details"] == 2
    )
    assert summary["integration"]["recorded_observation_detail_events"] == 1
    assert summary["state_delivery"]["recorded_state_publication_detail_events"] == 1
    assert "unique_source_observations_processed" not in summary["vision"]
    assert "unique_source_observations_published" not in summary["state_delivery"]


def test_trajectory_identity_and_publication_to_receipt_correlation() -> None:
    trajectory = trajectory_from_message(
        {
            "type": "trajectory",
            "data": {
                "sequence": 9,
                "source_frame_number": 41,
                "source_timestamp": 99.9,
                "source_observations": {
                    "vision_instance_id": "vision-a",
                    "source_process_id": 101,
                },
            },
        }
    )
    assert trajectory["stream_sequence"] == 9
    assert trajectory["source_frame_number"] == 41
    assert trajectory["source_process_id"] == 101

    summary = _delivery_latency_summary(
        [
            {
                "event_type": "state_received",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "stream_sequence": 4,
                "client_id": "client-a",
                "receive_monotonic_ns": 1_050_000,
                "source_publication_monotonic_ns": 970_000,
                "source_publication_to_client_receipt_ms": 0.08,
                "clock_comparable": True,
            }
        ],
        [
            {
                "event_type": "state_published",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "state",
                "stream_sequence": 4,
                "event_monotonic_ns": 1_000_000,
                "source_publication_monotonic_ns": 970_000,
                "component_age_seconds": {"balls": 0.02},
            },
            {
                "event_type": "websocket_delivery_complete",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "state",
                "client_id": "client-a",
                "stream_sequence": 4,
                "event_monotonic_ns": 1_030_000,
            },
        ],
        deadline_budget_ms=0.04,
    )
    assert summary["state_publication_to_receipt_ms"] == {
        "count": 1,
        "p50": 0.05,
        "p95": 0.05,
        "p99": 0.05,
    }
    assert summary["clocks_comparable"] is True
    assert summary["source_publication_to_state_ms"]["p95"] == 0.03
    assert summary["source_publication_to_client_receipt_ms"]["p95"] == 0.08
    assert summary["deadline_misses"]["state_publication_to_receipt"] == {
        "sample_count": 1,
        "missed_deadlines": 1,
    }
    assert summary["socket_send_to_receipt_ms"]["p50"] == 0.02
    assert summary["component_observation_age_at_publication_ms"]["balls"]["p95"] == 20


def test_overlapping_stream_sequences_keep_domain_latency_separate() -> None:
    summary = _delivery_latency_summary(
        [
            {
                "event_type": "state_received",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "stream_sequence": 1,
                "client_id": "client-a",
                "receive_monotonic_ns": 1_050_000,
                "source_publication_monotonic_ns": 980_000,
                "source_publication_to_client_receipt_ms": 0.07,
            },
            {
                "event_type": "trajectory_received",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "stream_sequence": 1,
                "client_id": "client-a",
                "receive_monotonic_ns": 2_080_000,
                "source_publication_monotonic_ns": 1_990_000,
                "source_publication_to_client_receipt_ms": 0.09,
            },
        ],
        [
            {
                "event_type": "state_published",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "state",
                "stream_sequence": 1,
                "event_monotonic_ns": 1_000_000,
                "source_publication_monotonic_ns": 980_000,
            },
            {
                "event_type": "trajectory_published",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "trajectory",
                "stream_sequence": 1,
                "event_monotonic_ns": 2_000_000,
            },
            {
                "event_type": "websocket_delivery_complete",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "state",
                "client_id": "client-a",
                "stream_sequence": 1,
                "event_monotonic_ns": 1_020_000,
            },
            {
                "event_type": "websocket_delivery_complete",
                "host_id": "audit-host",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "domain": "trajectory",
                "client_id": "client-a",
                "stream_sequence": 1,
                "event_monotonic_ns": 2_030_000,
            },
        ],
        deadline_budget_ms=0.06,
    )

    assert summary["state_publication_to_receipt_ms"] == {
        "count": 1,
        "p50": 0.05,
        "p95": 0.05,
        "p99": 0.05,
    }
    assert summary["trajectory_publication_to_receipt_ms"] == {
        "count": 1,
        "p50": 0.08,
        "p95": 0.08,
        "p99": 0.08,
    }
    assert summary["socket_send_to_receipt_ms"]["count"] == 2
    assert summary["trajectory_socket_send_to_receipt_ms"]["p95"] == 0.05
    assert summary["deadline_misses"]["state_publication_to_receipt"] == {
        "sample_count": 1,
        "missed_deadlines": 0,
    }


def test_host_name_does_not_establish_monotonic_clock_comparability() -> None:
    summary = _delivery_latency_summary(
        [
            {
                "event_type": "state_received",
                "host_id": "same-host-name",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "source_publication_wall_age_estimate_ms": 2.0,
                "source_publication_to_client_receipt_ms": 2.0,
            }
        ],
        [
            {
                "event_type": "state_published",
                "host_id": "same-host-name",
                "clock_domain_id": "darwin:boot-b:mach_absolute_time()",
                "event_monotonic_ns": 10_000_000,
                "source_publication_monotonic_ns": 8_000_000,
            }
        ],
    )

    assert summary["clocks_comparable"] is False
    assert summary["source_publication_to_state_ms"]["count"] == 0
    assert summary["source_publication_to_client_receipt_ms"]["count"] == 0


def test_instrumentation_off_baseline_uses_read_only_clock_identity() -> None:
    summary = _delivery_latency_summary(
        [
            {
                "event_type": "state_received",
                "clock_domain_id": "darwin:boot-a:mach_absolute_time()",
                "source_publication_to_client_receipt_ms": 14.0,
            }
        ],
        [],
        server_clock_domain_id="darwin:boot-a:mach_absolute_time()",
    )

    assert summary["clocks_comparable"] is True
    assert summary["source_publication_to_client_receipt_ms"]["p95"] == 14.0
    assert summary["source_publication_to_state_ms"]["count"] == 0


def test_rust_delivery_summary_matches_shared_fixture() -> None:
    fixture_path = (
        collector.ROOT
        / "perception"
        / "tests"
        / "fixtures"
        / "delivery-latency-summary.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = fixture["records"]
    config_record = records[0]
    client_events = [
        record["event"] for record in records if record["kind"] == "client"
    ]
    server_events = [
        record["event"] for record in records if record["kind"] == "server"
    ]

    summary = _delivery_latency_summary(
        client_events,
        server_events,
        deadline_budget_ms=config_record["deadline_budget_ms"],
        server_clock_domain_id=config_record["server_clock_domain_id"],
    )
    expected = fixture["expected"]
    expected["clock"] = time.get_clock_info("perf_counter").implementation
    assert summary == expected


def test_rust_delivery_summary_rejects_boolean_timestamps_and_metrics() -> None:
    clock_domain_id = "darwin:boot-a:mach_absolute_time()"
    summary = _delivery_latency_summary(
        [
            {
                "event_type": "state_received",
                "clock_domain_id": clock_domain_id,
                "stream_sequence": True,
                "receive_monotonic_ns": True,
                "source_publication_to_client_receipt_ms": True,
            }
        ],
        [
            {
                "event_type": "state_published",
                "clock_domain_id": clock_domain_id,
                "domain": "state",
                "stream_sequence": True,
                "event_monotonic_ns": True,
                "source_publication_monotonic_ns": 1,
            }
        ],
    )

    assert summary["clocks_comparable"] is True
    assert summary["state_publication_to_receipt_ms"]["count"] == 0
    assert summary["source_publication_to_state_ms"]["count"] == 0
    assert summary["source_publication_to_client_receipt_ms"]["count"] == 0


def test_server_summary_keeps_per_process_exporter_loss_status() -> None:
    exporter_status = {
        "dropped_events": 3,
        "sampled_out_events": 4,
        "export_limit_dropped_events": 1,
        "export_errors": 0,
    }
    summary = _server_measurement_summary(
        [
            {
                "event_type": "process_resource_sample",
                "host_id": "audit-host",
                "process_id": 101,
                "event_monotonic_ns": 1_000_000_000,
                "cpu_percent": 12.0,
                "resident_memory_bytes": 1024,
            },
            {
                "event_type": "process_resource_sample",
                "host_id": "audit-host",
                "process_id": 101,
                "event_monotonic_ns": 2_000_000_000,
                "cpu_percent": 14.0,
                "resident_memory_bytes": 2048,
                "audit_export_status": exporter_status,
            },
        ]
    )

    memory = summary["resources"]["by_process"]["audit-host:101"][
        "resident_memory_bytes"
    ]
    assert memory["change_first_to_last"] == 1024
    assert memory["observed_span_seconds"] == 1.0
    assert memory["change_bytes_per_second"] == 1024.0
    assert (
        summary["resources"]["by_process"]["audit-host:101"]["event_export_status"]
        == exporter_status
    )
    assert _event_export_status_by_process(summary["resources"]) == {
        "audit-host:101": exporter_status
    }


def test_export_status_report_handles_missing_or_malformed_process_resources() -> None:
    assert _event_export_status_by_process({}) == {}
    assert _event_export_status_by_process({"by_process": []}) == {}
    assert (
        _event_export_status_by_process(
            {"by_process": {"audit-host:101": "unavailable"}}
        )
        == {}
    )


def test_final_export_status_replaces_older_resource_snapshot() -> None:
    resources = {
        "by_process": {
            "audit-host:101": {
                "event_export_status": {
                    "process_id": 101,
                    "host_id": "audit-host",
                    "export_limit_dropped_events": 0,
                }
            }
        }
    }
    final_status = {
        "process_id": 101,
        "host_id": "audit-host",
        "export_limit_dropped_events": 16_309,
    }

    _merge_final_export_status(resources, final_status)

    assert _event_export_status_by_process(resources)["audit-host:101"] == final_status


def test_server_application_ping_is_answered_with_application_ping() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

    websocket = FakeWebSocket()
    replied = asyncio.run(
        _respond_to_server_ping(websocket, {"type": "ping", "data": {}})
    )

    assert replied is True
    assert json.loads(websocket.sent[0]) == {"type": "ping", "data": {}}


def test_client_application_heartbeat_is_sent_as_json_ping() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

    websocket = FakeWebSocket()
    asyncio.run(_send_client_heartbeat(websocket))

    assert json.loads(websocket.sent[0]) == {"type": "ping", "data": {}}


def test_non_ping_server_message_does_not_send_a_heartbeat() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

    websocket = FakeWebSocket()
    replied = asyncio.run(
        _respond_to_server_ping(websocket, {"type": "state", "data": {}})
    )

    assert replied is False
    assert websocket.sent == []


def test_collector_loads_service_config_before_building_parser_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Metadata and audit defaults must come from the backend's selected config."""
    config_path = tmp_path / "service-config.json"
    config_path.write_text(
        json.dumps({"performance_audit": {"measured_seconds": 17}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config, "_config_file", getattr(config, "_config_file", None), raising=False
    )
    monkeypatch.setattr(config, "_config_data", config.get_all(), raising=False)

    with pytest.raises(SystemExit) as result:
        main(["--config", str(config_path), "--help"])

    assert result.value.code == 0
    assert build_parser().parse_args([]).duration == 17


def test_trajectory_summary_does_not_overwrite_state_summary_fields() -> None:
    state = StateAccumulator()
    state.add(
        {
            "source_frame_number": 21,
            "vision_instance_id": "vision-a",
            "source_process_id": 10,
            "stream_sequence": 4,
            "source_publication_monotonic_ns": 8_000_000,
            "receive_monotonic_ns": 16_000_000,
            "source_publication_to_client_receipt_ms": 8.0,
            "clock_comparable": True,
            "component_status": {"balls": "observed"},
        }
    )
    trajectory = TrajectoryAccumulator()
    trajectory.add(
        {
            "source_frame_number": 21,
            "vision_instance_id": "vision-a",
            "source_process_id": 10,
            "stream_sequence": 7,
            "source_publication_to_client_receipt_ms": 12.0,
            "clock_comparable": True,
        }
    )

    summary = state.summary(1.0)
    summary.update(trajectory.summary(1.0))

    assert summary["unique_source_observations"] == 1
    assert summary["source_publication_to_client_receipt_ms"]["p50"] == 8.0
    assert summary["trajectory_unique_source_observations"] == 1
    assert summary["trajectory_source_publication_to_client_receipt_ms"]["p50"] == 12.0


def test_instrumentation_overhead_requires_matched_off_and_state_only_runs(
    tmp_path: Path,
) -> None:
    comparable_configuration: dict[str, dict[str, Any]] = {
        key: {}
        for key in (
            "camera",
            "detection",
            "processing",
            "vision_performance",
            "video",
            "calibration",
            "websocket_delivery",
            "stream_quality",
        )
    }
    metadata: dict[str, Any] = {
        "revision": "revision-a",
        "dirty_worktree_sha256": "dirty-a",
        "host": "audit-host",
        "platform": "platform-a",
        "processor": "processor-a",
        "hardware": {"logical_cpu_count": 8},
        "effective_configuration": {
            **comparable_configuration,
            "performance_audit": {
                "enabled": False,
                "state_enabled": True,
                "trajectory_enabled": False,
                "viewers_enabled": False,
                "sample_rate": 1.0,
                "history_size": 10000,
                "event_queue_capacity": 2048,
                "event_export_limit": 200000,
            },
        },
        "model": {"sha256": "model-a"},
        "configured_video_input": {"sha256": "input-a"},
    }
    baseline_runs: list[dict[str, Any]] = [
        {
            "unique_observations_per_second": 20.0,
            "streams": ["state"],
            "video_viewer": {"url": None},
            "subscriber_delay_ms": 0.0,
            "warmup_seconds": 30.0,
            "requested_measured_seconds": 120.0,
            "collector_settings": {
                "history_size": 10000,
                "queue_capacity": 2048,
                "sample_rate": 1.0,
                "event_export_limit": 200000,
            },
            "source_publication_to_client_receipt_ms": {
                "count": 100,
                "p95": 5.0,
            },
            "server_audit_domains": {
                "state": False,
                "trajectory": False,
                "viewers": False,
            },
        }
        for _ in range(3)
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"metadata": metadata, "repetitions": baseline_runs}),
        encoding="utf-8",
    )
    current_runs: list[dict[str, Any]] = [
        {
            "unique_observations_per_second": 19.8,
            "streams": ["state"],
            "video_viewer": {"url": None},
            "subscriber_delay_ms": 0.0,
            "warmup_seconds": 30.0,
            "requested_measured_seconds": 120.0,
            "collector_settings": {
                "history_size": 10000,
                "queue_capacity": 2048,
                "sample_rate": 1.0,
                "event_export_limit": 200000,
            },
            "source_publication_to_client_receipt_ms": {
                "count": 100,
                "p95": 5.05,
            },
            "server_audit_domains": {
                "state": True,
                "trajectory": False,
                "viewers": False,
            },
        }
        for _ in range(3)
    ]

    result = _compare_overhead(baseline_path, current_runs, metadata, 2.0)
    assert result is not None
    assert result["status"] == "pass"
    assert result["same_machine"] is True
    assert result["baseline_instrumentation_off"] is True
    assert result["current_state_audit_only"] is True
    assert result["same_workload"] is True
    assert result["same_collector_settings"] is True
    assert result["reference_state_only_workload"] is True

    current_runs[0]["subscriber_delay_ms"] = 10.0
    result = _compare_overhead(baseline_path, current_runs, metadata, 2.0)
    assert result is not None
    assert result["status"] == "inconclusive"
    assert result["same_workload"] is False
    assert result["reference_state_only_workload"] is False
    current_runs[0]["subscriber_delay_ms"] = 0.0

    current_runs[0]["collector_settings"]["sample_rate"] = 0.5
    result = _compare_overhead(baseline_path, current_runs, metadata, 2.0)
    assert result is not None
    assert result["status"] == "inconclusive"
    assert result["same_collector_settings"] is False
    current_runs[0]["collector_settings"]["sample_rate"] = 1.0

    current_metadata = {
        **metadata,
        "effective_configuration": {
            **metadata["effective_configuration"],
            "performance_audit": {
                **metadata["effective_configuration"]["performance_audit"],
                "sample_rate": 0.5,
            },
        },
    }
    result = _compare_overhead(baseline_path, current_runs, current_metadata, 2.0)
    assert result is not None
    assert result["status"] == "inconclusive"
    assert result["same_audit_configuration"] is False

    current_runs[0]["server_audit_domains"]["trajectory"] = True
    result = _compare_overhead(baseline_path, current_runs, metadata, 2.0)
    assert result is not None
    assert result["status"] == "inconclusive"
    assert result["current_state_audit_only"] is False

    unavailable_metadata = {
        **metadata,
        "revision": None,
        "dirty_worktree_sha256": None,
        "model": {"sha256": None},
        "configured_video_input": {"sha256": None},
    }
    baseline_path.write_text(
        json.dumps({"metadata": unavailable_metadata, "repetitions": baseline_runs}),
        encoding="utf-8",
    )
    result = _compare_overhead(baseline_path, baseline_runs, unavailable_metadata, 2.0)
    assert result is not None
    assert result["status"] == "inconclusive"
    assert result["same_revision"] is False
    assert result["same_dirty_worktree"] is False
    assert result["same_model_hash"] is False
    assert result["same_input_hash"] is False


def test_rust_overhead_comparison_matches_fixture_and_outcome_messages(
    tmp_path: Path,
) -> None:
    fixture_path = (
        collector.ROOT
        / "perception"
        / "tests"
        / "fixtures"
        / "overhead-comparison.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    input_document = fixture["input"]
    baseline_path = tmp_path / "baseline.json"

    def compare(document: dict[str, Any]) -> dict[str, Any]:
        baseline_path.write_text(
            json.dumps(
                {
                    "metadata": document["baseline_metadata"],
                    "repetitions": document["baseline_repetitions"],
                }
            ),
            encoding="utf-8",
        )
        return _compare_overhead(
            baseline_path,
            document["current_repetitions"],
            document["current_metadata"],
            document["budget_percent"],
            minimum_repetitions=document["minimum_repetitions"],
            minimum_delivery_samples=document["minimum_delivery_samples"],
        )

    expected = copy.deepcopy(fixture["expected"])
    expected["baseline_summary"] = str(baseline_path)
    assert compare(input_document) == expected

    over_budget = copy.deepcopy(input_document)
    for run in over_budget["current_repetitions"]:
        run["unique_observations_per_second"] = 90.0
        run["source_publication_to_client_receipt_ms"]["p95"] = 120.0
    over_budget_result = compare(over_budget)
    assert over_budget_result is not None
    assert over_budget_result["status"] == "over_budget"
    assert over_budget_result["interpretation"] == (
        "Instrumentation overhead exceeds the configured budget."
    )

    insufficient = copy.deepcopy(input_document)
    insufficient["current_repetitions"][0]["source_publication_to_client_receipt_ms"][
        "count"
    ] = 99
    inconclusive_result = compare(insufficient)
    assert inconclusive_result is not None
    assert inconclusive_result["status"] == "inconclusive"
    assert inconclusive_result["interpretation"].startswith(
        "No performance recommendation"
    )


def test_capture_transport_benchmark_is_opt_in_by_default() -> None:
    args = build_parser().parse_args([])
    assert args.capture_transport_benchmark_executable is None
    assert _capture_transport_settings(args) is None


def test_capture_transport_benchmark_requires_every_workload_bound_and_timeout() -> (
    None
):
    args = build_parser().parse_args(
        ["--capture-transport-benchmark-executable", "/tmp/bench"]
    )

    with pytest.raises(ValueError, match="requires explicit settings") as error:
        _capture_transport_settings(args)

    assert "--capture-transport-width-px" in str(error.value)
    assert "--capture-transport-timeout-seconds" in str(error.value)

    with pytest.raises(SystemExit) as cli_error:
        main(["--capture-transport-benchmark-executable", "/tmp/bench"])
    assert cli_error.value.code == 2


def test_capture_transport_benchmark_runs_with_explicit_arguments_and_validates_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = build_parser().parse_args(_capture_transport_args(tmp_path))
    settings = _capture_transport_settings(args)
    assert settings is not None
    document = _capture_transport_document()
    invocation: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        invocation["command"] = command
        invocation.update(kwargs)
        return subprocess.CompletedProcess(command, 0, json.dumps(document), "")

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    output_path = tmp_path / "capture-transport.json"
    summary = _run_capture_transport_benchmark(settings, output_path)

    assert invocation["command"] == [
        "/tmp/billiards-capture-transport-bench",
        "--width-px",
        "8",
        "--height-px",
        "6",
        "--warmup-frames",
        "2",
        "--measured-frames",
        "11",
        "--repetitions",
        "2",
        "--sampling-limit",
        "4",
        "--seed",
        "772",
        "--max-payload-bytes",
        "4096",
        "--max-total-frames",
        "100",
        "--max-total-samples",
        "16",
    ]
    assert invocation["timeout"] == 15
    assert invocation["cwd"] == collector.ROOT
    assert invocation["capture_output"] is True
    assert summary["build_profile"] == "release"
    assert (
        summary["repetitions"][0]["supervised_stdout_pipe"]["frames_per_second"]
        == 123.5
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == document


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (lambda report: report.update(schema_version=1), "schema_version 3"),
        (
            lambda report: report["repetitions"][0]["in_process"].update(
                measured_frames=10
            ),
            "frame count mismatch",
        ),
        (
            lambda report: report["repetitions"][0]["supervised_stdout_pipe"].update(
                frame_payload_sha256="0" * 64
            ),
            "mismatched.*SHA-256",
        ),
        (
            lambda report: report["workload"].update(seed=773),
            "workload metadata does not match",
        ),
    ],
)
def test_capture_transport_benchmark_rejects_invalid_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    error_match: str,
) -> None:
    args = build_parser().parse_args(_capture_transport_args(tmp_path))
    settings = _capture_transport_settings(args)
    assert settings is not None
    document = _capture_transport_document()
    mutation(document)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(document), "")

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    output_path = tmp_path / "capture-transport.json"

    with pytest.raises(ValueError, match=error_match):
        _run_capture_transport_benchmark(settings, output_path)

    assert not output_path.exists()


def test_capture_transport_process_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = build_parser().parse_args(_capture_transport_args(tmp_path))
    settings = _capture_transport_settings(args)
    assert settings is not None

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "", "configured limit rejected")

    monkeypatch.setattr(collector.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError, match="exited with status 2: configured limit rejected"
    ):
        _run_capture_transport_benchmark(settings, tmp_path / "capture-transport.json")


def test_capture_transport_report_is_saved_and_rendered_with_debug_caveat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = _capture_transport_document(repetitions=2, build_profile="debug")
    invocations: list[list[str]] = []
    run_process = subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "billiards-capture-transport-bench" not in command[0]:
            return run_process(command, **kwargs)
        invocations.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(document), "")

    async def fake_collect_repetition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "repetition": 1,
            "streams": ["state"],
            "state_messages_per_second": 10.0,
            "unique_observations_per_second": 9.0,
            "source_publication_to_client_receipt_ms": {"p95": None},
            "source_sequence_gaps": 0,
            "measurement_started_monotonic_ns": None,
            "measurement_ended_monotonic_ns": None,
            "monotonic_clocks_comparable": False,
            "server_audit_run_id": None,
            "server_event_export": None,
        }

    monkeypatch.setattr(collector.subprocess, "run", fake_run)
    monkeypatch.setattr(collector, "collect_repetition", fake_collect_repetition)
    monkeypatch.setattr(collector, "_project_metadata", lambda: {"revision": "test"})
    args = build_parser().parse_args(_capture_transport_args(tmp_path))

    run_directory = asyncio.run(run_audit(args))

    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (run_directory / "capture-transport.json").read_text(encoding="utf-8")
    )
    markdown = (run_directory / "report.md").read_text(encoding="utf-8")
    assert summary["schema_version"] == 6
    assert len(invocations) == 1
    assert artifact == document
    assert (
        summary["capture_transport_benchmark"]["artifact"] == "capture-transport.json"
    )
    assert summary["capture_transport_benchmark"]["build_profile"] == "debug"
    assert "## Capture transport microbenchmark" in markdown
    assert "Diagnostic only" in markdown
    assert "receipt p95" in markdown
    assert "not memory bandwidth" in markdown
    assert "Pipe logical MiB/s" in markdown
    assert "Producer/consumer model:" in markdown
    assert "The operating-system pipe capacity is not queried." in markdown
    assert (
        "Camera source-to-parent latency, process CPU time, and resident memory were not measured."
        in markdown
    )


def test_capture_transport_executable_is_not_invoked_for_normal_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_collect_repetition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "repetition": 1,
            "streams": ["state"],
            "state_messages_per_second": 0.0,
            "unique_observations_per_second": 0.0,
            "source_publication_to_client_receipt_ms": {"p95": None},
            "source_sequence_gaps": 0,
            "measurement_started_monotonic_ns": None,
            "measurement_ended_monotonic_ns": None,
            "monotonic_clocks_comparable": False,
            "server_audit_run_id": None,
            "server_event_export": None,
        }

    def fail_if_invoked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("capture transport benchmark should remain opt-in")

    monkeypatch.setattr(collector, "collect_repetition", fake_collect_repetition)
    monkeypatch.setattr(collector, "_run_capture_transport_benchmark", fail_if_invoked)
    args = build_parser().parse_args(
        _capture_transport_args(tmp_path, include_executable=False)
    )

    run_directory = asyncio.run(run_audit(args))

    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 6
    assert "capture_transport_benchmark" not in summary
    assert not (run_directory / "capture-transport.json").exists()


def test_rust_live_qualification_requires_explicit_selector_and_profile() -> None:
    args = build_parser().parse_args(["--rust-live-qualification"])

    with pytest.raises(ValueError, match="rust-live-settings"):
        collector._rust_live_profile_arguments(args)


def test_rust_live_settings_build_a_validated_source_defaults_request(
    tmp_path: Path,
) -> None:
    settings_path = _rust_live_settings_file(tmp_path)
    args = build_parser().parse_args(_rust_live_profile_args(settings_path))
    profile = collector._rust_live_profile_arguments(args)
    assert profile is not None

    settings = collector._read_rust_live_settings(settings_path, profile)
    requested = settings.capture_settings.requested

    assert settings.capture_settings.schema_version == 3
    assert requested["device_id"] == "gstreamer-device-v1:" + "a" * 64
    assert requested["backend"] == "gstreamer"
    assert requested["requested_profile"]["width_px"] == 640
    assert requested["requested_profile"]["height_px"] == 360
    assert requested["requested_profile"]["frame_rate"] == {
        "frames_numerator": 30,
        "seconds_denominator": 1,
    }
    assert requested["requested_profile"]["pixel_format"] == "NV12"
    assert requested["output_frame_geometry"] == {"width_px": 640, "height_px": 360}
    assert requested["camera_controls"] == {"mode": "source_defaults"}
    assert requested["timestamp_semantics"] == {"kind": "read_completion_only"}


def test_nested_frame_preview_settings_are_exported_as_json_objects(
    tmp_path: Path,
) -> None:
    from backend.perception.rust_frame_preview import (
        FramePreviewCalibrationControlSettings,
        FramePreviewProcessSettings,
    )

    preview = FramePreviewProcessSettings(
        bind_address="127.0.0.1",
        port=0,
        maximum_header_bytes=8192,
        maximum_jpeg_bytes=4_194_304,
        maximum_concurrent_encodes=1,
        jpeg_quality=85,
        request_timeout_ms=1000,
        write_timeout_ms=2000,
        calibration_control=FramePreviewCalibrationControlSettings(
            store_path=tmp_path / "calibration.json",
            store_lock_timeout_ms=100,
            maximum_store_file_bytes=65_536,
            maximum_request_bytes=65_536,
        ),
    )

    exported = collector._json_compatible_setting_value(preview)

    assert exported["bind_address"] == "127.0.0.1"
    assert exported["calibration_control"]["store_path"] == str(
        tmp_path / "calibration.json"
    )
    assert json.loads(json.dumps(exported)) == exported


def test_rust_live_qualification_rejects_non_loopback_ingress(
    tmp_path: Path,
) -> None:
    settings_path = _rust_live_settings_file(tmp_path)
    settings_document = json.loads(settings_path.read_text(encoding="utf-8"))
    settings_document["supervisor"]["bind_address"] = "0.0.0.0"
    settings_path.write_text(json.dumps(settings_document), encoding="utf-8")
    args = build_parser().parse_args(_rust_live_profile_args(settings_path))
    profile = collector._rust_live_profile_arguments(args)
    assert profile is not None

    with pytest.raises(ValueError, match="loopback"):
        collector._read_rust_live_settings(settings_path, profile)


def test_rust_live_state_accumulator_bounds_deduplication_and_accepts_zero_timestamp() -> (
    None
):
    accumulator = RustLiveStateAccumulator(maximum_samples=2)
    for frame_id, receipt in (
        ("frame-0001", 1_000_000),
        ("frame-0003", 2_000_000),
        ("frame-0002", 3_000_000),
        ("frame-0002", 4_000_000),
    ):
        accumulator.add(
            {
                "engine_instance_id": "engine-a",
                "source_session_id": "session-a",
                "source_frame_id": frame_id,
                "state_receipt_monotonic_ns": receipt,
                "source_timestamp_ticks": "0",
                "source_timestamp_stage": "camera_read_complete",
                "source_timestamp_is_exposure": False,
                "state_complete": True,
                "balls": [],
            }
        )

    summary = accumulator.summary(1.0)

    assert summary["unique_source_observations"] == 3
    assert summary["repeated_source_observations"] == 1
    assert summary["source_sequence_gaps"] == 1
    assert summary["source_out_of_order_messages"] == 1
    assert summary["complete_state_messages"] == 4
    assert summary["deduplication_window_observations"] == 2
    assert summary["retained_state_trace_count"] == 2
    assert summary["state_trace_truncated"] is True


def test_rust_live_child_resource_sampler_uses_native_bounded_helper() -> None:
    async def run_sampler() -> dict[str, Any]:
        sampler = collector.RustLiveChildResourceSampler(
            process_id=os.getpid(),
            interval_seconds=0.2,
            maximum_samples=2,
            startup_timeout_seconds=5,
            write_timeout_seconds=5,
            executable=_rust_resource_sampler_executable(),
        )
        await sampler.start()
        await asyncio.sleep(0.65)
        await sampler.close()
        return sampler.summary()

    summary = asyncio.run(run_sampler())

    assert summary["status"] == "available"
    assert summary["process_role"] == "rust_live_inference_child"
    assert summary["process_id"] == os.getpid()
    assert summary["process_start_time_unix_seconds"] is not None
    assert summary["sample_count"] >= 2
    assert summary["retained_sample_count"] == 2
    assert summary["samples_truncated"] is True
    assert summary["statistics_window"] == "retained_tail"
    assert summary["process_identity_resolution"] == (
        "pid_plus_sysinfo_start_time_unix_seconds"
    )
    assert summary["cpu_time_seconds"]["first"] is not None
    assert summary["resident_memory_bytes"]["maximum"] > 0
    assert all(
        sample["clock"] == "resource_sampler_process_monotonic"
        for sample in summary["retained_samples"]
    )


def test_rust_live_child_resource_sampler_failure_is_explicit() -> None:
    async def run_sampler() -> dict[str, Any]:
        sampler = collector.RustLiveChildResourceSampler(
            process_id=os.getpid(),
            interval_seconds=1,
            maximum_samples=100_000,
            startup_timeout_seconds=0.2,
            write_timeout_seconds=0.2,
            executable=Path("/missing/billiards-performance-audit-resources"),
        )
        await sampler.start()
        await sampler.close()
        return sampler.summary()

    summary = asyncio.run(run_sampler())
    assert summary["status"] == "unavailable"
    assert "does not exist" in summary["unavailable_reason"]
    assert summary["retention_limit"] == 50_000
    assert summary["requested_retention_limit"] == 100_000


def test_rust_live_child_resource_sampler_cancellation_reaps_helper(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "blocked-resource-helper.sh"
    helper.write_text(
        "#!/bin/sh\n"
        'printf \'{"kind":"ready","schema_version":2}\\n\'\n'
        "exec sleep 30\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)

    async def cancel_during_startup() -> int:
        sampler = collector.RustLiveChildResourceSampler(
            process_id=os.getpid(),
            interval_seconds=1,
            maximum_samples=2,
            startup_timeout_seconds=5,
            write_timeout_seconds=1,
            executable=helper,
        )
        start_task = asyncio.create_task(sampler.start())
        await asyncio.sleep(0.05)
        start_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start_task
        assert sampler._process is not None
        assert sampler._process.returncode is not None
        return sampler._process.returncode

    assert asyncio.run(cancel_during_startup()) != 0


def test_rust_live_timing_audit_flag_requires_live_qualification() -> None:
    args = build_parser().parse_args(["--rust-live-timing-audit"])
    with pytest.raises(ValueError, match="require --rust-live-qualification"):
        collector._rust_live_profile_arguments(args)


def test_rust_live_qualification_rejects_nonpositive_resource_sample_interval(
    tmp_path: Path,
) -> None:
    settings_path = _rust_live_settings_file(tmp_path)
    args = build_parser().parse_args(
        [
            *_rust_live_profile_args(settings_path),
            "--rust-live-resource-sample-interval",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="warm-up, duration and bounds"):
        asyncio.run(
            collector._run_rust_live_qualification(
                args, capture_transport_settings=None
            )
        )


def test_rust_live_qualification_runs_bounded_state_only_repetition_and_closes(
    tmp_path: Path,
) -> None:
    class FakeSupervisor:
        def __init__(self, _settings: Any, publisher: Any) -> None:
            self.publisher = publisher
            self.running = False
            self.closed = False
            self.transport_audit_enabled = False
            self.task: asyncio.Task[None] | None = None
            self._status = SimpleNamespace(
                ready=False,
                process_status="not_started",
                process_return_code=None,
                failure=None,
                ingress_accepted_snapshots=0,
                ingress_queue_replacements=0,
                state_snapshots_published=0,
            )

        @property
        def status(self) -> Any:
            return self._status

        async def start(self) -> Any:
            await self.publisher.publish(_qualification_scene(1))
            self.running = True
            self._status = SimpleNamespace(
                ready=True,
                process_status="running",
                process_return_code=None,
                failure=None,
                ingress_accepted_snapshots=1,
                ingress_queue_replacements=0,
                state_snapshots_published=1,
            )
            self.task = asyncio.create_task(self._emit_measurements())
            return self._status

        async def _emit_measurements(self) -> None:
            for frame_number in (2, 4, 3, 4):
                await asyncio.sleep(0.004)
                await self.publisher.publish(_qualification_scene(frame_number))
            await asyncio.Event().wait()

        def enable_transport_audit(self) -> None:
            self.transport_audit_enabled = True

        def finish_transport_audit(self) -> dict[str, Any]:
            assert self.transport_audit_enabled
            self.transport_audit_enabled = False
            return {
                "status": "available",
                "accepted_snapshots": 4,
                "payload_bytes_total": 4096,
                "payload_bytes_minimum": 900,
                "payload_bytes_mean": 1024.0,
                "payload_bytes_maximum": 1200,
                "sequence_gaps": 2,
                "queue_capacity": 1,
                "queue_depth_high_water": 1,
                "queue_depth_zero_before_enqueue": 2,
                "queue_depth_one_before_enqueue": 2,
                "queue_replacements": 2,
                "queue_replacement_fraction": 0.5,
            }

        async def close(self) -> None:
            self.closed = True
            self.running = False
            if self.task is not None:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self._status = SimpleNamespace(
                ready=False,
                process_status="stopped",
                process_return_code=0,
                failure=None,
                ingress_accepted_snapshots=5,
                ingress_queue_replacements=0,
                state_snapshots_published=5,
            )

    settings_path = _rust_live_settings_file(tmp_path)
    run_args = [
        *_rust_live_profile_args(settings_path),
        "--output-root",
        str(tmp_path / "audits"),
        "--warmup",
        "0",
        "--duration",
        "0.05",
        "--repetitions",
        "1",
        "--history-size",
        "8",
        "--rust-live-status-poll-interval",
        "0.005",
        "--quiet",
    ]
    args = build_parser().parse_args(run_args)
    supervisors: list[FakeSupervisor] = []

    def supervisor_factory(settings: Any, publisher: Any) -> FakeSupervisor:
        supervisor = FakeSupervisor(settings, publisher)
        supervisors.append(supervisor)
        return supervisor

    run_directory = asyncio.run(
        collector._run_rust_live_qualification(
            args,
            capture_transport_settings=None,
            supervisor_factory=supervisor_factory,
        )
    )

    assert len(supervisors) == 1
    assert supervisors[0].closed is True
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    repetition = summary["repetitions"][0]
    state = repetition["state"]
    assert repetition["warmup_state_messages"] == 1
    transport = repetition["rust_scene_transport"]
    assert transport["status"] == "available"
    assert transport["accepted_snapshots"] == 4
    assert transport["payload_bytes_total"] == 4096
    assert transport["queue_replacements"] == 2
    assert transport["queue_replacement_fraction"] == 0.5
    assert transport["measurement_window"]["clock"] == "collector_process_perf_counter"
    assert repetition["process_resources"]["status"] == "unavailable"
    assert "process id" in repetition["process_resources"]["unavailable_reason"]
    assert 0 < state["duration_seconds"] <= 0.15
    assert state["state_messages_received"] == 4
    assert state["unique_source_observations"] == 3
    assert state["repeated_source_observations"] == 1
    assert state["source_sequence_gaps"] == 1
    assert state["source_out_of_order_messages"] == 1
    assert state["observed_ball_measurements"] == 4
    assert state["identified_ball_measurements"] == 4
    assert state["table_coordinate_measurements"] == 4
    assert state["source_timestamp_stages"] == {"camera_read_complete": 4}
    assert state["source_exposure_timestamp_claims"] == 0
    assert summary["source_to_state_latency"]["status"].startswith("unavailable")
    assert summary["accuracy_evidence"].startswith("unavailable")
    assert summary["image_frame_persistence"].startswith("none")
    assert summary["capture_profile"]["camera_controls"] == {"mode": "source_defaults"}
    assert summary["event_export"]["written_events"] == 4
    assert "not a detector-accuracy result" in report
    assert "parent completion of the full frame-record read" in report
    assert "## Rust scene transport" in report
    assert "900 / 1024.0 / 1200" in report
    assert "Frame image persistence: `none" in report
    trace_path = Path(summary["state_event_file"])
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 4
    first_event = json.loads(trace_lines[0])
    assert first_event["event_type"] == "rust_live_state_received"
    assert first_event["balls"][0]["number"] == 8
    assert "rgb_frame" not in json.dumps(first_event)


@pytest.mark.parametrize(
    ("producer_dropped_records", "expected_status"),
    [(0, "complete"), (1, "partial")],
)
def test_rust_live_timing_audit_correlates_rust_records_and_exports_separately(
    tmp_path: Path, producer_dropped_records: int, expected_status: str
) -> None:
    class TimingSupervisor:
        def __init__(
            self, _settings: Any, publisher: Any, *, audit_ingress: Any
        ) -> None:
            self.publisher = publisher
            self.audit_ingress = audit_ingress
            self.process_id = os.getpid()
            self.writer: asyncio.StreamWriter | None = None
            self.task: asyncio.Task[None] | None = None
            self.records_written = 0
            self.status = SimpleNamespace(
                ready=False,
                process_status="not_started",
                process_return_code=None,
                failure=None,
                ingress_accepted_snapshots=0,
                ingress_queue_replacements=0,
                state_snapshots_published=0,
            )

        async def start(self) -> Any:
            _reader, self.writer = await asyncio.open_connection(
                *self.audit_ingress.address
            )
            await self.publisher.publish(_qualification_scene(1))
            self.status = SimpleNamespace(
                ready=True,
                process_status="running",
                process_return_code=None,
                failure=None,
                ingress_accepted_snapshots=1,
                ingress_queue_replacements=0,
                state_snapshots_published=1,
            )
            self.task = asyncio.create_task(self._emit_measurements())
            return self.status

        async def _write_record(self, record: dict[str, Any]) -> None:
            assert self.writer is not None
            payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
            self.writer.write(struct.pack(">I", len(payload)) + payload)
            await self.writer.drain()
            self.records_written += 1

        async def _emit_measurements(self) -> None:
            broadcaster = self.publisher._broadcaster
            while not broadcaster._measurement_open:
                await asyncio.sleep(0.001)
            for frame_number in (2, 4, 3, 4):
                await asyncio.sleep(0.004)
                scene = _qualification_scene(frame_number)
                await self.publisher.publish(scene)
                identity = {
                    "process_id": 4321,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": scene["source"]["session_id"],
                    "frame_id": scene["source"]["frame_id"],
                    "transport_sequence": scene["transport_sequence"],
                }
                await self._write_record(
                    {
                        "record_type": "frame",
                        "schema_version": 1,
                        **identity,
                        "configuration_revision": 1,
                        "tracking_configuration_revision": 1,
                        "rectification_revision": 1,
                        "calibration_revision": None,
                        "detection_count": len(scene["balls"]),
                        "track_count": len(scene["balls"]),
                        "source_timestamp": scene["source"]["timestamp"],
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
                )
                await self._write_record(
                    {
                        "record_type": "scene_socket_write",
                        "schema_version": 1,
                        **identity,
                        "write_attempt": 1,
                        "outcome": "written",
                        "duration_ms": 0.45,
                    }
                )
                if frame_number == 2:
                    await self._write_record(
                        {
                            "record_type": "observation_outcome",
                            "schema_version": 1,
                            "process_id": 4321,
                            "engine_instance_id": scene["engine_instance_id"],
                            "source_session_id": scene["source"]["session_id"],
                            "frame_id": scene["source"]["frame_id"],
                            "transport_sequence": scene["transport_sequence"],
                            "configuration_revision": 1,
                            "tracking_configuration_revision": 1,
                            "rectification_revision": 1,
                            "calibration_revision": None,
                            "source_timestamp": scene["source"]["timestamp"],
                            "outcome": "degraded",
                            "stage": "track_association",
                            "reason": "association_capacity_pressure_evicted_1",
                        }
                    )
            await asyncio.Event().wait()

        async def close(self) -> None:
            if self.task is not None:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            assert self.writer is not None
            await self._write_record(
                {
                    "record_type": "footer",
                    "schema_version": 1,
                    "export_status": "complete",
                    "records_written_before_footer": self.records_written,
                    "dropped_records": producer_dropped_records,
                    "scene_socket_write_successes": self.records_written // 2,
                    "scene_socket_write_failures": 0,
                    "skipped_observations": 0,
                    "rejected_observations": 0,
                    "failed_observations": 0,
                    "clean_shutdown": True,
                    "pipeline_counters": {
                        "association_capacity_pressure_frames": 1,
                        "association_capacity_evictions": 1,
                    },
                }
            )
            self.writer.close()
            await self.writer.wait_closed()
            self.status = SimpleNamespace(
                ready=False,
                process_status="stopped",
                process_return_code=0,
                failure=None,
                ingress_accepted_snapshots=5,
                ingress_queue_replacements=0,
                state_snapshots_published=5,
            )

    settings_path = _rust_live_settings_file(tmp_path)
    args = build_parser().parse_args(
        [
            *_rust_live_profile_args(settings_path),
            "--rust-live-timing-audit",
            "--rust-live-timing-summary-executable",
            str(_rust_timing_summary_executable()),
            "--output-root",
            str(tmp_path / "audits"),
            "--warmup",
            "0",
            "--duration",
            "0.2",
            "--rust-live-resource-sample-interval",
            "0.005",
            "--repetitions",
            "1",
            "--history-size",
            "16",
            "--rust-live-status-poll-interval",
            "0.005",
            "--quiet",
        ]
    )
    run_directory = asyncio.run(
        collector._run_rust_live_qualification(
            args,
            capture_transport_settings=None,
            supervisor_factory=TimingSupervisor,
        )
    )

    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    repetition = summary["repetitions"][0]
    timing = repetition["rust_pipeline_timing"]
    resources = repetition["process_resources"]
    assert repetition["rust_live_audit_ingress"]["status"] == expected_status
    assert timing["status"] == expected_status
    assert timing["producer_dropped_records"] == producer_dropped_records
    assert resources["status"] == "unavailable"
    assert "at least" in resources["unavailable_reason"]
    assert resources["process_role"] == "rust_live_inference_child"
    assert resources["process_id"] == os.getpid()
    assert resources["sample_count"] == 0
    assert timing["state_timing_correlation"]["matched_frames"] == 3
    assert timing["state_timing_correlation"]["duplicate_timing_frame_records"] == 1
    assert timing["stage_durations_ms"]["inference"]["p95"] == 2.4
    assert timing["matched_scene_socket_write_durations_ms"]["written"]["count"] == 4
    assert summary["event_export"]["written_events"] == 4
    assert summary["rust_pipeline_timing_event_export"]["written_events"] == 10
    assert (
        timing["association_capacity_pressure"]["measured_window_pressure_frames"] == 1
    )
    assert (
        timing["association_capacity_pressure"]["measured_window_evicted_tracks"] == 1
    )
    assert timing["association_capacity_pressure"]["run_total_pressure_frames"] == 1
    assert timing["association_capacity_pressure"]["run_total_evicted_tracks"] == 1
    state_trace = Path(summary["state_event_file"]).read_text(encoding="utf-8")
    timing_trace = Path(summary["rust_pipeline_timing_event_file"]).read_text(
        encoding="utf-8"
    )
    assert "rust_live_state_received" in state_trace
    assert "rust_live_pipeline_timing_record" in timing_trace
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "Rust pipeline timing (opt-in)" in report
    assert "## Rust child resources" in report
    assert "Pressure frames" in report


def test_rust_timing_summary_bridge_runs_native_summary_with_exact_state_join() -> None:
    async def run_bridge() -> dict[str, Any]:
        bridge = collector.RustLiveTimingSummaryBridge(
            maximum_samples=8,
            queue_capacity=8,
            maximum_line_bytes=1_048_576,
            write_timeout_seconds=1.0,
            executable=_rust_timing_summary_executable(),
        )
        await bridge.start()
        await bridge.begin_measurement(100)
        await asyncio.sleep(0)
        assert bridge.try_record(
            {
                "record_type": "frame",
                "engine_instance_id": "engine-a",
                "source_session_id": "session-a",
                "frame_id": "frame-1",
                "transport_sequence": 1,
                "detection_count": 1,
                "track_count": 1,
                "durations_ms": {"inference": 2.4},
                "audit_receiver_monotonic_ns": 120,
            }
        )
        assert bridge.try_record(
            {
                "record_type": "footer",
                "export_status": "complete",
                "dropped_records": 0,
                "audit_receiver_monotonic_ns": 190,
            }
        )
        await bridge.end_measurement(200)
        result = await bridge.finish(
            [
                {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "frame-1",
                    "source_transport_sequence": 1,
                }
            ],
            state_trace_truncated=False,
        )
        metadata = bridge.failure_metadata()
        await bridge.close()
        return {"result": result, "metadata": metadata}

    output = asyncio.run(run_bridge())
    summary = output["result"]["summary"]
    assert output["metadata"]["status"] == "complete"
    assert summary["status"] == "complete"
    assert summary["state_timing_correlation"]["matched_frames"] == 1
    assert summary["stage_durations_ms"]["inference"]["p50"] == 2.4


def test_rust_timing_summary_bridge_bounds_callback_and_reports_startup_failure() -> (
    None
):
    async def run_full_queue() -> tuple[dict[str, Any], dict[str, Any]]:
        bridge = collector.RustLiveTimingSummaryBridge(
            maximum_samples=8,
            queue_capacity=1,
            maximum_line_bytes=1_048_576,
            write_timeout_seconds=1.0,
            executable=_rust_timing_summary_executable(),
        )
        await bridge.start()
        first_accepted = bridge.try_record(
            {
                "record_type": "footer",
                "export_status": "complete",
                "dropped_records": 0,
            }
        )
        second_accepted = bridge.try_record({"record_type": "sample", "value": 2})
        result = await bridge.finish([], state_trace_truncated=False)
        metadata = bridge.failure_metadata()
        await bridge.close()
        return (
            {
                "first_accepted": first_accepted,
                "second_accepted": second_accepted,
                "result": result,
            },
            metadata,
        )

    outcome, metadata = asyncio.run(run_full_queue())
    assert outcome["first_accepted"] is True
    assert outcome["second_accepted"] is False
    assert metadata["dropped_records"] == 1
    assert metadata["status"] == "partial"
    assert outcome["result"]["bridge_dropped_records"] == 1
    assert outcome["result"]["summary"]["status"] == "partial"

    async def run_missing_executable() -> tuple[dict[str, Any], dict[str, Any]]:
        bridge = collector.RustLiveTimingSummaryBridge(
            maximum_samples=8,
            queue_capacity=1,
            maximum_line_bytes=1_048_576,
            write_timeout_seconds=0.1,
            executable=Path("/missing/billiards-performance-audit-timing-summary"),
        )
        await bridge.start()
        assert bridge.try_record({"record_type": "sample"}) is False
        result = await bridge.finish([], state_trace_truncated=False)
        metadata = bridge.failure_metadata()
        await bridge.close()
        return result, metadata

    result, failure_metadata = asyncio.run(run_missing_executable())
    assert result["summary"]["status"] == "partial"
    assert "does not exist" in failure_metadata["failure"]


def _trajectory_bridge_event(sequence: int, source_frame: int) -> dict[str, Any]:
    return {
        "stream_sequence": sequence,
        "source_frame_number": source_frame,
        "vision_instance_id": "vision-a",
        "source_process_id": 17,
        "source_publication_wall_age_estimate_ms": float(sequence),
        "source_publication_to_client_receipt_ms": float(sequence) / 2,
        "clock_comparable": True,
    }


def test_rust_trajectory_summary_bridge_runs_native_protocol_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE",
        str(_rust_trajectory_summary_executable()),
    )

    async def run_bridge() -> tuple[dict[str, Any], dict[str, Any]]:
        bridge = collector.RustTrajectorySummaryBridge(
            maximum_samples=8,
            maximum_records=16,
            queue_capacity=8,
            write_timeout_seconds=2.0,
            executable=_rust_trajectory_summary_executable(),
        )
        await bridge.start()
        assert bridge.try_record(_trajectory_bridge_event(10, 20))
        assert bridge.try_record(_trajectory_bridge_event(12, 22))
        summary = await bridge.finish(2.0, exporter_loss_count=3)
        metadata = bridge.failure_metadata()
        await bridge.close()
        assert summary is not None
        return summary, metadata

    summary, metadata = asyncio.run(run_bridge())

    assert metadata["status"] == "complete"
    assert metadata["return_code"] == 0
    assert summary["trajectory_messages_received"] == 2
    assert summary["trajectory_messages_per_second"] == 1.0
    assert summary["trajectory_sequence_gaps"] == 1
    assert summary["trajectory_source_sequence_gaps"] == 1
    assert summary["trajectory_exporter_dropped_records"] == 3
    assert summary["trajectory_bridge_dropped_records"] == 0
    assert summary["trajectory_source_publication_to_client_receipt_ms"]["count"] == 2


def test_rust_trajectory_summary_bridge_bounds_records_and_reports_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE",
        str(_rust_trajectory_summary_executable()),
    )

    async def run_bridge() -> tuple[dict[str, Any], dict[str, Any]]:
        bridge = collector.RustTrajectorySummaryBridge(
            maximum_samples=8,
            maximum_records=4,
            queue_capacity=4,
            write_timeout_seconds=2.0,
            executable=_rust_trajectory_summary_executable(),
        )
        await bridge.start()
        accepted = [
            bridge.try_record(_trajectory_bridge_event(1, 1)),
            bridge.try_record(_trajectory_bridge_event(2, 2)),
            bridge.try_record(_trajectory_bridge_event(3, 3)),
        ]
        summary = await bridge.finish(1.0)
        metadata = bridge.failure_metadata()
        await bridge.close()
        assert summary is not None
        return {"accepted": accepted, "summary": summary}, metadata

    result, metadata = asyncio.run(run_bridge())

    assert result["accepted"] == [True, True, False]
    assert result["summary"]["trajectory_messages_received"] == 2
    assert result["summary"]["trajectory_bridge_dropped_records"] == 1
    assert metadata["callback_rejected_records"] == 1
    assert metadata["status"] == "partial"


def test_rust_trajectory_summary_bridge_bounds_queue_and_keeps_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedStdin:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.records: list[bytes] = []

        def write(self, record: bytes) -> None:
            self.records.append(record)

        async def drain(self) -> None:
            self.entered.set()
            await self.release.wait()

        def close(self) -> None:
            pass

    class DelayedProcess:
        def __init__(self) -> None:
            self.stdin = DelayedStdin()
            self.returncode: int | None = None

        def terminate(self) -> None:
            self.returncode = -15

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    fake_process = DelayedProcess()
    monkeypatch.setenv(
        "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE", os.sys.executable
    )

    async def create_delayed_process(*_args: Any, **_kwargs: Any) -> DelayedProcess:
        return fake_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_delayed_process)

    async def run_bridge() -> dict[str, Any]:
        bridge = collector.RustTrajectorySummaryBridge(
            maximum_samples=8,
            maximum_records=8,
            queue_capacity=1,
            write_timeout_seconds=1.0,
            executable=Path(os.sys.executable),
        )
        await bridge.start()
        await asyncio.wait_for(fake_process.stdin.entered.wait(), timeout=1.0)
        accepted = bridge.try_record(_trajectory_bridge_event(1, 1))
        rejected = bridge.try_record(_trajectory_bridge_event(2, 2))

        ticked = asyncio.Event()

        async def event_loop_tick() -> None:
            await asyncio.sleep(0.01)
            ticked.set()

        tick_task = asyncio.create_task(event_loop_tick())
        await asyncio.wait_for(ticked.wait(), timeout=0.5)
        await tick_task
        await bridge.close()
        return {
            "accepted": accepted,
            "rejected": rejected,
            "metadata": bridge.failure_metadata(),
        }

    result = asyncio.run(run_bridge())
    assert result["accepted"] is True
    assert result["rejected"] is False
    assert result["metadata"]["callback_rejected_records"] == 1
    assert result["metadata"]["writer_dropped_records"] == 1
    assert result["metadata"]["status"] == "partial"


def test_rust_trajectory_summary_bridge_failure_leaves_python_summary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE",
        "/missing/billiards-performance-audit-trajectory-summary",
    )

    async def run_bridge() -> tuple[dict[str, Any] | None, dict[str, Any]]:
        bridge = collector.RustTrajectorySummaryBridge(
            maximum_samples=8,
            maximum_records=8,
            queue_capacity=2,
            write_timeout_seconds=0.1,
            executable=Path("/missing/billiards-performance-audit-trajectory-summary"),
        )
        await bridge.start()
        assert bridge.try_record(_trajectory_bridge_event(1, 10)) is False
        summary = await bridge.finish(1.0)
        metadata = bridge.failure_metadata()
        await bridge.close()
        return summary, metadata

    native_summary, metadata = asyncio.run(run_bridge())
    assert native_summary is None
    assert metadata["status"] == "partial"
    assert "missing or not executable" in metadata["failure"]

    # The caller can still produce the compatibility summary from the Python
    # accumulator when native startup fails; this keeps native tooling optional.
    fallback = TrajectoryAccumulator(maximum_samples=8)
    fallback.add(_trajectory_bridge_event(1, 10))
    assert fallback.summary(1.0)["trajectory_messages_received"] == 1


def test_collect_repetition_uses_native_trajectory_summary_and_skips_state_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _rust_trajectory_summary_executable()
    native_path = str(executable)
    missing_path = "/missing/billiards-performance-audit-trajectory-summary"
    clock_domain_id = "test-host:boot-a:monotonic"
    status_document = {
        "host_id": "test-host",
        "clock_domain_id": clock_domain_id,
        "audit_domains": {"state": True, "trajectory": True},
        "event_export": {"run_id": "run-a", "queue_depth": 0},
    }
    payloads_for_connection: list[dict[str, Any]] = []

    class FakeWebSocket:
        def __init__(self, payloads: list[dict[str, Any]]) -> None:
            self._messages = deque(
                [
                    json.dumps(
                        {"type": "connection", "data": {"client_id": "client-a"}}
                    ),
                    json.dumps({"type": "subscribed", "data": {}}),
                    *(json.dumps(payload) for payload in payloads),
                ]
            )

        async def recv(self) -> str:
            if self._messages:
                return self._messages.popleft()
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(self, _message: str) -> None:
            return None

    class FakeConnection:
        def __init__(self, websocket: FakeWebSocket) -> None:
            self._websocket = websocket

        async def __aenter__(self) -> FakeWebSocket:
            return self._websocket

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        collector.websockets,
        "connect",
        lambda *_args, **_kwargs: FakeConnection(
            FakeWebSocket(payloads_for_connection)
        ),
    )
    monkeypatch.setattr(
        collector, "_read_json_url", lambda *_args, **_kwargs: status_document
    )
    monkeypatch.setattr(
        collector,
        "_wait_for_audit_export_drain",
        lambda *_args, **_kwargs: status_document,
    )

    class FakeRecorder:
        def __init__(self) -> None:
            self.clock_domain_id = clock_domain_id

        def record(self, _event: dict[str, Any]) -> None:
            return None

        def status(self) -> dict[str, Any]:
            return {
                "dropped_events": 0,
                "export_limit_dropped_events": 0,
                "export_errors": 0,
            }

    recorder = FakeRecorder()
    original_resolver = collector._rust_audit_summary_executable
    resolved_binaries: list[str] = []

    def track_resolver(binary_name: str, configured_executable: str | None) -> Path:
        resolved_binaries.append(binary_name)
        return original_resolver(binary_name, configured_executable)

    monkeypatch.setattr(collector, "_rust_audit_summary_executable", track_resolver)

    async def collect_case(
        streams: list[str], payloads: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payloads_for_connection[:] = payloads
        return await collector.collect_repetition(
            "ws://127.0.0.1:8000/ws",
            streams=streams,
            warmup_seconds=0,
            measured_seconds=0.02,
            repetition=1,
            recorder=recorder,  # type: ignore[arg-type]
            progress=False,
            history_size=8,
            heartbeat_interval_seconds=10,
        )

    async def run_cases() -> tuple[dict[str, Any], dict[str, Any]]:
        monkeypatch.setenv(
            "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE", native_path
        )
        trajectory_payload = {
            "type": "trajectory",
            "data": {
                "sequence": 1,
                "source_frame_number": 42,
                "source_timestamp": time.time(),
                "source_timestamp_kind": "shared_memory_publication",
                "source_observations": {
                    "vision_instance_id": "vision-a",
                    "source_process_id": 17,
                    "source_publication_monotonic_ns": time.perf_counter_ns()
                    - 1_000_000,
                },
            },
        }
        trajectory_summary = await collect_case(
            ["state", "trajectory"], [trajectory_payload]
        )

        monkeypatch.setenv(
            "PERFORMANCE_AUDIT_TRAJECTORY_SUMMARY_EXECUTABLE", missing_path
        )
        state_payload = {
            "type": "state",
            "data": {
                "sequence": 1,
                "balls": [],
                "observation_status": {
                    "source_frame_number": 7,
                    "source_timestamp": time.time(),
                    "source_publication_monotonic_ns": 1,
                    "vision_instance_id": "vision-a",
                    "source_process_id": 17,
                    "component_observation_status": {"balls": "observed"},
                },
            },
        }
        state_summary = await collect_case(["state"], [state_payload])
        return trajectory_summary, state_summary

    trajectory_summary, state_summary = asyncio.run(run_cases())

    assert resolved_binaries == ["billiards-performance-audit-trajectory-summary"]
    assert trajectory_summary["trajectory_summary_engine"] == "rust"
    assert trajectory_summary["trajectory_summary_differential"] == {
        "status": "matched",
        "mismatched_fields": [],
    }
    assert trajectory_summary["trajectory_summary_bridge"]["status"] == "complete"
    assert trajectory_summary["trajectory_messages_received"] == 1
    assert state_summary["state_messages_received"] == 1
    assert "trajectory_summary_engine" not in state_summary


def test_rust_live_qualification_preserves_partial_state_and_process_diagnostics(
    tmp_path: Path,
) -> None:
    class InterruptingSupervisor:
        def __init__(self, _settings: Any, publisher: Any) -> None:
            self.publisher = publisher
            self._status = SimpleNamespace(
                ready=False,
                process_status="not_started",
                process_return_code=None,
                failure=None,
                source_health="not_started",
                stderr_tail="",
                ingress_accepted_snapshots=0,
                ingress_queue_replacements=0,
                state_snapshots_published=0,
            )
            self.task: asyncio.Task[None] | None = None

        @property
        def status(self) -> Any:
            return self._status

        async def start(self) -> Any:
            await self.publisher.publish(_qualification_scene(1))
            self._status.ready = True
            self._status.process_status = "running"
            self._status.source_health = "healthy"
            self.task = asyncio.create_task(self._interrupt())
            return self._status

        async def _interrupt(self) -> None:
            await asyncio.sleep(0.015)
            await self.publisher.publish(_qualification_scene(2))
            self._status.ready = False
            self._status.process_status = "failed"
            self._status.process_return_code = 9
            self._status.failure = "capture helper exited"
            self._status.source_health = "process_failed"
            self._status.stderr_tail = "camera source stopped unexpectedly"

        async def close(self) -> None:
            if self.task is not None:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self._status.ready = False
            self._status.process_status = "stopped"

    settings_path = _rust_live_settings_file(tmp_path)
    args = build_parser().parse_args(
        [
            *_rust_live_profile_args(settings_path),
            "--output-root",
            str(tmp_path / "audits"),
            "--warmup",
            "0",
            "--duration",
            "0.1",
            "--repetitions",
            "1",
            "--rust-live-status-poll-interval",
            "0.1",
            "--quiet",
        ]
    )
    run_directory = asyncio.run(
        collector._run_rust_live_qualification(
            args,
            capture_transport_settings=None,
            supervisor_factory=InterruptingSupervisor,
        )
    )

    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    repetition = summary["repetitions"][0]
    assert summary["qualification_status"] == "partial_measurement"
    assert repetition["measurement_status"] == "partial"
    assert repetition["measurement_failure"]["process_return_code"] == 9
    assert (
        repetition["measurement_failure"]["stderr_tail"]
        == "camera source stopped unexpectedly"
    )
    assert repetition["state"]["state_messages_received"] == 1
    assert repetition["state_trace"][0]["source_frame_id"] == "frame-0002"
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "ended early" in report
    assert "camera source stopped unexpectedly" in report


def test_rust_live_qualification_stops_supervisor_after_startup_failure(
    tmp_path: Path,
) -> None:
    class FailedSupervisor:
        def __init__(self, _settings: Any, _publisher: Any) -> None:
            self.closed = False
            self.status = SimpleNamespace(ready=False, failure="camera unavailable")

        async def start(self) -> Any:
            raise RuntimeError("simulated camera startup failure")

        async def close(self) -> None:
            self.closed = True

    settings_path = _rust_live_settings_file(tmp_path)
    args = build_parser().parse_args(
        [
            *_rust_live_profile_args(settings_path),
            "--output-root",
            str(tmp_path / "audits"),
            "--warmup",
            "0",
            "--duration",
            "0.01",
            "--repetitions",
            "1",
            "--quiet",
        ]
    )
    supervisors: list[FailedSupervisor] = []

    def supervisor_factory(settings: Any, publisher: Any) -> FailedSupervisor:
        supervisor = FailedSupervisor(settings, publisher)
        supervisors.append(supervisor)
        return supervisor

    with pytest.raises(RuntimeError, match="simulated camera startup failure"):
        asyncio.run(
            collector._run_rust_live_qualification(
                args,
                capture_transport_settings=None,
                supervisor_factory=supervisor_factory,
            )
        )

    assert len(supervisors) == 1
    assert supervisors[0].closed is True
    run_directories = list((tmp_path / "audits").iterdir())
    assert len(run_directories) == 1
    failure = json.loads((run_directories[0] / "failure.json").read_text())
    assert failure["outcome"] == "failed"
    assert failure["error"] == "simulated camera startup failure"
    assert failure["event_export"]["closed"] is True
    assert (
        "not an accuracy or performance result"
        in (run_directories[0] / "report.md").read_text()
    )


def test_rust_live_state_exporter_loss_is_observable(tmp_path: Path) -> None:
    def failed_writer(_event: dict[str, Any]) -> None:
        raise OSError("simulated state trace export failure")

    recorder = AsyncAuditRecorder(
        tmp_path / "failed-state-events.jsonl",
        writer=failed_writer,
    )
    broadcaster = collector.RustLiveQualificationBroadcaster(
        repetition=1,
        recorder=recorder,
        history_size=4,
    )
    broadcaster.begin_measurement()

    asyncio.run(
        broadcaster.broadcast_game_state(
            observation_status={
                "observation_identity": {
                    "engine_instance_id": "engine-a",
                    "session_id": "session-a",
                    "frame_id": "frame-0001",
                },
                "source_frame_id": "frame-0001",
                "source_timestamp": "0",
                "source_timestamp_kind": "monotonic:clock:camera_read_complete",
                "source_timestamp_is_exposure": False,
            },
            balls=[],
        )
    )
    broadcaster.end_measurement()
    recorder.close()

    assert broadcaster.measurement.summary(1.0)["complete_state_messages"] == 1
    assert recorder.status()["export_errors"] == 1
