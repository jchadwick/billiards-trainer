"""Tests for the unauthenticated, latest-only Rust scene ingress."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from backend.perception.capture_configuration import (
    CaptureSettingsSnapshot,
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
    resolve_capture_settings,
)
from backend.perception.scene_ingress import (
    SceneIngressConfig,
    SceneIngressError,
    SceneSnapshotIngress,
)

CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "perception/contracts/v1"
ACK_CONTRACT_ROOT = (
    Path(__file__).resolve().parents[3] / "perception/contracts/capture-session-ack/v1"
)
ACK_V2_CONTRACT_ROOT = ACK_CONTRACT_ROOT.parent / "v2"
PUBLIC_STATE_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "perception/contracts/public-state"
)


class EmptyConfiguration:
    """Select the numeric camera represented by live scene-test fixtures."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = copy.deepcopy(values) if values is not None else {}
        vision = self.values.setdefault("vision", {})
        assert isinstance(vision, dict)
        camera = vision.setdefault("camera", {})
        assert isinstance(camera, dict)
        camera.setdefault("video_file_path", "")

    def get_all(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)


def _fixture() -> dict[str, Any]:
    fixture = json.loads(
        (CONTRACT_ROOT / "scene-rust-tracked-success.json").read_text(encoding="utf-8")
    )
    if not isinstance(fixture, dict):
        raise AssertionError("scene fixture root must be a JSON object")
    return fixture


def _public_state_delivery_fixture() -> dict[str, Any]:
    fixture = json.loads(
        (PUBLIC_STATE_FIXTURE_ROOT / "number-8.json").read_text(encoding="utf-8")
    )
    scene = fixture["scene"]
    payload = fixture["expected"]
    payload["observation_status"].update(
        freshness_status="fresh", localization_age_ns="0"
    )
    return {
        "delivery_type": "billiards_public_state_delivery",
        "delivery_schema_version": 1,
        "source_snapshot": scene,
        "current_state": {
            "payload": payload,
            "epoch": 1,
            "updated_monotonic_ns": 1_000_000,
            "localization_received_monotonic_ns": 1_000_000,
            "localization_age_ns": 0,
            "ball_age_ns": {ball["id"]: 0 for ball in payload["balls"]},
            "freshness": "fresh",
        },
    }


def _next_scene(
    source: dict[str, Any], sequence: int, frame_id: str | None = None
) -> dict[str, Any]:
    scene = copy.deepcopy(source)
    scene["transport_sequence"] = sequence
    scene["snapshot_revision"] = sequence
    new_frame_id = frame_id or f"frame-{sequence}"
    scene["source"]["frame_id"] = new_frame_id
    scene["source"]["timestamp"]["ticks"] = str(912345670000 + sequence)
    scene["published_at"]["ticks"] = str(912367890000 + sequence)
    for ball in scene["balls"]:
        ball["last_observation"]["frame_id"] = new_frame_id
        ball["last_observation"]["timestamp"]["ticks"] = str(912345670000 + sequence)
    return scene


def _frame(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _capture_acknowledgement() -> dict[str, Any]:
    value = json.loads(
        (ACK_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise AssertionError("capture acknowledgement fixture must be an object")
    return value


def _capture_settings() -> CaptureSettingsSnapshotV1:
    snapshot = resolve_capture_settings(EmptyConfiguration(), schema_version=1)
    assert isinstance(snapshot, CaptureSettingsSnapshotV1)
    return snapshot


def _capture_settings_v2() -> CaptureSettingsSnapshotV2:
    snapshot = resolve_capture_settings(
        EmptyConfiguration(
            {"perception": {"capture": {"output_resolution": [1280, 720]}}}
        ),
        schema_version=2,
    )
    assert isinstance(snapshot, CaptureSettingsSnapshotV2)
    return snapshot


def _capture_settings_v3() -> CaptureSettingsSnapshotV3:
    snapshot = resolve_capture_settings(EmptyConfiguration())
    assert isinstance(snapshot, CaptureSettingsSnapshotV3)
    return snapshot


def _capture_acknowledgement_v2(
    settings: CaptureSettingsSnapshotV2 | CaptureSettingsSnapshotV3,
    session_id: str,
) -> dict[str, Any]:
    value = json.loads(
        (ACK_V2_CONTRACT_ROOT / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise AssertionError("v2 capture acknowledgement fixture must be an object")
    value["request_sha256"] = hashlib.sha256(
        settings.to_json().encode("utf-8")
    ).hexdigest()
    report = value["source_report"]
    report["session_id"] = session_id
    report["requested_profile"] = settings.requested["requested_profile"]
    report["negotiated_profile"] = settings.requested["expected_negotiated_profile"]
    report["applied_frame_geometry"]["camera_profile_id"] = settings.requested[
        "camera_profile_id"
    ]
    report["output_frame_geometry"] = settings.requested["output_frame_geometry"]
    return value


async def _read_json_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    header = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
    response_size = struct.unpack(">I", header)[0]
    response = json.loads(
        await asyncio.wait_for(reader.readexactly(response_size), timeout=1.0)
    )
    if not isinstance(response, dict):
        raise AssertionError("scene ingress response must be an object")
    return response


def _settings(**overrides: Any) -> SceneIngressConfig:
    values: dict[str, Any] = {
        "bind_address": "127.0.0.1",
        "port": 0,
        "maximum_message_bytes": 256 * 1024,
        "read_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return SceneIngressConfig(**values)


async def _connect(
    ingress: SceneSnapshotIngress,
    expected_last_sequence: int | None = None,
    identity: tuple[str, str] | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection(*ingress.address)
    fixture = _fixture()
    producer_identity = identity or (
        fixture["engine_instance_id"],
        fixture["source"]["session_id"],
    )
    writer.write(
        _frame(
            {
                "type": "hello",
                "protocol_version": 2,
                "engine_instance_id": producer_identity[0],
                "source_session_id": producer_identity[1],
            }
        )
    )
    await writer.drain()
    header = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
    response_size = struct.unpack(">I", header)[0]
    response = json.loads(
        await asyncio.wait_for(reader.readexactly(response_size), timeout=1.0)
    )
    assert response == {
        "type": "ready",
        "protocol_version": 2,
        "last_transport_sequence": expected_last_sequence,
    }
    return reader, writer


async def _wait_for_count(
    ingress: SceneSnapshotIngress, expected: int, *, rejected: bool = False
) -> None:
    for _ in range(100):
        actual = (
            ingress.stats.rejected_messages
            if rejected
            else ingress.stats.accepted_snapshots
        )
        if actual >= expected:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("scene ingress did not reach the expected message count")


def test_scene_ingress_configuration_requires_a_bounded_loopback_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        _settings(bind_address="192.0.2.10")
    with pytest.raises(ValueError, match="port"):
        _settings(port=65536)
    with pytest.raises(ValueError, match="message size"):
        _settings(maximum_message_bytes=0)
    with pytest.raises(ValueError, match="handshake"):
        _settings(maximum_message_bytes=64)
    with pytest.raises(ValueError, match="timeout"):
        _settings(read_timeout_seconds=float("nan"))


def test_scene_ingress_rejects_non_integer_capture_schema_version() -> None:
    malformed_settings = replace(cast("Any", _capture_settings()), schema_version=1.0)

    with pytest.raises(ValueError, match="version does not match"):
        _settings(capture_settings=malformed_settings)


def test_loopback_ingress_accepts_protocol_handshake_and_delivers_scene() -> None:
    async def run() -> tuple[dict[str, Any], Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        reader, writer = await _connect(ingress)
        del reader
        writer.write(_frame(_fixture()))
        await writer.drain()
        await _wait_for_count(ingress, 1)
        received = await ingress.receive_latest()
        writer.close()
        await writer.wait_closed()
        stats = ingress.stats
        await ingress.close()
        return received.snapshot, (received, stats)

    snapshot, (received, stats) = asyncio.run(run())

    assert snapshot["update_kind"] == "observation"
    assert snapshot["source"]["frame_id"] == "frame-0001"
    assert received.received_monotonic_ns > 0
    assert received.received_at.tzinfo is not None
    assert received.received_at.utcoffset() is not None
    assert len(received.payload_sha256) == 64
    assert received.peer_address[0] == "127.0.0.1"
    assert received.snapshot is not snapshot
    assert stats.accepted_snapshots == 1
    assert stats.producer_identity == (
        snapshot["engine_instance_id"],
        snapshot["source"]["session_id"],
    )


def test_slow_consumer_gets_only_latest_complete_snapshot() -> None:
    async def run() -> tuple[dict[str, Any], Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        base = _fixture()
        for sequence in range(1, 4):
            writer.write(_frame(_next_scene(base, sequence)))
        await writer.drain()
        await _wait_for_count(ingress, 3)
        latest = await ingress.receive_latest()
        writer.close()
        await writer.wait_closed()
        stats = ingress.stats
        await ingress.close()
        return latest.snapshot, stats

    latest, stats = asyncio.run(run())

    assert latest["transport_sequence"] == 3
    assert latest["source"]["frame_id"] == "frame-3"
    assert stats.accepted_snapshots == 3
    assert stats.queue_replacements == 2
    assert stats.sequence_gap_total == 0


def test_opt_in_transport_audit_excludes_warmup_and_measures_queue_pressure() -> None:
    async def run() -> tuple[Any, Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        base = _fixture()
        writer.write(_frame(_next_scene(base, 1)))
        await writer.drain()
        await _wait_for_count(ingress, 1)
        await ingress.receive_latest()

        ingress.enable_transport_audit()
        payload_sizes = []
        for sequence in (2, 3, 5):
            payload = _frame(_next_scene(base, sequence))
            payload_sizes.append(len(payload) - 4)
            writer.write(payload)
        await writer.drain()
        # Include the warm-up frame in the ingress total. Waiting for only three
        # frames can return after the first two measured frames, racing the
        # final queue and audit assertions below.
        await _wait_for_count(ingress, 4)
        await ingress.receive_latest()
        summary = ingress.finish_transport_audit()
        assert ingress.finish_transport_audit().status == "disabled"
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return summary, payload_sizes

    summary, payload_sizes = asyncio.run(run())

    assert summary.status == "available"
    assert summary.accepted_snapshots == 3
    assert summary.payload_bytes_total == sum(payload_sizes)
    assert summary.payload_bytes_minimum == min(payload_sizes)
    assert summary.payload_bytes_mean == pytest.approx(sum(payload_sizes) / 3)
    assert summary.payload_bytes_maximum == max(payload_sizes)
    assert summary.sequence_gaps == 1
    assert summary.queue_capacity == 1
    assert summary.queue_depth_high_water == 1
    assert summary.queue_depth_zero_before_enqueue == 1
    assert summary.queue_depth_one_before_enqueue == 2
    assert summary.queue_replacements == 2
    assert summary.queue_replacement_fraction == pytest.approx(2 / 3)


def test_transport_audit_is_disabled_until_explicitly_enabled() -> None:
    ingress = SceneSnapshotIngress(_settings())

    summary = ingress.finish_transport_audit()

    assert summary.status == "disabled"
    assert summary.accepted_snapshots == 0
    assert summary.payload_bytes_total == 0
    assert summary.queue_replacements == 0


def test_ingress_counts_skipped_transport_sequences_without_replaying_history() -> None:
    async def run() -> tuple[Any, Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        base = _fixture()
        writer.write(_frame(_next_scene(base, 1)))
        writer.write(_frame(_next_scene(base, 4)))
        await writer.drain()
        await _wait_for_count(ingress, 2)
        latest = await ingress.receive_latest()
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return latest.snapshot, stats

    latest, stats = asyncio.run(run())

    assert latest["transport_sequence"] == 4
    assert stats.sequence_gap_total == 2
    assert stats.queue_replacements == 1


def test_same_launched_producer_can_reconnect_without_resetting_sequence_identity() -> (
    None
):
    async def run() -> tuple[dict[str, Any], Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, first_writer = await _connect(ingress)
        first_writer.write(_frame(_fixture()))
        await first_writer.drain()
        await _wait_for_count(ingress, 1)
        first_writer.close()
        await first_writer.wait_closed()

        for _ in range(100):
            if not ingress.stats.connection_active:
                break
            await asyncio.sleep(0.005)
        _, second_writer = await _connect(ingress, expected_last_sequence=1)
        second_writer.write(_frame(_next_scene(_fixture(), 2)))
        await second_writer.drain()
        await _wait_for_count(ingress, 2)
        latest = await ingress.receive_latest()
        stats = ingress.stats
        second_writer.close()
        await second_writer.wait_closed()
        await ingress.close()
        return latest.snapshot, stats

    latest, stats = asyncio.run(run())

    assert latest["transport_sequence"] == 2
    assert stats.connection_count == 2
    assert stats.rejected_connections == 0
    assert stats.producer_identity == ("engine-rust-stage2-a", "replay-session-01")


def test_new_producer_epoch_resets_watermark_and_discards_previous_latest_scene() -> (
    None
):
    async def run() -> tuple[dict[str, Any], Any]:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, first_writer = await _connect(ingress)
        first_writer.write(_frame(_fixture()))
        await first_writer.drain()
        await _wait_for_count(ingress, 1)
        first_writer.close()
        await first_writer.wait_closed()

        for _ in range(100):
            if not ingress.stats.connection_active:
                break
            await asyncio.sleep(0.005)
        restarted_identity = ("engine-rust-restarted", "capture-session-02")
        _, restarted_writer = await _connect(
            ingress, expected_last_sequence=None, identity=restarted_identity
        )
        restarted_scene = _fixture()
        restarted_scene["engine_instance_id"] = restarted_identity[0]
        restarted_scene["source"]["session_id"] = restarted_identity[1]
        # Restarted producers begin their sequence counters over at one.
        restarted_writer.write(_frame(restarted_scene))
        await restarted_writer.drain()
        await _wait_for_count(ingress, 2)
        latest = await ingress.receive_latest()
        stats = ingress.stats
        restarted_writer.close()
        await restarted_writer.wait_closed()
        await ingress.close()
        return latest.snapshot, stats

    latest, stats = asyncio.run(run())

    assert latest["transport_sequence"] == 1
    assert latest["engine_instance_id"] == "engine-rust-restarted"
    assert latest["source"]["session_id"] == "capture-session-02"
    assert stats.producer_epoch_resets == 1
    assert stats.discarded_epoch_snapshots == 1
    assert stats.producer_identity == ("engine-rust-restarted", "capture-session-02")


def test_unsupported_protocol_version_never_admits_scene_data() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 1,
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                }
            )
        )
        await writer.drain()
        await asyncio.wait_for(reader.read(), timeout=1.0)
        for _ in range(100):
            if ingress.stats.rejected_handshakes:
                break
            await asyncio.sleep(0.005)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.rejected_handshakes == 1
    assert stats.accepted_snapshots == 0
    assert stats.producer_identity is None


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_sequence",
        "identity_change",
        "duplicate_observation",
        "stale_timestamp",
    ],
)
def test_ingress_rejects_stale_identity_and_duplicate_observations(
    failure: str,
) -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        first = _fixture()
        writer.write(_frame(first))
        if failure == "duplicate_sequence":
            stale = _next_scene(first, 1, "frame-stale")
        elif failure == "identity_change":
            stale = _next_scene(first, 2)
            stale["engine_instance_id"] = "different-engine"
        elif failure == "stale_timestamp":
            stale = _next_scene(first, 2)
            stale["source"]["timestamp"]["ticks"] = first["source"]["timestamp"][
                "ticks"
            ]
        else:
            stale = _next_scene(first, 2, first["source"]["frame_id"])
        for ball in stale["balls"]:
            ball["last_observation"]["timestamp"]["ticks"] = stale["source"][
                "timestamp"
            ]["ticks"]
        writer.write(_frame(stale))
        await writer.drain()
        await _wait_for_count(ingress, 1, rejected=True)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.accepted_snapshots == 1
    assert stats.rejected_messages == 1
    assert stats.last_transport_sequence == 1


@pytest.mark.parametrize(
    "malformed_frame",
    [b"\x00\x00", struct.pack(">I", 3) + b"{}", struct.pack(">I", 0)],
    ids=["truncated-header", "truncated-body", "zero-length"],
)
def test_ingress_rejects_truncated_and_zero_length_frames(
    malformed_frame: bytes,
) -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        writer.write(malformed_frame)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await _wait_for_count(ingress, 1, rejected=True)
        stats = ingress.stats
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.rejected_messages == 1
    assert stats.accepted_snapshots == 0


def test_ingress_rejects_oversized_frame_before_reading_its_body() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings(maximum_message_bytes=512))
        await ingress.start()
        reader, writer = await _connect(ingress)
        del reader
        writer.write(struct.pack(">I", 513))
        await writer.drain()
        for _ in range(100):
            if ingress.stats.rejected_messages:
                break
            await asyncio.sleep(0.005)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.rejected_messages == 1
    assert stats.accepted_snapshots == 0


def test_idle_producer_is_counted_as_a_read_timeout() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings(read_timeout_seconds=0.05))
        await ingress.start()
        _, writer = await _connect(ingress)
        for _ in range(100):
            if ingress.stats.read_timeouts:
                break
            await asyncio.sleep(0.005)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.read_timeouts == 1
    assert stats.rejected_messages == 0


def test_ingress_rejects_schema_invalid_scene_without_queueing_it() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        invalid = _fixture()
        invalid["unexpected"] = "not in v1"
        writer.write(_frame(invalid))
        await writer.drain()
        await _wait_for_count(ingress, 1, rejected=True)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.rejected_messages == 1
    assert stats.accepted_snapshots == 0
    assert stats.producer_identity is None


def test_ingress_rejects_non_finite_json_numbers_before_advancing_sequence() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        _, writer = await _connect(ingress)
        invalid = _fixture()
        invalid["balls"][0]["measurement"]["image_center_px"]["x"] = float("nan")
        writer.write(_frame(invalid))
        await writer.drain()
        await _wait_for_count(ingress, 1, rejected=True)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return stats

    stats = asyncio.run(run())

    assert stats.accepted_snapshots == 0
    assert stats.rejected_messages == 1
    assert stats.last_transport_sequence is None
    assert stats.producer_identity is None


def test_ingress_rejects_concurrent_producer_connections() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings())
        await ingress.start()
        first_reader, first_writer = await _connect(ingress)
        second_reader, second_writer = await asyncio.open_connection(*ingress.address)
        await asyncio.wait_for(second_reader.read(), timeout=1.0)
        for _ in range(100):
            if ingress.stats.rejected_connections:
                break
            await asyncio.sleep(0.005)
        stats = ingress.stats
        first_writer.close()
        second_writer.close()
        await first_writer.wait_closed()
        await second_writer.wait_closed()
        await ingress.close()
        return stats, first_reader

    stats, _ = asyncio.run(run())

    assert stats.connection_count == 1
    assert stats.rejected_connections == 1


def test_live_v3_handshake_validates_ack_before_ready_and_scene_admission() -> None:
    async def run() -> tuple[dict[str, Any], Any, Any]:
        ingress = SceneSnapshotIngress(_settings(capture_settings=_capture_settings()))
        await ingress.start()
        scene = _fixture()
        scene["source"]["width_px"] = 1920
        scene["source"]["height_px"] = 1080
        acknowledgement = _capture_acknowledgement()
        acknowledgement["source_report"]["session_id"] = scene["source"]["session_id"]
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": scene["source"]["session_id"],
                    "capture_session_ack": acknowledgement,
                }
            )
        )
        await writer.drain()
        ready = await _read_json_frame(reader)
        writer.write(_frame(scene))
        await writer.drain()
        await _wait_for_count(ingress, 1)
        received = await ingress.receive_latest()
        accepted_ack = ingress.capture_session_acknowledgement
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return ready, received.snapshot, (accepted_ack, stats)

    ready, scene, (accepted_ack, stats) = asyncio.run(run())

    assert ready == {
        "type": "ready",
        "protocol_version": 3,
        "configuration_revision": 1,
        "request_sha256": _capture_acknowledgement()["request_sha256"],
        "last_transport_sequence": None,
    }
    assert scene["source"]["session_id"] == accepted_ack.session_id
    assert accepted_ack.source_report["device_id"] == "device-index:0"
    assert stats.accepted_snapshots == 1


@pytest.mark.parametrize("schema_version", [2, 3])
def test_live_ingress_uses_v2_ack_schema_for_v2_or_v3_capture_settings(
    schema_version: int,
) -> None:
    async def run() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
        settings = (
            _capture_settings_v2() if schema_version == 2 else _capture_settings_v3()
        )
        ingress = SceneSnapshotIngress(_settings(capture_settings=settings))
        await ingress.start()
        scene = _fixture()
        output_geometry = settings.requested["output_frame_geometry"]
        assert isinstance(output_geometry, dict)
        scene["source"]["width_px"] = output_geometry["width_px"]
        scene["source"]["height_px"] = output_geometry["height_px"]
        source_session_id = scene["source"]["session_id"]

        # The v1 report shape must not be accepted for v2/v3 output geometry.
        legacy_reader, legacy_writer = await asyncio.open_connection(*ingress.address)
        legacy_ack = _capture_acknowledgement()
        legacy_ack["source_report"]["session_id"] = source_session_id
        legacy_writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": source_session_id,
                    "capture_session_ack": legacy_ack,
                }
            )
        )
        await legacy_writer.drain()
        legacy_closed = await asyncio.wait_for(legacy_reader.read(), timeout=1.0)
        legacy_writer.close()
        await legacy_writer.wait_closed()

        acknowledgement = _capture_acknowledgement_v2(settings, source_session_id)
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": source_session_id,
                    "capture_session_ack": acknowledgement,
                }
            )
        )
        await writer.drain()
        ready = await _read_json_frame(reader)
        writer.write(_frame(scene))
        await writer.drain()
        await _wait_for_count(ingress, 1)
        received = await ingress.receive_latest()
        accepted_ack = ingress.capture_session_acknowledgement
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return ready, received.snapshot, accepted_ack, (legacy_closed, stats)

    ready, scene, accepted_ack, (legacy_closed, stats) = asyncio.run(run())

    assert legacy_closed == b""
    assert ready["protocol_version"] == 3
    assert ready["request_sha256"] == accepted_ack.request_sha256
    expected_geometry = (
        _capture_settings_v2().requested["output_frame_geometry"]
        if schema_version == 2
        else _capture_settings_v3().requested["output_frame_geometry"]
    )
    assert accepted_ack.source_report["output_frame_geometry"] == expected_geometry
    assert scene["source"]["session_id"] == accepted_ack.session_id
    assert stats.rejected_handshakes == 1
    assert stats.accepted_snapshots == 1


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_live_ingress_rejects_scene_geometry_that_differs_from_capture_ack(
    schema_version: int,
) -> None:
    async def run() -> Any:
        if schema_version == 1:
            settings: CaptureSettingsSnapshot = _capture_settings()
        elif schema_version == 2:
            settings = _capture_settings_v2()
        else:
            settings = _capture_settings_v3()
        ingress = SceneSnapshotIngress(_settings(capture_settings=settings))
        await ingress.start()
        scene = _fixture()
        requested = settings.requested
        geometry = requested.get("output_frame_geometry")
        if not isinstance(geometry, dict):
            geometry = requested["expected_negotiated_profile"]
        assert isinstance(geometry, dict)
        scene["source"]["width_px"] = geometry["width_px"] + 1
        scene["source"]["height_px"] = geometry["height_px"]
        source_session_id = scene["source"]["session_id"]
        if schema_version == 1:
            acknowledgement = _capture_acknowledgement()
        else:
            assert isinstance(
                settings, (CaptureSettingsSnapshotV2, CaptureSettingsSnapshotV3)
            )
            acknowledgement = _capture_acknowledgement_v2(settings, source_session_id)
        acknowledgement["source_report"]["session_id"] = source_session_id
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": source_session_id,
                    "capture_session_ack": acknowledgement,
                }
            )
        )
        await writer.drain()
        ready = await _read_json_frame(reader)
        writer.write(_frame(scene))
        await writer.drain()
        await _wait_for_count(ingress, 1, rejected=True)
        stats = ingress.stats
        accepted_ack = ingress.capture_session_acknowledgement
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return ready, stats, accepted_ack

    ready, stats, accepted_ack = asyncio.run(run())

    assert ready["protocol_version"] == 3
    assert accepted_ack is not None
    assert stats.rejected_messages == 1
    assert stats.accepted_snapshots == 0
    assert stats.last_transport_sequence is None


def test_live_configuration_message_limit_includes_the_acknowledgement_hello() -> None:
    with pytest.raises(ValueError, match="cannot carry the ingress handshake"):
        _settings(
            capture_settings=_capture_settings(),
            maximum_message_bytes=1024,
        )


@pytest.mark.parametrize("protocol_version", [2, 3])
def test_live_handshake_rejects_legacy_or_mismatched_ack_before_ready(
    protocol_version: int,
) -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings(capture_settings=_capture_settings()))
        await ingress.start()
        scene = _fixture()
        acknowledgement = _capture_acknowledgement()
        acknowledgement["source_report"]["session_id"] = scene["source"]["session_id"]
        if protocol_version == 3:
            acknowledgement["request_sha256"] = "0" * 64
        hello = {
            "type": "hello",
            "protocol_version": protocol_version,
            "engine_instance_id": scene["engine_instance_id"],
            "source_session_id": scene["source"]["session_id"],
        }
        if protocol_version == 3:
            hello["capture_session_ack"] = acknowledgement
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(_frame(hello))
        await writer.drain()
        closed = await asyncio.wait_for(reader.read(), timeout=1.0)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return closed, stats, ingress.capture_session_acknowledgement

    closed, stats, accepted_ack = asyncio.run(run())

    assert closed == b""
    assert stats.rejected_handshakes == 1
    assert accepted_ack is None


def test_live_ingress_accepts_rust_public_state_delivery_and_exposes_raw_scene() -> (
    None
):
    delivery = _public_state_delivery_fixture()
    scene = delivery["source_snapshot"]
    settings = _settings(
        capture_settings=_capture_settings_v3(),
        accept_public_state_delivery=True,
    )
    ingress = SceneSnapshotIngress(settings)
    ingress._begin_producer_epoch(
        (scene["engine_instance_id"], scene["source"]["session_id"]), None
    )

    async def run() -> None:
        ingress._accept_payload(
            json.dumps(delivery, separators=(",", ":")).encode("utf-8"),
            ("127.0.0.1", 12345),
            "native-state-test",
        )
        received = await ingress.receive_latest()
        assert received.snapshot == scene
        assert received.public_state_delivery == delivery
        assert ingress.stats.accepted_snapshots == 1

    asyncio.run(run())


def test_replay_native_state_delivery_uses_identity_handshake_without_live_ack() -> (
    None
):
    delivery = _public_state_delivery_fixture()
    scene = delivery["source_snapshot"]

    async def run() -> tuple[dict[str, Any], Any, Any]:
        ingress = SceneSnapshotIngress(_settings(accept_public_state_delivery=True))
        await ingress.start()
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 2,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": scene["source"]["session_id"],
                }
            )
        )
        await writer.drain()
        ready = await _read_json_frame(reader)
        writer.write(_frame(delivery))
        await writer.drain()
        await _wait_for_count(ingress, 1)
        received = await ingress.receive_latest()
        acknowledgement = ingress.capture_session_acknowledgement
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return ready, received, (acknowledgement, stats)

    ready, received, (acknowledgement, stats) = asyncio.run(run())

    assert ready == {
        "type": "ready",
        "protocol_version": 2,
        "last_transport_sequence": None,
    }
    assert received.snapshot == scene
    assert received.public_state_delivery == delivery
    assert acknowledgement is None
    assert stats.accepted_snapshots == 1


def test_live_native_delivery_still_requires_request_bound_capture_ack() -> None:
    async def run() -> tuple[bytes, Any]:
        ingress = SceneSnapshotIngress(
            _settings(
                capture_settings=_capture_settings_v3(),
                accept_public_state_delivery=True,
            )
        )
        await ingress.start()
        scene = _fixture()
        reader, writer = await asyncio.open_connection(*ingress.address)
        writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 2,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": scene["source"]["session_id"],
                }
            )
        )
        await writer.drain()
        closed = await asyncio.wait_for(reader.read(), timeout=1.0)
        stats = ingress.stats
        writer.close()
        await writer.wait_closed()
        await ingress.close()
        return closed, stats

    closed, stats = asyncio.run(run())

    assert closed == b""
    assert stats.rejected_handshakes == 1
    assert stats.accepted_snapshots == 0


def test_native_rust_ingress_rejects_raw_scene_without_state_delivery() -> None:
    scene = _fixture()
    ingress = SceneSnapshotIngress(
        _settings(
            capture_settings=_capture_settings_v3(),
            accept_public_state_delivery=True,
        )
    )
    ingress._begin_producer_epoch(
        (scene["engine_instance_id"], scene["source"]["session_id"]), None
    )
    with pytest.raises(SceneIngressError, match="required public-state delivery"):
        ingress._accept_payload(
            json.dumps(scene, separators=(",", ":")).encode("utf-8"),
            ("127.0.0.1", 12345),
            "raw-scene-test",
        )


def test_live_session_report_cannot_change_under_a_reused_source_identity() -> None:
    async def run() -> Any:
        ingress = SceneSnapshotIngress(_settings(capture_settings=_capture_settings()))
        await ingress.start()
        scene = _fixture()
        source_session_id = scene["source"]["session_id"]

        first_ack = _capture_acknowledgement()
        first_ack["source_report"]["session_id"] = source_session_id
        first_reader, first_writer = await asyncio.open_connection(*ingress.address)
        first_writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": source_session_id,
                    "capture_session_ack": first_ack,
                }
            )
        )
        await first_writer.drain()
        await _read_json_frame(first_reader)
        first_writer.close()
        await first_writer.wait_closed()

        second_ack = _capture_acknowledgement()
        second_ack["source_report"]["session_id"] = source_session_id
        second_ack["source_report"]["applied_controls"]["brightness"] = {
            "status": "applied",
            "value": 0.5,
        }
        second_reader, second_writer = await asyncio.open_connection(*ingress.address)
        second_writer.write(
            _frame(
                {
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": scene["engine_instance_id"],
                    "source_session_id": source_session_id,
                    "capture_session_ack": second_ack,
                }
            )
        )
        await second_writer.drain()
        closed = await asyncio.wait_for(second_reader.read(), timeout=1.0)
        stats = ingress.stats
        second_writer.close()
        await second_writer.wait_closed()
        await ingress.close()
        return closed, stats

    closed, stats = asyncio.run(run())

    assert closed == b""
    assert stats.rejected_handshakes == 1
