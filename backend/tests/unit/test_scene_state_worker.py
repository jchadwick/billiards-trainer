"""End-to-end latest-only scene ingress to public state worker tests."""

from __future__ import annotations

import asyncio
import copy
import json
import struct
from pathlib import Path
from typing import Any

from backend.perception.scene_ingress import SceneIngressConfig, SceneSnapshotIngress
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher
from backend.perception.scene_state_worker import SceneSnapshotStateWorker

SCENE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "perception/contracts/v1/scene-rust-tracked-success.json"
)


def _scene(sequence: int) -> dict[str, Any]:
    value = json.loads(SCENE_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    value["snapshot_revision"] = sequence
    value["transport_sequence"] = sequence
    frame_id = f"frame-{sequence:04d}"
    timestamp = str(1_000_000_000 + sequence * 100_000_000)
    value["source"]["frame_id"] = frame_id
    value["source"]["timestamp"]["ticks"] = timestamp
    value["published_at"]["ticks"] = str(int(timestamp) + 10_000)
    cue_observation = value["cue"]["last_observation"]
    cue_observation["frame_id"] = frame_id
    cue_observation["timestamp"]["ticks"] = timestamp
    for ball in value["balls"]:
        ball["last_observation"]["frame_id"] = frame_id
        ball["last_observation"]["timestamp"]["ticks"] = timestamp
    return value


def _annotation(sequence: int) -> dict[str, Any]:
    value = _scene(sequence)
    value["update_kind"] = "annotation"
    value["outcome"] = "degraded"
    value["reason"] = "identity_update_pending"
    value["cue"] = {
        "status": "unavailable",
        "last_observation": None,
        "image_center_px": None,
        "image_space": None,
        "image_width_px": None,
        "image_height_px": None,
        "confidence": None,
    }
    value["balls"][0]["identity"]["number"] = 8
    value["balls"][0]["identity"]["confidence"] = 0.85
    value["balls"][0]["identity"]["evidence"] = [
        {"source": "test_classifier", "confidence": 0.85}
    ]
    return value


def _frame(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


async def _connect(ingress: SceneSnapshotIngress) -> tuple[Any, Any]:
    reader, writer = await asyncio.open_connection(*ingress.address)
    scene = json.loads(SCENE_FIXTURE.read_text(encoding="utf-8"))
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
    header = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
    response_size = struct.unpack(">I", header)[0]
    response = json.loads(await reader.readexactly(response_size))
    assert response == {
        "type": "ready",
        "protocol_version": 2,
        "last_transport_sequence": None,
    }
    return reader, writer


async def _wait_until(predicate: Any) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_slow_state_publisher_receives_first_and_latest_scene_only() -> None:
    async def run() -> tuple[list[dict[str, Any]], Any, Any]:
        first_publish_started = asyncio.Event()
        release_first_publish = asyncio.Event()
        second_publish_finished = asyncio.Event()

        class BlockingBroadcaster:
            def __init__(self) -> None:
                self.states: list[dict[str, Any]] = []

            async def broadcast_game_state(
                self,
                balls: object,
                cue: dict[str, Any] | None = None,
                table: dict[str, Any] | None = None,
                timestamp: Any = None,
                observation_status: dict[str, Any] | None = None,
                game_type: str | None = None,
            ) -> None:
                self.states.append(
                    {
                        "balls": balls,
                        "observation_status": observation_status,
                        "game_type": game_type,
                    }
                )
                if len(self.states) == 1:
                    first_publish_started.set()
                    await release_first_publish.wait()
                elif len(self.states) == 2:
                    second_publish_finished.set()

        ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address="127.0.0.1",
                port=0,
                maximum_message_bytes=256 * 1024,
                read_timeout_seconds=1.0,
            )
        )
        broadcaster = BlockingBroadcaster()
        worker = SceneSnapshotStateWorker(
            ingress,
            SceneSnapshotStatePublisher(broadcaster),
            source_silence_timeout_seconds=1.0,
        )
        await worker.start()
        _, writer = await _connect(ingress)

        writer.write(_frame(_scene(1)))
        await writer.drain()
        await asyncio.wait_for(first_publish_started.wait(), timeout=1.0)

        # The consumer remains occupied by the first downstream send while the
        # Ingress continues accepting and coalescing complete states.
        writer.write(_frame(_scene(2)))
        writer.write(_frame(_scene(3)))
        await writer.drain()
        await _wait_until(lambda: ingress.stats.accepted_snapshots == 3)
        release_first_publish.set()
        await asyncio.wait_for(second_publish_finished.wait(), timeout=1.0)

        writer.close()
        await writer.wait_closed()
        stats = worker.stats
        ingress_stats = ingress.stats
        states = broadcaster.states
        await worker.close()
        return states, stats, ingress_stats

    states, stats, ingress_stats = asyncio.run(run())

    assert len(states) == 2
    assert [state["observation_status"]["source_frame_id"] for state in states] == [
        "frame-0001",
        "frame-0003",
    ]
    assert stats.received == 2
    assert stats.published == 2
    assert stats.ignored_non_observations == 0
    assert stats.latest_transport_sequence == 3
    assert ingress_stats.accepted_snapshots == 3
    assert ingress_stats.queue_replacements == 1


def test_worker_skips_failed_and_annotation_snapshots_without_republishing_state() -> (
    None
):
    async def run() -> tuple[list[dict[str, Any]], Any]:
        first_publish_finished = asyncio.Event()

        class RecordingBroadcaster:
            def __init__(self) -> None:
                self.states: list[dict[str, Any]] = []

            async def broadcast_game_state(
                self,
                balls: object,
                cue: dict[str, Any] | None = None,
                table: dict[str, Any] | None = None,
                timestamp: Any = None,
                observation_status: dict[str, Any] | None = None,
                game_type: str | None = None,
            ) -> None:
                self.states.append(
                    {"balls": balls, "observation_status": observation_status}
                )
                first_publish_finished.set()

        ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address="127.0.0.1",
                port=0,
                maximum_message_bytes=256 * 1024,
                read_timeout_seconds=1.0,
            )
        )
        broadcaster = RecordingBroadcaster()
        worker = SceneSnapshotStateWorker(
            ingress,
            SceneSnapshotStatePublisher(broadcaster),
            source_silence_timeout_seconds=1.0,
        )
        await worker.start()
        _, writer = await _connect(ingress)

        first = _scene(1)
        annotation = _annotation(2)
        failed = _scene(3)
        failed["update_kind"] = "health"
        failed["outcome"] = "failed"
        failed["reason"] = "detector_unavailable"
        failed["cue"] = copy.deepcopy(annotation["cue"])
        failed["balls"] = []

        writer.write(_frame(first))
        await writer.drain()
        await asyncio.wait_for(first_publish_finished.wait(), timeout=1.0)
        writer.write(_frame(annotation))
        await writer.drain()
        await _wait_until(lambda: worker.stats.received == 2)
        writer.write(_frame(failed))
        await writer.drain()
        await _wait_until(lambda: worker.stats.received == 3)

        states = broadcaster.states
        stats = worker.stats
        writer.close()
        await writer.wait_closed()
        await worker.close()
        return states, stats

    states, stats = asyncio.run(run())

    assert len(states) == 2
    assert states[0]["observation_status"]["ball_positions_status"] == "observed"
    assert states[1]["observation_status"]["ball_positions_status"] == "cached"
    assert states[1]["observation_status"]["ball_detection_status"] == "failed"
    assert states[1]["balls"][0]["is_observed"] is False
    assert states[1]["balls"][0]["confidence"] == 0.0
    assert states[1]["balls"][0]["confidence_kind"] == "unavailable"
    assert states[1]["balls"][0]["last_observation_confidence"] == 0.97
    assert stats.published == 1
    assert stats.health_updates_published == 1
    assert stats.ignored_non_observations == 1
    assert stats.incompatible == 0
    assert stats.publication_errors == 0


def test_worker_marks_last_scene_stale_once_after_configured_source_silence() -> None:
    async def run() -> tuple[list[dict[str, Any]], Any]:
        second_state_published = asyncio.Event()

        class RecordingBroadcaster:
            def __init__(self) -> None:
                self.states: list[dict[str, Any]] = []

            async def broadcast_game_state(
                self,
                balls: object,
                cue: dict[str, Any] | None = None,
                table: dict[str, Any] | None = None,
                timestamp: Any = None,
                observation_status: dict[str, Any] | None = None,
                game_type: str | None = None,
            ) -> None:
                self.states.append(
                    {"balls": balls, "observation_status": observation_status}
                )
                if len(self.states) == 2:
                    second_state_published.set()

        ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address="127.0.0.1",
                port=0,
                maximum_message_bytes=256 * 1024,
                read_timeout_seconds=1.0,
            )
        )
        broadcaster = RecordingBroadcaster()
        worker = SceneSnapshotStateWorker(
            ingress,
            SceneSnapshotStatePublisher(broadcaster),
            source_silence_timeout_seconds=0.02,
        )
        await worker.start()
        _, writer = await _connect(ingress)
        writer.write(_frame(_scene(1)))
        await writer.drain()
        await _wait_until(lambda: len(broadcaster.states) == 1)
        await asyncio.wait_for(second_state_published.wait(), timeout=1.0)
        await asyncio.sleep(0.06)
        states = broadcaster.states
        stats = worker.stats
        writer.close()
        await writer.wait_closed()
        await worker.close()
        return states, stats

    states, stats = asyncio.run(run())

    assert len(states) == 2
    assert states[1]["balls"][0]["is_observed"] is False
    assert states[1]["observation_status"]["freshness_status"] == "source_silent"
    assert states[1]["observation_status"]["health_reason"] == "source_silence_timeout"
    assert "source_timestamp" not in states[1]["observation_status"]
    assert states[1]["observation_status"]["observation_identity"]["frame_id"] == (
        "frame-0001"
    )
    assert states[1]["balls"][0]["confidence"] == 0.0
    assert states[1]["balls"][0]["confidence_kind"] == "unavailable"
    assert states[1]["balls"][0]["last_observation_confidence"] == 0.97
    assert stats.health_updates_published == 1
    assert stats.source_health == "source_silent"
    assert stats.last_observation_age_seconds is not None
    assert stats.last_observation_age_seconds >= stats.source_silence_timeout_seconds


def test_annotations_do_not_extend_fresh_localization_deadline() -> None:
    async def run() -> tuple[list[dict[str, Any]], Any]:
        stale_state_published = asyncio.Event()

        class RecordingBroadcaster:
            def __init__(self) -> None:
                self.states: list[dict[str, Any]] = []

            async def broadcast_game_state(
                self,
                balls: object,
                cue: dict[str, Any] | None = None,
                table: dict[str, Any] | None = None,
                timestamp: Any = None,
                observation_status: dict[str, Any] | None = None,
                game_type: str | None = None,
            ) -> None:
                self.states.append(
                    {"balls": balls, "observation_status": observation_status}
                )
                if len(self.states) == 2:
                    stale_state_published.set()

        ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address="127.0.0.1",
                port=0,
                maximum_message_bytes=256 * 1024,
                read_timeout_seconds=1.0,
            )
        )
        broadcaster = RecordingBroadcaster()
        worker = SceneSnapshotStateWorker(
            ingress,
            SceneSnapshotStatePublisher(broadcaster),
            source_silence_timeout_seconds=0.08,
        )
        await worker.start()
        _, writer = await _connect(ingress)
        writer.write(_frame(_scene(1)))
        await writer.drain()
        await _wait_until(lambda: len(broadcaster.states) == 1)

        async def send_annotations() -> None:
            for sequence in range(2, 14):
                writer.write(_frame(_annotation(sequence)))
                await writer.drain()
                await _wait_until(
                    lambda expected=sequence: worker.stats.received >= expected
                )
                await asyncio.sleep(0.01)

        sender = asyncio.create_task(send_annotations())
        try:
            # An annotation arrives every 10 ms, faster than the 80 ms
            # localization timeout. It must not postpone the stale update.
            await asyncio.wait_for(stale_state_published.wait(), timeout=1.0)
            await sender
            await asyncio.sleep(0.04)
            states = broadcaster.states
            stats = worker.stats
        finally:
            if not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            writer.close()
            await writer.wait_closed()
            await worker.close()
        return states, stats

    states, stats = asyncio.run(run())

    assert len(states) == 2
    assert states[1]["observation_status"]["health_reason"] == "source_silence_timeout"
    assert states[1]["balls"][0]["is_observed"] is False
    assert stats.received == 13
    assert stats.ignored_non_observations == 12
    assert stats.health_updates_published == 1


def test_worker_propagates_paired_ingress_receipt_to_scene_read_model() -> None:
    class Broadcaster:
        async def broadcast_game_state(self, **kwargs: Any) -> None:
            return None

    async def run() -> tuple[Any, Any]:
        ingress = SceneSnapshotIngress(
            SceneIngressConfig(
                bind_address="127.0.0.1",
                port=0,
                maximum_message_bytes=256 * 1024,
                read_timeout_seconds=1.0,
            )
        )

        class RecordingPublisher:
            def __init__(self) -> None:
                self.inner = SceneSnapshotStatePublisher(Broadcaster())
                self.received_at = None

            async def publish(self, snapshot, *, received_at):
                self.received_at = received_at
                return await self.inner.publish(snapshot, received_at=received_at)

            async def publish_health(self, snapshot):
                return await self.inner.publish_health(snapshot)

            async def publish_source_silence(self):
                return await self.inner.publish_source_silence()

        publisher = RecordingPublisher()
        worker = SceneSnapshotStateWorker(
            ingress, publisher, source_silence_timeout_seconds=1.0
        )
        await worker.start()
        _, writer = await _connect(ingress)
        try:
            writer.write(_frame(_scene(1)))
            await writer.drain()
            await _wait_until(lambda: worker.stats.published == 1)
            read_model = publisher.inner.read_model
            assert read_model is not None
            assert publisher.received_at is not None
            assert read_model.localization_received_at == publisher.received_at
            assert (
                read_model.ball_localization_received_at["track-1"]
                == publisher.received_at
            )
            return publisher.received_at, read_model
        finally:
            writer.close()
            await writer.wait_closed()
            await worker.close()

    received_at, read_model = asyncio.run(run())
    assert received_at.tzinfo is not None
    assert received_at.utcoffset() is not None
    assert read_model.updated_at >= received_at
