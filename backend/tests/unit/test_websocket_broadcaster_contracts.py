"""Behavioral contracts for state publication and bounded WebSocket metadata."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import backend.api.websocket.broadcaster as broadcaster_module
from backend.api.websocket.broadcaster import (
    FrameBuffer,
    MessageBroadcaster,
    NumpyEncoder,
)
from backend.api.websocket.schemas import StreamType


def _ball(x: float = 10.0, y: float = 20.0) -> dict[str, object]:
    return {
        "id": "cue",
        "position": {"x": x, "y": y, "scale": [1.0, 1.0]},
    }


@pytest.fixture
def broadcaster(monkeypatch: pytest.MonkeyPatch) -> MessageBroadcaster:
    audit = Mock()
    audit.select_performance_audit_detail.return_value = False
    monkeypatch.setattr(broadcaster_module, "websocket_handler", audit)
    result = MessageBroadcaster()
    result._broadcast_to_stream = AsyncMock()
    return result


@pytest.mark.asyncio
async def test_game_state_publishes_every_valid_observation_even_when_shape_unchanged(
    broadcaster: MessageBroadcaster,
) -> None:
    first_timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_timestamp = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    await broadcaster.broadcast_game_state(
        [_ball()],
        timestamp=first_timestamp,
        observation_status={"source_frame_number": 1},
    )
    await broadcaster.broadcast_game_state(
        [_ball(11.0)],
        timestamp=second_timestamp,
        observation_status={"source_frame_number": 2},
    )

    assert broadcaster._broadcast_to_stream.await_count == 2
    first_stream, first_payload = broadcaster._broadcast_to_stream.await_args_list[
        0
    ].args
    second_stream, second_payload = broadcaster._broadcast_to_stream.await_args_list[
        1
    ].args
    assert first_stream is second_stream is StreamType.STATE
    assert [first_payload["sequence"], second_payload["sequence"]] == [1, 2]
    assert first_payload["observation_status"]["source_frame_number"] == 1
    assert second_payload["observation_status"]["source_frame_number"] == 2
    assert first_payload["timestamp"] != second_payload["timestamp"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "balls",
    [
        None,
        ["not-a-ball"],
        [{"id": "cue"}],
        [{"position": [10.0, 20.0]}],
        [{"position": {"x": 1.0, "y": 2.0}}],
        [{"position": {"x": 1.0, "y": 2.0, "scale": [0.0, 1.0]}}],
        [{"position": {"x": 1.0, "y": 2.0, "scale": [1.0]}}],
    ],
)
async def test_game_state_rejects_invalid_observations_without_publishing(
    broadcaster: MessageBroadcaster,
    balls: object,
) -> None:
    await broadcaster.broadcast_game_state(balls)

    broadcaster._broadcast_to_stream.assert_not_awaited()
    assert broadcaster.broadcast_stats.validation_failures == 1


@pytest.mark.asyncio
async def test_state_audit_records_source_identity_and_component_age(
    monkeypatch: pytest.MonkeyPatch,
    broadcaster: MessageBroadcaster,
) -> None:
    monkeypatch.setattr(broadcaster_module.time, "monotonic", lambda: 12.0)
    audit = broadcaster_module.websocket_handler
    audit.select_performance_audit_detail.return_value = True

    await broadcaster.broadcast_game_state(
        [_ball()],
        observation_status={
            "source_frame_number": 42,
            "source_timestamp_kind": "shared_memory_publication_monotonic",
            "component_observation_timestamps": {"balls": 10.5, "invalid": True},
            "map_revision": 8,
        },
    )

    audit.select_performance_audit_detail.assert_called_once_with(
        "state", 42, category="state_published"
    )
    event = audit.record_performance_audit.call_args.args[1]
    assert event["source_frame_number"] == 42
    assert event["source_timestamp_kind"] == "shared_memory_publication_monotonic"
    assert event["map_revision"] == 8
    assert event["component_age_seconds"] == {"balls": 1.5}


@pytest.mark.asyncio
async def test_alert_routes_targeted_and_broadcast_delivery(
    monkeypatch: pytest.MonkeyPatch,
    broadcaster: MessageBroadcaster,
) -> None:
    manager = Mock()
    manager.send_alert = AsyncMock()
    monkeypatch.setattr(broadcaster_module, "websocket_manager", manager)

    await broadcaster.broadcast_alert("warning", "stale", "STALE", target_clients=["a"])

    manager.send_alert.assert_awaited_once_with(
        "warning", "stale", "STALE", None, ["a"]
    )
    broadcaster._broadcast_to_stream.assert_not_awaited()

    await broadcaster.broadcast_alert("info", "ready", "READY")

    assert broadcaster._broadcast_to_stream.await_count == 1
    stream, payload = broadcaster._broadcast_to_stream.await_args.args
    assert stream is StreamType.ALERT
    assert payload["sequence"] == 2
    assert payload["details"] == {}


@pytest.mark.asyncio
async def test_config_and_performance_publications_use_their_own_streams(
    broadcaster: MessageBroadcaster,
) -> None:
    await broadcaster.broadcast_config_update("vision", {"fps": 30}, "changed")
    await broadcaster.broadcast_performance_stats({"fps": 29.5})

    assert [
        call.args[0] for call in broadcaster._broadcast_to_stream.await_args_list
    ] == [
        StreamType.CONFIG,
        StreamType.PERFORMANCE,
    ]
    assert broadcaster._broadcast_to_stream.await_args_list[0].args[1]["config"] == {
        "fps": 30
    }
    assert broadcaster._broadcast_to_stream.await_args_list[1].args[1] == {"fps": 29.5}


@pytest.mark.asyncio
async def test_broadcast_stats_update_success_and_isolate_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    broadcaster: MessageBroadcaster,
) -> None:
    manager = Mock()
    manager.broadcast_to_stream = AsyncMock()
    monkeypatch.setattr(broadcaster_module, "websocket_manager", manager)
    monkeypatch.setattr(
        broadcaster,
        "_broadcast_to_stream",
        MessageBroadcaster._broadcast_to_stream.__get__(broadcaster),
    )

    await broadcaster._broadcast_to_stream(StreamType.STATE, {"sequence": 1})
    assert broadcaster.broadcast_stats.messages_sent == 1
    assert broadcaster.broadcast_stats.bytes_sent > 0
    assert broadcaster.broadcast_stats.average_latency >= 0

    manager.broadcast_to_stream.side_effect = RuntimeError("client disconnected")
    await broadcaster._broadcast_to_stream(StreamType.STATE, {"sequence": 2})

    assert broadcaster.broadcast_stats.failed_sends == 1
    assert broadcaster.broadcast_stats.messages_sent == 1


def test_frame_buffer_is_bounded_and_uses_receipt_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((1.0, 2.0, 3.0, 4.1))
    monkeypatch.setattr(broadcaster_module.time, "time", lambda: next(times))
    buffer = FrameBuffer(max_size=2)
    buffer.add_frame({"sequence": 1})
    buffer.add_frame({"sequence": 2})
    buffer.add_frame({"sequence": 3})

    assert buffer.get_latest_frame() == {"sequence": 3}
    assert len(buffer.frames) == 2
    assert buffer.get_frame_rate() == 1.0

    buffer.cleanup_old_frames(max_age_seconds=2.0)
    assert list(buffer.frames) == [{"sequence": 3}]
    assert buffer.get_latest_frame() == {"sequence": 3}


def test_frame_rate_handles_insufficient_and_non_increasing_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((5.0, 5.0, 5.0))
    monkeypatch.setattr(broadcaster_module.time, "time", lambda: next(times))
    buffer = FrameBuffer(max_size=3)
    assert buffer.get_frame_rate() == 0.0
    buffer.add_frame({"sequence": 1})
    assert buffer.get_frame_rate() == 0.0
    buffer.add_frame({"sequence": 2})
    assert buffer.get_frame_rate() == 0.0


def test_numpy_encoder_converts_scalar_array_and_boolean_payloads() -> None:
    payload = {
        "integer": np.int64(3),
        "float": np.float32(1.25),
        "array": np.array([2, 4]),
        "boolean": np.bool_(True),
    }

    encoded = json.dumps(payload, cls=NumpyEncoder)

    assert '"integer": 3' in encoded
    assert '"float": 1.25' in encoded
    assert '"array": [2, 4]' in encoded
    assert '"boolean": true' in encoded


@pytest.mark.parametrize(
    ("payload", "expected_compressed"),
    [
        (b"short", False),
        (b"x" * 2048, True),
        (random.Random(7).randbytes(2048), False),
    ],
)
def test_compression_is_applied_only_when_it_meets_configured_savings(
    broadcaster: MessageBroadcaster,
    payload: bytes,
    expected_compressed: bool,
) -> None:
    broadcaster.compression_threshold = 100
    broadcaster.compression_ratio_threshold = 0.5

    result, compressed = broadcaster._compress_data(payload)

    assert compressed is expected_compressed
    if compressed:
        assert result != payload
    else:
        assert result == payload
