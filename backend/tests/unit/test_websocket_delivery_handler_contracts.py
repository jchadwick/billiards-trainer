"""Unit-level contracts for best-effort WebSocket state delivery."""

import asyncio
import json

import pytest

import backend.api.websocket.handler as handler_module
from backend.api.websocket.handler import WebSocketConnection, WebSocketHandler
from backend.api.websocket.manager import WebSocketManager


class RecordingSocket:
    """Capture writes and close requests without a network server."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.close_calls: list[tuple[int | None, str | None]] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))


class BlockingSocket(RecordingSocket):
    """Hold writes until the handler's configured send deadline expires."""

    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send_text(self, message: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.messages.append(message)


def _add_connection(handler: WebSocketHandler, client_id: str, socket) -> None:
    handler.connections[client_id] = WebSocketConnection(socket, client_id)


def _message_types(connection: WebSocketConnection) -> list[str]:
    return [json.loads(item.payload)["type"] for item in connection.outbound_messages]


def test_latest_update_replaces_its_slot_without_reordering_control_messages():
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "interleaved")

    updates = (
        {"type": "state", "data": {"sequence": 1}},
        {"type": "alert", "data": {"order": 1}},
        {"type": "trajectory", "data": {"sequence": 1}},
        {"type": "state", "data": {"sequence": 2}},
        {"type": "error", "data": {"order": 2}},
        {"type": "trajectory", "data": {"sequence": 2}},
    )
    for message in updates:
        assert handler._enqueue_message(connection, message)

    # Each replaceable stream keeps one queue slot. Replacements retain that
    # slot's place relative to ordered controls, and do not collapse controls.
    assert _message_types(connection) == [
        "state",
        "alert",
        "trajectory",
        "error",
    ]
    assert [
        json.loads(item.payload)["data"].get("sequence")
        for item in connection.outbound_messages
        if item.replaceable_key is not None
    ] == [2, 2]
    assert connection.control_queue_depth == 2
    assert connection.dropped_updates == 2


def test_replaceable_capacity_evicts_an_update_without_evicting_ordered_control():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.replaceable_queue_capacity = 1
    connection = WebSocketConnection(RecordingSocket(), "bounded-updates")

    assert handler._enqueue_message(
        connection, {"type": "state", "data": {"sequence": 1}}
    )
    assert handler._enqueue_message(connection, {"type": "alert", "data": {"order": 1}})
    assert handler._enqueue_message(
        connection, {"type": "trajectory", "data": {"sequence": 1}}
    )

    assert _message_types(connection) == ["alert", "trajectory"]
    assert connection.control_queue_depth == 1
    assert connection.replaceable_messages.keys() == {"trajectory"}
    assert connection.dropped_updates == 1


def test_control_capacity_is_independent_of_replaceable_updates():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.replaceable_queue_capacity = 1
    handler.control_queue_capacity = 1
    connection = WebSocketConnection(RecordingSocket(), "isolated-capacities")

    assert handler._enqueue_message(
        connection, {"type": "state", "data": {"sequence": 1}}
    )
    assert handler._enqueue_message(connection, {"type": "error", "data": {"order": 1}})
    # A state replacement consumes no control capacity and must remain
    # admissible even while the ordered control slot is full.
    assert handler._enqueue_message(
        connection, {"type": "state", "data": {"sequence": 2}}
    )
    assert _message_types(connection) == ["state", "error"]
    assert connection.control_queue_depth == 1

    assert not handler._enqueue_message(
        connection, {"type": "alert", "data": {"order": 2}}
    )
    assert connection.control_queue_overflows == 1
    assert handler.delivery_metrics["control_queue_overflows"] == 1


def test_nested_source_identity_rejects_older_frame_even_with_newer_publish_sequence(
    monkeypatch,
):
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "nested-source-identity")
    current = {
        "type": "state",
        "data": {
            "sequence": 800,
            "observation_status": {
                "observation_identity": {
                    "engine_instance_id": "engine-a",
                    "session_id": "session-a",
                    "frame_id": "42",
                }
            },
        },
    }
    assert handler._enqueue_message(connection, current)

    def fail_if_stale_state_is_serialized(_message):
        raise AssertionError("older camera frames must be discarded before encoding")

    monkeypatch.setattr(handler_module.json, "dumps", fail_if_stale_state_is_serialized)
    delayed_inference = {
        "type": "state",
        "data": {
            # A later publication sequence cannot make an older camera frame
            # current: inference may complete out of order.
            "sequence": 801,
            "observation_status": {
                "observation_identity": {
                    "engine_instance_id": "engine-a",
                    "session_id": "session-a",
                    "frame_id": "41",
                }
            },
        },
    }

    assert handler._enqueue_message(connection, delayed_inference)
    assert len(connection.outbound_messages) == 1
    assert connection.state_source_frame_number == 42
    assert connection.dropped_updates == 1


def test_new_source_epoch_allows_frame_counter_to_restart():
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "restarted-source")

    for sequence, engine_id, frame_id in (
        (700, "engine-before-restart", "900"),
        (1, "engine-after-restart", "1"),
        (2, "engine-after-restart", "2"),
    ):
        assert handler._enqueue_message(
            connection,
            {
                "type": "state",
                "data": {
                    "sequence": sequence,
                    "observation_status": {
                        "observation_identity": {
                            "engine_instance_id": engine_id,
                            "session_id": "session-a",
                            "frame_id": frame_id,
                        }
                    },
                },
            },
        )

    latest = json.loads(connection.outbound_messages[0].payload)
    assert latest["data"]["sequence"] == 2
    assert connection.state_source_epoch == (
        ("engine_instance_id", "engine-after-restart"),
        ("session_id", "session-a"),
    )
    assert connection.state_source_frame_number == 2
    assert connection.dropped_updates == 2


@pytest.mark.asyncio
async def test_stalled_client_timeout_does_not_block_delivery_to_healthy_client():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.replaceable_send_timeout_seconds = 0.02
    stalled_socket = BlockingSocket()
    healthy_socket = RecordingSocket()
    _add_connection(handler, "stalled", stalled_socket)
    _add_connection(handler, "healthy", healthy_socket)
    stalled_connection = handler.connections["stalled"]

    assert await handler.send_to_client(
        "stalled", {"type": "state", "data": {"sequence": 1}}
    )
    await asyncio.wait_for(stalled_socket.send_started.wait(), timeout=1)

    assert await handler.send_to_client(
        "healthy", {"type": "state", "data": {"sequence": 2}}
    )
    await handler.wait_for_outbound("healthy")
    await asyncio.wait_for(stalled_connection.closed_event.wait(), timeout=1)

    assert json.loads(healthy_socket.messages[0])["data"]["sequence"] == 2
    assert "stalled" not in handler.connections
    assert stalled_socket.close_calls == [(1013, "Slow consumer")]
    assert stalled_connection.send_timeouts == 1
    assert not stalled_connection.outbound_messages
    assert handler.delivery_metrics["disconnected_slow_clients"] == 1

    await handler.disconnect("healthy")
