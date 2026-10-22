"""Behavioral contracts for WebSocket freshness and connection cleanup.

These tests exercise the handler's asynchronous writer instead of only checking
its queue internals. The transport remains WebSocket/TCP, while replaceable
state updates use latest-wins and expiry semantics at the application layer.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from backend.api.websocket.handler import WebSocketConnection, WebSocketHandler
from backend.api.websocket.manager import StreamType, WebSocketManager


class MemorySocket:
    """Minimal socket that records the messages accepted by the ASGI layer."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.close_calls: list[tuple[int | None, str | None]] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))


class FailingSocket(MemorySocket):
    """Socket that fails one write as if the peer disappeared mid-send."""

    async def send_text(self, message: str) -> None:
        raise OSError("peer closed while writing")


class DisconnectingSocket(MemorySocket):
    """Socket that reports the normal ASGI disconnect exception on send."""

    async def send_text(self, message: str) -> None:
        raise WebSocketDisconnect(code=1001)


def _register_local_connection(
    handler: WebSocketHandler,
    client_id: str,
    socket: MemorySocket,
) -> WebSocketConnection:
    connection = WebSocketConnection(socket, client_id)
    handler.connections[client_id] = connection
    handler.rate_limit_windows[client_id] = []
    return connection


@pytest.mark.asyncio
async def test_writer_delivers_only_latest_state_and_refreshes_source_age(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    socket = MemorySocket()
    connection = _register_local_connection(handler, "latest-state", socket)
    handler.max_state_observation_age_seconds = 0.5
    monkeypatch.setattr(
        "backend.api.websocket.handler.time.perf_counter_ns", lambda: 1_000_000_000
    )

    for frame_number, sequence, publication_tick in (
        (51, 900, 700_000_000),
        (52, 901, 750_000_000),
    ):
        assert handler._enqueue_message(
            connection,
            {
                "type": "state",
                "data": {
                    "sequence": sequence,
                    "balls": [{"id": "cue", "x": float(frame_number)}],
                    "observation_status": {
                        "vision_instance_id": "vision-a",
                        "source_frame_number": frame_number,
                        "source_publication_monotonic_ns": publication_tick,
                        "frame_publication_age_seconds": 0.01,
                    },
                },
            },
        )

    handler._ensure_writer(connection)
    await asyncio.wait_for(connection.outbound_idle.wait(), timeout=1)

    assert len(socket.messages) == 1
    delivered = json.loads(socket.messages[0])
    assert delivered["data"]["sequence"] == 901
    assert delivered["data"]["balls"][0]["x"] == 52.0
    # The source age is recalculated at the final local delivery boundary.
    assert delivered["data"]["observation_status"][
        "frame_publication_age_seconds"
    ] == pytest.approx(0.25)
    assert connection.dropped_updates == 1
    assert connection.expired_updates == 0

    await handler.disconnect(connection.client_id)


@pytest.mark.asyncio
async def test_writer_drops_replaceable_update_that_ages_out_in_queue(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    socket = MemorySocket()
    connection = _register_local_connection(handler, "queued-expiry", socket)
    handler.max_replaceable_message_age_seconds = 0.01
    monotonic_ticks = iter((1_000_000_000, 1_020_000_000))
    monkeypatch.setattr(
        "backend.api.websocket.handler.time.monotonic_ns", lambda: next(monotonic_ticks)
    )

    assert handler._enqueue_message(
        connection, {"type": "trajectory", "data": {"sequence": 7}}
    )
    handler._ensure_writer(connection)
    await asyncio.wait_for(connection.outbound_idle.wait(), timeout=1)

    assert socket.messages == []
    assert connection.expired_updates == 1
    assert connection.dropped_updates == 1
    assert handler.delivery_metrics["expired_updates"] == 1
    assert connection.replaceable_messages == {}
    await handler.disconnect(connection.client_id)


@pytest.mark.asyncio
async def test_state_already_stale_at_enqueue_is_discarded_before_serialization(
    monkeypatch,
):
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(MemorySocket(), "source-expiry")
    handler.max_state_observation_age_seconds = 0.5

    def fail_if_serialized(_message):
        raise AssertionError("an already stale state must be dropped before encoding")

    monkeypatch.setattr("backend.api.websocket.handler.json.dumps", fail_if_serialized)
    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "observation_status": {
                    "freshness_status": "fresh",
                    "frame_publication_age_seconds": 0.7,
                }
            },
        },
    )

    assert list(connection.outbound_messages) == []
    assert connection.expired_updates == 1
    assert handler.delivery_metrics["dropped_updates"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("socket_type", [FailingSocket, DisconnectingSocket])
async def test_writer_failure_removes_client_and_cleans_all_subscriptions(socket_type):
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    client_id = f"failed-{socket_type.__name__}"
    socket = socket_type()
    connection = _register_local_connection(handler, client_id, socket)
    await manager.register_client(client_id)
    connection.add_subscription(StreamType.STATE.value)
    manager.stream_subscribers[StreamType.STATE].add(client_id)
    manager.stream_subscribers[StreamType.TRAJECTORY].add(client_id)

    assert await handler.send_to_client(
        client_id, {"type": "state", "data": {"sequence": 12}}
    )
    await asyncio.wait_for(connection.closed_event.wait(), timeout=1)
    # Also await the cleanup task after closed_event to avoid leaving an
    # asynchronous teardown continuation behind in the test loop.
    await handler.disconnect(client_id)

    assert client_id not in handler.connections
    assert client_id not in handler.rate_limit_windows
    assert client_id not in manager.sessions
    assert all(
        client_id not in clients for clients in manager.stream_subscribers.values()
    )
    assert not connection.is_alive
    assert not connection.outbound_messages
    assert connection.control_queue_depth == 0
    assert socket.close_calls == []


@pytest.mark.asyncio
async def test_unserializable_outbound_control_message_disconnects_client():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    client_id = "invalid-outbound"
    connection = _register_local_connection(handler, client_id, MemorySocket())
    await manager.register_client(client_id)

    assert not await handler.send_to_client(
        client_id, {"type": "alert", "data": {"not-json": object()}}
    )
    await asyncio.wait_for(connection.closed_event.wait(), timeout=1)
    await handler.disconnect(client_id)

    assert client_id not in handler.connections
    assert client_id not in manager.sessions
    assert not connection.is_alive


@pytest.mark.asyncio
async def test_shutdown_drains_active_connections_and_cleans_manager_state():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    connections = []
    for client_id in ("shutdown-one", "shutdown-two"):
        connection = _register_local_connection(handler, client_id, MemorySocket())
        await manager.register_client(client_id)
        connections.append(connection)
        handler._ensure_writer(connection)

    await handler.start_monitoring()
    await handler.shutdown()

    assert handler.connections == {}
    assert handler._disconnect_tasks == {}
    assert handler._monitoring_task is None
    assert manager.sessions == {}
    assert all(connection.closed_event.is_set() for connection in connections)
    assert all(not connection.is_alive for connection in connections)
