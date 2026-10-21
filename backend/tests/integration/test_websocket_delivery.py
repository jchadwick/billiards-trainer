"""Deterministic tests for bounded, per-client WebSocket delivery."""

import asyncio
import json

import pytest

import backend.api.websocket as websocket_package
from backend.api.websocket.handler import WebSocketConnection, WebSocketHandler
from backend.api.websocket.manager import WebSocketManager


class RecordingSocket:
    def __init__(self):
        self.messages: list[str] = []
        self.close_calls: list[tuple[int | None, str | None]] = []

    async def send_text(self, message: str) -> None:
        self.messages.append(message)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))


class GatedSocket(RecordingSocket):
    """Socket that blocks writes until a test releases its gate."""

    def __init__(self):
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send_text(self, message: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        self.messages.append(message)


class GatedCloseSocket(GatedSocket):
    """Socket whose close handshake can also be held by a test gate."""

    def __init__(self):
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))
        self.close_started.set()
        await self.release_close.wait()


class CancellationGatedSocket(GatedSocket):
    """Socket that holds its send coroutine open during writer cancellation."""

    def __init__(self):
        super().__init__()
        self.cancel_started = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def send_text(self, message: str) -> None:
        self.send_started.set()
        try:
            await self.release_send.wait()
        except asyncio.CancelledError:
            self.cancel_started.set()
            await self.release_cancel.wait()
        self.messages.append(message)


class CountingManager(WebSocketManager):
    def __init__(self):
        super().__init__()
        self.unregister_count = 0

    async def unregister_client(self, client_id: str) -> None:
        self.unregister_count += 1
        await super().unregister_client(client_id)


def add_connection(handler: WebSocketHandler, client_id: str, socket) -> None:
    handler.connections[client_id] = WebSocketConnection(socket, client_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_slow_client_does_not_block_healthy_client_and_updates_coalesce():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    handler.replaceable_queue_capacity = 2
    slow_socket = GatedSocket()
    healthy_socket = RecordingSocket()
    add_connection(handler, "slow", slow_socket)
    add_connection(handler, "healthy", healthy_socket)

    assert await handler.send_to_client("slow", {"type": "state", "data": {"n": 0}})
    await slow_socket.send_started.wait()

    # Distinct streams fill the pending replaceable capacity. The next one
    # evicts the oldest pending stream update while the first write is blocked.
    assert await handler.send_to_client(
        "slow", {"type": "trajectory", "data": {"n": 1}}
    )
    assert await handler.send_to_client(
        "slow", {"type": "performance", "data": {"n": 2}}
    )
    assert await handler.send_to_client("slow", {"type": "metrics", "data": {"n": 3}})
    assert await handler.send_to_client("healthy", {"type": "state", "data": {"n": 4}})
    await handler.wait_for_outbound("healthy")

    assert [json.loads(item)["data"]["n"] for item in healthy_socket.messages] == [4]
    slow_connection = handler.connections["slow"]
    assert len(slow_connection.outbound_messages) == 2
    assert slow_connection.dropped_updates == 1
    assert [item.replaceable_key for item in slow_connection.outbound_messages] == [
        "performance",
        "metrics",
    ]

    slow_socket.release_send.set()
    await handler.wait_for_outbound("slow")
    assert [json.loads(item)["data"]["n"] for item in slow_socket.messages] == [0, 2, 3]

    await handler.disconnect("slow")
    await handler.disconnect("healthy")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_messages_keep_order_and_queue_overflow_disconnects_client():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    handler.control_queue_capacity = 3
    socket = GatedSocket()
    add_connection(handler, "client", socket)

    assert await handler.send_to_client("client", {"type": "state", "data": {"n": 0}})
    await socket.send_started.wait()
    controls = [
        {"type": "error", "data": {"order": 1}},
        {"type": "alert", "data": {"order": 2}},
        {"type": "config", "data": {"order": 3}},
    ]
    for message in controls:
        assert await handler.send_to_client("client", message)

    socket.release_send.set()
    await handler.wait_for_outbound("client")
    assert [json.loads(item)["type"] for item in socket.messages] == [
        "state",
        "error",
        "alert",
        "config",
    ]

    # With the writer held inside a send, pending control messages have a hard
    # capacity. Overflow closes the client rather than losing an ordered event.
    socket2 = GatedSocket()
    add_connection(handler, "overflow", socket2)
    assert await handler.send_to_client("overflow", {"type": "state", "data": {}})
    await socket2.send_started.wait()
    handler.control_queue_capacity = 1
    assert await handler.send_to_client("overflow", {"type": "error", "data": {"n": 1}})
    overflow_connection = handler.connections["overflow"]
    assert not await handler.send_to_client(
        "overflow", {"type": "alert", "data": {"n": 2}}
    )

    assert "overflow" not in handler.connections
    await asyncio.wait_for(overflow_connection.closed_event.wait(), timeout=1)
    assert socket2.close_calls == [(1013, "Slow consumer")]
    assert handler.delivery_metrics["control_queue_overflows"] == 1
    assert handler.delivery_metrics["disconnected_slow_clients"] == 1
    await handler.disconnect("client")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_overflow_cleanup_does_not_block_other_clients():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.control_queue_capacity = 1
    slow_socket = GatedCloseSocket()
    healthy_socket = RecordingSocket()
    add_connection(handler, "slow-close", slow_socket)
    add_connection(handler, "healthy", healthy_socket)

    assert await handler.send_to_client("slow-close", {"type": "state", "data": {}})
    await slow_socket.send_started.wait()
    assert await handler.send_to_client(
        "slow-close", {"type": "error", "data": {"n": 1}}
    )
    slow_connection = handler.connections["slow-close"]

    overflow_task = asyncio.create_task(
        handler.send_to_client("slow-close", {"type": "alert", "data": {"n": 2}})
    )
    await slow_socket.close_started.wait()
    # The send caller must finish while cleanup is still blocked in close().
    assert overflow_task.done()
    assert await overflow_task is False
    assert "slow-close" not in handler.connections

    assert await handler.send_to_client("healthy", {"type": "state", "data": {"n": 3}})
    await handler.wait_for_outbound("healthy")
    assert json.loads(healthy_socket.messages[0])["data"]["n"] == 3
    assert not slow_socket.release_close.is_set()

    slow_socket.release_close.set()
    await asyncio.wait_for(slow_connection.closed_event.wait(), timeout=1)
    await handler.disconnect("healthy")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_system_shutdown_drains_cleanup_for_claimed_connection(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.control_queue_capacity = 1
    socket = GatedCloseSocket()
    add_connection(handler, "slow-close", socket)
    connection = handler.connections["slow-close"]

    assert await handler.send_to_client("slow-close", {"type": "state", "data": {}})
    await socket.send_started.wait()
    assert await handler.send_to_client(
        "slow-close", {"type": "error", "data": {"n": 1}}
    )
    assert not await handler.send_to_client(
        "slow-close", {"type": "alert", "data": {"n": 2}}
    )
    await socket.close_started.wait()

    class StoppedBroadcaster:
        stopped = False

        async def stop_streaming(self):
            self.stopped = True

    broadcaster = StoppedBroadcaster()
    monkeypatch.setattr(websocket_package, "websocket_handler", handler)
    monkeypatch.setattr(websocket_package, "message_broadcaster", broadcaster)

    shutdown_task = asyncio.create_task(websocket_package.shutdown_websocket_system())
    try:
        # shutdown has reached its drain await while close remains gated.
        await asyncio.sleep(0)
        assert not shutdown_task.done()
        assert "slow-close" not in handler.connections
        assert "slow-close" in handler._disconnect_tasks
    finally:
        socket.release_close.set()

    await shutdown_task
    assert broadcaster.stopped
    assert connection.closed_event.is_set()
    assert not handler._disconnect_tasks
    assert not handler.connections


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_disconnects_share_one_cleanup_without_key_errors():
    manager = CountingManager()
    handler = WebSocketHandler(manager=manager)
    socket = CancellationGatedSocket()
    add_connection(handler, "racing", socket)
    await manager.register_client("racing")
    connection = handler.connections["racing"]
    assert await handler.send_to_client("racing", {"type": "state", "data": {}})
    await socket.send_started.wait()
    first = asyncio.create_task(handler.disconnect("racing"))
    callers = [first]
    try:
        await socket.cancel_started.wait()
        assert "racing" not in handler.connections
        second = asyncio.create_task(handler.disconnect("racing"))
        callers.append(second)
    finally:
        socket.release_cancel.set()
    results = await asyncio.gather(*callers, return_exceptions=True)

    assert not any(isinstance(result, BaseException) for result in results)
    assert "racing" not in handler.connections
    assert connection.closed_event.is_set()
    assert manager.unregister_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_send_deadline_disconnects_a_stalled_client_and_reports_metrics():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.send_timeout_seconds = 0.01
    socket = GatedSocket()
    add_connection(handler, "stalled", socket)
    connection = handler.connections["stalled"]

    assert await handler.send_to_client("stalled", {"type": "state", "data": {}})
    await socket.send_started.wait()
    # The writer marks its queue idle only after handling the timed-out send.
    await asyncio.wait_for(connection.closed_event.wait(), timeout=1)

    assert "stalled" not in handler.connections
    assert handler.delivery_metrics["send_timeouts"] == 1
    assert handler.delivery_metrics["disconnected_slow_clients"] == 1
    assert socket.close_calls == [(1013, "Slow consumer")]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disconnect_cancels_writer_and_clears_pending_messages():
    handler = WebSocketHandler(manager=WebSocketManager())
    socket = GatedSocket()
    add_connection(handler, "closing", socket)

    assert await handler.send_to_client("closing", {"type": "state", "data": {}})
    await socket.send_started.wait()
    assert await handler.send_to_client("closing", {"type": "alert", "data": {}})
    connection = handler.connections["closing"]
    writer_task = connection.writer_task

    await handler.disconnect("closing")

    assert writer_task is not None
    assert writer_task.done()
    assert connection.closed_event.is_set()
    assert not connection.outbound_messages
    assert not connection.replaceable_messages
    assert connection.control_queue_depth == 0
