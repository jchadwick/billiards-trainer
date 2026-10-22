"""Deterministic contracts for WebSocket admission, lifecycle, and fan-out."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocketDisconnect, status

from backend.api.websocket import endpoints
from backend.api.websocket import manager as manager_module
from backend.api.websocket.manager import (
    ConnectionState,
    StreamType,
    SubscriptionFilter,
    WebSocketManager,
)
from backend.api.websocket.rate_limit import RateLimitUnavailable
from backend.api.websocket.schemas import SubscribeRequest, UnsubscribeRequest


class FakeWebSocket:
    """Small endpoint double with explicit, deterministic inbound messages."""

    def __init__(self, messages=()):
        self.client = SimpleNamespace(host="192.0.2.17")
        self.headers = {"user-agent": "contract-test"}
        self.messages = iter(messages)
        self.close_calls = []

    async def receive_text(self):
        try:
            item = next(self.messages)
        except StopIteration as error:
            raise WebSocketDisconnect() from error
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code=None, reason=None):
        self.close_calls.append((code, reason))


def _handler_connection(subscriptions=None):
    connection = SimpleNamespace(
        subscriptions=set(subscriptions or ()),
        get_connection_info=lambda: {"outbound_queue_depth": 0},
    )
    connection.add_subscription = connection.subscriptions.add
    connection.remove_subscription = connection.subscriptions.discard
    return connection


@pytest.mark.asyncio
async def test_rate_limited_endpoint_rejects_before_creating_a_connection(monkeypatch):
    socket = FakeWebSocket()
    handler = SimpleNamespace(connect=AsyncMock())
    limiter = SimpleNamespace(check_rate_limit=AsyncMock(return_value=False))
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "rate_limiter", limiter)

    await endpoints.websocket_endpoint(socket, client_ip="192.0.2.17")

    limiter.check_rate_limit.assert_awaited_once_with("192.0.2.17")
    handler.connect.assert_not_awaited()
    assert socket.close_calls == [
        (status.WS_1008_POLICY_VIOLATION, "Rate limit exceeded")
    ]


@pytest.mark.asyncio
async def test_management_subscription_endpoint_reports_valid_and_invalid_streams(
    monkeypatch,
):
    manager = SimpleNamespace(subscribe_to_stream=AsyncMock(return_value=True))
    monkeypatch.setattr(
        endpoints,
        "websocket_handler",
        SimpleNamespace(connections={"client": object()}),
    )
    monkeypatch.setattr(endpoints, "websocket_manager", manager)

    response = await endpoints.subscribe_client_to_streams(
        "client", SubscribeRequest(streams=["state", "invalid"])
    )

    assert response == {
        "results": [
            {"stream": "state", "success": True},
            {"stream": "invalid", "success": False, "error": "Invalid stream type"},
        ]
    }
    manager.subscribe_to_stream.assert_awaited_once_with("client", "state")


@pytest.mark.asyncio
async def test_management_routes_report_missing_connections_and_unsubscribe_results(
    monkeypatch,
):
    manager = SimpleNamespace(unsubscribe_from_stream=AsyncMock(return_value=False))
    handler = SimpleNamespace(connections={"client": object()}, disconnect=AsyncMock())
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "websocket_manager", manager)

    with pytest.raises(endpoints.HTTPException) as missing:
        await endpoints.subscribe_client_to_streams(
            "missing", SubscribeRequest(streams=["state"])
        )
    assert missing.value.status_code == 404

    response = await endpoints.unsubscribe_client_from_streams(
        "client", UnsubscribeRequest(streams=["state", "future-stream"])
    )
    assert response == {
        "results": [
            {"stream": "state", "success": False},
            {"stream": "future-stream", "success": False},
        ]
    }
    assert manager.unsubscribe_from_stream.await_count == 2

    with pytest.raises(endpoints.HTTPException) as missing_disconnect:
        await endpoints.disconnect_client("missing")
    assert missing_disconnect.value.status_code == 404
    assert await endpoints.disconnect_client("client") == {
        "message": "Client client disconnected successfully"
    }
    handler.disconnect.assert_awaited_once_with("client")


@pytest.mark.asyncio
async def test_unavailable_admission_store_closes_with_retryable_status(monkeypatch):
    socket = FakeWebSocket()
    handler = SimpleNamespace(connect=AsyncMock())
    limiter = SimpleNamespace(
        check_rate_limit=AsyncMock(side_effect=RateLimitUnavailable("offline"))
    )
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "rate_limiter", limiter)

    await endpoints.websocket_endpoint(socket, client_ip="192.0.2.17")

    handler.connect.assert_not_awaited()
    assert socket.close_calls == [
        (status.WS_1013_TRY_AGAIN_LATER, "Rate limit service unavailable")
    ]


@pytest.mark.asyncio
async def test_endpoint_protocol_errors_are_isolated_and_disconnect_cleans_up(
    monkeypatch,
):
    socket = FakeWebSocket(["{", "valid-looking-input"])
    handler = SimpleNamespace(
        connect=AsyncMock(return_value="client-1"),
        disconnect=AsyncMock(),
        handle_message=AsyncMock(
            side_effect=[
                json.JSONDecodeError("invalid", "{", 1),
                RuntimeError("isolated handler failure"),
            ]
        ),
        send_to_client=AsyncMock(return_value=True),
    )
    limiter = SimpleNamespace(check_rate_limit=AsyncMock(return_value=True))
    monitor = SimpleNamespace(
        client_metrics={},
        record_message_received=Mock(),
        record_error=Mock(),
    )
    remove_subscriptions = AsyncMock()
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "rate_limiter", limiter)
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)
    monkeypatch.setattr(
        endpoints.subscription_manager, "remove_all_subscriptions", remove_subscriptions
    )

    await endpoints.websocket_endpoint(socket, client_ip="192.0.2.17")

    assert handler.handle_message.await_count == 2
    error_message = handler.send_to_client.await_args.args[1]
    assert error_message["type"] == "error"
    assert error_message["data"]["code"] == "INVALID_JSON"
    assert error_message["data"]["request_id"]
    assert monitor.record_message_received.call_args_list == [
        (("client-1", 1),),
        (("client-1", 19),),
    ]
    monitor.record_error.assert_called_once_with("client-1", "protocol_error")
    handler.disconnect.assert_awaited_once_with("client-1")
    remove_subscriptions.assert_awaited_once_with("client-1")


@pytest.mark.asyncio
async def test_manager_broadcast_removes_only_failed_clients_and_keeps_fanout_best_effort(
    monkeypatch,
):
    manager = WebSocketManager()
    await manager.register_client("healthy")
    await manager.register_client("false-send")
    await manager.register_client("raised-send")
    manager.stream_subscribers[StreamType.STATE].update(
        {"healthy", "false-send", "raised-send"}
    )
    received = {}

    async def send(client_id, message):
        received[client_id] = message
        if client_id == "false-send":
            return False
        if client_id == "raised-send":
            raise OSError("client disconnected")
        return True

    monkeypatch.setattr(
        manager_module, "websocket_handler", SimpleNamespace(send_to_client=send)
    )

    await manager.broadcast_to_stream("state", {"source_frame_number": 19, "balls": []})

    assert set(received) == {"healthy", "false-send", "raised-send"}
    assert received["healthy"]["type"] == "state"
    assert received["healthy"]["data"]["source_frame_number"] == 19
    assert manager.stream_subscribers[StreamType.STATE] == {"healthy"}


@pytest.mark.asyncio
async def test_manager_drops_filtered_state_without_sending_it(monkeypatch):
    manager = WebSocketManager()
    await manager.register_client("client")
    await manager.subscribe_to_stream(
        "client",
        StreamType.STATE,
        SubscriptionFilter(
            stream_type=StreamType.STATE,
            conditions={"phase": "aiming"},
        ),
    )
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(
        manager_module, "websocket_handler", SimpleNamespace(send_to_client=send)
    )

    await manager.broadcast_to_stream(
        StreamType.STATE, {"phase": "settled", "frame_number": 22}
    )

    send.assert_not_awaited()
    assert manager.stream_subscribers[StreamType.STATE] == {"client"}


@pytest.mark.asyncio
async def test_manager_cleans_stale_subscribers_and_reports_session_counts():
    manager = WebSocketManager()
    await manager.register_client("live")
    manager.stream_subscribers[StreamType.STATE].update({"live", "stale"})
    manager.stream_subscribers[StreamType.FRAME].add("stale")

    cleanup = await manager.cleanup_stale_subscribers()
    sessions = await manager.get_all_sessions()

    assert cleanup == {"frame": 1, "state": 1}
    assert manager.stream_subscribers[StreamType.STATE] == {"live"}
    assert sessions["total_sessions"] == 1
    assert sessions["stream_subscribers"]["state"] == 1
    assert [session["client_id"] for session in sessions["sessions"]] == ["live"]


@pytest.mark.asyncio
async def test_unregister_removes_every_stream_and_cancels_reconnect_task():
    manager = WebSocketManager()
    await manager.register_client("client")
    for stream in StreamType:
        manager.stream_subscribers[stream].add("client")

    async def wait_until_cancelled():
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_until_cancelled())
    manager.reconnect_tasks["client"] = task
    await manager.unregister_client("client")
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "client" not in manager.sessions
    assert all(
        "client" not in subscribers
        for subscribers in manager.stream_subscribers.values()
    )
    assert "client" not in manager.reconnect_tasks


@pytest.mark.asyncio
async def test_connection_loss_schedules_capped_backoff_and_emits_reconnect_event(
    monkeypatch,
):
    manager = WebSocketManager()
    session = await manager.register_client("client")
    session.reconnect_delay = 8.0
    session.max_reconnect_attempts = 2
    manager.auto_reconnect_enabled = True
    delays = []
    events = []

    async def no_wait(delay):
        delays.append(delay)

    async def collect(event):
        events.append(event)

    monkeypatch.setattr(manager_module.config, "get", lambda key, default=None: 2.5)
    monkeypatch.setattr(manager_module.asyncio, "sleep", no_wait)
    manager.add_event_handler("reconnect_attempted", collect)

    await manager.handle_connection_lost("client")
    task = manager.reconnect_tasks["client"]
    await task

    assert delays == [2.5]
    assert events[0]["attempt"] == 1
    assert events[0]["delay"] == 2.5
    assert manager.sessions["client"].connection_state is ConnectionState.CONNECTING
    assert manager.sessions["client"].reconnect_attempts == 1
    assert "client" not in manager.reconnect_tasks


@pytest.mark.asyncio
async def test_connection_loss_unregisters_when_reconnect_is_disabled():
    manager = WebSocketManager()
    await manager.register_client("client")
    manager.auto_reconnect_enabled = False

    await manager.handle_connection_lost("client")

    assert "client" not in manager.sessions


@pytest.mark.asyncio
async def test_manager_event_handlers_support_sync_async_and_isolate_failures():
    manager = WebSocketManager()
    seen = []

    def sync_handler(data):
        seen.append(("sync", data["client_id"]))

    async def async_handler(data):
        seen.append(("async", data["client_id"]))

    def broken_handler(data):
        raise RuntimeError("observer failure")

    manager.add_event_handler("client_registered", sync_handler)
    manager.add_event_handler("client_registered", async_handler)
    manager.add_event_handler("client_registered", broken_handler)
    await manager.register_client("anonymous")
    manager.remove_event_handler("client_registered", sync_handler)
    manager.remove_event_handler("missing", sync_handler)

    assert seen == [("sync", "anonymous"), ("async", "anonymous")]
    assert manager.sessions["anonymous"].metadata == {}


@pytest.mark.asyncio
async def test_subscription_methods_reject_unknown_clients_and_stream_names(
    monkeypatch,
):
    manager = WebSocketManager()
    send_connection = _handler_connection()
    monkeypatch.setattr(
        manager_module,
        "websocket_handler",
        SimpleNamespace(connections={"client": send_connection}),
    )

    assert not await manager.subscribe_to_stream("missing", "state")
    assert not await manager.unsubscribe_from_stream("missing", "state")
    await manager.register_client("client")
    assert not await manager.subscribe_to_stream("client", "not-a-stream")
    assert not await manager.unsubscribe_from_stream("client", "not-a-stream")
    assert await manager.subscribe_to_stream("client", "state")
    assert send_connection.subscriptions == {"state"}
    assert await manager.unsubscribe_from_stream("client", "state")
    assert send_connection.subscriptions == set()


@pytest.mark.asyncio
async def test_message_filter_applies_field_selection_condition_and_frame_quality():
    manager = WebSocketManager()
    filter_config = SubscriptionFilter(
        stream_type=StreamType.FRAME,
        include_fields=["image", "phase"],
        conditions={"phase": "aiming"},
        quality_level="low",
    )
    message = {
        "type": "frame",
        "data": {"image": "encoded", "phase": "aiming", "internal": "omit"},
    }

    filtered = await manager._apply_message_filter(message, filter_config)
    rejected = await manager._apply_message_filter(
        {"type": "frame", "data": {"phase": "settled"}}, filter_config
    )

    assert filtered is not None
    assert filtered["data"] == {"image": "encoded", "phase": "aiming", "quality": "low"}
    assert message["data"] == {
        "image": "encoded",
        "phase": "aiming",
        "internal": "omit",
    }
    assert rejected is None


@pytest.mark.asyncio
async def test_targeted_alert_skips_unregistered_clients(monkeypatch):
    manager = WebSocketManager()
    await manager.register_client("live")
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(
        manager_module, "websocket_handler", SimpleNamespace(send_to_client=send)
    )

    await manager.send_alert(
        "warning", "current alert", "TABLE_STATE", target_clients=["live", "gone"]
    )

    send.assert_awaited_once()
    assert send.await_args.args[0] == "live"
    assert send.await_args.args[1]["type"] == "alert"
    assert send.await_args.args[1]["data"]["code"] == "TABLE_STATE"


@pytest.mark.asyncio
async def test_session_info_uses_handler_metadata_when_a_connection_is_present(
    monkeypatch,
):
    manager = WebSocketManager()
    await manager.register_client("client", metadata={"source": "local"})
    connection = _handler_connection({"state"})
    monkeypatch.setattr(
        manager_module,
        "websocket_handler",
        SimpleNamespace(connections={"client": connection}),
    )

    info = await manager.get_session_info("client")
    absent = await manager.get_session_info("missing")

    assert info is not None
    assert info["metadata"] == {"source": "local"}
    assert info["subscriptions"] == []
    assert info["outbound_queue_depth"] == 0
    assert absent is None
