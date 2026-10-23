"""Integration coverage for WebSocket subscription contracts and delivery."""

import json

import pytest

from backend.api.websocket import schemas as websocket_schemas
from backend.api.websocket.handler import WebSocketConnection, WebSocketHandler
from backend.api.websocket.manager import (
    StreamType,
    SubscriptionFilter,
    WebSocketManager,
)
from backend.api.websocket.subscriptions import SubscriptionManager


class RecordingSocket:
    """Minimal socket implementation that records handler output."""

    def __init__(self):
        self.messages = []

    async def accept(self):
        pass

    async def send_text(self, message):
        self.messages.append(message)

    async def close(self, code=None):
        pass


@pytest.mark.integration
def test_stream_types_have_one_shared_source_of_truth():
    """Manager, schemas, and REST validation expose the same streams."""
    assert StreamType is websocket_schemas.StreamType
    assert [
        stream_type.value for stream_type in StreamType
    ] == websocket_schemas.VALID_STREAM_TYPES
    assert {"frame", "state", "trajectory", "alert", "config"} == {
        stream_type.value for stream_type in StreamType
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_canonical_subscription_methods_preserve_supported_options():
    """The public lifecycle methods store the full supported configuration."""
    manager = SubscriptionManager()
    options = {
        "quality_level": "high",
        "max_fps": 24,
        "min_fps": 10,
        "include_fields": ["balls"],
        "exclude_fields": ["debug"],
        "filter_conditions": [{"field": "active", "operator": "eq", "value": True}],
        "enable_aggregation": True,
        "aggregation_window_ms": 40,
        "sample_rate": 0.5,
        "sample_offset": 0.25,
        "max_latency_ms": 80,
    }

    assert await manager.create_subscription("client", "state", **options)
    subscription = manager.subscriptions["client"][StreamType.STATE]
    assert subscription.quality_level.value == "high"
    assert subscription.max_fps == 24
    assert subscription.min_fps == 10
    assert subscription.include_fields == {"balls"}
    assert subscription.exclude_fields == {"debug"}
    assert subscription.filter_conditions[0].field_path == "active"
    assert subscription.enable_aggregation is True
    assert subscription.aggregation_window_ms == 40
    assert subscription.sample_rate == 0.5
    assert subscription.sample_offset == 0.25
    assert subscription.max_latency_ms == 80

    assert await manager.remove_subscription("client", "state") is True
    assert "client" not in manager.subscriptions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscription_manager_filters_messages_for_each_client():
    manager = SubscriptionManager()
    assert await manager.create_subscription(
        "client",
        StreamType.STATE,
        include_fields=["score", "active"],
        filter_conditions=[{"field": "active", "operator": "eq", "value": True}],
    )

    delivered = await manager.process_message(
        StreamType.STATE,
        {"score": 12, "active": True, "private": "hidden"},
    )
    filtered = await manager.process_message(
        StreamType.STATE,
        {"score": 8, "active": False, "private": "hidden"},
    )

    assert delivered == {"client": {"score": 12, "active": True}}
    assert filtered == {}
    subscription = manager.subscriptions["client"][StreamType.STATE]
    assert subscription.messages_sent == 1
    assert subscription.messages_filtered == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anonymous_handler_subscribes_and_unsubscribes_without_permission_token():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "client")
    handler.connections["client"] = connection
    await manager.register_client("client")

    await handler._handle_subscribe(
        "client",
        {
            "data": {
                "streams": ["state", "unknown"],
                "filters": {
                    "state": {
                        "include_fields": ["score"],
                        "conditions": {"active": True},
                    }
                },
            }
        },
    )
    await handler.wait_for_outbound("client")

    subscribed = json.loads(socket.messages[-1])
    assert subscribed["type"] == "subscribed"
    assert subscribed["data"]["streams"] == ["state"]
    assert connection.subscriptions == {"state"}
    assert manager.stream_subscribers[StreamType.STATE] == {"client"}
    assert manager.sessions["client"].subscription_filters[
        StreamType.STATE
    ].include_fields == ["score"]

    await handler._handle_unsubscribe(
        "client", {"data": {"streams": ["state", "unknown"]}}
    )
    await handler.wait_for_outbound("client")

    unsubscribed = json.loads(socket.messages[-1])
    assert unsubscribed["type"] == "unsubscribed"
    assert unsubscribed["data"]["streams"] == ["state"]
    assert connection.subscriptions == set()
    assert manager.stream_subscribers[StreamType.STATE] == set()
    assert StreamType.STATE not in manager.sessions["client"].subscription_filters
    await handler.disconnect("client")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unregistered_connection_cannot_subscribe_without_lifecycle_session():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "client")
    handler.connections["client"] = connection
    await handler._handle_subscribe("client", {"data": {"streams": ["state"]}})
    await handler.wait_for_outbound("client")

    response = json.loads(socket.messages[-1])
    assert response["type"] == "subscribed"
    assert response["data"]["streams"] == []
    assert connection.subscriptions == set()
    assert manager.stream_subscribers[StreamType.STATE] == set()
    await handler.disconnect("client")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anonymous_registered_client_can_subscribe_to_every_supported_stream():
    manager = WebSocketManager()
    await manager.register_client("anonymous-client")

    for stream_type in StreamType:
        assert await manager.subscribe_to_stream("anonymous-client", stream_type)

    assert all(
        subscribers == {"anonymous-client"}
        for subscribers in manager.stream_subscribers.values()
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_outputs_contain_no_user_or_permission_fields():
    manager = WebSocketManager()
    await manager.register_client("anonymous-client")

    session = await manager.get_session_info("anonymous-client")
    all_sessions = await manager.get_all_sessions()

    assert session is not None
    assert "user_id" not in session
    assert "permissions" not in session
    assert "total_users" not in all_sessions
    assert len(all_sessions["sessions"]) == 1
    assert all_sessions["sessions"][0]["client_id"] == "anonymous-client"
    assert "user_id" not in all_sessions["sessions"][0]
    assert "permissions" not in all_sessions["sessions"][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alert_delivery_keeps_per_client_targeting_and_anonymous_broadcasts(
    monkeypatch,
):
    from backend.api.websocket import manager as manager_module

    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    sockets = {client_id: RecordingSocket() for client_id in ("first", "second")}
    for client_id, socket in sockets.items():
        handler.connections[client_id] = WebSocketConnection(socket, client_id)
        await manager.register_client(client_id)
        assert await manager.subscribe_to_stream(client_id, StreamType.ALERT)
    monkeypatch.setattr(manager_module, "websocket_handler", handler)

    await manager.send_alert(
        "warning", "targeted", "TARGETED", target_clients=["first"]
    )
    await manager.send_alert("info", "broadcast", "BROADCAST")
    await handler.wait_for_outbound("first")
    await handler.wait_for_outbound("second")

    first_messages = [json.loads(message) for message in sockets["first"].messages]
    second_messages = [json.loads(message) for message in sockets["second"].messages]
    assert [message["data"]["code"] for message in first_messages] == [
        "TARGETED",
        "BROADCAST",
    ]
    assert [message["data"]["code"] for message in second_messages] == ["BROADCAST"]

    await handler.disconnect("first")
    await handler.disconnect("second")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_handler_errors_include_retained_connection_request_id():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    socket = RecordingSocket()
    metadata = {"request_id": "request-123", "user_agent": "test-client"}
    connection = WebSocketConnection(socket, "client", metadata=metadata)
    handler.connections["client"] = connection

    await handler.handle_message("client", '{"type":"not_supported"}')
    await handler.wait_for_outbound("client")

    error_message = json.loads(socket.messages[-1])
    assert error_message["type"] == "error"
    assert error_message["data"]["request_id"] == "request-123"
    assert connection.metadata == metadata
    assert connection.get_connection_info()["request_id"] == "request-123"
    await handler.disconnect("client")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_manager_delivers_filtered_message(monkeypatch):
    """The manager's registered subscriber receives its configured filtered data."""
    from backend.api.websocket import manager as manager_module

    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    socket = RecordingSocket()
    handler.connections["client"] = WebSocketConnection(socket, "client")
    monkeypatch.setattr(manager_module, "websocket_handler", handler)

    await manager.register_client("client")
    assert await manager.subscribe_to_stream(
        "client",
        StreamType.STATE,
        SubscriptionFilter(
            stream_type=StreamType.STATE,
            include_fields=["score"],
        ),
    )
    await manager.broadcast_to_stream(
        StreamType.STATE, {"score": 15, "internal": "not for clients"}
    )
    await handler.wait_for_outbound("client")

    assert len(socket.messages) == 1
    delivered = json.loads(socket.messages[0])
    assert delivered["type"] == "state"
    assert delivered["data"] == {"score": 15}
    await handler.disconnect("client")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disconnect_clears_manager_and_handler_subscription_state():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    connection = WebSocketConnection(RecordingSocket(), "client")
    connection.add_subscription("state")
    handler.connections["client"] = connection
    await manager.register_client("client")
    assert await manager.subscribe_to_stream("client", StreamType.STATE)

    await handler.disconnect("client")

    assert "client" not in handler.connections
    assert "client" not in manager.sessions
    assert all(
        "client" not in subscribers
        for subscribers in manager.stream_subscribers.values()
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connect_registers_session_before_connection_is_ready():
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)

    client_id = await handler.connect(RecordingSocket())

    assert client_id in handler.connections
    assert client_id in manager.sessions
    assert manager.sessions[client_id].client_id == client_id
    assert not hasattr(manager.sessions[client_id], "user_id")
    assert not hasattr(manager.sessions[client_id], "permissions")
    await handler.disconnect(client_id)
    await handler.disconnect(client_id)
