"""Versioned golden contracts for client-driven WebSocket control messages."""

import copy
import json
from pathlib import Path

import pytest

from backend.api.websocket.handler import WebSocketConnection, WebSocketHandler
from backend.api.websocket.manager import WebSocketManager

CONTRACT_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/contracts/v1/websocket_controls.json"
)
CONTRACT_TIME = "2025-01-01T00:00:00+00:00"


class RecordingSocket:
    """Minimal socket implementation that records production handler output."""

    def __init__(self):
        self.messages = []

    async def accept(self):
        pass

    async def send_text(self, message):
        self.messages.append(message)

    async def close(self, code=None):
        pass


def _normalize_control_timestamps(message):
    """Normalize dynamic wall-clock fields while preserving the wire shape."""
    normalized = copy.deepcopy(message)
    normalized["timestamp"] = CONTRACT_TIME
    if normalized["type"] == "pong":
        normalized["data"]["timestamp"] = CONTRACT_TIME
    return normalized


def _control_fixture():
    return json.loads(CONTRACT_FIXTURE.read_text(encoding="utf-8"))


async def _send_client_message(message, *, metadata=None):
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "client", metadata=metadata)
    handler.connections["client"] = connection
    # Anonymous clients receive lifecycle state without identity or grants.
    await manager.register_client("client", metadata=metadata)

    await handler.handle_message("client", json.dumps(message))
    await handler.wait_for_outbound("client")
    response = json.loads(socket.messages[-1])
    await handler.disconnect("client")
    return response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_subscription_ack_matches_versioned_websocket_control_fixture():
    response = await _send_client_message(
        {"type": "subscribe", "data": {"streams": ["state"]}}
    )

    assert (
        _normalize_control_timestamps(response)
        == _control_fixture()["subscription_ack"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_protocol_error_matches_versioned_websocket_control_fixture():
    response = await _send_client_message(
        {"type": "not_supported"}, metadata={"request_id": "request-123"}
    )

    assert (
        _normalize_control_timestamps(response)
        == _control_fixture()["unknown_message_error"]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ping_pong_matches_versioned_websocket_control_fixture():
    response = await _send_client_message({"type": "ping", "data": {}})

    assert _normalize_control_timestamps(response) == _control_fixture()["pong"]
