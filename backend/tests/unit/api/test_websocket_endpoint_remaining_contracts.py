"""Behavior contracts for WebSocket endpoint management and failure paths."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from backend.api.websocket import endpoints
from backend.api.websocket.schemas import SubscribeRequest, UnsubscribeRequest


class _Socket:
    """Minimal socket that disconnects after its configured inbound messages."""

    client = SimpleNamespace(host="192.0.2.40")
    headers = {"user-agent": "endpoint-contract"}

    def __init__(self, messages: tuple[Any, ...] | list[Any] = ()) -> None:
        self.messages = iter(messages)

    async def receive_text(self) -> str:
        try:
            message = next(self.messages)
        except StopIteration as error:
            raise WebSocketDisconnect() from error
        if isinstance(message, Exception):
            raise message
        return cast("str", message)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        return None


@pytest.mark.asyncio
async def test_message_monitor_failure_is_isolated_and_connection_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = SimpleNamespace(
        connect=AsyncMock(return_value="slow-client"),
        disconnect=AsyncMock(),
        handle_message=AsyncMock(),
    )
    metrics: dict[str, Any] = {}
    monitor = SimpleNamespace(
        client_metrics=metrics,
        record_message_received=Mock(side_effect=RuntimeError("monitor unavailable")),
        record_error=Mock(),
    )
    remove_subscriptions = AsyncMock()
    monkeypatch.setattr(
        endpoints,
        "rate_limiter",
        SimpleNamespace(check_rate_limit=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)
    monkeypatch.setattr(
        endpoints.subscription_manager, "remove_all_subscriptions", remove_subscriptions
    )

    await endpoints.websocket_endpoint(
        cast("WebSocket", _Socket(["{}"])), client_ip="192.0.2.40"
    )

    handler.handle_message.assert_not_awaited()
    monitor.record_error.assert_called_once_with("slow-client", "protocol_error")
    handler.disconnect.assert_awaited_once_with("slow-client")
    remove_subscriptions.assert_awaited_once_with("slow-client")


@pytest.mark.asyncio
async def test_failed_error_delivery_closes_connection_and_removes_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = SimpleNamespace(
        connect=AsyncMock(return_value="slow-client"),
        disconnect=AsyncMock(),
        handle_message=AsyncMock(side_effect=json.JSONDecodeError("invalid", "{", 0)),
        send_to_client=AsyncMock(side_effect=OSError("writer is unavailable")),
    )
    monitor = SimpleNamespace(
        client_metrics={},
        record_message_received=Mock(),
        record_error=Mock(),
    )
    remove_subscriptions = AsyncMock()
    monkeypatch.setattr(
        endpoints,
        "rate_limiter",
        SimpleNamespace(check_rate_limit=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)
    monkeypatch.setattr(
        endpoints.subscription_manager, "remove_all_subscriptions", remove_subscriptions
    )

    await endpoints.websocket_endpoint(
        cast("WebSocket", _Socket(["{"])), client_ip="192.0.2.40"
    )

    handler.send_to_client.assert_awaited_once()
    monitor.record_error.assert_called_once_with("slow-client", "disconnection")
    handler.disconnect.assert_awaited_once_with("slow-client")
    remove_subscriptions.assert_awaited_once_with("slow-client")


@pytest.mark.asyncio
async def test_get_active_connections_returns_stats_and_wraps_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = SimpleNamespace(get_connection_stats=Mock(return_value={"active": 2}))
    manager = SimpleNamespace(
        get_all_sessions=AsyncMock(return_value={"total_sessions": 2})
    )
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "websocket_manager", manager)

    response = await endpoints.get_active_connections()

    assert response["handler_stats"] == {"active": 2}
    assert response["session_info"] == {"total_sessions": 2}
    assert response["timestamp"]

    manager.get_all_sessions.side_effect = RuntimeError("manager unavailable")
    with pytest.raises(HTTPException) as error:
        await endpoints.get_active_connections()
    assert error.value.status_code == 500
    assert error.value.detail == "Failed to retrieve connection information"


@pytest.mark.asyncio
async def test_connection_detail_combines_session_health_and_subscriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SimpleNamespace(
        get_session_info=AsyncMock(return_value={"client_id": "c1"})
    )
    monitor = SimpleNamespace(
        get_client_health=Mock(return_value={"health_status": "good"})
    )
    subscriptions = SimpleNamespace(
        get_subscription_info=AsyncMock(return_value={"state": True})
    )
    monkeypatch.setattr(endpoints, "websocket_manager", manager)
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)
    monkeypatch.setattr(endpoints, "subscription_manager", subscriptions)

    response = await endpoints.get_connection_info("c1")

    assert response["session"] == {"client_id": "c1"}
    assert response["health"] == {"health_status": "good"}
    assert response["subscriptions"] == {"state": True}
    assert response["timestamp"]

    manager.get_session_info.return_value = None
    with pytest.raises(HTTPException) as missing:
        await endpoints.get_connection_info("gone")
    assert missing.value.status_code == 404

    manager.get_session_info.side_effect = RuntimeError("session store failed")
    with pytest.raises(HTTPException) as failed:
        await endpoints.get_connection_info("c1")
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_management_routes_distinguish_missing_clients_and_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = SimpleNamespace(connections={"c1": object()}, disconnect=AsyncMock())
    manager = SimpleNamespace(
        subscribe_to_stream=AsyncMock(return_value=True),
        unsubscribe_from_stream=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "websocket_manager", manager)

    cases: list[tuple[Any, Any, Any]] = [
        (
            endpoints.subscribe_client_to_streams,
            SubscribeRequest(streams=["state"], filters=None),
            manager.subscribe_to_stream,
        ),
        (
            endpoints.unsubscribe_client_from_streams,
            UnsubscribeRequest(streams=["state"]),
            manager.unsubscribe_from_stream,
        ),
    ]
    for operation, request, method in cases:
        with pytest.raises(HTTPException) as missing:
            await operation("gone", request)
        assert missing.value.status_code == 404

        method.side_effect = RuntimeError("manager failed")
        with pytest.raises(HTTPException) as failed:
            await operation("c1", request)
        assert failed.value.status_code == 500
        assert failed.value.detail == "Failed to update subscriptions"
        method.side_effect = None
        method.return_value = True

    handler.disconnect.side_effect = RuntimeError("disconnect failed")
    with pytest.raises(HTTPException) as failed_disconnect:
        await endpoints.disconnect_client("c1")
    assert failed_disconnect.value.status_code == 500


@pytest.mark.asyncio
async def test_health_endpoints_return_live_data_and_convert_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = SimpleNamespace(
        get_system_health=Mock(return_value={"status": "healthy"}),
        get_health_summary=Mock(return_value={"clients": 1}),
    )
    broadcaster = SimpleNamespace(get_broadcast_stats=Mock(return_value={"sent": 3}))
    subscriptions = SimpleNamespace(
        get_performance_stats=Mock(return_value={"active": 1})
    )
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)
    monkeypatch.setattr(endpoints, "message_broadcaster", broadcaster)
    monkeypatch.setattr(endpoints, "subscription_manager", subscriptions)

    health = await endpoints.get_websocket_health()
    assert health["system_health"] == {"status": "healthy"}
    assert health["broadcaster_stats"] == {"sent": 3}
    assert health["subscription_stats"] == {"active": 1}
    assert health["timestamp"]
    assert await endpoints.get_health_summary() == {"clients": 1}

    monitor.get_system_health.side_effect = RuntimeError("monitor failed")
    with pytest.raises(HTTPException) as health_error:
        await endpoints.get_websocket_health()
    assert health_error.value.status_code == 500
    monitor.get_health_summary.side_effect = RuntimeError("summary failed")
    with pytest.raises(HTTPException) as summary_error:
        await endpoints.get_health_summary()
    assert summary_error.value.status_code == 500


@pytest.mark.asyncio
async def test_test_frame_endpoint_builds_and_broadcasts_requested_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = SimpleNamespace(broadcast_frame=AsyncMock())
    monkeypatch.setattr(endpoints, "message_broadcaster", broadcaster)

    response = await endpoints.broadcast_test_frame(width=120, height=110, quality=61)

    call = broadcaster.broadcast_frame.await_args.kwargs
    assert call["image_data"].shape == (110, 120, 3)
    assert str(call["image_data"].dtype) == "uint8"
    assert call["width"] == 120
    assert call["height"] == 110
    assert call["quality"] == 61
    assert response["message"] == "Test frame broadcasted successfully"
    assert response["timestamp"]

    broadcaster.broadcast_frame.side_effect = RuntimeError("broadcast failed")
    with pytest.raises(HTTPException) as failed:
        await endpoints.broadcast_test_frame(width=120, height=110)
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_alert_broadcast_and_critical_health_alert_keep_payload_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = SimpleNamespace(broadcast_alert=AsyncMock())
    monkeypatch.setattr(endpoints, "message_broadcaster", broadcaster)

    assert await endpoints.broadcast_test_alert("warning", "watch", "TABLE") == {
        "message": "Test alert broadcasted successfully"
    }
    call = broadcaster.broadcast_alert.await_args.kwargs
    assert (call["level"], call["message"], call["code"]) == (
        "warning",
        "watch",
        "TABLE",
    )
    assert call["details"]["test"] is True

    broadcaster.broadcast_alert.reset_mock()
    await endpoints.handle_health_alert({"type": "routine", "message": "ignored"})
    broadcaster.broadcast_alert.assert_not_awaited()
    alert = {"type": "critical_connections", "message": "too many clients"}
    await endpoints.handle_health_alert(alert)
    assert broadcaster.broadcast_alert.await_args.kwargs == {
        "level": "warning",
        "message": "too many clients",
        "code": "HEALTH_ALERT",
        "details": alert,
    }

    broadcaster.broadcast_alert.side_effect = RuntimeError("alert failed")
    with pytest.raises(HTTPException) as failed:
        await endpoints.broadcast_test_alert()
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_system_start_and_stop_sequence_services_and_disconnect_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = SimpleNamespace(
        start_streaming=AsyncMock(), stop_streaming=AsyncMock()
    )
    handler = SimpleNamespace(
        start_monitoring=AsyncMock(),
        stop_monitoring=AsyncMock(),
        disconnect=AsyncMock(),
        connections={"first": object(), "second": object()},
    )
    monitor = SimpleNamespace(start_monitoring=AsyncMock(), stop_monitoring=AsyncMock())
    monkeypatch.setattr(endpoints, "message_broadcaster", broadcaster)
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "connection_monitor", monitor)

    assert await endpoints.start_websocket_system() == {
        "message": "WebSocket system started successfully"
    }
    assert await endpoints.stop_websocket_system() == {
        "message": "WebSocket system stopped successfully"
    }
    broadcaster.start_streaming.assert_awaited_once()
    handler.start_monitoring.assert_awaited_once()
    monitor.start_monitoring.assert_awaited_once()
    broadcaster.stop_streaming.assert_awaited_once()
    handler.stop_monitoring.assert_awaited_once()
    monitor.stop_monitoring.assert_awaited_once()
    assert [call.args[0] for call in handler.disconnect.await_args_list] == [
        "first",
        "second",
    ]

    monitor.start_monitoring.side_effect = RuntimeError("start failed")
    with pytest.raises(HTTPException) as start_error:
        await endpoints.start_websocket_system()
    assert start_error.value.status_code == 500
    handler.disconnect.side_effect = RuntimeError("stop failed")
    with pytest.raises(HTTPException) as stop_error:
        await endpoints.stop_websocket_system()
    assert stop_error.value.status_code == 500


def _broadcast_stats() -> SimpleNamespace:
    return SimpleNamespace(
        messages_sent=7,
        bytes_sent=101,
        failed_sends=1,
        average_latency=0.02,
        peak_latency=0.08,
        compression_enabled=True,
        frame_metrics=SimpleNamespace(
            frames_received=4,
            frames_sent=3,
            bytes_sent=99,
            compression_ratio=0.5,
            average_latency=0.03,
            dropped_frames=1,
            target_fps=30,
            actual_fps=24,
        ),
    )


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_delivery_counters_and_wraps_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcaster = SimpleNamespace(
        broadcast_stats=_broadcast_stats(),
        get_broadcast_stats=Mock(return_value={"clients": 2}),
    )
    handler = SimpleNamespace(get_connection_stats=Mock(return_value={"active": 2}))
    monkeypatch.setattr(endpoints, "message_broadcaster", broadcaster)
    monkeypatch.setattr(endpoints, "websocket_handler", handler)

    response = await endpoints.get_websocket_metrics()

    assert response["broadcast_stats"]["messages_sent"] == 7
    assert response["frame_metrics"]["dropped_frames"] == 1
    assert response["frame_metrics"]["actual_fps"] == 24
    assert response["connection_stats"] == {"active": 2}
    assert response["broadcaster_stats"] == {"clients": 2}
    assert response["timestamp"]

    handler.get_connection_stats.side_effect = RuntimeError("metrics failed")
    with pytest.raises(HTTPException) as failed:
        await endpoints.get_websocket_metrics()
    assert failed.value.status_code == 500
