"""Deterministic tests for bounded, per-client WebSocket delivery."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import backend.api.websocket as websocket_package
import backend.api.websocket.handler as handler_module
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


class AuditRecorder:
    def __init__(self, selected: bool) -> None:
        self.selected = selected
        self.events: list[tuple[dict, bool]] = []
        self.selections: list[tuple[str, int | str | None, str]] = []
        self.sample_rates: list[float] = []

    def select_detail_sample(self, domain, identity, sample_rate, *, category="detail"):
        self.selections.append((domain, identity, category))
        self.sample_rates.append(sample_rate)
        return self.selected

    def record(self, event, *, force_keep=False):
        self.events.append((event, force_keep))
        return True


@pytest.mark.integration
def test_unsampled_websocket_message_skips_timing_and_keeps_delivery(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    recorder = AuditRecorder(selected=False)
    handler.configure_performance_audit(recorder, {"state": True})
    connection = WebSocketConnection(RecordingSocket(), "sampled-out")

    def fail_if_timed():
        raise AssertionError("unsampled serialization must not take audit clocks")

    monkeypatch.setattr(handler_module.time, "perf_counter_ns", fail_if_timed)
    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 12,
                "observation_status": {"source_frame_number": 91},
            },
        },
    )

    queued = connection.outbound_messages[0]
    assert queued.audit_sampled is False
    assert json.loads(queued.payload)["data"]["sequence"] == 12
    assert recorder.selections == [("state", 91, "websocket_message")]
    assert recorder.events == []


@pytest.mark.integration
def test_audit_activation_refreshes_sampling_from_selected_config(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    selected_config = SimpleNamespace(
        get=lambda key, default=None: (
            0.05 if key == "performance_audit.detailed_event_sample_rate" else default
        )
    )
    monkeypatch.setattr(handler_module, "config", selected_config)
    recorder = AuditRecorder(selected=True)
    handler.configure_performance_audit(recorder, {"state": True})
    connection = WebSocketConnection(RecordingSocket(), "selected-config")

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 1,
                "observation_status": {"source_frame_number": 1},
            },
        },
    )

    assert recorder.sample_rates == [0.05]


@pytest.mark.integration
def test_websocket_delivery_samples_details_but_forces_coalescing_anomaly():
    handler = WebSocketHandler(manager=WebSocketManager())
    recorder = AuditRecorder(selected=False)
    handler.configure_performance_audit(recorder, {"state": True})
    connection = WebSocketConnection(RecordingSocket(), "coalescing")

    for frame_number in (1, 2):
        assert handler._enqueue_message(
            connection,
            {
                "type": "state",
                "data": {
                    "sequence": frame_number,
                    "observation_status": {"source_frame_number": frame_number},
                },
            },
        )

    assert len(connection.outbound_messages) == 1
    assert json.loads(connection.outbound_messages[0].payload)["data"]["sequence"] == 2
    assert len(recorder.events) == 1
    event, force_keep = recorder.events[0]
    assert event["event_type"] == "websocket_update_coalesced"
    assert force_keep is True


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
def test_expired_observation_is_dropped_before_serialization_or_enqueue(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_state_observation_age_seconds = 0.5
    connection = WebSocketConnection(RecordingSocket(), "expired-observation")

    def fail_if_serialized(_message):
        raise AssertionError("expired state must be dropped before serialization")

    monkeypatch.setattr(handler_module.json, "dumps", fail_if_serialized)
    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 17,
                "observation_status": {
                    "source_frame_number": 901,
                    "frame_publication_age_seconds": 0.75,
                    "ball_positions_status": "observed",
                },
            },
        },
    )

    assert not connection.outbound_messages
    assert connection.dropped_updates == 1
    assert connection.expired_updates == 1
    assert handler.delivery_metrics["expired_updates"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_writer_drops_replaceable_update_that_ages_in_queue():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_replaceable_message_age_seconds = 0.01
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "expired-queue")
    assert handler._enqueue_message(
        connection,
        {"type": "state", "data": {"sequence": 18}},
    )
    connection.outbound_messages[0].created_monotonic_ns = (
        time.monotonic_ns() - 20_000_000
    )
    connection.is_alive = False

    await handler._writer_loop(connection)

    assert socket.messages == []
    assert connection.expired_updates == 1
    assert connection.dropped_updates == 1
    assert handler.delivery_metrics["expired_updates"] == 1
    assert not connection.replaceable_messages


@pytest.mark.integration
@pytest.mark.asyncio
async def test_writer_refreshes_python_state_age_at_delivery(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_state_observation_age_seconds = 2.0
    handler.max_replaceable_message_age_seconds = 1.0
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "refreshed-source-age")
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(handler_module.time, "monotonic_ns", lambda: clock["now_ns"])
    monkeypatch.setattr(handler_module.time, "perf_counter_ns", lambda: clock["now_ns"])

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 21,
                "observation_status": {
                    "vision_instance_id": "vision-instance",
                    "source_publication_monotonic_ns": 9_000_000_000,
                    "frame_publication_age_seconds": 0.1,
                    "ball_positions_status": "observed",
                },
            },
        },
    )
    # Simulate work and queueing between state serialization and its socket write.
    clock["now_ns"] += 400_000_000
    connection.is_alive = False

    await handler._writer_loop(connection)

    delivered = json.loads(socket.messages[0])
    assert delivered["data"]["observation_status"][
        "frame_publication_age_seconds"
    ] == pytest.approx(1.4)
    assert connection.expired_updates == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_writer_drops_python_state_that_expires_while_waiting(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_state_observation_age_seconds = 0.5
    handler.max_replaceable_message_age_seconds = 2.0
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "expired-source-age")
    clock = {"now_ns": 10_000_000_000}
    monkeypatch.setattr(handler_module.time, "monotonic_ns", lambda: clock["now_ns"])
    monkeypatch.setattr(handler_module.time, "perf_counter_ns", lambda: clock["now_ns"])

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 22,
                "observation_status": {
                    "vision_instance_id": "vision-instance",
                    "source_publication_monotonic_ns": 10_000_000_000,
                    "frame_publication_age_seconds": 0.0,
                    "ball_positions_status": "observed",
                },
            },
        },
    )
    # Keep the message in the bounded queue but advance beyond the source-age
    # deadline before the writer is allowed to deliver it.
    clock["now_ns"] += 600_000_000
    connection.is_alive = False

    await handler._writer_loop(connection)

    assert socket.messages == []
    assert connection.expired_updates == 1
    assert handler.delivery_metrics["expired_updates"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_writer_drops_python_state_without_comparable_publication_tick(
    monkeypatch,
):
    handler = WebSocketHandler(manager=WebSocketManager())
    socket = RecordingSocket()
    connection = WebSocketConnection(socket, "missing-source-clock")
    monkeypatch.setattr(handler_module.time, "monotonic_ns", lambda: 10_000_000_000)
    monkeypatch.setattr(handler_module.time, "perf_counter_ns", lambda: 10_000_000_000)

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 23,
                "observation_status": {
                    "vision_instance_id": "vision-instance",
                    "frame_publication_age_seconds": 0.01,
                    "ball_positions_status": "observed",
                },
            },
        },
    )
    connection.is_alive = False

    await handler._writer_loop(connection)

    assert socket.messages == []
    assert connection.expired_updates == 1


@pytest.mark.integration
def test_source_silence_update_survives_source_age_expiry():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_state_observation_age_seconds = 0.5
    connection = WebSocketConnection(RecordingSocket(), "stale-health")

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 19,
                "observation_status": {
                    "frame_publication_age_seconds": 10.0,
                    "ball_positions_status": "cached",
                    "freshness_status": "source_silent",
                },
            },
        },
    )

    assert len(connection.outbound_messages) == 1
    assert connection.expired_updates == 0


@pytest.mark.integration
def test_older_source_observation_is_rejected_before_serialization(monkeypatch):
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "out-of-order-source")

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 40,
                "observation_status": {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "42",
                },
            },
        },
    )

    def fail_if_serialized(_message):
        raise AssertionError("an older source observation must be dropped early")

    monkeypatch.setattr(handler_module.json, "dumps", fail_if_serialized)
    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 41,
                "observation_status": {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "41",
                },
            },
        },
    )

    assert len(connection.outbound_messages) == 1
    assert json.loads(connection.outbound_messages[0].payload)["data"]["sequence"] == 40
    assert connection.state_source_frame_number == 42
    assert connection.dropped_updates == 1
    assert handler.delivery_metrics["dropped_updates"] == 1


@pytest.mark.integration
def test_equal_source_watermark_allows_stale_health_refresh():
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "equal-source-health")

    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 50,
                "observation_status": {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "52",
                    "ball_positions_status": "observed",
                },
            },
        },
    )
    assert handler._enqueue_message(
        connection,
        {
            "type": "state",
            "data": {
                "sequence": 51,
                "observation_status": {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "52",
                    "ball_positions_status": "cached",
                    "freshness_status": "source_silent",
                },
            },
        },
    )

    assert len(connection.outbound_messages) == 1
    refreshed = json.loads(connection.outbound_messages[0].payload)
    assert refreshed["data"]["sequence"] == 51
    assert refreshed["data"]["observation_status"]["freshness_status"] == (
        "source_silent"
    )
    assert connection.state_source_frame_number == 52
    assert connection.dropped_updates == 1  # the replaced state was coalesced


@pytest.mark.integration
def test_new_source_epoch_resets_state_frame_watermark():
    handler = WebSocketHandler(manager=WebSocketManager())
    connection = WebSocketConnection(RecordingSocket(), "new-source-epoch")

    for sequence, engine, frame_id in (
        (60, "engine-old", "900"),
        (1, "engine-new", "1"),
    ):
        assert handler._enqueue_message(
            connection,
            {
                "type": "state",
                "data": {
                    "sequence": sequence,
                    "observation_status": {
                        "engine_instance_id": engine,
                        "source_session_id": "session-a",
                        "source_frame_id": frame_id,
                    },
                },
            },
        )

    latest = json.loads(connection.outbound_messages[0].payload)
    assert latest["data"]["sequence"] == 1
    assert connection.state_source_epoch == (
        ("engine_instance_id", "engine-new"),
        ("source_session_id", "session-a"),
    )
    assert connection.state_source_frame_number == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_state_socket_write_is_cancelled_at_remaining_source_age_deadline():
    handler = WebSocketHandler(manager=WebSocketManager())
    handler.max_state_observation_age_seconds = 0.2
    handler.replaceable_send_timeout_seconds = 1.0
    socket = GatedSocket()
    add_connection(handler, "expires-during-send", socket)
    connection = handler.connections["expires-during-send"]

    assert await handler.send_to_client(
        "expires-during-send",
        {
            "type": "state",
            "data": {
                "sequence": 70,
                "observation_status": {
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "70",
                    # The write starts with about 100 ms of source-age lifetime.
                    "frame_publication_age_seconds": 0.1,
                    "ball_positions_status": "observed",
                },
            },
        },
    )

    async def release_after_source_expiry():
        await socket.send_started.wait()
        await asyncio.sleep(0.15)
        socket.release_send.set()

    delayed_release = asyncio.create_task(release_after_source_expiry())
    await asyncio.wait_for(connection.closed_event.wait(), timeout=0.5)
    await delayed_release

    assert socket.messages == []
    assert connection.send_timeouts == 1
    assert handler.delivery_metrics["send_timeouts"] == 1
    assert socket.close_calls == [(1013, "Slow consumer")]


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
    handler.replaceable_send_timeout_seconds = 0.01
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
