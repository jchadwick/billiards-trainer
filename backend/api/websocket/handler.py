"""WebSocket connection handler with message processing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket, WebSocketDisconnect, status

from backend.config import config

if TYPE_CHECKING:
    from .manager import StreamType, SubscriptionFilter

logger = logging.getLogger(__name__)

_REPLACEABLE_MESSAGE_TYPES = frozenset(
    {"state", "trajectory", "metrics", "performance"}
)


@dataclass
class _OutboundMessage:
    """One serialized message waiting for a connection's sole writer."""

    payload: str
    replaceable_key: str | None = None
    enqueued_monotonic_ns: int | None = None
    created_monotonic_ns: int | None = None
    message_type: str | None = None
    stream_sequence: Any = None
    source_observation_age_seconds: float | None = None
    source_observation_age_applies: bool = False
    source_publication_monotonic_ns: int | None = None
    source_publication_tick_applies: bool = False
    audit_domain: str | None = None
    audit_sampled: bool = False


class WebSocketConnection:
    """Represents a single WebSocket connection with metadata."""

    def __init__(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize WebSocket connection with metadata.

        Args:
            websocket: FastAPI WebSocket instance.
            client_id: Unique client identifier.
        """
        self.websocket = websocket
        self.client_id = client_id
        self.metadata = dict(metadata or {})
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.subscriptions: set[str] = set()
        self.message_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.quality_score = 1.0  # 0.0 to 1.0, higher is better
        self.is_alive = True
        self.outbound_messages: deque[_OutboundMessage] = deque()
        self.replaceable_messages: dict[str, _OutboundMessage] = {}
        self.control_queue_depth = 0
        self.outbound_event = asyncio.Event()
        self.outbound_idle = asyncio.Event()
        self.outbound_idle.set()
        self.closed_event = asyncio.Event()
        self.writer_task: asyncio.Task[None] | None = None
        # A connection represents one current state stream. Keep only its
        # latest comparable source watermark so late work from the same
        # producer epoch cannot replace a newer localization.
        self.state_source_epoch: tuple[tuple[str, str], ...] | None = None
        self.state_source_frame_number: int | None = None
        self.dropped_updates = 0
        self.expired_updates = 0
        self.send_timeouts = 0
        self.control_queue_overflows = 0
        self.max_outbound_queue_depth = 0

    def update_ping(self) -> None:
        """Update last ping timestamp."""
        self.last_ping = time.time()

    def update_pong(self) -> None:
        """Update last pong timestamp and recalculate quality."""
        self.last_pong = time.time()
        # Simple quality calculation based on ping-pong latency
        latency = self.last_pong - self.last_ping
        if latency < 0.05:  # < 50ms
            self.quality_score = 1.0
        elif latency < 0.1:  # < 100ms
            self.quality_score = 0.8
        elif latency < 0.2:  # < 200ms
            self.quality_score = 0.6
        else:
            self.quality_score = 0.3

    def add_subscription(self, stream_type: str) -> None:
        """Subscribe to a data stream."""
        self.subscriptions.add(stream_type)

    def remove_subscription(self, stream_type: str) -> None:
        """Unsubscribe from a data stream."""
        self.subscriptions.discard(stream_type)

    def is_subscribed(self, stream_type: str) -> bool:
        """Check if subscribed to a stream type."""
        return stream_type in self.subscriptions

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information for monitoring."""
        return {
            "client_id": self.client_id,
            "request_id": self.metadata.get("request_id"),
            "connected_at": self.connected_at.isoformat(),
            "uptime": (datetime.now(timezone.utc) - self.connected_at).total_seconds(),
            "subscriptions": list(self.subscriptions),
            "message_count": self.message_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "quality_score": self.quality_score,
            "last_ping_latency": self.last_pong - self.last_ping,
            "is_alive": self.is_alive,
            "outbound_queue_depth": len(self.outbound_messages),
            "replaceable_queue_depth": len(self.replaceable_messages),
            "control_queue_depth": self.control_queue_depth,
            "max_outbound_queue_depth": self.max_outbound_queue_depth,
            "dropped_updates": self.dropped_updates,
            "expired_updates": self.expired_updates,
            "send_timeouts": self.send_timeouts,
            "control_queue_overflows": self.control_queue_overflows,
        }


class WebSocketHandler:
    """Advanced WebSocket connection handler with subscriptions and monitoring."""

    def __init__(self, manager: Any = None) -> None:
        """Initialize WebSocket handler with connection tracking and monitoring."""
        self.connections: dict[str, WebSocketConnection] = {}
        self.ping_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.max_message_rate = 100  # messages per minute per connection
        self.rate_limit_windows: dict[str, list[float]] = {}
        self._monitoring_task: asyncio.Task[None] | None = None
        self.replaceable_queue_capacity = self._positive_int_setting(
            "api.websocket.delivery.replaceable_queue_capacity", 32
        )
        self.control_queue_capacity = self._positive_int_setting(
            "api.websocket.delivery.control_queue_capacity", 32
        )
        self.send_timeout_seconds = self._positive_float_setting(
            "api.websocket.delivery.send_timeout_seconds", 5.0
        )
        self.max_state_observation_age_seconds = self._positive_float_setting(
            "perception.freshness.maximum_observation_age_seconds", 0.5
        )
        self.max_replaceable_message_age_seconds = self._positive_float_setting(
            "api.websocket.delivery.max_replaceable_message_age_seconds", 0.25
        )
        self.replaceable_send_timeout_seconds = self._positive_float_setting(
            "api.websocket.delivery.replaceable_send_timeout_seconds", 0.5
        )
        self.delivery_metrics = {
            "dropped_updates": 0,
            "expired_updates": 0,
            "send_timeouts": 0,
            "disconnected_slow_clients": 0,
            "control_queue_overflows": 0,
        }
        self._disconnect_tasks: dict[str, asyncio.Task[None]] = {}
        # Resolve the global manager lazily to avoid the handler/manager import cycle.
        # Injection keeps tests and embedded handlers isolated from global state.
        self._manager_override = manager
        self._audit_recorder: Any = None
        self._audit_domains: dict[str, bool] = {}
        self._audit_detail_sample_rate = self._sample_rate_setting(
            "performance_audit.detailed_event_sample_rate", 0.05
        )

    def configure_performance_audit(
        self, recorder: Any = None, domains: dict[str, bool] | None = None
    ) -> None:
        """Attach optional non-blocking transport instrumentation."""
        # The global handler can be constructed while importing the ASGI app,
        # before backend.main selects its --config file. Refresh this setting
        # at activation so a selected runtime configuration controls sampling.
        self._audit_detail_sample_rate = self._sample_rate_setting(
            "performance_audit.detailed_event_sample_rate", 0.05
        )
        self._audit_recorder = recorder
        self._audit_domains = dict(domains or {})

    def _record_delivery_event(
        self, event: dict[str, Any], *, force_keep: bool = False
    ) -> None:
        """Enqueue transport events only when an audit session is active."""
        if self._audit_recorder is not None:
            self._audit_recorder.record(event, force_keep=force_keep)

    def select_performance_audit_detail(
        self, domain: str, identity: int | str | None, *, category: str
    ) -> bool:
        """Select detail before callers allocate event fields or timing spans."""
        if not self._audit_domains.get(domain, False):
            return False
        selector = getattr(self._audit_recorder, "select_detail_sample", None)
        if not callable(selector):
            return self._audit_recorder is not None
        return bool(
            selector(
                domain,
                identity,
                self._audit_detail_sample_rate,
                category=category,
            )
        )

    def record_performance_audit(
        self, domain: str, event: dict[str, Any], *, force_keep: bool = False
    ) -> None:
        """Record an event for an independently enabled audit domain."""
        if self._audit_domains.get(domain, False):
            self._record_delivery_event(
                {"domain": domain, **event}, force_keep=force_keep
            )

    def _get_manager(self) -> Any:
        """Return the subscription manager used by this handler."""
        if self._manager_override is not None:
            return self._manager_override

        from .manager import websocket_manager

        return websocket_manager

    @staticmethod
    def _positive_int_setting(key: str, default: int) -> int:
        """Read a positive integer setting, retaining a safe fallback."""
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return default
        return int(value)

    @staticmethod
    def _positive_float_setting(key: str, default: float) -> float:
        """Read a positive numeric setting, retaining a safe fallback."""
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        value = float(value)
        return value if value > 0 and math.isfinite(value) else default

    @staticmethod
    def _sample_rate_setting(key: str, default: float) -> float:
        """Read a validated probability, with a safe fallback for test hosts."""
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        value = float(value)
        return value if 0 < value <= 1 and math.isfinite(value) else default

    def _ensure_writer(self, connection: WebSocketConnection) -> None:
        """Start the connection's supervised writer if it is not already active."""
        if connection.writer_task is None or connection.writer_task.done():
            connection.writer_task = asyncio.create_task(
                self._writer_loop(connection),
                name=f"websocket-writer:{connection.client_id}",
            )

            def observe_writer(task: asyncio.Task[None]) -> None:
                self._writer_finished(connection.client_id, task)

            connection.writer_task.add_done_callback(observe_writer)

    def _writer_finished(self, client_id: str, task: asyncio.Task[None]) -> None:
        """Observe writer failures and remove connections whose writer stopped."""
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        logger.error("WebSocket writer crashed for %s: %s", client_id, error)
        connection = self.connections.get(client_id)
        if connection is not None:
            self._claim_disconnect(connection)

    @staticmethod
    def _replaceable_key(message: dict[str, Any]) -> str | None:
        """Return the stream key for update messages that may be coalesced."""
        message_type = message.get("type")
        if (
            isinstance(message_type, str)
            and message_type.lower() in _REPLACEABLE_MESSAGE_TYPES
        ):
            return message_type.lower()
        return None

    @staticmethod
    def _observation_freshness(
        message: dict[str, Any],
    ) -> tuple[float | None, bool]:
        """Extract source age without treating cached health as a new sample.

        Source age is a duration reported by the producer/gateway and has
        already crossed the source timestamp's clock boundary. Cached and
        source-silence states are useful health updates, but their old position
        data must not be judged as a newly observed localization here.
        """
        data = message.get("data")
        if not isinstance(data, dict):
            return None, False
        observation = data.get("observation_status")
        if not isinstance(observation, dict):
            return None, False
        age = observation.get("frame_publication_age_seconds")
        age_seconds: float | None = None
        if not isinstance(age, bool) and isinstance(age, (int, float)):
            try:
                numeric_age = float(age)
            except (OverflowError, ValueError):
                numeric_age = None
            if (
                numeric_age is not None
                and math.isfinite(numeric_age)
                and numeric_age >= 0
            ):
                age_seconds = numeric_age
        cached_or_silent = observation.get(
            "freshness_status"
        ) == "source_silent" or observation.get("ball_positions_status") in {
            "cached",
            "unavailable",
        }
        return age_seconds, not cached_or_silent

    @staticmethod
    def _python_publication_tick(
        message: dict[str, Any], *, observation_age_applies: bool
    ) -> tuple[bool, int | None]:
        """Return a same-host Python publication tick when this is a live sample.

        Python vision states carry ``vision_instance_id`` and publish a
        ``perf_counter_ns`` tick from the shared-memory frame reader. The Rust
        scene adapter instead supplies an age measured at snapshot publication;
        its clock domain cannot be compared with this process and therefore
        keeps the separate local queue-age check.
        """
        if not observation_age_applies or message.get("type") != "state":
            return False, None
        data = message.get("data")
        if not isinstance(data, dict):
            return False, None
        observation = data.get("observation_status")
        if not isinstance(observation, dict):
            return False, None
        tick_is_expected = (
            "vision_instance_id" in observation
            or "source_publication_monotonic_ns" in observation
        )
        if not tick_is_expected:
            return False, None
        tick = observation.get("source_publication_monotonic_ns")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick <= 0:
            return True, None
        return True, tick

    @staticmethod
    def _state_source_order(
        message: dict[str, Any],
    ) -> tuple[tuple[tuple[str, str], ...], int] | None:
        """Extract a source epoch and integer frame watermark when comparable.

        Publication sequence numbers are deliberately excluded: a delayed
        inference result can have a newer publication sequence while referring
        to an older camera frame. Rust frame IDs may be decimal strings; other
        opaque IDs are not ordered here.
        """
        if message.get("type") != "state":
            return None
        data = message.get("data")
        if not isinstance(data, dict):
            return None
        observation = data.get("observation_status")
        if not isinstance(observation, dict):
            return None

        epoch_fields = (
            "vision_instance_id",
            "engine_instance_id",
            "source_process_id",
            "source_session_id",
        )
        epoch: list[tuple[str, str]] = []
        for field in epoch_fields:
            value = observation.get(field)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                continue
            normalized = str(value).strip()
            if normalized:
                epoch.append((field, normalized))
        if not epoch:
            identity = observation.get("observation_identity")
            if isinstance(identity, dict):
                for field in ("engine_instance_id", "session_id"):
                    value = identity.get(field)
                    if isinstance(value, str) and value.strip():
                        epoch.append((field, value.strip()))
        if not epoch:
            return None

        frame_candidates = [
            observation.get("source_frame_number"),
            data.get("source_frame_number"),
            observation.get("source_frame_id"),
        ]
        identity = observation.get("observation_identity")
        if isinstance(identity, dict):
            frame_candidates.append(identity.get("frame_id"))
        for value in frame_candidates:
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return tuple(epoch), value
            if isinstance(value, str) and value.isdecimal():
                try:
                    return tuple(epoch), int(value)
                except ValueError:
                    # Extremely long decimal IDs are opaque rather than
                    # comparable; continue to another documented identity.
                    continue
        return None

    @staticmethod
    def _refresh_serialized_state_age(payload: str, age_seconds: float) -> str:
        """Update the client-visible source age at the final delivery boundary."""
        message = json.loads(payload)
        data = message.get("data")
        observation = data.get("observation_status") if isinstance(data, dict) else None
        if not isinstance(observation, dict):
            raise ValueError("state observation status is unavailable")
        observation["frame_publication_age_seconds"] = age_seconds
        return json.dumps(message)

    def _record_expired_update(
        self,
        connection: WebSocketConnection,
        *,
        message_type: str | None,
        stream_sequence: Any,
        audit_domain: str | None,
        reason: str,
        age_seconds: float | None,
        source_observation_age_seconds: float | None,
    ) -> None:
        """Count and optionally audit an intentional stale-update drop."""
        connection.dropped_updates += 1
        connection.expired_updates += 1
        self.delivery_metrics["dropped_updates"] += 1
        self.delivery_metrics["expired_updates"] += 1
        if audit_domain is not None:
            self._record_delivery_event(
                {
                    "event_type": "websocket_update_dropped",
                    "domain": audit_domain,
                    "message_type": message_type,
                    "client_id": connection.client_id,
                    "dropped_stream_sequence": stream_sequence,
                    "reason": reason,
                    "queue_age_seconds": age_seconds,
                    "source_observation_age_seconds": source_observation_age_seconds,
                },
                force_keep=True,
            )

    def _error_message(self, client_id: str, code: str, message: str) -> dict[str, Any]:
        """Build a safe protocol error correlated with its connection request."""
        data: dict[str, Any] = {"code": code, "message": message}
        connection = self.connections.get(client_id)
        request_id = connection.metadata.get("request_id") if connection else None
        if isinstance(request_id, str) and request_id:
            data["request_id"] = request_id
        return {"type": "error", "data": data}

    def _enqueue_message(
        self, connection: WebSocketConnection, message: dict[str, Any]
    ) -> bool:
        """Add a message without waiting for socket I/O; return false on overflow."""
        replaceable_key = self._replaceable_key(message)
        created_monotonic_ns = time.monotonic_ns() if replaceable_key else None
        message_copy = dict(message)
        message_copy.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        message_copy.setdefault("sequence", connection.message_count)
        message_type = message_copy.get("type")
        message_data = message_copy.get("data")
        data_sequence = (
            message_data.get("sequence") if isinstance(message_data, dict) else None
        )
        stream_sequence = (
            data_sequence if data_sequence is not None else message_copy.get("sequence")
        )
        if message_type == "state" and self._audit_domains.get("state", False):
            audit_domain = "state"
        elif message_type == "trajectory" and self._audit_domains.get(
            "trajectory", False
        ):
            audit_domain = "trajectory"
        else:
            audit_domain = None
        source_observation_age_seconds, source_observation_age_applies = (
            self._observation_freshness(message_copy)
            if replaceable_key is not None
            else (None, False)
        )
        source_publication_tick_applies, source_publication_monotonic_ns = (
            self._python_publication_tick(
                message_copy,
                observation_age_applies=(
                    replaceable_key == "state" and source_observation_age_applies
                ),
            )
        )
        source_order = (
            self._state_source_order(message_copy)
            if replaceable_key == "state"
            else None
        )
        if source_order is not None:
            source_epoch, source_frame_number = source_order
            if (
                source_epoch == connection.state_source_epoch
                and connection.state_source_frame_number is not None
                and source_frame_number < connection.state_source_frame_number
            ):
                connection.dropped_updates += 1
                self.delivery_metrics["dropped_updates"] += 1
                if audit_domain is not None:
                    self._record_delivery_event(
                        {
                            "event_type": "websocket_state_out_of_order_dropped",
                            "domain": audit_domain,
                            "message_type": message_type,
                            "client_id": connection.client_id,
                            "dropped_stream_sequence": stream_sequence,
                            "source_frame_number": source_frame_number,
                            "latest_source_frame_number": (
                                connection.state_source_frame_number
                            ),
                            "producer_epoch": source_epoch,
                            "reason": "older_source_observation",
                        },
                        force_keep=True,
                    )
                return True
        if (
            replaceable_key == "state"
            and source_observation_age_applies
            and source_observation_age_seconds is not None
            and source_observation_age_seconds >= self.max_state_observation_age_seconds
        ):
            self._record_expired_update(
                connection,
                message_type=(message_type if isinstance(message_type, str) else None),
                stream_sequence=stream_sequence,
                audit_domain=audit_domain,
                reason="state_observation_age_exceeded",
                age_seconds=0.0,
                source_observation_age_seconds=source_observation_age_seconds,
            )
            return True
        audit_identity: int | str | None = None
        if isinstance(message_data, dict):
            observation_status = message_data.get("observation_status")
            candidate_identity = message_data.get("source_frame_number")
            if candidate_identity is None and isinstance(observation_status, dict):
                candidate_identity = observation_status.get("source_frame_number")
            if candidate_identity is None:
                candidate_identity = message_data.get("sequence")
            if isinstance(candidate_identity, (int, str)) and not isinstance(
                candidate_identity, bool
            ):
                audit_identity = candidate_identity
        audit_sampled = (
            self.select_performance_audit_detail(
                audit_domain, audit_identity, category="websocket_message"
            )
            if audit_domain is not None
            else False
        )
        serialize_started_ns = time.perf_counter_ns() if audit_sampled else None
        serialized = json.dumps(message_copy)
        serialized_ns = time.perf_counter_ns() if audit_sampled else None
        outbound = _OutboundMessage(
            payload=serialized,
            replaceable_key=replaceable_key,
            enqueued_monotonic_ns=serialized_ns,
            created_monotonic_ns=created_monotonic_ns,
            message_type=message_type if isinstance(message_type, str) else None,
            stream_sequence=stream_sequence,
            source_observation_age_seconds=source_observation_age_seconds,
            source_observation_age_applies=source_observation_age_applies,
            source_publication_monotonic_ns=source_publication_monotonic_ns,
            source_publication_tick_applies=source_publication_tick_applies,
            audit_domain=audit_domain,
            audit_sampled=audit_sampled,
        )
        if audit_sampled and audit_domain is not None:
            assert serialize_started_ns is not None
            assert serialized_ns is not None
            self._record_delivery_event(
                {
                    "event_type": "websocket_message_serialized",
                    "domain": audit_domain,
                    "message_type": message_type,
                    "stream_sequence": stream_sequence,
                    "client_id": connection.client_id,
                    "serialization_seconds": max(
                        0, serialized_ns - serialize_started_ns
                    )
                    / 1e9,
                    "queue_depth_before_enqueue": len(connection.outbound_messages),
                    "payload_bytes": len(serialized.encode()),
                }
            )

        if replaceable_key is not None:
            existing = connection.replaceable_messages.get(replaceable_key)
            if existing is not None:
                # Keep its position relative to ordered control messages while
                # replacing stale data with the freshest update for this stream.
                if existing.audit_domain is not None:
                    self._record_delivery_event(
                        {
                            "event_type": "websocket_update_coalesced",
                            "domain": existing.audit_domain,
                            "message_type": message_type,
                            "client_id": connection.client_id,
                            "dropped_stream_sequence": existing.stream_sequence,
                            "replacement_stream_sequence": stream_sequence,
                            "queue_depth": len(connection.outbound_messages),
                        },
                        force_keep=True,
                    )
                existing.payload = outbound.payload
                existing.enqueued_monotonic_ns = outbound.enqueued_monotonic_ns
                existing.created_monotonic_ns = outbound.created_monotonic_ns
                existing.stream_sequence = outbound.stream_sequence
                existing.source_observation_age_seconds = (
                    outbound.source_observation_age_seconds
                )
                existing.source_observation_age_applies = (
                    outbound.source_observation_age_applies
                )
                existing.source_publication_monotonic_ns = (
                    outbound.source_publication_monotonic_ns
                )
                existing.source_publication_tick_applies = (
                    outbound.source_publication_tick_applies
                )
                existing.message_type = outbound.message_type
                existing.audit_domain = outbound.audit_domain
                existing.audit_sampled = outbound.audit_sampled
                if source_order is not None:
                    # Equal watermarks are accepted intentionally: a health
                    # or stale refresh may describe this same observation.
                    (
                        connection.state_source_epoch,
                        connection.state_source_frame_number,
                    ) = source_order
                connection.dropped_updates += 1
                self.delivery_metrics["dropped_updates"] += 1
                return True

            if len(connection.replaceable_messages) >= self.replaceable_queue_capacity:
                oldest = next(
                    (
                        item
                        for item in connection.outbound_messages
                        if item.replaceable_key
                    ),
                    None,
                )
                if oldest is None:
                    logger.error(
                        "Replaceable queue accounting failed for %s",
                        connection.client_id,
                    )
                    return False
                connection.outbound_messages.remove(oldest)
                if oldest.audit_domain is not None:
                    self._record_delivery_event(
                        {
                            "event_type": "websocket_update_dropped",
                            "domain": oldest.audit_domain,
                            "message_type": oldest.message_type,
                            "client_id": connection.client_id,
                            "dropped_stream_sequence": oldest.stream_sequence,
                            "replacement_stream_sequence": stream_sequence,
                            "reason": "replaceable_queue_capacity",
                            "queue_depth": len(connection.outbound_messages),
                        },
                        force_keep=True,
                    )
                oldest_key = oldest.replaceable_key
                if oldest_key is not None:
                    connection.replaceable_messages.pop(oldest_key, None)
                connection.dropped_updates += 1
                self.delivery_metrics["dropped_updates"] += 1

            connection.replaceable_messages[replaceable_key] = outbound
        else:
            if connection.control_queue_depth >= self.control_queue_capacity:
                connection.control_queue_overflows += 1
                self.delivery_metrics["control_queue_overflows"] += 1
                if audit_domain is not None:
                    self._record_delivery_event(
                        {
                            "event_type": "websocket_control_queue_overflow",
                            "domain": audit_domain,
                            "message_type": message_type,
                            "client_id": connection.client_id,
                            "stream_sequence": stream_sequence,
                            "queue_depth": connection.control_queue_depth,
                            "queue_capacity": self.control_queue_capacity,
                        },
                        force_keep=True,
                    )
                return False
            connection.control_queue_depth += 1

        connection.outbound_messages.append(outbound)
        if source_order is not None:
            connection.state_source_epoch, connection.state_source_frame_number = (
                source_order
            )
        connection.max_outbound_queue_depth = max(
            connection.max_outbound_queue_depth, len(connection.outbound_messages)
        )
        connection.outbound_idle.clear()
        connection.outbound_event.set()
        return True

    async def _writer_loop(self, connection: WebSocketConnection) -> None:
        """Write queued messages in order, isolating socket delays per client."""
        try:
            while connection.is_alive or connection.outbound_messages:
                if not connection.outbound_messages:
                    connection.outbound_idle.set()
                    connection.outbound_event.clear()
                    await connection.outbound_event.wait()
                    continue

                outbound = connection.outbound_messages.popleft()
                if outbound.replaceable_key is None:
                    connection.control_queue_depth -= 1
                else:
                    connection.replaceable_messages.pop(outbound.replaceable_key, None)

                delivery_payload = outbound.payload
                replaceable_age_remaining = math.inf
                queue_age_seconds = 0.0
                if outbound.replaceable_key is not None:
                    now_monotonic_ns = time.monotonic_ns()
                    queue_age_seconds = (
                        max(
                            0,
                            now_monotonic_ns
                            - (
                                outbound.created_monotonic_ns
                                if outbound.created_monotonic_ns is not None
                                else now_monotonic_ns
                            ),
                        )
                        / 1e9
                    )
                    if queue_age_seconds >= self.max_replaceable_message_age_seconds:
                        self._record_expired_update(
                            connection,
                            message_type=outbound.message_type,
                            stream_sequence=outbound.stream_sequence,
                            audit_domain=outbound.audit_domain,
                            reason="replaceable_queue_age_exceeded",
                            age_seconds=queue_age_seconds,
                            source_observation_age_seconds=(
                                outbound.source_observation_age_seconds
                            ),
                        )
                        continue
                    replaceable_age_remaining = (
                        self.max_replaceable_message_age_seconds - queue_age_seconds
                    )

                source_age_remaining = math.inf
                if (
                    outbound.replaceable_key == "state"
                    and outbound.source_observation_age_applies
                ):
                    if outbound.source_publication_tick_applies:
                        now_publication_clock_ns = time.perf_counter_ns()
                        source_tick = outbound.source_publication_monotonic_ns
                        if (
                            source_tick is None
                            or source_tick <= 0
                            or now_publication_clock_ns < source_tick
                        ):
                            self._record_expired_update(
                                connection,
                                message_type=outbound.message_type,
                                stream_sequence=outbound.stream_sequence,
                                audit_domain=outbound.audit_domain,
                                reason="state_source_publication_timestamp_unavailable",
                                age_seconds=queue_age_seconds,
                                source_observation_age_seconds=None,
                            )
                            continue

                        source_age_seconds = (
                            now_publication_clock_ns - source_tick
                        ) / 1e9
                        if (
                            not math.isfinite(source_age_seconds)
                            or source_age_seconds
                            >= self.max_state_observation_age_seconds
                        ):
                            self._record_expired_update(
                                connection,
                                message_type=outbound.message_type,
                                stream_sequence=outbound.stream_sequence,
                                audit_domain=outbound.audit_domain,
                                reason="state_observation_age_exceeded",
                                age_seconds=queue_age_seconds,
                                source_observation_age_seconds=source_age_seconds,
                            )
                            continue

                        source_age_remaining = (
                            self.max_state_observation_age_seconds - source_age_seconds
                        )
                        try:
                            delivery_payload = self._refresh_serialized_state_age(
                                outbound.payload, source_age_seconds
                            )
                        except (TypeError, ValueError, json.JSONDecodeError):
                            self._record_expired_update(
                                connection,
                                message_type=outbound.message_type,
                                stream_sequence=outbound.stream_sequence,
                                audit_domain=outbound.audit_domain,
                                reason="state_source_publication_timestamp_unavailable",
                                age_seconds=queue_age_seconds,
                                source_observation_age_seconds=source_age_seconds,
                            )
                            continue
                        outbound.source_observation_age_seconds = source_age_seconds
                    elif outbound.source_observation_age_seconds is not None:
                        # Rust's source age uses a separate clock domain. Add
                        # only elapsed queue residence measured on this host.
                        source_age_seconds = (
                            outbound.source_observation_age_seconds + queue_age_seconds
                        )
                        source_age_remaining = (
                            self.max_state_observation_age_seconds - source_age_seconds
                        )
                        if source_age_remaining <= 0:
                            self._record_expired_update(
                                connection,
                                message_type=outbound.message_type,
                                stream_sequence=outbound.stream_sequence,
                                audit_domain=outbound.audit_domain,
                                reason="state_observation_age_exceeded",
                                age_seconds=queue_age_seconds,
                                source_observation_age_seconds=source_age_seconds,
                            )
                            continue

                # A write timeout is bounded by both the configured slow-client
                # limit and the remaining freshness lifetime. This limits how
                # long the app waits for ASGI; it cannot retract bytes already
                # accepted by the server, kernel, TCP buffers, or a remote peer.
                write_timeout = (
                    self.replaceable_send_timeout_seconds
                    if outbound.replaceable_key is not None
                    else self.send_timeout_seconds
                )
                write_timeout = min(write_timeout, replaceable_age_remaining)
                if outbound.replaceable_key == "state":
                    write_timeout = min(write_timeout, source_age_remaining)
                if write_timeout <= 0:
                    self._record_expired_update(
                        connection,
                        message_type=outbound.message_type,
                        stream_sequence=outbound.stream_sequence,
                        audit_domain=outbound.audit_domain,
                        reason="state_observation_age_exceeded",
                        age_seconds=queue_age_seconds,
                        source_observation_age_seconds=(
                            outbound.source_observation_age_seconds
                        ),
                    )
                    continue

                send_started_ns = (
                    time.perf_counter_ns() if outbound.audit_sampled else None
                )
                try:
                    await asyncio.wait_for(
                        connection.websocket.send_text(delivery_payload),
                        timeout=write_timeout,
                    )
                    payload_bytes = len(delivery_payload.encode())
                    connection.bytes_sent += payload_bytes
                    if outbound.audit_sampled and outbound.audit_domain is not None:
                        assert send_started_ns is not None
                        self._record_delivery_event(
                            {
                                "event_type": "websocket_delivery_complete",
                                "domain": outbound.audit_domain,
                                "message_type": outbound.message_type,
                                "stream_sequence": outbound.stream_sequence,
                                "client_id": connection.client_id,
                                "queue_wait_seconds": max(
                                    0,
                                    send_started_ns
                                    - (
                                        outbound.enqueued_monotonic_ns
                                        if outbound.enqueued_monotonic_ns is not None
                                        else send_started_ns
                                    ),
                                )
                                / 1e9,
                                "socket_send_seconds": max(
                                    0, time.perf_counter_ns() - send_started_ns
                                )
                                / 1e9,
                                "payload_bytes": payload_bytes,
                            }
                        )
                except asyncio.TimeoutError:
                    if outbound.audit_domain is not None:
                        self._record_delivery_event(
                            {
                                "event_type": "websocket_send_timeout",
                                "domain": outbound.audit_domain,
                                "message_type": outbound.message_type,
                                "stream_sequence": outbound.stream_sequence,
                                "client_id": connection.client_id,
                                "queue_depth": len(connection.outbound_messages),
                            },
                            force_keep=True,
                        )
                    connection.send_timeouts += 1
                    self.delivery_metrics["send_timeouts"] += 1
                    logger.warning(
                        "WebSocket send timed out for slow client %s",
                        connection.client_id,
                    )
                    self._schedule_slow_client_disconnect(connection)
                    return
                except WebSocketDisconnect:
                    logger.debug(
                        "Client %s disconnected during send", connection.client_id
                    )
                    self._claim_disconnect(connection)
                    return
                except Exception as error:
                    logger.debug(
                        "Failed to send message to %s: %s", connection.client_id, error
                    )
                    self._claim_disconnect(connection)
                    return
                finally:
                    if not connection.outbound_messages:
                        connection.outbound_idle.set()
        finally:
            connection.outbound_idle.set()

    def _claim_disconnect(
        self, connection: WebSocketConnection, *, close_code: int | None = None
    ) -> asyncio.Task[None] | None:
        """Remove a connection synchronously and schedule exactly one cleanup."""
        client_id = connection.client_id
        existing_task = self._disconnect_tasks.get(client_id)
        if existing_task is not None:
            return existing_task
        if self.connections.get(client_id) is not connection:
            return None

        # Claim ownership before any await so concurrent disconnect callers can
        # only observe this cleanup task, never the same live connection twice.
        self.connections.pop(client_id)
        connection.is_alive = False
        connection.outbound_messages.clear()
        connection.replaceable_messages.clear()
        connection.control_queue_depth = 0
        connection.outbound_event.set()
        connection.outbound_idle.set()
        task = asyncio.create_task(
            self._finish_disconnect(connection, close_code=close_code),
            name=f"websocket-cleanup:{client_id}",
        )
        self._disconnect_tasks[client_id] = task

        def observe_disconnect(task: asyncio.Task[None]) -> None:
            self._observe_disconnect_task(client_id, task)

        task.add_done_callback(observe_disconnect)
        return task

    def _observe_disconnect_task(
        self, client_id: str, task: asyncio.Task[None]
    ) -> None:
        """Retrieve cleanup failures when the initiating caller did not await."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "WebSocket cleanup failed for %s: %s",
                client_id,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _schedule_slow_client_disconnect(self, connection: WebSocketConnection) -> None:
        """Queue slow-client close and teardown outside the broadcasting task."""
        self.delivery_metrics["disconnected_slow_clients"] += 1
        self._claim_disconnect(connection, close_code=1013)

    async def _finish_disconnect(
        self, connection: WebSocketConnection, *, close_code: int | None
    ) -> None:
        """Complete connection teardown after its state has been claimed."""
        client_id = connection.client_id
        try:
            writer_task = connection.writer_task
            if writer_task is not None and writer_task is not asyncio.current_task():
                writer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await writer_task

            if close_code is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        connection.websocket.close(
                            code=close_code, reason="Slow consumer"
                        ),
                        timeout=self.send_timeout_seconds,
                    )

            self.rate_limit_windows.pop(client_id, None)

            # Remove manager-owned subscriptions even when a session was already
            # lost during recovery.
            manager = self._get_manager()
            for stream_type in manager.stream_subscribers:
                manager.stream_subscribers[stream_type].discard(client_id)
            await manager.unregister_client(client_id)

            logger.info("WebSocket disconnected: %s", client_id)
            if not self.connections:
                await self.stop_monitoring()
        finally:
            connection.closed_event.set()
            if self._disconnect_tasks.get(client_id) is asyncio.current_task():
                self._disconnect_tasks.pop(client_id, None)

    async def wait_for_outbound(self, client_id: str) -> None:
        """Wait until currently queued messages have been written; useful for drains."""
        connection = self.connections.get(client_id)
        if connection is not None:
            await connection.outbound_idle.wait()

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_connections())

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

            self._monitoring_task = None

    async def connect(
        self,
        websocket: WebSocket,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Accept new WebSocket connection."""
        client_id: str | None = None
        try:
            # Accept the WebSocket connection
            await websocket.accept()

            # Generate unique client ID
            client_id = str(uuid.uuid4())

            # The manager owns stream delivery. Register first so a connection is
            # never exposed as ready before its subscription session exists.
            await self._get_manager().register_client(client_id, metadata=metadata)

            # Create connection object
            connection = WebSocketConnection(websocket, client_id, metadata=metadata)

            # Store the connection
            self.connections[client_id] = connection
            self._ensure_writer(connection)

            # Initialize rate limiting
            self.rate_limit_windows[client_id] = []

            logger.info("WebSocket connected: %s", client_id)

            # Start monitoring if not already running
            await self.start_monitoring()

            # Send welcome message
            await self.send_to_client(
                client_id,
                {
                    "type": "connection",
                    "data": {
                        "client_id": client_id,
                        "status": "connected",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                },
            )

            return client_id

        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            if client_id is not None:
                with contextlib.suppress(Exception):
                    await self.disconnect(client_id)
            # Don't raise HTTPException in WebSocket context - just close the socket
            try:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            except Exception:
                pass
            # Re-raise the original exception for proper error handling
            raise

    async def disconnect(self, client_id: str) -> None:
        """Handle WebSocket disconnection."""
        cleanup_task = self._disconnect_tasks.get(client_id)
        if cleanup_task is None:
            connection = self.connections.get(client_id)
            if connection is None:
                return
            cleanup_task = self._claim_disconnect(connection)
        if cleanup_task is None:
            return
        if cleanup_task is asyncio.current_task():
            return
        await asyncio.shield(cleanup_task)

    async def shutdown(self) -> None:
        """Stop monitoring and drain active connections plus pending cleanup."""
        await self.stop_monitoring()

        # Slow-client cleanup removes the connection synchronously, before its
        # close handshake completes. Drain the cleanup registry as well as any
        # connections that were still active when shutdown began.
        while self.connections or self._disconnect_tasks:
            for connection in tuple(self.connections.values()):
                self._claim_disconnect(connection)

            cleanup_tasks = tuple(self._disconnect_tasks.values())
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def handle_message(self, client_id: str, message: str) -> None:
        """Handle incoming WebSocket message from client."""
        if client_id not in self.connections:
            return

        connection = self.connections[client_id]

        # Rate limiting check
        if not self._check_rate_limit(client_id):
            await self.send_to_client(
                client_id,
                self._error_message(
                    client_id, "RATE_LIMIT", "Message rate limit exceeded"
                ),
            )
            return

        try:
            # Parse message
            data = json.loads(message)
            connection.bytes_received += len(message.encode())
            connection.message_count += 1

            # Handle different message types
            message_type = data.get("type")

            if message_type == "ping":
                await self._handle_ping(client_id, data)
            elif message_type == "subscribe":
                await self._handle_subscribe(client_id, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(client_id, data)
            elif message_type == "get_status":
                await self._handle_get_status(client_id, data)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                await self.send_to_client(
                    client_id,
                    self._error_message(
                        client_id,
                        "UNKNOWN_MESSAGE_TYPE",
                        f"Unknown message type: {message_type}",
                    ),
                )

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {client_id}: {message}")
            await self.send_to_client(
                client_id,
                self._error_message(client_id, "INVALID_JSON", "Invalid JSON format"),
            )
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(
                client_id,
                self._error_message(
                    client_id,
                    "MESSAGE_HANDLER_ERROR",
                    "Internal message handling error",
                ),
            )

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> bool:
        """Queue a message for a client without waiting for its socket to drain."""
        if client_id not in self.connections:
            return False

        connection = self.connections[client_id]
        if not connection.is_alive:
            return False

        self._ensure_writer(connection)
        try:
            queued = self._enqueue_message(connection, message)
        except (TypeError, ValueError) as error:
            logger.error(
                "Unable to serialize outbound message for %s: %s", client_id, error
            )
            await self.disconnect(client_id)
            return False

        if queued:
            return True

        # A full control queue means ordered events can no longer be preserved.
        # Disconnect promptly rather than silently dropping an acknowledgement,
        # alert, error, or configuration notification.
        logger.warning("Control queue is full for slow client %s", client_id)
        self._schedule_slow_client_disconnect(connection)
        return False

    async def broadcast_to_subscribers(
        self, stream_type: str, message: dict[str, Any]
    ) -> None:
        """Broadcast message to all clients subscribed to a stream type."""
        if not self.connections:
            return

        subscribers = [
            client_id
            for client_id, conn in self.connections.items()
            if conn.is_subscribed(stream_type) and conn.is_alive
        ]

        if not subscribers:
            return

        # Send to all subscribers concurrently
        tasks = [self.send_to_client(client_id, message) for client_id in subscribers]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Clean up failed clients
        failed_clients = []
        for i, result in enumerate(results):
            if not result or isinstance(result, Exception):
                client_id = subscribers[i]
                failed_clients.append(client_id)
                # Remove subscription for failed client
                if client_id in self.connections:
                    await self._unsubscribe_stream(client_id, stream_type)
                    logger.debug(
                        f"Removed {stream_type} subscription for client {client_id} due to send failure"
                    )

        if failed_clients:
            logger.debug(
                f"Cleaned up {len(failed_clients)}/{len(subscribers)} failed subscribers for {stream_type}"
            )

    async def broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self.connections:
            return

        # Send to all connected clients concurrently
        tasks = [
            self.send_to_client(client_id, message) for client_id in self.connections
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        failed_count = sum(
            1 for result in results if not result or isinstance(result, Exception)
        )
        if failed_count > 0:
            logger.warning(
                f"Failed to broadcast to {failed_count}/{len(tasks)} clients"
            )

    def get_connection_stats(self) -> dict[str, Any]:
        """Get overall connection statistics."""
        if not self.connections:
            return {
                "total_connections": 0,
                "average_quality": 0.0,
                "total_messages": 0,
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "delivery": {
                    **self.delivery_metrics,
                    "outbound_queue_depth": 0,
                    "replaceable_queue_depth": 0,
                    "control_queue_depth": 0,
                },
            }

        avg_quality = sum(
            conn.quality_score for conn in self.connections.values()
        ) / len(self.connections)
        total_messages = sum(conn.message_count for conn in self.connections.values())
        total_bytes_sent = sum(conn.bytes_sent for conn in self.connections.values())
        total_bytes_received = sum(
            conn.bytes_received for conn in self.connections.values()
        )

        return {
            "total_connections": len(self.connections),
            "average_quality": round(avg_quality, 3),
            "total_messages": total_messages,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "connections": [
                conn.get_connection_info() for conn in self.connections.values()
            ],
            "delivery": {
                **self.delivery_metrics,
                "outbound_queue_depth": sum(
                    len(conn.outbound_messages) for conn in self.connections.values()
                ),
                "replaceable_queue_depth": sum(
                    len(conn.replaceable_messages) for conn in self.connections.values()
                ),
                "control_queue_depth": sum(
                    conn.control_queue_depth for conn in self.connections.values()
                ),
            },
        }

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        window = self.rate_limit_windows.get(client_id, [])

        # Remove old entries (older than 1 minute)
        window = [timestamp for timestamp in window if now - timestamp < 60]

        # Check if under limit
        if len(window) >= self.max_message_rate:
            return False

        # Add current timestamp
        window.append(now)
        self.rate_limit_windows[client_id] = window

        return True

    async def _handle_ping(self, client_id: str, data: dict[str, Any]) -> None:
        """Handle ping message."""
        connection = self.connections[client_id]
        connection.update_ping()

        await self.send_to_client(
            client_id,
            {
                "type": "pong",
                "data": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "quality_score": connection.quality_score,
                },
            },
        )

        connection.update_pong()

    async def _handle_subscribe(self, client_id: str, data: dict[str, Any]) -> None:
        """Handle subscription request."""
        stream_types = data.get("data", {}).get("streams", [])

        if not isinstance(stream_types, list):
            await self.send_to_client(
                client_id,
                self._error_message(
                    client_id,
                    "INVALID_SUBSCRIBE_FORMAT",
                    "streams must be an array",
                ),
            )
            return

        connection = self.connections[client_id]
        manager = self._get_manager()
        from .manager import StreamType, SubscriptionFilter

        added_streams = []
        for requested_stream in stream_types:
            if not isinstance(requested_stream, str):
                logger.warning(
                    "Invalid stream type requested by %s: %r",
                    client_id,
                    requested_stream,
                )
                continue

            try:
                stream_type = StreamType(requested_stream.lower())
            except ValueError:
                logger.warning(
                    "Invalid stream type requested by %s: %s",
                    client_id,
                    requested_stream,
                )
                continue

            filter_config = self._build_subscription_filter(
                data.get("data", {}).get("filters"), stream_type, SubscriptionFilter
            )
            registered = await manager.subscribe_to_stream(
                client_id, stream_type, filter_config
            )
            if registered:
                connection.add_subscription(stream_type.value)
                added_streams.append(stream_type.value)

        await self.send_to_client(
            client_id,
            {
                "type": "subscribed",
                "data": {
                    "streams": added_streams,
                    "all_subscriptions": list(connection.subscriptions),
                },
            },
        )

    async def _handle_unsubscribe(self, client_id: str, data: dict[str, Any]) -> None:
        """Handle unsubscription request."""
        stream_types = data.get("data", {}).get("streams", [])

        if not isinstance(stream_types, list):
            await self.send_to_client(
                client_id,
                self._error_message(
                    client_id,
                    "INVALID_UNSUBSCRIBE_FORMAT",
                    "streams must be an array",
                ),
            )
            return

        connection = self.connections[client_id]
        from .manager import StreamType

        removed_streams = []
        for requested_stream in stream_types:
            if not isinstance(requested_stream, str):
                continue
            try:
                stream_type = StreamType(requested_stream.lower())
            except ValueError:
                continue

            if await self._unsubscribe_stream(client_id, stream_type):
                removed_streams.append(stream_type.value)

        await self.send_to_client(
            client_id,
            {
                "type": "unsubscribed",
                "data": {
                    "streams": removed_streams,
                    "all_subscriptions": list(connection.subscriptions),
                },
            },
        )

    def _build_subscription_filter(
        self,
        filters: Any,
        stream_type: StreamType,
        filter_type: type[SubscriptionFilter],
    ) -> SubscriptionFilter | None:
        """Build a manager filter from the optional protocol filter object."""
        if not isinstance(filters, dict):
            return None

        # Accept either a shared filter object or filters keyed by stream name.
        stream_filters = filters.get(stream_type.value, filters)
        if not isinstance(stream_filters, dict):
            return None

        supported_fields = {
            "min_fps",
            "max_fps",
            "quality_level",
            "include_fields",
            "exclude_fields",
            "conditions",
        }
        if not any(key in stream_filters for key in supported_fields):
            return None

        return filter_type(
            stream_type=stream_type,
            min_fps=stream_filters.get("min_fps"),
            max_fps=stream_filters.get("max_fps"),
            quality_level=stream_filters.get("quality_level", "auto"),
            include_fields=stream_filters.get("include_fields"),
            exclude_fields=stream_filters.get("exclude_fields"),
            conditions=stream_filters.get("conditions", {}),
        )

    async def _unsubscribe_stream(
        self, client_id: str, stream_type: str | StreamType
    ) -> bool:
        """Remove a stream from manager delivery and local connection state."""
        from .manager import StreamType as RuntimeStreamType

        if not isinstance(stream_type, RuntimeStreamType):
            try:
                stream_type = RuntimeStreamType(stream_type.lower())
            except ValueError:
                return False

        connection = self.connections.get(client_id)
        manager = self._get_manager()

        if connection is None:
            return False

        was_subscribed = connection.is_subscribed(stream_type.value)
        session_exists = client_id in manager.sessions
        manager_subscribed = client_id in manager.stream_subscribers[stream_type]
        if session_exists:
            manager_subscribed = manager_subscribed or (
                stream_type in manager.sessions[client_id].subscription_filters
            )
        if not was_subscribed and not manager_subscribed:
            return False

        removed = False
        if session_exists:
            removed = await manager.unsubscribe_from_stream(client_id, stream_type)
        else:
            manager.stream_subscribers[stream_type].discard(client_id)

        connection.remove_subscription(stream_type.value)
        # A stale local subscription still counts as successfully cleaned up.
        return removed or was_subscribed

    async def _handle_get_status(self, client_id: str, data: dict[str, Any]) -> None:
        """Handle status request."""
        connection = self.connections[client_id]

        await self.send_to_client(
            client_id, {"type": "status", "data": connection.get_connection_info()}
        )

    async def _monitor_connections(self) -> None:
        """Background task to monitor connection health."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)

                if not self.connections:
                    continue

                current_time = time.time()
                disconnected_clients = []

                # Check for stale connections
                for client_id, connection in self.connections.items():
                    time_since_pong = current_time - connection.last_pong

                    if time_since_pong > self.connection_timeout:
                        logger.warning(
                            f"Connection {client_id} timed out (no pong for {time_since_pong:.1f}s)"
                        )
                        disconnected_clients.append(client_id)
                    elif time_since_pong > self.ping_interval:
                        # Send ping to check if connection is alive
                        await self.send_to_client(
                            client_id,
                            {
                                "type": "ping",
                                "data": {
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                },
                            },
                        )

                # Disconnect stale connections
                for client_id in disconnected_clients:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")


# Global handler instance
websocket_handler = WebSocketHandler()
