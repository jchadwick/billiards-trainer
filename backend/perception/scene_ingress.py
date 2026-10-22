"""Unauthenticated, bounded loopback ingress for Rust scene snapshots.

The ingress is deliberately passive until :meth:`SceneSnapshotIngress.start`
is called. A recorded producer uses the v2 identity-scoped handshake; a live
producer is required to use v3 and present the request-bound capture-session
acknowledgement before it receives ``ready``. A configured native public-state
producer may use v2 for replay or the strict v3 handshake for live capture. Both
send complete scene-snapshot v1 records (or the explicitly enabled native
delivery envelope containing one) using the shared four-byte big-endian framing
contract. Producer IDs are provenance already present in scenes; they do not
authenticate a process. The endpoint is restricted to loopback and one active
connection. A single pending slot keeps consumers on the newest state when Core
or another optional consumer falls behind.

Receipt timestamps use this Python process's monotonic clock. The module does
not subtract producer timestamps or claim cross-process one-way latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from backend.perception.capture_acknowledgement import (
    ValidatedCaptureSessionAcknowledgement,
    validate_capture_session_acknowledgement,
)
from backend.perception.capture_configuration import (
    CaptureSettingsSnapshot,
    CaptureSettingsSnapshotV1,
    CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3,
)

logger = logging.getLogger(__name__)

_SCENE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "perception/contracts/v1/scene-snapshot.schema.json"
)
_PUBLIC_STATE_DELIVERY_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "perception/contracts/public-state/public-state-delivery-v1.schema.json"
)
_CAPTURE_ACK_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "perception/contracts/capture-session-ack"
)
_PROTOCOL_VERSION = 2
_MAX_IDENTIFIER_CHARACTERS = 128
_READY_RESPONSE_BASE = {"type": "ready", "protocol_version": _PROTOCOL_VERSION}


def _reject_non_finite_json_constant(value: str) -> None:
    """Reject Python's non-standard NaN and Infinity JSON extensions."""
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


class SceneIngressError(ValueError):
    """A protocol or configuration error at the local scene boundary."""


@dataclass(frozen=True)
class SceneIngressConfig:
    """Validated runtime settings supplied by the owning application."""

    bind_address: str
    port: int
    maximum_message_bytes: int
    read_timeout_seconds: float
    capture_settings: CaptureSettingsSnapshot | None = None
    accept_public_state_delivery: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bind_address, str):
            raise ValueError("scene ingress address must be a loopback IP")
        try:
            address = ipaddress.ip_address(self.bind_address)
        except ValueError as exc:
            raise ValueError("scene ingress address must be a loopback IP") from exc
        if not address.is_loopback:
            raise ValueError("scene ingress may bind only to a loopback IP")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 0 <= self.port <= 65535
        ):
            raise ValueError("scene ingress port must be between 0 and 65535")
        if (
            not isinstance(self.maximum_message_bytes, int)
            or isinstance(self.maximum_message_bytes, bool)
            or self.maximum_message_bytes <= 0
        ):
            raise ValueError("maximum scene message size must be positive")
        if (
            not isinstance(self.read_timeout_seconds, (int, float))
            or isinstance(self.read_timeout_seconds, bool)
            or not math.isfinite(self.read_timeout_seconds)
            or self.read_timeout_seconds <= 0
        ):
            raise ValueError("scene ingress read timeout must be finite and positive")
        if self.capture_settings is not None and not isinstance(
            self.capture_settings,
            (
                CaptureSettingsSnapshotV1,
                CaptureSettingsSnapshotV2,
                CaptureSettingsSnapshotV3,
            ),
        ):
            raise ValueError(
                "capture settings must be a validated v1, v2, or v3 request snapshot"
            )
        if not isinstance(self.accept_public_state_delivery, bool):
            raise ValueError("public-state delivery selection must be boolean")
        if (
            self.accept_public_state_delivery
            and self.capture_settings is not None
            and not isinstance(self.capture_settings, CaptureSettingsSnapshotV3)
        ):
            raise ValueError(
                "Rust public-state live delivery requires capture-settings v3"
            )
        hello_size = len(
            json.dumps(
                {
                    "type": "hello",
                    "protocol_version": _PROTOCOL_VERSION,
                    "engine_instance_id": "e" * _MAX_IDENTIFIER_CHARACTERS,
                    "source_session_id": "s" * _MAX_IDENTIFIER_CHARACTERS,
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        handshake_sizes = [
            hello_size,
            len(json.dumps({**_READY_RESPONSE_BASE, "last_transport_sequence": None})),
        ]
        if self.capture_settings is not None:
            version = self.capture_settings.schema_version
            expected_version = {
                CaptureSettingsSnapshotV1: 1,
                CaptureSettingsSnapshotV2: 2,
                CaptureSettingsSnapshotV3: 3,
            }[type(self.capture_settings)]
            if type(version) is not int or version != expected_version:
                raise ValueError(
                    "capture settings schema version does not match its snapshot type"
                )
            acknowledgement_version = 1 if version == 1 else 2
            acknowledgement_fixture = json.loads(
                (
                    _CAPTURE_ACK_FIXTURE_ROOT
                    / f"v{acknowledgement_version}"
                    / "capture-session-ack-default.json"
                ).read_text(encoding="utf-8")
            )
            if not isinstance(acknowledgement_fixture, dict):
                raise ValueError("capture acknowledgement fixture must be an object")
            source_report = acknowledgement_fixture.get("source_report")
            if isinstance(source_report, dict):
                source_report["session_id"] = "s" * _MAX_IDENTIFIER_CHARACTERS
            handshake_sizes.append(
                len(
                    json.dumps(
                        {
                            "type": "hello",
                            "protocol_version": 3,
                            "engine_instance_id": "e" * _MAX_IDENTIFIER_CHARACTERS,
                            "source_session_id": "s" * _MAX_IDENTIFIER_CHARACTERS,
                            "capture_session_ack": acknowledgement_fixture,
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            )
            handshake_sizes.append(
                len(
                    json.dumps(
                        {
                            "type": "ready",
                            "protocol_version": 3,
                            "configuration_revision": self.capture_settings.configuration_revision,
                            "request_sha256": "0" * 64,
                            "last_transport_sequence": None,
                        },
                        separators=(",", ":"),
                    )
                )
            )
        if self.maximum_message_bytes < max(handshake_sizes):
            raise ValueError("maximum message size cannot carry the ingress handshake")


@dataclass(frozen=True)
class ReceivedSceneSnapshot:
    """Immutable received bytes and local receipt provenance.

    The parsed snapshot is exposed as a fresh dictionary on each access so a
    Core adapter cannot mutate the copy retained for diagnostics or another
    consumer.
    """

    payload: bytes
    received_monotonic_ns: int
    received_at: datetime
    payload_sha256: str
    peer_address: tuple[str, int]
    connection_id: str

    @property
    def snapshot(self) -> dict[str, Any]:
        value = json.loads(self.payload)
        if not isinstance(value, dict):  # protected by schema validation
            raise SceneIngressError("validated scene snapshot is not an object")
        if value.get("delivery_type") == "billiards_public_state_delivery":
            snapshot = value.get("source_snapshot")
            if not isinstance(snapshot, dict):  # protected by delivery validation
                raise SceneIngressError("delivery source snapshot is not an object")
            return snapshot
        return value

    @property
    def public_state_delivery(self) -> dict[str, Any] | None:
        """Return a validated native-state envelope when one was delivered."""
        value = json.loads(self.payload)
        if not isinstance(value, dict):
            raise SceneIngressError("validated scene frame is not an object")
        if value.get("delivery_type") != "billiards_public_state_delivery":
            return None
        return value


@dataclass(frozen=True)
class SceneIngressStats:
    """Bounded counters describing accepted transport and queue behavior."""

    accepted_snapshots: int
    queue_replacements: int
    rejected_connections: int
    rejected_handshakes: int
    rejected_messages: int
    read_timeouts: int
    sequence_gap_total: int
    producer_epoch_resets: int
    discarded_epoch_snapshots: int
    connection_count: int
    connection_active: bool
    last_received_monotonic_ns: int | None
    last_transport_sequence: int | None
    producer_identity: tuple[str, str] | None


@dataclass(frozen=True)
class SceneIngressTransportAuditSummary:
    """Opt-in bounded aggregate measurements for one qualification window."""

    status: str
    accepted_snapshots: int
    payload_bytes_total: int
    payload_bytes_minimum: int | None
    payload_bytes_mean: float | None
    payload_bytes_maximum: int | None
    sequence_gaps: int
    queue_capacity: int
    queue_depth_high_water: int
    queue_depth_zero_before_enqueue: int
    queue_depth_one_before_enqueue: int
    queue_replacements: int
    queue_replacement_fraction: float | None


class _SceneIngressTransportAudit:
    """Accumulate scalar transport evidence without retaining frame history."""

    def __init__(self, queue_capacity: int) -> None:
        self._queue_capacity = queue_capacity
        self._accepted_snapshots = 0
        self._payload_bytes_total = 0
        self._payload_bytes_minimum: int | None = None
        self._payload_bytes_maximum: int | None = None
        self._sequence_gaps = 0
        self._queue_depth_high_water = 0
        self._queue_depth_zero_before_enqueue = 0
        self._queue_depth_one_before_enqueue = 0
        self._queue_replacements = 0

    def record(
        self, *, payload_bytes: int, queue_depth_before_enqueue: int, sequence_gaps: int
    ) -> None:
        self._accepted_snapshots += 1
        self._payload_bytes_total += payload_bytes
        self._payload_bytes_minimum = (
            payload_bytes
            if self._payload_bytes_minimum is None
            else min(self._payload_bytes_minimum, payload_bytes)
        )
        self._payload_bytes_maximum = (
            payload_bytes
            if self._payload_bytes_maximum is None
            else max(self._payload_bytes_maximum, payload_bytes)
        )
        self._sequence_gaps += sequence_gaps
        if queue_depth_before_enqueue == 0:
            self._queue_depth_zero_before_enqueue += 1
        elif queue_depth_before_enqueue == 1:
            self._queue_depth_one_before_enqueue += 1
            self._queue_replacements += 1
        self._queue_depth_high_water = max(
            self._queue_depth_high_water,
            min(self._queue_capacity, queue_depth_before_enqueue + 1),
        )

    def summary(self) -> SceneIngressTransportAuditSummary:
        count = self._accepted_snapshots
        return SceneIngressTransportAuditSummary(
            status="available" if count else "unavailable",
            accepted_snapshots=count,
            payload_bytes_total=self._payload_bytes_total,
            payload_bytes_minimum=self._payload_bytes_minimum,
            payload_bytes_mean=(self._payload_bytes_total / count if count else None),
            payload_bytes_maximum=self._payload_bytes_maximum,
            sequence_gaps=self._sequence_gaps,
            queue_capacity=self._queue_capacity,
            queue_depth_high_water=self._queue_depth_high_water,
            queue_depth_zero_before_enqueue=self._queue_depth_zero_before_enqueue,
            queue_depth_one_before_enqueue=self._queue_depth_one_before_enqueue,
            queue_replacements=self._queue_replacements,
            queue_replacement_fraction=(
                self._queue_replacements / count if count else None
            ),
        )


class SceneSnapshotIngress:
    """Receive one loopback scene producer with latest-only buffering."""

    def __init__(self, settings: SceneIngressConfig) -> None:
        self._settings = settings
        schema = json.loads(_SCENE_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        self._validator = Draft202012Validator(schema)
        self._public_state_validator: Draft202012Validator | None = None
        if settings.accept_public_state_delivery:
            delivery_schema = json.loads(
                _PUBLIC_STATE_DELIVERY_SCHEMA_PATH.read_text(encoding="utf-8")
            )
            Draft202012Validator.check_schema(delivery_schema)
            self._public_state_validator = Draft202012Validator(delivery_schema)

        self._server: asyncio.AbstractServer | None = None
        self._bound_address: tuple[str, int] | None = None
        self._latest: asyncio.Queue[ReceivedSceneSnapshot] = asyncio.Queue(maxsize=1)
        self._transport_audit: _SceneIngressTransportAudit | None = None
        self._active_writer: asyncio.StreamWriter | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._identity: tuple[str, str] | None = None
        self._accepted_identity: tuple[str, str] | None = None
        self._capture_session_acknowledgement: (
            ValidatedCaptureSessionAcknowledgement | None
        ) = None
        self._last_sequence: int | None = None
        self._last_snapshot_revision: int | None = None
        self._last_observation: tuple[str, str] | None = None
        self._last_observation_timestamp: tuple[str, str, int] | None = None
        self._accepted_snapshots = 0
        self._queue_replacements = 0
        self._rejected_connections = 0
        self._rejected_handshakes = 0
        self._rejected_messages = 0
        self._read_timeouts = 0
        self._sequence_gap_total = 0
        self._producer_epoch_resets = 0
        self._discarded_epoch_snapshots = 0
        self._connection_count = 0
        self._last_received_monotonic_ns: int | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Return the bound loopback address after startup."""
        if self._bound_address is None:
            raise RuntimeError("scene ingress has not been started")
        return self._bound_address

    @property
    def stats(self) -> SceneIngressStats:
        """Return a stable snapshot of counters without retaining event history."""
        return SceneIngressStats(
            accepted_snapshots=self._accepted_snapshots,
            queue_replacements=self._queue_replacements,
            rejected_connections=self._rejected_connections,
            rejected_handshakes=self._rejected_handshakes,
            rejected_messages=self._rejected_messages,
            read_timeouts=self._read_timeouts,
            sequence_gap_total=self._sequence_gap_total,
            producer_epoch_resets=self._producer_epoch_resets,
            discarded_epoch_snapshots=self._discarded_epoch_snapshots,
            connection_count=self._connection_count,
            connection_active=self._active_writer is not None,
            last_received_monotonic_ns=self._last_received_monotonic_ns,
            last_transport_sequence=self._last_sequence,
            producer_identity=self._accepted_identity,
        )

    def enable_transport_audit(self) -> None:
        """Begin bounded aggregate transport measurement for the next window."""
        if self._transport_audit is not None:
            raise RuntimeError("scene ingress transport audit is already enabled")
        self._transport_audit = _SceneIngressTransportAudit(self._latest.maxsize)

    def finish_transport_audit(self) -> SceneIngressTransportAuditSummary:
        """Return the current aggregate and disable transport measurement."""
        audit = self._transport_audit
        self._transport_audit = None
        if audit is None:
            return SceneIngressTransportAuditSummary(
                status="disabled",
                accepted_snapshots=0,
                payload_bytes_total=0,
                payload_bytes_minimum=None,
                payload_bytes_mean=None,
                payload_bytes_maximum=None,
                sequence_gaps=0,
                queue_capacity=self._latest.maxsize,
                queue_depth_high_water=0,
                queue_depth_zero_before_enqueue=0,
                queue_depth_one_before_enqueue=0,
                queue_replacements=0,
                queue_replacement_fraction=None,
            )
        return audit.summary()

    @property
    def capture_session_acknowledgement(
        self,
    ) -> ValidatedCaptureSessionAcknowledgement | None:
        """Return the last validated live-session report, if live mode is configured."""
        return self._capture_session_acknowledgement

    async def start(self) -> tuple[str, int]:
        """Bind the configured loopback endpoint and return its actual port."""
        if self._closed:
            raise RuntimeError("a closed scene ingress cannot be restarted")
        if self._server is not None:
            return self.address
        self._server = await asyncio.start_server(
            self._schedule_client,
            host=self._settings.bind_address,
            port=self._settings.port,
            limit=self._settings.maximum_message_bytes + 4,
        )
        socket = self._server.sockets[0]
        host, port = socket.getsockname()[:2]
        self._bound_address = (str(host), int(port))
        return self._bound_address

    async def receive_latest(self) -> ReceivedSceneSnapshot:
        """Wait for the newest complete snapshot, skipping superseded ones."""
        return await self._latest.get()

    async def close(self) -> None:
        """Stop accepting clients and release the current connection cleanly."""
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        writer = self._active_writer
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
        current = asyncio.current_task()
        tasks = [task for task in self._client_tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_writer = None

    def _schedule_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._closed or self._active_writer is not None:
            self._rejected_connections += 1
            await self._close_writer(writer)
            return

        self._active_writer = writer
        peer = writer.get_extra_info("peername")
        peer_address = (str(peer[0]), int(peer[1])) if peer else ("unknown", 0)
        connection_id = str(uuid.uuid4())
        handshake_accepted = False
        handshake_rejected = False
        self._connection_count += 1
        try:
            hello = await self._read_json_frame(reader)
            try:
                capture_acknowledgement = self._validate_handshake(hello)
            except SceneIngressError:
                self._rejected_handshakes += 1
                handshake_rejected = True
                raise
            identity = (hello["engine_instance_id"], hello["source_session_id"])
            try:
                self._begin_producer_epoch(identity, capture_acknowledgement)
            except SceneIngressError:
                self._rejected_handshakes += 1
                handshake_rejected = True
                raise
            handshake_accepted = True
            if capture_acknowledgement is None:
                ready = {
                    **_READY_RESPONSE_BASE,
                    "last_transport_sequence": self._last_sequence,
                }
            else:
                ready = {
                    "type": "ready",
                    "protocol_version": 3,
                    "configuration_revision": capture_acknowledgement.configuration_revision,
                    "request_sha256": capture_acknowledgement.request_sha256,
                    "last_transport_sequence": self._last_sequence,
                }
            await self._write_json_frame(writer, ready)

            while not self._closed:
                payload = await self._read_frame(reader)
                self._accept_payload(payload, peer_address, connection_id)
        except asyncio.TimeoutError:
            self._read_timeouts += 1
            logger.debug("Scene producer read timed out")
        except (SceneIngressError, ValidationError) as exc:
            if handshake_accepted:
                self._rejected_messages += 1
            elif not handshake_rejected:
                self._rejected_handshakes += 1
            logger.debug("Scene producer disconnected: %s", exc)
        except asyncio.IncompleteReadError:
            # Normal producer shutdown closes between complete messages.
            pass
        except (ConnectionError, OSError) as exc:
            logger.debug("Scene producer socket ended: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._rejected_messages += 1
            logger.exception("Unexpected error in scene stream")
        finally:
            if self._active_writer is writer:
                self._active_writer = None
            await self._close_writer(writer)

    @staticmethod
    def _valid_hello(message: Any) -> bool:
        if not isinstance(message, dict) or set(message) != {
            "type",
            "protocol_version",
            "engine_instance_id",
            "source_session_id",
        }:
            return False
        if (
            message.get("type") != "hello"
            or type(message.get("protocol_version")) is not int
            or message.get("protocol_version") != _PROTOCOL_VERSION
        ):
            return False
        for field in ("engine_instance_id", "source_session_id"):
            identifier = message.get(field)
            if (
                not isinstance(identifier, str)
                or not identifier
                or len(identifier) > _MAX_IDENTIFIER_CHARACTERS
                or any(
                    ord(character) < 32 or 127 <= ord(character) <= 159
                    for character in identifier
                )
            ):
                return False
        return True

    def _validate_handshake(
        self, message: Any
    ) -> ValidatedCaptureSessionAcknowledgement | None:
        if self._settings.capture_settings is None:
            if not self._valid_hello(message):
                raise SceneIngressError("producer handshake is invalid")
            return None

        if not self._valid_live_hello(message):
            raise SceneIngressError("live producer handshake is invalid")
        try:
            return validate_capture_session_acknowledgement(
                self._settings.capture_settings,
                message["capture_session_ack"],
                expected_session_id=message["source_session_id"],
            )
        except (TypeError, ValueError) as error:
            raise SceneIngressError(
                "live capture acknowledgement is invalid"
            ) from error

    @staticmethod
    def _valid_live_hello(message: Any) -> bool:
        if not isinstance(message, dict) or set(message) != {
            "type",
            "protocol_version",
            "engine_instance_id",
            "source_session_id",
            "capture_session_ack",
        }:
            return False
        if (
            message.get("type") != "hello"
            or type(message.get("protocol_version")) is not int
            or message.get("protocol_version") != 3
            or not isinstance(message.get("capture_session_ack"), dict)
        ):
            return False
        for field in ("engine_instance_id", "source_session_id"):
            identifier = message.get(field)
            if (
                not isinstance(identifier, str)
                or not identifier
                or len(identifier) > _MAX_IDENTIFIER_CHARACTERS
                or any(
                    ord(character) < 32 or 127 <= ord(character) <= 159
                    for character in identifier
                )
            ):
                return False
        return True

    def _begin_producer_epoch(
        self,
        identity: tuple[str, str],
        capture_acknowledgement: ValidatedCaptureSessionAcknowledgement | None,
    ) -> None:
        """Scope sequence watermarks and cached scenes to one source epoch.

        The hello fields are provenance copied from every scene, not a secret
        or permission check. A new identity intentionally starts a new sequence
        space so a restarted producer can publish sequence 1 immediately.
        """
        if identity == self._identity:
            current_ack = self._capture_session_acknowledgement
            if (current_ack is None) != (capture_acknowledgement is None):
                raise SceneIngressError(
                    "producer changed between replay and live mode within one epoch"
                )
            if (
                current_ack is not None
                and capture_acknowledgement is not None
                and (
                    current_ack.configuration_revision
                    != capture_acknowledgement.configuration_revision
                    or current_ack.request_sha256
                    != capture_acknowledgement.request_sha256
                    or current_ack.source_report
                    != capture_acknowledgement.source_report
                )
            ):
                raise SceneIngressError(
                    "live capture request changed without a new source session"
                )
            return

        if self._identity is not None:
            self._producer_epoch_resets += 1
            while True:
                try:
                    self._latest.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._discarded_epoch_snapshots += 1

        self._identity = identity
        self._accepted_identity = None
        self._capture_session_acknowledgement = capture_acknowledgement
        self._last_sequence = None
        self._last_snapshot_revision = None
        self._last_observation = None
        self._last_observation_timestamp = None
        self._last_received_monotonic_ns = None

    async def _read_json_frame(self, reader: asyncio.StreamReader) -> Any:
        payload = await self._read_frame(reader)
        try:
            return json.loads(payload, parse_constant=_reject_non_finite_json_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SceneIngressError("producer handshake is not valid JSON") from exc

    async def _read_frame(self, reader: asyncio.StreamReader) -> bytes:
        timeout = self._settings.read_timeout_seconds
        try:
            header = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                raise SceneIngressError("truncated frame header") from exc
            raise
        payload_size = struct.unpack(">I", header)[0]
        if payload_size == 0:
            raise SceneIngressError("zero-length frame is invalid")
        if payload_size > self._settings.maximum_message_bytes:
            raise SceneIngressError("frame exceeds configured maximum message size")
        try:
            return await asyncio.wait_for(
                reader.readexactly(payload_size), timeout=timeout
            )
        except asyncio.IncompleteReadError as exc:
            raise SceneIngressError("truncated frame body") from exc

    async def _write_json_frame(
        self, writer: asyncio.StreamWriter, message: dict[str, Any]
    ) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        if len(payload) > self._settings.maximum_message_bytes:
            raise SceneIngressError("protocol response exceeds configured maximum")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await asyncio.wait_for(
            writer.drain(), timeout=self._settings.read_timeout_seconds
        )

    def _accept_payload(
        self,
        payload: bytes,
        peer_address: tuple[str, int],
        connection_id: str,
    ) -> None:
        try:
            scene = json.loads(payload, parse_constant=_reject_non_finite_json_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SceneIngressError("scene frame is not valid JSON") from exc
        if not isinstance(scene, dict):
            raise SceneIngressError("scene frame must contain a JSON object")
        delivery: dict[str, Any] | None = None
        if scene.get("delivery_type") == "billiards_public_state_delivery":
            if self._public_state_validator is None:
                raise SceneIngressError(
                    "Rust public-state delivery is not enabled for this producer"
                )
            self._public_state_validator.validate(scene)
            delivery = scene
            raw_scene = delivery.get("source_snapshot")
            if not isinstance(raw_scene, dict):
                raise SceneIngressError("delivery source snapshot must be an object")
            self._validator.validate(raw_scene)
            scene = raw_scene
        else:
            if self._public_state_validator is not None:
                raise SceneIngressError(
                    "Rust producer sent raw scene v1 without its required public-state delivery"
                )
            self._validator.validate(scene)

        identity = (scene["engine_instance_id"], scene["source"]["session_id"])
        if identity != self._identity:
            raise SceneIngressError(
                "scene producer identity differs from its handshake"
            )

        acknowledgement = self._capture_session_acknowledgement
        if acknowledgement is not None:
            report = acknowledgement.source_report
            capture_settings = self._settings.capture_settings
            if capture_settings is not None and capture_settings.schema_version >= 2:
                expected_geometry = report.get("output_frame_geometry")
            else:
                # V1 has no distinct output geometry. Its emitted frames must
                # use the dimensions of the source-reported negotiated mode.
                expected_geometry = report["negotiated_profile"]
            if not isinstance(expected_geometry, dict) or (
                scene["source"]["width_px"] != expected_geometry.get("width_px")
                or scene["source"]["height_px"] != expected_geometry.get("height_px")
            ):
                raise SceneIngressError(
                    "scene dimensions differ from acknowledged capture geometry"
                )

        sequence = scene["transport_sequence"]
        revision = scene["snapshot_revision"]
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise SceneIngressError("transport sequence is duplicate or out of order")
        if (
            self._last_snapshot_revision is not None
            and revision <= self._last_snapshot_revision
        ):
            raise SceneIngressError("snapshot revision is duplicate or out of order")
        if self._last_sequence is not None and sequence > self._last_sequence + 1:
            sequence_gap = sequence - self._last_sequence - 1
        else:
            sequence_gap = 0

        observation = (identity[1], scene["source"]["frame_id"])
        observation_timestamp: tuple[str, str, int] | None = None
        if scene["update_kind"] == "observation":
            if observation == self._last_observation:
                raise SceneIngressError("duplicate source observation was republished")
            timestamp = scene["source"]["timestamp"]
            observation_timestamp = (
                timestamp["clock"],
                timestamp["domain_id"],
                int(timestamp["ticks"]),
            )
            previous_timestamp = self._last_observation_timestamp
            if (
                previous_timestamp is not None
                and observation_timestamp[:2] == previous_timestamp[:2]
                and observation_timestamp[2] <= previous_timestamp[2]
            ):
                raise SceneIngressError("source observation time is stale or repeated")

        receipt_ns = time.monotonic_ns()
        receipt_at = datetime.now(timezone.utc)
        received = ReceivedSceneSnapshot(
            payload=payload,
            received_monotonic_ns=receipt_ns,
            received_at=receipt_at,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            peer_address=peer_address,
            connection_id=connection_id,
        )
        audit = self._transport_audit
        queue_depth_before_enqueue = self._latest.qsize() if audit is not None else 0
        if self._latest.full():
            self._latest.get_nowait()
            self._queue_replacements += 1
        self._latest.put_nowait(received)
        if audit is not None:
            audit.record(
                payload_bytes=len(payload),
                queue_depth_before_enqueue=queue_depth_before_enqueue,
                sequence_gaps=sequence_gap,
            )
        self._accepted_identity = identity
        if observation_timestamp is not None:
            self._last_observation = observation
            self._last_observation_timestamp = observation_timestamp
        self._accepted_snapshots += 1
        self._last_received_monotonic_ns = receipt_ns
        self._last_sequence = sequence
        self._last_snapshot_revision = revision
        self._sequence_gap_total += sequence_gap

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
