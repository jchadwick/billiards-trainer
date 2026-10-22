"""Lifecycle contracts for the supervised Rust live process boundary.

The existing supervisor suite exercises the real Rust-facing wire fixture. These
tests replace only the operating-system child process so teardown timing and
session rejection remain deterministic while still driving the public startup
and close APIs against the real loopback ingress.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from backend.perception.rust_live_supervisor import (
    RustLiveAuditIngress,
    RustLiveStartupError,
    RustLiveSupervisor,
)
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher
from backend.tests.unit.test_rust_live_supervisor import ACK_FIXTURE_ROOT, _settings

if TYPE_CHECKING:
    from pathlib import Path


class RecordingBroadcaster:
    """Keep the supervisor's publication dependency inert and observable."""

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []

    async def broadcast_game_state(self, state: dict[str, Any], **kwargs: Any) -> None:
        del kwargs
        self.states.append(state)


class FakeStdin:
    """Capture the one-shot request and optionally run a child-side handshake."""

    def __init__(self, on_drain: Any = None) -> None:
        self.payload = bytearray()
        self._on_drain = on_drain
        self.closed = False

    def write(self, value: bytes) -> None:
        self.payload.extend(value)

    async def drain(self) -> None:
        if self._on_drain is not None:
            await self._on_drain(bytes(self.payload))

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return


class FakeProcess:
    """A child with explicit exit control and pipe readers for the supervisor."""

    def __init__(
        self,
        *,
        stdin: FakeStdin,
        terminate_exit_code: int | None = None,
    ) -> None:
        self.pid = 43210
        self.returncode: int | None = None
        self.stdin = stdin
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.terminate_count = 0
        self.kill_count = 0
        self._terminate_exit_code = terminate_exit_code
        self._exited = asyncio.Event()

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_count += 1
        if self._terminate_exit_code is not None:
            self.finish(self._terminate_exit_code)

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9
        self._exited.set()

    def finish(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._exited.set()


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-rust-live"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _supervisor(
    tmp_path: Path,
    *,
    startup_timeout_seconds: float = 0.3,
    stop_timeout_seconds: float = 0.03,
    audit_ingress: RustLiveAuditIngress | None = None,
) -> RustLiveSupervisor:
    settings = _settings(
        _executable(tmp_path),
        tmp_path,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    settings = replace(
        settings,
        stop_timeout_seconds=stop_timeout_seconds,
        graceful_stop_timeout_seconds=stop_timeout_seconds / 2,
    )
    return RustLiveSupervisor(
        settings,
        SceneSnapshotStatePublisher(RecordingBroadcaster()),
        audit_ingress=audit_ingress,
    )


def _frame(value: dict[str, Any]) -> bytes:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def test_startup_failure_reaps_child_after_bounded_kill_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that ignores terminate is killed and reaped within stop budget."""

    async def run() -> tuple[RustLiveSupervisor, FakeProcess, float]:
        supervisor = _supervisor(tmp_path)
        process = FakeProcess(
            stdin=FakeStdin(on_drain=lambda _request: _raise_broken_pipe())
        )

        async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
            del args, kwargs
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        started = time.monotonic()
        with pytest.raises(RustLiveStartupError, match="fake child rejected request"):
            await supervisor.start()
        return supervisor, process, time.monotonic() - started

    supervisor, process, elapsed = asyncio.run(run())

    assert elapsed < 0.5
    assert process.terminate_count == 1
    assert process.kill_count == 1
    assert process.returncode == -9
    assert supervisor.status.process_status == "failed"
    assert supervisor.status.process_return_code == -9


async def _raise_broken_pipe() -> None:
    raise BrokenPipeError("fake child rejected request")


def test_stale_capture_ack_session_is_rejected_before_state_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ACK from an earlier source session cannot make this run ready."""

    async def run() -> tuple[RustLiveSupervisor, FakeProcess]:
        supervisor = _supervisor(tmp_path, startup_timeout_seconds=0.08)
        process_holder: list[FakeProcess] = []
        arguments_holder: list[list[str]] = []

        async def send_stale_ack(request_bytes: bytes) -> None:
            arguments = arguments_holder[0]

            def argument(name: str) -> str:
                return arguments[arguments.index(name) + 1]

            host, port = argument("--scene-snapshots-tcp").rsplit(":", 1)
            request = json.loads(request_bytes)
            ack = json.loads(
                (
                    ACK_FIXTURE_ROOT / "v1" / "capture-session-ack-default.json"
                ).read_text(encoding="utf-8")
            )
            ack["request_sha256"] = hashlib.sha256(request_bytes).hexdigest()
            report = ack["source_report"]
            report["session_id"] = "previous-source-session"
            report["requested_profile"] = request["requested"]["requested_profile"]
            report["negotiated_profile"] = request["requested"][
                "expected_negotiated_profile"
            ]
            report["applied_frame_geometry"]["camera_profile_id"] = request[
                "requested"
            ]["camera_profile_id"]
            hello = {
                "type": "hello",
                "protocol_version": 3,
                "engine_instance_id": argument("--engine-instance-id"),
                "source_session_id": argument("--source-session-id"),
                "capture_session_ack": ack,
            }
            reader, writer = await asyncio.open_connection(host, int(port))
            writer.write(_frame(hello))
            await writer.drain()
            # A rejected handshake closes the connection before any ready
            # message. Reading to EOF ensures the rejection is processed before
            # the startup deadline expires.
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()

        stdin = FakeStdin(on_drain=send_stale_ack)
        process = FakeProcess(stdin=stdin, terminate_exit_code=0)
        process_holder.append(process)

        async def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
            del kwargs
            arguments_holder.append(list(args[1:]))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        with pytest.raises(RustLiveStartupError, match="startup deadline"):
            await supervisor.start()
        return supervisor, process_holder[0]

    supervisor, process = asyncio.run(run())

    assert supervisor.status.ingress_rejected_handshakes == 1
    assert supervisor.status.ingress_accepted_snapshots == 0
    assert supervisor.status.state_snapshots_published == 0
    assert not supervisor.status.ready
    assert process.terminate_count == 1
    assert process.returncode == 0


def test_spawn_error_closes_owned_audit_listener_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial startup releases the audit listener and later close is harmless."""

    async def run() -> tuple[RustLiveSupervisor, RustLiveAuditIngress]:
        audit = RustLiveAuditIngress(
            queue_capacity=4,
            maximum_record_bytes=4096,
            connect_timeout_ms=100,
            write_timeout_ms=100,
            startup_timeout_seconds=0.2,
            retained_record_capacity=4,
        )
        supervisor = _supervisor(tmp_path, audit_ingress=audit)

        async def fail_spawn(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise OSError("deterministic spawn failure")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)
        with pytest.raises(RustLiveStartupError, match="deterministic spawn failure"):
            await supervisor.start()
        await supervisor.close()
        await supervisor.close()
        return supervisor, audit

    supervisor, audit = asyncio.run(run())

    assert supervisor.status.process_status == "stopped"
    with pytest.raises(RuntimeError, match="cannot restart"):
        asyncio.run(audit.start())


def test_closed_supervisor_cannot_be_started_again(tmp_path: Path) -> None:
    async def run() -> RustLiveSupervisor:
        supervisor = _supervisor(tmp_path)
        await supervisor.close()
        await supervisor.close()
        with pytest.raises(RuntimeError, match="cannot restart"):
            await supervisor.start()
        return supervisor

    supervisor = asyncio.run(run())

    assert supervisor.status.process_status == "stopped"
    assert not supervisor.status.process_started
