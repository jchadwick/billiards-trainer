"""Runtime freshness and cleanup contracts for the Rust live supervisor.

These tests keep the real scene ingress, worker, and state adapter in the loop.
Only the child process boundary is replaced so frame bursts, source silence,
and process failure remain deterministic.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import struct
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from backend.perception.rust_live_supervisor import RustLiveSupervisor
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher
from backend.tests.unit.test_rust_live_supervisor import (
    ACK_FIXTURE_ROOT,
    SCENE_FIXTURE,
    _settings,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _RecordingBroadcaster:
    """Record state publications and optionally hold one publication open."""

    def __init__(self, *, block_after_first: bool = False) -> None:
        self.states: list[dict[str, Any]] = []
        self.block_after_first = block_after_first
        self.block_entered = asyncio.Event()
        self.release_block = asyncio.Event()

    async def broadcast_game_state(self, **state: Any) -> None:
        self.states.append(state)
        if self.block_after_first and len(self.states) == 2:
            self.block_entered.set()
            await self.release_block.wait()


class _FakeStdin:
    def __init__(self, on_drain: Any) -> None:
        self._on_drain = on_drain
        self.payload = bytearray()
        self.closed = False

    def write(self, value: bytes) -> None:
        self.payload.extend(value)

    async def drain(self) -> None:
        await self._on_drain(bytes(self.payload))

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeChild:
    """A live child with a genuine TCP scene session and controllable exit."""

    def __init__(self) -> None:
        self.pid = 74123
        self.returncode: int | None = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.writer: asyncio.StreamWriter | None = None
        self._exited = asyncio.Event()
        self.stdin = _FakeStdin(self._open_scene_session)
        self._request: dict[str, Any] | None = None
        self._arguments: list[str] = []
        self._engine_instance_id = ""
        self._source_session_id = ""

    def configure(self, arguments: list[str]) -> None:
        self._arguments = arguments
        self._engine_instance_id = self._argument("--engine-instance-id")
        self._source_session_id = self._argument("--source-session-id")

    def _argument(self, name: str) -> str:
        return self._arguments[self._arguments.index(name) + 1]

    async def _open_scene_session(self, request_bytes: bytes) -> None:
        self._request = json.loads(request_bytes)
        host, raw_port = self._argument("--scene-snapshots-tcp").rsplit(":", 1)
        reader, self.writer = await asyncio.open_connection(host, int(raw_port))
        acknowledgement = _acknowledgement(
            request_bytes,
            self._request,
            self._source_session_id,
        )
        await _write_frame(
            self.writer,
            {
                "type": "hello",
                "protocol_version": 3,
                "engine_instance_id": self._engine_instance_id,
                "source_session_id": self._source_session_id,
                "capture_session_ack": acknowledgement,
            },
        )
        ready = await _read_frame(reader)
        assert ready["type"] == "ready"
        await self.send_scene(1)

    async def send_scene(self, sequence: int) -> None:
        assert self.writer is not None
        assert self._request is not None
        scene = _scene(
            sequence,
            self._engine_instance_id,
            self._source_session_id,
            self._request["requested"]["expected_negotiated_profile"],
        )
        await _write_frame(self.writer, scene)

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.finish(-15)

    def kill(self) -> None:
        self.finish(-9)

    def finish(self, returncode: int) -> None:
        if self.returncode is not None:
            return
        self.returncode = returncode
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        if self.writer is not None:
            self.writer.close()
        self._exited.set()


def _acknowledgement(
    request_bytes: bytes,
    request: dict[str, Any],
    source_session_id: str,
) -> dict[str, Any]:
    acknowledgement = json.loads(
        (ACK_FIXTURE_ROOT / "v1" / "capture-session-ack-default.json").read_text(
            encoding="utf-8"
        )
    )
    acknowledgement["request_sha256"] = hashlib.sha256(request_bytes).hexdigest()
    report = acknowledgement["source_report"]
    report["session_id"] = source_session_id
    report["requested_profile"] = request["requested"]["requested_profile"]
    report["negotiated_profile"] = request["requested"]["expected_negotiated_profile"]
    report["applied_frame_geometry"]["camera_profile_id"] = request["requested"][
        "camera_profile_id"
    ]
    return acknowledgement


def _scene(
    sequence: int,
    engine_instance_id: str,
    source_session_id: str,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    scene = copy.deepcopy(json.loads(SCENE_FIXTURE.read_text(encoding="utf-8")))
    frame_id = f"runtime-frame-{sequence:04d}"
    ticks = str(1_000_000_000 + sequence * 40_000_000)
    timestamp = {
        "ticks": ticks,
        "unit": "nanoseconds",
        "clock": "monotonic",
        "domain_id": "supervisor-runtime-test",
        "stage": "shared_memory_publication",
    }
    scene["engine_instance_id"] = engine_instance_id
    scene["source"].update(
        {
            "session_id": source_session_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "width_px": geometry["width_px"],
            "height_px": geometry["height_px"],
        }
    )
    scene["published_at"]["ticks"] = str(2_000_000_000 + sequence * 40_000_000)
    scene["transport_sequence"] = sequence
    scene["snapshot_revision"] = sequence
    for ball in scene["balls"]:
        ball["last_observation"] = {
            "session_id": source_session_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
        }
        if ball["measurement"] is not None:
            ball["measurement"]["image_width_px"] = geometry["width_px"]
            ball["measurement"]["image_height_px"] = geometry["height_px"]
    scene["cue"]["last_observation"] = {
        "session_id": source_session_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
    }
    scene["cue"]["image_width_px"] = geometry["width_px"]
    scene["cue"]["image_height_px"] = geometry["height_px"]
    return scene


async def _write_frame(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    size = struct.unpack(">I", await reader.readexactly(4))[0]
    value = json.loads(await reader.readexactly(size))
    assert isinstance(value, dict)
    return value


async def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition was not reached before the test deadline")


def _mock_child(monkeypatch: pytest.MonkeyPatch, child: _FakeChild) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> _FakeChild:
        del kwargs
        child.configure(list(args[1:]))
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)


def test_supervisor_keeps_only_newest_scene_while_state_publication_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow client cannot make the supervisor replay its queued past."""

    async def run() -> tuple[list[dict[str, Any]], Any]:
        child = _FakeChild()
        _mock_child(monkeypatch, child)
        broadcaster = _RecordingBroadcaster(block_after_first=True)
        # The executable is validated before the monkeypatched spawn boundary.
        executable = tmp_path / "fake-rust-live"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        supervisor = RustLiveSupervisor(
            _settings(executable, tmp_path),
            SceneSnapshotStatePublisher(broadcaster),
        )
        try:
            assert (await supervisor.start()).ready
            await child.send_scene(2)
            await asyncio.wait_for(broadcaster.block_entered.wait(), timeout=1.0)
            await child.send_scene(3)
            await child.send_scene(4)
            await _wait_until(
                lambda: supervisor.status.ingress_accepted_snapshots == 4
                and supervisor.status.ingress_queue_replacements >= 1
            )
            broadcaster.release_block.set()
            await _wait_until(lambda: len(broadcaster.states) >= 3)

            delivered_frames = [
                state["observation_status"]["source_frame_id"]
                for state in broadcaster.states
            ]
            assert delivered_frames == [
                "runtime-frame-0001",
                "runtime-frame-0002",
                "runtime-frame-0004",
            ]
            return broadcaster.states, supervisor
        finally:
            broadcaster.release_block.set()
            await supervisor.close()

    states, supervisor = asyncio.run(run())

    assert len(states) == 3
    assert supervisor.status.process_status == "stopped"
    assert supervisor.worker_stats.published == 3
    assert supervisor.status.ingress_queue_replacements == 1


def test_supervisor_marks_cached_positions_stale_then_recovers_on_new_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silence invalidates freshness until a genuinely newer observation arrives."""

    async def run() -> tuple[_RecordingBroadcaster, RustLiveSupervisor]:
        executable = tmp_path / "fake-rust-live"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        child = _FakeChild()
        _mock_child(monkeypatch, child)
        broadcaster = _RecordingBroadcaster()
        settings = replace(
            _settings(executable, tmp_path),
            source_silence_timeout_seconds=0.04,
        )
        supervisor = RustLiveSupervisor(
            settings,
            SceneSnapshotStatePublisher(broadcaster),
        )
        try:
            assert (await supervisor.start()).ready
            await _wait_until(lambda: len(broadcaster.states) == 2, timeout=1.0)
            stale = broadcaster.states[-1]
            assert stale["observation_status"]["freshness_status"] == "source_silent"
            assert all(ball["is_observed"] is False for ball in stale["balls"])
            assert not supervisor.status.ready

            await child.send_scene(2)
            await _wait_until(lambda: len(broadcaster.states) == 3)
            current = broadcaster.states[-1]
            assert current["observation_status"]["source_frame_id"] == (
                "runtime-frame-0002"
            )
            assert all(ball["is_observed"] for ball in current["balls"])
            assert supervisor.status.ready
            return broadcaster, supervisor
        finally:
            await supervisor.close()

    broadcaster, supervisor = asyncio.run(run())

    assert len(broadcaster.states) == 3
    assert supervisor.worker_stats.health_updates_published == 1
    assert supervisor.status.process_status == "stopped"


def test_unexpected_child_exit_after_readiness_is_reported_and_fully_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead producer is unhealthy, and close still releases worker resources."""

    async def run() -> tuple[RustLiveSupervisor, _FakeChild]:
        executable = tmp_path / "fake-rust-live"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        child = _FakeChild()
        _mock_child(monkeypatch, child)
        supervisor = RustLiveSupervisor(
            _settings(executable, tmp_path),
            SceneSnapshotStatePublisher(_RecordingBroadcaster()),
        )
        assert (await supervisor.start()).ready

        child.finish(9)
        assert supervisor._process_wait_task is not None
        assert await asyncio.wait_for(supervisor._process_wait_task, timeout=1.0) == 9
        failed = supervisor.status
        assert failed.process_status == "failed"
        assert failed.process_return_code == 9
        assert failed.source_health == "process_failed"
        assert not failed.ready

        await supervisor.close()
        return supervisor, child

    supervisor, child = asyncio.run(run())

    assert child.returncode == 9
    assert supervisor.status.process_status == "stopped"
    assert supervisor._worker._task is None
    assert supervisor._ingress._server is None


def test_output_drain_retains_only_the_configured_diagnostic_tail(
    tmp_path: Path,
) -> None:
    """Noisy child output is drained without unbounded diagnostic retention."""

    async def run() -> bytearray:
        executable = tmp_path / "fake-rust-live"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        supervisor = RustLiveSupervisor(
            replace(
                _settings(executable, tmp_path),
                output_tail_capacity_bytes=8,
            ),
            SceneSnapshotStatePublisher(_RecordingBroadcaster()),
        )
        stream = asyncio.StreamReader()
        stream.feed_data(b"startup\nframe-001\nframe-002\n")
        stream.feed_eof()
        tail = bytearray()
        await supervisor._drain_output(stream, tail)
        return tail

    assert asyncio.run(run()) == b"ame-002\n"


def test_supervisor_can_close_after_process_exit_without_waiting_for_it_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process exit and repeated close preserve idempotent cleanup behavior."""

    async def run() -> RustLiveSupervisor:
        executable = tmp_path / "fake-rust-live"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        child = _FakeChild()
        _mock_child(monkeypatch, child)
        supervisor = RustLiveSupervisor(
            _settings(executable, tmp_path),
            SceneSnapshotStatePublisher(_RecordingBroadcaster()),
        )
        assert (await supervisor.start()).ready
        child.finish(0)
        assert supervisor._process_wait_task is not None
        await asyncio.wait_for(supervisor._process_wait_task, timeout=1.0)
        assert supervisor.status.process_status == "completed"
        await supervisor.close()
        await supervisor.close()
        return supervisor

    supervisor = asyncio.run(run())

    assert supervisor.status.process_status == "stopped"
    assert supervisor._worker._task is None
