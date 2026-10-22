"""Exercise Rust replay from API startup through HTTP and WebSocket delivery."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from backend.api import main
from backend.config import load_schema_reference
from backend.tests.support.compat_api import create_compatibility_app

SCENE_FIXTURE = (
    Path(__file__).parents[3]
    / "perception/contracts/v1/scene-rust-tracked-success.json"
)
PUBLIC_STATE_FIXTURE = (
    Path(__file__).parents[3] / "perception/contracts/public-state/number-8.json"
)


def _write_replay_assets(
    tmp_path: Path,
    *,
    child_exit_code: int | None = None,
    exit_signal_path: Path | None = None,
) -> dict[str, Any]:
    """Create validated model/input assets and a loopback-only fake producer."""
    if exit_signal_path is not None and child_exit_code is None:
        raise ValueError("a child exit code is required with an exit signal path")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX asset for lifecycle test")
    Path(f"{model_path}.manifest.json").write_text(
        json.dumps(
            {
                "model": {
                    "class_names": [
                        {"id": 0, "name": "cue"},
                        {"id": 1, "name": "ball"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"source checkpoint metadata")
    input_path = tmp_path / "frame.png"
    input_path.write_bytes(b"recorded input metadata")

    executable_path = tmp_path / "fake-rust-replay"
    executable_path.write_text(
        f"#!{sys.executable}\n"
        "import copy, json, socket, struct, sys, time\n"
        "from pathlib import Path\n"
        "def arg(name): return sys.argv[sys.argv.index(name) + 1]\n"
        "def read_exact(sock, count):\n"
        "    data = bytearray()\n"
        "    while len(data) < count:\n"
        "        chunk = sock.recv(count - len(data))\n"
        "        if not chunk: raise EOFError()\n"
        "        data.extend(chunk)\n"
        "    return bytes(data)\n"
        "def send_frame(sock, value):\n"
        "    data = json.dumps(value, separators=(',', ':')).encode()\n"
        "    sock.sendall(struct.pack('>I', len(data)) + data)\n"
        "def read_frame(sock):\n"
        "    size = struct.unpack('>I', read_exact(sock, 4))[0]\n"
        "    return json.loads(read_exact(sock, size))\n"
        "target_host, target_port = arg('--scene-snapshots-tcp').rsplit(':', 1)\n"
        "client = socket.create_connection((target_host, int(target_port)))\n"
        f"scene = json.loads(Path({str(SCENE_FIXTURE)!r}).read_text())\n"
        "scene['engine_instance_id'] = arg('--engine-instance-id')\n"
        "session_id = arg('--source-session-id')\n"
        "scene['source']['session_id'] = session_id\n"
        "send_frame(client, {'type': 'hello', 'protocol_version': 2, "
        "'engine_instance_id': scene['engine_instance_id'], "
        "'source_session_id': session_id})\n"
        "if read_frame(client).get('type') != 'ready': sys.exit(2)\n"
        f"for sequence in range(1, {2 if child_exit_code is not None and exit_signal_path is None else 1000000}):\n"
        "    frame_id = f'frame-{sequence:06d}'\n"
        "    ticks = str(time.monotonic_ns())\n"
        "    scene['snapshot_revision'] = sequence\n"
        "    scene['transport_sequence'] = sequence\n"
        "    scene['source']['frame_id'] = frame_id\n"
        "    scene['source']['timestamp']['ticks'] = ticks\n"
        "    scene['published_at']['ticks'] = str(time.monotonic_ns())\n"
        "    for item in [scene['cue'], *scene['balls']]:\n"
        "        observation = item.get('last_observation')\n"
        "        if observation is not None:\n"
        "            observation['session_id'] = session_id\n"
        "            observation['frame_id'] = frame_id\n"
        "            observation['timestamp']['ticks'] = ticks\n"
        f"    public_fixture = json.loads(Path({str(PUBLIC_STATE_FIXTURE)!r}).read_text())\n"
        "    current_state = copy.deepcopy(public_fixture['expected'])\n"
        "    status = current_state['observation_status']\n"
        "    status.update(engine_instance_id=scene['engine_instance_id'], source_session_id=session_id, source_frame_id=frame_id, source_timestamp=ticks, ball_observation_source_timestamp=ticks, snapshot_revision=sequence, transport_sequence=sequence)\n"
        "    status['observation_identity'].update(engine_instance_id=scene['engine_instance_id'], session_id=session_id, frame_id=frame_id, snapshot_revision=sequence)\n"
        "    for ball in current_state['balls']:\n"
        "        observation = ball['source_observation']\n"
        "        observation.update(engine_instance_id=scene['engine_instance_id'], session_id=session_id, frame_id=frame_id)\n"
        "        observation['timestamp']['ticks'] = ticks\n"
        "    current_state_delivery = {'delivery_type': 'billiards_public_state_delivery', 'delivery_schema_version': 1, 'source_snapshot': scene, 'current_state': {'payload': current_state, 'epoch': 1, 'updated_monotonic_ns': time.monotonic_ns(), 'localization_received_monotonic_ns': time.monotonic_ns(), 'localization_age_ns': 0, 'ball_age_ns': {ball['id']: 0 for ball in current_state['balls']}, 'freshness': 'fresh'}}\n"
        "    send_frame(client, current_state_delivery)\n"
        + (
            f"    if Path({str(exit_signal_path)!r}).exists(): sys.exit({child_exit_code})\n"
            if exit_signal_path is not None
            else ""
        )
        + f"    time.sleep({0.2 if child_exit_code is not None else 0.05})\n"
        + (f"sys.exit({child_exit_code})\n" if child_exit_code is not None else ""),
        encoding="utf-8",
    )
    executable_path.chmod(0o755)

    replay_defaults = load_schema_reference()["perception"]["rust_replay"]
    return {
        **replay_defaults,
        "executable_path": str(executable_path),
        "onnx_path": str(model_path),
        "source_checkpoint_path": str(checkpoint_path),
        "input_path": str(input_path),
        "startup_timeout_seconds": 10.0,
        "stop_timeout_seconds": 2.0,
    }


def test_configured_rust_replay_reaches_http_and_unauthenticated_websocket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anonymous state delivery remains explicit when its child later fails."""
    child_exit_signal = tmp_path / "stop-rust-child"
    replay_config = _write_replay_assets(
        tmp_path,
        child_exit_code=9,
        exit_signal_path=child_exit_signal,
    )
    replay_config["source_silence_timeout_seconds"] = 1.0
    app = create_compatibility_app(
        {
            "perception": {
                "provider": "rust",
                "mode": "recorded_replay",
                "rust_replay": replay_config,
            }
        }
    )

    async def allow_client(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(main.rate_limiter, "initialize", AsyncMock(return_value=True))
    monkeypatch.setattr(main.rate_limiter, "check_rate_limit", allow_client)
    monkeypatch.setattr(main.rate_limiter, "close", AsyncMock())
    monkeypatch.setattr(main, "setup_signal_handlers", Mock())
    monkeypatch.setattr(main, "_register_shutdown_functions", AsyncMock())
    monkeypatch.setattr(
        main,
        "health_monitor",
        SimpleNamespace(
            register_components=Mock(),
            start_monitoring=AsyncMock(),
            stop_monitoring=AsyncMock(),
        ),
    )

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/game/state")
            assert response.status_code == 200
            http_state = response.json()
            assert http_state["perception_source"] == "rust_scene"
            assert http_state["balls"][0]["source_observation"]["session_id"]
            assert http_state["balls"][0]["position"] == {
                "x": 1882.4,
                "y": 1024.7,
                "scale": [1.0, 1.0],
            }

            # No login, token, or query credential is part of local connection.
            with client.websocket_connect(
                "/api/v1/websocket/ws",
                headers={
                    "Authorization": "Bearer deliberately-invalid",
                    "X-API-Key": "deliberately-invalid",
                    "Cookie": "session=deliberately-invalid",
                },
            ) as websocket:
                welcome = websocket.receive_json()
                assert welcome["type"] == "connection"
                websocket.send_json(
                    {"type": "subscribe", "data": {"streams": ["state"]}}
                )
                received: list[dict[str, Any]] = []
                for _ in range(8):
                    message = websocket.receive_json()
                    received.append(message)
                    if message.get("type") == "state":
                        break

                state_message = next(
                    message for message in received if message.get("type") == "state"
                )
                assert any(message.get("type") == "subscribed" for message in received)
                websocket_ball = state_message["data"]["balls"][0]
                assert websocket_ball["position"] == http_state["balls"][0]["position"]
                assert websocket_ball["source_observation"]["session_id"] == (
                    http_state["balls"][0]["source_observation"]["session_id"]
                )
                assert websocket_ball.get("color") is None

            healthy_source = client.get("/api/v1/health/perception").json()
            assert healthy_source["process_status"] == "running"
            assert healthy_source["source_health"] == "healthy"

            # The process gets no credentials; a local test signal asks this
            # fake producer to exit after it has served the anonymous client.
            child_exit_signal.touch()
            deadline = time.monotonic() + 2.0
            failed_source = client.get("/api/v1/health/perception").json()
            while (
                failed_source.get("process_return_code") is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                failed_source = client.get("/api/v1/health/perception").json()
            assert failed_source["process_return_code"] == 9
            assert failed_source["process_status"] == "failed"
            assert failed_source["source_health"] == "process_failed"
            assert failed_source["last_observation_age_seconds"] is not None
            assert failed_source["source_silence_timeout_seconds"] == 1.0

            deadline = time.monotonic() + 2.0
            stale_response = client.get("/api/v1/game/state")
            while (
                stale_response.json()
                .get("observation_status", {})
                .get("freshness_status")
                != "source_silent"
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
                stale_response = client.get("/api/v1/game/state")
            assert stale_response.status_code == 200
            stale_state = stale_response.json()
            assert stale_state["frame_number"] is None
            assert stale_state["observation_status"]["freshness_status"] == (
                "source_silent"
            )
            assert stale_state["balls"][0]["is_observed"] is False
    finally:
        # The lifespan reaps the process; clear process-global app state for
        # any tests that run later in this interpreter.
        main.app_state.core_module = None
        main.app_state.perception_provider = None


def test_real_rust_model_replay_reaches_anonymous_http_and_websocket_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in black-box check of the real model through the local API gateway.

    This confirms end-to-end inference and delivery only. The still-image input
    has no temporal ground truth or applied calibration, so the test does not
    claim physical accuracy or live camera freshness.
    """
    environment_paths = {
        "executable_path": "BILLIARDS_RUST_REPLAY_AUDIT_EXECUTABLE",
        "onnx_path": "BILLIARDS_RUST_REPLAY_AUDIT_ONNX",
        "source_checkpoint_path": "BILLIARDS_RUST_REPLAY_AUDIT_CHECKPOINT",
        "input_path": "BILLIARDS_RUST_REPLAY_AUDIT_INPUT",
    }
    missing = [name for name in environment_paths.values() if not os.getenv(name)]
    if missing:
        pytest.skip(
            "set the real Rust replay audit paths to run this model-dependent check: "
            + ", ".join(missing)
        )

    replay_config = load_schema_reference()["perception"]["rust_replay"]
    replay_config.update(
        {
            key: os.environ[environment_name]
            for key, environment_name in environment_paths.items()
        }
    )
    replay_config.update(
        {
            "provider": "cpu",
            "source_image_space": "raw",
            "replay_rate_fps": 5.0,
            "startup_timeout_seconds": 30.0,
        }
    )
    replay_input = Path(replay_config["input_path"])
    if replay_input.is_file():
        # Repeat one known, ball-rich still so a finite replay continues to
        # publish after the anonymous WebSocket has subscribed.
        replay_directory = tmp_path / "real-model-replay"
        replay_directory.mkdir()
        for frame_number in range(8):
            shutil.copyfile(
                replay_input,
                replay_directory / f"frame-{frame_number:03d}{replay_input.suffix}",
            )
        replay_config["input_path"] = str(replay_directory)
    app = create_compatibility_app(
        {
            "perception": {
                "provider": "rust",
                "mode": "recorded_replay",
                "rust_replay": replay_config,
            }
        }
    )

    async def allow_client(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(main.rate_limiter, "initialize", AsyncMock(return_value=True))
    monkeypatch.setattr(main.rate_limiter, "check_rate_limit", allow_client)
    monkeypatch.setattr(main.rate_limiter, "close", AsyncMock())
    monkeypatch.setattr(main, "setup_signal_handlers", Mock())
    monkeypatch.setattr(main, "_register_shutdown_functions", AsyncMock())
    monkeypatch.setattr(
        main,
        "health_monitor",
        SimpleNamespace(
            register_components=Mock(),
            start_monitoring=AsyncMock(),
            stop_monitoring=AsyncMock(),
        ),
    )

    try:
        with TestClient(app) as client:
            http_response = client.get("/api/v1/game/state")
            assert http_response.status_code == 200
            http_state = http_response.json()
            assert http_state["perception_source"] == "rust_scene"
            assert http_state["balls"]
            assert any(ball.get("source_observation") for ball in http_state["balls"])

            with client.websocket_connect("/api/v1/websocket/ws") as websocket:
                welcome = websocket.receive_json()
                assert welcome["type"] == "connection"
                websocket.send_json(
                    {"type": "subscribe", "data": {"streams": ["state"]}}
                )
                messages: list[dict[str, Any]] = []
                for _ in range(8):
                    message = websocket.receive_json()
                    messages.append(message)
                    if message.get("type") == "state":
                        break

            state_message = next(
                message for message in messages if message.get("type") == "state"
            )
            assert any(message.get("type") == "subscribed" for message in messages)
            websocket_balls = state_message["data"]["balls"]
            assert websocket_balls
            assert any(ball.get("source_observation") for ball in websocket_balls)
            assert websocket_balls[0]["position"] == http_state["balls"][0]["position"]
    finally:
        main.app_state.core_module = None
        main.app_state.perception_provider = None
