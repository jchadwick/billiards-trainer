"""Lifecycle tests for the opt-in, replay-only Rust provider supervisor."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.config import load_schema_reference, validate_configuration
from backend.perception.rust_replay_supervisor import (
    RustReplayConfigurationError,
    RustReplaySettings,
    RustReplayStartupError,
    RustReplaySupervisor,
)
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCENE_FIXTURE = PROJECT_ROOT / "perception/contracts/v1/scene-rust-tracked-success.json"


class DictConfiguration:
    """Dot-path configuration stub with the same fallback contract as Config."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self._values
        for component in key.split("."):
            if not isinstance(current, dict) or component not in current:
                return default
            current = current[component]
        return current


class RecordingBroadcaster:
    """Capture the existing public state publication without a WebSocket."""

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []

    async def broadcast_game_state(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.states.append(kwargs)


def _settings(tmp_path: Path, **updates: Any) -> RustReplaySettings:
    executable = tmp_path / "rust-replay"
    executable.write_text(f"#!{sys.executable}\npass\n", encoding="utf-8")
    executable.chmod(0o755)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    Path(f"{onnx_path}.manifest.json").write_text(
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
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"recorded image")
    defaults = load_schema_reference()["perception"]["rust_replay"]
    values = {
        key: (
            str(value)
            if key
            in {
                "executable_path",
                "onnx_path",
                "source_checkpoint_path",
                "input_path",
            }
            else value
        )
        for key, value in defaults.items()
    }
    values.update(
        {
            "executable_path": str(executable),
            "onnx_path": str(onnx_path),
            "source_checkpoint_path": str(checkpoint_path),
            "input_path": str(input_path),
            **updates,
        }
    )
    return RustReplaySettings.from_configuration(
        DictConfiguration({"perception": {"rust_replay": values}}),
        project_root=tmp_path,
    )


def _fake_executable(
    settings: RustReplaySettings,
    tmp_path: Path,
    *,
    mode: str,
    record_path: Path | None = None,
    scene_fixture_path: Path = SCENE_FIXTURE,
) -> Path:
    executable = tmp_path / f"fake-{mode}"
    child_record = record_path or tmp_path / "child-record.json"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, socket, struct, sys, time\n"
        "from pathlib import Path\n"
        "def arg(name): return sys.argv[sys.argv.index(name) + 1]\n"
        f"record = Path({str(child_record)!r})\n"
        "record.write_text(json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ)}))\n"
        + (
            "batch = Path(arg('--detection-batches-jsonl'))\n"
            "batch.write_bytes(b'12345678901234567890')\n"
            + ("time.sleep(20)\n" if mode == "batch_limit" else "")
            if mode in {"batch_limit", "batch_fast_exit"}
            else ""
        )
        + (
            "print('fake producer failed', file=sys.stderr, flush=True)\n"
            "sys.exit(7)\n"
            if mode == "exit"
            else ""
        )
        + (
            (
                "target = arg('--scene-snapshots-tcp').rsplit(':', 1)\n"
                "client = socket.create_connection((target[0], int(target[1])))\n"
                "def read_frame():\n"
                "  size = struct.unpack('>I', client.recv(4))[0]\n"
                "  data = b''\n"
                "  while len(data) < size: data += client.recv(size-len(data))\n"
                "  return json.loads(data)\n"
                f"scene = json.loads(Path({str(scene_fixture_path)!r}).read_text())\n"
                "scene['engine_instance_id'] = arg('--engine-instance-id')\n"
                "scene['source']['session_id'] = arg('--source-session-id')\n"
                "hello = {'type':'hello','protocol_version':2, "
                "'engine_instance_id':scene['engine_instance_id'], "
                "'source_session_id':scene['source']['session_id']}\n"
                "data = json.dumps(hello, separators=(',',':')).encode()\n"
                "client.sendall(struct.pack('>I',len(data)) + data)\n"
                "read_frame()\n"
                "scene['transport_sequence'] = 1\n"
                "scene['snapshot_revision'] = 1\n"
                "data = json.dumps(scene, separators=(',',':')).encode()\n"
                "client.sendall(struct.pack('>I',len(data)) + data)\n"
                "client.close()\n"
                "print('fake replay complete', flush=True)\n"
                "print('z' * 70000, file=sys.stderr, flush=True)\n"
                + ("time.sleep(20)\n" if mode == "long" else "")
            )
            if mode not in {"exit", "empty_success", "batch_fast_exit"}
            else ""
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def test_provider_defaults_are_valid_and_replay_paths_are_explicit(
    tmp_path: Path,
) -> None:
    assert validate_configuration({"perception": {"provider": "rust"}})[0] == []
    with pytest.raises(RustReplayConfigurationError, match="executable"):
        RustReplaySettings.from_configuration(
            DictConfiguration({"perception": {"provider": "rust"}}),
            project_root=tmp_path,
        )


def test_replay_settings_validate_manifest_class_ids_and_replay_only_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(RustReplayConfigurationError, match="class IDs"):
        _settings(tmp_path, ball_class_id=8)

    input_path = tmp_path / "not-a-frame.mp4"
    input_path.write_bytes(b"recorded clip")
    with pytest.raises(RustReplayConfigurationError, match="PNG/JPEG"):
        _settings(tmp_path, input_path=str(input_path))


def test_config_schema_validates_replay_thresholds_and_enum_values() -> None:
    errors, _ = validate_configuration(
        {
            "perception": {
                "rust_replay": {
                    "provider": "mps",
                    "confidence": 1.1,
                    "image_size": 24,
                    "maximum_message_bytes": 0,
                }
            }
        }
    )

    assert {
        "perception.rust_replay.provider",
        "perception.rust_replay.confidence",
        "perception.rust_replay.image_size",
        "perception.rust_replay.maximum_message_bytes",
    } <= {error["field"] for error in errors}


@pytest.mark.parametrize("provider", ["cpu", "auto", "openvino"])
def test_config_schema_accepts_explicit_rust_execution_providers(
    provider: str,
) -> None:
    errors, _ = validate_configuration(
        {"perception": {"rust_replay": {"provider": provider}}}
    )

    assert "perception.rust_replay.provider" not in {error["field"] for error in errors}


@pytest.mark.parametrize("provider", ["cpu", "openvino"])
def test_supervisor_launches_replay_and_publishes_first_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    settings = _settings(tmp_path, provider=provider)
    record = tmp_path / "child-record.json"
    fake = _fake_executable(settings, tmp_path, mode="success", record_path=record)
    settings = RustReplaySettings.from_configuration(
        DictConfiguration(
            {
                "perception": {
                    "rust_replay": {
                        **{
                            field: (
                                getattr(settings, field).as_posix()
                                if isinstance(getattr(settings, field), Path)
                                else getattr(settings, field)
                            )
                            for field in settings.__dataclass_fields__
                            if field != "project_root"
                            and not field.endswith("_bundle_path")
                        },
                        "executable_path": str(fake),
                        "calibration_bundle_path": "",
                        "applied_camera_profile_id": "",
                    }
                }
            }
        ),
        project_root=tmp_path,
    )
    # Keep the child environment minimal. OpenVINO is the only provider that
    # receives the platform dynamic-library search path required by its runtime.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("PATH", "/test/parent/path")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/openvino/runtime/libs")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/test/openvino/runtime/dylibs")
    secret_environment = {
        "API_KEY": "test-api-key-secret",
        "AUTH_TOKEN": "test-auth-token-secret",
        "BILLIARDS_API_TOKEN": "test-app-token-secret",
        "DATABASE_PASSWORD": "test-database-password-secret",
        "DATABASE_URL": "postgresql://test-user:test-password@localhost/test-db",
        "GITHUB_TOKEN": "test-github-token-secret",
        "OPENAI_API_KEY": "test-provider-key-secret",
        "AWS_SECRET_ACCESS_KEY": "test-cloud-secret",
    }
    for name, value in secret_environment.items():
        monkeypatch.setenv(name, value)
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    spawned_environment: dict[str, str] = {}

    async def capture_environment(
        *args: Any, **kwargs: Any
    ) -> asyncio.subprocess.Process:
        spawned_environment.update(kwargs["env"])
        return await original_create_subprocess_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_environment)
    broadcaster = RecordingBroadcaster()
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(broadcaster)
    )

    async def run() -> None:
        status = await supervisor.start()
        assert status.mode == "recorded_replay"
        assert status.process_started is True
        assert status.first_observation_received is True
        assert status.live_camera is False
        assert status.preview == "unavailable"
        assert status.predictions == "unavailable"
        assert status.ingress_accepted_snapshots == 1
        assert status.state_snapshots_published == 1
        assert supervisor._process_wait_task is not None
        await supervisor._process_wait_task
        assert supervisor.status.process_status == "completed"
        assert supervisor.status.source_health != "process_failed"
        await supervisor.close()

    asyncio.run(run())
    record_data = json.loads(record.read_text(encoding="utf-8"))
    expected_environment = {"TMPDIR": str(tmp_path)}
    if provider == "openvino":
        expected_environment["LD_LIBRARY_PATH"] = "/test/openvino/runtime/libs"
        expected_environment["DYLD_LIBRARY_PATH"] = "/test/openvino/runtime/dylibs"
    assert spawned_environment == expected_environment
    # macOS may populate locale and CoreFoundation process metadata for the
    # interpreter launched from the test fixture after exec starts.
    assert record_data["env"].get("TMPDIR") == str(tmp_path)
    assert all(
        value not in record_data["env"].values()
        for value in secret_environment.values()
    )
    assert ("LD_LIBRARY_PATH" in record_data["env"]) is (provider == "openvino")
    assert ("DYLD_LIBRARY_PATH" in record_data["env"]) is (provider == "openvino")
    assert "--scene-snapshots-tcp" in record_data["argv"]
    assert "--detection-batches-jsonl" in record_data["argv"]
    provider_argument = record_data["argv"].index("--provider")
    assert record_data["argv"][provider_argument + 1] == provider
    assert len(broadcaster.states) == 1
    assert len(supervisor._stderr_tail) == 64 * 1024
    assert b"fake replay complete" in supervisor._stdout_tail


def test_nonzero_child_exit_fails_startup_without_python_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake = _fake_executable(settings, tmp_path, mode="exit")
    settings = RustReplaySettings(
        **{
            **settings.__dict__,
            "executable_path": fake,
            "startup_timeout_seconds": 0.5,
        }
    )
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )

    async def run() -> None:
        with pytest.raises(RustReplayStartupError, match="status 7"):
            await supervisor.start()
        assert supervisor.status.live_camera is False
        assert supervisor.status.first_observation_received is False

    asyncio.run(run())


def test_successful_child_exit_without_scene_fails_readiness_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        startup_timeout_seconds=0.5,
        output_limit_poll_interval_seconds=10.0,
    )
    record = tmp_path / "child-record.json"
    fake = _fake_executable(
        settings, tmp_path, mode="empty_success", record_path=record
    )
    settings = RustReplaySettings(**{**settings.__dict__, "executable_path": fake})
    broadcaster = RecordingBroadcaster()
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(broadcaster)
    )

    async def run() -> None:
        try:
            # Exit code zero cannot establish readiness: only a scene that
            # reaches the retained state publisher fulfills that contract.
            with pytest.raises(RustReplayStartupError, match="readiness deadline"):
                await supervisor.start()

            status = supervisor.status
            assert status.process_started is True
            assert status.process_running is False
            assert status.process_return_code == 0
            assert status.first_observation_received is False
            assert status.ingress_accepted_snapshots == 0
            assert status.state_snapshots_published == 0
            assert broadcaster.states == []

            # Startup failure must reap the child, cancel pipe/poll tasks, and
            # close the listener and publisher without requiring caller cleanup.
            tasks = (
                supervisor._process_wait_task,
                supervisor._stdout_task,
                supervisor._stderr_task,
                supervisor._output_limit_task,
            )
            assert all(task is not None and task.done() for task in tasks)
            assert supervisor._worker._task is None
            assert supervisor._ingress._server is None
            assert supervisor._output_directory is None
            arguments = json.loads(record.read_text(encoding="utf-8"))["argv"]
            batch_path = Path(
                arguments[arguments.index("--detection-batches-jsonl") + 1]
            )
            assert not batch_path.parent.exists()
        finally:
            await supervisor.close()

    asyncio.run(run())


def test_supervisor_terminates_child_when_batch_artifact_exceeds_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        maximum_detection_batch_output_bytes=10,
        output_limit_poll_interval_seconds=0.01,
        startup_timeout_seconds=1.0,
    )
    fake = _fake_executable(settings, tmp_path, mode="batch_limit")
    settings = RustReplaySettings(**{**settings.__dict__, "executable_path": fake})
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )

    async def run() -> None:
        with pytest.raises(RustReplayStartupError, match="byte limit"):
            await supervisor.start()
        assert supervisor.status.process_running is False

    asyncio.run(run())


def test_supervisor_checks_batch_limit_after_fast_successful_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(
        tmp_path,
        maximum_detection_batch_output_bytes=10,
        output_limit_poll_interval_seconds=10.0,
        startup_timeout_seconds=1.0,
    )
    fake = _fake_executable(settings, tmp_path, mode="batch_fast_exit")
    settings = RustReplaySettings(**{**settings.__dict__, "executable_path": fake})
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )

    async def run() -> None:
        with pytest.raises(RustReplayStartupError, match="byte limit"):
            await supervisor.start()
        status = supervisor.status
        assert status.process_return_code == 0
        assert status.process_status == "failed"
        assert status.source_health == "process_failed"

    asyncio.run(run())


def test_supervisor_close_terminates_a_running_replay_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    fake = _fake_executable(settings, tmp_path, mode="long")
    settings = RustReplaySettings(**{**settings.__dict__, "executable_path": fake})
    supervisor = RustReplaySupervisor(
        settings, SceneSnapshotStatePublisher(RecordingBroadcaster())
    )

    async def run() -> None:
        await supervisor.start()
        process = supervisor._process
        assert process is not None
        assert process.returncode is None
        await supervisor.close()
        assert process.returncode is not None

    asyncio.run(run())
