"""Subprocess and protocol contracts for the Rust configuration authority."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.perception import rust_config_authority
from backend.perception.rust_config_authority import (
    RustConfigurationAuthorityClient,
    RustConfigurationAuthorityLimits,
    RustConfigurationAuthorityOutputTooLarge,
    RustConfigurationAuthorityProcessError,
    RustConfigurationAuthorityProtocolError,
    RustConfigurationAuthorityRejected,
    RustConfigurationAuthoritySpawnError,
    RustConfigurationAuthorityTimeout,
)


def _write_executable(tmp_path: Path, source: str) -> Path:
    """Create a directly executable Python fixture using this test interpreter."""
    executable = tmp_path / "fake-rust-config"
    executable.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    executable.chmod(0o700)
    return executable


def _response(
    *,
    operation: str = "snapshot",
    ok: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "protocol_version": 1,
        "ok": ok,
        "operation": operation,
    }
    if ok:
        body.update({"revision": 7, "configuration": {"metadata": {"revision": 7}}})
    else:
        body["error"] = {"code": "stale_revision", "message": "revision changed"}
    if extra:
        body.update(extra)
    return body


def _client(
    executable: Path,
    *,
    limits: RustConfigurationAuthorityLimits | None = None,
    config_path: Path | None = None,
) -> RustConfigurationAuthorityClient:
    return RustConfigurationAuthorityClient(
        executable,
        config_path or Path("/tmp/billiards-config.json"),
        limits,
    )


def _request(**overrides: Any) -> dict[str, Any]:
    return {"protocol_version": 1, "operation": "snapshot", **overrides}


def _track_subprocesses(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    original_popen = rust_config_authority.subprocess.Popen
    processes: list[Any] = []

    def tracked_popen(*args: Any, **kwargs: Any) -> Any:
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(rust_config_authority.subprocess, "Popen", tracked_popen)
    return processes


def test_call_passes_explicit_paths_and_configured_bounds_and_returns_snapshot(
    tmp_path: Path,
) -> None:
    output = json.dumps(_response(), separators=(",", ":"))
    config_path = tmp_path / "active-config.json"
    expected_args = [
        "--config",
        str(config_path),
        "--max-input-bytes",
        "4096",
        "--max-output-bytes",
        "8192",
        "--max-config-bytes",
        "2048",
        "--lock-timeout-ms",
        "350",
    ]
    executable = _write_executable(
        tmp_path,
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request == {'protocol_version': 1, 'operation': 'snapshot'}\n"
        "assert sys.argv[1:] == " + repr(expected_args) + "\n"
        "print(" + repr(output) + ")\n",
    )
    limits = RustConfigurationAuthorityLimits(
        maximum_input_bytes=4096,
        maximum_output_bytes=8192,
        maximum_config_bytes=2048,
        lock_timeout_ms=350,
    )
    result = _client(executable, limits=limits, config_path=config_path).call(
        _request()
    )

    assert result.operation == "snapshot"
    assert result.revision == 7
    assert result.configuration == {"metadata": {"revision": 7}}


def test_call_raises_structured_rust_error_and_keeps_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    error_response = {
        "protocol_version": 1,
        "ok": False,
        "operation": "snapshot",
        "error": {"code": "stale_revision", "message": "revision changed"},
        "current_revision": 11,
    }
    executable = _write_executable(
        tmp_path,
        "import json, sys\n"
        "sys.stdin.read()\n"
        "sys.stderr.write('diagnostics-are-long')\n"
        "print(" + repr(json.dumps(error_response)) + ")\n"
        "raise SystemExit(1)\n",
    )

    with pytest.raises(RustConfigurationAuthorityRejected) as raised:
        _client(
            executable,
            limits=RustConfigurationAuthorityLimits(maximum_stderr_bytes=8),
        ).call(_request())

    assert raised.value.code == "stale_revision"
    assert raised.value.response == error_response
    assert raised.value.exit_code == 1
    assert raised.value.stderr == "diagnost"
    assert raised.value.stderr_truncated is True


def test_allows_versioned_bootstrap_error_without_operation_field(
    tmp_path: Path,
) -> None:
    bootstrap_error = {
        "protocol_version": 1,
        "ok": False,
        "error": {"code": "invalid_limits", "message": "invalid command settings"},
    }
    executable = _write_executable(
        tmp_path,
        "import sys\n"
        "sys.stdin.read()\n"
        "print(" + repr(json.dumps(bootstrap_error)) + ")\n"
        "raise SystemExit(1)\n",
    )

    with pytest.raises(RustConfigurationAuthorityRejected) as raised:
        _client(executable).call(_request())

    assert raised.value.code == "invalid_limits"
    assert raised.value.response == bootstrap_error


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("not-json", "invalid JSON"),
        (json.dumps([1, 2]), "JSON object"),
        (
            json.dumps({"protocol_version": 2, "ok": True, "operation": "snapshot"}),
            "protocol_version",
        ),
        (
            json.dumps({"protocol_version": 1, "ok": 1, "operation": "snapshot"}),
            "boolean ok",
        ),
        (
            json.dumps({"protocol_version": 1, "ok": True, "operation": "defaults"}),
            "does not match",
        ),
        (
            json.dumps({"protocol_version": 1, "ok": False, "operation": "snapshot"}),
            "missing its code or message",
        ),
    ],
)
def test_rejects_malformed_or_incompatible_responses(
    tmp_path: Path, response: str, message: str
) -> None:
    executable = _write_executable(
        tmp_path,
        "import sys\n"
        "sys.stdin.read()\n"
        "sys.stdout.write(" + repr(response) + ")\n",
    )

    with pytest.raises(RustConfigurationAuthorityProtocolError, match=message):
        _client(executable).call(_request())


def test_rejects_success_response_with_nonzero_exit_code(tmp_path: Path) -> None:
    executable = _write_executable(
        tmp_path,
        "import sys\n"
        "sys.stdin.read()\n"
        "print(" + repr(json.dumps(_response())) + ")\n"
        "raise SystemExit(6)\n",
    )

    with pytest.raises(RustConfigurationAuthorityProcessError) as raised:
        _client(executable).call(_request())

    assert raised.value.exit_code == 6
    assert raised.value.response == _response()


def test_missing_executable_is_a_typed_spawn_failure(tmp_path: Path) -> None:
    missing = tmp_path / "not-installed"

    with pytest.raises(RustConfigurationAuthoritySpawnError):
        _client(missing).call(_request())


def test_timeout_kills_reaps_and_reports_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = _write_executable(
        tmp_path,
        "import sys, time\n"
        "sys.stdin.read()\n"
        "sys.stderr.write('still-running')\n"
        "sys.stderr.flush()\n"
        "time.sleep(5)\n",
    )

    processes = _track_subprocesses(monkeypatch)
    with pytest.raises(RustConfigurationAuthorityTimeout) as raised:
        _client(
            executable,
            limits=RustConfigurationAuthorityLimits(operation_timeout_seconds=0.5),
        ).call(_request())

    assert "still-running" in raised.value.stderr
    assert raised.value.exit_code is not None
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_large_bounded_request_is_fully_written_to_child_stdin(tmp_path: Path) -> None:
    response = json.dumps(_response(), separators=(",", ":"))
    executable = _write_executable(
        tmp_path,
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert len(request['data']['large']) == 180000\n"
        "print(" + repr(response) + ")\n",
    )

    result = _client(
        executable,
        limits=RustConfigurationAuthorityLimits(maximum_input_bytes=262_144),
    ).call(_request(data={"large": "x" * 180_000}))

    assert result.revision == 7


def test_stdout_is_capped_and_oversized_child_is_killed_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = _write_executable(
        tmp_path,
        "import sys\n"
        "sys.stdin.read()\n"
        "sys.stdout.write('x' * 10000000)\n"
        "sys.stdout.flush()\n",
    )

    processes = _track_subprocesses(monkeypatch)
    with pytest.raises(RustConfigurationAuthorityOutputTooLarge):
        _client(
            executable,
            limits=RustConfigurationAuthorityLimits(maximum_output_bytes=256),
        ).call(_request())
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_invalid_or_relative_configuration_paths_are_rejected(tmp_path: Path) -> None:
    executable = tmp_path / "fake"
    executable.touch()

    with pytest.raises(ValueError, match="executable must be an absolute"):
        RustConfigurationAuthorityClient("relative-command", "/tmp/config.json")
    with pytest.raises(ValueError, match="file must be an absolute"):
        RustConfigurationAuthorityClient(executable, "relative-config.json")


def test_input_size_and_invalid_json_values_fail_before_process_start(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "must-not-be-started"
    client = _client(
        missing,
        limits=RustConfigurationAuthorityLimits(maximum_input_bytes=256),
    )

    with pytest.raises(ValueError, match="exceeds maximum_input_bytes"):
        client.call(_request(data={"large": "x" * 300}))
    with pytest.raises(ValueError, match="not JSON serializable"):
        client.call(_request(data={"value": float("nan")}))


@pytest.mark.parametrize(
    "limits",
    [
        RustConfigurationAuthorityLimits(maximum_output_bytes=255),
        RustConfigurationAuthorityLimits(maximum_input_bytes=16_777_217),
        RustConfigurationAuthorityLimits(lock_timeout_ms=60_001),
        RustConfigurationAuthorityLimits(operation_timeout_seconds=float("inf")),
        RustConfigurationAuthorityLimits(operation_timeout_seconds=60.1),
        RustConfigurationAuthorityLimits(maximum_stderr_bytes=0),
    ],
)
def test_rejects_out_of_range_limits(
    tmp_path: Path, limits: RustConfigurationAuthorityLimits
) -> None:
    executable = tmp_path / "unused"
    executable.touch()
    with pytest.raises(ValueError):
        RustConfigurationAuthorityClient(executable, tmp_path / "config.json", limits)


def test_defaults_match_rust_cli_documented_bounds() -> None:
    limits = RustConfigurationAuthorityLimits()

    assert limits.maximum_input_bytes == 1_048_576
    assert limits.maximum_output_bytes == 2_097_152
    assert limits.maximum_config_bytes == 1_048_576
    assert limits.lock_timeout_ms == 2_000
