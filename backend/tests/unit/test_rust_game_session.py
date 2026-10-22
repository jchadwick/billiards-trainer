"""Contracts for the bounded Python-to-Rust game-session adapter."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from backend.perception.rust_game_session import (
    RustGameSessionAuthorityClient,
    RustGameSessionAuthorityOutputTooLarge,
    RustGameSessionAuthoritySettings,
    RustGameSessionAuthorityTimeout,
    RustGameSessionSnapshot,
)

if TYPE_CHECKING:
    from pathlib import Path


def _settings(
    executable: Path,
    root: Path,
    *,
    maximum_response_bytes: int = 8192,
    operation_timeout_seconds: float = 2.0,
) -> RustGameSessionAuthoritySettings:
    executable.chmod(executable.stat().st_mode | 0o111)
    return RustGameSessionAuthoritySettings(
        executable=executable,
        store_path=root / "backend" / "calibration_data" / "game_session.json",
        maximum_store_file_bytes=4096,
        maximum_request_bytes=4096,
        maximum_response_bytes=maximum_response_bytes,
        lock_timeout_ms=100,
        operation_timeout_seconds=operation_timeout_seconds,
    )


def _executable(root: Path, body: str) -> Path:
    script = root / "game-session-authority"
    script.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8"
    )
    script.chmod(0o755)
    return script


def test_snapshot_wire_contract_is_strict_and_rejects_unrepresentable_scores() -> None:
    snapshot = RustGameSessionSnapshot.from_wire(
        {
            "schema_version": 1,
            "revision": 2,
            "game_type": "8ball",
            "active_player": 2,
            "scores": [3, 9],
            "is_break": False,
        }
    )
    assert snapshot.to_dict()["scores"] == [3, 9]

    invalid = snapshot.to_dict()
    invalid["scores"] = [0, 4_294_967_296]
    with pytest.raises(ValueError, match="scores"):
        RustGameSessionSnapshot.from_wire(invalid)

    invalid = snapshot.to_dict()
    invalid["extra"] = True
    with pytest.raises(ValueError, match="field set"):
        RustGameSessionSnapshot.from_wire(invalid)


def test_client_uses_bounded_line_protocol_for_snapshot_and_compare_and_swap_reset(
    tmp_path: Path,
) -> None:
    executable = _executable(
        tmp_path,
        """
        import json
        import sys

        request = json.load(sys.stdin)
        operation = request["operation"]
        session = {
            "schema_version": 1,
            "revision": 0 if operation == "snapshot" else request["expected_revision"] + 1,
            "game_type": "practice" if operation == "snapshot" else request["game_type"],
            "active_player": 1,
            "scores": [0, 0],
            "is_break": True,
        }
        print(json.dumps({"protocol_version": 1, "ok": True, "operation": operation, "session": session}))
        """,
    )
    client = RustGameSessionAuthorityClient(_settings(executable, tmp_path))

    assert client.snapshot().revision == 0
    result = client.reset(expected_revision=0, game_type="9ball")
    assert result.revision == 1
    assert result.game_type == "9ball"


def test_client_stops_an_authority_that_exceeds_its_response_limit(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path, "import sys\nsys.stdout.write('x' * 2000)\n")
    client = RustGameSessionAuthorityClient(
        _settings(executable, tmp_path, maximum_response_bytes=256)
    )
    with pytest.raises(RustGameSessionAuthorityOutputTooLarge):
        client.snapshot()


def test_client_kills_authority_after_configured_operation_deadline(
    tmp_path: Path,
) -> None:
    executable = _executable(
        tmp_path,
        "import time\ntime.sleep(2)\n",
    )
    client = RustGameSessionAuthorityClient(
        _settings(executable, tmp_path, operation_timeout_seconds=0.1)
    )
    with pytest.raises(RustGameSessionAuthorityTimeout):
        client.snapshot()


def test_configuration_resolves_trusted_sibling_binary_and_project_relative_store(
    tmp_path: Path,
) -> None:
    project_binary = tmp_path / "bin" / "billiards-perception-live"
    project_binary.parent.mkdir()
    project_binary.write_text("binary", encoding="utf-8")
    project_binary.chmod(0o755)
    authority_binary = _executable(
        tmp_path / "bin",
        "import sys\nsys.stdout.write('')\n",
    )
    authority_binary.rename(
        project_binary.with_name("billiards-perception-game-session")
    )
    untrusted_binary = _executable(tmp_path, "raise SystemExit(99)\n")

    class Configuration:
        values: dict[str, Any] = {
            # Stale values from older configs must not choose a command.
            "game_session.executable_path": str(untrusted_binary),
            "game_session.store_path": "backend/calibration_data/game_session.json",
            "game_session.maximum_store_file_bytes": 4096,
            "game_session.maximum_request_bytes": 4096,
            "game_session.maximum_response_bytes": 8192,
            "game_session.lock_timeout_ms": 2000,
            "game_session.operation_timeout_seconds": 5.0,
        }

        def get(self, key: str, default: Any = None) -> Any:
            return self.values.get(key, default)

    settings = RustGameSessionAuthoritySettings.from_configuration(
        Configuration(),
        perception_executable=project_binary,
        project_root=tmp_path,
    )
    assert settings.executable == project_binary.with_name(
        "billiards-perception-game-session"
    )
    assert settings.executable != untrusted_binary.resolve()
    assert settings.store_path == (
        tmp_path / "backend/calibration_data/game_session.json"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"maximum_response_bytes": 255},
        {"maximum_request_bytes": 127},
        {"maximum_request_bytes": 65_537},
        {"lock_timeout_ms": True},
        {"operation_timeout_seconds": float("nan")},
    ],
)
def test_settings_reject_unbounded_or_ambiguous_values(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    executable = _executable(tmp_path, "pass\n")
    values: dict[str, Any] = {
        "maximum_store_file_bytes": 4096,
        "maximum_request_bytes": 4096,
        "maximum_response_bytes": 8192,
        "lock_timeout_ms": 100,
        "operation_timeout_seconds": 2.0,
    }
    values.update(override)
    settings = RustGameSessionAuthoritySettings(
        executable=executable,
        store_path=tmp_path / "game_session.json",
        **values,
    )
    with pytest.raises(ValueError):
        settings.validate(project_root=tmp_path)
