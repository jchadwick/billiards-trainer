"""Bounded client for the Rust-owned game-session metadata authority.

This adapter intentionally transports only session metadata. Ball positions,
source identity, ordering and freshness remain with the Rust scene publisher.
The one-shot command is short-lived and uses a separate revisioned store, so
session requests never wait on the camera or detector process.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

_PROTOCOL_VERSION = 1
_GAME_TYPES = {"practice", "8ball", "9ball", "straight"}
_MAX_U64 = 9_007_199_254_740_991
_MAX_SCORE = 4_294_967_295
_MAX_TIMEOUT_SECONDS = 60.0


class ConfigurationReader(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class RustGameSessionAuthoritySettings:
    executable: Path
    store_path: Path
    maximum_store_file_bytes: int
    maximum_request_bytes: int
    maximum_response_bytes: int
    lock_timeout_ms: int
    operation_timeout_seconds: float

    @classmethod
    def from_configuration(
        cls,
        configuration: ConfigurationReader,
        *,
        perception_executable: Path,
        project_root: Path,
    ) -> RustGameSessionAuthoritySettings:
        """Resolve validated settings and the companion Rust binary path."""
        section = "game_session"

        def value(name: str) -> Any:
            result = configuration.get(f"{section}.{name}")
            if result is None:
                raise ValueError(f"{section}.{name} must be configured")
            return result

        # The API's configuration surface is intentionally unauthenticated.
        # Keep the executable selector deployment-owned by deriving the helper
        # from the installed perception binary instead of accepting a stored
        # path that could be changed remotely and executed after restart.
        executable = perception_executable.with_name(
            "billiards-perception-game-session"
        )
        store_raw = value("store_path")
        if not isinstance(store_raw, str) or not store_raw.strip():
            raise ValueError("game_session.store_path must be a non-empty path")
        store_path = Path(store_raw).expanduser()
        if not store_path.is_absolute():
            store_path = project_root / store_path

        settings = cls(
            executable=executable.resolve(),
            store_path=store_path.resolve(),
            maximum_store_file_bytes=value("maximum_store_file_bytes"),
            maximum_request_bytes=value("maximum_request_bytes"),
            maximum_response_bytes=value("maximum_response_bytes"),
            lock_timeout_ms=value("lock_timeout_ms"),
            operation_timeout_seconds=value("operation_timeout_seconds"),
        )
        settings.validate(project_root=project_root.resolve())
        return settings

    def validate(self, *, project_root: Path | None = None) -> None:
        if not self.executable.is_absolute() or not self.executable.is_file():
            raise ValueError(
                "game-session authority executable must be an absolute file"
            )
        if not os.access(self.executable, os.X_OK):
            raise ValueError("game-session authority executable must be executable")
        if not self.store_path.is_absolute() or self.store_path.name in {"", ".", ".."}:
            raise ValueError("game-session store path must be an absolute file path")
        if project_root is not None:
            try:
                self.store_path.parent.resolve().relative_to(project_root)
            except ValueError as error:
                raise ValueError(
                    "game-session store path must remain under the configured project root"
                ) from error
        for name, number, minimum, maximum in (
            ("maximum_store_file_bytes", self.maximum_store_file_bytes, 1, 65_536),
            # Leave room for every valid v1 request, including reset fields.
            ("maximum_request_bytes", self.maximum_request_bytes, 128, 65_536),
            ("maximum_response_bytes", self.maximum_response_bytes, 256, 65_536),
            ("lock_timeout_ms", self.lock_timeout_ms, 1, 60_000),
        ):
            if type(number) is not int or not minimum <= number <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        timeout = self.operation_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0 < timeout <= _MAX_TIMEOUT_SECONDS
        ):
            raise ValueError("operation_timeout_seconds must be finite and in (0, 60]")


@dataclass(frozen=True, slots=True)
class RustGameSessionSnapshot:
    schema_version: int
    revision: int
    game_type: Literal["practice", "8ball", "9ball", "straight"]
    active_player: int
    scores: tuple[int, int]
    is_break: bool

    @classmethod
    def from_wire(cls, value: Any) -> RustGameSessionSnapshot:
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "revision",
            "game_type",
            "active_player",
            "scores",
            "is_break",
        }:
            raise ValueError("session snapshot has an incompatible field set")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("session snapshot schema_version must be 1")
        revision = value["revision"]
        if type(revision) is not int or not 0 <= revision <= _MAX_U64:
            raise ValueError("session revision is outside the safe integer range")
        game_type = value["game_type"]
        if not isinstance(game_type, str) or game_type not in _GAME_TYPES:
            raise ValueError("session game_type is unsupported")
        active_player = value["active_player"]
        if type(active_player) is not int or active_player not in (1, 2):
            raise ValueError("session active_player must be 1 or 2")
        scores = value["scores"]
        if (
            not isinstance(scores, list)
            or len(scores) != 2
            or any(
                type(score) is not int or not 0 <= score <= _MAX_SCORE
                for score in scores
            )
        ):
            raise ValueError("session scores must contain two non-negative integers")
        if type(value["is_break"]) is not bool:
            raise ValueError("session is_break must be a boolean")
        return cls(
            schema_version=1,
            revision=revision,
            game_type=game_type,  # type: ignore[arg-type]
            active_player=active_player,
            scores=(scores[0], scores[1]),
            is_break=value["is_break"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "game_type": self.game_type,
            "active_player": self.active_player,
            "scores": list(self.scores),
            "is_break": self.is_break,
        }


class RustGameSessionAuthorityError(RuntimeError):
    """Base error for a failed Rust game-session operation."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        current_revision: int | None = None,
        stderr: str = "",
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.current_revision = current_revision
        self.stderr = stderr
        self.exit_code = exit_code


class RustGameSessionAuthorityTimeout(RustGameSessionAuthorityError):
    """The Rust command exceeded its configured deadline."""


class RustGameSessionAuthorityOutputTooLarge(RustGameSessionAuthorityError):
    """The Rust command exceeded its configured stdout bound."""


class RustGameSessionAuthorityProtocolError(RustGameSessionAuthorityError):
    """The Rust command returned a malformed or incompatible response."""


class RustGameSessionAuthorityRejected(RustGameSessionAuthorityError):
    """Rust rejected the operation or could not access its store."""


class RustGameSessionAuthorityClient:
    """Issue bounded snapshot/reset commands to the Rust session store."""

    def __init__(self, settings: RustGameSessionAuthoritySettings) -> None:
        settings.validate()
        self._settings = settings

    def snapshot(self) -> RustGameSessionSnapshot:
        return self._call({"protocol_version": 1, "operation": "snapshot"}, "snapshot")

    def reset(
        self,
        *,
        expected_revision: int,
        game_type: Literal["practice", "8ball", "9ball", "straight"],
    ) -> RustGameSessionSnapshot:
        if type(expected_revision) is not int or not 0 <= expected_revision <= _MAX_U64:
            raise ValueError("expected_revision is outside the safe integer range")
        if game_type not in _GAME_TYPES:
            raise ValueError("game_type is unsupported")
        return self._call(
            {
                "protocol_version": 1,
                "operation": "reset",
                "expected_revision": expected_revision,
                "game_type": game_type,
            },
            "reset",
        )

    def _call(self, request: dict[str, Any], operation: str) -> RustGameSessionSnapshot:
        encoded = json.dumps(
            request, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        if len(encoded) > self._settings.maximum_request_bytes:
            raise ValueError("game-session request exceeds maximum_request_bytes")
        args = [
            str(self._settings.executable),
            "--store",
            str(self._settings.store_path),
            "--max-input-bytes",
            str(self._settings.maximum_request_bytes),
            "--max-output-bytes",
            str(self._settings.maximum_response_bytes),
            "--max-store-bytes",
            str(self._settings.maximum_store_file_bytes),
            "--lock-timeout-ms",
            str(self._settings.lock_timeout_ms),
        ]
        stdout, stderr, code = self._run(args, encoded)
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RustGameSessionAuthorityProtocolError(
                f"Rust game-session command returned invalid JSON: {error}",
                stderr=stderr,
                exit_code=code,
            ) from error
        if (
            not isinstance(response, dict)
            or type(response.get("protocol_version")) is not int
            or response["protocol_version"] != _PROTOCOL_VERSION
            or response.get("operation") != operation
            or type(response.get("ok")) is not bool
        ):
            raise RustGameSessionAuthorityProtocolError(
                "Rust game-session response has an incompatible envelope",
                stderr=stderr,
                exit_code=code,
            )
        expected_fields = (
            {"protocol_version", "ok", "operation", "session"}
            if response["ok"]
            else {"protocol_version", "ok", "operation", "error"}
        )
        if set(response) != expected_fields:
            raise RustGameSessionAuthorityProtocolError(
                "Rust game-session response has an unexpected field set",
                stderr=stderr,
                exit_code=code,
            )
        if response["ok"] is False:
            rejection = response.get("error")
            if (
                not isinstance(rejection, dict)
                or not isinstance(rejection.get("code"), str)
                or not isinstance(rejection.get("message"), str)
            ):
                raise RustGameSessionAuthorityProtocolError(
                    "Rust game-session rejection is missing its error details",
                    stderr=stderr,
                    exit_code=code,
                )
            current_revision = rejection.get("current_revision")
            if set(rejection) not in (
                {"code", "message"},
                {"code", "message", "current_revision"},
            ):
                raise RustGameSessionAuthorityProtocolError(
                    "Rust game-session rejection has unexpected error fields",
                    stderr=stderr,
                    exit_code=code,
                )
            if current_revision is not None and (
                type(current_revision) is not int
                or not 0 <= current_revision <= _MAX_U64
            ):
                raise RustGameSessionAuthorityProtocolError(
                    "Rust game-session current_revision is invalid",
                    stderr=stderr,
                    exit_code=code,
                )
            raise RustGameSessionAuthorityRejected(
                rejection["message"],
                code=rejection["code"],
                current_revision=current_revision,
                stderr=stderr,
                exit_code=code,
            )
        if code != 0:
            raise RustGameSessionAuthorityProtocolError(
                "Rust game-session success response had a non-zero exit status",
                stderr=stderr,
                exit_code=code,
            )
        try:
            snapshot = RustGameSessionSnapshot.from_wire(response.get("session"))
        except ValueError as parse_error:
            raise RustGameSessionAuthorityProtocolError(
                str(parse_error), stderr=stderr, exit_code=code
            ) from parse_error
        if operation == "reset" and (
            snapshot.revision != request["expected_revision"] + 1
            or snapshot.game_type != request["game_type"]
        ):
            raise RustGameSessionAuthorityProtocolError(
                "Rust game-session reset returned an unexpected revision or game type",
                stderr=stderr,
                exit_code=code,
            )
        return snapshot

    def _run(self, args: list[str], request_bytes: bytes) -> tuple[bytes, str, int]:
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as error:
            raise RustGameSessionAuthorityError(
                f"Could not start Rust game-session command: {error}"
            ) from error
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdin = process.stdin
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        stdout_exceeded = threading.Event()
        stderr_truncated = threading.Event()
        kill_lock = threading.Lock()
        child_killed = False

        def kill_child() -> None:
            nonlocal child_killed
            with kill_lock:
                if child_killed:
                    return
                child_killed = True
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - deployed server targets macOS and Linux
                    process.kill()
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass

        def read_bounded(
            stream: Any, buffer: bytearray, limit: int, fatal: bool
        ) -> None:
            while True:
                chunk = stream.read(16_384)
                if not chunk:
                    return
                available = limit - len(buffer)
                if available > 0:
                    buffer.extend(chunk[:available])
                if len(chunk) > max(available, 0):
                    if fatal:
                        stdout_exceeded.set()
                        kill_child()
                        return
                    stderr_truncated.set()

        def write_request() -> None:
            try:
                remaining = memoryview(request_bytes)
                while remaining:
                    written = stdin.write(remaining)
                    if written is None or written <= 0:
                        raise OSError("request pipe accepted no bytes")
                    remaining = remaining[written:]
                stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    stdin.close()
                except OSError:
                    pass

        stdout_thread = threading.Thread(
            target=read_bounded,
            args=(
                process.stdout,
                stdout_buffer,
                self._settings.maximum_response_bytes + 1,
                True,
            ),
            name="rust-game-session-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_bounded,
            args=(
                process.stderr,
                stderr_buffer,
                self._settings.maximum_response_bytes,
                False,
            ),
            name="rust-game-session-stderr",
            daemon=True,
        )
        input_thread = threading.Thread(
            target=write_request, name="rust-game-session-stdin", daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        input_thread.start()
        try:
            code = process.wait(timeout=float(self._settings.operation_timeout_seconds))
        except subprocess.TimeoutExpired as error:
            kill_child()
            process.wait()
            self._join(input_thread, stdout_thread, stderr_thread)
            raise RustGameSessionAuthorityTimeout(
                "Rust game-session command exceeded its operation deadline",
                stderr=stderr_buffer.decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            ) from error
        self._join(input_thread, stdout_thread, stderr_thread)
        if stdout_exceeded.is_set():
            raise RustGameSessionAuthorityOutputTooLarge(
                "Rust game-session command exceeded maximum_response_bytes",
                stderr=stderr_buffer.decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            )
        # stderr_truncated is intentionally not surfaced in the API response;
        # keep stderr bounded and preserve only the diagnostic tail.
        _ = stderr_truncated.is_set()
        stdout = bytes(stdout_buffer)
        if stdout.endswith(b"\n"):
            stdout = stdout[:-1]
        if len(stdout) > self._settings.maximum_response_bytes:
            raise RustGameSessionAuthorityOutputTooLarge(
                "Rust game-session command exceeded maximum_response_bytes",
                stderr=stderr_buffer.decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            )
        return stdout, stderr_buffer.decode("utf-8", errors="replace"), code

    @staticmethod
    def _join(*threads: threading.Thread) -> None:
        for thread in threads:
            thread.join(timeout=1.0)


__all__ = [
    "RustGameSessionAuthorityClient",
    "RustGameSessionAuthorityError",
    "RustGameSessionAuthorityOutputTooLarge",
    "RustGameSessionAuthorityProtocolError",
    "RustGameSessionAuthorityRejected",
    "RustGameSessionAuthoritySettings",
    "RustGameSessionAuthorityTimeout",
    "RustGameSessionSnapshot",
]
