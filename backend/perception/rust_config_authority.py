"""Bounded client for the Rust-owned unified configuration authority.

Each operation launches the explicitly configured one-shot Rust command. This
module deliberately contains no configuration defaults, validation, or disk
fallback: a missing or unhealthy authority is an error for the caller to handle.
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
from typing import Any, Mapping

_PROTOCOL_VERSION = 1
_DEFAULT_MAX_INPUT_BYTES = 1_048_576
_DEFAULT_MAX_OUTPUT_BYTES = 2_097_152
_DEFAULT_MAX_CONFIG_BYTES = 1_048_576
_DEFAULT_LOCK_TIMEOUT_MS = 2_000
_MAX_INPUT_BYTES = 16_777_216
_MAX_OUTPUT_BYTES = 33_554_432
_MAX_CONFIG_BYTES = 16_777_216
_MAX_LOCK_TIMEOUT_MS = 60_000
_MAX_OPERATION_TIMEOUT_SECONDS = 60.0


class RustConfigurationAuthorityError(RuntimeError):
    """Base class for failures communicating with the Rust authority."""

    def __init__(
        self,
        message: str,
        *,
        stderr: str = "",
        stderr_truncated: bool = False,
        response: dict[str, Any] | None = None,
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stderr_truncated = stderr_truncated
        self.response = response
        self.exit_code = exit_code


class RustConfigurationAuthoritySpawnError(RustConfigurationAuthorityError):
    """The configured executable could not be started."""


class RustConfigurationAuthorityTimeout(RustConfigurationAuthorityError):
    """The Rust command did not finish before the configured deadline."""


class RustConfigurationAuthorityOutputTooLarge(RustConfigurationAuthorityError):
    """The Rust command exceeded its configured stdout limit."""


class RustConfigurationAuthorityProtocolError(RustConfigurationAuthorityError):
    """The Rust command returned a malformed or incompatible response."""


class RustConfigurationAuthorityProcessError(RustConfigurationAuthorityError):
    """The command failed without returning a valid Rust error response."""


class RustConfigurationAuthorityRejected(RustConfigurationAuthorityError):
    """Rust processed the request and returned a structured domain error."""

    def __init__(self, response: dict[str, Any], **kwargs: Any) -> None:
        error = response["error"]
        self.code = error["code"]
        self.message = error["message"]
        super().__init__(self.message, response=response, **kwargs)


@dataclass(frozen=True, slots=True)
class RustConfigurationAuthorityLimits:
    """Bound one subprocess operation with a five-second, 60-second-max deadline."""

    maximum_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
    maximum_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    maximum_config_bytes: int = _DEFAULT_MAX_CONFIG_BYTES
    lock_timeout_ms: int = _DEFAULT_LOCK_TIMEOUT_MS
    operation_timeout_seconds: float = 5.0
    maximum_stderr_bytes: int = 8_192

    def validate(self) -> None:
        for name, value, minimum, maximum in (
            ("maximum_input_bytes", self.maximum_input_bytes, 1, _MAX_INPUT_BYTES),
            (
                "maximum_output_bytes",
                self.maximum_output_bytes,
                256,
                _MAX_OUTPUT_BYTES,
            ),
            ("maximum_config_bytes", self.maximum_config_bytes, 1, _MAX_CONFIG_BYTES),
            ("lock_timeout_ms", self.lock_timeout_ms, 0, _MAX_LOCK_TIMEOUT_MS),
            ("maximum_stderr_bytes", self.maximum_stderr_bytes, 1, 1_048_576),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        if (
            isinstance(self.operation_timeout_seconds, bool)
            or not isinstance(self.operation_timeout_seconds, (int, float))
            or not math.isfinite(self.operation_timeout_seconds)
            or self.operation_timeout_seconds <= 0
            or self.operation_timeout_seconds > _MAX_OPERATION_TIMEOUT_SECONDS
        ):
            raise ValueError("operation_timeout_seconds must be finite and in (0, 60]")


@dataclass(frozen=True, slots=True)
class RustConfigurationAuthorityResponse:
    """A validated protocol-v1 success response from Rust."""

    operation: str
    body: dict[str, Any]

    @property
    def revision(self) -> int | None:
        value = self.body.get("revision")
        return value if type(value) is int else None

    @property
    def configuration(self) -> dict[str, Any] | None:
        value = self.body.get("configuration")
        return value if isinstance(value, dict) else None


class RustConfigurationAuthorityClient:
    """Run one explicitly selected Rust configuration operation per call."""

    def __init__(
        self,
        executable: str | Path,
        config_path: str | Path,
        limits: RustConfigurationAuthorityLimits | None = None,
    ) -> None:
        self._executable = Path(executable)
        self._config_path = Path(config_path)
        self._limits = limits or RustConfigurationAuthorityLimits()
        self._limits.validate()
        if not self._executable.is_absolute():
            raise ValueError("Rust configuration executable must be an absolute path")
        if not self._config_path.is_absolute():
            raise ValueError("Rust configuration file must be an absolute path")

    def call(self, request: Mapping[str, Any]) -> RustConfigurationAuthorityResponse:
        """Send a single versioned operation and validate its response envelope."""
        if not isinstance(request, Mapping):
            raise TypeError("configuration request must be a mapping")
        request_value = dict(request)
        operation = request_value.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError(
                "configuration request operation must be a non-empty string"
            )
        if (
            type(request_value.get("protocol_version")) is not int
            or request_value["protocol_version"] != _PROTOCOL_VERSION
        ):
            raise ValueError("configuration request protocol_version must be 1")
        try:
            request_bytes = json.dumps(
                request_value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise ValueError(
                f"configuration request is not JSON serializable: {error}"
            ) from error
        if len(request_bytes) > self._limits.maximum_input_bytes:
            raise ValueError(
                "configuration request exceeds maximum_input_bytes "
                f"({len(request_bytes)} > {self._limits.maximum_input_bytes})"
            )

        args = [
            str(self._executable),
            "--config",
            str(self._config_path),
            "--max-input-bytes",
            str(self._limits.maximum_input_bytes),
            "--max-output-bytes",
            str(self._limits.maximum_output_bytes),
            "--max-config-bytes",
            str(self._limits.maximum_config_bytes),
            "--lock-timeout-ms",
            str(self._limits.lock_timeout_ms),
        ]
        stdout, stderr, stderr_truncated, exit_code = self._run(args, request_bytes)
        response = self._parse_response(
            stdout,
            operation=operation,
            stderr=stderr,
            stderr_truncated=stderr_truncated,
            exit_code=exit_code,
        )
        if response.get("ok") is False:
            raise RustConfigurationAuthorityRejected(
                response,
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                exit_code=exit_code,
            )
        if exit_code != 0:
            raise RustConfigurationAuthorityProcessError(
                "Rust configuration command returned success data with a non-zero exit code",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                response=response,
                exit_code=exit_code,
            )
        return RustConfigurationAuthorityResponse(operation=operation, body=response)

    def _run(
        self, args: list[str], request_bytes: bytes
    ) -> tuple[bytes, str, bool, int]:
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
            raise RustConfigurationAuthoritySpawnError(
                f"Could not start Rust configuration executable: {error}"
            ) from error
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        input_stream = process.stdin

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        stderr_truncated = threading.Event()
        stdout_exceeded = threading.Event()
        kill_lock = threading.Lock()
        killed = False

        def kill_child() -> None:
            nonlocal killed
            with kill_lock:
                if killed:
                    return
                killed = True
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - project targets macOS and Linux
                    process.kill()
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass

        def read_bounded(
            stream: Any, buffer: bytearray, limit: int, *, fatal: bool
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
                request_view = memoryview(request_bytes)
                offset = 0
                while offset < len(request_view):
                    written = input_stream.write(request_view[offset:])
                    if not written:
                        raise OSError("configuration request write made no progress")
                    offset += written
                input_stream.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    input_stream.close()
                except OSError:
                    pass

        stdout_thread = threading.Thread(
            target=read_bounded,
            args=(process.stdout, stdout_buffer, self._limits.maximum_output_bytes),
            kwargs={"fatal": True},
            name="rust-config-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_bounded,
            args=(process.stderr, stderr_buffer, self._limits.maximum_stderr_bytes),
            kwargs={"fatal": False},
            name="rust-config-stderr",
            daemon=True,
        )
        input_thread = threading.Thread(
            target=write_request, name="rust-config-stdin", daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        input_thread.start()

        try:
            exit_code = process.wait(
                timeout=float(self._limits.operation_timeout_seconds)
            )
        except subprocess.TimeoutExpired as error:
            kill_child()
            process.wait()
            self._join_io_threads(input_thread, stdout_thread, stderr_thread)
            raise RustConfigurationAuthorityTimeout(
                "Rust configuration command exceeded its operation deadline",
                stderr=stderr_buffer.decode("utf-8", errors="replace"),
                stderr_truncated=stderr_truncated.is_set(),
                exit_code=process.returncode,
            ) from error

        if stdout_exceeded.is_set():
            kill_child()
            if process.poll() is None:
                process.wait()
        self._join_io_threads(input_thread, stdout_thread, stderr_thread)
        if stdout_exceeded.is_set():
            raise RustConfigurationAuthorityOutputTooLarge(
                "Rust configuration command exceeded maximum_output_bytes",
                stderr=stderr_buffer.decode("utf-8", errors="replace"),
                stderr_truncated=stderr_truncated.is_set(),
                exit_code=process.returncode,
            )
        return (
            bytes(stdout_buffer),
            stderr_buffer.decode("utf-8", errors="replace"),
            stderr_truncated.is_set(),
            exit_code,
        )

    @staticmethod
    def _join_io_threads(*threads: threading.Thread) -> None:
        for thread in threads:
            thread.join(timeout=1.0)

    def _parse_response(
        self,
        stdout: bytes,
        *,
        operation: str,
        stderr: str,
        stderr_truncated: bool,
        exit_code: int,
    ) -> dict[str, Any]:
        try:
            decoded = stdout.decode("utf-8")
            response = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as decode_error:
            raise RustConfigurationAuthorityProtocolError(
                f"Rust configuration command returned invalid JSON: {decode_error}",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                exit_code=exit_code,
            ) from decode_error
        if not isinstance(response, dict):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration response must be a JSON object",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                exit_code=exit_code,
            )
        if (
            type(response.get("protocol_version")) is not int
            or response["protocol_version"] != _PROTOCOL_VERSION
        ):
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration response has an unsupported protocol_version",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                response=response,
                exit_code=exit_code,
            )
        if type(response.get("ok")) is not bool:
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration response must contain a boolean ok field",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                response=response,
                exit_code=exit_code,
            )
        response_operation = response.get("operation")
        # The Rust command can fail while constructing the authority, before it
        # parses stdin and therefore before it knows the requested operation.
        # Such bootstrap errors still use the versioned error envelope.
        if response_operation is not None and response_operation != operation:
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration response operation does not match the request",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                response=response,
                exit_code=exit_code,
            )
        if response["ok"] is True and response_operation is None:
            raise RustConfigurationAuthorityProtocolError(
                "Rust configuration success response is missing its operation",
                stderr=stderr,
                stderr_truncated=stderr_truncated,
                response=response,
                exit_code=exit_code,
            )
        if response["ok"] is False:
            response_error = response.get("error")
            if (
                not isinstance(response_error, dict)
                or not isinstance(response_error.get("code"), str)
                or not isinstance(response_error.get("message"), str)
            ):
                raise RustConfigurationAuthorityProtocolError(
                    "Rust configuration error response is missing its code or message",
                    stderr=stderr,
                    stderr_truncated=stderr_truncated,
                    response=response,
                    exit_code=exit_code,
                )
        return response


__all__ = [
    "RustConfigurationAuthorityClient",
    "RustConfigurationAuthorityError",
    "RustConfigurationAuthorityLimits",
    "RustConfigurationAuthorityOutputTooLarge",
    "RustConfigurationAuthorityProcessError",
    "RustConfigurationAuthorityProtocolError",
    "RustConfigurationAuthorityRejected",
    "RustConfigurationAuthorityResponse",
    "RustConfigurationAuthoritySpawnError",
    "RustConfigurationAuthorityTimeout",
]
