"""Bounded client for the Rust-owned current-frame preview protocol.

The gateway forwards the encoded JPEG and its capture descriptor without
decoding pixels or choosing calibration values. All network work is local to
the supervised Rust child and has explicit size and time limits.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUEST_V1 = b"BTPV0001"
_CALIBRATION_READ_REQUEST_V1 = b"BTCR0001"
_CALIBRATION_APPLY_REQUEST_V1 = b"BTCA0001"
_READY_PREFIX = b"frame_preview_ready="
_MAX_U32 = (1 << 32) - 1
_MAX_PREVIEW_JPEG_BYTES = 16 * 1024 * 1024
_MAX_PREVIEW_ENCODERS = 4
_MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
_READY_FIELDS = {
    "schema_version",
    "address",
    "engine_instance_id",
    "capture_session_id",
}
_HEADER_FIELDS = {
    "schema_version",
    "status",
    "descriptor",
    "jpeg_bytes",
    "error",
}
_DESCRIPTOR_FIELDS = {
    "schema_version",
    "engine_instance_id",
    "capture_session_id",
    "source_frame_id",
    "width_px",
    "height_px",
    "pixel_format",
    "image_space",
    "camera_profile_id",
    "configuration_revision",
    "calibration_revision",
    "rectification_revision",
    "source_timestamp",
    "source_timestamp_semantics",
    "host_read_completed",
}
_CALIBRATION_CONTROL_STATUSES = {
    "ok",
    "invalid_request",
    "stale_context",
    "stale_revision",
    "busy",
    "store_error",
    "unsupported",
}


class FramePreviewError(RuntimeError):
    """Base error for a missing or invalid Rust frame-preview response."""


class FramePreviewUnavailable(FramePreviewError):
    """No fresh frame is currently available from the Rust capture owner."""


class FramePreviewBusy(FramePreviewError):
    """Every configured Rust preview encoder is currently occupied."""


class FramePreviewTransportError(FramePreviewError):
    """The local Rust preview service could not be reached in its deadline."""


class FramePreviewProtocolError(FramePreviewError):
    """The Rust child returned a malformed, inconsistent, or oversized record."""


class FramePreviewRequestTooLarge(FramePreviewError):
    """A calibration-control request exceeds the configured body limit."""


@dataclass(frozen=True, slots=True)
class FramePreviewEndpoint:
    """A loopback endpoint bound to one supervised Rust engine and session."""

    host: str
    port: int
    engine_instance_id: str
    capture_session_id: str


@dataclass(frozen=True, slots=True)
class FramePreviewCalibrationControlSettings:
    """Bounded Rust-owned calibration store configuration for the child."""

    store_path: Path
    store_lock_timeout_ms: int
    maximum_store_file_bytes: int
    maximum_request_bytes: int

    def validate(self, *, project_root: Path | None = None) -> None:
        """Require a bounded configuration path contained by the project root."""
        if not isinstance(self.store_path, Path) or not self.store_path.is_absolute():
            raise ValueError("calibration store path must be an absolute Path")
        resolved_path = self.store_path.resolve()
        if project_root is not None and not resolved_path.is_relative_to(
            project_root.resolve()
        ):
            raise ValueError("calibration store path must be inside the project root")
        if resolved_path.is_dir():
            raise ValueError("calibration store path must name a file, not a directory")
        for name, value, minimum, maximum in (
            ("store_lock_timeout_ms", self.store_lock_timeout_ms, 1, 60_000),
            ("maximum_store_file_bytes", self.maximum_store_file_bytes, 1, 16_777_216),
            ("maximum_request_bytes", self.maximum_request_bytes, 1_024, 1_048_576),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"calibration {name} must be an integer in [{minimum}, {maximum}]"
                )


@dataclass(frozen=True, slots=True)
class FramePreviewProcessSettings:
    """Validated Rust-owned listener limits resolved from application config."""

    bind_address: str
    port: int
    maximum_header_bytes: int
    maximum_jpeg_bytes: int
    maximum_concurrent_encodes: int
    jpeg_quality: int
    request_timeout_ms: int
    write_timeout_ms: int
    calibration_control: FramePreviewCalibrationControlSettings | None = None

    def validate(self, *, project_root: Path | None = None) -> None:
        try:
            address = ipaddress.ip_address(self.bind_address)
        except ValueError as error:
            raise ValueError(
                "frame-preview bind address must be a numeric IP"
            ) from error
        if not address.is_loopback:
            raise ValueError("frame-preview listener must bind to a loopback address")
        for name, value, minimum, maximum in (
            ("port", self.port, 0, 65_535),
            ("maximum_header_bytes", self.maximum_header_bytes, 128, 1_048_576),
            (
                "maximum_jpeg_bytes",
                self.maximum_jpeg_bytes,
                1,
                _MAX_PREVIEW_JPEG_BYTES,
            ),
            (
                "maximum_concurrent_encodes",
                self.maximum_concurrent_encodes,
                1,
                _MAX_PREVIEW_ENCODERS,
            ),
            ("jpeg_quality", self.jpeg_quality, 1, 100),
            ("request_timeout_ms", self.request_timeout_ms, 1, 60_000),
            ("write_timeout_ms", self.write_timeout_ms, 1, 60_000),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(
                    f"frame-preview {name} must be an integer in [{minimum}, {maximum}]"
                )
        if self.calibration_control is not None and self.maximum_header_bytes < 4096:
            raise ValueError(
                "frame-preview maximum_header_bytes must be at least 4096 when "
                "calibration control is enabled"
            )
        if self.calibration_control is not None:
            if not isinstance(
                self.calibration_control, FramePreviewCalibrationControlSettings
            ):
                raise ValueError(
                    "calibration_control must be a validated calibration settings object"
                )
            self.calibration_control.validate(project_root=project_root)

    @property
    def socket_address(self) -> str:
        """Return the Rust CLI's IPv4 or bracketed IPv6 socket syntax."""
        address = ipaddress.ip_address(self.bind_address)
        if address.version == 6:
            return f"[{address}]:{self.port}"
        return f"{address}:{self.port}"

    @property
    def client_limits(self) -> FramePreviewClientLimits:
        return FramePreviewClientLimits(
            maximum_header_bytes=self.maximum_header_bytes,
            maximum_jpeg_bytes=self.maximum_jpeg_bytes,
            request_timeout_ms=self.request_timeout_ms,
            write_timeout_ms=self.write_timeout_ms,
        )


@dataclass(frozen=True, slots=True)
class FramePreviewClientLimits:
    """Validated limits shared by the Rust encoder and the gateway reader."""

    maximum_header_bytes: int
    maximum_jpeg_bytes: int
    request_timeout_ms: int
    write_timeout_ms: int

    def validate(self) -> None:
        for name, value in (
            ("maximum_header_bytes", self.maximum_header_bytes),
            ("maximum_jpeg_bytes", self.maximum_jpeg_bytes),
            ("request_timeout_ms", self.request_timeout_ms),
            ("write_timeout_ms", self.write_timeout_ms),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_header_bytes > _MAX_U32:
            raise ValueError("maximum_header_bytes must fit the protocol-v1 u32")
        if self.maximum_header_bytes > 1_048_576:
            raise ValueError("maximum_header_bytes exceeds the bounded protocol limit")
        if self.maximum_jpeg_bytes > _MAX_PREVIEW_JPEG_BYTES:
            raise ValueError("maximum_jpeg_bytes exceeds the bounded protocol limit")
        if self.request_timeout_ms > 60_000 or self.write_timeout_ms > 60_000:
            raise ValueError("frame-preview timeouts exceed the bounded protocol limit")


@dataclass(frozen=True, slots=True)
class FramePreviewImage:
    """The exact JPEG bytes and descriptor returned together by Rust."""

    jpeg: bytes
    descriptor: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CalibrationControlResponse:
    """Opaque Rust-owned store result relayed by the Python service."""

    status: str
    snapshot: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned public protocol record without interpretation."""
        return {
            "schema_version": 1,
            "status": self.status,
            "snapshot": self.snapshot,
            "error": self.error,
        }


def parse_ready_record(
    line: bytes,
    *,
    expected_engine_instance_id: str,
    expected_capture_session_id: str,
    maximum_record_bytes: int,
) -> FramePreviewEndpoint:
    """Validate one `frame_preview_ready=` line from the supervised process."""
    if not isinstance(line, bytes) or len(line) > maximum_record_bytes:
        raise FramePreviewProtocolError("frame-preview readiness record is oversized")
    if not line.startswith(_READY_PREFIX):
        raise FramePreviewProtocolError("frame-preview readiness prefix is invalid")
    try:
        record = json.loads(line[len(_READY_PREFIX) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FramePreviewProtocolError(
            "frame-preview readiness record is not valid UTF-8 JSON"
        ) from error
    if not isinstance(record, dict) or set(record) != _READY_FIELDS:
        raise FramePreviewProtocolError("frame-preview readiness fields are invalid")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise FramePreviewProtocolError(
            "frame-preview readiness version is unsupported"
        )
    if record["engine_instance_id"] != expected_engine_instance_id:
        raise FramePreviewProtocolError("frame-preview engine identity does not match")
    if record["capture_session_id"] != expected_capture_session_id:
        raise FramePreviewProtocolError("frame-preview capture session does not match")
    if not isinstance(record["address"], str):
        raise FramePreviewProtocolError("frame-preview address must be a string")
    host, port = _parse_loopback_address(record["address"])
    return FramePreviewEndpoint(
        host=host,
        port=port,
        engine_instance_id=expected_engine_instance_id,
        capture_session_id=expected_capture_session_id,
    )


def _parse_loopback_address(value: str) -> tuple[str, int]:
    """Parse a numeric IPv4/IPv6 socket address and require loopback only."""
    if value.startswith("["):
        close = value.find("]")
        if close < 0 or value[close + 1 : close + 2] != ":":
            raise FramePreviewProtocolError("frame-preview IPv6 address is invalid")
        host = value[1:close]
        raw_port = value[close + 2 :]
    else:
        host, separator, raw_port = value.rpartition(":")
        if not separator or ":" in host:
            raise FramePreviewProtocolError("frame-preview address is invalid")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise FramePreviewProtocolError(
            "frame-preview address must use a numeric loopback IP"
        ) from error
    if not address.is_loopback:
        raise FramePreviewProtocolError("frame-preview address is not loopback")
    try:
        port = int(raw_port, 10)
    except ValueError as error:
        raise FramePreviewProtocolError("frame-preview port is invalid") from error
    if not 1 <= port <= 65_535:
        raise FramePreviewProtocolError("frame-preview port is outside its valid range")
    return str(address), port


async def fetch_frame_preview(
    endpoint: FramePreviewEndpoint,
    limits: FramePreviewClientLimits,
) -> FramePreviewImage:
    """Read one bounded JPEG+descriptor response from the Rust loopback service."""
    limits.validate()
    timeout_seconds = (limits.request_timeout_ms + limits.write_timeout_ms) / 1000.0
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    async def within_deadline(awaitable: Any) -> Any:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(awaitable, timeout=remaining)

    try:
        reader, writer = await within_deadline(
            asyncio.open_connection(
                endpoint.host,
                endpoint.port,
                limit=limits.maximum_header_bytes + 4,
            )
        )
    except (OSError, TimeoutError, asyncio.TimeoutError) as error:
        raise FramePreviewTransportError(
            "Rust frame-preview listener did not accept a local connection"
        ) from error

    try:
        writer.write(_REQUEST_V1)
        try:
            await within_deadline(writer.drain())
            prefix = await within_deadline(reader.readexactly(4))
            header_length = int.from_bytes(prefix, "big")
            if not 1 <= header_length <= limits.maximum_header_bytes:
                raise FramePreviewProtocolError(
                    "Rust frame-preview header length exceeds its configured limit"
                )
            raw_header = await within_deadline(reader.readexactly(header_length))
        except (
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
        ) as error:
            raise FramePreviewTransportError(
                "Rust frame-preview response missed its local deadline"
            ) from error

        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FramePreviewProtocolError(
                "Rust frame-preview header is not valid UTF-8 JSON"
            ) from error
        _validate_response_header(header)
        status = header["status"]
        if status == "unavailable":
            raise FramePreviewUnavailable("Rust has no fresh captured frame")
        if status == "busy":
            raise FramePreviewBusy("Rust frame-preview encoder capacity is full")
        if status != "ok":
            raise FramePreviewProtocolError(
                f"Rust frame-preview request failed with status {status!r}"
            )

        descriptor = header["descriptor"]
        _validate_descriptor(
            descriptor,
            expected_engine_instance_id=endpoint.engine_instance_id,
            expected_capture_session_id=endpoint.capture_session_id,
        )
        jpeg_length = header["jpeg_bytes"]
        if (
            type(jpeg_length) is not int
            or not 1 <= jpeg_length <= limits.maximum_jpeg_bytes
        ):
            raise FramePreviewProtocolError(
                "Rust frame-preview JPEG length exceeds its configured limit"
            )
        try:
            jpeg = await within_deadline(reader.readexactly(jpeg_length))
        except (
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
        ) as error:
            raise FramePreviewTransportError(
                "Rust frame-preview JPEG was incomplete or late"
            ) from error
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise FramePreviewProtocolError("Rust frame-preview payload is not a JPEG")
        return FramePreviewImage(jpeg=jpeg, descriptor=descriptor)
    finally:
        writer.close()
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.wait_for(writer.wait_closed(), timeout=remaining)
        except (OSError, TimeoutError, asyncio.TimeoutError):
            pass


async def read_calibration_store(
    endpoint: FramePreviewEndpoint,
    limits: FramePreviewClientLimits,
) -> CalibrationControlResponse:
    """Read the Rust-owned calibration snapshot over the shared loopback socket."""
    return await _calibration_control_request(
        endpoint,
        limits,
        magic=_CALIBRATION_READ_REQUEST_V1,
        request_body=None,
        maximum_request_bytes=0,
    )


async def submit_calibration_request(
    endpoint: FramePreviewEndpoint,
    limits: FramePreviewClientLimits,
    request_body: bytes,
    *,
    maximum_request_bytes: int,
) -> CalibrationControlResponse:
    """Relay the exact JSON body to Rust after enforcing its configured bound."""
    if not isinstance(request_body, bytes):
        raise TypeError("calibration request body must be bytes")
    if type(maximum_request_bytes) is not int or maximum_request_bytes <= 0:
        raise ValueError("maximum_request_bytes must be a positive integer")
    if len(request_body) > maximum_request_bytes or len(request_body) > _MAX_U32:
        raise FramePreviewRequestTooLarge(
            "calibration request exceeds its configured byte limit"
        )
    return await _calibration_control_request(
        endpoint,
        limits,
        magic=_CALIBRATION_APPLY_REQUEST_V1,
        request_body=request_body,
        maximum_request_bytes=maximum_request_bytes,
    )


async def _calibration_control_request(
    endpoint: FramePreviewEndpoint,
    limits: FramePreviewClientLimits,
    *,
    magic: bytes,
    request_body: bytes | None,
    maximum_request_bytes: int,
) -> CalibrationControlResponse:
    """Exchange one size-bounded calibration command and validate its envelope."""
    limits.validate()
    timeout_seconds = (limits.request_timeout_ms + limits.write_timeout_ms) / 1000.0
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    async def within_deadline(awaitable: Any) -> Any:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(awaitable, timeout=remaining)

    try:
        reader, writer = await within_deadline(
            asyncio.open_connection(
                endpoint.host,
                endpoint.port,
                limit=limits.maximum_header_bytes + 4,
            )
        )
    except (OSError, TimeoutError, asyncio.TimeoutError) as error:
        raise FramePreviewTransportError(
            "Rust calibration-control listener did not accept a local connection"
        ) from error

    try:
        writer.write(magic)
        if request_body is not None:
            if len(request_body) > maximum_request_bytes:
                raise FramePreviewRequestTooLarge(
                    "calibration request exceeds its configured byte limit"
                )
            writer.write(len(request_body).to_bytes(4, "big"))
            writer.write(request_body)
        try:
            await within_deadline(writer.drain())
            prefix = await within_deadline(reader.readexactly(4))
            response_length = int.from_bytes(prefix, "big")
            if not 1 <= response_length <= limits.maximum_header_bytes:
                raise FramePreviewProtocolError(
                    "Rust calibration-control response length exceeds its configured limit"
                )
            raw_response = await within_deadline(reader.readexactly(response_length))
        except FramePreviewProtocolError:
            raise
        except (
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
        ) as error:
            raise FramePreviewTransportError(
                "Rust calibration-control response missed its local deadline"
            ) from error
        try:
            value = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FramePreviewProtocolError(
                "Rust calibration-control response is not valid UTF-8 JSON"
            ) from error
        return _validate_calibration_control_response(value)
    finally:
        writer.close()
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.wait_for(writer.wait_closed(), timeout=remaining)
        except (OSError, TimeoutError, asyncio.TimeoutError):
            pass


def _validate_calibration_control_response(value: Any) -> CalibrationControlResponse:
    """Check only the versioned transport envelope; snapshot data stays opaque."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "status",
        "snapshot",
        "error",
    }:
        raise FramePreviewProtocolError(
            "Rust calibration-control response fields are invalid"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise FramePreviewProtocolError(
            "Rust calibration-control protocol version is unsupported"
        )
    status = value["status"]
    snapshot = value["snapshot"]
    error = value["error"]
    if not isinstance(status, str) or status not in _CALIBRATION_CONTROL_STATUSES:
        raise FramePreviewProtocolError("Rust calibration-control status is unknown")
    if status == "ok":
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != {"revision", "calibration", "binding_context"}
            or error is not None
        ):
            raise FramePreviewProtocolError(
                "successful Rust calibration-control response is incomplete"
            )
        revision = snapshot["revision"]
        if revision is not None and (
            type(revision) is not int or not 0 <= revision <= _MAX_SAFE_JSON_INTEGER
        ):
            raise FramePreviewProtocolError(
                "Rust calibration-control store revision is invalid"
            )
    elif snapshot is not None or not isinstance(error, str) or not error:
        raise FramePreviewProtocolError(
            "failed Rust calibration-control response carries inconsistent content"
        )
    return CalibrationControlResponse(status=status, snapshot=snapshot, error=error)


def _validate_response_header(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _HEADER_FIELDS:
        raise FramePreviewProtocolError(
            "Rust frame-preview response fields are invalid"
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise FramePreviewProtocolError(
            "Rust frame-preview protocol version is unsupported"
        )
    if type(value["jpeg_bytes"]) is not int or not 0 <= value["jpeg_bytes"] <= _MAX_U32:
        raise FramePreviewProtocolError("Rust frame-preview JPEG length is invalid")
    if not isinstance(value["status"], str) or value["status"] not in {
        "ok",
        "unavailable",
        "busy",
        "invalid_request",
        "oversized",
        "encode_failed",
    }:
        raise FramePreviewProtocolError("Rust frame-preview status is unknown")
    if value["status"] == "ok":
        if value["descriptor"] is None or value["error"] is not None:
            raise FramePreviewProtocolError(
                "successful Rust frame-preview response is incomplete"
            )
    elif (
        value["descriptor"] is not None
        or value["jpeg_bytes"] != 0
        or not isinstance(value["error"], str)
        or not value["error"]
    ):
        raise FramePreviewProtocolError(
            "failed Rust frame-preview response carries inconsistent content"
        )


def _validate_descriptor(
    value: Any,
    *,
    expected_engine_instance_id: str,
    expected_capture_session_id: str,
) -> None:
    if not isinstance(value, dict) or set(value) != _DESCRIPTOR_FIELDS:
        raise FramePreviewProtocolError("Rust frame descriptor fields are invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise FramePreviewProtocolError("Rust frame descriptor version is unsupported")
    if value["engine_instance_id"] != expected_engine_instance_id:
        raise FramePreviewProtocolError("Rust frame descriptor engine identity changed")
    if value["capture_session_id"] != expected_capture_session_id:
        raise FramePreviewProtocolError("Rust frame descriptor capture session changed")
    for name in ("source_frame_id", "camera_profile_id"):
        if (
            not isinstance(value[name], str)
            or not value[name]
            or len(value[name]) > 128
        ):
            raise FramePreviewProtocolError(f"Rust frame descriptor {name} is invalid")
    for name in ("width_px", "height_px"):
        if type(value[name]) is not int or not 1 <= value[name] <= _MAX_U32:
            raise FramePreviewProtocolError(f"Rust frame descriptor {name} is invalid")
    if (
        value["pixel_format"] != "rgb8"
        or not isinstance(value["image_space"], str)
        or value["image_space"] not in {"raw", "rectified"}
    ):
        raise FramePreviewProtocolError("Rust frame descriptor image format is invalid")
    for name, minimum in (
        ("configuration_revision", 1),
        ("calibration_revision", 0),
        ("rectification_revision", 0),
    ):
        if (
            type(value[name]) is not int
            or not minimum <= value[name] <= _MAX_SAFE_JSON_INTEGER
        ):
            raise FramePreviewProtocolError(f"Rust frame descriptor {name} is invalid")
    _validate_timestamp(value["source_timestamp"], optional=True)
    semantics = value["source_timestamp_semantics"]
    if (
        not isinstance(semantics, dict)
        or not isinstance(semantics.get("kind"), str)
        or semantics["kind"] not in {"read_completion_only", "source_timestamp"}
    ):
        raise FramePreviewProtocolError("Rust frame timestamp semantics are invalid")
    if semantics["kind"] == "read_completion_only":
        if set(semantics) != {"kind"} or value["source_timestamp"] is not None:
            raise FramePreviewProtocolError(
                "read-completion frames cannot carry a source timestamp"
            )
    else:
        if set(semantics) != {"kind", "unit", "clock", "domain_id", "stage"}:
            raise FramePreviewProtocolError(
                "Rust source timestamp semantics are incomplete"
            )
        _validate_timestamp(
            {
                "ticks": "0",
                **{
                    field: semantics[field]
                    for field in ("unit", "clock", "domain_id", "stage")
                },
            },
            optional=False,
        )
        timestamp = value["source_timestamp"]
        if timestamp is None or any(
            timestamp[field] != semantics[field]
            for field in ("unit", "clock", "domain_id", "stage")
        ):
            raise FramePreviewProtocolError(
                "Rust frame timestamp differs from its declared clock semantics"
            )
    _validate_host_timestamp(value["host_read_completed"], expected_engine_instance_id)


def _validate_timestamp(value: Any, *, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, dict) or set(value) != {
        "ticks",
        "unit",
        "clock",
        "domain_id",
        "stage",
    }:
        raise FramePreviewProtocolError("Rust frame source timestamp is malformed")
    if (
        not isinstance(value["ticks"], str)
        or len(value["ticks"]) > 20
        or not value["ticks"].isascii()
        or not value["ticks"].isdecimal()
        or not isinstance(value["domain_id"], str)
        or not value["domain_id"]
        or value["unit"] != "nanoseconds"
        or not isinstance(value["clock"], str)
        or value["clock"] not in {"sensor", "monotonic", "unix_epoch"}
        or not isinstance(value["stage"], str)
        or value["stage"]
        not in {
            "exposure",
            "camera_read_complete",
            "shared_memory_publication",
            "frame_arrival",
            "inference_started",
            "inference_complete",
            "batch_publication",
        }
    ):
        raise FramePreviewProtocolError(
            "Rust frame source timestamp values are invalid"
        )


def _validate_host_timestamp(value: Any, expected_engine_instance_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "ticks",
        "unit",
        "clock",
        "domain_id",
        "stage",
        "semantics",
    }:
        raise FramePreviewProtocolError("Rust host read timestamp is malformed")
    timestamp = {
        key: value[key] for key in ("ticks", "unit", "clock", "domain_id", "stage")
    }
    _validate_timestamp(timestamp, optional=False)
    if (
        value["unit"] != "nanoseconds"
        or value["clock"] != "monotonic"
        or value["domain_id"] != expected_engine_instance_id
        or value["stage"] != "camera_read_complete"
        or value["semantics"] != "read_completion_only"
    ):
        raise FramePreviewProtocolError(
            "Rust host timestamp must remain a labeled read-completion time"
        )
