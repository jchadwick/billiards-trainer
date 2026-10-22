"""Protocol bounds and descriptor provenance for Rust frame preview."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from backend.perception.rust_frame_preview import (
    CalibrationControlResponse,
    FramePreviewBusy,
    FramePreviewCalibrationControlSettings,
    FramePreviewClientLimits,
    FramePreviewEndpoint,
    FramePreviewError,
    FramePreviewImage,
    FramePreviewProcessSettings,
    FramePreviewProtocolError,
    FramePreviewRequestTooLarge,
    FramePreviewTransportError,
    FramePreviewUnavailable,
    _validate_calibration_control_response,
    _validate_descriptor,
    _validate_host_timestamp,
    _validate_response_header,
    _validate_timestamp,
    fetch_frame_preview,
    parse_ready_record,
    read_calibration_store,
    submit_calibration_request,
)


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine_instance_id": "engine-1",
        "capture_session_id": "session-1",
        "source_frame_id": "frame-9",
        "width_px": 1280,
        "height_px": 720,
        "pixel_format": "rgb8",
        "image_space": "raw",
        "camera_profile_id": "camera-default",
        "configuration_revision": 5,
        "calibration_revision": 2,
        "rectification_revision": 1,
        "source_timestamp": {
            "ticks": "987654",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "sensor-1",
            "stage": "exposure",
        },
        "source_timestamp_semantics": {
            "kind": "source_timestamp",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "sensor-1",
            "stage": "exposure",
        },
        "host_read_completed": {
            "ticks": "123456",
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": "engine-1",
            "stage": "camera_read_complete",
            "semantics": "read_completion_only",
        },
    }


def _endpoint(port: int) -> FramePreviewEndpoint:
    return FramePreviewEndpoint(
        host="127.0.0.1",
        port=port,
        engine_instance_id="engine-1",
        capture_session_id="session-1",
    )


def _limits(**overrides: int) -> FramePreviewClientLimits:
    values = {
        "maximum_header_bytes": 8192,
        "maximum_jpeg_bytes": 1_048_576,
        "request_timeout_ms": 250,
        "write_timeout_ms": 250,
    }
    values.update(overrides)
    return FramePreviewClientLimits(**values)


def _response(
    *,
    status: str = "ok",
    descriptor: dict[str, Any] | None = None,
    jpeg: bytes = b"\xff\xd8frame-data\xff\xd9",
    error: str | None = None,
) -> bytes:
    successful = status == "ok"
    header = {
        "schema_version": 1,
        "status": status,
        "descriptor": descriptor if successful else None,
        "jpeg_bytes": len(jpeg) if successful else 0,
        "error": error if not successful else None,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    body = jpeg if successful else b""
    return len(header_bytes).to_bytes(4, "big") + header_bytes + body


def _calibration_response(
    *,
    status: str = "ok",
    snapshot: dict[str, Any] | None = None,
    error: str | None = None,
) -> bytes:
    if status == "ok" and snapshot is None:
        snapshot = {"revision": None, "calibration": None, "binding_context": None}
    value = {
        "schema_version": 1,
        "status": status,
        "snapshot": snapshot if status == "ok" else None,
        "error": None if status == "ok" else error,
    }
    body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return len(body).to_bytes(4, "big") + body


def _serve_calibration_response(
    response: bytes,
    *,
    magic: bytes,
    request_body: bytes | None = None,
) -> tuple[int, CalibrationControlResponse]:
    async def run() -> tuple[int, CalibrationControlResponse]:
        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                assert await reader.readexactly(8) == magic
                if request_body is not None:
                    length = int.from_bytes(await reader.readexactly(4), "big")
                    assert length == len(request_body)
                    assert await reader.readexactly(length) == request_body
                writer.write(response)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        try:
            if magic == b"BTCR0001":
                result = await read_calibration_store(_endpoint(port), _limits())
            else:
                result = await submit_calibration_request(
                    _endpoint(port),
                    _limits(),
                    request_body or b"",
                    maximum_request_bytes=1024,
                )
            return port, result
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(run())


def _serve_response(
    response: bytes, *, wait_after_request: bool = False
) -> tuple[int, Any]:
    async def run() -> tuple[int, Any]:
        requested = asyncio.Event()

        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                assert await reader.readexactly(8) == b"BTPV0001"
                requested.set()
                if wait_after_request:
                    await asyncio.Event().wait()
                else:
                    writer.write(response)
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        try:
            if wait_after_request:
                with pytest.raises(FramePreviewTransportError):
                    await asyncio.wait_for(
                        fetch_frame_preview(
                            _endpoint(port),
                            _limits(request_timeout_ms=30, write_timeout_ms=30),
                        ),
                        timeout=1,
                    )
                await asyncio.wait_for(requested.wait(), timeout=1)
                return port, None
            return port, await fetch_frame_preview(_endpoint(port), _limits())
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(run())


def test_ready_record_accepts_only_loopback_endpoint_and_expected_process_identity() -> (
    None
):
    line = (
        b'frame_preview_ready={"schema_version":1,"address":"127.0.0.1:40231",'
        b'"engine_instance_id":"engine-1","capture_session_id":"session-1"}'
    )

    endpoint = parse_ready_record(
        line,
        expected_engine_instance_id="engine-1",
        expected_capture_session_id="session-1",
        maximum_record_bytes=1024,
    )

    assert endpoint == FramePreviewEndpoint("127.0.0.1", 40231, "engine-1", "session-1")
    with pytest.raises(FramePreviewProtocolError, match="loopback"):
        parse_ready_record(
            line.replace(b"127.0.0.1", b"192.0.2.1"),
            expected_engine_instance_id="engine-1",
            expected_capture_session_id="session-1",
            maximum_record_bytes=1024,
        )
    with pytest.raises(FramePreviewProtocolError, match="session"):
        parse_ready_record(
            line,
            expected_engine_instance_id="engine-1",
            expected_capture_session_id="other-session",
            maximum_record_bytes=1024,
        )


def test_ready_record_accepts_ipv6_and_rejects_wrong_engine_identity() -> None:
    line = (
        b'frame_preview_ready={"schema_version":1,"address":"[::1]:40231",'
        b'"engine_instance_id":"engine-1","capture_session_id":"session-1"}'
    )
    endpoint = parse_ready_record(
        line,
        expected_engine_instance_id="engine-1",
        expected_capture_session_id="session-1",
        maximum_record_bytes=1024,
    )
    assert (endpoint.host, endpoint.port) == ("::1", 40231)
    with pytest.raises(FramePreviewProtocolError, match="engine identity"):
        parse_ready_record(
            line,
            expected_engine_instance_id="another-engine",
            expected_capture_session_id="session-1",
            maximum_record_bytes=1024,
        )


def test_client_preserves_the_rust_jpeg_and_matching_frame_descriptor() -> None:
    jpeg = b"\xff\xd8exact-frame-bytes\xff\xd9"

    _port, preview = _serve_response(_response(descriptor=_descriptor(), jpeg=jpeg))

    assert preview == FramePreviewImage(jpeg=jpeg, descriptor=_descriptor())


def test_calibration_control_response_serializes_versioned_envelope() -> None:
    response = CalibrationControlResponse("ok", {"revision": 1}, None)
    assert response.to_dict() == {
        "schema_version": 1,
        "status": "ok",
        "snapshot": {"revision": 1},
        "error": None,
    }


@pytest.mark.parametrize(
    ("status", "error", "exception"),
    [
        ("unavailable", "no_current_frame", FramePreviewUnavailable),
        ("busy", "encode_capacity_full", FramePreviewBusy),
    ],
)
def test_client_surfaces_latest_frame_unavailable_and_overload_without_fallback(
    status: str,
    error: str,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _serve_response(_response(status=status, error=error))


def test_client_rejects_oversized_header_and_cross_session_descriptor() -> None:
    with pytest.raises(FramePreviewProtocolError, match="header length"):
        _serve_response((8193).to_bytes(4, "big"))

    wrong_session = _descriptor()
    wrong_session["capture_session_id"] = "different-session"
    with pytest.raises(FramePreviewProtocolError, match="capture session"):
        _serve_response(_response(descriptor=wrong_session))


def test_client_rejects_source_time_that_disagrees_with_its_clock_semantics() -> None:
    descriptor = _descriptor()
    descriptor["source_timestamp_semantics"]["domain_id"] = "different-clock"
    with pytest.raises(FramePreviewProtocolError, match="timestamp differs"):
        _serve_response(_response(descriptor=descriptor))


def test_client_request_has_a_total_deadline_when_rust_stops_responding() -> None:
    _serve_response(b"", wait_after_request=True)


def test_process_settings_require_bounded_loopback_configuration() -> None:
    settings = FramePreviewProcessSettings(
        bind_address="127.0.0.1",
        port=0,
        maximum_header_bytes=8192,
        maximum_jpeg_bytes=4_194_304,
        maximum_concurrent_encodes=1,
        jpeg_quality=85,
        request_timeout_ms=1000,
        write_timeout_ms=2000,
    )
    settings.validate()
    assert settings.socket_address == "127.0.0.1:0"
    assert settings.client_limits.maximum_jpeg_bytes == 4_194_304

    remote = replace(settings, bind_address="0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        remote.validate()


def test_calibration_control_read_and_submit_relay_versioned_rust_records() -> None:
    snapshot = {
        "revision": 2,
        "calibration": {"calibration_revision": 2},
        "binding_context": {"camera_profile_id": "profile-1"},
    }
    _port, read_result = _serve_calibration_response(
        _calibration_response(snapshot=snapshot), magic=b"BTCR0001"
    )
    assert read_result == CalibrationControlResponse("ok", snapshot, None)

    request_body = b'{"schema_version":1,"opaque":"forward-this-exactly"}'
    _port, submit_result = _serve_calibration_response(
        _calibration_response(snapshot=snapshot),
        magic=b"BTCA0001",
        request_body=request_body,
    )
    assert submit_result == CalibrationControlResponse("ok", snapshot, None)


def test_calibration_control_rejects_oversized_request_before_opening_socket() -> None:
    with pytest.raises(FramePreviewRequestTooLarge, match="byte limit"):
        asyncio.run(
            submit_calibration_request(
                _endpoint(1),
                _limits(),
                b"x" * 1025,
                maximum_request_bytes=1024,
            )
        )


def test_calibration_control_rejects_malformed_response_envelopes() -> None:
    malformed = b'{"schema_version":1,"status":"ok","snapshot":null,"error":null}'
    response = len(malformed).to_bytes(4, "big") + malformed

    with pytest.raises(FramePreviewProtocolError, match="incomplete"):
        _serve_calibration_response(response, magic=b"BTCR0001")


@pytest.mark.parametrize(
    "status",
    [
        "invalid_request",
        "stale_context",
        "stale_revision",
        "busy",
        "store_error",
        "unsupported",
    ],
)
def test_calibration_control_preserves_rust_status_values(status: str) -> None:
    _port, result = _serve_calibration_response(
        _calibration_response(status=status, error="rust-owned-status"),
        magic=b"BTCR0001",
    )
    assert result.status == status
    assert result.error == "rust-owned-status"


def test_calibration_store_settings_enforce_path_and_boundaries(
    tmp_path: Path,
) -> None:
    valid = FramePreviewCalibrationControlSettings(
        store_path=tmp_path / "calibration.json",
        store_lock_timeout_ms=100,
        maximum_store_file_bytes=4096,
        maximum_request_bytes=1024,
    )
    valid.validate(project_root=tmp_path)

    with pytest.raises(ValueError, match="absolute Path"):
        replace(valid, store_path=Path("relative.json")).validate()
    with pytest.raises(ValueError, match="inside the project root"):
        replace(valid, store_path=tmp_path.parent / "outside.json").validate(
            project_root=tmp_path
        )
    with pytest.raises(ValueError, match="directory"):
        replace(valid, store_path=tmp_path).validate()

    for field, value in (
        ("store_lock_timeout_ms", 0),
        ("store_lock_timeout_ms", True),
        ("maximum_store_file_bytes", 16_777_217),
        ("maximum_request_bytes", 1023),
    ):
        with pytest.raises(ValueError, match="must be an integer"):
            replace(valid, **{field: value}).validate()


def test_process_settings_validate_each_runtime_bound_and_ipv6_socket_syntax(
    tmp_path: Path,
) -> None:
    settings = FramePreviewProcessSettings(
        bind_address="127.0.0.1",
        port=0,
        maximum_header_bytes=8192,
        maximum_jpeg_bytes=1_048_576,
        maximum_concurrent_encodes=1,
        jpeg_quality=85,
        request_timeout_ms=100,
        write_timeout_ms=200,
    )
    with pytest.raises(ValueError, match="numeric IP"):
        replace(settings, bind_address="localhost").validate()
    with pytest.raises(ValueError, match="port must be an integer"):
        replace(settings, port=True).validate()
    with pytest.raises(ValueError, match="maximum_jpeg_bytes"):
        replace(settings, maximum_jpeg_bytes=16_777_217).validate()
    with pytest.raises(ValueError, match="maximum_concurrent_encodes"):
        replace(settings, maximum_concurrent_encodes=5).validate()
    with pytest.raises(ValueError, match="jpeg_quality"):
        replace(settings, jpeg_quality=0).validate()
    with pytest.raises(ValueError, match="request_timeout_ms"):
        replace(settings, request_timeout_ms=60_001).validate()
    with pytest.raises(ValueError, match="write_timeout_ms"):
        replace(settings, write_timeout_ms=False).validate()

    calibration = FramePreviewCalibrationControlSettings(
        store_path=tmp_path / "calibration.json",
        store_lock_timeout_ms=100,
        maximum_store_file_bytes=4096,
        maximum_request_bytes=1024,
    )
    with pytest.raises(ValueError, match="at least 4096"):
        replace(
            settings, maximum_header_bytes=4095, calibration_control=calibration
        ).validate(project_root=tmp_path)
    with pytest.raises(ValueError, match="validated calibration settings object"):
        replace(settings, calibration_control=object()).validate()
    with pytest.raises(ValueError, match="project root"):
        replace(settings, calibration_control=calibration).validate(
            project_root=tmp_path / "different-root"
        )

    ipv6 = replace(settings, bind_address="::1")
    ipv6.validate()
    assert ipv6.socket_address == "[::1]:0"


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (_limits(maximum_header_bytes=0), "positive integer"),
        (_limits(maximum_header_bytes=1_048_577), "bounded protocol limit"),
        (_limits(maximum_jpeg_bytes=16_777_217), "bounded protocol limit"),
        (_limits(request_timeout_ms=60_001), "timeouts exceed"),
        (_limits(write_timeout_ms=60_001), "timeouts exceed"),
        (_limits(maximum_jpeg_bytes=True), "positive integer"),
    ],
)
def test_client_limits_reject_unbounded_values(
    limits: FramePreviewClientLimits, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        limits.validate()


@pytest.mark.parametrize(
    ("line", "maximum_bytes", "expected_message"),
    [
        ("not-bytes", 1024, "oversized"),
        (b"frame_preview_not_ready={}", 1024, "prefix"),
        (b"frame_preview_ready=\xff", 1024, "UTF-8 JSON"),
        (b"frame_preview_ready={}", 4, "oversized"),
        (
            b'frame_preview_ready={"schema_version":1,"address":"127.0.0.1:1",'
            b'"engine_instance_id":"engine-1","capture_session_id":"session-1",'
            b'"unexpected":true}',
            1024,
            "fields",
        ),
        (
            b'frame_preview_ready={"schema_version":true,"address":"127.0.0.1:1",'
            b'"engine_instance_id":"engine-1","capture_session_id":"session-1"}',
            1024,
            "version",
        ),
        (
            b'frame_preview_ready={"schema_version":1,"address":4,'
            b'"engine_instance_id":"engine-1","capture_session_id":"session-1"}',
            1024,
            "address must be a string",
        ),
    ],
)
def test_ready_record_rejects_malformed_readiness_lines(
    line: bytes | str, maximum_bytes: int, expected_message: str
) -> None:
    with pytest.raises(FramePreviewProtocolError, match=expected_message):
        parse_ready_record(
            line,  # type: ignore[arg-type]
            expected_engine_instance_id="engine-1",
            expected_capture_session_id="session-1",
            maximum_record_bytes=maximum_bytes,
        )


@pytest.mark.parametrize(
    ("address", "message"),
    [
        ("[::1", "IPv6 address is invalid"),
        ("[::1]x42", "IPv6 address is invalid"),
        ("localhost:42", "numeric loopback IP"),
        ("192.0.2.1:42", "not loopback"),
        ("127.0.0.1:nope", "port is invalid"),
        ("127.0.0.1:0", "outside its valid range"),
        ("127.0.0.1:65536", "outside its valid range"),
        ("::1:42", "address is invalid"),
    ],
)
def test_readiness_address_parser_rejects_ambiguous_or_remote_sockets(
    address: str, message: str
) -> None:
    record = {
        "schema_version": 1,
        "address": address,
        "engine_instance_id": "engine-1",
        "capture_session_id": "session-1",
    }
    line = b"frame_preview_ready=" + json.dumps(record).encode()
    with pytest.raises(FramePreviewProtocolError, match=message):
        parse_ready_record(
            line,
            expected_engine_instance_id="engine-1",
            expected_capture_session_id="session-1",
            maximum_record_bytes=1024,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("width_px"), "fields"),
        (lambda value: value.update(schema_version=True), "version"),
        (
            lambda value: value.update(engine_instance_id="old-engine"),
            "engine identity",
        ),
        (
            lambda value: value.update(capture_session_id="old-session"),
            "capture session",
        ),
        (lambda value: value.update(source_frame_id=""), "source_frame_id"),
        (lambda value: value.update(camera_profile_id="x" * 129), "camera_profile_id"),
        (lambda value: value.update(width_px=True), "width_px"),
        (lambda value: value.update(height_px=0), "height_px"),
        (lambda value: value.update(pixel_format="bgr8"), "image format"),
        (lambda value: value.update(image_space="unknown"), "image format"),
        (
            lambda value: value.update(configuration_revision=0),
            "configuration_revision",
        ),
        (lambda value: value.update(calibration_revision=-1), "calibration_revision"),
        (
            lambda value: value.update(rectification_revision=9_007_199_254_740_992),
            "rectification_revision",
        ),
        (
            lambda value: value.update(source_timestamp_semantics={"kind": "unknown"}),
            "timestamp semantics",
        ),
        (
            lambda value: value.update(
                source_timestamp_semantics={
                    "kind": "read_completion_only",
                    "unit": "nanoseconds",
                }
            ),
            "read-completion frames",
        ),
        (
            lambda value: value.update(
                source_timestamp_semantics={"kind": "source_timestamp"}
            ),
            "semantics are incomplete",
        ),
        (lambda value: value.update(source_timestamp=None), "timestamp differs"),
        (lambda value: value.update(host_read_completed={}), "host read timestamp"),
    ],
)
def test_descriptor_validator_rejects_provenance_and_format_violations(
    mutation: Any, message: str
) -> None:
    descriptor = _descriptor()
    mutation(descriptor)
    with pytest.raises(FramePreviewProtocolError, match=message):
        _validate_descriptor(
            descriptor,
            expected_engine_instance_id="engine-1",
            expected_capture_session_id="session-1",
        )


@pytest.mark.parametrize(
    "timestamp",
    [
        {},
        {
            "ticks": "١",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "s",
            "stage": "exposure",
        },
        {
            "ticks": "1" * 21,
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "s",
            "stage": "exposure",
        },
        {
            "ticks": "-1",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "s",
            "stage": "exposure",
        },
        {
            "ticks": "1",
            "unit": "seconds",
            "clock": "sensor",
            "domain_id": "s",
            "stage": "exposure",
        },
        {
            "ticks": "1",
            "unit": "nanoseconds",
            "clock": "wall",
            "domain_id": "s",
            "stage": "exposure",
        },
        {
            "ticks": "1",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "",
            "stage": "exposure",
        },
        {
            "ticks": "1",
            "unit": "nanoseconds",
            "clock": "sensor",
            "domain_id": "s",
            "stage": "unknown",
        },
    ],
)
def test_timestamp_validator_rejects_invalid_clock_provenance(
    timestamp: Any,
) -> None:
    with pytest.raises(FramePreviewProtocolError, match="timestamp"):
        _validate_timestamp(timestamp, optional=False)


def test_optional_source_timestamps_allow_missing_but_host_time_must_be_read_completion() -> (
    None
):
    _validate_timestamp(None, optional=True)
    timestamp = _descriptor()["host_read_completed"]
    for field, value in (
        ("clock", "sensor"),
        ("domain_id", "other-engine"),
        ("stage", "exposure"),
        ("semantics", "source_timestamp"),
    ):
        invalid = {**timestamp, field: value}
        with pytest.raises(FramePreviewProtocolError, match="read-completion time"):
            _validate_host_timestamp(invalid, "engine-1")


def test_descriptor_accepts_explicit_read_completion_without_claiming_exposure_time() -> (
    None
):
    descriptor = _descriptor()
    descriptor["source_timestamp"] = None
    descriptor["source_timestamp_semantics"] = {"kind": "read_completion_only"}
    _validate_descriptor(
        descriptor,
        expected_engine_instance_id="engine-1",
        expected_capture_session_id="session-1",
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({}, "fields"),
        (
            {
                "schema_version": True,
                "status": "ok",
                "descriptor": {},
                "jpeg_bytes": 1,
                "error": None,
            },
            "version",
        ),
        (
            {
                "schema_version": 1,
                "status": "ok",
                "descriptor": {},
                "jpeg_bytes": True,
                "error": None,
            },
            "length is invalid",
        ),
        (
            {
                "schema_version": 1,
                "status": "unknown",
                "descriptor": None,
                "jpeg_bytes": 0,
                "error": "bad",
            },
            "status is unknown",
        ),
        (
            {
                "schema_version": 1,
                "status": "ok",
                "descriptor": None,
                "jpeg_bytes": 0,
                "error": None,
            },
            "incomplete",
        ),
        (
            {
                "schema_version": 1,
                "status": "busy",
                "descriptor": None,
                "jpeg_bytes": 0,
                "error": "",
            },
            "inconsistent",
        ),
    ],
)
def test_frame_response_header_rejects_malformed_envelopes(
    record: dict[str, Any], message: str
) -> None:
    with pytest.raises(FramePreviewProtocolError, match=message):
        _validate_response_header(record)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({}, "fields"),
        (
            {"schema_version": 2, "status": "ok", "snapshot": {}, "error": None},
            "version",
        ),
        (
            {"schema_version": 1, "status": "unknown", "snapshot": None, "error": "x"},
            "status is unknown",
        ),
        (
            {"schema_version": 1, "status": "ok", "snapshot": None, "error": None},
            "incomplete",
        ),
        (
            {
                "schema_version": 1,
                "status": "ok",
                "snapshot": {
                    "revision": True,
                    "calibration": None,
                    "binding_context": None,
                },
                "error": None,
            },
            "revision is invalid",
        ),
        (
            {
                "schema_version": 1,
                "status": "stale_context",
                "snapshot": None,
                "error": "",
            },
            "inconsistent",
        ),
    ],
)
def test_calibration_control_validator_rejects_malformed_envelopes(
    record: dict[str, Any], message: str
) -> None:
    with pytest.raises(FramePreviewProtocolError, match=message):
        _validate_calibration_control_response(record)


def test_calibration_request_requires_bytes_and_positive_integer_limit() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        asyncio.run(
            submit_calibration_request(
                _endpoint(1), _limits(), bytearray(b"{}"), maximum_request_bytes=10  # type: ignore[arg-type]
            )
        )
    for maximum in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            asyncio.run(
                submit_calibration_request(
                    _endpoint(1), _limits(), b"{}", maximum_request_bytes=maximum
                )
            )


def test_preview_client_reports_unreachable_listener_as_transport_error() -> None:
    async def run() -> None:
        server = await asyncio.start_server(
            lambda _reader, _writer: None, "127.0.0.1", 0
        )
        port = int(server.sockets[0].getsockname()[1])
        server.close()
        await server.wait_closed()
        with pytest.raises(FramePreviewTransportError, match="did not accept"):
            await fetch_frame_preview(_endpoint(port), _limits())

    asyncio.run(run())


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"\x00\x00\x00\x00", "header length"),
        (b"\x00\x00\x00\x05{}", "missed its local deadline"),
        (b"\x00\x00\x00\x01\xff", "UTF-8 JSON"),
        (_response(status="encode_failed", error="encoder failed"), "request failed"),
        (_response(descriptor=_descriptor(), jpeg=b"not-a-jpeg"), "not a JPEG"),
    ],
)
def test_preview_client_rejects_invalid_or_failed_responses(
    response: bytes, message: str
) -> None:
    with pytest.raises(FramePreviewError, match=message):
        _serve_response(response)


def test_calibration_store_rejects_invalid_response_length_and_unreachable_listener() -> (
    None
):
    with pytest.raises(FramePreviewProtocolError, match="response length"):
        _serve_calibration_response(b"\x00\x00\x00\x00", magic=b"BTCR0001")

    async def run() -> None:
        server = await asyncio.start_server(
            lambda _reader, _writer: None, "127.0.0.1", 0
        )
        port = int(server.sockets[0].getsockname()[1])
        server.close()
        await server.wait_closed()
        with pytest.raises(FramePreviewTransportError, match="did not accept"):
            await read_calibration_store(_endpoint(port), _limits())

    asyncio.run(run())
