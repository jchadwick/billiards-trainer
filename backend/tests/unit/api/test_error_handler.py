"""Focused checks for the public HTTP error contract."""

import asyncio
import logging
from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from backend.api.middleware.error_handler import (
    CustomHTTPException,
    ErrorHandlerConfig,
    general_exception_handler,
    get_error_handler,
    http_exception_handler,
    setup_error_handling,
)
from backend.api.middleware.logging import LoggingConfig, setup_logging_middleware
from backend.api.middleware.tracing import TracingConfig, setup_tracing_middleware
from backend.api.models.common import ErrorCode


def make_request(
    request_id: str = "request-test-123",
    headers: Optional[list[tuple[bytes, bytes]]] = None,
) -> Request:
    """Create the small ASGI request scope used by direct handler tests."""
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/test",
            "raw_path": b"/api/test",
            "query_string": b"",
            "headers": headers or [],
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 1234),
        }
    )
    request.state.request_id = request_id
    return request


@pytest.fixture(autouse=True)
def configure_handler():
    """Install a quiet handler so tests also exercise the public setup path."""
    setup_error_handling(FastAPI(), ErrorHandlerConfig(log_errors=False))


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (409, "GAME_002"),
        (422, "VAL_001"),
        (502, "NET_001"),
        (504, "NET_002"),
    ],
)
def test_http_status_maps_to_registered_error_code(status_code, error_code):
    request = make_request()
    response = asyncio.run(
        http_exception_handler(
            request,
            HTTPException(status_code=status_code, detail="expected request error"),
        )
    )

    assert response.status_code == status_code
    assert response.headers["x-request-id"] == "request-test-123"
    assert response.body
    assert error_code.encode() in response.body


@pytest.mark.parametrize(
    ("status_code", "detail", "header"),
    [
        (429, "slow down", ("Retry-After", "15")),
        (500, "private filesystem path", ("Retry-After", "2")),
    ],
)
def test_http_exception_preserves_headers_and_sanitizes_server_details(
    status_code, detail, header
):
    request = make_request()
    response = asyncio.run(
        http_exception_handler(
            request,
            HTTPException(
                status_code=status_code,
                detail=detail,
                headers={header[0]: header[1]},
            ),
        )
    )

    assert response.headers[header[0].lower()] == header[1]
    assert response.headers["x-request-id"] == "request-test-123"
    if status_code >= 500:
        assert detail.encode() not in response.body
        assert b"Internal server error" in response.body
    else:
        assert detail.encode() in response.body


@pytest.mark.parametrize(
    ("error_code", "expected_status", "expected_message"),
    [
        ("CAM_001", 503, "Camera unavailable"),
        ("PROC_001", 503, "Vision processing failed"),
    ],
)
def test_registered_domain_errors_keep_code_and_safe_message(
    error_code, expected_status, expected_message
):
    request = make_request()
    response = asyncio.run(
        http_exception_handler(
            request,
            CustomHTTPException(error_code, details={"step_failed": "frame_capture"}),
        )
    )

    assert response.status_code == expected_status
    assert error_code.encode() in response.body
    assert expected_message.encode() in response.body
    assert b"request-test-123" in response.body


def test_vision_failure_enum_matches_registered_error_code():
    assert ErrorCode.VISION_PROCESSING_FAILED.value == "PROC_001"


def test_unhandled_exception_does_not_leak_message_or_traceback():
    request = make_request()
    response = asyncio.run(
        general_exception_handler(
            request, RuntimeError("private model path and device failure")
        )
    )

    assert response.status_code == 500
    assert b"SYS_001" in response.body
    assert b"Internal server error" in response.body
    assert b"private model path" not in response.body
    assert b"RuntimeError" in response.body
    assert response.headers["x-request-id"] == "request-test-123"


def test_error_logs_ignore_untrusted_forwarded_client_address():
    request = make_request(headers=[(b"x-forwarded-for", b"203.0.113.99, 127.0.0.1")])

    handler = get_error_handler()
    assert handler is not None
    assert handler._get_client_ip(request) == "127.0.0.1"


def test_middleware_stack_shares_request_id_in_error_header_body_and_logs(caplog):
    app = FastAPI()
    setup_tracing_middleware(app, TracingConfig())
    setup_logging_middleware(app, LoggingConfig())
    setup_error_handling(app, ErrorHandlerConfig(log_errors=False))

    @app.get("/correlated-error")
    async def correlated_error():
        raise HTTPException(status_code=500, detail="private exception detail")

    caplog.set_level(logging.INFO, logger="api.requests")
    with TestClient(app) as client:
        response = client.get("/correlated-error")

    request_id = response.json()["request_id"]
    assert request_id
    assert response.headers["x-request-id"] == request_id
    assert "private exception detail" not in response.text

    request_response_records = [
        record
        for record in caplog.records
        if record.name == "api.requests"
        and getattr(record, "type", None) in {"request", "response"}
    ]
    assert {record.type for record in request_response_records} == {
        "request",
        "response",
    }
    assert {record.request_id for record in request_response_records} == {request_id}


def test_tracing_alone_keeps_client_request_id_header_behavior():
    app = FastAPI()
    setup_tracing_middleware(app, TracingConfig())
    setup_error_handling(app, ErrorHandlerConfig(log_errors=False))

    @app.get("/caller-id-error")
    async def caller_id_error():
        raise HTTPException(status_code=400, detail="bad request")

    with TestClient(app) as client:
        response = client.get(
            "/caller-id-error", headers={"X-Request-ID": "client-request-123"}
        )

    assert response.status_code == 400
    assert response.headers["x-request-id"] == "client-request-123"
    assert response.json()["request_id"] == "client-request-123"
