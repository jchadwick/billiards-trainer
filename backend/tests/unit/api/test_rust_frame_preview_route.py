"""Rust calibration preview is a descriptor-preserving gateway proxy."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_state
from backend.api.routes.stream import router
from backend.perception.rust_frame_preview import (
    CalibrationControlResponse,
    FramePreviewBusy,
    FramePreviewImage,
    FramePreviewTransportError,
    FramePreviewUnavailable,
)


def client_for(state: Any) -> TestClient:
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")
    application.dependency_overrides[get_app_state] = lambda: state
    return TestClient(application)


def descriptor() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "engine_instance_id": "engine-rust-test",
        "capture_session_id": "session-rust-test",
        "source_frame_id": "frame-19",
        "width_px": 1280,
        "height_px": 720,
        "pixel_format": "rgb8",
        "image_space": "raw",
        "camera_profile_id": "webcam-default",
        "configuration_revision": 4,
        "calibration_revision": 2,
        "rectification_revision": 3,
        "source_timestamp": None,
        "source_timestamp_semantics": {"kind": "read_completion_only"},
        "host_read_completed": {
            "ticks": "123456",
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": "engine-rust-test",
            "stage": "camera_read_complete",
            "semantics": "read_completion_only",
        },
    }


def rust_state(get_preview: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        perception_provider="rust",
        perception_runtime=SimpleNamespace(
            get_current_frame_preview=get_preview,
            get_calibration_store_snapshot=AsyncMock(),
            submit_calibration_store_request=AsyncMock(),
            calibration_control_max_request_bytes=1024,
        ),
        vision_module=SimpleNamespace(get_current_frame=Mock()),
    )


def test_rust_calibration_frame_relays_exact_jpeg_and_descriptor_without_camera_access() -> (
    None
):
    preview = FramePreviewImage(
        jpeg=b"\xff\xd8test-frame\xff\xd9",
        descriptor=descriptor(),
    )
    get_preview = AsyncMock(return_value=preview)
    state = rust_state(get_preview)

    response = client_for(state).get("/api/v1/stream/calibration-frame")

    assert response.status_code == 200
    assert response.content == preview.jpeg
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"
    encoded = response.headers["x-frame-preview-descriptor"]
    decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    assert json.loads(decoded) == preview.descriptor
    get_preview.assert_awaited_once_with()
    state.vision_module.get_current_frame.assert_not_called()


def test_python_provider_does_not_reach_the_rust_preview_runtime() -> None:
    get_preview = AsyncMock()
    state = rust_state(get_preview)
    state.perception_provider = "python"

    response = client_for(state).get("/api/v1/stream/calibration-frame")

    assert response.status_code == 409
    get_preview.assert_not_awaited()


@pytest.mark.parametrize(
    ("failure", "status_code"),
    [
        (FramePreviewBusy("busy"), 503),
        (FramePreviewUnavailable("no frame"), 503),
        (FramePreviewTransportError("offline"), 503),
    ],
)
def test_rust_preview_unavailable_states_are_reported_without_fallback(
    failure: Exception,
    status_code: int,
) -> None:
    get_preview = AsyncMock(side_effect=failure)
    state = rust_state(get_preview)

    response = client_for(state).get("/api/v1/stream/calibration-frame")

    assert response.status_code == status_code
    get_preview.assert_awaited_once_with()
    state.vision_module.get_current_frame.assert_not_called()


def test_rust_preview_requires_a_live_capture_runtime() -> None:
    state = SimpleNamespace(perception_provider="rust", perception_runtime=None)

    response = client_for(state).get("/api/v1/stream/calibration-frame")

    assert response.status_code == 503


def test_calibration_store_get_relays_rust_snapshot() -> None:
    snapshot = {
        "revision": None,
        "calibration": None,
        "binding_context": None,
    }
    get_preview = AsyncMock()
    state = rust_state(get_preview)
    result = CalibrationControlResponse("ok", snapshot, None)
    state.perception_runtime.get_calibration_store_snapshot.return_value = result

    response = client_for(state).get("/api/v1/stream/calibration-store")

    assert response.status_code == 200
    assert response.json() == result.to_dict()
    state.perception_runtime.get_calibration_store_snapshot.assert_awaited_once_with()


def test_calibration_store_post_forwards_exact_request_bytes() -> None:
    request_body = b'{"schema_version":1,"test":"exact bytes"}'
    get_preview = AsyncMock()
    state = rust_state(get_preview)
    result = CalibrationControlResponse(
        "ok",
        {"revision": 1, "calibration": {"opaque": True}, "binding_context": None},
        None,
    )
    state.perception_runtime.submit_calibration_store_request.return_value = result

    response = client_for(state).post(
        "/api/v1/stream/calibration-store",
        content=request_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == result.to_dict()
    state.perception_runtime.submit_calibration_store_request.assert_awaited_once_with(
        request_body
    )


def test_calibration_store_post_rejects_body_over_configured_limit() -> None:
    get_preview = AsyncMock()
    state = rust_state(get_preview)

    response = client_for(state).post(
        "/api/v1/stream/calibration-store", content=b"x" * 1025
    )

    assert response.status_code == 413
    state.perception_runtime.submit_calibration_store_request.assert_not_awaited()


@pytest.mark.parametrize(
    ("status", "status_code"),
    [
        ("invalid_request", 422),
        ("stale_context", 409),
        ("stale_revision", 409),
        ("busy", 503),
        ("store_error", 503),
        ("unsupported", 501),
    ],
)
def test_calibration_store_post_maps_rust_status_to_http(
    status: str, status_code: int
) -> None:
    state = rust_state(AsyncMock())
    result = CalibrationControlResponse(status, None, "rust status")
    state.perception_runtime.submit_calibration_store_request.return_value = result

    response = client_for(state).post("/api/v1/stream/calibration-store", content=b"{}")

    assert response.status_code == status_code
    assert response.json() == result.to_dict()
