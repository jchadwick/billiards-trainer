"""Failure contracts for diagnostics providers and endpoints.

These tests keep the diagnostic routes useful on machines with incomplete
telemetry, unavailable cameras, and invalid provider responses. All hardware,
time-consuming waits, and executor work is replaced with deterministic inputs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from fastapi import HTTPException

from backend.api.routes import diagnostics

if TYPE_CHECKING:
    from backend.api.dependencies import ApplicationState


class _ImmediateLoop:
    """Provide deterministic run_in_executor results to async route tests."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def run_in_executor(self, _executor: Any, _function: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.result


class _IncompletePsutil:
    """Expose valid resource values while optional CPU telemetry fails."""

    def cpu_freq(self) -> Any:
        raise OSError("frequency counter unavailable")

    def cpu_count(self, logical: bool = True) -> int:
        return 8 if logical else 4

    def cpu_percent(self, interval: float | None = None) -> float:
        raise OSError("usage counter unavailable")

    def virtual_memory(self) -> Any:
        return SimpleNamespace(total=100, available=70, used=30, percent=30.0)

    def disk_usage(self, path: str) -> Any:
        assert path == "/"
        return SimpleNamespace(total=100, used=25, free=75)

    def net_if_addrs(self) -> dict[str, list[Any]]:
        return {}


@pytest.mark.asyncio
async def test_hardware_route_keeps_cpu_device_when_optional_counters_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", _IncompletePsutil())

    devices = await diagnostics.detect_hardware()

    cpu = next(device for device in devices if device.id == "cpu")
    assert cpu.details == {
        "cores": 4,
        "logical_cores": 8,
        "frequency": 0,
        "usage": 0,
    }


@pytest.mark.asyncio
async def test_hardware_route_reports_malformed_disk_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidDiskPsutil(_IncompletePsutil):
        def disk_usage(self, path: str) -> Any:
            assert path == "/"
            return SimpleNamespace(total=0, used=0, free=0)

    monkeypatch.setattr(diagnostics, "psutil", InvalidDiskPsutil())

    with pytest.raises(HTTPException) as error:
        await diagnostics.detect_hardware()

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to detect hardware devices"


@pytest.mark.asyncio
async def test_summary_translates_malformed_memory_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidMemoryPsutil(_IncompletePsutil):
        def virtual_memory(self) -> Any:
            return SimpleNamespace(percent=30.0)

    monkeypatch.setattr(diagnostics, "psutil", InvalidMemoryPsutil())

    with pytest.raises(HTTPException) as error:
        await diagnostics.get_diagnostic_summary()

    assert error.value.status_code == 500
    detail = cast("dict[str, Any]", error.value.detail)
    assert detail["error"] == "Diagnostic Summary Failed"
    assert "total" in detail["details"]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "component", "issue"),
    [
        (
            diagnostics.diagnose_camera,
            "camera",
            "Camera diagnostic error: worker unavailable",
        ),
        (
            diagnostics.diagnose_fisheye_correction,
            "fisheye_correction",
            "Fisheye diagnostic error: worker unavailable",
        ),
        (
            diagnostics.diagnose_ball_detection,
            "ball_detection",
            "Detection diagnostic error: worker unavailable",
        ),
    ],
)
async def test_vision_route_returns_error_result_when_executor_fails(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: Any,
    component: str,
    issue: str,
) -> None:
    monkeypatch.setattr(
        diagnostics.asyncio,
        "get_event_loop",
        lambda: _ImmediateLoop(error=RuntimeError("worker unavailable")),
    )

    result = await endpoint()

    assert result.component == component
    assert result.status == "error"
    assert result.score == 0.0
    assert result.details == {"error": "worker unavailable"}
    assert result.issues == [issue]
    assert result.duration >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "expected_status"),
    [(49.9, "failed"), (50.0, "degraded"), (89.9, "degraded"), (90.0, "healthy")],
)
async def test_fisheye_route_applies_status_thresholds_at_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        diagnostics.asyncio,
        "get_event_loop",
        lambda: _ImmediateLoop(
            {"score": score, "details": {}, "issues": [], "recommendations": []}
        ),
    )

    result = await diagnostics.diagnose_fisheye_correction()

    assert result.status == expected_status
    assert result.score == score


def test_fisheye_diagnostic_reports_incomplete_calibration_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from backend.vision.calibration.camera import CameraCalibrator

    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr(CameraCalibrator, "__init__", lambda _self: None)
    monkeypatch.setattr(CameraCalibrator, "load_calibration", lambda _self, _path: True)
    monkeypatch.setattr(
        CameraCalibrator,
        "camera_params",
        property(
            lambda _self: SimpleNamespace(
                camera_matrix=None, distortion_coefficients=None
            )
        ),
        raising=False,
    )

    result = diagnostics._run_fisheye_diagnostic_sync()

    assert result["score"] == 50.0
    assert result["details"]["calibration_loaded"] is True
    assert "has_camera_matrix" not in result["details"]
    assert "has_distortion_coeffs" not in result["details"]
    assert result["issues"] == [
        "Camera matrix not available in calibration",
        "Distortion coefficients not available",
    ]


@pytest.mark.asyncio
async def test_fisheye_image_route_translates_unexpected_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics.asyncio,
        "get_event_loop",
        lambda: _ImmediateLoop(error=RuntimeError("image worker crashed")),
    )

    with pytest.raises(HTTPException) as error:
        await diagnostics.get_fisheye_test_image()

    assert error.value.status_code == 500
    assert error.value.detail == "image worker crashed"


@pytest.mark.asyncio
async def test_download_endpoint_rejects_oversized_request() -> None:
    with pytest.raises(HTTPException) as error:
        await diagnostics.test_download_endpoint(101)

    assert error.value.status_code == 400
    assert error.value.detail == "Size too large"


@pytest.mark.asyncio
async def test_download_and_upload_endpoints_return_transfer_metadata() -> None:
    download = await diagnostics.test_download_endpoint(0)
    upload = await diagnostics.test_upload_endpoint()

    assert download == {"size": 0, "message": "Downloaded 0MB test file"}
    assert upload == {"message": "Upload test completed", "received_size": 0}


@pytest.mark.asyncio
async def test_raw_frame_endpoint_returns_black_frame_warning_and_jpeg() -> None:
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    app_state = cast(
        "ApplicationState",
        SimpleNamespace(
            vision_module=SimpleNamespace(get_frame=lambda processed: frame)
        ),
    )

    response = await diagnostics.get_raw_camera_frame(app_state)

    assert response.media_type == "image/jpeg"
    assert bytes(response.body).startswith(b"\xff\xd8")
    assert "12x8" in response.headers["X-Frame-Stats"]
    assert "mostly black" in response.headers["X-Frame-Stats"]


@pytest.mark.asyncio
async def test_raw_frame_endpoint_reports_missing_module_and_frame() -> None:
    with pytest.raises(HTTPException) as no_module:
        await diagnostics.get_raw_camera_frame(
            cast("ApplicationState", SimpleNamespace())
        )
    assert no_module.value.status_code == 503
    assert "not initialized" in no_module.value.detail

    app_state = cast(
        "ApplicationState",
        SimpleNamespace(
            vision_module=SimpleNamespace(get_frame=lambda processed: None)
        ),
    )
    with pytest.raises(HTTPException) as no_frame:
        await diagnostics.get_raw_camera_frame(app_state)
    assert no_frame.value.status_code == 503
    assert no_frame.value.detail == "No frame available from camera"


@pytest.mark.asyncio
async def test_raw_frame_endpoint_handles_grayscale_and_invalid_provider_frame() -> (
    None
):
    gray_frame = np.full((5, 7), 80, dtype=np.uint8)
    app_state = cast(
        "ApplicationState",
        SimpleNamespace(
            vision_module=SimpleNamespace(get_frame=lambda processed: gray_frame)
        ),
    )
    response = await diagnostics.get_raw_camera_frame(app_state)
    assert response.media_type == "image/jpeg"
    assert "channels': 1" in response.headers["X-Frame-Stats"]
    assert "valid content" in response.headers["X-Frame-Stats"]

    malformed_state = cast(
        "ApplicationState",
        SimpleNamespace(
            vision_module=SimpleNamespace(get_frame=lambda processed: object())
        ),
    )
    with pytest.raises(HTTPException) as malformed:
        await diagnostics.get_raw_camera_frame(malformed_state)
    assert malformed.value.status_code == 500
    assert "shape" in malformed.value.detail


@pytest.mark.asyncio
async def test_raw_frame_endpoint_preserves_explicit_http_errors() -> None:
    def provider_failure(*, processed: bool) -> Any:
        assert processed is False
        raise HTTPException(status_code=503, detail="camera stopped")

    app_state = cast(
        "ApplicationState",
        SimpleNamespace(vision_module=SimpleNamespace(get_frame=provider_failure)),
    )

    with pytest.raises(HTTPException) as error:
        await diagnostics.get_raw_camera_frame(app_state)

    assert error.value.status_code == 503
    assert error.value.detail == "camera stopped"
