"""Behavioral contracts for diagnostics routes without real hardware/network use."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.api.routes import diagnostics


class _Psutil:
    """Small deterministic psutil surface for route contract tests."""

    def cpu_freq(self) -> Any:
        return SimpleNamespace(current=2400.0)

    def cpu_count(self, logical: bool = True) -> int:
        return 8 if logical else 4

    def cpu_percent(self, interval: float | None = None) -> float:
        return 23.0

    def virtual_memory(self) -> Any:
        return SimpleNamespace(
            total=16 * 1024**3,
            available=10 * 1024**3,
            used=6 * 1024**3,
            percent=37.5,
        )

    def disk_usage(self, path: str) -> Any:
        assert path == "/"
        return SimpleNamespace(total=1000, used=400, free=600)

    def net_if_addrs(self) -> dict[str, list[Any]]:
        family = SimpleNamespace(name="AF_INET")
        return {
            "lo": [SimpleNamespace(address="127.0.0.1", family=family)],
            "eth0": [SimpleNamespace(address="192.0.2.10", family=family)],
        }


@pytest.mark.asyncio
async def test_rust_provider_blocks_diagnostics_that_open_a_python_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_app_state",
        lambda: SimpleNamespace(perception_provider="rust"),
    )

    for route in (
        diagnostics.diagnose_camera,
        diagnostics.diagnose_ball_detection,
        diagnostics.diagnose_vision_system,
        diagnostics.get_fisheye_test_image,
    ):
        with pytest.raises(HTTPException) as error:
            await route()
        assert error.value.status_code == 503
        assert "while Rust owns live capture" in error.value.detail


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep simulated diagnostic delays from slowing contract tests."""

    async def sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(diagnostics.asyncio, "sleep", sleep)


@pytest.mark.asyncio
async def test_hardware_route_reports_supported_resources_and_simulated_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", _Psutil())

    result = await diagnostics.detect_hardware()

    by_id = {device.id: device for device in result}
    assert {
        "cpu",
        "memory",
        "disk",
        "network_eth0",
        "camera_overhead",
        "projector_main",
    } <= set(by_id)
    assert "network_lo" not in by_id
    assert by_id["cpu"].details == {
        "cores": 4,
        "logical_cores": 8,
        "frequency": 2400.0,
        "usage": 23.0,
    }
    assert by_id["memory"].details["percentage"] == 37.5
    assert by_id["disk"].details["percentage"] == 40.0
    assert by_id["network_eth0"].details == {
        "addresses": ["192.0.2.10"],
        "family": ["AF_INET"],
    }
    assert all(isinstance(device, diagnostics.HardwareDevice) for device in result)


@pytest.mark.asyncio
async def test_hardware_route_omits_unavailable_psutil_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", None)

    result = await diagnostics.detect_hardware()

    assert [device.id for device in result] == ["camera_overhead", "projector_main"]


@pytest.mark.asyncio
async def test_hardware_route_converts_resource_errors_to_http_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPsutil(_Psutil):
        def virtual_memory(self) -> Any:
            raise OSError("memory counters unavailable")

    monkeypatch.setattr(diagnostics, "psutil", BrokenPsutil())

    with pytest.raises(HTTPException) as error:
        await diagnostics.detect_hardware()

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to detect hardware devices"


@pytest.mark.asyncio
async def test_network_route_returns_stable_endpoint_contract_and_statuses(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    values = iter([0.95, 0.2, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(values))

    result = await diagnostics.test_network_endpoints()

    assert [endpoint.id for endpoint in result] == [
        "local_api",
        "local_ws",
        "external_dns",
        "internet",
    ]
    assert [endpoint.status for endpoint in result] == [
        "online",
        "online",
        "online",
        "offline",
    ]
    assert result[1].status_code == 101
    assert result[3].status_code == 500
    assert result[3].error == "Connection timeout"
    assert all(endpoint.response_time >= 0 for endpoint in result)
    assert all(isinstance(endpoint, diagnostics.NetworkEndpoint) for endpoint in result)


@pytest.mark.asyncio
async def test_bandwidth_route_maps_deterministic_samples_to_result_fields(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    values = iter([0.0, 0.5, 1.0, 0.25, 0.8])
    monkeypatch.setattr(random, "random", lambda: next(values))

    result = await diagnostics.test_network_bandwidth()

    assert result == diagnostics.BandwidthTestResult(
        download_speed=50.0,
        upload_speed=45.0,
        latency=50.0,
        packet_loss=0.5,
        jitter=4.0,
    )


@pytest.mark.asyncio
async def test_bandwidth_route_wraps_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_sleep(_delay: float) -> None:
        raise OSError("measurement unavailable")

    monkeypatch.setattr(diagnostics.asyncio, "sleep", fail_sleep)

    with pytest.raises(HTTPException) as error:
        await diagnostics.test_network_bandwidth()

    assert error.value.status_code == 500
    detail = cast("dict[str, Any]", error.value.detail)
    assert detail["error"] == "Bandwidth Test Failed"
    assert detail["details"]["error"] == "measurement unavailable"


@pytest.mark.asyncio
async def test_performance_benchmark_returns_cpu_memory_disk_results(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", _Psutil())

    results = await diagnostics.run_performance_benchmark()

    assert [result.test_name for result in results] == [
        "CPU Performance",
        "Memory Performance",
        "Disk Performance",
    ]
    assert results[0].score == 77.0
    assert results[0].passed is True
    assert results[0].metrics == {"cpu_usage": 23.0, "cores": 8.0}
    assert results[1].score == 62.5
    assert results[1].metrics["total_gb"] == 16.0
    assert results[2].score == 94.0
    assert results[2].metrics == {"read_speed": 120, "write_speed": 100}
    assert all(result.duration >= 0 for result in results)
    assert all(
        isinstance(result, diagnostics.PerformanceBenchmark) for result in results
    )


@pytest.mark.asyncio
async def test_performance_benchmark_uses_documented_fallback_without_psutil(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", None)

    results = await diagnostics.run_performance_benchmark()

    assert results[0].score == 80
    assert results[0].metrics == {"cpu_usage": 50, "cores": 4.0}
    assert results[1].score == 75
    assert results[1].metrics == {"memory_usage": 25, "total_gb": 16}


@pytest.mark.asyncio
async def test_performance_benchmark_translates_resource_failure(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    class BrokenPsutil(_Psutil):
        def cpu_percent(self, interval: float | None = None) -> float:
            raise OSError("CPU counter unavailable")

    monkeypatch.setattr(diagnostics, "psutil", BrokenPsutil())

    with pytest.raises(HTTPException) as error:
        await diagnostics.run_performance_benchmark()

    assert error.value.status_code == 500
    detail = cast("dict[str, Any]", error.value.detail)
    assert detail["error"] == "Performance Benchmark Failed"
    assert "CPU counter unavailable" in detail["details"]["error"]


@pytest.mark.asyncio
async def test_system_validation_returns_all_validation_categories(
    monkeypatch: pytest.MonkeyPatch, no_sleep: None
) -> None:
    results = await diagnostics.run_system_validation()

    assert [result.test_id for result in results] == [
        "config_validation",
        "module_communication",
        "data_integrity",
    ]
    assert [result.status for result in results] == ["passed"] * 3
    assert results[0].result["invalid_parameters"] == 0
    assert results[1].result["successful_connections"] == 5
    assert results[2].result["checksum_validation"] == "passed"
    assert all(
        isinstance(result, diagnostics.SystemValidationResult) for result in results
    )


@pytest.mark.asyncio
async def test_system_validation_reports_failure_as_http_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_sleep(_delay: float) -> None:
        raise RuntimeError("validation service failed")

    monkeypatch.setattr(diagnostics.asyncio, "sleep", fail_sleep)

    with pytest.raises(HTTPException) as error:
        await diagnostics.run_system_validation()

    assert error.value.status_code == 500
    detail = cast("dict[str, Any]", error.value.detail)
    assert detail["error"] == "System Validation Failed"
    assert detail["details"]["error"] == "validation service failed"


class _FakeCamera:
    def __init__(self, frame_data: tuple[Any, Any] | None, starts: bool = True):
        self.frame_data = frame_data
        self.starts = starts
        self.stop_calls = 0

    def start_capture(self) -> bool:
        return self.starts

    def get_latest_frame(self) -> tuple[Any, Any] | None:
        return self.frame_data

    def stop_capture(self) -> None:
        self.stop_calls += 1


def test_camera_diagnostic_scores_capture_and_frame_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = SimpleNamespace(shape=(720, 1280, 3))
    camera = _FakeCamera((frame, {"sequence": 4}))

    class CameraCapture:
        def __init__(self, config: dict[str, Any]):
            assert config["buffer_size"] == 1

        def start_capture(self) -> bool:
            return camera.start_capture()

        def get_latest_frame(self) -> tuple[Any, Any] | None:
            return camera.get_latest_frame()

        def stop_capture(self) -> None:
            camera.stop_capture()

    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)
    monkeypatch.setattr(diagnostics.time, "sleep", lambda _seconds: None)

    result = diagnostics._run_camera_diagnostic_sync()

    assert result["score"] == 100.0
    assert result["details"]["frame_size"] == "1280x720"
    assert result["details"]["frame_quality"] == "adequate"
    assert result["issues"] == []
    assert camera.stop_calls == 1


@pytest.mark.parametrize(
    ("starts", "frame_data", "expected_score", "issue"),
    [
        (False, None, 25.0, "Failed to start camera capture"),
        (True, None, 50.0, "Failed to capture frame from camera"),
        (
            True,
            (SimpleNamespace(shape=(240, 320, 3)), None),
            75.0,
            "Frame resolution below recommended minimum",
        ),
    ],
)
def test_camera_diagnostic_reports_camera_and_quality_failures(
    monkeypatch: pytest.MonkeyPatch,
    starts: bool,
    frame_data: tuple[Any, Any] | None,
    expected_score: float,
    issue: str,
) -> None:
    camera = _FakeCamera(frame_data, starts=starts)

    class CameraCapture:
        def __init__(self, _config: dict[str, Any]):
            pass

        def start_capture(self) -> bool:
            return camera.start_capture()

        def get_latest_frame(self) -> tuple[Any, Any] | None:
            return camera.get_latest_frame()

        def stop_capture(self) -> None:
            camera.stop_capture()

    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)
    monkeypatch.setattr(diagnostics.time, "sleep", lambda _seconds: None)

    result = diagnostics._run_camera_diagnostic_sync()

    assert result["score"] == expected_score
    assert issue in result["issues"]
    assert camera.stop_calls == int(starts)


def test_camera_diagnostic_catches_initialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CameraCapture:
        def __init__(self, _config: dict[str, Any]):
            raise OSError("camera driver unavailable")

    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)

    result = diagnostics._run_camera_diagnostic_sync()

    assert result["score"] == 0.0
    assert result["details"]["error"] == "camera driver unavailable"
    assert result["recommendations"] == [
        "Check camera hardware and driver installation"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "expected_status"),
    [(100, "healthy"), (75, "degraded"), (25, "failed"), (0, "error")],
)
async def test_camera_route_maps_diagnostic_score_to_status(
    monkeypatch: pytest.MonkeyPatch, score: float, expected_status: str
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_run_camera_diagnostic_sync",
        lambda: {"score": score, "details": {}, "issues": [], "recommendations": []},
    )

    result = await diagnostics.diagnose_camera()

    assert result.component == "camera"
    assert result.status == expected_status
    assert result.score == score
    assert result.duration >= 0


def test_fisheye_diagnostic_reports_missing_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    monkeypatch.setattr(Path, "exists", lambda _self: False)

    result = diagnostics._run_fisheye_diagnostic_sync()

    assert result["score"] == 0.0
    assert result["details"]["calibration_file_exists"] is False
    assert "Fisheye calibration file not found" in result["issues"]


def test_fisheye_diagnostic_reports_unreadable_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from backend.vision.calibration.camera import CameraCalibrator

    monkeypatch.setattr(Path, "exists", lambda _self: True)
    monkeypatch.setattr(CameraCalibrator, "__init__", lambda _self: None)
    monkeypatch.setattr(
        CameraCalibrator, "load_calibration", lambda _self, _path: False
    )

    result = diagnostics._run_fisheye_diagnostic_sync()

    assert result["score"] == 25.0
    assert result["details"]["calibration_file_exists"] is True
    assert result["details"]["calibration_loaded"] is False
    assert "Failed to load calibration file" in result["issues"]


def test_detection_diagnostic_reports_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _FakeCamera(None, starts=False)

    class CameraCapture:
        def __init__(self, _config: dict[str, Any]):
            pass

        def start_capture(self) -> bool:
            return camera.start_capture()

    class BallDetector:
        def __init__(self, _config: dict[str, Any]):
            pass

    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)
    monkeypatch.setattr("backend.vision.detection.balls.BallDetector", BallDetector)

    result = diagnostics._run_detection_diagnostic_sync()

    assert result["score"] == 25.0
    assert "Failed to start camera for detection test" in result["issues"]


def test_detection_diagnostic_reports_empty_frame_and_stops_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _FakeCamera(None)

    class CameraCapture:
        def __init__(self, _config: dict[str, Any]):
            pass

        def start_capture(self) -> bool:
            return camera.start_capture()

        def get_latest_frame(self) -> tuple[Any, Any] | None:
            return camera.get_latest_frame()

        def stop_capture(self) -> None:
            camera.stop_capture()

    class BallDetector:
        def __init__(self, _config: dict[str, Any]):
            pass

    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)
    monkeypatch.setattr("backend.vision.detection.balls.BallDetector", BallDetector)
    monkeypatch.setattr(diagnostics.time, "sleep", lambda _seconds: None)

    result = diagnostics._run_detection_diagnostic_sync()

    assert result["score"] == 25.0, result
    assert "Failed to capture test frame" in result["issues"]
    assert camera.stop_calls == 1


def test_detection_diagnostic_reports_confidence_and_slow_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = SimpleNamespace(shape=(720, 1280, 3))
    camera = _FakeCamera((frame, {"sequence": 9}))
    balls = [SimpleNamespace(confidence=0.2), SimpleNamespace(confidence=0.8)]

    class CameraCapture:
        def __init__(self, _config: dict[str, Any]):
            pass

        def start_capture(self) -> bool:
            return camera.start_capture()

        def get_latest_frame(self) -> tuple[Any, Any] | None:
            return camera.get_latest_frame()

        def stop_capture(self) -> None:
            camera.stop_capture()

    class BallDetector:
        def __init__(self, _config: dict[str, Any]):
            pass

        def detect_balls(self, _frame: Any) -> list[Any]:
            return balls

    times = iter([1.0, 1.2])
    monkeypatch.setattr("backend.vision.capture.CameraCapture", CameraCapture)
    monkeypatch.setattr("backend.vision.detection.balls.BallDetector", BallDetector)
    monkeypatch.setattr(diagnostics.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(diagnostics.time, "time", lambda: next(times))

    result = diagnostics._run_detection_diagnostic_sync()

    assert result["score"] == 100.0
    assert result["details"]["balls_detected"] == 2
    assert result["details"]["avg_confidence"] == 0.5
    assert result["details"]["min_confidence"] == 0.2
    assert result["details"]["max_confidence"] == 0.8
    assert result["details"]["detection_time_ms"] == 200.0
    assert "Low detection confidence" in result["issues"]
    assert any("slower than expected" in issue for issue in result["issues"])
    assert camera.stop_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("score", "expected_status"),
    [(100, "healthy"), (70, "degraded"), (20, "failed"), (0, "error")],
)
async def test_detection_route_maps_diagnostic_score_to_status(
    monkeypatch: pytest.MonkeyPatch, score: float, expected_status: str
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_run_detection_diagnostic_sync",
        lambda: {"score": score, "details": {}, "issues": [], "recommendations": []},
    )

    result = await diagnostics.diagnose_ball_detection()

    assert result.component == "ball_detection"
    assert result.status == expected_status


@pytest.mark.asyncio
async def test_vision_complete_runs_all_three_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def camera() -> diagnostics.VisionDiagnosticResult:
        return diagnostics.VisionDiagnosticResult(
            component="camera",
            status="healthy",
            score=100,
            details={},
            issues=[],
            recommendations=[],
            duration=0,
        )

    async def fisheye() -> diagnostics.VisionDiagnosticResult:
        return diagnostics.VisionDiagnosticResult(
            component="fisheye_correction",
            status="failed",
            score=0,
            details={},
            issues=[],
            recommendations=[],
            duration=0,
        )

    async def detection() -> diagnostics.VisionDiagnosticResult:
        return diagnostics.VisionDiagnosticResult(
            component="ball_detection",
            status="degraded",
            score=70,
            details={},
            issues=[],
            recommendations=[],
            duration=0,
        )

    monkeypatch.setattr(diagnostics, "diagnose_camera", camera)
    monkeypatch.setattr(diagnostics, "diagnose_fisheye_correction", fisheye)
    monkeypatch.setattr(diagnostics, "diagnose_ball_detection", detection)

    results = await diagnostics.diagnose_vision_system()

    assert [item.component for item in results] == [
        "camera",
        "fisheye_correction",
        "ball_detection",
    ]
    assert [item.status for item in results] == ["healthy", "failed", "degraded"]


@pytest.mark.asyncio
async def test_fisheye_image_route_returns_jpeg_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics, "_generate_fisheye_test_image", lambda: b"jpeg-data"
    )

    response = await diagnostics.get_fisheye_test_image()

    assert response.media_type == "image/jpeg"
    assert response.body == b"jpeg-data"


@pytest.mark.asyncio
async def test_fisheye_image_route_reports_missing_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "_generate_fisheye_test_image", lambda: None)

    with pytest.raises(HTTPException) as error:
        await diagnostics.get_fisheye_test_image()

    # The endpoint currently wraps its own 500 in the generic error handler.
    assert error.value.status_code == 500


@pytest.mark.asyncio
async def test_fisheye_image_route_maps_executor_timeout_to_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedOutLoop:
        def run_in_executor(
            self, _executor: Any, _function: Any
        ) -> asyncio.Future[Any]:
            return asyncio.get_running_loop().create_future()

    monkeypatch.setattr(diagnostics.asyncio, "get_event_loop", lambda: TimedOutLoop())
    monkeypatch.setattr(diagnostics.asyncio, "wait_for", _timeout_immediately)

    with pytest.raises(HTTPException) as error:
        await diagnostics.get_fisheye_test_image()

    assert error.value.status_code == 504


async def _timeout_immediately(_awaitable: Any, timeout: float) -> Any:
    assert timeout == 10.0
    raise asyncio.TimeoutError


@pytest.mark.asyncio
async def test_summary_serializes_aggregate_status_and_platform_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", _Psutil())

    result = await diagnostics.get_diagnostic_summary()

    assert result.total_tests == 15
    assert result.passed_tests == 13
    assert result.failed_tests == 1
    assert result.warnings == 1
    assert result.overall_score == pytest.approx(86.6667, abs=0.001)
    assert result.system_info["cpu_cores"] == 8
    assert result.system_info["memory_gb"] == 16.0
    assert result.system_info["disk_gb"] == 0.0
    assert len(result.recommendations) == 3
    assert result.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_summary_works_when_psutil_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "psutil", None)

    result = await diagnostics.get_diagnostic_summary()

    assert result.system_info["platform"] == "Unknown"
    assert "cpu_cores" not in result.system_info
    assert result.recommendations


def test_network_endpoint_model_rejects_missing_required_identity() -> None:
    with pytest.raises(ValidationError):
        diagnostics.NetworkEndpoint.model_validate(
            {
                "name": "Local API",
                "url": "http://localhost/health",
                "method": "GET",
                "status": "online",
                "response_time": 1.0,
            }
        )
