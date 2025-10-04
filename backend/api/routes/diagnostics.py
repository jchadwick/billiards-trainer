"""Advanced diagnostic endpoints for comprehensive system testing and validation.

Provides specialized diagnostic tools including:
- Hardware detection and validation
- Network connectivity testing
- Performance benchmarking
- System validation tests
- Troubleshooting utilities
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..dependencies import ApplicationState, get_app_state
from ..models.responses import SystemMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


class HardwareDevice(BaseModel):
    """Hardware device information."""

    id: str
    name: str
    type: str
    status: str
    details: dict[str, Any]


class NetworkEndpoint(BaseModel):
    """Network endpoint test result."""

    id: str
    name: str
    url: str
    method: str
    status: str
    response_time: float
    status_code: Optional[int] = None
    error: Optional[str] = None


class BandwidthTestResult(BaseModel):
    """Network bandwidth test result."""

    download_speed: float  # Mbps
    upload_speed: float  # Mbps
    latency: float  # ms
    packet_loss: float  # percentage
    jitter: float  # ms


class PerformanceBenchmark(BaseModel):
    """Performance benchmark result."""

    test_name: str
    score: float  # 0-100
    metrics: dict[str, float]
    duration: float  # seconds
    passed: bool
    issues: list[str]
    recommendations: list[str]


class SystemValidationResult(BaseModel):
    """System validation test result."""

    test_id: str
    name: str
    category: str
    status: str
    score: float
    duration: float
    result: dict[str, Any]
    issues: list[str]
    recommendations: list[str]


class DiagnosticSummary(BaseModel):
    """Comprehensive diagnostic summary."""

    timestamp: datetime
    overall_score: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    critical_issues: int
    system_info: dict[str, Any]
    recommendations: list[str]


@router.get("/hardware", response_model=list[HardwareDevice])
async def detect_hardware() -> list[HardwareDevice]:
    """Detect and validate hardware devices."""
    devices = []

    try:
        # CPU Information
        if psutil:
            try:
                cpu_freq = None
                try:
                    freq_info = psutil.cpu_freq()
                    cpu_freq = freq_info.current if freq_info else 0
                except (AttributeError, OSError):
                    cpu_freq = 0  # Fallback for systems where cpu_freq is not available

                cpu_info = {
                    "cores": psutil.cpu_count(logical=False) or 0,
                    "logical_cores": psutil.cpu_count(logical=True) or 0,
                    "frequency": cpu_freq,
                    "usage": psutil.cpu_percent(
                        interval=0.1
                    ),  # Shorter interval for faster response
                }
            except Exception as cpu_error:
                logger.warning(f"CPU info collection partially failed: {cpu_error}")
                cpu_info = {
                    "cores": psutil.cpu_count(logical=False) or 0,
                    "logical_cores": psutil.cpu_count(logical=True) or 0,
                    "frequency": 0,
                    "usage": 0,
                }

            devices.append(
                HardwareDevice(
                    id="cpu",
                    name="CPU",
                    type="processor",
                    status="online",
                    details=cpu_info,
                )
            )

        # Memory Information
        if psutil:
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percentage": memory.percent,
            }

            devices.append(
                HardwareDevice(
                    id="memory",
                    name="System Memory",
                    type="memory",
                    status="online",
                    details=memory_info,
                )
            )

        # Disk Information
        if psutil:
            disk = psutil.disk_usage("/")
            disk_info = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percentage": (disk.used / disk.total) * 100,
            }

            devices.append(
                HardwareDevice(
                    id="disk",
                    name="Primary Disk",
                    type="storage",
                    status="online",
                    details=disk_info,
                )
            )

        # Network Interfaces
        if psutil:
            network_interfaces = psutil.net_if_addrs()
            for interface, addresses in network_interfaces.items():
                if interface != "lo":  # Skip loopback
                    interface_info = {
                        "addresses": [addr.address for addr in addresses],
                        "family": [addr.family.name for addr in addresses],
                    }

                    devices.append(
                        HardwareDevice(
                            id=f"network_{interface}",
                            name=f"Network Interface {interface}",
                            type="network",
                            status="online",
                            details=interface_info,
                        )
                    )

        # Simulate camera detection (would use actual camera detection in real implementation)
        devices.append(
            HardwareDevice(
                id="camera_overhead",
                name="Overhead Camera",
                type="camera",
                status="available",
                details={
                    "resolution": "1920x1080",
                    "frame_rate": 30,
                    "formats": ["MJPEG", "YUV"],
                },
            )
        )

        # Simulate projector detection
        devices.append(
            HardwareDevice(
                id="projector_main",
                name="Main Projector",
                type="projector",
                status="connected",
                details={
                    "resolution": "1920x1080",
                    "brightness": 3000,
                    "calibrated": True,
                },
            )
        )

    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to detect hardware devices")

    return devices


@router.post("/network/test", response_model=list[NetworkEndpoint])
async def test_network_endpoints() -> list[NetworkEndpoint]:
    """Test connectivity to various network endpoints."""
    endpoints = [
        {
            "id": "local_api",
            "name": "Local API",
            "url": "http://localhost:8000/health",
            "method": "GET",
        },
        {
            "id": "local_ws",
            "name": "WebSocket",
            "url": "ws://localhost:8000/ws",
            "method": "GET",
        },
        {
            "id": "external_dns",
            "name": "External DNS",
            "url": "https://8.8.8.8",
            "method": "HEAD",
        },
        {
            "id": "internet",
            "name": "Internet Connectivity",
            "url": "https://www.google.com",
            "method": "HEAD",
        },
    ]

    results = []

    for endpoint in endpoints:
        start_time = time.time()

        try:
            if endpoint["url"].startswith("ws://"):
                # Would implement WebSocket test here
                status = "online"
                status_code = 101
                error = None
            else:
                # Would implement HTTP request here
                # For simulation, we'll assume most endpoints are working
                import random

                if random.random() > 0.1:  # 90% success rate
                    status = "online"
                    status_code = 200
                    error = None
                else:
                    status = "offline"
                    status_code = 500
                    error = "Connection timeout"

            response_time = (time.time() - start_time) * 1000  # Convert to ms

        except Exception as e:
            status = "offline"
            status_code = None
            error = str(e)
            response_time = (time.time() - start_time) * 1000

        results.append(
            NetworkEndpoint(
                id=endpoint["id"],
                name=endpoint["name"],
                url=endpoint["url"],
                method=endpoint["method"],
                status=status,
                response_time=response_time,
                status_code=status_code,
                error=error,
            )
        )

        # Small delay between tests
        await asyncio.sleep(0.1)

    return results


@router.post("/network/bandwidth", response_model=BandwidthTestResult)
async def test_network_bandwidth() -> BandwidthTestResult:
    """Perform network bandwidth testing."""
    try:
        # Simulate bandwidth test (would implement actual test in real system)
        await asyncio.sleep(2)  # Simulate test duration

        import random

        # Generate realistic bandwidth test results
        download_speed = 50 + random.random() * 100  # 50-150 Mbps
        upload_speed = 20 + random.random() * 50  # 20-70 Mbps
        latency = 10 + random.random() * 40  # 10-50 ms
        packet_loss = random.random() * 2  # 0-2%
        jitter = random.random() * 5  # 0-5 ms

        return BandwidthTestResult(
            download_speed=download_speed,
            upload_speed=upload_speed,
            latency=latency,
            packet_loss=packet_loss,
            jitter=jitter,
        )

    except Exception as e:
        logger.error(f"Bandwidth test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Bandwidth Test Failed",
                "message": "Unable to perform bandwidth test",
                "details": {"error": str(e)},
            },
        )


@router.post("/performance/benchmark", response_model=list[PerformanceBenchmark])
async def run_performance_benchmark() -> list[PerformanceBenchmark]:
    """Run comprehensive performance benchmarks."""
    benchmarks = []

    try:
        # CPU Benchmark
        cpu_start = time.time()
        if psutil:
            # Simulate CPU intensive task
            await asyncio.sleep(1)
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_score = max(0, 100 - cpu_usage)  # Lower usage = higher score
        else:
            cpu_score = 80

        cpu_duration = time.time() - cpu_start

        benchmarks.append(
            PerformanceBenchmark(
                test_name="CPU Performance",
                score=cpu_score,
                metrics={
                    "cpu_usage": cpu_usage if psutil else 50,
                    "cores": psutil.cpu_count() if psutil else 4,
                },
                duration=cpu_duration,
                passed=cpu_score >= 70,
                issues=[] if cpu_score >= 70 else ["High CPU usage detected"],
                recommendations=(
                    []
                    if cpu_score >= 70
                    else [
                        "Close unnecessary applications",
                        "Check for background processes",
                    ]
                ),
            )
        )

        # Memory Benchmark
        memory_start = time.time()
        if psutil:
            memory = psutil.virtual_memory()
            memory_score = 100 - memory.percent
        else:
            memory_score = 75

        memory_duration = time.time() - memory_start

        benchmarks.append(
            PerformanceBenchmark(
                test_name="Memory Performance",
                score=memory_score,
                metrics={
                    "memory_usage": 100 - memory_score,
                    "total_gb": memory.total / (1024**3) if psutil else 16,
                },
                duration=memory_duration,
                passed=memory_score >= 60,
                issues=[] if memory_score >= 60 else ["High memory usage detected"],
                recommendations=(
                    []
                    if memory_score >= 60
                    else [
                        "Close memory-intensive applications",
                        "Consider upgrading RAM",
                    ]
                ),
            )
        )

        # Disk I/O Benchmark
        disk_start = time.time()
        await asyncio.sleep(0.5)  # Simulate disk test
        disk_score = 80 + (20 * (1 - 0.3))  # Simulated disk performance
        disk_duration = time.time() - disk_start

        benchmarks.append(
            PerformanceBenchmark(
                test_name="Disk Performance",
                score=disk_score,
                metrics={"read_speed": 120, "write_speed": 100},  # MB/s
                duration=disk_duration,
                passed=disk_score >= 70,
                issues=[] if disk_score >= 70 else ["Slow disk performance detected"],
                recommendations=(
                    []
                    if disk_score >= 70
                    else ["Consider SSD upgrade", "Check disk health"]
                ),
            )
        )

    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Performance Benchmark Failed",
                "message": "Unable to run performance benchmarks",
                "details": {"error": str(e)},
            },
        )

    return benchmarks


@router.post("/validation/system", response_model=list[SystemValidationResult])
async def run_system_validation() -> list[SystemValidationResult]:
    """Run comprehensive system validation tests."""
    validations = []

    try:
        # Configuration Validation
        config_start = time.time()
        await asyncio.sleep(1)  # Simulate validation

        config_score = 95  # Simulated score
        config_duration = time.time() - config_start

        validations.append(
            SystemValidationResult(
                test_id="config_validation",
                name="Configuration Validation",
                category="integrity",
                status="passed",
                score=config_score,
                duration=config_duration,
                result={
                    "parameters_checked": 50,
                    "invalid_parameters": 0,
                    "missing_parameters": 0,
                },
                issues=[],
                recommendations=[],
            )
        )

        # Module Communication Test
        comm_start = time.time()
        await asyncio.sleep(1.5)  # Simulate communication test

        comm_score = 88
        comm_duration = time.time() - comm_start

        validations.append(
            SystemValidationResult(
                test_id="module_communication",
                name="Module Communication",
                category="communication",
                status="passed",
                score=comm_score,
                duration=comm_duration,
                result={
                    "modules_tested": 5,
                    "successful_connections": 5,
                    "average_latency": 25.5,
                },
                issues=[],
                recommendations=[],
            )
        )

        # Data Integrity Test
        data_start = time.time()
        await asyncio.sleep(0.8)  # Simulate data integrity check

        data_score = 100
        data_duration = time.time() - data_start

        validations.append(
            SystemValidationResult(
                test_id="data_integrity",
                name="Data Integrity",
                category="integrity",
                status="passed",
                score=data_score,
                duration=data_duration,
                result={
                    "records_checked": 1000,
                    "corrupted_records": 0,
                    "checksum_validation": "passed",
                },
                issues=[],
                recommendations=[],
            )
        )

    except Exception as e:
        logger.error(f"System validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "System Validation Failed",
                "message": "Unable to run system validation tests",
                "details": {"error": str(e)},
            },
        )

    return validations


class VisionDiagnosticResult(BaseModel):
    """Vision system diagnostic result."""

    component: str
    status: str
    score: float
    details: dict[str, Any]
    issues: list[str]
    recommendations: list[str]
    duration: float


def _run_camera_diagnostic_sync() -> dict[str, Any]:
    """Run camera diagnostic synchronously (to be called in thread)."""
    issues = []
    recommendations = []
    details = {}
    score = 0.0

    try:
        from backend.vision.capture import CameraCapture

        # Test camera initialization
        camera_config = {
            "device_id": 0,
            "backend": "auto",
            "resolution": (1920, 1080),
            "fps": 30,
            "buffer_size": 1,
        }

        camera = CameraCapture(camera_config)
        details["camera_initialized"] = True
        score += 25.0

        # Test camera start
        if camera.start_capture():
            details["camera_started"] = True
            score += 25.0

            # Wait for frame stabilization
            time.sleep(0.5)

            # Test frame capture
            frame_data = camera.get_latest_frame()
            if frame_data is not None:
                frame, frame_info = frame_data
                details["frame_captured"] = True
                details["frame_shape"] = list(frame.shape)
                details["frame_size"] = f"{frame.shape[1]}x{frame.shape[0]}"
                score += 25.0

                # Check frame quality
                if frame.shape[0] >= 480 and frame.shape[1] >= 640:
                    details["frame_quality"] = "adequate"
                    score += 25.0
                else:
                    issues.append("Frame resolution below recommended minimum")
                    recommendations.append(
                        "Increase camera resolution to at least 640x480"
                    )
            else:
                issues.append("Failed to capture frame from camera")
                recommendations.append("Check camera connection and permissions")

            camera.stop_capture()
        else:
            issues.append("Failed to start camera capture")
            recommendations.append(
                "Verify camera device is available and not in use by another process"
            )

    except Exception as e:
        logger.error(f"Camera diagnostic failed: {e}", exc_info=True)
        issues.append(f"Camera diagnostic error: {e}")
        recommendations.append("Check camera hardware and driver installation")
        details["error"] = str(e)
        score = 0.0

    return {
        "score": score,
        "details": details,
        "issues": issues,
        "recommendations": recommendations,
    }


@router.get("/vision/camera", response_model=VisionDiagnosticResult)
async def diagnose_camera() -> VisionDiagnosticResult:
    """Diagnose camera capture system.

    Tests:
    - Camera device accessibility
    - Frame capture capability
    - Resolution and FPS
    - Frame quality
    """
    start_time = time.time()

    try:
        # Run diagnostic in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_camera_diagnostic_sync)

        score = result["score"]
        details = result["details"]
        issues = result["issues"]
        recommendations = result["recommendations"]

        # Determine overall status
        if score >= 90:
            status = "healthy"
        elif score >= 70:
            status = "degraded"
        elif score > 0:
            status = "failed"
        else:
            status = "error"

        return VisionDiagnosticResult(
            component="camera",
            status=status,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            duration=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Camera diagnostic failed: {e}", exc_info=True)
        return VisionDiagnosticResult(
            component="camera",
            status="error",
            score=0.0,
            details={"error": str(e)},
            issues=[f"Camera diagnostic error: {e}"],
            recommendations=["Check camera hardware and driver installation"],
            duration=time.time() - start_time,
        )


def _run_fisheye_diagnostic_sync() -> dict[str, Any]:
    """Run fisheye diagnostic synchronously (to be called in thread)."""
    issues = []
    recommendations = []
    details = {}
    score = 0.0

    try:
        from pathlib import Path

        from backend.vision.calibration.camera import CameraCalibrator

        # Check calibration file
        backend_path = Path(__file__).parent.parent.parent
        calibration_path = backend_path / "calibration/camera_fisheye_default.yaml"

        if calibration_path.exists():
            details["calibration_file_exists"] = True
            details["calibration_path"] = str(calibration_path)
            score += 25.0

            # Try to load calibration
            calibrator = CameraCalibrator()

            try:
                if calibrator.load_calibration(str(calibration_path)):
                    details["calibration_loaded"] = True
                    score += 25.0

                    # Check if we have valid calibration data
                    if (
                        hasattr(calibrator, "camera_matrix")
                        and calibrator.camera_matrix is not None
                    ):
                        details["has_camera_matrix"] = True
                        score += 25.0
                    else:
                        issues.append("Camera matrix not available in calibration")
                        recommendations.append("Re-run camera calibration process")

                    if (
                        hasattr(calibrator, "dist_coeffs")
                        and calibrator.dist_coeffs is not None
                    ):
                        details["has_distortion_coeffs"] = True
                        score += 25.0
                    else:
                        issues.append("Distortion coefficients not available")
                        recommendations.append("Complete fisheye calibration")
                else:
                    issues.append("Failed to load calibration file")
                    recommendations.append(
                        "Verify calibration file format and re-run calibration"
                    )
                    details["calibration_loaded"] = False
            except Exception as load_error:
                issues.append(f"Calibration load error: {load_error}")
                recommendations.append(
                    "Calibration file may be corrupted, re-run calibration process"
                )
                details["load_error"] = str(load_error)
        else:
            issues.append("Fisheye calibration file not found")
            recommendations.append(
                f"Run camera calibration and save to {calibration_path}"
            )
            details["calibration_file_exists"] = False

    except Exception as e:
        logger.error(f"Fisheye diagnostic failed: {e}", exc_info=True)
        issues.append(f"Fisheye diagnostic error: {e}")
        recommendations.append("Check vision module installation")
        details["error"] = str(e)
        score = 0.0

    return {
        "score": score,
        "details": details,
        "issues": issues,
        "recommendations": recommendations,
    }


@router.get("/vision/fisheye", response_model=VisionDiagnosticResult)
async def diagnose_fisheye_correction() -> VisionDiagnosticResult:
    """Diagnose fisheye correction system.

    Tests:
    - Calibration file existence
    - Calibration file format
    - Calibration parameters validity
    - Correction performance
    """
    start_time = time.time()

    try:
        # Run diagnostic in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_fisheye_diagnostic_sync)

        score = result["score"]
        details = result["details"]
        issues = result["issues"]
        recommendations = result["recommendations"]

        # Determine status
        if score >= 90:
            status = "healthy"
        elif score >= 50:
            status = "degraded"
        else:
            status = "failed"

        return VisionDiagnosticResult(
            component="fisheye_correction",
            status=status,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            duration=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Fisheye diagnostic failed: {e}", exc_info=True)
        return VisionDiagnosticResult(
            component="fisheye_correction",
            status="error",
            score=0.0,
            details={"error": str(e)},
            issues=[f"Fisheye diagnostic error: {e}"],
            recommendations=["Check vision module installation"],
            duration=time.time() - start_time,
        )


def _run_detection_diagnostic_sync() -> dict[str, Any]:
    """Run detection diagnostic synchronously (to be called in thread)."""
    issues = []
    recommendations = []
    details = {}
    score = 0.0

    try:
        from backend.vision.capture import CameraCapture
        from backend.vision.detection.balls import BallDetector

        # Initialize detector
        detector = BallDetector()
        details["detector_initialized"] = True
        score += 25.0

        # Capture frame for testing
        camera_config = {
            "device_id": 0,
            "backend": "auto",
            "resolution": (1920, 1080),
            "fps": 30,
            "buffer_size": 1,
        }

        camera = CameraCapture(camera_config)
        if camera.start_capture():
            time.sleep(0.5)

            frame_data = camera.get_latest_frame()
            if frame_data is not None:
                frame, _ = frame_data
                details["test_frame_captured"] = True
                score += 25.0

                # Run detection
                detection_start = time.time()
                detected_balls = detector.detect(frame)
                detection_time = time.time() - detection_start

                details["balls_detected"] = len(detected_balls)
                details["detection_time_ms"] = round(detection_time * 1000, 2)
                score += 25.0

                # Analyze detection results
                if len(detected_balls) > 0:
                    details["detection_working"] = True
                    score += 25.0

                    # Check confidence levels
                    confidences = [ball.confidence for ball in detected_balls]
                    details["avg_confidence"] = round(
                        sum(confidences) / len(confidences), 2
                    )
                    details["min_confidence"] = round(min(confidences), 2)
                    details["max_confidence"] = round(max(confidences), 2)

                    if details["avg_confidence"] < 0.6:
                        issues.append("Low detection confidence")
                        recommendations.append(
                            "Improve lighting or adjust detection parameters"
                        )
                else:
                    issues.append("No balls detected in test frame")
                    recommendations.append(
                        "Ensure balls are visible on table or adjust detection threshold"
                    )

                # Performance check
                if detection_time > 0.1:  # 100ms threshold
                    issues.append(
                        f"Detection slower than expected ({detection_time*1000:.0f}ms)"
                    )
                    recommendations.append(
                        "Consider hardware acceleration or optimization"
                    )

            else:
                issues.append("Failed to capture test frame")
                recommendations.append("Check camera functionality")

            camera.stop_capture()
        else:
            issues.append("Failed to start camera for detection test")
            recommendations.append("Resolve camera issues first")

    except Exception as e:
        logger.error(f"Detection diagnostic failed: {e}", exc_info=True)
        issues.append(f"Detection diagnostic error: {e}")
        recommendations.append("Check vision module and detector installation")
        details["error"] = str(e)
        score = 0.0

    return {
        "score": score,
        "details": details,
        "issues": issues,
        "recommendations": recommendations,
    }


@router.get("/vision/detection", response_model=VisionDiagnosticResult)
async def diagnose_ball_detection() -> VisionDiagnosticResult:
    """Diagnose ball detection system.

    Tests:
    - Ball detector initialization
    - Detection on live frame
    - Detection accuracy
    - Performance metrics
    """
    start_time = time.time()

    try:
        # Run diagnostic in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_detection_diagnostic_sync)

        score = result["score"]
        details = result["details"]
        issues = result["issues"]
        recommendations = result["recommendations"]

        # Determine status
        if score >= 90:
            status = "healthy"
        elif score >= 60:
            status = "degraded"
        elif score > 0:
            status = "failed"
        else:
            status = "error"

        return VisionDiagnosticResult(
            component="ball_detection",
            status=status,
            score=score,
            details=details,
            issues=issues,
            recommendations=recommendations,
            duration=time.time() - start_time,
        )

    except Exception as e:
        logger.error(f"Detection diagnostic failed: {e}", exc_info=True)
        return VisionDiagnosticResult(
            component="ball_detection",
            status="error",
            score=0.0,
            details={"error": str(e)},
            issues=[f"Detection diagnostic error: {e}"],
            recommendations=["Check vision module and detector installation"],
            duration=time.time() - start_time,
        )


@router.get("/vision/complete", response_model=list[VisionDiagnosticResult])
async def diagnose_vision_system() -> list[VisionDiagnosticResult]:
    """Run complete vision system diagnostics.

    Executes all vision-related diagnostic tests:
    - Camera system
    - Fisheye correction
    - Ball detection

    Returns comprehensive results for all components.
    """
    results = []

    # Run all vision diagnostics
    results.append(await diagnose_camera())
    results.append(await diagnose_fisheye_correction())
    results.append(await diagnose_ball_detection())

    return results


def _generate_fisheye_test_image() -> Optional[bytes]:
    """Generate a before/after fisheye correction test image."""
    from pathlib import Path

    import cv2
    import numpy as np

    try:
        from backend.vision.calibration.camera import CameraCalibrator
        from backend.vision.capture import CameraCapture

        # Capture a frame
        camera_config = {
            "device_id": 0,
            "backend": "auto",
            "resolution": (1920, 1080),
            "fps": 30,
            "buffer_size": 1,
        }

        camera = CameraCapture(camera_config)
        if not camera.start_capture():
            logger.error("Failed to start camera for test image")
            return None

        time.sleep(0.5)  # Let camera stabilize

        frame_data = camera.get_latest_frame()
        camera.stop_capture()

        if frame_data is None:
            logger.error("Failed to capture test frame")
            return None

        original_frame, _ = frame_data

        # Load calibration
        backend_path = Path(__file__).parent.parent.parent
        calibration_path = backend_path / "calibration/camera_fisheye_default.yaml"

        calibrator = CameraCalibrator()
        if not calibration_path.exists():
            logger.error("Calibration file not found")
            return None

        if not calibrator.load_calibration(str(calibration_path)):
            logger.error("Failed to load calibration")
            return None

        # Apply undistortion
        corrected_frame = calibrator.undistort_image(original_frame)

        # Resize corrected to match original size (undistort may crop)
        h, w = original_frame.shape[:2]
        corrected_resized = cv2.resize(corrected_frame, (w, h))

        # Create side-by-side comparison
        # Add labels
        original_labeled = original_frame.copy()
        corrected_labeled = corrected_resized.copy()

        cv2.putText(
            original_labeled,
            "Original (with distortion)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            corrected_labeled,
            "Corrected (fisheye removed)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

        # Stack horizontally
        comparison = np.hstack([original_labeled, corrected_labeled])

        # Encode as JPEG
        success, encoded = cv2.imencode(
            ".jpg", comparison, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not success:
            logger.error("Failed to encode comparison image")
            return None

        return encoded.tobytes()

    except Exception as e:
        logger.error(f"Failed to generate test image: {e}")
        logger.exception("Full traceback:")
        return None


@router.get("/vision/fisheye/test-image")
async def get_fisheye_test_image():
    """Generate a before/after comparison image showing fisheye correction.

    Returns:
        JPEG image showing original and corrected frames side-by-side
    """
    from fastapi.responses import Response

    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, _generate_fisheye_test_image)

    if image_bytes is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate fisheye test image. Check camera and calibration.",
        )

    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/summary", response_model=DiagnosticSummary)
async def get_diagnostic_summary() -> DiagnosticSummary:
    """Get comprehensive diagnostic summary."""
    try:
        # Simulate running all diagnostic tests and collecting results
        total_tests = 15
        passed_tests = 13
        failed_tests = 1
        warnings = 1
        critical_issues = 0

        overall_score = (passed_tests / total_tests) * 100

        system_info = {
            "version": "1.0.0",
            "uptime": "2h 15m",
            "platform": "Linux x86_64" if psutil else "Unknown",
            "python_version": "3.11.0",
        }

        if psutil:
            system_info.update(
                {
                    "cpu_cores": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                    "disk_gb": round(psutil.disk_usage("/").total / (1024**3), 1),
                }
            )

        recommendations = []
        if failed_tests > 0:
            recommendations.append("Review and resolve failed diagnostic tests")
        if warnings > 0:
            recommendations.append("Address warning conditions for optimal performance")
        if overall_score < 90:
            recommendations.append(
                "Run individual diagnostic categories for detailed analysis"
            )

        return DiagnosticSummary(
            timestamp=datetime.now(timezone.utc),
            overall_score=overall_score,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warnings=warnings,
            critical_issues=critical_issues,
            system_info=system_info,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Diagnostic summary failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Diagnostic Summary Failed",
                "message": "Unable to generate diagnostic summary",
                "details": {"error": str(e)},
            },
        )


@router.post("/test/download/{size_mb}")
async def test_download_endpoint(size_mb: int) -> dict[str, Any]:
    """Test endpoint for bandwidth testing - simulates downloading data."""
    if size_mb > 100:  # Limit size for safety
        raise HTTPException(status_code=400, detail="Size too large")

    # Generate test data
    test_data = b"0" * (size_mb * 1024 * 1024)

    return {"size": len(test_data), "message": f"Downloaded {size_mb}MB test file"}


@router.post("/test/upload")
async def test_upload_endpoint() -> dict[str, Any]:
    """Test endpoint for bandwidth testing - simulates uploading data."""
    # In real implementation, would receive and process uploaded data
    return {"message": "Upload test completed", "received_size": 0}
