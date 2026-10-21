"""Health check endpoints for system monitoring and status reporting.

Provides comprehensive health information including:
- System health and status (FR-API-001)
- Version and capability information (FR-API-002)
- Performance metrics and statistics (FR-API-004)
- Graceful shutdown endpoint (FR-API-003)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Optional, Protocol, cast

try:
    _psutil: PsutilModule | None = cast("PsutilModule", import_module("psutil"))
except ImportError:
    _psutil = None

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from backend.api.dependencies import ApplicationState, get_app_state
from backend.api.models.responses import (
    CapabilityInfo,
    ComponentHealth,
    HealthResponse,
    HealthStatus,
    ShutdownResponse,
    SystemMetrics,
    VersionResponse,
)
from backend.api.shutdown import get_shutdown_progress, shutdown_coordinator
from backend.api.websocket.rate_limit import rate_limiter
from backend.system.health_monitor import health_monitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health & Status"])

# Shutdown state tracking
_shutdown_scheduled = False
_shutdown_time: datetime | None = None


class VirtualMemoryStats(Protocol):
    """Resource fields used from psutil's virtual-memory result."""

    percent: float


class DiskUsageStats(Protocol):
    """Resource fields used from psutil's disk-usage result."""

    used: int
    total: int


class NetworkIOStats(Protocol):
    """Network counters consumed by the health response."""

    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int


class PsutilModule(Protocol):
    """Small typed interface for the optional psutil dependency."""

    def cpu_percent(self, interval: float | None = None) -> float: ...

    def virtual_memory(self) -> VirtualMemoryStats: ...

    def disk_usage(self, path: str) -> DiskUsageStats: ...

    def net_io_counters(self) -> NetworkIOStats: ...


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    """Read a numeric health metric, using zero for absent or malformed values."""
    metric = metrics.get(name)
    if metric is None:
        return 0.0
    value = getattr(metric, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def get_system_metrics() -> SystemMetrics | None:
    """Get current system performance metrics from health monitor."""
    try:
        # Get system health from health monitor
        system_health = health_monitor.get_system_health()
        system_component = system_health.components.get("system")

        if not system_component:
            # Fallback to basic psutil if health monitor doesn't have system data
            if _psutil is None:
                return None

            cpu_percent = _psutil.cpu_percent(interval=0.1)
            memory = _psutil.virtual_memory()
            disk = _psutil.disk_usage("/")
            net_io = _psutil.net_io_counters()

            network_io = {
                "bytes_sent": float(net_io.bytes_sent),
                "bytes_recv": float(net_io.bytes_recv),
                "packets_sent": float(net_io.packets_sent),
                "packets_recv": float(net_io.packets_recv),
            }

            api_metrics = health_monitor.get_api_metrics()
            websocket_connections = health_monitor.get_websocket_connection_count()

            api_requests_per_second = _metric_value(
                api_metrics, "api_requests_per_second"
            )
            average_response_time = _metric_value(api_metrics, "api_avg_response_time")

            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                api_requests_per_second=api_requests_per_second,
                websocket_connections=websocket_connections,
                average_response_time=average_response_time,
            )

        # Extract metrics from health monitor
        metrics = system_component.metrics

        # Get basic system metrics
        cpu_usage = _metric_value(metrics, "cpu_usage")
        memory_usage = _metric_value(metrics, "memory_usage")
        disk_usage = _metric_value(metrics, "disk_usage")

        # Network metrics
        network_io = {
            "bytes_sent": _metric_value(metrics, "network_bytes_sent"),
            "bytes_recv": _metric_value(metrics, "network_bytes_recv"),
            "packets_sent": 0.0,  # Not tracked in our health monitor yet
            "packets_recv": 0.0,  # Not tracked in our health monitor yet
        }

        # Get API and WebSocket metrics from health monitor
        api_metrics = health_monitor.get_api_metrics()
        websocket_connections = health_monitor.get_websocket_connection_count()

        api_requests_per_second = _metric_value(api_metrics, "api_requests_per_second")
        average_response_time = _metric_value(api_metrics, "api_avg_response_time")

        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            api_requests_per_second=api_requests_per_second,
            websocket_connections=websocket_connections,
            average_response_time=average_response_time,
        )
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return None


@router.get("", response_model=HealthResponse)
async def health_check(
    include_details: bool = Query(
        False, description="Include detailed component health"
    ),
    include_metrics: bool = Query(False, description="Include performance metrics"),
    app_state: ApplicationState = Depends(get_app_state),
) -> HealthResponse:
    """System health check endpoint (FR-API-001) - No authentication required."""
    try:
        uptime = 0.0
        if app_state.startup_time:
            uptime = time.time() - app_state.startup_time

        # Determine overall status
        overall_status = (
            HealthStatus.HEALTHY if app_state.is_healthy else HealthStatus.UNHEALTHY
        )

        # Base response
        components: dict[str, ComponentHealth] = {}
        metrics = None

        # Include component details if requested
        if include_details:
            # Get component health from health monitor
            system_health = health_monitor.get_system_health()

            for component_name, component_health in system_health.components.items():
                # Convert health monitor status to API HealthStatus enum
                if component_health.status == "healthy":
                    status = HealthStatus.HEALTHY
                elif component_health.status == "degraded":
                    status = HealthStatus.DEGRADED
                elif component_health.status == "unhealthy":
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.UNHEALTHY

                # Create message from warnings or default
                message = (
                    "; ".join(component_health.warnings)
                    if component_health.warnings
                    else "Operating normally"
                )

                components[component_name] = ComponentHealth(
                    name=component_name,
                    status=status,
                    message=message,
                    last_check=datetime.fromtimestamp(
                        component_health.last_check, timezone.utc
                    ),
                    uptime=component_health.uptime,
                    errors=list(component_health.warnings),
                )

            # Add fallback components if not monitored by health monitor
            if "core" not in components:
                components["core"] = ComponentHealth(
                    name="core",
                    status=(
                        HealthStatus.HEALTHY
                        if app_state.core_module
                        else HealthStatus.UNHEALTHY
                    ),
                    message=(
                        "Operating normally"
                        if app_state.core_module
                        else "Core module unavailable"
                    ),
                    last_check=datetime.now(timezone.utc),
                    uptime=None,
                )

            if "config" not in components:
                components["config"] = ComponentHealth(
                    name="config",
                    status=(
                        HealthStatus.HEALTHY
                        if app_state.config_module
                        else HealthStatus.UNHEALTHY
                    ),
                    message=(
                        "Operating normally"
                        if app_state.config_module
                        else "Config module unavailable"
                    ),
                    last_check=datetime.now(timezone.utc),
                    uptime=None,
                )

            if "websocket" not in components:
                components["websocket"] = ComponentHealth(
                    name="websocket",
                    status=(
                        HealthStatus.HEALTHY
                        if app_state.websocket_manager
                        else HealthStatus.UNHEALTHY
                    ),
                    message=(
                        "Operating normally"
                        if app_state.websocket_manager
                        else "WebSocket manager unavailable"
                    ),
                    last_check=datetime.now(timezone.utc),
                    uptime=None,
                )

        # Include metrics if requested
        if include_metrics:
            metrics = get_system_metrics()

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            uptime=uptime,
            version="1.0.0",
            components=components,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Health Check Failed",
                "message": "Unable to determine system health",
                "code": "SYS_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/version", response_model=VersionResponse)
async def version_info() -> VersionResponse:
    """System version and capability information (FR-API-002) - No authentication required."""
    try:
        capabilities = CapabilityInfo(
            vision_processing=True,
            projector_support=True,
            calibration_modes=["table", "camera", "projector"],
            game_types=["practice", "8ball", "9ball", "straight"],
            export_formats=["json", "csv", "zip"],
            max_concurrent_sessions=10,
        )

        return VersionResponse(
            version="1.0.0",
            build_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            git_commit=os.environ.get("GIT_COMMIT", "dev"),
            capabilities=capabilities,
            api_version="v1",
            supported_clients=["web-ui-1.0", "mobile-app-1.0"],
        )

    except Exception as e:
        logger.error(f"Version info failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Version Info Failed",
                "message": "Unable to retrieve version information",
                "code": "SYS_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/metrics", response_model=SystemMetrics)
async def performance_metrics(
    time_range: str = Query(
        "5m", regex=r"^(5m|15m|1h|6h|24h)$", description="Time range for metrics"
    )
) -> SystemMetrics:
    """Performance metrics and statistics (FR-API-004) - No authentication required."""
    try:
        metrics = get_system_metrics()
        if not metrics:
            # Return safe defaults if psutil not available
            metrics = SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                api_requests_per_second=0.0,
                websocket_connections=0,
                average_response_time=0.0,
            )

        logger.info(f"Performance metrics requested for time range: {time_range}")
        return metrics

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Metrics Collection Failed",
                "message": "Unable to collect performance metrics",
                "code": "SYS_001",
                "details": {"error": str(e)},
            },
        )


async def perform_graceful_shutdown(
    delay: int = 0, force: bool = False, save_state: bool = True
) -> None:
    """Perform graceful shutdown after delay."""
    global _shutdown_scheduled, _shutdown_time

    if delay > 0:
        logger.info(f"Shutdown scheduled in {delay} seconds")
        await asyncio.sleep(delay)

    logger.info("Initiating graceful shutdown...")

    try:
        # Use the shutdown coordinator for comprehensive shutdown
        success = await shutdown_coordinator.initiate_shutdown(
            force=force, save_state=save_state
        )

        if success:
            logger.info("Graceful shutdown completed successfully")
        else:
            logger.error("Graceful shutdown completed with errors")

        # After successful shutdown, we can exit the application
        if success:
            # Give a brief moment for logging to flush
            await asyncio.sleep(0.1)

            # Exit the application
            import sys

            sys.exit(0)

    except Exception as e:
        logger.error(f"Shutdown process failed: {e}")
        # Even if shutdown fails, we should still exit
        import sys

        sys.exit(1)
    finally:
        _shutdown_scheduled = False
        _shutdown_time = None


@router.post("/shutdown", response_model=ShutdownResponse)
async def graceful_shutdown(
    background_tasks: BackgroundTasks,
    delay: int = Query(0, ge=0, le=300, description="Delay before shutdown in seconds"),
    force: bool = Query(
        False, description="Force shutdown without waiting for operations"
    ),
    save_state: bool = Query(True, description="Save current state before shutdown"),
) -> ShutdownResponse:
    """Schedule graceful shutdown through the directly accessible local API."""
    global _shutdown_scheduled, _shutdown_time

    try:
        if _shutdown_scheduled:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Shutdown Already Scheduled",
                    "message": "A shutdown is already in progress",
                    "code": "SYS_002",
                    "details": {
                        "scheduled_time": (
                            _shutdown_time.isoformat() if _shutdown_time else None
                        )
                    },
                },
            )

        # Schedule shutdown
        _shutdown_scheduled = True
        _shutdown_time = datetime.now(timezone.utc) + timedelta(seconds=delay)

        # Add shutdown task to background
        background_tasks.add_task(perform_graceful_shutdown, delay, force, save_state)

        # Get current active operations count from shutdown coordinator
        shutdown_status = get_shutdown_progress()
        active_operations = shutdown_status.get("active_operations", 0)

        logger.warning(
            f"Graceful shutdown requested with {delay}s delay, force={force}, save_state={save_state}"
        )

        return ShutdownResponse(
            acknowledged=True,
            scheduled_at=_shutdown_time,
            estimated_delay=delay,
            active_operations=active_operations,
            will_save_state=save_state,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Shutdown request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Shutdown Request Failed",
                "message": "Unable to schedule shutdown",
                "code": "SYS_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/shutdown/status")
async def shutdown_status() -> dict[str, Any]:
    """Get current shutdown status and progress - No authentication required."""
    try:
        return get_shutdown_progress()
    except Exception as e:
        logger.error(f"Failed to get shutdown status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Shutdown Status Failed",
                "message": "Unable to retrieve shutdown status",
                "code": "SYS_001",
                "details": {"error": str(e)},
            },
        )


@router.get("/ready")
async def readiness_check(
    app_state: ApplicationState = Depends(get_app_state),
) -> dict[str, Any]:
    """Kubernetes-style readiness check - No authentication required."""
    rate_limit_ready = await rate_limiter.check_readiness()
    components = {
        "core": "healthy" if app_state.core_module else "unavailable",
        "config": "healthy" if app_state.config_module else "unavailable",
        "websocket": "healthy" if app_state.websocket_manager else "unavailable",
        "rate_limit_store": "healthy" if rate_limit_ready else "unavailable",
    }

    all_healthy = all(status == "healthy" for status in components.values())

    if not all_healthy:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Service is not ready to accept requests",
                "components": components,
            },
        )

    return {
        "status": "ready",
        "components": components,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/perception")
async def perception_runtime_status(
    app_state: ApplicationState = Depends(get_app_state),
) -> dict[str, Any]:
    """Report provider mode and distinguish launch from first localization."""
    runtime = app_state.perception_runtime
    if runtime is None:
        provider = app_state.perception_provider or "unavailable"
        return {
            "provider": provider,
            "mode": "live" if provider == "python" else "unavailable",
            "process_started": None,
            "process_running": None,
            "process_status": "not_applicable",
            "first_observation_received": None,
            "source_health": (
                "python_provider" if provider == "python" else "unavailable"
            ),
            "last_observation_age_seconds": None,
            "source_silence_timeout_seconds": None,
            "live_camera": provider == "python",
            "preview": "python_provider" if provider == "python" else "unavailable",
            "predictions": "python_core" if provider == "python" else "unavailable",
        }

    status = runtime.status
    return {
        "provider": "rust",
        "mode": status.mode,
        "process_started": status.process_started,
        "process_running": status.process_running,
        "process_return_code": status.process_return_code,
        "process_status": status.process_status,
        "first_observation_received": status.first_observation_received,
        "source_health": status.source_health,
        "last_observation_age_seconds": status.last_observation_age_seconds,
        "source_silence_timeout_seconds": status.source_silence_timeout_seconds,
        "live_camera": status.live_camera,
        "preview": status.preview,
        "predictions": status.predictions,
        "ingress_accepted_snapshots": status.ingress_accepted_snapshots,
        "ingress_queue_replacements": status.ingress_queue_replacements,
        "state_snapshots_published": status.state_snapshots_published,
    }


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes-style liveness check - No authentication required."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Service is alive",
    }
