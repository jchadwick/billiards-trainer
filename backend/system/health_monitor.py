"""Real-time health monitoring system for all application components.

Provides comprehensive health monitoring for:
- Core module performance and status
- Vision module camera and processing health
- Configuration system health
- API server performance metrics
- WebSocket connection health
- System resource utilization
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Individual health metric with metadata."""

    name: str
    value: Any
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    healthy: bool = True
    message: str = "Operating normally"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Health status for a system component."""

    component: str
    status: str  # "healthy", "degraded", "unhealthy", "unavailable"
    metrics: dict[str, HealthMetric] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    error_count: int = 0
    warnings: list[str] = field(default_factory=list)
    uptime: float = 0.0
    version: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health aggregation."""

    overall_status: str  # "healthy", "degraded", "unhealthy"
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    system_metrics: dict[str, HealthMetric] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    active_alerts: list[str] = field(default_factory=list)


class HealthMonitor:
    """Real-time health monitoring for all system components."""

    def __init__(self):
        """Initialize health monitoring system."""
        self._lock = Lock()
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._check_interval = 5.0  # seconds

        # Component references
        self._core_module = None
        self._vision_module = None
        self._config_module = None
        self._websocket_manager = None
        self._startup_time = time.time()

        # Health data
        self._system_health = SystemHealth(overall_status="healthy")
        self._component_health: dict[str, ComponentHealth] = {}
        self._health_history: list[SystemHealth] = []
        self._max_history = 100

        # Monitoring callbacks
        self._health_callbacks: list[Callable[[SystemHealth], None]] = []
        self._alert_callbacks: list[Callable[[str, str, dict[str, Any]], None]] = []

        # Performance tracking
        self._api_request_count = 0
        self._api_request_times: list[float] = []
        self._api_error_count = 0
        self._max_request_times = 1000

        logger.info("Health monitor initialized")

    def register_components(
        self,
        core_module=None,
        vision_module=None,
        config_module=None,
        websocket_manager=None,
    ):
        """Register system components for monitoring.

        Args:
            core_module: CoreModule instance
            vision_module: VisionModule instance
            config_module: ConfigurationModule instance
            websocket_manager: WebSocketManager instance
        """
        with self._lock:
            if core_module:
                self._core_module = core_module
                logger.info("Core module registered for health monitoring")

            if vision_module:
                self._vision_module = vision_module
                logger.info("Vision module registered for health monitoring")

            if config_module:
                self._config_module = config_module
                logger.info("Configuration module registered for health monitoring")

            if websocket_manager:
                self._websocket_manager = websocket_manager
                logger.info("WebSocket manager registered for health monitoring")

    async def start_monitoring(self, check_interval: float = 5.0):
        """Start continuous health monitoring.

        Args:
            check_interval: Time between health checks in seconds
        """
        if self._is_running:
            logger.warning("Health monitoring is already running")
            return

        self._check_interval = check_interval
        self._is_running = True

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        logger.info(f"Health monitoring started with {check_interval}s interval")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._is_running:
            return

        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_running:
            try:
                start_time = time.time()

                # Perform health checks
                await self._check_all_components()

                # Update system health
                self._update_system_health()

                # Store in history
                self._add_to_history()

                # Notify callbacks
                await self._notify_health_callbacks()

                # Check for alerts
                await self._check_alerts()

                # Calculate sleep time
                check_time = time.time() - start_time
                sleep_time = max(0, self._check_interval - check_time)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    async def _check_all_components(self):
        """Check health of all registered components."""
        # Check core module
        if self._core_module:
            await self._check_core_module()

        # Check vision module
        if self._vision_module:
            await self._check_vision_module()

        # Check configuration module
        if self._config_module:
            await self._check_config_module()

        # Check WebSocket manager
        if self._websocket_manager:
            await self._check_websocket_manager()

        # Check system resources
        await self._check_system_resources()

    async def _check_core_module(self):
        """Check core module health."""
        try:
            component = "core"
            health = ComponentHealth(
                component=component,
                status="healthy",
                uptime=time.time() - self._startup_time,
            )

            # Get core module metrics
            if hasattr(self._core_module, "get_performance_metrics"):
                try:
                    metrics = self._core_module.get_performance_metrics()

                    # Process performance metrics
                    health.metrics["total_updates"] = HealthMetric(
                        name="total_updates",
                        value=metrics.total_updates,
                        unit="count",
                        healthy=True,
                    )

                    health.metrics["avg_update_time"] = HealthMetric(
                        name="avg_update_time",
                        value=metrics.avg_update_time * 1000,
                        unit="ms",
                        healthy=metrics.avg_update_time < 0.1,  # Alert if >100ms
                    )

                    health.metrics["avg_physics_time"] = HealthMetric(
                        name="avg_physics_time",
                        value=metrics.avg_physics_time * 1000,
                        unit="ms",
                        healthy=metrics.avg_physics_time < 0.05,  # Alert if >50ms
                    )

                    health.metrics["avg_analysis_time"] = HealthMetric(
                        name="avg_analysis_time",
                        value=metrics.avg_analysis_time * 1000,
                        unit="ms",
                        healthy=metrics.avg_analysis_time < 0.1,  # Alert if >100ms
                    )

                    health.metrics["errors_count"] = HealthMetric(
                        name="errors_count",
                        value=metrics.errors_count,
                        unit="count",
                        healthy=metrics.errors_count == 0,
                    )

                    # Update component status based on metrics
                    if metrics.errors_count > 10:
                        health.status = "unhealthy"
                        health.warnings.append(
                            f"High error count: {metrics.errors_count}"
                        )
                    elif (
                        metrics.avg_update_time > 0.1
                        or metrics.avg_physics_time > 0.05
                        or metrics.avg_analysis_time > 0.1
                    ):
                        health.status = "degraded"
                        health.warnings.append("Performance degradation detected")

                except Exception as e:
                    logger.warning(f"Failed to get core module metrics: {e}")
                    health.status = "degraded"
                    health.warnings.append("Unable to retrieve performance metrics")

            # Check if core module has current state
            if hasattr(self._core_module, "get_current_state"):
                try:
                    current_state = self._core_module.get_current_state()
                    health.metrics["has_current_state"] = HealthMetric(
                        name="has_current_state",
                        value=current_state is not None,
                        healthy=True,  # Not having state isn't necessarily unhealthy
                    )
                except Exception as e:
                    logger.warning(f"Failed to check core module state: {e}")

            # Validate configuration if available
            if hasattr(self._core_module, "validate_state"):
                try:
                    validation_result = await self._core_module.validate_state()
                    health.metrics["state_valid"] = HealthMetric(
                        name="state_valid",
                        value=validation_result.get("valid", False),
                        healthy=validation_result.get("valid", False),
                        details=validation_result,
                    )

                    if not validation_result.get("valid", False):
                        health.status = "degraded"
                        issues = validation_result.get("issues", [])
                        health.warnings.extend(issues[:3])  # Limit warnings

                except Exception as e:
                    logger.warning(f"Failed to validate core module state: {e}")

            health.last_check = time.time()
            self._component_health[component] = health

        except Exception as e:
            logger.error(f"Error checking core module health: {e}")
            self._component_health["core"] = ComponentHealth(
                component="core",
                status="unhealthy",
                error_count=1,
                warnings=[f"Health check failed: {str(e)}"],
                last_check=time.time(),
            )

    async def _check_vision_module(self):
        """Check vision module health."""
        try:
            component = "vision"
            health = ComponentHealth(
                component=component,
                status="healthy",
                uptime=time.time() - self._startup_time,
            )

            # Get vision statistics
            if hasattr(self._vision_module, "get_statistics"):
                try:
                    stats = self._vision_module.get_statistics()

                    # Process vision metrics
                    health.metrics["frames_processed"] = HealthMetric(
                        name="frames_processed",
                        value=stats.get("frames_processed", 0),
                        unit="count",
                        healthy=True,
                    )

                    health.metrics["frames_dropped"] = HealthMetric(
                        name="frames_dropped",
                        value=stats.get("frames_dropped", 0),
                        unit="count",
                        healthy=stats.get("frames_dropped", 0) < 100,
                    )

                    health.metrics["avg_processing_time"] = HealthMetric(
                        name="avg_processing_time",
                        value=stats.get("avg_processing_time_ms", 0),
                        unit="ms",
                        healthy=stats.get("avg_processing_time_ms", 0) < 50,
                    )

                    health.metrics["avg_fps"] = HealthMetric(
                        name="avg_fps",
                        value=stats.get("avg_fps", 0),
                        unit="fps",
                        healthy=stats.get("avg_fps", 0) > 15,  # Alert if FPS too low
                    )

                    health.metrics["is_running"] = HealthMetric(
                        name="is_running",
                        value=stats.get("is_running", False),
                        healthy=True,  # Not running isn't necessarily unhealthy
                    )

                    health.metrics["camera_connected"] = HealthMetric(
                        name="camera_connected",
                        value=stats.get("camera_connected", False),
                        healthy=stats.get("camera_connected", False),
                    )

                    # Detection accuracy metrics
                    detection_accuracy = stats.get("detection_accuracy", {})
                    for detection_type, accuracy in detection_accuracy.items():
                        health.metrics[f"{detection_type}_accuracy"] = HealthMetric(
                            name=f"{detection_type}_accuracy",
                            value=accuracy,
                            unit="ratio",
                            healthy=accuracy > 0.7,  # Alert if accuracy below 70%
                        )

                    # Update component status based on metrics
                    if not stats.get("camera_connected", False):
                        health.status = "unhealthy"
                        health.warnings.append("Camera not connected")
                    elif (
                        stats.get("frames_dropped", 0) > 100
                        or stats.get("avg_fps", 0) < 15
                    ):
                        health.status = "degraded"
                        health.warnings.append("Performance issues detected")

                    # Check for errors
                    if stats.get("last_error"):
                        health.status = "degraded"
                        health.warnings.append(f"Recent error: {stats['last_error']}")

                except Exception as e:
                    logger.warning(f"Failed to get vision module statistics: {e}")
                    health.status = "degraded"
                    health.warnings.append("Unable to retrieve statistics")

            # Check camera health if available
            if hasattr(self._vision_module, "camera") and self._vision_module.camera:
                try:
                    camera = self._vision_module.camera
                    if hasattr(camera, "get_health"):
                        camera_health = camera.get_health()
                        health.metrics["camera_health"] = HealthMetric(
                            name="camera_health",
                            value=camera_health.status.value,
                            healthy=camera_health.status.value
                            in ["connected", "streaming"],
                            details=camera_health.__dict__,
                        )
                except Exception as e:
                    logger.warning(f"Failed to check camera health: {e}")

            health.last_check = time.time()
            self._component_health[component] = health

        except Exception as e:
            logger.error(f"Error checking vision module health: {e}")
            self._component_health["vision"] = ComponentHealth(
                component="vision",
                status="unhealthy",
                error_count=1,
                warnings=[f"Health check failed: {str(e)}"],
                last_check=time.time(),
            )

    async def _check_config_module(self):
        """Check configuration module health."""
        try:
            component = "config"
            health = ComponentHealth(
                component=component,
                status="healthy",
                uptime=time.time() - self._startup_time,
            )

            # Check basic configuration access
            try:
                # Test basic get operation
                self._config_module.get("app.name", "test")
                health.metrics["config_accessible"] = HealthMetric(
                    name="config_accessible", value=True, healthy=True
                )
            except Exception as e:
                health.metrics["config_accessible"] = HealthMetric(
                    name="config_accessible",
                    value=False,
                    healthy=False,
                    message=f"Config access failed: {e}",
                )
                health.status = "unhealthy"
                health.warnings.append("Configuration not accessible")

            # Check validation status
            if hasattr(self._config_module, "validate"):
                try:
                    is_valid, errors = self._config_module.validate()
                    health.metrics["config_valid"] = HealthMetric(
                        name="config_valid",
                        value=is_valid,
                        healthy=is_valid,
                        details={"errors": errors},
                    )

                    if not is_valid:
                        health.status = "degraded"
                        health.warnings.extend(errors[:3])  # Limit warnings

                except Exception as e:
                    logger.warning(f"Failed to validate configuration: {e}")

            # Check hot reload status if available
            if hasattr(self._config_module, "is_hot_reload_enabled"):
                try:
                    hot_reload_enabled = self._config_module.is_hot_reload_enabled()
                    health.metrics["hot_reload_enabled"] = HealthMetric(
                        name="hot_reload_enabled",
                        value=hot_reload_enabled,
                        healthy=True,  # Not enabled isn't necessarily unhealthy
                    )
                except Exception as e:
                    logger.warning(f"Failed to check hot reload status: {e}")

            # Check configuration file access
            if hasattr(self._config_module, "get_watched_files"):
                try:
                    watched_files = self._config_module.get_watched_files()
                    accessible_files = 0
                    for file_path in watched_files:
                        if file_path.exists():
                            accessible_files += 1

                    health.metrics["config_files_accessible"] = HealthMetric(
                        name="config_files_accessible",
                        value=f"{accessible_files}/{len(watched_files)}",
                        healthy=accessible_files == len(watched_files),
                        details={
                            "total": len(watched_files),
                            "accessible": accessible_files,
                        },
                    )

                    if accessible_files < len(watched_files):
                        health.status = "degraded"
                        health.warnings.append(
                            "Some configuration files not accessible"
                        )

                except Exception as e:
                    logger.warning(f"Failed to check configuration files: {e}")

            health.last_check = time.time()
            self._component_health[component] = health

        except Exception as e:
            logger.error(f"Error checking configuration module health: {e}")
            self._component_health["config"] = ComponentHealth(
                component="config",
                status="unhealthy",
                error_count=1,
                warnings=[f"Health check failed: {str(e)}"],
                last_check=time.time(),
            )

    async def _check_websocket_manager(self):
        """Check WebSocket manager health."""
        try:
            component = "websocket"
            health = ComponentHealth(
                component=component,
                status="healthy",
                uptime=time.time() - self._startup_time,
            )

            # Get session information
            if hasattr(self._websocket_manager, "get_all_sessions"):
                try:
                    session_info = await self._websocket_manager.get_all_sessions()

                    health.metrics["total_sessions"] = HealthMetric(
                        name="total_sessions",
                        value=session_info.get("total_sessions", 0),
                        unit="count",
                        healthy=True,
                    )

                    health.metrics["total_users"] = HealthMetric(
                        name="total_users",
                        value=session_info.get("total_users", 0),
                        unit="count",
                        healthy=True,
                    )

                    # Stream subscriber counts
                    stream_subscribers = session_info.get("stream_subscribers", {})
                    for stream_type, count in stream_subscribers.items():
                        health.metrics[f"{stream_type}_subscribers"] = HealthMetric(
                            name=f"{stream_type}_subscribers",
                            value=count,
                            unit="count",
                            healthy=True,
                        )

                    # Check for active sessions
                    sessions = session_info.get("sessions", [])
                    active_sessions = len(
                        [
                            s
                            for s in sessions
                            if s.get("connection_state") == "connected"
                        ]
                    )

                    health.metrics["active_sessions"] = HealthMetric(
                        name="active_sessions",
                        value=active_sessions,
                        unit="count",
                        healthy=True,
                        details={"total": len(sessions), "active": active_sessions},
                    )

                except Exception as e:
                    logger.warning(f"Failed to get WebSocket session info: {e}")
                    health.status = "degraded"
                    health.warnings.append("Unable to retrieve session information")

            # Check individual session health
            if hasattr(self._websocket_manager, "sessions"):
                try:
                    sessions = self._websocket_manager.sessions
                    error_sessions = 0

                    for _session_id, session in sessions.items():
                        if hasattr(session, "connection_state"):
                            if session.connection_state.value == "error":
                                error_sessions += 1

                    health.metrics["error_sessions"] = HealthMetric(
                        name="error_sessions",
                        value=error_sessions,
                        unit="count",
                        healthy=error_sessions == 0,
                    )

                    if error_sessions > 0:
                        health.status = "degraded"
                        health.warnings.append(
                            f"{error_sessions} sessions in error state"
                        )

                except Exception as e:
                    logger.warning(f"Failed to check session states: {e}")

            health.last_check = time.time()
            self._component_health[component] = health

        except Exception as e:
            logger.error(f"Error checking WebSocket manager health: {e}")
            self._component_health["websocket"] = ComponentHealth(
                component="websocket",
                status="unhealthy",
                error_count=1,
                warnings=[f"Health check failed: {str(e)}"],
                last_check=time.time(),
            )

    async def _check_system_resources(self):
        """Check system resource utilization."""
        try:
            if not psutil:
                return

            component = "system"
            health = ComponentHealth(
                component=component,
                status="healthy",
                uptime=time.time() - self._startup_time,
            )

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            health.metrics["cpu_usage"] = HealthMetric(
                name="cpu_usage", value=cpu_percent, unit="%", healthy=cpu_percent < 80
            )

            # Memory usage
            memory = psutil.virtual_memory()
            health.metrics["memory_usage"] = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                healthy=memory.percent < 85,
            )

            health.metrics["memory_available"] = HealthMetric(
                name="memory_available",
                value=memory.available / (1024**3),  # GB
                unit="GB",
                healthy=memory.available > 1024**3,  # Alert if <1GB available
            )

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            health.metrics["disk_usage"] = HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                healthy=disk_percent < 90,
            )

            # Network I/O
            net_io = psutil.net_io_counters()
            health.metrics["network_bytes_sent"] = HealthMetric(
                name="network_bytes_sent",
                value=float(net_io.bytes_sent),
                unit="bytes",
                healthy=True,
            )

            health.metrics["network_bytes_recv"] = HealthMetric(
                name="network_bytes_recv",
                value=float(net_io.bytes_recv),
                unit="bytes",
                healthy=True,
            )

            # Update status based on resource usage
            if cpu_percent > 90 or memory.percent > 95 or disk_percent > 95:
                health.status = "unhealthy"
                health.warnings.append("Critical resource usage")
            elif cpu_percent > 80 or memory.percent > 85 or disk_percent > 90:
                health.status = "degraded"
                health.warnings.append("High resource usage")

            health.last_check = time.time()
            self._component_health[component] = health

        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            self._component_health["system"] = ComponentHealth(
                component="system",
                status="degraded",
                error_count=1,
                warnings=[f"Resource check failed: {str(e)}"],
                last_check=time.time(),
            )

    def _update_system_health(self):
        """Update overall system health based on component health."""
        with self._lock:
            components = dict(self._component_health)

            # Determine overall status
            unhealthy_count = sum(
                1 for h in components.values() if h.status == "unhealthy"
            )
            degraded_count = sum(
                1 for h in components.values() if h.status == "degraded"
            )

            if unhealthy_count > 0:
                overall_status = "unhealthy"
            elif degraded_count > 0:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            # Collect active alerts
            active_alerts = []
            for comp_health in components.values():
                active_alerts.extend(comp_health.warnings)

            # Update system health
            self._system_health = SystemHealth(
                overall_status=overall_status,
                components=components,
                timestamp=time.time(),
                active_alerts=active_alerts,
            )

    def _add_to_history(self):
        """Add current health to history."""
        with self._lock:
            self._health_history.append(self._system_health)

            # Limit history size
            if len(self._health_history) > self._max_history:
                self._health_history.pop(0)

    async def _notify_health_callbacks(self):
        """Notify registered health callbacks."""
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._system_health)
                else:
                    callback(self._system_health)
            except Exception as e:
                logger.warning(f"Error in health callback: {e}")

    async def _check_alerts(self):
        """Check for and emit health alerts."""
        for component_name, health in self._component_health.items():
            if health.status == "unhealthy":
                for callback in self._alert_callbacks:
                    try:
                        await callback(
                            "error",
                            f"Component {component_name} is unhealthy",
                            {"component": component_name, "warnings": health.warnings},
                        )
                    except Exception as e:
                        logger.warning(f"Error in alert callback: {e}")

    def get_system_health(self) -> SystemHealth:
        """Get current system health snapshot."""
        with self._lock:
            return self._system_health

    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get health for specific component."""
        return self._component_health.get(component)

    def get_health_history(self, count: int = 10) -> list[SystemHealth]:
        """Get recent health history."""
        with self._lock:
            return self._health_history[-count:]

    def add_health_callback(self, callback: Callable[[SystemHealth], None]):
        """Add callback for health updates."""
        self._health_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[str, str, dict[str, Any]], None]):
        """Add callback for health alerts."""
        self._alert_callbacks.append(callback)

    def record_api_request(self, response_time: float, success: bool = True):
        """Record API request metrics."""
        with self._lock:
            self._api_request_count += 1
            self._api_request_times.append(response_time)

            if not success:
                self._api_error_count += 1

            # Limit stored request times
            if len(self._api_request_times) > self._max_request_times:
                self._api_request_times.pop(0)

    def get_api_metrics(self) -> dict[str, HealthMetric]:
        """Get API performance metrics."""
        with self._lock:
            if not self._api_request_times:
                return {}

            avg_response_time = sum(self._api_request_times) / len(
                self._api_request_times
            )
            error_rate = self._api_error_count / max(1, self._api_request_count)

            # Calculate requests per second (rough estimate)
            uptime = time.time() - self._startup_time
            requests_per_second = self._api_request_count / max(1, uptime)

            return {
                "api_requests_total": HealthMetric(
                    name="api_requests_total",
                    value=self._api_request_count,
                    unit="count",
                    healthy=True,
                ),
                "api_requests_per_second": HealthMetric(
                    name="api_requests_per_second",
                    value=requests_per_second,
                    unit="req/s",
                    healthy=True,
                ),
                "api_avg_response_time": HealthMetric(
                    name="api_avg_response_time",
                    value=avg_response_time * 1000,  # Convert to ms
                    unit="ms",
                    healthy=avg_response_time < 0.5,  # Alert if >500ms
                ),
                "api_error_rate": HealthMetric(
                    name="api_error_rate",
                    value=error_rate * 100,  # Convert to percentage
                    unit="%",
                    healthy=error_rate < 0.05,  # Alert if >5% error rate
                ),
            }

    def get_websocket_connection_count(self) -> int:
        """Get current WebSocket connection count."""
        if not self._websocket_manager:
            return 0

        try:
            if hasattr(self._websocket_manager, "sessions"):
                return len(self._websocket_manager.sessions)
        except Exception:
            pass

        return 0


# Global health monitor instance
health_monitor = HealthMonitor()
