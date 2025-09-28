"""Performance monitoring and metrics collection system."""

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    timestamp: float = field(default_factory=time.time)

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: list[float] = field(default_factory=list)

    # Memory metrics
    memory_total: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    memory_available: int = 0

    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_percent: float = 0.0
    disk_free: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0

    # Process metrics
    process_count: int = 0
    thread_count: int = 0

    # GPU metrics (if available)
    gpu_percent: float = 0.0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0


@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics."""

    timestamp: float = field(default_factory=time.time)

    # Vision processing
    vision_fps: float = 0.0
    vision_latency: float = 0.0
    vision_accuracy: dict[str, float] = field(default_factory=dict)
    frames_processed: int = 0
    frames_dropped: int = 0

    # Core processing
    core_update_time: float = 0.0
    physics_calculation_time: float = 0.0
    analysis_time: float = 0.0
    prediction_accuracy: float = 0.0

    # API metrics
    api_requests_total: int = 0
    api_requests_per_second: float = 0.0
    api_response_time: float = 0.0
    api_error_rate: float = 0.0
    active_connections: int = 0

    # Projector metrics
    projector_latency: float = 0.0
    projector_accuracy: float = 0.0


@dataclass
class Alert:
    """System alert definition."""

    id: str
    type: str  # "warning", "error", "critical"
    message: str
    source: str
    threshold: float
    current_value: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class MetricsCollector:
    """Collects system and application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.system_modules: dict[str, Any] = {}
        self.is_running = False
        logger.info("Metrics Collector initialized")

    def register_module(self, name: str, module: Any) -> None:
        """Register a module for metrics collection.

        Args:
            name: Module name
            module: Module instance
        """
        self.system_modules[name] = module
        logger.debug(f"Module {name} registered for metrics collection")

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide performance metrics.

        Returns:
            System metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Load average (Unix-like systems only)
            load_average = []
            if hasattr(psutil, "getloadavg"):
                with contextlib.suppress(AttributeError):
                    load_average = list(psutil.getloadavg())

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage("/")

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process_count = len(psutil.pids())
            thread_count = sum(
                p.num_threads()
                for p in psutil.process_iter(["num_threads"])
                if p.info["num_threads"]
            )

            # GPU metrics (basic, would need specific GPU libraries for detailed info)
            gpu_percent = 0.0
            gpu_memory_used = 0
            gpu_memory_total = 0

            return SystemMetrics(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_average,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_percent=disk.percent,
                disk_free=disk.free,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                process_count=process_count,
                thread_count=thread_count,
                gpu_percent=gpu_percent,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()

    async def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics.

        Returns:
            Application metrics
        """
        try:
            metrics = ApplicationMetrics()

            # Vision module metrics
            if "vision" in self.system_modules:
                vision_module = self.system_modules["vision"]
                if hasattr(vision_module, "get_statistics"):
                    stats = vision_module.get_statistics()
                    metrics.vision_fps = stats.get("avg_fps", 0.0)
                    metrics.vision_latency = stats.get("avg_processing_time_ms", 0.0)
                    metrics.vision_accuracy = stats.get("detection_accuracy", {})
                    metrics.frames_processed = stats.get("frames_processed", 0)
                    metrics.frames_dropped = stats.get("frames_dropped", 0)

            # Core module metrics
            if "core" in self.system_modules:
                core_module = self.system_modules["core"]
                if hasattr(core_module, "get_performance_metrics"):
                    perf_metrics = core_module.get_performance_metrics()
                    metrics.core_update_time = perf_metrics.avg_update_time
                    metrics.physics_calculation_time = perf_metrics.avg_physics_time
                    metrics.analysis_time = perf_metrics.avg_analysis_time

            # API metrics would be collected from the API module
            # This would require API-specific metrics tracking

            # Projector metrics
            if "projector" in self.system_modules:
                self.system_modules["projector"]
                # Add projector-specific metrics collection here

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return ApplicationMetrics()


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_collector = MetricsCollector()
        self.is_running = False
        self.metrics_history: list[dict[str, Any]] = []
        self.max_history_size = 1000

        logger.info("Performance Monitor initialized")

    async def start(self) -> None:
        """Start performance monitoring."""
        self.is_running = True
        logger.info("Performance monitoring started")

    async def stop(self) -> None:
        """Stop performance monitoring."""
        self.is_running = False
        logger.info("Performance monitoring stopped")

    def register_module(self, name: str, module: Any) -> None:
        """Register a module for monitoring.

        Args:
            name: Module name
            module: Module instance
        """
        self.metrics_collector.register_module(name, module)

    async def collect_metrics(self) -> dict[str, Any]:
        """Collect comprehensive system metrics.

        Returns:
            Combined system and application metrics
        """
        try:
            system_metrics = await self.metrics_collector.collect_system_metrics()
            app_metrics = await self.metrics_collector.collect_application_metrics()

            combined_metrics = {
                "system": system_metrics,
                "application": app_metrics,
                "timestamp": time.time(),
            }

            # Store in history
            self._add_to_history(combined_metrics)

            return combined_metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}

    async def get_current_metrics(self) -> dict[str, Any]:
        """Get latest metrics.

        Returns:
            Latest collected metrics
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}

    async def get_metrics_history(
        self, count: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Get historical metrics.

        Args:
            count: Number of historical entries to return

        Returns:
            List of historical metrics
        """
        if count is None:
            return self.metrics_history.copy()
        return self.metrics_history[-count:]

    def _add_to_history(self, metrics: dict[str, Any]) -> None:
        """Add metrics to history with size limit."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)


class AlertManager:
    """System alerting and notification management."""

    def __init__(self):
        """Initialize alert manager."""
        self.alerts: list[Alert] = []
        self.thresholds: dict[str, dict[str, float]] = {}
        self.is_configured = False

        logger.info("Alert Manager initialized")

    async def configure(self, config: dict[str, Any]) -> None:
        """Configure alert thresholds.

        Args:
            config: Alert configuration
        """
        self.thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0},
            "vision_fps": {"warning": 15.0, "critical": 5.0},  # Below these values
            "api_error_rate": {"warning": 0.05, "critical": 0.10},  # Above these values
        }

        # Override with provided config
        if "max_cpu_percent" in config:
            self.thresholds["cpu_percent"]["critical"] = config["max_cpu_percent"]
        if "max_memory_mb" in config:
            # Convert to percentage (would need total memory info)
            pass

        self.is_configured = True
        logger.info("Alert thresholds configured")

    async def check_thresholds(self, metrics: dict[str, Any]) -> list[Alert]:
        """Check metrics against thresholds and generate alerts.

        Args:
            metrics: Current system metrics

        Returns:
            List of new alerts
        """
        if not self.is_configured:
            return []

        new_alerts = []

        try:
            system_metrics = metrics.get("system")
            app_metrics = metrics.get("application")

            if system_metrics:
                # Check system thresholds
                new_alerts.extend(self._check_system_thresholds(system_metrics))

            if app_metrics:
                # Check application thresholds
                new_alerts.extend(self._check_application_thresholds(app_metrics))

            # Add new alerts to list
            for alert in new_alerts:
                self.alerts.append(alert)

            # Clean up old alerts (older than 1 hour)
            current_time = time.time()
            self.alerts = [
                alert for alert in self.alerts if current_time - alert.timestamp < 3600
            ]

            return new_alerts

        except Exception as e:
            logger.error(f"Failed to check thresholds: {e}")
            return []

    async def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts.

        Returns:
            List of active alerts
        """
        return [alert for alert in self.alerts if not alert.acknowledged]

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge

        Returns:
            True if alert was acknowledged
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True

        return False

    def _check_system_thresholds(self, system_metrics) -> list[Alert]:
        """Check system metrics against thresholds."""
        alerts = []

        # CPU usage
        if hasattr(system_metrics, "cpu_percent"):
            cpu_percent = system_metrics.cpu_percent
            if cpu_percent >= self.thresholds["cpu_percent"]["critical"]:
                alerts.append(
                    Alert(
                        id=f"cpu_critical_{int(time.time())}",
                        type="critical",
                        message=f"Critical CPU usage: {cpu_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["cpu_percent"]["critical"],
                        current_value=cpu_percent,
                    )
                )
            elif cpu_percent >= self.thresholds["cpu_percent"]["warning"]:
                alerts.append(
                    Alert(
                        id=f"cpu_warning_{int(time.time())}",
                        type="warning",
                        message=f"High CPU usage: {cpu_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["cpu_percent"]["warning"],
                        current_value=cpu_percent,
                    )
                )

        # Memory usage
        if hasattr(system_metrics, "memory_percent"):
            memory_percent = system_metrics.memory_percent
            if memory_percent >= self.thresholds["memory_percent"]["critical"]:
                alerts.append(
                    Alert(
                        id=f"memory_critical_{int(time.time())}",
                        type="critical",
                        message=f"Critical memory usage: {memory_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["memory_percent"]["critical"],
                        current_value=memory_percent,
                    )
                )
            elif memory_percent >= self.thresholds["memory_percent"]["warning"]:
                alerts.append(
                    Alert(
                        id=f"memory_warning_{int(time.time())}",
                        type="warning",
                        message=f"High memory usage: {memory_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["memory_percent"]["warning"],
                        current_value=memory_percent,
                    )
                )

        # Disk usage
        if hasattr(system_metrics, "disk_percent"):
            disk_percent = system_metrics.disk_percent
            if disk_percent >= self.thresholds["disk_percent"]["critical"]:
                alerts.append(
                    Alert(
                        id=f"disk_critical_{int(time.time())}",
                        type="critical",
                        message=f"Critical disk usage: {disk_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["disk_percent"]["critical"],
                        current_value=disk_percent,
                    )
                )
            elif disk_percent >= self.thresholds["disk_percent"]["warning"]:
                alerts.append(
                    Alert(
                        id=f"disk_warning_{int(time.time())}",
                        type="warning",
                        message=f"High disk usage: {disk_percent:.1f}%",
                        source="system",
                        threshold=self.thresholds["disk_percent"]["warning"],
                        current_value=disk_percent,
                    )
                )

        return alerts

    def _check_application_thresholds(self, app_metrics) -> list[Alert]:
        """Check application metrics against thresholds."""
        alerts = []

        # Vision FPS
        if hasattr(app_metrics, "vision_fps"):
            fps = app_metrics.vision_fps
            if fps <= self.thresholds["vision_fps"]["critical"]:
                alerts.append(
                    Alert(
                        id=f"vision_fps_critical_{int(time.time())}",
                        type="critical",
                        message=f"Critical vision FPS: {fps:.1f}",
                        source="vision",
                        threshold=self.thresholds["vision_fps"]["critical"],
                        current_value=fps,
                    )
                )
            elif fps <= self.thresholds["vision_fps"]["warning"]:
                alerts.append(
                    Alert(
                        id=f"vision_fps_warning_{int(time.time())}",
                        type="warning",
                        message=f"Low vision FPS: {fps:.1f}",
                        source="vision",
                        threshold=self.thresholds["vision_fps"]["warning"],
                        current_value=fps,
                    )
                )

        return alerts


class PerformanceDashboard:
    """Real-time performance dashboard interface."""

    def __init__(
        self, performance_monitor: PerformanceMonitor, alert_manager: AlertManager
    ):
        """Initialize performance dashboard.

        Args:
            performance_monitor: Performance monitoring instance
            alert_manager: Alert management instance
        """
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager

        logger.info("Performance Dashboard initialized")

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive dashboard data.

        Returns:
            Dashboard data including metrics, alerts, and summaries
        """
        try:
            # Get current metrics
            current_metrics = await self.performance_monitor.get_current_metrics()

            # Get recent metrics history
            recent_history = await self.performance_monitor.get_metrics_history(
                60
            )  # Last 60 data points

            # Get active alerts
            active_alerts = await self.alert_manager.get_active_alerts()

            # Calculate summary statistics
            summary = self._calculate_summary_stats(recent_history)

            return {
                "current_metrics": current_metrics,
                "recent_history": recent_history,
                "active_alerts": [
                    {
                        "id": alert.id,
                        "type": alert.type,
                        "message": alert.message,
                        "source": alert.source,
                        "timestamp": alert.timestamp,
                    }
                    for alert in active_alerts
                ],
                "summary": summary,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}

    def _calculate_summary_stats(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate summary statistics from metrics history.

        Args:
            history: List of historical metrics

        Returns:
            Summary statistics
        """
        if not history:
            return {}

        try:
            # Extract key metrics
            cpu_values = []
            memory_values = []
            fps_values = []

            for entry in history:
                system = entry.get("system")
                app = entry.get("application")

                if system and hasattr(system, "cpu_percent"):
                    cpu_values.append(system.cpu_percent)
                if system and hasattr(system, "memory_percent"):
                    memory_values.append(system.memory_percent)
                if app and hasattr(app, "vision_fps"):
                    fps_values.append(app.vision_fps)

            # Calculate statistics
            summary = {}

            if cpu_values:
                summary["cpu"] = {
                    "avg": sum(cpu_values) / len(cpu_values),
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                }

            if memory_values:
                summary["memory"] = {
                    "avg": sum(memory_values) / len(memory_values),
                    "min": min(memory_values),
                    "max": max(memory_values),
                }

            if fps_values:
                summary["vision_fps"] = {
                    "avg": sum(fps_values) / len(fps_values),
                    "min": min(fps_values),
                    "max": max(fps_values),
                }

            return summary

        except Exception as e:
            logger.error(f"Failed to calculate summary stats: {e}")
            return {}
