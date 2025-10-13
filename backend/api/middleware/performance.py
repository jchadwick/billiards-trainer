"""Performance monitoring middleware for the billiards trainer API."""

import asyncio
import contextlib
import gc
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, median
from typing import Any, Callable, Optional

import psutil
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RequestMetrics:
    """Individual request performance metrics."""

    timestamp: datetime
    method: str
    path: str
    status_code: int
    duration_ms: float
    request_size: int
    response_size: int
    memory_usage_mb: float
    cpu_usage_percent: float
    error: Optional[str] = None


@dataclass
class EndpointStats:
    """Statistics for a specific endpoint."""

    path: str
    method: str
    total_requests: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0
    recent_durations: deque[float] = field(default_factory=deque)
    last_request: Optional[datetime] = None
    error_rate: float = 0


class PerformanceConfig(BaseModel):
    """Configuration for performance monitoring."""

    enable_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    enable_system_metrics: bool = Field(
        default=True, description="Enable system resource monitoring"
    )
    enable_endpoint_stats: bool = Field(
        default=True, description="Enable per-endpoint statistics"
    )
    enable_slow_query_detection: bool = Field(
        default=True, description="Enable slow request detection"
    )
    slow_request_threshold_ms: float = Field(
        default=1000.0, description="Threshold for slow requests in milliseconds"
    )
    metrics_retention_minutes: int = Field(
        default=60, description="How long to retain detailed metrics"
    )
    stats_window_size: int = Field(
        default=1000, description="Number of recent requests to keep for stats"
    )
    system_metrics_interval_seconds: int = Field(
        default=30, description="Interval for collecting system metrics"
    )
    memory_alert_threshold_mb: int = Field(
        default=512, description="Memory usage alert threshold in MB"
    )
    cpu_alert_threshold_percent: float = Field(
        default=80.0, description="CPU usage alert threshold percentage"
    )
    response_time_percentiles: list[int] = Field(
        default=[50, 90, 95, 99], description="Response time percentiles to track"
    )
    excluded_paths: list[str] = Field(
        default_factory=lambda: [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
        description="Paths to exclude from detailed monitoring",
    )
    endpoint_stats_recent_durations_maxlen: int = Field(
        default=100,
        description="Maximum number of recent durations to keep per endpoint",
    )
    cpu_percent_history_maxlen: int = Field(
        default=60, description="Maximum number of CPU measurements to keep in history"
    )
    memory_usage_history_maxlen: int = Field(
        default=60,
        description="Maximum number of memory measurements to keep in history",
    )


class SystemMetrics:
    """System resource metrics collector."""

    def __init__(self, cpu_history_maxlen: int = 60, memory_history_maxlen: int = 60):
        """Initialize system metrics collector."""
        self.process = psutil.Process()
        self.cpu_percent_history: deque[float] = deque(maxlen=cpu_history_maxlen)
        self.memory_usage_history: deque[float] = deque(maxlen=memory_history_maxlen)

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            system_cpu_percent = psutil.cpu_percent()

            # Memory metrics
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            system_memory = psutil.virtual_memory()

            # Disk metrics
            disk_usage = psutil.disk_usage("/")

            # Network metrics (basic)
            network_io = psutil.net_io_counters()

            # Update history
            self.cpu_percent_history.append(cpu_percent)
            self.memory_usage_history.append(memory_mb)

            return {
                "process": {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "threads": self.process.num_threads(),
                    "file_descriptors": (
                        self.process.num_fds()
                        if hasattr(self.process, "num_fds")
                        else 0
                    ),
                },
                "system": {
                    "cpu_percent": system_cpu_percent,
                    "memory_percent": system_memory.percent,
                    "memory_available_mb": system_memory.available / 1024 / 1024,
                    "disk_usage_percent": (disk_usage.used / disk_usage.total) * 100,
                    "disk_free_mb": disk_usage.free / 1024 / 1024,
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv,
                },
                "averages": {
                    "cpu_avg_1min": (
                        mean(self.cpu_percent_history)
                        if self.cpu_percent_history
                        else 0
                    ),
                    "memory_avg_1min": (
                        mean(self.memory_usage_history)
                        if self.memory_usage_history
                        else 0
                    ),
                },
            }
        except Exception as e:
            logging.getLogger(__name__).error(f"Error collecting system metrics: {e}")
            return {}


class PerformanceMonitor:
    """Main performance monitoring class."""

    def __init__(self, config: PerformanceConfig):
        """Initialize performance monitor."""
        self.config = config
        self.logger = logging.getLogger("api.performance")
        self.system_metrics = SystemMetrics(
            cpu_history_maxlen=config.cpu_percent_history_maxlen,
            memory_history_maxlen=config.memory_usage_history_maxlen,
        )

        # Request metrics storage
        self.request_metrics: deque[RequestMetrics] = deque(
            maxlen=config.stats_window_size
        )
        self.endpoint_stats: dict[str, EndpointStats] = defaultdict(
            lambda: EndpointStats(
                path="",
                method="",
                recent_durations=deque(
                    maxlen=config.endpoint_stats_recent_durations_maxlen
                ),
            )
        )

        # System monitoring
        self.system_metrics_cache: dict[str, Any] = {}
        self.last_system_metrics_update = time.time()

        # Performance alerts
        self.slow_requests_count = 0
        self.memory_alerts_count = 0
        self.cpu_alerts_count = 0

        # Start background tasks
        self._start_system_monitoring()

    def _start_system_monitoring(self) -> None:
        """Start background system monitoring."""
        if self.config.enable_system_metrics:

            def monitor_system():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._system_monitoring_loop())

            thread = threading.Thread(target=monitor_system, daemon=True)
            thread.start()

    async def _system_monitoring_loop(self) -> None:
        """Background loop for system monitoring."""
        while True:
            try:
                await asyncio.sleep(self.config.system_metrics_interval_seconds)
                metrics = self.system_metrics.get_current_metrics()
                self.system_metrics_cache = metrics
                self.last_system_metrics_update = time.time()

                # Check for alerts
                self._check_system_alerts(metrics)

            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")

    def _check_system_alerts(self, metrics: dict[str, Any]) -> None:
        """Check for system performance alerts."""
        if not metrics:
            return

        process_metrics = metrics.get("process", {})
        metrics.get("system", {})

        # Memory alert
        memory_mb = process_metrics.get("memory_mb", 0)
        if memory_mb > self.config.memory_alert_threshold_mb:
            self.memory_alerts_count += 1
            self.logger.warning(
                f"High memory usage detected: {memory_mb:.1f}MB "
                f"(threshold: {self.config.memory_alert_threshold_mb}MB)"
            )

        # CPU alert
        cpu_percent = process_metrics.get("cpu_percent", 0)
        if cpu_percent > self.config.cpu_alert_threshold_percent:
            self.cpu_alerts_count += 1
            self.logger.warning(
                f"High CPU usage detected: {cpu_percent:.1f}% "
                f"(threshold: {self.config.cpu_alert_threshold_percent}%)"
            )

    def should_monitor_path(self, path: str) -> bool:
        """Check if path should be monitored."""
        return path not in self.config.excluded_paths

    def record_request(
        self,
        request: Request,
        response: Response,
        duration_ms: float,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """Record request performance metrics."""
        if not self.config.enable_monitoring:
            return

        if not self.should_monitor_path(request.url.path):
            return

        # Get current system metrics for this request
        current_metrics = self.system_metrics_cache
        memory_mb = current_metrics.get("process", {}).get("memory_mb", 0)
        cpu_percent = current_metrics.get("process", {}).get("cpu_percent", 0)

        # Create request metrics
        metrics = RequestMetrics(
            timestamp=datetime.utcnow(),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            error=(
                None if response.status_code < 400 else f"HTTP {response.status_code}"
            ),
        )

        # Store metrics
        self.request_metrics.append(metrics)

        # Update endpoint statistics
        if self.config.enable_endpoint_stats:
            self._update_endpoint_stats(metrics)

        # Check for slow requests
        if (
            self.config.enable_slow_query_detection
            and duration_ms > self.config.slow_request_threshold_ms
        ):
            self.slow_requests_count += 1
            self.logger.warning(
                f"Slow request detected: {request.method} {request.url.path} "
                f"took {duration_ms:.2f}ms (threshold: {self.config.slow_request_threshold_ms}ms)"
            )

    def _update_endpoint_stats(self, metrics: RequestMetrics) -> None:
        """Update endpoint-specific statistics."""
        key = f"{metrics.method}:{metrics.path}"
        stats = self.endpoint_stats[key]

        # Initialize if new endpoint
        if stats.total_requests == 0:
            stats.path = metrics.path
            stats.method = metrics.method

        # Update basic counters
        stats.total_requests += 1
        stats.total_duration_ms += metrics.duration_ms
        stats.last_request = metrics.timestamp

        # Update duration statistics
        stats.min_duration_ms = min(stats.min_duration_ms, metrics.duration_ms)
        stats.max_duration_ms = max(stats.max_duration_ms, metrics.duration_ms)
        stats.recent_durations.append(metrics.duration_ms)

        # Update error statistics
        if metrics.error:
            stats.total_errors += 1

        # Calculate error rate
        stats.error_rate = (stats.total_errors / stats.total_requests) * 100

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        now = datetime.utcnow()

        # Filter recent metrics
        recent_cutoff = now - timedelta(minutes=self.config.metrics_retention_minutes)
        recent_metrics = [
            m for m in self.request_metrics if m.timestamp >= recent_cutoff
        ]

        if not recent_metrics:
            return {"message": "No recent metrics available"}

        # Calculate overall statistics
        durations = [m.duration_ms for m in recent_metrics]
        total_requests = len(recent_metrics)
        error_count = len([m for m in recent_metrics if m.error])

        summary = {
            "overview": {
                "total_requests": total_requests,
                "error_count": error_count,
                "error_rate_percent": (
                    (error_count / total_requests * 100) if total_requests > 0 else 0
                ),
                "slow_requests_count": self.slow_requests_count,
                "monitoring_period_minutes": self.config.metrics_retention_minutes,
            },
            "response_times": {
                "average_ms": mean(durations) if durations else 0,
                "median_ms": median(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
            },
            "system_metrics": self.system_metrics_cache,
            "alerts": {
                "memory_alerts": self.memory_alerts_count,
                "cpu_alerts": self.cpu_alerts_count,
                "slow_requests": self.slow_requests_count,
            },
            "timestamp": now.isoformat(),
        }

        # Add percentiles
        if durations:
            sorted_durations = sorted(durations)
            percentiles = {}
            for p in self.config.response_time_percentiles:
                index = int((p / 100) * len(sorted_durations)) - 1
                index = max(0, min(index, len(sorted_durations) - 1))
                percentiles[f"p{p}"] = sorted_durations[index]
            summary["response_times"]["percentiles"] = percentiles

        return summary

    def get_endpoint_statistics(self) -> dict[str, Any]:
        """Get per-endpoint statistics."""
        if not self.config.enable_endpoint_stats:
            return {"message": "Endpoint statistics disabled"}

        stats_data = {}
        for key, stats in self.endpoint_stats.items():
            if stats.total_requests > 0:
                avg_duration = stats.total_duration_ms / stats.total_requests
                recent_durations = list(stats.recent_durations)

                endpoint_data = {
                    "method": stats.method,
                    "path": stats.path,
                    "total_requests": stats.total_requests,
                    "total_errors": stats.total_errors,
                    "error_rate_percent": stats.error_rate,
                    "average_duration_ms": avg_duration,
                    "min_duration_ms": (
                        stats.min_duration_ms
                        if stats.min_duration_ms != float("inf")
                        else 0
                    ),
                    "max_duration_ms": stats.max_duration_ms,
                    "last_request": (
                        stats.last_request.isoformat() if stats.last_request else None
                    ),
                }

                # Add recent performance data
                if recent_durations:
                    endpoint_data.update(
                        {
                            "recent_average_ms": mean(recent_durations),
                            "recent_median_ms": median(recent_durations),
                            "recent_requests_count": len(recent_durations),
                        }
                    )

                stats_data[key] = endpoint_data

        return {
            "endpoints": stats_data,
            "total_endpoints": len(stats_data),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.request_metrics.clear()
        self.endpoint_stats.clear()
        self.slow_requests_count = 0
        self.memory_alerts_count = 0
        self.cpu_alerts_count = 0
        self.logger.info("Performance statistics reset")

    def force_garbage_collection(self) -> dict[str, Any]:
        """Force garbage collection and return memory info."""
        collected = gc.collect()
        memory_info = self.system_metrics.process.memory_info()

        return {
            "objects_collected": collected,
            "memory_before_mb": memory_info.rss / 1024 / 1024,
            "timestamp": datetime.utcnow().isoformat(),
        }


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""

    def __init__(self, app, config: Optional[PerformanceConfig] = None):
        """Initialize performance middleware."""
        super().__init__(app)
        self.config = config or PerformanceConfig()
        self.monitor = PerformanceMonitor(self.config)
        self.logger = logging.getLogger("api.performance")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        if not self.config.enable_monitoring:
            return await call_next(request)

        start_time = time.time()

        # Get request size
        # NOTE: We don't read the body here because BaseHTTPMiddleware has a bug
        # where reading the body can prevent endpoints from reading it later.
        # Request size tracking is less important than functional endpoints.
        request_size = 0

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Get response size
            response_size = 0
            if hasattr(response, "body"):
                with contextlib.suppress(Exception):
                    response_size = len(response.body)

            # Record metrics
            self.monitor.record_request(
                request=request,
                response=response,
                duration_ms=duration_ms,
                request_size=request_size,
                response_size=response_size,
            )

            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            # Record failed request
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Request failed after {duration_ms:.2f}ms: {str(e)}")
            raise


def setup_performance_monitoring(
    app: FastAPI, config: Optional[PerformanceConfig] = None
) -> None:
    """Setup performance monitoring middleware.

    Args:
        app: FastAPI application instance
        config: Performance monitoring configuration
    """
    if config is None:
        config = PerformanceConfig()

    app.add_middleware(PerformanceMiddleware, config=config)

    logger = logging.getLogger(__name__)
    logger.info("Performance monitoring middleware enabled")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor) -> None:
    """Set the global performance monitor instance."""
    global _performance_monitor
    _performance_monitor = monitor
