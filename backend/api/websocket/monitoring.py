"""WebSocket connection health monitoring and quality indicators."""

import asyncio
import contextlib
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from config import config_manager

from .handler import websocket_handler

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Connection health status levels."""

    EXCELLENT = "excellent"  # < 20ms latency, no packet loss
    GOOD = "good"  # < 50ms latency, minimal packet loss
    FAIR = "fair"  # < 100ms latency, some packet loss
    POOR = "poor"  # < 200ms latency, significant packet loss
    CRITICAL = "critical"  # > 200ms latency or high packet loss


class ConnectionIssue(Enum):
    """Types of connection issues."""

    HIGH_LATENCY = "high_latency"
    PACKET_LOSS = "packet_loss"
    BANDWIDTH_LIMIT = "bandwidth_limit"
    FREQUENT_DISCONNECTS = "frequent_disconnects"
    RATE_LIMITING = "rate_limiting"
    AUTHENTICATION_ISSUES = "authentication_issues"
    PROTOCOL_ERRORS = "protocol_errors"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    timestamp: datetime
    latency_ms: float
    message_type: str
    success: bool = True


@dataclass
class QualityMetrics:
    """Connection quality metrics over time."""

    client_id: str
    measurement_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    # Latency metrics
    latency_measurements: deque = field(
        default_factory=lambda: deque(
            maxlen=config_manager.get(
                "api.websocket.monitoring.measurements.max_latency_measurements", 1000
            )
        )
    )
    average_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0  # Standard deviation

    # Throughput metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Error metrics
    failed_sends: int = 0
    timeouts: int = 0
    disconnections: int = 0
    protocol_errors: int = 0

    # Quality indicators
    packet_loss_rate: float = 0.0
    bandwidth_utilization: float = 0.0  # 0.0 to 1.0
    connection_stability: float = 1.0  # 0.0 to 1.0

    # Health assessment
    overall_health: HealthStatus = HealthStatus.EXCELLENT
    active_issues: list[ConnectionIssue] = field(default_factory=list)

    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    monitoring_started: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics."""

    total_connections: int = 0
    healthy_connections: int = 0
    degraded_connections: int = 0
    critical_connections: int = 0

    average_system_latency: float = 0.0
    total_throughput_mbps: float = 0.0
    system_load: float = 0.0  # 0.0 to 1.0

    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    network_usage_percent: float = 0.0

    # Error rates
    total_errors: int = 0
    error_rate_per_minute: float = 0.0

    # Alerts
    active_alerts: list[dict[str, Any]] = field(default_factory=list)

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConnectionMonitor:
    """Advanced connection health monitoring and quality assessment."""

    def __init__(self):
        self.client_metrics: dict[str, QualityMetrics] = {}
        self.system_metrics = SystemHealthMetrics()
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_handlers: list[callable] = []

        # Configuration
        self.update_interval = config_manager.get(
            "api.websocket.monitoring.intervals.update_interval_seconds", 10.0
        )
        self.latency_threshold_ms = config_manager.get(
            "api.websocket.monitoring.thresholds.latency_ms", 100.0
        )
        self.packet_loss_threshold = config_manager.get(
            "api.websocket.monitoring.thresholds.packet_loss_rate", 0.05
        )
        self.bandwidth_threshold_mbps = config_manager.get(
            "api.websocket.monitoring.thresholds.bandwidth_mbps", 10.0
        )

        # Performance tracking
        performance_history_max = config_manager.get(
            "api.websocket.monitoring.intervals.performance_history_max_count", 288
        )
        self.performance_history = deque(maxlen=performance_history_max)

    async def start_monitoring(self):
        """Start connection monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Connection monitoring started")

    async def stop_monitoring(self):
        """Stop connection monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

            self.monitoring_task = None

        logger.info("Connection monitoring stopped")

    def record_latency(
        self,
        client_id: str,
        latency_ms: float,
        message_type: str = "ping",
        success: bool = True,
    ):
        """Record a latency measurement for a client."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = QualityMetrics(client_id=client_id)

        metrics = self.client_metrics[client_id]
        measurement = LatencyMeasurement(
            timestamp=datetime.now(timezone.utc),
            latency_ms=latency_ms,
            message_type=message_type,
            success=success,
        )

        metrics.latency_measurements.append(measurement)
        self._update_latency_metrics(metrics)

    def record_message_sent(self, client_id: str, message_size: int):
        """Record a message sent to a client."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = QualityMetrics(client_id=client_id)

        metrics = self.client_metrics[client_id]
        metrics.messages_sent += 1
        metrics.bytes_sent += message_size

    def record_message_received(self, client_id: str, message_size: int):
        """Record a message received from a client."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = QualityMetrics(client_id=client_id)

        metrics = self.client_metrics[client_id]
        metrics.messages_received += 1
        metrics.bytes_received += message_size

    def record_error(self, client_id: str, error_type: str):
        """Record an error for a client."""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = QualityMetrics(client_id=client_id)

        metrics = self.client_metrics[client_id]

        if error_type == "failed_send":
            metrics.failed_sends += 1
        elif error_type == "timeout":
            metrics.timeouts += 1
        elif error_type == "disconnection":
            metrics.disconnections += 1
        elif error_type == "protocol_error":
            metrics.protocol_errors += 1

    def get_client_health(self, client_id: str) -> Optional[dict[str, Any]]:
        """Get health information for a specific client."""
        if client_id not in self.client_metrics:
            return None

        metrics = self.client_metrics[client_id]

        return {
            "client_id": client_id,
            "health_status": metrics.overall_health.value,
            "latency": {
                "average_ms": metrics.average_latency_ms,
                "min_ms": (
                    metrics.min_latency_ms
                    if metrics.min_latency_ms != float("inf")
                    else 0
                ),
                "max_ms": metrics.max_latency_ms,
                "jitter_ms": metrics.latency_jitter_ms,
                "measurements": len(metrics.latency_measurements),
            },
            "throughput": {
                "messages_sent": metrics.messages_sent,
                "messages_received": metrics.messages_received,
                "bytes_sent": metrics.bytes_sent,
                "bytes_received": metrics.bytes_received,
                "bandwidth_utilization": metrics.bandwidth_utilization,
            },
            "errors": {
                "failed_sends": metrics.failed_sends,
                "timeouts": metrics.timeouts,
                "disconnections": metrics.disconnections,
                "protocol_errors": metrics.protocol_errors,
                "packet_loss_rate": metrics.packet_loss_rate,
            },
            "quality_indicators": {
                "connection_stability": metrics.connection_stability,
                "overall_score": self._calculate_quality_score(metrics),
            },
            "active_issues": [issue.value for issue in metrics.active_issues],
            "monitoring_duration": (
                datetime.now(timezone.utc) - metrics.monitoring_started
            ).total_seconds(),
            "last_updated": metrics.last_updated.isoformat(),
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health information."""
        self._update_system_metrics()

        return {
            "overall_status": self._get_overall_system_status(),
            "connections": {
                "total": self.system_metrics.total_connections,
                "healthy": self.system_metrics.healthy_connections,
                "degraded": self.system_metrics.degraded_connections,
                "critical": self.system_metrics.critical_connections,
                "health_distribution": self._get_health_distribution(),
            },
            "performance": {
                "average_latency_ms": self.system_metrics.average_system_latency,
                "total_throughput_mbps": self.system_metrics.total_throughput_mbps,
                "system_load": self.system_metrics.system_load,
            },
            "resources": {
                "cpu_usage_percent": self.system_metrics.cpu_usage_percent,
                "memory_usage_percent": self.system_metrics.memory_usage_percent,
                "network_usage_percent": self.system_metrics.network_usage_percent,
            },
            "errors": {
                "total_errors": self.system_metrics.total_errors,
                "error_rate_per_minute": self.system_metrics.error_rate_per_minute,
            },
            "alerts": self.system_metrics.active_alerts,
            "last_updated": self.system_metrics.last_updated.isoformat(),
        }

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of health status for all clients."""
        client_summaries = []

        for client_id, metrics in self.client_metrics.items():
            summary = {
                "client_id": client_id,
                "health_status": metrics.overall_health.value,
                "average_latency_ms": metrics.average_latency_ms,
                "packet_loss_rate": metrics.packet_loss_rate,
                "connection_stability": metrics.connection_stability,
                "active_issues_count": len(metrics.active_issues),
                "uptime_seconds": (
                    datetime.now(timezone.utc) - metrics.monitoring_started
                ).total_seconds(),
            }
            client_summaries.append(summary)

        return {
            "total_clients": len(client_summaries),
            "system_health": self._get_overall_system_status(),
            "clients": client_summaries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_alert_handler(self, handler: callable):
        """Add a handler for health alerts."""
        self.alert_handlers.append(handler)

    def remove_alert_handler(self, handler: callable):
        """Remove an alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.update_interval)

                # Update metrics for all clients
                for client_id in list(self.client_metrics.keys()):
                    if client_id not in websocket_handler.connections:
                        # Client disconnected, clean up metrics
                        del self.client_metrics[client_id]
                        continue

                    await self._update_client_metrics(client_id)

                # Update system metrics
                self._update_system_metrics()

                # Check for alerts
                await self._check_and_send_alerts()

                # Store performance history
                self._store_performance_snapshot()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _update_client_metrics(self, client_id: str):
        """Update metrics for a specific client."""
        if client_id not in self.client_metrics:
            return

        metrics = self.client_metrics[client_id]

        # Clean old measurements
        cutoff_time = datetime.now(timezone.utc) - metrics.measurement_window
        while (
            metrics.latency_measurements
            and metrics.latency_measurements[0].timestamp < cutoff_time
        ):
            metrics.latency_measurements.popleft()

        # Update health assessment
        metrics.overall_health = self._assess_client_health(metrics)
        metrics.active_issues = self._identify_client_issues(metrics)
        metrics.last_updated = datetime.now(timezone.utc)

        # Send ping to measure current latency
        if client_id in websocket_handler.connections:
            await self._send_health_ping(client_id)

    def _update_latency_metrics(self, metrics: QualityMetrics):
        """Update latency-related metrics."""
        if not metrics.latency_measurements:
            return

        # Filter successful measurements
        successful_measurements = [
            m.latency_ms for m in metrics.latency_measurements if m.success
        ]

        if not successful_measurements:
            return

        metrics.average_latency_ms = statistics.mean(successful_measurements)
        metrics.min_latency_ms = min(successful_measurements)
        metrics.max_latency_ms = max(successful_measurements)

        if len(successful_measurements) > 1:
            metrics.latency_jitter_ms = statistics.stdev(successful_measurements)

        # Calculate packet loss rate
        total_measurements = len(metrics.latency_measurements)
        failed_measurements = total_measurements - len(successful_measurements)
        metrics.packet_loss_rate = (
            failed_measurements / total_measurements if total_measurements > 0 else 0.0
        )

    def _assess_client_health(self, metrics: QualityMetrics) -> HealthStatus:
        """Assess overall health status for a client."""
        poor_latency = config_manager.get(
            "api.websocket.monitoring.health_status.poor_max_latency_ms", 200.0
        )
        poor_packet_loss = config_manager.get(
            "api.websocket.monitoring.health_status.poor_max_packet_loss", 0.1
        )
        fair_latency = config_manager.get(
            "api.websocket.monitoring.health_status.fair_max_latency_ms", 100.0
        )
        fair_packet_loss = config_manager.get(
            "api.websocket.monitoring.health_status.fair_max_packet_loss", 0.05
        )
        good_latency = config_manager.get(
            "api.websocket.monitoring.health_status.good_max_latency_ms", 50.0
        )
        good_packet_loss = config_manager.get(
            "api.websocket.monitoring.health_status.good_max_packet_loss", 0.02
        )
        excellent_latency = config_manager.get(
            "api.websocket.monitoring.health_status.excellent_max_latency_ms", 20.0
        )
        excellent_packet_loss = config_manager.get(
            "api.websocket.monitoring.health_status.excellent_max_packet_loss", 0.01
        )

        if (
            metrics.average_latency_ms > poor_latency
            or metrics.packet_loss_rate > poor_packet_loss
        ):
            return HealthStatus.CRITICAL
        elif (
            metrics.average_latency_ms > fair_latency
            or metrics.packet_loss_rate > fair_packet_loss
        ):
            return HealthStatus.POOR
        elif (
            metrics.average_latency_ms > good_latency
            or metrics.packet_loss_rate > good_packet_loss
        ):
            return HealthStatus.FAIR
        elif (
            metrics.average_latency_ms > excellent_latency
            or metrics.packet_loss_rate > excellent_packet_loss
        ):
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT

    def _identify_client_issues(self, metrics: QualityMetrics) -> list[ConnectionIssue]:
        """Identify specific issues for a client."""
        issues = []

        if metrics.average_latency_ms > self.latency_threshold_ms:
            issues.append(ConnectionIssue.HIGH_LATENCY)

        if metrics.packet_loss_rate > self.packet_loss_threshold:
            issues.append(ConnectionIssue.PACKET_LOSS)

        frequent_disconnects_threshold = config_manager.get(
            "api.websocket.monitoring.thresholds.frequent_disconnects_count", 5
        )
        if metrics.disconnections > frequent_disconnects_threshold:
            issues.append(ConnectionIssue.FREQUENT_DISCONNECTS)

        failed_sends_rate = config_manager.get(
            "api.websocket.monitoring.thresholds.failed_sends_rate", 0.1
        )
        if metrics.failed_sends > metrics.messages_sent * failed_sends_rate:
            issues.append(ConnectionIssue.RATE_LIMITING)

        if metrics.protocol_errors > 0:
            issues.append(ConnectionIssue.PROTOCOL_ERRORS)

        return issues

    def _update_system_metrics(self):
        """Update overall system metrics."""
        self.system_metrics.total_connections = len(self.client_metrics)

        # Count health distributions
        health_counts = defaultdict(int)
        total_latency = 0
        total_throughput = 0

        for metrics in self.client_metrics.values():
            health_counts[metrics.overall_health] += 1
            total_latency += metrics.average_latency_ms

            # Calculate throughput (rough estimation)
            uptime = (
                datetime.now(timezone.utc) - metrics.monitoring_started
            ).total_seconds()
            if uptime > 0:
                throughput_mbps = (
                    (metrics.bytes_sent + metrics.bytes_received)
                    * 8
                    / (uptime * 1_000_000)
                )
                total_throughput += throughput_mbps

        self.system_metrics.healthy_connections = (
            health_counts[HealthStatus.EXCELLENT] + health_counts[HealthStatus.GOOD]
        )
        self.system_metrics.degraded_connections = health_counts[HealthStatus.FAIR]
        self.system_metrics.critical_connections = (
            health_counts[HealthStatus.POOR] + health_counts[HealthStatus.CRITICAL]
        )

        if self.system_metrics.total_connections > 0:
            self.system_metrics.average_system_latency = (
                total_latency / self.system_metrics.total_connections
            )

        self.system_metrics.total_throughput_mbps = total_throughput
        self.system_metrics.last_updated = datetime.now(timezone.utc)

    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """Calculate an overall quality score (0.0 to 1.0)."""
        # Latency score (inverse relationship)
        if metrics.average_latency_ms == 0:
            latency_score = 1.0
        else:
            latency_score = max(
                0.0, 1.0 - (metrics.average_latency_ms / 200.0)
            )  # 200ms = 0 score

        # Packet loss score
        packet_loss_score = max(
            0.0, 1.0 - (metrics.packet_loss_rate * 10)
        )  # 10% loss = 0 score

        # Stability score based on disconnections
        if metrics.disconnections == 0:
            stability_score = 1.0
        else:
            stability_score = max(
                0.0, 1.0 - (metrics.disconnections / 10.0)
            )  # 10 disconnects = 0 score

        # Weighted average
        overall_score = (
            latency_score * 0.4 + packet_loss_score * 0.4 + stability_score * 0.2
        )
        return round(overall_score, 3)

    def _get_overall_system_status(self) -> str:
        """Get overall system health status."""
        if self.system_metrics.critical_connections > 0:
            return "critical"
        elif (
            self.system_metrics.degraded_connections
            > self.system_metrics.healthy_connections
        ):
            return "degraded"
        elif self.system_metrics.total_connections == 0:
            return "no_connections"
        else:
            return "healthy"

    def _get_health_distribution(self) -> dict[str, int]:
        """Get distribution of health statuses."""
        distribution = defaultdict(int)
        for metrics in self.client_metrics.values():
            distribution[metrics.overall_health.value] += 1
        return dict(distribution)

    async def _send_health_ping(self, client_id: str):
        """Send a health check ping to a client."""
        start_time = time.time()

        try:
            success = await websocket_handler.send_to_client(
                client_id,
                {
                    "type": "ping",
                    "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
                },
            )

            if success:
                # Estimate latency (this is not precise without client response)
                estimated_latency = (time.time() - start_time) * 1000
                self.record_latency(client_id, estimated_latency, "health_ping", True)
            else:
                self.record_latency(client_id, 0, "health_ping", False)
                self.record_error(client_id, "failed_send")

        except Exception as e:
            logger.debug(f"Health ping failed for {client_id}: {e}")
            self.record_error(client_id, "failed_send")

    async def _check_and_send_alerts(self):
        """Check for alert conditions and send notifications."""
        alerts = []

        # Check for critical connections
        critical_clients = [
            client_id
            for client_id, metrics in self.client_metrics.items()
            if metrics.overall_health == HealthStatus.CRITICAL
        ]

        if critical_clients:
            alerts.append(
                {
                    "type": "critical_connections",
                    "message": f"{len(critical_clients)} connections in critical state",
                    "clients": critical_clients,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Check for high system latency
        high_latency_threshold = config_manager.get(
            "api.websocket.monitoring.thresholds.high_system_latency_ms", 150.0
        )
        if self.system_metrics.average_system_latency > high_latency_threshold:
            alerts.append(
                {
                    "type": "high_system_latency",
                    "message": f"System average latency: {self.system_metrics.average_system_latency:.1f}ms",
                    "threshold": high_latency_threshold,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Update active alerts
        self.system_metrics.active_alerts = alerts

        # Send alerts to handlers
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

    def _store_performance_snapshot(self):
        """Store current performance snapshot for historical analysis."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc),
            "total_connections": self.system_metrics.total_connections,
            "healthy_connections": self.system_metrics.healthy_connections,
            "average_latency": self.system_metrics.average_system_latency,
            "total_throughput": self.system_metrics.total_throughput_mbps,
        }

        self.performance_history.append(snapshot)


# Global monitoring instance
connection_monitor = ConnectionMonitor()
