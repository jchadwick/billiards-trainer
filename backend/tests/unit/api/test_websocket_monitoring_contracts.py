"""Deterministic behavioral contracts for WebSocket connection monitoring."""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from backend.api.websocket import monitoring
from backend.api.websocket.monitoring import (
    ConnectionIssue,
    ConnectionMonitor,
    HealthStatus,
    LatencyMeasurement,
    QualityMetrics,
)


def _measurement(latency_ms: float, *, success: bool = True) -> LatencyMeasurement:
    return LatencyMeasurement(
        timestamp=datetime.now(timezone.utc),
        latency_ms=latency_ms,
        message_type="test",
        success=success,
    )


def test_latency_and_message_recording_builds_client_health_snapshot():
    monitor = ConnectionMonitor()

    assert monitor.get_client_health("new") is None
    monitor.record_latency("client", 10.0, "state", success=True)
    monitor.record_latency("client", 30.0, "state", success=True)
    monitor.record_latency("client", 500.0, "state", success=False)
    monitor.record_message_sent("client", 12)
    monitor.record_message_received("client", 7)
    for error in ("failed_send", "timeout", "disconnection", "protocol_error"):
        monitor.record_error("client", error)
    monitor.record_error("client", "unrecognized")

    health = monitor.get_client_health("client")
    assert health is not None
    assert health["latency"] == {
        "average_ms": 20.0,
        "min_ms": 10.0,
        "max_ms": 30.0,
        "jitter_ms": pytest.approx(14.1421356),
        "measurements": 3,
    }
    assert health["throughput"]["messages_sent"] == 1
    assert health["throughput"]["messages_received"] == 1
    assert health["throughput"]["bytes_sent"] == 12
    assert health["throughput"]["bytes_received"] == 7
    assert health["errors"] == {
        "failed_sends": 1,
        "timeouts": 1,
        "disconnections": 1,
        "protocol_errors": 1,
        "packet_loss_rate": pytest.approx(1 / 3),
    }
    assert health["quality_indicators"]["overall_score"] == 0.54


def test_latency_metrics_ignore_failed_samples_and_keep_empty_success_window():
    monitor = ConnectionMonitor()
    metrics = QualityMetrics(client_id="client")

    monitor._update_latency_metrics(metrics)
    assert metrics.average_latency_ms == 0.0
    assert metrics.min_latency_ms == float("inf")

    metrics.latency_measurements.append(_measurement(9.0, success=False))
    monitor._update_latency_metrics(metrics)
    assert metrics.average_latency_ms == 0.0
    assert metrics.packet_loss_rate == 0.0

    metrics.latency_measurements.append(_measurement(40.0))
    monitor._update_latency_metrics(metrics)
    assert metrics.average_latency_ms == 40.0
    assert metrics.min_latency_ms == metrics.max_latency_ms == 40.0
    assert metrics.packet_loss_rate == 0.5


def test_client_health_thresholds_cover_each_status_and_issue_type():
    monitor = ConnectionMonitor()
    examples = [
        (10.0, 0.0, HealthStatus.EXCELLENT),
        (25.0, 0.0, HealthStatus.GOOD),
        (75.0, 0.0, HealthStatus.FAIR),
        (150.0, 0.0, HealthStatus.POOR),
        (201.0, 0.0, HealthStatus.CRITICAL),
        (10.0, 0.11, HealthStatus.CRITICAL),
    ]

    for latency, packet_loss, expected in examples:
        metrics = QualityMetrics(client_id="client")
        metrics.average_latency_ms = latency
        metrics.packet_loss_rate = packet_loss
        assert monitor._assess_client_health(metrics) is expected

    metrics = QualityMetrics(client_id="issues")
    metrics.average_latency_ms = 101.0
    metrics.packet_loss_rate = 0.06
    metrics.disconnections = 6
    metrics.messages_sent = 2
    metrics.failed_sends = 1
    metrics.protocol_errors = 1
    assert monitor._identify_client_issues(metrics) == [
        ConnectionIssue.HIGH_LATENCY,
        ConnectionIssue.PACKET_LOSS,
        ConnectionIssue.FREQUENT_DISCONNECTS,
        ConnectionIssue.RATE_LIMITING,
        ConnectionIssue.PROTOCOL_ERRORS,
    ]
    metrics.average_latency_ms = 0.0
    metrics.packet_loss_rate = 0.0
    metrics.disconnections = 5
    metrics.failed_sends = 0
    metrics.protocol_errors = 0
    assert monitor._identify_client_issues(metrics) == []


def test_system_health_summary_and_quality_scores_reflect_connection_distribution():
    monitor = ConnectionMonitor()
    summary = monitor.get_health_summary()
    assert summary["total_clients"] == 0
    assert summary["system_health"] == "no_connections"
    assert summary["clients"] == []
    assert monitor.get_system_health()["overall_status"] == "no_connections"

    healthy = QualityMetrics(client_id="healthy")
    healthy.overall_health = HealthStatus.GOOD
    healthy.average_latency_ms = 20.0
    healthy.bytes_sent = 100_000
    healthy.bytes_received = 50_000
    degraded = QualityMetrics(client_id="degraded")
    degraded.overall_health = HealthStatus.FAIR
    critical = QualityMetrics(client_id="critical")
    critical.overall_health = HealthStatus.CRITICAL
    monitor.client_metrics.update(
        {"healthy": healthy, "degraded": degraded, "critical": critical}
    )

    result = monitor.get_system_health()
    assert result["overall_status"] == "critical"
    assert result["connections"] == {
        "total": 3,
        "healthy": 1,
        "degraded": 1,
        "critical": 1,
        "health_distribution": {"good": 1, "fair": 1, "critical": 1},
    }
    assert result["performance"]["average_latency_ms"] == pytest.approx(20 / 3)
    assert result["performance"]["total_throughput_mbps"] > 0
    assert set(monitor.get_health_summary()["clients"][0]) == {
        "client_id",
        "health_status",
        "average_latency_ms",
        "packet_loss_rate",
        "connection_stability",
        "active_issues_count",
        "uptime_seconds",
    }

    assert monitor._calculate_quality_score(healthy) == 0.96
    healthy.average_latency_ms = 250.0
    healthy.packet_loss_rate = 0.2
    healthy.disconnections = 10
    assert monitor._calculate_quality_score(healthy) == 0.0
    healthy.average_latency_ms = 0.0
    healthy.packet_loss_rate = 0.0
    healthy.disconnections = 0
    assert monitor._calculate_quality_score(healthy) == 1.0


def test_overall_status_distinguishes_degraded_healthy_and_critical():
    monitor = ConnectionMonitor()
    monitor.system_metrics.total_connections = 2
    monitor.system_metrics.healthy_connections = 1
    monitor.system_metrics.degraded_connections = 2
    assert monitor._get_overall_system_status() == "degraded"

    monitor.system_metrics.degraded_connections = 0
    monitor.system_metrics.critical_connections = 1
    assert monitor._get_overall_system_status() == "critical"

    monitor.system_metrics.critical_connections = 0
    assert monitor._get_overall_system_status() == "healthy"


@pytest.mark.asyncio
async def test_start_and_stop_monitoring_are_idempotent_and_cancel_owned_task(
    monkeypatch: pytest.MonkeyPatch,
):
    monitor = ConnectionMonitor()
    entered = asyncio.Event()

    async def waiting_loop():
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(monitor, "_monitoring_loop", waiting_loop)
    await monitor.start_monitoring()
    first_task = monitor.monitoring_task
    await entered.wait()
    await monitor.start_monitoring()
    assert monitor.monitoring_task is first_task
    assert monitor.monitoring_active is True

    await monitor.stop_monitoring()
    assert monitor.monitoring_active is False
    assert monitor.monitoring_task is None
    assert first_task is not None
    assert first_task.cancelled()
    await monitor.stop_monitoring()


@pytest.mark.asyncio
async def test_monitoring_loop_prunes_disconnected_clients_and_updates_live_clients(
    monkeypatch: pytest.MonkeyPatch,
):
    monitor = ConnectionMonitor()
    monitor.update_interval = 0.01
    monitor.client_metrics = {
        "connected": QualityMetrics(client_id="connected"),
        "gone": QualityMetrics(client_id="gone"),
    }
    monkeypatch.setattr(
        monitoring.websocket_handler, "connections", {"connected": object()}
    )
    ping = AsyncMock()
    monkeypatch.setattr(monitor, "_send_health_ping", ping)
    sleeps = 0

    async def controlled_sleep(_duration: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            monitor.monitoring_active = False

    monkeypatch.setattr(monitoring.asyncio, "sleep", controlled_sleep)
    monitor.monitoring_active = True
    await monitor._monitoring_loop()

    assert sleeps == 1
    assert set(monitor.client_metrics) == {"connected"}
    assert monitor.client_metrics["connected"].overall_health is HealthStatus.EXCELLENT
    assert monitor.system_metrics.total_connections == 1
    assert len(monitor.performance_history) == 1
    assert ping.await_count == 1


@pytest.mark.asyncio
async def test_monitoring_loop_recovers_after_a_client_update_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monitor = ConnectionMonitor()
    monitor.client_metrics["client"] = QualityMetrics(client_id="client")
    monkeypatch.setattr(
        monitoring.websocket_handler, "connections", {"client": object()}
    )
    sleeps = 0
    updates = 0

    async def controlled_sleep(_duration: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            monitor.monitoring_active = False

    async def sometimes_failing_update(_client_id: str) -> None:
        nonlocal updates
        updates += 1
        if updates == 1:
            raise RuntimeError("transient client update failure")

    monkeypatch.setattr(monitoring.asyncio, "sleep", controlled_sleep)
    monkeypatch.setattr(monitor, "_update_client_metrics", sometimes_failing_update)
    monitor.monitoring_active = True
    await monitor._monitoring_loop()

    assert updates == 2
    assert monitor.system_metrics.total_connections == 1
    assert len(monitor.performance_history) == 1


@pytest.mark.asyncio
async def test_update_client_metrics_prunes_old_samples_and_skips_missing_clients(
    monkeypatch: pytest.MonkeyPatch,
):
    monitor = ConnectionMonitor()
    await monitor._update_client_metrics("missing")
    metrics = QualityMetrics(
        client_id="client",
        measurement_window=timedelta(seconds=30),
        latency_measurements=deque(
            [
                LatencyMeasurement(
                    datetime.now(timezone.utc) - timedelta(seconds=31), 12, "old"
                ),
                _measurement(15),
            ]
        ),
    )
    monitor.client_metrics["client"] = metrics
    monkeypatch.setattr(monitoring.websocket_handler, "connections", {})
    ping = AsyncMock()
    monkeypatch.setattr(monitor, "_send_health_ping", ping)

    await monitor._update_client_metrics("client")

    assert len(metrics.latency_measurements) == 1
    assert metrics.overall_health is HealthStatus.EXCELLENT
    assert metrics.last_updated.tzinfo == timezone.utc
    ping.assert_not_awaited()

    monkeypatch.setattr(
        monitoring.websocket_handler, "connections", {"client": object()}
    )
    await monitor._update_client_metrics("client")
    ping.assert_awaited_once_with("client")


@pytest.mark.asyncio
async def test_health_ping_records_success_false_and_exception_paths(monkeypatch):
    monitor = ConnectionMonitor()
    sender = AsyncMock(side_effect=[True, False, OSError("disconnected")])
    monkeypatch.setattr(monitoring.websocket_handler, "send_to_client", sender)

    await monitor._send_health_ping("client")
    successful = monitor.client_metrics["client"]
    assert successful.latency_measurements[-1].success is True
    assert successful.latency_measurements[-1].message_type == "health_ping"

    await monitor._send_health_ping("client")
    assert successful.latency_measurements[-1].success is False
    assert successful.failed_sends == 1

    await monitor._send_health_ping("client")
    assert successful.failed_sends == 2
    assert sender.await_count == 3
    for await_item in sender.await_args_list:
        client_id, message = await_item.args
        assert client_id == "client"
        assert message["type"] == "ping"
        datetime.fromisoformat(message["data"]["timestamp"])


@pytest.mark.asyncio
async def test_alerts_reach_sync_and_async_handlers_and_isolate_handler_failure(
    caplog: pytest.LogCaptureFixture,
):
    monitor = ConnectionMonitor()
    monitor.system_metrics.average_system_latency = 151.0
    monitor.client_metrics["bad"] = QualityMetrics(client_id="bad")
    monitor.client_metrics["bad"].overall_health = HealthStatus.CRITICAL
    received: list[dict] = []

    def synchronous(alert):
        received.append(alert)

    async def asynchronous(alert):
        received.append(alert)

    def broken(_alert):
        raise RuntimeError("alert sink unavailable")

    monitor.add_alert_handler(synchronous)
    monitor.add_alert_handler(asynchronous)
    monitor.add_alert_handler(broken)
    await monitor._check_and_send_alerts()

    assert [alert["type"] for alert in received] == [
        "critical_connections",
        "critical_connections",
        "high_system_latency",
        "high_system_latency",
    ]
    assert [alert["type"] for alert in monitor.system_metrics.active_alerts] == [
        "critical_connections",
        "high_system_latency",
    ]
    assert "Error in alert handler" in caplog.text

    monitor.remove_alert_handler(synchronous)
    monitor.remove_alert_handler(synchronous)
    assert synchronous not in monitor.alert_handlers
    assert asynchronous in monitor.alert_handlers


def test_performance_history_is_bounded_and_snapshot_captures_current_system_stats():
    monitor = ConnectionMonitor()
    monitor.performance_history = deque(maxlen=1)
    monitor.system_metrics.total_connections = 4
    monitor.system_metrics.healthy_connections = 3
    monitor.system_metrics.average_system_latency = 22.5
    monitor.system_metrics.total_throughput_mbps = 1.25

    monitor._store_performance_snapshot()
    first = monitor.performance_history[0]
    monitor.system_metrics.total_connections = 5
    monitor._store_performance_snapshot()

    assert len(monitor.performance_history) == 1
    assert monitor.performance_history[0]["total_connections"] == 5
    assert monitor.performance_history[0]["healthy_connections"] == 3
    assert monitor.performance_history[0]["average_latency"] == 22.5
    assert monitor.performance_history[0]["total_throughput"] == 1.25
    assert isinstance(first["timestamp"], datetime)
