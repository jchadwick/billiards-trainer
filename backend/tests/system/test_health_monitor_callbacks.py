"""System health callbacks support both synchronous and asynchronous handlers."""

import pytest

from backend.system.health_monitor import ComponentHealth, HealthMonitor


@pytest.mark.system
@pytest.mark.asyncio
async def test_health_and_alert_callbacks_support_sync_and_async_handlers():
    """Both callback styles receive health updates and unhealthy alerts."""
    monitor = HealthMonitor()
    health_notifications: list[str] = []
    alerts: list[str] = []

    def sync_health_callback(_health):
        health_notifications.append("sync")

    async def async_health_callback(_health):
        health_notifications.append("async")

    def sync_alert_callback(_level, message, _details):
        alerts.append(f"sync: {message}")

    async def async_alert_callback(_level, message, _details):
        alerts.append(f"async: {message}")

    monitor.add_health_callback(sync_health_callback)
    monitor.add_health_callback(async_health_callback)
    monitor.add_alert_callback(sync_alert_callback)
    monitor.add_alert_callback(async_alert_callback)
    monitor._component_health["core"] = ComponentHealth(
        component="core", status="unhealthy"
    )

    await monitor._notify_health_callbacks()
    await monitor._check_alerts()

    assert health_notifications == ["sync", "async"]
    assert alerts == [
        "sync: Component core is unhealthy",
        "async: Component core is unhealthy",
    ]
