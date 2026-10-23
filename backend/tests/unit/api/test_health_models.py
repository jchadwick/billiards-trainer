"""API contracts for the standalone health and metrics models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.api.models.health_models import (
    AlertLevelEnum,
    AlertModel,
    AlertRequest,
    AlertsResponse,
    HealthCheckRequest,
    HealthCheckResult,
    HealthStatusEnum,
    HealthStatusModel,
    HealthStatusResponse,
    IntegrationMetricsModel,
    MetricDataPoint,
    MetricModel,
    MetricsCollectionModel,
    MetricsRequest,
    MetricsResponse,
    MetricTypeEnum,
    PerformanceMetricsModel,
    ProcessMetricsModel,
    ServiceHealthModel,
    ServiceHealthResponse,
    ServiceTypeEnum,
    SystemResourcesModel,
    SystemStatusResponse,
)


def test_health_models_build_nested_status_and_response_payloads() -> None:
    checked_at = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)
    service = ServiceHealthModel(
        name="vision",
        type=ServiceTypeEnum.VISION,
        status=HealthStatusEnum.HEALTHY,
        last_check=checked_at,
        response_time=0.012,
        details={"frames_per_second": 25.0},
        dependencies=["camera"],
        version="rust-openvino",
        uptime=3600.0,
    )
    check = HealthCheckResult(
        check_name="vision-ready",
        status=HealthStatusEnum.HEALTHY,
        duration=0.003,
        timestamp=checked_at,
    )
    resources = SystemResourcesModel(
        cpu_usage_percent=70,
        memory_usage_bytes=700,
        memory_total_bytes=1000,
        memory_usage_percent=70,
        disk_usage_bytes=200,
        disk_total_bytes=1000,
        disk_usage_percent=20,
        network_rx_bytes=10,
        network_tx_bytes=12,
        open_file_descriptors=4,
        load_average=[0.4, 0.3, 0.2],
    )
    process = ProcessMetricsModel(
        pid=41,
        threads_count=3,
        memory_rss=700,
        memory_vms=900,
        cpu_percent=55.0,
        memory_percent=4.0,
        create_time=checked_at.timestamp(),
        num_fds=4,
        num_connections=2,
    )
    status = HealthStatusModel(
        status=HealthStatusEnum.HEALTHY,
        timestamp=checked_at,
        version="1.2.3",
        uptime=120.0,
        services={"vision": service},
        health_checks=[check],
        system_resources=resources,
        process_metrics=process,
        performance_metrics=PerformanceMetricsModel(
            requests_total=12,
            requests_per_second=2.5,
            response_time_avg=0.01,
            response_time_p50=0.008,
            response_time_p95=0.02,
            response_time_p99=0.03,
            error_rate=0.01,
            cache_hit_rate=0.8,
            active_connections=2,
            queue_size=1,
        ),
        integration_metrics=IntegrationMetricsModel(
            modules_initialized=3,
            services_healthy=2,
            total_services=3,
            events_processed=8,
            cache_entries=4,
            background_tasks_active=1,
            integration_uptime=120,
            last_health_check=checked_at,
        ),
        environment="test",
        build_info={"revision": "abc123"},
    )
    response = HealthStatusResponse(
        health=status, checks_performed=1, processing_time=0.004
    )

    encoded = response.model_dump(mode="json")
    assert encoded["health"]["services"]["vision"]["type"] == "vision"
    assert encoded["health"]["services"]["vision"]["details"]["frames_per_second"] == 25
    assert encoded["health"]["system_resources"]["load_average"] == [0.4, 0.3, 0.2]
    assert encoded["health"]["health_checks"][0]["check_name"] == "vision-ready"
    assert encoded["checks_performed"] == 1


def test_metric_and_alert_models_preserve_values_tags_and_resolution_state() -> None:
    collected_at = datetime(2026, 10, 21, 18, 5, tzinfo=timezone.utc)
    point = MetricDataPoint(
        timestamp=collected_at,
        value=25.0,
        tags={"source": "vision", "unit": "fps"},
    )
    metric = MetricModel(
        name="observations_per_second",
        type=MetricTypeEnum.GAUGE,
        unit="frames/s",
        current_value=25.0,
        data_points=[point],
        labels={"runtime": "openvino"},
    )
    collection = MetricsCollectionModel(
        timestamp=collected_at,
        metrics=[metric],
        collection_duration=0.002,
        total_metrics=1,
        metadata={"sampled": True},
    )
    alert = AlertModel(
        id="camera-stale",
        level=AlertLevelEnum.WARNING,
        title="Camera state is stale",
        message="No fresh frame was observed.",
        source="vision",
        timestamp=collected_at,
        resolved=True,
        resolved_at=collected_at,
        tags={"component": "camera"},
        metadata={"last_age_seconds": 0.6},
    )
    response = MetricsResponse(metrics=collection, processing_time=0.003)

    assert response.metrics.metrics[0].data_points[0].value == 25.0
    assert response.metrics.metrics[0].data_points[0].tags["unit"] == "fps"
    assert alert.resolved
    assert alert.resolved_at == collected_at
    assert alert.level is AlertLevelEnum.WARNING


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SystemResourcesModel,
            {
                "cpu_usage_percent": 100.1,
                "memory_usage_bytes": 1,
                "memory_total_bytes": 1,
                "memory_usage_percent": 50,
            },
        ),
        (
            SystemResourcesModel,
            {
                "cpu_usage_percent": 10,
                "memory_usage_bytes": 1,
                "memory_total_bytes": 1,
                "memory_usage_percent": -0.1,
            },
        ),
        (PerformanceMetricsModel, {"error_rate": 1.1}),
        (PerformanceMetricsModel, {"cache_hit_rate": -0.1}),
        (
            HealthCheckResult,
            {
                "check_name": "negative-duration",
                "status": "healthy",
                "duration": -0.01,
            },
        ),
        (HealthCheckRequest, {"timeout": 0}),
        (MetricsRequest, {"time_range": 0}),
        (AlertRequest, {"limit": 1001}),
        (
            ServiceHealthModel,
            {
                "name": "vision",
                "type": "vision",
                "status": "healthy",
                "response_time": -0.1,
            },
        ),
    ],
)
def test_health_models_reject_invalid_ranges(model: type, payload: dict) -> None:
    with pytest.raises(ValidationError):
        model(**payload)


def test_request_and_summary_models_apply_defaults_and_bounds() -> None:
    health_request = HealthCheckRequest(services=["vision"])
    metrics_request = MetricsRequest(metric_names=["fps"], time_range=60)
    alert_request = AlertRequest(level=AlertLevelEnum.CRITICAL, limit=20)
    status = SystemStatusResponse(
        status=HealthStatusEnum.DEGRADED,
        healthy_services=2,
        total_services=3,
        uptime=10,
        last_check=datetime.now(timezone.utc),
        critical_alerts=1,
        processing_time=0.01,
    )
    alerts = AlertsResponse(
        alerts=[], total_alerts=0, active_alerts=0, processing_time=0
    )
    service_response = ServiceHealthResponse(
        service=ServiceHealthModel(name="api", type="api", status="healthy"),
        processing_time=0,
    )
    assert health_request.include_details
    assert health_request.include_metrics
    assert health_request.include_system_resources
    assert metrics_request.include_tags
    assert metrics_request.format == "json"
    assert alert_request.limit == 20
    assert alert_request.resolved is None
    assert status.critical_alerts == 1
    assert alerts.success
    assert alerts.active_alerts == 0
    assert service_response.service.name == "api"


def test_mutable_health_model_defaults_are_isolated_between_instances() -> None:
    first = ServiceHealthModel(name="first", type="api", status="healthy")
    second = ServiceHealthModel(name="second", type="api", status="healthy")

    first.details["state"] = "changed"
    first.dependencies.append("camera")

    assert second.details == {}
    assert second.dependencies == []
