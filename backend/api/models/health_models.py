"""Health and monitoring related API models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, validator


class HealthStatusEnum(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceTypeEnum(str, Enum):
    """Service types."""

    CORE = "core"
    CONFIGURATION = "configuration"
    VISION = "vision"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL = "external"


class MetricTypeEnum(str, Enum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevelEnum(str, Enum):
    """Alert levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Basic health models


class ServiceHealthModel(BaseModel):
    """Individual service health status."""

    name: str = Field(..., description="Service name")
    type: ServiceTypeEnum = Field(..., description="Service type")
    status: HealthStatusEnum = Field(..., description="Health status")
    last_check: datetime = Field(
        default_factory=datetime.now, description="Last health check time"
    )
    response_time: Optional[float] = Field(
        None, ge=0, description="Response time in seconds"
    )
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    details: dict[str, Any] = Field(default={}, description="Additional health details")
    dependencies: list[str] = Field(default=[], description="Service dependencies")
    version: Optional[str] = Field(None, description="Service version")
    uptime: Optional[float] = Field(None, ge=0, description="Service uptime in seconds")


class HealthCheckResult(BaseModel):
    """Health check result."""

    check_name: str = Field(..., description="Health check name")
    status: HealthStatusEnum = Field(..., description="Check status")
    message: Optional[str] = Field(None, description="Check message")
    duration: float = Field(..., ge=0, description="Check duration in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Check timestamp"
    )
    metadata: dict[str, Any] = Field(
        default={}, description="Additional check metadata"
    )


class SystemResourcesModel(BaseModel):
    """System resource metrics."""

    cpu_usage_percent: float = Field(
        ..., ge=0, le=100, description="CPU usage percentage"
    )
    memory_usage_bytes: int = Field(..., ge=0, description="Memory usage in bytes")
    memory_total_bytes: int = Field(..., ge=0, description="Total memory in bytes")
    memory_usage_percent: float = Field(
        ..., ge=0, le=100, description="Memory usage percentage"
    )
    disk_usage_bytes: Optional[int] = Field(
        None, ge=0, description="Disk usage in bytes"
    )
    disk_total_bytes: Optional[int] = Field(
        None, ge=0, description="Total disk space in bytes"
    )
    disk_usage_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Disk usage percentage"
    )
    network_rx_bytes: Optional[int] = Field(
        None, ge=0, description="Network bytes received"
    )
    network_tx_bytes: Optional[int] = Field(
        None, ge=0, description="Network bytes transmitted"
    )
    open_file_descriptors: Optional[int] = Field(
        None, ge=0, description="Open file descriptors"
    )
    load_average: Optional[list[float]] = Field(None, description="System load average")


class ProcessMetricsModel(BaseModel):
    """Process-specific metrics."""

    pid: int = Field(..., description="Process ID")
    threads_count: int = Field(..., ge=0, description="Number of threads")
    memory_rss: int = Field(..., ge=0, description="Resident set size in bytes")
    memory_vms: int = Field(..., ge=0, description="Virtual memory size in bytes")
    cpu_percent: float = Field(..., ge=0, description="CPU usage percentage")
    memory_percent: float = Field(..., ge=0, description="Memory usage percentage")
    create_time: float = Field(..., description="Process creation time")
    num_fds: Optional[int] = Field(None, ge=0, description="Number of file descriptors")
    num_connections: Optional[int] = Field(
        None, ge=0, description="Number of connections"
    )


class MetricDataPoint(BaseModel):
    """Individual metric data point."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Measurement timestamp"
    )
    value: Union[int, float, str] = Field(..., description="Metric value")
    tags: dict[str, str] = Field(default={}, description="Metric tags")


class MetricModel(BaseModel):
    """Generic metric model."""

    name: str = Field(..., description="Metric name")
    type: MetricTypeEnum = Field(..., description="Metric type")
    description: Optional[str] = Field(None, description="Metric description")
    unit: Optional[str] = Field(None, description="Measurement unit")
    current_value: Union[int, float, str] = Field(
        ..., description="Current metric value"
    )
    data_points: list[MetricDataPoint] = Field(
        default=[], description="Historical data points"
    )
    labels: dict[str, str] = Field(default={}, description="Metric labels")


class PerformanceMetricsModel(BaseModel):
    """Performance metrics collection."""

    requests_total: int = Field(default=0, ge=0, description="Total requests processed")
    requests_per_second: float = Field(
        default=0.0, ge=0, description="Current requests per second"
    )
    response_time_avg: float = Field(
        default=0.0, ge=0, description="Average response time in seconds"
    )
    response_time_p50: float = Field(
        default=0.0, ge=0, description="50th percentile response time"
    )
    response_time_p95: float = Field(
        default=0.0, ge=0, description="95th percentile response time"
    )
    response_time_p99: float = Field(
        default=0.0, ge=0, description="99th percentile response time"
    )
    error_rate: float = Field(default=0.0, ge=0, le=1, description="Error rate (0-1)")
    cache_hit_rate: float = Field(
        default=0.0, ge=0, le=1, description="Cache hit rate (0-1)"
    )
    active_connections: int = Field(default=0, ge=0, description="Active connections")
    queue_size: int = Field(default=0, ge=0, description="Current queue size")


class IntegrationMetricsModel(BaseModel):
    """Integration layer metrics."""

    modules_initialized: int = Field(
        default=0, ge=0, description="Number of initialized modules"
    )
    services_healthy: int = Field(
        default=0, ge=0, description="Number of healthy services"
    )
    total_services: int = Field(default=0, ge=0, description="Total number of services")
    events_processed: int = Field(default=0, ge=0, description="Total events processed")
    cache_entries: int = Field(default=0, ge=0, description="Number of cache entries")
    background_tasks_active: int = Field(
        default=0, ge=0, description="Active background tasks"
    )
    integration_uptime: float = Field(
        default=0.0, ge=0, description="Integration uptime in seconds"
    )
    last_health_check: Optional[datetime] = Field(
        None, description="Last health check time"
    )


class AlertModel(BaseModel):
    """System alert model."""

    id: str = Field(..., description="Alert identifier")
    level: AlertLevelEnum = Field(..., description="Alert level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Alert source")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Alert timestamp"
    )
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution time")
    tags: dict[str, str] = Field(default={}, description="Alert tags")
    metadata: dict[str, Any] = Field(
        default={}, description="Additional alert metadata"
    )


# Comprehensive health status model


class HealthStatusModel(BaseModel):
    """Overall system health status."""

    status: HealthStatusEnum = Field(..., description="Overall health status")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Health check timestamp"
    )
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    services: dict[str, ServiceHealthModel] = Field(
        ..., description="Individual service health"
    )
    health_checks: list[HealthCheckResult] = Field(
        default=[], description="Detailed health checks"
    )
    system_resources: Optional[SystemResourcesModel] = Field(
        None, description="System resource usage"
    )
    process_metrics: Optional[ProcessMetricsModel] = Field(
        None, description="Process metrics"
    )
    performance_metrics: Optional[PerformanceMetricsModel] = Field(
        None, description="Performance metrics"
    )
    integration_metrics: Optional[IntegrationMetricsModel] = Field(
        None, description="Integration metrics"
    )
    alerts: list[AlertModel] = Field(default=[], description="Active alerts")
    environment: str = Field(
        default="development", description="Deployment environment"
    )
    build_info: dict[str, str] = Field(default={}, description="Build information")


class MetricsCollectionModel(BaseModel):
    """Collection of system metrics."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Collection timestamp"
    )
    metrics: list[MetricModel] = Field(..., description="Collected metrics")
    collection_duration: float = Field(
        ..., ge=0, description="Collection duration in seconds"
    )
    total_metrics: int = Field(..., ge=0, description="Total number of metrics")
    metadata: dict[str, Any] = Field(default={}, description="Collection metadata")


# Request models


class HealthCheckRequest(BaseModel):
    """Health check request."""

    services: Optional[list[str]] = Field(
        None, description="Specific services to check"
    )
    include_details: bool = Field(
        default=True, description="Include detailed health information"
    )
    include_metrics: bool = Field(
        default=True, description="Include performance metrics"
    )
    include_system_resources: bool = Field(
        default=True, description="Include system resource metrics"
    )
    timeout: Optional[float] = Field(
        None, gt=0, description="Health check timeout in seconds"
    )


class MetricsRequest(BaseModel):
    """Metrics collection request."""

    metric_names: Optional[list[str]] = Field(
        None, description="Specific metrics to collect"
    )
    time_range: Optional[int] = Field(
        None, gt=0, description="Time range in seconds for historical data"
    )
    include_tags: bool = Field(default=True, description="Include metric tags")
    format: str = Field(default="json", description="Output format")


class AlertRequest(BaseModel):
    """Alert management request."""

    alert_id: Optional[str] = Field(None, description="Specific alert ID")
    level: Optional[AlertLevelEnum] = Field(None, description="Filter by alert level")
    resolved: Optional[bool] = Field(None, description="Filter by resolution status")
    since: Optional[datetime] = Field(None, description="Show alerts since timestamp")
    limit: int = Field(
        default=100, gt=0, le=1000, description="Maximum number of alerts"
    )


# Response models


class HealthStatusResponse(BaseModel):
    """Health status response."""

    success: bool = Field(default=True, description="Operation success")
    health: HealthStatusModel = Field(..., description="Health status information")
    checks_performed: int = Field(
        ..., ge=0, description="Number of health checks performed"
    )
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class MetricsResponse(BaseModel):
    """Metrics collection response."""

    success: bool = Field(default=True, description="Operation success")
    metrics: MetricsCollectionModel = Field(..., description="Collected metrics")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class AlertsResponse(BaseModel):
    """Alerts response."""

    success: bool = Field(default=True, description="Operation success")
    alerts: list[AlertModel] = Field(..., description="System alerts")
    total_alerts: int = Field(..., ge=0, description="Total number of alerts")
    active_alerts: int = Field(..., ge=0, description="Number of active alerts")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class ServiceHealthResponse(BaseModel):
    """Individual service health response."""

    success: bool = Field(default=True, description="Operation success")
    service: ServiceHealthModel = Field(..., description="Service health information")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class SystemStatusResponse(BaseModel):
    """System status summary response."""

    success: bool = Field(default=True, description="Operation success")
    status: HealthStatusEnum = Field(..., description="Overall system status")
    healthy_services: int = Field(..., ge=0, description="Number of healthy services")
    total_services: int = Field(..., ge=0, description="Total number of services")
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    last_check: datetime = Field(..., description="Last health check time")
    critical_alerts: int = Field(
        default=0, ge=0, description="Number of critical alerts"
    )
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


# Utility models


class ComponentStatus(BaseModel):
    """Generic component status."""

    name: str = Field(..., description="Component name")
    status: HealthStatusEnum = Field(..., description="Component status")
    message: Optional[str] = Field(None, description="Status message")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last status update"
    )


class DiagnosticsInfo(BaseModel):
    """System diagnostics information."""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="Diagnostics timestamp"
    )
    components: list[ComponentStatus] = Field(..., description="Component statuses")
    configuration_valid: bool = Field(
        ..., description="Configuration validation status"
    )
    dependencies_available: bool = Field(..., description="Dependencies availability")
    storage_accessible: bool = Field(..., description="Storage accessibility")
    network_connectivity: bool = Field(..., description="Network connectivity")
    performance_acceptable: bool = Field(..., description="Performance acceptability")
    errors: list[str] = Field(default=[], description="Diagnostic errors")
    warnings: list[str] = Field(default=[], description="Diagnostic warnings")


# Validators


@validator("cpu_usage_percent", "memory_usage_percent", "disk_usage_percent")
def validate_percentage(cls, v):
    """Validate percentage values."""
    if v is not None and not (0 <= v <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    return v


@validator("error_rate", "cache_hit_rate")
def validate_rate(cls, v):
    """Validate rate values (0-1)."""
    if not (0 <= v <= 1):
        raise ValueError("Rate must be between 0 and 1")
    return v


@validator("response_time")
def validate_response_time(cls, v):
    """Validate response time values."""
    if v is not None and v < 0:
        raise ValueError("Response time cannot be negative")
    return v


__all__ = [
    # Enums
    "HealthStatusEnum",
    "ServiceTypeEnum",
    "MetricTypeEnum",
    "AlertLevelEnum",
    # Health models
    "ServiceHealthModel",
    "HealthCheckResult",
    "HealthStatusModel",
    # Metrics models
    "SystemResourcesModel",
    "ProcessMetricsModel",
    "MetricDataPoint",
    "MetricModel",
    "PerformanceMetricsModel",
    "IntegrationMetricsModel",
    "MetricsCollectionModel",
    # Alert models
    "AlertModel",
    # Utility models
    "ComponentStatus",
    "DiagnosticsInfo",
    # Request models
    "HealthCheckRequest",
    "MetricsRequest",
    "AlertRequest",
    # Response models
    "HealthStatusResponse",
    "MetricsResponse",
    "AlertsResponse",
    "ServiceHealthResponse",
    "SystemStatusResponse",
]
