# Health Monitoring Integration

This document describes the comprehensive health monitoring system implemented for the Billiards Trainer application. The system provides real-time monitoring of all application components, API performance metrics, and system resources.

## Overview

The health monitoring system has been completely redesigned to replace placeholder values with real monitoring data from all system components:

- **Core Module**: Physics engine performance, state validation, error tracking
- **Vision Module**: Camera health, processing performance, detection accuracy
- **Configuration System**: Config validation, file access, hot reload status
- **API Server**: Request metrics, response times, error rates
- **WebSocket Manager**: Connection tracking, session health, stream monitoring
- **System Resources**: CPU, memory, disk usage, network I/O

## Architecture

### Core Components

1. **HealthMonitor** (`backend/system/health_monitor.py`)
   - Central monitoring system that tracks all components
   - Aggregates health data from multiple sources
   - Provides real-time metrics and alerts

2. **MetricsMiddleware** (`backend/api/middleware/metrics.py`)
   - Tracks API request performance automatically
   - Records response times, success rates, error counts
   - Integrates with health monitor for real-time metrics

3. **Enhanced Health Routes** (`backend/api/routes/health.py`)
   - Updated to use real monitoring data instead of placeholders
   - Provides detailed component health information
   - Supports Kubernetes-style health checks

## Features

### Real-Time Component Monitoring

**Core Module Health:**
- Performance metrics (update times, physics calculations)
- State validation and consistency checks
- Error tracking and cache statistics
- Physics validator integration

**Vision Module Health:**
- Camera connection status and health
- Frame processing performance and FPS
- Detection accuracy for tables, balls, and cues
- Processing queue and dropped frame tracking

**Configuration System Health:**
- Configuration validation and accessibility
- Hot reload functionality status
- Configuration file monitoring
- Backup and persistence system health

**WebSocket Health:**
- Active connection tracking
- Session state monitoring
- Stream subscription management
- Connection error tracking

**System Resource Monitoring:**
- Real-time CPU, memory, and disk usage
- Network I/O statistics
- Resource threshold alerting
- Performance degradation detection

### API Performance Metrics

- **Request Tracking**: Automatic tracking of all API requests
- **Response Time Monitoring**: Per-request and average response times
- **Error Rate Tracking**: Success/failure rates and error categorization
- **Throughput Metrics**: Requests per second calculations
- **Slow Request Detection**: Automatic logging of slow requests

### Health Aggregation

- **Multi-Source Data**: Combines data from all system components
- **Status Calculation**: Intelligent overall health status determination
- **Alert System**: Configurable alerts for component failures
- **Historical Tracking**: Health history and trend analysis

## API Endpoints

### Basic Health Check
```http
GET /api/v1/health/
```

Returns basic health status with optional detailed component information and metrics.

**Query Parameters:**
- `include_details=true`: Include detailed component health information
- `include_metrics=true`: Include system performance metrics

**Example Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": 3600.5,
  "version": "1.0.0",
  "components": {
    "core": {
      "name": "core",
      "status": "healthy",
      "message": "Operating normally",
      "uptime": 3600.5,
      "error_count": 0
    },
    "vision": {
      "name": "vision",
      "status": "healthy",
      "message": "Operating normally",
      "uptime": 3600.5,
      "error_count": 0
    }
  },
  "metrics": {
    "cpu_usage": 25.4,
    "memory_usage": 45.2,
    "disk_usage": 30.1,
    "api_requests_per_second": 12.5,
    "websocket_connections": 3,
    "average_response_time": 45.2
  }
}
```

### Performance Metrics
```http
GET /api/v1/health/metrics
```

Returns detailed system performance metrics.

**Query Parameters:**
- `time_range`: Time range for metrics (5m, 15m, 1h, 6h, 24h)

### Kubernetes-Style Health Checks

**Liveness Check:**
```http
GET /api/v1/health/live
```

**Readiness Check:**
```http
GET /api/v1/health/ready
```

## Integration

### Application Lifecycle

The health monitoring system is integrated into the main application lifecycle:

1. **Startup**: Components are registered with the health monitor
2. **Runtime**: Continuous monitoring with configurable intervals
3. **Shutdown**: Graceful monitoring shutdown with final health report

### Component Registration

Components are automatically registered during application startup:

```python
# Register components with health monitor
health_monitor.register_components(
    core_module=app_state.core_module,
    config_module=app_state.config_module,
    websocket_manager=app_state.websocket_manager
)

# Start monitoring
await health_monitor.start_monitoring(check_interval=5.0)
```

### Middleware Integration

API performance tracking is automatically enabled through middleware:

```python
# Automatic API metrics tracking
app.add_middleware(MetricsMiddleware)
```

## Configuration

### Health Monitor Settings

The health monitor can be configured with various parameters:

- **Check Interval**: How often to perform health checks (default: 5 seconds)
- **History Limit**: Number of health snapshots to retain (default: 100)
- **Alert Thresholds**: CPU, memory, and performance thresholds for alerts
- **Component Timeouts**: Per-component health check timeouts

### Excluded Paths

Certain paths are excluded from API metrics to avoid noise:

- `/health/live`
- `/health/ready`
- `/docs`
- `/redoc`
- `/openapi.json`

## Testing

### Comprehensive Test Suite

A comprehensive test suite is provided (`backend/test_health_monitoring.py`) that validates:

- Basic health endpoint functionality
- Detailed component health reporting
- Real-time metrics tracking
- API performance measurement
- WebSocket connection monitoring
- Error handling and edge cases
- Kubernetes-style health checks

### Running Tests

```bash
# Make the test script executable
chmod +x backend/test_health_monitoring.py

# Run comprehensive health monitoring tests
python backend/test_health_monitoring.py

# Results are saved to /tmp/health_monitoring_test_results.json
```

### Test Results

The test suite provides detailed results including:

- Overall success rate
- Individual test results
- Performance metrics
- Component health status
- Error analysis

## Monitoring and Alerts

### Health Status Levels

- **Healthy**: All components operating normally
- **Degraded**: Some performance issues detected but system functional
- **Unhealthy**: Critical issues affecting system operation

### Alert Conditions

Automatic alerts are triggered for:

- Component failures or errors
- High resource usage (CPU >80%, Memory >85%, Disk >90%)
- Slow API responses (>500ms average)
- High error rates (>5%)
- WebSocket connection issues
- Configuration validation failures

### Health History

The system maintains a rolling history of health snapshots for:

- Trend analysis
- Performance debugging
- Capacity planning
- Issue investigation

## Troubleshooting

### Common Issues

1. **Missing psutil**: Install with `pip install psutil` for system metrics
2. **Import Errors**: Health monitor includes fallback interfaces for missing components
3. **Permission Issues**: Ensure proper file permissions for configuration monitoring
4. **Resource Limits**: Monitor system resources to prevent performance degradation

### Debug Mode

Enable debug logging for detailed health monitoring information:

```python
logging.getLogger('backend.system.health_monitor').setLevel(logging.DEBUG)
```

## Implementation Details

### Removed Placeholders

The following TODO comments and placeholder implementations have been replaced with real monitoring:

1. **Line 64-68**: API requests per second, WebSocket connections, and response times now use real data
2. **Line 307**: Active operations count now uses shutdown coordinator data
3. **Component Health**: All component health checks now use real module data instead of simple availability checks

### Real Monitoring Sources

- **Core Module**: `get_performance_metrics()`, `validate_state()`, `get_current_state()`
- **Vision Module**: `get_statistics()`, camera health, processing performance
- **Config Module**: `validate()`, file accessibility, hot reload status
- **WebSocket Manager**: `get_all_sessions()`, session state tracking
- **API Metrics**: Middleware-tracked request performance
- **System Resources**: psutil-based real-time system monitoring

## Performance Impact

The health monitoring system is designed to have minimal performance impact:

- **Monitoring Interval**: Configurable (default 5 seconds)
- **Async Operations**: Non-blocking health checks
- **Efficient Caching**: Cached metrics with configurable refresh rates
- **Lightweight Middleware**: Minimal overhead for request tracking
- **Conditional Monitoring**: Components only monitored when available

## Future Enhancements

Potential future improvements:

1. **Distributed Monitoring**: Support for multi-instance deployments
2. **Custom Metrics**: User-defined health metrics and thresholds
3. **External Integrations**: Prometheus, Grafana, or other monitoring systems
4. **Predictive Analytics**: Trend analysis and predictive failure detection
5. **Health Dashboard**: Web-based real-time health monitoring interface

## Summary

The health monitoring integration provides a comprehensive, real-time view of system health with:

✅ **Real monitoring data** replacing all placeholder values
✅ **Component-specific health checks** for all major modules
✅ **API performance tracking** with detailed metrics
✅ **WebSocket connection monitoring** and session health
✅ **System resource monitoring** with alerting
✅ **Comprehensive test suite** for validation
✅ **Kubernetes-compatible** health endpoints
✅ **Production-ready** error handling and fallbacks

The system now provides actionable insights into application health and performance, enabling proactive monitoring and troubleshooting.
