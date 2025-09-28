# Billiards Trainer System Integration Documentation

## Overview

The Billiards Trainer system is a comprehensive computer vision and AI-powered training platform for billiards players. This document describes the complete system integration, including all modules, monitoring, deployment, and operational procedures.

## System Architecture

### Core Components

1. **System Orchestrator** (`backend/system/orchestrator.py`)
   - Main system coordinator and lifecycle manager
   - Handles module startup/shutdown sequences
   - Manages inter-module dependencies
   - Provides centralized configuration and monitoring

2. **Health Monitoring** (`backend/system/health.py`)
   - Real-time health checks for all modules
   - Module-specific health validation
   - System-wide health aggregation
   - Performance scoring and reporting

3. **Performance Monitoring** (`backend/system/monitoring.py`)
   - System resource monitoring (CPU, memory, disk)
   - Application-specific metrics collection
   - Real-time performance dashboard
   - Alert generation and management

4. **Auto-Recovery** (`backend/system/recovery.py`)
   - Automated fault detection and recovery
   - Configurable recovery policies
   - Module restart and state reset capabilities
   - System-wide recovery coordination

### Module Integration

#### Core Module
- Game state management and physics calculations
- Shot analysis and prediction algorithms
- Event coordination and caching
- Performance metrics and validation

#### Vision Module
- Real-time computer vision processing
- Ball and table detection
- Camera calibration and frame processing
- Object tracking and state updates

#### Projector Module
- Visual guidance projection
- Trajectory visualization
- Real-time rendering and display
- Network-based projector control

#### API Module
- RESTful API endpoints
- WebSocket real-time communication
- Middleware integration
- Request/response processing

#### Configuration Module
- Centralized configuration management
- Environment-specific settings
- Dynamic configuration updates
- Validation and type safety

## System Launcher

### Main Entry Point

The system launcher (`backend/system_launcher.py`) provides the primary entry point for starting the complete system.

```bash
# Start full system in development mode
python system_launcher.py --environment development

# Start production system
python system_launcher.py --environment production --host 0.0.0.0 --port 8000

# Start with specific modules disabled
python system_launcher.py --no-projector --no-vision

# Start with debug logging
python system_launcher.py --debug --log-level DEBUG
```

### Configuration Options

- `--environment`: Environment mode (development/production/testing)
- `--debug`: Enable debug mode
- `--no-vision`: Disable vision module
- `--no-projector`: Disable projector module
- `--no-api`: Disable API module
- `--no-core`: Disable core module
- `--no-monitoring`: Disable performance monitoring
- `--no-recovery`: Disable auto-recovery
- `--health-interval`: Health check interval in seconds
- `--host`: API server host
- `--port`: API server port
- `--workers`: Number of API server workers
- `--log-level`: Logging level
- `--log-file`: Log file path
- `--shutdown-timeout`: Graceful shutdown timeout

## Monitoring and Alerting

### Health Monitoring

The health monitoring system continuously checks the status of all modules:

#### System Health Levels
- **HEALTHY**: All systems operating normally
- **DEGRADED**: Some issues detected but system functional
- **UNHEALTHY**: Critical issues requiring attention
- **UNKNOWN**: Health status cannot be determined

#### Module-Specific Checks
- **Configuration Module**: Configuration accessibility and validity
- **Core Module**: State access, error rates, processing performance
- **Vision Module**: Camera connection, frame rates, detection accuracy
- **Projector Module**: Initialization status, network connectivity
- **API Module**: Application structure and endpoint availability

### Performance Metrics

#### System Metrics
- CPU usage and load average
- Memory usage and availability
- Disk usage and I/O
- Network traffic
- Process and thread counts

#### Application Metrics
- Vision processing FPS and latency
- Core module update times
- Physics calculation performance
- API request rates and response times
- Projector rendering latency

### Alerting System

The alert manager generates notifications based on configurable thresholds:

#### Alert Types
- **WARNING**: Performance degradation or minor issues
- **ERROR**: Significant problems affecting functionality
- **CRITICAL**: System-threatening conditions requiring immediate attention

#### Default Thresholds
- CPU usage: Warning at 70%, Critical at 90%
- Memory usage: Warning at 80%, Critical at 95%
- Disk usage: Warning at 85%, Critical at 95%
- Vision FPS: Warning below 15, Critical below 5
- API error rate: Warning above 5%, Critical above 10%

## Auto-Recovery System

### Recovery Policies

The auto-recovery system implements configurable recovery policies:

#### Recovery Actions
- **RESTART_MODULE**: Restart a specific module
- **RESTART_SYSTEM**: Restart the entire system
- **RESET_STATE**: Reset module state and clear caches
- **CLEAR_CACHE**: Clear module caches to free memory
- **FAILOVER**: Switch to backup systems
- **ALERT_ONLY**: Generate alert without automatic action

#### Default Policies
1. **Vision Module Recovery**: Restart module on unhealthy status (3 attempts)
2. **Core Module Recovery**: Reset state on unhealthy status (2 attempts)
3. **High Memory Recovery**: Clear caches when memory exceeds 90%
4. **System Critical Recovery**: Restart system when multiple modules fail

### Recovery Coordination

Recovery attempts are coordinated to prevent conflicts:
- Maximum attempt limits with exponential backoff
- Module dependency awareness
- System-wide impact assessment
- Recovery success validation

## Production Deployment

### Deployment Script

The deployment script (`deployment/deploy.py`) automates production deployment:

```bash
# Deploy to production
python deployment/deploy.py /opt/billiards-trainer --environment production

# Deploy without backup
python deployment/deploy.py /opt/billiards-trainer --skip-backup
```

#### Deployment Process
1. Pre-deployment validation
2. Backup creation
3. Service shutdown
4. Code deployment
5. Dependency installation
6. Configuration update
7. Service setup
8. Verification and rollback if needed

### System Service

The systemd service provides production process management:

```bash
# Service management
sudo systemctl start billiards-trainer
sudo systemctl stop billiards-trainer
sudo systemctl restart billiards-trainer
sudo systemctl status billiards-trainer

# Enable automatic startup
sudo systemctl enable billiards-trainer
```

### Nginx Configuration

Nginx provides reverse proxy and load balancing:
- Rate limiting for API endpoints
- WebSocket proxy support
- Static file serving
- Security headers and CORS
- SSL/TLS termination (when configured)

## Backup and Recovery

### Backup Types

#### Full System Backup
```bash
# Create full backup
python deployment/backup.py backup --system-dir /opt/billiards-trainer

# Create named backup
python deployment/backup.py backup --name "pre-upgrade-backup"
```

#### Data-Only Backup
```bash
# Backup configuration and data only
python deployment/backup.py backup --data-only
```

### Restore Operations

```bash
# List available backups
python deployment/backup.py list

# Restore from backup
python deployment/backup.py restore --backup-file /path/to/backup.tar.gz

# Restore data only
python deployment/backup.py restore --backup-file /path/to/backup.tar.gz --data-only
```

### Backup Management

```bash
# Clean up old backups (keep 10 most recent)
python deployment/backup.py cleanup --keep-count 10
```

#### Backup Contents
- **Full Backup**: Code, configuration, data, logs, system info
- **Data Backup**: Configuration and application data only
- **Incremental**: (Future enhancement) Changed files only

## Log Management

### Log Rotation

Automated log rotation is configured via logrotate:
- Daily rotation for application logs
- 30-day retention for system logs
- 14-day retention for nginx logs
- Compression and archival

### Log Locations

- **System Logs**: `/opt/billiards-trainer/logs/`
- **Nginx Logs**: `/var/log/nginx/billiards-trainer*.log`
- **Systemd Logs**: `journalctl -u billiards-trainer`

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical error conditions

## API Documentation

### REST Endpoints

#### Health and Status
- `GET /health` - System health check
- `GET /api/v1/system/status` - Detailed system status
- `GET /api/v1/system/metrics` - Performance metrics

#### Configuration
- `GET /api/v1/config` - Get configuration
- `PUT /api/v1/config` - Update configuration
- `POST /api/v1/config/reload` - Reload configuration

#### Game Operations
- `GET /api/v1/game/state` - Current game state
- `POST /api/v1/game/reset` - Reset game
- `POST /api/v1/game/analyze` - Analyze current shot

#### Calibration
- `POST /api/v1/calibration/camera` - Camera calibration
- `POST /api/v1/calibration/colors` - Color calibration
- `GET /api/v1/calibration/status` - Calibration status

### WebSocket Endpoints

#### Real-time Communication
- `/ws` - Main WebSocket endpoint
- Real-time game state updates
- Performance metrics streaming
- Alert notifications

## Troubleshooting

### Common Issues

#### System Won't Start
1. Check system logs: `journalctl -u billiards-trainer`
2. Verify configuration files
3. Check port availability
4. Validate dependencies

#### Vision Module Issues
1. Verify camera connection
2. Check camera permissions
3. Validate vision configuration
4. Test camera capture manually

#### Performance Problems
1. Monitor system resources
2. Check for memory leaks
3. Analyze processing bottlenecks
4. Review alert history

#### API Connectivity Issues
1. Check nginx configuration
2. Verify firewall settings
3. Test direct API access
4. Review rate limiting

### Diagnostic Commands

```bash
# System status
systemctl status billiards-trainer
journalctl -u billiards-trainer -f

# Resource monitoring
htop
iotop
df -h

# Network connectivity
netstat -tlnp | grep 8000
curl http://localhost/health

# Log analysis
tail -f /opt/billiards-trainer/logs/billiards-trainer.log
grep ERROR /opt/billiards-trainer/logs/*.log
```

## Security Considerations

### Production Security

1. **Service Isolation**: Run services with minimal privileges
2. **Network Security**: Configure firewall rules
3. **Access Control**: Implement authentication and authorization
4. **Data Protection**: Encrypt sensitive data
5. **Audit Logging**: Log security-relevant events

### Recommended Practices

1. Regular security updates
2. Secure configuration management
3. Network segmentation
4. Backup encryption
5. Access monitoring

## Performance Optimization

### System Tuning

1. **Resource Allocation**: Optimize CPU and memory usage
2. **I/O Performance**: Use SSD storage for logs and data
3. **Network Optimization**: Tune network buffers and timeouts
4. **Process Management**: Configure worker processes appropriately

### Application Optimization

1. **Caching Strategy**: Implement intelligent caching
2. **Database Optimization**: Optimize queries and indexing
3. **Image Processing**: Use GPU acceleration when available
4. **Memory Management**: Monitor and prevent memory leaks

## Maintenance Procedures

### Regular Maintenance

1. **System Updates**: Apply security patches monthly
2. **Log Cleanup**: Manage log file growth
3. **Backup Verification**: Test backup restore procedures
4. **Performance Review**: Analyze metrics and trends
5. **Configuration Review**: Validate settings and policies

### Scheduled Tasks

1. **Daily**: Automated backups and log rotation
2. **Weekly**: System health reports
3. **Monthly**: Security updates and maintenance
4. **Quarterly**: Full system review and optimization

## Support and Contact

For technical support and additional documentation:

- **System Logs**: Check application and system logs first
- **Health Dashboard**: Monitor real-time system status
- **Alert History**: Review past issues and resolutions
- **Documentation**: Refer to module-specific documentation

This integration documentation provides a comprehensive guide to operating and maintaining the Billiards Trainer system in production environments.
