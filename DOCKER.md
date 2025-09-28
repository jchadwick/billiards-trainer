# Docker Setup for Billiards Trainer

This document provides comprehensive instructions for building and running the Billiards Trainer backend API using Docker and Docker Compose.

## 🐳 Docker Configuration Overview

The project includes:
- **Multi-stage Dockerfile** for optimized production builds
- **Docker Compose** for orchestrating the complete stack
- **Development and production modes** with appropriate configurations
- **Health checks and monitoring** for production readiness

## 📁 File Structure

```
billiards-trainer/
├── backend/
│   ├── Dockerfile              # Multi-stage Docker configuration
│   ├── .dockerignore           # Build context optimization
│   └── requirements.txt        # Python dependencies
├── docker-compose.yml          # Complete stack orchestration
├── .env.docker                 # Docker environment template
├── monitoring/
│   └── prometheus.yml          # Monitoring configuration
└── DOCKER.md                   # This documentation
```

## 🚀 Quick Start

### 1. Set Up Environment

Copy the Docker environment template:
```bash
cp .env.docker .env
```

Edit the `.env` file and set secure values for production:
```bash
# IMPORTANT: Change these values for production!
JWT_SECRET_KEY=your-secure-secret-key-here
DEFAULT_API_KEY=your-secure-api-key-here
```

### 2. Build and Run (Development)

For development with hot reload:
```bash
docker-compose --profile development up backend-dev redis
```

### 3. Build and Run (Production)

For production deployment:
```bash
docker-compose up -d backend redis
```

### 4. Full Stack with Monitoring

To run with Prometheus and Grafana monitoring:
```bash
docker-compose --profile monitoring up -d
```

## 🔧 Configuration Options

### Environment Variables

Key environment variables that can be customized:

#### Application Settings
- `ENVIRONMENT`: `development` or `production`
- `DEBUG`: `true` or `false`
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`

#### API Configuration
- `API_HOST`: Host to bind to (default: `0.0.0.0`)
- `API_PORT`: API port (default: `8000`)
- `WS_PORT`: WebSocket port (default: `8001`)

#### Security
- `JWT_SECRET_KEY`: Secret key for JWT tokens (**required in production**)
- `API_KEY_HEADER`: Header name for API key (default: `X-API-Key`)
- `DEFAULT_API_KEY`: Default API key for development

#### Computer Vision
- `CAMERA_INDEX`: Camera device index (default: `0`)
- `CAMERA_WIDTH`: Camera resolution width (default: `1920`)
- `CAMERA_HEIGHT`: Camera resolution height (default: `1080`)
- `CAMERA_FPS`: Camera frame rate (default: `30`)

#### Performance
- `WORKERS`: Number of worker processes (default: `1`)
- `MAX_CONCURRENT_CONNECTIONS`: Max simultaneous connections (default: `100`)

### Docker Compose Profiles

The setup includes several profiles for different use cases:

- **Default**: Backend API + Redis
- **development**: Development mode with hot reload
- **monitoring**: Adds Prometheus and Grafana

## 📋 Available Commands

### Basic Operations

```bash
# Build the backend image
docker-compose build backend

# Start production stack
docker-compose up -d

# Start development stack
docker-compose --profile development up

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check container status
docker-compose ps

# Monitor container logs
docker-compose logs -f backend
```

### Scaling

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# Scale with custom worker count
WORKERS=4 docker-compose up -d backend
```

## 🔍 Development Workflow

### Hot Reload Development

```bash
# Start development environment
docker-compose --profile development up backend-dev redis

# The backend code is mounted as a volume for hot reload
# Changes to Python files will automatically restart the server
```

### Debugging

```bash
# Access development container shell
docker-compose exec backend-dev bash

# View detailed logs
docker-compose logs -f backend-dev

# Check Python environment
docker-compose exec backend-dev python -c "import sys; print(sys.path)"
```

### Testing

```bash
# Run tests inside container
docker-compose exec backend python -m pytest

# Run with coverage
docker-compose exec backend python -m pytest --cov=.
```

## 📊 Monitoring

### Prometheus Metrics

When using the monitoring profile:
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Metrics**: http://localhost:9090/metrics

### Health Checks

All services include comprehensive health checks:
- **Backend API**: `/api/v1/health` endpoint
- **Redis**: Redis ping command
- **Automatic restart** if health checks fail

## 🔒 Security Considerations

### Production Deployment

**Required security steps for production:**

1. **Set secure secrets:**
   ```bash
   # Generate secure JWT secret
   JWT_SECRET_KEY=$(openssl rand -base64 32)

   # Generate secure API key
   DEFAULT_API_KEY=$(openssl rand -hex 16)
   ```

2. **Limit network access:**
   - Remove external port mappings for Redis
   - Use proper firewall rules
   - Configure CORS appropriately

3. **Use HTTPS:**
   - Add reverse proxy (nginx/traefik)
   - Configure SSL certificates
   - Redirect HTTP to HTTPS

### Resource Limits

Production containers include resource limits:
- **Backend**: 4GB RAM, 2 CPU cores
- **Redis**: 512MB RAM, 0.5 CPU cores

Adjust in `docker-compose.yml` based on your hardware.

## 🛠️ Troubleshooting

### Common Issues

**1. Build failures:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache backend
```

**2. Permission issues:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data ./logs ./config
```

**3. Port conflicts:**
```bash
# Check what's using the port
lsof -i :8000

# Use different ports
API_PORT=8080 WS_PORT=8081 docker-compose up
```

**4. Memory issues:**
```bash
# Check container memory usage
docker stats

# Reduce worker count
WORKERS=1 docker-compose up backend
```

### Log Analysis

```bash
# View startup logs
docker-compose logs backend | grep "Starting"

# Check for errors
docker-compose logs backend | grep -i error

# Monitor real-time logs
docker-compose logs -f --tail=100 backend
```

## 🚀 Production Deployment

### Recommended Production Setup

1. **Use Docker Swarm or Kubernetes** for orchestration
2. **Set up proper secrets management**
3. **Configure external load balancer**
4. **Set up log aggregation** (ELK Stack, etc.)
5. **Monitor with external tools** (New Relic, Datadog, etc.)

### Example Production Command

```bash
# Set production environment
export ENVIRONMENT=production
export DEBUG=false
export JWT_SECRET_KEY="your-production-secret"

# Deploy with production settings
docker-compose -f docker-compose.yml up -d
```

## 📚 Additional Resources

- **Backend API Documentation**: http://localhost:8000/docs
- **OpenAPI Specification**: http://localhost:8000/openapi.json
- **Health Check Endpoint**: http://localhost:8000/api/v1/health
- **Metrics Endpoint**: http://localhost:9090/metrics

For more information about the application architecture and development, see the main README.md file.
