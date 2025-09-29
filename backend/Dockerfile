# Production-ready Dockerfile for Billiards Trainer Backend API
# ============================================================

# ========================
# Build Stage
# ========================
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt

# Install core dependencies first
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    numpy \
    scipy \
    pydantic \
    python-dotenv \
    PyYAML \
    python-multipart \
    websockets \
    python-jose[cryptography] \
    passlib[bcrypt] \
    psutil \
    watchdog

# Install OpenCV headless (no GUI dependencies)
RUN pip install opencv-python-headless

# Try to install remaining dependencies, but don't fail if some are missing
RUN pip install --no-cache-dir -r /tmp/requirements.txt || \
    echo "Some optional dependencies failed to install - continuing with core functionality"

# ========================
# Production Stage
# ========================
FROM python:3.11-slim AS production

# Set build arguments
ARG BUILD_DATE
ARG BUILD_VERSION=1.0.0
ARG BUILD_COMMIT

# Add metadata labels
LABEL org.opencontainers.image.title="Billiards Trainer Backend API" \
      org.opencontainers.image.description="FastAPI backend for billiards training system" \
      org.opencontainers.image.version=${BUILD_VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.source="https://github.com/your-org/billiards-trainer" \
      org.opencontainers.image.revision=${BUILD_COMMIT} \
      maintainer="Billiards Trainer Team"

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    # API configuration
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WS_PORT=8001 \
    # Performance
    MAX_CONCURRENT_CONNECTIONS=100 \
    API_TIMEOUT=30 \
    WEBSOCKET_TIMEOUT=60 \
    # Directories
    CONFIG_DIR=/app/config \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    TEMP_DIR=/tmp

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # System utilities
    curl \
    ca-certificates \
    # Create non-root user
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app/config /app/data /app/logs /app/temp

# Create startup script (before switching to non-root user)
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Billiards Trainer Backend API..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "Debug Mode: ${DEBUG:-false}"
echo "API Host: ${API_HOST:-0.0.0.0}"
echo "API Port: ${API_PORT:-8000}"

# Validate required environment variables in production
if [ "$ENVIRONMENT" = "production" ] && [ -z "$JWT_SECRET_KEY" ]; then
    echo "⚠️  WARNING: JWT_SECRET_KEY not set, using default (not secure for production)"
fi

# Create directories if they don't exist
mkdir -p "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$TEMP_DIR"

# Run health check
echo "🔍 Running pre-startup health check..."
python -c "
import sys
try:
    # Test critical imports
    import fastapi
    import uvicorn
    print('✅ FastAPI/Uvicorn loaded successfully')
except ImportError as e:
    print(f'❌ Critical web framework missing: {e}')
    sys.exit(1)

try:
    import cv2
    print('✅ OpenCV loaded successfully')
except ImportError as e:
    print(f'⚠️  OpenCV not available: {e}')

try:
    import numpy
    import scipy
    print('✅ Scientific libraries loaded successfully')
except ImportError as e:
    print(f'⚠️  Scientific libraries not available: {e}')

print('✅ Health check completed')
"

# Set default values
WORKERS=${WORKERS:-1}
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
LOG_LEVEL=${LOG_LEVEL:-info}

echo "🌟 Starting server with ${WORKERS} worker(s) on ${HOST}:${PORT}"

# Start the application
if [ "$ENVIRONMENT" = "development" ]; then
    # Development mode with hot reload
    exec python dev_server.py --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
else
    # Production mode
    exec uvicorn api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --no-use-colors
fi
EOF

# Make entrypoint executable and set proper ownership
RUN chmod +x /app/entrypoint.sh \
    && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/api/v1/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# ========================
# Development Stage
# ========================
FROM production AS development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    git \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install --no-cache-dir \
    ipython \
    ipdb \
    pytest \
    pytest-asyncio \
    httpx \
    || echo "Some development packages failed to install"

# Switch back to app user
USER appuser

# Override environment for development
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Development entrypoint with hot reload
CMD ["python", "dev_server.py", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
