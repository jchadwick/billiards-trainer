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

# Use the CPU-only PyTorch index. PyPI's default Linux wheels now bundle CUDA,
# which expanded this CPU deployment image by several gigabytes.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision \
    && python -c "import torch, torchvision; assert not torch.cuda.is_available(); print(f'CPU PyTorch {torch.__version__}, torchvision {torchvision.__version__}')"

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
    psutil \
    watchdog \
    sqlalchemy \
    redis \
    aiofiles \
    aiohttp \
    ultralytics \
    onnxruntime \
    onnx

# Install OpenCV headless (no GUI dependencies)
RUN pip install --no-cache-dir opencv-python-headless

# Install the remaining runtime dependencies. Calibration uses scikit-learn;
# test runners and projector display libraries stay out of the production image.
RUN pip install --no-cache-dir \
    scikit-learn \
    Pillow \
    email-validator \
    jsonschema

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
    # API configuration
    API_PORT=8000 \
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

# Copy entrypoint script
COPY --chown=root:root backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set proper ownership for app directory
RUN chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# The repository may use a symlink for its externally managed model directory.
# Normalize that path in the image so Docker volume mounts can target /app/models.
RUN if [ -L /app/models ]; then rm /app/models; fi \
    && mkdir -p /app/models \
    && chown appuser:appuser /app/models

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/api/v1/health || exit 1

# Expose ports
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
