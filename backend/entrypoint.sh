#!/bin/bash
set -e

# Set default values
WORKERS=${WORKERS:-1}
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
LOG_LEVEL=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
JWT_SECRET_KEY=${JWT_SECRET_KEY:-top_secret_key}


echo "🚀 Starting Billiards Trainer Backend API..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "Debug Mode: ${DEBUG:-false}"
echo "API Host: ${API_HOST:-0.0.0.0}"
echo "API Port: ${API_PORT:-8000}"
echo "Config Dir: ${CONFIG_DIR}"
echo "Data Dir: ${DATA_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "Temp Dir: ${TEMP_DIR}"


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

echo "🌟 Starting server with ${WORKERS} worker(s) on ${HOST}:${PORT}"

exec uvicorn api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --no-use-colors
