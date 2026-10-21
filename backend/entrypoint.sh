#!/bin/bash
set -e

# The application owns one camera and one game state per backend process. Keep
# this launcher single-process; distribute tables by running separate instances.
CONFIG_FILE=${CONFIG_FILE:-/app/config.json}

echo "🚀 Starting Billiards Trainer Backend API..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "Debug Mode: ${DEBUG:-false}"
echo "Configuration file: ${CONFIG_FILE}"
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

echo "🌟 Starting the application with one backend process per table"

exec python -m backend.main --config "$CONFIG_FILE"
