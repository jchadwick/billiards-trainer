#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}=== Billiards Trainer Deployment Packager ===${NC}"

# Clean dist folder
echo -e "${BLUE}Cleaning dist folder...${NC}"
rm -rf dist
mkdir -p dist

# Package backend
echo -e "${BLUE}Packaging backend...${NC}"
mkdir -p dist/backend

# Copy backend Python files
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    --exclude='*.egg-info' backend/ dist/backend/

# Copy requirements
cp backend/requirements.txt dist/requirements.txt

# Package frontend
echo -e "${BLUE}Packaging frontend...${NC}"
mkdir -p dist/frontend
cp -r frontend/web/dist/* dist/frontend/

# Copy config files
echo -e "${BLUE}Packaging config files...${NC}"
mkdir -p dist/config
if [ -d "config" ]; then
    cp -r config/* dist/config/ 2>/dev/null || true
fi
cp .env.example dist/.env.example

# Create runtime directories
echo -e "${BLUE}Creating runtime directories...${NC}"
mkdir -p dist/data
mkdir -p dist/logs

# Copy deployment scripts
echo -e "${BLUE}Creating deployment scripts...${NC}"
cp "$SCRIPT_DIR/run.sh" dist/run.sh
cp "$SCRIPT_DIR/setup_config.sh" dist/setup_config.sh
cp "$SCRIPT_DIR/update.sh" dist/update.sh
cp "$SCRIPT_DIR/README.md" dist/README.md
chmod +x dist/run.sh
chmod +x dist/setup_config.sh
chmod +x dist/update.sh

echo ""
echo -e "${GREEN}=== Deployment Package Complete ===${NC}"
echo ""
echo "Distribution created in: ./dist/"
echo ""
echo "To deploy:"
echo "  rsync -av dist/ user@server:/opt/billiards-trainer/"
echo "  ssh user@server 'cd /opt/billiards-trainer && ./run.sh'"
echo ""
echo "To update existing deployment:"
echo "  rsync -av dist/ user@server:/opt/billiards-trainer/"
echo "  (System will auto-restart)"
echo ""
