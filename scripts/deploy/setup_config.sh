#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== Billiards Trainer Configuration Setup ===${NC}"
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}Configuration file already exists.${NC}"
    read -p "Do you want to reconfigure? (y/N): " RECONFIGURE
    if [[ ! $RECONFIGURE =~ ^[Yy]$ ]]; then
        echo "Configuration unchanged."
        exit 0
    fi
    cp .env .env.backup
    echo -e "${GREEN}Backup created: .env.backup${NC}"
fi

# Copy from example
cp .env.example .env

echo -e "${BLUE}Please provide the following configuration:${NC}"
echo ""

# Generate secure keys
echo -e "${YELLOW}Generating secure keys...${NC}"
JWT_SECRET=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 16)
sed -i.tmp "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
sed -i.tmp "s/DEFAULT_API_KEY=.*/DEFAULT_API_KEY=$API_KEY/" .env
rm .env.tmp
echo -e "${GREEN}✓ Generated JWT secret and API key${NC}"

# API Configuration
echo ""
read -p "API Port [8000]: " API_PORT
API_PORT=${API_PORT:-8000}
sed -i.tmp "s/API_PORT=.*/API_PORT=$API_PORT/" .env
rm .env.tmp

# Camera Configuration
echo ""
echo -e "${BLUE}Camera Configuration:${NC}"
read -p "Camera index (0 for built-in, 1+ for external) [0]: " CAMERA_INDEX
CAMERA_INDEX=${CAMERA_INDEX:-0}
sed -i.tmp "s/CAMERA_INDEX=.*/CAMERA_INDEX=$CAMERA_INDEX/" .env
rm .env.tmp

read -p "Camera resolution - Width [1920]: " CAMERA_WIDTH
CAMERA_WIDTH=${CAMERA_WIDTH:-1920}
sed -i.tmp "s/CAMERA_WIDTH=.*/CAMERA_WIDTH=$CAMERA_WIDTH/" .env
rm .env.tmp

read -p "Camera resolution - Height [1080]: " CAMERA_HEIGHT
CAMERA_HEIGHT=${CAMERA_HEIGHT:-1080}
sed -i.tmp "s/CAMERA_HEIGHT=.*/CAMERA_HEIGHT=$CAMERA_HEIGHT/" .env
rm .env.tmp

# Projector Configuration
echo ""
echo -e "${BLUE}Projector Configuration:${NC}"
read -p "Projector display index [1]: " PROJECTOR_INDEX
PROJECTOR_INDEX=${PROJECTOR_INDEX:-1}
sed -i.tmp "s/PROJECTOR_DISPLAY_INDEX=.*/PROJECTOR_DISPLAY_INDEX=$PROJECTOR_INDEX/" .env
rm .env.tmp

read -p "Projector width [1920]: " PROJECTOR_WIDTH
PROJECTOR_WIDTH=${PROJECTOR_WIDTH:-1920}
sed -i.tmp "s/PROJECTOR_WIDTH=.*/PROJECTOR_WIDTH=$PROJECTOR_WIDTH/" .env
rm .env.tmp

read -p "Projector height [1080]: " PROJECTOR_HEIGHT
PROJECTOR_HEIGHT=${PROJECTOR_HEIGHT:-1080}
sed -i.tmp "s/PROJECTOR_HEIGHT=.*/PROJECTOR_HEIGHT=$PROJECTOR_HEIGHT/" .env
rm .env.tmp

read -p "Projector fullscreen? (true/false) [false]: " PROJECTOR_FULLSCREEN
PROJECTOR_FULLSCREEN=${PROJECTOR_FULLSCREEN:-false}
sed -i.tmp "s/PROJECTOR_FULLSCREEN=.*/PROJECTOR_FULLSCREEN=$PROJECTOR_FULLSCREEN/" .env
rm .env.tmp

# Environment
echo ""
read -p "Environment (production/development) [production]: " ENVIRONMENT
ENVIRONMENT=${ENVIRONMENT:-production}
sed -i.tmp "s/ENVIRONMENT=.*/ENVIRONMENT=$ENVIRONMENT/" .env
rm .env.tmp

if [ "$ENVIRONMENT" = "development" ]; then
    sed -i.tmp "s/DEBUG=.*/DEBUG=true/" .env
    sed -i.tmp "s/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/" .env
    rm .env.tmp
fi

echo ""
echo -e "${GREEN}=== Configuration Complete ===${NC}"
echo ""
echo -e "${BLUE}Generated credentials (save these securely):${NC}"
echo -e "API Key: ${YELLOW}$API_KEY${NC}"
echo ""
echo "Configuration saved to .env"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run ./run.sh to start the server"
echo "2. Access the web interface at http://localhost:$API_PORT"
echo ""
