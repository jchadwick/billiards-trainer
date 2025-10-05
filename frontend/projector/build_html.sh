#!/bin/bash

# Build LÖVE Projector as HTML/WebAssembly App
# Uses love.js to compile LÖVE game to web app

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Billiards Projector HTML Build ===${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
OUTPUT_DIR="${SCRIPT_DIR}/build/web"
LOVEJS_DIR="${SCRIPT_DIR}/love.js"
GAME_FILE="${SCRIPT_DIR}/billiards-projector.love"
GAME_TITLE="Billiards Trainer Projector"

# Check if love.js exists
if [ ! -d "$LOVEJS_DIR" ]; then
    echo -e "${YELLOW}love.js not found. Cloning repository...${NC}"
    git clone https://github.com/Davidobot/love.js.git "$LOVEJS_DIR"
fi

# Check if npm dependencies are installed
if [ ! -d "$LOVEJS_DIR/node_modules" ]; then
    echo -e "${BLUE}Installing love.js dependencies...${NC}"
    cd "$LOVEJS_DIR"
    npm install
    cd "$SCRIPT_DIR"
fi

# Create .love file
echo -e "${BLUE}Creating .love file...${NC}"
rm -f "$GAME_FILE"

# Create temporary directory for .love contents
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Copy all necessary files
cp -r "$SCRIPT_DIR"/*.lua "$TMP_DIR/" 2>/dev/null || true
cp -r "$SCRIPT_DIR"/core "$TMP_DIR/" 2>/dev/null || true
cp -r "$SCRIPT_DIR"/modules "$TMP_DIR/" 2>/dev/null || true
cp -r "$SCRIPT_DIR"/lib "$TMP_DIR/" 2>/dev/null || true
cp -r "$SCRIPT_DIR"/config "$TMP_DIR/" 2>/dev/null || true

# Remove test files
rm -f "$TMP_DIR"/conf_windowed.lua

# Create the .love file (which is just a zip)
cd "$TMP_DIR"
zip -9 -r "$GAME_FILE" . >/dev/null
cd "$SCRIPT_DIR"

echo -e "${GREEN}Created ${GAME_FILE}${NC}"

# Build with love.js
echo -e "${BLUE}Building web version with love.js...${NC}"

# Clean previous build
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run love.js build (compatibility mode for broader browser support)
npx love.js \
    --title "$GAME_TITLE" \
    --memory 67108864 \
    --compatibility \
    "$GAME_FILE" \
    "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/index.html" ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo -e "${BLUE}Output directory:${NC} $OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}To test locally:${NC}"
    echo "  cd $OUTPUT_DIR"
    echo "  python3 -m http.server 8080"
    echo "  # Then open http://localhost:8080 in your browser"
    echo ""
    echo -e "${YELLOW}To deploy to target:${NC}"
    echo "  rsync -av $OUTPUT_DIR/ jchadwick@192.168.1.31:/opt/billiards-trainer/frontend/projector/web/"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
