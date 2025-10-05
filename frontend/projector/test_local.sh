#!/bin/bash

# Local Projector Test Script
# Starts LÖVE projector and sends test UDP data in parallel

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Billiards Projector Local Test ===${NC}"
echo ""

# Check if LÖVE is installed
if ! command -v love &> /dev/null; then
    echo -e "${YELLOW}Error: LÖVE2D is not installed${NC}"
    echo "Install with: brew install --cask love"
    exit 1
fi

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
DURATION=${1:-30}
INTERVAL=${2:-0.5}
FULLSCREEN=${3:-false}

echo -e "${BLUE}Configuration:${NC}"
echo "  Duration: ${DURATION}s"
echo "  Update interval: ${INTERVAL}s"
echo "  Fullscreen: ${FULLSCREEN}"
echo ""

# Kill any existing LÖVE processes
pkill -f "love.*projector" || true
sleep 1

# Start LÖVE projector in background
if [ "$FULLSCREEN" = "true" ]; then
    echo -e "${GREEN}Starting projector in FULLSCREEN mode...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use open command
        open -a love "$SCRIPT_DIR" &
        LOVE_PID=$!
    else
        # Linux
        love . &
        LOVE_PID=$!
    fi
else
    echo -e "${GREEN}Starting projector in WINDOWED mode...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use open command with env var
        PROJECTOR_WINDOWED=true open -a love "$SCRIPT_DIR" &
        LOVE_PID=$!
    else
        # Linux
        PROJECTOR_WINDOWED=true love . &
        LOVE_PID=$!
    fi
fi

echo -e "${BLUE}Projector started (PID: $LOVE_PID)${NC}"
echo -e "${YELLOW}Waiting 3 seconds for projector to initialize...${NC}"
sleep 3

# Start sending test data
echo ""
echo -e "${GREEN}Starting test data stream...${NC}"
echo -e "${BLUE}Sending trajectory updates every ${INTERVAL}s for ${DURATION}s${NC}"
echo ""

python3 test_udp_sender.py continuous "$DURATION" "$INTERVAL" &
SENDER_PID=$!

echo ""
echo -e "${YELLOW}Test running...${NC}"
echo ""
echo "What you should see:"
echo "  - Animated green trajectory line moving in curves"
echo "  - Yellow collision markers appearing periodically"
echo "  - White dashed aim line rotating"
echo "  - Ghost ball (white circle with '8')"
echo "  - Network status: 'UDP: Connected (Port 9999)' in green"
echo ""
echo "Controls:"
echo "  C - Toggle calibration mode"
echo "  P - Pause/unpause"
echo "  ESC - Quit projector"
echo "  Ctrl+C - Stop test script"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Kill sender
    if kill -0 $SENDER_PID 2>/dev/null; then
        kill $SENDER_PID 2>/dev/null || true
        echo "Stopped test data sender"
    fi

    # Kill LÖVE (macOS requires different approach)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        pkill -f "love.*projector" || true
    else
        if kill -0 $LOVE_PID 2>/dev/null; then
            kill $LOVE_PID 2>/dev/null || true
        fi
    fi
    echo "Stopped projector"

    echo -e "${GREEN}Test complete!${NC}"
    exit 0
}

# Trap for graceful shutdown
trap cleanup SIGINT SIGTERM

# Wait for sender to finish
wait $SENDER_PID

# After sender finishes, wait a bit then cleanup
echo ""
echo -e "${YELLOW}Test data stream complete. Projector will remain open.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop, or ESC in the projector window to quit.${NC}"

# Keep script running until user stops it
wait
