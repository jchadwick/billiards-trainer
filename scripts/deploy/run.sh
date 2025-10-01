#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== Billiards Trainer Production Runner ===${NC}"

# Check if config exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No configuration found. Running setup...${NC}"
    ./setup_config.sh
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if watchdog is installed for auto-restart
if ! pip show watchdog &> /dev/null; then
    echo "Installing watchdog for auto-restart..."
    pip install -q watchdog
fi

# Kill any existing instances
pkill -f "python.*backend.main" || true
sleep 1

echo -e "${GREEN}Starting Billiards Trainer...${NC}"
echo "The server will auto-restart when files change."
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Function to start the server
start_server() {
    python -m backend.main &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
}

# Function to stop the server
stop_server() {
    if [ ! -z "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

# Trap for graceful shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    stop_server
    pkill -f "watchmedo" || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Watch for changes and auto-restart (watchmedo will start the server)
watchmedo auto-restart \
    --directory=. \
    --pattern="*.py;*.json;*.yaml;*.yml" \
    --recursive \
    --ignore-patterns="*/__pycache__/*;*/.venv/*;*/logs/*;*/data/*;*/.git/*" \
    -- python -m backend.main
