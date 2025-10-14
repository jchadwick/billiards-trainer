#!/bin/bash
# Test runner for visualizer module initialization

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting visualizer initialization tests..."
echo

# Backup main.lua
if [ -f main.lua ]; then
    cp main.lua main.lua.backup
fi

# Copy test file to main.lua
cp test_init.lua main.lua

# Run tests with LOVE2D
if love --version &>/dev/null; then
    echo "Running tests with LOVE2D..."
    love . 2>&1
    TEST_EXIT_CODE=$?
else
    echo "ERROR: LOVE2D not found. Please install LOVE2D to run tests."
    TEST_EXIT_CODE=1
fi

# Restore main.lua
if [ -f main.lua.backup ]; then
    mv main.lua.backup main.lua
else
    echo "WARNING: Could not restore main.lua backup"
fi

exit $TEST_EXIT_CODE
