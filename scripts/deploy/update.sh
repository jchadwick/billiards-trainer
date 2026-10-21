#!/usr/bin/env bash
# Restart the local application after replacing its distribution files.

set -euo pipefail

echo "Restarting local application after update..."
pkill -f "python.*backend.main" || true
sleep 2
./run.sh
