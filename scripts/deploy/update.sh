#!/bin/bash
# Simple update script - assumes you've rsync'd new files
echo "Restarting after update..."
pkill -f "python.*backend.main" || true
sleep 2
./run.sh
