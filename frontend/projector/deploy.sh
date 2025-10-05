#!/bin/bash

# Billiards Projector Deployment Script

TARGET_HOST="jchadwick@192.168.1.31"
TARGET_DIR="/opt/billiards-trainer/frontend/projector"

echo "Billiards Projector Deployment"
echo "=============================="

# Parse command line arguments
ACTION=${1:-deploy}

case "$ACTION" in
    deploy)
        echo "Deploying to $TARGET_HOST:$TARGET_DIR..."

        # Create target directory if it doesn't exist
        ssh $TARGET_HOST "mkdir -p $TARGET_DIR"

        # Sync files (exclude development files)
        rsync -av --exclude=".git" --exclude="*.swp" --exclude=".DS_Store" \
            ./ $TARGET_HOST:$TARGET_DIR/

        echo "Deployment complete!"
        echo ""
        echo "To run on target:"
        echo "  ssh $TARGET_HOST"
        echo "  cd $TARGET_DIR"
        echo "  love ."
        ;;

    install)
        echo "Installing LÖVE2D on target system..."
        ssh $TARGET_HOST "sudo apt-get update && sudo apt-get install -y love"
        ;;

    run)
        echo "Running projector on target system..."
        ssh -t $TARGET_HOST "cd $TARGET_DIR && love ."
        ;;

    test)
        echo "Running in test mode (windowed) on target..."
        # Create temporary conf.lua with windowed mode
        cat > /tmp/conf_test.lua << 'EOF'
function love.conf(t)
    t.window.fullscreen = false
    t.window.width = 1280
    t.window.height = 720
end
EOF
        scp /tmp/conf_test.lua $TARGET_HOST:$TARGET_DIR/conf_test.lua
        ssh -t $TARGET_HOST "cd $TARGET_DIR && love . --config conf_test.lua"
        ;;

    logs)
        echo "Viewing logs from target system..."
        ssh $TARGET_HOST "cd $TARGET_DIR && tail -f love.log 2>/dev/null || echo 'No logs found'"
        ;;

    status)
        echo "Checking status on target system..."
        ssh $TARGET_HOST "pgrep -fl love || echo 'Projector not running'"
        ;;

    stop)
        echo "Stopping projector on target system..."
        ssh $TARGET_HOST "pkill -f love || echo 'Projector was not running'"
        ;;

    *)
        echo "Usage: $0 [deploy|install|run|test|logs|status|stop]"
        echo ""
        echo "  deploy  - Deploy files to target system (default)"
        echo "  install - Install LÖVE2D on target system"
        echo "  run     - Run projector on target (fullscreen)"
        echo "  test    - Run projector in windowed mode for testing"
        echo "  logs    - View projector logs"
        echo "  status  - Check if projector is running"
        echo "  stop    - Stop projector"
        exit 1
        ;;
esac
