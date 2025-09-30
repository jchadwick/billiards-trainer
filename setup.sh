#!/bin/bash
# =================================================================
# Billiards Trainer Setup Script for Ubuntu
# =================================================================
# This script verifies system dependencies, versions, and environment
# configuration for the Billiards Trainer application.
#
# Usage: ./setup.sh [--fix]
#   --fix: Attempt to automatically install missing dependencies
# =================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
FIX_MODE=false
HAS_ERRORS=false
HAS_WARNINGS=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --fix)
            FIX_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--fix]"
            echo "  --fix: Attempt to automatically install missing dependencies"
            exit 0
            ;;
    esac
done

# =================================================================
# Helper Functions
# =================================================================

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    HAS_ERRORS=true
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    HAS_WARNINGS=true
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    local cmd=$1
    local name=$2
    local min_version=$3

    if command -v "$cmd" &> /dev/null; then
        local version=$("$cmd" --version 2>&1 | head -n 1)
        print_success "$name found: $version"

        if [ -n "$min_version" ]; then
            local current_version=$(echo "$version" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -n 1)
            if [ -n "$current_version" ]; then
                if ! version_gte "$current_version" "$min_version"; then
                    print_warning "$name version $current_version is below recommended $min_version"
                    return 1
                fi
            fi
        fi
        return 0
    else
        print_error "$name not found"
        return 1
    fi
}

version_gte() {
    # Returns 0 if $1 >= $2, 1 otherwise
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

install_package() {
    local package=$1
    if [ "$FIX_MODE" = true ]; then
        print_info "Installing $package..."
        sudo apt-get install -y "$package"
    else
        print_info "Run with --fix to automatically install $package"
    fi
}

# =================================================================
# System Checks
# =================================================================

print_header "System Information"

# Check OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    print_info "OS: $NAME $VERSION"

    # Verify Ubuntu
    if [[ "$ID" != "ubuntu" ]]; then
        print_warning "This script is designed for Ubuntu. Current OS: $ID"
    fi
else
    print_error "Cannot determine OS version"
fi

# Check architecture
ARCH=$(uname -m)
print_info "Architecture: $ARCH"

# Check kernel
KERNEL=$(uname -r)
print_info "Kernel: $KERNEL"

# =================================================================
# Core Dependencies
# =================================================================

print_header "Core Dependencies"

# Python
if ! check_command python3 "Python" "3.9"; then
    install_package python3
fi

# Pip
if ! check_command pip3 "pip" ""; then
    install_package python3-pip
fi

# Node.js
if ! check_command node "Node.js" "18.0"; then
    print_error "Node.js not found or version too old (need >=18.0)"
    if [ "$FIX_MODE" = true ]; then
        print_info "Installing Node.js 20.x..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
fi

# npm
if ! check_command npm "npm" "9.0"; then
    print_warning "npm not found or version too old"
fi

# Docker
if ! check_command docker "Docker" "24.0"; then
    install_package docker.io
fi

# Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose not found"
    install_package docker-compose-plugin
else
    if docker compose version &> /dev/null 2>&1; then
        version=$(docker compose version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
        print_success "Docker Compose found: $version"
    else
        version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
        print_success "Docker Compose found: $version"
    fi
fi

# Git
check_command git "Git" ""

# =================================================================
# System Libraries
# =================================================================

print_header "System Libraries"

check_pkg() {
    local pkg=$1
    if dpkg -l | grep -q "^ii  $pkg"; then
        print_success "$pkg installed"
        return 0
    else
        print_error "$pkg not installed"
        install_package "$pkg"
        return 1
    fi
}

# OpenCV dependencies
check_pkg libopencv-dev
check_pkg python3-opencv

# Camera/Video dependencies
check_pkg libv4l-dev
check_pkg v4l-utils

# Graphics libraries
check_pkg libgl1-mesa-glx
check_pkg libglib2.0-0
check_pkg libsm6
check_pkg libxext6
check_pkg libxrender-dev
check_pkg libgomp1

# OpenGL for projector
check_pkg libgl1-mesa-dev
check_pkg libglu1-mesa-dev
check_pkg freeglut3-dev

# SDL for pygame
check_pkg libsdl2-dev
check_pkg libsdl2-image-dev
check_pkg libsdl2-mixer-dev
check_pkg libsdl2-ttf-dev

# =================================================================
# Python Dependencies
# =================================================================

print_header "Python Dependencies"

# Check if requirements.txt exists
if [ -f "backend/requirements.txt" ]; then
    print_info "Checking Python packages from requirements.txt..."

    # Key packages to verify
    PYTHON_PACKAGES=(
        "opencv-python:4.8.0"
        "numpy:1.24.0"
        "fastapi:0.100.0"
        "uvicorn:0.23.0"
        "pygame:2.5.0"
    )

    for pkg_info in "${PYTHON_PACKAGES[@]}"; do
        IFS=':' read -r pkg min_ver <<< "$pkg_info"
        if python3 -c "import $pkg" 2>/dev/null; then
            version=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
            print_success "$pkg installed: $version"
        else
            print_error "$pkg not installed"
            if [ "$FIX_MODE" = true ]; then
                pip3 install "$pkg>=$min_ver"
            fi
        fi
    done
else
    print_warning "backend/requirements.txt not found"
fi

# =================================================================
# Node.js Dependencies
# =================================================================

print_header "Node.js Dependencies"

if [ -f "frontend/web/package.json" ]; then
    if [ -d "frontend/web/node_modules" ]; then
        print_success "node_modules directory exists"
    else
        print_warning "node_modules not found. Run: cd frontend/web && npm install"
        if [ "$FIX_MODE" = true ]; then
            print_info "Installing Node.js dependencies..."
            (cd frontend/web && npm install)
        fi
    fi
else
    print_warning "frontend/web/package.json not found"
fi

# =================================================================
# Camera Access
# =================================================================

print_header "Camera Access"

# Check for video devices
if [ -d "/dev" ]; then
    VIDEO_DEVICES=$(ls /dev/video* 2>/dev/null || echo "")
    if [ -n "$VIDEO_DEVICES" ]; then
        print_success "Video devices found:"
        for dev in $VIDEO_DEVICES; do
            echo "  - $dev"
        done

        # Check permissions
        for dev in $VIDEO_DEVICES; do
            if [ -r "$dev" ] && [ -w "$dev" ]; then
                print_success "User has read/write access to $dev"
            else
                print_warning "User may not have access to $dev"
                print_info "Add user to 'video' group: sudo usermod -aG video \$USER"
            fi
        done
    else
        print_warning "No video devices found at /dev/video*"
        print_info "Connect a camera or check that camera drivers are installed"
    fi
fi

# Check video group membership
if groups | grep -q video; then
    print_success "User is in 'video' group"
else
    print_warning "User is not in 'video' group"
    if [ "$FIX_MODE" = true ]; then
        sudo usermod -aG video "$USER"
        print_info "Added user to video group. Log out and back in for changes to take effect."
    fi
fi

# Test camera access with v4l2
if command -v v4l2-ctl &> /dev/null; then
    print_info "Testing camera with v4l2-ctl..."
    if v4l2-ctl --list-devices &> /dev/null; then
        print_success "Camera devices accessible"
        v4l2-ctl --list-devices | head -n 10
    else
        print_warning "Could not list camera devices"
    fi
fi

# =================================================================
# Environment Configuration
# =================================================================

print_header "Environment Configuration"

# Check for .env file
if [ -f ".env" ]; then
    print_success ".env file found"

    # Check required variables
    REQUIRED_VARS=(
        "JWT_SECRET_KEY"
        "CAMERA_INDEX"
        "API_PORT"
    )

    for var in "${REQUIRED_VARS[@]}"; do
        if grep -q "^${var}=" .env; then
            value=$(grep "^${var}=" .env | cut -d'=' -f2-)
            if [ -n "$value" ]; then
                print_success "$var is set"
            else
                print_warning "$var is defined but empty"
            fi
        else
            print_warning "$var not found in .env"
        fi
    done
else
    print_warning ".env file not found"
    print_info "Create .env file with required configuration"

    if [ "$FIX_MODE" = true ] && [ -f ".env.example" ]; then
        print_info "Copying .env.example to .env..."
        cp .env.example .env
        print_info "Please edit .env and set required values"
    fi
fi

# Check for config directory
if [ -d "config" ]; then
    print_success "config directory found"
else
    print_warning "config directory not found"
fi

# =================================================================
# Docker Environment
# =================================================================

print_header "Docker Environment"

# Check Docker daemon
if systemctl is-active --quiet docker; then
    print_success "Docker daemon is running"
elif service docker status &> /dev/null; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    if [ "$FIX_MODE" = true ]; then
        print_info "Starting Docker daemon..."
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
fi

# Check Docker permissions
if docker ps &> /dev/null; then
    print_success "User can run Docker commands"
else
    print_warning "User cannot run Docker commands without sudo"
    if [ "$FIX_MODE" = true ]; then
        print_info "Adding user to docker group..."
        sudo usermod -aG docker "$USER"
        print_info "Log out and back in for changes to take effect"
    else
        print_info "Add user to docker group: sudo usermod -aG docker \$USER"
    fi
fi

# =================================================================
# Display Configuration
# =================================================================

print_header "Display Configuration"

# Check for X server
if [ -n "$DISPLAY" ]; then
    print_success "DISPLAY environment variable set: $DISPLAY"

    # Check if X server is accessible
    if xdpyinfo &> /dev/null; then
        print_success "X server is accessible"
    else
        print_warning "Cannot connect to X server"
    fi
else
    print_warning "DISPLAY environment variable not set"
    print_info "Projector features require X server or Wayland"
fi

# Check available displays
if command -v xrandr &> /dev/null && xrandr &> /dev/null; then
    print_info "Available displays:"
    xrandr --listmonitors | tail -n +2
else
    print_warning "Cannot query display information (xrandr not available)"
fi

# =================================================================
# Network Configuration
# =================================================================

print_header "Network Configuration"

# Check if ports are available
check_port() {
    local port=$1
    local name=$2
    if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Port $port ($name) is already in use"
        return 1
    else
        print_success "Port $port ($name) is available"
        return 0
    fi
}

check_port 8000 "API"
check_port 8001 "WebSocket"
check_port 3000 "Frontend Dev"
check_port 6379 "Redis"

# =================================================================
# Directory Structure
# =================================================================

print_header "Directory Structure"

REQUIRED_DIRS=(
    "backend"
    "frontend/web"
    "config"
    "data"
    "logs"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "$dir/ exists"
    else
        print_warning "$dir/ not found"
        if [ "$FIX_MODE" = true ]; then
            mkdir -p "$dir"
            print_info "Created $dir/"
        fi
    fi
done

# Check write permissions
for dir in data logs; do
    if [ -d "$dir" ] && [ -w "$dir" ]; then
        print_success "$dir/ is writable"
    elif [ -d "$dir" ]; then
        print_error "$dir/ is not writable"
    fi
done

# =================================================================
# Summary
# =================================================================

print_header "Setup Summary"

if [ "$HAS_ERRORS" = true ]; then
    echo -e "${RED}Setup verification found errors that need to be addressed.${NC}"
    if [ "$FIX_MODE" = false ]; then
        echo -e "${YELLOW}Run '$0 --fix' to automatically fix some issues.${NC}"
    fi
    exit 1
elif [ "$HAS_WARNINGS" = true ]; then
    echo -e "${YELLOW}Setup verification completed with warnings.${NC}"
    echo -e "${YELLOW}Review warnings above and address as needed.${NC}"
    exit 0
else
    echo -e "${GREEN}All checks passed! System is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Install Python dependencies: pip3 install -r backend/requirements.txt"
    echo "  2. Install Node.js dependencies: cd frontend/web && npm install"
    echo "  3. Configure .env file with your settings"
    echo "  4. Start services: docker-compose up -d"
    exit 0
fi
