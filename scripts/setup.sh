#!/bin/bash

# Billiards Trainer Development Setup Script
# This script sets up the development environment for the billiards trainer project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Check if Python 3.9+ is available
check_python() {
    print_info "Checking Python installation..."

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        return 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."

    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi

    python3 -m venv .venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."

    # Install the project in development mode
    pip install -e .

    # Install development dependencies
    pip install -e ".[dev,test,docs,performance]"

    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_info "Creating project directories..."

    mkdir -p logs data temp debug_images
    touch logs/.gitkeep data/.gitkeep temp/.gitkeep

    print_success "Directories created"
}

# Setup git hooks
setup_git_hooks() {
    print_info "Setting up git hooks..."

    if command -v pre-commit >/dev/null 2>&1; then
        pre-commit install
        print_success "Git hooks installed"
    else
        print_warning "pre-commit not found, skipping git hooks setup"
    fi
}

# Create .env.local from .env if it doesn't exist
setup_env_file() {
    print_info "Setting up environment file..."

    if [ ! -f ".env.local" ]; then
        cp .env .env.local
        print_success "Created .env.local from .env template"
        print_warning "Please review and modify .env.local for your local setup"
    else
        print_info ".env.local already exists"
    fi
}

# Check system dependencies
check_system_deps() {
    print_info "Checking system dependencies..."

    # Check for required system libraries
    MISSING_DEPS=()

    # Check for OpenCV dependencies (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! ldconfig -p | grep -q libgtk; then
            MISSING_DEPS+=("libgtk-3-dev")
        fi
        if ! ldconfig -p | grep -q libgl1; then
            MISSING_DEPS+=("libgl1-mesa-glx")
        fi
    fi

    # Check for macOS dependencies
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            print_warning "Homebrew not found. Some OpenGL features may not work properly"
        fi
    fi

    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Some system dependencies may be missing:"
        for dep in "${MISSING_DEPS[@]}"; do
            echo "  - $dep"
        done
        print_info "Install them with your system package manager"
    else
        print_success "System dependencies check passed"
    fi
}

# Test the installation
test_installation() {
    print_info "Testing installation..."

    # Test Python imports
    python -c "
import sys
print(f'Python version: {sys.version}')

# Test core dependencies
try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV import failed: {e}')
    sys.exit(1)

try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
except ImportError as e:
    print(f'NumPy import failed: {e}')
    sys.exit(1)

try:
    import fastapi
    print(f'FastAPI version: {fastapi.__version__}')
except ImportError as e:
    print(f'FastAPI import failed: {e}')
    sys.exit(1)

try:
    import pydantic
    print(f'Pydantic version: {pydantic.__version__}')
except ImportError as e:
    print(f'Pydantic import failed: {e}')
    sys.exit(1)

print('All core dependencies imported successfully!')
"

    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Main setup function
main() {
    print_header "Billiards Trainer Development Setup"

    # Change to script directory
    cd "$(dirname "$0")/.."

    print_info "Setting up development environment in: $(pwd)"

    # Run setup steps
    check_python || exit 1
    check_system_deps
    create_venv || exit 1
    activate_venv || exit 1
    upgrade_pip || exit 1
    install_dependencies || exit 1
    create_directories || exit 1
    setup_git_hooks
    setup_env_file
    test_installation || exit 1

    print_header "Setup Complete!"

    echo -e "${GREEN}Development environment setup successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  1. Activate the virtual environment:"
    echo "     ${CYAN}source .venv/bin/activate${NC}"
    echo ""
    echo "  2. Review and modify .env.local for your setup"
    echo ""
    echo "  3. Run the application:"
    echo "     ${CYAN}make run${NC}"
    echo ""
    echo "  4. Run tests:"
    echo "     ${CYAN}make test${NC}"
    echo ""
    echo -e "${YELLOW}Available commands:${NC}"
    echo "     ${CYAN}make help${NC}    - Show all available commands"
    echo "     ${CYAN}make dev${NC}     - Setup development environment"
    echo "     ${CYAN}make test${NC}    - Run tests"
    echo "     ${CYAN}make lint${NC}    - Run code linting"
    echo "     ${CYAN}make format${NC}  - Format code"
    echo ""
    echo -e "${GREEN}Happy coding!${NC}"
}

# Run main function
main "$@"
