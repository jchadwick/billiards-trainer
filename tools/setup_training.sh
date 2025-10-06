#!/bin/bash
#
# Training Environment Setup Script for Billiards Trainer
#
# This script sets up the necessary dependencies for training YOLO models
# including PyTorch with appropriate GPU/CPU support and ultralytics.
#
# Usage:
#   ./tools/setup_training.sh [--cpu-only]
#
# Options:
#   --cpu-only    Force CPU-only installation (skip CUDA detection)
#

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Billiards Trainer - Training Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse arguments
FORCE_CPU=false
if [[ "$1" == "--cpu-only" ]]; then
    FORCE_CPU=true
    echo -e "${YELLOW}INFO: CPU-only mode requested${NC}"
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}ERROR: Python 3.8 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}WARNING: No virtual environment detected${NC}"
    echo "It's recommended to activate a virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# System information
echo -e "${BLUE}System Information:${NC}"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  CPU cores: $(python3 -c "import os; print(os.cpu_count())")"

# Check available memory
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    echo "  RAM: $TOTAL_MEM"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    TOTAL_MEM=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024) " GB"}')
    echo "  RAM: $TOTAL_MEM"
fi
echo ""

# Detect CUDA availability
CUDA_AVAILABLE=false
CUDA_VERSION=""

if [ "$FORCE_CPU" = false ]; then
    echo -e "${BLUE}Detecting CUDA...${NC}"

    # Check for nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ nvidia-smi found${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

        # Try to detect CUDA version
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
            echo "CUDA Toolkit version: $CUDA_VERSION"
            CUDA_AVAILABLE=true
        else
            echo -e "${YELLOW}WARNING: nvidia-smi found but nvcc not in PATH${NC}"
            echo "Attempting to detect CUDA from nvidia-smi driver..."
            DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
            echo "Driver version: $DRIVER_VERSION"
            # Assume recent CUDA if driver is recent enough
            CUDA_AVAILABLE=true
        fi
    else
        echo -e "${YELLOW}No CUDA-capable GPU detected${NC}"
    fi
else
    echo -e "${YELLOW}Skipping CUDA detection (CPU-only mode)${NC}"
fi
echo ""

# Determine PyTorch installation command
echo -e "${BLUE}Preparing PyTorch installation...${NC}"

if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "${GREEN}Installing PyTorch with CUDA support${NC}"

    # Determine CUDA version for PyTorch
    # PyTorch supports CUDA 11.8 and 12.1 as of late 2024
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    CUDA_DESC="CUDA 12.1"

    # If we detected a specific CUDA version, try to match it
    if [[ "$CUDA_VERSION" =~ ^11\. ]]; then
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        CUDA_DESC="CUDA 11.8"
    fi

    echo "Using PyTorch index: $TORCH_INDEX_URL ($CUDA_DESC)"
    TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url $TORCH_INDEX_URL"
else
    echo -e "${YELLOW}Installing PyTorch with CPU-only support${NC}"
    TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
fi
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install PyTorch
echo -e "${BLUE}Installing PyTorch (this may take several minutes)...${NC}"
echo "Command: $TORCH_INSTALL_CMD"
eval $TORCH_INSTALL_CMD
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install ultralytics and training dependencies
echo -e "${BLUE}Installing ultralytics and training dependencies...${NC}"
pip install ultralytics opencv-python-headless Pillow PyYAML scipy matplotlib seaborn pandas tqdm
echo -e "${GREEN}✓ Training dependencies installed${NC}"
echo ""

# Create requirements_training.txt
echo -e "${BLUE}Creating requirements_training.txt...${NC}"
cat > "$PROJECT_ROOT/requirements_training.txt" << 'EOF'
# Training dependencies for YOLO model training
# Install with: pip install -r requirements_training.txt
#
# For CUDA support, install PyTorch first using:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
# For CPU-only, use:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core training framework
ultralytics>=8.0.0

# PyTorch (installed separately via index-url, see above)
# torch>=2.0.0
# torchvision>=0.15.0
# torchaudio>=2.0.0

# Computer vision
opencv-python-headless>=4.8.0
Pillow>=10.0.0

# Data processing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Configuration
PyYAML>=6.0

# Visualization and monitoring
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0

# Utilities
tqdm>=4.65.0
psutil>=5.9.0

# Development
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
EOF

echo -e "${GREEN}✓ requirements_training.txt created${NC}"
echo ""

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python3 << 'VERIFY_SCRIPT'
import sys
import platform

print("\n" + "="*50)
print("Installation Verification")
print("="*50)

print(f"\nPython: {sys.version}")
print(f"Platform: {platform.platform()}")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("  Running in CPU mode")
except ImportError as e:
    print(f"\n✗ PyTorch not found: {e}")
    sys.exit(1)

# Check torchvision
try:
    import torchvision
    print(f"\n✓ torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"\n✗ torchvision not found: {e}")
    sys.exit(1)

# Check ultralytics
try:
    import ultralytics
    print(f"\n✓ ultralytics version: {ultralytics.__version__}")
    from ultralytics import YOLO
    print("  YOLO model class imported successfully")
except ImportError as e:
    print(f"\n✗ ultralytics not found: {e}")
    sys.exit(1)

# Check other key dependencies
deps = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'numpy': 'numpy',
    'yaml': 'PyYAML',
    'matplotlib': 'matplotlib',
    'tqdm': 'tqdm',
}

print("\nOther dependencies:")
for module, package in deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {package}: {version}")
    except ImportError:
        print(f"  ✗ {package}: not found")

print("\n" + "="*50)
print("Verification complete!")
print("="*50 + "\n")
VERIFY_SCRIPT

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All packages verified successfully${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi

# Create training configuration template if it doesn't exist
TRAINING_CONFIG="$PROJECT_ROOT/config/training_config.yaml"
if [ ! -f "$TRAINING_CONFIG" ]; then
    echo -e "${BLUE}Creating training configuration template...${NC}"
    mkdir -p "$PROJECT_ROOT/config"

    cat > "$TRAINING_CONFIG" << 'EOF'
# YOLO Training Configuration for Billiards Trainer
# Generated by setup_training.sh

# Model configuration
model:
  base_model: yolov8n.pt  # nano model for faster training (n/s/m/l/x available)
  img_size: 640

# Training parameters
training:
  epochs: 100
  batch: 16  # Adjust based on GPU memory
  patience: 50  # Early stopping patience
  workers: 8  # Data loader workers

  # Optimization
  optimizer: Adam
  lr0: 0.01  # Initial learning rate
  lrf: 0.01  # Final learning rate (lr0 * lrf)
  momentum: 0.937
  weight_decay: 0.0005

  # Augmentation
  hsv_h: 0.015  # HSV-Hue augmentation
  hsv_s: 0.7    # HSV-Saturation
  hsv_v: 0.4    # HSV-Value
  degrees: 0.0  # Rotation (+/- deg)
  translate: 0.1  # Translation (+/- fraction)
  scale: 0.5    # Scale (+/- gain)
  shear: 0.0    # Shear (+/- deg)
  perspective: 0.0  # Perspective
  flipud: 0.0   # Flip up-down
  fliplr: 0.5   # Flip left-right
  mosaic: 1.0   # Mosaic augmentation
  mixup: 0.0    # Mixup augmentation

# Hardware
device: 0  # GPU device (0, 1, 2, etc.) or 'cpu'

# Paths (relative to project root)
data:
  dataset: datasets/billiards/data.yaml
  output: runs/train

# Validation
val:
  split: 0.2  # Validation split
  conf: 0.25  # Confidence threshold
  iou: 0.7    # NMS IoU threshold
EOF

    echo -e "${GREEN}✓ Training configuration created at config/training_config.yaml${NC}"
else
    echo -e "${YELLOW}Training configuration already exists at config/training_config.yaml${NC}"
fi
echo ""

# Hardware recommendations
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Hardware Requirements & Recommendations${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Minimum requirements:"
echo "  - CPU: 4+ cores"
echo "  - RAM: 8GB+ (16GB recommended)"
echo "  - Storage: 10GB+ free space"
echo ""
echo "For GPU training:"
echo "  - NVIDIA GPU with 6GB+ VRAM (8GB+ recommended)"
echo "  - CUDA 11.8 or 12.1+"
echo "  - cuDNN 8.x+"
echo ""
echo "Training tips:"
echo "  - Use GPU for faster training (10-50x speedup)"
echo "  - Adjust batch size based on available VRAM"
echo "  - Start with a smaller model (yolov8n) for testing"
echo "  - Use data augmentation to improve generalization"
echo "  - Monitor training with tensorboard: tensorboard --logdir runs/train"
echo ""

if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "${GREEN}Your system appears ready for GPU training!${NC}"
else
    echo -e "${YELLOW}Your system will use CPU training (slower but functional)${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset using tools/dataset_creator.py"
echo "  2. Review config/training_config.yaml"
echo "  3. Start training with tools/train_yolo.py"
echo ""
echo "For help:"
echo "  python3 tools/train_yolo.py --help"
echo "  python3 tools/dataset_creator.py --help"
echo ""
