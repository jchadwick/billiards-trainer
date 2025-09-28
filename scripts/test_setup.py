#!/usr/bin/env python3
"""
Test script to verify the development environment setup.

This script checks that all required dependencies are installed and
the environment is properly configured for development.
"""

import os
import sys
from pathlib import Path


def print_header(message: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {message}")
    print(f"{'='*50}")


def print_status(message: str, success: bool) -> None:
    """Print a status message with color coding."""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {message}")


def check_python_version() -> bool:
    """Check if Python version is 3.9+."""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 9
    print_status(f"Python {version.major}.{version.minor}.{version.micro}", is_valid)
    return is_valid


def check_dependency(module_name: str, package_name: str = None) -> bool:
    """Check if a Python dependency is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print_status(f"{package_name} installed", True)
        return True
    except ImportError:
        print_status(f"{package_name} missing", False)
        return False


def check_dependencies() -> bool:
    """Check all required dependencies."""
    print_header("Checking Dependencies")

    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("jsonschema", "jsonschema"),
        ("scipy", "scipy"),
        ("OpenGL", "PyOpenGL"),
        ("pygame", "pygame"),
        ("moderngl", "moderngl"),
        ("websockets", "websockets"),
        ("jose", "python-jose"),
        ("passlib", "passlib"),
        ("PIL", "Pillow"),
        ("skimage", "scikit-image"),
        ("numba", "numba"),
        ("watchdog", "watchdog"),
    ]

    dev_dependencies = [
        ("pytest", "pytest"),
        ("black", "black"),
        ("isort", "isort"),
        ("mypy", "mypy"),
        ("ruff", "ruff"),
    ]

    # Check core dependencies
    core_success = all(check_dependency(mod, pkg) for mod, pkg in dependencies)

    print_header("Checking Development Dependencies")
    dev_success = all(check_dependency(mod, pkg) for mod, pkg in dev_dependencies)

    return core_success and dev_success


def check_environment_files() -> bool:
    """Check if environment files exist."""
    print_header("Checking Environment Files")

    files_to_check = [
        (".env", "Environment template"),
        ("pyproject.toml", "Project configuration"),
        ("config/logging.yaml", "Logging configuration"),
        ("Makefile", "Development commands"),
        (".gitignore", "Git ignore rules"),
        (".pre-commit-config.yaml", "Pre-commit hooks"),
    ]

    all_exist = True
    for file_path, description in files_to_check:
        exists = Path(file_path).exists()
        print_status(f"{description} ({file_path})", exists)
        if not exists:
            all_exist = False

    return all_exist


def check_directories() -> bool:
    """Check if required directories exist."""
    print_header("Checking Directories")

    directories = [
        "backend",
        "config",
        "scripts",
        "logs",
        "data",
        "temp",
    ]

    all_exist = True
    for directory in directories:
        exists = Path(directory).exists()
        print_status(f"{directory}/ directory", exists)
        if not exists:
            all_exist = False

    return all_exist


def check_logging_setup() -> bool:
    """Test logging configuration."""
    print_header("Testing Logging Setup")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))

        from backend.utils.logging import auto_setup_logging, get_logger

        # Setup logging
        auto_setup_logging()

        # Test logging
        logger = get_logger("test_setup")
        logger.info("Testing logging configuration")

        print_status("Logging configuration", True)
        return True

    except Exception as e:
        print_status(f"Logging configuration failed: {e}", False)
        return False


def check_import_backend() -> bool:
    """Test importing backend modules."""
    print_header("Testing Backend Imports")

    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))

        # Test importing backend utils
        from backend.utils import logging

        print_status("Backend utils import", True)
        return True

    except Exception as e:
        print_status(f"Backend import failed: {e}", False)
        return False


def main() -> None:
    """Run all setup checks."""
    print_header("Billiards Trainer Setup Verification")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment Files", check_environment_files),
        ("Directories", check_directories),
        ("Backend Imports", check_import_backend),
        ("Logging Setup", check_logging_setup),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_status(f"{name} check failed with error: {e}", False)
            results.append((name, False))

    # Summary
    print_header("Summary")
    all_passed = True
    for name, result in results:
        print_status(f"{name} check", result)
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Your development environment is ready.")
        print("\nNext steps:")
        print("  1. Activate virtual environment: source .venv/bin/activate")
        print("  2. Run the application: make run")
        print("  3. Run tests: make test")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed. Please review the output above.")
        print("\nTry running:")
        print("  make dev          # Setup development environment")
        print("  make install-dev  # Install development dependencies")
        sys.exit(1)


if __name__ == "__main__":
    main()
