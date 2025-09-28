#!/usr/bin/env python3
"""Basic test to verify the test suite setup works."""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test that basic imports work."""
    try:
        import pytest
        print("✅ pytest available")
    except ImportError as e:
        print(f"❌ pytest: {e}")
        return False

    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False

    try:
        import cv2
        print("✅ opencv available")
    except ImportError as e:
        print(f"❌ opencv: {e}")
        return False

    return True

def test_config_loading():
    """Test that configuration loading works."""
    try:
        from config.manager import ConfigurationModule
        config = ConfigurationModule()
        print("✅ Configuration manager works")
        return True
    except Exception as e:
        print(f"❌ Configuration: {e}")
        return False

def test_core_models():
    """Test that core models work."""
    try:
        from core.models import Vector2D, BallState

        # Test Vector2D
        v = Vector2D(1.0, 2.0)
        assert v.magnitude() > 0

        # Test BallState creation
        ball = BallState(
            id="test",
            position=Vector2D(1.0, 0.5)
        )
        assert ball.id == "test"

        print("✅ Core models work")
        return True
    except Exception as e:
        print(f"❌ Core models: {e}")
        return False

def test_pytest_execution():
    """Test that pytest can run a simple test."""
    try:
        import subprocess

        # Create a simple test file
        test_content = '''
import pytest

def test_simple():
    assert 1 + 1 == 2

def test_vector():
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))

    from core.models import Vector2D
    v = Vector2D(3.0, 4.0)
    assert abs(v.magnitude() - 5.0) < 0.001
'''

        test_file = backend_dir / "test_temp.py"
        with open(test_file, 'w') as f:
            f.write(test_content)

        # Run the test
        result = subprocess.run([
            sys.executable, "-m", "pytest", str(test_file), "-v"
        ], capture_output=True, text=True, cwd=backend_dir)

        # Clean up
        test_file.unlink()

        if result.returncode == 0:
            print("✅ pytest execution works")
            return True
        else:
            print(f"❌ pytest execution failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ pytest execution: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Testing billiards trainer test suite setup")
    print("=" * 50)

    tests = [
        test_imports,
        test_config_loading,
        test_core_models,
        test_pytest_execution
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")

    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed! Test suite is ready.")
        return True
    else:
        print("💥 Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
