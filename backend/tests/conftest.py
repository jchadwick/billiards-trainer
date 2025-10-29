"""Pytest configuration for backend tests.

This file sets up the Python path so that imports work correctly during testing.
"""

import sys
from pathlib import Path

# Add backend directory to Python path so imports work
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
