"""Vision test configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


@pytest.fixture()
def image_shape():
    """Standard image shape for tests (height, width)."""
    return (1080, 1920)


@pytest.fixture()
def class_names():
    """YOLO class names list."""
    return [
        "cue",
        "solid_1",
        "solid_2",
        "solid_3",
        "solid_4",
        "solid_5",
        "solid_6",
        "solid_7",
        "eight",
        "stripe_9",
        "stripe_10",
        "stripe_11",
        "stripe_12",
        "stripe_13",
        "stripe_14",
        "stripe_15",
        "cue_stick",
    ]
