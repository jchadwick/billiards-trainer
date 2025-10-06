"""Minimal test configuration for vision tests."""

import cv2
import numpy as np
import pytest


@pytest.fixture()
def sample_frame():
    """Create a sample BGR frame for testing."""
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Green background
    cv2.circle(frame, (320, 320), 20, (255, 255, 255), -1)  # White ball
    return frame
