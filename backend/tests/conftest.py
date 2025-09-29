"""Shared test configuration and fixtures for the billiards trainer backend."""

import asyncio
import os

# Ensure backend is in path
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from config.manager import ConfigurationModule
from core.models import BallState, GameState, ShotAnalysis, TableState, Vector2D
from vision.models import Ball, BallType, DetectionResult

# from api.main import create_app  # Skip API for now due to circular imports


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "camera": {
            "device_id": 0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "exposure": -6,
            "brightness": 50,
            "contrast": 50,
            "saturation": 50,
        },
        "table": {
            "width": 2.84,  # meters
            "height": 1.42,
            "pocket_radius": 0.057,
            "rail_height": 0.035,
            "felt_color": [34, 139, 34],  # forest green
        },
        "balls": {
            "radius": 0.028575,  # meters
            "colors": {
                "cue": [255, 255, 255],
                "1": [255, 255, 0],  # yellow
                "2": [0, 0, 255],  # blue
                "3": [255, 0, 0],  # red
                "4": [128, 0, 128],  # purple
                "5": [255, 165, 0],  # orange
                "6": [0, 128, 0],  # green
                "7": [128, 0, 0],  # maroon
                "8": [0, 0, 0],  # black
                "9": [255, 255, 0],  # yellow stripe
                "10": [0, 0, 255],  # blue stripe
                "11": [255, 0, 0],  # red stripe
                "12": [128, 0, 128],  # purple stripe
                "13": [255, 165, 0],  # orange stripe
                "14": [0, 128, 0],  # green stripe
                "15": [128, 0, 0],  # maroon stripe
            },
        },
        "physics": {
            "friction": 0.15,
            "restitution": 0.9,
            "gravity": 9.81,
            "air_resistance": 0.001,
        },
        "projector": {
            "width": 1920,
            "height": 1080,
            "position": [0, 0, 2.0],  # x, y, z
            "rotation": [0, 0, 0],  # rx, ry, rz
            "brightness": 1.0,
            "contrast": 1.0,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8001,
            "debug": True,
            "cors_origins": ["*"],
            "websocket_ping_interval": 30,
            "websocket_ping_timeout": 10,
        },
    }


@pytest.fixture()
def config_module(mock_config, temp_dir):
    """Create a configuration module instance for testing."""
    config_file = temp_dir / "test_config.yaml"

    # Write config to file
    import yaml

    with open(config_file, "w") as f:
        yaml.dump(mock_config, f)

    # Create config module
    config_module = ConfigurationModule()
    config_module.load_from_file(str(config_file))
    return config_module


@pytest.fixture()
def mock_camera_frame():
    """Create a mock camera frame for vision tests."""
    # Create a realistic test image (green table with white ball)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Green background

    # Add a white ball at center
    center = (960, 540)
    cv2.circle(frame, center, 30, (255, 255, 255), -1)

    return {
        "frame": frame,
        "timestamp": time.time(),
        "frame_id": 1,
        "width": 1920,
        "height": 1080,
    }


@pytest.fixture()
def mock_detection_result():
    """Create a mock detection result for testing."""
    from vision.models import FrameStatistics

    stats = FrameStatistics(frame_number=1, timestamp=time.time(), processing_time=10.0)

    return DetectionResult(
        frame_number=1,
        timestamp=time.time(),
        balls=[
            Ball(position=(960, 540), radius=30, ball_type=BallType.CUE),
            Ball(position=(800, 400), radius=30, ball_type=BallType.SOLID, number=1),
            Ball(position=(1100, 600), radius=30, ball_type=BallType.EIGHT, number=8),
        ],
        cue=None,
        table=None,
        statistics=stats,
    )


@pytest.fixture()
def mock_game_state():
    """Create a mock game state for testing."""
    table = TableState.standard_9ft_table()

    balls = [
        BallState(
            id="cue", position=Vector2D(1.42, 0.71), radius=0.028575, is_cue_ball=True
        ),
        BallState(id="1", position=Vector2D(1.0, 0.5), radius=0.028575, number=1),
        BallState(id="8", position=Vector2D(2.0, 0.9), radius=0.028575, number=8),
    ]

    return GameState(
        timestamp=time.time(),
        frame_number=0,
        balls=balls,
        table=table,
        current_player=1,
    )


@pytest.fixture()
def mock_shot():
    """Create a mock shot for testing."""
    from core.models import ShotType

    return ShotAnalysis(
        shot_type=ShotType.DIRECT,
        difficulty=0.5,
        success_probability=0.8,
        recommended_force=0.8,
        recommended_angle=45.0,
        target_ball_id="8",
    )


@pytest.fixture()
def test_client():
    """Create a test client for API testing."""
    # Skip for now due to circular imports
    return None


@pytest.fixture()
def mock_cv2_camera():
    """Mock cv2.VideoCapture for camera tests."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # FPS
    mock_cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))

    with patch("cv2.VideoCapture", return_value=mock_cap):
        yield mock_cap


@pytest.fixture()
def mock_websocket():
    """Mock WebSocket for real-time communication tests."""
    mock_ws = AsyncMock()
    mock_ws.accept = AsyncMock()
    mock_ws.send_text = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value='{"type": "ping"}')
    mock_ws.receive_json = AsyncMock(return_value={"type": "ping"})
    mock_ws.close = AsyncMock()
    return mock_ws


@pytest.fixture()
def performance_timer():
    """Timer fixture for performance tests."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

        @property
        def elapsed_ms(self):
            elapsed = self.elapsed
            return elapsed * 1000 if elapsed is not None else None

    return Timer()


@pytest.fixture()
def mock_opengl_context():
    """Mock OpenGL context for projector tests."""
    with patch("moderngl.create_context") as mock_ctx:
        mock_context = MagicMock()
        mock_context.clear.return_value = None
        mock_context.finish.return_value = None
        mock_ctx.return_value = mock_context
        yield mock_context


# Performance test helpers
def assert_performance(timer, max_time_ms: float, operation: str):
    """Assert that an operation completed within the specified time."""
    if timer.elapsed_ms is None:
        pytest.fail(f"Timer not properly used for {operation}")

    assert timer.elapsed_ms <= max_time_ms, (
        f"{operation} took {timer.elapsed_ms:.2f}ms, " f"expected <= {max_time_ms}ms"
    )


def assert_fps(frame_count: int, total_time: float, min_fps: float):
    """Assert that frame processing meets minimum FPS requirement."""
    actual_fps = frame_count / total_time
    assert actual_fps >= min_fps, (
        f"Processing achieved {actual_fps:.2f} FPS, " f"expected >= {min_fps} FPS"
    )


# Memory monitoring
@pytest.fixture()
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil

    process = psutil.Process()

    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = None
            self.peak_memory = None

        def start(self):
            self.initial_memory = process.memory_info().rss
            self.peak_memory = self.initial_memory

        def update(self):
            current_memory = process.memory_info().rss
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory

        @property
        def memory_increase_mb(self):
            if self.initial_memory is None or self.peak_memory is None:
                return None
            return (self.peak_memory - self.initial_memory) / 1024 / 1024

    return MemoryMonitor()


# Skip markers for optional dependencies
opencv_available = pytest.mark.skipif(not cv2, reason="OpenCV not available")

opengl_available = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None and os.name != "nt",
    reason="No display available for OpenGL tests",
)
