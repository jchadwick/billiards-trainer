"""Comprehensive tests for camera capture functionality.

Tests camera interface requirements:
- FR-VIS-001: Initialize and configure camera device with specified parameters
- FR-VIS-002: Capture continuous video stream at configurable frame rate (15-60 FPS)
- FR-VIS-003: Support multiple camera backends (V4L2, DirectShow, GStreamer)
- FR-VIS-004: Handle camera disconnection and reconnection gracefully
- FR-VIS-005: Provide camera status and health monitoring
"""

import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.vision.capture import CameraCapture, CameraHealth, CameraStatus, FrameInfo


class TestCameraCapture:
    """Test suite for CameraCapture class."""

    @pytest.fixture()
    def basic_config(self):
        """Basic camera configuration for testing."""
        return {
            "device_id": 0,
            "backend": "auto",
            "resolution": (640, 480),
            "fps": 30,
            "exposure_mode": "auto",
            "gain": 1.0,
            "buffer_size": 1,
            "auto_reconnect": True,
            "reconnect_delay": 0.1,  # Faster for testing
            "max_reconnect_attempts": 3,
        }

    @pytest.fixture()
    def mock_video_capture(self):
        """Mock cv2.VideoCapture for testing."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = True
            mock_instance.read.return_value = (
                True,
                np.zeros((480, 640, 3), dtype=np.uint8),
            )
            mock_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_EXPOSURE: 0.5,
                cv2.CAP_PROP_GAIN: 1.0,
                cv2.CAP_PROP_BRIGHTNESS: 0.5,
                cv2.CAP_PROP_CONTRAST: 0.5,
                cv2.CAP_PROP_SATURATION: 0.5,
                cv2.CAP_PROP_BUFFERSIZE: 1,
            }.get(prop, 0)
            mock_instance.set.return_value = True
            mock_cap.return_value = mock_instance
            yield mock_instance

    def test_initialization(self, basic_config):
        """Test camera initialization with valid configuration."""
        capture = CameraCapture(basic_config)

        assert capture._device_id == 0
        assert capture._backend == "auto"
        assert capture._resolution == (640, 480)
        assert capture._fps == 30
        assert capture._status == CameraStatus.DISCONNECTED
        assert not capture.is_connected()

    def test_initialization_with_defaults(self):
        """Test camera initialization with minimal configuration."""
        capture = CameraCapture({})

        assert capture._device_id == 0
        assert capture._backend == "auto"
        assert capture._resolution == (1920, 1080)
        assert capture._fps == 30

    def test_backend_conversion(self, basic_config):
        """Test OpenCV backend conversion."""
        capture = CameraCapture(basic_config)

        # Test all supported backends
        backends = {
            "auto": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "gstreamer": cv2.CAP_GSTREAMER,
            "opencv": cv2.CAP_OPENCV_MJPEG,
        }

        for backend_name, expected_value in backends.items():
            capture._backend = backend_name
            assert capture._get_opencv_backend() == expected_value

    def test_start_capture_success(self, basic_config, mock_video_capture):
        """Test successful camera capture start."""
        capture = CameraCapture(basic_config)

        assert capture.start_capture()
        assert capture.get_status() == CameraStatus.CONNECTED
        assert capture.is_connected()

        # Wait a moment for capture thread to start
        time.sleep(0.1)

        # Verify frame capture
        frame_data = capture.get_frame()
        assert frame_data is not None
        frame, frame_info = frame_data
        assert isinstance(frame, np.ndarray)
        assert isinstance(frame_info, FrameInfo)

        capture.stop_capture()

    def test_start_capture_failure(self, basic_config):
        """Test camera capture start failure."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.return_value = False
            mock_cap.return_value = mock_instance

            capture = CameraCapture(basic_config)
            assert not capture.start_capture()
            assert capture.get_status() == CameraStatus.ERROR
            assert not capture.is_connected()

    def test_stop_capture(self, basic_config, mock_video_capture):
        """Test camera capture stop."""
        capture = CameraCapture(basic_config)

        # Start then stop
        capture.start_capture()
        time.sleep(0.1)
        capture.stop_capture()

        assert capture.get_status() == CameraStatus.DISCONNECTED
        assert not capture.is_connected()

    def test_frame_queue_overflow(self, basic_config, mock_video_capture):
        """Test frame queue behavior when full."""
        config = basic_config.copy()
        config["buffer_size"] = 2
        config["fps"] = 100  # High frame rate to fill queue quickly

        capture = CameraCapture(config)
        capture.start_capture()

        # Wait for queue to fill and overflow
        time.sleep(0.5)

        health = capture.get_health()
        assert health.frames_captured > 0
        # Should have some dropped frames due to overflow

        capture.stop_capture()

    def test_get_latest_frame(self, basic_config, mock_video_capture):
        """Test getting the most recent frame."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        time.sleep(0.2)  # Let some frames accumulate

        latest = capture.get_latest_frame()
        assert latest is not None

        frame, frame_info = latest
        assert isinstance(frame, np.ndarray)
        assert frame_info.frame_number > 0

        capture.stop_capture()

    def test_status_callback(self, basic_config, mock_video_capture):
        """Test status change callback functionality."""
        status_changes = []

        def status_callback(status):
            status_changes.append(status)

        capture = CameraCapture(basic_config)
        capture.set_status_callback(status_callback)

        capture.start_capture()
        time.sleep(0.1)
        capture.stop_capture()

        # Should have at least CONNECTING -> CONNECTED -> DISCONNECTED
        assert len(status_changes) >= 3
        assert CameraStatus.CONNECTING in status_changes
        assert CameraStatus.CONNECTED in status_changes
        assert CameraStatus.DISCONNECTED in status_changes

    def test_camera_info(self, basic_config, mock_video_capture):
        """Test camera information retrieval."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        info = capture.get_camera_info()
        assert "device_id" in info
        assert "backend" in info
        assert "resolution" in info
        assert "fps" in info

        capture.stop_capture()

    def test_health_monitoring(self, basic_config, mock_video_capture):
        """Test camera health monitoring."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        time.sleep(0.2)

        health = capture.get_health()
        assert isinstance(health, CameraHealth)
        assert health.status == CameraStatus.CONNECTED
        assert health.frames_captured > 0
        assert health.fps > 0
        assert health.uptime > 0
        assert health.connection_attempts >= 1

        capture.stop_capture()

    def test_config_update(self, basic_config, mock_video_capture):
        """Test runtime configuration updates."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        # Test FPS update
        new_config = {"fps": 15}
        assert capture.update_config(new_config)
        assert capture._fps == 15

        # Test exposure update
        new_config = {"exposure_mode": "manual", "exposure_value": 0.3}
        assert capture.update_config(new_config)
        assert capture._exposure_value == 0.3

        capture.stop_capture()

    def test_context_manager(self, basic_config, mock_video_capture):
        """Test context manager functionality."""
        with CameraCapture(basic_config) as capture:
            assert capture.is_connected()
            time.sleep(0.1)

            frame_data = capture.get_frame()
            assert frame_data is not None

        # Should be disconnected after context exit
        assert not capture.is_connected()

    def test_reconnection_on_failure(self, basic_config):
        """Test automatic reconnection on camera failure."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()

            # First connection succeeds
            mock_instance.isOpened.side_effect = [
                True,
                True,
                False,
                True,
            ]  # Fail then recover
            mock_instance.read.side_effect = [
                (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Success
                (False, None),  # Failure
                (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Recovery
            ]
            mock_cap.return_value = mock_instance

            capture = CameraCapture(basic_config)
            capture.start_capture()

            # Wait for reconnection attempt
            time.sleep(0.5)

            health = capture.get_health()
            assert health.connection_attempts > 1
            assert health.error_count > 0

            capture.stop_capture()

    def test_no_reconnection_when_disabled(self, basic_config):
        """Test no reconnection when auto_reconnect is disabled."""
        config = basic_config.copy()
        config["auto_reconnect"] = False

        with patch("cv2.VideoCapture") as mock_cap:
            mock_instance = MagicMock()
            mock_instance.isOpened.side_effect = [True, False]  # Connect then fail
            mock_instance.read.return_value = (False, None)  # Immediate failure
            mock_cap.return_value = mock_instance

            capture = CameraCapture(config)
            capture.start_capture()

            # Wait for failure to be detected
            time.sleep(0.2)

            # Should go to error state and stay there
            assert capture.get_status() == CameraStatus.ERROR

            capture.stop_capture()

    def test_frame_rate_control(self, basic_config, mock_video_capture):
        """Test frame rate limiting functionality."""
        config = basic_config.copy()
        config["fps"] = 10  # Low frame rate for testing

        capture = CameraCapture(config)
        capture.start_capture()

        start_time = time.time()
        time.sleep(1.0)  # Wait 1 second

        health = capture.get_health()
        elapsed = time.time() - start_time

        # Should be approximately 10 FPS (allow some tolerance)
        expected_frames = elapsed * 10
        tolerance = expected_frames * 0.3  # 30% tolerance

        assert abs(health.frames_captured - expected_frames) < tolerance

        capture.stop_capture()

    def test_error_handling_in_callback(self, basic_config, mock_video_capture):
        """Test error handling when status callback raises exception."""

        def failing_callback(status):
            raise ValueError("Callback error")

        capture = CameraCapture(basic_config)
        capture.set_status_callback(failing_callback)

        # Should not crash even with failing callback
        capture.start_capture()
        time.sleep(0.1)
        capture.stop_capture()

    def test_thread_safety(self, basic_config, mock_video_capture):
        """Test thread safety of camera operations."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        results = []

        def get_frames():
            for _ in range(10):
                frame_data = capture.get_frame()
                results.append(frame_data is not None)
                time.sleep(0.01)

        # Start multiple threads accessing camera
        threads = [threading.Thread(target=get_frames) for _ in range(3)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        capture.stop_capture()

        # Should have successfully retrieved some frames from all threads
        assert any(results)

    def test_camera_configuration_validation(self, basic_config, mock_video_capture):
        """Test camera configuration parameter validation."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        # Verify configuration was applied
        info = capture.get_camera_info()
        assert info["resolution"] == (640, 480)
        assert info["fps"] == 30

        capture.stop_capture()

    def test_exposure_modes(self, basic_config, mock_video_capture):
        """Test different exposure modes."""
        # Test auto exposure
        config_auto = basic_config.copy()
        config_auto["exposure_mode"] = "auto"

        capture_auto = CameraCapture(config_auto)
        capture_auto.start_capture()
        capture_auto.stop_capture()

        # Test manual exposure
        config_manual = basic_config.copy()
        config_manual["exposure_mode"] = "manual"
        config_manual["exposure_value"] = 0.5

        capture_manual = CameraCapture(config_manual)
        capture_manual.start_capture()
        capture_manual.stop_capture()

    def test_backend_fallback(self, basic_config):
        """Test backend fallback when specific backend fails."""
        config = basic_config.copy()
        config["backend"] = "nonexistent"

        capture = CameraCapture(config)
        backend_id = capture._get_opencv_backend()

        # Should fall back to CAP_ANY for unknown backends
        assert backend_id == cv2.CAP_ANY

    def test_destructor_cleanup(self, basic_config, mock_video_capture):
        """Test that destructor properly cleans up resources."""
        capture = CameraCapture(basic_config)
        capture.start_capture()

        # Delete the capture object
        del capture

        # Should not hang or cause issues


class TestFrameInfo:
    """Test FrameInfo data structure."""

    def test_frame_info_creation(self):
        """Test FrameInfo creation and attributes."""
        timestamp = time.time()
        frame_info = FrameInfo(
            frame_number=42, timestamp=timestamp, size=(640, 480), channels=3
        )

        assert frame_info.frame_number == 42
        assert frame_info.timestamp == timestamp
        assert frame_info.size == (640, 480)
        assert frame_info.channels == 3


class TestCameraHealth:
    """Test CameraHealth data structure."""

    def test_camera_health_creation(self):
        """Test CameraHealth creation and attributes."""
        health = CameraHealth(
            status=CameraStatus.CONNECTED,
            frames_captured=100,
            frames_dropped=5,
            fps=29.5,
            last_frame_time=time.time(),
            error_count=2,
            last_error="Test error",
            connection_attempts=1,
            uptime=10.5,
        )

        assert health.status == CameraStatus.CONNECTED
        assert health.frames_captured == 100
        assert health.frames_dropped == 5
        assert health.fps == 29.5
        assert health.error_count == 2
        assert health.last_error == "Test error"
        assert health.connection_attempts == 1
        assert health.uptime == 10.5


# Integration test markers
@pytest.mark.integration()
class TestCameraIntegration:
    """Integration tests requiring actual camera hardware."""

    def test_real_camera_capture(self):
        """Test with real camera if available."""
        config = {
            "device_id": 0,
            "backend": "auto",
            "resolution": (640, 480),
            "fps": 15,
            "buffer_size": 1,
        }

        capture = CameraCapture(config)

        # Try to start capture - skip if no camera available
        if not capture.start_capture():
            pytest.skip("No camera device available")

        try:
            # Wait for a few frames
            time.sleep(1.0)

            # Verify we're getting frames
            frame_data = capture.get_latest_frame()
            assert frame_data is not None

            frame, frame_info = frame_data
            assert frame.shape[:2] == (480, 640)  # Height x Width
            assert frame_info.frame_number > 0

            # Check health
            health = capture.get_health()
            assert health.status == CameraStatus.CONNECTED
            assert health.frames_captured > 0
            assert health.fps > 0

        finally:
            capture.stop_capture()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
