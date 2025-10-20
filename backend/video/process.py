"""Video Module Process - Standalone video capture process.

This module runs as a separate OS process to capture video frames and write them
to shared memory for consumption by the Vision Module and API streaming endpoints.

Architecture:
    - Runs as independent process (python -m backend.video)
    - Captures video frames using CameraCapture
    - Writes frames to shared memory using SharedMemoryFrameWriter
    - Provides clean shutdown with resource cleanup
    - Handles signals (SIGTERM, SIGINT) gracefully

Performance Targets:
    - 30 FPS sustained capture rate
    - <5ms write latency to shared memory
    - <10% CPU usage
    - Stable memory usage (~20MB)
"""

import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from backend.config import config
from backend.video.ipc.shared_memory import FrameFormat, SharedMemoryFrameWriter
from backend.vision.capture import CameraCapture

# Configure logging
logger = logging.getLogger(__name__)


class VideoProcess:
    """Main orchestrator for Video Module process.

    This class manages the video capture lifecycle:
    1. Initialize camera from config
    2. Initialize shared memory writer with actual frame dimensions
    3. Run main capture loop (capture -> write to shared memory)
    4. Handle graceful shutdown and cleanup

    Usage:
        process = VideoProcess(config)
        exit_code = process.start()  # Blocking call
    """

    def __init__(self, config_instance):
        """Initialize VideoProcess.

        Args:
            config_instance: Config singleton instance with loaded configuration
        """
        self.config = config_instance
        self.camera: Optional[CameraCapture] = None
        self.ipc_writer: Optional[SharedMemoryFrameWriter] = None
        self.shutdown_event = threading.Event()

        # Load configuration
        self._shm_name = self.config.get("video.shared_memory_name", "billiards_video")
        self._shutdown_timeout = self.config.get("video.process.shutdown_timeout", 10.0)
        self._main_loop_sleep = self.config.get("video.process.main_loop_sleep", 0.001)

        # Statistics
        self._start_time = 0.0
        self._frames_written = 0
        self._frames_captured = 0
        self._last_stats_time = 0.0
        self._stats_interval = 10.0  # Log stats every 10 seconds

        logger.info("VideoProcess initialized")

    def _initialize_camera(self) -> bool:
        """Initialize camera from config.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing camera...")

            # Determine device_id: video_file_path takes precedence over device_id
            # This aligns with the logic in vision/config_manager.py
            video_file_path = self.config.get("vision.camera.video_file_path")

            if video_file_path:
                # Use video file path as device_id
                device_id = video_file_path
                logger.info(f"Using video file: {video_file_path}")
            else:
                # Use device_id setting (can be int for camera, or string for stream/file)
                device_id = self.config.get("vision.camera.device_id", 0)
                logger.info(f"Using camera device: {device_id}")

            # Get camera configuration from config
            camera_config = {
                "device_id": device_id,
                "backend": self.config.get("vision.camera.backend", "auto"),
                "resolution": self.config.get("vision.camera.resolution", [1920, 1080]),
                "fps": self.config.get("vision.camera.fps", 30),
                "exposure_mode": self.config.get("vision.camera.exposure_mode", "auto"),
                "exposure_value": self.config.get("vision.camera.exposure_value"),
                "gain": self.config.get("vision.camera.gain", 1.0),
                "buffer_size": self.config.get("vision.camera.buffer_size", 2),
                "auto_reconnect": self.config.get("vision.camera.auto_reconnect", True),
                "reconnect_delay": self.config.get(
                    "vision.camera.reconnect_delay", 1.0
                ),
                "max_reconnect_attempts": self.config.get(
                    "vision.camera.max_reconnect_attempts", 5
                ),
                "read_timeout": self.config.get("vision.camera.read_timeout", 5.0),
                "stop_timeout": self.config.get("vision.camera.stop_timeout", 5.0),
                "frame_log_interval": self.config.get(
                    "vision.camera.frame_log_interval", 30
                ),
                "fps_tracker_size": self.config.get(
                    "vision.camera.fps_tracker_size", 10
                ),
                "frame_sleep_compensation": self.config.get(
                    "vision.camera.frame_sleep_compensation", 0.001
                ),
                "loop_video": self.config.get("vision.camera.loop_video", False),
                "video_start_frame": self.config.get(
                    "vision.camera.video_start_frame", 0
                ),
                "video_end_frame": self.config.get("vision.camera.video_end_frame"),
            }

            # Create camera capture instance
            self.camera = CameraCapture(camera_config)

            # Start camera capture
            if not self.camera.start_capture():
                logger.error("Failed to start camera capture")
                return False

            logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}", exc_info=True)
            return False

    def _initialize_ipc_writer(self, width: int, height: int) -> bool:
        """Initialize shared memory writer with actual frame dimensions.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Initializing shared memory writer: {width}x{height}...")

            # Create shared memory writer
            # Using BGR24 format as that's what OpenCV uses by default
            self.ipc_writer = SharedMemoryFrameWriter(
                name=self._shm_name,
                width=width,
                height=height,
                frame_format=FrameFormat.BGR24,
            )

            # Initialize the shared memory segment
            self.ipc_writer.initialize()

            logger.info(
                f"Shared memory writer initialized: {self._shm_name} "
                f"({self.ipc_writer.total_size / 1024 / 1024:.2f} MB)"
            )
            return True

        except Exception as e:
            logger.error(f"Shared memory initialization failed: {e}", exc_info=True)
            return False

    def _main_loop(self) -> None:
        """Main capture loop - capture frames and write to shared memory.

        This loop runs continuously until shutdown_event is set:
        1. Get latest frame from camera
        2. Write frame to shared memory
        3. Update statistics
        4. Sleep briefly to maintain frame rate
        """
        logger.info("Starting main capture loop")
        self._start_time = time.time()
        self._last_stats_time = self._start_time

        while not self.shutdown_event.is_set():
            try:
                # Get latest frame from camera
                frame_data = self.camera.get_latest_frame()

                if frame_data is None:
                    # No frame available yet - sleep and continue
                    time.sleep(self._main_loop_sleep)
                    continue

                frame, frame_info = frame_data
                self._frames_captured += 1

                # Write frame to shared memory
                self.ipc_writer.write_frame(frame, frame_info.frame_number)
                self._frames_written += 1

                # Log statistics periodically
                current_time = time.time()
                if current_time - self._last_stats_time >= self._stats_interval:
                    self._log_statistics(current_time)
                    self._last_stats_time = current_time

                # Brief sleep to prevent CPU spinning
                # (frame rate is controlled by camera capture thread)
                time.sleep(self._main_loop_sleep)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Don't exit on error - try to continue
                time.sleep(0.1)

        logger.info("Main capture loop ended")

    def _log_statistics(self, current_time: float) -> None:
        """Log capture and write statistics.

        Args:
            current_time: Current timestamp in seconds
        """
        uptime = current_time - self._start_time
        fps = self._frames_written / uptime if uptime > 0 else 0

        # Get camera health info
        camera_health = self.camera.get_health()

        # Get writer stats
        writer_stats = self.ipc_writer.get_stats() if self.ipc_writer else {}

        logger.info(
            f"Stats: uptime={uptime:.1f}s, "
            f"frames_captured={self._frames_captured}, "
            f"frames_written={self._frames_written}, "
            f"fps={fps:.1f}, "
            f"camera_fps={camera_health.fps:.1f}, "
            f"camera_dropped={camera_health.frames_dropped}, "
            f"write_counter={writer_stats.get('write_counter', 0)}"
        )

    def _cleanup(self) -> None:
        """Clean up resources (camera and shared memory)."""
        logger.info("Cleaning up resources...")

        # Log final statistics
        if self._start_time > 0:
            final_time = time.time()
            self._log_statistics(final_time)

        # Stop camera
        if self.camera:
            try:
                logger.info("Stopping camera...")
                self.camera.stop_capture()
                logger.info("Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
            finally:
                self.camera = None

        # Close shared memory (writer owns unlink)
        if self.ipc_writer:
            try:
                logger.info("Closing shared memory...")
                self.ipc_writer.cleanup()
                logger.info("Shared memory closed")
            except Exception as e:
                logger.error(f"Error closing shared memory: {e}")
            finally:
                self.ipc_writer = None

        logger.info("Cleanup complete")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals (SIGTERM, SIGINT).

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), initiating shutdown...")
        self.shutdown_event.set()

    def start(self) -> int:
        """Start the Video Module process.

        This is the main entry point that:
        1. Sets up signal handlers
        2. Initializes camera
        3. Gets first frame to determine dimensions
        4. Initializes shared memory writer
        5. Runs main capture loop
        6. Handles shutdown and cleanup

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        logger.info("Starting Video Module process...")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("Signal handlers registered (SIGTERM, SIGINT)")

        try:
            # Initialize camera
            if not self._initialize_camera():
                logger.error("Camera initialization failed")
                return 1

            # Wait for first frame to get actual dimensions
            logger.info("Waiting for first frame to determine dimensions...")
            max_wait_time = 10.0  # 10 seconds timeout
            wait_start = time.time()
            first_frame_data = None

            while (time.time() - wait_start) < max_wait_time:
                first_frame_data = self.camera.get_latest_frame()
                if first_frame_data is not None:
                    break
                time.sleep(0.1)

            if first_frame_data is None:
                logger.error(
                    f"Failed to get first frame within {max_wait_time}s timeout"
                )
                return 1

            frame, frame_info = first_frame_data
            height, width = frame.shape[:2]
            logger.info(
                f"First frame received: {width}x{height}, channels={frame.shape[2] if len(frame.shape) > 2 else 1}"
            )

            # Initialize shared memory writer with actual dimensions
            if not self._initialize_ipc_writer(width, height):
                logger.error("Shared memory initialization failed")
                return 1

            # Run main loop (blocking)
            logger.info("Video Module process started successfully")
            self._main_loop()

            # Normal shutdown
            logger.info("Video Module process shutting down normally")
            return 0

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            return 0

        except Exception as e:
            logger.error(f"Fatal error in Video Module process: {e}", exc_info=True)
            return 1

        finally:
            # Always clean up resources
            self._cleanup()
