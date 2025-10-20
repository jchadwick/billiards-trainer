"""VideoConsumer - Simplified wrapper around SharedMemoryFrameReader for Vision Module.

This module provides a clean interface for the Vision Module to consume video frames
from the Video Module via shared memory IPC.
"""

import logging
from typing import Optional

import numpy as np

from ...video.ipc.shared_memory import SharedMemoryFrameReader

logger = logging.getLogger(__name__)


class VideoModuleNotAvailableError(Exception):
    """Raised when Video Module is not available or shared memory cannot be attached."""

    pass


class VideoConsumer:
    """Wraps SharedMemoryFrameReader with simplified interface and error handling.

    This class provides a clean interface for the Vision Module to consume frames
    from shared memory. It handles connection errors gracefully and provides clear
    error messages if the Video Module is not running.

    Usage:
        consumer = VideoConsumer()
        try:
            consumer.start()
            while running:
                frame = consumer.get_frame()
                if frame is not None:
                    process_frame(frame)
                time.sleep(0.001)
        finally:
            consumer.stop()
    """

    def __init__(self):
        """Initialize VideoConsumer.

        Configuration is loaded from the config system automatically.
        """
        self.reader: Optional[SharedMemoryFrameReader] = None
        self.is_running = False

        # Load configuration
        from ...config import config

        self._shm_name = config.get("video.shared_memory_name", "billiards_video")
        self._attach_timeout = config.get("video.shared_memory_attach_timeout_sec", 5.0)

        logger.info(
            f"VideoConsumer initialized: shm_name={self._shm_name}, "
            f"attach_timeout={self._attach_timeout}s"
        )

    def start(self) -> None:
        """Attach to Video Module's shared memory.

        This will wait for the shared memory segment to be created by the Video Module.

        Raises:
            VideoModuleNotAvailableError: If Video Module is not running or shared
                memory cannot be attached within the timeout period.
        """
        logger.info("VideoConsumer: Attaching to shared memory...")

        try:
            # Create reader and attach to shared memory
            self.reader = SharedMemoryFrameReader(name=self._shm_name)
            self.reader.attach(timeout=self._attach_timeout)

            self.is_running = True

            logger.info(
                f"VideoConsumer: Successfully attached to shared memory "
                f"({self.reader.width}x{self.reader.height} {self.reader.frame_format.name})"
            )

        except TimeoutError as e:
            error_msg = (
                f"Video Module not available. Is it running?\n"
                f"Start with: python -m backend.video\n"
                f"Shared memory segment '{self._shm_name}' not found within {self._attach_timeout}s timeout."
            )
            logger.error(error_msg)
            raise VideoModuleNotAvailableError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"Failed to attach to shared memory '{self._shm_name}': {e}\n"
                f"Ensure Video Module is running: python -m backend.video"
            )
            logger.error(error_msg)
            raise VideoModuleNotAvailableError(error_msg) from e

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking frame read. Returns None if no new frame available.

        This method reads the latest frame from shared memory without blocking.
        If no new frame has been written since the last read, it returns None.

        Returns:
            Frame data as numpy array (height, width, channels) or None if no
            new frame is available.

        Raises:
            RuntimeError: If consumer is not running (call start() first).
        """
        if not self.is_running:
            raise RuntimeError(
                "VideoConsumer not running. Call start() before get_frame()"
            )

        try:
            frame, metadata = self.reader.read_frame()

            if frame is not None:
                logger.debug(
                    f"VideoConsumer: Read frame #{metadata.frame_number}, "
                    f"timestamp={metadata.timestamp:.3f}s"
                )

            return frame

        except Exception as e:
            logger.error(f"VideoConsumer: Error reading frame: {e}")
            # Don't raise - just return None to allow graceful degradation
            return None

    def stop(self) -> None:
        """Detach from shared memory.

        This should be called during shutdown to properly release resources.
        """
        logger.info("VideoConsumer: Stopping...")

        if self.reader:
            try:
                self.reader.detach()
                logger.info("VideoConsumer: Detached from shared memory")
            except Exception as e:
                logger.error(f"VideoConsumer: Error detaching from shared memory: {e}")
            finally:
                self.reader = None

        self.is_running = False
        logger.info("VideoConsumer: Stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
