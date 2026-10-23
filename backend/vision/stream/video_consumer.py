"""Legacy Python shared-memory consumer retained for compatibility tests.

It does not consume frames from the supported Rust camera pipeline.
"""

import logging
from types import TracebackType
from typing import Optional

import numpy as np

from backend.video.ipc.shared_memory import FrameMetadata, SharedMemoryFrameReader

logger = logging.getLogger(__name__)


class VideoModuleNotAvailableError(Exception):
    """Raised when a legacy Python shared-memory source cannot be attached."""

    pass


class VideoConsumer:
    """Wraps SharedMemoryFrameReader with simplified interface and error handling.

    This compatibility class reads the old Python-only shared-memory protocol.
    The Rust runtime does not publish that protocol, so starting the supported
    application cannot satisfy this consumer.

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

    def __init__(self) -> None:
        """Initialize VideoConsumer.

        Configuration is loaded from the config system automatically.
        """
        self.reader: Optional[SharedMemoryFrameReader] = None
        self.is_running = False

        # Load configuration
        from backend.config import config

        self._shm_name = config.get("video.shared_memory_name", "billiards_video")
        self._attach_timeout = config.get("video.shared_memory_attach_timeout_sec", 5.0)

        logger.info(
            f"VideoConsumer initialized: shm_name={self._shm_name}, "
            f"attach_timeout={self._attach_timeout}s"
        )

    def start(self) -> None:
        """Attach to Video Module's shared memory.

        Wait for a legacy Python producer to create its shared-memory segment.

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
                f"({self.reader.width}x{self.reader.height} "
                f"{self.reader.frame_format.name if self.reader.frame_format else 'unknown'})"
            )

        except TimeoutError as e:
            error_msg = (
                "Legacy Python shared-memory source is unavailable; the Rust "
                "runtime does not publish this protocol. "
                f"Shared memory segment '{self._shm_name}' not found within "
                f"{self._attach_timeout}s timeout. See backend/video/README.md."
            )
            logger.error(error_msg)
            raise VideoModuleNotAvailableError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"Failed to attach to legacy Python shared memory "
                f"'{self._shm_name}': {e}. The Rust runtime does not publish "
                "this protocol; see backend/video/README.md."
            )
            logger.error(error_msg)
            raise VideoModuleNotAvailableError(error_msg) from e

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking frame read. Returns None if no new frame is available.

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

        frame, _metadata = self.get_frame_with_metadata()
        return frame

    def get_frame_with_metadata(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Read the latest frame and its shared-memory publication metadata.

        ``FrameMetadata.timestamp`` is written by the video publisher when it
        publishes the frame. It is not a camera exposure timestamp. The old
        :meth:`get_frame` method remains available for callers that only need
        image pixels.

        Returns:
            A ``(frame, metadata)`` pair, or ``(None, None)`` when there is no
            new frame or the shared-memory read fails.

        Raises:
            RuntimeError: If the consumer is not running.
        """
        if not self.is_running:
            raise RuntimeError(
                "VideoConsumer not running. Call start() before get_frame_with_metadata()"
            )

        try:
            if self.reader is None:
                raise RuntimeError(
                    "VideoConsumer is running without an attached reader"
                )
            frame, metadata = self.reader.read_frame()

            if frame is not None:
                if metadata is not None:
                    logger.debug(
                        f"VideoConsumer: Read frame #{metadata.frame_number}, "
                        f"publication_timestamp={metadata.timestamp:.3f}s"
                    )
                else:
                    logger.debug("VideoConsumer: Read frame without metadata")

            return frame, metadata

        except Exception as e:
            logger.error(f"VideoConsumer: Error reading frame: {e}")
            # Don't raise - just return None to allow graceful degradation
            return None, None

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

    def __enter__(self) -> "VideoConsumer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.stop()
