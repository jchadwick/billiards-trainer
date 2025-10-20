"""Shared Memory IPC for low-latency frame delivery to Vision Module.

This module implements a triple-buffered shared memory system for delivering
video frames from the Video Module to the Vision Module with <50ms latency.

Architecture:
    - Triple buffering to allow simultaneous read/write without blocking
    - Lock-free read operation for Vision Module
    - Minimal write-side locking in Video Module
    - Frame metadata includes timestamp, frame number, and resolution
    - Supports multiple readers (Vision + API if needed)

Memory Layout:
    The shared memory region is structured as:

    [Header Block (4KB)]
    [Frame Buffer 0 (configurable size)]
    [Frame Buffer 1 (configurable size)]
    [Frame Buffer 2 (configurable size)]

    Header Block Layout (256 bytes, rest is padding):
    - Magic number (8 bytes): 0x424954414C4C5344 ("BITALLSD" - Billiards Trainer)
    - Version (4 bytes): Protocol version number
    - Buffer count (4 bytes): Number of frame buffers (always 3)
    - Frame width (4 bytes): Frame width in pixels
    - Frame height (4 bytes): Frame height in pixels
    - Frame format (4 bytes): Format code (1=RGB24, 2=BGR24, 3=GRAY8, etc.)
    - Bytes per frame (8 bytes): Size of each frame buffer in bytes
    - Current write index (4 bytes): Index of buffer being written (0-2)
    - Current read index (4 bytes): Index of buffer ready to read (0-2)
    - Write counter (8 bytes): Total frames written (monotonic)
    - Frame number (8 bytes): Current frame number
    - Timestamp seconds (8 bytes): Unix timestamp seconds
    - Timestamp nanoseconds (8 bytes): Timestamp nanoseconds
    - Writer pid (4 bytes): Process ID of writer
    - Reader count (4 bytes): Number of active readers
    - Reserved (164 bytes): For future use

Performance Targets:
    - Latency: <50ms from capture to Vision access
    - Throughput: 30 FPS sustained
    - Memory: ~20MB for 1920x1080 RGB24 triple buffer
    - CPU: Minimal (zero-copy when possible)
"""

import logging
import mmap
import os
import struct
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Protocol constants
MAGIC_NUMBER = 0x424954414C4C5344  # "BITALLSD" in hex
PROTOCOL_VERSION = 1
HEADER_SIZE = 4096  # 4KB header block (256 bytes used, rest is padding)
BUFFER_COUNT = 3  # Triple buffering


def _sanitize_name(name: str) -> str:
    """Sanitize shared memory name for filesystem use."""
    return name.replace("/", "_")


_FALLBACK_DIR = Path(tempfile.gettempdir()) / "billiards_shm"


def _fallback_path(name: str) -> Path:
    """Compute filesystem-backed shared memory path."""
    return _FALLBACK_DIR / f"{_sanitize_name(name)}.bin"


def _remove_fallback_file(name: str) -> None:
    """Remove filesystem-backed shared memory artifact if it exists."""
    try:
        _fallback_path(name).unlink()
    except FileNotFoundError:
        pass


class FileBackedSharedMemory:
    """Fallback shared memory implementation using mmap'd files.

    Some environments (including macOS sandboxed shells) disallow shm_open.
    In that case we transparently back the shared segment with a temporary
    file so higher layers continue to operate.
    """

    def __init__(self, name: str, size: Optional[int], create: bool):
        self.name = name
        self.path = _fallback_path(name)
        self._file = None
        self._mmap = None
        self._size = size

        if create:
            if size is None:
                raise ValueError("FileBackedSharedMemory requires size when creating")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file_obj = open(self.path, "w+b")  # noqa: SIM115
            file_obj.truncate(size)
        else:
            file_obj = open(self.path, "r+b")  # noqa: SIM115
            if size is None:
                file_obj.seek(0, os.SEEK_END)
                self._size = file_obj.tell()
                file_obj.seek(0)

        if self._size is None:
            raise ValueError("Shared memory size could not be determined")

        self._file = file_obj
        self._mmap = mmap.mmap(self._file.fileno(), self._size)

    @property
    def buf(self) -> memoryview:
        return memoryview(self._mmap)

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.flush()
            except Exception:
                pass
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def unlink(self) -> None:
        self.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


from typing import Optional, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Protocol constants
MAGIC_NUMBER = 0x424954414C4C5344  # "BITALLSD" in hex
PROTOCOL_VERSION = 1
HEADER_SIZE = 4096  # 4KB header block (256 bytes used, rest is padding)
BUFFER_COUNT = 3  # Triple buffering


class FrameFormat(IntEnum):
    """Video frame format codes."""

    RGB24 = 1  # 24-bit RGB
    BGR24 = 2  # 24-bit BGR (OpenCV default)
    GRAY8 = 3  # 8-bit grayscale
    RGBA32 = 4  # 32-bit RGBA
    BGRA32 = 5  # 32-bit BGRA


@dataclass
class FrameMetadata:
    """Metadata for a video frame."""

    frame_number: int
    timestamp_sec: int
    timestamp_nsec: int
    width: int
    height: int
    format: FrameFormat

    @property
    def timestamp(self) -> float:
        """Get timestamp as float seconds."""
        return self.timestamp_sec + self.timestamp_nsec / 1e9


class SharedMemoryHeader:
    """Shared memory header structure.

    This class provides methods to read/write the header block at the
    start of the shared memory region. The header uses native byte order
    and alignment.
    """

    # Header format string for struct module
    # See memory layout documentation at top of file
    HEADER_FORMAT = (
        "Q"  # Magic number (8 bytes)
        "I"  # Version (4 bytes)
        "I"  # Buffer count (4 bytes)
        "I"  # Frame width (4 bytes)
        "I"  # Frame height (4 bytes)
        "I"  # Frame format (4 bytes)
        "Q"  # Bytes per frame (8 bytes)
        "I"  # Current write index (4 bytes)
        "I"  # Current read index (4 bytes)
        "Q"  # Write counter (8 bytes)
        "Q"  # Frame number (8 bytes)
        "Q"  # Timestamp seconds (8 bytes)
        "Q"  # Timestamp nanoseconds (8 bytes)
        "I"  # Writer pid (4 bytes)
        "I"  # Reader count (4 bytes)
        "164x"  # Reserved (164 bytes padding to 256 bytes)
    )

    HEADER_STRUCT = struct.Struct(HEADER_FORMAT)

    def __init__(self, memory_view: memoryview):
        """Initialize header wrapper.

        Args:
            memory_view: Memory view of the header region (first 4KB).
        """
        self.memory_view = memory_view

    def initialize(
        self, width: int, height: int, frame_format: FrameFormat, bytes_per_frame: int
    ) -> None:
        """Initialize the header with frame configuration.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            frame_format: Frame format code.
            bytes_per_frame: Size of each frame buffer in bytes.
        """
        header_data = self.HEADER_STRUCT.pack(
            MAGIC_NUMBER,
            PROTOCOL_VERSION,
            BUFFER_COUNT,
            width,
            height,
            int(frame_format),
            bytes_per_frame,
            0,  # write_index
            0,  # read_index
            0,  # write_counter
            0,  # frame_number
            0,  # timestamp_sec
            0,  # timestamp_nsec
            os.getpid(),  # writer_pid
            0,  # reader_count
        )
        self.memory_view[: len(header_data)] = header_data

    def validate(self) -> bool:
        """Validate header magic number and version.

        Returns:
            True if header is valid, False otherwise.
        """
        try:
            data = self.HEADER_STRUCT.unpack(
                self.memory_view[: self.HEADER_STRUCT.size]
            )
            magic = data[0]
            version = data[1]
            return magic == MAGIC_NUMBER and version == PROTOCOL_VERSION
        except struct.error:
            return False

    def read_config(self) -> tuple[int, int, FrameFormat, int]:
        """Read frame configuration from header.

        Returns:
            Tuple of (width, height, format, bytes_per_frame).
        """
        data = self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        return (data[3], data[4], FrameFormat(data[5]), data[6])

    def update_frame_metadata(self, metadata: FrameMetadata, buffer_index: int) -> None:
        """Update frame metadata and switch buffers atomically.

        This updates the write counter, frame number, timestamp, and then
        atomically switches the read index to the newly written buffer.

        Args:
            metadata: Frame metadata to write.
            buffer_index: Index of buffer that was written (0-2).
        """
        # Read current header
        data = list(
            self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        )

        # Update metadata fields
        data[7] = buffer_index  # write_index
        data[8] = buffer_index  # read_index (atomic switch)
        data[9] = data[9] + 1  # write_counter (increment)
        data[10] = metadata.frame_number
        data[11] = metadata.timestamp_sec
        data[12] = metadata.timestamp_nsec

        # Pack and write back
        header_data = self.HEADER_STRUCT.pack(*data)
        self.memory_view[: len(header_data)] = header_data

    def read_metadata(self) -> FrameMetadata:
        """Read current frame metadata.

        Returns:
            Current frame metadata.
        """
        data = self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        return FrameMetadata(
            frame_number=data[10],
            timestamp_sec=data[11],
            timestamp_nsec=data[12],
            width=data[3],
            height=data[4],
            format=FrameFormat(data[5]),
        )

    def get_read_index(self) -> int:
        """Get current read index (buffer ready to read).

        Returns:
            Buffer index (0-2).
        """
        data = self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        return data[8]

    def get_write_counter(self) -> int:
        """Get total write counter (monotonic frame count).

        Returns:
            Total number of frames written.
        """
        data = self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        return data[9]

    def increment_reader_count(self) -> None:
        """Increment reader count (called when reader attaches)."""
        data = list(
            self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        )
        data[14] = data[14] + 1  # reader_count
        header_data = self.HEADER_STRUCT.pack(*data)
        self.memory_view[: len(header_data)] = header_data

    def decrement_reader_count(self) -> None:
        """Decrement reader count (called when reader detaches)."""
        data = list(
            self.HEADER_STRUCT.unpack(self.memory_view[: self.HEADER_STRUCT.size])
        )
        data[14] = max(0, data[14] - 1)  # reader_count (don't go negative)
        header_data = self.HEADER_STRUCT.pack(*data)
        self.memory_view[: len(header_data)] = header_data


class SharedMemoryFrameWriter:
    """Writer-side shared memory frame publisher.

    This class is used by the Video Module to publish frames to shared memory
    for consumption by the Vision Module. It implements triple buffering to
    allow lock-free reading while writing new frames.

    Usage:
        writer = SharedMemoryFrameWriter(
            name="billiards_video",
            width=1920,
            height=1080,
            frame_format=FrameFormat.BGR24
        )

        try:
            writer.initialize()

            for frame_num, frame in enumerate(video_frames):
                writer.write_frame(frame, frame_num)
        finally:
            writer.cleanup()
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        frame_format: FrameFormat = FrameFormat.BGR24,
    ):
        """Initialize frame writer.

        Args:
            name: Shared memory segment name (without /dev/shm/ prefix).
            width: Frame width in pixels.
            height: Frame height in pixels.
            frame_format: Frame format (default BGR24 for OpenCV).
        """
        self.name = name
        self.width = width
        self.height = height
        self.frame_format = frame_format

        # Calculate frame buffer size
        if frame_format in (FrameFormat.RGB24, FrameFormat.BGR24):
            self.bytes_per_pixel = 3
        elif frame_format == FrameFormat.GRAY8:
            self.bytes_per_pixel = 1
        elif frame_format in (FrameFormat.RGBA32, FrameFormat.BGRA32):
            self.bytes_per_pixel = 4
        else:
            raise ValueError(f"Unsupported frame format: {frame_format}")

        self.bytes_per_frame = width * height * self.bytes_per_pixel
        self.total_size = HEADER_SIZE + (self.bytes_per_frame * BUFFER_COUNT)

        # State
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.header: Optional[SharedMemoryHeader] = None
        self.current_buffer_index = 0
        self.frame_counter = 0
        self.lock = threading.Lock()
        self._using_fallback = False

        logger.info(
            f"Initializing SharedMemoryFrameWriter: {name}, "
            f"{width}x{height} {frame_format.name}, "
            f"{self.bytes_per_frame} bytes/frame, "
            f"{self.total_size / 1024 / 1024:.2f} MB total"
        )

    def initialize(self) -> None:
        """Create and initialize the shared memory segment.

        Raises:
            OSError: If shared memory creation fails.
        """
        # Try to unlink any existing segment first
        try:
            existing_shm = shared_memory.SharedMemory(name=self.name)
        except FileNotFoundError:
            existing_shm = None
        except PermissionError:
            existing_shm = None
            logger.warning(
                "Unable to inspect existing shared memory '%s' due to permission error; "
                "assuming it does not exist.",
                self.name,
            )
        else:
            try:
                existing_shm.close()
                existing_shm.unlink()
            finally:
                existing_shm = None
            logger.warning(f"Removed existing shared memory segment: {self.name}")

        # Create shared memory segment (with fallback if permissions block us)
        try:
            self.shm = shared_memory.SharedMemory(
                create=True, size=self.total_size, name=self.name
            )
            self._using_fallback = False
        except PermissionError as exc:
            logger.warning(
                "Failed to create POSIX shared memory '%s' (%s). "
                "Falling back to mmap file in %s.",
                self.name,
                exc,
                _FALLBACK_DIR,
            )
            self.shm = FileBackedSharedMemory(self.name, self.total_size, create=True)
            self._using_fallback = True

        # Initialize header
        header_view = memoryview(self.shm.buf)[0:HEADER_SIZE]
        self.header = SharedMemoryHeader(header_view)
        self.header.initialize(
            self.width, self.height, self.frame_format, self.bytes_per_frame
        )

        logger.info(f"Shared memory initialized: {self.name} ({self.total_size} bytes)")

    def write_frame(
        self, frame: np.ndarray, frame_number: Optional[int] = None
    ) -> None:
        """Write a frame to shared memory.

        This method implements triple buffering by cycling through three frame
        buffers. The write is lock-free for readers - they can continue reading
        the current buffer while we write to the next one.

        Args:
            frame: Frame data as numpy array. Shape should be (height, width, channels)
                   for color frames or (height, width) for grayscale.
            frame_number: Optional explicit frame number. If None, uses internal counter.

        Raises:
            ValueError: If frame dimensions don't match configuration.
            RuntimeError: If writer not initialized.
        """
        if self.shm is None or self.header is None:
            raise RuntimeError("Writer not initialized. Call initialize() first.")

        # Validate frame dimensions
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame dimensions {frame.shape} don't match "
                f"configured {self.height}x{self.width}"
            )

        # Ensure frame is contiguous and correct type
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        # Get timestamp
        timestamp = time.time()
        timestamp_sec = int(timestamp)
        timestamp_nsec = int((timestamp - timestamp_sec) * 1e9)

        with self.lock:
            # Choose next buffer (triple buffering)
            buffer_index = (self.current_buffer_index + 1) % BUFFER_COUNT

            # Calculate buffer offset
            buffer_offset = HEADER_SIZE + (buffer_index * self.bytes_per_frame)

            # Write frame data to buffer
            frame_view = memoryview(self.shm.buf)[
                buffer_offset : buffer_offset + self.bytes_per_frame
            ]
            frame_bytes = frame.tobytes()
            frame_view[: len(frame_bytes)] = frame_bytes

            # Update metadata and switch buffers atomically
            metadata = FrameMetadata(
                frame_number=(
                    frame_number if frame_number is not None else self.frame_counter
                ),
                timestamp_sec=timestamp_sec,
                timestamp_nsec=timestamp_nsec,
                width=self.width,
                height=self.height,
                format=self.frame_format,
            )
            self.header.update_frame_metadata(metadata, buffer_index)

            # Update state
            self.current_buffer_index = buffer_index
            self.frame_counter += 1

    def get_stats(self) -> dict:
        """Get writer statistics.

        Returns:
            Dictionary with statistics (frames written, current buffer, etc.).
        """
        if self.header is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "frames_written": self.frame_counter,
            "current_buffer_index": self.current_buffer_index,
            "write_counter": self.header.get_write_counter(),
            "total_size_mb": self.total_size / 1024 / 1024,
        }

    def cleanup(self) -> None:
        """Clean up shared memory resources.

        This should be called during shutdown to properly release resources.
        """
        logger.info("Cleaning up SharedMemoryFrameWriter")

        # Close and unlink shared memory
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info(f"Removed shared memory segment: {self.name}")
            except Exception as e:
                logger.error(f"Failed to remove shared memory segment: {e}")
            finally:
                self.shm = None

        self.header = None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class SharedMemoryFrameReader:
    """Reader-side shared memory frame consumer.

    This class is used by the Vision Module to read frames from shared memory.
    Reading is lock-free - the reader can read the current buffer while the
    writer is writing to a different buffer.

    Usage:
        reader = SharedMemoryFrameReader(name="billiards_video")

        try:
            reader.attach()

            while running:
                frame, metadata = reader.read_frame()
                if frame is not None:
                    process_frame(frame, metadata)
                time.sleep(0.001)  # 1ms poll
        finally:
            reader.detach()
    """

    def __init__(self, name: str):
        """Initialize frame reader.

        Args:
            name: Shared memory segment name (must match writer).
        """
        self.name = name

        # State
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.header: Optional[SharedMemoryHeader] = None
        self.last_read_counter = -1  # Track which frames we've seen

        # Frame configuration (read from header)
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.frame_format: Optional[FrameFormat] = None
        self.bytes_per_frame: Optional[int] = None

        logger.info(f"Initializing SharedMemoryFrameReader: {name}")

    def attach(self, timeout: float = 5.0) -> None:
        """Attach to shared memory segment.

        This will wait for the shared memory segment to be created by the writer.

        Args:
            timeout: Maximum time to wait for segment in seconds.

        Raises:
            TimeoutError: If segment not created within timeout.
            OSError: If attachment fails.
        """
        # Wait for shared memory to exist
        start_time = time.time()
        while True:
            try:
                # Try to open existing shared memory
                self.shm = shared_memory.SharedMemory(name=self.name)
                break
            except FileNotFoundError:
                fallback_file = _fallback_path(self.name)
                if fallback_file.exists():
                    self.shm = FileBackedSharedMemory(
                        self.name, size=None, create=False
                    )
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Shared memory segment '{self.name}' not created within {timeout}s"
                    )
                time.sleep(0.1)
            except PermissionError:
                fallback_file = _fallback_path(self.name)
                if fallback_file.exists():
                    self.shm = FileBackedSharedMemory(
                        self.name, size=None, create=False
                    )
                    break
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Shared memory segment '{self.name}' not accessible within {timeout}s"
                    )
                time.sleep(0.1)

        # Read header
        header_view = memoryview(self.shm.buf)[0:HEADER_SIZE]
        self.header = SharedMemoryHeader(header_view)

        # Validate header
        if not self.header.validate():
            raise RuntimeError("Invalid shared memory header (magic/version mismatch)")

        # Read configuration
        self.width, self.height, self.frame_format, self.bytes_per_frame = (
            self.header.read_config()
        )

        logger.info(
            f"Attached to shared memory: {self.name}, "
            f"{self.width}x{self.height} {self.frame_format.name}"
        )

    def read_frame(self) -> tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Read the current frame from shared memory.

        This is a non-blocking read. If no new frame is available since the
        last read, returns (None, None).

        Returns:
            Tuple of (frame, metadata) where frame is a numpy array and metadata
            is FrameMetadata. Returns (None, None) if no new frame available.

        Raises:
            RuntimeError: If reader not attached.
        """
        if self.shm is None or self.header is None:
            raise RuntimeError("Reader not attached. Call attach() first.")

        # Check if there's a new frame
        current_counter = self.header.get_write_counter()
        if current_counter == 0 or current_counter == self.last_read_counter:
            # No frames written yet or no new frame since last read
            return None, None

        # Get current read index
        buffer_index = self.header.get_read_index()

        # Read metadata
        metadata = self.header.read_metadata()

        # Calculate buffer offset
        buffer_offset = HEADER_SIZE + (buffer_index * self.bytes_per_frame)

        # Read frame data
        frame_view = memoryview(self.shm.buf)[
            buffer_offset : buffer_offset + self.bytes_per_frame
        ]

        # Convert to numpy array
        if self.frame_format in (FrameFormat.RGB24, FrameFormat.BGR24):
            shape = (self.height, self.width, 3)
        elif self.frame_format == FrameFormat.GRAY8:
            shape = (self.height, self.width)
        elif self.frame_format in (FrameFormat.RGBA32, FrameFormat.BGRA32):
            shape = (self.height, self.width, 4)
        else:
            raise RuntimeError(f"Unsupported frame format: {self.frame_format}")

        frame = np.frombuffer(frame_view, dtype=np.uint8).reshape(shape).copy()

        # Update counter
        self.last_read_counter = current_counter

        return frame, metadata

    def wait_for_frame(
        self, timeout: float = 1.0
    ) -> tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Wait for a new frame with timeout.

        This is a blocking read that waits for a new frame to be available.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (frame, metadata) or (None, None) if timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            frame, metadata = self.read_frame()
            if frame is not None:
                return frame, metadata
            time.sleep(0.001)  # 1ms poll
        return None, None

    def detach(self) -> None:
        """Detach from shared memory.

        This should be called during shutdown to properly release resources.
        """
        logger.info("Detaching from shared memory")

        # Close shared memory (but don't unlink - writer owns it)
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception as e:
                logger.error(f"Failed to close shared memory: {e}")
            finally:
                self.shm = None

        self.header = None
        logger.info("Detached from shared memory")

    def __enter__(self):
        """Context manager entry."""
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.detach()
