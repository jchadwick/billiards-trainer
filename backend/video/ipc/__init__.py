"""IPC (Inter-Process Communication) module for video frames."""

from backend.video.ipc.shared_memory import (
    BUFFER_COUNT,
    HEADER_SIZE,
    MAGIC_NUMBER,
    PROTOCOL_VERSION,
    FrameFormat,
    FrameMetadata,
    SharedMemoryFrameReader,
    SharedMemoryFrameWriter,
)

__all__ = [
    "FrameFormat",
    "FrameMetadata",
    "SharedMemoryFrameReader",
    "SharedMemoryFrameWriter",
    "MAGIC_NUMBER",
    "PROTOCOL_VERSION",
    "HEADER_SIZE",
    "BUFFER_COUNT",
]
