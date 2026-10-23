"""Legacy Python shared-memory video compatibility types.

The supported application captures in Rust. These Python types remain only for
offline compatibility tests and are not connected to the Rust capture path.
"""

from backend.video.ipc.shared_memory import (
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
]
