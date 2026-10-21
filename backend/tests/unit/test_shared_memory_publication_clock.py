"""Compatibility tests for the optional shared-memory monotonic clock."""

from backend.video.ipc.shared_memory import (
    HEADER_SIZE,
    FrameFormat,
    FrameMetadata,
    SharedMemoryFrameWriter,
    SharedMemoryHeader,
)


def test_publication_monotonic_clock_uses_reserved_header_space() -> None:
    header_memory = memoryview(bytearray(HEADER_SIZE))
    header = SharedMemoryHeader(header_memory)

    header.initialize(8, 6, FrameFormat.BGR24, 8 * 6 * 3)
    assert header.HEADER_STRUCT.size == 256
    assert header.read_metadata().publication_monotonic_ns is None

    header.update_frame_metadata(
        FrameMetadata(
            frame_number=12,
            timestamp_sec=1_000,
            timestamp_nsec=123,
            width=8,
            height=6,
            format=FrameFormat.BGR24,
            writer_pid=55,
            publication_monotonic_ns=987_654_321,
        ),
        buffer_index=2,
    )

    metadata = header.read_metadata()
    assert metadata.frame_number == 12
    assert metadata.writer_pid == 55
    assert metadata.timestamp == 1_000.000000123
    assert metadata.publication_monotonic_ns == 987_654_321
    header.close()


def test_legacy_metadata_without_publication_clock_remains_supported() -> None:
    header_memory = memoryview(bytearray(HEADER_SIZE))
    header = SharedMemoryHeader(header_memory)
    header.initialize(8, 6, FrameFormat.BGR24, 8 * 6 * 3)
    header.update_frame_metadata(
        FrameMetadata(
            frame_number=1,
            timestamp_sec=1_000,
            timestamp_nsec=0,
            width=8,
            height=6,
            format=FrameFormat.BGR24,
        ),
        buffer_index=0,
    )

    assert header.read_metadata().publication_monotonic_ns is None
    header.close()


def test_writer_publication_clock_is_opt_in() -> None:
    ordinary_writer = SharedMemoryFrameWriter("ordinary", 8, 6)
    audit_writer = SharedMemoryFrameWriter(
        "audit", 8, 6, capture_publication_clock=True
    )

    assert ordinary_writer.capture_publication_clock is False
    assert audit_writer.capture_publication_clock is True
