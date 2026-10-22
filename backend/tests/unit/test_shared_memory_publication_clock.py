"""Compatibility tests for shared-memory publication clock metadata."""

import struct
import time
import uuid

import numpy as np

import backend.video.process as video_process_module
from backend.video.ipc.shared_memory import (
    BUFFER_COUNT,
    HEADER_SIZE,
    MAGIC_NUMBER,
    PROTOCOL_VERSION,
    FrameFormat,
    FrameMetadata,
    SharedMemoryFrameWriter,
    SharedMemoryHeader,
)
from backend.video.process import VideoProcess

_LEGACY_HEADER_STRUCT = struct.Struct("QIIIIIQIIQQQQII168x")


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
    # A pre-clock writer uses the same protocol version and leaves the former
    # reserved tail zeroed. The new reader should continue treating that value
    # as an unavailable optional timestamp.
    legacy_header = _LEGACY_HEADER_STRUCT.pack(
        MAGIC_NUMBER,
        PROTOCOL_VERSION,
        BUFFER_COUNT,
        8,
        6,
        int(FrameFormat.BGR24),
        8 * 6 * 3,
        0,
        0,
        1,
        1,
        1_000,
        0,
        55,
        0,
    )
    header_memory[: len(legacy_header)] = legacy_header

    assert header.validate() is True
    assert header.read_metadata().publication_monotonic_ns is None
    header.close()


def test_legacy_reader_ignores_nonzero_publication_clock_in_reserved_bytes() -> None:
    # Version 1 readers that predate the optional clock treat these eight bytes
    # as part of the existing reserved tail. Keep that layout readable without
    # changing the protocol version or moving any fields they consume.
    assert _LEGACY_HEADER_STRUCT.size == SharedMemoryHeader.HEADER_STRUCT.size

    header_memory = memoryview(bytearray(HEADER_SIZE))
    header = SharedMemoryHeader(header_memory)
    header.initialize(8, 6, FrameFormat.BGR24, 8 * 6 * 3)
    header.update_frame_metadata(
        FrameMetadata(
            frame_number=3,
            timestamp_sec=1_001,
            timestamp_nsec=456,
            width=8,
            height=6,
            format=FrameFormat.BGR24,
            publication_monotonic_ns=987_654_321,
        ),
        buffer_index=1,
    )

    legacy_fields = _LEGACY_HEADER_STRUCT.unpack(
        header_memory[: _LEGACY_HEADER_STRUCT.size]
    )
    assert legacy_fields[0] == MAGIC_NUMBER
    assert legacy_fields[1] == PROTOCOL_VERSION
    assert legacy_fields[2] == BUFFER_COUNT
    assert legacy_fields[10] == 3
    assert legacy_fields[11] == 1_001
    assert legacy_fields[12] == 456
    header.close()


def test_writer_captures_publication_clock_by_default() -> None:
    writer = SharedMemoryFrameWriter(f"clock-{uuid.uuid4().hex[:12]}", 8, 6)

    assert writer.capture_publication_clock is True
    try:
        writer.initialize()
        start_ns = time.perf_counter_ns()
        writer.write_frame(np.zeros((6, 8, 3), dtype=np.uint8), frame_number=4)
        finish_ns = time.perf_counter_ns()

        assert writer.header is not None
        publication_ns = writer.header.read_metadata().publication_monotonic_ns
        assert publication_ns is not None
        assert start_ns <= publication_ns <= finish_ns
    finally:
        writer.cleanup()


def test_legacy_writer_can_still_leave_publication_clock_unavailable() -> None:
    legacy_writer = SharedMemoryFrameWriter(
        f"legacy-{uuid.uuid4().hex[:12]}",
        8,
        6,
        capture_publication_clock=False,
    )

    assert legacy_writer.capture_publication_clock is False
    try:
        legacy_writer.initialize()
        legacy_writer.write_frame(np.zeros((6, 8, 3), dtype=np.uint8), frame_number=5)
        assert legacy_writer.header is not None
        assert legacy_writer.header.read_metadata().publication_monotonic_ns is None
    finally:
        legacy_writer.cleanup()


def test_video_process_enables_publication_clock_without_audit(monkeypatch) -> None:
    class ConfigStub:
        def get(self, _path, default=None):
            return default

    captured: dict[str, object] = {}

    class WriterStub:
        total_size = 0

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def initialize(self):
            return None

    monkeypatch.setattr(video_process_module, "SharedMemoryFrameWriter", WriterStub)
    process = VideoProcess(ConfigStub())

    assert process.audit_recorder is None
    assert process._initialize_ipc_writer(8, 6) is True
    assert captured["capture_publication_clock"] is True
