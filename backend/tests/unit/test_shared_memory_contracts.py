"""Contract tests for shared-memory framing, lifecycle, and fallback behavior."""

from __future__ import annotations

import logging
import os
import struct
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from backend.video.ipc import shared_memory as ipc

if TYPE_CHECKING:
    from pathlib import Path


def unique_name(prefix: str = "billiards-test") -> str:
    """Return a per-test IPC name so parallel test workers cannot collide."""
    return f"{prefix[:8]}-{uuid.uuid4().hex[:10]}"


def frame_for(frame_format: ipc.FrameFormat, width: int, height: int) -> np.ndarray:
    channels = {
        ipc.FrameFormat.RGB24: 3,
        ipc.FrameFormat.BGR24: 3,
        ipc.FrameFormat.GRAY8: 1,
        ipc.FrameFormat.RGBA32: 4,
        ipc.FrameFormat.BGRA32: 4,
    }[frame_format]
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)


def make_header() -> tuple[bytearray, ipc.SharedMemoryHeader]:
    storage = bytearray(ipc.HEADER_SIZE)
    return storage, ipc.SharedMemoryHeader(memoryview(storage))


@pytest.mark.parametrize(
    ("frame_format", "channels"),
    [
        (ipc.FrameFormat.RGB24, 3),
        (ipc.FrameFormat.BGR24, 3),
        (ipc.FrameFormat.GRAY8, 1),
        (ipc.FrameFormat.RGBA32, 4),
        (ipc.FrameFormat.BGRA32, 4),
    ],
)
def test_writer_reader_round_trip_and_latest_triple_buffer_index(
    frame_format: ipc.FrameFormat, channels: int
) -> None:
    width, height = 4, 3
    name = unique_name()
    writer = ipc.SharedMemoryFrameWriter(name, width, height, frame_format=frame_format)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        writer.initialize()
        reader.attach()
        assert reader.width == width
        assert reader.height == height
        assert reader.frame_format is frame_format
        assert reader.bytes_per_frame == width * height * channels
        assert reader.read_frame() == (None, None)

        # Publish more frames than there are slots without requiring the reader
        # to consume intermediate frames. It observes the newest header index.
        frames = [frame_for(frame_format, width, height) + value for value in range(3)]
        for number, frame in enumerate(frames, start=10):
            writer.write_frame(frame, frame_number=number)
        assert writer.current_buffer_index == (3 % ipc.BUFFER_COUNT)
        assert writer.header is not None
        assert writer.header.get_read_index() == writer.current_buffer_index
        assert writer.header.get_write_counter() == 3

        actual, metadata = reader.read_frame()
        assert actual is not None
        assert metadata is not None
        np.testing.assert_array_equal(actual, frames[-1])
        assert metadata.frame_number == 12
        assert metadata.width == width
        assert metadata.height == height
        assert metadata.format is frame_format
        assert metadata.writer_pid == os.getpid()
        assert metadata.timestamp_sec > 0
        assert 0 <= metadata.timestamp_nsec < 1_000_000_000
        assert metadata.publication_monotonic_ns is not None
        assert reader.read_frame() == (None, None)

        # Reader gets an owned copy, so the writer can overwrite the slot
        # without mutating the frame already handed to its caller.
        actual_snapshot = actual.copy()
        writer.write_frame(np.zeros_like(frames[0]), frame_number=13)
        np.testing.assert_array_equal(actual, actual_snapshot)
        newest, newest_metadata = reader.read_frame()
        assert newest is not None
        assert newest_metadata is not None
        assert newest_metadata.frame_number == 13
        np.testing.assert_array_equal(newest, np.zeros_like(frames[0]))
    finally:
        reader.detach()
        writer.cleanup()


def test_writer_uses_internal_frame_counter_and_can_disable_publication_clock() -> None:
    name = unique_name()
    frame = np.full((2, 3, 3), 17, dtype=np.uint8)
    with ipc.SharedMemoryFrameWriter(
        name, 3, 2, capture_publication_clock=False
    ) as writer:
        writer.write_frame(frame)
        writer.write_frame(frame)
        stats = writer.get_stats()
        assert stats["initialized"] is True
        assert stats["frames_written"] == 2
        assert stats["write_counter"] == 2
        assert stats["current_buffer_index"] == 2

        with ipc.SharedMemoryFrameReader(name) as reader:
            actual, metadata = reader.read_frame()
            assert actual is not None
            assert metadata is not None
            assert metadata.frame_number == 1
            assert metadata.publication_monotonic_ns is None
            np.testing.assert_array_equal(actual, frame)
    assert writer.get_stats() == {"initialized": False}
    writer.cleanup()  # cleanup is idempotent after context exit


def test_writer_rejects_uninitialized_and_mismatched_frames() -> None:
    writer = ipc.SharedMemoryFrameWriter(unique_name(), 2, 2)
    with pytest.raises(RuntimeError, match="not initialized"):
        writer.write_frame(np.zeros((2, 2, 3), dtype=np.uint8))

    writer.initialize()
    try:
        with pytest.raises(ValueError, match="don't match"):
            writer.write_frame(np.zeros((1, 2, 3), dtype=np.uint8))
    finally:
        writer.cleanup()


def test_writer_makes_non_contiguous_input_contiguous_before_publish() -> None:
    name = unique_name()
    source = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    non_contiguous = source[:, ::2, :]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    writer = ipc.SharedMemoryFrameWriter(name, 2, 2)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        writer.initialize()
        reader.attach()
        writer.write_frame(non_contiguous)
        actual, _ = reader.read_frame()
        assert actual is not None
        np.testing.assert_array_equal(actual, non_contiguous)
    finally:
        reader.detach()
        writer.cleanup()


def test_writer_rejects_unknown_frame_format() -> None:
    with pytest.raises(ValueError, match="Unsupported frame format"):
        ipc.SharedMemoryFrameWriter(unique_name(), 1, 1, frame_format=0)  # type: ignore[arg-type]


def test_header_round_trip_validation_metadata_and_reader_count() -> None:
    storage, header = make_header()
    header.initialize(640, 480, ipc.FrameFormat.BGR24, 640 * 480 * 3)
    assert header.validate()
    assert header.read_config() == (640, 480, ipc.FrameFormat.BGR24, 640 * 480 * 3)
    assert header.get_read_index() == 0
    assert header.get_write_counter() == 0

    header.increment_reader_count()
    header.increment_reader_count()
    values = header._unpack()
    assert values[14] == 2
    header.decrement_reader_count()
    header.decrement_reader_count()
    header.decrement_reader_count()
    assert header._unpack()[14] == 0

    metadata = ipc.FrameMetadata(
        frame_number=42,
        timestamp_sec=123,
        timestamp_nsec=456,
        width=640,
        height=480,
        format=ipc.FrameFormat.BGR24,
        writer_pid=9876,
        publication_monotonic_ns=555,
    )
    header.update_frame_metadata(metadata, buffer_index=2)
    assert header.get_read_index() == 2
    assert header.get_write_counter() == 1
    assert header.read_metadata() == metadata
    assert metadata.timestamp == pytest.approx(123.000000456)

    # Zero writer/tick values retain the protocol's fallback semantics.
    fallback_metadata = ipc.FrameMetadata(
        frame_number=43,
        timestamp_sec=124,
        timestamp_nsec=0,
        width=640,
        height=480,
        format=ipc.FrameFormat.BGR24,
        writer_pid=0,
        publication_monotonic_ns=0,
    )
    header.update_frame_metadata(fallback_metadata, buffer_index=1)
    published = header.read_metadata()
    assert published.writer_pid == os.getpid()
    assert published.publication_monotonic_ns is None
    assert header.get_write_counter() == 2
    header.close()
    assert storage  # keep backing storage alive until the exported view is released


@pytest.mark.parametrize(
    ("field_index", "bad_value"),
    [(0, 0), (1, ipc.PROTOCOL_VERSION + 1)],
)
def test_header_rejects_wrong_magic_or_protocol_version(
    field_index: int, bad_value: int
) -> None:
    _, header = make_header()
    header.initialize(1, 1, ipc.FrameFormat.GRAY8, 1)
    values = list(header._unpack())
    values[field_index] = bad_value
    header.memory_view[: header.HEADER_STRUCT.size] = header.HEADER_STRUCT.pack(*values)
    assert not header.validate()
    header.close()


def test_header_validate_handles_truncated_header_and_config_rejects_unknown_format() -> (
    None
):
    short_header = ipc.SharedMemoryHeader(memoryview(bytearray(3)))
    assert not short_header.validate()
    short_header.close()

    _, header = make_header()
    header.initialize(1, 1, ipc.FrameFormat.GRAY8, 1)
    values = list(header._unpack())
    values[5] = 99
    header.memory_view[: header.HEADER_STRUCT.size] = header.HEADER_STRUCT.pack(*values)
    with pytest.raises(ValueError):
        header.read_config()
    header.close()


def test_file_backed_shared_memory_create_open_close_and_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    name = "nested/name"
    assert ipc._sanitize_name(name) == "nested_name"
    assert ipc._fallback_path(name) == tmp_path / "nested_name.bin"

    with pytest.raises(ValueError, match="requires size"):
        ipc.FileBackedSharedMemory(name, None, create=True)

    writer = ipc.FileBackedSharedMemory(name, 128, create=True)
    writer.buf[:4] = b"test"
    assert writer.path.exists()
    reader = ipc.FileBackedSharedMemory(name, None, create=False)
    assert reader.buf[:4].tobytes() == b"test"
    assert reader._size == 128
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = reader.buf

    # Repeated close/unlink operations are safe, and unlink owns deletion.
    reader.close()
    writer.unlink()
    writer.unlink()
    assert not writer.path.exists()


def test_fallback_cleanup_tolerates_an_already_removed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    ipc._remove_fallback_file("not-created")
    assert not ipc._fallback_path("not-created").exists()


def test_file_backed_unlink_tolerates_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    handle = ipc.FileBackedSharedMemory("missing", 32, create=True)
    handle.path.unlink()
    handle.unlink()
    assert handle._mmap is None
    assert handle._file is None


def test_shared_buffer_rejects_closed_handle() -> None:
    class ClosedHandle:
        @property
        def buf(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="closed"):
        ipc._shared_buffer(ClosedHandle())  # type: ignore[arg-type]


def test_reader_requires_attach_and_attach_timeout() -> None:
    reader = ipc.SharedMemoryFrameReader(unique_name())
    with pytest.raises(RuntimeError, match="not attached"):
        reader.read_frame()
    with pytest.raises(TimeoutError, match="not created"):
        reader.attach(timeout=0)
    assert reader.shm is None
    assert reader.header is None


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_reader_retries_transient_attach_failures_until_timeout(
    failure: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    name = unique_name("retry")

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise failure("segment unavailable")

    clock_values = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", unavailable)
    monkeypatch.setattr(ipc.time, "time", lambda: next(clock_values))
    monkeypatch.setattr(ipc.time, "sleep", lambda _duration: None)

    reader = ipc.SharedMemoryFrameReader(name)
    message = "not created" if failure is FileNotFoundError else "not accessible"
    with pytest.raises(TimeoutError, match=message):
        reader.attach(timeout=0.5)
    assert reader.shm is None


def test_reader_fails_clearly_if_attachment_provider_returns_no_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", lambda **_kwargs: None)
    reader = ipc.SharedMemoryFrameReader(unique_name("empty-handle"))
    with pytest.raises(RuntimeError, match="Failed to attach"):
        reader.attach(timeout=0)


def test_reader_detach_is_idempotent_and_does_not_unlink_writer_segment() -> None:
    name = unique_name()
    writer = ipc.SharedMemoryFrameWriter(name, 2, 1, ipc.FrameFormat.GRAY8)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        writer.initialize()
        reader.attach()
        assert reader.shm is not None
        reader.detach()
        reader.detach()
        # Writer ownership is preserved; a fresh reader can still attach.
        other_reader = ipc.SharedMemoryFrameReader(name)
        try:
            other_reader.attach()
            writer.write_frame(np.array([[1, 2]], dtype=np.uint8))
            frame, _ = other_reader.read_frame()
            np.testing.assert_array_equal(frame, [[1, 2]])
        finally:
            other_reader.detach()
    finally:
        reader.detach()
        writer.cleanup()


def test_reader_attach_rejects_invalid_header_and_releases_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = unique_name()
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    writer.initialize()
    try:
        assert writer.shm is not None
        storage = ipc._shared_buffer(writer.shm)
        storage[:8] = struct.pack("Q", 0)
        reader = ipc.SharedMemoryFrameReader(name)
        with pytest.raises(RuntimeError, match="Invalid shared memory header"):
            reader.attach(timeout=0)
        assert reader.shm is None
        assert reader.header is None
    finally:
        writer.cleanup()


def test_reader_rejects_missing_configuration_and_unknown_runtime_format() -> None:
    name = unique_name("reader-config")
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        writer.initialize()
        reader.attach()
        reader.width = None
        with pytest.raises(RuntimeError, match="configuration is unavailable"):
            reader.read_frame()

        reader.width = 1
        reader.frame_format = 99  # type: ignore[assignment]
        writer.write_frame(np.array([[9]], dtype=np.uint8))
        with pytest.raises(RuntimeError, match="Unsupported frame format"):
            reader.read_frame()
    finally:
        reader.detach()
        writer.cleanup()


def test_wait_for_frame_returns_new_frame_and_times_out_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = ipc.SharedMemoryFrameReader(unique_name("wait"))
    metadata = ipc.FrameMetadata(
        frame_number=2,
        timestamp_sec=4,
        timestamp_nsec=5,
        width=1,
        height=1,
        format=ipc.FrameFormat.GRAY8,
    )
    expected = np.array([[2]], dtype=np.uint8)
    reads = iter(((None, None), (expected, metadata)))
    monkeypatch.setattr(reader, "read_frame", lambda: next(reads))
    actual, actual_metadata = reader.wait_for_frame(timeout=1)
    assert actual is expected
    assert actual_metadata is metadata

    clock_values = iter((10.0, 10.0, 12.0))
    monkeypatch.setattr(ipc.time, "time", lambda: next(clock_values))
    monkeypatch.setattr(ipc.time, "sleep", lambda _duration: None)
    monkeypatch.setattr(reader, "read_frame", lambda: (None, None))
    assert reader.wait_for_frame(timeout=1) == (None, None)


def test_reader_attach_releases_a_truncated_fallback_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    name = unique_name("truncated")
    short_file = ipc.FileBackedSharedMemory(name, 32, create=True)
    short_file.close()

    def missing_posix_segment(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", missing_posix_segment)
    reader = ipc.SharedMemoryFrameReader(name)
    with pytest.raises(RuntimeError, match="Invalid shared memory header"):
        reader.attach(timeout=0)
    assert reader.shm is None
    assert reader.header is None
    ipc._remove_fallback_file(name)


def test_writer_uses_file_fallback_when_posix_creation_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    calls = 0

    def denied_creation(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FileNotFoundError
        raise PermissionError("sandbox disallows POSIX shm")

    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", denied_creation)
    name = unique_name("fallback-writer")
    writer = ipc.SharedMemoryFrameWriter(name, 2, 1, ipc.FrameFormat.GRAY8)
    try:
        writer.initialize()
        assert writer._using_fallback
        assert isinstance(writer.shm, ipc.FileBackedSharedMemory)
        writer.write_frame(np.array([[4, 5]], dtype=np.uint8))
        assert writer.get_stats()["write_counter"] == 1
    finally:
        writer.cleanup()
    assert not ipc._fallback_path(name).exists()


def test_reader_attaches_to_fallback_and_discovers_file_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    name = unique_name("fallback-reader")
    writer = ipc.FileBackedSharedMemory(name, ipc.HEADER_SIZE + 3 * 2, create=True)
    header = ipc.SharedMemoryHeader(ipc._shared_buffer(writer)[: ipc.HEADER_SIZE])
    header.initialize(3, 2, ipc.FrameFormat.GRAY8, 6)
    frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    writer.buf[ipc.HEADER_SIZE : ipc.HEADER_SIZE + frame.nbytes] = frame.tobytes()
    header.update_frame_metadata(
        ipc.FrameMetadata(
            frame_number=8,
            timestamp_sec=1,
            timestamp_nsec=2,
            width=3,
            height=2,
            format=ipc.FrameFormat.GRAY8,
        ),
        0,
    )
    header.close()

    def missing_posix_segment(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", missing_posix_segment)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        reader.attach(timeout=0)
        assert reader.shm is not None
        assert isinstance(reader.shm, ipc.FileBackedSharedMemory)
        assert reader.bytes_per_frame == 6
        actual, metadata = reader.read_frame()
        assert actual is not None
        assert metadata is not None
        np.testing.assert_array_equal(actual, frame)
        assert metadata.frame_number == 8
    finally:
        reader.detach()
        writer.unlink()


def test_reader_attach_permission_fallback_and_timeout_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    name = unique_name("permission-fallback")
    backing = ipc.FileBackedSharedMemory(name, ipc.HEADER_SIZE + 1, create=True)
    header = ipc.SharedMemoryHeader(ipc._shared_buffer(backing)[: ipc.HEADER_SIZE])
    header.initialize(1, 1, ipc.FrameFormat.GRAY8, 1)
    header.close()

    def permission_denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("access restricted")

    monkeypatch.setattr(ipc.shared_memory, "SharedMemory", permission_denied)
    reader = ipc.SharedMemoryFrameReader(name)
    try:
        reader.attach(timeout=0)
        assert isinstance(reader.shm, ipc.FileBackedSharedMemory)
    finally:
        reader.detach()
        backing.unlink()

    missing_reader = ipc.SharedMemoryFrameReader(unique_name("permission-missing"))
    with pytest.raises(TimeoutError, match="not accessible"):
        missing_reader.attach(timeout=0)


def test_file_backed_close_ignores_flush_error_and_still_closes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ipc, "_FALLBACK_DIR", tmp_path)
    handle = ipc.FileBackedSharedMemory("flush-failure", 16, create=True)
    file_obj = handle._file
    assert file_obj is not None
    mapping = handle._mmap
    assert mapping is not None

    class FlushFailure:
        def flush(self) -> None:
            raise OSError("flush failed")

        def close(self) -> None:
            mapping.close()

    handle._mmap = FlushFailure()  # type: ignore[assignment]
    handle.close()
    assert file_obj.closed
    assert handle._file is None
    assert handle._mmap is None
    handle.path.unlink(missing_ok=True)


def test_writer_cleanup_logs_close_failure_and_clears_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = unique_name("close-error")
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    writer.initialize()
    real_handle = writer.shm
    assert real_handle is not None

    class CloseFailure:
        def close(self) -> None:
            real_handle.close()
            raise OSError("close failed")

        def unlink(self) -> None:
            real_handle.unlink()

        @property
        def buf(self) -> memoryview:
            return real_handle.buf

    writer.shm = CloseFailure()  # type: ignore[assignment]
    with caplog.at_level(logging.ERROR, logger=ipc.__name__):
        writer.cleanup()
    assert writer.shm is None
    assert writer.header is None
    assert "Failed to remove shared memory segment" in caplog.text
    # Cleanup's close failed before unlink, so explicitly release the POSIX name.
    try:
        real_handle.unlink()
    except FileNotFoundError:
        pass


def test_reader_detach_logs_close_failure_and_clears_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = unique_name("reader-close-error")
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    writer.initialize()
    reader = ipc.SharedMemoryFrameReader(name)
    reader.attach()
    real_reader_handle = reader.shm
    assert real_reader_handle is not None

    class CloseFailure:
        def close(self) -> None:
            real_reader_handle.close()
            raise OSError("reader close failed")

    reader.shm = CloseFailure()  # type: ignore[assignment]
    try:
        with caplog.at_level(logging.ERROR, logger=ipc.__name__):
            reader.detach()
        assert reader.shm is None
        assert reader.header is None
        assert "Failed to close shared memory" in caplog.text
    finally:
        writer.cleanup()


def test_writer_warns_when_existing_segment_cannot_be_inspected(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = unique_name("permission-inspection")
    calls = 0
    real_shared_memory = ipc.shared_memory.SharedMemory

    def permission_for_probe_then_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("cannot inspect")
        return real_shared_memory(*args, **kwargs)

    monkeypatch.setattr(
        ipc.shared_memory, "SharedMemory", permission_for_probe_then_create
    )
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    try:
        with caplog.at_level(logging.WARNING, logger=ipc.__name__):
            writer.initialize()
        assert "assuming it does not exist" in caplog.text
    finally:
        writer.cleanup()


def test_writer_replaces_an_existing_posix_segment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    name = unique_name("replace")
    stale = ipc.shared_memory.SharedMemory(create=True, size=64, name=name)
    stale.close()
    writer = ipc.SharedMemoryFrameWriter(name, 1, 1, ipc.FrameFormat.GRAY8)
    try:
        with caplog.at_level(logging.WARNING, logger=ipc.__name__):
            writer.initialize()
        assert "Removed existing shared memory segment" in caplog.text
        assert writer.shm is not None
    finally:
        writer.cleanup()
