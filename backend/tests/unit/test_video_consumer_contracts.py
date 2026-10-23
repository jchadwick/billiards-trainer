"""Deterministic contract tests for the shared-memory video consumer."""

from typing import Any

import numpy as np
import pytest

import backend.config as config_module
import backend.vision.stream.video_consumer as video_consumer_module
from backend.video.ipc.shared_memory import FrameFormat, FrameMetadata
from backend.vision.stream.video_consumer import (
    VideoConsumer,
    VideoModuleNotAvailableError,
)


class MappingConfig:
    """Small config double that exercises VideoConsumer's normal lookup path."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class ReaderStub:
    """In-memory reader double; no shared memory or devices are touched."""

    width = 640
    height = 480
    frame_format = FrameFormat.BGR24

    def __init__(
        self, *, read_result=(None, None), attach_error=None, detach_error=None
    ):
        self.read_result = read_result
        self.attach_error = attach_error
        self.detach_error = detach_error
        self.attach_timeouts: list[float] = []
        self.read_count = 0
        self.detach_count = 0

    def attach(self, *, timeout: float) -> None:
        self.attach_timeouts.append(timeout)
        if self.attach_error is not None:
            raise self.attach_error

    def read_frame(self):
        self.read_count += 1
        if isinstance(self.read_result, Exception):
            raise self.read_result
        return self.read_result

    def detach(self) -> None:
        self.detach_count += 1
        if self.detach_error is not None:
            raise self.detach_error


def configure(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any] | None = None):
    monkeypatch.setattr(config_module, "config", MappingConfig(values))
    return VideoConsumer()


def install_reader(
    monkeypatch: pytest.MonkeyPatch, reader: ReaderStub
) -> list[dict[str, Any]]:
    constructor_calls: list[dict[str, Any]] = []

    def create_reader(*, name: str) -> ReaderStub:
        constructor_calls.append({"name": name})
        return reader

    monkeypatch.setattr(video_consumer_module, "SharedMemoryFrameReader", create_reader)
    return constructor_calls


def sample_metadata(frame_number: int = 4) -> FrameMetadata:
    return FrameMetadata(
        frame_number=frame_number,
        timestamp_sec=12,
        timestamp_nsec=500_000_000,
        width=2,
        height=1,
        format=FrameFormat.BGR24,
        writer_pid=42,
        publication_monotonic_ns=12_500_000_000,
    )


def test_initialization_uses_configured_shared_memory_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = configure(
        monkeypatch,
        {
            "video.shared_memory_name": "table_camera_frames",
            "video.shared_memory_attach_timeout_sec": 1.25,
        },
    )

    assert consumer._shm_name == "table_camera_frames"
    assert consumer._attach_timeout == 1.25
    assert consumer.reader is None
    assert consumer.is_running is False


def test_initialization_uses_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = configure(monkeypatch)

    assert consumer._shm_name == "billiards_video"
    assert consumer._attach_timeout == 5.0


def test_start_attaches_configured_reader_and_marks_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = configure(
        monkeypatch,
        {
            "video.shared_memory_name": "replay_frames",
            "video.shared_memory_attach_timeout_sec": 0.75,
        },
    )
    reader = ReaderStub()
    constructor_calls = install_reader(monkeypatch, reader)

    consumer.start()

    assert constructor_calls == [{"name": "replay_frames"}]
    assert reader.attach_timeouts == [0.75]
    assert consumer.reader is reader
    assert consumer.is_running is True


def test_start_translates_attach_timeout_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = configure(monkeypatch)
    timeout = TimeoutError("segment missing")
    reader = ReaderStub(attach_error=timeout)
    install_reader(monkeypatch, reader)

    with pytest.raises(
        VideoModuleNotAvailableError, match="not found within 5.0s"
    ) as error:
        consumer.start()

    assert error.value.__cause__ is timeout
    assert consumer.is_running is False
    assert "Rust runtime does not publish this protocol" in str(error.value)
    assert "python -m backend.video" not in str(error.value)


def test_start_translates_other_attach_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = configure(monkeypatch, {"video.shared_memory_name": "broken_segment"})
    reader = ReaderStub(attach_error=OSError("permission denied"))
    install_reader(monkeypatch, reader)

    with pytest.raises(
        VideoModuleNotAvailableError,
        match=(
            "Failed to attach to legacy Python shared memory "
            "'broken_segment': permission denied"
        ),
    ) as error:
        consumer.start()

    assert isinstance(error.value.__cause__, OSError)
    assert consumer.is_running is False
    assert "Rust runtime does not publish this protocol" in str(error.value)
    assert "python -m backend.video" not in str(error.value)


def test_frame_reads_require_started_consumer() -> None:
    consumer = VideoConsumer()

    with pytest.raises(RuntimeError, match=r"Call start\(\) before get_frame\(\)"):
        consumer.get_frame()
    with pytest.raises(
        RuntimeError, match=r"Call start\(\) before get_frame_with_metadata\(\)"
    ):
        consumer.get_frame_with_metadata()


def test_get_frame_returns_pixels_and_metadata_reader_returns_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.arange(6, dtype=np.uint8).reshape((1, 2, 3))
    metadata = sample_metadata()
    reader = ReaderStub(read_result=(frame, metadata))
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)
    consumer.start()

    assert consumer.get_frame() is frame
    received_frame, received_metadata = consumer.get_frame_with_metadata()

    assert received_frame is frame
    assert received_metadata is metadata
    assert reader.read_count == 2


def test_frame_read_preserves_absence_and_optional_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = ReaderStub(read_result=(None, None))
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)
    consumer.start()

    assert consumer.get_frame_with_metadata() == (None, None)

    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    reader.read_result = (frame, None)
    received_frame, metadata = consumer.get_frame_with_metadata()
    assert received_frame is frame
    assert metadata is None


def test_running_consumer_without_reader_degrades_to_no_frame() -> None:
    consumer = VideoConsumer()
    consumer.is_running = True

    assert consumer.get_frame_with_metadata() == (None, None)


def test_read_failures_degrade_to_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReaderStub(read_result=OSError("mapping was closed"))
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)
    consumer.start()

    assert consumer.get_frame_with_metadata() == (None, None)


def test_stop_detaches_and_resets_state(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ReaderStub()
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)
    consumer.start()

    consumer.stop()

    assert reader.detach_count == 1
    assert consumer.reader is None
    assert consumer.is_running is False


def test_stop_swallows_detach_failure_and_still_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = ReaderStub(detach_error=OSError("already closed"))
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)
    consumer.start()

    consumer.stop()

    assert reader.detach_count == 1
    assert consumer.reader is None
    assert consumer.is_running is False


def test_stop_without_reader_is_safe_and_idempotent() -> None:
    consumer = VideoConsumer()

    consumer.stop()
    consumer.stop()

    assert consumer.reader is None
    assert consumer.is_running is False


def test_context_manager_starts_and_stops_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = ReaderStub()
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)

    with consumer as entered:
        assert entered is consumer
        assert consumer.is_running is True

    assert reader.detach_count == 1
    assert consumer.reader is None
    assert consumer.is_running is False


def test_context_manager_stops_after_body_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = ReaderStub()
    install_reader(monkeypatch, reader)
    consumer = configure(monkeypatch)

    with pytest.raises(ValueError, match="consumer body failed"):
        with consumer:
            raise ValueError("consumer body failed")

    assert reader.detach_count == 1
    assert consumer.is_running is False
