"""Deterministic contracts for freshness-first camera capture."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from backend.vision import capture as capture_module
from backend.vision.capture import CameraCapture, CameraStatus

if TYPE_CHECKING:
    import pytest


class FakeVideoCapture:
    """OpenCV-shaped source with controlled reads and driver properties."""

    def __init__(
        self,
        reads: list[tuple[bool, np.ndarray | None]] | None = None,
        *,
        opened: bool = True,
        fail_set: bool = False,
        after_read: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = deque(reads or [])
        self.fail_set = fail_set
        self.after_read = after_read
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        result = self.reads.popleft() if self.reads else (False, None)
        if self.after_read is not None:
            self.after_read(*result)
        return result

    def set(self, property_id: int, value: float) -> bool:
        if self.fail_set:
            raise RuntimeError("buffer property unsupported")
        self.set_calls.append((property_id, value))
        return True

    def get(self, _property_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self.released = True
        self.opened = False


def _frame(value: int) -> np.ndarray:
    return np.full((3, 5, 3), value, dtype=np.uint8)


def test_network_source_keeps_only_latest_copy_and_counts_producer_drops() -> None:
    reusable_buffer = _frame(1)

    def reuse_buffer(success: bool, _frame_value: np.ndarray | None) -> None:
        if success:
            reusable_buffer[:] += 1

    camera = FakeVideoCapture(
        reads=[
            (True, reusable_buffer),
            (True, reusable_buffer),
            (True, reusable_buffer),
            (False, None),
        ],
        after_read=reuse_buffer,
    )
    capture = CameraCapture(
        {
            "device_id": "http://127.0.0.1:8080/stream",
            "buffer_size": 8,
            "fps": 0,
            "auto_reconnect": False,
        }
    )
    capture._cap = camera  # type: ignore[assignment]

    capture._capture_loop()

    result = capture.get_latest_frame()
    assert capture._frame_queue.maxsize == 1
    assert result is not None
    delivered_frame, metadata = result
    assert metadata.frame_number == 3
    assert capture._frames_captured == 3
    assert capture._frames_dropped == 2
    # Each published frame is copied before the backend reuses its buffer.
    reusable_buffer[:] = 99
    assert np.all(delivered_frame == 4)
    assert np.all(reusable_buffer == 99)


def test_disconnected_source_reconnects_then_publishes_the_first_new_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = FakeVideoCapture(reads=[(True, _frame(7)), (False, None)])
    capture = CameraCapture({"auto_reconnect": True, "reconnect_delay": 0, "fps": 0})
    capture._cap = FakeVideoCapture(opened=False)  # type: ignore[assignment]
    statuses: list[CameraStatus] = []
    capture.set_status_callback(statuses.append)
    connect_calls = 0

    def reconnect() -> bool:
        nonlocal connect_calls
        connect_calls += 1
        capture._cap = replacement  # type: ignore[assignment]
        return True

    capture._connect_camera = reconnect  # type: ignore[method-assign]

    def finish_after_first_publication(_seconds: float) -> None:
        capture._stop_event.set()

    capture._fps = 30
    monkeypatch.setattr(capture_module.time, "sleep", finish_after_first_publication)

    capture._capture_loop()

    result = capture.get_frame()
    assert connect_calls == 1
    assert statuses == [CameraStatus.RECONNECTING, CameraStatus.CONNECTED]
    assert capture.get_status() is CameraStatus.CONNECTED
    assert result is not None
    frame, metadata = result
    assert metadata.frame_number == 1
    assert np.all(frame == 7)


def test_capture_read_exception_becomes_health_error_without_stale_delivery() -> None:
    class RaisingVideoCapture(FakeVideoCapture):
        def read(self) -> tuple[bool, np.ndarray | None]:
            raise OSError("device read interrupted")

    capture = CameraCapture({"auto_reconnect": False, "fps": 0})
    capture._cap = RaisingVideoCapture()  # type: ignore[assignment]
    statuses: list[CameraStatus] = []
    capture.set_status_callback(statuses.append)

    capture._capture_loop()

    assert capture.get_frame() is None
    assert capture.get_health().error_count == 1
    assert capture.get_health().last_error == "device read interrupted"
    assert capture.get_status() is CameraStatus.ERROR
    assert statuses == [CameraStatus.ERROR]


def test_stream_buffer_control_failure_does_not_disable_network_capture() -> None:
    capture = CameraCapture({"device_id": "http://127.0.0.1:8080/stream"})
    capture._cap = FakeVideoCapture(fail_set=True)  # type: ignore[assignment]

    assert capture._configure_camera() is True


def test_camera_configuration_failure_releases_newly_opened_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeVideoCapture(fail_set=True)
    monkeypatch.setattr(
        capture_module.cv2, "VideoCapture", lambda _device, _backend: camera
    )
    capture = CameraCapture({"device_id": 0})

    assert capture._connect_camera() is False
    assert camera.released is True
    assert capture._cap is None
    assert capture._error_count == 2
    assert capture._last_error == "Camera configuration failed"
