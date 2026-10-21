"""Hardware-free lifecycle checks for the optional Kinect capture adapter."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from backend.vision import kinect2_capture as kinect_module
from backend.vision.kinect2_capture import Kinect2Capture, Kinect2Status

if TYPE_CHECKING:
    import pytest


def make_capture(monkeypatch: pytest.MonkeyPatch, **config: Any) -> Kinect2Capture:
    """Construct the adapter without requiring the optional SDK package."""
    monkeypatch.setattr(kinect_module, "KINECT2_AVAILABLE", True)
    monkeypatch.setattr(kinect_module, "_load_kinect2_sdk", lambda: object())
    return Kinect2Capture(config)


def test_start_and_stop_status_transitions_do_not_reenter_the_state_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinishedThread:
        def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon
            self.alive = False

        def start(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.alive = False

    capture = make_capture(monkeypatch)
    monkeypatch.setattr(kinect_module.threading, "Thread", FinishedThread)
    capture._connect_kinect = lambda: True  # type: ignore[method-assign]
    statuses: list[Kinect2Status] = []
    capture.set_status_callback(statuses.append)

    assert capture.start_capture() is True
    capture.stop_capture()

    assert statuses == [
        Kinect2Status.CONNECTING,
        Kinect2Status.CONNECTED,
        Kinect2Status.DISCONNECTED,
    ]
    assert capture.get_status() is Kinect2Status.DISCONNECTED


def test_stop_timeout_keeps_sdk_device_owned_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_started = threading.Event()
    finish_read = threading.Event()

    class FakeDevice:
        def __init__(self) -> None:
            self.stop_count = 0
            self.close_count = 0
            self.read_active = False
            self.released_during_read = False

        def stop(self) -> None:
            self.released_during_read = self.read_active
            self.stop_count += 1

        def close(self) -> None:
            self.close_count += 1

    capture = make_capture(monkeypatch, stop_timeout=0.01)
    device = FakeDevice()
    capture._device = device  # type: ignore[assignment]
    capture._connect_kinect = lambda: True  # type: ignore[method-assign]

    def blocking_acquisition() -> None:
        device.read_active = True
        read_started.set()
        assert finish_read.wait(timeout=2.0)
        device.read_active = False

    capture._capture_loop = blocking_acquisition  # type: ignore[method-assign]

    assert capture.start_capture() is True
    assert read_started.wait(timeout=1.0)
    worker = capture._capture_thread
    assert worker is not None

    capture.stop_capture()

    assert worker.is_alive()
    assert capture._capture_thread is worker
    assert capture._device is device  # type: ignore[comparison-overlap]
    assert device.stop_count == 0
    assert device.close_count == 0
    assert capture.get_status() is Kinect2Status.ERROR
    assert capture.start_capture() is False

    finish_read.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert capture._device is None
    assert device.stop_count == 1
    assert device.close_count == 1
    assert device.released_during_read is False
    assert capture.get_status() is Kinect2Status.DISCONNECTED

    capture.stop_capture()
    assert capture._capture_thread is None
    assert device.stop_count == 1
