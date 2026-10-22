"""Adversarial contracts for camera restart, replay, and Kinect delegation paths."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np
import pytest

from backend.vision import capture as capture_module
from backend.vision.capture import CameraCapture, CameraStatus, FrameInfo


def _frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


class _VideoCapture:
    """OpenCV-shaped source with mutable position and deterministic frame reads."""

    def __init__(
        self,
        reads: list[tuple[bool, np.ndarray | None]] | None = None,
        *,
        opened: bool = True,
        position: float = 0.0,
        after_read: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = deque(reads or [])
        self.properties: dict[int, float] = {
            capture_module.cv2.CAP_PROP_POS_FRAMES: position
        }
        self.after_read = after_read
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        result = self.reads.popleft() if self.reads else (False, None)
        if result[0]:
            position_key = capture_module.cv2.CAP_PROP_POS_FRAMES
            self.properties[position_key] = self.properties.get(position_key, 0.0) + 1
        if self.after_read is not None:
            self.after_read(*result)
        return result

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.properties[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0.0)

    def release(self) -> None:
        self.released = True
        self.opened = False


def test_restarting_after_worker_exit_releases_old_source_and_resets_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinishedThread:
        def is_alive(self) -> bool:
            return False

    class StartedThread:
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

    old_source = _VideoCapture()
    new_source = _VideoCapture()
    sources = iter((new_source,))
    capture = CameraCapture({"buffer_size": 2})
    capture._cap = old_source  # type: ignore[assignment]
    capture._capture_thread = FinishedThread()  # type: ignore[assignment]
    capture._frames_captured = 27
    capture._frames_dropped = 4
    capture._fps_tracker[:] = [10.0, 10.1]
    capture._frame_queue.put_nowait((_frame(99), FrameInfo(27, 10.1, (3, 2), 3)))

    def connect_new_source() -> bool:
        capture._cap = next(sources)  # type: ignore[assignment]
        return True

    capture._connect_camera = connect_new_source  # type: ignore[method-assign]
    monkeypatch.setattr(capture_module.threading, "Thread", StartedThread)

    assert capture.start_capture() is True

    assert old_source.released is True
    assert capture._cap is new_source  # type: ignore[comparison-overlap]
    assert capture._frames_captured == 0
    assert capture._frames_dropped == 0
    assert capture._fps_tracker == []
    assert capture.get_frame() is None
    assert capture.get_status() is CameraStatus.CONNECTED

    capture.stop_capture()
    assert new_source.released is True


def test_looping_video_seeks_to_configured_start_after_reaching_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _VideoCapture(reads=[(True, _frame(7))], position=8)
    capture = CameraCapture(
        {
            "device_id": "recorded.mp4",
            "video_start_frame": 3,
            "video_end_frame": 8,
            "loop_video": True,
            "fps": 30,
        }
    )
    capture._cap = source  # type: ignore[assignment]

    def stop_after_publication(_seconds: float) -> None:
        capture._stop_event.set()

    monkeypatch.setattr(capture_module.time, "sleep", stop_after_publication)
    capture._capture_loop()

    assert source.set_calls == [(capture_module.cv2.CAP_PROP_POS_FRAMES, 3)]
    delivered = capture.get_frame()
    assert delivered is not None
    frame, metadata = delivered
    assert metadata.frame_number == 1
    assert np.array_equal(frame, _frame(7))


def test_video_eof_rewinds_when_looping_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _VideoCapture(
        reads=[(False, None), (True, _frame(8))],
        position=0,
    )
    capture = CameraCapture(
        {
            "device_id": "recorded.mp4",
            "video_start_frame": 2,
            "loop_video": True,
            "fps": 30,
        }
    )
    capture._cap = source  # type: ignore[assignment]

    def stop_after_publication(_seconds: float) -> None:
        capture._stop_event.set()

    monkeypatch.setattr(capture_module.time, "sleep", stop_after_publication)
    capture._capture_loop()

    assert source.set_calls == [(capture_module.cv2.CAP_PROP_POS_FRAMES, 2)]
    delivered = capture.get_frame()
    assert delivered is not None
    assert np.array_equal(delivered[0], _frame(8))
    assert capture._error_count == 0


def test_missing_kinect_runtime_fails_start_without_constructing_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture_module, "KINECT2_AVAILABLE", False)
    monkeypatch.setattr(
        capture_module,
        "Kinect2Capture",
        lambda _config: pytest.fail("unavailable SDK must not be constructed"),
    )
    capture = CameraCapture({"backend": "kinect2"})

    assert capture.start_capture() is False
    assert capture._kinect2 is None
    assert capture._capture_thread is None
    assert capture.get_status() is CameraStatus.ERROR


def test_kinect_constructor_failure_is_recorded_and_clears_partial_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_during_construction(_config: dict[str, Any]) -> Any:
        raise OSError("Kinect SDK initialization failed")

    monkeypatch.setattr(capture_module, "KINECT2_AVAILABLE", True)
    monkeypatch.setattr(capture_module, "Kinect2Capture", raise_during_construction)
    capture = CameraCapture({"backend": "kinect2"})

    assert capture._connect_kinect2() is False
    assert capture._kinect2 is None
    assert capture._error_count == 1
    assert capture._last_error == "Kinect SDK initialization failed"


def test_kinect_camera_info_and_data_helpers_delegate_only_when_connected() -> None:
    color = _frame(1)
    depth = np.full((2, 3), 1200.0)
    point = (1.25, 2.5, 1200.0)
    calibration = {"fx": 100.0, "fy": 101.0}

    class ConnectedKinect:
        def is_connected(self) -> bool:
            return True

        def get_device_info(self) -> dict[str, Any]:
            return {"color_resolution": "720p", "depth_enabled": False}

        def get_latest_frame(self) -> Any:
            return SimpleNamespace(color=color, depth=depth)

        def depth_to_3d(self, u: int, v: int, z: float) -> tuple[float, float, float]:
            assert (u, v, z) == (4, 5, 1200.0)
            return point

        def get_calibration_data(self) -> dict[str, float]:
            return calibration

    capture = CameraCapture(
        {"backend": "kinect2", "kinect2": {"min_depth": 400, "max_depth": 3500}}
    )
    kinect = ConnectedKinect()
    capture._kinect2 = cast("Any", kinect)

    info = capture.get_camera_info()
    assert info["device_id"] == "kinect2"
    assert info["depth_range"] == "400-3500mm"
    assert info["color_resolution"] == "720p"
    assert info["depth_enabled"] is False
    assert capture.get_depth_frame() is depth
    assert capture.get_3d_point(4, 5, 1200.0) == point
    assert capture.get_kinect_calibration() is calibration

    kinect.is_connected = lambda: False  # type: ignore[method-assign]
    assert capture.get_camera_info() == {}


def test_kinect_device_info_exception_returns_empty_info() -> None:
    class BrokenKinect:
        def is_connected(self) -> bool:
            return True

        def get_device_info(self) -> dict[str, Any]:
            raise OSError("device information unavailable")

    capture = CameraCapture({"backend": "kinect2"})
    capture._kinect2 = cast("Any", BrokenKinect())

    assert capture.get_camera_info() == {}


def test_kinect_missed_color_frame_recovers_and_publishes_converted_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :] = [10, 20, 30]

    class IntermittentKinect:
        def __init__(self) -> None:
            self.frames = iter(
                (SimpleNamespace(color=None), SimpleNamespace(color=rgb))
            )

        def is_connected(self) -> bool:
            return True

        def get_latest_frame(self) -> Any:
            return next(self.frames)

    capture = CameraCapture(
        {
            "backend": "kinect2",
            "auto_reconnect": True,
            "reconnect_delay": 0,
            "fps": 30,
        }
    )
    capture._kinect2 = cast("Any", IntermittentKinect())
    statuses: list[CameraStatus] = []
    capture.set_status_callback(statuses.append)
    sleep_count = 0

    def stop_after_successful_frame(_seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            capture._stop_event.set()

    monkeypatch.setattr(capture_module.time, "sleep", stop_after_successful_frame)
    capture._capture_loop()

    delivered = capture.get_frame()
    assert delivered is not None
    frame, metadata = delivered
    assert np.all(frame == [30, 20, 10])
    assert metadata.frame_number == 1
    assert capture._error_count == 1
    assert capture._last_error == "Kinect v2 frame capture failed"
    assert capture.get_status() is CameraStatus.CONNECTED
    assert statuses == [CameraStatus.RECONNECTING, CameraStatus.CONNECTED]
