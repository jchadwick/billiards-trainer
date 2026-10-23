"""Hardware-independent tests for camera acquisition and freshness behavior."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np
import pytest

from backend.perception.camera_source import resolve_configured_camera_source
from backend.vision import capture as capture_module
from backend.vision.capture import CameraCapture, CameraStatus, FrameInfo


class FakeVideoCapture:
    """Small OpenCV capture double with controllable reads and properties."""

    def __init__(
        self,
        reads: list[tuple[bool, np.ndarray | None]] | None = None,
        *,
        opened: bool = True,
        properties: dict[int, float] | None = None,
        fail_get: bool = False,
        fail_set: bool = False,
        after_read: Callable[[bool, np.ndarray | None], None] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = deque(reads or [])
        self.properties = dict(properties or {})
        self.fail_get = fail_get
        self.fail_set = fail_set
        self.after_read = after_read
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV method name
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.reads:
            result = self.reads.popleft()
        else:
            result = (False, None)
        if result[0]:
            position = capture_module.cv2.CAP_PROP_POS_FRAMES
            self.properties[position] = self.properties.get(position, 0.0) + 1
        if self.after_read is not None:
            self.after_read(*result)
        return result

    def set(self, property_id: int, value: float) -> bool:
        if self.fail_set:
            raise RuntimeError("property write failed")
        self.set_calls.append((property_id, value))
        self.properties[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        if self.fail_get:
            raise RuntimeError("property read failed")
        return self.properties.get(property_id, 0.0)

    def release(self) -> None:
        self.released = True
        self.opened = False


def make_frame(value: int) -> np.ndarray:
    return np.full((3, 5, 3), value, dtype=np.uint8)


def assert_capture_thread_stopped(capture: CameraCapture) -> None:
    """Assert stop_capture clears the worker reference after joining it."""
    assert capture._capture_thread is None


def test_stream_source_overrides_the_legacy_recorded_video_path() -> None:
    stream_uri = "http://127.0.0.1:8080/stream"

    assert (
        resolve_configured_camera_source(stream_uri, "assets/demo3.mp4") == stream_uri
    )


def test_local_device_keeps_legacy_recorded_video_fallback() -> None:
    assert resolve_configured_camera_source(0, "assets/demo3.mp4") == "assets/demo3.mp4"
    assert resolve_configured_camera_source(0, "") == 0


def test_backend_names_map_to_opencv_constants_and_unknown_falls_back() -> None:
    expected = {
        "auto": capture_module.cv2.CAP_ANY,
        "v4l2": capture_module.cv2.CAP_V4L2,
        "dshow": capture_module.cv2.CAP_DSHOW,
        "gstreamer": capture_module.cv2.CAP_GSTREAMER,
        "opencv": capture_module.cv2.CAP_OPENCV_MJPEG,
        "kinect2": -1,
    }

    for backend, code in expected.items():
        assert CameraCapture({"backend": backend})._get_opencv_backend() == code
    assert CameraCapture({"backend": "unknown"})._get_opencv_backend() == (
        capture_module.cv2.CAP_ANY
    )


def test_camera_configuration_applies_requested_manual_controls() -> None:
    capture = CameraCapture(
        {
            "resolution": [640, 480],
            "fps": 60,
            "buffer_size": 2,
            "control_policy": "require",
            "exposure_mode": "manual",
            "exposure_value": -6,
            "gain": 1.5,
        }
    )
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]

    assert capture._configure_camera() is True
    assert camera.set_calls == [
        (capture_module.cv2.CAP_PROP_FRAME_WIDTH, 640),
        (capture_module.cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (capture_module.cv2.CAP_PROP_FPS, 60),
        (capture_module.cv2.CAP_PROP_BUFFERSIZE, 2),
        (capture_module.cv2.CAP_PROP_AUTO_EXPOSURE, 0),
        (capture_module.cv2.CAP_PROP_EXPOSURE, -6),
        (capture_module.cv2.CAP_PROP_GAIN, 1.5),
    ]


def test_camera_configuration_sets_auto_exposure_and_skips_manual_value() -> None:
    capture = CameraCapture(
        {
            "control_policy": "require",
            "exposure_mode": "auto",
            "exposure_value": -4,
        }
    )
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]

    assert capture._configure_camera() is True
    assert (capture_module.cv2.CAP_PROP_AUTO_EXPOSURE, 1) in camera.set_calls
    assert (capture_module.cv2.CAP_PROP_EXPOSURE, -4) not in camera.set_calls


def test_camera_configuration_uses_driver_controls_by_default() -> None:
    capture = CameraCapture(
        {
            "resolution": [640, 480],
            "fps": 30,
            "buffer_size": 1,
            "exposure_mode": "manual",
            "exposure_value": -6,
            "gain": 1.5,
        }
    )
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]

    assert capture._configure_camera() is True
    assert camera.set_calls == [
        (capture_module.cv2.CAP_PROP_FRAME_WIDTH, 640),
        (capture_module.cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (capture_module.cv2.CAP_PROP_FPS, 30),
        (capture_module.cv2.CAP_PROP_BUFFERSIZE, 1),
    ]


@pytest.mark.parametrize("policy", [None, "", "automatic", 1, []])
def test_camera_control_policy_must_be_explicitly_supported(policy: object) -> None:
    config = {"control_policy": policy}
    with pytest.raises(ValueError, match="control_policy"):
        CameraCapture(config)


def test_camera_configuration_rejects_unopened_device() -> None:
    capture = CameraCapture({})
    capture._cap = FakeVideoCapture(opened=False)  # type: ignore[assignment]

    assert capture._configure_camera() is False


def test_video_file_keeps_native_camera_settings() -> None:
    capture = CameraCapture({"device_id": "recorded.mp4"})
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]

    assert capture._configure_camera() is True
    assert camera.set_calls == []


def test_http_mjpeg_source_skips_camera_controls_and_uses_latest_frame_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeVideoCapture()
    opened: list[tuple[object, int]] = []

    def open_source(device: object, backend: int) -> FakeVideoCapture:
        opened.append((device, backend))
        return camera

    monkeypatch.setattr(capture_module.cv2, "VideoCapture", open_source)
    capture = CameraCapture(
        {
            "device_id": "http://127.0.0.1:8080/stream",
            "backend": "v4l2",
            "resolution": [1920, 1080],
            "fps": 60,
            "buffer_size": 8,
        }
    )

    assert capture._is_stream_uri is True
    assert capture._is_video_file is False
    assert capture._frame_queue.maxsize == 1
    assert capture._connect_camera() is True
    assert opened == [("http://127.0.0.1:8080/stream", capture_module.cv2.CAP_ANY)]
    assert camera.set_calls == [(capture_module.cv2.CAP_PROP_BUFFERSIZE, 1)]


@pytest.mark.parametrize(
    "source",
    [
        "http://user:pass@camera.test/stream",
        "https://camera.test/stream#frame",
        "http://camera.test:invalid/stream",
    ],
)
def test_http_mjpeg_source_rejects_unsafe_or_malformed_uri(source: str) -> None:
    with pytest.raises(ValueError, match="camera source URI"):
        CameraCapture({"device_id": source})


def test_camera_configuration_failure_is_visible_in_health() -> None:
    capture = CameraCapture({})
    capture._cap = FakeVideoCapture(fail_get=True)  # type: ignore[assignment]

    assert capture._configure_camera() is False
    assert capture._error_count == 1
    assert capture._last_error == "property read failed"


@pytest.mark.parametrize("buffer_size", [0, -1, 1.5, True])
def test_camera_buffer_must_be_a_positive_integer(buffer_size: object) -> None:
    with pytest.raises(ValueError, match="buffer_size must be a positive integer"):
        CameraCapture({"buffer_size": buffer_size})


def test_video_connection_seeks_without_consuming_the_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeVideoCapture(
        reads=[(True, make_frame(4))],
        properties={capture_module.cv2.CAP_PROP_POS_FRAMES: 0},
    )
    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        lambda device, backend: camera,
    )
    capture = CameraCapture({"device_id": "recorded.mp4", "video_start_frame": 12})

    assert capture._connect_camera() is True
    assert (capture_module.cv2.CAP_PROP_POS_FRAMES, 12) in camera.set_calls
    assert capture._connection_attempts == 1
    assert capture._error_count == 0
    assert len(camera.reads) == 1


def test_live_camera_connection_preserves_probe_frame_for_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_frame = make_frame(5)
    next_frame = make_frame(6)
    camera = FakeVideoCapture(
        reads=[(True, probe_frame), (True, next_frame), (False, None)]
    )
    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        lambda device, backend: camera,
    )
    capture = CameraCapture(
        {"device_id": 0, "fps": 0, "auto_reconnect": False, "buffer_size": 2}
    )

    assert capture._connect_camera() is True
    capture._capture_loop()

    delivered = capture.get_frame()
    assert delivered is not None
    frame, _info = delivered
    assert np.array_equal(frame, probe_frame)


def test_video_replay_delivers_probe_frame_at_configured_end_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_frame = make_frame(12)
    camera = FakeVideoCapture(
        reads=[(True, initial_frame), (False, None)],
        properties={capture_module.cv2.CAP_PROP_POS_FRAMES: 0},
    )
    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        lambda device, backend: camera,
    )
    capture = CameraCapture(
        {
            "device_id": "recorded.mp4",
            "video_start_frame": 12,
            "video_end_frame": 13,
            "fps": 0,
        }
    )

    assert capture._connect_camera() is True
    capture._capture_loop()

    delivered = capture.get_frame()
    assert delivered is not None
    frame, info = delivered
    assert info.frame_number == 1
    assert np.array_equal(frame, initial_frame)
    assert capture.get_status() == CameraStatus.DISCONNECTED


def test_failed_camera_open_releases_capture_and_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeVideoCapture(opened=False)
    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        lambda device, backend: camera,
    )
    capture = CameraCapture({"device_id": 0})

    assert capture._connect_camera() is False
    assert camera.released is True
    assert capture._cap is None
    assert capture._connection_attempts == 1
    assert capture._error_count == 1
    assert "Failed to open camera" in (capture._last_error or "")


def test_live_camera_connection_does_not_start_a_competing_probe_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = FakeVideoCapture(reads=[(True, make_frame(1))])
    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        lambda device, backend: camera,
    )
    capture = CameraCapture({})

    assert capture._connect_camera() is True
    assert len(camera.reads) == 1
    assert camera.released is False


def test_get_frame_preserves_fifo_order_and_reports_empty_queue() -> None:
    capture = CameraCapture({"buffer_size": 2})
    first = (make_frame(1), FrameInfo(1, 1.0, (5, 3), 3))
    second = (make_frame(2), FrameInfo(2, 2.0, (5, 3), 3))
    capture._frame_queue.put_nowait(first)
    capture._frame_queue.put_nowait(second)

    assert capture.get_frame() is first
    assert capture.get_frame() is second
    assert capture.get_frame() is None


def test_capture_loop_drops_oldest_frame_and_keeps_latest_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_frame = make_frame(1)
    latest_frame = make_frame(2)
    camera = FakeVideoCapture(
        reads=[(True, first_frame), (True, latest_frame), (False, None)]
    )
    capture = CameraCapture(
        {
            "device_id": "recorded.mp4",
            "buffer_size": 1,
            "fps": 0,
            "frame_log_interval": 100,
        }
    )
    capture._cap = camera  # type: ignore[assignment]

    wall_times = iter((10.0, 10.1, 10.2))
    monkeypatch.setattr(capture_module.time, "time", lambda: next(wall_times))

    capture._capture_loop()

    queued = capture.get_frame()
    assert queued is not None
    frame, info = queued
    assert info.frame_number == 2
    assert info.timestamp == 10.1
    assert np.all(frame == 2)
    assert capture._frames_captured == 2
    assert capture._frames_dropped == 1
    assert capture._fps_tracker == [10.0, 10.1]


def test_capture_timestamp_is_taken_after_frame_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall_time = [100.0]

    def advance_clock(_success: bool, _frame: np.ndarray | None) -> None:
        wall_time[0] += 86_400.0

    camera = FakeVideoCapture(
        reads=[(True, make_frame(3)), (False, None)],
        after_read=advance_clock,
    )
    capture = CameraCapture(
        {
            "device_id": "recorded.mp4",
            "fps": 0,
        }
    )
    capture._cap = camera  # type: ignore[assignment]
    monkeypatch.setattr(capture_module.time, "time", lambda: wall_time[0])

    capture._capture_loop()

    delivered = capture.get_frame()
    assert delivered is not None
    _frame, info = delivered
    assert info.timestamp == 86_500.0


def test_non_video_read_failure_disables_capture_and_updates_status() -> None:
    capture = CameraCapture({"auto_reconnect": False, "fps": 0})
    capture._cap = FakeVideoCapture(reads=[(False, None)])  # type: ignore[assignment]
    statuses: list[CameraStatus] = []
    capture.set_status_callback(statuses.append)

    capture._capture_loop()

    assert capture._error_count == 1
    assert capture._last_error == "Frame capture failed"
    assert capture.get_status() is CameraStatus.ERROR
    assert statuses == [CameraStatus.ERROR]


def test_successful_read_restores_connected_status_after_transient_failure() -> None:
    capture: CameraCapture

    camera = FakeVideoCapture(
        reads=[(False, None), (True, make_frame(7))],
    )
    capture = CameraCapture({"auto_reconnect": True, "reconnect_delay": 0, "fps": 0})
    capture._cap = camera  # type: ignore[assignment]
    statuses: list[CameraStatus] = []

    def stop_after_recovery(status: CameraStatus) -> None:
        statuses.append(status)
        if status is CameraStatus.CONNECTED:
            capture._stop_event.set()

    capture.set_status_callback(stop_after_recovery)

    capture._capture_loop()

    assert capture._error_count == 1
    assert capture._frames_captured == 1
    assert capture.get_status() is CameraStatus.CONNECTED
    assert statuses == [CameraStatus.RECONNECTING, CameraStatus.CONNECTED]


def test_start_is_idempotent_and_stop_releases_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeThread:
        def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False
            self.alive = False
            created_threads.append(self)

        def start(self) -> None:
            self.started = True
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.alive = False

    camera = FakeVideoCapture()
    capture = CameraCapture({})

    def connect_camera() -> bool:
        capture._cap = camera  # type: ignore[assignment]
        return True

    capture._connect_camera = connect_camera  # type: ignore[method-assign]
    stale_frame = (make_frame(9), FrameInfo(9, 9.0, (5, 3), 3))
    capture._frame_queue.put_nowait(stale_frame)
    created_threads: list[FakeThread] = []
    monkeypatch.setattr(capture_module.threading, "Thread", FakeThread)

    assert capture.start_capture() is True
    thread = capture._capture_thread
    assert thread is not None
    assert len(created_threads) == 1
    assert created_threads[0].started
    assert capture.get_status() is CameraStatus.CONNECTED
    assert capture.get_frame() is None
    assert capture.start_capture() is True
    assert capture._capture_thread is thread

    capture.stop_capture()

    assert capture._stop_event.is_set()
    assert camera.released is True
    assert capture._cap is None
    assert capture.get_status() is CameraStatus.DISCONNECTED


def test_kinect_status_callbacks_do_not_deadlock_camera_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeThread:
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

    class FakeKinectCapture:
        def __init__(self) -> None:
            self.status_callback: Callable[[Any], None] | None = None

        def set_status_callback(self, callback: Callable[[Any], None]) -> None:
            self.status_callback = callback

        def start_capture(self) -> bool:
            assert self.status_callback is not None
            self.status_callback(capture_module.Kinect2Status.CONNECTED)
            return True

        def stop_capture(self) -> None:
            assert self.status_callback is not None
            self.status_callback(capture_module.Kinect2Status.DISCONNECTED)

    kinect = FakeKinectCapture()
    monkeypatch.setattr(capture_module, "KINECT2_AVAILABLE", True)
    monkeypatch.setattr(capture_module, "Kinect2Capture", lambda _config: kinect)
    monkeypatch.setattr(capture_module.threading, "Thread", FakeThread)
    capture = CameraCapture({"backend": "kinect2"})

    assert capture.start_capture() is True
    assert capture.get_status() is CameraStatus.CONNECTED
    capture.stop_capture()

    assert capture.get_status() is CameraStatus.DISCONNECTED


def test_capture_start_failure_leaves_camera_in_error_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = CameraCapture({})
    capture._connect_camera = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr(
        capture_module.threading,
        "Thread",
        lambda **kwargs: pytest.fail("thread must not start after connection failure"),
    )

    assert capture.start_capture() is False
    assert capture.get_status() is CameraStatus.ERROR
    assert capture._capture_thread is None


def test_stop_releases_source_after_capture_thread_reaches_natural_end() -> None:
    class FinishedThread:
        def is_alive(self) -> bool:
            return False

    capture = CameraCapture({})
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]
    capture._capture_thread = FinishedThread()  # type: ignore[assignment]

    capture.stop_capture()

    assert camera.released is True
    assert capture._cap is None
    assert capture._capture_thread is None
    assert capture.get_status() is CameraStatus.DISCONNECTED


def test_stop_joins_worker_without_holding_status_lock() -> None:
    import threading

    capture = CameraCapture({"stop_timeout": 1.0})
    capture._update_status(CameraStatus.CONNECTED)
    worker_finished = threading.Event()
    status_updates: list[CameraStatus] = []
    capture.set_status_callback(status_updates.append)

    def worker() -> None:
        capture._stop_event.wait(timeout=1.0)
        capture._update_status(CameraStatus.ERROR)
        worker_finished.set()

    thread = threading.Thread(target=worker)
    capture._capture_thread = thread
    thread.start()

    capture.stop_capture()

    assert worker_finished.is_set()
    assert not thread.is_alive()
    assert_capture_thread_stopped(capture)

    def assert_expected_status_updates(actual: list[CameraStatus]) -> None:
        assert actual == [CameraStatus.ERROR, CameraStatus.DISCONNECTED]

    assert_expected_status_updates(status_updates)


def test_stop_timeout_retains_native_source_until_blocked_read_exits() -> None:
    import threading

    class BlockingVideoCapture(FakeVideoCapture):
        def __init__(self) -> None:
            super().__init__()
            self.read_started = threading.Event()
            self.allow_read_to_finish = threading.Event()
            self.read_active = False
            self.released_during_read = False
            self.release_count = 0

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.read_active = True
            self.read_started.set()
            try:
                assert self.allow_read_to_finish.wait(timeout=2.0)
                return True, make_frame(11)
            finally:
                self.read_active = False

        def release(self) -> None:
            self.released_during_read = self.read_active
            self.release_count += 1
            super().release()

    camera = BlockingVideoCapture()
    capture = CameraCapture({"stop_timeout": 0.01, "fps": 0, "auto_reconnect": False})
    connection_count = 0

    def connect_blocking_source() -> bool:
        nonlocal connection_count
        connection_count += 1
        capture._cap = camera  # type: ignore[assignment]
        return True

    capture._connect_camera = connect_blocking_source  # type: ignore[method-assign]
    try:
        assert capture.start_capture() is True
        assert camera.read_started.wait(timeout=1.0)

        worker = capture._capture_thread
        assert worker is not None
        capture.stop_capture()

        assert worker.is_alive()
        assert capture._stop_event.is_set()
        assert capture._capture_thread is worker
        assert capture._cap is camera  # type: ignore[comparison-overlap]
        assert camera.release_count == 0
        assert capture.get_status() is CameraStatus.ERROR
        assert "source remains owned" in (capture._last_error or "")
        assert capture.start_capture() is False
        assert connection_count == 1

        # A repeated stop must preserve the same owner while the backend read
        # is still active, without releasing or reopening the native source.
        capture.stop_capture()
        assert capture._capture_thread is worker
        assert capture._cap is camera  # type: ignore[comparison-overlap]
        assert camera.release_count == 0
        assert capture.start_capture() is False
        assert connection_count == 1

        camera.allow_read_to_finish.set()
        worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert camera.released is True
        assert camera.release_count == 1
        assert camera.released_during_read is False
        assert capture._cap is None
        assert capture.get_status() is CameraStatus.DISCONNECTED
        assert capture.get_frame() is None
        assert capture.get_health().frames_captured == 0

        capture.stop_capture()
        assert_capture_thread_stopped(capture)
        assert camera.release_count == 1
    finally:
        # Failed assertions must also unblock and join the real worker; leaving
        # a daemon read behind could race other tests or hide lifecycle leaks.
        camera.allow_read_to_finish.set()
        worker_to_clean_up = capture._capture_thread
        if worker_to_clean_up is not None:
            worker_to_clean_up.join(timeout=1.0)
        capture.stop_capture()


def test_status_callback_exception_does_not_interrupt_state_transition() -> None:
    capture = CameraCapture({})

    def broken_callback(_status: CameraStatus) -> None:
        raise RuntimeError("observer failed")

    capture.set_status_callback(broken_callback)

    capture._update_status(CameraStatus.CONNECTED)

    assert capture.get_status() is CameraStatus.CONNECTED


def test_camera_health_reports_fps_and_capture_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = CameraCapture({})
    capture._start_time = 90.0
    capture._fps_tracker = [100.0, 100.1, 100.2]
    capture._frames_captured = 14
    capture._frames_dropped = 3
    capture._last_frame_time = 100.2
    capture._error_count = 2
    capture._last_error = "last read failed"
    capture._connection_attempts = 4
    monkeypatch.setattr(capture_module.time, "time", lambda: 101.0)

    health = capture.get_health()

    assert health.status is CameraStatus.DISCONNECTED
    assert health.frames_captured == 14
    assert health.frames_dropped == 3
    assert health.fps == pytest.approx(10.0)
    assert health.last_frame_time == 100.2
    assert health.error_count == 2
    assert health.last_error == "last read failed"
    assert health.connection_attempts == 4
    assert health.uptime == 11.0


def test_camera_info_reports_actual_driver_settings() -> None:
    capture = CameraCapture({"device_id": 2, "backend": "v4l2"})
    properties = {
        capture_module.cv2.CAP_PROP_FRAME_WIDTH: 1280,
        capture_module.cv2.CAP_PROP_FRAME_HEIGHT: 720,
        capture_module.cv2.CAP_PROP_FPS: 60,
        capture_module.cv2.CAP_PROP_EXPOSURE: -5,
        capture_module.cv2.CAP_PROP_GAIN: 2,
        capture_module.cv2.CAP_PROP_BRIGHTNESS: 0.2,
        capture_module.cv2.CAP_PROP_CONTRAST: 0.8,
        capture_module.cv2.CAP_PROP_SATURATION: 0.7,
        capture_module.cv2.CAP_PROP_BUFFERSIZE: 2,
    }
    capture._cap = FakeVideoCapture(properties=properties)  # type: ignore[assignment]

    info = capture.get_camera_info()

    assert info["resolution"] == (1280, 720)
    assert info["fps"] == 60
    assert info["exposure"] == -5
    assert info["buffer_size"] == 2
    assert info["backend"] == "v4l2"


def test_camera_info_is_empty_when_not_open_or_backend_query_fails() -> None:
    capture = CameraCapture({})
    assert capture.get_camera_info() == {}

    capture._cap = FakeVideoCapture(fail_get=True)  # type: ignore[assignment]
    assert capture.get_camera_info() == {}


def test_live_config_update_applies_exposure_fps_and_gain() -> None:
    capture = CameraCapture({})
    camera = FakeVideoCapture()
    capture._cap = camera  # type: ignore[assignment]

    assert (
        capture.update_config(
            {"fps": 50, "exposure_mode": "manual", "exposure_value": -7, "gain": 3}
        )
        is True
    )
    assert capture._fps == 50
    assert capture._exposure_value == -7
    assert capture._gain == 3
    assert camera.set_calls == [
        (capture_module.cv2.CAP_PROP_FPS, 50),
        (capture_module.cv2.CAP_PROP_AUTO_EXPOSURE, 0),
        (capture_module.cv2.CAP_PROP_EXPOSURE, -7),
        (capture_module.cv2.CAP_PROP_GAIN, 3),
    ]


def test_live_config_update_failure_is_reported() -> None:
    capture = CameraCapture({})
    capture._cap = FakeVideoCapture(fail_set=True)  # type: ignore[assignment]

    assert capture.update_config({"fps": 50}) is False
    assert capture._error_count == 1
    assert capture._last_error == "property write failed"


def test_frame_info_records_post_read_software_timestamp() -> None:
    info = FrameInfo(1, 123.0, (5, 3), 3)

    assert info.timestamp == 123.0
