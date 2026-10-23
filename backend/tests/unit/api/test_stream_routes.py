"""Contract tests for HTTP camera streaming routes and latest-frame delivery.

These tests use fake camera/shared-memory sources so validation, freshness, and
cleanup behavior can be checked without opening physical capture hardware.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, AsyncGenerator, cast
from unittest.mock import AsyncMock, Mock, call

import numpy as np
import pytest
from fastapi import HTTPException

import backend.api.routes.stream as stream_route
from backend.vision.capture import CameraStatus


class _Config:
    """Small dotted-path config fake with the same lookup contract as Config."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, path: str, default: Any = None) -> Any:
        return self.values.get(path, default)


class _FrameRequest:
    def __init__(self, disconnect_after: int) -> None:
        self.disconnect_after = disconnect_after
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks >= self.disconnect_after


class _SharedMemoryReader:
    def __init__(
        self,
        frames: list[tuple[np.ndarray | None, SimpleNamespace | None]],
        *,
        attach_error: Exception | None = None,
    ) -> None:
        self.frames = deque(frames)
        self.attach_error = attach_error
        self.attached_timeout: float | None = None
        self.detached = False
        self.read_count = 0

    def attach(self, *, timeout: float) -> None:
        self.attached_timeout = timeout
        if self.attach_error:
            raise self.attach_error

    def read_frame(self) -> tuple[np.ndarray | None, SimpleNamespace | None]:
        self.read_count += 1
        if self.frames:
            return self.frames.popleft()
        return None, None

    def detach(self) -> None:
        self.detached = True


class _Camera:
    def __init__(self, *, connected: bool = True, frame: np.ndarray | None = None):
        self.connected = connected
        self.frame = frame
        self.frame_calls: list[bool] = []

    def is_connected(self) -> bool:
        return self.connected

    def get_status(self) -> CameraStatus:
        return CameraStatus.CONNECTED if self.connected else CameraStatus.DISCONNECTED

    def get_current_frame(self, *, raw: bool = False) -> np.ndarray | None:
        self.frame_calls.append(raw)
        return self.frame


class _EnhancedCamera:
    def __init__(
        self,
        config: Any,
        *,
        event_loop: Any,
        frame_callback: Any,
        starts: bool = True,
    ) -> None:
        self.config = config
        self.event_loop = event_loop
        self.frame_callback = frame_callback
        self.starts = starts
        self.running = False
        self.stopped = False

    def start_capture(self) -> bool:
        self.running = self.starts
        return self.starts

    def stop_capture(self) -> None:
        self.running = False
        self.stopped = True

    def get_statistics(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "fps": 23.5,
            "resolution": (640, 480),
            "fisheye_correction_enabled": True,
            "preprocessing_enabled": True,
        }

    def get_frame(self, *, processed: bool) -> np.ndarray:
        del processed
        return np.zeros((480, 640, 3), dtype=np.uint8)


class _LiveStreamCamera:
    def __init__(
        self,
        frames: list[np.ndarray | None],
        *,
        connected: bool = True,
    ) -> None:
        self.frames = deque(frames)
        self.connected = connected
        self.frame_requests: list[tuple[float, bool]] = []

    def is_connected(self) -> bool:
        return self.connected

    def get_frame_for_streaming(self, *, scale: float, raw: bool) -> np.ndarray | None:
        self.frame_requests.append((scale, raw))
        return self.frames.popleft() if self.frames else None


@pytest.fixture
def clean_stream_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent module-level stream counters leaking between route tests."""
    monkeypatch.setattr(stream_route, "_streaming_clients", set())
    monkeypatch.setattr(
        stream_route,
        "_stream_stats",
        {
            "active_streams": 0,
            "total_frames_served": 0,
            "start_time": 1.0,
            "last_frame_time": 0.0,
        },
    )


def _stream_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "quality": {
            "default_jpeg_quality": 81,
            "single_frame_jpeg_quality": 91,
            "min_jpeg_quality": 10,
            "max_jpeg_quality": 95,
        },
        "framerate": {"default_fps": 24, "min_fps": 2, "max_fps": 50},
        "resolution": {
            "min_width": 16,
            "max_width": 4096,
            "min_height": 12,
            "max_height": 2160,
        },
    }
    config.update(overrides)
    return config


async def _unused_stream(**_kwargs: Any) -> AsyncGenerator[bytes, None]:
    """Provide a real async iterable for responses whose body stays unconsumed."""
    yield b""


@pytest.mark.asyncio
async def test_shared_memory_route_uses_config_defaults_and_no_cache_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SimpleNamespace(
        config_module=_Config({"api.stream": _stream_config()}),
        vision_module=None,
    )
    requested: dict[str, Any] = {}

    def capture_stream(**kwargs: Any) -> AsyncGenerator[bytes, None]:
        requested.update(kwargs)
        return _unused_stream(**kwargs)

    monkeypatch.setattr(stream_route, "generate_mjpeg_stream_from_shm", capture_stream)

    response = await stream_route.video_stream_shm(
        request=cast("Any", _FrameRequest(1)),
        app_state=cast("Any", app_state),
        quality=None,
        fps=None,
    )

    assert isinstance(response, stream_route.StreamingResponse)
    assert requested["quality"] == 81
    assert requested["fps"] == 24
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert response.headers["connection"] == "close"
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quality", "fps", "detail"),
    [
        (9, 24, "Quality must be between 10 and 95"),
        (96, 24, "Quality must be between 10 and 95"),
        (81, 1, "FPS must be between 2 and 50"),
        (81, 51, "FPS must be between 2 and 50"),
    ],
)
async def test_shared_memory_route_rejects_out_of_range_delivery_settings(
    quality: int,
    fps: int,
    detail: str,
) -> None:
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    with pytest.raises(HTTPException) as error:
        await stream_route.video_stream_shm(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            quality=quality,
            fps=fps,
        )

    assert error.value.status_code == 400
    assert error.value.detail == detail


@pytest.mark.asyncio
async def test_non_raw_route_ignores_unsupported_dimensions_and_uses_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))
    requested: dict[str, Any] = {}

    def capture_stream(**kwargs: Any) -> AsyncGenerator[bytes, None]:
        requested.update(kwargs)
        return _unused_stream(**kwargs)

    monkeypatch.setattr(stream_route, "generate_mjpeg_stream_from_shm", capture_stream)

    response = await stream_route.video_stream(
        request=cast("Any", _FrameRequest(1)),
        app_state=cast("Any", app_state),
        quality=None,
        fps=None,
        width=99999,
        height=99999,
    )

    assert requested["quality"] == 81
    assert requested["fps"] == 24
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"


@pytest.mark.asyncio
async def test_video_route_maps_stream_setup_failure_to_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    def fail_stream_setup(**_kwargs: Any) -> AsyncGenerator[bytes, None]:
        raise RuntimeError("shared memory setup failed")

    monkeypatch.setattr(
        stream_route, "generate_mjpeg_stream_from_shm", fail_stream_setup
    )

    with pytest.raises(HTTPException) as error:
        await stream_route.video_stream(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            quality=None,
            fps=None,
            width=None,
            height=None,
        )

    assert error.value.status_code == 500
    assert (
        error.value.detail
        == "Unable to initialize video stream: shared memory setup failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quality", "fps", "expected"),
    [
        (9, 24, "Quality must be between 10 and 95"),
        (81, 51, "FPS must be between 2 and 50"),
    ],
)
async def test_video_route_preserves_query_validation_errors(
    quality: int,
    fps: int,
    expected: str,
) -> None:
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    with pytest.raises(HTTPException) as error:
        await stream_route.video_stream(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            quality=quality,
            fps=fps,
            width=None,
            height=None,
        )

    assert error.value.status_code == 400
    assert error.value.detail == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quality", "fps", "width", "height", "expected"),
    [
        (9, 24, None, None, "Quality must be between 10 and 95"),
        (81, 1, None, None, "FPS must be between 2 and 50"),
        (81, 24, 15, None, "Width must be between 16 and 4096"),
        (81, 24, None, 11, "Height must be between 12 and 2160"),
    ],
)
async def test_raw_stream_route_validates_query_ranges_before_camera_use(
    quality: int,
    fps: int,
    width: int | None,
    height: int | None,
    expected: str,
) -> None:
    camera = _Camera(connected=False)
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    with pytest.raises(HTTPException) as error:
        await stream_route.video_stream_raw(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            vision_module=cast("Any", camera),
            quality=quality,
            fps=fps,
            width=width,
            height=height,
        )

    assert error.value.status_code == 400
    assert error.value.detail == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CameraStatus.DISCONNECTED, CameraStatus.ERROR])
async def test_raw_stream_route_refuses_unavailable_camera(
    status: CameraStatus,
) -> None:
    camera = _Camera(connected=False)
    camera.get_status = Mock(return_value=status)  # type: ignore[method-assign]
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    with pytest.raises(HTTPException) as error:
        await stream_route.video_stream_raw(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            vision_module=cast("Any", camera),
            quality=None,
            fps=None,
            width=None,
            height=None,
        )

    assert error.value.status_code == 503
    assert camera.frame_calls == []


@pytest.mark.asyncio
async def test_raw_stream_route_wires_raw_mode_and_configured_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _Camera()
    app_state = SimpleNamespace(
        config_module=_Config({"api.stream": _stream_config()}),
        vision_module=None,
    )
    requested: dict[str, Any] = {}

    def capture_stream(**kwargs: Any) -> AsyncGenerator[bytes, None]:
        requested.update(kwargs)
        return _unused_stream(**kwargs)

    monkeypatch.setattr(stream_route, "generate_mjpeg_stream", capture_stream)

    response = await stream_route.video_stream_raw(
        request=cast("Any", _FrameRequest(1)),
        app_state=cast("Any", app_state),
        vision_module=cast("Any", camera),
        quality=None,
        fps=None,
        width=320,
        height=240,
    )

    assert requested["raw"] is True
    assert requested["max_fps"] == 24
    assert requested["quality"] == 81
    assert requested["max_width"] == 320
    assert requested["max_height"] == 240
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"


@pytest.mark.asyncio
async def test_get_vision_module_lazily_creates_one_camera_and_broadcasts_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_config = SimpleNamespace(device_id="fake-webcam", resolution=(640, 480))
    monkeypatch.setattr(
        stream_route.EnhancedCameraConfig,
        "from_config",
        staticmethod(lambda _config: camera_config),
    )
    created: list[_EnhancedCamera] = []

    def create_camera(
        config: Any,
        *,
        event_loop: Any,
        frame_callback: Any,
    ) -> _EnhancedCamera:
        module = _EnhancedCamera(
            config,
            event_loop=event_loop,
            frame_callback=frame_callback,
        )
        created.append(module)
        return module

    monkeypatch.setattr(stream_route, "EnhancedCameraModule", create_camera)
    broadcast_frame = AsyncMock()
    monkeypatch.setattr(
        stream_route.message_broadcaster, "broadcast_frame", broadcast_frame
    )
    app_state = SimpleNamespace(
        config_module=_Config(
            {"api.stream": {"quality": {"websocket_jpeg_quality": 63}}}
        )
    )

    adapter = await stream_route.get_vision_module(cast("Any", app_state))
    cached_adapter = await stream_route.get_vision_module(cast("Any", app_state))

    assert adapter is cached_adapter
    assert len(created) == 1
    assert adapter.is_connected()
    assert adapter.get_status() == CameraStatus.CONNECTED
    assert adapter.get_camera_info() == {
        "device_id": "fake-webcam",
        "resolution": (640, 480),
        "fps": 23.5,
        "backend": "enhanced",
        "fisheye_correction": True,
        "preprocessing": True,
    }

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await created[0].frame_callback(frame, 640, 480)
    broadcast_frame.assert_awaited_once_with(
        image_data=frame,
        width=640,
        height=480,
        quality=63,
    )


@pytest.mark.asyncio
async def test_rust_provider_rejects_all_legacy_camera_stream_admission_paths() -> None:
    app_state = SimpleNamespace(
        perception_provider="rust",
        config_module=_Config({"api.stream": _stream_config()}),
    )
    routes = (
        lambda: stream_route.get_vision_module(cast("Any", app_state)),
        lambda: stream_route.video_stream(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            quality=None,
            fps=None,
            width=None,
            height=None,
        ),
        lambda: stream_route.video_stream_shm(
            request=cast("Any", _FrameRequest(1)),
            app_state=cast("Any", app_state),
            quality=None,
            fps=None,
        ),
    )

    for route in routes:
        with pytest.raises(HTTPException) as error:
            await route()
        assert error.value.status_code == 503
        assert "while Rust owns live capture" in error.value.detail

    assert not hasattr(app_state, "streaming_camera_module")


@pytest.mark.asyncio
async def test_get_vision_module_reports_camera_start_failure_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_config = SimpleNamespace(device_id=0, resolution=(320, 240))
    monkeypatch.setattr(
        stream_route.EnhancedCameraConfig,
        "from_config",
        staticmethod(lambda _config: camera_config),
    )
    monkeypatch.setattr(
        stream_route,
        "EnhancedCameraModule",
        lambda config, *, event_loop, frame_callback: _EnhancedCamera(
            config,
            event_loop=event_loop,
            frame_callback=frame_callback,
            starts=False,
        ),
    )
    app_state = SimpleNamespace(config_module=_Config({"api.stream": {}}))

    with pytest.raises(HTTPException) as error:
        await stream_route.get_vision_module(cast("Any", app_state))

    assert error.value.status_code == 503
    assert "Camera capture failed to start" in error.value.detail
    assert app_state.streaming_camera_module is None


@pytest.mark.asyncio
async def test_get_vision_module_wraps_missing_configuration_as_service_unavailable() -> (
    None
):
    app_state = SimpleNamespace(config_module=None)

    with pytest.raises(HTTPException) as error:
        await stream_route.get_vision_module(cast("Any", app_state))

    assert error.value.status_code == 503
    assert "Configuration module is not available" in error.value.detail
    assert app_state.streaming_camera_module is None


def test_camera_adapter_forwards_raw_flags_scaling_and_health_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _EnhancedCamera(
        SimpleNamespace(device_id=3, resolution=(640, 480)),
        event_loop=None,
        frame_callback=None,
    )
    camera.running = True
    adapter = stream_route.CameraModuleAdapter(cast("Any", camera))
    resized = np.zeros((240, 320, 3), dtype=np.uint8)
    resize = Mock(return_value=resized)
    monkeypatch.setattr(stream_route.cv2, "resize", resize)

    frame = adapter.get_frame_for_streaming(scale=0.5, raw=True)
    raw_frame = adapter.get_current_frame(raw=True)
    health = adapter.get_health()

    assert frame is resized
    resize.assert_called_once()
    assert resize.call_args.args[1] == (320, 240)
    assert resize.call_args.kwargs["interpolation"] == stream_route.cv2.INTER_AREA
    assert raw_frame is not None
    assert raw_frame.shape == (480, 640, 3)
    assert health.status == CameraStatus.CONNECTED
    assert health.fps == 23.5
    assert adapter.get_statistics() == {
        "is_running": True,
        "frames_processed": 0,
        "frames_dropped": 0,
        "avg_fps": 23.5,
        "avg_processing_time_ms": 0.0,
    }


def test_camera_adapter_handles_disconnected_state_and_unavailable_frame() -> None:
    module = SimpleNamespace(
        running=False,
        config=SimpleNamespace(device_id=1, resolution=(8, 6)),
        get_statistics=Mock(return_value={}),
        get_frame=Mock(return_value=None),
    )
    adapter = stream_route.CameraModuleAdapter(cast("Any", module))

    assert adapter.is_connected() is False
    assert adapter.get_status() == CameraStatus.DISCONNECTED
    assert adapter.get_frame_for_streaming(scale=1.0) is None
    assert adapter.get_current_frame(raw=True) is None
    module.get_frame.assert_has_calls([call(processed=True), call(processed=False)])


@pytest.mark.asyncio
async def test_live_mjpeg_generator_yields_resized_frames(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    source_frame = np.zeros((20, 40, 3), dtype=np.uint8)
    camera = _LiveStreamCamera([source_frame, source_frame])
    resized_frame = np.zeros((5, 10, 3), dtype=np.uint8)
    resize = Mock(return_value=resized_frame)
    encode = Mock(
        side_effect=[
            (True, np.asarray([65], dtype=np.uint8)),
            (True, np.asarray([66], dtype=np.uint8)),
        ]
    )
    monkeypatch.setattr(stream_route.cv2, "resize", resize)
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=72,
        max_fps=0,
        max_width=10,
        max_height=8,
        raw=True,
        stream_config={
            "performance": {"frame_log_interval": 1, "sleep_interval_ms": 0},
            "resolution": {"default_scale": 0.5},
        },
    )

    chunks = [await generator.__anext__(), await generator.__anext__()]
    stream_route._streaming_clients.clear()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert len(chunks) == 2
    assert chunks[0].endswith(b"A\r\n")
    assert chunks[1].endswith(b"B\r\n")
    assert camera.frame_requests == [(0.5, True), (0.5, True)]
    assert resize.call_count == 2
    assert all(call.args[1] == (10, 5) for call in resize.call_args_list)
    assert [call.args[2] for call in encode.call_args_list] == [
        [stream_route.cv2.IMWRITE_JPEG_QUALITY, 72],
        [stream_route.cv2.IMWRITE_JPEG_QUALITY, 72],
    ]
    assert stream_route._stream_stats["total_frames_served"] == 2
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_live_mjpeg_generator_retries_missing_frame_and_encode_failure(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    camera = _LiveStreamCamera([None, frame, frame])
    encode = Mock(
        side_effect=[
            (False, None),
            (True, np.asarray([79], dtype=np.uint8)),
        ]
    )
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=80,
        max_fps=0,
        stream_config={"performance": {"sleep_interval_ms": 0}},
    )

    chunk = await generator.__anext__()
    stream_route._streaming_clients.clear()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert chunk.endswith(b"O\r\n")
    assert len(camera.frame_requests) == 3
    assert encode.call_count == 2
    assert stream_route._stream_stats["total_frames_served"] == 1
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_live_mjpeg_generator_enforces_fps_without_upscaling(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    camera = _LiveStreamCamera([frame, frame])
    encode = Mock(
        return_value=(True, np.asarray([70], dtype=np.uint8)),
    )
    resize = Mock()
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)
    monkeypatch.setattr(stream_route.cv2, "resize", resize)
    wall_times = iter([100.0, 100.01, 100.2])
    monkeypatch.setattr(stream_route.time, "time", lambda: next(wall_times))
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=80,
        max_fps=10,
        max_width=100,
        max_height=100,
        stream_config={
            "performance": {"sleep_interval_ms": 0},
            "resolution": {"default_scale": 1.0},
        },
    )

    first = await generator.__anext__()
    second = await generator.__anext__()
    stream_route._streaming_clients.clear()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert first == second
    assert len(camera.frame_requests) == 2
    resize.assert_not_called()
    assert encode.call_count == 2
    assert stream_route._stream_stats["total_frames_served"] == 2


@pytest.mark.asyncio
async def test_live_mjpeg_generator_recovers_from_transient_camera_read_error(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    camera = SimpleNamespace(
        is_connected=Mock(return_value=True),
        get_frame_for_streaming=Mock(
            side_effect=[RuntimeError("temporary read error"), frame]
        ),
    )
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([82], dtype=np.uint8))),
    )
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=80,
        max_fps=0,
        stream_config={"performance": {"sleep_interval_ms": 0}},
    )

    chunk = await generator.__anext__()
    stream_route._streaming_clients.clear()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert chunk.endswith(b"R\r\n")
    assert camera.get_frame_for_streaming.call_count == 2
    assert stream_route._stream_stats["total_frames_served"] == 1
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_live_mjpeg_generator_disconnects_after_configured_send_backpressure(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    camera = _LiveStreamCamera([frame])
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([83], dtype=np.uint8))),
    )
    send_times = iter([100, 200_000_100])
    monkeypatch.setattr(stream_route.time, "perf_counter_ns", lambda: next(send_times))
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=80,
        max_fps=0,
        stream_config={
            "performance": {
                "sleep_interval_ms": 0,
                "maximum_send_backpressure_seconds": 0.05,
            }
        },
    )

    chunk = await generator.__anext__()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert chunk.endswith(b"S\r\n")
    assert camera.frame_requests == [(1.0, False)]
    assert stream_route._stream_stats["total_frames_served"] == 1
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_live_mjpeg_generator_rejects_disconnected_camera_and_cleans_client(
    clean_stream_globals: None,
) -> None:
    camera = _LiveStreamCamera([], connected=False)
    generator = stream_route.generate_mjpeg_stream(
        vision_module=cast("Any", camera),
        quality=80,
        max_fps=0,
    )

    with pytest.raises(stream_route.StreamingError, match="Camera not connected"):
        await generator.__anext__()

    assert not stream_route._streaming_clients
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_single_frame_route_resizes_preserving_aspect_and_returns_jpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    camera = _Camera(frame=original)
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))
    resize = Mock(return_value=np.zeros((25, 50, 3), dtype=np.uint8))
    encode = Mock(return_value=(True, np.asarray([7, 8, 9], dtype=np.uint8)))
    monkeypatch.setattr(stream_route.cv2, "resize", resize)
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)

    response = await stream_route.get_single_frame(
        app_state=cast("Any", app_state),
        vision_module=cast("Any", camera),
        quality=None,
        width=50,
        height=30,
    )

    resize.assert_called_once()
    assert resize.call_args.args[1] == (50, 25)
    assert camera.frame_calls == [False]
    assert encode.call_args.args[2] == [stream_route.cv2.IMWRITE_JPEG_QUALITY, 91]
    assert response.body == bytes((7, 8, 9))
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"


@pytest.mark.asyncio
async def test_raw_single_frame_route_requests_unprocessed_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _Camera(frame=np.zeros((20, 30, 3), dtype=np.uint8))
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))
    encode = Mock(return_value=(True, np.asarray([3, 2, 1], dtype=np.uint8)))
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)

    response = await stream_route.get_single_frame_raw(
        app_state=cast("Any", app_state),
        vision_module=cast("Any", camera),
        quality=None,
        width=None,
        height=None,
    )

    assert camera.frame_calls == [True]
    assert response.body == bytes((3, 2, 1))
    assert response.media_type == "image/jpeg"
    assert encode.call_args.args[2] == [stream_route.cv2.IMWRITE_JPEG_QUALITY, 91]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connected", "frame", "status", "detail"),
    [
        (False, np.zeros((20, 20, 3), dtype=np.uint8), 503, "Camera is not connected"),
        (True, None, 503, "Camera is not providing frames"),
    ],
)
async def test_single_frame_route_reports_unavailable_camera_or_frame(
    connected: bool,
    frame: np.ndarray | None,
    status: int,
    detail: str,
) -> None:
    camera = _Camera(connected=connected, frame=frame)
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))

    with pytest.raises(HTTPException) as error:
        await stream_route.get_single_frame(
            app_state=cast("Any", app_state),
            vision_module=cast("Any", camera),
            quality=None,
            width=None,
            height=None,
        )

    assert error.value.status_code == status
    assert error.value.detail == detail


@pytest.mark.asyncio
async def test_single_frame_route_surfaces_jpeg_encoding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = _Camera(frame=np.zeros((20, 20, 3), dtype=np.uint8))
    app_state = SimpleNamespace(config_module=_Config({"api.stream": _stream_config()}))
    monkeypatch.setattr(stream_route.cv2, "imencode", Mock(return_value=(False, None)))

    with pytest.raises(HTTPException) as error:
        await stream_route.get_single_frame(
            app_state=cast("Any", app_state),
            vision_module=cast("Any", camera),
            quality=None,
            width=None,
            height=None,
        )

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to encode frame as JPEG"


@pytest.mark.asyncio
async def test_shared_memory_stream_skips_duplicate_frames_and_delivers_latest_available(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    """A slow consumer resumes on the current source frame, not a queued history."""
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    reader = _SharedMemoryReader(
        [
            (frame, SimpleNamespace(frame_number=41, writer_pid=100)),
            (frame, SimpleNamespace(frame_number=41, writer_pid=100)),
            (frame, SimpleNamespace(frame_number=44, writer_pid=100)),
        ]
    )
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", lambda *, name: reader)
    encode = Mock(
        side_effect=[
            (True, np.asarray([41], dtype=np.uint8)),
            (True, np.asarray([44], dtype=np.uint8)),
        ]
    )
    monkeypatch.setattr(stream_route.cv2, "imencode", encode)
    request = _FrameRequest(disconnect_after=4)
    app_state = SimpleNamespace(
        config_module=_Config(
            {
                "video.shared_memory_name": "test-frames",
                "video.shared_memory_attach_timeout_sec": 0.25,
                "api.stream": {"performance": {"sleep_interval_ms": 0}},
            }
        ),
        vision_module=None,
    )

    chunks = [
        chunk
        async for chunk in stream_route.generate_mjpeg_stream_from_shm(
            request=cast("Any", request),
            app_state=cast("Any", app_state),
            quality=80,
            fps=0,
        )
    ]

    assert len(chunks) == 2
    assert b"Content-Length: 1\r\n\r\n)\r\n" in chunks[0]
    assert b"Content-Length: 1\r\n\r\n,\r\n" in chunks[1]
    assert reader.attached_timeout == 0.25
    assert reader.read_count == 3
    assert reader.detached
    assert request.checks == 4
    assert stream_route._stream_stats["total_frames_served"] == 2
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_shared_memory_stream_recovers_after_transient_reader_error(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    class _FlakyReader(_SharedMemoryReader):
        def read_frame(self) -> tuple[np.ndarray | None, SimpleNamespace | None]:
            self.read_count += 1
            if self.read_count == 1:
                raise OSError("temporary shared-memory read failure")
            return self.frames.popleft()

    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    reader = _FlakyReader([(frame, SimpleNamespace(frame_number=8, writer_pid=3))])
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", lambda *, name: reader)
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([83], dtype=np.uint8))),
    )
    request = _FrameRequest(disconnect_after=3)
    app_state = SimpleNamespace(
        config_module=_Config(
            {
                "video.shared_memory_name": "fresh-frames",
                "video.shared_memory_attach_timeout_sec": 0.2,
                "api.stream": {"performance": {"sleep_interval_ms": 0}},
            }
        ),
        vision_module=None,
    )

    chunks = [
        chunk
        async for chunk in stream_route.generate_mjpeg_stream_from_shm(
            request=cast("Any", request),
            app_state=cast("Any", app_state),
            quality=80,
            fps=0,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].endswith(b"S\r\n")
    assert reader.read_count == 2
    assert reader.detached
    assert stream_route._stream_stats["total_frames_served"] == 1
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_shared_memory_stream_disconnects_after_configured_send_backpressure(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    reader = _SharedMemoryReader(
        [(frame, SimpleNamespace(frame_number=9, writer_pid=4))]
    )
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", lambda *, name: reader)
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([83], dtype=np.uint8))),
    )
    send_times = iter([100, 200_000_100])
    monkeypatch.setattr(stream_route.time, "perf_counter_ns", lambda: next(send_times))
    app_state = SimpleNamespace(
        config_module=_Config(
            {
                "video.shared_memory_name": "fresh-frames",
                "video.shared_memory_attach_timeout_sec": 0.2,
                "api.stream": {
                    "performance": {
                        "sleep_interval_ms": 0,
                        "maximum_send_backpressure_seconds": 0.05,
                    }
                },
            }
        ),
        vision_module=None,
    )
    generator = stream_route.generate_mjpeg_stream_from_shm(
        request=cast("Any", _FrameRequest(disconnect_after=100)),
        app_state=cast("Any", app_state),
        quality=80,
        fps=0,
    )

    chunk = await generator.__anext__()
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert chunk.endswith(b"S\r\n")
    assert reader.read_count == 1
    assert reader.detached
    assert not stream_route._streaming_clients
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_stop_endpoint_terminates_shared_memory_stream_and_detaches_reader(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    reader = _SharedMemoryReader(
        [(frame, SimpleNamespace(frame_number=15, writer_pid=8))]
    )
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", lambda *, name: reader)
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([76], dtype=np.uint8))),
    )
    app_state = SimpleNamespace(
        config_module=_Config(
            {
                "video.shared_memory_name": "fresh-frames",
                "video.shared_memory_attach_timeout_sec": 0.2,
                "api.stream": {"performance": {"sleep_interval_ms": 0}},
            }
        ),
        vision_module=None,
    )
    request = _FrameRequest(disconnect_after=100)
    generator = stream_route.generate_mjpeg_stream_from_shm(
        request=cast("Any", request),
        app_state=cast("Any", app_state),
        quality=80,
        fps=0,
    )

    chunk = await generator.__anext__()
    assert stream_route._stream_stats["active_streams"] == 1
    assert len(stream_route._streaming_clients) == 1

    stopped = await stream_route.stop_video_capture(
        cast("Any", SimpleNamespace(stop_capture=Mock()))
    )

    assert stopped["active_streams_terminated"] == 1
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert chunk.endswith(b"L\r\n")
    assert request.checks == 1
    assert reader.read_count == 1
    assert reader.detached
    assert not stream_route._streaming_clients
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_shared_memory_stream_maps_missing_video_process_to_503_and_detaches(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    reader = _SharedMemoryReader([], attach_error=TimeoutError("not started"))
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", lambda *, name: reader)
    app_state = SimpleNamespace(
        config_module=_Config(
            {
                "video.shared_memory_name": "test-frames",
                "video.shared_memory_attach_timeout_sec": 0.25,
            }
        ),
        vision_module=None,
    )

    with pytest.raises(HTTPException) as error:
        generator = stream_route.generate_mjpeg_stream_from_shm(
            request=cast("Any", _FrameRequest(disconnect_after=1)),
            app_state=cast("Any", app_state),
            quality=80,
            fps=30,
        )
        await generator.__anext__()

    assert error.value.status_code == 503
    assert "Video Module is not running" in error.value.detail
    assert reader.detached
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_shared_memory_stream_reports_missing_configuration_before_reader_creation(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    reader_factory = Mock(side_effect=AssertionError("reader must not be created"))
    monkeypatch.setattr(stream_route, "SharedMemoryFrameReader", reader_factory)
    app_state = SimpleNamespace(config_module=None, vision_module=None)
    generator = stream_route.generate_mjpeg_stream_from_shm(
        request=cast("Any", _FrameRequest(disconnect_after=1)),
        app_state=cast("Any", app_state),
        quality=80,
        fps=30,
    )

    with pytest.raises(HTTPException) as error:
        await generator.__anext__()

    assert error.value.status_code == 503
    assert error.value.detail == "Configuration module is not available"
    reader_factory.assert_not_called()
    assert stream_route._stream_stats["active_streams"] == 0


@pytest.mark.asyncio
async def test_stream_status_is_read_only_when_camera_has_not_been_initialized(
    clean_stream_globals: None,
) -> None:
    app_state = SimpleNamespace()

    result = await stream_route.stream_status(cast("Any", app_state))

    assert result["camera"]["status"] == "not_initialized"
    assert result["camera"]["connected"] is False
    assert result["streaming"]["active_streams"] == 0
    assert result["streaming"]["total_frames_served"] == 0
    assert not hasattr(app_state, "streaming_camera_module")


@pytest.mark.asyncio
async def test_stream_status_reports_initialized_camera_and_aggregate_delivery_stats(
    monkeypatch: pytest.MonkeyPatch,
    clean_stream_globals: None,
) -> None:
    camera = _EnhancedCamera(
        SimpleNamespace(device_id=2, resolution=(640, 480)),
        event_loop=None,
        frame_callback=None,
    )
    camera.running = True
    app_state = SimpleNamespace(
        streaming_camera_module=stream_route.CameraModuleAdapter(cast("Any", camera))
    )
    stream_route._stream_stats.update(
        active_streams=2,
        total_frames_served=30,
        start_time=1.0,
        last_frame_time=10.5,
    )
    monkeypatch.setattr(stream_route.time, "time", lambda: 11.0)

    result = await stream_route.stream_status(cast("Any", app_state))

    assert result["camera"]["status"] == "connected"
    assert result["camera"]["connected"] is True
    assert result["camera"]["info"]["device_id"] == 2
    assert result["camera"]["health"]["fps"] == 23.5
    assert result["streaming"] == {
        "active_streams": 2,
        "total_frames_served": 30,
        "avg_fps": 3.0,
        "uptime": 10.0,
        "last_frame_time": 10.5,
    }
    assert result["vision"]["is_running"] is True
    assert result["timestamp"] == 11.0


@pytest.mark.asyncio
async def test_start_capture_reports_camera_state_and_stop_clears_active_clients(
    clean_stream_globals: None,
) -> None:
    camera = _EnhancedCamera(
        SimpleNamespace(device_id=5, resolution=(640, 480)),
        event_loop=None,
        frame_callback=None,
    )
    camera.running = True
    adapter = stream_route.CameraModuleAdapter(cast("Any", camera))

    running = await stream_route.start_video_capture(cast("Any", adapter))

    assert running["status"] == "running"
    assert running["camera_info"]["device_id"] == 5
    assert running["camera_health"]["status"] == CameraStatus.CONNECTED

    stream_route._streaming_clients.update({101, 202})
    stream_route._stream_stats["active_streams"] = 2
    stopped = await stream_route.stop_video_capture(cast("Any", adapter))

    assert stopped["status"] == "stopped"
    assert stopped["active_streams_terminated"] == 2
    assert camera.stopped
    assert not stream_route._streaming_clients
    assert stream_route._stream_stats["active_streams"] == 0
