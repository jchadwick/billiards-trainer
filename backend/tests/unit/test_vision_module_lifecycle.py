"""Hardware-free tests for VisionModule startup and frame capture contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import backend.vision as vision


def _vision_config(**overrides: Any) -> dict[str, Any]:
    """Supply every required setting so tests do not depend on local config."""
    values: dict[str, Any] = {
        "camera_device_id": 0,
        "camera_backend": "auto",
        "camera_resolution": (5, 5),
        "camera_fps": 30,
        "camera_buffer_size": 1,
        "target_fps": 30,
        "enable_threading": False,
        "max_frame_queue_size": 3,
        "enable_ball_detection": False,
        "enable_tracking": False,
        "frame_skip": 0,
        "roi_enabled": False,
        "background_image_path": None,
        "use_background_subtraction": False,
        "background_threshold": 30,
        "debug_mode": False,
        "save_debug_images": False,
        "debug_output_path": "/tmp/vision-test",
    }
    values.update(overrides)
    return values


class _FakeVideoConsumer:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.running = False
        self.frame_result: tuple[Any, Any] = (None, None)
        self.read_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.start_calls += 1
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def get_frame_with_metadata(self) -> tuple[Any, Any]:
        if self.read_error is not None:
            raise self.read_error
        self.running = False
        return self.frame_result


def _construct_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    consumer: _FakeVideoConsumer | None = None,
    **config_overrides: Any,
) -> tuple[vision.VisionModule, _FakeVideoConsumer]:
    fake_consumer = consumer or _FakeVideoConsumer()
    monkeypatch.setattr(vision, "VideoConsumer", lambda: fake_consumer)
    monkeypatch.setattr(vision, "_get_config_value", lambda _key, default: default)
    module = vision.VisionModule(_vision_config(**config_overrides))
    return module, fake_consumer


@pytest.mark.unit
def test_normal_initialization_leaves_detection_components_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _consumer = _construct_module(monkeypatch)

    assert module.detector is None
    assert module.tracker is None


@pytest.mark.unit
def test_capture_start_is_idempotent_resets_run_state_and_stop_drains_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, consumer = _construct_module(monkeypatch)
    module.stats.frames_processed = 12
    module.stats.frames_dropped = 4
    module._frame_number = 91
    module._phase_last_motion_timestamp = 123.0
    module._phase_calibration_revision = "previous-calibration"
    module._phase_last_source_frame_number = 90
    module._phase_last_source_process_id = 4567
    module._phase_last_source_timestamp = 122.5
    module._frame_queue.put_nowait({"source_frame_number": 90})
    module._result_queue.put_nowait({"frame_number": 90})

    assert module.start_capture() is True
    assert module.start_capture() is True
    assert consumer.start_calls == 1
    assert module.stats.frames_processed == 0
    assert module.stats.frames_dropped == 0
    assert module._frame_number == 0
    assert module._phase_last_motion_timestamp != 123.0
    state = cast("Any", module)
    assert state._phase_calibration_revision is None
    assert state._phase_last_source_frame_number is None
    assert state._phase_last_source_process_id is None
    assert state._phase_last_source_timestamp is None

    module.stop_capture()

    assert consumer.stop_calls == 1
    assert module._is_running is False
    assert module._frame_queue.empty()
    assert module._result_queue.empty()


@pytest.mark.unit
def test_capture_loop_preserves_source_identity_and_publication_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, consumer = _construct_module(monkeypatch)
    frame = np.full((5, 5, 3), 40, dtype=np.uint8)
    metadata = type(
        "SourceMetadata",
        (),
        {
            "timestamp": 123.25,
            "frame_number": 482,
            "writer_pid": 2764,
            "publication_monotonic_ns": 789_000_000,
        },
    )()
    consumer.frame_result = (frame, metadata)
    module.set_roi([(1, 1), (3, 1), (3, 3), (1, 3)])
    monkeypatch.setattr(vision.time, "monotonic", lambda: 1.0)

    def return_frame_once() -> tuple[Any, Any]:
        module._is_running = False
        return frame, metadata

    monkeypatch.setattr(consumer, "get_frame_with_metadata", return_frame_once)

    module._is_running = True
    module._capture_loop()

    captured = module._frame_queue.get_nowait()
    assert captured["source_frame_number"] == 482
    assert captured["frame_number"] == 482
    assert captured["timestamp"] == 123.25
    assert captured["source_timestamp"] == 123.25
    assert captured["source_timestamp_kind"] == "shared_memory_publication"
    assert captured["source_process_id"] == 2764
    assert captured["source_publication_monotonic_ns"] == 789_000_000
    assert captured["frame"][2, 2].tolist() == [40, 40, 40]
    assert captured["frame"][0, 0].tolist() == [0, 0, 0]
    current_frame = module.get_current_frame()
    assert current_frame is not None
    assert current_frame is not frame
    assert np.array_equal(current_frame, captured["frame"])


@pytest.mark.unit
def test_invalid_roi_retains_previous_region_and_reports_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _consumer = _construct_module(monkeypatch)
    valid_roi = [(0, 0), (4, 0), (4, 4), (0, 4)]
    module.set_roi(valid_roi)

    module.set_roi([(1, 1), (2, 2)])

    assert module._roi_corners == valid_roi
    assert module.config.roi_enabled is True
    assert module.stats.last_error == "ROI must have exactly 4 corners"


@pytest.mark.unit
def test_capture_read_failure_emits_error_and_retries_without_losing_running_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _FakeVideoConsumer()
    consumer.read_error = RuntimeError("shared-memory read failed")
    module, _consumer = _construct_module(monkeypatch, consumer=consumer)
    errors: list[dict[str, Any]] = []
    module.subscribe_to_events("error_occurred", errors.append)

    def stop_after_retry(_seconds: float) -> None:
        module._is_running = False

    monkeypatch.setattr(vision.time, "sleep", stop_after_retry)
    module._is_running = True

    module._capture_loop()

    assert module.stats.last_error == "shared-memory read failed"
    assert errors == [{"error": "shared-memory read failed"}]
    assert module._frame_queue.empty()


@pytest.mark.unit
def test_startup_attachment_failure_is_reported_without_marking_capture_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingVideoConsumer(_FakeVideoConsumer):
        def start(self) -> None:
            raise vision.VideoModuleNotAvailableError("producer unavailable")

    module, consumer = _construct_module(monkeypatch, consumer=MissingVideoConsumer())
    errors: list[dict[str, Any]] = []
    module.subscribe_to_events("error_occurred", errors.append)

    assert module.start_capture() is False

    assert consumer.start_calls == 0
    assert module._is_running is False
    assert "producer unavailable" in (module.stats.last_error or "")
    assert errors
    assert "producer unavailable" in errors[0]["error"]
