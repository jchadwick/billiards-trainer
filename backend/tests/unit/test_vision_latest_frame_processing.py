"""The live vision worker must discard queued history before inference."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

from backend.vision import VisionModule, VisionStatistics


def _frame_record(frame_number: int) -> dict[str, Any]:
    return {
        "frame": np.full((2, 2, 3), frame_number, dtype=np.uint8),
        "timestamp": float(frame_number),
        "frame_number": frame_number,
        "source_frame_number": frame_number,
        "source_timestamp": None,
        "source_timestamp_kind": "vision_capture_fallback",
        "source_process_id": None,
        "source_publication_monotonic_ns": None,
        "shared_memory_read_copy_seconds": None,
        "enqueued_monotonic": time.monotonic(),
    }


def _bare_module(frame_records: list[dict[str, Any]]) -> VisionModule:
    module = VisionModule.__new__(VisionModule)
    state = cast("Any", module)
    state._is_running = True
    state._frame_queue = queue.Queue(maxsize=max(1, len(frame_records)))
    for frame_record in frame_records:
        state._frame_queue.put_nowait(frame_record)
    state.config = SimpleNamespace()
    state.stats = VisionStatistics()
    state._lock = threading.Lock()
    state._current_result = None
    state._callbacks = {
        "frame_processed": [],
        "detection_complete": [],
        "error_occurred": [],
    }
    state.audit_domains = {"state": False}
    state.audit_recorder = None

    def no_op_stop_capture() -> None:
        return None

    state.stop_capture = no_op_stop_capture
    return module


@pytest.mark.unit
def test_processing_worker_infers_newest_frame_and_counts_skipped_history() -> None:
    module = _bare_module([_frame_record(10), _frame_record(11), _frame_record(12)])
    result = {"balls": ["current"]}

    def process_frame(*args: Any, **kwargs: Any) -> dict[str, Any]:
        module._is_running = False
        return result

    process_mock = Mock(side_effect=process_frame)
    cast("Any", module)._process_single_frame = process_mock
    frame_events: list[dict[str, Any]] = []
    detection_events: list[dict[str, Any]] = []
    module.subscribe_to_events("frame_processed", frame_events.append)
    module.subscribe_to_events("detection_complete", detection_events.append)

    module._processing_loop()

    args, kwargs = process_mock.call_args
    assert args[0][0, 0, 0] == 12
    assert args[1] == 12
    assert kwargs["source_frame_number"] == 12
    assert module.stats.frames_dropped == 2
    assert module.stats.frames_processed == 1
    assert cast("Any", module)._current_result is result
    assert frame_events[0]["source_frame_number"] == 12
    assert detection_events[0]["source_frame_number"] == 12


@pytest.mark.unit
def test_processing_worker_reports_inference_failure_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _bare_module([_frame_record(20)])
    cast("Any", module)._process_single_frame = Mock(
        side_effect=RuntimeError("inference failed")
    )
    errors: list[dict[str, Any]] = []
    module.subscribe_to_events("error_occurred", errors.append)

    def stop_after_retry_delay(_seconds: float) -> None:
        module._is_running = False

    monkeypatch.setattr("backend.vision.time.sleep", stop_after_retry_delay)

    module._processing_loop()

    assert module.stats.last_error == "inference failed"
    assert module.stats.frames_processed == 0
    assert errors == [{"error": "inference failed"}]


@pytest.mark.unit
def test_processing_event_callback_failure_does_not_kill_vision_worker() -> None:
    module = _bare_module([])
    seen: list[dict[str, Any]] = []

    def fail_callback(_event: dict[str, Any]) -> None:
        raise RuntimeError("client failed")

    module.subscribe_to_events("frame_processed", fail_callback)
    module.subscribe_to_events("frame_processed", seen.append)

    module._emit_event("frame_processed", {"frame_number": 5})

    assert seen == [{"frame_number": 5}]
    assert not module.subscribe_to_events("unsupported", seen.append)
