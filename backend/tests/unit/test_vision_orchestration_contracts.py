"""Deterministic orchestration tests for fresh live perception results."""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import backend.vision as vision
from backend.common.models import Ball
from backend.tests.unit.test_vision_phase_integration import (
    _make_module,
    _patch_cue_conversion,
    _settings,
    _StatusAwareFakeDetector,
)
from backend.vision import VisionModule, VisionStatistics
from backend.vision.detection.yolo_detector import (
    BallDetectionOutcome,
    BallDetectionStatus,
    DetectionFailureStage,
)
from backend.vision.models import BallType


def _frame_record(frame_number: int) -> dict[str, Any]:
    return {
        "frame": np.full((2, 2, 3), frame_number, dtype=np.uint8),
        "timestamp": float(frame_number),
        "frame_number": frame_number,
        "source_frame_number": frame_number,
        "source_timestamp": float(frame_number),
        "source_timestamp_kind": "shared_memory_publication",
        "source_process_id": 17,
        "source_publication_monotonic_ns": None,
        "enqueued_monotonic": 0.0,
    }


def _ball(x: float) -> Ball:
    return Ball(
        position=(x, 25.0),
        radius=5.0,
        ball_type=BallType.UNKNOWN,
        confidence=0.9,
        coordinate_space="pixel",
        source_resolution=(100, 50),
    )


@pytest.mark.unit
def test_processing_worker_recovers_after_transient_inference_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = VisionModule.__new__(VisionModule)
    module._is_running = True
    module._frame_queue = queue.Queue(maxsize=2)
    module._frame_queue.put_nowait(_frame_record(1))
    module.config = SimpleNamespace()
    module.stats = VisionStatistics()
    module._lock = threading.Lock()
    previous_result = object()
    module._current_result = previous_result
    module._callbacks = {
        "frame_processed": [],
        "detection_complete": [],
        "error_occurred": [],
    }
    module.stop_capture = lambda: None

    recovered_result = {"source_frame_number": 2}
    attempts: list[int] = []

    def process_frame(_frame: Any, frame_number: int, *_args: Any, **_kwargs: Any):
        attempts.append(frame_number)
        if len(attempts) == 1:
            # Capture continues while inference is busy. The next source frame
            # must remain available so the worker can recover on fresh input.
            module._frame_queue.put_nowait(_frame_record(2))
            raise RuntimeError("temporary detector failure")
        module._is_running = False
        return recovered_result

    cast("Any", module)._process_single_frame = process_frame
    monkeypatch.setattr(vision, "_get_config_value", lambda _key, default: default)
    monkeypatch.setattr(vision.time, "sleep", lambda _seconds: None)
    frame_events: list[dict[str, Any]] = []
    detection_events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    module.subscribe_to_events("frame_processed", frame_events.append)
    module.subscribe_to_events("detection_complete", detection_events.append)
    module.subscribe_to_events("error_occurred", errors.append)

    module._processing_loop()

    assert attempts == [1, 2]
    assert module._current_result is recovered_result
    assert module.process_frame() is recovered_result
    assert module.stats.frames_processed == 1
    assert module.stats.last_error == "temporary detector failure"
    assert errors == [{"error": "temporary detector failure"}]
    assert [event["source_frame_number"] for event in frame_events] == [2]
    assert [event["source_frame_number"] for event in detection_events] == [2]
    assert previous_result is not module._current_result


@pytest.mark.unit
def test_successful_ball_observation_replaces_cached_map_after_inference_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=100.0, cue_implies_aiming=False))
    module.detector = _StatusAwareFakeDetector(
        [
            BallDetectionOutcome(BallDetectionStatus.DETECTIONS, (_ball(20.0),)),
            BallDetectionOutcome(
                BallDetectionStatus.FAILED,
                failure_stage=DetectionFailureStage.INFERENCE,
            ),
            BallDetectionOutcome(BallDetectionStatus.DETECTIONS, (_ball(35.0),)),
        ]
    )
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    observed = module._process_single_frame(
        frame, 1, 100.0, source_frame_number=201, source_timestamp=10.0
    )
    module.phase_scheduler._next_due = dict.fromkeys(module.phase_scheduler._next_due)
    failed = module._process_single_frame(
        frame, 2, 100.1, source_frame_number=202, source_timestamp=10.1
    )
    module.phase_scheduler._next_due = dict.fromkeys(module.phase_scheduler._next_due)
    recovered = module._process_single_frame(
        frame, 3, 100.2, source_frame_number=203, source_timestamp=10.2
    )

    assert observed.ball_detection_status == "observed"
    assert observed.ball_observation_source_frame_number == 201
    assert failed.ball_detection_status == "failed"
    assert failed.ball_positions_status == "cached"
    assert failed.ball_observation_source_frame_number == 201
    assert failed.balls[0].position == observed.balls[0].position
    assert recovered.ball_detection_status == "observed"
    assert recovered.ball_positions_status == "observed"
    assert recovered.ball_observation_source_frame_number == 203
    assert recovered.ball_observation_source_timestamp == 10.2
    assert recovered.balls[0].position == pytest.approx((35.0, 25.0))
    assert len(module.tracker.calls) == 2


@pytest.mark.unit
def test_expired_cached_cue_is_cleared_while_ball_observation_remains_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cue_conversion(monkeypatch)
    module = _make_module(
        settings=_settings(
            interval=100.0,
            cue_implies_aiming=False,
            maximum_cue_age_seconds=0.25,
            maximum_motion_observation_gap_seconds=10.0,
            maximum_observation_age_seconds=10.0,
            quiet_dwell_seconds=10.0,
            consistent_full_observations_required=10,
        )
    )
    current_time = [1.0]
    module._clock = lambda: current_time[0]
    module.detector.ball_factory = lambda: [_ball(25.0)]
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    fresh = module._process_single_frame(
        frame, 1, 100.0, source_frame_number=301, source_timestamp=20.0
    )
    current_time[0] = 2.0
    aged = module._process_single_frame(
        frame, 2, 100.1, source_frame_number=302, source_timestamp=20.1
    )

    assert fresh.cue_detection_status == "observed"
    assert fresh.cue is not None
    assert aged.cue_detection_status == "stale"
    assert aged.cue is None
    assert aged.cue_observation_timestamp == fresh.cue_observation_timestamp
    assert aged.ball_detection_status == "skipped"
    assert aged.ball_positions_status == "cached"
    assert aged.ball_observation_source_frame_number == 301
    assert aged.component_observation_status["cue"] == "stale"
    assert aged.component_observation_status["balls"] == "skipped"
    assert aged.is_complete is True
