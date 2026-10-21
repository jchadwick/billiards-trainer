"""Integration checks for the opt-in phase-aware VisionModule path."""

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backend.common.models import Ball
from backend.vision import VisionModule, VisionStatistics
from backend.vision.detection.yolo_detector import (
    BallDetectionOutcome,
    BallDetectionStatus,
    DetectionFailureStage,
)
from backend.vision.models import BallType, CueStick
from backend.vision.motion_monitor import MotionAssessment
from backend.vision.phase_scheduler import VisionPhase, VisionPhaseScheduler


class _FakeMotionMonitor:
    def __init__(self):
        self.assessments = []
        self.commits = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def assess(self, frame, polygon, *, candidate_balls, timestamp):
        self.assessments.append(
            {
                "shape": frame.shape,
                "polygon": polygon,
                "candidate_balls": tuple(candidate_balls),
                "timestamp": timestamp,
            }
        )
        assessment = _quiet_assessment(timestamp, frame.shape[1], frame.shape[0])
        return assessment

    def commit_verified_reference(
        self, frame, polygon, *, settled, map_reconciled, timestamp
    ):
        self.commits.append((settled, map_reconciled, timestamp))
        return settled and map_reconciled


class _FakeTracker:
    def __init__(self):
        self.calls = []
        self.reset_calls = 0

    def update_tracking(self, balls, frame_number, timestamp):
        self.calls.append((list(balls), frame_number, timestamp))
        return balls

    def reset(self):
        self.reset_calls += 1


class _FakeDetector:
    def __init__(self, ball_factory=None):
        self.ball_factory = ball_factory or (lambda: [])
        self.ball_calls = []
        self.cue_calls = []
        self.fail_ball_calls = set()
        self.fail_cue_calls = set()

    def detect_balls_with_classification(
        self, frame, *, frame_key, classification_frame
    ):
        self.ball_calls.append((frame.shape, classification_frame.shape, frame_key))
        if len(self.ball_calls) in self.fail_ball_calls:
            raise RuntimeError("ball inference failed")
        return self.ball_factory()

    def detect_cue(self, frame, *, frame_key):
        self.cue_calls.append((frame.shape, frame_key))
        if len(self.cue_calls) in self.fail_cue_calls:
            raise RuntimeError("cue inference failed")
        return SimpleNamespace(confidence=0.95)


class _StatusAwareFakeDetector(_FakeDetector):
    """Test detector that exposes the richer, optional observation API."""

    def __init__(self, outcomes: list[BallDetectionOutcome[Any]]) -> None:
        super().__init__()
        self.outcomes = list(outcomes)

    def detect_balls_with_classification_outcome(
        self,
        frame: Any,
        *,
        frame_key: Any,
        classification_frame: Any,
    ) -> BallDetectionOutcome[Any]:
        self.ball_calls.append((frame.shape, classification_frame.shape, frame_key))
        return self.outcomes.pop(0)


def _quiet_assessment(timestamp, width=100, height=50):
    return MotionAssessment(
        valid=True,
        uncertain=False,
        uncertainty_reasons=(),
        motion_detected=False,
        motion_score=0.0,
        active_motion_fraction=0.0,
        persistent_change_detected=False,
        stable_scene_change_fraction=0.0,
        changed_regions=(),
        active_changed_regions=(),
        likely_occlusion=False,
        occlusion_score=0.0,
        illumination_change=False,
        illumination_change_score=0.0,
        illumination_median_delta=0.0,
        illumination_contrast_change=0.0,
        candidate_ball_changes=(),
        table_mask_pixels=100,
        frame_resolution=(width, height),
        timestamp=timestamp,
    )


def _settings(*, enabled=True, interval=0.00001, max_dimension=None, **extra):
    return {
        "enabled": enabled,
        "quiet_dwell_seconds": 0.0,
        "limited_motion_quiet_dwell_seconds": 0.0,
        "many_moving_exit_quiet_dwell_seconds": 0.0,
        "many_moving_enter_count": 3,
        "many_moving_exit_count": 1,
        "many_moving_enter_dwell_seconds": 0.0,
        "consistent_full_observations_required": 1,
        "minimum_detected_balls": 1,
        "minimum_ball_confidence": 0.5,
        "ball_position_tolerance_px": 6.0,
        "maximum_motion_observation_gap_seconds": 1.0,
        "maximum_ball_observation_gap_seconds": 2.0,
        "maximum_observation_age_seconds": 2.0,
        "maximum_cue_age_seconds": 2.0,
        "cue_implies_aiming": True,
        "processing_max_dimension": {
            phase.value: max_dimension for phase in VisionPhase
        },
        "intervals_seconds": {
            phase.value: {
                "motion_monitor": interval,
                "ball_map_verification": interval,
                "cue_detection": interval,
                "detailed_tracking": interval,
            }
            for phase in VisionPhase
        },
        **extra,
    }


def _make_module(*, settings=None, ball_factory=None):
    module = VisionModule.__new__(VisionModule)
    module.phase_scheduler = VisionPhaseScheduler(settings or _settings())
    module.motion_monitor = _FakeMotionMonitor()
    module._aiming_request_override = None
    module._phase_geometry_signature = None
    module._phase_calibration_revision = None
    module._phase_last_motion_assessment = None
    module._phase_last_motion_timestamp = None
    module._phase_last_ball_timestamp = None
    module._phase_last_ball_source_timestamp = None
    module._phase_last_ball_source_frame_number = None
    module._phase_last_ball_source_timestamp_kind = None
    module._phase_last_source_frame_number = None
    module._phase_last_source_timestamp = None
    module._phase_last_cue_timestamp = None
    module._phase_last_cue_detected = False
    module._phase_last_cue_reliable = False
    module._phase_last_cue = None
    module._phase_last_ball_map = []
    module.config = SimpleNamespace(enable_ball_detection=True, enable_tracking=True)
    module.detector = _FakeDetector(ball_factory)
    module.tracker = _FakeTracker()
    module.profiler = None
    module.stats = VisionStatistics()
    module._frame_number = 0
    module._apply_all_masks = lambda frame: frame.copy()
    module._is_ball_in_invalid_region = lambda x, y, width, height: False
    module._phase_table_geometry = lambda shape: (
        [(2.0, 2.0), (98.0, 2.0), (98.0, 48.0), (2.0, 48.0)],
        f"test-calibration:{shape[1]}x{shape[0]}",
    )
    return module


def _patch_cue_conversion(monkeypatch):
    def convert(detection, image_shape, min_confidence):
        height, width = image_shape
        return CueStick(
            tip_position=(1920.0, 1080.0),
            angle=12.0,
            length=80.0,
            confidence=detection.confidence,
            coordinate_space="4k_canonical",
            source_resolution=(width, height),
        )

    monkeypatch.setattr(
        "backend.vision.detection.detector_adapter.yolo_cue_to_cue_stick", convert
    )


@pytest.mark.unit
def test_disabled_scheduler_keeps_full_resolution_ball_and_cue_path(monkeypatch):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(enabled=False))
    module.detector.ball_factory = lambda: [
        Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            coordinate_space="4k_canonical",
            source_resolution=(100, 50),
        )
    ]
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    result = module._process_single_frame(
        frame,
        1,
        100.0,
        source_frame_number=44,
        source_timestamp=99.9,
        source_timestamp_kind="shared_memory_publication",
    )

    assert module.detector.ball_calls[0][0] == frame.shape
    assert module.detector.cue_calls[0][0] == frame.shape
    assert module.tracker.calls
    assert result.phase_aware is False
    assert result.aim_calculation_eligible is None
    assert result.source_frame_number == 44
    assert result.source_timestamp == 99.9
    assert result.ball_detection_status == "observed"
    assert result.ball_positions_status == "observed"
    assert result.ball_observation_source_frame_number == 44
    assert result.ball_observation_source_timestamp == 99.9
    assert result.ball_observation_source_timestamp_kind == "shared_memory_publication"
    assert result.cue_detection_status == "observed"
    assert result.balls[0].source_resolution == (100, 50)


@pytest.mark.unit
def test_tracker_receives_monotonic_time_not_wall_publication_time(monkeypatch):
    module = _make_module(settings=_settings(enabled=False))
    module._clock = lambda: 123.0

    module._process_single_frame(
        np.zeros((50, 100, 3), dtype=np.uint8), 7, 1_800_000_000.0
    )

    assert module.tracker.calls[0][1:] == (7, 123.0)


@pytest.mark.unit
def test_frame_publication_age_is_distinct_from_monotonic_observation_time(
    monkeypatch,
):
    module = _make_module(settings=_settings(enabled=False))
    module._clock = lambda: 123.0
    monkeypatch.setattr("backend.vision.time.time", lambda: 1000.0)

    result = module._process_single_frame(
        np.zeros((50, 100, 3), dtype=np.uint8),
        8,
        999.5,
        source_frame_number=8,
        source_timestamp=999.5,
        source_timestamp_kind="shared_memory_publication",
    )

    assert result.frame_publication_age_seconds == pytest.approx(0.5)
    assert result.timestamp == 999.5
    assert module.tracker.calls[0][2] == 123.0
    assert result.ball_observation_timestamp is None


@pytest.mark.unit
def test_source_sequence_restart_resets_temporal_state_and_rebuilds_map(monkeypatch):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings())
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    module._process_single_frame(
        frame,
        100,
        1000.0,
        source_frame_number=100,
        source_timestamp=1000.0,
    )
    restarted = module._process_single_frame(
        frame,
        1,
        1001.0,
        source_frame_number=1,
        source_timestamp=1001.0,
    )

    assert restarted is not None
    assert module.tracker.reset_calls == 1
    assert module.motion_monitor.reset_calls >= 2
    assert restarted.phase == "recovering"
    assert restarted.map_trusted is False


@pytest.mark.unit
def test_calibrated_ball_crop_restores_resize_offset_and_keeps_cue_full_frame(
    monkeypatch,
):
    _patch_cue_conversion(monkeypatch)
    crop_by_phase = {phase.value: True for phase in VisionPhase}
    module = _make_module(
        settings=_settings(
            max_dimension=50,
            ball_detection_crop=crop_by_phase,
            ball_crop_margin_fraction=0.0,
        )
    )
    module.detector.ball_factory = lambda: [
        Ball(
            position=(24.0, 11.5),
            radius=2.0,
            ball_type=BallType.CUE,
            confidence=0.95,
            coordinate_space="pixel",
            source_resolution=(48, 23),
        )
    ]

    result = module._process_single_frame(
        np.zeros((50, 100, 3), dtype=np.uint8),
        1,
        100.0,
        source_frame_number=1,
        source_timestamp=99.0,
    )

    assert module.detector.ball_calls[0][0] == (23, 48, 3)
    assert module.detector.ball_calls[0][1] == module.detector.ball_calls[0][0]
    assert module.detector.cue_calls[0][0] == (25, 50, 3)
    assert module.detector.ball_calls[0][2] != module.detector.cue_calls[0][1]
    assert result.ball_detection_view == "calibrated_table_crop"
    assert result.balls[0].position == pytest.approx((50.0, 25.0))
    assert result.balls[0].source_resolution == (100, 50)


@pytest.mark.unit
def test_enabled_scheduler_reports_observed_then_skipped_and_preserves_map(
    monkeypatch,
):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=100.0, cue_implies_aiming=False))
    module.detector.ball_factory = lambda: [
        Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            coordinate_space="4k_canonical",
            source_resolution=(100, 50),
        )
    ]
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    observed = module._process_single_frame(
        frame,
        1,
        90.0,
        source_frame_number=10,
        source_timestamp=90.0,
        source_timestamp_kind="shared_memory_publication",
    )
    skipped = module._process_single_frame(
        frame,
        2,
        90.1,
        source_frame_number=11,
        source_timestamp=90.1,
        source_timestamp_kind="shared_memory_publication",
    )

    assert observed.phase_aware
    assert observed.ball_detection_status == "observed"
    assert observed.cue_detection_status == "observed"
    assert observed.motion_monitor_status == "observed"
    assert observed.ball_positions_status == "observed"
    assert observed.ball_observation_source_frame_number == 10
    assert observed.ball_observation_source_timestamp == 90.0
    assert observed.balls[0].last_seen == 90.0
    assert skipped.ball_detection_status == "skipped"
    assert skipped.ball_positions_status == "cached"
    assert skipped.ball_observation_source_frame_number == 10
    assert skipped.ball_observation_source_timestamp == 90.0
    assert skipped.source_frame_number == 11
    assert skipped.source_timestamp == 90.1
    assert skipped.balls[0].last_seen == 90.0
    assert skipped.cue_detection_status == "skipped"
    assert skipped.motion_monitor_status == "skipped"
    assert len(skipped.balls) == 1
    assert len(module.detector.ball_calls) == 1
    assert len(module.detector.cue_calls) == 1
    assert len(module.tracker.calls) == 1


@pytest.mark.unit
def test_failed_observation_is_distinct_from_skip_and_does_not_erase_map(
    monkeypatch,
):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=100.0, cue_implies_aiming=False))
    module.detector.ball_factory = lambda: [
        Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            coordinate_space="4k_canonical",
            source_resolution=(100, 50),
        )
    ]
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    first = module._process_single_frame(
        frame, 1, 100.0, source_frame_number=20, source_timestamp=80.0
    )
    skipped = module._process_single_frame(
        frame, 2, 100.1, source_frame_number=21, source_timestamp=80.1
    )

    # Force the independent jobs due for one frame, then simulate model errors.
    module.phase_scheduler._next_due = dict.fromkeys(module.phase_scheduler._next_due)
    module.detector.fail_ball_calls.add(2)
    module.detector.fail_cue_calls.add(2)
    failed = module._process_single_frame(
        frame, 3, 100.2, source_frame_number=22, source_timestamp=80.2
    )

    assert len(first.balls) == len(skipped.balls) == len(failed.balls) == 1
    assert skipped.ball_detection_status == "skipped"
    assert failed.ball_detection_status == "failed"
    assert failed.ball_positions_status == "cached"
    assert failed.ball_observation_source_frame_number == 20
    assert failed.ball_observation_source_timestamp == 80.0
    assert failed.cue_detection_status == "failed"
    assert len(module.tracker.calls) == 1


@pytest.mark.unit
def test_status_aware_empty_inference_is_observed_and_can_clear_ball_positions() -> (
    None
):
    module = _make_module(settings=_settings(interval=0.00001))
    ball = Ball(
        position=(25.0, 25.0),
        radius=5.0,
        ball_type=BallType.UNKNOWN,
        confidence=0.9,
        coordinate_space="pixel",
        source_resolution=(100, 50),
    )
    module.detector = _StatusAwareFakeDetector(
        [
            BallDetectionOutcome(BallDetectionStatus.DETECTIONS, (ball,)),
            BallDetectionOutcome(BallDetectionStatus.NO_DETECTIONS),
        ]
    )
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    first = module._process_single_frame(frame, 1, 100.0, source_frame_number=1)
    result = module._process_single_frame(frame, 2, 100.1, source_frame_number=2)

    assert len(first.balls) == 1
    assert result.ball_detection_status == "observed"
    assert result.ball_positions_status == "observed"
    assert result.balls == []
    assert module._phase_last_ball_map == []
    assert result.ball_observation_source_frame_number == 2
    assert len(module.tracker.calls) == 2
    assert module.tracker.calls[1][0] == []
    assert module.tracker.calls[1][1] == 2
    assert result.is_complete is True
    assert result.has_errors is False
    assert result.error_messages == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_stage", "expected_error"),
    [
        (
            BallDetectionOutcome(BallDetectionStatus.MODEL_UNAVAILABLE),
            "unavailable",
            None,
            False,
        ),
        (
            BallDetectionOutcome(
                BallDetectionStatus.FAILED,
                failure_stage=DetectionFailureStage.CLASSIFICATION,
            ),
            "failed",
            "classification",
            True,
        ),
    ],
)
def test_status_aware_incomplete_outcomes_preserve_cached_map_without_tracking(
    monkeypatch: pytest.MonkeyPatch,
    outcome: BallDetectionOutcome[Any],
    expected_status: str,
    expected_stage: str | None,
    expected_error: bool,
) -> None:
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=100.0, cue_implies_aiming=False))
    ball = Ball(
        position=(1920.0, 1080.0),
        radius=36.0,
        ball_type=BallType.CUE,
        confidence=0.9,
        coordinate_space="4k_canonical",
        source_resolution=(100, 50),
    )
    module.detector = _StatusAwareFakeDetector(
        [BallDetectionOutcome(BallDetectionStatus.DETECTIONS, (ball,)), outcome]
    )
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    observed = module._process_single_frame(
        frame, 1, 100.0, source_frame_number=40, source_timestamp=10.0
    )

    module.phase_scheduler._next_due = dict.fromkeys(module.phase_scheduler._next_due)
    monkeypatch.setattr(
        module,
        "_normalise_balls_to_source",
        lambda *_args: pytest.fail("incomplete outcome must not be normalized"),
    )
    result = module._process_single_frame(
        frame, 2, 100.1, source_frame_number=41, source_timestamp=10.1
    )

    assert observed.ball_detection_status == "observed"
    assert result.ball_detection_status == expected_status
    assert result.component_observation_status["balls"] == expected_status
    assert result.ball_detection_failure_stage == expected_stage
    assert result.ball_positions_status == "cached"
    assert result.ball_observation_source_frame_number == 40
    assert result.balls == observed.balls
    assert len(module.tracker.calls) == 1
    assert result.is_complete is False
    assert result.has_errors is expected_error
    assert result.map_trusted is False
    if expected_error:
        assert result.error_messages == ["balls:failed:classification"]
    else:
        assert result.error_messages == []
    assert all("backend" not in message for message in result.error_messages)


@pytest.mark.unit
def test_trusted_map_and_fresh_cue_make_aim_eligible(monkeypatch):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=0.00001))
    module.detector.ball_factory = lambda: [
        Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            coordinate_space="4k_canonical",
            source_resolution=(100, 50),
        )
    ]
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    first = module._process_single_frame(frame, 1, 100.0)
    aiming = module._process_single_frame(frame, 2, 100.1)

    assert first.map_trusted
    assert first.vision_phase in {"idle", "aiming"}
    assert aiming.vision_phase == "aiming"
    assert aiming.aim_calculation_eligible is True
    assert aiming.map_trusted is True
    assert module.motion_monitor.commits
    assert module.motion_monitor.commits[0][:2] == (True, True)


@pytest.mark.unit
def test_many_moving_uses_fresh_quiet_monitor_as_zero_moving_estimate():
    module = _make_module(settings=_settings(interval=0.00001))
    module.phase_scheduler.phase = VisionPhase.MANY_MOVING
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    result = module._process_single_frame(frame, 1, 100.0)

    assert result.ball_detection_status == "skipped"
    assert result.cue_detection_status == "skipped"
    assert result.moving_ball_count == 0
    assert result.vision_phase == "recovering"
    assert not module.detector.ball_calls
    assert not module.detector.cue_calls


@pytest.mark.unit
def test_metadata_fields_are_declared_and_downscaled_coords_return_to_native(
    monkeypatch,
):
    _patch_cue_conversion(monkeypatch)
    module = _make_module(settings=_settings(interval=0.00001, max_dimension=10))

    def ball_factory():
        return [
            Ball(
                position=(1920.0, 1080.0),
                radius=18.0,
                ball_type=BallType.CUE,
                confidence=0.95,
                coordinate_space="4k_canonical",
                source_resolution=(10, 5),
            )
        ]

    module.detector.ball_factory = ball_factory
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    result = module._process_single_frame(frame, 1, 100.0)

    assert module.detector.ball_calls[0][0] == (5, 10, 3)
    assert module.detector.ball_calls[0][1] == (5, 10, 3)
    assert result.balls[0].source_resolution == (100, 50)
    assert result.balls[0].position == pytest.approx((1920.0, 1080.0))
    assert result.calibration_revision == "test-calibration:100x50"
    assert module.phase_scheduler._trusted_snapshot[0].x == pytest.approx(50.0)
    assert module.phase_scheduler._trusted_snapshot[0].y == pytest.approx(25.0)
    assert result.cue.source_resolution == (100, 50)

    model_fields = {field.name for field in fields(result)}
    assert {
        "source_frame_number",
        "source_timestamp",
        "source_timestamp_kind",
        "phase_aware",
        "vision_phase",
        "map_revision",
        "map_trusted",
        "aim_calculation_eligible",
        "ball_detection_status",
        "ball_positions_status",
        "ball_observation_source_timestamp",
        "ball_observation_source_frame_number",
        "ball_observation_source_timestamp_kind",
        "cue_detection_status",
        "motion_monitor_status",
        "ball_observation_timestamp",
        "cue_observation_timestamp",
        "motion_observation_timestamp",
    } <= model_fields
