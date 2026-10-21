"""Deterministic tests for opt-in billiards vision phase scheduling."""

from dataclasses import FrozenInstanceError, replace

import pytest

from backend.vision.motion_monitor import MotionAssessment
from backend.vision.phase_scheduler import (
    BallMapObservation,
    BallPosition,
    CueObservation,
    VisionPhase,
    VisionPhaseScheduler,
)


def _settings(**overrides):
    settings = {
        "enabled": True,
        "quiet_dwell_seconds": 0.2,
        "limited_motion_quiet_dwell_seconds": 0.2,
        "many_moving_exit_quiet_dwell_seconds": 0.2,
        "many_moving_enter_count": 3,
        "many_moving_exit_count": 1,
        "many_moving_enter_dwell_seconds": 0.1,
        "consistent_full_observations_required": 2,
        "ball_position_tolerance_px": 4.0,
        "minimum_ball_confidence": 0.6,
        "minimum_detected_balls": 1,
        "maximum_motion_observation_gap_seconds": 0.5,
        "maximum_ball_observation_gap_seconds": 0.8,
        "maximum_observation_age_seconds": 0.5,
        "maximum_cue_age_seconds": 0.15,
        "intervals_seconds": {
            phase.value: {
                "motion_monitor": 0.1,
                "ball_map_verification": 0.1,
                "cue_detection": 0.1,
                "detailed_tracking": 0.05,
            }
            for phase in VisionPhase
        },
    }
    settings.update(overrides)
    return settings


def _assessment(
    timestamp,
    *,
    moving=False,
    uncertain=False,
    occlusion=False,
    illumination=False,
    persistent=False,
    valid=True,
):
    reasons = []
    if moving:
        reasons.append("active_motion")
    if persistent:
        reasons.append("stable_scene_changed")
    if occlusion:
        reasons.append("likely_occlusion")
    if illumination:
        reasons.append("illumination_changed")
    if uncertain and not reasons:
        reasons.append("ambiguous_scene")
    return MotionAssessment(
        valid=valid,
        uncertain=bool(reasons) or uncertain,
        uncertainty_reasons=tuple(reasons),
        motion_detected=moving,
        motion_score=0.02 if moving else 0.0,
        active_motion_fraction=0.02 if moving else 0.0,
        persistent_change_detected=persistent,
        stable_scene_change_fraction=0.01 if persistent else 0.0,
        changed_regions=(),
        active_changed_regions=(),
        likely_occlusion=occlusion,
        occlusion_score=1.0 if occlusion else 0.0,
        illumination_change=illumination,
        illumination_change_score=1.0 if illumination else 0.0,
        illumination_median_delta=20.0 if illumination else 0.0,
        illumination_contrast_change=0.0,
        candidate_ball_changes=(),
        table_mask_pixels=1000,
        frame_resolution=(640, 360),
        timestamp=timestamp,
    )


def _map(
    timestamp,
    *,
    x=100.0,
    y=80.0,
    confidence=0.9,
    full=True,
    success=True,
    uncertain=False,
):
    return BallMapObservation(
        timestamp=timestamp,
        balls=(BallPosition(x=x, y=y, confidence=confidence),),
        full=full,
        successful=success,
        uncertain=uncertain,
    )


def _settle_to_idle(scheduler, start=0.0, *, calibration_revision="cal-a"):
    scheduler.update(
        now=start,
        motion_assessment=_assessment(start),
        ball_map_observation=_map(start),
        calibration_revision=calibration_revision,
    )
    scheduler.update(
        now=start + 0.1,
        motion_assessment=_assessment(start + 0.1),
        ball_map_observation=_map(start + 0.1, x=101.0),
        calibration_revision=calibration_revision,
    )
    return scheduler.update(
        now=start + 0.2,
        motion_assessment=_assessment(start + 0.2),
        calibration_revision=calibration_revision,
    )


@pytest.mark.unit
def test_default_disabled_is_legacy_passthrough_and_decision_is_immutable():
    scheduler = VisionPhaseScheduler({})

    decision = scheduler.update(
        now=1.0,
        motion_assessment=_assessment(1.0, moving=True),
        moving_ball_count=12,
    )

    assert decision.enabled is False
    assert decision.phase is VisionPhase.RECOVERING
    assert decision.run_motion_monitor
    assert decision.verify_ball_map
    assert decision.detect_cue
    assert decision.track_moving_balls
    assert decision.can_calculate_aim
    assert decision.processing_max_dimension is None
    with pytest.raises(FrozenInstanceError):
        decision.phase = VisionPhase.IDLE


@pytest.mark.unit
def test_requires_quiet_dwell_and_consecutive_consistent_full_observations():
    scheduler = VisionPhaseScheduler(_settings())

    first = scheduler.update(
        now=0.0,
        motion_assessment=_assessment(0.0),
        ball_map_observation=_map(0.0),
    )
    skipped = scheduler.update(
        now=0.1,
        motion_assessment=_assessment(0.1),
        ball_map_observation=None,
    )
    settled = scheduler.update(
        now=0.2,
        motion_assessment=_assessment(0.2),
        ball_map_observation=_map(0.2, x=102.0),
    )

    assert first.phase is VisionPhase.RECOVERING
    assert not first.map_trusted
    assert skipped.phase is VisionPhase.RECOVERING
    assert not skipped.map_trusted
    assert settled.phase is VisionPhase.IDLE
    assert settled.map_trusted


@pytest.mark.unit
def test_missing_motion_assessments_never_add_quiet_dwell_or_settle():
    scheduler = VisionPhaseScheduler(_settings())
    scheduler.update(
        now=0.0,
        motion_assessment=_assessment(0.0),
        ball_map_observation=_map(0.0),
    )

    second = scheduler.update(
        now=0.1,
        ball_map_observation=_map(0.1),
    )
    third = scheduler.update(
        now=0.2,
        ball_map_observation=_map(0.2),
    )

    assert not second.map_trusted
    assert not third.map_trusted
    assert third.phase is VisionPhase.RECOVERING


@pytest.mark.unit
def test_motion_monitor_can_acquire_stable_map_before_stable_reference_exists():
    scheduler = VisionPhaseScheduler(_settings())
    first_frame = replace(
        _assessment(0.0),
        uncertain=True,
        uncertainty_reasons=("previous_frame_unavailable",),
    )
    no_stable_reference = replace(
        _assessment(0.1),
        uncertain=True,
        uncertainty_reasons=("stable_reference_unavailable",),
    )
    later_without_reference = replace(
        _assessment(0.19),
        uncertain=True,
        uncertainty_reasons=("stable_reference_unavailable",),
    )
    final_quiet = replace(
        _assessment(0.3),
        uncertain=True,
        uncertainty_reasons=("stable_reference_unavailable",),
    )

    first = scheduler.update(now=0.0, motion_assessment=first_frame)
    second = scheduler.update(
        now=0.1,
        motion_assessment=no_stable_reference,
        ball_map_observation=_map(0.1),
    )
    third = scheduler.update(
        now=0.19,
        motion_assessment=later_without_reference,
        ball_map_observation=_map(0.19, x=101.0),
    )
    settled = scheduler.update(now=0.3, motion_assessment=final_quiet)

    assert first.phase is VisionPhase.RECOVERING
    assert not first.map_trusted
    assert not second.map_trusted
    assert not third.map_trusted
    assert settled.phase is VisionPhase.IDLE
    assert settled.map_trusted


@pytest.mark.unit
def test_motion_immediately_exits_aiming_and_disables_live_aim_eligibility():
    scheduler = VisionPhaseScheduler(_settings())
    _settle_to_idle(scheduler)
    aiming = scheduler.update(
        now=0.21,
        cue_observation=CueObservation(timestamp=0.21, detected=True),
        aiming_requested=True,
        calibration_revision="cal-a",
    )
    assert aiming.phase is VisionPhase.AIMING
    assert aiming.can_calculate_aim

    moving = scheduler.update(
        now=0.22,
        motion_assessment=_assessment(0.22, moving=True),
        moving_ball_count=1,
        aiming_requested=True,
        calibration_revision="cal-a",
    )

    assert moving.phase is VisionPhase.LIMITED_MOTION
    assert not moving.map_trusted
    assert not moving.can_calculate_aim
    assert moving.track_moving_balls


@pytest.mark.unit
def test_fresh_cue_enters_aiming_by_default_but_explicit_false_overrides_it():
    inferred = VisionPhaseScheduler(_settings())
    _settle_to_idle(inferred)
    automatic = inferred.update(
        now=0.21,
        cue_observation=CueObservation(timestamp=0.21, detected=True),
        calibration_revision="cal-a",
    )

    explicit = VisionPhaseScheduler(_settings())
    _settle_to_idle(explicit)
    disabled = explicit.update(
        now=0.21,
        cue_observation=CueObservation(timestamp=0.21, detected=True),
        aiming_requested=False,
        calibration_revision="cal-a",
    )

    assert automatic.phase is VisionPhase.AIMING
    assert automatic.can_calculate_aim
    assert disabled.phase is VisionPhase.IDLE
    assert not disabled.can_calculate_aim


@pytest.mark.unit
def test_motion_threshold_hysteresis_does_not_chatter_between_entry_and_exit():
    scheduler = VisionPhaseScheduler(_settings())
    _settle_to_idle(scheduler)

    entered = scheduler.update(
        now=0.21,
        motion_assessment=_assessment(0.21, moving=True),
        moving_ball_count=1,
        calibration_revision="cal-a",
    )
    hysteresis_band = scheduler.update(
        now=0.22,
        motion_assessment=replace(
            _assessment(0.22), motion_score=0.002, active_motion_fraction=0.002
        ),
        moving_ball_count=1,
        calibration_revision="cal-a",
    )
    remained_active_in_band = scheduler._motion_latched
    below_exit = scheduler.update(
        now=0.23,
        motion_assessment=replace(
            _assessment(0.23), motion_score=0.0005, active_motion_fraction=0.0005
        ),
        moving_ball_count=1,
        calibration_revision="cal-a",
    )

    assert entered.phase is VisionPhase.LIMITED_MOTION
    assert hysteresis_band.phase is VisionPhase.LIMITED_MOTION
    assert remained_active_in_band
    assert below_exit.phase is VisionPhase.LIMITED_MOTION
    assert not scheduler._motion_latched
    assert scheduler._last_motion_quiet


@pytest.mark.unit
def test_persistent_manual_repositioning_invalidates_once_then_allows_reconciliation():
    scheduler = VisionPhaseScheduler(_settings())
    _settle_to_idle(scheduler)

    first_changed_map = scheduler.update(
        now=0.21,
        motion_assessment=_assessment(0.21, persistent=True),
        ball_map_observation=_map(0.21, x=101.0),
        calibration_revision="cal-a",
    )
    second_changed_map = scheduler.update(
        now=0.31,
        motion_assessment=_assessment(0.31, persistent=True),
        ball_map_observation=_map(0.31, x=102.0),
        calibration_revision="cal-a",
    )
    reconciled = scheduler.update(
        now=0.42,
        motion_assessment=_assessment(0.42, persistent=True),
        ball_map_observation=_map(0.42, x=103.0),
        calibration_revision="cal-a",
    )

    assert first_changed_map.phase is VisionPhase.RECOVERING
    assert not first_changed_map.map_trusted
    assert second_changed_map.map_trusted
    assert second_changed_map.phase is VisionPhase.IDLE
    assert reconciled.phase is VisionPhase.IDLE
    assert reconciled.map_trusted


@pytest.mark.unit
def test_many_moving_uses_count_hysteresis_and_exits_through_recovery():
    scheduler = VisionPhaseScheduler(_settings())
    _settle_to_idle(scheduler)
    scheduler.update(
        now=0.21,
        motion_assessment=_assessment(0.21, moving=True),
        moving_ball_count=3,
    )
    many = scheduler.update(
        now=0.32,
        motion_assessment=_assessment(0.32, moving=True),
        moving_ball_count=3,
    )

    assert many.phase is VisionPhase.MANY_MOVING
    assert not many.track_moving_balls
    assert not many.verify_ball_map
    assert not many.detect_cue
    assert not many.can_calculate_aim

    scheduler.update(
        now=0.42,
        motion_assessment=_assessment(0.42),
        moving_ball_count=1,
    )
    scheduler.update(
        now=0.52,
        motion_assessment=_assessment(0.52),
        moving_ball_count=1,
    )
    recovering = scheduler.update(
        now=0.62,
        motion_assessment=_assessment(0.62),
        moving_ball_count=1,
    )

    assert recovering.phase is VisionPhase.RECOVERING
    assert not recovering.map_trusted


@pytest.mark.unit
def test_many_moving_does_not_exit_without_fresh_low_count_and_quiet_dwell():
    scheduler = VisionPhaseScheduler(_settings())
    scheduler.update(
        now=0.0,
        motion_assessment=_assessment(0.0, moving=True),
        moving_ball_count=3,
    )
    scheduler.update(
        now=0.1,
        motion_assessment=_assessment(0.1, moving=True),
        moving_ball_count=3,
    )
    assert scheduler.phase is VisionPhase.MANY_MOVING

    one_quiet_sample = scheduler.update(
        now=0.2,
        motion_assessment=_assessment(0.2),
        moving_ball_count=0,
    )
    still_many = scheduler.update(
        now=0.3,
        motion_assessment=_assessment(0.3),
        moving_ball_count=None,
    )

    assert one_quiet_sample.phase is VisionPhase.MANY_MOVING
    assert still_many.phase is VisionPhase.MANY_MOVING


@pytest.mark.unit
@pytest.mark.parametrize("uncertainty", ["occlusion", "illumination", "ambiguous"])
def test_scene_uncertainty_blocks_map_trust(uncertainty):
    scheduler = VisionPhaseScheduler(_settings())
    assessment = {
        "occlusion": _assessment(0.0, occlusion=True),
        "illumination": _assessment(0.0, illumination=True),
        "ambiguous": _assessment(0.0, uncertain=True),
    }[uncertainty]

    decision = scheduler.update(
        now=0.0,
        motion_assessment=assessment,
        ball_map_observation=_map(0.0),
    )
    assert decision.phase is VisionPhase.RECOVERING
    assert not decision.map_trusted


@pytest.mark.unit
def test_inconsistent_map_and_explicit_failed_observation_restart_reacquisition():
    scheduler = VisionPhaseScheduler(_settings())
    assert _settle_to_idle(scheduler).map_trusted

    changed = scheduler.update(
        now=0.3,
        motion_assessment=_assessment(0.3),
        ball_map_observation=_map(0.3, x=150.0),
        calibration_revision="cal-a",
    )
    assert changed.phase is VisionPhase.RECOVERING
    assert not changed.map_trusted

    failed = scheduler.update(
        now=0.4,
        motion_assessment=_assessment(0.4),
        ball_map_observation=BallMapObservation(
            timestamp=0.4,
            balls=(),
            full=True,
            successful=False,
        ),
        calibration_revision="cal-a",
    )
    assert failed.phase is VisionPhase.RECOVERING
    assert not failed.map_trusted


@pytest.mark.unit
def test_calibration_revision_change_resets_trust_and_requires_reacquisition():
    scheduler = VisionPhaseScheduler(_settings())
    assert _settle_to_idle(scheduler).map_trusted
    old_revision = scheduler.map_revision

    changed = scheduler.update(
        now=0.3,
        motion_assessment=_assessment(0.3),
        calibration_revision="cal-b",
    )

    assert changed.phase is VisionPhase.RECOVERING
    assert not changed.map_trusted
    assert changed.map_revision > old_revision


@pytest.mark.unit
def test_aiming_requires_fresh_reliable_cue_but_lost_cue_keeps_idle_map():
    scheduler = VisionPhaseScheduler(_settings())
    _settle_to_idle(scheduler)
    no_cue = scheduler.update(
        now=0.21,
        aiming_requested=True,
        calibration_revision="cal-a",
    )
    assert no_cue.phase is VisionPhase.IDLE
    assert no_cue.map_trusted

    aimed = scheduler.update(
        now=0.22,
        cue_observation=CueObservation(timestamp=0.22, detected=True, reliable=True),
        aiming_requested=True,
        calibration_revision="cal-a",
    )
    assert aimed.phase is VisionPhase.AIMING

    lost = scheduler.update(
        now=0.4,
        aiming_requested=True,
        calibration_revision="cal-a",
    )
    assert lost.phase is VisionPhase.IDLE
    assert lost.map_trusted


@pytest.mark.unit
def test_work_items_are_scheduled_independently_by_phase_and_due_time():
    intervals = _settings()["intervals_seconds"]
    intervals["IDLE"] = {
        "motion_monitor": 0.1,
        "ball_map_verification": 0.5,
        "cue_detection": 0.2,
        "detailed_tracking": 0.0,
    }
    scheduler = VisionPhaseScheduler(_settings(intervals_seconds=intervals))
    _settle_to_idle(scheduler)

    first = scheduler.update(
        now=0.21,
        calibration_revision="cal-a",
    )
    second = scheduler.update(
        now=0.31,
        calibration_revision="cal-a",
    )
    third = scheduler.update(
        now=0.42,
        calibration_revision="cal-a",
    )
    fourth = scheduler.update(
        now=0.72,
        calibration_revision="cal-a",
    )

    assert not first.run_motion_monitor
    assert not first.verify_ball_map
    assert not first.detect_cue
    assert not first.track_moving_balls
    assert second.run_motion_monitor
    assert not second.verify_ball_map
    assert not second.detect_cue
    assert third.run_motion_monitor
    assert not third.verify_ball_map
    assert third.detect_cue
    assert fourth.run_motion_monitor
    assert fourth.verify_ball_map
    assert fourth.detect_cue


@pytest.mark.unit
def test_phase_processing_dimension_is_opt_in_and_selected_for_current_phase():
    dimensions = {phase.value: None for phase in VisionPhase}
    dimensions[VisionPhase.RECOVERING.value] = 720
    decision = VisionPhaseScheduler(
        _settings(processing_max_dimension=dimensions)
    ).update(now=0.0)

    assert decision.phase is VisionPhase.RECOVERING
    assert decision.processing_max_dimension == 720


@pytest.mark.unit
def test_invalid_or_non_monotonic_scheduler_time_is_rejected():
    scheduler = VisionPhaseScheduler(_settings())
    with pytest.raises(ValueError, match="finite"):
        scheduler.update(now=float("nan"))
    scheduler.update(now=1.0)
    with pytest.raises(ValueError, match="monotonic"):
        scheduler.update(now=0.9)
