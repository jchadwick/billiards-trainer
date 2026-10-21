"""Hardware-free tests for calibrated table motion monitoring."""

import cv2
import numpy as np
import pytest

from backend.vision.motion_monitor import MotionMonitor


@pytest.fixture
def polygon():
    return [(10, 10), (245, 10), (245, 150), (10, 150)]


@pytest.fixture
def textured_frame():
    rng = np.random.default_rng(19)
    gray = rng.integers(35, 205, size=(160, 256), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _monitor(polygon):
    return MotionMonitor(
        {
            "difference_threshold": 16,
            "active_motion_fraction_threshold": 0.001,
            "persistent_change_fraction_threshold": 0.002,
            "minimum_region_area_px": 12,
            "minimum_region_area_fraction": 0.0001,
            "likely_occlusion_min_area_fraction": 0.008,
            "likely_occlusion_max_area_fraction": 0.4,
            "illumination_median_delta_threshold": 10,
            "illumination_contrast_change_threshold": 0.15,
        },
        table_polygon=polygon,
    )


@pytest.mark.unit
def test_missing_calibration_and_empty_frame_are_invalid(textured_frame):
    monitor = MotionMonitor({})

    missing = monitor.assess(textured_frame)
    empty = monitor.assess(np.empty((0, 0, 3), dtype=np.uint8), [])

    assert not missing.valid
    assert missing.uncertain
    assert "invalid_frame_or_calibration" in missing.uncertainty_reasons
    assert not empty.valid
    assert empty.uncertain


@pytest.mark.unit
def test_assessment_never_adapts_stable_reference_without_both_guards(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame,
        settled=True,
        map_reconciled=True,
        timestamp=1.0,
    )
    changed = textured_frame.copy()
    cv2.rectangle(changed, (80, 60), (100, 80), (245, 245, 245), -1)

    assert not monitor.commit_verified_reference(
        changed,
        settled=True,
        map_reconciled=False,
        timestamp=2.0,
    )
    first = monitor.assess(changed, timestamp=3.0)
    second = monitor.assess(changed, timestamp=4.0)

    assert first.persistent_change_detected
    assert first.stable_scene_change_fraction is not None
    assert first.stable_scene_change_fraction > 0
    assert second.persistent_change_detected
    assert second.active_motion_fraction == pytest.approx(0.0)
    assert monitor.stable_reference_timestamp == 1.0


@pytest.mark.unit
def test_verified_commit_replaces_reference_and_resets_active_baseline(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    changed = textured_frame.copy()
    cv2.rectangle(changed, (80, 60), (100, 80), (245, 245, 245), -1)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True, timestamp=1.0
    )
    assert monitor.assess(changed).persistent_change_detected

    assert monitor.commit_verified_reference(
        changed, settled=True, map_reconciled=True, timestamp=8.5
    )
    assessment = monitor.assess(changed)

    assert assessment.stable_scene_change_fraction == pytest.approx(0.0)
    assert assessment.active_motion_fraction == pytest.approx(0.0)
    assert not assessment.persistent_change_detected
    assert monitor.stable_reference_timestamp == 8.5


@pytest.mark.unit
def test_frame_motion_and_candidate_neighborhoods_are_reported(polygon, textured_frame):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True
    )
    first_ball = textured_frame.copy()
    second_ball = textured_frame.copy()
    cv2.circle(first_ball, (70, 80), 7, (250, 250, 250), -1)
    cv2.circle(second_ball, (95, 80), 7, (250, 250, 250), -1)

    monitor.assess(first_ball, candidate_balls=[(70, 80, 7)])
    assessment = monitor.assess(
        second_ball,
        candidate_balls=[{"position": (95, 80), "radius": 7}],
        timestamp=9.0,
    )

    assert assessment.motion_detected
    assert assessment.active_motion_fraction is not None
    assert assessment.active_motion_fraction > 0
    assert assessment.candidate_ball_changes[0].active_change_fraction > 0
    assert assessment.candidate_ball_changes[0].stable_change_fraction > 0
    assert assessment.active_changed_regions


@pytest.mark.unit
def test_global_illumination_shift_is_flagged_without_false_motion(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True
    )
    brighter = cv2.add(textured_frame, np.full_like(textured_frame, 24))

    assessment = monitor.assess(brighter)

    assert assessment.illumination_change
    assert assessment.illumination_median_delta == pytest.approx(24.0, abs=1.0)
    assert assessment.illumination_change_score >= 1.0
    assert not assessment.motion_detected
    assert not assessment.persistent_change_detected


@pytest.mark.unit
def test_persistent_localized_change_is_flagged_as_likely_occlusion(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True
    )
    obstruction = textured_frame.copy()
    cv2.rectangle(obstruction, (90, 60), (150, 95), (235, 235, 235), -1)

    assessment = monitor.assess(obstruction)

    assert assessment.persistent_change_detected
    assert assessment.changed_regions
    assert assessment.likely_occlusion
    assert assessment.occlusion_score > 0


@pytest.mark.unit
def test_changes_outside_calibrated_table_do_not_count_as_motion(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True
    )
    outside_change = textured_frame.copy()
    cv2.rectangle(outside_change, (0, 0), (8, 8), (255, 255, 255), -1)

    assessment = monitor.assess(outside_change)

    assert assessment.active_motion_fraction == pytest.approx(0.0)
    assert assessment.stable_scene_change_fraction == pytest.approx(0.0)
    assert not assessment.motion_detected


@pytest.mark.unit
def test_resolution_change_invalidates_references_conservatively(
    polygon, textured_frame
):
    monitor = _monitor(polygon)
    assert monitor.commit_verified_reference(
        textured_frame, settled=True, map_reconciled=True
    )
    resized = cv2.resize(textured_frame, (128, 80))
    resized_polygon = [(5, 5), (122, 5), (122, 75), (5, 75)]

    assessment = monitor.assess(resized, resized_polygon)

    assert assessment.valid
    assert assessment.uncertain
    assert assessment.active_motion_fraction is None
    assert assessment.stable_scene_change_fraction is None
    assert "calibration_or_resolution_changed" in assessment.uncertainty_reasons
    assert not monitor.has_stable_reference


@pytest.mark.unit
def test_invalid_commit_is_noop_and_configuration_values_are_sanitized(
    polygon, textured_frame
):
    monitor = MotionMonitor(
        {
            "difference_threshold": float("nan"),
            "active_motion_fraction_threshold": 4,
            "likely_occlusion_min_area_fraction": 0.5,
            "likely_occlusion_max_area_fraction": 0.1,
        },
        table_polygon=polygon,
    )
    assert monitor.settings["difference_threshold"] == 20
    assert monitor.settings["active_motion_fraction_threshold"] == 1.0
    assert monitor.settings["likely_occlusion_max_area_fraction"] == 0.5

    assert not monitor.commit_verified_reference(
        textured_frame, settled=False, map_reconciled=True
    )
    assert not monitor.has_stable_reference
