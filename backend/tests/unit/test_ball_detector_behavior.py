"""Behavior-focused tests for the classic OpenCV ball detector.

Synthetic frames keep the assertions deterministic while exercising the real
color, geometry, and output-model logic. Small detector fakes are used only at
the candidate-generation seam where the OpenCV algorithms themselves are not
the subject of the test.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from backend.common.models import Ball
from backend.vision.detection.balls import (
    BallDetectionConfig,
    BallDetector,
    DetectionMethod,
)
from backend.vision.models import BallType


def _detector(
    config: dict[str, Any] | None = None,
    *,
    resolution: tuple[int, int] = (1920, 1080),
) -> BallDetector:
    return BallDetector(config or {}, camera_resolution=resolution)


def _solid_frame(
    *,
    size: tuple[int, int] = (320, 240),
    balls: tuple[tuple[tuple[int, int], int, tuple[int, int, int]], ...] = (
        ((100, 100), 18, (0, 255, 255)),
    ),
) -> np.ndarray[Any, Any]:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for center, radius, bgr in balls:
        cv2.circle(frame, center, radius, bgr, -1)
    return frame


@pytest.mark.unit
def test_config_scales_ball_geometry_and_parses_detection_method():
    config = BallDetectionConfig.from_config(
        {
            "detection_method": "contour",
            "performance": {"roi_margin": 25},
        },
        camera_resolution=(1920, 1080),
    )

    assert config.detection_method is DetectionMethod.CONTOUR_BASED
    assert (config.min_radius, config.expected_radius, config.max_radius) == (
        12,
        18,
        23,
    )
    # Explicit config values are already in target-frame pixels.
    assert config.roi_margin == 25
    assert config.camera_resolution == (1920, 1080)


@pytest.mark.unit
def test_config_rejects_unknown_detection_method():
    with pytest.raises(ValueError, match="not-a-detector"):
        BallDetectionConfig.from_config({"detection_method": "not-a-detector"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("hsv", "expected_type", "expected_number"),
    [
        ((30, 220, 230), BallType.SOLID_1, 1),
        ((110, 220, 230), BallType.SOLID_2, 2),
        ((0, 0, 245), BallType.CUE, None),
    ],
)
def test_classifies_uniform_solid_colors_and_cue_ball(
    hsv, expected_type, expected_number
):
    detector = _detector()
    bgr_color = tuple(
        int(channel)
        for channel in cv2.cvtColor(
            np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0, 0]
    )
    region = np.full((48, 48, 3), bgr_color, dtype=np.uint8)

    ball_type, confidence, number = detector.classify_ball_type(
        region, (100.0, 100.0), 18.0
    )

    assert ball_type is expected_type
    assert number == expected_number
    assert confidence >= 0.9


@pytest.mark.unit
def test_classifies_eight_ball_with_a_specular_highlight():
    detector = _detector()
    region = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.circle(region, (24, 24), 3, (255, 255, 255), -1)

    ball_type, confidence, number = detector.classify_ball_type(
        region, (100.0, 100.0), 18.0
    )

    assert (ball_type, number) == (BallType.EIGHT, 8)
    assert confidence > 0.9


@pytest.mark.unit
def test_stripe_pattern_distinguishes_striped_from_solid_number():
    detector = _detector()
    striped = np.zeros((48, 48, 3), dtype=np.uint8)
    striped[:24] = (0, 255, 255)  # Yellow half
    striped[24:] = (245, 245, 245)  # White half
    solid = np.full((48, 48, 3), (0, 255, 255), dtype=np.uint8)

    assert detector._is_striped_ball(striped)[0] is True
    assert detector._color_to_ball_type("yellow", striped) == (BallType.STRIPE_9, 9)
    assert detector._color_to_ball_type("yellow", solid) == (BallType.SOLID_1, 1)


@pytest.mark.unit
def test_empty_or_unconvertible_classification_region_returns_unknown():
    detector = _detector()

    assert detector.classify_ball_type(
        np.empty((0, 0, 3), dtype=np.uint8), (0.0, 0.0), 18.0
    ) == (BallType.UNKNOWN, 0.0, None)
    # OpenCV cannot convert a one-channel frame using the requested BGR mode;
    # the public classifier deliberately degrades to unknown instead of raising.
    assert detector.classify_ball_type(
        np.zeros((16, 16), dtype=np.uint8), (0.0, 0.0), 5.0
    ) == (BallType.UNKNOWN, 0.0, None)


@pytest.mark.unit
def test_filter_validation_keeps_bright_correct_size_ball_and_rejects_bad_geometry():
    detector = _detector()
    frame = _solid_frame(
        size=(240, 180),
        balls=(((50, 50), 18, (0, 255, 255)), ((100, 50), 18, (0, 255, 255))),
    )
    candidates = [
        (50.0, 50.0, 18.0),  # valid
        (100.0, 50.0, 8.0),  # outside the configured radius tolerance
        (8.0, 50.0, 18.0),  # crosses the left image boundary
        (180.0, 120.0, 18.0),  # dark region, likely a shadow
    ]

    assert detector._filter_and_validate(candidates, frame) == [(50.0, 50.0, 18.0)]


@pytest.mark.unit
def test_filter_validation_excludes_candidates_inside_configured_pocket():
    detector = _detector({"pocket_detection": {"exclusion_radius_multiplier": 2.0}})
    detector.pocket_locations = [(50.0, 50.0, 12.0)]
    frame = _solid_frame(
        size=(240, 180),
        balls=(((50, 50), 18, (0, 255, 255)), ((120, 50), 18, (0, 255, 255))),
    )

    assert detector._filter_and_validate(
        [(50.0, 50.0, 18.0), (120.0, 50.0, 18.0)], frame
    ) == [(120.0, 50.0, 18.0)]


@pytest.mark.unit
def test_shadow_filter_removes_nearby_darker_candidate_but_keeps_isolated_ball():
    detector = _detector()
    frame = _solid_frame(
        size=(200, 100),
        balls=(
            ((40, 50), 10, (0, 255, 255)),
            ((60, 50), 10, (0, 0, 90)),
            ((150, 50), 10, (255, 0, 0)),
        ),
    )

    filtered = detector._filter_ball_shadows(
        [(40.0, 50.0, 10.0), (60.0, 50.0, 10.0), (150.0, 50.0, 10.0)], frame
    )

    assert filtered == [(40.0, 50.0, 10.0), (150.0, 50.0, 10.0)]


@pytest.mark.unit
def test_color_prefilter_retains_white_ball_and_excludes_table_and_dark_regions():
    detector = _detector()
    frame = np.full((80, 120, 3), (0, 180, 0), dtype=np.uint8)
    cv2.circle(frame, (35, 40), 20, (245, 245, 245), -1)
    cv2.circle(frame, (85, 40), 20, (0, 0, 0), -1)

    mask = detector._create_ball_color_mask(frame)

    assert mask[40, 35] == 255
    assert mask[40, 10] == 0  # green cloth
    assert mask[40, 85] == 0  # pocket or dark shadow


@pytest.mark.unit
def test_foreground_mask_tracks_a_new_ball_against_saved_background():
    detector = _detector()
    background = np.full((120, 160, 3), (0, 150, 0), dtype=np.uint8)
    detector.background_frame = background
    detector.background_threshold = 30
    current = background.copy()
    cv2.circle(current, (80, 60), 18, (0, 255, 255), -1)

    mask = detector._create_foreground_mask(current)

    assert mask[60, 80] == 255
    assert mask[5, 5] == 0
    detector.background_frame = None
    with pytest.raises(ValueError, match="background frame"):
        detector._create_foreground_mask(current)


@pytest.mark.unit
def test_set_background_frame_discovers_dark_pockets_and_cloth_area():
    detector = _detector()
    background = np.full((600, 800, 3), (0, 0, 180), dtype=np.uint8)
    cv2.rectangle(background, (40, 40), (760, 560), (20, 100, 20), -1)
    expected_centers = [
        (80, 80),
        (400, 80),
        (720, 80),
        (80, 520),
        (400, 520),
        (720, 520),
    ]
    for center in expected_centers:
        cv2.circle(background, center, 25, (0, 0, 0), -1)

    detector.set_background_frame(background)

    assert detector.use_background_subtraction is True
    assert detector.background_frame is not background
    assert len(detector.pocket_locations) == 6
    assert detector.playing_area_mask is not None
    assert detector.playing_area_mask[300, 400] == 255
    assert detector.playing_area_mask[0, 0] == 0
    assert all(
        min(math.dist(center, pocket[:2]) for pocket in detector.pocket_locations) < 2
        for center in expected_centers
    )


@pytest.mark.unit
def test_detect_balls_runs_selected_candidate_path_and_returns_pixel_provenance(
    monkeypatch,
):
    detector = _detector({"detection_method": "hough"})
    frame = _solid_frame(size=(320, 240))
    monkeypatch.setattr(
        detector, "_detect_hough_circles", lambda _frame: [(100.0, 100.0, 18.0)]
    )
    monkeypatch.setattr(
        detector,
        "_detect_contour_based",
        lambda _frame: pytest.fail("contour path should not run for hough mode"),
    )
    monkeypatch.setattr(
        detector,
        "_detect_blob_based",
        lambda _frame: pytest.fail("blob path should not run for hough mode"),
    )

    balls = detector.detect_balls(frame)

    assert len(balls) == 1
    ball = balls[0]
    assert (ball.ball_type, ball.number) == (BallType.SOLID_1, 1)
    assert ball.position == (100.0, 100.0)
    assert ball.coordinate_space == "pixel"
    assert ball.source_resolution == (320, 240)
    assert detector.stats["total_detections"] == 1
    assert detector.stats["successful_detections"] == 1


@pytest.mark.unit
def test_detect_balls_applies_table_mask_before_candidate_search(monkeypatch):
    detector = _detector({"detection_method": "hough"})
    frame = _solid_frame(size=(320, 240))
    seen_nonzero: list[int] = []

    def find_candidates(masked_frame: np.ndarray[Any, Any]):
        seen_nonzero.append(int(np.count_nonzero(masked_frame)))
        return [(100.0, 100.0, 18.0)] if seen_nonzero[-1] else []

    monkeypatch.setattr(detector, "_detect_hough_circles", find_candidates)
    table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    assert detector.detect_balls(frame, table_mask=table_mask) == []
    assert seen_nonzero == [0]
    assert detector.stats["total_detections"] == 1
    assert detector.stats["successful_detections"] == 0


@pytest.mark.unit
def test_detect_balls_handles_invalid_input_and_candidate_detector_failure(monkeypatch):
    detector = _detector({"detection_method": "hough"})
    assert detector.detect_balls(np.empty((0, 0, 3), dtype=np.uint8)) == []
    assert detector.stats["total_detections"] == 0

    def fail(_frame):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(detector, "_detect_hough_circles", fail)
    assert detector.detect_balls(_solid_frame()) == []
    # Failed frames do not masquerade as successful or completed observations.
    assert detector.stats["total_detections"] == 0


@pytest.mark.unit
def test_hough_candidates_are_rounded_and_detection_parameters_are_applied(monkeypatch):
    detector = _detector({"detection_method": "hough"})
    detector.config.min_radius = 12
    frame = np.full((240, 320, 3), 230, dtype=np.uint8)
    calls: list[dict[str, Any]] = []

    def hough(_image, method, **kwargs):
        calls.append({"method": method, **kwargs})
        return np.array([[[100.4, 80.6, 17.7]]], dtype=np.float32)

    monkeypatch.setattr(cv2, "HoughCircles", hough)
    candidates = detector._detect_hough_circles(frame)

    assert candidates == [(100.0, 81.0, 18.0)]
    assert calls[0]["minDist"] == int(12 * 2 * detector.config.hough_min_dist_ratio)
    assert calls[0]["minRadius"] == detector.config.min_radius
    assert calls[0]["maxRadius"] == detector.config.max_radius
    assert detector.stats["detection_method_stats"]["hough"] == 1


@pytest.mark.unit
def test_blob_path_enforces_radius_limits_and_updates_method_statistics():
    detector = _detector({"detection_method": "blob"})
    detector.blob_detector = SimpleNamespace(
        detect=lambda _image: [
            SimpleNamespace(pt=(40.0, 50.0), size=36.0),
            SimpleNamespace(pt=(80.0, 50.0), size=8.0),
            SimpleNamespace(pt=(120.0, 50.0), size=60.0),
        ]
    )

    candidates = detector._detect_blob_based(
        np.full((100, 180, 3), 220, dtype=np.uint8)
    )

    assert candidates == [(40.0, 50.0, 18.0)]
    assert detector.stats["detection_method_stats"]["blob"] == 1


@pytest.mark.unit
def test_combined_candidates_merge_duplicate_centers_and_keep_separate_ball(
    monkeypatch,
):
    detector = _detector()
    monkeypatch.setattr(
        detector, "_detect_hough_circles", lambda _f: [(50.0, 50.0, 18.0)]
    )
    monkeypatch.setattr(
        detector, "_detect_contour_based", lambda _f: [(52.0, 50.0, 20.0)]
    )
    monkeypatch.setattr(
        detector, "_detect_blob_based", lambda _f: [(120.0, 50.0, 18.0)]
    )

    combined = detector._detect_combined(np.zeros((200, 240, 3), dtype=np.uint8))

    assert combined == [(51.0, 50.0, 20.0), (120.0, 50.0, 18.0)]
    assert detector.stats["detection_method_stats"] == {
        "hough": 0,
        "contour": 0,
        "blob": 0,
        "combined": 2,
    }


@pytest.mark.unit
def test_overlap_resolution_merges_ghosts_and_keeps_nonoverlapping_ball():
    detector = _detector()
    balls = [
        Ball(
            position=(100.0, 80.0),
            radius=18.0,
            ball_type=BallType.SOLID_1,
            confidence=0.9,
        ),
        Ball(
            position=(103.0, 80.0),
            radius=20.0,
            ball_type=BallType.SOLID_1,
            confidence=0.6,
        ),
        Ball(
            position=(145.0, 80.0),
            radius=18.0,
            ball_type=BallType.SOLID_2,
            confidence=0.8,
        ),
    ]

    resolved = detector._remove_overlaps(balls)

    assert len(resolved) == 2
    merged = next(ball for ball in resolved if ball.ball_type is BallType.SOLID_1)
    assert merged.position == pytest.approx((101.2, 80.0))
    assert merged.radius == pytest.approx(18.8)
    assert merged.confidence == pytest.approx(0.75)
    assert any(ball.position == (145.0, 80.0) for ball in resolved)


@pytest.mark.unit
def test_type_conflicts_and_duplicate_numbers_keep_highest_confidence():
    detector = _detector()
    balls = [
        Ball(
            position=(20.0, 20.0), radius=18.0, ball_type=BallType.CUE, confidence=0.95
        ),
        Ball(
            position=(60.0, 20.0), radius=18.0, ball_type=BallType.CUE, confidence=0.75
        ),
        Ball(
            position=(100.0, 20.0),
            radius=18.0,
            ball_type=BallType.EIGHT,
            confidence=0.92,
        ),
        Ball(
            position=(140.0, 20.0),
            radius=18.0,
            ball_type=BallType.EIGHT,
            confidence=0.70,
        ),
        Ball(
            position=(180.0, 20.0),
            radius=18.0,
            ball_type=BallType.SOLID_3,
            confidence=0.9,
        ),
        Ball(
            position=(220.0, 20.0),
            radius=18.0,
            ball_type=BallType.SOLID_3,
            confidence=0.8,
        ),
    ]

    detector._resolve_ball_type_conflicts(balls)
    detector._apply_confidence_threshold_and_resolve_duplicates(balls)

    assert [ball.ball_type for ball in balls] == [
        BallType.CUE,
        BallType.UNKNOWN,
        BallType.EIGHT,
        BallType.UNKNOWN,
        BallType.SOLID_3,
        BallType.UNKNOWN,
    ]
    assert [balls[i].number for i in (1, 3, 5)] == [None, None, None]


@pytest.mark.unit
def test_debug_frames_are_copied_and_can_be_cleared():
    detector = _detector({"debug": {"debug_mode": True}})
    frame = _solid_frame(size=(320, 240))
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            detector, "_detect_hough_circles", lambda _f: [(100.0, 100.0, 18.0)]
        )
        detector.detect_balls(frame)
    finally:
        monkeypatch.undo()

    images = detector.get_debug_images()
    assert [name for name, _image in images] == ["input_frame", "candidates"]
    assert images[0][1] is not frame
    detector.clear_debug_images()
    assert detector.get_debug_images() == []
