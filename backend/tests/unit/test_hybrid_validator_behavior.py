"""Deterministic contracts for classical ball refinement and feature extraction."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from backend.common.models import Ball
from backend.vision.detection import hybrid_validator as hybrid
from backend.vision.models import BallType


@pytest.fixture
def frame() -> np.ndarray:
    """Return a deterministic synthetic BGR scene with a clear ball-like disk."""
    image = np.full((120, 160, 3), (30, 100, 30), dtype=np.uint8)
    cv2.circle(image, (80, 60), 18, (245, 245, 245), -1)
    cv2.circle(image, (80, 60), 18, (30, 30, 30), 2)
    cv2.rectangle(image, (65, 56), (95, 64), (40, 40, 40), -1)
    return image


def ball(
    *,
    position: tuple[float, float] = (80.0, 60.0),
    radius: float = 18.0,
    confidence: float = 0.9,
    coordinate_space: str = "pixel",
    source_resolution: tuple[int, int] = (160, 120),
    ball_type: BallType = BallType.CUE,
) -> Ball:
    return Ball(
        position=position,
        radius=radius,
        ball_type=ball_type,
        confidence=confidence,
        velocity=(1.0, 2.0),
        acceleration=(0.1, 0.2),
        is_moving=True,
        track_id=7,
        age=4,
        hit_count=2,
        color_hsv=(30, 40, 50),
        predicted_position=(82.0, 64.0),
        position_history=[(78.0, 58.0)],
        coordinate_space=coordinate_space,
        source_resolution=source_resolution,
    )


def test_low_confidence_skips_all_image_work_and_updates_fallback_stats(
    monkeypatch: pytest.MonkeyPatch, frame: np.ndarray
) -> None:
    refiner = hybrid.BallPositionRefiner(min_confidence_for_refinement=0.5)
    original = ball(confidence=0.49)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("low-confidence detections must not enter CV refinement")

    monkeypatch.setattr(refiner, "_extract_roi", unexpected)

    assert refiner.refine_ball_position(original, frame) is original
    assert refiner.get_statistics() == {
        "total_refinements": 1,
        "successful_refinements": 0,
        "failed_refinements": 0,
        "fallback_to_yolo": 1,
        "avg_refinement_distance": 0.0,
        "success_rate": 0.0,
    }


@pytest.mark.parametrize(
    ("original", "image", "roi", "detected"),
    [
        (ball(coordinate_space="world"), np.zeros((120, 160, 3), np.uint8), None, None),
        (
            ball(position=(2.0, 2.0), radius=2.0),
            np.zeros((120, 160, 3), np.uint8),
            None,
            None,
        ),
        (ball(), np.zeros((120, 160, 3), np.uint8), np.empty((0, 0), np.uint8), None),
        (
            ball(),
            np.zeros((120, 160, 3), np.uint8),
            np.zeros((30, 30), np.uint8),
            (None, None),
        ),
    ],
)
def test_refinement_failures_preserve_original_detection_and_count_failure(
    monkeypatch: pytest.MonkeyPatch,
    original: Ball,
    image: np.ndarray,
    roi: np.ndarray | None,
    detected: tuple[tuple[float, float] | None, float | None] | None,
) -> None:
    refiner = hybrid.BallPositionRefiner()
    if original.coordinate_space != "world":
        if roi is None:
            monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (None, (0, 0)))
        elif roi.size == 0:
            monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (roi, (0, 0)))
        else:
            monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (roi, (0, 0)))
            monkeypatch.setattr(refiner, "_preprocess_roi", lambda source: source)
            monkeypatch.setattr(
                refiner, "_detect_precise_circle", lambda *_args: detected
            )
            monkeypatch.setattr(
                refiner, "_detect_center_from_edges", lambda *_args: detected
            )

    assert refiner.refine_ball_position(original, image) is original
    assert refiner.get_statistics()["failed_refinements"] == 1
    assert refiner.get_statistics()["successful_refinements"] == 0


def test_refinement_falls_back_from_hough_to_edges_and_preserves_ball_metadata(
    monkeypatch: pytest.MonkeyPatch, frame: np.ndarray
) -> None:
    refiner = hybrid.BallPositionRefiner(max_refinement_distance=5.0)
    original = ball()
    roi = np.zeros((40, 40), dtype=np.uint8)
    calls: list[str] = []
    monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (roi, (60, 40)))
    monkeypatch.setattr(refiner, "_preprocess_roi", lambda image: image)
    monkeypatch.setattr(
        refiner,
        "_detect_precise_circle",
        lambda *_args: (calls.append("hough") or (None, None)),
    )
    monkeypatch.setattr(
        refiner,
        "_detect_center_from_edges",
        lambda *_args: (calls.append("edges") or ((20.5, 19.5), 17.0)),
    )
    monkeypatch.setattr(refiner, "_refine_subpixel", lambda *_args: (81.0, 59.0))

    refined = refiner.refine_ball_position(original, frame)

    assert calls == ["hough", "edges"]
    assert refined is not original
    assert refined.position == (81.0, 59.0)
    assert refined.radius == 17.0
    assert refined.ball_type == original.ball_type
    assert refined.number == original.number
    assert refined.confidence == original.confidence
    assert refined.velocity == original.velocity
    assert refined.acceleration == original.acceleration
    assert refined.is_moving is True
    assert refined.track_id == 7
    assert refined.last_seen == original.last_seen
    assert refined.age == 4
    assert refined.hit_count == 2
    assert refined.color_hsv == (30, 40, 50)
    assert refined.predicted_position == (82.0, 64.0)
    assert refined.position_history == original.position_history
    assert refined.position_history is not original.position_history
    assert refiner.get_statistics()["avg_refinement_distance"] == pytest.approx(
        math.sqrt(0.5**2 + 0.5**2)
    )


def test_refinement_scales_pixel_coordinates_from_source_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    refiner = hybrid.BallPositionRefiner()
    original = ball(
        position=(80.0, 60.0),
        radius=18.0,
        source_resolution=(160, 120),
    )
    roi = np.zeros((20, 20), dtype=np.uint8)
    monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (roi, (30, 20)))
    monkeypatch.setattr(refiner, "_preprocess_roi", lambda source: source)
    monkeypatch.setattr(
        refiner, "_detect_precise_circle", lambda *_args: ((10.0, 10.0), None)
    )
    monkeypatch.setattr(refiner, "_refine_subpixel", lambda *_args: None)

    refined = refiner.refine_ball_position(original, image)

    assert refined.position == (80.0, 60.0)
    assert refined.radius == 18.0
    assert refined.coordinate_space == "pixel"
    assert refined.source_resolution == (160, 120)


def test_refinement_rejects_far_candidate_and_catches_cv_failures(
    monkeypatch: pytest.MonkeyPatch, frame: np.ndarray
) -> None:
    original = ball()
    roi = np.zeros((40, 40), dtype=np.uint8)
    refiner = hybrid.BallPositionRefiner(max_refinement_distance=2.0)
    monkeypatch.setattr(refiner, "_extract_roi", lambda *_args: (roi, (0, 0)))
    monkeypatch.setattr(refiner, "_preprocess_roi", lambda source: source)
    monkeypatch.setattr(
        refiner, "_detect_precise_circle", lambda *_args: ((30.0, 30.0), 18.0)
    )
    assert refiner.refine_ball_position(original, frame) is original
    assert refiner.get_statistics()["failed_refinements"] == 1

    broken = hybrid.BallPositionRefiner()
    monkeypatch.setattr(
        broken, "_extract_roi", lambda *_args: (_ for _ in ()).throw(RuntimeError("cv"))
    )
    assert broken.refine_ball_position(original, frame) is original
    assert broken.get_statistics()["failed_refinements"] == 1


def test_invalid_geometry_and_roi_edges_are_handled_without_opencv_calls(
    frame: np.ndarray,
) -> None:
    invalid = ball(position=(1.0, 1.0), radius=0.1)
    assert hybrid.BallPositionRefiner._ball_geometry_in_frame(invalid, frame) == (
        (1.0, 1.0),
        0.1,
    )
    invalid_space = ball(coordinate_space="world")
    assert (
        hybrid.BallPositionRefiner._ball_geometry_in_frame(invalid_space, frame) is None
    )
    assert (
        hybrid.BallPositionRefiner._ball_geometry_in_frame(
            ball(source_resolution=(0, 120)), frame
        )
        is None
    )
    pixel_without_source = ball()
    pixel_without_source.source_resolution = None
    assert hybrid.BallPositionRefiner._ball_geometry_in_frame(
        pixel_without_source, frame
    ) == ((80.0, 60.0), 18.0)
    assert (
        hybrid.BallPositionRefiner._ball_geometry_in_frame(
            ball(), np.zeros((0, 0, 3), dtype=np.uint8)
        )
        is None
    )

    refiner = hybrid.BallPositionRefiner()
    invalid_roi, invalid_offset = refiner._extract_roi(invalid_space, frame)
    assert invalid_roi is None
    assert invalid_offset == (0, 0)
    roi, offset = refiner._extract_roi(ball(position=(0.0, 0.0)), frame)
    assert roi is not None
    assert roi.size
    assert offset == (0, 0)
    roi, offset = refiner._extract_roi(ball(position=(159.0, 119.0)), frame)
    assert roi is not None
    assert roi.size
    assert offset[0] >= 0
    assert offset[1] >= 0
    tiny, tiny_offset = refiner._extract_roi(
        ball(radius=1.0), np.zeros((8, 8, 3), np.uint8)
    )
    assert tiny is None
    assert tiny_offset == (0, 0)
    too_small, too_small_offset = refiner._extract_roi(
        ball(
            position=(4.0, 4.0),
            radius=2.0,
            source_resolution=(8, 8),
        ),
        np.zeros((8, 8, 3), np.uint8),
    )
    assert too_small is None
    assert too_small_offset == (0, 0)
    outside, outside_offset = refiner._extract_roi(
        ball(position=(-100.0, -100.0)), frame
    )
    assert outside is None
    assert outside_offset == (0, 0)


def test_preprocessing_supports_color_and_grayscale_roi(frame: np.ndarray) -> None:
    refiner = hybrid.BallPositionRefiner()
    color = refiner._preprocess_roi(frame[40:80, 60:100])
    gray = refiner._preprocess_roi(
        cv2.cvtColor(frame[40:80, 60:100], cv2.COLOR_BGR2GRAY)
    )
    assert color.shape == gray.shape == (40, 40)
    assert color.dtype == gray.dtype == np.uint8


def test_hough_circle_search_handles_empty_and_detected_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = hybrid.BallPositionRefiner()
    image = np.zeros((40, 40), dtype=np.uint8)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *_args, **_kwargs: None)
    assert refiner._detect_precise_circle(image, 10.0) == (None, None)

    monkeypatch.setattr(
        cv2,
        "HoughCircles",
        lambda *_args, **_kwargs: np.array([[[20.25, 19.5, 9.0], [5, 5, 8]]]),
    )
    assert refiner._detect_precise_circle(image, 10.0) == ((20.25, 19.5), 9.0)
    monkeypatch.setattr(
        cv2, "HoughCircles", lambda *_args, **_kwargs: np.empty((1, 0, 3))
    )
    assert refiner._detect_precise_circle(image, 3.0) == (None, None)


def test_edge_center_detection_filters_and_accepts_circular_contours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = hybrid.BallPositionRefiner()
    image = np.zeros((80, 80), dtype=np.uint8)
    monkeypatch.setattr(cv2, "Canny", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([], None))
    assert refiner._detect_center_from_edges(image, 12.0) == (None, None)

    # Tiny contours are filtered by expected area, while a round contour is accepted.
    circle = cv2.ellipse2Poly((40, 40), (12, 12), 0, 0, 360, 15).reshape(-1, 1, 2)
    tiny = np.array([[[1, 1]], [[2, 1]], [[2, 2]], [[1, 2]]], dtype=np.int32)
    rectangle = np.array(
        [[20, 20], [38, 20], [38, 38], [20, 38]], dtype=np.int32
    ).reshape(-1, 1, 2)
    monkeypatch.setattr(
        cv2, "findContours", lambda *_args: ([tiny, circle, rectangle], None)
    )
    center, radius = refiner._detect_center_from_edges(image, 12.0)
    assert center == pytest.approx((40.0, 40.0), abs=1.0)
    assert radius == pytest.approx(12.0, abs=1.0)

    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([circle], None))
    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 0.0)
    assert refiner._detect_center_from_edges(image, 12.0) == (None, None)

    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 50.0)
    monkeypatch.setattr(cv2, "moments", lambda *_args: {"m00": 0, "m10": 0, "m01": 0})
    assert refiner._detect_center_from_edges(image, 12.0) == (None, None)

    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([circle], None))
    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 0.0)
    assert refiner._detect_center_from_edges(image, 12.0) == (None, None)

    monkeypatch.setattr(cv2, "arcLength", lambda *_args: 100.0)
    monkeypatch.setattr(cv2, "moments", lambda *_args: {"m00": 0, "m10": 0, "m01": 0})
    assert refiner._detect_center_from_edges(image, 12.0) == (None, None)


def test_subpixel_refinement_success_none_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = hybrid.BallPositionRefiner()
    image = np.zeros((40, 40), dtype=np.uint8)
    monkeypatch.setattr(
        cv2,
        "cornerSubPix",
        lambda *_args: np.array([[[10.25, 8.75]]], dtype=np.float32),
    )
    assert refiner._refine_subpixel(image, (10.0, 9.0), (3, 4)) == pytest.approx(
        (13.25, 12.75)
    )
    monkeypatch.setattr(cv2, "cornerSubPix", lambda *_args: None)
    assert refiner._refine_subpixel(image, (10.0, 9.0), (3, 4)) is None
    monkeypatch.setattr(
        cv2, "cornerSubPix", lambda *_args: (_ for _ in ()).throw(cv2.error("bad"))
    )
    assert refiner._refine_subpixel(image, (10.0, 9.0), (3, 4)) is None


def test_statistics_reset_and_convenience_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refiner = hybrid.BallPositionRefiner(min_confidence_for_refinement=0.99)
    original = ball()
    assert (
        refiner.refine_ball_position(original, np.zeros((10, 10, 3), np.uint8))
        is original
    )
    assert refiner.get_statistics()["success_rate"] == 0.0
    refiner.reset_statistics()
    assert refiner.get_statistics()["total_refinements"] == 0

    monkeypatch.setattr(hybrid, "BallPositionRefiner", lambda: refiner)
    assert (
        hybrid.refine_ball_position(original, np.zeros((10, 10, 3), np.uint8))
        is original
    )


def test_feature_extraction_detects_fixed_ball_types_and_handles_bad_regions(
    frame: np.ndarray,
) -> None:
    cue_features = hybrid.extract_ball_features(ball(), frame, debug=True)
    assert cue_features.ball_type is BallType.CUE
    assert cue_features.detected_number is None
    assert cue_features.number_confidence == 1.0
    assert cue_features.color_histogram.shape == (692,)
    assert np.isfinite(cue_features.extraction_quality)

    eight_features = hybrid.extract_ball_features(ball(ball_type=BallType.EIGHT), frame)
    assert eight_features.detected_number == 8
    assert eight_features.number_confidence == 1.0

    outside = hybrid.extract_ball_features(ball(position=(1000.0, 1000.0)), frame)
    assert outside.extraction_quality == 0.0
    assert outside.ball_type is BallType.CUE
    assert outside.color_histogram.shape == (692,)


def test_feature_helpers_cover_empty_masks_and_number_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = np.zeros((32, 32, 3), dtype=np.uint8)
    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    color = hybrid._extract_color_features(region, empty_mask)
    assert color[0] == (0, 0, 0)
    assert color[1].shape == (692,)
    assert color[2:] == (0.0, 0.0, 0.0)
    assert hybrid._calculate_sharpness(region, empty_mask) == 0.0
    assert hybrid._extract_number_features(region, empty_mask, BallType.CUE) == (
        None,
        1.0,
    )
    assert hybrid._extract_number_features(region, empty_mask, BallType.EIGHT) == (
        8,
        1.0,
    )

    monkeypatch.setattr(
        hybrid, "_preprocess_for_number_recognition", lambda *_args: region[:, :, 0]
    )
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([], None))
    assert hybrid._extract_number_features(region, empty_mask, BallType.SOLID_1) == (
        None,
        0.0,
    )
    monkeypatch.setattr(
        hybrid,
        "_preprocess_for_number_recognition",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bad image")),
    )
    assert hybrid._extract_number_features(
        region, empty_mask, BallType.SOLID_1, debug=True
    ) == (None, 0.0)


def test_number_features_accepts_valid_contour_and_logs_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    square = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.int32).reshape(
        -1, 1, 2
    )
    thin = np.array([[30, 10], [33, 10], [33, 30], [30, 30]], dtype=np.int32).reshape(
        -1, 1, 2
    )
    monkeypatch.setattr(
        hybrid, "_preprocess_for_number_recognition", lambda *_args: region[:, :, 0]
    )
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([thin, square], None))

    number, confidence = hybrid._extract_number_features(
        region, mask, BallType.SOLID_1, debug=True
    )
    assert number == 6
    assert confidence == 0.6

    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([square], None))
    assert hybrid._extract_number_features(
        region, mask, BallType.UNKNOWN, debug=True
    ) == (None, 0.0)

    monkeypatch.setattr(
        hybrid,
        "_preprocess_for_number_recognition",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bad input")),
    )
    assert hybrid._extract_number_features(region, mask, BallType.SOLID_1) == (
        None,
        0.0,
    )


@pytest.mark.parametrize(
    ("ball_type", "contour", "expected"),
    [
        (
            BallType.SOLID_1,
            cv2.ellipse2Poly((30, 30), (10, 10), 0, 0, 360, 10).reshape(-1, 1, 2),
            6,
        ),
        (
            BallType.SOLID_1,
            np.array([[[1, 1]], [[3, 1]], [[3, 8]], [[1, 8]]], dtype=np.int32),
            1,
        ),
        (
            BallType.STRIPE_9,
            cv2.ellipse2Poly((30, 30), (10, 10), 0, 0, 360, 10).reshape(-1, 1, 2),
            9,
        ),
        (
            BallType.STRIPE_9,
            np.array([[[1, 1]], [[3, 1]], [[3, 8]], [[1, 8]]], dtype=np.int32),
            11,
        ),
        (
            BallType.CUE,
            np.array([[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], dtype=np.int32),
            None,
        ),
    ],
)
def test_contour_number_heuristics_are_stable_for_synthetic_shapes(
    ball_type: BallType, contour: np.ndarray, expected: int | None
) -> None:
    assert hybrid._analyze_contour_for_number(contour, ball_type) == expected


def test_pattern_features_on_solid_and_striped_synthetic_patches() -> None:
    height, width = 64, 64
    solid = np.full((height, width, 3), 220, dtype=np.uint8)
    striped = solid.copy()
    striped[:, 22:42] = 30
    mask = hybrid._create_circular_mask(solid, 24.0, 5)

    solid_result = hybrid._extract_pattern_features(solid, mask, 24.0, debug=True)
    stripe_result = hybrid._extract_pattern_features(striped, mask, 24.0)
    for result in (solid_result, stripe_result):
        assert len(result) == 4
        assert 0.0 <= result[1] <= 1.0
        assert 0.0 <= result[2] <= 1.0
        assert np.isfinite(result[3])


def test_number_contour_filtering_and_zero_perimeter_degrade_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    # Both contours are rejected by the area filter.
    tiny = np.array([[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], dtype=np.int32)
    monkeypatch.setattr(
        hybrid, "_preprocess_for_number_recognition", lambda *_args: region[:, :, 0]
    )
    monkeypatch.setattr(cv2, "findContours", lambda *_args: ([tiny], None))
    assert hybrid._extract_number_features(region, mask, BallType.SOLID_1) == (
        None,
        0.0,
    )

    degenerate = np.array([[[1, 1]], [[1, 1]], [[1, 1]]], dtype=np.int32)
    assert hybrid._analyze_contour_for_number(degenerate, BallType.SOLID_1) is None
    assert hybrid._analyze_contour_for_number(tiny, BallType.UNKNOWN) is None


def test_number_heuristics_cover_general_and_concave_shapes() -> None:
    triangle = np.array([[2, 2], [18, 3], [10, 18]], dtype=np.int32).reshape(-1, 1, 2)
    assert hybrid._analyze_contour_for_number(triangle, BallType.SOLID_2) == 2
    assert hybrid._analyze_contour_for_number(triangle, BallType.STRIPE_10) == 12

    concave = np.array(
        [[2, 2], [18, 2], [18, 6], [7, 6], [7, 14], [18, 14], [18, 18], [2, 18]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    assert hybrid._analyze_contour_for_number(concave, BallType.SOLID_3) == 4
    assert hybrid._analyze_contour_for_number(concave, BallType.STRIPE_11) == 14


def test_number_image_preprocessing_returns_binary_image() -> None:
    image = np.full((48, 48, 3), 220, dtype=np.uint8)
    cv2.putText(image, "8", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    mask = np.full((48, 48), 255, dtype=np.uint8)
    result = hybrid._preprocess_for_number_recognition(image, mask)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_number_heuristics_cover_general_and_concave_shapes() -> None:
    triangle = np.array([[2, 2], [18, 3], [10, 18]], dtype=np.int32).reshape(-1, 1, 2)
    assert hybrid._analyze_contour_for_number(triangle, BallType.SOLID_2) == 2
    assert hybrid._analyze_contour_for_number(triangle, BallType.STRIPE_10) == 12

    concave = np.array(
        [[2, 2], [18, 2], [18, 6], [7, 6], [7, 14], [18, 14], [18, 18], [2, 18]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    assert hybrid._analyze_contour_for_number(concave, BallType.SOLID_3) == 4
    assert hybrid._analyze_contour_for_number(concave, BallType.STRIPE_11) == 14


def test_number_image_preprocessing_returns_binary_image() -> None:
    image = np.full((48, 48, 3), 220, dtype=np.uint8)
    cv2.putText(image, "8", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    mask = np.full((48, 48), 255, dtype=np.uint8)
    result = hybrid._preprocess_for_number_recognition(image, mask)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_feature_extraction_degrades_when_stage_raises(
    monkeypatch: pytest.MonkeyPatch, frame: np.ndarray
) -> None:
    monkeypatch.setattr(
        hybrid,
        "_extract_color_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad color")),
    )
    actual = hybrid.extract_ball_features(ball(), frame)
    expected = hybrid._get_default_features(BallType.CUE)
    assert actual.dominant_color_hsv == expected.dominant_color_hsv
    np.testing.assert_array_equal(actual.color_histogram, expected.color_histogram)
    assert actual.extraction_quality == expected.extraction_quality == 0.0


def test_extraction_quality_combines_clamped_quality_components() -> None:
    mask = np.full((40, 40), 255, dtype=np.uint8)
    quality = hybrid._calculate_extraction_quality(
        np.zeros((40, 40, 3), dtype=np.uint8), mask, 300.0, 500.0, 0.8
    )
    assert quality == pytest.approx(0.95)
