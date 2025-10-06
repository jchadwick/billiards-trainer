"""Unit tests for YOLO detector adapter.

Tests the conversion of YOLO detection outputs to Vision module dataclasses,
including bbox conversion, radius estimation, class ID mapping, ball number
extraction, confidence handling, and edge cases.

Implements comprehensive testing for FR-VIS-020 through FR-VIS-024, FR-VIS-030, FR-VIS-031.
"""

import numpy as np
import pytest
from vision.detection.detector_adapter import (
    bbox_center_to_center_radius,
    bbox_to_center_radius,
    estimate_ball_radius_from_bbox,
    estimate_cue_angle_from_bbox,
    filter_ball_detections,
    filter_cue_detections,
    filter_detections_by_confidence,
    map_class_id_to_ball_type,
    parse_ball_class_name,
    process_yolo_detections,
    yolo_detections_to_balls,
    yolo_to_ball,
    yolo_to_cue,
)
from vision.models import Ball, BallType, CueState, CueStick

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture()
def image_shape():
    """Standard image shape for tests (height, width)."""
    return (1080, 1920)


@pytest.fixture()
def sample_ball_bbox_xyxy():
    """Sample ball bounding box in xyxy format (pixel coordinates)."""
    # Ball at center (960, 540) with radius 30
    # bbox: [x1, y1, x2, y2]
    return [930, 510, 990, 570]


@pytest.fixture()
def sample_ball_bbox_xywh():
    """Sample ball bounding box in xywh format (pixel coordinates)."""
    # Ball at center (960, 540) with radius 30
    # bbox: [x, y, w, h] where (x,y) is top-left
    return [930, 510, 60, 60]


@pytest.fixture()
def sample_ball_bbox_normalized():
    """Sample ball bounding box in normalized xywh format."""
    # Ball at center (0.5, 0.5) with radius ~30 pixels
    # bbox: [x, y, w, h] normalized to [0-1]
    return [0.484375, 0.4722, 0.03125, 0.0556]


@pytest.fixture()
def sample_cue_bbox():
    """Sample cue stick bounding box (horizontal orientation)."""
    # Cue stick: horizontal, center at (800, 400), length 200, width 20
    return [700, 390, 200, 20]


@pytest.fixture()
def sample_detection_cue():
    """Sample YOLO detection for cue ball."""
    return {
        "bbox": [930, 510, 990, 570],
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "cue",
    }


@pytest.fixture()
def sample_detection_solid():
    """Sample YOLO detection for solid ball #3."""
    return {
        "bbox": [800, 400, 860, 460],
        "confidence": 0.88,
        "class_id": 3,
        "class_name": "solid_3",
    }


@pytest.fixture()
def sample_detection_stripe():
    """Sample YOLO detection for stripe ball #12."""
    return {
        "bbox": [1100, 600, 1160, 660],
        "confidence": 0.82,
        "class_id": 12,
        "class_name": "stripe_12",
    }


@pytest.fixture()
def sample_detection_eight():
    """Sample YOLO detection for eight ball."""
    return {
        "bbox": [1000, 700, 1060, 760],
        "confidence": 0.91,
        "class_id": 8,
        "class_name": "eight",
    }


@pytest.fixture()
def class_names():
    """YOLO class names list."""
    return [
        "cue",
        "solid_1",
        "solid_2",
        "solid_3",
        "solid_4",
        "solid_5",
        "solid_6",
        "solid_7",
        "eight",
        "stripe_9",
        "stripe_10",
        "stripe_11",
        "stripe_12",
        "stripe_13",
        "stripe_14",
        "stripe_15",
        "cue_stick",
    ]


# =============================================================================
# Ball Class Name Parsing Tests
# =============================================================================


def test_parse_ball_class_name_cue():
    """Test parsing cue ball class names."""
    assert parse_ball_class_name("cue") == (BallType.CUE, None)
    assert parse_ball_class_name("CUE") == (BallType.CUE, None)
    assert parse_ball_class_name("cue_ball") == (BallType.CUE, None)
    assert parse_ball_class_name("ball_0") == (BallType.CUE, None)


def test_parse_ball_class_name_eight():
    """Test parsing eight ball class names."""
    assert parse_ball_class_name("eight") == (BallType.EIGHT, 8)
    assert parse_ball_class_name("EIGHT") == (BallType.EIGHT, 8)
    assert parse_ball_class_name("8ball") == (BallType.EIGHT, 8)
    assert parse_ball_class_name("ball_8") == (BallType.EIGHT, 8)


def test_parse_ball_class_name_solids():
    """Test parsing solid ball class names."""
    for i in range(1, 8):
        assert parse_ball_class_name(f"solid_{i}") == (BallType.SOLID, i)
        assert parse_ball_class_name(f"SOLID_{i}") == (BallType.SOLID, i)
        assert parse_ball_class_name(f"ball_{i}") == (BallType.SOLID, i)


def test_parse_ball_class_name_stripes():
    """Test parsing stripe ball class names."""
    for i in range(9, 16):
        assert parse_ball_class_name(f"stripe_{i}") == (BallType.STRIPE, i)
        assert parse_ball_class_name(f"STRIPE_{i}") == (BallType.STRIPE, i)
        assert parse_ball_class_name(f"ball_{i}") == (BallType.STRIPE, i)


def test_parse_ball_class_name_unknown():
    """Test parsing unknown class names."""
    assert parse_ball_class_name("unknown") == (BallType.UNKNOWN, None)
    assert parse_ball_class_name("ball_16") == (BallType.UNKNOWN, None)
    assert parse_ball_class_name("ball_99") == (BallType.UNKNOWN, None)
    assert parse_ball_class_name("table") == (BallType.UNKNOWN, None)


# =============================================================================
# Class ID Mapping Tests
# =============================================================================


def test_map_class_id_to_ball_type_with_names(class_names):
    """Test class ID mapping with class names provided."""
    assert map_class_id_to_ball_type(0, class_names) == (BallType.CUE, None)
    assert map_class_id_to_ball_type(3, class_names) == (BallType.SOLID, 3)
    assert map_class_id_to_ball_type(8, class_names) == (BallType.EIGHT, 8)
    assert map_class_id_to_ball_type(12, class_names) == (BallType.STRIPE, 12)


def test_map_class_id_to_ball_type_without_names():
    """Test class ID mapping without class names (fallback)."""
    assert map_class_id_to_ball_type(0, None) == (BallType.CUE, None)
    assert map_class_id_to_ball_type(3, None) == (BallType.SOLID, 3)
    assert map_class_id_to_ball_type(8, None) == (BallType.EIGHT, 8)
    assert map_class_id_to_ball_type(12, None) == (BallType.STRIPE, 12)


def test_map_class_id_to_ball_type_invalid():
    """Test class ID mapping with invalid IDs."""
    assert map_class_id_to_ball_type(99, None) == (BallType.UNKNOWN, None)
    assert map_class_id_to_ball_type(-1, None) == (BallType.UNKNOWN, None)
    assert map_class_id_to_ball_type(16, None) == (BallType.UNKNOWN, None)


# =============================================================================
# Bounding Box Conversion Tests
# =============================================================================


def test_bbox_to_center_radius_pixels(sample_ball_bbox_xywh, image_shape):
    """Test bbox to center/radius conversion with pixel coordinates."""
    center, radius = bbox_to_center_radius(
        sample_ball_bbox_xywh, image_shape, normalized=False
    )

    # Expected: center at (960, 540), radius 30
    assert center == pytest.approx((960, 540), abs=0.1)
    assert radius == pytest.approx(30, abs=0.1)


def test_bbox_to_center_radius_normalized(sample_ball_bbox_normalized, image_shape):
    """Test bbox to center/radius conversion with normalized coordinates."""
    center, radius = bbox_to_center_radius(
        sample_ball_bbox_normalized, image_shape, normalized=True
    )

    # Expected: center at (0.5, 0.5) * (1920, 1080) = (960, 540)
    # radius ~ 30 pixels
    assert center[0] == pytest.approx(960, abs=5)
    assert center[1] == pytest.approx(540, abs=5)
    assert radius == pytest.approx(30, abs=5)


def test_bbox_center_to_center_radius(image_shape):
    """Test center-format bbox conversion."""
    # Center format: [cx, cy, w, h]
    bbox_center = [960, 540, 60, 60]
    center, radius = bbox_center_to_center_radius(
        bbox_center, image_shape, normalized=False
    )

    assert center == (960, 540)
    assert radius == pytest.approx(30, abs=0.1)


def test_bbox_center_to_center_radius_normalized(image_shape):
    """Test center-format bbox conversion with normalized coordinates."""
    bbox_center = [0.5, 0.5, 0.03125, 0.0556]
    center, radius = bbox_center_to_center_radius(
        bbox_center, image_shape, normalized=True
    )

    assert center[0] == pytest.approx(960, abs=5)
    assert center[1] == pytest.approx(540, abs=5)


# =============================================================================
# Radius Estimation Tests
# =============================================================================


def test_radius_calculation_square_bbox():
    """Test radius estimation from square bounding box."""
    bbox = [0, 0, 60, 60]
    image_shape = (1080, 1920)
    radius = estimate_ball_radius_from_bbox(bbox, image_shape, normalized=False)

    # For square bbox, radius = (w + h) / 4 = (60 + 60) / 4 = 30
    assert radius == 30


def test_radius_calculation_rectangular_bbox():
    """Test radius estimation from rectangular bounding box."""
    bbox = [0, 0, 50, 70]
    image_shape = (1080, 1920)
    radius = estimate_ball_radius_from_bbox(bbox, image_shape, normalized=False)

    # For rectangular bbox, radius = (50 + 70) / 4 = 30
    assert radius == 30


def test_radius_calculation_normalized():
    """Test radius estimation with normalized coordinates."""
    bbox = [0.5, 0.5, 0.03125, 0.0556]  # ~60x60 pixels at 1920x1080
    image_shape = (1080, 1920)
    radius = estimate_ball_radius_from_bbox(bbox, image_shape, normalized=True)

    # 0.03125 * 1920 = 60, 0.0556 * 1080 = 60
    assert radius == pytest.approx(30, abs=1)


# =============================================================================
# YOLO to Ball Conversion Tests
# =============================================================================


def test_yolo_to_ball_conversion(sample_detection_cue, image_shape):
    """Test basic YOLO detection to Ball conversion."""
    ball = yolo_to_ball(sample_detection_cue, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.CUE
    assert ball.number is None
    assert ball.confidence == 0.95
    assert ball.radius == pytest.approx(30, abs=1)
    assert ball.position[0] == pytest.approx(960, abs=1)
    assert ball.position[1] == pytest.approx(540, abs=1)


def test_yolo_to_ball_solid(sample_detection_solid, image_shape):
    """Test conversion of solid ball detection."""
    ball = yolo_to_ball(sample_detection_solid, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.SOLID
    assert ball.number == 3
    assert ball.confidence == 0.88


def test_yolo_to_ball_stripe(sample_detection_stripe, image_shape):
    """Test conversion of stripe ball detection."""
    ball = yolo_to_ball(sample_detection_stripe, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.STRIPE
    assert ball.number == 12
    assert ball.confidence == 0.82


def test_yolo_to_ball_eight(sample_detection_eight, image_shape):
    """Test conversion of eight ball detection."""
    ball = yolo_to_ball(sample_detection_eight, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.EIGHT
    assert ball.number == 8
    assert ball.confidence == 0.91


def test_ball_type_mapping_via_class_name(image_shape):
    """Test ball type mapping via class_name field."""
    detection = {
        "bbox": [930, 510, 990, 570],
        "confidence": 0.9,
        "class_name": "stripe_15",
    }
    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.STRIPE
    assert ball.number == 15


def test_ball_number_extraction_from_class_id(image_shape, class_names):
    """Test ball number extraction from class ID."""
    detection = {
        "bbox": [930, 510, 990, 570],
        "confidence": 0.9,
        "class_id": 5,  # solid_5
    }
    ball = yolo_to_ball(detection, image_shape, class_names, bbox_format="xyxy")

    assert ball is not None
    assert ball.ball_type == BallType.SOLID
    assert ball.number == 5


# =============================================================================
# Confidence Handling Tests
# =============================================================================


def test_confidence_filtering_accept(sample_detection_cue, image_shape):
    """Test that detections above threshold are accepted."""
    ball = yolo_to_ball(
        sample_detection_cue, image_shape, min_confidence=0.5, bbox_format="xyxy"
    )

    assert ball is not None
    assert ball.confidence == 0.95


def test_confidence_filtering_reject(sample_detection_cue, image_shape):
    """Test that detections below threshold are rejected."""
    detection = sample_detection_cue.copy()
    detection["confidence"] = 0.15

    ball = yolo_to_ball(detection, image_shape, min_confidence=0.25, bbox_format="xyxy")

    assert ball is None


def test_confidence_filtering_boundary(sample_detection_cue, image_shape):
    """Test confidence filtering at exact threshold."""
    detection = sample_detection_cue.copy()
    detection["confidence"] = 0.25

    ball = yolo_to_ball(detection, image_shape, min_confidence=0.25, bbox_format="xyxy")

    assert ball is not None
    assert ball.confidence == 0.25


def test_filter_detections_by_confidence():
    """Test batch confidence filtering."""
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox": [10, 10, 20, 20], "confidence": 0.3},
        {"bbox": [20, 20, 30, 30], "confidence": 0.15},
        {"bbox": [30, 30, 40, 40], "conf": 0.7},  # Test "conf" key
    ]

    filtered = filter_detections_by_confidence(detections, min_confidence=0.25)

    assert len(filtered) == 3
    assert filtered[0]["confidence"] == 0.9
    assert filtered[1]["confidence"] == 0.3
    assert filtered[2]["conf"] == 0.7


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_empty_detections(image_shape):
    """Test handling of empty detection list."""
    balls = yolo_detections_to_balls([], image_shape)

    assert balls == []
    assert isinstance(balls, list)


def test_out_of_bounds_bbox(image_shape):
    """Test handling of out-of-bounds bounding box."""
    detection = {
        "bbox": [5000, 5000, 5100, 5100],  # Way outside image
        "confidence": 0.9,
        "class_id": 0,
    }

    # Should still convert, but with out-of-bounds coordinates
    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.position[0] > image_shape[1]  # Outside width


def test_missing_bbox_field(image_shape):
    """Test handling of detection missing bbox field."""
    detection = {
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is None


def test_missing_class_info(image_shape):
    """Test handling of detection missing class information."""
    detection = {
        "bbox": [930, 510, 990, 570],
        "confidence": 0.9,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is None


def test_invalid_bbox_format(image_shape):
    """Test handling of invalid bbox format."""
    detection = {
        "bbox": [930, 510],  # Only 2 values instead of 4
        "confidence": 0.9,
        "class_id": 0,
    }

    # Should handle gracefully
    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is None


def test_zero_confidence_detection(sample_detection_cue, image_shape):
    """Test handling of zero confidence detection."""
    detection = sample_detection_cue.copy()
    detection["confidence"] = 0.0

    ball = yolo_to_ball(detection, image_shape, min_confidence=0.0, bbox_format="xyxy")

    assert ball is not None
    assert ball.confidence == 0.0


def test_negative_bbox_coordinates(image_shape):
    """Test handling of negative bbox coordinates."""
    detection = {
        "bbox": [-10, -10, 50, 50],
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    # Should still convert, coordinates may be negative
    assert ball is not None


# =============================================================================
# CueStick Conversion Tests
# =============================================================================


def test_cue_conversion_basic(sample_cue_bbox, image_shape):
    """Test basic cue stick detection conversion."""
    detection = {
        "bbox": sample_cue_bbox,
        "confidence": 0.85,
    }

    cue = yolo_to_cue(detection, image_shape, bbox_format="xywh")

    assert cue is not None
    assert isinstance(cue, CueStick)
    assert cue.confidence == 0.85
    assert cue.length == 200  # Width of horizontal cue
    assert cue.state == CueState.AIMING


def test_cue_conversion_with_keypoints(image_shape):
    """Test cue conversion with keypoints."""
    detection = {
        "bbox": [700, 390, 200, 20],
        "confidence": 0.85,
        "keypoints": [(800, 400), (700, 400)],  # tip, butt
    }

    cue = yolo_to_cue(detection, image_shape, bbox_format="xywh")

    assert cue is not None
    assert cue.tip_position == (800.0, 400.0)


def test_cue_conversion_with_angle(image_shape):
    """Test cue conversion with explicit angle."""
    detection = {
        "bbox": [700, 390, 200, 20],
        "confidence": 0.85,
        "angle": 45.0,
    }

    cue = yolo_to_cue(detection, image_shape, bbox_format="xywh")

    assert cue is not None
    assert cue.angle == 45.0


def test_cue_angle_estimation_horizontal(image_shape):
    """Test cue angle estimation for horizontal orientation."""
    bbox = [700, 390, 200, 20]  # Wide and short = horizontal
    angle = estimate_cue_angle_from_bbox(bbox, image_shape)

    assert angle == 0.0


def test_cue_angle_estimation_vertical(image_shape):
    """Test cue angle estimation for vertical orientation."""
    bbox = [700, 390, 20, 200]  # Narrow and tall = vertical
    angle = estimate_cue_angle_from_bbox(bbox, image_shape)

    assert angle == 90.0


def test_cue_conversion_low_confidence(image_shape):
    """Test cue conversion with confidence below threshold."""
    detection = {
        "bbox": [700, 390, 200, 20],
        "confidence": 0.2,
    }

    cue = yolo_to_cue(detection, image_shape, min_confidence=0.3, bbox_format="xywh")

    assert cue is None


# =============================================================================
# Batch Processing Tests
# =============================================================================


def test_yolo_detections_to_balls_multiple(
    sample_detection_cue, sample_detection_solid, sample_detection_eight, image_shape
):
    """Test converting multiple detections to balls."""
    detections = [sample_detection_cue, sample_detection_solid, sample_detection_eight]

    balls = yolo_detections_to_balls(detections, image_shape, bbox_format="xyxy")

    assert len(balls) == 3
    assert balls[0].ball_type == BallType.CUE
    assert balls[1].ball_type == BallType.SOLID
    assert balls[2].ball_type == BallType.EIGHT


def test_yolo_detections_to_balls_with_filtering(image_shape):
    """Test batch conversion with confidence filtering."""
    detections = [
        {"bbox": [930, 510, 990, 570], "confidence": 0.9, "class_id": 0},
        {
            "bbox": [800, 400, 860, 460],
            "confidence": 0.15,
            "class_id": 1,
        },  # Below threshold
        {"bbox": [1000, 700, 1060, 760], "confidence": 0.85, "class_id": 8},
    ]

    balls = yolo_detections_to_balls(
        detections, image_shape, min_confidence=0.25, bbox_format="xyxy"
    )

    assert len(balls) == 2
    assert balls[0].confidence == 0.9
    assert balls[1].confidence == 0.85


# =============================================================================
# Detection Filtering Tests
# =============================================================================


def test_filter_ball_detections(class_names):
    """Test filtering to keep only ball detections."""
    detections = [
        {"class_id": 0, "class_name": "cue"},
        {"class_id": 3, "class_name": "solid_3"},
        {"class_id": 16, "class_name": "cue_stick"},  # Should be filtered out
        {"class_id": 8, "class_name": "eight"},
    ]

    filtered = filter_ball_detections(detections, class_names)

    assert len(filtered) == 3
    assert all(d["class_name"] != "cue_stick" for d in filtered)


def test_filter_cue_detections(class_names):
    """Test filtering to keep only cue stick detections."""
    detections = [
        {"class_id": 0, "class_name": "cue"},  # Cue ball, not cue stick
        {"class_id": 3, "class_name": "solid_3"},
        {"class_id": 16, "class_name": "cue_stick"},
    ]

    filtered = filter_cue_detections(detections, class_names)

    assert len(filtered) == 1
    assert filtered[0]["class_name"] == "cue_stick"


def test_filter_detections_by_class_id_only():
    """Test filtering when only class_id is available."""
    ball_detections = [
        {"class_id": 0},  # Cue ball
        {"class_id": 5},  # Solid
        {"class_id": 12},  # Stripe
    ]

    filtered = filter_ball_detections(ball_detections)

    assert len(filtered) == 3


# =============================================================================
# Batch Processing Integration Tests
# =============================================================================


def test_process_yolo_detections_complete(image_shape, class_names):
    """Test complete processing pipeline with balls and cue."""
    detections = [
        {"bbox": [930, 510, 990, 570], "confidence": 0.9, "class_id": 0},  # Cue ball
        {"bbox": [800, 400, 860, 460], "confidence": 0.85, "class_id": 3},  # Solid
        {"bbox": [1000, 700, 1060, 760], "confidence": 0.88, "class_id": 8},  # Eight
        {"bbox": [700, 390, 900, 410], "confidence": 0.75, "class_id": 16},  # Cue stick
    ]

    result = process_yolo_detections(
        detections, image_shape, class_names, bbox_format="xyxy"
    )

    assert result["ball_count"] == 3
    assert result["cue_count"] == 1
    assert len(result["balls"]) == 3
    assert len(result["cues"]) == 1


def test_process_yolo_detections_empty(image_shape):
    """Test processing empty detection list."""
    result = process_yolo_detections([], image_shape)

    assert result["ball_count"] == 0
    assert result["cue_count"] == 0
    assert result["balls"] == []
    assert result["cues"] == []


def test_process_yolo_detections_confidence_thresholds(image_shape):
    """Test processing with different confidence thresholds for balls and cue."""
    detections = [
        {"bbox": [930, 510, 990, 570], "confidence": 0.3, "class_id": 0},  # Ball
        {"bbox": [700, 390, 900, 410], "confidence": 0.25, "class_id": 16},  # Cue
    ]

    result = process_yolo_detections(
        detections,
        image_shape,
        min_ball_confidence=0.25,
        min_cue_confidence=0.3,
        bbox_format="xyxy",
    )

    assert result["ball_count"] == 1  # Ball passes threshold
    assert result["cue_count"] == 0  # Cue below threshold


# =============================================================================
# Bbox Format Tests
# =============================================================================


def test_bbox_format_xyxy(image_shape):
    """Test handling of xyxy bbox format."""
    detection = {
        "bbox": [930, 510, 990, 570],  # x1, y1, x2, y2
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=1)
    assert ball.radius == pytest.approx(30, abs=1)


def test_bbox_format_xywh(image_shape):
    """Test handling of xywh bbox format."""
    detection = {
        "bbox": [930, 510, 60, 60],  # x, y, w, h
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xywh")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=1)
    assert ball.radius == pytest.approx(30, abs=1)


def test_bbox_format_cxcywh(image_shape):
    """Test handling of center-xywh bbox format."""
    detection = {
        "bbox": [960, 540, 60, 60],  # cx, cy, w, h
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="cxcywh")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=1)
    assert ball.position[1] == pytest.approx(540, abs=1)


def test_bbox_format_normalized_xyxy(image_shape):
    """Test handling of normalized xyxy bbox format."""
    detection = {
        "bbox": [0.484375, 0.4722, 0.515625, 0.5278],  # Normalized xyxy
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="normalized_xyxy")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=5)
    assert ball.position[1] == pytest.approx(540, abs=5)


# =============================================================================
# Data Integrity Tests
# =============================================================================


def test_ball_dataclass_fields(sample_detection_cue, image_shape):
    """Test that Ball dataclass has all required fields."""
    ball = yolo_to_ball(sample_detection_cue, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert hasattr(ball, "position")
    assert hasattr(ball, "radius")
    assert hasattr(ball, "ball_type")
    assert hasattr(ball, "number")
    assert hasattr(ball, "confidence")
    assert hasattr(ball, "velocity")
    assert hasattr(ball, "acceleration")
    assert hasattr(ball, "is_moving")


def test_ball_default_values(sample_detection_cue, image_shape):
    """Test that Ball has correct default values."""
    ball = yolo_to_ball(sample_detection_cue, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.velocity == (0.0, 0.0)
    assert ball.acceleration == (0.0, 0.0)
    assert ball.is_moving is False


def test_cue_dataclass_fields(sample_cue_bbox, image_shape):
    """Test that CueStick dataclass has all required fields."""
    detection = {"bbox": sample_cue_bbox, "confidence": 0.85}
    cue = yolo_to_cue(detection, image_shape, bbox_format="xywh")

    assert cue is not None
    assert hasattr(cue, "tip_position")
    assert hasattr(cue, "angle")
    assert hasattr(cue, "length")
    assert hasattr(cue, "confidence")
    assert hasattr(cue, "state")
    assert hasattr(cue, "is_aiming")


def test_cue_default_values(sample_cue_bbox, image_shape):
    """Test that CueStick has correct default values."""
    detection = {"bbox": sample_cue_bbox, "confidence": 0.85}
    cue = yolo_to_cue(detection, image_shape, bbox_format="xywh")

    assert cue is not None
    assert cue.state == CueState.AIMING
    assert cue.is_aiming is True
    assert cue.tip_velocity == (0.0, 0.0)
    assert cue.angular_velocity == 0.0


# =============================================================================
# Numpy Array Compatibility Tests
# =============================================================================


def test_bbox_as_numpy_array(image_shape):
    """Test that bbox can be provided as numpy array."""
    detection = {
        "bbox": np.array([930, 510, 990, 570], dtype=np.float32),
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=1)


def test_bbox_as_list(image_shape):
    """Test that bbox can be provided as list."""
    detection = {
        "bbox": [930, 510, 990, 570],
        "confidence": 0.9,
        "class_id": 0,
    }

    ball = yolo_to_ball(detection, image_shape, bbox_format="xyxy")

    assert ball is not None
    assert ball.position[0] == pytest.approx(960, abs=1)
