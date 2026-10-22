"""Deterministic contracts for adapting detector output to vision objects.

These tests use real adapter functions and the production ``Detection`` shape,
but synthetic coordinates. They do not claim detector accuracy or require a
camera, model weights, or accelerator runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from backend.config import config
from backend.vision.detection.detector_adapter import (
    bbox_center_to_center_radius,
    bbox_to_center_radius,
    canonical_point_to_source_pixels,
    estimate_ball_radius_from_bbox,
    estimate_cue_angle_from_bbox,
    filter_ball_detections,
    filter_cue_detections,
    filter_detections_by_confidence,
    map_class_id_to_ball_type,
    parse_ball_class_name,
    process_yolo_detections,
    yolo_cue_to_cue_stick,
    yolo_detections_to_balls,
    yolo_to_ball,
    yolo_to_cue,
)
from backend.vision.detection.yolo_detector import Detection
from backend.vision.models import BallType, CueState


def _scaled_point(x: float, y: float) -> dict[str, Any]:
    """Make a configured point in 4K canonical coordinates."""
    return {"x": x, "y": y, "scale": [1.0, 1.0]}


def _table_geometry() -> list[dict[str, Any]]:
    return [
        _scaled_point(1200, 800),
        _scaled_point(3200, 800),
        _scaled_point(3200, 1360),
        _scaled_point(1200, 1360),
    ]


def _raw_cue(**overrides: Any) -> Detection:
    values: dict[str, Any] = {
        "class_id": 16,
        "class_name": "cue_stick",
        "confidence": 0.91,
        "bbox": (100.0, 45.0, 700.0, 55.0),
        "center": (400.0, 50.0),
        "width": 600.0,
        "height": 10.0,
        "angle": 0.0,
        "line_end1": (100.0, 50.0),
        "line_end2": (700.0, 50.0),
        "line_center": (400.0, 50.0),
    }
    values.update(overrides)
    return Detection(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected_type", "expected_number"),
    [
        ("ball", BallType.UNKNOWN, None),
        (" Cue_Ball ", BallType.CUE, None),
        ("eight", BallType.EIGHT, 8),
        ("8ball", BallType.EIGHT, 8),
        ("ball_8", BallType.EIGHT, 8),
        ("solid_1", BallType.UNKNOWN, None),
        ("stripe_15", BallType.UNKNOWN, None),
        ("ball_0", BallType.CUE, None),
        ("ball_14", BallType.UNKNOWN, None),
        ("ball_16", BallType.UNKNOWN, None),
        ("solid_bad", BallType.UNKNOWN, None),
        ("stripe_2", BallType.UNKNOWN, None),
        ("object_9", BallType.UNKNOWN, None),
        ("unknown", BallType.UNKNOWN, None),
    ],
)
def test_class_names_map_to_application_ball_identity(
    name: str, expected_type: BallType, expected_number: int | None
) -> None:
    assert parse_ball_class_name(name) == (expected_type, expected_number)


@pytest.mark.unit
def test_class_id_mapping_uses_names_when_valid_and_safe_fallbacks_otherwise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert map_class_id_to_ball_type(1, ["cue", "eight"]) == (BallType.EIGHT, 8)
    assert map_class_id_to_ball_type(0, []) == (BallType.CUE, None)
    assert map_class_id_to_ball_type(8) == (BallType.EIGHT, 8)
    assert map_class_id_to_ball_type(15) == (BallType.UNKNOWN, None)
    assert map_class_id_to_ball_type(-1) == (BallType.UNKNOWN, None)
    assert "Unknown class ID" in caplog.text


@pytest.mark.unit
def test_bbox_coordinate_helpers_cover_pixel_normalized_and_center_contracts() -> None:
    assert bbox_to_center_radius([10, 20, 30, 40], (100, 200)) == (
        (25.0, 40.0),
        17.5,
    )
    assert bbox_to_center_radius([0.1, 0.2, 0.2, 0.4], (100, 200), True) == (
        (40.0, 40.0),
        20.0,
    )
    assert bbox_center_to_center_radius([25, 40, 30, 40], (100, 200)) == (
        (25, 40),
        17.5,
    )
    assert bbox_center_to_center_radius([0.2, 0.4, 0.2, 0.4], (100, 200), True) == (
        (40.0, 40.0),
        20.0,
    )
    assert estimate_ball_radius_from_bbox([0, 0, 30, 40], (100, 200)) == 17.5
    assert estimate_ball_radius_from_bbox([0, 0, 0.2, 0.4], (100, 200), True) == 20
    assert estimate_cue_angle_from_bbox([0, 0, 30, 10], (100, 200)) == 0
    assert estimate_cue_angle_from_bbox([0, 0, 10, 30], (100, 200)) == 90
    assert estimate_cue_angle_from_bbox([0, 0, 0.3, 0.1], (100, 200), True) == 0


@pytest.mark.unit
def test_canonical_ball_position_round_trips_through_source_resolution() -> None:
    assert canonical_point_to_source_pixels((1920, 1080), (1920, 1080)) == (
        960.0,
        540.0,
    )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        canonical_point_to_source_pixels((10, 10), (0, 1080))


@pytest.mark.unit
def test_ball_adaptation_preserves_identity_confidence_and_coordinate_provenance() -> (
    None
):
    ball = yolo_to_ball(
        {
            "bbox": [942, 522, 978, 558],
            "confidence": 0.93,
            "class_id": 8,
        },
        (1080, 1920),
    )

    assert ball is not None
    assert ball.position == pytest.approx((1920, 1080))
    assert ball.radius == pytest.approx(36)
    assert ball.ball_type is BallType.EIGHT
    assert ball.number == 8
    assert ball.confidence == pytest.approx(0.93)
    assert ball.velocity == (0.0, 0.0)
    assert ball.acceleration == (0.0, 0.0)
    assert ball.is_moving is False
    assert ball.coordinate_space == "4k_canonical"
    assert ball.source_resolution == (1920, 1080)


@pytest.mark.unit
def test_ball_adaptation_accepts_supported_bbox_and_class_aliases() -> None:
    xywh = yolo_to_ball(
        {"box": [0.25, 0.25, 0.1, 0.1], "conf": 0.8, "class_name": "cue"},
        (100, 200),
        bbox_format="normalized_xywh",
    )
    center = yolo_to_ball(
        {"bbox": [100, 50, 20, 10], "conf": 0.7, "class": 1},
        (100, 200),
        class_names=["cue", "solid_2"],
        bbox_format="cxcywh",
    )

    assert xywh is not None
    assert xywh.position == pytest.approx((1152, 648))
    assert xywh.radius == pytest.approx(144)
    assert xywh.ball_type is BallType.CUE
    assert center is not None
    assert center.position == pytest.approx((1920, 1080))
    assert center.radius == pytest.approx(144)
    assert center.ball_type is BallType.UNKNOWN
    assert center.number is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "detection",
    [
        {"bbox": [0, 0, 10, 10], "confidence": 0.1, "class_id": 0},
        {"confidence": 0.9, "class_id": 0},
        {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox": [0, 0, 10], "confidence": 0.9, "class_id": 0},
        {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": "not-an-id"},
    ],
)
def test_invalid_ball_candidates_are_dropped_without_poisoning_batch(
    detection: dict[str, Any],
) -> None:
    assert yolo_to_ball(detection, (100, 200)) is None


@pytest.mark.unit
def test_ball_batch_conversion_keeps_valid_current_observations_in_input_order() -> (
    None
):
    detections = [
        {"bbox": [10, 10, 20, 20], "confidence": 0.9, "class_id": 0},
        {"bbox": [20, 20, 30, 30], "confidence": 0.1, "class_id": 8},
        {"bbox": [30, 30, 40, 40], "confidence": 0.8, "class_id": 8},
    ]
    balls = yolo_detections_to_balls(detections, (100, 200))

    assert len(balls) == 2
    assert [ball.ball_type for ball in balls] == [BallType.CUE, BallType.EIGHT]
    assert [ball.source_resolution for ball in balls] == [(200, 100)] * 2


@pytest.mark.unit
def test_cue_dict_adapter_handles_keypoints_angle_estimation_and_provenance() -> None:
    explicit = yolo_to_cue(
        {
            "bbox": [10, 20, 90, 40],
            "confidence": 0.8,
            "keypoints": [[25, 30]],
            "angle": 37,
        },
        (100, 200),
    )
    estimated = yolo_to_cue({"box": [10, 20, 90, 40], "conf": 0.7}, (100, 200))
    vertical = yolo_to_cue(
        {"bbox": [10, 20, 30, 80], "conf": 0.7},
        (100, 200),
        bbox_format="xywh",
    )

    assert explicit is not None
    assert explicit.tip_position == pytest.approx((480, 648))
    assert explicit.angle == 37
    assert explicit.length == pytest.approx(1536)
    assert explicit.confidence == pytest.approx(0.8)
    assert explicit.state is CueState.AIMING
    assert explicit.coordinate_space == "4k_canonical"
    assert explicit.source_resolution == (200, 100)
    assert estimated is not None
    assert estimated.tip_position == pytest.approx((960, 648))
    assert estimated.angle == 0
    assert vertical is not None
    assert vertical.angle == 90


@pytest.mark.unit
@pytest.mark.parametrize(
    "detection",
    [
        {"bbox": [0, 0, 10, 10], "confidence": 0.1},
        {"confidence": 0.9},
        {"bbox": [0, 0, 10], "confidence": 0.9},
    ],
)
def test_invalid_cue_candidates_are_dropped(detection: dict[str, Any]) -> None:
    assert yolo_to_cue(detection, (100, 200)) is None


@pytest.mark.unit
def test_detector_cue_adapter_uses_table_center_and_canonicalizes_real_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config,
        "get",
        lambda key, default=None: (
            _table_geometry() if key == "table.playing_area_corners" else default
        ),
    )
    cue = yolo_cue_to_cue_stick(_raw_cue(), (100, 800))

    assert cue is not None
    assert cue.tip_position == pytest.approx((3360, 1080))
    assert cue.butt_position == pytest.approx((480, 1080))
    assert cue.angle == pytest.approx(0)
    assert cue.length == pytest.approx(2880)
    assert cue.confidence == pytest.approx(0.91)
    assert cue.state is CueState.AIMING
    assert cue.coordinate_space == "4k_canonical"
    assert cue.source_resolution == (800, 100)


@pytest.mark.unit
def test_detector_cue_adapter_fallback_uses_detector_direction_without_geometry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config, "get", lambda _key, default=None: default)
    cue = yolo_cue_to_cue_stick(_raw_cue(angle=25), (100, 800))

    assert cue is not None
    assert cue.tip_position == pytest.approx((3360, 1080))
    assert cue.butt_position == pytest.approx((480, 1080))
    assert cue.angle == pytest.approx(25)
    assert "Playing area is not configured" in caplog.text


@pytest.mark.unit
def test_detector_cue_adapter_estimates_endpoints_when_line_fit_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "get", lambda _key, default=None: default)
    detection = _raw_cue(
        line_end1=None,
        line_end2=None,
        line_center=None,
        center=(400.0, 50.0),
        angle=0.0,
    )

    cue = yolo_cue_to_cue_stick(detection, (100, 800))

    assert cue is not None
    assert cue.tip_position == pytest.approx((3360, 1080))
    assert cue.butt_position == pytest.approx((480, 1080))
    assert cue.length == pytest.approx(2880)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("confidence", "frame_shape", "endpoints"),
    [
        (float("nan"), (100, 800), ((100, 50), (700, 50))),
        (0.9, (0, 800), ((100, 50), (700, 50))),
        (0.9, (100, 800), ((100, 50), (100, 50))),
        (0.9, (100, 800), ((float("nan"), 50), (700, 50))),
    ],
)
def test_detector_cue_adapter_rejects_invalid_confidence_resolution_or_geometry(
    monkeypatch: pytest.MonkeyPatch,
    confidence: float,
    frame_shape: tuple[int, int],
    endpoints: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    monkeypatch.setattr(config, "get", lambda _key, default=None: default)
    detection = _raw_cue(
        confidence=confidence,
        line_end1=endpoints[0],
        line_end2=endpoints[1],
    )

    assert yolo_cue_to_cue_stick(detection, frame_shape) is None


@pytest.mark.unit
def test_confidence_and_type_filters_keep_only_eligible_detection_records() -> None:
    detections = [
        {"confidence": 0.5, "class_name": "cue"},
        {"conf": 0.8, "class_name": "cue_stick"},
        {"confidence": 0.2, "class_id": 7},
        {"confidence": 0.9, "class_id": 17},
        {"confidence": 0.6, "class_name": "ball_8"},
        {"confidence": 0.7, "class_name": "cue_ball"},
    ]

    assert filter_detections_by_confidence(detections, 0.5) == [
        detections[0],
        detections[1],
        detections[3],
        detections[4],
        detections[5],
    ]
    assert filter_ball_detections(detections) == [
        detections[0],
        detections[2],
        detections[4],
        detections[5],
    ]
    assert filter_cue_detections(detections) == [detections[1]]


@pytest.mark.unit
def test_type_filters_use_class_names_and_safe_id_fallbacks() -> None:
    detections = [
        {"class_id": 0},
        {"class_id": 1},
        {"class_id": 2},
        {"class_id": -1},
        {"class_name": "Cue_Stick"},
        {"class_name": "cue_ball"},
        {"class_name": "stripe_10"},
        {"class_name": "stick_ball"},
    ]
    names = ["cue", "ball_2", "cue_stick"]

    assert filter_ball_detections(detections[:4], names) == detections[:2]
    assert filter_cue_detections(detections[:3], names) == [detections[2]]
    assert filter_ball_detections(detections[4:]) == [detections[5], detections[6]]
    assert filter_cue_detections(detections[4:]) == [detections[4], detections[7]]


@pytest.mark.unit
def test_process_yolo_detections_returns_consistent_counts_and_discards_low_scores() -> (
    None
):
    result = process_yolo_detections(
        [
            {"bbox": [10, 10, 20, 20], "confidence": 0.9, "class_name": "cue"},
            {
                "bbox": [20, 20, 40, 30],
                "confidence": 0.8,
                "class_name": "cue_stick",
            },
            {"bbox": [30, 30, 40, 40], "confidence": 0.1, "class_name": "ball"},
        ],
        (100, 200),
    )

    assert result["ball_count"] == len(result["balls"]) == 1
    assert result["cue_count"] == len(result["cues"]) == 1
    assert result["balls"][0].ball_type is BallType.CUE
    assert result["cues"][0].state is CueState.AIMING


@pytest.mark.unit
def test_process_yolo_detections_with_class_name_table_matches_detector_labels() -> (
    None
):
    result = process_yolo_detections(
        [
            {"bbox": [1, 1, 10, 10], "confidence": 0.8, "class_id": 8},
            {"bbox": [10, 10, 30, 30], "confidence": 0.8, "class_id": 16},
        ],
        (100, 200),
        class_names=[
            "cue",
            "ball_1",
            "ball_2",
            "ball_3",
            "ball_4",
            "ball_5",
            "ball_6",
            "ball_7",
            "eight",
            "ball_9",
            "ball_10",
            "ball_11",
            "ball_12",
            "ball_13",
            "ball_14",
            "ball_15",
            "cue_stick",
        ],
    )

    assert result["ball_count"] == 1
    assert result["balls"][0].ball_type is BallType.EIGHT
    assert result["balls"][0].number == 8
    assert result["cue_count"] == 1
