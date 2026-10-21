"""Configured table geometry uses each point's explicit canonical scale."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.config import config
from backend.vision import VisionModule
from backend.vision.coordinates import (
    point_is_within_scaled_radius,
    point_to_frame,
    points_to_frame,
    radius_to_frame_ellipse,
)
from backend.vision.detection.detector_adapter import yolo_cue_to_cue_stick


def _scaled_point(x: float, y: float, scale=(10.0, 10.0)) -> dict:
    return {"x": x, "y": y, "scale": list(scale)}


def _geometry_config() -> dict:
    return {
        "table.calibration_resolution_width": 640,
        "table.calibration_resolution_height": 360,
        "table.playing_area_corners": [
            _scaled_point(100, 100),
            _scaled_point(200, 100),
            _scaled_point(200, 200),
            _scaled_point(100, 200),
        ],
        "table.marker_dots": [_scaled_point(150, 150)],
        "vision.detection.marker_filtering": {
            "enabled": True,
            "mask_radius_px": 10,
        },
    }


def _install_config(monkeypatch, values: dict) -> None:
    monkeypatch.setattr(
        config, "get", lambda key, default=None: values.get(key, default)
    )


@pytest.mark.unit
def test_ground_truth_fixtures_declare_canonical_scale_matching_image_size():
    fixture_directory = Path("backend/vision/test_data")
    for annotation_path in fixture_directory.glob("*.json"):
        payload = json.loads(annotation_path.read_text())
        assert payload["coordinate_space"] == "4k_canonical"
        assert payload["coordinate_scale"] == [1.0, 1.0]
        with Image.open(fixture_directory / payload["image"]) as image:
            assert image.size == (3840, 2160)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("size", "expected"),
    [((1920, 1080), (750.0, 750.0)), ((3840, 2160), (1500.0, 1500.0))],
)
def test_point_and_bounds_follow_explicit_scale_at_multiple_resolutions(size, expected):
    point = _scaled_point(150, 150)
    assert point_to_frame(point, size) == expected
    corners = points_to_frame(
        [
            _scaled_point(100, 100),
            _scaled_point(200, 100),
            _scaled_point(200, 200),
            _scaled_point(100, 200),
        ],
        size,
    )
    frame_factor = size[0] / 3840
    assert corners == [
        (1000.0 * frame_factor, 1000.0 * frame_factor),
        (2000.0 * frame_factor, 1000.0 * frame_factor),
        (2000.0 * frame_factor, 2000.0 * frame_factor),
        (1000.0 * frame_factor, 2000.0 * frame_factor),
    ]


@pytest.mark.unit
def test_marker_radius_scales_in_declared_space_and_handles_anisotropic_frame():
    frame_size = (1920, 1200)
    scale = (10.0, 20.0)
    radius_x, radius_y = radius_to_frame_ellipse(10, scale, frame_size)
    assert radius_x == 50.0
    assert radius_y == pytest.approx(10 * 20 * 1200 / 2160)
    center = point_to_frame(_scaled_point(150, 150, scale), frame_size)
    assert point_is_within_scaled_radius(
        center[0] + 45, center[1], center, 10, scale, frame_size
    )
    assert point_is_within_scaled_radius(
        center[0], center[1] + 100, center, 10, scale, frame_size
    )
    assert not point_is_within_scaled_radius(
        center[0] + 55, center[1], center, 10, scale, frame_size
    )
    assert not point_is_within_scaled_radius(
        center[0], center[1] + radius_y + 1, center, 10, scale, frame_size
    )


@pytest.mark.unit
def test_invalid_region_uses_scaled_marker_and_boundary_not_calibration_ratio(
    monkeypatch,
):
    _install_config(monkeypatch, _geometry_config())
    module = VisionModule.__new__(VisionModule)

    # The explicit [10, 10] scale maps this point to (750, 750) in this frame.
    # A calibration-resolution-derived ratio would put it at (450, 450).
    assert module._is_ball_in_invalid_region(780, 750, 1920, 1080)
    assert not module._is_ball_in_invalid_region(900, 750, 1920, 1080)
    # At x=450 the ball is outside the actual scaled playing area, even though
    # the old calibration-ratio corners would have included it.
    assert module._is_ball_in_invalid_region(450, 750, 1920, 1080)


@pytest.mark.unit
def test_phase_bounds_use_explicit_point_scales(monkeypatch):
    _install_config(monkeypatch, _geometry_config())

    polygon, revision = VisionModule._phase_table_geometry((1080, 1920, 3))

    assert polygon == [
        (500.0, 500.0),
        (1000.0, 500.0),
        (1000.0, 1000.0),
        (500.0, 1000.0),
    ]
    assert "10.0" in revision


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frame_size", "center", "radius"),
    [
        ((1920, 1080), (750, 750), (50, 50)),
        ((3840, 2160), (1500, 1500), (100, 100)),
        ((1920, 1200), (750, 833), (50, 83)),
    ],
)
def test_mask_painting_scales_point_and_radius_with_frame(
    monkeypatch, frame_size, center, radius
):
    values = _geometry_config()
    values["vision.detection.marker_filtering"]["mask_radius_px"] = 10
    _install_config(monkeypatch, values)
    module = VisionModule.__new__(VisionModule)
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    masked = module._apply_all_masks(frame)

    assert tuple(masked[center[1], center[0]]) == (0, 128, 0)
    assert tuple(masked[center[1], center[0] + radius[0] - 1]) == (0, 128, 0)
    assert tuple(masked[center[1], center[0] + radius[0] + 2]) == (0, 0, 0)


@pytest.mark.unit
def test_cue_center_uses_scaled_corners_and_invalid_geometry_keeps_detected_angle(
    monkeypatch,
):
    corners = [
        _scaled_point(80, 100, (10, 5)),
        _scaled_point(120, 100, (10, 5)),
        _scaled_point(120, 300, (10, 5)),
        _scaled_point(80, 300, (10, 5)),
    ]
    values = _geometry_config()
    values["table.playing_area_corners"] = corners
    _install_config(monkeypatch, values)
    detection = SimpleNamespace(
        confidence=0.9,
        line_center=(50.0, 50.0),
        center=(50.0, 50.0),
        line_end1=(20.0, 50.0),
        line_end2=(80.0, 50.0),
        bbox=(20.0, 45.0, 80.0, 55.0),
        angle=37.0,
    )

    cue = yolo_cue_to_cue_stick(detection, (100, 100))
    assert cue is not None
    assert cue.angle == pytest.approx(180.0)

    values["table.playing_area_corners"] = [
        {"x": 80, "y": 100},
        _scaled_point(120, 100, (10, 5)),
        _scaled_point(120, 300, (10, 5)),
        _scaled_point(80, 300, (10, 5)),
    ]
    _install_config(monkeypatch, values)
    invalid_geometry_cue = yolo_cue_to_cue_stick(detection, (100, 100))
    assert invalid_geometry_cue is not None
    assert invalid_geometry_cue.angle == pytest.approx(37.0)
