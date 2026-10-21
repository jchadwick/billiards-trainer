"""Deterministic checks for explicit scaling in developer tools."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR.parent))
DetectorBenchmark = importlib.import_module("benchmark_detectors").DetectorBenchmark
scaling_module = importlib.import_module("coordinate_scaling")
CoordinateMetadata = scaling_module.CoordinateMetadata


def _metadata() -> CoordinateMetadata:
    return CoordinateMetadata.from_mapping(
        {
            "coordinate_space": "pixel",
            "source_resolution": [640, 360],
            "coordinate_scale": [6.0, 6.0],
        }
    )


def test_points_extents_radii_and_bounds_scale_to_actual_frame() -> None:
    metadata = _metadata()

    assert metadata.point_to_resolution(320, 180, (1920, 1080)) == (960, 540)
    assert metadata.extent_to_resolution(40, 20, (1920, 1080)) == (120, 60)
    assert metadata.radius_to_resolution(20, (1920, 1080)) == 60
    assert metadata.bounds_to_resolution(300, 100, 340, 140, (1920, 1080)) == (
        900,
        300,
        1020,
        420,
    )


def test_metadata_requires_source_resolution_and_matching_scale() -> None:
    with pytest.raises(ValueError, match="source_resolution"):
        CoordinateMetadata.from_mapping(
            {"coordinate_space": "pixel", "coordinate_scale": [6, 6]}
        )

    with pytest.raises(ValueError, match="does not match"):
        CoordinateMetadata.from_mapping(
            {
                "coordinate_space": "pixel",
                "source_resolution": [640, 360],
                "coordinate_scale": [1, 1],
            }
        )


def test_ground_truth_rejects_incomplete_or_invalid_extents() -> None:
    with pytest.raises(ValueError, match="both width and height"):
        DetectorBenchmark._validate_ground_truth_annotation(
            {"x": 1, "y": 1, "radius": 1, "width": 2}, 0
        )

    with pytest.raises(ValueError, match="positive and finite"):
        DetectorBenchmark._validate_ground_truth_annotation(
            {"x": 1, "y": 1, "radius": 1, "width": 2, "height": float("inf")}, 0
        )


def test_legacy_ground_truth_without_metadata_is_rejected(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy-ground-truth.json"
    legacy_path.write_text(json.dumps({"0": [{"x": 10, "y": 10, "radius": 5}]}))
    benchmark = DetectorBenchmark.__new__(DetectorBenchmark)
    benchmark.ground_truth_path = str(legacy_path)
    benchmark.ground_truth_metadata = None

    with pytest.raises(ValueError, match="coordinate_space"):
        benchmark._load_ground_truth()


def test_checked_in_ground_truth_converts_bounds_and_circle_to_frame() -> None:
    benchmark = DetectorBenchmark.__new__(DetectorBenchmark)
    benchmark.ground_truth_path = str(TOOLS_DIR / "example_ground_truth.json")
    benchmark.ground_truth_metadata = None
    frames = benchmark._load_ground_truth()

    assert len(frames) == 3
    annotation = benchmark._ground_truth_for_resolution(frames[0], (1920, 1080))[0]
    assert (annotation["x"], annotation["y"], annotation["radius"]) == pytest.approx(
        (961.5, 720.6, 60.3)
    )
    assert annotation["bounds"] == pytest.approx(
        {"left": 901.2, "top": 660.3, "right": 1021.8, "bottom": 780.9}
    )
    assert (annotation["width"], annotation["height"]) == pytest.approx((120.6, 120.6))


def test_opencv_sample_thresholds_are_nested_and_scaled() -> None:
    benchmark = DetectorBenchmark.__new__(DetectorBenchmark)
    benchmark.opencv_config_path = TOOLS_DIR / "example_opencv_config.json"
    benchmark._opencv_config = benchmark._load_opencv_config(
        benchmark.opencv_config_path
    )

    scaled = benchmark._config_for_resolution((1280, 720))

    assert scaled["size_constraints"]["min_radius"] == 30
    assert scaled["size_constraints"]["max_radius"] == 52
    assert scaled["size_constraints"]["expected_radius"] == 40
    assert scaled["performance"]["roi_margin"] == 100
    assert "min_radius" not in scaled

    detector_config = importlib.import_module(
        "vision.detection.balls"
    ).BallDetectionConfig.from_config(scaled, (1280, 720))
    assert detector_config.min_radius == 30
    assert detector_config.max_radius == 52
    assert detector_config.expected_radius == 40


def test_opencv_config_without_coordinate_metadata_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "ambiguous-detector.json"
    config_path.write_text(
        json.dumps({"size_constraints": {"min_radius": 10, "max_radius": 20}})
    )
    benchmark = DetectorBenchmark.__new__(DetectorBenchmark)

    with pytest.raises(ValueError, match="coordinate_space"):
        benchmark._load_opencv_config(config_path)


def test_canonical_ball_and_cue_render_at_non_4k_frame_resolution() -> None:
    debugger_module = importlib.import_module("video_debugger")
    cue_model = importlib.import_module("backend.vision.models").CueStick
    debugger = debugger_module.VideoDebugger.__new__(debugger_module.VideoDebugger)
    ball = SimpleNamespace(
        position=(1920, 1080),
        radius=40,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
        track_id=None,
    )
    cue = cue_model(
        tip_position=(1920, 1080),
        angle=0,
        length=200,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
    )
    debugger.vision_module = SimpleNamespace(
        config=SimpleNamespace(detection_backend="test"),
        _process_single_frame=lambda *_: SimpleNamespace(balls=[ball], cue=cue),
    )
    debugger.current_frame_num = 0
    debugger.current_tracked_balls = []
    debugger.current_trajectory = None
    debugger.table_state = None
    debugger._canonical_playing_area_corners = None
    debugger.show_playing_area = False
    debugger.show_trajectory = False
    debugger.show_cue = True
    debugger.show_detections = True
    debugger.show_track_ids = False
    debugger.paused = False
    debugger.playback_speed = 1.0

    frame = debugger.process_frame(np.zeros((540, 960, 3), dtype=np.uint8))

    # The cue's canonical endpoint at x=1920 is drawn at frame x=480; its
    # canonical length is likewise reduced to 50 frame pixels.
    assert frame[270, 455, 1] > 0
    # The canonical ball's center and radius are rendered at half/quarter size.
    assert tuple(frame[270, 490]) == (0, 0, 255)
    assert tuple(frame[270, 493]) == (0, 0, 0)


def test_cue_target_tolerance_is_scaled_into_canonical_space() -> None:
    debugger_module = importlib.import_module("video_debugger")
    cue_model = importlib.import_module("backend.vision.models").CueStick
    debugger = debugger_module.VideoDebugger.__new__(debugger_module.VideoDebugger)
    cue = cue_model(
        tip_position=(1000, 1000),
        angle=0,
        length=200,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
    )
    candidate = SimpleNamespace(
        position=(1100, 1060),
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
    )

    assert (
        debugger._find_ball_cue_is_pointing_at(cue, [candidate], (1920, 1080))
        is candidate
    )
    assert (
        debugger._find_ball_cue_is_pointing_at(cue, [candidate], (3840, 2160)) is None
    )


def test_canonical_cue_angle_uses_source_aspect_ratio_for_derived_endpoint() -> None:
    debugger_module = importlib.import_module("video_debugger")
    cue_model = importlib.import_module("backend.vision.models").CueStick
    cue = cue_model(
        tip_position=(1000, 1000),
        angle=90,
        length=200,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1200),
    )

    tip, butt = debugger_module.VideoDebugger._cue_endpoints_canonical(cue)

    assert tip == pytest.approx((1000, 1000))
    assert butt == pytest.approx((1000, 820))


def test_video_debugger_scales_canonical_playing_area_for_frame() -> None:
    debugger_module = importlib.import_module("video_debugger")
    vector_model = importlib.import_module("backend.core.models").Vector2D
    debugger = debugger_module.VideoDebugger.__new__(debugger_module.VideoDebugger)
    debugger._canonical_playing_area_corners = [
        vector_model.from_4k(x, y)
        for x, y in ((1920, 1080), (3000, 1080), (3000, 1800), (1920, 1800))
    ]
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    debugger.draw_playing_area(frame)

    assert tuple(frame[270, 480]) == (0, 255, 0)


def test_video_debugger_preserves_each_configured_corner_scale(monkeypatch) -> None:
    debugger_module = importlib.import_module("video_debugger")
    corners = [
        {"x": 320, "y": 180, "scale": [6, 6]},
        {"x": 500, "y": 180, "scale": [3.84, 3.84]},
        {"x": 500, "y": 300, "scale": [3.84, 3.84]},
        {"x": 320, "y": 300, "scale": [6, 6]},
    ]

    def get_config_value(path, default=None):
        return {
            "table.playing_area_corners": corners,
            "vision.calibration.camera.table_dimensions": {},
        }.get(path, default)

    monkeypatch.setitem(
        sys.modules,
        "backend.config",
        SimpleNamespace(config=SimpleNamespace(get=get_config_value)),
    )
    debugger = debugger_module.VideoDebugger.__new__(debugger_module.VideoDebugger)
    debugger._canonical_playing_area_corners = None
    debugger._table_dimensions_real = None

    table_state = debugger._load_table_state_from_config()

    assert table_state is not None
    actual_corners = [
        (corner.x, corner.y) for corner in debugger._canonical_playing_area_corners
    ]
    expected_corners = [(1920, 1080), (1920, 691.2), (1920, 1152), (1920, 1800)]
    assert len(actual_corners) == len(expected_corners)
    for actual, expected in zip(actual_corners, expected_corners):
        assert actual == pytest.approx(expected)


def test_benchmark_comparison_visualization_scales_canonical_yolo_output() -> None:
    benchmark = DetectorBenchmark.__new__(DetectorBenchmark)
    canonical_ball = SimpleNamespace(
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
        position=(1920, 1080),
        radius=40,
    )
    pixel_ball = SimpleNamespace(
        coordinate_space="pixel",
        source_resolution=(960, 540),
        position=(480, 270),
        radius=10,
    )

    comparison = benchmark._visualize_comparison(
        np.zeros((540, 960, 3), dtype=np.uint8),
        [pixel_ball],
        [canonical_ball],
        frame_number=0,
    )

    assert tuple(comparison[270, 480]) == (0, 0, 255)
    assert tuple(comparison[270, 1440]) == (0, 0, 255)
