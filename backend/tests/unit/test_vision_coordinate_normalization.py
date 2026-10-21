"""Regression coverage for source-resolution geometry in vision outputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.common.models import Ball
from backend.core.resolution_converter import ResolutionConverter
from backend.vision import VisionModule
from backend.vision.detection.balls import BallDetector
from backend.vision.detection.hybrid_validator import BallPositionRefiner
from backend.vision.models import BallType, CueStick, Pocket, PocketType, Table


@pytest.mark.unit
def test_classic_detector_marks_raw_ball_coordinates_with_input_resolution():
    detector = BallDetector.__new__(BallDetector)
    detector.raw_config = {}
    detector.config = SimpleNamespace(min_confidence=0.4)
    detector.classify_ball_type = lambda *_args: (BallType.CUE, 0.95, None)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    balls = detector._classify_balls([(960.0, 540.0, 18.0)], frame)

    assert len(balls) == 1
    assert balls[0].position == (960.0, 540.0)
    assert balls[0].radius == 18.0
    assert balls[0].coordinate_space == "pixel"
    assert balls[0].source_resolution == (1920, 1080)
    canonical = balls[0].to_4k_canonical()
    assert canonical.position == (1920.0, 1080.0)
    assert canonical.radius == 36.0


@pytest.mark.unit
def test_classic_overlap_merge_keeps_pixel_coordinate_provenance():
    detector = BallDetector.__new__(BallDetector)
    detector.raw_config = {"overlap_merging": {"extreme_overlap_ratio": 0.5}}
    detector.config = SimpleNamespace(max_overlap_ratio=0.3)
    balls = [
        Ball(
            position=(200.0, 100.0),
            radius=10.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            coordinate_space="pixel",
            source_resolution=(640, 360),
        ),
        Ball(
            position=(201.0, 101.0),
            radius=10.0,
            ball_type=BallType.CUE,
            confidence=0.8,
            coordinate_space="pixel",
            source_resolution=(640, 360),
        ),
    ]

    merged = detector._remove_overlaps(balls)

    assert len(merged) == 1
    assert merged[0].coordinate_space == "pixel"
    assert merged[0].source_resolution == (640, 360)


@pytest.mark.unit
def test_resized_and_full_resolution_ball_geometry_normalizes_equivalently():
    resized_ball = Ball(
        position=(480.0, 270.0),
        radius=9.0,
        ball_type=BallType.CUE,
        velocity=(2.0, 3.0),
        acceleration=(0.5, 0.25),
        predicted_position=(481.0, 272.0),
        position_history=[(470.0, 260.0)],
        coordinate_space="pixel",
        source_resolution=(960, 540),
    )
    full_resolution_ball = Ball(
        position=(1920.0, 1080.0),
        radius=36.0,
        ball_type=BallType.CUE,
        velocity=(8.0, 12.0),
        acceleration=(2.0, 1.0),
        predicted_position=(1924.0, 1088.0),
        position_history=[(1880.0, 1040.0)],
        coordinate_space="pixel",
        source_resolution=(3840, 2160),
    )

    resized = VisionModule._normalise_balls_to_source(
        [resized_ball], (1920, 1080), (960, 540)
    )[0]
    full_resolution = VisionModule._normalise_balls_to_source(
        [full_resolution_ball], (3840, 2160), (3840, 2160)
    )[0]
    resized_canonical = resized.to_4k_canonical()
    full_canonical = full_resolution.to_4k_canonical()

    assert resized_canonical.position == full_canonical.position == (1920.0, 1080.0)
    assert resized_canonical.radius == full_canonical.radius == 36.0
    assert resized_canonical.velocity == full_canonical.velocity == (8.0, 12.0)
    assert resized_canonical.acceleration == full_canonical.acceleration == (2.0, 1.0)
    assert resized_canonical.predicted_position == full_canonical.predicted_position
    assert resized_canonical.position_history == full_canonical.position_history

    # Re-normalizing an already source-sized detection does not scale it twice.
    VisionModule._normalise_balls_to_source([resized], (1920, 1080), (1920, 1080))
    assert resized.position == (960.0, 540.0)
    assert resized.radius == 18.0
    assert resized.velocity == (4.0, 6.0)


@pytest.mark.unit
def test_table_boundaries_pockets_and_extents_follow_source_resolution():
    table = Table(
        corners=[(0.0, 0.0), (960.0, 0.0), (960.0, 540.0), (0.0, 540.0)],
        pockets=[
            Pocket(
                position=(480.0, 270.0),
                pocket_type=PocketType.SIDE,
                radius=20.0,
                corners=[(460.0, 250.0), (500.0, 250.0), (480.0, 290.0)],
            )
        ],
        width=960.0,
        height=540.0,
        surface_color=(0, 0, 0),
        rails=[[(0.0, 10.0), (960.0, 10.0)]],
        playing_area=[(10.0, 10.0), (950.0, 10.0), (950.0, 530.0)],
        coordinate_space="pixel",
        source_resolution=(960, 540),
    )

    normalized = VisionModule._normalise_table_to_source(
        table, (1920, 1080), (960, 540)
    )

    assert normalized.source_resolution == (1920, 1080)
    assert normalized.coordinate_space == "pixel"
    assert normalized.corners == [
        (0.0, 0.0),
        (1920.0, 0.0),
        (1920.0, 1080.0),
        (0.0, 1080.0),
    ]
    assert (normalized.width, normalized.height) == (1920.0, 1080.0)
    assert normalized.rails == [[(0.0, 20.0), (1920.0, 20.0)]]
    assert normalized.playing_area[1] == (1900.0, 20.0)
    pocket = normalized.pockets[0]
    assert pocket.position == (960.0, 540.0)
    assert pocket.radius == 40.0
    assert pocket.corners[0] == (920.0, 500.0)

    canonical_corners = [
        ResolutionConverter.scale_to_4k(*point, normalized.source_resolution)
        for point in normalized.corners
    ]
    assert canonical_corners == [
        (0.0, 0.0),
        (3840.0, 0.0),
        (3840.0, 2160.0),
        (0.0, 2160.0),
    ]

    native_4k_table = Table(
        corners=canonical_corners,
        pockets=[
            Pocket(
                position=(1920.0, 1080.0),
                pocket_type=PocketType.SIDE,
                radius=80.0,
                corners=[(1840.0, 1000.0), (2000.0, 1000.0), (1920.0, 1160.0)],
            )
        ],
        width=3840.0,
        height=2160.0,
        surface_color=(0, 0, 0),
        coordinate_space="pixel",
        source_resolution=(3840, 2160),
    )
    canonical_1080p_pocket = ResolutionConverter.scale_to_4k(
        *pocket.position, normalized.source_resolution
    )
    canonical_4k_pocket = ResolutionConverter.scale_to_4k(
        *native_4k_table.pockets[0].position, native_4k_table.source_resolution
    )
    assert canonical_1080p_pocket == canonical_4k_pocket == (1920.0, 1080.0)
    assert (
        ResolutionConverter.scale_distance_to_4k(
            pocket.radius, normalized.source_resolution
        )
        == native_4k_table.pockets[0].radius
        == 80.0
    )


@pytest.mark.unit
def test_pixel_cue_normalization_scales_every_geometry_field_once():
    cue = CueStick(
        tip_position=(480.0, 270.0),
        butt_position=(400.0, 250.0),
        angle=15.0,
        length=100.0,
        width=3.0,
        tip_velocity=(4.0, 6.0),
        shaft_points=[(440.0, 260.0)],
        predicted_contact_point=(500.0, 280.0),
        aiming_line=[(480.0, 270.0), (500.0, 280.0)],
        coordinate_space="pixel",
        source_resolution=(960, 540),
    )

    normalized = VisionModule._normalise_cue_to_source(cue, (1920, 1080), (960, 540))

    assert normalized is cue
    assert normalized.source_resolution == (1920, 1080)
    assert normalized.tip_position == (960.0, 540.0)
    assert normalized.butt_position == (800.0, 500.0)
    assert normalized.tip_velocity == (8.0, 12.0)
    assert normalized.shaft_points == [(880.0, 520.0)]
    assert normalized.predicted_contact_point == (1000.0, 560.0)
    assert normalized.aiming_line == [(960.0, 540.0), (1000.0, 560.0)]
    assert normalized.length == 200.0
    assert normalized.width == 6.0


@pytest.mark.unit
def test_hybrid_refinement_converts_canonical_ball_to_frame_pixels(monkeypatch):
    refiner = BallPositionRefiner(max_refinement_distance=5.0)
    ball = Ball(
        position=(1920.0, 1080.0),
        radius=36.0,
        ball_type=BallType.CUE,
        confidence=0.95,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    roi = np.zeros((20, 20), dtype=np.uint8)
    observed_expected_radii = []

    monkeypatch.setattr(
        refiner, "_extract_roi", lambda _ball, _frame: (roi, (950, 533))
    )
    monkeypatch.setattr(refiner, "_preprocess_roi", lambda image: image)

    def detect_circle(_image, expected_radius):
        observed_expected_radii.append(expected_radius)
        return (10.0, 10.0), 9.0

    monkeypatch.setattr(refiner, "_detect_precise_circle", detect_circle)
    monkeypatch.setattr(refiner, "_refine_subpixel", lambda *_args: None)

    refined = refiner.refine_ball_position(ball, frame)

    assert observed_expected_radii == [18.0]
    assert refined is not ball
    assert refined.coordinate_space == "4k_canonical"
    assert refined.source_resolution == (1920, 1080)
    assert refined.position == (1920.0, 1086.0)
    assert refined.radius == 18.0
