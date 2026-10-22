"""Behavioral contracts for current-frame multi-ball tracking.

The elapsed-time suite covers stale-frame rejection and basic expiry. These tests
focus on association decisions and how the tracker presents the latest accepted
observation when identity, classification, and track lifecycle get complicated.
"""

import math

import pytest

from backend.common.models import Ball
from backend.vision.models import BallType
from backend.vision.tracking.tracker import ObjectTracker, TrackState


def _ball(
    x: float,
    *,
    y: float = 100.0,
    ball_type: BallType = BallType.UNKNOWN,
    number: int | None = None,
    radius: float = 15.0,
    confidence: float = 0.95,
    velocity: tuple[float, float] = (0.0, 0.0),
    coordinate_space: str = "4k_canonical",
    source_resolution: tuple[int, int] | None = None,
) -> Ball:
    return Ball(
        position=(x, y),
        radius=radius,
        ball_type=ball_type,
        number=number,
        confidence=confidence,
        velocity=velocity,
        coordinate_space=coordinate_space,
        source_resolution=source_resolution,
    )


def _tracker(**overrides: object) -> ObjectTracker:
    config: dict[str, object] = {
        "min_hits": 1,
        "max_age_seconds": 1.0,
        "lost_after_seconds": 0.3,
        "max_observation_gap_seconds": 0.8,
        "max_prediction_horizon_seconds": 0.25,
        "max_distance": 100.0,
        "return_tentative_tracks": True,
        "thresholds": {"min_confirmation_confidence": 0.3},
    }
    config.update(overrides)
    return ObjectTracker(config)


def test_assignment_preserves_two_ball_ids_when_detection_order_reverses():
    tracker = _tracker()
    first = _ball(100.0, ball_type=BallType.SOLID_1)
    second = _ball(500.0, ball_type=BallType.STRIPE_9)
    tracker.update_tracking([first, second], frame_number=1, timestamp=1.0)
    ids_by_number = {track.ball_number: track.track_id for track in tracker.tracks}

    # Detector ordering is not a stable identity signal. The Hungarian assignment
    # must use positions and preserve each ball's established track ID.
    output = tracker.update_tracking(
        [
            _ball(480.0, ball_type=BallType.STRIPE_9),
            _ball(120.0, ball_type=BallType.SOLID_1),
        ],
        frame_number=2,
        timestamp=1.1,
    )

    assert len(tracker.tracks) == 2
    assert {
        track.ball_number: track.track_id for track in tracker.tracks
    } == ids_by_number
    assert {ball.number: ball.track_id for ball in output} == ids_by_number


def test_unknown_detection_updates_position_without_erasing_classification():
    tracker = _tracker()
    tracker.update_tracking(
        [_ball(250.0, ball_type=BallType.EIGHT)], frame_number=1, timestamp=2.0
    )

    output = tracker.update_tracking(
        [_ball(253.0, ball_type=BallType.UNKNOWN)], frame_number=2, timestamp=2.1
    )

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].ball_type is BallType.EIGHT
    assert tracker.tracks[0].ball_number == 8
    assert output[0].position[0] > 250.0
    assert output[0].number == 8


@pytest.mark.parametrize(
    ("track_type", "detection_type"),
    [
        (BallType.CUE, BallType.SOLID_7),
        (BallType.SOLID_7, BallType.CUE),
    ],
)
def test_cue_and_known_object_ball_types_never_steal_each_others_tracks(
    track_type: BallType, detection_type: BallType
):
    tracker = _tracker()
    tracker.update_tracking(
        [_ball(200.0, ball_type=track_type)], frame_number=1, timestamp=2.5
    )
    original_id = tracker.tracks[0].track_id

    tracker.update_tracking(
        [_ball(201.0, ball_type=detection_type)], frame_number=2, timestamp=2.6
    )

    assert len(tracker.tracks) == 2
    original = next(track for track in tracker.tracks if track.track_id == original_id)
    replacement = next(
        track for track in tracker.tracks if track.track_id != original_id
    )
    assert original.ball_type is track_type
    assert original.miss_count == 1
    assert replacement.ball_type is detection_type


@pytest.mark.parametrize(
    ("track_type", "track_number", "detection_type", "detection_number"),
    [
        (BallType.SOLID_1, 1, BallType.STRIPE_9, 9),
        (BallType.UNKNOWN, 1, BallType.UNKNOWN, 2),
    ],
)
def test_incompatible_classification_cannot_steal_an_existing_identity(
    track_type: BallType,
    track_number: int,
    detection_type: BallType,
    detection_number: int,
):
    tracker = _tracker()
    tracker.update_tracking(
        [_ball(200.0, ball_type=track_type, number=track_number)],
        frame_number=1,
        timestamp=3.0,
    )
    original_id = tracker.tracks[0].track_id

    tracker.update_tracking(
        [_ball(201.0, ball_type=detection_type, number=detection_number)],
        frame_number=2,
        timestamp=3.1,
    )

    assert len(tracker.tracks) == 2
    original = next(track for track in tracker.tracks if track.track_id == original_id)
    replacement = next(
        track for track in tracker.tracks if track.track_id != original_id
    )
    assert original.ball_type is track_type
    assert original.miss_count == 1
    assert replacement.ball_type is detection_type


@pytest.mark.parametrize("radius", [7.0, 31.0])
def test_size_incompatible_detection_starts_a_new_track(radius: float):
    tracker = _tracker(size_compatibility={"min_ratio": 0.5, "max_ratio": 2.0})
    tracker.update_tracking([_ball(300.0, radius=15.0)], 1, 4.0)
    original_id = tracker.tracks[0].track_id

    tracker.update_tracking([_ball(301.0, radius=radius)], 2, 4.1)

    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].track_id == original_id
    assert tracker.tracks[0].miss_count == 1
    assert tracker.tracks[1].track_id != original_id
    assert tracker.track_statistics["total_tracks_created"] == 2


def test_far_detection_does_not_teleport_track_and_unmatched_ball_is_visible():
    tracker = _tracker(max_distance=40.0)
    tracker.update_tracking([_ball(100.0)], frame_number=1, timestamp=5.0)
    original_id = tracker.tracks[0].track_id

    current = tracker.update_tracking([_ball(400.0)], frame_number=2, timestamp=5.1)

    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].track_id == original_id
    assert tracker.tracks[0].miss_count == 1
    assert current[-1].track_id != original_id
    assert current[-1].is_observed is True


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf, "bad"])
def test_invalid_observation_time_does_not_advance_tracker(timestamp: object):
    tracker = _tracker()
    tracker.update_tracking([_ball(100.0)], frame_number=5, timestamp=6.0)
    track = tracker.tracks[0]
    before = track.kalman_filter.state.copy()

    output = tracker.update_tracking([_ball(200.0)], frame_number=6, timestamp=timestamp)  # type: ignore[arg-type]

    assert tracker.frame_number == 5
    assert tracker._last_observation_timestamp == 6.0
    assert track.detection_count == 1
    assert track.miss_count == 0
    assert track.kalman_filter.state.tolist() == before.tolist()
    assert output[0].track_id == track.track_id
    assert tracker.track_statistics["stale_observations_rejected"] == 1


def test_track_remains_associable_at_exact_observation_gap_boundary():
    tracker = _tracker(max_observation_gap_seconds=0.5)
    tracker.update_tracking([_ball(100.0)], frame_number=1, timestamp=10.0)
    track_id = tracker.tracks[0].track_id

    output = tracker.update_tracking([_ball(102.0)], frame_number=2, timestamp=10.5)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == track_id
    assert tracker.tracks[0].detection_count == 2
    assert output[0].track_id == track_id
    assert tracker.track_statistics["tracks_expired_after_gap"] == 0


def test_lost_track_reacquisition_restores_confirmed_state_and_freshness():
    tracker = _tracker(lost_after_seconds=0.2, max_observation_gap_seconds=0.8)
    tracker.update_tracking([_ball(150.0)], frame_number=1, timestamp=20.0)
    track = tracker.tracks[0]
    original_id = track.track_id

    tracker.update_tracking([], frame_number=2, timestamp=20.25)
    assert track.state is TrackState.LOST
    assert tracker._get_tracked_balls()[0].is_observed is False

    output = tracker.update_tracking([_ball(152.0)], frame_number=3, timestamp=20.4)

    assert len(tracker.tracks) == 1
    assert track.track_id == original_id
    assert track.state is TrackState.CONFIRMED
    assert track.miss_count == 0
    assert output[0].is_observed is True


def test_low_confidence_tentative_tracks_stay_hidden_until_confirmation():
    tracker = _tracker(
        min_hits=2,
        return_tentative_tracks=False,
        thresholds={"min_confirmation_confidence": 0.7},
    )

    assert (
        tracker.update_tracking(
            [_ball(100.0, confidence=0.4)], frame_number=1, timestamp=30.0
        )
        == []
    )
    assert tracker.tracks[0].state is TrackState.TENTATIVE
    assert (
        tracker.update_tracking(
            [_ball(101.0, confidence=0.5)], frame_number=2, timestamp=30.1
        )
        == []
    )
    assert tracker.tracks[0].state is TrackState.TENTATIVE

    output = tracker.update_tracking(
        [_ball(102.0, confidence=1.0)], frame_number=3, timestamp=30.2
    )

    assert tracker.tracks[0].state is TrackState.CONFIRMED
    assert len(output) == 1
    assert output[0].track_id == tracker.tracks[0].track_id
    assert tracker.track_statistics["ghost_balls_filtered"] == 2


def test_reset_starts_new_frame_epoch_and_clears_history_and_counters():
    tracker = _tracker()
    tracker.update_tracking([_ball(100.0)], frame_number=50, timestamp=40.0)
    old_id = tracker.tracks[0].track_id
    tracker.update_tracking([], frame_number=51, timestamp=40.9)
    assert tracker.track_history[old_id] == [(100.0, 100.0)]

    tracker.reset()
    output = tracker.update_tracking([_ball(500.0)], frame_number=1, timestamp=1.0)

    assert len(output) == 1
    assert output[0].track_id == 1
    assert tracker.frame_number == 1
    assert tracker.track_history == {}
    assert tracker.track_statistics["total_tracks_created"] == 1
    assert tracker.track_statistics["total_tracks_deleted"] == 0
    assert tracker.track_statistics["stale_observations_rejected"] == 0


@pytest.mark.parametrize("invalid", [None, 0, -1, math.nan, math.inf, "invalid"])
def test_invalid_duration_configuration_uses_documented_fallback(invalid: object):
    tracker = _tracker(
        max_age_seconds=invalid,
        max_observation_gap_seconds=invalid,
        max_prediction_horizon_seconds=invalid,
        lost_after_seconds=invalid,
        tentative_max_age_seconds=invalid,
    )

    assert tracker.max_age_seconds == 1.0
    assert tracker.max_observation_gap_seconds == 1.0
    assert tracker.max_prediction_horizon_seconds == 1.0
    assert tracker.lost_after_seconds == pytest.approx(0.333)
    assert tracker.tentative_max_age_seconds == 1.0


def test_output_preserves_detection_coordinate_metadata_and_last_seen():
    tracker = _tracker()
    detection = _ball(
        250.0,
        coordinate_space="pixel",
        source_resolution=(1280, 720),
    )
    detection.last_seen = 1_700_000_123.0

    output = tracker.update_tracking([detection], frame_number=1, timestamp=50.0)

    assert output[0].coordinate_space == "pixel"
    assert output[0].source_resolution == (1280, 720)
    assert output[0].last_seen == 1_700_000_123.0
