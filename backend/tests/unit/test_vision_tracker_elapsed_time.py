"""Elapsed-time tracking behavior and discontinuity safeguards."""

from backend.common.models import Ball
from backend.vision.models import BallType
from backend.vision.tracking.tracker import ObjectTracker, TrackState


def _ball(x: float, y: float = 100.0, *, velocity=(0.0, 0.0)) -> Ball:
    return Ball(
        position=(x, y),
        radius=15.0,
        ball_type=BallType.CUE,
        confidence=0.95,
        velocity=velocity,
    )


def _tracker(*, clock=None, **overrides) -> ObjectTracker:
    config = {
        "min_hits": 1,
        "max_age_seconds": 1.0,
        "max_observation_gap_seconds": 0.5,
        "max_prediction_horizon_seconds": 0.25,
        "max_distance": 200.0,
        "return_tentative_tracks": True,
    }
    config.update(overrides)
    return ObjectTracker(config, clock=clock)


def test_missing_timestamp_uses_injected_monotonic_clock():
    now = [3.0]
    tracker = _tracker(clock=lambda: now[0])

    tracker.update_tracking([_ball(100.0)], frame_number=1)
    now[0] = 3.2
    tracker.update_tracking([_ball(102.0)], frame_number=2)

    assert tracker._last_observation_timestamp == 3.2
    assert tracker.tracks[0].last_update_time == 3.2
    assert tracker.tracks[0].state is TrackState.CONFIRMED


def test_unobserved_tracks_preserve_wall_time_until_a_new_detection():
    tracker = _tracker(
        max_observation_gap_seconds=0.8,
        lost_after_seconds=0.6,
    )
    first = _ball(100.0)
    first.last_seen = 1_700_000_000.0

    tracker.update_tracking([first], frame_number=1, timestamp=10.0)
    missed = tracker.update_tracking([], frame_number=2, timestamp=10.2)

    assert len(missed) == 1
    assert missed[0].is_observed is False
    assert missed[0].last_seen == 1_700_000_000.0
    assert tracker.tracks[0].last_update_time == 10.0

    next_detection = _ball(104.0)
    next_detection.last_seen = 1_700_000_000.4
    observed = tracker.update_tracking([next_detection], frame_number=3, timestamp=10.4)

    assert len(observed) == 1
    assert observed[0].is_observed is True
    assert observed[0].last_seen == 1_700_000_000.4
    assert tracker.tracks[0].last_update_time == 10.4


def test_duplicate_and_out_of_order_observations_are_ignored():
    tracker = _tracker()
    tracker.update_tracking([_ball(100.0)], frame_number=10, timestamp=1.0)
    tracker.update_tracking([_ball(102.0)], frame_number=12, timestamp=1.2)
    before_position = tracker.tracks[0].kalman_filter.get_position()
    before_detection_count = tracker.tracks[0].detection_count

    tracker.update_tracking([_ball(300.0)], frame_number=12, timestamp=1.2)
    tracker.update_tracking([_ball(300.0)], frame_number=11, timestamp=1.1)
    tracker.update_tracking([_ball(300.0)], frame_number=11, timestamp=1.3)

    assert tracker.tracks[0].kalman_filter.get_position() == before_position
    assert tracker.tracks[0].detection_count == before_detection_count
    assert tracker.track_statistics["stale_observations_rejected"] == 3


def test_excessive_observation_gap_expires_and_reacquires_with_new_id():
    tracker = _tracker(max_observation_gap_seconds=0.4)
    tracker.update_tracking([_ball(100.0)], frame_number=1, timestamp=5.0)
    original_track_id = tracker.tracks[0].track_id

    tracked = tracker.update_tracking([_ball(120.0)], frame_number=2, timestamp=5.5)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id != original_track_id
    assert tracker.track_statistics["tracks_expired_after_gap"] == 1
    assert tracker.track_history[original_track_id] == [(100.0, 100.0)]
    assert tracked[0].track_id == tracker.tracks[0].track_id


def test_prediction_horizon_is_bounded_and_history_contains_observations_only():
    tracker = _tracker(max_prediction_horizon_seconds=0.25)
    tracker.update_tracking(
        [_ball(100.0, velocity=(20.0, 0.0))], frame_number=1, timestamp=1.0
    )
    track_id = tracker.tracks[0].track_id

    assert tracker.predict_positions(0.5) == {}
    assert len(tracker.predict_future_trajectory(track_id, time_steps=10, dt=0.1)) == 2
    tracker.update_tracking([], frame_number=2, timestamp=1.1)
    assert tracker.get_track_trajectories(track_id)[track_id] == [(100.0, 100.0)]


def test_missed_track_transitions_depend_on_elapsed_seconds_not_call_count():
    tracker = _tracker(lost_after_seconds=0.3, max_observation_gap_seconds=0.8)
    tracker.update_tracking([_ball(100.0)], frame_number=1, timestamp=0.0)

    for frame_number in range(2, 22):
        tracker.update_tracking(
            [], frame_number=frame_number, timestamp=frame_number / 1000
        )
    assert tracker.tracks[0].state is TrackState.CONFIRMED

    tracker.update_tracking([], frame_number=22, timestamp=0.31)
    assert tracker.tracks[0].state is TrackState.LOST


def test_missing_max_age_seconds_uses_elapsed_time_default():
    tracker = _tracker(
        max_age_seconds=None,
        max_observation_gap_seconds=None,
        max_prediction_horizon_seconds=None,
    )

    assert tracker.max_age_seconds == 1.0
    assert tracker.max_observation_gap_seconds == 1.0
    assert tracker.max_prediction_horizon_seconds == 1.0


def test_max_age_seconds_controls_lost_track_expiry_independently_of_frame_rate():
    tracker = _tracker(
        max_age_seconds=0.6,
        lost_after_seconds=0.2,
        max_observation_gap_seconds=1.0,
    )
    tracker.update_tracking([_ball(100.0)], frame_number=1, timestamp=1.0)
    track = tracker.tracks[0]

    tracker.update_tracking([], frame_number=2, timestamp=1.21)
    assert track.state is TrackState.LOST

    tracker.update_tracking([], frame_number=3, timestamp=1.6)
    assert track.state is TrackState.DELETED
