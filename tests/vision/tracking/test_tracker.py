"""Tests for object tracker implementation"""

import numpy as np
import pytest
from backend.vision.tracking.tracker import ObjectTracker, Track, TrackState
from backend.vision.tracking.kalman import KalmanFilter
from backend.vision.models import Ball, BallType


class TestTrack:
    """Test cases for Track class"""

    def setup_method(self):
        """Set up test configuration"""
        self.track_config = {
            'history': {
                'confidence_maxlen': 10,
                'position_maxlen': 50,
                'radius_maxlen': 10
            },
            'thresholds': {
                'default_ball_radius': 15.0,
                'movement_speed': 5.0,
                'collision_speed': 3.0,
                'lost_state_ratio': 0.333,
                'tentative_deletion_misses': 5,
                'lost_deletion_misses': 50
            },
            'penalties': {
                'type_mismatch': 2.0,
                'number_mismatch': 3.0,
                'invalid_measurement': 5.0
            },
            'size_compatibility': {
                'min_ratio': 0.5,
                'max_ratio': 2.0
            }
        }

    def test_track_creation(self):
        """Test track creation with detection"""
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.OTHER,
            number=1,
            confidence=0.8
        )

        kalman_filter = KalmanFilter(ball.position)
        track = Track(
            track_id=1,
            ball_type=ball.ball_type,
            ball_number=ball.number,
            kalman_filter=kalman_filter,
            min_hits=3,
            max_age=30,
            config=self.track_config
        )

        assert track.track_id == 1
        assert track.ball_type == BallType.OTHER
        assert track.ball_number == 1
        assert track.state == TrackState.TENTATIVE
        assert track.age == 0

    def test_track_update_with_detection(self):
        """Test updating track with new detection"""
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.OTHER,
            number=1,
            confidence=0.8
        )

        kalman_filter = KalmanFilter(ball.position)
        track = Track(
            track_id=1,
            ball_type=ball.ball_type,
            ball_number=ball.number,
            kalman_filter=kalman_filter,
            min_hits=3,
            max_age=30,
            config=self.track_config
        )

        # Update with detection
        track.update_with_detection(ball, frame_number=1)

        assert track.detection_count == 1
        assert track.miss_count == 0
        assert track.last_frame_number == 1
        assert len(track.confidence_history) > 0

    def test_track_confirmation(self):
        """Test track state transition to confirmed"""
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.OTHER,
            number=1,
            confidence=0.8
        )

        kalman_filter = KalmanFilter(ball.position)
        track = Track(
            track_id=1,
            ball_type=ball.ball_type,
            ball_number=ball.number,
            kalman_filter=kalman_filter,
            min_hits=3,
            max_age=30,
            config=self.track_config
        )

        # Update multiple times to confirm track
        for i in range(3):
            track.update_with_detection(ball, frame_number=i+1)

        assert track.state == TrackState.CONFIRMED

    def test_track_missing_and_deletion(self):
        """Test track state transitions when missing detections"""
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.OTHER,
            number=1,
            confidence=0.8
        )

        kalman_filter = KalmanFilter(ball.position)
        track = Track(
            track_id=1,
            ball_type=ball.ball_type,
            ball_number=ball.number,
            kalman_filter=kalman_filter,
            min_hits=3,
            max_age=30,
            config=self.track_config
        )

        # Confirm track first
        for i in range(3):
            track.update_with_detection(ball, frame_number=i+1)

        assert track.state == TrackState.CONFIRMED

        # Mark as missed multiple times
        for _ in range(11):
            track.mark_missed()

        assert track.state == TrackState.LOST

        # Mark as missed more times
        for _ in range(31):
            track.mark_missed()

        assert track.state == TrackState.DELETED

    def test_get_current_ball(self):
        """Test converting track to ball object"""
        original_ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.OTHER,
            number=1,
            confidence=0.8
        )

        kalman_filter = KalmanFilter(original_ball.position)
        track = Track(
            track_id=1,
            ball_type=original_ball.ball_type,
            ball_number=original_ball.number,
            kalman_filter=kalman_filter,
            min_hits=3,
            max_age=30,
            config=self.track_config
        )

        current_ball = track.get_current_ball()

        assert current_ball.ball_type == original_ball.ball_type
        assert current_ball.number == original_ball.number
        assert current_ball.position == original_ball.position


class TestObjectTracker:
    """Test cases for ObjectTracker class"""

    def setup_method(self):
        """Set up test configuration"""
        self.config = {
            'max_age': 30,
            'min_hits': 3,
            'max_distance': 50.0,
            'process_noise': 1.0,
            'measurement_noise': 10.0
        }

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = ObjectTracker(self.config)

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.max_distance == 50.0
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1

    def test_single_detection_tracking(self):
        """Test tracking single detection"""
        tracker = ObjectTracker(self.config)

        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.CUE,
            confidence=0.8
        )

        # Update tracker with detection
        tracked_balls = tracker.update_tracking([ball], frame_number=1)

        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].track_id == 1
        assert tracker.tracks[0].state == TrackState.TENTATIVE

        # Track should not be returned until confirmed
        assert len(tracked_balls) == 0

        # Update multiple times to confirm
        for i in range(2, 5):
            tracked_balls = tracker.update_tracking([ball], frame_number=i)

        assert tracker.tracks[0].state == TrackState.CONFIRMED
        assert len(tracked_balls) == 1

    def test_multiple_detections_tracking(self):
        """Test tracking multiple detections"""
        tracker = ObjectTracker(self.config)

        balls = [
            Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8),
            Ball(position=(200.0, 200.0), radius=15.0, ball_type=BallType.OTHER, number=1, confidence=0.8),
            Ball(position=(300.0, 300.0), radius=15.0, ball_type=BallType.OTHER, number=9, confidence=0.8)
        ]

        # Update multiple times to confirm tracks
        for frame in range(1, 5):
            tracked_balls = tracker.update_tracking(balls, frame_number=frame)

        assert len(tracker.tracks) == 3
        confirmed_tracks = [t for t in tracker.tracks if t.state == TrackState.CONFIRMED]
        assert len(confirmed_tracks) == 3
        assert len(tracked_balls) == 3

    def test_detection_association(self):
        """Test detection-to-track association"""
        tracker = ObjectTracker(self.config)

        # Initial detection
        ball1 = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball1], frame_number=frame)

        original_track_id = tracker.tracks[0].track_id

        # Move ball slightly and update
        ball1_moved = Ball(position=(105.0, 105.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)
        tracked_balls = tracker.update_tracking([ball1_moved], frame_number=5)

        # Should still be the same track
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].track_id == original_track_id

    def test_track_loss_and_recovery(self):
        """Test track loss and recovery"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        assert tracker.tracks[0].state == TrackState.CONFIRMED

        # Lose track for several frames
        for frame in range(5, 16):
            tracker.update_tracking([], frame_number=frame)

        assert tracker.tracks[0].state == TrackState.LOST

        # Recover track
        tracked_balls = tracker.update_tracking([ball], frame_number=16)
        assert tracker.tracks[0].state == TrackState.CONFIRMED

    def test_track_deletion(self):
        """Test automatic track deletion"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Create track but don't confirm it
        tracker.update_tracking([ball], frame_number=1)
        assert len(tracker.tracks) == 1

        # Miss too many detections
        for frame in range(2, 8):
            tracker.update_tracking([], frame_number=frame)

        # Track should be deleted
        assert len(tracker.tracks) == 0

    def test_prediction_without_measurement(self):
        """Test position prediction without new measurements"""
        tracker = ObjectTracker(self.config)

        ball = Ball(
            position=(100.0, 100.0),
            radius=15.0,
            ball_type=BallType.CUE,
            confidence=0.8,
            velocity=(10.0, 5.0)
        )

        # Confirm track with initial velocity
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        # Get predictions without new measurements
        predictions = tracker.predict_positions(0.1)  # 0.1 second prediction

        assert len(predictions) == 1
        track_id = list(predictions.keys())[0]
        predicted_pos = predictions[track_id]

        # Position should be predicted based on velocity
        assert predicted_pos[0] > 100.0  # Moved in x direction
        assert predicted_pos[1] > 100.0  # Moved in y direction

    def test_velocity_calculation(self):
        """Test velocity calculation for tracked objects"""
        tracker = ObjectTracker(self.config)

        # Create moving ball
        positions = [(100.0, 100.0), (110.0, 105.0), (120.0, 110.0), (130.0, 115.0)]

        for i, pos in enumerate(positions):
            ball = Ball(position=pos, radius=15.0, ball_type=BallType.CUE, confidence=0.8)
            tracker.update_tracking([ball], frame_number=i+1, timestamp=i*0.1)

        velocities = tracker.get_object_velocities()
        assert len(velocities) == 1

        # Should detect positive velocity in both directions
        velocity = list(velocities.values())[0]
        assert velocity[0] > 0  # Positive x velocity
        assert velocity[1] > 0  # Positive y velocity

    def test_trajectory_tracking(self):
        """Test trajectory history tracking"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Confirm track and build trajectory
        positions = [(100.0, 100.0), (105.0, 105.0), (110.0, 110.0), (115.0, 115.0)]

        for i, pos in enumerate(positions):
            ball.position = pos
            tracker.update_tracking([ball], frame_number=i+1)

        trajectories = tracker.get_track_trajectories()
        assert len(trajectories) == 1

        track_id = list(trajectories.keys())[0]
        trajectory = trajectories[track_id]
        assert len(trajectory) >= len(positions)

    def test_future_trajectory_prediction(self):
        """Test future trajectory prediction"""
        tracker = ObjectTracker(self.config)

        ball = Ball(
            position=(100.0, 100.0),
            radius=15.0,
            ball_type=BallType.CUE,
            confidence=0.8,
            velocity=(10.0, 5.0)
        )

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        track_id = tracker.tracks[0].track_id
        future_trajectory = tracker.predict_future_trajectory(track_id, 5, 0.1)

        assert len(future_trajectory) == 5
        # Each position should be further along the trajectory
        for i in range(1, len(future_trajectory)):
            assert future_trajectory[i][0] > future_trajectory[i-1][0]
            assert future_trajectory[i][1] > future_trajectory[i-1][1]

    def test_ball_type_compatibility(self):
        """Test ball type compatibility in association"""
        tracker = ObjectTracker(self.config)

        # Create cue ball track
        cue_ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        for frame in range(1, 5):
            tracker.update_tracking([cue_ball], frame_number=frame)

        # Try to associate with different ball type at same position
        solid_ball = Ball(position=(101.0, 101.0), radius=15.0, ball_type=BallType.OTHER, number=1, confidence=0.8)

        initial_track_count = len(tracker.tracks)
        tracker.update_tracking([solid_ball], frame_number=5)

        # Should create new track, not associate with existing cue ball track
        assert len(tracker.tracks) == initial_track_count + 1

    def test_tracking_statistics(self):
        """Test tracking statistics collection"""
        tracker = ObjectTracker(self.config)

        balls = [
            Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8),
            Ball(position=(200.0, 200.0), radius=15.0, ball_type=BallType.OTHER, number=1, confidence=0.8)
        ]

        # Create and confirm tracks
        for frame in range(1, 5):
            tracker.update_tracking(balls, frame_number=frame)

        stats = tracker.get_tracking_statistics()

        assert stats['total_tracks_created'] == 2
        assert stats['confirmed_tracks'] == 2
        assert stats['total_active_tracks'] == 2
        assert 'ball_type_distribution' in stats

    def test_tracker_reset(self):
        """Test tracker reset functionality"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Create some tracks
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        assert len(tracker.tracks) > 0
        assert tracker.next_track_id > 1

        # Reset tracker
        tracker.reset()

        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1
        assert tracker.frame_number == 0

    def test_large_displacement_handling(self):
        """Test handling of large displacement between frames"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        # Large displacement (should create new track)
        ball_far = Ball(position=(500.0, 500.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)
        tracker.update_tracking([ball_far], frame_number=5)

        # Should have created new track due to large distance
        assert len(tracker.tracks) == 2

    def test_occlusion_handling(self):
        """Test handling of temporary occlusions"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        # Simulate occlusion (no detections for several frames)
        for frame in range(5, 10):
            tracker.update_tracking([], frame_number=frame)

        # Ball reappears nearby
        ball_reappeared = Ball(position=(110.0, 110.0), radius=15.0, ball_type=BallType.CUE, confidence=0.8)
        tracker.update_tracking([ball_reappeared], frame_number=10)

        # Should recover the same track
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].state == TrackState.CONFIRMED

    @pytest.mark.parametrize("ball_type,expected_tracks", [
        (BallType.CUE, 1),
        (BallType.OTHER, 1),
        (BallType.OTHER, 1),
        (BallType.EIGHT, 1)
    ])
    def test_different_ball_types(self, ball_type, expected_tracks):
        """Test tracking different ball types"""
        tracker = ObjectTracker(self.config)

        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=ball_type, confidence=0.8)

        # Confirm track
        for frame in range(1, 5):
            tracker.update_tracking([ball], frame_number=frame)

        assert len(tracker.tracks) == expected_tracks
        assert tracker.tracks[0].ball_type == ball_type
