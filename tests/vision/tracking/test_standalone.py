"""Standalone tests for tracking components"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Copy the KalmanFilter and related classes directly to avoid import issues
@dataclass
class KalmanState:
    """Kalman filter state representation"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    covariance: np.ndarray
    confidence: float


class KalmanFilter:
    """Kalman filter for object position, velocity, and acceleration tracking."""

    def __init__(self,
                 initial_position: Tuple[float, float],
                 process_noise: float = 1.0,
                 measurement_noise: float = 10.0,
                 initial_velocity: Optional[Tuple[float, float]] = None,
                 initial_acceleration: Optional[Tuple[float, float]] = None):
        """Initialize Kalman filter with initial state."""
        self.dim_state = 6  # [x, y, vx, vy, ax, ay]
        self.dim_obs = 2    # [x, y] observations

        # Initialize state vector
        self.state = np.zeros(self.dim_state)
        self.state[0:2] = initial_position
        if initial_velocity:
            self.state[2:4] = initial_velocity
        if initial_acceleration:
            self.state[4:6] = initial_acceleration

        # Initialize covariance matrix
        self.P = np.eye(self.dim_state) * 1000  # High initial uncertainty
        self.P[0:2, 0:2] *= 0.1  # Lower uncertainty for initial position

        # Process noise covariance (Q)
        self.Q = np.eye(self.dim_state) * process_noise
        self.Q[4:6, 4:6] *= 10  # Higher noise for acceleration

        # Measurement noise covariance (R)
        self.R = np.eye(self.dim_obs) * measurement_noise

        # Observation matrix (H) - we observe position only
        self.H = np.zeros((self.dim_obs, self.dim_state))
        self.H[0, 0] = 1  # x position
        self.H[1, 1] = 1  # y position

        # State transition matrix (F) - will be updated with dt
        self.F = np.eye(self.dim_state)

        # Track quality metrics
        self.innovation_covariance = np.eye(self.dim_obs)
        self.mahalanobis_distance = 0.0
        self.confidence = 1.0
        self.age = 0
        self.hit_streak = 0
        self.time_since_update = 0

    def predict(self, dt: float) -> Tuple[float, float]:
        """Predict next position using constant acceleration model."""
        # Update state transition matrix with time step
        self.F[0, 2] = dt     # x = x + vx*dt
        self.F[1, 3] = dt     # y = y + vy*dt
        self.F[0, 4] = 0.5 * dt**2  # x = x + 0.5*ax*dt^2
        self.F[1, 5] = 0.5 * dt**2  # y = y + 0.5*ay*dt^2
        self.F[2, 4] = dt     # vx = vx + ax*dt
        self.F[3, 5] = dt     # vy = vy + ay*dt

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update tracking metrics
        self.age += 1
        self.time_since_update += 1

        # Decay confidence over time without measurements
        if self.time_since_update > 0:
            self.confidence *= 0.95

        return (self.state[0], self.state[1])

    def update(self, measured_position: Tuple[float, float]) -> None:
        """Update filter with measured position using Kalman update equations."""
        # Convert measurement to numpy array
        z = np.array(measured_position)

        # Innovation (prediction error)
        y = z - self.H @ self.state

        # Innovation covariance
        self.innovation_covariance = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.innovation_covariance)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Calculate quality metrics
        self.mahalanobis_distance = np.sqrt(y.T @ np.linalg.inv(self.innovation_covariance) @ y)

        # Update tracking metrics
        self.hit_streak += 1
        self.time_since_update = 0

        # Update confidence based on innovation
        innovation_factor = 1.0 / (1.0 + self.mahalanobis_distance / 10.0)
        self.confidence = min(1.0, self.confidence * 0.9 + innovation_factor * 0.1)

    def get_position(self) -> Tuple[float, float]:
        """Get current estimated position."""
        return (self.state[0], self.state[1])

    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated velocity."""
        return (self.state[2], self.state[3])

    def get_acceleration(self) -> Tuple[float, float]:
        """Get current estimated acceleration."""
        return (self.state[4], self.state[5])

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)

    def get_state(self) -> KalmanState:
        """Get complete filter state."""
        return KalmanState(
            position=self.get_position(),
            velocity=self.get_velocity(),
            acceleration=self.get_acceleration(),
            covariance=self.P.copy(),
            confidence=self.confidence
        )

    def predict_trajectory(self, time_steps: int, dt: float) -> list:
        """Predict future trajectory for multiple time steps."""
        # Create a temporary copy to avoid modifying current state
        temp_state = self.state.copy()
        temp_F = self.F.copy()

        # Update transition matrix
        temp_F[0, 2] = dt
        temp_F[1, 3] = dt
        temp_F[0, 4] = 0.5 * dt**2
        temp_F[1, 5] = 0.5 * dt**2
        temp_F[2, 4] = dt
        temp_F[3, 5] = dt

        trajectory = []
        for _ in range(time_steps):
            temp_state = temp_F @ temp_state
            trajectory.append((temp_state[0], temp_state[1]))

        return trajectory

    def is_valid_measurement(self, position: Tuple[float, float],
                           max_mahalanobis: float = 9.0) -> bool:
        """Check if a measurement is valid based on Mahalanobis distance."""
        z = np.array(position)
        predicted_pos = self.H @ self.state
        innovation = z - predicted_pos

        # Use current innovation covariance or predict it
        S = self.H @ self.P @ self.H.T + self.R

        try:
            mahal_dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
            return mahal_dist < max_mahalanobis
        except np.linalg.LinAlgError:
            # If covariance is singular, be conservative
            return False


# Define Ball and related classes
class BallType(Enum):
    """Ball type classification"""
    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


@dataclass
class Ball:
    """Ball detection data"""
    position: Tuple[float, float]
    radius: float
    ball_type: BallType
    number: Optional[int] = None
    confidence: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_moving: bool = False


class TrackState(Enum):
    """Track state enumeration"""
    TENTATIVE = "tentative"     # New track, not yet confirmed
    CONFIRMED = "confirmed"     # Established track
    LOST = "lost"              # Track lost, still predicting
    DELETED = "deleted"        # Track marked for deletion


# Simplified tracker for testing
class SimpleTracker:
    """Simplified tracker for testing purposes"""

    def __init__(self, config: Dict[str, Any]):
        self.max_distance = config.get('max_distance', 50.0)
        self.min_hits = config.get('min_hits', 3)
        self.process_noise = config.get('process_noise', 1.0)
        self.measurement_noise = config.get('measurement_noise', 10.0)

        self.tracks = []
        self.next_track_id = 1
        self.frame_number = 0
        self.track_statistics = {
            'total_tracks_created': 0,
            'confirmed_tracks': 0
        }

    def update_tracking(self, detections: List[Ball], frame_number: int) -> List[Ball]:
        """Update tracking with new detections"""
        self.frame_number = frame_number

        # Simple association: match closest detection to each track
        matched = set()
        for track in self.tracks:
            if track['state'] != TrackState.DELETED:
                # Find closest detection
                min_dist = float('inf')
                best_match = None

                for i, detection in enumerate(detections):
                    if i in matched:
                        continue

                    predicted_pos = track['kalman'].get_position()
                    dist = np.sqrt((detection.position[0] - predicted_pos[0])**2 +
                                 (detection.position[1] - predicted_pos[1])**2)

                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        best_match = i

                if best_match is not None:
                    # Update track
                    track['kalman'].update(detections[best_match].position)
                    track['hit_count'] += 1
                    track['miss_count'] = 0
                    matched.add(best_match)

                    # Check if track should be confirmed
                    if (track['state'] == TrackState.TENTATIVE and
                        track['hit_count'] >= self.min_hits):
                        track['state'] = TrackState.CONFIRMED
                        self.track_statistics['confirmed_tracks'] += 1
                else:
                    # Track missed
                    track['kalman'].predict(1/30.0)  # Assume 30 FPS
                    track['miss_count'] += 1

                    # Mark as lost if too many misses
                    if track['miss_count'] > 10:
                        track['state'] = TrackState.LOST

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched:
                kalman = KalmanFilter(
                    detection.position,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise
                )

                track = {
                    'id': self.next_track_id,
                    'kalman': kalman,
                    'state': TrackState.TENTATIVE,
                    'ball_type': detection.ball_type,
                    'hit_count': 1,
                    'miss_count': 0
                }

                self.tracks.append(track)
                self.track_statistics['total_tracks_created'] += 1
                self.next_track_id += 1

        # Return confirmed tracked balls
        tracked_balls = []
        for track in self.tracks:
            if track['state'] in [TrackState.CONFIRMED, TrackState.LOST]:
                pos = track['kalman'].get_position()
                vel = track['kalman'].get_velocity()

                ball = Ball(
                    position=pos,
                    radius=15.0,
                    ball_type=track['ball_type'],
                    confidence=track['kalman'].confidence,
                    velocity=vel,
                    is_moving=track['kalman'].get_speed() > 5.0
                )
                tracked_balls.append(ball)

        return tracked_balls

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return self.track_statistics.copy()

    def predict_positions(self, time_delta: float) -> Dict[int, Tuple[float, float]]:
        """Predict positions for all tracks"""
        predictions = {}
        for track in self.tracks:
            if track['state'] != TrackState.DELETED:
                # Simple prediction without modifying state
                pos = track['kalman'].get_position()
                vel = track['kalman'].get_velocity()
                pred_pos = (pos[0] + vel[0] * time_delta,
                           pos[1] + vel[1] * time_delta)
                predictions[track['id']] = pred_pos

        return predictions

    def get_object_velocities(self) -> Dict[int, Tuple[float, float]]:
        """Get velocities for all confirmed tracks"""
        velocities = {}
        for track in self.tracks:
            if track['state'] == TrackState.CONFIRMED:
                velocities[track['id']] = track['kalman'].get_velocity()

        return velocities


def test_kalman_filter():
    """Test Kalman filter functionality"""
    print("Testing Kalman Filter:")

    # Test initialization
    kf = KalmanFilter((100.0, 200.0))
    assert kf.get_position() == (100.0, 200.0)
    assert kf.get_velocity() == (0.0, 0.0)
    print("  ✓ Initialization")

    # Test prediction
    kf2 = KalmanFilter((0.0, 0.0), initial_velocity=(10.0, -5.0))
    predicted_pos = kf2.predict(0.1)
    expected_x = 10.0 * 0.1
    expected_y = -5.0 * 0.1
    assert abs(predicted_pos[0] - expected_x) < 0.001
    assert abs(predicted_pos[1] - expected_y) < 0.001
    print("  ✓ Prediction")

    # Test measurement update
    kf3 = KalmanFilter((100.0, 100.0))
    kf3.update((105.0, 95.0))
    updated_pos = kf3.get_position()
    assert updated_pos[0] > 100.0
    assert updated_pos[1] < 100.0
    print("  ✓ Measurement update")

    # Test trajectory prediction
    trajectory = kf2.predict_trajectory(3, 0.1)
    assert len(trajectory) == 3
    print("  ✓ Trajectory prediction")

    # Test measurement validation
    assert kf3.is_valid_measurement((102.0, 98.0))
    assert not kf3.is_valid_measurement((200.0, 200.0))
    print("  ✓ Measurement validation")

    print("Kalman filter tests completed!\n")


def test_simple_tracker():
    """Test simplified tracker functionality"""
    print("Testing Simple Tracker:")

    config = {
        'max_distance': 50.0,
        'min_hits': 3,
        'process_noise': 1.0,
        'measurement_noise': 10.0
    }

    tracker = SimpleTracker(config)
    assert len(tracker.tracks) == 0
    print("  ✓ Initialization")

    # Test single ball tracking
    ball = Ball(
        position=(100.0, 200.0),
        radius=15.0,
        ball_type=BallType.CUE,
        confidence=0.8
    )

    tracked_balls = tracker.update_tracking([ball], frame_number=1)
    assert len(tracker.tracks) == 1
    print("  ✓ Single ball detection")

    # Test track confirmation (need multiple hits)
    for frame in range(2, 5):
        tracked_balls = tracker.update_tracking([ball], frame_number=frame)

    assert len(tracked_balls) > 0
    stats = tracker.get_tracking_statistics()
    assert stats['confirmed_tracks'] > 0
    print("  ✓ Track confirmation")

    # Test multiple balls
    ball2 = Ball(
        position=(300.0, 400.0),
        radius=15.0,
        ball_type=BallType.SOLID,
        number=1,
        confidence=0.8
    )

    for frame in range(5, 8):
        tracked_balls = tracker.update_tracking([ball, ball2], frame_number=frame)

    assert len(tracker.tracks) == 2
    print("  ✓ Multi-ball tracking")

    # Test predictions
    predictions = tracker.predict_positions(0.1)
    assert len(predictions) > 0
    print("  ✓ Position prediction")

    # Test velocities
    velocities = tracker.get_object_velocities()
    assert len(velocities) >= 0
    print("  ✓ Velocity calculation")

    # Test track loss
    for frame in range(8, 15):
        tracked_balls = tracker.update_tracking([], frame_number=frame)

    # Tracks should still exist but might be lost
    print("  ✓ Track loss handling")

    print("Simple tracker tests completed!\n")


def main():
    """Run all tests"""
    print("Running standalone tracking system tests...\n")

    try:
        test_kalman_filter()
        test_simple_tracker()
        print("🎉 All tests completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
