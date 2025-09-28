"""Object tracking across frames."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Define local Ball and BallType to avoid import issues
from enum import Enum
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanFilter


class BallType(Enum):
    """Ball type classification."""

    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


@dataclass
class Ball:
    """Ball detection data."""

    position: tuple[float, float]
    radius: float
    ball_type: BallType
    number: Optional[int] = None
    confidence: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    is_moving: bool = False


class TrackState(Enum):
    """Track state enumeration."""

    TENTATIVE = "tentative"  # New track, not yet confirmed
    CONFIRMED = "confirmed"  # Established track
    LOST = "lost"  # Track lost, still predicting
    DELETED = "deleted"  # Track marked for deletion


@dataclass
class Track:
    """Individual object track."""

    track_id: int
    ball_type: BallType
    ball_number: Optional[int]
    kalman_filter: KalmanFilter
    state: TrackState = TrackState.TENTATIVE
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    last_frame_number: int = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=50))
    detection_count: int = 0
    miss_count: int = 0
    radius_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def __post_init__(self):
        """Initialize track after creation."""
        self.confidence_history.append(self.kalman_filter.confidence)
        self.position_history.append(self.kalman_filter.get_position())

    @property
    def age(self) -> int:
        """Get track age in frames."""
        return self.kalman_filter.age

    @property
    def time_since_update(self) -> int:
        """Get frames since last update."""
        return self.kalman_filter.time_since_update

    @property
    def average_confidence(self) -> float:
        """Get average confidence over history."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def average_radius(self) -> float:
        """Get average radius over history."""
        if not self.radius_history:
            return 15.0  # Default ball radius
        return sum(self.radius_history) / len(self.radius_history)

    def update_with_detection(self, detection: Ball, frame_number: int) -> None:
        """Update track with new detection."""
        self.kalman_filter.update(detection.position)
        self.last_update_time = time.time()
        self.last_frame_number = frame_number
        self.detection_count += 1
        self.miss_count = 0

        # Update history
        self.confidence_history.append(detection.confidence)
        self.position_history.append(detection.position)
        self.radius_history.append(detection.radius)

        # Update track state
        if self.state == TrackState.TENTATIVE and self.detection_count >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def predict(self, dt: float) -> tuple[float, float]:
        """Predict track position."""
        predicted_pos = self.kalman_filter.predict(dt)
        self.position_history.append(predicted_pos)
        return predicted_pos

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        self.miss_count += 1

        # Update track state based on miss count
        if self.state == TrackState.CONFIRMED and self.miss_count > 10:
            self.state = TrackState.LOST
        elif self.state == TrackState.TENTATIVE and self.miss_count > 3:
            self.state = TrackState.DELETED
        elif self.state == TrackState.LOST and self.miss_count > 30:
            self.state = TrackState.DELETED

    def is_valid(self) -> bool:
        """Check if track is valid (not deleted)."""
        return self.state != TrackState.DELETED

    def should_be_deleted(self) -> bool:
        """Check if track should be deleted."""
        return (
            self.state == TrackState.DELETED
            or (self.state == TrackState.TENTATIVE and self.miss_count > 5)
            or (self.state == TrackState.LOST and self.miss_count > 50)
        )

    def get_current_ball(self) -> Ball:
        """Convert track to Ball object."""
        position = self.kalman_filter.get_position()
        velocity = self.kalman_filter.get_velocity()
        speed = self.kalman_filter.get_speed()

        return Ball(
            position=position,
            radius=self.average_radius,
            ball_type=self.ball_type,
            number=self.ball_number,
            confidence=self.kalman_filter.confidence,
            velocity=velocity,
            is_moving=speed > 5.0,  # Threshold for movement
        )


class ObjectTracker:
    """Multi-object tracking for balls and cue using Kalman filters and Hungarian algorithm.

    Features:
    - Multi-object tracking with identity management
    - Kalman filter prediction for smooth tracking
    - Track association using Hungarian algorithm
    - Handle track loss and recovery
    - Predict positions during occlusions
    - Velocity and acceleration estimation
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize tracker with configuration.

        Args:
            config: Configuration dictionary with tracking parameters
        """
        # Tracking parameters
        self.max_age = config.get("max_age", 30)  # Max frames to keep lost tracks
        self.min_hits = config.get("min_hits", 3)  # Min detections to confirm track
        self.max_distance = config.get("max_distance", 50.0)  # Max association distance
        self.kalman_process_noise = config.get("process_noise", 1.0)
        self.kalman_measurement_noise = config.get("measurement_noise", 10.0)

        # State
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self.frame_number = 0
        self.last_frame_time = time.time()

        # Performance metrics
        self.track_statistics = {
            "total_tracks_created": 0,
            "total_tracks_deleted": 0,
            "current_active_tracks": 0,
            "average_track_length": 0.0,
        }

        # Track history for analysis
        self.track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.ball_type_tracks: dict[BallType, list[int]] = defaultdict(list)

    def update_tracking(
        self,
        detections: list[Ball],
        frame_number: int,
        timestamp: Optional[float] = None,
    ) -> list[Ball]:
        """Update tracking with new detections.

        Args:
            detections: List of detected balls
            frame_number: Current frame number
            timestamp: Frame timestamp (optional)

        Returns:
            List of tracked balls with IDs and predictions
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate time delta
        dt = timestamp - self.last_frame_time
        self.last_frame_time = timestamp
        self.frame_number = frame_number

        # Step 1: Predict all existing tracks
        self._predict_tracks(dt)

        # Step 2: Associate detections with tracks
        (
            matched_tracks,
            unmatched_detections,
            unmatched_tracks,
        ) = self._associate_detections_to_tracks(detections)

        # Step 3: Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            self.tracks[track_idx].update_with_detection(
                detections[detection_idx], frame_number
            )

        # Step 4: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._create_new_track(detections[detection_idx], frame_number)

        # Step 6: Delete old tracks
        self._delete_old_tracks()

        # Step 7: Update statistics
        self._update_statistics()

        # Step 8: Return current tracked balls
        return self._get_tracked_balls()

    def _predict_tracks(self, dt: float) -> None:
        """Predict positions for all tracks."""
        for track in self.tracks:
            if track.is_valid():
                track.predict(dt)

    def _associate_detections_to_tracks(
        self, detections: list[Ball]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections with existing tracks using Hungarian algorithm.

        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(detections)

        # Solve assignment problem
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Filter out invalid associations
        matched_tracks = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        for track_idx, detection_idx in zip(track_indices, detection_indices):
            cost = cost_matrix[track_idx, detection_idx]

            # Check if association is valid
            if (
                cost < self.max_distance
                and self.tracks[track_idx].is_valid()
                and self._is_compatible_detection(
                    self.tracks[track_idx], detections[detection_idx]
                )
            ):
                matched_tracks.append((track_idx, detection_idx))
                unmatched_detections.discard(detection_idx)
                unmatched_tracks.discard(track_idx)

        return matched_tracks, list(unmatched_detections), list(unmatched_tracks)

    def _build_cost_matrix(self, detections: list[Ball]) -> np.ndarray:
        """Build cost matrix for Hungarian algorithm."""
        valid_tracks = [i for i, track in enumerate(self.tracks) if track.is_valid()]

        if not valid_tracks:
            return np.empty((0, len(detections)))

        cost_matrix = np.full(
            (len(valid_tracks), len(detections)), self.max_distance + 1
        )

        for i, track_idx in enumerate(valid_tracks):
            track = self.tracks[track_idx]
            predicted_pos = track.kalman_filter.get_position()

            for j, detection in enumerate(detections):
                # Euclidean distance
                distance = np.sqrt(
                    (predicted_pos[0] - detection.position[0]) ** 2
                    + (predicted_pos[1] - detection.position[1]) ** 2
                )

                # Add penalty for type mismatch
                if (
                    track.ball_type != detection.ball_type
                    and detection.ball_type != BallType.CUE
                ):
                    distance *= 2.0

                # Add penalty for number mismatch
                if (
                    track.ball_number is not None
                    and detection.number is not None
                    and track.ball_number != detection.number
                ):
                    distance *= 3.0

                # Consider kalman validity
                if not track.kalman_filter.is_valid_measurement(detection.position):
                    distance *= 5.0

                cost_matrix[i, j] = distance

        return cost_matrix

    def _is_compatible_detection(self, track: Track, detection: Ball) -> bool:
        """Check if detection is compatible with track."""
        # Type compatibility
        if (
            track.ball_type != detection.ball_type
            and detection.ball_type != BallType.CUE
            and track.ball_type != BallType.CUE
        ):
            return False

        # Number compatibility (if both are known)
        if (
            track.ball_number is not None
            and detection.number is not None
            and track.ball_number != detection.number
        ):
            return False

        # Size compatibility
        size_ratio = detection.radius / track.average_radius
        if size_ratio < 0.5 or size_ratio > 2.0:
            return False

        return True

    def _create_new_track(self, detection: Ball, frame_number: int) -> None:
        """Create new track from detection."""
        kalman_filter = KalmanFilter(
            initial_position=detection.position,
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
            initial_velocity=detection.velocity
            if detection.velocity != (0, 0)
            else None,
        )

        track = Track(
            track_id=self.next_track_id,
            ball_type=detection.ball_type,
            ball_number=detection.number,
            kalman_filter=kalman_filter,
            last_frame_number=frame_number,
        )

        track.update_with_detection(detection, frame_number)
        self.tracks.append(track)

        # Update tracking data
        self.ball_type_tracks[detection.ball_type].append(self.next_track_id)
        self.track_statistics["total_tracks_created"] += 1

        self.next_track_id += 1

    def _delete_old_tracks(self) -> None:
        """Delete tracks that should be removed."""
        tracks_to_keep = []

        for track in self.tracks:
            if track.should_be_deleted():
                # Update statistics
                self.track_statistics["total_tracks_deleted"] += 1

                # Store final track history
                self.track_history[track.track_id] = list(track.position_history)
            else:
                tracks_to_keep.append(track)

        self.tracks = tracks_to_keep

    def _update_statistics(self) -> None:
        """Update tracking statistics."""
        self.track_statistics["current_active_tracks"] = len(
            [t for t in self.tracks if t.state == TrackState.CONFIRMED]
        )

        if self.track_statistics["total_tracks_deleted"] > 0:
            total_length = sum(len(history) for history in self.track_history.values())
            self.track_statistics["average_track_length"] = (
                total_length / self.track_statistics["total_tracks_deleted"]
            )

    def _get_tracked_balls(self) -> list[Ball]:
        """Get current tracked balls."""
        tracked_balls = []

        for track in self.tracks:
            if track.state == TrackState.CONFIRMED or track.state == TrackState.LOST:
                ball = track.get_current_ball()
                tracked_balls.append(ball)

        return tracked_balls

    def predict_positions(self, time_delta: float) -> dict[int, tuple[float, float]]:
        """Predict object positions for next frame.

        Args:
            time_delta: Time delta for prediction

        Returns:
            Dictionary mapping track IDs to predicted positions
        """
        predictions = {}

        for track in self.tracks:
            if track.is_valid():
                # Create a copy of kalman filter for prediction
                temp_state = track.kalman_filter.state.copy()
                temp_F = track.kalman_filter.F.copy()

                # Update transition matrix
                temp_F[0, 2] = time_delta
                temp_F[1, 3] = time_delta
                temp_F[0, 4] = 0.5 * time_delta**2
                temp_F[1, 5] = 0.5 * time_delta**2
                temp_F[2, 4] = time_delta
                temp_F[3, 5] = time_delta

                # Predict
                predicted_state = temp_F @ temp_state
                predictions[track.track_id] = (predicted_state[0], predicted_state[1])

        return predictions

    def get_object_velocities(self) -> dict[int, tuple[float, float]]:
        """Calculate velocities for tracked objects.

        Returns:
            Dictionary mapping track IDs to velocities
        """
        velocities = {}

        for track in self.tracks:
            if track.state == TrackState.CONFIRMED:
                velocities[track.track_id] = track.kalman_filter.get_velocity()

        return velocities

    def get_object_accelerations(self) -> dict[int, tuple[float, float]]:
        """Calculate accelerations for tracked objects.

        Returns:
            Dictionary mapping track IDs to accelerations
        """
        accelerations = {}

        for track in self.tracks:
            if track.state == TrackState.CONFIRMED:
                accelerations[track.track_id] = track.kalman_filter.get_acceleration()

        return accelerations

    def get_track_trajectories(
        self, track_id: Optional[int] = None
    ) -> dict[int, list[tuple[float, float]]]:
        """Get track trajectories.

        Args:
            track_id: Specific track ID, or None for all tracks

        Returns:
            Dictionary mapping track IDs to position lists
        """
        if track_id is not None:
            for track in self.tracks:
                if track.track_id == track_id:
                    return {track_id: list(track.position_history)}
            # Check historical tracks
            if track_id in self.track_history:
                return {track_id: self.track_history[track_id]}
            return {}

        # Return all trajectories
        trajectories = {}
        for track in self.tracks:
            trajectories[track.track_id] = list(track.position_history)

        # Add historical trajectories
        trajectories.update(self.track_history)

        return trajectories

    def predict_future_trajectory(
        self, track_id: int, time_steps: int, dt: float
    ) -> list[tuple[float, float]]:
        """Predict future trajectory for a specific track.

        Args:
            track_id: Track ID to predict
            time_steps: Number of future steps
            dt: Time delta for each step

        Returns:
            List of predicted positions
        """
        for track in self.tracks:
            if track.track_id == track_id and track.is_valid():
                return track.kalman_filter.predict_trajectory(time_steps, dt)

        return []

    def get_tracking_statistics(self) -> dict[str, Any]:
        """Get comprehensive tracking statistics."""
        stats = self.track_statistics.copy()

        # Add current state information
        stats.update(
            {
                "confirmed_tracks": len(
                    [t for t in self.tracks if t.state == TrackState.CONFIRMED]
                ),
                "tentative_tracks": len(
                    [t for t in self.tracks if t.state == TrackState.TENTATIVE]
                ),
                "lost_tracks": len(
                    [t for t in self.tracks if t.state == TrackState.LOST]
                ),
                "total_active_tracks": len(self.tracks),
                "frame_number": self.frame_number,
            }
        )

        # Add ball type distribution
        ball_type_counts = defaultdict(int)
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED:
                # Handle both enum and string ball types
                ball_type_value = (
                    track.ball_type.value
                    if hasattr(track.ball_type, "value")
                    else str(track.ball_type)
                )
                ball_type_counts[ball_type_value] += 1
        stats["ball_type_distribution"] = dict(ball_type_counts)

        return stats

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_number = 0
        self.track_history.clear()
        self.ball_type_tracks.clear()
        self.track_statistics = {
            "total_tracks_created": 0,
            "total_tracks_deleted": 0,
            "current_active_tracks": 0,
            "average_track_length": 0.0,
        }
