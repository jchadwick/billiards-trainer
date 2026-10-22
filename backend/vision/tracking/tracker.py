"""Object tracking across frames."""

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from backend.common.models import Ball
from backend.vision.models import BallType

from .kalman import KalmanFilter


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
    min_hits: int
    config: dict[str, Any]  # Configuration for thresholds and history sizes
    state: TrackState = TrackState.TENTATIVE
    creation_time: float = field(default_factory=time.monotonic)
    last_update_time: float = field(default_factory=time.monotonic)
    # Wall-clock timestamp of the most recent detector match. This is kept
    # separate from last_update_time, which is monotonic and drives tracking.
    last_observed_wall_time: float = field(default_factory=time.time)
    last_frame_number: int = 0
    confidence_history: deque[float] = field(default_factory=deque)
    position_history: deque[tuple[float, float]] = field(default_factory=deque)
    detection_count: int = 0
    miss_count: int = 0
    radius_history: deque[float] = field(default_factory=deque)

    # Coordinate space metadata - preserved from detections to prevent double-scaling
    coordinate_space: str = "4k_canonical"  # "4k_canonical", "pixel", or "world"
    source_resolution: Optional[tuple[int, int]] = (
        None  # (width, height) of source frame
    )

    def __post_init__(self) -> None:
        """Initialize track after creation."""
        # Initialize deques with maxlen from config
        history_config = self.config.get("history", {})
        self.confidence_history = deque(
            maxlen=history_config.get("confidence_maxlen", 10)
        )
        self.position_history = deque(maxlen=history_config.get("position_maxlen", 50))
        self.radius_history = deque(maxlen=history_config.get("radius_maxlen", 10))

        # Add initial values
        self.confidence_history.append(self.kalman_filter.confidence)
        self.position_history.append(self.kalman_filter.get_position())

    @property
    def age(self) -> int:
        """Get track age in frames."""
        return int(self.kalman_filter.age)

    @property
    def time_since_update(self) -> int:
        """Get frames since last update."""
        return int(self.kalman_filter.time_since_update)

    @property
    def average_confidence(self) -> float:
        """Get average confidence over history."""
        if not self.confidence_history:
            return 0.0
        return float(sum(self.confidence_history) / len(self.confidence_history))

    @property
    def average_radius(self) -> float:
        """Get average radius over history."""
        if not self.radius_history:
            thresholds = self.config.get("thresholds", {})
            return float(thresholds.get("default_ball_radius", 15.0))
        return float(sum(self.radius_history) / len(self.radius_history))

    def update_with_detection(
        self,
        detection: Ball,
        frame_number: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update track with new detection."""
        if timestamp is None:
            timestamp = time.monotonic()
        self.kalman_filter.update(detection.position)
        self.last_update_time = timestamp
        self.last_observed_wall_time = detection.last_seen
        self.last_frame_number = frame_number
        self.detection_count += 1
        self.miss_count = 0

        # Update history
        self.confidence_history.append(detection.confidence)
        if self.detection_count > 1:
            self.position_history.append(detection.position)
        self.radius_history.append(detection.radius)

        # Preserve coordinate metadata from detection to prevent double-scaling
        if hasattr(detection, "coordinate_space"):
            self.coordinate_space = detection.coordinate_space
        if hasattr(detection, "source_resolution"):
            self.source_resolution = detection.source_resolution

        # Update ball type if detection has more specific type information
        # Only update if detection has classified type (not UNKNOWN)
        if detection.ball_type != BallType.UNKNOWN:
            self.ball_type = detection.ball_type
            self.ball_number = detection.number

        # Update track state
        # Require both hit count AND minimum average confidence to confirm
        min_confidence_threshold = self.config.get("thresholds", {}).get(
            "min_confirmation_confidence", 0.3
        )
        if (
            self.state == TrackState.TENTATIVE
            and self.detection_count >= self.min_hits
            and self.average_confidence >= min_confidence_threshold
            or self.state == TrackState.LOST
        ):
            self.state = TrackState.CONFIRMED

    def predict(self, dt: float) -> tuple[float, float]:
        """Predict track position."""
        # Keep position_history observation-only. Predicted samples are useful for
        # association, but displaying them as a trajectory would imply continuity
        # across intervals in which the camera provided no ball observation.
        predicted_pos = self.kalman_filter.predict(dt)
        return (float(predicted_pos[0]), float(predicted_pos[1]))

    def mark_missed(self, timestamp: Optional[float] = None) -> None:
        """Mark a track unobserved, transitioning by elapsed time rather than calls."""
        if timestamp is None:
            timestamp = time.monotonic()
        self.miss_count += 1

        max_age_seconds = self.config["max_age_seconds"]
        lost_after_seconds = self.config["lost_after_seconds"]
        tentative_max_age_seconds = self.config["tentative_max_age_seconds"]
        elapsed = max(0.0, timestamp - self.last_update_time)

        if self.state == TrackState.CONFIRMED and elapsed >= float(lost_after_seconds):
            self.state = TrackState.LOST
        elif (
            self.state == TrackState.TENTATIVE
            and elapsed >= float(tentative_max_age_seconds)
        ) or (self.state == TrackState.LOST and elapsed >= float(max_age_seconds)):
            self.state = TrackState.DELETED

    def is_valid(self) -> bool:
        """Check if track is valid (not deleted)."""
        return self.state != TrackState.DELETED

    def should_be_deleted(self, timestamp: Optional[float] = None) -> bool:
        """Check expiry against elapsed observation time."""
        if timestamp is None:
            timestamp = time.monotonic()
        max_age_seconds = self.config["max_age_seconds"]
        tentative_max_age_seconds = self.config["tentative_max_age_seconds"]
        elapsed = max(0.0, timestamp - self.last_update_time)
        return (
            self.state == TrackState.DELETED
            or (
                self.state == TrackState.TENTATIVE
                and elapsed >= float(tentative_max_age_seconds)
            )
            or (self.state == TrackState.LOST and elapsed >= float(max_age_seconds))
        )

    def get_current_ball(self) -> Ball:
        """Convert track to Ball object."""
        position = self.kalman_filter.get_position()
        velocity = self.kalman_filter.get_velocity()
        speed = self.kalman_filter.get_speed()

        thresholds = self.config.get("thresholds", {})
        movement_speed = thresholds.get("movement_speed", 5.0)

        return Ball(
            position=position,
            radius=self.average_radius,
            ball_type=self.ball_type,
            number=self.ball_number,
            confidence=self.kalman_filter.confidence,
            is_observed=self.miss_count == 0,
            velocity=velocity,
            is_moving=speed > movement_speed,
            track_id=self.track_id,
            last_seen=self.last_observed_wall_time,
            coordinate_space=self.coordinate_space,
            source_resolution=self.source_resolution,
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
    - Ghost ball filtering to prevent false detections during high motion events
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """Initialize tracker with configuration.

        These default values have been tuned through empirical testing with video_debugger
        to provide optimal tracking stability and ghost ball rejection for pool table
        scenarios. The parameters balance between:
        - Quick initial detection of stationary balls
        - Stable tracking during motion
        - Aggressive filtering of ghost balls during collisions/high-speed events
        - Smooth velocity estimation for physics calculations

        Args:
            config: Configuration dictionary with tracking parameters

        Key Parameters:
            max_age_seconds (default: 1.0): Maximum lifetime after the last real
                observation. Track expiry is based on elapsed time, not frame rate.
            max_observation_gap_seconds: Gap beyond which a track is expired and a
                detection must reacquire with a new track ID.
            max_prediction_horizon_seconds: Upper bound for public extrapolation and
                filter prediction between observations.

            min_hits (default: 10): Number of consecutive detections required before a
                track is confirmed. This is CRITICAL for ghost ball filtering. Higher
                values (8-10) prevent transient false detections during ball collisions
                from being reported as real balls. Lower values (3-5) cause ghost balls
                to appear during collisions.

            max_distance (default: 100.0): Maximum pixel distance for associating a
                detection with a track. Very forgiving to handle stationary ball
                detection jitter and ensure balls don't lose their IDs.

            process_noise (default: 5.0): Kalman filter process noise. Higher values
                (5.0) make the filter more responsive to detection jitter, which is
                necessary because ball detections can vary by several pixels frame-to-frame
                even when stationary.

            measurement_noise (default: 20.0): Kalman filter measurement noise. Higher
                tolerance for noisy detections, smooths out jittery position estimates.

            collision_threshold (default: 60.0): Distance threshold (pixels) to detect
                potential ball collisions. Used to trigger more conservative track
                confirmation during high-motion events.

            min_hits_during_collision (default: 30): Higher confirmation threshold during
                collisions. Extremely conservative to prevent ANY ghost balls from being
                confirmed during the chaotic collision phase.

            return_tentative_tracks (default: False): When False, only returns tracks
                that have met min_hits threshold. Essential for ghost ball filtering -
                tentative tracks are never returned to prevent false positives.
        """
        # All tracker timing uses a monotonic clock. Callers may inject a clock for
        # deterministic tests; production integration should pass the observation's
        # monotonic timestamp to update_tracking when one is available.
        self.clock = clock or time.monotonic

        self.max_age_seconds = self._positive_seconds(
            config.get("max_age_seconds"), 1.0
        )
        self.max_observation_gap_seconds = self._positive_seconds(
            config.get("max_observation_gap_seconds"), self.max_age_seconds
        )
        self.max_prediction_horizon_seconds = self._positive_seconds(
            config.get("max_prediction_horizon_seconds"),
            self.max_observation_gap_seconds,
        )
        lost_state_ratio = config.get("thresholds", {}).get("lost_state_ratio", 0.333)
        self.lost_after_seconds = self._positive_seconds(
            config.get("lost_after_seconds"),
            self.max_age_seconds * float(lost_state_ratio),
        )
        self.tentative_max_age_seconds = self._positive_seconds(
            config.get("tentative_max_age_seconds"), self.max_age_seconds
        )

        # Track instances share the resolved settings but the caller's dictionary
        # is left untouched.
        self.config = dict(config)
        self.config.update(
            {
                "max_age_seconds": self.max_age_seconds,
                "max_observation_gap_seconds": self.max_observation_gap_seconds,
                "max_prediction_horizon_seconds": self.max_prediction_horizon_seconds,
                "lost_after_seconds": self.lost_after_seconds,
                "tentative_max_age_seconds": self.tentative_max_age_seconds,
            }
        )

        # Tracking parameters with tuned defaults from video_debugger testing
        self.min_hits = config.get(
            "min_hits", 3
        )  # Reduced from 10 to 3 for faster ball confirmation (100ms @ 30fps)
        self.max_distance = config.get(
            "max_distance", 200.0
        )  # Increased from 100 to handle fast-moving balls
        self.kalman_process_noise = config.get(
            "process_noise", 15.0
        )  # Increased from 10.0 for even more responsive motion tracking
        self.kalman_measurement_noise = config.get(
            "measurement_noise", 10.0
        )  # Reduced from 15.0 for tighter position lock

        # Ghost ball filtering parameters
        self.collision_threshold = config.get("collision_threshold", 60.0)
        self.min_hits_during_collision = config.get(
            "min_hits_during_collision", 5
        )  # Reduced from 30 to 5 for faster confirmation during motion
        self.motion_speed_threshold = config.get("motion_speed_threshold", 10.0)
        self.return_tentative_tracks = config.get("return_tentative_tracks", False)

        # State
        self.tracks: list[Track] = []
        self.next_track_id = 1
        self.frame_number = 0
        self.last_frame_time: Optional[float] = None
        self._last_observed_frame_number: Optional[int] = None
        self._last_observation_timestamp: Optional[float] = None
        self.collision_detected = False

        # Performance metrics
        self.track_statistics = {
            "total_tracks_created": 0,
            "total_tracks_deleted": 0,
            "current_active_tracks": 0,
            "average_track_length": 0.0,
            "ghost_balls_filtered": 0,
            "collision_frames": 0,
            "stale_observations_rejected": 0,
            "tracks_expired_after_gap": 0,
        }

        # Track history for analysis
        self.track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.ball_type_tracks: dict[BallType, list[int]] = defaultdict(list)

    @staticmethod
    def _positive_seconds(value: Any, fallback: float) -> float:
        """Return a finite positive duration, falling back for invalid config."""
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return seconds if math.isfinite(seconds) and seconds > 0.0 else float(fallback)

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
            timestamp = self.clock()
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            self.track_statistics["stale_observations_rejected"] += 1
            return self._get_tracked_balls()

        # Duplicate/out-of-order work must not mutate filter state or count as a
        # miss. Frame IDs complement timestamps so a restarted or replayed source
        # cannot silently splice observations into an existing track.
        if (
            not math.isfinite(timestamp)
            or (
                self._last_observation_timestamp is not None
                and timestamp <= self._last_observation_timestamp
            )
            or (
                self._last_observed_frame_number is not None
                and frame_number <= self._last_observed_frame_number
            )
        ):
            self.track_statistics["stale_observations_rejected"] += 1
            return self._get_tracked_balls()

        # Never extrapolate or associate a track after its observation has gone
        # stale. Expired tracks are archived with measured points only; detections
        # in this frame receive new IDs and form a visible discontinuity.
        self._expire_stale_tracks(timestamp)

        # Calculate time delta from observations, not processing call counts.
        dt = (
            0.0
            if self.last_frame_time is None
            else max(0.0, timestamp - self.last_frame_time)
        )
        self.last_frame_time = timestamp
        self.frame_number = frame_number
        self._last_observation_timestamp = timestamp
        self._last_observed_frame_number = frame_number

        # Step 1: Predict all existing tracks
        self._predict_tracks(min(dt, self.max_prediction_horizon_seconds))

        # Step 1.5: Detect collisions or high-motion events
        self.collision_detected = self._detect_collision_or_high_motion()
        if self.collision_detected:
            self.track_statistics["collision_frames"] += 1

        # Step 2: Associate detections with tracks
        (
            matched_tracks,
            unmatched_detections,
            unmatched_tracks,
        ) = self._associate_detections_to_tracks(detections)

        # Step 3: Update matched tracks
        for track_idx, detection_idx in matched_tracks:
            self.tracks[track_idx].update_with_detection(
                detections[detection_idx], frame_number, timestamp
            )

        # Step 4: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(timestamp)

        # Step 5: Create new tracks for unmatched detections
        # During collision, be more conservative about creating new tracks
        for detection_idx in unmatched_detections:
            self._create_new_track(detections[detection_idx], frame_number, timestamp)

        # Step 6: Delete old tracks
        self._delete_old_tracks(timestamp)

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

        # Build cost matrix and get per-track distance thresholds
        # NOTE: cost_matrix uses valid_tracks indices, not actual track indices
        cost_matrix, distance_thresholds, valid_track_indices = self._build_cost_matrix(
            detections
        )

        if len(valid_track_indices) == 0:
            # No valid tracks, all detections are unmatched
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # Solve assignment problem
        cost_matrix_row_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Filter out invalid associations
        matched_tracks = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))

        for cost_matrix_row_idx, detection_idx in zip(
            cost_matrix_row_indices, detection_indices
        ):
            # Map from cost matrix row index to actual track index
            actual_track_idx = valid_track_indices[cost_matrix_row_idx]

            cost = cost_matrix[cost_matrix_row_idx, detection_idx]
            threshold = distance_thresholds[cost_matrix_row_idx]

            # Check if association is valid using per-track threshold
            if (
                cost < threshold
                and self.tracks[actual_track_idx].is_valid()
                and self._is_compatible_detection(
                    self.tracks[actual_track_idx], detections[detection_idx]
                )
            ):
                matched_tracks.append((actual_track_idx, detection_idx))
                unmatched_detections.discard(detection_idx)
                unmatched_tracks.discard(actual_track_idx)

        return matched_tracks, list(unmatched_detections), list(unmatched_tracks)

    def _build_cost_matrix(
        self, detections: list[Ball]
    ) -> tuple[NDArray[np.float64], list[float], list[int]]:
        """Build cost matrix for Hungarian algorithm.

        Returns:
            Tuple of (cost_matrix, distance_thresholds, valid_track_indices) where:
            - cost_matrix: NxM matrix of association costs
            - distance_thresholds: per-track maximum association distances
            - valid_track_indices: mapping from cost_matrix row index to actual track index
        """
        valid_tracks = [i for i, track in enumerate(self.tracks) if track.is_valid()]

        if not valid_tracks:
            return np.empty((0, len(detections))), [], []

        cost_matrix = np.full(
            (len(valid_tracks), len(detections)), self.max_distance + 1
        )
        distance_thresholds = []

        for i, track_idx in enumerate(valid_tracks):
            track = self.tracks[track_idx]
            predicted_pos = track.kalman_filter.get_position()

            # Get velocity to adjust search radius for fast-moving balls
            velocity = track.kalman_filter.get_velocity()
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

            # Expand search radius for fast-moving balls
            # Allow up to 2x max_distance for balls moving > 20 pixels/frame
            speed_factor = min(2.0, 1.0 + (speed / 20.0))
            adjusted_max_distance = self.max_distance * speed_factor
            distance_thresholds.append(adjusted_max_distance)

            for j, detection in enumerate(detections):
                # Euclidean distance
                distance = np.sqrt(
                    (predicted_pos[0] - detection.position[0]) ** 2
                    + (predicted_pos[1] - detection.position[1]) ** 2
                )

                # Get penalty multipliers from config
                penalties = self.config.get("penalties", {})
                type_mismatch_penalty = penalties.get("type_mismatch", 2.0)
                number_mismatch_penalty = penalties.get("number_mismatch", 3.0)
                invalid_measurement_penalty = penalties.get("invalid_measurement", 5.0)

                # Add a penalty for different known types. UNKNOWN is a
                # generic detector result and may continue an established
                # track, but known identities (including CUE) must not match
                # a different known ball type.
                if (
                    track.ball_type != detection.ball_type
                    and detection.ball_type != BallType.UNKNOWN
                    and track.ball_type != BallType.UNKNOWN
                ):
                    distance *= type_mismatch_penalty

                # Add penalty for number mismatch
                if (
                    track.ball_number is not None
                    and detection.number is not None
                    and track.ball_number != detection.number
                ):
                    distance *= number_mismatch_penalty

                # Consider kalman validity
                if not track.kalman_filter.is_valid_measurement(detection.position):
                    distance *= invalid_measurement_penalty

                cost_matrix[i, j] = distance

        return cost_matrix, distance_thresholds, valid_tracks

    def _is_compatible_detection(self, track: Track, detection: Ball) -> bool:
        """Check if detection is compatible with track."""
        # Type compatibility
        # UNKNOWN detections may continue any track because the generic model
        # class does not identify a specific ball. Every known type, including
        # CUE, must otherwise match exactly to avoid switching identities.
        if (
            track.ball_type != detection.ball_type
            and detection.ball_type != BallType.UNKNOWN
            and track.ball_type != BallType.UNKNOWN
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
        size_compatibility = self.config.get("size_compatibility", {})
        min_ratio = size_compatibility.get("min_ratio", 0.5)
        max_ratio = size_compatibility.get("max_ratio", 2.0)

        size_ratio = detection.radius / track.average_radius
        return not (size_ratio < min_ratio or size_ratio > max_ratio)

    def _create_new_track(
        self, detection: Ball, frame_number: int, timestamp: float
    ) -> None:
        """Create new track from detection."""
        kalman_filter = KalmanFilter(
            initial_position=detection.position,
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
            initial_velocity=(
                detection.velocity if detection.velocity != (0, 0) else None
            ),
        )

        # Preserve coordinate metadata from detection to prevent double-scaling
        coordinate_space = (
            detection.coordinate_space
            if hasattr(detection, "coordinate_space")
            else "4k_canonical"
        )
        source_resolution = (
            detection.source_resolution
            if hasattr(detection, "source_resolution")
            else None
        )

        track = Track(
            track_id=self.next_track_id,
            ball_type=detection.ball_type,
            ball_number=detection.number,
            kalman_filter=kalman_filter,
            min_hits=self.min_hits,
            config=self.config,
            creation_time=timestamp,
            last_update_time=timestamp,
            last_frame_number=frame_number,
            coordinate_space=coordinate_space,
            source_resolution=source_resolution,
        )

        track.update_with_detection(detection, frame_number, timestamp)
        self.tracks.append(track)

        # Update tracking data
        self.ball_type_tracks[detection.ball_type].append(self.next_track_id)
        self.track_statistics["total_tracks_created"] += 1

        self.next_track_id += 1

    def _expire_stale_tracks(self, timestamp: float) -> None:
        """Archive tracks that cannot safely bridge the elapsed observation gap."""
        retained_tracks: list[Track] = []
        for track in self.tracks:
            observation_gap = timestamp - track.last_update_time
            if observation_gap > self.max_observation_gap_seconds:
                track.state = TrackState.DELETED
                self._archive_track(track)
                self.track_statistics["tracks_expired_after_gap"] += 1
            else:
                retained_tracks.append(track)
        self.tracks = retained_tracks

    def _archive_track(self, track: Track) -> None:
        """Store a track's observed history before removing it."""
        self.track_statistics["total_tracks_deleted"] += 1
        self.track_history[track.track_id] = list(track.position_history)

    def _delete_old_tracks(self, timestamp: float) -> None:
        """Delete tracks that should be removed."""
        tracks_to_keep = []

        for track in self.tracks:
            if track.should_be_deleted(timestamp):
                self._archive_track(track)
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

    def _detect_collision_or_high_motion(self) -> bool:
        """Detect if balls are colliding or in high-motion state.

        Collisions are characterized by:
        - Two or more balls very close together (< collision_threshold)
        - High velocity on one or more balls (> motion_speed_threshold)

        Returns:
            True if collision or high-motion detected, False otherwise
        """
        if len(self.tracks) < 2:
            return False

        # Check for ball proximity (potential collision)
        confirmed_tracks = [t for t in self.tracks if t.state == TrackState.CONFIRMED]

        for i, track_a in enumerate(confirmed_tracks):
            pos_a = track_a.kalman_filter.get_position()
            speed_a = track_a.kalman_filter.get_speed()

            # High-speed motion detected
            if speed_a > self.motion_speed_threshold:
                return True

            # Check proximity to other balls
            for track_b in confirmed_tracks[i + 1 :]:
                pos_b = track_b.kalman_filter.get_position()

                distance = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )

                # Balls are very close (likely collision or about to collide)
                if distance < self.collision_threshold:
                    speed_b = track_b.kalman_filter.get_speed()
                    thresholds = self.config.get("thresholds", {})
                    collision_speed = thresholds.get("collision_speed", 3.0)
                    # If at least one ball is moving, this is a collision event
                    if speed_a > collision_speed or speed_b > collision_speed:
                        return True

        return False

    def _get_tracked_balls(self) -> list[Ball]:
        """Get current tracked balls with ghost ball filtering.

        Filters out TENTATIVE tracks to prevent ghost balls from appearing.
        During collision events, applies even stricter filtering.
        """
        tracked_balls = []
        tentative_filtered = 0

        for track in self.tracks:
            # Always return CONFIRMED and LOST tracks
            if track.state == TrackState.CONFIRMED or track.state == TrackState.LOST:
                ball = track.get_current_ball()
                tracked_balls.append(ball)
            # Only return TENTATIVE tracks if explicitly enabled (for debugging)
            elif track.state == TrackState.TENTATIVE and self.return_tentative_tracks:
                # During collision, require higher confirmation threshold
                min_detections = (
                    self.min_hits_during_collision
                    if self.collision_detected
                    else self.min_hits
                )

                if track.detection_count >= min_detections:
                    ball = track.get_current_ball()
                    tracked_balls.append(ball)
                else:
                    tentative_filtered += 1
            else:
                tentative_filtered += 1

        # Track ghost ball filtering statistics
        if tentative_filtered > 0:
            self.track_statistics["ghost_balls_filtered"] += tentative_filtered

        return tracked_balls

    def predict_positions(self, time_delta: float) -> dict[int, tuple[float, float]]:
        """Predict object positions for next frame.

        Args:
            time_delta: Time delta for prediction

        Returns:
            Dictionary mapping track IDs to predicted positions
        """
        if (
            not math.isfinite(time_delta)
            or time_delta < 0.0
            or time_delta > self.max_prediction_horizon_seconds
        ):
            return {}

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
        if not math.isfinite(dt) or dt <= 0.0 or time_steps <= 0:
            return []
        bounded_steps = min(
            int(time_steps),
            int(self.max_prediction_horizon_seconds / dt),
        )
        if bounded_steps <= 0:
            return []

        for track in self.tracks:
            if track.track_id == track_id and track.is_valid():
                trajectory = track.kalman_filter.predict_trajectory(bounded_steps, dt)
                return list(trajectory)

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
        ball_type_counts: dict[str, int] = defaultdict(int)
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED:
                # Handle both enum and string ball types
                ball_type_value = (
                    track.ball_type.value
                    if hasattr(track.ball_type, "value")
                    else str(track.ball_type)
                )
                ball_type_counts[ball_type_value] += 1
        stats["ball_type_distribution"] = dict(ball_type_counts)  # type: ignore[assignment]

        return stats

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_number = 0
        self.last_frame_time = None
        self._last_observed_frame_number = None
        self._last_observation_timestamp = None
        self.track_history.clear()
        self.ball_type_tracks.clear()
        self.track_statistics = {
            "total_tracks_created": 0,
            "total_tracks_deleted": 0,
            "current_active_tracks": 0,
            "average_track_length": 0.0,
            "ghost_balls_filtered": 0,
            "collision_frames": 0,
            "stale_observations_rejected": 0,
            "tracks_expired_after_gap": 0,
        }
