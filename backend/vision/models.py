"""Vision module data models.

Provides comprehensive data structures for vision processing including:
- Ball detection and tracking
- Cue stick detection
- Table detection
- Camera capture information
- Processing statistics
- Calibration data
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Detection Object Types
# =============================================================================


class BallType(Enum):
    """Ball type classification."""

    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


class PocketType(Enum):
    """Pool table pocket types."""

    CORNER = "corner"
    SIDE = "side"


class CueState(Enum):
    """Cue stick states."""

    HIDDEN = "hidden"
    AIMING = "aiming"
    STRIKING = "striking"
    RETRACTING = "retracting"


# =============================================================================
# Detection Objects
# =============================================================================


@dataclass
class Ball:
    """Detected ball information with tracking data.

    Represents a single pool ball with position, type, movement, and tracking metadata.
    """

    position: tuple[float, float]  # (x, y) in pixels
    radius: float  # pixels
    ball_type: BallType
    number: Optional[int] = None  # 1-15 for numbered balls, None for cue/8-ball
    confidence: float = 0.0  # 0.0-1.0 detection confidence
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels/second
    acceleration: tuple[float, float] = (0.0, 0.0)  # (ax, ay) pixels/second²
    is_moving: bool = False

    # Tracking information
    track_id: Optional[int] = None  # Unique tracking ID
    last_seen: float = field(default_factory=time.time)
    age: int = 0  # Number of frames this ball has been tracked
    hit_count: int = 0  # Number of times this ball was hit

    # Physical properties
    color_hsv: Optional[tuple[int, int, int]] = None  # Average HSV color
    occlusion_state: float = 0.0  # 0.0 = fully visible, 1.0 = fully occluded

    # Prediction and filtering
    predicted_position: Optional[tuple[float, float]] = None
    position_history: list[tuple[float, float]] = field(default_factory=list)

    def update_history(self, max_history: int = 10) -> None:
        """Update position history, keeping only recent positions."""
        self.position_history.append(self.position)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)


@dataclass
class Pocket:
    """Pool table pocket information."""

    position: tuple[float, float]  # Center position in pixels
    pocket_type: PocketType
    radius: float  # Effective radius in pixels
    corners: list[tuple[float, float]]  # Corner points defining pocket shape


@dataclass
class CueStick:
    """Detected cue stick information with state tracking.

    Represents the cue stick with position, orientation, state, and motion data.
    """

    tip_position: tuple[float, float]  # Tip position in pixels
    angle: float  # degrees from horizontal
    length: float  # visible length in pixels
    confidence: float = 0.0  # 0.0-1.0 detection confidence

    # State information
    state: CueState = CueState.HIDDEN
    is_aiming: bool = False  # Backward compatibility

    # Motion tracking
    tip_velocity: tuple[float, float] = (0.0, 0.0)  # pixels/second
    angular_velocity: float = 0.0  # degrees/second

    # Geometry
    shaft_points: list[tuple[float, float]] = field(
        default_factory=list
    )  # Points along the shaft
    width: float = 0.0  # Estimated cue width in pixels

    # Interaction
    target_ball_id: Optional[int] = None  # ID of ball being aimed at
    predicted_contact_point: Optional[tuple[float, float]] = None
    aiming_line: Optional[list[tuple[float, float]]] = None  # Line from cue to target


@dataclass
class Table:
    """Detected table information with comprehensive geometry.

    Represents the pool table with boundaries, pockets, and surface properties.
    """

    corners: list[tuple[float, float]]  # 4 corners in clockwise order
    pockets: list[Pocket]  # All 6 pockets
    width: float  # pixels
    height: float  # pixels
    surface_color: tuple[int, int, int]  # Average HSV color

    # Geometric properties
    rails: list[list[tuple[float, float]]] = field(
        default_factory=list
    )  # Rail segments
    playing_area: list[tuple[float, float]] = field(
        default_factory=list
    )  # Inner boundary

    # Calibration data
    perspective_transform: Optional[NDArray[np.float64]] = None  # 3x3 transform matrix
    real_world_dimensions: Optional[tuple[float, float]] = (
        None  # (width, height) in meters
    )
    pixels_per_meter: Optional[float] = None

    # Table characteristics
    felt_condition: float = 1.0  # 0.0 = poor, 1.0 = excellent
    lighting_quality: float = 1.0  # 0.0 = poor, 1.0 = excellent

    def get_pocket_by_type(self, pocket_type: PocketType) -> list[Pocket]:
        """Get all pockets of specified type."""
        return [p for p in self.pockets if p.pocket_type == pocket_type]

    def nearest_pocket(self, position: tuple[float, float]) -> Optional[Pocket]:
        """Find nearest pocket to given position."""
        if not self.pockets:
            return None

        min_dist = float("inf")
        nearest = None
        x, y = position

        for pocket in self.pockets:
            px, py = pocket.position
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = pocket

        return nearest


# =============================================================================
# Processing Results
# =============================================================================


@dataclass
class CameraFrame:
    """Container for camera frame with metadata.

    Wraps a numpy frame array with metadata for passing through the vision pipeline.
    Provides convenient properties for frame dimensions and manipulation.
    """

    frame: NDArray[np.uint8]  # The actual image data
    timestamp: float
    frame_id: int
    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        """Calculate frame aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def center(self) -> tuple[int, int]:
        """Get frame center coordinates."""
        return (self.width // 2, self.height // 2)

    @property
    def size(self) -> tuple[int, int]:
        """Get frame dimensions as (width, height) tuple."""
        return (self.width, self.height)

    def copy(self) -> "CameraFrame":
        """Create a deep copy of the camera frame."""
        return CameraFrame(
            frame=self.frame.copy(),
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            width=self.width,
            height=self.height,
        )


@dataclass
class FrameStatistics:
    """Frame processing performance statistics."""

    frame_number: int
    timestamp: float
    processing_time: float  # Total processing time in milliseconds

    # Component timing breakdown
    capture_time: float = 0.0
    preprocessing_time: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    postprocessing_time: float = 0.0

    # Detection counts
    balls_detected: int = 0
    balls_tracked: int = 0
    cue_detected: bool = False
    table_detected: bool = False

    # Quality metrics
    detection_confidence: float = 0.0  # Average detection confidence
    tracking_quality: float = 0.0  # Average tracking quality
    frame_quality: float = 0.0  # Overall frame quality (0.0-1.0)


@dataclass
class DetectionResult:
    """Complete frame detection results with comprehensive information.

    Contains all detected objects, processing statistics, and metadata for a single frame.
    """

    frame_number: int
    timestamp: float
    balls: list[Ball]
    cue: Optional[CueStick]
    table: Optional[Table]
    statistics: FrameStatistics

    # Frame metadata
    frame_size: tuple[int, int] = (0, 0)  # (width, height)
    original_frame: Optional[NDArray[np.uint8]] = None  # Original captured frame
    processed_frame: Optional[NDArray[np.uint8]] = None  # Frame with overlays

    # Processing flags
    is_complete: bool = True  # All detection components completed successfully
    has_errors: bool = False  # Processing errors occurred
    error_messages: list[str] = field(default_factory=list)

    # Game state inference
    game_state: str = "unknown"  # "break", "aiming", "shooting", "moving", "stopped"
    shot_detected: bool = False
    balls_in_motion: int = 0

    def get_balls_by_type(self, ball_type: BallType) -> list[Ball]:
        """Get all balls of specified type."""
        return [ball for ball in self.balls if ball.ball_type == ball_type]

    def get_cue_ball(self) -> Optional[Ball]:
        """Get the cue ball if detected."""
        cue_balls = self.get_balls_by_type(BallType.CUE)
        return cue_balls[0] if cue_balls else None

    def get_moving_balls(self) -> list[Ball]:
        """Get all balls currently in motion."""
        return [ball for ball in self.balls if ball.is_moving]

    def get_stationary_balls(self) -> list[Ball]:
        """Get all stationary balls."""
        return [ball for ball in self.balls if not ball.is_moving]


# =============================================================================
# Calibration Data
# =============================================================================


@dataclass
class CameraCalibration:
    """Camera intrinsic calibration parameters."""

    camera_matrix: NDArray[np.float64]  # 3x3 intrinsic matrix
    distortion_coefficients: NDArray[np.float64]  # Distortion parameters
    resolution: tuple[int, int]  # (width, height) used for calibration
    reprojection_error: float  # RMS reprojection error
    calibration_date: float = field(default_factory=time.time)


@dataclass
class ColorCalibration:
    """Color threshold calibration data."""

    table_color_range: tuple[
        tuple[int, int, int], tuple[int, int, int]
    ]  # (lower_hsv, upper_hsv)
    ball_color_ranges: dict[BallType, tuple[tuple[int, int, int], tuple[int, int, int]]]
    white_balance: tuple[float, float, float]  # RGB multipliers
    ambient_light_level: float  # 0.0-1.0
    calibration_date: float = field(default_factory=time.time)


@dataclass
class GeometricCalibration:
    """Geometric transformation calibration."""

    table_corners_pixel: list[tuple[float, float]]  # Table corners in image coordinates
    table_corners_world: list[tuple[float, float]]  # Table corners in world coordinates
    homography_matrix: NDArray[np.float64]  # 3x3 homography matrix
    inverse_homography: NDArray[np.float64]  # Inverse homography
    table_dimensions_real: tuple[float, float]  # Real world table size (meters)
    pixels_per_meter: float  # Scaling factor
    calibration_quality: float  # 0.0-1.0 quality score
    calibration_date: float = field(default_factory=time.time)


@dataclass
class CalibrationData:
    """Complete calibration dataset."""

    camera: Optional[CameraCalibration] = None
    colors: Optional[ColorCalibration] = None
    geometry: Optional[GeometricCalibration] = None

    def is_complete(self) -> bool:
        """Check if all calibration components are available."""
        return all([self.camera, self.colors, self.geometry])

    def is_valid(self) -> bool:
        """Check if calibration data is still valid (not too old)."""
        if not self.is_complete():
            return False

        current_time = time.time()
        max_age = 24 * 3600  # 24 hours

        # Check if calibration components exist before accessing their attributes
        if self.camera is None or self.colors is None or self.geometry is None:
            return False

        return all(
            [
                current_time - self.camera.calibration_date < max_age,
                current_time - self.colors.calibration_date < max_age,
                current_time - self.geometry.calibration_date < max_age,
            ]
        )


# =============================================================================
# Session and History
# =============================================================================


@dataclass
class DetectionSession:
    """Session-level detection tracking."""

    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_frames: int = 0
    total_shots: int = 0

    # Performance statistics
    average_fps: float = 0.0
    average_processing_time: float = 0.0
    detection_accuracy: float = 0.0

    # Error tracking
    total_errors: int = 0
    error_rate: float = 0.0

    # Configuration snapshot
    config_snapshot: Optional[dict[str, Any]] = None


@dataclass
class ShotEvent:
    """Detected shot event with analysis."""

    shot_id: int
    timestamp: float
    cue_ball_position: tuple[float, float]
    target_ball_position: Optional[tuple[float, float]]
    cue_angle: float
    estimated_force: float
    contact_point: Optional[tuple[float, float]]

    # Results
    balls_potted: list[int] = field(default_factory=list)
    final_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    shot_quality: float = 0.0  # 0.0-1.0 quality assessment
