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
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from backend.common.models import Ball

# =============================================================================
# Detection Object Types
# =============================================================================


class BallType(Enum):
    """Ball type classification for all 16 pool balls.

    Supports full ball identification:
    - CUE: White cue ball (no number)
    - SOLID_1 through SOLID_7: Solid colored balls (1-7)
    - EIGHT: Black 8-ball
    - STRIPE_9 through STRIPE_15: Striped balls (9-15)
    - UNKNOWN: Classification failed
    """

    CUE = "cue"
    SOLID_1 = "solid_1"  # Yellow
    SOLID_2 = "solid_2"  # Blue
    SOLID_3 = "solid_3"  # Red
    SOLID_4 = "solid_4"  # Purple
    SOLID_5 = "solid_5"  # Orange
    SOLID_6 = "solid_6"  # Green
    SOLID_7 = "solid_7"  # Maroon
    EIGHT = "eight"  # Black
    STRIPE_9 = "stripe_9"  # Yellow stripe
    STRIPE_10 = "stripe_10"  # Blue stripe
    STRIPE_11 = "stripe_11"  # Red stripe
    STRIPE_12 = "stripe_12"  # Purple stripe
    STRIPE_13 = "stripe_13"  # Orange stripe
    STRIPE_14 = "stripe_14"  # Green stripe
    STRIPE_15 = "stripe_15"  # Maroon stripe
    UNKNOWN = "unknown"

    @property
    def number(self) -> Optional[int]:
        """Get the ball number (1-15), or None for cue ball."""
        if self == BallType.CUE or self == BallType.UNKNOWN:
            return None
        if self == BallType.EIGHT:
            return 8
        # Extract number from enum name (e.g., "SOLID_1" -> 1)
        name_parts = self.name.split("_")
        if len(name_parts) > 1:
            return int(name_parts[1])
        return None

    @property
    def is_solid(self) -> bool:
        """Check if this is a solid ball (1-7)."""
        return "SOLID" in self.name

    @property
    def is_stripe(self) -> bool:
        """Check if this is a stripe ball (9-15)."""
        return "STRIPE" in self.name

    @property
    def color_name(self) -> str:
        """Get the color name of the ball."""
        color_map = {
            BallType.CUE: "white",
            BallType.SOLID_1: "yellow",
            BallType.SOLID_2: "blue",
            BallType.SOLID_3: "red",
            BallType.SOLID_4: "purple",
            BallType.SOLID_5: "orange",
            BallType.SOLID_6: "green",
            BallType.SOLID_7: "maroon",
            BallType.EIGHT: "black",
            BallType.STRIPE_9: "yellow",
            BallType.STRIPE_10: "blue",
            BallType.STRIPE_11: "red",
            BallType.STRIPE_12: "purple",
            BallType.STRIPE_13: "orange",
            BallType.STRIPE_14: "green",
            BallType.STRIPE_15: "maroon",
        }
        return color_map.get(self, "unknown")


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
# NOTE: Ball class has been moved to backend.common.models
# Import it from there: from backend.common.models import Ball


@dataclass
class Pocket:
    """Pool table pocket information.

    Position, corners, and radius use the containing Table's coordinate space
    and source-resolution metadata. This avoids duplicating scale metadata on
    each pocket while ensuring pockets follow the exact same conversion as the
    table boundary that contains them.
    """

    position: tuple[float, float]  # Center position in pixel coordinates
    pocket_type: PocketType
    radius: float  # Effective radius in pixels
    corners: list[tuple[float, float]]  # Corner points in pixel coordinates


@dataclass
class CueStick:
    """Detected cue stick information with state tracking.

    Represents the cue stick with position, orientation, state, and motion data.

    Position vectors and pixel extents use ``coordinate_space``. Pixel-space
    cues carry the actual source frame resolution; canonical cues retain that
    resolution as provenance. The vision normalization path transforms tip,
    butt, shaft, aiming geometry, motion vectors, length, and width together.
    """

    tip_position: tuple[float, float]  # Tip position in coordinate_space
    angle: float  # degrees from horizontal
    length: float  # visible length in coordinate_space units
    confidence: float = 0.0  # 0.0-1.0 detection confidence

    # State information
    state: CueState = CueState.HIDDEN

    # Motion tracking
    tip_velocity: tuple[float, float] = (
        0.0,
        0.0,
    )  # in coordinate_space units/second
    angular_velocity: float = 0.0  # degrees/second

    # Geometry
    butt_position: tuple[float, float] = (
        0.0,
        0.0,
    )  # Butt (back end) position in coordinate_space
    shaft_points: list[tuple[float, float]] = field(
        default_factory=list
    )  # Points along the shaft in coordinate_space
    width: float = 0.0  # Estimated cue width in coordinate_space units

    # Interaction
    target_ball_id: Optional[int] = None  # ID of ball being aimed at
    predicted_contact_point: Optional[tuple[float, float]] = None  # in coordinate_space
    aiming_line: Optional[list[tuple[float, float]]] = (
        None  # Line from cue to target in coordinate_space
    )

    # Coordinate space metadata
    coordinate_space: str = "4k_canonical"  # "4k_canonical", "pixel", or "world"
    source_resolution: Optional[tuple[int, int]] = (
        None  # (width, height) of source frame
    )

    def to_world_coordinates(
        self, homography_matrix: Optional[NDArray[np.float64]] = None
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """Convert cue stick tip and butt positions to world coordinates.

        Args:
            homography_matrix: 3x3 homography matrix for perspective transformation.
                             If None, returns None.

        Returns:
            Tuple of (tip_position, butt_position) in world coordinates (meters),
            or None if conversion unavailable.
        """
        if homography_matrix is None or self.coordinate_space == "world":
            return None

        # Apply homography to tip position
        tip_point = np.array(
            [[self.tip_position[0], self.tip_position[1]]], dtype=np.float32
        )
        tip_transformed = np.dot(homography_matrix, np.append(tip_point[0], 1.0))
        tip_transformed = tip_transformed / tip_transformed[2]

        # Apply homography to butt position
        butt_point = np.array(
            [[self.butt_position[0], self.butt_position[1]]], dtype=np.float32
        )
        butt_transformed = np.dot(homography_matrix, np.append(butt_point[0], 1.0))
        butt_transformed = butt_transformed / butt_transformed[2]

        return (
            (float(tip_transformed[0]), float(tip_transformed[1])),
            (float(butt_transformed[0]), float(butt_transformed[1])),
        )


@dataclass
class Table:
    """Detected table information with comprehensive geometry.

    Represents the pool table with boundaries, pockets, and surface properties.

    Corners, rails, playing area, pixel dimensions, and nested pocket geometry
    share ``coordinate_space`` and ``source_resolution``. For pixel tables the
    source resolution identifies the exact frame in which every boundary and
    pixel extent is expressed; canonical tables carry it as provenance.
    """

    corners: list[tuple[float, float]]  # 4 corners in clockwise order, coordinate_space
    pockets: list[Pocket]  # All 6 pockets
    width: float  # width extent in coordinate_space units
    height: float  # height extent in coordinate_space units
    surface_color: tuple[int, int, int]  # Average HSV color

    # Geometric properties
    rails: list[list[tuple[float, float]]] = field(
        default_factory=list
    )  # Rail segments in coordinate_space
    playing_area: list[tuple[float, float]] = field(
        default_factory=list
    )  # Inner boundary in coordinate_space

    # Calibration data
    perspective_transform: Optional[NDArray[np.float64]] = None  # 3x3 transform matrix
    real_world_dimensions: Optional[tuple[float, float]] = (
        None  # (width, height) in meters
    )
    pixels_per_meter: Optional[float] = None

    # Table characteristics
    felt_condition: float = 1.0  # 0.0 = poor, 1.0 = excellent
    lighting_quality: float = 1.0  # 0.0 = poor, 1.0 = excellent

    # Coordinate space metadata
    coordinate_space: str = "4k_canonical"  # "4k_canonical", "pixel", or "world"
    source_resolution: Optional[tuple[int, int]] = (
        None  # (width, height) of source frame
    )

    def get_pocket_by_type(self, pocket_type: PocketType) -> list[Pocket]:
        """Get all pockets of specified type."""
        return [p for p in self.pockets if p.pocket_type == pocket_type]

    def nearest_pocket(self, position: tuple[float, float]) -> Optional[Pocket]:
        """Find nearest pocket to given position.

        Args:
            position: Position in pixel coordinates (x, y)

        Returns:
            Nearest pocket, or None if no pockets available.
        """
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

    def to_world_coordinates(
        self, point: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert a point from pixel coordinates to world coordinates.

        Uses the table's perspective_transform if available.

        Args:
            point: Point in pixel coordinates (x, y)

        Returns:
            Point in world coordinates (meters), or None if conversion unavailable.
        """
        if self.perspective_transform is None or self.coordinate_space == "world":
            return None

        # Apply homography transformation
        point_array = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed = np.dot(self.perspective_transform, np.append(point_array[0], 1.0))
        transformed = transformed / transformed[2]  # Normalize by w component

        return (float(transformed[0]), float(transformed[1]))


# =============================================================================
# Processing Results
# =============================================================================


@dataclass
class CameraFrame:
    """Container for camera frame with metadata.

    Wraps a numpy frame array with metadata for passing through the vision pipeline.
    Provides convenient properties for frame dimensions and manipulation.

    All pixel coordinates in this frame are in the coordinate space defined by
    width x height resolution.
    """

    frame: NDArray[np.uint8]  # The actual image data
    timestamp: float
    frame_id: int
    width: int
    height: int

    # Optional resolution metadata for coordinate space tracking
    original_resolution: Optional[tuple[int, int]] = (
        None  # Original capture resolution if frame was resized
    )
    coordinate_space: str = "pixel"  # "pixel" or other coordinate systems

    @property
    def aspect_ratio(self) -> float:
        """Calculate frame aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def center(self) -> tuple[int, int]:
        """Get frame center coordinates in pixel space."""
        return (self.width // 2, self.height // 2)

    @property
    def size(self) -> tuple[int, int]:
        """Get frame dimensions as (width, height) tuple."""
        return (self.width, self.height)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get current frame resolution.

        Returns:
            Current resolution as (width, height).
        """
        return (self.width, self.height)

    def copy(self) -> "CameraFrame":
        """Create a deep copy of the camera frame."""
        return CameraFrame(
            frame=self.frame.copy(),
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            width=self.width,
            height=self.height,
            original_resolution=self.original_resolution,
            coordinate_space=self.coordinate_space,
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
    balls: list["Ball"]
    cue: Optional[CueStick]
    table: Optional[Table]
    statistics: FrameStatistics

    # Frame metadata
    frame_size: tuple[int, int] = (0, 0)  # (width, height)
    original_frame: Optional[NDArray[np.uint8]] = None  # Original captured frame
    processed_frame: Optional[NDArray[np.uint8]] = None  # Frame with overlays

    # Processing flags
    # Processing failures or unavailable model components make the result
    # incomplete. Intentional scheduler skips may still yield a complete result.
    is_complete: bool = True
    # True only for components that failed; unavailable components are
    # incomplete but are not represented as runtime errors.
    has_errors: bool = False
    # Safe component/status/category codes only; never exception text.
    error_messages: list[str] = field(default_factory=list)

    # Game state inference
    game_state: str = "unknown"  # "break", "aiming", "shooting", "moving", "stopped"
    shot_detected: bool = False
    balls_in_motion: int = 0

    # Source-frame metadata. ``source_timestamp`` is the shared-memory
    # publication time when available; camera exposure time is not currently
    # supplied by the capture contract.
    source_frame_number: Optional[int] = None
    source_timestamp: Optional[float] = None
    source_timestamp_kind: Optional[str] = None
    source_process_id: Optional[int] = None
    source_publication_monotonic_ns: Optional[int] = None
    # Elapsed age since known publication/arrival time. Shared-memory records
    # use their monotonic publication timestamp; legacy records without it use
    # a wall-clock estimate. This is not camera exposure age or a moving-ball
    # observation gap.
    frame_publication_age_seconds: Optional[float] = None

    # Opt-in phase scheduler contract. ``None`` eligibility means no
    # phase-aware eligibility decision was made for this observation.
    phase_aware: bool = False
    phase: Optional[str] = None
    vision_phase: Optional[str] = None
    phase_reason: Optional[str] = None
    map_revision: Optional[int] = None
    map_trusted: Optional[bool] = None
    moving_ball_count: Optional[int] = None
    aim_calculation_eligible: Optional[bool] = None
    calibration_revision: Optional[str] = None

    # Per-component observation times use the monotonic scheduler clock.
    # Ball and cue status is "observed", "skipped", "failed", or
    # "unavailable". Motion monitoring may additionally report "stale".
    ball_detection_status: Optional[str] = None
    # Bounded detector stage category only (inference, decoding,
    # classification, or unknown); never backend exception text.
    ball_detection_failure_stage: Optional[str] = None
    cue_detection_status: Optional[str] = None
    motion_monitor_status: Optional[str] = None
    # The positions carried in ``balls`` may be retained from an earlier
    # successful inference while the current ball job is skipped or fails.
    # Consumers should append movement traces only when this is "observed".
    ball_positions_status: Optional[str] = None
    ball_observation_source_timestamp: Optional[float] = None
    ball_observation_source_frame_number: Optional[int] = None
    ball_observation_source_timestamp_kind: Optional[str] = None
    ball_observation_timestamp: Optional[float] = None
    maximum_ball_observation_gap_seconds: Optional[float] = None
    actual_path_history_max_points: Optional[int] = None
    ball_detection_view: Optional[str] = None
    cue_observation_timestamp: Optional[float] = None
    motion_observation_timestamp: Optional[float] = None
    component_observation_status: dict[str, str] = field(default_factory=dict)

    def get_balls_by_type(self, ball_type: BallType) -> list["Ball"]:
        """Get all balls of specified type."""
        return [ball for ball in self.balls if ball.ball_type == ball_type]

    def get_cue_ball(self) -> Optional["Ball"]:
        """Get the cue ball if detected."""
        cue_balls = self.get_balls_by_type(BallType.CUE)
        return cue_balls[0] if cue_balls else None

    def get_moving_balls(self) -> list["Ball"]:
        """Get all balls currently in motion."""
        return [ball for ball in self.balls if ball.is_moving]

    def get_stationary_balls(self) -> list["Ball"]:
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
    """Detected shot event with analysis.

    All position coordinates are in pixel space unless otherwise specified.
    """

    shot_id: int
    timestamp: float
    cue_ball_position: tuple[float, float]  # in pixel coordinates
    target_ball_position: Optional[tuple[float, float]]  # in pixel coordinates
    cue_angle: float
    estimated_force: float
    contact_point: Optional[tuple[float, float]]  # in pixel coordinates

    # Results
    balls_potted: list[int] = field(default_factory=list)
    final_positions: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )  # in pixel coordinates
    shot_quality: float = 0.0  # 0.0-1.0 quality assessment
