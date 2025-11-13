"""Common models shared across backend modules.

This module provides the unified Ball class that consolidates all ball
representations across vision, tracking, and core modules.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from backend.core.coordinates import Vector2D
from backend.vision.models import BallType

if TYPE_CHECKING:
    from backend.core.models import BallState


@dataclass
class Ball:
    """Unified ball model with coordinate metadata and tracking information.

    This class consolidates all ball representations used across vision,
    tracking, and core modules. It includes:
    - Detection data (position, radius, confidence)
    - Ball identification (type, number)
    - Motion data (velocity, acceleration, movement state)
    - Tracking metadata (ID, age, history)
    - Physical properties (color, occlusion)
    - Coordinate space metadata (prevents double-scaling bugs)

    All position coordinates use tuple[float, float] format (not Vector2D)
    to maintain compatibility with existing vision and tracking code.

    Coordinate Space Management:
        The coordinate_space and source_resolution fields prevent double-scaling
        bugs by tracking the coordinate system of position data:

        - coordinate_space: "4k_canonical", "pixel", or "world"
        - source_resolution: (width, height) of the original camera frame

        These fields enable safe coordinate conversions and ensure consistency
        when passing data between modules with different coordinate systems.

    Attributes:
        position: (x, y) coordinates in pixel space
        radius: Ball radius in pixels
        ball_type: BallType enum value (CUE, SOLID_1, etc.)
        number: Ball number (1-15), None for cue ball
        confidence: Detection confidence score [0.0-1.0]
        velocity: (vx, vy) velocity in pixels/second
        acceleration: (ax, ay) acceleration in pixels/second²
        is_moving: Whether the ball is currently moving
        track_id: Unique tracking identifier
        last_seen: Timestamp of last detection (seconds since epoch)
        age: Number of frames this ball has been tracked
        hit_count: Number of times this ball was hit
        color_hsv: Average HSV color tuple (h, s, v)
        occlusion_state: Occlusion level [0.0=visible, 1.0=occluded]
        predicted_position: Predicted next position (x, y)
        position_history: List of recent positions for tracking
        coordinate_space: Coordinate system identifier
        source_resolution: Original frame resolution (width, height)

    Example:
        >>> # Create ball from vision detection at 1080p
        >>> ball = Ball(
        ...     position=(960.0, 540.0),
        ...     radius=18.0,
        ...     ball_type=BallType.CUE,
        ...     coordinate_space="pixel",
        ...     source_resolution=(1920, 1080)
        ... )
        >>> # Convert to 4K canonical coordinates
        >>> ball_4k = ball.to_4k_canonical()
        >>> ball_4k.position
        (1920.0, 1080.0)
        >>> ball_4k.coordinate_space
        '4k_canonical'
    """

    # Core detection data
    position: tuple[float, float]  # (x, y) in pixel coordinates
    radius: float  # pixels
    ball_type: BallType

    # Ball identification
    number: Optional[int] = None  # 1-15 for numbered balls, None for cue ball
    confidence: float = 0.0  # 0.0-1.0 detection confidence

    # Motion data
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixels/second
    acceleration: tuple[float, float] = (0.0, 0.0)  # (ax, ay) pixels/second²
    is_moving: bool = False

    # Tracking information
    track_id: Optional[int] = None  # Unique tracking ID
    last_seen: float = field(default_factory=time.time)
    age: int = 0  # Number of frames tracked
    hit_count: int = 0  # Number of hits detected

    # Physical properties
    color_hsv: Optional[tuple[int, int, int]] = None  # Average HSV color
    occlusion_state: float = 0.0  # 0.0 = fully visible, 1.0 = fully occluded

    # Prediction and filtering
    predicted_position: Optional[tuple[float, float]] = None
    position_history: list[tuple[float, float]] = field(default_factory=list)

    # Coordinate space metadata - critical for preventing double-scaling bugs
    coordinate_space: str = "4k_canonical"  # "4k_canonical", "pixel", or "world"
    source_resolution: Optional[tuple[int, int]] = None  # (width, height)

    def __post_init__(self) -> None:
        """Validate ball state and populate derived fields.

        Raises:
            ValueError: If validation fails
        """
        # Populate number from ball_type if not explicitly set
        if self.number is None and hasattr(self.ball_type, "number"):
            self.number = self.ball_type.number

        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

        # Validate occlusion state
        if not 0.0 <= self.occlusion_state <= 1.0:
            raise ValueError(
                f"Occlusion state must be in [0.0, 1.0], got {self.occlusion_state}"
            )

        # Validate radius
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

        # Validate coordinate space
        valid_spaces = {"4k_canonical", "pixel", "world"}
        if self.coordinate_space not in valid_spaces:
            raise ValueError(
                f"Coordinate space must be one of {valid_spaces}, got {self.coordinate_space}"
            )

        # Validate source_resolution if coordinate_space is "pixel"
        if self.coordinate_space == "pixel" and self.source_resolution is None:
            raise ValueError(
                "source_resolution is required when coordinate_space is 'pixel'"
            )

    # =========================================================================
    # Position History Management
    # =========================================================================

    def update_history(self, max_history: int = 10) -> None:
        """Update position history, keeping only recent positions.

        Args:
            max_history: Maximum number of historical positions to retain
        """
        self.position_history.append(self.position)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)

    # =========================================================================
    # Coordinate Conversion Methods
    # =========================================================================

    def to_4k_canonical(self) -> "Ball":
        """Convert ball to 4K canonical coordinates (3840x2160).

        This method converts the ball's position, radius, velocity, and
        acceleration to 4K canonical space. It's idempotent - calling it
        on a ball already in 4K canonical space returns a copy unchanged.

        Returns:
            New Ball instance with coordinates in 4K canonical space

        Raises:
            ValueError: If coordinate conversion is not possible

        Example:
            >>> # Ball at 1080p
            >>> ball = Ball(
            ...     position=(960.0, 540.0),
            ...     radius=18.0,
            ...     ball_type=BallType.CUE,
            ...     velocity=(10.0, 5.0),
            ...     coordinate_space="pixel",
            ...     source_resolution=(1920, 1080)
            ... )
            >>> ball_4k = ball.to_4k_canonical()
            >>> ball_4k.position
            (1920.0, 1080.0)
            >>> ball_4k.radius
            36.0
            >>> ball_4k.velocity
            (20.0, 10.0)
        """
        # Already in 4K canonical space
        if self.coordinate_space == "4k_canonical":
            return Ball(
                position=self.position,
                radius=self.radius,
                ball_type=self.ball_type,
                number=self.number,
                confidence=self.confidence,
                velocity=self.velocity,
                acceleration=self.acceleration,
                is_moving=self.is_moving,
                track_id=self.track_id,
                last_seen=self.last_seen,
                age=self.age,
                hit_count=self.hit_count,
                color_hsv=self.color_hsv,
                occlusion_state=self.occlusion_state,
                predicted_position=self.predicted_position,
                position_history=self.position_history.copy(),
                coordinate_space="4k_canonical",
                source_resolution=None,  # 4K canonical doesn't need source resolution
            )

        # Convert from pixel space
        if self.coordinate_space == "pixel":
            if self.source_resolution is None:
                raise ValueError(
                    "source_resolution required for pixel to 4K conversion"
                )

            # Calculate scale factors
            scale_x = 3840.0 / self.source_resolution[0]
            scale_y = 2160.0 / self.source_resolution[1]

            # Scale position
            new_position = (
                self.position[0] * scale_x,
                self.position[1] * scale_y,
            )

            # Scale radius (use average of x/y scales)
            scale_avg = (scale_x + scale_y) / 2.0
            new_radius = self.radius * scale_avg

            # Scale velocity
            new_velocity = (
                self.velocity[0] * scale_x,
                self.velocity[1] * scale_y,
            )

            # Scale acceleration
            new_acceleration = (
                self.acceleration[0] * scale_x,
                self.acceleration[1] * scale_y,
            )

            # Scale predicted position if present
            new_predicted_position = None
            if self.predicted_position is not None:
                new_predicted_position = (
                    self.predicted_position[0] * scale_x,
                    self.predicted_position[1] * scale_y,
                )

            # Scale position history
            new_position_history = [
                (pos[0] * scale_x, pos[1] * scale_y) for pos in self.position_history
            ]

            return Ball(
                position=new_position,
                radius=new_radius,
                ball_type=self.ball_type,
                number=self.number,
                confidence=self.confidence,
                velocity=new_velocity,
                acceleration=new_acceleration,
                is_moving=self.is_moving,
                track_id=self.track_id,
                last_seen=self.last_seen,
                age=self.age,
                hit_count=self.hit_count,
                color_hsv=self.color_hsv,
                occlusion_state=self.occlusion_state,
                predicted_position=new_predicted_position,
                position_history=new_position_history,
                coordinate_space="4k_canonical",
                source_resolution=None,
            )

        # World coordinates not supported for this conversion
        raise ValueError(
            f"Cannot convert from coordinate_space '{self.coordinate_space}' to 4K canonical"
        )

    def to_world_coordinates(
        self, homography_matrix: Optional[NDArray[np.float64]] = None
    ) -> Optional[tuple[float, float]]:
        """Convert ball position to world coordinates using homography.

        Args:
            homography_matrix: 3x3 homography matrix for perspective transformation.
                             If None, returns None.

        Returns:
            Position in world coordinates (meters), or None if conversion unavailable.

        Example:
            >>> import numpy as np
            >>> ball = Ball(
            ...     position=(1920.0, 1080.0),
            ...     radius=36.0,
            ...     ball_type=BallType.CUE
            ... )
            >>> # Example homography (identity for simplicity)
            >>> H = np.eye(3)
            >>> world_pos = ball.to_world_coordinates(H)
        """
        if homography_matrix is None or self.coordinate_space == "world":
            return None

        # Apply homography transformation
        point = np.array([[self.position[0], self.position[1]]], dtype=np.float32)
        transformed = np.dot(homography_matrix, np.append(point[0], 1.0))
        transformed = transformed / transformed[2]  # Normalize by w component

        return (float(transformed[0]), float(transformed[1]))

    # =========================================================================
    # Conversion to Core BallState
    # =========================================================================

    def to_ball_state(
        self,
        ball_id: Optional[str] = None,
        mass: float = 0.17,
        spin: Optional[Vector2D] = None,
    ) -> "BallState":
        """Convert to core.models.BallState for physics simulation.

        This method converts the vision/tracking Ball to the core BallState
        format used by the physics engine. Coordinates are converted to 4K
        canonical space if not already.

        Args:
            ball_id: Unique identifier for the ball. If None, uses f"ball_{number}"
                    or "cue_ball" for the cue ball.
            mass: Ball mass in kg (defaults to standard pool ball: 0.17 kg)
            spin: Ball spin vector. If None, creates zero spin vector.

        Returns:
            BallState instance ready for physics simulation

        Example:
            >>> ball = Ball(
            ...     position=(1920.0, 1080.0),
            ...     radius=36.0,
            ...     ball_type=BallType.SOLID_1,
            ...     velocity=(100.0, 50.0)
            ... )
            >>> ball_state = ball.to_ball_state()
            >>> ball_state.id
            'ball_1'
            >>> ball_state.position.x
            1920.0
        """
        # Import here to avoid circular dependency
        from backend.core.models import BallState

        # Convert to 4K canonical if needed
        ball_4k = (
            self if self.coordinate_space == "4k_canonical" else self.to_4k_canonical()
        )

        # Generate ball ID if not provided
        if ball_id is None:
            if self.ball_type == BallType.CUE:
                ball_id = "cue_ball"
            elif self.number is not None:
                ball_id = f"ball_{self.number}"
            else:
                ball_id = f"ball_{self.ball_type.value}"

        # Create Vector2D position in 4K canonical space
        position = Vector2D.from_4k(ball_4k.position[0], ball_4k.position[1])

        # Create Vector2D velocity
        velocity = Vector2D.from_4k(ball_4k.velocity[0], ball_4k.velocity[1])

        # Use provided spin or create zero spin
        if spin is None:
            spin = Vector2D.zero()

        return BallState(
            id=ball_id,
            position=position,
            velocity=velocity,
            radius=ball_4k.radius,
            mass=mass,
            spin=spin,
            is_cue_ball=(self.ball_type == BallType.CUE),
            is_pocketed=False,
            number=self.number,
            confidence=self.confidence,
            last_update=self.last_seen,
        )

    # =========================================================================
    # Display and Utility Methods
    # =========================================================================

    @property
    def display_name(self) -> str:
        """Get human-readable display name for the ball.

        Returns:
            Display name string

        Example:
            >>> ball = Ball(
            ...     position=(0, 0),
            ...     radius=36,
            ...     ball_type=BallType.SOLID_1
            ... )
            >>> ball.display_name
            '1'
            >>> cue = Ball(
            ...     position=(0, 0),
            ...     radius=36,
            ...     ball_type=BallType.CUE
            ... )
            >>> cue.display_name
            'Cue Ball'
        """
        if self.number is not None:
            return str(self.number)
        elif self.ball_type == BallType.CUE:
            return "Cue Ball"
        elif self.ball_type == BallType.EIGHT:
            return "8"
        else:
            return self.ball_type.value.capitalize()
