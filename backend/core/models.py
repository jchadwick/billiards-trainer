"""Core data models and types for the billiards trainer system.

This module contains all the fundamental data structures that represent the game state,
physics objects, and analysis results. These models serve as the foundation for all
other core modules and provide the interface for data exchange between modules.

4K Coordinate System:
    This module uses the 4K-based Vector2D from coordinates.py. All positions are
    stored in 4K canonical pixels (3840×2160) with mandatory scale metadata for
    resolution tracking and conversion.

    Factory Methods for Creating Vectors:
        - Vector2D.from_4k(x, y) - Creates in 4K canonical coordinates (scale = 1.0)
        - Vector2D.from_resolution(x, y, resolution) - Creates from any resolution with auto-calculated scale

    Conversion Methods:
        - vector.to_4k_canonical() - Convert to 4K canonical coordinates
        - vector.to_resolution(target_resolution) - Convert to target resolution

    Example Usage:
        # Create position in 4K canonical
        pos = Vector2D.from_4k(1920.0, 1080.0)
        assert pos.scale == (1.0, 1.0)

        # Create from 1080p coordinates (auto-calculates scale = 2.0)
        pos_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        assert pos_1080p.scale == (2.0, 2.0)

        # Convert to 4K canonical
        pos_4k = pos_1080p.to_4k_canonical()
        assert (pos_4k.x, pos_4k.y) == (1920.0, 1080.0)
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

# Import Vector2D from coordinates module
try:
    # Try absolute import first (for when used as package)
    from backend.core.coordinates import Vector2D
except ImportError:
    # Fall back to relative import (for direct module usage)
    from .coordinates import Vector2D


class ShotType(Enum):
    """Types of billiards shots categorized by technique and strategy."""

    BREAK = "break"
    DIRECT = "direct"
    BANK = "bank"
    COMBINATION = "combination"
    SAFETY = "safety"
    MASSE = "masse"


class GameType(Enum):
    """Types of billiards games with different rules and objectives."""

    EIGHT_BALL = "8ball"
    NINE_BALL = "9ball"
    STRAIGHT_POOL = "straight"
    PRACTICE = "practice"


@dataclass
class BallState:
    """Complete ball state information.

    Represents all physical and logical properties of a billiards ball,
    including position, velocity, spin, and game-specific attributes.

    4K Coordinate System:
        Ball positions are stored using Vector2D with 4K canonical coordinates
        (3840×2160) and mandatory scale metadata for resolution tracking.

        Positions are created using Vector2D factory methods:
        - Vector2D.from_4k(x, y): Create in 4K canonical (scale = 1.0)
        - Vector2D.from_resolution(x, y, resolution): Create from any resolution

        Convert positions using Vector2D methods:
        - position.to_4k_canonical(): Convert to 4K canonical
        - position.to_resolution(target_resolution): Convert to target resolution
    """

    id: str
    position: Vector2D
    velocity: Vector2D = field(default_factory=Vector2D.zero)
    radius: float = (
        36.0  # Ball radius in 4K pixels (BALL_RADIUS_4K = 36 pixels, 72px diameter)
    )
    mass: float = 0.17  # kg, standard pool ball mass (physical property, not spatial)
    spin: Vector2D = field(default_factory=Vector2D.zero)  # Top/back/side spin
    is_cue_ball: bool = False
    is_pocketed: bool = False
    number: Optional[int] = None
    confidence: float = 1.0  # Detection confidence [0.0-1.0]
    last_update: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Validate ball state after initialization."""
        if self.radius <= 0:
            raise ValueError("Ball radius must be positive")
        if self.mass <= 0:
            raise ValueError("Ball mass must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        # Handle legacy last_seen field from older serialized data
        # Check in __dict__ to avoid AttributeError
        if "last_seen" in self.__dict__ and self.__dict__["last_seen"] is not None:
            self.last_update = self.__dict__["last_seen"]

    def kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the ball."""
        linear_ke = 0.5 * self.mass * self.velocity.magnitude_squared()
        # Rotational KE: 0.5 * I * ω², where I = (2/5) * m * r² for sphere
        moment_of_inertia = 0.4 * self.mass * self.radius**2
        angular_velocity = self.spin.magnitude() if self.spin is not None else 0.0
        rotational_ke = 0.5 * moment_of_inertia * angular_velocity**2
        return linear_ke + rotational_ke

    def is_moving(self, threshold: float = 0.001) -> bool:
        """Check if the ball is moving above the threshold velocity."""
        return self.velocity.magnitude() > threshold

    def is_spinning(self, threshold: float = 0.1) -> bool:
        """Check if the ball has significant spin."""
        return self.spin.magnitude() > threshold if self.spin is not None else False

    def distance_to(self, other: "BallState") -> float:
        """Calculate distance to another ball's center."""
        # Simple approach - always use raw distance calculation to avoid type issues
        dx = self.position.x - other.position.x
        dy = self.position.y - other.position.y
        return (dx * dx + dy * dy) ** 0.5

    def is_touching(self, other: "BallState", tolerance: float = 0.001) -> bool:
        """Check if this ball is touching another ball."""
        min_distance = self.radius + other.radius + tolerance
        return self.distance_to(other) <= min_distance

    def copy(self) -> "BallState":
        """Create a deep copy of the ball state.

        Preserves scale metadata from Vector2D instances.
        """
        # Copy position preserving scale metadata
        position = Vector2D(
            self.position.x,
            self.position.y,
            scale=self.position.scale,
        )

        # Copy velocity preserving scale metadata
        velocity = Vector2D(
            self.velocity.x,
            self.velocity.y,
            scale=self.velocity.scale,
        )

        # Copy spin preserving scale metadata
        spin = Vector2D(
            self.spin.x,
            self.spin.y,
            scale=self.spin.scale,
        )

        return BallState(
            id=self.id,
            position=position,
            velocity=velocity,
            radius=self.radius,
            mass=self.mass,
            spin=spin,
            is_cue_ball=self.is_cue_ball,
            is_pocketed=self.is_pocketed,
            number=self.number,
            confidence=self.confidence,
            last_update=self.last_update,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Includes coordinate space metadata when present.
        """
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "radius": self.radius,
            "mass": self.mass,
            "spin": self.spin.to_dict() if self.spin is not None else None,
            "is_cue_ball": self.is_cue_ball,
            "is_pocketed": self.is_pocketed,
            "number": self.number,
            "confidence": self.confidence,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BallState":
        """Create from dictionary.

        Automatically deserializes Vector2D with coordinate metadata when present.
        """
        # Parse position (Vector2D.from_dict handles metadata automatically)
        position = Vector2D.from_dict(data["position"])

        # Parse velocity
        velocity_data = data.get("velocity", {"x": 0, "y": 0})
        velocity = Vector2D.from_dict(velocity_data)

        # Parse spin
        spin_data = data.get("spin", {"x": 0, "y": 0})
        spin = Vector2D.from_dict(spin_data)

        return cls(
            id=data["id"],
            position=position,
            velocity=velocity,
            radius=data.get("radius", 36.0),  # Default to 4K pixels (BALL_RADIUS_4K)
            mass=data.get("mass", 0.17),
            spin=spin,
            is_cue_ball=data.get("is_cue_ball", False),
            is_pocketed=data.get("is_pocketed", False),
            number=data.get("number"),
            confidence=data.get("confidence", 1.0),
            last_update=data.get("last_update", datetime.now().timestamp()),
        )

    @classmethod
    def from_4k(
        cls, id: str, x: float, y: float, number: Optional[int] = None, **kwargs
    ) -> "BallState":
        """Create ball from 4K canonical coordinates.

        Args:
            id: Unique identifier for the ball
            x: X coordinate in 4K pixels (3840×2160)
            y: Y coordinate in 4K pixels
            number: Ball number (0-15, where 0 is cue ball)
            **kwargs: Additional ball properties (velocity, radius, etc.)

        Returns:
            BallState with position in 4K canonical space
        """
        position = Vector2D.from_4k(x, y)
        velocity = kwargs.pop("velocity", Vector2D.from_4k(0, 0))
        return cls(id=id, position=position, velocity=velocity, number=number, **kwargs)

    @classmethod
    def from_resolution(
        cls,
        id: str,
        x: float,
        y: float,
        resolution: tuple[int, int],
        number: Optional[int] = None,
        **kwargs,
    ) -> "BallState":
        """Create ball from any resolution coordinates.

        Args:
            id: Unique identifier for the ball
            x: X coordinate in source resolution
            y: Y coordinate in source resolution
            resolution: Source resolution as (width, height) tuple
            number: Ball number (0-15, where 0 is cue ball)
            **kwargs: Additional ball properties

        Returns:
            BallState with position scaled to 4K
        """
        position = Vector2D.from_resolution(x, y, resolution)
        velocity = kwargs.pop("velocity", Vector2D.from_4k(0, 0))
        return cls(id=id, position=position, velocity=velocity, number=number, **kwargs)


@dataclass
class TableState:
    """Pool table state information.

    Represents the physical properties of the billiards table including
    dimensions, pocket locations, and surface characteristics that affect
    ball physics.
    """

    width: float  # 4K pixels (3200) or legacy meters (2.54)
    height: float  # 4K pixels (1600) or legacy meters (1.27)
    pocket_positions: list[Vector2D]
    pocket_radius: float = (
        0.0635  # Standard pocket radius in meters (legacy) or pixels (4K)
    )
    cushion_elasticity: float = 0.85  # Coefficient of restitution
    surface_friction: float = 0.2  # Rolling friction coefficient
    surface_slope: float = 0.0  # Surface slope in degrees
    cushion_height: float = (
        0.064  # Height of cushions in meters (converted from rail_height)
    )
    playing_area_corners: Optional[list[Vector2D]] = (
        None  # Calibrated playing area corners (top-left, top-right, bottom-right, bottom-left)
    )

    def __post_init__(self) -> None:
        """Validate table state after initialization and handle legacy fields."""
        # Convert legacy mm measurements to meters if needed
        # NEW LOGIC: Don't convert 4K pixel values
        # 4K table dimensions are 3200×1600 (pixels)
        # Legacy meter values are 2.54×1.27 (< 10)
        # Legacy mm values are 2540×1270 (100-3000 range)
        # If EITHER dimension is >= 3000, assume we're in 4K pixel space (don't convert)
        # Otherwise, if in mm range (100-3000), convert to meters
        is_4k_pixels = self.width >= 3000 or self.height >= 1500

        if not is_4k_pixels:
            if 100 <= self.width < 3000:  # mm range
                self.width = self.width / 1000.0
            if 100 <= self.height < 3000:  # mm range
                self.height = self.height / 1000.0

        # Handle legacy rail_height field from older serialized data
        # Check in __dict__ to avoid AttributeError
        if "rail_height" in self.__dict__ and self.__dict__["rail_height"] > 0:
            rail_height_value = self.__dict__["rail_height"]
            self.cushion_height = (
                rail_height_value / 1000.0
                if rail_height_value > 1
                else rail_height_value
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Table dimensions must be positive")
        if self.pocket_radius <= 0:
            raise ValueError("Pocket radius must be positive")
        if not 0.0 <= self.cushion_elasticity <= 1.0:
            raise ValueError("Cushion elasticity must be between 0.0 and 1.0")
        if self.surface_friction < 0:
            raise ValueError("Surface friction must be non-negative")
        if len(self.pocket_positions) != 6:
            raise ValueError("Table must have exactly 6 pockets")

    def is_point_in_pocket(
        self, point: Vector2D, tolerance: float = 0.0
    ) -> tuple[bool, Optional[int]]:
        """Check if a point is in any pocket and return pocket index if found."""
        effective_radius = self.pocket_radius + tolerance
        for i, pocket_pos in enumerate(self.pocket_positions):
            # Calculate distance manually
            dx = point.x - pocket_pos.x
            dy = point.y - pocket_pos.y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= effective_radius:
                return True, i
        return False, None

    def is_point_on_table(self, point: Vector2D, ball_radius: float = 0.0) -> bool:
        """Check if a point (with optional ball radius) is within table bounds."""
        # Use playing area if available, otherwise fall back to rectangle bounds
        if self.playing_area_corners and len(self.playing_area_corners) == 4:
            return self.is_point_in_playing_area(point, ball_radius)

        margin = ball_radius
        return (
            margin <= point.x <= self.width - margin
            and margin <= point.y <= self.height - margin
        )

    def is_point_in_playing_area(self, point: Vector2D, margin: float = 0.0) -> bool:
        """Check if a point is within the calibrated playing area using polygon containment.

        Uses ray casting algorithm to determine if point is inside the playing area polygon.
        """
        if not self.playing_area_corners or len(self.playing_area_corners) != 4:
            # Fall back to rectangle check
            return (
                margin <= point.x <= self.width - margin
                and margin <= point.y <= self.height - margin
            )

        # Apply margin by shrinking the polygon slightly
        corners = self.playing_area_corners
        if margin > 0:
            # Calculate centroid
            cx = sum(c.x for c in corners) / 4
            cy = sum(c.y for c in corners) / 4
            # Shrink each corner towards centroid
            corners = [
                Vector2D(
                    c.x
                    + (cx - c.x)
                    * (margin / c.distance_to(Vector2D(cx, cy, scale=c.scale))),
                    c.y
                    + (cy - c.y)
                    * (margin / c.distance_to(Vector2D(cx, cy, scale=c.scale))),
                    scale=c.scale,
                )
                for c in corners
            ]

        # Ray casting algorithm
        inside = False
        j = len(corners) - 1
        for i in range(len(corners)):
            if ((corners[i].y > point.y) != (corners[j].y > point.y)) and (
                point.x
                < (corners[j].x - corners[i].x)
                * (point.y - corners[i].y)
                / (corners[j].y - corners[i].y)
                + corners[i].x
            ):
                inside = not inside
            j = i
        return inside

    def get_closest_cushion(self, point: Vector2D) -> tuple[str, float, Vector2D]:
        """Get the closest cushion to a point.

        Returns:
            Tuple of (cushion_name, distance, normal_vector)
        """
        # Calculate distances to each cushion
        distances = {
            "top": self.height - point.y,
            "bottom": point.y,
            "left": point.x,
            "right": self.width - point.x,
        }

        # Find closest cushion
        closest_cushion = min(distances, key=lambda k: distances[k])
        closest_distance = distances[closest_cushion]

        # Get normal vector pointing into the table
        normals = {
            "top": Vector2D(0, -1, scale=(1.0, 1.0)),
            "bottom": Vector2D(0, 1, scale=(1.0, 1.0)),
            "left": Vector2D(1, 0, scale=(1.0, 1.0)),
            "right": Vector2D(-1, 0, scale=(1.0, 1.0)),
        }

        return closest_cushion, closest_distance, normals[closest_cushion]

    def scale_playing_area_corners(
        self, from_width: float, from_height: float, to_width: float, to_height: float
    ) -> None:
        """Scale playing area corners from one resolution to another.

        Args:
            from_width: Source resolution width
            from_height: Source resolution height
            to_width: Target resolution width
            to_height: Target resolution height
        """
        if not self.playing_area_corners or len(self.playing_area_corners) != 4:
            return

        scale_x = to_width / from_width
        scale_y = to_height / from_height

        self.playing_area_corners = [
            Vector2D(corner.x * scale_x, corner.y * scale_y, scale=corner.scale)
            for corner in self.playing_area_corners
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "pocket_positions": [p.to_dict() for p in self.pocket_positions],
            "pocket_radius": self.pocket_radius,
            "cushion_elasticity": self.cushion_elasticity,
            "surface_friction": self.surface_friction,
            "surface_slope": self.surface_slope,
            "cushion_height": self.cushion_height,
            "playing_area_corners": (
                [p.to_dict() for p in self.playing_area_corners]
                if self.playing_area_corners
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableState":
        """Create from dictionary."""
        playing_area = data.get("playing_area_corners")
        return cls(
            width=data["width"],
            height=data["height"],
            pocket_positions=[Vector2D.from_dict(p) for p in data["pocket_positions"]],
            pocket_radius=data.get("pocket_radius", 0.0635),
            cushion_elasticity=data.get("cushion_elasticity", 0.85),
            surface_friction=data.get("surface_friction", 0.2),
            surface_slope=data.get("surface_slope", 0.0),
            cushion_height=data.get("cushion_height", 0.064),
            playing_area_corners=(
                [Vector2D.from_dict(p) for p in playing_area] if playing_area else None
            ),
        )

    @classmethod
    def from_calibration(
        cls,
        table_corners_pixel: list[tuple[float, float]],
        camera_resolution: tuple[int, int],
        table_dimensions_real: Optional[tuple[float, float]] = None,
        pocket_positions_pixel: Optional[list[tuple[float, float]]] = None,
    ) -> "TableState":
        """Create TableState from calibration data.

        Args:
            table_corners_pixel: Table corners in camera pixel coordinates (4 corners)
            camera_resolution: Camera resolution (width, height)
            table_dimensions_real: Optional real-world table dimensions in meters (width, height).
                                  If not provided, dimensions are calculated from corners.
            pocket_positions_pixel: Optional pocket positions in camera pixels (6 pockets)

        Returns:
            TableState with dimensions in 4K canonical coordinates
        """
        _ = table_dimensions_real  # Reserved for future use - currently calculate from corners
        from .constants_4k import CANONICAL_RESOLUTION
        from .resolution_converter import ResolutionConverter

        # Transform corners from camera pixels to 4K canonical
        corners_4k = []
        for corner in table_corners_pixel:
            x_4k, y_4k = ResolutionConverter.scale_to_4k(
                corner[0], corner[1], camera_resolution
            )
            corners_4k.append(Vector2D.from_4k(x_4k, y_4k))

        # Calculate table dimensions from corners (average of parallel sides)
        top_width = corners_4k[1].distance_to(corners_4k[0])
        bottom_width = corners_4k[2].distance_to(corners_4k[3])
        width_4k = (top_width + bottom_width) / 2

        left_height = corners_4k[3].distance_to(corners_4k[0])
        right_height = corners_4k[2].distance_to(corners_4k[1])
        height_4k = (left_height + right_height) / 2

        # Calculate pocket positions (6 standard positions)
        if pocket_positions_pixel:
            pockets = []
            for pocket in pocket_positions_pixel:
                x_4k, y_4k = ResolutionConverter.scale_to_4k(
                    pocket[0], pocket[1], camera_resolution
                )
                pockets.append(Vector2D.from_4k(x_4k, y_4k))
        else:
            # Generate standard pocket positions from corners
            pockets = [
                corners_4k[0],  # Top-left
                Vector2D.from_4k(
                    (corners_4k[0].x + corners_4k[1].x) / 2, corners_4k[0].y
                ),  # Top-middle
                corners_4k[1],  # Top-right
                corners_4k[3],  # Bottom-left
                Vector2D.from_4k(
                    (corners_4k[3].x + corners_4k[2].x) / 2, corners_4k[3].y
                ),  # Bottom-middle
                corners_4k[2],  # Bottom-right
            ]

        return cls(
            width=width_4k,
            height=height_4k,
            pocket_positions=pockets,
            playing_area_corners=corners_4k,
        )

    def calculate_pixels_per_meter(self) -> tuple[float, float]:
        """Calculate pixels per meter for this table.

        Returns:
            Tuple of (pixels_per_meter_width, pixels_per_meter_height)
        """
        # This requires physical dimensions, which should be stored
        # For now, return a reasonable default based on standard 9ft table
        # TODO: Store physical dimensions in TableState
        return (self.width / 2.54, self.height / 1.27)


@dataclass
class CueState:
    """Cue stick state information.

    Represents the position, orientation, and properties of the cue stick,
    including estimated force and impact point for shot analysis.
    """

    tip_position: Vector2D
    angle: float  # degrees (0 = pointing right, positive = counter-clockwise)
    elevation: float = 0.0  # degrees above horizontal
    estimated_force: float = 0.0  # Newtons
    impact_point: Optional[Vector2D] = None  # Impact point on cue ball surface
    length: float = 1.47  # Standard cue length in meters
    tip_radius: float = 0.006  # Cue tip radius in meters
    is_visible: bool = False  # Legacy field for compatibility
    confidence: float = 1.0  # Detection confidence [0.0-1.0]
    last_update: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self) -> None:
        """Validate cue state after initialization."""
        if self.length <= 0:
            raise ValueError("Cue length must be positive")
        if self.tip_radius <= 0:
            raise ValueError("Cue tip radius must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def get_direction_vector(self) -> Vector2D:
        """Get the direction vector the cue is pointing."""
        angle_rad = math.radians(self.angle)
        return Vector2D(math.cos(angle_rad), math.sin(angle_rad), scale=(1.0, 1.0))

    def get_aim_line(self, length: float = 2.0) -> tuple[Vector2D, Vector2D]:
        """Get the aim line from the cue tip.

        Returns:
            Tuple of (start_point, end_point) for the aim line
        """
        direction = self.get_direction_vector()
        start_point = self.tip_position
        end_point = start_point + (direction * length)
        return start_point, end_point

    def calculate_impact_velocity(self) -> Vector2D:
        """Calculate the velocity vector that would be imparted to the cue ball."""
        direction = self.get_direction_vector()
        # Simple force-to-velocity conversion (would be refined based on cue mass, contact time, etc.)
        velocity_magnitude = self.estimated_force * 0.1  # Simplified conversion factor
        return direction * velocity_magnitude

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tip_position": self.tip_position.to_dict(),
            "angle": self.angle,
            "elevation": self.elevation,
            "estimated_force": self.estimated_force,
            "impact_point": self.impact_point.to_dict() if self.impact_point else None,
            "length": self.length,
            "tip_radius": self.tip_radius,
            "is_visible": self.is_visible,
            "confidence": self.confidence,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CueState":
        """Create from dictionary."""
        impact_point = None
        if data.get("impact_point"):
            impact_point = Vector2D.from_dict(data["impact_point"])

        return cls(
            tip_position=Vector2D.from_dict(data["tip_position"]),
            angle=data["angle"],
            elevation=data.get("elevation", 0.0),
            estimated_force=data.get("estimated_force", 0.0),
            impact_point=impact_point,
            length=data.get("length", 1.47),
            tip_radius=data.get("tip_radius", 0.006),
            is_visible=data.get("is_visible", False),
            confidence=data.get("confidence", 1.0),
            last_update=data.get("last_update", datetime.now().timestamp()),
        )

    @classmethod
    def from_4k(
        cls, tip_x: float, tip_y: float, angle: float = 0.0, **kwargs
    ) -> "CueState":
        """Create cue from 4K canonical coordinates.

        Args:
            tip_x: Tip X coordinate in 4K pixels (3840×2160)
            tip_y: Tip Y coordinate in 4K pixels
            angle: Cue angle in degrees (0 = pointing right, positive = counter-clockwise)
            **kwargs: Additional cue properties (elevation, estimated_force, length, etc.)

        Returns:
            CueState with tip position in 4K canonical space
        """
        tip_position = Vector2D.from_4k(tip_x, tip_y)
        return cls(tip_position=tip_position, angle=angle, **kwargs)

    @classmethod
    def from_resolution(
        cls,
        tip_x: float,
        tip_y: float,
        resolution: tuple[int, int],
        angle: float = 0.0,
        **kwargs,
    ) -> "CueState":
        """Create cue from any resolution coordinates.

        Args:
            tip_x: Tip X coordinate in source resolution
            tip_y: Tip Y coordinate in source resolution
            resolution: Source resolution as (width, height) tuple
            angle: Cue angle in degrees
            **kwargs: Additional cue properties

        Returns:
            CueState with tip position scaled to 4K
        """
        tip_position = Vector2D.from_resolution(tip_x, tip_y, resolution)
        return cls(tip_position=tip_position, angle=angle, **kwargs)


@dataclass
class GameEvent:
    """Game event information."""

    timestamp: float
    event_type: str  # "shot", "pocket", "scratch", "foul", etc.
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    frame_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "data": self.data,
            "frame_number": self.frame_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameEvent":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            description=data["description"],
            data=data.get("data", {}),
            frame_number=data.get("frame_number", 0),
        )


@dataclass
class CoordinateMetadata:
    """Metadata describing the coordinate space and resolution of positions.

    This metadata helps clarify what coordinate system ball positions are in
    and provides information needed for coordinate transformations.
    """

    # Camera resolution (source of detections)
    camera_resolution: Optional[tuple[int, int]] = (
        None  # (width, height) e.g., (1920, 1080)
    )

    # Table playing area bounds in camera pixels
    table_bounds: Optional[tuple[float, float, float, float]] = (
        None  # (min_x, min_y, max_x, max_y)
    )

    # Coordinate space description
    coordinate_space: str = (
        "world_meters"  # "world_meters", "camera_pixels", "normalized"
    )

    # Scale factor for pixel to meter conversion (if applicable)
    pixels_per_meter: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "camera_resolution": (
                list(self.camera_resolution) if self.camera_resolution else None
            ),
            "table_bounds": list(self.table_bounds) if self.table_bounds else None,
            "coordinate_space": self.coordinate_space,
            "pixels_per_meter": self.pixels_per_meter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoordinateMetadata":
        """Create from dictionary."""
        camera_res = data.get("camera_resolution")
        if camera_res and isinstance(camera_res, list):
            camera_res = tuple(camera_res)

        table_bounds = data.get("table_bounds")
        if table_bounds and isinstance(table_bounds, list):
            table_bounds = tuple(table_bounds)

        return cls(
            camera_resolution=camera_res,
            table_bounds=table_bounds,
            coordinate_space=data.get("coordinate_space", "world_meters"),
            pixels_per_meter=data.get("pixels_per_meter"),
        )


@dataclass
class GameState:
    """Complete game state.

    Represents the entire state of the billiards game at a specific point in time,
    including all balls, table configuration, game rules, and metadata.

    Ball positions are stored in world meters (canonical coordinate system).
    The coordinate_metadata field provides information about the source resolution
    and coordinate transformations for clients that need to convert positions.
    """

    timestamp: float
    frame_number: int
    balls: list[BallState]
    table: TableState
    cue: Optional[CueState] = None
    game_type: GameType = GameType.PRACTICE
    current_player: Optional[int] = None
    scores: dict[int, int] = field(default_factory=dict)
    is_break: bool = False
    last_shot: Optional["ShotAnalysis"] = None  # Will be defined later
    events: list[GameEvent] = field(default_factory=list)
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    state_confidence: float = 1.0
    coordinate_metadata: Optional[CoordinateMetadata] = (
        None  # Resolution and coordinate space info
    )

    def __post_init__(self) -> None:
        """Validate game state after initialization."""
        if self.frame_number < 0:
            raise ValueError("Frame number must be non-negative")
        if not 0.0 <= self.state_confidence <= 1.0:
            raise ValueError("State confidence must be between 0.0 and 1.0")
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")

    def get_ball_by_id(self, ball_id: str) -> Optional[BallState]:
        """Get a ball by its ID."""
        for ball in self.balls:
            if ball.id == ball_id:
                return ball
        return None

    def get_cue_ball(self) -> Optional[BallState]:
        """Get the cue ball."""
        for ball in self.balls:
            if ball.is_cue_ball:
                return ball
        return None

    def get_numbered_balls(self) -> list[BallState]:
        """Get all numbered balls (non-cue balls)."""
        return [ball for ball in self.balls if not ball.is_cue_ball]

    def get_pocketed_balls(self) -> list[BallState]:
        """Get all pocketed balls."""
        return [ball for ball in self.balls if ball.is_pocketed]

    def get_active_balls(self) -> list[BallState]:
        """Get all non-pocketed balls."""
        return [ball for ball in self.balls if not ball.is_pocketed]

    def get_moving_balls(self, threshold: float = 0.001) -> list[BallState]:
        """Get all balls that are currently moving."""
        return [ball for ball in self.balls if ball.is_moving(threshold)]

    def is_table_clear(self, threshold: float = 0.001) -> bool:
        """Check if all balls have stopped moving."""
        return len(self.get_moving_balls(threshold)) == 0

    def count_balls_by_player(self, game_type: GameType) -> dict[int, int]:
        """Count balls belonging to each player based on game type."""
        if game_type == GameType.EIGHT_BALL:
            return self._count_eight_ball_groups()
        elif game_type == GameType.NINE_BALL:
            return self._count_nine_ball_remaining()
        else:
            return {}

    def _count_eight_ball_groups(self) -> dict[int, int]:
        """Count solids and stripes for 8-ball."""
        solids = sum(
            1
            for ball in self.balls
            if ball.number and 1 <= ball.number <= 7 and not ball.is_pocketed
        )
        stripes = sum(
            1
            for ball in self.balls
            if ball.number and 9 <= ball.number <= 15 and not ball.is_pocketed
        )
        return {1: solids, 2: stripes}  # Player 1: solids, Player 2: stripes

    def _count_nine_ball_remaining(self) -> dict[int, int]:
        """Count remaining balls for 9-ball."""
        remaining = sum(
            1
            for ball in self.balls
            if ball.number and 1 <= ball.number <= 9 and not ball.is_pocketed
        )
        return {1: remaining, 2: remaining}  # Both players target same balls

    def add_event(
        self,
        event: str,
        event_type: str = "info",
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a game event to the history."""
        if data is None:
            data = {}
        game_event = GameEvent(
            timestamp=self.timestamp,
            event_type=event_type,
            description=event,
            data=data,
            frame_number=self.frame_number,
        )
        self.events.append(game_event)

    def validate_consistency(self) -> list[str]:
        """Validate the consistency of the game state."""
        errors = []

        # Check for duplicate ball IDs
        ball_ids = [ball.id for ball in self.balls]
        if len(ball_ids) != len(set(ball_ids)):
            errors.append("Duplicate ball IDs found")

        # Check for exactly one cue ball
        cue_balls = [ball for ball in self.balls if ball.is_cue_ball]
        if len(cue_balls) != 1:
            errors.append(f"Expected 1 cue ball, found {len(cue_balls)}")

        # Check ball positions are within table bounds
        for ball in self.balls:
            if not ball.is_pocketed and not self.table.is_point_on_table(
                ball.position, ball.radius
            ):
                errors.append(f"Ball {ball.id} is outside table bounds")

        # Check for overlapping balls
        for i, ball1 in enumerate(self.balls):
            for ball2 in self.balls[i + 1 :]:
                if not ball1.is_pocketed and not ball2.is_pocketed:
                    if ball1.is_touching(
                        ball2, tolerance=-0.001
                    ):  # Slight overlap tolerance
                        errors.append(
                            f"Balls {ball1.id} and {ball2.id} are overlapping"
                        )

        return errors

    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        return GameState(
            timestamp=self.timestamp,
            frame_number=self.frame_number,
            balls=[ball.copy() for ball in self.balls],
            table=self.table,  # Tables are typically immutable during a game
            cue=self.cue,  # CueState is typically replaced, not modified
            game_type=self.game_type,
            current_player=self.current_player,
            scores=dict(self.scores),
            is_break=self.is_break,
            last_shot=self.last_shot,  # ShotAnalysis is typically immutable
            events=[GameEvent.from_dict(event.to_dict()) for event in self.events],
            is_valid=self.is_valid,
            validation_errors=list(self.validation_errors),
            state_confidence=self.state_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "balls": [ball.to_dict() for ball in self.balls],
            "table": self.table.to_dict(),
            "cue": self.cue.to_dict() if self.cue else None,
            "game_type": self.game_type.value,
            "current_player": self.current_player,
            "scores": self.scores,
            "is_break": self.is_break,
            "last_shot": self.last_shot.to_dict() if self.last_shot else None,
            "events": [event.to_dict() for event in self.events],
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "state_confidence": self.state_confidence,
            "coordinate_metadata": (
                self.coordinate_metadata.to_dict() if self.coordinate_metadata else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        """Create from dictionary."""
        cue = None
        if data.get("cue"):
            cue = CueState.from_dict(data["cue"])

        last_shot = None
        if data.get("last_shot"):
            # ShotAnalysis is now defined in this module
            last_shot = ShotAnalysis.from_dict(data["last_shot"])

        events = []
        if data.get("events"):
            events = [GameEvent.from_dict(event) for event in data["events"]]

        coordinate_metadata = None
        if data.get("coordinate_metadata"):
            coordinate_metadata = CoordinateMetadata.from_dict(
                data["coordinate_metadata"]
            )

        return cls(
            timestamp=data["timestamp"],
            frame_number=data["frame_number"],
            balls=[BallState.from_dict(ball) for ball in data["balls"]],
            table=TableState.from_dict(data["table"]),
            cue=cue,
            game_type=GameType(data.get("game_type", "practice")),
            current_player=data.get("current_player"),
            scores=data.get("scores", {}),
            is_break=data.get("is_break", False),
            last_shot=last_shot,
            events=events,
            is_valid=data.get("is_valid", True),
            validation_errors=data.get("validation_errors", []),
            state_confidence=data.get("state_confidence", 1.0),
            coordinate_metadata=coordinate_metadata,
        )

    @classmethod
    def create_initial_state(
        cls, table: TableState, game_type: GameType = GameType.PRACTICE
    ) -> "GameState":
        """Create an initial game state with standard ball setup.

        Args:
            table: TableState to use (must be provided, typically from calibration)
            game_type: Type of game to initialize

        Returns:
            GameState with initial ball setup
        """
        timestamp = datetime.now().timestamp()

        # Create standard 15-ball rack
        balls = []

        # Cue ball
        cue_ball = BallState(
            id="cue",
            position=Vector2D(table.width * 0.25, table.height * 0.5, scale=(1.0, 1.0)),
            is_cue_ball=True,
        )
        balls.append(cue_ball)

        # Numbered balls in triangle formation
        rack_x = table.width * 0.75
        rack_y = table.height * 0.5
        ball_radius = 0.028575  # Standard ball radius in meters

        # Standard 15-ball rack positions - adjusted for proper ball spacing
        rack_positions = [
            (0, 0),  # 1 ball (front)
            (-1, -0.5),
            (-1, 0.5),  # 2nd row
            (-2, -1),
            (-2, 0),
            (-2, 1),  # 3rd row
            (-3, -1.5),
            (-3, -0.5),
            (-3, 0.5),
            (-3, 1.5),  # 4th row
            (-4, -2),
            (-4, -1),
            (-4, 0),
            (-4, 1),
            (-4, 2),  # 5th row
        ]

        for i, (dx, dy) in enumerate(rack_positions):
            ball_number = i + 1
            # Use proper ball spacing (2 * radius with small gap)
            spacing = 2.1 * ball_radius
            x = rack_x + dx * spacing * 0.866  # 0.866 = sqrt(3)/2 for triangle spacing
            y = rack_y + dy * spacing * 0.5

            ball = BallState(
                id=f"ball_{ball_number}",
                position=Vector2D(x, y, scale=(1.0, 1.0)),
                number=ball_number,
            )
            balls.append(ball)

        return cls(
            timestamp=timestamp,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=game_type,
            is_break=True,
        )


@dataclass
class Collision:
    """Collision prediction information.

    Represents a predicted or actual collision between balls or between
    a ball and a table element (cushion, pocket).
    """

    time: float  # seconds from now (negative for past collisions)
    position: Vector2D  # Position where collision occurs
    ball1_id: str  # Primary ball involved
    ball2_id: Optional[str] = None  # Secondary ball (None for cushion/pocket collision)
    type: str = "ball"  # "ball", "cushion", "pocket"
    resulting_velocities: Optional[dict[str, Vector2D]] = None
    impact_force: float = 0.0  # Force of impact in Newtons
    confidence: float = 1.0  # Prediction confidence [0.0-1.0]

    def __post_init__(self) -> None:
        """Validate collision data after initialization."""
        valid_types = {"ball", "cushion", "pocket"}
        if self.type not in valid_types:
            raise ValueError(f"Collision type must be one of {valid_types}")
        if self.type == "ball" and self.ball2_id is None:
            raise ValueError("Ball collisions must have ball2_id")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def is_ball_collision(self) -> bool:
        """Check if this is a ball-to-ball collision."""
        return self.type == "ball" and self.ball2_id is not None

    def is_cushion_collision(self) -> bool:
        """Check if this is a ball-to-cushion collision."""
        return self.type == "cushion"

    def is_pocket_collision(self) -> bool:
        """Check if this is a ball going into a pocket."""
        return self.type == "pocket"

    def get_involved_balls(self) -> list[str]:
        """Get list of all ball IDs involved in the collision."""
        balls = [self.ball1_id]
        if self.ball2_id:
            balls.append(self.ball2_id)
        return balls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "time": self.time,
            "position": self.position.to_dict(),
            "ball1_id": self.ball1_id,
            "ball2_id": self.ball2_id,
            "type": self.type,
            "resulting_velocities": (
                {k: v.to_dict() for k, v in self.resulting_velocities.items()}
                if self.resulting_velocities
                else None
            ),
            "impact_force": self.impact_force,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Collision":
        """Create from dictionary."""
        resulting_velocities = None
        if data.get("resulting_velocities"):
            resulting_velocities = {
                k: Vector2D.from_dict(v)
                for k, v in data["resulting_velocities"].items()
            }

        return cls(
            time=data["time"],
            position=Vector2D.from_dict(data["position"]),
            ball1_id=data["ball1_id"],
            ball2_id=data.get("ball2_id"),
            type=data.get("type", "ball"),
            resulting_velocities=resulting_velocities,
            impact_force=data.get("impact_force", 0.0),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Trajectory:
    """Ball trajectory information.

    Represents the predicted or actual path of a ball, including
    collision points, final resting position, and timing information.
    """

    ball_id: str
    points: list[Vector2D]  # Path points at regular time intervals
    collisions: list[Collision]  # Predicted collisions along the path
    final_position: Vector2D
    final_velocity: Vector2D
    time_to_rest: float  # seconds until ball comes to rest
    will_be_pocketed: bool
    pocket_id: Optional[int] = None
    max_velocity: float = 0.0  # Maximum velocity during trajectory
    total_distance: float = 0.0  # Total distance traveled
    confidence: float = 1.0  # Prediction confidence [0.0-1.0]

    def __post_init__(self) -> None:
        """Validate and calculate derived properties after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.time_to_rest < 0:
            raise ValueError("Time to rest must be non-negative")
        if self.will_be_pocketed and self.pocket_id is None:
            raise ValueError("Pocketed balls must have a pocket_id")

        # Calculate total distance if not provided
        if self.total_distance == 0.0 and len(self.points) > 1:
            self.total_distance = self._calculate_total_distance()

    def _calculate_total_distance(self) -> float:
        """Calculate total distance from trajectory points."""
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i - 1].distance_to(self.points[i])
        return total

    def get_position_at_time(self, time: float) -> Optional[Vector2D]:
        """Get the ball position at a specific time.

        Args:
            time: Time in seconds from trajectory start

        Returns:
            Position vector if time is within trajectory, None otherwise
        """
        if time < 0 or time > self.time_to_rest or not self.points:
            return None

        if time >= self.time_to_rest:
            return self.final_position

        # Linear interpolation between trajectory points
        time_per_point = (
            self.time_to_rest / (len(self.points) - 1) if len(self.points) > 1 else 0
        )
        if time_per_point == 0:
            return self.points[0] if self.points else None

        index = time / time_per_point
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(self.points) - 1)

        if lower_idx >= len(self.points):
            return self.final_position

        if lower_idx == upper_idx:
            return self.points[lower_idx]

        # Interpolate between points
        t = index - lower_idx
        p1, p2 = self.points[lower_idx], self.points[upper_idx]
        return Vector2D(
            p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y), scale=p1.scale
        )

    def get_collisions_before_time(self, time: float) -> list[Collision]:
        """Get all collisions that occur before the specified time."""
        return [c for c in self.collisions if c.time <= time]

    def get_next_collision(self) -> Optional[Collision]:
        """Get the next collision in the trajectory."""
        future_collisions = [c for c in self.collisions if c.time > 0]
        return (
            min(future_collisions, key=lambda c: c.time) if future_collisions else None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ball_id": self.ball_id,
            "points": [p.to_dict() for p in self.points],
            "collisions": [c.to_dict() for c in self.collisions],
            "final_position": self.final_position.to_dict(),
            "final_velocity": self.final_velocity.to_dict(),
            "time_to_rest": self.time_to_rest,
            "will_be_pocketed": self.will_be_pocketed,
            "pocket_id": self.pocket_id,
            "max_velocity": self.max_velocity,
            "total_distance": self.total_distance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        return cls(
            ball_id=data["ball_id"],
            points=[Vector2D.from_dict(p) for p in data["points"]],
            collisions=[Collision.from_dict(c) for c in data["collisions"]],
            final_position=Vector2D.from_dict(data["final_position"]),
            final_velocity=Vector2D.from_dict(data["final_velocity"]),
            time_to_rest=data["time_to_rest"],
            will_be_pocketed=data["will_be_pocketed"],
            pocket_id=data.get("pocket_id"),
            max_velocity=data.get("max_velocity", 0.0),
            total_distance=data.get("total_distance", 0.0),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ShotAnalysis:
    """Shot analysis and recommendations.

    Contains comprehensive analysis of a potential or actual shot,
    including difficulty assessment, success probability, and recommendations.
    """

    shot_type: ShotType
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    success_probability: float  # 0.0 to 1.0
    recommended_force: float  # Newtons
    recommended_angle: float  # degrees
    target_ball_id: Optional[str] = None
    target_pocket_id: Optional[int] = None
    potential_problems: list[str] = field(default_factory=list)
    alternative_shots: list["ShotAnalysis"] = field(default_factory=list)
    risk_assessment: dict[str, float] = field(default_factory=dict)
    predicted_outcome: Optional[str] = None
    cue_ball_final_position: Optional[Vector2D] = None
    shot_confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate shot analysis after initialization."""
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("Difficulty must be between 0.0 and 1.0")
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("Success probability must be between 0.0 and 1.0")
        if not 0.0 <= self.shot_confidence <= 1.0:
            raise ValueError("Shot confidence must be between 0.0 and 1.0")
        if self.recommended_force < 0:
            raise ValueError("Recommended force must be non-negative")

    def is_safe_shot(self, safety_threshold: float = 0.7) -> bool:
        """Check if this is considered a safe shot."""
        return self.success_probability >= safety_threshold

    def is_high_risk(self, risk_threshold: float = 0.3) -> bool:
        """Check if this shot has high risk based on problems and probability."""
        return (
            len(self.potential_problems) > 2
            or self.success_probability < risk_threshold
        )

    def get_risk_score(self) -> float:
        """Calculate overall risk score (0.0 = low risk, 1.0 = high risk)."""
        base_risk = 1.0 - self.success_probability
        problem_risk = min(len(self.potential_problems) * 0.1, 0.5)
        difficulty_risk = self.difficulty * 0.3
        return min(base_risk + problem_risk + difficulty_risk, 1.0)

    def add_problem(self, problem: str) -> None:
        """Add a potential problem to the analysis."""
        if problem not in self.potential_problems:
            self.potential_problems.append(problem)

    def add_risk(self, risk_type: str, risk_value: float) -> None:
        """Add a risk assessment."""
        self.risk_assessment[risk_type] = max(0.0, min(1.0, risk_value))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "shot_type": self.shot_type.value,
            "difficulty": self.difficulty,
            "success_probability": self.success_probability,
            "recommended_force": self.recommended_force,
            "recommended_angle": self.recommended_angle,
            "target_ball_id": self.target_ball_id,
            "target_pocket_id": self.target_pocket_id,
            "potential_problems": self.potential_problems,
            "alternative_shots": [shot.to_dict() for shot in self.alternative_shots],
            "risk_assessment": self.risk_assessment,
            "predicted_outcome": self.predicted_outcome,
            "cue_ball_final_position": (
                self.cue_ball_final_position.to_dict()
                if self.cue_ball_final_position
                else None
            ),
            "shot_confidence": self.shot_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShotAnalysis":
        """Create from dictionary."""
        cue_ball_final_position = None
        if data.get("cue_ball_final_position"):
            cue_ball_final_position = Vector2D.from_dict(
                data["cue_ball_final_position"]
            )

        # Handle alternative shots (avoid infinite recursion)
        alternative_shots = []
        if data.get("alternative_shots"):
            for alt_data in data["alternative_shots"]:
                # Create alternative shots without their own alternatives
                alt_data_copy = dict(alt_data)
                alt_data_copy["alternative_shots"] = []
                alternative_shots.append(cls.from_dict(alt_data_copy))

        return cls(
            shot_type=ShotType(data["shot_type"]),
            difficulty=data["difficulty"],
            success_probability=data["success_probability"],
            recommended_force=data["recommended_force"],
            recommended_angle=data["recommended_angle"],
            target_ball_id=data.get("target_ball_id"),
            target_pocket_id=data.get("target_pocket_id"),
            potential_problems=data.get("potential_problems", []),
            alternative_shots=alternative_shots,
            risk_assessment=data.get("risk_assessment", {}),
            predicted_outcome=data.get("predicted_outcome"),
            cue_ball_final_position=cue_ball_final_position,
            shot_confidence=data.get("shot_confidence", 1.0),
        )


# Utility functions for model operations
def serialize_to_json(
    obj: Union[
        Vector2D,
        BallState,
        TableState,
        CueState,
        Collision,
        Trajectory,
        ShotAnalysis,
        GameState,
    ],
) -> str:
    """Serialize any model object to JSON string."""
    return json.dumps(obj.to_dict(), indent=2)


def deserialize_from_json(json_str: str, model_class: Any) -> Any:
    """Deserialize JSON string to model object."""
    data = json.loads(json_str)
    return model_class.from_dict(data)


def calculate_ball_separation(ball1: BallState, ball2: BallState) -> float:
    """Calculate the separation between two balls (edge to edge)."""
    center_distance = ball1.distance_to(
        ball2
    )  # Use BallState's distance_to which handles both types
    return max(0.0, center_distance - ball1.radius - ball2.radius)


def find_closest_ball(
    target_ball: BallState, other_balls: list[BallState]
) -> Optional[BallState]:
    """Find the closest ball to the target ball."""
    if not other_balls:
        return None

    closest_ball = None
    min_distance = float("inf")

    for ball in other_balls:
        if ball.id != target_ball.id and not ball.is_pocketed:
            distance = target_ball.distance_to(ball)
            if distance < min_distance:
                min_distance = distance
                closest_ball = ball

    return closest_ball


def validate_ball_physics(ball: BallState, max_velocity: float = 10.0) -> list[str]:
    """Validate ball physics constraints."""
    errors = []

    if ball.velocity.magnitude() > max_velocity:
        errors.append(
            f"Ball {ball.id} velocity {ball.velocity.magnitude():.2f} exceeds maximum {max_velocity}"
        )

    if ball.mass <= 0:
        errors.append(f"Ball {ball.id} has invalid mass: {ball.mass}")

    if ball.radius <= 0:
        errors.append(f"Ball {ball.id} has invalid radius: {ball.radius}")

    return errors


def create_standard_ball_set() -> list[BallState]:
    """Create a standard set of 15 numbered balls plus cue ball."""
    balls = []

    # Cue ball at a reasonable position
    cue_ball = BallState(
        id="cue",
        position=Vector2D(
            0.5, 0.5, scale=(1.0, 1.0)
        ),  # Safe position in middle of a standard table
        is_cue_ball=True,
    )
    balls.append(cue_ball)

    # Numbered balls 1-15 positioned in a simple line (for testing)
    ball_radius = 0.028575
    spacing = 2.2 * ball_radius
    start_x = 1.0
    start_y = 0.5

    for i in range(1, 16):
        x = start_x + (i - 1) * spacing
        ball = BallState(
            id=f"ball_{i}", position=Vector2D(x, start_y, scale=(1.0, 1.0)), number=i
        )
        balls.append(ball)

    return balls


# Backward compatibility aliases
Table = TableState
Ball = BallState
