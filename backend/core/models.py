"""Core data models and types for the billiards trainer system.

This module contains all the fundamental data structures that represent the game state,
physics objects, and analysis results. These models serve as the foundation for all
other core modules and provide the interface for data exchange between modules.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union


class ShotType(Enum):
    BREAK = "break"
    DIRECT = "direct"
    BANK = "bank"
    COMBINATION = "combination"
    SAFETY = "safety"
    MASSE = "masse"


class GameType(Enum):
    EIGHT_BALL = "8ball"
    NINE_BALL = "9ball"
    STRAIGHT_POOL = "straight"
    PRACTICE = "practice"


@dataclass
class Vector2D:
    """2D vector for positions, velocities, and other vector quantities.

    Provides mathematical operations commonly needed for physics calculations
    and geometric computations in the billiards simulation.
    """

    x: float
    y: float

    def magnitude(self) -> float:
        """Calculate the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (avoids sqrt for performance)."""
        return self.x**2 + self.y**2

    def normalize(self) -> "Vector2D":
        """Return a normalized (unit) vector in the same direction."""
        mag = self.magnitude()
        return Vector2D(self.x / mag, self.y / mag) if mag > 0 else Vector2D(0, 0)

    def dot(self, other: "Vector2D") -> float:
        """Calculate the dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2D") -> float:
        """Calculate the 2D cross product (returns scalar z-component)."""
        return self.x * other.y - self.y * other.x

    def distance_to(self, other: "Vector2D") -> float:
        """Calculate the distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "Vector2D") -> float:
        """Calculate the angle to another vector in radians."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle: float) -> "Vector2D":
        """Rotate the vector by the given angle in radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a
        )

    def scale(self, factor: float) -> "Vector2D":
        """Scale the vector by a scalar factor."""
        return Vector2D(self.x * factor, self.y * factor)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Add two vectors."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Subtract two vectors."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        """Multiply vector by a scalar."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        """Divide vector by a scalar."""
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        """Negate the vector."""
        return Vector2D(-self.x, -self.y)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Vector2D":
        """Create from dictionary."""
        return cls(data["x"], data["y"])

    @classmethod
    def zero(cls) -> "Vector2D":
        """Create a zero vector."""
        return cls(0.0, 0.0)

    @classmethod
    def unit_x(cls) -> "Vector2D":
        """Create a unit vector in the x direction."""
        return cls(1.0, 0.0)

    @classmethod
    def unit_y(cls) -> "Vector2D":
        """Create a unit vector in the y direction."""
        return cls(0.0, 1.0)


@dataclass
class BallState:
    """Complete ball state information.

    Represents all physical and logical properties of a billiards ball,
    including position, velocity, spin, and game-specific attributes.
    """

    id: str
    position: Vector2D
    velocity: Vector2D = field(default_factory=Vector2D.zero)
    radius: float = 0.028575  # Standard pool ball radius in meters (57.15mm diameter)
    mass: float = 0.17  # kg, standard pool ball mass
    spin: Vector2D = field(default_factory=Vector2D.zero)  # Top/back/side spin
    is_cue_ball: bool = False
    is_pocketed: bool = False
    number: Optional[int] = None
    confidence: float = 1.0  # Detection confidence [0.0-1.0]
    last_update: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self):
        """Validate ball state after initialization."""
        if self.radius <= 0:
            raise ValueError("Ball radius must be positive")
        if self.mass <= 0:
            raise ValueError("Ball mass must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        # Handle legacy last_seen field
        if hasattr(self, "last_seen") and self.last_seen is not None:
            self.last_update = self.last_seen

    def kinetic_energy(self) -> float:
        """Calculate the kinetic energy of the ball."""
        linear_ke = 0.5 * self.mass * self.velocity.magnitude_squared()
        # Rotational KE: 0.5 * I * ω², where I = (2/5) * m * r² for sphere
        moment_of_inertia = 0.4 * self.mass * self.radius**2
        angular_velocity = self.spin.magnitude()
        rotational_ke = 0.5 * moment_of_inertia * angular_velocity**2
        return linear_ke + rotational_ke

    def is_moving(self, threshold: float = 0.001) -> bool:
        """Check if the ball is moving above the threshold velocity."""
        return self.velocity.magnitude() > threshold

    def is_spinning(self, threshold: float = 0.1) -> bool:
        """Check if the ball has significant spin."""
        return self.spin.magnitude() > threshold

    def distance_to(self, other: "BallState") -> float:
        """Calculate distance to another ball's center."""
        return self.position.distance_to(other.position)

    def is_touching(self, other: "BallState", tolerance: float = 0.001) -> bool:
        """Check if this ball is touching another ball."""
        min_distance = self.radius + other.radius + tolerance
        return self.distance_to(other) <= min_distance

    def copy(self) -> "BallState":
        """Create a deep copy of the ball state."""
        return BallState(
            id=self.id,
            position=Vector2D(self.position.x, self.position.y),
            velocity=Vector2D(self.velocity.x, self.velocity.y),
            radius=self.radius,
            mass=self.mass,
            spin=Vector2D(self.spin.x, self.spin.y) if self.spin else None,
            is_cue_ball=self.is_cue_ball,
            is_pocketed=self.is_pocketed,
            number=self.number,
            confidence=self.confidence,
            last_update=self.last_update,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "radius": self.radius,
            "mass": self.mass,
            "spin": self.spin.to_dict(),
            "is_cue_ball": self.is_cue_ball,
            "is_pocketed": self.is_pocketed,
            "number": self.number,
            "confidence": self.confidence,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BallState":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            position=Vector2D.from_dict(data["position"]),
            velocity=Vector2D.from_dict(data.get("velocity", {"x": 0, "y": 0})),
            radius=data.get("radius", 0.028575),
            mass=data.get("mass", 0.17),
            spin=Vector2D.from_dict(data.get("spin", {"x": 0, "y": 0})),
            is_cue_ball=data.get("is_cue_ball", False),
            is_pocketed=data.get("is_pocketed", False),
            number=data.get("number"),
            confidence=data.get("confidence", 1.0),
            last_update=data.get("last_update", datetime.now().timestamp()),
        )


@dataclass
class TableState:
    """Pool table state information.

    Represents the physical properties of the billiards table including
    dimensions, pocket locations, and surface characteristics that affect
    ball physics.
    """

    width: float  # meters (converted from mm in legacy systems)
    height: float  # meters (converted from mm in legacy systems)
    pocket_positions: list[Vector2D]
    pocket_radius: float = 0.0635  # Standard pocket radius in meters
    cushion_elasticity: float = 0.85  # Coefficient of restitution
    surface_friction: float = 0.2  # Rolling friction coefficient
    surface_slope: float = 0.0  # Surface slope in degrees
    cushion_height: float = (
        0.064  # Height of cushions in meters (converted from rail_height)
    )

    def __post_init__(self):
        """Validate table state after initialization and handle legacy fields."""
        # Convert legacy mm measurements to meters if needed
        if self.width > 10:  # Assume values > 10 are in mm
            self.width = self.width / 1000.0
        if self.height > 10:  # Assume values > 10 are in mm
            self.height = self.height / 1000.0

        # Handle legacy rail_height field
        if hasattr(self, "rail_height") and self.rail_height > 0:
            self.cushion_height = (
                self.rail_height / 1000.0 if self.rail_height > 1 else self.rail_height
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
            if point.distance_to(pocket_pos) <= effective_radius:
                return True, i
        return False, None

    def is_point_on_table(self, point: Vector2D, ball_radius: float = 0.0) -> bool:
        """Check if a point (with optional ball radius) is within table bounds."""
        margin = ball_radius
        return (
            margin <= point.x <= self.width - margin
            and margin <= point.y <= self.height - margin
        )

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
        closest_cushion = min(distances, key=distances.get)
        closest_distance = distances[closest_cushion]

        # Get normal vector pointing into the table
        normals = {
            "top": Vector2D(0, -1),
            "bottom": Vector2D(0, 1),
            "left": Vector2D(1, 0),
            "right": Vector2D(-1, 0),
        }

        return closest_cushion, closest_distance, normals[closest_cushion]

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
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableState":
        """Create from dictionary."""
        return cls(
            width=data["width"],
            height=data["height"],
            pocket_positions=[Vector2D.from_dict(p) for p in data["pocket_positions"]],
            pocket_radius=data.get("pocket_radius", 0.0635),
            cushion_elasticity=data.get("cushion_elasticity", 0.85),
            surface_friction=data.get("surface_friction", 0.2),
            surface_slope=data.get("surface_slope", 0.0),
            cushion_height=data.get("cushion_height", 0.064),
        )

    @classmethod
    def standard_9ft_table(cls) -> "TableState":
        """Create a standard 9-foot pool table."""
        width = 2.54  # 9 feet in meters
        height = 1.27  # 4.5 feet in meters

        # Standard pocket positions for 9-foot table
        pocket_positions = [
            Vector2D(0, 0),  # Bottom left corner
            Vector2D(width / 2, 0),  # Bottom middle
            Vector2D(width, 0),  # Bottom right corner
            Vector2D(0, height),  # Top left corner
            Vector2D(width / 2, height),  # Top middle
            Vector2D(width, height),  # Top right corner
        ]

        return cls(width=width, height=height, pocket_positions=pocket_positions)


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

    def __post_init__(self):
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
        return Vector2D(math.cos(angle_rad), math.sin(angle_rad))

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
class GameState:
    """Complete game state.

    Represents the entire state of the billiards game at a specific point in time,
    including all balls, table configuration, game rules, and metadata.
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

    def __post_init__(self):
        """Validate game state after initialization."""
        if self.frame_number < 0:
            raise ValueError("Frame number must be non-negative")
        if not 0.0 <= self.state_confidence <= 1.0:
            raise ValueError("State confidence must be between 0.0 and 1.0")
        if self.timestamp < 0:
            raise ValueError("Timestamp must be non-negative")
        # Handle legacy validation_errors None
        if self.validation_errors is None:
            self.validation_errors = []

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
        self, event: str, event_type: str = "info", data: dict[str, Any] = None
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
        )

    @classmethod
    def create_initial_state(
        cls, game_type: GameType = GameType.PRACTICE
    ) -> "GameState":
        """Create an initial game state with standard ball setup."""
        timestamp = datetime.now().timestamp()
        table = TableState.standard_9ft_table()

        # Create standard 15-ball rack
        balls = []

        # Cue ball
        cue_ball = BallState(
            id="cue",
            position=Vector2D(table.width * 0.25, table.height * 0.5),
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
                id=f"ball_{ball_number}", position=Vector2D(x, y), number=ball_number
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

    def __post_init__(self):
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
            "resulting_velocities": {
                k: v.to_dict() for k, v in self.resulting_velocities.items()
            }
            if self.resulting_velocities
            else None,
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

    def __post_init__(self):
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
        return Vector2D(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))

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

    def __post_init__(self):
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
            "cue_ball_final_position": self.cue_ball_final_position.to_dict()
            if self.cue_ball_final_position
            else None,
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
    ]
) -> str:
    """Serialize any model object to JSON string."""
    return json.dumps(obj.to_dict(), indent=2)


def deserialize_from_json(json_str: str, model_class) -> Any:
    """Deserialize JSON string to model object."""
    data = json.loads(json_str)
    return model_class.from_dict(data)


def calculate_ball_separation(ball1: BallState, ball2: BallState) -> float:
    """Calculate the separation between two balls (edge to edge)."""
    center_distance = ball1.position.distance_to(ball2.position)
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
        position=Vector2D(0.5, 0.5),  # Safe position in middle of a standard table
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
        ball = BallState(id=f"ball_{i}", position=Vector2D(x, start_y), number=i)
        balls.append(ball)

    return balls
