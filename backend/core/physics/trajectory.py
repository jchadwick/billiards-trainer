"""Comprehensive trajectory calculation algorithms for billiards simulation.

All calculations are performed in 4K pixels (3840×2160) as the canonical coordinate system.
Vector2D instances must include scale metadata for proper resolution handling.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..constants_4k import BALL_RADIUS_4K, PIXELS_PER_METER_REFERENCE
from ..coordinates import Vector2D
from ..models import BallState, CueState, TableState
from ..utils.cache import CacheManager
from ..utils.geometry import GeometryUtils
from ..utils.math import MathUtils


class CollisionType(Enum):
    """Types of collisions that can occur."""

    BALL_BALL = "ball_ball"
    BALL_CUSHION = "ball_cushion"
    BALL_POCKET = "ball_pocket"
    NONE = "none"


class TrajectoryQuality(Enum):
    """Quality levels for trajectory calculation."""

    LOW = "low"  # Fast, less accurate
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # Precise, slower
    ULTRA = "ultra"  # Maximum precision


@dataclass
class TrajectoryPoint:
    """A single point along a ball's trajectory.

    All Vector2D values should have scale=[1.0, 1.0] (4K canonical coordinates).
    """

    time: float  # Time from start (seconds)
    position: Vector2D  # Ball position in 4K pixels
    velocity: Vector2D  # Ball velocity in pixels/second
    acceleration: Vector2D  # Ball acceleration in pixels/second²
    spin: Vector2D  # Spin state (angular velocity)
    energy: float  # Kinetic energy in joules


@dataclass
class PredictedCollision:
    """Detailed collision prediction."""

    time: float  # Time until collision
    position: Vector2D  # Collision point
    type: CollisionType  # Type of collision
    ball1_id: str  # Primary ball
    ball2_id: Optional[str]  # Secondary ball (None for cushion/pocket)
    impact_angle: float  # Angle of impact
    impact_velocity: float  # Velocity at impact
    resulting_velocities: dict[str, Vector2D]  # Post-collision velocities
    confidence: float  # Prediction confidence (0-1)
    cushion_normal: Optional[Vector2D] = None  # For cushion collisions
    pocket_id: Optional[int] = None  # For pocket collisions


@dataclass
class TrajectoryBranch:
    """Alternative trajectory possibility."""

    probability: float  # Likelihood of this outcome
    description: str  # Human-readable description
    points: list[TrajectoryPoint]
    collisions: list[PredictedCollision]
    final_state: BallState
    success_metrics: dict[str, float]  # Various success measurements


@dataclass
class Trajectory:
    """Complete ball trajectory information."""

    ball_id: str
    initial_state: BallState
    points: list[TrajectoryPoint] = field(default_factory=list)
    collisions: list[PredictedCollision] = field(default_factory=list)
    branches: list[TrajectoryBranch] = field(default_factory=list)
    final_position: Vector2D = None
    final_velocity: Vector2D = None
    time_to_rest: float = 0.0
    total_distance: float = 0.0
    will_be_pocketed: bool = False
    pocket_id: Optional[int] = None
    success_probability: float = 0.0
    quality: TrajectoryQuality = TrajectoryQuality.MEDIUM
    calculation_time: float = 0.0
    cache_key: Optional[str] = None
    triggered_by_ball: Optional[str] = None  # ID of ball that caused this ball to move


@dataclass
class MultiballTrajectoryResult:
    """Container for trajectories of multiple balls in a shot sequence."""

    primary_ball_id: str  # The cue ball
    trajectories: dict[str, Trajectory]  # ball_id -> trajectory
    collision_sequence: list[PredictedCollision]  # All collisions in order
    total_calculation_time: float = 0.0

    def get_position_at_time(self, t: float) -> Optional[Vector2D]:
        """Get ball position at specific time."""
        if not self.points:
            return None

        # Find closest time points
        for i, point in enumerate(self.points):
            if point.time >= t:
                if i == 0:
                    return point.position

                # Interpolate between points
                prev_point = self.points[i - 1]
                ratio = (t - prev_point.time) / (point.time - prev_point.time)

                return Vector2D(
                    prev_point.position.x
                    + ratio * (point.position.x - prev_point.position.x),
                    prev_point.position.y
                    + ratio * (point.position.y - prev_point.position.y),
                )

        # Time is beyond trajectory end
        return self.points[-1].position if self.points else None

    def get_velocity_at_time(self, t: float) -> Optional[Vector2D]:
        """Get ball velocity at specific time."""
        if not self.points:
            return None

        for i, point in enumerate(self.points):
            if point.time >= t:
                if i == 0:
                    return point.velocity

                prev_point = self.points[i - 1]
                ratio = (t - prev_point.time) / (point.time - prev_point.time)

                return Vector2D(
                    prev_point.velocity.x
                    + ratio * (point.velocity.x - prev_point.velocity.x),
                    prev_point.velocity.y
                    + ratio * (point.velocity.y - prev_point.velocity.y),
                )

        return self.points[-1].velocity if self.points else None


class TrajectoryCalculator:
    """Comprehensive ball trajectory calculations."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize trajectory calculator.

        Args:
            cache_manager: Optional cache manager for performance optimization
        """
        self.cache_manager = cache_manager or CacheManager()
        self.geometry = GeometryUtils()
        self.math_utils = MathUtils()

        # Import from collision module (no longer in physics to avoid import chain)
        from backend.core.collision.geometric_collision import (
            GeometricCollisionDetector,
        )

        self.geometric_detector = GeometricCollisionDetector()

    def calculate_trajectory(
        self,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState] = None,
        quality: TrajectoryQuality = TrajectoryQuality.MEDIUM,
        time_limit: float = None,
    ) -> Trajectory:
        """Calculate complete trajectory for a ball using geometric approach.

        This uses a simple geometric trajectory calculation:
        1. Ball travels in straight lines
        2. Reflects off cushions using angle reflection
        3. Stops at ball collisions or pockets
        4. Max 2-3 bounces for reasonable prediction
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(
            ball_state, table_state, other_balls, quality
        )
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        other_balls = other_balls or []

        trajectory = Trajectory(
            ball_id=ball_state.id,
            initial_state=ball_state,
            quality=quality,
            cache_key=cache_key,
        )

        # Calculate geometric trajectory segments
        segments = self._calculate_geometric_trajectory(
            ball_state.position,
            ball_state.velocity,
            table_state,
            other_balls,
            ball_state.radius,
            ball_id=ball_state.id,
        )

        # Convert segments to trajectory points
        current_time = 0.0
        for i, segment in enumerate(segments):
            start_pos = segment["start"]
            end_pos = segment["end"]
            segment_type = segment["type"]

            # Add start point
            if i == 0:
                trajectory.points.append(
                    TrajectoryPoint(
                        time=current_time,
                        position=start_pos,
                        velocity=ball_state.velocity,
                        acceleration=Vector2D(0, 0),
                        spin=Vector2D(0, 0),
                        energy=0.5
                        * ball_state.mass
                        * ball_state.velocity.magnitude() ** 2,
                    )
                )

            # Estimate time for this segment (simplified - no friction)
            segment_distance = start_pos.distance_to(end_pos)
            velocity_mag = ball_state.velocity.magnitude()
            if velocity_mag > 0:
                segment_time = segment_distance / velocity_mag
            else:
                segment_time = 0.0
            current_time += segment_time

            # Add end point
            trajectory.points.append(
                TrajectoryPoint(
                    time=current_time,
                    position=end_pos,
                    velocity=segment.get("velocity", ball_state.velocity),
                    acceleration=Vector2D(0, 0),
                    spin=Vector2D(0, 0),
                    energy=0.5
                    * ball_state.mass
                    * segment.get("velocity", ball_state.velocity).magnitude() ** 2,
                )
            )

            # Record collision if this segment ended in one
            if segment_type == "cushion":
                collision = PredictedCollision(
                    time=current_time,
                    position=end_pos,
                    type=CollisionType.BALL_CUSHION,
                    ball1_id=ball_state.id,
                    ball2_id=None,
                    impact_angle=segment.get("angle", 0.0),
                    impact_velocity=ball_state.velocity.magnitude(),
                    resulting_velocities={
                        ball_state.id: segment.get("velocity", ball_state.velocity)
                    },
                    confidence=0.9,
                    cushion_normal=segment.get("cushion_normal"),
                )
                trajectory.collisions.append(collision)
            elif segment_type == "ball":
                collision = PredictedCollision(
                    time=current_time,
                    position=end_pos,
                    type=CollisionType.BALL_BALL,
                    ball1_id=ball_state.id,
                    ball2_id=segment.get("hit_ball_id"),
                    impact_angle=segment.get("angle", 0.0),
                    impact_velocity=ball_state.velocity.magnitude(),
                    resulting_velocities=segment.get("resulting_velocities", {}),
                    confidence=0.85,
                )
                trajectory.collisions.append(collision)
            elif segment_type == "pocket":
                collision = PredictedCollision(
                    time=current_time,
                    position=end_pos,
                    type=CollisionType.BALL_POCKET,
                    ball1_id=ball_state.id,
                    ball2_id=None,
                    impact_angle=0.0,
                    impact_velocity=ball_state.velocity.magnitude(),
                    resulting_velocities={ball_state.id: Vector2D(0, 0)},
                    confidence=0.8,
                    pocket_id=segment.get("pocket_id"),
                )
                trajectory.collisions.append(collision)
                trajectory.will_be_pocketed = True
                trajectory.pocket_id = segment.get("pocket_id")

        # Finalize trajectory
        if trajectory.points:
            trajectory.final_position = trajectory.points[-1].position
            trajectory.final_velocity = trajectory.points[-1].velocity
        else:
            trajectory.final_position = ball_state.position
            trajectory.final_velocity = Vector2D(0, 0)

        trajectory.time_to_rest = current_time
        trajectory.total_distance = self._calculate_total_distance(trajectory.points)
        trajectory.calculation_time = time.time() - start_time

        # Calculate success probability and generate alternatives
        self._calculate_success_metrics(trajectory, table_state)
        self._generate_alternative_trajectories(trajectory, table_state, other_balls)

        # Cache result
        self.cache_manager.set(cache_key, trajectory)

        return trajectory

    def _generate_cache_key(
        self,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState],
        quality: TrajectoryQuality,
    ) -> str:
        """Generate cache key for trajectory calculation."""
        ball_hash = f"{ball_state.position.x:.3f},{ball_state.position.y:.3f},{ball_state.velocity.x:.3f},{ball_state.velocity.y:.3f}"
        table_hash = f"{table_state.width}x{table_state.height},{table_state.cushion_elasticity},{table_state.surface_friction}"
        others_hash = ",".join(
            [f"{b.position.x:.3f},{b.position.y:.3f}" for b in other_balls or []]
        )
        return f"traj_{ball_hash}_{table_hash}_{others_hash}_{quality.value}"

    def _copy_ball_state(self, ball_state: BallState) -> BallState:
        """Create a copy of ball state for simulation."""
        return BallState(
            id=ball_state.id,
            position=Vector2D(ball_state.position.x, ball_state.position.y),
            velocity=Vector2D(ball_state.velocity.x, ball_state.velocity.y),
            radius=ball_state.radius,
            mass=ball_state.mass,
            spin=(
                Vector2D(ball_state.spin.x, ball_state.spin.y)
                if ball_state.spin
                else Vector2D(0, 0)
            ),
            is_cue_ball=ball_state.is_cue_ball,
            is_pocketed=ball_state.is_pocketed,
            number=ball_state.number,
        )

    def _calculate_geometric_trajectory(
        self,
        start_position: Vector2D,
        velocity: Vector2D,
        table: TableState,
        other_balls: list[BallState],
        ball_radius: float,
        max_bounces: int = 3,
        ball_id: str = "unknown",
    ) -> list[dict]:
        """Calculate geometric trajectory using line segments and reflections.

        Based on cassapa pool.cpp CalculateBallTrajectory() function (line 451).
        Returns list of trajectory segments, each containing:
        - start: Vector2D start position
        - end: Vector2D end position
        - type: "line", "cushion", "ball", or "pocket"
        - velocity: resulting velocity (for cushion bounces)
        - other metadata
        """
        segments = []
        current_pos = Vector2D(start_position.x, start_position.y)
        current_velocity = Vector2D(velocity.x, velocity.y)

        if current_velocity.magnitude() < 1e-6:
            # Ball is not moving
            return segments

        for bounce in range(max_bounces):
            # Find where the line hits: cushion, ball, or pocket
            direction = current_velocity.normalize()

            # Calculate intersection with all four cushions
            cushion_hit = self._find_cushion_intersection(
                current_pos, direction, table, ball_radius
            )

            # Check for ball-ball collision along this line
            ball_hit = self._find_ball_intersection(
                current_pos, direction, other_balls, ball_radius, moving_ball_id=ball_id
            )

            # Check for pocket
            pocket_hit = self._find_pocket_intersection(
                current_pos, direction, table, ball_radius
            )

            # Determine which happens first
            hit_distance = float("inf")
            hit_type = "none"
            hit_point = None
            hit_data = {}

            if cushion_hit:
                dist = current_pos.distance_to(cushion_hit["position"])
                if dist < hit_distance:
                    hit_distance = dist
                    hit_type = "cushion"
                    hit_point = cushion_hit["position"]
                    hit_data = cushion_hit

            if ball_hit:
                dist = current_pos.distance_to(ball_hit["position"])
                if dist < hit_distance:
                    hit_distance = dist
                    hit_type = "ball"
                    hit_point = ball_hit["position"]
                    hit_data = ball_hit

            if pocket_hit:
                dist = current_pos.distance_to(pocket_hit["position"])
                if dist < hit_distance:
                    hit_distance = dist
                    hit_type = "pocket"
                    hit_point = pocket_hit["position"]
                    hit_data = pocket_hit

            # Create segment
            if hit_type == "none":
                # No hit - trajectory continues to edge of table (shouldn't happen)
                break

            segment = {
                "start": current_pos,
                "end": hit_point,
                "type": hit_type,
            }

            if hit_type == "cushion":
                # Calculate reflected velocity using cassapa's formula (lines 612-614)
                reflected_velocity = self._calculate_geometric_reflection(
                    current_velocity, hit_data["cushion_side"]
                )
                segment["velocity"] = reflected_velocity
                segment["cushion_normal"] = hit_data.get("normal")
                segment["angle"] = math.atan2(current_velocity.y, current_velocity.x)
                segments.append(segment)

                # Continue with reflected velocity
                current_pos = hit_point
                current_velocity = reflected_velocity

            elif hit_type == "ball":
                # Ball collision - record collision and continue with new velocity
                segment["hit_ball_id"] = hit_data.get("ball_id")
                segment["angle"] = math.atan2(
                    hit_point.y - current_pos.y, hit_point.x - current_pos.x
                )
                # Calculate resulting velocities for both balls
                resulting_velocities = hit_data.get("resulting_velocities", {})
                segment["resulting_velocities"] = resulting_velocities
                segments.append(segment)

                # Get the moving ball's new velocity after collision
                new_velocity = resulting_velocities.get(ball_id)
                if new_velocity and new_velocity.magnitude() > 0.01:
                    # Continue trajectory with new velocity
                    current_pos = hit_point
                    current_velocity = new_velocity
                    # Remove the hit ball from other_balls to avoid hitting it again
                    hit_ball_id = hit_data.get("ball_id")
                    other_balls = [b for b in other_balls if b.id != hit_ball_id]
                else:
                    # Ball has stopped, end trajectory
                    break

            elif hit_type == "pocket":
                # Pocket - trajectory ends
                segment["pocket_id"] = hit_data.get("pocket_id")
                segments.append(segment)
                break

        return segments

    def _line_segment_intersection(
        self,
        ray_start: Vector2D,
        ray_dir: Vector2D,
        seg_start: Vector2D,
        seg_end: Vector2D,
    ) -> Optional[tuple[Vector2D, float]]:
        """Find intersection between a ray and a line segment.

        Args:
            ray_start: Starting point of the ray
            ray_dir: Direction vector of the ray (should be normalized)
            seg_start: Start point of the line segment
            seg_end: End point of the line segment

        Returns:
            Tuple of (intersection_point, distance_from_ray_start) or None if no intersection
        """
        # Line segment as vector
        seg_vec = Vector2D(seg_end.x - seg_start.x, seg_end.y - seg_start.y)

        # Calculate denominator for intersection equations
        denom = ray_dir.cross(seg_vec)

        # If denominator is 0, lines are parallel
        if abs(denom) < 1e-10:
            return None

        # Vector from ray start to segment start
        start_diff = Vector2D(seg_start.x - ray_start.x, seg_start.y - ray_start.y)

        # Calculate parameters
        t = start_diff.cross(seg_vec) / denom  # Parameter along ray
        u = start_diff.cross(ray_dir) / denom  # Parameter along segment

        # Check if intersection is valid (ray forward, within segment)
        if t > 0.001 and 0 <= u <= 1:
            intersection = Vector2D(
                ray_start.x + t * ray_dir.x, ray_start.y + t * ray_dir.y
            )
            return (intersection, t)

        return None

    def _find_cushion_intersection_polygon(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[dict]:
        """Find intersection with cushions defined by playing area polygon.

        Args:
            position: Current ball position
            direction: Ball direction (normalized)
            table: Table state with playing_area_corners
            ball_radius: Radius of the ball

        Returns:
            Dict with intersection info or None
        """
        corners = table.playing_area_corners
        if not corners or len(corners) != 4:
            return None

        # Define the 4 edges of the playing area
        # Corners are: top-left, top-right, bottom-right, bottom-left
        edges = [
            (corners[0], corners[1], "top"),  # Top edge
            (corners[1], corners[2], "right"),  # Right edge
            (corners[2], corners[3], "bottom"),  # Bottom edge
            (corners[3], corners[0], "left"),  # Left edge
        ]

        closest_intersection = None
        min_distance = float("inf")

        for seg_start, seg_end, cushion_name in edges:
            # Calculate edge normal (pointing inward to table)
            edge_vec = Vector2D(seg_end.x - seg_start.x, seg_end.y - seg_start.y)
            edge_normal = Vector2D(-edge_vec.y, edge_vec.x).normalize()

            # Offset the edge inward by ball_radius to account for ball size
            offset_start = Vector2D(
                seg_start.x + edge_normal.x * ball_radius,
                seg_start.y + edge_normal.y * ball_radius,
            )
            offset_end = Vector2D(
                seg_end.x + edge_normal.x * ball_radius,
                seg_end.y + edge_normal.y * ball_radius,
            )

            # Find intersection with this edge
            result = self._line_segment_intersection(
                position, direction, offset_start, offset_end
            )

            if result:
                intersection_point, distance = result
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = {
                        "position": intersection_point,
                        "distance": distance,
                        "cushion_side": cushion_name,
                        "normal": edge_normal,
                    }

        return closest_intersection

    def _find_cushion_intersection(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[dict]:
        """Find intersection point with nearest cushion using geometric collision detector.

        Based on cassapa pool.cpp CalculateBallTrajectory() lines 471-504.
        Updated to support calibrated playing area corners.
        """
        # Use geometric collision detector
        collision = self.geometric_detector.find_cushion_intersection(
            position, direction, table, ball_radius
        )

        if collision:
            return {
                "position": collision.hit_point,
                "distance": collision.distance,
                "cushion_side": collision.cushion_side,
                "normal": collision.cushion_normal,
            }

        return None

    def _find_ball_intersection(
        self,
        position: Vector2D,
        direction: Vector2D,
        other_balls: list[BallState],
        ball_radius: float,
        moving_ball_id: str = "unknown",
    ) -> Optional[dict]:
        """Find intersection with nearest ball using geometric collision detector.

        Based on cassapa pool.cpp CheckIfLineCrossesBall() lines 737-826.
        """
        # Use geometric collision detector
        collision = self.geometric_detector.find_closest_ball_collision(
            position, direction, other_balls, moving_ball_id, ball_radius
        )

        if collision:
            # Calculate resulting velocities using geometric approach
            # Find the target ball state
            target_ball = None
            for ball in other_balls:
                if ball.id == collision.ball_id:
                    target_ball = ball
                    break

            resulting_velocities = {}
            if target_ball:
                # Create temporary ball state for moving ball
                moving_ball = BallState(
                    id=moving_ball_id,
                    position=position,
                    velocity=direction,  # Direction is velocity direction
                    radius=ball_radius,
                )

                # Calculate post-collision velocities
                moving_vel, target_vel = (
                    self.geometric_detector.calculate_ball_collision_velocities(
                        moving_ball, target_ball, collision.hit_point
                    )
                )

                resulting_velocities = {
                    moving_ball_id: moving_vel,
                    collision.ball_id: target_vel,
                }

            return {
                "position": collision.hit_point,
                "ball_id": collision.ball_id,
                "distance": collision.distance,
                "resulting_velocities": resulting_velocities,
            }

        return None

    def _find_pocket_intersection(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[dict]:
        """Check if trajectory line intersects any pocket using geometric detector."""
        # Use geometric collision detector
        collision = self.geometric_detector.find_pocket_intersection(
            position, direction, table, ball_radius
        )

        if collision:
            return {
                "position": collision.hit_point,
                "pocket_id": collision.pocket_id,
                "distance": collision.distance,
            }

        return None

    def _calculate_geometric_reflection(
        self, velocity: Vector2D, cushion_side: str
    ) -> Vector2D:
        """Calculate reflected velocity using geometric collision detector.

        Based on cassapa pool.cpp lines 612-614.
        """
        # Use geometric collision detector for consistent reflection calculation
        return self.geometric_detector.calculate_geometric_reflection(
            velocity, cushion_side, elasticity=0.95
        )

    def _calculate_total_distance(self, points: list[TrajectoryPoint]) -> float:
        """Calculate total distance traveled along trajectory."""
        if len(points) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(points)):
            dx = points[i].position.x - points[i - 1].position.x
            dy = points[i].position.y - points[i - 1].position.y
            total += math.sqrt(dx * dx + dy * dy)

        return total

    def _calculate_success_metrics(
        self, trajectory: Trajectory, table: TableState
    ) -> None:
        """Calculate various success probability metrics."""
        # Basic success metric: did ball reach intended destination?
        if trajectory.will_be_pocketed:
            trajectory.success_probability = 0.8  # Base probability for pocketing

            # Adjust based on trajectory complexity
            complexity_factor = max(0.5, 1.0 - len(trajectory.collisions) * 0.1)
            trajectory.success_probability *= complexity_factor

            # Adjust based on final velocity (slower = more accurate)
            if trajectory.points:
                final_velocity = trajectory.points[-1].velocity.magnitude()
                velocity_factor = min(1.0, 2.0 / (1.0 + final_velocity))
                trajectory.success_probability *= velocity_factor
        else:
            # Position-based success (how close to intended target)
            trajectory.success_probability = 0.3  # Base for non-pocketing shots

    def _generate_alternative_trajectories(
        self,
        main_trajectory: Trajectory,
        table: TableState,
        other_balls: list[BallState],
    ) -> None:
        """Generate alternative trajectory possibilities."""
        # Skip alternative generation to avoid recursion for now
        # In a production system, this would use a separate simplified calculator
        # that doesn't generate alternatives to avoid infinite recursion

        # For now, create simple theoretical alternatives without full calculation
        base_velocity = main_trajectory.initial_state.velocity
        velocity_variations = [
            (0.95, "Softer shot"),
            (1.05, "Harder shot"),
        ]

        for factor, description in velocity_variations:
            # Create simplified trajectory branch without full calculation
            Vector2D(base_velocity.x * factor, base_velocity.y * factor)

            # Estimate final position based on modified velocity (simplified)
            estimated_final = Vector2D(
                main_trajectory.final_position.x + (factor - 1.0) * 0.1,
                main_trajectory.final_position.y + (factor - 1.0) * 0.1,
            )

            # Create trajectory branch with estimated values
            branch = TrajectoryBranch(
                probability=0.3,  # Simplified probability
                description=description,
                points=[],  # Empty for now to avoid recursion
                collisions=[],  # Empty for now to avoid recursion
                final_state=BallState(
                    id=main_trajectory.initial_state.id,
                    position=estimated_final,
                    velocity=Vector2D(0, 0),  # Assume comes to rest
                    radius=main_trajectory.initial_state.radius,
                    mass=main_trajectory.initial_state.mass,
                ),
                success_metrics={
                    "pocket_probability": main_trajectory.success_probability * factor
                },
            )

            main_trajectory.branches.append(branch)

    def predict_cue_shot(
        self,
        cue_state: CueState,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState] = None,
        quality: TrajectoryQuality = TrajectoryQuality.MEDIUM,
    ) -> Trajectory:
        """Predict trajectory from cue shot parameters."""
        # Calculate initial velocity from cue state
        force_to_velocity_ratio = 0.5  # Simplified conversion
        initial_speed = cue_state.estimated_force * force_to_velocity_ratio

        # Calculate direction from cue angle
        angle_rad = math.radians(cue_state.angle)
        initial_velocity = Vector2D(
            initial_speed * math.cos(angle_rad), initial_speed * math.sin(angle_rad)
        )

        # Create initial ball state with calculated velocity
        shot_ball = self._copy_ball_state(ball_state)
        shot_ball.velocity = initial_velocity

        # Add spin if cue impact point is off-center
        if cue_state.impact_point:
            center_offset = Vector2D(
                cue_state.impact_point.x - ball_state.position.x,
                cue_state.impact_point.y - ball_state.position.y,
            )
            # Convert offset to spin (simplified)
            spin_factor = 2.0
            shot_ball.spin = Vector2D(
                center_offset.x * spin_factor, center_offset.y * spin_factor
            )

        return self.calculate_trajectory(shot_ball, table_state, other_balls, quality)

    def export_visualization_data(self, trajectory: Trajectory) -> dict:
        """Export trajectory data for visualization."""
        # Keep points for backwards compatibility
        points = [
            {
                "time": p.time,
                "x": p.position.x,
                "y": p.position.y,
                "vx": p.velocity.x,
                "vy": p.velocity.y,
                "energy": p.energy,
            }
            for p in trajectory.points
        ]

        # Convert points to lines (consecutive point pairs) for frontend rendering
        lines = []
        for i in range(len(trajectory.points) - 1):
            start_point = trajectory.points[i]
            end_point = trajectory.points[i + 1]
            lines.append(
                {
                    "start": [start_point.position.x, start_point.position.y],
                    "end": [end_point.position.x, end_point.position.y],
                    "type": "primary",  # Type of trajectory segment
                    "confidence": trajectory.success_probability,
                }
            )

        return {
            "ball_id": trajectory.ball_id,
            "points": points,  # Keep for backwards compatibility
            "lines": lines,  # Add line segments for frontend
            "collisions": [
                {
                    "time": c.time,
                    "position": [
                        c.position.x,
                        c.position.y,
                    ],  # Frontend expects array format
                    "x": c.position.x,  # Keep for backwards compatibility
                    "y": c.position.y,  # Keep for backwards compatibility
                    "type": c.type.value,
                    "ball_id": c.ball2_id,  # Target ball ID for frontend compatibility
                    "ball1_id": c.ball1_id,  # The moving ball (e.g., cue ball)
                    "ball2_id": c.ball2_id,  # The target ball (None for cushion/pocket)
                    "angle": c.impact_angle if hasattr(c, "impact_angle") else 0.0,
                    "confidence": c.confidence,
                }
                for c in trajectory.collisions
            ],
            "success_probability": trajectory.success_probability,
            "will_be_pocketed": trajectory.will_be_pocketed,
            "pocket_id": trajectory.pocket_id,
            "total_distance": trajectory.total_distance,
            "time_to_rest": trajectory.time_to_rest,
            "alternatives": [
                {
                    "description": branch.description,
                    "probability": branch.probability,
                    "success_metrics": branch.success_metrics,
                    # Include full trajectory data for secondary balls (branches with ball_id)
                    "ball_id": branch.success_metrics.get("ball_id"),
                    "points": (
                        [
                            {
                                "time": p.time,
                                "x": p.position.x,
                                "y": p.position.y,
                                "vx": p.velocity.x,
                                "vy": p.velocity.y,
                                "energy": p.energy,
                            }
                            for p in branch.points
                        ]
                        if branch.points
                        else []
                    ),
                    "lines": (
                        [
                            {
                                "start": [
                                    branch.points[i].position.x,
                                    branch.points[i].position.y,
                                ],
                                "end": [
                                    branch.points[i + 1].position.x,
                                    branch.points[i + 1].position.y,
                                ],
                                "type": "secondary",  # Mark as secondary trajectory
                                "confidence": branch.probability,
                            }
                            for i in range(len(branch.points) - 1)
                        ]
                        if len(branch.points) > 1
                        else []
                    ),
                    "collisions": (
                        [
                            {
                                "time": c.time,
                                "position": [c.position.x, c.position.y],
                                "x": c.position.x,
                                "y": c.position.y,
                                "type": c.type.value,
                                "ball_id": c.ball2_id,
                                "ball1_id": c.ball1_id,
                                "ball2_id": c.ball2_id,
                                "angle": (
                                    c.impact_angle
                                    if hasattr(c, "impact_angle")
                                    else 0.0
                                ),
                                "confidence": c.confidence,
                            }
                            for c in branch.collisions
                        ]
                        if branch.collisions
                        else []
                    ),
                }
                for branch in trajectory.branches
            ],
        }

    def predict_multiball_cue_shot(
        self,
        cue_state: CueState,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState] = None,
        quality: TrajectoryQuality = TrajectoryQuality.MEDIUM,
        max_collision_depth: int = 5,
    ) -> MultiballTrajectoryResult:
        """Predict trajectory for cue shot including all subsequent ball movements.

        Uses geometric approach to calculate:
        1. Cue ball trajectory until it hits a target ball (or stops)
        2. Target ball trajectory after being hit (from impact point)
        3. Any subsequent collisions recursively up to max_collision_depth

        Based on cassapa pool.cpp FindBallHitByThisBall() (line 687) for recursive ball tracking.

        Args:
            cue_state: Cue stick state
            ball_state: Cue ball state
            table_state: Table state
            other_balls: Other balls on table
            quality: Trajectory quality
            max_collision_depth: Maximum collision chain depth to calculate

        Returns:
            MultiballTrajectoryResult with trajectories for all affected balls
        """
        start_time = time.time()
        other_balls = other_balls or []

        # Debug logging
        import logging

        logging.info(
            f"predict_multiball_cue_shot: cue_ball={ball_state.id}, other_balls={len(other_balls)}"
        )
        for ball in other_balls:
            logging.info(
                f"  Other ball: {ball.id} at ({ball.position.x:.1f}, {ball.position.y:.1f})"
            )

        # Create result container
        result = MultiballTrajectoryResult(
            primary_ball_id=ball_state.id,
            trajectories={},
            collision_sequence=[],
        )

        # Calculate initial cue ball trajectory
        cue_trajectory = self.predict_cue_shot(
            cue_state, ball_state, table_state, other_balls, quality
        )

        # Debug: log cue trajectory results
        logging.info(
            f"Cue trajectory has {len(cue_trajectory.points)} points, {len(cue_trajectory.collisions)} collisions"
        )
        for collision in cue_trajectory.collisions:
            logging.info(
                f"  Collision type: {collision.type}, ball1={collision.ball1_id}, ball2={collision.ball2_id}"
            )

        # Store cue ball trajectory
        result.trajectories[ball_state.id] = cue_trajectory
        result.collision_sequence.extend(cue_trajectory.collisions)

        # Process ball-ball collisions to generate secondary trajectories
        processed_ball_ids = {ball_state.id}
        balls_to_process = [(cue_trajectory, 0)]  # (trajectory, depth)

        while balls_to_process:
            current_traj, depth = balls_to_process.pop(0)

            if depth >= max_collision_depth:
                continue

            # Find ball-ball collisions in this trajectory
            for collision in current_traj.collisions:
                if collision.type != CollisionType.BALL_BALL:
                    continue

                # Debug: log collision found
                import logging

                logging.info(
                    f"Found ball-ball collision: {current_traj.ball_id} -> {collision.ball2_id}"
                )

                # Get the ball that was hit
                hit_ball_id = collision.ball2_id
                if hit_ball_id == current_traj.ball_id:
                    hit_ball_id = collision.ball1_id

                # Skip if we've already processed this ball
                if hit_ball_id in processed_ball_ids:
                    logging.info(f"Already processed ball {hit_ball_id}, skipping")
                    continue

                # Find the ball state for the hit ball
                hit_ball_state = None
                for ball in other_balls:
                    if ball.id == hit_ball_id:
                        hit_ball_state = self._copy_ball_state(ball)
                        break

                if not hit_ball_state:
                    logging.warning(
                        f"Could not find ball state for {hit_ball_id} in other_balls!"
                    )
                    continue

                logging.info(
                    f"Calculating trajectory for {hit_ball_id} from collision point"
                )

                # Calculate trajectory from collision point
                # The hit ball starts moving from the collision position
                # with a velocity determined by the collision geometry
                hit_ball_state.position = collision.position

                # Use the actual post-collision velocity from the collision physics
                # This was calculated by the geometric collision detector
                hit_ball_velocity = collision.resulting_velocities.get(hit_ball_id)
                if hit_ball_velocity:
                    hit_ball_state.velocity = hit_ball_velocity
                else:
                    # Fallback: use simplified velocity calculation
                    impact_direction = current_traj.initial_state.velocity.normalize()
                    impact_speed = current_traj.initial_state.velocity.magnitude() * 0.7
                    hit_ball_state.velocity = impact_direction * impact_speed
                    logging.warning(
                        f"No resulting velocity found for {hit_ball_id}, using fallback"
                    )

                # Calculate trajectory for this ball from impact point
                remaining_balls = [
                    b
                    for b in other_balls
                    if b.id != hit_ball_id and b.id not in processed_ball_ids
                ]

                hit_ball_trajectory = self.calculate_trajectory(
                    hit_ball_state,
                    table_state,
                    remaining_balls,
                    quality,
                )

                # Mark that this trajectory was triggered by a collision
                hit_ball_trajectory.triggered_by_ball = current_traj.ball_id

                # Store the trajectory
                result.trajectories[hit_ball_id] = hit_ball_trajectory
                result.collision_sequence.extend(hit_ball_trajectory.collisions)
                logging.info(
                    f"Stored trajectory for {hit_ball_id}: {len(hit_ball_trajectory.points)} points"
                )

                # Add to processing queue for further collisions
                balls_to_process.append((hit_ball_trajectory, depth + 1))
                processed_ball_ids.add(hit_ball_id)

        result.total_calculation_time = time.time() - start_time
        logging.info(
            f"Final result has {len(result.trajectories)} trajectories: {list(result.trajectories.keys())}"
        )
        return result

    def clear_cache(self) -> None:
        """Clear trajectory calculation cache."""
        self.cache_manager.clear()

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.cache_manager.get_stats()
