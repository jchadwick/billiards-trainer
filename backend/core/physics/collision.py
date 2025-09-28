"""Collision detection and response algorithms.

This module implements comprehensive collision detection and response for billiards physics:
- Ball-to-ball collision detection and response
- Ball-to-cushion collision detection and response
- Collision timing and positioning
- Momentum and energy conservation
- Multi-ball collision handling
- Collision prediction for trajectory calculation
- Edge case handling (simultaneous collisions, near-misses)
- Performance optimizations for real-time simulation
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from backend.core.models import BallState, TableState, Vector2D


class CollisionType(Enum):
    """Types of collisions that can occur."""

    BALL_BALL = "ball_ball"
    BALL_CUSHION = "ball_cushion"
    BALL_POCKET = "ball_pocket"


@dataclass
class CollisionPoint:
    """Collision point information."""

    position: Vector2D  # World coordinates
    normal: Vector2D  # Collision normal vector
    time: float  # Time when collision occurs (seconds)
    relative_velocity: float  # Relative velocity at collision point


@dataclass
class CollisionResult:
    """Result of collision detection or resolution."""

    collision_type: CollisionType
    time: float  # Time of collision
    point: Optional[CollisionPoint]  # Collision point details
    ball1_velocity: Vector2D  # Ball 1 velocity after collision
    ball2_velocity: Optional[Vector2D] = None  # Ball 2 velocity (if applicable)
    ball1_spin: Optional[Vector2D] = None  # Ball 1 spin after collision
    ball2_spin: Optional[Vector2D] = None  # Ball 2 spin after collision
    energy_lost: float = 0.0  # Energy lost in collision
    ball1_id: Optional[str] = None  # Ball 1 ID
    ball2_id: Optional[str] = None  # Ball 2 ID


@dataclass
class CushionSegment:
    """Table cushion segment definition."""

    start: Vector2D  # Start point
    end: Vector2D  # End point
    normal: Vector2D  # Outward normal vector
    restitution: float = 0.85  # Cushion elasticity
    friction: float = 0.2  # Cushion friction


class CollisionDetector:
    """Advanced collision detection algorithms."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize collision detector with configuration.

        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.collision_threshold = self.config.get("collision_threshold", 0.001)  # mm
        self.time_step = self.config.get("time_step", 0.001)  # seconds
        self.max_iterations = self.config.get("max_iterations", 1000)

    def detect_ball_collision(
        self, ball1: BallState, ball2: BallState, dt: float
    ) -> Optional[CollisionResult]:
        """Detect collision between two balls using continuous collision detection.

        Args:
            ball1: First ball state
            ball2: Second ball state
            dt: Time step for detection

        Returns:
            CollisionResult if collision detected, None otherwise
        """
        # Calculate relative position and velocity
        rel_pos = Vector2D(
            ball2.position.x - ball1.position.x, ball2.position.y - ball1.position.y
        )
        rel_vel = Vector2D(
            ball2.velocity.x - ball1.velocity.x, ball2.velocity.y - ball1.velocity.y
        )

        # Combined radius
        combined_radius = ball1.radius + ball2.radius

        # Current distance
        current_dist = rel_pos.magnitude()

        # Check if already colliding
        if current_dist <= combined_radius + self.collision_threshold:
            # Calculate collision point and normal
            if current_dist > 0:
                normal = rel_pos.normalize()
            else:
                # Balls are exactly on top of each other - use arbitrary normal
                normal = Vector2D(1.0, 0.0)

            collision_point = Vector2D(
                ball1.position.x + normal.x * ball1.radius,
                ball1.position.y + normal.y * ball1.radius,
            )

            # Calculate relative velocity magnitude
            rel_vel_mag = rel_vel.magnitude()

            return CollisionResult(
                collision_type=CollisionType.BALL_BALL,
                time=0.0,
                point=CollisionPoint(
                    position=collision_point,
                    normal=normal,
                    time=0.0,
                    relative_velocity=rel_vel_mag,
                ),
                ball1_velocity=ball1.velocity,
                ball2_velocity=ball2.velocity,
                ball1_id=ball1.id,
                ball2_id=ball2.id,
            )

        # Quadratic equation for continuous collision detection
        # Distance equation: |rel_pos + rel_vel * t|^2 = combined_radius^2
        a = rel_vel.x**2 + rel_vel.y**2
        b = 2 * (rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y)
        c = rel_pos.x**2 + rel_pos.y**2 - combined_radius**2

        # No relative motion - no collision
        if abs(a) < 1e-10:
            return None

        discriminant = b**2 - 4 * a * c

        # No collision if discriminant is negative
        if discriminant < 0:
            return None

        # Calculate collision times
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Find the earliest positive collision time within the time step
        collision_time = None
        if 0 <= t1 <= dt:
            collision_time = t1
        elif 0 <= t2 <= dt:
            collision_time = t2

        if collision_time is None:
            return None

        # Calculate collision point and normal
        collision_pos1 = Vector2D(
            ball1.position.x + ball1.velocity.x * collision_time,
            ball1.position.y + ball1.velocity.y * collision_time,
        )
        collision_pos2 = Vector2D(
            ball2.position.x + ball2.velocity.x * collision_time,
            ball2.position.y + ball2.velocity.y * collision_time,
        )

        # Normal vector points from ball1 to ball2
        rel_collision_pos = Vector2D(
            collision_pos2.x - collision_pos1.x, collision_pos2.y - collision_pos1.y
        )
        normal_length = rel_collision_pos.magnitude()

        if normal_length > 0:
            normal = rel_collision_pos.normalize()
        else:
            normal = Vector2D(1.0, 0.0)

        collision_point = Vector2D(
            collision_pos1.x + normal.x * ball1.radius,
            collision_pos1.y + normal.y * ball1.radius,
        )

        # Calculate relative velocity magnitude
        rel_vel_mag = rel_vel.magnitude()

        return CollisionResult(
            collision_type=CollisionType.BALL_BALL,
            time=collision_time,
            point=CollisionPoint(
                position=collision_point,
                normal=normal,
                time=collision_time,
                relative_velocity=rel_vel_mag,
            ),
            ball1_velocity=ball1.velocity,
            ball2_velocity=ball2.velocity,
            ball1_id=ball1.id,
            ball2_id=ball2.id,
        )

    def detect_cushion_collision(
        self, ball: BallState, table: TableState, dt: float
    ) -> Optional[CollisionResult]:
        """Detect collision with table cushions.

        Args:
            ball: Ball state
            table: Table state with cushion information
            dt: Time step for detection

        Returns:
            CollisionResult if collision detected, None otherwise
        """
        # Create cushion segments from table boundaries
        cushions = self._create_table_cushions(table)

        closest_collision = None
        closest_time = float("inf")

        for cushion in cushions:
            collision = self._detect_ball_line_collision(ball, cushion, dt)
            if collision and collision.time < closest_time:
                closest_collision = collision
                closest_time = collision.time

        return closest_collision

    def _create_table_cushions(self, table: TableState) -> list[CushionSegment]:
        """Create cushion segments from table boundaries.

        Args:
            table: Table state

        Returns:
            List of cushion segments
        """
        cushions = []

        # Left cushion
        cushions.append(
            CushionSegment(
                start=Vector2D(0, 0),
                end=Vector2D(0, table.height),
                normal=Vector2D(1, 0),
                restitution=table.cushion_elasticity,
                friction=table.surface_friction,
            )
        )

        # Right cushion
        cushions.append(
            CushionSegment(
                start=Vector2D(table.width, 0),
                end=Vector2D(table.width, table.height),
                normal=Vector2D(-1, 0),
                restitution=table.cushion_elasticity,
                friction=table.surface_friction,
            )
        )

        # Bottom cushion
        cushions.append(
            CushionSegment(
                start=Vector2D(0, 0),
                end=Vector2D(table.width, 0),
                normal=Vector2D(0, 1),
                restitution=table.cushion_elasticity,
                friction=table.surface_friction,
            )
        )

        # Top cushion
        cushions.append(
            CushionSegment(
                start=Vector2D(0, table.height),
                end=Vector2D(table.width, table.height),
                normal=Vector2D(0, -1),
                restitution=table.cushion_elasticity,
                friction=table.surface_friction,
            )
        )

        return cushions

    def _detect_ball_line_collision(
        self, ball: BallState, cushion: CushionSegment, dt: float
    ) -> Optional[CollisionResult]:
        """Detect collision between ball and line segment (cushion).

        Args:
            ball: Ball state
            cushion: Cushion segment
            dt: Time step

        Returns:
            CollisionResult if collision detected, None otherwise
        """
        # Line segment vector
        line_vec = Vector2D(
            cushion.end.x - cushion.start.x, cushion.end.y - cushion.start.y
        )
        line_length = line_vec.magnitude()

        if line_length < 1e-10:
            return None

        # Normalized line direction
        line_dir = line_vec.normalize()

        # Project ball position onto line
        ball_to_start = Vector2D(
            ball.position.x - cushion.start.x, ball.position.y - cushion.start.y
        )
        projection_length = ball_to_start.x * line_dir.x + ball_to_start.y * line_dir.y

        # Clamp projection to line segment
        projection_length = max(0, min(line_length, projection_length))

        # Closest point on line segment
        closest_point = Vector2D(
            cushion.start.x + line_dir.x * projection_length,
            cushion.start.y + line_dir.y * projection_length,
        )

        # Distance from ball center to closest point
        dist_vec = Vector2D(
            ball.position.x - closest_point.x, ball.position.y - closest_point.y
        )
        current_dist = dist_vec.magnitude()

        # Check if already colliding
        if current_dist <= ball.radius + self.collision_threshold:
            normal = dist_vec.normalize() if current_dist > 0 else cushion.normal

            collision_point = Vector2D(
                ball.position.x - normal.x * ball.radius,
                ball.position.y - normal.y * ball.radius,
            )

            return CollisionResult(
                collision_type=CollisionType.BALL_CUSHION,
                time=0.0,
                point=CollisionPoint(
                    position=collision_point,
                    normal=normal,
                    time=0.0,
                    relative_velocity=ball.velocity.magnitude(),
                ),
                ball1_velocity=ball.velocity,
                ball1_id=ball.id,
            )

        # Check if ball is moving towards the cushion
        vel_towards_line = -(
            dist_vec.x * ball.velocity.x + dist_vec.y * ball.velocity.y
        )
        if current_dist > 0:
            vel_towards_line /= current_dist

        if vel_towards_line <= 0:
            return None  # Moving away from cushion

        # Calculate time to collision (simplified)
        collision_time = (current_dist - ball.radius) / vel_towards_line

        if collision_time < 0 or collision_time > dt:
            return None

        # Calculate collision position
        collision_ball_pos = Vector2D(
            ball.position.x + ball.velocity.x * collision_time,
            ball.position.y + ball.velocity.y * collision_time,
        )

        # Recalculate closest point for collision position
        ball_to_start_collision = Vector2D(
            collision_ball_pos.x - cushion.start.x,
            collision_ball_pos.y - cushion.start.y,
        )
        projection_length_collision = (
            ball_to_start_collision.x * line_dir.x
            + ball_to_start_collision.y * line_dir.y
        )
        projection_length_collision = max(
            0, min(line_length, projection_length_collision)
        )

        closest_point_collision = Vector2D(
            cushion.start.x + line_dir.x * projection_length_collision,
            cushion.start.y + line_dir.y * projection_length_collision,
        )

        # Normal at collision
        dist_vec_collision = Vector2D(
            collision_ball_pos.x - closest_point_collision.x,
            collision_ball_pos.y - closest_point_collision.y,
        )
        dist_collision = dist_vec_collision.magnitude()

        if dist_collision > 0:
            normal = dist_vec_collision.normalize()
        else:
            normal = cushion.normal

        collision_point = Vector2D(
            collision_ball_pos.x - normal.x * ball.radius,
            collision_ball_pos.y - normal.y * ball.radius,
        )

        return CollisionResult(
            collision_type=CollisionType.BALL_CUSHION,
            time=collision_time,
            point=CollisionPoint(
                position=collision_point,
                normal=normal,
                time=collision_time,
                relative_velocity=ball.velocity.magnitude(),
            ),
            ball1_velocity=ball.velocity,
            ball1_id=ball.id,
        )

    def detect_multiple_collisions(
        self, balls: list[BallState], table: TableState, dt: float
    ) -> list[CollisionResult]:
        """Detect all collisions in a time step.

        Args:
            balls: List of ball states
            table: Table state
            dt: Time step

        Returns:
            List of collision results sorted by time
        """
        collisions = []

        # Ball-to-ball collisions
        for i in range(len(balls)):
            if balls[i].is_pocketed:
                continue
            for j in range(i + 1, len(balls)):
                if balls[j].is_pocketed:
                    continue
                collision = self.detect_ball_collision(balls[i], balls[j], dt)
                if collision:
                    collisions.append(collision)

        # Ball-to-cushion collisions
        for ball in balls:
            if ball.is_pocketed:
                continue
            collision = self.detect_cushion_collision(ball, table, dt)
            if collision:
                collisions.append(collision)

        # Sort by collision time
        collisions.sort(key=lambda c: c.time)

        return collisions


class CollisionResolver:
    """Advanced collision response calculations with realistic physics."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize collision resolver with configuration.

        Args:
            config: Configuration dictionary with physics parameters
        """
        self.config = config or {}
        self.energy_loss_factor = self.config.get("energy_loss_factor", 0.02)
        self.spin_transfer_factor = self.config.get("spin_transfer_factor", 0.3)
        self.default_restitution = self.config.get("default_restitution", 0.95)
        self.friction_coefficient = self.config.get("friction_coefficient", 0.1)

    def resolve_ball_collision(
        self, ball1: BallState, ball2: BallState, collision: CollisionResult
    ) -> CollisionResult:
        """Calculate velocities after ball-to-ball collision using realistic physics.

        Args:
            ball1: First ball state
            ball2: Second ball state
            collision: Collision information

        Returns:
            Updated CollisionResult with new velocities and spins
        """
        if not collision.point:
            raise ValueError("Collision point required for ball collision resolution")

        normal = collision.point.normal

        # Calculate relative velocity
        rel_vel = Vector2D(
            ball1.velocity.x - ball2.velocity.x, ball1.velocity.y - ball2.velocity.y
        )

        # Relative velocity in collision normal direction
        vel_along_normal = rel_vel.x * normal.x + rel_vel.y * normal.y

        # Do not resolve if velocities are separating
        if vel_along_normal > 0:
            collision.ball1_velocity = ball1.velocity
            collision.ball2_velocity = ball2.velocity
            return collision

        # Calculate restitution
        restitution = self.default_restitution

        # Calculate impulse scalar using proper physics formula
        impulse_scalar = -(1 + restitution) * vel_along_normal
        impulse_scalar /= 1 / ball1.mass + 1 / ball2.mass

        # Calculate impulse vector
        impulse = Vector2D(impulse_scalar * normal.x, impulse_scalar * normal.y)

        # Update velocities using conservation of momentum
        new_vel1 = Vector2D(
            ball1.velocity.x + impulse.x / ball1.mass,
            ball1.velocity.y + impulse.y / ball1.mass,
        )
        new_vel2 = Vector2D(
            ball2.velocity.x - impulse.x / ball2.mass,
            ball2.velocity.y - impulse.y / ball2.mass,
        )

        # Calculate energy loss
        kinetic_before = (
            0.5 * ball1.mass * ball1.velocity.magnitude() ** 2
            + 0.5 * ball2.mass * ball2.velocity.magnitude() ** 2
        )
        kinetic_after = (
            0.5 * ball1.mass * new_vel1.magnitude() ** 2
            + 0.5 * ball2.mass * new_vel2.magnitude() ** 2
        )
        energy_lost = kinetic_before - kinetic_after

        # Handle spin transfer (simplified model)
        new_spin1 = ball1.spin
        new_spin2 = ball2.spin

        if (
            abs(vel_along_normal) > 0.1 and ball1.spin and ball2.spin
        ):  # Only transfer spin for significant collisions
            # Transfer some spin between balls
            spin_transfer = self.spin_transfer_factor * abs(vel_along_normal)

            # Tangential component of relative velocity affects spin
            tangent = Vector2D(-normal.y, normal.x)  # Perpendicular to normal
            rel_vel_tangent = rel_vel.x * tangent.x + rel_vel.y * tangent.y

            # Apply spin changes (simplified - real physics would be more complex)
            spin_change = rel_vel_tangent * spin_transfer / ball1.radius
            new_spin1 = Vector2D(ball1.spin.x, ball1.spin.y + spin_change)
            new_spin2 = Vector2D(ball2.spin.x, ball2.spin.y - spin_change)

        # Update collision result
        collision.ball1_velocity = new_vel1
        collision.ball2_velocity = new_vel2
        collision.ball1_spin = new_spin1
        collision.ball2_spin = new_spin2
        collision.energy_lost = energy_lost

        return collision

    def resolve_cushion_collision(
        self, ball: BallState, collision: CollisionResult
    ) -> CollisionResult:
        """Calculate velocity after ball-to-cushion collision.

        Args:
            ball: Ball state
            collision: Collision information

        Returns:
            Updated CollisionResult with new velocity and spin
        """
        if not collision.point:
            raise ValueError(
                "Collision point required for cushion collision resolution"
            )

        normal = collision.point.normal

        # Velocity components
        vel_normal = ball.velocity.x * normal.x + ball.velocity.y * normal.y
        vel_tangent = ball.velocity.x * (-normal.y) + ball.velocity.y * normal.x

        # Apply restitution to normal component
        cushion_restitution = 0.85  # Default cushion restitution
        new_vel_normal = -vel_normal * cushion_restitution

        # Apply friction to tangential component
        friction_force = self.friction_coefficient * abs(vel_normal)
        if abs(vel_tangent) > friction_force:
            new_vel_tangent = vel_tangent - math.copysign(friction_force, vel_tangent)
        else:
            new_vel_tangent = 0.0

        # Combine components
        new_velocity = Vector2D(
            new_vel_normal * normal.x + new_vel_tangent * (-normal.y),
            new_vel_normal * normal.y + new_vel_tangent * normal.x,
        )

        # Calculate energy loss
        kinetic_before = 0.5 * ball.mass * ball.velocity.magnitude() ** 2
        kinetic_after = 0.5 * ball.mass * new_velocity.magnitude() ** 2
        energy_lost = kinetic_before - kinetic_after

        # Update spin (cushion interaction affects spin)
        new_spin = ball.spin
        if abs(vel_tangent) > 0.1 and ball.spin:
            # Cushion friction affects ball spin
            spin_change = friction_force / ball.radius
            new_spin = Vector2D(
                ball.spin.x, ball.spin.y + math.copysign(spin_change, vel_tangent)
            )

        # Update collision result
        collision.ball1_velocity = new_velocity
        collision.ball1_spin = new_spin
        collision.energy_lost = energy_lost

        return collision

    def resolve_simultaneous_collisions(
        self, collisions: list[CollisionResult], balls: list[BallState]
    ) -> list[CollisionResult]:
        """Handle simultaneous or near-simultaneous collisions.

        Args:
            collisions: List of collisions to resolve
            balls: Current ball states

        Returns:
            List of resolved collision results
        """
        if not collisions:
            return []

        # Group collisions by time (within small tolerance)
        time_tolerance = 0.0001  # 0.1ms tolerance
        collision_groups = []
        current_group = [collisions[0]]
        current_time = collisions[0].time

        for collision in collisions[1:]:
            if abs(collision.time - current_time) <= time_tolerance:
                current_group.append(collision)
            else:
                collision_groups.append(current_group)
                current_group = [collision]
                current_time = collision.time
        collision_groups.append(current_group)

        resolved_collisions = []

        # Resolve each group
        for group in collision_groups:
            if len(group) == 1:
                # Single collision - resolve normally
                collision = group[0]
                if collision.collision_type == CollisionType.BALL_BALL:
                    ball1 = self._find_ball_by_id(balls, collision.ball1_id)
                    ball2 = self._find_ball_by_id(balls, collision.ball2_id)
                    if ball1 and ball2:
                        resolved_collision = self.resolve_ball_collision(
                            ball1, ball2, collision
                        )
                    else:
                        resolved_collision = collision
                else:  # BALL_CUSHION
                    ball = self._find_ball_by_id(balls, collision.ball1_id)
                    if ball:
                        resolved_collision = self.resolve_cushion_collision(
                            ball, collision
                        )
                    else:
                        resolved_collision = collision
                resolved_collisions.append(resolved_collision)
            else:
                # Multiple simultaneous collisions - use iterative approach
                resolved_group = self._resolve_simultaneous_group(group, balls)
                resolved_collisions.extend(resolved_group)

        return resolved_collisions

    def _find_ball_by_id(
        self, balls: list[BallState], ball_id: Optional[str]
    ) -> Optional[BallState]:
        """Find ball by ID in list."""
        if ball_id is None:
            return None
        for ball in balls:
            if ball.id == ball_id:
                return ball
        return None

    def _resolve_simultaneous_group(
        self, collisions: list[CollisionResult], balls: list[BallState]
    ) -> list[CollisionResult]:
        """Resolve a group of simultaneous collisions using iterative method.

        Args:
            collisions: Simultaneous collisions
            balls: Ball states

        Returns:
            List of resolved collisions
        """
        # For simplicity, resolve collisions in order of impact magnitude
        # A more sophisticated approach would use iterative impulse resolution

        # Sort by relative velocity (higher impact first)
        sorted_collisions = sorted(
            collisions,
            key=lambda c: c.point.relative_velocity if c.point else 0,
            reverse=True,
        )

        resolved = []
        ball_dict = {ball.id: ball for ball in balls}

        for collision in sorted_collisions:
            if collision.collision_type == CollisionType.BALL_BALL:
                ball1 = ball_dict.get(collision.ball1_id)
                ball2 = ball_dict.get(collision.ball2_id)
                if ball1 and ball2:
                    resolved_collision = self.resolve_ball_collision(
                        ball1, ball2, collision
                    )
                    # Update ball states for subsequent collision resolutions
                    ball1.velocity = resolved_collision.ball1_velocity
                    ball2.velocity = resolved_collision.ball2_velocity
                    if resolved_collision.ball1_spin:
                        ball1.spin = resolved_collision.ball1_spin
                    if resolved_collision.ball2_spin:
                        ball2.spin = resolved_collision.ball2_spin
                else:
                    resolved_collision = collision
            else:  # BALL_CUSHION
                ball = ball_dict.get(collision.ball1_id)
                if ball:
                    resolved_collision = self.resolve_cushion_collision(ball, collision)
                    # Update ball state
                    ball.velocity = resolved_collision.ball1_velocity
                    if resolved_collision.ball1_spin:
                        ball.spin = resolved_collision.ball1_spin
                else:
                    resolved_collision = collision

            resolved.append(resolved_collision)

        return resolved


class CollisionPredictor:
    """Collision prediction for trajectory calculation."""

    def __init__(self, detector: CollisionDetector, resolver: CollisionResolver):
        """Initialize collision predictor.

        Args:
            detector: Collision detector instance
            resolver: Collision resolver instance
        """
        self.detector = detector
        self.resolver = resolver

    def predict_trajectory_collisions(
        self,
        ball: BallState,
        other_balls: list[BallState],
        table: TableState,
        max_time: float = 10.0,
        time_step: float = 0.01,
    ) -> list[CollisionResult]:
        """Predict all collisions along a ball's trajectory.

        Args:
            ball: Primary ball to predict trajectory for
            other_balls: Other balls on table
            table: Table state
            max_time: Maximum prediction time
            time_step: Time step for prediction

        Returns:
            List of predicted collisions in chronological order
        """
        predicted_collisions = []
        current_ball = BallState(
            id=ball.id,
            position=Vector2D(ball.position.x, ball.position.y),
            velocity=Vector2D(ball.velocity.x, ball.velocity.y),
            radius=ball.radius,
            mass=ball.mass,
            spin=Vector2D(ball.spin.x, ball.spin.y) if ball.spin else None,
            is_cue_ball=ball.is_cue_ball,
            is_pocketed=ball.is_pocketed,
            number=ball.number,
        )
        current_time = 0.0
        all_balls = [current_ball] + [b for b in other_balls if not b.is_pocketed]

        while current_time < max_time:
            remaining_time = min(time_step, max_time - current_time)

            # Detect collisions in this time step
            collisions = self.detector.detect_multiple_collisions(
                all_balls, table, remaining_time
            )

            if not collisions:
                # No collisions - advance time step
                current_time += remaining_time
                # Update ball position (simplified - no forces)
                current_ball.position.x += current_ball.velocity.x * remaining_time
                current_ball.position.y += current_ball.velocity.y * remaining_time
                continue

            # Process first collision
            first_collision = collisions[0]
            collision_time = current_time + first_collision.time

            # Resolve collision
            if first_collision.collision_type == CollisionType.BALL_BALL:
                ball1 = self._find_ball_in_list(all_balls, first_collision.ball1_id)
                ball2 = self._find_ball_in_list(all_balls, first_collision.ball2_id)
                if ball1 and ball2:
                    resolved = self.resolver.resolve_ball_collision(
                        ball1, ball2, first_collision
                    )
                else:
                    resolved = first_collision
            else:  # BALL_CUSHION
                ball_in_collision = self._find_ball_in_list(
                    all_balls, first_collision.ball1_id
                )
                if ball_in_collision:
                    resolved = self.resolver.resolve_cushion_collision(
                        ball_in_collision, first_collision
                    )
                else:
                    resolved = first_collision

            # Update collision time
            resolved.time = collision_time
            predicted_collisions.append(resolved)

            # Update ball states after collision
            if resolved.ball1_id == current_ball.id:
                current_ball.position.x += (
                    current_ball.velocity.x * first_collision.time
                )
                current_ball.position.y += (
                    current_ball.velocity.y * first_collision.time
                )
                current_ball.velocity = resolved.ball1_velocity
                if resolved.ball1_spin:
                    current_ball.spin = resolved.ball1_spin

            current_time += first_collision.time

            # Stop if ball velocity becomes very small
            if current_ball.velocity.magnitude() < 0.01:
                break

        return predicted_collisions

    def _find_ball_in_list(
        self, balls: list[BallState], ball_id: Optional[str]
    ) -> Optional[BallState]:
        """Find ball by ID in list."""
        if ball_id is None:
            return None
        for ball in balls:
            if ball.id == ball_id:
                return ball
        return None


class CollisionOptimizer:
    """Performance optimizations for real-time collision detection."""

    def __init__(self):
        """Initialize collision optimizer."""
        self.spatial_grid_size = 100.0  # mm
        self.spatial_grid = {}

    def build_spatial_grid(self, balls: list[BallState]):
        """Build spatial partitioning grid for efficient collision detection.

        Args:
            balls: List of ball states
        """
        self.spatial_grid.clear()

        for i, ball in enumerate(balls):
            if ball.is_pocketed:
                continue

            grid_x = int(ball.position.x / self.spatial_grid_size)
            grid_y = int(ball.position.y / self.spatial_grid_size)

            # Add ball to multiple grid cells based on radius
            radius_cells = int(math.ceil(ball.radius / self.spatial_grid_size))

            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    cell = (grid_x + dx, grid_y + dy)
                    if cell not in self.spatial_grid:
                        self.spatial_grid[cell] = []
                    self.spatial_grid[cell].append(i)

    def get_collision_candidates(
        self, ball_index: int, balls: list[BallState]
    ) -> list[int]:
        """Get potential collision candidates using spatial partitioning.

        Args:
            ball_index: Index of ball to find candidates for
            balls: List of ball states

        Returns:
            List of ball indices that could potentially collide
        """
        ball = balls[ball_index]
        grid_x = int(ball.position.x / self.spatial_grid_size)
        grid_y = int(ball.position.y / self.spatial_grid_size)

        candidates = set()

        # Check surrounding cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (grid_x + dx, grid_y + dy)
                if cell in self.spatial_grid:
                    for candidate_index in self.spatial_grid[cell]:
                        if candidate_index != ball_index:
                            candidates.add(candidate_index)

        return list(candidates)
