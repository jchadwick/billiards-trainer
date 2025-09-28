"""Main physics engine for ball trajectory calculations."""

import math
from dataclasses import dataclass
from typing import Any, Optional

from backend.core.game_state import BallState, TableState, Vector2D
from backend.core.models import Collision


class PhysicsConstants:
    """Physical constants for billiard ball simulation."""

    # Ball properties - all in SI units (meters)
    BALL_RADIUS = 0.028575  # m (57.15mm diameter / 2)
    BALL_MASS = 0.17  # kg (standard pool ball)

    # Table properties
    TABLE_FRICTION_COEFFICIENT = 0.2  # Rolling friction
    CUSHION_RESTITUTION = 0.85  # Energy retained in cushion bounce
    BALL_RESTITUTION = 0.95  # Energy retained in ball-ball collision

    # Physics simulation
    GRAVITY = 9.81  # m/s^2
    TIME_STEP = 0.001  # s (1ms for accurate simulation)
    MIN_VELOCITY = 0.001  # m/s (below this, ball is considered stopped)

    # Pocket properties
    POCKET_RADIUS = 0.0635  # m (standard corner pocket)
    POCKET_CAPTURE_SPEED = 2.0  # m/s (max speed for clean pocket entry)


@dataclass
class TrajectoryPoint:
    """Single point in a ball's trajectory."""

    time: float
    position: Vector2D
    velocity: Vector2D
    collision_type: Optional[str] = None  # "ball", "cushion", "pocket", None


class PhysicsEngine:
    """Main physics simulation engine for billiard ball trajectories."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize physics engine with configuration."""
        self.config = config or {}
        self.constants = PhysicsConstants()

        # Simulation parameters (can be overridden by config)
        self.time_step = self.config.get("time_step", self.constants.TIME_STEP)
        self.max_simulation_time = self.config.get(
            "max_simulation_time", 30.0
        )  # seconds
        self.friction_enabled = self.config.get("friction_enabled", True)
        self.spin_enabled = self.config.get("spin_enabled", False)  # Advanced feature

    def calculate_trajectory(
        self,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState],
        time_limit: float = 5.0,
    ) -> list[TrajectoryPoint]:
        """Calculate complete ball trajectory using physics simulation.

        Args:
            ball_state: Initial state of the ball to simulate
            table_state: Table configuration and properties
            other_balls: List of other balls on the table
            time_limit: Maximum simulation time in seconds

        Returns:
            List of trajectory points showing ball path over time
        """
        if ball_state.velocity.magnitude() < self.constants.MIN_VELOCITY:
            return [TrajectoryPoint(0.0, ball_state.position, ball_state.velocity)]

        trajectory = []
        current_ball = BallState(
            id=ball_state.id,
            position=Vector2D(ball_state.position.x, ball_state.position.y),
            velocity=Vector2D(ball_state.velocity.x, ball_state.velocity.y),
            radius=ball_state.radius,
            mass=ball_state.mass,
            spin=ball_state.spin,
            is_cue_ball=ball_state.is_cue_ball,
            is_pocketed=ball_state.is_pocketed,
            number=ball_state.number,
        )

        # Create copies of other balls to track their positions during simulation
        sim_balls = [
            self._copy_ball_state(ball) for ball in other_balls if not ball.is_pocketed
        ]

        current_time = 0.0

        while current_time < min(time_limit, self.max_simulation_time):
            # Record current state
            trajectory.append(
                TrajectoryPoint(
                    time=current_time,
                    position=Vector2D(current_ball.position.x, current_ball.position.y),
                    velocity=Vector2D(current_ball.velocity.x, current_ball.velocity.y),
                )
            )

            # Check if ball has stopped
            if current_ball.velocity.magnitude() < self.constants.MIN_VELOCITY:
                break

            # Detect collisions in next time step
            collision = self._detect_next_collision(
                current_ball, sim_balls, table_state, self.time_step
            )

            if collision is None:
                # No collision, advance ball using numerical integration
                self._integrate_motion(current_ball, table_state, self.time_step)
                current_time += self.time_step
            else:
                # Collision detected, advance to collision time
                collision_time = collision.time
                if collision_time > 0:
                    self._integrate_motion(current_ball, table_state, collision_time)
                    current_time += collision_time

                # Handle collision
                self._handle_collision(current_ball, sim_balls, collision, table_state)

                # Record collision point
                trajectory.append(
                    TrajectoryPoint(
                        time=current_time,
                        position=Vector2D(
                            current_ball.position.x, current_ball.position.y
                        ),
                        velocity=Vector2D(
                            current_ball.velocity.x, current_ball.velocity.y
                        ),
                        collision_type=collision.type,
                    )
                )

                # Check if ball was pocketed
                if collision.type == "pocket":
                    current_ball.is_pocketed = True
                    break

                # Continue with remaining time in step
                remaining_time = self.time_step - collision_time
                if remaining_time > 0:
                    self._integrate_motion(current_ball, table_state, remaining_time)

                current_time += remaining_time

        return trajectory

    def _copy_ball_state(self, ball: BallState) -> BallState:
        """Create a deep copy of a ball state for simulation."""
        return BallState(
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

    def _integrate_motion(self, ball: BallState, table: TableState, dt: float) -> None:
        """Integrate ball motion using numerical methods (Euler integration)
        Applies friction and updates position/velocity.
        """
        if not self.friction_enabled:
            # Simple ballistic motion
            ball.position.x += ball.velocity.x * dt
            ball.position.y += ball.velocity.y * dt
            return

        # Calculate friction force
        velocity_magnitude = ball.velocity.magnitude()
        if velocity_magnitude < self.constants.MIN_VELOCITY:
            ball.velocity.x = 0.0
            ball.velocity.y = 0.0
            return

        # Rolling friction deceleration
        friction_deceleration = table.surface_friction * self.constants.GRAVITY

        # Apply friction (opposing velocity direction)
        velocity_unit = ball.velocity.normalize()
        friction_force_x = -friction_deceleration * velocity_unit.x
        friction_force_y = -friction_deceleration * velocity_unit.y

        # Update velocity (F = ma, a = F/m)
        acceleration_x = friction_force_x
        acceleration_y = friction_force_y

        # Euler integration
        new_velocity_x = ball.velocity.x + acceleration_x * dt
        new_velocity_y = ball.velocity.y + acceleration_y * dt

        # Check if friction would reverse velocity (ball stops)
        if (ball.velocity.x > 0 and new_velocity_x < 0) or (
            ball.velocity.x < 0 and new_velocity_x > 0
        ):
            new_velocity_x = 0.0
        if (ball.velocity.y > 0 and new_velocity_y < 0) or (
            ball.velocity.y < 0 and new_velocity_y > 0
        ):
            new_velocity_y = 0.0

        # Update position using average velocity
        avg_velocity_x = (ball.velocity.x + new_velocity_x) / 2.0
        avg_velocity_y = (ball.velocity.y + new_velocity_y) / 2.0

        ball.position.x += avg_velocity_x * dt
        ball.position.y += avg_velocity_y * dt

        # Update velocity
        ball.velocity.x = new_velocity_x
        ball.velocity.y = new_velocity_y

    def _detect_next_collision(
        self,
        ball: BallState,
        other_balls: list[BallState],
        table: TableState,
        max_time: float,
    ) -> Optional[Collision]:
        """Detect the next collision (ball-ball, ball-cushion, or ball-pocket)
        within the given time frame.
        """
        closest_collision = None
        closest_time = max_time

        # Check ball-ball collisions
        for other_ball in other_balls:
            if other_ball.id == ball.id or other_ball.is_pocketed:
                continue

            collision_time = self._calculate_ball_collision_time(ball, other_ball)
            if collision_time is not None and 0 < collision_time < closest_time:
                closest_time = collision_time
                closest_collision = Collision(
                    time=collision_time,
                    position=self._calculate_collision_position(ball, collision_time),
                    ball1_id=ball.id,
                    ball2_id=other_ball.id,
                    type="ball",
                )

        # Check cushion collisions
        cushion_collision = self._detect_cushion_collision(ball, table, max_time)
        if cushion_collision and cushion_collision.time < closest_time:
            closest_time = cushion_collision.time
            closest_collision = cushion_collision

        # Check pocket collisions
        pocket_collision = self._detect_pocket_collision(ball, table, max_time)
        if pocket_collision and pocket_collision.time < closest_time:
            closest_collision = pocket_collision

        return closest_collision

    def _calculate_ball_collision_time(
        self, ball1: BallState, ball2: BallState
    ) -> Optional[float]:
        """Calculate time until collision between two balls
        Returns None if no collision will occur.
        """
        # Relative position and velocity
        dx = ball2.position.x - ball1.position.x
        dy = ball2.position.y - ball1.position.y
        dvx = ball2.velocity.x - ball1.velocity.x
        dvy = ball2.velocity.y - ball1.velocity.y

        # Quadratic equation coefficients for collision detection
        # |relative_position + relative_velocity * t| = ball1.radius + ball2.radius
        a = dvx * dvx + dvy * dvy
        b = 2 * (dx * dvx + dy * dvy)
        c = dx * dx + dy * dy - (ball1.radius + ball2.radius) ** 2

        # Check if balls are moving apart or parallel
        if a == 0:
            return None

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None  # No collision

        # Get the earlier collision time
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        # Return the smallest positive time
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return None  # Collision in the past

    def _calculate_collision_position(self, ball: BallState, time: float) -> Vector2D:
        """Calculate ball position at collision time."""
        return Vector2D(
            ball.position.x + ball.velocity.x * time,
            ball.position.y + ball.velocity.y * time,
        )

    def _detect_cushion_collision(
        self, ball: BallState, table: TableState, max_time: float
    ) -> Optional[Collision]:
        """Detect collision with table cushions."""
        # Calculate time to hit each cushion
        times = []

        # Left cushion (x = ball.radius)
        if ball.velocity.x < 0:
            t = (ball.radius - ball.position.x) / ball.velocity.x
            if 0 < t <= max_time:
                y_at_collision = ball.position.y + ball.velocity.y * t
                if 0 <= y_at_collision <= table.height:
                    times.append((t, "left", Vector2D(ball.radius, y_at_collision)))

        # Right cushion (x = table.width - ball.radius)
        if ball.velocity.x > 0:
            t = (table.width - ball.radius - ball.position.x) / ball.velocity.x
            if 0 < t <= max_time:
                y_at_collision = ball.position.y + ball.velocity.y * t
                if 0 <= y_at_collision <= table.height:
                    times.append(
                        (
                            t,
                            "right",
                            Vector2D(table.width - ball.radius, y_at_collision),
                        )
                    )

        # Bottom cushion (y = ball.radius)
        if ball.velocity.y < 0:
            t = (ball.radius - ball.position.y) / ball.velocity.y
            if 0 < t <= max_time:
                x_at_collision = ball.position.x + ball.velocity.x * t
                if 0 <= x_at_collision <= table.width:
                    times.append((t, "bottom", Vector2D(x_at_collision, ball.radius)))

        # Top cushion (y = table.height - ball.radius)
        if ball.velocity.y > 0:
            t = (table.height - ball.radius - ball.position.y) / ball.velocity.y
            if 0 < t <= max_time:
                x_at_collision = ball.position.x + ball.velocity.x * t
                if 0 <= x_at_collision <= table.width:
                    times.append(
                        (t, "top", Vector2D(x_at_collision, table.height - ball.radius))
                    )

        if not times:
            return None

        # Return the earliest collision
        earliest = min(times, key=lambda x: x[0])
        return Collision(
            time=earliest[0],
            position=earliest[2],
            ball1_id=ball.id,
            ball2_id=None,
            type="cushion",
        )

    def _detect_pocket_collision(
        self, ball: BallState, table: TableState, max_time: float
    ) -> Optional[Collision]:
        """Detect collision with table pockets."""
        for pocket_pos in table.pocket_positions:
            # Calculate time to reach pocket center
            dx = pocket_pos.x - ball.position.x
            dy = pocket_pos.y - ball.position.y

            # Check if ball is heading toward pocket
            if ball.velocity.x * dx + ball.velocity.y * dy <= 0:
                continue  # Ball moving away from pocket

            # Calculate closest approach time
            velocity_mag_sq = ball.velocity.x**2 + ball.velocity.y**2
            if velocity_mag_sq == 0:
                continue

            t = (dx * ball.velocity.x + dy * ball.velocity.y) / velocity_mag_sq

            if 0 < t <= max_time:
                # Position at closest approach
                x_closest = ball.position.x + ball.velocity.x * t
                y_closest = ball.position.y + ball.velocity.y * t

                # Distance to pocket center at closest approach
                distance_to_pocket = math.sqrt(
                    (x_closest - pocket_pos.x) ** 2 + (y_closest - pocket_pos.y) ** 2
                )

                # Check if ball enters pocket (with some tolerance for partial entry)
                if distance_to_pocket <= table.pocket_radius - ball.radius * 0.5:
                    return Collision(
                        time=t,
                        position=Vector2D(x_closest, y_closest),
                        ball1_id=ball.id,
                        ball2_id=None,
                        type="pocket",
                    )

        return None

    def _handle_collision(
        self,
        ball: BallState,
        other_balls: list[BallState],
        collision: Collision,
        table: TableState,
    ) -> dict[str, Any]:
        """Handle collision and update ball velocities."""
        if collision.type == "ball":
            return self._resolve_ball_collision(ball, other_balls, collision)
        elif collision.type == "cushion":
            return self._resolve_cushion_collision(ball, collision, table)
        elif collision.type == "pocket":
            return self._resolve_pocket_collision(ball, collision)

        return {}

    def _resolve_ball_collision(
        self, ball1: BallState, other_balls: list[BallState], collision: Collision
    ) -> dict[str, Any]:
        """Resolve collision between two balls using conservation of momentum and energy."""
        # Find the other ball
        ball2 = None
        for ball in other_balls:
            if ball.id == collision.ball2_id:
                ball2 = ball
                break

        if ball2 is None:
            return {}

        # Collision normal (from ball1 to ball2)
        dx = ball2.position.x - ball1.position.x
        dy = ball2.position.y - ball1.position.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance == 0:
            return {}  # Balls at same position, skip

        # Unit normal vector
        nx = dx / distance
        ny = dy / distance

        # Unit tangent vector
        tx = -ny
        ty = nx

        # Project velocities onto normal and tangent directions
        v1n = ball1.velocity.x * nx + ball1.velocity.y * ny
        v1t = ball1.velocity.x * tx + ball1.velocity.y * ty
        v2n = ball2.velocity.x * nx + ball2.velocity.y * ny
        v2t = ball2.velocity.x * tx + ball2.velocity.y * ty

        # Conservation of momentum in normal direction (assuming equal masses)
        m1, m2 = ball1.mass, ball2.mass
        v1n_new = ((m1 - m2) * v1n + 2 * m2 * v2n) / (m1 + m2)
        v2n_new = ((m2 - m1) * v2n + 2 * m1 * v1n) / (m1 + m2)

        # Apply restitution
        restitution = self.constants.BALL_RESTITUTION
        v1n_new *= restitution
        v2n_new *= restitution

        # Tangent velocities remain unchanged (no friction)
        v1t_new = v1t
        v2t_new = v2t

        # Convert back to x,y coordinates
        ball1.velocity.x = v1n_new * nx + v1t_new * tx
        ball1.velocity.y = v1n_new * ny + v1t_new * ty
        ball2.velocity.x = v2n_new * nx + v2t_new * tx
        ball2.velocity.y = v2n_new * ny + v2t_new * ty

        # Separate balls to prevent overlap
        overlap = (ball1.radius + ball2.radius) - distance
        if overlap > 0:
            separation = overlap / 2.0
            ball1.position.x -= separation * nx
            ball1.position.y -= separation * ny
            ball2.position.x += separation * nx
            ball2.position.y += separation * ny

        return {
            "type": "ball_collision",
            "ball1_velocity": Vector2D(ball1.velocity.x, ball1.velocity.y),
            "ball2_velocity": Vector2D(ball2.velocity.x, ball2.velocity.y),
        }

    def _resolve_cushion_collision(
        self, ball: BallState, collision: Collision, table: TableState
    ) -> dict[str, Any]:
        """Resolve collision with table cushion."""
        # Determine which cushion was hit based on collision position
        x, y = collision.position.x, collision.position.y

        if x <= ball.radius or x >= table.width - ball.radius:  # Left cushion
            ball.velocity.x = -ball.velocity.x * table.cushion_elasticity
        elif y <= ball.radius or y >= table.height - ball.radius:  # Bottom cushion
            ball.velocity.y = -ball.velocity.y * table.cushion_elasticity

        # Ensure ball is not inside cushion
        ball.position.x = max(
            ball.radius, min(table.width - ball.radius, ball.position.x)
        )
        ball.position.y = max(
            ball.radius, min(table.height - ball.radius, ball.position.y)
        )

        return {
            "type": "cushion_collision",
            "velocity": Vector2D(ball.velocity.x, ball.velocity.y),
        }

    def _resolve_pocket_collision(
        self, ball: BallState, collision: Collision
    ) -> dict[str, Any]:
        """Resolve ball entering pocket."""
        ball.velocity.x = 0.0
        ball.velocity.y = 0.0
        ball.is_pocketed = True

        return {"type": "pocket", "ball_id": ball.id, "pocketed": True}
