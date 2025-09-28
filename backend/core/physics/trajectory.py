"""Comprehensive trajectory calculation algorithms for billiards simulation."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.core.game_state import BallState, CueState, TableState, Vector2D
from backend.core.utils.cache import CacheManager
from backend.core.utils.geometry import GeometryUtils
from backend.core.utils.math import MathUtils


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
    """A single point along a ball's trajectory."""

    time: float  # Time from start (seconds)
    position: Vector2D  # Ball position
    velocity: Vector2D  # Ball velocity
    acceleration: Vector2D  # Ball acceleration (friction, etc.)
    spin: Vector2D  # Spin state
    energy: float  # Kinetic energy


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


class TrajectoryOptimizer:
    """Optimizes trajectory calculations for performance."""

    def __init__(self):
        self.adaptive_timestep = True
        self.early_termination = True
        self.collision_prediction_depth = 5
        self.min_velocity_threshold = 0.01  # m/s

    def get_optimal_timestep(
        self, velocity: Vector2D, quality: TrajectoryQuality
    ) -> float:
        """Calculate optimal timestep based on velocity and quality."""
        base_timesteps = {
            TrajectoryQuality.LOW: 0.01,
            TrajectoryQuality.MEDIUM: 0.005,
            TrajectoryQuality.HIGH: 0.002,
            TrajectoryQuality.ULTRA: 0.001,
        }

        base_dt = base_timesteps[quality]

        if self.adaptive_timestep:
            # Reduce timestep for higher velocities
            velocity_factor = max(0.1, 1.0 / (1.0 + velocity.magnitude() / 2.0))
            return base_dt * velocity_factor

        return base_dt

    def should_terminate_early(self, ball: BallState, table: TableState) -> bool:
        """Determine if trajectory calculation should terminate early."""
        if not self.early_termination:
            return False

        # Terminate if ball is moving too slowly
        if ball.velocity.magnitude() < self.min_velocity_threshold:
            return True

        # Terminate if ball is off table (using proper table coordinates 0 to width/height)
        return bool(
            ball.position.x < 0
            or ball.position.x > table.width
            or ball.position.y < 0
            or ball.position.y > table.height
        )


class TrajectoryCalculator:
    """Comprehensive ball trajectory calculations."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.optimizer = TrajectoryOptimizer()
        self.geometry = GeometryUtils()
        self.math_utils = MathUtils()

        # Physics constants
        self.GRAVITY = 9.81  # m/s²
        self.AIR_RESISTANCE = 0.02
        self.ROLLING_FRICTION = 0.015
        self.SLIDING_FRICTION = 0.25
        self.BALL_BALL_RESTITUTION = 0.95
        self.BALL_CUSHION_RESTITUTION = 0.85

        # Calculation settings
        self.max_simulation_time = 10.0  # seconds
        self.max_collisions = 20
        self.precision_tolerance = 1e-6

    def calculate_trajectory(
        self,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState] = None,
        quality: TrajectoryQuality = TrajectoryQuality.MEDIUM,
        time_limit: float = None,
    ) -> Trajectory:
        """Calculate complete trajectory for a ball."""
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(
            ball_state, table_state, other_balls, quality
        )
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        time_limit = time_limit or self.max_simulation_time
        other_balls = other_balls or []

        trajectory = Trajectory(
            ball_id=ball_state.id,
            initial_state=ball_state,
            quality=quality,
            cache_key=cache_key,
        )

        # Initialize simulation state
        current_ball = self._copy_ball_state(ball_state)
        current_time = 0.0
        collision_count = 0

        while (
            current_time < time_limit
            and collision_count < self.max_collisions
            and not self.optimizer.should_terminate_early(current_ball, table_state)
        ):
            # Calculate optimal timestep
            dt = self.optimizer.get_optimal_timestep(current_ball.velocity, quality)

            # Predict next collision
            next_collision = self._predict_next_collision(
                current_ball, table_state, other_balls, dt, time_limit - current_time
            )

            if next_collision and next_collision.time <= dt:
                # Collision will occur within this timestep
                # Move to collision point
                collision_time = next_collision.time
                self._integrate_motion(current_ball, collision_time, table_state)
                current_time += collision_time

                # Record trajectory point at collision
                trajectory.points.append(
                    TrajectoryPoint(
                        time=current_time,
                        position=Vector2D(
                            current_ball.position.x, current_ball.position.y
                        ),
                        velocity=Vector2D(
                            current_ball.velocity.x, current_ball.velocity.y
                        ),
                        acceleration=self._calculate_acceleration(
                            current_ball, table_state
                        ),
                        spin=(
                            Vector2D(current_ball.spin.x, current_ball.spin.y)
                            if current_ball.spin
                            else Vector2D(0, 0)
                        ),
                        energy=0.5
                        * current_ball.mass
                        * current_ball.velocity.magnitude() ** 2,
                    )
                )

                # Process collision
                self._process_collision(
                    current_ball, next_collision, table_state, other_balls
                )
                trajectory.collisions.append(next_collision)
                collision_count += 1

                # Check if ball was pocketed
                if next_collision.type == CollisionType.BALL_POCKET:
                    trajectory.will_be_pocketed = True
                    trajectory.pocket_id = next_collision.pocket_id
                    break

            else:
                # No collision in this timestep, integrate motion normally
                self._integrate_motion(current_ball, dt, table_state)
                current_time += dt

                # Record trajectory point
                trajectory.points.append(
                    TrajectoryPoint(
                        time=current_time,
                        position=Vector2D(
                            current_ball.position.x, current_ball.position.y
                        ),
                        velocity=Vector2D(
                            current_ball.velocity.x, current_ball.velocity.y
                        ),
                        acceleration=self._calculate_acceleration(
                            current_ball, table_state
                        ),
                        spin=(
                            Vector2D(current_ball.spin.x, current_ball.spin.y)
                            if current_ball.spin
                            else Vector2D(0, 0)
                        ),
                        energy=0.5
                        * current_ball.mass
                        * current_ball.velocity.magnitude() ** 2,
                    )
                )

        # Finalize trajectory
        trajectory.final_position = current_ball.position
        trajectory.final_velocity = current_ball.velocity
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

    def _calculate_acceleration(self, ball: BallState, table: TableState) -> Vector2D:
        """Calculate ball acceleration due to friction and other forces."""
        if ball.velocity.magnitude() < 1e-6:
            return Vector2D(0, 0)

        # Rolling friction (opposite to velocity direction)
        velocity_unit = ball.velocity.normalize()
        friction_magnitude = self.ROLLING_FRICTION * self.GRAVITY

        # Air resistance (proportional to velocity squared)
        air_resistance_magnitude = self.AIR_RESISTANCE * ball.velocity.magnitude()

        total_deceleration = friction_magnitude + air_resistance_magnitude

        return Vector2D(
            -velocity_unit.x * total_deceleration, -velocity_unit.y * total_deceleration
        )

    def _integrate_motion(self, ball: BallState, dt: float, table: TableState) -> None:
        """Integrate ball motion over time step using Runge-Kutta 4th order."""
        # Current state
        pos = ball.position
        vel = ball.velocity

        # k1
        k1_vel = vel
        k1_acc = self._calculate_acceleration(ball, table)

        # k2
        k2_pos = Vector2D(pos.x + 0.5 * dt * k1_vel.x, pos.y + 0.5 * dt * k1_vel.y)
        k2_vel = Vector2D(vel.x + 0.5 * dt * k1_acc.x, vel.y + 0.5 * dt * k1_acc.y)
        temp_ball = BallState(
            id=ball.id,
            position=k2_pos,
            velocity=k2_vel,
            radius=ball.radius,
            mass=ball.mass,
        )
        k2_acc = self._calculate_acceleration(temp_ball, table)

        # k3
        k3_pos = Vector2D(pos.x + 0.5 * dt * k2_vel.x, pos.y + 0.5 * dt * k2_vel.y)
        k3_vel = Vector2D(vel.x + 0.5 * dt * k2_acc.x, vel.y + 0.5 * dt * k2_acc.y)
        temp_ball.position = k3_pos
        temp_ball.velocity = k3_vel
        k3_acc = self._calculate_acceleration(temp_ball, table)

        # k4
        k4_pos = Vector2D(pos.x + dt * k3_vel.x, pos.y + dt * k3_vel.y)
        k4_vel = Vector2D(vel.x + dt * k3_acc.x, vel.y + dt * k3_acc.y)
        temp_ball.position = k4_pos
        temp_ball.velocity = k4_vel
        k4_acc = self._calculate_acceleration(temp_ball, table)

        # Final integration
        ball.position.x += (
            dt / 6.0 * (k1_vel.x + 2 * k2_vel.x + 2 * k3_vel.x + k4_vel.x)
        )
        ball.position.y += (
            dt / 6.0 * (k1_vel.y + 2 * k2_vel.y + 2 * k3_vel.y + k4_vel.y)
        )
        ball.velocity.x += (
            dt / 6.0 * (k1_acc.x + 2 * k2_acc.x + 2 * k3_acc.x + k4_acc.x)
        )
        ball.velocity.y += (
            dt / 6.0 * (k1_acc.y + 2 * k2_acc.y + 2 * k3_acc.y + k4_acc.y)
        )

        # Apply spin effects if present
        if ball.spin and ball.spin.magnitude() > 1e-6:
            self._apply_spin_effects(ball, dt)

    def _apply_spin_effects(self, ball: BallState, dt: float) -> None:
        """Apply spin effects to ball motion."""
        # Magnus force due to spin
        if ball.velocity.magnitude() > 1e-6:
            # Cross product of spin and velocity (simplified 2D)
            magnus_force_magnitude = (
                0.1 * ball.spin.magnitude() * ball.velocity.magnitude()
            )

            # Perpendicular to velocity
            vel_perp = Vector2D(-ball.velocity.y, ball.velocity.x).normalize()

            # Apply Magnus acceleration
            magnus_acc = Vector2D(
                vel_perp.x * magnus_force_magnitude / ball.mass,
                vel_perp.y * magnus_force_magnitude / ball.mass,
            )

            ball.velocity.x += magnus_acc.x * dt
            ball.velocity.y += magnus_acc.y * dt

        # Decay spin over time
        spin_decay = 0.95  # per second
        decay_factor = spin_decay**dt
        ball.spin.x *= decay_factor
        ball.spin.y *= decay_factor

    def _predict_next_collision(
        self,
        ball: BallState,
        table: TableState,
        other_balls: list[BallState],
        dt: float,
        max_time: float,
    ) -> Optional[PredictedCollision]:
        """Predict the next collision for the ball."""
        collisions = []

        # Check ball-ball collisions
        for other_ball in other_balls:
            if other_ball.id != ball.id and not other_ball.is_pocketed:
                collision = self._predict_ball_ball_collision(
                    ball, other_ball, max_time
                )
                if collision:
                    collisions.append(collision)

        # Check cushion collisions
        cushion_collision = self._predict_cushion_collision(ball, table, max_time)
        if cushion_collision:
            collisions.append(cushion_collision)

        # Check pocket collisions
        pocket_collision = self._predict_pocket_collision(ball, table, max_time)
        if pocket_collision:
            collisions.append(pocket_collision)

        # Return earliest collision
        if collisions:
            return min(collisions, key=lambda c: c.time)

        return None

    def _predict_ball_ball_collision(
        self, ball1: BallState, ball2: BallState, max_time: float
    ) -> Optional[PredictedCollision]:
        """Predict collision between two balls."""
        # Relative position and velocity
        rel_pos = Vector2D(
            ball2.position.x - ball1.position.x, ball2.position.y - ball1.position.y
        )
        rel_vel = Vector2D(
            ball2.velocity.x - ball1.velocity.x, ball2.velocity.y - ball1.velocity.y
        )

        # Check if balls are approaching
        if rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y >= 0:
            return None  # Moving away from each other

        # Quadratic equation for collision time
        combined_radius = ball1.radius + ball2.radius

        a = rel_vel.x**2 + rel_vel.y**2
        b = 2 * (rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y)
        c = rel_pos.x**2 + rel_pos.y**2 - combined_radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0 or abs(a) < 1e-10:
            return None  # No collision

        t = (-b - math.sqrt(discriminant)) / (2 * a)

        if t < 0 or t > max_time:
            return None  # Collision in past or beyond time limit

        # Calculate collision point
        collision_pos = Vector2D(
            ball1.position.x + ball1.velocity.x * t,
            ball1.position.y + ball1.velocity.y * t,
        )

        # Calculate impact angle
        impact_vector = Vector2D(
            ball2.position.x + ball2.velocity.x * t - collision_pos.x,
            ball2.position.y + ball2.velocity.y * t - collision_pos.y,
        )
        impact_angle = math.atan2(impact_vector.y, impact_vector.x)

        # Calculate post-collision velocities
        v1_new, v2_new = self._calculate_ball_collision_velocities(
            ball1, ball2, impact_vector
        )

        return PredictedCollision(
            time=t,
            position=collision_pos,
            type=CollisionType.BALL_BALL,
            ball1_id=ball1.id,
            ball2_id=ball2.id,
            impact_angle=impact_angle,
            impact_velocity=ball1.velocity.magnitude(),
            resulting_velocities={ball1.id: v1_new, ball2.id: v2_new},
            confidence=0.95,
        )

    def _predict_cushion_collision(
        self, ball: BallState, table: TableState, max_time: float
    ) -> Optional[PredictedCollision]:
        """Predict collision with table cushions."""
        # Check each cushion (left, right, top, bottom)
        # Note: Table coordinates are from (0,0) to (width, height)
        cushion_times = []

        # Left cushion (x = 0)
        if ball.velocity.x < 0:
            t = (ball.radius - ball.position.x) / ball.velocity.x
            if 0 < t <= max_time:
                y = ball.position.y + ball.velocity.y * t
                if 0 <= y <= table.height:
                    cushion_times.append((t, "left", Vector2D(1, 0)))

        # Right cushion (x = table.width)
        if ball.velocity.x > 0:
            t = (table.width - ball.radius - ball.position.x) / ball.velocity.x
            if 0 < t <= max_time:
                y = ball.position.y + ball.velocity.y * t
                if 0 <= y <= table.height:
                    cushion_times.append((t, "right", Vector2D(-1, 0)))

        # Bottom cushion (y = 0)
        if ball.velocity.y < 0:
            t = (ball.radius - ball.position.y) / ball.velocity.y
            if 0 < t <= max_time:
                x = ball.position.x + ball.velocity.x * t
                if 0 <= x <= table.width:
                    cushion_times.append((t, "bottom", Vector2D(0, 1)))

        # Top cushion (y = table.height)
        if ball.velocity.y > 0:
            t = (table.height - ball.radius - ball.position.y) / ball.velocity.y
            if 0 < t <= max_time:
                x = ball.position.x + ball.velocity.x * t
                if 0 <= x <= table.width:
                    cushion_times.append((t, "top", Vector2D(0, -1)))

        if not cushion_times:
            return None

        # Get earliest collision
        t, cushion, normal = min(cushion_times, key=lambda x: x[0])

        collision_pos = Vector2D(
            ball.position.x + ball.velocity.x * t, ball.position.y + ball.velocity.y * t
        )

        # Calculate reflected velocity
        new_velocity = self._calculate_cushion_reflection(
            ball.velocity, normal, table.cushion_elasticity
        )

        return PredictedCollision(
            time=t,
            position=collision_pos,
            type=CollisionType.BALL_CUSHION,
            ball1_id=ball.id,
            ball2_id=None,
            impact_angle=math.atan2(ball.velocity.y, ball.velocity.x),
            impact_velocity=ball.velocity.magnitude(),
            resulting_velocities={ball.id: new_velocity},
            confidence=0.90,
            cushion_normal=normal,
        )

    def _predict_pocket_collision(
        self, ball: BallState, table: TableState, max_time: float
    ) -> Optional[PredictedCollision]:
        """Predict if ball will go into a pocket."""
        for i, pocket_pos in enumerate(table.pocket_positions):
            # Calculate if trajectory intersects pocket
            # Simplified: check if ball center will be within pocket radius

            # Time to reach pocket x-coordinate
            if abs(ball.velocity.x) < 1e-10:
                if abs(ball.position.x - pocket_pos.x) > ball.radius:
                    continue
                t_x = float("inf")
            else:
                t_x = (pocket_pos.x - ball.position.x) / ball.velocity.x

            # Time to reach pocket y-coordinate
            if abs(ball.velocity.y) < 1e-10:
                if abs(ball.position.y - pocket_pos.y) > ball.radius:
                    continue
                t_y = float("inf")
            else:
                t_y = (pocket_pos.y - ball.position.y) / ball.velocity.y

            # Check if times are approximately equal (ball passes through pocket center)
            if 0 < t_x <= max_time and abs(t_x - t_y) < 0.01:
                # Verify ball actually enters pocket
                ball_pos_at_pocket = Vector2D(
                    ball.position.x + ball.velocity.x * t_x,
                    ball.position.y + ball.velocity.y * t_x,
                )

                distance_to_pocket = math.sqrt(
                    (ball_pos_at_pocket.x - pocket_pos.x) ** 2
                    + (ball_pos_at_pocket.y - pocket_pos.y) ** 2
                )

                if distance_to_pocket <= table.pocket_radius - ball.radius:
                    return PredictedCollision(
                        time=t_x,
                        position=ball_pos_at_pocket,
                        type=CollisionType.BALL_POCKET,
                        ball1_id=ball.id,
                        ball2_id=None,
                        impact_angle=math.atan2(ball.velocity.y, ball.velocity.x),
                        impact_velocity=ball.velocity.magnitude(),
                        resulting_velocities={ball.id: Vector2D(0, 0)},
                        confidence=0.85,
                        pocket_id=i,
                    )

        return None

    def _calculate_ball_collision_velocities(
        self, ball1: BallState, ball2: BallState, impact_normal: Vector2D
    ) -> tuple[Vector2D, Vector2D]:
        """Calculate post-collision velocities for two balls."""
        # Normalize impact vector
        normal = impact_normal.normalize()

        # Relative velocity in collision normal direction
        rel_vel = Vector2D(
            ball1.velocity.x - ball2.velocity.x, ball1.velocity.y - ball2.velocity.y
        )
        vel_along_normal = rel_vel.x * normal.x + rel_vel.y * normal.y

        # Do not resolve if velocities are separating
        if vel_along_normal > 0:
            return ball1.velocity, ball2.velocity

        # Calculate restitution
        restitution = self.BALL_BALL_RESTITUTION

        # Calculate impulse scalar
        impulse = 2 * vel_along_normal / (ball1.mass + ball2.mass)

        # Calculate new velocities
        impulse_vector = Vector2D(impulse * normal.x, impulse * normal.y)

        v1_new = Vector2D(
            ball1.velocity.x - impulse_vector.x * ball2.mass / ball1.mass * restitution,
            ball1.velocity.y - impulse_vector.y * ball2.mass / ball1.mass * restitution,
        )

        v2_new = Vector2D(
            ball2.velocity.x + impulse_vector.x * ball1.mass / ball2.mass * restitution,
            ball2.velocity.y + impulse_vector.y * ball1.mass / ball2.mass * restitution,
        )

        return v1_new, v2_new

    def _calculate_cushion_reflection(
        self, velocity: Vector2D, normal: Vector2D, elasticity: float
    ) -> Vector2D:
        """Calculate velocity after cushion collision."""
        # Reflect velocity across normal
        dot_product = velocity.x * normal.x + velocity.y * normal.y

        reflected_vel = Vector2D(
            velocity.x - 2 * dot_product * normal.x,
            velocity.y - 2 * dot_product * normal.y,
        )

        # Apply elasticity
        return Vector2D(reflected_vel.x * elasticity, reflected_vel.y * elasticity)

    def _process_collision(
        self,
        ball: BallState,
        collision: PredictedCollision,
        table: TableState,
        other_balls: list[BallState],
    ) -> None:
        """Process collision and update ball state."""
        if collision.type == CollisionType.BALL_BALL:
            # Update both balls' velocities
            ball.velocity = collision.resulting_velocities[ball.id]

            # Find and update other ball
            for other_ball in other_balls:
                if other_ball.id == collision.ball2_id:
                    other_ball.velocity = collision.resulting_velocities[other_ball.id]
                    break

        elif collision.type == CollisionType.BALL_CUSHION:
            ball.velocity = collision.resulting_velocities[ball.id]

        elif collision.type == CollisionType.BALL_POCKET:
            ball.velocity = Vector2D(0, 0)
            ball.is_pocketed = True

        # Update ball position to collision point
        ball.position = collision.position

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
        return {
            "ball_id": trajectory.ball_id,
            "points": [
                {
                    "time": p.time,
                    "x": p.position.x,
                    "y": p.position.y,
                    "vx": p.velocity.x,
                    "vy": p.velocity.y,
                    "energy": p.energy,
                }
                for p in trajectory.points
            ],
            "collisions": [
                {
                    "time": c.time,
                    "x": c.position.x,
                    "y": c.position.y,
                    "type": c.type.value,
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
                }
                for branch in trajectory.branches
            ],
        }

    def clear_cache(self) -> None:
        """Clear trajectory calculation cache."""
        self.cache_manager.clear()

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.cache_manager.get_stats()
