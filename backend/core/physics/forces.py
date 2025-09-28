"""Force calculation algorithms for billiard ball physics."""

import math
from dataclasses import dataclass
from typing import Optional

from backend.core.game_state import BallState, TableState, Vector2D


@dataclass
class ForceComponents:
    """Force components acting on a ball."""

    friction: Vector2D
    spin: Vector2D
    gravity: Vector2D
    air_resistance: Vector2D
    total: Vector2D


class ForceCalculator:
    """Advanced force and acceleration calculations for billiard balls."""

    def __init__(self, gravity: float = 9.81):
        """Initialize force calculator.

        Args:
            gravity: Gravitational acceleration in m/s^2
        """
        self.gravity = gravity

        # Physical constants
        self.air_density = 1.225  # kg/m^3 at sea level
        self.ball_drag_coefficient = 0.47  # Sphere drag coefficient

        # Advanced physics coefficients
        self.rolling_resistance_coefficient = 0.01  # Rolling resistance
        self.sliding_friction_coefficient = 0.3  # Sliding friction
        self.spin_decay_rate = 0.95  # Rate at which spin decays per second

    def calculate_all_forces(
        self, ball: BallState, table: TableState
    ) -> ForceComponents:
        """Calculate all forces acting on a ball.

        Args:
            ball: Current ball state
            table: Table properties

        Returns:
            ForceComponents with all calculated forces
        """
        # Calculate individual force components
        friction_force = self.calculate_friction_force(ball, table)
        spin_force = self.calculate_spin_force(ball, table)
        gravity_force = self.calculate_gravity_force(ball, table)
        air_resistance = self.calculate_air_resistance(ball)

        # Sum all forces
        total_force = Vector2D(
            friction_force.x + spin_force.x + gravity_force.x + air_resistance.x,
            friction_force.y + spin_force.y + gravity_force.y + air_resistance.y,
        )

        return ForceComponents(
            friction=friction_force,
            spin=spin_force,
            gravity=gravity_force,
            air_resistance=air_resistance,
            total=total_force,
        )

    def calculate_friction_force(self, ball: BallState, table: TableState) -> Vector2D:
        """Calculate friction force on a moving ball.

        Args:
            ball: Ball state with velocity information
            table: Table with surface friction properties

        Returns:
            Friction force vector opposing motion
        """
        velocity_magnitude = ball.velocity.magnitude()

        if velocity_magnitude < 1e-6:  # Ball essentially stopped
            return Vector2D(0.0, 0.0)

        # Ball radius and velocity are already in SI units (meters)

        # Check if ball is rolling (simplified - assumes most balls roll)
        is_rolling = True  # Simplified assumption

        if is_rolling:
            # Rolling friction
            friction_coefficient = self.rolling_resistance_coefficient
        else:
            # Sliding friction
            friction_coefficient = self.sliding_friction_coefficient

        # Calculate normal force (weight of ball on horizontal surface)
        normal_force = ball.mass * self.gravity

        # Friction force magnitude
        friction_magnitude = friction_coefficient * normal_force

        # Friction force direction (opposite to velocity)
        velocity_unit = ball.velocity.normalize()
        friction_force = Vector2D(
            -friction_magnitude * velocity_unit.x, -friction_magnitude * velocity_unit.y
        )

        # Apply table surface friction scaling
        friction_force.x *= table.surface_friction
        friction_force.y *= table.surface_friction

        return friction_force

    def calculate_spin_force(self, ball: BallState, table: TableState) -> Vector2D:
        """Calculate force from ball spin effects (Magnus effect).

        Args:
            ball: Ball state with spin information
            table: Table properties

        Returns:
            Force vector from spin effects
        """
        if ball.spin is None:
            return Vector2D(0.0, 0.0)

        spin_magnitude = ball.spin.magnitude()
        velocity_magnitude = ball.velocity.magnitude()

        if spin_magnitude < 1e-6 or velocity_magnitude < 1e-6:
            return Vector2D(0.0, 0.0)

        # Magnus effect coefficient (simplified)
        magnus_coefficient = 0.1  # Empirical value for pool balls

        # Cross product of spin and velocity vectors (2D approximation)
        # spin = (spin_x, spin_y, 0), velocity = (vx, vy, 0)
        # cross_product_z = spin_x * vy - spin_y * vx
        cross_product_z = ball.spin.x * ball.velocity.y - ball.spin.y * ball.velocity.x

        # Magnus force perpendicular to velocity
        velocity_unit = ball.velocity.normalize()
        magnus_force_magnitude = magnus_coefficient * abs(cross_product_z)

        # Force direction perpendicular to velocity
        if cross_product_z > 0:
            # Force to the left of motion direction
            magnus_force = Vector2D(-velocity_unit.y, velocity_unit.x)
        else:
            # Force to the right of motion direction
            magnus_force = Vector2D(velocity_unit.y, -velocity_unit.x)

        magnus_force.x *= magnus_force_magnitude
        magnus_force.y *= magnus_force_magnitude

        return magnus_force

    def calculate_gravity_force(self, ball: BallState, table: TableState) -> Vector2D:
        """Calculate gravitational force considering table slope.

        Args:
            ball: Ball state
            table: Table with slope information

        Returns:
            Gravity force vector
        """
        if abs(table.surface_slope) < 1e-6:
            return Vector2D(0.0, 0.0)  # Level table

        # Convert slope from degrees to radians
        slope_rad = math.radians(table.surface_slope)

        # Gravitational force component along slope
        gravity_force_magnitude = ball.mass * self.gravity * math.sin(slope_rad)

        # Assume slope is in the negative y direction (balls roll toward y=0)
        gravity_force = Vector2D(0.0, -gravity_force_magnitude)

        return gravity_force

    def calculate_air_resistance(self, ball: BallState) -> Vector2D:
        """Calculate air resistance force (drag).

        Args:
            ball: Ball state with velocity

        Returns:
            Air resistance force vector
        """
        velocity_magnitude = ball.velocity.magnitude()

        if velocity_magnitude < 1e-6:
            return Vector2D(0.0, 0.0)

        # Ball properties are already in SI units (meters)
        velocity_m_per_s = velocity_magnitude  # Already in m/s
        ball_radius_m = ball.radius  # Already in meters

        # Cross-sectional area
        cross_sectional_area = math.pi * ball_radius_m**2

        # Drag force magnitude: F_drag = 0.5 * ρ * C_d * A * v²
        drag_magnitude = (
            0.5
            * self.air_density
            * self.ball_drag_coefficient
            * cross_sectional_area
            * velocity_m_per_s**2
        )

        # Force is in Newtons (SI units)

        # Drag force direction (opposite to velocity)
        velocity_unit = ball.velocity.normalize()
        drag_force = Vector2D(
            -drag_magnitude * velocity_unit.x, -drag_magnitude * velocity_unit.y
        )

        return drag_force

    def calculate_acceleration(
        self, forces: ForceComponents, ball_mass: float
    ) -> Vector2D:
        """Calculate ball acceleration from total forces.

        Args:
            forces: All force components
            ball_mass: Mass of the ball in kg

        Returns:
            Acceleration vector in mm/s^2
        """
        # F = ma, so a = F/m
        acceleration = Vector2D(forces.total.x / ball_mass, forces.total.y / ball_mass)

        return acceleration

    def update_spin(self, ball: BallState, dt: float) -> None:
        """Update ball spin over time (spin decay).

        Args:
            ball: Ball state to update
            dt: Time step in seconds
        """
        if ball.spin is None:
            return

        # Exponential decay of spin
        decay_factor = self.spin_decay_rate**dt
        ball.spin.x *= decay_factor
        ball.spin.y *= decay_factor

        # Stop spin if it becomes negligible
        if ball.spin.magnitude() < 0.01:  # rad/s threshold
            ball.spin.x = 0.0
            ball.spin.y = 0.0

    def calculate_cue_impact_force(
        self,
        cue_velocity: float,
        impact_point: Vector2D,
        ball_center: Vector2D,
        ball_radius: float,
    ) -> tuple[Vector2D, Optional[Vector2D]]:
        """Calculate initial velocity and spin from cue stick impact.

        Args:
            cue_velocity: Cue stick velocity at impact (m/s)
            impact_point: Point of impact on ball surface
            ball_center: Center of the ball
            ball_radius: Ball radius in mm

        Returns:
            Tuple of (initial_velocity, initial_spin)
        """
        # Vector from ball center to impact point
        impact_vector = Vector2D(
            impact_point.x - ball_center.x, impact_point.y - ball_center.y
        )

        impact_distance = impact_vector.magnitude()

        if impact_distance > ball_radius:
            # Impact point outside ball, clamp to surface
            impact_vector = impact_vector.normalize()
            impact_vector.x *= ball_radius
            impact_vector.y *= ball_radius
            impact_distance = ball_radius

        # Calculate velocity transfer efficiency
        # Center hit transfers most momentum, edge hits transfer less
        efficiency = 1.0 - (impact_distance / ball_radius) * 0.3

        # Initial velocity in direction of cue (already in m/s)
        initial_velocity = Vector2D(
            cue_velocity * efficiency,  # Already in m/s
            0.0,  # Assume cue strikes horizontally
        )

        # Calculate spin from off-center hits
        if impact_distance > ball_radius * 0.1:  # Significant off-center hit
            # Spin proportional to off-center distance
            spin_magnitude = (
                (impact_distance / ball_radius) * cue_velocity * 10
            )  # rad/s

            # Spin direction perpendicular to impact vector
            spin_direction = Vector2D(-impact_vector.y, impact_vector.x).normalize()
            initial_spin = Vector2D(
                spin_direction.x * spin_magnitude, spin_direction.y * spin_magnitude
            )
        else:
            initial_spin = Vector2D(0.0, 0.0)

        return initial_velocity, initial_spin

    def calculate_rolling_velocity(self, ball: BallState) -> Vector2D:
        """Calculate the velocity at which a ball should roll without slipping.

        Args:
            ball: Ball state with current spin

        Returns:
            Rolling velocity vector
        """
        if ball.spin is None:
            return Vector2D(0.0, 0.0)

        # For a sphere: v = ω × r
        # In 2D, this becomes v = ω_z * r * perpendicular_unit_vector
        ball_radius_m = ball.radius  # Already in meters

        # Rolling velocity magnitude
        rolling_speed = ball.spin.magnitude() * ball_radius_m  # Already in m/s

        if rolling_speed < 1e-6:
            return Vector2D(0.0, 0.0)

        # Rolling direction (perpendicular to spin axis)
        spin_unit = ball.spin.normalize()
        rolling_direction = Vector2D(-spin_unit.y, spin_unit.x)

        return Vector2D(
            rolling_direction.x * rolling_speed, rolling_direction.y * rolling_speed
        )
