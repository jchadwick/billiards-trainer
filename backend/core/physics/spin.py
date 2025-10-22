"""Spin and English effects simulation.

This module implements comprehensive spin physics for billiards balls including:
- English (side spin) application from cue impact
- Spin transfer during ball-to-ball collisions
- Spin decay over time due to friction
- Spin-induced trajectory modifications
- Magnus force effects on ball movement
- Surface interaction effects on spin

All spatial calculations use 4K pixels (3840×2160) as the canonical coordinate system.
Ball radius and distances are in pixels; velocities in pixels/second.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..constants_4k import BALL_MASS_KG, BALL_RADIUS_4K, PIXELS_PER_METER_REFERENCE
from ..coordinates import Vector2D
from ..models import BallState, Collision


@dataclass
class SpinState:
    """Complete spin state of a ball."""

    # Angular velocity components (rad/s)
    top_spin: float = 0.0  # Forward/backward spin (ωx)
    side_spin: float = 0.0  # Left/right spin (ωy)
    back_spin: float = 0.0  # Up/down spin relative to table (ωz)

    # Spin decay rates (per second)
    decay_rate: float = 2.0  # How quickly spin decays due to friction

    def magnitude(self) -> float:
        """Calculate total spin magnitude."""
        return math.sqrt(self.top_spin**2 + self.side_spin**2 + self.back_spin**2)

    def to_vector2d(self) -> Vector2D:
        """Convert to Vector2D for compatibility with existing code."""
        return Vector2D(self.side_spin, self.top_spin)

    def from_vector2d(self, spin_vec: Vector2D) -> None:
        """Update from Vector2D."""
        self.side_spin = spin_vec.x
        self.top_spin = spin_vec.y


class SpinPhysics:
    """Advanced spin physics calculations in 4K pixel space."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize spin physics with configuration.

        Args:
            config: Physics configuration parameters
        """
        self.config = config or {}

        # Physical constants - converted to 4K pixel space
        self.ball_radius = BALL_RADIUS_4K  # Ball radius in 4K pixels
        self.ball_mass = BALL_MASS_KG  # Mass in kg (not spatial)
        self.moment_of_inertia = (
            0.4 * self.ball_mass * (self.ball_radius / PIXELS_PER_METER_REFERENCE) ** 2
        )  # Solid sphere (using radius in meters for calculation)

        # Surface interaction parameters (dimensionless)
        self.cloth_friction = self.config.get("cloth_friction", 0.3)
        self.roll_slip_threshold = (
            self.config.get("roll_slip_threshold", 0.1) * PIXELS_PER_METER_REFERENCE
        )  # pixels/s
        self.magnus_coefficient = self.config.get("magnus_coefficient", 0.25)

        # Spin transfer coefficients (dimensionless)
        self.spin_transfer_efficiency = self.config.get("spin_transfer_efficiency", 0.7)
        self.cushion_spin_retention = self.config.get("cushion_spin_retention", 0.6)

        # Cue interaction parameters (dimensionless)
        self.cue_efficiency = self.config.get("cue_efficiency", 0.8)
        self.english_effectiveness = self.config.get("english_effectiveness", 0.9)

    def apply_english_from_cue(
        self,
        cue_ball: BallState,
        impact_point: Vector2D,
        force_magnitude: float,
        cue_angle: float,
        cue_elevation: float = 0.0,
    ) -> SpinState:
        """Apply English (spin) to cue ball from cue impact.

        Args:
            cue_ball: Cue ball state
            impact_point: Point where cue hits ball (relative to ball center)
            force_magnitude: Force of cue impact (N)
            cue_angle: Cue angle in degrees (0 = horizontal right)
            cue_elevation: Cue elevation in degrees (0 = horizontal)

        Returns:
            New spin state for the cue ball
        """
        # Convert angles to radians
        angle_rad = math.radians(cue_angle)
        elevation_rad = math.radians(cue_elevation)

        # Calculate impact offset from ball center (normalized by radius)
        offset_x = impact_point.x / self.ball_radius
        offset_y = impact_point.y / self.ball_radius

        # Ensure impact point is within ball surface
        offset_magnitude = math.sqrt(offset_x**2 + offset_y**2)
        if offset_magnitude > 1.0:
            offset_x /= offset_magnitude
            offset_y /= offset_magnitude
            offset_magnitude = 1.0

        # Calculate cue direction vector
        cue_dir = Vector2D(
            math.cos(angle_rad) * math.cos(elevation_rad),
            math.sin(angle_rad) * math.cos(elevation_rad),
        )

        # Calculate contact force components
        force_normal = force_magnitude * math.cos(elevation_rad)
        force_tangent_x = -offset_y * force_normal * self.english_effectiveness
        force_tangent_y = offset_x * force_normal * self.english_effectiveness

        # Calculate linear velocity imparted
        impulse = force_magnitude * 0.01  # Simplified contact time
        velocity_change = Vector2D(
            impulse * cue_dir.x / self.ball_mass, impulse * cue_dir.y / self.ball_mass
        )

        # Update ball velocity
        cue_ball.velocity = Vector2D(
            cue_ball.velocity.x + velocity_change.x,
            cue_ball.velocity.y + velocity_change.y,
        )

        # Calculate angular impulse and resulting spin
        torque_x = force_tangent_x * self.ball_radius
        torque_y = force_tangent_y * self.ball_radius
        torque_z = 0.0  # Simplified - real calculation would be more complex

        angular_impulse_x = torque_x * 0.01  # Contact time
        angular_impulse_y = torque_y * 0.01
        angular_impulse_z = torque_z * 0.01

        # Convert to angular velocity
        spin_state = SpinState(
            top_spin=angular_impulse_y / self.moment_of_inertia,
            side_spin=angular_impulse_x / self.moment_of_inertia,
            back_spin=angular_impulse_z / self.moment_of_inertia,
        )

        # Add follow/draw effect from elevation
        elevation_effect = (
            math.sin(elevation_rad) * force_magnitude * self.cue_efficiency
        )
        spin_state.top_spin += elevation_effect / (
            self.moment_of_inertia * self.ball_radius
        )

        return spin_state

    def update_spin_with_motion(
        self, ball: BallState, spin_state: SpinState, dt: float
    ) -> tuple[SpinState, Vector2D]:
        """Update spin state and calculate Magnus force during motion.

        Args:
            ball: Ball state
            spin_state: Current spin state
            dt: Time step

        Returns:
            Tuple of (updated_spin_state, magnus_force)
        """
        # Check if ball is rolling or slipping
        linear_velocity = ball.velocity.magnitude()
        expected_roll_velocity = spin_state.top_spin * self.ball_radius

        slip_velocity = abs(linear_velocity - expected_roll_velocity)
        is_slipping = slip_velocity > self.roll_slip_threshold

        # Calculate spin decay due to surface friction
        if is_slipping:
            # Rapid spin decay when slipping
            decay_factor = math.exp(-spin_state.decay_rate * 3.0 * dt)
        else:
            # Slower decay when rolling
            decay_factor = math.exp(-spin_state.decay_rate * dt)

        # Apply spin decay
        new_spin = SpinState(
            top_spin=spin_state.top_spin * decay_factor,
            side_spin=spin_state.side_spin * decay_factor,
            back_spin=spin_state.back_spin * decay_factor,
            decay_rate=spin_state.decay_rate,
        )

        # Calculate Magnus force (perpendicular to velocity and spin axis)
        if linear_velocity > 0.01 and new_spin.magnitude() > 0.1:
            # Simplified Magnus force calculation
            # Real calculation would involve cross product of velocity and angular velocity
            velocity_unit = ball.velocity.normalize()

            # Side spin creates curve
            magnus_x = -new_spin.side_spin * velocity_unit.y * self.magnus_coefficient
            magnus_y = new_spin.side_spin * velocity_unit.x * self.magnus_coefficient

            # Top/back spin affects trajectory less in 2D simulation
            # but can affect ball interaction with cloth

            magnus_force = Vector2D(magnus_x, magnus_y)
        else:
            magnus_force = Vector2D(0.0, 0.0)

        # Gradually align roll with linear motion (for realistic physics)
        if is_slipping and linear_velocity > 0.01:
            target_roll_spin = linear_velocity / self.ball_radius
            spin_alignment_rate = 5.0  # How quickly spin aligns with motion

            spin_difference = target_roll_spin - new_spin.top_spin
            alignment_change = spin_difference * spin_alignment_rate * dt
            new_spin.top_spin += alignment_change

        return new_spin, magnus_force

    def transfer_spin_collision(
        self,
        ball1: BallState,
        ball2: BallState,
        spin1: SpinState,
        spin2: SpinState,
        collision: Collision,
    ) -> tuple[SpinState, SpinState]:
        """Transfer spin between balls during collision.

        Args:
            ball1: First ball state
            ball2: Second ball state
            spin1: First ball spin state
            spin2: Second ball spin state
            collision: Collision information

        Returns:
            Tuple of (new_spin1, new_spin2)
        """
        # Calculate collision geometry
        if collision.position:
            # Collision normal (from ball1 to ball2)
            dx = ball2.position.x - ball1.position.x
            dy = ball2.position.y - ball1.position.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance > 0:
                normal_x = dx / distance
                normal_y = dy / distance
            else:
                normal_x, normal_y = 1.0, 0.0
        else:
            normal_x, normal_y = 1.0, 0.0

        # Calculate relative velocity at contact point
        rel_vel_x = ball1.velocity.x - ball2.velocity.x
        rel_vel_y = ball1.velocity.y - ball2.velocity.y

        # Calculate surface velocities due to spin
        surface_vel1_x = -spin1.side_spin * self.ball_radius
        surface_vel1_y = spin1.top_spin * self.ball_radius
        surface_vel2_x = -spin2.side_spin * self.ball_radius
        surface_vel2_y = spin2.top_spin * self.ball_radius

        # Calculate relative surface velocity
        rel_surface_vel_x = surface_vel1_x - surface_vel2_x
        rel_surface_vel_y = surface_vel1_y - surface_vel2_y

        # Calculate tangential component of relative velocity
        tangent_x = -normal_y
        tangent_y = normal_x

        rel_tangent_vel = (rel_vel_x + rel_surface_vel_x) * tangent_x + (
            rel_vel_y + rel_surface_vel_y
        ) * tangent_y

        # Calculate spin transfer based on friction and collision dynamics
        if abs(rel_tangent_vel) > 0.01:
            # Friction force during collision
            normal_force = (
                abs(ball1.velocity.magnitude() - ball2.velocity.magnitude())
                * self.ball_mass
            )
            friction_force = self.cloth_friction * normal_force

            # Angular impulse from friction
            angular_impulse = (
                friction_force * 0.001 * self.ball_radius
            )  # Very short contact time
            spin_change = angular_impulse / self.moment_of_inertia

            # Transfer spin based on collision efficiency
            transfer_factor = (
                self.spin_transfer_efficiency * abs(rel_tangent_vel) / 10.0
            )
            transfer_factor = min(transfer_factor, 1.0)

            if rel_tangent_vel > 0:
                # Ball1 spinning faster, transfer to ball2
                spin_transfer = spin_change * transfer_factor
                new_spin1_side = spin1.side_spin - spin_transfer
                new_spin2_side = (
                    spin2.side_spin + spin_transfer * 0.5
                )  # Partial transfer
            else:
                # Ball2 spinning faster, transfer to ball1
                spin_transfer = spin_change * transfer_factor
                new_spin1_side = spin1.side_spin + spin_transfer * 0.5
                new_spin2_side = spin2.side_spin - spin_transfer
        else:
            new_spin1_side = spin1.side_spin
            new_spin2_side = spin2.side_spin

        # Top spin transfer is usually less significant in ball-to-ball collisions
        new_spin1_top = spin1.top_spin * 0.9  # Slight reduction
        new_spin2_top = spin2.top_spin * 0.9

        # Back spin typically not affected much by ball collisions
        new_spin1_back = spin1.back_spin * 0.95
        new_spin2_back = spin2.back_spin * 0.95

        new_spin1 = SpinState(
            top_spin=new_spin1_top,
            side_spin=new_spin1_side,
            back_spin=new_spin1_back,
            decay_rate=spin1.decay_rate,
        )

        new_spin2 = SpinState(
            top_spin=new_spin2_top,
            side_spin=new_spin2_side,
            back_spin=new_spin2_back,
            decay_rate=spin2.decay_rate,
        )

        return new_spin1, new_spin2

    def cushion_spin_interaction(
        self, ball: BallState, spin_state: SpinState, cushion_normal: Vector2D
    ) -> SpinState:
        """Calculate spin changes when ball hits cushion.

        Args:
            ball: Ball state
            spin_state: Current spin state
            cushion_normal: Normal vector of cushion surface

        Returns:
            New spin state after cushion interaction
        """
        # Calculate velocity components relative to cushion
        vel_normal = (
            ball.velocity.x * cushion_normal.x + ball.velocity.y * cushion_normal.y
        )
        vel_tangent = (
            ball.velocity.x * (-cushion_normal.y) + ball.velocity.y * cushion_normal.x
        )

        # Side spin affects angle of rebound
        side_spin_effect = spin_state.side_spin * self.ball_radius
        effective_tangent_vel = vel_tangent + side_spin_effect

        # Calculate friction force during cushion contact
        friction_force = self.cloth_friction * abs(vel_normal)
        contact_time = 0.002  # Typical cushion contact time

        # Angular impulse from cushion friction
        angular_impulse = friction_force * contact_time * self.ball_radius
        spin_change = angular_impulse / self.moment_of_inertia

        # Apply spin changes
        if cushion_normal.x != 0:  # Vertical cushion (affects side spin)
            new_side_spin = -spin_state.side_spin * self.cushion_spin_retention
            new_top_spin = spin_state.top_spin + spin_change * math.copysign(
                1, effective_tangent_vel
            )
        else:  # Horizontal cushion (affects top/back spin)
            new_side_spin = spin_state.side_spin + spin_change * math.copysign(
                1, effective_tangent_vel
            )
            new_top_spin = -spin_state.top_spin * self.cushion_spin_retention

        new_back_spin = spin_state.back_spin * self.cushion_spin_retention

        return SpinState(
            top_spin=new_top_spin,
            side_spin=new_side_spin,
            back_spin=new_back_spin,
            decay_rate=spin_state.decay_rate,
        )

    def calculate_masse_shot(
        self,
        cue_ball: BallState,
        target_position: Vector2D,
        cue_elevation: float,
        english_amount: float,
    ) -> tuple[Vector2D, SpinState]:
        """Calculate masse shot trajectory and spin.

        Args:
            cue_ball: Cue ball state
            target_position: Desired target position
            cue_elevation: Cue elevation angle (degrees)
            english_amount: Amount of English (-1.0 to 1.0)

        Returns:
            Tuple of (initial_velocity, initial_spin)
        """
        # Calculate required force and angle for masse shot
        distance = cue_ball.position.distance_to(target_position)

        # High elevation creates more spin, less linear motion
        elevation_rad = math.radians(cue_elevation)
        elevation_factor = math.sin(elevation_rad)

        # Calculate direction to target
        direction = Vector2D(
            target_position.x - cue_ball.position.x,
            target_position.y - cue_ball.position.y,
        ).normalize()

        # For masse shots, initial velocity is reduced due to high elevation
        base_velocity = min(distance * 2.0, 5.0)  # Moderate speed
        velocity_reduction = (
            elevation_factor * 0.7
        )  # High elevation reduces forward motion

        initial_velocity = Vector2D(
            direction.x * base_velocity * (1.0 - velocity_reduction),
            direction.y * base_velocity * (1.0 - velocity_reduction),
        )

        # Calculate masse spin (high side spin, significant back spin)
        masse_spin = SpinState(
            top_spin=-elevation_factor * 15.0,  # Strong back spin
            side_spin=english_amount * elevation_factor * 20.0,  # Strong side spin
            back_spin=elevation_factor * 10.0,  # Additional back spin component
            decay_rate=1.5,  # Masse shots decay faster due to extreme spin
        )

        return initial_velocity, masse_spin


class SpinCalculator:
    """High-level interface for spin calculations."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize spin calculator.

        Args:
            config: Configuration parameters
        """
        self.physics = SpinPhysics(config)
        self.spin_states: dict[str, Vector2D] = {}  # Track spin for each ball

    def apply_english(
        self,
        cue_ball: BallState,
        impact_point: Vector2D,
        force: float,
        cue_angle: float = 0.0,
        cue_elevation: float = 0.0,
    ) -> None:
        """Apply spin from cue English.

        Args:
            cue_ball: Cue ball to apply English to
            impact_point: Point where cue hits ball
            force: Force of cue impact
            cue_angle: Cue angle in degrees
            cue_elevation: Cue elevation in degrees
        """
        spin_state = self.physics.apply_english_from_cue(
            cue_ball, impact_point, force, cue_angle, cue_elevation
        )

        # Store spin state and update ball's spin vector for compatibility
        self.spin_states[cue_ball.id] = spin_state
        cue_ball.spin = spin_state.to_vector2d()

    def transfer_spin(
        self, ball1: BallState, ball2: BallState, collision: Collision
    ) -> None:
        """Transfer spin between balls during collision.

        Args:
            ball1: First ball
            ball2: Second ball
            collision: Collision information
        """
        # Get current spin states
        spin1 = self.spin_states.get(ball1.id, SpinState())
        spin2 = self.spin_states.get(ball2.id, SpinState())

        # Update from ball spin vectors if available
        if ball1.spin:
            spin1.from_vector2d(ball1.spin)
        if ball2.spin:
            spin2.from_vector2d(ball2.spin)

        # Calculate spin transfer
        new_spin1, new_spin2 = self.physics.transfer_spin_collision(
            ball1, ball2, spin1, spin2, collision
        )

        # Update stored states and ball spin vectors
        self.spin_states[ball1.id] = new_spin1
        self.spin_states[ball2.id] = new_spin2
        ball1.spin = new_spin1.to_vector2d()
        ball2.spin = new_spin2.to_vector2d()

    def update_ball_motion(self, ball: BallState, dt: float) -> Vector2D:
        """Update ball motion considering spin effects.

        Args:
            ball: Ball to update
            dt: Time step

        Returns:
            Magnus force to apply to ball
        """
        spin_state = self.spin_states.get(ball.id, SpinState())

        # Update from ball spin vector if available
        if ball.spin:
            spin_state.from_vector2d(ball.spin)

        # Update spin and get Magnus force
        new_spin, magnus_force = self.physics.update_spin_with_motion(
            ball, spin_state, dt
        )

        # Store updated state and update ball spin vector
        self.spin_states[ball.id] = new_spin
        ball.spin = new_spin.to_vector2d()

        return magnus_force

    def handle_cushion_collision(
        self, ball: BallState, cushion_normal: Vector2D
    ) -> None:
        """Handle spin changes during cushion collision.

        Args:
            ball: Ball hitting cushion
            cushion_normal: Normal vector of cushion
        """
        spin_state = self.spin_states.get(ball.id, SpinState())

        # Update from ball spin vector if available
        if ball.spin:
            spin_state.from_vector2d(ball.spin)

        # Calculate new spin after cushion interaction
        new_spin = self.physics.cushion_spin_interaction(
            ball, spin_state, cushion_normal
        )

        # Store updated state and update ball spin vector
        self.spin_states[ball.id] = new_spin
        ball.spin = new_spin.to_vector2d()

    def calculate_masse_shot(
        self,
        cue_ball: BallState,
        target_position: Vector2D,
        cue_elevation: float,
        english_amount: float = 0.0,
    ) -> tuple[Vector2D, Vector2D]:
        """Calculate masse shot parameters.

        Args:
            cue_ball: Cue ball
            target_position: Target position
            cue_elevation: Cue elevation angle
            english_amount: Amount of English

        Returns:
            Tuple of (initial_velocity, initial_spin_vector)
        """
        initial_velocity, spin_state = self.physics.calculate_masse_shot(
            cue_ball, target_position, cue_elevation, english_amount
        )

        # Store spin state
        self.spin_states[cue_ball.id] = spin_state

        return initial_velocity, spin_state.to_vector2d()

    def get_spin_state(self, ball_id: str) -> Optional[SpinState]:
        """Get detailed spin state for a ball.

        Args:
            ball_id: Ball ID

        Returns:
            Spin state if available
        """
        return self.spin_states.get(ball_id)

    def reset_spin(self, ball_id: str) -> None:
        """Reset spin for a ball.

        Args:
            ball_id: Ball ID to reset
        """
        if ball_id in self.spin_states:
            del self.spin_states[ball_id]
