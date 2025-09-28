"""Physics validation algorithms.

This module provides comprehensive physics validation to ensure system reliability
and catch invalid physics states. It validates trajectories, collisions, ball states,
forces, and conservation laws.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

from backend.core.models import BallState, Collision, TableState, Trajectory, Vector2D
from backend.core.physics.engine import PhysicsConstants, TrajectoryPoint


@dataclass
class ValidationError:
    """Physics validation error information."""

    error_type: str  # Type of validation error
    severity: str  # "warning", "error", "critical"
    message: str  # Human-readable error message
    details: dict[str, Any]  # Additional error details
    suggested_fix: Optional[str] = None  # Suggested fix for the error


@dataclass
class ValidationResult:
    """Result of physics validation."""

    is_valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]
    confidence: float  # 0.0 to 1.0 confidence in validation


class PhysicsValidator:
    """Comprehensive physics consistency validation.

    Validates physics calculations, conservation laws, and system constraints
    to ensure reliable physics simulation and catch invalid states.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize physics validator with configuration.

        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.constants = PhysicsConstants()

        # Validation tolerances
        self.energy_tolerance = self.config.get(
            "energy_tolerance", 0.05
        )  # 5% tolerance
        self.momentum_tolerance = self.config.get(
            "momentum_tolerance", 0.01
        )  # 1% tolerance
        self.velocity_tolerance = self.config.get("velocity_tolerance", 0.001)  # m/s
        self.position_tolerance = self.config.get("position_tolerance", 0.001)  # m

        # Physics limits
        self.max_velocity = self.config.get("max_velocity", 20.0)  # m/s
        self.max_acceleration = self.config.get("max_acceleration", 100.0)  # m/s²
        self.max_spin = self.config.get("max_spin", 100.0)  # rad/s
        self.max_force = self.config.get("max_force", 50.0)  # N

        # Performance settings
        self.detailed_validation = self.config.get("detailed_validation", True)
        self.conservation_checks = self.config.get("conservation_checks", True)

    def validate_trajectory(
        self, trajectory: Union[Trajectory, list[TrajectoryPoint]]
    ) -> ValidationResult:
        """Validate trajectory physics validity.

        Args:
            trajectory: Trajectory object or list of trajectory points

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []

        # Handle both Trajectory objects and list of TrajectoryPoint
        if isinstance(trajectory, Trajectory):
            points = [
                TrajectoryPoint(
                    time=i
                    * (trajectory.time_to_rest / max(1, len(trajectory.points) - 1)),
                    position=point,
                    velocity=(
                        trajectory.final_velocity
                        if i == len(trajectory.points) - 1
                        else Vector2D(0, 0)
                    ),
                )
                for i, point in enumerate(trajectory.points)
            ]
            ball_id = trajectory.ball_id
        else:
            points = trajectory
            ball_id = points[0].position if points else "unknown"

        if not points:
            errors.append(
                ValidationError(
                    error_type="empty_trajectory",
                    severity="error",
                    message="Trajectory contains no points",
                    details={"ball_id": ball_id},
                )
            )
            return ValidationResult(False, errors, warnings, 0.0)

        # Validate trajectory continuity
        for i in range(1, len(points)):
            prev_point = points[i - 1]
            curr_point = points[i]

            # Check time sequence
            if curr_point.time <= prev_point.time:
                errors.append(
                    ValidationError(
                        error_type="time_sequence",
                        severity="error",
                        message=f"Invalid time sequence at point {i}",
                        details={
                            "prev_time": prev_point.time,
                            "curr_time": curr_point.time,
                            "point_index": i,
                        },
                        suggested_fix="Ensure trajectory points have monotonically increasing time",
                    )
                )

            # Check position continuity
            dt = curr_point.time - prev_point.time
            if dt > 0:
                # Calculate expected position based on previous velocity
                if hasattr(prev_point, "velocity") and prev_point.velocity:
                    expected_pos = Vector2D(
                        prev_point.position.x + prev_point.velocity.x * dt,
                        prev_point.position.y + prev_point.velocity.y * dt,
                    )
                    position_error = curr_point.position.distance_to(expected_pos)

                    if position_error > self.position_tolerance:
                        warnings.append(
                            ValidationError(
                                error_type="position_discontinuity",
                                severity="warning",
                                message=f"Position discontinuity at point {i}",
                                details={
                                    "expected_position": expected_pos.to_dict(),
                                    "actual_position": curr_point.position.to_dict(),
                                    "error_distance": position_error,
                                    "point_index": i,
                                },
                            )
                        )

        # Validate velocity constraints
        for i, point in enumerate(points):
            if hasattr(point, "velocity") and point.velocity:
                velocity_mag = point.velocity.magnitude()
                if velocity_mag > self.max_velocity:
                    errors.append(
                        ValidationError(
                            error_type="velocity_limit",
                            severity="error",
                            message=f"Velocity exceeds limit at point {i}",
                            details={
                                "velocity": velocity_mag,
                                "max_velocity": self.max_velocity,
                                "point_index": i,
                            },
                            suggested_fix=f"Reduce velocity to below {self.max_velocity} m/s",
                        )
                    )

        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)

        return ValidationResult(is_valid, errors, warnings, max(0.0, confidence))

    def validate_collision(
        self, collision: Collision, ball1: BallState, ball2: Optional[BallState] = None
    ) -> ValidationResult:
        """Verify collision calculations and physics validity.

        Args:
            collision: Collision object to validate
            ball1: First ball involved in collision
            ball2: Second ball (None for cushion/pocket collisions)

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []

        # Validate collision type consistency
        if collision.type == "ball" and ball2 is None:
            errors.append(
                ValidationError(
                    error_type="collision_type_mismatch",
                    severity="error",
                    message="Ball collision requires two balls",
                    details={"collision_type": collision.type, "ball2_provided": False},
                )
            )

        # Validate collision time
        if collision.time < 0:
            warnings.append(
                ValidationError(
                    error_type="negative_collision_time",
                    severity="warning",
                    message="Collision time is negative (past collision)",
                    details={"collision_time": collision.time},
                )
            )

        # Validate collision position
        if collision.type == "ball" and ball2:
            # Check if collision position is reasonable
            distance_to_ball1 = collision.position.distance_to(ball1.position)
            collision.position.distance_to(ball2.position)

            expected_distance = ball1.radius
            if abs(distance_to_ball1 - expected_distance) > self.position_tolerance:
                warnings.append(
                    ValidationError(
                        error_type="collision_position",
                        severity="warning",
                        message="Collision position inconsistent with ball1 radius",
                        details={
                            "distance_to_ball1": distance_to_ball1,
                            "expected_distance": expected_distance,
                            "tolerance": self.position_tolerance,
                        },
                    )
                )

        # Validate resulting velocities if provided
        if (
            hasattr(collision, "resulting_velocities")
            and collision.resulting_velocities
        ):
            for ball_id, velocity in collision.resulting_velocities.items():
                velocity_mag = velocity.magnitude()
                if velocity_mag > self.max_velocity:
                    errors.append(
                        ValidationError(
                            error_type="post_collision_velocity",
                            severity="error",
                            message=f"Post-collision velocity exceeds limit for ball {ball_id}",
                            details={
                                "ball_id": ball_id,
                                "velocity": velocity_mag,
                                "max_velocity": self.max_velocity,
                            },
                        )
                    )

        # Validate impact force
        if collision.impact_force > self.max_force:
            warnings.append(
                ValidationError(
                    error_type="impact_force",
                    severity="warning",
                    message="Impact force is unusually high",
                    details={
                        "impact_force": collision.impact_force,
                        "max_force": self.max_force,
                    },
                )
            )

        is_valid = len(errors) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.4 + len(warnings) * 0.1))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def validate_ball_state(
        self, ball_state: BallState, table: Optional[TableState] = None
    ) -> ValidationResult:
        """Check ball physics properties and constraints.

        Args:
            ball_state: Ball state to validate
            table: Optional table state for boundary checking

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []

        # Validate basic properties
        if ball_state.radius <= 0:
            errors.append(
                ValidationError(
                    error_type="invalid_radius",
                    severity="error",
                    message="Ball radius must be positive",
                    details={"radius": ball_state.radius},
                    suggested_fix="Set radius to standard ball radius (0.028575 m)",
                )
            )

        if ball_state.mass <= 0:
            errors.append(
                ValidationError(
                    error_type="invalid_mass",
                    severity="error",
                    message="Ball mass must be positive",
                    details={"mass": ball_state.mass},
                    suggested_fix="Set mass to standard ball mass (0.17 kg)",
                )
            )

        # Validate velocity
        velocity_mag = ball_state.velocity.magnitude()
        if velocity_mag > self.max_velocity:
            errors.append(
                ValidationError(
                    error_type="velocity_limit",
                    severity="error",
                    message="Ball velocity exceeds physical limits",
                    details={
                        "velocity": velocity_mag,
                        "max_velocity": self.max_velocity,
                        "ball_id": ball_state.id,
                    },
                    suggested_fix=f"Reduce velocity to below {self.max_velocity} m/s",
                )
            )

        # Validate spin
        if ball_state.spin:
            spin_mag = ball_state.spin.magnitude()
            if spin_mag > self.max_spin:
                warnings.append(
                    ValidationError(
                        error_type="spin_limit",
                        severity="warning",
                        message="Ball spin is unusually high",
                        details={
                            "spin": spin_mag,
                            "max_spin": self.max_spin,
                            "ball_id": ball_state.id,
                        },
                    )
                )

        # Validate position relative to table
        if table and not ball_state.is_pocketed:
            if not table.is_point_on_table(ball_state.position, ball_state.radius):
                errors.append(
                    ValidationError(
                        error_type="position_out_of_bounds",
                        severity="error",
                        message="Ball position is outside table bounds",
                        details={
                            "position": ball_state.position.to_dict(),
                            "table_width": table.width,
                            "table_height": table.height,
                            "ball_id": ball_state.id,
                        },
                        suggested_fix="Move ball to valid position within table bounds",
                    )
                )

        # Validate kinetic energy
        kinetic_energy = ball_state.kinetic_energy()
        max_kinetic_energy = 0.5 * ball_state.mass * self.max_velocity**2
        if kinetic_energy > max_kinetic_energy:
            errors.append(
                ValidationError(
                    error_type="kinetic_energy",
                    severity="error",
                    message="Ball kinetic energy exceeds physical limits",
                    details={
                        "kinetic_energy": kinetic_energy,
                        "max_kinetic_energy": max_kinetic_energy,
                        "ball_id": ball_state.id,
                    },
                )
            )

        is_valid = len(errors) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def validate_forces(self, forces: Vector2D, mass: float) -> ValidationResult:
        """Verify force calculations are reasonable and within limits.

        Args:
            forces: Total force vector acting on object
            mass: Mass of the object the forces are acting on

        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []

        # Validate force magnitude
        force_magnitude = forces.magnitude()
        if force_magnitude > self.max_force:
            warnings.append(
                ValidationError(
                    error_type="force_magnitude",
                    severity="warning",
                    message="Force magnitude is unusually high",
                    details={
                        "force_magnitude": force_magnitude,
                        "max_force": self.max_force,
                    },
                )
            )

        # Validate resulting acceleration
        if mass > 0:
            acceleration = force_magnitude / mass
            if acceleration > self.max_acceleration:
                errors.append(
                    ValidationError(
                        error_type="acceleration_limit",
                        severity="error",
                        message="Resulting acceleration exceeds physical limits",
                        details={
                            "acceleration": acceleration,
                            "max_acceleration": self.max_acceleration,
                            "force": force_magnitude,
                            "mass": mass,
                        },
                        suggested_fix=f"Reduce force or increase mass to limit acceleration to {self.max_acceleration} m/s²",
                    )
                )
        else:
            errors.append(
                ValidationError(
                    error_type="invalid_mass",
                    severity="error",
                    message="Mass must be positive for force validation",
                    details={"mass": mass},
                )
            )

        # Check for NaN or infinite forces
        if not (math.isfinite(forces.x) and math.isfinite(forces.y)):
            errors.append(
                ValidationError(
                    error_type="invalid_force_values",
                    severity="critical",
                    message="Force contains invalid values (NaN or infinite)",
                    details={"force_x": forces.x, "force_y": forces.y},
                    suggested_fix="Check force calculation for division by zero or invalid operations",
                )
            )

        is_valid = len(errors) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.4 + len(warnings) * 0.1))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def validate_energy_conservation(
        self, before_states: list[BallState], after_states: list[BallState]
    ) -> ValidationResult:
        """Validate energy conservation in collisions and system evolution.

        Args:
            before_states: Ball states before interaction
            after_states: Ball states after interaction

        Returns:
            ValidationResult with energy conservation validation
        """
        errors = []
        warnings = []

        if not self.conservation_checks:
            return ValidationResult(True, [], [], 1.0)

        # Calculate total kinetic energy before and after
        energy_before = sum(ball.kinetic_energy() for ball in before_states)
        energy_after = sum(ball.kinetic_energy() for ball in after_states)

        if energy_before == 0:
            # Can't validate conservation if no initial energy
            warnings.append(
                ValidationError(
                    error_type="zero_initial_energy",
                    severity="warning",
                    message="Cannot validate energy conservation with zero initial energy",
                    details={"energy_before": energy_before},
                )
            )
            return ValidationResult(True, [], warnings, 0.5)

        # Calculate energy change
        energy_change = abs(energy_after - energy_before)
        energy_ratio = energy_change / energy_before

        if energy_ratio > self.energy_tolerance:
            severity = "error" if energy_ratio > 0.2 else "warning"
            errors.append(
                ValidationError(
                    error_type="energy_conservation",
                    severity=severity,
                    message="Energy conservation violation detected",
                    details={
                        "energy_before": energy_before,
                        "energy_after": energy_after,
                        "energy_change": energy_change,
                        "relative_change": energy_ratio,
                        "tolerance": self.energy_tolerance,
                    },
                    suggested_fix="Check collision resolution and force calculations",
                )
            )

        # Energy should generally decrease (due to friction, inelastic collisions)
        if energy_after > energy_before * (1 + self.energy_tolerance):
            errors.append(
                ValidationError(
                    error_type="energy_increase",
                    severity="error",
                    message="Energy increased without external input",
                    details={
                        "energy_before": energy_before,
                        "energy_after": energy_after,
                        "energy_gain": energy_after - energy_before,
                    },
                    suggested_fix="Check for energy-adding bugs in physics calculations",
                )
            )

        is_valid = len([e for e in errors if e.severity == "error"]) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def validate_momentum_conservation(
        self, before_states: list[BallState], after_states: list[BallState]
    ) -> ValidationResult:
        """Validate momentum conservation in collisions.

        Args:
            before_states: Ball states before collision
            after_states: Ball states after collision

        Returns:
            ValidationResult with momentum conservation validation
        """
        errors = []
        warnings = []

        if not self.conservation_checks:
            return ValidationResult(True, [], [], 1.0)

        # Calculate total momentum before and after
        momentum_before = Vector2D(0, 0)
        momentum_after = Vector2D(0, 0)

        for ball in before_states:
            momentum_before.x += ball.mass * ball.velocity.x
            momentum_before.y += ball.mass * ball.velocity.y

        for ball in after_states:
            momentum_after.x += ball.mass * ball.velocity.x
            momentum_after.y += ball.mass * ball.velocity.y

        # Calculate momentum change
        momentum_change = Vector2D(
            momentum_after.x - momentum_before.x, momentum_after.y - momentum_before.y
        )
        momentum_change_mag = momentum_change.magnitude()

        # Calculate relative change
        momentum_before_mag = momentum_before.magnitude()
        if momentum_before_mag > 0:
            relative_change = momentum_change_mag / momentum_before_mag

            if relative_change > self.momentum_tolerance:
                severity = "error" if relative_change > 0.1 else "warning"
                errors.append(
                    ValidationError(
                        error_type="momentum_conservation",
                        severity=severity,
                        message="Momentum conservation violation detected",
                        details={
                            "momentum_before": momentum_before.to_dict(),
                            "momentum_after": momentum_after.to_dict(),
                            "momentum_change": momentum_change.to_dict(),
                            "relative_change": relative_change,
                            "tolerance": self.momentum_tolerance,
                        },
                        suggested_fix="Check collision resolution calculations",
                    )
                )
        else:
            # No initial momentum - after momentum should also be zero or small
            if momentum_change_mag > 0.001:  # Small absolute tolerance
                warnings.append(
                    ValidationError(
                        error_type="momentum_creation",
                        severity="warning",
                        message="Momentum created from zero initial momentum",
                        details={
                            "momentum_after": momentum_after.to_dict(),
                            "momentum_magnitude": momentum_change_mag,
                        },
                    )
                )

        is_valid = len([e for e in errors if e.severity == "error"]) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def validate_system_state(
        self, balls: list[BallState], table: TableState
    ) -> ValidationResult:
        """Comprehensive validation of entire system state.

        Args:
            balls: All balls in the system
            table: Table state

        Returns:
            ValidationResult with comprehensive system validation
        """
        errors = []
        warnings = []

        # Validate each ball individually
        for ball in balls:
            ball_result = self.validate_ball_state(ball, table)
            errors.extend(ball_result.errors)
            warnings.extend(ball_result.warnings)

        # Check for ball overlaps
        active_balls = [b for b in balls if not b.is_pocketed]
        for i, ball1 in enumerate(active_balls):
            for ball2 in active_balls[i + 1 :]:
                distance = ball1.distance_to(ball2)
                min_distance = ball1.radius + ball2.radius

                if distance < min_distance - self.position_tolerance:
                    errors.append(
                        ValidationError(
                            error_type="ball_overlap",
                            severity="error",
                            message=f"Balls {ball1.id} and {ball2.id} are overlapping",
                            details={
                                "ball1_id": ball1.id,
                                "ball2_id": ball2.id,
                                "distance": distance,
                                "min_distance": min_distance,
                                "overlap": min_distance - distance,
                            },
                            suggested_fix="Separate overlapping balls",
                        )
                    )

        # Check for exactly one cue ball
        cue_balls = [b for b in balls if b.is_cue_ball and not b.is_pocketed]
        if len(cue_balls) != 1:
            errors.append(
                ValidationError(
                    error_type="cue_ball_count",
                    severity="error",
                    message=f"Expected exactly 1 active cue ball, found {len(cue_balls)}",
                    details={"cue_ball_count": len(cue_balls)},
                    suggested_fix="Ensure exactly one cue ball is active on the table",
                )
            )

        # Validate total system energy
        total_energy = sum(ball.kinetic_energy() for ball in balls)
        max_reasonable_energy = len(balls) * 0.5 * 0.17 * self.max_velocity**2

        if total_energy > max_reasonable_energy:
            warnings.append(
                ValidationError(
                    error_type="system_energy",
                    severity="warning",
                    message="Total system energy is unusually high",
                    details={
                        "total_energy": total_energy,
                        "max_reasonable": max_reasonable_energy,
                    },
                )
            )

        is_valid = len([e for e in errors if e.severity in ["error", "critical"]]) == 0
        confidence = max(0.0, 1.0 - (len(errors) * 0.2 + len(warnings) * 0.05))

        return ValidationResult(is_valid, errors, warnings, confidence)

    def get_validation_summary(self, results: list[ValidationResult]) -> dict[str, Any]:
        """Generate a summary of multiple validation results.

        Args:
            results: List of validation results to summarize

        Returns:
            Dictionary with validation summary statistics
        """
        total_errors = []
        total_warnings = []

        for result in results:
            total_errors.extend(result.errors)
            total_warnings.extend(result.warnings)

        # Count errors by type
        error_types = {}
        warning_types = {}

        for error in total_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1

        for warning in total_warnings:
            warning_types[warning.error_type] = (
                warning_types.get(warning.error_type, 0) + 1
            )

        # Calculate overall confidence
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
        else:
            avg_confidence = 1.0

        return {
            "total_validations": len(results),
            "passed_validations": len([r for r in results if r.is_valid]),
            "failed_validations": len([r for r in results if not r.is_valid]),
            "total_errors": len(total_errors),
            "total_warnings": len(total_warnings),
            "error_types": error_types,
            "warning_types": warning_types,
            "average_confidence": avg_confidence,
            "overall_valid": all(r.is_valid for r in results),
        }
