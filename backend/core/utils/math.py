"""Mathematical utility functions for billiards calculations."""

import math
from typing import Callable, Optional

from ...config.manager import ConfigurationModule
from ..models import Vector2D

# Global configuration instance (lazy loaded)
_config: Optional[ConfigurationModule] = None


def _get_config() -> ConfigurationModule:
    """Get or create configuration instance."""
    global _config
    if _config is None:
        from backend.config import config_manager

        _config = config_manager
    return _config


# Convenience functions that wrap MathUtils methods
def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + t * (b - a)


def normalize_angle(
    angle: float, min_angle: float = 0.0, max_angle: float = 360.0
) -> float:
    """Normalize angle to specified range (default: [0, 360) for degrees).

    Args:
        angle: Angle to normalize
        min_angle: Minimum angle in range (default 0.0)
        max_angle: Maximum angle in range (default 360.0)

    Returns:
        Angle normalized to [min_angle, max_angle) range
    """
    range_size = max_angle - min_angle
    while angle < min_angle:
        angle += range_size
    while angle >= max_angle:
        angle -= range_size
    return angle


class MathUtils:
    """Mathematical calculations and utilities for billiards physics."""

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b."""
        return a + t * (b - a)

    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0

    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """Convert radians to degrees."""
        return radians * 180.0 / math.pi

    @staticmethod
    def smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Smooth step function for smooth interpolation."""
        t = MathUtils.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def sign(value: float) -> int:
        """Return the sign of a value (-1, 0, or 1)."""
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    @staticmethod
    def approximately_equal(a: float, b: float, tolerance: float | None = None) -> bool:
        """Check if two floating point values are approximately equal.

        Args:
            a: First value
            b: Second value
            tolerance: Tolerance for comparison. If None, uses config value.

        Returns:
            True if values are approximately equal
        """
        if tolerance is None:
            config = _get_config()
            tolerance = config.get("core.utils.math.tolerance.default", default=1e-6)
        return abs(a - b) <= tolerance

    @staticmethod
    def approximately_zero(value: float, tolerance: float | None = None) -> bool:
        """Check if a floating point value is approximately zero.

        Args:
            value: Value to check
            tolerance: Tolerance for comparison. If None, uses config value.

        Returns:
            True if value is approximately zero
        """
        if tolerance is None:
            config = _get_config()
            tolerance = config.get(
                "core.utils.math.tolerance.zero_threshold", default=1e-6
            )
        return abs(value) <= tolerance

    @staticmethod
    def wrap_angle(
        angle: float,
        min_angle: float | None = None,
        max_angle: float | None = None,
    ) -> float:
        """Wrap angle to specified range (default: [-π, π]).

        Args:
            angle: Angle to wrap
            min_angle: Minimum angle in range. If None, uses config value.
            max_angle: Maximum angle in range. If None, uses config value.

        Returns:
            Wrapped angle
        """
        if min_angle is None or max_angle is None:
            config = _get_config()
            if min_angle is None:
                min_angle = config.get(
                    "core.utils.math.angles.wrap_min_radians", default=-math.pi
                )
            if max_angle is None:
                max_angle = config.get(
                    "core.utils.math.angles.wrap_max_radians", default=math.pi
                )

        range_size = max_angle - min_angle
        while angle < min_angle:
            angle += range_size
        while angle >= max_angle:
            angle -= range_size
        return angle

    @staticmethod
    def normalize_angle_0_2pi(angle: float) -> float:
        """Normalize angle to range [0, 2π]."""
        return MathUtils.wrap_angle(angle, 0, 2 * math.pi)

    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """Calculate the smallest difference between two angles."""
        diff = angle2 - angle1
        return MathUtils.wrap_angle(diff)

    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> list[float]:
        """Solve quadratic equation ax² + bx + c = 0."""
        if MathUtils.approximately_zero(a):
            # Linear equation
            if MathUtils.approximately_zero(b):
                return []  # No solution or infinite solutions
            return [-c / b]

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []  # No real solutions

        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-b - sqrt_discriminant) / (2 * a)
        x2 = (-b + sqrt_discriminant) / (2 * a)

        if MathUtils.approximately_equal(x1, x2):
            return [x1]  # One solution (repeated root)
        else:
            return sorted([x1, x2])  # Two solutions

    @staticmethod
    def remap(
        value: float, from_min: float, from_max: float, to_min: float, to_max: float
    ) -> float:
        """Remap value from one range to another."""
        if MathUtils.approximately_equal(from_max, from_min):
            return to_min
        return to_min + (value - from_min) * (to_max - to_min) / (from_max - from_min)

    @staticmethod
    def calculate_polynomial(coefficients: list[float], x: float) -> float:
        """Calculate polynomial value using Horner's method."""
        if not coefficients:
            return 0.0

        result = coefficients[0]
        for coeff in coefficients[1:]:
            result = result * x + coeff
        return result

    @staticmethod
    def moving_average(values: list[float], window_size: int) -> list[float]:
        """Calculate moving average with specified window size."""
        if window_size <= 0 or window_size > len(values):
            return values.copy()

        averaged = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            window_values = values[start_idx:end_idx]
            averaged.append(sum(window_values) / len(window_values))

        return averaged

    @staticmethod
    def gaussian(x: float, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """Calculate Gaussian (normal) distribution value."""
        variance = std_dev * std_dev
        return (1.0 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(
            -0.5 * ((x - mean) ** 2) / variance
        )

    @staticmethod
    def sigmoid(x: float, steepness: float = 1.0) -> float:
        """Calculate sigmoid function value."""
        return 1.0 / (1.0 + math.exp(-steepness * x))

    @staticmethod
    def spring_interpolation(
        current: float,
        target: float,
        velocity: float,
        spring_strength: float | None = None,
        damping: float | None = None,
        dt: float | None = None,
    ) -> tuple[float, float]:
        """Calculate spring-based interpolation for smooth motion.

        Args:
            current: Current position
            target: Target position
            velocity: Current velocity
            spring_strength: Spring strength coefficient. If None, uses config value.
            damping: Damping coefficient. If None, uses config value.
            dt: Time step. If None, uses config value.

        Returns:
            Tuple of (new_position, new_velocity)
        """
        config = _get_config()
        if spring_strength is None:
            spring_strength = config.get(
                "core.utils.math.interpolation.spring_strength", default=100.0
            )
        if damping is None:
            damping = config.get(
                "core.utils.math.interpolation.spring_damping", default=10.0
            )
        if dt is None:
            dt = config.get(
                "core.utils.math.interpolation.spring_timestep", default=0.016
            )

        spring_force = (target - current) * spring_strength
        damping_force = -velocity * damping
        acceleration = spring_force + damping_force

        new_velocity = velocity + acceleration * dt
        new_position = current + new_velocity * dt

        return new_position, new_velocity

    @staticmethod
    def exponential_decay(
        current_value: float, target_value: float, decay_rate: float, dt: float
    ) -> float:
        """Apply exponential decay towards target value."""
        return target_value + (current_value - target_value) * math.exp(
            -decay_rate * dt
        )

    @staticmethod
    def calculate_trajectory_parabola(
        initial_pos: Vector2D, initial_vel: Vector2D, gravity: float, time: float
    ) -> Vector2D:
        """Calculate position along parabolic trajectory (for projectile motion)."""
        x = initial_pos.x + initial_vel.x * time
        y = initial_pos.y + initial_vel.y * time - 0.5 * gravity * time * time
        return Vector2D(x, y)

    @staticmethod
    def find_roots_bisection(
        func: Callable[[float], float],
        left: float,
        right: float,
        tolerance: float | None = None,
        max_iterations: int | None = None,
    ) -> Optional[float]:
        """Find root of function using bisection method.

        Args:
            func: Function to find root for
            left: Left bound of interval
            right: Right bound of interval
            tolerance: Tolerance for convergence. If None, uses config value.
            max_iterations: Maximum iterations. If None, uses config value.

        Returns:
            Root of function or None if not found
        """
        config = _get_config()
        if tolerance is None:
            tolerance = config.get("core.utils.math.tolerance.bisection", default=1e-6)
        if max_iterations is None:
            max_iterations = config.get(
                "core.utils.math.numerical_methods.bisection_max_iterations",
                default=100,
            )

        if func(left) * func(right) > 0:
            return None  # No root in interval

        for _ in range(max_iterations):
            mid = (left + right) / 2
            mid_value = func(mid)

            if abs(mid_value) < tolerance:
                return mid

            if func(left) * mid_value < 0:
                right = mid
            else:
                left = mid

        return (left + right) / 2

    @staticmethod
    def calculate_centripetal_force(
        mass: float, velocity: float, radius: float
    ) -> float:
        """Calculate centripetal force for circular motion."""
        if MathUtils.approximately_zero(radius):
            return 0.0
        return mass * velocity * velocity / radius

    @staticmethod
    def calculate_angular_velocity(linear_velocity: float, radius: float) -> float:
        """Calculate angular velocity from linear velocity and radius."""
        if MathUtils.approximately_zero(radius):
            return 0.0
        return linear_velocity / radius

    @staticmethod
    def rotate_vector_2d(vector: Vector2D, angle: float) -> Vector2D:
        """Rotate a 2D vector by given angle (radians)."""
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        return Vector2D(
            vector.x * cos_angle - vector.y * sin_angle,
            vector.x * sin_angle + vector.y * cos_angle,
        )

    @staticmethod
    def reflect_vector_2d(incident: Vector2D, normal: Vector2D) -> Vector2D:
        """Reflect a vector across a surface normal."""
        normal_unit = normal.normalize()
        dot_product = incident.dot(normal_unit)
        return incident - (normal_unit * (2 * dot_product))

    @staticmethod
    def project_vector_onto_line(
        vector: Vector2D, line_direction: Vector2D
    ) -> Vector2D:
        """Project vector onto a line direction."""
        line_unit = line_direction.normalize()
        dot_product = vector.dot(line_unit)
        return line_unit * dot_product

    @staticmethod
    def perpendicular_vector_2d(vector: Vector2D) -> Vector2D:
        """Get perpendicular vector (90 degrees counter-clockwise)."""
        return Vector2D(-vector.y, vector.x)

    @staticmethod
    def calculate_momentum(mass: float, velocity: Vector2D) -> Vector2D:
        """Calculate momentum vector."""
        return velocity * mass

    @staticmethod
    def calculate_kinetic_energy(mass: float, velocity: Vector2D) -> float:
        """Calculate kinetic energy."""
        speed_squared = velocity.magnitude_squared()
        return 0.5 * mass * speed_squared

    @staticmethod
    def elastic_collision_1d(
        m1: float, v1: float, m2: float, v2: float, restitution: float = 1.0
    ) -> tuple[float, float]:
        """Calculate velocities after 1D elastic collision."""
        total_mass = m1 + m2
        if MathUtils.approximately_zero(total_mass):
            return v1, v2

        # Conservation of momentum
        momentum_transfer = 2 * (m1 * v1 + m2 * v2) / total_mass

        # Apply restitution
        relative_velocity = v1 - v2
        impulse = restitution * relative_velocity

        v1_new = (
            v1
            - (m2 / total_mass) * (momentum_transfer - 2 * v1)
            - (m2 / total_mass) * impulse
        )
        v2_new = (
            v2
            + (m1 / total_mass) * (momentum_transfer - 2 * v2)
            + (m1 / total_mass) * impulse
        )

        return v1_new, v2_new

    @staticmethod
    def calculate_friction_force(
        normal_force: float, friction_coefficient: float
    ) -> float:
        """Calculate friction force magnitude."""
        return normal_force * friction_coefficient

    @staticmethod
    def apply_drag_force(
        velocity: Vector2D, drag_coefficient: float, dt: float
    ) -> Vector2D:
        """Apply drag force to velocity."""
        speed = velocity.magnitude()
        if MathUtils.approximately_zero(speed):
            return velocity

        drag_magnitude = drag_coefficient * speed * speed
        drag_direction = velocity.normalize() * -1
        drag_acceleration = drag_direction * drag_magnitude

        new_velocity = velocity + (drag_acceleration * dt)

        # Prevent velocity reversal due to drag
        if new_velocity.dot(velocity) < 0:
            return Vector2D.zero()

        return new_velocity

    @staticmethod
    def calculate_rolling_resistance(
        velocity: Vector2D,
        mass: float,
        friction_coefficient: float,
        gravity: float | None = None,
    ) -> Vector2D:
        """Calculate rolling resistance force.

        Args:
            velocity: Velocity vector
            mass: Mass of object
            friction_coefficient: Coefficient of rolling friction
            gravity: Gravitational acceleration. If None, uses config value.

        Returns:
            Rolling resistance force vector
        """
        config = _get_config()
        if gravity is None:
            gravity = config.get(
                "core.utils.math.physics.default_gravity", default=9.81
            )

        velocity_threshold = config.get(
            "core.utils.math.physics.rolling_velocity_threshold", default=1e-6
        )

        if velocity.magnitude() < velocity_threshold:
            return Vector2D.zero()

        normal_force = mass * gravity
        friction_force = normal_force * friction_coefficient
        direction = velocity.normalize() * -1

        return direction * friction_force

    @staticmethod
    def is_point_in_circle(point: Vector2D, center: Vector2D, radius: float) -> bool:
        """Check if point is inside circle."""
        distance_squared = (point - center).magnitude_squared()
        return distance_squared <= radius * radius

    @staticmethod
    def circle_line_intersection_times(
        line_start: Vector2D,
        line_velocity: Vector2D,
        circle_center: Vector2D,
        circle_radius: float,
    ) -> list[float]:
        """Calculate times when moving point intersects circle."""
        # Relative position
        rel_pos = line_start - circle_center

        # Quadratic equation coefficients
        a = line_velocity.magnitude_squared()
        b = 2 * rel_pos.dot(line_velocity)
        c = rel_pos.magnitude_squared() - circle_radius * circle_radius

        return MathUtils.solve_quadratic(a, b, c)

    @staticmethod
    def calculate_impact_parameter(
        trajectory_start: Vector2D,
        trajectory_direction: Vector2D,
        target_center: Vector2D,
    ) -> float:
        """Calculate impact parameter (closest approach distance) for trajectory."""
        rel_pos = trajectory_start - target_center
        trajectory_unit = trajectory_direction.normalize()

        # Project relative position onto perpendicular to trajectory
        perpendicular = MathUtils.perpendicular_vector_2d(trajectory_unit)
        impact_parameter = abs(rel_pos.dot(perpendicular))

        return impact_parameter

    @staticmethod
    def numerical_derivative(
        func: Callable[[float], float], x: float, h: float | None = None
    ) -> float:
        """Calculate numerical derivative using central difference.

        Args:
            func: Function to differentiate
            x: Point at which to evaluate derivative
            h: Step size. If None, uses config value.

        Returns:
            Numerical derivative value
        """
        if h is None:
            config = _get_config()
            h = config.get(
                "core.utils.math.tolerance.numerical_derivative_step", default=1e-8
            )
        return float((func(x + h) - func(x - h)) / (2 * h))

    @staticmethod
    def numerical_integration_trapezoidal(
        func: Callable[[float], float], a: float, b: float, n: int | None = None
    ) -> float:
        """Numerical integration using trapezoidal rule.

        Args:
            func: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of steps. If None, uses config value.

        Returns:
            Numerical integration result
        """
        if n is None:
            config = _get_config()
            n = config.get(
                "core.utils.math.numerical_methods.integration_default_steps",
                default=1000,
            )

        h = (b - a) / n
        result = 0.5 * (func(a) + func(b))

        for i in range(1, n):
            x = a + i * h
            result += func(x)

        return float(result * h)

    @staticmethod
    def weighted_average(values: list[float], weights: list[float]) -> float:
        """Calculate weighted average."""
        if len(values) != len(weights) or len(values) == 0:
            return 0.0

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)

        if MathUtils.approximately_zero(weight_sum):
            return 0.0

        return weighted_sum / weight_sum
