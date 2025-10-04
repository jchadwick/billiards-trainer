"""Geometric calculation utilities for projector rendering.

This module provides comprehensive geometric utilities for coordinate transformations,
shape calculations, and spatial analysis for the projector system.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class CoordinateSystem(Enum):
    """Coordinate system types."""

    SCREEN = "screen"  # Screen coordinates (0,0 at top-left)
    CARTESIAN = "cartesian"  # Cartesian coordinates (0,0 at bottom-left)
    TABLE = "table"  # Table coordinates (physical measurements)
    NORMALIZED = "normalized"  # Normalized coordinates (0-1 range)


@dataclass
class Point2D:
    """2D point with utility methods."""

    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "Point2D") -> float:
        """Calculate angle to another point in radians."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle: float, center: Optional["Point2D"] = None) -> "Point2D":
        """Rotate point around center by angle (radians)."""
        if center is None:
            center = Point2D(0, 0)

        # Translate to origin
        x = self.x - center.x
        y = self.y - center.y

        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a

        # Translate back
        return Point2D(x_new + center.x, y_new + center.y)

    def scale(self, factor_x: float, factor_y: Optional[float] = None) -> "Point2D":
        """Scale point by given factors."""
        if factor_y is None:
            factor_y = factor_x
        return Point2D(self.x * factor_x, self.y * factor_y)

    def translate(self, dx: float, dy: float) -> "Point2D":
        """Translate point by given offsets."""
        return Point2D(self.x + dx, self.y + dy)

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def to_homogeneous(self) -> np.ndarray:
        """Convert to homogeneous coordinates."""
        return np.array([self.x, self.y, 1.0])

    @classmethod
    def from_tuple(cls, point: tuple[float, float]) -> "Point2D":
        """Create point from tuple."""
        return cls(point[0], point[1])

    @classmethod
    def from_homogeneous(cls, point: np.ndarray) -> "Point2D":
        """Create point from homogeneous coordinates."""
        if point[2] != 0:
            return cls(point[0] / point[2], point[1] / point[2])
        else:
            return cls(point[0], point[1])


@dataclass
class Vector2D:
    """2D vector with utility methods."""

    x: float
    y: float

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> "Vector2D":
        """Normalize vector to unit length."""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other: "Vector2D") -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2D") -> float:
        """Calculate cross product (z-component) with another vector."""
        return self.x * other.y - self.y * other.x

    def angle(self) -> float:
        """Get vector angle in radians."""
        return math.atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2D":
        """Rotate vector by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_new = self.x * cos_a - self.y * sin_a
        y_new = self.x * sin_a + self.y * cos_a
        return Vector2D(x_new, y_new)

    def scale(self, factor: float) -> "Vector2D":
        """Scale vector by factor."""
        return Vector2D(self.x * factor, self.y * factor)

    def perpendicular(self) -> "Vector2D":
        """Get perpendicular vector (90 degrees counter-clockwise)."""
        return Vector2D(-self.y, self.x)

    @classmethod
    def from_points(cls, start: Point2D, end: Point2D) -> "Vector2D":
        """Create vector from two points."""
        return cls(end.x - start.x, end.y - start.y)


@dataclass
class Line2D:
    """2D line with utility methods."""

    start: Point2D
    end: Point2D

    def length(self) -> float:
        """Calculate line length."""
        return self.start.distance_to(self.end)

    def direction(self) -> Vector2D:
        """Get line direction vector."""
        return Vector2D.from_points(self.start, self.end)

    def midpoint(self) -> Point2D:
        """Get line midpoint."""
        return Point2D((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    def point_at_parameter(self, t: float) -> Point2D:
        """Get point at parameter t (0=start, 1=end)."""
        x = self.start.x + t * (self.end.x - self.start.x)
        y = self.start.y + t * (self.end.y - self.start.y)
        return Point2D(x, y)

    def distance_to_point(self, point: Point2D) -> float:
        """Calculate distance from line to point."""
        # Use point-to-line distance formula
        line_vec = self.direction()
        line_length = self.length()

        if line_length == 0:
            return self.start.distance_to(point)

        # Project point onto line
        to_point = Vector2D.from_points(self.start, point)
        projection = to_point.dot(line_vec) / line_length

        # Clamp to line segment
        projection = max(0, min(line_length, projection))

        # Find closest point on line
        closest = self.point_at_parameter(projection / line_length)
        return closest.distance_to(point)

    def intersect_line(self, other: "Line2D") -> Optional[Point2D]:
        """Find intersection point with another line."""
        # Line 1: start + t1 * (end - start)
        # Line 2: other.start + t2 * (other.end - other.start)

        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x3, y3 = other.start.x, other.start.y
        x4, y4 = other.end.x, other.end.y

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:  # Lines are parallel
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Point2D(x, y)

        return None


@dataclass
class Circle:
    """Circle with utility methods."""

    center: Point2D
    radius: float

    def contains_point(self, point: Point2D) -> bool:
        """Check if circle contains point."""
        return self.center.distance_to(point) <= self.radius

    def intersect_line(self, line: Line2D) -> list[Point2D]:
        """Find intersection points with line."""
        # Vector from line start to circle center
        to_center = Vector2D.from_points(line.start, self.center)
        line_dir = line.direction().normalize()

        # Project center onto line
        projection_length = to_center.dot(line_dir)
        closest_point = Point2D(
            line.start.x + projection_length * line_dir.x,
            line.start.y + projection_length * line_dir.y,
        )

        # Distance from center to line
        distance = self.center.distance_to(closest_point)

        if distance > self.radius:
            return []  # No intersection

        if distance == self.radius:
            return [closest_point]  # Tangent

        # Two intersections
        chord_half_length = math.sqrt(self.radius**2 - distance**2)

        point1 = Point2D(
            closest_point.x - chord_half_length * line_dir.x,
            closest_point.y - chord_half_length * line_dir.y,
        )
        point2 = Point2D(
            closest_point.x + chord_half_length * line_dir.x,
            closest_point.y + chord_half_length * line_dir.y,
        )

        return [point1, point2]

    def intersect_circle(self, other: "Circle") -> list[Point2D]:
        """Find intersection points with another circle."""
        d = self.center.distance_to(other.center)

        # No intersection cases
        if d > self.radius + other.radius:  # Too far apart
            return []
        if d < abs(self.radius - other.radius):  # One inside the other
            return []
        if d == 0 and self.radius == other.radius:  # Same circle
            return []

        # Calculate intersection points
        a = (self.radius**2 - other.radius**2 + d**2) / (2 * d)
        h = math.sqrt(self.radius**2 - a**2)

        # Point on line between centers
        px = self.center.x + a * (other.center.x - self.center.x) / d
        py = self.center.y + a * (other.center.y - self.center.y) / d

        if h == 0:  # Tangent
            return [Point2D(px, py)]

        # Two intersection points
        point1 = Point2D(
            px + h * (other.center.y - self.center.y) / d,
            py - h * (other.center.x - self.center.x) / d,
        )
        point2 = Point2D(
            px - h * (other.center.y - self.center.y) / d,
            py + h * (other.center.x - self.center.x) / d,
        )

        return [point1, point2]


@dataclass
class Rectangle:
    """Rectangle with utility methods."""

    x: float
    y: float
    width: float
    height: float

    def contains_point(self, point: Point2D) -> bool:
        """Check if rectangle contains point."""
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def intersect_line(self, line: Line2D) -> list[Point2D]:
        """Find intersection points with line."""
        intersections = []

        # Create rectangle edges
        edges = [
            Line2D(
                Point2D(self.x, self.y), Point2D(self.x + self.width, self.y)
            ),  # Top
            Line2D(
                Point2D(self.x + self.width, self.y),
                Point2D(self.x + self.width, self.y + self.height),
            ),  # Right
            Line2D(
                Point2D(self.x + self.width, self.y + self.height),
                Point2D(self.x, self.y + self.height),
            ),  # Bottom
            Line2D(
                Point2D(self.x, self.y + self.height), Point2D(self.x, self.y)
            ),  # Left
        ]

        for edge in edges:
            intersection = line.intersect_line(edge)
            if intersection:
                intersections.append(intersection)

        return intersections

    def center(self) -> Point2D:
        """Get rectangle center."""
        return Point2D(self.x + self.width / 2, self.y + self.height / 2)

    def corners(self) -> list[Point2D]:
        """Get rectangle corners."""
        return [
            Point2D(self.x, self.y),  # Top-left
            Point2D(self.x + self.width, self.y),  # Top-right
            Point2D(self.x + self.width, self.y + self.height),  # Bottom-right
            Point2D(self.x, self.y + self.height),  # Bottom-left
        ]


class GeometryUtils:
    """Utility functions for geometric calculations."""

    @staticmethod
    def angle_between_vectors(v1: Vector2D, v2: Vector2D) -> float:
        """Calculate angle between two vectors in radians."""
        dot_product = v1.dot(v2)
        magnitudes = v1.magnitude() * v2.magnitude()

        if magnitudes == 0:
            return 0

        cos_angle = dot_product / magnitudes
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        return math.acos(cos_angle)

    @staticmethod
    def reflect_vector(incident: Vector2D, normal: Vector2D) -> Vector2D:
        """Reflect vector off surface with given normal."""
        normal_unit = normal.normalize()
        dot_product = incident.dot(normal_unit)
        reflected = Vector2D(
            incident.x - 2 * dot_product * normal_unit.x,
            incident.y - 2 * dot_product * normal_unit.y,
        )
        return reflected


class CoordinateTransform:
    """Coordinate system transformation utilities."""

    @staticmethod
    def create_transform_matrix(
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        rotation: float = 0.0,
        translation_x: float = 0.0,
        translation_y: float = 0.0,
    ) -> np.ndarray:
        """Create 2D transformation matrix.

        Args:
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
            rotation: Rotation in radians
            translation_x: X-axis translation
            translation_y: Y-axis translation

        Returns:
            3x3 homogeneous transformation matrix
        """
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        return np.array(
            [
                [scale_x * cos_r, -scale_x * sin_r, translation_x],
                [scale_y * sin_r, scale_y * cos_r, translation_y],
                [0, 0, 1],
            ]
        )

    @staticmethod
    def apply_transform(points: list[Point2D], transform: np.ndarray) -> list[Point2D]:
        """Apply transformation matrix to list of points."""
        transformed = []
        for point in points:
            homogeneous = point.to_homogeneous()
            transformed_homogeneous = transform @ homogeneous
            transformed.append(Point2D.from_homogeneous(transformed_homogeneous))
        return transformed

    @staticmethod
    def screen_to_cartesian(point: Point2D, screen_height: float) -> Point2D:
        """Convert screen coordinates to cartesian coordinates."""
        return Point2D(point.x, screen_height - point.y)

    @staticmethod
    def cartesian_to_screen(point: Point2D, screen_height: float) -> Point2D:
        """Convert cartesian coordinates to screen coordinates."""
        return Point2D(point.x, screen_height - point.y)

    @staticmethod
    def normalize_coordinates(point: Point2D, width: float, height: float) -> Point2D:
        """Normalize coordinates to 0-1 range."""
        return Point2D(point.x / width, point.y / height)

    @staticmethod
    def denormalize_coordinates(point: Point2D, width: float, height: float) -> Point2D:
        """Convert normalized coordinates back to pixel coordinates."""
        return Point2D(point.x * width, point.y * height)


# Convenience functions for common geometric operations


def calculate_trajectory_reflection(
    incident_point: Point2D,
    incident_direction: Vector2D,
    surface_start: Point2D,
    surface_end: Point2D,
) -> tuple[Point2D, Vector2D]:
    """Calculate reflection of trajectory off surface.

    Args:
        incident_point: Point where trajectory hits surface
        incident_direction: Direction of incoming trajectory
        surface_start: Start point of reflecting surface
        surface_end: End point of reflecting surface

    Returns:
        Tuple of (reflection_point, reflection_direction)
    """
    # Calculate surface normal
    surface_vector = Vector2D.from_points(surface_start, surface_end)
    surface_normal = surface_vector.perpendicular().normalize()

    # Ensure normal points away from incident direction
    if incident_direction.dot(surface_normal) > 0:
        surface_normal = Vector2D(-surface_normal.x, -surface_normal.y)

    # Calculate reflection direction
    reflection_direction = GeometryUtils.reflect_vector(
        incident_direction, surface_normal
    )

    return incident_point, reflection_direction


def find_circle_tangent_points(
    circle: Circle, external_point: Point2D
) -> list[Point2D]:
    """Find tangent points from external point to circle.

    Args:
        circle: Circle to find tangents to
        external_point: External point

    Returns:
        List of tangent points (0, 1, or 2 points)
    """
    distance = circle.center.distance_to(external_point)

    if distance < circle.radius:
        return []  # Point is inside circle

    if distance == circle.radius:
        return [external_point]  # Point is on circle

    # Calculate tangent points
    angle_to_center = external_point.angle_to(circle.center)
    tangent_angle = math.asin(circle.radius / distance)

    angle1 = angle_to_center + tangent_angle
    angle2 = angle_to_center - tangent_angle

    tangent1 = Point2D(
        circle.center.x + circle.radius * math.cos(angle1 + math.pi / 2),
        circle.center.y + circle.radius * math.sin(angle1 + math.pi / 2),
    )
    tangent2 = Point2D(
        circle.center.x + circle.radius * math.cos(angle2 - math.pi / 2),
        circle.center.y + circle.radius * math.sin(angle2 - math.pi / 2),
    )

    return [tangent1, tangent2]


def calculate_ball_collision_angles(
    ball1_center: Point2D, ball2_center: Point2D, ball1_velocity: Vector2D
) -> tuple[Vector2D, Vector2D]:
    """Calculate post-collision velocities for two balls.

    Args:
        ball1_center: Center of first ball
        ball2_center: Center of second ball
        ball1_velocity: Velocity of first ball (second ball assumed stationary)

    Returns:
        Tuple of (ball1_new_velocity, ball2_new_velocity)
    """
    # Normal vector from ball1 to ball2
    collision_normal = Vector2D.from_points(ball1_center, ball2_center).normalize()

    # Tangent vector (perpendicular to normal)
    collision_tangent = collision_normal.perpendicular()

    # Project velocities onto normal and tangent
    v1_normal = ball1_velocity.dot(collision_normal)
    v1_tangent = ball1_velocity.dot(collision_tangent)

    # After collision (assuming equal masses and elastic collision)
    # Ball 1 transfers all normal velocity to ball 2, keeps tangent velocity
    ball1_new = Vector2D(
        v1_tangent * collision_tangent.x, v1_tangent * collision_tangent.y
    )

    ball2_new = Vector2D(v1_normal * collision_normal.x, v1_normal * collision_normal.y)

    return ball1_new, ball2_new
