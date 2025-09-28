"""Geometric utility functions for billiards calculations."""

import math

from ..models import Vector2D


class GeometryUtils:
    """Geometric calculations and utilities for billiards physics."""

    @staticmethod
    def distance_between_points(
        p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def distance_between_vectors(v1: Vector2D, v2: Vector2D) -> float:
        """Calculate distance between two vector points."""
        return math.sqrt((v2.x - v1.x) ** 2 + (v2.y - v1.y) ** 2)

    @staticmethod
    def angle_between_vectors(
        v1: tuple[float, float], v2: tuple[float, float]
    ) -> float:
        """Calculate angle between two vectors in radians."""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = dot_product / (mag1 * mag2)
        # Clamp to prevent numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)

    @staticmethod
    def angle_from_points(p1: Vector2D, p2: Vector2D) -> float:
        """Calculate angle from point p1 to point p2 in radians."""
        return math.atan2(p2.y - p1.y, p2.x - p1.x)

    @staticmethod
    def line_circle_intersection(
        line_start: Vector2D, line_end: Vector2D, circle_center: Vector2D, radius: float
    ) -> list[Vector2D]:
        """Find intersection points between line segment and circle."""
        # Convert to relative coordinates (circle at origin)
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y
        fx = line_start.x - circle_center.x
        fy = line_start.y - circle_center.y

        # Quadratic equation coefficients
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return []  # No intersection

        sqrt_discriminant = math.sqrt(discriminant)

        # Calculate intersection parameters
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)

        intersections = []

        # Check if intersections are within line segment (0 <= t <= 1)
        for t in [t1, t2]:
            if 0 <= t <= 1:
                x = line_start.x + t * dx
                y = line_start.y + t * dy
                intersections.append(Vector2D(x, y))

        return intersections

    @staticmethod
    def point_line_distance(
        point: Vector2D, line_start: Vector2D, line_end: Vector2D
    ) -> float:
        """Calculate shortest distance from point to line segment."""
        # Vector from line_start to line_end
        line_vec = Vector2D(line_end.x - line_start.x, line_end.y - line_start.y)
        line_length_sq = line_vec.x**2 + line_vec.y**2

        if line_length_sq == 0:
            # Line is a point
            return GeometryUtils.distance_between_vectors(point, line_start)

        # Vector from line_start to point
        point_vec = Vector2D(point.x - line_start.x, point.y - line_start.y)

        # Project point onto line
        t = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_length_sq
        t = max(0, min(1, t))  # Clamp to line segment

        # Find closest point on line segment
        closest = Vector2D(line_start.x + t * line_vec.x, line_start.y + t * line_vec.y)

        return GeometryUtils.distance_between_vectors(point, closest)

    @staticmethod
    def reflect_vector(incident: Vector2D, normal: Vector2D) -> Vector2D:
        """Reflect a vector across a surface normal."""
        # Ensure normal is normalized
        n = normal.normalize()

        # Calculate reflection: v' = v - 2(v·n)n
        dot_product = incident.x * n.x + incident.y * n.y
        return Vector2D(
            incident.x - 2 * dot_product * n.x, incident.y - 2 * dot_product * n.y
        )

    @staticmethod
    def rotate_point(point: Vector2D, center: Vector2D, angle: float) -> Vector2D:
        """Rotate a point around a center by given angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Translate to origin
        dx = point.x - center.x
        dy = point.y - center.y

        # Rotate
        rotated_x = dx * cos_a - dy * sin_a
        rotated_y = dx * sin_a + dy * cos_a

        # Translate back
        return Vector2D(rotated_x + center.x, rotated_y + center.y)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to range [0, 2π)."""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle

    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """Calculate the smallest angle difference between two angles."""
        diff = angle2 - angle1
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        return diff

    @staticmethod
    def is_point_in_polygon(point: Vector2D, polygon: list[Vector2D]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point.x, point.y
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def circle_circle_intersection(
        center1: Vector2D, radius1: float, center2: Vector2D, radius2: float
    ) -> list[Vector2D]:
        """Find intersection points between two circles."""
        # Distance between centers
        d = GeometryUtils.distance_between_vectors(center1, center2)

        # Check for no intersection cases
        if d > radius1 + radius2:  # Too far apart
            return []
        if d < abs(radius1 - radius2):  # One circle inside the other
            return []
        if d == 0 and radius1 == radius2:  # Same circle
            return []  # Infinite intersections

        # Calculate intersection points
        a = (radius1**2 - radius2**2 + d**2) / (2 * d)
        h = math.sqrt(radius1**2 - a**2)

        # Point P2 is a + h*perpendicular to line between centers
        p2_x = center1.x + a * (center2.x - center1.x) / d
        p2_y = center1.y + a * (center2.y - center1.y) / d

        # Intersection points
        intersection1 = Vector2D(
            p2_x + h * (center2.y - center1.y) / d,
            p2_y - h * (center2.x - center1.x) / d,
        )

        intersection2 = Vector2D(
            p2_x - h * (center2.y - center1.y) / d,
            p2_y + h * (center2.x - center1.x) / d,
        )

        return [intersection1, intersection2]

    @staticmethod
    def calculate_triangle_area(p1: Vector2D, p2: Vector2D, p3: Vector2D) -> float:
        """Calculate area of triangle formed by three points."""
        return abs(
            (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2.0
        )

    @staticmethod
    def point_in_triangle(
        point: Vector2D, p1: Vector2D, p2: Vector2D, p3: Vector2D
    ) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        # Calculate area of main triangle
        area = GeometryUtils.calculate_triangle_area(p1, p2, p3)

        if area == 0:
            return False  # Degenerate triangle

        # Calculate areas of sub-triangles
        area1 = GeometryUtils.calculate_triangle_area(point, p2, p3)
        area2 = GeometryUtils.calculate_triangle_area(p1, point, p3)
        area3 = GeometryUtils.calculate_triangle_area(p1, p2, point)

        # Point is inside if sum of sub-triangle areas equals main triangle area
        return abs(area - (area1 + area2 + area3)) < 1e-10

    @staticmethod
    def lerp(start: Vector2D, end: Vector2D, t: float) -> Vector2D:
        """Linear interpolation between two points."""
        return Vector2D(
            start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)
        )

    @staticmethod
    def bezier_curve(
        p0: Vector2D, p1: Vector2D, p2: Vector2D, p3: Vector2D, t: float
    ) -> Vector2D:
        """Calculate point on cubic Bezier curve at parameter t."""
        u = 1 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t

        p = Vector2D(
            uuu * p0.x + 3 * uu * t * p1.x + 3 * u * tt * p2.x + ttt * p3.x,
            uuu * p0.y + 3 * uu * t * p1.y + 3 * u * tt * p2.y + ttt * p3.y,
        )

        return p

    @staticmethod
    def smooth_path(
        points: list[Vector2D], smoothing_factor: float = 0.1
    ) -> list[Vector2D]:
        """Apply smoothing to a path of points."""
        if len(points) < 3:
            return points.copy()

        smoothed = [points[0]]  # Keep first point

        for i in range(1, len(points) - 1):
            # Calculate smoothed position using neighboring points
            prev_point = points[i - 1]
            curr_point = points[i]
            next_point = points[i + 1]

            # Simple smoothing: weighted average
            smoothed_x = (prev_point.x + curr_point.x + next_point.x) / 3
            smoothed_y = (prev_point.y + curr_point.y + next_point.y) / 3

            # Blend with original point
            final_x = curr_point.x + smoothing_factor * (smoothed_x - curr_point.x)
            final_y = curr_point.y + smoothing_factor * (smoothed_y - curr_point.y)

            smoothed.append(Vector2D(final_x, final_y))

        smoothed.append(points[-1])  # Keep last point
        return smoothed
