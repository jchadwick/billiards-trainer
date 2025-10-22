"""Geometric utility functions for billiards calculations.

All spatial operations work in 4K canonical pixel coordinates (3840×2160).
Distances, positions, and radii are expressed in pixels.

IMPORTANT: All Vector2D instances should include scale metadata to ensure
proper coordinate space tracking. See constants_4k.py for canonical dimensions.
"""

import math

from config import Config, config

from ..coordinates import Vector2D


def _get_config() -> Config:
    """Get the global configuration instance."""
    return config


# Convenience functions for simple coordinate-based calculations
def angle_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate angle in degrees from point (x1, y1) to point (x2, y2).

    Args:
        x1: X coordinate of first point in 4K pixels
        y1: Y coordinate of first point in 4K pixels
        x2: X coordinate of second point in 4K pixels
        y2: Y coordinate of second point in 4K pixels

    Returns:
        Angle in degrees from point 1 to point 2

    Note:
        All coordinates should be in 4K canonical space for consistency.
    """
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        x1: X coordinate of first point in 4K pixels
        y1: Y coordinate of first point in 4K pixels
        x2: X coordinate of second point in 4K pixels
        y2: Y coordinate of second point in 4K pixels

    Returns:
        Distance between the two points in 4K pixels

    Note:
        All coordinates should be in 4K canonical space for consistency.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_vector(x: float, y: float) -> tuple[float, float]:
    """Normalize a 2D vector.

    Args:
        x: X component of vector
        y: Y component of vector

    Returns:
        Tuple of (normalized_x, normalized_y). Returns (0, 0) for zero vector.
    """
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return (0.0, 0.0)
    return (x / magnitude, y / magnitude)


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Check if point (x, y) is inside polygon using ray casting algorithm.

    Args:
        x: X coordinate of point to test
        y: Y coordinate of point to test
        polygon: List of (x, y) tuples defining the polygon vertices

    Returns:
        True if point is inside polygon, False otherwise
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class GeometryUtils:
    """Geometric calculations and utilities for billiards physics."""

    @staticmethod
    def distance_between_points(
        p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        """Calculate distance between two points.

        Args:
            p1: First point (x, y) in 4K pixels
            p2: Second point (x, y) in 4K pixels

        Returns:
            Distance in 4K pixels
        """
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def distance_between_vectors(v1: Vector2D, v2: Vector2D) -> float:
        """Calculate distance between two vector points.

        Args:
            v1: First point in 4K pixels
            v2: Second point in 4K pixels

        Returns:
            Distance in 4K pixels

        Note:
            Both vectors should be in the same coordinate space (4K canonical).
        """
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
        """Find intersection points between line segment and circle.

        Args:
            line_start: Line start point in 4K pixels
            line_end: Line end point in 4K pixels
            circle_center: Circle center in 4K pixels
            radius: Circle radius in 4K pixels

        Returns:
            List of intersection points in 4K pixels (0-2 points)

        Note:
            All coordinates should be in 4K canonical space.
        """
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
                intersections.append(Vector2D(x, y, scale=line_start.scale))

        return intersections

    @staticmethod
    def point_line_distance(
        point: Vector2D, line_start: Vector2D, line_end: Vector2D
    ) -> float:
        """Calculate shortest distance from point to line segment.

        Args:
            point: Point in 4K pixels
            line_start: Line start point in 4K pixels
            line_end: Line end point in 4K pixels

        Returns:
            Distance in 4K pixels

        Note:
            All coordinates should be in 4K canonical space.
        """
        # Vector from line_start to line_end
        line_vec = Vector2D(
            line_end.x - line_start.x, line_end.y - line_start.y, scale=line_start.scale
        )
        line_length_sq = line_vec.x**2 + line_vec.y**2

        if line_length_sq == 0:
            # Line is a point
            return GeometryUtils.distance_between_vectors(point, line_start)

        # Vector from line_start to point
        point_vec = Vector2D(
            point.x - line_start.x, point.y - line_start.y, scale=point.scale
        )

        # Project point onto line
        t = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_length_sq
        t = max(0, min(1, t))  # Clamp to line segment

        # Find closest point on line segment
        closest = Vector2D(
            line_start.x + t * line_vec.x,
            line_start.y + t * line_vec.y,
            scale=line_start.scale,
        )

        return GeometryUtils.distance_between_vectors(point, closest)

    @staticmethod
    def reflect_vector(incident: Vector2D, normal: Vector2D) -> Vector2D:
        """Reflect a vector across a surface normal."""
        # Ensure normal is normalized
        n = normal.normalize()

        # Calculate reflection: v' = v - 2(v·n)n
        dot_product = incident.x * n.x + incident.y * n.y
        return Vector2D(
            incident.x - 2 * dot_product * n.x,
            incident.y - 2 * dot_product * n.y,
            scale=incident.scale,
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
        return Vector2D(rotated_x + center.x, rotated_y + center.y, scale=point.scale)

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
        """Find intersection points between two circles.

        Args:
            center1: First circle center in 4K pixels
            radius1: First circle radius in 4K pixels
            center2: Second circle center in 4K pixels
            radius2: Second circle radius in 4K pixels

        Returns:
            List of intersection points in 4K pixels (0-2 points)

        Note:
            All coordinates should be in 4K canonical space.
        """
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
            scale=center1.scale,
        )

        intersection2 = Vector2D(
            p2_x - h * (center2.y - center1.y) / d,
            p2_y + h * (center2.x - center1.x) / d,
            scale=center1.scale,
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

        # Get tolerance from config
        config = _get_config()
        tolerance = config.get(
            "core.utils.geometry.tolerance.triangle_point_test", default=1e-10
        )

        # Point is inside if sum of sub-triangle areas equals main triangle area
        return abs(area - (area1 + area2 + area3)) < tolerance

    @staticmethod
    def lerp(start: Vector2D, end: Vector2D, t: float) -> Vector2D:
        """Linear interpolation between two points."""
        return Vector2D(
            start.x + t * (end.x - start.x),
            start.y + t * (end.y - start.y),
            scale=start.scale,
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
            scale=p0.scale,
        )

        return p

    @staticmethod
    def smooth_path(
        points: list[Vector2D], smoothing_factor: float | None = None
    ) -> list[Vector2D]:
        """Apply smoothing to a path of points.

        Args:
            points: List of points defining the path
            smoothing_factor: Smoothing factor (0.0 = no smoothing, 1.0 = full smoothing).
                            If None, uses config value.

        Returns:
            Smoothed path
        """
        if len(points) < 3:
            return points.copy()

        # Get smoothing factor from config if not provided
        if smoothing_factor is None:
            config = _get_config()
            smoothing_factor = config.get(
                "core.utils.geometry.smoothing.default_factor", default=0.1
            )

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

            smoothed.append(Vector2D(final_x, final_y, scale=curr_point.scale))

        smoothed.append(points[-1])  # Keep last point
        return smoothed


def find_ball_cue_is_pointing_at(
    cue_tip: tuple[float, float] | Vector2D,
    cue_direction: tuple[float, float] | Vector2D | None = None,
    cue_angle: float | None = None,
    balls: list[tuple[float, float] | Vector2D] = None,
    max_perpendicular_distance: float = 40.0,
) -> int | None:
    """Find which ball the cue is currently pointing at.

    This function determines which ball (if any) lies along the cue stick's
    trajectory using perpendicular distance from the cue line. It's useful for
    determining aim assistance, target ball identification, and shot prediction.

    REFACTORING NOTE: This is a NEW utility function created during the refactoring
    to centralize ball-targeting logic that was previously duplicated in video_debugger.py
    and integration_service.py. Now used system-wide for consistent ball targeting.

    Algorithm:
    1. Calculate cue direction vector from either:
       - Provided direction vector (dx, dy), or
       - Calculated from angle (angle_rad = radians(cue_angle))
    2. Normalize direction vector to unit length
    3. For each ball:
       - Calculate vector from cue tip to ball center
       - Project onto cue direction (dot product) = distance along cue line
       - Calculate perpendicular distance (cross product magnitude)
       - Skip balls behind cue tip (negative projection)
       - Keep balls within max_perpendicular_distance threshold
    4. Return index of closest ball along cue direction

    Args:
        cue_tip: Position of the cue tip in 4K pixels as (x, y) tuple or Vector2D
        cue_direction: Direction vector the cue is pointing as (dx, dy) tuple or Vector2D.
            If None, cue_angle must be provided.
        cue_angle: Angle of the cue in degrees (0 = pointing right, counter-clockwise positive).
            Only used if cue_direction is None.
        balls: List of ball positions in 4K pixels as (x, y) tuples or Vector2D objects
        max_perpendicular_distance: Maximum perpendicular distance in 4K pixels from the
            cue line for a ball to be considered a target. Default is 40 pixels (approximately
            one ball diameter in 4K canonical space where ball radius = 36 pixels).

    Returns:
        Index of the ball the cue is pointing at (index in balls list), or None if no
        ball is close enough to the cue line.

    Raises:
        ValueError: If neither cue_direction nor cue_angle is provided, or if required
            parameters are missing.

    Example:
        >>> # All coordinates in 4K pixels
        >>> cue_tip = (1920, 1080)  # Table center
        >>> cue_angle = 45.0  # pointing up-right
        >>> balls = [(2100, 1260), (2300, 1440), (2500, 1000)]
        >>> target_idx = find_ball_cue_is_pointing_at(
        ...     cue_tip, cue_angle=cue_angle, balls=balls
        ... )
        >>> if target_idx is not None:
        ...     print(f"Aiming at ball at {balls[target_idx]}")

    Note:
        - All coordinates should be in 4K canonical space (3840×2160)
        - The function uses perpendicular distance (cross product) to determine proximity
        - Only balls in front of the cue tip (positive projection) are considered
        - If multiple balls are within threshold, the closest one along cue direction is returned
        - Works with both tuple (x, y) and Vector2D inputs for flexibility
        - Default threshold of 40px ≈ 1 ball diameter (ball radius in 4K is 36px)
    """
    if balls is None or len(balls) == 0:
        return None

    # Convert cue_tip to tuple if needed
    if isinstance(cue_tip, Vector2D):
        tip_x, tip_y = cue_tip.x, cue_tip.y
    else:
        tip_x, tip_y = cue_tip

    # Calculate direction vector
    if cue_direction is not None:
        if isinstance(cue_direction, Vector2D):
            dir_x, dir_y = cue_direction.x, cue_direction.y
        else:
            dir_x, dir_y = cue_direction
    elif cue_angle is not None:
        # Convert angle to direction vector
        angle_rad = math.radians(cue_angle)
        dir_x = math.cos(angle_rad)
        dir_y = math.sin(angle_rad)
    else:
        raise ValueError("Either cue_direction or cue_angle must be provided")

    # Normalize direction vector
    dir_length = math.sqrt(dir_x**2 + dir_y**2)
    if dir_length == 0:
        return None

    dir_x /= dir_length
    dir_y /= dir_length

    # Find the closest ball along the cue direction
    closest_ball_idx = None
    min_distance_along_cue = float("inf")

    for i, ball_pos in enumerate(balls):
        # Convert ball position to tuple if needed
        if isinstance(ball_pos, Vector2D):
            ball_x, ball_y = ball_pos.x, ball_pos.y
        else:
            ball_x, ball_y = ball_pos

        # Vector from cue tip to ball center
        ball_dx = ball_x - tip_x
        ball_dy = ball_y - tip_y

        # Distance along cue direction (dot product)
        distance_along_cue = ball_dx * dir_x + ball_dy * dir_y

        # Skip balls behind the cue tip
        if distance_along_cue < 0:
            continue

        # Calculate perpendicular distance from cue line to ball center (cross product)
        perpendicular_distance = abs(ball_dx * dir_y - ball_dy * dir_x)

        # Check if ball is within tolerance and closer than current closest
        if (
            perpendicular_distance < max_perpendicular_distance
            and distance_along_cue < min_distance_along_cue
        ):
            min_distance_along_cue = distance_along_cue
            closest_ball_idx = i

    return closest_ball_idx
