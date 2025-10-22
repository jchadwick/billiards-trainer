"""Geometric collision detection for billiards trajectory prediction.

This module implements elegant geometric collision detection inspired by the Cassapa
pool detection system. It uses pure geometry rather than complex physics:
- Line-circle intersection using quadratic formula for ball collisions
- Geometric reflection (θ_reflected = π - 2θ_incident) for cushion bounces
- Simple force transfer along collision normal for ball-ball impacts
- Optimized with bounding box quick rejection

Based on cassapa/code/detector/pool.cpp implementation.

4K Migration (Group 5):
All collision detection now operates in 4K pixels (3840×2160).
- Use BALL_RADIUS_4K (36 pixels) for all ball radius calculations
- All distances are in pixels
- All collision points include scale metadata [1.0, 1.0] (4K canonical)
"""

import math
from dataclasses import dataclass
from typing import Optional

from ..constants_4k import BALL_RADIUS_4K
from ..coordinates import Vector2D
from ..models import BallState, TableState


@dataclass
class GeometricCollision:
    """Result of geometric collision detection.

    4K Migration (Group 5):
    - distance: in 4K pixels
    - hit_point: Vector2D in 4K pixels with scale=[1.0, 1.0]
    - cushion_normal: Vector2D (direction only, no scale)
    """

    distance: float  # Distance along ray to collision point (4K pixels)
    hit_point: Vector2D  # Collision point coordinates (4K pixels, scale=[1.0, 1.0])
    collision_type: str  # "ball", "cushion", "pocket", "none"
    ball_id: Optional[str] = None  # ID of ball that was hit (for ball collisions)
    cushion_side: Optional[str] = None  # "top", "bottom", "left", "right"
    cushion_normal: Optional[Vector2D] = (
        None  # Normal vector for cushion (direction only)
    )
    pocket_id: Optional[int] = None  # Pocket ID (for pocket collisions)


class GeometricCollisionDetector:
    """Geometric collision detection using line-circle intersection mathematics.

    This implementation follows the elegant approach from cassapa pool.cpp:
    - Minimal complexity, maximum clarity
    - Pure geometric calculations
    - No complex physics engine overhead
    - Optimized with early rejection tests

    4K Migration (Group 5):
    All collision detection operates in 4K pixels (3840×2160):
    - Default ball radius is BALL_RADIUS_4K (36 pixels)
    - All distances are in pixels
    - All collision points have scale=[1.0, 1.0] (4K canonical)
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize geometric collision detector.

        4K Migration: All coordinates and radii should be in 4K pixels.
        Use BALL_RADIUS_4K (36 pixels) for ball_radius parameters.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.epsilon = 1e-6  # Small value to avoid division by zero
        self.ball_radius_4k = BALL_RADIUS_4K  # Standard ball radius in 4K pixels (36)

    def check_line_circle_intersection(
        self,
        line_start: Vector2D,
        line_end: Vector2D,
        circle_center: Vector2D,
        circle_radius: float,
    ) -> Optional[GeometricCollision]:
        """Check if a line segment intersects a circle using quadratic formula.

        Based on cassapa pool.cpp CheckIfLineCrossesBall() lines 737-826.

        Algorithm:
        1. Quick rejection test using bounding box
        2. Handle vertical/near-vertical lines specially
        3. For general case, solve line-circle intersection using quadratic formula:
           Given circle (p, q, r) and line y = mx + h:
           A*x² + B*x + C = 0 where:
           A = m² + 1
           B = 2(mh - mq - p)
           C = q² - r² + p² - 2hq + h²

        4K Migration (Group 5):
        All coordinates are in 4K pixels (3840×2160).
        Use BALL_RADIUS_4K (36 pixels) for circle_radius when checking ball collisions.

        Args:
            line_start: Start point of line segment (4K pixels)
            line_end: End point of line segment (4K pixels)
            circle_center: Center of circle (4K pixels)
            circle_radius: Radius of circle (4K pixels, use BALL_RADIUS_4K for balls)

        Returns:
            GeometricCollision if intersection found, None otherwise
            - hit_point will have scale=[1.0, 1.0] (4K canonical)
            - distance will be in 4K pixels
        """
        # Quick rejection: bounding box test (cassapa pool.cpp:737-747)
        if not self._bounding_box_check(
            line_start, line_end, circle_center, circle_radius
        ):
            return None

        # Calculate line equation: y = mx + h
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y

        # Handle vertical or near-vertical lines (cassapa pool.cpp:750-768)
        if abs(dx) < 3.0:
            return self._vertical_line_circle_intersection(
                line_start, line_end, circle_center, circle_radius
            )

        # Calculate line slope and intercept
        m = dy / dx
        h = line_start.y - m * line_start.x

        # Quadratic formula for line-circle intersection (cassapa pool.cpp:770-826)
        p = circle_center.x
        q = circle_center.y
        r = circle_radius

        A = m**2 + 1.0
        B = 2.0 * (m * h - m * q - p)
        C = q**2 - r**2 + p**2 - 2.0 * h * q + h**2

        discriminant = B**2 - 4.0 * A * C

        # Tolerance for tangent case (cassapa pool.cpp:785-789)
        if discriminant > -0.00001:
            if discriminant < 0.0001:
                discriminant = 0.0  # Treat as tangent

            # Two intersection points
            sqrt_d = math.sqrt(discriminant)
            xi1 = (-B - sqrt_d) / (2.0 * A)
            xi2 = (-B + sqrt_d) / (2.0 * A)
            yi1 = m * xi1 + h
            yi2 = m * xi2 + h

            # Choose the closer intersection to line_start (cassapa pool.cpp:800-812)
            dist1 = self._point_distance(line_start.x, line_start.y, xi1, yi1)
            dist2 = self._point_distance(line_start.x, line_start.y, xi2, yi2)

            if dist1 < dist2:
                hit_x, hit_y, distance = xi1, yi1, dist1
            else:
                hit_x, hit_y, distance = xi2, yi2, dist2

            # Verify hit point is within line segment bounds
            if self._point_on_segment(line_start, line_end, Vector2D(hit_x, hit_y)):
                return GeometricCollision(
                    distance=distance,
                    hit_point=Vector2D(hit_x, hit_y),
                    collision_type="ball",
                )

        return None

    def _bounding_box_check(
        self,
        line_start: Vector2D,
        line_end: Vector2D,
        circle_center: Vector2D,
        circle_radius: float,
    ) -> bool:
        """Quick rejection test using bounding box (cassapa pool.cpp:737-747).

        Args:
            line_start: Line start point
            line_end: Line end point
            circle_center: Circle center
            circle_radius: Circle radius

        Returns:
            True if circle could intersect line segment, False otherwise
        """
        # Get bounding box of line segment
        if line_start.x < line_end.x:
            xa, xb = line_start.x, line_end.x
        else:
            xa, xb = line_end.x, line_start.x

        if line_start.y < line_end.y:
            ya, yb = line_start.y, line_end.y
        else:
            ya, yb = line_end.y, line_start.y

        # Check if circle is completely outside bounding box (with radius margin)
        if circle_center.x + circle_radius < xa or circle_center.x - circle_radius > xb:
            return False

        if circle_center.y + circle_radius < ya or circle_center.y - circle_radius > yb:
            return False

        return True

    def _vertical_line_circle_intersection(
        self,
        line_start: Vector2D,
        line_end: Vector2D,
        circle_center: Vector2D,
        circle_radius: float,
    ) -> Optional[GeometricCollision]:
        """Handle vertical/near-vertical line-circle intersection (cassapa pool.cpp:750-768).

        Uses trigonometry instead of line equations to avoid numerical issues.

        Args:
            line_start: Line start point
            line_end: Line end point
            circle_center: Circle center
            circle_radius: Circle radius

        Returns:
            GeometricCollision if intersection found, None otherwise
        """
        hit_x = line_start.x
        dx = abs(line_start.x - circle_center.x)

        # Circle doesn't reach the line
        if dx > circle_radius:
            return None

        # Use trigonometry to find intersection
        angle = math.acos(dx / circle_radius)
        dy = abs(math.sin(angle)) * circle_radius

        # Determine which intersection point based on line direction
        if line_start.y < line_end.y:  # Line going up
            hit_y = circle_center.y - dy
        else:  # Line going down
            hit_y = circle_center.y + dy

        # Check if hit point is within line segment
        hit_point = Vector2D(hit_x, hit_y)
        if self._point_on_segment(line_start, line_end, hit_point):
            distance = self._point_distance(line_start.x, line_start.y, hit_x, hit_y)
            return GeometricCollision(
                distance=distance,
                hit_point=hit_point,
                collision_type="ball",
            )

        return None

    def _point_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points (cassapa pool_utils.cpp:33-40).

        Args:
            x1: First point x
            y1: First point y
            x2: Second point x
            y2: Second point y

        Returns:
            Distance between points
        """
        d1 = x1 - x2
        d2 = y1 - y2
        return math.sqrt(d1**2 + d2**2)

    def _point_on_segment(
        self, seg_start: Vector2D, seg_end: Vector2D, point: Vector2D
    ) -> bool:
        """Check if point lies on line segment.

        Args:
            seg_start: Segment start
            seg_end: Segment end
            point: Point to check

        Returns:
            True if point is on segment, False otherwise
        """
        # Check if point is within segment bounding box (with small epsilon)
        epsilon = 0.001
        min_x = min(seg_start.x, seg_end.x) - epsilon
        max_x = max(seg_start.x, seg_end.x) + epsilon
        min_y = min(seg_start.y, seg_end.y) - epsilon
        max_y = max(seg_start.y, seg_end.y) + epsilon

        return min_x <= point.x <= max_x and min_y <= point.y <= max_y

    def find_closest_ball_collision(
        self,
        ray_start: Vector2D,
        ray_direction: Vector2D,
        balls: list[BallState],
        moving_ball_id: str,
        ball_radius: float,
        max_distance: Optional[float] = None,
    ) -> Optional[GeometricCollision]:
        """Find closest ball collision along a ray (cassapa pool.cpp FindBallHitByThisBall:687-736).

        4K Migration (Group 5):
        All coordinates and distances are in 4K pixels.
        Use BALL_RADIUS_4K (36 pixels) for ball_radius parameter.

        Args:
            ray_start: Starting point of ray (4K pixels)
            ray_direction: Direction vector (should be normalized)
            balls: List of all balls (positions in 4K pixels)
            moving_ball_id: ID of moving ball (to exclude from checks)
            ball_radius: Radius of balls (4K pixels, use BALL_RADIUS_4K = 36)
            max_distance: Maximum distance to check in 4K pixels (optional)

        Returns:
            GeometricCollision for closest ball hit, or None
            - hit_point will have scale=[1.0, 1.0] (4K canonical)
            - distance will be in 4K pixels
        """
        closest_collision = None
        min_distance = max_distance if max_distance else float("inf")

        # Extend ray to a far endpoint for line segment representation
        ray_length = max_distance if max_distance else 10000.0
        ray_end = Vector2D(
            ray_start.x + ray_direction.x * ray_length,
            ray_start.y + ray_direction.y * ray_length,
        )

        for ball in balls:
            # Skip the moving ball itself and pocketed balls
            if ball.id == moving_ball_id or ball.is_pocketed:
                continue

            # Check intersection with combined radius (2 * ball_radius)
            collision = self.check_line_circle_intersection(
                ray_start, ray_end, ball.position, 2.0 * ball_radius
            )

            if collision and collision.distance < min_distance:
                min_distance = collision.distance
                collision.ball_id = ball.id
                closest_collision = collision

        return closest_collision

    def calculate_geometric_reflection(
        self, velocity: Vector2D, cushion_side: str, elasticity: float = 0.95
    ) -> Vector2D:
        """Calculate reflected velocity using geometric reflection (cassapa pool.cpp:612-614).

        Formula: θ_reflected = π - 2θ_incident

        4K Migration (Group 5):
        Velocity components are in 4K pixels/second (or pixels/frame).

        Args:
            velocity: Incident velocity vector (4K pixels/time)
            cushion_side: "top", "bottom", "left", or "right"
            elasticity: Energy retention coefficient (default 0.95)

        Returns:
            Reflected velocity vector (4K pixels/time)
        """
        speed = velocity.magnitude() * elasticity

        # Current angle
        angle = math.atan2(velocity.y, velocity.x)

        # Calculate reflection based on cushion orientation
        if cushion_side in ["top", "bottom"]:
            # Reflect across horizontal (negate y component)
            new_angle = -angle
        else:  # "left" or "right"
            # Reflect across vertical (π - angle)
            new_angle = math.pi - angle

        return Vector2D(speed * math.cos(new_angle), speed * math.sin(new_angle))

    def calculate_ball_collision_velocities(
        self,
        moving_ball: BallState,
        target_ball: BallState,
        collision_point: Vector2D,
    ) -> tuple[Vector2D, Vector2D]:
        """Calculate post-collision velocities for ball-ball collision.

        Uses simplified collision physics from cassapa: force transfers along
        the line connecting ball centers (cassapa pool.cpp:103-116).

        4K Migration (Group 5):
        All positions and velocities are in 4K pixels.
        Ball positions should be in 4K pixels with scale=[1.0, 1.0].

        Args:
            moving_ball: The ball that is moving (position/velocity in 4K pixels)
            target_ball: The ball that is hit (position in 4K pixels)
            collision_point: Point where collision occurs (4K pixels)

        Returns:
            Tuple of (moving_ball_new_velocity, target_ball_new_velocity)
            Both velocities in 4K pixels/time
        """
        # Direction from collision point to target ball center
        # This represents the force transfer direction
        force_direction = Vector2D(
            target_ball.position.x - collision_point.x,
            target_ball.position.y - collision_point.y,
        )
        force_direction = force_direction.normalize()

        # Project moving ball's velocity onto collision normal
        moving_ball.velocity.magnitude()
        normal_component = (
            moving_ball.velocity.x * force_direction.x
            + moving_ball.velocity.y * force_direction.y
        )

        # Simplified physics: target ball gets velocity in collision direction
        # Speed transfer is proportional to normal component
        transfer_speed = abs(normal_component) * 0.85  # Energy loss factor

        target_velocity = Vector2D(
            force_direction.x * transfer_speed,
            force_direction.y * transfer_speed,
        )

        # Moving ball retains tangential component and reduced normal component
        tangent = Vector2D(-force_direction.y, force_direction.x)
        tangent_component = (
            moving_ball.velocity.x * tangent.x + moving_ball.velocity.y * tangent.y
        )

        # Reduced normal component (some energy transferred to target ball)
        remaining_normal = normal_component * 0.3

        moving_velocity = Vector2D(
            force_direction.x * remaining_normal + tangent.x * tangent_component,
            force_direction.y * remaining_normal + tangent.y * tangent_component,
        )

        return moving_velocity, target_velocity

    def find_cushion_intersection(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[GeometricCollision]:
        """Find intersection with table cushions (cassapa pool.cpp:471-504).

        Supports both calibrated playing area corners and rectangular bounds.

        4K Migration (Group 5):
        All coordinates and distances are in 4K pixels.
        Use BALL_RADIUS_4K (36 pixels) for ball_radius parameter.

        Args:
            position: Current ball position (4K pixels)
            direction: Ball direction (normalized)
            table: Table state (dimensions in 4K pixels)
            ball_radius: Ball radius (4K pixels, use BALL_RADIUS_4K = 36)

        Returns:
            GeometricCollision for cushion hit, or None
            - hit_point will have scale=[1.0, 1.0] (4K canonical)
            - distance will be in 4K pixels
            - cushion_normal will be a unit vector (no scale)
        """
        # Use playing area corners if available (calibrated system)
        if table.playing_area_corners and len(table.playing_area_corners) == 4:
            return self._find_cushion_intersection_polygon(
                position, direction, table, ball_radius
            )

        # Fall back to rectangular bounds
        return self._find_cushion_intersection_rectangular(
            position, direction, table, ball_radius
        )

    def _find_cushion_intersection_polygon(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[GeometricCollision]:
        """Find cushion intersection for calibrated playing area polygon.

        Args:
            position: Current ball position
            direction: Ball direction (normalized)
            table: Table state with playing_area_corners
            ball_radius: Ball radius

        Returns:
            GeometricCollision for cushion hit, or None
        """
        corners = table.playing_area_corners
        if not corners or len(corners) != 4:
            return None

        # Define 4 edges: top, right, bottom, left
        edges = [
            (corners[0], corners[1], "top"),
            (corners[1], corners[2], "right"),
            (corners[2], corners[3], "bottom"),
            (corners[3], corners[0], "left"),
        ]

        closest_collision = None
        min_distance = float("inf")

        # Extend ray for intersection testing
        ray_end = Vector2D(
            position.x + direction.x * 10000.0, position.y + direction.y * 10000.0
        )

        for seg_start, seg_end, cushion_name in edges:
            # Calculate inward normal
            edge_vec = Vector2D(seg_end.x - seg_start.x, seg_end.y - seg_start.y)
            edge_normal = Vector2D(-edge_vec.y, edge_vec.x).normalize()

            # Offset edge inward by ball radius
            offset_start = Vector2D(
                seg_start.x + edge_normal.x * ball_radius,
                seg_start.y + edge_normal.y * ball_radius,
            )
            offset_end = Vector2D(
                seg_end.x + edge_normal.x * ball_radius,
                seg_end.y + edge_normal.y * ball_radius,
            )

            # Find line-line intersection
            result = self._line_segment_intersection(
                position, ray_end, offset_start, offset_end
            )

            if result:
                hit_point, distance = result
                if distance < min_distance:
                    min_distance = distance
                    closest_collision = GeometricCollision(
                        distance=distance,
                        hit_point=hit_point,
                        collision_type="cushion",
                        cushion_side=cushion_name,
                        cushion_normal=edge_normal,
                    )

        return closest_collision

    def _find_cushion_intersection_rectangular(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[GeometricCollision]:
        """Find cushion intersection for rectangular table bounds.

        Args:
            position: Current ball position
            direction: Ball direction (normalized)
            table: Table state
            ball_radius: Ball radius

        Returns:
            GeometricCollision for cushion hit, or None
        """
        candidates = []

        # Right cushion
        if direction.x > 0:
            t = (table.width - ball_radius - position.x) / direction.x
            if t > 0:
                y = position.y + direction.y * t
                if ball_radius <= y <= table.height - ball_radius:
                    candidates.append(
                        GeometricCollision(
                            distance=t,
                            hit_point=Vector2D(table.width - ball_radius, y),
                            collision_type="cushion",
                            cushion_side="right",
                            cushion_normal=Vector2D(-1, 0),
                        )
                    )

        # Left cushion
        if direction.x < 0:
            t = (ball_radius - position.x) / direction.x
            if t > 0:
                y = position.y + direction.y * t
                if ball_radius <= y <= table.height - ball_radius:
                    candidates.append(
                        GeometricCollision(
                            distance=t,
                            hit_point=Vector2D(ball_radius, y),
                            collision_type="cushion",
                            cushion_side="left",
                            cushion_normal=Vector2D(1, 0),
                        )
                    )

        # Top cushion
        if direction.y > 0:
            t = (table.height - ball_radius - position.y) / direction.y
            if t > 0:
                x = position.x + direction.x * t
                if ball_radius <= x <= table.width - ball_radius:
                    candidates.append(
                        GeometricCollision(
                            distance=t,
                            hit_point=Vector2D(x, table.height - ball_radius),
                            collision_type="cushion",
                            cushion_side="top",
                            cushion_normal=Vector2D(0, -1),
                        )
                    )

        # Bottom cushion
        if direction.y < 0:
            t = (ball_radius - position.y) / direction.y
            if t > 0:
                x = position.x + direction.x * t
                if ball_radius <= x <= table.width - ball_radius:
                    candidates.append(
                        GeometricCollision(
                            distance=t,
                            hit_point=Vector2D(x, ball_radius),
                            collision_type="cushion",
                            cushion_side="bottom",
                            cushion_normal=Vector2D(0, 1),
                        )
                    )

        # Return nearest cushion
        if candidates:
            return min(candidates, key=lambda c: c.distance)
        return None

    def _line_segment_intersection(
        self,
        line1_start: Vector2D,
        line1_end: Vector2D,
        line2_start: Vector2D,
        line2_end: Vector2D,
    ) -> Optional[tuple[Vector2D, float]]:
        """Find intersection between two line segments.

        Args:
            line1_start: First line start
            line1_end: First line end
            line2_start: Second line start
            line2_end: Second line end

        Returns:
            Tuple of (intersection_point, distance_from_line1_start) or None
        """
        # Direction vectors
        dir1 = Vector2D(line1_end.x - line1_start.x, line1_end.y - line1_start.y)
        dir2 = Vector2D(line2_end.x - line2_start.x, line2_end.y - line2_start.y)

        # Calculate cross product
        denom = dir1.x * dir2.y - dir1.y * dir2.x

        # Lines are parallel
        if abs(denom) < self.epsilon:
            return None

        # Vector from line1 start to line2 start
        diff = Vector2D(line2_start.x - line1_start.x, line2_start.y - line1_start.y)

        # Calculate parameters
        t = (diff.x * dir2.y - diff.y * dir2.x) / denom
        u = (diff.x * dir1.y - diff.y * dir1.x) / denom

        # Check if intersection is within both segments
        if t > 0.001 and 0 <= u <= 1:
            intersection = Vector2D(
                line1_start.x + t * dir1.x, line1_start.y + t * dir1.y
            )
            distance = math.sqrt((t * dir1.x) ** 2 + (t * dir1.y) ** 2)
            return (intersection, distance)

        return None

    def find_pocket_intersection(
        self,
        position: Vector2D,
        direction: Vector2D,
        table: TableState,
        ball_radius: float,
    ) -> Optional[GeometricCollision]:
        """Find intersection with table pockets (cassapa pool.cpp:326-346).

        4K Migration (Group 5):
        All coordinates and distances are in 4K pixels.
        Use BALL_RADIUS_4K (36 pixels) for ball_radius parameter.

        Args:
            position: Current ball position (4K pixels)
            direction: Ball direction (normalized)
            table: Table state with pocket positions (4K pixels)
            ball_radius: Ball radius (4K pixels, use BALL_RADIUS_4K = 36)

        Returns:
            GeometricCollision for pocket hit, or None
            - hit_point will have scale=[1.0, 1.0] (4K canonical)
            - distance will be in 4K pixels
        """
        if not table.pocket_positions:
            return None

        closest_pocket = None
        min_distance = float("inf")

        # Extend ray for testing
        ray_end = Vector2D(
            position.x + direction.x * 10000.0, position.y + direction.y * 10000.0
        )

        for i, pocket_pos in enumerate(table.pocket_positions):
            # Check line-circle intersection with pocket
            collision = self.check_line_circle_intersection(
                position, ray_end, pocket_pos, table.pocket_radius
            )

            if collision and collision.distance < min_distance:
                min_distance = collision.distance
                collision.collision_type = "pocket"
                collision.pocket_id = i
                closest_pocket = collision

        return closest_pocket
