"""Coordinate transformation utilities for vision system.

This module provides comprehensive coordinate transformation capabilities for
converting between different coordinate systems used in the billiards vision
system, including camera coordinates, world coordinates, projector coordinates,
and table coordinates.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoordinateSystem(Enum):
    """Different coordinate systems used in the vision pipeline."""

    CAMERA = "camera"  # Camera image coordinates (pixels)
    WORLD = "world"  # Real-world coordinates (millimeters)
    TABLE = "table"  # Table surface coordinates (millimeters)
    PROJECTOR = "projector"  # Projector display coordinates (pixels)
    NORMALIZED = "normalized"  # Normalized coordinates (0.0 to 1.0)


@dataclass
class Point2D:
    """2D point representation."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)

    def distance_to(self, other: "Point2D") -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: "Point2D") -> "Point2D":
        """Add two points."""
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """Subtract two points."""
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """Multiply point by scalar."""
        return Point2D(self.x * scalar, self.y * scalar)


@dataclass
class Point3D:
    """3D point representation."""

    x: float
    y: float
    z: float

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_2d(self) -> Point2D:
        """Project to 2D by dropping z coordinate."""
        return Point2D(self.x, self.y)


@dataclass
class TransformationMatrix:
    """Container for transformation matrices and metadata."""

    matrix: np.ndarray
    source_system: CoordinateSystem
    target_system: CoordinateSystem
    is_homogeneous: bool = True
    calibration_error: Optional[float] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Validate transformation matrix."""
        if self.is_homogeneous:
            if self.matrix.shape not in [(3, 3), (4, 4)]:
                raise ValueError(
                    f"Homogeneous matrix must be 3x3 or 4x4, got {self.matrix.shape}"
                )
        else:
            if self.matrix.shape[1] != self.matrix.shape[0]:
                raise ValueError(
                    f"Transformation matrix must be square, got {self.matrix.shape}"
                )

    @property
    def inverse(self) -> "TransformationMatrix":
        """Get inverse transformation."""
        try:
            inv_matrix = np.linalg.inv(self.matrix)
            return TransformationMatrix(
                matrix=inv_matrix,
                source_system=self.target_system,
                target_system=self.source_system,
                is_homogeneous=self.is_homogeneous,
                calibration_error=self.calibration_error,
                timestamp=self.timestamp,
            )
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix is not invertible: {e}")


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""

    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients
    image_size: tuple[int, int]  # (width, height)
    reprojection_error: float = 0.0
    calibration_flags: int = 0

    @property
    def focal_length(self) -> tuple[float, float]:
        """Get focal lengths (fx, fy)."""
        return (self.camera_matrix[0, 0], self.camera_matrix[1, 1])

    @property
    def principal_point(self) -> tuple[float, float]:
        """Get principal point (cx, cy)."""
        return (self.camera_matrix[0, 2], self.camera_matrix[1, 2])


class CoordinateTransformer:
    """Comprehensive coordinate transformation system.

    Handles transformations between different coordinate systems used in the
    billiards vision pipeline, including camera calibration, perspective
    correction, and coordinate system conversions.
    """

    def __init__(self):
        """Initialize coordinate transformer."""
        # Transformation matrices
        self.transformations: dict[
            tuple[CoordinateSystem, CoordinateSystem], TransformationMatrix
        ] = {}

        # Camera calibration
        self.camera_calibration: Optional[CameraCalibration] = None

        # Table calibration
        self.table_corners_camera: Optional[list[Point2D]] = None
        self.table_corners_world: Optional[list[Point2D]] = None
        self.table_size: Optional[tuple[float, float]] = None  # (width, height) in mm

        logger.info("CoordinateTransformer initialized")

    def set_camera_calibration(self, calibration: CameraCalibration) -> None:
        """Set camera calibration parameters.

        Args:
            calibration: Camera calibration data
        """
        self.camera_calibration = calibration
        logger.info(
            f"Camera calibration set: {calibration.image_size}, error={calibration.reprojection_error:.3f}"
        )

    def set_table_calibration(
        self,
        camera_corners: list[Point2D],
        world_corners: list[Point2D],
        table_size: tuple[float, float],
    ) -> None:
        """Set table calibration data.

        Args:
            camera_corners: Table corners in camera coordinates
            world_corners: Table corners in world coordinates
            table_size: Table size (width, height) in millimeters
        """
        if len(camera_corners) != 4 or len(world_corners) != 4:
            raise ValueError("Exactly 4 corners required for table calibration")

        self.table_corners_camera = camera_corners
        self.table_corners_world = world_corners
        self.table_size = table_size

        # Compute homography from camera to table coordinates
        self._compute_table_homography()

        logger.info(f"Table calibration set: {table_size}mm")

    def add_transformation(self, transform: TransformationMatrix) -> None:
        """Add a transformation matrix.

        Args:
            transform: Transformation matrix with metadata
        """
        key = (transform.source_system, transform.target_system)
        self.transformations[key] = transform
        logger.debug(
            f"Added transformation: {transform.source_system.value} -> {transform.target_system.value}"
        )

    def transform_point(
        self,
        point: Union[Point2D, Point3D, tuple[float, float], tuple[float, float, float]],
        source_system: CoordinateSystem,
        target_system: CoordinateSystem,
    ) -> Union[Point2D, Point3D]:
        """Transform a point between coordinate systems.

        Args:
            point: Point to transform
            source_system: Source coordinate system
            target_system: Target coordinate system

        Returns:
            Transformed point

        Raises:
            ValueError: If transformation is not available
        """
        # Convert input to Point object
        if isinstance(point, tuple):
            if len(point) == 2:
                point = Point2D(point[0], point[1])
            elif len(point) == 3:
                point = Point3D(point[0], point[1], point[2])
            else:
                raise ValueError(
                    f"Point tuple must have 2 or 3 elements, got {len(point)}"
                )

        # Check for direct transformation
        key = (source_system, target_system)
        if key in self.transformations:
            return self._apply_transformation(point, self.transformations[key])

        # Check for inverse transformation
        inverse_key = (target_system, source_system)
        if inverse_key in self.transformations:
            return self._apply_transformation(
                point, self.transformations[inverse_key].inverse
            )

        # Try to find path through intermediate systems
        path = self._find_transformation_path(source_system, target_system)
        if path:
            current_point = point
            for i in range(len(path) - 1):
                current_point = self.transform_point(
                    current_point, path[i], path[i + 1]
                )
            return current_point

        raise ValueError(
            f"No transformation available from {source_system.value} to {target_system.value}"
        )

    def transform_points(
        self,
        points: list[Union[Point2D, Point3D, tuple[float, float]]],
        source_system: CoordinateSystem,
        target_system: CoordinateSystem,
    ) -> list[Union[Point2D, Point3D]]:
        """Transform multiple points between coordinate systems.

        Args:
            points: List of points to transform
            source_system: Source coordinate system
            target_system: Target coordinate system

        Returns:
            List of transformed points
        """
        return [self.transform_point(p, source_system, target_system) for p in points]

    def undistort_point(self, point: Point2D) -> Point2D:
        """Remove camera distortion from a point.

        Args:
            point: Distorted camera point

        Returns:
            Undistorted camera point

        Raises:
            ValueError: If camera calibration is not set
        """
        if not self.camera_calibration:
            raise ValueError("Camera calibration required for undistortion")

        # Convert to numpy format
        pts = np.array([[point.x, point.y]], dtype=np.float32)

        # Undistort
        undistorted = cv2.undistortPoints(
            pts,
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coeffs,
            P=self.camera_calibration.camera_matrix,
        )

        return Point2D(undistorted[0][0][0], undistorted[0][0][1])

    def undistort_points(self, points: list[Point2D]) -> list[Point2D]:
        """Remove camera distortion from multiple points.

        Args:
            points: List of distorted camera points

        Returns:
            List of undistorted camera points
        """
        if not points:
            return []

        if not self.camera_calibration:
            raise ValueError("Camera calibration required for undistortion")

        # Convert to numpy format
        pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)

        # Undistort
        undistorted = cv2.undistortPoints(
            pts,
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coeffs,
            P=self.camera_calibration.camera_matrix,
        )

        return [Point2D(pt[0][0], pt[0][1]) for pt in undistorted]

    def normalize_coordinates(
        self, point: Point2D, image_size: tuple[int, int]
    ) -> Point2D:
        """Normalize pixel coordinates to 0-1 range.

        Args:
            point: Point in pixel coordinates
            image_size: Image size (width, height)

        Returns:
            Normalized point (0-1 range)
        """
        return Point2D(point.x / image_size[0], point.y / image_size[1])

    def denormalize_coordinates(
        self, point: Point2D, image_size: tuple[int, int]
    ) -> Point2D:
        """Convert normalized coordinates back to pixel coordinates.

        Args:
            point: Normalized point (0-1 range)
            image_size: Image size (width, height)

        Returns:
            Point in pixel coordinates
        """
        return Point2D(point.x * image_size[0], point.y * image_size[1])

    def compute_perspective_transform(
        self, source_points: list[Point2D], target_points: list[Point2D]
    ) -> np.ndarray:
        """Compute perspective transformation matrix.

        Args:
            source_points: Source points (at least 4)
            target_points: Target points (same count as source)

        Returns:
            3x3 perspective transformation matrix

        Raises:
            ValueError: If insufficient points or mismatched counts
        """
        if len(source_points) < 4 or len(target_points) < 4:
            raise ValueError(
                "At least 4 point pairs required for perspective transform"
            )

        if len(source_points) != len(target_points):
            raise ValueError("Source and target point counts must match")

        # Convert to numpy arrays
        src_pts = np.array([p.to_tuple() for p in source_points[:4]], dtype=np.float32)
        dst_pts = np.array([p.to_tuple() for p in target_points[:4]], dtype=np.float32)

        # Compute perspective transform
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    def apply_perspective_transform(
        self, points: list[Point2D], transform_matrix: np.ndarray
    ) -> list[Point2D]:
        """Apply perspective transformation to points.

        Args:
            points: Points to transform
            transform_matrix: 3x3 perspective transformation matrix

        Returns:
            Transformed points
        """
        if not points:
            return []

        # Convert to homogeneous coordinates
        pts = np.array([p.to_tuple() for p in points], dtype=np.float32)
        pts = np.column_stack([pts, np.ones(len(pts))])

        # Apply transformation
        transformed = (transform_matrix @ pts.T).T

        # Convert back to Cartesian coordinates
        result = []
        for pt in transformed:
            if pt[2] != 0:
                result.append(Point2D(pt[0] / pt[2], pt[1] / pt[2]))
            else:
                result.append(Point2D(pt[0], pt[1]))

        return result

    def estimate_table_plane(
        self, table_points_3d: list[Point3D]
    ) -> tuple[np.ndarray, float]:
        """Estimate table plane from 3D points.

        Args:
            table_points_3d: 3D points on the table surface

        Returns:
            Tuple of (plane_normal, plane_distance)
        """
        if len(table_points_3d) < 3:
            raise ValueError("At least 3 points required to estimate plane")

        # Convert to numpy array
        points = np.array([p.to_tuple() for p in table_points_3d])

        # Fit plane using SVD
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD to find normal vector
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last row is the normal vector

        # Compute distance from origin
        distance = np.dot(normal, centroid)

        return normal, distance

    def project_to_table_plane(
        self, point_3d: Point3D, plane_normal: np.ndarray, plane_distance: float
    ) -> Point3D:
        """Project a 3D point onto the table plane.

        Args:
            point_3d: 3D point to project
            plane_normal: Table plane normal vector
            plane_distance: Distance from origin to plane

        Returns:
            Projected point on table plane
        """
        point = point_3d.to_array()

        # Calculate distance from point to plane
        dist_to_plane = np.dot(plane_normal, point) - plane_distance

        # Project onto plane
        projected = point - dist_to_plane * plane_normal

        return Point3D(projected[0], projected[1], projected[2])

    def compute_camera_to_world_transform(
        self, camera_points: list[Point2D], world_points: list[Point3D]
    ) -> TransformationMatrix:
        """Compute transformation from camera to world coordinates using PnP.

        Args:
            camera_points: 2D points in camera coordinates
            world_points: Corresponding 3D points in world coordinates

        Returns:
            Camera to world transformation matrix

        Raises:
            ValueError: If camera calibration is not set or insufficient points
        """
        if not self.camera_calibration:
            raise ValueError("Camera calibration required for PnP")

        if len(camera_points) < 4 or len(world_points) < 4:
            raise ValueError("At least 4 point correspondences required for PnP")

        if len(camera_points) != len(world_points):
            raise ValueError("Camera and world point counts must match")

        # Convert to numpy arrays
        image_points = np.array([p.to_tuple() for p in camera_points], dtype=np.float32)
        object_points = np.array([p.to_tuple() for p in world_points], dtype=np.float32)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coeffs,
        )

        if not success:
            raise ValueError("PnP solution failed")

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = tvec.flatten()

        return TransformationMatrix(
            matrix=transform,
            source_system=CoordinateSystem.CAMERA,
            target_system=CoordinateSystem.WORLD,
            is_homogeneous=True,
        )

    def _compute_table_homography(self) -> None:
        """Compute homography from camera to table coordinates."""
        if not self.table_corners_camera or not self.table_corners_world:
            return

        # Compute homography
        homography = self.compute_perspective_transform(
            self.table_corners_camera, self.table_corners_world
        )

        # Store transformation
        transform = TransformationMatrix(
            matrix=homography,
            source_system=CoordinateSystem.CAMERA,
            target_system=CoordinateSystem.TABLE,
            is_homogeneous=True,
        )

        self.add_transformation(transform)

    def _apply_transformation(
        self, point: Union[Point2D, Point3D], transform: TransformationMatrix
    ) -> Union[Point2D, Point3D]:
        """Apply a transformation matrix to a point."""
        if isinstance(point, Point2D):
            if transform.is_homogeneous:
                # Homogeneous 2D transformation
                homogeneous = np.array([point.x, point.y, 1.0])
                transformed = transform.matrix @ homogeneous

                if transformed.shape[0] >= 3 and transformed[2] != 0:
                    return Point2D(
                        transformed[0] / transformed[2], transformed[1] / transformed[2]
                    )
                else:
                    return Point2D(transformed[0], transformed[1])
            else:
                # Non-homogeneous 2D transformation
                coord = np.array([point.x, point.y])
                transformed = transform.matrix @ coord
                return Point2D(transformed[0], transformed[1])

        elif isinstance(point, Point3D):
            if transform.is_homogeneous:
                # Homogeneous 3D transformation
                homogeneous = np.array([point.x, point.y, point.z, 1.0])
                transformed = transform.matrix @ homogeneous

                if transformed.shape[0] >= 4 and transformed[3] != 0:
                    return Point3D(
                        transformed[0] / transformed[3],
                        transformed[1] / transformed[3],
                        transformed[2] / transformed[3],
                    )
                else:
                    return Point3D(transformed[0], transformed[1], transformed[2])
            else:
                # Non-homogeneous 3D transformation
                coord = np.array([point.x, point.y, point.z])
                transformed = transform.matrix @ coord
                return Point3D(transformed[0], transformed[1], transformed[2])

        else:
            raise ValueError(f"Unsupported point type: {type(point)}")

    def _find_transformation_path(
        self, source: CoordinateSystem, target: CoordinateSystem
    ) -> Optional[list[CoordinateSystem]]:
        """Find a path of transformations from source to target."""
        # Simple BFS to find transformation path
        visited = set()
        queue = [(source, [source])]

        while queue:
            current, path = queue.pop(0)

            if current == target:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Find all systems we can transform to from current
            for src, dst in self.transformations:
                if src == current and dst not in visited:
                    queue.append((dst, path + [dst]))
                elif dst == current and src not in visited:
                    queue.append((src, path + [src]))

        return None

    def get_transformation_error(
        self, source_system: CoordinateSystem, target_system: CoordinateSystem
    ) -> Optional[float]:
        """Get calibration error for a transformation.

        Args:
            source_system: Source coordinate system
            target_system: Target coordinate system

        Returns:
            Calibration error or None if not available
        """
        key = (source_system, target_system)
        if key in self.transformations:
            return self.transformations[key].calibration_error

        inverse_key = (target_system, source_system)
        if inverse_key in self.transformations:
            return self.transformations[inverse_key].calibration_error

        return None

    def is_transformation_available(
        self, source_system: CoordinateSystem, target_system: CoordinateSystem
    ) -> bool:
        """Check if transformation is available between coordinate systems.

        Args:
            source_system: Source coordinate system
            target_system: Target coordinate system

        Returns:
            True if transformation is available
        """
        if source_system == target_system:
            return True

        key = (source_system, target_system)
        if key in self.transformations:
            return True

        inverse_key = (target_system, source_system)
        if inverse_key in self.transformations:
            return True

        # Check for path through intermediate systems
        return self._find_transformation_path(source_system, target_system) is not None

    def get_available_transformations(
        self,
    ) -> list[tuple[CoordinateSystem, CoordinateSystem]]:
        """Get list of available transformations.

        Returns:
            List of (source, target) coordinate system pairs
        """
        transformations = list(self.transformations.keys())

        # Add inverse transformations
        for source, target in list(transformations):
            transformations.append((target, source))

        return list(set(transformations))


# Convenience functions for common transformations


def create_perspective_matrix(
    source_corners: list[tuple[float, float]], target_corners: list[tuple[float, float]]
) -> np.ndarray:
    """Create perspective transformation matrix from corner points.

    Args:
        source_corners: Source corner points
        target_corners: Target corner points

    Returns:
        3x3 perspective transformation matrix
    """
    transformer = CoordinateTransformer()
    src_points = [Point2D(x, y) for x, y in source_corners]
    dst_points = [Point2D(x, y) for x, y in target_corners]
    return transformer.compute_perspective_transform(src_points, dst_points)


def apply_perspective_correction(
    points: list[tuple[float, float]],
    source_corners: list[tuple[float, float]],
    target_corners: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Apply perspective correction to a list of points.

    Args:
        points: Points to transform
        source_corners: Source quadrilateral corners
        target_corners: Target quadrilateral corners

    Returns:
        Transformed points
    """
    transformer = CoordinateTransformer()

    # Create transformation matrix
    matrix = create_perspective_matrix(source_corners, target_corners)

    # Transform points
    point_objects = [Point2D(x, y) for x, y in points]
    transformed = transformer.apply_perspective_transform(point_objects, matrix)

    return [p.to_tuple() for p in transformed]


def normalize_table_coordinates(
    points: list[tuple[float, float]], table_size: tuple[float, float]
) -> list[tuple[float, float]]:
    """Normalize table coordinates to 0-1 range.

    Args:
        points: Points in table coordinates (mm)
        table_size: Table size (width, height) in mm

    Returns:
        Normalized points (0-1 range)
    """
    width, height = table_size
    return [(x / width, y / height) for x, y in points]


def denormalize_table_coordinates(
    points: list[tuple[float, float]], table_size: tuple[float, float]
) -> list[tuple[float, float]]:
    """Convert normalized coordinates back to table coordinates.

    Args:
        points: Normalized points (0-1 range)
        table_size: Table size (width, height) in mm

    Returns:
        Points in table coordinates (mm)
    """
    width, height = table_size
    return [(x * width, y * height) for x, y in points]
