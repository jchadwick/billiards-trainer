"""Camera calibration routines."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from config import config as config_manager
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class CameraParameters:
    """Camera intrinsic and distortion parameters."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    resolution: tuple[int, int]
    calibration_error: float
    calibration_date: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "resolution": self.resolution,
            "calibration_error": self.calibration_error,
            "calibration_date": self.calibration_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraParameters":
        """Create from dictionary."""
        return cls(
            camera_matrix=np.array(data["camera_matrix"]),
            distortion_coefficients=np.array(data["distortion_coefficients"]),
            resolution=tuple(data["resolution"]),
            calibration_error=data["calibration_error"],
            calibration_date=data["calibration_date"],
        )


@dataclass
class TableTransform:
    """Table coordinate transformation parameters."""

    homography_matrix: np.ndarray
    table_corners_pixel: list[tuple[float, float]]
    table_corners_world: list[tuple[float, float]]
    table_dimensions: tuple[float, float]  # Real world dimensions in meters
    transformation_error: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "homography_matrix": self.homography_matrix.tolist(),
            "table_corners_pixel": self.table_corners_pixel,
            "table_corners_world": self.table_corners_world,
            "table_dimensions": self.table_dimensions,
            "transformation_error": self.transformation_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableTransform":
        """Create from dictionary."""
        return cls(
            homography_matrix=np.array(data["homography_matrix"]),
            table_corners_pixel=data["table_corners_pixel"],
            table_corners_world=data["table_corners_world"],
            table_dimensions=tuple(data["table_dimensions"]),
            transformation_error=data["transformation_error"],
        )


class CameraCalibrator:
    """Camera intrinsic and extrinsic calibration.

    Implements requirements FR-VIS-039 to FR-VIS-043:
    - Automatic camera calibration
    - Camera intrinsic parameter calculation
    - Camera-to-table transformation
    - Lens distortion compensation
    - Manual calibration adjustment
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize camera calibrator.

        Args:
            cache_dir: Directory to cache calibration results
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.cwd() / "calibration_cache"
        )
        self.cache_dir.mkdir(exist_ok=True)

        # Standard billiards table dimensions (9-foot table)
        self.standard_table_width = config_manager.get(
            "vision.calibration.camera.table_dimensions.standard_width_meters", 2.54
        )
        self.standard_table_height = config_manager.get(
            "vision.calibration.camera.table_dimensions.standard_height_meters", 1.27
        )

        self.camera_params: Optional[CameraParameters] = None
        self.table_transform: Optional[TableTransform] = None

        # Chessboard pattern for intrinsic calibration
        chessboard_pattern = config_manager.get(
            "vision.calibration.camera.chessboard.pattern_size", [9, 6]
        )
        self.chessboard_size = tuple(chessboard_pattern)  # Internal corners
        self.square_size = config_manager.get(
            "vision.calibration.camera.chessboard.square_size_meters", 0.025
        )

    def generate_chessboard_points(self) -> NDArray[np.float64]:
        """Generate 3D points for chessboard pattern."""
        pattern_points = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        pattern_points[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        pattern_points *= self.square_size
        return pattern_points

    def calibrate_intrinsics(
        self, calibration_images: list[np.ndarray]
    ) -> tuple[bool, Optional[CameraParameters]]:
        """Calibrate camera intrinsic parameters using chessboard images.

        Args:
            calibration_images: List of images containing chessboard patterns

        Returns:
            Tuple of (success, camera_parameters)
        """
        min_calibration_images = config_manager.get(
            "vision.calibration.camera.intrinsic_calibration.min_calibration_images", 10
        )
        if len(calibration_images) < min_calibration_images:
            logger.warning(
                f"Need at least {min_calibration_images} calibration images, got {len(calibration_images)}"
            )
            return False, None

        # Prepare object points and image points
        pattern_points = self.generate_chessboard_points()
        object_points = []
        image_points = []

        image_size = None

        for i, image in enumerate(calibration_images):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if image_size is None:
                image_size = gray.shape[::-1]

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                # Refine corner positions
                max_iter = config_manager.get(
                    "vision.calibration.camera.chessboard.corner_refinement.max_iterations",
                    30,
                )
                epsilon = config_manager.get(
                    "vision.calibration.camera.chessboard.corner_refinement.epsilon",
                    0.001,
                )
                window_size = config_manager.get(
                    "vision.calibration.camera.chessboard.corner_refinement.window_size",
                    [11, 11],
                )
                zero_zone = config_manager.get(
                    "vision.calibration.camera.chessboard.corner_refinement.zero_zone",
                    [-1, -1],
                )
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iter,
                    epsilon,
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, tuple(window_size), tuple(zero_zone), criteria
                )

                object_points.append(pattern_points)
                image_points.append(corners_refined)
                logger.info(
                    f"Found chessboard in image {i+1}/{len(calibration_images)}"
                )
            else:
                logger.warning(f"Could not find chessboard in image {i+1}")

        min_valid_detections = config_manager.get(
            "vision.calibration.camera.intrinsic_calibration.min_valid_detections", 5
        )
        if len(object_points) < min_valid_detections:
            logger.error(
                f"Need at least {min_valid_detections} valid chessboard detections, got {len(object_points)}"
            )
            return False, None

        # Perform camera calibration
        logger.info(f"Calibrating camera with {len(object_points)} valid images...")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )

        if not ret:
            logger.error("Camera calibration failed")
            return False, None

        # Calculate calibration error
        total_error = 0
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(
                projected_points
            )
            total_error += error

        mean_error = total_error / len(object_points)

        # Create camera parameters object
        from datetime import datetime

        self.camera_params = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coefficients=dist_coeffs,
            resolution=image_size,
            calibration_error=mean_error,
            calibration_date=datetime.now().isoformat(),
        )

        logger.info(
            f"Camera calibration successful! Mean error: {mean_error:.3f} pixels"
        )

        # Cache the results
        self._save_camera_params()

        return True, self.camera_params

    def calibrate_table_transform(
        self,
        image: NDArray[np.uint8],
        table_corners: list[tuple[float, float]],
        table_dimensions: Optional[tuple[float, float]] = None,
    ) -> Optional[TableTransform]:
        """Calculate camera-to-table transformation matrix.

        Args:
            image: Image containing the table
            table_corners: List of 4 table corners in pixel coordinates (clockwise from top-left)
            table_dimensions: Real world table dimensions (width, height) in meters

        Returns:
            TableTransform object or None if failed
        """
        if len(table_corners) != 4:
            logger.error("Need exactly 4 table corners for transformation")
            return None

        # Use standard table dimensions if not provided
        if table_dimensions is None:
            table_dimensions = (self.standard_table_width, self.standard_table_height)

        # Define world coordinates (table coordinate system)
        # Origin at center of table, x-axis along width, y-axis along height
        w, h = table_dimensions
        table_corners_world = [
            (-w / 2, -h / 2),  # Top-left
            (w / 2, -h / 2),  # Top-right
            (w / 2, h / 2),  # Bottom-right
            (-w / 2, h / 2),  # Bottom-left
        ]

        # Convert to numpy arrays
        src_points = np.array(table_corners, dtype=np.float32)
        dst_points = np.array(table_corners_world, dtype=np.float32)

        # Calculate homography matrix
        ransac_threshold = config_manager.get(
            "vision.calibration.camera.homography.ransac_threshold", 5.0
        )
        homography, mask = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, ransac_threshold
        )

        if homography is None:
            logger.error("Failed to calculate homography matrix")
            return None

        # Calculate transformation error
        projected_points = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2), homography
        )
        projected_points = projected_points.reshape(-1, 2)
        error = np.mean(np.linalg.norm(projected_points - dst_points, axis=1))

        self.table_transform = TableTransform(
            homography_matrix=homography,
            table_corners_pixel=table_corners,
            table_corners_world=table_corners_world,
            table_dimensions=table_dimensions,
            transformation_error=error,
        )

        logger.info(f"Table transformation calculated. Error: {error:.6f} meters")

        # Cache the results
        self._save_table_transform()

        return self.table_transform

    def undistort_image(
        self,
        image: NDArray[np.uint8],
        camera_matrix: Optional[NDArray[np.float64]] = None,
        distortion: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Remove lens distortion from image.

        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix (uses cached if None)
            distortion: Distortion coefficients (uses cached if None)

        Returns:
            Undistorted image
        """
        if camera_matrix is None or distortion is None:
            if self.camera_params is None:
                logger.warning("No camera parameters available for undistortion")
                return image
            camera_matrix = self.camera_params.camera_matrix
            distortion = self.camera_params.distortion_coefficients

        # Get optimal new camera matrix
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion, (w, h), 1, (w, h)
        )

        # Undistort image
        undistorted = cv2.undistort(
            image, camera_matrix, distortion, None, new_camera_matrix
        )

        # Crop to valid region
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y : y + h_roi, x : x + w_roi]

        return undistorted

    def pixel_to_world(
        self, pixel_point: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert pixel coordinates to world coordinates.

        Args:
            pixel_point: Point in pixel coordinates

        Returns:
            Point in world coordinates (meters) or None if no transform available
        """
        if self.table_transform is None:
            logger.warning("No table transformation available")
            return None

        # Convert to homogeneous coordinates
        point = np.array([[pixel_point]], dtype=np.float32)

        # Apply transformation
        world_point = cv2.perspectiveTransform(
            point, self.table_transform.homography_matrix
        )

        return tuple(world_point[0, 0])

    def world_to_pixel(
        self, world_point: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert world coordinates to pixel coordinates.

        Args:
            world_point: Point in world coordinates (meters)

        Returns:
            Point in pixel coordinates or None if no transform available
        """
        if self.table_transform is None:
            logger.warning("No table transformation available")
            return None

        # Use inverse transformation
        inverse_homography = np.linalg.inv(self.table_transform.homography_matrix)

        # Convert to homogeneous coordinates
        point = np.array([[world_point]], dtype=np.float32)

        # Apply inverse transformation
        pixel_point = cv2.perspectiveTransform(point, inverse_homography)

        return tuple(pixel_point[0, 0])

    def manual_adjust_corners(
        self,
        image: NDArray[np.uint8],
        initial_corners: list[tuple[float, float]],
        adjustment_vectors: list[tuple[float, float]],
    ) -> Optional[TableTransform]:
        """Manually adjust table corner positions.

        Args:
            image: Current image
            initial_corners: Initial corner positions
            adjustment_vectors: Adjustment vectors for each corner

        Returns:
            Updated table transformation
        """
        adjusted_corners = []
        for (x, y), (dx, dy) in zip(initial_corners, adjustment_vectors):
            adjusted_corners.append((x + dx, y + dy))

        return self.calibrate_table_transform(image, adjusted_corners)

    def validate_calibration(
        self,
        test_image: NDArray[np.float64],
        known_world_points: list[tuple[float, float]],
        known_pixel_points: list[tuple[float, float]],
    ) -> dict[str, float]:
        """Validate calibration accuracy using known points.

        Args:
            test_image: Test image
            known_world_points: Known points in world coordinates
            known_pixel_points: Corresponding points in pixel coordinates

        Returns:
            Dictionary with validation metrics
        """
        if len(known_world_points) != len(known_pixel_points):
            raise ValueError("World and pixel point lists must have same length")

        # Calculate projection errors
        pixel_errors = []
        world_errors = []

        for world_pt, pixel_pt in zip(known_world_points, known_pixel_points):
            # World to pixel error
            projected_pixel = self.world_to_pixel(world_pt)
            if projected_pixel:
                pixel_error = np.linalg.norm(
                    np.array(projected_pixel) - np.array(pixel_pt)
                )
                pixel_errors.append(pixel_error)

            # Pixel to world error
            projected_world = self.pixel_to_world(pixel_pt)
            if projected_world:
                world_error = np.linalg.norm(
                    np.array(projected_world) - np.array(world_pt)
                )
                world_errors.append(world_error)

        return {
            "mean_pixel_error": np.mean(pixel_errors) if pixel_errors else float("inf"),
            "max_pixel_error": np.max(pixel_errors) if pixel_errors else float("inf"),
            "mean_world_error": np.mean(world_errors) if world_errors else float("inf"),
            "max_world_error": np.max(world_errors) if world_errors else float("inf"),
            "num_test_points": len(known_world_points),
        }

    def _save_camera_params(self) -> None:
        """Save camera parameters to cache."""
        if self.camera_params:
            cache_file = self.cache_dir / "camera_params.json"
            with open(cache_file, "w") as f:
                json.dump(self.camera_params.to_dict(), f, indent=2)

    def _save_table_transform(self) -> None:
        """Save table transformation to cache."""
        if self.table_transform:
            cache_file = self.cache_dir / "table_transform.json"
            with open(cache_file, "w") as f:
                json.dump(self.table_transform.to_dict(), f, indent=2)

    def load_camera_params(self, filepath: Optional[str] = None) -> bool:
        """Load camera parameters from file.

        Args:
            filepath: Path to camera parameters file (uses cache if None)

        Returns:
            True if loaded successfully
        """
        try:
            if filepath is None:
                filepath = self.cache_dir / "camera_params.json"

            with open(filepath) as f:
                data = json.load(f)

            self.camera_params = CameraParameters.from_dict(data)
            logger.info("Camera parameters loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load camera parameters: {e}")
            return False

    def load_table_transform(self, filepath: Optional[str] = None) -> bool:
        """Load table transformation from file.

        Args:
            filepath: Path to table transform file (uses cache if None)

        Returns:
            True if loaded successfully
        """
        try:
            if filepath is None:
                filepath = self.cache_dir / "table_transform.json"

            with open(filepath) as f:
                data = json.load(f)

            self.table_transform = TableTransform.from_dict(data)
            logger.info("Table transformation loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load table transformation: {e}")
            return False

    def export_calibration(self, filepath: str) -> bool:
        """Export complete calibration data.

        Args:
            filepath: Path to export file

        Returns:
            True if exported successfully
        """
        try:
            export_data = {
                "camera_params": (
                    self.camera_params.to_dict() if self.camera_params else None
                ),
                "table_transform": (
                    self.table_transform.to_dict() if self.table_transform else None
                ),
                "calibrator_version": "1.0",
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Calibration exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export calibration: {e}")
            return False

    def import_calibration(self, filepath: str) -> bool:
        """Import complete calibration data.

        Args:
            filepath: Path to import file

        Returns:
            True if imported successfully
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            if data.get("camera_params"):
                self.camera_params = CameraParameters.from_dict(data["camera_params"])

            if data.get("table_transform"):
                self.table_transform = TableTransform.from_dict(data["table_transform"])

            logger.info(f"Calibration imported from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import calibration: {e}")
            return False

    def calibrate_fisheye_from_table(
        self,
        image: NDArray[np.uint8],
        table_corners: list[tuple[float, float]],
        table_dimensions: Optional[tuple[float, float]] = None,
    ) -> tuple[bool, Optional[CameraParameters]]:
        """Automatically calibrate fisheye distortion using table corners.

        This method uses the known rectangular geometry of a billiards table
        to estimate fisheye distortion parameters. Since the table is a perfect
        rectangle, any deviation from rectangular geometry indicates distortion.

        NOTE: Single-frame fisheye calibration with limited points is challenging.
        This method uses a simplified radial distortion model (k1, k2) instead of
        full fisheye model for better stability.

        Args:
            image: Input image containing the table
            table_corners: List of 4 detected table corners in pixel coordinates
            table_dimensions: Real world table dimensions (width, height) in meters

        Returns:
            Tuple of (success, camera_parameters)
        """
        if len(table_corners) != 4:
            logger.error("Need exactly 4 table corners for fisheye calibration")
            return False, None

        # Use standard table dimensions if not provided
        if table_dimensions is None:
            table_dimensions = (self.standard_table_width, self.standard_table_height)

        h, w = image.shape[:2]
        image_size = (w, h)

        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners_array = np.array(table_corners, dtype=np.float32)

        # Calculate center
        center = np.mean(corners_array, axis=0)

        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])

        corners_with_angles = [
            (corner, angle_from_center(corner)) for corner in corners_array
        ]
        corners_with_angles.sort(key=lambda x: x[1])
        sorted_corners = [corner for corner, _ in corners_with_angles]

        # Find top-left (smallest x + y sum)
        sum_coords = [corner[0] + corner[1] for corner in sorted_corners]
        top_left_idx = sum_coords.index(min(sum_coords))
        sorted_corners = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]

        # Generate additional points along table edges
        # This gives calibration more data, especially important when edges are near frame boundaries
        points_per_edge = config_manager.get(
            "vision.calibration.camera.fisheye.points_per_edge", 5
        )  # Sample points along each edge
        detected_points = []
        object_points_3d = []

        table_width, table_height = table_dimensions

        for i in range(4):
            corner1 = sorted_corners[i]
            corner2 = sorted_corners[(i + 1) % 4]

            # Determine which edge this is and corresponding 3D coordinates
            if i == 0:  # Top edge: top-left to top-right
                edge_start_3d = np.array([0, 0, 0])
                edge_end_3d = np.array([table_width, 0, 0])
            elif i == 1:  # Right edge: top-right to bottom-right
                edge_start_3d = np.array([table_width, 0, 0])
                edge_end_3d = np.array([table_width, table_height, 0])
            elif i == 2:  # Bottom edge: bottom-right to bottom-left
                edge_start_3d = np.array([table_width, table_height, 0])
                edge_end_3d = np.array([0, table_height, 0])
            else:  # Left edge: bottom-left to top-left
                edge_start_3d = np.array([0, table_height, 0])
                edge_end_3d = np.array([0, 0, 0])

            # Sample points along this edge
            for j in range(points_per_edge):
                t = j / (points_per_edge - 1)  # Interpolation parameter [0, 1]

                # Interpolate 2D point
                point_2d = corner1 * (1 - t) + corner2 * t
                detected_points.append(point_2d)

                # Interpolate 3D point
                point_3d = edge_start_3d * (1 - t) + edge_end_3d * t
                object_points_3d.append(point_3d)

        detected_points = np.array(detected_points, dtype=np.float32)
        object_points_3d = np.array(object_points_3d, dtype=np.float32)

        logger.info(f"Generated {len(detected_points)} points for fisheye calibration")

        # Use standard calibrateCamera with FIXED focal length and principal point
        # Only optimize distortion coefficients for stability with limited points

        # Fixed camera matrix - DO NOT optimize these
        # Use diagonal FOV of ~70 degrees as typical for webcams
        focal_length_factor = config_manager.get(
            "vision.calibration.camera.fisheye.focal_length_factor", 0.8
        )
        fx = fy = w * focal_length_factor  # Conservative focal length estimate
        cx, cy = w / 2, h / 2  # Principal point at image center

        camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

        # Initialize distortion coefficients - only k1 and k2 will be optimized
        distortion_coeffs = np.zeros((5, 1), dtype=np.float64)

        # Prepare data for standard calibration
        obj_points = [object_points_3d.astype(np.float32)]
        img_points = [detected_points.astype(np.float32)]

        logger.info(f"Using {len(detected_points)} points for calibration")
        logger.info(f"Image size: {image_size}")
        logger.info(
            f"Fixed focal length: {fx:.1f}, principal point: ({cx:.1f}, {cy:.1f})"
        )

        try:
            # FIXED camera intrinsics, only optimize distortion
            # This prevents extreme focal length values from limited calibration data
            calibration_flags = (
                cv2.CALIB_USE_INTRINSIC_GUESS  # Use our camera_matrix as fixed
                | cv2.CALIB_FIX_PRINCIPAL_POINT  # Don't optimize principal point
                | cv2.CALIB_FIX_ASPECT_RATIO  # Keep fx = fy
                | cv2.CALIB_FIX_FOCAL_LENGTH  # Don't optimize focal length
                | cv2.CALIB_ZERO_TANGENT_DIST  # Assume no tangential distortion
                | cv2.CALIB_FIX_K3  # Only use k1, k2 (not k3)
            )

            max_iter = config_manager.get(
                "vision.calibration.camera.fisheye.max_iterations", 100
            )
            epsilon = config_manager.get(
                "vision.calibration.camera.fisheye.convergence_epsilon", 1e-6
            )
            rms_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_size,
                camera_matrix,
                distortion_coeffs,
                flags=calibration_flags,
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    max_iter,
                    epsilon,
                ),
            )

            logger.info(
                f"Camera calibration from table completed with RMS error: {rms_error:.4f}"
            )
            logger.info(f"Camera matrix (fixed):\n{K}")
            logger.info(f"Distortion coefficients (k1,k2,p1,p2,k3): {D.ravel()}")

            # Validate distortion coefficients are reasonable
            warning_threshold = config_manager.get(
                "vision.calibration.camera.fisheye.distortion_limits.warning_threshold",
                0.3,
            )
            clamp_threshold = config_manager.get(
                "vision.calibration.camera.fisheye.distortion_limits.clamp_threshold",
                0.25,
            )
            k1, k2 = float(D[0]), float(D[1])
            if abs(k1) > warning_threshold or abs(k2) > warning_threshold:
                logger.warning(
                    f"High distortion values detected: k1={k1:.3f}, k2={k2:.3f}"
                )
                logger.warning(
                    f"Clamping to conservative range [-{clamp_threshold}, {clamp_threshold}]"
                )
                k1 = np.clip(k1, -clamp_threshold, clamp_threshold)
                k2 = np.clip(k2, -clamp_threshold, clamp_threshold)
                D[0], D[1] = k1, k2
                logger.info(f"Clamped to: k1={k1:.3f}, k2={k2:.3f}")

            # Convert to fisheye-compatible format for backwards compatibility
            fisheye_dist = np.zeros((4, 1), dtype=np.float64)
            fisheye_dist[0] = D[0]  # k1
            fisheye_dist[1] = D[1]  # k2

            logger.info(
                f"Final distortion coefficients: k1={fisheye_dist[0,0]:.4f}, k2={fisheye_dist[1,0]:.4f}"
            )

            # Create camera parameters object
            from datetime import datetime

            self.camera_params = CameraParameters(
                camera_matrix=K,
                distortion_coefficients=fisheye_dist,
                resolution=image_size,
                calibration_error=rms_error,
                calibration_date=datetime.now().isoformat(),
            )

            # Cache the results
            self._save_camera_params()

            return True, self.camera_params

        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            logger.exception("Full traceback:")
            return False, None

    def load_calibration(self, filepath: str) -> bool:
        """Load camera calibration from YAML file.

        Args:
            filepath: Path to YAML calibration file

        Returns:
            True if loaded successfully
        """
        try:
            # Read using OpenCV FileStorage
            fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)

            if not fs.isOpened():
                logger.error(f"Failed to open calibration file: {filepath}")
                return False

            # Read camera matrix
            camera_matrix = fs.getNode("camera_matrix").mat()
            if camera_matrix is None:
                logger.error("No camera_matrix found in calibration file")
                fs.release()
                return False

            # Read distortion coefficients
            dist_coeffs = fs.getNode("dist_coeffs").mat()
            if dist_coeffs is None:
                logger.error("No dist_coeffs found in calibration file")
                fs.release()
                return False

            # Read resolution
            image_width = fs.getNode("image_width").real()
            image_height = fs.getNode("image_height").real()

            if image_width <= 0 or image_height <= 0:
                logger.warning("Invalid resolution in calibration file, using defaults")
                image_width = config_manager.get(
                    "vision.calibration.camera.default_resolution.width", 1920
                )
                image_height = config_manager.get(
                    "vision.calibration.camera.default_resolution.height", 1080
                )

            # Read calibration error (optional)
            calibration_error_node = fs.getNode("calibration_error")
            calibration_error = (
                calibration_error_node.real()
                if not calibration_error_node.empty()
                else 0.0
            )

            # Read calibration date (optional)
            calibration_date_node = fs.getNode("calibration_date")
            calibration_date = (
                calibration_date_node.string()
                if not calibration_date_node.empty()
                else "unknown"
            )

            fs.release()

            # Create camera parameters object
            self.camera_params = CameraParameters(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                resolution=(int(image_width), int(image_height)),
                calibration_error=calibration_error,
                calibration_date=calibration_date,
            )

            # Store additional attributes for backward compatibility
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

            logger.info(f"Calibration loaded from {filepath}")
            logger.info(f"  Resolution: {int(image_width)}x{int(image_height)}")
            logger.info(f"  Calibration error: {calibration_error:.4f}")
            logger.info(f"  Distortion coeffs: {dist_coeffs.ravel()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load calibration from YAML: {e}")
            logger.exception("Full traceback:")
            return False

    def save_fisheye_calibration_yaml(
        self, filepath: str, camera_params: Optional[CameraParameters] = None
    ) -> bool:
        """Save fisheye calibration to OpenCV YAML format.

        Args:
            filepath: Path to save YAML file
            camera_params: Camera parameters to save (uses cached if None)

        Returns:
            True if saved successfully
        """
        if camera_params is None:
            camera_params = self.camera_params

        if camera_params is None:
            logger.error("No camera parameters to save")
            return False

        try:
            # Create directory if needed
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Write using OpenCV FileStorage for compatibility
            fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", camera_params.camera_matrix)
            fs.write("dist_coeffs", camera_params.distortion_coefficients)
            fs.write("image_width", camera_params.resolution[0])
            fs.write("image_height", camera_params.resolution[1])
            fs.write("calibration_error", float(camera_params.calibration_error))
            fs.write("calibration_date", camera_params.calibration_date)
            fs.write("calibration_method", "table_rectangle")
            fs.write(
                "notes",
                "Fisheye calibration computed from billiards table rectangular geometry",
            )
            fs.release()

            logger.info(f"Fisheye calibration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save fisheye calibration: {e}")
            return False
