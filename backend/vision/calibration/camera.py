"""Camera calibration routines."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

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

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize camera calibrator.

        Args:
            cache_dir: Directory to cache calibration results
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.cwd() / "calibration_cache"
        )
        self.cache_dir.mkdir(exist_ok=True)

        # Standard billiards table dimensions (9-foot table)
        self.standard_table_width = 2.54  # meters (100 inches)
        self.standard_table_height = 1.27  # meters (50 inches)

        self.camera_params: Optional[CameraParameters] = None
        self.table_transform: Optional[TableTransform] = None

        # Chessboard pattern for intrinsic calibration
        self.chessboard_size = (9, 6)  # Internal corners
        self.square_size = 0.025  # 25mm squares

    def generate_chessboard_points(self) -> np.ndarray:
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
        if len(calibration_images) < 10:
            logger.warning(
                f"Need at least 10 calibration images, got {len(calibration_images)}"
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
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

                object_points.append(pattern_points)
                image_points.append(corners_refined)
                logger.info(
                    f"Found chessboard in image {i+1}/{len(calibration_images)}"
                )
            else:
                logger.warning(f"Could not find chessboard in image {i+1}")

        if len(object_points) < 5:
            logger.error(
                f"Need at least 5 valid chessboard detections, got {len(object_points)}"
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
        image: np.ndarray,
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
        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

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
        image: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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
        image: np.ndarray,
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
        test_image: np.ndarray,
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

    def _save_camera_params(self):
        """Save camera parameters to cache."""
        if self.camera_params:
            cache_file = self.cache_dir / "camera_params.json"
            with open(cache_file, "w") as f:
                json.dump(self.camera_params.to_dict(), f, indent=2)

    def _save_table_transform(self):
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
