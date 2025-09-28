"""Keystone calibration for perspective correction.

This module handles perspective correction and keystone adjustment for projector
displays. It provides tools for correcting trapezoidal distortion caused by
angled projection and calculates homography transformations.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeystoneParams:
    """Keystone correction parameters."""

    horizontal: float = 0.0  # -1.0 to 1.0
    vertical: float = 0.0  # -1.0 to 1.0
    rotation: float = 0.0  # degrees -180 to 180
    barrel_distortion: float = 0.0  # -1.0 to 1.0


@dataclass
class CornerPoints:
    """Four corner points for perspective transformation."""

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array format for OpenCV."""
        return np.array(
            [
                [self.top_left[0], self.top_left[1]],
                [self.top_right[0], self.top_right[1]],
                [self.bottom_right[0], self.bottom_right[1]],
                [self.bottom_left[0], self.bottom_left[1]],
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, points: np.ndarray) -> "CornerPoints":
        """Create from numpy array."""
        return cls(
            top_left=(float(points[0][0]), float(points[0][1])),
            top_right=(float(points[1][0]), float(points[1][1])),
            bottom_right=(float(points[2][0]), float(points[2][1])),
            bottom_left=(float(points[3][0]), float(points[3][1])),
        )


class KeystoneCalibrator:
    """Handles perspective correction and keystone adjustment for projectors.

    This class provides methods to:
    - Calculate perspective transformation matrices
    - Apply keystone corrections for angled projection
    - Handle barrel/pincushion distortion correction
    - Generate calibration grids and test patterns
    """

    def __init__(self, display_width: int, display_height: int):
        """Initialize keystone calibrator.

        Args:
            display_width: Display width in pixels
            display_height: Display height in pixels
        """
        self.display_width = display_width
        self.display_height = display_height

        # Current calibration state
        self.keystone_params = KeystoneParams()
        self.corner_points: Optional[CornerPoints] = None
        self.homography_matrix: Optional[np.ndarray] = None
        self.distortion_coefficients: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None

        # Default reference points (ideal rectangle)
        self.reference_points = CornerPoints(
            top_left=(0.0, 0.0),
            top_right=(float(display_width), 0.0),
            bottom_right=(float(display_width), float(display_height)),
            bottom_left=(0.0, float(display_height)),
        )

        logger.info(
            f"KeystoneCalibrator initialized for {display_width}x{display_height}"
        )

    def set_corner_points(self, points: CornerPoints) -> None:
        """Set the corner points for perspective correction.

        Args:
            points: Four corner points defining the projection area
        """
        self.corner_points = points
        self._calculate_homography()
        logger.debug("Corner points updated and homography recalculated")

    def adjust_corner_point(
        self, corner_index: int, new_position: tuple[float, float]
    ) -> None:
        """Adjust a single corner point.

        Args:
            corner_index: Corner index (0=TL, 1=TR, 2=BR, 3=BL)
            new_position: New (x, y) position for the corner
        """
        if not self.corner_points:
            # Initialize with default rectangle if not set
            self.corner_points = CornerPoints(
                top_left=(0.0, 0.0),
                top_right=(float(self.display_width), 0.0),
                bottom_right=(float(self.display_width), float(self.display_height)),
                bottom_left=(0.0, float(self.display_height)),
            )

        # Update the specified corner
        if corner_index == 0:  # Top Left
            self.corner_points.top_left = new_position
        elif corner_index == 1:  # Top Right
            self.corner_points.top_right = new_position
        elif corner_index == 2:  # Bottom Right
            self.corner_points.bottom_right = new_position
        elif corner_index == 3:  # Bottom Left
            self.corner_points.bottom_left = new_position
        else:
            raise ValueError(f"Invalid corner index: {corner_index}")

        self._calculate_homography()
        logger.debug(f"Corner {corner_index} adjusted to {new_position}")

    def set_keystone_params(self, params: KeystoneParams) -> None:
        """Set keystone correction parameters.

        Args:
            params: Keystone parameters to apply
        """
        self.keystone_params = params
        self._apply_keystone_correction()
        logger.debug(
            f"Keystone parameters updated: h={params.horizontal:.3f}, v={params.vertical:.3f}"
        )

    def _apply_keystone_correction(self) -> None:
        """Apply keystone correction based on parameters."""
        if not self.corner_points:
            return

        # Get base corner points
        points = self.corner_points.to_array()

        # Apply horizontal keystone
        if abs(self.keystone_params.horizontal) > 0.001:
            h_offset = self.keystone_params.horizontal * self.display_width * 0.2
            points[0][0] += h_offset  # Top left
            points[3][0] += h_offset  # Bottom left
            points[1][0] -= h_offset  # Top right
            points[2][0] -= h_offset  # Bottom right

        # Apply vertical keystone
        if abs(self.keystone_params.vertical) > 0.001:
            v_offset = self.keystone_params.vertical * self.display_height * 0.2
            points[0][1] += v_offset  # Top left
            points[1][1] += v_offset  # Top right
            points[2][1] -= v_offset  # Bottom right
            points[3][1] -= v_offset  # Bottom left

        # Apply rotation
        if abs(self.keystone_params.rotation) > 0.1:
            center_x = self.display_width / 2
            center_y = self.display_height / 2
            angle_rad = np.deg2rad(self.keystone_params.rotation)

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            for i in range(4):
                # Translate to origin
                x = points[i][0] - center_x
                y = points[i][1] - center_y

                # Rotate
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a

                # Translate back
                points[i][0] = new_x + center_x
                points[i][1] = new_y + center_y

        # Update corner points
        self.corner_points = CornerPoints.from_array(points)
        self._calculate_homography()

    def _calculate_homography(self) -> None:
        """Calculate homography matrix from corner points."""
        if not self.corner_points:
            return

        try:
            # Source points (distorted/actual projection)
            src_points = self.corner_points.to_array()

            # Destination points (ideal rectangle)
            dst_points = self.reference_points.to_array()

            # Calculate homography matrix
            self.homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

            logger.debug("Homography matrix calculated successfully")

        except Exception as e:
            logger.error(f"Failed to calculate homography: {e}")
            self.homography_matrix = None

    def setup_distortion_correction(self, barrel_distortion: float = 0.0) -> None:
        """Set up lens distortion correction.

        Args:
            barrel_distortion: Barrel distortion coefficient (-1.0 to 1.0)
        """
        try:
            # Create camera matrix (simplified for projector)
            focal_length = max(self.display_width, self.display_height)
            self.camera_matrix = np.array(
                [
                    [focal_length, 0, self.display_width / 2],
                    [0, focal_length, self.display_height / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            # Set distortion coefficients
            # [k1, k2, p1, p2, k3] format
            self.distortion_coefficients = np.array(
                [
                    barrel_distortion,  # k1 - radial distortion
                    0.0,  # k2 - radial distortion
                    0.0,  # p1 - tangential distortion
                    0.0,  # p2 - tangential distortion
                    0.0,  # k3 - radial distortion
                ],
                dtype=np.float32,
            )

            logger.debug(
                f"Distortion correction set up with barrel={barrel_distortion:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to set up distortion correction: {e}")

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform a point using the current calibration.

        Args:
            x: Input X coordinate
            y: Input Y coordinate

        Returns:
            Transformed (x, y) coordinates
        """
        if not self.homography_matrix is not None:
            return x, y

        try:
            # Create point in homogeneous coordinates
            point = np.array([[x, y]], dtype=np.float32)

            # Apply perspective transformation
            transformed = cv2.perspectiveTransform(
                point.reshape(1, -1, 2), self.homography_matrix
            )

            return float(transformed[0][0][0]), float(transformed[0][0][1])

        except Exception as e:
            logger.warning(f"Point transformation failed: {e}")
            return x, y

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points using current calibration.

        Args:
            points: Array of points to transform, shape (N, 2)

        Returns:
            Transformed points array
        """
        if self.homography_matrix is None:
            return points

        try:
            # Reshape for OpenCV
            points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)

            # Apply transformation
            transformed = cv2.perspectiveTransform(
                points_reshaped, self.homography_matrix
            )

            return transformed.reshape(-1, 2)

        except Exception as e:
            logger.warning(f"Points transformation failed: {e}")
            return points

    def generate_calibration_grid(
        self, grid_size: int = 10
    ) -> list[list[tuple[float, float]]]:
        """Generate calibration grid points.

        Args:
            grid_size: Number of grid lines in each dimension

        Returns:
            Grid lines as list of point lists
        """
        lines = []

        # Vertical lines
        for i in range(grid_size + 1):
            x = (i / grid_size) * self.display_width
            line = []
            for j in range(11):  # More points for smooth lines
                y = (j / 10) * self.display_height
                line.append((x, y))
            lines.append(line)

        # Horizontal lines
        for i in range(grid_size + 1):
            y = (i / grid_size) * self.display_height
            line = []
            for j in range(11):  # More points for smooth lines
                x = (j / 10) * self.display_width
                line.append((x, y))
            lines.append(line)

        return lines

    def generate_crosshairs(
        self, size: float = 50.0
    ) -> list[list[tuple[float, float]]]:
        """Generate crosshair patterns for corner adjustment.

        Args:
            size: Size of crosshairs in pixels

        Returns:
            Crosshair lines for each corner
        """
        if not self.corner_points:
            return []

        crosshairs = []
        corners = [
            self.corner_points.top_left,
            self.corner_points.top_right,
            self.corner_points.bottom_right,
            self.corner_points.bottom_left,
        ]

        for corner in corners:
            cx, cy = corner

            # Horizontal line
            h_line = [(cx - size, cy), (cx + size, cy)]
            crosshairs.append(h_line)

            # Vertical line
            v_line = [(cx, cy - size), (cx, cy + size)]
            crosshairs.append(v_line)

        return crosshairs

    def get_calibration_data(self) -> dict:
        """Get current calibration data for persistence.

        Returns:
            Dictionary containing calibration parameters
        """
        data = {
            "display_resolution": [self.display_width, self.display_height],
            "keystone_params": {
                "horizontal": self.keystone_params.horizontal,
                "vertical": self.keystone_params.vertical,
                "rotation": self.keystone_params.rotation,
                "barrel_distortion": self.keystone_params.barrel_distortion,
            },
        }

        if self.corner_points:
            data["corner_points"] = {
                "top_left": self.corner_points.top_left,
                "top_right": self.corner_points.top_right,
                "bottom_right": self.corner_points.bottom_right,
                "bottom_left": self.corner_points.bottom_left,
            }

        if self.homography_matrix is not None:
            data["homography_matrix"] = self.homography_matrix.tolist()

        return data

    def load_calibration_data(self, data: dict) -> bool:
        """Load calibration data from persistence.

        Args:
            data: Dictionary containing calibration parameters

        Returns:
            True if loaded successfully
        """
        try:
            # Validate display resolution
            if "display_resolution" in data:
                res = data["display_resolution"]
                if res != [self.display_width, self.display_height]:
                    logger.warning(
                        f"Resolution mismatch: saved={res}, current=[{self.display_width}, {self.display_height}]"
                    )

            # Load keystone parameters
            if "keystone_params" in data:
                params = data["keystone_params"]
                self.keystone_params = KeystoneParams(
                    horizontal=params.get("horizontal", 0.0),
                    vertical=params.get("vertical", 0.0),
                    rotation=params.get("rotation", 0.0),
                    barrel_distortion=params.get("barrel_distortion", 0.0),
                )

            # Load corner points
            if "corner_points" in data:
                points = data["corner_points"]
                self.corner_points = CornerPoints(
                    top_left=tuple(points["top_left"]),
                    top_right=tuple(points["top_right"]),
                    bottom_right=tuple(points["bottom_right"]),
                    bottom_left=tuple(points["bottom_left"]),
                )

            # Load homography matrix
            if "homography_matrix" in data:
                self.homography_matrix = np.array(
                    data["homography_matrix"], dtype=np.float32
                )
            else:
                # Recalculate if not saved
                self._calculate_homography()

            # Apply distortion correction if specified
            if self.keystone_params.barrel_distortion != 0.0:
                self.setup_distortion_correction(self.keystone_params.barrel_distortion)

            logger.info("Keystone calibration data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            return False

    def validate_calibration(self) -> tuple[bool, list[str]]:
        """Validate current calibration setup.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if corner points are set
        if not self.corner_points:
            errors.append("Corner points not defined")
        else:
            # Check if corners form a valid quadrilateral
            points = self.corner_points.to_array()

            # Check for degenerate cases
            for i in range(4):
                for j in range(i + 1, 4):
                    p1, p2 = points[i], points[j]
                    if np.allclose(p1, p2, atol=1.0):
                        errors.append(f"Corner points too close: {i} and {j}")

            # Check if points are within display bounds
            for i, point in enumerate(points):
                x, y = point
                if x < -100 or x > self.display_width + 100:
                    errors.append(f"Corner {i} X coordinate out of bounds: {x}")
                if y < -100 or y > self.display_height + 100:
                    errors.append(f"Corner {i} Y coordinate out of bounds: {y}")

        # Check homography matrix
        if self.homography_matrix is None:
            errors.append("Homography matrix not calculated")
        else:
            # Check if matrix is invertible
            try:
                det = cv2.determinant(self.homography_matrix)
                if abs(det) < 1e-6:
                    errors.append("Homography matrix is singular")
            except Exception:
                errors.append("Invalid homography matrix")

        return len(errors) == 0, errors

    def reset_calibration(self) -> None:
        """Reset calibration to default state."""
        self.keystone_params = KeystoneParams()
        self.corner_points = None
        self.homography_matrix = None
        self.distortion_coefficients = None
        self.camera_matrix = None

        logger.info("Keystone calibration reset to defaults")
