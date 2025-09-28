"""Geometric calibration for table-to-projector coordinate mapping.

This module handles the mapping between physical table coordinates and projector
display coordinates. It provides tools for establishing coordinate transforms,
handling table dimensions, and managing spatial accuracy.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TableDimensions:
    """Physical table dimensions and characteristics."""

    length: float  # Table length in standard units (e.g., meters)
    width: float  # Table width in standard units
    pocket_radius: float = 0.0  # Pocket radius if applicable
    rail_width: float = 0.0  # Rail width

    # Corner pocket positions (relative to table corners)
    corner_pockets: list[tuple[float, float]] = None
    side_pockets: list[tuple[float, float]] = None

    def __post_init__(self):
        """Initialize default pocket positions for standard pool table."""
        if self.corner_pockets is None:
            self.corner_pockets = [
                (0.0, 0.0),  # Top-left
                (self.length, 0.0),  # Top-right
                (self.length, self.width),  # Bottom-right
                (0.0, self.width),  # Bottom-left
            ]

        if self.side_pockets is None:
            self.side_pockets = [
                (self.length / 2, 0.0),  # Top center
                (self.length / 2, self.width),  # Bottom center
            ]


@dataclass
class CalibrationTarget:
    """A calibration target point with both table and display coordinates."""

    table_x: float
    table_y: float
    display_x: float
    display_y: float
    accuracy: float = 1.0  # Confidence/accuracy score
    label: str = ""  # Optional label for the point


class GeometricCalibrator:
    """Handles geometric mapping between table and projector coordinates.

    This class provides methods to:
    - Establish coordinate transforms between table and display space
    - Calibrate using reference points and measurements
    - Handle scale, rotation, and translation transforms
    - Validate calibration accuracy with test patterns
    """

    def __init__(
        self, table_dimensions: TableDimensions, display_width: int, display_height: int
    ):
        """Initialize geometric calibrator.

        Args:
            table_dimensions: Physical table dimensions
            display_width: Display width in pixels
            display_height: Display height in pixels
        """
        self.table_dimensions = table_dimensions
        self.display_width = display_width
        self.display_height = display_height

        # Calibration state
        self.calibration_targets: list[CalibrationTarget] = []
        self.transform_matrix: Optional[np.ndarray] = None
        self.inverse_transform: Optional[np.ndarray] = None

        # Calibration metrics
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.rotation_angle: float = 0.0
        self.translation_x: float = 0.0
        self.translation_y: float = 0.0
        self.calibration_error: float = 0.0

        logger.info(
            f"GeometricCalibrator initialized for {table_dimensions.length}x{table_dimensions.width} table"
        )

    def add_calibration_target(
        self,
        table_x: float,
        table_y: float,
        display_x: float,
        display_y: float,
        label: str = "",
    ) -> None:
        """Add a calibration target point.

        Args:
            table_x: Table X coordinate (physical units)
            table_y: Table Y coordinate (physical units)
            display_x: Display X coordinate (pixels)
            display_y: Display Y coordinate (pixels)
            label: Optional label for the point
        """
        target = CalibrationTarget(
            table_x=table_x,
            table_y=table_y,
            display_x=display_x,
            display_y=display_y,
            label=label,
        )

        self.calibration_targets.append(target)
        logger.debug(
            f"Added calibration target: {label} table=({table_x:.3f}, {table_y:.3f}) display=({display_x:.1f}, {display_y:.1f})"
        )

    def remove_calibration_target(self, index: int) -> bool:
        """Remove a calibration target by index.

        Args:
            index: Index of target to remove

        Returns:
            True if removed successfully
        """
        if 0 <= index < len(self.calibration_targets):
            removed = self.calibration_targets.pop(index)
            logger.debug(f"Removed calibration target: {removed.label}")
            return True
        return False

    def clear_calibration_targets(self) -> None:
        """Clear all calibration targets."""
        self.calibration_targets.clear()
        self.transform_matrix = None
        self.inverse_transform = None
        logger.debug("All calibration targets cleared")

    def add_table_corner_targets(
        self, corner_displays: list[tuple[float, float]]
    ) -> None:
        """Add calibration targets for table corners.

        Args:
            corner_displays: Display coordinates for each table corner
                           in order: [top_left, top_right, bottom_right, bottom_left]
        """
        if len(corner_displays) != 4:
            raise ValueError("Exactly 4 corner display coordinates required")

        # Table corner coordinates (physical)
        table_corners = [
            (0.0, 0.0),  # Top-left
            (self.table_dimensions.length, 0.0),  # Top-right
            (self.table_dimensions.length, self.table_dimensions.width),  # Bottom-right
            (0.0, self.table_dimensions.width),  # Bottom-left
        ]

        corner_labels = ["top_left", "top_right", "bottom_right", "bottom_left"]

        for i, ((table_x, table_y), (display_x, display_y)) in enumerate(
            zip(table_corners, corner_displays)
        ):
            self.add_calibration_target(
                table_x,
                table_y,
                display_x,
                display_y,
                label=f"corner_{corner_labels[i]}",
            )

        logger.info("Table corner calibration targets added")

    def calculate_transform(self) -> bool:
        """Calculate the geometric transform from calibration targets.

        Returns:
            True if transform calculated successfully
        """
        if len(self.calibration_targets) < 3:
            logger.error("At least 3 calibration targets required")
            return False

        try:
            # Extract source and destination points
            table_points = np.array(
                [
                    [target.table_x, target.table_y]
                    for target in self.calibration_targets
                ],
                dtype=np.float32,
            )

            display_points = np.array(
                [
                    [target.display_x, target.display_y]
                    for target in self.calibration_targets
                ],
                dtype=np.float32,
            )

            # Calculate transform based on number of points
            if len(self.calibration_targets) == 3:
                # Affine transform for 3 points
                self.transform_matrix = cv2.getAffineTransform(
                    table_points[:3], display_points[:3]
                )
                # Convert to 3x3 homogeneous matrix
                self.transform_matrix = np.vstack([self.transform_matrix, [0, 0, 1]])

            elif len(self.calibration_targets) == 4:
                # Perspective transform for 4 points
                self.transform_matrix = cv2.getPerspectiveTransform(
                    table_points[:4], display_points[:4]
                )

            else:
                # Use least squares for more than 4 points
                self.transform_matrix = self._calculate_least_squares_transform(
                    table_points, display_points
                )

            # Calculate inverse transform
            self.inverse_transform = np.linalg.inv(self.transform_matrix)

            # Extract transformation components
            self._extract_transform_components()

            # Calculate calibration error
            self._calculate_calibration_error()

            logger.info(
                f"Geometric transform calculated with {len(self.calibration_targets)} targets"
            )
            logger.info(f"Calibration error: {self.calibration_error:.3f} pixels RMS")
            return True

        except Exception as e:
            logger.error(f"Failed to calculate geometric transform: {e}")
            return False

    def _calculate_least_squares_transform(
        self, table_points: np.ndarray, display_points: np.ndarray
    ) -> np.ndarray:
        """Calculate transform using least squares for overdetermined system."""
        # Set up homogeneous coordinate system
        n_points = len(table_points)

        # Create coefficient matrix for homogeneous transformation
        A = np.zeros((2 * n_points, 8))
        b = np.zeros(2 * n_points)

        for i in range(n_points):
            x, y = table_points[i]
            u, v = display_points[i]

            # Set up equations for perspective transform
            A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
            A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
            b[2 * i] = u
            b[2 * i + 1] = v

        # Solve least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Reconstruct homography matrix
        H = np.array(
            [
                [coeffs[0], coeffs[1], coeffs[2]],
                [coeffs[3], coeffs[4], coeffs[5]],
                [coeffs[6], coeffs[7], 1.0],
            ]
        )

        return H

    def _extract_transform_components(self) -> None:
        """Extract scale, rotation, and translation from transform matrix."""
        if self.transform_matrix is None:
            return

        try:
            # Extract components from 2x3 or 3x3 matrix
            if self.transform_matrix.shape == (3, 3):
                # Use the 2x2 linear part for decomposition
                linear_part = self.transform_matrix[:2, :2]
                translation = self.transform_matrix[:2, 2]
            else:
                linear_part = self.transform_matrix
                translation = self.transform_matrix[:, 2]

            # Extract scale and rotation using SVD
            U, s, Vt = np.linalg.svd(linear_part)

            # Scale factors
            self.scale_x = s[0]
            self.scale_y = s[1]

            # Rotation angle (from the U matrix)
            self.rotation_angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

            # Translation
            self.translation_x = translation[0]
            self.translation_y = translation[1]

            logger.debug(
                f"Transform components - Scale: ({self.scale_x:.3f}, {self.scale_y:.3f}), "
                f"Rotation: {self.rotation_angle:.2f}°, "
                f"Translation: ({self.translation_x:.1f}, {self.translation_y:.1f})"
            )

        except Exception as e:
            logger.warning(f"Failed to extract transform components: {e}")

    def _calculate_calibration_error(self) -> None:
        """Calculate RMS error of calibration targets."""
        if not self.calibration_targets or self.transform_matrix is None:
            self.calibration_error = float("inf")
            return

        total_error = 0.0
        n_points = len(self.calibration_targets)

        for target in self.calibration_targets:
            # Transform table point to display coordinates
            predicted_x, predicted_y = self.table_to_display(
                target.table_x, target.table_y
            )

            # Calculate error
            error_x = predicted_x - target.display_x
            error_y = predicted_y - target.display_y
            error = np.sqrt(error_x**2 + error_y**2)
            total_error += error**2

        self.calibration_error = np.sqrt(total_error / n_points)

    def table_to_display(self, table_x: float, table_y: float) -> tuple[float, float]:
        """Transform table coordinates to display coordinates.

        Args:
            table_x: Table X coordinate
            table_y: Table Y coordinate

        Returns:
            Display coordinates (x, y)
        """
        if self.transform_matrix is None:
            logger.warning("No transform matrix available")
            return table_x, table_y

        try:
            # Create homogeneous point
            point = np.array([table_x, table_y, 1.0])

            # Apply transformation
            transformed = self.transform_matrix @ point

            # Normalize if perspective transform
            if abs(transformed[2]) > 1e-6:
                display_x = transformed[0] / transformed[2]
                display_y = transformed[1] / transformed[2]
            else:
                display_x = transformed[0]
                display_y = transformed[1]

            return float(display_x), float(display_y)

        except Exception as e:
            logger.warning(f"Table to display transform failed: {e}")
            return table_x, table_y

    def display_to_table(
        self, display_x: float, display_y: float
    ) -> tuple[float, float]:
        """Transform display coordinates to table coordinates.

        Args:
            display_x: Display X coordinate
            display_y: Display Y coordinate

        Returns:
            Table coordinates (x, y)
        """
        if self.inverse_transform is None:
            logger.warning("No inverse transform matrix available")
            return display_x, display_y

        try:
            # Create homogeneous point
            point = np.array([display_x, display_y, 1.0])

            # Apply inverse transformation
            transformed = self.inverse_transform @ point

            # Normalize if perspective transform
            if abs(transformed[2]) > 1e-6:
                table_x = transformed[0] / transformed[2]
                table_y = transformed[1] / transformed[2]
            else:
                table_x = transformed[0]
                table_y = transformed[1]

            return float(table_x), float(table_y)

        except Exception as e:
            logger.warning(f"Display to table transform failed: {e}")
            return display_x, display_y

    def transform_table_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple table points to display coordinates.

        Args:
            points: Array of table points, shape (N, 2)

        Returns:
            Array of display points, shape (N, 2)
        """
        if self.transform_matrix is None:
            return points

        try:
            # Add homogeneous coordinate
            homogeneous_points = np.column_stack([points, np.ones(len(points))])

            # Apply transformation
            transformed = homogeneous_points @ self.transform_matrix.T

            # Normalize and extract 2D coordinates
            transformed[:, 0] /= transformed[:, 2]
            transformed[:, 1] /= transformed[:, 2]

            return transformed[:, :2].astype(np.float32)

        except Exception as e:
            logger.warning(f"Bulk transform failed: {e}")
            return points

    def generate_table_grid(
        self, grid_spacing: float = 0.1
    ) -> list[list[tuple[float, float]]]:
        """Generate a grid pattern in table coordinates.

        Args:
            grid_spacing: Spacing between grid lines in table units

        Returns:
            Grid lines as lists of table coordinate points
        """
        lines = []

        # Vertical lines
        x = 0.0
        while x <= self.table_dimensions.length:
            line = []
            y = 0.0
            while y <= self.table_dimensions.width:
                line.append((x, y))
                y += grid_spacing / 10  # Finer resolution for smooth lines
            lines.append(line)
            x += grid_spacing

        # Horizontal lines
        y = 0.0
        while y <= self.table_dimensions.width:
            line = []
            x = 0.0
            while x <= self.table_dimensions.length:
                line.append((x, y))
                x += grid_spacing / 10  # Finer resolution for smooth lines
            lines.append(line)
            y += grid_spacing

        return lines

    def generate_test_pattern(self) -> dict[str, list[tuple[float, float]]]:
        """Generate test patterns for validation.

        Returns:
            Dictionary of test patterns with table coordinates
        """
        patterns = {}

        # Table outline
        patterns["table_outline"] = [
            (0.0, 0.0),
            (self.table_dimensions.length, 0.0),
            (self.table_dimensions.length, self.table_dimensions.width),
            (0.0, self.table_dimensions.width),
            (0.0, 0.0),  # Close the rectangle
        ]

        # Center lines
        center_x = self.table_dimensions.length / 2
        center_y = self.table_dimensions.width / 2

        patterns["center_cross"] = [
            # Horizontal line
            [(0.0, center_y), (self.table_dimensions.length, center_y)],
            # Vertical line
            [(center_x, 0.0), (center_x, self.table_dimensions.width)],
        ]

        # Corner markers
        marker_size = (
            min(self.table_dimensions.length, self.table_dimensions.width) * 0.05
        )
        corners = [
            (0.0, 0.0),
            (self.table_dimensions.length, 0.0),
            (self.table_dimensions.length, self.table_dimensions.width),
            (0.0, self.table_dimensions.width),
        ]

        patterns["corner_markers"] = []
        for corner_x, corner_y in corners:
            # Small cross at each corner
            patterns["corner_markers"].extend(
                [
                    [
                        (corner_x - marker_size, corner_y),
                        (corner_x + marker_size, corner_y),
                    ],
                    [
                        (corner_x, corner_y - marker_size),
                        (corner_x, corner_y + marker_size),
                    ],
                ]
            )

        # Pocket positions if available
        if self.table_dimensions.corner_pockets:
            patterns["corner_pockets"] = []
            for pocket_x, pocket_y in self.table_dimensions.corner_pockets:
                # Circle approximation
                angles = np.linspace(0, 2 * np.pi, 16)
                radius = self.table_dimensions.pocket_radius or marker_size
                circle = [
                    (
                        pocket_x + radius * np.cos(angle),
                        pocket_y + radius * np.sin(angle),
                    )
                    for angle in angles
                ]
                circle.append(circle[0])  # Close the circle
                patterns["corner_pockets"].append(circle)

        return patterns

    def validate_calibration(self, tolerance: float = 5.0) -> tuple[bool, list[str]]:
        """Validate the current calibration.

        Args:
            tolerance: Maximum acceptable error in pixels

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if we have enough calibration targets
        if len(self.calibration_targets) < 3:
            errors.append("Insufficient calibration targets (minimum 3 required)")

        # Check if transform matrix exists
        if self.transform_matrix is None:
            errors.append("Transform matrix not calculated")

        # Check calibration error
        if self.calibration_error > tolerance:
            errors.append(
                f"Calibration error too high: {self.calibration_error:.1f} pixels (tolerance: {tolerance})"
            )

        # Check for reasonable scale factors
        if self.scale_x <= 0 or self.scale_y <= 0:
            errors.append("Invalid scale factors")

        if self.scale_x > 1000 or self.scale_y > 1000:
            errors.append("Scale factors unreasonably large")

        # Check if all calibration targets are within reasonable bounds
        for i, target in enumerate(self.calibration_targets):
            if (
                target.display_x < -100
                or target.display_x > self.display_width + 100
                or target.display_y < -100
                or target.display_y > self.display_height + 100
            ):
                errors.append(
                    f"Calibration target {i} display coordinates out of bounds"
                )

        return len(errors) == 0, errors

    def get_calibration_data(self) -> dict[str, Any]:
        """Get current calibration data for persistence.

        Returns:
            Dictionary containing calibration data
        """
        data = {
            "table_dimensions": {
                "length": self.table_dimensions.length,
                "width": self.table_dimensions.width,
                "pocket_radius": self.table_dimensions.pocket_radius,
                "rail_width": self.table_dimensions.rail_width,
            },
            "display_resolution": [self.display_width, self.display_height],
            "calibration_targets": [
                {
                    "table_x": target.table_x,
                    "table_y": target.table_y,
                    "display_x": target.display_x,
                    "display_y": target.display_y,
                    "label": target.label,
                }
                for target in self.calibration_targets
            ],
            "transform_components": {
                "scale_x": self.scale_x,
                "scale_y": self.scale_y,
                "rotation_angle": self.rotation_angle,
                "translation_x": self.translation_x,
                "translation_y": self.translation_y,
            },
            "calibration_error": self.calibration_error,
        }

        if self.transform_matrix is not None:
            data["transform_matrix"] = self.transform_matrix.tolist()

        return data

    def load_calibration_data(self, data: dict[str, Any]) -> bool:
        """Load calibration data from persistence.

        Args:
            data: Dictionary containing calibration data

        Returns:
            True if loaded successfully
        """
        try:
            # Load table dimensions
            if "table_dimensions" in data:
                dims = data["table_dimensions"]
                self.table_dimensions = TableDimensions(
                    length=dims["length"],
                    width=dims["width"],
                    pocket_radius=dims.get("pocket_radius", 0.0),
                    rail_width=dims.get("rail_width", 0.0),
                )

            # Load calibration targets
            if "calibration_targets" in data:
                self.calibration_targets.clear()
                for target_data in data["calibration_targets"]:
                    target = CalibrationTarget(
                        table_x=target_data["table_x"],
                        table_y=target_data["table_y"],
                        display_x=target_data["display_x"],
                        display_y=target_data["display_y"],
                        label=target_data.get("label", ""),
                    )
                    self.calibration_targets.append(target)

            # Load transform matrix
            if "transform_matrix" in data:
                self.transform_matrix = np.array(
                    data["transform_matrix"], dtype=np.float32
                )
                self.inverse_transform = np.linalg.inv(self.transform_matrix)

            # Load transform components
            if "transform_components" in data:
                comp = data["transform_components"]
                self.scale_x = comp.get("scale_x", 1.0)
                self.scale_y = comp.get("scale_y", 1.0)
                self.rotation_angle = comp.get("rotation_angle", 0.0)
                self.translation_x = comp.get("translation_x", 0.0)
                self.translation_y = comp.get("translation_y", 0.0)

            # Load calibration error
            self.calibration_error = data.get("calibration_error", 0.0)

            logger.info("Geometric calibration data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            return False

    def reset_calibration(self) -> None:
        """Reset calibration to default state."""
        self.calibration_targets.clear()
        self.transform_matrix = None
        self.inverse_transform = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.rotation_angle = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.calibration_error = 0.0

        logger.info("Geometric calibration reset to defaults")
