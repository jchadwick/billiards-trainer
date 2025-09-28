"""Geometric calibration utilities."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveCorrection:
    """Perspective correction parameters."""

    source_points: list[tuple[float, float]]  # Source quadrilateral corners
    target_points: list[tuple[float, float]]  # Target rectangle corners
    transform_matrix: np.ndarray  # 3x3 perspective transform matrix
    inverse_matrix: np.ndarray  # Inverse transform matrix
    correction_quality: float  # Quality metric (0.0-1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "source_points": self.source_points,
            "target_points": self.target_points,
            "transform_matrix": self.transform_matrix.tolist(),
            "inverse_matrix": self.inverse_matrix.tolist(),
            "correction_quality": self.correction_quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerspectiveCorrection":
        """Create from dictionary."""
        return cls(
            source_points=data["source_points"],
            target_points=data["target_points"],
            transform_matrix=np.array(data["transform_matrix"]),
            inverse_matrix=np.array(data["inverse_matrix"]),
            correction_quality=data["correction_quality"],
        )


@dataclass
class CoordinateMapping:
    """Coordinate system mapping parameters."""

    pixel_bounds: tuple[
        tuple[float, float], tuple[float, float]
    ]  # ((min_x, min_y), (max_x, max_y))
    world_bounds: tuple[
        tuple[float, float], tuple[float, float]
    ]  # World coordinate bounds
    scale_factor: float  # Pixels per world unit
    rotation_angle: float  # Rotation angle in degrees
    translation: tuple[float, float]  # Translation offset

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoordinateMapping":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GeometricCalibration:
    """Complete geometric calibration data."""

    perspective_correction: Optional[PerspectiveCorrection]
    coordinate_mapping: Optional[CoordinateMapping]
    table_dimensions_real: tuple[float, float]  # Real world dimensions (meters)
    table_corners_pixel: list[tuple[float, float]]  # Table corners in pixels
    calibration_error: float  # RMS error in pixels
    calibration_date: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "perspective_correction": (
                self.perspective_correction.to_dict()
                if self.perspective_correction
                else None
            ),
            "coordinate_mapping": (
                self.coordinate_mapping.to_dict() if self.coordinate_mapping else None
            ),
            "table_dimensions_real": self.table_dimensions_real,
            "table_corners_pixel": self.table_corners_pixel,
            "calibration_error": self.calibration_error,
            "calibration_date": self.calibration_date,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeometricCalibration":
        """Create from dictionary."""
        return cls(
            perspective_correction=(
                PerspectiveCorrection.from_dict(data["perspective_correction"])
                if data["perspective_correction"]
                else None
            ),
            coordinate_mapping=(
                CoordinateMapping.from_dict(data["coordinate_mapping"])
                if data["coordinate_mapping"]
                else None
            ),
            table_dimensions_real=tuple(data["table_dimensions_real"]),
            table_corners_pixel=data["table_corners_pixel"],
            calibration_error=data["calibration_error"],
            calibration_date=data["calibration_date"],
        )


class GeometricCalibrator:
    """Geometric calibration and perspective correction.

    Provides comprehensive geometric calibration including:
    - Perspective correction for keystone distortion
    - Coordinate mapping between pixel and world coordinates
    - Table boundary detection and rectification
    - Geometric validation and accuracy testing
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize geometric calibrator.

        Args:
            cache_dir: Directory to cache calibration results
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "geometry_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Standard pool table dimensions (9-foot table)
        self.standard_table_width = 2.54  # meters
        self.standard_table_height = 1.27  # meters

        self.current_calibration: Optional[GeometricCalibration] = None

    def detect_table_corners(
        self,
        frame: np.ndarray,
        manual_corners: Optional[list[tuple[float, float]]] = None,
    ) -> list[tuple[float, float]]:
        """Detect or use manually specified table corners.

        Args:
            frame: Input frame
            manual_corners: Optional manually specified corners

        Returns:
            List of 4 corner points in clockwise order from top-left
        """
        if manual_corners is not None and len(manual_corners) == 4:
            logger.info("Using manually specified table corners")
            return manual_corners

        # Automatic corner detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive threshold to find edges
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find largest quadrilateral contour
        largest_area = 0
        best_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's a quadrilateral
            if len(approx) == 4 and area > largest_area:
                largest_area = area
                best_contour = approx

        if best_contour is None:
            logger.warning(
                "Could not detect table corners automatically, using default"
            )
            h, w = frame.shape[:2]
            margin = 50
            return [
                (margin, margin),
                (w - margin, margin),
                (w - margin, h - margin),
                (margin, h - margin),
            ]

        # Extract corner points and sort clockwise from top-left
        corners = [tuple(point[0]) for point in best_contour]
        corners = self._sort_corners_clockwise(corners)

        logger.info(f"Auto-detected table corners: {corners}")
        return corners

    def _sort_corners_clockwise(
        self, corners: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Sort corners in clockwise order starting from top-left."""
        # Convert to numpy array for easier manipulation
        corners_array = np.array(corners)

        # Find center point
        center = np.mean(corners_array, axis=0)

        # Calculate angles from center
        angles = []
        for corner in corners_array:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)

        # Sort by angle (clockwise from top-left)
        sorted_indices = np.argsort(angles)

        # Adjust to start from top-left (smallest y coordinate)
        y_coords = corners_array[:, 1]
        top_indices = np.where(y_coords < center[1])[0]

        if len(top_indices) >= 2:
            # Among top corners, choose the leftmost as starting point
            top_corners = corners_array[top_indices]
            leftmost_top = top_indices[np.argmin(top_corners[:, 0])]

            # Rotate the sorted list to start from top-left
            start_idx = np.where(sorted_indices == leftmost_top)[0][0]
            sorted_indices = np.roll(sorted_indices, -start_idx)

        return [tuple(corners_array[i]) for i in sorted_indices]

    def calculate_perspective_transform(
        self,
        source_points: list[tuple[float, float]],
        target_points: Optional[list[tuple[float, float]]] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> PerspectiveCorrection:
        """Calculate perspective transformation matrix.

        Args:
            source_points: Source quadrilateral corners (table corners in image)
            target_points: Target rectangle corners (optional, auto-generated if None)
            target_size: Target image size (width, height)

        Returns:
            PerspectiveCorrection object
        """
        if len(source_points) != 4:
            raise ValueError("Need exactly 4 source points")

        # Auto-generate target points if not provided
        if target_points is None:
            if target_size is None:
                # Use standard aspect ratio
                width = 800
                height = int(
                    width * self.standard_table_height / self.standard_table_width
                )
            else:
                width, height = target_size

            target_points = [
                (0, 0),  # Top-left
                (width, 0),  # Top-right
                (width, height),  # Bottom-right
                (0, height),  # Bottom-left
            ]

        # Convert to numpy arrays
        src_points = np.array(source_points, dtype=np.float32)
        dst_points = np.array(target_points, dtype=np.float32)

        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # Calculate correction quality based on how much distortion is corrected
        quality = self._calculate_correction_quality(
            src_points, dst_points, transform_matrix
        )

        correction = PerspectiveCorrection(
            source_points=source_points,
            target_points=target_points,
            transform_matrix=transform_matrix,
            inverse_matrix=inverse_matrix,
            correction_quality=quality,
        )

        logger.info(f"Calculated perspective transform with quality: {quality:.3f}")
        return correction

    def _calculate_correction_quality(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> float:
        """Calculate the quality of perspective correction."""
        # Transform source points and compare with target
        transformed = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2), transform_matrix
        )
        transformed = transformed.reshape(-1, 2)

        # Calculate RMS error
        errors = np.linalg.norm(transformed - dst_points, axis=1)
        rms_error = np.sqrt(np.mean(errors**2))

        # Convert to quality score (0.0-1.0, where 1.0 is perfect)
        max_expected_error = 50.0  # pixels
        quality = max(0.0, 1.0 - (rms_error / max_expected_error))

        return quality

    def correct_keystone_distortion(
        self, frame: np.ndarray, correction: PerspectiveCorrection
    ) -> np.ndarray:
        """Apply keystone correction to frame.

        Args:
            frame: Input frame
            correction: Perspective correction parameters

        Returns:
            Corrected frame
        """
        if correction is None:
            logger.warning("No perspective correction available")
            return frame

        # Get target size from correction parameters
        target_points = np.array(correction.target_points)
        width = int(np.max(target_points[:, 0]))
        height = int(np.max(target_points[:, 1]))

        # Apply perspective transformation
        corrected = cv2.warpPerspective(
            frame, correction.transform_matrix, (width, height)
        )

        return corrected

    def create_coordinate_mapping(
        self,
        pixel_corners: list[tuple[float, float]],
        world_dimensions: tuple[float, float],
    ) -> CoordinateMapping:
        """Create coordinate mapping between pixel and world coordinates.

        Args:
            pixel_corners: Table corners in pixel coordinates
            world_dimensions: Real world table dimensions (width, height) in meters

        Returns:
            CoordinateMapping object
        """
        # Calculate pixel bounds
        pixels_array = np.array(pixel_corners)
        pixel_min = np.min(pixels_array, axis=0)
        pixel_max = np.max(pixels_array, axis=0)
        pixel_bounds = (tuple(pixel_min), tuple(pixel_max))

        # World coordinate system: origin at table center
        world_width, world_height = world_dimensions
        world_bounds = (
            (-world_width / 2, -world_height / 2),
            (world_width / 2, world_height / 2),
        )

        # Calculate scale factor (pixels per meter)
        pixel_width = pixel_max[0] - pixel_min[0]
        pixel_height = pixel_max[1] - pixel_min[1]
        scale_x = pixel_width / world_width
        scale_y = pixel_height / world_height
        scale_factor = (scale_x + scale_y) / 2  # Average scale

        # Calculate rotation (assume minimal rotation for now)
        rotation_angle = 0.0

        # Calculate translation (pixel center to world origin)
        pixel_center = (
            (pixel_min[0] + pixel_max[0]) / 2,
            (pixel_min[1] + pixel_max[1]) / 2,
        )
        translation = pixel_center

        mapping = CoordinateMapping(
            pixel_bounds=pixel_bounds,
            world_bounds=world_bounds,
            scale_factor=scale_factor,
            rotation_angle=rotation_angle,
            translation=translation,
        )

        logger.info(
            f"Created coordinate mapping with scale: {scale_factor:.2f} pixels/meter"
        )
        return mapping

    def pixel_to_world_coordinates(
        self,
        pixel_pos: tuple[float, float],
        mapping: Optional[CoordinateMapping] = None,
    ) -> tuple[float, float]:
        """Convert pixel coordinates to world coordinates.

        Args:
            pixel_pos: Position in pixel coordinates
            mapping: Coordinate mapping (uses current if None)

        Returns:
            Position in world coordinates (meters)
        """
        if mapping is None:
            if (
                self.current_calibration is None
                or self.current_calibration.coordinate_mapping is None
            ):
                logger.warning("No coordinate mapping available")
                return pixel_pos
            mapping = self.current_calibration.coordinate_mapping

        px, py = pixel_pos
        tx, ty = mapping.translation

        # Translate to origin and scale
        world_x = (px - tx) / mapping.scale_factor
        world_y = (py - ty) / mapping.scale_factor

        # Apply rotation if needed
        if abs(mapping.rotation_angle) > 0.01:
            angle = np.radians(mapping.rotation_angle)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            world_x_rot = world_x * cos_a - world_y * sin_a
            world_y_rot = world_x * sin_a + world_y * cos_a
            world_x, world_y = world_x_rot, world_y_rot

        return (world_x, world_y)

    def world_to_pixel_coordinates(
        self,
        world_pos: tuple[float, float],
        mapping: Optional[CoordinateMapping] = None,
    ) -> tuple[float, float]:
        """Convert world coordinates to pixel coordinates.

        Args:
            world_pos: Position in world coordinates (meters)
            mapping: Coordinate mapping (uses current if None)

        Returns:
            Position in pixel coordinates
        """
        if mapping is None:
            if (
                self.current_calibration is None
                or self.current_calibration.coordinate_mapping is None
            ):
                logger.warning("No coordinate mapping available")
                return world_pos
            mapping = self.current_calibration.coordinate_mapping

        wx, wy = world_pos

        # Apply inverse rotation if needed
        if abs(mapping.rotation_angle) > 0.01:
            angle = np.radians(-mapping.rotation_angle)  # Inverse rotation
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            wx_rot = wx * cos_a - wy * sin_a
            wy_rot = wx * sin_a + wy * cos_a
            wx, wy = wx_rot, wy_rot

        # Scale and translate
        tx, ty = mapping.translation
        pixel_x = wx * mapping.scale_factor + tx
        pixel_y = wy * mapping.scale_factor + ty

        return (pixel_x, pixel_y)

    def calibrate_table_geometry(
        self,
        frame: np.ndarray,
        table_corners: Optional[list[tuple[float, float]]] = None,
        table_dimensions: Optional[tuple[float, float]] = None,
    ) -> GeometricCalibration:
        """Perform complete geometric calibration.

        Args:
            frame: Input frame containing table
            table_corners: Optional manual table corners
            table_dimensions: Real world table dimensions (meters)

        Returns:
            Complete geometric calibration
        """
        if table_dimensions is None:
            table_dimensions = (self.standard_table_width, self.standard_table_height)

        # Detect or use provided table corners
        corners = self.detect_table_corners(frame, table_corners)

        # Calculate perspective correction
        perspective_correction = self.calculate_perspective_transform(corners)

        # Create coordinate mapping
        coordinate_mapping = self.create_coordinate_mapping(corners, table_dimensions)

        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(
            corners, perspective_correction
        )

        # Create calibration object
        from datetime import datetime

        calibration = GeometricCalibration(
            perspective_correction=perspective_correction,
            coordinate_mapping=coordinate_mapping,
            table_dimensions_real=table_dimensions,
            table_corners_pixel=corners,
            calibration_error=calibration_error,
            calibration_date=datetime.now().isoformat(),
        )

        self.current_calibration = calibration
        self._save_calibration()

        logger.info(
            f"Geometric calibration completed with error: {calibration_error:.2f} pixels"
        )
        return calibration

    def _calculate_calibration_error(
        self, corners: list[tuple[float, float]], correction: PerspectiveCorrection
    ) -> float:
        """Calculate overall calibration error."""
        # Transform corners and check if they form a proper rectangle
        corners_array = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(
            corners_array, correction.transform_matrix
        )
        transformed = transformed.reshape(-1, 2)

        # Check if transformed corners form a rectangle
        target_array = np.array(correction.target_points)
        errors = np.linalg.norm(transformed - target_array, axis=1)

        return float(np.sqrt(np.mean(errors**2)))

    def validate_geometry(
        self,
        test_points: list[tuple[float, float]],
        expected_world_points: list[tuple[float, float]],
    ) -> dict[str, float]:
        """Validate geometric calibration accuracy.

        Args:
            test_points: Test points in pixel coordinates
            expected_world_points: Expected world coordinates for test points

        Returns:
            Validation metrics
        """
        if len(test_points) != len(expected_world_points):
            raise ValueError("Test points and expected points must have same length")

        if self.current_calibration is None:
            raise ValueError("No calibration available for validation")

        # Convert test points to world coordinates
        converted_points = []
        for pixel_point in test_points:
            world_point = self.pixel_to_world_coordinates(pixel_point)
            converted_points.append(world_point)

        # Calculate errors
        errors = []
        for converted, expected in zip(converted_points, expected_world_points):
            error = np.linalg.norm(np.array(converted) - np.array(expected))
            errors.append(error)

        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
            "num_test_points": len(test_points),
            "errors": errors,
        }

    def correct_barrel_distortion(
        self, frame: np.ndarray, k1: float = 0.0, k2: float = 0.0
    ) -> np.ndarray:
        """Correct barrel/pincushion distortion.

        Args:
            frame: Input frame
            k1, k2: Radial distortion coefficients

        Returns:
            Distortion-corrected frame
        """
        if abs(k1) < 1e-6 and abs(k2) < 1e-6:
            return frame  # No correction needed

        h, w = frame.shape[:2]

        # Create camera matrix (assume center at image center)
        camera_matrix = np.array(
            [[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32
        )

        # Distortion coefficients
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

        # Undistort image
        corrected = cv2.undistort(frame, camera_matrix, dist_coeffs)

        return corrected

    def create_rectification_grid(
        self, frame_size: tuple[int, int], grid_size: int = 20
    ) -> np.ndarray:
        """Create rectification grid for visual calibration assessment.

        Args:
            frame_size: Frame dimensions (width, height)
            grid_size: Grid spacing in pixels

        Returns:
            Grid image for overlay
        """
        width, height = frame_size
        grid = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(grid, (x, 0), (x, height), (0, 255, 0), 1)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(grid, (0, y), (width, y), (0, 255, 0), 1)

        return grid

    def _save_calibration(self):
        """Save current calibration to cache."""
        if self.current_calibration:
            cache_file = self.cache_dir / "geometric_calibration.json"
            with open(cache_file, "w") as f:
                json.dump(self.current_calibration.to_dict(), f, indent=2)

    def load_calibration(self, filepath: Optional[str] = None) -> bool:
        """Load geometric calibration from file.

        Args:
            filepath: Path to calibration file (uses cache if None)

        Returns:
            True if loaded successfully
        """
        try:
            if filepath is None:
                filepath = self.cache_dir / "geometric_calibration.json"

            with open(filepath) as f:
                data = json.load(f)

            self.current_calibration = GeometricCalibration.from_dict(data)
            logger.info("Geometric calibration loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load geometric calibration: {e}")
            return False

    def export_calibration(self, filepath: str) -> bool:
        """Export geometric calibration.

        Args:
            filepath: Export file path

        Returns:
            True if exported successfully
        """
        try:
            if self.current_calibration is None:
                logger.error("No calibration to export")
                return False

            with open(filepath, "w") as f:
                json.dump(self.current_calibration.to_dict(), f, indent=2)

            logger.info(f"Geometric calibration exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export calibration: {e}")
            return False

    def import_calibration(self, filepath: str) -> bool:
        """Import geometric calibration.

        Args:
            filepath: Import file path

        Returns:
            True if imported successfully
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            self.current_calibration = GeometricCalibration.from_dict(data)
            logger.info(f"Geometric calibration imported from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import calibration: {e}")
            return False
