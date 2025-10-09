"""Table detection algorithms."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray


class PocketType(Enum):
    """Types of pockets on a pool table."""

    CORNER = "corner"
    SIDE = "side"


@dataclass
class Pocket:
    """Individual pocket information."""

    position: tuple[float, float]  # Center position (x, y)
    size: float  # Effective radius in pixels
    pocket_type: PocketType
    confidence: float  # Detection confidence 0.0-1.0


@dataclass
class TableCorners:
    """Table corner positions with sub-pixel accuracy."""

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]

    def to_list(self) -> list[tuple[float, float]]:
        """Convert to list format."""
        return [self.top_left, self.top_right, self.bottom_left, self.bottom_right]


@dataclass
class TableDetectionResult:
    """Complete table detection result."""

    corners: TableCorners
    pockets: list[Pocket]
    surface_color: tuple[int, int, int]  # Average HSV color
    width: float  # Table width in pixels
    height: float  # Table height in pixels
    confidence: float  # Overall detection confidence
    perspective_transform: Optional[NDArray[np.float64]] = None  # Transformation matrix


class TableDetector:
    """Pool table detection and boundary identification."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize table detector with configuration."""
        self.config = config

        # Load color ranges configuration
        color_ranges_config = config.get("color_ranges", {})
        self.table_color_ranges = {}
        for color_name, color_range in color_ranges_config.items():
            self.table_color_ranges[color_name] = {
                "lower": np.array(color_range.get("lower_hsv", [0, 0, 0])),
                "upper": np.array(color_range.get("upper_hsv", [179, 255, 255])),
            }

        # Table geometry constraints
        geometry_config = config.get("geometry", {})
        self.expected_aspect_ratio = geometry_config.get(
            "expected_aspect_ratio", 2.0
        )  # Standard 9ft table is 2:1
        self.aspect_ratio_tolerance = geometry_config.get("aspect_ratio_tolerance", 0.3)
        self.side_length_tolerance = geometry_config.get(
            "side_length_tolerance", 0.1
        )  # Max difference between opposite sides
        self.min_table_area_ratio = geometry_config.get(
            "min_table_area_ratio", 0.1
        )  # Minimum % of image
        self.max_table_area_ratio = geometry_config.get(
            "max_table_area_ratio", 0.9
        )  # Maximum % of image

        # Edge detection parameters
        edge_config = config.get("edge_detection", {})
        self.canny_low = edge_config.get("canny_low_threshold", 50)
        self.canny_high = edge_config.get("canny_high_threshold", 150)
        self.contour_epsilon_factor = edge_config.get("contour_epsilon_factor", 0.02)
        self.contour_epsilon_multipliers = edge_config.get(
            "contour_epsilon_multipliers", [0.01, 0.03, 0.04, 0.05]
        )

        # Corner refinement parameters
        corner_config = config.get("corner_refinement", {})
        self.corner_window_size = corner_config.get("window_size", 5)
        self.corner_max_iterations = corner_config.get("max_iterations", 30)
        self.corner_epsilon = corner_config.get("epsilon", 0.001)

        # Pocket detection parameters
        pocket_config = config.get("pocket_detection", {})
        self.pocket_color_threshold = pocket_config.get("color_threshold", 30)
        self.min_pocket_area = pocket_config.get("min_area", 100)
        self.max_pocket_area = pocket_config.get("max_area", 2000)
        self.min_pocket_confidence = pocket_config.get("min_confidence", 0.5)
        self.max_expected_pocket_distance = pocket_config.get(
            "max_expected_distance", 100
        )

        # Morphology parameters
        morphology_config = config.get("morphology", {})
        large_kernel = morphology_config.get("large_kernel_size", [5, 5])
        small_kernel = morphology_config.get("small_kernel_size", [3, 3])
        self.large_kernel = tuple(large_kernel)
        self.small_kernel = tuple(small_kernel)

        # Temporal stability parameters
        temporal_config = config.get("temporal_stability", {})
        self.blending_alpha = temporal_config.get("blending_alpha", 0.7)
        self.min_previous_confidence = temporal_config.get(
            "min_previous_confidence", 0.5
        )

        # Confidence weights
        confidence_config = config.get("confidence_weights", {})
        self.confidence_weight_geometry = confidence_config.get("geometry", 0.4)
        self.confidence_weight_pockets = confidence_config.get("pockets", 0.3)
        self.confidence_weight_surface = confidence_config.get("surface", 0.3)

        # Debug settings
        self.debug_mode = config.get("debug", False)
        self.debug_images: list[tuple[str, NDArray[np.uint8]]] = []

    def detect_table_boundaries(
        self, frame: NDArray[np.uint8]
    ) -> Optional[TableCorners]:
        """Detect table edges and corners (FR-VIS-011, FR-VIS-012).

        Uses combined color and edge detection for robust boundary identification.
        Achieves sub-pixel accuracy through corner refinement.
        """
        if frame is None or frame.size == 0:
            return None

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for table surface color
        table_mask = self._create_table_color_mask(hsv)

        if self.debug_mode:
            self.debug_images.append(("table_mask", table_mask))

        # Find table contour using color mask
        table_contour = self._find_table_contour(table_mask)

        if table_contour is None:
            return None

        # Refine corners with edge detection
        corners = self._refine_corners_with_edges(frame, table_contour)

        if corners is None:
            return None

        # Validate geometry
        if not self._validate_table_geometry(corners):
            return None

        return corners

    def detect_table_surface(
        self, frame: NDArray[np.uint8]
    ) -> Optional[tuple[NDArray[np.uint8], tuple[int, int, int]]]:
        """Distinguish table surface from surrounding environment (FR-VIS-013).

        Returns binary mask and average color of detected table surface.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Try different color ranges to find the best match
        best_mask = None
        best_color = None
        best_area = 0

        for _color_name, color_range in self.table_color_ranges.items():
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])

            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.large_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find the largest contour
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > best_area:
                    best_area = float(area)
                    best_mask = mask
                    # Calculate average color in the masked region
                    mean_color = cv2.mean(hsv, mask=mask)
                    best_color = (
                        int(mean_color[0]),
                        int(mean_color[1]),
                        int(mean_color[2]),
                    )

        if (
            best_mask is not None
            and best_area > frame.shape[0] * frame.shape[1] * self.min_table_area_ratio
        ):
            return best_mask, best_color

        return None

    def detect_pockets(
        self, frame: NDArray[np.uint8], table_corners: TableCorners
    ) -> list[Pocket]:
        """Detect pocket locations (FR-VIS-016 to FR-VIS-019).

        Locates all six pockets and determines their type and characteristics.
        """
        if table_corners is None:
            return []

        # Convert to grayscale for dark region detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create region of interest based on table boundaries
        table_mask = self._create_table_roi_mask(frame.shape[:2], table_corners)

        # Find dark regions (potential pockets)
        dark_mask = cv2.inRange(gray, 0, self.pocket_color_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, table_mask)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.small_kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        if self.debug_mode:
            self.debug_images.append(("pocket_mask", dark_mask))

        # Find contours for potential pockets
        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        pockets = []
        corner_positions = self._get_expected_pocket_positions(table_corners)

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_pocket_area <= area <= self.max_pocket_area:
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
                center = (center_x, center_y)

                # Determine pocket type based on position
                pocket_type, confidence = self._classify_pocket_type(
                    center, corner_positions
                )

                if confidence > self.min_pocket_confidence:
                    # Calculate effective radius
                    radius = math.sqrt(area / math.pi)

                    pocket = Pocket(
                        position=center,
                        size=radius,
                        pocket_type=pocket_type,
                        confidence=confidence,
                    )
                    pockets.append(pocket)

        # Sort pockets and ensure we have exactly 6 (or close to it)
        pockets = self._validate_and_sort_pockets(pockets, corner_positions)

        return pockets

    def handle_occlusions(
        self,
        frame: NDArray[np.uint8],
        previous_detection: Optional[TableDetectionResult] = None,
    ) -> Optional[TableDetectionResult]:
        """Handle partial table visibility and occlusions (FR-VIS-014).

        Uses previous detection results to maintain stable tracking.
        """
        current_detection = self.detect_complete_table(frame)

        if current_detection is None and previous_detection is not None:
            # Try to use previous detection with validation
            if self._validate_previous_detection(frame, previous_detection):
                return previous_detection

        if current_detection is not None and previous_detection is not None:
            # Blend current and previous detections for stability
            return self._blend_detections(current_detection, previous_detection)

        return current_detection

    def validate_table_dimensions(self, corners: TableCorners) -> bool:
        """Validate detected table dimensions against expected ratios (FR-VIS-015)."""
        return self._validate_table_geometry(corners)

    def detect_complete_table(
        self, frame: NDArray[np.uint8]
    ) -> Optional[TableDetectionResult]:
        """Complete table detection pipeline combining all detection methods."""
        # Detect table boundaries
        corners = self.detect_table_boundaries(frame)

        if corners is None:
            return None

        # Detect table surface
        surface_result = self.detect_table_surface(frame)

        if surface_result is None:
            return None

        surface_mask, surface_color = surface_result

        # Detect pockets
        pockets = self.detect_pockets(frame, corners)

        # Calculate table dimensions
        width = self._calculate_table_width(corners)
        height = self._calculate_table_height(corners)

        # Calculate overall confidence
        confidence = self._calculate_detection_confidence(
            corners, pockets, surface_mask
        )

        # Generate perspective correction transform
        transform = self._generate_perspective_transform(corners, width, height)

        return TableDetectionResult(
            corners=corners,
            pockets=pockets,
            surface_color=surface_color,
            width=width,
            height=height,
            confidence=confidence,
            perspective_transform=transform,
        )

    def calibrate_table(self, frame: NDArray[np.uint8]) -> dict:
        """Perform table calibration for perspective correction."""
        detection = self.detect_complete_table(frame)

        if detection is None:
            return {"success": False, "error": "Could not detect table"}

        calibration_data = {
            "success": True,
            "table_corners": detection.corners.to_list(),
            "table_dimensions": (detection.width, detection.height),
            "surface_color": detection.surface_color,
            "perspective_transform": (
                detection.perspective_transform.tolist()
                if detection.perspective_transform is not None
                else None
            ),
            "pocket_count": len(detection.pockets),
            "confidence": detection.confidence,
        }

        return calibration_data

    def get_debug_images(self) -> list[tuple[str, np.ndarray]]:
        """Get debug visualization images."""
        return self.debug_images

    def clear_debug_images(self) -> None:
        """Clear debug image buffer."""
        self.debug_images.clear()

    # Private helper methods

    def _create_table_color_mask(self, hsv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Create binary mask for table surface color."""
        masks = []

        for color_range in self.table_color_ranges.values():
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            masks.append(mask)

        # Combine all color masks
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.large_kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        return combined_mask

    def _find_table_contour(
        self, mask: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float64]]:
        """Find the largest table contour from color mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if contour is large enough
        area = cv2.contourArea(largest_contour)
        image_area = mask.shape[0] * mask.shape[1]

        if area < image_area * self.min_table_area_ratio:
            return None

        return largest_contour

    def _refine_corners_with_edges(
        self, frame: NDArray[np.uint8], contour: NDArray[np.float64]
    ) -> Optional[TableCorners]:
        """Refine corner detection using edge detection for sub-pixel accuracy."""
        # Approximate contour to quadrilateral
        epsilon = self.contour_epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If we don't get exactly 4 points, try different epsilon values
        if len(approx) != 4:
            for epsilon_mult in self.contour_epsilon_multipliers:
                epsilon = epsilon_mult * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    break

        if len(approx) != 4:
            return None

        # Extract corner points
        corners = approx.reshape(-1, 2).astype(np.float32)

        # Sort corners to consistent order (top-left, top-right, bottom-right, bottom-left)
        corners = self._sort_corners(corners)

        # Refine corners using edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        refined_corners = []
        for corner in corners:
            refined_corner = self._refine_corner_subpixel(gray, corner)
            refined_corners.append(refined_corner)

        return TableCorners(
            top_left=tuple(refined_corners[0]),
            top_right=tuple(refined_corners[1]),
            bottom_left=tuple(refined_corners[2]),
            bottom_right=tuple(refined_corners[3]),
        )

    def _sort_corners(self, corners: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sort corners to consistent order: top-left, top-right, bottom-left, bottom-right."""
        # Calculate center point
        center = np.mean(corners, axis=0)

        # Sort by angle from center
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])

        # Sort corners by angle
        corners_with_angles = [
            (corner, angle_from_center(corner)) for corner in corners
        ]
        corners_with_angles.sort(key=lambda x: x[1])

        sorted_corners = [corner for corner, _ in corners_with_angles]

        # Ensure correct order: top-left, top-right, bottom-right, bottom-left
        # The first corner with smallest angle should be the rightmost
        # We need to rotate the list to get top-left first

        # Find the top-left corner (smallest x + y)
        sum_coords = [corner[0] + corner[1] for corner in sorted_corners]
        top_left_idx = sum_coords.index(min(sum_coords))

        # Rotate list to start with top-left
        sorted_corners = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]

        return np.array(sorted_corners)

    def _refine_corner_subpixel(
        self,
        gray: NDArray[np.float64],
        corner: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Refine corner position to sub-pixel accuracy."""
        x, y = int(corner[0]), int(corner[1])

        # Define search window
        half_window = self.corner_window_size // 2
        x1, y1 = max(0, x - half_window), max(0, y - half_window)
        x2, y2 = (
            min(gray.shape[1], x + half_window + 1),
            min(gray.shape[0], y + half_window + 1),
        )

        if x2 - x1 < 3 or y2 - y1 < 3:
            return corner

        # Extract region of interest
        gray[y1:y2, x1:x2]

        # Use corner sub-pixel refinement
        corners_refined = cv2.cornerSubPix(
            gray,
            np.array([[corner]], dtype=np.float32),
            (half_window, half_window),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                self.corner_max_iterations,
                self.corner_epsilon,
            ),
        )

        return corners_refined[0][0]

    def _validate_table_geometry(self, corners: TableCorners) -> bool:
        """Validate that detected corners form a valid table shape."""
        corner_list = corners.to_list()

        # Calculate side lengths
        width1 = self._distance(corner_list[0], corner_list[1])  # top edge
        width2 = self._distance(corner_list[2], corner_list[3])  # bottom edge
        height1 = self._distance(corner_list[0], corner_list[2])  # left edge
        height2 = self._distance(corner_list[1], corner_list[3])  # right edge

        # Check if opposite sides are approximately equal
        width_diff = abs(width1 - width2) / max(width1, width2)
        height_diff = abs(height1 - height2) / max(height1, height2)

        if (
            width_diff > self.side_length_tolerance
            or height_diff > self.side_length_tolerance
        ):
            return False

        # Check aspect ratio
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        aspect_ratio = avg_width / avg_height

        expected_ratio = self.expected_aspect_ratio
        return not abs(aspect_ratio - expected_ratio) > self.aspect_ratio_tolerance

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _create_table_roi_mask(
        self, image_shape: tuple[int, int], corners: TableCorners
    ) -> NDArray[np.float64]:
        """Create a mask for the table region of interest."""
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Create polygon mask from table corners
        corner_points = np.array(corners.to_list(), dtype=np.int32)
        cv2.fillPoly(mask, [corner_points], 255)

        return mask

    def _get_expected_pocket_positions(
        self, corners: TableCorners
    ) -> dict[str, tuple[float, float]]:
        """Calculate expected pocket positions based on table corners."""
        corner_list = corners.to_list()

        # Corner pockets are at the table corners
        corner_pockets = {
            "corner_tl": corner_list[0],  # top-left
            "corner_tr": corner_list[1],  # top-right
            "corner_bl": corner_list[2],  # bottom-left
            "corner_br": corner_list[3],  # bottom-right
        }

        # Side pockets are at the midpoints of the long sides
        side_pockets = {
            "side_top": (
                (corner_list[0][0] + corner_list[1][0]) / 2,
                (corner_list[0][1] + corner_list[1][1]) / 2,
            ),
            "side_bottom": (
                (corner_list[2][0] + corner_list[3][0]) / 2,
                (corner_list[2][1] + corner_list[3][1]) / 2,
            ),
        }

        return {**corner_pockets, **side_pockets}

    def _classify_pocket_type(
        self,
        center: tuple[float, float],
        expected_positions: dict[str, tuple[float, float]],
    ) -> tuple[PocketType, float]:
        """Classify pocket type based on position relative to expected locations."""
        min_distance = float("inf")
        best_match = None

        for pocket_name, expected_pos in expected_positions.items():
            distance = self._distance(center, expected_pos)
            if distance < min_distance:
                min_distance = distance
                best_match = pocket_name

        # Determine pocket type
        if best_match and best_match.startswith("corner"):
            pocket_type = PocketType.CORNER
        else:
            pocket_type = PocketType.SIDE

        # Calculate confidence based on distance (closer = higher confidence)
        confidence = max(0.0, 1.0 - (min_distance / self.max_expected_pocket_distance))

        return pocket_type, confidence

    def _validate_and_sort_pockets(
        self, pockets: list[Pocket], expected_positions: dict[str, tuple[float, float]]
    ) -> list[Pocket]:
        """Validate and sort pockets to ensure consistent ordering."""
        if len(pockets) > 6:
            # Too many pockets detected, keep the 6 most confident
            pockets.sort(key=lambda p: p.confidence, reverse=True)
            pockets = pockets[:6]

        # Sort pockets by position (top to bottom, left to right)
        pockets.sort(key=lambda p: (p.position[1], p.position[0]))

        return pockets

    def _calculate_table_width(self, corners: TableCorners) -> float:
        """Calculate table width in pixels."""
        corner_list = corners.to_list()
        width1 = self._distance(corner_list[0], corner_list[1])  # top edge
        width2 = self._distance(corner_list[2], corner_list[3])  # bottom edge
        return (width1 + width2) / 2

    def _calculate_table_height(self, corners: TableCorners) -> float:
        """Calculate table height in pixels."""
        corner_list = corners.to_list()
        height1 = self._distance(corner_list[0], corner_list[2])  # left edge
        height2 = self._distance(corner_list[1], corner_list[3])  # right edge
        return (height1 + height2) / 2

    def _calculate_detection_confidence(
        self,
        corners: TableCorners,
        pockets: list[Pocket],
        surface_mask: NDArray[np.float64],
    ) -> float:
        """Calculate overall detection confidence."""
        # Base confidence from geometry validation
        geometry_confidence = 1.0 if self._validate_table_geometry(corners) else 0.0

        # Pocket confidence (ideal is 6 pockets)
        pocket_confidence = min(1.0, len(pockets) / 6.0)

        # Surface area confidence
        total_pixels = surface_mask.shape[0] * surface_mask.shape[1]
        surface_pixels = np.sum(surface_mask > 0)
        surface_confidence = min(
            1.0, surface_pixels / (total_pixels * self.min_table_area_ratio)
        )

        # Weighted average
        overall_confidence = (
            geometry_confidence * self.confidence_weight_geometry
            + pocket_confidence * self.confidence_weight_pockets
            + surface_confidence * self.confidence_weight_surface
        )

        return overall_confidence

    def _generate_perspective_transform(
        self, corners: TableCorners, width: float, height: float
    ) -> NDArray[np.float64]:
        """Generate perspective transformation matrix for table rectification."""
        # Source points (detected corners)
        src_points = np.array(corners.to_list(), dtype=np.float32)

        # Destination points (rectified rectangle)
        dst_points = np.array(
            [
                [0, 0],  # top-left
                [width, 0],  # top-right
                [0, height],  # bottom-left
                [width, height],  # bottom-right
            ],
            dtype=np.float32,
        )

        # Calculate perspective transformation matrix
        transform = cv2.getPerspectiveTransform(src_points, dst_points)

        return transform

    def _validate_previous_detection(
        self, frame: NDArray[np.uint8], previous: TableDetectionResult
    ) -> bool:
        """Validate if previous detection is still valid for current frame."""
        # Simple validation - check if the previous corners still make sense
        # This could be enhanced with more sophisticated tracking
        return previous.confidence > self.min_previous_confidence

    def _blend_detections(
        self,
        current: TableDetectionResult,
        previous: TableDetectionResult,
    ) -> TableDetectionResult:
        """Blend current and previous detections for temporal stability."""
        # Simple blending of corner positions
        # This could be enhanced with Kalman filtering

        current_corners = current.corners.to_list()
        previous_corners = previous.corners.to_list()

        blended_corners = []
        for curr, prev in zip(current_corners, previous_corners):
            blended_x = (
                self.blending_alpha * curr[0] + (1 - self.blending_alpha) * prev[0]
            )
            blended_y = (
                self.blending_alpha * curr[1] + (1 - self.blending_alpha) * prev[1]
            )
            blended_corners.append((blended_x, blended_y))

        blended_table_corners = TableCorners(
            top_left=blended_corners[0],
            top_right=blended_corners[1],
            bottom_left=blended_corners[2],
            bottom_right=blended_corners[3],
        )

        # Use current detection for other properties but blended corners
        return TableDetectionResult(
            corners=blended_table_corners,
            pockets=current.pockets,
            surface_color=current.surface_color,
            width=current.width,
            height=current.height,
            confidence=max(current.confidence, previous.confidence),
            perspective_transform=self._generate_perspective_transform(
                blended_table_corners, current.width, current.height
            ),
        )
