"""Table detection algorithms."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from core.constants_4k import POCKET_RADIUS_4K
from core.resolution_converter import ResolutionConverter
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

    def __init__(
        self, config: dict[str, Any], camera_resolution: tuple[int, int] = (1920, 1080)
    ) -> None:
        """Initialize table detector with configuration.

        Args:
            config: Configuration dictionary
            camera_resolution: Camera resolution (width, height) for scaling pixel-based parameters.
                             Defaults to 1080p. All pixel values are scaled from 4K canonical.
        """
        self.config = config
        self.camera_resolution = camera_resolution

        # Calculate scale factor from 4K to camera resolution
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(
            camera_resolution
        )
        self.scale = scale_x  # Use X scale for consistency

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
        self.playing_surface_inset_ratio = geometry_config.get(
            "playing_surface_inset_ratio", 0.05
        )  # Ratio to inset for playing surface

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

        # Pocket detection parameters (RESOLUTION-AWARE)
        pocket_config = config.get("pocket_detection", {})
        self.pocket_color_threshold = pocket_config.get("color_threshold", 30)

        # Scale pocket dimensions from 4K
        # POCKET_RADIUS_4K = 72 pixels in 4K
        pocket_radius_scaled = POCKET_RADIUS_4K * self.scale
        default_min_pocket_area = int(
            math.pi * (pocket_radius_scaled * 0.3) ** 2
        )  # 30% of pocket area
        default_max_pocket_area = int(
            math.pi * (pocket_radius_scaled * 1.5) ** 2
        )  # 150% of pocket area

        self.min_pocket_area = pocket_config.get("min_area", default_min_pocket_area)
        self.max_pocket_area = pocket_config.get("max_area", default_max_pocket_area)
        self.min_pocket_confidence = pocket_config.get("min_confidence", 0.5)
        self.max_expected_pocket_distance = pocket_config.get(
            "max_expected_distance", int(200 * self.scale)
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
        self, frame: NDArray[np.uint8], use_pocket_detection: bool = False
    ) -> Optional[TableCorners]:
        """Detect table edges and corners (FR-VIS-011, FR-VIS-012).

        Uses combined color and edge detection for robust boundary identification.
        Achieves sub-pixel accuracy through corner refinement.

        If use_pocket_detection is True, attempts to detect corner pockets and
        use their centers as the table corners for more accurate boundaries.
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

        # Try pocket-based detection if enabled
        if use_pocket_detection:
            pocket_corners = self._detect_corners_from_pockets(frame, corners)
            if pocket_corners is not None:
                corners = pocket_corners

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
        """Complete table detection pipeline combining all detection methods.

        Returns table detection with all coordinates converted to 4K canonical (3840×2160).
        This ensures consistent coordinate space regardless of camera resolution.
        """
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

        # Convert all coordinates to 4K canonical
        source_resolution = (
            frame.shape[1],
            frame.shape[0],
        )  # (width, height) from (height, width)

        # Convert corners to 4K
        corner_list = corners.to_list()
        corners_4k = []
        for corner in corner_list:
            x_4k, y_4k = ResolutionConverter.scale_to_4k(
                corner[0], corner[1], source_resolution
            )
            corners_4k.append((x_4k, y_4k))

        corners_4k_obj = TableCorners(
            top_left=corners_4k[0],
            top_right=corners_4k[1],
            bottom_left=corners_4k[2],
            bottom_right=corners_4k[3],
        )

        # Convert pockets to 4K
        pockets_4k = []
        for pocket in pockets:
            pos_x_4k, pos_y_4k = ResolutionConverter.scale_to_4k(
                pocket.position[0], pocket.position[1], source_resolution
            )
            size_4k = ResolutionConverter.scale_distance_to_4k(
                pocket.size, source_resolution
            )
            pockets_4k.append(
                Pocket(
                    position=(pos_x_4k, pos_y_4k),
                    size=size_4k,
                    pocket_type=pocket.pocket_type,
                    confidence=pocket.confidence,
                )
            )

        # Convert dimensions to 4K
        width_4k = ResolutionConverter.scale_distance_to_4k(width, source_resolution)
        height_4k = ResolutionConverter.scale_distance_to_4k(height, source_resolution)

        return TableDetectionResult(
            corners=corners_4k_obj,
            pockets=pockets_4k,
            surface_color=surface_color,
            width=width_4k,
            height=height_4k,
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

        # If no color ranges configured, create a default mask using green felt
        if not masks:
            # Default green felt color range in HSV
            default_lower = np.array([35, 40, 40])
            default_upper = np.array([85, 255, 255])
            combined_mask = cv2.inRange(hsv, default_lower, default_upper)
        else:
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

        # If we still don't have 4 points, use convex hull (handles pockets)
        if len(approx) != 4:
            hull = cv2.convexHull(contour)
            # Try to approximate the convex hull to a quadrilateral
            for epsilon_mult in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                epsilon = epsilon_mult * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, epsilon, True)
                if len(approx) == 4:
                    break

        if len(approx) != 4:
            return None

        # Extract corner points
        corners = approx.reshape(-1, 2).astype(np.float32)

        # Sort corners to consistent order (top-left, top-right, bottom-right, bottom-left)
        corners = self._sort_corners(corners)

        # Apply inset to find the cushion edge (playing surface boundary)
        corners = self._inset_corners(corners)

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
        # Sort points by y-coordinate to separate top and bottom rows
        corners_by_y = sorted(corners, key=lambda p: p[1])

        # Top two points (smallest y values)
        top_points = corners_by_y[:2]
        # Bottom two points (largest y values)
        bottom_points = corners_by_y[2:]

        # Sort each pair by x-coordinate (left to right)
        top_points = sorted(top_points, key=lambda p: p[0])
        bottom_points = sorted(bottom_points, key=lambda p: p[0])

        # Arrange in the expected order: top-left, top-right, bottom-left, bottom-right
        sorted_corners = [
            top_points[0],  # top-left
            top_points[1],  # top-right
            bottom_points[0],  # bottom-left
            bottom_points[1],  # bottom-right
        ]

        return np.array(sorted_corners)

    def _inset_corners(self, corners: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inset corners to find the playing surface (inner rectangle without rails).

        Takes the detected table boundary and moves each corner inward by a ratio
        of the table dimensions to approximate the playing surface.
        """
        # Calculate table dimensions
        top_left, top_right, bottom_left, bottom_right = corners

        # Calculate width and height
        width = np.linalg.norm(top_right - top_left)
        height = np.linalg.norm(bottom_left - top_left)

        # Calculate inset distance
        inset_horizontal = width * self.playing_surface_inset_ratio
        inset_vertical = height * self.playing_surface_inset_ratio

        # Calculate direction vectors for each edge
        # Top edge: left to right
        top_dir = (top_right - top_left) / width
        # Bottom edge: left to right
        bottom_dir = (bottom_right - bottom_left) / np.linalg.norm(
            bottom_right - bottom_left
        )
        # Left edge: top to bottom
        left_dir = (bottom_left - top_left) / height
        # Right edge: top to bottom
        right_dir = (bottom_right - top_right) / np.linalg.norm(
            bottom_right - top_right
        )

        # Inset each corner
        # Top-left: move right and down
        new_top_left = top_left + top_dir * inset_horizontal + left_dir * inset_vertical

        # Top-right: move left and down
        new_top_right = (
            top_right - top_dir * inset_horizontal + right_dir * inset_vertical
        )

        # Bottom-left: move right and up
        new_bottom_left = (
            bottom_left + bottom_dir * inset_horizontal - left_dir * inset_vertical
        )

        # Bottom-right: move left and up
        new_bottom_right = (
            bottom_right - bottom_dir * inset_horizontal - right_dir * inset_vertical
        )

        return np.array(
            [new_top_left, new_top_right, new_bottom_left, new_bottom_right]
        )

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

    def _detect_corners_from_pockets(
        self, frame: NDArray[np.uint8], approximate_corners: TableCorners
    ) -> Optional[TableCorners]:
        """Detect table corners by finding the centers of corner pockets.

        Uses the approximate corners to establish a search region, then detects
        the four corner pockets and uses their centers as the precise table corners.
        """
        # Convert to grayscale for dark region detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create an expanded region of interest to search for pockets
        # We'll search slightly beyond the approximate corners
        corner_list = approximate_corners.to_list()
        expansion_factor = 0.15  # Expand search region by 15%

        # Calculate bounds with expansion
        min_x = min(c[0] for c in corner_list)
        max_x = max(c[0] for c in corner_list)
        min_y = min(c[1] for c in corner_list)
        max_y = max(c[1] for c in corner_list)

        width = max_x - min_x
        height = max_y - min_y

        search_min_x = max(0, int(min_x - width * expansion_factor))
        search_max_x = min(gray.shape[1], int(max_x + width * expansion_factor))
        search_min_y = max(0, int(min_y - height * expansion_factor))
        search_max_y = min(gray.shape[0], int(max_y + height * expansion_factor))

        # Create ROI mask for pocket search
        roi_mask = np.zeros(gray.shape, dtype=np.uint8)
        roi_mask[search_min_y:search_max_y, search_min_x:search_max_x] = 255

        # Find dark regions (potential pockets) with lower threshold for better detection
        dark_mask = cv2.inRange(gray, 0, self.pocket_color_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, roi_mask)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        if self.debug_mode:
            self.debug_images.append(("corner_pockets_mask", dark_mask))

        # Find contours for potential pockets
        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area and find corner pockets
        pocket_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Use a relaxed area constraint for pockets
            if self.min_pocket_area * 0.5 <= area <= self.max_pocket_area * 2.0:
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]

                # Calculate how "corner-like" this pocket is (should be near the extremes)
                corner_score = 0
                if abs(center_x - min_x) < width * 0.2:  # Near left edge
                    corner_score += 1
                if abs(center_x - max_x) < width * 0.2:  # Near right edge
                    corner_score += 1
                if abs(center_y - min_y) < height * 0.2:  # Near top edge
                    corner_score += 1
                if abs(center_y - max_y) < height * 0.2:  # Near bottom edge
                    corner_score += 1

                # Corner pockets should be near two edges
                if corner_score >= 2:
                    pocket_candidates.append((center_x, center_y, area, corner_score))

        # Sort by corner_score (descending) and area (descending)
        pocket_candidates.sort(key=lambda p: (p[3], p[2]), reverse=True)

        # We need exactly 4 corner pockets
        if len(pocket_candidates) < 4:
            return None  # Fall back to approximate corners

        # Instead of just taking the top 4, we need to ensure we get one pocket
        # from each corner quadrant (TL, TR, BL, BR)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Group candidates by quadrant
        quadrants = {"TL": [], "TR": [], "BL": [], "BR": []}
        for cx, cy, area, score in pocket_candidates:
            if cx < center_x and cy < center_y:
                quadrants["TL"].append((cx, cy, area, score))
            elif cx >= center_x and cy < center_y:
                quadrants["TR"].append((cx, cy, area, score))
            elif cx < center_x and cy >= center_y:
                quadrants["BL"].append((cx, cy, area, score))
            else:
                quadrants["BR"].append((cx, cy, area, score))

        # Check if we have at least one candidate in each quadrant
        if not all(quadrants.values()):
            return None  # Missing pocket in at least one quadrant

        # Take the best candidate from each quadrant
        pocket_centers = []
        for quadrant in ["TL", "TR", "BL", "BR"]:
            # Sort by score then area
            quadrants[quadrant].sort(key=lambda p: (p[3], p[2]), reverse=True)
            best = quadrants[quadrant][0]
            pocket_centers.append((best[0], best[1]))

        # Sort the pocket centers to match the expected corner order
        # top-left, top-right, bottom-left, bottom-right
        pocket_centers_array = np.array(pocket_centers, dtype=np.float32)
        sorted_pockets = self._sort_corners(pocket_centers_array)

        return TableCorners(
            top_left=tuple(sorted_pockets[0]),
            top_right=tuple(sorted_pockets[1]),
            bottom_left=tuple(sorted_pockets[2]),
            bottom_right=tuple(sorted_pockets[3]),
        )

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
