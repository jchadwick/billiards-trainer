"""Ball detection and classification algorithms for the vision module.

Provides comprehensive ball detection including:
- Circle detection using Hough transforms
- Color-based ball classification
- Number/pattern recognition
- Size and position validation
- Motion detection

Implements requirements:
- FR-VIS-020: Detect all balls on the table surface
- FR-VIS-021: Distinguish between different ball types (cue, solid, stripe, 8-ball)
- FR-VIS-022: Identify ball numbers/patterns when visible
- FR-VIS-023: Track ball positions with ±2 pixel accuracy
- FR-VIS-024: Measure ball radius for size validation
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..models import Ball, BallType

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Ball detection methods."""

    HOUGH_CIRCLES = "hough"
    CONTOUR_BASED = "contour"
    BLOB_DETECTION = "blob"
    COMBINED = "combined"


@dataclass
class BallDetectionConfig:
    """Configuration for ball detection."""

    # Detection method
    detection_method: DetectionMethod = DetectionMethod.COMBINED

    # Hough circle parameters
    hough_dp: float = 1.0
    hough_min_dist_ratio: float = 0.8
    hough_param1: int = 50
    hough_param2: int = 30
    hough_accumulator_threshold: int = 15
    hough_gaussian_blur_kernel: int = 9
    hough_gaussian_blur_sigma: int = 2

    # Size constraints
    min_radius: int = 15
    max_radius: int = 26
    expected_radius: int = 20
    radius_tolerance: float = 0.30

    # Color classification
    ball_colors: dict[str, dict[str, tuple[int, int, int]]] = None

    # Quality filters
    min_circularity: float = 0.75
    min_confidence: float = 0.4
    max_overlap_ratio: float = 0.30
    min_convexity: float = 0.8
    min_inertia_ratio: float = 0.5

    # Performance optimization
    roi_enabled: bool = True
    roi_margin: int = 50

    # Debug settings
    debug_mode: bool = False
    save_debug_images: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BallDetectionConfig":
        """Create BallDetectionConfig from nested configuration dictionary.

        Args:
            config: Nested configuration dictionary with sections like
                   hough_circles, size_constraints, etc.

        Returns:
            BallDetectionConfig instance
        """
        # Extract detection method
        method_str = config.get("detection_method", "combined")
        if isinstance(method_str, str):
            detection_method = DetectionMethod(method_str)
        else:
            detection_method = method_str

        # Extract hough parameters
        hough_config = config.get("hough_circles", {})

        # Extract size constraints
        size_config = config.get("size_constraints", {})

        # Extract quality filters (support both "quality" and "quality_filters" keys)
        quality_config = config.get("quality_filters", config.get("quality", {}))

        # Extract performance settings
        perf_config = config.get("performance", {})

        # Extract debug settings
        debug_config = config.get("debug", {})

        # Extract ball colors and convert to tuple format
        ball_colors_raw = config.get("ball_colors", {})
        ball_colors = {}
        for color_name, color_range in ball_colors_raw.items():
            ball_colors[color_name] = {
                "lower": tuple(color_range.get("lower", [0, 0, 0])),
                "upper": tuple(color_range.get("upper", [180, 255, 255])),
            }

        return cls(
            detection_method=detection_method,
            # Hough parameters
            hough_dp=hough_config.get("dp", 1.0),
            hough_min_dist_ratio=hough_config.get("min_dist_ratio", 0.8),
            hough_param1=hough_config.get("param1", 50),
            hough_param2=hough_config.get("param2", 30),
            hough_accumulator_threshold=hough_config.get("accumulator_threshold", 15),
            hough_gaussian_blur_kernel=hough_config.get("gaussian_blur_kernel", 9),
            hough_gaussian_blur_sigma=hough_config.get("gaussian_blur_sigma", 2),
            # Size constraints
            min_radius=size_config.get("min_radius", 15),
            max_radius=size_config.get("max_radius", 26),
            expected_radius=size_config.get("expected_radius", 20),
            radius_tolerance=size_config.get("radius_tolerance", 0.30),
            # Ball colors
            ball_colors=ball_colors if ball_colors else None,
            # Quality filters
            min_circularity=quality_config.get("min_circularity", 0.75),
            min_confidence=quality_config.get("min_confidence", 0.4),
            max_overlap_ratio=quality_config.get("max_overlap_ratio", 0.30),
            min_convexity=quality_config.get("min_convexity", 0.8),
            min_inertia_ratio=quality_config.get("min_inertia_ratio", 0.5),
            # Performance
            roi_enabled=perf_config.get("roi_enabled", True),
            roi_margin=perf_config.get("roi_margin", 50),
            # Debug
            debug_mode=debug_config.get("debug_mode", False),
            save_debug_images=debug_config.get("save_debug_images", False),
        )

    def __post_init__(self) -> None:
        """Initialize ball colors with default HSV ranges if not provided."""
        if self.ball_colors is None:
            self.ball_colors = {
                "cue": {
                    "lower": (0, 0, 150),  # White ball - lowered threshold
                    "upper": (180, 50, 255),  # Increased saturation tolerance
                },
                "yellow": {
                    "lower": (10, 80, 80),  # Yellow (1, 9) - relaxed thresholds
                    "upper": (35, 255, 255),
                },
                "blue": {
                    "lower": (90, 80, 80),  # Blue (2, 10) - expanded range
                    "upper": (140, 255, 255),
                },
                "red": {
                    "lower": (0, 80, 80),  # Red (3, 11) - relaxed
                    "upper": (15, 255, 255),
                },
                "purple": {
                    "lower": (120, 80, 80),  # Purple (4, 12) - expanded
                    "upper": (170, 255, 255),
                },
                "orange": {
                    "lower": (8, 80, 80),  # Orange (5, 13) - expanded
                    "upper": (25, 255, 255),
                },
                "green": {
                    "lower": (30, 80, 80),  # Green (6, 14) - expanded
                    "upper": (90, 255, 255),
                },
                "maroon": {
                    "lower": (
                        0,
                        60,
                        30,
                    ),  # Maroon (7, 15) - wider range including reddish-brown
                    "upper": (15, 255, 120),  # Darker value range for brown tones
                },
                "black": {
                    "lower": (0, 0, 0),  # Black (8-ball) - expanded
                    "upper": (180, 255, 70),
                },
            }


class BallDetector:
    """Ball detection and classification with multiple detection methods.

    Features:
    - Multiple detection algorithms (Hough, contour, blob)
    - Color-based ball type classification
    - Size and geometry validation
    - Overlap detection and resolution
    - Motion state detection
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize ball detector with configuration.

        Args:
            config: Configuration dictionary
        """
        # Store raw config for accessing nested values
        self.raw_config = config
        self.config = BallDetectionConfig.from_config(config)

        # Initialize detection components
        self._initialize_detectors()

        # Background subtraction
        self.background_frame: Optional[NDArray[np.uint8]] = None
        bg_config = config.get("background_subtraction", {})
        self.use_background_subtraction = bg_config.get("enabled", False)
        self.background_threshold = bg_config.get("threshold", 30)

        # Pocket locations (detected from background frame)
        self.pocket_locations: list[tuple[float, float, float]] = []  # (x, y, radius)
        pocket_config = config.get("pocket_detection", {})
        self.pocket_exclusion_radius_multiplier = pocket_config.get(
            "exclusion_radius_multiplier", 2.0
        )

        # Playing area mask (excludes rails and pockets)
        self.playing_area_mask: Optional[NDArray[np.uint8]] = None

        # Statistics tracking
        self.stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "false_positives": 0,
            "avg_confidence": 0.0,
            "detection_method_stats": {method.value: 0 for method in DetectionMethod},
        }

        # Debug storage
        self.debug_images = []

        logger.info(
            f"Ball detector initialized with method: {self.config.detection_method}"
        )

    def _initialize_detectors(self) -> None:
        """Initialize detection algorithm components."""
        # Blob detector for alternative detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = math.pi * (self.config.min_radius**2)
        params.maxArea = math.pi * (self.config.max_radius**2)
        params.filterByCircularity = True
        params.minCircularity = self.config.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.config.min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = self.config.min_inertia_ratio

        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        logger.debug("Detection algorithms initialized")

    def _create_foreground_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Create foreground mask using background subtraction.

        Args:
            frame: Current frame

        Returns:
            Binary mask where foreground pixels are 255
        """
        # Convert both frames to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if len(self.background_frame.shape) == 3:
            bg_gray = cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2GRAY)
        else:
            bg_gray = self.background_frame

        # Compute absolute difference
        diff = cv2.absdiff(gray, bg_gray)

        # Get background subtraction config
        bg_config = self.raw_config.get("background_subtraction", {})
        threshold_adj = bg_config.get("threshold_adjustment", 10)
        morph_kernel_size = bg_config.get("morph_kernel_size", 5)
        morph_iterations = bg_config.get("morph_iterations", 1)

        # Threshold to get foreground mask
        # Higher threshold = less sensitive = fewer motion artifacts
        _, fg_mask = cv2.threshold(
            diff, self.background_threshold + threshold_adj, 255, cv2.THRESH_BINARY
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Less aggressive dilation to reduce motion blur
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=morph_iterations)

        return fg_mask

    def _detect_pockets_from_background(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Detect pocket locations from background frame.

        Pockets are dark circular regions on the table.

        Args:
            frame: Background frame (empty table)

        Returns:
            List of (x, y, radius) tuples for each pocket
        """
        # Get pocket detection config
        pocket_config = self.raw_config.get("pocket_detection", {})
        threshold = pocket_config.get("threshold", 40)
        morph_kernel_size = pocket_config.get("morph_kernel_size", 9)
        min_radius = pocket_config.get("min_radius", 35)
        max_radius = pocket_config.get("max_radius", 80)
        min_circularity = pocket_config.get("min_circularity", 0.65)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pockets are VERY dark (nearly black), use stricter threshold
        _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        pockets = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Pockets are significantly larger than balls
            min_pocket_area = math.pi * (min_radius**2)
            max_pocket_area = math.pi * (max_radius**2)

            if min_pocket_area <= area <= max_pocket_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter**2)

                    # Pockets should be fairly circular
                    if circularity >= min_circularity:
                        # Calculate center and radius
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            radius = math.sqrt(area / math.pi)
                            pockets.append((cx, cy, radius))

        return pockets

    def _detect_playing_area_from_background(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.uint8]]:
        """Detect the playing area (cloth surface) excluding rails and pockets.

        The playing area is the green/blue cloth where balls can rest.
        Rails (cushions) are typically darker or have different texture.

        Args:
            frame: Background frame (empty table)

        Returns:
            Binary mask where 255 = playing area, 0 = rails/pockets/outside
        """
        # Get playing area config
        playing_area_config = self.raw_config.get("playing_area", {})
        green_cloth = playing_area_config.get("green_cloth", {})
        blue_cloth = playing_area_config.get("blue_cloth", {})
        min_area_ratio = playing_area_config.get("min_area_ratio", 0.1)
        morph_kernel_size = playing_area_config.get("morph_kernel_size", 15)
        erode_kernel_size = playing_area_config.get("erode_kernel_size", 20)
        erode_iterations = playing_area_config.get("erode_iterations", 1)

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect table cloth color (green or blue felt)
        # Use broader ranges to capture the playing surface
        green_lower = np.array(
            [
                green_cloth.get("hue_min", 35),
                green_cloth.get("saturation_min", 30),
                green_cloth.get("value_min", 30),
            ]
        )
        green_upper = np.array(
            [
                green_cloth.get("hue_max", 85),
                green_cloth.get("saturation_max", 255),
                green_cloth.get("value_max", 255),
            ]
        )
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        blue_lower = np.array(
            [
                blue_cloth.get("hue_min", 90),
                blue_cloth.get("saturation_min", 30),
                blue_cloth.get("value_min", 30),
            ]
        )
        blue_upper = np.array(
            [
                blue_cloth.get("hue_max", 130),
                blue_cloth.get("saturation_max", 255),
                blue_cloth.get("value_max", 255),
            ]
        )
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Combine green and blue masks
        cloth_mask = cv2.bitwise_or(green_mask, blue_mask)

        # Morphological operations to clean up and fill small holes
        kernel_large = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel_large)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel_large)

        # Find the largest contour (the playing area)
        contours, _ = cv2.findContours(
            cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("Could not detect playing area - no cloth surface found")
            return None

        # Get the largest contour (the main playing surface)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Validate that the detected area is reasonable
        frame_area = frame.shape[0] * frame.shape[1]
        if area < frame_area * min_area_ratio:  # Too small
            logger.warning(
                f"Detected playing area too small: {area} pixels ({area/frame_area*100:.1f}% of frame)"
            )
            return None

        # Create mask from the largest contour
        playing_area_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(playing_area_mask, [largest_contour], 0, 255, -1)

        # Erode the mask slightly to exclude the rail edges
        # This ensures we don't detect balls that are partially on the rail
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size)
        )
        playing_area_mask = cv2.erode(
            playing_area_mask, kernel_erode, iterations=erode_iterations
        )

        logger.info(
            f"Playing area detected: {cv2.countNonZero(playing_area_mask)} pixels ({cv2.countNonZero(playing_area_mask)/frame_area*100:.1f}% of frame)"
        )

        return playing_area_mask

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set the background reference frame (empty table).

        Args:
            frame: Reference frame of empty table
        """
        self.background_frame = frame.copy()
        self.use_background_subtraction = True

        # Detect pockets from background frame
        self.pocket_locations = self._detect_pockets_from_background(frame)
        logger.info(
            f"Background frame set for ball detection, detected {len(self.pocket_locations)} pockets"
        )

        # Detect playing area (excludes rails and pockets)
        self.playing_area_mask = self._detect_playing_area_from_background(frame)
        if self.playing_area_mask is not None:
            logger.info("Playing area mask detected successfully")
        else:
            logger.warning(
                "Could not detect playing area - ball detection will not be confined to playing surface"
            )

    def detect_balls(
        self, frame: NDArray[np.uint8], table_mask: Optional[NDArray[np.float64]] = None
    ) -> list[Ball]:
        """Detect all balls in frame using configured method.

        Args:
            frame: Input frame in BGR format
            table_mask: Optional mask for table region

        Returns:
            List of detected balls
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Apply background subtraction if enabled
            if self.use_background_subtraction and self.background_frame is not None:
                fg_mask = self._create_foreground_mask(frame)
                # Combine with table mask if provided
                if table_mask is not None:
                    combined_mask = cv2.bitwise_and(fg_mask, table_mask)
                    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
                else:
                    masked_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
            # Apply table mask if provided (and no background subtraction)
            elif table_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=table_mask)
            else:
                masked_frame = frame

            if self.config.debug_mode:
                self.debug_images.append(("input_frame", masked_frame.copy()))

            # Detect ball candidates using selected method
            if self.config.detection_method == DetectionMethod.HOUGH_CIRCLES:
                candidates = self._detect_hough_circles(masked_frame)
            elif self.config.detection_method == DetectionMethod.CONTOUR_BASED:
                candidates = self._detect_contour_based(masked_frame)
            elif self.config.detection_method == DetectionMethod.BLOB_DETECTION:
                candidates = self._detect_blob_based(masked_frame)
            else:  # COMBINED
                candidates = self._detect_combined(masked_frame)

            if self.config.debug_mode:
                self._draw_candidates(masked_frame, candidates)

            # Filter and validate candidates
            valid_balls = self._filter_and_validate(candidates, frame)

            # Classify ball types
            classified_balls = self._classify_balls(valid_balls, frame)

            # Remove overlapping detections
            final_balls = self._remove_overlaps(classified_balls)

            # Resolve ball type conflicts (multiple cue/8-balls)
            final_balls = self._resolve_ball_type_conflicts(final_balls)

            # Update statistics
            self._update_stats(final_balls)

            return final_balls

        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            return []

    def classify_ball_type(
        self,
        ball_region: NDArray[np.float64],
        position: tuple[float, float],
        radius: float,
    ) -> tuple[BallType, float, Optional[int]]:
        """Classify ball type from image region.

        Args:
            ball_region: Cropped image region containing the ball
            position: Ball center position
            radius: Ball radius

        Returns:
            Tuple of (ball_type, confidence, ball_number)
        """
        if ball_region is None or ball_region.size == 0:
            return BallType.UNKNOWN, 0.0, None

        try:
            # Get brightness config for mask ratio
            brightness_config = self.raw_config.get("brightness_filters", {})
            ball_mask_inner_ratio = brightness_config.get("ball_mask_inner_ratio", 0.8)

            # Convert to HSV for color analysis
            hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

            # Create circular mask to focus on ball area
            mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
            center = (ball_region.shape[1] // 2, ball_region.shape[0] // 2)
            cv2.circle(mask, center, int(radius * ball_mask_inner_ratio), 255, -1)

            # Analyze color distribution
            color_scores = {}
            for color_name, color_range in self.config.ball_colors.items():
                color_mask = cv2.inRange(
                    hsv_region, color_range["lower"], color_range["upper"]
                )
                color_mask = cv2.bitwise_and(color_mask, mask)
                score = np.sum(color_mask > 0) / np.sum(mask > 0)
                color_scores[color_name] = score

            # Determine ball type based on color
            best_color = max(color_scores, key=color_scores.get)
            confidence = color_scores[best_color]

            # Map color to ball type and number
            ball_type, ball_number = self._color_to_ball_type(best_color, ball_region)

            return ball_type, confidence, ball_number

        except Exception as e:
            logger.warning(f"Ball classification failed: {e}")
            return BallType.UNKNOWN, 0.0, None

    # Private helper methods

    def _create_ball_color_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Create mask that excludes non-ball colors (table, skin, pockets).

        This is a pre-filter to eliminate obvious false positives BEFORE geometry detection.

        Args:
            frame: Input frame in BGR format

        Returns:
            Binary mask where 255 = possible ball location, 0 = definitely not a ball
        """
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color filtering config
        color_filter_config = self.raw_config.get("color_filtering", {})
        green_table = color_filter_config.get("green_table", {})
        blue_table = color_filter_config.get("blue_table", {})
        skin_tones = color_filter_config.get("skin_tones", {})
        dark_regions = color_filter_config.get("dark_regions", {})
        morph_kernel_size = color_filter_config.get("morph_kernel_size", 5)

        # Create mask that EXCLUDES non-ball regions
        exclude_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 1. Exclude GREEN/BLUE table surface (biggest source of false positives)
        green_table_lower = np.array(
            [
                green_table.get("hue_min", 35),
                green_table.get("saturation_min", 40),
                green_table.get("value_min", 30),
            ]
        )
        green_table_upper = np.array(
            [
                green_table.get("hue_max", 85),
                green_table.get("saturation_max", 255),
                green_table.get("value_max", 255),
            ]
        )
        green_table_mask = cv2.inRange(hsv, green_table_lower, green_table_upper)
        exclude_mask = cv2.bitwise_or(exclude_mask, green_table_mask)

        blue_table_lower = np.array(
            [
                blue_table.get("hue_min", 95),
                blue_table.get("saturation_min", 40),
                blue_table.get("value_min", 30),
            ]
        )
        blue_table_upper = np.array(
            [
                blue_table.get("hue_max", 125),
                blue_table.get("saturation_max", 255),
                blue_table.get("value_max", 255),
            ]
        )
        blue_table_mask = cv2.inRange(hsv, blue_table_lower, blue_table_upper)
        exclude_mask = cv2.bitwise_or(exclude_mask, blue_table_mask)

        # 2. Exclude SKIN TONES (hands, arms)
        skin_lower = np.array(
            [
                skin_tones.get("hue_min", 0),
                skin_tones.get("saturation_min", 20),
                skin_tones.get("value_min", 50),
            ]
        )
        skin_upper = np.array(
            [
                skin_tones.get("hue_max", 20),
                skin_tones.get("saturation_max", 150),
                skin_tones.get("value_max", 255),
            ]
        )
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        exclude_mask = cv2.bitwise_or(exclude_mask, skin_mask)

        # 3. Exclude VERY DARK regions (pockets, shadows)
        dark_lower = np.array(
            [
                dark_regions.get("hue_min", 0),
                dark_regions.get("saturation_min", 0),
                dark_regions.get("value_min", 0),
            ]
        )
        dark_upper = np.array(
            [
                dark_regions.get("hue_max", 180),
                dark_regions.get("saturation_max", 255),
                dark_regions.get("value_max", 30),
            ]
        )
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        exclude_mask = cv2.bitwise_or(exclude_mask, dark_mask)

        # Invert: what's left is POSSIBLE ball locations
        ball_color_mask = cv2.bitwise_not(exclude_mask)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        ball_color_mask = cv2.morphologyEx(ball_color_mask, cv2.MORPH_OPEN, kernel)

        return ball_color_mask

    def _detect_hough_circles(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Detect circles using Hough transform with adaptive thresholding."""
        # FIRST: Apply color pre-filter to eliminate obvious non-balls
        color_mask = self._create_ball_color_mask(frame)
        masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur_kernel = self.config.hough_gaussian_blur_kernel
        blur_sigma = self.config.hough_gaussian_blur_sigma
        # Ensure kernel size is odd
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), blur_sigma)

        # Calculate dynamic parameters
        min_dist = int(self.config.min_radius * 2 * self.config.hough_min_dist_ratio)

        # Adaptive param2 based on image content
        # Key insight: Empty tables need high param2 (strict), tables with balls need lower param2 (lenient)
        # Analyze color mask to determine image context
        scaled_param2 = self._calculate_adaptive_param2(frame, color_mask)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=min_dist,
            param1=self.config.hough_param1,
            param2=scaled_param2,
            minRadius=self.config.min_radius,
            maxRadius=self.config.max_radius,
        )

        candidates = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                candidates.append((float(x), float(y), float(r)))

        self.stats["detection_method_stats"]["hough"] += len(candidates)
        return candidates

    def _calculate_adaptive_param2(
        self, frame: NDArray[np.uint8], color_mask: NDArray[np.uint8]
    ) -> int:
        """Calculate param2 threshold with optimized scaling for high-resolution images.

        For 4K images with 50-65px radius balls, we need param2 ~38-42 to balance:
        - Avoiding false positives on empty tables
        - Detecting real balls effectively

        Args:
            frame: Original frame
            color_mask: Pre-computed color mask (currently not used reliably)

        Returns:
            Scaled param2 value
        """
        base_param2 = self.config.hough_param2

        # Resolution-based scaling using square root to avoid over-scaling
        # This provides gentler adjustment that works across different ball sizes
        image_width = frame.shape[1]
        baseline_width = 1920.0  # 1080p baseline
        scale_factor = math.sqrt(image_width / baseline_width)

        # Apply scaling with a cap to prevent param2 from getting too high
        scaled_param2 = int(base_param2 * scale_factor)
        # Cap at reasonable maximum to ensure we can still detect balls
        max_param2 = base_param2 + 15
        scaled_param2 = min(scaled_param2, max_param2)

        if self.config.debug_mode:
            logger.debug(
                f"Adaptive param2: width={image_width}, "
                f"scale_factor={scale_factor:.2f}, "
                f"base={base_param2}, scaled={scaled_param2}"
            )

        return scaled_param2

    def _detect_contour_based(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Detect circles using contour analysis."""
        # Apply color pre-filter
        color_mask = self._create_ball_color_mask(frame)
        masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Get contour detection config
        contour_config = self.raw_config.get("contour_detection", {})
        block_size = contour_config.get("adaptive_threshold_block_size", 11)
        c_value = contour_config.get("adaptive_threshold_c", 2)

        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            min_area = math.pi * (self.config.min_radius**2)
            max_area = math.pi * (self.config.max_radius**2)

            if min_area <= area <= max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter**2)

                    if circularity >= self.config.min_circularity:
                        # Calculate center and radius
                        M = cv2.moments(contour)
                        if M["m00"] > 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            radius = math.sqrt(area / math.pi)
                            candidates.append((cx, cy, radius))

        self.stats["detection_method_stats"]["contour"] += len(candidates)
        return candidates

    def _detect_blob_based(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Detect circles using blob detection."""
        # Apply color pre-filter
        color_mask = self._create_ball_color_mask(frame)
        masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Invert image for blob detection (blobs are dark on light background)
        inverted = cv2.bitwise_not(gray)

        # Detect blobs
        keypoints = self.blob_detector.detect(inverted)

        candidates = []
        for keypoint in keypoints:
            x, y = keypoint.pt
            radius = keypoint.size / 2

            # Validate radius
            if self.config.min_radius <= radius <= self.config.max_radius:
                candidates.append((x, y, radius))

        self.stats["detection_method_stats"]["blob"] += len(candidates)
        return candidates

    def _detect_combined(
        self, frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Combined detection using multiple methods with optimized merging."""
        # Get candidates from all methods
        hough_candidates = self._detect_hough_circles(frame)
        contour_candidates = self._detect_contour_based(frame)
        blob_candidates = self._detect_blob_based(frame)

        # Combine all candidates
        all_candidates = hough_candidates + contour_candidates + blob_candidates

        if not all_candidates:
            return []

        # Use more sophisticated merging with confidence weighting
        merged_candidates = []
        used_indices = set()

        for i, candidate in enumerate(all_candidates):
            if i in used_indices:
                continue

            # Find all candidates within merge distance
            cluster = [candidate]
            cluster_indices = [i]

            for j, other_candidate in enumerate(all_candidates[i + 1 :], i + 1):
                if j in used_indices:
                    continue

                distance = math.sqrt(
                    (candidate[0] - other_candidate[0]) ** 2
                    + (candidate[1] - other_candidate[1]) ** 2
                )

                # Get merge distance ratio from config
                combined_config = self.raw_config.get("combined_detection", {})
                merge_ratio = combined_config.get("merge_distance_ratio", 0.8)
                merge_distance = max(candidate[2], other_candidate[2]) * merge_ratio

                if distance < merge_distance:
                    cluster.append(other_candidate)
                    cluster_indices.append(j)

            # Mark all cluster candidates as used
            used_indices.update(cluster_indices)

            if len(cluster) == 1:
                merged_candidates.append(candidate)
            else:
                # Merge cluster using weighted average
                total_weight = len(cluster)
                avg_x = sum(c[0] for c in cluster) / total_weight
                avg_y = sum(c[1] for c in cluster) / total_weight

                # Use median radius for better robustness
                radii = sorted([c[2] for c in cluster])
                median_r = radii[len(radii) // 2]

                merged_candidates.append((avg_x, avg_y, median_r))

        self.stats["detection_method_stats"]["combined"] += len(merged_candidates)
        return merged_candidates

    def _is_near_pocket(self, x: float, y: float) -> bool:
        """Check if a detection is near a pocket location.

        Args:
            x, y: Detection center coordinates

        Returns:
            True if detection is near a pocket, False otherwise
        """
        for px, py, pr in self.pocket_locations:
            distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            # Exclude if within pocket exclusion radius
            if distance < pr * self.pocket_exclusion_radius_multiplier:
                return True
        return False

    def _filter_and_validate(
        self, candidates: list[tuple[float, float, float]], frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Filter and validate ball candidates."""
        valid_candidates = []

        for x, y, r in candidates:
            # Check if near pocket (exclude pocket detections)
            if self.pocket_locations and self._is_near_pocket(x, y):
                continue

            # Basic boundary checks
            if (
                r < frame.shape[1]
                and r < frame.shape[0]
                and x - r >= 0
                and x + r < frame.shape[1]
                and y - r >= 0
                and y + r < frame.shape[0]
            ):
                # Radius validation
                expected_r = self.config.expected_radius
                radius_ratio = abs(r - expected_r) / expected_r

                if radius_ratio <= self.config.radius_tolerance:
                    # Additional shadow/darkness filter
                    if self._is_bright_enough(frame, x, y, r):
                        valid_candidates.append((x, y, r))

        # Filter out ball shadows (darker detections near brighter ones)
        valid_candidates = self._filter_ball_shadows(valid_candidates, frame)

        return valid_candidates

    def _is_bright_enough(
        self, frame: NDArray[np.uint8], x: float, y: float, r: float
    ) -> bool:
        """Check if the region is bright enough to be a ball (not a shadow).

        Args:
            frame: Input frame
            x, y: Center coordinates
            r: Radius

        Returns:
            True if bright enough to be a ball, False if likely a shadow
        """
        # Extract region
        x1, y1 = max(0, int(x - r)), max(0, int(y - r))
        x2, y2 = min(frame.shape[1], int(x + r)), min(frame.shape[0], int(y + r))
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return False

        # Convert to HSV for better brightness analysis
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Get brightness filter config
        brightness_config = self.raw_config.get("brightness_filters", {})
        ball_mask_radius_ratio = brightness_config.get("ball_mask_radius_ratio", 0.7)
        min_avg_brightness = brightness_config.get("min_avg_brightness", 40)
        min_max_brightness = brightness_config.get("min_max_brightness", 80)
        eight_ball_min_highlight = brightness_config.get(
            "eight_ball_min_highlight", 150
        )
        eight_ball_max_avg = brightness_config.get("eight_ball_max_avg", 50)

        # Create circular mask for the ball center
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        center = (region.shape[1] // 2, region.shape[0] // 2)
        cv2.circle(mask, center, int(r * ball_mask_radius_ratio), 255, -1)

        # Get average brightness (V channel) and saturation (S channel) in HSV
        masked_v = hsv_region[:, :, 2][mask > 0]
        masked_s = hsv_region[:, :, 1][mask > 0]

        if masked_v.size == 0 or masked_s.size == 0:
            return False

        avg_brightness = np.mean(masked_v)
        np.mean(masked_s)
        max_brightness = np.max(masked_v)

        # Multi-criteria shadow detection:
        # Ball shadows are VERY dark compared to actual balls
        # Even dark colored balls (like the 8-ball) have specular highlights
        #
        # Key insight: Shadows are the DARKEST circular regions on the table
        # Real balls always have some brightness due to lighting
        #
        # The 8-ball is special: it's black but has bright specular highlights
        # Shadows have no bright spots at all

        # Special case for very dark regions (potential 8-ball)
        # If average is dark BUT has bright highlights, it's likely the 8-ball
        if (
            avg_brightness < eight_ball_max_avg
            and max_brightness >= eight_ball_min_highlight
        ):
            # This is likely the 8-ball (dark with strong highlights)
            return True

        # First check: Must have reasonable average brightness OR be dark with highlights
        if avg_brightness < min_avg_brightness:
            return False

        # Second check: Must have at least SOME bright pixels (specular highlights)
        # This is the key filter: shadows have NO bright spots
        if max_brightness < min_max_brightness:
            return False

        return True

    def _filter_ball_shadows(
        self, candidates: list[tuple[float, float, float]], frame: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Filter out ball shadows by removing darker detections near brighter ones.

        Ball shadows are typically:
        - Darker than the ball itself
        - Close to the ball (within 1-2 ball diameters)
        - Similar size to the ball

        Args:
            candidates: List of (x, y, r) tuples
            frame: Input frame

        Returns:
            Filtered list without shadow detections
        """
        if len(candidates) <= 1:
            return candidates

        # Calculate brightness for each candidate
        brightnesses = []
        for x, y, r in candidates:
            x1, y1 = max(0, int(x - r)), max(0, int(y - r))
            x2, y2 = min(frame.shape[1], int(x + r)), min(frame.shape[0], int(y + r))
            region = frame[y1:y2, x1:x2]

            if region.size > 0:
                hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                avg_brightness = np.mean(hsv_region[:, :, 2])
                brightnesses.append(avg_brightness)
            else:
                brightnesses.append(0)

        # Get shadow detection config
        shadow_config = self.raw_config.get("shadow_detection", {})
        max_shadow_dist_mult = shadow_config.get("max_shadow_distance_multiplier", 1.5)
        min_brightness_diff = shadow_config.get("min_brightness_diff", 15)
        min_brightness_ratio = shadow_config.get("min_brightness_ratio", 0.7)

        # Filter out shadows
        filtered = []
        for i, (x, y, r) in enumerate(candidates):
            is_shadow = False

            # Check if this is a shadow of a nearby brighter ball
            for j, (other_x, other_y, other_r) in enumerate(candidates):
                if i == j:
                    continue

                # Calculate distance between candidates
                distance = math.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)

                # Shadows are typically within some distance of the ball
                max_shadow_distance = (r + other_r) * max_shadow_dist_mult

                if distance < max_shadow_distance:
                    # If this candidate is significantly darker than the nearby one,
                    # it's likely a shadow
                    brightness_diff = brightnesses[j] - brightnesses[i]

                    # Shadow detection using brightness diff and ratio
                    brightness_ratio = brightnesses[i] / (brightnesses[j] + 1)

                    if (
                        brightness_diff > min_brightness_diff
                        or brightness_ratio < min_brightness_ratio
                    ):
                        is_shadow = True
                        break

            if not is_shadow:
                filtered.append((x, y, r))

        return filtered

    def _classify_balls(
        self, candidates: list[tuple[float, float, float]], frame: NDArray[np.uint8]
    ) -> list[Ball]:
        """Classify ball candidates."""
        balls = []

        # Get brightness config for region expansion
        brightness_config = self.raw_config.get("brightness_filters", {})
        region_expansion_ratio = brightness_config.get("region_expansion_ratio", 1.2)

        for x, y, r in candidates:
            # Extract ball region
            x1, y1 = max(0, int(x - r * region_expansion_ratio)), max(
                0, int(y - r * region_expansion_ratio)
            )
            x2, y2 = min(frame.shape[1], int(x + r * region_expansion_ratio)), min(
                frame.shape[0], int(y + r * region_expansion_ratio)
            )

            ball_region = frame[y1:y2, x1:x2]

            if ball_region.size > 0:
                ball_type, confidence, ball_number = self.classify_ball_type(
                    ball_region, (x, y), r
                )

                if confidence >= self.config.min_confidence:
                    ball = Ball(
                        position=(x, y),
                        radius=r,
                        ball_type=ball_type,
                        number=ball_number,
                        confidence=confidence,
                        velocity=(0.0, 0.0),  # Will be calculated by tracking
                        is_moving=False,  # Will be determined by tracking
                    )
                    balls.append(ball)

        return balls

    def _remove_overlaps(self, balls: list[Ball]) -> list[Ball]:
        """Remove overlapping ball detections, aggressively merging extreme overlaps.

        Strategy:
        - Extreme overlap (>70% center distance < 0.5 radii sum): Merge into one ball
        - High overlap (50-70%): Remove lower confidence
        - Touching balls (centers ~2 radii apart): Keep both
        """
        if len(balls) <= 1:
            return balls

        # Get overlap merging config
        overlap_config = self.raw_config.get("overlap_merging", {})
        extreme_overlap_ratio = overlap_config.get("extreme_overlap_ratio", 0.5)

        # Sort by confidence (highest first)
        sorted_balls = sorted(balls, key=lambda b: b.confidence, reverse=True)

        final_balls = []
        merged_indices = set()

        for i, ball in enumerate(sorted_balls):
            if i in merged_indices:
                continue

            # Check for extreme overlaps to merge
            balls_to_merge = [ball]
            merge_indices = [i]

            for j, other_ball in enumerate(sorted_balls[i + 1 :], i + 1):
                if j in merged_indices:
                    continue

                distance = math.sqrt(
                    (ball.position[0] - other_ball.position[0]) ** 2
                    + (ball.position[1] - other_ball.position[1]) ** 2
                )

                avg_radius = (ball.radius + other_ball.radius) / 2

                # Extreme overlap: centers very close
                # This catches ghost detections around the same ball
                if distance < avg_radius * extreme_overlap_ratio:
                    balls_to_merge.append(other_ball)
                    merge_indices.append(j)
                    merged_indices.add(j)

            # If multiple balls detected at same location, merge them
            if len(balls_to_merge) > 1:
                # Merge by averaging positions weighted by confidence
                total_confidence = sum(b.confidence for b in balls_to_merge)
                merged_x = (
                    sum(b.position[0] * b.confidence for b in balls_to_merge)
                    / total_confidence
                )
                merged_y = (
                    sum(b.position[1] * b.confidence for b in balls_to_merge)
                    / total_confidence
                )
                merged_r = (
                    sum(b.radius * b.confidence for b in balls_to_merge)
                    / total_confidence
                )
                merged_confidence = total_confidence / len(balls_to_merge)

                # Create merged ball with highest confidence ball's type
                merged_ball = Ball(
                    position=(merged_x, merged_y),
                    radius=merged_r,
                    ball_type=ball.ball_type,
                    number=ball.number,
                    confidence=merged_confidence,
                    velocity=(0.0, 0.0),
                    is_moving=False,
                )
                final_balls.append(merged_ball)
            else:
                # Single ball, check for regular overlaps
                is_overlap = False
                for existing_ball in final_balls:
                    distance = math.sqrt(
                        (ball.position[0] - existing_ball.position[0]) ** 2
                        + (ball.position[1] - existing_ball.position[1]) ** 2
                    )

                    # Standard overlap check (for moderate overlaps)
                    min_distance = (ball.radius + existing_ball.radius) * (
                        1 - self.config.max_overlap_ratio
                    )

                    if distance < min_distance:
                        is_overlap = True
                        break

                if not is_overlap:
                    final_balls.append(ball)

        return final_balls

    def _resolve_ball_type_conflicts(self, balls: list[Ball]) -> list[Ball]:
        """Resolve conflicts when multiple cue or 8-balls are detected.

        Rule: Only one cue ball and one 8-ball allowed on table.
        If multiple detected, highest confidence wins, others are reclassified:
        - Extra cue balls → OTHER
        - Extra 8-balls → OTHER

        Args:
            balls: List of detected balls

        Returns:
            List of balls with conflicts resolved
        """
        if len(balls) <= 1:
            return balls

        # Find all cue balls
        cue_balls = [ball for ball in balls if ball.ball_type == BallType.CUE]

        # If multiple cue balls, keep highest confidence, convert others to OTHER
        if len(cue_balls) > 1:
            # Sort by confidence (highest first)
            cue_balls.sort(key=lambda b: b.confidence, reverse=True)
            best_cue = cue_balls[0]

            logger.info(
                f"Multiple cue balls detected ({len(cue_balls)}), keeping highest confidence "
                f"({best_cue.confidence:.3f}), converting others to OTHER"
            )

            # Convert all but the best to OTHER
            for ball in cue_balls[1:]:
                ball.ball_type = BallType.OTHER
                ball.number = None
                logger.debug(
                    f"Converting cue ball at ({ball.position[0]:.1f}, {ball.position[1]:.1f}) "
                    f"with confidence {ball.confidence:.3f} to OTHER"
                )

        # Find all 8-balls
        eight_balls = [ball for ball in balls if ball.ball_type == BallType.EIGHT]

        # If multiple 8-balls, keep highest confidence, convert others to OTHER
        if len(eight_balls) > 1:
            # Sort by confidence (highest first)
            eight_balls.sort(key=lambda b: b.confidence, reverse=True)
            best_eight = eight_balls[0]

            logger.info(
                f"Multiple 8-balls detected ({len(eight_balls)}), keeping highest confidence "
                f"({best_eight.confidence:.3f}), converting others to OTHER"
            )

            # Convert all but the best to OTHER
            for ball in eight_balls[1:]:
                ball.ball_type = BallType.OTHER
                ball.number = None
                logger.debug(
                    f"Converting 8-ball at ({ball.position[0]:.1f}, {ball.position[1]:.1f}) "
                    f"with confidence {ball.confidence:.3f} to OTHER"
                )

        return balls

    def _color_to_ball_type(
        self, color_name: str, ball_region: NDArray[np.float64]
    ) -> tuple[BallType, Optional[int]]:
        """Map detected color to ball type (simplified - no stripe/solid distinction).

        Returns only three ball types: CUE, EIGHT, OTHER.
        Ball numbers are no longer detected or returned.
        Stripe/solid classification removed due to poor accuracy.
        """
        # Map color to base ball type
        color_type_map = {
            "cue": BallType.CUE,
            "black": BallType.EIGHT,
            # All colored balls are classified as OTHER (no stripe/solid distinction)
            "yellow": BallType.OTHER,
            "blue": BallType.OTHER,
            "red": BallType.OTHER,
            "purple": BallType.OTHER,
            "orange": BallType.OTHER,
            "green": BallType.OTHER,
            "maroon": BallType.OTHER,
        }

        if color_name not in color_type_map:
            return BallType.UNKNOWN, None

        ball_type = color_type_map[color_name]

        return ball_type, None

    # NOTE: Stripe detection disabled - stripe/solid classification removed due to poor accuracy
    # The method below is preserved for reference but is no longer called
    #
    # def _is_striped_ball(self, ball_region: NDArray[np.float64]) -> bool:
    #     """DEPRECATED: Stripe detection disabled due to unreliable classification.
    #
    #     This method previously detected if a ball had stripe pattern using multi-method
    #     pattern recognition. However, stripe/solid classification proved too sensitive to:
    #     - Lighting conditions
    #     - Ball orientation
    #     - Image quality
    #     - Shadows and reflections
    #
    #     All colored balls are now classified as BallType.OTHER.
    #     """
    #     return False

    def _draw_candidates(
        self, frame: NDArray[np.uint8], candidates: list[tuple[float, float, float]]
    ):
        """Draw detected candidates for debugging."""
        debug_frame = frame.copy()

        for x, y, r in candidates:
            cv2.circle(debug_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(debug_frame, (int(x), int(y)), 2, (0, 0, 255), 3)

        self.debug_images.append(("candidates", debug_frame))

    def _update_stats(self, balls: list[Ball]) -> None:
        """Update detection statistics."""
        self.stats["total_detections"] += 1
        if balls:
            self.stats["successful_detections"] += 1
            avg_conf = sum(ball.confidence for ball in balls) / len(balls)
            self.stats["avg_confidence"] = (
                self.stats["avg_confidence"] * (self.stats["successful_detections"] - 1)
                + avg_conf
            ) / self.stats["successful_detections"]

    def get_debug_images(self) -> list[tuple[str, np.ndarray]]:
        """Get debug images."""
        return self.debug_images

    def clear_debug_images(self) -> None:
        """Clear debug image storage."""
        self.debug_images.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()
