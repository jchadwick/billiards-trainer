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

    # Hough circle parameters - optimized for better detection
    hough_dp: float = 1.0
    hough_min_dist_ratio: float = 0.6  # Reduced to allow closer balls
    hough_param1: int = 30  # Reduced for more sensitive detection
    hough_param2: int = 20  # Reduced threshold for more detections
    hough_accumulator_threshold: int = 10

    # Size constraints - more tolerant
    min_radius: int = 12
    max_radius: int = 30
    expected_radius: int = 20  # Expected ball radius for validation
    radius_tolerance: float = 0.5  # Increased to ±50% radius tolerance

    # Color classification
    ball_colors: dict[str, dict[str, tuple[int, int, int]]] = None

    # Quality filters - more permissive for higher detection rate
    min_circularity: float = 0.5  # Reduced for more detections
    min_confidence: float = 0.2  # Lowered threshold
    max_overlap_ratio: float = 0.4  # Slightly more permissive

    # Performance optimization
    roi_enabled: bool = True
    roi_margin: int = 50  # Margin around table boundaries

    # Debug settings
    debug_mode: bool = False
    save_debug_images: bool = False

    def __post_init__(self):
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
                    "lower": (155, 80, 40),  # Maroon (7, 15) - relaxed
                    "upper": (180, 255, 180),
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

    def __init__(self, config: dict[str, Any]):
        """Initialize ball detector with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = BallDetectionConfig(**config)

        # Initialize detection components
        self._initialize_detectors()

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

    def _initialize_detectors(self):
        """Initialize detection algorithm components."""
        # Blob detector for alternative detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = math.pi * (self.config.min_radius**2)
        params.maxArea = math.pi * (self.config.max_radius**2)
        params.filterByCircularity = True
        params.minCircularity = self.config.min_circularity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        self.blob_detector = cv2.SimpleBlobDetector_create(params)

        logger.debug("Detection algorithms initialized")

    def detect_balls(
        self, frame: np.ndarray, table_mask: Optional[np.ndarray] = None
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
            # Apply table mask if provided
            if table_mask is not None:
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

            # Update statistics
            self._update_stats(final_balls)

            return final_balls

        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            return []

    def classify_ball_type(
        self, ball_region: np.ndarray, position: tuple[float, float], radius: float
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
            # Convert to HSV for color analysis
            hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

            # Create circular mask to focus on ball area
            mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
            center = (ball_region.shape[1] // 2, ball_region.shape[0] // 2)
            cv2.circle(mask, center, int(radius * 0.8), 255, -1)

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

    def identify_ball_number(
        self, ball_region: np.ndarray, ball_type: BallType
    ) -> Optional[int]:
        """Identify ball number if visible using pattern recognition.

        Args:
            ball_region: Cropped image region containing the ball
            ball_type: Previously classified ball type

        Returns:
            Ball number (1-15) or None if not identifiable
        """
        # This is a simplified implementation
        # A full implementation would use OCR or template matching
        # to identify numbers and stripe patterns

        if ball_type == BallType.CUE:
            return None  # Cue ball has no number
        elif ball_type == BallType.EIGHT:
            return 8
        else:
            # For now, return None - would need advanced pattern recognition
            # to distinguish between solid/stripe numbers of same color
            return None

    # Private helper methods

    def _detect_hough_circles(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """Detect circles using Hough transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Calculate dynamic parameters
        min_dist = int(self.config.min_radius * 2 * self.config.hough_min_dist_ratio)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.config.hough_dp,
            minDist=min_dist,
            param1=self.config.hough_param1,
            param2=self.config.hough_param2,
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

    def _detect_contour_based(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """Detect circles using contour analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
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

    def _detect_blob_based(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Detect circles using blob detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    def _detect_combined(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
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

                # More generous merge distance for better coverage
                merge_distance = max(candidate[2], other_candidate[2]) * 0.8

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

    def _filter_and_validate(
        self, candidates: list[tuple[float, float, float]], frame: np.ndarray
    ) -> list[tuple[float, float, float]]:
        """Filter and validate ball candidates."""
        valid_candidates = []

        for x, y, r in candidates:
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
                    valid_candidates.append((x, y, r))

        return valid_candidates

    def _classify_balls(
        self, candidates: list[tuple[float, float, float]], frame: np.ndarray
    ) -> list[Ball]:
        """Classify ball candidates."""
        balls = []

        for x, y, r in candidates:
            # Extract ball region
            x1, y1 = max(0, int(x - r * 1.2)), max(0, int(y - r * 1.2))
            x2, y2 = min(frame.shape[1], int(x + r * 1.2)), min(
                frame.shape[0], int(y + r * 1.2)
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
        """Remove overlapping ball detections, keeping the best ones."""
        if len(balls) <= 1:
            return balls

        # Sort by confidence (highest first)
        sorted_balls = sorted(balls, key=lambda b: b.confidence, reverse=True)

        final_balls = []
        for ball in sorted_balls:
            is_overlap = False

            for existing_ball in final_balls:
                distance = math.sqrt(
                    (ball.position[0] - existing_ball.position[0]) ** 2
                    + (ball.position[1] - existing_ball.position[1]) ** 2
                )

                min_distance = (ball.radius + existing_ball.radius) * (
                    1 - self.config.max_overlap_ratio
                )

                if distance < min_distance:
                    is_overlap = True
                    break

            if not is_overlap:
                final_balls.append(ball)

        return final_balls

    def _color_to_ball_type(
        self, color_name: str, ball_region: np.ndarray
    ) -> tuple[BallType, Optional[int]]:
        """Map detected color to ball type and number."""
        color_map = {
            "cue": (BallType.CUE, None),
            "black": (BallType.EIGHT, 8),
            "yellow": (BallType.SOLID, 1),  # Could also be 9-stripe
            "blue": (BallType.SOLID, 2),  # Could also be 10-stripe
            "red": (BallType.SOLID, 3),  # Could also be 11-stripe
            "purple": (BallType.SOLID, 4),  # Could also be 12-stripe
            "orange": (BallType.SOLID, 5),  # Could also be 13-stripe
            "green": (BallType.SOLID, 6),  # Could also be 14-stripe
            "maroon": (BallType.SOLID, 7),  # Could also be 15-stripe
        }

        if color_name in color_map:
            ball_type, number = color_map[color_name]

            # For colored balls, try to determine if it's solid or stripe
            if ball_type == BallType.SOLID and number is not None and number <= 7:
                if self._is_striped_ball(ball_region):
                    return BallType.STRIPE, number + 8

            return ball_type, number

        return BallType.UNKNOWN, None

    def _is_striped_ball(self, ball_region: np.ndarray) -> bool:
        """Detect if a ball has stripe pattern (simplified implementation)."""
        # This is a placeholder - would need sophisticated pattern recognition
        # to reliably distinguish stripes from solids
        gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)

        # Calculate variance - striped balls typically have higher variance
        variance = np.var(gray)

        # Threshold determined empirically (would need calibration)
        return variance > 1000

    def _draw_candidates(
        self, frame: np.ndarray, candidates: list[tuple[float, float, float]]
    ):
        """Draw detected candidates for debugging."""
        debug_frame = frame.copy()

        for x, y, r in candidates:
            cv2.circle(debug_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(debug_frame, (int(x), int(y)), 2, (0, 0, 255), 3)

        self.debug_images.append(("candidates", debug_frame))

    def _update_stats(self, balls: list[Ball]):
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

    def clear_debug_images(self):
        """Clear debug image storage."""
        self.debug_images.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()
