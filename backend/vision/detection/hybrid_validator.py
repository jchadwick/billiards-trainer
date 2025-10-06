"""Hybrid YOLO + Classical CV validation and refinement.

Provides ball position refinement using classical computer vision techniques
to achieve sub-pixel accuracy from YOLO detections.

Features:
- Sub-pixel position refinement using Hough circles
- Edge-based center detection with cornerSubPix
- Confidence validation and fallback handling
- ROI extraction and preprocessing

Implements requirements:
- FR-VIS-023: Track ball positions with ±2 pixel accuracy
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from ..models import Ball, BallType

logger = logging.getLogger(__name__)


@dataclass
class BallFeatures:
    """Extracted features from a detected ball.

    Contains detailed feature information for ball classification,
    validation, and tracking consistency.
    """

    # Color features
    dominant_color_hsv: tuple[int, int, int]  # Dominant color in HSV
    color_histogram: NDArray[np.float64]  # HSV histogram (normalized)
    color_variance: float  # Color variance indicator

    # Pattern features
    has_stripe: bool  # Whether stripe pattern detected
    stripe_confidence: float  # Confidence of stripe detection (0.0-1.0)
    edge_density: float  # Edge density metric
    pattern_score: float  # Overall pattern complexity score

    # Number recognition
    detected_number: Optional[int]  # Recognized ball number (1-15)
    number_confidence: float  # Confidence of number recognition (0.0-1.0)

    # Quality metrics
    brightness: float  # Average brightness (V channel)
    saturation: float  # Average saturation (S channel)
    sharpness: float  # Sharpness metric (Laplacian variance)

    # Additional metadata
    ball_type: BallType  # Classified ball type
    extraction_quality: float  # Overall feature extraction quality (0.0-1.0)


class BallPositionRefiner:
    """Refines YOLO ball detections using classical CV techniques for sub-pixel accuracy."""

    def __init__(
        self,
        hough_dp: float = 1.2,
        hough_param1: int = 50,
        hough_param2: int = 30,
        hough_min_dist_ratio: float = 0.8,
        subpix_window_size: int = 5,
        subpix_zero_zone: int = -1,
        subpix_criteria_max_iter: int = 30,
        subpix_criteria_epsilon: float = 0.01,
        max_refinement_distance: float = 5.0,
        min_confidence_for_refinement: float = 0.3,
    ):
        """Initialize the ball position refiner.

        Args:
            hough_dp: Inverse ratio of accumulator resolution to image resolution
            hough_param1: Higher threshold for Canny edge detector
            hough_param2: Accumulator threshold for circle centers
            hough_min_dist_ratio: Minimum distance ratio between circles (relative to radius)
            subpix_window_size: Half of the side length of the search window for cornerSubPix
            subpix_zero_zone: Half of the dead zone size (usually -1)
            subpix_criteria_max_iter: Maximum number of iterations for cornerSubPix
            subpix_criteria_epsilon: Desired accuracy for cornerSubPix
            max_refinement_distance: Maximum allowed distance between YOLO and refined position
            min_confidence_for_refinement: Minimum YOLO confidence to attempt refinement
        """
        self.hough_dp = hough_dp
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.hough_min_dist_ratio = hough_min_dist_ratio
        self.subpix_window_size = subpix_window_size
        self.subpix_zero_zone = subpix_zero_zone
        self.subpix_criteria_max_iter = subpix_criteria_max_iter
        self.subpix_criteria_epsilon = subpix_criteria_epsilon
        self.max_refinement_distance = max_refinement_distance
        self.min_confidence_for_refinement = min_confidence_for_refinement

        # Statistics tracking
        self.stats = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "failed_refinements": 0,
            "fallback_to_yolo": 0,
            "avg_refinement_distance": 0.0,
        }

        logger.info("Ball position refiner initialized")

    def refine_ball_position(self, yolo_ball: Ball, frame: NDArray[np.uint8]) -> Ball:
        """Refine ball position using classical CV techniques for sub-pixel accuracy.

        This method extracts an ROI around the YOLO detection, applies Hough circle
        detection for precise center localization, and optionally uses sub-pixel
        edge detection for even greater accuracy. Falls back to YOLO position if
        refinement fails.

        Args:
            yolo_ball: Ball detected by YOLO with initial position and radius
            frame: Input frame in BGR format

        Returns:
            Ball with refined position (or original position if refinement fails)
        """
        self.stats["total_refinements"] += 1

        # Skip refinement for low-confidence detections
        if yolo_ball.confidence < self.min_confidence_for_refinement:
            logger.debug(
                f"Skipping refinement for low confidence ball: {yolo_ball.confidence:.2f}"
            )
            self.stats["fallback_to_yolo"] += 1
            return yolo_ball

        try:
            # Extract ROI around YOLO detection
            roi, roi_offset = self._extract_roi(yolo_ball, frame)
            if roi is None or roi.size == 0:
                logger.debug("Failed to extract ROI, using YOLO position")
                self.stats["failed_refinements"] += 1
                return yolo_ball

            # Apply preprocessing to enhance circle detection
            preprocessed = self._preprocess_roi(roi)

            # Attempt Hough circle detection for precise center
            refined_center, refined_radius = self._detect_precise_circle(
                preprocessed, yolo_ball.radius
            )

            if refined_center is None:
                logger.debug("Hough circle detection failed, trying edge-based method")
                # Fallback to edge-based center detection
                refined_center, refined_radius = self._detect_center_from_edges(
                    preprocessed, yolo_ball.radius
                )

            if refined_center is None:
                logger.debug("All refinement methods failed, using YOLO position")
                self.stats["failed_refinements"] += 1
                return yolo_ball

            # Convert ROI coordinates back to frame coordinates
            refined_x = refined_center[0] + roi_offset[0]
            refined_y = refined_center[1] + roi_offset[1]

            # Validate refined position is not too far from YOLO detection
            distance = math.sqrt(
                (refined_x - yolo_ball.position[0]) ** 2
                + (refined_y - yolo_ball.position[1]) ** 2
            )

            if distance > self.max_refinement_distance:
                logger.debug(
                    f"Refined position too far from YOLO ({distance:.2f} px), using YOLO position"
                )
                self.stats["failed_refinements"] += 1
                return yolo_ball

            # Apply sub-pixel refinement for even greater accuracy
            subpix_center = self._refine_subpixel(
                preprocessed, refined_center, roi_offset
            )

            if subpix_center is not None:
                refined_x, refined_y = subpix_center

            # Create refined ball with updated position
            refined_ball = Ball(
                position=(refined_x, refined_y),
                radius=(
                    refined_radius if refined_radius is not None else yolo_ball.radius
                ),
                ball_type=yolo_ball.ball_type,
                number=yolo_ball.number,
                confidence=yolo_ball.confidence,
                velocity=yolo_ball.velocity,
                acceleration=yolo_ball.acceleration,
                is_moving=yolo_ball.is_moving,
                track_id=yolo_ball.track_id,
                last_seen=yolo_ball.last_seen,
                age=yolo_ball.age,
                hit_count=yolo_ball.hit_count,
                color_hsv=yolo_ball.color_hsv,
                occlusion_state=yolo_ball.occlusion_state,
                predicted_position=yolo_ball.predicted_position,
                position_history=yolo_ball.position_history.copy(),
            )

            # Update statistics
            self.stats["successful_refinements"] += 1
            avg_dist = self.stats["avg_refinement_distance"]
            count = self.stats["successful_refinements"]
            self.stats["avg_refinement_distance"] = (
                avg_dist * (count - 1) + distance
            ) / count

            logger.debug(
                f"Ball position refined: ({yolo_ball.position[0]:.1f}, {yolo_ball.position[1]:.1f}) -> "
                f"({refined_x:.2f}, {refined_y:.2f}), distance: {distance:.2f} px"
            )

            return refined_ball

        except Exception as e:
            logger.warning(f"Ball position refinement failed: {e}")
            self.stats["failed_refinements"] += 1
            return yolo_ball

    def _extract_roi(
        self, ball: Ball, frame: NDArray[np.uint8]
    ) -> tuple[Optional[NDArray[np.uint8]], tuple[int, int]]:
        """Extract region of interest around ball detection.

        Args:
            ball: Ball with position and radius
            frame: Full frame

        Returns:
            Tuple of (ROI image, (x_offset, y_offset)) or (None, (0, 0)) if extraction fails
        """
        # Calculate ROI bounds with margin (2x radius for good context)
        margin = int(ball.radius * 2)
        x, y = ball.position
        x1 = max(0, int(x - margin))
        y1 = max(0, int(y - margin))
        x2 = min(frame.shape[1], int(x + margin))
        y2 = min(frame.shape[0], int(y + margin))

        # Validate ROI size
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        # Validate ROI is large enough
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return None, (0, 0)

        return roi, (x1, y1)

    def _preprocess_roi(self, roi: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Preprocess ROI to enhance circle detection.

        Args:
            roi: ROI image in BGR format

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Apply bilateral filter to reduce noise while preserving edges
        # This is crucial for accurate circle detection
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Optional: Apply CLAHE for better contrast in varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        return enhanced

    def _detect_precise_circle(
        self, preprocessed: NDArray[np.uint8], expected_radius: float
    ) -> tuple[Optional[tuple[float, float]], Optional[float]]:
        """Detect precise circle center using Hough transform.

        Args:
            preprocessed: Preprocessed grayscale ROI
            expected_radius: Expected ball radius from YOLO

        Returns:
            Tuple of ((x, y), radius) or (None, None) if detection fails
        """
        # Calculate search parameters based on expected radius
        min_radius = max(int(expected_radius * 0.7), 5)
        max_radius = int(expected_radius * 1.3)
        min_dist = int(expected_radius * self.hough_min_dist_ratio)

        # Apply Hough circle detection with precise parameters
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None or len(circles[0]) == 0:
            return None, None

        # Get the best circle (Hough circles are already sorted by quality)
        best_circle = circles[0][0]
        center = (float(best_circle[0]), float(best_circle[1]))
        radius = float(best_circle[2])

        return center, radius

    def _detect_center_from_edges(
        self, preprocessed: NDArray[np.uint8], expected_radius: float
    ) -> tuple[Optional[tuple[float, float]], Optional[float]]:
        """Detect ball center from edge analysis when Hough circles fail.

        Uses edge detection and contour analysis to find the ball center.

        Args:
            preprocessed: Preprocessed grayscale ROI
            expected_radius: Expected ball radius from YOLO

        Returns:
            Tuple of ((x, y), radius) or (None, None) if detection fails
        """
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed, self.hough_param1 // 2, self.hough_param1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None

        # Find the most circular contour near expected size
        best_contour = None
        best_circularity = 0.0
        best_radius = None

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by expected area
            expected_area = math.pi * (expected_radius**2)
            if area < expected_area * 0.5 or area > expected_area * 2.0:
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter**2)

                if circularity > best_circularity:
                    best_circularity = circularity
                    best_contour = contour
                    best_radius = math.sqrt(area / math.pi)

        if best_contour is None or best_circularity < 0.6:
            return None, None

        # Calculate center from moments
        moments = cv2.moments(best_contour)
        if moments["m00"] == 0:
            return None, None

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        return (cx, cy), best_radius

    def _refine_subpixel(
        self,
        preprocessed: NDArray[np.uint8],
        center: tuple[float, float],
        roi_offset: tuple[int, int],
    ) -> Optional[tuple[float, float]]:
        """Refine center position to sub-pixel accuracy using cornerSubPix.

        Args:
            preprocessed: Preprocessed grayscale ROI
            center: Initial center position in ROI coordinates
            roi_offset: ROI offset in frame coordinates

        Returns:
            Refined center in frame coordinates or None if refinement fails
        """
        try:
            # Prepare corner point (must be float32)
            corners = np.array([[center]], dtype=np.float32)

            # Define termination criteria
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                self.subpix_criteria_max_iter,
                self.subpix_criteria_epsilon,
            )

            # Apply sub-pixel refinement
            # Note: cornerSubPix works best on corner-like features, but can also
            # refine circular centers by finding the edge transition points
            refined_corners = cv2.cornerSubPix(
                preprocessed,
                corners,
                (self.subpix_window_size, self.subpix_window_size),
                (self.subpix_zero_zone, self.subpix_zero_zone),
                criteria,
            )

            if refined_corners is None or len(refined_corners) == 0:
                return None

            # Convert back to frame coordinates
            refined_x = refined_corners[0][0][0] + roi_offset[0]
            refined_y = refined_corners[0][0][1] + roi_offset[1]

            return (refined_x, refined_y)

        except Exception as e:
            logger.debug(f"Sub-pixel refinement failed: {e}")
            return None

    def get_statistics(self) -> dict:
        """Get refinement statistics.

        Returns:
            Dictionary of refinement statistics
        """
        stats = self.stats.copy()

        # Calculate success rate
        if stats["total_refinements"] > 0:
            stats["success_rate"] = (
                stats["successful_refinements"] / stats["total_refinements"]
            )
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset refinement statistics."""
        self.stats = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "failed_refinements": 0,
            "fallback_to_yolo": 0,
            "avg_refinement_distance": 0.0,
        }


# Convenience function for one-off refinements
def refine_ball_position(yolo_ball: Ball, frame: NDArray[np.uint8]) -> Ball:
    """Convenience function to refine a single ball position.

    Args:
        yolo_ball: Ball detected by YOLO
        frame: Input frame in BGR format

    Returns:
        Ball with refined position
    """
    refiner = BallPositionRefiner()
    return refiner.refine_ball_position(yolo_ball, frame)


# =============================================================================
# Ball Feature Extraction
# =============================================================================


def extract_ball_features(
    ball: Ball, frame: NDArray[np.uint8], debug: bool = False
) -> BallFeatures:
    """Extract comprehensive features from a detected ball.

    Performs multi-stage feature extraction including:
    1. Color histogram analysis in HSV space
    2. Stripe pattern detection using edge detection and frequency analysis
    3. Ball number recognition using contour analysis

    Reuses stripe detection and number recognition logic from balls.py.

    Args:
        ball: Detected ball object with position, radius, and type
        frame: Original BGR frame containing the ball
        debug: Enable debug logging

    Returns:
        BallFeatures dataclass with all extracted features
    """
    try:
        # Extract ball region with padding for better feature extraction
        x, y = ball.position
        r = ball.radius

        # Add 20% padding
        padding = int(r * 0.2)
        x1 = max(0, int(x - r - padding))
        y1 = max(0, int(y - r - padding))
        x2 = min(frame.shape[1], int(x + r + padding))
        y2 = min(frame.shape[0], int(y + r + padding))

        ball_region = frame[y1:y2, x1:x2]

        if ball_region.size == 0:
            raise ValueError("Ball region is empty")

        # Create circular mask to focus on ball area
        mask = _create_circular_mask(ball_region, r, padding)

        # Extract color features
        dominant_color, color_hist, color_var, brightness, saturation = (
            _extract_color_features(ball_region, mask, debug)
        )

        # Extract pattern features (stripe detection)
        has_stripe, stripe_conf, edge_dens, pattern_sc = _extract_pattern_features(
            ball_region, mask, r, debug
        )

        # Extract number recognition features
        detected_num, num_conf = _extract_number_features(
            ball_region, mask, ball.ball_type, debug
        )

        # Calculate sharpness
        sharpness = _calculate_sharpness(ball_region, mask)

        # Calculate overall extraction quality
        extraction_quality = _calculate_extraction_quality(
            ball_region,
            mask,
            brightness,
            sharpness,
            stripe_conf if has_stripe else 1.0 - stripe_conf,
        )

        return BallFeatures(
            dominant_color_hsv=dominant_color,
            color_histogram=color_hist,
            color_variance=color_var,
            has_stripe=has_stripe,
            stripe_confidence=stripe_conf,
            edge_density=edge_dens,
            pattern_score=pattern_sc,
            detected_number=detected_num,
            number_confidence=num_conf,
            brightness=brightness,
            saturation=saturation,
            sharpness=sharpness,
            ball_type=ball.ball_type,
            extraction_quality=extraction_quality,
        )

    except Exception as e:
        logger.error(f"Ball feature extraction failed: {e}")
        return _get_default_features(ball.ball_type)


def _create_circular_mask(
    ball_region: NDArray[np.uint8], radius: float, padding: int
) -> NDArray[np.uint8]:
    """Create circular mask for ball region."""
    mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
    center = (ball_region.shape[1] // 2, ball_region.shape[0] // 2)
    cv2.circle(mask, center, int(radius * 0.9), 255, -1)
    return mask


def _extract_color_features(
    ball_region: NDArray[np.uint8], mask: NDArray[np.uint8], debug: bool = False
) -> tuple[tuple[int, int, int], NDArray[np.float64], float, float, float]:
    """Extract color-based features from ball region."""
    hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)

    # Calculate histograms
    h_hist = cv2.calcHist([hsv_region], [0], mask, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_region], [1], mask, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_region], [2], mask, [256], [0, 256])

    # Normalize
    h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-6)
    s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-6)
    v_hist = v_hist.flatten() / (np.sum(v_hist) + 1e-6)

    color_histogram = np.concatenate([h_hist, s_hist, v_hist])

    dominant_hue = int(np.argmax(h_hist))

    masked_hsv = hsv_region[mask > 0]
    if masked_hsv.size == 0:
        return (0, 0, 0), color_histogram, 0.0, 0.0, 0.0

    avg_saturation = float(np.mean(masked_hsv[:, 1]))
    avg_brightness = float(np.mean(masked_hsv[:, 2]))

    dominant_color = (
        dominant_hue,
        int(np.median(masked_hsv[:, 1])),
        int(np.median(masked_hsv[:, 2])),
    )

    color_variance = float(np.var(masked_hsv[:, 0]))

    if debug:
        logger.debug(
            f"Color features - Dominant: {dominant_color}, "
            f"Variance: {color_variance:.1f}, "
            f"Brightness: {avg_brightness:.1f}, "
            f"Saturation: {avg_saturation:.1f}"
        )

    return (
        dominant_color,
        color_histogram,
        color_variance,
        avg_brightness,
        avg_saturation,
    )


def _extract_pattern_features(
    ball_region: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    radius: float,
    debug: bool = False,
) -> tuple[bool, float, float, float]:
    """Extract pattern features for stripe detection using multi-method approach."""
    gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Method 1: Variance Check
    variance = np.var(masked_gray[mask > 0])
    variance_score = 1.0 if variance > 800 else 0.0

    # Method 2: Edge Detection
    blurred = cv2.bilateralFilter(masked_gray, 5, 50, 50)
    median_intensity = np.median(blurred[mask > 0])
    lower_thresh = int(max(0, 0.66 * median_intensity))
    upper_thresh = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(blurred, lower_thresh, upper_thresh)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)

    edge_pixels = np.count_nonzero(edges_masked)
    total_pixels = np.count_nonzero(mask)
    edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0.0
    edge_score = 1.0 if 0.10 <= edge_density <= 0.35 else 0.0

    # Method 3: Frequency Analysis
    h_projection = np.sum(edges_masked, axis=0)
    v_projection = np.sum(edges_masked, axis=1)
    h_std = np.std(h_projection) / (np.mean(h_projection) + 1e-6)
    v_std = np.std(v_projection) / (np.mean(v_projection) + 1e-6)
    max_projection_std = max(h_std, v_std)
    frequency_score = 1.0 if max_projection_std > 1.5 else 0.0

    # Method 4: Hough Line Detection
    height, width = masked_gray.shape
    min_size = min(width, height)
    lines = cv2.HoughLinesP(
        edges_masked,
        rho=1,
        theta=np.pi / 180,
        threshold=int(min_size * 0.3),
        minLineLength=int(min_size * 0.4),
        maxLineGap=int(min_size * 0.2),
    )
    line_count = 0 if lines is None else len(lines)
    line_score = 1.0 if 1 <= line_count <= 4 else 0.0

    # Combine scores (same weights as balls.py)
    combined_score = (
        variance_score * 0.15
        + edge_score * 0.25
        + frequency_score * 0.30
        + line_score * 0.30
    )

    has_stripe = combined_score > 0.5
    pattern_score = (
        edge_density * 0.4 + max_projection_std * 0.3 + variance / 1000 * 0.3
    )

    if debug:
        logger.debug(
            f"Pattern - Var: {variance:.1f}, Edge: {edge_density:.3f}, "
            f"Freq: {max_projection_std:.2f}, Lines: {line_count}, "
            f"Score: {combined_score:.3f} -> {'STRIPE' if has_stripe else 'SOLID'}"
        )

    return has_stripe, combined_score, edge_density, pattern_score


def _extract_number_features(
    ball_region: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    ball_type: BallType,
    debug: bool = False,
) -> tuple[Optional[int], float]:
    """Extract number recognition features using contour analysis."""
    if ball_type == BallType.CUE:
        return None, 1.0
    if ball_type == BallType.EIGHT:
        return 8, 1.0

    try:
        preprocessed = _preprocess_for_number_recognition(ball_region, mask)
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, 0.0

        # Filter contours
        number_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0.0
                if 0.3 < aspect_ratio < 2.0:
                    number_contours.append(contour)

        if not number_contours:
            return None, 0.0

        largest_contour = max(number_contours, key=cv2.contourArea)
        number = _analyze_contour_for_number(largest_contour, ball_type)
        confidence = 0.6 if number is not None else 0.0

        if debug and number is not None:
            logger.debug(f"Number detected: {number} (confidence: {confidence:.2f})")

        return number, confidence

    except Exception as e:
        if debug:
            logger.debug(f"Number recognition failed: {e}")
        return None, 0.0


def _preprocess_for_number_recognition(
    ball_region: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """Preprocess ball region for number recognition."""
    size = (64, 64)
    resized = cv2.resize(ball_region, size)
    resized_mask = cv2.resize(mask, size)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    masked = cv2.bitwise_and(enhanced, enhanced, mask=resized_mask)
    blurred = cv2.GaussianBlur(masked, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary


def _analyze_contour_for_number(
    contour: NDArray[np.float64], ball_type: BallType
) -> Optional[int]:
    """Analyze contour shape to infer ball number using heuristics."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None

    compactness = 4 * math.pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Heuristic classification (same as balls.py)
    if ball_type == BallType.SOLID:
        if compactness > 0.7:
            return 6
        elif aspect_ratio < 0.5:
            return 1
        elif solidity < 0.8:
            return 4
        else:
            return 2
    elif ball_type == BallType.STRIPE:
        if compactness > 0.7:
            return 9
        elif aspect_ratio < 0.5:
            return 11
        elif solidity < 0.8:
            return 14
        else:
            return 12

    return None


def _calculate_sharpness(
    ball_region: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> float:
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    masked_laplacian = laplacian[mask > 0]

    if masked_laplacian.size == 0:
        return 0.0

    return float(np.var(masked_laplacian))


def _calculate_extraction_quality(
    ball_region: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    brightness: float,
    sharpness: float,
    pattern_confidence: float,
) -> float:
    """Calculate overall feature extraction quality score."""
    region_area = np.count_nonzero(mask)
    size_quality = min(1.0, region_area / 1000.0)
    brightness_quality = min(1.0, brightness / 200.0)
    sharpness_quality = min(1.0, sharpness / 100.0)

    quality = (
        size_quality * 0.25
        + brightness_quality * 0.25
        + sharpness_quality * 0.25
        + pattern_confidence * 0.25
    )

    return quality


def _get_default_features(ball_type: BallType) -> BallFeatures:
    """Get default features when extraction fails."""
    return BallFeatures(
        dominant_color_hsv=(0, 0, 0),
        color_histogram=np.zeros(180 + 256 + 256),
        color_variance=0.0,
        has_stripe=False,
        stripe_confidence=0.0,
        edge_density=0.0,
        pattern_score=0.0,
        detected_number=None,
        number_confidence=0.0,
        brightness=0.0,
        saturation=0.0,
        sharpness=0.0,
        ball_type=ball_type,
        extraction_quality=0.0,
    )
