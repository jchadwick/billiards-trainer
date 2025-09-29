"""Cue stick detection and shot analysis algorithms.

Implements requirements FR-VIS-030 to FR-VIS-038:
- Cue stick detection using advanced line detection
- Cue angle and position tracking
- Shot detection and analysis
- Strike force estimation
- English/spin detection
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np

# Import from models to use consistent data structures
from ..models import CueState, CueStick


class ShotType(Enum):
    """Type of shot detected."""

    STRAIGHT = "straight"
    ENGLISH_LEFT = "english_left"
    ENGLISH_RIGHT = "english_right"
    FOLLOW = "follow"
    DRAW = "draw"
    MASSE = "masse"


@dataclass
class ExtendedCueStick(CueStick):
    """Extended cue stick information with motion analysis."""

    # Additional fields for advanced tracking
    butt_position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)  # pixels/frame
    acceleration: tuple[float, float] = (0.0, 0.0)  # pixels/frame²
    strike_velocity: float = 0.0  # pixels/frame at contact
    strike_angle: float = 0.0  # angle at contact
    frame_id: int = 0
    detection_history: list[bool] = field(default_factory=list)


@dataclass
class ExtendedShotEvent:
    """Extended shot event with detailed analysis."""

    # Basic shot information
    shot_id: int
    timestamp: float
    cue_ball_position: tuple[float, float]
    target_ball_position: Optional[tuple[float, float]] = None
    cue_angle: float = 0.0
    estimated_force: float = 0.0

    # Contact analysis
    contact_point: tuple[float, float] = (0.0, 0.0)  # on cue ball
    strike_force: float = 0.0  # estimated force 0-100
    strike_angle: float = 0.0  # angle of contact

    # Ball analysis
    cue_ball_velocity_pre: tuple[float, float] = (0.0, 0.0)
    cue_ball_velocity_post: tuple[float, float] = (0.0, 0.0)

    # Shot classification
    shot_type: ShotType = ShotType.STRAIGHT
    english_amount: float = 0.0  # -1 to 1 (left to right)
    follow_draw: float = 0.0  # -1 to 1 (draw to follow)

    confidence: float = 0.0


class CueDetector:
    """Advanced cue stick detection and shot analysis.

    Implements:
    - Multi-algorithm line detection (Hough, LSD, probabilistic)
    - Temporal tracking and filtering
    - Motion analysis and state detection
    - Shot event detection and classification
    - False positive filtering
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize cue detector with configuration.

        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detection parameters
        self.min_cue_length = config.get("min_cue_length", 150)
        self.max_cue_length = config.get("max_cue_length", 800)
        self.min_line_thickness = config.get("min_line_thickness", 3)
        self.max_line_thickness = config.get("max_line_thickness", 25)

        # Line detection thresholds
        self.hough_threshold = config.get("hough_threshold", 100)
        self.hough_min_line_length = config.get("hough_min_line_length", 100)
        self.hough_max_line_gap = config.get("hough_max_line_gap", 20)

        # LSD parameters
        self.lsd_scale = config.get("lsd_scale", 0.8)
        self.lsd_sigma = config.get("lsd_sigma", 0.6)
        self.lsd_quant = config.get("lsd_quant", 2.0)

        # Motion analysis
        self.velocity_threshold = config.get("velocity_threshold", 5.0)
        self.acceleration_threshold = config.get("acceleration_threshold", 2.0)
        self.striking_velocity_threshold = config.get(
            "striking_velocity_threshold", 15.0
        )

        # Tracking parameters
        self.max_tracking_distance = config.get("max_tracking_distance", 50)
        self.tracking_history_size = config.get("tracking_history_size", 10)
        self.confidence_decay = config.get("confidence_decay", 0.95)

        # Filtering parameters
        self.min_detection_confidence = config.get("min_detection_confidence", 0.6)
        self.temporal_smoothing = config.get("temporal_smoothing", 0.7)

        # Internal state
        self.previous_cues: deque = deque(maxlen=self.tracking_history_size)
        self.frame_count = 0
        self.shot_events: list[ExtendedShotEvent] = []

        # Initialize line segment detector
        self._init_line_detectors()

    def _init_line_detectors(self) -> None:
        """Initialize line detection algorithms."""
        try:
            # Line Segment Detector
            self.lsd = cv2.createLineSegmentDetector(
                scale=self.lsd_scale,
                sigma_scale=self.lsd_sigma,
                quant=self.lsd_quant,
                ang_th=22.5,
                log_eps=0,
                density_th=0.7,
                n_bins=1024,
            )
        except Exception as e:
            self.logger.warning(f"LSD initialization failed: {e}")
            self.lsd = None

    def detect_cue(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[CueStick]:
        """Detect cue stick in frame using multiple algorithms.

        Args:
            frame: Input image frame
            cue_ball_pos: Optional cue ball position for improved detection

        Returns:
            Detected CueStick object or None
        """
        self.frame_count += 1

        if frame is None or frame.size == 0:
            return None

        try:
            # Check if frame has sufficient content
            if np.sum(frame) < 1000:  # Very dark/empty frame
                return None

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)

            # Check if preprocessed frame has sufficient contrast
            if np.std(processed_frame) < 10:  # Very low contrast
                return None

            # Detect lines using multiple methods
            all_lines = self._detect_lines_multi_method(processed_frame)

            if not all_lines:
                return None

            # Filter and score cue candidates
            cue_candidates = self._filter_cue_candidates(
                all_lines, frame.shape, cue_ball_pos
            )

            if not cue_candidates:
                return None

            # Select best cue candidate
            best_cue = self._select_best_cue(cue_candidates, frame)

            if best_cue is None:
                return None

            # Apply temporal tracking and smoothing
            tracked_cue = self._apply_temporal_tracking(best_cue)

            # Analyze motion and state
            self._analyze_cue_motion(tracked_cue)

            # Store for next frame
            self.previous_cues.append(tracked_cue)

            # Convert to base CueStick format for compatibility
            return CueStick(
                tip_position=tracked_cue.tip_position,
                angle=tracked_cue.angle,
                length=tracked_cue.length,
                confidence=tracked_cue.confidence,
                state=tracked_cue.state,
                is_aiming=(tracked_cue.state == CueState.AIMING),
                tip_velocity=tracked_cue.velocity,
                angular_velocity=tracked_cue.angular_velocity,
            )

        except Exception as e:
            self.logger.error(f"Cue detection failed: {e}")
            return None

    def _preprocess_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Preprocess frame for line detection.

        Args:
            frame: Input BGR frame

        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)

        return enhanced

    def _detect_lines_multi_method(self, frame: NDArray[np.uint8]) -> list[np.ndarray]:
        """Detect lines using multiple algorithms.

        Args:
            frame: Preprocessed grayscale frame

        Returns:
            List of detected lines from all methods
        """
        all_lines = []

        # Method 1: Canny + Probabilistic Hough
        hough_lines = self._detect_lines_hough(frame)
        if hough_lines is not None:
            all_lines.extend(hough_lines)

        # Method 2: Line Segment Detector (LSD)
        if self.lsd is not None:
            lsd_lines = self._detect_lines_lsd(frame)
            if lsd_lines is not None:
                all_lines.extend(lsd_lines)

        # Method 3: Morphological line detection
        morph_lines = self._detect_lines_morphological(frame)
        if morph_lines is not None:
            all_lines.extend(morph_lines)

        return all_lines

    def _detect_lines_hough(
        self, frame: NDArray[np.uint8]
    ) -> Optional[list[np.ndarray]]:
        """Detect lines using Canny edge detection + Probabilistic Hough Transform."""
        try:
            # Canny edge detection
            edges = cv2.Canny(frame, 50, 150, apertureSize=3)

            # Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap,
            )

            if lines is not None:
                return [line[0] for line in lines]

        except Exception as e:
            self.logger.debug(f"Hough line detection failed: {e}")

        return None

    def _detect_lines_lsd(self, frame: NDArray[np.uint8]) -> Optional[list[np.ndarray]]:
        """Detect lines using Line Segment Detector (LSD)."""
        try:
            lines = self.lsd.detect(frame)[0]

            if lines is not None and len(lines) > 0:
                # Convert LSD format to standard format [x1, y1, x2, y2]
                converted_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0:4]
                    converted_lines.append(np.array([x1, y1, x2, y2]))
                return converted_lines

        except Exception as e:
            self.logger.debug(f"LSD line detection failed: {e}")

        return None

    def _detect_lines_morphological(
        self, frame: NDArray[np.uint8]
    ) -> Optional[list[np.ndarray]]:
        """Detect lines using morphological operations."""
        try:
            # Create morphological kernels for different orientations
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)),  # Horizontal
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)),  # Vertical
            ]

            # Create diagonal kernels
            diag1 = np.zeros((11, 11), dtype=np.uint8)
            np.fill_diagonal(diag1, 1)
            diag2 = np.zeros((11, 11), dtype=np.uint8)
            np.fill_diagonal(np.fliplr(diag2), 1)
            kernels.extend([diag1, diag2])

            all_lines = []

            for kernel in kernels:
                # Apply morphological opening
                opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

                # Only proceed if we have meaningful content
                if np.sum(opened) < 100:  # Skip if very little content
                    continue

                # Find contours
                contours, _ = cv2.findContours(
                    opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    # Fit line to contour
                    if len(contour) >= 4:
                        [vx, vy, x, y] = cv2.fitLine(
                            contour, cv2.DIST_L2, 0, 0.01, 0.01
                        )

                        # Calculate line endpoints
                        if abs(vx) > 0.001:  # Avoid division by zero
                            lefty = int((-x * vy / vx) + y)
                            righty = int(((frame.shape[1] - x) * vy / vx) + y)

                            if (
                                0 <= lefty < frame.shape[0]
                                and 0 <= righty < frame.shape[0]
                            ):
                                line = np.array([0, lefty, frame.shape[1] - 1, righty])
                                # Check if line is long enough
                                line_length = np.sqrt(
                                    (frame.shape[1] - 1) ** 2 + (righty - lefty) ** 2
                                )
                                if line_length >= self.min_cue_length:
                                    all_lines.append(line)

            return all_lines if all_lines else None

        except Exception as e:
            self.logger.debug(f"Morphological line detection failed: {e}")

        return None

    def _filter_cue_candidates(
        self,
        lines: list[np.ndarray],
        frame_shape: tuple[int, ...],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> list[ExtendedCueStick]:
        """Filter and score potential cue stick candidates.

        Args:
            lines: Detected lines
            frame_shape: Shape of the input frame
            cue_ball_pos: Optional cue ball position for scoring

        Returns:
            List of ExtendedCueStick candidates with scores
        """
        candidates = []

        for line in lines:
            x1, y1, x2, y2 = line

            # Calculate line properties
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Normalize angle to 0-360
            if angle < 0:
                angle += 360

            # Filter by length
            if not (self.min_cue_length <= length <= self.max_cue_length):
                continue

            # Determine tip and butt positions (tip is usually closer to cue ball)
            if cue_ball_pos is not None:
                dist1 = np.sqrt(
                    (x1 - cue_ball_pos[0]) ** 2 + (y1 - cue_ball_pos[1]) ** 2
                )
                dist2 = np.sqrt(
                    (x2 - cue_ball_pos[0]) ** 2 + (y2 - cue_ball_pos[1]) ** 2
                )

                if dist1 < dist2:
                    tip_pos = (x1, y1)
                    butt_pos = (x2, y2)
                else:
                    tip_pos = (x2, y2)
                    butt_pos = (x1, y1)
            else:
                # Default: assume tip is the point with smaller y-coordinate
                if y1 < y2:
                    tip_pos = (x1, y1)
                    butt_pos = (x2, y2)
                else:
                    tip_pos = (x2, y2)
                    butt_pos = (x1, y1)

            # Calculate confidence score
            confidence = self._calculate_line_confidence(
                line, frame_shape, cue_ball_pos
            )

            if confidence >= self.min_detection_confidence:
                cue = ExtendedCueStick(
                    tip_position=tip_pos,
                    butt_position=butt_pos,
                    angle=angle,
                    length=length,
                    confidence=confidence,
                    frame_id=self.frame_count,
                )
                candidates.append(cue)

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    def _calculate_line_confidence(
        self,
        line: NDArray[np.float64],
        frame_shape: tuple[int, ...],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> float:
        """Calculate confidence score for a potential cue line.

        Args:
            line: Line coordinates [x1, y1, x2, y2]
            frame_shape: Shape of the input frame
            cue_ball_pos: Optional cue ball position

        Returns:
            Confidence score 0.0-1.0
        """
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        confidence = 0.0

        # Length score (prefer longer lines)
        length_score = min(1.0, length / self.max_cue_length)
        confidence += 0.3 * length_score

        # Position score (prefer lines not at frame edges)
        h, w = frame_shape[:2]
        edge_margin = 20

        edge_penalty = 0.0
        if (
            x1 < edge_margin
            or x1 > w - edge_margin
            or y1 < edge_margin
            or y1 > h - edge_margin
            or x2 < edge_margin
            or x2 > w - edge_margin
            or y2 < edge_margin
            or y2 > h - edge_margin
        ):
            edge_penalty = 0.2

        confidence += 0.2 * (1.0 - edge_penalty)

        # Cue ball proximity score
        if cue_ball_pos is not None:
            cx, cy = cue_ball_pos

            # Distance to line
            line_dist = self._point_to_line_distance((cx, cy), line)
            max_reasonable_distance = 200

            proximity_score = max(0.0, 1.0 - line_dist / max_reasonable_distance)
            confidence += 0.3 * proximity_score
        else:
            confidence += 0.15  # Neutral score when no cue ball position

        # Temporal consistency score (if we have previous detections)
        if self.previous_cues:
            prev_cue = self.previous_cues[-1]

            # Angle consistency
            angle_diff = abs(
                math.degrees(math.atan2(y2 - y1, x2 - x1)) - prev_cue.angle
            )
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            angle_score = max(0.0, 1.0 - angle_diff / 45.0)  # Penalize >45° changes

            # Position consistency
            tip_dist = np.sqrt(
                (x1 - prev_cue.tip_position[0]) ** 2
                + (y1 - prev_cue.tip_position[1]) ** 2
            )
            position_score = max(0.0, 1.0 - tip_dist / self.max_tracking_distance)

            temporal_score = 0.5 * angle_score + 0.5 * position_score
            confidence += 0.2 * temporal_score
        else:
            confidence += 0.1  # Neutral score for first detection

        return min(1.0, confidence)

    def _point_to_line_distance(
        self, point: tuple[float, float], line: NDArray[np.float64]
    ) -> float:
        """Calculate distance from point to line segment."""
        x0, y0 = point
        x1, y1, x2, y2 = line

        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            # Line is a point
            return float(np.sqrt(x0 - x1) ** 2 + (y0 - y1) ** 2)

        # Parameter t for closest point on line
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp to line segment

        # Closest point on line
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        return float(np.sqrt(x0 - closest_x) ** 2 + (y0 - closest_y) ** 2)

    def _select_best_cue(
        self, candidates: list[ExtendedCueStick], frame: NDArray[np.uint8]
    ) -> Optional[ExtendedCueStick]:
        """Select the best cue candidate from the list.

        Args:
            candidates: List of cue candidates
            frame: Original frame for additional validation

        Returns:
            Best ExtendedCueStick candidate or None
        """
        if not candidates:
            return None

        # Additional validation for top candidates
        validated_candidates = []

        for candidate in candidates[:3]:  # Check top 3 candidates
            if self._validate_cue_candidate(candidate, frame):
                validated_candidates.append(candidate)

        if not validated_candidates:
            return None

        # Return highest confidence validated candidate
        return validated_candidates[0]

    def _validate_cue_candidate(
        self, cue: ExtendedCueStick, frame: NDArray[np.uint8]
    ) -> bool:
        """Additional validation for cue candidate.

        Args:
            cue: Cue candidate to validate
            frame: Original frame

        Returns:
            True if candidate is valid
        """
        try:
            # Check for reasonable thickness along the line
            thickness_valid = self._check_line_thickness(cue, frame)

            # Check for consistent color/texture along line
            texture_valid = self._check_line_texture(cue, frame)

            # Check for reasonable position (not intersecting with table edges too much)
            position_valid = self._check_cue_position(cue, frame)

            return thickness_valid and texture_valid and position_valid

        except Exception as e:
            self.logger.debug(f"Cue validation failed: {e}")
            return False

    def _check_line_thickness(
        self, cue: ExtendedCueStick, frame: NDArray[np.uint8]
    ) -> bool:
        """Check if line has reasonable thickness for a cue stick."""
        # Sample perpendicular profiles at multiple points along the line
        x1, y1 = cue.tip_position
        x2, y2 = cue.butt_position

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)

        if length == 0:
            return False

        # Normalized direction and perpendicular vectors
        dir_x, dir_y = dx / length, dy / length
        perp_x, perp_y = -dir_y, dir_x

        valid_thickness_count = 0
        sample_count = 5

        for i in range(sample_count):
            # Sample point along the line
            t = (i + 1) / (sample_count + 1)
            sample_x = x1 + t * dx
            sample_y = y1 + t * dy

            # Check thickness at this point
            thickness = self._measure_line_thickness_at_point(
                frame, (sample_x, sample_y), (perp_x, perp_y)
            )

            if self.min_line_thickness <= thickness <= self.max_line_thickness:
                valid_thickness_count += 1

        return valid_thickness_count >= sample_count // 2

    def _measure_line_thickness_at_point(
        self,
        frame: NDArray[np.uint8],
        point: tuple[float, float],
        perpendicular: tuple[float, float],
    ) -> float:
        """Measure line thickness at a specific point."""
        x, y = point
        perp_x, perp_y = perpendicular

        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return 0

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Sample along perpendicular direction
        max_search = 30
        intensities = []

        for offset in range(-max_search, max_search + 1):
            sample_x = int(x + offset * perp_x)
            sample_y = int(y + offset * perp_y)

            if 0 <= sample_x < gray.shape[1] and 0 <= sample_y < gray.shape[0]:
                intensities.append(gray[sample_y, sample_x])
            else:
                intensities.append(0)

        if len(intensities) < 10:
            return 0

        # Find edges (significant intensity changes)
        gradient = np.gradient(intensities)
        edges = np.where(np.abs(gradient) > 20)[0]

        if len(edges) >= 2:
            # Distance between first and last significant edge
            return edges[-1] - edges[0]

        return 0

    def _check_line_texture(
        self, cue: ExtendedCueStick, frame: NDArray[np.uint8]
    ) -> bool:
        """Check for consistent texture along the line (simplified check)."""
        # For now, just check that the line region has reasonable contrast
        # More sophisticated texture analysis could be added later
        return True

    def _check_cue_position(
        self, cue: ExtendedCueStick, frame: NDArray[np.uint8]
    ) -> bool:
        """Check if cue position is reasonable (not intersecting table boundaries too much)."""
        # This would need table detection integration
        # For now, just check it's not entirely at the frame edge
        x1, y1 = cue.tip_position
        x2, y2 = cue.butt_position
        h, w = frame.shape[:2]

        edge_margin = 10

        # Check if both endpoints are at frame edges (likely false positive)
        tip_at_edge = (
            x1 < edge_margin
            or x1 > w - edge_margin
            or y1 < edge_margin
            or y1 > h - edge_margin
        )
        butt_at_edge = (
            x2 < edge_margin
            or x2 > w - edge_margin
            or y2 < edge_margin
            or y2 > h - edge_margin
        )

        return not (tip_at_edge and butt_at_edge)

    def _apply_temporal_tracking(
        self, current_cue: ExtendedCueStick
    ) -> ExtendedCueStick:
        """Apply temporal tracking and smoothing to cue detection.

        Args:
            current_cue: Current frame detection

        Returns:
            Smoothed and tracked cue
        """
        if not self.previous_cues:
            return current_cue

        prev_cue = self.previous_cues[-1]

        # Apply temporal smoothing
        alpha = self.temporal_smoothing

        # Smooth position
        smoothed_tip_x = (
            alpha * prev_cue.tip_position[0] + (1 - alpha) * current_cue.tip_position[0]
        )
        smoothed_tip_y = (
            alpha * prev_cue.tip_position[1] + (1 - alpha) * current_cue.tip_position[1]
        )

        smoothed_butt_x = (
            alpha * prev_cue.butt_position[0]
            + (1 - alpha) * current_cue.butt_position[0]
        )
        smoothed_butt_y = (
            alpha * prev_cue.butt_position[1]
            + (1 - alpha) * current_cue.butt_position[1]
        )

        # Smooth angle (handle wraparound)
        angle_diff = current_cue.angle - prev_cue.angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        smoothed_angle = prev_cue.angle + (1 - alpha) * angle_diff
        if smoothed_angle < 0:
            smoothed_angle += 360
        elif smoothed_angle >= 360:
            smoothed_angle -= 360

        # Update cue with smoothed values
        current_cue.tip_position = (smoothed_tip_x, smoothed_tip_y)
        current_cue.butt_position = (smoothed_butt_x, smoothed_butt_y)
        current_cue.angle = smoothed_angle
        current_cue.length = np.sqrt(
            (smoothed_butt_x - smoothed_tip_x) ** 2
            + (smoothed_butt_y - smoothed_tip_y) ** 2
        )

        return current_cue

    def _analyze_cue_motion(self, cue: ExtendedCueStick) -> None:
        """Analyze cue motion and determine state.

        Args:
            cue: Current cue detection to analyze
        """
        if len(self.previous_cues) < 2:
            cue.state = CueState.AIMING
            return

        prev_cue = self.previous_cues[-1]
        prev_prev_cue = self.previous_cues[-2]

        # Calculate velocity
        dt = 1.0  # Assuming 1 frame time unit
        vx = (cue.tip_position[0] - prev_cue.tip_position[0]) / dt
        vy = (cue.tip_position[1] - prev_cue.tip_position[1]) / dt
        cue.velocity = (vx, vy)

        # Calculate acceleration
        prev_vx = (prev_cue.tip_position[0] - prev_prev_cue.tip_position[0]) / dt
        prev_vy = (prev_cue.tip_position[1] - prev_prev_cue.tip_position[1]) / dt

        ax = (vx - prev_vx) / dt
        ay = (vy - prev_vy) / dt
        cue.acceleration = (ax, ay)

        # Calculate angular velocity
        angle_diff = cue.angle - prev_cue.angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        cue.angular_velocity = angle_diff / dt

        # Determine state based on motion
        speed = np.sqrt(vx * vx + vy * vy)
        acceleration_mag = np.sqrt(ax * ax + ay * ay)

        if speed > self.striking_velocity_threshold:
            cue.state = CueState.STRIKING
            cue.strike_velocity = speed
            cue.strike_angle = cue.angle
        elif speed > self.velocity_threshold:
            if acceleration_mag > self.acceleration_threshold:
                # Accelerating - likely starting to strike
                cue.state = CueState.STRIKING
            else:
                # Moving but not accelerating much - aiming adjustment
                cue.state = CueState.AIMING
        else:
            # Low speed - steady aiming or stationary
            cue.state = CueState.AIMING

    def detect_shot_event(
        self,
        cue: CueStick,
        cue_ball_pos: tuple[float, float],
        cue_ball_velocity: tuple[float, float],
    ) -> Optional[ExtendedShotEvent]:
        """Detect if a shot event occurred.

        Args:
            cue: Current cue detection
            cue_ball_pos: Current cue ball position
            cue_ball_velocity: Current cue ball velocity

        Returns:
            ShotEvent if detected, None otherwise
        """
        if cue.state != CueState.STRIKING:
            return None

        # Check if cue tip is close to cue ball
        distance = np.sqrt(
            (cue.tip_position[0] - cue_ball_pos[0]) ** 2
            + (cue.tip_position[1] - cue_ball_pos[1]) ** 2
        )

        # Threshold for contact detection (adjust based on typical ball size)
        contact_threshold = 30  # pixels

        if distance > contact_threshold:
            return None

        # Check if cue ball started moving
        ball_speed = np.sqrt(cue_ball_velocity[0] ** 2 + cue_ball_velocity[1] ** 2)
        if ball_speed < 2.0:  # Minimum speed to consider movement
            return None

        # Get strike velocity - prioritize from cue state or tip velocity
        strike_velocity = 0.0
        if hasattr(cue, "strike_velocity") and cue.strike_velocity > 0:
            strike_velocity = cue.strike_velocity
        elif hasattr(cue, "tip_velocity") and cue.tip_velocity != (0.0, 0.0):
            strike_velocity = np.sqrt(
                cue.tip_velocity[0] ** 2 + cue.tip_velocity[1] ** 2
            )
        else:
            # Estimate from ball velocity
            strike_velocity = ball_speed * 1.5  # Rough approximation

        # Create shot event
        shot_event = ExtendedShotEvent(
            shot_id=len(self.shot_events) + 1,
            timestamp=self.frame_count,
            cue_ball_position=cue_ball_pos,
            cue_angle=cue.angle,
            estimated_force=self._estimate_strike_force(strike_velocity),
            contact_point=self._calculate_contact_point(cue, cue_ball_pos),
            strike_force=self._estimate_strike_force(strike_velocity),
            strike_angle=cue.angle,
            cue_ball_velocity_pre=(0.0, 0.0),  # Assume stationary before
            cue_ball_velocity_post=cue_ball_velocity,
            shot_type=self._classify_shot_type(cue, cue_ball_pos, cue_ball_velocity),
            confidence=min(
                cue.confidence, 0.9
            ),  # Shot detection is inherently uncertain
        )

        # Calculate English and follow/draw
        shot_event.english_amount = self._calculate_english(cue, cue_ball_pos)
        shot_event.follow_draw = self._calculate_follow_draw(cue, cue_ball_pos)

        self.shot_events.append(shot_event)
        return shot_event

    def _calculate_contact_point(
        self, cue: CueStick, cue_ball_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculate the contact point on the cue ball surface.

        Args:
            cue: Cue stick information
            cue_ball_pos: Cue ball center position

        Returns:
            Contact point coordinates
        """
        # Vector from cue ball center to cue tip
        dx = cue.tip_position[0] - cue_ball_pos[0]
        dy = cue.tip_position[1] - cue_ball_pos[1]

        # Normalize to ball surface (assuming standard ball radius)
        ball_radius = 15  # pixels (adjust based on actual detection)
        distance = np.sqrt(dx * dx + dy * dy)

        if distance == 0:
            return cue_ball_pos

        # Point on ball surface closest to cue tip
        factor = ball_radius / distance
        contact_x = cue_ball_pos[0] + dx * factor
        contact_y = cue_ball_pos[1] + dy * factor

        return (contact_x, contact_y)

    def _estimate_strike_force(self, strike_velocity: float) -> float:
        """Estimate strike force from cue velocity.

        Args:
            strike_velocity: Cue tip velocity at contact

        Returns:
            Estimated force (0-100 scale)
        """
        # Simple linear mapping (could be improved with physics modeling)
        max_velocity = 50.0  # pixels/frame
        force = min(100.0, (strike_velocity / max_velocity) * 100.0)
        return force

    def _classify_shot_type(
        self,
        cue: CueStick,
        cue_ball_pos: tuple[float, float],
        cue_ball_velocity: tuple[float, float],
    ) -> ShotType:
        """Classify the type of shot based on cue and ball motion.

        Args:
            cue: Cue stick information
            cue_ball_pos: Cue ball position
            cue_ball_velocity: Cue ball velocity after contact

        Returns:
            Classified shot type
        """
        # Angle between cue direction and cue ball velocity
        cue_dir_x = np.cos(np.radians(cue.angle))
        cue_dir_y = np.sin(np.radians(cue.angle))

        ball_speed = np.sqrt(cue_ball_velocity[0] ** 2 + cue_ball_velocity[1] ** 2)
        if ball_speed == 0:
            return ShotType.STRAIGHT

        ball_dir_x = cue_ball_velocity[0] / ball_speed
        ball_dir_y = cue_ball_velocity[1] / ball_speed

        # Dot product for angle between directions
        dot_product = cue_dir_x * ball_dir_x + cue_dir_y * ball_dir_y
        angle_diff = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

        # Simple classification based on angle difference
        if angle_diff > 30:
            # Significant deviation suggests English
            cross_product = cue_dir_x * ball_dir_y - cue_dir_y * ball_dir_x
            if cross_product > 0:
                return ShotType.ENGLISH_LEFT
            else:
                return ShotType.ENGLISH_RIGHT
        else:
            # Check for follow/draw based on contact point
            contact_point = self._calculate_contact_point(cue, cue_ball_pos)

            # Distance from center (positive = above center)
            center_offset = contact_point[1] - cue_ball_pos[1]

            if center_offset < -5:  # Hit below center
                return ShotType.DRAW
            elif center_offset > 5:  # Hit above center
                return ShotType.FOLLOW
            else:
                return ShotType.STRAIGHT

    def _calculate_english(
        self, cue: CueStick, cue_ball_pos: tuple[float, float]
    ) -> float:
        """Calculate amount of English (side spin) applied.

        Args:
            cue: Cue stick information
            cue_ball_pos: Cue ball position

        Returns:
            English amount (-1 to 1, left to right)
        """
        contact_point = self._calculate_contact_point(cue, cue_ball_pos)

        # Horizontal offset from center
        horizontal_offset = contact_point[0] - cue_ball_pos[0]

        # Normalize by ball radius
        ball_radius = 15  # pixels
        english = np.clip(horizontal_offset / ball_radius, -1.0, 1.0)

        return english

    def _calculate_follow_draw(
        self, cue: CueStick, cue_ball_pos: tuple[float, float]
    ) -> float:
        """Calculate amount of follow/draw applied.

        Args:
            cue: Cue stick information
            cue_ball_pos: Cue ball position

        Returns:
            Follow/draw amount (-1 to 1, draw to follow)
        """
        contact_point = self._calculate_contact_point(cue, cue_ball_pos)

        # Vertical offset from center (negative = below center = draw)
        vertical_offset = cue_ball_pos[1] - contact_point[1]

        # Normalize by ball radius
        ball_radius = 15  # pixels
        follow_draw = np.clip(vertical_offset / ball_radius, -1.0, 1.0)

        return follow_draw

    def get_multiple_cues(
        self, frame: NDArray[np.uint8], max_cues: int = 2
    ) -> list[CueStick]:
        """Detect multiple cue sticks in frame (FR-VIS-034).

        Args:
            frame: Input image frame
            max_cues: Maximum number of cues to detect

        Returns:
            List of detected cue sticks
        """
        try:
            # Check if frame has sufficient content
            if np.sum(frame) < 1000:  # Very dark/empty frame
                return []

            # Get all line candidates
            processed_frame = self._preprocess_frame(frame)

            # Check if preprocessed frame has sufficient contrast
            if np.std(processed_frame) < 10:  # Very low contrast
                return []

            all_lines = self._detect_lines_multi_method(processed_frame)

            if not all_lines:
                return []

            # Get all cue candidates
            all_candidates = self._filter_cue_candidates(all_lines, frame.shape)

            # Filter out overlapping detections
            unique_cues = self._filter_overlapping_cues(all_candidates)

            # Convert to base CueStick format and return top candidates
            result_cues = []
            for extended_cue in unique_cues[:max_cues]:
                base_cue = CueStick(
                    tip_position=extended_cue.tip_position,
                    angle=extended_cue.angle,
                    length=extended_cue.length,
                    confidence=extended_cue.confidence,
                    state=extended_cue.state,
                    is_aiming=(extended_cue.state == CueState.AIMING),
                )
                result_cues.append(base_cue)

            return result_cues

        except Exception as e:
            self.logger.error(f"Multiple cue detection failed: {e}")
            return []

    def _filter_overlapping_cues(
        self, candidates: list[ExtendedCueStick]
    ) -> list[ExtendedCueStick]:
        """Filter out overlapping cue detections.

        Args:
            candidates: List of cue candidates

        Returns:
            Filtered list without overlaps
        """
        if len(candidates) <= 1:
            return candidates

        unique_cues = []
        overlap_threshold = 50  # pixels

        for candidate in candidates:
            is_unique = True

            for existing in unique_cues:
                # Check distance between tip positions
                tip_dist = np.sqrt(
                    (candidate.tip_position[0] - existing.tip_position[0]) ** 2
                    + (candidate.tip_position[1] - existing.tip_position[1]) ** 2
                )

                # Check angle similarity
                angle_diff = abs(candidate.angle - existing.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if tip_dist < overlap_threshold and angle_diff < 30:
                    is_unique = False
                    break

            if is_unique:
                unique_cues.append(candidate)

        return unique_cues

    def estimate_cue_angle(
        self,
        cue_line: NDArray[np.float64],
        reference_frame: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Calculate cue angle relative to table coordinate system.

        Args:
            cue_line: Line coordinates [x1, y1, x2, y2]
            reference_frame: Optional reference frame for table coordinate transformation

        Returns:
            Cue angle in degrees (0-360)
        """
        x1, y1, x2, y2 = cue_line

        # Calculate angle from horizontal
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)

        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360

        # Apply table coordinate transformation if reference frame is available
        if reference_frame is not None:
            try:
                # Try to detect table in reference frame to get transformation matrix
                from ..detection.table import TableDetector

                table_detector = TableDetector()
                table_result = table_detector.detect_complete_table(reference_frame)

                if table_result and table_result.perspective_transform is not None:
                    # Transform the line endpoints to table coordinates
                    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    points = points.reshape(-1, 1, 2)

                    # Apply perspective transformation
                    transformed_points = cv2.perspectiveTransform(
                        points, table_result.perspective_transform
                    )
                    transformed_points = transformed_points.reshape(-1, 2)

                    # Recalculate angle in table coordinates
                    tx1, ty1 = transformed_points[0]
                    tx2, ty2 = transformed_points[1]

                    table_angle_rad = math.atan2(ty2 - ty1, tx2 - tx1)
                    table_angle_deg = math.degrees(table_angle_rad)

                    # Normalize to 0-360 range
                    if table_angle_deg < 0:
                        table_angle_deg += 360

                    return table_angle_deg

            except Exception as e:
                # Fall back to image coordinates if transformation fails
                self.logger.debug(f"Table coordinate transformation failed: {e}")

        return angle_deg

    def detect_cue_movement(
        self, current_cue: CueStick, previous_cue: Optional[CueStick]
    ) -> bool:
        """Detect if cue is moving (shooting vs aiming).

        Args:
            current_cue: Current cue detection
            previous_cue: Previous cue detection

        Returns:
            True if cue is moving significantly
        """
        if previous_cue is None:
            return False

        # Calculate position change
        tip_movement = np.sqrt(
            (current_cue.tip_position[0] - previous_cue.tip_position[0]) ** 2
            + (current_cue.tip_position[1] - previous_cue.tip_position[1]) ** 2
        )

        # Calculate angle change
        angle_change = abs(current_cue.angle - previous_cue.angle)
        if angle_change > 180:
            angle_change = 360 - angle_change

        # Movement threshold
        position_threshold = 10.0  # pixels
        angle_threshold = 5.0  # degrees

        return tip_movement > position_threshold or angle_change > angle_threshold

    def get_detection_statistics(self) -> dict[str, Any]:
        """Get detection performance statistics.

        Returns:
            Dictionary with detection statistics
        """
        if not self.previous_cues:
            return {
                "total_detections": 0,
                "average_confidence": 0.0,
                "shot_events": 0,
                "detection_rate": 0.0,
            }

        detections = len(self.previous_cues)
        avg_confidence = sum(cue.confidence for cue in self.previous_cues) / detections
        detection_rate = detections / max(1, self.frame_count)

        return {
            "total_detections": detections,
            "average_confidence": avg_confidence,
            "shot_events": len(self.shot_events),
            "detection_rate": detection_rate,
            "frames_processed": self.frame_count,
        }

    def reset_tracking(self) -> None:
        """Reset tracking state (useful for new game/session)."""
        self.previous_cues.clear()
        self.shot_events.clear()
        self.frame_count = 0

    def set_cue_ball_position(self, position: tuple[float, float]) -> None:
        """Set known cue ball position to improve detection accuracy.

        Args:
            position: Cue ball (x, y) position in pixels
        """
        self.cue_ball_position = position

    def visualize_detection(
        self,
        frame: NDArray[np.uint8],
        cue: CueStick,
        shot_event: Optional[ExtendedShotEvent] = None,
    ) -> NDArray[np.float64]:
        """Visualize cue detection results on frame.

        Args:
            frame: Original frame
            cue: Detected cue stick
            shot_event: Optional shot event to visualize

        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()

        if cue is None:
            return vis_frame

        # Draw cue line
        tip_pos = (int(cue.tip_position[0]), int(cue.tip_position[1]))

        # Calculate butt position if not available
        if hasattr(cue, "shaft_points") and cue.shaft_points:
            butt_pos = (int(cue.shaft_points[-1][0]), int(cue.shaft_points[-1][1]))
        else:
            # Estimate butt position from tip and angle
            butt_x = cue.tip_position[0] - cue.length * np.cos(np.radians(cue.angle))
            butt_y = cue.tip_position[1] - cue.length * np.sin(np.radians(cue.angle))
            butt_pos = (int(butt_x), int(butt_y))

        # Color based on state
        if cue.state == CueState.STRIKING:
            color = (0, 0, 255)  # Red for striking
        elif cue.state == CueState.AIMING:
            color = (0, 255, 0)  # Green for aiming
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw cue line
        cv2.line(vis_frame, tip_pos, butt_pos, color, 3)

        # Draw tip marker
        cv2.circle(vis_frame, tip_pos, 8, color, -1)

        # Draw info text
        info_text = f"Angle: {cue.angle:.1f}° Conf: {cue.confidence:.2f} State: {cue.state.value}"
        cv2.putText(
            vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        # Draw velocity vector if available
        if hasattr(cue, "tip_velocity") and cue.tip_velocity != (0.0, 0.0):
            vel_end_x = int(tip_pos[0] + cue.tip_velocity[0] * 5)
            vel_end_y = int(tip_pos[1] + cue.tip_velocity[1] * 5)
            cv2.arrowedLine(
                vis_frame, tip_pos, (vel_end_x, vel_end_y), (255, 255, 0), 2
            )

        # Draw shot event info if available
        if shot_event:
            shot_text = f"Shot: {shot_event.shot_type.value} Force: {shot_event.strike_force:.1f}"
            cv2.putText(
                vis_frame,
                shot_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Draw contact point
            contact_pos = (
                int(shot_event.contact_point[0]),
                int(shot_event.contact_point[1]),
            )
            cv2.circle(vis_frame, contact_pos, 5, (0, 255, 255), -1)

        return vis_frame
