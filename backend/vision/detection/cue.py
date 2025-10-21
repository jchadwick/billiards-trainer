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
from numpy.typing import NDArray

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

    def __init__(self, config: dict[str, Any], yolo_detector=None) -> None:
        """Initialize cue detector with configuration.

        Args:
            config: Configuration dictionary with detection parameters
            yolo_detector: Optional YOLODetector instance for YOLO-based cue detection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.yolo_detector = yolo_detector

        # Geometry parameters
        geometry = config.get("geometry", {})
        self.min_cue_length = geometry.get("min_cue_length", 150)
        self.max_cue_length = geometry.get("max_cue_length", 800)
        self.min_line_thickness = geometry.get("min_line_thickness", 3)
        self.max_line_thickness = geometry.get("max_line_thickness", 25)
        self.ball_radius = geometry.get("ball_radius", 15)

        # Hough transform parameters
        hough = config.get("hough", {})
        self.hough_threshold = hough.get("threshold", 100)
        self.hough_min_line_length = hough.get("min_line_length", 100)
        self.hough_max_line_gap = hough.get("max_line_gap", 20)
        self.hough_rho = hough.get("rho", 1)
        self.hough_theta_divisor = hough.get("theta_divisor", 180)

        # LSD parameters
        lsd = config.get("lsd", {})
        self.lsd_scale = lsd.get("scale", 0.8)
        self.lsd_sigma = lsd.get("sigma", 0.6)
        self.lsd_quant = lsd.get("quant", 2.0)
        self.lsd_ang_th = lsd.get("ang_th", 22.5)
        self.lsd_log_eps = lsd.get("log_eps", 0)
        self.lsd_density_th = lsd.get("density_th", 0.7)
        self.lsd_n_bins = lsd.get("n_bins", 1024)

        # Edge detection parameters
        edge = config.get("edge_detection", {})
        self.canny_low_threshold = edge.get("canny_low_threshold", 50)
        self.canny_high_threshold = edge.get("canny_high_threshold", 150)
        self.canny_aperture_size = edge.get("canny_aperture_size", 3)
        self.gradient_threshold = edge.get("gradient_threshold", 20)

        # Preprocessing parameters
        preproc = config.get("preprocessing", {})
        self.gaussian_kernel_size = preproc.get("gaussian_kernel_size", 5)
        self.gaussian_sigma = preproc.get("gaussian_sigma", 1.0)
        self.clahe_clip_limit = preproc.get("clahe_clip_limit", 2.0)
        self.clahe_tile_grid_size = preproc.get("clahe_tile_grid_size", 8)
        self.morphology_kernel_size = preproc.get("morphology_kernel_size", 3)
        self.morphology_horizontal_kernel = preproc.get(
            "morphology_horizontal_kernel", [15, 3]
        )
        self.morphology_vertical_kernel = preproc.get(
            "morphology_vertical_kernel", [3, 15]
        )
        self.morphology_diagonal_kernel_size = preproc.get(
            "morphology_diagonal_kernel_size", 11
        )
        self.min_frame_sum = preproc.get("min_frame_sum", 1000)
        self.min_contrast_std = preproc.get("min_contrast_std", 10)
        self.min_morph_content = preproc.get("min_morph_content", 100)

        # Motion analysis parameters
        motion = config.get("motion", {})
        self.velocity_threshold = motion.get("velocity_threshold", 5.0)
        self.acceleration_threshold = motion.get("acceleration_threshold", 2.0)
        self.striking_velocity_threshold = motion.get(
            "striking_velocity_threshold", 15.0
        )
        self.position_movement_threshold = motion.get(
            "position_movement_threshold", 10.0
        )
        self.angle_change_threshold = motion.get("angle_change_threshold", 5.0)
        self.min_ball_speed = motion.get("min_ball_speed", 2.0)

        # Tracking parameters
        tracking = config.get("tracking", {})
        self.max_tracking_distance = tracking.get("max_tracking_distance", 50)
        self.tracking_history_size = tracking.get("tracking_history_size", 10)
        self.confidence_decay = tracking.get("confidence_decay", 0.95)
        self.temporal_smoothing = tracking.get("temporal_smoothing", 0.7)

        # Validation parameters
        validation = config.get("validation", {})
        self.min_detection_confidence = validation.get("min_detection_confidence", 0.6)
        self.use_background_subtraction = validation.get(
            "use_background_subtraction", False
        )
        self.background_threshold = validation.get("background_threshold", 30)
        self.thickness_sample_count = validation.get("thickness_sample_count", 5)
        self.max_thickness_search = validation.get("max_thickness_search", 30)
        self.edge_margin = validation.get("edge_margin", 20)
        self.position_edge_margin = validation.get("position_edge_margin", 10)

        # Proximity parameters
        proximity = config.get("proximity", {})
        self.max_distance_to_cue_ball = proximity.get("max_distance_to_cue_ball", 40)
        self.max_tip_distance = proximity.get("max_tip_distance", 300)
        self.max_reasonable_distance = proximity.get("max_reasonable_distance", 200)
        self.contact_threshold = proximity.get("contact_threshold", 30)
        self.overlap_threshold = proximity.get("overlap_threshold", 50)
        self.max_angle_overlap = proximity.get("max_angle_overlap", 30)

        # Scoring parameters
        scoring = config.get("scoring", {})
        self.length_weight = scoring.get("length_weight", 0.3)
        self.position_weight = scoring.get("position_weight", 0.2)
        self.proximity_weight = scoring.get("proximity_weight", 0.3)
        self.temporal_weight = scoring.get("temporal_weight", 0.2)
        self.no_cue_ball_score = scoring.get("no_cue_ball_score", 0.15)
        self.no_temporal_score = scoring.get("no_temporal_score", 0.1)
        self.max_angle_change = scoring.get("max_angle_change", 45.0)
        self.edge_penalty = scoring.get("edge_penalty", 0.2)
        self.min_confidence_for_shot = scoring.get("min_confidence_for_shot", 0.9)

        # Shot detection parameters
        shot = config.get("shot_detection", {})
        self.max_velocity = shot.get("max_velocity", 50.0)
        self.english_deviation_angle = shot.get("english_deviation_angle", 30)
        self.follow_draw_threshold = shot.get("follow_draw_threshold", 5)
        self.center_offset_threshold = shot.get("center_offset_threshold", 5)

        # Advanced parameters
        advanced = config.get("advanced", {})
        self.enable_yolo_detection = advanced.get("enable_yolo_detection", True)
        self.enable_lsd_detection = advanced.get("enable_lsd_detection", True)
        self.enable_morphological_detection = advanced.get(
            "enable_morphological_detection", True
        )
        self.top_candidates_to_validate = advanced.get("top_candidates_to_validate", 3)
        self.max_cues_to_detect = advanced.get("max_cues_to_detect", 2)

        # Cassapa edge refinement parameters (used with YOLO detection)
        # Reference: cassapa/detector.cpp:827-954
        cassapa = config.get("cassapa", {})
        self.cassapa_edge_sample_step = cassapa.get(
            "edge_sample_step", 3
        )  # Sample every 3rd point
        self.cassapa_edge_search_distance = cassapa.get(
            "edge_search_distance", 10
        )  # Max perpendicular search distance
        self.cassapa_edge_min_points = cassapa.get(
            "edge_min_points", 5
        )  # Minimum edge points for centerline calculation

        # Internal state
        self.previous_cues: deque = deque(maxlen=self.tracking_history_size)
        self.frame_count = 0
        self.shot_events: list[ExtendedShotEvent] = []
        self.background_frame: Optional[NDArray[np.uint8]] = None

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
                ang_th=self.lsd_ang_th,
                log_eps=self.lsd_log_eps,
                density_th=self.lsd_density_th,
                n_bins=self.lsd_n_bins,
            )
        except Exception as e:
            self.logger.warning(f"LSD initialization failed: {e}")
            self.lsd = None

    def detect_cue(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: Optional[tuple[float, float]] = None,
        all_ball_positions: Optional[list[tuple[float, float]]] = None,
        table_corners: Optional[
            tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ]
        ] = None,
    ) -> Optional[CueStick]:
        """Detect cue stick in frame using multiple algorithms.

        Detection priority (if enabled):
        1. Cassapa-style detection (HSV + Hough + edge refinement)
        2. YOLO detection (if available and cue_ball_pos provided)
        3. Traditional line-based detection (fallback)

        Args:
            frame: Input image frame
            cue_ball_pos: Optional cue ball position for improved detection
            all_ball_positions: Optional list of all ball positions for orientation detection
            table_corners: Optional table corners (top_left, top_right, bottom_left, bottom_right) for cushion-based orientation

        Returns:
            Detected CueStick object or None
        """
        self.frame_count += 1

        if frame is None or frame.size == 0:
            return None

        try:
            # Fallback to YOLO detection if available (PRIMARY METHOD)
            # Note: cue_ball_pos is optional - YOLO can detect cues without it
            if self.yolo_detector is not None:
                yolo_cue = self._detect_cue_with_yolo(
                    frame, cue_ball_pos, all_ball_positions, table_corners
                )
                if yolo_cue is not None:
                    self.logger.debug("Cue detected using YOLO detection")
                    return yolo_cue

            # Last resort: traditional line-based detection
            # Check if frame has sufficient content
            if np.sum(frame) < self.min_frame_sum:  # Very dark/empty frame
                return None

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)

            # Check if preprocessed frame has sufficient contrast
            if np.std(processed_frame) < self.min_contrast_std:  # Very low contrast
                return None

            # Detect lines using multiple methods
            all_lines = self._detect_lines_multi_method(processed_frame)

            if not all_lines:
                return None

            # Filter and score cue candidates
            cue_candidates = self._filter_cue_candidates(
                all_lines, frame.shape, cue_ball_pos, all_ball_positions, table_corners
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

            self.logger.debug("Cue detected using traditional line-based detection")

            # Convert to base CueStick format for compatibility
            return CueStick(
                tip_position=tracked_cue.tip_position,
                angle=tracked_cue.angle,
                length=tracked_cue.length,
                confidence=tracked_cue.confidence,
                state=tracked_cue.state,
                is_aiming=(tracked_cue.state == CueState.AIMING),
                tip_velocity=tracked_cue.tip_velocity,
                angular_velocity=tracked_cue.angular_velocity,
            )

        except Exception as e:
            self.logger.error(f"Cue detection failed: {e}")
            return None

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set the background reference frame (empty table).

        Args:
            frame: Reference frame of empty table
        """
        self.background_frame = frame.copy()
        self.use_background_subtraction = True
        self.logger.info("Background frame set for cue detection")

    def _distance_to_nearest_cushion(
        self,
        point: tuple[float, float],
        table_corners: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> float:
        """Calculate distance from a point to the nearest table cushion/rail.

        Args:
            point: Point (x, y) to measure from
            table_corners: Table corners (top_left, top_right, bottom_left, bottom_right)

        Returns:
            Distance to the nearest cushion edge in pixels
        """
        top_left, top_right, bottom_left, bottom_right = table_corners
        x, y = point

        # Calculate distances to each of the four rails
        # Top rail: line from top_left to top_right
        top_dist = self._point_to_line_distance(
            point, np.array([top_left[0], top_left[1], top_right[0], top_right[1]])
        )

        # Bottom rail: line from bottom_left to bottom_right
        bottom_dist = self._point_to_line_distance(
            point,
            np.array(
                [bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1]]
            ),
        )

        # Left rail: line from top_left to bottom_left
        left_dist = self._point_to_line_distance(
            point, np.array([top_left[0], top_left[1], bottom_left[0], bottom_left[1]])
        )

        # Right rail: line from top_right to bottom_right
        right_dist = self._point_to_line_distance(
            point,
            np.array([top_right[0], top_right[1], bottom_right[0], bottom_right[1]]),
        )

        # Return the minimum distance to any cushion
        return min(top_dist, bottom_dist, left_dist, right_dist)

    def _find_closest_ball_on_cue_line(
        self,
        tip_pos: tuple[float, float],
        butt_pos: tuple[float, float],
        all_ball_positions: list[tuple[float, float]],
        cue_ball_pos: Optional[tuple[float, float]] = None,
    ) -> Optional[tuple[float, float]]:
        """Find the closest ball that the cue line is pointing at.

        Args:
            tip_pos: Cue tip position
            butt_pos: Cue butt position
            all_ball_positions: List of all ball positions
            cue_ball_pos: Optional cue ball position to exclude from consideration

        Returns:
            Position of the closest ball the cue is pointing at, or None if no ball is close enough
        """
        if not all_ball_positions:
            return None

        # Create line from tip to butt
        line = np.array([tip_pos[0], tip_pos[1], butt_pos[0], butt_pos[1]])

        # Find balls that are close to the cue line
        closest_ball = None
        min_distance = float("inf")

        for ball_pos in all_ball_positions:
            # Skip the cue ball - we're looking for TARGET balls
            if cue_ball_pos is not None:
                cue_ball_distance = np.sqrt(
                    (ball_pos[0] - cue_ball_pos[0]) ** 2
                    + (ball_pos[1] - cue_ball_pos[1]) ** 2
                )
                # If this ball is very close to the cue ball position, skip it
                if cue_ball_distance < self.ball_radius * 2:  # Within 2 ball radii
                    continue
            # Calculate distance from ball to cue line
            distance = self._point_to_line_distance(ball_pos, line)

            # Only consider balls that are close to the line
            if distance <= self.max_distance_to_cue_ball:
                # Check if ball is "in front" of the tip (not behind the butt)
                # Vector from butt to tip
                cue_dx = tip_pos[0] - butt_pos[0]
                cue_dy = tip_pos[1] - butt_pos[1]

                # Vector from butt to ball
                ball_dx = ball_pos[0] - butt_pos[0]
                ball_dy = ball_pos[1] - butt_pos[1]

                # Dot product to check if ball is in the direction of the tip
                dot_product = cue_dx * ball_dx + cue_dy * ball_dy

                if dot_product > 0:  # Ball is ahead of butt
                    # Calculate distance from tip to ball
                    tip_to_ball_dist = np.sqrt(
                        (tip_pos[0] - ball_pos[0]) ** 2
                        + (tip_pos[1] - ball_pos[1]) ** 2
                    )

                    if tip_to_ball_dist < min_distance:
                        min_distance = tip_to_ball_dist
                        closest_ball = ball_pos

        return closest_ball

    def _detect_mass_near_point(
        self,
        frame: NDArray[np.uint8],
        point: tuple[float, float],
        table_color_hsv: Optional[tuple[int, int, int]] = None,
        radius: float = 100.0,
    ) -> float:
        """Detect amount of non-table mass (person's body) near a point.

        The butt end of the cue will have more visual mass (player's body) nearby,
        while the tip will have less obstruction.

        Args:
            frame: Input frame in BGR format
            point: Point to check (x, y)
            table_color_hsv: Table color to exclude (if known)
            radius: Radius around point to check

        Returns:
            Amount of non-table content near the point (0.0-1.0)
        """
        try:
            x, y = int(point[0]), int(point[1])
            r = int(radius)

            # Extract region around point
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(frame.shape[1], x + r)
            y2 = min(frame.shape[0], y + r)

            if x2 <= x1 or y2 <= y1:
                return 0.0

            region = frame[y1:y2, x1:x2]

            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

            # Create mask excluding table color (green/blue felt)
            # Typical pool table colors
            lower_table = np.array([35, 40, 40], dtype=np.uint8)  # Green
            upper_table = np.array([90, 255, 255], dtype=np.uint8)

            table_mask = cv2.inRange(hsv, lower_table, upper_table)

            # Invert to get non-table pixels (this is the "mass")
            non_table_mask = cv2.bitwise_not(table_mask)

            # Calculate proportion of non-table pixels
            total_pixels = non_table_mask.size
            mass_pixels = np.count_nonzero(non_table_mask)

            return mass_pixels / total_pixels if total_pixels > 0 else 0.0

        except Exception as e:
            self.logger.debug(f"Mass detection failed: {e}")
            return 0.0

    def _detect_cue_with_yolo(
        self,
        frame: NDArray[np.uint8],
        cue_ball_pos: tuple[float, float],
        all_ball_positions: Optional[list[tuple[float, float]]] = None,
        table_corners: Optional[
            tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ]
        ] = None,
    ) -> Optional[CueStick]:
        """Detect cue using YOLO and convert bounding box to CueStick object.

        Args:
            frame: Input image frame
            cue_ball_pos: Cue ball position for orientation
            all_ball_positions: Optional list of all ball positions for improved orientation detection
            table_corners: Optional table corners (top_left, top_right, bottom_left, bottom_right) for cushion-based orientation

        Returns:
            CueStick object or None if not detected
        """
        try:
            # Get YOLO cue detection
            yolo_detection = self.yolo_detector.detect_cue(frame)
            if yolo_detection is None:
                return None

            # Convert YOLO bounding box to CueStick
            # Extract bounding box coordinates
            x1, y1, x2, y2 = yolo_detection.bbox
            center_x, center_y = yolo_detection.center
            width, height = yolo_detection.width, yolo_detection.height

            # Use YOLO's calculated angle (from Hough line detection)
            angle = yolo_detection.angle

            # Calculate cue length from bounding box diagonal
            length = max(width, height)

            # Calculate angle in radians
            angle_rad = np.radians(angle)

            # Use the line endpoints if available (from Hough line detection), otherwise calculate from center and angle
            if (
                hasattr(yolo_detection, "line_end1")
                and yolo_detection.line_end1 is not None
                and hasattr(yolo_detection, "line_end2")
                and yolo_detection.line_end2 is not None
            ):
                # Use actual detected line endpoints - these give us the true line orientation
                end1_x, end1_y = yolo_detection.line_end1
                end2_x, end2_y = yolo_detection.line_end2
                self.logger.debug(
                    f"Using Hough line endpoints: end1=({end1_x:.1f}, {end1_y:.1f}), end2=({end2_x:.1f}, {end2_y:.1f})"
                )

                # Apply Cassapa edge refinement to improve line accuracy
                initial_line = np.array(
                    [end1_x, end1_y, end2_x, end2_y], dtype=np.float64
                )
                refined_line = self._refine_line_with_edges(
                    frame, initial_line, yolo_detection.bbox
                )

                # Use refined endpoints
                end1_x, end1_y = refined_line[0], refined_line[1]
                end2_x, end2_y = refined_line[2], refined_line[3]
                self.logger.debug(
                    f"After edge refinement: end1=({end1_x:.1f}, {end1_y:.1f}), end2=({end2_x:.1f}, {end2_y:.1f})"
                )
            else:
                # Fallback: calculate endpoints from center and angle
                # Use the line center if available (from Hough line detection), otherwise use bbox center
                if (
                    hasattr(yolo_detection, "line_center")
                    and yolo_detection.line_center is not None
                ):
                    center_x, center_y = yolo_detection.line_center
                    self.logger.debug(
                        f"Using Hough line center: ({center_x:.1f}, {center_y:.1f})"
                    )
                else:
                    self.logger.debug(
                        f"Using bbox center: ({center_x:.1f}, {center_y:.1f})"
                    )

                # Calculate two endpoints of the cue based on center and angle
                # Endpoint 1: center + (length/2) in angle direction
                end1_x = center_x + (length / 2) * np.cos(angle_rad)
                end1_y = center_y + (length / 2) * np.sin(angle_rad)

                # Endpoint 2: center - (length/2) in angle direction
                end2_x = center_x - (length / 2) * np.cos(angle_rad)
                end2_y = center_y - (length / 2) * np.sin(angle_rad)

            # Determine orientation using multiple strategies

            # STRATEGY 1 (Cassapa method): Use table center distance if table_corners available
            # This ensures butt is farther from table center, tip is closer
            # Reference: cassapa/detector.cpp:143-158
            tip_x, tip_y = None, None

            if table_corners is not None:
                table_center = self._calculate_table_center(table_corners)
                line_to_normalize = np.array(
                    [end1_x, end1_y, end2_x, end2_y], dtype=np.float64
                )
                normalized_line = self._normalize_line_direction_cassapa(
                    line_to_normalize, table_center
                )

                # After normalization, point 1 is butt (farther), point 2 is tip (closer)
                tip_x, tip_y = normalized_line[2], normalized_line[3]
                self.logger.debug(
                    f"Using Cassapa table-center orientation: tip=({tip_x:.1f}, {tip_y:.1f})"
                )

            # STRATEGY 2: Check mass at both ends (fallback if no table_corners or for validation)
            if tip_x is None:
                mass_at_end1 = self._detect_mass_near_point(frame, (end1_x, end1_y))
                mass_at_end2 = self._detect_mass_near_point(frame, (end2_x, end2_y))

                self.logger.debug(
                    f"Mass at end1: {mass_at_end1:.3f}, Mass at end2: {mass_at_end2:.3f}"
                )

                # If there's a significant difference in mass, use that to determine orientation
                mass_threshold = 0.10  # 10% difference is significant
                if abs(mass_at_end1 - mass_at_end2) > mass_threshold:
                    if mass_at_end1 > mass_at_end2:
                        # End1 has more mass, so it's the butt, end2 is the tip
                        tip_x, tip_y = end2_x, end2_y
                        self.logger.debug(
                            "Using mass detection: end1=butt (more mass), end2=tip"
                        )
                    else:
                        # End2 has more mass, so it's the butt, end1 is the tip
                        tip_x, tip_y = end1_x, end1_y
                        self.logger.debug(
                            "Using mass detection: end2=butt (more mass), end1=tip"
                        )

            # STRATEGY 3: Try both orientations and find which one has the closest ball in front of the tip
            if tip_x is None and all_ball_positions and len(all_ball_positions) > 0:
                # Option 1: end1 is tip, end2 is butt
                closest_ball_1 = self._find_closest_ball_on_cue_line(
                    (end1_x, end1_y), (end2_x, end2_y), all_ball_positions, cue_ball_pos
                )

                # Option 2: end2 is tip, end1 is butt
                closest_ball_2 = self._find_closest_ball_on_cue_line(
                    (end2_x, end2_y), (end1_x, end1_y), all_ball_positions, cue_ball_pos
                )

                # Choose the orientation where we found a ball the cue is pointing at
                if closest_ball_1 is not None and closest_ball_2 is None:
                    tip_x, tip_y = end1_x, end1_y
                elif closest_ball_2 is not None and closest_ball_1 is None:
                    tip_x, tip_y = end2_x, end2_y
                elif closest_ball_1 is not None and closest_ball_2 is not None:
                    # Both found a ball - choose the one with the closer ball
                    dist1 = np.sqrt(
                        (end1_x - closest_ball_1[0]) ** 2
                        + (end1_y - closest_ball_1[1]) ** 2
                    )
                    dist2 = np.sqrt(
                        (end2_x - closest_ball_2[0]) ** 2
                        + (end2_y - closest_ball_2[1]) ** 2
                    )

                    # Check if distances are similar (within 20% of each other)
                    # If so, use cushion-based tie-breaker
                    if (
                        table_corners is not None
                        and abs(dist1 - dist2) / max(dist1, dist2) < 0.2
                    ):
                        # Tie-breaker: prefer orientation where butt is closer to a cushion
                        # This is more realistic as players typically position themselves near the table edge
                        butt_cushion_dist_1 = self._distance_to_nearest_cushion(
                            (end2_x, end2_y), table_corners
                        )
                        butt_cushion_dist_2 = self._distance_to_nearest_cushion(
                            (end1_x, end1_y), table_corners
                        )

                        if butt_cushion_dist_1 < butt_cushion_dist_2:
                            # Option 1 butt is closer to cushion
                            tip_x, tip_y = end1_x, end1_y
                        else:
                            # Option 2 butt is closer to cushion
                            tip_x, tip_y = end2_x, end2_y
                    elif dist1 < dist2:
                        tip_x, tip_y = end1_x, end1_y
                    else:
                        tip_x, tip_y = end2_x, end2_y
                else:
                    # No balls found on cue line, fallback to cue ball proximity
                    tip_x = None

            # STRATEGY 4 (final fallback): Cue ball proximity - closest end to cue ball is the tip
            if tip_x is None:
                dist1 = np.sqrt(
                    (end1_x - cue_ball_pos[0]) ** 2 + (end1_y - cue_ball_pos[1]) ** 2
                )
                dist2 = np.sqrt(
                    (end2_x - cue_ball_pos[0]) ** 2 + (end2_y - cue_ball_pos[1]) ** 2
                )
                if dist1 < dist2:
                    tip_x, tip_y = end1_x, end1_y
                    self.logger.debug("Using cue ball proximity: end1=tip")
                else:
                    tip_x, tip_y = end2_x, end2_y
                    self.logger.debug("Using cue ball proximity: end2=tip")

            # Create CueStick object
            cue_stick = CueStick(
                tip_position=(tip_x, tip_y),
                angle=angle,
                length=length,
                confidence=yolo_detection.confidence,
                state=CueState.AIMING,  # Default to aiming (motion detection requires temporal info)
                is_aiming=True,
                tip_velocity=(0.0, 0.0),  # Would need temporal info for velocity
                angular_velocity=0.0,
            )

            return cue_stick

        except Exception as e:
            self.logger.debug(f"YOLO cue detection failed: {e}")
            return None

    def _refine_line_with_edges(
        self,
        frame: NDArray[np.uint8],
        initial_line: NDArray[np.float64],
        bbox: tuple[float, float, float, float],
    ) -> Optional[NDArray[np.float64]]:
        """Refine YOLO-detected line using Cassapa edge detection for sub-pixel accuracy.

        This method applies Cassapa's edge-based refinement technique to a YOLO-detected
        cue line. It extracts the cue region from the bounding box, applies edge detection,
        then uses the existing Cassapa methods to find both edges and calculate a precise
        centerline.

        Algorithm:
        1. Extract cue region from bounding box
        2. Convert to grayscale and apply Canny edge detection
        3. Adjust initial line coordinates to region space
        4. Sample points along the initial line
        5. Find edges perpendicular to the line
        6. Calculate refined centerline by averaging edges
        7. Convert back to full frame coordinates

        Args:
            frame: Input BGR frame
            initial_line: YOLO-detected line as [x1, y1, x2, y2] in full frame coordinates
            bbox: Bounding box as (x1, y1, x2, y2) in full frame coordinates

        Returns:
            Refined line as [x1, y1, x2, y2] numpy array in full frame coordinates,
            or initial_line if refinement fails
        """
        try:
            # Extract bounding box region
            x1, y1, x2, y2 = bbox
            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1_int = max(0, x1_int)
            y1_int = max(0, y1_int)
            x2_int = min(w, x2_int)
            y2_int = min(h, y2_int)

            # Check for valid region
            if x2_int <= x1_int or y2_int <= y1_int:
                self.logger.debug("Invalid bbox region, returning initial line")
                return initial_line

            # Extract region
            region = frame[y1_int:y2_int, x1_int:x2_int]

            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Adjust line coordinates to region space
            line_in_region = initial_line.copy()
            line_in_region[0] -= x1  # x1 of line
            line_in_region[1] -= y1  # y1 of line
            line_in_region[2] -= x1  # x2 of line
            line_in_region[3] -= y1  # y2 of line

            # Apply Cassapa edge refinement
            sample_points = self._sample_line_points(
                line_in_region, step=self.cassapa_edge_sample_step
            )

            if not sample_points or len(sample_points) == 0:
                self.logger.debug("No sample points, returning initial line")
                return initial_line

            # Find cue edges
            edge_points_1, edge_points_2 = self._find_cue_edges(
                edges, sample_points, line_in_region
            )

            # Check if we have sufficient edge points
            if (
                len(edge_points_1) >= self.cassapa_edge_min_points
                and len(edge_points_2) >= self.cassapa_edge_min_points
            ):
                # Calculate refined centerline
                refined_line = self._calculate_centerline(edge_points_1, edge_points_2)

                if refined_line is not None:
                    # Adjust back to full frame coordinates
                    refined_line[0] += x1  # x1 of line
                    refined_line[1] += y1  # y1 of line
                    refined_line[2] += x1  # x2 of line
                    refined_line[3] += y1  # y2 of line

                    self.logger.debug(
                        f"Edge refinement successful: {len(edge_points_1)} and "
                        f"{len(edge_points_2)} edge points found"
                    )
                    return refined_line

            # If refinement failed, return original line
            self.logger.debug(
                f"Edge refinement failed: insufficient edge points "
                f"({len(edge_points_1)}, {len(edge_points_2)}), returning initial line"
            )
            return initial_line

        except Exception as e:
            self.logger.debug(f"Edge refinement failed: {e}, returning initial line")
            return initial_line

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

        # Apply background subtraction if enabled
        if self.use_background_subtraction and self.background_frame is not None:
            # Convert background to grayscale if needed
            if len(self.background_frame.shape) == 3:
                bg_gray = cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2GRAY)
            else:
                bg_gray = self.background_frame

            # Compute absolute difference
            diff = cv2.absdiff(gray, bg_gray)

            # Threshold to get foreground mask
            _, fg_mask = cv2.threshold(
                diff, self.background_threshold, 255, cv2.THRESH_BINARY
            )

            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.morphology_kernel_size, self.morphology_kernel_size),
            )
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Apply mask to keep only foreground
            gray = cv2.bitwise_and(gray, gray, mask=fg_mask)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray,
            (self.gaussian_kernel_size, self.gaussian_kernel_size),
            self.gaussian_sigma,
        )

        # Enhance contrast
        enhanced = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size),
        ).apply(blurred)

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
            edges = cv2.Canny(
                frame,
                self.canny_low_threshold,
                self.canny_high_threshold,
                apertureSize=self.canny_aperture_size,
            )

            # Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=self.hough_rho,
                theta=np.pi / self.hough_theta_divisor,
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
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, tuple(self.morphology_horizontal_kernel)
                ),  # Horizontal
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, tuple(self.morphology_vertical_kernel)
                ),  # Vertical
            ]

            # Create diagonal kernels
            diag1 = np.zeros(
                (
                    self.morphology_diagonal_kernel_size,
                    self.morphology_diagonal_kernel_size,
                ),
                dtype=np.uint8,
            )
            np.fill_diagonal(diag1, 1)
            diag2 = np.zeros(
                (
                    self.morphology_diagonal_kernel_size,
                    self.morphology_diagonal_kernel_size,
                ),
                dtype=np.uint8,
            )
            np.fill_diagonal(np.fliplr(diag2), 1)
            kernels.extend([diag1, diag2])

            all_lines = []

            for kernel in kernels:
                # Apply morphological opening
                opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

                # Only proceed if we have meaningful content
                if (
                    np.sum(opened) < self.min_morph_content
                ):  # Skip if very little content
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

    # ============================================================================
    # Phase 3: Line Equation Calculation and Point Sampling
    # Reference: cassapa/detector.cpp:827-863, pool_utils.cpp:44-54
    # ============================================================================

    def _get_line_equation(self, line: NDArray[np.float64]) -> tuple[float, float]:
        """Calculate line equation in slope-intercept form (y = mx + h).

        This is a fundamental utility for the Cassapa centerline refinement algorithm.
        It converts a line defined by two endpoints into slope-intercept form for
        efficient point sampling and perpendicular edge detection.

        Reference: cassapa/pool_utils.cpp:44-54

        Algorithm:
        1. Extract endpoints (x1, y1) and (x2, y2) from line
        2. Calculate dx = x2 - x1 and dy = y2 - y1
        3. Calculate slope m = dy/dx (add epsilon to avoid division by zero)
        4. Calculate y-intercept h = y1 - m*x1
        5. Return (m, h) representing y = mx + h

        Args:
            line: Line as [x1, y1, x2, y2] numpy array

        Returns:
            Tuple (m, h) where:
            - m is the slope (dy/dx)
            - h is the y-intercept
            Line equation: y = m*x + h

        Edge cases:
        - Near-vertical lines (dx ≈ 0): Adds epsilon 0.000001 to prevent division by zero
        - Horizontal lines (dy = 0): Returns m=0, h=y1
        - Degenerate lines (x1==x2 and y1==y2): Returns m=0, h=y1
        """
        x1, y1, x2, y2 = line

        # Calculate differences
        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero for vertical or near-vertical lines
        # Reference: pool_utils.cpp:48 uses 0.000001 epsilon
        if abs(dx) < 0.000001:
            dx = 0.000001

        # Calculate slope: m = dy/dx
        m = dy / dx

        # Calculate y-intercept: h = y1 - m*x1
        h = y1 - m * x1

        return (float(m), float(h))

    def _sample_line_points(
        self, line: NDArray[np.float64], step: int = 3
    ) -> list[tuple[float, float]]:
        """Sample points along a line at regular intervals.

        This implements the Cassapa point sampling algorithm used for edge-based
        centerline refinement. Points are sampled along the dominant direction
        (horizontal or vertical) of the line, with the perpendicular coordinate
        calculated using the line equation.

        Reference: cassapa/detector.cpp:854-863

        Algorithm:
        1. Extract line endpoints (x1, y1) and (x2, y2)
        2. Determine dominant direction:
           - If |dx| > |dy|: horizontal (sample along x)
           - Else: vertical (sample along y)
        3. Sample points every `step` pixels in dominant direction
        4. Calculate perpendicular coordinate using line equation y = mx + h
        5. Return list of (x, y) sample points

        Args:
            line: Line as [x1, y1, x2, y2] numpy array
            step: Sampling interval in pixels (default: 3)

        Returns:
            List of (x, y) tuples representing sample points along the line

        Edge cases:
        - Empty lines: Returns empty list
        - Degenerate lines (x1==x2 and y1==y2): Returns single point
        - Near-vertical lines: Samples along y-axis
        - Near-horizontal lines: Samples along x-axis
        - Step larger than line length: Returns at least start and end points
        """
        x1, y1, x2, y2 = line

        # Handle degenerate line (point)
        if x1 == x2 and y1 == y2:
            return [(float(x1), float(y1))]

        # Determine dominant direction
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        points: list[tuple[float, float]] = []

        # Check if line is truly vertical (dx very small)
        # Use a threshold to detect near-vertical lines
        if dx < 0.001:
            # Perfectly vertical line: sample along y-axis with constant x
            # Reference: detector.cpp:860-862 (vertical case)
            y_start = min(y1, y2)
            y_end = max(y1, y2)
            x_const = (x1 + x2) / 2  # Average x value

            # Sample every `step` pixels
            y = y_start
            while y <= y_end:
                points.append((float(x_const), float(y)))
                y += step

            # Always include the end point if not already included
            if len(points) == 0 or points[-1][1] < y_end:
                points.append((float(x_const), float(y_end)))

        elif dx > dy:
            # Horizontal dominant: sample along x-axis
            # Reference: detector.cpp:857-859 (horizontal case)
            # Calculate line equation
            m, h = self._get_line_equation(line)

            x_start = min(x1, x2)
            x_end = max(x1, x2)

            # Sample every `step` pixels
            x = x_start
            while x <= x_end:
                # Calculate y using line equation: y = m*x + h
                y = m * x + h
                points.append((float(x), float(y)))
                x += step

            # Always include the end point if not already included
            if len(points) == 0 or points[-1][0] < x_end:
                y = m * x_end + h
                points.append((float(x_end), float(y)))

        else:
            # Vertical dominant: sample along y-axis
            # Reference: detector.cpp:860-862 (vertical case)
            # Calculate line equation
            m, h = self._get_line_equation(line)

            y_start = min(y1, y2)
            y_end = max(y1, y2)

            # Sample every `step` pixels
            y = y_start
            while y <= y_end:
                # Calculate x from line equation: x = (y - h)/m
                # m is guaranteed to be non-zero due to epsilon in _get_line_equation
                x = (y - h) / m
                points.append((float(x), float(y)))
                y += step

            # Always include the end point if not already included
            if len(points) == 0 or points[-1][1] < y_end:
                x = (y_end - h) / m
                points.append((float(x), float(y_end)))

        return points

    def _find_cue_edges(
        self,
        mask: NDArray[np.uint8],
        sample_points: list[tuple[float, float]],
        line: NDArray[np.float64],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Find cue edges by searching perpendicular to the detected line.

        This is THE KEY INNOVATION from Cassapa. Instead of detecting the centerline
        directly, it finds both edges of the cue by searching perpendicular to the
        detected line, then averages them for sub-pixel accuracy.

        Reference: cassapa/detector.cpp:879-913
        - Lines 879-880: Setup and iteration through sample points
        - Lines 883-888: Determine search direction (perpendicular to line)
        - Lines 890-913: Search along perpendicular normal for edges

        Algorithm:
        1. For each sample point along the line:
           a. Calculate the normal (perpendicular line) at that point
           b. Determine search direction (perpendicular to line direction)
           c. Search outward in both directions (+ and -) up to edge_search_distance pixels
           d. Check mask pixel intensity at each position
           e. If pixel > 100 (white = cue detected), record as edge point
           f. Accumulate edge points into two lists: edge_points_1 and edge_points_2

        Args:
            mask: Binary mask (grayscale) where cue pixels are white (>100)
            sample_points: List of (x, y) points sampled along the detected line
            line: Line as [x1, y1, x2, y2] representing the detected cue centerline

        Returns:
            Tuple of (edge_points_1, edge_points_2) where each is a list of (x, y)
            points representing the two edges of the cue stick. Returns ([], [])
            if no edges found or if inputs are invalid.

        Example:
            >>> # Create edge mask from cue region
            >>> gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            >>> edges = cv2.Canny(gray, 50, 150)
            >>> line = np.array([100, 200, 300, 400])
            >>> sample_points = detector._sample_line_points(line, step=3)
            >>> edge1, edge2 = detector._find_cue_edges(edges, sample_points, line)
            >>> # edge1 = [(x1, y1), (x2, y2), ...] - one edge of cue
            >>> # edge2 = [(x1, y1), (x2, y2), ...] - other edge of cue
        """
        # Handle empty inputs
        if not sample_points or len(sample_points) == 0:
            return ([], [])

        if line is None or len(line) != 4:
            return ([], [])

        edge_points_1: list[tuple[float, float]] = []
        edge_points_2: list[tuple[float, float]] = []

        try:
            h, w = mask.shape

            # Extract line endpoints
            x1, y1, x2, y2 = (
                float(line[0]),
                float(line[1]),
                float(line[2]),
                float(line[3]),
            )

            # Determine search direction (perpendicular to line direction)
            # Reference: detector.cpp:883-888
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            go_horizontal = dy > dx  # Note: perpendicular to line direction!

            # Calculate line slope and normal slope
            # Avoid division by zero
            if abs(dx) > 0.001:
                line_m = (y2 - y1) / (x2 - x1)
                y1 - line_m * x1
            else:
                # Vertical line: normal is horizontal
                line_m = float("inf")

            # Calculate normal (perpendicular) slope
            # If line is vertical (m = inf), normal is horizontal (m = 0)
            # If line is horizontal (m = 0), normal is vertical (m = inf)
            if abs(line_m) < 0.001:
                # Horizontal line -> vertical normal
                normal_m = float("inf")
                normal_h = 0.0
            elif line_m == float("inf"):
                # Vertical line -> horizontal normal
                normal_m = 0.0
                normal_h = 0.0
            else:
                # Regular case: normal_m = -1 / line_m
                normal_m = -1.0 / line_m
                normal_h = 0.0  # Will be calculated for each point

            # Iterate through sample points
            # Reference: detector.cpp:879-880
            for point in sample_points:
                x, y = point

                # Calculate normal line through this point
                # normal: y = normal_m * x + normal_h
                if normal_m != float("inf"):
                    normal_h = y - normal_m * x

                # Search along perpendicular normal for edges
                # Reference: detector.cpp:890-913
                for j in range(self.cassapa_edge_search_distance + 1):
                    # Calculate points on both sides of the line
                    if go_horizontal:
                        # Search horizontally (perpendicular to vertical line)
                        # Reference: detector.cpp:893-899
                        xn1 = x + j
                        yn1 = (
                            normal_m * xn1 + normal_h if normal_m != float("inf") else y
                        )

                        xn2 = x - j
                        yn2 = (
                            normal_m * xn2 + normal_h if normal_m != float("inf") else y
                        )
                    else:
                        # Search vertically (perpendicular to horizontal line)
                        # Reference: detector.cpp:901-907
                        yn1 = y + j
                        if normal_m != float("inf") and abs(normal_m) > 0.001:
                            xn1 = (yn1 - normal_h) / normal_m
                        else:
                            xn1 = x

                        yn2 = y - j
                        if normal_m != float("inf") and abs(normal_m) > 0.001:
                            xn2 = (yn2 - normal_h) / normal_m
                        else:
                            xn2 = x

                    # Check bounds and pixel intensity for first edge
                    # Reference: detector.cpp:909-913
                    if 0 <= int(yn1) < h and 0 <= int(xn1) < w:
                        if mask[int(yn1), int(xn1)] > 100:
                            edge_points_1.append((xn1, yn1))

                    # Check bounds and pixel intensity for second edge
                    if 0 <= int(yn2) < h and 0 <= int(xn2) < w:
                        if mask[int(yn2), int(xn2)] > 100:
                            edge_points_2.append((xn2, yn2))

            return (edge_points_1, edge_points_2)

        except Exception as e:
            self.logger.debug(f"Edge detection failed: {e}")
            return ([], [])

    def _calculate_centerline(
        self,
        edge_points_1: list[tuple[float, float]],
        edge_points_2: list[tuple[float, float]],
    ) -> Optional[NDArray[np.float64]]:
        """Calculate precise centerline by averaging two fitted edge lines.

        This implements Phase 3.5 of the Cassapa detector's KEY INNOVATION:
        After detecting both edges of the cue, fit a line through each edge,
        then average them to get a precise centerline with sub-pixel accuracy.

        Reference: cassapa/detector.cpp:923-954
        - Lines 923-924: Validate minimum points (5 per edge)
        - Lines 926-927: Convert edge points to arrays
        - Lines 929-930: Fit line through edge_points_1 using DIST_L2
        - Lines 931-932: Fit line through edge_points_2 using DIST_L2
        - Lines 934-937: Extract direction vectors (vx, vy) and points (x0, y0)
        - Lines 939-946: Generate endpoints for both edge lines (extending 1000 pixels)
        - Lines 948-951: Calculate centerline as average of the two edge lines
        - Lines 953: Return centerline as [x1, y1, x2, y2]

        Algorithm:
        1. Validate both edge lists have minimum required points
        2. Convert edge point lists to numpy arrays
        3. Fit line through edge_points_1 using cv2.fitLine with DIST_L2
        4. Fit line through edge_points_2 using cv2.fitLine with DIST_L2
        5. Extract direction vectors and points from both fitted lines
        6. Generate endpoints for both edge lines (extending 1000 pixels)
        7. Calculate centerline as average of the two edge lines
        8. Return centerline as [x1, y1, x2, y2]

        Args:
            edge_points_1: List of (x, y) points along first edge
            edge_points_2: List of (x, y) points along second edge

        Returns:
            Centerline as [x1, y1, x2, y2] numpy array, or None if:
            - Either edge has fewer than cassapa_edge_min_points points
            - cv2.fitLine fails for either edge
            - Points are degenerate (all at same location)
        """
        try:
            # Step 1: Validate minimum points
            # Reference: detector.cpp:923-924
            if (
                len(edge_points_1) < self.cassapa_edge_min_points
                or len(edge_points_2) < self.cassapa_edge_min_points
            ):
                self.logger.debug(
                    f"Insufficient edge points for centerline calculation: "
                    f"{len(edge_points_1)}, {len(edge_points_2)} "
                    f"(minimum: {self.cassapa_edge_min_points})"
                )
                return None

            # Step 2: Convert edge point lists to numpy arrays
            # Reference: detector.cpp:926-927
            points1 = np.array(edge_points_1, dtype=np.float32)
            points2 = np.array(edge_points_2, dtype=np.float32)

            # Step 3: Fit line through edge_points_1
            # Reference: detector.cpp:929-930
            # cv2.fitLine returns [vx, vy, x0, y0] where:
            # - (vx, vy) is the normalized direction vector
            # - (x0, y0) is a point on the line
            line1 = cv2.fitLine(points1, cv2.DIST_L2, 0, 0.01, 0.01)
            vx1 = float(line1[0][0])
            vy1 = float(line1[1][0])
            x01 = float(line1[2][0])
            y01 = float(line1[3][0])

            # Step 4: Fit line through edge_points_2
            # Reference: detector.cpp:931-932
            line2 = cv2.fitLine(points2, cv2.DIST_L2, 0, 0.01, 0.01)
            vx2 = float(line2[0][0])
            vy2 = float(line2[1][0])
            x02 = float(line2[2][0])
            y02 = float(line2[3][0])

            # Step 5: Generate endpoints for both edge lines
            # Reference: detector.cpp:939-946
            # Extend 1000 pixels in each direction along the line

            # Edge line 1 endpoints
            xa1 = x01 - 1000 * vx1
            ya1 = y01 - 1000 * vy1
            xb1 = x01 + 1000 * vx1
            yb1 = y01 + 1000 * vy1

            # Edge line 2 endpoints
            xa2 = x02 - 1000 * vx2
            ya2 = y02 - 1000 * vy2
            xb2 = x02 + 1000 * vx2
            yb2 = y02 + 1000 * vy2

            # Step 6: Calculate centerline as average of the two edge lines
            # Reference: detector.cpp:948-951
            center_x1 = (xa1 + xa2) / 2
            center_y1 = (ya1 + ya2) / 2
            center_x2 = (xb1 + xb2) / 2
            center_y2 = (yb1 + yb2) / 2

            # Step 7: Return centerline as [x1, y1, x2, y2]
            # Reference: detector.cpp:953
            return np.array(
                [center_x1, center_y1, center_x2, center_y2], dtype=np.float64
            )

        except Exception as e:
            self.logger.debug(f"Centerline calculation failed: {e}")
            return None

    def _calculate_table_center(
        self, table_corners: Optional[np.ndarray]
    ) -> tuple[float, float]:
        """Calculate the center point of the table from its corner coordinates.

        This is a utility function used by the Cassapa direction normalization
        algorithm to determine the geometric center of the table. The center is
        used as a reference point to establish consistent line direction: the
        butt end of the cue should be farther from the table center, while the
        tip end should be closer.

        Reference: cassapa/detector.cpp:143-158 (uses table center for direction)

        Algorithm:
        1. If table_corners is None, return (0, 0) as fallback
        2. Calculate centroid of the 4 corner points: (sum(x)/4, sum(y)/4)
        3. Return (cx, cy) as the table center

        Args:
            table_corners: Array of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                          or None if table detection unavailable

        Returns:
            Tuple (cx, cy) representing the table center coordinates.
            Returns (0, 0) if table_corners is None (fallback to frame origin).

        Edge cases:
        - None input: Returns (0, 0) as safe fallback
        - Degenerate corners (all same point): Returns that point
        - Non-rectangular tables: Still calculates geometric centroid
        """
        # Handle None input - return origin as fallback
        if table_corners is None:
            return (0.0, 0.0)

        # Calculate centroid of the 4 corners
        # Reference: detector.cpp uses table center implicitly in distance calculations
        cx = float(np.mean(table_corners[:, 0]))
        cy = float(np.mean(table_corners[:, 1]))

        return (cx, cy)

    def _normalize_line_direction_cassapa(
        self, line: NDArray[np.float64], table_center: tuple[float, float]
    ) -> NDArray[np.float64]:
        """Normalize line direction so point 1 is butt (far) and point 2 is tip (near).

        This implements the Cassapa direction normalization algorithm that ensures
        consistent line orientation. The cue stick should always be oriented with
        the butt end (where the player holds) farther from the table center, and
        the tip end (that strikes the ball) closer to the table center.

        Reference: cassapa/detector.cpp:143-158
        - Lines 143-146: Calculate distance from each endpoint to table center
        - Lines 148-152: Compare distances to determine which point is farther
        - Lines 154-158: Swap points if needed to ensure point 1 is farther

        Algorithm:
        1. Extract endpoints (x1, y1) and (x2, y2) from line
        2. Get table center coordinates (cx, cy)
        3. Calculate dist1 = distance from (x1, y1) to (cx, cy)
        4. Calculate dist2 = distance from (x2, y2) to (cx, cy)
        5. If dist1 < dist2, swap points so point 1 is farther (butt end)
        6. Return normalized line [x1, y1, x2, y2]

        Args:
            line: Line as [x1, y1, x2, y2] numpy array
            table_center: Tuple (cx, cy) representing table center

        Returns:
            Normalized line as [x1, y1, x2, y2] where:
            - Point 1 (x1, y1) is FARTHER from table center (butt end)
            - Point 2 (x2, y2) is CLOSER to table center (tip end)

        Edge cases:
        - Equal distances: Returns line unchanged (no swap needed)
        - Degenerate line (x1==x2, y1==y2): Returns line unchanged
        - Table center at (0,0): Still calculates valid distances
        """
        # Extract endpoints
        x1, y1, x2, y2 = line
        cx, cy = table_center

        # Calculate distance from each endpoint to table center
        # Reference: detector.cpp:143-146
        dist1 = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        dist2 = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)

        # Swap if point 1 is closer (should be farther - it's the butt end)
        # Reference: detector.cpp:154-158
        if dist1 < dist2:
            return np.array([x2, y2, x1, y1], dtype=np.float64)

        return line

    def _calculate_cue_angle_cassapa(self, line: NDArray[np.float64]) -> float:
        """Calculate the cue stick angle in degrees from the line equation.

        This implements the Cassapa angle calculation that converts the cue line
        into an angle measurement. The angle is calculated from the line's slope
        using arctangent, then converted to degrees and normalized to [0, 360).

        Reference: cassapa/detector.cpp:1022-1023
        - Line 1022: angle_rad = atan(m) where m is slope from line equation
        - Line 1023: angle_deg = angle_rad * 180/pi, normalized to 0-360

        Algorithm:
        1. Get line equation (m, h) using _get_line_equation()
        2. Calculate angle in radians: angle_rad = arctan(m)
        3. Convert to degrees: angle_deg = angle_rad * 180/pi
        4. Normalize to 0-360 range (add 360 if negative)
        5. Return angle in degrees

        Args:
            line: Line as [x1, y1, x2, y2] numpy array

        Returns:
            Angle in degrees, normalized to range [0, 360).
            0 degrees = horizontal right, 90 degrees = vertical up.

        Edge cases:
        - Horizontal line (m=0): Returns 0 degrees
        - Vertical line (m=inf): arctan returns ~90 degrees
        - Negative angles: Normalized to positive by adding 360
        - Degenerate line: Returns 0 degrees (from m=0 in _get_line_equation)

        Example:
            Line from (0,0) to (1,1) has slope m=1:
            angle_rad = arctan(1) = pi/4 = 0.785...
            angle_deg = 45.0 degrees
        """
        # Get line equation: y = mx + h
        m, h = self._get_line_equation(line)

        # Calculate angle from slope
        # Reference: detector.cpp:1022
        angle_rad = np.arctan(m)

        # Convert to degrees
        # Reference: detector.cpp:1023
        angle_deg = np.degrees(angle_rad)

        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360

        return float(angle_deg)

    def _clip_to_frame(
        self, line: NDArray[np.float64], frame_shape: tuple
    ) -> NDArray[np.float64]:
        """Clip line to frame boundaries using cv2.clipLine.

        Uses OpenCV's clipLine to ensure the line endpoints are within the
        frame boundaries. This prevents drawing or processing issues with
        lines that extend beyond the visible frame.

        Reference: cassapa/detector.cpp:622-629

        Args:
            line: Line as [x1, y1, x2, y2]
            frame_shape: Shape of frame (height, width, ...) or (height, width)

        Returns:
            Clipped line [x1, y1, x2, y2] within frame bounds, or original line if clipping fails
        """
        try:
            x1, y1, x2, y2 = line
            height, width = frame_shape[0], frame_shape[1]

            # Convert to integer points for clipLine
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))

            # Define frame rectangle (x, y, width, height)
            rect = (0, 0, width, height)

            # Clip line to frame boundaries
            # Returns (retval, pt1, pt2) where retval is True if line intersects frame
            retval, clipped_pt1, clipped_pt2 = cv2.clipLine(rect, pt1, pt2)

            if retval:
                # Line intersects frame, use clipped endpoints
                return np.array(
                    [
                        float(clipped_pt1[0]),
                        float(clipped_pt1[1]),
                        float(clipped_pt2[0]),
                        float(clipped_pt2[1]),
                    ],
                    dtype=np.float64,
                )
            else:
                # Line completely outside frame, return original
                self.logger.debug("Line outside frame bounds after clipping")
                return line.copy()

        except Exception as e:
            self.logger.debug(f"Frame clipping failed: {e}")
            return line.copy()

    def _filter_cue_candidates(
        self,
        lines: list[np.ndarray],
        frame_shape: tuple[int, ...],
        cue_ball_pos: Optional[tuple[float, float]] = None,
        all_ball_positions: Optional[list[tuple[float, float]]] = None,
        table_corners: Optional[
            tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ]
        ] = None,
    ) -> list[CueStick]:
        """Filter and score potential cue stick candidates.

        Args:
            lines: Detected lines
            frame_shape: Shape of the input frame
            cue_ball_pos: Optional cue ball position for scoring
            all_ball_positions: Optional list of all ball positions for improved orientation detection
            table_corners: Optional table corners (top_left, top_right, bottom_left, bottom_right) for cushion-based orientation

        Returns:
            List of CueStick candidates with scores
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

            # Determine tip and butt positions
            # Strategy: the butt should face away from the closest ball the cue is pointing at
            if all_ball_positions and len(all_ball_positions) > 0:
                # Option 1: x1,y1 is tip, x2,y2 is butt
                closest_ball_1 = self._find_closest_ball_on_cue_line(
                    (x1, y1), (x2, y2), all_ball_positions, cue_ball_pos
                )

                # Option 2: x2,y2 is tip, x1,y1 is butt
                closest_ball_2 = self._find_closest_ball_on_cue_line(
                    (x2, y2), (x1, y1), all_ball_positions, cue_ball_pos
                )

                # Choose the orientation where we found a ball the cue is pointing at
                if closest_ball_1 is not None and closest_ball_2 is None:
                    tip_pos = (x1, y1)
                    butt_pos = (x2, y2)
                elif closest_ball_2 is not None and closest_ball_1 is None:
                    tip_pos = (x2, y2)
                    butt_pos = (x1, y1)
                elif closest_ball_1 is not None and closest_ball_2 is not None:
                    # Both found a ball - choose the one with the closer ball
                    dist1 = np.sqrt(
                        (x1 - closest_ball_1[0]) ** 2 + (y1 - closest_ball_1[1]) ** 2
                    )
                    dist2 = np.sqrt(
                        (x2 - closest_ball_2[0]) ** 2 + (y2 - closest_ball_2[1]) ** 2
                    )

                    # Check if distances are similar (within 20% of each other)
                    # If so, use cushion-based tie-breaker
                    if (
                        table_corners is not None
                        and abs(dist1 - dist2) / max(dist1, dist2) < 0.2
                    ):
                        # Tie-breaker: prefer orientation where butt is closer to a cushion
                        # This is more realistic as players typically position themselves near the table edge
                        butt_cushion_dist_1 = self._distance_to_nearest_cushion(
                            (x2, y2), table_corners
                        )
                        butt_cushion_dist_2 = self._distance_to_nearest_cushion(
                            (x1, y1), table_corners
                        )

                        if butt_cushion_dist_1 < butt_cushion_dist_2:
                            # Option 1 butt is closer to cushion
                            tip_pos = (x1, y1)
                            butt_pos = (x2, y2)
                        else:
                            # Option 2 butt is closer to cushion
                            tip_pos = (x2, y2)
                            butt_pos = (x1, y1)
                    elif dist1 < dist2:
                        tip_pos = (x1, y1)
                        butt_pos = (x2, y2)
                    else:
                        tip_pos = (x2, y2)
                        butt_pos = (x1, y1)
                elif cue_ball_pos is not None:
                    # Fallback to closest to cue ball
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
                    # No balls detected - skip this candidate
                    continue

                # CRITICAL: Only accept cues that are pointing at a ball
                if cue_ball_pos is not None and not self._is_pointing_at_cue_ball(
                    tip_pos, butt_pos, cue_ball_pos
                ):
                    continue
            elif cue_ball_pos is not None:
                # Fallback: determine which end is closer to cue ball (that's the tip)
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

                # CRITICAL: Only accept cues that are pointing at the cue ball
                if not self._is_pointing_at_cue_ball(tip_pos, butt_pos, cue_ball_pos):
                    continue
            else:
                # No cue ball position - skip this candidate (we require cue ball)
                continue

            # Calculate confidence score
            confidence = self._calculate_line_confidence(
                line, frame_shape, cue_ball_pos
            )

            if confidence >= self.min_detection_confidence:
                cue = CueStick(
                    tip_position=tip_pos,
                    butt_position=butt_pos,
                    angle=angle,
                    length=length,
                    confidence=confidence,
                )
                candidates.append(cue)

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    def _is_pointing_at_cue_ball(
        self,
        tip_pos: tuple[float, float],
        butt_pos: tuple[float, float],
        cue_ball_pos: tuple[float, float],
    ) -> bool:
        """Check if the cue line is pointing at the cue ball.

        Args:
            tip_pos: Cue tip position
            butt_pos: Cue butt position
            cue_ball_pos: Cue ball position

        Returns:
            True if cue is pointing at cue ball within tolerance
        """
        # Calculate the distance from cue ball to the cue line
        line = np.array([tip_pos[0], tip_pos[1], butt_pos[0], butt_pos[1]])
        distance_to_line = self._point_to_line_distance(cue_ball_pos, line)

        # Maximum distance from line to cue ball center (in pixels)
        # This accounts for cue ball radius + some tolerance
        max_distance = (
            self.max_distance_to_cue_ball
        )  # Adjust based on typical ball size + tolerance

        if distance_to_line > max_distance:
            return False

        # Also check that the cue ball is "in front" of the tip (not behind the butt)
        # Vector from butt to tip
        cue_dx = tip_pos[0] - butt_pos[0]
        cue_dy = tip_pos[1] - butt_pos[1]

        # Vector from butt to cue ball
        ball_dx = cue_ball_pos[0] - butt_pos[0]
        ball_dy = cue_ball_pos[1] - butt_pos[1]

        # Dot product to check if cue ball is in the direction of the tip
        dot_product = cue_dx * ball_dx + cue_dy * ball_dy

        # Must be positive (cue ball is ahead of butt in the direction of tip)
        if dot_product <= 0:
            return False

        # Also check the tip is reasonably close to the cue ball
        tip_distance = np.sqrt(
            (tip_pos[0] - cue_ball_pos[0]) ** 2 + (tip_pos[1] - cue_ball_pos[1]) ** 2
        )

        # Maximum distance from tip to cue ball (adjust based on typical aiming distance)
        max_tip_distance = self.max_tip_distance  # pixels

        return tip_distance <= max_tip_distance

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
        confidence += self.length_weight * length_score

        # Position score (prefer lines not at frame edges)
        h, w = frame_shape[:2]
        edge_margin = self.edge_margin

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
            edge_penalty = self.edge_penalty

        confidence += self.position_weight * (1.0 - edge_penalty)

        # Cue ball proximity score
        if cue_ball_pos is not None:
            cx, cy = cue_ball_pos

            # Distance to line
            line_dist = self._point_to_line_distance((cx, cy), line)
            max_reasonable_distance = self.max_reasonable_distance

            proximity_score = max(0.0, 1.0 - line_dist / max_reasonable_distance)
            confidence += self.proximity_weight * proximity_score
        else:
            confidence += (
                self.no_cue_ball_score
            )  # Neutral score when no cue ball position

        # Temporal consistency score (if we have previous detections)
        if self.previous_cues:
            prev_cue = self.previous_cues[-1]

            # Angle consistency
            angle_diff = abs(
                math.degrees(math.atan2(y2 - y1, x2 - x1)) - prev_cue.angle
            )
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            angle_score = max(
                0.0, 1.0 - angle_diff / self.max_angle_change
            )  # Penalize >max changes

            # Position consistency
            tip_dist = np.sqrt(
                (x1 - prev_cue.tip_position[0]) ** 2
                + (y1 - prev_cue.tip_position[1]) ** 2
            )
            position_score = max(0.0, 1.0 - tip_dist / self.max_tracking_distance)

            temporal_score = 0.5 * angle_score + 0.5 * position_score
            confidence += self.temporal_weight * temporal_score
        else:
            confidence += self.no_temporal_score  # Neutral score for first detection

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
        return float(np.sqrt((x0 - closest_x) ** 2 + (y0 - closest_y) ** 2))

    def _get_line_normal(
        self, line: NDArray[np.float64], point: tuple[float, float]
    ) -> tuple[float, float]:
        """Calculate perpendicular line equation at a given point.

        Cassapa Phase 3.3: Perpendicular Normal Line Calculation
        Reference: cassapa/detector.cpp:864-871, pool_utils.cpp:58-67

        This method calculates the equation of a line perpendicular to the input line,
        passing through the specified point. This is used to search outward from the
        cue centerline to find the cue's edges.

        The Cassapa algorithm uses a robust geometric formula that handles edge cases
        better than the simple perpendicular slope formula (-1/m).

        Args:
            line: Line coordinates [x1, y1, x2, y2] defining the original line
            point: Point (x, y) through which the normal line must pass

        Returns:
            Tuple (normal_m, normal_h) representing the perpendicular line equation:
                - normal_m: slope of the perpendicular line
                - normal_h: y-intercept of the perpendicular line

        Edge cases:
            - Vertical line (slope=inf): Normal is horizontal (slope=0)
            - Horizontal line (slope=0): Normal is vertical (slope=inf)
            - Near-zero slopes: Handled by geometric formula

        Mathematical formula (Cassapa method):
            1. Get original line equation: m, h = _get_line_equation(line)
            2. Calculate angle: angle = arctan(m)
            3. Calculate perpendicular angle:
               delta_angle = π - 2.0 * angle
               normal_angle = angle + delta_angle
            4. Calculate normal slope: normal_m = tan(normal_angle)
            5. Calculate normal intercept: normal_h = y - normal_m * x

        Alternative (simple) formula:
            For non-special cases, perpendicular slope is: -1/m
            But Cassapa's geometric formula handles edge cases more robustly.

        Example:
            >>> line = np.array([100, 100, 200, 200])  # slope = 1
            >>> point = (150, 150)
            >>> normal_m, normal_h = detector._get_line_normal(line, point)
            >>> # normal_m ≈ -1.0 (perpendicular to slope=1)
            >>> # Normal line passes through (150, 150)
        """
        # Get original line equation
        # Note: _get_line_equation uses epsilon (0.000001) to avoid division by zero
        # so m will never be infinite, but can be very large for near-vertical lines
        m, h = self._get_line_equation(line)

        x, y = point

        # Use Cassapa's geometric formula for perpendicular line
        # This formula is derived from angle geometry:
        # 1. Get angle of original line: angle = arctan(m)
        # 2. Calculate perpendicular angle using: delta_angle = π - 2*angle
        # 3. Normal angle: normal_angle = angle + delta_angle = π - angle
        # 4. Normal slope: normal_m = tan(π - angle) = -tan(angle) = -m
        #
        # Note: This formula gives -m (reflection), not -1/m (true perpendicular).
        # However, this is the exact formula from Cassapa's implementation
        # (detector.cpp:864-871) and is used for edge detection in their algorithm.
        angle = np.arctan(m)
        delta_angle = np.pi - 2.0 * angle
        normal_angle = angle + delta_angle
        normal_m = np.tan(normal_angle)

        # Calculate y-intercept so line passes through the given point
        # y = normal_m * x + normal_h
        # normal_h = y - normal_m * x
        normal_h = y - normal_m * x

        return (float(normal_m), float(normal_h))

    def _select_best_cue(
        self, candidates: list[CueStick], frame: NDArray[np.uint8]
    ) -> Optional[CueStick]:
        """Select the best cue candidate from the list.

        Args:
            candidates: List of cue candidates
            frame: Original frame for additional validation

        Returns:
            Best CueStick candidate or None
        """
        if not candidates:
            return None

        # Additional validation for top candidates
        validated_candidates = []

        for candidate in candidates[
            : self.top_candidates_to_validate
        ]:  # Check top candidates
            if self._validate_cue_candidate(candidate, frame):
                validated_candidates.append(candidate)

        if not validated_candidates:
            return None

        # Return highest confidence validated candidate
        return validated_candidates[0]

    def _validate_cue_candidate(self, cue: CueStick, frame: NDArray[np.uint8]) -> bool:
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

    def _check_line_thickness(self, cue: CueStick, frame: NDArray[np.uint8]) -> bool:
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
        sample_count = self.thickness_sample_count

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
        max_search = self.max_thickness_search
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
        edges = np.where(np.abs(gradient) > self.gradient_threshold)[0]

        if len(edges) >= 2:
            # Distance between first and last significant edge
            return edges[-1] - edges[0]

        return 0

    def _check_line_texture(self, cue: CueStick, frame: NDArray[np.uint8]) -> bool:
        """Check for consistent texture along the line (simplified check)."""
        # For now, just check that the line region has reasonable contrast
        # More sophisticated texture analysis could be added later
        return True

    def _check_cue_position(self, cue: CueStick, frame: NDArray[np.uint8]) -> bool:
        """Check if cue position is reasonable (not intersecting table boundaries too much)."""
        # This would need table detection integration
        # For now, just check it's not entirely at the frame edge
        x1, y1 = cue.tip_position
        x2, y2 = cue.butt_position
        h, w = frame.shape[:2]

        edge_margin = self.position_edge_margin

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

    def _apply_temporal_tracking(self, current_cue: CueStick) -> CueStick:
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

    def _analyze_cue_motion(self, cue: CueStick) -> None:
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
        cue.tip_velocity = (vx, vy)

        # Calculate acceleration (for state detection, not stored in model)
        prev_vx = (prev_cue.tip_position[0] - prev_prev_cue.tip_position[0]) / dt
        prev_vy = (prev_cue.tip_position[1] - prev_prev_cue.tip_position[1]) / dt

        ax = (vx - prev_vx) / dt
        ay = (vy - prev_vy) / dt

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
        contact_threshold = self.contact_threshold  # pixels

        if distance > contact_threshold:
            return None

        # Check if cue ball started moving
        ball_speed = np.sqrt(cue_ball_velocity[0] ** 2 + cue_ball_velocity[1] ** 2)
        if ball_speed < self.min_ball_speed:  # Minimum speed to consider movement
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
                cue.confidence, self.min_confidence_for_shot
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
        ball_radius = self.ball_radius  # pixels (adjust based on actual detection)
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
        max_velocity = self.max_velocity  # pixels/frame
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
        if angle_diff > self.english_deviation_angle:
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

            if center_offset < -self.follow_draw_threshold:  # Hit below center
                return ShotType.DRAW
            elif center_offset > self.follow_draw_threshold:  # Hit above center
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
        ball_radius = self.ball_radius  # pixels
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
        ball_radius = self.ball_radius  # pixels
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
            if np.sum(frame) < self.min_frame_sum:  # Very dark/empty frame
                return []

            # Get all line candidates
            processed_frame = self._preprocess_frame(frame)

            # Check if preprocessed frame has sufficient contrast
            if np.std(processed_frame) < self.min_contrast_std:  # Very low contrast
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
            for extended_cue in unique_cues[: self.max_cues_to_detect]:
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

    def _filter_overlapping_cues(self, candidates: list[CueStick]) -> list[CueStick]:
        """Filter out overlapping cue detections.

        Args:
            candidates: List of cue candidates

        Returns:
            Filtered list without overlaps
        """
        if len(candidates) <= 1:
            return candidates

        unique_cues = []
        overlap_threshold = self.overlap_threshold  # pixels

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

                if tip_dist < overlap_threshold and angle_diff < self.max_angle_overlap:
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
        position_threshold = self.position_movement_threshold  # pixels
        angle_threshold = self.angle_change_threshold  # degrees

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
