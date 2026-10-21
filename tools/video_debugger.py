#!/usr/bin/env python3
"""Video debugging tool for ball tracking and trajectory prediction.

This tool allows you to play back video files (MKV/MP4) with ball detection,
tracking, cue stick detection, and predicted trajectory overlaid on the frames.

Uses the VisionModule from backend for all detection/tracking logic.

Usage:
    python tools/video_debugger.py <video_file> [--loop] [--log-level LEVEL]

Examples:
    # Play video with default settings
    python tools/video_debugger.py demo.mkv

    # Loop video playback
    python tools/video_debugger.py demo.mkv --loop

    # Enable debug logging
    python tools/video_debugger.py demo.mkv --log-level DEBUG

Controls:
    SPACE - Play/Pause
    RIGHT - Step forward one frame (when paused)
    LEFT  - Step backward one frame (when paused)
    R     - Reset to beginning
    +/-   - Increase/decrease playback speed
    T     - Toggle trajectory prediction
    C     - Toggle cue stick display
    D     - Toggle detection circles
    I     - Toggle track IDs
    P     - Toggle playing area display
    A     - Re-detect table playing area
    Q/ESC - Quit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

# Configure logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections import deque

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend components
from backend.core.models import BallState, CueState, TableState, Vector2D
from backend.core.physics.trajectory import (
    MultiballTrajectoryResult,
    TrajectoryCalculator,
    TrajectoryQuality,
)
from backend.vision import VisionModule
from backend.vision.models import CueState as VisionCueState

try:
    from .coordinate_scaling import (
        CANONICAL_RESOLUTION,
        CoordinateMetadata,
        canonical_point_to_resolution,
        point_for_frame,
        scalar_extent_for_frame,
    )
except ImportError:  # Direct ``python tools/video_debugger.py`` execution.
    from coordinate_scaling import (
        CANONICAL_RESOLUTION,
        CoordinateMetadata,
        canonical_point_to_resolution,
        point_for_frame,
        scalar_extent_for_frame,
    )


class VideoDebugger:
    """Video playback debugger with ball tracking visualization."""

    def __init__(
        self,
        video_path: str,
        loop: bool = False,
        max_trace_length: int = 100,
        detection_backend: str = "yolo",
        yolo_model_path: str | None = None,
        yolo_device: str = "cpu",
    ):
        """Initialize video debugger.

        Args:
            video_path: Path to video file
            loop: Whether to loop the video
            max_trace_length: Maximum number of trace points per ball
            detection_backend: Backend to use ("yolo" or "opencv")
            yolo_model_path: Path to YOLO model (for YOLO backend)
            yolo_device: Device to use for YOLO ("cpu", "cuda", or "tpu")
        """
        self.video_path = video_path
        self.loop = loop
        self.max_trace_length = max_trace_length

        # Playback state
        self.paused = False
        self.playback_speed = 1.0
        self.current_frame_num = 0

        # Visualization toggles
        self.show_trajectory = True
        self.show_detections = True
        self.show_track_ids = True
        self.show_cue = True
        self.show_playing_area = True

        # Tracking data
        self.ball_traces: dict[int, deque] = {}  # track_id -> deque of positions
        self.current_tracked_balls = []  # Store current frame's tracked balls for HUD

        # Initialize VisionModule using backend config system
        # VisionModule will load all settings from backend/config/default.json
        # The video path is passed as camera_device_id override
        logger.info(f"Initializing video debugger for: {video_path}")
        logger.info("Using backend config system for all vision settings")

        # Initialize VisionModule with minimal overrides
        # All other settings come from backend config
        vision_config = {
            "camera_device_id": video_path,
            "enable_threading": False,  # Disable threading for video file
        }

        # Initialize VisionModule
        self.vision_module = VisionModule(vision_config)

        # Initialize trajectory calculator
        self.trajectory_calculator = TrajectoryCalculator()

        # Table configuration - load from saved config if available
        self._canonical_playing_area_corners: list[Vector2D] | None = None
        self._table_dimensions_real: tuple[float, float] | None = None
        self.table_state = self._load_table_state_from_config()

        # Current trajectory prediction
        self.current_trajectory = None

        logger.info("Video debugger initialized")

    def _load_table_state_from_config(self) -> TableState | None:
        """Load table state from backend configuration system.

        Returns:
            A calibrated table with explicit source scales, or None when no
            valid calibration is configured.
        """
        try:
            from backend.config import config

            # Extract table config from backend config system
            playing_area_corners_data = config.get("table.playing_area_corners")
            table_dimensions = config.get(
                "vision.calibration.camera.table_dimensions", {}
            )
            if isinstance(table_dimensions, dict):
                table_width = table_dimensions.get("standard_width_meters")
                table_height = table_dimensions.get("standard_height_meters")
                if table_width is not None and table_height is not None:
                    self._table_dimensions_real = (
                        float(table_width),
                        float(table_height),
                    )

            if not playing_area_corners_data or len(playing_area_corners_data) != 4:
                logger.warning("No valid playing_area_corners in backend config")
                self._canonical_playing_area_corners = None
                return None

            # Each point's explicit scale identifies its source pixels. Do not
            # infer it from an image-size setting: saved calibration points may
            # have different scales and remain valid after camera reconfiguration.
            playing_area_corners = [
                Vector2D(
                    corner["x"],
                    corner["y"],
                    scale=tuple(corner["scale"]),
                ).to_4k_canonical()
                for corner in playing_area_corners_data
            ]
            self._canonical_playing_area_corners = playing_area_corners.copy()

            # Build table extents and pockets from the canonicalized boundary.
            table_state = TableState.from_calibration(
                [(corner.x, corner.y) for corner in playing_area_corners],
                CANONICAL_RESOLUTION,
                table_dimensions_real=self._table_dimensions_real,
            )

            logger.info("Loaded scale-aware playing area corners from backend config:")
            for i, corner in enumerate(playing_area_corners):
                logger.info(f"  Corner {i+1}: ({corner.x:.1f}, {corner.y:.1f})")

            return table_state

        except Exception as e:
            logger.warning(f"Failed to load table config from backend: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            self._canonical_playing_area_corners = None
            return None

    def draw_ball(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        radius: float,
        color: tuple,
        label: str = "",
    ) -> None:
        """Draw a ball detection on the frame.

        Args:
            frame: Frame to draw on
            x, y: Ball center position
            radius: Ball radius
            color: BGR color tuple
            label: Optional label text
        """
        center = (int(x), int(y))
        r = int(radius)

        # Draw circle
        cv2.circle(frame, center, r, color, 2)

        # Draw center point
        cv2.circle(frame, center, 3, color, -1)

        # Draw label if provided
        if label:
            label_pos = (center[0] - r, center[1] - r - 5)
            cv2.putText(
                frame,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def draw_cue_stick(
        self, frame: np.ndarray, cue_stick, color: tuple = (0, 255, 255)
    ) -> None:
        """Draw cue stick on frame.

        Args:
            frame: Frame to draw on
            cue_stick: CueStick object
            color: BGR color tuple
        """
        if cue_stick is None:
            return

        frame_resolution = (int(frame.shape[1]), int(frame.shape[0]))
        (tip_x_canonical, tip_y_canonical), (
            butt_x_canonical,
            butt_y_canonical,
        ) = self._cue_endpoints_canonical(cue_stick)
        tip_x, tip_y = canonical_point_to_resolution(
            tip_x_canonical, tip_y_canonical, frame_resolution
        )
        butt_x, butt_y = canonical_point_to_resolution(
            butt_x_canonical, butt_y_canonical, frame_resolution
        )
        tip = (int(tip_x), int(tip_y))
        butt = (int(butt_x), int(butt_y))

        # Draw cue stick line
        cv2.line(frame, tip, butt, color, 3)

        # Draw tip marker
        cv2.circle(frame, tip, 6, color, -1)

        # Draw state indicator
        state_color = (
            (0, 255, 0) if cue_stick.state == VisionCueState.AIMING else (0, 0, 255)
        )
        cv2.circle(frame, tip, 10, state_color, 2)

    @staticmethod
    def _cue_endpoints_canonical(
        cue_stick,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return cue tip and butt in canonical pixels using declared geometry."""
        metadata = CoordinateMetadata.for_values(
            cue_stick.coordinate_space, cue_stick.source_resolution
        )
        tip = metadata.to_canonical(
            cue_stick.tip_position[0], cue_stick.tip_position[1]
        )
        if cue_stick.butt_position != (0.0, 0.0):
            butt = metadata.to_canonical(
                cue_stick.butt_position[0], cue_stick.butt_position[1]
            )
        else:
            # Length and angle are expressed in the cue's declared coordinate
            # space. Scale vector components independently for non-square pixels.
            angle_rad = np.radians(cue_stick.angle)
            scale_x, scale_y = metadata.coordinate_scale
            if metadata.coordinate_space == "4k_canonical":
                # Canonical cue length uses the runtime's X-axis scalar length
                # convention. Its source resolution is provenance for the
                # detector angle, so adjust the Y component when that source
                # did not have the canonical aspect ratio.
                if cue_stick.source_resolution is not None:
                    source_metadata = CoordinateMetadata.for_values(
                        "pixel", cue_stick.source_resolution
                    )
                    scale_y /= source_metadata.coordinate_scale[0]
                    scale_y *= source_metadata.coordinate_scale[1]
            butt = (
                tip[0] - cue_stick.length * scale_x * np.cos(angle_rad),
                tip[1] - cue_stick.length * scale_y * np.sin(angle_rad),
            )
        return tip, butt

    def draw_playing_area(self, frame: np.ndarray) -> None:
        """Draw the playing area polygon on the frame.

        Args:
            frame: Frame to draw on
        """
        corners = self._canonical_playing_area_corners
        if corners is None or len(corners) != 4:
            return

        frame_resolution = (int(frame.shape[1]), int(frame.shape[0]))
        # Keep canonical physics geometry intact and convert only the display copy.
        points = np.array(
            [
                [
                    int(x),
                    int(y),
                ]
                for x, y in (
                    canonical_point_to_resolution(
                        corner.to_4k_canonical().x,
                        corner.to_4k_canonical().y,
                        frame_resolution,
                    )
                    for corner in corners
                )
            ],
            dtype=np.int32,
        )

        # Draw polygon outline
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw corner markers
        for i, point in enumerate(points):
            center = (int(point[0]), int(point[1]))
            # Draw circle
            cv2.circle(frame, center, 8, (0, 255, 0), -1)
            # Draw corner number
            cv2.putText(
                frame,
                str(i + 1),
                (center[0] - 10, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Draw label
        if corners:
            label_pos = (10, frame.shape[0] - 60)
            cv2.putText(
                frame,
                "Playing Area (Press 'A' to re-detect)",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    def draw_trajectory(self, frame: np.ndarray, multiball_result) -> None:
        """Draw predicted trajectories for all balls on frame.

        Based on cassapa projector_preview.cpp DrawAids() lines 227-266.
        Renders trajectory as line segments with circles at endpoints.

        Args:
            frame: Frame to draw on
            multiball_result: MultiballTrajectoryResult with multiple ball trajectories
        """
        if multiball_result is None:
            return
        frame_resolution = (int(frame.shape[1]), int(frame.shape[0]))

        # Define colors matching cassapa:
        # White ball (cue) = white (255, 255, 255)
        # Hit ball = yellow (0, 255, 255) in BGR
        cue_color = (255, 255, 255)  # White for cue ball
        hit_color = (0, 255, 255)  # Yellow for first hit ball

        # Get primary (cue) ball trajectory
        primary_id = multiball_result.primary_ball_id
        if primary_id not in multiball_result.trajectories:
            return

        cue_trajectory = multiball_result.trajectories[primary_id]

        # Draw cue ball trajectory (white lines)
        if cue_trajectory.points and len(cue_trajectory.points) >= 2:
            # Draw line segments for each trajectory piece
            for i in range(len(cue_trajectory.points) - 1):
                p1 = cue_trajectory.points[i]
                p2 = cue_trajectory.points[i + 1]

                point1 = p1.position.to_resolution(frame_resolution)
                point2 = p2.position.to_resolution(frame_resolution)
                pt1 = (int(point1.x), int(point1.y))
                pt2 = (int(point2.x), int(point2.y))

                # Draw the line segment
                cv2.line(frame, pt1, pt2, cue_color, 4)

                # Draw circle at endpoint (shows ball position at end of segment)
                ball_radius = int(
                    scalar_extent_for_frame(
                        cue_trajectory.initial_state.radius,
                        "4k_canonical",
                        None,
                        frame_resolution,
                    )
                )
                cv2.circle(frame, pt2, ball_radius, cue_color, 2)

        # Find the first hit ball (if any) and draw its trajectory
        hit_ball_id = None
        for collision in cue_trajectory.collisions:
            if collision.type.value == "ball_ball" and collision.ball2_id:
                hit_ball_id = collision.ball2_id
                break

        if hit_ball_id and hit_ball_id in multiball_result.trajectories:
            hit_trajectory = multiball_result.trajectories[hit_ball_id]

            if hit_trajectory.points and len(hit_trajectory.points) >= 2:
                # Draw hit ball trajectory (yellow lines)
                for i in range(len(hit_trajectory.points) - 1):
                    p1 = hit_trajectory.points[i]
                    p2 = hit_trajectory.points[i + 1]

                    point1 = p1.position.to_resolution(frame_resolution)
                    point2 = p2.position.to_resolution(frame_resolution)
                    pt1 = (int(point1.x), int(point1.y))
                    pt2 = (int(point2.x), int(point2.y))

                    cv2.line(frame, pt1, pt2, hit_color, 4)

                # Draw white circle at collision point where cue ball hits target ball
                if cue_trajectory.points:
                    collision_point = cue_trajectory.points[-1].position
                    ball_radius = int(
                        scalar_extent_for_frame(
                            cue_trajectory.initial_state.radius,
                            "4k_canonical",
                            None,
                            frame_resolution,
                        )
                    )
                    collision_point_frame = collision_point.to_resolution(
                        frame_resolution
                    )
                    cv2.circle(
                        frame,
                        (int(collision_point_frame.x), int(collision_point_frame.y)),
                        ball_radius,
                        cue_color,
                        2,
                    )

                # Draw yellow circle at end position of hit ball
                if hit_trajectory.points:
                    final_point = hit_trajectory.points[-1].position
                    ball_radius = int(
                        scalar_extent_for_frame(
                            hit_trajectory.initial_state.radius,
                            "4k_canonical",
                            None,
                            frame_resolution,
                        )
                    )
                    final_point_frame = final_point.to_resolution(frame_resolution)
                    cv2.circle(
                        frame,
                        (int(final_point_frame.x), int(final_point_frame.y)),
                        ball_radius,
                        hit_color,
                        2,
                    )

    def _find_ball_cue_is_pointing_at(
        self, cue, balls, frame_resolution: tuple[int, int]
    ):
        """Find which ball the cue is currently pointing at.

        Args:
            cue: Detected cue stick with tip and butt positions
            balls: List of detected balls

        Returns:
            The ball the cue is pointing at, or None
        """
        if not cue or not balls:
            return None

        (tip_x, tip_y), (butt_x, butt_y) = self._cue_endpoints_canonical(cue)

        # Calculate cue direction vector
        cue_dx = tip_x - butt_x
        cue_dy = tip_y - butt_y
        cue_length = np.sqrt(cue_dx**2 + cue_dy**2)

        if cue_length == 0:
            return None

        # Normalize direction
        cue_dx /= cue_length
        cue_dy /= cue_length

        # Find the closest ball along the cue direction
        closest_ball = None
        min_distance = float("inf")
        # Preserve the historical 40 source-pixel tolerance while comparing
        # canonical points. The actual frame dimensions declare that source.
        max_perpendicular_distance = scalar_extent_for_frame(
            40,
            "pixel",
            frame_resolution,
            CANONICAL_RESOLUTION,
        )
        cue_metadata = CoordinateMetadata.for_values(
            cue.coordinate_space, cue.source_resolution
        )

        for ball in balls:
            # Vector from cue tip to ball center
            ball_metadata = CoordinateMetadata.for_values(
                ball.coordinate_space, ball.source_resolution
            )
            ball_x, ball_y = ball_metadata.to_canonical(
                ball.position[0], ball.position[1]
            )
            cue_tip_x, cue_tip_y = cue_metadata.to_canonical(
                cue.tip_position[0], cue.tip_position[1]
            )
            ball_dx = ball_x - cue_tip_x
            ball_dy = ball_y - cue_tip_y

            # Distance along cue direction (projection)
            distance_along_cue = ball_dx * cue_dx + ball_dy * cue_dy

            # Skip balls behind the cue tip
            if distance_along_cue < 0:
                continue

            # Calculate perpendicular distance from cue line to ball center
            perpendicular_distance = abs(ball_dx * cue_dy - ball_dy * cue_dx)

            # Check if ball is within tolerance and closer than current closest
            if (
                perpendicular_distance < max_perpendicular_distance
                and distance_along_cue < min_distance
            ):
                min_distance = distance_along_cue
                closest_ball = ball

        return closest_ball

    def draw_hud(self, frame: np.ndarray, detection_backend: str) -> None:
        """Draw heads-up display with info and controls.

        Args:
            frame: Frame to draw on
            detection_backend: Detection backend name for display
        """
        h, w = frame.shape[:2]

        # Draw semi-transparent background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Display status info
        status_text = [
            f"Frame: {self.current_frame_num}",
            f"Speed: {self.playback_speed:.1f}x",
            f"State: {'PAUSED' if self.paused else 'PLAYING'}",
            f"Backend: {detection_backend.upper()}",
            f"Balls: {len(self.current_tracked_balls)}",
        ]

        y_offset = 20
        for text in status_text:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        # Display controls
        controls_text = [
            "SPACE: Play/Pause | LEFT/RIGHT: Step | +/-: Speed | A: Auto-detect Table",
            "T: Trajectory | D: Detections | I: IDs | C: Cue | P: Playing Area | R: Reset | Q: Quit",
        ]

        y_offset = h - 40
        for text in controls_text:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            y_offset += 15

        # Draw toggle indicators
        toggle_y = 20
        toggle_x = w - 150
        toggles = [
            ("Trajectory", self.show_trajectory),
            ("Detections", self.show_detections),
            ("IDs", self.show_track_ids),
            ("Cue", self.show_cue),
            ("Playing Area", self.show_playing_area),
        ]

        for name, enabled in toggles:
            color = (0, 255, 0) if enabled else (0, 0, 255)
            text = f"{name}: {'ON' if enabled else 'OFF'}"
            cv2.putText(
                frame,
                text,
                (toggle_x, toggle_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            toggle_y += 15

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with detection, tracking, and visualization.

        This is now a thin rendering layer - all detection/tracking is done by VisionModule.

        Args:
            frame: Input frame

        Returns:
            Frame with visualizations drawn
        """
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        # Create a copy for drawing
        display_frame = frame.copy()

        # Use VisionModule to process frame - this handles all detection/tracking
        detection_result = self.vision_module._process_single_frame(
            frame, self.current_frame_num, None
        )

        if detection_result is None:
            logger.warning(f"Frame {self.current_frame_num}: No detection result")
            return display_frame

        # Extract results from VisionModule
        detected_balls = detection_result.balls
        detected_cue = detection_result.cue

        # Debug logging every 30 frames
        if self.current_frame_num % 30 == 0:
            logger.info(
                f"Frame {self.current_frame_num}: {len(detected_balls)} balls, cue={'detected' if detected_cue else 'not detected'}"
            )

        # Store current tracked balls for HUD display
        self.current_tracked_balls = detected_balls

        # Calculate trajectory if cue is detected
        if detected_cue and detected_balls and self.table_state is not None:
            try:
                # Find which ball the cue is pointing at
                target_ball = self._find_ball_cue_is_pointing_at(
                    detected_cue,
                    detected_balls,
                    (frame_width, frame_height),
                )

                if target_ball is None:
                    self.current_trajectory = None
                else:
                    # Create CueState for trajectory calculation
                    cue_metadata = CoordinateMetadata.for_values(
                        detected_cue.coordinate_space,
                        detected_cue.source_resolution,
                    )
                    cue_tip_canonical = cue_metadata.to_canonical(
                        detected_cue.tip_position[0], detected_cue.tip_position[1]
                    )
                    target_metadata = CoordinateMetadata.for_values(
                        target_ball.coordinate_space,
                        target_ball.source_resolution,
                    )
                    target_position_canonical = target_metadata.to_canonical(
                        target_ball.position[0], target_ball.position[1]
                    )
                    target_radius_canonical = (
                        target_ball.radius * target_metadata.coordinate_scale[0]
                    )
                    cue_state = CueState(
                        angle=detected_cue.angle,
                        estimated_force=5.0,  # Default moderate force
                        impact_point=Vector2D.from_4k(*cue_tip_canonical),
                        tip_position=Vector2D.from_4k(*cue_tip_canonical),
                        elevation=getattr(detected_cue, "elevation", 0.0),
                        is_visible=True,
                        confidence=detected_cue.confidence,
                        last_update=0,
                    )

                    # Create BallState for the ball the cue is pointing at
                    ball_state = BallState(
                        id="target_ball",
                        position=Vector2D.from_4k(*target_position_canonical),
                        velocity=Vector2D.from_4k(0, 0),
                        radius=target_radius_canonical,
                        mass=0.17,  # Standard pool ball mass in kg
                        is_cue_ball=False,
                        confidence=getattr(target_ball, "confidence", 1.0),
                        last_update=0,
                    )

                    # Create BallState objects for other balls
                    other_ball_states = []
                    for i, ball in enumerate(detected_balls):
                        if ball != target_ball:
                            ball_metadata = CoordinateMetadata.for_values(
                                ball.coordinate_space, ball.source_resolution
                            )
                            ball_position_canonical = ball_metadata.to_canonical(
                                ball.position[0], ball.position[1]
                            )
                            ball_radius_canonical = (
                                ball.radius * ball_metadata.coordinate_scale[0]
                            )
                            other_ball_states.append(
                                BallState(
                                    id=f"ball_{i}",
                                    position=Vector2D.from_4k(*ball_position_canonical),
                                    velocity=Vector2D.from_4k(0, 0),
                                    radius=ball_radius_canonical,
                                    mass=0.17,
                                    confidence=getattr(ball, "confidence", 1.0),
                                    last_update=0,
                                )
                            )

                    # Calculate multiball trajectory using backend calculator
                    self.current_trajectory = (
                        self.trajectory_calculator.predict_multiball_cue_shot(
                            cue_state,
                            ball_state,
                            self.table_state,
                            other_ball_states,
                            quality=TrajectoryQuality.LOW,
                            max_collision_depth=5,
                        )
                    )
            except Exception as e:
                if self.current_frame_num % 30 == 0:
                    logger.error(f"Trajectory calculation failed: {e}")
                self.current_trajectory = None
        else:
            self.current_trajectory = None

        # Draw playing area first (so it's behind other overlays)
        if self.show_playing_area:
            self.draw_playing_area(display_frame)

        # Draw trajectory prediction
        if self.show_trajectory and self.current_trajectory:
            self.draw_trajectory(display_frame, self.current_trajectory)

        # Draw cue stick
        if self.show_cue and detected_cue:
            self.draw_cue_stick(display_frame, detected_cue)

        # Draw ball detections
        for i, ball in enumerate(detected_balls):
            x, y = point_for_frame(
                ball.position[0],
                ball.position[1],
                ball.coordinate_space,
                ball.source_resolution,
                (frame_width, frame_height),
            )
            radius = scalar_extent_for_frame(
                ball.radius,
                ball.coordinate_space,
                ball.source_resolution,
                (frame_width, frame_height),
            )
            track_id = getattr(ball, "track_id", None)

            # Use track_id for tracked balls, or index for raw detections
            ball_id = track_id if track_id is not None else i

            # Choose color based on ball type
            if hasattr(ball, "ball_type") and ball.ball_type:
                # Color by ball type
                type_colors = {
                    "CUE": (255, 255, 255),
                    "SOLID": (0, 0, 255),
                    "STRIPE": (0, 255, 255),
                    "EIGHT": (0, 0, 0),
                }
                color = type_colors.get(str(ball.ball_type), (255, 0, 255))
            else:
                # Use unique color per ball_id
                hue = (ball_id * 30) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[
                    0
                ][0]
                color = tuple(int(c) for c in color)

            # Draw detection
            if self.show_detections:
                label = ""
                if self.show_track_ids and track_id is not None:
                    # Use display_name if available, otherwise fall back to track_id
                    if hasattr(ball, "display_name"):
                        label = ball.display_name
                    else:
                        label = f"ID:{track_id}"
                    if getattr(ball, "is_moving", False):
                        label += " *"
                elif track_id is None:
                    label = f"#{i}"  # Show detection index for untracked balls

                self.draw_ball(display_frame, x, y, radius, color, label)

        # Draw HUD
        detection_backend = getattr(
            self.vision_module.config, "detection_backend", "yolo"
        )
        self.draw_hud(display_frame, detection_backend)

        return display_frame

    def run(self) -> None:
        """Run the video debugger main loop."""
        logger.info("Starting video debugger...")

        # Start camera capture via VisionModule
        if not self.vision_module.start_capture():
            logger.error("Failed to start video capture")
            return

        # Give capture thread time to fill buffer
        import time

        time.sleep(0.5)

        try:
            window_name = "Video Debugger - Ball Tracking"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            frame_count = 0
            no_frame_count = 0

            while True:
                # Get frame if not paused
                if not self.paused:
                    frame_data = self.vision_module.camera.get_latest_frame()

                    if frame_data is not None:
                        no_frame_count = 0  # Reset counter
                        frame, frame_info = frame_data
                        self.current_frame_num = frame_info.frame_number

                        # Process and display frame
                        display_frame = self.process_frame(frame)
                        cv2.imshow(window_name, display_frame)

                        # Calculate delay based on playback speed
                        delay = int(33 / self.playback_speed)  # ~30 FPS base
                        frame_count += 1
                    else:
                        # No frame available
                        no_frame_count += 1

                        # Only exit if we've had multiple consecutive failures
                        if no_frame_count > 10:
                            if not self.loop:
                                logger.info(
                                    f"Video playback completed (processed {frame_count} frames)"
                                )
                                break
                        delay = 10
                else:
                    # Paused - just wait for key
                    delay = 100

                # Handle keyboard input
                key = cv2.waitKey(delay) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    logger.info("Quit requested")
                    break
                elif key == ord(" "):  # SPACE
                    self.paused = not self.paused
                    logger.info(f"Playback {'paused' if self.paused else 'resumed'}")
                elif key == ord("t"):  # Toggle trajectory
                    self.show_trajectory = not self.show_trajectory
                    logger.info(f"Trajectory: {self.show_trajectory}")
                elif key == ord("c"):  # Toggle cue
                    self.show_cue = not self.show_cue
                    logger.info(f"Cue: {self.show_cue}")
                elif key == ord("d"):  # Toggle detections
                    self.show_detections = not self.show_detections
                    logger.info(f"Detections: {self.show_detections}")
                elif key == ord("i"):  # Toggle IDs
                    self.show_track_ids = not self.show_track_ids
                    logger.info(f"Track IDs: {self.show_track_ids}")
                elif key == ord("r"):  # Reset
                    logger.info("Resetting video...")
                    self.vision_module.stop_capture()
                    self.ball_traces.clear()
                    self.current_frame_num = 0
                    self.vision_module.start_capture()
                elif key == ord("+") or key == ord("="):  # Increase speed
                    self.playback_speed = min(4.0, self.playback_speed + 0.25)
                    logger.info(f"Speed: {self.playback_speed:.2f}x")
                elif key == ord("-") or key == ord("_"):  # Decrease speed
                    self.playback_speed = max(0.25, self.playback_speed - 0.25)
                    logger.info(f"Speed: {self.playback_speed:.2f}x")
                elif key == ord("a"):  # Auto-detect table playing area
                    if "frame" in locals() and frame is not None:
                        logger.info("Auto-detecting table playing area...")
                        # Use VisionModule's table detector
                        if self.vision_module.table_detector:
                            try:
                                calibration_result = (
                                    self.vision_module.table_detector.calibrate_table(
                                        frame
                                    )
                                )
                                if calibration_result.get("success"):
                                    table_corners = calibration_result["table_corners"]
                                    if table_corners and len(table_corners) == 4:
                                        # Convert corner tuples to Vector2D objects
                                        frame_width, frame_height = (
                                            frame.shape[1],
                                            frame.shape[0],
                                        )
                                        # Construct canonical physics geometry from
                                        # corners in this frame's declared pixels.
                                        self.table_state = TableState.from_calibration(
                                            table_corners,
                                            (frame_width, frame_height),
                                            table_dimensions_real=self._table_dimensions_real,
                                        )
                                        self._canonical_playing_area_corners = [
                                            corner.to_4k_canonical()
                                            for corner in self.table_state.playing_area_corners
                                        ]
                                        logger.info(
                                            "Playing area detection successful!"
                                        )
                                    else:
                                        logger.error("Invalid table corners detected")
                                else:
                                    logger.error(
                                        f"Table detection failed: {calibration_result.get('error')}"
                                    )
                            except Exception as e:
                                logger.error(f"Failed to detect playing area: {e}")
                        else:
                            logger.error("Table detector not available")
                elif key == ord("p"):  # Toggle playing area display
                    self.show_playing_area = not self.show_playing_area
                    logger.info(f"Playing Area: {self.show_playing_area}")

        finally:
            logger.info("Shutting down...")
            self.vision_module.stop_capture()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video debugger for ball tracking and tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("video", type=str, help="Path to video file (MKV/MP4)")
    parser.add_argument("--loop", action="store_true", help="Loop video playback")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check if video file exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Create and run debugger with sensible defaults
    # Note: yolo_model_path is read from configuration, pass None to use config value
    debugger = VideoDebugger(
        video_path=args.video,
        loop=args.loop,
        max_trace_length=100,
        detection_backend="yolo",
        yolo_model_path=None,  # Use configuration value
        yolo_device="cpu",
    )

    debugger.run()


if __name__ == "__main__":
    main()
