#!/usr/bin/env python3
"""Video debugging tool for ball tracking and trajectory prediction.

This tool allows you to play back video files (MKV/MP4) with ball detection,
tracking, cue stick detection, and predicted trajectory overlaid on the frames.

Uses YOLO detection with the custom-trained billiards model and automatically
detects the table playing area on the first frame.

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

import argparse
import json
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Configure logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import minimal components directly, bypassing __init__ files that have missing dependencies
import importlib.util

def import_from_path(module_name, file_path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get base path
base_path = Path(__file__).parent.parent

# Import capture (has dependencies we need to satisfy first)
# First import kinect2_capture stub
kinect2_path = base_path / "backend" / "vision" / "kinect2_capture.py"
try:
    kinect2_module = import_from_path("backend.vision.kinect2_capture", kinect2_path)
except Exception as e:
    logger.warning(f"Could not import kinect2_capture: {e}")
    # Create a stub
    import types
    kinect2_module = types.ModuleType("backend.vision.kinect2_capture")
    kinect2_module.KINECT2_AVAILABLE = False
    kinect2_module.Kinect2Capture = None
    kinect2_module.Kinect2Status = None
    sys.modules["backend.vision.kinect2_capture"] = kinect2_module

# Import capture
capture_path = base_path / "backend" / "vision" / "capture.py"
capture_module = import_from_path("backend.vision.capture", capture_path)
CameraCapture = capture_module.CameraCapture

# Import models (needed by detector/tracker)
models_path = base_path / "backend" / "vision" / "models.py"
models_module = import_from_path("backend.vision.models", models_path)

# Import detector (use BallDetector directly to avoid circular imports)
balls_path = base_path / "backend" / "vision" / "detection" / "balls.py"
balls_module = import_from_path("backend.vision.detection.balls", balls_path)
BallDetector = balls_module.BallDetector

# Import YOLO detector
yolo_detector_path = base_path / "backend" / "vision" / "detection" / "yolo_detector.py"
yolo_detector_module = import_from_path("backend.vision.detection.yolo_detector", yolo_detector_path)
YOLODetector = yolo_detector_module.YOLODetector

# Import kalman filter first (needed by tracker)
kalman_path = base_path / "backend" / "vision" / "tracking" / "kalman.py"
kalman_module = import_from_path("backend.vision.tracking.kalman", kalman_path)

# Import tracker
tracker_path = base_path / "backend" / "vision" / "tracking" / "tracker.py"
tracker_module = import_from_path("backend.vision.tracking.tracker", tracker_path)
ObjectTracker = tracker_module.ObjectTracker

# Import cue detector
cue_path = base_path / "backend" / "vision" / "detection" / "cue.py"
cue_module = import_from_path("backend.vision.detection.cue", cue_path)
CueDetector = cue_module.CueDetector

# Import table detector
table_path = base_path / "backend" / "vision" / "detection" / "table.py"
table_module = import_from_path("backend.vision.detection.table", table_path)
TableDetector = table_module.TableDetector

# Import core models from backend
models_core_path = base_path / "backend" / "core" / "models.py"
models_core_module = import_from_path("backend.core.models", models_core_path)
sys.modules["backend.core.models"] = models_core_module
Vector2D = models_core_module.Vector2D
BallState = models_core_module.BallState
CueState = models_core_module.CueState
TableState = models_core_module.TableState

# Import utility modules FIRST (before game_state or trajectory) to avoid circular imports
cache_path = base_path / "backend" / "core" / "utils" / "cache.py"
cache_module = import_from_path("backend.core.utils.cache", cache_path)
sys.modules["backend.core.utils.cache"] = cache_module

geometry_path = base_path / "backend" / "core" / "utils" / "geometry.py"
geometry_module = import_from_path("backend.core.utils.geometry", geometry_path)
sys.modules["backend.core.utils.geometry"] = geometry_module

math_utils_path = base_path / "backend" / "core" / "utils" / "math.py"
math_utils_module = import_from_path("backend.core.utils.math", math_utils_path)
sys.modules["backend.core.utils.math"] = math_utils_module

# Import geometric collision module FIRST (needed by trajectory)
# Note: moved to collision module to avoid import chain issues
geometric_collision_path = base_path / "backend" / "core" / "collision" / "geometric_collision.py"
geometric_collision_module = import_from_path("backend.core.collision.geometric_collision", geometric_collision_path)
sys.modules["backend.core.collision.geometric_collision"] = geometric_collision_module

# Import trajectory calculator from backend (do this BEFORE game_state to avoid circular import)
# The trajectory module needs models which we already loaded
trajectory_path = base_path / "backend" / "core" / "physics" / "trajectory.py"
trajectory_module = import_from_path("backend.core.physics.trajectory", trajectory_path)
TrajectoryCalculator = trajectory_module.TrajectoryCalculator
TrajectoryQuality = trajectory_module.TrajectoryQuality
MultiballTrajectoryResult = trajectory_module.MultiballTrajectoryResult


class VideoDebugger:
    """Video playback debugger with ball tracking visualization."""

    def __init__(
        self,
        video_path: str,
        loop: bool = False,
        max_trace_length: int = 100,
        detection_backend: str = "yolo",
        yolo_model_path: Optional[str] = None,
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

        # Initialize components
        logger.info(f"Initializing video debugger for: {video_path}")
        logger.info(f"Detection backend: {detection_backend}")
        if detection_backend == "yolo":
            logger.info(f"YOLO device: {yolo_device}")
            logger.info(f"YOLO model: {yolo_model_path or 'default'}")

        # Configure camera capture for video file
        camera_config = {
            "device_id": video_path,
            "backend": "auto",
            "loop_video": loop,
            "buffer_size": 1,
            "auto_reconnect": False,
        }
        self.camera = CameraCapture(camera_config)

        # Initialize ball detector
        self.detection_backend = detection_backend
        self.detector = None
        self.yolo_detector = None

        if detection_backend == "yolo":
            # Use YOLO detector with OpenCV classification refinement
            logger.info(f"Initializing YOLO detector with model: {yolo_model_path or 'default'}")
            try:
                self.yolo_detector = YOLODetector(
                    model_path=yolo_model_path,
                    device=yolo_device,
                    confidence=0.15,  # Very low for cue detection
                    nms_threshold=0.45,
                    auto_fallback=True,
                    enable_opencv_classification=True,  # Enable hybrid YOLO+OpenCV detection
                    min_ball_size=20,  # Filter out small markers and noise
                )
                logger.info("YOLO detector initialized successfully with OpenCV classification")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO detector: {e}")
                logger.warning("Falling back to OpenCV detection")
                self.detection_backend = "opencv"

        # Initialize OpenCV detector if YOLO not available or selected
        if self.detection_backend == "opencv" or self.yolo_detector is None:
            detector_config = {
                "detection_method": "combined",
                "debug_mode": False,
            }
            self.detector = BallDetector(detector_config)
            self.detection_backend = "opencv"

        # Initialize tracker with ghost ball filtering
        self.tracker_config = {
            "max_age": 30,  # Keep tracks alive (1 second at 30fps)
            "min_hits": 10,  # Require 5 consecutive hits to confirm track (filters ghosts)
            "max_distance": 100.0,  # Very forgiving for stationary balls
            "process_noise": 5.0,  # Higher process noise to handle detection jitter
            "measurement_noise": 20.0,  # Higher tolerance for detection noise
            "collision_threshold": 60.0,  # Distance to detect collision
            "min_hits_during_collision": 30,  # Higher threshold during collision
            "motion_speed_threshold": 10.0,  # Speed to consider "moving"
            "return_tentative_tracks": False,  # Only return confirmed tracks (min_hits+)
        }
        self.tracker = ObjectTracker(self.tracker_config)

        # Initialize cue detector (with relaxed settings for testing)
        cue_config = {
            "min_cue_length": 100,  # Reduced from 150
            "max_cue_length": 1000,  # Increased from 800
            "min_line_thickness": 2,  # Reduced from 3
            "max_line_thickness": 40,  # Increased from 25
            "hough_threshold": 50,  # Reduced from 80 (more sensitive)
            "min_detection_confidence": 0.3,  # Reduced from 0.5 (more lenient)
            "temporal_smoothing": 0.7,
        }
        # Pass YOLO detector to CueDetector so it can use YOLO for detection
        self.cue_detector = CueDetector(cue_config, yolo_detector=self.yolo_detector)

        # Test mode - always show a sample trajectory for testing
        self.test_mode = False  # Set to True to show test trajectory
        self.auto_detect_on_first_frame = False  # Set to True to auto-detect on first frame
        self.auto_detect_done = False  # Track if auto-detect has been performed

        # Initialize trajectory calculator from backend
        self.trajectory_calculator = TrajectoryCalculator()

        # Table configuration - load from saved config if available
        self.table_state = self._load_table_state_from_config()

        # Initialize table detector for playing area detection
        table_detector_config = {}
        self.table_detector = TableDetector(table_detector_config)

        # Current trajectory prediction
        self.current_trajectory = None

        logger.info("Video debugger initialized")

    def _load_table_state_from_config(self) -> TableState:
        """Load table state from saved configuration file.

        Returns:
            TableState with playing_area_corners from config if available,
            otherwise returns a standard table.
        """
        try:
            # Try to load config from standard location
            config_path = Path(__file__).parent.parent / "config" / "current.json"

            if not config_path.exists():
                logger.warning(f"Config file not found at {config_path}, using standard table")
                return TableState.standard_9ft_table()

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract table config
            table_config = config.get("table", {})
            playing_area_corners_data = table_config.get("playing_area_corners")
            calibration_res = table_config.get("calibration_resolution", {})

            if not playing_area_corners_data or len(playing_area_corners_data) != 4:
                logger.warning("No valid playing_area_corners in config, using standard table")
                return TableState.standard_9ft_table()

            # Store calibration resolution for later scaling
            self.calibration_resolution = (
                calibration_res.get("width", 640),
                calibration_res.get("height", 360)
            )

            # Convert corner dicts to Vector2D objects (in calibration resolution)
            # These will be scaled when we get the first frame
            playing_area_corners = [
                Vector2D(corner["x"], corner["y"])
                for corner in playing_area_corners_data
            ]

            # Create table state with corners (dimensions will be updated from frame)
            table_state = TableState.standard_9ft_table()
            table_state.playing_area_corners = playing_area_corners

            logger.info(f"Loaded playing area corners from config (calibration res: {self.calibration_resolution[0]}x{self.calibration_resolution[1]}):")
            for i, corner in enumerate(playing_area_corners):
                logger.info(f"  Corner {i+1}: ({corner.x:.1f}, {corner.y:.1f})")

            return table_state

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Set default calibration resolution
            self.calibration_resolution = (640, 360)
            return TableState.standard_9ft_table()

    def draw_ball(
        self, frame: np.ndarray, x: float, y: float, radius: float, color: tuple, label: str = ""
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

    def draw_cue_stick(self, frame: np.ndarray, cue_stick, color: tuple = (0, 255, 255)) -> None:
        """Draw cue stick on frame.

        Args:
            frame: Frame to draw on
            cue_stick: CueStick object
            color: BGR color tuple
        """
        if cue_stick is None:
            return

        tip_x, tip_y = int(cue_stick.tip_position[0]), int(cue_stick.tip_position[1])

        # Calculate butt position from angle and length
        angle_rad = np.radians(cue_stick.angle)
        butt_x = int(tip_x - cue_stick.length * np.cos(angle_rad))
        butt_y = int(tip_y - cue_stick.length * np.sin(angle_rad))

        # Draw cue stick line
        cv2.line(frame, (tip_x, tip_y), (butt_x, butt_y), color, 3)

        # Draw tip marker
        cv2.circle(frame, (tip_x, tip_y), 6, color, -1)

        # Draw state indicator
        state_color = (0, 255, 0) if cue_stick.is_aiming else (0, 0, 255)
        cv2.circle(frame, (tip_x, tip_y), 10, state_color, 2)

    def detect_and_set_playing_area(self, frame: np.ndarray) -> bool:
        """Detect table boundaries and set playing area in TableState.

        Args:
            frame: Frame to detect table from

        Returns:
            True if detection successful, False otherwise
        """
        try:
            logger.info("Detecting table playing area...")
            calibration_result = self.table_detector.calibrate_table(frame)

            if not calibration_result.get("success"):
                logger.error(f"Table detection failed: {calibration_result.get('error', 'Unknown error')}")
                return False

            table_corners = calibration_result["table_corners"]
            if not table_corners or len(table_corners) != 4:
                logger.error(f"Invalid table corners detected: {table_corners}")
                return False

            # Convert corner tuples to Vector2D objects
            playing_area_corners = [
                Vector2D(corner[0], corner[1]) for corner in table_corners
            ]

            # Update table state with detected playing area
            self.table_state.playing_area_corners = playing_area_corners

            logger.info(f"Playing area detected successfully:")
            logger.info(f"  Corners: {[(c.x, c.y) for c in playing_area_corners]}")
            logger.info(f"  Confidence: {calibration_result.get('confidence', 0.0):.2f}")
            logger.info(f"  Pockets: {calibration_result.get('pocket_count', 0)}")

            return True

        except Exception as e:
            logger.error(f"Failed to detect playing area: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def draw_playing_area(self, frame: np.ndarray) -> None:
        """Draw the playing area polygon on the frame.

        Args:
            frame: Frame to draw on
        """
        if not self.table_state.playing_area_corners or len(self.table_state.playing_area_corners) != 4:
            return

        # Convert Vector2D to integer points for drawing
        points = np.array([
            [int(corner.x), int(corner.y)]
            for corner in self.table_state.playing_area_corners
        ], dtype=np.int32)

        # Draw polygon outline
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw corner markers
        for i, corner in enumerate(self.table_state.playing_area_corners):
            center = (int(corner.x), int(corner.y))
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
        if self.table_state.playing_area_corners:
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

        # Define colors matching cassapa:
        # White ball (cue) = white (255, 255, 255)
        # Hit ball = yellow (0, 255, 255) in BGR
        cue_color = (255, 255, 255)  # White for cue ball
        hit_color = (0, 255, 255)    # Yellow for first hit ball

        # Get primary (cue) ball trajectory
        primary_id = multiball_result.primary_ball_id
        if primary_id not in multiball_result.trajectories:
            return

        cue_trajectory = multiball_result.trajectories[primary_id]

        # Draw cue ball trajectory (white lines)
        # Based on cassapa lines 228-234
        if cue_trajectory.points and len(cue_trajectory.points) >= 2:
            # Draw line segments for each trajectory piece
            for i in range(len(cue_trajectory.points) - 1):
                p1 = cue_trajectory.points[i]
                p2 = cue_trajectory.points[i + 1]

                pt1 = (int(p1.position.x), int(p1.position.y))
                pt2 = (int(p2.position.x), int(p2.position.y))

                # Draw the line segment
                cv2.line(frame, pt1, pt2, cue_color, 4)

                # Draw circle at endpoint (shows ball position at end of segment)
                # Get ball radius from initial state
                ball_radius = int(cue_trajectory.initial_state.radius)
                cv2.circle(frame, pt2, ball_radius, cue_color, 2)

        # Find the first hit ball (if any) and draw its trajectory
        # Based on cassapa lines 238-265
        hit_ball_id = None
        for collision in cue_trajectory.collisions:
            if collision.type.value == "ball_ball" and collision.ball2_id:
                hit_ball_id = collision.ball2_id
                break

        if hit_ball_id and hit_ball_id in multiball_result.trajectories:
            hit_trajectory = multiball_result.trajectories[hit_ball_id]

            if hit_trajectory.points and len(hit_trajectory.points) >= 2:
                # Draw hit ball trajectory (yellow lines) - cassapa lines 242-246
                for i in range(len(hit_trajectory.points) - 1):
                    p1 = hit_trajectory.points[i]
                    p2 = hit_trajectory.points[i + 1]

                    pt1 = (int(p1.position.x), int(p1.position.y))
                    pt2 = (int(p2.position.x), int(p2.position.y))

                    cv2.line(frame, pt1, pt2, hit_color, 4)

                # Draw white circle at collision point where cue ball hits target ball
                # cassapa lines 248-255
                if cue_trajectory.points:
                    collision_point = cue_trajectory.points[-1].position
                    ball_radius = int(cue_trajectory.initial_state.radius)
                    cv2.circle(frame,
                             (int(collision_point.x), int(collision_point.y)),
                             ball_radius, cue_color, 2)

                # Draw yellow circle at end position of hit ball
                # cassapa lines 257-264
                if hit_trajectory.points:
                    final_point = hit_trajectory.points[-1].position
                    ball_radius = int(hit_trajectory.initial_state.radius)
                    cv2.circle(frame,
                             (int(final_point.x), int(final_point.y)),
                             ball_radius, hit_color, 2)

    def _find_ball_cue_is_pointing_at(self, cue, balls):
        """Find which ball the cue is currently pointing at.

        Args:
            cue: Detected cue stick with tip and butt positions
            balls: List of detected balls

        Returns:
            The ball the cue is pointing at, or None
        """
        if not cue or not balls:
            return None

        # Get butt position (use calculated if not set)
        if cue.butt_position and cue.butt_position != (0.0, 0.0):
            butt_x, butt_y = cue.butt_position
        else:
            # Fallback: calculate from tip, angle, and length
            angle_rad = np.radians(cue.angle)
            butt_x = cue.tip_position[0] - cue.length * np.cos(angle_rad)
            butt_y = cue.tip_position[1] - cue.length * np.sin(angle_rad)

        # Calculate cue direction vector
        cue_dx = cue.tip_position[0] - butt_x
        cue_dy = cue.tip_position[1] - butt_y
        cue_length = np.sqrt(cue_dx**2 + cue_dy**2)

        if cue_length == 0:
            return None

        # Normalize direction
        cue_dx /= cue_length
        cue_dy /= cue_length

        # Find the closest ball along the cue direction
        closest_ball = None
        min_distance = float('inf')
        max_perpendicular_distance = 40  # pixels tolerance

        for ball in balls:
            # Vector from cue tip to ball center
            ball_dx = ball.position[0] - cue.tip_position[0]
            ball_dy = ball.position[1] - cue.tip_position[1]

            # Distance along cue direction (projection)
            distance_along_cue = ball_dx * cue_dx + ball_dy * cue_dy

            # Skip balls behind the cue tip
            if distance_along_cue < 0:
                continue

            # Calculate perpendicular distance from cue line to ball center
            perpendicular_distance = abs(ball_dx * cue_dy - ball_dy * cue_dx)

            # Check if ball is within tolerance and closer than current closest
            if perpendicular_distance < max_perpendicular_distance and distance_along_cue < min_distance:
                min_distance = distance_along_cue
                closest_ball = ball

        return closest_ball

    def draw_hud(self, frame: np.ndarray) -> None:
        """Draw heads-up display with info and controls.

        Args:
            frame: Frame to draw on
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
            f"Backend: {self.detection_backend.upper()}",
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
            "SPACE: Play/Pause | LEFT/RIGHT: Step | +/-: Speed | B: Set Background | A: Auto-detect Table",
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

        Args:
            frame: Input frame

        Returns:
            Frame with visualizations drawn
        """
        # Update table dimensions from frame if needed, and scale corners if resolution changed
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        if self.table_state.width != frame_width or self.table_state.height != frame_height:
            # Scale playing area corners from calibration resolution to current frame resolution
            if hasattr(self, 'calibration_resolution') and self.table_state.playing_area_corners:
                calib_width, calib_height = self.calibration_resolution

                # Use the TableState method to scale the corners
                self.table_state.scale_playing_area_corners(
                    from_width=calib_width,
                    from_height=calib_height,
                    to_width=frame_width,
                    to_height=frame_height
                )

                logger.info(f"Scaled playing area corners from {calib_width}x{calib_height} to {frame_width}x{frame_height}:")
                for i, corner in enumerate(self.table_state.playing_area_corners):
                    logger.info(f"  Corner {i+1}: ({corner.x:.1f}, {corner.y:.1f})")

            self.table_state.width = frame_width
            self.table_state.height = frame_height

        # Create a copy for drawing
        display_frame = frame.copy()

        # Detect balls using appropriate backend
        if self.yolo_detector is not None:
            # Use hybrid YOLO+OpenCV detection (backend handles classification)
            detected_balls = self.yolo_detector.detect_balls_with_classification(
                frame,
                min_confidence=0.25
            )
        else:
            # Use OpenCV detector
            detected_balls = self.detector.detect_balls(frame)

        # Update tracker
        tracked_balls = self.tracker.update_tracking(
            detected_balls, self.current_frame_num, None
        )

        # Debug logging every 30 frames
        if self.current_frame_num % 30 == 0:
            logger.info(f"Frame {self.current_frame_num}: {len(detected_balls)} detections, {len(tracked_balls) if tracked_balls else 0} confirmed tracks")
            if hasattr(self.tracker, 'tracks'):
                tentative = sum(1 for t in self.tracker.tracks if t.state.value == "tentative")
                confirmed = sum(1 for t in self.tracker.tracks if t.state.value == "confirmed")
                logger.info(f"  Tracker state: {confirmed} confirmed, {tentative} tentative, {len(self.tracker.tracks)} total")
                # Log details about tentative tracks
                for t in self.tracker.tracks:
                    if t.state.value == "tentative":
                        logger.info(f"    Track {t.track_id}: {t.detection_count}/{t.min_hits} hits, {t.miss_count} misses")

        # Store current tracked balls for HUD display
        self.current_tracked_balls = tracked_balls if tracked_balls else []

        # Use tracked balls if available, otherwise show detections
        balls_to_draw = tracked_balls if tracked_balls else detected_balls

        # Find cue ball (for cue detection - needs a reference point)
        cue_ball = None
        cue_ball_pos = None
        for ball in balls_to_draw:
            if hasattr(ball, "ball_type") and str(ball.ball_type) == "CUE":
                cue_ball = ball
                cue_ball_pos = ball.position
                break

        # If no cue ball identified, use first detected ball as approximation for cue detection
        if cue_ball is None and balls_to_draw:
            cue_ball = balls_to_draw[0]
            cue_ball_pos = balls_to_draw[0].position

        # Detect cue stick (CueDetector handles YOLO + line-based fallback internally)
        detected_cue = None
        if cue_ball_pos:
            detected_cue = self.cue_detector.detect_cue(frame, cue_ball_pos)
            if detected_cue and self.current_frame_num % 30 == 0:
                logger.info(f"Cue detected: angle={detected_cue.angle:.1f}, confidence={detected_cue.confidence:.2f}, aiming={detected_cue.is_aiming}")
            elif detected_cue is None and self.current_frame_num % 30 == 0:
                logger.info("No cue detected")

        # Test mode: create a fake trajectory for visualization testing
        if self.test_mode and cue_ball and self.current_frame_num % 30 == 0:
            # Create a test trajectory pointing to the right
            logger.info("Creating test trajectory")
            test_cue_state = CueState(
                angle=0.0,  # Point to the right
                estimated_force=3.0,
                impact_point=Vector2D(cue_ball_pos[0], cue_ball_pos[1]),
                tip_position=Vector2D(cue_ball_pos[0] - 50, cue_ball_pos[1]),
                elevation=0.0,
                is_visible=True,
                confidence=1.0,
                last_update=0,
            )
            test_ball_state = BallState(
                id="cue",
                position=Vector2D(cue_ball_pos[0], cue_ball_pos[1]),
                velocity=Vector2D(0, 0),
                radius=cue_ball.radius,
                mass=0.17,
                is_cue_ball=True,
                confidence=1.0,
                last_update=0,
            )
            # Create other ball states for test mode
            test_other_balls = []
            for i, ball in enumerate(balls_to_draw):
                if ball != cue_ball:
                    test_other_balls.append(
                        BallState(
                            id=f"ball_{i}",
                            position=Vector2D(ball.position[0], ball.position[1]),
                            velocity=Vector2D(0, 0),
                            radius=ball.radius,
                            mass=0.17,
                            confidence=1.0,
                            last_update=0,
                        )
                    )

            self.current_trajectory = self.trajectory_calculator.predict_multiball_cue_shot(
                test_cue_state,
                test_ball_state,
                self.table_state,
                test_other_balls,
                quality=TrajectoryQuality.LOW,
                max_collision_depth=5,
            )

        # Calculate trajectory if cue is detected (regardless of aiming state for visualization)
        elif detected_cue and balls_to_draw:
            try:
                # Find which ball the cue is pointing at (not just cue ball!)
                target_ball = self._find_ball_cue_is_pointing_at(detected_cue, balls_to_draw)

                if target_ball is None:
                    if self.current_frame_num % 30 == 0:
                        logger.info("No target ball found for cue trajectory")
                    self.current_trajectory = None
                else:
                    if self.current_frame_num % 30 == 0:
                        logger.info(f"Target ball found at ({target_ball.position[0]:.1f}, {target_ball.position[1]:.1f})")
                    # Create CueState for trajectory calculation
                    cue_state = CueState(
                        angle=detected_cue.angle,
                        estimated_force=5.0,  # Default moderate force
                        impact_point=Vector2D(detected_cue.tip_position[0], detected_cue.tip_position[1]),
                        tip_position=Vector2D(detected_cue.tip_position[0], detected_cue.tip_position[1]),
                        elevation=getattr(detected_cue, 'elevation', 0.0),
                        is_visible=True,
                        confidence=detected_cue.confidence,
                        last_update=0,
                    )

                    # Create BallState for the ball the cue is pointing at
                    ball_state = BallState(
                        id=f"target_ball",
                        position=Vector2D(target_ball.position[0], target_ball.position[1]),
                        velocity=Vector2D(0, 0),
                        radius=target_ball.radius,
                        mass=0.17,  # Standard pool ball mass in kg
                        is_cue_ball=False,  # Might not be cue ball!
                        confidence=getattr(target_ball, 'confidence', 1.0),
                        last_update=0,
                    )

                    # Create BallState objects for other balls
                    other_ball_states = []
                    for i, ball in enumerate(balls_to_draw):
                        if ball != target_ball:
                            other_ball_states.append(
                                BallState(
                                    id=f"ball_{i}",
                                    position=Vector2D(ball.position[0], ball.position[1]),
                                    velocity=Vector2D(0, 0),
                                    radius=ball.radius,
                                    mass=0.17,
                                    confidence=getattr(ball, 'confidence', 1.0),
                                    last_update=0,
                                )
                            )

                    # Calculate multiball trajectory using backend calculator
                    if self.current_frame_num % 30 == 0:
                        logger.info(f"Calculating trajectory: cue angle={cue_state.angle:.1f}, ball pos=({ball_state.position.x:.1f}, {ball_state.position.y:.1f}), other_balls={len(other_ball_states)}")

                    self.current_trajectory = self.trajectory_calculator.predict_multiball_cue_shot(
                        cue_state,
                        ball_state,
                        self.table_state,
                        other_ball_states,
                        quality=TrajectoryQuality.LOW,  # Use low quality for real-time visualization
                        max_collision_depth=5,  # Calculate up to 5 collision levels deep
                    )

                    if self.current_frame_num % 30 == 0:
                        if self.current_trajectory and self.current_trajectory.trajectories:
                            logger.info(f"Trajectory calculated successfully: {len(self.current_trajectory.trajectories)} ball(s)")
                        else:
                            logger.info("Trajectory calculation returned empty result")
            except Exception as e:
                if self.current_frame_num % 30 == 0:
                    logger.error(f"Trajectory calculation failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
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
        for i, ball in enumerate(balls_to_draw):
            x, y = ball.position
            radius = ball.radius
            track_id = getattr(ball, 'track_id', None)

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
                color = cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                )[0][0]
                color = tuple(int(c) for c in color)

            # Draw detection
            if self.show_detections:
                label = ""
                if self.show_track_ids and track_id is not None:
                    # Use display_name if available, otherwise fall back to track_id
                    if hasattr(ball, 'display_name'):
                        label = ball.display_name
                    else:
                        label = f"ID:{track_id}"
                    if getattr(ball, 'is_moving', False):
                        label += " *"
                elif track_id is None:
                    label = f"#{i}"  # Show detection index for untracked balls

                self.draw_ball(display_frame, x, y, radius, color, label)

        # Draw HUD
        self.draw_hud(display_frame)

        return display_frame

    def run(self) -> None:
        """Run the video debugger main loop."""
        logger.info("Starting video debugger...")

        # Start camera capture
        if not self.camera.start_capture():
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
                    frame_data = self.camera.get_latest_frame()

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
                        # This avoids exiting on temporary buffer issues
                        if no_frame_count > 10:
                            if not self.loop:
                                logger.info(f"Video playback completed (processed {frame_count} frames)")
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
                    self.camera.stop_capture()
                    self.ball_traces.clear()
                    self.tracker = ObjectTracker(self.tracker_config)
                    self.current_frame_num = 0
                    self.camera.start_capture()
                elif key == ord("+") or key == ord("="):  # Increase speed
                    self.playback_speed = min(4.0, self.playback_speed + 0.25)
                    logger.info(f"Speed: {self.playback_speed:.2f}x")
                elif key == ord("-") or key == ord("_"):  # Decrease speed
                    self.playback_speed = max(0.25, self.playback_speed - 0.25)
                    logger.info(f"Speed: {self.playback_speed:.2f}x")
                elif key == ord("b"):  # Set background frame
                    if 'frame' in locals() and frame is not None:
                        self.cue_detector.set_background_frame(frame)
                        if self.detector is not None:
                            self.detector.set_background_frame(frame)
                        logger.info("Background frame set for cue and ball detection")
                elif key == ord("a"):  # Auto-detect table playing area
                    if 'frame' in locals() and frame is not None:
                        logger.info("Auto-detecting table playing area...")
                        if self.detect_and_set_playing_area(frame):
                            logger.info("Playing area detection successful!")
                        else:
                            logger.error("Playing area detection failed")
                elif key == ord("p"):  # Toggle playing area display
                    self.show_playing_area = not self.show_playing_area
                    logger.info(f"Playing Area: {self.show_playing_area}")

        finally:
            logger.info("Shutting down...")
            self.camera.stop_capture()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video debugger for ball tracking and tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("video", type=str, help="Path to video file (MKV/MP4)")
    parser.add_argument(
        "--loop", action="store_true", help="Loop video playback"
    )
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
    debugger = VideoDebugger(
        video_path=args.video,
        loop=args.loop,
        max_trace_length=100,
        detection_backend="yolo",
        yolo_model_path="models/yolov8n-pool-1280.onnx",  # High-res 1280x1280 model
        yolo_device="cpu",
    )

    # Always auto-detect table on first frame
    logger.info("Auto-detect table mode enabled - will detect on first frame")
    debugger.auto_detect_on_first_frame = True

    debugger.run()


if __name__ == "__main__":
    main()
