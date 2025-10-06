#!/usr/bin/env python3
"""Video debugging tool for ball tracking and trajectory prediction.

This tool allows you to play back video files (MKV/MP4) with ball detection,
tracking, cue stick detection, and predicted trajectory overlaid on the frames.

Supports both YOLO (deep learning) and OpenCV (traditional) detection backends.

Usage:
    python tools/video_debugger.py <video_file> [options]

Examples:
    # Use YOLO detection with CPU
    python tools/video_debugger.py demo.mkv

    # Use YOLO detection with custom model
    python tools/video_debugger.py demo.mkv --yolo-model models/yolov8n-pool.onnx

    # Use YOLO detection with Coral TPU
    python tools/video_debugger.py demo.mkv --yolo-device tpu

    # Use OpenCV detection (traditional method)
    python tools/video_debugger.py demo.mkv --backend opencv

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
    B     - Set current frame as background (for cue detection)
    Q/ESC - Quit
"""

import argparse
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

# Import detector factory
factory_path = base_path / "backend" / "vision" / "detection" / "detector_factory.py"
factory_module = import_from_path("backend.vision.detection.detector_factory", factory_path)
DetectorFactory = factory_module.DetectorFactory

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

# Define simplified data classes for trajectory calculation
# (avoiding complex imports that have circular dependencies)
@dataclass
class Vector2D:
    """Simple 2D vector."""
    x: float
    y: float

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return np.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        """Return normalized vector."""
        mag = self.magnitude()
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)


@dataclass
class BallState:
    """Simplified ball state."""
    id: str
    position: Vector2D
    velocity: Vector2D
    radius: float
    mass: float
    spin: Optional[Vector2D] = None
    is_cue_ball: bool = False
    is_pocketed: bool = False
    number: Optional[int] = None


@dataclass
class CueState:
    """Simplified cue state."""
    angle: float
    estimated_force: float
    impact_point: Optional[Vector2D] = None


@dataclass
class TableState:
    """Simplified table state."""
    width: float
    height: float
    cushion_elasticity: float = 0.85
    surface_friction: float = 0.015
    pocket_radius: float = 30.0
    pocket_positions: list = None

    def __post_init__(self):
        if self.pocket_positions is None:
            self.pocket_positions = []

# Simplified trajectory result classes
@dataclass
class TrajectoryPoint:
    """A point along a trajectory."""
    position: Vector2D
    velocity: Vector2D


@dataclass
class SimpleTrajectory:
    """Simplified trajectory result."""
    points: list
    collisions: list
    final_position: Optional[Vector2D] = None
    will_be_pocketed: bool = False


class SimplifiedTrajectoryCalculator:
    """Simplified trajectory calculator for real-time visualization."""

    def __init__(self):
        self.friction = 0.015
        self.timestep = 0.05  # seconds
        self.max_time = 5.0  # seconds

    def predict_cue_shot(self, cue_state, ball_state, table_state, other_balls=None, quality=None):
        """Calculate simple trajectory from cue shot."""
        # Calculate initial velocity from cue angle and force
        angle_rad = math.radians(cue_state.angle)
        speed = cue_state.estimated_force * 0.5  # m/s

        initial_velocity = Vector2D(
            speed * math.cos(angle_rad),
            speed * math.sin(angle_rad)
        )

        # Simulate trajectory
        points = []
        current_pos = Vector2D(ball_state.position.x, ball_state.position.y)
        current_vel = Vector2D(initial_velocity.x, initial_velocity.y)

        time = 0
        while time < self.max_time and current_vel.magnitude() > 0.01:
            # Add point
            points.append(TrajectoryPoint(
                position=Vector2D(current_pos.x, current_pos.y),
                velocity=Vector2D(current_vel.x, current_vel.y)
            ))

            # Update velocity (friction)
            vel_mag = current_vel.magnitude()
            if vel_mag > 0:
                friction_decel = self.friction * 9.81 * self.timestep
                new_mag = max(0, vel_mag - friction_decel)
                current_vel.x = current_vel.x / vel_mag * new_mag
                current_vel.y = current_vel.y / vel_mag * new_mag

            # Update position
            current_pos.x += current_vel.x * self.timestep
            current_pos.y += current_vel.y * self.timestep

            # Check table boundaries (simple bounce)
            if current_pos.x < ball_state.radius:
                current_pos.x = ball_state.radius
                current_vel.x = -current_vel.x * 0.8
            elif current_pos.x > table_state.width - ball_state.radius:
                current_pos.x = table_state.width - ball_state.radius
                current_vel.x = -current_vel.x * 0.8

            if current_pos.y < ball_state.radius:
                current_pos.y = ball_state.radius
                current_vel.y = -current_vel.y * 0.8
            elif current_pos.y > table_state.height - ball_state.radius:
                current_pos.y = table_state.height - ball_state.radius
                current_vel.y = -current_vel.y * 0.8

            time += self.timestep

        return SimpleTrajectory(
            points=points,
            collisions=[],
            final_position=current_pos
        )


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

        # Tracking data
        self.ball_traces: dict[int, deque] = {}  # track_id -> deque of positions

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

        # Initialize ball detector using factory pattern
        detector_config = {
            "detection_backend": detection_backend,
            "debug_mode": False,
        }

        # Add YOLO-specific config if using YOLO backend
        if detection_backend == "yolo":
            detector_config.update({
                "yolo_model_path": yolo_model_path,
                "yolo_device": yolo_device,
                "yolo_confidence": 0.4,
                "yolo_nms_threshold": 0.45,
                "use_opencv_validation": True,  # Use hybrid validation
                "fallback_to_opencv": True,
            })
        else:
            # OpenCV-specific config
            detector_config.update({
                "detection_method": "combined",
            })

        self.detector = DetectorFactory.create_detector(detector_config)
        self.detection_backend = detection_backend

        # Initialize tracker (lenient settings to maintain tracks longer)
        tracker_config = {
            "max_age": 90,  # Keep tracks alive much longer (3 seconds at 30fps)
            "min_hits": 2,  # Require 2 hits to confirm track
            "max_distance": 80.0,  # More forgiving distance matching
            "process_noise": 2.0,  # Higher process noise for motion tolerance
            "measurement_noise": 15.0,  # Higher measurement noise tolerance
        }
        self.tracker = ObjectTracker(tracker_config)

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
        self.cue_detector = CueDetector(cue_config)

        # Test mode - always show a sample trajectory for testing
        self.test_mode = False  # Set to True to show test trajectory

        # Initialize trajectory calculator (simplified version)
        self.trajectory_calculator = SimplifiedTrajectoryCalculator()

        # Table configuration (approximate - adjust based on your video)
        # Assuming standard pool table dimensions in pixels
        self.table_state = TableState(
            width=1920,  # Will be updated from frame size
            height=1080,
            cushion_elasticity=0.85,
            surface_friction=0.015,
            pocket_radius=30.0,
            pocket_positions=[
                Vector2D(50, 50),  # Top-left
                Vector2D(960, 50),  # Top-center
                Vector2D(1870, 50),  # Top-right
                Vector2D(50, 1030),  # Bottom-left
                Vector2D(960, 1030),  # Bottom-center
                Vector2D(1870, 1030),  # Bottom-right
            ],
        )

        # Current trajectory prediction
        self.current_trajectory = None

        logger.info("Video debugger initialized")

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

    def draw_trajectory(self, frame: np.ndarray, trajectory) -> None:
        """Draw predicted trajectory on frame.

        Args:
            frame: Frame to draw on
            trajectory: Trajectory object from trajectory calculator
        """
        if trajectory is None or not trajectory.points:
            return

        # Draw main trajectory path
        points = trajectory.points
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            # Color based on velocity (red = fast, yellow = slow)
            speed = p1.velocity.magnitude()
            max_speed = 3.0  # Adjusted for pixel-based velocity
            speed_ratio = min(1.0, speed / max_speed)

            # Gradient from yellow (slow) to red (fast)
            b = int(0)
            g = int(255 * (1 - speed_ratio))
            r = int(255)
            color = (b, g, r)

            # Draw trajectory segment
            pt1 = (int(p1.position.x), int(p1.position.y))
            pt2 = (int(p2.position.x), int(p2.position.y))
            cv2.line(frame, pt1, pt2, color, 2)

        # Draw collision points (if any)
        if trajectory.collisions:
            for collision in trajectory.collisions:
                pos = (int(collision.position.x), int(collision.position.y))

                # Color by collision type
                collision_type = getattr(collision, 'type', None)
                if collision_type:
                    type_val = collision_type.value if hasattr(collision_type, 'value') else str(collision_type)
                    if type_val == "ball_ball":
                        collision_color = (255, 0, 255)  # Magenta
                        cv2.circle(frame, pos, 8, collision_color, 2)
                        cv2.circle(frame, pos, 3, collision_color, -1)
                    elif type_val == "ball_cushion":
                        collision_color = (255, 255, 0)  # Cyan
                        cv2.circle(frame, pos, 6, collision_color, 2)
                    elif type_val == "ball_pocket":
                        collision_color = (0, 255, 0)  # Green
                        cv2.circle(frame, pos, 10, collision_color, 3)
                        cv2.circle(frame, pos, 5, collision_color, -1)

        # Draw final position
        if trajectory.final_position:
            final_pos = (int(trajectory.final_position.x), int(trajectory.final_position.y))
            final_color = (0, 255, 0) if trajectory.will_be_pocketed else (128, 128, 128)
            cv2.circle(frame, final_pos, 12, final_color, 2)
            cv2.circle(frame, final_pos, 4, final_color, -1)

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
            f"Balls: {len(self.ball_traces)}",
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
            "SPACE: Play/Pause | LEFT/RIGHT: Step | +/-: Speed | B: Set Background",
            "T: Trajectory | D: Detections | I: IDs | C: Cue | R: Reset | Q: Quit",
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
        # Update table dimensions from frame if needed
        if self.table_state.width != frame.shape[1] or self.table_state.height != frame.shape[0]:
            self.table_state.width = frame.shape[1]
            self.table_state.height = frame.shape[0]

        # Create a copy for drawing
        display_frame = frame.copy()

        # Detect balls
        detected_balls = self.detector.detect_balls(frame)

        # Update tracker
        tracked_balls = self.tracker.update_tracking(
            detected_balls, self.current_frame_num, None
        )

        # Use tracked balls if available, otherwise show detections
        balls_to_draw = tracked_balls if tracked_balls else detected_balls

        # Find cue ball
        cue_ball = None
        cue_ball_pos = None
        for ball in balls_to_draw:
            if hasattr(ball, "ball_type") and str(ball.ball_type) == "CUE":
                cue_ball = ball
                cue_ball_pos = ball.position
                break

        # If no cue ball identified, use first detected ball as approximation
        if cue_ball is None and balls_to_draw:
            cue_ball = balls_to_draw[0]
            cue_ball_pos = balls_to_draw[0].position

        # Detect cue stick
        detected_cue = None
        if cue_ball_pos:
            detected_cue = self.cue_detector.detect_cue(frame, cue_ball_pos)
            if detected_cue:
                if self.current_frame_num % 30 == 0:  # Log every 30 frames to avoid spam
                    logger.info(f"Cue detected: angle={detected_cue.angle:.1f}, confidence={detected_cue.confidence:.2f}, aiming={detected_cue.is_aiming}")
            elif self.current_frame_num % 30 == 0:
                logger.info("No cue detected")

        # Test mode: create a fake trajectory for visualization testing
        if self.test_mode and cue_ball and self.current_frame_num % 30 == 0:
            # Create a test trajectory pointing to the right
            logger.info("Creating test trajectory")
            test_cue_state = CueState(
                angle=0.0,  # Point to the right
                estimated_force=3.0,
                impact_point=Vector2D(cue_ball_pos[0], cue_ball_pos[1]),
            )
            test_ball_state = BallState(
                id="cue",
                position=Vector2D(cue_ball_pos[0], cue_ball_pos[1]),
                velocity=Vector2D(0, 0),
                radius=cue_ball.radius,
                mass=0.17,
                is_cue_ball=True,
            )
            self.current_trajectory = self.trajectory_calculator.predict_cue_shot(
                test_cue_state,
                test_ball_state,
                self.table_state,
                [],
            )

        # Calculate trajectory if cue is detected (regardless of aiming state for visualization)
        elif detected_cue and cue_ball:
            try:
                # Create CueState for trajectory calculation
                cue_state = CueState(
                    angle=detected_cue.angle,
                    estimated_force=5.0,  # Default moderate force
                    impact_point=Vector2D(detected_cue.tip_position[0], detected_cue.tip_position[1]),
                )

                # Create BallState for cue ball
                ball_state = BallState(
                    id="cue",
                    position=Vector2D(cue_ball_pos[0], cue_ball_pos[1]),
                    velocity=Vector2D(0, 0),
                    radius=cue_ball.radius,
                    mass=0.17,  # Standard pool ball mass in kg
                    is_cue_ball=True,
                )

                # Create BallState objects for other balls
                other_ball_states = []
                for i, ball in enumerate(balls_to_draw):
                    if ball != cue_ball:
                        other_ball_states.append(
                            BallState(
                                id=f"ball_{i}",
                                position=Vector2D(ball.position[0], ball.position[1]),
                                velocity=Vector2D(0, 0),
                                radius=ball.radius,
                                mass=0.17,
                            )
                        )

                # Calculate trajectory
                self.current_trajectory = self.trajectory_calculator.predict_cue_shot(
                    cue_state,
                    ball_state,
                    self.table_state,
                    other_ball_states,
                )
            except Exception as e:
                logger.debug(f"Trajectory calculation failed: {e}")
                self.current_trajectory = None
        else:
            self.current_trajectory = None

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
                    self.tracker = ObjectTracker(
                        {
                            "max_age": 30,
                            "min_hits": 3,
                            "max_distance": 50.0,
                        }
                    )
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
                        self.detector.set_background_frame(frame)
                        logger.info("Background frame set for cue and ball detection")

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
        "--trace-length",
        type=int,
        default=100,
        help="Maximum trace length (default: 100)",
    )
    parser.add_argument(
        "--backend",
        choices=["yolo", "opencv"],
        default="yolo",
        help="Detection backend (default: yolo)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        help="Path to YOLO model (default: models/yolov8n-pool.onnx)",
    )
    parser.add_argument(
        "--yolo-device",
        choices=["cpu", "cuda", "tpu"],
        default="cpu",
        help="YOLO inference device (default: cpu)",
    )
    parser.add_argument(
        "--background",
        type=str,
        help="Path to background image file (for background subtraction)",
    )
    parser.add_argument(
        "--test-trajectory",
        action="store_true",
        help="Enable test trajectory mode (shows sample trajectory for testing)",
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

    # Create and run debugger
    debugger = VideoDebugger(
        video_path=args.video,
        loop=args.loop,
        max_trace_length=args.trace_length,
        detection_backend=args.backend,
        yolo_model_path=args.yolo_model,
        yolo_device=args.yolo_device,
    )

    # Enable test mode if requested
    if args.test_trajectory:
        debugger.test_mode = True
        logger.info("Test trajectory mode enabled")

    # Load background image if provided
    if args.background:
        if not Path(args.background).exists():
            logger.error(f"Background image not found: {args.background}")
            sys.exit(1)

        logger.info(f"Loading background image: {args.background}")
        background_frame = cv2.imread(args.background)
        if background_frame is None:
            logger.error(f"Failed to load background image: {args.background}")
            sys.exit(1)

        # Set background for cue detection
        debugger.cue_detector.set_background_frame(background_frame)

        # Detect pockets from background to exclude them from ball detection
        # Note: We do NOT enable background_subtraction (which hurts stationary ball detection)
        debugger.detector.pocket_locations = debugger.detector._detect_pockets_from_background(background_frame)
        logger.info(f"Background frame loaded for cue detection, detected {len(debugger.detector.pocket_locations)} pockets")

    debugger.run()


if __name__ == "__main__":
    main()
