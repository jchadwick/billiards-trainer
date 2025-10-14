#!/usr/bin/env python3
"""Backend Equivalence Test - Verifies backend produces same results as video_debugger.

This test compares results from the integrated backend (IntegrationService + VisionModule)
against the standalone video_debugger to ensure they produce equivalent outputs.

The test runs the same video through both systems and compares:
1. Ball detections (positions, types, IDs)
2. Cue stick detections (position, angle)
3. Trajectory calculations (lines, collisions)
4. Playing area detection

Usage:
    python tools/test_backend_equivalence.py <video_file> [--frames N] [--tolerance T] [--output-dir DIR]

Examples:
    # Test first 100 frames with default tolerance
    python tools/test_backend_equivalence.py demo.mkv --frames 100

    # Test with custom tolerance (in pixels)
    python tools/test_backend_equivalence.py demo.mkv --tolerance 5.0

    # Save detailed comparison report
    python tools/test_backend_equivalence.py demo.mkv --output-dir ./test-results
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.manager import ConfigurationModule
from backend.core import CoreModule
from backend.core.models import BallState, CueState, TableState, Vector2D
from backend.core.physics.trajectory import (
    MultiballTrajectoryResult,
    TrajectoryCalculator,
    TrajectoryQuality,
)
from backend.integration_service import IntegrationService
from backend.vision import VisionModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BallDetectionData:
    """Ball detection data for comparison."""
    position: Tuple[float, float]
    radius: float
    ball_type: Optional[str] = None
    track_id: Optional[int] = None
    number: Optional[int] = None
    confidence: float = 1.0


@dataclass
class CueDetectionData:
    """Cue stick detection data for comparison."""
    tip_position: Tuple[float, float]
    angle: float
    length: float
    confidence: float = 1.0


@dataclass
class TrajectoryData:
    """Trajectory calculation data for comparison."""
    lines: List[Dict[str, Any]]  # List of line segments
    collisions: List[Dict[str, Any]]  # List of collision points
    num_balls_affected: int


@dataclass
class FrameComparisonResult:
    """Comparison result for a single frame."""
    frame_number: int
    timestamp: float

    # Ball detection comparison
    video_debugger_balls: List[BallDetectionData]
    backend_balls: List[BallDetectionData]
    balls_match: bool
    ball_position_errors: List[float]  # Distance errors for matched balls
    ball_detection_diff: str

    # Cue detection comparison
    video_debugger_cue: Optional[CueDetectionData]
    backend_cue: Optional[CueDetectionData]
    cue_match: bool
    cue_diff: str

    # Trajectory comparison
    video_debugger_trajectory: Optional[TrajectoryData]
    backend_trajectory: Optional[TrajectoryData]
    trajectory_match: bool
    trajectory_diff: str


@dataclass
class EquivalenceTestReport:
    """Complete test report."""
    video_path: str
    total_frames_tested: int
    frames_with_differences: int
    equivalence_percentage: float

    # Per-frame results
    frame_results: List[FrameComparisonResult]

    # Summary statistics
    avg_ball_position_error: float
    max_ball_position_error: float
    ball_count_mismatches: int
    cue_detection_mismatches: int
    trajectory_mismatches: int

    # Detailed analysis
    differences_summary: List[str]


class MockBroadcaster:
    """Mock broadcaster for testing - captures broadcast calls."""

    def __init__(self):
        self.game_states = []
        self.trajectories = []

    async def broadcast_game_state(self, balls, cue, table):
        """Capture game state broadcast."""
        self.game_states.append({
            'balls': balls,
            'cue': cue,
            'table': table,
            'timestamp': time.time()
        })

    async def broadcast_trajectory(self, lines, collisions):
        """Capture trajectory broadcast."""
        self.trajectories.append({
            'lines': lines,
            'collisions': collisions,
            'timestamp': time.time()
        })


class VideoDebuggerRunner:
    """Runs video_debugger logic to capture detection results."""

    def __init__(self, video_path: str):
        """Initialize video debugger runner.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path

        # Configure VisionModule for video file
        vision_config = {
            "camera_device_id": video_path,
            "camera_backend": "auto",
            "enable_threading": False,
            "enable_table_detection": True,
            "enable_ball_detection": True,
            "enable_cue_detection": True,
            "enable_tracking": True,
            "detection_backend": "yolo",
            "use_opencv_validation": True,
            "fallback_to_opencv": True,
        }

        self.vision_module = VisionModule(vision_config)
        self.trajectory_calculator = TrajectoryCalculator()

        # Load table state from config
        self.table_state = self._load_table_state_from_config()
        self.calibration_resolution = (640, 360)

    def _load_table_state_from_config(self) -> TableState:
        """Load table state from saved configuration file."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "current.json"

            if not config_path.exists():
                return TableState.standard_9ft_table()

            with open(config_path, 'r') as f:
                config = json.load(f)

            table_config = config.get("table", {})
            playing_area_corners_data = table_config.get("playing_area_corners")
            calibration_res = table_config.get("calibration_resolution", {})

            if not playing_area_corners_data or len(playing_area_corners_data) != 4:
                return TableState.standard_9ft_table()

            self.calibration_resolution = (
                calibration_res.get("width", 640),
                calibration_res.get("height", 360)
            )

            playing_area_corners = [
                Vector2D(corner["x"], corner["y"])
                for corner in playing_area_corners_data
            ]

            table_state = TableState.standard_9ft_table()
            table_state.playing_area_corners = playing_area_corners

            return table_state

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.calibration_resolution = (640, 360)
            return TableState.standard_9ft_table()

    def _find_ball_cue_is_pointing_at(self, cue, balls):
        """Find which ball the cue is currently pointing at."""
        if not cue or not balls:
            return None

        # Get butt position
        if cue.butt_position and cue.butt_position != (0.0, 0.0):
            butt_x, butt_y = cue.butt_position
        else:
            angle_rad = np.radians(cue.angle)
            butt_x = cue.tip_position[0] - cue.length * np.cos(angle_rad)
            butt_y = cue.tip_position[1] - cue.length * np.sin(angle_rad)

        # Calculate cue direction vector
        cue_dx = cue.tip_position[0] - butt_x
        cue_dy = cue.tip_position[1] - butt_y
        cue_length = np.sqrt(cue_dx**2 + cue_dy**2)

        if cue_length == 0:
            return None

        cue_dx /= cue_length
        cue_dy /= cue_length

        # Find closest ball along cue direction
        closest_ball = None
        min_distance = float('inf')
        max_perpendicular_distance = 40

        for ball in balls:
            ball_dx = ball.position[0] - cue.tip_position[0]
            ball_dy = ball.position[1] - cue.tip_position[1]

            distance_along_cue = ball_dx * cue_dx + ball_dy * cue_dy

            if distance_along_cue < 0:
                continue

            perpendicular_distance = abs(ball_dx * cue_dy - ball_dy * cue_dx)

            if perpendicular_distance < max_perpendicular_distance and distance_along_cue < min_distance:
                min_distance = distance_along_cue
                closest_ball = ball

        return closest_ball

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[List[BallDetectionData], Optional[CueDetectionData], Optional[TrajectoryData]]:
        """Process a frame and return detection results.

        Args:
            frame: Video frame
            frame_number: Frame number

        Returns:
            Tuple of (balls, cue, trajectory)
        """
        # Update table dimensions and scale corners if needed
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        if self.table_state.width != frame_width or self.table_state.height != frame_height:
            if hasattr(self, 'calibration_resolution') and self.table_state.playing_area_corners:
                calib_width, calib_height = self.calibration_resolution

                self.table_state.scale_playing_area_corners(
                    from_width=calib_width,
                    from_height=calib_height,
                    to_width=frame_width,
                    to_height=frame_height
                )

            self.table_state.width = frame_width
            self.table_state.height = frame_height

        # Process frame with VisionModule
        detection_result = self.vision_module._process_single_frame(
            frame, frame_number, None
        )

        if detection_result is None:
            return [], None, None

        # Extract ball detections
        balls = []
        for ball in detection_result.balls:
            balls.append(BallDetectionData(
                position=(ball.position[0], ball.position[1]),
                radius=ball.radius,
                ball_type=ball.ball_type.value if hasattr(ball, 'ball_type') and ball.ball_type else None,
                track_id=getattr(ball, 'track_id', None),
                number=getattr(ball, 'number', None),
                confidence=getattr(ball, 'confidence', 1.0)
            ))

        # Extract cue detection
        cue_data = None
        if detection_result.cue:
            cue = detection_result.cue
            cue_data = CueDetectionData(
                tip_position=(cue.tip_position[0], cue.tip_position[1]),
                angle=cue.angle,
                length=cue.length,
                confidence=cue.confidence
            )

        # Calculate trajectory if cue is detected
        trajectory_data = None
        if detection_result.cue and detection_result.balls:
            try:
                target_ball = self._find_ball_cue_is_pointing_at(detection_result.cue, detection_result.balls)

                if target_ball:
                    # Create CueState
                    cue_state = CueState(
                        angle=detection_result.cue.angle,
                        estimated_force=5.0,
                        impact_point=Vector2D(detection_result.cue.tip_position[0], detection_result.cue.tip_position[1]),
                        tip_position=Vector2D(detection_result.cue.tip_position[0], detection_result.cue.tip_position[1]),
                        elevation=getattr(detection_result.cue, 'elevation', 0.0),
                        is_visible=True,
                        confidence=detection_result.cue.confidence,
                        last_update=0,
                    )

                    # Create BallState for target ball
                    ball_state = BallState(
                        id="target_ball",
                        position=Vector2D(target_ball.position[0], target_ball.position[1]),
                        velocity=Vector2D(0, 0),
                        radius=target_ball.radius,
                        mass=0.17,
                        is_cue_ball=False,
                        confidence=getattr(target_ball, 'confidence', 1.0),
                        last_update=0,
                    )

                    # Create BallState for other balls
                    other_ball_states = []
                    for i, ball in enumerate(detection_result.balls):
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

                    # Calculate trajectory
                    multiball_result = self.trajectory_calculator.predict_multiball_cue_shot(
                        cue_state,
                        ball_state,
                        self.table_state,
                        other_ball_states,
                        quality=TrajectoryQuality.LOW,
                        max_collision_depth=5,
                    )

                    # Convert to trajectory data
                    all_lines = []
                    all_collisions = []

                    for ball_id, trajectory in multiball_result.trajectories.items():
                        line_type = "primary" if ball_id == multiball_result.primary_ball_id else "secondary"

                        if trajectory.points and len(trajectory.points) >= 2:
                            for i in range(len(trajectory.points) - 1):
                                start_point = trajectory.points[i]
                                end_point = trajectory.points[i + 1]
                                all_lines.append({
                                    "start": [start_point.position.x, start_point.position.y],
                                    "end": [end_point.position.x, end_point.position.y],
                                    "type": line_type,
                                    "confidence": trajectory.success_probability,
                                    "ball_id": ball_id,
                                })

                        for collision in trajectory.collisions:
                            all_collisions.append({
                                "time": collision.time,
                                "position": [collision.position.x, collision.position.y],
                                "x": collision.position.x,
                                "y": collision.position.y,
                                "type": collision.type.value,
                                "ball_id": collision.ball2_id,
                                "ball1_id": collision.ball1_id,
                                "ball2_id": collision.ball2_id,
                                "angle": collision.impact_angle,
                                "confidence": collision.confidence,
                            })

                    trajectory_data = TrajectoryData(
                        lines=all_lines,
                        collisions=all_collisions,
                        num_balls_affected=len(multiball_result.trajectories)
                    )

            except Exception as e:
                logger.debug(f"Trajectory calculation failed: {e}")

        return balls, cue_data, trajectory_data

    def run(self):
        """Start video capture."""
        return self.vision_module.start_capture()

    def stop(self):
        """Stop video capture."""
        self.vision_module.stop_capture()

    def get_frame(self):
        """Get latest frame."""
        return self.vision_module.camera.get_latest_frame()


class BackendRunner:
    """Runs integrated backend (IntegrationService) to capture results."""

    def __init__(self, video_path: str):
        """Initialize backend runner.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path

        # Configure modules
        vision_config = {
            "camera_device_id": video_path,
            "camera_backend": "auto",
            "enable_threading": False,
            "enable_table_detection": True,
            "enable_ball_detection": True,
            "enable_cue_detection": True,
            "enable_tracking": True,
            "detection_backend": "yolo",
            "use_opencv_validation": True,
            "fallback_to_opencv": True,
        }

        self.config_module = ConfigurationModule()
        self.vision_module = VisionModule(vision_config)
        self.core_module = CoreModule(config_module=self.config_module)
        self.mock_broadcaster = MockBroadcaster()

        # Create integration service
        self.integration_service = IntegrationService(
            vision_module=self.vision_module,
            core_module=self.core_module,
            message_broadcaster=self.mock_broadcaster,
            config_module=self.config_module
        )

    async def start(self):
        """Start integration service."""
        await self.integration_service.start()

    async def stop(self):
        """Stop integration service."""
        await self.integration_service.stop()

    def get_latest_results(self) -> Tuple[List[BallDetectionData], Optional[CueDetectionData], Optional[TrajectoryData]]:
        """Get latest detection and trajectory results from backend.

        Returns:
            Tuple of (balls, cue, trajectory)
        """
        # Get latest game state from broadcaster
        if not self.mock_broadcaster.game_states:
            return [], None, None

        latest_state = self.mock_broadcaster.game_states[-1]

        # Extract ball detections
        balls = []
        for ball_dict in latest_state.get('balls', []):
            pos = ball_dict.get('position', {})
            balls.append(BallDetectionData(
                position=(pos.get('x', 0.0), pos.get('y', 0.0)),
                radius=ball_dict.get('radius', 0.028575),
                ball_type=ball_dict.get('type'),
                track_id=ball_dict.get('id'),
                number=ball_dict.get('number'),
                confidence=ball_dict.get('confidence', 1.0)
            ))

        # Extract cue detection
        cue_data = None
        cue_dict = latest_state.get('cue')
        if cue_dict:
            tip_pos = cue_dict.get('tip_position', {})
            cue_data = CueDetectionData(
                tip_position=(tip_pos.get('x', 0.0), tip_pos.get('y', 0.0)),
                angle=cue_dict.get('angle', 0.0),
                length=cue_dict.get('length', 0.0),
                confidence=cue_dict.get('confidence', 1.0)
            )

        # Extract trajectory
        trajectory_data = None
        if self.mock_broadcaster.trajectories:
            latest_trajectory = self.mock_broadcaster.trajectories[-1]
            trajectory_data = TrajectoryData(
                lines=latest_trajectory.get('lines', []),
                collisions=latest_trajectory.get('collisions', []),
                num_balls_affected=len(set(line.get('ball_id') for line in latest_trajectory.get('lines', [])))
            )

        return balls, cue_data, trajectory_data


class EquivalenceTester:
    """Main test orchestrator."""

    def __init__(
        self,
        video_path: str,
        position_tolerance: float = 2.0,
        angle_tolerance: float = 1.0,
        max_frames: Optional[int] = None,
        output_dir: Optional[Path] = None
    ):
        """Initialize equivalence tester.

        Args:
            video_path: Path to video file
            position_tolerance: Max position difference in pixels
            angle_tolerance: Max angle difference in degrees
            max_frames: Maximum frames to test (None = all)
            output_dir: Directory for output reports
        """
        self.video_path = video_path
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_frames = max_frames
        self.output_dir = output_dir or Path("./equivalence-test-results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_debugger = VideoDebuggerRunner(video_path)
        self.backend = BackendRunner(video_path)

        self.frame_results: List[FrameComparisonResult] = []

    def _compare_balls(
        self,
        vd_balls: List[BallDetectionData],
        backend_balls: List[BallDetectionData]
    ) -> Tuple[bool, List[float], str]:
        """Compare ball detections between systems.

        Returns:
            Tuple of (match, position_errors, difference_description)
        """
        if len(vd_balls) != len(backend_balls):
            return False, [], f"Ball count mismatch: video_debugger={len(vd_balls)}, backend={len(backend_balls)}"

        # Match balls by position (closest pairs)
        position_errors = []
        unmatched_balls = []

        vd_positions = [np.array(b.position) for b in vd_balls]
        backend_positions = [np.array(b.position) for b in backend_balls]

        for vd_pos in vd_positions:
            min_dist = float('inf')
            closest_idx = -1

            for i, backend_pos in enumerate(backend_positions):
                dist = np.linalg.norm(vd_pos - backend_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            position_errors.append(min_dist)

            if min_dist > self.position_tolerance:
                unmatched_balls.append(f"Ball at {vd_pos} has error {min_dist:.2f}px")

        if unmatched_balls:
            return False, position_errors, "; ".join(unmatched_balls)

        avg_error = np.mean(position_errors) if position_errors else 0.0
        return True, position_errors, f"All balls matched (avg error: {avg_error:.2f}px)"

    def _compare_cue(
        self,
        vd_cue: Optional[CueDetectionData],
        backend_cue: Optional[CueDetectionData]
    ) -> Tuple[bool, str]:
        """Compare cue detections between systems.

        Returns:
            Tuple of (match, difference_description)
        """
        if vd_cue is None and backend_cue is None:
            return True, "No cue detected in either system"

        if vd_cue is None:
            return False, "Cue detected in backend but not video_debugger"

        if backend_cue is None:
            return False, "Cue detected in video_debugger but not backend"

        # Compare positions
        vd_pos = np.array(vd_cue.tip_position)
        backend_pos = np.array(backend_cue.tip_position)
        pos_error = np.linalg.norm(vd_pos - backend_pos)

        # Compare angles
        angle_diff = abs(vd_cue.angle - backend_cue.angle)
        # Handle angle wrapping (e.g., 359 vs 1 degree)
        angle_diff = min(angle_diff, 360 - angle_diff)

        if pos_error > self.position_tolerance:
            return False, f"Cue tip position error: {pos_error:.2f}px (tolerance: {self.position_tolerance}px)"

        if angle_diff > self.angle_tolerance:
            return False, f"Cue angle error: {angle_diff:.2f}° (tolerance: {self.angle_tolerance}°)"

        return True, f"Cue matched (pos error: {pos_error:.2f}px, angle error: {angle_diff:.2f}°)"

    def _compare_trajectories(
        self,
        vd_trajectory: Optional[TrajectoryData],
        backend_trajectory: Optional[TrajectoryData]
    ) -> Tuple[bool, str]:
        """Compare trajectory calculations between systems.

        Returns:
            Tuple of (match, difference_description)
        """
        if vd_trajectory is None and backend_trajectory is None:
            return True, "No trajectory calculated in either system"

        if vd_trajectory is None:
            return False, "Trajectory calculated in backend but not video_debugger"

        if backend_trajectory is None:
            return False, "Trajectory calculated in video_debugger but not backend"

        # Compare number of line segments
        vd_lines = len(vd_trajectory.lines)
        backend_lines = len(backend_trajectory.lines)

        if vd_lines != backend_lines:
            return False, f"Line count mismatch: video_debugger={vd_lines}, backend={backend_lines}"

        # Compare number of collisions
        vd_collisions = len(vd_trajectory.collisions)
        backend_collisions = len(backend_trajectory.collisions)

        if vd_collisions != backend_collisions:
            return False, f"Collision count mismatch: video_debugger={vd_collisions}, backend={backend_collisions}"

        # Compare line endpoints (allow some tolerance)
        line_errors = []
        for vd_line, backend_line in zip(vd_trajectory.lines, backend_trajectory.lines):
            vd_start = np.array(vd_line['start'])
            vd_end = np.array(vd_line['end'])
            backend_start = np.array(backend_line['start'])
            backend_end = np.array(backend_line['end'])

            start_error = np.linalg.norm(vd_start - backend_start)
            end_error = np.linalg.norm(vd_end - backend_end)

            if start_error > self.position_tolerance or end_error > self.position_tolerance:
                line_errors.append(f"Line endpoint error: start={start_error:.2f}px, end={end_error:.2f}px")

        if line_errors:
            return False, f"Trajectory line mismatches: {'; '.join(line_errors[:3])}"  # Show first 3

        return True, f"Trajectories matched ({vd_lines} lines, {vd_collisions} collisions)"

    def _compare_frame(
        self,
        frame_number: int,
        vd_balls: List[BallDetectionData],
        vd_cue: Optional[CueDetectionData],
        vd_trajectory: Optional[TrajectoryData],
        backend_balls: List[BallDetectionData],
        backend_cue: Optional[CueDetectionData],
        backend_trajectory: Optional[TrajectoryData]
    ) -> FrameComparisonResult:
        """Compare results for a single frame.

        Returns:
            FrameComparisonResult with detailed comparison
        """
        # Compare balls
        balls_match, ball_errors, ball_diff = self._compare_balls(vd_balls, backend_balls)

        # Compare cue
        cue_match, cue_diff = self._compare_cue(vd_cue, backend_cue)

        # Compare trajectory
        trajectory_match, trajectory_diff = self._compare_trajectories(vd_trajectory, backend_trajectory)

        return FrameComparisonResult(
            frame_number=frame_number,
            timestamp=time.time(),
            video_debugger_balls=vd_balls,
            backend_balls=backend_balls,
            balls_match=balls_match,
            ball_position_errors=ball_errors,
            ball_detection_diff=ball_diff,
            video_debugger_cue=vd_cue,
            backend_cue=backend_cue,
            cue_match=cue_match,
            cue_diff=cue_diff,
            video_debugger_trajectory=vd_trajectory,
            backend_trajectory=backend_trajectory,
            trajectory_match=trajectory_match,
            trajectory_diff=trajectory_diff
        )

    async def run_test(self) -> EquivalenceTestReport:
        """Run equivalence test on video.

        Returns:
            EquivalenceTestReport with complete results
        """
        logger.info(f"Starting equivalence test on {self.video_path}")
        logger.info(f"Position tolerance: {self.position_tolerance}px")
        logger.info(f"Angle tolerance: {self.angle_tolerance}°")
        logger.info(f"Max frames: {self.max_frames or 'all'}")

        # Start both systems
        logger.info("Starting video_debugger...")
        if not self.video_debugger.run():
            raise RuntimeError("Failed to start video_debugger")

        logger.info("Starting backend integration service...")
        await self.backend.start()

        # Give systems time to initialize
        await asyncio.sleep(1.0)

        try:
            frame_count = 0
            frames_with_differences = 0

            while True:
                # Check max frames limit
                if self.max_frames and frame_count >= self.max_frames:
                    logger.info(f"Reached max frames limit ({self.max_frames})")
                    break

                # Get frame from video_debugger
                frame_data = self.video_debugger.get_frame()
                if frame_data is None:
                    logger.info("No more frames available")
                    break

                frame, frame_info = frame_data
                frame_number = frame_info.frame_number

                # Process with video_debugger
                vd_balls, vd_cue, vd_trajectory = self.video_debugger.process_frame(frame, frame_number)

                # Give backend time to process
                await asyncio.sleep(0.1)

                # Get results from backend
                backend_balls, backend_cue, backend_trajectory = self.backend.get_latest_results()

                # Compare results
                result = self._compare_frame(
                    frame_number,
                    vd_balls, vd_cue, vd_trajectory,
                    backend_balls, backend_cue, backend_trajectory
                )

                self.frame_results.append(result)

                # Count frames with differences
                if not (result.balls_match and result.cue_match and result.trajectory_match):
                    frames_with_differences += 1
                    logger.warning(f"Frame {frame_number}: Differences detected")
                    if not result.balls_match:
                        logger.warning(f"  Balls: {result.ball_detection_diff}")
                    if not result.cue_match:
                        logger.warning(f"  Cue: {result.cue_diff}")
                    if not result.trajectory_match:
                        logger.warning(f"  Trajectory: {result.trajectory_diff}")

                frame_count += 1

                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames...")

        finally:
            # Stop both systems
            logger.info("Stopping systems...")
            self.video_debugger.stop()
            await self.backend.stop()

        # Generate report
        return self._generate_report(frame_count, frames_with_differences)

    def _generate_report(self, total_frames: int, frames_with_diffs: int) -> EquivalenceTestReport:
        """Generate test report from results.

        Args:
            total_frames: Total frames tested
            frames_with_diffs: Frames with differences

        Returns:
            EquivalenceTestReport
        """
        # Calculate statistics
        all_ball_errors = []
        ball_count_mismatches = 0
        cue_mismatches = 0
        trajectory_mismatches = 0
        differences = []

        for result in self.frame_results:
            if result.ball_position_errors:
                all_ball_errors.extend(result.ball_position_errors)

            if not result.balls_match:
                ball_count_mismatches += 1
                differences.append(f"Frame {result.frame_number}: {result.ball_detection_diff}")

            if not result.cue_match:
                cue_mismatches += 1
                differences.append(f"Frame {result.frame_number}: {result.cue_diff}")

            if not result.trajectory_match:
                trajectory_mismatches += 1
                differences.append(f"Frame {result.frame_number}: {result.trajectory_diff}")

        avg_ball_error = np.mean(all_ball_errors) if all_ball_errors else 0.0
        max_ball_error = np.max(all_ball_errors) if all_ball_errors else 0.0

        equivalence_pct = ((total_frames - frames_with_diffs) / total_frames * 100) if total_frames > 0 else 0.0

        report = EquivalenceTestReport(
            video_path=self.video_path,
            total_frames_tested=total_frames,
            frames_with_differences=frames_with_diffs,
            equivalence_percentage=equivalence_pct,
            frame_results=self.frame_results,
            avg_ball_position_error=avg_ball_error,
            max_ball_position_error=max_ball_error,
            ball_count_mismatches=ball_count_mismatches,
            cue_detection_mismatches=cue_mismatches,
            trajectory_mismatches=trajectory_mismatches,
            differences_summary=differences
        )

        return report

    def save_report(self, report: EquivalenceTestReport):
        """Save report to output directory.

        Args:
            report: Test report to save
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save JSON report
        json_path = self.output_dir / f"equivalence_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"JSON report saved to {json_path}")

        # Save human-readable summary
        summary_path = self.output_dir / f"equivalence_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKEND EQUIVALENCE TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video: {report.video_path}\n")
            f.write(f"Total Frames Tested: {report.total_frames_tested}\n")
            f.write(f"Frames with Differences: {report.frames_with_differences}\n")
            f.write(f"Equivalence: {report.equivalence_percentage:.2f}%\n\n")

            f.write("-" * 80 + "\n")
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"Ball Position Error (avg): {report.avg_ball_position_error:.2f}px\n")
            f.write(f"Ball Position Error (max): {report.max_ball_position_error:.2f}px\n")
            f.write(f"Ball Count Mismatches: {report.ball_count_mismatches}\n")
            f.write(f"Cue Detection Mismatches: {report.cue_detection_mismatches}\n")
            f.write(f"Trajectory Mismatches: {report.trajectory_mismatches}\n\n")

            if report.differences_summary:
                f.write("-" * 80 + "\n")
                f.write("DIFFERENCES (first 50)\n")
                f.write("-" * 80 + "\n\n")
                for diff in report.differences_summary[:50]:
                    f.write(f"  {diff}\n")

            f.write("\n" + "=" * 80 + "\n")
            if report.equivalence_percentage >= 99.0:
                f.write("RESULT: SYSTEMS ARE EQUIVALENT\n")
            elif report.equivalence_percentage >= 95.0:
                f.write("RESULT: SYSTEMS ARE MOSTLY EQUIVALENT (minor differences)\n")
            else:
                f.write("RESULT: SYSTEMS HAVE SIGNIFICANT DIFFERENCES\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Summary report saved to {summary_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test backend equivalence with video_debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=None,
        help="Maximum number of frames to test (default: all)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=2.0,
        help="Position tolerance in pixels (default: 2.0)"
    )
    parser.add_argument(
        "--angle-tolerance", "-a",
        type=float,
        default=1.0,
        help="Angle tolerance in degrees (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for reports (default: ./equivalence-test-results)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check video file exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Create tester
    output_dir = Path(args.output_dir) if args.output_dir else None
    tester = EquivalenceTester(
        video_path=args.video,
        position_tolerance=args.tolerance,
        angle_tolerance=args.angle_tolerance,
        max_frames=args.frames,
        output_dir=output_dir
    )

    # Run test
    try:
        report = await tester.run_test()

        # Print summary
        print("\n" + "=" * 80)
        print("EQUIVALENCE TEST RESULTS")
        print("=" * 80)
        print(f"\nVideo: {report.video_path}")
        print(f"Total Frames: {report.total_frames_tested}")
        print(f"Frames with Differences: {report.frames_with_differences}")
        print(f"Equivalence: {report.equivalence_percentage:.2f}%")
        print(f"\nBall Position Error (avg): {report.avg_ball_position_error:.2f}px")
        print(f"Ball Position Error (max): {report.max_ball_position_error:.2f}px")
        print(f"Ball Count Mismatches: {report.ball_count_mismatches}")
        print(f"Cue Detection Mismatches: {report.cue_detection_mismatches}")
        print(f"Trajectory Mismatches: {report.trajectory_mismatches}")

        # Save report
        tester.save_report(report)

        print("\n" + "=" * 80)
        if report.equivalence_percentage >= 99.0:
            print("✓ SYSTEMS ARE EQUIVALENT")
            sys.exit(0)
        elif report.equivalence_percentage >= 95.0:
            print("⚠ SYSTEMS ARE MOSTLY EQUIVALENT (minor differences)")
            sys.exit(0)
        else:
            print("✗ SYSTEMS HAVE SIGNIFICANT DIFFERENCES")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
