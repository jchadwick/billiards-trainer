"""Integration Service - Connects Vision → Core → Broadcast data flow.

This service is the critical missing piece that ties all modules together:
1. Polls Vision for detection results
2. Updates Core game state with detection data
3. Subscribes to Core events for state changes
4. Triggers WebSocket broadcasts on state updates
5. Calculates and broadcasts trajectories when cue detected
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from backend.config import Config, config
from backend.core import CoreModule
from backend.core.models import BallState, CueState, TableState, Vector2D
from backend.core.physics.trajectory import (
    MultiballTrajectoryResult,
    TrajectoryCalculator,
    TrajectoryQuality,
)
from backend.core.validation.physics import PhysicsValidator
from backend.core.validation.table_state import TableStateValidator
from backend.integration_service_conversion_helpers import StateConversionHelpers
from backend.vision import VisionModule
from backend.vision.models import Ball, CueStick, DetectionResult

logger = logging.getLogger(__name__)


class BroadcastErrorType(Enum):
    """Types of broadcast errors for retry logic."""

    TRANSIENT = "transient"  # Network errors, temporary failures - retry
    VALIDATION = "validation"  # Data validation errors - don't retry
    UNKNOWN = "unknown"  # Unknown errors - retry with caution


@dataclass
class BroadcastMetrics:
    """Metrics for broadcast operations."""

    successful_broadcasts: int = 0
    failed_broadcasts: int = 0
    total_retries: int = 0

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.successful_broadcasts + self.failed_broadcasts
        return (self.successful_broadcasts / total * 100) if total > 0 else 0.0


class CircuitBreaker:
    """Circuit breaker pattern for broadcast failures.

    Implements the Circuit Breaker pattern to prevent cascade failures in WebSocket
    broadcasting. When multiple consecutive broadcast failures occur (e.g., all clients
    disconnect), the circuit "opens" and stops attempting broadcasts for a timeout period.

    REFACTORING NOTE: This is a new addition to improve system resilience. Previously,
    broadcast failures would continuously spam error logs and waste CPU cycles attempting
    to send to disconnected clients.

    States:
    - CLOSED: Normal operation, broadcasts are attempted
    - OPEN: Too many failures, broadcasts are blocked
    - HALF-OPEN: Timeout elapsed, trying one broadcast to test if system recovered

    Behavior:
    - After N consecutive failures (default 10), circuit opens
    - While open, all broadcast attempts are blocked (return immediately)
    - After timeout period (default 30s), circuit enters half-open state
    - One successful broadcast in half-open state closes the circuit
    - One failed broadcast in half-open state reopens the circuit

    This prevents:
    - CPU waste from attempting broadcasts when all clients are disconnected
    - Log spam from repeated broadcast failures
    - Resource exhaustion from queued broadcast tasks
    """

    def __init__(self, failure_threshold: int = 10, timeout_seconds: float = 30.0):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before circuit opens
                             (default: 10 failures)
            timeout_seconds: Seconds to wait before attempting to close circuit
                           (default: 30 seconds)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.consecutive_failures = 0
        self.circuit_open_time: Optional[float] = None
        self.is_open = False

    def record_success(self) -> None:
        """Record a successful operation.

        Resets the circuit breaker to CLOSED state and clears failure counters.
        This is called after any successful broadcast attempt.
        """
        if self.is_open:
            logger.info("Circuit breaker closing after successful operation")
        # Reset all state - back to normal operation
        self.consecutive_failures = 0
        self.circuit_open_time = None
        self.is_open = False

    def record_failure(self) -> None:
        """Record a failed operation.

        Increments the consecutive failure counter. If threshold is reached,
        opens the circuit and starts the timeout period.
        """
        self.consecutive_failures += 1

        # Check if we've hit the threshold to open the circuit
        if not self.is_open and self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            self.circuit_open_time = time.time()
            logger.error(
                f"Circuit breaker OPENED after {self.consecutive_failures} consecutive failures. "
                f"Will retry after {self.timeout_seconds} seconds."
            )

    def can_attempt(self) -> bool:
        """Check if operation can be attempted.

        This is called before each broadcast attempt to determine if the operation
        should proceed or be blocked.

        Returns:
            True if circuit is closed (normal operation) or timeout has elapsed
            (entering half-open state). False if circuit is open and timeout has
            not yet elapsed (blocking broadcasts).
        """
        # Circuit closed - normal operation
        if not self.is_open:
            return True

        # Circuit open - check if timeout has elapsed to enter half-open state
        if (
            self.circuit_open_time
            and (time.time() - self.circuit_open_time) >= self.timeout_seconds
        ):
            logger.info(
                f"Circuit breaker timeout elapsed ({self.timeout_seconds}s). "
                "Attempting to close circuit..."
            )
            # Half-open state: allow one attempt to test if system recovered
            # If this attempt succeeds, record_success() will close the circuit
            # If this attempt fails, record_failure() will reopen the circuit
            self.is_open = False
            return True

        # Circuit still open and timeout not elapsed - block the operation
        return False

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        return {
            "is_open": self.is_open,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "seconds_until_retry": (
                max(0, self.timeout_seconds - (time.time() - self.circuit_open_time))
                if self.circuit_open_time
                else 0
            ),
        }


class IntegrationService:
    """Manages data flow between Vision, Core, and Broadcasting systems."""

    def __init__(
        self,
        vision_module: VisionModule,
        core_module: CoreModule,
        message_broadcaster: Any,  # MessageBroadcaster from api.websocket
        config_module: Optional[Config] = None,
    ):
        """Initialize integration service.

        Args:
            vision_module: Vision processing module
            core_module: Core game state and physics module
            message_broadcaster: WebSocket broadcaster
            config_module: Configuration module (optional, will use singleton if not provided)
        """
        self.vision = vision_module
        self.core = core_module
        self.broadcaster = message_broadcaster
        self.running = False
        self.integration_task: Optional[asyncio.Task[None]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize trajectory calculator for multiball predictions
        self.trajectory_calculator = TrajectoryCalculator()

        # Initialize validators
        self.physics_validator = PhysicsValidator()
        self.table_state_validator = TableStateValidator()

        # Configuration (use provided or singleton)
        self.config = config_module if config_module is not None else config

        # Initialize state conversion helpers
        self.state_converter = StateConversionHelpers(
            config=self.config, physics_validator=self.physics_validator
        )
        self.target_fps = self.config.get("integration.target_fps", 30)
        self.log_interval_frames = self.config.get(
            "integration.log_interval_frames", 300
        )
        self.error_retry_delay = self.config.get(
            "integration.error_retry_delay_sec", 0.1
        )
        self.shot_speed_estimate = self.config.get(
            "integration.shot_speed_estimate_m_per_s", 2.0
        )

        # Validation thresholds
        self.max_ball_velocity = self.config.get(
            "integration.max_ball_velocity_m_per_s", 10.0
        )
        self.max_position_x = self.config.get("integration.max_position_x", 3.0)
        self.max_position_y = self.config.get("integration.max_position_y", 2.0)

        # Broadcast retry configuration
        self.max_retries = self.config.get("integration.broadcast_max_retries", 3)
        self.retry_base_delay = self.config.get(
            "integration.broadcast_retry_base_delay_sec", 0.1
        )
        self.circuit_breaker_threshold = self.config.get(
            "integration.circuit_breaker_threshold", 10
        )
        self.circuit_breaker_timeout = self.config.get(
            "integration.circuit_breaker_timeout_sec", 30.0
        )

        # Performance tracking
        self.frame_count = 0
        self.error_count = 0
        self.last_detection_time = 0.0
        self.trajectory_frame_count = 0  # Track frames for periodic trajectory logging

        # Broadcast error recovery
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker_threshold,
            timeout_seconds=self.circuit_breaker_timeout,
        )
        self.broadcast_metrics = BroadcastMetrics()

    async def start(self) -> None:
        """Start the integration service."""
        if self.running:
            logger.warning("Integration service already running")
            return

        logger.info("Starting integration service...")
        self.running = True

        # Capture the event loop for use in event callbacks
        self._event_loop = asyncio.get_running_loop()

        # Start vision module camera capture
        logger.info("Starting vision module camera capture...")
        try:
            if self.vision.start_capture():
                logger.info("Vision camera capture started successfully")
            else:
                logger.error(
                    "Failed to start vision camera capture - integration will continue "
                    "but no detections will be available until camera is started"
                )
        except Exception as e:
            logger.error(
                f"Error starting vision camera capture: {e} - "
                "integration will continue but no detections will be available"
            )

        # Check vision calibration status
        try:
            # Check if vision module has calibration data
            if hasattr(self.vision, "calibrator") and self.vision.calibrator:
                if not self.vision.calibrator.is_calibrated():
                    logger.warning(
                        "Vision module is not calibrated - detection accuracy may be poor. "
                        "Please run camera calibration via API endpoint: "
                        "POST /api/v1/vision/calibration/camera/auto-calibrate"
                    )
            else:
                logger.warning(
                    "Vision module has no calibrator - using uncalibrated camera"
                )
        except Exception as e:
            logger.debug(f"Could not check calibration status: {e}")

        # Subscribe to Core events
        self._subscribe_to_core_events()

        # Start integration loop
        self.integration_task = asyncio.create_task(self._integration_loop())
        logger.info("Integration service started successfully")

    async def stop(self) -> None:
        """Stop the integration service."""
        if not self.running:
            return

        logger.info("Stopping integration service...")
        self.running = False

        if self.integration_task:
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass

        # Stop vision module camera capture
        logger.info("Stopping vision module camera capture...")
        try:
            self.vision.stop_capture()
            logger.info("Vision camera capture stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping vision camera capture: {e}")

        logger.info("Integration service stopped")

    def _subscribe_to_core_events(self) -> None:
        """Subscribe to Core module events for broadcasting."""
        # Subscribe to state updates
        self.core.subscribe_to_events("state_updated", self._on_state_updated)

        # Subscribe to trajectory calculations
        self.core.subscribe_to_events(
            "trajectory_calculated", self._on_trajectory_calculated
        )

        logger.info("Subscribed to Core module events")

    async def _integration_loop(self) -> None:
        """Main integration loop - processes vision detections at configured FPS."""
        frame_interval = 1.0 / self.target_fps

        logger.info(f"Integration loop starting at {self.target_fps} FPS")

        while self.running:
            try:
                loop_start = asyncio.get_event_loop().time()

                # Get detection result from vision
                detection_result = await self._get_vision_detection()

                if detection_result:
                    # Convert to Core format and update state
                    await self._process_detection(detection_result)

                    self.frame_count += 1

                    # Log progress periodically
                    if self.frame_count % self.log_interval_frames == 0:
                        logger.info(
                            f"Integration: {self.frame_count} frames processed, "
                            f"{self.error_count} errors"
                        )

                # Maintain target FPS
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in integration loop: {e}", exc_info=True)
                await asyncio.sleep(self.error_retry_delay)

        logger.info("Integration loop stopped")

    async def _get_vision_detection(self) -> Optional[DetectionResult]:
        """Get detection result from vision module.

        Returns:
            DetectionResult if available, None otherwise
        """
        try:
            # Process current frame
            detection = self.vision.process_frame()
            return detection
        except Exception as e:
            logger.error(f"Failed to get vision detection: {e}")
            return None

    async def _process_detection(self, detection: DetectionResult) -> None:
        """Process detection result and update Core.

        Args:
            detection: Detection result from vision
        """
        # Convert detection to Core format
        detection_data = self._convert_detection_to_core_format(detection)

        # Update Core state
        try:
            await self.core.update_state(detection_data)
        except Exception as e:
            logger.error(f"Failed to update Core state: {e}", exc_info=True)

        # Check if we should calculate trajectory
        await self._check_trajectory_calculation(detection)

    def vision_cue_to_cue_state(
        self, detected_cue: CueStick, timestamp: Optional[float] = None
    ) -> Optional[CueState]:
        """Convert vision CueStick detection to core CueState with validation.

        This is the recommended public method for converting CueStick detections.

        Args:
            detected_cue: Vision module CueStick detection result
            timestamp: Optional timestamp for last_update (defaults to current time)

        Returns:
            Core CueState object with all required fields, or None if validation fails
        """
        return self.state_converter.vision_cue_to_cue_state(detected_cue, timestamp)

    def vision_ball_to_ball_state(
        self,
        ball: Ball,
        is_target: bool = False,
        timestamp: Optional[float] = None,
    ) -> Optional[BallState]:
        """Convert vision Ball detection to core BallState with validation.

        This is the recommended public method for converting Ball detections.

        Args:
            ball: Vision module Ball detection result
            is_target: Whether this is the target ball for trajectory calculation
            timestamp: Optional timestamp for last_update (defaults to current time)

        Returns:
            Core BallState object with all required fields, or None if validation fails
        """
        return self.state_converter.vision_ball_to_ball_state(
            ball, is_target, timestamp
        )

    def _create_cue_state(self, detected_cue: CueStick) -> CueState:
        """Create a core CueState from a vision CueStick detection.

        DEPRECATED: Use vision_cue_to_cue_state() instead for better validation.

        This method converts Vision module's CueStick detection (pixel coordinates, degrees)
        into Core module's CueState (physics-ready with force estimation).

        Args:
            detected_cue: Vision module CueStick detection result

        Returns:
            Core CueState object with all required fields for physics calculations
        """
        return self.vision_cue_to_cue_state(detected_cue)

    def _create_ball_state(self, ball: Ball, is_target: bool = False) -> BallState:
        """Create a core BallState from a vision Ball detection.

        DEPRECATED: Use vision_ball_to_ball_state() instead for better validation.

        This method converts Vision module's Ball detection (pixel coordinates, OpenCV types)
        into Core module's BallState (physics-ready with SI units and persistent IDs).

        Args:
            ball: Vision module Ball detection result
            is_target: Whether this is the target ball for trajectory calculation

        Returns:
            Core BallState object with all required fields for physics calculations
        """
        return self.vision_ball_to_ball_state(ball, is_target)

    def _create_ball_states(
        self, balls: list[Ball], exclude_ball: Optional[Ball] = None
    ) -> list[BallState]:
        """Create a list of core BallState objects from vision Ball detections.

        DEPRECATED: This is kept for backwards compatibility but uses the new
        validation-enabled conversion internally.

        Args:
            balls: List of vision module Ball detection results
            exclude_ball: Optional ball to exclude from the list (e.g., the target ball)

        Returns:
            List of core BallState objects (non-None results only)
        """
        ball_states = []
        for ball in balls:
            # Skip the excluded ball if specified
            if exclude_ball is not None and ball == exclude_ball:
                continue

            ball_state = self.vision_ball_to_ball_state(ball, is_target=False)
            # Only include successfully converted balls
            if ball_state is not None:
                ball_states.append(ball_state)

        return ball_states

    def _convert_detection_to_core_format(
        self, detection: DetectionResult
    ) -> dict[str, Any]:
        """Convert vision DetectionResult to Core's expected format.

        Args:
            detection: Vision detection result

        Returns:
            Dictionary in Core's expected format
        """
        detection_data: dict[str, Any] = {
            "timestamp": detection.timestamp,
            "frame_number": detection.frame_number,
        }

        # Convert balls
        if detection.balls:
            detection_data["balls"] = [
                {
                    # Use track_id if available, otherwise generate ID from position
                    "id": (
                        ball.track_id
                        if ball.track_id is not None
                        else hash(
                            (
                                round(ball.position[0]),
                                round(ball.position[1]),
                                ball.ball_type.value,
                            )
                        )
                        % 10000
                    ),
                    "position": {"x": ball.position[0], "y": ball.position[1]},
                    "velocity": {"x": ball.velocity[0], "y": ball.velocity[1]},
                    "is_moving": ball.is_moving,
                    "number": ball.number,
                    "type": ball.ball_type.value if ball.ball_type else "unknown",
                    "is_cue_ball": (
                        ball.ball_type.value == "cue" if ball.ball_type else False
                    ),
                    "confidence": ball.confidence,
                }
                for ball in detection.balls
            ]

        # Convert cue
        if detection.cue:
            cue = detection.cue
            detection_data["cue"] = {
                "tip_position": {"x": cue.tip_position[0], "y": cue.tip_position[1]},
                "angle": cue.angle,
                "length": cue.length,
                "state": cue.state.value if cue.state else "unknown",
                "confidence": cue.confidence,
            }

        # Convert table
        if detection.table:
            table = detection.table
            detection_data["table"] = {
                "corners": [
                    {"x": float(corner[0]), "y": float(corner[1])}
                    for corner in table.corners
                ],
                "width": table.width,
                "height": table.height,
                "pockets": [
                    {
                        "position": {"x": pocket.position[0], "y": pocket.position[1]},
                        "type": (
                            pocket.pocket_type.value
                            if pocket.pocket_type
                            else "unknown"
                        ),
                    }
                    for pocket in (table.pockets or [])
                ],
            }

        return detection_data

    def _find_ball_cue_is_pointing_at(self, cue: Any, balls: Any) -> Optional[Any]:
        """Find which ball the cue is currently pointing at.

        Args:
            cue: Detected cue stick with tip and tail position
            balls: List of detected balls

        Returns:
            The ball the cue is pointing at, or None
        """
        if not cue or not balls:
            return None

        # Calculate cue direction vector
        cue_dx = cue.tip_position[0] - cue.butt_position[0]
        cue_dy = cue.tip_position[1] - cue.butt_position[1]
        cue_length = np.sqrt(cue_dx**2 + cue_dy**2)

        if cue_length == 0:
            return None

        # Normalize direction
        cue_dx /= cue_length
        cue_dy /= cue_length

        # Find the closest ball along the cue direction
        closest_ball = None
        min_distance = float("inf")
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
            if (
                perpendicular_distance < max_perpendicular_distance
                and distance_along_cue < min_distance
            ):
                min_distance = distance_along_cue
                closest_ball = ball

        return closest_ball

    async def _check_trajectory_calculation(self, detection: DetectionResult) -> None:
        """Check if trajectory should be calculated and trigger if needed.

        Args:
            detection: Current detection result
        """
        # Increment trajectory frame counter for periodic logging
        self.trajectory_frame_count += 1
        should_log = (self.trajectory_frame_count % 30) == 0  # Log every 30 frames

        # Only calculate trajectory if:
        # 1. Cue is detected (removed aiming state requirement for more flexibility)
        # 2. All balls are stationary
        # 3. At least one ball is detected

        if not detection.cue:
            if should_log:
                logger.debug("Trajectory calculation skipped: No cue detected")
            return

        # Log cue detection with parameters
        if should_log:
            logger.debug(
                f"Cue detected - angle: {detection.cue.angle:.2f}deg, "
                f"tip: ({detection.cue.tip_position[0]:.1f}, {detection.cue.tip_position[1]:.1f}), "
                f"confidence: {detection.cue.confidence:.2f}"
            )

        # Check if all balls are stationary (no velocity data in detection)
        # This is a simplification - in real implementation, check Core state
        if not detection.balls:
            if should_log:
                logger.warning("Trajectory calculation skipped: No balls detected")
            return

        if should_log:
            logger.debug(
                f"Ball positions being used for trajectory: {len(detection.balls)} balls - "
                + ", ".join(
                    [
                        f"ball_{i}@({ball.position[0]:.1f},{ball.position[1]:.1f})"
                        for i, ball in enumerate(detection.balls)
                    ]
                )
            )

        # Find the ball the cue is pointing at (could be any ball, not just cue ball)
        target_ball = self._find_ball_cue_is_pointing_at(detection.cue, detection.balls)
        if not target_ball:
            if should_log:
                logger.warning(
                    f"Trajectory calculation skipped: No target ball found in cue direction. "
                    f"Cue pointing at angle {detection.cue.angle:.2f}deg from "
                    f"({detection.cue.tip_position[0]:.1f}, {detection.cue.tip_position[1]:.1f})"
                )
            return

        if should_log:
            logger.debug(
                f"Target ball found at ({target_ball.position[0]:.1f}, {target_ball.position[1]:.1f}), "
                f"type: {target_ball.ball_type.value if target_ball.ball_type else 'unknown'}"
            )

        # Calculate multiball trajectory using trajectory calculator directly
        try:
            if should_log:
                logger.debug("Starting trajectory calculation...")

            # Create CueState from detected cue using state converter (with validation)
            cue_state = self.state_converter.vision_cue_to_cue_state(
                detection.cue, timestamp=detection.timestamp
            )
            if cue_state is None:
                logger.error(
                    "Failed to convert cue state - aborting trajectory calculation"
                )
                return

            # Create BallState for target ball using state converter (with validation)
            ball_state = self.state_converter.vision_ball_to_ball_state(
                target_ball, is_target=True, timestamp=detection.timestamp
            )
            if ball_state is None:
                logger.error(
                    "Failed to convert target ball state - aborting trajectory calculation"
                )
                return

            # Create list of BallState for other balls using state converter (with validation)
            other_ball_states = []
            for ball in detection.balls:
                if ball == target_ball:
                    continue  # Skip the target ball
                ball_state_converted = self.state_converter.vision_ball_to_ball_state(
                    ball, is_target=False, timestamp=detection.timestamp
                )
                if ball_state_converted is not None:
                    other_ball_states.append(ball_state_converted)

            if should_log:
                logger.debug(
                    f"Trajectory calculation inputs - "
                    f"cue angle: {cue_state.angle:.2f}deg, "
                    f"cue force: {cue_state.estimated_force:.2f}, "
                    f"target ball: ({ball_state.position.x:.1f}, {ball_state.position.y:.1f}), "
                    f"other balls: {len(other_ball_states)}"
                )

            # Get table state from Core's current state
            if not self.core._current_state:
                logger.warning(
                    "Trajectory calculation skipped: No current game state available. "
                    "Core module may not be initialized properly."
                )
                return

            table_state = self.core._current_state.table

            # Validate table state before trajectory calculation
            is_valid, validation_errors = (
                self.table_state_validator.validate_for_trajectory(
                    table_state, require_playing_area=False
                )
            )
            if not is_valid:
                logger.error(
                    f"Trajectory calculation skipped: Invalid table state. "
                    f"Errors: {'; '.join(validation_errors)}"
                )
                return

            # Log validation summary periodically
            if should_log:
                summary = self.table_state_validator.get_validation_summary(table_state)
                logger.debug(f"Table state validation: {summary}")

            # Call trajectory_calculator.predict_multiball_cue_shot() directly
            multiball_result = self.trajectory_calculator.predict_multiball_cue_shot(
                cue_state=cue_state,
                ball_state=ball_state,
                table_state=table_state,
                other_balls=other_ball_states,
                quality=TrajectoryQuality.LOW,  # Use low quality for real-time performance
                max_collision_depth=5,  # Calculate up to 5 collision levels deep
            )

            # Log trajectory calculation results
            total_lines = sum(
                len(traj.points) - 1 if len(traj.points) >= 2 else 0
                for traj in multiball_result.trajectories.values()
            )
            total_collisions = sum(
                len(traj.collisions) for traj in multiball_result.trajectories.values()
            )

            if should_log:
                logger.debug(
                    f"Trajectory calculation results - "
                    f"balls affected: {len(multiball_result.trajectories)}, "
                    f"total line segments: {total_lines}, "
                    f"total collisions: {total_collisions}"
                )

                # Log detailed collision info
                for ball_id, traj in multiball_result.trajectories.items():
                    if traj.collisions:
                        collision_types = [c.type.value for c in traj.collisions]
                        logger.debug(
                            f"  Ball {ball_id}: {len(traj.collisions)} collisions - "
                            f"types: {', '.join(collision_types)}"
                        )

            # Convert multiball result to event format and emit
            await self._emit_multiball_trajectory(multiball_result)

            logger.info(
                f"Trajectory calculated and broadcast - "
                f"{len(multiball_result.trajectories)} ball(s) affected, "
                f"{total_lines} line segments, {total_collisions} collisions"
            )

        except Exception as e:
            logger.error(
                f"Trajectory calculation failed - "
                f"Error: {str(e)}, "
                f"Context: cue_angle={detection.cue.angle:.2f}deg, "
                f"cue_tip=({detection.cue.tip_position[0]:.1f},{detection.cue.tip_position[1]:.1f}), "
                f"balls_count={len(detection.balls)}, "
                f"target_ball={'found' if target_ball else 'not_found'}",
                exc_info=True,
            )

    def _estimate_velocity_from_cue(self, cue: Any) -> dict[str, float]:
        """Estimate initial velocity from cue angle and estimated force.

        Args:
            cue: Detected cue stick

        Returns:
            Velocity dictionary with vx, vy components
        """
        # Estimate force based on cue position/state
        # This is a simplified estimation - real implementation would use
        # cue motion tracking or user input
        estimated_speed = self.shot_speed_estimate

        # Convert angle to velocity components
        angle_rad = np.deg2rad(cue.angle)
        vx = estimated_speed * np.cos(angle_rad)
        vy = estimated_speed * np.sin(angle_rad)

        return {"vx": vx, "vy": vy, "vz": 0.0}

    def _classify_broadcast_error(self, error: Exception) -> BroadcastErrorType:
        """Classify a broadcast error to determine retry strategy.

        Args:
            error: The exception that occurred

        Returns:
            BroadcastErrorType indicating how to handle the error
        """
        error_message = str(error).lower()

        # Validation errors - don't retry
        validation_keywords = [
            "validation",
            "invalid",
            "schema",
            "type error",
            "value error",
            "missing required",
        ]
        if any(keyword in error_message for keyword in validation_keywords):
            return BroadcastErrorType.VALIDATION

        # Transient errors - retry
        transient_keywords = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "unavailable",
            "refused",
        ]
        if any(keyword in error_message for keyword in transient_keywords):
            return BroadcastErrorType.TRANSIENT

        # Unknown - retry with caution
        return BroadcastErrorType.UNKNOWN

    async def _broadcast_with_retry(
        self,
        broadcast_func: Callable[..., Any],
        operation_name: str,
        data_summary: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Execute a broadcast with retry logic and circuit breaker.

        Args:
            broadcast_func: Async function to call for broadcasting
            operation_name: Name of the operation for logging
            data_summary: Summary of data being broadcast for logging
            *args: Arguments to pass to broadcast function
            **kwargs: Keyword arguments to pass to broadcast function

        Returns:
            True if broadcast succeeded, False otherwise
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            logger.warning(
                f"Circuit breaker OPEN - skipping {operation_name}. "
                f"Status: {self.circuit_breaker.get_status()}"
            )
            self.broadcast_metrics.failed_broadcasts += 1
            return False

        last_error = None
        error_type = BroadcastErrorType.UNKNOWN

        # Attempt broadcast with retries
        for attempt in range(self.max_retries + 1):
            try:
                await broadcast_func(*args, **kwargs)

                # Success!
                self.circuit_breaker.record_success()
                self.broadcast_metrics.successful_broadcasts += 1

                if attempt > 0:
                    logger.info(
                        f"{operation_name} succeeded after {attempt} retries. "
                        f"Data: {data_summary}"
                    )

                return True

            except Exception as e:
                last_error = e
                error_type = self._classify_broadcast_error(e)

                # Log error with context
                logger.error(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Error type: {error_type.value}. "
                    f"Data: {data_summary}. "
                    f"Error: {str(e)}"
                )

                # Don't retry validation errors
                if error_type == BroadcastErrorType.VALIDATION:
                    logger.error(
                        f"{operation_name} validation error - will not retry. "
                        f"Data: {data_summary}. "
                        f"Error: {str(e)}"
                    )
                    break

                # Calculate exponential backoff delay for retry
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt)
                    logger.debug(f"Retrying {operation_name} in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    self.broadcast_metrics.total_retries += 1

        # All attempts failed
        self.circuit_breaker.record_failure()
        self.broadcast_metrics.failed_broadcasts += 1

        logger.error(
            f"{operation_name} failed after {self.max_retries + 1} attempts. "
            f"Error type: {error_type.value}. "
            f"Data: {data_summary}. "
            f"Final error: {str(last_error)}. "
            f"Action: Giving up. "
            f"Circuit breaker status: {self.circuit_breaker.get_status()}"
        )

        return False

    def _on_state_updated(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Handle Core state update events.

        This is a synchronous callback that schedules async work in the background.

        Args:
            event_type: Type of event (e.g., "state_updated")
            event_data: Event data containing updated state (flattened GameState dict)
        """
        # Schedule async processing using the stored event loop
        if self._event_loop is None:
            logger.error(
                "No event loop available - cannot process state update. "
                "Integration service may not be started properly."
            )
            return

        # Schedule the coroutine on the stored event loop
        try:
            self._event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._on_state_updated_async(event_type, event_data),
                    loop=self._event_loop,
                )
            )
        except Exception as e:
            logger.error(f"Failed to schedule state update processing: {e}")

    async def _on_state_updated_async(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Async handler for Core state update events.

        Args:
            event_type: Type of event (e.g., "state_updated")
            event_data: Event data containing updated state (flattened GameState dict)
        """
        # Validate event_data structure
        if not isinstance(event_data, dict):
            logger.error(
                f"Invalid event_data type: expected dict, got {type(event_data).__name__}. "
                f"Event will be skipped."
            )
            return

        # Event data is now flattened (directly from asdict(GameState))
        # Validate required fields exist
        required_fields = ["balls", "table", "timestamp"]
        missing_fields = [field for field in required_fields if field not in event_data]
        if missing_fields:
            logger.error(
                f"Missing required fields in event_data: {missing_fields}. "
                f"Available keys: {list(event_data.keys())}. "
                "Event will be skipped."
            )
            return

        # Extract data for broadcasting (directly from flattened event_data)
        balls = event_data.get("balls", [])
        cue = event_data.get("cue")
        table = event_data.get("table")

        # Validate balls is a list
        if not isinstance(balls, list):
            logger.error(
                f"Invalid balls type: expected list, got {type(balls).__name__}. "
                f"Balls value: {balls}. "
                "Event will be skipped."
            )
            return

        # Validate each ball is a dict
        for i, ball in enumerate(balls):
            if not isinstance(ball, dict):
                logger.error(
                    f"Invalid ball type at index {i}: expected dict, got {type(ball).__name__}. "
                    f"Ball value: {ball}. "
                    "Event will be skipped."
                )
                return

        # Validate cue is dict or None
        if cue is not None and not isinstance(cue, dict):
            logger.error(
                f"Invalid cue type: expected dict or None, got {type(cue).__name__}. "
                f"Cue value: {cue}. "
                "Event will be skipped."
            )
            return

        # Validate table is dict or None
        if table is not None and not isinstance(table, dict):
            logger.error(
                f"Invalid table type: expected dict or None, got {type(table).__name__}. "
                f"Table value: {table}. "
                "Event will be skipped."
            )
            return

        # Convert position dicts to lists for broadcaster compatibility
        # The broadcaster expects positions as [x, y] but asdict() converts Vector2D to {'x': ..., 'y': ...}
        balls_converted = []
        for ball in balls:
            ball_copy = ball.copy()
            position = ball_copy.get("position")
            if isinstance(position, dict) and "x" in position and "y" in position:
                ball_copy["position"] = [position["x"], position["y"]]
            # Also convert velocity if present
            velocity = ball_copy.get("velocity")
            if isinstance(velocity, dict) and "x" in velocity and "y" in velocity:
                ball_copy["velocity"] = [velocity["x"], velocity["y"]]
            balls_converted.append(ball_copy)

        # Create summary for logging
        data_summary = f"{len(balls)} balls, cue={'present' if cue else 'absent'}, table={'present' if table else 'absent'}"

        # Log detailed structure on first broadcast to help with debugging
        if not hasattr(self, "_logged_state_structure"):
            logger.debug(
                f"State structure sample - "
                f"balls keys: {list(balls[0].keys()) if balls else 'no balls'}, "
                f"cue keys: {list(cue.keys()) if cue else 'no cue'}, "
                f"table keys: {list(table.keys()) if table else 'no table'}"
            )
            self._logged_state_structure = True

        # Broadcast with retry logic
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_game_state,
            "broadcast_game_state",
            data_summary,
            balls_converted,
            cue,
            table,
        )

    def _on_trajectory_calculated(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Handle trajectory calculation events.

        This is a synchronous callback that schedules async work in the background.

        Args:
            event_type: Type of event (e.g., "trajectory_calculated")
            event_data: Event data containing trajectory
        """
        # Schedule async processing using the stored event loop
        if self._event_loop is None:
            logger.error(
                "No event loop available - cannot process trajectory calculation. "
                "Integration service may not be started properly."
            )
            return

        # Schedule the coroutine on the stored event loop
        try:
            self._event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._on_trajectory_calculated_async(event_type, event_data),
                    loop=self._event_loop,
                )
            )
        except Exception as e:
            logger.error(f"Failed to schedule trajectory calculation processing: {e}")

    async def _on_trajectory_calculated_async(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Async handler for trajectory calculation events.

        Args:
            event_type: Type of event (e.g., "trajectory_calculated")
            event_data: Event data containing trajectory
        """
        trajectory = event_data.get("trajectory")
        if not trajectory:
            return

        # Extract trajectory lines and collision points
        lines = trajectory.get("lines", [])
        collisions = trajectory.get("collisions", [])

        # Create summary for logging
        data_summary = f"{len(lines)} lines, {len(collisions)} collisions"

        # Broadcast with retry logic
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_trajectory,
            "broadcast_trajectory",
            data_summary,
            lines,
            collisions,
        )

    async def _emit_multiball_trajectory(
        self, multiball_result: MultiballTrajectoryResult
    ) -> None:
        """Convert MultiballTrajectoryResult to event format and emit/broadcast.

        Args:
            multiball_result: Multiball trajectory calculation result
        """
        # Collect all lines and collisions from all ball trajectories
        all_lines = []
        all_collisions = []

        # Process each trajectory in the result
        for ball_id, trajectory in multiball_result.trajectories.items():
            # Determine line type: primary for cue ball, secondary for hit balls
            line_type = (
                "primary"
                if ball_id == multiball_result.primary_ball_id
                else "secondary"
            )

            # Convert trajectory points to line segments
            if trajectory.points and len(trajectory.points) >= 2:
                for i in range(len(trajectory.points) - 1):
                    start_point = trajectory.points[i]
                    end_point = trajectory.points[i + 1]
                    all_lines.append(
                        {
                            "start": [start_point.position.x, start_point.position.y],
                            "end": [end_point.position.x, end_point.position.y],
                            "type": line_type,
                            "confidence": trajectory.success_probability,
                            "ball_id": ball_id,
                        }
                    )

            # Convert collisions to event format
            for collision in trajectory.collisions:
                all_collisions.append(
                    {
                        "time": collision.time,
                        "position": [collision.position.x, collision.position.y],
                        "x": collision.position.x,
                        "y": collision.position.y,
                        "type": collision.type.value,
                        "ball_id": collision.ball2_id,  # Target ball for frontend compatibility
                        "ball1_id": collision.ball1_id,  # Moving ball
                        "ball2_id": collision.ball2_id,  # Target ball (None for cushion/pocket)
                        "angle": collision.impact_angle,
                        "confidence": collision.confidence,
                    }
                )

        # Create summary for logging
        data_summary = (
            f"{len(all_lines)} lines, {len(all_collisions)} collisions "
            f"across {len(multiball_result.trajectories)} ball(s)"
        )

        # Broadcast with retry logic
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_trajectory,
            "broadcast_multiball_trajectory",
            data_summary,
            all_lines,
            all_collisions,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get integration service statistics.

        Returns:
            Dictionary with service statistics
        """
        return {
            "running": self.running,
            "frames_processed": self.frame_count,
            "errors": self.error_count,
            "error_rate": (
                self.error_count / self.frame_count if self.frame_count > 0 else 0
            ),
            "broadcast_metrics": {
                "successful_broadcasts": self.broadcast_metrics.successful_broadcasts,
                "failed_broadcasts": self.broadcast_metrics.failed_broadcasts,
                "total_retries": self.broadcast_metrics.total_retries,
                "success_rate_percent": self.broadcast_metrics.success_rate(),
            },
            "circuit_breaker": self.circuit_breaker.get_status(),
        }
