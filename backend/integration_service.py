"""Integration Service - Connects Vision → Core → Broadcast data flow.

This service is the critical missing piece that ties all modules together:
1. Polls Vision for detection results
2. Updates Core game state with detection data
3. Subscribes to Core events for state changes
4. Triggers WebSocket/UDP broadcasts on state updates
5. Calculates and broadcasts trajectories when cue detected
"""

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from backend.core import CoreModule
from backend.vision import VisionModule
from backend.vision.models import DetectionResult

logger = logging.getLogger(__name__)


class IntegrationService:
    """Manages data flow between Vision, Core, and Broadcasting systems."""

    def __init__(
        self,
        vision_module: VisionModule,
        core_module: CoreModule,
        message_broadcaster: Any,  # MessageBroadcaster from api.websocket
    ):
        """Initialize integration service.

        Args:
            vision_module: Vision processing module
            core_module: Core game state and physics module
            message_broadcaster: WebSocket/UDP broadcaster
        """
        self.vision = vision_module
        self.core = core_module
        self.broadcaster = message_broadcaster
        self.running = False
        self.integration_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.frame_count = 0
        self.error_count = 0
        self.last_detection_time = 0.0

    async def start(self) -> None:
        """Start the integration service."""
        if self.running:
            logger.warning("Integration service already running")
            return

        logger.info("Starting integration service...")
        self.running = True

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
        """Main integration loop - processes vision detections at 30 FPS."""
        target_fps = 30
        frame_interval = 1.0 / target_fps

        logger.info(f"Integration loop starting at {target_fps} FPS")

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
                    if self.frame_count % 300 == 0:  # Every 10 seconds at 30fps
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
                await asyncio.sleep(0.1)  # Brief pause on error

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
                    "id": ball.id,
                    "position": {"x": ball.position[0], "y": ball.position[1]},
                    "velocity": {"x": ball.velocity[0], "y": ball.velocity[1]},
                    "is_moving": ball.is_moving,
                    "number": ball.number,
                    "type": ball.ball_type.value if ball.ball_type else "unknown",
                    "is_cue_ball": ball.number == 0,
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

    async def _check_trajectory_calculation(self, detection: DetectionResult) -> None:
        """Check if trajectory should be calculated and trigger if needed.

        Args:
            detection: Current detection result
        """
        # Only calculate trajectory if:
        # 1. Cue is detected and in aiming state
        # 2. All balls are stationary
        # 3. Cue ball is detected

        if not detection.cue:
            return

        # Check if cue is aiming (not striking)
        if detection.cue.state.value not in ["aiming", "ready"]:
            return

        # Check if all balls are stationary (no velocity data in detection)
        # This is a simplification - in real implementation, check Core state
        if not detection.balls:
            return

        # Find cue ball
        cue_ball = next((b for b in detection.balls if b.number == 0), None)
        if not cue_ball:
            return

        # Estimate velocity from cue angle and force
        try:
            velocity = self._estimate_velocity_from_cue(detection.cue)

            # Calculate trajectory
            await self.core.calculate_trajectory(
                ball_id=cue_ball.id, initial_velocity=velocity
            )

            # Trajectory broadcast will be handled by event subscription
            logger.debug(f"Trajectory calculated for cue ball {cue_ball.id}")

        except Exception as e:
            logger.error(f"Failed to calculate trajectory: {e}")

    def _estimate_velocity_from_cue(self, cue) -> dict[str, float]:
        """Estimate initial velocity from cue angle and estimated force.

        Args:
            cue: Detected cue stick

        Returns:
            Velocity dictionary with vx, vy components
        """
        # Estimate force based on cue position/state
        # This is a simplified estimation - real implementation would use
        # cue motion tracking or user input
        estimated_speed = 2.0  # m/s - moderate shot

        # Convert angle to velocity components
        angle_rad = np.deg2rad(cue.angle)
        vx = estimated_speed * np.cos(angle_rad)
        vy = estimated_speed * np.sin(angle_rad)

        return {"vx": vx, "vy": vy, "vz": 0.0}

    async def _on_state_updated(self, event_data: dict[str, Any]) -> None:
        """Handle Core state update events.

        Args:
            event_data: Event data containing updated state
        """
        try:
            state = event_data.get("state")
            if not state:
                return

            # Extract data for broadcasting
            balls = state.get("balls", [])
            cue = state.get("cue")
            table = state.get("table")

            # Broadcast to WebSocket clients
            await self.broadcaster.broadcast_game_state(balls, cue, table)

            logger.debug(f"Broadcast game state: {len(balls)} balls")

        except Exception as e:
            logger.error(f"Failed to broadcast state update: {e}", exc_info=True)

    async def _on_trajectory_calculated(self, event_data: dict[str, Any]) -> None:
        """Handle trajectory calculation events.

        Args:
            event_data: Event data containing trajectory
        """
        try:
            trajectory = event_data.get("trajectory")
            if not trajectory:
                return

            # Extract trajectory lines and collision points
            lines = trajectory.get("lines", [])
            collisions = trajectory.get("collisions", [])

            # Broadcast to WebSocket/UDP
            await self.broadcaster.broadcast_trajectory(lines, collisions)

            logger.debug(
                f"Broadcast trajectory: {len(lines)} lines, {len(collisions)} collisions"
            )

        except Exception as e:
            logger.error(f"Failed to broadcast trajectory: {e}", exc_info=True)

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
        }
