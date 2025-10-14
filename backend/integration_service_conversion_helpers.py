"""State conversion helper methods for IntegrationService.

This module contains improved conversion methods that:
1. Create proper BallState/CueState objects that match trajectory calculation needs
2. Add validation during conversion
3. Provide helper methods for common conversions
4. Ensure consistency between state conversion and trajectory calculation
"""

import logging
import time
from typing import Optional

from backend.core.models import BallState, CueState, Vector2D
from backend.core.validation.physics import PhysicsValidator
from backend.vision.models import Ball, CueStick

logger = logging.getLogger(__name__)


class StateConversionHelpers:
    """Helper class for converting Vision detections to Core states with validation."""

    def __init__(
        self, config=None, physics_validator: Optional[PhysicsValidator] = None
    ):
        """Initialize state conversion helpers.

        Args:
            config: Configuration object
            physics_validator: Optional physics validator for validation during conversion
        """
        self.config = config
        self.physics_validator = physics_validator or PhysicsValidator()

        # Validation thresholds from config or defaults
        if config:
            self.max_ball_velocity = config.get(
                "integration.max_ball_velocity_m_per_s", 10.0
            )
            self.max_position_x = config.get("integration.max_position_x", 3.0)
            self.max_position_y = config.get("integration.max_position_y", 2.0)
            self.min_ball_confidence = config.get(
                "integration.min_ball_confidence", 0.1
            )
            self.min_cue_confidence = config.get("integration.min_cue_confidence", 0.1)
        else:
            self.max_ball_velocity = 10.0
            self.max_position_x = 3.0
            self.max_position_y = 2.0
            self.min_ball_confidence = 0.1
            self.min_cue_confidence = 0.1

        # Conversion counters for periodic logging
        self._ball_conversion_count = 0
        self._cue_conversion_count = 0
        self._validation_warnings = 0

    def vision_ball_to_ball_state(
        self,
        ball: Ball,
        is_target: bool = False,
        timestamp: Optional[float] = None,
        validate: bool = True,
    ) -> Optional[BallState]:
        """Convert vision Ball detection to core BallState with validation.

        Args:
            ball: Vision module Ball detection result
            is_target: Whether this is the target ball for trajectory calculation
            timestamp: Optional timestamp for last_update (defaults to current time)
            validate: Whether to validate the ball state (recommended: True)

        Returns:
            Core BallState object with all required fields, or None if validation fails critically
        """
        self._ball_conversion_count += 1

        # Validate confidence threshold
        if ball.confidence < self.min_ball_confidence:
            logger.warning(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"Low confidence {ball.confidence:.3f} < {self.min_ball_confidence}, "
                f"but including anyway"
            )
            self._validation_warnings += 1

        # Validate position is reasonable
        if not self._validate_position(ball.position[0], ball.position[1], "ball"):
            # Position is out of bounds - log warning but continue
            self._validation_warnings += 1

        # Validate velocity is reasonable
        if not self._validate_velocity(ball.velocity[0], ball.velocity[1], "ball"):
            # Velocity is too high - clamp it
            velocity_mag = (ball.velocity[0] ** 2 + ball.velocity[1] ** 2) ** 0.5
            if velocity_mag > 0:
                scale = self.max_ball_velocity / velocity_mag
                ball.velocity = (ball.velocity[0] * scale, ball.velocity[1] * scale)
                logger.warning(
                    f"Ball conversion #{self._ball_conversion_count}: "
                    f"Velocity clamped from {velocity_mag:.2f} to {self.max_ball_velocity:.2f} m/s"
                )
                self._validation_warnings += 1

        # Generate ball ID - use track_id if available, otherwise create from position/type
        ball_id = self._generate_ball_id(ball, is_target)

        # Create BallState
        ball_state = BallState(
            id=ball_id,
            position=Vector2D(ball.position[0], ball.position[1]),
            velocity=Vector2D(ball.velocity[0], ball.velocity[1]),
            radius=(
                ball.radius if ball.radius > 0 else 0.028575
            ),  # Standard radius fallback
            mass=0.17,  # Standard pool ball mass in kg
            spin=Vector2D.zero(),  # Vision doesn't detect spin yet
            is_cue_ball=(ball.ball_type.value == "cue" if ball.ball_type else False),
            is_pocketed=False,  # Vision detections are only for balls on table
            number=ball.number,
            confidence=ball.confidence,
            last_update=timestamp if timestamp is not None else time.time(),
        )

        # Validate the ball state using physics validator if requested
        if validate:
            validation_result = self.physics_validator.validate_ball_state(
                ball_state, table=None
            )
            if not validation_result.is_valid:
                logger.error(
                    f"Ball conversion #{self._ball_conversion_count}: "
                    f"FAILED validation - "
                    f"Errors: {[e.message for e in validation_result.errors]}"
                )
                # For now, still return the ball state but log the error
                # In strict mode, we could return None
                self._validation_warnings += len(validation_result.errors)

            if validation_result.warnings:
                logger.debug(
                    f"Ball conversion #{self._ball_conversion_count}: "
                    f"Warnings: {[w.message for w in validation_result.warnings]}"
                )
                self._validation_warnings += len(validation_result.warnings)

        # Log conversion details periodically
        if self._ball_conversion_count % 100 == 0:
            logger.debug(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"id={ball_state.id}, "
                f"pos=({ball_state.position.x:.1f},{ball_state.position.y:.1f}), "
                f"vel=({ball_state.velocity.x:.2f},{ball_state.velocity.y:.2f}), "
                f"confidence={ball_state.confidence:.2f}, "
                f"is_cue={ball_state.is_cue_ball}"
            )
            logger.info(
                f"Ball conversion stats: {self._ball_conversion_count} conversions, "
                f"{self._validation_warnings} warnings"
            )

        return ball_state

    def vision_cue_to_cue_state(
        self,
        detected_cue: CueStick,
        timestamp: Optional[float] = None,
        validate: bool = True,
    ) -> Optional[CueState]:
        """Convert vision CueStick detection to core CueState with validation.

        Args:
            detected_cue: Vision module CueStick detection result
            timestamp: Optional timestamp for last_update (defaults to current time)
            validate: Whether to validate the cue state (recommended: True)

        Returns:
            Core CueState object with all required fields, or None if validation fails critically
        """
        self._cue_conversion_count += 1

        # Validate confidence threshold
        if detected_cue.confidence < self.min_cue_confidence:
            logger.warning(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Low confidence {detected_cue.confidence:.3f} < {self.min_cue_confidence}"
            )
            self._validation_warnings += 1

        # Validate cue angle is reasonable (0-360 degrees)
        if detected_cue.angle < 0 or detected_cue.angle > 360:
            logger.warning(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Angle {detected_cue.angle:.2f}deg out of range [0, 360], normalizing"
            )
            # Normalize angle to 0-360
            detected_cue.angle = detected_cue.angle % 360
            self._validation_warnings += 1

        # Validate tip position is reasonable
        if not self._validate_position(
            detected_cue.tip_position[0], detected_cue.tip_position[1], "cue"
        ):
            self._validation_warnings += 1

        # Get estimated force from config or use default
        estimated_force = (
            self.config.get("cue.default_force", 5.0) if self.config else 5.0
        )

        # Validate force is reasonable
        if estimated_force < 0 or estimated_force > 50.0:
            logger.warning(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Force {estimated_force:.2f}N out of reasonable range [0, 50], clamping"
            )
            estimated_force = max(0.0, min(50.0, estimated_force))
            self._validation_warnings += 1

        # Create tip position vector
        tip_position = Vector2D(
            detected_cue.tip_position[0], detected_cue.tip_position[1]
        )

        # Create cue state
        cue_state = CueState(
            angle=detected_cue.angle,
            estimated_force=estimated_force,
            impact_point=tip_position,  # For now, impact point is same as tip position
            tip_position=tip_position,
            elevation=0.0,  # Vision doesn't currently detect elevation
            length=detected_cue.length if hasattr(detected_cue, "length") else 1.47,
            is_visible=True,
            confidence=detected_cue.confidence,
            last_update=timestamp if timestamp is not None else time.time(),
        )

        # Log conversion details periodically
        if self._cue_conversion_count % 100 == 0:
            logger.debug(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"angle={cue_state.angle:.2f}deg, "
                f"force={cue_state.estimated_force:.2f}N, "
                f"tip=({cue_state.tip_position.x:.1f},{cue_state.tip_position.y:.1f}), "
                f"confidence={cue_state.confidence:.2f}"
            )
            logger.info(
                f"Cue conversion stats: {self._cue_conversion_count} conversions, "
                f"{self._validation_warnings} warnings"
            )

        return cue_state

    def _generate_ball_id(self, ball: Ball, is_target: bool = False) -> str:
        """Generate consistent ball ID from ball detection.

        Args:
            ball: Vision Ball detection
            is_target: Whether this is the target ball for trajectory calculation

        Returns:
            Ball ID string
        """
        # If this is marked as target, use special ID for trajectory engine
        if is_target:
            return "target_ball"

        # Use track_id if available (preferred for consistency)
        if ball.track_id is not None:
            return str(ball.track_id)

        # Fallback: Generate ID from position and type
        # This is less ideal as ball positions change, but provides some stability
        position_hash = hash(
            (
                round(ball.position[0] / 10)
                * 10,  # Round to nearest 10 pixels for stability
                round(ball.position[1] / 10) * 10,
                ball.ball_type.value if ball.ball_type else "unknown",
            )
        )
        return f"ball_{position_hash % 10000}"

    def _validate_position(self, x: float, y: float, obj_type: str) -> bool:
        """Validate position is within reasonable bounds.

        Args:
            x: X position
            y: Y position
            obj_type: Type of object for logging (e.g., "ball", "cue")

        Returns:
            True if position is valid, False otherwise
        """
        is_valid = True

        if x < 0 or x > self.max_position_x:
            logger.warning(
                f"{obj_type.capitalize()} position X={x:.2f} out of bounds [0, {self.max_position_x}]"
            )
            is_valid = False

        if y < 0 or y > self.max_position_y:
            logger.warning(
                f"{obj_type.capitalize()} position Y={y:.2f} out of bounds [0, {self.max_position_y}]"
            )
            is_valid = False

        return is_valid

    def _validate_velocity(self, vx: float, vy: float, obj_type: str) -> bool:
        """Validate velocity is within reasonable bounds.

        Args:
            vx: X velocity
            vy: Y velocity
            obj_type: Type of object for logging (e.g., "ball")

        Returns:
            True if velocity is valid, False otherwise
        """
        velocity_mag = (vx**2 + vy**2) ** 0.5

        if velocity_mag > self.max_ball_velocity:
            logger.warning(
                f"{obj_type.capitalize()} velocity {velocity_mag:.2f} m/s exceeds max {self.max_ball_velocity} m/s"
            )
            return False

        return True

    def get_conversion_stats(self) -> dict:
        """Get conversion statistics.

        Returns:
            Dictionary with conversion statistics
        """
        return {
            "ball_conversions": self._ball_conversion_count,
            "cue_conversions": self._cue_conversion_count,
            "validation_warnings": self._validation_warnings,
            "ball_warning_rate": (
                self._validation_warnings / self._ball_conversion_count
                if self._ball_conversion_count > 0
                else 0.0
            ),
        }
