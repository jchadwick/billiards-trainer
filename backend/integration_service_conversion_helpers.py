"""State conversion helper methods for IntegrationService.

This module contains improved conversion methods that:
1. Create proper BallState/CueState objects that match trajectory calculation needs
2. Add validation during conversion
3. Provide helper methods for common conversions
4. Ensure consistency between state conversion and trajectory calculation
5. Track and convert coordinate spaces (camera pixels -> 4K canonical)
6. Use resolution-based scaling with Vector2D.from_resolution()

COORDINATE SYSTEM UPDATE (2025-10-21):
--------------------------------------
This module has been updated to use the new 4K pixel-based coordinate system.

OLD APPROACH (DEPRECATED):
- Vision detections in pixels → meters conversion using pixels_per_meter calibration
- Required table corner detection and homography transforms
- Complex CoordinateConverter with perspective transforms

NEW APPROACH (CURRENT):
- Vision detections in pixels → 4K canonical pixels using resolution scale
- Automatic scale calculation: scale = 4K_resolution / source_resolution
- Simple Vector2D.from_resolution() → to_4k_canonical() conversion
- No calibration required - pure pixel-based coordinates throughout

All positions, velocities, and sizes are now in 4K canonical pixel space (3840×2160).
The scale metadata is preserved in Vector2D for future coordinate transformations.
"""

import logging
import time
from typing import Optional, Sequence, Tuple

# Resolution type for camera/source resolution metadata
Resolution = tuple[int, int]  # (width, height)

from core.coordinates import Vector2D
from core.models import BallState, CueState
from core.validation.physics import PhysicsValidator

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

        # Get table dimensions from config or use default 9ft table
        # Standard 9ft table dimensions in 4K pixels: 3200px × 1600px (using ~1260 px/m scale)
        # Note: These dimensions are used for validation bounds only
        self.table_width_4k = 3200.0  # 9 feet in 4K pixels (~2.54m * 1260px/m)
        self.table_height_4k = 1600.0  # 4.5 feet in 4K pixels (~1.27m * 1260px/m)

        # Validation thresholds from config or defaults
        # All thresholds are now in 4K pixel units
        if config:
            # Velocity threshold: ~10 m/s * 1260 px/m = 12600 px/s in 4K
            self.max_ball_velocity_4k = config.get(
                "integration.max_ball_velocity_4k_px_per_s", 12600.0
            )
            # Position bounds: 4K frame is 3840×2160, table playing area is ~(320,280) to (3520,1880)
            self.max_position_x_4k = config.get("integration.max_position_x_4k", 3840.0)
            self.max_position_y_4k = config.get("integration.max_position_y_4k", 2160.0)
            self.min_ball_confidence = config.get(
                "integration.min_ball_confidence", 0.1
            )
            self.min_cue_confidence = config.get("integration.min_cue_confidence", 0.05)
            camera_width = config.get("camera.resolution.width", 1920)
            camera_height = config.get("camera.resolution.height", 1080)
        else:
            # Default values in 4K pixels
            self.max_ball_velocity_4k = 12600.0  # ~10 m/s in 4K pixels/second
            self.max_position_x_4k = 3840.0  # 4K frame width
            self.max_position_y_4k = 2160.0  # 4K frame height
            self.min_ball_confidence = 0.1
            self.min_cue_confidence = 0.05  # Lowered to minimize intermittent cue drops
            camera_width = 1920
            camera_height = 1080

        # Store camera resolution for coordinate conversions
        # Camera resolution is the source resolution for vision detections
        self.camera_resolution: Resolution = (camera_width, camera_height)

        # Conversion counters for periodic logging
        self._ball_conversion_count = 0
        self._cue_conversion_count = 0
        self._validation_warnings = 0
        self._coordinate_conversions = 0

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

        # Generate ball ID - use track_id if available, otherwise create from position/type
        ball_id = self._generate_ball_id(ball, is_target)

        # Track coordinate space information for logging/debugging
        source_space = (
            ball.coordinate_space if hasattr(ball, "coordinate_space") else "pixel"
        )

        # Convert coordinates from camera pixels to 4K canonical using scale
        # Vision Ball positions are in pixel coordinates with source_resolution metadata
        # Core BallState expects positions in 4K canonical format
        if source_space == "pixel":
            # Get source resolution from ball metadata
            # Ball detections include source_resolution=(width, height) from camera
            source_resolution = (
                ball.source_resolution
                if hasattr(ball, "source_resolution") and ball.source_resolution
                else self.camera_resolution
            )

            # Use Vector2D.from_resolution() to create position with automatic scale
            # This calculates scale = 4K / source_resolution and stores it in the vector
            position_with_scale = Vector2D.from_resolution(
                ball.position[0], ball.position[1], source_resolution
            )

            # Convert to 4K canonical (scale=[1.0, 1.0])
            position_4k = position_with_scale.to_4k_canonical()

            # Convert velocity using the same scale
            # Velocity is a displacement vector, so apply the same scaling
            velocity_with_scale = Vector2D.from_resolution(
                ball.velocity[0], ball.velocity[1], source_resolution
            )
            velocity_4k = velocity_with_scale.to_4k_canonical()

            # Convert radius using the scale factor
            scale_x = 3840 / source_resolution[0]
            radius_4k = (
                ball.radius * scale_x if ball.radius > 0 else 36.0
            )  # Default ball radius in 4K pixels

            self._coordinate_conversions += 1

            # Log coordinate conversion periodically
            if self._coordinate_conversions % 100 == 0:
                logger.debug(
                    f"Coordinate conversion #{self._coordinate_conversions}: "
                    f"source_res={source_resolution}, "
                    f"scale=[{scale_x:.2f},{3840/source_resolution[0]:.2f}], "
                    f"pixel({ball.position[0]:.1f},{ball.position[1]:.1f}) -> "
                    f"4K({position_4k.x:.1f},{position_4k.y:.1f})"
                )
        else:
            # Already in 4K canonical or other coordinate space
            position_4k = Vector2D(ball.position[0], ball.position[1], scale=(1.0, 1.0))
            velocity_4k = Vector2D(ball.velocity[0], ball.velocity[1], scale=(1.0, 1.0))
            radius_4k = ball.radius if ball.radius > 0 else 36.0
            logger.debug(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"Ball already in target coordinate space, no conversion needed"
            )

        # Validate position is reasonable (in 4K pixels)
        # Table bounds in 4K: roughly (320, 280) to (3520, 1880)
        # Allow some margin for balls near rails
        if (
            position_4k.x < 0
            or position_4k.x > 3840
            or position_4k.y < 0
            or position_4k.y > 2160
        ):
            logger.warning(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"Position ({position_4k.x:.1f}, {position_4k.y:.1f}) outside 4K frame, clamping"
            )
            position_4k = Vector2D(
                max(0, min(3840, position_4k.x)),
                max(0, min(2160, position_4k.y)),
                scale=(1.0, 1.0),
            )
            self._validation_warnings += 1

        # Validate velocity is reasonable (in 4K pixels/second)
        # Max reasonable velocity: ~10 m/s * 1260 px/m ≈ 12600 px/s
        velocity_mag = (velocity_4k.x**2 + velocity_4k.y**2) ** 0.5
        max_velocity_px = 12600.0  # Roughly 10 m/s in 4K pixels/second
        if velocity_mag > max_velocity_px:
            logger.warning(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"Velocity {velocity_mag:.1f} px/s exceeds max {max_velocity_px:.1f} px/s, clamping"
            )
            if velocity_mag > 0:
                scale = max_velocity_px / velocity_mag
                velocity_4k = Vector2D(
                    velocity_4k.x * scale, velocity_4k.y * scale, scale=(1.0, 1.0)
                )
            self._validation_warnings += 1

        # Create BallState using from_4k() factory method
        # This ensures the position has proper 4K canonical scale metadata
        ball_state = BallState.from_4k(
            id=ball_id,
            x=position_4k.x,
            y=position_4k.y,
            vx=velocity_4k.x,
            vy=velocity_4k.y,
            radius=radius_4k,
            mass=0.17,  # Standard pool ball mass in kg
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
                f"source_space={source_space}, "
                f"pos_4k=({ball_state.position.x:.1f},{ball_state.position.y:.1f})px, "
                f"vel_4k=({ball_state.velocity.x:.1f},{ball_state.velocity.y:.1f})px/s, "
                f"radius_4k={ball_state.radius:.1f}px, "
                f"confidence={ball_state.confidence:.2f}, "
                f"is_cue={ball_state.is_cue_ball}"
            )
            logger.info(
                f"Ball conversion stats: {self._ball_conversion_count} conversions, "
                f"{self._coordinate_conversions} coordinate conversions, "
                f"{self._validation_warnings} warnings"
            )

        return ball_state

    def vision_cue_to_cue_state(
        self,
        detected_cue: CueStick,
        timestamp: Optional[float] = None,
        validate: bool = True,  # noqa: ARG002 - validate param reserved for future use
    ) -> Optional[CueState]:
        """Convert vision CueStick detection to core CueState with validation.

        Args:
            detected_cue: Vision module CueStick detection result
            timestamp: Optional timestamp for last_update (defaults to current time)
            validate: Whether to validate the cue state (reserved for future use)

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

        # Track coordinate space information for logging/debugging
        source_space = (
            detected_cue.coordinate_space
            if hasattr(detected_cue, "coordinate_space")
            else "pixel"
        )

        # Convert coordinates from camera pixels to 4K canonical using scale
        # Vision CueStick positions are in pixel coordinates with source_resolution metadata
        # Core CueState expects positions in 4K canonical format
        if source_space == "pixel":
            # Get source resolution from cue metadata
            source_resolution = (
                detected_cue.source_resolution
                if hasattr(detected_cue, "source_resolution")
                and detected_cue.source_resolution
                else self.camera_resolution
            )

            # Use Vector2D.from_resolution() to create tip position with automatic scale
            tip_position_with_scale = Vector2D.from_resolution(
                detected_cue.tip_position[0],
                detected_cue.tip_position[1],
                source_resolution,
            )

            # Convert to 4K canonical
            tip_position_4k = tip_position_with_scale.to_4k_canonical()

            # Convert length using the same scale
            scale_x = 3840 / source_resolution[0]
            length_4k = (
                detected_cue.length * scale_x
                if hasattr(detected_cue, "length") and detected_cue.length > 0
                else 1851.0
            )  # Default cue length in 4K pixels (~1.47m * 1260px/m)

            self._coordinate_conversions += 1

            # Log coordinate conversion periodically
            if self._coordinate_conversions % 100 == 0:
                logger.debug(
                    f"Cue coordinate conversion #{self._coordinate_conversions}: "
                    f"source_res={source_resolution}, "
                    f"scale=[{scale_x:.2f}], "
                    f"pixel_tip({detected_cue.tip_position[0]:.1f},{detected_cue.tip_position[1]:.1f}) -> "
                    f"4K_tip({tip_position_4k.x:.1f},{tip_position_4k.y:.1f})"
                )
        else:
            # Already in 4K canonical or other coordinate space
            tip_position_4k = Vector2D(
                detected_cue.tip_position[0],
                detected_cue.tip_position[1],
                scale=(1.0, 1.0),
            )
            length_4k = (
                detected_cue.length if hasattr(detected_cue, "length") else 1851.0
            )
            logger.debug(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Cue already in target coordinate space, no conversion needed"
            )

        # Validate tip position is reasonable (in 4K pixels)
        if (
            tip_position_4k.x < 0
            or tip_position_4k.x > 3840
            or tip_position_4k.y < 0
            or tip_position_4k.y > 2160
        ):
            logger.warning(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Tip position ({tip_position_4k.x:.1f}, {tip_position_4k.y:.1f}) outside 4K frame, clamping"
            )
            tip_position_4k = Vector2D(
                max(0, min(3840, tip_position_4k.x)),
                max(0, min(2160, tip_position_4k.y)),
                scale=(1.0, 1.0),
            )
            self._validation_warnings += 1

        # Create cue state using from_4k() factory method
        # This ensures the position has proper 4K canonical scale metadata
        cue_state = CueState.from_4k(
            angle=detected_cue.angle,
            estimated_force=estimated_force,
            tip_x=tip_position_4k.x,
            tip_y=tip_position_4k.y,
            elevation=0.0,  # Vision doesn't currently detect elevation
            length=length_4k,
            is_visible=True,
            confidence=detected_cue.confidence,
            last_update=timestamp if timestamp is not None else time.time(),
        )

        # Log conversion details periodically
        if self._cue_conversion_count % 100 == 0:
            logger.debug(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"source_space={source_space}, "
                f"angle={cue_state.angle:.2f}deg, "
                f"force={cue_state.estimated_force:.2f}N, "
                f"tip_4k=({cue_state.tip_position.x:.1f},{cue_state.tip_position.y:.1f})px, "
                f"length_4k={cue_state.length:.1f}px, "
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
        """Validate position is within reasonable bounds (4K pixel coordinates).

        Args:
            x: X position in 4K pixels
            y: Y position in 4K pixels
            obj_type: Type of object for logging (e.g., "ball", "cue")

        Returns:
            True if position is valid, False otherwise
        """
        is_valid = True

        if x < 0 or x > self.max_position_x_4k:
            logger.warning(
                f"{obj_type.capitalize()} position X={x:.2f}px out of bounds [0, {self.max_position_x_4k}px] (4K)"
            )
            is_valid = False

        if y < 0 or y > self.max_position_y_4k:
            logger.warning(
                f"{obj_type.capitalize()} position Y={y:.2f}px out of bounds [0, {self.max_position_y_4k}px] (4K)"
            )
            is_valid = False

        return is_valid

    def _validate_velocity(self, vx: float, vy: float, obj_type: str) -> bool:
        """Validate velocity is within reasonable bounds (4K pixel units).

        Args:
            vx: X velocity in 4K pixels/second
            vy: Y velocity in 4K pixels/second
            obj_type: Type of object for logging (e.g., "ball")

        Returns:
            True if velocity is valid, False otherwise
        """
        velocity_mag = (vx**2 + vy**2) ** 0.5

        if velocity_mag > self.max_ball_velocity_4k:
            logger.warning(
                f"{obj_type.capitalize()} velocity {velocity_mag:.2f} px/s exceeds max {self.max_ball_velocity_4k:.2f} px/s (4K)"
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
            "coordinate_conversions": self._coordinate_conversions,
            "validation_warnings": self._validation_warnings,
            "ball_warning_rate": (
                self._validation_warnings / self._ball_conversion_count
                if self._ball_conversion_count > 0
                else 0.0
            ),
            "coordinate_system": "4K_canonical",
            "resolution_based_conversion": True,
        }
