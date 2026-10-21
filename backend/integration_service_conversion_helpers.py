"""State conversion helper methods for IntegrationService.

This module contains improved conversion methods that:
1. Create proper BallState/CueState objects that match trajectory calculation needs
2. Add validation during conversion
3. Provide helper methods for common conversions
4. Ensure consistency between state conversion and trajectory calculation
5. Track and convert coordinate spaces (camera pixels -> 4K canonical)
6. Require explicit source-space and resolution metadata for pixel geometry

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
- One shared source-resolution mapping into canonical 4K pixels
- No calibration required - pure pixel-based coordinates throughout

All positions, velocities, and sizes are now in 4K canonical pixel space (3840×2160).
The scale metadata is preserved in Vector2D for future coordinate transformations.
"""

import logging
import math
import time
from typing import Any, Optional, Sequence

from backend.common.models import Ball
from backend.core.constants_4k import BALL_MASS_KG, CANONICAL_HEIGHT, CANONICAL_WIDTH
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState
from backend.core.validation.physics import PhysicsValidator
from backend.vision.models import CueStick

logger = logging.getLogger(__name__)


class CoordinateConversionError(ValueError):
    """Raised when detection geometry cannot be mapped safely into canonical pixels."""


def _validated_resolution(
    resolution: Optional[tuple[int, int]], context: str
) -> tuple[int, int]:
    """Return a usable (width, height), rejecting absent or invalid frame metadata."""
    try:
        valid = (
            resolution is not None
            and len(resolution) == 2
            and all(
                not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value).is_integer()
                and value > 0
                for value in resolution
            )
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid or resolution is None:
        raise CoordinateConversionError(
            f"{context} uses pixel coordinates but has no valid source resolution"
        )
    return int(resolution[0]), int(resolution[1])


def _coordinate_scale(
    coordinate_space: Any,
    source_resolution: Optional[tuple[int, int]],
    frame_resolution: Optional[tuple[int, int]],
    context: str,
) -> tuple[float, float]:
    """Resolve the declared coordinate space into scale factors to canonical 4K."""
    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        raise CoordinateConversionError(
            f"{context} is missing or has invalid coordinate space metadata"
        )
    space = coordinate_space.strip().lower()
    if space == "4k_canonical":
        return (1.0, 1.0)
    if space != "pixel":
        raise CoordinateConversionError(
            f"{context} uses unsupported coordinate space {coordinate_space!r}; "
            "only pixel and 4k_canonical geometry can enter Core state"
        )

    resolution = _validated_resolution(
        source_resolution if source_resolution is not None else frame_resolution,
        context,
    )
    return (CANONICAL_WIDTH / resolution[0], CANONICAL_HEIGHT / resolution[1])


def _scaled_xy(
    point: Sequence[float], scale: tuple[float, float], context: str
) -> dict[str, Any]:
    """Map a point and its pixel extents into explicit canonical coordinates."""
    if (
        not isinstance(point, Sequence)
        or isinstance(point, (str, bytes))
        or len(point) != 2
    ):
        raise CoordinateConversionError(f"{context} must contain exactly two values")
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise CoordinateConversionError(
            f"{context} must contain numeric coordinates"
        ) from error
    if not math.isfinite(x) or not math.isfinite(y):
        raise CoordinateConversionError(f"{context} contains non-finite coordinates")
    scaled_x, scaled_y = x * scale[0], y * scale[1]
    if not math.isfinite(scaled_x) or not math.isfinite(scaled_y):
        raise CoordinateConversionError(f"{context} overflows canonical coordinates")
    return {"x": scaled_x, "y": scaled_y, "scale": [1.0, 1.0]}


def _scaled_extent(
    extent: float, factor: float, context: str
) -> tuple[float, list[float]]:
    """Map a one-dimensional pixel extent to canonical pixels with metadata."""
    try:
        value = float(extent)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoordinateConversionError(
            f"{context} must be a finite pixel extent"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise CoordinateConversionError(
            f"{context} must be a finite, non-negative pixel extent"
        )
    canonical = value * factor
    if not math.isfinite(canonical):
        raise CoordinateConversionError(f"{context} overflows canonical pixels")
    return canonical, [1.0, 1.0]


def _coordinate_space_metadata(scale: tuple[float, float]) -> dict[str, Any]:
    """Describe the source-to-canonical mapping for a converted object."""
    return {
        "coordinate_space": "4k_canonical",
        "source_to_canonical_scale": [scale[0], scale[1]],
        "scale": [1.0, 1.0],
    }


def _object_coordinate_scale(
    value: Any, frame_resolution: Optional[tuple[int, int]], context: str
) -> tuple[float, float]:
    """Resolve object-local metadata, using the detection frame only as pixel context."""
    return _coordinate_scale(
        getattr(value, "coordinate_space", None),
        getattr(value, "source_resolution", None),
        frame_resolution,
        context,
    )


def _canonical_geometry_point(
    value: Sequence[float], scale: tuple[float, float], context: str
) -> dict[str, Any]:
    """Return the boundary point as canonical 4K x/y plus mandatory scale metadata."""
    return _scaled_xy(value, scale, context)


def serialize_vision_ball(
    ball: Ball, frame_resolution: Optional[tuple[int, int]]
) -> dict[str, Any]:
    """Serialize one Vision ball to canonical Core payload geometry."""
    context = "ball detection"
    try:
        observation_timestamp = float(ball.last_seen)
    except (TypeError, ValueError, OverflowError) as error:
        raise CoordinateConversionError(
            "ball observation timestamp must be a finite wall-clock value"
        ) from error
    if not math.isfinite(observation_timestamp):
        raise CoordinateConversionError(
            "ball observation timestamp must be a finite wall-clock value"
        )
    scale = _object_coordinate_scale(ball, frame_resolution, context)
    position = _canonical_geometry_point(ball.position, scale, f"{context} position")
    ball_id = ball.track_id
    if ball_id is None:
        ball_id = (
            hash(
                (
                    round(position["x"]),
                    round(position["y"]),
                    ball.ball_type.value if ball.ball_type else "unknown",
                )
            )
            % 10000
        )
        context = f"ball {ball_id}"
    radius, radius_scale = _scaled_extent(ball.radius, scale[0], f"{context} radius")
    radius_extent = _canonical_geometry_point(
        (ball.radius, ball.radius), scale, f"{context} radius extent"
    )
    data: dict[str, Any] = {
        "id": ball_id,
        "position": position,
        "velocity": _canonical_geometry_point(
            ball.velocity, scale, f"{context} velocity"
        ),
        "acceleration": _canonical_geometry_point(
            ball.acceleration, scale, f"{context} acceleration"
        ),
        "radius": radius,
        "radius_scale": radius_scale,
        "radius_extent": radius_extent,
        "is_moving": ball.is_moving,
        "is_observed": ball.is_observed,
        "number": ball.number,
        "type": ball.ball_type.value if ball.ball_type else "unknown",
        "is_cue_ball": bool(ball.ball_type and ball.ball_type.value == "cue"),
        "confidence": ball.confidence,
        # Core's last_update field is wall-clock seconds. Tracker intervals use
        # a separate monotonic clock and must never be serialized here.
        "timestamp": observation_timestamp,
        **_coordinate_space_metadata(scale),
    }
    if ball.predicted_position is not None:
        data["predicted_position"] = _canonical_geometry_point(
            ball.predicted_position, scale, f"{context} predicted position"
        )
    data["position_history"] = [
        _canonical_geometry_point(point, scale, f"{context} history point")
        for point in ball.position_history
    ]
    return data


def serialize_vision_cue(
    cue: CueStick, frame_resolution: Optional[tuple[int, int]]
) -> dict[str, Any]:
    """Serialize cue geometry and motion to canonical 4K pixels."""
    scale = _object_coordinate_scale(cue, frame_resolution, "cue geometry")
    if not math.isfinite(cue.angle):
        raise CoordinateConversionError("cue angle must be finite")
    angle = math.radians(cue.angle)
    dx = math.cos(angle) * scale[0]
    dy = math.sin(angle) * scale[1]
    canonical_angle = math.degrees(math.atan2(dy, dx)) % 360
    length_factor = math.hypot(dx, dy)
    length, length_scale = _scaled_extent(cue.length * length_factor, 1.0, "cue length")
    perpendicular_factor = math.hypot(
        math.sin(angle) * scale[0], math.cos(angle) * scale[1]
    )
    width, width_scale = _scaled_extent(
        cue.width * perpendicular_factor, 1.0, "cue width"
    )

    data: dict[str, Any] = {
        "tip_position": _canonical_geometry_point(
            cue.tip_position, scale, "cue tip position"
        ),
        "angle": canonical_angle,
        "length": length,
        "length_scale": length_scale,
        "width": width,
        "width_scale": width_scale,
        "tip_velocity": _canonical_geometry_point(
            cue.tip_velocity, scale, "cue tip velocity"
        ),
        "angular_velocity": cue.angular_velocity,
        "shaft_points": [
            _canonical_geometry_point(point, scale, "cue shaft point")
            for point in cue.shaft_points
        ],
        "aiming_line": [
            _canonical_geometry_point(point, scale, "cue aiming line point")
            for point in (cue.aiming_line or [])
        ],
        "state": cue.state.value if cue.state else "unknown",
        "confidence": cue.confidence,
        "target_ball_id": cue.target_ball_id,
        **_coordinate_space_metadata(scale),
    }
    # CueStick historically defaults an unavailable butt endpoint to (0, 0).
    # Keep that legacy default optional at this boundary so older cue detectors
    # do not publish a false measured endpoint. The YOLO endpoint converter
    # supplies a concrete butt point whenever it detects the shaft line.
    cue_butt_position = getattr(cue, "butt_position", None)
    if cue_butt_position is not None and tuple(cue_butt_position) != (0.0, 0.0):
        data["butt_position"] = _canonical_geometry_point(
            cue_butt_position, scale, "cue butt position"
        )
    if cue.predicted_contact_point is not None:
        data["predicted_contact_point"] = _canonical_geometry_point(
            cue.predicted_contact_point, scale, "cue predicted contact point"
        )
    return data


def serialize_vision_table(
    table: Any, frame_resolution: Optional[tuple[int, int]]
) -> dict[str, Any]:
    """Serialize table boundaries, pocket geometry, and pixel extents canonically."""
    scale = _object_coordinate_scale(table, frame_resolution, "table geometry")
    width, width_scale = _scaled_extent(table.width, scale[0], "table width")
    height, height_scale = _scaled_extent(table.height, scale[1], "table height")
    pockets = []
    for index, pocket in enumerate(table.pockets or []):
        pocket_context = f"pocket {index}"
        radius, radius_scale = _scaled_extent(
            pocket.radius, scale[0], f"{pocket_context} radius"
        )
        radius_extent = _canonical_geometry_point(
            (pocket.radius, pocket.radius), scale, f"{pocket_context} radius extent"
        )
        pockets.append(
            {
                "position": _canonical_geometry_point(
                    pocket.position, scale, f"{pocket_context} position"
                ),
                "corners": [
                    _canonical_geometry_point(point, scale, f"{pocket_context} corner")
                    for point in pocket.corners
                ],
                "radius": radius,
                "radius_scale": radius_scale,
                "radius_extent": radius_extent,
                "type": (pocket.pocket_type.value if pocket.pocket_type else "unknown"),
                **_coordinate_space_metadata(scale),
            }
        )

    data: dict[str, Any] = {
        "corners": [
            _canonical_geometry_point(point, scale, "table corner")
            for point in table.corners
        ],
        "rails": [
            [
                _canonical_geometry_point(point, scale, "table rail point")
                for point in rail
            ]
            for rail in table.rails
        ],
        "playing_area": [
            _canonical_geometry_point(point, scale, "table playing-area point")
            for point in table.playing_area
        ],
        "playing_area_corners": [
            _canonical_geometry_point(point, scale, "table playing-area corner")
            for point in table.playing_area
        ],
        "width": width,
        "width_scale": width_scale,
        "height": height,
        "height_scale": height_scale,
        "pockets": pockets,
        "surface_color": table.surface_color,
        "real_world_dimensions": table.real_world_dimensions,
        "real_world_dimensions_unit": "meters",
        "pixels_per_meter": (
            table.pixels_per_meter * math.sqrt(scale[0] * scale[1])
            if table.pixels_per_meter is not None
            else None
        ),
        **_coordinate_space_metadata(scale),
    }
    if pockets:
        data["pocket_radius"] = pockets[0]["radius"]
        data["pocket_radius_scale"] = [1.0, 1.0]
    source_resolution = getattr(table, "source_resolution", None) or frame_resolution
    if source_resolution is not None:
        data["source_resolution"] = list(source_resolution)
    return data


class StateConversionHelpers:
    """Helper class for converting Vision detections to Core states with validation."""

    def __init__(
        self, config: Any = None, physics_validator: Optional[PhysicsValidator] = None
    ):
        """Initialize state conversion helpers.

        Args:
            config: Configuration object
            physics_validator: Optional physics validator for validation during conversion
        """
        self.config = config
        self.physics_validator = physics_validator or PhysicsValidator()

        # NOTE: Table dimensions come from calibration/detection, not hardcoded values
        # The system requires table detection/calibration to determine actual table bounds

        # Validation thresholds from config or defaults
        # All thresholds are now in 4K pixel units
        if config:
            # Velocity threshold: ~10 m/s * 1260 px/m = 12600 px/s in 4K
            self.max_ball_velocity_4k = config.get(
                "integration.max_ball_velocity_4k_px_per_s", 12600.0
            )
            # Position bounds: 4K frame is 3840×2160
            # These are frame bounds for validation, not table bounds
            self.max_position_x_4k = config.get(
                "integration.max_position_x_4k", CANONICAL_WIDTH
            )
            self.max_position_y_4k = config.get(
                "integration.max_position_y_4k", CANONICAL_HEIGHT
            )
            self.min_ball_confidence = config.get(
                "integration.min_ball_confidence", 0.1
            )
            self.min_cue_confidence = config.get("integration.min_cue_confidence", 0.05)
        else:
            # Default values in 4K pixels
            self.max_ball_velocity_4k = 12600.0  # ~10 m/s in 4K pixels/second
            self.max_position_x_4k = CANONICAL_WIDTH  # 4K frame width
            self.max_position_y_4k = CANONICAL_HEIGHT  # 4K frame height
            self.min_ball_confidence = 0.1
            self.min_cue_confidence = 0.05  # Lowered to minimize intermittent cue drops
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

        source_space = getattr(ball, "coordinate_space", None)
        source_resolution = getattr(ball, "source_resolution", None)
        scale = _coordinate_scale(
            source_space, source_resolution, None, f"ball {ball_id}"
        )
        position = _scaled_xy(ball.position, scale, f"ball {ball_id} position")
        velocity = _scaled_xy(ball.velocity, scale, f"ball {ball_id} velocity")
        position_4k = Vector2D.from_4k(position["x"], position["y"])
        velocity_4k = Vector2D.from_4k(velocity["x"], velocity["y"])
        radius_4k, _ = _scaled_extent(ball.radius, scale[0], f"ball {ball_id} radius")

        if source_space == "pixel":
            self._coordinate_conversions += 1
            if self._coordinate_conversions % 100 == 0:
                logger.debug(
                    "Coordinate conversion #%s: source_res=%s, scale=%s, "
                    "pixel(%s,%s) -> 4K(%s,%s)",
                    self._coordinate_conversions,
                    source_resolution,
                    scale,
                    ball.position[0],
                    ball.position[1],
                    position_4k.x,
                    position_4k.y,
                )

        # Validate position is reasonable (in 4K pixels)
        # 4K frame bounds: (0, 0) to (3840, 2160)
        # Table bounds vary based on calibration
        if (
            position_4k.x < 0
            or position_4k.x > self.max_position_x_4k
            or position_4k.y < 0
            or position_4k.y > self.max_position_y_4k
        ):
            logger.warning(
                f"Ball conversion #{self._ball_conversion_count}: "
                f"Position ({position_4k.x:.1f}, {position_4k.y:.1f}) outside 4K frame, clamping"
            )
            # Clamp to frame bounds
            position_4k = Vector2D(
                max(0, min(self.max_position_x_4k, position_4k.x)),
                max(0, min(self.max_position_y_4k, position_4k.y)),
                scale=(1.0, 1.0),
            )
            self._validation_warnings += 1

        # Validate velocity is reasonable (in 4K pixels/second)
        # Max reasonable velocity: ~10 m/s * 1260 px/m ≈ 12600 px/s
        velocity_mag = (velocity_4k.x**2 + velocity_4k.y**2) ** 0.5
        max_velocity_px = self.max_ball_velocity_4k
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
            velocity=velocity_4k,
            radius=radius_4k,
            mass=BALL_MASS_KG,
            is_cue_ball=(ball.ball_type.value == "cue" if ball.ball_type else False),
            is_pocketed=False,  # Vision detections are only for balls on table
            number=ball.number,
            confidence=ball.confidence,
            is_observed=ball.is_observed,
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

        # Keep the source detection immutable; normalize the local value only.
        source_angle = detected_cue.angle % 360
        if source_angle != detected_cue.angle:
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

        source_space = getattr(detected_cue, "coordinate_space", None)
        source_resolution = getattr(detected_cue, "source_resolution", None)
        scale = _coordinate_scale(source_space, source_resolution, None, "cue geometry")
        tip = _scaled_xy(detected_cue.tip_position, scale, "cue tip position")
        tip_position_4k = Vector2D.from_4k(tip["x"], tip["y"])
        butt_position_4k = None
        cue_butt_position = getattr(detected_cue, "butt_position", None)
        if cue_butt_position is not None and tuple(cue_butt_position) != (0.0, 0.0):
            butt = _scaled_xy(cue_butt_position, scale, "cue butt position")
            butt_position_4k = Vector2D.from_4k(butt["x"], butt["y"])
        angle_radians = math.radians(source_angle)
        direction_x = math.cos(angle_radians) * scale[0]
        direction_y = math.sin(angle_radians) * scale[1]
        transformed_angle = math.degrees(math.atan2(direction_y, direction_x)) % 360
        length_4k = float(detected_cue.length) * math.hypot(direction_x, direction_y)
        if not math.isfinite(length_4k) or length_4k < 0:
            raise CoordinateConversionError(
                "cue length must be finite and non-negative"
            )
        if source_space == "pixel":
            self._coordinate_conversions += 1

        # Validate tip position is reasonable (in 4K pixels)
        # 4K frame bounds: (0, 0) to (3840, 2160)
        # Table bounds vary based on calibration
        if (
            tip_position_4k.x < 0
            or tip_position_4k.x > self.max_position_x_4k
            or tip_position_4k.y < 0
            or tip_position_4k.y > self.max_position_y_4k
        ):
            logger.warning(
                f"Cue conversion #{self._cue_conversion_count}: "
                f"Tip position ({tip_position_4k.x:.1f}, {tip_position_4k.y:.1f}) outside 4K frame, clamping"
            )
            # Clamp to frame bounds
            tip_position_4k = Vector2D(
                max(0, min(self.max_position_x_4k, tip_position_4k.x)),
                max(0, min(self.max_position_y_4k, tip_position_4k.y)),
                scale=(1.0, 1.0),
            )
            self._validation_warnings += 1

        # Create cue state using from_4k() factory method
        # This ensures the position has proper 4K canonical scale metadata
        cue_state = CueState.from_4k(
            angle=transformed_angle,
            estimated_force=estimated_force,
            tip_x=tip_position_4k.x,
            tip_y=tip_position_4k.y,
            butt_position=butt_position_4k,
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

        # Fallback IDs must be stable across source resolutions. Hash canonical
        # coordinates so the same observed ball does not change identity when
        # the camera capture resolution changes.
        coordinate_scale = _coordinate_scale(
            getattr(ball, "coordinate_space", None),
            getattr(ball, "source_resolution", None),
            None,
            "ball ID position",
        )
        position = _scaled_xy(ball.position, coordinate_scale, "ball ID position")
        position_hash = hash(
            (
                round(position["x"] / 10) * 10,
                round(position["y"] / 10) * 10,
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

    def get_conversion_stats(self) -> dict[str, Any]:
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
