"""Game State Management.

Main game state management for tracking balls, cue, table state.
Implements FR-CORE-001 through FR-CORE-015 requirements.
"""

import json
import logging
import math
import pickle
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

from .constants_4k import BALL_MASS_KG, BALL_RADIUS_4K, POCKET_RADIUS_4K
from .coordinates import Vector2D
from .events.manager import EventManager
from .models import (
    BallState,
    CoordinateMetadata,
    CueState,
    GameEvent,
    GameState,
    GameType,
    TableState,
)
from .validation import StateValidator, ValidationResult

logger = logging.getLogger(__name__)


class StateValidationError(Exception):
    """Exception raised when state validation fails."""

    pass


class GameStateManager:
    """Game state management implementing FR-CORE-001 through FR-CORE-015.

    Features:
    - Real-time state tracking and validation
    - Frame history management
    - Event detection and notification
    - State persistence and recovery
    - Thread-safe operations
    - Integration with vision module
    """

    def __init__(
        self,
        table: Optional[TableState] = None,
        max_history_frames: int = 1000,
        persistence_path: Optional[Path] = None,
    ):
        """Initialize game state manager.

        Args:
            table: TableState from calibration (required for operation)
            max_history_frames: Maximum number of frames to keep in history
            persistence_path: Path for state persistence files
        """
        self._current_state: Optional[GameState] = None
        self._state_history: deque[GameState] = deque(maxlen=max_history_frames)
        self._event_manager = EventManager()
        self._lock = threading.RLock()
        self._frame_counter = 0
        self._start_time = time.time()
        self._persistence_path = persistence_path or Path("./game_state_data")
        self._persistence_path.mkdir(parents=True, exist_ok=True)

        # State validation configuration
        self._validation_enabled = True
        self._auto_correct_enabled = True
        self._state_validator = StateValidator(
            enable_auto_correction=self._auto_correct_enabled
        )

        # Table must be provided or system cannot operate
        if table is None:
            logger.warning(
                "GameStateManager initialized without table. "
                "Table detection/calibration must be completed before processing frames."
            )
            self._default_table = None
        else:
            self._default_table = table

        # Event detection state
        self._last_ball_positions: dict[str, Vector2D] = {}
        self._motion_threshold = 5.0  # canonical pixels/frame

        # Change detection configuration
        self._position_change_threshold = (
            5.0  # pixels - minimum position change to consider significant
        )
        self._angle_change_threshold = (
            2.0  # degrees - minimum angle change to consider significant
        )
        self._last_cue_state: Optional[CueState] = None

        logger.info("GameStateManager initialized")

    def update_state(self, detection_data: dict[str, Any]) -> GameState:
        """Update game state from vision detection data (FR-CORE-001).

        Args:
            detection_data: Vision module detection results

        Returns:
            Updated game state

        Raises:
            StateValidationError: If state validation fails
        """
        with self._lock:
            timestamp = time.time()
            self._frame_counter += 1

            # A detector failure or unavailable model carries no negative
            # evidence about the table. Keep the last authoritative ball
            # observation intact while still applying cue/table updates from
            # this frame. A successful observed-empty result deliberately
            # follows the normal extraction path and clears the ball list.
            if self._ball_detection_is_incomplete(detection_data):
                balls = (
                    [ball.copy() for ball in self._current_state.balls]
                    if self._current_state is not None
                    else []
                )
            else:
                balls = self._extract_ball_states(detection_data)

            # Extract cue state if present
            cue = self._extract_cue_state(detection_data)

            # Extract table state (convert from dict if needed)
            table = self._extract_table_state(detection_data)

            # Detect events based on state changes
            events = self._detect_events(balls)

            # Create coordinate metadata from detection data and table
            coordinate_metadata = self._create_coordinate_metadata(
                detection_data, table
            )

            # Create new game state
            new_state = GameState(
                timestamp=timestamp,
                frame_number=self._frame_counter,
                balls=balls,
                table=table,
                cue=cue,
                game_type=(
                    self._current_state.game_type
                    if self._current_state
                    else GameType.PRACTICE
                ),
                current_player=(
                    self._current_state.current_player if self._current_state else None
                ),
                scores=self._current_state.scores if self._current_state else {},
                is_break=self._current_state.is_break if self._current_state else False,
                last_shot=(
                    self._current_state.last_shot if self._current_state else None
                ),
                events=events,
                coordinate_metadata=coordinate_metadata,
            )

            # Validate state
            if self._validation_enabled:
                self._validate_state(new_state)

            # Check if state has changed significantly
            has_changed = self._has_significant_change(new_state, self._current_state)

            # Update state and history
            if self._current_state is not None:
                self._state_history.append(self._current_state)
            self._current_state = new_state

            # Update ball position tracking for event detection
            self._last_ball_positions = {ball.id: ball.position for ball in balls}

            # Update cue tracking for change detection
            self._last_cue_state = cue

            # Emit state change event ONLY if state changed significantly
            if has_changed:
                self._event_manager.emit_event(
                    "state_updated",
                    {
                        "frame_number": self._frame_counter,
                        "timestamp": timestamp,
                        "balls_count": len(balls),
                        "events": [event.to_dict() for event in events],
                    },
                )
                logger.debug(f"State changed for frame {self._frame_counter}")
            else:
                logger.debug(
                    f"No significant state change for frame {self._frame_counter}"
                )

            return new_state

    @staticmethod
    def _ball_detection_is_incomplete(detection_data: dict[str, Any]) -> bool:
        """Return whether this frame lacks authoritative ball evidence.

        Vision's status contract separates a successful empty detection from
        a detector failure or unavailable model. The latter statuses must not
        turn missing detections into a false "all balls disappeared" update.
        This boundary accepts status enums as well as their serialized string
        values without importing the Vision module into Core.
        """
        observation_status = detection_data.get("observation_status")
        if not isinstance(observation_status, dict):
            return False

        status = observation_status.get("ball_detection_status")
        status_value = getattr(status, "value", status)
        return isinstance(status_value, str) and status_value.strip().lower() in {
            "failed",
            "unavailable",
            "model_unavailable",
        }

    def _has_significant_change(
        self, new_state: GameState, old_state: Optional[GameState]
    ) -> bool:
        """Detect if state has changed significantly enough to warrant an event.

        Args:
            new_state: New game state to compare
            old_state: Previous game state (None on first frame)

        Returns:
            True if state changed significantly, False otherwise
        """
        # Always emit on first frame
        if old_state is None:
            return True

        # Check ball count change
        if len(new_state.balls) != len(old_state.balls):
            logger.debug(
                f"Ball count changed: {len(old_state.balls)} → {len(new_state.balls)}"
            )
            return True

        # Check if any ball position changed significantly
        # Create position maps keyed by ball ID for comparison
        old_positions = {ball.id: ball.position for ball in old_state.balls}
        new_positions = {ball.id: ball.position for ball in new_state.balls}

        for ball_id in new_positions:
            if ball_id not in old_positions:
                # New ball appeared
                logger.debug(f"New ball appeared: {ball_id}")
                return True

            old_pos = old_positions[ball_id]
            new_pos = new_positions[ball_id]

            # Calculate distance moved (in pixels)
            dx = new_pos.x - old_pos.x
            dy = new_pos.y - old_pos.y
            distance = (dx * dx + dy * dy) ** 0.5

            if distance >= self._position_change_threshold:
                logger.debug(
                    f"Ball {ball_id} moved {distance:.2f} pixels (threshold: {self._position_change_threshold})"
                )
                return True

        # Check for balls that disappeared
        for ball_id in old_positions:
            if ball_id not in new_positions:
                logger.debug(f"Ball disappeared: {ball_id}")
                return True

        # Check cue appearance/disappearance
        old_has_cue = old_state.cue is not None
        new_has_cue = new_state.cue is not None

        if old_has_cue != new_has_cue:
            logger.debug(f"Cue {'appeared' if new_has_cue else 'disappeared'}")
            return True

        # Check cue position/angle change
        if new_has_cue and old_has_cue:
            old_cue = old_state.cue
            new_cue = new_state.cue
            if isinstance(old_cue, CueState) and isinstance(new_cue, CueState):
                tip_distance = old_cue.tip_position.distance_to(new_cue.tip_position)
                if tip_distance >= self._position_change_threshold:
                    logger.debug(f"Cue tip moved {tip_distance:.2f} pixels")
                    return True

                angle_diff = abs(new_cue.angle - old_cue.angle)
                # Normalize to 0-180 range.
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff >= self._angle_change_threshold:
                    logger.debug(f"Cue angle changed {angle_diff:.2f} degrees")
                    return True

        # Check for game events (pocket, collision, etc.)
        if len(new_state.events) > 0:
            logger.debug(f"Game events detected: {len(new_state.events)}")
            return True

        # No significant changes detected
        return False

    @staticmethod
    def _coordinate_vector(
        payload: Any,
        *,
        context: str,
        fallback_resolution: Optional[tuple[int, int]] = None,
        coordinate_space: Optional[str] = None,
    ) -> Vector2D:
        """Parse a point/vector and normalize it to canonical 4K pixels.

        Serialized points need their own scale or an explicit pixel source
        resolution. A coordinate-space label alone is not a scale.
        """
        supported_spaces = {"4k_canonical", "pixel"}
        declared_space = (
            payload.get("coordinate_space", coordinate_space)
            if isinstance(payload, dict)
            else coordinate_space
        )
        if declared_space is not None and declared_space not in supported_spaces:
            raise ValueError(
                f"{context} uses unsupported coordinate space {declared_space!r}"
            )
        if isinstance(payload, Vector2D):
            return payload.to_4k_canonical()
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            raise ValueError(
                f"{context} must be a scale-aware mapping with x and y coordinates"
            )

        point_space = declared_space
        if payload.get("scale") is not None:
            return Vector2D.from_dict(payload).to_4k_canonical()
        resolution = payload.get("source_resolution", fallback_resolution)
        if resolution is not None and point_space != "4k_canonical":
            normalized_resolution = GameStateManager._normalize_resolution(
                resolution, f"{context} source resolution"
            )
            return Vector2D.from_resolution(
                payload["x"], payload["y"], normalized_resolution
            ).to_4k_canonical()
        raise ValueError(f"{context} is missing coordinate scale metadata")

    @staticmethod
    def _normalize_resolution(resolution: Any, context: str) -> tuple[int, int]:
        """Validate an image resolution without truncating fractional values."""
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError(f"{context} must contain a width/height pair")
        normalized: list[int] = []
        for value in resolution:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or not float(value).is_integer()
            ):
                raise ValueError(f"{context} must contain positive whole pixels")
            normalized.append(int(value))
        return normalized[0], normalized[1]

    @staticmethod
    def _source_resolution(
        payload: dict[str, Any], detection_data: dict[str, Any]
    ) -> Optional[tuple[int, int]]:
        """Find the exact resolution associated with one coordinate payload."""
        resolution = payload.get("source_resolution")
        if resolution is None:
            resolution = detection_data.get("source_resolution")
        if resolution is None:
            return None
        return GameStateManager._normalize_resolution(
            resolution, "Coordinate source resolution"
        )

    def _extract_ball_states(self, detection_data: dict[str, Any]) -> list[BallState]:
        """Extract ball states from detection data."""
        balls = []
        ball_detections = detection_data.get("balls", [])

        for detection in ball_detections:
            if isinstance(detection, BallState):
                balls.append(detection.copy())
                continue
            if not isinstance(detection, dict):
                raise ValueError("Ball detections must be mappings or BallState values")
            # Vision serialization supplies nested, scale-aware position and
            # velocity objects. Flat x/y and vx/vy records are no longer a
            # supported physics-state schema.
            source_resolution = self._source_resolution(detection, detection_data)
            coordinate_space = detection.get(
                "coordinate_space", detection_data.get("coordinate_space")
            )
            position_data = detection.get("position")
            if position_data is None:
                raise ValueError("Ball detection requires a nested position mapping")
            position = self._coordinate_vector(
                position_data,
                context="ball position",
                fallback_resolution=source_resolution,
                coordinate_space=coordinate_space,
            )

            velocity_data = detection.get("velocity")
            if velocity_data is None:
                raise ValueError("Ball detection requires a nested velocity mapping")
            velocity = self._coordinate_vector(
                velocity_data,
                context="ball velocity",
                fallback_resolution=source_resolution,
                coordinate_space=coordinate_space,
            )

            radius = detection.get("radius", BALL_RADIUS_4K)
            radius_scale = detection.get("radius_scale")
            if radius_scale is None:
                raise ValueError("Ball radius scale metadata is required")
            if detection.get("mass_unit", "kg") != "kg":
                raise ValueError("Ball mass must use kg")

            observation_timestamp = detection.get("timestamp")
            if observation_timestamp is None:
                observation_timestamp = detection_data.get("timestamp")
            if observation_timestamp is None:
                observation_timestamp = time.time()

            ball = BallState(
                id=detection.get("id", f"ball_{len(balls)}"),
                position=position,
                velocity=velocity,
                radius=radius,
                radius_scale=tuple(radius_scale),
                mass=detection.get("mass", BALL_MASS_KG),
                spin=(
                    Vector2D.from_4k(
                        detection.get("spin_x", 0.0), detection.get("spin_y", 0.0)
                    )
                    if "spin_x" in detection or "spin_y" in detection
                    else Vector2D.zero()
                ),
                is_cue_ball=detection.get("is_cue_ball", False),
                is_pocketed=detection.get("is_pocketed", False),
                number=detection.get("number"),
                confidence=detection.get("confidence", 1.0),
                is_observed=detection.get("is_observed", True),
                last_update=observation_timestamp,
            )
            balls.append(ball)

        return balls

    def _extract_cue_state(self, detection_data: dict[str, Any]) -> Optional[CueState]:
        """Extract cue state from detection data."""
        cue_data = detection_data.get("cue")
        if not cue_data:
            return None
        if any(key in cue_data for key in ("tip_x", "tip_y", "impact_x", "impact_y")):
            raise ValueError("Cue coordinates must use nested scale-aware points")
        if cue_data.get("length_unit", "meters") != "meters":
            raise ValueError("Cue length must use meters")
        if cue_data.get("tip_radius_unit", "meters") != "meters":
            raise ValueError("Cue tip radius must use meters")

        # Vision serialization supplies a nested scale-aware tip position.
        source_resolution = self._source_resolution(cue_data, detection_data)
        coordinate_space = cue_data.get(
            "coordinate_space", detection_data.get("coordinate_space")
        )
        tip_position_data = cue_data.get("tip_position")
        if tip_position_data is None:
            raise ValueError("Cue tip position requires a nested mapping")
        tip_position = self._coordinate_vector(
            tip_position_data,
            context="cue tip position",
            fallback_resolution=source_resolution,
            coordinate_space=coordinate_space,
        )

        # Handle impact point
        impact_point = None
        if "impact_point" in cue_data:
            impact_data = cue_data["impact_point"]
            if impact_data is not None:
                impact_point = self._coordinate_vector(
                    impact_data,
                    context="cue impact point",
                    fallback_resolution=source_resolution,
                    coordinate_space=coordinate_space,
                )

        butt_position = None
        if cue_data.get("butt_position") is not None:
            butt_position = self._coordinate_vector(
                cue_data["butt_position"],
                context="cue butt position",
                fallback_resolution=source_resolution,
                coordinate_space=coordinate_space,
            )

        cue_dimensions: dict[str, float] = {}
        if "length" in cue_data:
            cue_dimensions["length"] = cue_data["length"]
        if "tip_radius" in cue_data:
            cue_dimensions["tip_radius"] = cue_data["tip_radius"]

        return CueState(
            tip_position=tip_position,
            angle=cue_data.get("angle", 0.0),
            elevation=cue_data.get("elevation", 0.0),
            estimated_force=cue_data.get("force", 0.0),
            impact_point=impact_point,
            butt_position=butt_position,
            is_visible=cue_data.get("is_visible", True),
            confidence=cue_data.get("confidence", 1.0),
            last_update=cue_data.get("timestamp", time.time()),
            **cue_dimensions,
        )

    def _create_coordinate_metadata(
        self, detection_data: dict[str, Any], table: TableState
    ) -> Optional[CoordinateMetadata]:
        """Create metadata whose spatial values use canonical 4K pixels."""
        from backend.config import config

        camera_resolution = self._source_resolution({}, detection_data)
        # Legacy callers that omit this field may use the configured camera
        # size as a compatibility fallback. An explicit null from the current
        # observation means its source dimensions are unavailable and must not
        # be replaced with a potentially different camera mode.
        if camera_resolution is None and "source_resolution" not in detection_data:
            configured_resolution = config.get("vision.camera.resolution")
            if (
                isinstance(configured_resolution, (tuple, list))
                and len(configured_resolution) == 2
            ):
                configured = (
                    int(configured_resolution[0]),
                    int(configured_resolution[1]),
                )
                if all(value > 0 for value in configured):
                    camera_resolution = configured

        table_bounds = None
        if table.playing_area_corners and len(table.playing_area_corners) >= 4:
            canonical_corners = [
                corner.to_4k_canonical() for corner in table.playing_area_corners
            ]
            x_coords = [corner.x for corner in canonical_corners]
            y_coords = [corner.y for corner in canonical_corners]
            table_bounds = (
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            )

        pixels_per_meter = detection_data.get("pixels_per_meter")
        if pixels_per_meter is None:
            pixel_scales = table.calculate_pixels_per_meter()
            if pixel_scales is not None:
                pixels_per_meter = pixel_scales[0]

        return CoordinateMetadata(
            camera_resolution=camera_resolution,
            table_bounds=table_bounds,
            table_bounds_scale=(1.0, 1.0),
            coordinate_space="4k_canonical",
            pixels_per_meter=pixels_per_meter,
        )

    def _extract_table_state(self, detection_data: dict[str, Any]) -> TableState:
        """Parse scale-aware table geometry into canonical Core state."""
        table_data = detection_data.get("table")
        if table_data is None:
            if self._default_table is None:
                raise ValueError(
                    "No table data available and no default table configured"
                )
            return self._default_table
        if isinstance(table_data, TableState):
            return table_data
        if not isinstance(table_data, dict):
            raise ValueError("Table detection must be a mapping or TableState value")
        deprecated_fields = {
            "rail_height",
            "real_width_meters",
            "real_height_meters",
        }.intersection(table_data)
        if deprecated_fields:
            names = ", ".join(sorted(deprecated_fields))
            raise ValueError(f"Unsupported legacy table fields: {names}")

        source_resolution = self._source_resolution(table_data, detection_data)
        coordinate_space = table_data.get(
            "coordinate_space", detection_data.get("coordinate_space")
        )
        if coordinate_space not in {None, "4k_canonical", "pixel"}:
            raise ValueError(f"Unsupported table coordinate space {coordinate_space!r}")
        declared_pixel_space = coordinate_space == "pixel"

        raw_corners = (
            table_data.get("playing_area_corners") or table_data.get("corners") or []
        )
        corners = [
            self._coordinate_vector(
                corner,
                context=f"table boundary corner {index}",
                fallback_resolution=source_resolution,
                coordinate_space=coordinate_space,
            )
            for index, corner in enumerate(raw_corners)
        ]
        if corners and len(corners) != 4:
            raise ValueError("Table boundaries require exactly four corners")

        raw_width = table_data.get("width")
        raw_height = table_data.get("height")
        if raw_width is None or raw_height is None:
            if len(corners) == 4:
                raw_width = (
                    corners[0].distance_to(corners[1])
                    + corners[2].distance_to(corners[3])
                ) / 2
                raw_height = (
                    corners[1].distance_to(corners[2])
                    + corners[3].distance_to(corners[0])
                ) / 2
                dimensions_scale = (1.0, 1.0)
            elif self._default_table is not None:
                return self._default_table
            else:
                raise ValueError("Table dimensions are required with its boundary")
        else:
            dimensions_unit = table_data.get("dimensions_unit")
            if dimensions_unit not in {None, "canonical_pixels", "pixels"}:
                raise ValueError("Table dimensions must use pixel units")
            if dimensions_unit == "pixels" and (
                not declared_pixel_space or source_resolution is None
            ):
                raise ValueError(
                    "Pixel table dimensions require pixel coordinate space and source resolution"
                )
            raw_dimensions_scale = table_data.get("dimensions_scale")
            if raw_dimensions_scale is not None:
                if (
                    not isinstance(raw_dimensions_scale, (tuple, list))
                    or len(raw_dimensions_scale) != 2
                ):
                    raise ValueError("Table dimensions scale must contain two factors")
                dimensions_scale = (
                    float(raw_dimensions_scale[0]),
                    float(raw_dimensions_scale[1]),
                )
            else:
                width_scale = table_data.get("width_scale")
                height_scale = table_data.get("height_scale")
                if width_scale is not None and height_scale is not None:
                    if (
                        not isinstance(width_scale, (tuple, list))
                        or len(width_scale) != 2
                        or not isinstance(height_scale, (tuple, list))
                        or len(height_scale) != 2
                    ):
                        raise ValueError(
                            "Table width and height scales must each contain two factors"
                        )
                    dimensions_scale = (width_scale[0], height_scale[1])
                elif (
                    dimensions_unit == "pixels"
                    and declared_pixel_space
                    and source_resolution is not None
                ):
                    from .resolution_converter import ResolutionConverter

                    dimensions_scale = ResolutionConverter.calculate_scale_to_4k(
                        source_resolution
                    )
                else:
                    raise ValueError("Table dimensions scale metadata is required")

        table_width = float(raw_width) * dimensions_scale[0]
        table_height = float(raw_height) * dimensions_scale[1]
        if not all(
            math.isfinite(value) and value > 0 for value in (table_width, table_height)
        ):
            raise ValueError(
                "Table dimensions must be finite positive canonical pixel extents"
            )

        pocket_positions: list[Vector2D] = []
        raw_pockets = table_data.get("pockets")
        if raw_pockets is not None:
            for index, pocket in enumerate(raw_pockets):
                if not isinstance(pocket, dict) or "position" not in pocket:
                    raise ValueError(
                        f"Pocket {index} is missing its scale-aware position"
                    )
                pocket_positions.append(
                    self._coordinate_vector(
                        pocket["position"],
                        context=f"pocket {index} position",
                        fallback_resolution=self._source_resolution(pocket, table_data),
                        coordinate_space=pocket.get(
                            "coordinate_space", coordinate_space
                        ),
                    )
                )

        if not pocket_positions and len(corners) == 4:
            top_left, top_right, bottom_right, bottom_left = corners
            pocket_positions = [
                top_left,
                Vector2D.from_4k(
                    (top_left.x + top_right.x) / 2, (top_left.y + top_right.y) / 2
                ),
                top_right,
                bottom_left,
                Vector2D.from_4k(
                    (bottom_left.x + bottom_right.x) / 2,
                    (bottom_left.y + bottom_right.y) / 2,
                ),
                bottom_right,
            ]
        elif not pocket_positions and self._default_table is not None:
            width_ratio = table_width / self._default_table.width
            height_ratio = table_height / self._default_table.height
            pocket_positions = [
                Vector2D.from_4k(pocket.x * width_ratio, pocket.y * height_ratio)
                for pocket in self._default_table.pocket_positions
            ]
        elif not pocket_positions:
            from .constants_4k import CANONICAL_HEIGHT, CANONICAL_WIDTH

            left = (CANONICAL_WIDTH - table_width) / 2
            top = (CANONICAL_HEIGHT - table_height) / 2
            right, bottom = left + table_width, top + table_height
            middle = (left + right) / 2
            pocket_positions = [
                Vector2D.from_4k(left, top),
                Vector2D.from_4k(middle, top),
                Vector2D.from_4k(right, top),
                Vector2D.from_4k(left, bottom),
                Vector2D.from_4k(middle, bottom),
                Vector2D.from_4k(right, bottom),
            ]
        if len(pocket_positions) != 6:
            raise ValueError("Table state requires exactly six pocket positions")

        real_dimensions = table_data.get("real_world_dimensions")
        if isinstance(real_dimensions, dict):
            if real_dimensions.get("unit") != "meters":
                raise ValueError(
                    "Real table dimensions must be explicitly measured in meters"
                )
            real_width = real_dimensions.get("width")
            real_height = real_dimensions.get("height")
        elif real_dimensions is not None:
            if table_data.get("real_world_dimensions_unit") != "meters":
                raise ValueError(
                    "Real table dimensions require an explicit meters unit"
                )
            real_width, real_height = real_dimensions
        else:
            real_width = None
            real_height = None

        kwargs: dict[str, Any] = {}
        if table_data.get("cushion_height_unit", "meters") != "meters":
            raise ValueError("Cushion height must use meters")
        for field_name in (
            "cushion_elasticity",
            "surface_friction",
            "surface_slope",
            "cushion_height",
        ):
            if field_name in table_data:
                kwargs[field_name] = table_data[field_name]
            elif self._default_table is not None:
                kwargs[field_name] = getattr(self._default_table, field_name)
        if self._default_table is not None:
            default_radius = self._default_table.pocket_radius
        else:
            default_radius = POCKET_RADIUS_4K
        radius = table_data.get("pocket_radius", default_radius)
        radius_scale = table_data.get("pocket_radius_scale")
        if "pocket_radius" in table_data and radius_scale is None:
            raise ValueError("Pocket radius scale metadata is required")
        if radius_scale is None:
            radius_scale = (1.0, 1.0)

        return TableState(
            width=table_width,
            height=table_height,
            dimensions_scale=(1.0, 1.0),
            pocket_positions=pocket_positions,
            pocket_radius=radius,
            pocket_radius_scale=tuple(radius_scale),
            playing_area_corners=corners or None,
            real_width_meters=real_width,
            real_height_meters=real_height,
            **kwargs,
        )

    def _detect_events(self, balls: list[BallState]) -> list[GameEvent]:
        """Detect game events based on state changes (FR-CORE-005)."""
        events = []
        timestamp = time.time()

        # Detect ball motion
        for ball in balls:
            if ball.id in self._last_ball_positions:
                last_pos = self._last_ball_positions[ball.id]
                distance = (
                    (ball.position.x - last_pos.x) ** 2
                    + (ball.position.y - last_pos.y) ** 2
                ) ** 0.5

                if distance > self._motion_threshold:
                    events.append(
                        GameEvent(
                            timestamp=timestamp,
                            event_type="ball_motion",
                            description=f"Ball {ball.id} moved {distance:.1f} pixels",
                            data={
                                "ball_id": ball.id,
                                "distance": distance,
                                "from_position": last_pos.to_dict(),
                                "to_position": ball.position.to_dict(),
                            },
                            frame_number=self._frame_counter,
                        )
                    )

        # Detect pocketed balls
        for ball in balls:
            if ball.is_pocketed:
                events.append(
                    GameEvent(
                        timestamp=timestamp,
                        event_type="ball_pocketed",
                        description=f"Ball {ball.number or ball.id} pocketed",
                        data={
                            "ball_id": ball.id,
                            "ball_number": ball.number,
                            "is_cue_ball": ball.is_cue_ball,
                        },
                        frame_number=self._frame_counter,
                    )
                )

        return events

    def set_game_type(self, game_type: GameType) -> None:
        """Set the current game type."""
        with self._lock:
            if self._current_state:
                self._current_state.game_type = game_type
                logger.info(f"Game type set to {game_type.value}")

    def get_current_state(self) -> Optional[GameState]:
        """Get current game state (FR-CORE-002)."""
        with self._lock:
            return self._current_state

    def get_state_history(self, frames: int = 100) -> list[GameState]:
        """Get historical game states (FR-CORE-004)."""
        with self._lock:
            # Return the last N frames, filtering out None values
            history = [state for state in self._state_history if state is not None]
            return history[-frames:] if frames > 0 else history

    def get_ball_by_id(self, ball_id: str) -> Optional[BallState]:
        """Get specific ball state by ID."""
        if not self._current_state:
            return None

        for ball in self._current_state.balls:
            if ball.id == ball_id:
                return ball
        return None

    def get_cue_ball(self) -> Optional[BallState]:
        """Get cue ball state."""
        if not self._current_state:
            return None

        for ball in self._current_state.balls:
            if ball.is_cue_ball:
                return ball
        return None

    def _required_table(self, table_config: Optional[TableState]) -> TableState:
        """Return the selected table or reject a reset before state mutation."""
        table = table_config or self._default_table
        if table is None:
            raise ValueError("A calibrated table is required to reset the game")
        return table

    def reset_game(
        self,
        game_type: GameType = GameType.PRACTICE,
        table_config: Optional[TableState] = None,
    ) -> None:
        """Reset game to initial state (FR-CORE-003)."""
        selected_table = self._required_table(table_config)
        with self._lock:
            self._frame_counter = 0
            self._start_time = time.time()
            self._state_history.clear()
            self._last_ball_positions.clear()

            # Create initial state with empty table
            self._current_state = GameState(
                timestamp=self._start_time,
                frame_number=0,
                balls=[],
                table=selected_table,
                cue=None,
                game_type=game_type,
                current_player=1,
                scores={1: 0, 2: 0},
                is_break=True,
                last_shot=None,
                events=[],
            )

            self._event_manager.emit_event(
                "game_reset",
                {"game_type": game_type.value, "timestamp": self._start_time},
            )

            logger.info(f"Game reset to {game_type.value}")

    def _validate_state(self, state: GameState) -> None:
        """Validate game state for consistency using comprehensive StateValidator.

        Raises:
            StateValidationError: If validation fails
        """
        # Run comprehensive validation
        validation_result = self._state_validator.validate_game_state(state)

        # Add frame sequence validation (specific to GameStateManager)
        if (
            self._current_state
            and state.frame_number <= self._current_state.frame_number
        ):
            validation_result.add_error("Frame number must increase")

        # Apply auto-corrections if enabled
        if self._auto_correct_enabled and validation_result.corrected_values:
            self._apply_corrections(state, validation_result.corrected_values)

        # Update state validation status
        state.is_valid = validation_result.is_valid
        state.validation_errors = validation_result.errors.copy()

        # Log validation results
        if validation_result.warnings:
            logger.warning(
                f"State validation warnings: {'; '.join(validation_result.warnings)}"
            )

        if validation_result.errors:
            error_msg = (
                f"State validation failed: {'; '.join(validation_result.errors)}"
            )
            if not self._auto_correct_enabled:
                raise StateValidationError(error_msg)
            else:
                logger.debug(error_msg)

        # Log validation statistics
        if validation_result.errors or validation_result.warnings:
            logger.debug(f"Validation confidence: {validation_result.confidence:.2f}")

    def _apply_corrections(self, state: GameState, corrections: dict[str, Any]) -> None:
        """Apply auto-corrections to the game state."""
        corrections_applied = []

        for field, value in corrections.items():
            if field.startswith("ball_") and field.endswith("_position"):
                # Extract ball ID and update position
                ball_id = field.replace("ball_", "").replace("_position", "")
                for ball in state.balls:
                    if ball.id == ball_id:
                        ball.position = value
                        corrections_applied.append(f"position of ball {ball_id}")
                        break

            elif field.startswith("ball_") and field.endswith("_velocity"):
                # Extract ball ID and update velocity
                ball_id = field.replace("ball_", "").replace("_velocity", "")
                for ball in state.balls:
                    if ball.id == ball_id:
                        ball.velocity = value
                        corrections_applied.append(f"velocity of ball {ball_id}")
                        break

            elif field.startswith("ball_") and field.endswith("_pocketed"):
                # Extract ball ID and update pocketed status
                ball_id = field.replace("ball_", "").replace("_pocketed", "")
                for ball in state.balls:
                    if ball.id == ball_id:
                        ball.is_pocketed = value
                        corrections_applied.append(f"pocketed status of ball {ball_id}")
                        break

        if corrections_applied:
            logger.info(f"Auto-corrections applied: {', '.join(corrections_applied)}")

    def subscribe_to_events(self, event_type: str, callback: Callable[..., Any]) -> str:
        """Subscribe to state change events (FR-CORE-011)."""
        return self._event_manager.subscribe_to_events(event_type, callback)

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events (FR-CORE-012)."""
        return self._event_manager.unsubscribe(subscription_id)

    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """Save current state to disk (FR-CORE-013).

        Args:
            filepath: Custom save path, or None for auto-generated

        Returns:
            Path where state was saved
        """
        if filepath is None:
            timestamp = int(time.time())
            filepath = self._persistence_path / f"game_state_{timestamp}.pkl"

        with self._lock:
            save_data = {
                "current_state": self._current_state,
                "state_history": list(self._state_history),
                "frame_counter": self._frame_counter,
                "start_time": self._start_time,
            }

            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)

        logger.info(f"State saved to {filepath}")
        return filepath

    def load_state(self, filepath: Path) -> None:
        """Load state from disk (FR-CORE-014).

        Args:
            filepath: Path to saved state file

        Raises:
            FileNotFoundError: If save file doesn't exist
            ValueError: If save file is corrupted
        """
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)

            if not isinstance(save_data, dict):
                raise ValueError("save root must be a mapping")

            # Decode and validate every saved field before mutating live state.
            # A corrupt save must not leave a new current state paired with old
            # history, counters, or cached positions.
            current_state = save_data["current_state"]
            saved_history = save_data["state_history"]
            frame_counter = save_data["frame_counter"]
            start_time = save_data["start_time"]
            if current_state is not None and not isinstance(current_state, GameState):
                raise ValueError("current_state has an unsupported type")
            if not isinstance(saved_history, (list, tuple, deque)) or any(
                not isinstance(state, GameState) for state in saved_history
            ):
                raise ValueError("state_history has an unsupported shape")
            if (
                isinstance(frame_counter, bool)
                or not isinstance(frame_counter, int)
                or frame_counter < 0
            ):
                raise ValueError("frame_counter must be a non-negative integer")
            if (
                isinstance(start_time, bool)
                or not isinstance(start_time, (int, float))
                or not math.isfinite(start_time)
            ):
                raise ValueError("start_time must be finite")

            restored_history = deque(saved_history, maxlen=self._state_history.maxlen)
            restored_ball_positions = (
                {ball.id: ball.position for ball in current_state.balls}
                if current_state is not None
                else {}
            )
        except (
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            raise ValueError(f"Corrupted state file: {e}")

        with self._lock:
            self._current_state = current_state
            self._state_history = restored_history
            self._frame_counter = frame_counter
            self._start_time = float(start_time)
            self._last_ball_positions = restored_ball_positions

        self._event_manager.emit_event(
            "state_loaded",
            {"filepath": str(filepath), "frame_number": self._frame_counter},
        )

        logger.info(f"State loaded from {filepath}")

    def export_state_json(self, filepath: Path) -> None:
        """Export current state as JSON for debugging."""
        if not self._current_state:
            raise ValueError("No current state to export")

        # Use model serializers so canonical scales and physical units survive
        # state export and reload as part of the same transport contract.
        state_dict = self._current_state.to_dict()

        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

        logger.info(f"State exported to JSON: {filepath}")

    def get_statistics(self) -> dict[str, Any]:
        """Get state management statistics (FR-CORE-015)."""
        with self._lock:
            return {
                "current_frame": self._frame_counter,
                "history_size": len(self._state_history),
                "uptime_seconds": time.time() - self._start_time,
                "validation_enabled": self._validation_enabled,
                "auto_correct_enabled": self._auto_correct_enabled,
                "current_balls_count": (
                    len(self._current_state.balls) if self._current_state else 0
                ),
                "active_balls_count": (
                    len([b for b in self._current_state.balls if not b.is_pocketed])
                    if self._current_state
                    else 0
                ),
                "last_update": (
                    self._current_state.timestamp if self._current_state else None
                ),
            }

    def set_validation_config(
        self, enabled: bool = True, auto_correct: bool = True
    ) -> None:
        """Configure state validation behavior."""
        self._validation_enabled = enabled
        self._auto_correct_enabled = auto_correct

        # Update validator configuration
        self._state_validator.enable_auto_correction = auto_correct

        logger.info(
            f"Validation config: enabled={enabled}, auto_correct={auto_correct}"
        )

    def force_validation(self) -> tuple[bool, list[str]]:
        """Force validation of current state and return results."""
        if not self._current_state:
            return False, ["No current state"]

        try:
            # Run comprehensive validation without frame sequence check
            validation_result = self._state_validator.validate_game_state(
                self._current_state
            )

            return validation_result.is_valid, validation_result.errors
        except Exception as e:
            logger.error(f"Error during force validation: {e}")
            return False, [str(e)]

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get detailed validation statistics and configuration."""
        base_stats = self.get_statistics()
        validator_stats = self._state_validator.get_validation_statistics()

        return {
            **base_stats,
            "validation_config": validator_stats,
            "validator_type": "StateValidator",
        }

    def validate_with_physics(
        self, previous_state: Optional[GameState] = None, dt: float = 1 / 30
    ) -> ValidationResult:
        """Perform physics validation between states.

        Args:
            previous_state: Previous game state for physics comparison
            dt: Time delta between states in seconds

        Returns:
            ValidationResult with physics validation details
        """
        if not self._current_state:
            result = ValidationResult()
            result.add_error("No current state to validate")
            return result

        return self._state_validator.validate_physics(
            self._current_state, previous_state, dt
        )
