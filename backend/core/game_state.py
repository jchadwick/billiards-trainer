"""Game State Management.

Main game state management for tracking balls, cue, table state.
Implements FR-CORE-001 through FR-CORE-015 requirements.
"""

import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

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
from .rules import GameRules
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
        self._state_history: deque = deque(maxlen=max_history_frames)
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
        self._motion_threshold = 0.005  # meters/frame threshold for motion detection

        # Change detection configuration
        self._position_change_threshold = (
            5.0  # pixels - minimum position change to consider significant
        )
        self._angle_change_threshold = (
            2.0  # degrees - minimum angle change to consider significant
        )
        self._last_cue_state: Optional[CueState] = None

        self._game_rules: Optional[GameRules] = None

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

            # Extract ball positions and states
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
                        "events": [asdict(event) for event in events],
                    },
                )
                logger.debug(f"State changed for frame {self._frame_counter}")
            else:
                logger.debug(
                    f"No significant state change for frame {self._frame_counter}"
                )

            return new_state

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

            # Check position change (tip or base)
            if hasattr(old_cue, "tip") and hasattr(new_cue, "tip"):
                if old_cue.tip and new_cue.tip:
                    dx = new_cue.tip.x - old_cue.tip.x
                    dy = new_cue.tip.y - old_cue.tip.y
                    tip_distance = (dx * dx + dy * dy) ** 0.5
                    if tip_distance >= self._position_change_threshold:
                        logger.debug(f"Cue tip moved {tip_distance:.2f} pixels")
                        return True

            # Check angle change
            if hasattr(old_cue, "angle") and hasattr(new_cue, "angle"):
                if old_cue.angle is not None and new_cue.angle is not None:
                    angle_diff = abs(new_cue.angle - old_cue.angle)
                    # Normalize to 0-180 range
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

    def _extract_ball_states(self, detection_data: dict[str, Any]) -> list[BallState]:
        """Extract ball states from detection data."""
        balls = []
        ball_detections = detection_data.get("balls", [])

        for detection in ball_detections:
            # Handle both nested dict format (from integration_service) and flat format (legacy)
            position_data = detection.get("position", {})
            if isinstance(position_data, dict):
                pos_x = position_data.get("x", 0.0)
                pos_y = position_data.get("y", 0.0)
            else:
                # Fallback to flat format
                pos_x = detection.get("x", 0.0)
                pos_y = detection.get("y", 0.0)

            velocity_data = detection.get("velocity", {})
            if isinstance(velocity_data, dict):
                vel_x = velocity_data.get("x", 0.0)
                vel_y = velocity_data.get("y", 0.0)
            else:
                # Fallback to flat format
                vel_x = detection.get("vx", 0.0)
                vel_y = detection.get("vy", 0.0)

            ball = BallState(
                id=detection.get("id", f"ball_{len(balls)}"),
                position=Vector2D.from_4k(pos_x, pos_y),
                velocity=Vector2D.from_4k(vel_x, vel_y),
                radius=36.0,  # Standard ball radius in 4K pixels (BALL_RADIUS_4K)
                mass=0.17,  # Standard ball mass in kg
                spin=(
                    Vector2D.from_4k(
                        detection.get("spin_x", 0.0), detection.get("spin_y", 0.0)
                    )
                    if "spin_x" in detection
                    else None
                ),
                is_cue_ball=detection.get("is_cue_ball", False),
                is_pocketed=detection.get("is_pocketed", False),
                number=detection.get("number"),
                confidence=detection.get("confidence", 1.0),
                last_update=detection.get("timestamp", time.time()),
            )
            balls.append(ball)

        return balls

    def _extract_cue_state(self, detection_data: dict[str, Any]) -> Optional[CueState]:
        """Extract cue state from detection data."""
        cue_data = detection_data.get("cue")
        if not cue_data:
            return None

        # Handle both nested dict format (from integration_service) and flat format (legacy)
        tip_position_data = cue_data.get("tip_position", {})
        if isinstance(tip_position_data, dict):
            tip_x = tip_position_data.get("x", 0.0)
            tip_y = tip_position_data.get("y", 0.0)
        else:
            # Fallback to flat format
            tip_x = cue_data.get("tip_x", 0.0)
            tip_y = cue_data.get("tip_y", 0.0)

        # Handle impact point
        impact_point = None
        if "impact_point" in cue_data:
            impact_data = cue_data["impact_point"]
            if isinstance(impact_data, dict):
                impact_point = Vector2D.from_4k(
                    impact_data.get("x", 0.0), impact_data.get("y", 0.0)
                )
        elif "impact_x" in cue_data and "impact_y" in cue_data:
            # Fallback to flat format
            impact_point = Vector2D.from_4k(cue_data["impact_x"], cue_data["impact_y"])

        return CueState(
            tip_position=Vector2D.from_4k(tip_x, tip_y),
            angle=cue_data.get("angle", 0.0),
            elevation=cue_data.get("elevation", 0.0),
            estimated_force=cue_data.get("force", 0.0),
            impact_point=impact_point,
            is_visible=cue_data.get("is_visible", True),
            confidence=cue_data.get("confidence", 1.0),
            last_update=cue_data.get("timestamp", time.time()),
        )

    def _create_coordinate_metadata(
        self, detection_data: dict[str, Any], table: TableState
    ) -> Optional[CoordinateMetadata]:
        """Create coordinate metadata from detection data and table state.

        Args:
            detection_data: Vision detection results
            table: Table state with playing area information

        Returns:
            CoordinateMetadata instance with camera resolution and table bounds
        """
        from config import config

        # Get camera resolution from config (fallback to default)
        camera_res_list = config.get("vision.camera.resolution", [1920, 1080])
        camera_resolution = (
            tuple(camera_res_list)
            if camera_res_list and len(camera_res_list) >= 2
            else None
        )

        # Calculate table bounds from playing area corners if available
        table_bounds = None
        if table.playing_area_corners and len(table.playing_area_corners) >= 4:
            x_coords = [corner.x for corner in table.playing_area_corners]
            y_coords = [corner.y for corner in table.playing_area_corners]
            table_bounds = (
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            )

        # Get pixels_per_meter from detection_data if available (from calibration)
        pixels_per_meter = detection_data.get("pixels_per_meter")

        return CoordinateMetadata(
            camera_resolution=camera_resolution,
            table_bounds=table_bounds,
            coordinate_space="world_meters",  # Ball positions are in world meters
            pixels_per_meter=pixels_per_meter,
        )

    def _extract_table_state(self, detection_data: dict[str, Any]) -> TableState:
        """Extract table state from detection data."""
        table_data = detection_data.get("table")

        # If no table data, use default (which may be None)
        if table_data is None:
            if self._default_table is None:
                raise ValueError(
                    "No table data available and no default table configured"
                )
            return self._default_table

        if isinstance(table_data, TableState):
            return table_data

        # Convert from dict format (from integration_service)
        # Extract pocket positions from pockets list
        pocket_positions = []
        if "pockets" in table_data:
            for pocket in table_data["pockets"]:
                pos_data = pocket.get("position", {})
                if isinstance(pos_data, dict):
                    pocket_positions.append(
                        Vector2D.from_4k(pos_data.get("x", 0.0), pos_data.get("y", 0.0))
                    )

        # Use actual detected/calibrated dimensions
        table_width = table_data.get("width")
        table_height = table_data.get("height")

        if table_width is None or table_height is None:
            if self._default_table is None:
                raise ValueError(
                    "Table dimensions not provided and no default available"
                )
            return self._default_table

        # Extract corners early so we can use them for pocket calculation if needed
        detected_corners = None
        if "corners" in table_data:
            corners = []
            for corner in table_data["corners"]:
                if isinstance(corner, dict):
                    corners.append(
                        Vector2D.from_4k(corner.get("x", 0.0), corner.get("y", 0.0))
                    )
            if len(corners) >= 4:
                detected_corners = corners

        # If no pockets defined, generate standard 6-pocket positions
        if not pocket_positions:
            if detected_corners:
                # Use detected corners to calculate pocket positions
                # Corners are typically: [top-left, top-right, bottom-right, bottom-left]
                tl, tr, br, bl = (
                    detected_corners[0],
                    detected_corners[1],
                    detected_corners[2],
                    detected_corners[3],
                )

                # Calculate middle points along top and bottom edges
                top_middle = Vector2D.from_4k((tl.x + tr.x) / 2, (tl.y + tr.y) / 2)
                bottom_middle = Vector2D.from_4k((bl.x + br.x) / 2, (bl.y + br.y) / 2)

                pocket_positions = [
                    tl,  # Top-left
                    top_middle,  # Top-middle
                    tr,  # Top-right
                    bl,  # Bottom-left
                    bottom_middle,  # Bottom-middle
                    br,  # Bottom-right
                ]
            elif self._default_table is not None:
                # Scale default pocket positions to match actual table dimensions
                width_scale = table_width / self._default_table.width
                height_scale = table_height / self._default_table.height

                pocket_positions = [
                    Vector2D.from_4k(pocket.x * width_scale, pocket.y * height_scale)
                    for pocket in self._default_table.pocket_positions
                ]
            else:
                # Generate standard pocket positions from table dimensions
                # Assume table is centered in 4K frame and calculate corner/middle positions
                from .constants_4k import CANONICAL_HEIGHT, CANONICAL_WIDTH

                # Calculate approximate table position (centered in frame)
                left = (CANONICAL_WIDTH - table_width) / 2
                top = (CANONICAL_HEIGHT - table_height) / 2
                right = left + table_width
                bottom = top + table_height
                center_x = (left + right) / 2

                pocket_positions = [
                    Vector2D.from_4k(left, top),  # Top-left
                    Vector2D.from_4k(center_x, top),  # Top-middle
                    Vector2D.from_4k(right, top),  # Top-right
                    Vector2D.from_4k(left, bottom),  # Bottom-left
                    Vector2D.from_4k(center_x, bottom),  # Bottom-middle
                    Vector2D.from_4k(right, bottom),  # Bottom-right
                ]

        # Ensure we have exactly 6 pockets (required for pool table)
        if len(pocket_positions) != 6:
            logger.warning(
                f"Expected 6 pockets but got {len(pocket_positions)}. "
                "Generating standard pocket positions."
            )
            # Generate standard positions as fallback
            from .constants_4k import CANONICAL_HEIGHT, CANONICAL_WIDTH

            left = (CANONICAL_WIDTH - table_width) / 2
            top = (CANONICAL_HEIGHT - table_height) / 2
            right = left + table_width
            bottom = top + table_height
            center_x = (left + right) / 2

            pocket_positions = [
                Vector2D.from_4k(left, top),
                Vector2D.from_4k(center_x, top),
                Vector2D.from_4k(right, top),
                Vector2D.from_4k(left, bottom),
                Vector2D.from_4k(center_x, bottom),
                Vector2D.from_4k(right, bottom),
            ]

        # Use detected corners for playing area if available
        playing_area_corners = detected_corners

        # Get default values from default table if available, otherwise use standard defaults
        default_pocket_radius = (
            self._default_table.pocket_radius if self._default_table else 72.0
        )
        default_cushion_elasticity = (
            self._default_table.cushion_elasticity if self._default_table else 0.8
        )
        default_surface_friction = (
            self._default_table.surface_friction if self._default_table else 0.2
        )
        default_surface_slope = (
            self._default_table.surface_slope if self._default_table else 0.0
        )
        default_cushion_height = (
            self._default_table.cushion_height if self._default_table else 48.0
        )

        # Create TableState with extracted data
        return TableState(
            width=table_width,
            height=table_height,
            pocket_positions=pocket_positions,
            pocket_radius=table_data.get("pocket_radius", default_pocket_radius),
            cushion_elasticity=table_data.get(
                "cushion_elasticity", default_cushion_elasticity
            ),
            surface_friction=table_data.get(
                "surface_friction", default_surface_friction
            ),
            surface_slope=table_data.get("surface_slope", default_surface_slope),
            cushion_height=table_data.get("cushion_height", default_cushion_height),
            playing_area_corners=playing_area_corners,
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
                            description=f"Ball {ball.id} moved {distance:.1f}mm",
                            data={
                                "ball_id": ball.id,
                                "distance": distance,
                                "from_position": asdict(last_pos),
                                "to_position": asdict(ball.position),
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
                self._game_rules = GameRules(game_type)
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

    def validate_shot(self, target_ball: BallState) -> bool:
        """Validate a shot based on the game rules."""
        if self._game_rules and self._current_state:
            return self._game_rules.validate_shot(self._current_state, target_ball)
        return True

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        if self._game_rules and self._current_state:
            return self._game_rules.check_game_over(self._current_state)
        return False

    def reset_game(
        self,
        game_type: GameType = GameType.PRACTICE,
        table_config: Optional[TableState] = None,
    ) -> None:
        """Reset game to initial state (FR-CORE-003)."""
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
                table=table_config or self._default_table,
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

            self._game_rules = GameRules(game_type)
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

    def _apply_corrections(self, state: GameState, corrections: dict) -> None:
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

    def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
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

            with self._lock:
                self._current_state = save_data["current_state"]
                self._state_history = deque(
                    save_data["state_history"], maxlen=self._state_history.maxlen
                )
                self._frame_counter = save_data["frame_counter"]
                self._start_time = save_data["start_time"]

                # Rebuild ball position tracking
                if self._current_state:
                    self._last_ball_positions = {
                        ball.id: ball.position for ball in self._current_state.balls
                    }

            self._event_manager.emit_event(
                "state_loaded",
                {"filepath": str(filepath), "frame_number": self._frame_counter},
            )

            logger.info(f"State loaded from {filepath}")

        except (pickle.PickleError, KeyError) as e:
            raise ValueError(f"Corrupted state file: {e}")

    def export_state_json(self, filepath: Path) -> None:
        """Export current state as JSON for debugging."""
        if not self._current_state:
            raise ValueError("No current state to export")

        # Convert dataclasses to dictionaries
        state_dict = asdict(self._current_state)

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

    def get_validation_statistics(self) -> dict:
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
