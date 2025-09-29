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

from .events.manager import EventManager
from .models import (
    BallState,
    CueState,
    GameEvent,
    GameState,
    GameType,
    TableState,
    Vector2D,
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
        self, max_history_frames: int = 1000, persistence_path: Optional[Path] = None
    ):
        """Initialize game state manager.

        Args:
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

        # Default table configuration (standard 9-foot table)
        self._default_table = self._create_default_table()

        # Event detection state
        self._last_ball_positions: dict[str, Vector2D] = {}
        self._motion_threshold = 0.005  # meters/frame threshold for motion detection

        self._game_rules: Optional[GameRules] = None

        logger.info("GameStateManager initialized")

    def _create_default_table(self) -> TableState:
        """Create default table configuration for 9-foot pool table."""
        return TableState.standard_9ft_table()

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

            # Get table state (typically static unless recalibrated)
            table = detection_data.get("table", self._default_table)

            # Detect events based on state changes
            events = self._detect_events(balls)

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
            )

            # Validate state
            if self._validation_enabled:
                self._validate_state(new_state)

            # Update state and history
            self._state_history.append(self._current_state)
            self._current_state = new_state

            # Update ball position tracking for event detection
            self._last_ball_positions = {ball.id: ball.position for ball in balls}

            # Emit state change event
            self._event_manager.emit_event(
                "state_updated",
                {
                    "frame_number": self._frame_counter,
                    "timestamp": timestamp,
                    "balls_count": len(balls),
                    "events": [asdict(event) for event in events],
                },
            )

            logger.debug(f"State updated for frame {self._frame_counter}")
            return new_state

    def _extract_ball_states(self, detection_data: dict[str, Any]) -> list[BallState]:
        """Extract ball states from detection data."""
        balls = []
        ball_detections = detection_data.get("balls", [])

        for detection in ball_detections:
            ball = BallState(
                id=detection.get("id", f"ball_{len(balls)}"),
                position=Vector2D(detection["x"], detection["y"]),
                velocity=Vector2D(detection.get("vx", 0.0), detection.get("vy", 0.0)),
                radius=detection.get(
                    "radius", 0.028575
                ),  # Standard ball radius in meters
                mass=0.17,  # Standard ball mass in kg
                spin=(
                    Vector2D(detection.get("spin_x", 0.0), detection.get("spin_y", 0.0))
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

        return CueState(
            tip_position=Vector2D(cue_data["tip_x"], cue_data["tip_y"]),
            angle=cue_data.get("angle", 0.0),
            elevation=cue_data.get("elevation", 0.0),
            estimated_force=cue_data.get("force", 0.0),
            impact_point=(
                Vector2D(cue_data["impact_x"], cue_data["impact_y"])
                if "impact_x" in cue_data
                else None
            ),
            is_visible=cue_data.get("is_visible", True),
            confidence=cue_data.get("confidence", 1.0),
            last_update=cue_data.get("timestamp", time.time()),
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
                logger.warning(error_msg)

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
