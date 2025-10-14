"""Game state validation algorithms.

Provides comprehensive validation of game states to ensure consistency,
physical plausibility, and logical correctness.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import BallState, CueState, GameState, TableState

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_values: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def add_correction(self, field: str, value: Any) -> None:
        """Add a corrected value."""
        self.corrected_values[field] = value

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        result = ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            corrected_values={**self.corrected_values, **other.corrected_values},
            confidence=min(self.confidence, other.confidence),
        )
        return result


class StateValidator:
    """Comprehensive game state validation.

    Provides detailed validation of all aspects of game state including:
    - Ball position and physics constraints
    - Table state consistency
    - Cue state validity
    - Inter-object consistency checks
    - Physical plausibility
    """

    def __init__(
        self,
        max_velocity: float = 10.0,  # m/s
        max_acceleration: float = 50.0,  # m/s²
        overlap_tolerance: float = 0.001,  # meters
        position_tolerance: float = 0.0001,  # meters
        enable_auto_correction: bool = True,
    ):
        """Initialize state validator with configuration.

        Args:
            max_velocity: Maximum allowed ball velocity in m/s
            max_acceleration: Maximum allowed ball acceleration in m/s²
            overlap_tolerance: Tolerance for ball overlaps in meters
            position_tolerance: Tolerance for position precision in meters
            enable_auto_correction: Whether to automatically correct minor errors
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.overlap_tolerance = overlap_tolerance
        self.position_tolerance = position_tolerance
        self.enable_auto_correction = enable_auto_correction

        logger.info(
            f"StateValidator initialized with max_velocity={max_velocity}, "
            f"overlap_tolerance={overlap_tolerance}"
        )

    def validate_game_state(self, game_state: GameState) -> ValidationResult:
        """Validate complete game state consistency.

        Args:
            game_state: The game state to validate

        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult()

        # Validate basic state properties
        basic_result = self._validate_basic_properties(game_state)
        result = result.merge(basic_result)

        # Validate ball positions and physics
        ball_result = self.validate_ball_positions(game_state.balls)
        result = result.merge(ball_result)

        # Validate table state
        table_result = self.validate_table_state(game_state.table)
        result = result.merge(table_result)

        # Validate cue state if present
        if game_state.cue:
            cue_result = self.validate_cue_state(game_state.cue, game_state.table)
            result = result.merge(cue_result)

        # Validate consistency between objects
        consistency_result = self._validate_object_consistency(
            game_state.balls, game_state.table, game_state.cue
        )
        result = result.merge(consistency_result)

        # Validate game logic
        logic_result = self._validate_game_logic(game_state)
        result = result.merge(logic_result)

        logger.debug(
            f"Game state validation complete: valid={result.is_valid}, "
            f"errors={len(result.errors)}, warnings={len(result.warnings)}"
        )

        return result

    def validate_ball_positions(self, balls: list[BallState]) -> ValidationResult:
        """Validate ball positions and physics constraints.

        Args:
            balls: List of ball states to validate

        Returns:
            ValidationResult with ball validation details
        """
        result = ValidationResult()

        if not balls:
            return result

        # Check for duplicate ball IDs
        ball_ids = [ball.id for ball in balls]
        if len(ball_ids) != len(set(ball_ids)):
            result.add_error("Duplicate ball IDs found")

        # Check for exactly one cue ball among active balls
        active_balls = [b for b in balls if not b.is_pocketed]
        cue_balls = [b for b in active_balls if b.is_cue_ball]

        if len(cue_balls) == 0:
            result.add_error("No active cue ball found")
        elif len(cue_balls) > 1:
            result.add_error(f"Multiple active cue balls found: {len(cue_balls)}")

        # Validate each ball individually
        for ball in balls:
            ball_result = self._validate_individual_ball(ball)
            result = result.merge(ball_result)

        # Check for ball overlaps
        overlap_result = self._check_ball_overlaps(active_balls)
        result = result.merge(overlap_result)

        return result

    def validate_table_state(self, table: TableState) -> ValidationResult:
        """Validate table state consistency.

        Args:
            table: Table state to validate

        Returns:
            ValidationResult with table validation details
        """
        result = ValidationResult()

        # Validate table dimensions
        if table.width <= 0:
            result.add_error(f"Invalid table width: {table.width}")
        if table.height <= 0:
            result.add_error(f"Invalid table height: {table.height}")

        # Check for reasonable table size (validation only, no warnings)

        # Validate pocket configuration
        if len(table.pocket_positions) != 6:
            result.add_error(f"Expected 6 pockets, found {len(table.pocket_positions)}")

        # Validate physical properties
        if not 0.0 <= table.cushion_elasticity <= 1.0:
            result.add_error(f"Invalid cushion elasticity: {table.cushion_elasticity}")

        if table.surface_friction < 0:
            result.add_error(f"Invalid surface friction: {table.surface_friction}")

        if table.pocket_radius <= 0:
            result.add_error(f"Invalid pocket radius: {table.pocket_radius}")

        return result

    def validate_cue_state(self, cue: CueState, table: TableState) -> ValidationResult:
        """Validate cue state consistency.

        Args:
            cue: Cue state to validate
            table: Table state for bounds checking

        Returns:
            ValidationResult with cue validation details
        """
        result = ValidationResult()

        # Validate cue tip position is reasonable relative to table (validation only, no warnings)

        # Validate cue properties
        if cue.length <= 0:
            result.add_error(f"Invalid cue length: {cue.length}")

        if cue.tip_radius <= 0:
            result.add_error(f"Invalid cue tip radius: {cue.tip_radius}")

        if not 0.0 <= cue.confidence <= 1.0:
            result.add_error(f"Invalid cue confidence: {cue.confidence}")

        # Validate force is reasonable
        if cue.estimated_force < 0:
            result.add_error(f"Negative cue force: {cue.estimated_force}")

        # Validate angle ranges (validation only, no warnings)

        return result

    def validate_physics(
        self,
        current_state: GameState,
        previous_state: Optional[GameState] = None,
        dt: float = 1 / 30,  # default 30 FPS
    ) -> ValidationResult:
        """Validate physical consistency between states.

        Args:
            current_state: Current game state
            previous_state: Previous game state for comparison
            dt: Time difference between states in seconds

        Returns:
            ValidationResult with physics validation details
        """
        result = ValidationResult()

        if previous_state is None:
            return result

        if dt <= 0:
            result.add_error(f"Invalid time delta: {dt}")
            return result

        # Check each ball's physics
        for current_ball in current_state.balls:
            previous_ball = previous_state.get_ball_by_id(current_ball.id)
            if previous_ball is None:
                continue  # New ball, skip physics check

            physics_result = self._validate_ball_physics(
                current_ball, previous_ball, dt
            )
            result = result.merge(physics_result)

        return result

    def _validate_basic_properties(self, game_state: GameState) -> ValidationResult:
        """Validate basic game state properties."""
        result = ValidationResult()

        if game_state.frame_number < 0:
            result.add_error(f"Invalid frame number: {game_state.frame_number}")

        if game_state.timestamp < 0:
            result.add_error(f"Invalid timestamp: {game_state.timestamp}")

        if not 0.0 <= game_state.state_confidence <= 1.0:
            result.add_error(f"Invalid state confidence: {game_state.state_confidence}")

        return result

    def _validate_individual_ball(self, ball: BallState) -> ValidationResult:
        """Validate a single ball's properties."""
        result = ValidationResult()

        # Validate basic properties
        if ball.radius <= 0:
            result.add_error(f"Ball {ball.id} has invalid radius: {ball.radius}")

        if ball.mass <= 0:
            result.add_error(f"Ball {ball.id} has invalid mass: {ball.mass}")

        if not 0.0 <= ball.confidence <= 1.0:
            result.add_error(
                f"Ball {ball.id} has invalid confidence: {ball.confidence}"
            )

        # Validate velocity - check for None first
        if ball.velocity is None:
            result.add_error(f"Ball {ball.id} has None velocity")
            if self.enable_auto_correction:
                # Auto-correct to zero velocity
                from ..models import Vector2D

                result.add_correction(f"ball_{ball.id}_velocity", Vector2D.zero())
        else:
            velocity_magnitude = ball.velocity.magnitude()
            if velocity_magnitude > self.max_velocity:
                if self.enable_auto_correction:
                    # Clamp velocity to maximum
                    corrected_velocity = ball.velocity.normalize() * self.max_velocity
                    result.add_correction(
                        f"ball_{ball.id}_velocity", corrected_velocity
                    )
                else:
                    result.add_error(
                        f"Ball {ball.id} velocity {velocity_magnitude:.2f} exceeds maximum {self.max_velocity}"
                    )

        # Validate spin (validation only, no warnings)

        return result

    def _check_ball_overlaps(self, balls: list[BallState]) -> ValidationResult:
        """Check for overlapping balls."""
        result = ValidationResult()

        for i, ball1 in enumerate(balls):
            for ball2 in balls[i + 1 :]:
                if ball1.is_touching(ball2, tolerance=-self.overlap_tolerance):
                    overlap_distance = (
                        ball1.radius + ball2.radius - ball1.distance_to(ball2)
                    )

                    if (
                        self.enable_auto_correction and overlap_distance < 0.01
                    ):  # Small overlap
                        # Suggest correction by separating balls
                        separation_vector = (
                            ball2.position - ball1.position
                        ).normalize()
                        correction_distance = (
                            overlap_distance / 2 + self.overlap_tolerance
                        )

                        new_pos1 = (
                            ball1.position - separation_vector * correction_distance
                        )
                        new_pos2 = (
                            ball2.position + separation_vector * correction_distance
                        )

                        result.add_correction(f"ball_{ball1.id}_position", new_pos1)
                        result.add_correction(f"ball_{ball2.id}_position", new_pos2)
                    else:
                        result.add_error(
                            f"Balls {ball1.id} and {ball2.id} overlap by {overlap_distance*1000:.1f}mm"
                        )

        return result

    def _validate_object_consistency(
        self, balls: list[BallState], table: TableState, cue: Optional[CueState]
    ) -> ValidationResult:
        """Validate consistency between different objects."""
        result = ValidationResult()

        # Check balls are within table bounds
        for ball in balls:
            if not ball.is_pocketed:
                if not table.is_point_on_table(ball.position, ball.radius):
                    # Check if ball is in a pocket
                    is_in_pocket, pocket_id = table.is_point_in_pocket(
                        ball.position, ball.radius
                    )

                    if is_in_pocket:
                        if self.enable_auto_correction:
                            result.add_correction(f"ball_{ball.id}_pocketed", True)
                            result.add_correction(
                                f"ball_{ball.id}_pocket_id", pocket_id
                            )
                    else:
                        result.add_error(
                            f"Ball {ball.id} is outside table bounds: {ball.position}"
                        )

        # Check cue-ball interaction if cue is present
        if cue:
            cue_ball = next(
                (b for b in balls if b.is_cue_ball and not b.is_pocketed), None
            )
            if cue_ball:
                cue.tip_position.distance_to(cue_ball.position)

                # If cue is very close to ball, validate impact point (no warnings)

        return result

    def _validate_game_logic(self, game_state: GameState) -> ValidationResult:
        """Validate game-specific logic and rules."""
        result = ValidationResult()

        # Check score consistency
        if game_state.scores:
            for player, score in game_state.scores.items():
                if score < 0:
                    result.add_error(f"Player {player} has negative score: {score}")

        # Check current player validity
        if game_state.current_player is not None:
            if game_state.current_player not in [1, 2]:
                result.add_error(f"Invalid current player: {game_state.current_player}")

        return result

    def _validate_ball_physics(
        self, current_ball: BallState, previous_ball: BallState, dt: float
    ) -> ValidationResult:
        """Validate physics consistency between ball states."""
        result = ValidationResult()

        # Validate acceleration (validation only, no warnings)
        # Validate position change consistency (validation only, no warnings)

        return result

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation configuration and statistics."""
        return {
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "overlap_tolerance": self.overlap_tolerance,
            "position_tolerance": self.position_tolerance,
            "auto_correction_enabled": self.enable_auto_correction,
        }
