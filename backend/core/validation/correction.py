"""Error correction algorithms.

This module provides automated error correction for common issues detected
in billiards game states, including overlapping balls, out-of-bounds positions,
invalid velocities, and physics violations.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from backend.core.models import BallState, GameState, TableState, Vector2D


class CorrectionStrategy(Enum):
    """Error correction strategies."""

    GRADUAL = "gradual"  # Apply corrections gradually over time
    IMMEDIATE = "immediate"  # Apply corrections immediately
    CONFIDENCE_BASED = "confidence_based"  # Only correct obvious errors


class CorrectionType(Enum):
    """Types of corrections that can be applied."""

    BALL_OVERLAP = "ball_overlap"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_VELOCITY = "invalid_velocity"
    PHYSICS_VIOLATION = "physics_violation"
    STATE_INCONSISTENCY = "state_inconsistency"


@dataclass
class CorrectionRecord:
    """Record of a correction that was applied."""

    timestamp: float
    correction_type: CorrectionType
    ball_id: str
    description: str
    original_value: Any
    corrected_value: Any
    confidence: float = 1.0
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "correction_type": self.correction_type.value,
            "ball_id": self.ball_id,
            "description": self.description,
            "original_value": str(self.original_value),
            "corrected_value": str(self.corrected_value),
            "confidence": self.confidence,
            "severity": self.severity,
        }


@dataclass
class CorrectionStats:
    """Statistics about corrections applied."""

    total_corrections: int = 0
    corrections_by_type: dict[str, int] = field(default_factory=dict)
    corrections_by_severity: dict[str, int] = field(default_factory=dict)
    success_rate: float = 1.0
    average_confidence: float = 1.0
    last_correction_time: Optional[float] = None

    def add_correction(self, record: CorrectionRecord) -> None:
        """Add a correction to the statistics."""
        self.total_corrections += 1

        # Update by type
        correction_type = record.correction_type.value
        self.corrections_by_type[correction_type] = (
            self.corrections_by_type.get(correction_type, 0) + 1
        )

        # Update by severity
        self.corrections_by_severity[record.severity] = (
            self.corrections_by_severity.get(record.severity, 0) + 1
        )

        # Update averages
        if self.total_corrections > 0:
            old_avg = self.average_confidence
            self.average_confidence = (
                old_avg * (self.total_corrections - 1) + record.confidence
            ) / self.total_corrections

        self.last_correction_time = record.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_corrections": self.total_corrections,
            "corrections_by_type": self.corrections_by_type,
            "corrections_by_severity": self.corrections_by_severity,
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "last_correction_time": self.last_correction_time,
        }


class ErrorCorrector:
    """Game state error correction.

    Provides automated correction for common issues in billiards game states,
    including ball overlaps, out-of-bounds positions, invalid velocities,
    and physics violations.
    """

    def __init__(
        self,
        strategy: CorrectionStrategy = CorrectionStrategy.CONFIDENCE_BASED,
        max_velocity: float = 10.0,
        confidence_threshold: float = 0.5,
        enable_logging: bool = True,
        oscillation_prevention: bool = True,
    ):
        """Initialize the error corrector.

        Args:
            strategy: Correction strategy to use
            max_velocity: Maximum allowed velocity for balls
            confidence_threshold: Minimum confidence to apply corrections
            enable_logging: Whether to log corrections
            oscillation_prevention: Whether to prevent correction oscillations
        """
        self.strategy = strategy
        self.max_velocity = max_velocity
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self.oscillation_prevention = oscillation_prevention

        # Correction history and statistics
        self.correction_history: list[CorrectionRecord] = []
        self.stats = CorrectionStats()

        # Oscillation prevention
        self._recent_corrections: dict[str, list[float]] = {}  # ball_id -> timestamps
        self._oscillation_window = 5.0  # seconds
        self._max_corrections_per_window = 3

        # Setup logging
        if self.enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None

    def correct_errors(self, state: GameState, errors: list[str]) -> GameState:
        """Attempt to correct detected errors in the game state.

        Args:
            state: The game state to correct
            errors: List of error descriptions from validation

        Returns:
            Corrected game state
        """
        corrected_state = state.copy()
        corrections_applied = []

        # Parse errors and apply appropriate corrections
        for error in errors:
            try:
                corrections = self._parse_and_correct_error(corrected_state, error)
                corrections_applied.extend(corrections)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to correct error '{error}': {e}")

        # Update statistics
        for correction in corrections_applied:
            self.correction_history.append(correction)
            self.stats.add_correction(correction)

            if self.logger:
                self.logger.info(
                    f"Applied correction: {correction.description} "
                    f"(confidence: {correction.confidence:.2f})"
                )

        # Mark state as corrected and add correction info
        if corrections_applied:
            corrected_state.add_event(
                f"Applied {len(corrections_applied)} corrections",
                "correction",
                {"corrections": [c.to_dict() for c in corrections_applied]},
            )

        return corrected_state

    def _parse_and_correct_error(
        self, state: GameState, error: str
    ) -> list[CorrectionRecord]:
        """Parse error message and apply appropriate correction."""
        corrections = []

        if "overlapping" in error.lower():
            corrections.extend(self._correct_overlapping_balls_from_error(state, error))
        elif "outside table bounds" in error.lower():
            corrections.extend(self._correct_out_of_bounds_from_error(state, error))
        elif "velocity" in error.lower() and "exceeds" in error.lower():
            corrections.extend(self._correct_invalid_velocity_from_error(state, error))
        elif "duplicate ball" in error.lower():
            corrections.extend(self._correct_duplicate_ids(state))
        elif "expected 1 cue ball" in error.lower():
            corrections.extend(self._correct_cue_ball_count(state))

        return corrections

    def correct_overlapping_balls(self, balls: list[BallState]) -> list[BallState]:
        """Fix ball overlap issues by separating overlapping balls.

        Args:
            balls: List of balls that may be overlapping

        Returns:
            List of balls with overlaps resolved
        """
        corrected_balls = [ball.copy() for ball in balls]
        corrections = []

        # Find overlapping pairs
        for i in range(len(corrected_balls)):
            for j in range(i + 1, len(corrected_balls)):
                ball1, ball2 = corrected_balls[i], corrected_balls[j]

                if ball1.is_pocketed or ball2.is_pocketed:
                    continue

                if ball1.is_touching(ball2, tolerance=-0.001):  # Overlapping
                    if self._should_apply_correction(ball1.id):
                        self._separate_balls(ball1, ball2)

                        record = CorrectionRecord(
                            timestamp=datetime.now().timestamp(),
                            correction_type=CorrectionType.BALL_OVERLAP,
                            ball_id=f"{ball1.id},{ball2.id}",
                            description=f"Separated overlapping balls {ball1.id} and {ball2.id}",
                            original_value=f"distance: {ball1.distance_to(ball2):.4f}",
                            corrected_value=f"distance: {ball1.distance_to(ball2):.4f}",
                            confidence=min(ball1.confidence, ball2.confidence),
                            severity="medium",
                        )
                        corrections.append(record)

        # Store corrections for statistics
        for correction in corrections:
            self.correction_history.append(correction)
            self.stats.add_correction(correction)

        return corrected_balls

    def correct_out_of_bounds_balls(
        self, balls: list[BallState], table: TableState
    ) -> list[BallState]:
        """Move balls back onto the table if they're outside bounds.

        Args:
            balls: List of balls to check
            table: Table state for boundary information

        Returns:
            List of balls with positions corrected
        """
        corrected_balls = []
        corrections = []

        for ball in balls:
            corrected_ball = ball.copy()

            if ball.is_pocketed:
                corrected_balls.append(corrected_ball)
                continue

            if not table.is_point_on_table(ball.position, ball.radius):
                if self._should_apply_correction(ball.id):
                    original_pos = corrected_ball.position
                    corrected_ball.position = self._move_ball_to_table(
                        ball.position, ball.radius, table
                    )

                    record = CorrectionRecord(
                        timestamp=datetime.now().timestamp(),
                        correction_type=CorrectionType.OUT_OF_BOUNDS,
                        ball_id=ball.id,
                        description=f"Moved ball {ball.id} back onto table",
                        original_value=f"({original_pos.x:.3f}, {original_pos.y:.3f})",
                        corrected_value=f"({corrected_ball.position.x:.3f}, {corrected_ball.position.y:.3f})",
                        confidence=ball.confidence,
                        severity="high",
                    )
                    corrections.append(record)

            corrected_balls.append(corrected_ball)

        # Store corrections for statistics
        for correction in corrections:
            self.correction_history.append(correction)
            self.stats.add_correction(correction)

        return corrected_balls

    def correct_invalid_velocities(self, balls: list[BallState]) -> list[BallState]:
        """Fix unrealistic velocities in ball states.

        Args:
            balls: List of balls to check

        Returns:
            List of balls with velocities corrected
        """
        corrected_balls = []
        corrections = []

        for ball in balls:
            corrected_ball = ball.copy()
            original_velocity = corrected_ball.velocity
            velocity_magnitude = original_velocity.magnitude()

            # Check for invalid velocities
            needs_correction = False
            correction_reason = ""

            if math.isnan(velocity_magnitude) or math.isinf(velocity_magnitude):
                corrected_ball.velocity = Vector2D.zero()
                needs_correction = True
                correction_reason = "NaN/Inf velocity"
            elif velocity_magnitude > self.max_velocity:
                # Scale down to maximum allowed velocity
                direction = original_velocity.normalize()
                corrected_ball.velocity = direction * self.max_velocity
                needs_correction = True
                correction_reason = f"velocity exceeds maximum ({velocity_magnitude:.2f} > {self.max_velocity})"
            elif velocity_magnitude < 0:
                corrected_ball.velocity = Vector2D.zero()
                needs_correction = True
                correction_reason = "negative velocity magnitude"

            if needs_correction and self._should_apply_correction(ball.id):
                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.INVALID_VELOCITY,
                    ball_id=ball.id,
                    description=f"Corrected {correction_reason} for ball {ball.id}",
                    original_value=f"({original_velocity.x:.3f}, {original_velocity.y:.3f})",
                    corrected_value=f"({corrected_ball.velocity.x:.3f}, {corrected_ball.velocity.y:.3f})",
                    confidence=ball.confidence,
                    severity=(
                        "medium" if velocity_magnitude > self.max_velocity else "high"
                    ),
                )
                corrections.append(record)

            corrected_balls.append(corrected_ball)

        # Store corrections for statistics
        for correction in corrections:
            self.correction_history.append(correction)
            self.stats.add_correction(correction)

        return corrected_balls

    def correct_physics_violations(self, state: GameState) -> GameState:
        """Fix physics inconsistencies in the game state.

        Args:
            state: Game state to check and correct

        Returns:
            Corrected game state
        """
        corrected_state = state.copy()
        corrections = []

        for _i, ball in enumerate(corrected_state.balls):
            original_ball = ball.copy()

            # Fix invalid mass
            if ball.mass <= 0:
                ball.mass = 0.17  # Standard pool ball mass

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.PHYSICS_VIOLATION,
                    ball_id=ball.id,
                    description=f"Corrected invalid mass for ball {ball.id}",
                    original_value=str(original_ball.mass),
                    corrected_value=str(ball.mass),
                    confidence=0.9,  # High confidence in standard mass
                    severity="critical",
                )
                corrections.append(record)

            # Fix invalid radius
            if ball.radius <= 0:
                ball.radius = 0.028575  # Standard pool ball radius

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.PHYSICS_VIOLATION,
                    ball_id=ball.id,
                    description=f"Corrected invalid radius for ball {ball.id}",
                    original_value=str(original_ball.radius),
                    corrected_value=str(ball.radius),
                    confidence=0.9,  # High confidence in standard radius
                    severity="critical",
                )
                corrections.append(record)

            # Fix invalid confidence
            if not 0.0 <= ball.confidence <= 1.0:
                ball.confidence = max(0.0, min(1.0, ball.confidence))

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.PHYSICS_VIOLATION,
                    ball_id=ball.id,
                    description=f"Corrected invalid confidence for ball {ball.id}",
                    original_value=str(original_ball.confidence),
                    corrected_value=str(ball.confidence),
                    confidence=0.8,
                    severity="low",
                )
                corrections.append(record)

        # Store corrections for statistics
        for correction in corrections:
            self.correction_history.append(correction)
            self.stats.add_correction(correction)

        return corrected_state

    def _should_apply_correction(self, ball_id: str) -> bool:
        """Check if correction should be applied based on strategy and oscillation prevention."""
        current_time = datetime.now().timestamp()

        # Check oscillation prevention
        if self.oscillation_prevention:
            if ball_id not in self._recent_corrections:
                self._recent_corrections[ball_id] = []

            recent_times = self._recent_corrections[ball_id]
            # Remove old corrections outside the window
            recent_times = [
                t for t in recent_times if current_time - t <= self._oscillation_window
            ]
            self._recent_corrections[ball_id] = recent_times

            if len(recent_times) >= self._max_corrections_per_window:
                if self.logger:
                    self.logger.warning(
                        f"Skipping correction for {ball_id} due to oscillation prevention "
                        f"({len(recent_times)} corrections in {self._oscillation_window}s)"
                    )
                return False

            # Record this correction attempt
            self._recent_corrections[ball_id].append(current_time)

        # Apply correction based on strategy
        if self.strategy == CorrectionStrategy.IMMEDIATE:
            return True
        elif self.strategy == CorrectionStrategy.GRADUAL:
            # Apply corrections gradually (could implement more sophisticated logic)
            return True
        elif self.strategy == CorrectionStrategy.CONFIDENCE_BASED:
            # Only apply if we have sufficient confidence (implementation specific)
            return True

        return True

    def _separate_balls(self, ball1: BallState, ball2: BallState) -> None:
        """Separate two overlapping balls."""
        # Calculate the direction vector from ball1 to ball2
        direction = ball2.position - ball1.position
        distance = direction.magnitude()

        if distance == 0:
            # Balls are at exactly the same position, use arbitrary direction
            direction = Vector2D(1.0, 0.0)
            distance = 0.0
        else:
            direction = direction.normalize()

        # Calculate minimum separation distance
        min_distance = ball1.radius + ball2.radius + 0.001  # Small buffer

        # Calculate how much to move each ball
        overlap = min_distance - distance
        move_distance = overlap / 2.0

        # Move balls apart
        ball1.position = ball1.position - (direction * move_distance)
        ball2.position = ball2.position + (direction * move_distance)

    def _move_ball_to_table(
        self, position: Vector2D, radius: float, table: TableState
    ) -> Vector2D:
        """Move a ball position to be within table bounds."""
        # Clamp position to table bounds with ball radius consideration
        margin = radius + 0.001  # Small buffer

        x = max(margin, min(table.width - margin, position.x))
        y = max(margin, min(table.height - margin, position.y))

        return Vector2D(x, y)

    def _correct_overlapping_balls_from_error(
        self, state: GameState, error: str
    ) -> list[CorrectionRecord]:
        """Parse overlapping ball error and correct it."""
        corrections = []

        # Extract ball IDs from error message if possible
        # Format: "Balls ball1_id and ball2_id are overlapping"
        words = error.split()
        ball_ids = []

        for i, word in enumerate(words):
            if word == "Balls" and i + 3 < len(words) and words[i + 2] == "and":
                ball_ids = [words[i + 1], words[i + 3]]
                break

        if len(ball_ids) == 2:
            # Find the specific balls
            ball1 = state.get_ball_by_id(ball_ids[0])
            ball2 = state.get_ball_by_id(ball_ids[1])

            if ball1 and ball2:
                if self._should_apply_correction(ball1.id):
                    original_distance = ball1.distance_to(ball2)
                    self._separate_balls(ball1, ball2)
                    new_distance = ball1.distance_to(ball2)

                    record = CorrectionRecord(
                        timestamp=datetime.now().timestamp(),
                        correction_type=CorrectionType.BALL_OVERLAP,
                        ball_id=f"{ball1.id},{ball2.id}",
                        description=f"Separated overlapping balls {ball1.id} and {ball2.id}",
                        original_value=f"distance: {original_distance:.4f}",
                        corrected_value=f"distance: {new_distance:.4f}",
                        confidence=min(ball1.confidence, ball2.confidence),
                        severity="medium",
                    )
                    corrections.append(record)
        else:
            # General overlap correction - check all balls
            state.balls = self.correct_overlapping_balls(state.balls)

        return corrections

    def _correct_out_of_bounds_from_error(
        self, state: GameState, error: str
    ) -> list[CorrectionRecord]:
        """Parse out of bounds error and correct it."""
        corrections = []

        # Extract ball ID from error message
        # Format: "Ball ball_id is outside table bounds"
        words = error.split()
        ball_id = None

        for i, word in enumerate(words):
            if word == "Ball" and i + 1 < len(words):
                ball_id = words[i + 1]
                break

        if ball_id:
            ball = state.get_ball_by_id(ball_id)
            if ball and self._should_apply_correction(ball_id):
                original_pos = ball.position
                ball.position = self._move_ball_to_table(
                    ball.position, ball.radius, state.table
                )

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.OUT_OF_BOUNDS,
                    ball_id=ball_id,
                    description=f"Moved ball {ball_id} back onto table",
                    original_value=f"({original_pos.x:.3f}, {original_pos.y:.3f})",
                    corrected_value=f"({ball.position.x:.3f}, {ball.position.y:.3f})",
                    confidence=ball.confidence,
                    severity="high",
                )
                corrections.append(record)

        return corrections

    def _correct_invalid_velocity_from_error(
        self, state: GameState, error: str
    ) -> list[CorrectionRecord]:
        """Parse velocity error and correct it."""
        corrections = []

        # Extract ball ID from error message
        # Format: "Ball ball_id velocity X.XX exceeds maximum Y.YY"
        words = error.split()
        ball_id = None

        for i, word in enumerate(words):
            if word == "Ball" and i + 1 < len(words):
                ball_id = words[i + 1]
                break

        if ball_id:
            ball = state.get_ball_by_id(ball_id)
            if ball and self._should_apply_correction(ball_id):
                original_velocity = ball.velocity
                corrected_balls = self.correct_invalid_velocities([ball])
                if corrected_balls:
                    ball.velocity = corrected_balls[0].velocity

                    record = CorrectionRecord(
                        timestamp=datetime.now().timestamp(),
                        correction_type=CorrectionType.INVALID_VELOCITY,
                        ball_id=ball_id,
                        description=f"Corrected excessive velocity for ball {ball_id}",
                        original_value=f"({original_velocity.x:.3f}, {original_velocity.y:.3f})",
                        corrected_value=f"({ball.velocity.x:.3f}, {ball.velocity.y:.3f})",
                        confidence=ball.confidence,
                        severity="medium",
                    )
                    corrections.append(record)

        return corrections

    def _correct_duplicate_ids(self, state: GameState) -> list[CorrectionRecord]:
        """Correct duplicate ball IDs."""
        corrections = []
        seen_ids = set()

        for i, ball in enumerate(state.balls):
            if ball.id in seen_ids:
                original_id = ball.id
                ball.id = f"{ball.id}_dup_{i}"

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.STATE_INCONSISTENCY,
                    ball_id=ball.id,
                    description="Corrected duplicate ball ID",
                    original_value=original_id,
                    corrected_value=ball.id,
                    confidence=0.9,
                    severity="high",
                )
                corrections.append(record)

            seen_ids.add(ball.id)

        return corrections

    def _correct_cue_ball_count(self, state: GameState) -> list[CorrectionRecord]:
        """Correct cue ball count issues."""
        corrections = []
        cue_balls = [
            ball for ball in state.balls if ball.is_cue_ball and not ball.is_pocketed
        ]

        if len(cue_balls) == 0:
            # No cue ball - designate one (prefer ball with "cue" in ID)
            for ball in state.balls:
                if not ball.is_pocketed and "cue" in ball.id.lower():
                    ball.is_cue_ball = True

                    record = CorrectionRecord(
                        timestamp=datetime.now().timestamp(),
                        correction_type=CorrectionType.STATE_INCONSISTENCY,
                        ball_id=ball.id,
                        description=f"Designated ball {ball.id} as cue ball",
                        original_value="is_cue_ball: False",
                        corrected_value="is_cue_ball: True",
                        confidence=0.8,
                        severity="high",
                    )
                    corrections.append(record)
                    break

        elif len(cue_balls) > 1:
            # Multiple cue balls - keep only the first one
            for _i, ball in enumerate(cue_balls[1:], 1):
                ball.is_cue_ball = False

                record = CorrectionRecord(
                    timestamp=datetime.now().timestamp(),
                    correction_type=CorrectionType.STATE_INCONSISTENCY,
                    ball_id=ball.id,
                    description=f"Removed extra cue ball designation from {ball.id}",
                    original_value="is_cue_ball: True",
                    corrected_value="is_cue_ball: False",
                    confidence=0.9,
                    severity="medium",
                )
                corrections.append(record)

        return corrections

    def get_correction_statistics(self) -> dict[str, Any]:
        """Get comprehensive correction statistics."""
        return {
            "stats": self.stats.to_dict(),
            "total_history_records": len(self.correction_history),
            "recent_corrections": [
                record.to_dict() for record in self.correction_history[-10:]
            ],  # Last 10 corrections
            "correction_frequency": self._calculate_correction_frequency(),
            "oscillation_status": self._get_oscillation_status(),
        }

    def _calculate_correction_frequency(self) -> dict[str, float]:
        """Calculate correction frequency statistics."""
        if not self.correction_history:
            return {"per_minute": 0.0, "per_hour": 0.0}

        now = datetime.now().timestamp()
        recent_corrections = [
            record
            for record in self.correction_history
            if now - record.timestamp <= 3600  # Last hour
        ]

        corrections_per_hour = len(recent_corrections)
        corrections_per_minute = corrections_per_hour / 60.0

        return {
            "per_minute": corrections_per_minute,
            "per_hour": corrections_per_hour,
        }

    def _get_oscillation_status(self) -> dict[str, Any]:
        """Get oscillation prevention status."""
        current_time = datetime.now().timestamp()
        active_windows = {}

        for ball_id, timestamps in self._recent_corrections.items():
            recent_count = len(
                [t for t in timestamps if current_time - t <= self._oscillation_window]
            )
            if recent_count > 0:
                active_windows[ball_id] = {
                    "recent_corrections": recent_count,
                    "max_allowed": self._max_corrections_per_window,
                    "is_limited": recent_count >= self._max_corrections_per_window,
                }

        return {
            "oscillation_prevention_enabled": self.oscillation_prevention,
            "window_duration": self._oscillation_window,
            "max_corrections_per_window": self._max_corrections_per_window,
            "active_windows": active_windows,
        }

    def reset_statistics(self) -> None:
        """Reset all correction statistics and history."""
        self.correction_history.clear()
        self.stats = CorrectionStats()
        self._recent_corrections.clear()

        if self.logger:
            self.logger.info("Reset correction statistics and history")

    def set_strategy(self, strategy: CorrectionStrategy) -> None:
        """Change the correction strategy."""
        old_strategy = self.strategy
        self.strategy = strategy

        if self.logger:
            self.logger.info(
                f"Changed correction strategy from {old_strategy.value} to {strategy.value}"
            )
