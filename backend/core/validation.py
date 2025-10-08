"""State validation module stub.

Minimal implementation to satisfy imports. Full implementation TBD.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Result of state validation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_values: dict = field(default_factory=dict)
    confidence: float = 1.0

    def add_error(self, error: str) -> None:
        """Add an error message.

        Args:
            error: Error message to add
        """
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message.

        Args:
            warning: Warning message to add
        """
        self.warnings.append(warning)


class StateValidator:
    """State validator - stub implementation."""

    def __init__(self, enable_auto_correction: bool = False):
        """Initialize validator.

        Args:
            enable_auto_correction: Whether to auto-correct invalid states
        """
        self.enable_auto_correction = enable_auto_correction

    def validate_game_state(self, game_state) -> ValidationResult:
        """Validate game state.

        Args:
            game_state: Game state to validate

        Returns:
            Validation result (stub always returns valid)
        """
        return ValidationResult(is_valid=True)

    def validate_physics(
        self, current_state, previous_state: Optional = None, dt: float = 1 / 30
    ) -> ValidationResult:
        """Validate physics between states.

        Args:
            current_state: Current game state
            previous_state: Previous game state
            dt: Time delta in seconds

        Returns:
            Validation result (stub always returns valid)
        """
        return ValidationResult(is_valid=True)

    def get_validation_statistics(self) -> dict:
        """Get validation statistics.

        Returns:
            Statistics dict (stub returns empty dict)
        """
        return {}
