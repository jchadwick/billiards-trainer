"""Validation module for game state and physics consistency.

This module provides comprehensive validation capabilities for the billiards
trainer system, ensuring data integrity and physical plausibility with automatic
error correction and system reliability improvements.
"""

from .correction import (
    CorrectionRecord,
    CorrectionStats,
    CorrectionStrategy,
    CorrectionType,
    ErrorCorrector,
)

# from .manager import ValidationManager, ValidationReport, validate_game_state
from .physics import PhysicsValidator, ValidationError
from .state import StateValidator, ValidationResult

__all__ = [
    "StateValidator",
    "ValidationResult",
    "PhysicsValidator",
    "ValidationError",
    "ErrorCorrector",
    "CorrectionStrategy",
    "CorrectionType",
    "CorrectionRecord",
    "CorrectionStats",
    # "ValidationManager",
    # "ValidationReport",
    # "validate_game_state",
]
