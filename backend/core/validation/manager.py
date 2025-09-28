"""Integrated validation and correction manager.

This module provides a unified interface for validation and automatic error
correction, combining state validation, physics validation, and error correction
into a single coordinated system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from backend.core.models import GameState

from .correction import CorrectionStrategy, ErrorCorrector
from .physics import PhysicsValidator
from .physics import ValidationError as PhysicsValidationError
from .state import StateValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Comprehensive validation and correction report."""

    timestamp: float
    is_valid: bool
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)
    physics_errors: list[PhysicsValidationError] = field(default_factory=list)
    corrections_applied: list[dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 1.0
    processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
            "validation_results": {
                name: {
                    "is_valid": result.is_valid,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "confidence": result.confidence,
                }
                for name, result in self.validation_results.items()
            },
            "physics_errors": [
                {
                    "error_type": error.error_type,
                    "severity": error.severity,
                    "message": error.message,
                    "details": error.details,
                }
                for error in self.physics_errors
            ],
            "corrections_applied": self.corrections_applied,
            "overall_confidence": self.overall_confidence,
            "processing_time": self.processing_time,
        }


class ValidationManager:
    """Integrated validation and correction manager.

    Coordinates state validation, physics validation, and automatic error
    correction to ensure system reliability and data consistency.
    """

    def __init__(
        self,
        auto_correct: bool = True,
        correction_strategy: CorrectionStrategy = CorrectionStrategy.CONFIDENCE_BASED,
        max_correction_attempts: int = 3,
        enable_physics_validation: bool = True,
        enable_detailed_logging: bool = True,
    ):
        """Initialize the validation manager.

        Args:
            auto_correct: Whether to automatically apply corrections
            correction_strategy: Strategy for applying corrections
            max_correction_attempts: Maximum attempts to correct errors
            enable_physics_validation: Whether to perform physics validation
            enable_detailed_logging: Whether to enable detailed logging
        """
        self.auto_correct = auto_correct
        self.max_correction_attempts = max_correction_attempts
        self.enable_physics_validation = enable_physics_validation
        self.enable_detailed_logging = enable_detailed_logging

        # Initialize validators and corrector
        self.state_validator = StateValidator(
            enable_auto_correction=False
        )  # We handle corrections
        self.physics_validator = (
            PhysicsValidator() if enable_physics_validation else None
        )
        self.error_corrector = ErrorCorrector(
            strategy=correction_strategy, enable_logging=enable_detailed_logging
        )

        # Statistics
        self.validation_count = 0
        self.correction_count = 0
        self.total_processing_time = 0.0

        logger.info(
            f"ValidationManager initialized: auto_correct={auto_correct}, "
            f"strategy={correction_strategy.value}, physics={enable_physics_validation}"
        )

    def validate_and_correct(
        self,
        game_state: GameState,
        previous_state: Optional[GameState] = None,
        force_validation: bool = False,
    ) -> tuple[GameState, ValidationReport]:
        """Perform comprehensive validation and correction.

        Args:
            game_state: Current game state to validate
            previous_state: Previous state for physics validation
            force_validation: Force full validation even if confidence is high

        Returns:
            Tuple of (corrected_state, validation_report)
        """
        start_time = datetime.now().timestamp()
        self.validation_count += 1

        report = ValidationReport(timestamp=start_time, is_valid=True)

        corrected_state = game_state.copy()

        try:
            # Step 1: Basic state validation
            state_result = self.state_validator.validate_game_state(corrected_state)
            report.validation_results["state"] = state_result

            if not state_result.is_valid:
                report.is_valid = False

                if self.auto_correct:
                    # Apply corrections for state errors
                    corrected_state = self._apply_state_corrections(
                        corrected_state, state_result, report
                    )

            # Step 2: Physics validation (if enabled and previous state available)
            if self.physics_validator and previous_state:
                physics_result = self.state_validator.validate_physics(
                    corrected_state, previous_state
                )
                report.validation_results["physics"] = physics_result

                if not physics_result.is_valid:
                    report.is_valid = False

                    if self.auto_correct:
                        # Apply corrections for physics errors
                        corrected_state = self._apply_physics_corrections(
                            corrected_state, physics_result, report
                        )

            # Step 3: Additional physics validation with specialized validator
            if self.physics_validator:
                # Validate system state
                system_validation = self.physics_validator.validate_system_state(
                    corrected_state.balls, corrected_state.table
                )

                if not system_validation.is_valid:
                    report.physics_errors.extend(system_validation.errors)
                    report.is_valid = False

                    if self.auto_correct:
                        corrected_state = self._apply_physics_validator_corrections(
                            corrected_state, system_validation, report
                        )

            # Step 4: Final validation pass if corrections were applied
            if report.corrections_applied and self.auto_correct:
                final_validation = self.state_validator.validate_game_state(
                    corrected_state
                )
                report.validation_results["final"] = final_validation
                report.is_valid = final_validation.is_valid

            # Calculate overall confidence
            report.overall_confidence = self._calculate_overall_confidence(report)

        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            report.is_valid = False
            report.overall_confidence = 0.0

        # Record processing time
        end_time = datetime.now().timestamp()
        report.processing_time = end_time - start_time
        self.total_processing_time += report.processing_time

        if self.enable_detailed_logging:
            logger.info(
                f"Validation complete: valid={report.is_valid}, "
                f"corrections={len(report.corrections_applied)}, "
                f"time={report.processing_time:.3f}s"
            )

        return corrected_state, report

    def _apply_state_corrections(
        self,
        state: GameState,
        validation_result: ValidationResult,
        report: ValidationReport,
    ) -> GameState:
        """Apply corrections based on state validation results."""
        all_errors = validation_result.errors + validation_result.warnings

        if not all_errors:
            return state

        attempt = 0
        current_state = state

        while attempt < self.max_correction_attempts and all_errors:
            corrected_state = self.error_corrector.correct_errors(
                current_state, all_errors
            )

            # Record corrections in report
            correction_stats = self.error_corrector.get_correction_statistics()
            if correction_stats["recent_corrections"]:
                report.corrections_applied.extend(
                    correction_stats["recent_corrections"]
                )
                self.correction_count += len(correction_stats["recent_corrections"])

            # Check if corrections were effective
            revalidation = self.state_validator.validate_game_state(corrected_state)
            new_errors = revalidation.errors + revalidation.warnings

            if len(new_errors) >= len(all_errors):
                # No improvement, stop trying
                logger.warning(f"Corrections ineffective after attempt {attempt + 1}")
                break

            current_state = corrected_state
            all_errors = new_errors
            attempt += 1

        return current_state

    def _apply_physics_corrections(
        self,
        state: GameState,
        validation_result: ValidationResult,
        report: ValidationReport,
    ) -> GameState:
        """Apply corrections based on physics validation results."""
        # Convert physics validation errors to string format for error corrector
        physics_errors = []

        for error in validation_result.errors:
            if (
                "velocity" in error.lower()
                and "exceeds" in error.lower()
                or "acceleration" in error.lower()
            ):
                physics_errors.append(error)
            elif "position change inconsistent" in error.lower():
                # This might require position correction
                physics_errors.append(error)

        if physics_errors:
            corrected_state = self.error_corrector.correct_errors(state, physics_errors)

            # Record corrections
            correction_stats = self.error_corrector.get_correction_statistics()
            if correction_stats["recent_corrections"]:
                report.corrections_applied.extend(
                    correction_stats["recent_corrections"]
                )
                self.correction_count += len(correction_stats["recent_corrections"])

            return corrected_state

        return state

    def _apply_physics_validator_corrections(
        self, state: GameState, validation_result, report: ValidationReport
    ) -> GameState:
        """Apply corrections based on physics validator results."""
        # Convert physics validation errors to correctable format
        correctable_errors = []

        for error in validation_result.errors:
            if error.error_type == "ball_overlap":
                correctable_errors.append(
                    f"Balls {error.details.get('ball1_id', 'unknown')} and {error.details.get('ball2_id', 'unknown')} are overlapping"
                )
            elif error.error_type == "position_out_of_bounds":
                correctable_errors.append(
                    f"Ball {error.details.get('ball_id', 'unknown')} is outside table bounds"
                )
            elif error.error_type == "velocity_limit":
                correctable_errors.append(
                    f"Ball {error.details.get('ball_id', 'unknown')} velocity {error.details.get('velocity', 0):.2f} exceeds maximum {error.details.get('max_velocity', 0)}"
                )
            elif error.error_type in ["invalid_mass", "invalid_radius"]:
                # These are handled by physics violation correction
                corrected_state = self.error_corrector.correct_physics_violations(state)
                return corrected_state

        if correctable_errors:
            corrected_state = self.error_corrector.correct_errors(
                state, correctable_errors
            )

            # Record corrections
            correction_stats = self.error_corrector.get_correction_statistics()
            if correction_stats["recent_corrections"]:
                report.corrections_applied.extend(
                    correction_stats["recent_corrections"]
                )
                self.correction_count += len(correction_stats["recent_corrections"])

            return corrected_state

        return state

    def _calculate_overall_confidence(self, report: ValidationReport) -> float:
        """Calculate overall confidence based on validation results."""
        if not report.validation_results:
            return 0.0

        # Average confidence from all validation results
        confidences = [
            result.confidence for result in report.validation_results.values()
        ]
        base_confidence = sum(confidences) / len(confidences)

        # Reduce confidence based on number of corrections
        correction_penalty = min(len(report.corrections_applied) * 0.1, 0.5)

        # Reduce confidence based on physics errors
        physics_penalty = min(len(report.physics_errors) * 0.15, 0.6)

        overall_confidence = max(
            0.0, base_confidence - correction_penalty - physics_penalty
        )

        return overall_confidence

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health metrics."""
        correction_stats = self.error_corrector.get_correction_statistics()

        # Calculate success rate
        success_rate = 1.0
        if self.validation_count > 0:
            success_rate = 1.0 - (self.correction_count / self.validation_count)

        # Calculate average processing time
        avg_processing_time = 0.0
        if self.validation_count > 0:
            avg_processing_time = self.total_processing_time / self.validation_count

        return {
            "validation_count": self.validation_count,
            "correction_count": self.correction_count,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "correction_statistics": correction_stats,
            "validators": {
                "state_validator": self.state_validator.get_validation_statistics(),
                "physics_validator_enabled": self.physics_validator is not None,
                "auto_correction_enabled": self.auto_correct,
            },
        }

    def reset_statistics(self) -> None:
        """Reset all validation and correction statistics."""
        self.validation_count = 0
        self.correction_count = 0
        self.total_processing_time = 0.0
        self.error_corrector.reset_statistics()

        logger.info("ValidationManager statistics reset")

    def set_correction_strategy(self, strategy: CorrectionStrategy) -> None:
        """Change the correction strategy."""
        self.error_corrector.set_strategy(strategy)
        logger.info(
            f"ValidationManager correction strategy changed to {strategy.value}"
        )

    def enable_auto_correction(self, enabled: bool) -> None:
        """Enable or disable automatic correction."""
        self.auto_correct = enabled
        logger.info(
            f"ValidationManager auto-correction {'enabled' if enabled else 'disabled'}"
        )


# Convenience function for easy integration
def validate_game_state(
    game_state: GameState,
    previous_state: Optional[GameState] = None,
    auto_correct: bool = True,
) -> tuple[GameState, ValidationReport]:
    """Convenience function for validating and correcting a game state.

    Args:
        game_state: Game state to validate
        previous_state: Previous state for physics validation
        auto_correct: Whether to automatically apply corrections

    Returns:
        Tuple of (corrected_state, validation_report)
    """
    manager = ValidationManager(auto_correct=auto_correct)
    return manager.validate_and_correct(game_state, previous_state)
