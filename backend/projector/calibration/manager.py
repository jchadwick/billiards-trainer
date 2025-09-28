"""Calibration manager for coordinating the complete calibration process.

This module provides the main CalibrationManager class that coordinates
keystone calibration, geometric calibration, and persistence operations.
It provides a unified interface for the calibration system.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .geometric import GeometricCalibrator, TableDimensions
from .keystone import CornerPoints, KeystoneCalibrator, KeystoneParams
from .persistence import CalibrationPersistence

logger = logging.getLogger(__name__)


class CalibrationState(Enum):
    """Calibration process states."""

    IDLE = "idle"
    KEYSTONE_SETUP = "keystone_setup"
    KEYSTONE_ADJUSTMENT = "keystone_adjustment"
    GEOMETRIC_SETUP = "geometric_setup"
    GEOMETRIC_POINTS = "geometric_points"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"


class CalibrationMethod(Enum):
    """Calibration methods."""

    MANUAL = "manual"
    AUTOMATIC = "automatic"
    ASSISTED = "assisted"


@dataclass
class CalibrationStatus:
    """Current calibration status."""

    state: CalibrationState
    progress: float  # 0.0 to 1.0
    message: str
    errors: list[str]
    warnings: list[str]


@dataclass
class CalibrationSettings:
    """Calibration process settings."""

    method: CalibrationMethod = CalibrationMethod.MANUAL
    auto_save: bool = True
    validation_tolerance: float = 5.0  # pixels
    grid_size: int = 10
    corner_adjustment_step: float = 1.0  # pixels
    keystone_adjustment_step: float = 0.01
    show_grid: bool = True
    show_crosshairs: bool = True
    show_test_patterns: bool = True


class CalibrationManager:
    """Manages the complete projector calibration process.

    This class coordinates keystone calibration, geometric calibration,
    and persistence operations. It provides:
    - Unified calibration workflow
    - Progress tracking and status updates
    - Error handling and validation
    - Profile management
    - Integration with display system
    """

    def __init__(
        self,
        display_width: int,
        display_height: int,
        table_dimensions: TableDimensions,
        calibration_dir: Optional[Path] = None,
    ):
        """Initialize calibration manager.

        Args:
            display_width: Display width in pixels
            display_height: Display height in pixels
            table_dimensions: Physical table dimensions
            calibration_dir: Directory for calibration files
        """
        self.display_width = display_width
        self.display_height = display_height
        self.table_dimensions = table_dimensions

        # Initialize calibration components
        self.keystone_calibrator = KeystoneCalibrator(display_width, display_height)
        self.geometric_calibrator = GeometricCalibrator(
            table_dimensions, display_width, display_height
        )
        self.persistence = CalibrationPersistence(
            calibration_dir or Path("config/calibration")
        )

        # Calibration state
        self.state = CalibrationState.IDLE
        self.settings = CalibrationSettings()
        self.status = CalibrationStatus(
            state=CalibrationState.IDLE,
            progress=0.0,
            message="Ready for calibration",
            errors=[],
            warnings=[],
        )

        # Callbacks for status updates
        self.status_callbacks: list[Callable[[CalibrationStatus], None]] = []

        # Current calibration session data
        self.session_start_time: Optional[float] = None
        self.current_profile_name: Optional[str] = None
        self.calibration_complete: bool = False

        logger.info(
            f"CalibrationManager initialized for {display_width}x{display_height} display"
        )

    def add_status_callback(
        self, callback: Callable[[CalibrationStatus], None]
    ) -> None:
        """Add a callback for status updates.

        Args:
            callback: Function to call when status changes
        """
        self.status_callbacks.append(callback)

    def remove_status_callback(
        self, callback: Callable[[CalibrationStatus], None]
    ) -> None:
        """Remove a status update callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)

    def _update_status(
        self,
        state: Optional[CalibrationState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        errors: Optional[list[str]] = None,
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Update calibration status and notify callbacks."""
        if state is not None:
            self.state = state
            self.status.state = state

        if progress is not None:
            self.status.progress = max(0.0, min(1.0, progress))

        if message is not None:
            self.status.message = message

        if errors is not None:
            self.status.errors = errors.copy()

        if warnings is not None:
            self.status.warnings = warnings.copy()

        # Notify callbacks
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")

        logger.debug(
            f"Calibration status: {self.status.state.value} - {self.status.message}"
        )

    def start_calibration(
        self, method: CalibrationMethod = CalibrationMethod.MANUAL
    ) -> bool:
        """Start the calibration process.

        Args:
            method: Calibration method to use

        Returns:
            True if calibration started successfully
        """
        try:
            if self.state != CalibrationState.IDLE:
                logger.warning("Calibration already in progress")
                return False

            self.settings.method = method
            self.session_start_time = time.time()
            self.calibration_complete = False

            # Reset calibration state
            self.keystone_calibrator.reset_calibration()
            self.geometric_calibrator.reset_calibration()

            self._update_status(
                state=CalibrationState.KEYSTONE_SETUP,
                progress=0.1,
                message="Starting keystone calibration setup",
                errors=[],
                warnings=[],
            )

            logger.info(f"Calibration started using {method.value} method")
            return True

        except Exception as e:
            logger.error(f"Failed to start calibration: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Failed to start calibration: {e}",
                errors=[str(e)],
            )
            return False

    def setup_keystone_corners(self, corner_points: CornerPoints) -> bool:
        """Set up initial keystone corner points.

        Args:
            corner_points: Initial corner points for keystone correction

        Returns:
            True if setup successful
        """
        try:
            if self.state != CalibrationState.KEYSTONE_SETUP:
                logger.error("Not in keystone setup state")
                return False

            self.keystone_calibrator.set_corner_points(corner_points)

            self._update_status(
                state=CalibrationState.KEYSTONE_ADJUSTMENT,
                progress=0.2,
                message="Keystone corners set. Ready for adjustment.",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to setup keystone corners: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Keystone setup failed: {e}",
                errors=[str(e)],
            )
            return False

    def adjust_keystone_corner(
        self, corner_index: int, new_position: tuple[float, float]
    ) -> bool:
        """Adjust a keystone corner point.

        Args:
            corner_index: Corner index to adjust (0-3)
            new_position: New position for the corner

        Returns:
            True if adjustment successful
        """
        try:
            if self.state != CalibrationState.KEYSTONE_ADJUSTMENT:
                logger.error("Not in keystone adjustment state")
                return False

            self.keystone_calibrator.adjust_corner_point(corner_index, new_position)

            self._update_status(
                message=f"Adjusted corner {corner_index} to {new_position}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to adjust keystone corner: {e}")
            self._update_status(errors=[str(e)])
            return False

    def set_keystone_params(self, params: KeystoneParams) -> bool:
        """Set keystone correction parameters.

        Args:
            params: Keystone parameters

        Returns:
            True if parameters set successfully
        """
        try:
            if self.state not in [
                CalibrationState.KEYSTONE_SETUP,
                CalibrationState.KEYSTONE_ADJUSTMENT,
            ]:
                logger.error("Not in keystone calibration state")
                return False

            self.keystone_calibrator.set_keystone_params(params)

            self._update_status(message="Keystone parameters updated")

            return True

        except Exception as e:
            logger.error(f"Failed to set keystone parameters: {e}")
            self._update_status(errors=[str(e)])
            return False

    def complete_keystone_calibration(self) -> bool:
        """Complete keystone calibration and move to geometric setup.

        Returns:
            True if keystone calibration completed successfully
        """
        try:
            if self.state != CalibrationState.KEYSTONE_ADJUSTMENT:
                logger.error("Not in keystone adjustment state")
                return False

            # Validate keystone calibration
            is_valid, errors = self.keystone_calibrator.validate_calibration()
            if not is_valid:
                self._update_status(
                    message="Keystone calibration validation failed", errors=errors
                )
                return False

            self._update_status(
                state=CalibrationState.GEOMETRIC_SETUP,
                progress=0.4,
                message="Keystone calibration complete. Starting geometric setup.",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to complete keystone calibration: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Keystone completion failed: {e}",
                errors=[str(e)],
            )
            return False

    def add_geometric_target(
        self,
        table_x: float,
        table_y: float,
        display_x: float,
        display_y: float,
        label: str = "",
    ) -> bool:
        """Add a geometric calibration target.

        Args:
            table_x: Table X coordinate
            table_y: Table Y coordinate
            display_x: Display X coordinate
            display_y: Display Y coordinate
            label: Optional label

        Returns:
            True if target added successfully
        """
        try:
            if self.state not in [
                CalibrationState.GEOMETRIC_SETUP,
                CalibrationState.GEOMETRIC_POINTS,
            ]:
                logger.error("Not in geometric calibration state")
                return False

            self.geometric_calibrator.add_calibration_target(
                table_x, table_y, display_x, display_y, label
            )

            # Update state if this is the first target
            if self.state == CalibrationState.GEOMETRIC_SETUP:
                self._update_status(
                    state=CalibrationState.GEOMETRIC_POINTS,
                    progress=0.5,
                    message="Adding geometric calibration points",
                )

            target_count = len(self.geometric_calibrator.calibration_targets)
            self._update_status(
                message=f"Added geometric target {target_count}: {label or 'unlabeled'}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add geometric target: {e}")
            self._update_status(errors=[str(e)])
            return False

    def add_table_corner_targets(
        self, corner_displays: list[tuple[float, float]]
    ) -> bool:
        """Add calibration targets for table corners.

        Args:
            corner_displays: Display coordinates for table corners

        Returns:
            True if corners added successfully
        """
        try:
            if self.state not in [
                CalibrationState.GEOMETRIC_SETUP,
                CalibrationState.GEOMETRIC_POINTS,
            ]:
                logger.error("Not in geometric calibration state")
                return False

            self.geometric_calibrator.add_table_corner_targets(corner_displays)

            self._update_status(
                state=CalibrationState.GEOMETRIC_POINTS,
                progress=0.6,
                message="Table corner targets added",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add table corner targets: {e}")
            self._update_status(errors=[str(e)])
            return False

    def calculate_geometric_transform(self) -> bool:
        """Calculate the geometric transformation.

        Returns:
            True if transform calculated successfully
        """
        try:
            if self.state != CalibrationState.GEOMETRIC_POINTS:
                logger.error("Not in geometric points state")
                return False

            success = self.geometric_calibrator.calculate_transform()
            if not success:
                self._update_status(
                    message="Failed to calculate geometric transform",
                    errors=["Insufficient or invalid calibration targets"],
                )
                return False

            self._update_status(
                state=CalibrationState.VALIDATION,
                progress=0.8,
                message="Geometric transform calculated. Validating calibration...",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to calculate geometric transform: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Transform calculation failed: {e}",
                errors=[str(e)],
            )
            return False

    def validate_calibration(self) -> bool:
        """Validate the complete calibration.

        Returns:
            True if calibration is valid
        """
        try:
            if self.state != CalibrationState.VALIDATION:
                logger.error("Not in validation state")
                return False

            errors = []
            warnings = []

            # Validate keystone calibration
            (
                keystone_valid,
                keystone_errors,
            ) = self.keystone_calibrator.validate_calibration()
            if not keystone_valid:
                errors.extend([f"Keystone: {err}" for err in keystone_errors])

            # Validate geometric calibration
            (
                geometric_valid,
                geometric_errors,
            ) = self.geometric_calibrator.validate_calibration(
                self.settings.validation_tolerance
            )
            if not geometric_valid:
                errors.extend([f"Geometric: {err}" for err in geometric_errors])

            # Check calibration error
            if (
                self.geometric_calibrator.calibration_error
                > self.settings.validation_tolerance
            ):
                warnings.append(
                    f"Calibration error ({self.geometric_calibrator.calibration_error:.1f}px) "
                    f"exceeds tolerance ({self.settings.validation_tolerance}px)"
                )

            if errors:
                self._update_status(
                    message="Calibration validation failed",
                    errors=errors,
                    warnings=warnings,
                )
                return False

            self._update_status(
                state=CalibrationState.COMPLETE,
                progress=1.0,
                message="Calibration validation successful",
                errors=[],
                warnings=warnings,
            )

            self.calibration_complete = True
            return True

        except Exception as e:
            logger.error(f"Failed to validate calibration: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Validation failed: {e}",
                errors=[str(e)],
            )
            return False

    def save_calibration(self, profile_name: Optional[str] = None) -> bool:
        """Save the current calibration.

        Args:
            profile_name: Optional profile name, saves as current if None

        Returns:
            True if saved successfully
        """
        try:
            if not self.calibration_complete:
                logger.error("Calibration not complete")
                return False

            # Get calibration data
            keystone_data = self.keystone_calibrator.get_calibration_data()
            geometric_data = self.geometric_calibrator.get_calibration_data()

            # Create metadata
            metadata = {
                "calibration_time": self.session_start_time,
                "completion_time": time.time(),
                "method": self.settings.method.value,
                "display_resolution": [self.display_width, self.display_height],
                "table_dimensions": {
                    "length": self.table_dimensions.length,
                    "width": self.table_dimensions.width,
                },
                "calibration_error": self.geometric_calibrator.calibration_error,
                "target_count": len(self.geometric_calibrator.calibration_targets),
            }

            if profile_name:
                # Save as named profile
                success = self.persistence.save_profile(
                    name=profile_name,
                    description=f"Calibration created on {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    keystone_data=keystone_data,
                    geometric_data=geometric_data,
                    metadata=metadata,
                )
                if success:
                    self.current_profile_name = profile_name
            else:
                # Save as current calibration
                success = self.persistence.save_current_calibration(
                    keystone_data=keystone_data,
                    geometric_data=geometric_data,
                    metadata=metadata,
                )

            if success:
                logger.info(
                    f"Calibration saved{'as profile ' + profile_name if profile_name else ''}"
                )
                self._update_status(
                    message=f"Calibration saved{'as profile ' + profile_name if profile_name else ''}"
                )
            else:
                self._update_status(errors=["Failed to save calibration"])

            return success

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            self._update_status(errors=[f"Save failed: {e}"])
            return False

    def load_calibration(self, profile_name: Optional[str] = None) -> bool:
        """Load a saved calibration.

        Args:
            profile_name: Profile name to load, loads current if None

        Returns:
            True if loaded successfully
        """
        try:
            if profile_name:
                # Load named profile
                profile = self.persistence.load_profile(profile_name)
                if not profile:
                    return False

                keystone_data = profile.keystone_data
                geometric_data = profile.geometric_data
                self.current_profile_name = profile_name

            else:
                # Load current calibration
                (
                    keystone_data,
                    geometric_data,
                    _,
                ) = self.persistence.load_current_calibration()
                if keystone_data is None or geometric_data is None:
                    return False

            # Apply calibration data
            keystone_success = self.keystone_calibrator.load_calibration_data(
                keystone_data
            )
            geometric_success = self.geometric_calibrator.load_calibration_data(
                geometric_data
            )

            if keystone_success and geometric_success:
                self.calibration_complete = True
                self._update_status(
                    state=CalibrationState.COMPLETE,
                    progress=1.0,
                    message=f"Calibration loaded{'from profile ' + profile_name if profile_name else ''}",
                )

                logger.info(
                    f"Calibration loaded{'from profile ' + profile_name if profile_name else ''}"
                )
                return True
            else:
                self._update_status(
                    state=CalibrationState.ERROR,
                    message="Failed to apply loaded calibration data",
                    errors=["Invalid calibration data format"],
                )
                return False

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self._update_status(
                state=CalibrationState.ERROR,
                message=f"Load failed: {e}",
                errors=[str(e)],
            )
            return False

    def cancel_calibration(self) -> None:
        """Cancel the current calibration process."""
        try:
            self._update_status(
                state=CalibrationState.IDLE,
                progress=0.0,
                message="Calibration cancelled",
                errors=[],
                warnings=[],
            )

            self.session_start_time = None
            self.calibration_complete = False

            logger.info("Calibration cancelled")

        except Exception as e:
            logger.error(f"Failed to cancel calibration: {e}")

    def reset_calibration(self) -> None:
        """Reset calibration to initial state."""
        try:
            self.keystone_calibrator.reset_calibration()
            self.geometric_calibrator.reset_calibration()

            self._update_status(
                state=CalibrationState.IDLE,
                progress=0.0,
                message="Calibration reset",
                errors=[],
                warnings=[],
            )

            self.session_start_time = None
            self.current_profile_name = None
            self.calibration_complete = False

            logger.info("Calibration reset")

        except Exception as e:
            logger.error(f"Failed to reset calibration: {e}")

    def get_calibration_data(self) -> dict[str, Any]:
        """Get complete calibration data.

        Returns:
            Dictionary containing all calibration data
        """
        return {
            "state": self.state.value,
            "complete": self.calibration_complete,
            "profile_name": self.current_profile_name,
            "settings": {
                "method": self.settings.method.value,
                "validation_tolerance": self.settings.validation_tolerance,
                "grid_size": self.settings.grid_size,
            },
            "keystone": self.keystone_calibrator.get_calibration_data(),
            "geometric": self.geometric_calibrator.get_calibration_data(),
            "status": {
                "state": self.status.state.value,
                "progress": self.status.progress,
                "message": self.status.message,
                "errors": self.status.errors.copy(),
                "warnings": self.status.warnings.copy(),
            },
        }

    def transform_point(self, table_x: float, table_y: float) -> tuple[float, float]:
        """Transform a point from table coordinates to display coordinates.

        Args:
            table_x: Table X coordinate
            table_y: Table Y coordinate

        Returns:
            Display coordinates (x, y)
        """
        # First apply geometric transform
        display_x, display_y = self.geometric_calibrator.table_to_display(
            table_x, table_y
        )

        # Then apply keystone transform
        return self.keystone_calibrator.transform_point(display_x, display_y)

    def inverse_transform_point(
        self, display_x: float, display_y: float
    ) -> tuple[float, float]:
        """Transform a point from display coordinates to table coordinates.

        Args:
            display_x: Display X coordinate
            display_y: Display Y coordinate

        Returns:
            Table coordinates (x, y)
        """
        # This requires inverse operations in reverse order
        # Note: This is a simplified implementation
        # For accurate inverse transform, we'd need proper inverse of the composed transform
        return self.geometric_calibrator.display_to_table(display_x, display_y)

    def is_calibration_valid(self) -> bool:
        """Check if current calibration is valid and complete.

        Returns:
            True if calibration is valid
        """
        return (
            self.calibration_complete
            and self.state == CalibrationState.COMPLETE
            and len(self.status.errors) == 0
        )

    def get_profiles(self) -> list[dict[str, Any]]:
        """Get list of available calibration profiles.

        Returns:
            List of profile information
        """
        return self.persistence.list_profiles()

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a calibration profile.

        Args:
            profile_name: Name of profile to delete

        Returns:
            True if deleted successfully
        """
        return self.persistence.delete_profile(profile_name)
