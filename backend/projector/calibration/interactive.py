"""Interactive calibration interface.

This module provides an interactive interface for projector calibration,
supporting both manual and guided calibration workflows.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .manager import CalibrationManager, CalibrationMethod

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Interaction modes for calibration."""

    MANUAL_ADJUSTMENT = "manual_adjustment"
    GUIDED_WORKFLOW = "guided_workflow"
    QUICK_SETUP = "quick_setup"


@dataclass
class InteractionEvent:
    """User interaction event during calibration."""

    event_type: str
    data: dict[str, Any]
    timestamp: float


class InteractiveCalibration:
    """Interactive projector calibration interface.

    This class provides methods for interactive calibration including:
    - Guided calibration workflows
    - Manual corner adjustment
    - Real-time feedback
    - User input handling
    """

    def __init__(self, calibration_manager: CalibrationManager):
        """Initialize interactive calibration.

        Args:
            calibration_manager: The calibration manager instance
        """
        self.calibration_manager = calibration_manager
        self.interaction_mode = InteractionMode.MANUAL_ADJUSTMENT

        # Interactive state
        self.current_step = 0
        self.total_steps = 0
        self.user_instructions = ""
        self.feedback_message = ""

        # Event handling
        self.event_callbacks: dict[str, list[Callable]] = {}
        self.interaction_history: list[InteractionEvent] = []

        # UI state
        self.show_grid = True
        self.show_crosshairs = True
        self.show_instructions = True
        self.grid_opacity = 0.6
        self.crosshair_size = 30.0

        logger.info("InteractiveCalibration initialized")

    def start_calibration(
        self, method: CalibrationMethod = CalibrationMethod.MANUAL
    ) -> bool:
        """Start interactive calibration procedure.

        Args:
            method: Calibration method to use

        Returns:
            True if calibration started successfully
        """
        try:
            if method == CalibrationMethod.GUIDED_WORKFLOW:
                self.interaction_mode = InteractionMode.GUIDED_WORKFLOW
                return self._start_guided_calibration()
            elif method == CalibrationMethod.QUICK_SETUP:
                self.interaction_mode = InteractionMode.QUICK_SETUP
                return self._start_quick_setup()
            else:
                self.interaction_mode = InteractionMode.MANUAL_ADJUSTMENT
                return self._start_manual_calibration()

        except Exception as e:
            logger.error(f"Failed to start interactive calibration: {e}")
            return False

    def _start_guided_calibration(self) -> bool:
        """Start guided calibration workflow."""
        self.total_steps = 6
        self.current_step = 1

        # Start the calibration process
        success = self.calibration_manager.start_calibration(CalibrationMethod.MANUAL)
        if not success:
            return False

        self.user_instructions = (
            "Step 1/6: Position corner markers\n\n"
            "Click and drag the corner markers to align them with the physical table corners.\n"
            "Use precise movements for better accuracy."
        )

        self._emit_event(
            "calibration_started",
            {
                "mode": "guided",
                "step": self.current_step,
                "total_steps": self.total_steps,
                "instructions": self.user_instructions,
            },
        )

        return True

    def _start_quick_setup(self) -> bool:
        """Start quick setup calibration."""
        self.total_steps = 3
        self.current_step = 1

        success = self.calibration_manager.start_calibration(CalibrationMethod.MANUAL)
        if not success:
            return False

        self.user_instructions = (
            "Quick Setup 1/3: Basic corner alignment\n\n"
            "Roughly position the corners to match your table.\n"
            "Fine adjustments will be made automatically."
        )

        self._emit_event(
            "calibration_started",
            {
                "mode": "quick",
                "step": self.current_step,
                "total_steps": self.total_steps,
                "instructions": self.user_instructions,
            },
        )

        return True

    def _start_manual_calibration(self) -> bool:
        """Start manual calibration mode."""
        success = self.calibration_manager.start_calibration(CalibrationMethod.MANUAL)
        if not success:
            return False

        self.user_instructions = (
            "Manual Calibration Mode\n\n"
            "Use the controls to adjust keystone and geometric calibration manually.\n"
            "All adjustments are applied in real-time."
        )

        self._emit_event(
            "calibration_started",
            {"mode": "manual", "instructions": self.user_instructions},
        )

        return True

    def adjust_corner_point(
        self, corner_index: int, new_position: tuple[float, float]
    ) -> bool:
        """Adjust calibration corner point.

        Args:
            corner_index: Corner index (0-3)
            new_position: New position (x, y) in display coordinates

        Returns:
            True if adjustment successful
        """
        try:
            # Get current keystone calibrator
            keystone_cal = self.calibration_manager.keystone_calibrator

            # Apply the adjustment
            keystone_cal.adjust_corner_point(corner_index, new_position)

            # Update feedback
            self.feedback_message = f"Corner {corner_index} moved to ({new_position[0]:.1f}, {new_position[1]:.1f})"

            # Record the interaction
            self._record_interaction(
                "corner_adjusted",
                {"corner_index": corner_index, "position": new_position},
            )

            # Emit event for UI updates
            self._emit_event(
                "corner_adjusted",
                {
                    "corner_index": corner_index,
                    "position": new_position,
                    "feedback": self.feedback_message,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to adjust corner point: {e}")
            self.feedback_message = f"Error adjusting corner: {e}"
            return False

    def advance_guided_step(self) -> bool:
        """Advance to the next step in guided calibration.

        Returns:
            True if advanced successfully
        """
        if self.interaction_mode != InteractionMode.GUIDED_WORKFLOW:
            return False

        try:
            if self.current_step == 1:
                # Complete keystone setup
                success = self.calibration_manager.complete_keystone_calibration()
                if not success:
                    self.feedback_message = "Keystone calibration validation failed"
                    return False

                self.current_step = 2
                self.user_instructions = (
                    "Step 2/6: Mark table corners\n\n"
                    "Click on each physical table corner in order:\n"
                    "1. Top-left\n2. Top-right\n3. Bottom-right\n4. Bottom-left"
                )

            elif self.current_step == 2:
                # Table corner marking completed in UI
                self.current_step = 3
                self.user_instructions = (
                    "Step 3/6: Add center reference\n\n"
                    "Click on the center of the table to improve calibration accuracy."
                )

            elif self.current_step == 3:
                # Calculate geometric transform
                success = self.calibration_manager.calculate_geometric_transform()
                if not success:
                    self.feedback_message = "Failed to calculate geometric transform"
                    return False

                self.current_step = 4
                self.user_instructions = (
                    "Step 4/6: Validation\n\n"
                    "Validating calibration accuracy...\n"
                    "Check that the overlay aligns with the physical table."
                )

            elif self.current_step == 4:
                # Validate calibration
                success = self.calibration_manager.validate_calibration()
                if not success:
                    self.feedback_message = (
                        "Calibration validation failed - please adjust"
                    )
                    return False

                self.current_step = 5
                self.user_instructions = (
                    "Step 5/6: Save calibration\n\n"
                    "Calibration is complete and valid.\n"
                    "Save this calibration for future use?"
                )

            elif self.current_step == 5:
                # Save calibration
                profile_name = f"calibration_{int(time.time())}"
                success = self.calibration_manager.save_calibration(profile_name)
                if not success:
                    self.feedback_message = "Failed to save calibration"
                    return False

                self.current_step = 6
                self.user_instructions = (
                    "Step 6/6: Complete\n\n"
                    "🎉 Calibration completed successfully!\n"
                    f"Saved as profile: {profile_name}"
                )

                self._emit_event(
                    "calibration_completed",
                    {"profile_name": profile_name, "success": True},
                )

            # Emit step advancement event
            self._emit_event(
                "step_advanced",
                {
                    "step": self.current_step,
                    "total_steps": self.total_steps,
                    "instructions": self.user_instructions,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to advance guided step: {e}")
            self.feedback_message = f"Step advancement failed: {e}"
            return False

    def add_table_corner_click(self, display_position: tuple[float, float]) -> bool:
        """Add a table corner click during geometric calibration.

        Args:
            display_position: Clicked position in display coordinates

        Returns:
            True if corner added successfully
        """
        try:
            # Get current number of targets
            target_count = len(
                self.calibration_manager.geometric_calibrator.calibration_targets
            )

            # Determine table position based on corner order
            table_dims = self.calibration_manager.table_dimensions
            corner_positions = [
                (0.0, 0.0),  # Top-left
                (table_dims.length, 0.0),  # Top-right
                (table_dims.length, table_dims.width),  # Bottom-right
                (0.0, table_dims.width),  # Bottom-left
            ]

            if target_count < 4:
                table_pos = corner_positions[target_count]
                label = ["top_left", "top_right", "bottom_right", "bottom_left"][
                    target_count
                ]

                success = self.calibration_manager.add_geometric_target(
                    table_pos[0],
                    table_pos[1],
                    display_position[0],
                    display_position[1],
                    label,
                )

                if success:
                    self.feedback_message = f"Added {label} corner at ({display_position[0]:.1f}, {display_position[1]:.1f})"
                    self._emit_event(
                        "corner_added",
                        {
                            "corner_name": label,
                            "table_position": table_pos,
                            "display_position": display_position,
                            "count": target_count + 1,
                        },
                    )

                return success

            return False

        except Exception as e:
            logger.error(f"Failed to add table corner: {e}")
            return False

    def add_center_point_click(self, display_position: tuple[float, float]) -> bool:
        """Add center point for improved calibration accuracy.

        Args:
            display_position: Clicked position in display coordinates

        Returns:
            True if center point added successfully
        """
        try:
            table_dims = self.calibration_manager.table_dimensions
            center_x = table_dims.length / 2
            center_y = table_dims.width / 2

            success = self.calibration_manager.add_geometric_target(
                center_x, center_y, display_position[0], display_position[1], "center"
            )

            if success:
                self.feedback_message = f"Added center point at ({display_position[0]:.1f}, {display_position[1]:.1f})"
                self._emit_event(
                    "center_added",
                    {
                        "table_position": (center_x, center_y),
                        "display_position": display_position,
                    },
                )

            return success

        except Exception as e:
            logger.error(f"Failed to add center point: {e}")
            return False

    def get_display_elements(self) -> dict[str, Any]:
        """Get current display elements for rendering.

        Returns:
            Dictionary containing elements to display
        """
        elements = {
            "show_grid": self.show_grid,
            "show_crosshairs": self.show_crosshairs,
            "show_instructions": self.show_instructions,
            "instructions": self.user_instructions,
            "feedback": self.feedback_message,
            "progress": (
                self.current_step / max(self.total_steps, 1)
                if self.total_steps > 0
                else 0.0
            ),
            "step_info": (
                f"{self.current_step}/{self.total_steps}"
                if self.total_steps > 0
                else "Manual"
            ),
        }

        # Add grid if requested
        if self.show_grid:
            grid = self.calibration_manager.geometric_calibrator.generate_table_grid(
                0.2
            )
            elements["grid_lines"] = grid
            elements["grid_opacity"] = self.grid_opacity

        # Add crosshairs if requested
        if self.show_crosshairs:
            crosshairs = (
                self.calibration_manager.keystone_calibrator.generate_crosshairs(
                    self.crosshair_size
                )
            )
            elements["crosshairs"] = crosshairs

        return elements

    def set_display_options(
        self,
        show_grid: Optional[bool] = None,
        show_crosshairs: Optional[bool] = None,
        show_instructions: Optional[bool] = None,
        grid_opacity: Optional[float] = None,
        crosshair_size: Optional[float] = None,
    ) -> None:
        """Set display options for calibration UI.

        Args:
            show_grid: Whether to show calibration grid
            show_crosshairs: Whether to show corner crosshairs
            show_instructions: Whether to show instruction text
            grid_opacity: Grid opacity (0.0-1.0)
            crosshair_size: Crosshair size in pixels
        """
        if show_grid is not None:
            self.show_grid = show_grid

        if show_crosshairs is not None:
            self.show_crosshairs = show_crosshairs

        if show_instructions is not None:
            self.show_instructions = show_instructions

        if grid_opacity is not None:
            self.grid_opacity = max(0.0, min(1.0, grid_opacity))

        if crosshair_size is not None:
            self.crosshair_size = max(10.0, min(100.0, crosshair_size))

        self._emit_event(
            "display_options_changed",
            {
                "show_grid": self.show_grid,
                "show_crosshairs": self.show_crosshairs,
                "show_instructions": self.show_instructions,
                "grid_opacity": self.grid_opacity,
                "crosshair_size": self.crosshair_size,
            },
        )

    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for calibration events.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a calibration event to registered callbacks.

        Args:
            event_type: Type of event
            data: Event data
        """
        event = InteractionEvent(
            event_type=event_type, data=data, timestamp=time.time()
        )

        # Record in history
        self.interaction_history.append(event)

        # Call registered callbacks
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Event callback failed: {e}")

    def _record_interaction(self, interaction_type: str, data: dict[str, Any]) -> None:
        """Record user interaction for analysis.

        Args:
            interaction_type: Type of interaction
            data: Interaction data
        """
        self._emit_event(
            "user_interaction", {"interaction_type": interaction_type, **data}
        )

    def get_calibration_status(self) -> dict[str, Any]:
        """Get current calibration status.

        Returns:
            Status information dictionary
        """
        return {
            "mode": self.interaction_mode.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": (
                self.current_step / max(self.total_steps, 1)
                if self.total_steps > 0
                else 0.0
            ),
            "instructions": self.user_instructions,
            "feedback": self.feedback_message,
            "calibration_state": self.calibration_manager.state.value,
            "calibration_valid": self.calibration_manager.is_calibration_valid(),
            "interaction_count": len(self.interaction_history),
        }

    def cancel_calibration(self) -> None:
        """Cancel the current calibration process."""
        self.calibration_manager.cancel_calibration()
        self.current_step = 0
        self.total_steps = 0
        self.user_instructions = "Calibration cancelled"
        self.feedback_message = ""

        self._emit_event("calibration_cancelled", {"timestamp": time.time()})
