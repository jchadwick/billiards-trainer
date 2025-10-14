"""Table State Validation for Trajectory Calculations.

This module provides validation to ensure TableState is properly configured
before trajectory calculations are performed.
"""

import logging
from typing import Optional

from ..models import TableState

logger = logging.getLogger(__name__)


class TableStateValidationError(Exception):
    """Exception raised when table state is invalid for trajectory calculation."""

    pass


class TableStateValidator:
    """Validates table state for trajectory calculations."""

    def __init__(self):
        """Initialize table state validator."""
        self.last_validation_warnings = []

    def validate_for_trajectory(
        self, table_state: Optional[TableState], require_playing_area: bool = False
    ) -> tuple[bool, list[str]]:
        """Validate table state for trajectory calculation.

        Args:
            table_state: Table state to validate
            require_playing_area: If True, require playing_area_corners to be present

        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        self.last_validation_warnings = []
        errors = []

        # Check if table state exists
        if table_state is None:
            errors.append("Table state is None - cannot calculate trajectory")
            return False, errors

        # Validate basic dimensions
        if table_state.width <= 0:
            errors.append(f"Invalid table width: {table_state.width}")
        if table_state.height <= 0:
            errors.append(f"Invalid table height: {table_state.height}")

        # Validate pocket configuration
        if not table_state.pocket_positions:
            errors.append("Table has no pocket positions defined")
        elif len(table_state.pocket_positions) != 6:
            errors.append(
                f"Table must have exactly 6 pockets, found {len(table_state.pocket_positions)}"
            )

        if table_state.pocket_radius <= 0:
            errors.append(f"Invalid pocket radius: {table_state.pocket_radius}")

        # Validate physics parameters
        if not (0.0 <= table_state.cushion_elasticity <= 1.0):
            errors.append(
                f"Invalid cushion elasticity: {table_state.cushion_elasticity} "
                "(must be between 0.0 and 1.0)"
            )

        if table_state.surface_friction < 0:
            errors.append(
                f"Invalid surface friction: {table_state.surface_friction} "
                "(must be non-negative)"
            )

        # Check playing area corners
        if table_state.playing_area_corners is None:
            if require_playing_area:
                errors.append(
                    "Playing area corners not calibrated - trajectory accuracy will be reduced. "
                    "Please calibrate the table using the vision module."
                )
            else:
                self.last_validation_warnings.append(
                    "Playing area corners not defined - using rectangular bounds. "
                    "For better accuracy, calibrate the table with the vision module."
                )
        elif len(table_state.playing_area_corners) != 4:
            errors.append(
                f"Playing area corners must have exactly 4 points, "
                f"found {len(table_state.playing_area_corners)}"
            )
        else:
            # Validate corner coordinates are within table bounds
            for i, corner in enumerate(table_state.playing_area_corners):
                if corner.x < 0 or corner.x > table_state.width:
                    errors.append(
                        f"Corner {i} x-coordinate ({corner.x}) is outside table width "
                        f"(0-{table_state.width})"
                    )
                if corner.y < 0 or corner.y > table_state.height:
                    errors.append(
                        f"Corner {i} y-coordinate ({corner.y}) is outside table height "
                        f"(0-{table_state.height})"
                    )

            # Validate corners form a reasonable polygon
            if not self._validate_corner_geometry(table_state.playing_area_corners):
                errors.append(
                    "Playing area corners do not form a valid convex quadrilateral"
                )

        # Log warnings
        for warning in self.last_validation_warnings:
            logger.warning(warning)

        return len(errors) == 0, errors

    def _validate_corner_geometry(self, corners: list) -> bool:
        """Validate that corners form a reasonable quadrilateral.

        Args:
            corners: List of 4 Vector2D corner points

        Returns:
            True if corners form a valid quadrilateral
        """
        if len(corners) != 4:
            return False

        # Check that corners are not all collinear
        # Use cross product to check if consecutive edges are not parallel
        edge1 = (corners[1].x - corners[0].x, corners[1].y - corners[0].y)
        edge2 = (corners[2].x - corners[1].x, corners[2].y - corners[1].y)

        # Cross product
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

        # If cross product is near zero, points are collinear
        if abs(cross) < 1.0:  # Allow some tolerance
            return False

        # Check minimum area
        area = self._calculate_polygon_area(corners)
        if area < 1000:  # Minimum reasonable area in pixels
            return False

        return True

    def _calculate_polygon_area(self, corners: list) -> float:
        """Calculate area of polygon defined by corners using shoelace formula.

        Args:
            corners: List of Vector2D corner points

        Returns:
            Area of polygon
        """
        if len(corners) < 3:
            return 0.0

        area = 0.0
        for i in range(len(corners)):
            j = (i + 1) % len(corners)
            area += corners[i].x * corners[j].y
            area -= corners[j].x * corners[i].y

        return abs(area) / 2.0

    def get_validation_summary(self, table_state: Optional[TableState]) -> str:
        """Get a human-readable validation summary.

        Args:
            table_state: Table state to summarize

        Returns:
            Human-readable summary string
        """
        if table_state is None:
            return "Table state: None (INVALID)"

        is_valid, errors = self.validate_for_trajectory(table_state)

        summary_parts = [
            f"Table dimensions: {table_state.width:.1f}x{table_state.height:.1f}",
            f"Pockets: {len(table_state.pocket_positions) if table_state.pocket_positions else 0}",
            f"Playing area: {'calibrated' if table_state.playing_area_corners else 'uncalibrated'}",
        ]

        if is_valid:
            summary_parts.append("Status: VALID")
        else:
            summary_parts.append(f"Status: INVALID ({len(errors)} errors)")

        return ", ".join(summary_parts)
