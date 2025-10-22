"""4K Coordinate System Constants.

This module defines the canonical 4K resolution (3840×2160) coordinate system
that serves as the foundation for all spatial calculations in the billiards trainer.

The 4K resolution is the coordinate system canvas - think of it as the coordinate space.
Table dimensions within this canvas vary based on camera positioning and must come from
calibration data. Use TableState.from_calibration() to create tables with proper dimensions.

Only physical constants (ball diameter, pocket size, etc.) that are independent of
camera positioning are defined here.
"""

# ============================================================================
# CANONICAL RESOLUTION
# ============================================================================

CANONICAL_RESOLUTION = (3840, 2160)  # 4K UHD
CANONICAL_WIDTH = 3840
CANONICAL_HEIGHT = 2160


# ============================================================================
# TABLE DIMENSIONS - REFERENCE VALUES ONLY
# ============================================================================

# WARNING: These are reference values for a standard 9ft table in idealized positioning.
# PRODUCTION CODE MUST USE ACTUAL CALIBRATED TABLE DIMENSIONS from TableState.
# These constants are kept ONLY for:
# 1. Default/fallback values in analysis code
# 2. Unit tests that don't require calibration
# 3. Documentation and reference
#
# DO NOT use these for actual game state management or trajectory calculations.
# Use TableState.from_calibration() or TableState from vision detection instead.

# Standard 9ft pool table mapped to 4K space (idealized centered positioning)
# Physical: 2.54m × 1.27m (100" × 50")
# Aspect Ratio: 2:1 (maintained in pixel space)
TABLE_WIDTH_4K = 3200  # pixels - reference only, actual width from calibration
TABLE_HEIGHT_4K = 1600  # pixels - reference only, actual height from calibration

# Table positioning (idealized - assumes centered in 4K frame)
TABLE_CENTER_4K = (1920, 1080)  # Center of 4K frame

# Derived bounds (reference only - actual bounds from calibration)
TABLE_LEFT_4K = TABLE_CENTER_4K[0] - TABLE_WIDTH_4K // 2  # 320
TABLE_TOP_4K = TABLE_CENTER_4K[1] - TABLE_HEIGHT_4K // 2  # 280
TABLE_RIGHT_4K = TABLE_LEFT_4K + TABLE_WIDTH_4K  # 3520
TABLE_BOTTOM_4K = TABLE_TOP_4K + TABLE_HEIGHT_4K  # 1880

# Standard 6-pocket positions (reference only - actual positions from calibration)
POCKET_POSITIONS_4K = [
    # Top row
    (TABLE_LEFT_4K, TABLE_TOP_4K),  # Top-left corner
    (TABLE_CENTER_4K[0], TABLE_TOP_4K),  # Top-middle
    (TABLE_RIGHT_4K, TABLE_TOP_4K),  # Top-right corner
    # Bottom row
    (TABLE_LEFT_4K, TABLE_BOTTOM_4K),  # Bottom-left corner
    (TABLE_CENTER_4K[0], TABLE_BOTTOM_4K),  # Bottom-middle
    (TABLE_RIGHT_4K, TABLE_BOTTOM_4K),  # Bottom-right corner
]


# ============================================================================
# BALL DIMENSIONS
# ============================================================================

# Standard pool ball: 57.15mm (2.25 inches) diameter
BALL_DIAMETER_MM = 57.15  # millimeters
BALL_RADIUS_4K = 36  # pixels in 4K (half of 72px diameter)
BALL_DIAMETER_4K = 72  # pixels in 4K
BALL_MASS_KG = 0.17  # Mass in kilograms (not spatial - kept in SI units)


# ============================================================================
# POCKET DIMENSIONS
# ============================================================================

# Standard pocket: ~114.3mm (4.5 inches) diameter
POCKET_DIAMETER_MM = 114.3  # millimeters
POCKET_RADIUS_4K = 72  # pixels in 4K (approximate)


# ============================================================================
# CUSHION DIMENSIONS
# ============================================================================

# Standard cushion width: ~38.1mm (1.5 inches)
CUSHION_WIDTH_MM = 38.1  # millimeters
CUSHION_WIDTH_4K = 48  # pixels in 4K


# ============================================================================
# REFERENCE CONVERSIONS (DOCUMENTATION ONLY - NOT USED IN CODE)
# ============================================================================

# These values are for human understanding and validation only.
# They are NOT used in production code.
# All calculations should use pixels directly.

# Physical table dimensions (standard 9ft pool table)
PHYSICAL_TABLE_WIDTH_MM = 2540.0  # 2.54m = 2540mm (100 inches)
PHYSICAL_TABLE_HEIGHT_MM = 1270.0  # 1.27m = 1270mm (50 inches)

# Conversion factors (REFERENCE ONLY - actual values depend on calibration)
# These are calculated from the reference table dimensions above
PIXELS_PER_METER_REFERENCE = TABLE_WIDTH_4K / 2.54  # ~1259.84 pixels/meter
MM_PER_PIXEL_REFERENCE = PHYSICAL_TABLE_WIDTH_MM / TABLE_WIDTH_4K  # ~0.79375 mm/pixel
PIXELS_PER_MM_REFERENCE = (
    TABLE_WIDTH_4K / PHYSICAL_TABLE_WIDTH_MM
)  # ~1.259842 pixels/mm

# NOTE: These conversion factors depend on actual table dimensions in the calibrated
# coordinate space. For production use, calculate these from TableState dimensions.


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def is_valid_4k_coordinate(x: float, y: float) -> bool:
    """Check if coordinates are within 4K frame bounds.

    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels

    Returns:
        True if coordinates are within 4K bounds
    """
    return 0 <= x <= CANONICAL_WIDTH and 0 <= y <= CANONICAL_HEIGHT


def is_on_table(x: float, y: float, include_cushions: bool = True) -> bool:
    """Check if coordinates are on the reference table surface.

    WARNING: Uses reference table dimensions. For production use, check against
    actual TableState bounds from calibration.

    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        include_cushions: If True, include cushion area in bounds

    Returns:
        True if coordinates are on reference table
    """
    if include_cushions:
        # Include cushion width in bounds
        return (
            TABLE_LEFT_4K - CUSHION_WIDTH_4K <= x <= TABLE_RIGHT_4K + CUSHION_WIDTH_4K
            and TABLE_TOP_4K - CUSHION_WIDTH_4K
            <= y
            <= TABLE_BOTTOM_4K + CUSHION_WIDTH_4K
        )
    else:
        # Just playing surface
        return (
            TABLE_LEFT_4K <= x <= TABLE_RIGHT_4K
            and TABLE_TOP_4K <= y <= TABLE_BOTTOM_4K
        )


def get_table_bounds_4k(
    include_cushions: bool = False,
) -> tuple[float, float, float, float]:
    """Get reference table bounds in 4K coordinates.

    WARNING: Returns reference table dimensions. For production use, get bounds from
    actual TableState from calibration.

    Args:
        include_cushions: If True, include cushion area in bounds

    Returns:
        Tuple of (left, top, right, bottom) in 4K pixels
    """
    if include_cushions:
        return (
            TABLE_LEFT_4K - CUSHION_WIDTH_4K,
            TABLE_TOP_4K - CUSHION_WIDTH_4K,
            TABLE_RIGHT_4K + CUSHION_WIDTH_4K,
            TABLE_BOTTOM_4K + CUSHION_WIDTH_4K,
        )
    else:
        return (TABLE_LEFT_4K, TABLE_TOP_4K, TABLE_RIGHT_4K, TABLE_BOTTOM_4K)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Resolution
    "CANONICAL_RESOLUTION",
    "CANONICAL_WIDTH",
    "CANONICAL_HEIGHT",
    # Table dimensions (reference values only - use TableState for actual dimensions)
    "TABLE_WIDTH_4K",
    "TABLE_HEIGHT_4K",
    "TABLE_CENTER_4K",
    "TABLE_LEFT_4K",
    "TABLE_TOP_4K",
    "TABLE_RIGHT_4K",
    "TABLE_BOTTOM_4K",
    "POCKET_POSITIONS_4K",
    # Ball dimensions
    "BALL_RADIUS_4K",
    "BALL_DIAMETER_4K",
    "BALL_DIAMETER_MM",
    "BALL_MASS_KG",
    # Pocket dimensions
    "POCKET_RADIUS_4K",
    "POCKET_DIAMETER_MM",
    # Cushion dimensions
    "CUSHION_WIDTH_4K",
    "CUSHION_WIDTH_MM",
    # Physical reference dimensions
    "PHYSICAL_TABLE_WIDTH_MM",
    "PHYSICAL_TABLE_HEIGHT_MM",
    # Conversion factors (reference only)
    "PIXELS_PER_METER_REFERENCE",
    "MM_PER_PIXEL_REFERENCE",
    "PIXELS_PER_MM_REFERENCE",
    # Validation helpers
    "is_valid_4k_coordinate",
    "is_on_table",
    "get_table_bounds_4k",
]
