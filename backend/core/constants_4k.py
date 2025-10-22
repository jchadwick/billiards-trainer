"""4K Coordinate System Constants.

All measurements are in 4K pixels (3840×2160).
Standard 9ft pool table dimensions mapped to pixel space.

This module defines the canonical 4K resolution-based coordinate system
that serves as the foundation for all spatial calculations in the billiards trainer.
"""

# ============================================================================
# CANONICAL RESOLUTION
# ============================================================================

CANONICAL_RESOLUTION = (3840, 2160)  # 4K UHD
CANONICAL_WIDTH = 3840
CANONICAL_HEIGHT = 2160


# ============================================================================
# TABLE DIMENSIONS (in 4K pixels)
# ============================================================================

# Standard 9ft pool table mapped to 4K space
# Physical: 2.54m × 1.27m (100" × 50")
# Aspect Ratio: 2:1 (maintained in pixel space)
TABLE_WIDTH_4K = 3200  # pixels - maintains 2:1 aspect ratio
TABLE_HEIGHT_4K = 1600  # pixels


# ============================================================================
# TABLE POSITIONING (center of 4K frame)
# ============================================================================

TABLE_CENTER_4K = (1920, 1080)  # Center of 4K frame

# Derived bounds (table centered in frame)
TABLE_LEFT_4K = TABLE_CENTER_4K[0] - TABLE_WIDTH_4K // 2  # 320
TABLE_TOP_4K = TABLE_CENTER_4K[1] - TABLE_HEIGHT_4K // 2  # 280
TABLE_RIGHT_4K = TABLE_LEFT_4K + TABLE_WIDTH_4K  # 3520
TABLE_BOTTOM_4K = TABLE_TOP_4K + TABLE_HEIGHT_4K  # 1880


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
# POCKET POSITIONS (in 4K pixels)
# ============================================================================

# Standard 6-pocket positions (4 corners + 2 middle)
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
# REFERENCE CONVERSIONS (DOCUMENTATION ONLY - NOT USED IN CODE)
# ============================================================================

# These values are for human understanding and validation only.
# They are NOT used in production code.
# All calculations should use pixels directly.

# Physical table dimensions
PHYSICAL_TABLE_WIDTH_MM = 2540.0  # 2.54m = 2540mm
PHYSICAL_TABLE_HEIGHT_MM = 1270.0  # 1.27m = 1270mm

# Conversion factors (REFERENCE ONLY)
PIXELS_PER_METER_REFERENCE = TABLE_WIDTH_4K / 2.54  # ~1259.84 pixels/meter
MM_PER_PIXEL_REFERENCE = PHYSICAL_TABLE_WIDTH_MM / TABLE_WIDTH_4K  # ~0.79375 mm/pixel
PIXELS_PER_MM_REFERENCE = (
    TABLE_WIDTH_4K / PHYSICAL_TABLE_WIDTH_MM
)  # ~1.259842 pixels/mm


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
    """Check if coordinates are on the table surface.

    Args:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        include_cushions: If True, include cushion area in bounds

    Returns:
        True if coordinates are on table
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
    """Get table bounds in 4K coordinates.

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
    # Table dimensions
    "TABLE_WIDTH_4K",
    "TABLE_HEIGHT_4K",
    "TABLE_CENTER_4K",
    "TABLE_LEFT_4K",
    "TABLE_TOP_4K",
    "TABLE_RIGHT_4K",
    "TABLE_BOTTOM_4K",
    # Ball dimensions
    "BALL_RADIUS_4K",
    "BALL_DIAMETER_4K",
    "BALL_MASS_KG",
    # Pocket dimensions
    "POCKET_RADIUS_4K",
    "POCKET_DIAMETER_MM",
    "POCKET_POSITIONS_4K",
    # Cushion dimensions
    "CUSHION_WIDTH_4K",
    "CUSHION_WIDTH_MM",
    # Validation helpers
    "is_valid_4k_coordinate",
    "is_on_table",
    "get_table_bounds_4k",
]
