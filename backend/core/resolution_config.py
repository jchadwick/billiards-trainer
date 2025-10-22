"""Resolution configuration utilities for the 4K canonical coordinate system.

This module provides standard video resolutions and utilities for working with
resolution-based scaling in the billiards trainer system.

Key Features:
- Standard video resolutions (HD, Full HD, 4K, etc.)
- Resolution validation and helpers
- Aspect ratio calculations
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple


class StandardResolution(Enum):
    """Standard video resolutions commonly used in vision systems.

    Each resolution is represented as (width, height) in pixels.
    Format: WIDTH x HEIGHT
    """

    # SD Resolutions
    VGA = (640, 480)  # 4:3 aspect ratio
    SVGA = (800, 600)  # 4:3 aspect ratio

    # HD Resolutions
    HD_720 = (1280, 720)  # 720p, 16:9 aspect ratio
    HD_900 = (1600, 900)  # 900p, 16:9 aspect ratio
    HD_1080 = (1920, 1080)  # 1080p (Full HD), 16:9 aspect ratio

    # QHD Resolutions
    QHD = (2560, 1440)  # 1440p (2K), 16:9 aspect ratio

    # UHD/4K Resolutions
    UHD_4K = (3840, 2160)  # 4K UHD, 16:9 aspect ratio
    DCI_4K = (4096, 2160)  # DCI 4K (cinema standard)

    # 8K Resolution
    UHD_8K = (7680, 4320)  # 8K UHD, 16:9 aspect ratio

    @property
    def width(self) -> int:
        """Get the width in pixels."""
        return self.value[0]

    @property
    def height(self) -> int:
        """Get the height in pixels."""
        return self.value[1]

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio (width/height)."""
        return self.value[0] / self.value[1]

    @property
    def total_pixels(self) -> int:
        """Get the total number of pixels."""
        return self.value[0] * self.value[1]

    def __str__(self) -> str:
        """String representation showing resolution name and dimensions."""
        return f"{self.name} ({self.width}x{self.height})"


# Convenience functions


def get_standard_resolution(name: str) -> tuple[int, int] | None:
    """Get a standard resolution by name.

    Args:
        name: Resolution name (e.g., "HD_1080", "4K", "1080p")

    Returns:
        Tuple of (width, height) or None if not found

    Examples:
        >>> get_standard_resolution("HD_1080")
        (1920, 1080)
        >>> get_standard_resolution("4K")
        (3840, 2160)
    """
    # Normalize name
    name = name.upper().replace("P", "").replace("-", "_")

    # Handle common aliases
    aliases = {
        "720": "HD_720",
        "1080": "HD_1080",
        "1440": "QHD",
        "2K": "QHD",
        "4K": "UHD_4K",
        "8K": "UHD_8K",
        "FHD": "HD_1080",
        "FULLHD": "HD_1080",
        "FULL_HD": "HD_1080",
    }

    # Check if it's an alias
    if name in aliases:
        name = aliases[name]

    # Try to get from enum
    try:
        resolution = StandardResolution[name]
        return resolution.value
    except KeyError:
        return None


def validate_point_in_bounds(
    x: float, y: float, width: float, height: float, margin: float = 0.0
) -> bool:
    """Quick validation that a point is within rectangular bounds.

    Args:
        x: X coordinate
        y: Y coordinate
        width: Width of bounding rectangle
        height: Height of bounding rectangle
        margin: Optional margin from edges

    Returns:
        True if point is within bounds
    """
    return margin <= x <= width - margin and margin <= y <= height - margin


# Export main classes and functions
__all__ = [
    "StandardResolution",
    "get_standard_resolution",
    "validate_point_in_bounds",
]
