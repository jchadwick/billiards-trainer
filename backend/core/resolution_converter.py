"""Resolution Converter - Simple scaling between resolutions.

Replaces the complex CoordinateConverter with simple resolution scaling.
All coordinates are stored in 4K canonical format and scaled to/from
other resolutions as needed.

This module provides lightweight conversion utilities for scaling coordinates
between different resolutions while maintaining the 4K canonical standard.
"""

from typing import Tuple

from .constants_4k import CANONICAL_RESOLUTION


class ResolutionConverter:
    """Simple resolution scaler for converting between resolutions.

    This class provides static methods for calculating scale factors
    and performing resolution conversions. All conversions assume
    4K (3840×2160) as the canonical storage format.

    Example:
        >>> # Calculate scale from 1080p to 4K
        >>> scale = ResolutionConverter.calculate_scale_to_4k((1920, 1080))
        >>> # scale = (2.0, 2.0)
        >>>
        >>> # Scale coordinates from 1080p to 4K
        >>> x_4k, y_4k = ResolutionConverter.scale_to_4k(960, 540, (1920, 1080))
        >>> # x_4k = 1920.0, y_4k = 1080.0
    """

    @staticmethod
    def calculate_scale_to_4k(
        source_resolution: tuple[int, int],
    ) -> tuple[float, float]:
        """Calculate scale factors from source resolution to 4K.

        Args:
            source_resolution: Source (width, height) in pixels

        Returns:
            Tuple of (scale_x, scale_y) factors to multiply source coords by

        Example:
            >>> ResolutionConverter.calculate_scale_to_4k((1920, 1080))
            (2.0, 2.0)
            >>> ResolutionConverter.calculate_scale_to_4k((1280, 720))
            (3.0, 3.0)
        """
        scale_x = CANONICAL_RESOLUTION[0] / source_resolution[0]
        scale_y = CANONICAL_RESOLUTION[1] / source_resolution[1]
        return (scale_x, scale_y)

    @staticmethod
    def calculate_scale_from_4k(
        target_resolution: tuple[int, int],
    ) -> tuple[float, float]:
        """Calculate scale factors from 4K to target resolution.

        Args:
            target_resolution: Target (width, height) in pixels

        Returns:
            Tuple of (scale_x, scale_y) factors to multiply 4K coords by

        Example:
            >>> ResolutionConverter.calculate_scale_from_4k((1920, 1080))
            (0.5, 0.5)
            >>> ResolutionConverter.calculate_scale_from_4k((1280, 720))
            (0.333..., 0.333...)
        """
        scale_x = target_resolution[0] / CANONICAL_RESOLUTION[0]
        scale_y = target_resolution[1] / CANONICAL_RESOLUTION[1]
        return (scale_x, scale_y)

    @staticmethod
    def scale_to_4k(
        x: float, y: float, source_resolution: tuple[int, int]
    ) -> tuple[float, float]:
        """Convert coordinates from source resolution to 4K canonical.

        Args:
            x: X coordinate in source resolution
            y: Y coordinate in source resolution
            source_resolution: Source (width, height) in pixels

        Returns:
            Tuple of (x_4k, y_4k) in 4K canonical pixels

        Example:
            >>> ResolutionConverter.scale_to_4k(960, 540, (1920, 1080))
            (1920.0, 1080.0)
        """
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(source_resolution)
        x_4k = x * scale_x
        y_4k = y * scale_y
        return (x_4k, y_4k)

    @staticmethod
    def scale_from_4k(
        x_4k: float, y_4k: float, target_resolution: tuple[int, int]
    ) -> tuple[float, float]:
        """Convert coordinates from 4K canonical to target resolution.

        Args:
            x_4k: X coordinate in 4K canonical pixels
            y_4k: Y coordinate in 4K canonical pixels
            target_resolution: Target (width, height) in pixels

        Returns:
            Tuple of (x, y) in target resolution pixels

        Example:
            >>> ResolutionConverter.scale_from_4k(1920, 1080, (1920, 1080))
            (960.0, 540.0)
        """
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(
            target_resolution
        )
        x = x_4k * scale_x
        y = y_4k * scale_y
        return (x, y)

    @staticmethod
    def scale_between_resolutions(
        x: float,
        y: float,
        source_resolution: tuple[int, int],
        target_resolution: tuple[int, int],
    ) -> tuple[float, float]:
        """Convert coordinates between arbitrary resolutions.

        This performs a two-step conversion:
        1. Source resolution → 4K canonical
        2. 4K canonical → Target resolution

        Args:
            x: X coordinate in source resolution
            y: Y coordinate in source resolution
            source_resolution: Source (width, height) in pixels
            target_resolution: Target (width, height) in pixels

        Returns:
            Tuple of (x_target, y_target) in target resolution

        Example:
            >>> ResolutionConverter.scale_between_resolutions(
            ...     640, 360, (1280, 720), (1920, 1080)
            ... )
            (960.0, 540.0)
        """
        # Convert to 4K canonical first
        x_4k, y_4k = ResolutionConverter.scale_to_4k(x, y, source_resolution)

        # Then convert to target resolution
        x_target, y_target = ResolutionConverter.scale_from_4k(
            x_4k, y_4k, target_resolution
        )

        return (x_target, y_target)

    @staticmethod
    def scale_distance_to_4k(
        distance: float, source_resolution: tuple[int, int]
    ) -> float:
        """Scale a distance/radius from source resolution to 4K.

        Uses the X-axis scale factor for simplicity.
        For anisotropic scaling, use separate X/Y calculations.

        Args:
            distance: Distance/radius in source resolution pixels
            source_resolution: Source (width, height) in pixels

        Returns:
            Distance/radius in 4K canonical pixels

        Example:
            >>> ResolutionConverter.scale_distance_to_4k(18, (1920, 1080))
            36.0
        """
        scale_x, _ = ResolutionConverter.calculate_scale_to_4k(source_resolution)
        return distance * scale_x

    @staticmethod
    def scale_distance_from_4k(
        distance_4k: float, target_resolution: tuple[int, int]
    ) -> float:
        """Scale a distance/radius from 4K to target resolution.

        Uses the X-axis scale factor for simplicity.
        For anisotropic scaling, use separate X/Y calculations.

        Args:
            distance_4k: Distance/radius in 4K canonical pixels
            target_resolution: Target (width, height) in pixels

        Returns:
            Distance/radius in target resolution pixels

        Example:
            >>> ResolutionConverter.scale_distance_from_4k(36, (1920, 1080))
            18.0
        """
        scale_x, _ = ResolutionConverter.calculate_scale_from_4k(target_resolution)
        return distance_4k * scale_x

    @staticmethod
    def is_4k_canonical(resolution: tuple[int, int]) -> bool:
        """Check if a resolution is 4K canonical.

        Args:
            resolution: Resolution (width, height) to check

        Returns:
            True if resolution matches 4K canonical (3840×2160)
        """
        return resolution == CANONICAL_RESOLUTION

    @staticmethod
    def get_aspect_ratio(resolution: tuple[int, int]) -> float:
        """Get aspect ratio of a resolution.

        Args:
            resolution: Resolution (width, height) in pixels

        Returns:
            Aspect ratio (width / height)

        Example:
            >>> ResolutionConverter.get_aspect_ratio((1920, 1080))
            1.7777...
            >>> ResolutionConverter.get_aspect_ratio((3840, 2160))
            1.7777...
        """
        return resolution[0] / resolution[1]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def to_4k(
    x: float, y: float, source_resolution: tuple[int, int]
) -> tuple[float, float]:
    """Convenience function to convert coordinates to 4K canonical.

    Args:
        x: X coordinate in source resolution
        y: Y coordinate in source resolution
        source_resolution: Source (width, height) in pixels

    Returns:
        Tuple of (x_4k, y_4k) in 4K canonical pixels
    """
    return ResolutionConverter.scale_to_4k(x, y, source_resolution)


def from_4k(
    x_4k: float, y_4k: float, target_resolution: tuple[int, int]
) -> tuple[float, float]:
    """Convenience function to convert coordinates from 4K canonical.

    Args:
        x_4k: X coordinate in 4K canonical pixels
        y_4k: Y coordinate in 4K canonical pixels
        target_resolution: Target (width, height) in pixels

    Returns:
        Tuple of (x, y) in target resolution pixels
    """
    return ResolutionConverter.scale_from_4k(x_4k, y_4k, target_resolution)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ResolutionConverter",
    "to_4k",
    "from_4k",
]
