"""Color management utilities for projector rendering.

This module provides comprehensive color handling, conversion, and manipulation
utilities for the projector system.
"""

import colorsys
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class ColorSpace(Enum):
    """Color space definitions."""

    RGB = "rgb"
    HSV = "hsv"
    HSL = "hsl"
    LAB = "lab"
    XYZ = "xyz"


@dataclass
class ColorPalette:
    """Color palette for consistent theming."""

    primary: tuple[int, int, int, int]
    secondary: tuple[int, int, int, int]
    accent: tuple[int, int, int, int]
    background: tuple[int, int, int, int]
    text: tuple[int, int, int, int]
    success: tuple[int, int, int, int]
    warning: tuple[int, int, int, int]
    error: tuple[int, int, int, int]


class ColorUtils:
    """Comprehensive color manipulation utilities."""

    @staticmethod
    def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
        """Convert RGB to HSV color space.

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)

        Returns:
            HSV tuple (h: 0-360, s: 0-1, v: 0-1)
        """
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        return h * 360, s, v

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
        """Convert HSV to RGB color space.

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            v: Value (0-1)

        Returns:
            RGB tuple (r, g, b) with values 0-255
        """
        h_norm = h / 360.0
        r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
        """Convert RGB to HSL color space.

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)

        Returns:
            HSL tuple (h: 0-360, s: 0-1, l: 0-1)
        """
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        return h * 360, s, l

    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
        """Convert HSL to RGB color space.

        Args:
            h: Hue (0-360)
            s: Saturation (0-1)
            l: Lightness (0-1)

        Returns:
            RGB tuple (r, g, b) with values 0-255
        """
        h_norm = h / 360.0
        r, g, b = colorsys.hls_to_rgb(h_norm, l, s)
        return int(r * 255), int(g * 255), int(b * 255)

    @staticmethod
    def blend_colors(
        color1: tuple[int, int, int, int],
        color2: tuple[int, int, int, int],
        factor: float,
    ) -> tuple[int, int, int, int]:
        """Blend two RGBA colors.

        Args:
            color1: First color (r, g, b, a)
            color2: Second color (r, g, b, a)
            factor: Blend factor (0.0 = color1, 1.0 = color2)

        Returns:
            Blended color (r, g, b, a)
        """
        factor = max(0.0, min(1.0, factor))
        inv_factor = 1.0 - factor

        r = int(color1[0] * inv_factor + color2[0] * factor)
        g = int(color1[1] * inv_factor + color2[1] * factor)
        b = int(color1[2] * inv_factor + color2[2] * factor)
        a = int(color1[3] * inv_factor + color2[3] * factor)

        return (r, g, b, a)

    @staticmethod
    def lighten_color(
        color: tuple[int, int, int, int], amount: float
    ) -> tuple[int, int, int, int]:
        """Lighten a color by a specified amount.

        Args:
            color: RGBA color tuple
            amount: Lighten amount (0.0 = no change, 1.0 = white)

        Returns:
            Lightened color
        """
        r, g, b, a = color
        h, s, l = ColorUtils.rgb_to_hsl(r, g, b)

        # Increase lightness
        l = min(1.0, l + amount)
        r_new, g_new, b_new = ColorUtils.hsl_to_rgb(h, s, l)

        return (r_new, g_new, b_new, a)

    @staticmethod
    def darken_color(
        color: tuple[int, int, int, int], amount: float
    ) -> tuple[int, int, int, int]:
        """Darken a color by a specified amount.

        Args:
            color: RGBA color tuple
            amount: Darken amount (0.0 = no change, 1.0 = black)

        Returns:
            Darkened color
        """
        r, g, b, a = color
        h, s, l = ColorUtils.rgb_to_hsl(r, g, b)

        # Decrease lightness
        l = max(0.0, l - amount)
        r_new, g_new, b_new = ColorUtils.hsl_to_rgb(h, s, l)

        return (r_new, g_new, b_new, a)

    @staticmethod
    def adjust_saturation(
        color: tuple[int, int, int, int], amount: float
    ) -> tuple[int, int, int, int]:
        """Adjust color saturation.

        Args:
            color: RGBA color tuple
            amount: Saturation adjustment (-1.0 to 1.0)

        Returns:
            Color with adjusted saturation
        """
        r, g, b, a = color
        h, s, l = ColorUtils.rgb_to_hsl(r, g, b)

        # Adjust saturation
        s = max(0.0, min(1.0, s + amount))
        r_new, g_new, b_new = ColorUtils.hsl_to_rgb(h, s, l)

        return (r_new, g_new, b_new, a)

    @staticmethod
    def get_complementary_color(
        color: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Get complementary color.

        Args:
            color: RGBA color tuple

        Returns:
            Complementary color
        """
        r, g, b, a = color
        h, s, v = ColorUtils.rgb_to_hsv(r, g, b)

        # Rotate hue by 180 degrees
        h_comp = (h + 180) % 360
        r_comp, g_comp, b_comp = ColorUtils.hsv_to_rgb(h_comp, s, v)

        return (r_comp, g_comp, b_comp, a)

    @staticmethod
    def create_gradient(
        color1: tuple[int, int, int, int], color2: tuple[int, int, int, int], steps: int
    ) -> list[tuple[int, int, int, int]]:
        """Create color gradient between two colors.

        Args:
            color1: Start color
            color2: End color
            steps: Number of gradient steps

        Returns:
            List of gradient colors
        """
        if steps <= 1:
            return [color1]

        gradient = []
        for i in range(steps):
            factor = i / (steps - 1)
            blended = ColorUtils.blend_colors(color1, color2, factor)
            gradient.append(blended)

        return gradient

    @staticmethod
    def get_luminance(color: tuple[int, int, int, int]) -> float:
        """Calculate relative luminance of a color.

        Args:
            color: RGBA color tuple

        Returns:
            Relative luminance (0.0 to 1.0)
        """
        r, g, b, _ = color

        # Convert to linear RGB
        def to_linear(c):
            c_norm = c / 255.0
            if c_norm <= 0.03928:
                return c_norm / 12.92
            else:
                return pow((c_norm + 0.055) / 1.055, 2.4)

        r_lin = to_linear(r)
        g_lin = to_linear(g)
        b_lin = to_linear(b)

        # Calculate luminance using ITU-R BT.709 coefficients
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    @staticmethod
    def get_contrast_ratio(
        color1: tuple[int, int, int, int], color2: tuple[int, int, int, int]
    ) -> float:
        """Calculate contrast ratio between two colors.

        Args:
            color1: First color
            color2: Second color

        Returns:
            Contrast ratio (1:1 to 21:1)
        """
        lum1 = ColorUtils.get_luminance(color1)
        lum2 = ColorUtils.get_luminance(color2)

        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
        """Convert hex color to RGBA.

        Args:
            hex_color: Hex color string (e.g., "#FF0000" or "FF0000")
            alpha: Alpha value (0-255)

        Returns:
            RGBA color tuple
        """
        hex_color = hex_color.lstrip("#")

        if len(hex_color) == 3:
            # Short form (e.g., "F0A")
            hex_color = "".join([c * 2 for c in hex_color])

        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")

        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b, alpha)
        except ValueError:
            raise ValueError(f"Invalid hex color: {hex_color}")

    @staticmethod
    def rgba_to_hex(color: tuple[int, int, int, int]) -> str:
        """Convert RGBA color to hex string.

        Args:
            color: RGBA color tuple

        Returns:
            Hex color string (e.g., "#FF0000")
        """
        r, g, b, _ = color
        return f"#{r:02X}{g:02X}{b:02X}"


class BilliardsPalette:
    """Color palettes specifically designed for billiards visualization."""

    @staticmethod
    def classic_green() -> ColorPalette:
        """Classic green billiards table palette."""
        return ColorPalette(
            primary=(34, 139, 34, 255),  # Forest green
            secondary=(0, 100, 0, 255),  # Dark green
            accent=(255, 215, 0, 255),  # Gold
            background=(0, 50, 0, 255),  # Very dark green
            text=(255, 255, 255, 255),  # White
            success=(0, 255, 0, 255),  # Bright green
            warning=(255, 255, 0, 255),  # Yellow
            error=(255, 0, 0, 255),  # Red
        )

    @staticmethod
    def blue_felt() -> ColorPalette:
        """Blue felt table palette."""
        return ColorPalette(
            primary=(25, 25, 112, 255),  # Midnight blue
            secondary=(0, 0, 139, 255),  # Dark blue
            accent=(255, 215, 0, 255),  # Gold
            background=(0, 0, 50, 255),  # Very dark blue
            text=(255, 255, 255, 255),  # White
            success=(0, 255, 0, 255),  # Bright green
            warning=(255, 255, 0, 255),  # Yellow
            error=(255, 0, 0, 255),  # Red
        )

    @staticmethod
    def burgundy_luxury() -> ColorPalette:
        """Luxury burgundy table palette."""
        return ColorPalette(
            primary=(128, 0, 32, 255),  # Burgundy
            secondary=(75, 0, 20, 255),  # Dark burgundy
            accent=(255, 215, 0, 255),  # Gold
            background=(40, 0, 10, 255),  # Very dark burgundy
            text=(255, 255, 255, 255),  # White
            success=(0, 255, 0, 255),  # Bright green
            warning=(255, 255, 0, 255),  # Yellow
            error=(255, 0, 0, 255),  # Red
        )


class TrajectoryColors:
    """Color schemes for trajectory visualization."""

    @staticmethod
    def get_velocity_color(
        velocity: float, max_velocity: float = 5.0
    ) -> tuple[int, int, int, int]:
        """Get color based on velocity magnitude.

        Args:
            velocity: Velocity magnitude
            max_velocity: Maximum expected velocity

        Returns:
            Color representing velocity intensity
        """
        # Normalize velocity (0.0 to 1.0)
        intensity = min(velocity / max_velocity, 1.0)

        if intensity < 0.33:
            # Green to yellow
            factor = intensity / 0.33
            return ColorUtils.blend_colors((0, 255, 0, 255), (255, 255, 0, 255), factor)
        elif intensity < 0.66:
            # Yellow to orange
            factor = (intensity - 0.33) / 0.33
            return ColorUtils.blend_colors(
                (255, 255, 0, 255), (255, 165, 0, 255), factor
            )
        else:
            # Orange to red
            factor = (intensity - 0.66) / 0.34
            return ColorUtils.blend_colors((255, 165, 0, 255), (255, 0, 0, 255), factor)

    @staticmethod
    def get_probability_color(probability: float) -> tuple[int, int, int, int]:
        """Get color based on success probability.

        Args:
            probability: Success probability (0.0 to 1.0)

        Returns:
            Color representing probability
        """
        if probability < 0.3:
            # Red (low probability)
            return (255, 0, 0, 255)
        elif probability < 0.7:
            # Yellow (medium probability)
            factor = (probability - 0.3) / 0.4
            return ColorUtils.blend_colors((255, 0, 0, 255), (255, 255, 0, 255), factor)
        else:
            # Green (high probability)
            factor = (probability - 0.7) / 0.3
            return ColorUtils.blend_colors((255, 255, 0, 255), (0, 255, 0, 255), factor)

    @staticmethod
    def get_ball_type_color(ball_type: str) -> tuple[int, int, int, int]:
        """Get color for different ball types.

        Args:
            ball_type: Ball type identifier

        Returns:
            Color representing ball type
        """
        ball_colors = {
            "cue": (255, 255, 255, 255),  # White
            "8": (0, 0, 0, 255),  # Black
            "solids": (255, 0, 0, 255),  # Red for solids group
            "stripes": (255, 255, 0, 255),  # Yellow for stripes group
            "target": (0, 255, 0, 255),  # Green for target ball
            "obstacle": (128, 128, 128, 255),  # Gray for obstacles
        }
        return ball_colors.get(ball_type, (255, 255, 255, 255))


# Convenience functions for common color operations


def get_high_contrast_text_color(
    background: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Get high contrast text color for given background.

    Args:
        background: Background color

    Returns:
        High contrast text color (white or black)
    """
    luminance = ColorUtils.get_luminance(background)
    if luminance > 0.5:
        return (0, 0, 0, 255)  # Black text on light background
    else:
        return (255, 255, 255, 255)  # White text on dark background
