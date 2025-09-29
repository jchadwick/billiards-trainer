"""Projector Module - Display and rendering for pool table projection.

This module provides the display foundation and rendering capabilities
for projecting visual overlays onto the pool table surface.

Main Components:
- DisplayManager: Handles display initialization and window management
- BasicRenderer: Provides shape rendering and visual effects
- ProjectorModule: Main interface (from main.py)

Features:
- OpenGL/pygame-based display management
- Multi-monitor support with fullscreen and windowed modes
- Basic shape rendering (lines, circles, rectangles)
- Color management and visual effects
- Coordinate transformation system
- Performance monitoring and error handling
"""

import logging
from typing import Any

# Core display and rendering
from .display import (
    DisplayConfig,
    DisplayError,
    DisplayInfo,
    DisplayManager,
    DisplayMode,
)

# Main projector interface
from .main import CalibrationPoints, Circle
from .main import DisplayMode as MainDisplayMode
from .main import Line
from .main import LineStyle as MainLineStyle
from .main import Point2D as MainPoint2D
from .main import ProjectorModule, RenderFrame, RenderQuality, Text
from .rendering import (
    BasicRenderer,
    BlendMode,
    Color,
    Colors,
    LineStyle,
    Point2D,
    RendererError,
    RenderStats,
)
from .rendering.text import (
    FontDescriptor,
    FontWeight,
    TextAlign,
    TextRenderer,
    TextStyle,
    VerticalAlign,
    create_debug_text_style,
    create_error_text_style,
    create_heading_text_style,
    create_info_text_style,
)
from .utils import (
    BilliardsPalette,
    ColorPalette,
    ColorUtils,
    GeometryUtils,
    TrajectoryColors,
)

# Set up module logging
logger = logging.getLogger(__name__)


def create_projector(config: dict[str, Any]) -> ProjectorModule:
    """Create and initialize a projector module instance.

    Args:
        config: Configuration dictionary for the projector

    Returns:
        Initialized ProjectorModule instance

    Raises:
        DisplayError: If initialization fails
    """
    try:
        projector = ProjectorModule(config)
        logger.info("Projector module created successfully")
        return projector
    except Exception as e:
        logger.error(f"Failed to create projector module: {e}")
        raise DisplayError(f"Projector creation failed: {e}")


def get_available_displays() -> list[DisplayInfo]:
    """Get information about available display devices.

    Returns:
        List of available display devices
    """
    try:
        # Create temporary display manager to detect displays
        display_manager = DisplayManager()
        return display_manager.display_info
    except Exception as e:
        logger.warning(f"Failed to detect displays: {e}")
        return []


# Export main classes and functions
__all__ = [
    # Main interface
    "ProjectorModule",
    "create_projector",
    # Display management
    "DisplayManager",
    "DisplayMode",
    "DisplayConfig",
    "DisplayInfo",
    "DisplayError",
    "get_available_displays",
    # Rendering
    "BasicRenderer",
    "Color",
    "Colors",
    "Point2D",
    "LineStyle",
    "BlendMode",
    "RenderStats",
    "RendererError",
    # Text rendering
    "TextRenderer",
    "TextStyle",
    "FontDescriptor",
    "FontWeight",
    "TextAlign",
    "VerticalAlign",
    "create_info_text_style",
    "create_heading_text_style",
    "create_debug_text_style",
    "create_error_text_style",
    # Specification data classes
    "CalibrationPoints",
    "RenderFrame",
    "RenderQuality",
    "Line",
    "Circle",
    "Text",
    # Utilities
    "ColorUtils",
    "ColorPalette",
    "BilliardsPalette",
    "TrajectoryColors",
    "GeometryUtils",
    # Compatibility exports from main.py
    "MainDisplayMode",
    "MainLineStyle",
    "MainPoint2D",
]


# Module version
__version__ = "1.0.0"
