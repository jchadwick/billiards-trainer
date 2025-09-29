"""Utilities module for projector system."""

from .colors import (
    BilliardsPalette,
    ColorPalette,
    ColorSpace,
    ColorUtils,
    TrajectoryColors,
    apply_color_temperature,
    create_fade_effect,
    get_high_contrast_text_color,
)
from .geometry import (
    Circle,
    CoordinateSystem,
    CoordinateTransform,
    GeometryUtils,
    Line2D,
    Point2D,
    Rectangle,
    Vector2D,
    calculate_ball_collision_angles,
    calculate_trajectory_reflection,
    find_circle_tangent_points,
)
from .performance import PerformanceMonitor

__all__ = [
    # Color utilities
    "ColorUtils",
    "ColorPalette",
    "ColorSpace",
    "BilliardsPalette",
    "TrajectoryColors",
    "create_fade_effect",
    "get_high_contrast_text_color",
    "apply_color_temperature",
    # Geometry utilities
    "Point2D",
    "Vector2D",
    "Line2D",
    "Circle",
    "Rectangle",
    "GeometryUtils",
    "CoordinateSystem",
    "CoordinateTransform",
    "calculate_trajectory_reflection",
    "find_circle_tangent_points",
    "calculate_ball_collision_angles",
    # Performance monitoring
    "PerformanceMonitor",
]
