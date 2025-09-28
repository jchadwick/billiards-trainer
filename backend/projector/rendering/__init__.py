"""Rendering module for projector graphics with trajectory visualization."""

from .effects import (
    AnimationState,
    Effect,
    EffectsConfig,
    EffectsSystem,
    EffectType,
    Particle,
)
from .renderer import (
    BasicRenderer,
    BlendMode,
    Color,
    Colors,
    LineStyle,
    Point2D,
    RendererError,
    RenderStats,
)
from .trajectory import (
    GhostBallStyle,
    PowerIndicatorStyle,
    TrajectoryRenderData,
    TrajectoryRenderer,
    TrajectoryStyle,
    TrajectoryVisualConfig,
)

__all__ = [
    # Basic rendering
    "BasicRenderer",
    "Color",
    "Colors",
    "Point2D",
    "LineStyle",
    "BlendMode",
    "RenderStats",
    "RendererError",
    # Trajectory rendering
    "TrajectoryRenderer",
    "TrajectoryVisualConfig",
    "TrajectoryStyle",
    "GhostBallStyle",
    "PowerIndicatorStyle",
    "TrajectoryRenderData",
    # Effects system
    "EffectsSystem",
    "EffectsConfig",
    "Effect",
    "Particle",
    "EffectType",
    "AnimationState",
]
