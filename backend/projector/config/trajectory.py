"""Configuration management for trajectory rendering system.

This module provides comprehensive configuration management for all trajectory
rendering aspects including visual styles, effects, performance settings,
and user preferences with validation and persistence.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

from ...core.config.base import ConfigBase
from ..rendering.trajectory import (
    GhostBallStyle,
    PowerIndicatorStyle,
    TrajectoryStyle,
)

logger = logging.getLogger(__name__)


class TrajectoryQualityLevel(Enum):
    """Trajectory rendering quality levels."""

    PERFORMANCE = "performance"  # Optimized for speed
    BALANCED = "balanced"  # Good balance of quality and performance
    QUALITY = "quality"  # High quality rendering
    MAXIMUM = "maximum"  # Maximum quality, may impact performance


class AssistanceLevel(Enum):
    """User assistance levels."""

    DISABLED = "disabled"  # No trajectory assistance
    MINIMAL = "minimal"  # Basic trajectory only
    STANDARD = "standard"  # Standard assistance features
    ADVANCED = "advanced"  # Full assistance with all features
    EXPERT = "expert"  # All features with detailed analytics


@dataclass
class ColorConfig:
    """Color configuration with validation."""

    r: float = 0.0
    g: float = 1.0
    b: float = 0.0
    a: float = 0.8

    def __post_init__(self):
        """Validate color values."""
        for component in [self.r, self.g, self.b, self.a]:
            if not 0.0 <= component <= 1.0:
                raise ValueError(
                    f"Color component must be between 0.0 and 1.0, got {component}"
                )

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: int = 255) -> "ColorConfig":
        """Create from RGB values (0-255)."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @classmethod
    def from_hex(cls, hex_color: str) -> "ColorConfig":
        """Create from hex string."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            hex_color += "FF"
        elif len(hex_color) != 8:
            raise ValueError("Hex color must be #RRGGBB or #RRGGBBAA format")

        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return cls(r, g, b, a)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, self.a)


@dataclass
class TrajectoryRenderingConfig:
    """Configuration for trajectory line rendering."""

    # Basic line properties
    style: TrajectoryStyle = TrajectoryStyle.SOLID
    width: float = 3.0
    opacity: float = 0.8
    max_segments: int = 100

    # Colors for different trajectory types
    primary_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(0.0, 1.0, 0.0, 0.8)
    )  # Green
    secondary_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 1.0, 0.0, 0.8)
    )  # Yellow
    collision_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 0.0, 0.0, 0.8)
    )  # Red
    reflection_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(0.0, 1.0, 1.0, 0.8)
    )  # Cyan
    spin_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 0.0, 1.0, 0.6)
    )  # Magenta

    # Dynamic color coding
    enable_speed_coloring: bool = True
    enable_probability_coloring: bool = True
    speed_color_threshold_low: float = 1.0  # m/s
    speed_color_threshold_high: float = 3.0  # m/s

    # Animation settings
    enable_animations: bool = True
    fade_in_duration: float = 0.3
    fade_out_duration: float = 0.5
    animation_speed: float = 1.0

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.1 <= self.width <= 20.0:
            raise ValueError(
                f"Line width must be between 0.1 and 20.0, got {self.width}"
            )
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be between 0.0 and 1.0, got {self.opacity}")
        if not 10 <= self.max_segments <= 1000:
            raise ValueError(
                f"Max segments must be between 10 and 1000, got {self.max_segments}"
            )


@dataclass
class CollisionVisualizationConfig:
    """Configuration for collision visualization."""

    # Collision markers
    show_collision_markers: bool = True
    marker_style: str = "cross"  # cross, circle, star
    marker_radius: float = 15.0
    marker_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 0.5, 0.0, 0.9)
    )

    # Collision effects
    enable_impact_effects: bool = True
    particle_count: int = 20
    burst_size: float = 30.0
    effect_duration: float = 1.0

    # Angle indicators
    show_impact_angles: bool = False
    angle_line_length: float = 40.0
    angle_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 1.0, 1.0, 0.7)
    )

    def __post_init__(self):
        """Validate configuration values."""
        if not 5.0 <= self.marker_radius <= 50.0:
            raise ValueError(
                f"Marker radius must be between 5.0 and 50.0, got {self.marker_radius}"
            )
        if not 1 <= self.particle_count <= 100:
            raise ValueError(
                f"Particle count must be between 1 and 100, got {self.particle_count}"
            )


@dataclass
class GhostBallConfig:
    """Configuration for ghost ball visualization."""

    enabled: bool = True
    style: GhostBallStyle = GhostBallStyle.TRANSPARENT
    opacity: float = 0.4
    color: ColorConfig = field(default_factory=lambda: ColorConfig(1.0, 1.0, 1.0, 0.4))

    # Multiple ghost balls for uncertainty
    show_uncertainty_balls: bool = False
    uncertainty_count: int = 3
    uncertainty_spread: float = 10.0  # pixels

    # Animation
    enable_pulsing: bool = False
    pulse_speed: float = 2.0
    pulse_amplitude: float = 0.2

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(
                f"Ghost ball opacity must be between 0.0 and 1.0, got {self.opacity}"
            )
        if not 1 <= self.uncertainty_count <= 10:
            raise ValueError(
                f"Uncertainty count must be between 1 and 10, got {self.uncertainty_count}"
            )


@dataclass
class PowerIndicatorConfig:
    """Configuration for power/force indicators."""

    enabled: bool = True
    style: PowerIndicatorStyle = PowerIndicatorStyle.GRADIENT
    scale: float = 2.0
    max_size: float = 50.0

    # Color coding by power level
    low_power_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(0.0, 1.0, 0.0, 0.8)
    )  # Green
    medium_power_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 1.0, 0.0, 0.8)
    )  # Yellow
    high_power_color: ColorConfig = field(
        default_factory=lambda: ColorConfig(1.0, 0.0, 0.0, 0.8)
    )  # Red

    # Thresholds for color changes
    medium_power_threshold: float = 2.0  # m/s
    high_power_threshold: float = 5.0  # m/s

    # Animation
    enable_animation: bool = True
    animation_duration: float = 0.5

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.1 <= self.scale <= 10.0:
            raise ValueError(
                f"Power scale must be between 0.1 and 10.0, got {self.scale}"
            )
        if not 10.0 <= self.max_size <= 200.0:
            raise ValueError(
                f"Max size must be between 10.0 and 200.0, got {self.max_size}"
            )


@dataclass
class EffectsConfig:
    """Configuration for visual effects system."""

    # Ball trails
    enable_ball_trails: bool = True
    trail_length: int = 10
    trail_fade_rate: float = 0.1
    trail_width_decay: float = 0.9

    # Particle effects
    enable_particle_effects: bool = True
    max_particles: int = 200
    particle_lifetime_base: float = 1.0

    # Success/failure indicators
    enable_outcome_indicators: bool = True
    indicator_duration: float = 2.0
    success_threshold: float = 0.5

    # Spin visualization
    enable_spin_effects: bool = True
    spin_particle_rate: float = 10.0
    min_spin_threshold: float = 0.1

    # Performance limits
    max_active_effects: int = 50
    effect_quality_scaling: bool = True  # Reduce quality under load

    def __post_init__(self):
        """Validate configuration values."""
        if not 1 <= self.trail_length <= 50:
            raise ValueError(
                f"Trail length must be between 1 and 50, got {self.trail_length}"
            )
        if not 10 <= self.max_particles <= 1000:
            raise ValueError(
                f"Max particles must be between 10 and 1000, got {self.max_particles}"
            )


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # Quality settings
    quality_level: TrajectoryQualityLevel = TrajectoryQualityLevel.BALANCED
    auto_quality_adjustment: bool = True
    target_fps: int = 60
    min_fps_threshold: float = 45.0

    # Rendering optimizations
    enable_level_of_detail: bool = True
    max_trajectory_points: int = 500
    trajectory_simplification_threshold: float = 1.0  # pixels
    enable_frustum_culling: bool = True

    # Update rates
    trajectory_update_rate: float = 30.0  # Hz
    effects_update_rate: float = 60.0  # Hz
    stats_update_interval: float = 1.0  # seconds

    # Memory management
    max_cached_trajectories: int = 10
    cache_cleanup_interval: float = 30.0  # seconds

    def __post_init__(self):
        """Validate configuration values."""
        if not 10 <= self.target_fps <= 120:
            raise ValueError(
                f"Target FPS must be between 10 and 120, got {self.target_fps}"
            )
        if not 100 <= self.max_trajectory_points <= 2000:
            raise ValueError(
                f"Max trajectory points must be between 100 and 2000, got {self.max_trajectory_points}"
            )


@dataclass
class UserPreferencesConfig:
    """Configuration for user preferences and assistance."""

    # Assistance level
    assistance_level: AssistanceLevel = AssistanceLevel.STANDARD

    # Trajectory display preferences
    show_primary_trajectory: bool = True
    show_collision_trajectories: bool = True
    show_reflection_paths: bool = True
    show_spin_effects: bool = True
    max_trajectory_bounces: int = 3

    # Information display
    show_success_probability: bool = True
    show_angle_measurements: bool = False
    show_velocity_indicators: bool = True
    show_alternative_shots: bool = False

    # Training modes
    beginner_mode: bool = False
    expert_mode: bool = False
    practice_mode: bool = True
    competition_mode: bool = False

    # Accessibility
    high_contrast_mode: bool = False
    colorblind_friendly: bool = False
    large_text_mode: bool = False

    def get_effective_settings(self) -> dict[str, bool]:
        """Get effective settings based on assistance level."""
        base_settings = {
            "show_primary_trajectory": self.show_primary_trajectory,
            "show_collision_trajectories": self.show_collision_trajectories,
            "show_reflection_paths": self.show_reflection_paths,
            "show_spin_effects": self.show_spin_effects,
            "show_success_probability": self.show_success_probability,
            "show_angle_measurements": self.show_angle_measurements,
            "show_velocity_indicators": self.show_velocity_indicators,
            "show_alternative_shots": self.show_alternative_shots,
        }

        # Override based on assistance level
        if self.assistance_level == AssistanceLevel.DISABLED:
            return {key: False for key in base_settings}
        elif self.assistance_level == AssistanceLevel.MINIMAL:
            return {
                **{key: False for key in base_settings},
                "show_primary_trajectory": True,
            }
        elif self.assistance_level == AssistanceLevel.ADVANCED:
            return {
                **base_settings,
                "show_alternative_shots": True,
                "show_angle_measurements": True,
            }
        elif self.assistance_level == AssistanceLevel.EXPERT:
            return {key: True for key in base_settings}

        return base_settings


@dataclass
class TrajectoryConfig(ConfigBase):
    """Complete trajectory rendering configuration."""

    # Core components
    rendering: TrajectoryRenderingConfig = field(
        default_factory=TrajectoryRenderingConfig
    )
    collisions: CollisionVisualizationConfig = field(
        default_factory=CollisionVisualizationConfig
    )
    ghost_balls: GhostBallConfig = field(default_factory=GhostBallConfig)
    power_indicators: PowerIndicatorConfig = field(default_factory=PowerIndicatorConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    user_preferences: UserPreferencesConfig = field(
        default_factory=UserPreferencesConfig
    )

    # Global settings
    enabled: bool = True
    debug_mode: bool = False
    log_performance: bool = False

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        try:
            # Validate each component
            components = [
                self.rendering,
                self.collisions,
                self.ghost_balls,
                self.power_indicators,
                self.effects,
                self.performance,
            ]

            for component in components:
                # This will trigger __post_init__ validation
                component.__post_init__()

        except ValueError as e:
            errors.append(str(e))

        return errors

    def apply_theme(self, theme: str) -> None:
        """Apply a visual theme to the configuration."""
        themes = {
            "classic": {
                "primary_color": ColorConfig.from_hex("#00FF00"),
                "secondary_color": ColorConfig.from_hex("#FFFF00"),
                "collision_color": ColorConfig.from_hex("#FF0000"),
            },
            "neon": {
                "primary_color": ColorConfig.from_hex("#00FFFF"),
                "secondary_color": ColorConfig.from_hex("#FF00FF"),
                "collision_color": ColorConfig.from_hex("#FF8000"),
            },
            "pastel": {
                "primary_color": ColorConfig.from_hex("#80FF80"),
                "secondary_color": ColorConfig.from_hex("#FFFF80"),
                "collision_color": ColorConfig.from_hex("#FF8080"),
            },
            "high_contrast": {
                "primary_color": ColorConfig.from_hex("#FFFFFF"),
                "secondary_color": ColorConfig.from_hex("#FFFF00"),
                "collision_color": ColorConfig.from_hex("#FF0000"),
            },
        }

        if theme in themes:
            theme_colors = themes[theme]
            for attr, color in theme_colors.items():
                if hasattr(self.rendering, attr):
                    setattr(self.rendering, attr, color)

    def optimize_for_performance(self) -> None:
        """Optimize configuration for better performance."""
        self.performance.quality_level = TrajectoryQualityLevel.PERFORMANCE
        self.performance.max_trajectory_points = 200
        self.effects.enable_particle_effects = False
        self.effects.max_particles = 50
        self.rendering.enable_animations = False
        self.rendering.max_segments = 50

    def optimize_for_quality(self) -> None:
        """Optimize configuration for best visual quality."""
        self.performance.quality_level = TrajectoryQualityLevel.MAXIMUM
        self.performance.max_trajectory_points = 1000
        self.effects.enable_particle_effects = True
        self.effects.max_particles = 500
        self.rendering.enable_animations = True
        self.rendering.max_segments = 200

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryConfig":
        """Create configuration from dictionary."""
        # Reconstruct nested dataclasses
        config = cls()

        if "rendering" in data:
            config.rendering = TrajectoryRenderingConfig(**data["rendering"])
        if "collisions" in data:
            config.collisions = CollisionVisualizationConfig(**data["collisions"])
        if "ghost_balls" in data:
            config.ghost_balls = GhostBallConfig(**data["ghost_balls"])
        if "power_indicators" in data:
            config.power_indicators = PowerIndicatorConfig(**data["power_indicators"])
        if "effects" in data:
            config.effects = EffectsConfig(**data["effects"])
        if "performance" in data:
            config.performance = PerformanceConfig(**data["performance"])
        if "user_preferences" in data:
            config.user_preferences = UserPreferencesConfig(**data["user_preferences"])

        # Set global settings
        config.enabled = data.get("enabled", True)
        config.debug_mode = data.get("debug_mode", False)
        config.log_performance = data.get("log_performance", False)

        return config

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Trajectory configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "TrajectoryConfig":
        """Load configuration from JSON file."""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}. Using defaults.")
            return cls()

        try:
            with open(filepath) as f:
                data = json.load(f)

            config = cls.from_dict(data)
            logger.info(f"Trajectory configuration loaded from {filepath}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return cls()


# Preset configurations for common use cases
class TrajectoryConfigPresets:
    """Predefined trajectory configuration presets."""

    @staticmethod
    def beginner() -> TrajectoryConfig:
        """Configuration optimized for beginners."""
        config = TrajectoryConfig()
        config.user_preferences.assistance_level = AssistanceLevel.ADVANCED
        config.user_preferences.beginner_mode = True
        config.rendering.width = 4.0  # Thicker lines
        config.ghost_balls.opacity = 0.6  # More visible ghost balls
        config.power_indicators.enabled = True
        config.collisions.show_collision_markers = True
        return config

    @staticmethod
    def expert() -> TrajectoryConfig:
        """Configuration for expert players."""
        config = TrajectoryConfig()
        config.user_preferences.assistance_level = AssistanceLevel.MINIMAL
        config.user_preferences.expert_mode = True
        config.rendering.width = 2.0  # Thinner lines
        config.ghost_balls.opacity = 0.2  # Subtle ghost balls
        config.effects.enable_ball_trails = False
        config.collisions.show_collision_markers = False
        return config

    @staticmethod
    def practice() -> TrajectoryConfig:
        """Configuration for practice sessions."""
        config = TrajectoryConfig()
        config.user_preferences.practice_mode = True
        config.user_preferences.show_success_probability = True
        config.user_preferences.show_alternative_shots = True
        config.effects.enable_outcome_indicators = True
        return config

    @staticmethod
    def competition() -> TrajectoryConfig:
        """Configuration for competition/demo mode."""
        config = TrajectoryConfig()
        config.user_preferences.competition_mode = True
        config.user_preferences.assistance_level = AssistanceLevel.DISABLED
        config.effects.enable_ball_trails = True
        config.effects.enable_particle_effects = True
        config.rendering.enable_animations = True
        return config

    @staticmethod
    def performance() -> TrajectoryConfig:
        """Configuration optimized for performance."""
        config = TrajectoryConfig()
        config.optimize_for_performance()
        return config

    @staticmethod
    def quality() -> TrajectoryConfig:
        """Configuration optimized for visual quality."""
        config = TrajectoryConfig()
        config.optimize_for_quality()
        return config
