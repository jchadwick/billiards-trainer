"""Trajectory rendering system for the projector module.

This module provides comprehensive trajectory visualization capabilities including:
- Ball path rendering with various styles
- Collision point markers
- Ghost ball positioning
- Angle and power indicators
- Real-time trajectory updates
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.game_state import Vector2D
from core.physics.trajectory import CollisionType, Trajectory

from .renderer import BasicRenderer, Color, Colors, LineStyle, Point2D

logger = logging.getLogger(__name__)


class TrajectoryStyle(Enum):
    """Trajectory line rendering styles."""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    GRADIENT = "gradient"
    ARROW = "arrow"
    FADE = "fade"


class GhostBallStyle(Enum):
    """Ghost ball rendering styles."""

    OUTLINE = "outline"
    FILLED = "filled"
    TRANSPARENT = "transparent"
    PULSING = "pulsing"


class PowerIndicatorStyle(Enum):
    """Power indicator visualization styles."""

    BAR = "bar"
    CIRCLE = "circle"
    ARROW = "arrow"
    GRADIENT = "gradient"


@dataclass
class TrajectoryVisualConfig:
    """Configuration for trajectory visualization."""

    # Line rendering
    primary_color: Color = field(default_factory=lambda: Colors.GREEN)
    secondary_color: Color = field(default_factory=lambda: Colors.YELLOW)
    collision_color: Color = field(default_factory=lambda: Colors.RED)
    reflection_color: Color = field(default_factory=lambda: Colors.CYAN)

    # Line properties
    line_width: float = 3.0
    style: TrajectoryStyle = TrajectoryStyle.SOLID
    opacity: float = 0.8
    max_segments: int = 100

    # Collision markers
    collision_marker_radius: float = 15.0
    collision_marker_style: str = "cross"  # cross, circle, star
    show_collision_angle: bool = True

    # Ghost balls
    ghost_ball_style: GhostBallStyle = GhostBallStyle.TRANSPARENT
    ghost_ball_opacity: float = 0.4
    show_ghost_balls: bool = True

    # Power indicators
    power_indicator_style: PowerIndicatorStyle = PowerIndicatorStyle.GRADIENT
    show_power_indicator: bool = True
    power_scale: float = 2.0

    # Animation
    enable_animations: bool = True
    animation_speed: float = 1.0
    fade_in_duration: float = 0.3
    fade_out_duration: float = 0.5

    # Performance
    max_trajectory_points: int = 500
    update_interval: float = 0.016  # 60 FPS
    level_of_detail: bool = True


@dataclass
class TrajectoryRenderData:
    """Rendered trajectory data for a single frame."""

    trajectory: Trajectory
    line_segments: list[tuple[Point2D, Point2D]]
    collision_markers: list[tuple[Point2D, CollisionType]]
    ghost_balls: list[tuple[Point2D, float]]  # position, radius
    power_indicators: list[dict]
    confidence_regions: list[dict]
    animation_state: dict
    last_update: float


class TrajectoryRenderer:
    """Advanced trajectory rendering system for billiards projector.

    Provides comprehensive visualization of ball trajectories including:
    - Primary and secondary ball paths
    - Collision prediction markers
    - Ghost ball positioning for aiming
    - Power and angle indicators
    - Success probability visualization
    """

    def __init__(
        self, renderer: BasicRenderer, config: Optional[TrajectoryVisualConfig] = None
    ):
        """Initialize trajectory renderer.

        Args:
            renderer: Base renderer for drawing primitives
            config: Visual configuration settings
        """
        self.renderer = renderer
        self.config = config or TrajectoryVisualConfig()

        # Rendering state
        self._active_trajectories: dict[str, TrajectoryRenderData] = {}
        self._animation_time = 0.0
        self._last_frame_time = 0.0

        # Performance tracking
        self._render_stats = {
            "trajectories_rendered": 0,
            "segments_drawn": 0,
            "markers_drawn": 0,
            "total_render_time": 0.0,
            "last_frame_time": 0.0,
        }

        # Precomputed geometry for markers
        self._collision_markers = self._generate_collision_markers()

        logger.info("TrajectoryRenderer initialized")

    def update_trajectory(self, trajectory: Trajectory, fade_in: bool = True) -> None:
        """Update or add a trajectory for rendering.

        Args:
            trajectory: Trajectory data to render
            fade_in: Whether to animate trajectory appearance
        """
        current_time = time.time()

        # Convert trajectory to render data
        render_data = self._convert_trajectory_to_render_data(trajectory, current_time)

        # Set up fade-in animation if requested
        if fade_in and self.config.enable_animations:
            render_data.animation_state = {
                "fade_in_start": current_time,
                "fade_in_duration": self.config.fade_in_duration,
                "current_alpha": 0.0,
            }
        else:
            render_data.animation_state = {"current_alpha": self.config.opacity}

        self._active_trajectories[trajectory.ball_id] = render_data
        logger.debug(f"Updated trajectory for ball {trajectory.ball_id}")

    def remove_trajectory(self, ball_id: str, fade_out: bool = True) -> None:
        """Remove a trajectory from rendering.

        Args:
            ball_id: ID of ball whose trajectory to remove
            fade_out: Whether to animate trajectory disappearance
        """
        if ball_id not in self._active_trajectories:
            return

        if fade_out and self.config.enable_animations:
            # Start fade-out animation
            current_time = time.time()
            self._active_trajectories[ball_id].animation_state.update(
                {
                    "fade_out_start": current_time,
                    "fade_out_duration": self.config.fade_out_duration,
                    "fading_out": True,
                }
            )
        else:
            # Remove immediately
            del self._active_trajectories[ball_id]

        logger.debug(f"Removing trajectory for ball {ball_id}")

    def clear_all_trajectories(self, fade_out: bool = True) -> None:
        """Clear all trajectories.

        Args:
            fade_out: Whether to animate disappearance
        """
        ball_ids = list(self._active_trajectories.keys())
        for ball_id in ball_ids:
            self.remove_trajectory(ball_id, fade_out)

    def render_frame(self) -> None:
        """Render all active trajectories for current frame."""
        frame_start = time.time()
        self._animation_time = frame_start

        # Update animations and remove expired trajectories
        self._update_animations()

        # Reset frame stats
        self._render_stats["segments_drawn"] = 0
        self._render_stats["markers_drawn"] = 0

        # Render each active trajectory
        for _ball_id, render_data in self._active_trajectories.items():
            self._render_trajectory(render_data)

        # Update performance stats
        frame_time = time.time() - frame_start
        self._render_stats["last_frame_time"] = frame_time
        self._render_stats["total_render_time"] += frame_time
        self._render_stats["trajectories_rendered"] = len(self._active_trajectories)
        self._last_frame_time = frame_start

    def _convert_trajectory_to_render_data(
        self, trajectory: Trajectory, current_time: float
    ) -> TrajectoryRenderData:
        """Convert physics trajectory to renderable data."""
        # Extract line segments from trajectory points
        line_segments = []
        if len(trajectory.points) >= 2:
            for i in range(len(trajectory.points) - 1):
                start = Point2D(
                    trajectory.points[i].position.x, trajectory.points[i].position.y
                )
                end = Point2D(
                    trajectory.points[i + 1].position.x,
                    trajectory.points[i + 1].position.y,
                )
                line_segments.append((start, end))

        # Extract collision markers
        collision_markers = []
        for collision in trajectory.collisions:
            marker_pos = Point2D(collision.position.x, collision.position.y)
            collision_markers.append((marker_pos, collision.type))

        # Generate ghost ball positions
        ghost_balls = []
        if self.config.show_ghost_balls and trajectory.points:
            # Show ghost ball at final position if not pocketed
            if not trajectory.will_be_pocketed and trajectory.final_position:
                final_pos = Point2D(
                    trajectory.final_position.x, trajectory.final_position.y
                )
                # Use radius from initial state
                radius = trajectory.initial_state.radius
                ghost_balls.append((final_pos, radius))

        # Generate power indicators
        power_indicators = []
        if self.config.show_power_indicator and trajectory.points:
            initial_velocity = (
                trajectory.points[0].velocity if trajectory.points else Vector2D(0, 0)
            )
            speed = initial_velocity.magnitude()
            if speed > 0.1:  # Only show for meaningful velocities
                start_pos = Point2D(
                    trajectory.points[0].position.x, trajectory.points[0].position.y
                )
                power_indicators.append(
                    {
                        "type": self.config.power_indicator_style,
                        "position": start_pos,
                        "magnitude": speed,
                        "direction": math.atan2(initial_velocity.y, initial_velocity.x),
                    }
                )

        return TrajectoryRenderData(
            trajectory=trajectory,
            line_segments=line_segments,
            collision_markers=collision_markers,
            ghost_balls=ghost_balls,
            power_indicators=power_indicators,
            confidence_regions=[],
            animation_state={},
            last_update=current_time,
        )

    def _render_trajectory(self, render_data: TrajectoryRenderData) -> None:
        """Render a single trajectory's visual elements."""
        # Get current alpha for animations
        alpha = render_data.animation_state.get("current_alpha", self.config.opacity)
        if alpha <= 0:
            return

        # Render main trajectory line
        self._render_trajectory_line(render_data, alpha)

        # Render collision markers
        self._render_collision_markers(render_data, alpha)

        # Render ghost balls
        self._render_ghost_balls(render_data, alpha)

        # Render power indicators
        self._render_power_indicators(render_data, alpha)

        # Render success probability regions
        self._render_confidence_regions(render_data, alpha)

    def _render_trajectory_line(
        self, render_data: TrajectoryRenderData, alpha: float
    ) -> None:
        """Render the main trajectory path."""
        if not render_data.line_segments:
            return

        # Determine color based on trajectory success probability
        base_color = self.config.primary_color
        if render_data.trajectory.success_probability < 0.3:
            base_color = self.config.collision_color
        elif render_data.trajectory.success_probability < 0.6:
            base_color = self.config.secondary_color

        # Apply alpha
        line_color = base_color.with_alpha(alpha)

        # Set rendering properties
        self.renderer.set_color(line_color)
        self.renderer.set_line_width(self.config.line_width)

        # Convert trajectory style to line style
        line_style = self._convert_trajectory_style(self.config.style)

        # Render based on style
        if self.config.style == TrajectoryStyle.GRADIENT:
            self._render_gradient_line(render_data.line_segments, base_color, alpha)
        elif self.config.style == TrajectoryStyle.FADE:
            self._render_fade_line(render_data.line_segments, base_color, alpha)
        else:
            # Standard line rendering
            for start, end in render_data.line_segments:
                self.renderer.draw_line(start, end, style=line_style)
                self._render_stats["segments_drawn"] += 1

    def _render_collision_markers(
        self, render_data: TrajectoryRenderData, alpha: float
    ) -> None:
        """Render collision prediction markers."""
        for marker_pos, collision_type in render_data.collision_markers:
            # Choose color based on collision type
            if collision_type == CollisionType.BALL_BALL:
                color = Colors.ORANGE.with_alpha(alpha)
            elif collision_type == CollisionType.BALL_CUSHION:
                color = Colors.CYAN.with_alpha(alpha)
            elif collision_type == CollisionType.BALL_POCKET:
                color = Colors.GREEN.with_alpha(alpha)
            else:
                color = Colors.WHITE.with_alpha(alpha)

            # Render marker based on configured style
            if self.config.collision_marker_style == "cross":
                self._render_cross_marker(marker_pos, color)
            elif self.config.collision_marker_style == "circle":
                self._render_circle_marker(marker_pos, color)
            elif self.config.collision_marker_style == "star":
                self._render_star_marker(marker_pos, color)

            self._render_stats["markers_drawn"] += 1

    def _render_ghost_balls(
        self, render_data: TrajectoryRenderData, alpha: float
    ) -> None:
        """Render ghost ball positions."""
        if not self.config.show_ghost_balls:
            return

        for ghost_pos, radius in render_data.ghost_balls:
            ghost_alpha = self.config.ghost_ball_opacity * alpha
            ghost_color = Colors.CUE_BALL.with_alpha(ghost_alpha)

            if self.config.ghost_ball_style == GhostBallStyle.OUTLINE:
                self.renderer.draw_circle(
                    ghost_pos, radius, ghost_color, filled=False, outline_width=2.0
                )
            elif (
                self.config.ghost_ball_style == GhostBallStyle.FILLED
                or self.config.ghost_ball_style == GhostBallStyle.TRANSPARENT
            ):
                self.renderer.draw_circle(ghost_pos, radius, ghost_color, filled=True)
            elif self.config.ghost_ball_style == GhostBallStyle.PULSING:
                # Animate radius based on time
                pulse_factor = 0.8 + 0.2 * math.sin(self._animation_time * 3.0)
                pulsing_radius = radius * pulse_factor
                self.renderer.draw_circle(
                    ghost_pos,
                    pulsing_radius,
                    ghost_color,
                    filled=False,
                    outline_width=2.0,
                )

    def _render_power_indicators(
        self, render_data: TrajectoryRenderData, alpha: float
    ) -> None:
        """Render power/force indicators."""
        if not self.config.show_power_indicator:
            return

        for indicator in render_data.power_indicators:
            pos = indicator["position"]
            magnitude = indicator["magnitude"]
            direction = indicator["direction"]

            # Scale magnitude for visualization
            visual_magnitude = magnitude * self.config.power_scale

            # Create gradient color based on power
            if magnitude < 1.0:
                power_color = Colors.GREEN
            elif magnitude < 3.0:
                power_color = Colors.YELLOW
            else:
                power_color = Colors.RED

            power_color = power_color.with_alpha(alpha)

            if self.config.power_indicator_style == PowerIndicatorStyle.ARROW:
                self._render_power_arrow(pos, direction, visual_magnitude, power_color)
            elif self.config.power_indicator_style == PowerIndicatorStyle.BAR:
                self._render_power_bar(pos, magnitude, power_color)
            elif self.config.power_indicator_style == PowerIndicatorStyle.CIRCLE:
                self._render_power_circle(pos, magnitude, power_color)
            elif self.config.power_indicator_style == PowerIndicatorStyle.GRADIENT:
                self._render_power_gradient(
                    pos, direction, visual_magnitude, power_color
                )

    def _render_confidence_regions(
        self, render_data: TrajectoryRenderData, alpha: float
    ) -> None:
        """Render success probability and confidence regions."""
        # This would render uncertainty regions around the trajectory
        # For now, implement as a simple overlay at the final position
        if not render_data.trajectory.final_position:
            return

        success_prob = render_data.trajectory.success_probability
        if success_prob > 0.1:  # Only show if there's meaningful probability
            final_pos = Point2D(
                render_data.trajectory.final_position.x,
                render_data.trajectory.final_position.y,
            )

            # Color-code by probability
            if success_prob > 0.7:
                region_color = Colors.GREEN.with_alpha(alpha * 0.3)
            elif success_prob > 0.4:
                region_color = Colors.YELLOW.with_alpha(alpha * 0.3)
            else:
                region_color = Colors.RED.with_alpha(alpha * 0.3)

            # Draw confidence circle
            confidence_radius = 20.0 * (
                1.0 - success_prob
            )  # Larger circle = less confident
            self.renderer.draw_circle(
                final_pos, confidence_radius, region_color, filled=True
            )

    def _update_animations(self) -> None:
        """Update animation states and remove expired trajectories."""
        current_time = self._animation_time
        expired_trajectories = []

        for ball_id, render_data in self._active_trajectories.items():
            anim_state = render_data.animation_state

            # Update fade-in animation
            if "fade_in_start" in anim_state:
                elapsed = current_time - anim_state["fade_in_start"]
                duration = anim_state["fade_in_duration"]

                if elapsed >= duration:
                    # Fade-in complete
                    anim_state["current_alpha"] = self.config.opacity
                    del anim_state["fade_in_start"]
                    del anim_state["fade_in_duration"]
                else:
                    # Interpolate alpha
                    progress = elapsed / duration
                    anim_state["current_alpha"] = self.config.opacity * progress

            # Update fade-out animation
            if anim_state.get("fading_out", False):
                elapsed = current_time - anim_state["fade_out_start"]
                duration = anim_state["fade_out_duration"]

                if elapsed >= duration:
                    # Fade-out complete, mark for removal
                    expired_trajectories.append(ball_id)
                else:
                    # Interpolate alpha
                    progress = elapsed / duration
                    start_alpha = anim_state.get("current_alpha", self.config.opacity)
                    anim_state["current_alpha"] = start_alpha * (1.0 - progress)

        # Remove expired trajectories
        for ball_id in expired_trajectories:
            del self._active_trajectories[ball_id]

    # Helper methods for rendering specific elements

    def _convert_trajectory_style(self, style: TrajectoryStyle) -> LineStyle:
        """Convert trajectory style to basic line style."""
        style_map = {
            TrajectoryStyle.SOLID: LineStyle.SOLID,
            TrajectoryStyle.DASHED: LineStyle.DASHED,
            TrajectoryStyle.DOTTED: LineStyle.DOTTED,
            TrajectoryStyle.ARROW: LineStyle.ARROW,
            TrajectoryStyle.GRADIENT: LineStyle.SOLID,  # Special handling
            TrajectoryStyle.FADE: LineStyle.SOLID,  # Special handling
        }
        return style_map.get(style, LineStyle.SOLID)

    def _render_gradient_line(
        self, segments: list[tuple[Point2D, Point2D]], base_color: Color, alpha: float
    ) -> None:
        """Render line with color gradient along length."""
        total_segments = len(segments)
        if total_segments == 0:
            return

        for i, (start, end) in enumerate(segments):
            # Calculate gradient position (0.0 to 1.0)
            gradient_pos = i / max(1, total_segments - 1)

            # Interpolate from primary to secondary color
            r = base_color.r + gradient_pos * (
                self.config.secondary_color.r - base_color.r
            )
            g = base_color.g + gradient_pos * (
                self.config.secondary_color.g - base_color.g
            )
            b = base_color.b + gradient_pos * (
                self.config.secondary_color.b - base_color.b
            )

            segment_color = Color(r, g, b, alpha)
            self.renderer.draw_line(start, end, segment_color)
            self._render_stats["segments_drawn"] += 1

    def _render_fade_line(
        self, segments: list[tuple[Point2D, Point2D]], base_color: Color, alpha: float
    ) -> None:
        """Render line with alpha fade along length."""
        total_segments = len(segments)
        if total_segments == 0:
            return

        for i, (start, end) in enumerate(segments):
            # Calculate fade factor (1.0 to 0.3)
            fade_pos = i / max(1, total_segments - 1)
            segment_alpha = alpha * (1.0 - 0.7 * fade_pos)

            segment_color = base_color.with_alpha(segment_alpha)
            self.renderer.draw_line(start, end, segment_color)
            self._render_stats["segments_drawn"] += 1

    def _render_cross_marker(self, pos: Point2D, color: Color) -> None:
        """Render cross-shaped collision marker."""
        radius = self.config.collision_marker_radius

        # Horizontal line
        start_h = Point2D(pos.x - radius, pos.y)
        end_h = Point2D(pos.x + radius, pos.y)
        self.renderer.draw_line(start_h, end_h, color)

        # Vertical line
        start_v = Point2D(pos.x, pos.y - radius)
        end_v = Point2D(pos.x, pos.y + radius)
        self.renderer.draw_line(start_v, end_v, color)

    def _render_circle_marker(self, pos: Point2D, color: Color) -> None:
        """Render circular collision marker."""
        radius = self.config.collision_marker_radius
        self.renderer.draw_circle(pos, radius, color, filled=False, outline_width=3.0)

    def _render_power_gradient(
        self, pos: Point2D, direction: float, magnitude: float, color: Color
    ) -> None:
        """Render gradient-style power indicator."""
        # Draw multiple circles with decreasing alpha
        max_radius = min(magnitude * 3, 40.0)

        for i in range(5):
            radius = max_radius * (0.2 + 0.8 * i / 4)
            alpha = color.a * (1.0 - i / 5)
            circle_color = Color(color.r, color.g, color.b, alpha)
            self.renderer.draw_circle(pos, radius, circle_color, filled=True)

    def _generate_collision_markers(self) -> dict:
        """Pre-generate collision marker geometry for performance."""
        # This could pre-compute marker vertices for efficient rendering
        return {}

    # Public API methods

    def set_config(self, config: TrajectoryVisualConfig) -> None:
        """Update visual configuration."""
        self.config = config
        logger.debug("Trajectory renderer configuration updated")

    def get_render_stats(self) -> dict:
        """Get rendering performance statistics."""
        return self._render_stats.copy()

    def get_active_trajectory_count(self) -> int:
        """Get number of active trajectories."""
        return len(self._active_trajectories)

    def has_trajectory(self, ball_id: str) -> bool:
        """Check if trajectory exists for ball."""
        return ball_id in self._active_trajectories

    def get_trajectory_alpha(self, ball_id: str) -> float:
        """Get current alpha value for trajectory."""
        if ball_id not in self._active_trajectories:
            return 0.0
        return self._active_trajectories[ball_id].animation_state.get(
            "current_alpha", 0.0
        )
