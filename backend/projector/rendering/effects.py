"""Visual effects system for trajectory rendering.

This module provides advanced visual effects for enhancing trajectory visualization:
- Ball trail effects with physics-based rendering
- Collision impact animations
- Shot outcome indicators with success/failure feedback
- Dynamic color coding based on physics parameters
- Particle systems for enhanced visual feedback
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from backend.core.game_state import Vector2D
from backend.core.physics.trajectory import (
    CollisionType,
    PredictedCollision,
    Trajectory,
)

from .renderer import BasicRenderer, Color, Colors, Point2D

logger = logging.getLogger(__name__)


class EffectType(Enum):
    """Types of visual effects."""

    BALL_TRAIL = "ball_trail"
    COLLISION_IMPACT = "collision_impact"
    SUCCESS_INDICATOR = "success_indicator"
    FAILURE_INDICATOR = "failure_indicator"
    POWER_BURST = "power_burst"
    SPIN_VISUALIZATION = "spin_visualization"
    POCKET_ATTRACTION = "pocket_attraction"
    UNCERTAINTY_CLOUD = "uncertainty_cloud"


class AnimationState(Enum):
    """Animation lifecycle states."""

    STARTING = "starting"
    ACTIVE = "active"
    FADING = "fading"
    COMPLETE = "complete"


@dataclass
class Particle:
    """Single particle in a particle system."""

    position: Point2D
    velocity: Vector2D
    color: Color
    size: float
    life_time: float
    max_life: float
    gravity: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    fade_rate: float = 1.0
    size_decay: float = 1.0

    @property
    def age_ratio(self) -> float:
        """Get particle age as ratio (0.0 = new, 1.0 = dead)."""
        return min(1.0, self.life_time / self.max_life)

    @property
    def is_alive(self) -> bool:
        """Check if particle is still alive."""
        return self.life_time < self.max_life and self.size > 0.1

    def update(self, dt: float) -> None:
        """Update particle state."""
        self.life_time += dt

        # Update position
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt

        # Apply gravity
        self.velocity.x += self.gravity.x * dt
        self.velocity.y += self.gravity.y * dt

        # Apply fade and size decay
        1.0 - self.age_ratio
        self.color.a *= (1.0 - self.fade_rate * dt) if self.fade_rate > 0 else 1.0
        self.size *= (1.0 - self.size_decay * dt) if self.size_decay > 0 else 1.0


@dataclass
class Effect:
    """Base class for visual effects."""

    effect_type: EffectType
    position: Point2D
    start_time: float
    duration: float
    state: AnimationState = AnimationState.STARTING
    particles: list[Particle] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Get effect age in seconds."""
        return time.time() - self.start_time

    @property
    def progress(self) -> float:
        """Get effect progress (0.0 = start, 1.0 = complete)."""
        return min(1.0, self.age / self.duration) if self.duration > 0 else 1.0

    @property
    def is_complete(self) -> bool:
        """Check if effect has completed."""
        return self.state == AnimationState.COMPLETE or self.age >= self.duration

    def update(self, dt: float) -> None:
        """Update effect state."""
        # Update particles
        self.particles = [p for p in self.particles if p.is_alive]
        for particle in self.particles:
            particle.update(dt)

        # Update animation state based on progress
        progress = self.progress
        if progress < 0.1:
            self.state = AnimationState.STARTING
        elif progress < 0.9:
            self.state = AnimationState.ACTIVE
        elif progress < 1.0:
            self.state = AnimationState.FADING
        else:
            self.state = AnimationState.COMPLETE


class EffectsConfig:
    """Configuration for visual effects system."""

    def __init__(self):
        # Ball trail effects
        self.trail_enabled = True
        self.trail_length = 10  # Number of trail segments
        self.trail_fade_rate = 0.1
        self.trail_width_decay = 0.9
        self.trail_color_shift = True

        # Collision effects
        self.collision_effects_enabled = True
        self.collision_particle_count = 20
        self.collision_burst_size = 30.0
        self.collision_duration = 1.0

        # Success/failure indicators
        self.outcome_indicators_enabled = True
        self.success_color = Colors.GREEN
        self.failure_color = Colors.RED
        self.indicator_duration = 2.0

        # Power effects
        self.power_effects_enabled = True
        self.power_burst_particles = 15
        self.power_effect_scale = 1.5

        # Spin visualization
        self.spin_effects_enabled = True
        self.spin_trail_length = 5
        self.spin_particle_rate = 10.0

        # Performance settings
        self.max_active_effects = 50
        self.particle_limit = 200
        self.enable_gpu_effects = True


class EffectsSystem:
    """Advanced visual effects system for trajectory rendering.

    Manages particle systems, animations, and visual enhancements
    to create engaging feedback for billiards training.
    """

    def __init__(self, renderer: BasicRenderer, config: Optional[EffectsConfig] = None):
        """Initialize effects system.

        Args:
            renderer: Base renderer for drawing primitives
            config: Effects configuration
        """
        self.renderer = renderer
        self.config = config or EffectsConfig()

        # Active effects
        self._active_effects: list[Effect] = []
        self._effect_generators: dict[EffectType, Callable] = {}

        # Performance tracking
        self._stats = {
            "active_effects": 0,
            "active_particles": 0,
            "effects_created": 0,
            "particles_created": 0,
            "render_time": 0.0,
        }

        # Initialize effect generators
        self._setup_effect_generators()

        logger.info("EffectsSystem initialized")

    def _setup_effect_generators(self) -> None:
        """Setup effect generation functions."""
        self._effect_generators = {
            EffectType.BALL_TRAIL: self._create_ball_trail_effect,
            EffectType.COLLISION_IMPACT: self._create_collision_impact_effect,
            EffectType.SUCCESS_INDICATOR: self._create_success_indicator_effect,
            EffectType.FAILURE_INDICATOR: self._create_failure_indicator_effect,
            EffectType.POWER_BURST: self._create_power_burst_effect,
            EffectType.SPIN_VISUALIZATION: self._create_spin_visualization_effect,
            EffectType.POCKET_ATTRACTION: self._create_pocket_attraction_effect,
            EffectType.UNCERTAINTY_CLOUD: self._create_uncertainty_cloud_effect,
        }

    def create_ball_trail(
        self, trajectory: Trajectory, trail_length: Optional[int] = None
    ) -> None:
        """Create ball trail effect for trajectory.

        Args:
            trajectory: Ball trajectory to visualize
            trail_length: Number of trail segments (uses config default if None)
        """
        if not self.config.trail_enabled:
            return

        length = trail_length or self.config.trail_length
        if len(trajectory.points) < 2:
            return

        # Create trail effect for each segment
        for i in range(min(length, len(trajectory.points) - 1)):
            point = trajectory.points[i]
            position = Point2D(point.position.x, point.position.y)

            effect = self._effect_generators[EffectType.BALL_TRAIL](
                position=position,
                velocity=point.velocity,
                energy=point.energy,
                segment_index=i,
                total_segments=length,
            )

            self._add_effect(effect)

    def create_collision_impact(
        self, collision: PredictedCollision, intensity: float = 1.0
    ) -> None:
        """Create collision impact effect.

        Args:
            collision: Collision data
            intensity: Effect intensity multiplier
        """
        if not self.config.collision_effects_enabled:
            return

        position = Point2D(collision.position.x, collision.position.y)

        effect = self._effect_generators[EffectType.COLLISION_IMPACT](
            position=position,
            collision_type=collision.type,
            impact_velocity=collision.impact_velocity,
            intensity=intensity,
        )

        self._add_effect(effect)

    def create_success_indicator(self, position: Point2D, success_level: float) -> None:
        """Create success outcome indicator.

        Args:
            position: Position to show indicator
            success_level: Success probability (0.0 to 1.0)
        """
        if not self.config.outcome_indicators_enabled:
            return

        effect = self._effect_generators[EffectType.SUCCESS_INDICATOR](
            position=position, success_level=success_level
        )

        self._add_effect(effect)

    def create_failure_indicator(self, position: Point2D, failure_reason: str) -> None:
        """Create failure outcome indicator.

        Args:
            position: Position to show indicator
            failure_reason: Reason for failure
        """
        if not self.config.outcome_indicators_enabled:
            return

        effect = self._effect_generators[EffectType.FAILURE_INDICATOR](
            position=position, failure_reason=failure_reason
        )

        self._add_effect(effect)

    def create_power_burst(
        self, position: Point2D, power_level: float, direction: float
    ) -> None:
        """Create power burst effect.

        Args:
            position: Burst origin position
            power_level: Power magnitude
            direction: Burst direction in radians
        """
        if not self.config.power_effects_enabled:
            return

        effect = self._effect_generators[EffectType.POWER_BURST](
            position=position, power_level=power_level, direction=direction
        )

        self._add_effect(effect)

    def create_spin_visualization(
        self, position: Point2D, spin: Vector2D, ball_radius: float
    ) -> None:
        """Create spin visualization effect.

        Args:
            position: Ball position
            spin: Spin vector
            ball_radius: Ball radius
        """
        if not self.config.spin_effects_enabled or spin.magnitude() < 0.1:
            return

        effect = self._effect_generators[EffectType.SPIN_VISUALIZATION](
            position=position, spin=spin, ball_radius=ball_radius
        )

        self._add_effect(effect)

    def update_effects(self, dt: float) -> None:
        """Update all active effects.

        Args:
            dt: Time delta in seconds
        """
        # Update all effects
        for effect in self._active_effects:
            effect.update(dt)

        # Remove completed effects
        self._active_effects = [e for e in self._active_effects if not e.is_complete]

        # Enforce limits
        self._enforce_limits()

        # Update stats
        self._update_stats()

    def render_effects(self) -> None:
        """Render all active effects."""
        start_time = time.time()

        for effect in self._active_effects:
            self._render_effect(effect)

        self._stats["render_time"] = time.time() - start_time

    def clear_all_effects(self) -> None:
        """Clear all active effects."""
        self._active_effects.clear()

    # Effect creation methods

    def _create_ball_trail_effect(
        self,
        position: Point2D,
        velocity: Vector2D,
        energy: float,
        segment_index: int,
        total_segments: int,
    ) -> Effect:
        """Create ball trail effect."""
        effect = Effect(
            effect_type=EffectType.BALL_TRAIL,
            position=position,
            start_time=time.time(),
            duration=2.0,
            properties={
                "velocity": velocity,
                "energy": energy,
                "segment_index": segment_index,
                "total_segments": total_segments,
            },
        )

        # Create trail particles
        trail_alpha = 1.0 - (segment_index / max(1, total_segments - 1)) * 0.7
        speed = velocity.magnitude()

        # Color based on energy/speed
        if speed > 3.0:
            base_color = Colors.RED
        elif speed > 1.5:
            base_color = Colors.YELLOW
        else:
            base_color = Colors.GREEN

        trail_color = base_color.with_alpha(trail_alpha)

        # Create single particle for this trail segment
        particle = Particle(
            position=Point2D(position.x, position.y),
            velocity=Vector2D(velocity.x * 0.1, velocity.y * 0.1),
            color=trail_color,
            size=5.0 * (1.0 - segment_index / max(1, total_segments)),
            life_time=0.0,
            max_life=1.5,
            fade_rate=0.5,
        )

        effect.particles.append(particle)
        return effect

    def _create_collision_impact_effect(
        self,
        position: Point2D,
        collision_type: CollisionType,
        impact_velocity: float,
        intensity: float,
    ) -> Effect:
        """Create collision impact effect."""
        effect = Effect(
            effect_type=EffectType.COLLISION_IMPACT,
            position=position,
            start_time=time.time(),
            duration=self.config.collision_duration,
            properties={
                "collision_type": collision_type,
                "impact_velocity": impact_velocity,
                "intensity": intensity,
            },
        )

        # Choose colors based on collision type
        if collision_type == CollisionType.BALL_BALL:
            primary_color = Colors.ORANGE
            secondary_color = Colors.YELLOW
        elif collision_type == CollisionType.BALL_CUSHION:
            primary_color = Colors.CYAN
            secondary_color = Colors.BLUE
        elif collision_type == CollisionType.BALL_POCKET:
            primary_color = Colors.GREEN
            secondary_color = Colors.WHITE
        else:
            primary_color = Colors.WHITE
            secondary_color = Colors.WHITE

        # Create burst particles
        particle_count = int(self.config.collision_particle_count * intensity)
        burst_size = self.config.collision_burst_size * intensity

        for i in range(particle_count):
            angle = (i / particle_count) * 2 * math.pi
            speed = (0.5 + 0.5 * (i % 2)) * burst_size * impact_velocity * 0.1

            velocity = Vector2D(speed * math.cos(angle), speed * math.sin(angle))

            # Alternate colors
            color = primary_color if i % 2 == 0 else secondary_color

            particle = Particle(
                position=Point2D(position.x, position.y),
                velocity=velocity,
                color=Color(color.r, color.g, color.b, 0.8),
                size=3.0 + 2.0 * (i % 3),
                life_time=0.0,
                max_life=0.5 + 0.5 * (i / particle_count),
                gravity=Vector2D(0, -10.0),  # Slight downward gravity
                fade_rate=2.0,
                size_decay=1.5,
            )

            effect.particles.append(particle)

        return effect

    def _create_success_indicator_effect(
        self, position: Point2D, success_level: float
    ) -> Effect:
        """Create success indicator effect."""
        effect = Effect(
            effect_type=EffectType.SUCCESS_INDICATOR,
            position=position,
            start_time=time.time(),
            duration=self.config.indicator_duration,
            properties={"success_level": success_level},
        )

        # Color intensity based on success level
        green_intensity = success_level
        success_color = Color(0.0, green_intensity, 0.0, 0.8)

        # Create expanding ring particles
        for i in range(12):  # 12 particles in a circle
            angle = (i / 12) * 2 * math.pi
            radius = 20.0

            start_pos = Point2D(
                position.x + radius * math.cos(angle),
                position.y + radius * math.sin(angle),
            )

            # Particles move outward
            velocity = Vector2D(30.0 * math.cos(angle), 30.0 * math.sin(angle))

            particle = Particle(
                position=start_pos,
                velocity=velocity,
                color=success_color,
                size=4.0,
                life_time=0.0,
                max_life=1.5,
                fade_rate=0.8,
                size_decay=0.5,
            )

            effect.particles.append(particle)

        return effect

    def _create_failure_indicator_effect(
        self, position: Point2D, failure_reason: str
    ) -> Effect:
        """Create failure indicator effect."""
        effect = Effect(
            effect_type=EffectType.FAILURE_INDICATOR,
            position=position,
            start_time=time.time(),
            duration=self.config.indicator_duration,
            properties={"failure_reason": failure_reason},
        )

        # Red X pattern
        failure_color = Color(0.8, 0.0, 0.0, 0.9)

        # Create X pattern with particles
        for line in range(2):  # Two lines for X
            for i in range(8):  # Points along each line
                if line == 0:  # First diagonal
                    t = i / 7.0
                    x = position.x + (t - 0.5) * 40.0
                    y = position.y + (t - 0.5) * 40.0
                else:  # Second diagonal
                    t = i / 7.0
                    x = position.x + (t - 0.5) * 40.0
                    y = position.y - (t - 0.5) * 40.0

                particle = Particle(
                    position=Point2D(x, y),
                    velocity=Vector2D(0, 0),  # Static particles
                    color=failure_color,
                    size=3.0,
                    life_time=0.0,
                    max_life=2.0,
                    fade_rate=0.5,
                )

                effect.particles.append(particle)

        return effect

    def _create_power_burst_effect(
        self, position: Point2D, power_level: float, direction: float
    ) -> Effect:
        """Create power burst effect."""
        effect = Effect(
            effect_type=EffectType.POWER_BURST,
            position=position,
            start_time=time.time(),
            duration=1.0,
            properties={"power_level": power_level, "direction": direction},
        )

        # Color based on power level
        if power_level > 5.0:
            power_color = Colors.RED
        elif power_level > 2.0:
            power_color = Colors.YELLOW
        else:
            power_color = Colors.GREEN

        particle_count = int(self.config.power_burst_particles * min(2.0, power_level))

        for i in range(particle_count):
            # Spread particles around the main direction
            angle_spread = math.pi / 4  # 45 degree spread
            angle = direction + (i / particle_count - 0.5) * angle_spread

            speed = power_level * 20.0 * (0.8 + 0.4 * (i % 2))

            velocity = Vector2D(speed * math.cos(angle), speed * math.sin(angle))

            particle = Particle(
                position=Point2D(position.x, position.y),
                velocity=velocity,
                color=Color(power_color.r, power_color.g, power_color.b, 0.7),
                size=2.0 + power_level * 0.5,
                life_time=0.0,
                max_life=0.8,
                fade_rate=1.2,
                size_decay=1.0,
            )

            effect.particles.append(particle)

        return effect

    def _create_spin_visualization_effect(
        self, position: Point2D, spin: Vector2D, ball_radius: float
    ) -> Effect:
        """Create spin visualization effect."""
        effect = Effect(
            effect_type=EffectType.SPIN_VISUALIZATION,
            position=position,
            start_time=time.time(),
            duration=0.5,  # Short duration, continuously renewed
            properties={"spin": spin, "ball_radius": ball_radius},
        )

        spin_magnitude = spin.magnitude()
        spin_color = Colors.CYAN.with_alpha(0.6)

        # Create spiral particles around the ball
        num_particles = max(3, int(spin_magnitude * 2))
        for i in range(num_particles):
            angle = (i / num_particles) * 2 * math.pi
            radius = ball_radius * 1.2

            particle_pos = Point2D(
                position.x + radius * math.cos(angle),
                position.y + radius * math.sin(angle),
            )

            # Tangential velocity for spin effect
            tangent_velocity = Vector2D(
                -radius * math.sin(angle) * spin_magnitude * 0.1,
                radius * math.cos(angle) * spin_magnitude * 0.1,
            )

            particle = Particle(
                position=particle_pos,
                velocity=tangent_velocity,
                color=spin_color,
                size=2.0,
                life_time=0.0,
                max_life=0.3,
                fade_rate=3.0,
            )

            effect.particles.append(particle)

        return effect

    def _create_pocket_attraction_effect(self, position: Point2D, **kwargs) -> Effect:
        """Create pocket attraction effect."""
        return Effect(
            effect_type=EffectType.POCKET_ATTRACTION,
            position=position,
            start_time=time.time(),
            duration=1.0,
        )

    def _create_uncertainty_cloud_effect(self, position: Point2D, **kwargs) -> Effect:
        """Create uncertainty cloud effect."""
        return Effect(
            effect_type=EffectType.UNCERTAINTY_CLOUD,
            position=position,
            start_time=time.time(),
            duration=2.0,
        )

    # Rendering methods

    def _render_effect(self, effect: Effect) -> None:
        """Render a single effect."""
        # Render all particles in the effect
        for particle in effect.particles:
            self._render_particle(particle)

        # Render effect-specific elements
        if effect.effect_type == EffectType.SUCCESS_INDICATOR:
            self._render_success_indicator_base(effect)
        elif effect.effect_type == EffectType.FAILURE_INDICATOR:
            self._render_failure_indicator_base(effect)

    def _render_particle(self, particle: Particle) -> None:
        """Render a single particle."""
        if particle.size > 0.1 and particle.color.a > 0.01:
            self.renderer.draw_circle(
                particle.position, particle.size, particle.color, filled=True
            )

    def _render_success_indicator_base(self, effect: Effect) -> None:
        """Render base elements for success indicator."""
        progress = effect.progress
        alpha = 1.0 - progress

        # Draw expanding circle
        radius = 30.0 * progress
        circle_color = Colors.GREEN.with_alpha(alpha * 0.3)

        self.renderer.draw_circle(
            effect.position, radius, circle_color, filled=False, outline_width=3.0
        )

    def _render_failure_indicator_base(self, effect: Effect) -> None:
        """Render base elements for failure indicator."""
        progress = effect.progress
        alpha = 1.0 - progress

        # Draw pulsing X
        line_color = Colors.RED.with_alpha(alpha)
        size = 20.0

        # X lines
        self.renderer.draw_line(
            Point2D(effect.position.x - size, effect.position.y - size),
            Point2D(effect.position.x + size, effect.position.y + size),
            line_color,
            width=4.0,
        )

        self.renderer.draw_line(
            Point2D(effect.position.x - size, effect.position.y + size),
            Point2D(effect.position.x + size, effect.position.y - size),
            line_color,
            width=4.0,
        )

    # Utility methods

    def _add_effect(self, effect: Effect) -> None:
        """Add effect to active list."""
        self._active_effects.append(effect)
        self._stats["effects_created"] += 1
        self._stats["particles_created"] += len(effect.particles)

    def _enforce_limits(self) -> None:
        """Enforce performance limits on effects and particles."""
        # Limit total effects
        if len(self._active_effects) > self.config.max_active_effects:
            # Remove oldest effects
            self._active_effects = self._active_effects[
                -self.config.max_active_effects :
            ]

        # Limit total particles
        total_particles = sum(len(effect.particles) for effect in self._active_effects)
        if total_particles > self.config.particle_limit:
            # Remove particles from oldest effects
            for effect in self._active_effects:
                if total_particles <= self.config.particle_limit:
                    break
                particles_to_remove = min(
                    len(effect.particles), total_particles - self.config.particle_limit
                )
                effect.particles = effect.particles[particles_to_remove:]
                total_particles -= particles_to_remove

    def _update_stats(self) -> None:
        """Update performance statistics."""
        self._stats["active_effects"] = len(self._active_effects)
        self._stats["active_particles"] = sum(
            len(effect.particles) for effect in self._active_effects
        )

    # Public API

    def get_stats(self) -> dict:
        """Get effects system statistics."""
        return self._stats.copy()

    def set_config(self, config: EffectsConfig) -> None:
        """Update effects configuration."""
        self.config = config

    def get_active_effect_count(self) -> int:
        """Get number of active effects."""
        return len(self._active_effects)

    def get_active_particle_count(self) -> int:
        """Get total number of active particles."""
        return sum(len(effect.particles) for effect in self._active_effects)
