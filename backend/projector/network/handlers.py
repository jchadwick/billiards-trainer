"""Message handlers for projector WebSocket client."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

# Import core models with fallback for different import contexts
try:
    from ...core.game_state import BallState
    from ...core.physics.trajectory import Trajectory
except ImportError:
    # If running from the backend directory directly
    from core.game_state import BallState
    from core.physics.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Try to import rendering modules, use fallbacks if not available
try:
    from ..rendering.renderer import Color, Colors, Point2D
except ImportError:
    # Create fallback classes for testing
    class Color:
        def __init__(self, r, g, b, a=1.0):
            self.r = r
            self.g = g
            self.b = b
            self.a = a

        def with_alpha(self, alpha):
            return Color(self.r, self.g, self.b, alpha)

        def to_dict(self):
            return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    class Point2D:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        def to_tuple(self):
            return (self.x, self.y)

    class Colors:
        RED = Color(1.0, 0.0, 0.0, 1.0)
        GREEN = Color(0.0, 1.0, 0.0, 1.0)
        BLUE = Color(0.0, 0.0, 1.0, 1.0)
        YELLOW = Color(1.0, 1.0, 0.0, 1.0)
        WHITE = Color(1.0, 1.0, 1.0, 1.0)
        BLACK = Color(0.0, 0.0, 0.0, 1.0)
        CYAN = Color(0.0, 1.0, 1.0, 1.0)
        ORANGE = Color(1.0, 0.5, 0.0, 1.0)


class ProjectorInterface(Protocol):
    """Protocol for projector module interface."""

    def render_trajectory(self, trajectory: Trajectory, fade_in: bool = True) -> None:
        """Render a complete trajectory with physics data."""
        ...

    def update_ball_state(self, ball_state: BallState) -> None:
        """Update ball state for real-time effects."""
        ...

    def clear_display(self) -> None:
        """Clear all rendered content."""
        ...

    def remove_trajectory(self, ball_id: str, fade_out: bool = True) -> None:
        """Remove trajectory for specific ball."""
        ...

    def render_success_indicator(
        self, position: Any, success_probability: float
    ) -> None:
        """Render success probability indicator."""
        ...

    def set_trajectory_config(self, config: Any) -> None:
        """Update trajectory rendering configuration."""
        ...


@dataclass
class HandlerConfig:
    """Configuration for message handlers."""

    # Frame handling
    enable_frame_processing: bool = False  # Projector typically doesn't need frames
    max_frame_rate: float = 30.0

    # State handling
    enable_ball_tracking: bool = True
    enable_cue_tracking: bool = True
    enable_table_updates: bool = True

    # Trajectory handling
    enable_trajectory_rendering: bool = True
    trajectory_fade_in: bool = True
    trajectory_fade_out: bool = True
    clear_old_trajectories: bool = True

    # Alert handling
    enable_alert_display: bool = True
    alert_display_duration: float = 5.0

    # Configuration handling
    enable_config_updates: bool = True
    auto_apply_config: bool = True

    # Performance
    debounce_state_updates: bool = True
    debounce_interval: float = 0.1  # seconds


class ProjectorMessageHandlers:
    """Message handlers for projector WebSocket client.

    Handles incoming messages from the backend API and translates them
    into appropriate projector actions for real-time display updates.
    """

    def __init__(
        self, projector: ProjectorInterface, config: Optional[HandlerConfig] = None
    ):
        """Initialize message handlers.

        Args:
            projector: Projector module instance
            config: Handler configuration
        """
        self.projector = projector
        self.config = config or HandlerConfig()

        # State tracking
        self.last_game_state: Optional[dict[str, Any]] = None
        self.last_state_update: Optional[float] = None
        self.active_trajectories: dict[str, dict[str, Any]] = {}
        self.active_alerts: list[dict[str, Any]] = []

        # Performance tracking
        self.handler_stats = {
            "frames_processed": 0,
            "states_processed": 0,
            "trajectories_processed": 0,
            "alerts_processed": 0,
            "config_updates_processed": 0,
            "errors": 0,
        }

        logger.info("Projector message handlers initialized")

    async def handle_frame(self, data: dict[str, Any]) -> None:
        """Handle frame data from backend.

        Note: Projector typically doesn't need to process frames as it
        generates its own display output. This handler is mainly for
        monitoring and debugging purposes.

        Args:
            data: Frame data containing image, dimensions, etc.
        """
        if not self.config.enable_frame_processing:
            return

        try:
            width = data.get("width", 0)
            height = data.get("height", 0)
            fps = data.get("fps", 0.0)
            quality = data.get("quality", 0)

            logger.debug(
                f"Frame received: {width}x{height} @ {fps:.1f} FPS, quality {quality}"
            )
            self.handler_stats["frames_processed"] += 1

        except Exception as e:
            logger.error(f"Error handling frame data: {e}")
            self.handler_stats["errors"] += 1

    async def handle_state(self, data: dict[str, Any]) -> None:
        """Handle game state updates from backend.

        Updates projector with current ball positions, cue position,
        and table state for real-time tracking effects.

        Args:
            data: Game state data with balls, cue, table info
        """
        try:
            import time

            current_time = time.time()

            # Debounce state updates if enabled
            if (
                self.config.debounce_state_updates
                and self.last_state_update
                and current_time - self.last_state_update
                < self.config.debounce_interval
            ):
                return

            self.last_state_update = current_time

            # Extract game state components
            balls_data = data.get("balls", [])
            cue_data = data.get("cue")
            table_data = data.get("table")

            # Process ball states
            if self.config.enable_ball_tracking and balls_data:
                await self._process_ball_states(balls_data)

            # Process cue state
            if self.config.enable_cue_tracking and cue_data:
                await self._process_cue_state(cue_data)

            # Process table state
            if self.config.enable_table_updates and table_data:
                await self._process_table_state(table_data)

            self.last_game_state = data
            self.handler_stats["states_processed"] += 1

            logger.debug(f"Processed game state with {len(balls_data)} balls")

        except Exception as e:
            logger.error(f"Error handling game state: {e}")
            self.handler_stats["errors"] += 1

    async def handle_trajectory(self, data: dict[str, Any]) -> None:
        """Handle trajectory prediction data from backend.

        Renders predicted ball trajectories on the projector display
        with appropriate visual effects and confidence indicators.

        Args:
            data: Trajectory data with lines, collisions, confidence
        """
        if not self.config.enable_trajectory_rendering:
            return

        try:
            # Extract trajectory information
            lines = data.get("lines", [])
            collisions = data.get("collisions", [])
            confidence = data.get("confidence", 1.0)
            data.get("calculation_time_ms", 0.0)

            # Create trajectory ID (use timestamp if not provided)
            trajectory_id = data.get("id", f"traj_{int(time.time() * 1000)}")

            # Clear old trajectories if enabled
            if self.config.clear_old_trajectories:
                await self._clear_old_trajectories(trajectory_id)

            # Convert trajectory data to internal format
            trajectory = await self._convert_trajectory_data(data)

            if trajectory:
                # Render trajectory with fade-in effect
                self.projector.render_trajectory(
                    trajectory, fade_in=self.config.trajectory_fade_in
                )

                # Store trajectory for tracking
                self.active_trajectories[trajectory_id] = {
                    "data": data,
                    "timestamp": time.time(),
                    "confidence": confidence,
                }

                # Render collision predictions with confidence indicators
                for collision in collisions:
                    await self._render_collision_prediction(collision, confidence)

                logger.debug(
                    f"Rendered trajectory {trajectory_id}: {len(lines)} lines, "
                    f"{len(collisions)} collisions, {confidence:.2f} confidence"
                )

            self.handler_stats["trajectories_processed"] += 1

        except Exception as e:
            logger.error(f"Error handling trajectory data: {e}")
            self.handler_stats["errors"] += 1

    async def handle_alert(self, data: dict[str, Any]) -> None:
        """Handle alert/notification messages from backend.

        Displays alerts on the projector as visual overlays with
        appropriate severity levels and automatic dismissal.

        Args:
            data: Alert data with level, message, code, details
        """
        if not self.config.enable_alert_display:
            return

        try:
            level = data.get("level", "info")
            message = data.get("message", "No message")
            code = data.get("code", "NO_CODE")
            details = data.get("details", {})

            # Create alert entry
            alert = {
                "level": level,
                "message": message,
                "code": code,
                "details": details,
                "timestamp": time.time(),
            }

            # Add to active alerts
            self.active_alerts.append(alert)

            # Display alert based on severity
            await self._display_alert(alert)

            # Schedule alert removal
            if self.config.alert_display_duration > 0:
                import asyncio

                asyncio.create_task(
                    self._remove_alert_after_delay(
                        alert, self.config.alert_display_duration
                    )
                )

            self.handler_stats["alerts_processed"] += 1

            logger.info(f"Alert [{level.upper()}] {code}: {message}")

        except Exception as e:
            logger.error(f"Error handling alert: {e}")
            self.handler_stats["errors"] += 1

    async def handle_config(self, data: dict[str, Any]) -> None:
        """Handle configuration updates from backend.

        Updates projector configuration in real-time based on
        backend configuration changes.

        Args:
            data: Configuration data with section and config updates
        """
        if not self.config.enable_config_updates:
            return

        try:
            section = data.get("section", "unknown")
            config_data = data.get("config", {})
            change_summary = data.get("change_summary", "")

            # Apply configuration updates based on section
            await self._apply_config_update(section, config_data)

            self.handler_stats["config_updates_processed"] += 1

            logger.info(f"Applied config update for {section}: {change_summary}")

        except Exception as e:
            logger.error(f"Error handling config update: {e}")
            self.handler_stats["errors"] += 1

    async def handle_metrics(self, data: dict[str, Any]) -> None:
        """Handle performance metrics from backend.

        Processes system performance metrics for monitoring and
        potential display adjustments.

        Args:
            data: Metrics data with performance statistics
        """
        try:
            # Extract metrics
            broadcast_stats = data.get("broadcast_stats", {})
            data.get("frame_metrics", {})
            data.get("connection_stats", {})

            # Log performance info (could be displayed on projector)
            logger.debug(f"Backend metrics - Broadcast: {broadcast_stats}")

        except Exception as e:
            logger.error(f"Error handling metrics: {e}")
            self.handler_stats["errors"] += 1

    async def handle_connection(self, data: dict[str, Any]) -> None:
        """Handle connection status messages.

        Args:
            data: Connection status data
        """
        try:
            client_id = data.get("client_id", "unknown")
            status = data.get("status", "unknown")
            data.get("timestamp")

            logger.info(f"Connection status for {client_id}: {status}")

        except Exception as e:
            logger.error(f"Error handling connection message: {e}")
            self.handler_stats["errors"] += 1

    async def handle_error(self, data: dict[str, Any]) -> None:
        """Handle error messages from backend.

        Args:
            data: Error data with code, message, details
        """
        try:
            error_code = data.get("code", "UNKNOWN")
            error_message = data.get("message", "Unknown error")
            details = data.get("details", {})

            # Display error as alert
            await self.handle_alert(
                {
                    "level": "error",
                    "message": f"Backend Error: {error_message}",
                    "code": error_code,
                    "details": details,
                }
            )

        except Exception as e:
            logger.error(f"Error handling error message: {e}")
            self.handler_stats["errors"] += 1

    def get_handler_stats(self) -> dict[str, Any]:
        """Get message handler statistics.

        Returns:
            Dictionary with handler performance statistics
        """
        return {
            "stats": self.handler_stats.copy(),
            "active_trajectories": len(self.active_trajectories),
            "active_alerts": len(self.active_alerts),
            "last_state_update": self.last_state_update,
            "config": {
                "enable_trajectory_rendering": self.config.enable_trajectory_rendering,
                "enable_ball_tracking": self.config.enable_ball_tracking,
                "enable_alert_display": self.config.enable_alert_display,
            },
        }

    async def _process_ball_states(self, balls_data: list[dict[str, Any]]) -> None:
        """Process ball state updates."""
        for ball_data in balls_data:
            try:
                # Extract ball information
                ball_data.get("id", "unknown")
                ball_data.get("position", [0, 0])
                ball_data.get("velocity")
                ball_data.get("radius", 10.0)
                confidence = ball_data.get("confidence", 1.0)
                visible = ball_data.get("visible", True)

                if not visible or confidence < 0.5:
                    continue

                # Create ball state object (simplified - adapt to your models)
                ball_state = self._create_ball_state(ball_data)

                # Update projector with ball state
                if ball_state:
                    self.projector.update_ball_state(ball_state)

            except Exception as e:
                logger.error(f"Error processing ball state: {e}")

    async def _process_cue_state(self, cue_data: dict[str, Any]) -> None:
        """Process cue state updates."""
        try:
            detected = cue_data.get("detected", False)
            confidence = cue_data.get("confidence", 0.0)

            if not detected or confidence < 0.5:
                return

            # Process cue data (adapt based on your needs)
            angle = cue_data.get("angle", 0.0)
            position = cue_data.get("position", [0, 0])

            logger.debug(f"Cue detected: angle={angle:.1f}°, position={position}")

        except Exception as e:
            logger.error(f"Error processing cue state: {e}")

    async def _process_table_state(self, table_data: dict[str, Any]) -> None:
        """Process table state updates."""
        try:
            calibrated = table_data.get("calibrated", False)
            corners = table_data.get("corners", [])
            pockets = table_data.get("pockets", [])

            if calibrated and len(corners) == 4:
                logger.debug(f"Table calibrated with {len(pockets)} pockets")

        except Exception as e:
            logger.error(f"Error processing table state: {e}")

    async def _convert_trajectory_data(
        self, data: dict[str, Any]
    ) -> Optional[Trajectory]:
        """Convert WebSocket trajectory data to internal Trajectory object."""
        try:
            # This is a simplified conversion - adapt based on your Trajectory class
            lines = data.get("lines", [])
            data.get("collisions", [])
            confidence = data.get("confidence", 1.0)

            if not lines:
                return None

            # Create a simple trajectory (you may need to adapt this)
            # For now, we'll just create a basic trajectory representation
            trajectory_id = data.get("ball_id", "cue")

            # Create trajectory object (simplified - adapt to your actual Trajectory class)
            from ...core.models import Vector2D
            from ...core.physics.trajectory import Trajectory

            # Convert first line to start position
            first_line = lines[0]
            start_pos = Vector2D(first_line["start"][0], first_line["start"][1])

            # Create basic trajectory
            trajectory = Trajectory(
                ball_id=trajectory_id,
                initial_position=start_pos,
                initial_velocity=Vector2D(1.0, 0.0),  # Placeholder
                confidence=confidence,
            )

            return trajectory

        except Exception as e:
            logger.error(f"Error converting trajectory data: {e}")
            return None

    async def _clear_old_trajectories(self, current_trajectory_id: str) -> None:
        """Clear old trajectories from display."""
        try:
            import time

            current_time = time.time()
            max_age = 5.0  # 5 seconds

            # Remove old trajectories
            old_trajectories = [
                traj_id
                for traj_id, traj_data in self.active_trajectories.items()
                if current_time - traj_data["timestamp"] > max_age
            ]

            for traj_id in old_trajectories:
                self.projector.remove_trajectory(
                    traj_id, fade_out=self.config.trajectory_fade_out
                )
                del self.active_trajectories[traj_id]

        except Exception as e:
            logger.error(f"Error clearing old trajectories: {e}")

    async def _render_collision_prediction(
        self, collision: dict[str, Any], confidence: float
    ) -> None:
        """Render collision prediction with confidence indicator."""
        try:
            position = collision.get("position", [0, 0])
            collision.get("ball_id", "unknown")

            # Create position object for projector
            from ..rendering.renderer import Point2D

            collision_point = Point2D(position[0], position[1])

            # Render success indicator based on confidence
            self.projector.render_success_indicator(collision_point, confidence)

        except Exception as e:
            logger.error(f"Error rendering collision prediction: {e}")

    async def _display_alert(self, alert: dict[str, Any]) -> None:
        """Display alert on projector using visual overlay system."""
        try:
            level = alert["level"]
            message = alert["message"]
            code = alert["code"]
            details = alert.get("details", {})

            # Log the alert
            logger.info(f"Displaying alert [{level.upper()}]: {message}")

            # Import required modules for visual alerts (fallbacks defined at module level)
            try:
                from ..rendering.text import TextRenderer
            except ImportError:
                TextRenderer = None

            # Get alert display configuration
            alert_config = self._get_alert_display_config(level)

            # Create visual alert overlay
            await self._create_alert_overlay(
                level=level,
                message=message,
                code=code,
                details=details,
                config=alert_config
            )

            # Create appropriate effect based on alert level
            await self._create_alert_effect(level, alert_config["position"])

        except Exception as e:
            logger.error(f"Error displaying alert: {e}")

    def _get_alert_display_config(self, level: str) -> dict[str, Any]:
        """Get display configuration for alert level."""
        # Define alert configurations by severity level
        alert_configs = {
            "error": {
                "background_color": Colors.RED.with_alpha(0.8),
                "text_color": Colors.WHITE,
                "border_color": Colors.RED,
                "position": Point2D(50, 50),  # Top-left area
                "size": (400, 100),
                "icon": "✕",
                "priority": 4,
                "pulsing": True,
                "sound_enabled": True
            },
            "warning": {
                "background_color": Colors.YELLOW.with_alpha(0.7),
                "text_color": Colors.BLACK,
                "border_color": Colors.ORANGE,
                "position": Point2D(50, 170),  # Below error alerts
                "size": (350, 80),
                "icon": "⚠",
                "priority": 3,
                "pulsing": False,
                "sound_enabled": False
            },
            "info": {
                "background_color": Colors.BLUE.with_alpha(0.6),
                "text_color": Colors.WHITE,
                "border_color": Colors.CYAN,
                "position": Point2D(50, 270),  # Below warning alerts
                "size": (300, 70),
                "icon": "ℹ",
                "priority": 2,
                "pulsing": False,
                "sound_enabled": False
            },
            "success": {
                "background_color": Colors.GREEN.with_alpha(0.6),
                "text_color": Colors.WHITE,
                "border_color": Colors.GREEN,
                "position": Point2D(50, 360),  # Below info alerts
                "size": (300, 70),
                "icon": "✓",
                "priority": 1,
                "pulsing": False,
                "sound_enabled": False
            }
        }

        return alert_configs.get(level, alert_configs["info"])

    async def _create_alert_overlay(
        self,
        level: str,
        message: str,
        code: str,
        details: dict[str, Any],
        config: dict[str, Any]
    ) -> None:
        """Create visual alert overlay on projector display."""
        try:
            # Check if projector has required rendering capabilities
            if not hasattr(self.projector, 'render_text_overlay'):
                # Fall back to effects-based alert display
                await self._create_effects_based_alert(level, message, config)
                return

            # Create text overlay components
            title_text = f"{config['icon']} {level.upper()}"
            message_text = message
            code_text = f"Code: {code}" if code != "NO_CODE" else ""

            # Position calculations
            position = config["position"]
            width, height = config["size"]

            # Create background overlay
            overlay_data = {
                "type": "alert_overlay",
                "level": level,
                "position": {"x": position.x, "y": position.y},
                "size": {"width": width, "height": height},
                "background": {
                    "color": config["background_color"].to_dict(),
                    "border_color": config["border_color"].to_dict(),
                    "border_width": 3.0,
                    "corner_radius": 8.0
                },
                "text": {
                    "title": {
                        "content": title_text,
                        "color": config["text_color"].to_dict(),
                        "font_size": 18,
                        "font_weight": "bold",
                        "position": {"x": position.x + 15, "y": position.y + 15}
                    },
                    "message": {
                        "content": message_text,
                        "color": config["text_color"].to_dict(),
                        "font_size": 14,
                        "font_weight": "normal",
                        "position": {"x": position.x + 15, "y": position.y + 45},
                        "max_width": width - 30
                    },
                    "code": {
                        "content": code_text,
                        "color": config["text_color"].with_alpha(0.8).to_dict(),
                        "font_size": 10,
                        "font_weight": "normal",
                        "position": {"x": position.x + 15, "y": position.y + height - 20}
                    } if code_text else None
                },
                "animation": {
                    "pulsing": config["pulsing"],
                    "fade_in_duration": 0.3,
                    "display_duration": self.config.alert_display_duration,
                    "fade_out_duration": 0.5
                },
                "details": details
            }

            # Render the overlay on projector
            self.projector.render_text_overlay(overlay_data)

        except Exception as e:
            logger.error(f"Error creating alert overlay: {e}")
            # Fall back to basic effects
            await self._create_effects_based_alert(level, message, config)

    async def _create_effects_based_alert(
        self,
        level: str,
        message: str,
        config: dict[str, Any]
    ) -> None:
        """Create alert using effects system as fallback."""
        try:
            # Check if projector has effects system
            if not hasattr(self.projector, 'effects_system'):
                logger.warning("No effects system available for alert display")
                return

            effects_system = self.projector.effects_system
            position = config["position"]

            # Create appropriate effect based on alert level
            if level == "error":
                effects_system.create_failure_indicator(position, f"Error: {message}")
            elif level == "success":
                effects_system.create_success_indicator(position, 1.0)
            else:
                # Create a generic indicator effect
                try:
                    from ..rendering.effects import Effect, EffectType

                    effect = Effect(
                        effect_type=EffectType.UNCERTAINTY_CLOUD,
                        position=position,
                        start_time=time.time(),
                        duration=self.config.alert_display_duration,
                        properties={"level": level, "message": message}
                    )

                    effects_system._add_effect(effect)
                except ImportError:
                    logger.debug("Effect classes not available for generic alert effect")

        except Exception as e:
            logger.error(f"Error creating effects-based alert: {e}")

    async def _create_alert_effect(self, level: str, position) -> None:
        """Create visual effect to accompany alert display."""
        try:
            # Check if projector has effects system
            if not hasattr(self.projector, 'effects_system'):
                return

            effects_system = self.projector.effects_system

            # Create effect based on alert level
            if level == "error":
                # Create pulsing red effect around alert
                effects_system.create_failure_indicator(position, "Alert")
            elif level == "warning":
                # Create yellow warning burst
                effects_system.create_power_burst(position, 2.0, 0.0)
            elif level == "success":
                # Create green success indicator
                effects_system.create_success_indicator(position, 1.0)
            elif level == "info":
                # Create subtle blue effect
                try:
                    from ..rendering.effects import Effect, EffectType, Particle
                    from ..rendering.renderer import Colors
                    from ...core.models import Vector2D
                except ImportError:
                    # Use fallback classes if imports fail
                    Effect = EffectType = Particle = None
                    class Vector2D:
                        def __init__(self, x, y):
                            self.x = x
                            self.y = y
                import math

                if Effect and EffectType and Particle:
                    effect = Effect(
                        effect_type=EffectType.UNCERTAINTY_CLOUD,
                        position=position,
                        start_time=time.time(),
                        duration=1.0,
                        properties={"level": level}
                    )

                    # Add gentle particles for info alerts
                    for i in range(8):
                        angle = (i / 8) * 2 * math.pi
                        particle_pos = Point2D(
                            position.x + 30 * math.cos(angle),
                            position.y + 30 * math.sin(angle)
                        )

                        particle = Particle(
                            position=particle_pos,
                            velocity=Vector2D(0, 0),
                            color=Colors.CYAN.with_alpha(0.5),
                            size=3.0,
                            life_time=0.0,
                            max_life=1.0,
                            fade_rate=1.0
                        )

                        effect.particles.append(particle)

                    effects_system._add_effect(effect)
                else:
                    logger.debug("Effect classes not available, skipping info effect")

        except Exception as e:
            logger.error(f"Error creating alert effect: {e}")

    async def _remove_alert_after_delay(
        self, alert: dict[str, Any], delay: float
    ) -> None:
        """Remove alert after specified delay."""
        try:
            import asyncio

            await asyncio.sleep(delay)

            # Remove from active alerts
            if alert in self.active_alerts:
                self.active_alerts.remove(alert)

            logger.debug(f"Removed alert after {delay}s delay")

        except Exception as e:
            logger.error(f"Error removing alert: {e}")

    async def _apply_config_update(
        self, section: str, config_data: dict[str, Any]
    ) -> None:
        """Apply configuration update to projector."""
        try:
            if section == "trajectory":
                # Update trajectory rendering configuration
                await self._update_trajectory_config(config_data)
            elif section == "visual":
                # Update visual configuration
                await self._update_visual_config(config_data)
            elif section == "effects":
                # Update effects configuration
                await self._update_effects_config(config_data)
            else:
                logger.debug(f"Unknown config section: {section}")

        except Exception as e:
            logger.error(f"Error applying config update: {e}")

    async def _update_trajectory_config(self, config_data: dict[str, Any]) -> None:
        """Update trajectory rendering configuration."""
        try:
            # Extract trajectory configuration
            from ..rendering.renderer import Color
            from ..rendering.trajectory import TrajectoryVisualConfig

            # Create new trajectory config
            trajectory_config = TrajectoryVisualConfig()

            # Apply updates
            if "primary_color" in config_data:
                color_data = config_data["primary_color"]
                trajectory_config.primary_color = Color.from_rgb(*color_data)

            if "line_width" in config_data:
                trajectory_config.line_width = config_data["line_width"]

            if "opacity" in config_data:
                trajectory_config.opacity = config_data["opacity"]

            # Apply to projector
            self.projector.set_trajectory_config(trajectory_config)

            logger.debug("Updated trajectory configuration")

        except Exception as e:
            logger.error(f"Error updating trajectory config: {e}")

    async def _update_visual_config(self, config_data: dict[str, Any]) -> None:
        """Update visual configuration."""
        try:
            # Update visual settings (implement based on your projector capabilities)
            logger.debug(f"Updated visual configuration: {config_data}")

        except Exception as e:
            logger.error(f"Error updating visual config: {e}")

    async def _update_effects_config(self, config_data: dict[str, Any]) -> None:
        """Update effects configuration."""
        try:
            # Update effects settings
            from ..rendering.effects import EffectsConfig

            effects_config = EffectsConfig()

            if "trail_enabled" in config_data:
                effects_config.trail_enabled = config_data["trail_enabled"]

            if "collision_effects_enabled" in config_data:
                effects_config.collision_effects_enabled = config_data[
                    "collision_effects_enabled"
                ]

            # Apply to projector
            self.projector.set_effects_config(effects_config)

            logger.debug("Updated effects configuration")

        except Exception as e:
            logger.error(f"Error updating effects config: {e}")

    def _create_ball_state(self, ball_data: dict[str, Any]) -> Optional[BallState]:
        """Create BallState object from ball data."""
        try:
            from ...core.models import Vector2D

            # Extract ball data
            position = ball_data.get("position", [0, 0])
            velocity_data = ball_data.get("velocity")
            radius = ball_data.get("radius", 10.0)

            # Create position vector
            position_vector = Vector2D(position[0], position[1])

            # Create velocity vector
            velocity_vector = Vector2D(0, 0)
            if velocity_data and len(velocity_data) >= 2:
                velocity_vector = Vector2D(velocity_data[0], velocity_data[1])

            # Create ball state
            ball_state = BallState(
                position=position_vector,
                velocity=velocity_vector,
                radius=radius,
                spin=None,  # Not provided in basic data
            )

            return ball_state

        except Exception as e:
            logger.error(f"Error creating ball state: {e}")
            return None
