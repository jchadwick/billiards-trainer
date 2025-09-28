"""Main projector module entry point with trajectory rendering integration."""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Optional

import moderngl
import pygame

# Import core models with fallback for different import contexts
try:
    from ..core.game_state import BallState
    from ..core.physics.trajectory import PredictedCollision, Trajectory
except ImportError:
    # If running from the backend directory directly
    from core.game_state import BallState
    from core.physics.trajectory import PredictedCollision, Trajectory

from .calibration.manager import CalibrationManager
from .display.manager import DisplayManager
from .network.client import WebSocketClient
from .network.handlers import HandlerConfig, ProjectorMessageHandlers
from .rendering.effects import EffectsConfig, EffectsSystem
from .rendering.renderer import BasicRenderer, Color, Point2D
from .rendering.trajectory import TrajectoryRenderer, TrajectoryVisualConfig

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    FULLSCREEN = "fullscreen"
    WINDOW = "window"
    BORDERLESS = "borderless"


class LineStyle(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    ARROW = "arrow"


class ProjectorModule:
    """Enhanced projector interface with trajectory rendering capabilities.

    Provides comprehensive billiards trajectory visualization including:
    - Real-time trajectory rendering
    - Visual effects and animations
    - Collision prediction markers
    - Success probability indicators
    - Performance optimization
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize projector module with configuration.

        Args:
            config: Complete projector configuration
        """
        self.config = config
        self._initialized = False
        self._running = False

        # Core components
        self.display_manager: Optional[DisplayManager] = None
        self.calibration_manager: Optional[CalibrationManager] = None
        self.basic_renderer: Optional[BasicRenderer] = None
        self.trajectory_renderer: Optional[TrajectoryRenderer] = None
        self.effects_system: Optional[EffectsSystem] = None

        # Network components
        self.websocket_client: Optional[WebSocketClient] = None
        self.message_handlers: Optional[ProjectorMessageHandlers] = None
        self._network_enabled = config.get("network", {}).get("enabled", True)
        self._network_config = config.get("network", {})

        # OpenGL context
        self._gl_context: Optional[moderngl.Context] = None

        # Performance tracking
        self._frame_count = 0
        self._last_fps_time = 0.0
        self._current_fps = 0.0
        self._target_fps = config.get("rendering", {}).get("max_fps", 60)

        # Active trajectory data
        self._active_trajectories: dict[str, Trajectory] = {}

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

        logger.info(
            "ProjectorModule initialized with trajectory rendering and network support"
        )

    def initialize(self) -> bool:
        """Initialize all projector subsystems.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize pygame for display management
            pygame.init()
            pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)

            # Create OpenGL context
            self._gl_context = moderngl.create_context()

            # Initialize display manager
            self.display_manager = DisplayManager(self.config.get("display", {}))

            # Initialize calibration manager
            self.calibration_manager = CalibrationManager(
                self.config.get("calibration", {})
            )

            # Initialize basic renderer
            self.basic_renderer = BasicRenderer(self._gl_context)

            # Initialize trajectory renderer
            trajectory_config = TrajectoryVisualConfig()
            # Apply config overrides
            visual_config = self.config.get("visual", {})
            if visual_config:
                trajectory_config.primary_color = Color.from_rgb(
                    *visual_config.get("trajectory_color", (0, 255, 0))
                )
                trajectory_config.line_width = visual_config.get(
                    "trajectory_width", 3.0
                )
                trajectory_config.opacity = visual_config.get("trajectory_opacity", 0.8)

            self.trajectory_renderer = TrajectoryRenderer(
                self.basic_renderer, trajectory_config
            )

            # Initialize effects system
            effects_config = EffectsConfig()
            # Apply config overrides
            effects_settings = self.config.get("effects", {})
            if effects_settings:
                effects_config.trail_enabled = effects_settings.get(
                    "enable_trails", True
                )
                effects_config.collision_effects_enabled = effects_settings.get(
                    "enable_collision_effects", True
                )

            self.effects_system = EffectsSystem(self.basic_renderer, effects_config)

            # Initialize network components if enabled
            if self._network_enabled:
                self._initialize_network()

            self._initialized = True
            logger.info("ProjectorModule initialization complete")
            return True

        except Exception as e:
            logger.error(f"ProjectorModule initialization failed: {e}")
            return False

    def start_display(self, mode: DisplayMode = DisplayMode.FULLSCREEN) -> bool:
        """Start projector display output.

        Args:
            mode: Display mode to use

        Returns:
            True if display started successfully
        """
        if not self._initialized and not self.initialize():
            return False

        try:
            # Configure display based on mode
            display_config = self.config.get("display", {})

            if mode == DisplayMode.FULLSCREEN:
                screen_size = pygame.display.list_modes()[0]
                pygame.display.set_mode(
                    screen_size, pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF
                )
            elif mode == DisplayMode.BORDERLESS:
                screen_size = pygame.display.list_modes()[0]
                pygame.display.set_mode(
                    screen_size, pygame.NOFRAME | pygame.OPENGL | pygame.DOUBLEBUF
                )
            else:  # WINDOW
                resolution = display_config.get("resolution", [1920, 1080])
                pygame.display.set_mode(resolution, pygame.OPENGL | pygame.DOUBLEBUF)

            # Set up OpenGL viewport
            width, height = pygame.display.get_surface().get_size()
            self._gl_context.viewport = (0, 0, width, height)

            # Configure basic renderer projection
            self.basic_renderer.set_projection_matrix(width, height)

            # Enable blending for transparency
            self._gl_context.enable(moderngl.BLEND)
            self._gl_context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

            self._running = True
            self._last_fps_time = time.time()

            logger.info(f"Display started in {mode.value} mode ({width}x{height})")
            return True

        except Exception as e:
            logger.error(f"Failed to start display: {e}")
            return False

    def stop_display(self) -> None:
        """Stop projector display."""
        self._running = False

        if self.trajectory_renderer:
            self.trajectory_renderer.clear_all_trajectories()

        if self.effects_system:
            self.effects_system.clear_all_effects()

        # Stop network if running (async)
        if self.websocket_client and self.websocket_client.connected:
            logger.info("Scheduling network shutdown...")
            asyncio.create_task(self.stop_network())

        pygame.quit()
        logger.info("Display stopped")

    async def stop_display_async(self) -> None:
        """Stop projector display (async version)."""
        self._running = False

        if self.trajectory_renderer:
            self.trajectory_renderer.clear_all_trajectories()

        if self.effects_system:
            self.effects_system.clear_all_effects()

        # Stop network
        await self.stop_network()

        pygame.quit()
        logger.info("Display stopped")

    def render_trajectory(self, trajectory: Trajectory, fade_in: bool = True) -> None:
        """Render a complete trajectory with physics data.

        Args:
            trajectory: Physics trajectory to visualize
            fade_in: Whether to animate trajectory appearance
        """
        if not self.trajectory_renderer:
            logger.warning("Trajectory renderer not initialized")
            return

        # Store trajectory for frame rendering
        self._active_trajectories[trajectory.ball_id] = trajectory

        # Update trajectory renderer
        self.trajectory_renderer.update_trajectory(trajectory, fade_in)

        # Create effects for collisions
        if self.effects_system:
            for collision in trajectory.collisions:
                self.effects_system.create_collision_impact(collision)

        logger.debug(f"Rendered trajectory for ball {trajectory.ball_id}")

    def render_collision_prediction(
        self, collision: PredictedCollision, intensity: float = 1.0
    ) -> None:
        """Render collision prediction with visual effects.

        Args:
            collision: Predicted collision data
            intensity: Effect intensity multiplier
        """
        if not self.effects_system:
            logger.warning("Effects system not initialized")
            return

        self.effects_system.create_collision_impact(collision, intensity)

    def render_success_indicator(
        self, position: Point2D, success_probability: float
    ) -> None:
        """Render success probability indicator.

        Args:
            position: Position to show indicator
            success_probability: Success probability (0.0 to 1.0)
        """
        if not self.effects_system:
            return

        if success_probability > 0.5:
            self.effects_system.create_success_indicator(position, success_probability)
        else:
            self.effects_system.create_failure_indicator(
                position, "Low success probability"
            )

    def update_ball_state(self, ball_state: BallState) -> None:
        """Update ball state for real-time effects.

        Args:
            ball_state: Current ball state
        """
        if not self.effects_system:
            return

        position = Point2D(ball_state.position.x, ball_state.position.y)

        # Create spin visualization if ball has significant spin
        if ball_state.spin and ball_state.spin.magnitude() > 0.1:
            self.effects_system.create_spin_visualization(
                position, ball_state.spin, ball_state.radius
            )

        # Create power burst if ball has high velocity
        if ball_state.velocity.magnitude() > 2.0:
            import math

            direction = math.atan2(ball_state.velocity.y, ball_state.velocity.x)
            power_level = ball_state.velocity.magnitude()
            self.effects_system.create_power_burst(position, power_level, direction)

    def render_frame(self) -> None:
        """Render a complete frame with all visual elements."""
        if not self._running or not self.basic_renderer:
            return

        frame_start = time.time()

        # Clear frame
        self._gl_context.clear(0.0, 0.0, 0.0, 1.0)  # Clear to black

        # Begin rendering
        self.basic_renderer.begin_frame()

        # Update and render effects
        if self.effects_system:
            dt = 1.0 / self._target_fps  # Fixed timestep for consistency
            self.effects_system.update_effects(dt)
            self.effects_system.render_effects()

        # Render trajectories
        if self.trajectory_renderer:
            self.trajectory_renderer.render_frame()

        # End rendering
        self.basic_renderer.end_frame()

        # Swap buffers
        pygame.display.flip()

        # Update performance stats
        self._update_performance_stats(frame_start)

    def clear_display(self) -> None:
        """Clear all rendered content."""
        if self.trajectory_renderer:
            self.trajectory_renderer.clear_all_trajectories()

        if self.effects_system:
            self.effects_system.clear_all_effects()

        self._active_trajectories.clear()

        # Clear frame buffer
        if self._gl_context:
            self._gl_context.clear(0.0, 0.0, 0.0, 1.0)
            pygame.display.flip()

    def remove_trajectory(self, ball_id: str, fade_out: bool = True) -> None:
        """Remove trajectory for specific ball.

        Args:
            ball_id: ID of ball whose trajectory to remove
            fade_out: Whether to animate removal
        """
        if ball_id in self._active_trajectories:
            del self._active_trajectories[ball_id]

        if self.trajectory_renderer:
            self.trajectory_renderer.remove_trajectory(ball_id, fade_out)

    def set_trajectory_config(self, config: TrajectoryVisualConfig) -> None:
        """Update trajectory rendering configuration.

        Args:
            config: New trajectory visual configuration
        """
        if self.trajectory_renderer:
            self.trajectory_renderer.set_config(config)

    def set_effects_config(self, config: EffectsConfig) -> None:
        """Update effects system configuration.

        Args:
            config: New effects configuration
        """
        if self.effects_system:
            self.effects_system.set_config(config)

    def get_render_stats(self) -> dict:
        """Get comprehensive rendering statistics.

        Returns:
            Dictionary with performance and rendering statistics
        """
        stats = {
            "fps": self._current_fps,
            "frame_count": self._frame_count,
            "active_trajectories": len(self._active_trajectories),
            "running": self._running,
            "initialized": self._initialized,
        }

        if self.basic_renderer:
            stats["basic_renderer"] = self.basic_renderer.get_stats()

        if self.trajectory_renderer:
            stats["trajectory_renderer"] = self.trajectory_renderer.get_render_stats()

        if self.effects_system:
            stats["effects_system"] = self.effects_system.get_stats()

        return stats

    def is_running(self) -> bool:
        """Check if projector is currently running."""
        return self._running

    def get_trajectory_count(self) -> int:
        """Get number of active trajectories."""
        return len(self._active_trajectories)

    # Legacy compatibility methods

    def render_simple_trajectory(
        self, points: list[Point2D], color: tuple[int, int, int, int]
    ) -> None:
        """Render simple trajectory for legacy compatibility.

        Args:
            points: List of trajectory points
            color: RGBA color tuple
        """
        if not self.basic_renderer or len(points) < 2:
            return

        # Convert color to Color object
        render_color = Color.from_rgb(color[0], color[1], color[2], color[3])

        # Draw line segments between points
        for i in range(len(points) - 1):
            self.basic_renderer.draw_line(points[i], points[i + 1], render_color)

    # Private methods

    def _update_performance_stats(self, frame_start: float) -> None:
        """Update performance statistics."""
        self._frame_count += 1

        # Calculate FPS every second
        current_time = time.time()
        if current_time - self._last_fps_time >= 1.0:
            self._current_fps = self._frame_count / (current_time - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = current_time

        # Log performance warnings
        frame_time = current_time - frame_start
        if frame_time > 1.0 / (self._target_fps * 0.8):  # 80% of target frame time
            logger.warning(
                f"Frame time exceeded target: {frame_time:.3f}s (target: {1.0/self._target_fps:.3f}s)"
            )

    def _initialize_network(self) -> None:
        """Initialize WebSocket client and message handlers."""
        try:
            # Get network configuration
            api_url = self._network_config.get("api_url", "ws://localhost:8000/ws")
            auth_token = self._network_config.get("auth_token")
            client_id = self._network_config.get(
                "client_id", f"projector_{int(time.time())}"
            )

            # Create WebSocket client
            self.websocket_client = WebSocketClient(
                api_url=api_url,
                client_id=client_id,
                auth_token=auth_token,
                reconnect_enabled=self._network_config.get("auto_reconnect", True),
                max_reconnect_attempts=self._network_config.get(
                    "max_reconnect_attempts", -1
                ),
                ping_interval=self._network_config.get("ping_interval", 30.0),
            )

            # Create message handlers
            handler_config = HandlerConfig(
                enable_trajectory_rendering=True,
                enable_ball_tracking=True,
                enable_alert_display=True,
                enable_config_updates=True,
                trajectory_fade_in=True,
                trajectory_fade_out=True,
            )

            self.message_handlers = ProjectorMessageHandlers(
                projector=self, config=handler_config
            )

            # Register message handlers
            self._register_message_handlers()

            # Register connection handlers
            self.websocket_client.add_connection_handler(self._on_connection_changed)
            self.websocket_client.add_error_handler(self._on_network_error)

            logger.info(f"Network components initialized - API: {api_url}")

        except Exception as e:
            logger.error(f"Failed to initialize network components: {e}")
            self._network_enabled = False

    def _register_message_handlers(self) -> None:
        """Register WebSocket message handlers."""
        if not self.websocket_client or not self.message_handlers:
            return

        # Register handlers for each message type
        self.websocket_client.add_message_handler(
            "frame", self.message_handlers.handle_frame
        )
        self.websocket_client.add_message_handler(
            "state", self.message_handlers.handle_state
        )
        self.websocket_client.add_message_handler(
            "trajectory", self.message_handlers.handle_trajectory
        )
        self.websocket_client.add_message_handler(
            "alert", self.message_handlers.handle_alert
        )
        self.websocket_client.add_message_handler(
            "config", self.message_handlers.handle_config
        )
        self.websocket_client.add_message_handler(
            "metrics", self.message_handlers.handle_metrics
        )
        self.websocket_client.add_message_handler(
            "connection", self.message_handlers.handle_connection
        )
        self.websocket_client.add_message_handler(
            "error", self.message_handlers.handle_error
        )

        logger.debug("WebSocket message handlers registered")

    def _on_connection_changed(self, connected: bool) -> None:
        """Handle WebSocket connection state changes."""
        if connected:
            logger.info(
                "WebSocket connected - projector is now receiving real-time data"
            )
            # Subscribe to required data streams
            asyncio.create_task(self._subscribe_to_streams())
        else:
            logger.warning(
                "WebSocket disconnected - projector operating in standalone mode"
            )

    def _on_network_error(self, error_message: str) -> None:
        """Handle network errors."""
        logger.error(f"Network error: {error_message}")

    async def _subscribe_to_streams(self) -> None:
        """Subscribe to required data streams."""
        if not self.websocket_client or not self.websocket_client.connected:
            return

        # Define required streams for projector
        required_streams = ["state", "trajectory", "alert", "config"]

        try:
            success = await self.websocket_client.subscribe(required_streams)
            if success:
                logger.info(f"Subscribed to streams: {required_streams}")
            else:
                logger.warning("Failed to subscribe to required streams")

        except Exception as e:
            logger.error(f"Error subscribing to streams: {e}")

    async def start_network(self) -> bool:
        """Start WebSocket connection to backend API.

        Returns:
            True if network started successfully
        """
        if not self._network_enabled or not self.websocket_client:
            logger.info("Network is disabled")
            return False

        try:
            logger.info("Starting WebSocket connection...")
            success = await self.websocket_client.connect()

            if success:
                logger.info("WebSocket connection established")
                return True
            else:
                logger.error("Failed to establish WebSocket connection")
                return False

        except Exception as e:
            logger.error(f"Error starting network: {e}")
            return False

    async def stop_network(self) -> None:
        """Stop WebSocket connection."""
        if self.websocket_client:
            logger.info("Stopping WebSocket connection...")
            await self.websocket_client.disconnect()

            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            self._background_tasks.clear()
            logger.info("Network stopped")

    def get_network_status(self) -> dict[str, Any]:
        """Get current network status and statistics.

        Returns:
            Dictionary with network status information
        """
        if not self._network_enabled:
            return {"enabled": False, "status": "disabled"}

        if not self.websocket_client:
            return {"enabled": True, "status": "not_initialized"}

        status = {
            "enabled": True,
            "connection_info": self.websocket_client.get_connection_info(),
        }

        if self.message_handlers:
            status["handler_stats"] = self.message_handlers.get_handler_stats()

        return status

    def is_network_connected(self) -> bool:
        """Check if WebSocket is connected.

        Returns:
            True if connected to backend API
        """
        return (
            self._network_enabled
            and self.websocket_client is not None
            and self.websocket_client.connected
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_display()
