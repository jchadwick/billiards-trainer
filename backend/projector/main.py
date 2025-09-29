"""Main projector module entry point with trajectory rendering integration."""

import asyncio
import logging
import time
from dataclasses import dataclass
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
from .rendering.text import (
    FontDescriptor,
    TextRenderer,
    TextStyle,
    create_info_text_style,
)
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


class RenderQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class CalibrationPoints:
    """Projector calibration corner points."""

    top_left: Point2D
    top_right: Point2D
    bottom_right: Point2D
    bottom_left: Point2D


@dataclass
class Line:
    """Rendered line segment."""

    start: Point2D
    end: Point2D
    color: tuple[int, int, int, int]  # RGBA
    width: float
    style: LineStyle
    glow: bool = False
    animated: bool = False


@dataclass
class Circle:
    """Rendered circle (for balls, impact points)."""

    center: Point2D
    radius: float
    color: tuple[int, int, int, int]
    filled: bool
    width: float = 1.0


@dataclass
class Text:
    """Rendered text overlay."""

    position: Point2D
    content: str
    size: int
    color: tuple[int, int, int, int]
    background: Optional[tuple[int, int, int, int]] = None
    anchor: str = "center"  # center, left, right


@dataclass
class RenderFrame:
    """Complete frame to be rendered."""

    trajectories: list[Line]
    collision_points: list[Circle]
    ghost_balls: list[Circle]
    text_overlays: list[Text]
    highlight_zones: list[dict]
    timestamp: float


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
        self.text_renderer: Optional[TextRenderer] = None
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

            # Initialize text renderer
            self.text_renderer = TextRenderer(self.basic_renderer)

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

    # Specification-required interface methods

    def calibrate(self, interactive: bool = True) -> CalibrationPoints:
        """Run calibration procedure and return calibration points.

        Args:
            interactive: Whether to run interactive calibration

        Returns:
            CalibrationPoints with corner coordinates

        Raises:
            RuntimeError: If calibration fails or is not initialized
        """
        if not self.calibration_manager:
            if not self.display_manager:
                raise RuntimeError("Display manager not initialized")

            # Initialize calibration manager from display manager
            self.calibration_manager = self.display_manager.calibration_manager

            if not self.calibration_manager:
                raise RuntimeError("Calibration manager not available")

        try:
            if interactive:
                # Enable calibration mode in display manager
                if (
                    self.display_manager
                    and not self.display_manager.enable_calibration_mode()
                ):
                    raise RuntimeError("Failed to enable calibration mode")

                # Start calibration process
                if not self.calibration_manager.start_calibration():
                    raise RuntimeError("Failed to start calibration process")

                logger.info(
                    "Interactive calibration started - use display interface to complete"
                )

                # For interactive mode, we would typically wait for user input
                # This is a simplified version that returns default corners
                # In a real implementation, this would integrate with the interactive UI

            # Get calibration data
            calibration_data = self.calibration_manager.get_calibration_data()
            keystone_data = calibration_data.get("keystone", {})
            corner_points = keystone_data.get("corner_points", {})

            if not corner_points:
                # Return default calibration points based on display size
                width = self.display_manager.width if self.display_manager else 1920
                height = self.display_manager.height if self.display_manager else 1080

                return CalibrationPoints(
                    top_left=Point2D(0, 0),
                    top_right=Point2D(width, 0),
                    bottom_right=Point2D(width, height),
                    bottom_left=Point2D(0, height),
                )

            # Convert from calibration format to CalibrationPoints
            return CalibrationPoints(
                top_left=Point2D(
                    corner_points.get("top_left", {}).get("x", 0),
                    corner_points.get("top_left", {}).get("y", 0),
                ),
                top_right=Point2D(
                    corner_points.get("top_right", {}).get("x", 1920),
                    corner_points.get("top_right", {}).get("y", 0),
                ),
                bottom_right=Point2D(
                    corner_points.get("bottom_right", {}).get("x", 1920),
                    corner_points.get("bottom_right", {}).get("y", 1080),
                ),
                bottom_left=Point2D(
                    corner_points.get("bottom_left", {}).get("x", 0),
                    corner_points.get("bottom_left", {}).get("y", 1080),
                ),
            )

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise RuntimeError(f"Calibration failed: {e}")

    def set_calibration(self, points: CalibrationPoints) -> None:
        """Apply calibration points to the projector.

        Args:
            points: Calibration corner points to apply

        Raises:
            RuntimeError: If calibration application fails
        """
        if not self.calibration_manager:
            raise RuntimeError("Calibration manager not initialized")

        try:
            # Convert CalibrationPoints to calibration manager format
            from .calibration.keystone import CornerPoints

            corner_points = CornerPoints(
                top_left=(points.top_left.x, points.top_left.y),
                top_right=(points.top_right.x, points.top_right.y),
                bottom_right=(points.bottom_right.x, points.bottom_right.y),
                bottom_left=(points.bottom_left.x, points.bottom_left.y),
            )

            # Apply to calibration manager
            if not self.calibration_manager.setup_keystone_corners(corner_points):
                raise RuntimeError("Failed to setup keystone corners")

            if not self.calibration_manager.complete_keystone_calibration():
                raise RuntimeError("Failed to complete keystone calibration")

            # Save calibration
            if not self.calibration_manager.save_calibration():
                logger.warning("Failed to save calibration")

            logger.info("Calibration points applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply calibration: {e}")
            raise RuntimeError(f"Failed to apply calibration: {e}")

    def render_text(
        self,
        text: str,
        position: Point2D,
        size: int = 24,
        color: tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> None:
        """Render text overlay on the display.

        Args:
            text: Text content to render
            position: Position to render text at
            size: Font size in pixels
            color: RGBA color tuple
        """
        if not self.text_renderer:
            logger.warning("Text renderer not initialized")
            return

        try:
            # Create text style for rendering
            style = TextStyle(font=FontDescriptor(size=size), color=color)

            # Use the proper text renderer
            self.text_renderer.render_text(text, position, style)

            logger.debug(f"Text rendered: '{text}' at ({position.x}, {position.y})")

        except Exception as e:
            logger.error(f"Failed to render text: {e}")
            # Fallback to basic indicator
            if self.basic_renderer:
                text_color = Color.from_rgb(color[0], color[1], color[2], color[3])
                self.basic_renderer.draw_circle(position, 5.0, text_color)

    def set_render_quality(self, quality: RenderQuality) -> None:
        """Set rendering quality level.

        Args:
            quality: Target rendering quality
        """
        try:
            # Update quality settings based on the quality level
            if quality == RenderQuality.LOW:
                self._target_fps = 30
                # Disable effects for performance
                if self.effects_system:
                    config = self.effects_system._config
                    config.trail_enabled = False
                    config.collision_effects_enabled = False
                    self.effects_system.set_config(config)

            elif quality == RenderQuality.MEDIUM:
                self._target_fps = 45
                if self.effects_system:
                    config = self.effects_system._config
                    config.trail_enabled = True
                    config.collision_effects_enabled = True
                    config.particle_count_multiplier = 0.5
                    self.effects_system.set_config(config)

            elif quality == RenderQuality.HIGH:
                self._target_fps = 60
                if self.effects_system:
                    config = self.effects_system._config
                    config.trail_enabled = True
                    config.collision_effects_enabled = True
                    config.particle_count_multiplier = 1.0
                    self.effects_system.set_config(config)

            elif quality == RenderQuality.ULTRA:
                self._target_fps = 120
                if self.effects_system:
                    config = self.effects_system._config
                    config.trail_enabled = True
                    config.collision_effects_enabled = True
                    config.particle_count_multiplier = 2.0
                    self.effects_system.set_config(config)

            logger.info(f"Render quality set to {quality.value}")

        except Exception as e:
            logger.error(f"Failed to set render quality: {e}")

    def get_display_info(self) -> dict:
        """Get display device information.

        Returns:
            Dictionary with display information and status
        """
        try:
            info = {
                "initialized": self._initialized,
                "running": self._running,
                "fps": self._current_fps,
                "target_fps": self._target_fps,
                "frame_count": self._frame_count,
            }

            # Add display manager info if available
            if self.display_manager:
                display_info = self.display_manager.get_display_info()
                info.update(display_info)

            # Add calibration info
            info["calibration"] = {
                "enabled": bool(self.calibration_manager),
                "valid": (
                    self.calibration_manager.is_calibration_valid()
                    if self.calibration_manager
                    else False
                ),
            }

            # Add rendering stats
            if self.basic_renderer:
                info["renderer"] = self.basic_renderer.get_stats()

            if self.trajectory_renderer:
                info["trajectory_renderer"] = (
                    self.trajectory_renderer.get_render_stats()
                )

            if self.effects_system:
                info["effects"] = self.effects_system.get_stats()

            return info

        except Exception as e:
            logger.error(f"Failed to get display info: {e}")
            return {"error": str(e)}

    def render_frame(self, frame: RenderFrame) -> None:
        """Render a complete frame with all visual elements.

        Args:
            frame: RenderFrame containing all elements to render
        """
        if not self._running or not self.basic_renderer:
            return

        try:
            frame_start = time.time()

            # Clear frame
            self._gl_context.clear(0.0, 0.0, 0.0, 1.0)
            self.basic_renderer.begin_frame()

            # Render trajectories
            for trajectory_line in frame.trajectories:
                line_color = Color.from_rgb(*trajectory_line.color)
                self.basic_renderer.draw_line(
                    trajectory_line.start,
                    trajectory_line.end,
                    line_color,
                    trajectory_line.width,
                )

            # Render collision points
            for collision_point in frame.collision_points:
                point_color = Color.from_rgb(*collision_point.color)
                if collision_point.filled:
                    self.basic_renderer.draw_circle(
                        collision_point.center, collision_point.radius, point_color
                    )
                else:
                    # Draw circle outline
                    self.basic_renderer.draw_circle(
                        collision_point.center, collision_point.radius, point_color
                    )

            # Render ghost balls
            for ghost_ball in frame.ghost_balls:
                ghost_color = Color.from_rgb(*ghost_ball.color)
                self.basic_renderer.draw_circle(
                    ghost_ball.center, ghost_ball.radius, ghost_color
                )

            # Render text overlays
            for text_overlay in frame.text_overlays:
                self.render_text(
                    text_overlay.content,
                    text_overlay.position,
                    text_overlay.size,
                    text_overlay.color,
                )

            # Render highlight zones (simplified)
            for zone in frame.highlight_zones:
                if "center" in zone and "radius" in zone:
                    center = Point2D(zone["center"]["x"], zone["center"]["y"])
                    radius = zone["radius"]
                    color = zone.get("color", (255, 255, 0, 128))
                    zone_color = Color.from_rgb(*color)
                    self.basic_renderer.draw_circle(center, radius, zone_color)

            self.basic_renderer.end_frame()
            pygame.display.flip()

            # Update performance stats
            self._update_performance_stats(frame_start)

            logger.debug(
                f"Frame rendered with {len(frame.trajectories)} trajectories, "
                f"{len(frame.collision_points)} collision points, "
                f"{len(frame.text_overlays)} text overlays"
            )

        except Exception as e:
            logger.error(f"Failed to render frame: {e}")

    def render_collision(
        self,
        position: Point2D,
        radius: float = 20,
        color: tuple[int, int, int, int] = (255, 0, 0, 128),
    ) -> None:
        """Render collision indicator at specified position.

        Args:
            position: Position to render collision indicator
            radius: Radius of collision indicator
            color: RGBA color for the indicator
        """
        if not self.basic_renderer:
            logger.warning("Basic renderer not initialized")
            return

        try:
            collision_color = Color.from_rgb(*color)
            self.basic_renderer.draw_circle(position, radius, collision_color)
            logger.debug(
                f"Collision indicator rendered at ({position.x}, {position.y})"
            )

        except Exception as e:
            logger.error(f"Failed to render collision indicator: {e}")

    def connect_to_backend(self, url: str) -> bool:
        """Connect to backend WebSocket.

        Args:
            url: WebSocket URL to connect to

        Returns:
            True if connection successful
        """
        try:
            if not self._network_enabled:
                logger.warning("Network is disabled")
                return False

            # Update network config with new URL
            self._network_config["api_url"] = url

            # Reinitialize network components if needed
            if not self.websocket_client:
                self._initialize_network()

            # Start network connection (async)
            async def connect_async():
                return await self.start_network()

            # Run async connection
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(connect_async())
            except RuntimeError:
                # No event loop running, create new one
                return asyncio.run(connect_async())

        except Exception as e:
            logger.error(f"Failed to connect to backend: {e}")
            return False

    def disconnect_from_backend(self) -> None:
        """Disconnect from backend WebSocket."""
        try:
            if self.websocket_client:
                # Stop network connection (async)
                async def disconnect_async():
                    await self.stop_network()

                # Run async disconnection
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(disconnect_async())
                except RuntimeError:
                    # No event loop running, create new one
                    asyncio.run(disconnect_async())

                logger.info("Disconnected from backend")
            else:
                logger.warning("No active connection to disconnect")

        except Exception as e:
            logger.error(f"Failed to disconnect from backend: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_display()
