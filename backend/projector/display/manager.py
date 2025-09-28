"""Display management for the projector module.

This module handles the creation and management of projection displays,
including OpenGL/pygame initialization, window management, and coordinate systems.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import moderngl
import numpy as np
import pygame

from ..calibration import (
    CalibrationManager,
    TableDimensions,
)

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display output modes."""

    FULLSCREEN = "fullscreen"
    WINDOW = "window"
    BORDERLESS = "borderless"


class DisplayStatus(Enum):
    """Display status states."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DisplayInfo:
    """Information about available display devices."""

    index: int
    name: str
    width: int
    height: int
    refresh_rate: int
    is_primary: bool


@dataclass
class DisplayConfig:
    """Configuration for display management."""

    mode: DisplayMode = DisplayMode.FULLSCREEN
    monitor_index: int = 0
    resolution: tuple[int, int] = (1920, 1080)
    refresh_rate: int = 60
    vsync: bool = True
    background_color: tuple[int, int, int] = (0, 0, 0)
    title: str = "Billiards Trainer Projector"


class DisplayError(Exception):
    """Display-related errors."""

    pass


class DisplayManager:
    """Manages projection display output using OpenGL and pygame.

    This class provides the foundation for rendering overlays onto the pool table
    via a projector. It handles display initialization, window management,
    coordinate system setup, and the basic rendering loop.

    Features:
    - Fullscreen, windowed, and borderless display modes
    - Multiple monitor support
    - OpenGL rendering context
    - Display device detection and configuration
    - Proper error handling and recovery
    """

    def __init__(
        self,
        config: Optional[DisplayConfig] = None,
        table_dimensions: Optional[TableDimensions] = None,
        calibration_dir: Optional[Path] = None,
    ):
        """Initialize the display manager.

        Args:
            config: Display configuration, uses defaults if None
            table_dimensions: Physical table dimensions for calibration
            calibration_dir: Directory for calibration files
        """
        self.config = config or DisplayConfig()
        self.status = DisplayStatus.INITIALIZED

        # Display properties
        self.screen: Optional[pygame.Surface] = None
        self.gl_context: Optional[moderngl.Context] = None
        self.display_info: list[DisplayInfo] = []
        self.current_display: Optional[DisplayInfo] = None

        # Rendering properties
        self.width = 0
        self.height = 0
        self.aspect_ratio = 1.0
        self.coordinate_transform = np.identity(3)  # 2D homogeneous transform

        # Calibration system
        self.calibration_manager: Optional[CalibrationManager] = None
        self.table_dimensions = table_dimensions
        self.calibration_enabled = False
        self.calibration_mode = False  # True when in calibration mode

        # Performance tracking
        self.frame_count = 0
        self.last_fps_update = 0.0
        self.current_fps = 0.0

        # Initialize pygame and OpenGL
        self._initialize_pygame()
        self._detect_displays()

        # Initialize calibration system if table dimensions provided
        if self.table_dimensions:
            self._initialize_calibration(calibration_dir)

        logger.info("DisplayManager initialized successfully")

    def _initialize_pygame(self) -> None:
        """Initialize pygame for display management."""
        try:
            pygame.init()
            pygame.display.init()

            # Set OpenGL attributes for better rendering quality
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)  # 4x MSAA

            logger.debug("Pygame initialized with OpenGL context")

        except Exception as e:
            logger.error(f"Failed to initialize pygame: {e}")
            raise DisplayError(f"Pygame initialization failed: {e}")

    def _detect_displays(self) -> None:
        """Detect available display devices."""
        try:
            self.display_info.clear()

            # Get display count
            num_displays = pygame.display.get_num_displays()
            logger.debug(f"Detected {num_displays} display(s)")

            for i in range(num_displays):
                try:
                    # Get display bounds
                    bounds = pygame.display.get_desktop_sizes()[i]

                    # Create display info
                    display = DisplayInfo(
                        index=i,
                        name=f"Display {i + 1}",
                        width=bounds[0],
                        height=bounds[1],
                        refresh_rate=60,  # Default, would need platform-specific code to get actual
                        is_primary=(i == 0),
                    )

                    self.display_info.append(display)
                    logger.debug(f"Display {i}: {bounds[0]}x{bounds[1]}")

                except Exception as e:
                    logger.warning(f"Failed to get info for display {i}: {e}")

            if not self.display_info:
                # Fallback to default display
                self.display_info.append(
                    DisplayInfo(
                        index=0,
                        name="Default Display",
                        width=1920,
                        height=1080,
                        refresh_rate=60,
                        is_primary=True,
                    )
                )
                logger.warning("No displays detected, using default")

        except Exception as e:
            logger.error(f"Display detection failed: {e}")
            # Create minimal fallback display info
            self.display_info = [
                DisplayInfo(
                    index=0,
                    name="Fallback Display",
                    width=1920,
                    height=1080,
                    refresh_rate=60,
                    is_primary=True,
                )
            ]

    def start_display(self, mode: Optional[DisplayMode] = None) -> bool:
        """Start the projection display output.

        Args:
            mode: Display mode to use, defaults to config mode

        Returns:
            True if display started successfully

        Raises:
            DisplayError: If display startup fails
        """
        try:
            if self.status == DisplayStatus.RUNNING:
                logger.warning("Display already running")
                return True

            display_mode = mode or self.config.mode
            logger.info(f"Starting display in {display_mode.value} mode")

            # Select target display
            if self.config.monitor_index < len(self.display_info):
                self.current_display = self.display_info[self.config.monitor_index]
            else:
                self.current_display = self.display_info[0]
                logger.warning(
                    f"Monitor index {self.config.monitor_index} not available, using primary"
                )

            # Set display position for multi-monitor support
            if len(self.display_info) > 1 and self.config.monitor_index > 0:
                # Position window on specified monitor
                pos_x = sum(
                    d.width for d in self.display_info[: self.config.monitor_index]
                )
                pos_y = 0
                import os

                os.environ["SDL_VIDEO_WINDOW_POS"] = f"{pos_x},{pos_y}"

            # Create display surface based on mode
            if display_mode == DisplayMode.FULLSCREEN:
                self._create_fullscreen_display()
            elif display_mode == DisplayMode.BORDERLESS:
                self._create_borderless_display()
            else:  # WINDOW
                self._create_windowed_display()

            # Initialize OpenGL context
            self._initialize_opengl()

            # Set up coordinate system
            self._setup_coordinate_system()

            # Clear display
            self.clear_display()

            self.status = DisplayStatus.RUNNING
            logger.info(f"Display started successfully: {self.width}x{self.height}")
            return True

        except Exception as e:
            self.status = DisplayStatus.ERROR
            logger.error(f"Failed to start display: {e}")
            raise DisplayError(f"Display startup failed: {e}")

    def _create_fullscreen_display(self) -> None:
        """Create fullscreen display."""
        if self.current_display:
            self.width = self.current_display.width
            self.height = self.current_display.height
        else:
            self.width, self.height = self.config.resolution

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        if self.config.vsync:
            flags |= pygame.HWSURFACE

        self.screen = pygame.display.set_mode(
            (self.width, self.height), flags, display=self.config.monitor_index
        )
        pygame.display.set_caption(self.config.title)

    def _create_borderless_display(self) -> None:
        """Create borderless windowed display."""
        if self.current_display:
            self.width = self.current_display.width
            self.height = self.current_display.height
        else:
            self.width, self.height = self.config.resolution

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.NOFRAME
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(self.config.title)

    def _create_windowed_display(self) -> None:
        """Create windowed display."""
        self.width, self.height = self.config.resolution
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(self.config.title)

    def _initialize_opengl(self) -> None:
        """Initialize OpenGL context and settings."""
        try:
            # Create moderngl context
            self.gl_context = moderngl.create_context()

            # Set basic OpenGL state
            self.gl_context.enable(moderngl.BLEND)
            self.gl_context.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )

            # Set viewport
            self.gl_context.viewport = (0, 0, self.width, self.height)

            # Clear color
            bg = self.config.background_color
            self.gl_context.clear_color = (bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)

            logger.debug("OpenGL context initialized")

        except Exception as e:
            logger.error(f"OpenGL initialization failed: {e}")
            raise DisplayError(f"OpenGL setup failed: {e}")

    def _setup_coordinate_system(self) -> None:
        """Set up the display coordinate system.

        Establishes a coordinate transform that maps from table coordinates
        to display coordinates. This will be enhanced during calibration.
        """
        try:
            self.aspect_ratio = self.width / self.height

            # Create basic identity transform for now
            # This will be replaced with calibration transform later
            self.coordinate_transform = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            logger.debug(
                f"Coordinate system set up: {self.width}x{self.height}, aspect={self.aspect_ratio:.2f}"
            )

        except Exception as e:
            logger.error(f"Coordinate system setup failed: {e}")
            raise DisplayError(f"Coordinate system setup failed: {e}")

    def stop_display(self) -> None:
        """Stop the projection display."""
        try:
            if self.status != DisplayStatus.RUNNING:
                logger.warning("Display not running")
                return

            logger.info("Stopping display")

            # Clean up OpenGL context
            if self.gl_context:
                self.gl_context.release()
                self.gl_context = None

            # Clean up pygame
            if self.screen:
                pygame.display.quit()
                self.screen = None

            self.status = DisplayStatus.STOPPED
            logger.info("Display stopped successfully")

        except Exception as e:
            self.status = DisplayStatus.ERROR
            logger.error(f"Error stopping display: {e}")

    def clear_display(self) -> None:
        """Clear the display with background color."""
        try:
            if not self.gl_context:
                return

            self.gl_context.clear()

        except Exception as e:
            logger.warning(f"Display clear failed: {e}")

    def present_frame(self) -> None:
        """Present the current frame to the display.

        This swaps the buffers and updates FPS tracking.
        """
        try:
            if not self.screen:
                return

            pygame.display.flip()

            # Update FPS tracking
            self.frame_count += 1
            current_time = time.time()

            if current_time - self.last_fps_update >= 1.0:
                self.current_fps = self.frame_count / (
                    current_time - self.last_fps_update
                )
                self.frame_count = 0
                self.last_fps_update = current_time

        except Exception as e:
            logger.warning(f"Frame presentation failed: {e}")

    def handle_events(self) -> bool:
        """Handle pygame events.

        Returns:
            False if quit event received, True otherwise
        """
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                elif event.type == pygame.VIDEORESIZE:
                    if self.config.mode == DisplayMode.WINDOW:
                        self.width = event.w
                        self.height = event.h
                        self.aspect_ratio = self.width / self.height
                        if self.gl_context:
                            self.gl_context.viewport = (0, 0, self.width, self.height)
                        logger.debug(f"Window resized to {self.width}x{self.height}")

            return True

        except Exception as e:
            logger.warning(f"Event handling failed: {e}")
            return True

    def is_running(self) -> bool:
        """Check if display is currently running."""
        return self.status == DisplayStatus.RUNNING

    def get_display_info(self) -> dict[str, Any]:
        """Get information about the current display setup.

        Returns:
            Dictionary containing display information
        """
        return {
            "status": self.status.value,
            "mode": self.config.mode.value,
            "resolution": (self.width, self.height),
            "aspect_ratio": self.aspect_ratio,
            "fps": self.current_fps,
            "monitor_index": self.config.monitor_index,
            "available_displays": [
                {
                    "index": d.index,
                    "name": d.name,
                    "resolution": (d.width, d.height),
                    "refresh_rate": d.refresh_rate,
                    "is_primary": d.is_primary,
                }
                for d in self.display_info
            ],
            "current_display": (
                {
                    "index": self.current_display.index,
                    "name": self.current_display.name,
                    "resolution": (
                        self.current_display.width,
                        self.current_display.height,
                    ),
                }
                if self.current_display
                else None
            ),
        }

    def set_coordinate_transform(self, transform: np.ndarray) -> None:
        """Set the coordinate transformation matrix.

        Args:
            transform: 3x3 homogeneous transformation matrix
        """
        if transform.shape != (3, 3):
            raise ValueError("Transform must be a 3x3 matrix")

        self.coordinate_transform = transform.astype(np.float32)
        logger.debug("Coordinate transform updated")

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform a point from table coordinates to display coordinates.

        Args:
            x: Table X coordinate
            y: Table Y coordinate

        Returns:
            Tuple of (display_x, display_y)
        """
        # Apply homogeneous transformation
        point = np.array([x, y, 1.0], dtype=np.float32)
        transformed = self.coordinate_transform @ point

        # Convert to display coordinates (normalize if needed)
        if transformed[2] != 0:
            display_x = transformed[0] / transformed[2]
            display_y = transformed[1] / transformed[2]
        else:
            display_x = transformed[0]
            display_y = transformed[1]

        return display_x, display_y

    # Calibration methods

    def _initialize_calibration(self, calibration_dir: Optional[Path]) -> None:
        """Initialize calibration system.

        Args:
            calibration_dir: Directory for calibration files
        """
        try:
            # Wait for display dimensions to be available
            if self.width == 0 or self.height == 0:
                # Use config resolution as fallback
                width, height = self.config.resolution
            else:
                width, height = self.width, self.height

            self.calibration_manager = CalibrationManager(
                display_width=width,
                display_height=height,
                table_dimensions=self.table_dimensions,
                calibration_dir=calibration_dir,
            )

            # Try to load existing calibration
            if self.calibration_manager.load_calibration():
                self.calibration_enabled = True
                self._apply_calibration_transform()
                logger.info("Existing calibration loaded and applied")
            else:
                logger.info("No existing calibration found")

        except Exception as e:
            logger.error(f"Failed to initialize calibration: {e}")

    def enable_calibration_mode(self) -> bool:
        """Enable calibration mode for interactive calibration.

        Returns:
            True if calibration mode enabled successfully
        """
        if not self.calibration_manager:
            logger.error("Calibration manager not initialized")
            return False

        self.calibration_mode = True
        logger.info("Calibration mode enabled")
        return True

    def disable_calibration_mode(self) -> None:
        """Disable calibration mode and return to normal operation."""
        self.calibration_mode = False
        logger.info("Calibration mode disabled")

    def start_calibration(self) -> bool:
        """Start the calibration process.

        Returns:
            True if calibration started successfully
        """
        if not self.calibration_manager:
            logger.error("Calibration manager not initialized")
            return False

        if not self.calibration_mode:
            logger.error("Must enable calibration mode first")
            return False

        success = self.calibration_manager.start_calibration()
        if success:
            logger.info("Calibration process started")

        return success

    def get_calibration_status(self) -> Optional[dict[str, Any]]:
        """Get current calibration status.

        Returns:
            Calibration status dictionary or None if not available
        """
        if not self.calibration_manager:
            return None

        return self.calibration_manager.get_calibration_data()

    def save_calibration(self, profile_name: Optional[str] = None) -> bool:
        """Save current calibration.

        Args:
            profile_name: Optional profile name

        Returns:
            True if saved successfully
        """
        if not self.calibration_manager:
            logger.error("Calibration manager not initialized")
            return False

        success = self.calibration_manager.save_calibration(profile_name)
        if success:
            self.calibration_enabled = True
            self._apply_calibration_transform()
            logger.info("Calibration saved and applied")

        return success

    def load_calibration(self, profile_name: Optional[str] = None) -> bool:
        """Load a saved calibration.

        Args:
            profile_name: Optional profile name to load

        Returns:
            True if loaded successfully
        """
        if not self.calibration_manager:
            logger.error("Calibration manager not initialized")
            return False

        success = self.calibration_manager.load_calibration(profile_name)
        if success:
            self.calibration_enabled = True
            self._apply_calibration_transform()
            logger.info("Calibration loaded and applied")

        return success

    def _apply_calibration_transform(self) -> None:
        """Apply calibration transform to coordinate system."""
        if (
            not self.calibration_manager
            or not self.calibration_manager.is_calibration_valid()
        ):
            return

        try:
            # Get the complete transform matrix from calibration manager
            # This combines both geometric and keystone transforms
            calibration_data = self.calibration_manager.get_calibration_data()

            # For now, we'll use the geometric transform as the base
            geometric_data = calibration_data.get("geometric", {})
            if "transform_matrix" in geometric_data:
                transform_matrix = np.array(
                    geometric_data["transform_matrix"], dtype=np.float32
                )
                self.set_coordinate_transform(transform_matrix)
                logger.debug("Calibration transform applied to coordinate system")

        except Exception as e:
            logger.error(f"Failed to apply calibration transform: {e}")

    def transform_point_with_calibration(
        self, table_x: float, table_y: float
    ) -> tuple[float, float]:
        """Transform a point using calibration if available.

        Args:
            table_x: Table X coordinate
            table_y: Table Y coordinate

        Returns:
            Display coordinates (x, y)
        """
        if self.calibration_enabled and self.calibration_manager:
            return self.calibration_manager.transform_point(table_x, table_y)
        else:
            # Fall back to basic transform
            return self.transform_point(table_x, table_y)

    def inverse_transform_point_with_calibration(
        self, display_x: float, display_y: float
    ) -> tuple[float, float]:
        """Inverse transform a point using calibration if available.

        Args:
            display_x: Display X coordinate
            display_y: Display Y coordinate

        Returns:
            Table coordinates (x, y)
        """
        if self.calibration_enabled and self.calibration_manager:
            return self.calibration_manager.inverse_transform_point(
                display_x, display_y
            )
        else:
            # Fall back to basic inverse (identity for now)
            return display_x, display_y

    def get_calibration_grid(self) -> Optional[list[list[tuple[float, float]]]]:
        """Get calibration grid pattern for display.

        Returns:
            Grid lines as lists of display coordinate points or None
        """
        if not self.calibration_manager:
            return None

        try:
            # Get table grid from geometric calibrator
            table_grid = (
                self.calibration_manager.geometric_calibrator.generate_table_grid()
            )

            # Transform to display coordinates
            display_grid = []
            for line in table_grid:
                display_line = []
                for table_x, table_y in line:
                    display_x, display_y = self.transform_point_with_calibration(
                        table_x, table_y
                    )
                    display_line.append((display_x, display_y))
                display_grid.append(display_line)

            return display_grid

        except Exception as e:
            logger.error(f"Failed to generate calibration grid: {e}")
            return None

    def get_calibration_crosshairs(self) -> Optional[list[list[tuple[float, float]]]]:
        """Get calibration crosshairs for corner adjustment.

        Returns:
            Crosshair lines as lists of display coordinate points or None
        """
        if not self.calibration_manager:
            return None

        try:
            return self.calibration_manager.keystone_calibrator.generate_crosshairs()
        except Exception as e:
            logger.error(f"Failed to generate calibration crosshairs: {e}")
            return None

    def is_calibration_valid(self) -> bool:
        """Check if current calibration is valid.

        Returns:
            True if calibration is valid and active
        """
        return (
            self.calibration_enabled
            and self.calibration_manager is not None
            and self.calibration_manager.is_calibration_valid()
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_display()

    def __str__(self) -> str:
        """String representation."""
        return f"DisplayManager(status={self.status.value}, resolution={self.width}x{self.height})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"DisplayManager("
            f"status={self.status.value}, "
            f"mode={self.config.mode.value}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.current_fps:.1f}"
            f")"
        )
