"""Text rendering system for projector displays.

This module provides comprehensive text rendering capabilities for the projector,
including font management, text measurement, and various text overlay styles.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import moderngl
import numpy as np

try:
    import pygame
    import pygame.freetype

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available, text rendering will be limited")

from .renderer import BasicRenderer, Color, Point2D

logger = logging.getLogger(__name__)


class FontWeight(Enum):
    """Font weight options."""

    LIGHT = "light"
    NORMAL = "normal"
    BOLD = "bold"
    EXTRA_BOLD = "extra_bold"


class TextAlign(Enum):
    """Text alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VerticalAlign(Enum):
    """Vertical alignment options."""

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    BASELINE = "baseline"


@dataclass
class FontDescriptor:
    """Font descriptor for text rendering."""

    family: str = "Arial"
    size: int = 24
    weight: FontWeight = FontWeight.NORMAL
    italic: bool = False


@dataclass
class TextStyle:
    """Text rendering style configuration."""

    font: FontDescriptor
    color: tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA
    background_color: Optional[tuple[int, int, int, int]] = None
    outline_color: Optional[tuple[int, int, int, int]] = None
    outline_width: float = 0.0
    shadow_offset: tuple[float, float] = (0.0, 0.0)
    shadow_color: tuple[int, int, int, int] = (0, 0, 0, 128)
    align: TextAlign = TextAlign.CENTER
    vertical_align: VerticalAlign = VerticalAlign.MIDDLE
    line_spacing: float = 1.2
    max_width: Optional[float] = None
    antialiasing: bool = True


@dataclass
class TextMetrics:
    """Text measurement metrics."""

    width: float
    height: float
    ascent: float
    descent: float
    line_count: int
    lines: list[str]


class FontManager:
    """Manages font loading and caching for text rendering."""

    def __init__(self):
        """Initialize font manager."""
        self._font_cache: dict[str, object] = {}
        self._default_fonts = [
            "Arial",
            "Helvetica",
            "sans-serif",
            "Times New Roman",
            "serif",
            "Courier New",
            "monospace",
        ]
        self._font_paths: dict[str, str] = {}

        # Initialize pygame fonts if available
        if PYGAME_AVAILABLE:
            try:
                pygame.freetype.init()
                self._discover_system_fonts()
            except Exception as e:
                logger.warning(f"Failed to initialize pygame fonts: {e}")

    def _discover_system_fonts(self) -> None:
        """Discover available system fonts."""
        try:
            if PYGAME_AVAILABLE:
                # Get system font paths
                system_fonts = pygame.freetype.get_fonts()
                for font_name in system_fonts[
                    :50
                ]:  # Limit to prevent excessive loading
                    try:
                        font_path = pygame.freetype.match_font(font_name)
                        if font_path:
                            self._font_paths[font_name.lower()] = font_path
                    except Exception:
                        continue

                logger.debug(f"Discovered {len(self._font_paths)} system fonts")
        except Exception as e:
            logger.warning(f"Font discovery failed: {e}")

    def get_font(self, descriptor: FontDescriptor) -> Optional[object]:
        """Get or load a font based on descriptor.

        Args:
            descriptor: Font descriptor

        Returns:
            Font object or None if not available
        """
        if not PYGAME_AVAILABLE:
            return None

        try:
            # Create cache key
            cache_key = f"{descriptor.family}_{descriptor.size}_{descriptor.weight.value}_{descriptor.italic}"

            if cache_key in self._font_cache:
                return self._font_cache[cache_key]

            # Try to find font file
            font_path = self._find_font_file(descriptor)
            if not font_path:
                # Fall back to default font
                font_path = (
                    pygame.freetype.match_font("Arial")
                    or pygame.freetype.get_default_font()
                )

            if font_path:
                font = pygame.freetype.Font(font_path, descriptor.size)
                if descriptor.weight == FontWeight.BOLD:
                    font.strong = True
                if descriptor.italic:
                    font.oblique = True

                self._font_cache[cache_key] = font
                return font

        except Exception as e:
            logger.warning(f"Failed to load font {descriptor.family}: {e}")

        return None

    def _find_font_file(self, descriptor: FontDescriptor) -> Optional[str]:
        """Find font file for descriptor."""
        family_lower = descriptor.family.lower()

        # Check exact match first
        if family_lower in self._font_paths:
            return self._font_paths[family_lower]

        # Check partial matches
        for font_name, font_path in self._font_paths.items():
            if family_lower in font_name or font_name in family_lower:
                return font_path

        # Try pygame's font matching
        if PYGAME_AVAILABLE:
            try:
                return pygame.freetype.match_font(
                    descriptor.family, bold=(descriptor.weight == FontWeight.BOLD)
                )
            except Exception:
                pass

        return None

    def measure_text(self, text: str, style: TextStyle) -> TextMetrics:
        """Measure text dimensions with given style.

        Args:
            text: Text to measure
            style: Text style

        Returns:
            Text metrics
        """
        if not PYGAME_AVAILABLE:
            # Fallback text measurement
            char_width = style.font.size * 0.6
            line_height = style.font.size * style.line_spacing
            lines = text.split("\n")
            max_width = max(len(line) * char_width for line in lines) if lines else 0
            height = len(lines) * line_height

            return TextMetrics(
                width=max_width,
                height=height,
                ascent=style.font.size * 0.8,
                descent=style.font.size * 0.2,
                line_count=len(lines),
                lines=lines,
            )

        try:
            font = self.get_font(style.font)
            if not font:
                # Fallback measurement
                return self.measure_text(text, style)

            lines = text.split("\n")
            if not lines:
                lines = [""]

            max_width = 0
            total_height = 0
            ascent = 0
            descent = 0

            for i, line in enumerate(lines):
                if line:
                    text_rect = font.get_rect(line)
                    line_width = text_rect.width
                    line_height = text_rect.height
                    # Use font metrics from pygame.freetype
                    metrics = font.get_metrics(line)
                    if metrics:
                        line_ascent = (
                            max(m[1] for m in metrics)
                            if metrics
                            else style.font.size * 0.8
                        )
                        line_descent = (
                            max(abs(m[2]) for m in metrics)
                            if metrics
                            else style.font.size * 0.2
                        )
                    else:
                        line_ascent = style.font.size * 0.8
                        line_descent = style.font.size * 0.2
                else:
                    line_width = 0
                    line_height = style.font.size
                    line_ascent = style.font.size * 0.8
                    line_descent = style.font.size * 0.2

                max_width = max(max_width, line_width)
                ascent = max(ascent, line_ascent)
                descent = max(descent, abs(line_descent))

                if i > 0:
                    total_height += line_height * style.line_spacing
                else:
                    total_height += line_height

            return TextMetrics(
                width=max_width,
                height=total_height,
                ascent=ascent,
                descent=descent,
                line_count=len(lines),
                lines=lines,
            )

        except Exception as e:
            logger.warning(f"Text measurement failed: {e}")
            # Return fallback metrics
            return TextMetrics(
                width=len(text) * style.font.size * 0.6,
                height=style.font.size,
                ascent=style.font.size * 0.8,
                descent=style.font.size * 0.2,
                line_count=1,
                lines=[text],
            )


class TextRenderer:
    """Advanced text rendering system for projector displays."""

    def __init__(self, basic_renderer: BasicRenderer):
        """Initialize text renderer.

        Args:
            basic_renderer: Basic renderer for drawing operations
        """
        self.basic_renderer = basic_renderer
        self.font_manager = FontManager()

        # Text rendering cache for performance
        self._text_cache: dict[str, object] = {}
        self._cache_size_limit = 100

        logger.info("Text renderer initialized")

    def render_text(
        self, text: str, position: Point2D, style: Optional[TextStyle] = None
    ) -> None:
        """Render text at specified position with given style.

        Args:
            text: Text content to render
            position: Position to render text at
            style: Text style (uses default if None)
        """
        if not text:
            return

        style = style or TextStyle(FontDescriptor())

        try:
            if PYGAME_AVAILABLE:
                self._render_text_pygame(text, position, style)
            else:
                self._render_text_fallback(text, position, style)

        except Exception as e:
            logger.error(f"Text rendering failed: {e}")
            # Fall back to simple indicator
            self._render_text_fallback(text, position, style)

    def _render_text_pygame(
        self, text: str, position: Point2D, style: TextStyle
    ) -> None:
        """Render text using pygame fonts."""
        font = self.font_manager.get_font(style.font)
        if not font:
            self._render_text_fallback(text, position, style)
            return

        try:
            # Measure text for positioning
            metrics = self.font_manager.measure_text(text, style)

            # Calculate actual position based on alignment
            render_x, render_y = self._calculate_text_position(position, metrics, style)

            # Render background if specified
            if style.background_color:
                self._render_text_background(render_x, render_y, metrics, style)

            # Render shadow if specified
            if style.shadow_offset != (0.0, 0.0):
                shadow_x = render_x + style.shadow_offset[0]
                shadow_y = render_y + style.shadow_offset[1]
                self._render_text_lines(
                    metrics.lines,
                    shadow_x,
                    shadow_y,
                    style,
                    color=style.shadow_color,
                    font=font,
                )

            # Render outline if specified
            if style.outline_width > 0 and style.outline_color:
                for dx in [-style.outline_width, 0, style.outline_width]:
                    for dy in [-style.outline_width, 0, style.outline_width]:
                        if dx != 0 or dy != 0:
                            outline_x = render_x + dx
                            outline_y = render_y + dy
                            self._render_text_lines(
                                metrics.lines,
                                outline_x,
                                outline_y,
                                style,
                                color=style.outline_color,
                                font=font,
                            )

            # Render main text
            self._render_text_lines(
                metrics.lines, render_x, render_y, style, color=style.color, font=font
            )

        except Exception as e:
            logger.warning(f"Pygame text rendering failed: {e}")
            self._render_text_fallback(text, position, style)

    def _render_text_lines(
        self,
        lines: list[str],
        x: float,
        y: float,
        style: TextStyle,
        color: tuple[int, int, int, int],
        font: object,
    ) -> None:
        """Render multiple lines of text."""
        current_y = y
        line_height = style.font.size * style.line_spacing

        for line in lines:
            if line:  # Skip empty lines for rendering but preserve spacing
                # Create surface with text
                text_surface, text_rect = font.render(
                    line,
                    color[:3],
                    style.background_color[:3] if style.background_color else None,
                )

                # Convert pygame surface to OpenGL texture and render
                try:
                    # Convert pygame surface to OpenGL texture
                    texture = self._surface_to_texture(text_surface)

                    if texture:
                        # Render the texture to the screen
                        self._render_texture_at_position(
                            texture, x, current_y, text_rect.width, text_rect.height
                        )
                    else:
                        raise Exception("Failed to create texture from surface")

                except Exception as tex_error:
                    logger.warning(f"Failed to render text as texture: {tex_error}")
                    # Fallback to rectangle placeholder
                    text_color = Color.from_rgb(*color)
                    self.basic_renderer.draw_rectangle(
                        x, current_y, text_rect.width, text_rect.height, text_color
                    )

            current_y += line_height

    def _render_text_fallback(
        self, text: str, position: Point2D, style: TextStyle
    ) -> None:
        """Fallback text rendering using basic shapes."""
        try:
            # Estimate text dimensions
            char_width = style.font.size * 0.6
            text_width = len(text) * char_width
            text_height = style.font.size

            # Calculate position
            if style.align == TextAlign.CENTER:
                x = position.x - text_width / 2
            elif style.align == TextAlign.RIGHT:
                x = position.x - text_width
            else:  # LEFT
                x = position.x

            if style.vertical_align == VerticalAlign.MIDDLE:
                y = position.y - text_height / 2
            elif style.vertical_align == VerticalAlign.BOTTOM:
                y = position.y - text_height
            else:  # TOP
                y = position.y

            # Render background if specified
            if style.background_color:
                bg_color = Color.from_rgb(*style.background_color)
                self.basic_renderer.draw_rectangle(
                    x - 2, y - 2, text_width + 4, text_height + 4, bg_color
                )

            # Render text indicator
            text_color = Color.from_rgb(*style.color)

            # Draw simple text representation
            for i, char in enumerate(text[:20]):  # Limit to prevent performance issues
                char_x = x + i * char_width

                # Draw character as small rectangle
                if char != " ":
                    self.basic_renderer.draw_rectangle(
                        char_x, y, char_width * 0.8, text_height, text_color
                    )

            # Draw indicator circle for text position
            self.basic_renderer.draw_circle(position, 3.0, text_color)

        except Exception as e:
            logger.warning(f"Fallback text rendering failed: {e}")

    def _calculate_text_position(
        self, position: Point2D, metrics: TextMetrics, style: TextStyle
    ) -> tuple[float, float]:
        """Calculate actual rendering position based on alignment."""
        x = position.x
        y = position.y

        # Horizontal alignment
        if style.align == TextAlign.CENTER:
            x -= metrics.width / 2
        elif style.align == TextAlign.RIGHT:
            x -= metrics.width

        # Vertical alignment
        if style.vertical_align == VerticalAlign.MIDDLE:
            y -= metrics.height / 2
        elif style.vertical_align == VerticalAlign.BOTTOM:
            y -= metrics.height
        elif style.vertical_align == VerticalAlign.BASELINE:
            y -= metrics.ascent

        return x, y

    def _render_text_background(
        self, x: float, y: float, metrics: TextMetrics, style: TextStyle
    ) -> None:
        """Render text background."""
        if not style.background_color:
            return

        try:
            bg_color = Color.from_rgb(*style.background_color)
            padding = style.font.size * 0.1

            self.basic_renderer.draw_rectangle(
                x - padding,
                y - padding,
                metrics.width + 2 * padding,
                metrics.height + 2 * padding,
                bg_color,
            )
        except Exception as e:
            logger.warning(f"Background rendering failed: {e}")

    def measure_text(self, text: str, style: Optional[TextStyle] = None) -> TextMetrics:
        """Measure text dimensions.

        Args:
            text: Text to measure
            style: Text style (uses default if None)

        Returns:
            Text metrics
        """
        style = style or TextStyle(FontDescriptor())
        return self.font_manager.measure_text(text, style)

    def create_text_style(
        self,
        family: str = "Arial",
        size: int = 24,
        color: tuple[int, int, int, int] = (255, 255, 255, 255),
        weight: FontWeight = FontWeight.NORMAL,
        align: TextAlign = TextAlign.CENTER,
        background: Optional[tuple[int, int, int, int]] = None,
    ) -> TextStyle:
        """Create a text style with common parameters.

        Args:
            family: Font family name
            size: Font size in pixels
            color: Text color RGBA
            weight: Font weight
            align: Text alignment
            background: Background color RGBA (optional)

        Returns:
            Configured text style
        """
        font = FontDescriptor(family=family, size=size, weight=weight)
        return TextStyle(
            font=font, color=color, background_color=background, align=align
        )

    def get_font_families(self) -> list[str]:
        """Get list of available font families.

        Returns:
            List of font family names
        """
        return list(self.font_manager._font_paths.keys())

    def clear_cache(self) -> None:
        """Clear text rendering cache."""
        self._text_cache.clear()
        self.font_manager._font_cache.clear()
        logger.debug("Text rendering cache cleared")

    def _surface_to_texture(self, surface) -> Optional[moderngl.Texture]:
        """Convert pygame surface to OpenGL texture.

        Args:
            surface: Pygame surface containing rendered text

        Returns:
            ModernGL texture object or None if conversion fails
        """
        try:
            # Get surface data
            width, height = surface.get_size()

            # Convert surface to RGBA format if needed
            if surface.get_bitsize() != 32:
                surface = surface.convert_alpha()

            # Get raw pixel data
            # Use pygame.image.tobytes() instead of deprecated tostring()
            try:
                # Modern pygame uses tobytes()
                raw_data = pygame.image.tobytes(surface, "RGBA", flipped=True)
            except AttributeError:
                # Fallback for older pygame versions
                raw_data = pygame.image.tostring(surface, "RGBA", flipped=True)

            # Create OpenGL texture with proper format
            # RGBA format requires 4 components
            texture = self.basic_renderer.ctx.texture((width, height), 4, raw_data)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

            # Enable blending for alpha transparency
            texture.use(0)

            return texture

        except Exception as e:
            logger.error(f"Failed to convert surface to texture: {e}")
            logger.debug(
                f"Surface info - size: {surface.get_size() if surface else 'None'}, "
                f"bitsize: {surface.get_bitsize() if surface else 'None'}"
            )
            return None

    def _render_texture_at_position(
        self, texture: moderngl.Texture, x: float, y: float, width: float, height: float
    ) -> None:
        """Render a texture at the specified position.

        Args:
            texture: OpenGL texture to render
            x: X position
            y: Y position
            width: Width of the rendered texture
            height: Height of the rendered texture
        """
        try:
            # Get OpenGL context
            ctx = self.basic_renderer.ctx

            # Create vertex data for a quad
            vertices = np.array(
                [
                    # Position (x, y)    # Texture coords (u, v)
                    x,
                    y,
                    0.0,
                    1.0,  # Bottom-left
                    x + width,
                    y,
                    1.0,
                    1.0,  # Bottom-right
                    x + width,
                    y + height,
                    1.0,
                    0.0,  # Top-right
                    x,
                    y + height,
                    0.0,
                    0.0,  # Top-left
                ],
                dtype=np.float32,
            )

            # Create indices for two triangles making a quad
            indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

            # Create vertex buffer and vertex array
            vbo = ctx.buffer(vertices.tobytes())
            ibo = ctx.buffer(indices.tobytes())

            # Create a simple shader program for texture rendering
            vertex_shader = """
            #version 330 core
            in vec2 position;
            in vec2 texcoord;
            out vec2 uv;
            uniform mat4 projection;

            void main() {
                gl_Position = projection * vec4(position, 0.0, 1.0);
                uv = texcoord;
            }
            """

            fragment_shader = """
            #version 330 core
            in vec2 uv;
            out vec4 fragColor;
            uniform sampler2D textTexture;

            void main() {
                fragColor = texture(textTexture, uv);
            }
            """

            # Create shader program
            program = ctx.program(
                vertex_shader=vertex_shader, fragment_shader=fragment_shader
            )

            # Create vertex array object
            vao = ctx.vertex_array(
                program, [(vbo, "2f 2f", "position", "texcoord")], ibo
            )

            # Set up OpenGL state for blending (for text transparency)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            # Bind texture and set uniforms
            texture.use(0)
            program["textTexture"].value = 0

            # Set projection matrix (should match the basic renderer's projection)
            if hasattr(self.basic_renderer, "projection_matrix"):
                program["projection"].write(
                    self.basic_renderer.projection_matrix.tobytes()
                )
            else:
                # Create a simple orthographic projection if not available
                # This assumes a coordinate system where (0,0) is bottom-left
                proj = np.array(
                    [
                        [2.0 / 1920, 0, 0, -1],
                        [0, 2.0 / 1080, 0, -1],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
                program["projection"].write(proj.tobytes())

            # Render the quad
            vao.render()

            # Clean up
            vao.release()
            vbo.release()
            ibo.release()
            program.release()

        except Exception as e:
            logger.error(f"Failed to render texture: {e}")
            # Fallback: render as a colored rectangle
            color = Color.from_rgb(255, 255, 255)
            self.basic_renderer.draw_rectangle(x, y, width, height, color)


# Convenience functions for common text rendering tasks


def create_info_text_style() -> TextStyle:
    """Create text style for information overlays."""
    return TextStyle(
        font=FontDescriptor(family="Arial", size=18, weight=FontWeight.NORMAL),
        color=(255, 255, 255, 255),
        background_color=(0, 0, 0, 128),
        align=TextAlign.CENTER,
        vertical_align=VerticalAlign.MIDDLE,
    )


def create_heading_text_style() -> TextStyle:
    """Create text style for headings."""
    return TextStyle(
        font=FontDescriptor(family="Arial", size=32, weight=FontWeight.BOLD),
        color=(255, 255, 255, 255),
        align=TextAlign.CENTER,
        vertical_align=VerticalAlign.MIDDLE,
        outline_color=(0, 0, 0, 255),
        outline_width=1.0,
    )


def create_debug_text_style() -> TextStyle:
    """Create text style for debug information."""
    return TextStyle(
        font=FontDescriptor(family="Courier New", size=14, weight=FontWeight.NORMAL),
        color=(0, 255, 0, 255),
        background_color=(0, 0, 0, 180),
        align=TextAlign.LEFT,
        vertical_align=VerticalAlign.TOP,
    )


def create_error_text_style() -> TextStyle:
    """Create text style for error messages."""
    return TextStyle(
        font=FontDescriptor(family="Arial", size=20, weight=FontWeight.BOLD),
        color=(255, 255, 255, 255),
        background_color=(255, 0, 0, 200),
        align=TextAlign.CENTER,
        vertical_align=VerticalAlign.MIDDLE,
        outline_color=(128, 0, 0, 255),
        outline_width=1.0,
    )
