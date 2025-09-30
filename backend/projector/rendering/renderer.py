"""Rendering engine for the projector module.

This module provides basic shape rendering capabilities for projecting
visual elements onto the pool table surface.
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import moderngl
import numpy as np

logger = logging.getLogger(__name__)


class LineStyle(Enum):
    """Line rendering styles."""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    ARROW = "arrow"


class BlendMode(Enum):
    """Blending modes for rendering."""

    NORMAL = "normal"
    ADDITIVE = "additive"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"


@dataclass
class Color:
    """RGBA color representation."""

    r: float  # 0.0 - 1.0
    g: float  # 0.0 - 1.0
    b: float  # 0.0 - 1.0
    a: float = 1.0  # 0.0 - 1.0 (alpha/transparency)

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: int = 255) -> "Color":
        """Create color from RGB values (0-255)."""
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @classmethod
    def from_hex(cls, hex_color: str) -> "Color":
        """Create color from hex string (#RRGGBB or #RRGGBBAA)."""
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

    def with_alpha(self, alpha: float) -> "Color":
        """Return a new color with different alpha."""
        return Color(self.r, self.g, self.b, alpha)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}


@dataclass
class Point2D:
    """2D point representation."""

    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def distance_to(self, other: "Point2D") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class RenderStats:
    """Rendering performance statistics."""

    frames_rendered: int = 0
    total_render_time: float = 0.0
    average_frame_time: float = 0.0
    shapes_rendered: int = 0
    draw_calls: int = 0
    last_frame_time: float = 0.0


class RendererError(Exception):
    """Renderer-related errors."""

    pass


class BasicRenderer:
    """Basic rendering engine for projector overlays.

    Provides fundamental shape rendering capabilities including:
    - Lines with various styles
    - Circles (filled and outlined)
    - Rectangles
    - Color management
    - Basic rendering pipeline

    Uses ModernGL for efficient OpenGL rendering.
    """

    def __init__(self, gl_context: moderngl.Context):
        """Initialize the renderer.

        Args:
            gl_context: ModernGL context for rendering
        """
        self.ctx = gl_context
        self.stats = RenderStats()

        # Shader programs
        self._line_program: Optional[moderngl.Program] = None
        self._circle_program: Optional[moderngl.Program] = None
        self._rect_program: Optional[moderngl.Program] = None

        # Vertex buffers and arrays
        self._line_vao: Optional[moderngl.VertexArray] = None
        self._circle_vao: Optional[moderngl.VertexArray] = None
        self._rect_vao: Optional[moderngl.VertexArray] = None

        # Rendering state
        self._current_color = Color(1.0, 1.0, 1.0, 1.0)
        self._current_line_width = 2.0
        self._current_blend_mode = BlendMode.NORMAL

        # Store projection matrix for external use (e.g., text rendering)
        self.projection_matrix: Optional[np.ndarray] = None

        # Initialize shaders and buffers
        self._initialize_shaders()
        self._initialize_buffers()

        logger.info("BasicRenderer initialized successfully")

    def _initialize_shaders(self) -> None:
        """Initialize shader programs for different primitive types."""
        try:
            # Line shader
            line_vertex_shader = """
            #version 330 core
            in vec2 in_position;
            uniform mat4 u_projection;
            uniform vec4 u_color;
            out vec4 v_color;

            void main() {
                gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
                v_color = u_color;
            }
            """

            line_fragment_shader = """
            #version 330 core
            in vec4 v_color;
            out vec4 fragColor;

            void main() {
                fragColor = v_color;
            }
            """

            self._line_program = self.ctx.program(
                vertex_shader=line_vertex_shader, fragment_shader=line_fragment_shader
            )

            # Circle shader (for filled circles)
            circle_vertex_shader = """
            #version 330 core
            in vec2 in_position;
            in vec2 in_uv;
            uniform mat4 u_projection;
            uniform vec4 u_color;
            out vec4 v_color;
            out vec2 v_uv;

            void main() {
                gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
                v_color = u_color;
                v_uv = in_uv;
            }
            """

            circle_fragment_shader = """
            #version 330 core
            in vec4 v_color;
            in vec2 v_uv;
            out vec4 fragColor;

            void main() {
                float dist = length(v_uv - vec2(0.5, 0.5));
                if (dist > 0.5) {
                    discard;
                }
                fragColor = v_color;
            }
            """

            self._circle_program = self.ctx.program(
                vertex_shader=circle_vertex_shader,
                fragment_shader=circle_fragment_shader,
            )

            # Rectangle shader
            rect_vertex_shader = """
            #version 330 core
            in vec2 in_position;
            uniform mat4 u_projection;
            uniform vec4 u_color;
            out vec4 v_color;

            void main() {
                gl_Position = u_projection * vec4(in_position, 0.0, 1.0);
                v_color = u_color;
            }
            """

            rect_fragment_shader = """
            #version 330 core
            in vec4 v_color;
            out vec4 fragColor;

            void main() {
                fragColor = v_color;
            }
            """

            self._rect_program = self.ctx.program(
                vertex_shader=rect_vertex_shader, fragment_shader=rect_fragment_shader
            )

            logger.debug("Shaders initialized successfully")

        except Exception as e:
            logger.error(f"Shader initialization failed: {e}")
            raise RendererError(f"Shader setup failed: {e}")

    def _initialize_buffers(self) -> None:
        """Initialize vertex buffers and arrays."""
        try:
            # Line buffer (will be updated per draw)
            line_vertices = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            line_vbo = self.ctx.buffer(line_vertices.tobytes())
            self._line_vao = self.ctx.vertex_array(
                self._line_program, [(line_vbo, "2f", "in_position")]
            )

            # Circle buffer (quad with UV coordinates)
            circle_vertices = np.array(
                [
                    # Position  UV
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,  # Bottom-left
                    1.0,
                    -1.0,
                    1.0,
                    0.0,  # Bottom-right
                    1.0,
                    1.0,
                    1.0,
                    1.0,  # Top-right
                    -1.0,
                    1.0,
                    0.0,
                    1.0,  # Top-left
                ],
                dtype=np.float32,
            )

            circle_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

            circle_vbo = self.ctx.buffer(circle_vertices.tobytes())
            circle_ibo = self.ctx.buffer(circle_indices.tobytes())

            self._circle_vao = self.ctx.vertex_array(
                self._circle_program,
                [(circle_vbo, "2f 2f", "in_position", "in_uv")],
                circle_ibo,
            )

            # Rectangle buffer (will be updated per draw)
            rect_vertices = np.array(
                [
                    0.0,
                    0.0,  # Bottom-left
                    1.0,
                    0.0,  # Bottom-right
                    1.0,
                    1.0,  # Top-right
                    0.0,
                    1.0,  # Top-left
                ],
                dtype=np.float32,
            )

            rect_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

            rect_vbo = self.ctx.buffer(rect_vertices.tobytes())
            rect_ibo = self.ctx.buffer(rect_indices.tobytes())

            self._rect_vao = self.ctx.vertex_array(
                self._rect_program, [(rect_vbo, "2f", "in_position")], rect_ibo
            )

            logger.debug("Vertex buffers initialized successfully")

        except Exception as e:
            logger.error(f"Buffer initialization failed: {e}")
            raise RendererError(f"Buffer setup failed: {e}")

    def begin_frame(self) -> None:
        """Begin a new rendering frame."""
        self.stats.frames_rendered += 1
        self.stats.shapes_rendered = 0
        self.stats.draw_calls = 0
        self._frame_start_time = time.time()

    def end_frame(self) -> None:
        """End the current rendering frame."""
        frame_time = time.time() - self._frame_start_time
        self.stats.last_frame_time = frame_time
        self.stats.total_render_time += frame_time

        if self.stats.frames_rendered > 0:
            self.stats.average_frame_time = (
                self.stats.total_render_time / self.stats.frames_rendered
            )

    def set_projection_matrix(self, width: int, height: int) -> None:
        """Set the projection matrix for orthographic rendering.

        Args:
            width: Display width in pixels
            height: Display height in pixels
        """
        # Create orthographic projection matrix
        projection = np.array(
            [
                [2.0 / width, 0.0, 0.0, -1.0],
                [0.0, -2.0 / height, 0.0, 1.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Store projection matrix for external use
        self.projection_matrix = projection

        # Update all shader programs
        for program in [self._line_program, self._circle_program, self._rect_program]:
            if program and "u_projection" in program:
                program["u_projection"].write(projection.tobytes())

    def set_color(self, color: Union[Color, tuple[float, float, float, float]]) -> None:
        """Set the current rendering color.

        Args:
            color: Color to use for subsequent rendering
        """
        if isinstance(color, tuple):
            self._current_color = Color(*color)
        else:
            self._current_color = color

    def set_line_width(self, width: float) -> None:
        """Set the current line width.

        Args:
            width: Line width in pixels
        """
        self._current_line_width = max(1.0, width)

    def set_blend_mode(self, mode: BlendMode) -> None:
        """Set the blending mode for rendering.

        Args:
            mode: Blending mode to use
        """
        self._current_blend_mode = mode

        # Apply blend mode
        if mode == BlendMode.NORMAL:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        elif mode == BlendMode.ADDITIVE:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        elif mode == BlendMode.MULTIPLY:
            self.ctx.blend_func = moderngl.DST_COLOR, moderngl.ZERO
        elif mode == BlendMode.OVERLAY:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def draw_line(
        self,
        start: Point2D,
        end: Point2D,
        color: Optional[Color] = None,
        width: Optional[float] = None,
        style: LineStyle = LineStyle.SOLID,
    ) -> None:
        """Draw a line between two points.

        Args:
            start: Starting point
            end: Ending point
            color: Line color (uses current if None)
            width: Line width (uses current if None)
            style: Line style
        """
        try:
            render_color = color or self._current_color
            render_width = width or self._current_line_width

            # Set line width
            self.ctx.line_width = render_width

            # Update line vertices
            vertices = np.array([start.x, start.y, end.x, end.y], dtype=np.float32)
            line_vbo = self.ctx.buffer(vertices.tobytes())

            # Create new VAO for this line
            line_vao = self.ctx.vertex_array(
                self._line_program, [(line_vbo, "2f", "in_position")]
            )

            # Set color uniform
            self._line_program["u_color"] = render_color.to_tuple()

            # Render based on style
            if style == LineStyle.SOLID:
                line_vao.render(moderngl.LINES)
            elif style == LineStyle.DASHED:
                # For dashed lines, we'll need to break the line into segments
                self._draw_dashed_line(start, end, render_color, render_width)
                return
            elif style == LineStyle.DOTTED:
                self._draw_dotted_line(start, end, render_color, render_width)
                return
            elif style == LineStyle.ARROW:
                # Draw line + arrowhead
                line_vao.render(moderngl.LINES)
                self._draw_arrowhead(start, end, render_color, render_width)

            self.stats.shapes_rendered += 1
            self.stats.draw_calls += 1

        except Exception as e:
            logger.warning(f"Line drawing failed: {e}")

    def draw_circle(
        self,
        center: Point2D,
        radius: float,
        color: Optional[Color] = None,
        filled: bool = True,
        outline_width: float = 2.0,
    ) -> None:
        """Draw a circle.

        Args:
            center: Circle center point
            radius: Circle radius
            color: Circle color (uses current if None)
            filled: Whether to fill the circle
            outline_width: Width of outline if not filled
        """
        try:
            render_color = color or self._current_color

            if filled:
                self._draw_filled_circle(center, radius, render_color)
            else:
                self._draw_circle_outline(center, radius, render_color, outline_width)

            self.stats.shapes_rendered += 1
            self.stats.draw_calls += 1

        except Exception as e:
            logger.warning(f"Circle drawing failed: {e}")

    def draw_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Optional[Color] = None,
        filled: bool = True,
        outline_width: float = 2.0,
    ) -> None:
        """Draw a rectangle.

        Args:
            x: Left edge X coordinate
            y: Bottom edge Y coordinate
            width: Rectangle width
            height: Rectangle height
            color: Rectangle color (uses current if None)
            filled: Whether to fill the rectangle
            outline_width: Width of outline if not filled
        """
        try:
            render_color = color or self._current_color

            if filled:
                self._draw_filled_rectangle(x, y, width, height, render_color)
            else:
                self._draw_rectangle_outline(
                    x, y, width, height, render_color, outline_width
                )

            self.stats.shapes_rendered += 1
            self.stats.draw_calls += 1

        except Exception as e:
            logger.warning(f"Rectangle drawing failed: {e}")

    def _draw_filled_circle(self, center: Point2D, radius: float, color: Color) -> None:
        """Draw a filled circle using a quad with circle shader."""
        # Calculate transform matrix for the circle
        np.array(
            [
                [radius, 0.0, 0.0, center.x],
                [0.0, radius, 0.0, center.y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Set uniforms and render
        self._circle_program["u_color"] = color.to_tuple()
        # Note: For full implementation, we'd multiply projection * transform
        # For now, we'll use the basic circle VAO
        self._circle_vao.render()

    def _draw_circle_outline(
        self, center: Point2D, radius: float, color: Color, width: float
    ) -> None:
        """Draw a circle outline using line segments."""
        num_segments = max(16, int(radius * 0.5))  # More segments for larger circles
        angle_step = 2 * math.pi / num_segments

        points = []
        for i in range(num_segments + 1):  # +1 to close the circle
            angle = i * angle_step
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append(Point2D(x, y))

        # Draw line segments
        self.ctx.line_width = width
        self._line_program["u_color"] = color.to_tuple()

        for i in range(len(points) - 1):
            vertices = np.array(
                [points[i].x, points[i].y, points[i + 1].x, points[i + 1].y],
                dtype=np.float32,
            )

            vbo = self.ctx.buffer(vertices.tobytes())
            vao = self.ctx.vertex_array(
                self._line_program, [(vbo, "2f", "in_position")]
            )
            vao.render(moderngl.LINES)

    def _draw_filled_rectangle(
        self, x: float, y: float, width: float, height: float, color: Color
    ) -> None:
        """Draw a filled rectangle."""
        vertices = np.array(
            [
                x,
                y,  # Bottom-left
                x + width,
                y,  # Bottom-right
                x + width,
                y + height,  # Top-right
                x,
                y + height,  # Top-left
            ],
            dtype=np.float32,
        )

        vbo = self.ctx.buffer(vertices.tobytes())
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        ibo = self.ctx.buffer(indices.tobytes())

        vao = self.ctx.vertex_array(
            self._rect_program, [(vbo, "2f", "in_position")], ibo
        )

        self._rect_program["u_color"] = color.to_tuple()
        vao.render()

    def _draw_rectangle_outline(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Color,
        line_width: float,
    ) -> None:
        """Draw a rectangle outline."""
        # Draw four lines
        lines = [
            (Point2D(x, y), Point2D(x + width, y)),  # Bottom
            (Point2D(x + width, y), Point2D(x + width, y + height)),  # Right
            (Point2D(x + width, y + height), Point2D(x, y + height)),  # Top
            (Point2D(x, y + height), Point2D(x, y)),  # Left
        ]

        for start, end in lines:
            self.draw_line(start, end, color, line_width)

    def _draw_dashed_line(
        self, start: Point2D, end: Point2D, color: Color, width: float
    ) -> None:
        """Draw a dashed line."""
        dash_length = 10.0
        gap_length = 5.0
        total_length = start.distance_to(end)

        if total_length == 0:
            return

        # Calculate direction vector
        dx = (end.x - start.x) / total_length
        dy = (end.y - start.y) / total_length

        current_pos = 0.0
        drawing_dash = True

        while current_pos < total_length:
            segment_length = dash_length if drawing_dash else gap_length
            next_pos = min(current_pos + segment_length, total_length)

            if drawing_dash:
                segment_start = Point2D(
                    start.x + dx * current_pos, start.y + dy * current_pos
                )
                segment_end = Point2D(start.x + dx * next_pos, start.y + dy * next_pos)
                self.draw_line(
                    segment_start, segment_end, color, width, LineStyle.SOLID
                )

            current_pos = next_pos
            drawing_dash = not drawing_dash

    def _draw_dotted_line(
        self, start: Point2D, end: Point2D, color: Color, width: float
    ) -> None:
        """Draw a dotted line."""
        dot_spacing = 8.0
        total_length = start.distance_to(end)

        if total_length == 0:
            return

        num_dots = int(total_length / dot_spacing)
        if num_dots == 0:
            return

        # Calculate direction vector
        dx = (end.x - start.x) / total_length
        dy = (end.y - start.y) / total_length

        for i in range(num_dots + 1):
            pos = i * dot_spacing
            if pos > total_length:
                break

            dot_x = start.x + dx * pos
            dot_y = start.y + dy * pos
            dot_center = Point2D(dot_x, dot_y)

            # Draw a small circle for each dot
            self.draw_circle(dot_center, width * 0.5, color, filled=True)

    def _draw_arrowhead(
        self, start: Point2D, end: Point2D, color: Color, width: float
    ) -> None:
        """Draw an arrowhead at the end of a line."""
        arrow_length = width * 3
        arrow_angle = math.pi / 6  # 30 degrees

        if start.distance_to(end) == 0:
            return

        # Calculate line direction
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.sqrt(dx * dx + dy * dy)
        dx /= length
        dy /= length

        # Calculate arrowhead points
        head_x = end.x - arrow_length * dx
        head_y = end.y - arrow_length * dy

        # Calculate perpendicular vector
        perp_x = -dy
        perp_y = dx

        # Arrowhead side points
        side_offset = arrow_length * math.sin(arrow_angle)
        left_x = head_x + perp_x * side_offset
        left_y = head_y + perp_y * side_offset
        right_x = head_x - perp_x * side_offset
        right_y = head_y - perp_y * side_offset

        # Draw arrowhead lines
        self.draw_line(end, Point2D(left_x, left_y), color, width, LineStyle.SOLID)
        self.draw_line(end, Point2D(right_x, right_y), color, width, LineStyle.SOLID)

    def get_stats(self) -> RenderStats:
        """Get rendering performance statistics."""
        return self.stats

    def clear_stats(self) -> None:
        """Clear rendering statistics."""
        self.stats = RenderStats()

    def __str__(self) -> str:
        """String representation."""
        return f"BasicRenderer(frames={self.stats.frames_rendered}, avg_time={self.stats.average_frame_time:.4f}s)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"BasicRenderer("
            f"frames={self.stats.frames_rendered}, "
            f"shapes={self.stats.shapes_rendered}, "
            f"calls={self.stats.draw_calls}, "
            f"avg_time={self.stats.average_frame_time:.4f}s"
            f")"
        )


# Color constants for convenience
class Colors:
    """Predefined colors for common use."""

    WHITE = Color(1.0, 1.0, 1.0, 1.0)
    BLACK = Color(0.0, 0.0, 0.0, 1.0)
    RED = Color(1.0, 0.0, 0.0, 1.0)
    GREEN = Color(0.0, 1.0, 0.0, 1.0)
    BLUE = Color(0.0, 0.0, 1.0, 1.0)
    YELLOW = Color(1.0, 1.0, 0.0, 1.0)
    CYAN = Color(0.0, 1.0, 1.0, 1.0)
    MAGENTA = Color(1.0, 0.0, 1.0, 1.0)
    ORANGE = Color(1.0, 0.5, 0.0, 1.0)
    PURPLE = Color(0.5, 0.0, 1.0, 1.0)
    TRANSPARENT = Color(0.0, 0.0, 0.0, 0.0)

    # Pool-specific colors
    CUE_BALL = Color(1.0, 1.0, 0.9, 1.0)  # Off-white
    TRAJECTORY = Color(0.0, 1.0, 0.0, 0.8)  # Semi-transparent green
    COLLISION = Color(1.0, 0.0, 0.0, 0.8)  # Semi-transparent red
    GHOST_BALL = Color(1.0, 1.0, 1.0, 0.3)  # Very transparent white
