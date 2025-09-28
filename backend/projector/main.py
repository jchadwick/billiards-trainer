"""Main projector module entry point."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DisplayMode(Enum):
    FULLSCREEN = "fullscreen"
    WINDOW = "window"
    BORDERLESS = "borderless"


class LineStyle(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    ARROW = "arrow"


@dataclass
class Point2D:
    x: float
    y: float


class ProjectorModule:
    """Main projector interface."""

    def __init__(self, config: dict[str, Any]):
        """Initialize projector module with configuration."""
        pass

    def start_display(self, mode: DisplayMode = DisplayMode.FULLSCREEN) -> bool:
        """Start projector display output."""
        pass

    def stop_display(self) -> None:
        """Stop projector display."""
        pass

    def render_trajectory(
        self, points: list[Point2D], color: tuple[int, int, int, int]
    ) -> None:
        """Render a trajectory path."""
        pass

    def clear_display(self) -> None:
        """Clear all rendered content."""
        pass
