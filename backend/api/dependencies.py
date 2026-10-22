"""Dependency injection functions for FastAPI routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, HTTPException

from backend.config import Config, config

from .websocket.manager import WebSocketManager, websocket_manager

if TYPE_CHECKING:
    import asyncio

    from backend.core import CoreModule


class ApplicationState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state with default values."""
        self.core_module: CoreModule | None = None
        self.config_module: Config | None = config  # Use singleton config
        self.websocket_manager = websocket_manager
        self.websocket_handler: Any | None = None
        self.message_broadcaster: Any | None = None  # MessageBroadcaster instance
        self.integration_service: Any | None = None  # IntegrationService instance
        # Optional Rust scene publisher. When present, HTTP bootstrap reads the
        # same defensive scene snapshot that the publisher sends to WebSocket.
        # Keeping this optional preserves the Python-only response path.
        self.scene_state_publisher: Any | None = None
        # Rust owns game-session metadata in Rust live/replay modes. The
        # transitional gateway keeps only a client to that authority.
        self.game_session_authority: Any | None = None
        self.perception_provider: str | None = None
        self.perception_runtime: Any | None = None
        self.performance_broadcast_task: asyncio.Task[None] | None = None
        self.startup_time: float | None = None
        self.is_healthy: bool = False
        self.vision_module: Any | None = None  # Will be VisionModule when imported


# Global application state
app_state = ApplicationState()


def get_core_module() -> CoreModule:
    """Get the core module instance."""
    if not app_state.core_module:
        raise HTTPException(status_code=503, detail="Core module not available")
    return app_state.core_module


def get_config_module() -> Config:
    """Get the configuration module instance."""
    if not app_state.config_module:
        raise HTTPException(
            status_code=503, detail="Configuration module not available"
        )
    return app_state.config_module


def get_websocket_manager() -> WebSocketManager:
    """Get the WebSocket manager instance."""
    if not app_state.websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")
    return app_state.websocket_manager


def get_app_state() -> ApplicationState:
    """Get the application state."""
    return app_state


def get_optional_core_module(
    application: ApplicationState = Depends(get_app_state),
) -> CoreModule | None:
    """Return Core when available without coupling scene localization to Core."""
    return getattr(application, "core_module", None)
