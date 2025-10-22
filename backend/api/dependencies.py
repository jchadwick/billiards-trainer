"""Dependency injection functions for FastAPI routes."""

# Import modules - ensure backend dir is in path
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, HTTPException

backend_dir = Path(__file__).parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import Config, config
from core import CoreModule

from .websocket.manager import websocket_manager


class ApplicationState:
    """Application state container."""

    def __init__(self):
        """Initialize application state with default values."""
        self.core_module: Optional[CoreModule] = None
        self.config_module: Optional[Config] = config  # Use singleton config
        self.websocket_manager = websocket_manager
        self.websocket_handler = None
        self.message_broadcaster: Optional[Any] = None  # MessageBroadcaster instance
        self.integration_service: Optional[Any] = None  # IntegrationService instance
        self.startup_time: Optional[float] = None
        self.is_healthy: bool = False
        self.vision_module: Optional[Any] = None  # Will be VisionModule when imported


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


def get_websocket_manager():
    """Get the WebSocket manager instance."""
    if not app_state.websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not available")
    return app_state.websocket_manager


def get_app_state() -> ApplicationState:
    """Get the application state."""
    return app_state
