"""Dependency injection functions for FastAPI routes."""

from typing import Optional

from fastapi import HTTPException

# Import modules with fallback for different import contexts
try:
    from backend.config import ConfigurationModule
    from backend.core import CoreModule
except ImportError:
    # If running from the backend directory directly
    from core import CoreModule

    from config import ConfigurationModule

from .websocket.manager import websocket_manager


class ApplicationState:
    """Application state container."""

    def __init__(self):
        self.core_module: Optional[CoreModule] = None
        self.config_module: Optional[ConfigurationModule] = None
        self.websocket_manager = websocket_manager
        self.websocket_handler = None
        self.startup_time: Optional[float] = None
        self.is_healthy: bool = False


# Global application state
app_state = ApplicationState()


def get_core_module() -> CoreModule:
    """Get the core module instance."""
    if not app_state.core_module:
        raise HTTPException(status_code=503, detail="Core module not available")
    return app_state.core_module


def get_config_module() -> ConfigurationModule:
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
