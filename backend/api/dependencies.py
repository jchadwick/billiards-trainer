"""Dependency injection functions for FastAPI routes."""

from typing import Any, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Import modules with fallback for different import contexts
try:
    from ..config import ConfigurationModule
    from ..core import CoreModule
except ImportError:
    # If running from the backend directory directly
    from core import CoreModule

    from config import ConfigurationModule

from .middleware.authentication import verify_jwt_token
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
        self.vision_module: Optional[Any] = None  # Will be VisionModule when imported


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


# Security scheme for Bearer token
security_scheme = HTTPBearer(auto_error=False)


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled in configuration."""
    if not app_state.config_module:
        return False
    try:
        return app_state.config_module.get("api.authentication.enabled", default=False)
    except Exception:
        return False


def _get_unauthenticated_user(role: str = "admin") -> dict[str, Any]:
    """Get default user for unauthenticated mode."""
    return {
        "user_id": "unauthenticated_user",
        "username": "unauthenticated",
        "role": role,
        "auth_type": "unauthenticated",
        "permissions": ["read", "write", "admin"] if role == "admin" else ["read"],
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
    _config: ConfigurationModule = Depends(get_config_module),
) -> dict[str, Any]:
    """Get current user from JWT token or return unauthenticated user."""
    auth_enabled = _is_auth_enabled()

    if not auth_enabled:
        return _get_unauthenticated_user("admin")

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_jwt_token(credentials.credentials)
        return {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "role": payload.get("role", "viewer"),
            "auth_type": "jwt",
            "permissions": payload.get("permissions", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def require_viewer(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Require viewer role or higher."""
    if user["role"] not in ["viewer", "operator", "admin"]:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions: viewer role required"
        )
    return user


async def require_operator(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Require operator role or higher."""
    if user["role"] not in ["operator", "admin"]:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions: operator role required"
        )
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Require admin role."""
    if user["role"] != "admin":
        raise HTTPException(
            status_code=403, detail="Insufficient permissions: admin role required"
        )
    return user


# Legacy aliases for backward compatibility
ViewerRequired = require_viewer
OperatorRequired = require_operator
AdminRequired = require_admin


# Development mode bypass dependencies (deprecated)
def dev_viewer_required() -> dict[str, Any]:
    """Development mode bypass for ViewerRequired."""
    return _get_unauthenticated_user("viewer")


def dev_admin_required() -> dict[str, Any]:
    """Development mode bypass for AdminRequired."""
    return _get_unauthenticated_user("admin")


def dev_operator_required() -> dict[str, Any]:
    """Development mode bypass for OperatorRequired."""
    return _get_unauthenticated_user("operator")
