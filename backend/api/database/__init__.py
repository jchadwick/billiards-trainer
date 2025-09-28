"""Database package for authentication and user management."""

from .connection import DatabaseManager, get_db
from .models import (
    APIKey,
    APIKeyCreateRequest,
    APIKeyResponse,
    Base,
    PasswordReset,
    PasswordResetConfirm,
    PasswordResetRequest,
    SecurityEvent,
    SecurityEventResponse,
    SessionResponse,
    User,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
    UserSession,
    UserUpdateRequest,
)
from .repositories import (
    APIKeyRepository,
    PasswordResetRepository,
    SecurityEventRepository,
    UserRepository,
    UserSessionRepository,
)

__all__ = [
    # Database connection
    "DatabaseManager",
    "get_db",
    # Models
    "Base",
    "User",
    "UserSession",
    "APIKey",
    "SecurityEvent",
    "PasswordReset",
    # Request/Response models
    "UserCreateRequest",
    "UserUpdateRequest",
    "UserResponse",
    "UserListResponse",
    "APIKeyCreateRequest",
    "APIKeyResponse",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "SecurityEventResponse",
    "SessionResponse",
    # Repositories
    "UserRepository",
    "UserSessionRepository",
    "APIKeyRepository",
    "SecurityEventRepository",
    "PasswordResetRepository",
]
