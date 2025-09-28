"""Authentication and authorization endpoints."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator

from ..middleware.authentication import (
    get_current_user,
    require_admin,
    session_manager,
)
from ..utils.security import (
    APIKeyUtils,
    JWTUtils,
    PasswordUtils,
    SecurityEventType,
    UserRole,
    input_validator,
    security_logger,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# For demo purposes - in production, use a proper user database
DEMO_USERS = {
    "admin": {
        "user_id": "admin",
        "username": "admin",
        "email": "admin@billiards-trainer.local",
        "password_hash": PasswordUtils.hash_password("admin123!"),
        "role": UserRole.ADMIN,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
        "failed_attempts": 0,
    },
    "operator": {
        "user_id": "operator",
        "username": "operator",
        "email": "operator@billiards-trainer.local",
        "password_hash": PasswordUtils.hash_password("operator123!"),
        "role": UserRole.OPERATOR,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
        "failed_attempts": 0,
    },
    "viewer": {
        "user_id": "viewer",
        "username": "viewer",
        "email": "viewer@billiards-trainer.local",
        "password_hash": PasswordUtils.hash_password("viewer123!"),
        "role": UserRole.VIEWER,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
        "failed_attempts": 0,
    },
}


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1)
    remember_me: bool = False

    @validator("username")
    def validate_username(cls, v):
        # Sanitize input
        sanitized = input_validator.sanitize_string(v, max_length=50)

        # Check for injection attempts
        if input_validator.detect_sql_injection(
            sanitized
        ) or input_validator.detect_xss(sanitized):
            raise ValueError("Invalid characters in username")

        # Validate format
        is_valid, error_msg = input_validator.validate_username(sanitized)
        if not is_valid:
            raise ValueError(error_msg)

        return sanitized


class LoginResponse(BaseModel):
    """Login response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request model."""

    current_password: str
    new_password: str

    @validator("new_password")
    def validate_new_password(cls, v):
        is_valid, errors = PasswordUtils.validate_password_strength(v)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        return v


class CreateAPIKeyRequest(BaseModel):
    """Create API key request model."""

    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole
    expires_days: Optional[int] = Field(None, ge=1, le=365)

    @validator("name")
    def validate_name(cls, v):
        sanitized = input_validator.sanitize_string(v, max_length=100)
        if input_validator.detect_sql_injection(
            sanitized
        ) or input_validator.detect_xss(sanitized):
            raise ValueError("Invalid characters in API key name")
        return sanitized


class APIKeyResponse(BaseModel):
    """API key response model."""

    key_id: str
    name: str
    role: str
    api_key: str  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime]


class APIKeyInfo(BaseModel):
    """API key info (without the key itself)."""

    key_id: str
    name: str
    role: str
    created_at: datetime
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool


class UserInfo(BaseModel):
    """User information model."""

    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class SessionInfo(BaseModel):
    """Session information model."""

    jti: str
    user_id: str
    role: str
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str]
    is_active: bool


# Helper functions
def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    """Get user by username (demo implementation)."""
    return DEMO_USERS.get(username)


def authenticate_user(username: str, password: str) -> Optional[dict[str, Any]]:
    """Authenticate user credentials."""
    user = get_user_by_username(username)
    if not user:
        return None

    if not user.get("is_active", False):
        return None

    if not PasswordUtils.verify_password(password, user["password_hash"]):
        return None

    return user


# Authentication endpoints
@router.post("/login", response_model=LoginResponse)
async def login(request: Request, login_data: LoginRequest):
    """Authenticate user and return access tokens."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    # Check for rate limiting/lockout
    if session_manager.is_locked_out(client_ip):
        security_logger.log_login_attempt(
            user_id=login_data.username,
            success=False,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"reason": "ip_locked_out"},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed attempts. Please try again later.",
        )

    if session_manager.is_locked_out(login_data.username):
        security_logger.log_login_attempt(
            user_id=login_data.username,
            success=False,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"reason": "user_locked_out"},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Account temporarily locked due to failed login attempts.",
        )

    # Authenticate user
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        # Record failed attempt
        session_manager.record_failed_attempt(client_ip, "login")
        session_manager.record_failed_attempt(login_data.username, "login")

        security_logger.log_login_attempt(
            user_id=login_data.username,
            success=False,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"reason": "invalid_credentials"},
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Clear failed attempts on successful login
    session_manager.clear_failed_attempts(client_ip)
    session_manager.clear_failed_attempts(login_data.username)

    # Create tokens
    access_token_expires = timedelta(minutes=30)
    if login_data.remember_me:
        access_token_expires = timedelta(hours=24)

    access_token = JWTUtils.create_access_token(
        subject=user["username"], role=user["role"], expires_delta=access_token_expires
    )

    refresh_token = JWTUtils.create_refresh_token(
        subject=user["username"], role=user["role"]
    )

    # Extract JWT ID for session management
    token_data = JWTUtils.decode_token(access_token)
    if token_data:
        # Create session
        session_manager.create_session(
            user_id=user["username"],
            jti=token_data.jti,
            role=user["role"],
            ip_address=client_ip,
        )

    # Update user last login
    user["last_login"] = datetime.now(timezone.utc)

    # Log successful login
    security_logger.log_login_attempt(
        user_id=user["username"],
        success=True,
        ip_address=client_ip,
        user_agent=user_agent,
        details={"remember_me": login_data.remember_me},
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds()),
        user={
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"].value,
        },
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: Request, refresh_data: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    client_ip = request.client.host if request.client else "unknown"

    # Decode refresh token
    token_data = JWTUtils.decode_token(refresh_data.refresh_token)
    if not token_data or token_data.token_type != "refresh":
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_REFRESH,
                "timestamp": datetime.now(timezone.utc),
                "ip_address": client_ip,
                "details": {"reason": "invalid_refresh_token"},
                "success": False,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    # Check if token is expired
    if JWTUtils.is_token_expired(token_data):
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_REFRESH,
                "timestamp": datetime.now(timezone.utc),
                "user_id": token_data.sub,
                "ip_address": client_ip,
                "details": {"reason": "expired_refresh_token"},
                "success": False,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired"
        )

    # Get user
    user = get_user_by_username(token_data.sub)
    if not user or not user.get("is_active", False):
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_REFRESH,
                "timestamp": datetime.now(timezone.utc),
                "user_id": token_data.sub,
                "ip_address": client_ip,
                "details": {"reason": "user_not_found_or_inactive"},
                "success": False,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Create new tokens
    access_token = JWTUtils.create_access_token(
        subject=user["username"], role=user["role"]
    )

    new_refresh_token = JWTUtils.create_refresh_token(
        subject=user["username"], role=user["role"]
    )

    # Create new session
    new_token_data = JWTUtils.decode_token(access_token)
    if new_token_data:
        session_manager.create_session(
            user_id=user["username"],
            jti=new_token_data.jti,
            role=user["role"],
            ip_address=client_ip,
        )

    # Log successful refresh
    security_logger.log_event(
        {
            "event_type": SecurityEventType.TOKEN_REFRESH,
            "timestamp": datetime.now(timezone.utc),
            "user_id": user["username"],
            "ip_address": client_ip,
            "details": {},
            "success": True,
        }
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=30 * 60,  # 30 minutes
        user={
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"].value,
        },
    )


@router.post("/logout")
async def logout(
    request: Request, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Logout user and invalidate session."""
    client_ip = request.client.host if request.client else "unknown"

    # Invalidate session if JWT auth
    if current_user.get("auth_type") == "jwt":
        jti = current_user.get("jti")
        if jti:
            session_manager.invalidate_session(jti)
            session_manager.blacklist_token(jti)

    # Log logout
    security_logger.log_event(
        {
            "event_type": SecurityEventType.LOGOUT,
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {"auth_type": current_user.get("auth_type")},
            "success": True,
        }
    )

    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all(
    request: Request, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Logout user from all sessions."""
    client_ip = request.client.host if request.client else "unknown"

    # Invalidate all user sessions
    invalidated_count = session_manager.invalidate_user_sessions(
        current_user["user_id"]
    )

    # Log logout all
    security_logger.log_event(
        {
            "event_type": SecurityEventType.LOGOUT,
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {"logout_all": True, "sessions_invalidated": invalidated_count},
            "success": True,
        }
    )

    return {"message": f"Successfully logged out from {invalidated_count} sessions"}


@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: ChangePasswordRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Change user password."""
    # Only allow JWT-authenticated users to change passwords
    if current_user.get("auth_type") != "jwt":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change requires user authentication",
        )

    user = get_user_by_username(current_user["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Verify current password
    if not PasswordUtils.verify_password(
        password_data.current_password, user["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user["password_hash"] = PasswordUtils.hash_password(password_data.new_password)

    # Invalidate all other sessions for this user
    invalidated_count = session_manager.invalidate_user_sessions(
        current_user["user_id"]
    )

    # Log password change
    client_ip = request.client.host if request.client else "unknown"
    security_logger.log_event(
        {
            "event_type": SecurityEventType.LOGIN_SUCCESS,  # Could add PASSWORD_CHANGE event type
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {
                "password_changed": True,
                "sessions_invalidated": invalidated_count,
            },
            "success": True,
        }
    )

    return {"message": "Password changed successfully. Please log in again."}


# API Key Management Endpoints
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: Request,
    api_key_data: CreateAPIKeyRequest,
    current_user: dict[str, Any] = Depends(require_admin),
):
    """Create a new API key."""
    # Generate API key
    api_key = APIKeyUtils.generate_api_key()
    hashed_key = APIKeyUtils.hash_api_key(api_key)

    # Calculate expiration
    expires_at = None
    if api_key_data.expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=api_key_data.expires_days
        )

    # Create key data
    key_id = f"key_{len(session_manager._api_keys) + 1:04d}"
    key_info = {
        "key_id": key_id,
        "name": api_key_data.name,
        "role": api_key_data.role.value,
        "created_at": datetime.now(timezone.utc),
        "created_by": current_user["user_id"],
        "last_used": None,
        "is_active": True,
        "expires_at": expires_at,
    }

    # Store hashed key
    session_manager.add_api_key(hashed_key, key_info)

    # Log API key creation
    client_ip = request.client.host if request.client else "unknown"
    security_logger.log_event(
        {
            "event_type": SecurityEventType.API_KEY_USAGE,
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {
                "action": "created",
                "key_id": key_id,
                "key_name": api_key_data.name,
                "key_role": api_key_data.role.value,
            },
            "success": True,
        }
    )

    return APIKeyResponse(
        key_id=key_id,
        name=api_key_data.name,
        role=api_key_data.role.value,
        api_key=api_key,  # Only returned on creation
        created_at=key_info["created_at"],
        expires_at=expires_at,
    )


@router.get("/api-keys", response_model=list[APIKeyInfo])
async def list_api_keys(current_user: dict[str, Any] = Depends(require_admin)):
    """List all API keys."""
    api_keys = []
    for _hashed_key, key_data in session_manager._api_keys.items():
        api_keys.append(
            APIKeyInfo(
                key_id=key_data["key_id"],
                name=key_data["name"],
                role=key_data["role"],
                created_at=key_data["created_at"],
                last_used=key_data.get("last_used"),
                expires_at=key_data.get("expires_at"),
                is_active=key_data.get("is_active", True),
            )
        )

    return api_keys


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str, request: Request, current_user: dict[str, Any] = Depends(require_admin)
):
    """Revoke an API key."""
    # Find and revoke the key
    revoked = False
    for hashed_key, key_data in session_manager._api_keys.items():
        if key_data["key_id"] == key_id:
            session_manager.revoke_api_key(hashed_key)
            revoked = True
            break

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Log API key revocation
    client_ip = request.client.host if request.client else "unknown"
    security_logger.log_event(
        {
            "event_type": SecurityEventType.API_KEY_USAGE,
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {"action": "revoked", "key_id": key_id},
            "success": True,
        }
    )

    return {"message": f"API key {key_id} revoked successfully"}


# User and Session Management
@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Get current user information."""
    if current_user.get("auth_type") == "api_key":
        # API key authentication
        return UserInfo(
            user_id=current_user["user_id"],
            username=current_user.get("key_name", "API Key"),
            email="",
            role=current_user["role"].value,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            last_login=None,
        )

    # JWT authentication
    user = get_user_by_username(current_user["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserInfo(
        user_id=user["user_id"],
        username=user["username"],
        email=user["email"],
        role=user["role"].value,
        is_active=user["is_active"],
        created_at=user["created_at"],
        last_login=user.get("last_login"),
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def get_user_sessions(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get all active sessions for current user."""
    # Only available for JWT users
    if current_user.get("auth_type") != "jwt":
        return []

    user_id = current_user["user_id"]
    sessions = []

    # Get all sessions for this user
    if user_id in session_manager._user_sessions:
        for jti in session_manager._user_sessions[user_id]:
            session_data = session_manager.get_session(jti)
            if session_data and session_data.get("is_active", False):
                sessions.append(
                    SessionInfo(
                        jti=jti,
                        user_id=session_data["user_id"],
                        role=session_data["role"],
                        created_at=session_data["created_at"],
                        last_activity=session_data["last_activity"],
                        ip_address=session_data.get("ip_address"),
                        is_active=session_data["is_active"],
                    )
                )

    return sessions


@router.delete("/sessions/{jti}")
async def revoke_session(
    jti: str, request: Request, current_user: dict[str, Any] = Depends(get_current_user)
):
    """Revoke a specific session."""
    # Only allow users to revoke their own sessions, or admins to revoke any session
    session_data = session_manager.get_session(jti)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Check permissions
    if (
        current_user["user_id"] != session_data["user_id"]
        and current_user["role"] != UserRole.ADMIN
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only revoke your own sessions",
        )

    # Revoke session
    session_manager.invalidate_session(jti)
    session_manager.blacklist_token(jti)

    # Log session revocation
    client_ip = request.client.host if request.client else "unknown"
    security_logger.log_event(
        {
            "event_type": SecurityEventType.LOGOUT,
            "timestamp": datetime.now(timezone.utc),
            "user_id": current_user["user_id"],
            "ip_address": client_ip,
            "details": {"action": "session_revoked", "target_jti": jti},
            "success": True,
        }
    )

    return {"message": "Session revoked successfully"}


# System endpoints
@router.get("/status")
async def auth_status():
    """Get authentication system status."""
    return {
        "service": "authentication",
        "status": "healthy",
        "active_sessions": len(
            [
                s
                for s in session_manager._active_sessions.values()
                if s.get("is_active", False)
            ]
        ),
        "active_api_keys": len(
            [k for k in session_manager._api_keys.values() if k.get("is_active", False)]
        ),
        "blacklisted_tokens": len(session_manager._blacklisted_tokens),
    }
