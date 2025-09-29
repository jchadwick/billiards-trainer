"""Authentication and authorization endpoints with database persistence."""

import logging
from datetime import timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ..database import (
    APIKeyCreateRequest,
    APIKeyResponse,
    PasswordResetConfirm,
    PasswordResetRequest,
    SecurityEventResponse,
    SessionResponse,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
    UserUpdateRequest,
    get_db,
)
from ..middleware.authentication import get_current_user, require_admin
from ..services.auth_service import AuthenticationService
from ..utils.security import (
    JWTUtils,
    PasswordUtils,
    SecurityEventType,
    UserRole,
    input_validator,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


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


# Keep existing request/response models that are not defined in database package
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


# Helper functions
def get_auth_service(db: Session = Depends(get_db)) -> AuthenticationService:
    """Get authentication service instance."""
    return AuthenticationService(db)


# Authentication endpoints
@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Authenticate user and return access tokens."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    # Authenticate user using database service
    user = auth_service.authenticate_user(
        username=login_data.username,
        password=login_data.password,
        ip_address=client_ip,
        user_agent=user_agent,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Create tokens
    access_token_expires = timedelta(minutes=30)
    if login_data.remember_me:
        access_token_expires = timedelta(hours=24)

    access_token = JWTUtils.create_access_token(
        subject=user.username, role=user.role, expires_delta=access_token_expires
    )

    refresh_token = JWTUtils.create_refresh_token(subject=user.username, role=user.role)

    # Extract JWT ID for session management
    token_data = JWTUtils.decode_token(access_token)
    if token_data:
        # Create session
        auth_service.create_session(
            user=user,
            jti=token_data.jti,
            ip_address=client_ip,
            user_agent=user_agent,
            expires_in_minutes=int(access_token_expires.total_seconds() / 60),
        )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_token_expires.total_seconds()),
        user={
            "user_id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
        },
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Refresh access token using refresh token."""
    client_ip = request.client.host if request.client else "unknown"

    # Decode refresh token
    token_data = JWTUtils.decode_token(refresh_data.refresh_token)
    if not token_data or token_data.token_type != "refresh":
        auth_service.log_security_event(
            event_type=SecurityEventType.TOKEN_REFRESH.value,
            success=False,
            ip_address=client_ip,
            details={"reason": "invalid_refresh_token"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    # Check if token is expired
    if JWTUtils.is_token_expired(token_data):
        auth_service.log_security_event(
            event_type=SecurityEventType.TOKEN_REFRESH.value,
            success=False,
            user_id=token_data.sub,
            ip_address=client_ip,
            details={"reason": "expired_refresh_token"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired"
        )

    # Get user
    user = auth_service.get_user_by_username(token_data.sub)
    if not user or not user.can_login():
        auth_service.log_security_event(
            event_type=SecurityEventType.TOKEN_REFRESH.value,
            success=False,
            user_id=token_data.sub,
            ip_address=client_ip,
            details={"reason": "user_not_found_or_inactive"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Create new tokens
    access_token = JWTUtils.create_access_token(subject=user.username, role=user.role)

    new_refresh_token = JWTUtils.create_refresh_token(
        subject=user.username, role=user.role
    )

    # Create new session
    new_token_data = JWTUtils.decode_token(access_token)
    if new_token_data:
        auth_service.create_session(
            user=user,
            jti=new_token_data.jti,
            ip_address=client_ip,
        )

    # Log successful refresh
    auth_service.log_security_event(
        event_type=SecurityEventType.TOKEN_REFRESH.value,
        success=True,
        user_id=str(user.id),
        ip_address=client_ip,
        details={},
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=30 * 60,  # 30 minutes
        user={
            "user_id": str(user.id),
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
        },
    )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Logout user and invalidate session."""
    client_ip = request.client.host if request.client else "unknown"

    # Invalidate session if JWT auth
    if current_user.get("auth_type") == "jwt":
        jti = current_user.get("jti")
        if jti:
            auth_service.invalidate_session(jti)

    # Log logout
    auth_service.log_security_event(
        event_type=SecurityEventType.LOGOUT.value,
        success=True,
        user_id=current_user["user_id"],
        ip_address=client_ip,
        details={"auth_type": current_user.get("auth_type")},
    )

    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Logout user from all sessions."""
    client_ip = request.client.host if request.client else "unknown"

    # Invalidate all user sessions
    invalidated_count = auth_service.invalidate_user_sessions(
        user_id=current_user["user_id"],
        ip_address=client_ip,
    )

    return {"message": f"Successfully logged out from {invalidated_count} sessions"}


@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: ChangePasswordRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Change user password."""
    # Only allow JWT-authenticated users to change passwords
    if current_user.get("auth_type") != "jwt":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change requires user authentication",
        )

    user = auth_service.get_user(current_user["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Verify current password
    if not user.verify_password(password_data.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password using service
    auth_service.update_user(user_id=str(user.id), password=password_data.new_password)

    # Invalidate all other sessions for this user
    client_ip = request.client.host if request.client else "unknown"
    auth_service.invalidate_user_sessions(
        user_id=str(user.id),
        exclude_jti=current_user.get("jti"),
        ip_address=client_ip,
    )

    return {
        "message": "Password changed successfully. Other sessions have been invalidated."
    }


# API Key Management Endpoints
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: Request,
    api_key_data: APIKeyCreateRequest,
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Create a new API key."""
    client_ip = request.client.host if request.client else "unknown"

    # Create API key using service
    api_key_record, api_key = auth_service.create_api_key(
        user_id=current_user["user_id"],
        name=api_key_data.name,
        role=api_key_data.role,
        expires_days=api_key_data.expires_days,
        created_by_ip=client_ip,
    )

    return APIKeyResponse(
        id=str(api_key_record.id),
        name=api_key_record.name,
        role=api_key_record.role.value,
        api_key=api_key,  # Only returned on creation
        is_active=api_key_record.is_active,
        last_used=api_key_record.last_used,
        usage_count=api_key_record.usage_count,
        created_at=api_key_record.created_at,
        expires_at=api_key_record.expires_at,
    )


@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
    page: int = 1,
    per_page: int = 50,
):
    """List all API keys."""
    api_keys, _total = auth_service.list_api_keys(
        page=page,
        per_page=per_page,
    )

    return [
        APIKeyResponse(
            id=str(key.id),
            name=key.name,
            role=key.role.value,
            api_key=None,  # Never return the actual key
            is_active=key.is_active,
            last_used=key.last_used,
            usage_count=key.usage_count,
            created_at=key.created_at,
            expires_at=key.expires_at,
        )
        for key in api_keys
    ]


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    request: Request,
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Revoke an API key."""
    client_ip = request.client.host if request.client else "unknown"

    # Revoke API key using service
    revoked = auth_service.revoke_api_key(
        api_key_id=key_id,
        revoked_by_user_id=current_user["user_id"],
        ip_address=client_ip,
    )

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    return {"message": f"API key {key_id} revoked successfully"}


# User and Session Management
@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Get current user information."""
    if current_user.get("auth_type") == "api_key":
        # API key authentication - get the actual user
        user = auth_service.get_user(current_user["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            first_name=user.first_name,
            last_name=user.last_name,
            avatar_url=user.avatar_url,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login,
        )

    # JWT authentication
    user = auth_service.get_user(current_user["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        first_name=user.first_name,
        last_name=user.last_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def get_user_sessions(
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Get all active sessions for current user."""
    # Only available for JWT users
    if current_user.get("auth_type") != "jwt":
        return []

    sessions = auth_service.get_user_sessions(current_user["user_id"])

    return [
        SessionResponse(
            id=str(session.id),
            user_id=str(session.user_id),
            jti=session.jti,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            is_active=session.is_active,
            created_at=session.created_at,
            last_activity=session.last_activity,
            expires_at=session.expires_at,
        )
        for session in sessions
    ]


@router.delete("/sessions/{jti}")
async def revoke_session(
    jti: str,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Revoke a specific session."""
    # Validate session exists and get session data
    session = auth_service.validate_session(jti)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Check permissions - users can revoke their own sessions, admins can revoke any
    if (
        current_user["user_id"] != str(session.user_id)
        and current_user["role"] != UserRole.ADMIN
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only revoke your own sessions",
        )

    # Revoke session
    revoked = auth_service.invalidate_session(jti)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Log session revocation
    client_ip = request.client.host if request.client else "unknown"
    auth_service.log_security_event(
        event_type=SecurityEventType.LOGOUT.value,
        success=True,
        user_id=current_user["user_id"],
        ip_address=client_ip,
        details={"action": "session_revoked", "target_jti": jti},
    )

    return {"message": "Session revoked successfully"}


# User Registration and Management
@router.post("/register", response_model=UserResponse)
async def register_user(
    request: Request,
    user_data: UserCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    current_user: dict[str, Any] = Depends(
        require_admin
    ),  # Only admins can create users
):
    """Register a new user (admin only)."""
    user = auth_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        role=user_data.role,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        is_verified=True,  # Admin-created users are verified by default
    )

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        first_name=user.first_name,
        last_name=user.last_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
    page: int = 1,
    per_page: int = 50,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
):
    """List users with pagination and filtering (admin only)."""
    users, total = auth_service.list_users(
        page=page,
        per_page=per_page,
        role=role,
        is_active=is_active,
        search=search,
    )

    total_pages = (total + per_page - 1) // per_page

    return UserListResponse(
        users=[
            UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                role=user.role.value,
                is_active=user.is_active,
                is_verified=user.is_verified,
                first_name=user.first_name,
                last_name=user.last_name,
                avatar_url=user.avatar_url,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login=user.last_login,
            )
            for user in users
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Get user by ID (users can view their own profile, admins can view any)."""
    # Users can view their own profile, admins can view any
    if current_user["user_id"] != user_id and current_user["role"] != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view your own profile",
        )

    user = auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        first_name=user.first_name,
        last_name=user.last_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Update user (users can update their own profile, admins can update any)."""
    # Users can update their own profile, admins can update any
    # But only admins can change role and is_active
    if current_user["user_id"] != user_id:
        if current_user["role"] != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Can only update your own profile",
            )
    else:
        # Regular users cannot change their role or active status
        if user_data.role is not None or user_data.is_active is not None:
            if current_user["role"] != UserRole.ADMIN:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only admins can change role or active status",
                )

    # Update user
    update_data = user_data.dict(exclude_unset=True)
    user = auth_service.update_user(user_id=user_id, **update_data)

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role.value,
        is_active=user.is_active,
        is_verified=user.is_verified,
        first_name=user.first_name,
        last_name=user.last_name,
        avatar_url=user.avatar_url,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
    hard_delete: bool = False,
):
    """Delete user (admin only)."""
    # Prevent deleting self
    if current_user["user_id"] == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    deleted = auth_service.delete_user(user_id, hard_delete=hard_delete)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    action = "permanently deleted" if hard_delete else "deactivated"
    return {"message": f"User {action} successfully"}


# Password Reset Endpoints
@router.post("/password-reset")
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Request password reset."""
    client_ip = request.client.host if request.client else "unknown"

    # Request password reset (always returns success to prevent email enumeration)
    token = auth_service.request_password_reset(
        email=reset_data.email,
        ip_address=client_ip,
    )

    # In a real application, you would send the token via email
    # For demo purposes, we'll return it (DO NOT DO THIS IN PRODUCTION)
    return {
        "message": "If an account with that email exists, a password reset link has been sent.",
        "reset_token": token,  # REMOVE THIS IN PRODUCTION
    }


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    request: Request,
    reset_data: PasswordResetConfirm,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Confirm password reset with token."""
    client_ip = request.client.host if request.client else "unknown"

    success = auth_service.reset_password(
        token=reset_data.token,
        new_password=reset_data.new_password,
        ip_address=client_ip,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    return {
        "message": "Password reset successfully. Please log in with your new password."
    }


# Security and Audit Endpoints
@router.get("/security-events", response_model=list[SecurityEventResponse])
async def get_security_events(
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
    page: int = 1,
    per_page: int = 100,
    user_id: Optional[str] = None,
    event_type: Optional[str] = None,
    success: Optional[bool] = None,
):
    """Get security events (admin only)."""
    events, _total = auth_service.get_security_events(
        page=page,
        per_page=per_page,
        user_id=user_id,
        event_type=event_type,
        success=success,
    )

    return [
        SecurityEventResponse(
            id=str(event.id),
            user_id=str(event.user_id) if event.user_id else None,
            event_type=event.event_type,
            ip_address=event.ip_address,
            success=event.success,
            details=event.details,
            timestamp=event.timestamp,
        )
        for event in events
    ]


@router.get("/security-stats")
async def get_security_stats(
    current_user: dict[str, Any] = Depends(require_admin),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Get security statistics (admin only)."""
    return auth_service.get_security_stats()


# System endpoints
@router.get("/status")
async def auth_status(
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """Get authentication system status."""
    return auth_service.get_auth_status()
