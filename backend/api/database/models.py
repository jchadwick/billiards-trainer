"""Database models for authentication and user management."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import JSON, Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from ..utils.security import PasswordUtils, UserRole, input_validator

Base = declarative_base()


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)

    # Profile information
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    preferences = Column(JSON, nullable=True, default=dict)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    last_login = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Relationships
    sessions = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan"
    )
    security_events = relationship(
        "SecurityEvent", back_populates="user", cascade="all, delete-orphan"
    )
    password_resets = relationship(
        "PasswordReset", back_populates="user", cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        """Set user password with validation."""
        is_valid, errors = PasswordUtils.validate_password_strength(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")

        self.password_hash = PasswordUtils.hash_password(password)
        self.password_changed_at = datetime.now(timezone.utc)

    def verify_password(self, password: str) -> bool:
        """Verify user password."""
        return PasswordUtils.verify_password(password, self.password_hash)

    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def can_login(self) -> bool:
        """Check if user can login."""
        return self.is_active and not self.is_locked()

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login,
        }


class UserSession(Base):
    """User session model."""

    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    last_activity = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        """Check if session is valid."""
        return self.is_active and not self.is_expired()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "jti": self.jti,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "expires_at": self.expires_at,
        }


class APIKey(Base):
    """API key model."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(SQLEnum(UserRole), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Usage tracking
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        """Check if API key is valid."""
        return self.is_active and not self.is_expired()

    def record_usage(self) -> None:
        """Record API key usage."""
        self.last_used = datetime.now(timezone.utc)
        self.usage_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert API key to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "name": self.name,
            "role": self.role.value,
            "is_active": self.is_active,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class SecurityEvent(Base):
    """Security event model for audit logging."""

    __tablename__ = "security_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    event_type = Column(String(50), nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False)
    details = Column(JSON, nullable=True, default=dict)

    # Timestamps
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )

    # Relationships
    user = relationship("User", back_populates="security_events")

    def to_dict(self) -> dict[str, Any]:
        """Convert security event to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "event_type": self.event_type,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class PasswordReset(Base):
    """Password reset token model."""

    __tablename__ = "password_resets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    is_used = Column(Boolean, nullable=False, default=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="password_resets")

    def is_expired(self) -> bool:
        """Check if reset token is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        """Check if reset token is valid."""
        return not self.is_used and not self.is_expired()

    def to_dict(self) -> dict[str, Any]:
        """Convert password reset to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "is_used": self.is_used,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "used_at": self.used_at,
        }


# Pydantic models for API requests/responses


class UserCreateRequest(BaseModel):
    """User creation request model."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = UserRole.VIEWER

    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        sanitized = input_validator.sanitize_string(v, max_length=50)

        if input_validator.detect_sql_injection(
            sanitized
        ) or input_validator.detect_xss(sanitized):
            raise ValueError("Invalid characters in username")

        is_valid, error_msg = input_validator.validate_username(sanitized)
        if not is_valid:
            raise ValueError(error_msg)

        return sanitized

    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        is_valid, errors = PasswordUtils.validate_password_strength(v)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        return v


class UserUpdateRequest(BaseModel):
    """User update request model."""

    email: Optional[EmailStr] = None
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    avatar_url: Optional[str] = Field(None, max_length=500)
    preferences: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


class UserResponse(BaseModel):
    """User response model."""

    id: str
    username: str
    email: str
    role: str
    is_active: bool
    is_verified: bool
    first_name: Optional[str]
    last_name: Optional[str]
    avatar_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]


class UserListResponse(BaseModel):
    """User list response model."""

    users: list[UserResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class PasswordResetRequest(BaseModel):
    """Password reset request model."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""

    token: str
    new_password: str = Field(..., min_length=8)

    @validator("new_password")
    def validate_password(cls, v):
        """Validate new password strength."""
        is_valid, errors = PasswordUtils.validate_password_strength(v)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        return v


class SecurityEventResponse(BaseModel):
    """Security event response model."""

    id: str
    user_id: Optional[str]
    event_type: str
    ip_address: Optional[str]
    success: bool
    details: dict[str, Any]
    timestamp: datetime


class SessionResponse(BaseModel):
    """Session response model."""

    id: str
    user_id: str
    jti: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    is_active: bool
    created_at: datetime
    last_activity: datetime
    expires_at: datetime


class APIKeyCreateRequest(BaseModel):
    """API key creation request model."""

    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole
    expires_days: Optional[int] = Field(None, ge=1, le=365)

    @validator("name")
    def validate_name(cls, v):
        """Validate API key name."""
        sanitized = input_validator.sanitize_string(v, max_length=100)
        if input_validator.detect_sql_injection(
            sanitized
        ) or input_validator.detect_xss(sanitized):
            raise ValueError("Invalid characters in API key name")
        return sanitized


class APIKeyResponse(BaseModel):
    """API key response model."""

    id: str
    name: str
    role: str
    api_key: Optional[str] = None  # Only returned on creation
    is_active: bool
    last_used: Optional[datetime]
    usage_count: int
    created_at: datetime
    expires_at: Optional[datetime]
