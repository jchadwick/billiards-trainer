"""Database repositories for user management."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from ..utils.security import APIKeyUtils, UserRole
from .models import APIKey, PasswordReset, SecurityEvent, User, UserSession


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session


class UserRepository(BaseRepository):
    """Repository for user management operations."""

    def create(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_verified: bool = False,
        **kwargs,
    ) -> User:
        """Create a new user."""
        user = User(
            username=username,
            email=email,
            role=role,
            first_name=first_name,
            last_name=last_name,
            is_verified=is_verified,
            **kwargs,
        )
        user.set_password(password)

        self.session.add(user)
        self.session.flush()  # Get the ID without committing
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.session.query(User).filter(User.id == user_id).first()

    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.session.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(User.email == email).first()

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        # Try username first, then email
        user = (
            self.session.query(User)
            .filter(or_(User.username == username, User.email == username))
            .first()
        )

        if user and user.verify_password(password):
            return user
        return None

    def update(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user information."""
        user = self.get_by_id(user_id)
        if not user:
            return None

        # Handle special fields
        if "password" in kwargs:
            user.set_password(kwargs.pop("password"))

        # Update other fields
        for key, value in kwargs.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)

        user.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        return user

    def list_users(
        self,
        page: int = 1,
        per_page: int = 50,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> tuple[list[User], int]:
        """List users with pagination and filtering."""
        query = self.session.query(User)

        # Apply filters
        if role is not None:
            query = query.filter(User.role == role)

        if is_active is not None:
            query = query.filter(User.is_active == is_active)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    User.username.ilike(search_term),
                    User.email.ilike(search_term),
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term),
                )
            )

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * per_page
        users = (
            query.order_by(User.created_at.desc()).offset(offset).limit(per_page).all()
        )

        return users, total

    def count(self) -> int:
        """Get total user count."""
        return self.session.query(User).count()

    def delete(self, user_id: str) -> bool:
        """Delete user (soft delete by deactivating)."""
        user = self.get_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        return True

    def hard_delete(self, user_id: str) -> bool:
        """Permanently delete user and all related data."""
        user = self.get_by_id(user_id)
        if not user:
            return False

        self.session.delete(user)
        self.session.flush()
        return True

    def record_login(self, user_id: str, _ip_address: Optional[str] = None) -> User:
        """Record successful login."""
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.now(timezone.utc)
            user.failed_login_attempts = 0
            user.locked_until = None
            self.session.flush()
        return user

    def record_failed_login(
        self, username: str, max_attempts: int = 5
    ) -> Optional[User]:
        """Record failed login attempt and lock if necessary."""
        user = self.get_by_username(username)
        if not user:
            return None

        user.failed_login_attempts += 1

        # Lock account if too many failed attempts
        if user.failed_login_attempts >= max_attempts:
            user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)

        self.session.flush()
        return user

    def unlock_user(self, user_id: str) -> bool:
        """Unlock user account."""
        user = self.get_by_id(user_id)
        if not user:
            return False

        user.failed_login_attempts = 0
        user.locked_until = None
        self.session.flush()
        return True


class UserSessionRepository(BaseRepository):
    """Repository for user session management."""

    def create(
        self,
        user_id: str,
        jti: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_in_minutes: int = 60,
    ) -> UserSession:
        """Create a new user session."""
        session = UserSession(
            user_id=user_id,
            jti=jti,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.now(timezone.utc)
            + timedelta(minutes=expires_in_minutes),
        )

        self.session.add(session)
        self.session.flush()
        return session

    def get_by_jti(self, jti: str) -> Optional[UserSession]:
        """Get session by JWT ID."""
        return self.session.query(UserSession).filter(UserSession.jti == jti).first()

    def get_user_sessions(
        self, user_id: str, active_only: bool = True
    ) -> list[UserSession]:
        """Get all sessions for a user."""
        query = self.session.query(UserSession).filter(UserSession.user_id == user_id)

        if active_only:
            query = query.filter(
                and_(
                    UserSession.is_active is True,
                    UserSession.expires_at > datetime.now(timezone.utc),
                )
            )

        return query.order_by(UserSession.created_at.desc()).all()

    def update_activity(self, jti: str) -> bool:
        """Update session last activity."""
        session = self.get_by_jti(jti)
        if not session:
            return False

        session.last_activity = datetime.now(timezone.utc)
        self.session.flush()
        return True

    def invalidate(self, jti: str) -> bool:
        """Invalidate a session."""
        session = self.get_by_jti(jti)
        if not session:
            return False

        session.is_active = False
        self.session.flush()
        return True

    def invalidate_user_sessions(
        self, user_id: str, exclude_jti: Optional[str] = None
    ) -> int:
        """Invalidate all sessions for a user."""
        query = self.session.query(UserSession).filter(
            and_(UserSession.user_id == user_id, UserSession.is_active is True)
        )

        if exclude_jti:
            query = query.filter(UserSession.jti != exclude_jti)

        sessions = query.all()
        count = 0

        for session in sessions:
            session.is_active = False
            count += 1

        self.session.flush()
        return count

    def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired_sessions = (
            self.session.query(UserSession).filter(UserSession.expires_at <= now).all()
        )

        count = 0
        for session in expired_sessions:
            session.is_active = False
            count += 1

        self.session.flush()
        return count

    def is_valid(self, jti: str) -> bool:
        """Check if session is valid."""
        session = self.get_by_jti(jti)
        return session is not None and session.is_valid()


class APIKeyRepository(BaseRepository):
    """Repository for API key management."""

    def create(
        self,
        user_id: str,
        name: str,
        role: UserRole,
        expires_days: Optional[int] = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key."""
        # Generate API key
        api_key = APIKeyUtils.generate_api_key()
        key_hash = APIKeyUtils.hash_api_key(api_key)

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

        # Create API key record
        api_key_record = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            role=role,
            expires_at=expires_at,
        )

        self.session.add(api_key_record)
        self.session.flush()
        return api_key_record, api_key

    def get_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        return self.session.query(APIKey).filter(APIKey.key_hash == key_hash).first()

    def get_by_id(self, api_key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self.session.query(APIKey).filter(APIKey.id == api_key_id).first()

    def get_user_keys(self, user_id: str, active_only: bool = True) -> list[APIKey]:
        """Get all API keys for a user."""
        query = self.session.query(APIKey).filter(APIKey.user_id == user_id)

        if active_only:
            now = datetime.now(timezone.utc)
            query = query.filter(
                and_(
                    APIKey.is_active is True,
                    or_(APIKey.expires_at.is_(None), APIKey.expires_at > now),
                )
            )

        return query.order_by(APIKey.created_at.desc()).all()

    def list_keys(
        self,
        page: int = 1,
        per_page: int = 50,
        user_id: Optional[str] = None,
        active_only: bool = True,
    ) -> tuple[list[APIKey], int]:
        """List API keys with pagination."""
        query = self.session.query(APIKey)

        if user_id:
            query = query.filter(APIKey.user_id == user_id)

        if active_only:
            now = datetime.now(timezone.utc)
            query = query.filter(
                and_(
                    APIKey.is_active is True,
                    or_(APIKey.expires_at.is_(None), APIKey.expires_at > now),
                )
            )

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * per_page
        keys = (
            query.order_by(APIKey.created_at.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )

        return keys, total

    def record_usage(self, key_hash: str) -> bool:
        """Record API key usage."""
        api_key = self.get_by_hash(key_hash)
        if not api_key:
            return False

        api_key.record_usage()
        self.session.flush()
        return True

    def revoke(self, api_key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self.get_by_id(api_key_id)
        if not api_key:
            return False

        api_key.is_active = False
        self.session.flush()
        return True

    def authenticate(self, api_key: str) -> Optional[APIKey]:
        """Authenticate with API key."""
        key_hash = APIKeyUtils.hash_api_key(api_key)
        api_key_record = self.get_by_hash(key_hash)

        if api_key_record and api_key_record.is_valid():
            self.record_usage(key_hash)
            return api_key_record

        return None


class SecurityEventRepository(BaseRepository):
    """Repository for security event logging."""

    def create(
        self,
        event_type: str,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> SecurityEvent:
        """Create a new security event."""
        event = SecurityEvent(
            event_type=event_type,
            success=success,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )

        self.session.add(event)
        self.session.flush()
        return event

    def list_events(
        self,
        page: int = 1,
        per_page: int = 100,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[list[SecurityEvent], int]:
        """List security events with filtering."""
        query = self.session.query(SecurityEvent)

        # Apply filters
        if user_id:
            query = query.filter(SecurityEvent.user_id == user_id)

        if event_type:
            query = query.filter(SecurityEvent.event_type == event_type)

        if success is not None:
            query = query.filter(SecurityEvent.success == success)

        if start_date:
            query = query.filter(SecurityEvent.timestamp >= start_date)

        if end_date:
            query = query.filter(SecurityEvent.timestamp <= end_date)

        # Get total count
        total = query.count()

        # Apply pagination
        offset = (page - 1) * per_page
        events = (
            query.order_by(SecurityEvent.timestamp.desc())
            .offset(offset)
            .limit(per_page)
            .all()
        )

        return events, total

    def get_event_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get security event statistics."""
        query = self.session.query(SecurityEvent)

        if start_date:
            query = query.filter(SecurityEvent.timestamp >= start_date)

        if end_date:
            query = query.filter(SecurityEvent.timestamp <= end_date)

        # Get counts by event type
        event_counts = (
            query.with_entities(
                SecurityEvent.event_type, func.count(SecurityEvent.id).label("count")
            )
            .group_by(SecurityEvent.event_type)
            .all()
        )

        # Get success/failure counts
        success_counts = (
            query.with_entities(
                SecurityEvent.success, func.count(SecurityEvent.id).label("count")
            )
            .group_by(SecurityEvent.success)
            .all()
        )

        return {
            "event_types": dict(event_counts),
            "success_failure": {
                str(success): count for success, count in success_counts
            },
            "total_events": query.count(),
        }

    def cleanup_old_events(self, days_to_keep: int = 90) -> int:
        """Clean up old security events."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        events_to_delete = self.session.query(SecurityEvent).filter(
            SecurityEvent.timestamp < cutoff_date
        )

        count = events_to_delete.count()
        events_to_delete.delete()
        self.session.flush()

        return count


class PasswordResetRepository(BaseRepository):
    """Repository for password reset management."""

    def create(
        self, user_id: str, expires_hours: int = 24
    ) -> tuple[PasswordReset, str]:
        """Create a new password reset token."""
        # Generate reset token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Create reset record
        reset = PasswordReset(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours),
        )

        self.session.add(reset)
        self.session.flush()
        return reset, token

    def get_by_token(self, token: str) -> Optional[PasswordReset]:
        """Get password reset by token."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return (
            self.session.query(PasswordReset)
            .filter(PasswordReset.token_hash == token_hash)
            .first()
        )

    def use_token(self, token: str) -> Optional[PasswordReset]:
        """Mark token as used."""
        reset = self.get_by_token(token)
        if not reset or not reset.is_valid():
            return None

        reset.is_used = True
        reset.used_at = datetime.now(timezone.utc)
        self.session.flush()
        return reset

    def cleanup_expired(self) -> int:
        """Clean up expired reset tokens."""
        now = datetime.now(timezone.utc)
        expired_resets = self.session.query(PasswordReset).filter(
            PasswordReset.expires_at <= now
        )

        count = expired_resets.count()
        expired_resets.delete()
        self.session.flush()

        return count

    def invalidate_user_tokens(self, user_id: str) -> int:
        """Invalidate all reset tokens for a user."""
        user_resets = self.session.query(PasswordReset).filter(
            and_(PasswordReset.user_id == user_id, PasswordReset.is_used is False)
        )

        count = 0
        for reset in user_resets:
            reset.is_used = True
            reset.used_at = datetime.now(timezone.utc)
            count += 1

        self.session.flush()
        return count
