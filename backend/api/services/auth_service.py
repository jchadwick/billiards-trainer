"""Authentication service with database persistence."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..database import (
    APIKeyRepository,
    PasswordResetRepository,
    SecurityEventRepository,
    User,
    UserRepository,
    UserSession,
    UserSessionRepository,
)
from ..utils.security import SecurityEventType, UserRole

logger = logging.getLogger(__name__)


class AuthenticationService:
    """Authentication service with database persistence."""

    def __init__(self, session: Session):
        """Initialize authentication service."""
        self.session = session
        self.user_repo = UserRepository(session)
        self.session_repo = UserSessionRepository(session)
        self.api_key_repo = APIKeyRepository(session)
        self.security_repo = SecurityEventRepository(session)
        self.password_reset_repo = PasswordResetRepository(session)

    # User Management

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_verified: bool = False,
    ) -> User:
        """Create a new user."""
        # Check if username or email already exists
        if self.user_repo.get_by_username(username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        if self.user_repo.get_by_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists"
            )

        try:
            user = self.user_repo.create(
                username=username,
                email=email,
                password=password,
                role=role,
                first_name=first_name,
                last_name=last_name,
                is_verified=is_verified,
            )

            # Log user creation
            self.security_repo.create(
                event_type=SecurityEventType.LOGIN_SUCCESS.value,  # Could add USER_CREATED
                success=True,
                user_id=str(user.id),
                details={"action": "user_created", "role": role.value},
            )

            self.session.commit()
            logger.info(f"User created: {username} ({user.id})")
            return user

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating user {username}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user",
            )

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.user_repo.get_by_id(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.user_repo.get_by_username(username)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.user_repo.get_by_email(email)

    def update_user(self, user_id: str, **kwargs) -> User:
        """Update user information."""
        user = self.user_repo.update(user_id, **kwargs)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        try:
            self.session.commit()
            logger.info(f"User updated: {user.username} ({user_id})")
            return user
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user",
            )

    def list_users(
        self,
        page: int = 1,
        per_page: int = 50,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> tuple[list[User], int]:
        """List users with pagination and filtering."""
        return self.user_repo.list_users(
            page=page,
            per_page=per_page,
            role=role,
            is_active=is_active,
            search=search,
        )

    def delete_user(self, user_id: str, hard_delete: bool = False) -> bool:
        """Delete user (soft delete by default)."""
        try:
            if hard_delete:
                result = self.user_repo.hard_delete(user_id)
            else:
                result = self.user_repo.delete(user_id)

            if result:
                self.session.commit()
                logger.info(f"User {'hard ' if hard_delete else ''}deleted: {user_id}")
            return result

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user",
            )

    # Authentication

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[User]:
        """Authenticate user with username/email and password."""
        try:
            user = self.user_repo.authenticate(username, password)

            if user:
                if not user.can_login():
                    # Log failed login attempt
                    self.security_repo.create(
                        event_type=SecurityEventType.LOGIN_FAILURE.value,
                        success=False,
                        user_id=str(user.id),
                        ip_address=ip_address,
                        user_agent=user_agent,
                        details={"reason": "account_locked_or_inactive"},
                    )
                    self.session.commit()
                    return None

                # Record successful login
                self.user_repo.record_login(str(user.id), ip_address)

                # Log successful login
                self.security_repo.create(
                    event_type=SecurityEventType.LOGIN_SUCCESS.value,
                    success=True,
                    user_id=str(user.id),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={},
                )

                self.session.commit()
                return user
            else:
                # Try to find user to record failed attempt
                user_record = self.user_repo.get_by_username(
                    username
                ) or self.user_repo.get_by_email(username)

                if user_record:
                    self.user_repo.record_failed_login(username)

                # Log failed login attempt
                self.security_repo.create(
                    event_type=SecurityEventType.LOGIN_FAILURE.value,
                    success=False,
                    user_id=str(user_record.id) if user_record else None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={"reason": "invalid_credentials", "username": username},
                )

                self.session.commit()
                return None

        except Exception as e:
            self.session.rollback()
            logger.error(f"Authentication error for {username}: {e}")
            return None

    def create_session(
        self,
        user: User,
        jti: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_in_minutes: int = 60,
    ) -> UserSession:
        """Create a new user session."""
        try:
            session = self.session_repo.create(
                user_id=str(user.id),
                jti=jti,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_in_minutes=expires_in_minutes,
            )

            self.session.commit()
            return session

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating session for user {user.id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session",
            )

    def validate_session(self, jti: str) -> Optional[UserSession]:
        """Validate session by JWT ID."""
        session = self.session_repo.get_by_jti(jti)
        if session and session.is_valid():
            self.session_repo.update_activity(jti)
            self.session.commit()
            return session
        return None

    def invalidate_session(self, jti: str) -> bool:
        """Invalidate a session."""
        try:
            result = self.session_repo.invalidate(jti)
            if result:
                self.session.commit()
            return result
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error invalidating session {jti}: {e}")
            return False

    def invalidate_user_sessions(
        self,
        user_id: str,
        exclude_jti: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> int:
        """Invalidate all sessions for a user."""
        try:
            count = self.session_repo.invalidate_user_sessions(user_id, exclude_jti)

            # Log session invalidation
            self.security_repo.create(
                event_type=SecurityEventType.LOGOUT.value,
                success=True,
                user_id=user_id,
                ip_address=ip_address,
                details={"action": "logout_all", "sessions_invalidated": count},
            )

            self.session.commit()
            return count

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error invalidating user sessions for {user_id}: {e}")
            return 0

    def get_user_sessions(self, user_id: str) -> list[UserSession]:
        """Get all active sessions for a user."""
        return self.session_repo.get_user_sessions(user_id, active_only=True)

    # API Key Management

    def create_api_key(
        self,
        user_id: str,
        name: str,
        role: UserRole,
        expires_days: Optional[int] = None,
        created_by_ip: Optional[str] = None,
    ) -> tuple[Any, str]:  # Returns (APIKey, raw_key)
        """Create a new API key."""
        try:
            api_key_record, api_key = self.api_key_repo.create(
                user_id=user_id,
                name=name,
                role=role,
                expires_days=expires_days,
            )

            # Log API key creation
            self.security_repo.create(
                event_type=SecurityEventType.API_KEY_USAGE.value,
                success=True,
                user_id=user_id,
                ip_address=created_by_ip,
                details={
                    "action": "api_key_created",
                    "key_id": str(api_key_record.id),
                    "key_name": name,
                    "role": role.value,
                },
            )

            self.session.commit()
            logger.info(f"API key created: {name} for user {user_id}")
            return api_key_record, api_key

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating API key for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create API key",
            )

    def authenticate_api_key(
        self,
        api_key: str,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[Any]:  # Returns APIKey or None
        """Authenticate with API key."""
        try:
            api_key_record = self.api_key_repo.authenticate(api_key)

            if api_key_record:
                # Log successful API key usage
                self.security_repo.create(
                    event_type=SecurityEventType.API_KEY_USAGE.value,
                    success=True,
                    user_id=str(api_key_record.user_id),
                    ip_address=ip_address,
                    details={
                        "key_id": str(api_key_record.id),
                        "endpoint": endpoint,
                    },
                )
                self.session.commit()
                return api_key_record
            else:
                # Log failed API key usage
                self.security_repo.create(
                    event_type=SecurityEventType.API_KEY_INVALID.value,
                    success=False,
                    ip_address=ip_address,
                    details={"endpoint": endpoint},
                )
                self.session.commit()
                return None

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error authenticating API key: {e}")
            return None

    def list_api_keys(
        self,
        user_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> tuple[list[Any], int]:
        """List API keys."""
        return self.api_key_repo.list_keys(
            page=page,
            per_page=per_page,
            user_id=user_id,
            active_only=True,
        )

    def revoke_api_key(
        self,
        api_key_id: str,
        revoked_by_user_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Revoke an API key."""
        try:
            result = self.api_key_repo.revoke(api_key_id)

            if result:
                # Log API key revocation
                self.security_repo.create(
                    event_type=SecurityEventType.API_KEY_USAGE.value,
                    success=True,
                    user_id=revoked_by_user_id,
                    ip_address=ip_address,
                    details={
                        "action": "api_key_revoked",
                        "key_id": api_key_id,
                    },
                )

                self.session.commit()
                logger.info(f"API key revoked: {api_key_id}")

            return result

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error revoking API key {api_key_id}: {e}")
            return False

    # Password Reset

    def request_password_reset(
        self,
        email: str,
        ip_address: Optional[str] = None,
    ) -> Optional[str]:  # Returns reset token or None
        """Request password reset for user."""
        user = self.user_repo.get_by_email(email)
        if not user or not user.is_active:
            # Don't reveal if email exists
            return None

        try:
            # Invalidate existing reset tokens
            self.password_reset_repo.invalidate_user_tokens(str(user.id))

            # Create new reset token
            reset, token = self.password_reset_repo.create(str(user.id))

            # Log password reset request
            self.security_repo.create(
                event_type=SecurityEventType.LOGIN_SUCCESS.value,  # Could add PASSWORD_RESET_REQUESTED
                success=True,
                user_id=str(user.id),
                ip_address=ip_address,
                details={"action": "password_reset_requested"},
            )

            self.session.commit()
            logger.info(f"Password reset requested for user {user.id}")
            return token

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error requesting password reset for {email}: {e}")
            return None

    def reset_password(
        self,
        token: str,
        new_password: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Reset password using reset token."""
        try:
            reset = self.password_reset_repo.use_token(token)
            if not reset:
                return False

            # Update user password
            user = self.user_repo.get_by_id(str(reset.user_id))
            if not user:
                return False

            user.set_password(new_password)

            # Invalidate all user sessions
            self.session_repo.invalidate_user_sessions(str(user.id))

            # Log password reset
            self.security_repo.create(
                event_type=SecurityEventType.LOGIN_SUCCESS.value,  # Could add PASSWORD_RESET_COMPLETED
                success=True,
                user_id=str(user.id),
                ip_address=ip_address,
                details={"action": "password_reset_completed"},
            )

            self.session.commit()
            logger.info(f"Password reset completed for user {user.id}")
            return True

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error resetting password: {e}")
            return False

    # Security Events

    def log_security_event(
        self,
        event_type: str,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a security event."""
        try:
            self.security_repo.create(
                event_type=event_type,
                success=success,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
            )
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error logging security event: {e}")

    def get_security_events(
        self,
        page: int = 1,
        per_page: int = 100,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[list[Any], int]:
        """Get security events with filtering."""
        return self.security_repo.list_events(
            page=page,
            per_page=per_page,
            user_id=user_id,
            event_type=event_type,
            success=success,
            start_date=start_date,
            end_date=end_date,
        )

    def get_security_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get security event statistics."""
        return self.security_repo.get_event_stats(start_date, end_date)

    # Maintenance

    def cleanup_expired_data(self) -> dict[str, int]:
        """Clean up expired sessions, tokens, and old events."""
        try:
            results = {
                "expired_sessions": self.session_repo.cleanup_expired(),
                "expired_reset_tokens": self.password_reset_repo.cleanup_expired(),
                "old_security_events": self.security_repo.cleanup_old_events(),
            }

            self.session.commit()
            logger.info(f"Cleanup completed: {results}")
            return results

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}

    def get_auth_status(self) -> dict[str, Any]:
        """Get authentication system status."""
        try:
            active_sessions = len(
                [
                    s
                    for s in self.session_repo.session.query(UserSession).all()
                    if s.is_valid()
                ]
            )

            active_api_keys = len(
                [
                    k
                    for k in self.api_key_repo.session.query(
                        self.api_key_repo.session.query
                    ).all()
                    if k.is_valid()
                ]
            )

            return {
                "service": "authentication",
                "status": "healthy",
                "database_connected": True,
                "total_users": self.user_repo.count(),
                "active_sessions": active_sessions,
                "active_api_keys": active_api_keys,
            }

        except Exception as e:
            logger.error(f"Error getting auth status: {e}")
            return {
                "service": "authentication",
                "status": "unhealthy",
                "database_connected": False,
                "error": str(e),
            }
