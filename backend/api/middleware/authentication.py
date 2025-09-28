"""Authentication middleware for FastAPI."""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..utils.security import (
    APIKeyUtils,
    JWTUtils,
    SecurityEventType,
    TokenData,
    UserRole,
    check_role_permissions,
    security_logger,
)

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class SessionManager:
    """Manages user sessions and token blacklisting."""

    def __init__(self):
        # In-memory storage for demo - in production use Redis
        self._active_sessions: dict[str, dict[str, Any]] = {}
        self._blacklisted_tokens: set = set()
        self._user_sessions: dict[str, set] = {}  # user_id -> set of jti
        self._api_keys: dict[str, dict[str, Any]] = {}  # hashed_key -> key_data
        self._failed_attempts: dict[str, dict[str, Any]] = {}  # ip -> attempts info
        self._session_timeout = timedelta(minutes=60)

    def create_session(
        self, user_id: str, jti: str, role: UserRole, ip_address: str = None
    ) -> None:
        """Create a new session."""
        session_data = {
            "user_id": user_id,
            "role": role.value,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "ip_address": ip_address,
            "is_active": True,
        }

        self._active_sessions[jti] = session_data

        # Track user sessions
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = set()
        self._user_sessions[user_id].add(jti)

    def get_session(self, jti: str) -> Optional[dict[str, Any]]:
        """Get session data."""
        return self._active_sessions.get(jti)

    def update_session_activity(self, jti: str) -> bool:
        """Update session last activity."""
        if jti in self._active_sessions:
            self._active_sessions[jti]["last_activity"] = datetime.now(timezone.utc)
            return True
        return False

    def invalidate_session(self, jti: str) -> bool:
        """Invalidate a specific session."""
        if jti in self._active_sessions:
            session = self._active_sessions[jti]
            session["is_active"] = False
            user_id = session["user_id"]

            # Remove from user sessions
            if user_id in self._user_sessions:
                self._user_sessions[user_id].discard(jti)

            return True
        return False

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        count = 0
        if user_id in self._user_sessions:
            sessions_to_invalidate = self._user_sessions[user_id].copy()
            for jti in sessions_to_invalidate:
                if self.invalidate_session(jti):
                    count += 1
            self._user_sessions[user_id].clear()
        return count

    def is_session_active(self, jti: str) -> bool:
        """Check if session is active and not expired."""
        session = self.get_session(jti)
        if not session or not session.get("is_active", False):
            return False

        # Check session timeout
        last_activity = session["last_activity"]
        if datetime.now(timezone.utc) - last_activity > self._session_timeout:
            self.invalidate_session(jti)
            return False

        return True

    def blacklist_token(self, jti: str) -> None:
        """Add token to blacklist."""
        self._blacklisted_tokens.add(jti)

    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        return jti in self._blacklisted_tokens

    def add_api_key(self, hashed_key: str, key_data: dict[str, Any]) -> None:
        """Add API key to storage."""
        self._api_keys[hashed_key] = key_data

    def get_api_key(self, hashed_key: str) -> Optional[dict[str, Any]]:
        """Get API key data."""
        return self._api_keys.get(hashed_key)

    def revoke_api_key(self, hashed_key: str) -> bool:
        """Revoke an API key."""
        if hashed_key in self._api_keys:
            self._api_keys[hashed_key]["is_active"] = False
            return True
        return False

    def record_failed_attempt(
        self, identifier: str, attempt_type: str = "login"
    ) -> None:
        """Record a failed authentication attempt."""
        now = datetime.now(timezone.utc)
        if identifier not in self._failed_attempts:
            self._failed_attempts[identifier] = {
                "count": 0,
                "first_attempt": now,
                "last_attempt": now,
                "type": attempt_type,
            }

        self._failed_attempts[identifier]["count"] += 1
        self._failed_attempts[identifier]["last_attempt"] = now

    def is_locked_out(
        self, identifier: str, max_attempts: int = 5, lockout_duration: int = 15
    ) -> bool:
        """Check if identifier is locked out due to failed attempts."""
        if identifier not in self._failed_attempts:
            return False

        attempts = self._failed_attempts[identifier]
        if attempts["count"] < max_attempts:
            return False

        # Check if lockout period has expired
        lockout_end = attempts["last_attempt"] + timedelta(minutes=lockout_duration)
        if datetime.now(timezone.utc) > lockout_end:
            # Reset attempts after lockout period
            del self._failed_attempts[identifier]
            return False

        return True

    def clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts for identifier."""
        self._failed_attempts.pop(identifier, None)

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed."""
        now = datetime.now(timezone.utc)
        expired_sessions = []

        for jti, session in self._active_sessions.items():
            last_activity = session["last_activity"]
            if now - last_activity > self._session_timeout:
                expired_sessions.append(jti)

        for jti in expired_sessions:
            self.invalidate_session(jti)

        return len(expired_sessions)


# Global session manager instance
session_manager = SessionManager()


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware to handle JWT and API key authentication."""

    def __init__(self, app, require_https: bool = True):
        super().__init__(app)
        self.require_https = require_https

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request through authentication middleware."""
        # HTTPS enforcement
        if self.require_https and request.url.scheme != "https":
            # Allow HTTP for localhost/127.0.0.1 in development
            host = request.url.hostname
            if host not in ("localhost", "127.0.0.1"):
                return Response(
                    content="HTTPS required",
                    status_code=status.HTTP_426_UPGRADE_REQUIRED,
                    headers={"Upgrade": "TLS/1.3"},
                )

        # Clean up expired sessions periodically
        if hasattr(request.state, "session_cleanup_time"):
            last_cleanup = getattr(request.state, "session_cleanup_time", 0)
            if time.time() - last_cleanup > 300:  # Every 5 minutes
                asyncio.create_task(self._cleanup_sessions())
                request.state.session_cleanup_time = time.time()

        response = await call_next(request)
        return response

    async def _cleanup_sessions(self):
        """Background task to cleanup expired sessions."""
        try:
            removed = session_manager.cleanup_expired_sessions()
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired sessions")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")


async def get_current_user_jwt(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[TokenData]:
    """Extract and validate JWT token from request."""
    if not credentials:
        return None

    token = credentials.credentials
    if not token:
        return None

    # Decode token
    token_data = JWTUtils.decode_token(token)
    if not token_data:
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_INVALID,
                "timestamp": datetime.now(timezone.utc),
                "ip_address": request.client.host if request.client else None,
                "details": {"reason": "invalid_token"},
                "success": False,
            }
        )
        return None

    # Check if token is blacklisted
    if session_manager.is_token_blacklisted(token_data.jti):
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_INVALID,
                "timestamp": datetime.now(timezone.utc),
                "user_id": token_data.sub,
                "ip_address": request.client.host if request.client else None,
                "details": {"reason": "blacklisted_token"},
                "success": False,
            }
        )
        return None

    # Check if token is expired
    if JWTUtils.is_token_expired(token_data):
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_INVALID,
                "timestamp": datetime.now(timezone.utc),
                "user_id": token_data.sub,
                "ip_address": request.client.host if request.client else None,
                "details": {"reason": "expired_token"},
                "success": False,
            }
        )
        return None

    # Check session validity
    if not session_manager.is_session_active(token_data.jti):
        security_logger.log_event(
            {
                "event_type": SecurityEventType.TOKEN_INVALID,
                "timestamp": datetime.now(timezone.utc),
                "user_id": token_data.sub,
                "ip_address": request.client.host if request.client else None,
                "details": {"reason": "inactive_session"},
                "success": False,
            }
        )
        return None

    # Update session activity
    session_manager.update_session_activity(token_data.jti)

    return token_data


async def get_current_user_api_key(
    request: Request, api_key: Optional[str] = Security(api_key_header)
) -> Optional[dict[str, Any]]:
    """Extract and validate API key from request."""
    if not api_key:
        return None

    # Validate API key format
    if not APIKeyUtils.verify_api_key_format(api_key):
        security_logger.log_api_key_usage(
            key_id="invalid_format",
            success=False,
            ip_address=request.client.host if request.client else None,
            endpoint=str(request.url.path),
        )
        return None

    # Hash the API key for lookup
    hashed_key = APIKeyUtils.hash_api_key(api_key)

    # Get API key data
    key_data = session_manager.get_api_key(hashed_key)
    if not key_data:
        security_logger.log_api_key_usage(
            key_id="unknown",
            success=False,
            ip_address=request.client.host if request.client else None,
            endpoint=str(request.url.path),
        )
        return None

    # Check if API key is active
    if not key_data.get("is_active", False):
        security_logger.log_api_key_usage(
            key_id=key_data.get("key_id", "unknown"),
            success=False,
            ip_address=request.client.host if request.client else None,
            endpoint=str(request.url.path),
        )
        return None

    # Check if API key is expired
    expires_at = key_data.get("expires_at")
    if expires_at and datetime.now(timezone.utc) > expires_at:
        security_logger.log_api_key_usage(
            key_id=key_data.get("key_id", "unknown"),
            success=False,
            ip_address=request.client.host if request.client else None,
            endpoint=str(request.url.path),
        )
        return None

    # Update last used timestamp
    key_data["last_used"] = datetime.now(timezone.utc)

    # Log successful API key usage
    security_logger.log_api_key_usage(
        key_id=key_data.get("key_id", "unknown"),
        success=True,
        ip_address=request.client.host if request.client else None,
        endpoint=str(request.url.path),
    )

    return key_data


async def get_current_user(
    request: Request,
    jwt_user: Optional[TokenData] = Depends(get_current_user_jwt),
    api_key_user: Optional[dict[str, Any]] = Depends(get_current_user_api_key),
) -> dict[str, Any]:
    """Get current authenticated user from JWT or API key."""
    # Check JWT authentication first
    if jwt_user:
        return {
            "user_id": jwt_user.sub,
            "role": jwt_user.role,
            "auth_type": "jwt",
            "jti": jwt_user.jti,
        }

    # Check API key authentication
    if api_key_user:
        return {
            "user_id": api_key_user.get("key_id"),
            "role": UserRole(api_key_user.get("role")),
            "auth_type": "api_key",
            "key_name": api_key_user.get("name"),
        }

    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(required_role: UserRole):
    """Dependency to require specific role."""

    async def role_checker(
        request: Request, current_user: dict[str, Any] = Depends(get_current_user)
    ) -> dict[str, Any]:
        user_role = current_user["role"]

        if not check_role_permissions(user_role, required_role):
            security_logger.log_access_denied(
                user_id=current_user["user_id"],
                resource=str(request.url.path),
                required_role=required_role,
                user_role=user_role,
                ip_address=request.client.host if request.client else None,
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_role.value}",
            )

        return current_user

    return role_checker


def require_admin():
    """Dependency to require admin role."""
    return require_role(UserRole.ADMIN)


def require_operator():
    """Dependency to require operator role or higher."""
    return require_role(UserRole.OPERATOR)


def require_viewer():
    """Dependency to require viewer role or higher (any authenticated user)."""
    return require_role(UserRole.VIEWER)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(
        self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_buckets: dict[str, dict[str, Any]] = {}
        self._hour_buckets: dict[str, dict[str, Any]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limits
        if self._is_rate_limited(client_ip):
            security_logger.log_event(
                {
                    "event_type": SecurityEventType.RATE_LIMIT_EXCEEDED,
                    "timestamp": datetime.now(timezone.utc),
                    "ip_address": client_ip,
                    "details": {"endpoint": str(request.url.path)},
                    "success": False,
                }
            )

            return Response(
                content="Rate limit exceeded",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "60"},
            )

        # Record request
        self._record_request(client_ip)

        response = await call_next(request)
        return response

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()

        # Check minute bucket
        minute_key = f"{client_ip}:{int(now // 60)}"
        if minute_key in self._minute_buckets:
            if self._minute_buckets[minute_key]["count"] >= self.requests_per_minute:
                return True

        # Check hour bucket
        hour_key = f"{client_ip}:{int(now // 3600)}"
        if hour_key in self._hour_buckets:
            if self._hour_buckets[hour_key]["count"] >= self.requests_per_hour:
                return True

        return False

    def _record_request(self, client_ip: str):
        """Record a request for rate limiting."""
        now = time.time()

        # Update minute bucket
        minute_key = f"{client_ip}:{int(now // 60)}"
        if minute_key not in self._minute_buckets:
            self._minute_buckets[minute_key] = {"count": 0, "window_start": now}
        self._minute_buckets[minute_key]["count"] += 1

        # Update hour bucket
        hour_key = f"{client_ip}:{int(now // 3600)}"
        if hour_key not in self._hour_buckets:
            self._hour_buckets[hour_key] = {"count": 0, "window_start": now}
        self._hour_buckets[hour_key]["count"] += 1

        # Clean up old buckets
        self._cleanup_buckets(now)

    def _cleanup_buckets(self, current_time: float):
        """Clean up expired rate limit buckets."""
        # Clean minute buckets (keep last 2 minutes)
        cutoff_minute = current_time - 120
        self._minute_buckets = {
            k: v
            for k, v in self._minute_buckets.items()
            if v["window_start"] > cutoff_minute
        }

        # Clean hour buckets (keep last 2 hours)
        cutoff_hour = current_time - 7200
        self._hour_buckets = {
            k: v
            for k, v in self._hour_buckets.items()
            if v["window_start"] > cutoff_hour
        }


# Authentication dependencies for different roles
AdminRequired = require_admin()
OperatorRequired = require_operator()
ViewerRequired = require_viewer()
