"""Security utilities for authentication and authorization."""

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# API Key configuration
API_KEY_LENGTH = 32
API_KEY_PREFIX = "bt_"


class UserRole(str, Enum):
    """User roles for role-based access control."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class SecurityEventType(str, Enum):
    """Types of security events to log."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_INVALID = "token_invalid"
    API_KEY_USAGE = "api_key_usage"
    API_KEY_INVALID = "api_key_invalid"
    ACCESS_DENIED = "access_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class TokenData(BaseModel):
    """Token payload data."""

    sub: str  # Subject (username)
    role: UserRole
    exp: datetime
    iat: datetime
    jti: str  # JWT ID
    token_type: str = "access"


class APIKeyData(BaseModel):
    """API key data."""

    key_id: str
    name: str
    role: UserRole
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True
    expires_at: Optional[datetime] = None


class SecurityEvent(BaseModel):
    """Security event for logging."""

    event_type: SecurityEventType
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: dict[str, Any] = {}
    success: bool = True


class PasswordPolicy(BaseModel):
    """Password policy configuration."""

    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special: bool = True
    forbidden_patterns: list[str] = ["password", "123456", "admin"]


class SecurityConfig(BaseModel):
    """Security configuration."""

    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_policy: PasswordPolicy = Field(default_factory=PasswordPolicy)
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    require_https: bool = True
    api_key_expire_days: int = 365


class PasswordUtils:
    """Password hashing and verification utilities."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        # Truncate password to 72 bytes max for bcrypt compatibility
        if len(password.encode("utf-8")) > 72:
            password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
        try:
            return pwd_context.hash(password)
        except Exception:
            # Fallback to basic hash if bcrypt fails
            import hashlib

            salt = secrets.token_hex(16)
            return f"fallback${salt}${hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()}"

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        # Truncate password to 72 bytes max for bcrypt compatibility
        if len(plain_password.encode("utf-8")) > 72:
            plain_password = plain_password.encode("utf-8")[:72].decode(
                "utf-8", errors="ignore"
            )

        try:
            # Check if it's a fallback hash
            if hashed_password.startswith("fallback$"):
                parts = hashed_password.split("$")
                if len(parts) >= 3:
                    salt = parts[1]
                    stored_hash = parts[2]
                    import hashlib

                    computed_hash = hashlib.pbkdf2_hmac(
                        "sha256", plain_password.encode(), salt.encode(), 100000
                    ).hex()
                    return hmac.compare_digest(stored_hash, computed_hash)
                return False

            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False

    @staticmethod
    def validate_password_strength(
        password: str, policy: PasswordPolicy = None
    ) -> tuple[bool, list[str]]:
        """Validate password against policy."""
        if policy is None:
            policy = PasswordPolicy()

        errors = []

        # Check length
        if len(password) < policy.min_length:
            errors.append(
                f"Password must be at least {policy.min_length} characters long"
            )

        # Check character requirements
        if policy.require_uppercase and not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")

        if policy.require_lowercase and not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")

        if policy.require_digits and not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")

        if policy.require_special and not re.search(
            r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>?]', password
        ):
            errors.append("Password must contain at least one special character")

        # Check forbidden patterns
        password_lower = password.lower()
        for pattern in policy.forbidden_patterns:
            if pattern.lower() in password_lower:
                errors.append(f"Password cannot contain '{pattern}'")

        return len(errors) == 0, errors


class JWTUtils:
    """JWT token utilities."""

    @staticmethod
    def create_access_token(
        subject: str, role: UserRole, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=ACCESS_TOKEN_EXPIRE_MINUTES
            )

        # Generate unique JWT ID
        jti = secrets.token_urlsafe(16)

        token_data = {
            "sub": subject,
            "role": role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": jti,
            "token_type": "access",
        }

        encoded_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def create_refresh_token(subject: str, role: UserRole) -> str:
        """Create a new refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        jti = secrets.token_urlsafe(16)

        token_data = {
            "sub": subject,
            "role": role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": jti,
            "token_type": "refresh",
        }

        encoded_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Optional[TokenData]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Validate required fields
            if not all(
                field in payload for field in ["sub", "role", "exp", "iat", "jti"]
            ):
                return None

            # Convert timestamps
            exp_timestamp = payload["exp"]
            iat_timestamp = payload["iat"]

            # Handle both timestamp formats
            if isinstance(exp_timestamp, (int, float)):
                exp = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            else:
                exp = exp_timestamp

            if isinstance(iat_timestamp, (int, float)):
                iat = datetime.fromtimestamp(iat_timestamp, tz=timezone.utc)
            else:
                iat = iat_timestamp

            # Validate role
            try:
                role = UserRole(payload["role"])
            except ValueError:
                logger.warning(f"Invalid role in token: {payload['role']}")
                return None

            return TokenData(
                sub=payload["sub"],
                role=role,
                exp=exp,
                iat=iat,
                jti=payload["jti"],
                token_type=payload.get("token_type", "access"),
            )

        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error decoding token: {e}")
            return None

    @staticmethod
    def is_token_expired(token_data: TokenData) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > token_data.exp


class APIKeyUtils:
    """API key utilities."""

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key."""
        key = secrets.token_urlsafe(API_KEY_LENGTH)
        return f"{API_KEY_PREFIX}{key}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def verify_api_key_format(api_key: str) -> bool:
        """Verify API key format."""
        if not api_key.startswith(API_KEY_PREFIX):
            return False

        key_part = api_key[len(API_KEY_PREFIX) :]
        if len(key_part) != 43:  # Base64 URL-safe encoding of 32 bytes
            return False

        # Check if it's valid base64 URL-safe
        try:
            import base64

            base64.urlsafe_b64decode(key_part + "==")  # Add padding
            return True
        except Exception:
            return False


class SecurityEventLogger:
    """Security event logging utility."""

    def __init__(self):
        self.logger = logging.getLogger("security")

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event."""
        log_message = (
            f"Security Event: {event.event_type.value} | "
            f"User: {event.user_id or 'unknown'} | "
            f"IP: {event.ip_address or 'unknown'} | "
            f"Success: {event.success} | "
            f"Details: {event.details}"
        )

        if event.success:
            self.logger.info(log_message)
        else:
            self.logger.warning(log_message)

    def log_login_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: dict[str, Any] = None,
    ) -> None:
        """Log a login attempt."""
        event = SecurityEvent(
            event_type=(
                SecurityEventType.LOGIN_SUCCESS
                if success
                else SecurityEventType.LOGIN_FAILURE
            ),
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
        )
        self.log_event(event)

    def log_api_key_usage(
        self,
        key_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Log API key usage."""
        event = SecurityEvent(
            event_type=(
                SecurityEventType.API_KEY_USAGE
                if success
                else SecurityEventType.API_KEY_INVALID
            ),
            timestamp=datetime.now(timezone.utc),
            user_id=key_id,
            ip_address=ip_address,
            details={"endpoint": endpoint} if endpoint else {},
            success=success,
        )
        self.log_event(event)

    def log_access_denied(
        self,
        user_id: str,
        resource: str,
        required_role: UserRole,
        user_role: UserRole,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log access denied event."""
        event = SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            ip_address=ip_address,
            details={
                "resource": resource,
                "required_role": required_role.value,
                "user_role": user_role.value,
            },
            success=False,
        )
        self.log_event(event)


class InputValidator:
    """Input validation and sanitization utilities."""

    @staticmethod
    def sanitize_string(
        value: str, max_length: int = 255, allowed_chars: str = None
    ) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Remove null bytes and control characters
        sanitized = "".join(
            char for char in value if ord(char) >= 32 or char in "\t\n\r"
        )

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        # Filter allowed characters if specified
        if allowed_chars:
            sanitized = "".join(char for char in sanitized if char in allowed_chars)

        return sanitized.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(email_pattern, email))

    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        """Validate username format."""
        if not username:
            return False, "Username cannot be empty"

        if len(username) < 3:
            return False, "Username must be at least 3 characters"

        if len(username) > 50:
            return False, "Username cannot exceed 50 characters"

        if not re.match(r"^[a-zA-Z0-9_-]+$", username):
            return (
                False,
                "Username can only contain letters, numbers, hyphens, and underscores",
            )

        return True, ""

    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection patterns."""
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(\b(UNION|OR|AND)\b.*\b(SELECT|INSERT|UPDATE|DELETE)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(XP_|SP_)\w+)",
            r"(\b(OPENROWSET|OPENDATASOURCE)\b)",
        ]

        value_upper = value.upper()
        return any(
            re.search(pattern, value_upper, re.IGNORECASE) for pattern in sql_patterns
        )

    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS patterns."""
        xss_patterns = [
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe\b[^>]*>",
            r"<object\b[^>]*>",
            r"<embed\b[^>]*>",
            r"<form\b[^>]*>",
        ]

        return any(re.search(pattern, value, re.IGNORECASE) for pattern in xss_patterns)


# Global instances
security_logger = SecurityEventLogger()
input_validator = InputValidator()


def check_role_permissions(user_role: UserRole, required_role: UserRole) -> bool:
    """Check if user role has required permissions."""
    role_hierarchy = {UserRole.VIEWER: 1, UserRole.OPERATOR: 2, UserRole.ADMIN: 3}

    return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)


def generate_secure_random_string(length: int = 32) -> str:
    """Generate a cryptographically secure random string."""
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """Constant time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())
