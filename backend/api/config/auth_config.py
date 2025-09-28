"""Authentication configuration management."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..utils.security import PasswordPolicy, SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class AuthenticationConfig:
    """Authentication system configuration."""

    # Database settings
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./billiards_trainer.db"
        )
    )
    database_echo: bool = field(
        default_factory=lambda: os.getenv("DATABASE_ECHO", "false").lower() == "true"
    )

    # JWT settings
    jwt_secret_key: str = field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    jwt_algorithm: str = field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    access_token_expire_minutes: int = field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    )
    refresh_token_expire_days: int = field(
        default_factory=lambda: int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    )

    # Session settings
    session_timeout_minutes: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    )
    max_sessions_per_user: int = field(
        default_factory=lambda: int(os.getenv("MAX_SESSIONS_PER_USER", "10"))
    )

    # Security settings
    max_login_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    )
    lockout_duration_minutes: int = field(
        default_factory=lambda: int(os.getenv("LOCKOUT_DURATION_MINUTES", "15"))
    )
    require_https: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_HTTPS", "true").lower() == "true"
    )

    # Password policy
    password_min_length: int = field(
        default_factory=lambda: int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
    )
    password_require_uppercase: bool = field(
        default_factory=lambda: os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower()
        == "true"
    )
    password_require_lowercase: bool = field(
        default_factory=lambda: os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower()
        == "true"
    )
    password_require_digits: bool = field(
        default_factory=lambda: os.getenv("PASSWORD_REQUIRE_DIGITS", "true").lower()
        == "true"
    )
    password_require_special: bool = field(
        default_factory=lambda: os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower()
        == "true"
    )
    password_forbidden_patterns: list = field(
        default_factory=lambda: ["password", "123456", "admin"]
    )

    # API key settings
    api_key_expire_days: int = field(
        default_factory=lambda: int(os.getenv("API_KEY_EXPIRE_DAYS", "365"))
    )
    api_key_prefix: str = field(
        default_factory=lambda: os.getenv("API_KEY_PREFIX", "bt_")
    )

    # Rate limiting
    rate_limit_requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    )
    rate_limit_requests_per_hour: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", "1000"))
    )

    # Logging and audit
    security_log_level: str = field(
        default_factory=lambda: os.getenv("SECURITY_LOG_LEVEL", "INFO")
    )
    audit_log_retention_days: int = field(
        default_factory=lambda: int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))
    )

    # Email settings (for password reset)
    smtp_host: Optional[str] = field(default_factory=lambda: os.getenv("SMTP_HOST"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_username: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_USERNAME")
    )
    smtp_password: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD")
    )
    smtp_use_tls: bool = field(
        default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    )
    email_from: str = field(
        default_factory=lambda: os.getenv(
            "EMAIL_FROM", "noreply@billiards-trainer.local"
        )
    )

    # Development settings
    create_default_users: bool = field(
        default_factory=lambda: os.getenv("CREATE_DEFAULT_USERS", "true").lower()
        == "true"
    )
    allow_registration: bool = field(
        default_factory=lambda: os.getenv("ALLOW_REGISTRATION", "false").lower()
        == "true"
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.jwt_secret_key:
            import secrets

            self.jwt_secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "JWT_SECRET_KEY not set, generated random key. This should be set in production!"
            )

        if self.password_min_length < 8:
            logger.warning(
                "Password minimum length is less than 8 characters, this is not recommended"
            )

        if not self.require_https:
            logger.warning(
                "HTTPS is not required, this is not recommended for production"
            )

    def get_password_policy(self) -> PasswordPolicy:
        """Get password policy from configuration."""
        return PasswordPolicy(
            min_length=self.password_min_length,
            require_uppercase=self.password_require_uppercase,
            require_lowercase=self.password_require_lowercase,
            require_digits=self.password_require_digits,
            require_special=self.password_require_special,
            forbidden_patterns=self.password_forbidden_patterns,
        )

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return SecurityConfig(
            jwt_secret_key=self.jwt_secret_key,
            jwt_algorithm=self.jwt_algorithm,
            access_token_expire_minutes=self.access_token_expire_minutes,
            refresh_token_expire_days=self.refresh_token_expire_days,
            password_policy=self.get_password_policy(),
            max_login_attempts=self.max_login_attempts,
            lockout_duration_minutes=self.lockout_duration_minutes,
            session_timeout_minutes=self.session_timeout_minutes,
            require_https=self.require_https,
            api_key_expire_days=self.api_key_expire_days,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_url": self.database_url,
            "database_echo": self.database_echo,
            "jwt_algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "refresh_token_expire_days": self.refresh_token_expire_days,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_sessions_per_user": self.max_sessions_per_user,
            "max_login_attempts": self.max_login_attempts,
            "lockout_duration_minutes": self.lockout_duration_minutes,
            "require_https": self.require_https,
            "password_policy": {
                "min_length": self.password_min_length,
                "require_uppercase": self.password_require_uppercase,
                "require_lowercase": self.password_require_lowercase,
                "require_digits": self.password_require_digits,
                "require_special": self.password_require_special,
                "forbidden_patterns": self.password_forbidden_patterns,
            },
            "api_key_expire_days": self.api_key_expire_days,
            "api_key_prefix": self.api_key_prefix,
            "rate_limiting": {
                "requests_per_minute": self.rate_limit_requests_per_minute,
                "requests_per_hour": self.rate_limit_requests_per_hour,
            },
            "audit_log_retention_days": self.audit_log_retention_days,
            "create_default_users": self.create_default_users,
            "allow_registration": self.allow_registration,
        }

    @classmethod
    def from_file(cls, config_path: Path) -> "AuthenticationConfig":
        """Load configuration from file."""
        try:
            with open(config_path) as f:
                data = json.load(f)

            # Create instance with defaults
            config = cls()

            # Update with file data
            for key, value in data.items():
                if hasattr(config, key):
                    if key == "password_policy" and isinstance(value, dict):
                        # Handle password policy nested structure
                        for policy_key, policy_value in value.items():
                            attr_name = f"password_{policy_key}"
                            if hasattr(config, attr_name):
                                setattr(config, attr_name, policy_value)
                    elif key == "rate_limiting" and isinstance(value, dict):
                        # Handle rate limiting nested structure
                        for rate_key, rate_value in value.items():
                            attr_name = f"rate_limit_{rate_key}"
                            if hasattr(config, attr_name):
                                setattr(config, attr_name, rate_value)
                    else:
                        setattr(config, key, value)

            logger.info(f"Loaded authentication configuration from {config_path}")
            return config

        except FileNotFoundError:
            logger.warning(
                f"Configuration file {config_path} not found, using defaults"
            )
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return cls()

    def save_to_file(self, config_path: Path) -> bool:
        """Save configuration to file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove sensitive data before saving
            data = self.to_dict()
            data.pop("jwt_secret_key", None)
            data.pop("database_url", None)

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Saved authentication configuration to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration settings."""
        errors = []

        # Validate JWT settings
        if len(self.jwt_secret_key) < 32:
            errors.append("JWT secret key should be at least 32 characters long")

        # Validate timeouts
        if self.access_token_expire_minutes <= 0:
            errors.append("Access token expiration must be positive")

        if self.refresh_token_expire_days <= 0:
            errors.append("Refresh token expiration must be positive")

        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")

        # Validate security settings
        if self.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")

        if self.lockout_duration_minutes <= 0:
            errors.append("Lockout duration must be positive")

        # Validate password policy
        if self.password_min_length < 4:
            errors.append("Password minimum length should be at least 4")

        # Validate rate limiting
        if self.rate_limit_requests_per_minute <= 0:
            errors.append("Rate limit requests per minute must be positive")

        if self.rate_limit_requests_per_hour <= 0:
            errors.append("Rate limit requests per hour must be positive")

        # Validate email settings if SMTP is configured
        if self.smtp_host:
            if not self.smtp_username:
                errors.append("SMTP username is required when SMTP host is configured")
            if not self.smtp_password:
                errors.append("SMTP password is required when SMTP host is configured")
            if not self.email_from:
                errors.append("Email from address is required when SMTP is configured")

        return len(errors) == 0, errors


class AuthConfigManager:
    """Authentication configuration manager."""

    def __init__(self, config_dir: Path = None):
        """Initialize configuration manager."""
        self.config_dir = config_dir or Path("config")
        self.config_file = self.config_dir / "auth_config.json"
        self._config: Optional[AuthenticationConfig] = None

    def get_config(self) -> AuthenticationConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> AuthenticationConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            return AuthenticationConfig.from_file(self.config_file)
        else:
            # Create default configuration
            config = AuthenticationConfig()
            self.save_config(config)
            return config

    def save_config(self, config: AuthenticationConfig) -> bool:
        """Save configuration to file."""
        success = config.save_to_file(self.config_file)
        if success:
            self._config = config
        return success

    def update_config(self, **kwargs) -> bool:
        """Update configuration with new values."""
        config = self.get_config()

        # Update values
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

        return self.save_config(config)

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate current configuration."""
        return self.get_config().validate()

    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        config = AuthenticationConfig()
        return self.save_config(config)


# Global configuration manager instance
auth_config_manager = AuthConfigManager()


def get_auth_config() -> AuthenticationConfig:
    """Get authentication configuration."""
    return auth_config_manager.get_config()


def update_auth_config(**kwargs) -> bool:
    """Update authentication configuration."""
    return auth_config_manager.update_config(**kwargs)


def validate_auth_config() -> tuple[bool, list[str]]:
    """Validate authentication configuration."""
    return auth_config_manager.validate_config()
