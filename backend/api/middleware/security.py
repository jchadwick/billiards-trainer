"""Security headers middleware for the billiards trainer API."""

import hashlib
import logging
import secrets
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityConfig(BaseModel):
    """Configuration for security headers middleware."""

    # Content Security Policy
    enable_csp: bool = Field(default=True, description="Enable Content Security Policy")
    csp_directives: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "blob:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'", "ws:", "wss:"],
            "media-src": ["'self'"],
            "object-src": ["'none'"],
            "frame-src": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
        },
        description="Content Security Policy directives",
    )

    # HTTP Strict Transport Security
    enable_hsts: bool = Field(
        default=True, description="Enable HTTP Strict Transport Security"
    )
    hsts_max_age: int = Field(
        default=31536000, description="HSTS max age in seconds (1 year default)"
    )
    hsts_include_subdomains: bool = Field(
        default=True, description="Include subdomains in HSTS"
    )
    hsts_preload: bool = Field(default=False, description="Enable HSTS preload")

    # X-Frame-Options
    enable_frame_options: bool = Field(
        default=True, description="Enable X-Frame-Options header"
    )
    frame_options: str = Field(
        default="DENY",
        description="X-Frame-Options value (DENY, SAMEORIGIN, ALLOW-FROM)",
    )

    # X-Content-Type-Options
    enable_content_type_options: bool = Field(
        default=True, description="Enable X-Content-Type-Options header"
    )

    # X-XSS-Protection
    enable_xss_protection: bool = Field(
        default=True, description="Enable X-XSS-Protection header"
    )
    xss_protection_mode: str = Field(
        default="1; mode=block", description="X-XSS-Protection mode"
    )

    # Referrer Policy
    enable_referrer_policy: bool = Field(
        default=True, description="Enable Referrer-Policy header"
    )
    referrer_policy: str = Field(
        default="strict-origin-when-cross-origin", description="Referrer policy value"
    )

    # Permissions Policy (formerly Feature Policy)
    enable_permissions_policy: bool = Field(
        default=True, description="Enable Permissions-Policy header"
    )
    permissions_policy: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "camera": ["'none'"],
            "microphone": ["'none'"],
            "geolocation": ["'none'"],
            "payment": ["'none'"],
            "usb": ["'none'"],
        },
        description="Permissions policy directives",
    )

    # Cross-Origin Embedder Policy
    enable_coep: bool = Field(
        default=False, description="Enable Cross-Origin-Embedder-Policy header"
    )
    coep_value: str = Field(default="require-corp", description="COEP header value")

    # Cross-Origin Opener Policy
    enable_coop: bool = Field(
        default=False, description="Enable Cross-Origin-Opener-Policy header"
    )
    coop_value: str = Field(default="same-origin", description="COOP header value")

    # Cross-Origin Resource Policy
    enable_corp: bool = Field(
        default=False, description="Enable Cross-Origin-Resource-Policy header"
    )
    corp_value: str = Field(default="same-origin", description="CORP header value")

    # Server header
    hide_server_header: bool = Field(
        default=True, description="Hide or modify Server header"
    )
    custom_server_header: Optional[str] = Field(
        default=None, description="Custom Server header value"
    )

    # X-Powered-By header
    hide_powered_by: bool = Field(
        default=True, description="Remove X-Powered-By header"
    )

    # Custom security headers
    custom_headers: dict[str, str] = Field(
        default_factory=dict, description="Additional custom security headers"
    )

    # Rate limiting headers
    expose_rate_limit_headers: bool = Field(
        default=True, description="Expose rate limiting information in headers"
    )

    # Content sniffing protection
    enable_content_sniffing_protection: bool = Field(
        default=True, description="Enable content sniffing protection"
    )

    # Development mode overrides
    development_mode: bool = Field(
        default=False, description="Enable development mode (relaxed security)"
    )


class NonceGenerator:
    """Generates cryptographic nonces for CSP."""

    @staticmethod
    def generate_nonce() -> str:
        """Generate a cryptographically secure nonce."""
        return secrets.token_urlsafe(16)

    @staticmethod
    def generate_hash(content: str, algorithm: str = "sha256") -> str:
        """Generate a hash for CSP hash sources."""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode("utf-8"))
        return f"'{algorithm}-{hash_obj.digest().hex()}'"


class SecurityHeadersManager:
    """Manages security headers for responses."""

    def __init__(self, config: SecurityConfig):
        """Initialize security headers manager."""
        self.config = config
        self.logger = logging.getLogger("api.security")

    def build_csp_header(self, nonce: Optional[str] = None) -> str:
        """Build Content Security Policy header value."""
        if not self.config.enable_csp:
            return ""

        directives = []
        for directive, sources in self.config.csp_directives.items():
            sources_list = sources.copy()

            # Add nonce to script-src and style-src if provided
            if nonce and directive in ["script-src", "style-src"]:
                sources_list.append(f"'nonce-{nonce}'")

            directive_value = f"{directive} {' '.join(sources_list)}"
            directives.append(directive_value)

        return "; ".join(directives)

    def build_hsts_header(self) -> str:
        """Build HTTP Strict Transport Security header value."""
        if not self.config.enable_hsts:
            return ""

        hsts_parts = [f"max-age={self.config.hsts_max_age}"]

        if self.config.hsts_include_subdomains:
            hsts_parts.append("includeSubDomains")

        if self.config.hsts_preload:
            hsts_parts.append("preload")

        return "; ".join(hsts_parts)

    def build_permissions_policy_header(self) -> str:
        """Build Permissions Policy header value."""
        if not self.config.enable_permissions_policy:
            return ""

        directives = []
        for directive, allowlist in self.config.permissions_policy.items():
            if allowlist:
                allowlist_str = " ".join(allowlist)
                directives.append(f"{directive}=({allowlist_str})")
            else:
                directives.append(f"{directive}=()")

        return ", ".join(directives)

    def apply_security_headers(
        self, response: Response, request: Request, nonce: Optional[str] = None
    ) -> None:
        """Apply security headers to response."""
        headers_applied = []

        try:
            # Content Security Policy
            if self.config.enable_csp:
                csp_value = self.build_csp_header(nonce)
                if csp_value:
                    response.headers["Content-Security-Policy"] = csp_value
                    headers_applied.append("CSP")

            # HTTP Strict Transport Security (only for HTTPS)
            if self.config.enable_hsts and request.url.scheme == "https":
                hsts_value = self.build_hsts_header()
                if hsts_value:
                    response.headers["Strict-Transport-Security"] = hsts_value
                    headers_applied.append("HSTS")

            # X-Frame-Options
            if self.config.enable_frame_options:
                response.headers["X-Frame-Options"] = self.config.frame_options
                headers_applied.append("X-Frame-Options")

            # X-Content-Type-Options
            if self.config.enable_content_type_options:
                response.headers["X-Content-Type-Options"] = "nosniff"
                headers_applied.append("X-Content-Type-Options")

            # X-XSS-Protection
            if self.config.enable_xss_protection:
                response.headers["X-XSS-Protection"] = self.config.xss_protection_mode
                headers_applied.append("X-XSS-Protection")

            # Referrer Policy
            if self.config.enable_referrer_policy:
                response.headers["Referrer-Policy"] = self.config.referrer_policy
                headers_applied.append("Referrer-Policy")

            # Permissions Policy
            if self.config.enable_permissions_policy:
                permissions_value = self.build_permissions_policy_header()
                if permissions_value:
                    response.headers["Permissions-Policy"] = permissions_value
                    headers_applied.append("Permissions-Policy")

            # Cross-Origin Embedder Policy
            if self.config.enable_coep:
                response.headers[
                    "Cross-Origin-Embedder-Policy"
                ] = self.config.coep_value
                headers_applied.append("COEP")

            # Cross-Origin Opener Policy
            if self.config.enable_coop:
                response.headers["Cross-Origin-Opener-Policy"] = self.config.coop_value
                headers_applied.append("COOP")

            # Cross-Origin Resource Policy
            if self.config.enable_corp:
                response.headers[
                    "Cross-Origin-Resource-Policy"
                ] = self.config.corp_value
                headers_applied.append("CORP")

            # Server header modification
            if self.config.hide_server_header:
                if self.config.custom_server_header:
                    response.headers["Server"] = self.config.custom_server_header
                else:
                    # Remove server header if present
                    response.headers.pop("Server", None)
                headers_applied.append("Server")

            # X-Powered-By removal
            if self.config.hide_powered_by:
                response.headers.pop("X-Powered-By", None)
                headers_applied.append("X-Powered-By-Hidden")

            # Custom headers
            for header_name, header_value in self.config.custom_headers.items():
                response.headers[header_name] = header_value
                headers_applied.append(f"Custom-{header_name}")

            # Content sniffing protection
            if self.config.enable_content_sniffing_protection:
                response.headers["X-Download-Options"] = "noopen"
                headers_applied.append("X-Download-Options")

            self.logger.debug(f"Applied security headers: {', '.join(headers_applied)}")

        except Exception as e:
            self.logger.error(f"Error applying security headers: {e}")

    def get_development_config(self) -> SecurityConfig:
        """Get relaxed security configuration for development."""
        dev_config = SecurityConfig(
            development_mode=True,
            enable_hsts=False,  # Don't require HTTPS in dev
            csp_directives={
                "default-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
                "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
                "style-src": ["'self'", "'unsafe-inline'"],
                "img-src": ["'self'", "data:", "blob:", "*"],
                "connect-src": ["'self'", "ws:", "wss:", "*"],
            },
            frame_options="SAMEORIGIN",  # Allow embedding for dev tools
            enable_coep=False,
            enable_coop=False,
            enable_corp=False,
        )
        return dev_config


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for applying security headers to responses."""

    def __init__(self, app, config: Optional[SecurityConfig] = None):
        """Initialize security headers middleware."""
        super().__init__(app)
        self.config = config or SecurityConfig()
        self.security_manager = SecurityHeadersManager(self.config)
        self.logger = logging.getLogger("api.security")

        # Warn about development mode
        if self.config.development_mode:
            self.logger.warning(
                "Security headers middleware running in development mode - some protections are relaxed"
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and apply security headers to response."""
        # Generate nonce for CSP if needed
        nonce = None
        if self.config.enable_csp and any(
            "'nonce-" in sources
            for sources in self.config.csp_directives.values()
            for sources in sources
            if isinstance(sources, list)
        ):
            nonce = NonceGenerator.generate_nonce()
            request.state.csp_nonce = nonce

        # Process request
        response = await call_next(request)

        # Apply security headers
        self.security_manager.apply_security_headers(response, request, nonce)

        # Add security monitoring headers
        if self.config.expose_rate_limit_headers:
            # These would be populated by rate limiting middleware
            pass

        return response


def setup_security_headers(
    app: FastAPI,
    config: Optional[SecurityConfig] = None,
    development_mode: bool = False,
) -> None:
    """Setup security headers middleware.

    Args:
        app: FastAPI application instance
        config: Security configuration
        development_mode: Enable development mode with relaxed security
    """
    if config is None:
        config = SecurityConfig()

    # Override with development config if requested
    if development_mode:
        config = SecurityHeadersManager(config).get_development_config()

    app.add_middleware(SecurityHeadersMiddleware, config=config)

    logger = logging.getLogger(__name__)
    mode = "development" if development_mode else "production"
    logger.info(f"Security headers middleware enabled in {mode} mode")


def get_csp_nonce(request: Request) -> Optional[str]:
    """Get CSP nonce from request state."""
    return getattr(request.state, "csp_nonce", None)


def create_secure_cookie_settings(
    secure: bool = True, httponly: bool = True, samesite: str = "strict"
) -> dict[str, Any]:
    """Create secure cookie settings.

    Args:
        secure: Set secure flag (HTTPS only)
        httponly: Set HttpOnly flag
        samesite: SameSite policy (strict, lax, none)

    Returns:
        Dictionary of cookie settings
    """
    return {"secure": secure, "httponly": httponly, "samesite": samesite}


class SecurityAuditLogger:
    """Logger for security-related events."""

    def __init__(self):
        """Initialize security audit logger."""
        self.logger = logging.getLogger("api.security.audit")

    def log_security_event(
        self,
        event_type: str,
        request: Request,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a security event."""
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "details": details or {},
        }

        self.logger.warning(f"Security event: {event_type}", extra=event_data)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Security event types
class SecurityEventType:
    """Security event type constants."""

    SUSPICIOUS_REQUEST = "suspicious_request"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    MALFORMED_REQUEST = "malformed_request"
    POTENTIAL_ATTACK = "potential_attack"
    CSP_VIOLATION = "csp_violation"


# Global security audit logger
_security_audit_logger = SecurityAuditLogger()


def log_security_event(
    event_type: str, request: Request, details: Optional[dict[str, Any]] = None
) -> None:
    """Log a security event using the global audit logger."""
    _security_audit_logger.log_security_event(event_type, request, details)


# Security header validation
def validate_security_headers(response: Response) -> dict[str, bool]:
    """Validate that required security headers are present.

    Args:
        response: HTTP response to validate

    Returns:
        Dictionary of header presence validation results
    """
    required_headers = {
        "Content-Security-Policy": "Content-Security-Policy" in response.headers,
        "X-Frame-Options": "X-Frame-Options" in response.headers,
        "X-Content-Type-Options": "X-Content-Type-Options" in response.headers,
        "X-XSS-Protection": "X-XSS-Protection" in response.headers,
        "Referrer-Policy": "Referrer-Policy" in response.headers,
    }

    return required_headers
