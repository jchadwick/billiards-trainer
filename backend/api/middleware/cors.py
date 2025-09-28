"""CORS middleware configuration for the billiards trainer API."""

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class CORSConfig(BaseModel):
    """CORS configuration model."""

    allow_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="List of origins that are allowed to make cross-origin requests",
    )
    allow_credentials: bool = Field(
        default=True,
        description="Whether to allow cookies to be included in cross-origin requests",
    )
    allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="List of HTTP methods that are allowed for cross-origin requests",
    )
    allow_headers: list[str] = Field(
        default=["*"],
        description="List of headers that are allowed for cross-origin requests",
    )
    expose_headers: list[str] = Field(
        default=["X-Request-ID", "X-Response-Time"],
        description="List of headers that are exposed to the browser",
    )
    max_age: int = Field(
        default=600, description="Maximum age in seconds for preflight cache"
    )


def setup_cors_middleware(
    app: FastAPI, config: Optional[CORSConfig] = None, development_mode: bool = False
) -> None:
    """Setup CORS middleware with appropriate configuration.

    Args:
        app: FastAPI application instance
        config: CORS configuration object
        development_mode: If True, allows all origins for development
    """
    if config is None:
        config = CORSConfig()

    # In development mode, allow all origins for easier testing
    if development_mode:
        allow_origins = ["*"]
        allow_credentials = False  # Cannot use credentials with wildcard origins
    else:
        allow_origins = config.allow_origins
        allow_credentials = config.allow_credentials

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        expose_headers=config.expose_headers,
        max_age=config.max_age,
    )


def get_cors_config_from_env() -> CORSConfig:
    """Get CORS configuration from environment variables.

    Returns:
        CORSConfig object with values from environment or defaults
    """
    import os

    # Parse origins from comma-separated string
    origins_str = os.getenv("CORS_ALLOW_ORIGINS", "")
    if origins_str:
        origins = [origin.strip() for origin in origins_str.split(",")]
    else:
        origins = CORSConfig().allow_origins

    # Parse methods from comma-separated string
    methods_str = os.getenv("CORS_ALLOW_METHODS", "")
    if methods_str:
        methods = [method.strip() for method in methods_str.split(",")]
    else:
        methods = CORSConfig().allow_methods

    # Parse headers from comma-separated string
    headers_str = os.getenv("CORS_ALLOW_HEADERS", "")
    if headers_str:
        headers = [header.strip() for header in headers_str.split(",")]
    else:
        headers = CORSConfig().allow_headers

    return CORSConfig(
        allow_origins=origins,
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
        allow_methods=methods,
        allow_headers=headers,
        max_age=int(os.getenv("CORS_MAX_AGE", "600")),
    )


def validate_cors_config(config: CORSConfig) -> None:
    """Validate CORS configuration for security issues.

    Args:
        config: CORS configuration to validate

    Raises:
        ValueError: If configuration has security issues
    """
    # Check for wildcard origins with credentials
    if "*" in config.allow_origins and config.allow_credentials:
        raise ValueError(
            "Cannot use wildcard origins (*) with allow_credentials=True. "
            "This is a security risk."
        )

    # Warn about overly permissive origins in production
    if "*" in config.allow_origins:
        import logging

        logging.warning(
            "Using wildcard (*) for CORS origins. "
            "This should only be used in development."
        )

    # Check for suspicious origins
    suspicious_patterns = ["file://", "data:", "javascript:"]
    for origin in config.allow_origins:
        for pattern in suspicious_patterns:
            if pattern in origin.lower():
                raise ValueError(f"Suspicious origin detected: {origin}")
