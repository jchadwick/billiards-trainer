"""Comprehensive error handling middleware for the billiards trainer API."""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    error: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request correlation ID"
    )
    path: Optional[str] = Field(default=None, description="Request path")
    method: Optional[str] = Field(default=None, description="HTTP method")


# Error code mappings from SPECS.md
ERROR_CODES = {
    # Client errors (4xx)
    "AUTH_001": {"status": 401, "message": "Invalid credentials"},
    "AUTH_002": {"status": 401, "message": "Token expired"},
    "AUTH_003": {"status": 403, "message": "Insufficient permissions"},
    "VAL_001": {"status": 400, "message": "Invalid request format"},
    "VAL_002": {"status": 400, "message": "Missing required parameter"},
    "VAL_003": {"status": 400, "message": "Parameter out of range"},
    "RES_001": {"status": 404, "message": "Resource not found"},
    "RATE_001": {"status": 429, "message": "Rate limit exceeded"},
    # Server errors (5xx)
    "SYS_001": {"status": 500, "message": "Internal server error"},
    "CAM_001": {"status": 503, "message": "Camera unavailable"},
    "PROC_001": {"status": 503, "message": "Vision processing failed"},
    "WS_001": {"status": 503, "message": "WebSocket service unavailable"},
    # Additional error codes
    "CONFIG_001": {"status": 400, "message": "Invalid configuration"},
    "CONFIG_002": {"status": 500, "message": "Configuration load failed"},
    "CALIB_001": {"status": 400, "message": "Calibration in progress"},
    "CALIB_002": {"status": 500, "message": "Calibration failed"},
    "GAME_001": {"status": 404, "message": "Game session not found"},
    "GAME_002": {"status": 409, "message": "Game state conflict"},
    "NET_001": {"status": 502, "message": "Network connectivity error"},
    "NET_002": {"status": 504, "message": "Request timeout"},
}


class ErrorHandlerConfig(BaseModel):
    """Error handler configuration."""

    include_traceback: bool = Field(
        default=False, description="Include stack traces in error responses"
    )
    log_errors: bool = Field(
        default=True, description="Log errors to the application logger"
    )
    log_level: str = Field(default="ERROR", description="Log level for error messages")
    include_request_details: bool = Field(
        default=True, description="Include request details in error responses"
    )
    sanitize_errors: bool = Field(
        default=True, description="Sanitize error messages for security"
    )
    max_detail_length: int = Field(
        default=1000, description="Maximum length of error detail strings"
    )


class CustomHTTPException(HTTPException):
    """Custom HTTP exception with error codes."""

    def __init__(
        self,
        error_code: str,
        details: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize custom HTTP exception.

        Args:
            error_code: Error code from ERROR_CODES
            details: Additional error details
            headers: HTTP headers to include
        """
        if error_code not in ERROR_CODES:
            raise ValueError(f"Unknown error code: {error_code}")

        error_info = ERROR_CODES[error_code]
        super().__init__(
            status_code=error_info["status"],
            detail=error_info["message"],
            headers=headers,
        )
        self.error_code = error_code
        self.error_details = details or {}


class ErrorHandler:
    """Centralized error handler for the application."""

    def __init__(self, config: ErrorHandlerConfig):
        """Initialize error handler with configuration."""
        self.config = config
        self.error_count = 0
        self.last_errors = []  # Keep track of recent errors for monitoring

    def format_error_response(
        self,
        error_code: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
        request: Optional[Request] = None,
        request_id: Optional[str] = None,
    ) -> ErrorResponse:
        """Format standardized error response.

        Args:
            error_code: Error code identifier
            message: Error message
            details: Additional error details
            request: FastAPI request object
            request_id: Request correlation ID

        Returns:
            Formatted error response
        """
        # Sanitize error details if configured
        if details and self.config.sanitize_errors:
            details = self._sanitize_details(details)

        error_response = ErrorResponse(
            error=error_code, message=message, details=details, request_id=request_id
        )

        # Add request details if configured and available
        if request and self.config.include_request_details:
            error_response.path = str(request.url.path)
            error_response.method = request.method

        return error_response

    def log_error(
        self,
        error: Exception,
        request: Optional[Request] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log error with context information.

        Args:
            error: Exception that occurred
            request: FastAPI request object
            request_id: Request correlation ID
            additional_context: Additional context information
        """
        if not self.config.log_errors:
            return

        context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "request_id": request_id,
        }

        if request:
            context.update(
                {
                    "method": request.method,
                    "path": str(request.url.path),
                    "query_params": dict(request.query_params),
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("User-Agent", ""),
                }
            )

        if additional_context:
            context.update(additional_context)

        # Include traceback if configured
        if self.config.include_traceback:
            context["traceback"] = traceback.format_exc()

        # Log at configured level
        log_level = getattr(logging, self.config.log_level.upper(), logging.ERROR)
        logger.log(log_level, f"Error occurred: {str(error)}", extra=context)

        # Track error for monitoring
        self._track_error(error, context)

    def _sanitize_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """Sanitize error details for security.

        Args:
            details: Raw error details

        Returns:
            Sanitized error details
        """
        sanitized = {}
        sensitive_keys = {"password", "token", "secret", "key", "auth", "credential"}

        for key, value in details.items():
            # Check for sensitive information
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > self.config.max_detail_length:
                sanitized[key] = value[: self.config.max_detail_length] + "..."
            else:
                sanitized[key] = value

        return sanitized

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _track_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Track error for monitoring purposes."""
        self.error_count += 1
        error_info = {
            "timestamp": datetime.utcnow(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }

        # Keep only last 100 errors to prevent memory issues
        self.last_errors.append(error_info)
        if len(self.last_errors) > 100:
            self.last_errors.pop(0)

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics for monitoring."""
        recent_errors = [
            err
            for err in self.last_errors
            if (datetime.utcnow() - err["timestamp"]).total_seconds() < 3600
        ]

        error_types = {}
        for error in recent_errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": self.error_count,
            "recent_errors_1h": len(recent_errors),
            "error_types": error_types,
            "last_error": self.last_errors[-1] if self.last_errors else None,
        }


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_request_id(request: Request) -> str:
    """Get or generate request ID for correlation."""
    # Check if request ID was set by tracing middleware
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id

    # Generate new request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    global _error_handler

    request_id = get_request_id(request)

    # Handle custom HTTP exceptions
    if isinstance(exc, CustomHTTPException):
        error_response = (
            _error_handler.format_error_response(
                error_code=exc.error_code,
                message=exc.detail,
                details=exc.error_details,
                request=request,
                request_id=request_id,
            )
            if _error_handler
            else ErrorResponse(
                error=exc.error_code,
                message=exc.detail,
                details=exc.error_details,
                request_id=request_id,
            )
        )
    else:
        # Map status codes to error codes
        status_code_mapping = {
            400: "VAL_001",
            401: "AUTH_001",
            403: "AUTH_003",
            404: "RES_001",
            429: "RATE_001",
            500: "SYS_001",
            503: "SYS_001",
        }

        error_code = status_code_mapping.get(exc.status_code, "SYS_001")
        error_response = (
            _error_handler.format_error_response(
                error_code=error_code,
                message=exc.detail,
                request=request,
                request_id=request_id,
            )
            if _error_handler
            else ErrorResponse(
                error=error_code, message=exc.detail, request_id=request_id
            )
        )

    # Log the error
    if _error_handler:
        _error_handler.log_error(exc, request, request_id)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    global _error_handler

    request_id = get_request_id(request)

    # Format validation errors
    error_details = {"validation_errors": []}

    for error in exc.errors():
        error_details["validation_errors"].append(
            {
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input"),
            }
        )

    error_response = (
        _error_handler.format_error_response(
            error_code="VAL_001",
            message="Request validation failed",
            details=error_details,
            request=request,
            request_id=request_id,
        )
        if _error_handler
        else ErrorResponse(
            error="VAL_001",
            message="Request validation failed",
            details=error_details,
            request_id=request_id,
        )
    )

    # Log the error
    if _error_handler:
        _error_handler.log_error(exc, request, request_id)

    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions."""
    global _error_handler

    request_id = get_request_id(request)

    # Determine error code based on exception type
    error_code = "SYS_001"  # Default to internal server error

    # Map specific exception types to error codes
    if "camera" in str(exc).lower() or "cv2" in str(exc).lower():
        error_code = "CAM_001"
    elif "vision" in str(exc).lower() or "processing" in str(exc).lower():
        error_code = "PROC_001"
    elif "websocket" in str(exc).lower():
        error_code = "WS_001"
    elif "timeout" in str(exc).lower():
        error_code = "NET_002"

    error_details = {"exception_type": type(exc).__name__}
    if _error_handler and _error_handler.config.include_traceback:
        error_details["traceback"] = traceback.format_exc()

    error_response = (
        _error_handler.format_error_response(
            error_code=error_code,
            message=ERROR_CODES[error_code]["message"],
            details=error_details,
            request=request,
            request_id=request_id,
        )
        if _error_handler
        else ErrorResponse(
            error=error_code,
            message=ERROR_CODES[error_code]["message"],
            details=error_details,
            request_id=request_id,
        )
    )

    # Log the error
    if _error_handler:
        _error_handler.log_error(
            exc, request, request_id, {"original_message": str(exc)}
        )

    return JSONResponse(
        status_code=ERROR_CODES[error_code]["status"],
        content=error_response.model_dump(mode="json"),
        headers={"X-Request-ID": request_id},
    )


def setup_error_handling(
    app: FastAPI, config: Optional[ErrorHandlerConfig] = None
) -> None:
    """Setup comprehensive error handling for FastAPI application.

    Args:
        app: FastAPI application instance
        config: Error handler configuration
    """
    global _error_handler

    if config is None:
        config = ErrorHandlerConfig()

    _error_handler = ErrorHandler(config)

    # Register exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Error handling middleware configured")


def get_error_handler() -> Optional[ErrorHandler]:
    """Get the global error handler instance."""
    return _error_handler


def raise_error(error_code: str, details: Optional[dict[str, Any]] = None) -> None:
    """Raise a custom HTTP exception with error code.

    Args:
        error_code: Error code from ERROR_CODES
        details: Additional error details

    Raises:
        CustomHTTPException: With the specified error code
    """
    raise CustomHTTPException(error_code=error_code, details=details)
