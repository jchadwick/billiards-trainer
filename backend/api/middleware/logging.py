"""Request and response logging middleware for the billiards trainer API."""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse


class LoggingConfig(BaseModel):
    """Configuration for request/response logging."""

    enable_request_logging: bool = Field(
        default=True, description="Enable request logging"
    )
    enable_response_logging: bool = Field(
        default=True, description="Enable response logging"
    )
    log_level: str = Field(
        default="INFO", description="Log level for request/response logs"
    )
    log_headers: bool = Field(default=False, description="Include headers in logs")
    log_body: bool = Field(
        default=False, description="Include request/response body in logs"
    )
    max_body_size: int = Field(
        default=1024, description="Maximum body size to log (bytes)"
    )
    excluded_paths: set[str] = Field(
        default_factory=lambda: {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        },
        description="Paths to exclude from logging",
    )
    excluded_headers: set[str] = Field(
        default_factory=lambda: {
            "authorization",
            "cookie",
            "x-api-key",
            "x-auth-token",
        },
        description="Headers to exclude from logging (case insensitive)",
    )
    include_performance_metrics: bool = Field(
        default=True, description="Include performance metrics in logs"
    )
    slow_request_threshold: float = Field(
        default=1.0, description="Threshold in seconds to mark requests as slow"
    )


class RequestMetrics(BaseModel):
    """Request performance metrics."""

    request_id: str
    method: str
    path: str
    status_code: Optional[int] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    error_message: Optional[str] = None


class RequestLogger:
    """Request/response logger with configurable options."""

    def __init__(self, config: LoggingConfig):
        """Initialize request logger with configuration."""
        self.config = config
        self.logger = logging.getLogger("api.requests")
        self.performance_logger = logging.getLogger("api.performance")
        self.metrics_storage = []  # Store recent metrics for monitoring

    def should_log_path(self, path: str) -> bool:
        """Check if path should be logged."""
        return path not in self.config.excluded_paths

    def sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive headers from logging."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.config.excluded_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized

    def truncate_body(self, body: bytes) -> str:
        """Truncate body for logging if it's too large."""
        if len(body) > self.config.max_body_size:
            truncated = body[: self.config.max_body_size].decode(
                "utf-8", errors="replace"
            )
            return f"{truncated}... (truncated, total size: {len(body)} bytes)"
        return body.decode("utf-8", errors="replace")

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    async def read_body(self, request: Request) -> bytes:
        """Read request body for logging."""
        try:
            return await request.body()
        except Exception:
            return b""

    def log_request(
        self, request: Request, request_id: str, body: bytes = None
    ) -> None:
        """Log incoming request."""
        if not self.config.enable_request_logging:
            return

        if not self.should_log_path(request.url.path):
            return

        log_data = {
            "request_id": request_id,
            "type": "request",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self.get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.config.log_headers:
            log_data["headers"] = self.sanitize_headers(dict(request.headers))

        if self.config.log_body and body:
            log_data["body"] = self.truncate_body(body)
            log_data["body_size"] = len(body)

        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.log(
            log_level, f"Request {request.method} {request.url.path}", extra=log_data
        )

    def log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        duration_ms: float,
        response_body: bytes = None,
    ) -> None:
        """Log outgoing response."""
        if not self.config.enable_response_logging:
            return

        if not self.should_log_path(request.url.path):
            return

        log_data = {
            "request_id": request_id,
            "type": "response",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.config.log_headers and hasattr(response, "headers"):
            log_data["response_headers"] = dict(response.headers)

        if self.config.log_body and response_body:
            log_data["response_body"] = self.truncate_body(response_body)
            log_data["response_size"] = len(response_body)

        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = logging.ERROR
        elif response.status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        self.logger.log(
            log_level,
            f"Response {response.status_code} for {request.method} {request.url.path} in {duration_ms:.2f}ms",
            extra=log_data,
        )

    def log_performance_metrics(self, metrics: RequestMetrics) -> None:
        """Log performance metrics."""
        if not self.config.include_performance_metrics:
            return

        # Store metrics for monitoring
        self.metrics_storage.append(metrics)
        # Keep only last 1000 metrics to prevent memory issues
        if len(self.metrics_storage) > 1000:
            self.metrics_storage.pop(0)

        # Log slow requests
        if (
            metrics.duration_ms
            and metrics.duration_ms / 1000 > self.config.slow_request_threshold
        ):
            self.performance_logger.warning(
                f"Slow request detected: {metrics.method} {metrics.path} "
                f"took {metrics.duration_ms:.2f}ms",
                extra=metrics.model_dump(),
            )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of recent request metrics."""
        if not self.metrics_storage:
            return {"message": "No metrics available"}

        recent_metrics = self.metrics_storage[-100:]  # Last 100 requests

        total_requests = len(recent_metrics)
        successful_requests = len(
            [m for m in recent_metrics if m.status_code and m.status_code < 400]
        )
        avg_duration = (
            sum(m.duration_ms for m in recent_metrics if m.duration_ms) / total_requests
        )

        status_codes = {}
        for metric in recent_metrics:
            if metric.status_code:
                status_codes[metric.status_code] = (
                    status_codes.get(metric.status_code, 0) + 1
                )

        slow_requests = len(
            [
                m
                for m in recent_metrics
                if m.duration_ms
                and m.duration_ms / 1000 > self.config.slow_request_threshold
            ]
        )

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": (
                successful_requests / total_requests * 100 if total_requests > 0 else 0
            ),
            "average_duration_ms": avg_duration,
            "slow_requests": slow_requests,
            "status_code_distribution": status_codes,
        }


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    def __init__(self, app, config: Optional[LoggingConfig] = None):
        """Initialize logging middleware."""
        super().__init__(app)
        self.config = config or LoggingConfig()
        self.request_logger = RequestLogger(self.config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging."""
        # Generate request ID for correlation
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        start_datetime = datetime.utcnow()

        # Initialize metrics
        metrics = RequestMetrics(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            start_time=start_datetime,
            client_ip=self.request_logger.get_client_ip(request),
            user_agent=request.headers.get("User-Agent", ""),
        )

        # Read request body for logging
        request_body = b""
        if self.config.log_body and request.method in ["POST", "PUT", "PATCH"]:
            request_body = await self.request_logger.read_body(request)
            metrics.request_size = len(request_body)

        # Log request
        self.request_logger.log_request(request, request_id, request_body)

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Update metrics
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = duration_ms
            metrics.status_code = response.status_code

            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            # Get response body for logging if needed
            response_body = b""
            if self.config.log_body and not isinstance(response, StreamingResponse):
                # For regular responses, we can access the body
                # Note: This is complex for FastAPI responses, simplified here
                pass

            # Log response
            self.request_logger.log_response(
                request, response, request_id, duration_ms, response_body
            )

            # Log performance metrics
            self.request_logger.log_performance_metrics(metrics)

            return response

        except Exception as e:
            # Calculate duration for failed requests
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Update metrics for error case
            metrics.end_time = datetime.utcnow()
            metrics.duration_ms = duration_ms
            metrics.error_message = str(e)

            # Log the error
            self.request_logger.logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {duration_ms:.2f}ms - {str(e)}",
                extra=metrics.model_dump(),
                exc_info=True,
            )

            # Log performance metrics even for failed requests
            self.request_logger.log_performance_metrics(metrics)

            raise


def setup_logging_middleware(
    app: FastAPI, config: Optional[LoggingConfig] = None
) -> None:
    """Setup request/response logging middleware.

    Args:
        app: FastAPI application instance
        config: Logging configuration
    """
    if config is None:
        config = LoggingConfig()

    app.add_middleware(LoggingMiddleware, config=config)

    # Configure logging format
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Request/response logging middleware enabled")


def get_logging_metrics() -> dict[str, Any]:
    """Get current logging metrics."""
    # This would be accessible from the middleware instance
    # For now, return a placeholder
    return {"message": "Metrics not available - middleware not initialized"}


# Utility function to configure structured logging
def configure_structured_logging():
    """Configure structured JSON logging for production."""
    import sys

    class JSONFormatter(logging.Formatter):
        """JSON formatter for structured logging."""

        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add extra fields if present
            if hasattr(record, "__dict__"):
                for key, value in record.__dict__.items():
                    if key not in [
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                    ]:
                        log_entry[key] = value

            return json.dumps(log_entry)

    # Configure root logger with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    logging.basicConfig(handlers=[handler], level=logging.INFO)
