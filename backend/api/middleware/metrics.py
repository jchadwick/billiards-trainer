"""Middleware for tracking API request metrics and performance."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Import health monitor with fallback
try:
    from backend.system.health_monitor import health_monitor
except ImportError:
    try:
        from system.health_monitor import health_monitor
    except ImportError:
        # Fallback: create a minimal health monitor interface
        class MockHealthMonitor:
            def record_api_request(self, response_time: float, success: bool = True):
                pass

        health_monitor = MockHealthMonitor()

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track API request metrics for health monitoring."""

    def __init__(self, app, exclude_paths: list[str] = None):
        """Initialize metrics middleware.

        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from metrics (e.g., health checks)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health/live",
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics.

        Args:
            request: HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response
        """
        # Skip metrics for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Record request start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time

            # Determine if request was successful
            success = 200 <= response.status_code < 400

            # Record metrics
            health_monitor.record_api_request(response_time, success)

            # Add response headers with timing information
            response.headers["X-Response-Time"] = f"{response_time:.4f}"

            # Log slow requests
            if response_time > 1.0:  # Log requests taking more than 1 second
                logger.warning(
                    f"Slow API request: {request.method} {request.url.path} "
                    f"took {response_time:.4f}s (status: {response.status_code})"
                )

            return response

        except Exception as e:
            # Calculate response time even for errors
            response_time = time.time() - start_time

            # Record failed request
            health_monitor.record_api_request(response_time, success=False)

            # Log error
            logger.error(
                f"API request failed: {request.method} {request.url.path} "
                f"after {response_time:.4f}s - {str(e)}"
            )

            # Re-raise the exception
            raise
