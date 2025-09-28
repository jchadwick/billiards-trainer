"""Rate limiting middleware for the billiards trainer API."""

import logging
import time
from collections import defaultdict, deque
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration model."""

    requests_per_minute: int = Field(
        default=60, description="Maximum number of requests per minute per client"
    )
    requests_per_hour: int = Field(
        default=1000, description="Maximum number of requests per hour per client"
    )
    burst_size: int = Field(
        default=10, description="Maximum number of requests allowed in a burst"
    )
    enable_per_endpoint_limits: bool = Field(
        default=True, description="Whether to enable per-endpoint rate limiting"
    )
    cleanup_interval: int = Field(
        default=300, description="Interval in seconds to cleanup old rate limit data"
    )
    # Per-endpoint rate limits (endpoint_pattern: requests_per_minute)
    endpoint_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "/api/v1/stream/video": 1000,  # High limit for video streaming
            "/api/v1/game/state": 120,  # Higher limit for real-time game state
            "/api/v1/config": 30,  # Lower limit for configuration changes
            "/api/v1/calibration": 10,  # Low limit for calibration operations
        }
    )


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens in the bucket
            refill_rate: Number of tokens added per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        now = time.time()
        # Add tokens based on time elapsed
        tokens_to_add = (now - self.last_refill) * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_token(self) -> float:
        """Get time in seconds until next token is available."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for time-based rate limiting."""

    def __init__(self, window_size: int):
        """Initialize sliding window counter.

        Args:
            window_size: Window size in seconds
        """
        self.window_size = window_size
        self.requests = deque()

    def add_request(self, timestamp: float = None) -> None:
        """Add a request timestamp."""
        if timestamp is None:
            timestamp = time.time()
        self.requests.append(timestamp)
        self._cleanup_old_requests()

    def get_request_count(self) -> int:
        """Get number of requests in the current window."""
        self._cleanup_old_requests()
        return len(self.requests)

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the current window."""
        now = time.time()
        while self.requests and (now - self.requests[0]) > self.window_size:
            self.requests.popleft()


class RateLimiter:
    """Rate limiter using token bucket and sliding window algorithms."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        # Client rate limit storage: client_ip -> (token_bucket, sliding_windows)
        self.client_buckets: dict[str, TokenBucket] = {}
        self.client_windows: dict[str, dict[str, SlidingWindowCounter]] = defaultdict(
            lambda: {
                "minute": SlidingWindowCounter(60),
                "hour": SlidingWindowCounter(3600),
            }
        )
        # Per-endpoint rate limits: (client_ip, endpoint) -> sliding_window
        self.endpoint_windows: dict[tuple[str, str], SlidingWindowCounter] = {}
        self.last_cleanup = time.time()

    def is_allowed(self, client_ip: str, endpoint: str = None) -> tuple[bool, dict]:
        """Check if request is allowed based on rate limits.

        Args:
            client_ip: Client IP address
            endpoint: API endpoint path

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        now = time.time()
        self._cleanup_if_needed()

        # Initialize token bucket for client if not exists
        if client_ip not in self.client_buckets:
            self.client_buckets[client_ip] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=self.config.requests_per_minute / 60.0,
            )

        # Check token bucket (burst protection)
        bucket = self.client_buckets[client_ip]
        if not bucket.consume():
            return False, {
                "limit_type": "burst",
                "limit": self.config.burst_size,
                "reset_time": now + bucket.time_until_token(),
            }

        # Check sliding window limits
        windows = self.client_windows[client_ip]

        # Check minute limit
        minute_count = windows["minute"].get_request_count()
        if minute_count >= self.config.requests_per_minute:
            return False, {
                "limit_type": "minute",
                "limit": self.config.requests_per_minute,
                "current": minute_count,
                "reset_time": now + (60 - (now % 60)),
            }

        # Check hour limit
        hour_count = windows["hour"].get_request_count()
        if hour_count >= self.config.requests_per_hour:
            return False, {
                "limit_type": "hour",
                "limit": self.config.requests_per_hour,
                "current": hour_count,
                "reset_time": now + (3600 - (now % 3600)),
            }

        # Check per-endpoint limits
        if endpoint and self.config.enable_per_endpoint_limits:
            endpoint_limit = self._get_endpoint_limit(endpoint)
            if endpoint_limit:
                endpoint_key = (client_ip, endpoint)
                if endpoint_key not in self.endpoint_windows:
                    self.endpoint_windows[endpoint_key] = SlidingWindowCounter(60)

                endpoint_window = self.endpoint_windows[endpoint_key]
                endpoint_count = endpoint_window.get_request_count()
                if endpoint_count >= endpoint_limit:
                    return False, {
                        "limit_type": "endpoint",
                        "endpoint": endpoint,
                        "limit": endpoint_limit,
                        "current": endpoint_count,
                        "reset_time": now + (60 - (now % 60)),
                    }

        # Request is allowed, record it
        windows["minute"].add_request(now)
        windows["hour"].add_request(now)
        if endpoint and self.config.enable_per_endpoint_limits:
            endpoint_key = (client_ip, endpoint)
            if endpoint_key in self.endpoint_windows:
                self.endpoint_windows[endpoint_key].add_request(now)

        return True, {
            "limit_type": "allowed",
            "minute_remaining": self.config.requests_per_minute - minute_count - 1,
            "hour_remaining": self.config.requests_per_hour - hour_count - 1,
        }

    def _get_endpoint_limit(self, endpoint: str) -> Optional[int]:
        """Get rate limit for specific endpoint."""
        for pattern, limit in self.config.endpoint_limits.items():
            if pattern in endpoint:
                return limit
        return None

    def _cleanup_if_needed(self) -> None:
        """Cleanup old rate limit data if needed."""
        now = time.time()
        if now - self.last_cleanup > self.config.cleanup_interval:
            self._cleanup_old_data()
            self.last_cleanup = now

    def _cleanup_old_data(self) -> None:
        """Remove old rate limit data to prevent memory leaks."""
        now = time.time()

        # Remove old client data (older than 1 hour)
        clients_to_remove = []
        for client_ip, windows in self.client_windows.items():
            if windows["hour"].get_request_count() == 0 and (
                not windows["hour"].requests
                or now - windows["hour"].requests[-1] > 3600
            ):
                clients_to_remove.append(client_ip)

        for client_ip in clients_to_remove:
            self.client_windows.pop(client_ip, None)
            self.client_buckets.pop(client_ip, None)

        # Remove old endpoint data
        endpoints_to_remove = []
        for endpoint_key, window in self.endpoint_windows.items():
            if window.get_request_count() == 0 and (
                not window.requests or now - window.requests[-1] > 3600
            ):
                endpoints_to_remove.append(endpoint_key)

        for endpoint_key in endpoints_to_remove:
            self.endpoint_windows.pop(endpoint_key, None)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


class RateLimitMiddleware:
    """Rate limiting middleware class for FastAPI."""

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiting middleware."""
        self.app = app
        if config is None:
            config = RateLimitConfig()

        global _rate_limiter
        _rate_limiter = RateLimiter(config)
        logger.info(
            f"Rate limiting middleware initialized: {config.requests_per_minute}/min, {config.requests_per_hour}/hour"
        )

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting."""
        return await rate_limit_middleware(request, call_next)


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address
    """
    # Check for forwarded headers (in case of proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    forwarded_ip = request.headers.get("X-Real-IP")
    if forwarded_ip:
        return forwarded_ip

    # Fall back to direct connection IP
    return request.client.host


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """Rate limiting middleware function.

    Args:
        request: FastAPI request object
        call_next: Next middleware in chain

    Returns:
        Response object
    """
    global _rate_limiter

    if _rate_limiter is None:
        # Rate limiting not configured, skip
        return await call_next(request)

    client_ip = get_client_ip(request)
    endpoint = request.url.path

    # Check rate limits
    is_allowed, limit_info = _rate_limiter.is_allowed(client_ip, endpoint)

    if not is_allowed:
        # Rate limit exceeded
        logger.warning(
            f"Rate limit exceeded for {client_ip} on {endpoint}: {limit_info}"
        )

        headers = {
            "X-RateLimit-Limit": str(limit_info.get("limit", "")),
            "X-RateLimit-Remaining": "0",
            "Retry-After": str(int(limit_info.get("reset_time", 0) - time.time())),
        }

        return JSONResponse(
            status_code=429,
            content={
                "error": "RATE_001",
                "message": "Rate limit exceeded",
                "details": {
                    "limit_type": limit_info.get("limit_type"),
                    "retry_after": int(limit_info.get("reset_time", 0) - time.time()),
                },
            },
            headers=headers,
        )

    # Add rate limit headers to response
    response = await call_next(request)
    if limit_info.get("limit_type") == "allowed":
        response.headers["X-RateLimit-Limit-Minute"] = str(
            _rate_limiter.config.requests_per_minute
        )
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            limit_info.get("minute_remaining", 0)
        )
        response.headers["X-RateLimit-Limit-Hour"] = str(
            _rate_limiter.config.requests_per_hour
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            limit_info.get("hour_remaining", 0)
        )

    return response


def setup_rate_limiting(app: FastAPI, config: Optional[RateLimitConfig] = None) -> None:
    """Setup rate limiting middleware.

    Args:
        app: FastAPI application instance
        config: Rate limiting configuration
    """
    global _rate_limiter

    if config is None:
        config = RateLimitConfig()

    _rate_limiter = RateLimiter(config)
    app.middleware("http")(rate_limit_middleware)

    logger.info(
        f"Rate limiting enabled: {config.requests_per_minute}/min, {config.requests_per_hour}/hour"
    )


def get_rate_limit_status(client_ip: str) -> dict:
    """Get current rate limit status for a client.

    Args:
        client_ip: Client IP address

    Returns:
        Dictionary with rate limit status
    """
    global _rate_limiter

    if _rate_limiter is None:
        return {"error": "Rate limiting not configured"}

    windows = _rate_limiter.client_windows.get(client_ip)
    if not windows:
        return {
            "minute_used": 0,
            "minute_remaining": _rate_limiter.config.requests_per_minute,
            "hour_used": 0,
            "hour_remaining": _rate_limiter.config.requests_per_hour,
        }

    minute_used = windows["minute"].get_request_count()
    hour_used = windows["hour"].get_request_count()

    return {
        "minute_used": minute_used,
        "minute_remaining": _rate_limiter.config.requests_per_minute - minute_used,
        "hour_used": hour_used,
        "hour_remaining": _rate_limiter.config.requests_per_hour - hour_used,
    }
