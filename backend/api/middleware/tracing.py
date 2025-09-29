"""Request tracing and correlation ID middleware for the billiards trainer API."""

import asyncio
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Thread-local storage for correlation context
_correlation_context = threading.local()


@dataclass
class TraceSpan:
    """Represents a trace span for tracking operations."""

    span_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # active, completed, error

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def finish(self, status: str = "completed") -> None:
        """Finish the span."""
        self.end_time = time.time()
        self.status = status

    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span."""
        self.tags[key] = value

    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": level,
            **kwargs,
        }
        self.logs.append(log_entry)

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }


@dataclass
class TraceContext:
    """Represents a complete trace context."""

    trace_id: str
    request_id: str
    spans: dict[str, TraceSpan] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_span(
        self, operation_name: str, parent_span_id: Optional[str] = None
    ) -> TraceSpan:
        """Create a new span within this trace."""
        span_id = str(uuid.uuid4())
        span = TraceSpan(
            span_id=span_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
        )
        self.spans[span_id] = span
        return span

    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Get a span by ID."""
        return self.spans.get(span_id)

    def finish_trace(self) -> None:
        """Finish all active spans in the trace."""
        for span in self.spans.values():
            if span.status == "active":
                span.finish()

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "start_time": self.start_time,
            "total_duration_ms": (time.time() - self.start_time) * 1000,
            "spans": {span_id: span.to_dict() for span_id, span in self.spans.items()},
            "metadata": self.metadata,
        }


class TracingConfig(BaseModel):
    """Configuration for request tracing."""

    enable_tracing: bool = Field(default=True, description="Enable request tracing")
    enable_correlation_ids: bool = Field(
        default=True, description="Enable correlation ID generation"
    )
    trace_header_name: str = Field(
        default="X-Trace-ID", description="Header name for trace ID"
    )
    request_id_header_name: str = Field(
        default="X-Request-ID", description="Header name for request ID"
    )
    parent_span_header_name: str = Field(
        default="X-Parent-Span-ID", description="Header name for parent span ID"
    )
    max_trace_duration: int = Field(
        default=300, description="Maximum trace duration in seconds"
    )
    auto_cleanup_interval: int = Field(
        default=60, description="Auto cleanup interval in seconds"
    )
    include_request_headers: bool = Field(
        default=False, description="Include request headers in trace metadata"
    )
    include_response_headers: bool = Field(
        default=False, description="Include response headers in trace metadata"
    )
    excluded_paths: list[str] = Field(
        default_factory=lambda: ["/health", "/metrics", "/docs", "/redoc"],
        description="Paths to exclude from tracing",
    )
    sample_rate: float = Field(
        default=1.0, description="Trace sampling rate (0.0 to 1.0)", ge=0.0, le=1.0
    )


class TraceManager:
    """Manages trace contexts and spans."""

    def __init__(self, config: TracingConfig):
        """Initialize trace manager."""
        self.config = config
        self.active_traces: dict[str, TraceContext] = {}
        self.logger = logging.getLogger("api.tracing")
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start the cleanup task for old traces."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_traces())

    async def _cleanup_old_traces(self) -> None:
        """Cleanup old traces periodically."""
        while True:
            try:
                await asyncio.sleep(self.config.auto_cleanup_interval)
                self._remove_expired_traces()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in trace cleanup: {e}")

    def _remove_expired_traces(self) -> None:
        """Remove traces that have exceeded the maximum duration."""
        current_time = time.time()
        expired_traces = []

        for trace_id, trace_context in self.active_traces.items():
            if current_time - trace_context.start_time > self.config.max_trace_duration:
                expired_traces.append(trace_id)

        for trace_id in expired_traces:
            trace_context = self.active_traces.pop(trace_id, None)
            if trace_context:
                trace_context.finish_trace()
                self.logger.debug(f"Cleaned up expired trace: {trace_id}")

    def should_sample_trace(self) -> bool:
        """Determine if a trace should be sampled."""
        import random

        return random.random() < self.config.sample_rate

    def create_trace_context(
        self,
        request: Request,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> TraceContext:
        """Create a new trace context."""
        if not self.should_sample_trace():
            # Return a minimal trace context for non-sampled requests
            return TraceContext(
                trace_id=trace_id or str(uuid.uuid4()),
                request_id=request_id or str(uuid.uuid4()),
            )

        trace_id = trace_id or str(uuid.uuid4())
        request_id = request_id or str(uuid.uuid4())

        trace_context = TraceContext(trace_id=trace_id, request_id=request_id)

        # Add request metadata
        trace_context.metadata.update(
            {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("User-Agent", ""),
                "content_type": request.headers.get("Content-Type", ""),
            }
        )

        if self.config.include_request_headers:
            trace_context.metadata["request_headers"] = dict(request.headers)

        self.active_traces[trace_id] = trace_context
        return trace_context

    def get_trace_context(self, trace_id: str) -> Optional[TraceContext]:
        """Get an existing trace context."""
        return self.active_traces.get(trace_id)

    def finish_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Finish a trace and return its data."""
        trace_context = self.active_traces.pop(trace_id, None)
        if trace_context:
            trace_context.finish_trace()
            return trace_context.to_dict()
        return None

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def get_trace_statistics(self) -> dict[str, Any]:
        """Get trace statistics."""
        active_count = len(self.active_traces)
        total_spans = sum(len(trace.spans) for trace in self.active_traces.values())

        return {
            "active_traces": active_count,
            "total_spans": total_spans,
            "sampling_rate": self.config.sample_rate,
            "cleanup_interval": self.config.auto_cleanup_interval,
            "max_trace_duration": self.config.max_trace_duration,
        }


# Global trace manager
_trace_manager: Optional[TraceManager] = None


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from thread-local context."""
    return getattr(_correlation_context, "correlation_id", None)


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from thread-local context."""
    return getattr(_correlation_context, "trace_id", None)


def get_current_span_id() -> Optional[str]:
    """Get the current span ID from thread-local context."""
    return getattr(_correlation_context, "current_span_id", None)


def get_trace_context() -> Optional[TraceContext]:
    """Get the current trace context."""
    trace_id = get_trace_id()
    if trace_id and _trace_manager:
        return _trace_manager.get_trace_context(trace_id)
    return None


@contextmanager
def trace_operation(operation_name: str, **tags):
    """Context manager for tracing an operation."""
    trace_context = get_trace_context()
    if not trace_context:
        yield None
        return

    # Create a new span
    parent_span_id = get_current_span_id()
    span = trace_context.create_span(operation_name, parent_span_id)

    # Add tags
    for key, value in tags.items():
        span.add_tag(key, value)

    # Set as current span
    old_span_id = getattr(_correlation_context, "current_span_id", None)
    _correlation_context.current_span_id = span.span_id

    try:
        yield span
        span.finish("completed")
    except Exception as e:
        span.add_tag("error", True)
        span.add_tag("error_message", str(e))
        span.add_log(f"Operation failed: {str(e)}", level="error")
        span.finish("error")
        raise
    finally:
        # Restore previous span
        _correlation_context.current_span_id = old_span_id


def add_trace_log(message: str, level: str = "info", **kwargs) -> None:
    """Add a log entry to the current span."""
    span_id = get_current_span_id()
    if span_id:
        trace_context = get_trace_context()
        if trace_context:
            span = trace_context.get_span(span_id)
            if span:
                span.add_log(message, level, **kwargs)


def add_trace_tag(key: str, value: Any) -> None:
    """Add a tag to the current span."""
    span_id = get_current_span_id()
    if span_id:
        trace_context = get_trace_context()
        if trace_context:
            span = trace_context.get_span(span_id)
            if span:
                span.add_tag(key, value)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and correlation IDs."""

    def __init__(self, app, config: Optional[TracingConfig] = None):
        """Initialize tracing middleware."""
        super().__init__(app)
        self.config = config or TracingConfig()
        global _trace_manager
        _trace_manager = TraceManager(self.config)
        self.logger = logging.getLogger("api.tracing")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        # Check if path should be traced
        if request.url.path in self.config.excluded_paths:
            return await call_next(request)

        # Extract or generate trace and request IDs
        trace_id = request.headers.get(self.config.trace_header_name)
        request_id = request.headers.get(self.config.request_id_header_name)
        parent_span_id = request.headers.get(self.config.parent_span_header_name)

        # Create trace context
        trace_context = _trace_manager.create_trace_context(
            request, trace_id, request_id
        )

        # Set correlation context
        _correlation_context.correlation_id = trace_context.request_id
        _correlation_context.trace_id = trace_context.trace_id

        # Create root span for the request
        root_span = trace_context.create_span(
            f"{request.method} {request.url.path}", parent_span_id
        )
        root_span.add_tag("http.method", request.method)
        root_span.add_tag("http.url", str(request.url))
        root_span.add_tag("http.scheme", request.url.scheme)
        root_span.add_tag("http.host", request.url.hostname)

        _correlation_context.current_span_id = root_span.span_id

        # Store IDs in request state for access by other middleware/routes
        request.state.trace_id = trace_context.trace_id
        request.state.request_id = trace_context.request_id
        request.state.span_id = root_span.span_id

        try:
            # Process request
            response = await call_next(request)

            # Add response information to span
            root_span.add_tag("http.status_code", response.status_code)
            root_span.add_tag(
                "http.response_size",
                len(response.body) if hasattr(response, "body") else 0,
            )

            # Add response headers to trace if configured
            if self.config.include_response_headers:
                trace_context.metadata["response_headers"] = dict(response.headers)

            # Add tracing headers to response
            response.headers[self.config.trace_header_name] = trace_context.trace_id
            response.headers[self.config.request_id_header_name] = (
                trace_context.request_id
            )

            # Finish the root span
            status = "completed" if response.status_code < 400 else "error"
            root_span.finish(status)

            # Log trace completion
            self.logger.debug(
                f"Trace completed: {trace_context.trace_id} "
                f"({root_span.duration_ms:.2f}ms)"
            )

            return response

        except Exception as e:
            # Handle errors
            root_span.add_tag("error", True)
            root_span.add_tag("error_type", type(e).__name__)
            root_span.add_tag("error_message", str(e))
            root_span.add_log(f"Request failed: {str(e)}", level="error")
            root_span.finish("error")

            self.logger.error(f"Trace failed: {trace_context.trace_id} - {str(e)}")
            raise

        finally:
            # Clean up correlation context
            _correlation_context.correlation_id = None
            _correlation_context.trace_id = None
            _correlation_context.current_span_id = None


def setup_tracing_middleware(
    app: FastAPI, config: Optional[TracingConfig] = None
) -> None:
    """Setup tracing middleware.

    Args:
        app: FastAPI application instance
        config: Tracing configuration
    """
    if config is None:
        config = TracingConfig()

    app.add_middleware(TracingMiddleware, config=config)

    logger = logging.getLogger(__name__)
    logger.info("Request tracing and correlation ID middleware enabled")


def get_trace_manager() -> Optional[TraceManager]:
    """Get the global trace manager instance."""
    return _trace_manager


# Helper function to extract correlation info from request
def extract_correlation_info(request: Request) -> dict[str, Optional[str]]:
    """Extract correlation information from request."""
    return {
        "trace_id": getattr(request.state, "trace_id", None),
        "request_id": getattr(request.state, "request_id", None),
        "span_id": getattr(request.state, "span_id", None),
    }


# Decorator for automatic function tracing
def traced(operation_name: Optional[str] = None):
    """Decorator to automatically trace function calls."""

    def decorator(func):
        actual_operation_name = operation_name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with trace_operation(actual_operation_name) as span:
                    if span:
                        span.add_tag("function.name", func.__name__)
                        span.add_tag("function.module", func.__module__)
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with trace_operation(actual_operation_name) as span:
                    if span:
                        span.add_tag("function.name", func.__name__)
                        span.add_tag("function.module", func.__module__)
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
