"""Behavioral contracts for request tracing and correlation propagation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from backend.api.middleware import tracing


@pytest.fixture(autouse=True)
def restore_tracing_globals():
    """Keep the module-level manager and thread-local context isolated per test."""
    previous_manager = tracing._trace_manager
    previous_context = {
        name: getattr(tracing._correlation_context, name, None)
        for name in ("correlation_id", "trace_id", "current_span_id")
    }
    yield
    tracing._trace_manager = previous_manager
    for name, value in previous_context.items():
        setattr(tracing._correlation_context, name, value)


def make_request(
    path: str = "/api/balls",
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    query_string: bytes = b"mode=live",
) -> Request:
    """Build a deterministic Starlette request without starting an ASGI server."""
    encoded_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string,
        "headers": encoded_headers,
        "server": ("table.local", 8000),
        "client": ("192.0.2.10", 4321),
    }
    return Request(scope)


async def close_manager(manager: tracing.TraceManager) -> None:
    """Cancel the manager's periodic task so tests leave no pending task behind."""
    task = manager._cleanup_task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize("sample_rate", [-0.01, 1.01])
def test_tracing_config_rejects_sample_rates_outside_unit_interval(sample_rate):
    with pytest.raises(ValueError):
        tracing.TracingConfig(sample_rate=sample_rate)


def test_trace_span_records_duration_tags_logs_and_serialized_status(monkeypatch):
    clock = SimpleNamespace(value=100.0)
    monkeypatch.setattr(tracing, "time", SimpleNamespace(time=lambda: clock.value))
    span = tracing.TraceSpan(
        span_id="span-1", operation_name="detector.infer", start_time=99.5
    )
    assert span.duration_ms is None

    span.add_tag("model", "balls-v1")
    span.add_log("inference started", level="debug", frame_id=17)
    clock.value = 100.25
    span.finish("completed")

    result = span.to_dict()
    assert result["duration_ms"] == pytest.approx(750.0)
    assert result["status"] == "completed"
    assert result["tags"] == {"model": "balls-v1"}
    assert result["logs"] == [
        {
            "timestamp": 100.0,
            "message": "inference started",
            "level": "debug",
            "frame_id": 17,
        }
    ]


def test_trace_context_finishes_only_active_spans(monkeypatch):
    now = SimpleNamespace(value=50.0)
    monkeypatch.setattr(tracing, "time", SimpleNamespace(time=lambda: now.value))
    context = tracing.TraceContext("trace-a", "request-a", start_time=49.0)
    completed = context.create_span("already-done")
    completed.finish("completed")
    active = context.create_span("still-running")
    now.value = 52.0

    context.finish_trace()
    serialized = context.to_dict()

    assert active.status == "completed"
    assert active.duration_ms == pytest.approx(2000.0)
    assert completed.end_time == 50.0
    assert serialized["trace_id"] == "trace-a"
    assert serialized["request_id"] == "request-a"
    assert set(serialized["spans"]) == {active.span_id, completed.span_id}


@pytest.mark.asyncio
async def test_trace_manager_sampling_metadata_forwarding_and_statistics():
    config = tracing.TracingConfig(
        include_request_headers=True,
        sample_rate=1.0,
        auto_cleanup_interval=3600,
    )
    manager = tracing.TraceManager(config)
    try:
        request = make_request(
            headers={
                "User-Agent": "audit-client/1",
                "Content-Type": "application/json",
                "X-Forwarded-For": "198.51.100.7, 192.0.2.1",
                "X-Session": "session-value",
            }
        )
        context = manager.create_trace_context(
            request, "trace-explicit", "req-explicit"
        )
        context.create_span("root")

        assert context.metadata["method"] == "GET"
        assert context.metadata["path"] == "/api/balls"
        assert context.metadata["query_params"] == {"mode": "live"}
        assert context.metadata["client_ip"] == "198.51.100.7"
        assert context.metadata["user_agent"] == "audit-client/1"
        assert context.metadata["content_type"] == "application/json"
        assert context.metadata["request_headers"]["x-session"] == "session-value"
        assert manager.get_trace_context("trace-explicit") is context
        assert manager.get_trace_statistics() == {
            "active_traces": 1,
            "total_spans": 1,
            "sampling_rate": 1.0,
            "cleanup_interval": 3600,
            "max_trace_duration": 300,
        }

        manager.should_sample_trace = lambda: False
        unsampled = manager.create_trace_context(request, "unsampled", "request-u")
        assert unsampled.metadata == {}
        assert "unsampled" not in manager.active_traces
        assert manager.finish_trace("missing") is None
        finished = manager.finish_trace("trace-explicit")
        assert finished is not None
        assert finished["trace_id"] == "trace-explicit"
        assert manager.finish_trace("trace-explicit") is None
    finally:
        await close_manager(manager)


@pytest.mark.asyncio
async def test_trace_manager_expiry_finishes_spans_and_removes_expired_contexts(
    monkeypatch,
):
    now = SimpleNamespace(value=100.0)
    monkeypatch.setattr(tracing, "time", SimpleNamespace(time=lambda: now.value))
    manager = tracing.TraceManager(
        tracing.TracingConfig(max_trace_duration=10, auto_cleanup_interval=3600)
    )
    try:
        expired = tracing.TraceContext("old", "req-old", start_time=89.0)
        expired_span = expired.create_span("active")
        current = tracing.TraceContext("new", "req-new", start_time=95.0)
        manager.active_traces.update({"old": expired, "new": current})

        manager._remove_expired_traces()

        assert set(manager.active_traces) == {"new"}
        assert expired_span.status == "completed"
        assert expired_span.end_time == 100.0
    finally:
        await close_manager(manager)


@pytest.mark.asyncio
async def test_cleanup_loop_logs_maintenance_failure_and_continues(monkeypatch, caplog):
    second_sleep = asyncio.Event()
    real_sleep = asyncio.sleep
    calls = 0

    async def flaky_sleep(_delay: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated cleanup timer failure")
        second_sleep.set()
        await real_sleep(3600)

    # Isolate the test clock namespace so replacing sleep does not affect pytest.
    monkeypatch.setattr(
        tracing,
        "asyncio",
        SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            create_task=asyncio.create_task,
            sleep=flaky_sleep,
        ),
    )
    manager = tracing.TraceManager(tracing.TracingConfig(auto_cleanup_interval=17))
    try:
        await asyncio.wait_for(second_sleep.wait(), timeout=1)
        assert calls == 2
        assert any(
            "Error in trace cleanup: simulated cleanup timer failure" in record.message
            for record in caplog.records
        )
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()
    finally:
        task = manager._cleanup_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_cleanup_loop_removes_expired_contexts_after_interval(monkeypatch):
    next_interval = asyncio.Event()
    real_sleep = asyncio.sleep
    calls = 0
    now = SimpleNamespace(value=100.0)
    monkeypatch.setattr(tracing, "time", SimpleNamespace(time=lambda: now.value))

    async def return_once_then_wait(_delay: float) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            next_interval.set()
            await real_sleep(3600)

    monkeypatch.setattr(
        tracing,
        "asyncio",
        SimpleNamespace(
            CancelledError=asyncio.CancelledError,
            create_task=asyncio.create_task,
            sleep=return_once_then_wait,
        ),
    )
    manager = tracing.TraceManager(
        tracing.TracingConfig(max_trace_duration=10, auto_cleanup_interval=2)
    )
    expired = tracing.TraceContext("expired-by-loop", "request-expired", start_time=89)
    manager.active_traces[expired.trace_id] = expired
    try:
        await asyncio.wait_for(next_interval.wait(), timeout=1)
        assert expired.trace_id not in manager.active_traces
    finally:
        task = manager._cleanup_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_cleanup_task_restarts_after_previous_task_finishes():
    manager = tracing.TraceManager(tracing.TracingConfig(auto_cleanup_interval=3600))
    try:
        previous_task = manager._cleanup_task
        assert previous_task is not None
        manager._start_cleanup_task()
        assert manager._cleanup_task is previous_task
        previous_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await previous_task

        manager._start_cleanup_task()

        assert manager._cleanup_task is not None
        assert manager._cleanup_task is not previous_task
        assert not manager._cleanup_task.done()
    finally:
        await close_manager(manager)


def test_nested_trace_operations_propagate_parent_and_restore_context(monkeypatch):
    context = tracing.TraceContext("trace-nested", "request-nested")
    manager = SimpleNamespace(get_trace_context=lambda trace_id: context)
    monkeypatch.setattr(tracing, "_trace_manager", manager)
    tracing._correlation_context.trace_id = context.trace_id
    tracing._correlation_context.current_span_id = None

    with tracing.trace_operation("outer", frame_id=8) as outer:
        assert outer is not None
        with tracing.trace_operation("inner", detector="ball") as inner:
            assert inner is not None
            tracing.add_trace_tag("confidence", 0.91)
            tracing.add_trace_log("candidate accepted", candidate_count=4)
            assert tracing.get_current_span_id() == inner.span_id
        assert inner.parent_span_id == outer.span_id
        assert inner.tags["confidence"] == 0.91
        assert inner.logs[0]["candidate_count"] == 4
        assert tracing.get_current_span_id() == outer.span_id

    assert outer.status == "completed"
    assert tracing.get_current_span_id() is None


def test_trace_operation_marks_errors_and_restores_parent_context(monkeypatch):
    context = tracing.TraceContext("trace-error", "request-error")
    manager = SimpleNamespace(get_trace_context=lambda _trace_id: context)
    monkeypatch.setattr(tracing, "_trace_manager", manager)
    tracing._correlation_context.trace_id = context.trace_id
    tracing._correlation_context.current_span_id = "upstream-span"

    with pytest.raises(ValueError, match="detector unavailable"):
        with tracing.trace_operation("detector.infer") as span:
            assert span is not None
            raise ValueError("detector unavailable")

    assert span is not None
    assert span.parent_span_id == "upstream-span"
    assert span.status == "error"
    assert span.tags["error"] is True
    assert span.tags["error_message"] == "detector unavailable"
    assert span.logs[0]["level"] == "error"
    assert tracing.get_current_span_id() == "upstream-span"


def test_trace_log_and_tag_helpers_are_safe_without_a_current_span(monkeypatch):
    context = tracing.TraceContext("trace-no-span", "request-no-span")
    monkeypatch.setattr(
        tracing,
        "_trace_manager",
        SimpleNamespace(
            get_trace_context=lambda trace_id: (
                context if trace_id == context.trace_id else None
            )
        ),
    )
    tracing._correlation_context.trace_id = context.trace_id
    tracing._correlation_context.current_span_id = None

    tracing.add_trace_log("ignored without span")
    tracing.add_trace_tag("ignored", True)

    tracing._correlation_context.current_span_id = "unknown-span"
    tracing.add_trace_log("ignored without matching context span")
    tracing.add_trace_tag("ignored", True)

    tracing._correlation_context.trace_id = "missing-context"
    tracing._correlation_context.current_span_id = "unknown-span"
    tracing.add_trace_log("ignored without trace context")
    tracing.add_trace_tag("ignored", True)
    assert context.spans == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_status"), [(200, "completed"), (503, "error")]
)
async def test_middleware_propagates_ids_parent_span_and_response_metadata(
    status_code, expected_status
):
    config = tracing.TracingConfig(
        include_response_headers=True,
        auto_cleanup_interval=3600,
    )
    middleware = tracing.TracingMiddleware(FastAPI(), config)
    request = make_request(
        method="POST",
        headers={
            "X-Trace-ID": "caller-trace",
            "X-Request-ID": "caller-request",
            "X-Parent-Span-ID": "caller-parent",
        },
    )
    response = Response(
        content=b"state payload",
        status_code=status_code,
        headers={"x-response-kind": "full-state"},
    )

    async def call_next(received_request):
        assert received_request is request
        assert tracing.get_correlation_id() == "caller-request"
        assert tracing.get_trace_id() == "caller-trace"
        assert tracing.get_current_span_id() == request.state.span_id
        return response

    try:
        result = await middleware.dispatch(request, call_next)
        assert result.headers["X-Trace-ID"] == "caller-trace"
        assert result.headers["X-Request-ID"] == "caller-request"
        assert tracing.extract_correlation_info(request) == {
            "trace_id": "caller-trace",
            "request_id": "caller-request",
            "span_id": request.state.span_id,
        }
        assert tracing.get_correlation_id() is None
        assert tracing.get_trace_id() is None
        assert tracing.get_current_span_id() is None

        context = middleware.trace_manager.get_trace_context("caller-trace")
        assert context is not None
        root = context.get_span(request.state.span_id)
        assert root is not None
        assert root.parent_span_id == "caller-parent"
        assert root.operation_name == "POST /api/balls"
        assert root.tags["http.status_code"] == status_code
        assert root.tags["http.response_size"] == len(b"state payload")
        assert root.status == expected_status
        assert context.metadata["response_headers"]["x-response-kind"] == "full-state"
    finally:
        await close_manager(middleware.trace_manager)


@pytest.mark.asyncio
async def test_excluded_path_bypasses_trace_creation_and_id_injection():
    middleware = tracing.TracingMiddleware(
        FastAPI(), tracing.TracingConfig(auto_cleanup_interval=3600)
    )
    request = make_request("/health")
    response = Response(content=b"ok")
    called = False

    async def call_next(_request):
        nonlocal called
        called = True
        return response

    try:
        result = await middleware.dispatch(request, call_next)
        assert called
        assert result is response
        assert "x-trace-id" not in result.headers
        assert "x-request-id" not in result.headers
        assert middleware.trace_manager.active_traces == {}
        assert not hasattr(request.state, "trace_id")
    finally:
        await close_manager(middleware.trace_manager)


@pytest.mark.asyncio
async def test_middleware_handles_unsampled_requests_without_retaining_trace():
    middleware = tracing.TracingMiddleware(
        FastAPI(),
        tracing.TracingConfig(sample_rate=0, auto_cleanup_interval=3600),
    )
    request = make_request(headers={"X-Request-ID": "client-request"})

    async def call_next(_request):
        return Response(content=b"latest")

    try:
        response = await middleware.dispatch(request, call_next)
        assert response.headers["x-request-id"] == "client-request"
        assert response.headers["x-trace-id"] == request.state.trace_id
        assert middleware.trace_manager.active_traces == {}
        assert request.state.request_id == "client-request"
    finally:
        await close_manager(middleware.trace_manager)


@pytest.mark.asyncio
async def test_middleware_records_exception_and_clears_correlation_context(caplog):
    middleware = tracing.TracingMiddleware(
        FastAPI(), tracing.TracingConfig(auto_cleanup_interval=3600)
    )
    request = make_request(headers={"X-Trace-ID": "failed-trace"})

    async def call_next(_request):
        raise RuntimeError("camera stream interrupted")

    try:
        with pytest.raises(RuntimeError, match="camera stream interrupted"):
            await middleware.dispatch(request, call_next)

        context = middleware.trace_manager.get_trace_context("failed-trace")
        assert context is not None
        root = context.get_span(request.state.span_id)
        assert root is not None
        assert root.status == "error"
        assert root.tags["error_type"] == "RuntimeError"
        assert root.tags["error_message"] == "camera stream interrupted"
        assert root.logs[0]["level"] == "error"
        assert tracing.get_correlation_id() is None
        assert tracing.get_trace_id() is None
        assert tracing.get_current_span_id() is None
        assert "Trace failed: failed-trace" in caplog.text
    finally:
        await close_manager(middleware.trace_manager)


@pytest.mark.asyncio
async def test_middleware_prefers_request_state_id_and_supports_custom_headers():
    config = tracing.TracingConfig(
        trace_header_name="Trace-Key",
        request_id_header_name="Request-Key",
        parent_span_header_name="Parent-Key",
        auto_cleanup_interval=3600,
    )
    middleware = tracing.TracingMiddleware(FastAPI(), config)
    request = make_request(
        headers={
            "Trace-Key": "trace-from-header",
            "Request-Key": "request-from-header",
            "Parent-Key": "parent-from-header",
        }
    )
    request.state.request_id = "request-from-state"

    async def call_next(_request):
        return Response(content=b"state")

    try:
        response = await middleware.dispatch(request, call_next)
        assert response.headers["Trace-Key"] == "trace-from-header"
        assert response.headers["Request-Key"] == "request-from-state"
        context = middleware.trace_manager.get_trace_context("trace-from-header")
        assert context is not None
        assert context.request_id == "request-from-state"
        assert (
            context.get_span(request.state.span_id).parent_span_id
            == "parent-from-header"
        )
    finally:
        await close_manager(middleware.trace_manager)


def test_traced_decorator_handles_sync_calls_and_is_noop_without_context():
    @tracing.traced("sample.sync")
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5


def test_sync_traced_decorator_records_function_metadata(monkeypatch):
    context = tracing.TraceContext("trace-sync", "request-sync")
    monkeypatch.setattr(
        tracing,
        "_trace_manager",
        SimpleNamespace(get_trace_context=lambda _trace_id: context),
    )
    tracing._correlation_context.trace_id = context.trace_id

    @tracing.traced()
    def multiply(left: int, right: int) -> int:
        return left * right

    assert multiply(4, 5) == 20
    span = next(iter(context.spans.values()))
    assert span.operation_name == f"{__name__}.multiply"
    assert span.tags["function.name"] == "multiply"
    assert span.tags["function.module"] == __name__


@pytest.mark.asyncio
async def test_traced_decorator_records_async_function_metadata():
    context = tracing.TraceContext("trace-async", "request-async")
    monkeypatch_manager = SimpleNamespace(get_trace_context=lambda _trace_id: context)
    tracing._trace_manager = monkeypatch_manager
    tracing._correlation_context.trace_id = context.trace_id

    @tracing.traced("sample.async")
    async def inspect(value: int) -> int:
        return value * 2

    assert await inspect(6) == 12
    spans = list(context.spans.values())
    assert len(spans) == 1
    assert spans[0].operation_name == "sample.async"
    assert spans[0].tags["function.name"] == "inspect"
    assert spans[0].tags["function.module"] == __name__
    assert spans[0].status == "completed"


@pytest.mark.asyncio
async def test_async_traced_decorator_runs_without_trace_context():
    tracing._correlation_context.trace_id = None

    @tracing.traced("sample.async.noop")
    async def inspect(value: int) -> int:
        return value + 1

    assert await inspect(9) == 10


@pytest.mark.asyncio
async def test_custom_client_ip_fallback_and_non_sampled_trace_context():
    manager = tracing.TraceManager(
        tracing.TracingConfig(sample_rate=0, auto_cleanup_interval=3600)
    )
    try:
        no_client_request = make_request()
        no_client_request.scope["client"] = None
        assert manager._get_client_ip(no_client_request) == "unknown"

        manager.should_sample_trace = lambda: False
        minimal = manager.create_trace_context(no_client_request)
        assert minimal.trace_id
        assert minimal.request_id
        assert minimal.spans == {}
        assert minimal.metadata == {}
    finally:
        await close_manager(manager)


def test_setup_function_registers_explicit_or_default_configuration(caplog):
    caplog.set_level(logging.INFO)
    manager_before = tracing.get_trace_manager()
    explicit_app = FastAPI()
    explicit_config = tracing.TracingConfig(excluded_paths=["/internal"])
    tracing.setup_tracing_middleware(explicit_app, explicit_config)

    default_app = FastAPI()
    tracing.setup_tracing_middleware(default_app)

    explicit_middleware = explicit_app.user_middleware[0]
    default_middleware = default_app.user_middleware[0]
    assert explicit_middleware.cls is tracing.TracingMiddleware
    assert explicit_middleware.kwargs["config"] is explicit_config
    assert default_middleware.cls is tracing.TracingMiddleware
    assert isinstance(default_middleware.kwargs["config"], tracing.TracingConfig)
    assert "Request tracing and correlation ID middleware enabled" in caplog.text
    assert tracing.get_trace_manager() is manager_before
