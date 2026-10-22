"""Tests for API metrics middleware registration and request recording."""

import pytest
from fastapi.testclient import TestClient

from backend.api.middleware.metrics import MetricsConfig, MetricsMiddleware
from backend.system.health_monitor import health_monitor
from backend.tests.support.compat_api import create_compatibility_app


def test_create_app_registers_configured_metrics_middleware_and_records_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API stack records request outcomes without starting hardware services."""
    recorded_requests: list[tuple[float, bool]] = []
    monkeypatch.setattr(
        health_monitor,
        "record_api_request",
        lambda response_time, success=True: recorded_requests.append(
            (response_time, success)
        ),
    )
    app = create_compatibility_app(
        {
            "development_mode": True,
            "api": {
                "middleware": {
                    "metrics": {
                        "excluded_paths": [],
                        "slow_request_threshold": 0.25,
                    }
                }
            },
        }
    )

    metrics_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is MetricsMiddleware
    )
    metrics_config = metrics_middleware.kwargs["config"]
    assert isinstance(metrics_config, MetricsConfig)
    assert metrics_config.slow_request_threshold == 0.25
    assert metrics_config.excluded_paths == []

    @app.get("/test/metrics-recording")
    async def metrics_recording_route() -> dict[str, bool]:
        return {"ok": True}

    # Do not enter TestClient as a context manager: that would run the app
    # lifespan and initialize camera and vision services in this unit test.
    response = TestClient(app).get("/test/metrics-recording")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers.get("X-Response-Time") is not None
    assert len(recorded_requests) == 1
    assert recorded_requests[0][0] >= 0
    assert recorded_requests[0][1] is True


def test_create_app_uses_metrics_defaults_when_configuration_is_absent() -> None:
    """The configured middleware stack constructs metrics defaults as needed."""
    app = create_compatibility_app({"development_mode": True})

    metrics_middleware = next(
        middleware
        for middleware in app.user_middleware
        if middleware.cls is MetricsMiddleware
    )

    assert metrics_middleware.kwargs["config"] == MetricsConfig()
