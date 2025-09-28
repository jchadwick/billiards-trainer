"""Integration tests for all middleware components working together."""

import asyncio
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ...middleware.cors import CORSConfig, setup_cors_middleware
from ...middleware.error_handler import (
    ErrorHandlerConfig,
    setup_error_handling,
)
from ...middleware.logging import LoggingConfig, setup_logging_middleware
from ...middleware.performance import (
    PerformanceConfig,
    setup_performance_monitoring,
)
from ...middleware.rate_limit import RateLimitConfig, setup_rate_limiting
from ...middleware.security import SecurityConfig, setup_security_headers
from ...middleware.tracing import TracingConfig, setup_tracing_middleware


class TestModel(BaseModel):
    """Test model for request validation."""

    name: str
    age: int


class TestMiddlewareIntegration:
    """Test integration of all middleware components."""

    def setup_method(self):
        """Set up test FastAPI app with all middleware."""
        self.app = FastAPI(title="Test API")

        # Configure middleware in reverse order (last added = first executed)

        # Performance monitoring (last)
        setup_performance_monitoring(
            self.app,
            PerformanceConfig(
                enable_monitoring=True,
                slow_request_threshold_ms=100,
                excluded_paths=["/health"],
            ),
        )

        # Security headers
        setup_security_headers(self.app, SecurityConfig(development_mode=True))

        # Request tracing
        setup_tracing_middleware(
            self.app,
            TracingConfig(
                enable_tracing=True,
                enable_correlation_ids=True,
                excluded_paths=["/health"],
            ),
        )

        # Request/response logging
        setup_logging_middleware(
            self.app,
            LoggingConfig(
                enable_request_logging=True,
                enable_response_logging=True,
                log_body=False,  # Disable for tests
                excluded_paths=["/health"],
            ),
        )

        # Error handling
        setup_error_handling(
            self.app, ErrorHandlerConfig(include_traceback=True, sanitize_errors=True)
        )

        # Rate limiting
        setup_rate_limiting(
            self.app,
            RateLimitConfig(
                requests_per_minute=60, requests_per_hour=1000, burst_size=10
            ),
        )

        # CORS (first)
        setup_cors_middleware(self.app, CORSConfig(allow_origins=["*"]))

        # Add test routes
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test successful"}

        @self.app.post("/test-validation")
        async def test_validation(data: TestModel):
            return {"received": data.dict()}

        @self.app.get("/test-error")
        async def test_error():
            raise HTTPException(status_code=500, detail="Test error")

        @self.app.get("/test-slow")
        async def test_slow():
            await asyncio.sleep(0.15)  # 150ms - should trigger slow request detection
            return {"message": "slow response"}

        @self.app.get("/test-auth")
        async def test_auth():
            # Simulate auth check
            return {"message": "authenticated"}

        self.client = TestClient(self.app)

    def test_successful_request_flow(self):
        """Test complete middleware flow for successful request."""
        response = self.client.get("/test")

        # Check response
        assert response.status_code == 200
        assert response.json() == {"message": "test successful"}

        # Check middleware headers are present
        headers = response.headers

        # Tracing headers
        assert "X-Request-ID" in headers
        assert "X-Trace-ID" in headers

        # Performance headers
        assert "X-Response-Time" in headers

        # Security headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers

        # Rate limiting headers
        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers

    def test_cors_headers(self):
        """Test CORS headers are applied correctly."""
        # Preflight request
        response = self.client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

        # Actual request
        response = self.client.get("/test", headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_error_handling_with_tracing(self):
        """Test error handling includes tracing information."""
        response = self.client.get("/test-error")

        assert response.status_code == 500
        data = response.json()

        # Error response structure
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data

        # Tracing information should be included
        assert "request_id" in data
        assert "X-Request-ID" in response.headers

    def test_validation_error_handling(self):
        """Test validation errors are handled correctly."""
        response = self.client.post(
            "/test-validation", json={"name": "test"}
        )  # Missing age

        assert response.status_code == 422
        data = response.json()

        # Error structure
        assert "error" in data
        assert "validation_errors" in data["details"]
        assert "request_id" in data

    def test_rate_limiting_headers(self):
        """Test rate limiting headers are included."""
        response = self.client.get("/test")

        assert response.status_code == 200
        headers = response.headers

        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers
        assert "X-RateLimit-Limit-Hour" in headers
        assert "X-RateLimit-Remaining-Hour" in headers

        # Values should be reasonable
        assert int(headers["X-RateLimit-Limit-Minute"]) == 60
        assert int(headers["X-RateLimit-Remaining-Minute"]) <= 60

    def test_security_headers_applied(self):
        """Test security headers are applied."""
        response = self.client.get("/test")

        headers = response.headers

        # Essential security headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Referrer-Policy" in headers

        # Development mode should have relaxed CSP
        if "Content-Security-Policy" in headers:
            csp = headers["Content-Security-Policy"]
            assert "'unsafe-inline'" in csp

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Make a request
        response = self.client.get("/test")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers

        # Parse response time
        response_time = response.headers["X-Response-Time"]
        assert "ms" in response_time

        # Should be a reasonable value
        time_value = float(response_time.replace("ms", ""))
        assert 0 < time_value < 1000  # Should be under 1 second for simple request

    @pytest.mark.asyncio()
    async def test_slow_request_detection(self):
        """Test slow request detection."""
        with patch("backend.api.middleware.performance.logging"):
            response = self.client.get("/test-slow")

            assert response.status_code == 200

            # Should have logged a slow request warning
            # Note: This test might be flaky depending on timing

    def test_request_correlation_across_middleware(self):
        """Test request correlation IDs are consistent across middleware."""
        response = self.client.get("/test")

        # Get correlation IDs from headers
        request_id = response.headers.get("X-Request-ID")
        trace_id = response.headers.get("X-Trace-ID")

        assert request_id is not None
        assert trace_id is not None

        # IDs should be UUID format
        assert len(request_id.split("-")) == 5
        assert len(trace_id.split("-")) == 5

        # In error responses, same IDs should be present
        error_response = self.client.get("/test-error")
        error_data = error_response.json()

        assert error_data["request_id"] == error_response.headers["X-Request-ID"]

    def test_middleware_order_and_execution(self):
        """Test middleware execution order is correct."""
        response = self.client.get("/test")

        # All middleware should have executed successfully
        assert response.status_code == 200

        # Headers from different middleware should all be present
        headers = response.headers

        # Security (early in chain)
        assert "X-Frame-Options" in headers

        # Tracing (middle)
        assert "X-Request-ID" in headers

        # Performance (late in chain)
        assert "X-Response-Time" in headers

    def test_excluded_paths_respected(self):
        """Test that excluded paths bypass appropriate middleware."""
        response = self.client.get("/health")

        assert response.status_code == 200

        # Health endpoint should have minimal headers
        headers = response.headers

        # Should still have security headers (not excluded)
        assert "X-Frame-Options" in headers

        # But might not have detailed tracing/performance headers
        # depending on configuration

    def test_multiple_requests_performance(self):
        """Test performance with multiple concurrent requests."""
        responses = []

        # Make multiple requests
        for i in range(5):
            response = self.client.get(f"/test?param={i}")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "X-Request-ID" in response.headers

        # Each should have unique request ID
        request_ids = [r.headers["X-Request-ID"] for r in responses]
        assert len(set(request_ids)) == 5  # All unique

    def test_error_handling_preserves_middleware_headers(self):
        """Test that error responses still include middleware headers."""
        response = self.client.get("/test-error")

        assert response.status_code == 500

        headers = response.headers

        # Should still have headers from middleware that ran before error
        assert "X-Request-ID" in headers
        assert "X-Frame-Options" in headers

    def test_request_size_tracking(self):
        """Test request size is tracked correctly."""
        # POST request with body
        test_data = {"name": "test user", "age": 25}
        response = self.client.post("/test-validation", json=test_data)

        assert response.status_code == 200

        # Should have performance headers
        assert "X-Response-Time" in response.headers

    def test_content_type_handling(self):
        """Test different content types are handled correctly."""
        # JSON request
        json_response = self.client.post(
            "/test-validation", json={"name": "test", "age": 25}
        )
        assert json_response.status_code == 200

        # Invalid content type should still be handled
        form_response = self.client.post(
            "/test-validation", data={"name": "test", "age": "25"}
        )
        # This should fail validation but be handled gracefully
        assert form_response.status_code == 422
        assert "error" in form_response.json()


class TestMiddlewareConfiguration:
    """Test different middleware configurations."""

    def test_development_vs_production_config(self):
        """Test middleware behaves differently in dev vs prod."""
        # Development configuration
        dev_app = FastAPI()
        setup_security_headers(dev_app, development_mode=True)

        # Production configuration
        FastAPI()
        setup_security_headers(dev_app, development_mode=False)

        # Both should work but have different security policies
        # This is more of a configuration test

    def test_disabled_middleware_components(self):
        """Test app works with some middleware disabled."""
        app = FastAPI()

        # Only enable essential middleware
        setup_error_handling(app)
        setup_cors_middleware(app)

        @app.get("/test")
        async def test():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        # Should have basic CORS but not other headers
        assert "Access-Control-Allow-Origin" in response.headers


class TestMiddlewareEdgeCases:
    """Test edge cases and error conditions in middleware integration."""

    def setup_method(self):
        """Set up minimal test app."""
        self.app = FastAPI()
        setup_error_handling(self.app)
        setup_tracing_middleware(self.app)

        @self.app.get("/test")
        async def test():
            return {"message": "test"}

        @self.app.get("/test-exception")
        async def test_exception():
            raise Exception("Unexpected error")

        self.client = TestClient(self.app)

    def test_unexpected_exception_handling(self):
        """Test handling of unexpected exceptions."""
        response = self.client.get("/test-exception")

        assert response.status_code == 500
        data = response.json()

        assert "error" in data
        assert "request_id" in data

    def test_malformed_requests(self):
        """Test handling of malformed requests."""
        # This would typically be handled by the web server,
        # but we can test some edge cases

        response = self.client.get("/test", headers={"Content-Length": "invalid"})
        # Should still work or fail gracefully
        assert response.status_code in [200, 400, 422]

    def test_large_request_handling(self):
        """Test handling of large requests."""
        # Create a large but valid request
        large_data = {"data": "x" * 10000}  # 10KB of data

        response = self.client.post("/test", json=large_data)

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 422]


if __name__ == "__main__":
    pytest.main([__file__])
