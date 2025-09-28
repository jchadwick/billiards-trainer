"""Tests for error handling middleware."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from ...middleware.error_handler import (
    ERROR_CODES,
    CustomHTTPException,
    ErrorHandler,
    ErrorHandlerConfig,
    get_error_handler,
    raise_error,
    setup_error_handling,
)
from ...models.responses import ErrorResponse


class TestErrorHandlerConfig:
    """Test ErrorHandlerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ErrorHandlerConfig()

        assert config.include_traceback is False
        assert config.log_errors is True
        assert config.log_level == "ERROR"
        assert config.include_request_details is True
        assert config.sanitize_errors is True
        assert config.max_detail_length == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ErrorHandlerConfig(
            include_traceback=True,
            log_errors=False,
            log_level="DEBUG",
            sanitize_errors=False,
            max_detail_length=500,
        )

        assert config.include_traceback is True
        assert config.log_errors is False
        assert config.log_level == "DEBUG"
        assert config.sanitize_errors is False
        assert config.max_detail_length == 500


class TestCustomHTTPException:
    """Test CustomHTTPException class."""

    def test_valid_error_code(self):
        """Test creation with valid error code."""
        exc = CustomHTTPException("AUTH_001", {"user_id": "123"})

        assert exc.error_code == "AUTH_001"
        assert exc.status_code == 401
        assert exc.detail == "Invalid credentials"
        assert exc.error_details == {"user_id": "123"}

    def test_invalid_error_code(self):
        """Test creation with invalid error code raises ValueError."""
        with pytest.raises(ValueError, match="Unknown error code"):
            CustomHTTPException("INVALID_CODE")

    def test_without_details(self):
        """Test creation without error details."""
        exc = CustomHTTPException("VAL_001")

        assert exc.error_code == "VAL_001"
        assert exc.error_details == {}


class TestErrorHandler:
    """Test ErrorHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ErrorHandlerConfig(
            include_traceback=True, sanitize_errors=True, max_detail_length=100
        )
        self.handler = ErrorHandler(self.config)

    def test_format_error_response(self):
        """Test error response formatting."""
        response = self.handler.format_error_response(
            error_code="VAL_001",
            message="Test error",
            details={"field": "value"},
            request_id="test-123",
        )

        assert response.error == "VAL_001"
        assert response.message == "Test error"
        assert response.details == {"field": "value"}
        assert response.request_id == "test-123"
        assert isinstance(response.timestamp, datetime)

    def test_format_error_response_with_request(self):
        """Test error response formatting with request details."""
        mock_request = Mock()
        mock_request.url.path = "/api/test"
        mock_request.method = "POST"

        response = self.handler.format_error_response(
            error_code="SYS_001",
            message="System error",
            request=mock_request,
            request_id="req-456",
        )

        assert response.path == "/api/test"
        assert response.method == "POST"

    @patch("backend.api.middleware.error_handler.logger")
    def test_log_error(self, mock_logger):
        """Test error logging."""
        error = ValueError("Test error")
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.query_params = {}
        mock_request.headers = {"User-Agent": "test-agent"}
        mock_request.client.host = "127.0.0.1"

        self.handler.log_error(error, mock_request, "req-123")

        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][1] == "Error occurred: Test error"

    def test_sanitize_details(self):
        """Test details sanitization."""
        details = {
            "password": "secret123",
            "token": "abc123",
            "safe_field": "safe_value",
            "long_field": "x" * 200,
        }

        sanitized = self.handler._sanitize_details(details)

        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["safe_field"] == "safe_value"
        assert len(sanitized["long_field"]) == 103  # 100 chars + "..."

    def test_track_error(self):
        """Test error tracking."""
        error = RuntimeError("Test error")
        context = {"test": "value"}

        initial_count = self.handler.error_count
        self.handler._track_error(error, context)

        assert self.handler.error_count == initial_count + 1
        assert len(self.handler.last_errors) == 1
        assert self.handler.last_errors[0]["error_type"] == "RuntimeError"

    def test_get_error_stats(self):
        """Test error statistics."""
        # Track some errors
        error1 = ValueError("Error 1")
        error2 = RuntimeError("Error 2")
        error3 = ValueError("Error 3")

        self.handler._track_error(error1, {})
        self.handler._track_error(error2, {})
        self.handler._track_error(error3, {})

        stats = self.handler.get_error_stats()

        assert stats["total_errors"] == 3
        assert "ValueError" in stats["error_types"]
        assert "RuntimeError" in stats["error_types"]
        assert stats["error_types"]["ValueError"] == 2
        assert stats["error_types"]["RuntimeError"] == 1


class TestErrorHandlerIntegration:
    """Test error handler integration with FastAPI."""

    def setup_method(self):
        """Set up test FastAPI app."""
        self.app = FastAPI()

        # Set up error handling
        config = ErrorHandlerConfig(include_traceback=True)
        setup_error_handling(self.app, config)

        # Add test routes
        @self.app.get("/test-success")
        async def test_success():
            return {"message": "success"}

        @self.app.get("/test-http-error")
        async def test_http_error():
            raise HTTPException(status_code=404, detail="Not found")

        @self.app.get("/test-custom-error")
        async def test_custom_error():
            raise CustomHTTPException("AUTH_001", {"user_id": "123"})

        @self.app.get("/test-validation-error")
        async def test_validation_error():
            raise RequestValidationError(
                [
                    {
                        "loc": ("field",),
                        "msg": "field required",
                        "type": "value_error.missing",
                    }
                ]
            )

        @self.app.get("/test-general-error")
        async def test_general_error():
            raise ValueError("Unexpected error")

        @self.app.post("/test-validation-body")
        async def test_validation_body(data: TestModel):
            return {"received": data.dict()}

        self.client = TestClient(self.app)

    def test_successful_request(self):
        """Test successful request handling."""
        response = self.client.get("/test-success")

        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_http_exception_handling(self):
        """Test standard HTTP exception handling."""
        response = self.client.get("/test-http-error")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "RES_001"
        assert data["message"] == "Not found"
        assert "request_id" in data

    def test_custom_http_exception_handling(self):
        """Test custom HTTP exception handling."""
        response = self.client.get("/test-custom-error")

        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "AUTH_001"
        assert data["message"] == "Invalid credentials"
        assert data["details"]["user_id"] == "123"

    def test_validation_exception_handling(self):
        """Test request validation exception handling."""
        response = self.client.post("/test-validation-body", json={})

        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "VAL_001"
        assert "validation_errors" in data["details"]

    def test_general_exception_handling(self):
        """Test general exception handling."""
        response = self.client.get("/test-general-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "SYS_001"
        assert data["message"] == ERROR_CODES["SYS_001"]["message"]
        assert "request_id" in data

    def test_request_id_in_headers(self):
        """Test that request ID is included in response headers."""
        response = self.client.get("/test-http-error")

        assert "X-Request-ID" in response.headers
        data = response.json()
        assert response.headers["X-Request-ID"] == data["request_id"]


class TestModel(BaseModel):
    """Test model for validation testing."""

    required_field: str = Field(..., description="Required field")


class TestErrorUtils:
    """Test error utility functions."""

    def test_raise_error(self):
        """Test raise_error utility function."""
        with pytest.raises(CustomHTTPException) as exc_info:
            raise_error("VAL_001", {"field": "test"})

        exc = exc_info.value
        assert exc.error_code == "VAL_001"
        assert exc.error_details == {"field": "test"}

    def test_get_error_handler(self):
        """Test get_error_handler function."""
        # Should return None initially
        handler = get_error_handler()
        assert handler is None

        # Set up error handling and test again
        app = FastAPI()
        setup_error_handling(app)

        handler = get_error_handler()
        assert handler is not None
        assert isinstance(handler, ErrorHandler)


class TestErrorCodes:
    """Test error code mappings."""

    def test_error_codes_exist(self):
        """Test that all expected error codes exist."""
        expected_codes = [
            "AUTH_001",
            "AUTH_002",
            "AUTH_003",
            "VAL_001",
            "VAL_002",
            "VAL_003",
            "RES_001",
            "RATE_001",
            "SYS_001",
            "CAM_001",
            "PROC_001",
            "WS_001",
        ]

        for code in expected_codes:
            assert code in ERROR_CODES
            assert "status" in ERROR_CODES[code]
            assert "message" in ERROR_CODES[code]

    def test_error_code_status_mapping(self):
        """Test that error codes have correct HTTP status codes."""
        # 4xx errors
        assert ERROR_CODES["AUTH_001"]["status"] == 401
        assert ERROR_CODES["VAL_001"]["status"] == 400
        assert ERROR_CODES["RES_001"]["status"] == 404
        assert ERROR_CODES["RATE_001"]["status"] == 429

        # 5xx errors
        assert ERROR_CODES["SYS_001"]["status"] == 500
        assert ERROR_CODES["CAM_001"]["status"] == 503


class TestErrorResponseModel:
    """Test ErrorResponse model."""

    def test_error_response_creation(self):
        """Test ErrorResponse model creation."""
        response = ErrorResponse(
            error="TEST_001",
            message="Test error",
            details={"key": "value"},
            request_id="req-123",
        )

        assert response.error == "TEST_001"
        assert response.message == "Test error"
        assert response.details == {"key": "value"}
        assert response.request_id == "req-123"
        assert isinstance(response.timestamp, datetime)

    def test_error_response_serialization(self):
        """Test ErrorResponse JSON serialization."""
        response = ErrorResponse(error="TEST_001", message="Test error")

        data = response.model_dump(mode="json")

        assert data["error"] == "TEST_001"
        assert data["message"] == "Test error"
        assert "timestamp" in data


@pytest.fixture()
def mock_request():
    """Mock request fixture."""
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = "/test"
    request.query_params = {}
    request.headers = {"User-Agent": "test-agent"}
    request.client.host = "127.0.0.1"
    request.state = Mock()
    return request


class TestErrorHandlerEdgeCases:
    """Test edge cases and error conditions."""

    def test_handler_with_disabled_logging(self):
        """Test error handler with logging disabled."""
        config = ErrorHandlerConfig(log_errors=False)
        handler = ErrorHandler(config)

        # Should not raise exception when logging is disabled
        error = ValueError("Test")
        handler.log_error(error)  # Should do nothing

    def test_handler_with_none_request(self):
        """Test error handler with None request."""
        config = ErrorHandlerConfig()
        handler = ErrorHandler(config)

        response = handler.format_error_response(
            error_code="SYS_001", message="Test", request=None
        )

        assert response.path is None
        assert response.method is None

    def test_sanitize_empty_details(self):
        """Test sanitizing empty details."""
        config = ErrorHandlerConfig()
        handler = ErrorHandler(config)

        result = handler._sanitize_details({})
        assert result == {}

    def test_error_tracking_limit(self):
        """Test error tracking with limit."""
        config = ErrorHandlerConfig()
        handler = ErrorHandler(config)

        # Add more than 100 errors to test limit
        for i in range(105):
            error = ValueError(f"Error {i}")
            handler._track_error(error, {})

        # Should only keep last 100
        assert len(handler.last_errors) == 100
        assert handler.error_count == 105


if __name__ == "__main__":
    pytest.main([__file__])
