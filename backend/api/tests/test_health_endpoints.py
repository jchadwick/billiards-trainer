"""Unit tests for health check endpoints."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_app_state():
    """Mock application state for testing."""
    mock_state = Mock()
    mock_state.is_healthy = True
    mock_state.startup_time = datetime.now(timezone.utc).timestamp()
    mock_state.core_module = Mock()
    mock_state.config_module = Mock()
    mock_state.websocket_manager = Mock()
    return mock_state


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self, client, mock_app_state):
        """Test basic health check without details."""
        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "uptime" in data
            assert data["version"] == "1.0.0"
            assert data["components"] == {}
            assert data["metrics"] is None

    def test_health_check_with_details(self, client, mock_app_state):
        """Test health check with component details."""
        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/?include_details=true")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "components" in data
            assert len(data["components"]) > 0

            # Check component structure
            for _component_name, component in data["components"].items():
                assert "name" in component
                assert "status" in component
                assert "message" in component
                assert "last_check" in component

    @patch("backend.api.routes.health.psutil")
    def test_health_check_with_metrics(self, mock_psutil, client, mock_app_state):
        """Test health check with performance metrics."""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
        mock_psutil.disk_usage.return_value = Mock(
            used=100 * 1024**3, total=500 * 1024**3
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000000, bytes_recv=2000000, packets_sent=1000, packets_recv=2000
        )

        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/?include_metrics=true")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "healthy"
            assert "metrics" in data
            assert data["metrics"] is not None

            metrics = data["metrics"]
            assert "cpu_usage" in metrics
            assert "memory_usage" in metrics
            assert "disk_usage" in metrics
            assert "network_io" in metrics

    def test_health_check_unhealthy_state(self, client, mock_app_state):
        """Test health check when system is unhealthy."""
        mock_app_state.is_healthy = False

        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "unhealthy"

    def test_version_endpoint(self, client):
        """Test version information endpoint."""
        response = client.get("/api/v1/health/version")

        assert response.status_code == 200
        data = response.json()

        assert "version" in data
        assert "build_date" in data
        assert "capabilities" in data
        assert "api_version" in data
        assert "supported_clients" in data

        # Check capabilities structure
        capabilities = data["capabilities"]
        assert "vision_processing" in capabilities
        assert "projector_support" in capabilities
        assert "calibration_modes" in capabilities
        assert "game_types" in capabilities
        assert "export_formats" in capabilities
        assert "max_concurrent_sessions" in capabilities

    @patch("backend.api.routes.health.psutil")
    def test_metrics_endpoint(self, mock_psutil, client):
        """Test performance metrics endpoint."""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 35.0
        mock_psutil.virtual_memory.return_value = Mock(percent=45.0)
        mock_psutil.disk_usage.return_value = Mock(
            used=200 * 1024**3, total=1000 * 1024**3
        )
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=5000000,
            bytes_recv=10000000,
            packets_sent=5000,
            packets_recv=10000,
        )

        response = client.get("/api/v1/health/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        assert "network_io" in data
        assert "api_requests_per_second" in data
        assert "websocket_connections" in data
        assert "average_response_time" in data

    def test_metrics_endpoint_with_time_range(self, client):
        """Test metrics endpoint with different time ranges."""
        time_ranges = ["5m", "15m", "1h", "6h", "24h"]

        for time_range in time_ranges:
            response = client.get(f"/api/v1/health/metrics?time_range={time_range}")
            assert response.status_code == 200

    def test_metrics_endpoint_invalid_time_range(self, client):
        """Test metrics endpoint with invalid time range."""
        response = client.get("/api/v1/health/metrics?time_range=invalid")
        assert response.status_code == 422  # Validation error

    def test_shutdown_endpoint(self, client):
        """Test graceful shutdown endpoint."""
        response = client.post("/api/v1/health/shutdown")

        # This endpoint typically requires admin auth, but we're testing without auth
        # In a real test, you'd mock the authentication
        assert response.status_code in [200, 401, 403]  # Depends on auth setup

    def test_shutdown_endpoint_with_delay(self, client):
        """Test shutdown endpoint with delay parameter."""
        response = client.post("/api/v1/health/shutdown?delay=30")

        # This endpoint typically requires admin auth
        assert response.status_code in [200, 401, 403]

    def test_shutdown_endpoint_invalid_delay(self, client):
        """Test shutdown endpoint with invalid delay."""
        response = client.post("/api/v1/health/shutdown?delay=9999")
        assert response.status_code == 422  # Validation error

    def test_readiness_check_healthy(self, client, mock_app_state):
        """Test readiness check when system is healthy."""
        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/ready")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "ready"
            assert "components" in data
            assert "timestamp" in data

    def test_readiness_check_unhealthy(self, client, mock_app_state):
        """Test readiness check when system is unhealthy."""
        mock_app_state.is_healthy = False
        mock_app_state.core_module = None

        with patch(
            "backend.api.routes.health.get_app_state", return_value=mock_app_state
        ):
            response = client.get("/api/v1/health/ready")

            assert response.status_code == 503
            data = response.json()

            assert data["status"] == "not_ready"

    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "message" in data

    def test_health_check_error_handling(self, client):
        """Test health check error handling."""
        with patch(
            "backend.api.routes.health.get_app_state",
            side_effect=Exception("Test error"),
        ):
            response = client.get("/api/v1/health/")

            assert response.status_code == 500
            data = response.json()

            assert "error" in data
            assert "message" in data

    @patch("backend.api.routes.health.psutil", None)
    def test_metrics_without_psutil(self, client):
        """Test metrics endpoint when psutil is not available."""
        response = client.get("/api/v1/health/metrics")

        assert response.status_code == 200
        data = response.json()

        # Should return default values when psutil is not available
        assert data["cpu_usage"] == 0.0
        assert data["memory_usage"] == 0.0
        assert data["disk_usage"] == 0.0
