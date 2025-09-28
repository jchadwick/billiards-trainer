"""Unit tests for configuration management endpoints."""

import json
from unittest.mock import Mock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from ..main import create_app


@pytest.fixture()
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def mock_config_module():
    """Mock configuration module for testing."""
    mock_config = Mock()
    mock_config.get_configuration.return_value = {
        "system": {"debug": False, "log_level": "INFO"},
        "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30},
        "vision": {"sensitivity": 0.8},
    }
    return mock_config


@pytest.fixture()
def mock_user():
    """Mock authenticated user."""
    return {"user_id": "test_user", "username": "testuser", "role": "admin"}


class TestConfigurationEndpoints:
    """Test configuration management endpoints."""

    def test_get_configuration_success(self, client, mock_config_module, mock_user):
        """Test successful configuration retrieval."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/")

            assert response.status_code == 200
            data = response.json()

            assert "timestamp" in data
            assert "values" in data
            assert "schema_version" in data
            assert "last_modified" in data
            assert "is_valid" in data
            assert "validation_errors" in data

            # Check configuration values
            values = data["values"]
            assert "system" in values
            assert "camera" in values
            assert "vision" in values

    def test_get_configuration_by_section(self, client, mock_config_module, mock_user):
        """Test configuration retrieval by specific section."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/?section=camera")

            assert response.status_code == 200
            data = response.json()

            assert "values" in data
            values = data["values"]
            assert "camera" in values
            assert len(values) == 1  # Only camera section

    def test_get_configuration_invalid_section(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration retrieval with invalid section."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/?section=invalid_section")

            assert response.status_code == 400
            data = response.json()
            assert "error" in data

    def test_update_configuration_success(self, client, mock_config_module, mock_user):
        """Test successful configuration update."""
        update_data = {"camera": {"fps": 60, "resolution": [1280, 720]}}

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put("/api/v1/config/", json=update_data)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "updated_fields" in data
            assert "validation_errors" in data
            assert "warnings" in data
            assert "rollback_available" in data
            assert "restart_required" in data

    def test_update_configuration_validation_only(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration update in validation-only mode."""
        update_data = {"camera": {"fps": 60}}

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put(
                "/api/v1/config/?validate_only=true", json=update_data
            )

            assert response.status_code == 200
            data = response.json()

            assert "success" in data
            assert data["updated_fields"] == []  # No fields updated in validation mode

    def test_update_configuration_with_warnings(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration update with warnings."""
        update_data = {"camera": {"fps": 120}}  # This should trigger a warning

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put("/api/v1/config/", json=update_data)

            assert response.status_code == 400
            data = response.json()
            assert "warnings" in data["details"]

    def test_update_configuration_force_update(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration update with force flag."""
        update_data = {
            "camera": {"fps": 120}  # This triggers warning but force_update=true
        }

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put("/api/v1/config/?force_update=true", json=update_data)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_update_configuration_validation_errors(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration update with validation errors."""
        update_data = {"camera": {"fps": "invalid"}}  # Should be integer

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put("/api/v1/config/", json=update_data)

            assert response.status_code == 422
            data = response.json()
            assert "validation_errors" in data["details"]

    def test_update_configuration_empty_data(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration update with empty data."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.put("/api/v1/config/", json={})

            assert response.status_code == 422
            data = response.json()
            assert "validation_errors" in data["details"]

    def test_reset_configuration_success(self, client, mock_config_module, mock_user):
        """Test successful configuration reset."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.post("/api/v1/config/reset?confirm=true")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "message" in data

    def test_reset_configuration_without_confirmation(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration reset without confirmation."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            response = client.post("/api/v1/config/reset?confirm=false")

            assert response.status_code == 400
            data = response.json()
            assert "Reset Not Confirmed" in data["message"]

    def test_reset_configuration_with_backup(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration reset with backup creation."""
        mock_config_module.get_configuration.return_value = {"test": "config"}

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            response = client.post(
                "/api/v1/config/reset?confirm=true&backup_current=true"
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            mock_file.assert_called()

    def test_import_configuration_json(self, client, mock_config_module, mock_user):
        """Test configuration import from JSON file."""
        config_data = {
            "camera": {"fps": 30, "resolution": [1920, 1080]},
            "system": {"debug": False},
        }

        json_content = json.dumps(config_data)

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            files = {"file": ("config.json", json_content, "application/json")}
            response = client.post("/api/v1/config/import", files=files)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "sections_imported" in data["data"]

    def test_import_configuration_yaml(self, client, mock_config_module, mock_user):
        """Test configuration import from YAML file."""
        yaml_content = """
camera:
  fps: 30
  resolution: [1920, 1080]
system:
  debug: false
"""

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            files = {"file": ("config.yaml", yaml_content, "application/yaml")}
            response = client.post("/api/v1/config/import", files=files)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True

    def test_import_configuration_invalid_format(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration import with invalid file format."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            files = {"file": ("config.txt", "invalid content", "text/plain")}
            response = client.post("/api/v1/config/import", files=files)

            assert response.status_code == 400
            data = response.json()
            assert "Invalid File Format" in data["message"]

    def test_import_configuration_invalid_json(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration import with invalid JSON."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            files = {"file": ("config.json", "invalid json{", "application/json")}
            response = client.post("/api/v1/config/import", files=files)

            assert response.status_code == 400
            data = response.json()
            assert "File Parse Error" in data["message"]

    def test_import_configuration_validation_only(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration import in validation-only mode."""
        config_data = {"camera": {"fps": 30}}
        json_content = json.dumps(config_data)

        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.AdminRequired", return_value=mock_user),
        ):
            files = {"file": ("config.json", json_content, "application/json")}
            response = client.post(
                "/api/v1/config/import?validate_only=true", files=files
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "sections_found" in data["data"]

    def test_export_configuration_json(self, client, mock_config_module, mock_user):
        """Test configuration export in JSON format."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/export?format=json")

            assert response.status_code == 200
            data = response.json()

            assert "format" in data
            assert "size" in data
            assert "checksum" in data
            assert "timestamp" in data
            assert "data" in data

    def test_export_configuration_yaml(self, client, mock_config_module, mock_user):
        """Test configuration export in YAML format."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/export?format=yaml")

            assert response.status_code == 200
            data = response.json()

            assert data["format"] == "yaml"

    def test_export_configuration_specific_sections(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration export for specific sections."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get(
                "/api/v1/config/export?sections=camera&sections=system"
            )

            assert response.status_code == 200
            data = response.json()

            # Should only include specified sections
            assert "data" in data

    def test_export_configuration_with_metadata(
        self, client, mock_config_module, mock_user
    ):
        """Test configuration export with metadata."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/export?include_metadata=true")

            assert response.status_code == 200
            data = response.json()

            assert "data" in data
            # Should include metadata in the exported data

    def test_download_configuration(self, client, mock_config_module, mock_user):
        """Test configuration download as file."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                return_value=mock_config_module,
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            response = client.get("/api/v1/config/export/download?format=json")

            # Should return a file response
            assert response.status_code == 200
            mock_file.assert_called()

    def test_get_configuration_schema(self, client, mock_user):
        """Test configuration schema retrieval."""
        with patch("backend.api.routes.config.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/config/schema")

            assert response.status_code == 200
            data = response.json()

            assert "type" in data
            assert "properties" in data
            assert data["type"] == "object"

    def test_get_configuration_schema_by_section(self, client, mock_user):
        """Test configuration schema retrieval for specific section."""
        with patch("backend.api.routes.config.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/config/schema?section=camera")

            assert response.status_code == 200
            data = response.json()

            assert "type" in data
            assert "properties" in data

    def test_configuration_error_handling(self, client, mock_user):
        """Test configuration endpoint error handling."""
        with (
            patch(
                "backend.api.routes.config.get_config_module",
                side_effect=Exception("Test error"),
            ),
            patch("backend.api.routes.config.ViewerRequired", return_value=mock_user),
        ):
            response = client.get("/api/v1/config/")

            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    def test_authentication_required(self, client):
        """Test that configuration endpoints require authentication."""
        # Without proper authentication, these should fail
        response = client.get("/api/v1/config/")
        assert response.status_code in [401, 403]

        response = client.put("/api/v1/config/", json={})
        assert response.status_code in [401, 403]

        response = client.post("/api/v1/config/reset?confirm=true")
        assert response.status_code in [401, 403]
