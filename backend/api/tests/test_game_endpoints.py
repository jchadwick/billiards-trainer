"""Unit tests for game state management endpoints."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


@pytest.fixture()
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def mock_core_module():
    """Mock core module for testing."""
    mock_core = Mock()

    # Mock game state
    mock_game_state = Mock()
    mock_game_state.timestamp = datetime.now(timezone.utc).timestamp()
    mock_game_state.frame_number = 123
    mock_game_state.balls = []
    mock_game_state.cue = None
    mock_game_state.table = Mock()
    mock_game_state.game_type = Mock()
    mock_game_state.game_type.value = "practice"
    mock_game_state.is_valid = True
    mock_game_state.state_confidence = 0.95
    mock_game_state.events = []
    mock_game_state.to_dict.return_value = {"test": "data"}

    mock_core.get_current_state.return_value = mock_game_state
    return mock_core


@pytest.fixture()
def mock_user():
    """Mock authenticated user."""
    return {"user_id": "test_user", "username": "testuser", "role": "operator"}


class TestGameStateEndpoints:
    """Test game state management endpoints."""

    def test_get_current_game_state_success(self, client, mock_core_module, mock_user):
        """Test successful current game state retrieval."""
        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state")

            assert response.status_code == 200
            data = response.json()

            assert "timestamp" in data
            assert "frame_number" in data
            assert "balls" in data
            assert "table" in data
            assert "game_type" in data
            assert "is_valid" in data
            assert "confidence" in data
            assert "events" in data

    def test_get_current_game_state_no_events(
        self, client, mock_core_module, mock_user
    ):
        """Test current game state retrieval without events."""
        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state?include_events=false")

            assert response.status_code == 200
            data = response.json()

            assert data["events"] == []

    def test_get_current_game_state_with_trajectories(
        self, client, mock_core_module, mock_user
    ):
        """Test current game state retrieval with trajectories."""
        mock_core_module.predict_trajectories.return_value = []

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state?include_trajectories=true")

            assert response.status_code == 200
            response.json()

            # Trajectory data would be included in a real implementation
            mock_core_module.predict_trajectories.assert_called_once()

    def test_get_current_game_state_no_active_game(
        self, client, mock_core_module, mock_user
    ):
        """Test current game state retrieval when no game is active."""
        mock_core_module.get_current_state.return_value = None

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state")

            assert response.status_code == 200
            data = response.json()

            assert data["frame_number"] == 0
            assert data["balls"] == []
            assert data["game_type"] == "practice"

    def test_get_game_history_success(self, client, mock_core_module, mock_user):
        """Test successful game history retrieval."""
        mock_history = {"states": [], "total_count": 0}
        mock_core_module.get_game_history.return_value = mock_history

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/history")

            assert response.status_code == 200
            data = response.json()

            assert "states" in data
            assert "total_count" in data
            assert "has_more" in data
            assert "time_range" in data

    def test_get_game_history_with_filters(self, client, mock_core_module, mock_user):
        """Test game history retrieval with filters."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_history = {"states": [], "total_count": 0}
        mock_core_module.get_game_history.return_value = mock_history

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get(
                f"/api/v1/game/history?start_time={start_time.isoformat()}&end_time={end_time.isoformat()}&game_type=8ball&limit=50"
            )

            assert response.status_code == 200
            response.json()

            mock_core_module.get_game_history.assert_called_once()

    def test_get_game_history_invalid_time_range(
        self, client, mock_core_module, mock_user
    ):
        """Test game history retrieval with invalid time range."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc) - timedelta(hours=1)  # End before start

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get(
                f"/api/v1/game/history?start_time={start_time.isoformat()}&end_time={end_time.isoformat()}"
            )

            assert response.status_code == 400
            data = response.json()
            assert "Invalid Time Range" in data["message"]

    def test_get_game_history_pagination(self, client, mock_core_module, mock_user):
        """Test game history pagination."""
        mock_history = {"states": [], "total_count": 150}
        mock_core_module.get_game_history.return_value = mock_history

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/history?limit=50&offset=100")

            assert response.status_code == 200
            data = response.json()

            assert data["total_count"] == 150
            assert data["has_more"] is False  # 100 + 0 states < 150 total

    def test_reset_game_state_success(self, client, mock_core_module, mock_user):
        """Test successful game state reset."""
        mock_new_state = Mock()
        mock_new_state.timestamp = datetime.now(timezone.utc).timestamp()
        mock_new_state.frame_number = 0
        mock_new_state.balls = []
        mock_new_state.cue = None
        mock_new_state.table = Mock()
        mock_new_state.game_type = Mock()
        mock_new_state.game_type.value = "practice"
        mock_new_state.is_valid = True
        mock_new_state.state_confidence = 1.0
        mock_new_state.events = []

        mock_core_module.reset_game_state.return_value = mock_new_state

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.OperatorRequired", return_value=mock_user
        ), patch(
            "backend.api.routes.game.GameType"
        ) as mock_game_type:
            mock_game_type.return_value = Mock()

            response = client.post("/api/v1/game/reset?game_type=8ball")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert "new_state" in data
            assert "backup_created" in data
            assert "reset_at" in data

    def test_reset_game_state_with_backup(self, client, mock_core_module, mock_user):
        """Test game state reset with backup creation."""
        # Mock current state for backup
        mock_current_state = Mock()
        mock_current_state.to_dict.return_value = {"backup": "data"}
        mock_core_module.get_current_state.return_value = mock_current_state

        # Mock new state after reset
        mock_new_state = Mock()
        mock_new_state.timestamp = datetime.now(timezone.utc).timestamp()
        mock_new_state.frame_number = 0
        mock_new_state.balls = []
        mock_new_state.cue = None
        mock_new_state.table = Mock()
        mock_new_state.game_type = Mock()
        mock_new_state.game_type.value = "practice"
        mock_new_state.is_valid = True
        mock_new_state.state_confidence = 1.0
        mock_new_state.events = []

        mock_core_module.reset_game_state.return_value = mock_new_state

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.OperatorRequired", return_value=mock_user
        ), patch(
            "backend.api.routes.game.GameType"
        ) as mock_game_type, patch(
            "builtins.open", mock_open()
        ) as mock_file:
            mock_game_type.return_value = Mock()

            response = client.post("/api/v1/game/reset?create_backup=true")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["backup_created"] is True
            mock_file.assert_called()

    def test_reset_game_state_invalid_game_type(
        self, client, mock_core_module, mock_user
    ):
        """Test game state reset with invalid game type."""
        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.OperatorRequired", return_value=mock_user
        ), patch(
            "backend.api.routes.game.GameType",
            side_effect=ValueError("Invalid game type"),
        ):
            response = client.post("/api/v1/game/reset?game_type=invalid")

            assert response.status_code == 400
            data = response.json()
            assert "Invalid Game Type" in data["message"]

    def test_export_session_data_success(self, client, mock_core_module, mock_user):
        """Test successful session data export."""
        mock_current_state = Mock()
        mock_current_state.to_dict.return_value = {"session": "data"}
        mock_core_module.get_current_state.return_value = mock_current_state

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.ViewerRequired", return_value=mock_user
        ), patch(
            "builtins.open", mock_open()
        ), patch(
            "backend.api.routes.game.Path"
        ) as mock_path:
            mock_path.return_value.stat.return_value.st_size = 1024

            response = client.post("/api/v1/game/export")

            assert response.status_code == 200
            data = response.json()

            assert "export_id" in data
            assert "format" in data
            assert "size" in data
            assert "file_path" in data
            assert "checksum" in data
            assert "created_at" in data
            assert "expires_at" in data

    def test_export_session_data_zip_format(self, client, mock_core_module, mock_user):
        """Test session data export in ZIP format."""
        mock_current_state = Mock()
        mock_current_state.to_dict.return_value = {"session": "data"}
        mock_core_module.get_current_state.return_value = mock_current_state

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.ViewerRequired", return_value=mock_user
        ), patch(
            "backend.api.routes.game.zipfile.ZipFile"
        ) as mock_zipfile, patch(
            "backend.api.routes.game.Path"
        ) as mock_path:
            mock_path.return_value.stat.return_value.st_size = 2048

            response = client.post(
                "/api/v1/game/export?format=zip&include_raw_frames=true"
            )

            assert response.status_code == 200
            data = response.json()

            assert data["format"] == "zip"
            mock_zipfile.assert_called()

    def test_export_session_data_with_history(
        self, client, mock_core_module, mock_user
    ):
        """Test session data export including history."""
        mock_current_state = Mock()
        mock_current_state.to_dict.return_value = {"session": "data"}
        mock_core_module.get_current_state.return_value = mock_current_state

        mock_history = {"states": [{"history": "data"}]}
        mock_core_module.get_game_history.return_value = mock_history

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch(
            "backend.api.routes.game.ViewerRequired", return_value=mock_user
        ), patch(
            "builtins.open", mock_open()
        ), patch(
            "backend.api.routes.game.Path"
        ) as mock_path:
            mock_path.return_value.stat.return_value.st_size = 1024

            response = client.post("/api/v1/game/export?include_processed_data=true")

            assert response.status_code == 200
            response.json()

            mock_core_module.get_game_history.assert_called()

    def test_download_session_export_success(self, client, mock_user):
        """Test successful session export download."""
        export_id = "test_export_123"

        with patch(
            "backend.api.routes.game.ViewerRequired", return_value=mock_user
        ), patch("backend.api.routes.game.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.name = f"session_export_{export_id}.zip"

            response = client.get(f"/api/v1/game/export/{export_id}/download")

            assert response.status_code == 200

    def test_download_session_export_not_found(self, client, mock_user):
        """Test session export download when file not found."""
        export_id = "nonexistent_export"

        with patch(
            "backend.api.routes.game.ViewerRequired", return_value=mock_user
        ), patch("backend.api.routes.game.Path") as mock_path, patch(
            "glob.glob", return_value=[]
        ):
            mock_path.return_value.exists.return_value = False

            response = client.get(f"/api/v1/game/export/{export_id}/download")

            assert response.status_code == 404
            data = response.json()
            assert "Export Not Found" in data["message"]

    def test_get_game_statistics_success(self, client, mock_core_module, mock_user):
        """Test successful game statistics retrieval."""
        mock_stats = {
            "games": {"total_sessions": 10, "total_shots": 150},
            "performance": {"average_fps": 29.5, "tracking_accuracy": 0.92},
        }
        mock_core_module.get_game_statistics.return_value = mock_stats

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/stats")

            assert response.status_code == 200
            data = response.json()

            assert "time_range" in data
            assert "games" in data
            assert "performance" in data

    def test_get_game_statistics_different_time_ranges(
        self, client, mock_core_module, mock_user
    ):
        """Test game statistics with different time ranges."""
        time_ranges = ["1h", "6h", "24h", "7d", "30d"]

        mock_core_module.get_game_statistics.return_value = {}

        for time_range in time_ranges:
            with patch(
                "backend.api.routes.game.get_core_module", return_value=mock_core_module
            ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
                response = client.get(f"/api/v1/game/stats?time_range={time_range}")
                assert response.status_code == 200

    def test_get_game_statistics_invalid_time_range(
        self, client, mock_core_module, mock_user
    ):
        """Test game statistics with invalid time range."""
        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core_module
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/stats?time_range=invalid")
            assert response.status_code == 422

    def test_game_endpoints_error_handling(self, client, mock_user):
        """Test game endpoint error handling."""
        with patch(
            "backend.api.routes.game.get_core_module",
            side_effect=Exception("Test error"),
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state")

            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    def test_game_endpoints_without_core_module(self, client, mock_user):
        """Test game endpoints when core module methods are not available."""
        mock_core = Mock()
        # Remove methods to simulate unavailable functionality
        del mock_core.get_current_state
        del mock_core.get_game_history

        with patch(
            "backend.api.routes.game.get_core_module", return_value=mock_core
        ), patch("backend.api.routes.game.ViewerRequired", return_value=mock_user):
            response = client.get("/api/v1/game/state")
            # Should still work with fallback implementation
            assert response.status_code == 200

    def test_authentication_required(self, client):
        """Test that game endpoints require authentication."""
        # Without proper authentication, these should fail
        response = client.get("/api/v1/game/state")
        assert response.status_code in [401, 403]

        response = client.get("/api/v1/game/history")
        assert response.status_code in [401, 403]

        response = client.post("/api/v1/game/reset")
        assert response.status_code in [401, 403]

        response = client.post("/api/v1/game/export")
        assert response.status_code in [401, 403]
