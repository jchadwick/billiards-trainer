"""Unit tests for calibration management endpoints."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ..main import create_app


@pytest.fixture()
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def mock_core_module():
    """Mock core module for testing."""
    mock_core = Mock()
    mock_core.apply_calibration.return_value = True
    return mock_core


@pytest.fixture()
def mock_user():
    """Mock authenticated user."""
    return {"user_id": "test_user", "username": "testuser", "role": "operator"}


@pytest.fixture()
def sample_calibration_session():
    """Sample calibration session data."""
    return {
        "session_id": "cal_20240115_120000_abcd1234",
        "calibration_type": "standard",
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=300),
        "points_captured": 4,
        "points_required": 9,
        "status": "in_progress",
        "created_by": "test_user",
        "points": [
            {
                "point_id": "corner_1",
                "screen_x": 100.0,
                "screen_y": 100.0,
                "world_x": 0.1,
                "world_y": 0.1,
                "confidence": 0.95,
                "captured_at": datetime.now(timezone.utc),
                "captured_by": "test_user",
            }
        ],
        "metadata": {
            "timeout_seconds": 300,
            "force_restart": False,
            "user_agent": "api",
        },
    }


class TestCalibrationEndpoints:
    """Test calibration management endpoints."""

    def test_start_calibration_success(self, client, mock_core_module, mock_user):
        """Test successful calibration start."""
        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", {}),
        ):
            response = client.post(
                "/api/v1/calibration/start?calibration_type=standard"
            )

            assert response.status_code == 200
            data = response.json()

            assert "session" in data
            assert "instructions" in data
            assert "expected_points" in data
            assert "timeout" in data

            session = data["session"]
            assert session["calibration_type"] == "standard"
            assert session["status"] == "in_progress"
            assert session["points_captured"] == 0
            assert session["points_required"] == 9  # Standard mode requires 9 points

    def test_start_calibration_different_types(
        self, client, mock_core_module, mock_user
    ):
        """Test starting calibration with different types."""
        calibration_types = [("quick", 4), ("standard", 9), ("advanced", 16)]

        for cal_type, expected_points in calibration_types:
            with (
                patch(
                    "backend.api.routes.calibration.get_core_module",
                    return_value=mock_core_module,
                ),
                patch(
                    "backend.api.routes.calibration.OperatorRequired",
                    return_value=mock_user,
                ),
                patch("backend.api.routes.calibration._calibration_sessions", {}),
            ):
                response = client.post(
                    f"/api/v1/calibration/start?calibration_type={cal_type}"
                )

                assert response.status_code == 200
                data = response.json()
                assert data["expected_points"] == expected_points

    def test_start_calibration_with_active_session(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test starting calibration when one is already active."""
        # Setup an active session
        existing_sessions = {"existing_session": sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch(
                "backend.api.routes.calibration._calibration_sessions",
                existing_sessions,
            ),
        ):
            response = client.post("/api/v1/calibration/start")

            assert response.status_code == 409
            data = response.json()
            assert "Calibration Already In Progress" in data["message"]

    def test_start_calibration_force_restart(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test starting calibration with force restart."""
        # Setup an active session
        existing_sessions = {"existing_session": sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch(
                "backend.api.routes.calibration._calibration_sessions",
                existing_sessions,
            ),
        ):
            response = client.post("/api/v1/calibration/start?force_restart=true")

            assert response.status_code == 200
            response.json()

            # Original session should be cancelled
            assert existing_sessions["existing_session"]["status"] == "cancelled"

    def test_start_calibration_custom_timeout(
        self, client, mock_core_module, mock_user
    ):
        """Test starting calibration with custom timeout."""
        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", {}),
        ):
            response = client.post("/api/v1/calibration/start?timeout_seconds=600")

            assert response.status_code == 200
            data = response.json()
            assert data["timeout"] == 600

    def test_start_calibration_invalid_timeout(
        self, client, mock_core_module, mock_user
    ):
        """Test starting calibration with invalid timeout."""
        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
        ):
            response = client.post(
                "/api/v1/calibration/start?timeout_seconds=30"
            )  # Too low
            assert response.status_code == 422

            response = client.post(
                "/api/v1/calibration/start?timeout_seconds=2000"
            )  # Too high
            assert response.status_code == 422

    def test_capture_calibration_point_success(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test successful calibration point capture."""
        session_id = sample_calibration_session["session_id"]
        sessions = {session_id: sample_calibration_session}

        point_data = {
            "point_id": "corner_2",
            "screen_position": [200.0, 200.0],
            "world_position": [0.2, 0.2],
            "confidence": 0.98,
        }

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/points", params=point_data
            )

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["point_id"] == "corner_2"
            assert data["total_points"] == 2  # Original 1 + new 1
            assert data["remaining_points"] == 7  # 9 required - 2 captured

    def test_capture_calibration_point_invalid_session(
        self, client, mock_core_module, mock_user
    ):
        """Test capturing point with invalid session ID."""
        point_data = {
            "point_id": "corner_1",
            "screen_position": [100.0, 100.0],
            "world_position": [0.1, 0.1],
        }

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", {}),
        ):
            response = client.post(
                "/api/v1/calibration/invalid_session/points", params=point_data
            )

            assert response.status_code == 404
            data = response.json()
            assert "Calibration Session Not Found" in data["message"]

    def test_capture_calibration_point_invalid_coordinates(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test capturing point with invalid coordinates."""
        session_id = sample_calibration_session["session_id"]
        sessions = {session_id: sample_calibration_session}

        # Test invalid screen position
        point_data = {
            "point_id": "corner_1",
            "screen_position": [100.0],  # Only one coordinate
            "world_position": [0.1, 0.1],
        }

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/points", params=point_data
            )

            assert response.status_code == 400
            data = response.json()
            assert "Invalid Screen Position" in data["message"]

    def test_capture_calibration_point_out_of_range(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test capturing point with coordinates out of range."""
        session_id = sample_calibration_session["session_id"]
        sessions = {session_id: sample_calibration_session}

        point_data = {
            "point_id": "corner_1",
            "screen_position": [5000.0, 5000.0],  # Out of range
            "world_position": [0.1, 0.1],
        }

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/points", params=point_data
            )

            assert response.status_code == 400
            data = response.json()
            assert "Invalid Screen Coordinates" in data["message"]

    def test_capture_calibration_point_completion(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test calibration completion when enough points are captured."""
        session_id = sample_calibration_session["session_id"]
        # Set up session with 8 points already (need 9 total)
        sample_calibration_session["points_captured"] = 8
        sample_calibration_session["points"] = [
            {"point_id": f"point_{i}"} for i in range(8)
        ]
        sessions = {session_id: sample_calibration_session}

        point_data = {
            "point_id": "final_point",
            "screen_position": [300.0, 300.0],
            "world_position": [0.3, 0.3],
        }

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/points", params=point_data
            )

            assert response.status_code == 200
            data = response.json()

            assert data["can_proceed"] is True
            assert sessions[session_id]["status"] == "ready_to_apply"

    def test_apply_calibration_success(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test successful calibration application."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "ready_to_apply"
        sample_calibration_session["points_captured"] = 9
        sample_calibration_session["accuracy"] = 0.95
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
            patch("builtins.open"),
            patch(
                "backend.api.routes.calibration.CalibrationMath.calculate_homography",
                return_value=np.eye(3),
            ),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/apply")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["accuracy"] == 0.95
            assert "applied_at" in data
            assert "transformation_matrix" in data
            assert sessions[session_id]["status"] == "applied"

    def test_apply_calibration_not_ready(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test applying calibration when not ready."""
        session_id = sample_calibration_session["session_id"]
        # Keep status as "in_progress"
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/apply")

            assert response.status_code == 400
            data = response.json()
            assert "Calibration Not Ready" in data["message"]

    def test_apply_calibration_insufficient_points(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test applying calibration with insufficient points."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "ready_to_apply"
        sample_calibration_session["points_captured"] = 2  # Less than minimum 4
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/apply")

            assert response.status_code == 400
            data = response.json()
            assert "Insufficient Calibration Points" in data["message"]

    def test_apply_calibration_low_accuracy(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test applying calibration with low accuracy."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "ready_to_apply"
        sample_calibration_session["points_captured"] = 9
        sample_calibration_session["accuracy"] = 0.5  # Below threshold
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/apply")

            assert response.status_code == 400
            data = response.json()
            assert "Low Calibration Accuracy" in data["message"]

    def test_apply_calibration_force_apply(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test force applying calibration despite low accuracy."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "ready_to_apply"
        sample_calibration_session["points_captured"] = 9
        sample_calibration_session["accuracy"] = 0.5  # Below threshold
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
            patch("builtins.open"),
            patch(
                "backend.api.routes.calibration.CalibrationMath.calculate_homography",
                return_value=np.eye(3),
            ),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/apply?force_apply=true"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_validate_calibration_success(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test successful calibration validation."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "applied"
        sample_calibration_session["points_captured"] = 9
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/validate")

            assert response.status_code == 200
            data = response.json()

            assert "is_valid" in data
            assert "accuracy" in data
            assert "test_results" in data
            assert "errors" in data
            assert "recommendations" in data

    def test_validate_calibration_with_test_points(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test calibration validation with custom test points."""
        session_id = sample_calibration_session["session_id"]
        sample_calibration_session["status"] = "applied"
        sessions = {session_id: sample_calibration_session}

        test_points = [{"screen": [150.0, 150.0], "world": [0.15, 0.15]}]

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(
                f"/api/v1/calibration/{session_id}/validate",
                json={"test_points": test_points},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["test_results"]) > 0

    def test_validate_calibration_not_ready(
        self, client, mock_core_module, mock_user, sample_calibration_session
    ):
        """Test validating calibration when not ready."""
        session_id = sample_calibration_session["session_id"]
        # Keep status as "in_progress"
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.get_core_module",
                return_value=mock_core_module,
            ),
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.post(f"/api/v1/calibration/{session_id}/validate")

            assert response.status_code == 400
            data = response.json()
            assert "Calibration Not Ready for Validation" in data["message"]

    def test_get_calibration_session_success(
        self, client, mock_user, sample_calibration_session
    ):
        """Test successful calibration session retrieval."""
        session_id = sample_calibration_session["session_id"]
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.get(f"/api/v1/calibration/{session_id}")

            assert response.status_code == 200
            data = response.json()

            assert data["session_id"] == session_id
            assert data["calibration_type"] == "standard"
            assert data["status"] == "in_progress"

    def test_get_calibration_session_not_found(self, client, mock_user):
        """Test retrieving non-existent calibration session."""
        with (
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", {}),
        ):
            response = client.get("/api/v1/calibration/nonexistent")

            assert response.status_code == 404

    def test_list_calibration_sessions_success(
        self, client, mock_user, sample_calibration_session
    ):
        """Test successful calibration sessions listing."""
        sessions = {"session1": sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.get("/api/v1/calibration/")

            assert response.status_code == 200
            data = response.json()

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["session_id"] == sample_calibration_session["session_id"]

    def test_list_calibration_sessions_with_status_filter(
        self, client, mock_user, sample_calibration_session
    ):
        """Test listing calibration sessions with status filter."""
        # Create sessions with different statuses
        session1 = sample_calibration_session.copy()
        session1["status"] = "in_progress"
        session2 = sample_calibration_session.copy()
        session2["status"] = "applied"

        sessions = {"session1": session1, "session2": session2}

        with (
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.get("/api/v1/calibration/?status=applied")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 1
            assert data[0]["status"] == "applied"

    def test_list_calibration_sessions_with_limit(
        self, client, mock_user, sample_calibration_session
    ):
        """Test listing calibration sessions with limit."""
        # Create multiple sessions
        sessions = {f"session{i}": sample_calibration_session.copy() for i in range(10)}

        with (
            patch(
                "backend.api.routes.calibration.OperatorRequired",
                return_value=mock_user,
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.get("/api/v1/calibration/?limit=5")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 5

    def test_delete_calibration_session_success(
        self, client, mock_user, sample_calibration_session
    ):
        """Test successful calibration session deletion."""
        session_id = sample_calibration_session["session_id"]
        sessions = {session_id: sample_calibration_session}

        with (
            patch(
                "backend.api.routes.calibration.AdminRequired", return_value=mock_user
            ),
            patch("backend.api.routes.calibration._calibration_sessions", sessions),
        ):
            response = client.delete(f"/api/v1/calibration/{session_id}")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert session_id not in sessions

    def test_delete_calibration_session_not_found(self, client, mock_user):
        """Test deleting non-existent calibration session."""
        with (
            patch(
                "backend.api.routes.calibration.AdminRequired", return_value=mock_user
            ),
            patch("backend.api.routes.calibration._calibration_sessions", {}),
        ):
            response = client.delete("/api/v1/calibration/nonexistent")

            assert response.status_code == 404

    def test_calibration_math_calculate_accuracy(self):
        """Test calibration math accuracy calculation."""
        from ..routes.calibration import CalibrationMath

        # Test with empty points
        result = CalibrationMath.calculate_accuracy([])
        assert result["accuracy"] == 0.0

        # Test with valid points
        points = [
            {"expected_x": 0.0, "expected_y": 0.0, "actual_x": 0.1, "actual_y": 0.1}
        ]
        result = CalibrationMath.calculate_accuracy(points)
        assert "accuracy" in result
        assert "max_error" in result
        assert "mean_error" in result

    def test_calibration_math_calculate_homography(self):
        """Test calibration math homography calculation."""
        from ..routes.calibration import CalibrationMath

        src_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        dst_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        result = CalibrationMath.calculate_homography(src_points, dst_points)
        assert result is not None
        assert result.shape == (3, 3)

        # Test with insufficient points
        result = CalibrationMath.calculate_homography([(0, 0)], [(0.0, 0.0)])
        assert result is None

    def test_authentication_required(self, client):
        """Test that calibration endpoints require authentication."""
        # Without proper authentication, these should fail
        response = client.post("/api/v1/calibration/start")
        assert response.status_code in [401, 403]

        response = client.post("/api/v1/calibration/session_id/points")
        assert response.status_code in [401, 403]

        response = client.post("/api/v1/calibration/session_id/apply")
        assert response.status_code in [401, 403]

        response = client.get("/api/v1/calibration/")
        assert response.status_code in [401, 403]
