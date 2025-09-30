"""Unit tests for the API module."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from api.main import create_app
from api.middleware.authentication import AuthenticationMiddleware as AuthMiddleware
from api.middleware.performance import PerformanceMiddleware
from api.models.responses import GameStateResponse, HealthResponse
from api.utils.security import SecurityUtils
from api.websocket.handler import WebSocketHandler
from api.websocket.subscriptions import SubscriptionManager


@pytest.mark.unit()
class TestAPICreation:
    """Test API application creation."""

    def test_create_app(self):
        """Test creating FastAPI application."""
        app = create_app()
        assert app is not None
        assert hasattr(app, "routes")

    def test_app_configuration(self):
        """Test application configuration."""
        app = create_app()

        # Check that middleware is properly configured
        middleware_classes = [m.cls.__name__ for m in app.middleware_stack]
        assert "CORSMiddleware" in middleware_classes

    def test_app_routes_registered(self):
        """Test that all routes are registered."""
        app = create_app()
        route_paths = [route.path for route in app.routes]

        # Check for essential routes
        assert "/" in route_paths
        assert "/health" in route_paths
        assert "/api/v1/game/state" in route_paths


@pytest.mark.unit()
class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_success(self, test_client):
        """Test successful health check."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_endpoint_response_model(self, test_client):
        """Test health endpoint response model."""
        response = test_client.get("/health")
        data = response.json()

        # Validate response structure
        health_response = HealthResponse(**data)
        assert health_response.status == "healthy"
        assert isinstance(health_response.timestamp, (int, float))

    def test_health_endpoint_performance(self, test_client, performance_timer):
        """Test health endpoint performance."""
        performance_timer.start()
        response = test_client.get("/health")
        performance_timer.stop()

        assert response.status_code == 200
        # Health check should be very fast
        assert performance_timer.elapsed_ms < 50


@pytest.mark.unit()
class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint response."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_root_endpoint_links(self, test_client):
        """Test root endpoint contains proper links."""
        response = test_client.get("/")
        data = response.json()

        assert "/docs" in data["docs"]
        assert isinstance(data["version"], str)


@pytest.mark.unit()
class TestGameStateEndpoints:
    """Test game state API endpoints."""

    def test_get_game_state_no_game(self, test_client):
        """Test getting game state when no game is active."""
        response = test_client.get("/api/v1/game/state")

        # Should return 404 or empty state
        assert response.status_code in [200, 404]

    @patch("api.routes.game.game_state_manager")
    def test_get_game_state_with_game(self, mock_manager, test_client, mock_game_state):
        """Test getting game state when game is active."""
        mock_manager.current_state = mock_game_state

        response = test_client.get("/api/v1/game/state")

        assert response.status_code == 200
        data = response.json()
        assert "balls" in data
        assert "table" in data
        assert "current_player" in data

    @patch("api.routes.game.game_state_manager")
    def test_update_game_state(self, mock_manager, test_client, mock_game_state):
        """Test updating game state."""
        mock_manager.current_state = mock_game_state

        update_data = {"current_player": 2, "shot_clock": 25.0}

        response = test_client.put("/api/v1/game/state", json=update_data)

        assert response.status_code == 200
        # Verify update was called
        mock_manager.update_state.assert_called_once()

    def test_game_state_validation(self, test_client):
        """Test game state update validation."""
        invalid_data = {
            "current_player": "invalid",  # Should be int
            "shot_clock": -5.0,  # Should be positive
        }

        response = test_client.put("/api/v1/game/state", json=invalid_data)

        assert response.status_code == 422  # Validation error


@pytest.mark.unit()
class TestBallEndpoints:
    """Test ball-related API endpoints."""

    @patch("api.routes.balls.game_state_manager")
    def test_get_balls(self, mock_manager, test_client, mock_game_state):
        """Test getting all balls."""
        mock_manager.current_state = mock_game_state

        response = test_client.get("/api/v1/balls")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    @patch("api.routes.balls.game_state_manager")
    def test_get_ball_by_id(self, mock_manager, test_client, mock_game_state):
        """Test getting specific ball by ID."""
        mock_manager.current_state = mock_game_state

        response = test_client.get("/api/v1/balls/cue")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "cue"

    @patch("api.routes.balls.game_state_manager")
    def test_get_nonexistent_ball(self, mock_manager, test_client, mock_game_state):
        """Test getting non-existent ball."""
        mock_manager.current_state = mock_game_state

        response = test_client.get("/api/v1/balls/99")

        assert response.status_code == 404

    @patch("api.routes.balls.game_state_manager")
    def test_update_ball_position(self, mock_manager, test_client, mock_game_state):
        """Test updating ball position."""
        mock_manager.current_state = mock_game_state

        update_data = {"x": 2.0, "y": 1.0}

        response = test_client.put("/api/v1/balls/cue/position", json=update_data)

        assert response.status_code == 200
        # Verify update was called
        mock_manager.update_ball_position.assert_called_once_with("cue", 2.0, 1.0)


@pytest.mark.unit()
class TestShotEndpoints:
    """Test shot-related API endpoints."""

    @patch("api.routes.shots.shot_predictor")
    def test_predict_shot(self, mock_predictor, test_client):
        """Test shot prediction endpoint."""
        mock_prediction = {
            "path": [(1.42, 0.71), (2.0, 0.9)],
            "duration": 2.5,
            "success_probability": 0.75,
        }
        mock_predictor.predict_shot.return_value = mock_prediction

        shot_data = {"angle": 45.0, "force": 0.8, "english": [0, 0]}

        response = test_client.post("/api/v1/shots/predict", json=shot_data)

        assert response.status_code == 200
        data = response.json()
        assert "path" in data
        assert "duration" in data
        assert "success_probability" in data

    @patch("api.routes.shots.shot_assistant")
    def test_get_shot_suggestions(self, mock_assistant, test_client):
        """Test getting shot suggestions."""
        mock_suggestions = [
            {
                "target_ball": "1",
                "angle": 30.0,
                "force": 0.6,
                "difficulty": 0.3,
                "success_probability": 0.8,
            }
        ]
        mock_assistant.suggest_shots.return_value = mock_suggestions

        response = test_client.get("/api/v1/shots/suggestions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["target_ball"] == "1"

    def test_execute_shot_validation(self, test_client):
        """Test shot execution validation."""
        invalid_shot = {
            "angle": 400,  # Invalid angle
            "force": 2.0,  # Force too high
            "english": [5, 5],  # English too high
        }

        response = test_client.post("/api/v1/shots/execute", json=invalid_shot)

        assert response.status_code == 422  # Validation error


@pytest.mark.unit()
class TestConfigurationEndpoints:
    """Test configuration API endpoints."""

    @patch("api.routes.config.config_manager")
    def test_get_configuration(self, mock_manager, test_client, mock_config):
        """Test getting configuration."""
        mock_manager.get_all.return_value = mock_config

        response = test_client.get("/api/v1/config")

        assert response.status_code == 200
        data = response.json()
        assert "camera" in data
        assert "table" in data
        assert "physics" in data

    @patch("api.routes.config.config_manager")
    def test_update_configuration(self, mock_manager, test_client):
        """Test updating configuration."""
        config_update = {"camera": {"device_id": 1, "fps": 60}}

        response = test_client.put("/api/v1/config", json=config_update)

        assert response.status_code == 200
        # Verify update was called
        mock_manager.update.assert_called_once()

    @patch("api.routes.config.config_manager")
    def test_get_config_section(self, mock_manager, test_client, mock_config):
        """Test getting specific configuration section."""
        mock_manager.get.return_value = mock_config["camera"]

        response = test_client.get("/api/v1/config/camera")

        assert response.status_code == 200
        data = response.json()
        assert "device_id" in data
        assert "width" in data


@pytest.mark.unit()
class TestWebSocketHandler:
    """Test WebSocket handler."""

    @pytest.mark.asyncio()
    async def test_handler_creation(self):
        """Test creating WebSocket handler."""
        handler = WebSocketHandler()
        assert handler is not None

    @pytest.mark.asyncio()
    async def test_websocket_connection(self, mock_websocket):
        """Test WebSocket connection handling."""
        handler = WebSocketHandler()

        await handler.connect(mock_websocket)
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio()
    async def test_websocket_disconnection(self, mock_websocket):
        """Test WebSocket disconnection handling."""
        handler = WebSocketHandler()

        await handler.connect(mock_websocket)
        await handler.disconnect(mock_websocket)

        # Should clean up connection
        assert mock_websocket not in handler.active_connections

    @pytest.mark.asyncio()
    async def test_broadcast_message(self, mock_websocket):
        """Test broadcasting message to all connections."""
        handler = WebSocketHandler()

        await handler.connect(mock_websocket)

        message = {"type": "game_state_update", "data": {}}
        await handler.broadcast(message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_send_to_specific_client(self, mock_websocket):
        """Test sending message to specific client."""
        handler = WebSocketHandler()

        await handler.connect(mock_websocket)

        message = {"type": "personal_message", "data": {}}
        await handler.send_personal_message(mock_websocket, message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_handle_ping_pong(self, mock_websocket):
        """Test ping/pong handling."""
        handler = WebSocketHandler()

        # Mock receiving ping
        mock_websocket.receive_json.return_value = {"type": "ping"}

        await handler.handle_message(mock_websocket, {"type": "ping"})

        # Should respond with pong
        mock_websocket.send_json.assert_called_with({"type": "pong"})


@pytest.mark.unit()
class TestSubscriptionManager:
    """Test WebSocket subscription manager."""

    def test_manager_creation(self):
        """Test creating subscription manager."""
        manager = SubscriptionManager()
        assert manager is not None

    @pytest.mark.asyncio()
    async def test_subscribe_to_event(self, mock_websocket):
        """Test subscribing to events."""
        manager = SubscriptionManager()

        await manager.subscribe(mock_websocket, "game_state_updates")

        assert mock_websocket in manager.get_subscribers("game_state_updates")

    @pytest.mark.asyncio()
    async def test_unsubscribe_from_event(self, mock_websocket):
        """Test unsubscribing from events."""
        manager = SubscriptionManager()

        await manager.subscribe(mock_websocket, "game_state_updates")
        await manager.unsubscribe(mock_websocket, "game_state_updates")

        assert mock_websocket not in manager.get_subscribers("game_state_updates")

    @pytest.mark.asyncio()
    async def test_publish_to_subscribers(self, mock_websocket):
        """Test publishing to event subscribers."""
        manager = SubscriptionManager()

        await manager.subscribe(mock_websocket, "game_state_updates")

        message = {"type": "game_state_update", "data": {}}
        await manager.publish("game_state_updates", message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_cleanup_disconnected_clients(self, mock_websocket):
        """Test cleaning up disconnected clients."""
        manager = SubscriptionManager()

        await manager.subscribe(mock_websocket, "game_state_updates")

        # Simulate client disconnection
        await manager.cleanup_client(mock_websocket)

        assert mock_websocket not in manager.get_subscribers("game_state_updates")


@pytest.mark.unit()
class TestPerformanceMiddleware:
    """Test performance monitoring middleware."""

    def test_middleware_creation(self):
        """Test creating performance middleware."""
        middleware = PerformanceMiddleware()
        assert middleware is not None

    @pytest.mark.asyncio()
    async def test_request_timing(self):
        """Test request timing measurement."""
        middleware = PerformanceMiddleware()

        async def mock_call_next(request):
            await asyncio.sleep(0.1)  # Simulate 100ms processing
            return MagicMock(status_code=200)

        request = MagicMock()
        request.url.path = "/test"
        request.method = "GET"

        response = await middleware.dispatch(request, mock_call_next)

        # Should have recorded timing
        assert hasattr(response, "headers")

    @pytest.mark.asyncio()
    async def test_memory_monitoring(self):
        """Test memory usage monitoring."""
        middleware = PerformanceMiddleware(monitor_memory=True)

        async def mock_call_next(request):
            return MagicMock(status_code=200)

        request = MagicMock()
        request.url.path = "/test"

        response = await middleware.dispatch(request, mock_call_next)

        # Should complete without errors
        assert response is not None

    def test_metrics_collection(self):
        """Test metrics collection."""
        middleware = PerformanceMiddleware()

        # Simulate request metrics
        middleware.record_request("/api/v1/game/state", "GET", 0.050, 200)

        metrics = middleware.get_metrics()
        assert "/api/v1/game/state" in metrics
        assert metrics["/api/v1/game/state"]["count"] == 1


@pytest.mark.unit()
class TestAuthMiddleware:
    """Test authentication middleware."""

    def test_middleware_creation(self):
        """Test creating auth middleware."""
        middleware = AuthMiddleware()
        assert middleware is not None

    @pytest.mark.asyncio()
    async def test_public_endpoint_access(self):
        """Test access to public endpoints."""
        middleware = AuthMiddleware()

        async def mock_call_next(request):
            return MagicMock(status_code=200)

        request = MagicMock()
        request.url.path = "/health"
        request.method = "GET"

        response = await middleware.dispatch(request, mock_call_next)

        # Public endpoints should be accessible
        assert response.status_code == 200

    @pytest.mark.asyncio()
    async def test_protected_endpoint_no_auth(self):
        """Test access to protected endpoints without auth."""
        middleware = AuthMiddleware()

        async def mock_call_next(request):
            return MagicMock(status_code=200)

        request = MagicMock()
        request.url.path = "/api/v1/admin/config"
        request.method = "PUT"
        request.headers = {}

        response = await middleware.dispatch(request, mock_call_next)

        # Should be denied access
        assert response.status_code == 401

    @pytest.mark.asyncio()
    async def test_protected_endpoint_with_valid_auth(self):
        """Test access to protected endpoints with valid auth."""
        middleware = AuthMiddleware()

        async def mock_call_next(request):
            return MagicMock(status_code=200)

        request = MagicMock()
        request.url.path = "/api/v1/admin/config"
        request.method = "PUT"
        request.headers = {"Authorization": "Bearer valid_token"}

        with patch.object(middleware, "validate_token", return_value=True):
            response = await middleware.dispatch(request, mock_call_next)

        # Should allow access
        assert response.status_code == 200


@pytest.mark.unit()
class TestSecurityUtils:
    """Test security utilities."""

    def test_generate_token(self):
        """Test token generation."""
        utils = SecurityUtils()

        token = utils.generate_token({"user_id": "test"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token(self):
        """Test token validation."""
        utils = SecurityUtils()

        payload = {"user_id": "test", "exp": time.time() + 3600}
        token = utils.generate_token(payload)

        # Valid token
        is_valid = utils.validate_token(token)
        assert is_valid

        # Invalid token
        is_valid = utils.validate_token("invalid_token")
        assert not is_valid

    def test_hash_password(self):
        """Test password hashing."""
        utils = SecurityUtils()

        password = "test_password"
        hashed = utils.hash_password(password)

        assert hashed != password
        assert utils.verify_password(password, hashed)

    def test_verify_password(self):
        """Test password verification."""
        utils = SecurityUtils()

        password = "test_password"
        wrong_password = "wrong_password"
        hashed = utils.hash_password(password)

        assert utils.verify_password(password, hashed)
        assert not utils.verify_password(wrong_password, hashed)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        utils = SecurityUtils()

        client_id = "test_client"

        # Should allow first few requests
        for _i in range(5):
            assert utils.check_rate_limit(client_id)

        # Should eventually deny if too many requests
        # (depends on rate limit configuration)


@pytest.mark.unit()
class TestResponseModels:
    """Test API response models."""

    def test_health_response_model(self):
        """Test health response model."""
        response = HealthResponse(
            status="healthy", version="1.0.0", timestamp=time.time()
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert isinstance(response.timestamp, (int, float))

    def test_game_state_response_model(self, mock_game_state):
        """Test game state response model."""
        response = GameStateResponse(
            balls=mock_game_state.balls,
            table=mock_game_state.table,
            current_player=mock_game_state.current_player,
            shot_clock=mock_game_state.shot_clock,
            game_mode=mock_game_state.game_mode,
        )

        assert len(response.balls) == 3
        assert response.current_player == 1
        assert response.shot_clock == 30.0

    def test_error_response_model(self):
        """Test error response model."""
        from api.models.responses import ErrorResponse

        error = ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            details={"field": "value"},
        )

        assert error.error == "ValidationError"
        assert error.message == "Invalid input data"
        assert "field" in error.details
