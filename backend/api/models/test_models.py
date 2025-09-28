"""Comprehensive Model Validation Tests.

This module contains comprehensive tests for all API models including:
- Request model validation
- Response model validation
- WebSocket message validation
- Model conversion testing
- Error handling validation
- Performance testing

All tests use pytest and include both positive and negative test cases.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from .common import (
    Coordinate2D,
    PaginationRequest,
    ValidationResult,
    validate_coordinate_bounds,
)
from .converters import validate_ball_state_conversion, vector2d_to_coordinate2d
from .factories import (
    create_ball_info,
    create_config_update_request,
    create_frame_message,
    create_game_state_response,
    create_login_request,
    create_test_dataset,
)
from .requests import (
    CalibrationPointRequest,
    ConfigUpdateRequest,
    GameStateResetRequest,
    LoginRequest,
)
from .responses import (
    BallInfo,
    ConfigResponse,
    ErrorCode,
    ErrorResponse,
    GameStateResponse,
    HealthResponse,
    HealthStatus,
)
from .websocket import (
    AlertLevel,
    AlertMessage,
    BallStateData,
    FrameMessage,
    GameStateData,
    MessageType,
    StateMessage,
    validate_websocket_message,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture()
def valid_login_data():
    """Valid login request data."""
    return {"username": "testuser", "password": "testpass123", "remember_me": False}


@pytest.fixture()
def invalid_login_data():
    """Invalid login request data."""
    return {
        "username": "u",  # Too short
        "password": "short",  # Too short
        "remember_me": "not_boolean",  # Wrong type
    }


@pytest.fixture()
def valid_config_data():
    """Valid configuration data."""
    return {
        "camera": {"resolution": [1920, 1080], "fps": 30, "exposure": "auto"},
        "vision": {"detection_threshold": 0.8, "tracking_enabled": True},
    }


@pytest.fixture()
def valid_ball_data():
    """Valid ball state data."""
    return {
        "id": "ball_1",
        "number": 1,
        "position": [1.0, 0.5],
        "velocity": [0.1, -0.2],
        "radius": 0.028575,
        "is_cue_ball": False,
        "is_pocketed": False,
        "confidence": 0.95,
    }


@pytest.fixture()
def valid_websocket_frame_data():
    """Valid WebSocket frame message data."""
    return {
        "type": "frame",
        "timestamp": datetime.now().isoformat(),
        "sequence": 1001,
        "frame_number": 12345,
        "data": {
            "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "metadata": {
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "quality": "high",
                "format": "jpeg",
            },
        },
    }


# =============================================================================
# Request Model Tests
# =============================================================================


class TestLoginRequest:
    """Test LoginRequest model validation."""

    def test_valid_login_request(self, valid_login_data):
        """Test creating a valid login request."""
        request = LoginRequest(**valid_login_data)
        assert request.username == "testuser"
        assert request.password.get_secret_value() == "testpass123"
        assert request.remember_me is False

    def test_username_too_short(self):
        """Test username length validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoginRequest(username="ab", password="validpass123")

        errors = exc_info.value.errors()
        assert any("at least 3 characters" in str(error) for error in errors)

    def test_username_invalid_characters(self):
        """Test username character validation."""
        with pytest.raises(ValidationError):
            LoginRequest(username="user@domain", password="validpass123")

    def test_password_too_short(self):
        """Test password length validation."""
        with pytest.raises(ValidationError) as exc_info:
            LoginRequest(username="validuser", password="short")

        errors = exc_info.value.errors()
        assert any("at least 8 characters" in str(error) for error in errors)

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        with pytest.raises(ValidationError):
            LoginRequest(username="testuser")  # Missing password

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            LoginRequest(
                username="testuser", password="testpass123", extra_field="not_allowed"
            )


class TestConfigUpdateRequest:
    """Test ConfigUpdateRequest model validation."""

    def test_valid_config_update(self, valid_config_data):
        """Test creating a valid config update request."""
        request = ConfigUpdateRequest(config_data=valid_config_data, validate_only=True)
        assert request.config_data == valid_config_data
        assert request.validate_only is True

    def test_empty_config_data(self):
        """Test validation with empty config data."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigUpdateRequest(config_data={})

        errors = exc_info.value.errors()
        assert any("cannot be empty" in str(error) for error in errors)

    def test_config_section_specified(self, valid_config_data):
        """Test config update with specific section."""
        request = ConfigUpdateRequest(
            config_section="camera", config_data=valid_config_data["camera"]
        )
        assert request.config_section == "camera"
        assert request.config_data == valid_config_data["camera"]


class TestCalibrationPointRequest:
    """Test CalibrationPointRequest model validation."""

    def test_valid_calibration_point(self):
        """Test creating a valid calibration point request."""
        request = CalibrationPointRequest(
            session_id="cal_session_123",
            point_id="corner_1",
            screen_position=[100.0, 200.0],
            world_position=[0.1, 0.2],
            confidence=0.95,
        )
        assert request.session_id == "cal_session_123"
        assert request.screen_position == [100.0, 200.0]
        assert request.confidence == 0.95

    def test_invalid_coordinate_length(self):
        """Test validation with incorrect coordinate length."""
        with pytest.raises(ValidationError):
            CalibrationPointRequest(
                session_id="test",
                point_id="test",
                screen_position=[100.0],  # Only one coordinate
                world_position=[0.1, 0.2],
            )

    def test_confidence_out_of_range(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            CalibrationPointRequest(
                session_id="test",
                point_id="test",
                screen_position=[100.0, 200.0],
                world_position=[0.1, 0.2],
                confidence=1.5,  # Invalid confidence
            )


class TestGameStateResetRequest:
    """Test GameStateResetRequest model validation."""

    def test_valid_game_types(self):
        """Test valid game type values."""
        valid_types = ["practice", "8ball", "9ball", "straight"]
        for game_type in valid_types:
            request = GameStateResetRequest(game_type=game_type)
            assert request.game_type == game_type

    def test_invalid_game_type(self):
        """Test invalid game type validation."""
        with pytest.raises(ValidationError):
            GameStateResetRequest(game_type="invalid_type")


# =============================================================================
# Response Model Tests
# =============================================================================


class TestHealthResponse:
    """Test HealthResponse model validation."""

    def test_valid_health_response(self):
        """Test creating a valid health response."""
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            timestamp=datetime.now(),
            uptime=3600.0,
            version="1.0.0",
            components={},
            metrics=None,
        )
        assert response.status == HealthStatus.HEALTHY
        assert response.version == "1.0.0"

    def test_health_status_enum(self):
        """Test health status enum validation."""
        valid_statuses = [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        for status in valid_statuses:
            response = HealthResponse(
                status=status, timestamp=datetime.now(), uptime=3600.0, version="1.0.0"
            )
            assert response.status == status


class TestBallInfo:
    """Test BallInfo model validation."""

    def test_valid_ball_info(self, valid_ball_data):
        """Test creating a valid ball info."""
        ball = BallInfo(**valid_ball_data, last_update=datetime.now())
        assert ball.id == "ball_1"
        assert ball.position == [1.0, 0.5]
        assert ball.confidence == 0.95

    def test_confidence_validation(self, valid_ball_data):
        """Test ball confidence validation."""
        # Test valid confidence
        ball = BallInfo(**valid_ball_data, last_update=datetime.now())
        assert ball.confidence == 0.95

        # Test invalid confidence
        with pytest.raises(ValidationError):
            invalid_data = valid_ball_data.copy()
            invalid_data["confidence"] = 1.5
            BallInfo(**invalid_data, last_update=datetime.now())

    def test_cue_ball_properties(self, valid_ball_data):
        """Test cue ball specific properties."""
        cue_data = valid_ball_data.copy()
        cue_data.update({"id": "cue", "is_cue_ball": True, "number": None})

        ball = BallInfo(**cue_data, last_update=datetime.now())
        assert ball.is_cue_ball is True
        assert ball.number is None


class TestErrorResponse:
    """Test ErrorResponse model validation."""

    def test_valid_error_response(self):
        """Test creating a valid error response."""
        response = ErrorResponse(
            error="validation_error",
            message="Invalid input provided",
            code=ErrorCode.AUTH_INVALID_CREDENTIALS.value,
            timestamp=datetime.now(),
        )
        assert response.error == "validation_error"
        assert response.code == ErrorCode.AUTH_INVALID_CREDENTIALS.value

    def test_error_with_details(self):
        """Test error response with details."""
        details = {"field": "username", "value": "invalid"}
        response = ErrorResponse(
            error="validation_error",
            message="Invalid username",
            code="VAL_001",
            details=details,
            timestamp=datetime.now(),
        )
        assert response.details == details


# =============================================================================
# WebSocket Message Tests
# =============================================================================


class TestFrameMessage:
    """Test FrameMessage model validation."""

    def test_valid_frame_message(self, valid_websocket_frame_data):
        """Test creating a valid frame message."""
        # Convert timestamp string to datetime
        data = valid_websocket_frame_data.copy()
        data["timestamp"] = datetime.now()

        message = FrameMessage(**data)
        assert message.type == MessageType.FRAME
        assert message.frame_number == 12345

    def test_invalid_base64_image(self):
        """Test validation with invalid base64 image data."""
        with pytest.raises(ValidationError):
            from .websocket import FrameData, FrameMetadata

            metadata = FrameMetadata(width=640, height=480, fps=30.0, quality="high")

            FrameData(image_data="invalid_base64_data!@#", metadata=metadata)


class TestStateMessage:
    """Test StateMessage model validation."""

    def test_valid_state_message(self, valid_ball_data):
        """Test creating a valid state message."""
        ball_state = BallStateData(**valid_ball_data)

        game_state = GameStateData(
            frame_number=1000,
            balls=[ball_state],
            table={
                "width": 2.84,
                "height": 1.42,
                "pocket_positions": [[0, 0], [1.42, 0]],
                "pocket_radius": 0.0635,
            },
            game_type="practice",
        )

        message = StateMessage(timestamp=datetime.now(), sequence=1001, data=game_state)
        assert message.type == MessageType.STATE
        assert len(message.data.balls) == 1


class TestAlertMessage:
    """Test AlertMessage model validation."""

    def test_valid_alert_levels(self):
        """Test all valid alert levels."""
        levels = [
            AlertLevel.INFO,
            AlertLevel.WARNING,
            AlertLevel.ERROR,
            AlertLevel.CRITICAL,
        ]

        for level in levels:
            message = AlertMessage(
                timestamp=datetime.now(),
                sequence=1001,
                data={
                    "level": level,
                    "message": f"Test {level.value} message",
                    "code": "TEST_001",
                },
            )
            assert message.data.level == level

    def test_alert_code_format(self):
        """Test alert code format validation."""
        # Valid code format
        AlertMessage(
            timestamp=datetime.now(),
            sequence=1001,
            data={
                "level": AlertLevel.INFO,
                "message": "Test message",
                "code": "TEST_001",
            },
        )

        # Invalid code format should raise validation error
        with pytest.raises(ValidationError):
            AlertMessage(
                timestamp=datetime.now(),
                sequence=1001,
                data={
                    "level": AlertLevel.INFO,
                    "message": "Test message",
                    "code": "invalid_code",
                },
            )


# =============================================================================
# Model Conversion Tests
# =============================================================================


class TestModelConverters:
    """Test model conversion utilities."""

    def test_coordinate_conversion(self):
        """Test coordinate conversions."""
        from ...core.models import Vector2D

        # Test Vector2D to Coordinate2D
        vector = Vector2D(1.5, 2.3)
        coord = vector2d_to_coordinate2d(vector)
        assert coord.x == 1.5
        assert coord.y == 2.3

    def test_ball_state_conversion(self, valid_ball_data):
        """Test ball state conversion validation."""
        result = validate_ball_state_conversion(valid_ball_data)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invalid_ball_state_conversion(self):
        """Test ball state conversion with invalid data."""
        invalid_data = {
            "id": "test",
            "position": [1.0],  # Missing y coordinate
            "confidence": 1.5,  # Invalid confidence
        }

        result = validate_ball_state_conversion(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) > 0


# =============================================================================
# Common Model Tests
# =============================================================================


class TestCoordinate2D:
    """Test Coordinate2D model and utilities."""

    def test_coordinate_creation(self):
        """Test coordinate creation and properties."""
        coord = Coordinate2D(x=1.5, y=2.5)
        assert coord.x == 1.5
        assert coord.y == 2.5

    def test_distance_calculation(self):
        """Test distance calculation between coordinates."""
        coord1 = Coordinate2D(x=0.0, y=0.0)
        coord2 = Coordinate2D(x=3.0, y=4.0)
        distance = coord1.distance_to(coord2)
        assert distance == 5.0  # 3-4-5 triangle

    def test_coordinate_bounds_validation(self):
        """Test coordinate bounds validation."""
        # Valid coordinates
        coords = [1.0, 2.0]
        validated = validate_coordinate_bounds(coords)
        assert validated == coords

        # Invalid coordinates (out of bounds)
        with pytest.raises(ValueError):
            validate_coordinate_bounds([2000.0, 3000.0])

    def test_list_conversion(self):
        """Test conversion to/from list format."""
        coord = Coordinate2D(x=1.5, y=2.5)
        coord_list = coord.to_list()
        assert coord_list == [1.5, 2.5]

        coord_from_list = Coordinate2D.from_list([1.5, 2.5])
        assert coord_from_list.x == 1.5
        assert coord_from_list.y == 2.5


class TestPaginationRequest:
    """Test PaginationRequest model validation."""

    def test_valid_pagination(self):
        """Test valid pagination parameters."""
        request = PaginationRequest(page=2, size=25, sort_by="timestamp")
        assert request.page == 2
        assert request.size == 25
        assert request.offset == 25  # (page - 1) * size

    def test_page_size_limits(self):
        """Test page size validation."""
        # Valid size
        PaginationRequest(page=1, size=100)

        # Invalid size (too large)
        with pytest.raises(ValidationError):
            PaginationRequest(page=1, size=2000)

    def test_invalid_page_number(self):
        """Test page number validation."""
        with pytest.raises(ValidationError):
            PaginationRequest(page=0, size=20)  # Page must be >= 1


class TestValidationResult:
    """Test ValidationResult model functionality."""

    def test_empty_validation_result(self):
        """Test creating an empty validation result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.has_errors is False

    def test_adding_errors(self):
        """Test adding validation errors."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")
        assert result.is_valid is False
        assert result.has_errors is True
        assert len(result.errors) == 1

    def test_adding_warnings(self):
        """Test adding validation warnings."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid is True  # Warnings don't affect validity
        assert result.has_warnings is True
        assert len(result.warnings) == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test model factory functions."""

    def test_login_request_factory(self):
        """Test login request factory."""
        request = create_login_request()
        assert isinstance(request, LoginRequest)
        assert len(request.username) >= 3
        assert len(request.password.get_secret_value()) >= 8

    def test_config_update_factory(self):
        """Test config update request factory."""
        request = create_config_update_request()
        assert isinstance(request, ConfigUpdateRequest)
        assert isinstance(request.config_data, dict)
        assert len(request.config_data) > 0

    def test_game_state_response_factory(self):
        """Test game state response factory."""
        response = create_game_state_response(ball_count=10)
        assert isinstance(response, GameStateResponse)
        assert len(response.balls) == 10

    def test_ball_info_factory(self):
        """Test ball info factory."""
        ball = create_ball_info()
        assert isinstance(ball, BallInfo)
        assert 0.0 <= ball.confidence <= 1.0

    def test_frame_message_factory(self):
        """Test frame message factory."""
        message = create_frame_message()
        assert isinstance(message, FrameMessage)
        assert message.type == MessageType.FRAME

    def test_test_dataset_factory(self):
        """Test comprehensive test dataset factory."""
        dataset = create_test_dataset(ball_count=8, frame_count=5)
        assert isinstance(dataset, dict)
        assert "metadata" in dataset
        assert "requests" in dataset
        assert "responses" in dataset
        assert "websocket_messages" in dataset


# =============================================================================
# WebSocket Message Validation Tests
# =============================================================================


class TestWebSocketValidation:
    """Test WebSocket message validation utilities."""

    def test_validate_valid_message(self, valid_websocket_frame_data):
        """Test validation of valid WebSocket message."""
        # Convert timestamp to datetime object
        data = valid_websocket_frame_data.copy()
        data["timestamp"] = datetime.now()

        message = validate_websocket_message(data)
        assert isinstance(message, FrameMessage)
        assert message.type == MessageType.FRAME

    def test_validate_invalid_message_type(self):
        """Test validation with invalid message type."""
        invalid_data = {
            "type": "invalid_type",
            "timestamp": datetime.now(),
            "sequence": 1001,
        }

        with pytest.raises(ValueError) as exc_info:
            validate_websocket_message(invalid_data)

        assert "Invalid message type" in str(exc_info.value)

    def test_validate_missing_type(self):
        """Test validation with missing message type."""
        invalid_data = {"timestamp": datetime.now(), "sequence": 1001}

        with pytest.raises(ValueError) as exc_info:
            validate_websocket_message(invalid_data)

        assert "must have a 'type' field" in str(exc_info.value)


# =============================================================================
# Performance Tests
# =============================================================================


class TestModelPerformance:
    """Test model performance and memory usage."""

    def test_bulk_model_creation(self):
        """Test creating many models quickly."""
        import time

        start_time = time.time()

        # Create 1000 ball info objects
        balls = []
        for i in range(1000):
            ball = create_ball_info(f"ball_{i}")
            balls.append(ball)

        end_time = time.time()
        creation_time = end_time - start_time

        assert len(balls) == 1000
        assert creation_time < 5.0  # Should create 1000 objects in under 5 seconds

    def test_model_serialization_performance(self):
        """Test model serialization performance."""
        import time

        # Create a complex game state
        game_state = create_game_state_response(ball_count=16)

        start_time = time.time()

        # Serialize 100 times
        for _ in range(100):
            json_data = game_state.model_dump_json()
            assert isinstance(json_data, str)

        end_time = time.time()
        serialization_time = end_time - start_time

        assert serialization_time < 2.0  # Should serialize 100 times in under 2 seconds

    def test_validation_performance(self):
        """Test validation performance."""
        import time

        valid_data = {
            "username": "testuser",
            "password": "testpass123",
            "remember_me": False,
        }

        start_time = time.time()

        # Validate 1000 times
        for _ in range(1000):
            request = LoginRequest(**valid_data)
            assert request.username == "testuser"

        end_time = time.time()
        validation_time = end_time - start_time

        assert validation_time < 3.0  # Should validate 1000 times in under 3 seconds


# =============================================================================
# Integration Tests
# =============================================================================


class TestModelIntegration:
    """Test model integration with other components."""

    def test_request_response_cycle(self):
        """Test complete request-response cycle."""
        # Create a config update request
        request = create_config_update_request()

        # Simulate processing and create response
        response = ConfigResponse(
            timestamp=datetime.now(),
            values=request.config_data,
            schema_version="1.0.0",
            last_modified=datetime.now(),
            is_valid=True,
            validation_errors=[],
        )

        assert response.values == request.config_data
        assert response.is_valid is True

    def test_websocket_state_conversion(self, valid_ball_data):
        """Test converting between API and WebSocket models."""
        # Create API ball info
        ball_info = BallInfo(**valid_ball_data, last_update=datetime.now())

        # Convert to WebSocket format
        ball_state_data = BallStateData(
            id=ball_info.id,
            number=ball_info.number,
            position=ball_info.position,
            velocity=ball_info.velocity,
            is_cue_ball=ball_info.is_cue_ball,
            is_pocketed=ball_info.is_pocketed,
            confidence=ball_info.confidence,
        )

        assert ball_state_data.id == ball_info.id
        assert ball_state_data.position == ball_info.position


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in model validation."""

    def test_multiple_validation_errors(self):
        """Test handling multiple validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            LoginRequest(
                username="a",  # Too short
                password="short",  # Too short
                remember_me="not_bool",  # Wrong type
            )

        errors = exc_info.value.errors()
        assert len(errors) >= 3  # Should have multiple errors

    def test_nested_model_validation_errors(self):
        """Test validation errors in nested models."""
        with pytest.raises(ValidationError):
            ConfigUpdateRequest(
                config_data={
                    "camera": {
                        "resolution": "invalid",  # Should be list
                        "fps": -30,  # Should be positive
                    }
                }
            )

    def test_custom_validation_error_messages(self):
        """Test custom validation error messages."""
        with pytest.raises(ValidationError) as exc_info:
            CalibrationPointRequest(
                session_id="test",
                point_id="test",
                screen_position=[100.0, 200.0, 300.0],  # Too many coordinates
                world_position=[0.1, 0.2],
            )

        errors = exc_info.value.errors()
        # Should have a meaningful error message about coordinate length
        assert any("2" in str(error) for error in errors)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
