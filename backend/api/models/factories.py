"""Model Factory Functions for Testing.

This module provides factory functions to create test data for all API models.
These factories are useful for:
- Unit testing
- Integration testing
- API documentation examples
- Development and debugging

All factories can generate both valid and invalid data for comprehensive testing.
"""

import base64
import json
import random
import string
from datetime import datetime, timedelta
from typing import Any, Optional

from .common import (
    BoundingBox,
    ColorProfile,
    Coordinate2D,
    PaginationRequest,
    PaginationResponse,
    TimeRange,
    ValidationResult,
)
from .requests import (
    BallPositionUpdateRequest,
    CalibrationPointRequest,
    CalibrationStartRequest,
    ConfigUpdateRequest,
    GameStateResetRequest,
    LoginRequest,
    TokenRefreshRequest,
    WebSocketSubscribeRequest,
)
from .responses import (
    BallInfo,
    CueInfo,
    ErrorCode,
    ErrorResponse,
    GameStateResponse,
    HealthResponse,
    HealthStatus,
    SuccessResponse,
    TableInfo,
)
from .websocket import (
    AlertData,
    AlertLevel,
    AlertMessage,
    BallStateData,
    CueStateData,
    ErrorMessage,
    FrameData,
    FrameMessage,
    FrameMetadata,
    GameStateData,
    StateMessage,
    StreamQuality,
    TableStateData,
)

# =============================================================================
# Utility Functions
# =============================================================================


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_id() -> str:
    """Generate a random ID string."""
    return f"{generate_random_string(8)}_{random.randint(1000, 9999)}"


def generate_timestamp(
    days_ago: int = 0, hours_ago: int = 0, minutes_ago: int = 0
) -> datetime:
    """Generate a timestamp with optional offset."""
    now = datetime.now()
    offset = timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
    return now - offset


def generate_coordinate(
    x_range: tuple = (-10.0, 10.0), y_range: tuple = (-10.0, 10.0)
) -> Coordinate2D:
    """Generate a random coordinate within specified ranges."""
    return Coordinate2D(x=random.uniform(*x_range), y=random.uniform(*y_range))


def generate_table_coordinate() -> list[float]:
    """Generate coordinates within table bounds."""
    return [random.uniform(0.0, 2.84), random.uniform(0.0, 1.42)]  # Standard 9ft table


def generate_base64_image(width: int = 640, height: int = 480) -> str:
    """Generate a base64 encoded dummy image."""
    # Create a simple dummy image data (just random bytes)
    size = width * height * 3  # RGB
    image_bytes = bytes([random.randint(0, 255) for _ in range(size)])
    return base64.b64encode(image_bytes).decode("utf-8")


# =============================================================================
# Common Model Factories
# =============================================================================


def create_coordinate2d(
    x: Optional[float] = None, y: Optional[float] = None
) -> Coordinate2D:
    """Create a Coordinate2D with optional specific values."""
    return Coordinate2D(
        x=x if x is not None else random.uniform(-10.0, 10.0),
        y=y if y is not None else random.uniform(-10.0, 10.0),
    )


def create_bounding_box(
    x: Optional[float] = None,
    y: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> BoundingBox:
    """Create a BoundingBox with optional specific values."""
    return BoundingBox(
        x=x if x is not None else random.uniform(0.0, 100.0),
        y=y if y is not None else random.uniform(0.0, 100.0),
        width=width if width is not None else random.uniform(10.0, 50.0),
        height=height if height is not None else random.uniform(10.0, 50.0),
    )


def create_color_profile(name: Optional[str] = None) -> ColorProfile:
    """Create a ColorProfile with optional specific name."""
    return ColorProfile(
        hue_range=(random.randint(0, 89), random.randint(90, 179)),
        saturation_range=(random.randint(0, 127), random.randint(128, 255)),
        value_range=(random.randint(0, 127), random.randint(128, 255)),
    )


def create_time_range(duration_hours: int = 1) -> TimeRange:
    """Create a TimeRange with specified duration."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=duration_hours)
    return TimeRange(start_time=start_time, end_time=end_time)


def create_pagination_request(page: int = 1, size: int = 20) -> PaginationRequest:
    """Create a PaginationRequest with specified values."""
    return PaginationRequest(
        page=page,
        size=size,
        sort_by=random.choice(["timestamp", "name", "id", None]),
        sort_order=random.choice(["asc", "desc"]),
    )


def create_pagination_response(
    page: int = 1, size: int = 20, total_items: int = 100
) -> PaginationResponse:
    """Create a PaginationResponse with specified values."""
    return PaginationResponse.create(page, size, total_items)


def create_validation_result(
    is_valid: bool = True, error_count: int = 0, warning_count: int = 0
) -> ValidationResult:
    """Create a ValidationResult with specified error/warning counts."""
    result = ValidationResult(is_valid=is_valid)

    for i in range(error_count):
        result.add_error(f"Test error {i + 1}")

    for i in range(warning_count):
        result.add_warning(f"Test warning {i + 1}")

    return result


# =============================================================================
# Request Model Factories
# =============================================================================


def create_login_request(
    username: str = "testuser", password: str = "testpass123", remember_me: bool = False
) -> LoginRequest:
    """Create a LoginRequest with specified or default values."""
    return LoginRequest(username=username, password=password, remember_me=remember_me)


def create_token_refresh_request(token: Optional[str] = None) -> TokenRefreshRequest:
    """Create a TokenRefreshRequest with specified or generated token."""
    return TokenRefreshRequest(
        refresh_token=token or f"refresh_token_{generate_random_string(32)}"
    )


def create_config_update_request(
    section: Optional[str] = None, config_data: Optional[dict[str, Any]] = None
) -> ConfigUpdateRequest:
    """Create a ConfigUpdateRequest with test data."""
    if config_data is None:
        config_data = {
            "camera": {"resolution": [1920, 1080], "fps": 30, "exposure": "auto"},
            "vision": {"detection_threshold": 0.8, "tracking_enabled": True},
        }

    return ConfigUpdateRequest(
        config_section=section,
        config_data=config_data,
        validate_only=False,
        force_update=False,
        client_timestamp=datetime.now(),
    )


def create_calibration_start_request(
    calibration_type: str = "standard",
) -> CalibrationStartRequest:
    """Create a CalibrationStartRequest with specified type."""
    return CalibrationStartRequest(
        calibration_type=calibration_type,
        force_restart=False,
        timeout_seconds=300,
        client_timestamp=datetime.now(),
    )


def create_calibration_point_request(
    session_id: Optional[str] = None, point_id: Optional[str] = None
) -> CalibrationPointRequest:
    """Create a CalibrationPointRequest with test data."""
    return CalibrationPointRequest(
        session_id=session_id or f"cal_session_{generate_random_string(8)}",
        point_id=point_id or f"point_{random.randint(1, 10)}",
        screen_position=[random.uniform(0, 1920), random.uniform(0, 1080)],
        world_position=[random.uniform(0, 2.84), random.uniform(0, 1.42)],
        confidence=random.uniform(0.8, 1.0),
        client_timestamp=datetime.now(),
    )


def create_game_state_reset_request(
    game_type: str = "practice",
) -> GameStateResetRequest:
    """Create a GameStateResetRequest with specified game type."""
    return GameStateResetRequest(
        game_type=game_type,
        preserve_table=True,
        custom_setup=None,
        client_timestamp=datetime.now(),
    )


def create_ball_position_update_request(
    ball_count: int = 3,
) -> BallPositionUpdateRequest:
    """Create a BallPositionUpdateRequest with specified number of balls."""
    ball_updates = []
    for i in range(ball_count):
        ball_updates.append(
            {
                "id": f"ball_{i + 1}",
                "position": generate_table_coordinate(),
                "velocity": [0.0, 0.0],
                "is_pocketed": False,
            }
        )

    return BallPositionUpdateRequest(
        ball_updates=ball_updates,
        validate_positions=True,
        check_collisions=True,
        client_timestamp=datetime.now(),
    )


def create_websocket_subscribe_request(
    streams: Optional[list[str]] = None,
) -> WebSocketSubscribeRequest:
    """Create a WebSocketSubscribeRequest with specified streams."""
    if streams is None:
        streams = ["frames", "state", "trajectories"]

    return WebSocketSubscribeRequest(
        streams=streams, quality="high", frame_rate=30, filters={}
    )


# =============================================================================
# Response Model Factories
# =============================================================================


def create_health_response(
    status: HealthStatus = HealthStatus.HEALTHY,
) -> HealthResponse:
    """Create a HealthResponse with specified status."""
    from .responses import ComponentHealth, SystemMetrics

    components = {
        "vision": ComponentHealth(
            name="vision",
            status=status,
            message="Vision processing operational",
            last_check=datetime.now(),
            uptime=random.uniform(3600, 86400),
            errors=[],
        ),
        "camera": ComponentHealth(
            name="camera",
            status=status,
            message="Camera connected and streaming",
            last_check=datetime.now(),
            uptime=random.uniform(3600, 86400),
            errors=[],
        ),
    }

    metrics = SystemMetrics(
        cpu_usage=random.uniform(10, 80),
        memory_usage=random.uniform(20, 90),
        disk_usage=random.uniform(30, 70),
        network_io={"bytes_sent": 1024000, "bytes_received": 2048000},
        api_requests_per_second=random.uniform(10, 100),
        websocket_connections=random.randint(1, 50),
        average_response_time=random.uniform(10, 100),
    )

    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        uptime=random.uniform(3600, 86400),
        version="1.0.0",
        components=components,
        metrics=metrics,
    )


def create_ball_info(
    ball_id: Optional[str] = None, is_cue_ball: bool = False, is_pocketed: bool = False
) -> BallInfo:
    """Create a BallInfo with specified properties."""
    return BallInfo(
        id=ball_id or f"ball_{random.randint(1, 16)}",
        number=None if is_cue_ball else random.randint(1, 15),
        position=generate_table_coordinate(),
        velocity=[random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)],
        is_cue_ball=is_cue_ball,
        is_pocketed=is_pocketed,
        confidence=random.uniform(0.8, 1.0),
        last_update=datetime.now(),
    )


def create_cue_info(is_visible: bool = True) -> CueInfo:
    """Create a CueInfo with specified visibility."""
    return CueInfo(
        tip_position=generate_table_coordinate(),
        angle=random.uniform(-180, 180),
        elevation=random.uniform(-30, 30),
        estimated_force=random.uniform(0, 50),
        is_visible=is_visible,
        confidence=random.uniform(0.7, 1.0) if is_visible else 0.0,
    )


def create_table_info() -> TableInfo:
    """Create a standard TableInfo."""
    # Standard 9-foot table
    width, height = 2.84, 1.42
    pocket_positions = [
        [0, 0],
        [width / 2, 0],
        [width, 0],  # Bottom pockets
        [0, height],
        [width / 2, height],
        [width, height],  # Top pockets
    ]

    return TableInfo(
        width=width,
        height=height,
        pocket_positions=pocket_positions,
        pocket_radius=0.0635,
        surface_friction=0.2,
    )


def create_game_state_response(ball_count: int = 16) -> GameStateResponse:
    """Create a GameStateResponse with specified number of balls."""
    balls = []

    # Create cue ball
    balls.append(create_ball_info("cue", is_cue_ball=True))

    # Create numbered balls
    for i in range(1, ball_count):
        balls.append(create_ball_info(f"ball_{i}"))

    return GameStateResponse(
        timestamp=datetime.now(),
        frame_number=random.randint(1000, 10000),
        balls=balls,
        cue=create_cue_info(),
        table=create_table_info(),
        game_type="practice",
        is_valid=True,
        confidence=random.uniform(0.8, 1.0),
        events=[],
    )


def create_error_response(
    error_code: ErrorCode = ErrorCode.AUTH_INVALID_CREDENTIALS,
    message: Optional[str] = None,
) -> ErrorResponse:
    """Create an ErrorResponse with specified error code."""
    return ErrorResponse(
        error=error_code.name.lower().replace("_", " "),
        message=message or f"Test error: {error_code.value}",
        code=error_code.value,
        details={"test": True},
        timestamp=datetime.now(),
        request_id=generate_random_id(),
    )


def create_success_response(message: str = "Operation successful") -> SuccessResponse:
    """Create a SuccessResponse with specified message."""
    return SuccessResponse(
        success=True, message=message, timestamp=datetime.now(), data={"test": True}
    )


# =============================================================================
# WebSocket Message Factories
# =============================================================================


def create_frame_message(
    width: int = 640, height: int = 480, frame_number: Optional[int] = None
) -> FrameMessage:
    """Create a FrameMessage with specified dimensions."""
    metadata = FrameMetadata(
        width=width,
        height=height,
        fps=30.0,
        quality=StreamQuality.HIGH,
        format="jpeg",
        compression_level=0.8,
    )

    frame_data = FrameData(
        image_data=generate_base64_image(width, height), metadata=metadata
    )

    return FrameMessage(
        timestamp=datetime.now(),
        sequence=random.randint(1, 10000),
        frame_number=frame_number or random.randint(1, 10000),
        data=frame_data,
        processing_time_ms=random.uniform(5.0, 50.0),
    )


def create_ball_state_data(
    ball_id: Optional[str] = None, is_cue_ball: bool = False
) -> BallStateData:
    """Create a BallStateData with specified properties."""
    return BallStateData(
        id=ball_id or f"ball_{random.randint(1, 16)}",
        number=None if is_cue_ball else random.randint(1, 15),
        position=generate_table_coordinate(),
        velocity=[random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)],
        radius=0.028575,
        is_cue_ball=is_cue_ball,
        is_pocketed=False,
        confidence=random.uniform(0.8, 1.0),
    )


def create_game_state_data(ball_count: int = 16) -> GameStateData:
    """Create a GameStateData with specified number of balls."""
    balls = []

    # Create cue ball
    balls.append(create_ball_state_data("cue", is_cue_ball=True))

    # Create numbered balls
    for i in range(1, ball_count):
        balls.append(create_ball_state_data(f"ball_{i}"))

    cue_data = CueStateData(
        tip_position=generate_table_coordinate(),
        angle=random.uniform(-180, 180),
        elevation=random.uniform(-30, 30),
        estimated_force=random.uniform(0, 50),
        is_visible=True,
        confidence=random.uniform(0.7, 1.0),
    )

    table_data = TableStateData(
        width=2.84,
        height=1.42,
        pocket_positions=[
            [0, 0],
            [1.42, 0],
            [2.84, 0],
            [0, 1.42],
            [1.42, 1.42],
            [2.84, 1.42],
        ],
        pocket_radius=0.0635,
    )

    return GameStateData(
        frame_number=random.randint(1000, 10000),
        balls=balls,
        cue=cue_data,
        table=table_data,
        game_type="practice",
        is_valid=True,
        confidence=random.uniform(0.8, 1.0),
    )


def create_state_message(ball_count: int = 16) -> StateMessage:
    """Create a StateMessage with specified number of balls."""
    return StateMessage(
        timestamp=datetime.now(),
        sequence=random.randint(1, 10000),
        data=create_game_state_data(ball_count),
        changes=["balls", "cue"],
        processing_time_ms=random.uniform(5.0, 50.0),
    )


def create_alert_message(
    level: AlertLevel = AlertLevel.INFO, code: Optional[str] = None
) -> AlertMessage:
    """Create an AlertMessage with specified level."""
    alert_data = AlertData(
        level=level,
        message=f"Test {level.value} alert message",
        code=code or f"TEST_{random.randint(100, 999)}",
        component="test_component",
        details={"test": True},
        auto_dismiss=level in [AlertLevel.INFO, AlertLevel.WARNING],
        dismiss_timeout=30 if level == AlertLevel.WARNING else None,
        actions=["dismiss", "details"],
    )

    return AlertMessage(
        timestamp=datetime.now(), sequence=random.randint(1, 10000), data=alert_data
    )


def create_error_message(
    error_code: str = "TEST_001", recoverable: bool = True
) -> ErrorMessage:
    """Create an ErrorMessage with specified properties."""
    from .websocket import ErrorData

    error_data = ErrorData(
        error_code=error_code,
        message=f"Test error message for {error_code}",
        details={"test": True},
        recoverable=recoverable,
    )

    return ErrorMessage(
        timestamp=datetime.now(), sequence=random.randint(1, 10000), data=error_data
    )


# =============================================================================
# Batch Factory Functions
# =============================================================================


def create_multiple_ball_info(count: int = 16) -> list[BallInfo]:
    """Create multiple BallInfo objects."""
    balls = [create_ball_info("cue", is_cue_ball=True)]
    for i in range(1, count):
        balls.append(create_ball_info(f"ball_{i}"))
    return balls


def create_test_dataset(
    ball_count: int = 16, frame_count: int = 10, include_errors: bool = True
) -> dict[str, Any]:
    """Create a comprehensive test dataset."""
    dataset = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "ball_count": ball_count,
            "frame_count": frame_count,
            "version": "1.0.0",
        },
        "requests": {
            "login": create_login_request(),
            "config_update": create_config_update_request(),
            "calibration_start": create_calibration_start_request(),
            "game_reset": create_game_state_reset_request(),
        },
        "responses": {
            "health": create_health_response(),
            "game_state": create_game_state_response(ball_count),
            "success": create_success_response(),
        },
        "websocket_messages": {
            "frame": create_frame_message(),
            "state": create_state_message(ball_count),
            "alert": create_alert_message(),
        },
        "common_models": {
            "coordinate": create_coordinate2d(),
            "bounding_box": create_bounding_box(),
            "time_range": create_time_range(),
            "pagination": create_pagination_request(),
        },
    }

    if include_errors:
        dataset["errors"] = {
            "response": create_error_response(),
            "websocket": create_error_message(),
            "validation": create_validation_result(False, 2, 1),
        }

    return dataset


def export_test_data_as_json(
    dataset: dict[str, Any], filename: str = "test_data.json"
) -> str:
    """Export test dataset as JSON file."""

    def json_serializer(obj):
        """Custom JSON serializer for datetime and other objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)

    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2, default=json_serializer)

    return filename
