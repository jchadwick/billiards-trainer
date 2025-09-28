"""API Request Models.

This module defines all Pydantic models for API request validation, including:
- Authentication requests
- Configuration update requests
- Calibration operation requests
- Game state manipulation requests

All models include comprehensive validation, documentation, and OpenAPI schema support.
"""

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

# =============================================================================
# Base Request Models
# =============================================================================


class BaseRequest(BaseModel):
    """Base class for all API requests with common validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class TimestampedRequest(BaseRequest):
    """Base request with client timestamp for synchronization."""

    client_timestamp: Optional[datetime] = Field(
        default=None,
        description="Client-side timestamp for request synchronization",
        examples=["2024-01-15T10:30:00Z"],
    )


# =============================================================================
# Authentication Request Models
# =============================================================================


class LoginRequest(BaseRequest):
    """User login request."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Username for authentication",
        examples=["admin"],
    )
    password: SecretStr = Field(
        ..., min_length=8, description="User password", examples=["secure_password123"]
    )
    remember_me: bool = Field(
        default=False, description="Whether to extend session duration"
    )


class TokenRefreshRequest(BaseRequest):
    """Token refresh request."""

    refresh_token: str = Field(
        ...,
        description="Refresh token for generating new access token",
        examples=["eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."],
    )


class ChangePasswordRequest(BaseRequest):
    """Password change request."""

    current_password: SecretStr = Field(
        ..., description="Current password for verification"
    )
    new_password: SecretStr = Field(..., min_length=8, description="New password")
    confirm_password: SecretStr = Field(..., description="Confirmation of new password")

    @model_validator(mode="after")
    def validate_password_match(self):
        """Validate that new password and confirmation match."""
        if (
            self.new_password.get_secret_value()
            != self.confirm_password.get_secret_value()
        ):
            raise ValueError("New password and confirmation do not match")
        return self


class CreateUserRequest(BaseRequest):
    """User creation request."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique username",
        examples=["operator1"],
    )
    email: Optional[EmailStr] = Field(
        default=None,
        description="User email address",
        examples=["operator@example.com"],
    )
    password: SecretStr = Field(..., min_length=8, description="Initial password")
    role: str = Field(
        default="viewer",
        pattern=r"^(admin|operator|viewer)$",
        description="User role",
        examples=["operator"],
    )
    enabled: bool = Field(default=True, description="Whether user account is enabled")


# =============================================================================
# Health Check Request Models
# =============================================================================


class HealthCheckRequest(BaseRequest):
    """Request model for health check endpoint."""

    include_details: bool = Field(
        False, description="Include detailed component health"
    )
    include_metrics: bool = Field(False, description="Include performance metrics")


# =============================================================================
# Configuration Request Models
# =============================================================================


class ConfigUpdateRequest(TimestampedRequest):
    """Configuration update request."""

    config_section: Optional[str] = Field(
        default=None,
        description="Specific configuration section to update (null for full config)",
        examples=["camera"],
    )
    config_data: dict[str, Any] = Field(
        ...,
        description="Configuration data to update",
        examples=[{"camera": {"resolution": [1920, 1080], "fps": 30}}],
    )
    validate_only: bool = Field(
        default=False,
        description="Only validate configuration without applying changes",
    )
    force_update: bool = Field(
        default=False, description="Force update even if validation warnings exist"
    )

    @field_validator("config_data")
    @classmethod
    def validate_config_data_not_empty(cls, v):
        if not v:
            raise ValueError("Configuration data cannot be empty")
        return v


class ConfigImportRequest(BaseRequest):
    """Configuration import request."""

    config_data: Union[dict[str, Any], str] = Field(
        ..., description="Configuration data to import (object or string)"
    )
    format: str = Field(
        default="json",
        pattern=r"^(json|yaml)$",
        description="Import format",
        examples=["json"],
    )
    merge_strategy: str = Field(
        default="replace",
        pattern=r"^(replace|merge|append)$",
        description="How to handle existing configuration",
        examples=["merge"],
    )
    validate_only: bool = Field(
        default=False, description="Only validate import without applying"
    )

    @field_validator("config_data")
    @classmethod
    def validate_config_data_not_empty(cls, v):
        if not v:
            raise ValueError("Configuration data cannot be empty")
        return v


class ConfigExportRequest(BaseRequest):
    """Configuration export request."""

    sections: Optional[list[str]] = Field(
        default=None,
        description="Specific sections to export (null for all)",
        examples=[["camera", "vision"]],
    )
    format: str = Field(
        default="json",
        pattern=r"^(json|yaml)$",
        description="Export format",
        examples=["json"],
    )
    include_defaults: bool = Field(
        default=False, description="Include default values in export"
    )
    include_metadata: bool = Field(
        default=True, description="Include configuration metadata"
    )


# =============================================================================
# Calibration Request Models
# =============================================================================


class CalibrationStartRequest(TimestampedRequest):
    """Calibration sequence start request."""

    calibration_type: str = Field(
        default="standard",
        pattern=r"^(standard|advanced|quick)$",
        description="Type of calibration to perform",
        examples=["standard"],
    )
    force_restart: bool = Field(
        default=False, description="Force restart if calibration already in progress"
    )
    timeout_seconds: int = Field(
        default=300,
        ge=60,
        le=1800,
        description="Calibration timeout in seconds",
        examples=[300],
    )


class CalibrationPointRequest(TimestampedRequest):
    """Calibration point capture request."""

    session_id: str = Field(
        ..., description="Active calibration session ID", examples=["cal_session_123"]
    )
    point_id: str = Field(
        ..., description="Calibration point identifier", examples=["corner_1"]
    )
    screen_position: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Screen coordinates [x, y]",
        examples=[[100.0, 200.0]],
    )
    world_position: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Real-world coordinates [x, y] in meters",
        examples=[[0.1, 0.2]],
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Point detection confidence",
        examples=[0.95],
    )

    @field_validator("screen_position", "world_position")
    @classmethod
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Coordinates must have exactly 2 values [x, y]")
        return v


class CalibrationApplyRequest(BaseRequest):
    """Calibration application request."""

    session_id: str = Field(
        ..., description="Calibration session ID to apply", examples=["cal_session_123"]
    )
    save_as_default: bool = Field(
        default=True, description="Save as default calibration"
    )
    backup_previous: bool = Field(
        default=True, description="Backup previous calibration"
    )


class CalibrationValidateRequest(BaseRequest):
    """Calibration validation request."""

    session_id: str = Field(
        ...,
        description="Calibration session ID to validate",
        examples=["cal_session_123"],
    )
    test_points: Optional[list[dict[str, list[float]]]] = Field(
        default=None,
        description="Additional test points for validation",
        examples=[[{"screen": [150.0, 250.0], "world": [0.15, 0.25]}]],
    )
    accuracy_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Required accuracy threshold",
        examples=[0.95],
    )


# =============================================================================
# Game State Request Models
# =============================================================================


class GameStateResetRequest(TimestampedRequest):
    """Game state reset request."""

    game_type: str = Field(
        default="practice",
        pattern=r"^(practice|8ball|9ball|straight)$",
        description="Type of game to initialize",
        examples=["8ball"],
    )
    preserve_table: bool = Field(
        default=True, description="Keep existing table configuration"
    )
    custom_setup: Optional[dict[str, Any]] = Field(
        default=None, description="Custom ball positions and game rules"
    )


class BallPositionUpdateRequest(TimestampedRequest):
    """Manual ball position update request."""

    ball_updates: list[dict[str, Any]] = Field(
        ...,
        description="List of ball position updates",
        examples=[
            [
                {
                    "id": "cue",
                    "position": [0.5, 0.6],
                    "velocity": [0.0, 0.0],
                    "is_pocketed": False,
                }
            ]
        ],
    )
    validate_positions: bool = Field(
        default=True, description="Validate ball positions against table bounds"
    )
    check_collisions: bool = Field(default=True, description="Check for ball overlaps")


class GameEventRequest(TimestampedRequest):
    """Game event recording request."""

    event_type: str = Field(
        ...,
        pattern=r"^(shot|pocket|scratch|foul|break|safety)$",
        description="Type of game event",
        examples=["shot"],
    )
    description: str = Field(
        ...,
        max_length=200,
        description="Human-readable event description",
        examples=["Player 1 attempted bank shot on 8-ball"],
    )
    player_id: Optional[int] = Field(
        default=None, description="Player associated with event", examples=[1]
    )
    ball_ids: Optional[list[str]] = Field(
        default=None,
        description="Balls involved in event",
        examples=[["cue", "ball_8"]],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional event metadata"
    )


class GameStateExportRequest(BaseModel):
    """Request model for game state export."""

    format: str = Field("json", description="Export format", pattern="^(json|csv)$")
    include_history: bool = Field(True, description="Include historical states")
    include_events: bool = Field(True, description="Include game events")
    time_range: Optional[dict[str, datetime]] = Field(
        None, description="Time range for export"
    )
    compression: bool = Field(False, description="Compress export data")


class GameHistoryRequest(BaseModel):
    """Request model for game history retrieval."""

    start_time: Optional[datetime] = Field(
        None, description="Start time for history query"
    )
    end_time: Optional[datetime] = Field(None, description="End time for history query")
    limit: int = Field(
        100, description="Maximum number of states to return", ge=1, le=10000
    )
    offset: int = Field(0, description="Offset for pagination", ge=0)
    include_events: bool = Field(False, description="Include game events in results")

    @field_validator("end_time")
    @classmethod
    def validate_time_range(cls, v, info):
        if v and info.data.get("start_time") and v <= info.data["start_time"]:
            raise ValueError("End time must be after start time")
        return v


class ShutdownRequest(BaseModel):
    """Request model for graceful shutdown."""

    delay: int = Field(0, description="Delay before shutdown in seconds", ge=0, le=300)
    force: bool = Field(
        False, description="Force shutdown without waiting for operations"
    )
    save_state: bool = Field(True, description="Save current state before shutdown")


class MetricsRequest(BaseModel):
    """Request model for performance metrics."""

    time_range: str = Field(
        "1h", description="Time range for metrics", pattern="^(5m|15m|1h|6h|24h|7d)$"
    )
    include_system: bool = Field(True, description="Include system metrics")
    include_vision: bool = Field(True, description="Include vision processing metrics")
    include_api: bool = Field(True, description="Include API metrics")
    format: str = Field(
        "json", description="Response format", pattern="^(json|prometheus)$"
    )


class ConfigResetRequest(BaseModel):
    """Request model for configuration reset."""

    confirm: bool = Field(..., description="Confirmation that reset is intended")
    backup_current: bool = Field(True, description="Create backup before reset")
    reset_type: str = Field(
        "all", description="Type of reset to perform", pattern="^(all|module|user)$"
    )
    modules: Optional[list[str]] = Field(None, description="Specific modules to reset")

    @field_validator("confirm")
    @classmethod
    def validate_confirmation(cls, v):
        if not v:
            raise ValueError("Configuration reset must be explicitly confirmed")
        return v


# =============================================================================
# System Control Request Models
# =============================================================================


class SystemControlRequest(BaseRequest):
    """System control request."""

    action: str = Field(
        ...,
        pattern=r"^(start|stop|restart|pause|resume|shutdown)$",
        description="System action to perform",
        examples=["restart"],
    )
    component: Optional[str] = Field(
        default=None,
        pattern=r"^(vision|projector|api|websocket|all)$",
        description="Specific component to control (null for all)",
        examples=["vision"],
    )
    force: bool = Field(default=False, description="Force action even if unsafe")
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Action timeout in seconds", examples=[30]
    )


class SystemMaintenanceRequest(BaseRequest):
    """System maintenance request."""

    operation: str = Field(
        ...,
        pattern=r"^(cleanup|optimize|backup|restore|update)$",
        description="Maintenance operation to perform",
        examples=["cleanup"],
    )
    target: Optional[str] = Field(
        default=None, description="Specific target for operation", examples=["logs"]
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific parameters"
    )


# =============================================================================
# Hardware Control Request Models
# =============================================================================


class CameraControlRequest(TimestampedRequest):
    """Camera control request."""

    action: str = Field(
        ...,
        pattern=r"^(start|stop|restart|configure|capture)$",
        description="Camera action to perform",
        examples=["configure"],
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Camera parameters",
        examples=[{"resolution": [1920, 1080], "fps": 30, "exposure": "auto"}],
    )


class ProjectorControlRequest(TimestampedRequest):
    """Projector control request."""

    action: str = Field(
        ...,
        pattern=r"^(start|stop|restart|configure|display|hide)$",
        description="Projector action to perform",
        examples=["display"],
    )
    content_type: Optional[str] = Field(
        default=None,
        pattern=r"^(trajectory|overlay|calibration|test)$",
        description="Type of content to display",
        examples=["trajectory"],
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Projector parameters"
    )


# =============================================================================
# Data Export/Import Request Models
# =============================================================================


class DataExportRequest(BaseRequest):
    """Data export request."""

    data_type: str = Field(
        ...,
        pattern=r"^(game_states|trajectories|events|all)$",
        description="Type of data to export",
        examples=["game_states"],
    )
    format: str = Field(
        default="json",
        pattern=r"^(json|csv|xlsx)$",
        description="Export format",
        examples=["json"],
    )
    start_time: Optional[datetime] = Field(
        default=None,
        description="Start time for data range",
        examples=["2024-01-01T00:00:00Z"],
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="End time for data range",
        examples=["2024-01-31T23:59:59Z"],
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Additional data filters"
    )


class DataImportRequest(BaseRequest):
    """Data import request."""

    data_type: str = Field(
        ...,
        pattern=r"^(configuration|calibration|game_data)$",
        description="Type of data to import",
        examples=["configuration"],
    )
    format: str = Field(
        default="json",
        pattern=r"^(json|csv|xlsx|yaml)$",
        description="Data format",
        examples=["json"],
    )
    merge_strategy: str = Field(
        default="replace",
        pattern=r"^(replace|merge|append|skip)$",
        description="How to handle existing data",
        examples=["merge"],
    )
    validate_only: bool = Field(
        default=False, description="Only validate without importing"
    )


# =============================================================================
# WebSocket Subscription Request Models
# =============================================================================


class WebSocketSubscribeRequest(BaseRequest):
    """WebSocket subscription request."""

    streams: list[str] = Field(
        ...,
        description="List of streams to subscribe to",
        examples=[["frames", "state", "trajectories"]],
    )
    quality: str = Field(
        default="high",
        pattern=r"^(low|medium|high)$",
        description="Stream quality level",
        examples=["high"],
    )
    frame_rate: Optional[int] = Field(
        default=None,
        ge=1,
        le=60,
        description="Requested frame rate (null for max)",
        examples=[30],
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Stream-specific filters"
    )


class WebSocketUnsubscribeRequest(BaseRequest):
    """WebSocket unsubscription request."""

    streams: list[str] = Field(
        ..., description="List of streams to unsubscribe from", examples=[["frames"]]
    )


class SessionDataExportRequest(BaseRequest):
    """Session data export request."""

    session_id: Optional[str] = Field(
        default=None, description="Specific session ID to export"
    )
    include_raw_frames: bool = Field(
        default=False, description="Include raw frame data"
    )
    include_processed_data: bool = Field(
        default=True, description="Include processed vision data"
    )
    include_trajectories: bool = Field(
        default=True, description="Include trajectory calculations"
    )
    compression_level: int = Field(
        default=6, ge=0, le=9, description="Compression level (0-9)"
    )
    format: str = Field(
        default="zip", pattern=r"^(zip|tar|json)$", description="Export format"
    )


# =============================================================================
# Request Validation Utilities
# =============================================================================


def validate_coordinate_list(
    coords: list[float], expected_length: int = 2
) -> list[float]:
    """Validate coordinate list format and values."""
    if len(coords) != expected_length:
        raise ValueError(f"Expected {expected_length} coordinates, got {len(coords)}")

    for i, coord in enumerate(coords):
        if not isinstance(coord, (int, float)):
            raise ValueError(f"Coordinate {i} must be numeric")
        if not -1000 <= coord <= 1000:  # Reasonable bounds
            raise ValueError(f"Coordinate {i} out of reasonable bounds")

    return coords


def validate_session_id(session_id: str) -> str:
    """Validate session ID format."""
    if not session_id or len(session_id) < 8:
        raise ValueError("Session ID must be at least 8 characters")

    if not session_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Session ID must be alphanumeric with optional _ or -")

    return session_id
