"""WebSocket message protocol schemas using Pydantic for validation and documentation."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class MessageType(str, Enum):
    """WebSocket message types."""

    # Outbound data streams
    FRAME = "frame"
    STATE = "state"
    TRAJECTORY = "trajectory"
    ALERT = "alert"
    CONFIG = "config"
    METRICS = "metrics"

    # Connection management
    CONNECTION = "connection"
    PING = "ping"
    PONG = "pong"

    # Subscription management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # Status and control
    STATUS = "status"
    ERROR = "error"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityLevel(str, Enum):
    """Video quality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


# Base message schema
class WebSocketMessage(BaseModel):
    """Base WebSocket message structure."""

    type: MessageType = Field(..., description="Message type identifier")
    timestamp: datetime = Field(..., description="Message timestamp in ISO format")
    sequence: Optional[int] = Field(None, description="Message sequence number")
    data: dict[str, Any] = Field(..., description="Message payload data")

    @field_serializer("timestamp")
    def serialize_timestamp(self, timestamp: datetime) -> str:
        """Serialize timestamp to ISO format."""
        return timestamp.isoformat()


# Data stream message schemas
class FrameData(BaseModel):
    """Video frame data structure."""

    image: str = Field(..., description="Base64 encoded image data")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    format: str = Field("jpeg", description="Image format (jpeg, png, etc.)")
    quality: int = Field(85, ge=1, le=100, description="Image quality (1-100)")
    compressed: bool = Field(False, description="Whether image data is compressed")
    fps: float = Field(30.0, ge=0, description="Current frames per second")
    size_bytes: int = Field(..., ge=0, description="Original image size in bytes")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate image format is one of the supported types."""
        allowed_formats = ["jpeg", "jpg", "png", "webp"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Format must be one of: {allowed_formats}")
        return v.lower()


class BallData(BaseModel):
    """Individual ball data structure."""

    id: str = Field(..., description="Ball identifier (cue, 1, 2, ..., 8ball, etc.)")
    position: list[float] = Field(
        ..., min_length=2, max_length=2, description="[x, y] position"
    )
    radius: float = Field(..., gt=0, description="Ball radius in pixels")
    color: str = Field(..., description="Ball color name or hex code")
    velocity: Optional[list[float]] = Field(
        None, min_length=2, max_length=2, description="[vx, vy] velocity"
    )
    confidence: float = Field(1.0, ge=0, le=1, description="Detection confidence")
    visible: bool = Field(True, description="Whether ball is visible/detected")

    @field_validator("position", "velocity")
    @classmethod
    def validate_coordinates(cls, v):
        """Validate coordinate arrays are exactly [x, y] format."""
        if v is not None and len(v) != 2:
            raise ValueError("Coordinates must be [x, y] format")
        return v


class CueData(BaseModel):
    """Cue stick data structure."""

    angle: float = Field(..., description="Cue angle in degrees")
    position: list[float] = Field(
        ..., min_length=2, max_length=2, description="[x, y] cue position"
    )
    detected: bool = Field(True, description="Whether cue is detected")
    confidence: float = Field(1.0, ge=0, le=1, description="Detection confidence")
    length: Optional[float] = Field(
        None, gt=0, description="Detected cue length in pixels"
    )
    tip_position: Optional[list[float]] = Field(
        None, min_length=2, max_length=2, description="Cue tip position"
    )


class TableData(BaseModel):
    """Pool table data structure."""

    corners: list[list[float]] = Field(
        ..., min_length=4, max_length=4, description="Table corner coordinates"
    )
    pockets: list[list[float]] = Field(..., description="Pocket center coordinates")
    rails: Optional[list[dict[str, Any]]] = Field(None, description="Rail segment data")
    calibrated: bool = Field(True, description="Whether table is properly calibrated")
    dimensions: Optional[dict[str, float]] = Field(
        None, description="Real-world table dimensions"
    )

    @field_validator("corners")
    @classmethod
    def validate_corners(cls, v):
        """Validate table has exactly 4 corners with [x, y] coordinates."""
        if len(v) != 4:
            raise ValueError("Table must have exactly 4 corners")
        for corner in v:
            if len(corner) != 2:
                raise ValueError("Each corner must be [x, y] coordinates")
        return v


class GameStateData(BaseModel):
    """Complete game state data structure."""

    balls: list[BallData] = Field(..., description="List of detected balls")
    cue: Optional[CueData] = Field(None, description="Cue stick data")
    table: Optional[TableData] = Field(None, description="Table geometry data")
    ball_count: int = Field(..., ge=0, description="Total number of balls detected")
    frame_number: Optional[int] = Field(None, ge=0, description="Frame sequence number")


class TrajectoryLine(BaseModel):
    """Individual trajectory line segment."""

    start: list[float] = Field(
        ..., min_length=2, max_length=2, description="Line start coordinates"
    )
    end: list[float] = Field(
        ..., min_length=2, max_length=2, description="Line end coordinates"
    )
    type: Literal["primary", "reflection", "collision"] = Field(
        ..., description="Line type"
    )
    confidence: float = Field(1.0, ge=0, le=1, description="Prediction confidence")


class CollisionData(BaseModel):
    """Ball collision prediction data."""

    position: list[float] = Field(
        ..., min_length=2, max_length=2, description="Collision point"
    )
    ball_id: str = Field(..., description="ID of ball being hit")
    angle: float = Field(..., description="Collision angle in degrees")
    velocity_before: Optional[list[float]] = Field(
        None, description="Ball velocity before collision"
    )
    velocity_after: Optional[list[float]] = Field(
        None, description="Ball velocity after collision"
    )
    time_to_collision: Optional[float] = Field(
        None, ge=0, description="Time to collision in seconds"
    )


class TrajectoryData(BaseModel):
    """Trajectory prediction data structure."""

    lines: list[TrajectoryLine] = Field(..., description="Trajectory line segments")
    collisions: list[CollisionData] = Field(
        default_factory=list, description="Predicted collisions"
    )
    confidence: float = Field(
        1.0, ge=0, le=1, description="Overall prediction confidence"
    )
    calculation_time_ms: float = Field(
        0.0, ge=0, description="Calculation time in milliseconds"
    )
    line_count: int = Field(..., ge=0, description="Number of trajectory lines")
    collision_count: int = Field(
        ..., ge=0, description="Number of predicted collisions"
    )


class AlertData(BaseModel):
    """Alert/notification data structure."""

    level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(..., description="Human-readable alert message")
    code: str = Field(..., description="Alert code for programmatic handling")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional alert details"
    )


class ConfigData(BaseModel):
    """Configuration update data structure."""

    section: str = Field(..., description="Configuration section name")
    config: dict[str, Any] = Field(..., description="Configuration data")
    change_summary: Optional[str] = Field(None, description="Summary of changes made")


class ConnectionData(BaseModel):
    """Connection status data structure."""

    client_id: str = Field(..., description="Unique client identifier")
    status: Literal["connected", "reconnecting", "disconnected"] = Field(
        ..., description="Connection status"
    )
    timestamp: datetime = Field(..., description="Status change timestamp")


class StatusData(BaseModel):
    """Client status information."""

    client_id: str = Field(..., description="Client identifier")
    user_id: Optional[str] = Field(None, description="User identifier if authenticated")
    connected_at: datetime = Field(..., description="Connection establishment time")
    uptime: float = Field(..., ge=0, description="Connection uptime in seconds")
    subscriptions: list[str] = Field(
        default_factory=list, description="Active subscriptions"
    )
    message_count: int = Field(0, ge=0, description="Total messages sent to client")
    bytes_sent: int = Field(0, ge=0, description="Total bytes sent to client")
    bytes_received: int = Field(0, ge=0, description="Total bytes received from client")
    quality_score: float = Field(
        1.0, ge=0, le=1, description="Connection quality score"
    )
    last_ping_latency: float = Field(
        0.0, ge=0, description="Last ping latency in seconds"
    )
    is_alive: bool = Field(True, description="Whether connection is active")


class ErrorData(BaseModel):
    """Error message data structure."""

    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class MetricsData(BaseModel):
    """Performance metrics data structure."""

    broadcast_stats: dict[str, Union[int, float, bool]] = Field(
        ..., description="Broadcasting statistics"
    )
    frame_metrics: dict[str, Union[int, float]] = Field(
        ..., description="Frame streaming metrics"
    )
    connection_stats: dict[str, Any] = Field(..., description="Connection statistics")


# Client-sent message schemas
class SubscribeRequest(BaseModel):
    """Client subscription request."""

    streams: list[str] = Field(..., description="List of stream types to subscribe to")
    filters: Optional[dict[str, Any]] = Field(
        None, description="Optional subscription filters"
    )


class UnsubscribeRequest(BaseModel):
    """Client unsubscription request."""

    streams: list[str] = Field(
        ..., description="List of stream types to unsubscribe from"
    )


class PingRequest(BaseModel):
    """Client ping request."""

    timestamp: datetime = Field(..., description="Client timestamp")


class StatusRequest(BaseModel):
    """Client status request."""

    include_details: bool = Field(
        True, description="Include detailed status information"
    )


# Response message schemas
class SubscribedResponse(BaseModel):
    """Subscription confirmation response."""

    streams: list[str] = Field(..., description="Successfully subscribed streams")
    all_subscriptions: list[str] = Field(..., description="All active subscriptions")


class UnsubscribedResponse(BaseModel):
    """Unsubscription confirmation response."""

    streams: list[str] = Field(..., description="Successfully unsubscribed streams")
    all_subscriptions: list[str] = Field(
        ..., description="Remaining active subscriptions"
    )


class PongResponse(BaseModel):
    """Ping response data."""

    timestamp: datetime = Field(..., description="Server timestamp")
    quality_score: float = Field(
        ..., ge=0, le=1, description="Connection quality score"
    )


# Message factory functions
def create_frame_message(
    frame_data: FrameData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a frame message."""
    return WebSocketMessage(
        type=MessageType.FRAME,
        timestamp=datetime.now(),
        sequence=sequence,
        data=frame_data.dict(),
    )


def create_state_message(
    state_data: GameStateData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a game state message."""
    return WebSocketMessage(
        type=MessageType.STATE,
        timestamp=datetime.now(),
        sequence=sequence,
        data=state_data.dict(),
    )


def create_trajectory_message(
    trajectory_data: TrajectoryData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a trajectory message."""
    return WebSocketMessage(
        type=MessageType.TRAJECTORY,
        timestamp=datetime.now(),
        sequence=sequence,
        data=trajectory_data.dict(),
    )


def create_alert_message(
    alert_data: AlertData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create an alert message."""
    return WebSocketMessage(
        type=MessageType.ALERT,
        timestamp=datetime.now(),
        sequence=sequence,
        data=alert_data.dict(),
    )


def create_config_message(
    config_data: ConfigData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a configuration update message."""
    return WebSocketMessage(
        type=MessageType.CONFIG,
        timestamp=datetime.now(),
        sequence=sequence,
        data=config_data.dict(),
    )


def create_connection_message(
    connection_data: ConnectionData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a connection status message."""
    return WebSocketMessage(
        type=MessageType.CONNECTION,
        timestamp=datetime.now(),
        sequence=sequence,
        data=connection_data.dict(),
    )


def create_error_message(
    error_data: ErrorData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create an error message."""
    return WebSocketMessage(
        type=MessageType.ERROR,
        timestamp=datetime.now(),
        sequence=sequence,
        data=error_data.dict(),
    )


def create_status_message(
    status_data: StatusData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a status message."""
    return WebSocketMessage(
        type=MessageType.STATUS,
        timestamp=datetime.now(),
        sequence=sequence,
        data=status_data.dict(),
    )


def create_metrics_message(
    metrics_data: MetricsData, sequence: Optional[int] = None
) -> WebSocketMessage:
    """Create a metrics message."""
    return WebSocketMessage(
        type=MessageType.METRICS,
        timestamp=datetime.now(),
        sequence=sequence,
        data=metrics_data.dict(),
    )


# Validation helper functions
def validate_websocket_message(message_dict: dict[str, Any]) -> WebSocketMessage:
    """Validate and parse a WebSocket message."""
    return WebSocketMessage(**message_dict)


def validate_client_message(message_type: str, data: dict[str, Any]) -> BaseModel:
    """Validate client-sent message data based on type."""
    message_type_map = {
        "subscribe": SubscribeRequest,
        "unsubscribe": UnsubscribeRequest,
        "ping": PingRequest,
        "get_status": StatusRequest,
    }

    if message_type not in message_type_map:
        raise ValueError(f"Unknown message type: {message_type}")

    schema_class = message_type_map[message_type]
    return schema_class(**data)


# Constants for validation
VALID_STREAM_TYPES = [
    stream_type.value
    for stream_type in MessageType
    if stream_type.value in ["frame", "state", "trajectory", "alert", "config"]
]
VALID_ALERT_LEVELS = [level.value for level in AlertLevel]
VALID_QUALITY_LEVELS = [level.value for level in QualityLevel]
