"""WebSocket Message Models.

This module defines all Pydantic models for WebSocket message validation, including:
- Frame message format for video streaming
- State message format for game state updates
- Trajectory message format for shot predictions
- Alert message format for system notifications
- Configuration message format for real-time updates

All models include comprehensive validation, documentation, and serialization support.
"""

import base64
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Base WebSocket Message Models
# =============================================================================


class MessageType(str, Enum):
    """WebSocket message type enumeration."""

    FRAME = "frame"
    STATE = "state"
    TRAJECTORY = "trajectory"
    ALERT = "alert"
    CONFIG = "config"
    METRICS = "metrics"
    EVENT = "event"
    COMMAND = "command"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StreamQuality(str, Enum):
    """Stream quality levels."""

    LOW = "low"  # 480p, low fps
    MEDIUM = "medium"  # 720p, medium fps
    HIGH = "high"  # 1080p, high fps
    ULTRA = "ultra"  # 4K, max fps


class BaseWebSocketMessage(BaseModel):
    """Base class for all WebSocket messages."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    type: MessageType = Field(..., description="Message type identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    sequence: int = Field(..., description="Message sequence number for ordering")
    connection_id: Optional[str] = Field(
        default=None, description="WebSocket connection identifier"
    )
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL, description="Message priority level"
    )


class TimestampedMessage(BaseWebSocketMessage):
    """WebSocket message with client/server timing information."""

    client_timestamp: Optional[datetime] = Field(
        default=None, description="Client-side timestamp for latency calculation"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Server processing time in milliseconds", ge=0.0
    )


# =============================================================================
# Frame Message Models
# =============================================================================


class FrameMetadata(BaseModel):
    """Video frame metadata."""

    width: int = Field(..., description="Frame width in pixels", gt=0)
    height: int = Field(..., description="Frame height in pixels", gt=0)
    fps: float = Field(..., description="Current frames per second", gt=0.0)
    quality: StreamQuality = Field(..., description="Stream quality level")
    format: str = Field(
        default="jpeg", pattern=r"^(jpeg|png|webp)$", description="Image format"
    )
    compression_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Compression quality (0=lowest, 1=highest)",
    )
    encoding_time_ms: Optional[float] = Field(
        default=None, description="Frame encoding time in milliseconds", ge=0.0
    )


class FrameData(BaseModel):
    """Video frame data container."""

    image_data: str = Field(..., description="Base64 encoded image data")
    metadata: FrameMetadata = Field(..., description="Frame metadata")

    @field_validator("image_data")
    @classmethod
    def validate_base64_image(cls, v: str) -> str:
        """Validate that image_data is valid base64."""
        try:
            # Try to decode to validate base64
            base64.b64decode(v, validate=True)
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")


class FrameMessage(TimestampedMessage):
    """Video frame streaming message."""

    type: MessageType = Field(default=MessageType.FRAME, frozen=True)
    frame_number: int = Field(..., description="Sequential frame number", ge=0)
    data: FrameData = Field(..., description="Frame image data and metadata")
    annotations: Optional[dict[str, Any]] = Field(
        default=None, description="Optional frame annotations (detected objects, etc.)"
    )


# =============================================================================
# Game State Message Models
# =============================================================================


class BallStateData(BaseModel):
    """Ball state information for WebSocket messages."""

    id: str = Field(..., description="Ball identifier")
    number: Optional[int] = Field(None, description="Ball number (1-15)")
    position: list[float] = Field(
        ..., min_length=2, max_length=2, description="Ball position [x, y] in meters"
    )
    velocity: list[float] = Field(
        ..., min_length=2, max_length=2, description="Ball velocity [x, y] in m/s"
    )
    radius: float = Field(default=0.028575, gt=0.0, description="Ball radius in meters")
    is_cue_ball: bool = Field(default=False, description="Whether this is the cue ball")
    is_pocketed: bool = Field(default=False, description="Whether ball is pocketed")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence"
    )


class CueStateData(BaseModel):
    """Cue stick state information for WebSocket messages."""

    tip_position: list[float] = Field(
        ..., min_length=2, max_length=2, description="Cue tip position [x, y] in meters"
    )
    angle: float = Field(..., ge=-180.0, le=180.0, description="Cue angle in degrees")
    elevation: float = Field(
        default=0.0, ge=-90.0, le=90.0, description="Cue elevation in degrees"
    )
    estimated_force: float = Field(
        default=0.0, ge=0.0, description="Estimated shot force in Newtons"
    )
    is_visible: bool = Field(default=False, description="Whether cue is detected")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence"
    )


class TableStateData(BaseModel):
    """Table state information for WebSocket messages."""

    width: float = Field(..., gt=0.0, description="Table width in meters")
    height: float = Field(..., gt=0.0, description="Table height in meters")
    pocket_positions: list[list[float]] = Field(
        ..., description="Pocket positions [[x, y], ...]"
    )
    pocket_radius: float = Field(
        default=0.0635, gt=0.0, description="Pocket radius in meters"
    )


class GameStateData(BaseModel):
    """Complete game state data for WebSocket messages."""

    frame_number: int = Field(..., ge=0, description="Frame number")
    balls: list[BallStateData] = Field(..., description="Ball states")
    cue: Optional[CueStateData] = Field(None, description="Cue stick state")
    table: TableStateData = Field(..., description="Table configuration")
    game_type: str = Field(
        default="practice",
        pattern=r"^(practice|8ball|9ball|straight)$",
        description="Type of game",
    )
    is_valid: bool = Field(default=True, description="State validity")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall state confidence"
    )


class StateMessage(TimestampedMessage):
    """Game state update message."""

    type: MessageType = Field(default=MessageType.STATE, frozen=True)
    data: GameStateData = Field(..., description="Game state data")
    changes: Optional[list[str]] = Field(
        default=None, description="List of changed fields since last update"
    )


# =============================================================================
# Trajectory Message Models
# =============================================================================


class TrajectoryPoint(BaseModel):
    """Single point on a trajectory path."""

    position: list[float] = Field(
        ..., min_length=2, max_length=2, description="Point position [x, y] in meters"
    )
    time: float = Field(..., ge=0.0, description="Time offset in seconds")
    velocity: Optional[list[float]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Velocity at this point [x, y] in m/s",
    )


class CollisionInfo(BaseModel):
    """Collision prediction information."""

    position: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Collision position [x, y] in meters",
    )
    time: float = Field(..., ge=0.0, description="Time until collision in seconds")
    type: str = Field(
        ..., pattern=r"^(ball|cushion|pocket)$", description="Type of collision"
    )
    ball1_id: str = Field(..., description="Primary ball ID")
    ball2_id: Optional[str] = Field(
        None, description="Secondary ball ID (for ball collisions)"
    )
    impact_angle: float = Field(
        ..., ge=-180.0, le=180.0, description="Impact angle in degrees"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Collision prediction confidence"
    )


class TrajectoryData(BaseModel):
    """Trajectory prediction data."""

    ball_id: str = Field(..., description="Ball identifier")
    points: list[TrajectoryPoint] = Field(..., description="Trajectory path points")
    collisions: list[CollisionInfo] = Field(
        default_factory=list, description="Predicted collisions"
    )
    will_be_pocketed: bool = Field(
        default=False, description="Whether ball will be pocketed"
    )
    pocket_id: Optional[int] = Field(
        default=None, description="Target pocket ID if pocketed"
    )
    time_to_rest: float = Field(
        ..., ge=0.0, description="Time until ball comes to rest in seconds"
    )
    max_velocity: float = Field(
        default=0.0, ge=0.0, description="Maximum velocity during trajectory in m/s"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall trajectory confidence"
    )


class TrajectoryMessage(TimestampedMessage):
    """Trajectory prediction message."""

    type: MessageType = Field(default=MessageType.TRAJECTORY, frozen=True)
    trajectories: list[TrajectoryData] = Field(
        ..., description="Ball trajectory predictions"
    )
    shot_analysis: Optional[dict[str, Any]] = Field(
        default=None, description="Optional shot analysis data"
    )


# =============================================================================
# Alert Message Models
# =============================================================================


class AlertData(BaseModel):
    """Alert notification data."""

    level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(
        ..., min_length=1, max_length=500, description="Human-readable alert message"
    )
    code: str = Field(
        ...,
        pattern=r"^[A-Z]{2,5}_\d{3}$",
        description="Standardized alert code (e.g., CAM_001)",
    )
    component: Optional[str] = Field(
        default=None, description="System component that generated the alert"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional alert details"
    )
    auto_dismiss: bool = Field(
        default=False, description="Whether alert should auto-dismiss"
    )
    dismiss_timeout: Optional[int] = Field(
        default=None, ge=1, description="Auto-dismiss timeout in seconds"
    )
    actions: list[str] = Field(
        default_factory=list, description="Available user actions"
    )


class AlertMessage(BaseWebSocketMessage):
    """System alert notification message."""

    type: MessageType = Field(default=MessageType.ALERT, frozen=True)
    priority: MessagePriority = Field(default=MessagePriority.HIGH)
    data: AlertData = Field(..., description="Alert information")


# =============================================================================
# Configuration Message Models
# =============================================================================


class ConfigChangeData(BaseModel):
    """Configuration change notification data."""

    section: str = Field(..., description="Configuration section changed")
    key: str = Field(..., description="Configuration key changed")
    old_value: Any = Field(..., description="Previous value")
    new_value: Any = Field(..., description="New value")
    source: str = Field(
        ...,
        pattern=r"^(api|file|environment|runtime)$",
        description="Source of the change",
    )
    requires_restart: bool = Field(
        default=False, description="Whether change requires system restart"
    )


class ConfigMessage(BaseWebSocketMessage):
    """Configuration update notification message."""

    type: MessageType = Field(default=MessageType.CONFIG, frozen=True)
    changes: list[ConfigChangeData] = Field(..., description="Configuration changes")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors if any"
    )


# =============================================================================
# Metrics Message Models
# =============================================================================


class PerformanceMetrics(BaseModel):
    """System performance metrics data."""

    cpu_usage: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(
        ..., ge=0.0, le=100.0, description="Memory usage percentage"
    )
    disk_usage: float = Field(
        ..., ge=0.0, le=100.0, description="Disk usage percentage"
    )
    network_io: dict[str, float] = Field(
        default_factory=dict, description="Network I/O statistics"
    )
    processing_fps: float = Field(
        default=0.0, ge=0.0, description="Video processing frames per second"
    )
    api_requests_per_second: float = Field(
        default=0.0, ge=0.0, description="API requests per second"
    )
    websocket_connections: int = Field(
        default=0, ge=0, description="Active WebSocket connections"
    )
    error_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Error rate percentage"
    )


class MetricsMessage(BaseWebSocketMessage):
    """System metrics update message."""

    type: MessageType = Field(default=MessageType.METRICS, frozen=True)
    metrics: PerformanceMetrics = Field(..., description="Performance metrics data")
    collection_interval: int = Field(
        default=5, ge=1, description="Metrics collection interval in seconds"
    )


# =============================================================================
# Event Message Models
# =============================================================================


class GameEventData(BaseModel):
    """Game event notification data."""

    event_type: str = Field(
        ...,
        pattern=r"^(shot|pocket|scratch|foul|break|safety|game_start|game_end)$",
        description="Type of game event",
    )
    description: str = Field(
        ..., max_length=200, description="Human-readable event description"
    )
    player_id: Optional[int] = Field(None, description="Player involved in event")
    ball_ids: list[str] = Field(
        default_factory=list, description="Balls involved in event"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional event metadata"
    )


class EventMessage(BaseWebSocketMessage):
    """Game event notification message."""

    type: MessageType = Field(default=MessageType.EVENT, frozen=True)
    data: GameEventData = Field(..., description="Game event data")


# =============================================================================
# Command and Response Message Models
# =============================================================================


class CommandData(BaseModel):
    """WebSocket command data."""

    action: str = Field(..., description="Command action")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Command parameters"
    )
    timeout: Optional[int] = Field(
        default=None, ge=1, description="Command timeout in seconds"
    )
    require_response: bool = Field(
        default=True, description="Whether command requires a response"
    )


class CommandMessage(BaseWebSocketMessage):
    """WebSocket command message."""

    type: MessageType = Field(default=MessageType.COMMAND, frozen=True)
    command_id: str = Field(..., description="Unique command identifier")
    data: CommandData = Field(..., description="Command data")


class ResponseData(BaseModel):
    """WebSocket response data."""

    success: bool = Field(..., description="Command execution success")
    result: Optional[Any] = Field(None, description="Command result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Command execution time in milliseconds"
    )


class ResponseMessage(BaseWebSocketMessage):
    """WebSocket response message."""

    type: MessageType = Field(default=MessageType.RESPONSE, frozen=True)
    command_id: str = Field(..., description="Original command identifier")
    data: ResponseData = Field(..., description="Response data")


# =============================================================================
# Heartbeat and Error Message Models
# =============================================================================


class HeartbeatMessage(BaseWebSocketMessage):
    """WebSocket heartbeat/ping message."""

    type: MessageType = Field(default=MessageType.HEARTBEAT, frozen=True)
    server_time: datetime = Field(..., description="Server timestamp")
    uptime: float = Field(..., ge=0.0, description="Server uptime in seconds")


class ErrorData(BaseModel):
    """WebSocket error data."""

    error_code: str = Field(..., description="Standardized error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional error details"
    )
    recoverable: bool = Field(default=True, description="Whether error is recoverable")


class ErrorMessage(BaseWebSocketMessage):
    """WebSocket error notification message."""

    type: MessageType = Field(default=MessageType.ERROR, frozen=True)
    priority: MessagePriority = Field(default=MessagePriority.HIGH)
    data: ErrorData = Field(..., description="Error information")


# =============================================================================
# Message Factory Functions
# =============================================================================


def create_frame_message(
    frame_data: bytes,
    width: int,
    height: int,
    fps: float,
    frame_number: int,
    sequence: int,
    quality: StreamQuality = StreamQuality.HIGH,
    format: str = "jpeg",
) -> FrameMessage:
    """Create a frame message from raw image data."""
    image_b64 = base64.b64encode(frame_data).decode("utf-8")

    metadata = FrameMetadata(
        width=width, height=height, fps=fps, quality=quality, format=format
    )

    frame_data_obj = FrameData(image_data=image_b64, metadata=metadata)

    return FrameMessage(
        timestamp=datetime.now(),
        sequence=sequence,
        frame_number=frame_number,
        data=frame_data_obj,
    )


def create_alert_message(
    level: AlertLevel,
    message: str,
    code: str,
    sequence: int,
    component: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> AlertMessage:
    """Create an alert message."""
    alert_data = AlertData(
        level=level,
        message=message,
        code=code,
        component=component,
        details=details or {},
    )

    return AlertMessage(timestamp=datetime.now(), sequence=sequence, data=alert_data)


def create_error_message(
    error_code: str,
    message: str,
    sequence: int,
    details: Optional[dict[str, Any]] = None,
    recoverable: bool = True,
) -> ErrorMessage:
    """Create an error message."""
    error_data = ErrorData(
        error_code=error_code,
        message=message,
        details=details or {},
        recoverable=recoverable,
    )

    return ErrorMessage(timestamp=datetime.now(), sequence=sequence, data=error_data)


# =============================================================================
# Message Validation Utilities
# =============================================================================


def validate_websocket_message(message_dict: dict[str, Any]) -> BaseWebSocketMessage:
    """Validate and parse a WebSocket message from dictionary."""
    message_type = message_dict.get("type")

    if not message_type:
        raise ValueError("Message must have a 'type' field")

    # Message type mapping
    message_classes = {
        MessageType.FRAME: FrameMessage,
        MessageType.STATE: StateMessage,
        MessageType.TRAJECTORY: TrajectoryMessage,
        MessageType.ALERT: AlertMessage,
        MessageType.CONFIG: ConfigMessage,
        MessageType.METRICS: MetricsMessage,
        MessageType.EVENT: EventMessage,
        MessageType.COMMAND: CommandMessage,
        MessageType.RESPONSE: ResponseMessage,
        MessageType.HEARTBEAT: HeartbeatMessage,
        MessageType.ERROR: ErrorMessage,
    }

    try:
        msg_type = MessageType(message_type)
        message_class = message_classes.get(msg_type)

        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")

        return message_class.model_validate(message_dict)

    except ValueError as e:
        raise ValueError(f"Invalid message type '{message_type}': {e}")


def serialize_websocket_message(message: BaseWebSocketMessage) -> dict[str, Any]:
    """Serialize a WebSocket message to dictionary."""
    return message.model_dump(mode="json", exclude_none=True)
