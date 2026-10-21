"""WebSocket message protocol schemas using Pydantic for validation and documentation."""

import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from backend.api.models.common import PositionWithScale


def _validate_scale_factors(value: list[float]) -> list[float]:
    """Validate scale metadata for pixel dimensions that are not points."""
    if len(value) != 2 or not all(math.isfinite(item) and item > 0 for item in value):
        raise ValueError("Scale factors must contain two positive finite values")
    return value


class StreamType(str, Enum):
    """Data streams supported by the connection manager and API."""

    FRAME = "frame"
    STATE = "state"
    TRAJECTORY = "trajectory"
    ALERT = "alert"
    CONFIG = "config"
    PERFORMANCE = "performance"


class MessageType(str, Enum):
    """WebSocket message types."""

    # Outbound data streams
    FRAME = StreamType.FRAME.value
    STATE = StreamType.STATE.value
    TRAJECTORY = StreamType.TRAJECTORY.value
    ALERT = StreamType.ALERT.value
    CONFIG = StreamType.CONFIG.value
    PERFORMANCE = StreamType.PERFORMANCE.value
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
    def validate_format(cls: type[Any], v: str) -> str:
        """Validate image format is one of the supported types."""
        allowed_formats = ["jpeg", "jpg", "png", "webp"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Format must be one of: {allowed_formats}")
        return v.lower()


class BallData(BaseModel):
    """Individual ball data structure shared by Python and Rust providers.

    Rust localization intentionally does not invent color or identity before
    there is evidence. The optional scene fields describe the measurement and
    its provenance while retaining the established state stream shape.
    """

    model_config = ConfigDict(extra="allow")

    id: Union[str, int] = Field(
        ..., description="Provider track identifier; may be numeric for legacy states"
    )
    position: PositionWithScale = Field(
        ..., description="Ball center in scaled image coordinates"
    )
    radius: float = Field(..., gt=0, description="Ball radius in pixels")
    radius_scale: list[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Scale factors for radius in the source image axes",
    )
    color: Optional[str] = Field(
        None,
        description="Ball color name or hex code; null when color is not established",
    )
    velocity: Optional[PositionWithScale] = Field(
        None, description="Velocity in scaled image-pixel units per second"
    )
    confidence: float = Field(1.0, ge=0, le=1, description="Detection confidence")
    visible: bool = Field(True, description="Whether ball is visible/detected")
    is_observed: bool = Field(
        True,
        description="Whether this position has a detector match in the latest frame",
    )
    number: Optional[int] = Field(
        None, ge=1, le=15, description="Numbered-ball identity when established"
    )
    identity_status: Optional[Literal["unknown", "identified"]] = Field(
        None, description="Whether numbered-ball identity is known"
    )
    identity_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence of the optional identity result"
    )
    identity_evidence: list[dict[str, Any]] = Field(
        default_factory=list, description="Evidence supporting optional identity"
    )
    confidence_kind: Optional[Literal["detector_observation", "unavailable"]] = Field(
        None,
        description="Whether confidence describes a current detector match",
    )
    last_observation_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Historical confidence from the last detection"
    )
    tracking_status: Optional[Literal["observed", "tracked", "occluded", "lost"]] = (
        Field(None, description="Location track state for this source observation")
    )
    position_kind: Optional[str] = Field(
        None, description="Measured, estimated, or cached position semantics"
    )
    radius_unit: Optional[Literal["canonical_pixels"]] = Field(
        None, description="Unit of the compatibility radius field"
    )
    radius_source: Optional[str] = Field(
        None, description="Whether radius is measured or a compatibility assumption"
    )
    table_position: Optional["TablePositionData"] = Field(
        None, description="Calibrated position in physical table coordinates"
    )
    source_observation: Optional["SourceObservationData"] = Field(
        None, description="Original source identity and capture-stage timestamp"
    )
    image_space: Optional[Literal["raw", "rectified"]] = Field(
        None, description="Image coordinate space for the reported position"
    )
    image_width_px: Optional[int] = Field(None, gt=0)
    image_height_px: Optional[int] = Field(None, gt=0)

    @field_validator("radius_scale")
    @classmethod
    def validate_radius_scale(cls: type[Any], v: list[float]) -> list[float]:
        """Reject scales that cannot map the radius into canonical pixels."""
        return _validate_scale_factors(v)

    type: Optional[str] = Field(
        None, description="Ball type when established; unknown remains explicit"
    )


class TablePositionData(BaseModel):
    """Physical table-plane position, separate from image pixels."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., allow_inf_nan=False)
    y: float = Field(..., allow_inf_nan=False)
    unit: Literal["millimeters"]
    origin: Literal["playing_surface_center"]


class SourceObservationTimestamp(BaseModel):
    """Source clock evidence; ticks are not gateway wall-clock timestamps."""

    model_config = ConfigDict(extra="forbid")

    ticks: str = Field(..., pattern=r"^[0-9]+$")
    unit: Literal["nanoseconds"]
    clock: str
    domain_id: str
    stage: str


class SourceObservationData(BaseModel):
    """Frame identity and source timestamp for one ball location."""

    model_config = ConfigDict(extra="forbid")

    engine_instance_id: str
    session_id: str
    frame_id: str
    timestamp: SourceObservationTimestamp


class CueData(BaseModel):
    """Cue stick data structure."""

    angle: float = Field(..., description="Cue angle in degrees")
    position: PositionWithScale = Field(
        ..., description="Cue position in scaled image coordinates"
    )
    detected: bool = Field(True, description="Whether cue is detected")
    confidence: float = Field(1.0, ge=0, le=1, description="Detection confidence")
    length: Optional[float] = Field(
        None, gt=0, description="Detected cue length in pixels"
    )
    length_scale: Optional[list[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        validate_default=True,
        description="Scale factors for length in the source image axes",
    )
    tip_position: Optional[PositionWithScale] = Field(
        None, description="Cue tip position in scaled image coordinates"
    )
    butt_position: Optional[PositionWithScale] = Field(
        None, description="Measured cue butt endpoint in scaled image coordinates"
    )

    @field_validator("length_scale")
    @classmethod
    def validate_length_scale(
        cls: type[Any], value: Optional[list[float]], info: Any
    ) -> Optional[list[float]]:
        """Require scale whenever a pixel length is supplied."""
        if info.data.get("length") is not None and value is None:
            raise ValueError("length_scale is required when length is supplied")
        if value is not None:
            return _validate_scale_factors(value)
        return value


class RailSegmentData(BaseModel):
    """A table boundary segment with scaled endpoints."""

    start: PositionWithScale = Field(..., description="Rail start point")
    end: PositionWithScale = Field(..., description="Rail end point")


class TableDimensionsData(BaseModel):
    """Physical table dimensions, represented in meters rather than pixels."""

    width: float = Field(..., gt=0, allow_inf_nan=False)
    height: float = Field(..., gt=0, allow_inf_nan=False)
    unit: Literal["meters"] = Field(..., description="Dimension unit")


class TableData(BaseModel):
    """Pool table data structure."""

    corners: list[PositionWithScale] = Field(
        ..., min_length=4, max_length=4, description="Scaled table corner coordinates"
    )
    pockets: list[PositionWithScale] = Field(
        ..., description="Scaled pocket center coordinates"
    )
    rails: Optional[list[RailSegmentData]] = Field(
        None, description="Scaled rail segments"
    )
    calibrated: bool = Field(True, description="Whether table is properly calibrated")
    dimensions: Optional[TableDimensionsData] = Field(
        None, description="Real-world table dimensions in meters"
    )

    @field_validator("corners")
    @classmethod
    def validate_corners(
        cls: type[Any], v: list[PositionWithScale]
    ) -> list[PositionWithScale]:
        """Validate table has its four scaled boundary corners."""
        if len(v) != 4:
            raise ValueError("Table must have exactly 4 corners")
        return v


class GameStateData(BaseModel):
    """Complete game state data structure."""

    model_config = ConfigDict(extra="allow")

    balls: list[BallData] = Field(..., description="List of detected balls")
    cue: Optional[CueData] = Field(None, description="Cue stick data")
    table: Optional[TableData] = Field(None, description="Table geometry data")
    game_type: Optional[Literal["practice", "8ball", "9ball", "straight"]] = Field(
        None, description="Current billiards game mode"
    )
    ball_count: int = Field(..., ge=0, description="Total number of balls detected")
    frame_number: Optional[int] = Field(None, ge=0, description="Frame sequence number")
    observation_status: Optional[dict[str, Any]] = Field(
        None,
        description="Phase, map trust and source observation freshness from vision",
    )
    timestamp: Optional[datetime] = Field(
        None, description="Gateway publication time for this state message"
    )
    sequence: Optional[int] = Field(
        None, ge=0, description="State stream sequence number"
    )


class TrajectoryLine(BaseModel):
    """Individual trajectory line segment."""

    start: PositionWithScale = Field(..., description="Scaled line start point")
    end: PositionWithScale = Field(..., description="Scaled line end point")
    type: Literal["primary", "secondary", "reflection", "collision"] = Field(
        ..., description="Line type"
    )
    confidence: float = Field(1.0, ge=0, le=1, description="Prediction confidence")
    ball_id: Optional[str] = Field(
        None, description="Ball whose path this line follows"
    )


class CollisionData(BaseModel):
    """Collision prediction data using canonical participant identifiers."""

    model_config = ConfigDict(extra="forbid")

    position: PositionWithScale = Field(..., description="Scaled collision point")
    ball1_id: str = Field(..., description="ID of the primary ball involved")
    ball2_id: Optional[str] = Field(
        None,
        description="ID of the target ball; absent for cushion and pocket collisions",
    )
    type: Optional[str] = Field(None, description="Collision kind")
    time: Optional[float] = Field(
        None, ge=0, allow_inf_nan=False, description="Time to collision in seconds"
    )
    angle: float = Field(..., description="Collision angle in degrees")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    velocity_before: Optional[PositionWithScale] = Field(
        None, description="Scaled ball velocity before collision"
    )
    velocity_after: Optional[PositionWithScale] = Field(
        None, description="Scaled ball velocity after collision"
    )
    time_to_collision: Optional[float] = Field(
        None, ge=0, description="Time to collision in seconds"
    )


class TrajectoryData(BaseModel):
    """Trajectory prediction data structure."""

    model_config = ConfigDict(extra="forbid")

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
    timestamp: Optional[datetime] = Field(
        None, description="Publication timestamp for this trajectory"
    )
    sequence: Optional[int] = Field(None, description="Trajectory stream sequence")
    prediction_status: str = Field(
        "live", description="live, frozen_reference, or invalidated"
    )
    spatial_valid: bool = Field(
        True, description="Whether geometry matches the current calibration"
    )
    phase: Optional[str] = Field(None, description="Current vision processing phase")
    source_frame_number: Optional[int] = Field(
        None, ge=0, description="Source sequence that produced the prediction"
    )
    source_timestamp: Optional[float] = Field(
        None, description="Source publication timestamp, not camera exposure time"
    )
    source_timestamp_kind: Optional[str] = Field(
        None, description="Meaning/source of the publication timestamp"
    )
    map_revision: Optional[Any] = Field(None, description="Ball-map revision used")
    calibration_revision: Optional[Any] = Field(
        None, description="Calibration revision used"
    )
    prediction_settings: dict[str, Any] = Field(
        default_factory=dict, description="Settings used to calculate the prediction"
    )
    source_observations: dict[str, Any] = Field(
        default_factory=dict, description="Freshness/status of source observations"
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
    message_type_map: dict[str, type[BaseModel]] = {
        "subscribe": SubscribeRequest,
        "unsubscribe": UnsubscribeRequest,
        "ping": PingRequest,
        "get_status": StatusRequest,
    }

    if message_type not in message_type_map:
        raise ValueError(f"Unknown message type: {message_type}")

    schema_class = message_type_map[message_type]
    return schema_class.model_validate(data)


# Constants for validation
VALID_STREAM_TYPES = [stream_type.value for stream_type in StreamType]
VALID_ALERT_LEVELS = [level.value for level in AlertLevel]
VALID_QUALITY_LEVELS = [level.value for level in QualityLevel]
