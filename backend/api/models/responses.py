"""API Response Models.

This module defines all Pydantic models for API responses, including:
- Health check responses
- Configuration data responses
- Game state responses
- Error responses with proper error codes
- WebSocket message responses

All models include comprehensive validation, documentation, and OpenAPI schema support.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .common import PositionWithScale

# =============================================================================
# Base Response Models and Enums
# =============================================================================


class BaseResponse(BaseModel):
    """Base class for all API responses."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ApiVersion(str, Enum):
    """API version enumeration."""

    V1 = "v1"
    V2 = "v2"


class StatusCode(str, Enum):
    """Operation status codes."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    PENDING = "pending"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    """Standardized error codes."""

    # Authentication errors (AUTH_xxx)
    AUTH_INVALID_CREDENTIALS = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"
    AUTH_ACCOUNT_DISABLED = "AUTH_004"

    # Validation errors (VAL_xxx)
    VAL_INVALID_FORMAT = "VAL_001"
    VAL_MISSING_PARAMETER = "VAL_002"
    VAL_PARAMETER_OUT_OF_RANGE = "VAL_003"
    VAL_INVALID_CONFIGURATION = "VAL_004"

    # Resource errors (RES_xxx)
    RES_NOT_FOUND = "RES_001"
    RES_ALREADY_EXISTS = "RES_002"
    RES_ACCESS_DENIED = "RES_003"
    RES_LOCKED = "RES_004"

    # Rate limiting (RATE_xxx)
    RATE_LIMIT_EXCEEDED = "RATE_001"
    RATE_QUOTA_EXCEEDED = "RATE_002"

    # System errors (SYS_xxx)
    SYS_INTERNAL_ERROR = "SYS_001"
    SYS_SERVICE_UNAVAILABLE = "SYS_002"
    SYS_TIMEOUT = "SYS_003"
    SYS_MAINTENANCE = "SYS_004"

    # Hardware errors (HW_xxx)
    HW_CAMERA_UNAVAILABLE = "HW_001"
    HW_PROJECTOR_UNAVAILABLE = "HW_002"
    HW_CALIBRATION_FAILED = "HW_003"

    # Processing errors (PROC_xxx)
    PROC_VISION_FAILED = "PROC_001"
    PROC_ANALYSIS_FAILED = "PROC_002"
    PROC_TRACKING_LOST = "PROC_003"

    # WebSocket errors (WS_xxx)
    WS_CONNECTION_FAILED = "WS_001"
    WS_STREAM_UNAVAILABLE = "WS_002"
    WS_INVALID_MESSAGE = "WS_003"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


# =============================================================================
# Health Check Response Models
# =============================================================================


class ComponentHealth(BaseResponse):
    """Health information for a system component."""

    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Component health status")
    message: Optional[str] = Field(None, description="Status message")
    last_check: datetime = Field(..., description="Last health check timestamp")
    uptime: Optional[float] = Field(None, description="Component uptime in seconds")
    errors: list[str] = Field(
        default_factory=list, description="Current error messages"
    )


class SystemMetrics(BaseResponse):
    """System performance metrics."""

    cpu_usage: float = Field(..., description="CPU usage percentage", ge=0, le=100)
    memory_usage: float = Field(
        ..., description="Memory usage percentage", ge=0, le=100
    )
    disk_usage: float = Field(..., description="Disk usage percentage", ge=0, le=100)
    network_io: dict[str, float] = Field(
        default_factory=dict, description="Network I/O statistics"
    )
    api_requests_per_second: float = Field(..., description="API requests per second")
    websocket_connections: int = Field(..., description="Active WebSocket connections")
    average_response_time: float = Field(
        ..., description="Average API response time in ms"
    )


class HealthResponse(BaseResponse):
    """Health check response."""

    status: HealthStatus = Field(..., description="Overall system health")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict, description="Component health details"
    )
    metrics: Optional[SystemMetrics] = Field(None, description="Performance metrics")


class CapabilityInfo(BaseResponse):
    """System capability information."""

    vision_processing: bool = Field(..., description="Vision processing available")
    projector_support: bool = Field(..., description="Projector support available")
    calibration_modes: list[str] = Field(..., description="Available calibration modes")
    game_types: list[str] = Field(..., description="Supported game types")
    export_formats: list[str] = Field(..., description="Supported export formats")
    max_concurrent_sessions: int = Field(..., description="Maximum concurrent sessions")


class VersionResponse(BaseResponse):
    """Version and capability information response."""

    version: str = Field(..., description="System version")
    build_date: datetime = Field(..., description="Build date")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    capabilities: CapabilityInfo = Field(..., description="System capabilities")
    api_version: str = Field(..., description="API version")
    supported_clients: list[str] = Field(..., description="Supported client versions")


# =============================================================================
# Configuration Response Models
# =============================================================================


class ValidationResult(BaseResponse):
    """Generic validation result."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    field_errors: dict[str, list[str]] = Field(
        default_factory=dict, description="Field-specific errors"
    )


class ConfigValidationError(BaseResponse):
    """Configuration validation error."""

    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Error message")
    current_value: Any = Field(..., description="Current invalid value")
    expected_type: str = Field(..., description="Expected value type or format")


class ConfigResponse(BaseResponse):
    """Configuration response."""

    timestamp: datetime = Field(..., description="Configuration timestamp")
    values: dict[str, Any] = Field(..., description="Configuration values")
    schema_version: str = Field(..., description="Configuration schema version")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    is_valid: bool = Field(..., description="Configuration validity status")
    validation_errors: list[ConfigValidationError] = Field(
        default_factory=list, description="Validation errors"
    )


class ConfigUpdateResponse(BaseResponse):
    """Configuration update response."""

    success: bool = Field(..., description="Update success status")
    updated_fields: list[str] = Field(..., description="Fields that were updated")
    validation_errors: list[ConfigValidationError] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: list[str] = Field(default_factory=list, description="Update warnings")
    rollback_available: bool = Field(..., description="Whether rollback is available")
    restart_required: bool = Field(
        ..., description="Whether system restart is required"
    )


class ConfigExportResponse(BaseResponse):
    """Configuration export response."""

    format: str = Field(..., description="Export format")
    size: int = Field(..., description="Export size in bytes")
    checksum: str = Field(..., description="Export data checksum")
    timestamp: datetime = Field(..., description="Export timestamp")
    data: Union[dict[str, Any], str] = Field(
        ..., description="Exported configuration data"
    )


# =============================================================================
# Calibration Response Models
# =============================================================================


class CalibrationSession(BaseResponse):
    """Calibration session information."""

    session_id: str = Field(..., description="Unique session identifier")
    calibration_type: str = Field(..., description="Type of calibration")
    status: str = Field(..., description="Session status")
    created_at: datetime = Field(..., description="Session creation time")
    expires_at: datetime = Field(..., description="Session expiration time")
    points_captured: int = Field(..., description="Number of points captured")
    points_required: int = Field(..., description="Number of points required")
    accuracy: Optional[float] = Field(None, description="Current calibration accuracy")
    errors: list[str] = Field(default_factory=list, description="Calibration errors")


class CalibrationStartResponse(BaseResponse):
    """Calibration start response."""

    session: CalibrationSession = Field(..., description="Created calibration session")
    instructions: list[str] = Field(..., description="Calibration instructions")
    expected_points: int = Field(..., description="Number of points expected")
    timeout: int = Field(..., description="Session timeout in seconds")


class CalibrationPointResponse(BaseResponse):
    """Calibration point capture response."""

    success: bool = Field(..., description="Point capture success")
    point_id: str = Field(..., description="Captured point identifier")
    accuracy: float = Field(..., description="Point accuracy estimate")
    total_points: int = Field(..., description="Total points captured")
    remaining_points: int = Field(..., description="Points still needed")
    can_proceed: bool = Field(..., description="Whether calibration can proceed")


class CalibrationApplyResponse(BaseResponse):
    """Calibration apply response."""

    success: bool = Field(..., description="Apply operation success")
    accuracy: float = Field(..., description="Final calibration accuracy")
    backup_created: bool = Field(..., description="Whether backup was created")
    applied_at: datetime = Field(..., description="Application timestamp")
    transformation_matrix: list[list[float]] = Field(
        ..., description="Applied transformation matrix"
    )
    errors: list[str] = Field(default_factory=list, description="Application errors")


class CalibrationValidationResponse(BaseResponse):
    """Calibration validation response."""

    is_valid: bool = Field(..., description="Validation result")
    accuracy: float = Field(..., description="Measured accuracy")
    test_results: list[dict[str, Any]] = Field(
        ..., description="Individual test results"
    )
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


# =============================================================================
# Game State Response Models
# =============================================================================


class BallInfo(BaseResponse):
    """Ball information for game state.

    Note: position and velocity now use dict format with mandatory scale metadata.
    Format: {"x": float, "y": float, "scale": [sx, sy]}
    """

    id: str = Field(..., description="Ball identifier")
    number: Optional[int] = Field(None, description="Ball number")
    position: PositionWithScale = Field(
        ..., description="Ball position with scale metadata"
    )
    velocity: PositionWithScale = Field(
        ..., description="Ball velocity with scale metadata"
    )
    is_cue_ball: bool = Field(..., description="Whether this is the cue ball")
    is_pocketed: bool = Field(..., description="Whether ball is pocketed")
    confidence: float = Field(..., description="Detection confidence")
    last_update: datetime = Field(..., description="Last update timestamp")


class CueInfo(BaseResponse):
    """Cue stick information.

    Note: tip_position now uses dict format with mandatory scale metadata.
    Format: {"x": float, "y": float, "scale": [sx, sy]}
    """

    tip_position: PositionWithScale = Field(
        ..., description="Cue tip position with scale metadata"
    )
    angle: float = Field(..., description="Cue angle in degrees")
    elevation: float = Field(..., description="Cue elevation in degrees")
    estimated_force: float = Field(..., description="Estimated shot force")
    is_visible: bool = Field(..., description="Whether cue is visible")
    confidence: float = Field(..., description="Detection confidence")


class TableInfo(BaseResponse):
    """Table information.

    Note: pocket_positions now uses dict format with mandatory scale metadata.
    Each position has format: {"x": float, "y": float, "scale": [sx, sy]}
    """

    width: float = Field(..., description="Table width in meters")
    height: float = Field(..., description="Table height in meters")
    pocket_positions: list[PositionWithScale] = Field(
        ..., description="Pocket positions with scale metadata"
    )
    pocket_radius: float = Field(..., description="Pocket radius in meters")
    surface_friction: float = Field(..., description="Surface friction coefficient")


class GameEvent(BaseResponse):
    """Game event information."""

    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Type of event")
    description: str = Field(..., description="Event description")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Additional event data"
    )


class GameStateResponse(BaseResponse):
    """Game state response."""

    timestamp: datetime = Field(..., description="State timestamp")
    frame_number: int = Field(..., description="Frame number")
    balls: list[BallInfo] = Field(..., description="Ball states")
    cue: Optional[CueInfo] = Field(None, description="Cue stick state")
    table: TableInfo = Field(..., description="Table configuration")
    game_type: str = Field(..., description="Type of game")
    is_valid: bool = Field(..., description="State validity")
    confidence: float = Field(..., description="Overall state confidence")
    events: list[GameEvent] = Field(default_factory=list, description="Recent events")


class GameHistoryResponse(BaseResponse):
    """Game history response."""

    states: list[GameStateResponse] = Field(..., description="Historical game states")
    total_count: int = Field(..., description="Total available states")
    has_more: bool = Field(..., description="Whether more states are available")
    time_range: dict[str, datetime] = Field(
        ..., description="Actual time range of results"
    )


class GameResetResponse(BaseResponse):
    """Game reset response."""

    success: bool = Field(..., description="Reset operation success")
    new_state: GameStateResponse = Field(..., description="New game state after reset")
    backup_created: bool = Field(..., description="Whether backup was created")
    reset_at: datetime = Field(..., description="Reset timestamp")


# =============================================================================
# Data Export/Import Response Models
# =============================================================================


class SessionExportResponse(BaseResponse):
    """Session export response."""

    export_id: str = Field(..., description="Export operation identifier")
    format: str = Field(..., description="Export format")
    size: int = Field(..., description="Export size in bytes")
    file_path: str = Field(..., description="Export file path")
    checksum: str = Field(..., description="Export file checksum")
    created_at: datetime = Field(..., description="Export creation timestamp")
    expires_at: datetime = Field(..., description="Export expiration timestamp")


# =============================================================================
# System Control Response Models
# =============================================================================


class ShutdownResponse(BaseResponse):
    """Shutdown response."""

    acknowledged: bool = Field(..., description="Shutdown request acknowledged")
    scheduled_at: datetime = Field(..., description="Scheduled shutdown time")
    estimated_delay: int = Field(..., description="Estimated delay in seconds")
    active_operations: int = Field(..., description="Number of active operations")
    will_save_state: bool = Field(..., description="Whether state will be saved")


# =============================================================================
# Generic Response Models
# =============================================================================


class ErrorResponse(BaseResponse):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Error code")
    details: Optional[dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )


class SuccessResponse(BaseResponse):
    """Generic success response."""

    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(..., description="Operation timestamp")
    data: Optional[dict[str, Any]] = Field(None, description="Additional response data")


# =============================================================================
# Hardware Control Response Models
# =============================================================================


class CameraControlResponse(BaseResponse):
    """Camera control operation response."""

    success: bool = Field(..., description="Operation success status")
    action: str = Field(..., description="Performed action")
    camera_id: str = Field(..., description="Camera identifier")
    current_settings: dict[str, Any] = Field(..., description="Current camera settings")
    timestamp: datetime = Field(..., description="Operation timestamp")
    errors: list[str] = Field(default_factory=list, description="Operation errors")


class ProjectorControlResponse(BaseResponse):
    """Projector control operation response."""

    success: bool = Field(..., description="Operation success status")
    action: str = Field(..., description="Performed action")
    projector_id: str = Field(..., description="Projector identifier")
    current_settings: dict[str, Any] = Field(
        ..., description="Current projector settings"
    )
    display_content: Optional[str] = Field(
        None, description="Currently displayed content type"
    )
    timestamp: datetime = Field(..., description="Operation timestamp")
    errors: list[str] = Field(default_factory=list, description="Operation errors")


# =============================================================================
# Authentication Response Models
# =============================================================================


class LoginResponse(BaseResponse):
    """User login response."""

    success: bool = Field(..., description="Login success status")
    access_token: Optional[str] = Field(None, description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    expires_in: Optional[int] = Field(
        None, description="Token expiration time in seconds"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    role: Optional[str] = Field(None, description="User role")
    permissions: list[str] = Field(default_factory=list, description="User permissions")
    timestamp: datetime = Field(..., description="Login timestamp")


class TokenRefreshResponse(BaseResponse):
    """Token refresh response."""

    success: bool = Field(..., description="Refresh success status")
    access_token: Optional[str] = Field(None, description="New JWT access token")
    expires_in: Optional[int] = Field(
        None, description="Token expiration time in seconds"
    )
    timestamp: datetime = Field(..., description="Refresh timestamp")


class UserCreateResponse(BaseResponse):
    """User creation response."""

    success: bool = Field(..., description="Creation success status")
    user_id: str = Field(..., description="Created user identifier")
    username: str = Field(..., description="Created username")
    role: str = Field(..., description="Assigned role")
    created_at: datetime = Field(..., description="Creation timestamp")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors"
    )


# =============================================================================
# Advanced Game Analysis Response Models
# =============================================================================


class TrajectoryInfo(BaseResponse):
    """Ball trajectory information.

    Note: points now uses dict format with mandatory scale metadata.
    Each point has format: {"x": float, "y": float, "scale": [sx, sy]}
    """

    ball_id: str = Field(..., description="Ball identifier")
    points: list[PositionWithScale] = Field(
        ..., description="Trajectory points with scale metadata"
    )
    will_be_pocketed: bool = Field(..., description="Whether ball will be pocketed")
    pocket_id: Optional[int] = Field(None, description="Target pocket ID")
    time_to_rest: float = Field(..., description="Time until ball stops")
    max_velocity: float = Field(..., description="Maximum velocity during trajectory")
    confidence: float = Field(..., description="Trajectory prediction confidence")


class ShotAnalysisResponse(BaseResponse):
    """Shot analysis response."""

    shot_type: str = Field(..., description="Type of shot")
    difficulty: float = Field(..., description="Shot difficulty (0-1)")
    success_probability: float = Field(..., description="Success probability (0-1)")
    recommended_force: float = Field(..., description="Recommended shot force")
    recommended_angle: float = Field(..., description="Recommended shot angle")
    target_ball_id: Optional[str] = Field(None, description="Target ball ID")
    target_pocket_id: Optional[int] = Field(None, description="Target pocket ID")
    potential_problems: list[str] = Field(
        default_factory=list, description="Potential shot problems"
    )
    risk_assessment: dict[str, float] = Field(
        default_factory=dict, description="Risk factors"
    )
    trajectories: list[TrajectoryInfo] = Field(
        default_factory=list, description="Predicted trajectories"
    )


# =============================================================================
# Data Processing Response Models
# =============================================================================


class DataImportResponse(BaseResponse):
    """Data import operation response."""

    success: bool = Field(..., description="Import success status")
    import_id: str = Field(..., description="Import operation identifier")
    data_type: str = Field(..., description="Type of imported data")
    records_processed: int = Field(..., description="Number of records processed")
    records_imported: int = Field(
        ..., description="Number of records successfully imported"
    )
    records_rejected: int = Field(..., description="Number of records rejected")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: list[str] = Field(default_factory=list, description="Import warnings")
    timestamp: datetime = Field(..., description="Import timestamp")


class DataExportResponse(BaseResponse):
    """Data export operation response."""

    success: bool = Field(..., description="Export success status")
    export_id: str = Field(..., description="Export operation identifier")
    data_type: str = Field(..., description="Type of exported data")
    format: str = Field(..., description="Export format")
    file_path: str = Field(..., description="Export file path")
    file_size: int = Field(..., description="Export file size in bytes")
    record_count: int = Field(..., description="Number of records exported")
    checksum: str = Field(..., description="File checksum")
    expires_at: datetime = Field(..., description="Export expiration time")
    timestamp: datetime = Field(..., description="Export timestamp")


# =============================================================================
# System Monitoring Response Models
# =============================================================================


class SystemResourcesResponse(BaseResponse):
    """System resources information."""

    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    available_memory: int = Field(..., description="Available memory in bytes")
    available_disk: int = Field(..., description="Available disk space in bytes")
    network_stats: dict[str, Any] = Field(
        default_factory=dict, description="Network statistics"
    )
    timestamp: datetime = Field(..., description="Measurement timestamp")


class ProcessingStatsResponse(BaseResponse):
    """Processing performance statistics."""

    frames_per_second: float = Field(..., description="Processing FPS")
    average_processing_time: float = Field(
        ..., description="Average processing time in ms"
    )
    queue_length: int = Field(..., description="Processing queue length")
    dropped_frames: int = Field(..., description="Number of dropped frames")
    error_rate: float = Field(..., description="Processing error rate")
    accuracy_metrics: dict[str, float] = Field(
        default_factory=dict, description="Accuracy metrics"
    )
    timestamp: datetime = Field(..., description="Statistics timestamp")


class ApiMetricsResponse(BaseResponse):
    """API performance metrics."""

    requests_per_second: float = Field(..., description="Requests per second")
    average_response_time: float = Field(..., description="Average response time in ms")
    error_rate: float = Field(..., description="Error rate percentage")
    active_connections: int = Field(..., description="Active WebSocket connections")
    total_requests: int = Field(..., description="Total requests since startup")
    endpoint_stats: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-endpoint statistics"
    )
    timestamp: datetime = Field(..., description="Metrics timestamp")


# =============================================================================
# WebSocket Connection Response Models
# =============================================================================


class WebSocketConnectionResponse(BaseResponse):
    """WebSocket connection establishment response."""

    success: bool = Field(..., description="Connection success status")
    connection_id: str = Field(..., description="WebSocket connection identifier")
    available_streams: list[str] = Field(..., description="Available data streams")
    supported_qualities: list[str] = Field(..., description="Supported quality levels")
    max_frame_rate: int = Field(..., description="Maximum frame rate")
    timestamp: datetime = Field(..., description="Connection timestamp")


class WebSocketSubscriptionResponse(BaseResponse):
    """WebSocket subscription response."""

    success: bool = Field(..., description="Subscription success status")
    subscribed_streams: list[str] = Field(
        ..., description="Successfully subscribed streams"
    )
    failed_streams: list[str] = Field(
        default_factory=list, description="Failed stream subscriptions"
    )
    quality: str = Field(..., description="Active quality level")
    frame_rate: int = Field(..., description="Active frame rate")
    timestamp: datetime = Field(..., description="Subscription timestamp")


# =============================================================================
# Camera Fisheye Calibration Response Models
# =============================================================================


class CameraCalibrationModeResponse(BaseResponse):
    """Camera calibration mode start/stop response."""

    success: bool = Field(..., description="Operation success status")
    mode_active: bool = Field(..., description="Whether calibration mode is active")
    message: str = Field(..., description="Status message")
    chessboard_size: tuple[int, int] = Field(
        default=(9, 6), description="Chessboard internal corners (cols, rows)"
    )
    square_size: float = Field(
        default=0.025, description="Chessboard square size in meters"
    )
    min_images: int = Field(
        default=10, description="Minimum images required for calibration"
    )
    instructions: list[str] = Field(
        default_factory=list, description="Calibration instructions"
    )


class CameraImageCaptureResponse(BaseResponse):
    """Camera calibration image capture response."""

    success: bool = Field(..., description="Capture success status")
    image_id: str = Field(..., description="Unique image identifier")
    chessboard_found: bool = Field(..., description="Whether chessboard was detected")
    corners_detected: int = Field(
        default=0, description="Number of corners detected in image"
    )
    total_images: int = Field(..., description="Total images captured so far")
    images_required: int = Field(..., description="Minimum images required")
    can_process: bool = Field(
        ..., description="Whether enough images have been captured to process"
    )
    message: str = Field(..., description="Status message")
    image_preview: Optional[str] = Field(
        None, description="Base64 encoded preview image (optional)"
    )


class CameraCalibrationProcessResponse(BaseResponse):
    """Camera calibration processing response."""

    success: bool = Field(..., description="Calibration success status")
    calibration_error: float = Field(
        ..., description="Mean reprojection error in pixels"
    )
    images_used: int = Field(..., description="Number of images used in calibration")
    camera_matrix: list[list[float]] = Field(
        ..., description="Camera intrinsic matrix (3x3)"
    )
    distortion_coefficients: list[float] = Field(
        ..., description="Distortion coefficients"
    )
    resolution: tuple[int, int] = Field(
        ..., description="Calibration resolution (w, h)"
    )
    saved_to: str = Field(..., description="Path where calibration was saved")
    quality_rating: str = Field(
        ..., description="Quality rating: excellent, good, fair, poor"
    )
    message: str = Field(..., description="Status message")


class CameraCalibrationApplyResponse(BaseResponse):
    """Camera calibration apply response."""

    success: bool = Field(..., description="Apply success status")
    calibration_loaded: bool = Field(
        ..., description="Whether calibration was successfully loaded"
    )
    fisheye_correction_enabled: bool = Field(
        ..., description="Whether fisheye correction is now enabled"
    )
    message: str = Field(..., description="Status message")


# =============================================================================
# Utility Functions
# =============================================================================


def create_success_response(
    message: str, data: Optional[dict[str, Any]] = None
) -> SuccessResponse:
    """Create a standardized success response."""
    return SuccessResponse(
        success=True, message=message, timestamp=datetime.now(), data=data or {}
    )
