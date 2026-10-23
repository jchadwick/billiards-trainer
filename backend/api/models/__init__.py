"""API Models Package.

Comprehensive Pydantic models for the Billiards Trainer API including:
- Request models for all API endpoints
- Response models with proper error handling
- WebSocket message schemas for real-time communication
- Model conversion utilities for backend integration

All models provide type safety, validation, and OpenAPI schema support.
"""

# Import common models that are not superseded by the response-layer contract.
# The similarly named response enums and helpers below remain the package-level
# versions, matching the historical wildcard import order.
from .common import (
    ApiMetrics,
    BoundingBox,
    ColorProfile,
    Coordinate2D,
    NetworkStats,
    Operation,
    OperationStatus,
    PaginationRequest,
    PaginationResponse,
    PositionWithScale,
    ProcessingStats,
    SystemResources,
    TimeRange,
    WorldPoint2DModel,
)

# Import utilities
from .converters import (
    ball_state_to_ball_info,
    convert_ball_states_to_api,
    game_state_to_response,
    safe_convert_game_state,
    validate_ball_state_conversion,
    vector2d_to_coordinate2d,
)
from .requests import *
from .responses import *
from .websocket import *

# Export comprehensive list of all models and utilities
__all__ = [
    # Request Models
    "BaseRequest",
    "TimestampedRequest",
    "ConfigUpdateRequest",
    "ConfigImportRequest",
    "ConfigExportRequest",
    "CalibrationStartRequest",
    "CalibrationPointRequest",
    "CalibrationTestPoint",
    "CalibrationApplyRequest",
    "CalibrationValidateRequest",
    "GameStateResetRequest",
    "BallPositionUpdateRequest",
    "BallPositionUpdate",
    "GameEventRequest",
    "SystemControlRequest",
    "CameraControlRequest",
    "ProjectorControlRequest",
    "DataExportRequest",
    "DataImportRequest",
    "WebSocketSubscribeRequest",
    "WebSocketUnsubscribeRequest",
    # Response Models
    "BaseResponse",
    "HealthResponse",
    "VersionResponse",
    "ConfigResponse",
    "ConfigUpdateResponse",
    "ConfigExportResponse",
    "CalibrationStartResponse",
    "CalibrationPointResponse",
    "CalibrationApplyResponse",
    "CalibrationValidationResponse",
    "GameStateResponse",
    "GameHistoryResponse",
    "GameResetResponse",
    "SessionExportResponse",
    "ShutdownResponse",
    "ErrorResponse",
    "SuccessResponse",
    "BallInfo",
    "CueInfo",
    "TableInfo",
    "CameraControlResponse",
    "ProjectorControlResponse",
    "TrajectoryInfo",
    "ShotAnalysisResponse",
    "DataImportResponse",
    "DataExportResponse",
    "SystemResourcesResponse",
    "ProcessingStatsResponse",
    "ApiMetricsResponse",
    "WebSocketConnectionResponse",
    "WebSocketSubscriptionResponse",
    # WebSocket Message Models
    "BaseWebSocketMessage",
    "FrameMessage",
    "StateMessage",
    "TrajectoryMessage",
    "AlertMessage",
    "ConfigMessage",
    "EventMessage",
    "CommandMessage",
    "ResponseMessage",
    "HeartbeatMessage",
    "ErrorMessage",
    "FrameData",
    "FrameMetadata",
    "BallStateData",
    "CueStateData",
    "TableStateData",
    "GameStateData",
    "TrajectoryData",
    "AlertData",
    "GameEventData",
    "CommandData",
    "ResponseData",
    "ErrorData",
    # Common Models
    "Coordinate2D",
    "PositionWithScale",
    "WorldPoint2DModel",
    "BoundingBox",
    "ColorProfile",
    "PaginationRequest",
    "PaginationResponse",
    "TimeRange",
    "SystemResources",
    "NetworkStats",
    "ProcessingStats",
    "ApiMetrics",
    "Operation",
    "ValidationResult",
    # Enums
    "HealthStatus",
    "ApiVersion",
    "StatusCode",
    "ErrorCode",
    "LogLevel",
    "OperationStatus",
    "MessageType",
    "MessagePriority",
    "AlertLevel",
    "StreamQuality",
    # Conversion Utilities
    "vector2d_to_coordinate2d",
    "ball_state_to_ball_info",
    "game_state_to_response",
    "convert_ball_states_to_api",
    "validate_ball_state_conversion",
    "safe_convert_game_state",
    # Utility Functions
    "create_success_response",
    "validate_coordinate_bounds",
    "validate_confidence",
    "validate_websocket_message",
    "serialize_websocket_message",
]
