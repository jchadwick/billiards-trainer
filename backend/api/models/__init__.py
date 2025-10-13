"""API Models Package.

Comprehensive Pydantic models for the Billiards Trainer API including:
- Request models for all API endpoints
- Response models with proper error handling
- WebSocket message schemas for real-time communication
- Model conversion utilities for backend integration
- Factory functions for testing and development
- JSON schema generation for documentation
- Example data for API documentation

All models provide type safety, validation, and OpenAPI schema support.
"""

# Import all model categories
from .common import *

# Import utilities
from .converters import (
    ball_state_to_ball_info,
    convert_ball_states_to_api,
    game_state_to_response,
    safe_convert_game_state,
    validate_ball_state_conversion,
    vector2d_to_coordinate2d,
)

# TODO: These modules don't exist yet - commenting out for now
# from .examples import (
#     EXAMPLES,
#     get_all_examples_for_model,
#     get_example_by_path,
#     list_available_examples,
# )
# from .factories import (
#     create_config_update_request,
#     create_frame_message,
#     create_game_state_response,
#     create_login_request,
#     create_test_dataset,
# )
from .requests import *
from .responses import *

# from .schema_generator import (
#     export_schemas_to_files,
#     generate_all_api_schemas,
#     generate_model_schema,
#     generate_openapi_schema,
# )
from .websocket import *

# Export comprehensive list of all models and utilities
__all__ = [
    # Request Models
    "BaseRequest",
    "TimestampedRequest",
    "LoginRequest",
    "TokenRefreshRequest",
    "ChangePasswordRequest",
    "CreateUserRequest",
    "ConfigUpdateRequest",
    "ConfigImportRequest",
    "ConfigExportRequest",
    "CalibrationStartRequest",
    "CalibrationPointRequest",
    "CalibrationApplyRequest",
    "CalibrationValidateRequest",
    "GameStateResetRequest",
    "BallPositionUpdateRequest",
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
    "LoginResponse",
    "TokenRefreshResponse",
    "UserCreateResponse",
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
    "MetricsMessage",
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
    "PerformanceMetrics",
    "GameEventData",
    "CommandData",
    "ResponseData",
    "ErrorData",
    # Common Models
    "BaseAPIModel",
    "Coordinate2D",
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
    "FieldError",
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
    # Factory Functions (TODO: Not implemented yet)
    # "create_login_request",
    # "create_config_update_request",
    # "create_game_state_response",
    # "create_frame_message",
    # "create_test_dataset",
    # Schema Generation (TODO: Not implemented yet)
    # "generate_model_schema",
    # "generate_all_api_schemas",
    # "generate_openapi_schema",
    # "export_schemas_to_files",
    # Examples (TODO: Not implemented yet)
    # "EXAMPLES",
    # "get_example_by_path",
    # "get_all_examples_for_model",
    # "list_available_examples",
    # Utility Functions
    "create_success_response",
    "validate_coordinate_bounds",
    "validate_confidence",
    "validate_websocket_message",
    "serialize_websocket_message",
    # TODO: These are from factories module which doesn't exist yet
    # "create_frame_message",
    # "create_alert_message",
    # "create_error_message",
]
