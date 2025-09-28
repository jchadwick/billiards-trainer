"""Common models and utilities for API endpoints."""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class ApiVersion(str, Enum):
    """API version enumeration."""

    V1 = "v1"


class StatusCode(int, Enum):
    """Extended HTTP status codes for API responses."""

    # Success codes
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204

    # Client error codes
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429

    # Server error codes
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class ErrorCode(str, Enum):
    """Application-specific error codes."""

    # Authentication/Authorization
    AUTH_INVALID_CREDENTIALS = "AUTH_001"
    AUTH_TOKEN_EXPIRED = "AUTH_002"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_003"

    # Validation
    VALIDATION_INVALID_FORMAT = "VAL_001"
    VALIDATION_MISSING_PARAMETER = "VAL_002"
    VALIDATION_OUT_OF_RANGE = "VAL_003"
    VALIDATION_INVALID_TYPE = "VAL_004"

    # Resource errors
    RESOURCE_NOT_FOUND = "RES_001"
    RESOURCE_ALREADY_EXISTS = "RES_002"
    RESOURCE_LOCKED = "RES_003"

    # System errors
    SYSTEM_INTERNAL_ERROR = "SYS_001"
    SYSTEM_UNAVAILABLE = "SYS_002"
    SYSTEM_OVERLOADED = "SYS_003"

    # Hardware/Device errors
    CAMERA_UNAVAILABLE = "CAM_001"
    CAMERA_INITIALIZATION_FAILED = "CAM_002"
    PROJECTOR_UNAVAILABLE = "PROJ_001"
    PROJECTOR_CALIBRATION_FAILED = "PROJ_002"

    # Processing errors
    VISION_PROCESSING_FAILED = "VIS_001"
    CALIBRATION_FAILED = "CAL_001"
    CALIBRATION_INCOMPLETE = "CAL_002"
    CALIBRATION_EXPIRED = "CAL_003"

    # Configuration errors
    CONFIG_INVALID = "CFG_001"
    CONFIG_READONLY = "CFG_002"
    CONFIG_BACKUP_FAILED = "CFG_003"

    # Game state errors
    GAME_INVALID_STATE = "GAME_001"
    GAME_OPERATION_NOT_ALLOWED = "GAME_002"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationResult(BaseModel):
    """Result of validation operation."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    field_errors: dict[str, list[str]] = Field(
        default_factory=dict, description="Field-specific errors"
    )


class PaginationRequest(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(1, description="Page number (1-based)", ge=1)
    page_size: int = Field(50, description="Items per page", ge=1, le=1000)
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", description="Sort order", pattern="^(asc|desc)$")

    @validator("sort_by")
    def validate_sort_field(cls, v):
        if v and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError("Sort field must be a valid identifier")
        return v


class PaginationResponse(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class TimeRange(BaseModel):
    """Time range specification."""

    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")

    @validator("end")
    def validate_time_range(cls, v, values):
        if "start" in values and v <= values["start"]:
            raise ValueError("End time must be after start time")
        return v


class Coordinate2D(BaseModel):
    """2D coordinate representation."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class BoundingBox(BaseModel):
    """Bounding box representation."""

    x: float = Field(..., description="Left edge X coordinate")
    y: float = Field(..., description="Top edge Y coordinate")
    width: float = Field(..., description="Box width", gt=0)
    height: float = Field(..., description="Box height", gt=0)


class ColorProfile(BaseModel):
    """Color profile specification."""

    name: str = Field(..., description="Profile name")
    hue_range: list[int] = Field(
        ..., description="HSV hue range [min, max]", min_items=2, max_items=2
    )
    saturation_range: list[int] = Field(
        ..., description="HSV saturation range [min, max]", min_items=2, max_items=2
    )
    value_range: list[int] = Field(
        ..., description="HSV value range [min, max]", min_items=2, max_items=2
    )

    @validator("hue_range", "saturation_range", "value_range")
    def validate_ranges(cls, v):
        if len(v) != 2:
            raise ValueError("Range must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("Range minimum must be less than maximum")
        return v


class SystemResources(BaseModel):
    """System resource utilization."""

    cpu_percent: float = Field(..., description="CPU usage percentage", ge=0, le=100)
    memory_percent: float = Field(
        ..., description="Memory usage percentage", ge=0, le=100
    )
    memory_used_mb: float = Field(..., description="Memory used in MB", ge=0)
    memory_total_mb: float = Field(..., description="Total memory in MB", ge=0)
    disk_percent: float = Field(..., description="Disk usage percentage", ge=0, le=100)
    disk_used_gb: float = Field(..., description="Disk used in GB", ge=0)
    disk_total_gb: float = Field(..., description="Total disk in GB", ge=0)


class NetworkStats(BaseModel):
    """Network statistics."""

    bytes_sent: int = Field(..., description="Total bytes sent", ge=0)
    bytes_received: int = Field(..., description="Total bytes received", ge=0)
    packets_sent: int = Field(..., description="Total packets sent", ge=0)
    packets_received: int = Field(..., description="Total packets received", ge=0)
    connections_active: int = Field(..., description="Active connections", ge=0)


class ProcessingStats(BaseModel):
    """Processing performance statistics."""

    frames_processed: int = Field(..., description="Total frames processed", ge=0)
    frames_per_second: float = Field(..., description="Current FPS", ge=0)
    average_processing_time: float = Field(
        ..., description="Average processing time per frame (ms)", ge=0
    )
    max_processing_time: float = Field(
        ..., description="Maximum processing time (ms)", ge=0
    )
    queue_size: int = Field(..., description="Current processing queue size", ge=0)
    dropped_frames: int = Field(..., description="Total dropped frames", ge=0)


class ApiMetrics(BaseModel):
    """API performance metrics."""

    total_requests: int = Field(..., description="Total API requests", ge=0)
    requests_per_second: float = Field(
        ..., description="Current requests per second", ge=0
    )
    average_response_time: float = Field(
        ..., description="Average response time (ms)", ge=0
    )
    error_rate: float = Field(..., description="Error rate percentage", ge=0, le=100)
    active_connections: int = Field(
        ..., description="Active WebSocket connections", ge=0
    )


class OperationStatus(str, Enum):
    """Operation status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Operation(BaseModel):
    """Asynchronous operation tracking."""

    operation_id: str = Field(..., description="Unique operation identifier")
    status: OperationStatus = Field(..., description="Current operation status")
    progress: float = Field(
        ..., description="Operation progress percentage", ge=0, le=100
    )
    message: str = Field(..., description="Current operation message")
    started_at: datetime = Field(..., description="Operation start time")
    completed_at: Optional[datetime] = Field(
        None, description="Operation completion time"
    )
    result: Optional[dict[str, Any]] = Field(None, description="Operation result data")
    error: Optional[str] = Field(None, description="Error message if failed")


def create_error_response(
    error_type: str,
    message: str,
    code: ErrorCode,
    details: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a standardized error response."""
    return {
        "error": error_type,
        "message": message,
        "code": code.value,
        "details": details or {},
        "timestamp": datetime.utcnow(),
        "request_id": request_id,
    }


def create_success_response(
    message: str, data: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Create a standardized success response."""
    return {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow(),
        "data": data or {},
    }
