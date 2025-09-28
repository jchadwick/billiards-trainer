"""Vision-related API models for data transformation."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class CameraStatusEnum(str, Enum):
    """Camera status."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    CAPTURING = "capturing"
    ERROR = "error"


class DetectionTypeEnum(str, Enum):
    """Detection types."""

    TABLE = "table"
    BALLS = "balls"
    CUE = "cue"
    POCKETS = "pockets"


class CalibrationTypeEnum(str, Enum):
    """Calibration types."""

    CAMERA = "camera"
    COLOR = "color"
    GEOMETRY = "geometry"
    DISTORTION = "distortion"


class BallTypeEnum(str, Enum):
    """Ball types for vision detection."""

    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


class CueStateEnum(str, Enum):
    """Cue stick states."""

    NOT_DETECTED = "not_detected"
    DETECTED = "detected"
    AIMING = "aiming"
    STRIKING = "striking"


# Basic geometry models


class Point2DModel(BaseModel):
    """2D point model for vision coordinates."""

    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")


class BoundingBoxModel(BaseModel):
    """Bounding box model."""

    x: float = Field(..., description="Top-left X coordinate")
    y: float = Field(..., description="Top-left Y coordinate")
    width: float = Field(..., ge=0, description="Box width")
    height: float = Field(..., ge=0, description="Box height")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence"
    )


class ContourModel(BaseModel):
    """Contour model for shape detection."""

    points: list[Point2DModel] = Field(..., description="Contour points")
    area: float = Field(..., ge=0, description="Contour area")
    perimeter: float = Field(..., ge=0, description="Contour perimeter")
    center: Point2DModel = Field(..., description="Contour center")


# Vision detection models


class VisionBallModel(BaseModel):
    """Ball model for vision detection."""

    id: str = Field(..., description="Ball identifier")
    center: Point2DModel = Field(..., description="Ball center in pixels")
    radius: float = Field(..., gt=0, description="Ball radius in pixels")
    color: tuple[int, int, int] = Field(..., description="Detected ball color (RGB)")
    ball_type: BallTypeEnum = Field(
        default=BallTypeEnum.UNKNOWN, description="Ball type"
    )
    number: Optional[int] = Field(None, ge=0, le=15, description="Ball number")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: BoundingBoxModel = Field(..., description="Ball bounding box")
    contour: Optional[ContourModel] = Field(None, description="Ball contour")
    is_moving: bool = Field(default=False, description="Whether ball is in motion")
    velocity_pixels: Optional[Point2DModel] = Field(
        None, description="Velocity in pixels/frame"
    )


class VisionPocketModel(BaseModel):
    """Pocket model for vision detection."""

    id: str = Field(..., description="Pocket identifier")
    center: Point2DModel = Field(..., description="Pocket center in pixels")
    radius: float = Field(..., gt=0, description="Pocket radius in pixels")
    type: str = Field(default="corner", description="Pocket type (corner/side)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: BoundingBoxModel = Field(..., description="Pocket bounding box")


class VisionTableModel(BaseModel):
    """Table model for vision detection."""

    corners: list[Point2DModel] = Field(
        ..., min_items=4, max_items=4, description="Table corner points"
    )
    rails: list[list[Point2DModel]] = Field(default=[], description="Rail boundaries")
    pockets: list[VisionPocketModel] = Field(default=[], description="Detected pockets")
    surface_color: tuple[int, int, int] = Field(
        ..., description="Table surface color (RGB)"
    )
    width_pixels: float = Field(..., gt=0, description="Table width in pixels")
    height_pixels: float = Field(..., gt=0, description="Table height in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    transform_matrix: Optional[list[list[float]]] = Field(
        None, description="Perspective transform matrix"
    )


class VisionCueModel(BaseModel):
    """Cue stick model for vision detection."""

    tip_position: Point2DModel = Field(..., description="Cue tip position in pixels")
    end_position: Point2DModel = Field(..., description="Cue end position in pixels")
    angle: float = Field(..., description="Cue angle in radians")
    length_pixels: float = Field(..., gt=0, description="Cue length in pixels")
    state: CueStateEnum = Field(default=CueStateEnum.DETECTED, description="Cue state")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bounding_box: BoundingBoxModel = Field(..., description="Cue bounding box")
    is_visible: bool = Field(default=True, description="Whether cue is fully visible")


class DetectionResultModel(BaseModel):
    """Complete detection result model."""

    frame_number: int = Field(..., ge=0, description="Frame sequence number")
    timestamp: float = Field(..., description="Detection timestamp")
    processing_time: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    balls: list[VisionBallModel] = Field(default=[], description="Detected balls")
    table: Optional[VisionTableModel] = Field(None, description="Detected table")
    cue: Optional[VisionCueModel] = Field(None, description="Detected cue")
    frame_width: int = Field(..., gt=0, description="Frame width in pixels")
    frame_height: int = Field(..., gt=0, description="Frame height in pixels")
    detection_quality: dict[str, float] = Field(
        default={}, description="Quality metrics per detection type"
    )
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata")


# Camera and configuration models


class CameraConfigModel(BaseModel):
    """Camera configuration model."""

    device_id: int = Field(default=0, ge=0, description="Camera device ID")
    resolution: tuple[int, int] = Field(
        default=(1920, 1080), description="Camera resolution (width, height)"
    )
    fps: int = Field(default=30, gt=0, le=120, description="Frames per second")
    exposure: Optional[float] = Field(None, description="Camera exposure")
    gain: Optional[float] = Field(None, description="Camera gain")
    white_balance: Optional[int] = Field(None, description="White balance setting")
    focus: Optional[float] = Field(None, description="Focus setting")
    auto_exposure: bool = Field(default=True, description="Auto exposure enabled")
    auto_white_balance: bool = Field(
        default=True, description="Auto white balance enabled"
    )
    buffer_size: int = Field(default=1, ge=1, le=10, description="Frame buffer size")


class ColorThresholdModel(BaseModel):
    """Color threshold model for detection."""

    name: str = Field(..., description="Color name")
    hsv_lower: tuple[int, int, int] = Field(..., description="Lower HSV threshold")
    hsv_upper: tuple[int, int, int] = Field(..., description="Upper HSV threshold")
    rgb_reference: tuple[int, int, int] = Field(..., description="Reference RGB color")
    tolerance: float = Field(default=0.1, ge=0.0, le=1.0, description="Color tolerance")


class VisionConfigModel(BaseModel):
    """Vision module configuration model."""

    camera: CameraConfigModel = Field(
        default=CameraConfigModel(), description="Camera configuration"
    )
    detection_enabled: dict[str, bool] = Field(
        default={"table": True, "balls": True, "cue": True, "pockets": True},
        description="Enabled detection types",
    )
    color_thresholds: list[ColorThresholdModel] = Field(
        default=[], description="Color detection thresholds"
    )
    preprocessing: dict[str, Any] = Field(
        default={}, description="Image preprocessing settings"
    )
    tracking_enabled: bool = Field(default=True, description="Object tracking enabled")
    roi_enabled: bool = Field(default=False, description="Region of interest enabled")
    debug_mode: bool = Field(default=False, description="Debug mode enabled")
    performance_mode: str = Field(
        default="balanced", description="Performance mode (fast/balanced/accurate)"
    )


# Calibration models


class CalibrationPointModel(BaseModel):
    """Calibration point model."""

    id: str = Field(..., description="Point identifier")
    image_point: Point2DModel = Field(..., description="Point in image coordinates")
    world_point: Optional[Point2DModel] = Field(
        None, description="Point in world coordinates"
    )
    is_valid: bool = Field(default=True, description="Whether point is valid")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Point confidence"
    )


class CameraCalibrationModel(BaseModel):
    """Camera calibration model."""

    intrinsic_matrix: list[list[float]] = Field(
        ..., description="Camera intrinsic matrix"
    )
    distortion_coefficients: list[float] = Field(
        ..., description="Distortion coefficients"
    )
    resolution: tuple[int, int] = Field(..., description="Calibration resolution")
    reprojection_error: float = Field(..., ge=0, description="Reprojection error")
    calibration_date: datetime = Field(
        default_factory=datetime.now, description="Calibration timestamp"
    )
    is_valid: bool = Field(default=True, description="Whether calibration is valid")


class GeometricCalibrationModel(BaseModel):
    """Geometric calibration model."""

    homography_matrix: list[list[float]] = Field(
        ..., description="Homography transformation matrix"
    )
    table_corners_image: list[Point2DModel] = Field(
        ..., description="Table corners in image coordinates"
    )
    table_corners_world: list[Point2DModel] = Field(
        ..., description="Table corners in world coordinates"
    )
    pixels_per_meter: float = Field(..., gt=0, description="Pixels per meter ratio")
    calibration_error: float = Field(
        ..., ge=0, description="Geometric calibration error"
    )
    calibration_date: datetime = Field(
        default_factory=datetime.now, description="Calibration timestamp"
    )
    is_valid: bool = Field(default=True, description="Whether calibration is valid")


class ColorCalibrationModel(BaseModel):
    """Color calibration model."""

    ball_colors: dict[str, ColorThresholdModel] = Field(
        default={}, description="Ball color thresholds"
    )
    table_color: ColorThresholdModel = Field(
        ..., description="Table surface color threshold"
    )
    lighting_conditions: str = Field(
        default="normal", description="Lighting conditions"
    )
    calibration_date: datetime = Field(
        default_factory=datetime.now, description="Calibration timestamp"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Calibration quality score"
    )
    is_valid: bool = Field(default=True, description="Whether calibration is valid")


class CalibrationDataModel(BaseModel):
    """Complete calibration data model."""

    camera: Optional[CameraCalibrationModel] = Field(
        None, description="Camera calibration"
    )
    geometry: Optional[GeometricCalibrationModel] = Field(
        None, description="Geometric calibration"
    )
    color: Optional[ColorCalibrationModel] = Field(
        None, description="Color calibration"
    )
    version: str = Field(default="1.0", description="Calibration data version")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# ROI and area models


class ROIModel(BaseModel):
    """Region of Interest model."""

    name: str = Field(..., description="ROI name")
    corners: list[Point2DModel] = Field(
        ..., min_items=3, description="ROI corner points"
    )
    is_active: bool = Field(default=True, description="Whether ROI is active")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )


# Statistics and monitoring models


class VisionStatisticsModel(BaseModel):
    """Vision processing statistics."""

    frames_processed: int = Field(default=0, ge=0, description="Total frames processed")
    frames_dropped: int = Field(default=0, ge=0, description="Total frames dropped")
    average_fps: float = Field(
        default=0.0, ge=0, description="Average frames per second"
    )
    average_processing_time: float = Field(
        default=0.0, ge=0, description="Average processing time (ms)"
    )
    detection_accuracy: dict[str, float] = Field(
        default={}, description="Detection accuracy per type"
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0, description="Module uptime in seconds"
    )
    last_error: Optional[str] = Field(None, description="Last error message")
    camera_status: CameraStatusEnum = Field(
        default=CameraStatusEnum.DISCONNECTED, description="Camera status"
    )
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class FrameStatisticsModel(BaseModel):
    """Individual frame statistics."""

    frame_number: int = Field(..., ge=0, description="Frame number")
    timestamp: float = Field(..., description="Frame timestamp")
    processing_time: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    detections_count: dict[str, int] = Field(
        default={}, description="Number of detections per type"
    )
    quality_scores: dict[str, float] = Field(
        default={}, description="Quality scores per detection type"
    )
    frame_size_bytes: Optional[int] = Field(None, description="Frame size in bytes")


# Request models


class CameraConfigRequest(BaseModel):
    """Request to update camera configuration."""

    device_id: Optional[int] = Field(None, ge=0, description="Camera device ID")
    resolution: Optional[tuple[int, int]] = Field(None, description="Camera resolution")
    fps: Optional[int] = Field(None, gt=0, le=120, description="Frames per second")
    exposure: Optional[float] = Field(None, description="Camera exposure")
    gain: Optional[float] = Field(None, description="Camera gain")
    auto_exposure: Optional[bool] = Field(None, description="Auto exposure enabled")


class CalibrationRequest(BaseModel):
    """Request to perform calibration."""

    calibration_type: CalibrationTypeEnum = Field(
        ..., description="Type of calibration to perform"
    )
    sample_frames: int = Field(
        default=10, ge=1, le=50, description="Number of sample frames"
    )
    auto_detect: bool = Field(
        default=True, description="Auto-detect calibration points"
    )
    manual_points: Optional[list[CalibrationPointModel]] = Field(
        None, description="Manual calibration points"
    )
    save_results: bool = Field(default=True, description="Save calibration results")


class ROIRequest(BaseModel):
    """Request to set region of interest."""

    name: str = Field(..., description="ROI name")
    corners: list[Point2DModel] = Field(
        ..., min_items=3, description="ROI corner points"
    )
    replace_existing: bool = Field(default=True, description="Replace existing ROI")


class DetectionConfigRequest(BaseModel):
    """Request to update detection configuration."""

    detection_types: Optional[dict[str, bool]] = Field(
        None, description="Enable/disable detection types"
    )
    sensitivity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Detection sensitivity"
    )
    color_thresholds: Optional[list[ColorThresholdModel]] = Field(
        None, description="Color thresholds"
    )
    tracking_enabled: Optional[bool] = Field(
        None, description="Object tracking enabled"
    )
    debug_mode: Optional[bool] = Field(None, description="Debug mode enabled")


# Response models


class CameraStatusResponse(BaseModel):
    """Camera status response."""

    success: bool = Field(default=True, description="Operation success")
    status: CameraStatusEnum = Field(..., description="Camera status")
    configuration: CameraConfigModel = Field(
        ..., description="Current camera configuration"
    )
    capabilities: dict[str, Any] = Field(default={}, description="Camera capabilities")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class DetectionResponse(BaseModel):
    """Detection result response."""

    success: bool = Field(default=True, description="Operation success")
    result: DetectionResultModel = Field(..., description="Detection results")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class CalibrationResultModel(BaseModel):
    """Calibration result response model."""

    success: bool = Field(..., description="Calibration success")
    calibration_type: CalibrationTypeEnum = Field(
        ..., description="Type of calibration performed"
    )
    data: Optional[CalibrationDataModel] = Field(None, description="Calibration data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    quality_metrics: dict[str, float] = Field(default={}, description="Quality metrics")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")


class VisionConfigResponse(BaseModel):
    """Vision configuration response."""

    success: bool = Field(default=True, description="Operation success")
    configuration: VisionConfigModel = Field(..., description="Vision configuration")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class VisionStatisticsResponse(BaseModel):
    """Vision statistics response."""

    success: bool = Field(default=True, description="Operation success")
    statistics: VisionStatisticsModel = Field(
        ..., description="Vision processing statistics"
    )
    frame_history: list[FrameStatisticsModel] = Field(
        default=[], description="Recent frame statistics"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ROIResponse(BaseModel):
    """ROI operation response."""

    success: bool = Field(default=True, description="Operation success")
    roi: ROIModel = Field(..., description="Region of interest")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


# Validators


@validator("corners")
def validate_corners(cls, v):
    """Validate corner points."""
    if len(v) < 3:
        raise ValueError("Must have at least 3 corner points")
    return v


@validator("hsv_lower", "hsv_upper")
def validate_hsv_values(cls, v):
    """Validate HSV values."""
    if len(v) != 3:
        raise ValueError("HSV values must have exactly 3 components")
    h, s, v_val = v
    if not (0 <= h <= 179):
        raise ValueError("Hue must be between 0 and 179")
    if not (0 <= s <= 255):
        raise ValueError("Saturation must be between 0 and 255")
    if not (0 <= v_val <= 255):
        raise ValueError("Value must be between 0 and 255")
    return v


@validator("rgb_reference")
def validate_rgb_values(cls, v):
    """Validate RGB values."""
    if len(v) != 3:
        raise ValueError("RGB values must have exactly 3 components")
    if not all(0 <= c <= 255 for c in v):
        raise ValueError("RGB values must be between 0 and 255")
    return v


__all__ = [
    # Enums
    "CameraStatusEnum",
    "DetectionTypeEnum",
    "CalibrationTypeEnum",
    "BallTypeEnum",
    "CueStateEnum",
    # Basic models
    "Point2DModel",
    "BoundingBoxModel",
    "ContourModel",
    # Detection models
    "VisionBallModel",
    "VisionPocketModel",
    "VisionTableModel",
    "VisionCueModel",
    "DetectionResultModel",
    # Configuration models
    "CameraConfigModel",
    "ColorThresholdModel",
    "VisionConfigModel",
    # Calibration models
    "CalibrationPointModel",
    "CameraCalibrationModel",
    "GeometricCalibrationModel",
    "ColorCalibrationModel",
    "CalibrationDataModel",
    # ROI and area models
    "ROIModel",
    # Statistics models
    "VisionStatisticsModel",
    "FrameStatisticsModel",
    # Request models
    "CameraConfigRequest",
    "CalibrationRequest",
    "ROIRequest",
    "DetectionConfigRequest",
    # Response models
    "CameraStatusResponse",
    "DetectionResponse",
    "CalibrationResultModel",
    "VisionConfigResponse",
    "VisionStatisticsResponse",
    "ROIResponse",
]
