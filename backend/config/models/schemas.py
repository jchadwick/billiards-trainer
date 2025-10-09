"""Comprehensive Pydantic configuration schemas for all modules.

This module defines the complete data models for system configuration, providing:
- Type-safe configuration validation
- Default values and constraints
- JSON Schema generation
- Configuration inheritance and profiles
- Field descriptions for documentation

Based on specifications from all module SPECS.md files.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Data Classes for Configuration Management
# =============================================================================


@dataclass
class ConfigValue:
    """Configuration value with metadata."""

    key: str
    value: Any
    source: "ConfigSource"
    timestamp: float
    validated: bool
    schema: Optional[dict] = None
    description: Optional[str] = None


@dataclass
class ConfigChange:
    """Configuration change event."""

    key: str
    old_value: Any
    new_value: Any
    source: "ConfigSource"
    timestamp: float
    applied: bool


@dataclass
class ConfigProfile:
    """Named configuration profile."""

    name: str
    description: str
    settings: dict[str, Any]
    parent: Optional[str] = None  # Parent profile to inherit from
    conditions: Optional[dict] = None  # Auto-activation conditions


# =============================================================================
# Base Configuration Classes
# =============================================================================


class ConfigSource(str, Enum):
    """Configuration source enumeration."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    CLI = "cli"
    API = "api"
    RUNTIME = "runtime"


class ConfigFormat(str, Enum):
    """Configuration file format enumeration."""

    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class BaseConfig(BaseModel):
    """Base configuration class with common functionality."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            Path: str,
            datetime: lambda v: v.isoformat(),
        },
    )


# =============================================================================
# Configuration Metadata
# =============================================================================


class ConfigMetadata(BaseConfig):
    """Configuration file metadata."""

    version: str = Field(default="1.0.0", description="Configuration schema version")
    application: str = Field(
        default="billiards-trainer", description="Application name"
    )
    created: datetime = Field(
        default_factory=datetime.now, description="Configuration creation timestamp"
    )
    modified: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )
    profile: Optional[str] = Field(
        default=None, description="Configuration profile name"
    )
    environment: Optional[str] = Field(
        default=None, description="Target environment (dev, test, prod)"
    )
    description: Optional[str] = Field(
        default=None, description="Configuration description"
    )


# =============================================================================
# System Configuration
# =============================================================================


class PerformanceMode(str, Enum):
    """System performance mode."""

    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"
    CUSTOM = "custom"


class SystemPaths(BaseConfig):
    """System file and directory paths."""

    config_dir: Path = Field(
        default=Path("config"), description="Configuration files directory"
    )
    data_dir: Path = Field(default=Path("data"), description="Data files directory")
    log_dir: Path = Field(default=Path("logs"), description="Log files directory")
    cache_dir: Path = Field(default=Path(".cache"), description="Cache files directory")
    profiles_dir: Path = Field(
        default=Path("config/profiles"), description="Configuration profiles directory"
    )
    temp_dir: Path = Field(
        default=Path("/tmp"), description="Temporary files directory"
    )


class LoggingFormatter(BaseConfig):
    """Logging formatter configuration."""

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    datefmt: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log messages",
    )


class LoggingFileHandler(BaseConfig):
    """Logging file handler configuration."""

    filename: str = Field(..., description="Log file name")
    level: str = Field(default="DEBUG", description="Handler logging level")
    max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum log file size in bytes",
    )
    backup_count: int = Field(
        default=5, ge=1, le=20, description="Number of backup log files to keep"
    )
    encoding: str = Field(default="utf-8", description="Log file encoding")


class LoggingEnvironmentDefaults(BaseConfig):
    """Environment-specific logging defaults."""

    level: str = Field(default="INFO", description="Default log level for environment")
    log_system_info: bool = Field(
        default=False, description="Whether to log system info on startup"
    )


class LoggingSystemInfo(BaseConfig):
    """System information logging configuration."""

    log_python_version: bool = Field(default=True, description="Log Python version")
    log_platform: bool = Field(default=True, description="Log platform information")
    log_architecture: bool = Field(default=True, description="Log architecture")
    log_processor: bool = Field(default=True, description="Log processor information")
    log_working_directory: bool = Field(
        default=True, description="Log working directory"
    )
    header_separator: str = Field(
        default="=== System Information ===", description="Header separator"
    )
    footer_separator: str = Field(
        default="==========================", description="Footer separator"
    )


class LoggingUvicorn(BaseConfig):
    """Uvicorn logging configuration."""

    disable_access_logging: bool = Field(
        default=True, description="Disable uvicorn access logging"
    )
    configure_error_logging: bool = Field(
        default=True, description="Configure uvicorn error logging"
    )
    propagate_errors: bool = Field(default=True, description="Propagate error logs")


class SystemLogging(BaseConfig):
    """System logging configuration."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Global logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file_logging: bool = Field(default=True, description="Enable file logging")
    console_logging: bool = Field(default=True, description="Enable console logging")
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum log file size in bytes",
    )
    backup_count: int = Field(
        default=5, ge=1, le=20, description="Number of backup log files to keep"
    )
    log_modules: dict[str, str] = Field(
        default_factory=lambda: {
            "vision": "INFO",
            "core": "INFO",
            "api": "INFO",
            "projector": "INFO",
            "config": "INFO",
        },
        description="Per-module logging levels",
    )
    default_config_path: str = Field(
        default="config/logging.yaml", description="Default logging config file path"
    )
    env_key: str = Field(
        default="LOG_CFG", description="Environment variable for logging config path"
    )
    default_log_dir: str = Field(default="logs", description="Default log directory")
    log_dir_env_key: str = Field(
        default="LOG_DIR", description="Environment variable for log directory"
    )
    file_handlers: dict[str, LoggingFileHandler] = Field(
        default_factory=lambda: {
            "app_log": LoggingFileHandler(
                filename="app.log",
                level="DEBUG",
                max_bytes=10 * 1024 * 1024,
                backup_count=5,
                encoding="utf-8",
            ),
            "error_log": LoggingFileHandler(
                filename="error.log",
                level="ERROR",
                max_bytes=10 * 1024 * 1024,
                backup_count=5,
                encoding="utf-8",
            ),
        },
        description="File handler configurations",
    )
    formatters: dict[str, LoggingFormatter] = Field(
        default_factory=lambda: {
            "detailed": LoggingFormatter(
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ),
            "simple": LoggingFormatter(
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ),
        },
        description="Formatter configurations",
    )
    environment_defaults: dict[str, LoggingEnvironmentDefaults] = Field(
        default_factory=lambda: {
            "development": LoggingEnvironmentDefaults(
                level="DEBUG", log_system_info=True
            ),
            "production": LoggingEnvironmentDefaults(
                level="INFO", log_system_info=False
            ),
            "testing": LoggingEnvironmentDefaults(
                level="WARNING", log_system_info=False
            ),
        },
        description="Environment-specific defaults",
    )
    system_info: LoggingSystemInfo = Field(
        default_factory=LoggingSystemInfo, description="System info logging config"
    )
    uvicorn: LoggingUvicorn = Field(
        default_factory=LoggingUvicorn, description="Uvicorn logging config"
    )


class SystemPerformance(BaseConfig):
    """System performance configuration."""

    mode: PerformanceMode = Field(
        default=PerformanceMode.BALANCED, description="Performance optimization mode"
    )
    max_memory_mb: int = Field(
        default=2048, ge=512, le=16384, description="Maximum memory usage in MB"
    )
    max_cpu_percent: int = Field(
        default=80, ge=10, le=100, description="Maximum CPU usage percentage"
    )
    thread_pool_size: int = Field(
        default=4, ge=1, le=32, description="Thread pool size for async operations"
    )
    enable_profiling: bool = Field(
        default=False, description="Enable performance profiling"
    )
    profile_interval: int = Field(
        default=60, ge=1, description="Profiling interval in seconds"
    )


class SystemConfig(BaseConfig):
    """System-wide configuration."""

    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Runtime environment")
    timezone: str = Field(default="UTC", description="System timezone")
    paths: SystemPaths = Field(
        default_factory=SystemPaths, description="System file paths"
    )
    logging: SystemLogging = Field(
        default_factory=SystemLogging, description="Logging configuration"
    )
    performance: SystemPerformance = Field(
        default_factory=SystemPerformance, description="Performance configuration"
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v):
        """Validate timezone string."""
        try:
            import pytz

            pytz.timezone(v)
            return v
        except (ImportError, Exception):
            # Fall back to basic validation if pytz is not available
            if v in ["UTC", "GMT"]:
                return v
            raise ValueError(f"Unknown timezone: {v}")


class ConfigSources(BaseModel):
    """Configuration source priorities."""

    enable_files: bool = True
    enable_environment: bool = True
    enable_cli: bool = True
    file_paths: list[Path] = [Path("default.json"), Path("local.json")]
    env_prefix: str = "BILLIARDS_"
    precedence: list[str] = ["cli", "environment", "file", "default"]


class ValidationRules(BaseModel):
    """Configuration validation settings."""

    strict_mode: bool = False
    allow_unknown: bool = False
    type_checking: bool = True
    range_checking: bool = True
    dependency_checking: bool = True
    auto_correct: bool = False


class PersistenceSettings(BaseModel):
    """Configuration persistence settings."""

    auto_save: bool = True
    save_interval: int = 60  # seconds
    backup_count: int = 5
    compression: bool = True
    encryption: bool = False
    atomic_writes: bool = True


class HotReloadSettings(BaseModel):
    """Hot reload configuration."""

    enabled: bool = True
    watch_files: bool = True
    watch_interval: int = 1  # seconds
    reload_delay: int = 0  # milliseconds
    notify_modules: bool = True
    validation_before_reload: bool = True


# =============================================================================
# Camera Configuration (Vision Module)
# =============================================================================


class CameraBackend(str, Enum):
    """Camera backend types."""

    AUTO = "auto"
    V4L2 = "v4l2"
    DSHOW = "dshow"
    GSTREAMER = "gstreamer"
    OPENCV = "opencv"


class ExposureMode(str, Enum):
    """Camera exposure modes."""

    AUTO = "auto"
    MANUAL = "manual"
    APERTURE_PRIORITY = "aperture_priority"
    SHUTTER_PRIORITY = "shutter_priority"


class VideoSourceType(str, Enum):
    """Video source types."""

    CAMERA = "camera"
    FILE = "file"
    STREAM = "stream"


class CameraSettings(BaseConfig):
    """Camera hardware settings."""

    device_id: int = Field(default=0, ge=0, le=10, description="Camera device index")
    backend: CameraBackend = Field(
        default=CameraBackend.AUTO, description="Camera backend to use"
    )
    resolution: tuple[int, int] = Field(
        default=(1920, 1080), description="Camera resolution (width, height)"
    )
    fps: int = Field(default=30, ge=15, le=120, description="Frames per second")
    exposure_mode: ExposureMode = Field(
        default=ExposureMode.AUTO, description="Camera exposure mode"
    )
    exposure_value: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Manual exposure value (0.0-1.0)"
    )
    gain: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Camera gain multiplier"
    )
    brightness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Camera brightness (0.0-1.0)"
    )
    contrast: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Camera contrast (0.0-1.0)"
    )
    saturation: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Camera saturation (0.0-1.0)"
    )
    buffer_size: int = Field(
        default=1, ge=1, le=10, description="Camera frame buffer size"
    )
    auto_focus: bool = Field(default=True, description="Enable automatic focus")
    focus_value: Optional[int] = Field(
        default=None, ge=0, le=255, description="Manual focus value (0-255)"
    )
    auto_reconnect: bool = Field(
        default=True, description="Enable automatic camera reconnection"
    )
    reconnect_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay before reconnection attempts (seconds)",
    )
    max_reconnect_attempts: int = Field(
        default=5, ge=1, le=100, description="Maximum number of reconnection attempts"
    )

    # Video source configuration
    video_source_type: VideoSourceType = Field(
        default=VideoSourceType.CAMERA,
        description="Video source type (camera, file, or stream)",
    )
    video_file_path: Optional[str] = Field(
        default=None,
        description="Path to video file when video_source_type='file'",
    )
    stream_url: Optional[str] = Field(
        default=None,
        description="Stream URL when video_source_type='stream'",
    )

    # Video playback control
    loop_video: bool = Field(
        default=False, description="Loop video playback for file sources"
    )
    video_start_frame: int = Field(
        default=0, ge=0, description="Starting frame number for video files"
    )
    video_end_frame: Optional[int] = Field(
        default=None, ge=0, description="Ending frame number for video files"
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        """Validate camera resolution."""
        if v[0] < 640 or v[1] < 480:
            raise ValueError("Minimum resolution is 640x480")
        if v[0] > 4096 or v[1] > 4096:
            raise ValueError("Maximum resolution is 4096x4096")
        return v

    @model_validator(mode="after")
    def validate_video_config(self):
        """Validate video source configuration."""
        # Check that required fields are present for each source type
        if self.video_source_type == VideoSourceType.FILE and not self.video_file_path:
            raise ValueError(
                "video_file_path is required when video_source_type='file'"
            )

        if self.video_source_type == VideoSourceType.STREAM and not self.stream_url:
            raise ValueError("stream_url is required when video_source_type='stream'")

        # Validate frame range
        if (
            self.video_end_frame is not None
            and self.video_start_frame >= self.video_end_frame
        ):
            raise ValueError(
                f"video_start_frame ({self.video_start_frame}) must be less than "
                f"video_end_frame ({self.video_end_frame})"
            )

        return self


class ColorThresholds(BaseConfig):
    """HSV color threshold ranges."""

    hue_min: int = Field(default=0, ge=0, le=179, description="Minimum hue value")
    hue_max: int = Field(default=179, ge=0, le=179, description="Maximum hue value")
    saturation_min: int = Field(
        default=0, ge=0, le=255, description="Minimum saturation value"
    )
    saturation_max: int = Field(
        default=255, ge=0, le=255, description="Maximum saturation value"
    )
    value_min: int = Field(
        default=0, ge=0, le=255, description="Minimum value (brightness)"
    )
    value_max: int = Field(
        default=255, ge=0, le=255, description="Maximum value (brightness)"
    )

    @field_validator("hue_max")
    @classmethod
    def validate_hue_range(cls, v, info):
        """Validate hue range."""
        if info.data and "hue_min" in info.data and v < info.data["hue_min"]:
            raise ValueError("hue_max must be >= hue_min")
        return v


class DetectionMethod(str, Enum):
    """Ball detection methods."""

    HOUGH = "hough"
    CONTOUR = "contour"
    BLOB = "blob"
    TEMPLATE = "template"
    YOLO = "yolo"


class DetectionBackend(str, Enum):
    """Detection backend types."""

    OPENCV = "opencv"
    YOLO = "yolo"


class DetectionSettings(BaseConfig):
    """Vision detection algorithm settings."""

    # Detection backend selection
    detection_backend: DetectionBackend = Field(
        default=DetectionBackend.OPENCV,
        description="Detection backend to use (opencv or yolo)",
    )

    # YOLO configuration
    yolo_model_path: str = Field(
        default="models/yolov8n-pool.onnx",
        description="Path to YOLO model file (ONNX format)",
    )
    yolo_confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="YOLO detection confidence threshold",
    )
    yolo_nms_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="YOLO non-maximum suppression threshold",
    )
    yolo_device: str = Field(
        default="cpu", description="YOLO inference device (cpu, cuda, or tpu)"
    )
    use_opencv_validation: bool = Field(
        default=True,
        description="Use OpenCV validation on YOLO detections",
    )
    fallback_to_opencv: bool = Field(
        default=True,
        description="Fall back to OpenCV if YOLO fails or unavailable",
    )

    # Google Coral Edge TPU configuration
    tpu_device_path: Optional[str] = Field(
        default=None,
        description="Coral TPU device path (e.g., '/dev/bus/usb/001/002', 'usb', 'pcie', or None for auto-detect)",
    )
    tpu_model_path: str = Field(
        default="models/yolov8n-pool_edgetpu.tflite",
        description="Path to Edge TPU compiled model file (.tflite)",
    )

    # Table detection
    table_color: ColorThresholds = Field(
        default_factory=lambda: ColorThresholds(
            hue_min=35,
            hue_max=85,  # Green table
            saturation_min=50,
            saturation_max=255,
            value_min=50,
            value_max=255,
        ),
        description="Table surface color thresholds",
    )
    table_edge_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Table edge detection threshold"
    )
    min_table_area: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Minimum table area as fraction of image",
    )

    # Ball detection
    ball_colors: dict[str, ColorThresholds] = Field(
        default_factory=lambda: {
            "cue": ColorThresholds(
                hue_min=0, hue_max=179, saturation_min=0, saturation_max=50
            ),
            "solid": ColorThresholds(
                hue_min=0, hue_max=179, saturation_min=100, saturation_max=255
            ),
            "stripe": ColorThresholds(
                hue_min=0, hue_max=179, saturation_min=50, saturation_max=255
            ),
        },
        description="Ball color detection thresholds by type",
    )
    min_ball_radius: int = Field(
        default=10, ge=5, le=50, description="Minimum ball radius in pixels"
    )
    max_ball_radius: int = Field(
        default=40, ge=20, le=100, description="Maximum ball radius in pixels"
    )
    ball_detection_method: DetectionMethod = Field(
        default=DetectionMethod.HOUGH, description="Ball detection algorithm"
    )
    ball_sensitivity: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Ball detection sensitivity"
    )

    # Cue detection
    cue_detection_enabled: bool = Field(
        default=True, description="Enable cue stick detection"
    )
    min_cue_length: int = Field(
        default=100, ge=50, le=500, description="Minimum cue length in pixels"
    )
    cue_line_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Cue line detection threshold"
    )
    cue_color: Optional[ColorThresholds] = Field(
        default=None, description="Cue color thresholds (optional)"
    )


class ProcessingSettings(BaseConfig):
    """Image processing settings."""

    use_gpu: bool = Field(
        default=False, description="Use GPU acceleration when available"
    )
    enable_preprocessing: bool = Field(
        default=True, description="Enable image preprocessing"
    )
    blur_kernel_size: int = Field(
        default=5, ge=3, le=15, description="Gaussian blur kernel size (must be odd)"
    )
    morphology_kernel_size: int = Field(
        default=3, ge=3, le=15, description="Morphological operations kernel size"
    )
    enable_tracking: bool = Field(
        default=True, description="Enable object tracking across frames"
    )
    tracking_max_distance: int = Field(
        default=50, ge=10, le=200, description="Maximum tracking distance in pixels"
    )
    frame_skip: int = Field(
        default=0, ge=0, le=10, description="Process every Nth frame (0 = process all)"
    )
    roi_enabled: bool = Field(
        default=False, description="Enable region of interest processing"
    )
    roi_coordinates: Optional[tuple[int, int, int, int]] = Field(
        default=None, description="ROI coordinates (x, y, width, height)"
    )

    @field_validator("blur_kernel_size", "morphology_kernel_size")
    @classmethod
    def validate_odd_kernel_size(cls, v):
        """Validate kernel size is odd."""
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v


class VisionConfig(BaseConfig):
    """Complete vision module configuration."""

    camera: CameraSettings = Field(
        default_factory=CameraSettings, description="Camera hardware settings"
    )
    detection: DetectionSettings = Field(
        default_factory=DetectionSettings, description="Detection algorithm settings"
    )
    processing: ProcessingSettings = Field(
        default_factory=ProcessingSettings, description="Image processing settings"
    )
    debug: bool = Field(default=False, description="Enable vision debug mode")
    save_debug_images: bool = Field(
        default=False, description="Save debug images to disk"
    )
    debug_output_path: Path = Field(
        default=Path("/tmp/vision_debug"), description="Debug output directory"
    )
    calibration_auto_save: bool = Field(
        default=True, description="Automatically save calibration data"
    )


# =============================================================================
# Core Module Configuration
# =============================================================================


class PhysicsConfig(BaseConfig):
    """Physics simulation parameters."""

    gravity: float = Field(
        default=9.81, ge=0.0, description="Gravitational acceleration (m/s²)"
    )
    air_resistance: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Air resistance coefficient"
    )
    rolling_friction: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Rolling friction coefficient"
    )
    sliding_friction: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Sliding friction coefficient"
    )
    cushion_coefficient: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Cushion elasticity coefficient"
    )
    spin_decay_rate: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Spin decay rate per second"
    )
    max_iterations: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum physics simulation iterations",
    )
    time_step: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Physics simulation time step (seconds)",
    )
    enable_spin_effects: bool = Field(
        default=True, description="Enable ball spin physics"
    )
    enable_cushion_compression: bool = Field(
        default=True, description="Enable cushion compression effects"
    )


class PredictionConfig(BaseConfig):
    """Trajectory prediction parameters."""

    max_prediction_time: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Maximum prediction time horizon (seconds)",
    )
    prediction_resolution: int = Field(
        default=100, ge=10, le=1000, description="Number of points per trajectory"
    )
    collision_threshold: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Collision detection threshold (meters)",
    )
    monte_carlo_samples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Monte Carlo samples for uncertainty estimation",
    )
    max_bounces: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of cushion bounces to predict",
    )
    stop_velocity_threshold: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Velocity threshold to consider ball stopped (m/s)",
    )


class AssistanceConfig(BaseConfig):
    """Assistance feature configuration."""

    difficulty_levels: dict[str, float] = Field(
        default_factory=lambda: {
            "beginner": 0.2,
            "intermediate": 0.5,
            "advanced": 0.8,
            "expert": 1.0,
        },
        description="Difficulty level mappings",
    )
    show_alternative_shots: bool = Field(
        default=True, description="Show alternative shot suggestions"
    )
    max_alternatives: int = Field(
        default=3, ge=1, le=10, description="Maximum number of alternative shots"
    )
    highlight_best_shot: bool = Field(
        default=True, description="Highlight the recommended shot"
    )
    show_success_probability: bool = Field(
        default=True, description="Display shot success probability"
    )
    enable_shot_analysis: bool = Field(
        default=True, description="Enable detailed shot analysis"
    )
    analysis_depth: int = Field(
        default=3, ge=1, le=10, description="Analysis depth (number of shots ahead)"
    )


class CoreValidationConfig(BaseConfig):
    """State validation parameters."""

    max_ball_velocity: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Maximum realistic ball velocity (m/s)",
    )
    min_ball_separation: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Minimum ball separation distance (meters)",
    )
    position_tolerance: float = Field(
        default=0.005,
        ge=0.001,
        le=0.05,
        description="Position validation tolerance (meters)",
    )
    velocity_tolerance: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Velocity validation tolerance (m/s)"
    )
    enable_physics_validation: bool = Field(
        default=True, description="Enable physics consistency validation"
    )
    enable_continuity_check: bool = Field(
        default=True, description="Enable state continuity checking"
    )
    auto_correction: bool = Field(
        default=True, description="Automatically correct detected errors"
    )


class CoreConfig(BaseConfig):
    """Core module configuration."""

    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig, description="Physics simulation configuration"
    )
    prediction: PredictionConfig = Field(
        default_factory=PredictionConfig,
        description="Trajectory prediction configuration",
    )
    assistance: AssistanceConfig = Field(
        default_factory=AssistanceConfig, description="Player assistance configuration"
    )
    validation: CoreValidationConfig = Field(
        default_factory=CoreValidationConfig,
        description="State validation configuration",
    )
    state_history_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of historical states to maintain",
    )
    event_buffer_size: int = Field(
        default=100, ge=10, le=1000, description="Event buffer size"
    )
    update_frequency: float = Field(
        default=60.0, ge=1.0, le=120.0, description="State update frequency (Hz)"
    )
    enable_game_rules: bool = Field(
        default=False, description="Enable game rule enforcement"
    )
    default_game_type: str = Field(default="practice", description="Default game type")


# =============================================================================
# API Module Configuration
# =============================================================================


class CorsConfig(BaseConfig):
    """CORS configuration."""

    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: list[str] = Field(
        default=["*"], description="Allowed origins for CORS"
    )
    allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods",
    )
    allow_headers: list[str] = Field(default=["*"], description="Allowed headers")
    allow_credentials: bool = Field(
        default=True, description="Allow credentials in CORS requests"
    )
    max_age: int = Field(
        default=600, ge=0, description="CORS preflight cache time (seconds)"
    )


class RateLimitConfig(BaseConfig):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=100, ge=1, le=10000, description="Requests per minute per client"
    )
    burst_size: int = Field(default=20, ge=1, le=1000, description="Burst request size")
    websocket_connections_per_ip: int = Field(
        default=10, ge=1, le=100, description="WebSocket connections per IP"
    )
    enable_distributed: bool = Field(
        default=False, description="Enable distributed rate limiting (requires Redis)"
    )
    redis_url: Optional[str] = Field(
        default=None, description="Redis URL for distributed rate limiting"
    )


class NetworkConfig(BaseConfig):
    """Network configuration."""

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(
        default=1, ge=1, le=32, description="Number of worker processes"
    )
    max_connections: int = Field(
        default=100, ge=1, le=10000, description="Maximum concurrent connections"
    )
    websocket_ping_interval: int = Field(
        default=30, ge=5, le=300, description="WebSocket ping interval (seconds)"
    )
    websocket_ping_timeout: int = Field(
        default=10, ge=1, le=60, description="WebSocket ping timeout (seconds)"
    )
    request_timeout: int = Field(
        default=30, ge=1, le=300, description="HTTP request timeout (seconds)"
    )
    enable_compression: bool = Field(
        default=True, description="Enable response compression"
    )
    ssl_enabled: bool = Field(default=False, description="Enable SSL/TLS")
    ssl_cert_file: Optional[str] = Field(
        default=None, description="SSL certificate file path"
    )
    ssl_key_file: Optional[str] = Field(
        default=None, description="SSL private key file path"
    )


class VideoFeedConfig(BaseConfig):
    """Video feed API configuration."""

    enabled: bool = Field(default=False, description="Enable video feed API endpoints")
    endpoint: str = Field(
        default="/api/v1/video/feed", description="Video feed endpoint path"
    )
    mjpeg_stream: bool = Field(default=False, description="Enable MJPEG streaming")
    mjpeg_endpoint: str = Field(
        default="/api/v1/video/stream", description="MJPEG stream endpoint path"
    )
    max_clients: int = Field(
        default=5, ge=1, le=100, description="Maximum concurrent video clients"
    )
    buffer_size: int = Field(
        default=10, ge=1, le=100, description="Video frame buffer size"
    )


class APIConfig(BaseConfig):
    """API module configuration."""

    network: NetworkConfig = Field(
        default_factory=NetworkConfig, description="Network configuration"
    )
    cors: CorsConfig = Field(
        default_factory=CorsConfig, description="CORS configuration"
    )
    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limiting configuration"
    )
    video_feed: VideoFeedConfig = Field(
        default_factory=VideoFeedConfig, description="Video feed configuration"
    )
    enable_docs: bool = Field(
        default=True, description="Enable API documentation endpoints"
    )
    docs_url: str = Field(default="/docs", description="API documentation URL path")
    openapi_url: str = Field(
        default="/openapi.json", description="OpenAPI schema URL path"
    )
    api_prefix: str = Field(default="/api/v1", description="API URL prefix")
    websocket_path: str = Field(default="/ws", description="WebSocket endpoint path")
    health_check_path: str = Field(
        default="/health", description="Health check endpoint path"
    )
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")


# =============================================================================
# Projector Module Configuration
# =============================================================================


class DisplayMode(str, Enum):
    """Display output modes."""

    FULLSCREEN = "fullscreen"
    WINDOW = "window"
    BORDERLESS = "borderless"


class RenderQuality(str, Enum):
    """Rendering quality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class DisplayConfig(BaseConfig):
    """Display output configuration."""

    mode: DisplayMode = Field(
        default=DisplayMode.FULLSCREEN, description="Display mode"
    )
    monitor_index: int = Field(
        default=0, ge=0, le=10, description="Monitor index to use"
    )
    resolution: tuple[int, int] = Field(
        default=(1920, 1080), description="Display resolution (width, height)"
    )
    refresh_rate: int = Field(
        default=60, ge=30, le=144, description="Display refresh rate (Hz)"
    )
    vsync: bool = Field(default=True, description="Enable vertical sync")
    gamma: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Display gamma correction"
    )
    brightness: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Display brightness multiplier"
    )
    contrast: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Display contrast multiplier"
    )


class ProjectorCalibrationConfig(BaseConfig):
    """Geometric calibration configuration."""

    calibration_points: Optional[list[tuple[float, float]]] = Field(
        default=None,
        description="Calibration corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]",
    )
    keystone_horizontal: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Horizontal keystone correction"
    )
    keystone_vertical: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Vertical keystone correction"
    )
    rotation: float = Field(
        default=0.0, ge=-180.0, le=180.0, description="Display rotation (degrees)"
    )
    barrel_distortion: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Barrel distortion correction"
    )
    edge_blend: dict[str, float] = Field(
        default_factory=lambda: {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
        description="Edge blending settings",
    )
    auto_calibration: bool = Field(
        default=False, description="Enable automatic calibration"
    )
    calibration_grid_size: int = Field(
        default=10, ge=5, le=50, description="Calibration grid size"
    )

    @field_validator("calibration_points")
    @classmethod
    def validate_calibration_points(cls, v):
        """Validate calibration points."""
        if v is not None and len(v) != 4:
            raise ValueError("Calibration points must contain exactly 4 points")
        return v


class VisualRenderingConfig(BaseConfig):
    """Visual rendering configuration."""

    # Line rendering
    trajectory_width: float = Field(
        default=3.0, ge=1.0, le=10.0, description="Trajectory line width (pixels)"
    )
    trajectory_color: tuple[int, int, int] = Field(
        default=(0, 255, 0), description="Primary trajectory color (RGB)"
    )
    collision_color: tuple[int, int, int] = Field(
        default=(255, 0, 0), description="Collision indicator color (RGB)"
    )
    reflection_color: tuple[int, int, int] = Field(
        default=(255, 255, 0), description="Reflection trajectory color (RGB)"
    )

    # Effects
    enable_glow: bool = Field(default=True, description="Enable glow effects")
    glow_intensity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Glow effect intensity"
    )
    enable_animations: bool = Field(
        default=True, description="Enable visual animations"
    )
    animation_speed: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Animation speed multiplier"
    )
    fade_duration: float = Field(
        default=0.5, ge=0.0, le=2.0, description="Fade effect duration (seconds)"
    )

    # Transparency
    trajectory_opacity: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Trajectory line opacity"
    )
    ghost_ball_opacity: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Ghost ball opacity"
    )
    overlay_opacity: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Text overlay opacity"
    )

    # Fonts and text
    font_family: str = Field(
        default="Arial", description="Font family for text rendering"
    )
    font_size: int = Field(default=24, ge=8, le=72, description="Default font size")
    text_color: tuple[int, int, int] = Field(
        default=(255, 255, 255), description="Default text color (RGB)"
    )


class RenderingConfig(BaseConfig):
    """Rendering engine configuration."""

    renderer: str = Field(
        default="opengl",
        description="Rendering backend (opengl, directx, vulkan, software)",
    )
    quality: RenderQuality = Field(
        default=RenderQuality.HIGH, description="Rendering quality level"
    )
    antialiasing: str = Field(
        default="4x", description="Antialiasing level (none, 2x, 4x, 8x, 16x)"
    )
    texture_filtering: str = Field(
        default="anisotropic", description="Texture filtering method"
    )
    max_fps: int = Field(default=60, ge=30, le=144, description="Maximum frame rate")
    buffer_frames: int = Field(default=2, ge=1, le=4, description="Frame buffer count")
    use_gpu: bool = Field(default=True, description="Use GPU acceleration")
    gpu_memory_limit_mb: int = Field(
        default=1024, ge=256, le=8192, description="GPU memory limit (MB)"
    )


class ProjectorNetworkConfig(BaseConfig):
    """Projector network configuration."""

    backend_url: str = Field(
        default="ws://localhost:8000/ws", description="Backend WebSocket URL"
    )
    reconnect_interval: int = Field(
        default=5, ge=1, le=60, description="Reconnection interval (seconds)"
    )
    heartbeat_interval: int = Field(
        default=30, ge=5, le=300, description="Heartbeat interval (seconds)"
    )
    buffer_size: int = Field(
        default=1024 * 1024, ge=1024, description="Network buffer size (bytes)"  # 1MB
    )
    compression: bool = Field(default=True, description="Enable message compression")
    timeout: int = Field(
        default=10, ge=1, le=60, description="Connection timeout (seconds)"
    )

    # Video Feed Configuration
    stream_video_feed: bool = Field(
        default=False, description="Enable video feed streaming to projector"
    )
    video_feed_quality: int = Field(
        default=85, ge=1, le=100, description="Video feed JPEG quality (1-100)"
    )
    video_feed_fps: int = Field(
        default=15, ge=1, le=60, description="Video feed frames per second"
    )
    video_feed_scale: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Video feed resolution scale factor"
    )
    video_feed_format: str = Field(
        default="jpeg", description="Video feed image format (jpeg, png, webp)"
    )


class AssistanceDisplayConfig(BaseConfig):
    """Assistance feature display configuration."""

    show_primary_trajectory: bool = Field(
        default=True, description="Show primary ball trajectory"
    )
    show_collision_trajectories: bool = Field(
        default=True, description="Show collision result trajectories"
    )
    show_ghost_balls: bool = Field(
        default=True, description="Show ghost ball positions"
    )
    show_impact_points: bool = Field(
        default=True, description="Show collision impact points"
    )
    show_angle_measurements: bool = Field(
        default=False, description="Show angle measurements"
    )
    show_force_indicators: bool = Field(
        default=True, description="Show force/power indicators"
    )
    show_probability: bool = Field(default=True, description="Show success probability")
    show_alternative_shots: bool = Field(
        default=False, description="Show alternative shot suggestions"
    )
    max_trajectory_bounces: int = Field(
        default=3, ge=1, le=10, description="Maximum trajectory bounces to display"
    )
    difficulty_filter: Optional[str] = Field(
        default=None, description="Filter by difficulty level"
    )


class ProjectorConfig(BaseConfig):
    """Complete projector configuration."""

    display: DisplayConfig = Field(
        default_factory=DisplayConfig, description="Display configuration"
    )
    calibration: ProjectorCalibrationConfig = Field(
        default_factory=ProjectorCalibrationConfig, description="Geometric calibration"
    )
    visual: VisualRenderingConfig = Field(
        default_factory=VisualRenderingConfig, description="Visual rendering settings"
    )
    rendering: RenderingConfig = Field(
        default_factory=RenderingConfig, description="Rendering engine settings"
    )
    network: ProjectorNetworkConfig = Field(
        default_factory=ProjectorNetworkConfig, description="Network configuration"
    )
    assistance: AssistanceDisplayConfig = Field(
        default_factory=AssistanceDisplayConfig,
        description="Assistance display settings",
    )
    debug: bool = Field(default=False, description="Enable projector debug mode")
    debug_overlay: bool = Field(
        default=False, description="Show debug information overlay"
    )


# =============================================================================
# Enhanced Profile Management
# =============================================================================


class ConfigProfileEnhanced(BaseConfig):
    """Enhanced named configuration profile."""

    name: str = Field(description="Profile name")
    description: Optional[str] = Field(default=None, description="Profile description")
    parent: Optional[str] = Field(
        default=None, description="Parent profile to inherit from"
    )
    conditions: Optional[dict[str, Any]] = Field(
        default=None, description="Auto-activation conditions"
    )
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Profile-specific settings"
    )
    created: datetime = Field(
        default_factory=datetime.now, description="Profile creation timestamp"
    )
    modified: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )
    tags: list[str] = Field(
        default_factory=list, description="Profile tags for organization"
    )
    is_default: bool = Field(
        default=False, description="Whether this is the default profile"
    )


# =============================================================================
# Complete Application Configuration
# =============================================================================


class ApplicationConfig(BaseConfig):
    """Complete application configuration."""

    # Metadata
    metadata: ConfigMetadata = Field(
        default_factory=ConfigMetadata, description="Configuration metadata"
    )

    # Core configurations
    system: SystemConfig = Field(
        default_factory=SystemConfig, description="System configuration"
    )
    vision: VisionConfig = Field(
        default_factory=VisionConfig, description="Vision module configuration"
    )
    core: CoreConfig = Field(
        default_factory=CoreConfig, description="Core module configuration"
    )
    api: APIConfig = Field(
        default_factory=APIConfig, description="API module configuration"
    )
    projector: ProjectorConfig = Field(
        default_factory=ProjectorConfig, description="Projector module configuration"
    )

    # Configuration management
    sources: ConfigSources = Field(
        default_factory=ConfigSources, description="Configuration sources"
    )
    validation: ValidationRules = Field(
        default_factory=ValidationRules, description="Validation rules"
    )
    persistence: PersistenceSettings = Field(
        default_factory=PersistenceSettings, description="Persistence settings"
    )
    hot_reload: HotReloadSettings = Field(
        default_factory=HotReloadSettings, description="Hot reload settings"
    )

    # Profiles and customization
    profiles: dict[str, ConfigProfileEnhanced] = Field(
        default_factory=dict, description="Configuration profiles"
    )
    active_profile: Optional[str] = Field(
        default=None, description="Currently active profile"
    )
    user_preferences: dict[str, Any] = Field(
        default_factory=dict, description="User-specific preferences"
    )
    feature_flags: dict[str, bool] = Field(
        default_factory=dict, description="Feature flags"
    )

    # Module-specific extensions
    custom_modules: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Custom module configurations"
    )

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate complete configuration consistency."""
        # Validate profile references
        if self.active_profile and self.active_profile not in self.profiles:
            raise ValueError(
                f"Active profile '{self.active_profile}' not found in profiles"
            )

        # Validate profile inheritance
        for profile_name, profile in self.profiles.items():
            if profile.parent and profile.parent not in self.profiles:
                raise ValueError(
                    f"Parent profile '{profile.parent}' not found "
                    f"for profile '{profile_name}'"
                )

        return self

    def get_json_schema(self) -> dict[str, Any]:
        """Generate JSON Schema for the configuration."""
        return self.model_json_schema()

    def merge_profile(self, profile_name: str) -> "ApplicationConfig":
        """Merge a profile into the configuration."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found")

        profile = self.profiles[profile_name]
        config_dict = self.model_dump()

        # Apply profile settings
        for key, value in profile.settings.items():
            keys = key.split(".")
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        return ApplicationConfig(**config_dict)


# =============================================================================
# Configuration Factory Functions
# =============================================================================


def create_default_config() -> ApplicationConfig:
    """Create a default application configuration."""
    return ApplicationConfig()


# Backward compatibility - keep existing aliases
ConfigurationSettings = ApplicationConfig
CameraConfig = CameraSettings  # Alias for backward compatibility
