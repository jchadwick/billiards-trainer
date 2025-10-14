"""Configuration validation module for backend startup.

This module validates all configuration parameters at startup, ensures required
parameters are present, validates ranges, and provides clear error messages.
Implements fail-fast behavior for critical configuration issues.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .manager import ConfigurationModule

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails with critical errors."""

    pass


class ConfigValidator:
    """Validates configuration at backend startup."""

    # Proven defaults from video_debugger and current working configuration
    PROVEN_DEFAULTS = {
        "vision": {
            "camera": {
                "device_id": 0,
                "resolution": [1920, 1080],
                "fps": 30,
                "backend": "auto",
                "exposure_mode": "auto",
                "gain": 1.0,
                "buffer_size": 1,
                "auto_reconnect": True,
                "reconnect_delay": 1.0,
                "max_reconnect_attempts": 5,
                "loop_video": False,
                "video_start_frame": 0,
                "video_end_frame": None,
            },
            "detection": {
                "detection_backend": "yolo",
                "yolo_confidence": 0.15,
                "yolo_nms_threshold": 0.45,
                "yolo_device": "cpu",
                "yolo_auto_fallback": True,
                "yolo_enable_opencv_classification": True,
                "yolo_min_ball_size": 20,
                "use_opencv_validation": True,
                "fallback_to_opencv": True,
                "table_edge_threshold": 0.7,
                "min_table_area": 0.3,
                "min_ball_radius": 10,
                "max_ball_radius": 40,
                "ball_detection_method": "hough",
                "ball_sensitivity": 0.8,
                "cue_detection_enabled": True,
                "min_cue_length": 100,
                "cue_line_threshold": 0.6,
                "enable_table_detection": True,
                "enable_ball_detection": True,
                "enable_cue_detection": True,
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 10,
                "max_distance": 100.0,
                "process_noise": 5.0,
                "measurement_noise": 20.0,
                "collision_threshold": 60.0,
                "min_hits_during_collision": 30,
                "motion_speed_threshold": 10.0,
                "return_tentative_tracks": False,
            },
            "processing": {
                "use_gpu": False,
                "enable_preprocessing": False,
                "blur_kernel_size": 5,
                "morphology_kernel_size": 3,
                "enable_tracking": True,
                "tracking_max_distance": 50,
                "frame_skip": 0,
            },
            "debug": False,
            "save_debug_images": False,
            "debug_output_path": "/tmp/vision_debug",
            "calibration_auto_save": True,
        }
    }

    # Validation rules for configuration parameters
    VALIDATION_RULES = {
        "vision.camera.fps": {
            "type": int,
            "min": 1,
            "max": 120,
            "description": "Camera frame rate",
        },
        "vision.camera.buffer_size": {
            "type": int,
            "min": 1,
            "max": 10,
            "description": "Camera buffer size",
        },
        "vision.camera.gain": {
            "type": (int, float),
            "min": 0.0,
            "max": 10.0,
            "description": "Camera gain",
        },
        "vision.camera.reconnect_delay": {
            "type": (int, float),
            "min": 0.1,
            "max": 60.0,
            "description": "Camera reconnect delay in seconds",
        },
        "vision.camera.max_reconnect_attempts": {
            "type": int,
            "min": 0,
            "max": 100,
            "description": "Maximum camera reconnect attempts",
        },
        "vision.detection.yolo_confidence": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "YOLO detection confidence threshold",
        },
        "vision.detection.yolo_nms_threshold": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "YOLO NMS threshold",
        },
        "vision.detection.table_edge_threshold": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "Table edge detection threshold",
        },
        "vision.detection.min_table_area": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "Minimum table area ratio",
        },
        "vision.detection.min_ball_radius": {
            "type": int,
            "min": 1,
            "max": 100,
            "description": "Minimum ball radius in pixels",
        },
        "vision.detection.max_ball_radius": {
            "type": int,
            "min": 1,
            "max": 200,
            "description": "Maximum ball radius in pixels",
        },
        "vision.detection.ball_sensitivity": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "Ball detection sensitivity",
        },
        "vision.detection.min_cue_length": {
            "type": int,
            "min": 10,
            "max": 1000,
            "description": "Minimum cue stick length in pixels",
        },
        "vision.detection.cue_line_threshold": {
            "type": (int, float),
            "min": 0.0,
            "max": 1.0,
            "description": "Cue line detection threshold",
        },
        "vision.tracking.max_age": {
            "type": int,
            "min": 1,
            "max": 1000,
            "description": "Maximum track age in frames",
        },
        "vision.tracking.min_hits": {
            "type": int,
            "min": 1,
            "max": 100,
            "description": "Minimum hits to establish track",
        },
        "vision.tracking.max_distance": {
            "type": (int, float),
            "min": 1.0,
            "max": 1000.0,
            "description": "Maximum distance for track association",
        },
        "vision.tracking.process_noise": {
            "type": (int, float),
            "min": 0.1,
            "max": 100.0,
            "description": "Kalman filter process noise",
        },
        "vision.tracking.measurement_noise": {
            "type": (int, float),
            "min": 0.1,
            "max": 100.0,
            "description": "Kalman filter measurement noise",
        },
        "vision.tracking.collision_threshold": {
            "type": (int, float),
            "min": 1.0,
            "max": 200.0,
            "description": "Collision detection threshold in pixels",
        },
        "vision.tracking.min_hits_during_collision": {
            "type": int,
            "min": 1,
            "max": 100,
            "description": "Minimum hits during collision detection",
        },
        "vision.tracking.motion_speed_threshold": {
            "type": (int, float),
            "min": 0.0,
            "max": 1000.0,
            "description": "Motion speed threshold in pixels/frame",
        },
        "vision.processing.blur_kernel_size": {
            "type": int,
            "min": 1,
            "max": 31,
            "description": "Blur kernel size (must be odd)",
            "validator": lambda x: x % 2 == 1,
            "validator_message": "Blur kernel size must be odd",
        },
        "vision.processing.morphology_kernel_size": {
            "type": int,
            "min": 1,
            "max": 31,
            "description": "Morphology kernel size (must be odd)",
            "validator": lambda x: x % 2 == 1,
            "validator_message": "Morphology kernel size must be odd",
        },
        "vision.processing.tracking_max_distance": {
            "type": (int, float),
            "min": 1.0,
            "max": 1000.0,
            "description": "Maximum tracking distance in pixels",
        },
        "vision.processing.frame_skip": {
            "type": int,
            "min": 0,
            "max": 10,
            "description": "Number of frames to skip",
        },
    }

    # Suboptimal value warnings
    SUBOPTIMAL_WARNINGS = {
        "vision.tracking.min_hits": {
            "recommended": 10,
            "warning_if": lambda x: x < 5,
            "message": "min_hits < 5 may cause unstable tracking. Recommended: 10",
        },
        "vision.detection.yolo_confidence": {
            "recommended": 0.15,
            "warning_if": lambda x: x > 0.5,
            "message": "yolo_confidence > 0.5 may miss detections. Recommended: 0.15",
        },
        "vision.tracking.max_distance": {
            "recommended": 100.0,
            "warning_if": lambda x: x > 200.0,
            "message": "max_distance > 200.0 may cause incorrect track associations. Recommended: 100.0",
        },
    }

    def __init__(self, config_manager: "ConfigurationModule"):
        """Initialize the configuration validator.

        Args:
            config_manager: Configuration manager instance to validate
        """
        self.config_manager = config_manager
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _get_nested_value(self, config: dict, path: str) -> Any:
        """Get nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "vision.camera.fps")

        Returns:
            Configuration value or None if not found
        """
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_nested_value(self, config: dict, path: str, value: Any) -> None:
        """Set nested configuration value using dot notation.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "vision.camera.fps")
            value: Value to set
        """
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def apply_defaults(self) -> None:
        """Apply proven default values for missing configuration parameters."""
        logger.info("Applying default configuration values...")

        defaults_applied = []

        # Flatten proven defaults
        def flatten_dict(d: dict, parent_key: str = "") -> dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened_defaults = flatten_dict(self.PROVEN_DEFAULTS)

        # Apply missing defaults - only if value is truly None
        for key, default_value in flattened_defaults.items():
            current_value = self.config_manager.get(key)
            if current_value is None:
                # Convert dot notation to nested dict and update
                self.config_manager.set(key, default_value, persist=False)
                defaults_applied.append(f"{key} = {default_value}")

        if defaults_applied:
            logger.info(f"Applied {len(defaults_applied)} default configuration values")
            for default in defaults_applied[:5]:  # Show first 5
                logger.debug(f"  {default}")
            if len(defaults_applied) > 5:
                logger.debug(f"  ... and {len(defaults_applied) - 5} more")

    def validate_parameter(self, path: str, value: Any) -> bool:
        """Validate a single configuration parameter.

        Args:
            path: Parameter path (e.g., "vision.camera.fps")
            value: Parameter value

        Returns:
            True if valid, False otherwise
        """
        if path not in self.VALIDATION_RULES:
            return True  # No validation rule, assume valid

        rule = self.VALIDATION_RULES[path]
        param_desc = rule.get("description", path)

        # Type validation
        expected_type = rule.get("type")
        if expected_type:
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"{param_desc} ({path}): Expected type {expected_type}, got {type(value).__name__}"
                    )
                    return False
            else:
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"{param_desc} ({path}): Expected type {expected_type.__name__}, got {type(value).__name__}"
                    )
                    return False

        # Range validation
        if isinstance(value, (int, float)):
            min_val = rule.get("min")
            max_val = rule.get("max")

            if min_val is not None and value < min_val:
                self.errors.append(
                    f"{param_desc} ({path}): Value {value} is below minimum {min_val}"
                )
                return False

            if max_val is not None and value > max_val:
                self.errors.append(
                    f"{param_desc} ({path}): Value {value} is above maximum {max_val}"
                )
                return False

        # Custom validator
        validator = rule.get("validator")
        if validator and not validator(value):
            msg = rule.get("validator_message", f"Invalid value for {param_desc}")
            self.errors.append(f"{param_desc} ({path}): {msg}")
            return False

        return True

    def check_suboptimal_values(self) -> None:
        """Check for suboptimal configuration values and log warnings."""
        for path, warning_rule in self.SUBOPTIMAL_WARNINGS.items():
            value = self.config_manager.get(path)
            if value is None:
                continue

            warning_check = warning_rule.get("warning_if")
            if warning_check and warning_check(value):
                message = warning_rule["message"]
                warning_rule.get("recommended")
                warning_msg = f"{path} = {value}: {message}"
                self.warnings.append(warning_msg)

    def validate_vision_config(self) -> bool:
        """Validate vision-specific configuration.

        Returns:
            True if valid, False otherwise
        """
        vision_config = self.config_manager.get("vision")

        if not vision_config:
            self.errors.append(
                "CRITICAL: Vision configuration is missing. Cannot start vision system."
            )
            return False

        # Validate all parameters with rules
        for path in self.VALIDATION_RULES:
            if path.startswith("vision."):
                value = self.config_manager.get(path)
                if value is not None:
                    self.validate_parameter(path, value)

        # Additional cross-field validations
        min_radius = self.config_manager.get("vision.detection.min_ball_radius")
        max_radius = self.config_manager.get("vision.detection.max_ball_radius")

        if (
            min_radius is not None
            and max_radius is not None
            and min_radius >= max_radius
        ):
            self.errors.append(
                f"min_ball_radius ({min_radius}) must be less than max_ball_radius ({max_radius})"
            )

        # Validate YOLO model path if YOLO backend is enabled
        detection_backend = self.config_manager.get(
            "vision.detection.detection_backend"
        )
        if detection_backend == "yolo":
            model_path = self.config_manager.get("vision.detection.yolo_model_path")
            if model_path:
                model_file = Path(model_path)
                if not model_file.exists():
                    self.warnings.append(
                        f"YOLO model file not found: {model_path}. "
                        "YOLO detection will fail until model is available."
                    )

        # Validate camera resolution
        resolution = self.config_manager.get("vision.camera.resolution")
        if resolution is not None:
            if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
                self.errors.append(
                    "camera.resolution must be a list/tuple of 2 integers [width, height]"
                )
            elif not all(isinstance(x, int) and x > 0 for x in resolution):
                self.errors.append("camera.resolution must contain positive integers")

        return len(self.errors) == 0

    def validate_all(self) -> bool:
        """Validate all configuration.

        Returns:
            True if all validation passes, False otherwise
        """
        logger.info("Validating backend configuration...")

        # Clear previous errors/warnings
        self.errors.clear()
        self.warnings.clear()

        # Apply defaults first
        self.apply_defaults()

        # Validate vision config
        vision_valid = self.validate_vision_config()

        # Check for suboptimal values
        self.check_suboptimal_values()

        # Log results
        if self.errors:
            logger.error(
                f"Configuration validation failed with {len(self.errors)} errors:"
            )
            for error in self.errors:
                logger.error(f"  ❌ {error}")

        if self.warnings:
            logger.warning(
                f"Configuration has {len(self.warnings)} warnings (non-critical):"
            )
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")

        if not self.errors and not self.warnings:
            logger.info("✅ Configuration validation passed")

        return vision_valid and len(self.errors) == 0


def validate_configuration(
    config_manager: "ConfigurationModule",
) -> None:
    """Validate configuration at backend startup.

    This function should be called during backend initialization to ensure
    all configuration is valid before starting services.

    Args:
        config_manager: Configuration manager instance

    Raises:
        ConfigValidationError: If critical configuration validation fails
    """
    validator = ConfigValidator(config_manager)

    if not validator.validate_all():
        error_msg = "\n".join(
            [
                "Configuration validation failed!",
                "",
                "Errors:",
                *[f"  - {error}" for error in validator.errors],
            ]
        )
        logger.error(error_msg)
        raise ConfigValidationError(error_msg)

    logger.info("Configuration validation complete")
