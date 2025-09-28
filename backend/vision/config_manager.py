"""Vision Configuration Manager.

Integrates the vision module with the central configuration system.
Provides type-safe configuration loading, validation, and runtime updates.

Features:
- Integration with the configuration module
- Real-time configuration updates
- Validation and error handling
- Profile-based configuration
- Camera-specific settings management
"""

import logging
from typing import Any, Callable, Optional

try:
    from ..config.manager import ConfigurationManager
    from ..config.models.schemas import CameraSettings, VisionConfig
except ImportError:
    # Fallback for development/testing
    logging.warning("Configuration module not available, using fallback")
    ConfigurationManager = None
    VisionConfig = None
    CameraSettings = None

from .capture import CameraCapture

logger = logging.getLogger(__name__)


class VisionConfigurationManager:
    """Configuration manager for the vision module.

    Provides integration with the central configuration system,
    enabling type-safe configuration management and real-time updates.
    """

    def __init__(self, config_manager: Optional["ConfigurationManager"] = None):
        """Initialize vision configuration manager.

        Args:
            config_manager: Central configuration manager instance
        """
        self._config_manager = config_manager
        self._vision_config: Optional[VisionConfig] = None
        self._camera_capture: Optional[CameraCapture] = None

        # Configuration change callbacks
        self._config_callbacks: list[Callable[[VisionConfig], None]] = []

        # Default configuration
        self._default_config = self._create_default_config()

    def _create_default_config(self) -> dict[str, Any]:
        """Create default vision configuration."""
        return {
            "camera": {
                "device_id": 0,
                "backend": "auto",
                "resolution": (1920, 1080),
                "fps": 30,
                "exposure_mode": "auto",
                "exposure_value": None,
                "gain": 1.0,
                "buffer_size": 1,
                "auto_reconnect": True,
                "reconnect_delay": 1.0,
                "max_reconnect_attempts": 5,
            },
            "detection": {
                "table_edge_threshold": 0.7,
                "min_table_area": 0.3,
                "min_ball_radius": 10,
                "max_ball_radius": 40,
                "ball_detection_method": "hough",
                "ball_sensitivity": 0.8,
                "cue_detection_enabled": True,
                "min_cue_length": 100,
                "cue_line_threshold": 0.6,
            },
            "processing": {
                "use_gpu": False,
                "enable_preprocessing": True,
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

    def initialize(self) -> bool:
        """Initialize configuration manager and load settings.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self._config_manager and VisionConfig:
                # Load configuration from central manager
                self._vision_config = self._config_manager.get_module_config(
                    "vision", VisionConfig
                )

                # Register for configuration updates
                self._config_manager.register_update_callback(
                    "vision", self._on_config_update
                )

                logger.info("Vision configuration loaded from central manager")
            else:
                # Use default configuration
                self._vision_config = self._default_config
                logger.info("Using default vision configuration")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision configuration: {e}")
            self._vision_config = self._default_config
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current vision configuration.

        Returns:
            Current configuration dictionary
        """
        if self._vision_config is None:
            return self._default_config.copy()

        if hasattr(self._vision_config, "model_dump"):
            # Pydantic model
            return self._vision_config.model_dump()
        else:
            # Dictionary
            return self._vision_config.copy()

    def get_camera_config(self) -> dict[str, Any]:
        """Get camera-specific configuration.

        Returns:
            Camera configuration dictionary suitable for CameraCapture
        """
        config = self.get_config()
        camera_config = config.get("camera", {})

        # Convert to format expected by CameraCapture
        return {
            "device_id": camera_config.get("device_id", 0),
            "backend": camera_config.get("backend", "auto"),
            "resolution": tuple(camera_config.get("resolution", [1920, 1080])),
            "fps": camera_config.get("fps", 30),
            "exposure_mode": camera_config.get("exposure_mode", "auto"),
            "exposure_value": camera_config.get("exposure_value"),
            "gain": camera_config.get("gain", 1.0),
            "buffer_size": camera_config.get("buffer_size", 1),
            "auto_reconnect": camera_config.get("auto_reconnect", True),
            "reconnect_delay": camera_config.get("reconnect_delay", 1.0),
            "max_reconnect_attempts": camera_config.get("max_reconnect_attempts", 5),
        }

    def update_config(self, updates: dict[str, Any]) -> bool:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Returns:
            True if update successful, False otherwise
        """
        try:
            if self._config_manager and hasattr(self._vision_config, "model_copy"):
                # Update Pydantic model
                current_dict = self._vision_config.model_dump()
                current_dict.update(updates)

                # Validate new configuration
                new_config = VisionConfig(**current_dict)

                # Update in central manager
                self._config_manager.set_module_config("vision", new_config)
                self._vision_config = new_config

            else:
                # Update dictionary directly
                if isinstance(self._vision_config, dict):
                    self._vision_config.update(updates)
                else:
                    config_dict = self._default_config.copy()
                    config_dict.update(updates)
                    self._vision_config = config_dict

            # Notify callbacks
            self._notify_config_callbacks()

            # Update camera capture if active
            if self._camera_capture and "camera" in updates:
                camera_updates = updates["camera"]
                self._camera_capture.update_config(camera_updates)

            logger.info(f"Vision configuration updated: {updates}")
            return True

        except Exception as e:
            logger.error(f"Failed to update vision configuration: {e}")
            return False

    def set_camera_capture(self, camera_capture: CameraCapture) -> None:
        """Set camera capture instance for configuration updates.

        Args:
            camera_capture: Camera capture instance
        """
        self._camera_capture = camera_capture

    def register_config_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register callback for configuration changes.

        Args:
            callback: Function to call when configuration changes
        """
        self._config_callbacks.append(callback)

    def _notify_config_callbacks(self) -> None:
        """Notify all registered callbacks of configuration changes."""
        config = self.get_config()
        for callback in self._config_callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Configuration callback error: {e}")

    def _on_config_update(self, new_config: VisionConfig) -> None:
        """Handle configuration update from central manager."""
        self._vision_config = new_config
        self._notify_config_callbacks()

        # Update camera capture if active
        if self._camera_capture:
            camera_config = self.get_camera_config()
            self._camera_capture.update_config(camera_config)

        logger.info("Vision configuration updated from central manager")

    def save_config(self, profile_name: Optional[str] = None) -> bool:
        """Save current configuration to storage.

        Args:
            profile_name: Optional profile name to save as

        Returns:
            True if save successful, False otherwise
        """
        try:
            if self._config_manager:
                return self._config_manager.save_profile(profile_name or "default")
            else:
                logger.warning("No configuration manager available for saving")
                return False

        except Exception as e:
            logger.error(f"Failed to save vision configuration: {e}")
            return False

    def load_profile(self, profile_name: str) -> bool:
        """Load configuration from named profile.

        Args:
            profile_name: Name of profile to load

        Returns:
            True if load successful, False otherwise
        """
        try:
            if self._config_manager:
                profile = self._config_manager.load_profile(profile_name)
                if profile and "vision" in profile:
                    vision_config = profile["vision"]

                    if VisionConfig:
                        self._vision_config = VisionConfig(**vision_config)
                    else:
                        self._vision_config = vision_config

                    self._notify_config_callbacks()
                    logger.info(
                        f"Vision configuration loaded from profile: {profile_name}"
                    )
                    return True
                else:
                    logger.warning(
                        f"No vision configuration in profile: {profile_name}"
                    )
                    return False
            else:
                logger.warning("No configuration manager available for loading")
                return False

        except Exception as e:
            logger.error(f"Failed to load profile {profile_name}: {e}")
            return False

    def get_debug_config(self) -> dict[str, Any]:
        """Get debug-specific configuration.

        Returns:
            Debug configuration dictionary
        """
        config = self.get_config()
        return {
            "debug": config.get("debug", False),
            "save_debug_images": config.get("save_debug_images", False),
            "debug_output_path": config.get("debug_output_path", "/tmp/vision_debug"),
        }

    def get_detection_config(self) -> dict[str, Any]:
        """Get detection algorithm configuration.

        Returns:
            Detection configuration dictionary
        """
        config = self.get_config()
        return config.get(
            "detection",
            {
                "table_edge_threshold": 0.7,
                "min_table_area": 0.3,
                "min_ball_radius": 10,
                "max_ball_radius": 40,
                "ball_detection_method": "hough",
                "ball_sensitivity": 0.8,
                "cue_detection_enabled": True,
                "min_cue_length": 100,
                "cue_line_threshold": 0.6,
            },
        )

    def get_processing_config(self) -> dict[str, Any]:
        """Get image processing configuration.

        Returns:
            Processing configuration dictionary
        """
        config = self.get_config()
        return config.get(
            "processing",
            {
                "use_gpu": False,
                "enable_preprocessing": True,
                "blur_kernel_size": 5,
                "morphology_kernel_size": 3,
                "enable_tracking": True,
                "tracking_max_distance": 50,
                "frame_skip": 0,
            },
        )

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Validate camera configuration
            camera_config = config.get("camera", {})

            # Check required camera fields
            if "device_id" in camera_config:
                device_id = camera_config["device_id"]
                if not isinstance(device_id, int) or device_id < 0:
                    errors.append("camera.device_id must be a non-negative integer")

            if "fps" in camera_config:
                fps = camera_config["fps"]
                if not isinstance(fps, int) or not (15 <= fps <= 120):
                    errors.append("camera.fps must be an integer between 15 and 120")

            if "resolution" in camera_config:
                resolution = camera_config["resolution"]
                if (
                    not isinstance(resolution, (list, tuple))
                    or len(resolution) != 2
                    or not all(isinstance(x, int) and x > 0 for x in resolution)
                ):
                    errors.append(
                        "camera.resolution must be a tuple of two positive integers"
                    )

            # Validate detection configuration
            detection_config = config.get("detection", {})

            if (
                "min_ball_radius" in detection_config
                and "max_ball_radius" in detection_config
            ):
                min_r = detection_config["min_ball_radius"]
                max_r = detection_config["max_ball_radius"]
                if min_r >= max_r:
                    errors.append(
                        "detection.min_ball_radius must be less than max_ball_radius"
                    )

            # Use Pydantic validation if available
            if VisionConfig:
                try:
                    VisionConfig(**config)
                except Exception as e:
                    errors.append(f"Pydantic validation error: {e}")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return len(errors) == 0, errors

    def create_camera_capture(self) -> CameraCapture:
        """Create a new CameraCapture instance with current configuration.

        Returns:
            Configured CameraCapture instance
        """
        camera_config = self.get_camera_config()
        camera_capture = CameraCapture(camera_config)

        # Store reference for configuration updates
        self._camera_capture = camera_capture

        return camera_capture

    def get_calibration_config(self) -> dict[str, Any]:
        """Get calibration-related configuration.

        Returns:
            Calibration configuration dictionary
        """
        config = self.get_config()
        return {
            "calibration_auto_save": config.get("calibration_auto_save", True),
            "debug_output_path": config.get("debug_output_path", "/tmp/vision_debug"),
        }


# Factory function for easy integration
def create_vision_config_manager(
    config_manager: Optional["ConfigurationManager"] = None,
) -> VisionConfigurationManager:
    """Factory function to create and initialize vision configuration manager.

    Args:
        config_manager: Optional central configuration manager

    Returns:
        Initialized VisionConfigurationManager instance
    """
    manager = VisionConfigurationManager(config_manager)
    manager.initialize()
    return manager
