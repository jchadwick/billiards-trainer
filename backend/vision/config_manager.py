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
import os
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from config import Config, config
except ImportError:
    # Fallback for development/testing
    logging.warning("Configuration module not available, using fallback")
    Config = None  # type: ignore
    config = None  # type: ignore

from .capture import CameraCapture

logger = logging.getLogger(__name__)


class VisionConfigurationManager:
    """Configuration manager for the vision module.

    Provides integration with the central configuration system,
    enabling type-safe configuration management and real-time updates.
    """

    def __init__(self, config_manager: Optional[Any] = None) -> None:
        """Initialize vision configuration manager.

        Args:
            config_manager: Central configuration manager instance
        """
        self._config_manager = config_manager
        self._vision_config: Optional[dict[str, Any]] = None
        self._camera_capture: Optional[CameraCapture] = None

        # Configuration change callbacks
        self._config_callbacks: list[Callable[[dict[str, Any]], None]] = []

    def _get_default_config(self) -> dict[str, Any]:
        """Get default vision configuration from config manager.

        Returns:
            Default vision configuration dictionary
        """
        if self._config_manager:
            # Get defaults from central configuration manager
            try:
                return self._config_manager.get_module_config("vision", dict)
            except Exception:
                pass

        # Minimal fallback if config manager is not available
        # Note: All actual defaults should be defined in backend/config/default.json
        return {
            "camera": {},
            "detection": {},
            "processing": {},
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
            if self._config_manager:
                # Load configuration from central manager
                self._vision_config = self._config_manager.get_module_config(
                    "vision", dict
                )

                # Register for configuration updates
                self._config_manager.register_update_callback(
                    "vision", self._on_config_update
                )

                logger.info("Vision configuration loaded from central manager")
            else:
                # Use default configuration from config system
                self._vision_config = self._get_default_config()
                logger.info("Using default vision configuration")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision configuration: {e}")
            self._vision_config = self._get_default_config()
            return False

    def get_config(self) -> dict[str, Any]:
        """Get current vision configuration.

        Returns:
            Current configuration dictionary
        """
        if self._vision_config is None:
            return self._get_default_config()

        if hasattr(self._vision_config, "model_dump"):
            # Pydantic model
            return self._vision_config.model_dump()
        else:
            # Dictionary
            if isinstance(self._vision_config, dict):
                return self._vision_config.copy()
            return self._vision_config

    def is_video_file_input(self) -> bool:
        """Check if video input is from a file.

        Returns:
            True if video_file_path is set or device_id is a string (backward compatibility)
        """
        config = self.get_config()
        camera_config = config.get("camera", {})

        # Check if video_file_path is set
        if camera_config.get("video_file_path"):
            return True

        # Backward compatibility: check if device_id is a string
        device_id = camera_config.get("device_id", 0)
        return isinstance(device_id, str)

    def get_video_file_path(self) -> Optional[str]:
        """Get the video file path if configured for file input.

        Returns:
            Absolute path to video file, or None if not configured for file input
            For backward compatibility, also checks device_id if it's a string

        Raises:
            FileNotFoundError: If video file path is configured but file doesn't exist
        """
        config = self.get_config()
        camera_config = config.get("camera", {})

        video_path = None

        # Check video_file_path field first
        video_path = camera_config.get("video_file_path")

        # Backward compatibility: check device_id if it's a string
        if video_path is None:
            device_id = camera_config.get("device_id")
            if isinstance(device_id, str):
                video_path = device_id

        if video_path is None:
            return None

        # Convert to Path object for validation
        path_obj = Path(video_path)

        # Make path absolute if it's relative
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj

        # Validate file exists and is readable
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {path_obj}")

        if not path_obj.is_file():
            raise ValueError(f"Video path is not a file: {path_obj}")

        if not os.access(path_obj, os.R_OK):
            raise PermissionError(f"Video file is not readable: {path_obj}")

        return str(path_obj.resolve())

    def resolve_camera_config(self) -> dict[str, Any]:
        """Resolve camera configuration for CameraCapture.

        Returns:
            CameraCapture-compatible configuration dictionary with:
            - device_id: video_file_path if set, otherwise device_id setting, otherwise 0
            - All camera settings (resolution, fps, etc.)
            - Video file settings (loop_video, video_start_frame, video_end_frame)
            - Handles backward compatibility with legacy device_id configurations
        """
        config = self.get_config()
        camera_config = config.get("camera", {})

        # Determine device_id: video_file_path takes precedence, then device_id, then default to 0
        video_file_path = camera_config.get("video_file_path")

        if video_file_path:
            # Use video file path as device_id
            device_id = self.get_video_file_path()
        else:
            # Use device_id setting (can be int for camera, or string for stream/file)
            device_id = camera_config.get("device_id", 0)

        # Get default camera config for fallback values
        default_config = self._get_default_config()
        default_camera = default_config.get("camera", {})

        return {
            "device_id": device_id,
            "backend": camera_config.get(
                "backend", default_camera.get("backend", "auto")
            ),
            "resolution": tuple(
                camera_config.get(
                    "resolution", default_camera.get("resolution", [1920, 1080])
                )
            ),
            "fps": camera_config.get("fps", default_camera.get("fps", 30)),
            "exposure_mode": camera_config.get(
                "exposure_mode", default_camera.get("exposure_mode", "auto")
            ),
            "exposure_value": camera_config.get(
                "exposure_value", default_camera.get("exposure_value")
            ),
            "gain": camera_config.get("gain", default_camera.get("gain", 1.0)),
            "buffer_size": camera_config.get(
                "buffer_size", default_camera.get("buffer_size", 1)
            ),
            "auto_reconnect": camera_config.get(
                "auto_reconnect", default_camera.get("auto_reconnect", True)
            ),
            "reconnect_delay": camera_config.get(
                "reconnect_delay", default_camera.get("reconnect_delay", 1.0)
            ),
            "max_reconnect_attempts": camera_config.get(
                "max_reconnect_attempts",
                default_camera.get("max_reconnect_attempts", 5),
            ),
            "loop_video": camera_config.get(
                "loop_video", default_camera.get("loop_video", False)
            ),
            "video_start_frame": camera_config.get(
                "video_start_frame", default_camera.get("video_start_frame", 0)
            ),
            "video_end_frame": camera_config.get(
                "video_end_frame", default_camera.get("video_end_frame")
            ),
        }

    def get_camera_config(self) -> dict[str, Any]:
        """Get camera-specific configuration.

        Returns:
            Camera configuration dictionary suitable for CameraCapture

        Note:
            This method now uses resolve_camera_config() internally.
            Consider using resolve_camera_config() directly for new code.
        """
        return self.resolve_camera_config()

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
                new_config = current_dict

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

    def _on_config_update(self, new_config: dict[str, Any]) -> None:
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

                    # Always use dict format
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
        default_config = self._get_default_config()

        return {
            "debug": config.get("debug", default_config.get("debug", False)),
            "save_debug_images": config.get(
                "save_debug_images", default_config.get("save_debug_images", False)
            ),
            "debug_output_path": config.get(
                "debug_output_path",
                default_config.get("debug_output_path", "/tmp/vision_debug"),
            ),
        }

    def get_detection_config(self) -> dict[str, Any]:
        """Get detection algorithm configuration.

        Returns:
            Detection configuration dictionary
        """
        config = self.get_config()
        default_config = self._get_default_config()

        # Return detection config with defaults as fallback
        return config.get("detection", default_config.get("detection", {}))

    def get_processing_config(self) -> dict[str, Any]:
        """Get image processing configuration.

        Returns:
            Processing configuration dictionary
        """
        config = self.get_config()
        default_config = self._get_default_config()

        # Return processing config with defaults as fallback
        return config.get("processing", default_config.get("processing", {}))

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

            # Check video file path if configured
            video_file_path = camera_config.get("video_file_path")
            if video_file_path:
                # Validate file exists and is readable
                try:
                    path_obj = Path(video_file_path)
                    if not path_obj.is_absolute():
                        path_obj = Path.cwd() / path_obj

                    if not path_obj.exists():
                        errors.append(
                            f"camera.video_file_path points to non-existent file: {video_file_path}"
                        )
                    elif not path_obj.is_file():
                        errors.append(
                            f"camera.video_file_path is not a file: {video_file_path}"
                        )
                    elif not os.access(path_obj, os.R_OK):
                        errors.append(
                            f"camera.video_file_path is not readable: {video_file_path}"
                        )
                except Exception as e:
                    errors.append(f"camera.video_file_path validation error: {e}")

            # Validate video frame range
            video_start_frame = camera_config.get("video_start_frame", 0)
            video_end_frame = camera_config.get("video_end_frame")

            if video_end_frame is not None:
                if not isinstance(video_start_frame, int) or video_start_frame < 0:
                    errors.append(
                        "camera.video_start_frame must be a non-negative integer"
                    )
                if not isinstance(video_end_frame, int) or video_end_frame < 0:
                    errors.append(
                        "camera.video_end_frame must be a non-negative integer"
                    )
                if (
                    isinstance(video_start_frame, int)
                    and isinstance(video_end_frame, int)
                    and video_start_frame >= video_end_frame
                ):
                    errors.append(
                        f"camera.video_start_frame ({video_start_frame}) must be less than "
                        f"camera.video_end_frame ({video_end_frame})"
                    )

            # Check required camera fields
            if "device_id" in camera_config:
                device_id = camera_config["device_id"]
                # device_id can be int for camera, or string for file/stream (backward compatibility)
                if not isinstance(device_id, (int, str)):
                    errors.append(
                        "camera.device_id must be a non-negative integer or string"
                    )
                elif isinstance(device_id, int) and device_id < 0:
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

            # Basic validation passed
            pass

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
        default_config = self._get_default_config()

        return {
            "calibration_auto_save": config.get(
                "calibration_auto_save",
                default_config.get("calibration_auto_save", True),
            ),
            "debug_output_path": config.get(
                "debug_output_path",
                default_config.get("debug_output_path", "/tmp/vision_debug"),
            ),
        }


# Factory function for easy integration
def create_vision_config_manager(
    config_manager: Optional[Any] = None,
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
