"""Video source configuration migration for backward compatibility.

This migration handles the transition from the legacy device_id-based video
source configuration to the new explicit video_source_type configuration.

Legacy format:
    vision.camera.device_id = 0 (int) -> camera
    vision.camera.device_id = "/path/to/video.mp4" (str) -> video file
    vision.camera.loop_video = true/false

New format:
    vision.camera.video_source_type = "camera" | "file"
    vision.camera.device_id = 0 (int, for cameras only)
    vision.camera.video_file_path = "/path/to/video.mp4" (str, for files only)
    vision.camera.loop_video = true/false (for files only)
"""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class VideoSourceConfigMigration:
    """Migration handler for video source configuration changes.

    This migration ensures backward compatibility when transitioning from the
    legacy device_id-based configuration to the new explicit video_source_type
    configuration format.

    Attributes:
        version: Migration version identifier
        description: Human-readable description of what this migration does
    """

    version = "1.0.0"
    description = (
        "Migrate legacy device_id configurations to new video_source_type format"
    )

    def should_migrate(self, config: dict) -> bool:
        """Check if this configuration needs migration.

        A configuration needs migration if:
        1. device_id is a string (file path) but video_source_type is not set
        2. loop_video is set but video_source_type is not "file"
        3. device_id exists but video_source_type is missing

        Args:
            config: Configuration dictionary to check

        Returns:
            True if migration is needed, False otherwise
        """
        # Navigate to vision.camera configuration
        vision_config = config.get("vision", {})
        camera_config = vision_config.get("camera", {})

        # If no camera config exists, nothing to migrate
        if not camera_config:
            return False

        # Check if video_source_type is already set
        has_video_source_type = "video_source_type" in camera_config
        has_device_id = "device_id" in camera_config
        has_loop_video = "loop_video" in camera_config

        # If video_source_type is already set, check if it's consistent
        if has_video_source_type:
            video_source_type = camera_config["video_source_type"]
            device_id = camera_config.get("device_id")

            # Check for inconsistencies
            if video_source_type == "file":
                # File source should have video_file_path, not int device_id
                if (
                    isinstance(device_id, int)
                    and "video_file_path" not in camera_config
                ):
                    return True
            elif video_source_type == "camera":
                # Camera source should have int device_id, not string
                if isinstance(device_id, str):
                    return True
                # Camera source shouldn't have loop_video
                if has_loop_video:
                    return True

            # No migration needed if everything is consistent
            return False

        # If no video_source_type, check if we can infer it
        if has_device_id:
            device_id = camera_config["device_id"]

            # String device_id indicates file source
            if isinstance(device_id, str):
                return True

            # Int device_id with loop_video indicates file source misconfiguration
            if isinstance(device_id, int) and has_loop_video:
                return True

            # Int device_id without video_source_type needs migration
            if isinstance(device_id, int):
                return True

        return False

    def migrate(self, config: dict) -> dict:
        """Perform the configuration migration.

        This method transforms the legacy configuration format to the new format:
        - If device_id is string: set video_source_type="file",
          video_file_path=device_id, remove string device_id
        - If device_id is int: set video_source_type="camera" (if not already set)
        - Preserve all existing settings
        - Remove deprecated or inconsistent settings

        Args:
            config: Configuration dictionary to migrate

        Returns:
            Migrated configuration dictionary

        Note:
            This method modifies the configuration in-place and also returns it.
        """
        # Navigate to vision.camera configuration
        if "vision" not in config:
            config["vision"] = {}
        if "camera" not in config["vision"]:
            config["vision"]["camera"] = {}

        camera_config = config["vision"]["camera"]
        device_id = camera_config.get("device_id")

        # Case 1: device_id is a string (file path)
        if isinstance(device_id, str):
            logger.info(
                f"Migrating file-based video source: device_id='{device_id}' -> "
                f"video_source_type='file', video_file_path='{device_id}'"
            )

            camera_config["video_source_type"] = "file"
            camera_config["video_file_path"] = device_id

            # Remove the string device_id (it's now in video_file_path)
            del camera_config["device_id"]

            # Preserve loop_video if it exists
            if "loop_video" not in camera_config:
                camera_config["loop_video"] = False

        # Case 2: device_id is an int (camera device)
        elif isinstance(device_id, int):
            # Check if loop_video is mistakenly set for camera
            if "loop_video" in camera_config:
                logger.warning(
                    f"Removing loop_video setting for camera device (device_id={device_id}). "
                    "loop_video is only applicable to video files."
                )
                del camera_config["loop_video"]

            # Set video_source_type if not already set
            if "video_source_type" not in camera_config:
                logger.info(
                    f"Migrating camera-based video source: device_id={device_id} -> "
                    f"video_source_type='camera'"
                )
                camera_config["video_source_type"] = "camera"

        # Case 3: video_source_type exists but configuration is inconsistent
        elif "video_source_type" in camera_config:
            video_source_type = camera_config["video_source_type"]

            if video_source_type == "file":
                # Ensure video_file_path exists
                if "video_file_path" not in camera_config and device_id:
                    if isinstance(device_id, str):
                        camera_config["video_file_path"] = device_id
                        del camera_config["device_id"]
                    else:
                        logger.warning(
                            f"video_source_type='file' but device_id is int ({device_id}). "
                            "This configuration is invalid."
                        )

                # Ensure loop_video has a default
                if "loop_video" not in camera_config:
                    camera_config["loop_video"] = False

            elif video_source_type == "camera":
                # Remove loop_video if it exists
                if "loop_video" in camera_config:
                    logger.warning(
                        "Removing loop_video setting for camera source. "
                        "loop_video is only applicable to video files."
                    )
                    del camera_config["loop_video"]

                # Ensure device_id is int
                if isinstance(device_id, str):
                    logger.warning(
                        f"video_source_type='camera' but device_id is string ('{device_id}'). "
                        "Converting to default camera (device_id=0)."
                    )
                    camera_config["device_id"] = 0
                elif device_id is None:
                    camera_config["device_id"] = 0

        return config

    def validate(self, config: dict) -> list[str]:
        """Validate configuration and return warnings about deprecated usage.

        This method checks for:
        - Deprecated configuration patterns
        - Inconsistent settings
        - Missing required fields for the configured source type

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of warning messages (empty list if no warnings)
        """
        warnings = []

        # Navigate to vision.camera configuration
        vision_config = config.get("vision", {})
        camera_config = vision_config.get("camera", {})

        if not camera_config:
            return warnings

        video_source_type = camera_config.get("video_source_type")
        device_id = camera_config.get("device_id")
        video_file_path = camera_config.get("video_file_path")
        loop_video = camera_config.get("loop_video")

        # Validate video_source_type is set
        if not video_source_type:
            warnings.append(
                "DEPRECATED: 'video_source_type' is not set. "
                "Please explicitly set 'vision.camera.video_source_type' to 'camera' or 'file'."
            )

        # Validate file source configuration
        if video_source_type == "file":
            if not video_file_path:
                warnings.append(
                    "WARNING: video_source_type='file' but 'video_file_path' is not set. "
                    "Please set 'vision.camera.video_file_path' to the path of your video file."
                )

            if device_id is not None:
                if isinstance(device_id, str):
                    warnings.append(
                        "DEPRECATED: Using 'device_id' as a file path is deprecated. "
                        f"Please use 'video_file_path' instead. Current value: '{device_id}'"
                    )
                else:
                    warnings.append(
                        f"WARNING: video_source_type='file' but 'device_id' is set to {device_id}. "
                        "For file sources, use 'video_file_path' instead of 'device_id'."
                    )

        # Validate camera source configuration
        elif video_source_type == "camera":
            if device_id is None:
                warnings.append(
                    "WARNING: video_source_type='camera' but 'device_id' is not set. "
                    "Defaulting to device_id=0."
                )
            elif not isinstance(device_id, int):
                warnings.append(
                    f"ERROR: video_source_type='camera' but 'device_id' is not an integer: {device_id}. "
                    "For camera sources, 'device_id' must be an integer (e.g., 0, 1, 2)."
                )

            if loop_video:
                warnings.append(
                    "WARNING: 'loop_video' is set for camera source. "
                    "This setting is only applicable to file sources and will be ignored."
                )

            if video_file_path:
                warnings.append(
                    f"WARNING: video_source_type='camera' but 'video_file_path' is set to '{video_file_path}'. "
                    "This setting is only applicable to file sources and will be ignored."
                )

        # Validate unknown video_source_type
        elif video_source_type is not None:
            warnings.append(
                f"ERROR: Unknown video_source_type: '{video_source_type}'. "
                "Valid values are 'camera' or 'file'."
            )

        return warnings

    def get_suggested_configuration(self, config: dict) -> str:
        """Generate suggested configuration format for users.

        This provides a helpful message showing the recommended configuration
        format based on the current settings.

        Args:
            config: Current configuration dictionary

        Returns:
            Formatted string with suggested configuration
        """
        vision_config = config.get("vision", {})
        camera_config = vision_config.get("camera", {})

        if not camera_config:
            return ""

        video_source_type = camera_config.get("video_source_type")
        device_id = camera_config.get("device_id")
        video_file_path = camera_config.get("video_file_path")
        loop_video = camera_config.get("loop_video", False)

        suggestion = "\nSuggested configuration format:\n\n"

        if video_source_type == "camera" or (
            isinstance(device_id, int) and not video_file_path
        ):
            suggestion += "For camera sources:\n"
            suggestion += "{\n"
            suggestion += '  "vision": {\n'
            suggestion += '    "camera": {\n'
            suggestion += '      "video_source_type": "camera",\n'
            suggestion += (
                f'      "device_id": {device_id if device_id is not None else 0},\n'
            )
            suggestion += '      "resolution": [1920, 1080],\n'
            suggestion += '      "fps": 30\n'
            suggestion += "    }\n"
            suggestion += "  }\n"
            suggestion += "}\n"

        elif (
            video_source_type == "file" or isinstance(device_id, str) or video_file_path
        ):
            file_path = video_file_path or (
                device_id if isinstance(device_id, str) else "/path/to/video.mp4"
            )
            suggestion += "For video file sources:\n"
            suggestion += "{\n"
            suggestion += '  "vision": {\n'
            suggestion += '    "camera": {\n'
            suggestion += '      "video_source_type": "file",\n'
            suggestion += f'      "video_file_path": "{file_path}",\n'
            suggestion += f'      "loop_video": {str(loop_video).lower()},\n'
            suggestion += '      "resolution": [1920, 1080],\n'
            suggestion += '      "fps": 30\n'
            suggestion += "    }\n"
            suggestion += "  }\n"
            suggestion += "}\n"

        else:
            suggestion += (
                "Please set 'video_source_type' to either 'camera' or 'file'.\n"
            )

        return suggestion


def migrate_config(config: dict) -> tuple[dict, list[str]]:
    """Convenience function to migrate and validate configuration.

    This is a helper function that applies the migration and returns both
    the migrated configuration and any validation warnings.

    Args:
        config: Configuration dictionary to migrate

    Returns:
        Tuple of (migrated_config, warnings_list)
    """
    migration = VideoSourceConfigMigration()

    # Check if migration is needed
    if migration.should_migrate(config):
        logger.info(
            f"Applying video source configuration migration (version {migration.version})"
        )
        config = migration.migrate(config)

    # Validate and collect warnings
    warnings = migration.validate(config)

    # Log warnings
    for warning in warnings:
        if warning.startswith("ERROR:"):
            logger.error(warning)
        elif warning.startswith("WARNING:"):
            logger.warning(warning)
        else:
            logger.info(warning)

    return config, warnings
