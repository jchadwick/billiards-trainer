"""Video Module - Standalone Process Entry Point.

This module provides the entry point for running the Video Module as a standalone
process. It handles configuration loading, environment overrides, and process
initialization.

Usage:
    # Run with default config.json
    python -m backend.video

    # Run with custom config file
    python -m backend.video /path/to/config.json

    # Run with environment overrides (useful for testing)
    VIDEO_FILE=/path/to/video.mp4 python -m backend.video

Environment Variables:
    VIDEO_FILE: Override camera device_id with video file path (enables loop mode)
    LOG_LEVEL: Override logging level (DEBUG, INFO, WARNING, ERROR)
"""

import logging
import os
import sys
from pathlib import Path

from backend.config import Config
from backend.video.process import VideoProcess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    """Configure logging from config and environment.

    Args:
        config: Config instance with loaded configuration
    """
    # Get log level from environment or config
    log_level = os.environ.get("LOG_LEVEL")
    if not log_level:
        log_level = config.get("system.logging.level", "INFO")

    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.getLogger().setLevel(numeric_level)

    logger.info(f"Logging configured: level={log_level}")


def apply_environment_overrides(config: Config) -> None:
    """Apply environment variable overrides to configuration.

    This allows testing with video files and other overrides without
    modifying the config file.

    Args:
        config: Config instance to modify
    """
    overrides_applied = []

    # VIDEO_FILE override - useful for testing
    video_file = os.environ.get("VIDEO_FILE")
    if video_file:
        config.set("vision.camera.video_file_path", video_file)
        config.set("vision.camera.loop_video", True)
        overrides_applied.append(f"video_file_path={video_file} (loop=True)")

    # LOG_LEVEL override
    log_level = os.environ.get("LOG_LEVEL")
    if log_level:
        overrides_applied.append(f"log_level={log_level}")

    if overrides_applied:
        logger.info(f"Environment overrides applied: {', '.join(overrides_applied)}")


def main() -> int:
    """Main entry point for Video Module process.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger.info("Video Module process starting...")

    try:
        # Load config
        config_instance = Config()

        # Check for custom config file path
        if len(sys.argv) > 1:
            config_path = Path(sys.argv[1])
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return 1

            logger.info(f"Loading config from: {config_path}")
            Config.set_config_file(config_path)
        else:
            # Use default config.json from current directory
            default_config = Path.cwd() / "config.json"
            if default_config.exists():
                logger.info(f"Loading config from: {default_config}")
                Config.set_config_file(default_config)
            else:
                logger.warning(
                    f"No config file found at {default_config}, using defaults"
                )

        # Apply environment overrides
        apply_environment_overrides(config_instance)

        # Setup logging after config is loaded
        setup_logging(config_instance)

        # Create and start Video Module process
        logger.info("Creating VideoProcess instance...")
        process = VideoProcess(config_instance)

        logger.info("Starting VideoProcess (blocking call)...")
        exit_code = process.start()

        logger.info(f"VideoProcess exited with code: {exit_code}")
        return exit_code

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting...")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
