"""Main entry point for the billiards trainer application."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn

from backend.api.main import app, lifespan
from backend.config import Config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Billiards Trainer - AI-powered billiards training system"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: ./config.json)",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    # Set the config file path before any imports that might use config
    Config.set_config_file(args.config)

    # Get the config instance for convenience
    config = Config()

    # Configure VAAPI for GPU hardware acceleration BEFORE any video operations
    # This must be done early before OpenCV initializes
    try:
        from backend.vision.gpu_utils import configure_vaapi_env

        configure_vaapi_env()
        logger.info("VAAPI GPU acceleration configured")
    except Exception as e:
        logger.warning(f"Could not configure VAAPI: {e}")

    # Note: Signal handlers are set up in api.shutdown.setup_signal_handlers()
    # which is called during the FastAPI lifespan startup

    # Get API server settings from configuration
    api_host = config.get("api.server.host", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", str(config.get("api.server.port", 8000))))
    reload = config.get("api.server.reload", False)
    log_level = config.get(
        "api.server.log_level", config.get("logging.level", "info")
    ).lower()

    logger.info(f"Starting Billiards Trainer on {api_host}:{api_port}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Reload: {reload}")

    # Start the API server (this never returns normally)
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=api_host,
        port=api_port,
        log_level=log_level,
        reload=reload,
    )

    # This line should never be reached, but satisfies type checker
    raise SystemExit(0)


if __name__ == "__main__":
    main()
