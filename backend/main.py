"""Main entry point for the billiards trainer application."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn

from backend.api.main import app, lifespan
from backend.config.manager import ConfigurationModule
from backend.config.validation import ConfigValidationError, validate_configuration

logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """Main entry point for the application."""
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

    # Load configuration
    config_module = ConfigurationModule()

    # Validate configuration at startup (fail fast on critical errors)
    try:
        validate_configuration(config_module)
    except ConfigValidationError as e:
        logger.critical("Configuration validation failed!")
        logger.critical(str(e))
        logger.critical("Cannot start backend with invalid configuration")
        sys.exit(1)

    # Get API server settings from configuration
    api_host = config_module.get("api.server.host", "0.0.0.0")
    api_port = int(
        os.getenv("API_PORT", str(config_module.get("api.server.port", 8000)))
    )
    reload = config_module.get("api.server.reload", False)
    log_level = config_module.get(
        "api.server.log_level", config_module.get("logging.level", "info")
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
