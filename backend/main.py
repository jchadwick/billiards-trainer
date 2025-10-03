"""Main entry point for the billiards trainer application."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn

# Add backend to path
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from api.main import app, lifespan
from config.manager import ConfigurationModule

logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """Main entry point for the application."""
    # Configure VAAPI for GPU hardware acceleration BEFORE any video operations
    # This must be done early before OpenCV initializes
    try:
        from vision.gpu_utils import configure_vaapi_env

        configure_vaapi_env()
        logger.info("VAAPI GPU acceleration configured")
    except Exception as e:
        logger.warning(f"Could not configure VAAPI: {e}")

    # Note: Signal handlers are set up in api.shutdown.setup_signal_handlers()
    # which is called during the FastAPI lifespan startup

    # Load configuration
    config_module = ConfigurationModule()

    # Get API settings - default to 0.0.0.0 for LAN access
    api_host = "0.0.0.0"  # Bind to all interfaces
    api_port = int(os.getenv("API_PORT", "8000"))
    log_level = config_module.get("logging.level", "INFO").lower()
    logger.info(f"Starting Billiards Trainer on {api_host}:{api_port}")
    logger.info(f"Log level: {log_level}")

    # Start the API server (this never returns normally)
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=api_host,
        port=api_port,
        log_level=log_level,
        reload=False,
    )

    # This line should never be reached, but satisfies type checker
    raise SystemExit(0)


if __name__ == "__main__":
    main()
