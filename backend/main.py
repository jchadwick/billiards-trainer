"""Main entry point for the billiards trainer application."""

import asyncio
import logging
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


def signal_handler(signum: int, frame: object) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main() -> NoReturn:
    """Main entry point for the application."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    config_module = ConfigurationModule()
    config = config_module.get_config()

    # Get API settings
    api_host = config.get("api.host", "0.0.0.0")
    api_port = config.get("api.port", 8000)
    log_level = config.get("logging.level", "INFO").lower()

    logger.info(f"Starting Billiards Trainer on {api_host}:{api_port}")
    logger.info(f"Log level: {log_level}")

    # Start the API server
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=api_host,
        port=api_port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
