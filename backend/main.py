"""Main entry point for the billiards trainer application."""

import argparse
import logging
import os
from typing import Any, NoReturn

from backend.api.main import app, lifespan
from backend.config import Config

logger = logging.getLogger(__name__)


def build_uvicorn_options(
    host: str, port: int, log_level: str, reload: bool
) -> dict[str, Any]:
    """Build server options without Uvicorn's implicit proxy-header rewriting.

    Client IP forwarding is evaluated by the WebSocket limiter against its
    configured trusted proxy networks. Uvicorn must leave the transport peer
    untouched so an untrusted forwarded header cannot influence that check.
    """
    return {
        "host": host,
        "port": port,
        "log_level": log_level,
        "reload": reload,
        "proxy_headers": False,
        "forwarded_allow_ips": "",
    }


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
    parser.add_argument(
        "--performance-audit",
        action="store_true",
        help="Enable opt-in performance audit instrumentation for this process",
    )
    return parser.parse_args()


def main() -> NoReturn:
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    # Set the config file path before any imports that might use config
    Config.set_config_file(args.config)
    if getattr(args, "performance_audit", False):
        # Uvicorn imports the application in its serving process. Keep this
        # override process-local and avoid persisting a diagnostic setting.
        os.environ["BT_PERFORMANCE_AUDIT"] = "1"

    # Get the config instance for convenience
    config = Config()

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
        **build_uvicorn_options(
            host=api_host,
            port=api_port,
            log_level=log_level,
            reload=reload,
        ),
    )

    # This line should never be reached, but satisfies type checker
    raise SystemExit(0)


if __name__ == "__main__":
    main()
