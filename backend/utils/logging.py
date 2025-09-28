"""Logging configuration utilities for the billiards trainer backend."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
    environment: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        config_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable key for config path override
        environment: Environment name (development, production, testing)
    """
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Determine config path
    if config_path is None:
        config_path = os.getenv(env_key, "config/logging.yaml")

    config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

            # Apply environment-specific overrides
            if environment and environment in config:
                env_config = config[environment]

                # Override formatters
                if "formatters" in env_config:
                    config["formatters"].update(env_config["formatters"])

                # Override handlers
                if "handlers" in env_config:
                    config["handlers"].update(env_config["handlers"])

                # Override loggers
                if "loggers" in env_config:
                    config["loggers"].update(env_config["loggers"])

            logging.config.dictConfig(config)

        except Exception as e:
            print(f"Error loading logging configuration from {config_path}: {e}")
            print("Using default logging configuration")
            logging.basicConfig(
                level=default_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
    else:
        print(
            f"Logging config file {config_path} not found. Using default configuration."
        )
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(logger_name: str, level: int) -> None:
    """Set log level for a specific logger.

    Args:
        logger_name: Name of the logger
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


def configure_uvicorn_logging() -> None:
    """Configure uvicorn logging to work with our logging setup."""
    # Disable uvicorn's default access logging
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True

    # Configure uvicorn error logging
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_error.handlers = []
    uvicorn_error.propagate = True


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    import platform
    import sys

    logger = get_logger(__name__)

    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("==========================")


def setup_development_logging() -> None:
    """Setup logging specifically for development environment."""
    environment = os.getenv("ENVIRONMENT", "development")
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # Convert string log level to integer
    numeric_level = getattr(logging, log_level, logging.DEBUG)

    setup_logging(default_level=numeric_level, environment=environment)

    # Log system info in development
    if environment == "development":
        log_system_info()


def setup_production_logging() -> None:
    """Setup logging specifically for production environment."""
    setup_logging(default_level=logging.INFO, environment="production")

    # Configure uvicorn for production
    configure_uvicorn_logging()


def setup_testing_logging() -> None:
    """Setup logging specifically for testing environment."""
    setup_logging(default_level=logging.WARNING, environment="testing")


# Convenience function to setup logging based on environment
def auto_setup_logging() -> None:
    """Automatically setup logging based on the ENVIRONMENT variable."""
    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        setup_production_logging()
    elif environment == "testing":
        setup_testing_logging()
    else:
        setup_development_logging()


# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    auto_setup_logging()

    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
