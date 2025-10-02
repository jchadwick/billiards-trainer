"""Logging configuration utilities for the billiards trainer backend."""

import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG",
    environment: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Setup logging configuration.

    Args:
        config_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable key for config path override
        environment: Environment name (development, production, testing)
        log_dir: Directory for log files (defaults to ./logs)
    """
    # Ensure logs directory exists
    if log_dir is None:
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
    logs_dir = Path(log_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

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
            _setup_default_logging(default_level, logs_dir)
    else:
        print(
            f"Logging config file {config_path} not found. Using default configuration."
        )
        _setup_default_logging(default_level, logs_dir)


def _setup_default_logging(level: int, logs_dir: Path) -> None:
    """Setup default logging with both console and file handlers.

    Args:
        level: Logging level
        logs_dir: Directory for log files
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)

    # Rotating file handler for all logs
    all_logs_file = logs_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        all_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Rotating file handler for errors only
    error_logs_file = logs_dir / "error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_log_directory() -> Path:
    """Get the configured log directory path.

    Returns:
        Path to the log directory
    """
    return Path(os.getenv("LOG_DIR", "logs"))


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
