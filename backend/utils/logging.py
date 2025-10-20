"""Logging configuration utilities for the billiards trainer backend."""

import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml


def _get_config_value(key: str, default: any = None) -> any:
    """Get configuration value from config manager or environment.

    Args:
        key: Configuration key (dot notation supported)
        default: Default value if key not found

    Returns:
        Configuration value
    """
    try:
        from backend.config import config

        return config.get(key, default)
    except (ImportError, Exception):
        # Fallback to default if config not available
        return default


def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: Optional[str] = None,
    environment: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Setup logging configuration.

    Args:
        config_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable key for config path override
        environment: Environment name (development, production, testing)
        log_dir: Directory for log files (defaults to configured value)
    """
    # Get configuration values
    if env_key is None:
        env_key = _get_config_value("system.logging.env_key", "LOG_CFG")

    default_config_path = _get_config_value(
        "system.logging.default_config_path", "config/logging.yaml"
    )
    default_log_dir = _get_config_value("system.logging.default_log_dir", "logs")
    log_dir_env_key = _get_config_value("system.logging.log_dir_env_key", "LOG_DIR")

    # Ensure logs directory exists
    if log_dir is None:
        log_dir = Path(os.getenv(log_dir_env_key, default_log_dir))
    logs_dir = Path(log_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Determine config path
    if config_path is None:
        config_path = os.getenv(env_key, default_config_path)

    config_path = Path(config_path)

    if config_path.exists():
        try:
            config_encoding = _get_config_value(
                "system.logging.config_file_encoding", "utf-8"
            )
            with open(config_path, encoding=config_encoding) as config_file:
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
    # Get formatter configuration
    detailed_fmt = _get_config_value(
        "system.logging.formatters.detailed.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    detailed_datefmt = _get_config_value(
        "system.logging.formatters.detailed.datefmt", "%Y-%m-%d %H:%M:%S"
    )
    simple_fmt = _get_config_value(
        "system.logging.formatters.simple.format",
        "%(asctime)s - %(levelname)s - %(message)s",
    )
    simple_datefmt = _get_config_value(
        "system.logging.formatters.simple.datefmt", "%Y-%m-%d %H:%M:%S"
    )

    # Create formatters
    detailed_formatter = logging.Formatter(
        detailed_fmt,
        datefmt=detailed_datefmt,
    )
    simple_formatter = logging.Formatter(
        simple_fmt,
        datefmt=simple_datefmt,
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)

    # Get file handler configuration for app log
    app_log_filename = _get_config_value(
        "system.logging.file_handlers.app_log.filename", "app.log"
    )
    app_log_level = _get_config_value(
        "system.logging.file_handlers.app_log.level", "DEBUG"
    )
    app_log_max_bytes = _get_config_value(
        "system.logging.file_handlers.app_log.max_bytes", 10 * 1024 * 1024
    )
    app_log_backup_count = _get_config_value(
        "system.logging.file_handlers.app_log.backup_count", 5
    )
    app_log_encoding = _get_config_value(
        "system.logging.file_handlers.app_log.encoding", "utf-8"
    )

    # Rotating file handler for all logs
    all_logs_file = logs_dir / app_log_filename
    file_handler = logging.handlers.RotatingFileHandler(
        all_logs_file,
        maxBytes=app_log_max_bytes,
        backupCount=app_log_backup_count,
        encoding=app_log_encoding,
    )
    file_handler.setLevel(getattr(logging, app_log_level.upper(), logging.DEBUG))
    file_handler.setFormatter(detailed_formatter)

    # Get file handler configuration for error log
    error_log_filename = _get_config_value(
        "system.logging.file_handlers.error_log.filename", "error.log"
    )
    error_log_level = _get_config_value(
        "system.logging.file_handlers.error_log.level", "ERROR"
    )
    error_log_max_bytes = _get_config_value(
        "system.logging.file_handlers.error_log.max_bytes", 10 * 1024 * 1024
    )
    error_log_backup_count = _get_config_value(
        "system.logging.file_handlers.error_log.backup_count", 5
    )
    error_log_encoding = _get_config_value(
        "system.logging.file_handlers.error_log.encoding", "utf-8"
    )

    # Rotating file handler for errors only
    error_logs_file = logs_dir / error_log_filename
    error_handler = logging.handlers.RotatingFileHandler(
        error_logs_file,
        maxBytes=error_log_max_bytes,
        backupCount=error_log_backup_count,
        encoding=error_log_encoding,
    )
    error_handler.setLevel(getattr(logging, error_log_level.upper(), logging.ERROR))
    error_handler.setFormatter(detailed_formatter)

    # Configure root logger
    root_logger_level_str = _get_config_value(
        "system.logging.root_logger_level", "DEBUG"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(
        getattr(logging, root_logger_level_str.upper(), logging.DEBUG)
    )  # Capture everything, handlers will filter
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
    default_log_dir = _get_config_value("system.logging.default_log_dir", "logs")
    log_dir_env_key = _get_config_value("system.logging.log_dir_env_key", "LOG_DIR")
    return Path(os.getenv(log_dir_env_key, default_log_dir))


def configure_uvicorn_logging() -> None:
    """Configure uvicorn logging to work with our logging setup."""
    # Get uvicorn configuration
    disable_access = _get_config_value(
        "system.logging.uvicorn.disable_access_logging", True
    )
    configure_error = _get_config_value(
        "system.logging.uvicorn.configure_error_logging", True
    )
    propagate_errors = _get_config_value(
        "system.logging.uvicorn.propagate_errors", True
    )

    # Disable uvicorn's default access logging
    if disable_access:
        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_access.disabled = True

    # Configure uvicorn error logging
    if configure_error:
        uvicorn_error = logging.getLogger("uvicorn.error")
        uvicorn_error.handlers = []
        uvicorn_error.propagate = propagate_errors


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    import platform
    import sys

    logger = get_logger(__name__)

    # Get system info configuration
    log_python = _get_config_value(
        "system.logging.system_info.log_python_version", True
    )
    log_platform_info = _get_config_value(
        "system.logging.system_info.log_platform", True
    )
    log_arch = _get_config_value("system.logging.system_info.log_architecture", True)
    log_proc = _get_config_value("system.logging.system_info.log_processor", True)
    log_wd = _get_config_value("system.logging.system_info.log_working_directory", True)
    header_sep = _get_config_value(
        "system.logging.system_info.header_separator", "=== System Information ==="
    )
    footer_sep = _get_config_value(
        "system.logging.system_info.footer_separator", "=========================="
    )

    logger.info(header_sep)
    if log_python:
        logger.info(f"Python version: {sys.version}")
    if log_platform_info:
        logger.info(f"Platform: {platform.platform()}")
    if log_arch:
        logger.info(f"Architecture: {platform.architecture()}")
    if log_proc:
        logger.info(f"Processor: {platform.processor()}")
    if log_wd:
        logger.info(f"Working directory: {os.getcwd()}")
    logger.info(footer_sep)


def setup_development_logging() -> None:
    """Setup logging specifically for development environment."""
    default_env = _get_config_value("system.logging.default_environment", "development")
    env_key = _get_config_value("system.logging.environment_env_key", "ENVIRONMENT")
    environment = os.getenv(env_key, default_env)
    log_level = os.getenv("LOG_LEVEL")

    # Get environment defaults if log level not set in environment
    if log_level is None:
        log_level = _get_config_value(
            "system.logging.environment_defaults.development.level", "DEBUG"
        )

    log_level = log_level.upper()

    # Convert string log level to integer
    numeric_level = getattr(logging, log_level, logging.DEBUG)

    setup_logging(default_level=numeric_level, environment=environment)

    # Get log system info setting and environment name
    log_sys_info = _get_config_value(
        "system.logging.environment_defaults.development.log_system_info", True
    )
    dev_env_name = _get_config_value(
        "system.logging.environment_names.development", "development"
    )

    # Log system info in development
    if environment == dev_env_name and log_sys_info:
        log_system_info()


def setup_production_logging() -> None:
    """Setup logging specifically for production environment."""
    # Get production default log level and environment name
    prod_level = _get_config_value(
        "system.logging.environment_defaults.production.level", "INFO"
    )
    prod_env_name = _get_config_value(
        "system.logging.environment_names.production", "production"
    )
    numeric_level = getattr(logging, prod_level.upper(), logging.INFO)

    setup_logging(default_level=numeric_level, environment=prod_env_name)

    # Configure uvicorn for production
    configure_uvicorn_logging()


def setup_testing_logging() -> None:
    """Setup logging specifically for testing environment."""
    # Get testing default log level and environment name
    test_level = _get_config_value(
        "system.logging.environment_defaults.testing.level", "WARNING"
    )
    test_env_name = _get_config_value(
        "system.logging.environment_names.testing", "testing"
    )
    numeric_level = getattr(logging, test_level.upper(), logging.WARNING)

    setup_logging(default_level=numeric_level, environment=test_env_name)


# Convenience function to setup logging based on environment
def auto_setup_logging() -> None:
    """Automatically setup logging based on the ENVIRONMENT variable."""
    default_env = _get_config_value("system.logging.default_environment", "development")
    env_key = _get_config_value("system.logging.environment_env_key", "ENVIRONMENT")
    prod_env_name = _get_config_value(
        "system.logging.environment_names.production", "production"
    )
    test_env_name = _get_config_value(
        "system.logging.environment_names.testing", "testing"
    )
    environment = os.getenv(env_key, default_env)

    if environment == prod_env_name:
        setup_production_logging()
    elif environment == test_env_name:
        setup_testing_logging()
    else:
        setup_development_logging()
