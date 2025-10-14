"""Configuration module for the billiards trainer backend."""

from pathlib import Path

from .manager import ConfigurationModule
from .validation import ConfigValidationError, ConfigValidator, validate_configuration

# Global configuration manager singleton - config_dir should point to the config folder
config_manager = ConfigurationModule(config_dir=Path("backend/config"))

__all__ = [
    "ConfigurationModule",
    "config_manager",
    "ConfigValidator",
    "ConfigValidationError",
    "validate_configuration",
]
