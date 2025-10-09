"""Configuration module for the billiards trainer backend."""

from pathlib import Path

from .manager import ConfigurationModule

# Global configuration manager singleton - config_dir should point to parent of 'config' folder
config_manager = ConfigurationModule(config_dir=Path("backend"))

__all__ = ["ConfigurationModule", "config_manager"]
