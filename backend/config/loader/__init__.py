"""Configuration loader package.

This package provides various configuration loaders for different sources
and a merger utility to combine configurations with proper precedence.

Supports the following requirements:
- FR-CFG-001: Load configuration from multiple sources (files, environment, CLI)
- FR-CFG-002: Support configuration file formats (JSON, YAML, INI)
- FR-CFG-003: Merge configurations with proper precedence rules
- FR-CFG-004: Provide default values for all settings
- FR-CFG-005: Support configuration inheritance and overrides
"""

from .cli import CLILoader
from .env import EnvironmentLoader
from .file import FileLoader
from .merger import ConfigurationMerger

__all__ = ["FileLoader", "EnvironmentLoader", "CLILoader", "ConfigurationMerger"]
