"""Configuration migration system for handling backward compatibility.

This module provides a migration framework for configuration changes across
versions. Migrations are automatically applied when configurations are loaded
to ensure backward compatibility with older configuration formats.
"""

from .video_source_migration import VideoSourceConfigMigration

__all__ = ["VideoSourceConfigMigration"]
