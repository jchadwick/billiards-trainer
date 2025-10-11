"""Main configuration management module."""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .loader.cli import CLILoader
from .loader.env import EnvironmentLoader
from .migrations import VideoSourceConfigMigration
from .models.schemas import (
    ConfigChange,
    ConfigFormat,
    ConfigProfile,
    ConfigSource,
    ConfigValue,
    create_default_config,
)
from .profiles import ProfileManager, ProfileManagerError
from .storage.backup import BackupError, BackupMetadata, ConfigBackup
from .storage.persistence import ConfigPersistence, ConfigPersistenceError
from .utils.watcher import ConfigChangeEvent, ConfigWatcher


class ConfigurationModule:
    """Main configuration interface providing centralized settings management."""

    def __init__(
        self, config_dir: Path = Path("config"), enable_hot_reload: bool = True
    ):
        """Initialize configuration module.

        Args:
            config_dir: Directory containing configuration files
            enable_hot_reload: Whether to enable hot reload functionality
        """
        self.config_dir = Path(config_dir)
        self._settings = create_default_config()
        self._data: dict[str, ConfigValue] = {}
        self._schemas: dict[str, dict] = {}
        self._subscriptions: dict[str, Callable[[ConfigChange], None]] = {}
        self._history: list[ConfigChange] = []
        self._module_specs: dict[str, dict] = {}

        # Hot reload functionality
        self._enable_hot_reload = enable_hot_reload
        self._config_watcher: Optional[ConfigWatcher] = None
        self._watched_files: list[Path] = []

        # Initialize persistence system
        self._persistence = ConfigPersistence(base_dir=self.config_dir)

        # Initialize loaders
        self._env_loader = EnvironmentLoader()
        self._cli_loader = CLILoader()

        # Initialize backup system
        self._backup = ConfigBackup(
            backup_dir=self.config_dir / "backups",
            max_backups=self._settings.persistence.backup_count,
            compression=self._settings.persistence.compression,
            verify_integrity=True,
        )

        # Initialize profile manager
        self._profile_manager = ProfileManager(
            persistence=self._persistence, config_dir=self.config_dir
        )

        # Initialize migrations
        self._migrations = [VideoSourceConfigMigration()]

        # Initialize directory structure
        self._init_directories()

        # Load default configuration
        self._load_defaults()

        # Load configuration files if they exist
        self._load_initial_config()

        # Initialize hot reload if enabled
        if self._enable_hot_reload:
            try:
                if hasattr(self, "_init_hot_reload") and callable(
                    getattr(self, "_init_hot_reload", None)
                ):
                    self._init_hot_reload()
                else:
                    print("Warning: _init_hot_reload method not found or not callable")
                    self._enable_hot_reload = False
            except Exception as e:
                print(f"Error initializing hot reload during construction: {e}")
                self._enable_hot_reload = False

    def _init_directories(self) -> None:
        """Initialize required directories."""
        directories = [
            self.config_dir,
            self.config_dir / "profiles",
            self._settings.system.paths.data_dir,
            self._settings.system.paths.log_dir,
            self._settings.system.paths.cache_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        # Basic default configuration
        defaults = {
            "app": {"name": "billiards-trainer", "version": "1.0.0", "debug": False},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "api": {
                "host": "localhost",
                "port": 8000,
                "cors_origins": ["http://localhost:3000"],
            },
            "vision": {
                "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30},
                "detection": {"sensitivity": 0.8},
            },
            "projector": {
                "display": {"width": 1920, "height": 1080, "fullscreen": False}
            },
        }

        # Set defaults with proper metadata
        for key, value in self._flatten_dict(defaults).items():
            self._set_internal(
                key=key, value=value, source=ConfigSource.DEFAULT, persist=False
            )

    def _load_initial_config(self) -> None:
        """Load initial configuration from files and environment."""
        # Load configuration files
        for file_path in self._settings.sources.file_paths:
            full_path = self.config_dir / file_path
            if full_path.exists():
                self.load_config(full_path)

        # Load environment variables
        if self._settings.sources.enable_environment:
            self.load_environment_variables()

        # Track loaded files for hot reload
        for file_path in self._settings.sources.file_paths:
            full_path = self.config_dir / file_path
            if full_path.exists():
                self._watched_files.append(full_path)

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten nested dictionary into dot-notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _unflatten_dict(self, d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
        """Convert dot-notation keys back to nested dictionary."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    def _apply_migrations(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply all registered migrations to configuration.

        Args:
            config: Configuration dictionary to migrate

        Returns:
            Migrated configuration dictionary
        """
        for migration in self._migrations:
            if migration.should_migrate(config):
                print(
                    f"Applying migration: {migration.description} "
                    f"(version {migration.version})"
                )
                config = migration.migrate(config)

                # Validate and show warnings
                warnings = migration.validate(config)
                for warning in warnings:
                    if warning.startswith(("ERROR:", "WARNING:", "DEPRECATED:")):
                        print(f"  {warning}")

                # Show suggested configuration if there are warnings
                if warnings:
                    suggestion = migration.get_suggested_configuration(config)
                    if suggestion:
                        print(suggestion)

        return config

    def load_config(self, path: Path, format: Optional[ConfigFormat] = None) -> bool:
        """Load configuration from file.

        Args:
            path: Path to configuration file
            format: File format (auto-detected if None)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not path.exists():
                return False

            # Use the persistence system to load configuration
            data = self._persistence.load_config(path, format)

            # Apply migrations
            data = self._apply_migrations(data)

            # Flatten and store configuration
            flattened = self._flatten_dict(data)
            for key, value in flattened.items():
                self._set_internal(
                    key=key, value=value, source=ConfigSource.FILE, persist=False
                )

            return True

        except ConfigPersistenceError as e:
            print(f"Error loading config from {path}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading config from {path}: {e}")
            return False

    def save_config(
        self, path: Optional[Path] = None, format: Optional[ConfigFormat] = None
    ) -> bool:
        """Save current configuration to file.

        Args:
            path: Path to save configuration (defaults to config/current.json)
            format: File format (auto-detected if None)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if path is None:
                path = self.config_dir / "current.json"

            # Get all non-default configuration values
            config_data = {}
            for key, config_value in self._data.items():
                if config_value.source != ConfigSource.DEFAULT:
                    config_data[key] = config_value.value

            # Convert to nested structure
            nested_data = self._unflatten_dict(config_data)

            # Use the persistence system to save configuration
            return self._persistence.save_config(nested_data, path, format)

        except Exception as e:
            print(f"Error saving config to {path}: {e}")
            return False

    def reload_config(self) -> bool:
        """Reload configuration from all sources.

        Returns:
            True if reloaded successfully
        """
        try:
            # Clear current non-default configuration
            keys_to_remove = [
                key
                for key, config_value in self._data.items()
                if config_value.source != ConfigSource.DEFAULT
            ]
            for key in keys_to_remove:
                del self._data[key]

            # Reload from files and environment
            self._load_initial_config()

            return True

        except Exception as e:
            print(f"Error reloading configuration: {e}")
            return False

    def get(
        self, key: str, default: Any = None, type_hint: Optional[type] = None
    ) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            type_hint: Expected type for validation

        Returns:
            Configuration value or default
        """
        if key in self._data:
            value = self._data[key].value

            # Type checking if requested
            if type_hint is not None and not isinstance(value, type_hint):
                try:
                    value = type_hint(value)
                except (ValueError, TypeError):
                    return default

            return value

        return default

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.RUNTIME,
        persist: bool = False,
    ) -> bool:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
            source: Source of the configuration change
            persist: Whether to persist the change to file

        Returns:
            True if set successfully, False otherwise
        """
        return self._set_internal(key, value, source, persist)

    def _set_internal(
        self, key: str, value: Any, source: ConfigSource, persist: bool
    ) -> bool:
        """Internal method to set configuration value."""
        try:
            old_value = self._data.get(key)
            timestamp = time.time()

            # Create new configuration value
            config_value = ConfigValue(
                key=key,
                value=value,
                source=source,
                timestamp=timestamp,
                validated=False,  # Will be validated later
            )

            # Validate if schema exists
            if key in self._schemas:
                is_valid, errors = self._validate_value(key, value, self._schemas[key])
                if not is_valid:
                    return False
                config_value.validated = True

            # Store the value
            self._data[key] = config_value

            # Create change event
            change = ConfigChange(
                key=key,
                old_value=old_value.value if old_value else None,
                new_value=value,
                source=source,
                timestamp=timestamp,
                applied=True,
            )

            # Add to history
            self._history.append(change)

            # Notify subscribers
            self._notify_subscribers(change)

            # Persist if requested
            if persist:
                self.save_config()

            return True

        except Exception as e:
            print(f"Error setting {key}: {e}")
            return False

    def get_all(self, prefix: Optional[str] = None) -> dict[str, Any]:
        """Get all configuration values.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            Dictionary of configuration values
        """
        result = {}
        for key, config_value in self._data.items():
            if prefix is None or key.startswith(prefix):
                result[key] = config_value.value
        return result

    def update(
        self, values: dict[str, Any], source: ConfigSource = ConfigSource.RUNTIME
    ) -> list[ConfigChange]:
        """Update multiple configuration values.

        Args:
            values: Dictionary of key-value pairs to update
            source: Source of the configuration change

        Returns:
            List of configuration changes made
        """
        changes = []
        for key, value in values.items():
            old_value = self._data.get(key)
            if self._set_internal(key, value, source, False):
                change = ConfigChange(
                    key=key,
                    old_value=old_value.value if old_value else None,
                    new_value=value,
                    source=source,
                    timestamp=time.time(),
                    applied=True,
                )
                changes.append(change)
        return changes

    def load_environment_variables(self) -> None:
        """Load configuration from environment variables using EnvironmentLoader."""
        try:
            env_config = self._env_loader.load_environment()
            for key, value in env_config.items():
                self._set_internal(
                    key=key,
                    value=value,
                    source=ConfigSource.ENVIRONMENT,
                    persist=False,
                )
        except Exception as e:
            print(f"Warning: Failed to load environment variables: {e}")

    def load_cli_arguments(self, args: Optional[list[str]] = None) -> None:
        """Load configuration from CLI arguments using CLILoader."""
        try:
            # Update CLI loader with current schema if available
            if self._schemas:
                self._cli_loader = CLILoader(schema=self._schemas)

            cli_config = self._cli_loader.load(args)
            for key, value in cli_config.items():
                self._set_internal(
                    key=key,
                    value=value,
                    source=ConfigSource.CLI,
                    persist=False,
                )
        except SystemExit:
            # Re-raise SystemExit to allow help/version exits
            raise
        except Exception as e:
            print(f"Warning: Failed to load CLI arguments: {e}")

    def validate(
        self, key: Optional[str] = None, schema: Optional[dict] = None
    ) -> tuple[bool, list[str]]:
        """Validate configuration values.

        Args:
            key: Specific key to validate (validates all if None)
            schema: Schema to validate against

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if key is not None:
            # Validate specific key
            if key in self._data:
                value = self._data[key].value
                validation_schema = schema or self._schemas.get(key)
                if validation_schema:
                    is_valid, key_errors = self._validate_value(
                        key, value, validation_schema
                    )
                    errors.extend(key_errors)
            else:
                errors.append(f"Configuration key '{key}' not found")
        else:
            # Validate all keys with schemas
            for config_key, config_value in self._data.items():
                if config_key in self._schemas:
                    is_valid, key_errors = self._validate_value(
                        config_key, config_value.value, self._schemas[config_key]
                    )
                    errors.extend(key_errors)

        return len(errors) == 0, errors

    def _validate_value(
        self, key: str, value: Any, schema: dict
    ) -> tuple[bool, list[str]]:
        """Validate a single value against its schema."""
        errors = []

        # Basic type checking
        expected_type = schema.get("type")
        if expected_type:
            if expected_type == "integer" and not isinstance(value, int):
                errors.append(f"{key}: Expected integer, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"{key}: Expected number, got {type(value).__name__}")
            elif expected_type == "string" and not isinstance(value, str):
                errors.append(f"{key}: Expected string, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"{key}: Expected boolean, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"{key}: Expected array, got {type(value).__name__}")

        # Range checking for numbers
        if isinstance(value, (int, float)):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{key}: Value {value} below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{key}: Value {value} above maximum {schema['maximum']}")

        return len(errors) == 0, errors

    def register_schema(self, prefix: str, schema: dict) -> None:
        """Register validation schema for configuration prefix.

        Args:
            prefix: Configuration key prefix
            schema: JSON schema for validation
        """
        self._schemas[prefix] = schema

    def subscribe(self, pattern: str, callback: Callable[[ConfigChange], None]) -> str:
        """Subscribe to configuration changes.

        Args:
            pattern: Key pattern to match (supports wildcards)
            callback: Function to call on changes

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid.uuid4())
        self._subscriptions[subscription_id] = callback
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from configuration changes.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def _notify_subscribers(self, change: ConfigChange) -> None:
        """Notify all subscribers of configuration change."""
        for callback in self._subscriptions.values():
            try:
                callback(change)
            except Exception as e:
                print(f"Error in configuration change callback: {e}")

    def register_module(self, module_name: str, config_spec: dict) -> None:
        """Register module configuration requirements.

        Args:
            module_name: Name of the module
            config_spec: Module configuration specification
        """
        self._module_specs[module_name] = config_spec

        # Register schemas from the module spec
        if "configuration" in config_spec:
            for key, schema in config_spec["configuration"].items():
                full_key = f"{module_name}.{key}"
                self.register_schema(full_key, schema)

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get module-specific configuration.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary of module configuration
        """
        prefix = f"{module_name}."
        module_config = {}

        for key, config_value in self._data.items():
            if key.startswith(prefix):
                # Remove module prefix
                local_key = key[len(prefix) :]
                module_config[local_key] = config_value.value

        return self._unflatten_dict(module_config)

    def get_metadata(self, key: str) -> Optional[ConfigValue]:
        """Get configuration metadata.

        Args:
            key: Configuration key

        Returns:
            ConfigValue with metadata or None
        """
        return self._data.get(key)

    def get_history(
        self, key: Optional[str] = None, limit: int = 10
    ) -> list[ConfigChange]:
        """Get configuration change history.

        Args:
            key: Optional key to filter history
            limit: Maximum number of changes to return

        Returns:
            List of configuration changes
        """
        history = self._history

        if key is not None:
            history = [change for change in history if change.key == key]

        # Return most recent changes first
        return sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]

    # Additional utility methods for basic functionality
    def reset_to_defaults(self, prefix: Optional[str] = None) -> None:
        """Reset configuration to defaults.

        Args:
            prefix: Optional prefix to reset (resets all if None)
        """
        keys_to_reset = []
        for key, config_value in self._data.items():
            if prefix is None or key.startswith(prefix):
                if config_value.source != ConfigSource.DEFAULT:
                    keys_to_reset.append(key)

        for key in keys_to_reset:
            del self._data[key]

        # Reload defaults for the prefix
        if prefix is None:
            self._load_defaults()

    def create_profile(
        self,
        name: str,
        settings: Optional[dict] = None,
        description: Optional[str] = None,
        parent: Optional[str] = None,
        conditions: Optional[dict] = None,
    ) -> ConfigProfile:
        """Create new configuration profile.

        Args:
            name: Profile name
            settings: Optional settings to include (uses current settings if None)
            description: Profile description
            parent: Parent profile name for inheritance
            conditions: Auto-activation conditions

        Returns:
            Created configuration profile
        """
        if settings is None:
            settings = self.get_all()

        # Use ProfileManager for enhanced functionality
        success = self._profile_manager.create_profile(
            name=name,
            config_data=settings,
            description=description,
            parent=parent,
            conditions=conditions,
        )

        if success:
            # Return legacy ConfigProfile for backward compatibility
            return ConfigProfile(
                name=name,
                description=description or f"Configuration profile: {name}",
                settings=settings,
            )
        else:
            # Fallback to legacy behavior
            profile = ConfigProfile(
                name=name,
                description=f"Configuration profile: {name}",
                settings=settings,
            )
            return profile

    def save_profile(self, name: str, settings: Optional[dict] = None) -> bool:
        """Save a configuration profile using the persistence system.

        Args:
            name: Profile name
            settings: Settings to save (uses current settings if None)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if settings is None:
                settings = self.get_all()

            # Use ProfileManager if profile already exists, otherwise use persistence directly
            if name in self._profile_manager.list_profiles():
                return self._profile_manager.update_profile(name, settings=settings)
            else:
                return self._profile_manager.create_profile(name, config_data=settings)

        except Exception as e:
            print(f"Error saving profile '{name}': {e}")
            return False

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile.

        Args:
            name: Profile name to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            return self._profile_manager.delete_profile(name)
        except ProfileManagerError as e:
            print(f"Error deleting profile '{name}': {e}")
            return False
        except Exception as e:
            print(f"Unexpected error deleting profile '{name}': {e}")
            return False

    def cleanup_backups(self, max_backups: int = 10) -> int:
        """Clean up old backup files.

        Args:
            max_backups: Maximum number of backups to keep per file

        Returns:
            Number of backups removed
        """
        return self._persistence.cleanup_backups(max_backups)

    def load_profile(self, name: str) -> bool:
        """Load and apply configuration profile.

        Args:
            name: Profile name to load

        Returns:
            True if loaded successfully
        """
        try:
            # Use ProfileManager to get merged settings with inheritance
            merged_settings = self._profile_manager.merge_profile_settings(
                name, include_inherited=True
            )
            if not merged_settings:
                return False

            # Switch to the profile in ProfileManager
            if not self._profile_manager.switch_profile(name):
                return False

            # Apply profile settings to configuration
            flattened = self._flatten_dict(merged_settings)
            for key, value in flattened.items():
                self._set_internal(
                    key=key, value=value, source=ConfigSource.FILE, persist=False
                )

            return True

        except Exception as e:
            print(f"Error loading profile '{name}': {e}")
            return False

    def list_profiles(self) -> list[ConfigProfile]:
        """List available configuration profiles.

        Returns:
            List of available profiles
        """
        profiles = []
        try:
            profile_names = self._profile_manager.list_profiles()
            for profile_name in profile_names:
                try:
                    profile = self._profile_manager.load_profile(profile_name)
                    if profile:
                        # Convert to legacy ConfigProfile format for backward compatibility
                        profiles.append(
                            ConfigProfile(
                                name=profile.name,
                                description=profile.description or "",
                                settings=profile.settings,
                            )
                        )
                except Exception:
                    continue
        except Exception:
            pass
        return profiles

    def export_profile(self, name: str, path: Path) -> bool:
        """Export profile to file.

        Args:
            name: Profile name
            path: Export path

        Returns:
            True if exported successfully
        """
        try:
            profile_data = {
                "name": name,
                "description": f"Exported profile: {name}",
                "settings": self.get_all(),
            }
            with open(path, "w") as f:
                json.dump(profile_data, f, indent=2, default=str)
            return True
        except Exception:
            return False

    def diff(
        self, other: Optional[dict] = None, profile: Optional[str] = None
    ) -> dict[str, tuple[Any, Any]]:
        """Compare configurations.

        Args:
            other: Other configuration to compare against
            profile: Profile name to compare against

        Returns:
            Dictionary of differences
        """
        current = self.get_all()

        if profile:
            profile_path = self.config_dir / "profiles" / f"{profile}.json"
            if profile_path.exists():
                with open(profile_path) as f:
                    profile_data = json.load(f)
                other = profile_data.get("settings", {})

        if other is None:
            return {}

        differences = {}
        all_keys = set(current.keys()) | set(other.keys())

        for key in all_keys:
            current_val = current.get(key)
            other_val = other.get(key)
            if current_val != other_val:
                differences[key] = (current_val, other_val)

        return differences

    async def initialize(self) -> None:
        """Async initialization method for compatibility with API main.py.

        This method is called during application startup.
        """
        # The initialization is already done in __init__, so this is a no-op
        # but provides the async interface expected by the API main.py
        pass

    async def get_configuration(self) -> dict[str, Any]:
        """Get the complete configuration as a nested dictionary.

        Returns:
            Dictionary with the complete configuration in nested format
        """
        # Get all configuration values
        all_config = self.get_all()

        # Convert flat dict to nested structure
        return self._unflatten_dict(all_config)

    # =============================================================================
    # Backup and Restore Methods
    # =============================================================================

    def create_backup(
        self, backup_name: str, description: str = None, tags: list[str] = None
    ) -> str:
        """Create a named backup of current configuration.

        Args:
            backup_name: Name for the backup
            description: Optional description
            tags: Optional tags for organization

        Returns:
            Path to created backup file

        Raises:
            BackupError: If backup creation fails
        """
        try:
            # Get current configuration data
            config_data = self.get_all()

            # Create backup
            backup_path = self._backup.create_backup(
                config_data, backup_name, description, tags
            )

            print(f"Configuration backup created: {backup_name}")
            return backup_path

        except BackupError as e:
            print(f"Failed to create backup '{backup_name}': {e}")
            raise

    def create_auto_backup(self, reason: str = "configuration_change") -> str:
        """Create automatic backup before configuration changes.

        Args:
            reason: Reason for the automatic backup

        Returns:
            Path to created backup file
        """
        try:
            # Get current configuration data
            config_data = self.get_all()

            # Create automatic backup
            backup_path = self._backup.create_auto_backup(config_data, reason)

            print(f"Automatic backup created: {reason}")
            return backup_path

        except BackupError as e:
            print(f"Failed to create automatic backup: {e}")
            # Don't raise for auto backups - just log and continue
            return ""

    def list_config_backups(
        self,
        include_auto: bool = True,
        include_manual: bool = True,
        tags: list[str] = None,
    ) -> list[BackupMetadata]:
        """List available configuration backups.

        Args:
            include_auto: Include automatic backups
            include_manual: Include manual backups
            tags: Filter by tags

        Returns:
            List of backup metadata
        """
        return self._backup.list_backups(include_auto, include_manual, tags)

    def restore_backup(self, backup_name: str, validate_first: bool = True) -> bool:
        """Restore configuration from backup.

        Args:
            backup_name: Name of backup to restore
            validate_first: Whether to validate backup integrity first

        Returns:
            True if restore successful, False otherwise
        """
        try:
            # Create auto backup before restore
            self.create_auto_backup("before_restore")

            # Restore configuration data
            config_data = self._backup.restore_backup(backup_name, validate_first)

            # Clear current non-default configuration
            keys_to_remove = [
                key
                for key, config_value in self._data.items()
                if config_value.source != ConfigSource.DEFAULT
            ]
            for key in keys_to_remove:
                del self._data[key]

            # Apply restored configuration
            flattened = self._flatten_dict(config_data)
            changes = []
            for key, value in flattened.items():
                if self._set_internal(key, value, ConfigSource.FILE, False):
                    change = ConfigChange(
                        key=key,
                        old_value=None,  # Cleared above
                        new_value=value,
                        source=ConfigSource.FILE,
                        timestamp=time.time(),
                        applied=True,
                    )
                    changes.append(change)

            # Save restored configuration
            self.save_config()

            print(f"Configuration restored from backup: {backup_name}")
            return True

        except (BackupError, Exception) as e:
            print(f"Failed to restore backup '{backup_name}': {e}")
            return False

    def delete_backup(self, backup_name: str) -> bool:
        """Delete a configuration backup.

        Args:
            backup_name: Name of backup to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            success = self._backup.delete_backup(backup_name)
            if success:
                print(f"Backup deleted: {backup_name}")
            return success
        except Exception as e:
            print(f"Failed to delete backup '{backup_name}': {e}")
            return False

    def verify_backup(self, backup_name: str) -> bool:
        """Verify backup integrity.

        Args:
            backup_name: Name of backup to verify

        Returns:
            True if backup is valid, False otherwise
        """
        return self._backup.verify_backup_integrity(backup_name)

    def get_backup_info(self, backup_name: str) -> Optional[BackupMetadata]:
        """Get backup metadata.

        Args:
            backup_name: Name of backup

        Returns:
            Backup metadata or None if not found
        """
        return self._backup.get_backup_info(backup_name)

    def cleanup_old_config_backups(
        self, max_age_days: int = 30, keep_manual: bool = True
    ) -> int:
        """Clean up old configuration backups.

        Args:
            max_age_days: Maximum age for automatic backups
            keep_manual: Whether to keep manual backups regardless of age

        Returns:
            Number of backups cleaned up
        """
        try:
            cleaned_count = self._backup.cleanup_old_backups(max_age_days, keep_manual)
            if cleaned_count > 0:
                print(f"Cleaned up {cleaned_count} old backups")
            return cleaned_count
        except Exception as e:
            print(f"Failed to cleanup old backups: {e}")
            return 0

    def export_backup(self, backup_name: str, export_path: Path) -> bool:
        """Export backup to external location.

        Args:
            backup_name: Name of backup to export
            export_path: Path to export to

        Returns:
            True if export successful, False otherwise
        """
        try:
            success = self._backup.export_backup(backup_name, export_path)
            if success:
                print(f"Backup exported: {backup_name} -> {export_path}")
            return success
        except Exception as e:
            print(f"Failed to export backup '{backup_name}': {e}")
            return False

    def import_backup(self, import_path: Path, backup_name: str = None) -> str:
        """Import backup from external location.

        Args:
            import_path: Path to backup file to import
            backup_name: Optional name for imported backup

        Returns:
            Name of imported backup

        Raises:
            BackupError: If import fails
        """
        try:
            result = self._backup.import_backup(import_path, backup_name)
            print(f"Backup imported: {result}")
            return result
        except BackupError as e:
            print(f"Failed to import backup from {import_path}: {e}")
            raise

    # =============================================================================
    # Hot Reload Methods
    # =============================================================================

    def _init_hot_reload(self) -> None:
        """Initialize hot reload functionality.

        This method is called during initialization when hot reload is enabled.
        It sets up file watching capabilities for configuration files.
        """
        try:
            # Ensure we have the required attributes
            if not hasattr(self, "_watched_files"):
                print(
                    "Warning: _watched_files attribute not found, initializing empty list"
                )
                self._watched_files = []

            if not hasattr(self, "_settings"):
                print(
                    "Warning: _settings attribute not found, cannot initialize hot reload"
                )
                return

            if not self._watched_files:
                # No configuration files to watch
                print("Hot reload: No configuration files to watch")
                return

            # Get hot reload settings from configuration
            try:
                hot_reload_config = self._settings.hot_reload
                debounce_delay = (
                    hot_reload_config.reload_delay / 1000.0
                )  # Convert ms to seconds
            except AttributeError as e:
                print(f"Warning: Hot reload settings not found in configuration: {e}")
                debounce_delay = 1.0  # Default debounce delay

            # Initialize configuration watcher
            try:
                self._config_watcher = ConfigWatcher(debounce_delay=debounce_delay)
            except Exception as e:
                print(f"Error creating ConfigWatcher: {e}")
                self._config_watcher = None
                return

            # Register callbacks with error handling
            try:
                self._config_watcher.on_file_changed(self._handle_config_file_change)
                self._config_watcher.on_validation_needed(self._validate_config_file)
                self._config_watcher.on_rollback_needed(self._handle_config_rollback)
            except Exception as e:
                print(f"Error registering hot reload callbacks: {e}")
                self._config_watcher = None
                return

            # Start watching files
            try:
                if self._config_watcher.start_watching(self._watched_files):
                    print(
                        f"Hot reload enabled for {len(self._watched_files)} configuration files"
                    )
                else:
                    print("Failed to start hot reload functionality")
                    self._config_watcher = None
            except Exception as e:
                print(f"Error starting file watching: {e}")
                self._config_watcher = None

        except Exception as e:
            print(f"Error initializing hot reload: {e}")
            # Ensure watcher is cleared on any error
            self._config_watcher = None

    def _handle_config_file_change(self, event: ConfigChangeEvent) -> None:
        """Handle configuration file change events.

        Args:
            event: Configuration change event
        """
        try:
            print(f"Configuration file changed: {event.file_path}")

            # Create backup before reloading
            try:
                backup_info = self._backup.create_backup(
                    source_config=self.get_all(),
                    metadata={
                        "trigger": "hot_reload",
                        "file_changed": str(event.file_path),
                        "timestamp": event.timestamp,
                    },
                )
                print(f"Created backup: {backup_info.backup_id}")
            except Exception as e:
                print(f"Warning: Failed to create backup before reload: {e}")

            # Reload the specific configuration file
            if self.load_config(event.file_path):
                # Create change notification for subscribers
                for key, new_value in self._flatten_dict(
                    event.new_content or {}
                ).items():
                    old_value = self._flatten_dict(event.old_content or {}).get(key)

                    if old_value != new_value:
                        change = ConfigChange(
                            key=key,
                            old_value=old_value,
                            new_value=new_value,
                            source=ConfigSource.FILE,
                            timestamp=event.timestamp,
                            applied=True,
                        )

                        # Add to history
                        self._history.append(change)

                        # Notify subscribers
                        self._notify_subscribers(change)

                print(f"Successfully reloaded configuration from {event.file_path}")
            else:
                print(f"Failed to reload configuration from {event.file_path}")

        except Exception as e:
            print(f"Error handling configuration file change: {e}")

    def _validate_config_file(self, file_path: Path, content: dict[str, Any]) -> bool:
        """Validate configuration file content.

        Args:
            file_path: Path to configuration file
            content: Configuration content to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Basic validation - check if it's valid JSON/YAML structure
            if not isinstance(content, dict):
                print(
                    f"Configuration file {file_path} must contain a dictionary/object"
                )
                return False

            # Validate against schemas if available
            flattened_content = self._flatten_dict(content)

            for key, value in flattened_content.items():
                if key in self._schemas:
                    is_valid, errors = self._validate_value(
                        key, value, self._schemas[key]
                    )
                    if not is_valid:
                        print(f"Validation failed for {key} in {file_path}: {errors}")
                        return False

            # Additional custom validation can be added here
            return True

        except Exception as e:
            print(f"Error validating configuration file {file_path}: {e}")
            return False

    def _handle_config_rollback(self, file_path: Path, error: Exception) -> None:
        """Handle configuration rollback scenarios.

        Args:
            file_path: Path to problematic configuration file
            error: Exception that caused the rollback
        """
        try:
            print(f"Configuration rollback triggered for {file_path}: {error}")

            # Try to restore from most recent backup
            try:
                backups = self._backup.list_backups()
                if backups:
                    latest_backup = backups[0]  # Most recent backup
                    restored_config = self._backup.restore_backup(
                        latest_backup.backup_id
                    )

                    # Apply restored configuration
                    for key, value in self._flatten_dict(restored_config).items():
                        self._set_internal(key, value, ConfigSource.FILE, persist=False)

                    print(
                        f"Successfully restored configuration from backup: {latest_backup.backup_id}"
                    )
                else:
                    print("No backups available for rollback")

            except Exception as backup_error:
                print(f"Failed to restore from backup: {backup_error}")
                print(
                    "Continuing with previous configuration due to validation failure"
                )

        except Exception as e:
            print(f"Error during configuration rollback handling: {e}")

    def enable_hot_reload(self) -> bool:
        """Enable hot reload functionality.

        Returns:
            True if hot reload was enabled successfully
        """
        if self._config_watcher is not None:
            print("Hot reload is already enabled")
            return True

        self._enable_hot_reload = True
        self._init_hot_reload()
        return self._config_watcher is not None

    def disable_hot_reload(self) -> bool:
        """Disable hot reload functionality.

        Returns:
            True if hot reload was disabled successfully
        """
        if self._config_watcher is None:
            return True

        try:
            success = self._config_watcher.stop_watching()
            self._config_watcher = None
            self._enable_hot_reload = False
            print("Hot reload disabled")
            return success
        except Exception as e:
            print(f"Error disabling hot reload: {e}")
            return False

    def is_hot_reload_enabled(self) -> bool:
        """Check if hot reload is currently enabled.

        Returns:
            True if hot reload is enabled and active
        """
        return self._config_watcher is not None and self._config_watcher._is_watching

    def add_watched_file(self, file_path: Path) -> bool:
        """Add a configuration file to the watch list.

        Args:
            file_path: Path to configuration file to watch

        Returns:
            True if file was added successfully
        """
        try:
            # Add to our tracked files
            if file_path not in self._watched_files:
                self._watched_files.append(file_path)

            # Add to watcher if it's active
            if self._config_watcher is not None:
                return self._config_watcher.add_file(file_path)

            return True

        except Exception as e:
            print(f"Error adding file to watch list: {e}")
            return False

    def remove_watched_file(self, file_path: Path) -> bool:
        """Remove a configuration file from the watch list.

        Args:
            file_path: Path to configuration file to stop watching

        Returns:
            True if file was removed successfully
        """
        try:
            # Remove from our tracked files
            if file_path in self._watched_files:
                self._watched_files.remove(file_path)

            # Remove from watcher if it's active
            if self._config_watcher is not None:
                return self._config_watcher.remove_file(file_path)

            return True

        except Exception as e:
            print(f"Error removing file from watch list: {e}")
            return False

    def get_watched_files(self) -> list[Path]:
        """Get list of currently watched configuration files.

        Returns:
            List of watched file paths
        """
        return self._watched_files.copy()

    async def force_reload(self, file_path: Optional[Path] = None) -> bool:
        """Force reload configuration files.

        Args:
            file_path: Specific file to reload, or None for all files

        Returns:
            True if reload was successful
        """
        try:
            if self._config_watcher is not None:
                return await self._config_watcher.reload_configuration(file_path)
            else:
                # Manual reload without watcher
                if file_path:
                    return self.load_config(file_path)
                else:
                    return self.reload_config()

        except Exception as e:
            print(f"Error during force reload: {e}")
            return False

    # =============================================================================
    # Enhanced Profile Management Methods
    # =============================================================================

    def switch_profile(self, name: str) -> bool:
        """Switch to a different configuration profile.

        Args:
            name: Profile name to switch to

        Returns:
            True if switched successfully, False otherwise
        """
        try:
            return self.load_profile(name)
        except Exception as e:
            print(f"Error switching to profile '{name}': {e}")
            return False

    def get_active_profile(self) -> Optional[str]:
        """Get the name of the currently active profile.

        Returns:
            Active profile name or None if no profile is active
        """
        return self._profile_manager.get_active_profile()

    def get_profile_metadata(self, name: str) -> Optional[dict]:
        """Get profile metadata.

        Args:
            name: Profile name

        Returns:
            Profile metadata dictionary or None
        """
        return self._profile_manager.get_profile_metadata(name)

    def validate_profile(self, name: str) -> tuple[bool, list[str]]:
        """Validate a profile for consistency and correctness.

        Args:
            name: Profile name to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self._profile_manager.validate_profile(name)

    def update_profile(
        self,
        name: str,
        settings: Optional[dict] = None,
        description: Optional[str] = None,
        conditions: Optional[dict] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Update an existing profile.

        Args:
            name: Profile name to update
            settings: New settings (merged with existing)
            description: New description
            conditions: New conditions
            tags: New tags

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            return self._profile_manager.update_profile(
                name=name,
                settings=settings,
                description=description,
                conditions=conditions,
                tags=tags,
            )
        except Exception as e:
            print(f"Error updating profile '{name}': {e}")
            return False

    def auto_select_profile(self, context: Optional[dict] = None) -> bool:
        """Automatically select and switch to the best matching profile.

        Args:
            context: Context information for condition evaluation

        Returns:
            True if a profile was selected and switched, False otherwise
        """
        try:
            return self._profile_manager.auto_select_profile(context)
        except Exception as e:
            print(f"Error auto-selecting profile: {e}")
            return False

    def check_profile_conditions(self, context: Optional[dict] = None) -> Optional[str]:
        """Check profile conditions and return the best matching profile.

        Args:
            context: Context information for condition evaluation

        Returns:
            Name of the best matching profile or None
        """
        try:
            return self._profile_manager.check_profile_conditions(context)
        except Exception as e:
            print(f"Error checking profile conditions: {e}")
            return None

    def get_profiles_by_tag(self, tag: str) -> list[str]:
        """Get profiles that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of profile names with the tag
        """
        try:
            return self._profile_manager.get_profiles_by_tag(tag)
        except Exception as e:
            print(f"Error searching profiles by tag '{tag}': {e}")
            return []

    def merge_profile_settings(
        self, profile_name: str, include_inherited: bool = True
    ) -> dict:
        """Get merged profile settings including inheritance.

        Args:
            profile_name: Profile name
            include_inherited: Whether to include settings from parent profiles

        Returns:
            Merged settings dictionary
        """
        try:
            return self._profile_manager.merge_profile_settings(
                profile_name, include_inherited
            )
        except Exception as e:
            print(f"Error merging settings for profile '{profile_name}': {e}")
            return {}

    def export_profile_enhanced(
        self, name: str, export_path: Path, format: ConfigFormat = ConfigFormat.JSON
    ) -> bool:
        """Export profile to file with enhanced format support.

        Args:
            name: Profile name to export
            export_path: Path to export to
            format: Export format

        Returns:
            True if exported successfully, False otherwise
        """
        try:
            return self._profile_manager.export_profile(name, export_path, format)
        except Exception as e:
            print(f"Error exporting profile '{name}': {e}")
            return False

    def import_profile_enhanced(
        self, import_path: Path, profile_name: Optional[str] = None
    ) -> bool:
        """Import profile from file with enhanced format support.

        Args:
            import_path: Path to import from
            profile_name: Name for imported profile (uses file name if None)

        Returns:
            True if imported successfully, False otherwise
        """
        try:
            return self._profile_manager.import_profile(import_path, profile_name)
        except Exception as e:
            print(f"Error importing profile from '{import_path}': {e}")
            return False

    def get_profile_manager(self) -> ProfileManager:
        """Get the ProfileManager instance for advanced operations.

        Returns:
            ProfileManager instance
        """
        return self._profile_manager

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self._config_watcher is not None:
                self._config_watcher.stop_watching()
        except Exception:
            pass  # Ignore errors during cleanup


# Global configuration manager instance
config_manager = ConfigurationModule(enable_hot_reload=False)
