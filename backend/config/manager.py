"""Main configuration management module."""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .models.schemas import (
    ConfigChange,
    ConfigFormat,
    ConfigProfile,
    ConfigSource,
    ConfigurationSettings,
    ConfigValue,
)


class ConfigurationModule:
    """Main configuration interface providing centralized settings management."""

    def __init__(self, config_dir: Path = Path("config")):
        """Initialize configuration module.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._settings = ConfigurationSettings()
        self._data: dict[str, ConfigValue] = {}
        self._schemas: dict[str, dict] = {}
        self._subscriptions: dict[str, Callable[[ConfigChange], None]] = {}
        self._history: list[ConfigChange] = []
        self._module_specs: dict[str, dict] = {}

        # Initialize directory structure
        self._init_directories()

        # Load default configuration
        self._load_defaults()

        # Load configuration files if they exist
        self._load_initial_config()

    def _init_directories(self) -> None:
        """Initialize required directories."""
        directories = [
            self.config_dir,
            self.config_dir / "profiles",
            self._settings.paths.data_dir,
            self._settings.paths.log_dir,
            self._settings.paths.cache_dir,
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

    def load_config(self, path: Path, format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Load configuration from file.

        Args:
            path: Path to configuration file
            format: File format (JSON, YAML, INI)

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not path.exists():
                return False

            if format == ConfigFormat.JSON:
                with open(path) as f:
                    data = json.load(f)
            else:
                # For now, only implement JSON support
                raise NotImplementedError(f"Format {format} not yet implemented")

            # Flatten and store configuration
            flattened = self._flatten_dict(data)
            for key, value in flattened.items():
                self._set_internal(
                    key=key, value=value, source=ConfigSource.FILE, persist=False
                )

            return True

        except Exception as e:
            # Log error (would use proper logging in real implementation)
            print(f"Error loading config from {path}: {e}")
            return False

    def save_config(
        self, path: Optional[Path] = None, format: ConfigFormat = ConfigFormat.JSON
    ) -> bool:
        """Save current configuration to file.

        Args:
            path: Path to save configuration (defaults to config/current.json)
            format: File format to save as

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

            if format == ConfigFormat.JSON:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(nested_data, f, indent=2, default=str)
            else:
                raise NotImplementedError(f"Format {format} not yet implemented")

            return True

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
        """Load configuration from environment variables."""
        prefix = self._settings.sources.env_prefix

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and convert to lowercase with dots
                config_key = env_key[len(prefix) :].lower().replace("_", ".")

                # Try to parse as JSON first, then as string
                try:
                    value = json.loads(env_value)
                except json.JSONDecodeError:
                    value = env_value

                self._set_internal(
                    key=config_key,
                    value=value,
                    source=ConfigSource.ENVIRONMENT,
                    persist=False,
                )

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
        self, name: str, settings: Optional[dict] = None
    ) -> ConfigProfile:
        """Create new configuration profile.

        Args:
            name: Profile name
            settings: Optional settings to include

        Returns:
            Created configuration profile
        """
        if settings is None:
            settings = self.get_all()

        profile = ConfigProfile(
            name=name, description=f"Configuration profile: {name}", settings=settings
        )
        return profile

    def load_profile(self, name: str) -> bool:
        """Load and apply configuration profile.

        Args:
            name: Profile name to load

        Returns:
            True if loaded successfully
        """
        profile_path = self.config_dir / "profiles" / f"{name}.json"
        if profile_path.exists():
            return self.load_config(profile_path)
        return False

    def list_profiles(self) -> list[ConfigProfile]:
        """List available configuration profiles.

        Returns:
            List of available profiles
        """
        profiles = []
        profiles_dir = self.config_dir / "profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    with open(profile_file) as f:
                        data = json.load(f)
                    profiles.append(
                        ConfigProfile(
                            name=profile_file.stem,
                            description=data.get("description", ""),
                            settings=data.get("settings", {}),
                        )
                    )
                except Exception:
                    continue
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
