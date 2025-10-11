"""Environment variable configuration loader.

Supports loading configuration from environment variables with type conversion,
prefix filtering, and nested structure support.
Implements FR-CFG-001 and FR-CFG-038.
"""

import json
import logging
import os
import re
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class EnvironmentError(Exception):
    """Exception raised for environment variable errors."""

    pass


class TypeConversionError(EnvironmentError):
    """Exception raised when type conversion fails."""

    pass


class EnvironmentLoader:
    """Environment variable configuration loader.

    Supports:
    - Automatic type conversion
    - Nested dictionary structure from dot notation
    - Custom type converters
    - Default values
    """

    def __init__(
        self,
        separator: str = "_",
        nested_separator: str = "__",
        type_converters: Optional[dict[str, Callable]] = None,
    ):
        """Initialize the environment loader.

        Args:
            separator: Separator for splitting variable names
            nested_separator: Separator for nested keys
            type_converters: Custom type conversion functions
        """
        self.separator = separator
        self.nested_separator = nested_separator
        self.type_converters = type_converters or {}

        # Default type converters
        self._default_converters = {
            "str": str,
            "int": int,
            "float": float,
            "bool": self._convert_bool,
            "json": self._convert_json,
            "list": self._convert_list,
        }

    def load_environment(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Load configuration from environment variables.

        Args:
            include_patterns: Regex patterns to include variables
            exclude_patterns: Regex patterns to exclude variables

        Returns:
            Configuration dictionary with nested structure
        """
        env_vars = self._get_filtered_env_vars(include_patterns, exclude_patterns)

        if not env_vars:
            logger.debug("No environment variables found")
            return {}

        config = {}

        for env_key, env_value in env_vars.items():
            try:
                # Remove prefix and convert to config key
                config_key = self._env_key_to_config_key(env_key)

                # Convert value with type inference
                converted_value = self._convert_value(env_value, env_key)

                # Set nested value in config
                self._set_nested_value(config, config_key, converted_value)

                logger.debug(
                    f"Loaded env var: {env_key} -> {config_key} = {converted_value}"
                )

            except Exception as e:
                logger.warning(f"Failed to process environment variable {env_key}: {e}")
                continue

        logger.info(f"Loaded {len(env_vars)} environment variables")
        return config

    def load_with_schema(
        self, schema: dict[str, Any], strict: bool = False
    ) -> dict[str, Any]:
        """Load environment variables according to a schema.

        Args:
            schema: Configuration schema with types and defaults
            strict: Whether to raise errors for missing required values

        Returns:
            Configuration dictionary
        """
        config = {}

        for key, spec in schema.items():
            env_key = self._config_key_to_env_key(key)

            # Get value from environment
            env_value = os.environ.get(env_key)

            if env_value is not None:
                # Convert using schema type
                try:
                    value_type = spec.get("type", "str")
                    converted_value = self._convert_value_with_type(
                        env_value, value_type
                    )
                    self._set_nested_value(config, key, converted_value)
                except Exception as e:
                    if strict:
                        raise TypeConversionError(f"Failed to convert {env_key}: {e}")
                    logger.warning(f"Failed to convert {env_key}, using raw value: {e}")
                    self._set_nested_value(config, key, env_value)

            elif "default" in spec:
                # Use default value
                self._set_nested_value(config, key, spec["default"])

            elif spec.get("required", False) and strict:
                raise EnvironmentError(
                    f"Required environment variable not found: {env_key}"
                )

        return config

    def get_env_var(
        self,
        config_key: str,
        default: Any = None,
        value_type: Optional[Union[str, type]] = None,
    ) -> Any:
        """Get a single environment variable with type conversion.

        Args:
            config_key: Configuration key (will be converted to env var name)
            default: Default value if not found
            value_type: Type to convert to

        Returns:
            Converted value or default
        """
        env_key = self._config_key_to_env_key(config_key)
        env_value = os.environ.get(env_key)

        if env_value is None:
            return default

        if value_type is None:
            return self._convert_value(env_value, env_key)
        else:
            return self._convert_value_with_type(env_value, value_type)

    def set_env_var(self, config_key: str, value: Any) -> None:
        """Set an environment variable from a config key.

        Args:
            config_key: Configuration key
            value: Value to set
        """
        env_key = self._config_key_to_env_key(config_key)
        os.environ[env_key] = str(value)

    def list_relevant_env_vars(self) -> list[str]:
        """List all environment variables that match the loader's criteria.

        Returns:
            List of environment variable names
        """
        return list(self._get_filtered_env_vars().keys())

    def _get_filtered_env_vars(
        self,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """Get filtered environment variables."""
        env_vars = {}

        for key, value in os.environ.items():
            # Apply include patterns
            if include_patterns:
                if not any(re.match(pattern, key) for pattern in include_patterns):
                    continue

            # Apply exclude patterns
            if exclude_patterns:
                if any(re.match(pattern, key) for pattern in exclude_patterns):
                    continue

            env_vars[key] = value

        return env_vars

    def _env_key_to_config_key(self, env_key: str) -> str:
        """Convert environment variable name to config key."""
        # Convert to lowercase and replace separators with dots
        key = env_key.lower()
        key = key.replace(self.nested_separator, ".")
        key = key.replace(self.separator, ".")

        return key

    def _config_key_to_env_key(self, config_key: str) -> str:
        """Convert config key to environment variable name."""
        # Replace dots with nested separator
        key = config_key.replace(".", self.nested_separator)

        # Convert to uppercase
        key = key.upper()

        return key

    def _convert_value(self, value: str, env_key: str) -> Any:
        """Convert string value with automatic type inference."""
        # Try to infer type from the value
        value = value.strip()

        # Boolean values
        if value.lower() in ("true", "false", "yes", "no", "on", "off"):
            return self._convert_bool(value)

        # Numeric values
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)

        try:
            return float(value)
        except ValueError:
            pass

        # JSON values (lists, dicts)
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Comma-separated lists
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # Default to string
        return value

    def _convert_value_with_type(self, value: str, value_type: Union[str, type]) -> Any:
        """Convert value with explicit type."""
        if isinstance(value_type, str):
            converter = self.type_converters.get(
                value_type
            ) or self._default_converters.get(value_type)
            if converter:
                return converter(value)
            else:
                raise TypeConversionError(f"Unknown type: {value_type}")
        else:
            # Direct type conversion
            return value_type(value)

    def _convert_bool(self, value: str) -> bool:
        """Convert string to boolean."""
        value = value.lower().strip()
        if value in ("true", "yes", "on", "1"):
            return True
        elif value in ("false", "no", "off", "0"):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")

    def _convert_json(self, value: str) -> Any:
        """Convert JSON string to Python object."""
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def _convert_list(self, value: str) -> list[str]:
        """Convert comma-separated string to list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",")]

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        keys = key.split(".")
        current = config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                # Convert non-dict values to dict if needed
                current[k] = {}
            current = current[k]

        # Set the final value
        current[keys[-1]] = value
