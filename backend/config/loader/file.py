"""File-based configuration loader.

Supports loading configuration from JSON, YAML, and INI files.
Implements FR-CFG-001, FR-CFG-002, FR-CFG-004, and FR-CFG-005.
"""

import configparser
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats."""

    JSON = "json"
    YAML = "yaml"
    YML = "yml"
    INI = "ini"
    TOML = "toml"


class ConfigurationError(Exception):
    """Base exception for configuration errors."""

    pass


class FileLoadError(ConfigurationError):
    """Exception raised when file loading fails."""

    pass


class FormatError(ConfigurationError):
    """Exception raised when file format is unsupported or invalid."""

    pass


class FileLoader:
    """Configuration file loader with support for multiple formats.

    Supports:
    - JSON files
    - YAML files
    - INI files
    - Default value provision
    - Configuration inheritance
    """

    def __init__(
        self, default_values: Optional[dict[str, Any]] = None, encoding: str = "utf-8"
    ):
        """Initialize the file loader.

        Args:
            default_values: Default configuration values
            encoding: File encoding to use
        """
        self.default_values = default_values or {}
        self.encoding = encoding
        self._loaded_files = {}  # Cache for loaded files

    def load_file(
        self, file_path: Union[str, Path], format: Optional[ConfigFormat] = None
    ) -> dict[str, Any]:
        """Load configuration from a single file.

        Args:
            file_path: Path to configuration file
            format: File format (auto-detected if None)

        Returns:
            Configuration dictionary

        Raises:
            FileLoadError: If file cannot be loaded
            FormatError: If file format is unsupported
        """
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return {}

        if not path.is_file():
            raise FileLoadError(f"Path is not a file: {path}")

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(path)

        try:
            with open(path, encoding=self.encoding) as f:
                content = f.read()

            config = self._parse_content(content, format)

            # Cache the loaded file
            self._loaded_files[str(path)] = {
                "config": config,
                "format": format,
                "mtime": path.stat().st_mtime,
            }

            logger.info(f"Loaded configuration from {path} ({format.value})")
            return config

        except OSError as e:
            raise FileLoadError(f"Failed to read file {path}: {e}")
        except Exception as e:
            raise FileLoadError(f"Failed to load configuration from {path}: {e}")

    def load_multiple(
        self, file_paths: list[Union[str, Path]], ignore_missing: bool = True
    ) -> list[dict[str, Any]]:
        """Load configuration from multiple files.

        Args:
            file_paths: List of configuration file paths
            ignore_missing: Whether to ignore missing files

        Returns:
            List of configuration dictionaries
        """
        configs = []

        for file_path in file_paths:
            try:
                config = self.load_file(file_path)
                if config:  # Only add non-empty configs
                    configs.append(config)
            except FileLoadError as e:
                if ignore_missing and "not found" in str(e):
                    logger.debug(f"Ignoring missing file: {file_path}")
                    continue
                else:
                    raise

        return configs

    def load_with_inheritance(
        self, file_path: Union[str, Path], parent_key: str = "inherit"
    ) -> dict[str, Any]:
        """Load configuration with inheritance support.

        Supports FR-CFG-005: Configuration inheritance and overrides.

        Args:
            file_path: Path to configuration file
            parent_key: Key that specifies parent configuration file

        Returns:
            Merged configuration with inheritance applied
        """
        config = self.load_file(file_path)

        if parent_key in config:
            parent_path = config.pop(parent_key)

            # Resolve parent path relative to current file
            current_dir = Path(file_path).parent
            if not Path(parent_path).is_absolute():
                parent_path = current_dir / parent_path

            # Load parent configuration recursively
            parent_config = self.load_with_inheritance(parent_path, parent_key)

            # Merge parent config with current config (current takes precedence)
            merged_config = self._deep_merge(parent_config, config)
            return merged_config

        return config

    def load_with_defaults(self, file_path: Union[str, Path]) -> dict[str, Any]:
        """Load configuration file with default values applied.

        Implements FR-CFG-004: Provide default values for all settings.

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration with defaults applied
        """
        config = self.load_file(file_path)
        return self._deep_merge(self.default_values, config)

    def is_file_modified(self, file_path: Union[str, Path]) -> bool:
        """Check if a previously loaded file has been modified.

        Args:
            file_path: Path to check

        Returns:
            True if file has been modified since last load
        """
        path_str = str(file_path)
        if path_str not in self._loaded_files:
            return True

        try:
            current_mtime = Path(file_path).stat().st_mtime
            cached_mtime = self._loaded_files[path_str]["mtime"]
            return current_mtime != cached_mtime
        except (OSError, KeyError):
            return True

    def clear_cache(self):
        """Clear the file cache."""
        self._loaded_files.clear()

    def _detect_format(self, path: Path) -> ConfigFormat:
        """Auto-detect configuration file format from extension.

        Args:
            path: File path

        Returns:
            Detected format

        Raises:
            FormatError: If format cannot be detected
        """
        suffix = path.suffix.lower()

        format_map = {
            ".json": ConfigFormat.JSON,
            ".yaml": ConfigFormat.YAML,
            ".yml": ConfigFormat.YML,
            ".ini": ConfigFormat.INI,
            ".toml": ConfigFormat.TOML,
        }

        if suffix in format_map:
            return format_map[suffix]

        raise FormatError(f"Unsupported file format: {suffix}")

    def _parse_content(self, content: str, format: ConfigFormat) -> dict[str, Any]:
        """Parse configuration content based on format.

        Args:
            content: File content
            format: Configuration format

        Returns:
            Parsed configuration dictionary

        Raises:
            FormatError: If parsing fails
        """
        try:
            if format in (ConfigFormat.JSON,):
                return json.loads(content)

            elif format in (ConfigFormat.YAML, ConfigFormat.YML):
                return yaml.safe_load(content) or {}

            elif format == ConfigFormat.INI:
                parser = configparser.ConfigParser()
                parser.read_string(content)

                # Convert ConfigParser to dict
                config = {}
                for section_name in parser.sections():
                    config[section_name] = dict(parser[section_name])

                # Handle the DEFAULT section if it exists
                if parser.defaults():
                    config["DEFAULT"] = dict(parser.defaults())

                return config

            elif format == ConfigFormat.TOML:
                try:
                    import tomllib

                    return tomllib.loads(content)
                except ImportError:
                    try:
                        import toml

                        return toml.loads(content)
                    except ImportError:
                        raise FormatError(
                            "TOML support requires 'tomllib' (Python 3.11+) or 'toml' package"
                        )

            else:
                raise FormatError(f"Unsupported format: {format}")

        except Exception as e:
            raise FormatError(f"Failed to parse {format.value} content: {e}")

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result
