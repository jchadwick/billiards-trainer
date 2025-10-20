"""Simple configuration system for billiards-trainer.

A minimal, straightforward configuration loader that:
- Loads from a single JSON file (default: ./config.json)
- Provides dot-notation access (e.g., config.get("vision.camera.device_id", 0))
- Automatically saves changes to file asynchronously
- Is a singleton instance
- Has no dependencies beyond stdlib
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Config:
    """Simple configuration class with dot-notation access support.

    This is a singleton that loads configuration from a specified file
    (default: ./config.json relative to project root) and provides
    get/set methods with default value fallbacks.

    Example:
        config = Config()
        device_id = config.get("vision.camera.device_id", 0)
        config.set("vision.camera.device_id", 1)
    """

    _instance: Optional["Config"] = None
    _config_data: dict[str, Any] = {}
    _config_file: Optional[Path] = None

    def __new__(cls) -> "Config":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Don't load config yet - wait for set_config_file or use default
        return cls._instance

    @classmethod
    def set_config_file(cls, config_path: str | Path) -> None:
        """Set the configuration file path and load it.

        Args:
            config_path: Path to the configuration file
        """
        if cls._instance is None:
            cls._instance = cls()

        cls._config_file = Path(config_path).resolve()
        cls._instance._load_config()

    def _load_config(self) -> None:
        """Load configuration from config.json file."""
        # Use default if no file specified
        if self._config_file is None:
            # Default to ./config.json relative to project root (cwd)
            self._config_file = Path(os.getcwd()) / "config.json"

        try:
            if self._config_file.exists():
                with open(self._config_file, encoding="utf-8") as f:
                    self._config_data = json.load(f)
                logger.info(f"Loaded configuration from {self._config_file}")
            else:
                logger.warning(
                    f"Config file not found: {self._config_file}, using empty config"
                )
                self._config_data = {}
        except Exception as e:
            logger.error(f"Error loading config from {self._config_file}: {e}")
            self._config_data = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "vision.camera.device_id")
            default: Default value if key not found

        Returns:
            Configuration value or default if not found

        Example:
            >>> config = Config()
            >>> device_id = config.get("vision.camera.device_id", 0)
            >>> host = config.get("api.server.host", "0.0.0.0")
        """
        # Lazy load config on first access if not already loaded
        if not self._config_data and self._config_file is None:
            self._load_config()

        keys = key.split(".")
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Updates the in-memory configuration immediately and triggers an
        asynchronous save to the config file.

        Args:
            key: Dot-separated key path (e.g., "vision.camera.device_id")
            value: Value to set

        Example:
            >>> config = Config()
            >>> config.set("vision.camera.device_id", 1)
        """
        keys = key.split(".")
        data = self._config_data

        # Navigate to the parent of the target key, creating dicts as needed
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]

        # Set the final key (in-memory update first)
        data[keys[-1]] = value

        # Always save to file asynchronously
        self._save_config_async()

    def _save_config_async(self) -> None:
        """Save configuration to file asynchronously in a background thread."""
        thread = threading.Thread(target=self._save_config, daemon=True)
        thread.start()

    def _save_config(self) -> None:
        """Save configuration to config.json file (runs in background thread)."""
        if self._config_file is None:
            self._config_file = Path(os.getcwd()) / "config.json"

        try:
            # Create a copy of the data to avoid race conditions
            data_to_save = self._config_data.copy()

            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved configuration to {self._config_file}")
        except Exception as e:
            logger.error(f"Error saving config to {self._config_file}: {e}")

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def get_all(self) -> dict[str, Any]:
        """Get entire configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config_data.copy()


# Singleton instance for easy import
config = Config()

# Backwards compatibility aliases
config_manager = config
ConfigurationModule = Config  # For old imports
