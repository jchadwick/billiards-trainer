"""File watching for configuration hot reload."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """Represents a configuration file change event."""

    def __init__(
        self,
        file_path: Path,
        event_type: str,
        timestamp: float,
        old_content: Optional[dict[str, Any]] = None,
        new_content: Optional[dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.event_type = event_type
        self.timestamp = timestamp
        self.old_content = old_content
        self.new_content = new_content


class ConfigFileHandler(FileSystemEventHandler):
    """Handles file system events for configuration files."""

    def __init__(self, watcher: "ConfigWatcher"):
        self.watcher = watcher
        self.last_modified = {}

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_path = Path(event.src_path)

            # Debounce rapid file changes (some editors save multiple times)
            now = time.time()
            if file_path in self.last_modified:
                if now - self.last_modified[file_path] < 0.1:  # 100ms debounce
                    return
            self.last_modified[file_path] = now

            # Check if this file is being watched
            if file_path in self.watcher._watched_files:
                asyncio.create_task(
                    self.watcher._handle_file_change(file_path, "modified")
                )


class ConfigWatcher:
    """Advanced configuration file watcher with hot reload capabilities.

    Features:
    - Watch multiple configuration files simultaneously
    - Debounce rapid file changes
    - Validate configuration before applying
    - Rollback on validation errors
    - Notify modules of configuration changes
    - Support for different file formats (JSON, YAML, etc.)
    """

    def __init__(self, debounce_delay: float = 0.5):
        """Initialize the configuration watcher.

        Args:
            debounce_delay: Delay in seconds to debounce rapid changes
        """
        self._observer: Optional[Observer] = None
        self._watched_files: set[Path] = set()
        self._watched_directories: set[Path] = set()
        self._change_callbacks: list[Callable[[ConfigChangeEvent], None]] = []
        self._validation_callbacks: list[Callable[[Path, dict[str, Any]], bool]] = []
        self._rollback_callbacks: list[Callable[[Path, Exception], None]] = []
        self._debounce_delay = debounce_delay
        self._pending_changes: dict[Path, asyncio.Task] = {}
        self._is_watching = False
        self._file_contents: dict[Path, dict[str, Any]] = {}

        # File format handlers
        self._format_handlers = {
            ".json": self._load_json,
            ".yaml": self._load_yaml,
            ".yml": self._load_yaml,
        }

    def start_watching(self, config_files: list[Path]) -> bool:
        """Start monitoring configuration files.

        Args:
            config_files: List of configuration file paths to monitor

        Returns:
            True if watching started successfully, False otherwise
        """
        try:
            if self._is_watching:
                logger.warning("Watcher is already running")
                return True

            # Initialize observer
            self._observer = Observer()
            handler = ConfigFileHandler(self)

            # Track directories and files
            directories_to_watch = set()

            for config_file in config_files:
                config_path = Path(config_file).resolve()

                if not config_path.exists():
                    logger.warning(f"Configuration file does not exist: {config_path}")
                    continue

                self._watched_files.add(config_path)

                # Add directory to watch list
                directory = config_path.parent
                directories_to_watch.add(directory)

                # Load initial content
                try:
                    content = self._load_file(config_path)
                    self._file_contents[config_path] = content
                    logger.info(f"Loaded initial content for {config_path}")
                except Exception as e:
                    logger.error(
                        f"Failed to load initial content for {config_path}: {e}"
                    )
                    self._file_contents[config_path] = {}

            # Watch directories
            for directory in directories_to_watch:
                self._observer.schedule(handler, str(directory), recursive=False)
                self._watched_directories.add(directory)
                logger.info(f"Watching directory: {directory}")

            # Start observer
            self._observer.start()
            self._is_watching = True

            logger.info(
                f"Started watching {len(self._watched_files)} configuration files"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            return False

    def stop_watching(self) -> bool:
        """Stop file monitoring.

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self._is_watching:
                return True

            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            # Cancel pending change tasks
            for task in self._pending_changes.values():
                if not task.done():
                    task.cancel()
            self._pending_changes.clear()

            self._is_watching = False
            self._watched_files.clear()
            self._watched_directories.clear()
            self._file_contents.clear()

            logger.info("Stopped configuration file watching")
            return True

        except Exception as e:
            logger.error(f"Failed to stop file watching: {e}")
            return False

    def on_file_changed(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Register a callback for configuration file changes.

        Args:
            callback: Function to call when files change
        """
        if callback not in self._change_callbacks:
            self._change_callbacks.append(callback)
            logger.debug(f"Registered file change callback: {callback.__name__}")

    def on_validation_needed(
        self, callback: Callable[[Path, dict[str, Any]], bool]
    ) -> None:
        """Register a callback for configuration validation.

        Args:
            callback: Function to validate configuration. Should return True if valid.
        """
        if callback not in self._validation_callbacks:
            self._validation_callbacks.append(callback)
            logger.debug(f"Registered validation callback: {callback.__name__}")

    def on_rollback_needed(self, callback: Callable[[Path, Exception], None]) -> None:
        """Register a callback for configuration rollback scenarios.

        Args:
            callback: Function to call when rollback is needed
        """
        if callback not in self._rollback_callbacks:
            self._rollback_callbacks.append(callback)
            logger.debug(f"Registered rollback callback: {callback.__name__}")

    async def reload_configuration(self, config_path: Optional[Path] = None) -> bool:
        """Manually reload configuration from files.

        Args:
            config_path: Specific file to reload, or None for all watched files

        Returns:
            True if reload was successful, False otherwise
        """
        try:
            files_to_reload = (
                [config_path] if config_path else list(self._watched_files)
            )

            for file_path in files_to_reload:
                if file_path not in self._watched_files:
                    logger.warning(f"File not being watched: {file_path}")
                    continue

                await self._process_file_change(file_path, "manual_reload")

            logger.info(f"Manual reload completed for {len(files_to_reload)} files")
            return True

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False

    def get_watched_files(self) -> list[Path]:
        """Get list of currently watched files.

        Returns:
            List of watched file paths
        """
        return list(self._watched_files)

    def is_watching_file(self, file_path: Path) -> bool:
        """Check if a specific file is being watched.

        Args:
            file_path: Path to check

        Returns:
            True if file is being watched
        """
        return Path(file_path).resolve() in self._watched_files

    def add_file(self, file_path: Path) -> bool:
        """Add a new file to watch list.

        Args:
            file_path: Path of file to start watching

        Returns:
            True if file was added successfully
        """
        try:
            config_path = Path(file_path).resolve()

            if config_path in self._watched_files:
                logger.debug(f"File already being watched: {config_path}")
                return True

            if not config_path.exists():
                logger.warning(f"File does not exist: {config_path}")
                return False

            # Load initial content
            content = self._load_file(config_path)
            self._file_contents[config_path] = content
            self._watched_files.add(config_path)

            # If we're actively watching, we may need to add the directory
            if self._is_watching and self._observer:
                directory = config_path.parent
                if directory not in self._watched_directories:
                    handler = ConfigFileHandler(self)
                    self._observer.schedule(handler, str(directory), recursive=False)
                    self._watched_directories.add(directory)

            logger.info(f"Added file to watch list: {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to add file to watch list: {e}")
            return False

    def remove_file(self, file_path: Path) -> bool:
        """Remove a file from watch list.

        Args:
            file_path: Path of file to stop watching

        Returns:
            True if file was removed successfully
        """
        try:
            config_path = Path(file_path).resolve()

            if config_path not in self._watched_files:
                logger.debug(f"File not being watched: {config_path}")
                return True

            self._watched_files.discard(config_path)
            self._file_contents.pop(config_path, None)

            # Cancel any pending changes for this file
            if config_path in self._pending_changes:
                self._pending_changes[config_path].cancel()
                del self._pending_changes[config_path]

            logger.info(f"Removed file from watch list: {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove file from watch list: {e}")
            return False

    async def _handle_file_change(self, file_path: Path, event_type: str) -> None:
        """Handle a file change event with debouncing.

        Args:
            file_path: Path of changed file
            event_type: Type of change event
        """
        # Cancel any existing pending task for this file
        if file_path in self._pending_changes:
            self._pending_changes[file_path].cancel()

        # Create a new debounced task
        task = asyncio.create_task(self._debounced_file_change(file_path, event_type))
        self._pending_changes[file_path] = task

    async def _debounced_file_change(self, file_path: Path, event_type: str) -> None:
        """Process a file change after debounce delay.

        Args:
            file_path: Path of changed file
            event_type: Type of change event
        """
        try:
            # Wait for debounce delay
            await asyncio.sleep(self._debounce_delay)

            # Process the change
            await self._process_file_change(file_path, event_type)

        except asyncio.CancelledError:
            # Task was cancelled due to another change
            pass
        except Exception as e:
            logger.error(f"Error in debounced file change processing: {e}")
        finally:
            # Clean up the task reference
            self._pending_changes.pop(file_path, None)

    async def _process_file_change(self, file_path: Path, event_type: str) -> None:
        """Process a configuration file change.

        Args:
            file_path: Path of changed file
            event_type: Type of change event
        """
        try:
            logger.info(f"Processing {event_type} event for {file_path}")

            # Get old content
            old_content = self._file_contents.get(file_path, {}).copy()

            # Load new content
            new_content = self._load_file(file_path)

            # Validate new configuration
            is_valid = await self._validate_configuration(file_path, new_content)

            if not is_valid:
                logger.error(f"Configuration validation failed for {file_path}")
                await self._handle_rollback(
                    file_path, ValueError("Configuration validation failed")
                )
                return

            # Update stored content
            self._file_contents[file_path] = new_content

            # Create change event
            change_event = ConfigChangeEvent(
                file_path=file_path,
                event_type=event_type,
                timestamp=time.time(),
                old_content=old_content,
                new_content=new_content,
            )

            # Notify callbacks
            await self._notify_change_callbacks(change_event)

            logger.info(f"Successfully processed configuration change for {file_path}")

        except Exception as e:
            logger.error(f"Failed to process file change for {file_path}: {e}")
            await self._handle_rollback(file_path, e)

    async def _validate_configuration(
        self, file_path: Path, content: dict[str, Any]
    ) -> bool:
        """Validate configuration content using registered validators.

        Args:
            file_path: Path of configuration file
            content: Configuration content to validate

        Returns:
            True if configuration is valid
        """
        try:
            # Run all validation callbacks
            for validator in self._validation_callbacks:
                try:
                    if not validator(file_path, content):
                        logger.warning(
                            f"Validation failed for {file_path} using {validator.__name__}"
                        )
                        return False
                except Exception as e:
                    logger.error(
                        f"Validation callback {validator.__name__} raised exception: {e}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            return False

    async def _handle_rollback(self, file_path: Path, error: Exception) -> None:
        """Handle rollback when configuration change fails.

        Args:
            file_path: Path of problematic configuration file
            error: Exception that caused the rollback
        """
        try:
            logger.warning(f"Initiating rollback for {file_path} due to: {error}")

            # Notify rollback callbacks
            for callback in self._rollback_callbacks:
                try:
                    callback(file_path, error)
                except Exception as e:
                    logger.error(f"Rollback callback {callback.__name__} failed: {e}")

        except Exception as e:
            logger.error(f"Error during rollback handling: {e}")

    async def _notify_change_callbacks(self, change_event: ConfigChangeEvent) -> None:
        """Notify all registered change callbacks.

        Args:
            change_event: The configuration change event
        """
        for callback in self._change_callbacks:
            try:
                # Handle both sync and async callbacks
                result = callback(change_event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Change callback {callback.__name__} failed: {e}")

    def _load_file(self, file_path: Path) -> dict[str, Any]:
        """Load configuration file content.

        Args:
            file_path: Path to configuration file

        Returns:
            Dictionary containing configuration data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix not in self._format_handlers:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

        return self._format_handlers[suffix](file_path)

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        """Load JSON configuration file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary containing configuration data
        """
        import json

        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def _load_yaml(self, file_path: Path) -> dict[str, Any]:
        """Load YAML configuration file.

        Args:
            file_path: Path to YAML file

        Returns:
            Dictionary containing configuration data
        """
        try:
            import yaml

            with open(file_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ValueError("PyYAML not installed - cannot load YAML files")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")


# Legacy compatibility
class FileWatcher:
    """Legacy file watcher for backward compatibility."""

    def __init__(self):
        self._config_watcher = ConfigWatcher()
        logger.warning("FileWatcher is deprecated. Use ConfigWatcher instead.")

    def watch(self, file_path: str, callback) -> None:
        """Watch file for changes."""

        def change_handler(event: ConfigChangeEvent):
            if str(event.file_path) == file_path:
                callback(event)

        self._config_watcher.on_file_changed(change_handler)
        self._config_watcher.start_watching([Path(file_path)])

    def stop_watching(self, file_path: str) -> None:
        """Stop watching file."""
        self._config_watcher.remove_file(Path(file_path))
