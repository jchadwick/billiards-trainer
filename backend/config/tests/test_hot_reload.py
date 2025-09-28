"""Tests for configuration hot reload functionality."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from backend.config.manager import ConfigurationModule
from backend.config.models.schemas import ConfigSource
from backend.config.utils.watcher import ConfigChangeEvent, ConfigWatcher


class TestConfigWatcher:
    """Test the ConfigWatcher class."""

    def test_config_watcher_initialization(self):
        """Test ConfigWatcher initialization."""
        watcher = ConfigWatcher(debounce_delay=0.1)

        assert watcher._debounce_delay == 0.1
        assert not watcher._is_watching
        assert len(watcher._watched_files) == 0
        assert len(watcher._change_callbacks) == 0
        assert len(watcher._validation_callbacks) == 0
        assert len(watcher._rollback_callbacks) == 0

    def test_callback_registration(self):
        """Test callback registration methods."""
        watcher = ConfigWatcher()

        # Test change callback registration
        def change_callback(event):
            pass

        watcher.on_file_changed(change_callback)
        assert change_callback in watcher._change_callbacks

        # Test validation callback registration
        def validation_callback(file_path, content):
            return True

        watcher.on_validation_needed(validation_callback)
        assert validation_callback in watcher._validation_callbacks

        # Test rollback callback registration
        def rollback_callback(file_path, error):
            pass

        watcher.on_rollback_needed(rollback_callback)
        assert rollback_callback in watcher._rollback_callbacks

    def test_file_loading_json(self):
        """Test JSON file loading."""
        watcher = ConfigWatcher()

        test_config = {"app": {"name": "test", "debug": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            temp_path = Path(f.name)

        try:
            loaded_config = watcher._load_file(temp_path)
            assert loaded_config == test_config
        finally:
            temp_path.unlink()

    def test_file_loading_yaml(self):
        """Test YAML file loading."""
        pytest.importorskip("yaml")  # Skip if PyYAML not available

        watcher = ConfigWatcher()

        yaml_content = """
app:
  name: test
  debug: true
database:
  host: localhost
  port: 5432
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            loaded_config = watcher._load_file(temp_path)
            assert loaded_config["app"]["name"] == "test"
            assert loaded_config["app"]["debug"] is True
            assert loaded_config["database"]["host"] == "localhost"
            assert loaded_config["database"]["port"] == 5432
        finally:
            temp_path.unlink()

    def test_file_loading_invalid_format(self):
        """Test loading file with unsupported format."""
        watcher = ConfigWatcher()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                watcher._load_file(temp_path)
        finally:
            temp_path.unlink()

    def test_add_remove_files(self):
        """Test adding and removing files from watch list."""
        watcher = ConfigWatcher()

        test_config = {"test": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            temp_path = Path(f.name)

        try:
            # Test adding file
            assert watcher.add_file(temp_path)
            # Check if the resolved path is in watched files
            assert temp_path.resolve() in watcher._watched_files
            assert watcher.is_watching_file(temp_path)

            # Test removing file
            assert watcher.remove_file(temp_path)
            assert temp_path.resolve() not in watcher._watched_files
            assert not watcher.is_watching_file(temp_path)

        finally:
            temp_path.unlink()

    def test_add_nonexistent_file(self):
        """Test adding a non-existent file."""
        watcher = ConfigWatcher()

        nonexistent_path = Path("/tmp/nonexistent_config.json")
        assert not watcher.add_file(nonexistent_path)


class TestConfigurationModuleHotReload:
    """Test hot reload functionality in ConfigurationModule."""

    def test_hot_reload_initialization(self):
        """Test hot reload initialization in ConfigurationModule."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create a test configuration file
            test_config = {"app": {"name": "test-app", "version": "1.0"}}
            config_file = config_dir / "test.json"
            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Initialize configuration module with hot reload enabled
            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Check that hot reload is initialized but may not be active
            # (since no files may be in the default watched list)
            assert hasattr(config_module, "_config_watcher")
            assert hasattr(config_module, "_watched_files")
            assert hasattr(config_module, "_enable_hot_reload")
            assert config_module._enable_hot_reload is True

    def test_hot_reload_disabled(self):
        """Test ConfigurationModule with hot reload disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=False
            )

            assert config_module._enable_hot_reload is False
            assert config_module._config_watcher is None

    def test_enable_disable_hot_reload(self):
        """Test enabling and disabling hot reload."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Start with hot reload disabled
            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=False
            )

            assert not config_module.is_hot_reload_enabled()

            # Enable hot reload
            config_module.enable_hot_reload()
            # Note: May not be enabled if no files to watch

            # Disable hot reload
            assert config_module.disable_hot_reload()
            assert not config_module.is_hot_reload_enabled()

    def test_add_remove_watched_files(self):
        """Test adding and removing watched files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Create a test configuration file
            test_config = {"test": {"key": "value"}}
            config_file = config_dir / "test.json"
            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Add file to watch list
            assert config_module.add_watched_file(config_file)
            watched_files = config_module.get_watched_files()
            assert config_file in watched_files

            # Remove file from watch list
            assert config_module.remove_watched_file(config_file)
            watched_files = config_module.get_watched_files()
            assert config_file not in watched_files

    def test_config_file_validation(self):
        """Test configuration file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Test valid configuration
            valid_config = {"app": {"name": "test"}}
            config_file = Path(temp_dir) / "valid.json"

            assert config_module._validate_config_file(config_file, valid_config)

            # Test invalid configuration (not a dict)
            invalid_config = "not a dictionary"
            assert not config_module._validate_config_file(config_file, invalid_config)

    @pytest.mark.asyncio
    async def test_force_reload(self):
        """Test force reload functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create initial configuration
            initial_config = {"app": {"name": "initial", "version": "1.0"}}
            config_file = config_dir / "test.json"
            with open(config_file, "w") as f:
                json.dump(initial_config, f)

            config_module = ConfigurationModule(
                config_dir=config_dir,
                enable_hot_reload=False,  # Test without hot reload
            )

            # Load initial configuration
            config_module.load_config(config_file)
            assert config_module.get("app.name") == "initial"

            # Update configuration file
            updated_config = {"app": {"name": "updated", "version": "2.0"}}
            with open(config_file, "w") as f:
                json.dump(updated_config, f)

            # Force reload
            success = await config_module.force_reload(config_file)
            assert success
            assert config_module.get("app.name") == "updated"
            assert config_module.get("app.version") == "2.0"

    def test_config_change_event_creation(self):
        """Test ConfigChangeEvent creation and attributes."""
        file_path = Path("/tmp/test.json")
        old_content = {"app": {"name": "old"}}
        new_content = {"app": {"name": "new"}}
        timestamp = time.time()

        event = ConfigChangeEvent(
            file_path=file_path,
            event_type="modified",
            timestamp=timestamp,
            old_content=old_content,
            new_content=new_content,
        )

        assert event.file_path == file_path
        assert event.event_type == "modified"
        assert event.timestamp == timestamp
        assert event.old_content == old_content
        assert event.new_content == new_content

    def test_config_rollback_handling(self):
        """Test configuration rollback handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            config_file = Path(temp_dir) / "test.json"
            error = ValueError("Test error")

            # Test rollback handling (should not raise exception)
            config_module._handle_config_rollback(config_file, error)

    @patch("backend.config.utils.watcher.Observer")
    def test_hot_reload_with_file_watching(self, mock_observer):
        """Test hot reload with actual file watching (mocked)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create configuration files
            config_files = []
            for i in range(2):
                config = {"module": f"module{i}", "value": i}
                config_file = config_dir / f"config{i}.json"
                with open(config_file, "w") as f:
                    json.dump(config, f)
                config_files.append(config_file)

            # Mock observer
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance

            # Create watcher
            watcher = ConfigWatcher()

            # Start watching
            success = watcher.start_watching(config_files)
            assert success
            assert watcher._is_watching

            # Check that observer was configured
            mock_observer_instance.start.assert_called_once()

            # Stop watching
            success = watcher.stop_watching()
            assert success
            assert not watcher._is_watching

            mock_observer_instance.stop.assert_called_once()
            mock_observer_instance.join.assert_called_once()

    def test_subscription_and_notification(self):
        """Test configuration change subscription and notification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Create mock subscriber
            changes_received = []

            def mock_subscriber(change):
                changes_received.append(change)

            # Subscribe to changes
            subscription_id = config_module.subscribe("app.*", mock_subscriber)
            assert subscription_id

            # Simulate configuration change
            config_module.set("app.name", "test-app", source=ConfigSource.FILE)

            # Check that subscriber was notified
            assert len(changes_received) > 0
            assert changes_received[-1].key == "app.name"
            assert changes_received[-1].new_value == "test-app"

            # Unsubscribe
            assert config_module.unsubscribe(subscription_id)

    @pytest.mark.asyncio
    async def test_debounced_file_changes(self):
        """Test that rapid file changes are debounced."""
        watcher = ConfigWatcher(debounce_delay=0.1)

        # Mock the _process_file_change method
        process_calls = []

        async def mock_process_file_change(file_path, event_type):
            process_calls.append((file_path, event_type))

        watcher._process_file_change = mock_process_file_change

        test_file = Path("/tmp/test.json")

        # Simulate rapid file changes
        await watcher._handle_file_change(test_file, "modified")
        await watcher._handle_file_change(test_file, "modified")
        await watcher._handle_file_change(test_file, "modified")

        # Wait for debounce delay plus a bit extra
        await asyncio.sleep(0.15)

        # Should have only processed the last change
        assert len(process_calls) == 1
        assert process_calls[0] == (test_file, "modified")


class TestIntegrationScenarios:
    """Integration tests for hot reload functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_hot_reload_simulation(self):
        """Test complete hot reload workflow simulation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create initial configuration
            initial_config = {
                "app": {"name": "billiards-trainer", "debug": False},
                "vision": {"camera": {"fps": 30}},
            }
            config_file = config_dir / "app.json"
            with open(config_file, "w") as f:
                json.dump(initial_config, f)

            # Initialize configuration module
            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Load initial configuration
            config_module.load_config(config_file)

            # Verify initial values
            assert config_module.get("app.name") == "billiards-trainer"
            assert config_module.get("app.debug") is False
            assert config_module.get("vision.camera.fps") == 30

            # Add file to watch list
            config_module.add_watched_file(config_file)

            # Simulate configuration change
            updated_config = {
                "app": {"name": "billiards-trainer", "debug": True},
                "vision": {"camera": {"fps": 60}},
            }

            # Create change event
            change_event = ConfigChangeEvent(
                file_path=config_file,
                event_type="modified",
                timestamp=time.time(),
                old_content=initial_config,
                new_content=updated_config,
            )

            # Update the file
            with open(config_file, "w") as f:
                json.dump(updated_config, f)

            # Simulate hot reload event
            config_module._handle_config_file_change(change_event)

            # Verify updated values
            assert config_module.get("app.debug") is True
            assert config_module.get("vision.camera.fps") == 60

            # Check history
            history = config_module.get_history()
            assert len(history) > 0

            # Find the debug change in history
            debug_changes = [h for h in history if h.key == "app.debug"]
            assert len(debug_changes) > 0
            assert debug_changes[0].old_value is False
            assert debug_changes[0].new_value is True

    def test_multiple_file_watching(self):
        """Test watching multiple configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create multiple configuration files
            configs = [
                {"module": "vision", "enabled": True},
                {"module": "core", "threads": 4},
                {"module": "api", "port": 8000},
            ]

            config_files = []
            for i, config in enumerate(configs):
                config_file = config_dir / f"module{i}.json"
                with open(config_file, "w") as f:
                    json.dump(config, f)
                config_files.append(config_file)

            # Initialize watcher
            watcher = ConfigWatcher()

            # Add all files
            for config_file in config_files:
                assert watcher.add_file(config_file)

            # Verify all files are being watched
            watched_files = watcher.get_watched_files()
            assert len(watched_files) == len(config_files)
            # Check resolved paths since the watcher stores resolved paths
            watched_resolved = {f.resolve() for f in watched_files}
            for config_file in config_files:
                assert config_file.resolve() in watched_resolved

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Create invalid JSON file
            invalid_config_file = config_dir / "invalid.json"
            with open(invalid_config_file, "w") as f:
                f.write("{ invalid json content")

            # Test validation - pass empty dict which should be valid
            # Testing with actual invalid content requires loading the file
            assert config_module._validate_config_file(invalid_config_file, {})

            # Test with invalid content (not a dict)
            assert not config_module._validate_config_file(
                invalid_config_file, "not a dict"
            )

            # Test rollback handling
            error = ValueError("Invalid JSON")
            config_module._handle_config_rollback(invalid_config_file, error)

    def test_cleanup_on_destruction(self):
        """Test that resources are cleaned up when ConfigurationModule is destroyed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_module = ConfigurationModule(
                config_dir=config_dir, enable_hot_reload=True
            )

            # Simulate having a watcher
            config_module._config_watcher = Mock()

            # Test cleanup
            config_module.__del__()

            # Verify stop_watching was called
            config_module._config_watcher.stop_watching.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
