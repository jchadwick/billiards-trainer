"""Unit tests for the ConfigurationModule."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.config.manager import ConfigurationModule
from backend.config.models.schemas import ConfigSource


class TestConfigurationModule:
    """Test suite for ConfigurationModule."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config_module(self, temp_config_dir):
        """Create a ConfigurationModule instance for testing."""
        return ConfigurationModule(config_dir=temp_config_dir)

    def test_initialization(self, config_module, temp_config_dir):
        """Test that the configuration module initializes properly."""
        assert config_module.config_dir == temp_config_dir
        assert config_module._data is not None
        assert config_module._schemas is not None
        assert config_module._subscriptions is not None
        assert config_module._history is not None

        # Check that default values are loaded
        assert config_module.get("app.name") == "billiards-trainer"
        assert config_module.get("app.version") == "1.0.0"
        assert config_module.get("api.host") == "localhost"
        assert config_module.get("api.port") == 8000

    def test_directory_creation(self, temp_config_dir):
        """Test that required directories are created."""
        config_module = ConfigurationModule(config_dir=temp_config_dir)

        assert (temp_config_dir / "profiles").exists()
        assert (config_module._settings.paths.data_dir).exists()
        assert (config_module._settings.paths.log_dir).exists()
        assert (config_module._settings.paths.cache_dir).exists()

    def test_get_and_set_basic(self, config_module):
        """Test basic get and set functionality."""
        # Test setting a new value
        assert config_module.set("test.key", "test_value")
        assert config_module.get("test.key") == "test_value"

        # Test getting non-existent key with default
        assert config_module.get("non.existent", "default") == "default"

        # Test setting with different types
        assert config_module.set("test.number", 42)
        assert config_module.get("test.number") == 42

        assert config_module.set("test.boolean", True)
        assert config_module.get("test.boolean") is True

        assert config_module.set("test.list", [1, 2, 3])
        assert config_module.get("test.list") == [1, 2, 3]

    def test_get_with_type_hint(self, config_module):
        """Test type checking in get method."""
        config_module.set("test.string_number", "123")

        # Should convert string to int
        assert config_module.get("test.string_number", type_hint=int) == 123

        # Should return default if conversion fails
        config_module.set("test.invalid_number", "abc")
        assert config_module.get("test.invalid_number", default=0, type_hint=int) == 0

    def test_get_all(self, config_module):
        """Test getting all configuration values."""
        config_module.set("test.key1", "value1")
        config_module.set("test.key2", "value2")
        config_module.set("other.key", "value3")

        all_configs = config_module.get_all()
        assert "test.key1" in all_configs
        assert "test.key2" in all_configs
        assert "other.key" in all_configs

        # Test with prefix
        test_configs = config_module.get_all(prefix="test.")
        assert "test.key1" in test_configs
        assert "test.key2" in test_configs
        assert "other.key" not in test_configs

    def test_load_config_file(self, config_module, temp_config_dir):
        """Test loading configuration from JSON file."""
        config_data = {
            "app": {"name": "test-app", "debug": True},
            "api": {"port": 3000},
        }

        config_file = temp_config_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        assert config_module.load_config(config_file)

        # Check that values were loaded
        assert config_module.get("app.name") == "test-app"
        assert config_module.get("app.debug") is True
        assert config_module.get("api.port") == 3000

    def test_load_config_file_not_exists(self, config_module, temp_config_dir):
        """Test loading from non-existent file."""
        non_existent = temp_config_dir / "non_existent.json"
        assert not config_module.load_config(non_existent)

    def test_save_config(self, config_module, temp_config_dir):
        """Test saving configuration to file."""
        # Set some values
        config_module.set("test.key", "value", source=ConfigSource.RUNTIME)
        config_module.set("test.number", 42, source=ConfigSource.RUNTIME)

        # Save configuration
        save_path = temp_config_dir / "saved_config.json"
        assert config_module.save_config(save_path)
        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            saved_data = json.load(f)

        assert saved_data["test"]["key"] == "value"
        assert saved_data["test"]["number"] == 42

    def test_environment_variables(self, config_module):
        """Test loading configuration from environment variables."""
        env_vars = {
            "BILLIARDS_APP_NAME": "env-app",
            "BILLIARDS_API_PORT": "9000",
            "BILLIARDS_DEBUG_MODE": "true",
        }

        with patch.dict(os.environ, env_vars):
            config_module.load_environment_variables()

        assert config_module.get("app.name") == "env-app"
        assert config_module.get("api.port") == 9000  # JSON parsed as integer
        assert config_module.get("debug.mode") is True  # JSON parsed as boolean

    def test_validation_basic(self, config_module):
        """Test basic validation functionality."""
        # Register a schema
        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        config_module.register_schema("test.number", schema)

        # Set a valid value
        assert config_module.set("test.number", 50)
        is_valid, errors = config_module.validate("test.number")
        assert is_valid
        assert len(errors) == 0

        # Try to set an invalid value
        config_module._schemas["test.number"] = schema  # Ensure schema is registered
        is_valid, errors = config_module._validate_value("test.number", 150, schema)
        assert not is_valid
        assert len(errors) > 0

    def test_validation_type_checking(self, config_module):
        """Test type validation."""
        schema = {"type": "string"}
        config_module.register_schema("test.string", schema)

        # Valid string
        is_valid, errors = config_module._validate_value("test.string", "hello", schema)
        assert is_valid

        # Invalid type
        is_valid, errors = config_module._validate_value("test.string", 123, schema)
        assert not is_valid
        assert "Expected string" in errors[0]

    def test_subscriptions(self, config_module):
        """Test configuration change subscriptions."""
        changes = []

        def callback(change):
            changes.append(change)

        # Subscribe to changes
        sub_id = config_module.subscribe("test.*", callback)
        assert sub_id is not None

        # Make a change
        config_module.set("test.key", "value")

        # Check that callback was called
        assert len(changes) == 1
        assert changes[0].key == "test.key"
        assert changes[0].new_value == "value"

        # Unsubscribe
        assert config_module.unsubscribe(sub_id)

        # Make another change
        config_module.set("test.key2", "value2")

        # Should still be only one change (callback not called again)
        assert len(changes) == 1

    def test_history(self, config_module):
        """Test configuration change history."""
        # Make some changes
        config_module.set("test.key1", "value1")
        config_module.set("test.key2", "value2")
        config_module.set("test.key1", "updated_value1")

        # Get history
        history = config_module.get_history()
        assert len(history) >= 3

        # Get history for specific key
        key1_history = config_module.get_history("test.key1")
        assert len(key1_history) >= 2

        # Check that most recent is first
        assert key1_history[0].new_value == "updated_value1"

    def test_module_registration(self, config_module):
        """Test module configuration registration."""
        module_spec = {
            "module_name": "vision",
            "configuration": {
                "camera.device_id": {"type": "integer", "default": 0, "minimum": 0},
                "camera.resolution": {"type": "array", "default": [1920, 1080]},
            },
        }

        config_module.register_module("vision", module_spec)

        # Set some values for the module
        config_module.set("vision.camera.device_id", 1)
        config_module.set("vision.camera.resolution", [1280, 720])

        # Get module configuration
        vision_config = config_module.get_module_config("vision")
        assert vision_config["camera"]["device_id"] == 1
        assert vision_config["camera"]["resolution"] == [1280, 720]

    def test_reload_config(self, config_module, temp_config_dir):
        """Test configuration reload functionality."""
        # Create a config file
        config_data = {"app": {"name": "reloaded-app"}}
        config_file = temp_config_dir / "config" / "default.json"
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Set a runtime value
        config_module.set("runtime.value", "test")

        # Reload
        assert config_module.reload_config()

        # Runtime value should be gone, file value should be loaded
        assert config_module.get("runtime.value") is None
        assert config_module.get("app.name") == "reloaded-app"

    def test_profiles(self, config_module, temp_config_dir):
        """Test configuration profiles."""
        # Create a profile
        settings = {"test.key": "profile_value"}
        profile = config_module.create_profile("test_profile", settings)

        assert profile.name == "test_profile"
        assert profile.settings == settings

        # Export profile
        profile_path = temp_config_dir / "test_profile.json"
        assert config_module.export_profile("test_profile", profile_path)
        assert profile_path.exists()

    def test_flatten_unflatten_dict(self, config_module):
        """Test dictionary flattening and unflattening."""
        nested = {"level1": {"level2": {"key": "value"}, "simple": "test"}}

        flattened = config_module._flatten_dict(nested)
        assert flattened["level1.level2.key"] == "value"
        assert flattened["level1.simple"] == "test"

        unflattened = config_module._unflatten_dict(flattened)
        assert unflattened == nested

    def test_config_metadata(self, config_module):
        """Test configuration metadata tracking."""
        config_module.set("test.key", "value")

        metadata = config_module.get_metadata("test.key")
        assert metadata is not None
        assert metadata.key == "test.key"
        assert metadata.value == "value"
        assert metadata.source == ConfigSource.RUNTIME
        assert metadata.timestamp > 0
