"""Tests for environment variable configuration loader.

Tests all functionality including type conversion, prefix filtering,
nested structure support, and schema-based loading.
"""

import os

import pytest

from ..loader.env import EnvironmentError, EnvironmentLoader, TypeConversionError


class TestEnvironmentLoader:
    """Test cases for EnvironmentLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = EnvironmentLoader(prefix="TEST_")
        # Save original environment
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_init_with_defaults(self):
        """Test EnvironmentLoader initialization with defaults."""
        loader = EnvironmentLoader()
        assert loader.prefix == ""
        assert loader.separator == "_"
        assert loader.nested_separator == "__"

    def test_init_with_custom_params(self):
        """Test EnvironmentLoader initialization with custom parameters."""
        loader = EnvironmentLoader(
            prefix="MYAPP_", separator="-", nested_separator="___"
        )
        assert loader.prefix == "MYAPP_"
        assert loader.separator == "-"
        assert loader.nested_separator == "___"

    def test_env_key_to_config_key(self):
        """Test conversion from environment key to config key."""
        # Simple key
        result = self.loader._env_key_to_config_key("TEST_DATABASE_HOST")
        assert result == "database.host"

        # Nested key
        result = self.loader._env_key_to_config_key("TEST_APP__FEATURES__VISION")
        assert result == "app.features.vision"

    def test_config_key_to_env_key(self):
        """Test conversion from config key to environment key."""
        # Simple key
        result = self.loader._config_key_to_env_key("database.host")
        assert result == "TEST_DATABASE__HOST"

        # Nested key
        result = self.loader._config_key_to_env_key("app.features.vision")
        assert result == "TEST_APP__FEATURES__VISION"

    def test_convert_bool_true_values(self):
        """Test boolean conversion for true values."""
        true_values = ["true", "True", "TRUE", "yes", "YES", "on", "ON", "1"]
        for value in true_values:
            assert self.loader._convert_bool(value) is True

    def test_convert_bool_false_values(self):
        """Test boolean conversion for false values."""
        false_values = ["false", "False", "FALSE", "no", "NO", "off", "OFF", "0"]
        for value in false_values:
            assert self.loader._convert_bool(value) is False

    def test_convert_bool_invalid(self):
        """Test boolean conversion for invalid values."""
        with pytest.raises(ValueError):
            self.loader._convert_bool("invalid")

    def test_convert_json_valid(self):
        """Test JSON conversion for valid JSON."""
        result = self.loader._convert_json('{"key": "value", "number": 42}')
        assert result == {"key": "value", "number": 42}

        result = self.loader._convert_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_convert_json_invalid(self):
        """Test JSON conversion for invalid JSON."""
        with pytest.raises(ValueError):
            self.loader._convert_json("invalid json")

    def test_convert_list(self):
        """Test list conversion."""
        result = self.loader._convert_list("item1,item2,item3")
        assert result == ["item1", "item2", "item3"]

        result = self.loader._convert_list("single")
        assert result == ["single"]

        result = self.loader._convert_list("")
        assert result == []

    def test_convert_value_auto_inference(self):
        """Test automatic type inference."""
        # Boolean
        assert self.loader._convert_value("true", "TEST_KEY") is True
        assert self.loader._convert_value("false", "TEST_KEY") is False

        # Integer
        assert self.loader._convert_value("42", "TEST_KEY") == 42
        assert self.loader._convert_value("-42", "TEST_KEY") == -42

        # Float
        assert self.loader._convert_value("3.14", "TEST_KEY") == 3.14

        # JSON
        assert self.loader._convert_value('{"key": "value"}', "TEST_KEY") == {
            "key": "value"
        }

        # List
        assert self.loader._convert_value("a,b,c", "TEST_KEY") == ["a", "b", "c"]

        # String (default)
        assert (
            self.loader._convert_value("simple string", "TEST_KEY") == "simple string"
        )

    def test_convert_value_with_type(self):
        """Test value conversion with explicit type."""
        assert self.loader._convert_value_with_type("42", "int") == 42
        assert self.loader._convert_value_with_type("3.14", "float") == 3.14
        assert self.loader._convert_value_with_type("true", "bool") is True
        assert self.loader._convert_value_with_type('{"key": "value"}', "json") == {
            "key": "value"
        }
        assert self.loader._convert_value_with_type("a,b,c", "list") == ["a", "b", "c"]

    def test_convert_value_unknown_type(self):
        """Test value conversion with unknown type."""
        with pytest.raises(TypeConversionError):
            self.loader._convert_value_with_type("value", "unknown_type")

    def test_set_nested_value(self):
        """Test setting nested values in configuration."""
        config = {}

        self.loader._set_nested_value(config, "app.name", "test-app")
        assert config == {"app": {"name": "test-app"}}

        self.loader._set_nested_value(config, "app.debug", True)
        assert config == {"app": {"name": "test-app", "debug": True}}

        self.loader._set_nested_value(config, "database.host", "localhost")
        assert config == {
            "app": {"name": "test-app", "debug": True},
            "database": {"host": "localhost"},
        }

    def test_load_environment_basic(self):
        """Test basic environment variable loading."""
        # Set test environment variables
        os.environ["TEST_APP_NAME"] = "test-app"
        os.environ["TEST_APP_DEBUG"] = "true"
        os.environ["TEST_DATABASE_HOST"] = "localhost"
        os.environ["TEST_DATABASE_PORT"] = "5432"

        result = self.loader.load_environment()

        assert result["app"]["name"] == "test-app"
        assert result["app"]["debug"] is True
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432

    def test_load_environment_nested(self):
        """Test loading environment variables with nested structure."""
        os.environ["TEST_APP__FEATURES__VISION"] = "true"
        os.environ["TEST_APP__FEATURES__PROJECTOR"] = "false"

        result = self.loader.load_environment()

        assert result["app"]["features"]["vision"] is True
        assert result["app"]["features"]["projector"] is False

    def test_load_environment_with_prefix_filter(self):
        """Test environment loading with prefix filtering."""
        os.environ["TEST_VALID_KEY"] = "value1"
        os.environ["OTHER_INVALID_KEY"] = "value2"

        result = self.loader.load_environment()

        assert "valid" in result
        assert result["valid"]["key"] == "value1"
        assert "other" not in result

    def test_load_environment_empty(self):
        """Test loading when no matching environment variables exist."""
        # Clear any TEST_ variables
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

        result = self.loader.load_environment()
        assert result == {}

    def test_load_with_schema(self):
        """Test loading environment variables with schema."""
        schema = {
            "app.name": {"type": "str", "default": "default-app"},
            "app.debug": {"type": "bool", "default": False},
            "database.port": {"type": "int", "default": 5432},
            "required_key": {"type": "str", "required": True},
        }

        # Set some environment variables
        os.environ["TEST_APP__NAME"] = "test-app"
        os.environ["TEST_APP__DEBUG"] = "true"
        os.environ["TEST_REQUIRED_KEY"] = "required-value"

        result = self.loader.load_with_schema(schema)

        assert result["app"]["name"] == "test-app"
        assert result["app"]["debug"] is True
        assert result["database"]["port"] == 5432  # default value
        assert result["required_key"] == "required-value"

    def test_load_with_schema_missing_required(self):
        """Test schema loading with missing required value in strict mode."""
        schema = {"required_key": {"type": "str", "required": True}}

        with pytest.raises(EnvironmentError):
            self.loader.load_with_schema(schema, strict=True)

    def test_load_with_schema_type_conversion_error(self):
        """Test schema loading with type conversion error."""
        schema = {"app.port": {"type": "int"}}

        os.environ["TEST_APP__PORT"] = "not-a-number"

        # In non-strict mode, should use raw value
        result = self.loader.load_with_schema(schema, strict=False)
        assert result["app"]["port"] == "not-a-number"

        # In strict mode, should raise error
        with pytest.raises(TypeConversionError):
            self.loader.load_with_schema(schema, strict=True)

    def test_get_env_var(self):
        """Test getting a single environment variable."""
        os.environ["TEST_SINGLE_KEY"] = "test-value"

        result = self.loader.get_env_var("single.key")
        assert result == "test-value"

        # Test with default
        result = self.loader.get_env_var("nonexistent.key", default="default-value")
        assert result == "default-value"

        # Test with type conversion
        os.environ["TEST_NUMBER_KEY"] = "42"
        result = self.loader.get_env_var("number.key", value_type="int")
        assert result == 42

    def test_set_env_var(self):
        """Test setting an environment variable."""
        self.loader.set_env_var("test.key", "test-value")
        assert os.environ["TEST_TEST__KEY"] == "test-value"

    def test_list_relevant_env_vars(self):
        """Test listing relevant environment variables."""
        os.environ["TEST_KEY1"] = "value1"
        os.environ["TEST_KEY2"] = "value2"
        os.environ["OTHER_KEY"] = "value3"

        relevant_vars = self.loader.list_relevant_env_vars()

        assert "TEST_KEY1" in relevant_vars
        assert "TEST_KEY2" in relevant_vars
        assert "OTHER_KEY" not in relevant_vars

    def test_load_environment_with_include_patterns(self):
        """Test loading environment variables with include patterns."""
        os.environ["TEST_APP_NAME"] = "app-name"
        os.environ["TEST_DB_HOST"] = "db-host"
        os.environ["TEST_CACHE_URL"] = "cache-url"

        # Only include variables matching APP or DB
        result = self.loader.load_environment(include_patterns=[r".*APP.*", r".*DB.*"])

        assert "app" in result
        assert "db" in result
        assert "cache" not in result

    def test_load_environment_with_exclude_patterns(self):
        """Test loading environment variables with exclude patterns."""
        os.environ["TEST_APP_NAME"] = "app-name"
        os.environ["TEST_SECRET_KEY"] = "secret"
        os.environ["TEST_DB_HOST"] = "db-host"

        # Exclude SECRET variables
        result = self.loader.load_environment(exclude_patterns=[r".*SECRET.*"])

        assert "app" in result
        assert "db" in result
        assert "secret" not in result


class TestEnvironmentLoaderIntegration:
    """Integration tests for EnvironmentLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Clean up after tests."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_complex_environment_loading(self):
        """Test loading a complex environment configuration."""
        # Set up complex environment
        env_vars = {
            "BILLIARDS_APP__NAME": "billiards-trainer",
            "BILLIARDS_APP__VERSION": "1.0.0",
            "BILLIARDS_APP__DEBUG": "true",
            "BILLIARDS_APP__FEATURES__VISION": "true",
            "BILLIARDS_APP__FEATURES__PROJECTOR": "false",
            "BILLIARDS_DATABASE__HOST": "localhost",
            "BILLIARDS_DATABASE__PORT": "5432",
            "BILLIARDS_DATABASE__NAME": "billiards",
            "BILLIARDS_VISION__CAMERA__DEVICE_ID": "0",
            "BILLIARDS_VISION__CAMERA__RESOLUTION": "[1920, 1080]",
            "BILLIARDS_VISION__DETECTION__SENSITIVITY": "0.8",
            "BILLIARDS_PROJECTOR__BRIGHTNESS": "75",
            "BILLIARDS_LOGGING__LEVEL": "DEBUG",
            "BILLIARDS_LOGGING__HANDLERS": "console,file",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        loader = EnvironmentLoader(prefix="BILLIARDS_")
        result = loader.load_environment()

        # Check structure and values
        assert result["app"]["name"] == "billiards-trainer"
        assert result["app"]["version"] == "1.0.0"
        assert result["app"]["debug"] is True
        assert result["app"]["features"]["vision"] is True
        assert result["app"]["features"]["projector"] is False

        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432
        assert result["database"]["name"] == "billiards"

        assert result["vision"]["camera"]["device_id"] == 0
        assert result["vision"]["camera"]["resolution"] == [1920, 1080]
        assert result["vision"]["detection"]["sensitivity"] == 0.8

        assert result["projector"]["brightness"] == 75

        assert result["logging"]["level"] == "DEBUG"
        assert result["logging"]["handlers"] == ["console", "file"]
