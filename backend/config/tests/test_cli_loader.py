"""Tests for CLI argument configuration loader."""

import argparse
from unittest.mock import patch

import pytest

from ..loader.cli import ArgumentError, CLIError, CLILoader, TypeConversionError


class TestCLILoader:
    """Test CLI argument configuration loader."""

    def test_init_default(self):
        """Test default initialization."""
        loader = CLILoader()

        assert loader.prog_name == "billiards-trainer"
        assert loader.description == "Professional Billiards Training System"
        assert loader.prefix == "--"
        assert loader.nested_separator == "."
        assert isinstance(loader.parser, argparse.ArgumentParser)

    def test_init_custom(self):
        """Test custom initialization."""
        schema = {"test": {"type": "str", "description": "Test option"}}
        loader = CLILoader(
            prog_name="test-app",
            description="Test Description",
            prefix="-",
            nested_separator="__",
            schema=schema,
        )

        assert loader.prog_name == "test-app"
        assert loader.description == "Test Description"
        assert loader.prefix == "-"
        assert loader.nested_separator == "__"
        assert loader.schema == schema

    def test_load_empty_args(self):
        """Test loading with no arguments."""
        loader = CLILoader()
        config = loader.load([])

        assert isinstance(config, dict)
        # Should be empty since no arguments provided
        assert len(config) == 0

    def test_load_basic_arguments(self):
        """Test loading basic arguments."""
        loader = CLILoader()
        args = ["--debug", "--log-level", "DEBUG", "--port", "8080", "--camera", "1"]

        config = loader.load(args)

        assert config["system"]["debug"] is True
        assert config["system"]["logging"]["level"] == "DEBUG"
        assert config["api"]["network"]["port"] == 8080
        assert config["vision"]["camera"]["device_id"] == 1

    def test_load_nested_configuration(self):
        """Test loading nested configuration paths."""
        loader = CLILoader()
        args = ["--host", "localhost", "--resolution", "1920x1080"]

        config = loader.load(args)

        assert config["api"]["network"]["host"] == "localhost"
        assert config["vision"]["camera"]["resolution"] == (1920, 1080)

    def test_load_with_config_file(self):
        """Test loading with config file specification."""
        loader = CLILoader()
        args = [
            "--config",
            "/path/to/config.json",
            "--config-format",
            "json",
            "--profile",
            "development",
        ]

        config = loader.load(args)

        assert config["config_file"] == "/path/to/config.json"
        assert config["config_format"] == "json"
        assert config["active_profile"] == "development"

    def test_type_conversions(self):
        """Test automatic type conversions."""
        loader = CLILoader()

        # Test boolean conversion
        assert loader._convert_bool("true") is True
        assert loader._convert_bool("false") is False
        assert loader._convert_bool("yes") is True
        assert loader._convert_bool("no") is False
        assert loader._convert_bool("1") is True
        assert loader._convert_bool("0") is False

        with pytest.raises(ValueError):
            loader._convert_bool("invalid")

    def test_json_conversion(self):
        """Test JSON string conversion."""
        loader = CLILoader()

        # Test valid JSON
        result = loader._convert_json('{"key": "value"}')
        assert result == {"key": "value"}

        result = loader._convert_json("[1, 2, 3]")
        assert result == [1, 2, 3]

        # Test invalid JSON
        with pytest.raises(ValueError):
            loader._convert_json("invalid json")

    def test_list_conversion(self):
        """Test comma-separated list conversion."""
        loader = CLILoader()

        result = loader._convert_list("a,b,c")
        assert result == ["a", "b", "c"]

        result = loader._convert_list("a, b , c ")
        assert result == ["a", "b", "c"]

        result = loader._convert_list("")
        assert result == []

    def test_resolution_conversion(self):
        """Test resolution string conversion."""
        loader = CLILoader()

        result = loader._convert_string_value("1920x1080", "vision.camera.resolution")
        assert result == (1920, 1080)

        result = loader._convert_string_value("640x480", "resolution")
        assert result == (640, 480)

        with pytest.raises(TypeConversionError):
            loader._convert_string_value("invalid", "resolution")

    def test_nested_value_setting(self):
        """Test setting nested values."""
        loader = CLILoader()
        config = {}

        loader._set_nested_value(config, "system.debug", True)
        assert config["system"]["debug"] is True

        loader._set_nested_value(config, "api.network.port", 8080)
        assert config["api"]["network"]["port"] == 8080

        loader._set_nested_value(config, "simple", "value")
        assert config["simple"] == "value"

    def test_add_custom_argument(self):
        """Test adding custom arguments."""
        loader = CLILoader()

        loader.add_argument(
            "custom_option",
            help="Custom option",
            config_key="custom.option",
            value_type="str",
        )

        config = loader.load(["--custom-option", "test_value"])
        assert config["custom"]["option"] == "test_value"

    def test_schema_based_arguments(self):
        """Test schema-based argument generation."""
        schema = {
            "database": {
                "host": {
                    "type": "str",
                    "default": "localhost",
                    "description": "Database host",
                },
                "port": {
                    "type": "int",
                    "default": 5432,
                    "description": "Database port",
                },
                "ssl": {"type": "bool", "default": False, "description": "Enable SSL"},
            }
        }

        loader = CLILoader(schema=schema)

        # Test that schema arguments are available
        help_text = loader.get_help_text()
        assert "--database-host" in help_text
        assert "--database-port" in help_text
        assert "--database-ssl" in help_text

    def test_load_with_schema(self):
        """Test loading with a specific schema."""
        schema = {"test_option": {"type": "str", "description": "Test option"}}

        loader = CLILoader()
        config = loader.load_with_schema(schema, ["--test-option", "value"])

        assert config["test_option"] == "value"

    def test_validate_arguments(self):
        """Test argument validation."""
        loader = CLILoader()

        # Valid arguments
        valid, errors = loader.validate_arguments(["--debug", "--port", "8080"])
        assert valid is True
        assert len(errors) == 0

        # Invalid arguments (missing value for port)
        valid, errors = loader.validate_arguments(["--port"])
        assert valid is False
        assert len(errors) > 0

    def test_configuration_precedence(self):
        """Test that CLI arguments override other sources."""
        loader = CLILoader()
        args = [
            "--debug",  # This should override any default/file/env setting
            "--port",
            "9000",
        ]

        config = loader.load(args)

        # CLI arguments should be present in config
        assert config["system"]["debug"] is True
        assert config["api"]["network"]["port"] == 9000

    def test_module_arguments(self):
        """Test adding module-specific arguments."""
        loader = CLILoader()
        module_schema = {
            "enabled": {"type": "bool", "description": "Enable the module"},
            "config_option": {
                "type": "str",
                "description": "Module configuration option",
            },
        }

        loader.add_module_arguments("test_module", module_schema)

        # Check that module arguments are added
        help_text = loader.get_help_text()
        assert "Test_Module Module" in help_text

    def test_error_handling(self):
        """Test error handling for invalid arguments."""
        loader = CLILoader()

        # Test invalid port (non-integer)
        with pytest.raises(SystemExit):
            loader.load(["--port", "invalid"])

    def test_special_arguments(self):
        """Test special arguments like --help-config and --version."""
        loader = CLILoader()

        # Test --version
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                loader.load(["--version"])
                mock_exit.assert_called_once_with(0)
                mock_print.assert_called()

    def test_environment_choices(self):
        """Test environment argument with choices."""
        loader = CLILoader()
        config = loader.load(["--env", "production"])

        assert config["environment"] == "production"

        # Test invalid environment
        with pytest.raises(SystemExit):
            loader.load(["--env", "invalid"])

    def test_complex_nested_arguments(self):
        """Test complex nested argument handling."""
        # Add a schema that supports these nested arguments
        schema = {
            "vision": {
                "camera": {
                    "device_id": {"type": "int", "description": "Camera device ID"}
                }
            },
            "api": {"network": {"host": {"type": "str", "description": "API host"}}},
            "system": {
                "logging": {"level": {"type": "str", "description": "Log level"}}
            },
        }

        loader = CLILoader(schema=schema)
        args = [
            "--vision-camera-device-id",
            "2",
            "--api-network-host",
            "0.0.0.0",
            "--system-logging-level",
            "WARNING",
        ]

        config = loader.load(args)

        assert config["vision"]["camera"]["device_id"] == 2
        assert config["api"]["network"]["host"] == "0.0.0.0"
        assert config["system"]["logging"]["level"] == "WARNING"

    def test_boolean_flags(self):
        """Test boolean flag handling."""
        loader = CLILoader()

        # Test setting flags
        config = loader.load(["--debug", "--enable-vision", "--enable-api"])

        assert config["system"]["debug"] is True
        assert config["modules"]["vision"]["enabled"] is True
        assert config["modules"]["api"]["enabled"] is True

        # Test without flags
        config = loader.load([])

        # Flags should not be present if not set
        assert "system" not in config or "debug" not in config.get("system", {})

    def test_list_argument_handling(self):
        """Test handling of list-type arguments."""
        loader = CLILoader()

        # Add a custom list argument
        loader.add_argument(
            "--modules", help="Comma-separated list of modules", dest="enabled_modules"
        )

        config = loader.load(["--modules", "vision,api,projector"])
        assert config["enabled_modules"] == ["vision", "api", "projector"]

    def test_json_argument_handling(self):
        """Test handling of JSON arguments."""
        loader = CLILoader()

        # Add a custom JSON argument
        loader.add_argument(
            "metadata",
            help="JSON metadata",
            config_key="custom_metadata",
            value_type="json",
        )

        json_value = '{"key": "value", "number": 42}'
        config = loader.load(["--metadata", json_value])
        assert config["custom_metadata"] == {"key": "value", "number": 42}

    def test_get_help_text(self):
        """Test help text generation."""
        loader = CLILoader()
        help_text = loader.get_help_text()

        assert "Professional Billiards Training System" in help_text
        assert "--debug" in help_text
        assert "--port" in help_text
        assert "Configuration Precedence" in help_text

    def test_config_validation(self):
        """Test configuration validation against schema."""
        schema = {"custom_port": {"type": "int", "minimum": 1, "maximum": 65535}}

        loader = CLILoader(schema=schema)

        # Valid port
        config = loader.load(["--custom-port", "8080"])
        assert config["custom_port"] == 8080

        # Invalid port (out of range) - this should be caught by schema validation
        with pytest.raises(CLIError):
            loader.load(["--custom-port", "999999"])

    def test_argument_error_handling(self):
        """Test handling of argument errors."""
        loader = CLILoader()

        # Test unknown argument
        with pytest.raises(SystemExit):
            loader.load(["--unknown-argument", "value"])

    def test_empty_schema_handling(self):
        """Test handling of empty or missing schema."""
        loader = CLILoader(schema={})

        # Should still work with built-in arguments
        config = loader.load(["--debug"])
        assert config["system"]["debug"] is True

    def test_schema_argument_conflicts(self):
        """Test handling of schema argument conflicts."""
        # Schema with argument that might conflict with built-ins
        schema = {"port": {"type": "int", "description": "Custom port"}}

        loader = CLILoader(schema=schema)

        # Should handle conflicts gracefully (built-in takes precedence)
        config = loader.load(["--port", "8080"])
        # Either the built-in api.network.port or schema port should work
        assert "port" in str(config) or "8080" in str(config)
