"""Integration tests for CLI loader with configuration system."""

import json
import tempfile
from pathlib import Path

import pytest

from ..loader.cli import CLILoader
from ..loader.env import EnvironmentLoader
from ..loader.file import FileLoader
from ..loader.merger import ConfigurationMerger, ConfigSource


class TestCLIIntegration:
    """Test CLI loader integration with configuration system."""

    def test_cli_precedence_over_file(self):
        """Test that CLI arguments override file configuration."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "system": {
                    "debug": False,
                    "logging": {
                        "level": "INFO"
                    }
                },
                "api": {
                    "network": {
                        "port": 8000
                    }
                }
            }
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Load from file
            file_loader = FileLoader()
            file_config = file_loader.load_file(Path(config_file))

            # Load from CLI (should override file)
            cli_loader = CLILoader()
            cli_config = cli_loader.load([
                "--debug",
                "--log-level", "DEBUG",
                "--port", "9000"
            ])

            # Merge with proper precedence
            merger = ConfigurationMerger()
            merged_config = merger.merge_configurations(
                [file_config, cli_config],
                sources=[ConfigSource.FILE, ConfigSource.CLI]
            )

            # CLI should override file
            assert merged_config["system"]["debug"] is True  # CLI override
            assert merged_config["system"]["logging"]["level"] == "DEBUG"  # CLI override
            assert merged_config["api"]["network"]["port"] == 9000  # CLI override

        finally:
            Path(config_file).unlink()

    def test_cli_precedence_over_environment(self):
        """Test that CLI arguments override environment variables."""
        import os

        # Set environment variables
        old_env = os.environ.copy()
        try:
            os.environ.update({
                "BILLIARDS_SYSTEM__DEBUG": "false",
                "BILLIARDS_API__NETWORK__PORT": "8000",
                "BILLIARDS_SYSTEM__LOGGING__LEVEL": "INFO"
            })

            # Load from environment
            env_loader = EnvironmentLoader(prefix="BILLIARDS_")
            env_config = env_loader.load_environment()

            # Load from CLI (should override environment)
            cli_loader = CLILoader()
            cli_config = cli_loader.load([
                "--debug",
                "--port", "9000"
            ])

            # Merge with proper precedence
            merger = ConfigurationMerger()
            merged_config = merger.merge_configurations(
                [env_config, cli_config],
                sources=[ConfigSource.ENVIRONMENT, ConfigSource.CLI]
            )

            # CLI should override environment
            assert merged_config["system"]["debug"] is True  # CLI override
            assert merged_config["api"]["network"]["port"] == 9000  # CLI override
            # Environment should be preserved where CLI doesn't override
            assert merged_config["system"]["logging"]["level"] == "INFO"  # From env

        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_full_precedence_chain(self):
        """Test full precedence chain: CLI > Environment > File > Default."""
        import os

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_config_data = {
                "system": {
                    "debug": False,
                    "logging": {
                        "level": "WARNING"
                    }
                },
                "api": {
                    "network": {
                        "port": 8000,
                        "host": "127.0.0.1"
                    }
                },
                "vision": {
                    "camera": {
                        "device_id": 0
                    }
                }
            }
            json.dump(file_config_data, f)
            config_file = f.name

        old_env = os.environ.copy()
        try:
            # Set environment variables (should override file)
            os.environ.update({
                "BILLIARDS_API__NETWORK__PORT": "8080",
                "BILLIARDS_VISION__CAMERA__DEVICE_ID": "1"
            })

            # Load from all sources
            file_loader = FileLoader()
            file_config = file_loader.load_file(Path(config_file))

            env_loader = EnvironmentLoader(prefix="BILLIARDS_")
            env_config = env_loader.load_environment()

            cli_loader = CLILoader()
            cli_config = cli_loader.load([
                "--debug",  # CLI override
                "--camera", "2"  # CLI override
            ])

            # Default config
            default_config = {
                "system": {
                    "timezone": "UTC"
                },
                "api": {
                    "network": {
                        "workers": 1
                    }
                }
            }

            # Merge with proper precedence: CLI > Env > File > Default
            merger = ConfigurationMerger()
            merged_config = merger.merge_configurations(
                [default_config, file_config, env_config, cli_config],
                sources=[ConfigSource.DEFAULT, ConfigSource.FILE, ConfigSource.ENVIRONMENT, ConfigSource.CLI]
            )

            # Verify precedence
            assert merged_config["system"]["debug"] is True  # CLI
            assert merged_config["system"]["logging"]["level"] == "WARNING"  # File
            assert merged_config["api"]["network"]["port"] == 8080  # Environment
            assert merged_config["api"]["network"]["host"] == "127.0.0.1"  # File
            assert merged_config["api"]["network"]["workers"] == 1  # Default
            assert merged_config["vision"]["camera"]["device_id"] == 2  # CLI
            assert merged_config["system"]["timezone"] == "UTC"  # Default

        finally:
            Path(config_file).unlink()
            os.environ.clear()
            os.environ.update(old_env)

    def test_schema_integration(self):
        """Test CLI loader integration with schema validation."""
        from ..models.schemas import SystemConfig, VisionConfig

        # Create schema from Pydantic models
        schema = {
            "system": {
                "debug": {"type": "bool", "default": False},
                "logging": {
                    "level": {"type": "str", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                }
            },
            "vision": {
                "camera": {
                    "device_id": {"type": "int", "minimum": 0, "maximum": 10},
                    "fps": {"type": "int", "minimum": 15, "maximum": 120}
                }
            }
        }

        cli_loader = CLILoader(schema=schema)

        # Valid configuration
        config = cli_loader.load([
            "--system-debug",
            "--system-logging-level", "DEBUG",
            "--vision-camera-device-id", "1",
            "--vision-camera-fps", "30"
        ])

        assert config["system"]["debug"] is True
        assert config["system"]["logging"]["level"] == "DEBUG"
        assert config["vision"]["camera"]["device_id"] == 1
        assert config["vision"]["camera"]["fps"] == 30

    def test_config_profiles_with_cli(self):
        """Test CLI arguments with configuration profiles."""
        # Create temporary profile files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Base config
            base_config = {
                "system": {"debug": False},
                "api": {"network": {"port": 8000}}
            }
            base_file = Path(temp_dir) / "base.json"
            with open(base_file, 'w') as f:
                json.dump(base_config, f)

            # Development profile
            dev_config = {
                "system": {"debug": True},
                "api": {"network": {"port": 8080}}
            }
            dev_file = Path(temp_dir) / "development.json"
            with open(dev_file, 'w') as f:
                json.dump(dev_config, f)

            # Load base config
            file_loader = FileLoader()
            base_config_loaded = file_loader.load_file(base_file)

            # Simulate profile selection via CLI
            cli_loader = CLILoader()
            cli_config = cli_loader.load([
                "--profile", "development",
                "--port", "9000"  # CLI should override profile
            ])

            # Merge: Base < Profile < CLI
            merger = ConfigurationMerger()

            # First merge base with CLI
            merged_config = merger.merge_configurations(
                [base_config_loaded, cli_config],
                sources=[ConfigSource.FILE, ConfigSource.CLI]
            )

            # CLI should override everything
            assert merged_config["api"]["network"]["port"] == 9000

    def test_nested_configuration_paths(self):
        """Test deep nested configuration paths via CLI."""
        schema = {
            "projector": {
                "display": {
                    "resolution": {
                        "width": {"type": "int"},
                        "height": {"type": "int"}
                    },
                    "calibration": {
                        "keystone": {
                            "horizontal": {"type": "float"},
                            "vertical": {"type": "float"}
                        }
                    }
                }
            }
        }

        cli_loader = CLILoader(schema=schema)
        config = cli_loader.load([
            "--projector-display-resolution-width", "1920",
            "--projector-display-resolution-height", "1080",
            "--projector-display-calibration-keystone-horizontal", "0.1",
            "--projector-display-calibration-keystone-vertical", "-0.05"
        ])

        assert config["projector"]["display"]["resolution"]["width"] == 1920
        assert config["projector"]["display"]["resolution"]["height"] == 1080
        assert config["projector"]["display"]["calibration"]["keystone"]["horizontal"] == 0.1
        assert config["projector"]["display"]["calibration"]["keystone"]["vertical"] == -0.05

    def test_complex_type_conversions(self):
        """Test complex type conversions in integrated environment."""
        cli_loader = CLILoader()

        # Test resolution conversion
        config = cli_loader.load(["--resolution", "1920x1080"])
        assert config["vision"]["camera"]["resolution"] == (1920, 1080)

        # Add JSON argument
        cli_loader.add_argument(
            "--metadata",
            dest="custom_metadata",
            help="JSON metadata"
        )

        # Test JSON conversion
        json_data = '{"cameras": [{"id": 0, "name": "main"}], "enabled": true}'
        config = cli_loader.load(["--metadata", json_data])
        expected = {"cameras": [{"id": 0, "name": "main"}], "enabled": True}
        assert config["custom_metadata"] == expected

        # Add list argument
        cli_loader.add_argument(
            "--modules",
            dest="enabled_modules",
            help="Enabled modules list"
        )

        # Test list conversion
        config = cli_loader.load(["--modules", "vision,api,projector"])
        assert config["enabled_modules"] == ["vision", "api", "projector"]

    def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        cli_loader = CLILoader()

        # Test invalid argument
        with pytest.raises(SystemExit):
            cli_loader.load(["--invalid-argument", "value"])

        # Test argument validation with schema
        schema = {
            "test_field": {
                "type": "int",
                "minimum": 1,
                "maximum": 100
            }
        }

        cli_loader = CLILoader(schema=schema)

        # Test out-of-range value (caught by schema validation)
        from ..loader.cli import CLIError
        with pytest.raises(CLIError):
            cli_loader.load(["--test-field", "999"])

    def test_help_and_documentation(self):
        """Test help text generation with schema integration."""
        schema = {
            "database": {
                "host": {
                    "type": "str",
                    "default": "localhost",
                    "description": "Database host address"
                },
                "port": {
                    "type": "int",
                    "default": 5432,
                    "description": "Database port number"
                }
            }
        }

        cli_loader = CLILoader(schema=schema)
        help_text = cli_loader.get_help_text()

        # Check that schema-generated arguments appear in help
        assert "--database-host" in help_text
        assert "--database-port" in help_text
        assert "Database host address" in help_text
        assert "Database port number" in help_text

        # Check precedence information
        assert "Configuration Precedence" in help_text
        assert "Command-line arguments" in help_text

    def test_module_specific_configuration(self):
        """Test module-specific CLI arguments."""
        cli_loader = CLILoader()

        # Add vision module arguments
        vision_schema = {
            "enabled": {"type": "bool", "description": "Enable vision module"},
            "camera": {
                "device_id": {"type": "int", "description": "Camera device ID"},
                "resolution": {"type": "str", "description": "Camera resolution"}
            }
        }

        api_schema = {
            "enabled": {"type": "bool", "description": "Enable API module"},
            "network": {
                "port": {"type": "int", "description": "API server port"},
                "host": {"type": "str", "description": "API server host"}
            }
        }

        cli_loader.add_module_arguments("vision", vision_schema)
        cli_loader.add_module_arguments("api", api_schema)

        config = cli_loader.load([
            "--vision-enabled",
            "--vision-camera-device-id", "1",
            "--api-network-port", "8080"
        ])

        assert config["vision"]["enabled"] is True
        assert config["vision"]["camera"]["device_id"] == 1
        assert config["api"]["network"]["port"] == 8080

        # Check help includes module sections
        help_text = cli_loader.get_help_text()
        assert "Vision Module" in help_text
        assert "Api Module" in help_text