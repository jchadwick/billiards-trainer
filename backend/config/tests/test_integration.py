"""Integration tests for configuration loading system.

Tests the complete system working together with multiple sources,
inheritance, and complex configuration scenarios.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from ..loader.env import EnvironmentLoader
from ..loader.file import FileLoader
from ..loader.merger import ConfigSource, ConfigurationMerger, MergeStrategy


class TestConfigurationSystemIntegration:
    """Integration tests for the complete configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_full_configuration_loading_pipeline(self):
        """Test the complete configuration loading pipeline."""
        # 1. Define default configuration
        default_config = {
            "app": {
                "name": "billiards-trainer",
                "version": "1.0.0",
                "debug": False,
                "features": {"vision": True, "projector": True, "ai_analysis": False},
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "billiards",
                "ssl": False,
                "pool_size": 10,
            },
            "vision": {
                "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30},
                "detection": {"sensitivity": 0.8, "min_confidence": 0.7},
            },
            "projector": {"brightness": 100, "contrast": 50, "color_temperature": 6500},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console"],
            },
        }

        # 2. Create file configuration (overrides some defaults)
        file_config = {
            "app": {"debug": True, "features": {"ai_analysis": True}},
            "database": {"host": "config-db-host", "ssl": True},
            "vision": {"camera": {"fps": 60}, "detection": {"sensitivity": 0.9}},
            "logging": {"level": "DEBUG", "handlers": ["console", "file"]},
        }

        # 3. Set up environment variables (highest precedence)
        env_vars = {
            "BILLIARDS_APP__DEBUG": "false",  # Override file config
            "BILLIARDS_DATABASE__HOST": "production-db",  # Override both
            "BILLIARDS_DATABASE__PORT": "3306",  # Override default
            "BILLIARDS_VISION__CAMERA__DEVICE_ID": "1",  # Override default
            "BILLIARDS_PROJECTOR__BRIGHTNESS": "80",  # Override default
            "BILLIARDS_LOGGING__LEVEL": "WARNING",  # Override file config
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        # 4. Create temporary configuration file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(file_config, f)
            config_file_path = f.name

        try:
            # 5. Load configurations using the system components
            file_loader = FileLoader(default_values=default_config)
            env_loader = EnvironmentLoader(prefix="BILLIARDS_")
            merger = ConfigurationMerger()

            # Load from file (with defaults merged)
            file_loaded_config = file_loader.load_with_defaults(config_file_path)

            # Load from environment
            env_config = env_loader.load_environment()

            # Merge all configurations with proper precedence
            configs = [file_loaded_config, env_config]
            sources = [ConfigSource.FILE, ConfigSource.ENVIRONMENT]

            final_config = merger.merge_configurations(configs, sources)

            # 6. Verify the final configuration
            # App configuration
            assert final_config["app"]["name"] == "billiards-trainer"  # from default
            assert final_config["app"]["version"] == "1.0.0"  # from default
            assert final_config["app"]["debug"] is False  # env overrides file
            assert final_config["app"]["features"]["vision"] is True  # from default
            assert final_config["app"]["features"]["projector"] is True  # from default
            assert final_config["app"]["features"]["ai_analysis"] is True  # from file

            # Database configuration
            assert final_config["database"]["host"] == "production-db"  # from env
            assert final_config["database"]["port"] == 3306  # from env
            assert final_config["database"]["name"] == "billiards"  # from default
            assert final_config["database"]["ssl"] is True  # from file
            assert final_config["database"]["pool_size"] == 10  # from default

            # Vision configuration
            assert final_config["vision"]["camera"]["device_id"] == 1  # from env
            assert final_config["vision"]["camera"]["resolution"] == [
                1920,
                1080,
            ]  # from default
            assert final_config["vision"]["camera"]["fps"] == 60  # from file
            assert (
                final_config["vision"]["detection"]["sensitivity"] == 0.9
            )  # from file
            assert (
                final_config["vision"]["detection"]["min_confidence"] == 0.7
            )  # from default

            # Projector configuration
            assert final_config["projector"]["brightness"] == 80  # from env
            assert final_config["projector"]["contrast"] == 50  # from default
            assert (
                final_config["projector"]["color_temperature"] == 6500
            )  # from default

            # Logging configuration
            assert final_config["logging"]["level"] == "WARNING"  # from env
            assert (
                final_config["logging"]["format"]
                == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )  # from default
            assert final_config["logging"]["handlers"] == [
                "console",
                "file",
            ]  # from file

        finally:
            os.unlink(config_file_path)

    def test_configuration_inheritance_with_multiple_files(self):
        """Test configuration inheritance across multiple files."""
        # Base configuration
        base_config = {
            "app": {"name": "billiards-trainer", "version": "1.0.0"},
            "database": {"host": "localhost", "port": 5432},
        }

        # Development configuration (inherits from base)
        dev_config = {
            "inherit": "base.json",
            "app": {"debug": True},
            "database": {"name": "billiards_dev"},
            "logging": {"level": "DEBUG"},
        }

        # Production configuration (inherits from base)
        prod_config = {
            "inherit": "base.json",
            "app": {"debug": False},
            "database": {"host": "prod-db-host", "name": "billiards_prod", "ssl": True},
            "logging": {"level": "WARNING"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration files
            base_path = Path(temp_dir) / "base.json"
            dev_path = Path(temp_dir) / "dev.json"
            prod_path = Path(temp_dir) / "prod.json"

            with open(base_path, "w") as f:
                json.dump(base_config, f)

            with open(dev_path, "w") as f:
                json.dump(dev_config, f)

            with open(prod_path, "w") as f:
                json.dump(prod_config, f)

            # Load configurations with inheritance
            file_loader = FileLoader()

            dev_result = file_loader.load_with_inheritance(dev_path)
            prod_result = file_loader.load_with_inheritance(prod_path)

            # Verify development configuration
            assert dev_result["app"]["name"] == "billiards-trainer"  # inherited
            assert dev_result["app"]["version"] == "1.0.0"  # inherited
            assert dev_result["app"]["debug"] is True  # overridden
            assert dev_result["database"]["host"] == "localhost"  # inherited
            assert dev_result["database"]["port"] == 5432  # inherited
            assert dev_result["database"]["name"] == "billiards_dev"  # overridden
            assert dev_result["logging"]["level"] == "DEBUG"  # added

            # Verify production configuration
            assert prod_result["app"]["name"] == "billiards-trainer"  # inherited
            assert prod_result["app"]["version"] == "1.0.0"  # inherited
            assert prod_result["app"]["debug"] is False  # overridden
            assert prod_result["database"]["host"] == "prod-db-host"  # overridden
            assert prod_result["database"]["port"] == 5432  # inherited
            assert prod_result["database"]["name"] == "billiards_prod"  # overridden
            assert prod_result["database"]["ssl"] is True  # added
            assert prod_result["logging"]["level"] == "WARNING"  # added

    def test_complex_merge_strategies(self):
        """Test complex merge strategies with different configuration sources."""
        # Configuration with lists and complex structures
        base_config = {
            "modules": ["core", "vision"],
            "features": {"enabled": ["basic_detection"], "experimental": []},
            "plugins": {
                "vision": {"enabled": True, "config": {"sensitivity": 0.8}},
                "projector": {"enabled": False},
            },
        }

        override_config = {
            "modules": ["projector", "ai"],
            "features": {
                "enabled": ["advanced_detection"],
                "experimental": ["neural_analysis"],
            },
            "plugins": {
                "vision": {"config": {"sensitivity": 0.9, "fps": 30}},
                "ai": {"enabled": True, "model": "yolo"},
            },
        }

        # Test different merge strategies
        merger = ConfigurationMerger()

        # 1. Test with override strategy (default for lists)
        result_override = merger.merge_two(
            base_config, override_config, MergeStrategy.OVERRIDE
        )
        assert result_override["modules"] == ["projector", "ai"]

        # 2. Test with list merge strategy

        # For this test, we'll use the base merge_two method which uses deep merge by default
        result_deep = merger.merge_two(
            base_config, override_config, MergeStrategy.MERGE_DEEP
        )

        # Check that deep merge worked for nested structures
        assert result_deep["plugins"]["vision"]["enabled"] is True  # from base
        assert (
            result_deep["plugins"]["vision"]["config"]["sensitivity"] == 0.9
        )  # from override
        assert result_deep["plugins"]["vision"]["config"]["fps"] == 30  # from override
        assert result_deep["plugins"]["projector"]["enabled"] is False  # from base
        assert result_deep["plugins"]["ai"]["enabled"] is True  # from override
        assert result_deep["plugins"]["ai"]["model"] == "yolo"  # from override

    def test_error_handling_and_validation(self):
        """Test error handling and validation in the configuration system."""
        # Test file loading errors
        file_loader = FileLoader()

        # Non-existent file should return empty config, not error
        result = file_loader.load_file("/totally/nonexistent/path.json")
        assert result == {}

        # Test environment loading with invalid types
        env_loader = EnvironmentLoader(prefix="TEST_")

        # Set invalid environment variable
        os.environ["TEST_INVALID_JSON"] = '{"incomplete": json'

        # Should handle invalid JSON gracefully
        result = env_loader.load_environment()
        # The value should be treated as a string since JSON parsing failed
        assert "invalid" in result
        assert isinstance(result["invalid"]["json"], str)

        # Test merger error handling
        merger = ConfigurationMerger()

        # Test with mismatched configs and sources
        with pytest.raises(Exception):  # Should be MergeError but testing the concept
            merger.merge_configurations(
                [{"key": "value1"}, {"key": "value2"}],
                [ConfigSource.FILE],  # Only one source for two configs
            )

    def test_real_world_billiards_configuration(self):
        """Test a realistic billiards trainer configuration scenario."""
        # This test simulates a real-world configuration for the billiards trainer

        # Default system configuration
        system_defaults = {
            "system": {
                "name": "Billiards Trainer",
                "version": "2.0.0",
                "mode": "training",
            },
            "hardware": {
                "camera": {
                    "count": 1,
                    "primary_device": 0,
                    "backup_device": 1,
                    "resolution": {"width": 1920, "height": 1080},
                    "fps": 30,
                    "exposure": "auto",
                },
                "projector": {
                    "enabled": True,
                    "brightness": 100,
                    "contrast": 50,
                    "keystone_correction": {"horizontal": 0, "vertical": 0},
                },
            },
            "vision": {
                "ball_detection": {
                    "algorithm": "opencv_contours",
                    "min_radius": 10,
                    "max_radius": 50,
                    "sensitivity": 0.8,
                },
                "table_detection": {
                    "algorithm": "edge_detection",
                    "corner_detection": True,
                    "pocket_detection": True,
                },
                "tracking": {"max_disappeared": 30, "max_distance": 100},
            },
            "analysis": {
                "shot_analysis": {
                    "enabled": True,
                    "trajectory_prediction": True,
                    "spin_detection": False,
                },
                "game_state": {"auto_detection": True, "manual_override": True},
            },
        }

        # Local configuration file (user customizations)
        local_config = {
            "hardware": {
                "camera": {
                    "fps": 60,  # Higher FPS for better tracking
                    "exposure": "manual",
                    "exposure_value": 100,
                },
                "projector": {
                    "brightness": 80,  # Dimmer for eye comfort
                    "keystone_correction": {"horizontal": -2, "vertical": 1},
                },
            },
            "vision": {
                "ball_detection": {
                    "sensitivity": 0.9,  # More sensitive detection
                    "algorithm": "neural_network",
                },
                "tracking": {"max_disappeared": 20},  # Faster re-acquisition
            },
            "analysis": {
                "shot_analysis": {"spin_detection": True}  # Enable advanced feature
            },
            "ui": {"theme": "dark", "show_trajectory": True, "show_guidelines": True},
        }

        # Environment-specific overrides (production settings)
        env_vars = {
            "BILLIARDS_SYSTEM__MODE": "competition",
            "BILLIARDS_HARDWARE__CAMERA__FPS": "120",  # Ultra-high FPS for competition
            "BILLIARDS_VISION__BALL_DETECTION__SENSITIVITY": "0.95",
            "BILLIARDS_ANALYSIS__SHOT_ANALYSIS__TRAJECTORY_PREDICTION": "true",
            "BILLIARDS_LOGGING__LEVEL": "INFO",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        # Create temporary local config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(local_config, f)
            local_config_path = f.name

        try:
            # Load complete configuration
            file_loader = FileLoader(default_values=system_defaults)
            env_loader = EnvironmentLoader(prefix="BILLIARDS_")
            merger = ConfigurationMerger()

            # Load configurations
            file_config = file_loader.load_with_defaults(local_config_path)
            env_config = env_loader.load_environment()

            # Merge with proper precedence: defaults < file < environment
            final_config = merger.merge_configurations(
                [file_config, env_config], [ConfigSource.FILE, ConfigSource.ENVIRONMENT]
            )

            # Verify final configuration has correct values from all sources

            # System settings
            assert final_config["system"]["name"] == "Billiards Trainer"  # default
            assert final_config["system"]["version"] == "2.0.0"  # default
            assert final_config["system"]["mode"] == "competition"  # environment

            # Hardware settings
            assert final_config["hardware"]["camera"]["count"] == 1  # default
            assert (
                final_config["hardware"]["camera"]["fps"] == 120
            )  # environment overrides file
            assert final_config["hardware"]["camera"]["exposure"] == "manual"  # file
            assert final_config["hardware"]["camera"]["exposure_value"] == 100  # file
            assert final_config["hardware"]["projector"]["brightness"] == 80  # file
            assert (
                final_config["hardware"]["projector"]["keystone_correction"][
                    "horizontal"
                ]
                == -2
            )  # file

            # Vision settings
            assert (
                final_config["vision"]["ball_detection"]["algorithm"]
                == "neural_network"
            )  # file
            assert (
                final_config["vision"]["ball_detection"]["sensitivity"] == 0.95
            )  # environment
            assert (
                final_config["vision"]["ball_detection"]["min_radius"] == 10
            )  # default
            assert final_config["vision"]["tracking"]["max_disappeared"] == 20  # file

            # Analysis settings
            assert (
                final_config["analysis"]["shot_analysis"]["spin_detection"] is True
            )  # file
            assert (
                final_config["analysis"]["shot_analysis"]["trajectory_prediction"]
                is True
            )  # environment

            # UI settings (only in file)
            assert final_config["ui"]["theme"] == "dark"
            assert final_config["ui"]["show_trajectory"] is True

        finally:
            os.unlink(local_config_path)
