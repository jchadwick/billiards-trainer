"""Unit tests for the configuration module."""

from unittest.mock import patch

import pytest
import yaml
from config.loader.env import EnvironmentLoader
from config.loader.file import FileLoader
from config.manager import ConfigurationModule
from config.storage.persistence import ConfigPersistence
from config.validator.schema import SchemaValidator


@pytest.mark.unit()
class TestConfigurationModule:
    """Test the main configuration module."""

    def test_init(self):
        """Test configuration module initialization."""
        config = ConfigurationModule()
        assert config is not None
        assert hasattr(config, "data")
        assert hasattr(config, "validator")

    def test_load_from_file(self, temp_dir, mock_config):
        """Test loading configuration from file."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(mock_config, f)

        config = ConfigurationModule()
        config.load_config(config_file)

        assert config.get("camera.device_id") == 0
        assert config.get("table.width") == 2.84
        assert config.get("api.port") == 8001

    def test_load_from_dict(self, mock_config):
        """Test loading configuration from dictionary."""
        config = ConfigurationModule()
        config.load_from_dict(mock_config)

        assert config.get("camera.device_id") == 0
        assert config.get("table.width") == 2.84

    def test_get_nested_value(self, config_module):
        """Test getting nested configuration values."""
        assert config_module.get("camera.device_id") == 0
        assert config_module.get("camera.width") == 1920
        assert config_module.get("physics.friction") == 0.15

    def test_get_default_value(self, config_module):
        """Test getting values with default fallback."""
        assert config_module.get("nonexistent.key", "default") == "default"
        assert config_module.get("camera.nonexistent", 1080) == 1080

    def test_set_value(self, config_module):
        """Test setting configuration values."""
        config_module.set("camera.device_id", 1)
        assert config_module.get("camera.device_id") == 1

        config_module.set("new.nested.key", "value")
        assert config_module.get("new.nested.key") == "value"

    def test_validation_success(self, mock_config):
        """Test successful configuration validation."""
        config = ConfigurationModule()
        config.load_from_dict(mock_config)

        # Should not raise any validation errors
        assert config.validate()

    def test_validation_failure(self):
        """Test configuration validation failure."""
        invalid_config = {
            "camera": {
                "device_id": "invalid",  # Should be int
                "width": -100,  # Should be positive
            }
        }

        config = ConfigurationModule()
        config.load_from_dict(invalid_config)

        with pytest.raises(ValueError):
            config.validate(strict=True)

    def test_save_to_file(self, config_module, temp_dir):
        """Test saving configuration to file."""
        output_file = temp_dir / "output_config.yaml"
        config_module.save_to_file(str(output_file))

        assert output_file.exists()

        # Verify content
        with open(output_file) as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["camera"]["device_id"] == 0
        assert saved_config["table"]["width"] == 2.84


@pytest.mark.unit()
class TestSchemaValidator:
    """Test the schema validator."""

    def test_validate_camera_config(self):
        """Test camera configuration validation."""
        validator = SchemaValidator()

        valid_camera = {"device_id": 0, "width": 1920, "height": 1080, "fps": 30}

        # Should not raise
        validator.validate_camera(valid_camera)

    def test_validate_camera_config_invalid(self):
        """Test camera configuration validation with invalid data."""
        validator = SchemaValidator()

        invalid_camera = {"device_id": "not_an_int", "width": -100, "fps": 0}

        with pytest.raises(ValueError):
            validator.validate_camera(invalid_camera)

    def test_validate_table_config(self):
        """Test table configuration validation."""
        validator = SchemaValidator()

        valid_table = {"width": 2.84, "height": 1.42, "pocket_radius": 0.057}

        validator.validate_table(valid_table)

    def test_validate_physics_config(self):
        """Test physics configuration validation."""
        validator = SchemaValidator()

        valid_physics = {"friction": 0.15, "restitution": 0.9, "gravity": 9.81}

        validator.validate_physics(valid_physics)


@pytest.mark.unit()
class TestFileLoader:
    """Test the file loader."""

    def test_load_yaml_file(self, temp_dir):
        """Test loading YAML configuration file."""
        config_data = {"camera": {"device_id": 0}, "table": {"width": 2.84}}

        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = FileLoader()
        loaded_data = loader.load(str(config_file))

        assert loaded_data == config_data

    def test_load_json_file(self, temp_dir):
        """Test loading JSON configuration file."""
        import json

        config_data = {"camera": {"device_id": 0}, "table": {"width": 2.84}}

        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        loader = FileLoader()
        loaded_data = loader.load(str(config_file))

        assert loaded_data == config_data

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = FileLoader()

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.yaml")

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML file."""
        config_file = temp_dir / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")

        loader = FileLoader()

        with pytest.raises(yaml.YAMLError):
            loader.load(str(config_file))


@pytest.mark.unit()
class TestEnvironmentLoader:
    """Test the environment variable loader."""

    def test_load_env_vars(self):
        """Test loading environment variables."""
        with patch.dict(
            "os.environ",
            {
                "CAMERA_DEVICE_ID": "1",
                "API_PORT": "8002",
                "TABLE_WIDTH": "3.0",
            },
        ):
            loader = EnvironmentLoader()
            config = loader.load()

            assert config["camera"]["device_id"] == 1
            assert config["api"]["port"] == 8002
            assert config["table"]["width"] == 3.0

    def test_load_env_vars_no_prefix(self):
        """Test loading environment variables without prefix."""
        with patch.dict("os.environ", {"CAMERA_DEVICE_ID": "1", "API_PORT": "8002"}):
            loader = EnvironmentLoader()
            config = loader.load()

            assert config["camera"]["device_id"] == 1
            assert config["api"]["port"] == 8002

    def test_type_conversion(self):
        """Test automatic type conversion from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "CAMERA_DEVICE_ID": "1",  # int
                "TABLE_WIDTH": "2.84",  # float
                "API_DEBUG": "true",  # bool
                "API_CORS_ORIGINS": '["*"]',  # list
            },
        ):
            loader = EnvironmentLoader()
            config = loader.load()

            assert isinstance(config["camera"]["device_id"], int)
            assert isinstance(config["table"]["width"], float)
            assert isinstance(config["api"]["debug"], bool)
            assert isinstance(config["api"]["cors_origins"], list)


@pytest.mark.unit()
class TestPersistenceManager:
    """Test the persistence manager."""

    def test_save_and_load(self, temp_dir):
        """Test saving and loading configuration."""
        config_data = {"camera": {"device_id": 0}, "table": {"width": 2.84}}

        persistence = ConfigPersistence(base_dir=str(temp_dir))

        # Save configuration
        persistence.save("test_config", config_data)

        # Load configuration
        loaded_data = persistence.load("test_config")

        assert loaded_data == config_data

    def test_load_nonexistent_config(self, temp_dir):
        """Test loading non-existent configuration."""
        persistence = ConfigPersistence(base_dir=str(temp_dir))

        with pytest.raises(FileNotFoundError):
            persistence.load("nonexistent_config")

    def test_list_configurations(self, temp_dir):
        """Test listing saved configurations."""
        persistence = ConfigPersistence(base_dir=str(temp_dir))

        # Save multiple configurations
        persistence.save("config1", {"key": "value1"})
        persistence.save("config2", {"key": "value2"})

        configs = persistence.list_configurations()
        assert "config1" in configs
        assert "config2" in configs

    def test_delete_configuration(self, temp_dir):
        """Test deleting saved configuration."""
        persistence = ConfigPersistence(base_dir=str(temp_dir))

        # Save and delete configuration
        persistence.save("test_config", {"key": "value"})
        assert persistence.exists("test_config")

        persistence.delete("test_config")
        assert not persistence.exists("test_config")

    def test_backup_and_restore(self, temp_dir):
        """Test configuration backup and restore."""
        config_data = {"camera": {"device_id": 0}, "table": {"width": 2.84}}

        persistence = ConfigPersistence(base_dir=str(temp_dir))

        # Save original configuration
        persistence.save("main_config", config_data)

        # Create backup
        backup_name = persistence.create_backup("main_config")
        assert backup_name is not None

        # Modify original
        modified_data = config_data.copy()
        modified_data["camera"]["device_id"] = 1
        persistence.save("main_config", modified_data)

        # Restore from backup
        persistence.restore_backup("main_config", backup_name)

        # Verify restoration
        restored_data = persistence.load("main_config")
        assert restored_data["camera"]["device_id"] == 0


@pytest.mark.unit()
class TestConfigSchema:
    """Test the configuration schema."""

    def test_camera_schema(self):
        """Test camera configuration schema."""
        from config.models.schemas import CameraConfig

        camera_data = {
            "device_id": 0,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "exposure": -6,
        }

        camera_config = CameraConfig(**camera_data)
        assert camera_config.device_id == 0
        assert camera_config.width == 1920
        assert camera_config.fps == 30

    def test_table_schema(self):
        """Test table configuration schema."""
        from config.models.schemas import TableConfig

        table_data = {"width": 2.84, "height": 1.42, "pocket_radius": 0.057}

        table_config = TableConfig(**table_data)
        assert table_config.width == 2.84
        assert table_config.height == 1.42

    def test_invalid_schema(self):
        """Test invalid configuration schema."""
        from config.models.schemas import CameraConfig

        invalid_camera_data = {
            "device_id": "invalid",  # Should be int
            "width": -100,  # Should be positive
            "fps": 0,  # Should be positive
        }

        with pytest.raises(ValueError):
            CameraConfig(**invalid_camera_data)
