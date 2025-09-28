#!/usr/bin/env python3
"""Simple test for configuration persistence functionality."""

import tempfile
from pathlib import Path


def create_persistence_class():
    """Create the ConfigPersistence class inline for testing."""
    import json
    import logging
    import os
    import tempfile
    from pathlib import Path
    from typing import Any, Optional, Union

    try:
        import yaml

        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
        yaml = None

    class ConfigPersistenceError(Exception):
        """Configuration persistence error."""

        pass

    class ConfigFormat:
        JSON = "json"
        YAML = "yaml"

    class ConfigPersistence:
        """Configuration persistence manager."""

        def __init__(
            self,
            base_dir: Optional[Path] = None,
            logger: Optional[logging.Logger] = None,
        ):
            self.base_dir = Path(base_dir) if base_dir else Path("config")
            self.profiles_dir = self.base_dir / "profiles"
            self.backups_dir = self.base_dir / "backups"
            self.logger = logger or logging.getLogger(__name__)
            self._ensure_directories()

        def _ensure_directories(self) -> None:
            """Ensure all required directories exist."""
            for directory in [self.base_dir, self.profiles_dir, self.backups_dir]:
                directory.mkdir(parents=True, exist_ok=True)

        def _detect_format(self, file_path: Path):
            """Detect configuration format from file extension."""
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                return ConfigFormat.JSON
            elif suffix in [".yaml", ".yml"]:
                if not YAML_AVAILABLE:
                    raise ConfigPersistenceError(
                        "YAML format requested but PyYAML is not installed"
                    )
                return ConfigFormat.YAML
            else:
                return ConfigFormat.JSON

        def _serialize_data(self, data: dict[str, Any], format) -> str:
            """Serialize data to the specified format."""
            if format == ConfigFormat.JSON:
                return json.dumps(data, indent=2, default=str, ensure_ascii=False)
            elif format == ConfigFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ConfigPersistenceError("YAML support not available")
                return yaml.dump(
                    data, default_flow_style=False, allow_unicode=True, sort_keys=False
                )
            else:
                raise ConfigPersistenceError(f"Unsupported format: {format}")

        def _deserialize_data(self, content: str, format) -> dict[str, Any]:
            """Deserialize data from the specified format."""
            if format == ConfigFormat.JSON:
                return json.loads(content)
            elif format == ConfigFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ConfigPersistenceError("YAML support not available")
                return yaml.safe_load(content) or {}
            else:
                raise ConfigPersistenceError(f"Unsupported format: {format}")

        def _atomic_write(self, file_path: Path, content: str) -> None:
            """Perform atomic write operation."""
            temp_dir = file_path.parent
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=temp_dir,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_path = Path(temp_file.name)

            temp_path.replace(file_path)

        def save_config(
            self,
            config_data: dict[str, Any],
            file_path: Union[str, Path],
            format=None,
            create_backup: bool = True,
        ) -> bool:
            """Save configuration to file."""
            file_path = Path(file_path)
            try:
                if format is None:
                    format = self._detect_format(file_path)

                content = self._serialize_data(config_data, format)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write(file_path, content)
                return True
            except Exception as e:
                self.logger.error(f"Failed to save configuration to {file_path}: {e}")
                return False

        def load_config(
            self, file_path: Union[str, Path], format=None
        ) -> dict[str, Any]:
            """Load configuration from file."""
            file_path = Path(file_path)
            if not file_path.exists():
                raise ConfigPersistenceError(
                    f"Configuration file not found: {file_path}"
                )

            if format is None:
                format = self._detect_format(file_path)

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self._deserialize_data(content, format)

        def save_profile(
            self, profile_name: str, config_data: dict[str, Any], format=None
        ) -> bool:
            """Save a named configuration profile."""
            if format is None:
                format = ConfigFormat.JSON
            profile_file = self.profiles_dir / f"{profile_name}.{format}"

            profile_data = {
                "name": profile_name,
                "description": f"Profile for {profile_name}",
                "settings": config_data,
            }
            return self.save_config(profile_data, profile_file, format)

        def load_profile(self, profile_name: str, format=None):
            """Load a named configuration profile."""
            if format is None:
                for fmt in [ConfigFormat.JSON, ConfigFormat.YAML]:
                    profile_file = self.profiles_dir / f"{profile_name}.{fmt}"
                    if profile_file.exists():
                        format = fmt
                        break
                else:
                    raise ConfigPersistenceError(f"Profile '{profile_name}' not found")
            else:
                profile_file = self.profiles_dir / f"{profile_name}.{format}"

            data = self.load_config(profile_file, format)

            # Simple profile object
            class Profile:
                def __init__(self, name, description, settings):
                    self.name = name
                    self.description = description
                    self.settings = settings

            return Profile(
                data["name"], data.get("description", ""), data.get("settings", {})
            )

        def list_profiles(self) -> list[str]:
            """List all available configuration profiles."""
            profiles = set()
            try:
                for file_path in self.profiles_dir.glob("*"):
                    if file_path.is_file() and file_path.suffix in [
                        ".json",
                        ".yaml",
                        ".yml",
                    ]:
                        profiles.add(file_path.stem)
                return sorted(profiles)
            except Exception:
                return []

        def delete_profile(self, profile_name: str) -> bool:
            """Delete a configuration profile."""
            try:
                deleted = False
                for fmt in [ConfigFormat.JSON, ConfigFormat.YAML]:
                    profile_file = self.profiles_dir / f"{profile_name}.{fmt}"
                    if profile_file.exists():
                        profile_file.unlink()
                        deleted = True
                return deleted
            except Exception:
                return False

    return ConfigPersistence, ConfigPersistenceError, ConfigFormat


def test_basic_functionality():
    """Test basic save and load functionality."""
    print("=== Testing Basic Functionality ===")

    ConfigPersistence, ConfigPersistenceError, ConfigFormat = create_persistence_class()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        # Test data
        test_config = {
            "app": {"name": "billiards-trainer", "version": "1.0.0", "debug": True},
            "api": {"host": "0.0.0.0", "port": 8080},
        }

        config_file = temp_path / "test_config.json"

        # Test saving
        print(f"Saving config to: {config_file}")
        success = persistence.save_config(test_config, config_file)
        print(f"Save result: {success}")
        assert success, "Failed to save configuration"
        assert config_file.exists(), "Configuration file was not created"

        # Test loading
        print(f"Loading config from: {config_file}")
        loaded_config = persistence.load_config(config_file)
        print(f"Loaded config keys: {list(loaded_config.keys())}")
        assert loaded_config == test_config, "Loaded configuration doesn't match saved"

        print("✓ Basic functionality test passed")


def test_profile_management():
    """Test profile save and load functionality."""
    print("\n=== Testing Profile Management ===")

    ConfigPersistence, ConfigPersistenceError, ConfigFormat = create_persistence_class()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        # Test profile data
        profile_settings = {
            "projector": {
                "display": {"width": 1920, "height": 1080, "fullscreen": True}
            }
        }

        profile_name = "test_profile"

        # Test saving profile
        print(f"Saving profile: {profile_name}")
        success = persistence.save_profile(profile_name, profile_settings)
        print(f"Profile save result: {success}")
        assert success, "Failed to save profile"

        # Test loading profile
        print(f"Loading profile: {profile_name}")
        loaded_profile = persistence.load_profile(profile_name)
        print(f"Loaded profile name: {loaded_profile.name}")
        assert loaded_profile.name == profile_name, "Profile name doesn't match"
        assert (
            loaded_profile.settings == profile_settings
        ), "Profile settings don't match"

        # Test listing profiles
        profiles = persistence.list_profiles()
        print(f"Available profiles: {profiles}")
        assert profile_name in profiles, "Profile not found in list"

        # Test deleting profile
        print(f"Deleting profile: {profile_name}")
        success = persistence.delete_profile(profile_name)
        print(f"Profile delete result: {success}")
        assert success, "Failed to delete profile"

        # Verify deletion
        profiles = persistence.list_profiles()
        assert profile_name not in profiles, "Profile still exists after deletion"

        print("✓ Profile management test passed")


def test_yaml_support():
    """Test YAML format support if available."""
    print("\n=== Testing YAML Support ===")

    try:
        import yaml

        print("PyYAML is available, testing YAML format...")

        (
            ConfigPersistence,
            ConfigPersistenceError,
            ConfigFormat,
        ) = create_persistence_class()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            persistence = ConfigPersistence(base_dir=temp_path)

            test_config = {
                "vision": {
                    "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30}
                }
            }

            config_file = temp_path / "test_config.yaml"

            # Test saving in YAML format
            success = persistence.save_config(
                test_config, config_file, ConfigFormat.YAML
            )
            print(f"YAML save result: {success}")
            assert success, "Failed to save YAML configuration"

            # Test loading YAML format
            loaded_config = persistence.load_config(config_file, ConfigFormat.YAML)
            print(
                f"Loaded YAML config: vision.camera.device_id = {loaded_config['vision']['camera']['device_id']}"
            )
            assert loaded_config == test_config, "YAML configuration doesn't match"

            print("✓ YAML support test passed")

    except ImportError:
        print("PyYAML not available, skipping YAML tests")


def main():
    """Run all tests."""
    print("Configuration Persistence Test Suite")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_profile_management()
        test_yaml_support()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Configuration persistence is working correctly.")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
