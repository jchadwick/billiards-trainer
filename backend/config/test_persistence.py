#!/usr/bin/env python3
"""Test script for configuration persistence functionality."""

import json
import logging
import sys
import tempfile
from pathlib import Path

# Add the config directory to sys.path for proper imports
sys.path.insert(0, str(Path(__file__).parent))

from models.schemas import ConfigFormat
from storage.persistence import ConfigPersistence, ConfigPersistenceError


def test_basic_persistence():
    """Test basic save and load functionality."""
    print("=== Testing Basic Persistence ===")

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
        print(f"Loaded config: {loaded_config}")
        assert loaded_config == test_config, "Loaded configuration doesn't match saved"

        print("✓ Basic persistence test passed")


def test_yaml_support():
    """Test YAML format support (if available)."""
    print("\n=== Testing YAML Support ===")

    try:
        import yaml

        print("PyYAML is available, testing YAML format...")

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
            print(f"Loaded YAML config: {loaded_config}")
            assert loaded_config == test_config, "YAML configuration doesn't match"

            print("✓ YAML support test passed")

    except ImportError:
        print("PyYAML not available, skipping YAML tests")


def test_profile_management():
    """Test profile save and load functionality."""
    print("\n=== Testing Profile Management ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        # Test profile data
        profile_settings = {
            "projector": {
                "display": {"width": 1920, "height": 1080, "fullscreen": True}
            },
            "vision": {"detection": {"sensitivity": 0.9}},
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
        print(f"Loaded profile: {loaded_profile.name}")
        print(f"Profile settings: {loaded_profile.settings}")
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


def test_atomic_writes():
    """Test atomic write functionality."""
    print("\n=== Testing Atomic Writes ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        config_file = temp_path / "atomic_test.json"

        # Create initial config
        initial_config = {"test": "initial"}
        persistence.save_config(initial_config, config_file)

        # Verify file exists and has correct content
        assert config_file.exists(), "Initial config file not created"
        with open(config_file) as f:
            content = json.load(f)
        assert content == initial_config, "Initial content incorrect"

        # Update config
        updated_config = {"test": "updated", "new_field": 42}
        success = persistence.save_config(updated_config, config_file)

        assert success, "Failed to update configuration"

        # Verify update
        with open(config_file) as f:
            content = json.load(f)
        assert content == updated_config, "Updated content incorrect"

        print("✓ Atomic writes test passed")


def test_error_handling():
    """Test error handling scenarios."""
    print("\n=== Testing Error Handling ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        # Test loading non-existent file
        non_existent = temp_path / "does_not_exist.json"
        try:
            persistence.load_config(non_existent)
            raise AssertionError("Should have raised ConfigPersistenceError")
        except ConfigPersistenceError:
            print("✓ Correctly handled missing file")

        # Test loading non-existent profile
        try:
            persistence.load_profile("non_existent_profile")
            raise AssertionError("Should have raised ConfigPersistenceError")
        except ConfigPersistenceError:
            print("✓ Correctly handled missing profile")

        # Test saving to invalid path (permission error simulation)
        try:
            invalid_path = Path(
                "/root/cannot_write.json"
            )  # Assuming no write permission
            result = persistence.save_config({"test": "data"}, invalid_path)
            # This might succeed in some environments, so we just check that it doesn't crash
            print(f"Save to invalid path result: {result}")
        except Exception as e:
            print(f"✓ Handled permission error gracefully: {type(e).__name__}")

        print("✓ Error handling test passed")


def test_backup_functionality():
    """Test backup creation."""
    print("\n=== Testing Backup Functionality ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        config_file = temp_path / "backup_test.json"

        # Create initial config
        initial_config = {"version": 1}
        persistence.save_config(initial_config, config_file, create_backup=False)

        # Update config (should create backup)
        updated_config = {"version": 2}
        success = persistence.save_config(
            updated_config, config_file, create_backup=True
        )

        assert success, "Failed to save with backup"

        # Check if backup was created
        backups_dir = temp_path / "backups"
        backups = list(backups_dir.glob("backup_test_*.json"))
        print(f"Backups created: {len(backups)}")
        assert len(backups) >= 1, "No backup files created"

        # Test backup cleanup
        removed_count = persistence.cleanup_backups(max_backups=1)
        print(f"Backups removed during cleanup: {removed_count}")

        print("✓ Backup functionality test passed")


def test_format_detection():
    """Test automatic format detection."""
    print("\n=== Testing Format Detection ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        persistence = ConfigPersistence(base_dir=temp_path)

        test_config = {"format": "auto_detected"}

        # Test JSON format detection
        json_file = temp_path / "auto_detect.json"
        success = persistence.save_config(
            test_config, json_file
        )  # format=None for auto-detection
        assert success, "Failed to save with auto-detected JSON format"

        loaded = persistence.load_config(json_file)  # format=None for auto-detection
        assert loaded == test_config, "Auto-detected JSON load failed"

        # Test unknown extension fallback
        unknown_file = temp_path / "unknown.xyz"
        success = persistence.save_config(
            test_config, unknown_file
        )  # Should default to JSON
        assert success, "Failed to save with unknown extension"

        print("✓ Format detection test passed")


def main():
    """Run all tests."""
    print("Configuration Persistence Test Suite")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

    try:
        test_basic_persistence()
        test_yaml_support()
        test_profile_management()
        test_atomic_writes()
        test_error_handling()
        test_backup_functionality()
        test_format_detection()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Configuration persistence is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
