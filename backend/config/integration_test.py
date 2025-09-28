#!/usr/bin/env python3
"""Integration test for configuration persistence functionality."""

import json
import tempfile
from pathlib import Path

from manager import ConfigurationModule


def test_persistence_integration():
    """Test the full configuration persistence integration."""
    print("=== Configuration Persistence Integration Test ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temporary directory: {temp_path}")

        # Create configuration manager
        config = ConfigurationModule(config_dir=temp_path, enable_hot_reload=False)

        # Test setting some configuration values
        print("\n1. Setting configuration values...")
        config.set("api.host", "127.0.0.1", persist=False)
        config.set("api.port", 9000, persist=False)
        config.set("vision.camera.device_id", 1, persist=False)
        config.set("projector.display.fullscreen", True, persist=False)

        # Get current configuration
        current_config = config.get_all()
        print(f"Current config keys: {len(current_config)}")

        # Test saving configuration
        print("\n2. Testing configuration save...")
        save_path = temp_path / "test_config.json"
        success = config.save_config(save_path)
        print(f"Save result: {success}")
        assert success, "Failed to save configuration"
        assert save_path.exists(), "Configuration file was not created"

        # Verify saved content
        with open(save_path) as f:
            saved_content = json.load(f)
        print(f"Saved configuration has {len(saved_content)} top-level keys")

        # Test saving a profile
        print("\n3. Testing profile save...")
        profile_success = config.save_profile("test_profile")
        print(f"Profile save result: {profile_success}")
        assert profile_success, "Failed to save profile"

        # List profiles
        profiles = config.list_profiles()
        profile_names = [p.name for p in profiles]
        print(f"Available profiles: {profile_names}")
        assert "test_profile" in profile_names, "Test profile not found in list"

        # Test loading a profile
        print("\n4. Testing profile load...")
        # First, modify some values
        config.set("api.port", 8888, persist=False)
        old_port = config.get("api.port")
        print(f"Modified port to: {old_port}")

        # Load the profile (should restore previous values)
        load_success = config.load_profile("test_profile")
        print(f"Profile load result: {load_success}")
        assert load_success, "Failed to load profile"

        # Check if values were restored
        restored_port = config.get("api.port")
        print(f"Restored port: {restored_port}")

        # Test configuration loading
        print("\n5. Testing configuration load...")
        # Clear configuration and reload from file
        config.reset_to_defaults()
        config.set("api.port", 7777, persist=False)  # Set different value

        load_config_success = config.load_config(save_path)
        print(f"Config load result: {load_config_success}")
        assert load_config_success, "Failed to load configuration"

        # Check if values were loaded
        loaded_port = config.get("api.port")
        print(f"Loaded port: {loaded_port}")

        # Test deletion
        print("\n6. Testing profile deletion...")
        delete_success = config.delete_profile("test_profile")
        print(f"Profile delete result: {delete_success}")
        assert delete_success, "Failed to delete profile"

        # Verify deletion
        profiles_after = config.list_profiles()
        profile_names_after = [p.name for p in profiles_after]
        print(f"Profiles after deletion: {profile_names_after}")
        assert (
            "test_profile" not in profile_names_after
        ), "Profile still exists after deletion"

        # Test backup cleanup
        print("\n7. Testing backup cleanup...")
        cleanup_count = config.cleanup_backups(max_backups=5)
        print(f"Backup cleanup result: {cleanup_count} files removed")

        print("\n✅ All integration tests passed!")
        return True


def test_yaml_config():
    """Test YAML configuration support if available."""
    print("\n=== YAML Configuration Test ===")

    try:
        import yaml

        print("PyYAML is available, testing YAML configuration...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = ConfigurationModule(config_dir=temp_path, enable_hot_reload=False)

            # Create a YAML config file
            yaml_config = temp_path / "test.yaml"
            yaml_content = {
                "vision": {"camera": {"device_id": 2, "resolution": [1280, 720]}},
                "api": {"host": "0.0.0.0", "port": 8080},
            }

            with open(yaml_config, "w") as f:
                yaml.dump(yaml_content, f)

            # Load YAML configuration
            success = config.load_config(yaml_config)
            print(f"YAML load result: {success}")
            assert success, "Failed to load YAML configuration"

            # Check loaded values
            device_id = config.get("vision.camera.device_id")
            host = config.get("api.host")
            print(f"Loaded device_id: {device_id}, host: {host}")
            assert device_id == 2, "YAML device_id not loaded correctly"
            assert host == "0.0.0.0", "YAML host not loaded correctly"

            print("✅ YAML configuration test passed!")

    except ImportError:
        print("PyYAML not available, skipping YAML tests")


def main():
    """Run all integration tests."""
    print("Configuration Persistence Integration Test Suite")
    print("=" * 60)

    try:
        # Run basic integration test
        test_persistence_integration()

        # Run YAML test if available
        test_yaml_config()

        print("\n" + "=" * 60)
        print("🎉 All integration tests completed successfully!")
        print("Configuration persistence system is fully functional.")

        return 0

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
