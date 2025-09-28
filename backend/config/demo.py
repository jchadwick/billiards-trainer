#!/usr/bin/env python3
"""Demonstration of the ConfigurationModule usage for other modules.

This script shows how to:
1. Initialize the configuration module
2. Register module-specific configuration schemas
3. Get configuration values with type hints
4. Subscribe to configuration changes
5. Load configuration from files and environment
"""

import json
import os
import tempfile
from pathlib import Path

from .manager import ConfigurationModule


def demo_basic_usage():
    """Demonstrate basic configuration usage."""
    print("=== ConfigurationModule Basic Usage Demo ===\n")

    # Initialize configuration module
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ConfigurationModule(config_dir=Path(temp_dir) / "config")
        print("1. ✓ Configuration module initialized")

        # Get default values
        print("\n2. Default configuration values:")
        print(f"   App name: {config.get('app.name')}")
        print(f"   API port: {config.get('api.port')}")
        print(f"   Vision camera device: {config.get('vision.camera.device_id')}")
        print(f"   Vision sensitivity: {config.get('vision.detection.sensitivity')}")

        # Set custom values
        config.set("vision.camera.device_id", 1)
        config.set("vision.detection.sensitivity", 0.9)
        config.set("custom.feature.enabled", True)
        print("\n3. ✓ Set custom configuration values")

        # Get with type hints
        camera_id = config.get("vision.camera.device_id", type_hint=int)
        sensitivity = config.get("vision.detection.sensitivity", type_hint=float)
        print("\n4. Retrieved with type hints:")
        print(f"   Camera ID: {camera_id} (type: {type(camera_id).__name__})")
        print(f"   Sensitivity: {sensitivity} (type: {type(sensitivity).__name__})")

        return config


def demo_module_registration():
    """Demonstrate module configuration registration."""
    print("\n=== Module Registration Demo ===\n")

    config = ConfigurationModule()

    # Register a vision module configuration
    vision_spec = {
        "module_name": "vision",
        "configuration": {
            "camera.device_id": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 10,
                "description": "Camera device index",
            },
            "camera.resolution": {
                "type": "array",
                "default": [1920, 1080],
                "description": "Camera resolution [width, height]",
            },
            "detection.sensitivity": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Detection sensitivity threshold",
            },
        },
    }

    config.register_module("vision", vision_spec)
    print("1. ✓ Vision module registered with configuration schema")

    # Set some vision-specific values
    config.set("vision.camera.device_id", 2)
    config.set("vision.camera.resolution", [1280, 720])
    config.set("vision.detection.sensitivity", 0.75)

    # Get module-specific configuration
    vision_config = config.get_module_config("vision")
    print("\n2. Vision module configuration:")
    print(f"   Camera: {vision_config.get('camera', {})}")
    print(f"   Detection: {vision_config.get('detection', {})}")

    return config


def demo_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Demo ===\n")

    config = ConfigurationModule()

    # Register schema for a custom setting
    schema = {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Number of threads (1-100)",
    }
    config.register_schema("performance.threads", schema)
    print("1. ✓ Registered validation schema for performance.threads")

    # Test valid value
    config.set("performance.threads", 8)
    is_valid, errors = config.validate("performance.threads")
    print(f"\n2. Valid value (8): {is_valid}, errors: {errors}")

    # Test invalid value (this will fail validation during set)
    try:
        # Manually validate an invalid value
        is_valid, errors = config._validate_value("performance.threads", 150, schema)
        print(f"\n3. Invalid value (150): {is_valid}, errors: {errors}")
    except Exception as e:
        print(f"\n3. Validation error: {e}")

    return config


def demo_subscriptions():
    """Demonstrate configuration change subscriptions."""
    print("\n=== Configuration Change Subscriptions Demo ===\n")

    config = ConfigurationModule()
    changes_received = []

    def on_config_change(change):
        """Handle configuration changes."""
        changes_received.append(change)
        print(
            f"   📢 Config changed: {change.key} = {change.new_value} (was: {change.old_value})"
        )

    # Subscribe to changes
    sub_id = config.subscribe("api.*", on_config_change)
    print("1. ✓ Subscribed to API configuration changes")

    # Make some changes
    config.set("api.port", 9000)
    config.set("api.host", "0.0.0.0")
    config.set("other.setting", "ignored")  # This won't trigger the subscription

    print(f"\n2. Received {len(changes_received)} notifications for 'api.*' changes")

    # Unsubscribe
    config.unsubscribe(sub_id)
    config.set("api.port", 8080)  # This won't trigger notification
    print(f"\n3. After unsubscribing: {len(changes_received)} total notifications")

    return config


def demo_file_and_env_loading():
    """Demonstrate loading from files and environment."""
    print("\n=== File and Environment Loading Demo ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()

        # Create a configuration file
        config_data = {
            "app": {"name": "Demo Billiards Trainer", "debug": True},
            "api": {
                "port": 3000,
                "cors_origins": ["http://localhost:3000", "http://demo.local"],
            },
            "vision": {"camera": {"device_id": 1, "fps": 60}},
        }

        config_file = config_dir / "demo.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        print("1. ✓ Created demo configuration file")

        # Set environment variables
        env_vars = {
            "BILLIARDS_API_HOST": "demo.example.com",
            "BILLIARDS_VISION_DETECTION_SENSITIVITY": "0.95",
        }

        # Save original environment
        original_env = {}
        for key in env_vars:
            original_env[key] = os.environ.get(key)

        try:
            # Set environment variables
            for key, value in env_vars.items():
                os.environ[key] = value

            print("2. ✓ Set environment variables")

            # Initialize config and load file
            config = ConfigurationModule(config_dir=config_dir)
            config.load_config(config_file)
            config.load_environment_variables()

            print("\n3. Configuration values from different sources:")
            print(f"   App name (file): {config.get('app.name')}")
            print(f"   API port (file): {config.get('api.port')}")
            print(f"   API host (env): {config.get('api.host')}")
            print(
                f"   Vision sensitivity (env): {config.get('vision.detection.sensitivity')}"
            )
            print(f"   Camera device (file): {config.get('vision.camera.device_id')}")

            # Show all configuration
            print(f"\n4. Total configuration keys: {len(config.get_all())}")

        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return config


def main():
    """Run all configuration demos."""
    print("🎱 Billiards Trainer Configuration Module Demo\n")

    try:
        demo_basic_usage()
        demo_module_registration()
        demo_validation()
        demo_subscriptions()
        demo_file_and_env_loading()

        print("\n✅ All configuration demos completed successfully!")
        print("\n📋 Summary of Features Demonstrated:")
        print("   • Basic get/set operations with type hints")
        print("   • Module registration and configuration schemas")
        print("   • Configuration validation with detailed error messages")
        print("   • Change subscriptions and notifications")
        print("   • Loading from JSON files and environment variables")
        print("   • Hierarchical configuration with dot notation")
        print("   • Default values and automatic directory creation")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
