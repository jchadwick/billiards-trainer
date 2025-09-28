"""Example usage of the configuration loading system.

This demonstrates how to use the FileLoader, EnvironmentLoader, and
ConfigurationMerger classes to create a complete configuration system
that supports multiple sources with proper precedence.
"""

import json
import os
import tempfile
from pathlib import Path

from .loader.env import EnvironmentLoader
from .loader.file import FileLoader
from .loader.merger import ConfigSource, ConfigurationMerger


def main():
    """Demonstrate the configuration loading system."""
    print("=== Billiards Trainer Configuration Loading System Demo ===\n")

    # 1. Define default configuration
    print("1. Setting up default configuration...")
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
        },
        "vision": {
            "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30},
            "detection": {"sensitivity": 0.8},
        },
        "projector": {"brightness": 100, "contrast": 50},
        "logging": {"level": "INFO", "handlers": ["console"]},
    }
    print(f"✓ Default config loaded with {len(default_config)} top-level sections")

    # 2. Create a configuration file
    print("\n2. Creating configuration file...")
    file_config = {
        "app": {"debug": True, "features": {"ai_analysis": True}},
        "vision": {"camera": {"fps": 60}, "detection": {"sensitivity": 0.9}},
        "logging": {"level": "DEBUG", "handlers": ["console", "file"]},
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(file_config, f)
        config_file_path = f.name

    print(f"✓ Configuration file created at {config_file_path}")

    # 3. Set up environment variables
    print("\n3. Setting up environment variables...")
    env_vars = {
        "BILLIARDS_APP__DEBUG": "false",  # Override file config
        "BILLIARDS_DATABASE__HOST": "production-db",
        "BILLIARDS_DATABASE__PORT": "3306",
        "BILLIARDS_VISION__CAMERA__DEVICE_ID": "1",
        "BILLIARDS_PROJECTOR__BRIGHTNESS": "80",
        "BILLIARDS_LOGGING__LEVEL": "WARNING",
    }

    # Save original environment
    original_env = os.environ.copy()

    # Set test environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    print(f"✓ Set {len(env_vars)} environment variables with BILLIARDS_ prefix")

    try:
        # 4. Load configurations using the system
        print("\n4. Loading configurations...")

        # Initialize loaders
        file_loader = FileLoader(default_values=default_config)
        env_loader = EnvironmentLoader(prefix="BILLIARDS_")
        merger = ConfigurationMerger()

        # Load from file (with defaults merged)
        print("   Loading file configuration with defaults...")
        file_loaded_config = file_loader.load_with_defaults(config_file_path)

        # Load from environment
        print("   Loading environment variables...")
        env_config = env_loader.load_environment()

        print(
            f"   ✓ Environment config loaded with {len(env_config)} top-level sections"
        )

        # 5. Merge configurations with proper precedence
        print(
            "\n5. Merging configurations (precedence: defaults < file < environment)..."
        )

        configs = [file_loaded_config, env_config]
        sources = [ConfigSource.FILE, ConfigSource.ENVIRONMENT]

        final_config = merger.merge_configurations(configs, sources)

        print("✓ Final configuration merged successfully")

        # 6. Display the results
        print("\n6. Configuration Results:")
        print("=" * 60)

        print("\nApp Configuration:")
        print(f"  Name: {final_config['app']['name']}")
        print(f"  Version: {final_config['app']['version']}")
        print(f"  Debug: {final_config['app']['debug']} (from environment)")
        print("  Features:")
        for feature, enabled in final_config["app"]["features"].items():
            source = "file" if feature == "ai_analysis" else "default"
            print(f"    {feature}: {enabled} (from {source})")

        print("\nDatabase Configuration:")
        print(f"  Host: {final_config['database']['host']} (from environment)")
        print(f"  Port: {final_config['database']['port']} (from environment)")
        print(f"  Name: {final_config['database']['name']} (from default)")
        print(f"  SSL: {final_config['database']['ssl']} (from default)")

        print("\nVision Configuration:")
        print(
            f"  Camera Device ID: {final_config['vision']['camera']['device_id']} (from environment)"
        )
        print(
            f"  Camera Resolution: {final_config['vision']['camera']['resolution']} (from default)"
        )
        print(f"  Camera FPS: {final_config['vision']['camera']['fps']} (from file)")
        print(
            f"  Detection Sensitivity: {final_config['vision']['detection']['sensitivity']} (from file)"
        )

        print("\nProjector Configuration:")
        print(
            f"  Brightness: {final_config['projector']['brightness']} (from environment)"
        )
        print(f"  Contrast: {final_config['projector']['contrast']} (from default)")

        print("\nLogging Configuration:")
        print(f"  Level: {final_config['logging']['level']} (from environment)")
        print(f"  Handlers: {final_config['logging']['handlers']} (from file)")

        # 7. Demonstrate inheritance
        print("\n\n7. Demonstrating Configuration Inheritance:")
        print("=" * 60)

        # Create base and child configurations
        base_config = {
            "app": {"name": "billiards-trainer-base", "version": "1.0.0"},
            "common_settings": {"theme": "light", "auto_save": True},
        }

        child_config = {
            "inherit": "base.json",
            "app": {"name": "billiards-trainer-dev", "debug": True},
            "common_settings": {"theme": "dark"},  # Override parent
        }

        # Create temporary files for inheritance demo
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base.json"
            child_path = Path(temp_dir) / "child.json"

            with open(base_path, "w") as f:
                json.dump(base_config, f)

            with open(child_path, "w") as f:
                json.dump(child_config, f)

            # Load with inheritance
            inherited_config = file_loader.load_with_inheritance(child_path)

            print("Base configuration loaded")
            print(
                "Child configuration inherits from base and overrides specific values"
            )
            print("\nInherited Configuration Result:")
            print(
                f"  App Name: {inherited_config['app']['name']} (overridden in child)"
            )
            print(
                f"  App Version: {inherited_config['app']['version']} (inherited from base)"
            )
            print(f"  App Debug: {inherited_config['app']['debug']} (added in child)")
            print(
                f"  Theme: {inherited_config['common_settings']['theme']} (overridden in child)"
            )
            print(
                f"  Auto Save: {inherited_config['common_settings']['auto_save']} (inherited from base)"
            )

        print("\n\n=== Demo Complete ===")
        print("The configuration loading system successfully demonstrates:")
        print(
            "✓ FR-CFG-001: Load configuration from multiple sources (files, environment)"
        )
        print("✓ FR-CFG-002: Support configuration file formats (JSON)")
        print("✓ FR-CFG-003: Merge configurations with proper precedence rules")
        print("✓ FR-CFG-004: Provide default values for all settings")
        print("✓ FR-CFG-005: Support configuration inheritance and overrides")

    finally:
        # Clean up
        os.unlink(config_file_path)
        os.environ.clear()
        os.environ.update(original_env)


if __name__ == "__main__":
    main()
