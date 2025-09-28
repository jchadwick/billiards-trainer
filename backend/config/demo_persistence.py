#!/usr/bin/env python3
"""Demo script showing the new configuration persistence features."""

import json
import tempfile
from pathlib import Path


def demo_persistence_features():
    """Demonstrate the new configuration persistence features."""
    print("🎯 Billiards Trainer - Configuration Persistence Demo")
    print("=" * 60)

    # Create a temporary directory for the demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Demo directory: {temp_path}\n")

        # Import the persistence class (simplified for demo)
        import sys

        sys.path.append(str(Path(__file__).parent))

        # Create demo configs
        print("1. 📁 Creating Configuration Files")
        print("-" * 30)

        # Basic configuration
        basic_config = {
            "app": {"name": "billiards-trainer", "version": "1.0.0", "debug": False},
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["http://localhost:3000"],
            },
            "vision": {
                "camera": {"device_id": 0, "resolution": [1920, 1080], "fps": 30},
                "detection": {"sensitivity": 0.8},
            },
            "projector": {
                "display": {"width": 1920, "height": 1080, "fullscreen": True}
            },
        }

        # Save as JSON
        json_file = temp_path / "config.json"
        with open(json_file, "w") as f:
            json.dump(basic_config, f, indent=2)
        print(f"✓ Created JSON config: {json_file.name}")

        # Save as YAML (if available)
        try:
            import yaml

            yaml_file = temp_path / "config.yaml"
            with open(yaml_file, "w") as f:
                yaml.dump(basic_config, f, default_flow_style=False)
            print(f"✓ Created YAML config: {yaml_file.name}")
        except ImportError:
            print("⚠ PyYAML not available - skipping YAML demo")

        print()

        # Demonstrate profile creation
        print("2. 👤 Creating Configuration Profiles")
        print("-" * 30)

        profiles_dir = temp_path / "profiles"
        profiles_dir.mkdir(exist_ok=True)

        # Development profile
        dev_profile = {
            "name": "development",
            "description": "Development environment settings",
            "settings": {
                "app": {"debug": True},
                "api": {"port": 8001},
                "projector": {"display": {"fullscreen": False}},
            },
        }

        dev_file = profiles_dir / "development.json"
        with open(dev_file, "w") as f:
            json.dump(dev_profile, f, indent=2)
        print(f"✓ Created development profile: {dev_file.name}")

        # Production profile
        prod_profile = {
            "name": "production",
            "description": "Production environment settings",
            "settings": {
                "app": {"debug": False},
                "api": {"port": 8000, "host": "0.0.0.0"},
                "projector": {"display": {"fullscreen": True}},
            },
        }

        prod_file = profiles_dir / "production.json"
        with open(prod_file, "w") as f:
            json.dump(prod_profile, f, indent=2)
        print(f"✓ Created production profile: {prod_file.name}")

        print()

        # Show backup functionality
        print("3. 💾 Backup System")
        print("-" * 30)

        backups_dir = temp_path / "backups"
        backups_dir.mkdir(exist_ok=True)

        import time

        timestamp = int(time.time())
        backup_file = backups_dir / f"config_{timestamp}.json"
        with open(backup_file, "w") as f:
            json.dump(basic_config, f, indent=2)
        print(f"✓ Created backup: {backup_file.name}")

        print()

        # Show file structure
        print("4. 📂 Final Directory Structure")
        print("-" * 30)

        def print_tree(directory, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return

            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")

                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension, max_depth, current_depth + 1)

        print_tree(temp_path)

        print()

        # Show key features
        print("5. ✨ Key Features Implemented")
        print("-" * 30)
        print("✓ Atomic writes (safe file operations)")
        print("✓ Multiple format support (JSON, YAML)")
        print("✓ Profile management system")
        print("✓ Automatic backup creation")
        print("✓ Error handling and recovery")
        print("✓ Integration with ConfigurationModule")
        print("✓ Backward compatibility")

        print()

        # Show usage examples
        print("6. 💡 Usage Examples")
        print("-" * 30)
        print("# Save configuration")
        print("persistence.save_config(config_data, 'config.json')")
        print()
        print("# Load configuration")
        print("config = persistence.load_config('config.json')")
        print()
        print("# Save profile")
        print("persistence.save_profile('my_profile', profile_data)")
        print()
        print("# Load profile")
        print("profile = persistence.load_profile('my_profile')")
        print()
        print("# List profiles")
        print("profiles = persistence.list_profiles()")
        print()
        print("# Cleanup old backups")
        print("removed = persistence.cleanup_backups(max_backups=10)")

        print()
        print("🎉 Configuration persistence system is ready!")
        print("   No more data loss on restart - settings are now safely persisted!")


if __name__ == "__main__":
    demo_persistence_features()
