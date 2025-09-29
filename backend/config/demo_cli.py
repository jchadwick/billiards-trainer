#!/usr/bin/env python3
"""Demonstration of the CLI configuration loader.

This script showcases the comprehensive CLI argument loading system
with type conversion, nested configuration paths, schema validation,
and integration with the existing configuration pipeline.
"""

import json
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader.cli import CLILoader
from config.loader.env import EnvironmentLoader
from config.loader.file import FileLoader
from config.loader.merger import ConfigurationMerger, ConfigSource


def demo_basic_usage():
    """Demonstrate basic CLI argument loading."""
    print("=== Basic CLI Usage Demo ===\n")

    loader = CLILoader()

    # Simulate command line arguments
    test_args = [
        "--debug",
        "--log-level", "DEBUG",
        "--port", "8080",
        "--camera", "1",
        "--resolution", "1920x1080"
    ]

    print(f"CLI arguments: {' '.join(test_args)}")

    config = loader.load(test_args)
    print("\nLoaded configuration:")
    print(json.dumps(config, indent=2))


def demo_schema_integration():
    """Demonstrate schema-based argument generation."""
    print("\n=== Schema Integration Demo ===\n")

    # Define configuration schema
    schema = {
        "vision": {
            "camera": {
                "device_id": {
                    "type": "int",
                    "default": 0,
                    "description": "Camera device ID"
                },
                "fps": {
                    "type": "int",
                    "default": 30,
                    "minimum": 15,
                    "maximum": 120,
                    "description": "Camera frames per second"
                }
            },
            "detection": {
                "enabled": {
                    "type": "bool",
                    "default": True,
                    "description": "Enable object detection"
                }
            }
        },
        "api": {
            "network": {
                "host": {
                    "type": "str",
                    "default": "localhost",
                    "description": "API server host"
                },
                "port": {
                    "type": "int",
                    "default": 8000,
                    "minimum": 1,
                    "maximum": 65535,
                    "description": "API server port"
                }
            }
        }
    }

    loader = CLILoader(schema=schema)

    # Test schema-generated arguments
    test_args = [
        "--vision-camera-device-id", "2",
        "--vision-camera-fps", "60",
        "--vision-detection-enabled",
        "--api-network-host", "0.0.0.0",
        "--api-network-port", "9000"
    ]

    print(f"CLI arguments: {' '.join(test_args)}")

    config = loader.load(test_args)
    print("\nLoaded configuration with schema:")
    print(json.dumps(config, indent=2))


def demo_type_conversions():
    """Demonstrate automatic type conversions."""
    print("\n=== Type Conversion Demo ===\n")

    loader = CLILoader()

    # Add custom arguments with different types
    loader.add_argument("--json-data", dest="metadata", help="JSON metadata")
    loader.add_argument("--list-data", dest="modules", help="Comma-separated modules")

    test_args = [
        "--resolution", "2560x1440",  # Tuple conversion
        "--json-data", '{"version": "1.0", "features": ["vision", "api"]}',  # JSON conversion
        "--list-data", "vision,core,projector,api",  # List conversion
        "--debug"  # Boolean flag
    ]

    print(f"CLI arguments: {' '.join(test_args)}")

    config = loader.load(test_args)
    print("\nType conversions:")
    print(f"Resolution (tuple): {config.get('vision', {}).get('camera', {}).get('resolution')}")
    print(f"JSON data (dict): {config.get('metadata')}")
    print(f"List data (list): {config.get('modules')}")
    print(f"Debug flag (bool): {config.get('system', {}).get('debug')}")


def demo_precedence_chain():
    """Demonstrate configuration precedence chain."""
    print("\n=== Configuration Precedence Demo ===\n")

    # Create a temporary config file
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        file_config = {
            "system": {
                "debug": False,
                "logging": {"level": "WARNING"}
            },
            "api": {
                "network": {"port": 8000, "host": "127.0.0.1"}
            }
        }
        json.dump(file_config, f)
        config_file = f.name

    try:
        # Set environment variables
        import os
        old_env = os.environ.copy()
        os.environ.update({
            "BILLIARDS_API__NETWORK__PORT": "8080",
            "BILLIARDS_SYSTEM__LOGGING__LEVEL": "INFO"
        })

        # Load from all sources
        file_loader = FileLoader()
        file_config_loaded = file_loader.load_file(Path(config_file))

        env_loader = EnvironmentLoader(prefix="BILLIARDS_")
        env_config = env_loader.load_environment()

        cli_loader = CLILoader()
        cli_config = cli_loader.load([
            "--debug",  # CLI override
            "--port", "9000"  # CLI override
        ])

        print("File configuration:")
        print(json.dumps(file_config_loaded, indent=2))

        print("\nEnvironment configuration:")
        print(json.dumps(env_config, indent=2))

        print("\nCLI configuration:")
        print(json.dumps(cli_config, indent=2))

        # Merge with proper precedence
        merger = ConfigurationMerger()
        merged_config = merger.merge_configurations(
            [file_config_loaded, env_config, cli_config],
            sources=[ConfigSource.FILE, ConfigSource.ENVIRONMENT, ConfigSource.CLI]
        )

        print("\nMerged configuration (CLI > Env > File):")
        print(json.dumps(merged_config, indent=2))

        print("\nPrecedence verification:")
        print(f"- Debug (CLI override): {merged_config['system']['debug']}")
        print(f"- Port (CLI override): {merged_config['api']['network']['port']}")
        print(f"- Log level (Env override): {merged_config['system']['logging']['level']}")
        print(f"- Host (File original): {merged_config['api']['network']['host']}")

    finally:
        Path(config_file).unlink()
        os.environ.clear()
        os.environ.update(old_env)


def demo_help_generation():
    """Demonstrate help text generation."""
    print("\n=== Help Generation Demo ===\n")

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
            },
            "ssl": {
                "type": "bool",
                "default": False,
                "description": "Enable SSL connection"
            }
        }
    }

    loader = CLILoader(schema=schema)

    print("Generated help text:")
    print(loader.get_help_text())


def demo_module_arguments():
    """Demonstrate module-specific arguments."""
    print("\n=== Module-Specific Arguments Demo ===\n")

    loader = CLILoader()

    # Add vision module arguments
    vision_schema = {
        "enabled": {"type": "bool", "description": "Enable vision module"},
        "camera": {
            "device_id": {"type": "int", "description": "Camera device ID"},
            "resolution": {"type": "str", "description": "Camera resolution"}
        },
        "detection": {
            "ball_sensitivity": {"type": "float", "description": "Ball detection sensitivity"}
        }
    }

    # Add API module arguments
    api_schema = {
        "enabled": {"type": "bool", "description": "Enable API module"},
        "network": {
            "port": {"type": "int", "description": "API server port"},
            "workers": {"type": "int", "description": "Number of worker processes"}
        }
    }

    loader.add_module_arguments("vision", vision_schema)
    loader.add_module_arguments("api", api_schema)

    test_args = [
        "--vision-enabled",
        "--vision-camera-device-id", "1",
        "--vision-camera-resolution", "1920x1080",
        "--vision-detection-ball-sensitivity", "0.85",
        "--api-enabled",
        "--api-network-port", "8080",
        "--api-network-workers", "4"
    ]

    print(f"CLI arguments: {' '.join(test_args)}")

    config = loader.load(test_args)
    print("\nModule-specific configuration:")
    print(json.dumps(config, indent=2))


def demo_error_handling():
    """Demonstrate error handling and validation."""
    print("\n=== Error Handling Demo ===\n")

    schema = {
        "port": {
            "type": "int",
            "minimum": 1,
            "maximum": 65535,
            "description": "Network port"
        }
    }

    loader = CLILoader(schema=schema)

    print("Testing validation errors:")

    # Test 1: Valid value
    try:
        config = loader.load(["--port", "8080"])
        print(f"✓ Valid port (8080): {config['port']}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Test 2: Invalid value (out of range)
    try:
        config = loader.load(["--port", "999999"])
        print(f"✗ Should have failed for port 999999")
    except Exception as e:
        print(f"✓ Caught validation error: {e}")

    # Test 3: Invalid argument
    try:
        config = loader.load(["--nonexistent-argument", "value"])
        print(f"✗ Should have failed for nonexistent argument")
    except SystemExit:
        print(f"✓ Caught argument error for nonexistent argument")


def main():
    """Run all demonstrations."""
    print("Billiards Trainer CLI Configuration Loader Demo")
    print("=" * 50)

    try:
        demo_basic_usage()
        demo_schema_integration()
        demo_type_conversions()
        demo_precedence_chain()
        demo_help_generation()
        demo_module_arguments()
        demo_error_handling()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nThe CLI loader provides:")
        print("✓ Comprehensive argument parsing with argparse")
        print("✓ Automatic type conversion based on context")
        print("✓ Schema-based argument generation")
        print("✓ Nested configuration path support")
        print("✓ Integration with existing configuration pipeline")
        print("✓ Proper precedence handling (CLI > Env > File > Default)")
        print("✓ Validation and error handling")
        print("✓ Module-specific argument grouping")
        print("✓ Comprehensive help text generation")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())