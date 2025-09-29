"""CLI argument configuration loader.

Supports loading configuration from command-line arguments with comprehensive
argument parsing, type conversion, nested structure support, and integration
with the existing configuration system.
Implements FR-CFG-001, FR-CFG-003, and FR-CFG-052.
"""

import argparse
import json
import logging
import sys
from enum import Enum
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class CLIError(Exception):
    """Exception raised for CLI argument errors."""

    pass


class TypeConversionError(CLIError):
    """Exception raised when type conversion fails."""

    pass


class ArgumentError(CLIError):
    """Exception raised for invalid arguments."""

    pass


class CLILoader:
    """CLI argument configuration loader.

    Supports:
    - Comprehensive argument parsing with argparse
    - Automatic type conversion based on configuration schema
    - Nested configuration paths using dot notation
    - Value overrides for any configuration option
    - Help text generation from schema descriptions
    - Integration with existing configuration pipeline
    - Validation and error handling
    """

    def __init__(
        self,
        prog_name: str = "billiards-trainer",
        description: str = "Professional Billiards Training System",
        prefix: str = "--",
        nested_separator: str = ".",
        type_converters: Optional[dict[str, Callable]] = None,
        schema: Optional[dict[str, Any]] = None,
    ):
        """Initialize the CLI loader.

        Args:
            prog_name: Program name for help text
            description: Program description for help text
            prefix: Argument prefix (default: '--')
            nested_separator: Separator for nested keys (default: '.')
            type_converters: Custom type conversion functions
            schema: Configuration schema for automatic argument generation
        """
        self.prog_name = prog_name
        self.description = description
        self.prefix = prefix
        self.nested_separator = nested_separator
        self.type_converters = type_converters or {}
        self.schema = schema or {}

        # Default type converters
        self._default_converters = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": self._convert_bool,
            "boolean": self._convert_bool,
            "json": self._convert_json,
            "list": self._convert_list,
            "array": self._convert_list,
        }

        # Create argument parser
        self.parser = argparse.ArgumentParser(
            prog=self.prog_name,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._generate_epilog(),
        )

        # Add built-in arguments
        self._add_builtin_arguments()

        # Generate schema-based arguments if schema provided
        if self.schema:
            self._add_schema_arguments()

    def load(self, args: Optional[list[str]] = None) -> dict[str, Any]:
        """Load configuration from CLI arguments.

        Args:
            args: List of command-line arguments (None to use sys.argv)

        Returns:
            Configuration dictionary with nested structure

        Raises:
            CLIError: If argument parsing or conversion fails
        """
        try:
            # Parse arguments
            if args is None:
                args = sys.argv[1:]

            parsed_args = self.parser.parse_args(args)

            # Handle special cases
            if hasattr(parsed_args, "help_config") and parsed_args.help_config:
                self._print_config_help()
                sys.exit(0)

            if hasattr(parsed_args, "version") and parsed_args.version:
                self._print_version()
                sys.exit(0)

            # Convert namespace to configuration dictionary
            config = self._namespace_to_config(parsed_args)

            # Apply type conversions
            config = self._apply_type_conversions(config)

            # Validate against schema if available
            if self.schema:
                self._validate_config(config)

            logger.info(f"Loaded {len(config)} configuration values from CLI")
            return config

        except SystemExit:
            # Re-raise SystemExit to allow normal help/error exits
            raise
        except Exception as e:
            raise CLIError(f"Failed to load CLI configuration: {e}")

    def load_with_schema(
        self,
        schema: dict[str, Any],
        args: Optional[list[str]] = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Load CLI arguments according to a specific schema.

        Args:
            schema: Configuration schema with types and defaults
            args: List of command-line arguments
            strict: Whether to raise errors for unknown arguments

        Returns:
            Configuration dictionary
        """
        # Update schema and regenerate arguments
        old_schema = self.schema
        self.schema = schema

        try:
            # Recreate parser with new schema
            self._recreate_parser()

            # Load with updated schema
            config = self.load(args)

            return config

        finally:
            # Restore original schema
            self.schema = old_schema
            self._recreate_parser()

    def add_argument(
        self,
        name: str,
        *args,
        config_key: Optional[str] = None,
        value_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a custom CLI argument.

        Args:
            name: Argument name (with or without prefix)
            config_key: Configuration key to map to (defaults to name)
            value_type: Type for conversion
            *args, **kwargs: Additional arguments for argparse
        """
        # Ensure name has prefix
        if not name.startswith(self.prefix):
            name = self.prefix + name.replace("_", "-")

        # Set default config key
        if config_key is None:
            config_key = name.lstrip(self.prefix).replace("-", "_")

        # Add type conversion info
        if value_type:
            kwargs["dest"] = config_key
            kwargs["metavar"] = value_type.upper()

        # Add argument to parser
        try:
            self.parser.add_argument(name, *args, **kwargs)
        except argparse.ArgumentError as e:
            logger.warning(f"Could not add argument {name}: {e}")

    def add_module_arguments(
        self, module_name: str, module_schema: dict[str, Any]
    ) -> None:
        """Add arguments for a specific module.

        Args:
            module_name: Name of the module
            module_schema: Schema for the module's configuration
        """
        group = self.parser.add_argument_group(
            f"{module_name.title()} Module",
            f"Configuration options for the {module_name} module",
        )

        self._add_schema_arguments_to_group(group, module_schema, module_name)

    def get_help_text(self) -> str:
        """Get formatted help text for all arguments.

        Returns:
            Formatted help text
        """
        return self.parser.format_help()

    def validate_arguments(self, args: list[str]) -> tuple[bool, list[str]]:
        """Validate arguments without loading configuration.

        Args:
            args: List of command-line arguments

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            self.parser.parse_args(args)
            return True, []
        except SystemExit:
            return False, ["Invalid arguments or help requested"]
        except Exception as e:
            return False, [str(e)]

    def _add_builtin_arguments(self) -> None:
        """Add built-in CLI arguments."""
        # Configuration file options
        config_group = self.parser.add_argument_group(
            "Configuration", "Configuration file and loading options"
        )

        config_group.add_argument(
            "--config",
            "-c",
            dest="config_file",
            metavar="FILE",
            help="Configuration file path",
        )

        config_group.add_argument(
            "--config-format",
            choices=["json", "yaml", "ini"],
            dest="config_format",
            help="Configuration file format (auto-detected if not specified)",
        )

        config_group.add_argument(
            "--profile",
            dest="active_profile",
            metavar="NAME",
            help="Configuration profile to use",
        )

        config_group.add_argument(
            "--no-config",
            action="store_true",
            dest="ignore_config_files",
            help="Ignore configuration files",
        )

        # Environment options
        env_group = self.parser.add_argument_group(
            "Environment", "Environment and runtime options"
        )

        env_group.add_argument(
            "--env",
            dest="environment",
            choices=["development", "testing", "production"],
            help="Runtime environment",
        )

        env_group.add_argument(
            "--debug",
            action="store_true",
            dest="system.debug",
            help="Enable debug mode",
        )

        env_group.add_argument(
            "--log-level",
            dest="system.logging.level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set logging level",
        )

        # Module enable/disable
        modules_group = self.parser.add_argument_group(
            "Modules", "Module enable/disable options"
        )

        modules_group.add_argument(
            "--enable-vision",
            action="store_true",
            dest="modules.vision.enabled",
            help="Enable vision module",
        )

        modules_group.add_argument(
            "--enable-projector",
            action="store_true",
            dest="modules.projector.enabled",
            help="Enable projector module",
        )

        modules_group.add_argument(
            "--enable-api",
            action="store_true",
            dest="modules.api.enabled",
            help="Enable API module",
        )

        # Network options
        network_group = self.parser.add_argument_group(
            "Network", "Network and connectivity options"
        )

        network_group.add_argument(
            "--host",
            dest="api.network.host",
            metavar="HOST",
            help="API server host address",
        )

        network_group.add_argument(
            "--port",
            "-p",
            dest="api.network.port",
            type=int,
            metavar="PORT",
            help="API server port",
        )

        # Hardware options
        hardware_group = self.parser.add_argument_group(
            "Hardware", "Hardware and device options"
        )

        hardware_group.add_argument(
            "--camera",
            dest="vision.camera.device_id",
            type=int,
            metavar="ID",
            help="Camera device ID",
        )

        hardware_group.add_argument(
            "--resolution",
            dest="vision.camera.resolution",
            metavar="WIDTHxHEIGHT",
            help="Camera resolution (e.g., 1920x1080)",
        )

        # Utility options
        util_group = self.parser.add_argument_group(
            "Utility", "Utility and information options"
        )

        util_group.add_argument(
            "--help-config",
            action="store_true",
            help="Show detailed configuration help",
        )

        util_group.add_argument(
            "--version", "-V", action="store_true", help="Show version information"
        )

        util_group.add_argument(
            "--validate-only",
            action="store_true",
            dest="validate_only",
            help="Validate configuration and exit",
        )

    def _add_schema_arguments(self) -> None:
        """Add arguments based on configuration schema."""
        if not self.schema:
            return

        # Process schema to generate arguments
        self._process_schema_section(self.schema, "")

    def _add_schema_arguments_to_group(
        self, group: argparse._ArgumentGroup, schema: dict[str, Any], prefix: str = ""
    ) -> None:
        """Add schema-based arguments to a specific group."""
        for key, spec in schema.items():
            if isinstance(spec, dict):
                if "type" in spec or "default" in spec:
                    # This is a field specification
                    self._add_single_argument_to_group(group, key, spec, prefix)
                else:
                    # This is a nested section
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    self._add_schema_arguments_to_group(group, spec, new_prefix)

    def _process_schema_section(self, schema: dict[str, Any], prefix: str) -> None:
        """Process a section of the schema to generate arguments."""
        for key, spec in schema.items():
            if isinstance(spec, dict):
                if "type" in spec or "default" in spec:
                    # This is a field specification
                    self._add_single_argument(key, spec, prefix)
                else:
                    # This is a nested section
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    self._process_schema_section(spec, new_prefix)

    def _add_single_argument(self, key: str, spec: dict[str, Any], prefix: str) -> None:
        """Add a single argument based on schema specification."""
        # Build argument name
        full_key = f"{prefix}.{key}" if prefix else key
        arg_name = self.prefix + full_key.replace(".", "-").replace("_", "-")

        # Check for conflicts with existing arguments
        existing_args = [action.option_strings for action in self.parser._actions]
        existing_args_flat = [arg for sublist in existing_args for arg in sublist]

        if arg_name in existing_args_flat:
            logger.debug(
                f"Skipping schema argument {arg_name}: conflicts with existing argument"
            )
            return

        # Get argument properties
        arg_type = spec.get("type", "str")
        default = spec.get("default")
        description = spec.get("description", f"Set {full_key}")
        choices = spec.get("enum")
        required = spec.get("required", False)

        # Build argument kwargs
        kwargs = {
            "dest": full_key,
            "help": description,
        }

        if default is not None:
            kwargs["default"] = argparse.SUPPRESS  # Don't override with None
            kwargs["help"] += f" (default: {default})"

        if choices:
            kwargs["choices"] = choices

        if required:
            kwargs["required"] = True

        # Handle type conversion
        if arg_type == "bool" or arg_type == "boolean":
            kwargs["action"] = "store_true"
        else:
            converter = self._get_type_converter(arg_type)
            if converter:
                kwargs["type"] = converter
            kwargs["metavar"] = arg_type.upper()

        # Add argument
        try:
            self.parser.add_argument(arg_name, **kwargs)
        except argparse.ArgumentError as e:
            logger.warning(f"Could not add argument {arg_name}: {e}")

    def _add_single_argument_to_group(
        self,
        group: argparse._ArgumentGroup,
        key: str,
        spec: dict[str, Any],
        prefix: str,
    ) -> None:
        """Add a single argument to a specific group."""
        # Similar to _add_single_argument but adds to group
        full_key = f"{prefix}.{key}" if prefix else key
        arg_name = self.prefix + full_key.replace(".", "-").replace("_", "-")

        arg_type = spec.get("type", "str")
        default = spec.get("default")
        description = spec.get("description", f"Set {full_key}")
        choices = spec.get("enum")

        kwargs = {
            "dest": full_key,
            "help": description,
        }

        if default is not None:
            kwargs["default"] = argparse.SUPPRESS
            kwargs["help"] += f" (default: {default})"

        if choices:
            kwargs["choices"] = choices

        if arg_type == "bool" or arg_type == "boolean":
            kwargs["action"] = "store_true"
        else:
            converter = self._get_type_converter(arg_type)
            if converter:
                kwargs["type"] = converter
            kwargs["metavar"] = arg_type.upper()

        try:
            group.add_argument(arg_name, **kwargs)
        except argparse.ArgumentError as e:
            logger.warning(f"Could not add argument {arg_name} to group: {e}")

    def _namespace_to_config(self, namespace: argparse.Namespace) -> dict[str, Any]:
        """Convert argparse namespace to nested configuration dictionary."""
        config = {}

        # Track which arguments were actually provided
        # We need to check if the argument was explicitly set
        for action in self.parser._actions:
            dest = action.dest
            if dest == "help":
                continue

            if hasattr(namespace, dest):
                value = getattr(namespace, dest)

                # For boolean actions, only include if True (explicitly set)
                if isinstance(action, argparse._StoreTrueAction):
                    if value is True:
                        self._set_nested_value(config, dest, value)
                # For other types, include if not None and not the default
                elif value is not None and value != action.default:
                    self._set_nested_value(config, dest, value)

        return config

    def _apply_type_conversions(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply type conversions to configuration values."""

        def convert_recursive(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                return {
                    k: convert_recursive(v, f"{path}.{k}" if path else k)
                    for k, v in obj.items()
                }
            elif isinstance(obj, str):
                # Try to convert special string values
                return self._convert_string_value(obj, path)
            else:
                return obj

        return convert_recursive(config)

    def _convert_string_value(self, value: str, path: str) -> Any:
        """Convert string value based on context and patterns."""
        # Handle resolution format (WIDTHxHEIGHT)
        if "resolution" in path:
            if "x" in value:
                try:
                    parts = value.split("x")
                    if len(parts) != 2:
                        raise TypeConversionError(f"Invalid resolution format: {value}")
                    width, height = parts
                    return (int(width), int(height))
                except (ValueError, TypeError):
                    raise TypeConversionError(f"Invalid resolution format: {value}")
            else:
                # If it's meant to be a resolution but doesn't have 'x', it's invalid
                raise TypeConversionError(f"Invalid resolution format: {value}")

        # Handle JSON strings
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # Handle comma-separated lists
        if "," in value and not value.startswith("{"):
            return [item.strip() for item in value.split(",")]

        return value

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration against schema."""
        errors = []

        def validate_recursive(
            obj: dict[str, Any], schema: dict[str, Any], path: str = ""
        ) -> None:
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if key in schema:
                    spec = schema[key]
                    if isinstance(spec, dict) and "type" in spec:
                        # Validate type
                        expected_type = spec["type"]
                        if not self._check_type(value, expected_type):
                            errors.append(
                                f"{current_path}: Expected {expected_type}, got {type(value).__name__}"
                            )

                        # Validate range for numbers
                        if isinstance(value, (int, float)):
                            minimum = spec.get("minimum")
                            maximum = spec.get("maximum")

                            if minimum is not None and value < minimum:
                                errors.append(
                                    f"{current_path}: Value {value} below minimum {minimum}"
                                )

                            if maximum is not None and value > maximum:
                                errors.append(
                                    f"{current_path}: Value {value} above maximum {maximum}"
                                )

                    elif isinstance(spec, dict) and isinstance(value, dict):
                        # Recurse into nested objects
                        validate_recursive(value, spec, current_path)

        validate_recursive(config, self.schema)

        if errors:
            raise ArgumentError(f"Configuration validation failed: {'; '.join(errors)}")

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True  # Unknown type, skip validation

    def _get_type_converter(self, type_name: str) -> Optional[Callable]:
        """Get type converter function for a type name."""
        return self.type_converters.get(type_name) or self._default_converters.get(
            type_name
        )

    def _convert_bool(self, value: str) -> bool:
        """Convert string to boolean."""
        value = value.lower().strip()
        if value in ("true", "yes", "on", "1", "y"):
            return True
        elif value in ("false", "no", "off", "0", "n"):
            return False
        else:
            raise ValueError(f"Cannot convert '{value}' to boolean")

    def _convert_json(self, value: str) -> Any:
        """Convert JSON string to Python object."""
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def _convert_list(self, value: str) -> list[str]:
        """Convert comma-separated string to list."""
        if not value:
            return []
        return [item.strip() for item in value.split(",")]

    def _set_nested_value(self, config: dict[str, Any], key: str, value: Any) -> None:
        """Set a nested value in the configuration dictionary."""
        if self.nested_separator in key:
            keys = key.split(self.nested_separator)
            current = config

            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                elif not isinstance(current[k], dict):
                    # Convert non-dict values to dict if needed
                    current[k] = {}
                current = current[k]

            # Set final value
            current[keys[-1]] = value
        else:
            config[key] = value

    def _recreate_parser(self) -> None:
        """Recreate the argument parser with current settings."""
        self.parser = argparse.ArgumentParser(
            prog=self.prog_name,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._generate_epilog(),
        )

        self._add_builtin_arguments()

        if self.schema:
            self._add_schema_arguments()

    def _generate_epilog(self) -> str:
        """Generate epilog text for help."""
        return """
Configuration Precedence (highest to lowest):
  1. Command-line arguments (--option value)
  2. Environment variables (BILLIARDS_OPTION)
  3. Configuration files (--config file.json)
  4. Default values

Examples:
  # Use specific config file
  %(prog)s --config production.json

  # Override specific settings
  %(prog)s --debug --log-level DEBUG

  # Set nested configuration
  %(prog)s --vision-camera-device-id 1 --api-network-port 8080

  # Disable modules
  %(prog)s --no-vision --no-projector

  # Use specific profile
  %(prog)s --profile development
"""

    def _print_config_help(self) -> None:
        """Print detailed configuration help."""
        print("\nBilliards Trainer Configuration Help")
        print("===================================\n")

        print("Configuration can be provided through multiple sources:")
        print("1. Configuration files (JSON, YAML, INI)")
        print("2. Environment variables (with BILLIARDS_ prefix)")
        print("3. Command-line arguments (shown below)")
        print("4. Default values\n")

        if self.schema:
            print("Available configuration sections:")
            self._print_schema_help(self.schema)

        print("\nFor more details, use --help")

    def _print_schema_help(
        self, schema: dict[str, Any], prefix: str = "", indent: int = 0
    ) -> None:
        """Print help for schema sections."""
        for key, spec in schema.items():
            current_prefix = f"{prefix}.{key}" if prefix else key

            if isinstance(spec, dict):
                if "type" in spec or "description" in spec:
                    # This is a field
                    description = spec.get("description", "No description")
                    field_type = spec.get("type", "unknown")
                    default = spec.get("default")

                    print("  " * indent + f"--{current_prefix.replace('.', '-')}")
                    print("  " * (indent + 1) + f"Type: {field_type}")
                    print("  " * (indent + 1) + f"Description: {description}")
                    if default is not None:
                        print("  " * (indent + 1) + f"Default: {default}")
                    print()
                else:
                    # This is a section
                    print("  " * indent + f"[{current_prefix.upper()}]")
                    self._print_schema_help(spec, current_prefix, indent + 1)

    def _print_version(self) -> None:
        """Print version information."""
        print(f"{self.prog_name} Configuration Loader")
        print("Part of the Professional Billiards Training System")
        print("Version: 1.0.0")
