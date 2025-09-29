"""Type conversion utilities for configuration."""

import json
import logging
import re
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class TypeConversionError(Exception):
    """Exception raised when type conversion fails."""

    pass


class TypeConverter:
    """Type conversion utilities for configuration values."""

    def __init__(self):
        """Initialize the type converter with default converters."""
        self._converters: dict[Union[str, type], Callable[[Any], Any]] = {
            # String converters
            "str": str,
            "string": str,
            str: str,
            # Integer converters
            "int": self._convert_int,
            "integer": self._convert_int,
            int: self._convert_int,
            # Float converters
            "float": self._convert_float,
            "number": self._convert_float,
            float: self._convert_float,
            # Boolean converters
            "bool": self._convert_bool,
            "boolean": self._convert_bool,
            bool: self._convert_bool,
            # List converters
            "list": self._convert_list,
            "array": self._convert_list,
            list: self._convert_list,
            # Dict converters
            "dict": self._convert_dict,
            "object": self._convert_dict,
            dict: self._convert_dict,
            # JSON converter
            "json": self._convert_json,
        }

    def convert(self, value: Any, target_type: Union[str, type]) -> Any:
        """Convert value to target type.

        Args:
            value: Value to convert
            target_type: Target type (string name or type object)

        Returns:
            Converted value

        Raises:
            TypeConversionError: If conversion fails
        """
        if value is None:
            return None

        try:
            # If value is already the target type, return as-is
            if isinstance(target_type, type) and isinstance(value, target_type):
                return value

            # Get converter function
            converter = self._converters.get(target_type)
            if converter:
                return converter(value)

            # Fallback: try direct type conversion
            if isinstance(target_type, type):
                return target_type(value)

            # If target_type is a string, try to resolve it
            if isinstance(target_type, str):
                # Handle generic types like List[str], Dict[str, int], etc.
                if "[" in target_type and "]" in target_type:
                    return self._convert_generic_type(value, target_type)

                # Unknown string type
                raise TypeConversionError(f"Unknown type: {target_type}")

            raise TypeConversionError(f"Cannot convert to type: {target_type}")

        except Exception as e:
            if isinstance(e, TypeConversionError):
                raise
            raise TypeConversionError(
                f"Failed to convert {value} to {target_type}: {e}"
            )

    def auto_convert(self, value: str) -> Any:
        """Auto-detect type and convert.

        Args:
            value: String value to convert

        Returns:
            Converted value with auto-detected type
        """
        if not isinstance(value, str):
            return value

        value = value.strip()

        if not value:
            return value

        # Try JSON values (objects and arrays) first
        if value.startswith(("{", "[")):
            try:
                return self._convert_json(value)
            except (ValueError, TypeConversionError):
                pass

        # Try integer
        try:
            return self._convert_int(value)
        except (ValueError, TypeConversionError):
            pass

        # Try float
        try:
            return self._convert_float(value)
        except (ValueError, TypeConversionError):
            pass

        # Try boolean values last (since numbers can be interpreted as bool)
        try:
            return self._convert_bool(value)
        except (ValueError, TypeConversionError):
            pass

        # Try comma-separated list
        if "," in value:
            try:
                return self._convert_list(value)
            except (ValueError, TypeConversionError):
                pass

        # Default to string
        return value

    def _convert_int(self, value: Any) -> int:
        """Convert value to integer."""
        if isinstance(value, int):
            return value

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            value = value.strip()

            # Handle empty string
            if not value:
                raise TypeConversionError("Cannot convert empty string to int")

            # If it contains a decimal point, it's not an integer
            if "." in value:
                raise TypeConversionError(
                    f"Cannot convert '{value}' to int (contains decimal)"
                )

            # Handle binary, octal, hex
            if value.startswith("0b"):
                return int(value, 2)
            elif value.startswith("0o"):
                return int(value, 8)
            elif value.startswith("0x"):
                return int(value, 16)

            # Handle regular integers
            try:
                return int(value)
            except ValueError:
                raise TypeConversionError(f"Cannot convert '{value}' to int")

        raise TypeConversionError(f"Cannot convert {type(value).__name__} to int")

    def _convert_float(self, value: Any) -> float:
        """Convert value to float."""
        if isinstance(value, float):
            return value

        if isinstance(value, (int, bool)):
            return float(value)

        if isinstance(value, str):
            value = value.strip()

            # Handle empty string
            if not value:
                raise TypeConversionError("Cannot convert empty string to float")

            # Handle special float values
            lower_value = value.lower()
            if lower_value in ("inf", "infinity", "+inf", "+infinity"):
                return float("inf")
            elif lower_value in ("-inf", "-infinity"):
                return float("-inf")
            elif lower_value in ("nan", "none", "null"):
                return float("nan")

            try:
                return float(value)
            except ValueError:
                raise TypeConversionError(f"Cannot convert '{value}' to float")

        raise TypeConversionError(f"Cannot convert {type(value).__name__} to float")

    def _convert_bool(self, value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            value = value.strip().lower()

            if value in ("true", "yes", "on", "1", "enable", "enabled"):
                return True
            elif value in ("false", "no", "off", "0", "disable", "disabled", ""):
                return False
            else:
                raise TypeConversionError(f"Cannot convert '{value}' to bool")

        raise TypeConversionError(f"Cannot convert {type(value).__name__} to bool")

    def _convert_list(self, value: Any) -> list[Any]:
        """Convert value to list."""
        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, str):
            value = value.strip()

            # Handle empty string
            if not value:
                return []

            # Try JSON first
            if value.startswith("[") and value.endswith("]"):
                try:
                    return self._convert_json(value)
                except (ValueError, TypeConversionError):
                    pass

            # Split by comma and auto-convert each item
            items = [item.strip() for item in value.split(",")]
            return [self.auto_convert(item) for item in items if item]

        # Convert single value to list
        return [value]

    def _convert_dict(self, value: Any) -> dict[str, Any]:
        """Convert value to dictionary."""
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            value = value.strip()

            # Handle empty string
            if not value:
                return {}

            # Try JSON
            if value.startswith("{") and value.endswith("}"):
                try:
                    return self._convert_json(value)
                except (ValueError, TypeConversionError):
                    pass

            # Try key=value format
            if "=" in value:
                result = {}
                pairs = value.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, val = pair.split("=", 1)
                        result[key.strip()] = self.auto_convert(val.strip())
                return result

        raise TypeConversionError(f"Cannot convert {type(value).__name__} to dict")

    def _convert_json(self, value: Any) -> Any:
        """Convert JSON string to Python object."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise TypeConversionError(f"Invalid JSON: {e}")

        # If not a string, return as-is (already parsed)
        return value

    def _convert_generic_type(self, value: Any, type_spec: str) -> Any:
        """Convert value to generic type like List[str], Dict[str, int], etc.

        Args:
            value: Value to convert
            type_spec: Type specification string like "List[str]"

        Returns:
            Converted value
        """
        # Parse the generic type
        match = re.match(r"(\w+)\[(.+)\]", type_spec)
        if not match:
            raise TypeConversionError(f"Invalid generic type: {type_spec}")

        container_type = match.group(1).lower()
        inner_types = match.group(2)

        if container_type in ("list", "array"):
            # Convert to list and convert each element
            list_value = self._convert_list(value)
            inner_type = inner_types.strip()
            return [self.convert(item, inner_type) for item in list_value]

        elif container_type in ("dict", "object"):
            # Convert to dict and convert values
            dict_value = self._convert_dict(value)

            # Parse Dict[key_type, value_type]
            if "," in inner_types:
                key_type, value_type = (t.strip() for t in inner_types.split(",", 1))
                result = {}
                for k, v in dict_value.items():
                    converted_key = self.convert(k, key_type)
                    converted_value = self.convert(v, value_type)
                    result[converted_key] = converted_value
                return result
            else:
                # Only value type specified
                value_type = inner_types.strip()
                return {k: self.convert(v, value_type) for k, v in dict_value.items()}

        else:
            raise TypeConversionError(f"Unsupported generic type: {container_type}")

    def register_converter(
        self, type_name: Union[str, type], converter: Callable[[Any], Any]
    ) -> None:
        """Register a custom type converter.

        Args:
            type_name: Type name or type object
            converter: Converter function
        """
        self._converters[type_name] = converter
        logger.info(f"Registered custom converter for: {type_name}")

    def get_available_types(self) -> list[str]:
        """Get list of available type conversions.

        Returns:
            List of supported type names
        """
        return [str(t) for t in self._converters if isinstance(t, str)]

    def convert_batch(
        self, values: dict[str, Any], type_mapping: dict[str, Union[str, type]]
    ) -> dict[str, Any]:
        """Convert multiple values according to type mapping.

        Args:
            values: Dictionary of values to convert
            type_mapping: Dictionary mapping keys to target types

        Returns:
            Dictionary with converted values

        Raises:
            TypeConversionError: If any conversion fails
        """
        result = {}
        errors = []

        for key, value in values.items():
            if key in type_mapping:
                try:
                    result[key] = self.convert(value, type_mapping[key])
                except TypeConversionError as e:
                    errors.append(f"{key}: {e}")
            else:
                # No type mapping, try auto-conversion
                try:
                    result[key] = (
                        self.auto_convert(value) if isinstance(value, str) else value
                    )
                except Exception:
                    # If auto-conversion fails, keep original value
                    result[key] = value

        if errors:
            raise TypeConversionError(f"Batch conversion failed: {'; '.join(errors)}")

        return result

    def is_convertible(self, value: Any, target_type: Union[str, type]) -> bool:
        """Check if a value can be converted to target type.

        Args:
            value: Value to check
            target_type: Target type

        Returns:
            True if conversion is possible
        """
        try:
            self.convert(value, target_type)
            return True
        except (TypeConversionError, Exception):
            return False

    def safe_convert(
        self, value: Any, target_type: Union[str, type], default: Any = None
    ) -> Any:
        """Safely convert value, returning default if conversion fails.

        Args:
            value: Value to convert
            target_type: Target type
            default: Default value if conversion fails

        Returns:
            Converted value or default
        """
        try:
            return self.convert(value, target_type)
        except (TypeConversionError, Exception) as e:
            logger.debug(f"Safe conversion failed for {value} -> {target_type}: {e}")
            return default
