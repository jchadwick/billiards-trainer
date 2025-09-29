"""Validation rules for configuration."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation rule violations."""

    def __init__(self, field_name: str, message: str, value: Any = None):
        self.field_name = field_name
        self.message = message
        self.value = value
        super().__init__(f"{field_name}: {message}")


class ValidationRules:
    """Configuration validation rules with comprehensive type and range checking."""

    def __init__(self, strict_mode: bool = False, auto_correct: bool = False):
        """Initialize validation rules.

        Args:
            strict_mode: If True, raise exceptions on validation failures
            auto_correct: If True, attempt to auto-correct invalid values
        """
        self.strict_mode = strict_mode
        self.auto_correct = auto_correct
        self._validation_errors: list[str] = []

        # Mapping of string type names to Python types
        self._type_mapping = {
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
            "path": (str, Path),
            "none": type(None),
            "null": type(None),
        }

    def check_range(
        self,
        value: Any,
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None,
        field_name: str = "value",
    ) -> bool:
        """Check if numeric value is within specified range.

        Args:
            value: Value to check
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            field_name: Name of the field being validated

        Returns:
            True if value is within range, False otherwise

        Raises:
            ValidationError: If strict_mode is True and validation fails
        """
        try:
            # Convert value to numeric if possible
            if isinstance(value, str):
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    error_msg = f"Cannot convert '{value}' to numeric value"
                    self._handle_validation_error(field_name, error_msg, value)
                    return False

            if not isinstance(value, (int, float)):
                error_msg = f"Range validation requires numeric value, got {type(value).__name__}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            # Check minimum value
            if min_val is not None and value < min_val:
                if self.auto_correct:
                    logger.warning(
                        f"Auto-correcting {field_name}: {value} -> {min_val} (below minimum)"
                    )
                    return min_val
                error_msg = f"Value {value} below minimum {min_val}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            # Check maximum value
            if max_val is not None and value > max_val:
                if self.auto_correct:
                    logger.warning(
                        f"Auto-correcting {field_name}: {value} -> {max_val} (above maximum)"
                    )
                    return max_val
                error_msg = f"Value {value} above maximum {max_val}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            logger.debug(f"Range validation passed for {field_name}: {value}")
            return True

        except Exception as e:
            error_msg = f"Range validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def check_type(
        self,
        value: Any,
        expected_type: Union[type, str, tuple[type, ...]],
        field_name: str = "value",
    ) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected type (can be type, string name, or tuple of types)
            field_name: Name of the field being validated

        Returns:
            True if type matches, False otherwise

        Raises:
            ValidationError: If strict_mode is True and validation fails
        """
        try:
            # Handle string type specifications
            if isinstance(expected_type, str):
                expected_type = expected_type.lower()
                if expected_type in self._type_mapping:
                    expected_type = self._type_mapping[expected_type]
                else:
                    error_msg = f"Unknown type specification: {expected_type}"
                    self._handle_validation_error(field_name, error_msg, value)
                    return False

            # Check type match
            if isinstance(value, expected_type):
                logger.debug(
                    f"Type validation passed for {field_name}: {type(value).__name__}"
                )
                return True

            # Special handling for numeric conversions
            if expected_type in (int, float, (int, float)):
                if isinstance(value, str) and self.auto_correct:
                    try:
                        if expected_type == int:
                            corrected_value = int(float(value))  # Handle "1.0" -> 1
                            logger.warning(
                                f"Auto-correcting {field_name}: '{value}' -> {corrected_value}"
                            )
                            return corrected_value
                        elif expected_type == float:
                            corrected_value = float(value)
                            logger.warning(
                                f"Auto-correcting {field_name}: '{value}' -> {corrected_value}"
                            )
                            return corrected_value
                    except ValueError:
                        pass

            # Special handling for Path objects
            if expected_type in (Path, (str, Path)):
                if isinstance(value, str) and self.auto_correct:
                    logger.warning(f"Auto-correcting {field_name}: string -> Path")
                    return Path(value)

            # Special handling for boolean values
            if expected_type == bool and self.auto_correct:
                if isinstance(value, str):
                    lower_val = value.lower()
                    if lower_val in ("true", "1", "yes", "on", "enabled"):
                        logger.warning(
                            f"Auto-correcting {field_name}: '{value}' -> True"
                        )
                        return True
                    elif lower_val in ("false", "0", "no", "off", "disabled"):
                        logger.warning(
                            f"Auto-correcting {field_name}: '{value}' -> False"
                        )
                        return False
                elif isinstance(value, (int, float)):
                    corrected_value = bool(value)
                    logger.warning(
                        f"Auto-correcting {field_name}: {value} -> {corrected_value}"
                    )
                    return corrected_value

            # Type mismatch
            if isinstance(expected_type, tuple):
                expected_names = [t.__name__ for t in expected_type]
                error_msg = (
                    f"Expected one of {expected_names}, got {type(value).__name__}"
                )
            else:
                error_msg = (
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )

            self._handle_validation_error(field_name, error_msg, value)
            return False

        except Exception as e:
            error_msg = f"Type validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def validate_pattern(
        self, value: str, pattern: str, field_name: str = "value"
    ) -> bool:
        """Validate string against regex pattern.

        Args:
            value: String value to validate
            pattern: Regular expression pattern
            field_name: Name of the field being validated

        Returns:
            True if pattern matches, False otherwise
        """
        try:
            if not isinstance(value, str):
                error_msg = f"Pattern validation requires string value, got {type(value).__name__}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            if re.match(pattern, value):
                logger.debug(f"Pattern validation passed for {field_name}")
                return True
            else:
                error_msg = f"Value '{value}' does not match pattern '{pattern}'"
                self._handle_validation_error(field_name, error_msg, value)
                return False

        except re.error as e:
            error_msg = f"Invalid regex pattern '{pattern}': {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False
        except Exception as e:
            error_msg = f"Pattern validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def validate_enum(
        self, value: Any, allowed_values: list[Any], field_name: str = "value"
    ) -> bool:
        """Validate value is in allowed set.

        Args:
            value: Value to validate
            allowed_values: List of allowed values
            field_name: Name of the field being validated

        Returns:
            True if value is allowed, False otherwise
        """
        try:
            if value in allowed_values:
                logger.debug(f"Enum validation passed for {field_name}")
                return True
            else:
                error_msg = f"Value '{value}' not in allowed values: {allowed_values}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

        except Exception as e:
            error_msg = f"Enum validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def validate_length(
        self,
        value: Union[str, list, dict],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "value",
    ) -> bool:
        """Validate length of string, list, or dict.

        Args:
            value: Value to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            field_name: Name of the field being validated

        Returns:
            True if length is valid, False otherwise
        """
        try:
            if not hasattr(value, "__len__"):
                error_msg = f"Length validation requires object with length, got {type(value).__name__}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            length = len(value)

            if min_length is not None and length < min_length:
                error_msg = f"Length {length} below minimum {min_length}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            if max_length is not None and length > max_length:
                error_msg = f"Length {length} above maximum {max_length}"
                self._handle_validation_error(field_name, error_msg, value)
                return False

            logger.debug(f"Length validation passed for {field_name}: {length}")
            return True

        except Exception as e:
            error_msg = f"Length validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def validate_nested(
        self, value: dict, schema: dict[str, dict[str, Any]], field_name: str = "value"
    ) -> bool:
        """Validate nested dictionary against schema.

        Args:
            value: Dictionary to validate
            schema: Schema definition with field rules
            field_name: Name of the field being validated

        Returns:
            True if all nested validations pass, False otherwise
        """
        try:
            if not isinstance(value, dict):
                error_msg = (
                    f"Nested validation requires dictionary, got {type(value).__name__}"
                )
                self._handle_validation_error(field_name, error_msg, value)
                return False

            all_valid = True

            for key, rules in schema.items():
                nested_field_name = f"{field_name}.{key}"

                # Check if field is required
                if rules.get("required", False) and key not in value:
                    error_msg = "Required field missing"
                    self._handle_validation_error(nested_field_name, error_msg, None)
                    all_valid = False
                    continue

                if key not in value:
                    continue  # Optional field not present

                field_value = value[key]

                # Apply all rules to the field
                if "type" in rules:
                    if not self.check_type(
                        field_value, rules["type"], nested_field_name
                    ):
                        all_valid = False

                if "min" in rules or "max" in rules:
                    if not self.check_range(
                        field_value,
                        rules.get("min"),
                        rules.get("max"),
                        nested_field_name,
                    ):
                        all_valid = False

                if "pattern" in rules:
                    if not self.validate_pattern(
                        field_value, rules["pattern"], nested_field_name
                    ):
                        all_valid = False

                if "enum" in rules:
                    if not self.validate_enum(
                        field_value, rules["enum"], nested_field_name
                    ):
                        all_valid = False

                if "min_length" in rules or "max_length" in rules:
                    if not self.validate_length(
                        field_value,
                        rules.get("min_length"),
                        rules.get("max_length"),
                        nested_field_name,
                    ):
                        all_valid = False

            return all_valid

        except Exception as e:
            error_msg = f"Nested validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def validate_custom(
        self, value: Any, validator_func: callable, field_name: str = "value"
    ) -> bool:
        """Validate using custom validation function.

        Args:
            value: Value to validate
            validator_func: Custom validation function that returns bool or raises exception
            field_name: Name of the field being validated

        Returns:
            True if custom validation passes, False otherwise
        """
        try:
            result = validator_func(value)
            if result is True:
                logger.debug(f"Custom validation passed for {field_name}")
                return True
            else:
                error_msg = "Custom validation failed"
                self._handle_validation_error(field_name, error_msg, value)
                return False

        except Exception as e:
            error_msg = f"Custom validation error: {str(e)}"
            self._handle_validation_error(field_name, error_msg, value)
            return False

    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors from last validation run.

        Returns:
            List of error messages
        """
        return self._validation_errors.copy()

    def clear_validation_errors(self) -> None:
        """Clear validation error list."""
        self._validation_errors.clear()

    def _handle_validation_error(
        self, field_name: str, message: str, value: Any
    ) -> None:
        """Handle validation error according to configured mode.

        Args:
            field_name: Name of the field that failed validation
            message: Error message
            value: Value that failed validation
        """
        error_msg = f"{field_name}: {message}"
        self._validation_errors.append(error_msg)

        logger.warning(f"Validation error - {error_msg}")

        if self.strict_mode:
            raise ValidationError(field_name, message, value)
