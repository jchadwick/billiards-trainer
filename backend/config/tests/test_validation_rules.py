"""Tests for validation rules."""

from pathlib import Path

import pytest

from backend.config.validator.rules import ValidationError, ValidationRules


class TestValidationRules:
    """Test cases for ValidationRules class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ValidationRules()
        self.strict_validator = ValidationRules(strict_mode=True)
        self.auto_correct_validator = ValidationRules(auto_correct=True)

    def test_init(self):
        """Test ValidationRules initialization."""
        assert not self.validator.strict_mode
        assert not self.validator.auto_correct
        assert self.validator._validation_errors == []

    def test_check_range_valid_int(self):
        """Test range validation with valid integers."""
        assert self.validator.check_range(5, 0, 10, "test_field")
        assert self.validator.check_range(0, 0, 10, "test_field")
        assert self.validator.check_range(10, 0, 10, "test_field")

    def test_check_range_valid_float(self):
        """Test range validation with valid floats."""
        assert self.validator.check_range(5.5, 0.0, 10.0, "test_field")
        assert self.validator.check_range(0.0, 0.0, 10.0, "test_field")
        assert self.validator.check_range(10.0, 0.0, 10.0, "test_field")

    def test_check_range_invalid_below_min(self):
        """Test range validation with value below minimum."""
        assert not self.validator.check_range(-1, 0, 10, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "below minimum" in errors[0]

    def test_check_range_invalid_above_max(self):
        """Test range validation with value above maximum."""
        assert not self.validator.check_range(11, 0, 10, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "above maximum" in errors[0]

    def test_check_range_auto_correct(self):
        """Test range validation with auto-correction."""
        # Below minimum
        result = self.auto_correct_validator.check_range(-1, 0, 10, "test_field")
        assert result == 0  # Auto-corrected to minimum

        # Above maximum
        result = self.auto_correct_validator.check_range(11, 0, 10, "test_field")
        assert result == 10  # Auto-corrected to maximum

    def test_check_range_strict_mode(self):
        """Test range validation in strict mode."""
        with pytest.raises(ValidationError) as exc_info:
            self.strict_validator.check_range(-1, 0, 10, "test_field")
        assert "test_field" in str(exc_info.value)
        assert "below minimum" in str(exc_info.value)

    def test_check_range_string_numeric_conversion(self):
        """Test range validation with string numeric values."""
        assert self.validator.check_range("5", 0, 10, "test_field")
        assert self.validator.check_range("5.5", 0.0, 10.0, "test_field")

    def test_check_range_invalid_string(self):
        """Test range validation with invalid string."""
        assert not self.validator.check_range("not_a_number", 0, 10, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "Cannot convert" in errors[0]

    def test_check_range_non_numeric(self):
        """Test range validation with non-numeric values."""
        assert not self.validator.check_range([], 0, 10, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "requires numeric value" in errors[0]

    def test_check_type_valid_basic_types(self):
        """Test type validation with valid basic types."""
        assert self.validator.check_type("hello", str, "test_field")
        assert self.validator.check_type(42, int, "test_field")
        assert self.validator.check_type(3.14, float, "test_field")
        assert self.validator.check_type(True, bool, "test_field")
        assert self.validator.check_type([], list, "test_field")
        assert self.validator.check_type({}, dict, "test_field")

    def test_check_type_string_specifications(self):
        """Test type validation with string type specifications."""
        assert self.validator.check_type("hello", "str", "test_field")
        assert self.validator.check_type("hello", "string", "test_field")
        assert self.validator.check_type(42, "int", "test_field")
        assert self.validator.check_type(42, "integer", "test_field")
        assert self.validator.check_type(3.14, "float", "test_field")
        assert self.validator.check_type(42, "number", "test_field")  # int/float tuple
        assert self.validator.check_type(3.14, "number", "test_field")

    def test_check_type_invalid_basic_types(self):
        """Test type validation with invalid basic types."""
        assert not self.validator.check_type("hello", int, "test_field")
        assert not self.validator.check_type(42, str, "test_field")
        assert not self.validator.check_type([], dict, "test_field")

    def test_check_type_auto_correction(self):
        """Test type validation with auto-correction."""
        # String to int
        result = self.auto_correct_validator.check_type("42", int, "test_field")
        assert result == 42

        # String to float
        result = self.auto_correct_validator.check_type("3.14", float, "test_field")
        assert result == 3.14

        # String to bool
        result = self.auto_correct_validator.check_type("true", bool, "test_field")
        assert result is True

        result = self.auto_correct_validator.check_type("false", bool, "test_field")
        assert result is False

        # String to Path
        result = self.auto_correct_validator.check_type(
            "/path/to/file", Path, "test_field"
        )
        assert result == Path("/path/to/file")

    def test_check_type_tuple_of_types(self):
        """Test type validation with tuple of possible types."""
        type_spec = (int, float)
        assert self.validator.check_type(42, type_spec, "test_field")
        assert self.validator.check_type(3.14, type_spec, "test_field")
        assert not self.validator.check_type("hello", type_spec, "test_field")

    def test_check_type_unknown_string_spec(self):
        """Test type validation with unknown string specification."""
        assert not self.validator.check_type("hello", "unknown_type", "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "Unknown type specification" in errors[0]

    def test_validate_pattern_valid(self):
        """Test pattern validation with valid patterns."""
        assert self.validator.validate_pattern("hello123", r"^hello\d+$", "test_field")
        assert self.validator.validate_pattern(
            "test@example.com", r"^[\w\.-]+@[\w\.-]+\.\w+$", "test_field"
        )

    def test_validate_pattern_invalid(self):
        """Test pattern validation with invalid patterns."""
        assert not self.validator.validate_pattern("hello", r"^\d+$", "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "does not match pattern" in errors[0]

    def test_validate_pattern_non_string(self):
        """Test pattern validation with non-string value."""
        assert not self.validator.validate_pattern(123, r"^\d+$", "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "requires string value" in errors[0]

    def test_validate_pattern_invalid_regex(self):
        """Test pattern validation with invalid regex."""
        assert not self.validator.validate_pattern("test", r"[", "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "Invalid regex pattern" in errors[0]

    def test_validate_enum_valid(self):
        """Test enum validation with valid values."""
        allowed_values = ["red", "green", "blue"]
        assert self.validator.validate_enum("red", allowed_values, "test_field")
        assert self.validator.validate_enum("green", allowed_values, "test_field")

    def test_validate_enum_invalid(self):
        """Test enum validation with invalid values."""
        allowed_values = ["red", "green", "blue"]
        assert not self.validator.validate_enum("yellow", allowed_values, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "not in allowed values" in errors[0]

    def test_validate_length_string(self):
        """Test length validation with strings."""
        assert self.validator.validate_length("hello", 3, 10, "test_field")
        assert self.validator.validate_length("hi", 1, 5, "test_field")
        assert not self.validator.validate_length("h", 2, 5, "test_field")
        assert not self.validator.validate_length("toolong", 1, 5, "test_field")

    def test_validate_length_list(self):
        """Test length validation with lists."""
        assert self.validator.validate_length([1, 2, 3], 2, 5, "test_field")
        assert not self.validator.validate_length([1], 2, 5, "test_field")
        assert not self.validator.validate_length(
            [1, 2, 3, 4, 5, 6], 2, 5, "test_field"
        )

    def test_validate_length_dict(self):
        """Test length validation with dictionaries."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        assert self.validator.validate_length(test_dict, 2, 5, "test_field")

    def test_validate_length_no_length_attribute(self):
        """Test length validation with object without length."""
        assert not self.validator.validate_length(42, 1, 5, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "requires object with length" in errors[0]

    def test_validate_nested_valid(self):
        """Test nested validation with valid data."""
        schema = {
            "name": {"type": "str", "required": True},
            "age": {"type": "int", "min": 0, "max": 150},
            "email": {"type": "str", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        }
        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}
        assert self.validator.validate_nested(data, schema, "person")

    def test_validate_nested_missing_required(self):
        """Test nested validation with missing required field."""
        schema = {"name": {"type": "str", "required": True}, "age": {"type": "int"}}
        data = {"age": 30}
        assert not self.validator.validate_nested(data, schema, "person")
        errors = self.validator.get_validation_errors()
        assert any("Required field missing" in error for error in errors)

    def test_validate_nested_wrong_type(self):
        """Test nested validation with wrong type."""
        schema = {"name": {"type": "str"}, "age": {"type": "int"}}
        data = {"name": "John Doe", "age": "thirty"}  # Should be int
        assert not self.validator.validate_nested(data, schema, "person")

    def test_validate_nested_non_dict(self):
        """Test nested validation with non-dictionary value."""
        schema = {"field": {"type": "str"}}
        assert not self.validator.validate_nested("not_a_dict", schema, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "requires dictionary" in errors[0]

    def test_validate_custom_valid(self):
        """Test custom validation with valid function."""

        def is_even(value):
            return isinstance(value, int) and value % 2 == 0

        assert self.validator.validate_custom(4, is_even, "test_field")
        assert not self.validator.validate_custom(3, is_even, "test_field")

    def test_validate_custom_exception(self):
        """Test custom validation with function that raises exception."""

        def always_fails(value):
            raise ValueError("Custom validation failed")

        assert not self.validator.validate_custom(42, always_fails, "test_field")
        errors = self.validator.get_validation_errors()
        assert len(errors) > 0
        assert "Custom validation error" in errors[0]

    def test_error_management(self):
        """Test validation error management."""
        # Initially empty
        assert len(self.validator.get_validation_errors()) == 0

        # Add some errors
        self.validator.check_range(-1, 0, 10, "field1")
        self.validator.check_type("hello", int, "field2")

        errors = self.validator.get_validation_errors()
        assert len(errors) == 2

        # Clear errors
        self.validator.clear_validation_errors()
        assert len(self.validator.get_validation_errors()) == 0

    def test_strict_mode_exception_handling(self):
        """Test that strict mode raises appropriate exceptions."""
        with pytest.raises(ValidationError) as exc_info:
            self.strict_validator.check_type("hello", int, "test_field")

        error = exc_info.value
        assert error.field_name == "test_field"
        assert "Expected int" in error.message
        assert error.value == "hello"
