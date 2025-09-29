"""Tests for type checker."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest

from backend.config.validator.types import TypeChecker, TypeCheckError


class ConfigTestEnum(Enum):
    """Test enum for type checking."""

    OPTION_A = "a"
    OPTION_B = "b"
    OPTION_C = "c"


try:
    from pydantic import BaseModel

    class ConfigTestModel(BaseModel):
        """Test Pydantic model for type checking."""

        name: str
        age: int
        active: bool = True

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    ConfigTestModel = None


class TestTypeChecker:
    """Test cases for TypeChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = TypeChecker()
        self.strict_checker = TypeChecker(strict_mode=True)
        self.no_coercion_checker = TypeChecker(allow_coercion=False)

    def test_init(self):
        """Test TypeChecker initialization."""
        assert not self.checker.strict_mode
        assert self.checker.allow_coercion
        assert self.checker._type_errors == []

    def test_basic_type_checking(self):
        """Test basic type checking."""
        assert self.checker.check("hello", str)
        assert self.checker.check(42, int)
        assert self.checker.check(3.14, float)
        assert self.checker.check(True, bool)
        assert self.checker.check([], list)
        assert self.checker.check({}, dict)
        assert self.checker.check((1, 2), tuple)
        assert self.checker.check({1, 2}, set)

    def test_basic_type_mismatch(self):
        """Test basic type mismatches."""
        assert not self.checker.check("hello", int)
        assert not self.checker.check(42, str)
        assert not self.checker.check([], dict)
        assert not self.checker.check({}, list)

    def test_string_type_specifications(self):
        """Test string type specifications."""
        assert self.checker.check("hello", "str")
        assert self.checker.check("hello", "string")
        assert self.checker.check(42, "int")
        assert self.checker.check(42, "integer")
        assert self.checker.check(3.14, "float")
        assert self.checker.check(42, "number")  # accepts int
        assert self.checker.check(3.14, "number")  # accepts float
        assert self.checker.check(True, "bool")
        assert self.checker.check(True, "boolean")
        assert self.checker.check([], "list")
        assert self.checker.check([], "array")
        assert self.checker.check({}, "dict")
        assert self.checker.check({}, "object")

    def test_none_type_checking(self):
        """Test None type checking."""
        assert self.checker.check(None, type(None))
        assert self.checker.check(None, "none")
        assert self.checker.check(None, "null")
        assert not self.checker.check(None, str)
        assert not self.checker.check("hello", type(None))

    def test_union_type_checking(self):
        """Test Union type checking."""
        union_type = Union[str, int]
        assert self.checker.check("hello", union_type)
        assert self.checker.check(42, union_type)
        assert not self.checker.check(3.14, union_type)  # float not in union
        assert not self.checker.check([], union_type)

    def test_optional_type_checking(self):
        """Test Optional type checking."""
        optional_str = Optional[str]
        assert self.checker.check("hello", optional_str)
        assert self.checker.check(None, optional_str)
        assert not self.checker.check(42, optional_str)

    def test_list_type_checking(self):
        """Test List generic type checking."""
        list_str = list[str]
        assert self.checker.check(["a", "b", "c"], list_str)
        assert not self.checker.check([1, 2, 3], list_str)  # wrong element type
        assert not self.checker.check("not_a_list", list_str)  # wrong container type

        # Mixed types in list
        assert not self.checker.check(["a", 2, "c"], list_str)

    def test_dict_type_checking(self):
        """Test Dict generic type checking."""
        dict_str_int = dict[str, int]
        assert self.checker.check({"a": 1, "b": 2}, dict_str_int)
        assert not self.checker.check({1: "a", 2: "b"}, dict_str_int)  # wrong key type
        assert not self.checker.check({"a": "b"}, dict_str_int)  # wrong value type
        assert not self.checker.check(
            "not_a_dict", dict_str_int
        )  # wrong container type

    def test_tuple_type_checking(self):
        """Test Tuple generic type checking."""
        # Fixed-length tuple
        tuple_str_int = tuple[str, int]
        assert self.checker.check(("hello", 42), tuple_str_int)
        assert not self.checker.check(("hello",), tuple_str_int)  # wrong length
        assert not self.checker.check(("hello", "world"), tuple_str_int)  # wrong type

        # Variable-length tuple
        tuple_str_var = tuple[str, ...]
        assert self.checker.check(("a", "b", "c"), tuple_str_var)
        assert not self.checker.check((1, 2, 3), tuple_str_var)  # wrong element type

    def test_enum_type_checking(self):
        """Test Enum type checking."""
        assert self.checker.check(ConfigTestEnum.OPTION_A, ConfigTestEnum)
        assert not self.checker.check("not_an_enum_value", ConfigTestEnum)

        # Test coercion
        assert self.checker.check("a", ConfigTestEnum)  # by value
        assert self.checker.check("OPTION_A", ConfigTestEnum)  # by name (uppercase)

    def test_enum_no_coercion(self):
        """Test Enum checking without coercion."""
        assert self.no_coercion_checker.check(ConfigTestEnum.OPTION_A, ConfigTestEnum)
        assert not self.no_coercion_checker.check("a", ConfigTestEnum)
        assert not self.no_coercion_checker.check("OPTION_A", ConfigTestEnum)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_model_checking(self):
        """Test Pydantic model checking."""
        model_instance = ConfigTestModel(name="John", age=30)
        assert self.checker.check(model_instance, ConfigTestModel)

        # Test dict coercion
        valid_dict = {"name": "Jane", "age": 25}
        assert self.checker.check(valid_dict, ConfigTestModel)

        # Invalid dict
        invalid_dict = {"name": "Bob"}  # missing required field
        assert not self.checker.check(invalid_dict, ConfigTestModel)

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_no_coercion(self):
        """Test Pydantic model checking without coercion."""
        model_instance = ConfigTestModel(name="John", age=30)
        assert self.no_coercion_checker.check(model_instance, ConfigTestModel)

        valid_dict = {"name": "Jane", "age": 25}
        assert not self.no_coercion_checker.check(valid_dict, ConfigTestModel)

    def test_path_type_checking(self):
        """Test Path type checking."""
        path_obj = Path("/test/path")
        assert self.checker.check(path_obj, Path)
        assert self.checker.check("/test/path", Path)  # string coercion

        assert not self.no_coercion_checker.check("/test/path", Path)

    def test_type_coercion(self):
        """Test type coercion functionality."""
        # String to int
        assert self.checker.check("42", int)
        assert not self.no_coercion_checker.check("42", int)

        # String to float
        assert self.checker.check("3.14", float)
        assert not self.no_coercion_checker.check("3.14", float)

        # Float to int (if integer)
        assert self.checker.check(42.0, int)
        assert not self.checker.check(42.5, int)

        # Various to bool
        assert self.checker.check("true", bool)
        assert self.checker.check("false", bool)
        assert self.checker.check("1", bool)
        assert self.checker.check("0", bool)
        assert self.checker.check(1, bool)
        assert self.checker.check(0, bool)

    def test_tuple_of_types(self):
        """Test checking against tuple of possible types."""
        type_spec = (str, int, float)
        assert self.checker.check("hello", type_spec)
        assert self.checker.check(42, type_spec)
        assert self.checker.check(3.14, type_spec)
        assert not self.checker.check([], type_spec)

    def test_complex_nested_types(self):
        """Test complex nested type specifications."""
        # List of dicts
        list_dict_type = list[dict[str, int]]
        valid_data = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        assert self.checker.check(valid_data, list_dict_type)

        invalid_data = [{"a": 1, "b": "not_int"}]
        assert not self.checker.check(invalid_data, list_dict_type)

        # Dict with Union values
        dict_union_type = dict[str, Union[str, int]]
        valid_union_data = {"name": "John", "age": 30, "city": "NYC"}
        assert self.checker.check(valid_union_data, dict_union_type)

    def test_string_generic_patterns(self):
        """Test string patterns for generic types."""
        # Optional pattern
        assert self.checker.check(None, "Optional[str]")
        assert self.checker.check("hello", "Optional[str]")
        assert not self.checker.check(42, "Optional[str]")

        # List pattern
        assert self.checker.check(["a", "b"], "List[str]")
        assert not self.checker.check("not_a_list", "List[str]")

        # Dict pattern
        assert self.checker.check({"a": 1}, "Dict[str]")
        assert not self.checker.check("not_a_dict", "Dict[str]")

    def test_unknown_type_specification(self):
        """Test handling of unknown type specifications."""
        assert not self.checker.check("test", "UnknownType")
        errors = self.checker.get_type_errors()
        assert len(errors) > 0
        assert "Unknown type specification" in errors[0]

    def test_error_management(self):
        """Test type error management."""
        # Initially empty
        assert len(self.checker.get_type_errors()) == 0

        # Add some errors
        self.checker.check("hello", int)
        self.checker.check([], str)

        errors = self.checker.get_type_errors()
        assert len(errors) >= 2

        # Clear errors
        self.checker.clear_type_errors()
        assert len(self.checker.get_type_errors()) == 0

    def test_strict_mode_exceptions(self):
        """Test strict mode exception handling."""
        with pytest.raises(TypeCheckError):
            self.strict_checker.check("hello", int)

        with pytest.raises(TypeCheckError):
            self.strict_checker.check([], dict)

    def test_validate_schema_types(self):
        """Test schema validation functionality."""
        schema = {"name": str, "age": int, "active": bool, "scores": list[float]}

        valid_data = {
            "name": "John",
            "age": 30,
            "active": True,
            "scores": [85.5, 92.0, 78.5],
        }

        is_valid, errors = self.checker.validate_schema_types(valid_data, schema)
        assert is_valid
        assert len(errors) == 0

        invalid_data = {
            "name": 123,  # should be str
            "age": "thirty",  # should be int
            "active": True,
            "scores": [85.5, "not_float", 78.5],  # contains invalid element
        }

        is_valid, errors = self.checker.validate_schema_types(invalid_data, schema)
        assert not is_valid
        assert len(errors) > 0

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty containers
        assert self.checker.check([], list[str])
        assert self.checker.check({}, dict[str, int])
        assert self.checker.check((), tuple[str, ...])

        # Very nested types
        deeply_nested = list[dict[str, list[int]]]
        valid_nested = [{"numbers": [1, 2, 3]}, {"more_numbers": [4, 5, 6]}]
        assert self.checker.check(valid_nested, deeply_nested)

    def test_datetime_type_checking(self):
        """Test datetime type checking."""
        now = datetime.now()
        assert self.checker.check(now, datetime)
        assert self.checker.check(now, "datetime")
        assert not self.checker.check("not_datetime", datetime)

    def test_any_type_checking(self):
        """Test Any type checking."""
        assert self.checker.check("anything", object)  # object represents Any
        assert self.checker.check(42, object)
        assert self.checker.check([], object)
        assert self.checker.check({}, object)

    def test_fallback_isinstance_check(self):
        """Test fallback isinstance functionality."""

        # Create a custom class
        class CustomClass:
            pass

        instance = CustomClass()
        assert self.checker.check(instance, CustomClass)
        assert not self.checker.check("not_instance", CustomClass)

    def test_recursive_type_error_handling(self):
        """Test error handling in recursive type checking."""
        # This should not cause infinite recursion
        recursive_dict = {}
        recursive_dict["self"] = recursive_dict

        # Should handle gracefully without crashing
        result = self.checker.check(recursive_dict, dict[str, Any])
        # The exact result doesn't matter as much as not crashing
        assert isinstance(result, bool)
