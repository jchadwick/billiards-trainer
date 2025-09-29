"""Type checking for configuration."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_origin, get_args
import sys

# Handle different Python versions for typing
if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None
import inspect
from pathlib import Path
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

logger = logging.getLogger(__name__)


class TypeCheckError(Exception):
    """Exception raised for type checking errors."""

    def __init__(self, value: Any, expected_type: Any, message: str = None):
        self.value = value
        self.expected_type = expected_type
        self.message = message or f"Expected {expected_type}, got {type(value).__name__}"
        super().__init__(self.message)


class TypeChecker:
    """Configuration type checker with support for complex types and Pydantic models."""

    def __init__(self, strict_mode: bool = False, allow_coercion: bool = True):
        """Initialize type checker.

        Args:
            strict_mode: If True, raise exceptions on type mismatches
            allow_coercion: If True, attempt type coercion for compatible types
        """
        self.strict_mode = strict_mode
        self.allow_coercion = allow_coercion
        self._type_errors: List[str] = []

        # Built-in type mappings for string specifications
        self._type_mapping = {
            'str': str,
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'number': (int, float),
            'bool': bool,
            'boolean': bool,
            'list': list,
            'array': list,
            'dict': dict,
            'object': dict,
            'tuple': tuple,
            'set': set,
            'path': Path,
            'datetime': datetime,
            'none': type(None),
            'null': type(None),
            'any': object,
        }

    def check(self, value: Any, type_spec: Any) -> bool:
        """Check if value matches type specification.

        Args:
            value: Value to type check
            type_spec: Type specification (can be type, string, generic, Union, etc.)

        Returns:
            True if type matches, False otherwise

        Raises:
            TypeCheckError: If strict_mode is True and type check fails
        """
        try:
            self._type_errors.clear()

            # Handle None values
            if value is None:
                return self._check_none_type(type_spec)

            # Handle string type specifications
            if isinstance(type_spec, str):
                return self._check_string_type_spec(value, type_spec)

            # Handle Union types (e.g., Union[int, str])
            if self._is_union_type(type_spec):
                return self._check_union_type(value, type_spec)

            # Handle Optional types (which are Union[T, None])
            if self._is_optional_type(type_spec):
                return self._check_optional_type(value, type_spec)

            # Handle generic types (e.g., List[str], Dict[str, int])
            if self._is_generic_type(type_spec):
                return self._check_generic_type(value, type_spec)

            # Handle Pydantic models
            if PYDANTIC_AVAILABLE and inspect.isclass(type_spec) and issubclass(type_spec, BaseModel):
                return self._check_pydantic_model(value, type_spec)

            # Handle Enum types
            if inspect.isclass(type_spec) and issubclass(type_spec, Enum):
                return self._check_enum_type(value, type_spec)

            # Handle basic types
            if inspect.isclass(type_spec):
                return self._check_basic_type(value, type_spec)

            # Handle tuple of types
            if isinstance(type_spec, tuple):
                return self._check_tuple_of_types(value, type_spec)

            # Default: try isinstance check
            return self._check_isinstance(value, type_spec)

        except Exception as e:
            error_msg = f"Type checking error: {str(e)}"
            self._handle_type_error(error_msg)
            return False

    def _check_none_type(self, type_spec: Any) -> bool:
        """Check if None is allowed by type specification."""
        if type_spec is type(None) or type_spec == 'none' or type_spec == 'null':
            return True

        # Check if it's Optional type
        if self._is_optional_type(type_spec):
            return True

        # Check if it's Union with None
        if self._is_union_type(type_spec):
            union_args = get_args(type_spec)
            if type(None) in union_args:
                return True

        error_msg = f"None value not allowed for type {type_spec}"
        self._handle_type_error(error_msg)
        return False

    def _check_string_type_spec(self, value: Any, type_spec: str) -> bool:
        """Check value against string type specification."""
        type_spec_lower = type_spec.lower()

        if type_spec_lower in self._type_mapping:
            expected_type = self._type_mapping[type_spec_lower]
            return self._check_basic_type(value, expected_type)

        # Handle complex string specifications like "List[str]" or "Optional[int]"
        try:
            # This would require more sophisticated parsing
            # For now, handle common patterns
            if type_spec_lower.startswith('optional[') and type_spec_lower.endswith(']'):
                inner_type = type_spec[9:-1]  # Remove "Optional[" and "]"
                if value is None:
                    return True
                return self._check_string_type_spec(value, inner_type)

            if type_spec_lower.startswith('list[') and type_spec_lower.endswith(']'):
                if not isinstance(value, list):
                    self._handle_type_error(f"Expected list, got {type(value).__name__}")
                    return False
                # For now, don't check inner types from string spec
                return True

            if type_spec_lower.startswith('dict[') and type_spec_lower.endswith(']'):
                if not isinstance(value, dict):
                    self._handle_type_error(f"Expected dict, got {type(value).__name__}")
                    return False
                return True

        except Exception:
            pass

        error_msg = f"Unknown type specification: {type_spec}"
        self._handle_type_error(error_msg)
        return False

    def _check_union_type(self, value: Any, type_spec: Any) -> bool:
        """Check value against Union type."""
        union_args = get_args(type_spec)

        # Temporarily disable strict mode to avoid exceptions during Union checking
        original_strict = self.strict_mode
        self.strict_mode = False

        for union_type in union_args:
            try:
                # Clear errors before each check
                self.clear_type_errors()
                if self.check(value, union_type):
                    self.strict_mode = original_strict
                    return True
            except:
                continue

        # Restore strict mode
        self.strict_mode = original_strict

        type_names = [getattr(t, '__name__', str(t)) for t in union_args]
        error_msg = f"Value does not match any type in Union[{', '.join(type_names)}]"
        self._handle_type_error(error_msg)
        return False

    def _check_optional_type(self, value: Any, type_spec: Any) -> bool:
        """Check value against Optional type."""
        if value is None:
            return True

        # Get the inner type from Optional[T] (which is Union[T, None])
        args = get_args(type_spec)
        if args:
            inner_type = args[0]  # First arg is the actual type
            return self.check(value, inner_type)

        return True  # If no args, treat as Any

    def _check_generic_type(self, value: Any, type_spec: Any) -> bool:
        """Check value against generic type (e.g., List[str], Dict[str, int])."""
        origin = get_origin(type_spec)
        args = get_args(type_spec)

        # Check container type first
        if not isinstance(value, origin):
            error_msg = f"Expected {origin.__name__}, got {type(value).__name__}"
            self._handle_type_error(error_msg)
            return False

        # Check element types
        if origin is list or origin is List:
            return self._check_list_elements(value, args)
        elif origin is dict or origin is Dict:
            return self._check_dict_elements(value, args)
        elif origin is tuple or origin is Tuple:
            return self._check_tuple_elements(value, args)
        elif origin in (set, frozenset):
            return self._check_set_elements(value, args)
        else:
            # For unknown generic types, just check container
            logger.debug(f"Unknown generic type {origin}, only checking container")
            return True

    def _check_list_elements(self, value: list, args: tuple) -> bool:
        """Check list element types."""
        if not args:
            return True  # No type constraint

        element_type = args[0]
        all_valid = True

        # Temporarily disable strict mode to collect all errors
        original_strict = self.strict_mode
        self.strict_mode = False

        for i, item in enumerate(value):
            if not self.check(item, element_type):
                self._handle_type_error(f"List element at index {i} has wrong type")
                all_valid = False

        # Restore strict mode
        self.strict_mode = original_strict

        return all_valid

    def _check_dict_elements(self, value: dict, args: tuple) -> bool:
        """Check dictionary key and value types."""
        if not args:
            return True  # No type constraint

        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any

        all_valid = True

        for k, v in value.items():
            if not self.check(k, key_type):
                self._handle_type_error(f"Dict key '{k}' has wrong type")
                all_valid = False
                if self.strict_mode:
                    break

            if not self.check(v, value_type):
                self._handle_type_error(f"Dict value for key '{k}' has wrong type")
                all_valid = False
                if self.strict_mode:
                    break

        return all_valid

    def _check_tuple_elements(self, value: tuple, args: tuple) -> bool:
        """Check tuple element types."""
        if not args:
            return True  # No type constraint

        # Handle Tuple[int, ...] (variable length)
        if len(args) == 2 and args[1] is ...:
            element_type = args[0]
            for i, item in enumerate(value):
                if not self.check(item, element_type):
                    self._handle_type_error(f"Tuple element at index {i} has wrong type")
                    return False
            return True

        # Handle fixed-length tuples
        if len(value) != len(args):
            self._handle_type_error(f"Tuple length mismatch: expected {len(args)}, got {len(value)}")
            return False

        for i, (item, expected_type) in enumerate(zip(value, args)):
            if not self.check(item, expected_type):
                self._handle_type_error(f"Tuple element at index {i} has wrong type")
                return False

        return True

    def _check_set_elements(self, value: Union[set, frozenset], args: tuple) -> bool:
        """Check set element types."""
        if not args:
            return True  # No type constraint

        element_type = args[0]

        for item in value:
            if not self.check(item, element_type):
                self._handle_type_error(f"Set element has wrong type")
                return False

        return True

    def _check_pydantic_model(self, value: Any, type_spec: Type[BaseModel]) -> bool:
        """Check value against Pydantic model."""
        if isinstance(value, type_spec):
            return True

        if isinstance(value, dict) and self.allow_coercion:
            try:
                # Try to create model instance from dict
                type_spec(**value)
                logger.debug(f"Dict can be coerced to {type_spec.__name__}")
                return True
            except Exception as e:
                error_msg = f"Dict cannot be coerced to {type_spec.__name__}: {str(e)}"
                self._handle_type_error(error_msg)
                return False

        error_msg = f"Expected {type_spec.__name__}, got {type(value).__name__}"
        self._handle_type_error(error_msg)
        return False

    def _check_enum_type(self, value: Any, type_spec: Type[Enum]) -> bool:
        """Check value against Enum type."""
        if isinstance(value, type_spec):
            return True

        if self.allow_coercion:
            # Try to match by value
            try:
                type_spec(value)
                logger.debug(f"Value can be coerced to {type_spec.__name__}")
                return True
            except ValueError:
                pass

            # Try to match by name
            if isinstance(value, str):
                try:
                    getattr(type_spec, value.upper())
                    logger.debug(f"String can be coerced to {type_spec.__name__}")
                    return True
                except AttributeError:
                    pass

        error_msg = f"Expected {type_spec.__name__}, got {type(value).__name__}"
        self._handle_type_error(error_msg)
        return False

    def _check_basic_type(self, value: Any, type_spec: Type) -> bool:
        """Check value against basic type."""
        if isinstance(value, type_spec):
            return True

        if self.allow_coercion:
            coerced_result = self._try_coercion(value, type_spec)
            if isinstance(coerced_result, bool):
                if coerced_result:
                    return True
            else:
                # Coercion returned a value, meaning it succeeded
                return True

        error_msg = f"Expected {type_spec.__name__}, got {type(value).__name__}"
        self._handle_type_error(error_msg)
        return False

    def _check_tuple_of_types(self, value: Any, type_spec: tuple) -> bool:
        """Check value against tuple of possible types."""
        for possible_type in type_spec:
            if self.check(value, possible_type):
                return True

        type_names = [getattr(t, '__name__', str(t)) for t in type_spec]
        error_msg = f"Value does not match any type in ({', '.join(type_names)})"
        self._handle_type_error(error_msg)
        return False

    def _check_isinstance(self, value: Any, type_spec: Any) -> bool:
        """Fallback isinstance check."""
        try:
            if isinstance(value, type_spec):
                return True
        except TypeError:
            pass

        error_msg = f"isinstance check failed for {type_spec}"
        self._handle_type_error(error_msg)
        return False

    def _try_coercion(self, value: Any, target_type: Type) -> bool:
        """Attempt to coerce value to target type."""
        try:
            # Common coercion patterns
            if target_type == str:
                str(value)
                return True
            elif target_type == int:
                if isinstance(value, float) and value.is_integer():
                    return True
                elif isinstance(value, str):
                    int(value)
                    return True
            elif target_type == float:
                if isinstance(value, (int, str)):
                    float(value)
                    return True
            elif target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', 'false', '1', '0', 'yes', 'no')
                elif isinstance(value, (int, float)):
                    return True
            elif target_type == Path:
                if isinstance(value, str):
                    return True
            elif target_type == list:
                if isinstance(value, (tuple, set)):
                    return True
            elif target_type == dict:
                return hasattr(value, 'items')

        except (ValueError, TypeError):
            pass

        return False

    def _is_union_type(self, type_spec: Any) -> bool:
        """Check if type specification is a Union type."""
        origin = get_origin(type_spec)
        return origin is Union or (UnionType is not None and origin is UnionType)

    def _is_optional_type(self, type_spec: Any) -> bool:
        """Check if type specification is Optional (Union[T, None])."""
        origin = get_origin(type_spec)
        if origin is Union or (UnionType is not None and origin is UnionType):
            args = get_args(type_spec)
            return len(args) == 2 and type(None) in args
        return False

    def _is_generic_type(self, type_spec: Any) -> bool:
        """Check if type specification is a generic type."""
        return get_origin(type_spec) is not None

    def get_type_errors(self) -> List[str]:
        """Get list of type errors from last check.

        Returns:
            List of error messages
        """
        return self._type_errors.copy()

    def clear_type_errors(self) -> None:
        """Clear type error list."""
        self._type_errors.clear()

    def _handle_type_error(self, message: str) -> None:
        """Handle type error according to configured mode.

        Args:
            message: Error message
        """
        self._type_errors.append(message)
        logger.debug(f"Type check error: {message}")

        if self.strict_mode:
            raise TypeCheckError(None, None, message)

    def validate_schema_types(self, data: dict, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all types in a data dictionary against a schema.

        Args:
            data: Data dictionary to validate
            schema: Schema with field type specifications

        Returns:
            Tuple of (all_valid, error_messages)
        """
        all_valid = True
        all_errors = []

        for field_name, type_spec in schema.items():
            if field_name in data:
                self.clear_type_errors()
                if not self.check(data[field_name], type_spec):
                    all_valid = False
                    errors = self.get_type_errors()
                    for error in errors:
                        all_errors.append(f"{field_name}: {error}")

        return all_valid, all_errors
