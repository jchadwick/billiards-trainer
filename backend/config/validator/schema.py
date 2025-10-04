"""Schema validation for configuration."""

import logging
from typing import Any, Optional, Union

from pydantic import BaseModel, ValidationError

from ..models.schemas import (
    APIConfig,
    ApplicationConfig,
    CameraSettings,
    CoreConfig,
    ProjectorConfig,
    SystemConfig,
    VisionConfig,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors."""

    pass


class SchemaValidator:
    """Configuration schema validator using Pydantic models."""

    def __init__(self):
        """Initialize the schema validator with known schema models."""
        self._schema_models = {
            "application": ApplicationConfig,
            "system": SystemConfig,
            "vision": VisionConfig,
            "core": CoreConfig,
            "api": APIConfig,
            "projector": ProjectorConfig,
            "camera": CameraSettings,
        }

    def validate(self, data: dict, schema: dict) -> tuple[bool, list]:
        """Validate data against schema.

        Args:
            data: Configuration data to validate
            schema: Schema definition (can be Pydantic model name or JSON schema)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            errors = []

            # Handle different schema types
            if isinstance(schema, dict):
                if "model" in schema:
                    # Schema specifies a Pydantic model
                    model_name = schema["model"]
                    if model_name in self._schema_models:
                        model_class = self._schema_models[model_name]
                        return self._validate_with_pydantic_model(data, model_class)
                    else:
                        errors.append(f"Unknown schema model: {model_name}")
                        return False, errors
                else:
                    # JSON Schema validation
                    return self._validate_with_json_schema(data, schema)
            elif isinstance(schema, str):
                # Schema is a model name
                if schema in self._schema_models:
                    model_class = self._schema_models[schema]
                    return self._validate_with_pydantic_model(data, model_class)
                else:
                    errors.append(f"Unknown schema model: {schema}")
                    return False, errors
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                # Schema is a Pydantic model class
                return self._validate_with_pydantic_model(data, schema)
            else:
                errors.append(f"Invalid schema type: {type(schema)}")
                return False, errors

        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return False, [f"Validation error: {str(e)}"]

    def _validate_with_pydantic_model(
        self, data: dict, model_class: type[BaseModel]
    ) -> tuple[bool, list[str]]:
        """Validate data using a Pydantic model.

        Args:
            data: Data to validate
            model_class: Pydantic model class

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            # Attempt to create model instance - this validates the data
            model_class(**data)
            return True, []
        except ValidationError as e:
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            return False, errors
        except Exception as e:
            return False, [f"Unexpected validation error: {str(e)}"]

    def _validate_with_json_schema(
        self, data: dict, schema: dict
    ) -> tuple[bool, list[str]]:
        """Validate data using JSON Schema.

        Args:
            data: Data to validate
            schema: JSON schema definition

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Basic JSON Schema validation
        for key, value in data.items():
            if key in schema:
                field_schema = schema[key]
                field_errors = self._validate_field(key, value, field_schema)
                errors.extend(field_errors)

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field missing: {field}")

        return len(errors) == 0, errors

    def _validate_field(
        self, field_name: str, value: Any, field_schema: dict
    ) -> list[str]:
        """Validate a single field against its schema.

        Args:
            field_name: Name of the field
            value: Value to validate
            field_schema: Schema for the field

        Returns:
            List of validation errors
        """
        errors = []

        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            if not self._check_type(value, expected_type):
                errors.append(
                    f"{field_name}: Expected {expected_type}, got {type(value).__name__}"
                )
                return errors  # Skip further validation if type is wrong

        # Range validation for numbers
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")

            if minimum is not None and value < minimum:
                errors.append(f"{field_name}: Value {value} below minimum {minimum}")

            if maximum is not None and value > maximum:
                errors.append(f"{field_name}: Value {value} above maximum {maximum}")

        # Length validation for strings and arrays
        if isinstance(value, (str, list)):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")

            if min_length is not None and len(value) < min_length:
                errors.append(
                    f"{field_name}: Length {len(value)} below minimum {min_length}"
                )

            if max_length is not None and len(value) > max_length:
                errors.append(
                    f"{field_name}: Length {len(value)} above maximum {max_length}"
                )

        # Pattern validation for strings
        if isinstance(value, str):
            pattern = field_schema.get("pattern")
            if pattern:
                import re

                if not re.match(pattern, value):
                    errors.append(
                        f"{field_name}: Value does not match pattern {pattern}"
                    )

        # Enum validation
        enum_values = field_schema.get("enum")
        if enum_values and value not in enum_values:
            errors.append(f"{field_name}: Value must be one of {enum_values}")

        return errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected type name

        Returns:
            True if type matches
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True  # Unknown type, skip validation

    def register_schema_model(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a new Pydantic model for validation.

        Args:
            name: Name to register the model under
            model_class: Pydantic model class
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must be a Pydantic BaseModel")

        self._schema_models[name] = model_class
        logger.info(f"Registered schema model: {name}")

    def get_available_models(self) -> list[str]:
        """Get list of available schema models.

        Returns:
            List of registered model names
        """
        return list(self._schema_models.keys())

    def validate_partial(
        self,
        data: dict,
        schema: Union[dict, str, type[BaseModel]],
        allow_extra: bool = True,
    ) -> tuple[bool, list[str]]:
        """Validate partial configuration data.

        This allows validation of incomplete configurations where only some
        fields are present.

        Args:
            data: Partial configuration data
            schema: Schema to validate against
            allow_extra: Whether to allow extra fields not in schema

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # For Pydantic models, create a partial validation
                try:
                    # Create model with only the provided fields
                    schema(**data)
                    return True, []
                except ValidationError as e:
                    errors = []
                    for error in e.errors():
                        # Skip "missing" errors for partial validation
                        if error["type"] != "missing":
                            loc = ".".join(str(x) for x in error["loc"])
                            msg = error["msg"]
                            errors.append(f"{loc}: {msg}")
                    return len(errors) == 0, errors
            else:
                # Use regular validation for non-Pydantic schemas
                return self.validate(data, schema)

        except Exception as e:
            return False, [f"Partial validation error: {str(e)}"]
