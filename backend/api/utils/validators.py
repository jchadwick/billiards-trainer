"""Input validation utilities for the billiards trainer API."""

import ipaddress
import logging
import re
from typing import Any, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error with detailed information."""

    def __init__(self, field: str, value: Any, message: str, code: str = "VAL_001"):
        """Initialize validation error.

        Args:
            field: Field name that failed validation
            value: Value that failed validation
            message: Human-readable error message
            code: Error code for categorization
        """
        self.field = field
        self.value = value
        self.message = message
        self.code = code
        super().__init__(f"Validation failed for field '{field}': {message}")


class ValidationResult(BaseModel):
    """Result of validation operation."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="List of validation errors"
    )
    sanitized_value: Optional[Any] = Field(
        default=None, description="Sanitized/normalized value"
    )


class BaseValidator:
    """Base class for custom validators."""

    def __init__(self, error_code: str = "VAL_001"):
        """Initialize validator with error code."""
        self.error_code = error_code

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Validate a value.

        Args:
            value: Value to validate
            field_name: Name of the field being validated

        Returns:
            ValidationResult with validation outcome
        """
        raise NotImplementedError("Subclasses must implement validate method")

    def _create_error(self, field: str, value: Any, message: str) -> dict[str, Any]:
        """Create standardized error dictionary."""
        return {
            "field": field,
            "value": str(value) if value is not None else None,
            "message": message,
            "code": self.error_code,
        }


class StringValidator(BaseValidator):
    """Validator for string values with various constraints."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[list[str]] = None,
        forbidden_patterns: Optional[list[str]] = None,
        trim_whitespace: bool = True,
        error_code: str = "VAL_003",
    ):
        """Initialize string validator.

        Args:
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern the string must match
            allowed_values: List of allowed string values
            forbidden_patterns: List of forbidden regex patterns
            trim_whitespace: Whether to trim whitespace
            error_code: Error code for validation failures
        """
        super().__init__(error_code)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_values = set(allowed_values) if allowed_values else None
        self.forbidden_patterns = (
            [re.compile(p) for p in forbidden_patterns] if forbidden_patterns else []
        )
        self.trim_whitespace = trim_whitespace

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Validate string value."""
        errors = []

        # Type check
        if not isinstance(value, str):
            errors.append(self._create_error(field_name, value, "Must be a string"))
            return ValidationResult(is_valid=False, errors=errors)

        # Normalize value
        sanitized_value = value.strip() if self.trim_whitespace else value

        # Length checks
        if self.min_length is not None and len(sanitized_value) < self.min_length:
            errors.append(
                self._create_error(
                    field_name,
                    value,
                    f"Must be at least {self.min_length} characters long",
                )
            )

        if self.max_length is not None and len(sanitized_value) > self.max_length:
            errors.append(
                self._create_error(
                    field_name,
                    value,
                    f"Must be at most {self.max_length} characters long",
                )
            )

        # Pattern matching
        if self.pattern and not self.pattern.match(sanitized_value):
            errors.append(
                self._create_error(
                    field_name, value, f"Must match pattern: {self.pattern.pattern}"
                )
            )

        # Allowed values check
        if self.allowed_values and sanitized_value not in self.allowed_values:
            errors.append(
                self._create_error(
                    field_name,
                    value,
                    f"Must be one of: {', '.join(self.allowed_values)}",
                )
            )

        # Forbidden patterns check
        for forbidden_pattern in self.forbidden_patterns:
            if forbidden_pattern.search(sanitized_value):
                errors.append(
                    self._create_error(
                        field_name,
                        value,
                        f"Contains forbidden pattern: {forbidden_pattern.pattern}",
                    )
                )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_value
        )


class NumericValidator(BaseValidator):
    """Validator for numeric values."""

    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_negative: bool = True,
        allow_zero: bool = True,
        decimal_places: Optional[int] = None,
        error_code: str = "VAL_003",
    ):
        """Initialize numeric validator.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_negative: Whether negative values are allowed
            allow_zero: Whether zero is allowed
            decimal_places: Maximum number of decimal places
            error_code: Error code for validation failures
        """
        super().__init__(error_code)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
        self.allow_zero = allow_zero
        self.decimal_places = decimal_places

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Validate numeric value."""
        errors = []

        # Type check and conversion
        try:
            if isinstance(value, str):
                # Try to convert string to number
                numeric_value = float(value) if "." in value else int(value)
            elif isinstance(value, (int, float)):
                numeric_value = value
            else:
                errors.append(self._create_error(field_name, value, "Must be a number"))
                return ValidationResult(is_valid=False, errors=errors)
        except (ValueError, TypeError):
            errors.append(
                self._create_error(field_name, value, "Must be a valid number")
            )
            return ValidationResult(is_valid=False, errors=errors)

        # Range checks
        if self.min_value is not None and numeric_value < self.min_value:
            errors.append(
                self._create_error(
                    field_name, value, f"Must be at least {self.min_value}"
                )
            )

        if self.max_value is not None and numeric_value > self.max_value:
            errors.append(
                self._create_error(
                    field_name, value, f"Must be at most {self.max_value}"
                )
            )

        # Sign checks
        if not self.allow_negative and numeric_value < 0:
            errors.append(
                self._create_error(field_name, value, "Negative values not allowed")
            )

        if not self.allow_zero and numeric_value == 0:
            errors.append(self._create_error(field_name, value, "Zero not allowed"))

        # Decimal places check
        if self.decimal_places is not None and isinstance(numeric_value, float):
            decimal_str = str(numeric_value).split(".")
            if len(decimal_str) > 1 and len(decimal_str[1]) > self.decimal_places:
                errors.append(
                    self._create_error(
                        field_name,
                        value,
                        f"Must have at most {self.decimal_places} decimal places",
                    )
                )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=numeric_value
        )


class EmailValidator(BaseValidator):
    """Validator for email addresses."""

    def __init__(self, error_code: str = "VAL_003"):
        """Initialize email validator."""
        super().__init__(error_code)
        # RFC 5322 compliant email regex (simplified)
        self.email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )

    def validate(self, value: Any, field_name: str = "email") -> ValidationResult:
        """Validate email address."""
        errors = []

        if not isinstance(value, str):
            errors.append(self._create_error(field_name, value, "Must be a string"))
            return ValidationResult(is_valid=False, errors=errors)

        sanitized_value = value.strip().lower()

        if not self.email_pattern.match(sanitized_value):
            errors.append(
                self._create_error(field_name, value, "Must be a valid email address")
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_value
        )


class IPAddressValidator(BaseValidator):
    """Validator for IP addresses."""

    def __init__(
        self,
        allow_ipv4: bool = True,
        allow_ipv6: bool = True,
        allow_private: bool = True,
        error_code: str = "VAL_003",
    ):
        """Initialize IP address validator.

        Args:
            allow_ipv4: Whether IPv4 addresses are allowed
            allow_ipv6: Whether IPv6 addresses are allowed
            allow_private: Whether private IP addresses are allowed
            error_code: Error code for validation failures
        """
        super().__init__(error_code)
        self.allow_ipv4 = allow_ipv4
        self.allow_ipv6 = allow_ipv6
        self.allow_private = allow_private

    def validate(self, value: Any, field_name: str = "ip_address") -> ValidationResult:
        """Validate IP address."""
        errors = []

        if not isinstance(value, str):
            errors.append(self._create_error(field_name, value, "Must be a string"))
            return ValidationResult(is_valid=False, errors=errors)

        sanitized_value = value.strip()

        try:
            ip = ipaddress.ip_address(sanitized_value)

            # Check IP version
            if isinstance(ip, ipaddress.IPv4Address) and not self.allow_ipv4:
                errors.append(
                    self._create_error(field_name, value, "IPv4 addresses not allowed")
                )

            if isinstance(ip, ipaddress.IPv6Address) and not self.allow_ipv6:
                errors.append(
                    self._create_error(field_name, value, "IPv6 addresses not allowed")
                )

            # Check if private
            if not self.allow_private and ip.is_private:
                errors.append(
                    self._create_error(
                        field_name, value, "Private IP addresses not allowed"
                    )
                )

        except ValueError:
            errors.append(
                self._create_error(field_name, value, "Must be a valid IP address")
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_value
        )


class CoordinateValidator(BaseValidator):
    """Validator for coordinate pairs (x, y)."""

    def __init__(
        self,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        min_y: Optional[float] = None,
        max_y: Optional[float] = None,
        error_code: str = "VAL_003",
    ):
        """Initialize coordinate validator.

        Args:
            min_x: Minimum X coordinate value
            max_x: Maximum X coordinate value
            min_y: Minimum Y coordinate value
            max_y: Maximum Y coordinate value
            error_code: Error code for validation failures
        """
        super().__init__(error_code)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def validate(self, value: Any, field_name: str = "coordinates") -> ValidationResult:
        """Validate coordinate pair."""
        errors = []

        # Check if it's a list/tuple with 2 elements
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            errors.append(
                self._create_error(
                    field_name,
                    value,
                    "Must be a list or tuple with exactly 2 elements [x, y]",
                )
            )
            return ValidationResult(is_valid=False, errors=errors)

        try:
            x, y = float(value[0]), float(value[1])
        except (ValueError, TypeError):
            errors.append(
                self._create_error(field_name, value, "Coordinates must be numeric")
            )
            return ValidationResult(is_valid=False, errors=errors)

        # Range checks
        if self.min_x is not None and x < self.min_x:
            errors.append(
                self._create_error(
                    field_name, value, f"X coordinate must be at least {self.min_x}"
                )
            )

        if self.max_x is not None and x > self.max_x:
            errors.append(
                self._create_error(
                    field_name, value, f"X coordinate must be at most {self.max_x}"
                )
            )

        if self.min_y is not None and y < self.min_y:
            errors.append(
                self._create_error(
                    field_name, value, f"Y coordinate must be at least {self.min_y}"
                )
            )

        if self.max_y is not None and y > self.max_y:
            errors.append(
                self._create_error(
                    field_name, value, f"Y coordinate must be at most {self.max_y}"
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=[x, y]
        )


class ColorValidator(BaseValidator):
    """Validator for color values (hex, rgb, hsl)."""

    def __init__(self, error_code: str = "VAL_003"):
        """Initialize color validator."""
        super().__init__(error_code)
        self.hex_pattern = re.compile(r"^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")

    def validate(self, value: Any, field_name: str = "color") -> ValidationResult:
        """Validate color value."""
        errors = []

        if not isinstance(value, str):
            errors.append(self._create_error(field_name, value, "Must be a string"))
            return ValidationResult(is_valid=False, errors=errors)

        sanitized_value = value.strip()

        # Check hex color
        if self.hex_pattern.match(sanitized_value):
            # Normalize hex color (add # if missing, expand 3-digit to 6-digit)
            if not sanitized_value.startswith("#"):
                sanitized_value = "#" + sanitized_value
            if len(sanitized_value) == 4:  # #RGB -> #RRGGBB
                sanitized_value = "#" + "".join(c * 2 for c in sanitized_value[1:])
        else:
            errors.append(
                self._create_error(
                    field_name,
                    value,
                    "Must be a valid hex color (e.g., #FF0000 or #F00)",
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_value
        )


# Pre-configured validators for common use cases
USERNAME_VALIDATOR = StringValidator(
    min_length=3,
    max_length=50,
    pattern=r"^[a-zA-Z0-9_.-]+$",
    forbidden_patterns=[r"admin", r"root", r"system"],
    error_code="VAL_003",
)

PASSWORD_VALIDATOR = StringValidator(
    min_length=8,
    max_length=128,
    pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]",
    error_code="VAL_003",
)

COORDINATE_VALIDATOR = CoordinateValidator(
    min_x=0,
    max_x=1920,  # Assuming HD resolution
    min_y=0,
    max_y=1080,
    error_code="VAL_003",
)

RADIUS_VALIDATOR = NumericValidator(
    min_value=1,
    max_value=100,
    allow_negative=False,
    allow_zero=False,
    decimal_places=2,
    error_code="VAL_003",
)

ANGLE_VALIDATOR = NumericValidator(
    min_value=0,
    max_value=360,
    allow_negative=False,
    decimal_places=2,
    error_code="VAL_003",
)


def validate_request_data(
    data: dict[str, Any], validators: dict[str, BaseValidator]
) -> dict[str, Any]:
    """Validate request data against a set of validators.

    Args:
        data: Dictionary of data to validate
        validators: Dictionary mapping field names to validators

    Returns:
        Dictionary of sanitized data

    Raises:
        HTTPException: If validation fails
    """
    errors = []
    sanitized_data = {}

    for field_name, validator in validators.items():
        if field_name in data:
            result = validator.validate(data[field_name], field_name)
            if result.is_valid:
                sanitized_data[field_name] = result.sanitized_value
            else:
                errors.extend(result.errors)
        # Note: Required field validation should be handled by Pydantic models

    if errors:
        logger.warning(f"Validation failed: {errors}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VAL_001",
                "message": "Request validation failed",
                "validation_errors": errors,
            },
        )

    return sanitized_data


def sanitize_string_input(value: str, max_length: int = 1000) -> str:
    """Sanitize string input for security.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)

    # Remove null bytes and control characters
    sanitized = "".join(char for char in value if ord(char) >= 32 or char in "\t\n\r")

    # Trim whitespace
    sanitized = sanitized.strip()

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def validate_file_upload(
    filename: str,
    content_type: str,
    file_size: int,
    allowed_extensions: list[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB default
) -> ValidationResult:
    """Validate file upload parameters.

    Args:
        filename: Name of the uploaded file
        content_type: MIME type of the file
        file_size: Size of the file in bytes
        allowed_extensions: List of allowed file extensions
        max_size: Maximum file size in bytes

    Returns:
        ValidationResult with validation outcome
    """
    errors = []

    # Check file size
    if file_size > max_size:
        errors.append(
            {
                "field": "file_size",
                "value": file_size,
                "message": f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)",
                "code": "VAL_003",
            }
        )

    # Check file extension
    if allowed_extensions:
        file_ext = filename.lower().split(".")[-1] if "." in filename else ""
        if file_ext not in [ext.lower().lstrip(".") for ext in allowed_extensions]:
            errors.append(
                {
                    "field": "filename",
                    "value": filename,
                    "message": f"File extension '{file_ext}' not allowed. Allowed: {', '.join(allowed_extensions)}",
                    "code": "VAL_003",
                }
            )

    # Basic filename sanitization
    sanitized_filename = sanitize_string_input(filename, 255)

    return ValidationResult(
        is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized_filename
    )
