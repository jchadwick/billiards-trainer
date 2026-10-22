"""Behavioral contracts for API input validators and sanitizers."""

from typing import Any, cast

import pytest
from fastapi import HTTPException

from backend.api.utils.validators import (
    ANGLE_VALIDATOR,
    COORDINATE_VALIDATOR,
    RADIUS_VALIDATOR,
    BaseValidator,
    ColorValidator,
    CoordinateValidator,
    EmailValidator,
    IPAddressValidator,
    NumericValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    sanitize_string_input,
    validate_file_upload,
    validate_request_data,
)


def test_validation_error_retains_structured_details_and_message() -> None:
    error = ValidationError("radius", -1, "must be positive", "BALL_001")

    assert error.field == "radius"
    assert error.value == -1
    assert error.message == "must be positive"
    assert error.code == "BALL_001"
    assert str(error) == "Validation failed for field 'radius': must be positive"


def test_validation_result_uses_fresh_error_lists_and_serializes() -> None:
    first = ValidationResult(is_valid=True, sanitized_value=(1, 2))
    second = ValidationResult(is_valid=False)
    first.errors.append({"field": "x"})

    assert second.errors == []
    assert first.model_dump(mode="json")["sanitized_value"] == [1, 2]


def test_base_validator_defines_the_subclass_contract() -> None:
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        BaseValidator().validate("value")


def test_base_validator_error_shape_stringifies_non_null_values() -> None:
    validator = BaseValidator("CUSTOM")

    assert validator._create_error("field", 17, "bad") == {
        "field": "field",
        "value": "17",
        "message": "bad",
        "code": "CUSTOM",
    }
    assert validator._create_error("field", None, "missing")["value"] is None


def test_string_validator_normalizes_and_accepts_configured_value() -> None:
    validator = StringValidator(
        min_length=3,
        max_length=8,
        pattern=r"^[a-z]+$",
        allowed_values=["table", "balls"],
        forbidden_patterns=[r"admin"],
    )

    result = validator.validate(" table ", "name")

    assert result.is_valid
    assert result.errors == []
    assert result.sanitized_value == "table"


def test_string_validator_rejects_non_string_without_sanitized_value() -> None:
    result = StringValidator().validate(42, "name")

    assert not result.is_valid
    assert result.sanitized_value is None
    assert result.errors == [
        {
            "field": "name",
            "value": "42",
            "message": "Must be a string",
            "code": "VAL_003",
        }
    ]


def test_string_validator_accumulates_length_pattern_allowlist_and_forbidden_errors() -> (
    None
):
    validator = StringValidator(
        min_length=5,
        max_length=7,
        pattern=r"^[a-z]+$",
        allowed_values=["safe"],
        forbidden_patterns=[r"bad", r"!"],
    )

    result = validator.validate(" bad! ", "label")

    assert not result.is_valid
    assert result.sanitized_value == "bad!"
    assert [error["message"] for error in result.errors] == [
        "Must be at least 5 characters long",
        "Must match pattern: ^[a-z]+$",
        "Must be one of: safe",
        "Contains forbidden pattern: bad",
        "Contains forbidden pattern: !",
    ]
    assert all(error["field"] == "label" for error in result.errors)


def test_string_validator_can_preserve_whitespace_and_reports_maximum_length() -> None:
    result = StringValidator(max_length=3, trim_whitespace=False).validate("  abc  ")

    assert not result.is_valid
    assert result.sanitized_value == "  abc  "
    assert result.errors[0]["message"] == "Must be at most 3 characters long"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("17", 17), ("2.5", 2.5), (8, 8), (3.75, 3.75)],
)
def test_numeric_validator_converts_supported_inputs(
    value: object, expected: int | float
) -> None:
    result = NumericValidator().validate(value, "amount")

    assert result.is_valid
    assert result.sanitized_value == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [([1], "Must be a number"), ("not-a-number", "Must be a valid number")],
)
def test_numeric_validator_rejects_unsupported_or_malformed_inputs(
    value: object, message: str
) -> None:
    result = NumericValidator().validate(value)

    assert not result.is_valid
    assert result.sanitized_value is None
    assert result.errors[0]["message"] == message


def test_numeric_validator_accumulates_range_sign_zero_and_precision_errors() -> None:
    validator = NumericValidator(
        min_value=2,
        max_value=5,
        allow_negative=False,
        allow_zero=False,
        decimal_places=1,
    )

    result = validator.validate(-1.25, "speed")

    assert not result.is_valid
    assert result.sanitized_value == -1.25
    assert [error["message"] for error in result.errors] == [
        "Must be at least 2",
        "Negative values not allowed",
        "Must have at most 1 decimal places",
    ]

    too_large = validator.validate(5.01, "speed")
    assert [error["message"] for error in too_large.errors] == [
        "Must be at most 5",
        "Must have at most 1 decimal places",
    ]


def test_numeric_validator_checks_zero_and_only_checks_decimal_places_for_floats() -> (
    None
):
    validator = NumericValidator(allow_zero=False, decimal_places=0)

    zero = validator.validate("0")
    integer = validator.validate(1)

    assert [error["message"] for error in zero.errors] == ["Zero not allowed"]
    assert integer.is_valid


def test_preconfigured_coordinate_radius_and_angle_validators_enforce_bounds() -> None:
    assert COORDINATE_VALIDATOR.validate([1920, 1080]).is_valid
    assert not COORDINATE_VALIDATOR.validate([-1, 1081]).is_valid
    assert RADIUS_VALIDATOR.validate("12.25").sanitized_value == 12.25
    assert not RADIUS_VALIDATOR.validate(0).is_valid
    assert ANGLE_VALIDATOR.validate(360).is_valid
    assert not ANGLE_VALIDATOR.validate(-0.1).is_valid


def test_email_validator_trims_lowercases_and_rejects_invalid_shapes() -> None:
    validator = EmailValidator()
    valid = validator.validate("  PLAYER+cue@Example.COM  ")
    invalid = validator.validate("player@localhost")
    wrong_type = validator.validate(None)

    assert valid.is_valid
    assert valid.sanitized_value == "player+cue@example.com"
    assert invalid.errors[0]["message"] == "Must be a valid email address"
    assert wrong_type.errors[0]["message"] == "Must be a string"


def test_ip_address_validator_normalizes_and_checks_versions_and_private_ranges() -> (
    None
):
    assert IPAddressValidator().validate("  192.168.1.31 ").sanitized_value == (
        "192.168.1.31"
    )
    assert (
        IPAddressValidator(allow_ipv4=False).validate("8.8.8.8").errors[0]["message"]
        == "IPv4 addresses not allowed"
    )
    assert (
        IPAddressValidator(allow_ipv6=False)
        .validate("2001:4860:4860::8888")
        .errors[0]["message"]
        == "IPv6 addresses not allowed"
    )
    assert (
        IPAddressValidator(allow_private=False)
        .validate("192.168.1.31")
        .errors[0]["message"]
        == "Private IP addresses not allowed"
    )


def test_ip_address_validator_rejects_invalid_and_non_string_values() -> None:
    invalid = IPAddressValidator().validate("999.1.1.1", "host")
    wrong_type = IPAddressValidator().validate(123, "host")

    assert invalid.errors[0]["message"] == "Must be a valid IP address"
    assert wrong_type.errors[0]["message"] == "Must be a string"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1,2", "Must be a list or tuple with exactly 2 elements [x, y]"),
        ([1], "Must be a list or tuple with exactly 2 elements [x, y]"),
        ([1, object()], "Coordinates must be numeric"),
    ],
)
def test_coordinate_validator_rejects_invalid_shapes_and_values(
    value: object, message: str
) -> None:
    result = CoordinateValidator().validate(value)

    assert not result.is_valid
    assert result.sanitized_value is None
    assert result.errors[0]["message"] == message


def test_coordinate_validator_converts_tuple_and_reports_each_axis_bound() -> None:
    validator = CoordinateValidator(min_x=0, max_x=10, min_y=0, max_y=20)

    accepted = validator.validate(("2.5", "4"))
    too_small = validator.validate((-1, 4))
    too_low_y = validator.validate((2, -1))
    too_large = validator.validate((11, 21))

    assert accepted.is_valid
    assert accepted.sanitized_value == [2.5, 4.0]
    assert [error["message"] for error in too_small.errors] == [
        "X coordinate must be at least 0"
    ]
    assert [error["message"] for error in too_low_y.errors] == [
        "Y coordinate must be at least 0"
    ]
    assert [error["message"] for error in too_large.errors] == [
        "X coordinate must be at most 10",
        "Y coordinate must be at most 20",
    ]


def test_color_validator_normalizes_short_and_full_hex_and_rejects_other_forms() -> (
    None
):
    validator = ColorValidator()

    assert validator.validate(" F0a ").sanitized_value == "#FF00aa"
    assert validator.validate("abcdef").sanitized_value == "#abcdef"
    assert validator.validate("#f00").sanitized_value == "#ff0000"
    invalid = validator.validate("rgb(1,2,3)")
    wrong_type = validator.validate([255, 0, 0])

    assert invalid.errors[0]["message"].startswith("Must be a valid hex color")
    assert wrong_type.errors[0]["message"] == "Must be a string"


def test_validate_request_data_returns_only_present_fields_and_sanitized_values() -> (
    None
):
    result = validate_request_data(
        {"name": "  Cue  ", "radius": "12", "ignored": "not validated"},
        {"name": StringValidator(), "radius": NumericValidator()},
    )

    assert result == {"name": "Cue", "radius": 12}
    assert validate_request_data({}, {"optional": StringValidator()}) == {}


def test_validate_request_data_aggregates_field_errors_in_http_response() -> None:
    with pytest.raises(HTTPException) as raised:
        validate_request_data(
            {"name": 3, "radius": -2},
            {
                "name": StringValidator(error_code="NAME"),
                "radius": NumericValidator(min_value=0, error_code="RADIUS"),
            },
        )

    assert raised.value.status_code == 400
    assert cast("Any", raised.value.detail) == {
        "error": "VAL_001",
        "message": "Request validation failed",
        "validation_errors": [
            {
                "field": "name",
                "value": "3",
                "message": "Must be a string",
                "code": "NAME",
            },
            {
                "field": "radius",
                "value": "-2",
                "message": "Must be at least 0",
                "code": "RADIUS",
            },
        ],
    }


@pytest.mark.parametrize(
    ("value", "max_length", "expected"),
    [
        ("  cue\x00\x01 ball  ", 100, "cue ball"),
        ("abcdef", 3, "abc"),
        ("a\tb\nc\rd", 100, "a\tb\nc\rd"),
        (123, 2, "123"),
    ],
)
def test_sanitize_string_input_removes_controls_trims_and_truncates(
    value: object, max_length: int, expected: str
) -> None:
    assert sanitize_string_input(value, max_length) == expected


def test_validate_file_upload_accepts_case_insensitive_extension_and_sanitizes_name() -> (
    None
):
    result = validate_file_upload(
        "table\x00.PNG",
        "image/png",
        file_size=10,
        allowed_extensions=[".png", "jpg"],
        max_size=10,
    )

    assert result.is_valid
    assert result.errors == []
    assert result.sanitized_value == "table.PNG"


def test_validate_file_upload_accumulates_size_and_extension_errors() -> None:
    result = validate_file_upload(
        "malware.exe", "application/octet-stream", 11, ["png", ".jpg"], max_size=10
    )

    assert not result.is_valid
    assert result.sanitized_value == "malware.exe"
    assert [error["field"] for error in result.errors] == ["file_size", "filename"]
    assert [error["code"] for error in result.errors] == ["VAL_003", "VAL_003"]


@pytest.mark.parametrize("filename", ["plain", "", "archive.tar.gz"])
def test_validate_file_upload_handles_missing_or_multiple_extensions(
    filename: str,
) -> None:
    result = validate_file_upload(filename, "application/octet-stream", 0, ["gz"])

    assert result.is_valid is (filename == "archive.tar.gz")


def test_validate_file_upload_without_extension_allowlist_skips_extension_check() -> (
    None
):
    result = validate_file_upload("file.anything", "application/octet-stream", 0)

    assert result.is_valid
    assert result.sanitized_value == "file.anything"
