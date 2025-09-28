"""Configuration-related API models for data transformation."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, validator


class ConfigSourceEnum(str, Enum):
    """Configuration sources."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    USER = "user"
    RUNTIME = "runtime"


class ConfigFormatEnum(str, Enum):
    """Configuration formats."""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"


class ConfigTypeEnum(str, Enum):
    """Configuration value types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    URL = "url"


class ConfigValueModel(BaseModel):
    """Individual configuration value model."""

    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    type: ConfigTypeEnum = Field(..., description="Value type")
    source: ConfigSourceEnum = Field(
        default=ConfigSourceEnum.DEFAULT, description="Value source"
    )
    description: Optional[str] = Field(None, description="Value description")
    default_value: Optional[Any] = Field(None, description="Default value")
    is_required: bool = Field(default=False, description="Whether value is required")
    is_secret: bool = Field(
        default=False, description="Whether value contains sensitive data"
    )
    validation_rules: Optional[dict[str, Any]] = Field(
        None, description="Validation rules"
    )
    last_modified: Optional[datetime] = Field(
        None, description="Last modification time"
    )
    modified_by: Optional[str] = Field(None, description="Who last modified this value")


class ConfigSectionModel(BaseModel):
    """Configuration section model."""

    name: str = Field(..., description="Section name")
    description: Optional[str] = Field(None, description="Section description")
    values: dict[str, ConfigValueModel] = Field(
        default={}, description="Section values"
    )
    subsections: dict[str, "ConfigSectionModel"] = Field(
        default={}, description="Subsections"
    )
    is_readonly: bool = Field(default=False, description="Whether section is read-only")


class ConfigurationModel(BaseModel):
    """Complete configuration model."""

    version: str = Field(default="1.0", description="Configuration version")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Configuration timestamp"
    )
    sections: dict[str, ConfigSectionModel] = Field(
        default={}, description="Configuration sections"
    )
    metadata: dict[str, Any] = Field(default={}, description="Configuration metadata")
    checksum: Optional[str] = Field(None, description="Configuration checksum")


class ConfigProfileModel(BaseModel):
    """Configuration profile model."""

    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    configuration: ConfigurationModel = Field(..., description="Profile configuration")
    is_active: bool = Field(default=False, description="Whether profile is active")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    tags: list[str] = Field(default=[], description="Profile tags")


class ConfigChangeModel(BaseModel):
    """Configuration change record."""

    id: str = Field(..., description="Change ID")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Change timestamp"
    )
    key: str = Field(..., description="Changed configuration key")
    old_value: Optional[Any] = Field(None, description="Previous value")
    new_value: Any = Field(..., description="New value")
    source: ConfigSourceEnum = Field(..., description="Change source")
    user: Optional[str] = Field(None, description="User who made the change")
    reason: Optional[str] = Field(None, description="Reason for change")


class ConfigValidationResult(BaseModel):
    """Configuration validation result."""

    is_valid: bool = Field(..., description="Whether configuration is valid")
    errors: list[str] = Field(default=[], description="Validation errors")
    warnings: list[str] = Field(default=[], description="Validation warnings")
    checked_keys: list[str] = Field(default=[], description="Keys that were validated")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Validation timestamp"
    )


# Request models


class ConfigUpdateRequest(BaseModel):
    """Request to update a configuration value."""

    key: str = Field(..., description="Configuration key to update")
    value: Any = Field(..., description="New value")
    source: ConfigSourceEnum = Field(
        default=ConfigSourceEnum.USER, description="Update source"
    )
    reason: Optional[str] = Field(None, description="Reason for update")
    validate: bool = Field(
        default=True, description="Whether to validate the new value"
    )


class ConfigBatchUpdateRequest(BaseModel):
    """Request to update multiple configuration values."""

    updates: list[ConfigUpdateRequest] = Field(
        ..., description="List of updates to apply"
    )
    atomic: bool = Field(
        default=True, description="Whether to apply all updates atomically"
    )
    validate_all: bool = Field(
        default=True, description="Whether to validate all values"
    )
    reason: Optional[str] = Field(None, description="Reason for batch update")


class ConfigGetRequest(BaseModel):
    """Request to get configuration values."""

    keys: Optional[list[str]] = Field(
        None, description="Specific keys to retrieve (all if None)"
    )
    include_metadata: bool = Field(
        default=False, description="Include metadata in response"
    )
    include_defaults: bool = Field(default=True, description="Include default values")
    filter_secrets: bool = Field(default=True, description="Filter out secret values")


class ConfigExportRequest(BaseModel):
    """Request to export configuration."""

    format: ConfigFormatEnum = Field(
        default=ConfigFormatEnum.JSON, description="Export format"
    )
    include_secrets: bool = Field(default=False, description="Include secret values")
    include_metadata: bool = Field(default=True, description="Include metadata")
    keys: Optional[list[str]] = Field(None, description="Specific keys to export")
    filename: Optional[str] = Field(None, description="Export filename")


class ConfigImportRequest(BaseModel):
    """Request to import configuration."""

    data: Union[str, dict[str, Any]] = Field(
        ..., description="Configuration data to import"
    )
    format: ConfigFormatEnum = Field(
        default=ConfigFormatEnum.JSON, description="Data format"
    )
    merge: bool = Field(
        default=True, description="Whether to merge with existing config"
    )
    validate: bool = Field(
        default=True, description="Whether to validate imported data"
    )
    backup_current: bool = Field(
        default=True, description="Whether to backup current config"
    )


class ConfigProfileRequest(BaseModel):
    """Request to manage configuration profiles."""

    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    configuration: Optional[dict[str, Any]] = Field(
        None, description="Profile configuration"
    )
    tags: list[str] = Field(default=[], description="Profile tags")


class ConfigSearchRequest(BaseModel):
    """Request to search configuration."""

    query: str = Field(..., description="Search query")
    search_keys: bool = Field(default=True, description="Search in keys")
    search_values: bool = Field(default=True, description="Search in values")
    search_descriptions: bool = Field(
        default=True, description="Search in descriptions"
    )
    case_sensitive: bool = Field(default=False, description="Case sensitive search")
    regex: bool = Field(default=False, description="Use regex search")


class ConfigValidationRequest(BaseModel):
    """Request to validate configuration."""

    keys: Optional[list[str]] = Field(None, description="Specific keys to validate")
    strict: bool = Field(default=False, description="Strict validation mode")
    check_dependencies: bool = Field(
        default=True, description="Check configuration dependencies"
    )


# Response models


class ConfigValueResponse(BaseModel):
    """Configuration value response."""

    success: bool = Field(default=True, description="Operation success")
    value: ConfigValueModel = Field(..., description="Configuration value")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigurationResponse(BaseModel):
    """Configuration response."""

    success: bool = Field(default=True, description="Operation success")
    configuration: ConfigurationModel = Field(..., description="Configuration data")
    total_keys: int = Field(..., description="Total number of configuration keys")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigUpdateResponse(BaseModel):
    """Configuration update response."""

    success: bool = Field(default=True, description="Operation success")
    updated_keys: list[str] = Field(..., description="Successfully updated keys")
    failed_keys: list[str] = Field(default=[], description="Failed to update keys")
    validation_results: Optional[ConfigValidationResult] = Field(
        None, description="Validation results"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigExportResponse(BaseModel):
    """Configuration export response."""

    success: bool = Field(default=True, description="Operation success")
    data: Union[str, dict[str, Any]] = Field(
        ..., description="Exported configuration data"
    )
    format: ConfigFormatEnum = Field(..., description="Export format")
    filename: Optional[str] = Field(None, description="Export filename")
    size: int = Field(..., description="Export data size in bytes")
    key_count: int = Field(..., description="Number of exported keys")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigImportResponse(BaseModel):
    """Configuration import response."""

    success: bool = Field(default=True, description="Operation success")
    imported_keys: list[str] = Field(..., description="Successfully imported keys")
    skipped_keys: list[str] = Field(default=[], description="Skipped keys")
    failed_keys: list[str] = Field(default=[], description="Failed to import keys")
    validation_results: Optional[ConfigValidationResult] = Field(
        None, description="Validation results"
    )
    backup_file: Optional[str] = Field(None, description="Backup file path")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigProfileResponse(BaseModel):
    """Configuration profile response."""

    success: bool = Field(default=True, description="Operation success")
    profile: ConfigProfileModel = Field(..., description="Configuration profile")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigProfileListResponse(BaseModel):
    """Configuration profile list response."""

    success: bool = Field(default=True, description="Operation success")
    profiles: list[ConfigProfileModel] = Field(..., description="Available profiles")
    total_profiles: int = Field(..., description="Total number of profiles")
    active_profile: Optional[str] = Field(None, description="Currently active profile")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigSearchResponse(BaseModel):
    """Configuration search response."""

    success: bool = Field(default=True, description="Operation success")
    results: list[ConfigValueModel] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Search query used")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigValidationResponse(BaseModel):
    """Configuration validation response."""

    success: bool = Field(default=True, description="Operation success")
    validation_result: ConfigValidationResult = Field(
        ..., description="Validation results"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ConfigHistoryResponse(BaseModel):
    """Configuration change history response."""

    success: bool = Field(default=True, description="Operation success")
    changes: list[ConfigChangeModel] = Field(..., description="Configuration changes")
    total_changes: int = Field(..., description="Total number of changes")
    from_time: Optional[datetime] = Field(None, description="History start time")
    to_time: Optional[datetime] = Field(None, description="History end time")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


# Validators


@validator("key", pre=True)
def validate_config_key(cls, v):
    """Validate configuration key format."""
    if not isinstance(v, str):
        raise ValueError("Configuration key must be a string")
    if not v.strip():
        raise ValueError("Configuration key cannot be empty")
    # Allow dots for hierarchical keys
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not all(c in allowed_chars for c in v):
        raise ValueError("Configuration key contains invalid characters")
    return v.strip()


@validator("value")
def validate_config_value(cls, v, values):
    """Validate configuration value based on type."""
    if "type" in values:
        config_type = values["type"]
        if config_type == ConfigTypeEnum.STRING and not isinstance(v, str):
            raise ValueError("Value must be a string")
        elif config_type == ConfigTypeEnum.INTEGER and not isinstance(v, int):
            raise ValueError("Value must be an integer")
        elif config_type == ConfigTypeEnum.FLOAT and not isinstance(v, (int, float)):
            raise ValueError("Value must be a number")
        elif config_type == ConfigTypeEnum.BOOLEAN and not isinstance(v, bool):
            raise ValueError("Value must be a boolean")
        elif config_type == ConfigTypeEnum.LIST and not isinstance(v, list):
            raise ValueError("Value must be a list")
        elif config_type == ConfigTypeEnum.DICT and not isinstance(v, dict):
            raise ValueError("Value must be a dictionary")
    return v


# Update forward references
ConfigSectionModel.update_forward_refs()


__all__ = [
    # Enums
    "ConfigSourceEnum",
    "ConfigFormatEnum",
    "ConfigTypeEnum",
    # Data models
    "ConfigValueModel",
    "ConfigSectionModel",
    "ConfigurationModel",
    "ConfigProfileModel",
    "ConfigChangeModel",
    "ConfigValidationResult",
    # Request models
    "ConfigUpdateRequest",
    "ConfigBatchUpdateRequest",
    "ConfigGetRequest",
    "ConfigExportRequest",
    "ConfigImportRequest",
    "ConfigProfileRequest",
    "ConfigSearchRequest",
    "ConfigValidationRequest",
    # Response models
    "ConfigValueResponse",
    "ConfigurationResponse",
    "ConfigUpdateResponse",
    "ConfigExportResponse",
    "ConfigImportResponse",
    "ConfigProfileResponse",
    "ConfigProfileListResponse",
    "ConfigSearchResponse",
    "ConfigValidationResponse",
    "ConfigHistoryResponse",
]
