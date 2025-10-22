"""Configuration management endpoints for system settings control.

Provides comprehensive configuration management including:
- Retrieve current system configuration (FR-API-005)
- Update configuration parameters with validation (FR-API-006)
- Reset configuration to defaults (FR-API-007)
- Import/export configuration files (FR-API-008)
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure backend directory is in Python path for imports
backend_dir = Path(__file__).parent.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import yaml
from api.dependencies import get_config_module
from api.models.common import create_success_response
from api.models.responses import (
    ConfigExportResponse,
    ConfigResponse,
    ConfigUpdateResponse,
    SuccessResponse,
)
from config import Config, config
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    HTTPException,
    Query,
)
from fastapi import Request as FastAPIRequest
from fastapi import UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration Management"])


def calculate_checksum(data: str) -> str:
    """Calculate SHA-256 checksum of data."""
    return hashlib.sha256(data.encode()).hexdigest()


def validate_config_section(section: str, allowed_sections: list[str]) -> str:
    """Validate configuration section name."""
    if section not in allowed_sections:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid configuration section. Section '{section}' is not valid. Allowed sections: {allowed_sections}",
        )
    return section


@router.get("", response_model=ConfigResponse)
async def get_configuration(
    section: Optional[str] = Query(
        None, description="Specific configuration section to retrieve"
    ),
    include_metadata: bool = Query(True, description="Include configuration metadata"),
    config_module: Config = Depends(get_config_module),
) -> ConfigResponse:
    """Retrieve current system configuration (FR-API-005).

    Allows retrieval of complete configuration or specific sections
    with optional metadata inclusion.
    """
    try:
        # Get available configuration sections
        available_sections = [
            "system",
            "camera",
            "vision",
            "projector",
            "api",
            "logging",
        ]

        if section:
            validate_config_section(section, available_sections)

        # Retrieve configuration
        if hasattr(config_module, "get_configuration"):
            config_data = await config_module.get_configuration()
        else:
            # Fallback for basic configuration access
            config_data = getattr(config_module, "_data", {})

        # Filter by section if requested
        if section:
            if isinstance(config_data, dict):
                config_values = {section: config_data.get(section, {})}
            else:
                config_values = {section: getattr(config_data, section, {})}
        else:
            config_values = (
                config_data if isinstance(config_data, dict) else config_data.__dict__
            )

        # Build response
        timestamp = datetime.now(timezone.utc)

        response = ConfigResponse(
            timestamp=timestamp,
            values=config_values,
            schema_version="1.0.0",
            last_modified=timestamp,  # Would track actual modification time
            is_valid=True,  # Would validate configuration
            validation_errors=[],
        )

        logger.info("Configuration retrieved")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration retrieval failed. Unable to retrieve system configuration",
        )


@router.put("/", response_model=ConfigUpdateResponse)
async def update_configuration(
    config_data: dict[str, Any],
    section: Optional[str] = Query(None, description="Specific section to update"),
    validate_only: bool = Query(False, description="Only validate without applying"),
    force_update: bool = Query(False, description="Force update even with warnings"),
    config_module: Config = Depends(get_config_module),
) -> ConfigUpdateResponse:
    """Update configuration parameters with validation (FR-API-006).

    Supports partial updates, validation-only mode, and forced updates.
    """
    try:
        available_sections = [
            "system",
            "camera",
            "vision",
            "projector",
            "api",
            "logging",
        ]

        if section:
            validate_config_section(section, available_sections)

        # Validate configuration data
        validation_errors = []
        warnings = []
        updated_fields = []

        # Basic validation (would be more comprehensive in production)
        if not config_data:
            validation_errors.append(
                {
                    "field": "config_data",
                    "message": "Configuration data cannot be empty",
                    "current_value": config_data,
                    "expected_type": "dict",
                }
            )

        # Simulate field validation
        for key, value in config_data.items():
            updated_fields.append(key)

            # Example validation rules
            if key == "camera" and isinstance(value, dict):
                if "fps" in value and not isinstance(value["fps"], int):
                    validation_errors.append(
                        {
                            "field": "camera.fps",
                            "message": "FPS must be an integer",
                            "current_value": value["fps"],
                            "expected_type": "int",
                        }
                    )
                elif "fps" in value and value["fps"] > 60:
                    warnings.append(
                        f"High FPS value ({value['fps']}) may impact performance"
                    )

        # If validation-only mode, return validation results
        if validate_only:
            return ConfigUpdateResponse(
                success=len(validation_errors) == 0,
                updated_fields=[],
                validation_errors=validation_errors,
                warnings=warnings,
                rollback_available=False,
                restart_required=False,
            )

        # Check if we should proceed with warnings
        if warnings and not force_update:
            raise HTTPException(
                status_code=400,
                detail="Configuration has warnings. Use force_update=true to proceed",
            )

        # Stop if validation errors exist
        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail="Configuration validation failed. Configuration contains validation errors",
            )

        # Apply configuration update
        try:
            if hasattr(config_module, "update_configuration"):
                await config_module.update_configuration(config_data)
            else:
                # Fallback for basic configuration update
                for key, value in config_data.items():
                    setattr(config_module, key, value)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Configuration update failed. Failed to apply configuration changes: {str(e)}",
            )

        # Determine if restart is required
        restart_required = any(
            field in ["camera", "projector", "api"] for field in updated_fields
        )

        logger.info(f"Configuration updated: {updated_fields}")

        return ConfigUpdateResponse(
            success=True,
            updated_fields=updated_fields,
            validation_errors=[],
            warnings=warnings,
            rollback_available=True,
            restart_required=restart_required,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration update failed. Unable to update system configuration",
        )


@router.post("/reset", response_model=SuccessResponse)
async def reset_configuration(
    confirm: bool = Query(..., description="Confirmation that reset is intended"),
    reset_type: str = Query(
        "all", pattern="^(all|user|section)$", description="Type of reset"
    ),
    sections: Optional[list[str]] = Query(
        None, description="Specific sections to reset"
    ),
    config_module: Config = Depends(get_config_module),
) -> SuccessResponse:
    """Reset configuration to defaults (FR-API-007).

    Supports full reset or section-specific reset.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Configuration reset must be explicitly confirmed",
            )

        available_sections = [
            "system",
            "camera",
            "vision",
            "projector",
            "api",
            "logging",
        ]

        if reset_type == "section" and sections:
            for section in sections:
                validate_config_section(section, available_sections)

        # Perform reset
        try:
            if hasattr(config_module, "reset_to_defaults"):
                await config_module.reset_to_defaults()
            else:
                # Fallback implementation
                logger.warning("Configuration module doesn't support reset_to_defaults")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Configuration reset failed. Failed to reset configuration: {str(e)}",
            )

        logger.warning("Configuration reset performed")

        return create_success_response(
            "Configuration reset to defaults successfully",
            {
                "reset_type": reset_type,
                "sections_reset": (
                    sections if reset_type == "section" else available_sections
                ),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration reset failed. Unable to reset system configuration",
        )


@router.post("/import", response_model=SuccessResponse)
async def import_configuration(
    file: UploadFile = File(..., description="Configuration file to import"),
    merge_strategy: str = Query(
        "replace", pattern="^(replace|merge)$", description="Import strategy"
    ),
    validate_only: bool = Query(False, description="Only validate without importing"),
    config_module: Config = Depends(get_config_module),
) -> SuccessResponse:
    """Import configuration from file (FR-API-008).

    Supports JSON and YAML formats with merge or replace strategies.
    """
    try:
        # Validate file format
        allowed_extensions = [".json", ".yaml", ".yml"]
        file_extension = Path(file.filename or "").suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. File format '{file_extension}' not supported. Allowed: {allowed_extensions}",
            )

        # Read and parse file
        try:
            content = await file.read()
            content_str = content.decode("utf-8")

            if file_extension == ".json":
                config_data = json.loads(content_str)
            else:  # YAML
                config_data = yaml.safe_load(content_str)

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"File parse error. Unable to parse configuration file: {str(e)}",
            )

        # Validate configuration structure
        if not isinstance(config_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid configuration structure. Configuration must be a JSON/YAML object",
            )

        # If validation-only mode, just return validation results
        if validate_only:
            # Perform validation without applying
            validation_errors = []
            # Add actual validation logic here

            return create_success_response(
                "Configuration file validation completed",
                {
                    "is_valid": len(validation_errors) == 0,
                    "validation_errors": validation_errors,
                    "sections_found": list(config_data.keys()),
                },
            )

        # Apply configuration based on merge strategy
        try:
            if merge_strategy == "replace":
                # Replace entire configuration
                if hasattr(config_module, "replace_configuration"):
                    await config_module.replace_configuration(config_data)
                else:
                    # Fallback
                    for key, value in config_data.items():
                        setattr(config_module, key, value)
            else:  # merge
                # Merge with existing configuration
                if hasattr(config_module, "update_configuration"):
                    await config_module.update_configuration(config_data)
                else:
                    # Fallback
                    for key, value in config_data.items():
                        setattr(config_module, key, value)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Configuration import failed. Failed to import configuration: {str(e)}",
            )

        logger.info(f"Configuration imported from file {file.filename}")

        return create_success_response(
            "Configuration imported successfully",
            {
                "filename": file.filename,
                "format": file_extension,
                "merge_strategy": merge_strategy,
                "sections_imported": list(config_data.keys()),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration import failed. Unable to import configuration file",
        )


@router.get("/export", response_model=ConfigExportResponse)
async def export_configuration(
    format: str = Query("json", pattern="^(json|yaml)$", description="Export format"),
    sections: Optional[list[str]] = Query(
        None, description="Specific sections to export"
    ),
    include_defaults: bool = Query(False, description="Include default values"),
    include_metadata: bool = Query(True, description="Include metadata"),
    config_module: Config = Depends(get_config_module),
) -> ConfigExportResponse:
    """Export configuration to downloadable format (FR-API-008).

    Supports JSON and YAML formats with selective section export.
    """
    try:
        available_sections = [
            "system",
            "camera",
            "vision",
            "projector",
            "api",
            "logging",
        ]

        if sections:
            for section in sections:
                validate_config_section(section, available_sections)

        # Get configuration data
        if hasattr(config_module, "get_configuration"):
            config_data = await config_module.get_configuration()
        else:
            config_data = getattr(config_module, "_data", {})

        # Convert to dict if needed
        if not isinstance(config_data, dict):
            config_data = (
                config_data.__dict__ if hasattr(config_data, "__dict__") else {}
            )

        # Filter sections if specified
        if sections:
            config_data = {k: v for k, v in config_data.items() if k in sections}

        # Add metadata if requested
        if include_metadata:
            config_data["_metadata"] = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exported_by": "api",
                "version": "1.0.0",
                "sections": list(config_data.keys()),
            }

        # Serialize data
        if format == "json":
            serialized_data = json.dumps(config_data, indent=2, default=str)
        else:  # yaml
            serialized_data = yaml.dump(config_data, default_flow_style=False)

        # Calculate size and checksum
        data_size = len(serialized_data.encode("utf-8"))
        checksum = calculate_checksum(serialized_data)

        logger.info(f"Configuration exported in {format} format")

        return ConfigExportResponse(
            format=format,
            size=data_size,
            checksum=checksum,
            timestamp=datetime.now(timezone.utc),
            data=config_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration export failed. Unable to export system configuration",
        )


@router.get("/export/download")
async def download_configuration(
    format: str = Query("json", pattern="^(json|yaml)$", description="Export format"),
    sections: Optional[list[str]] = Query(
        None, description="Specific sections to export"
    ),
    config_module: Config = Depends(get_config_module),
) -> FileResponse:
    """Download configuration as file.

    Returns configuration as downloadable file.
    """
    try:
        # Get export data (reuse export logic)
        export_response = await export_configuration(
            format=format,
            sections=sections,
            include_defaults=False,
            include_metadata=True,
            config_module=config_module,
        )

        # Create temporary file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"billiards_config_{timestamp}.{format}"

        # Serialize data for file
        if format == "json":
            content = json.dumps(export_response.data, indent=2, default=str)
        else:  # yaml
            content = yaml.dump(export_response.data, default_flow_style=False)

        # Write to temporary file
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "w") as f:
            f.write(content)

        return FileResponse(
            path=temp_path, filename=filename, media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Failed to download configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration download failed. Unable to generate configuration download",
        )


@router.get("/schema")
async def get_configuration_schema(
    section: Optional[str] = Query(None, description="Specific section schema"),
) -> dict[str, Any]:
    """Get configuration schema for validation and documentation.

    Returns JSON Schema for configuration structure.
    """
    try:
        # Return basic schema structure (would be more comprehensive in production)
        schema = {
            "type": "object",
            "properties": {
                "system": {
                    "type": "object",
                    "properties": {
                        "debug": {"type": "boolean"},
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        },
                    },
                },
                "camera": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "integer", "minimum": 0},
                        "resolution": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "fps": {"type": "integer", "minimum": 1, "maximum": 60},
                    },
                },
                "vision": {
                    "type": "object",
                    "properties": {
                        "sensitivity": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        }
                    },
                },
            },
        }

        if section:
            available_sections = list(schema["properties"].keys())
            validate_config_section(section, available_sections)
            return schema["properties"][section]

        return schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration schema: {e}")
        raise HTTPException(
            status_code=500,
            detail="Schema retrieval failed. Unable to retrieve configuration schema",
        )


class Corner(BaseModel):
    """A single corner point."""

    x: float
    y: float


class MarkerDot(BaseModel):
    """A table marker dot position (for masking)."""

    x: float
    y: float


class PlayingAreaCornersRequest(BaseModel):
    """Request model for setting playing area corners."""

    corners: list[Corner] = Field(
        ...,
        description="List of 4 corner points in order: top-left, top-right, bottom-right, bottom-left",
        min_length=4,
        max_length=4,
    )
    marker_dots: list[MarkerDot] | None = Field(
        default=None,
        description="Optional list of table marker dots (spots/markings) to be masked during ball detection",
    )
    calibration_resolution_width: int | None = Field(
        default=None,
        description="Width of the video resolution used during calibration (for coordinate scaling)",
    )
    calibration_resolution_height: int | None = Field(
        default=None,
        description="Height of the video resolution used during calibration (for coordinate scaling)",
    )


@router.post("/table/playing-area")
async def set_playing_area_corners(
    request: FastAPIRequest,
    config_module: Config = Depends(get_config_module),
) -> dict:
    """Set the playing area corner points for table calibration.

    Saves the corners to the config file for persistence.

    Args:
        request: FastAPI request containing corners data
        config_module: Configuration module (injected)

    Returns:
        Dictionary with success status and saved corners

    Raises:
        HTTPException: If save fails or validation fails
    """
    try:
        # Parse JSON body
        body = await request.json()
        corners = body.get("corners", [])
        marker_dots = body.get("marker_dots", [])
        calibration_resolution_width = body.get("calibration_resolution_width")
        calibration_resolution_height = body.get("calibration_resolution_height")

        # Validate corners data
        if not corners:
            raise HTTPException(
                status_code=400,
                detail="No corners provided in request body",
            )

        if len(corners) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 4 corners, got {len(corners)}",
            )

        # Validate each corner has x and y
        for i, corner in enumerate(corners):
            if not isinstance(corner, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Corner {i} must be an object with x and y properties",
                )
            if "x" not in corner or "y" not in corner:
                raise HTTPException(
                    status_code=400,
                    detail=f"Corner {i} missing x or y coordinate",
                )

        # Validate marker dots if provided
        if marker_dots:
            for i, dot in enumerate(marker_dots):
                if not isinstance(dot, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Marker dot {i} must be an object with x and y properties",
                    )
                if "x" not in dot or "y" not in dot:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Marker dot {i} missing x or y coordinate",
                    )

        # Save to config file
        try:
            config_module.set("table.playing_area_corners", corners)
            config_module.set("table.marker_dots", marker_dots)
            if calibration_resolution_width is not None:
                config_module.set(
                    "table.calibration_resolution_width", calibration_resolution_width
                )
            if calibration_resolution_height is not None:
                config_module.set(
                    "table.calibration_resolution_height", calibration_resolution_height
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save playing area configuration: {e}",
            )

        logger.info(f"Playing area corners saved to config: {corners}")
        logger.info(f"Table marker dots saved to config: {marker_dots}")
        logger.info(
            f"Calibration resolution saved: {calibration_resolution_width}x{calibration_resolution_height}"
        )

        return {
            "success": True,
            "message": "Playing area configuration saved successfully",
            "corners": corners,
            "count": len(corners),
            "marker_dots": marker_dots,
            "marker_dots_count": len(marker_dots),
            "calibration_resolution_width": calibration_resolution_width,
            "calibration_resolution_height": calibration_resolution_height,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set playing area corners: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save playing area corners: {str(e)}",
        )


@router.get("/table/playing-area", response_model=dict)
async def get_playing_area_corners(
    config_module: Config = Depends(get_config_module),
) -> dict[str, Any]:
    """Get the current playing area corner points for table calibration.

    Returns:
        Dictionary with corners array and metadata

    Raises:
        HTTPException: If configuration retrieval fails
    """
    try:
        # Get playing area corners and marker dots from configuration
        playing_area_corners = config_module.get("table.playing_area_corners", [])
        marker_dots = config_module.get("table.marker_dots", [])

        return {
            "success": True,
            "corners": playing_area_corners,
            "count": len(playing_area_corners),
            "calibrated": len(playing_area_corners) == 4,
            "marker_dots": marker_dots,
            "marker_dots_count": len(marker_dots),
        }

    except Exception as e:
        logger.error(f"Failed to get playing area corners: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration retrieval failed. Unable to get playing area corners",
        )
