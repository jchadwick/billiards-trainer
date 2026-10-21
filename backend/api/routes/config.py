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
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Protocol, cast

from fastapi import APIRouter, Depends, File, HTTPException, Query
from fastapi import Request as FastAPIRequest
from fastapi import UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from backend.api.dependencies import get_config_module
from backend.api.models.common import PositionWithScale, create_success_response
from backend.api.models.responses import (
    ConfigExportResponse,
    ConfigResponse,
    ConfigUpdateResponse,
)
from backend.api.models.responses import (
    ConfigValidationError as APIConfigValidationError,
)
from backend.api.models.responses import SuccessResponse
from backend.config import Config, ConfigValidationError
from backend.config import get_configuration_schema as build_configuration_schema
from backend.config import load_schema_reference, validate_configuration
from backend.core.constants_4k import CANONICAL_HEIGHT, CANONICAL_WIDTH


class _YamlLibrary(Protocol):
    """Typed surface used from PyYAML without relying on external stubs."""

    def safe_load(self, stream: str) -> Any: ...

    def dump(self, data: Any, **kwargs: Any) -> str: ...


yaml = cast("_YamlLibrary", import_module("yaml"))

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


def _configuration_sections() -> list[str]:
    """Return sections from the same runtime defaults used by validation."""
    return list(load_schema_reference())


def _normalize_section_update(
    config_data: dict[str, Any], section: Optional[str]
) -> dict[str, Any]:
    if not section:
        return config_data
    if len(config_data) == 1 and section in config_data:
        value = config_data[section]
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=422, detail="Section update must be an object"
            )
        return {section: value}
    return {section: config_data}


def _restart_required(updated_fields: list[str]) -> bool:
    restart_prefixes = (
        "perception.",
        "vision.",
        "core.",
        "integration.",
        "camera.",
        "cue.",
        "projector.",
        "api.server.",
        "api.network.",
        "api.middleware.",
        "api.rate_limiting.",
        "api.websocket.",
        "streaming.",
    )
    return any(field.startswith(restart_prefixes) for field in updated_fields)


def _validation_detail(errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {"message": "Configuration validation failed", "errors": errors}


def _response_validation_errors(
    errors: list[dict[str, Any]],
) -> list[APIConfigValidationError]:
    """Map store validation records into the documented response schema."""
    return [APIConfigValidationError(**error) for error in errors]


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
        available_sections = _configuration_sections()

        if section:
            validate_config_section(section, available_sections)

        config_data = config_module.get_all()

        # Filter by section if requested
        if section:
            config_values = {section: config_data.get(section, {})}
        else:
            config_values = config_data

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
        available_sections = _configuration_sections()

        if section:
            validate_config_section(section, available_sections)

        if not config_data:
            raise HTTPException(
                status_code=422, detail="Configuration data cannot be empty"
            )

        updates = _normalize_section_update(config_data, section)
        current = config_module.get_all()
        validation_errors, warnings = validate_configuration(updates, baseline=current)

        # If validation-only mode, return validation results
        if validate_only:
            return ConfigUpdateResponse(
                success=len(validation_errors) == 0,
                updated_fields=[],
                validation_errors=_response_validation_errors(validation_errors),
                warnings=warnings,
                rollback_available=False,
                restart_required=False,
            )

        # Stop if validation errors exist
        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail=_validation_detail(validation_errors),
            )

        # Warnings can only be bypassed after all errors are clear.
        if warnings and not force_update:
            raise HTTPException(
                status_code=400,
                detail="Configuration has warnings. Use force_update=true to proceed",
            )

        # Apply configuration update
        try:
            updated_fields = config_module.apply_updates(updates)
        except ConfigValidationError as exc:
            raise HTTPException(
                status_code=422, detail=_validation_detail(exc.errors)
            ) from exc
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Configuration update failed. Changes were not persisted.",
            )

        restart_required = _restart_required(updated_fields)

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
) -> dict[str, Any]:
    """Reset configuration to defaults (FR-API-007).

    Supports full reset or section-specific reset.
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Configuration reset must be explicitly confirmed",
            )

        available_sections = _configuration_sections()

        if reset_type == "section" and sections:
            for section in sections:
                validate_config_section(section, available_sections)

        # Reset through the same validated atomic store operation used by
        # updates and imports. The existing `user` mode historically reset the
        # complete store because configuration has no user-only namespace.
        try:
            defaults = load_schema_reference()
            if reset_type == "section":
                if not sections:
                    raise HTTPException(
                        status_code=422,
                        detail="At least one section is required for a section reset",
                    )
                replacement = config_module.get_all()
                for name in sections:
                    replacement[name] = defaults[name]
                config_module.replace_configuration(replacement)
            else:
                config_module.replace_configuration(defaults)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Configuration reset failed. Changes were not persisted.",
            ) from e

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
    force_update: bool = Query(
        False, description="Proceed when validation reports warnings"
    ),
    config_module: Config = Depends(get_config_module),
) -> dict[str, Any]:
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

        # Exports add this transport-only envelope; it is not a runtime key.
        config_data.pop("_metadata", None)
        current = config_module.get_all()
        validation_base = current if merge_strategy == "merge" else {}
        validation_errors, warnings = validate_configuration(
            config_data, baseline=validation_base
        )

        if validate_only:
            return create_success_response(
                "Configuration file validation completed",
                {
                    "is_valid": not validation_errors,
                    "validation_errors": validation_errors,
                    "warnings": warnings,
                    "sections_found": list(config_data.keys()),
                },
            )

        if validation_errors:
            raise HTTPException(
                status_code=422, detail=_validation_detail(validation_errors)
            )
        if warnings and not force_update:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Configuration has warnings. Use force_update=true to proceed",
                    "warnings": warnings,
                },
            )

        try:
            if merge_strategy == "replace":
                imported_fields = config_module.replace_configuration(config_data)
            else:  # merge
                imported_fields = config_module.apply_updates(config_data)
        except ConfigValidationError as exc:
            raise HTTPException(
                status_code=422, detail=_validation_detail(exc.errors)
            ) from exc
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Configuration import failed. Changes were not persisted.",
            )

        logger.info(f"Configuration imported from file {file.filename}")

        return create_success_response(
            "Configuration imported successfully",
            {
                "filename": file.filename,
                "format": file_extension,
                "merge_strategy": merge_strategy,
                "sections_imported": list(config_data.keys()),
                "updated_fields": imported_fields,
                "warnings": warnings,
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
        available_sections = _configuration_sections()

        if sections:
            for section in sections:
                validate_config_section(section, available_sections)

        config_data = config_module.get_all()

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
        if section:
            validate_config_section(section, _configuration_sections())
        return build_configuration_schema(section)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration schema: {e}")
        raise HTTPException(
            status_code=500,
            detail="Schema retrieval failed. Unable to retrieve configuration schema",
        )


class Corner(PositionWithScale):
    """A table corner in its source image space with scale-to-4K metadata."""

    model_config = ConfigDict(extra="forbid")


class MarkerDot(PositionWithScale):
    """A table marker dot in its source image space with explicit scale."""

    model_config = ConfigDict(extra="forbid")


class PlayingAreaCornersRequest(BaseModel):
    """Request model for setting playing area corners."""

    model_config = ConfigDict(extra="forbid")

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
        gt=0,
        description="Width of the video resolution used during calibration (for coordinate scaling)",
    )
    calibration_resolution_height: int | None = Field(
        default=None,
        gt=0,
        description="Height of the video resolution used during calibration (for coordinate scaling)",
    )

    @model_validator(mode="after")
    def validate_declared_scale(self) -> "PlayingAreaCornersRequest":
        """Keep scale metadata consistent with the declared source resolution."""
        width = self.calibration_resolution_width
        height = self.calibration_resolution_height
        if (width is None) != (height is None):
            raise ValueError(
                "Calibration resolution width and height must be provided together"
            )
        if width is None or height is None:
            return self

        expected_scale = (CANONICAL_WIDTH / width, CANONICAL_HEIGHT / height)
        for point in [*self.corners, *(self.marker_dots or [])]:
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(point.scale, expected_scale)
            ):
                raise ValueError(
                    "Point scale must map the declared calibration resolution to canonical 4K"
                )
        return self


@router.post("/table/playing-area")
async def set_playing_area_corners(
    request: FastAPIRequest,
    config_module: Config = Depends(get_config_module),
) -> dict[str, Any]:
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
        try:
            payload = PlayingAreaCornersRequest.model_validate(await request.json())
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=exc.errors(include_context=False)
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Malformed JSON request body"
            ) from exc

        corners = [point.model_dump() for point in payload.corners]
        marker_dots = (
            [point.model_dump() for point in payload.marker_dots]
            if payload.marker_dots is not None
            else None
        )
        calibration_resolution_width = payload.calibration_resolution_width
        calibration_resolution_height = payload.calibration_resolution_height
        baseline = config_module.get_all()
        table_baseline = baseline.get("table", {})
        effective_width = calibration_resolution_width or table_baseline.get(
            "calibration_resolution_width"
        )
        effective_height = calibration_resolution_height or table_baseline.get(
            "calibration_resolution_height"
        )
        if effective_width is not None or effective_height is not None:
            merged_payload = payload.model_dump()
            if payload.marker_dots is None:
                merged_payload["marker_dots"] = table_baseline.get("marker_dots")
            merged_payload["calibration_resolution_width"] = effective_width
            merged_payload["calibration_resolution_height"] = effective_height
            try:
                PlayingAreaCornersRequest.model_validate(merged_payload)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422, detail=exc.errors(include_context=False)
                ) from exc

        updates: dict[str, Any] = {"table": {"playing_area_corners": corners}}
        if marker_dots is not None:
            updates["table"]["marker_dots"] = marker_dots
        if calibration_resolution_width is not None:
            updates["table"][
                "calibration_resolution_width"
            ] = calibration_resolution_width
        if calibration_resolution_height is not None:
            updates["table"][
                "calibration_resolution_height"
            ] = calibration_resolution_height

        try:
            validation_errors, _ = validate_configuration(updates, baseline=baseline)
            if validation_errors:
                raise HTTPException(
                    status_code=422, detail=_validation_detail(validation_errors)
                )
            config_module.apply_updates(updates)
        except ConfigValidationError as exc:
            raise HTTPException(
                status_code=422, detail=_validation_detail(exc.errors)
            ) from exc
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Failed to persist playing area configuration.",
            ) from e

        marker_dots = config_module.get("table.marker_dots", [])
        calibration_resolution_width = config_module.get(
            "table.calibration_resolution_width"
        )
        calibration_resolution_height = config_module.get(
            "table.calibration_resolution_height"
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
        calibration_resolution_width = config_module.get(
            "table.calibration_resolution_width"
        )
        calibration_resolution_height = config_module.get(
            "table.calibration_resolution_height"
        )

        return {
            "success": True,
            "corners": playing_area_corners,
            "count": len(playing_area_corners),
            "calibrated": len(playing_area_corners) == 4,
            "marker_dots": marker_dots,
            "marker_dots_count": len(marker_dots),
            "calibration_resolution_width": calibration_resolution_width,
            "calibration_resolution_height": calibration_resolution_height,
        }

    except Exception as e:
        logger.error(f"Failed to get playing area corners: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration retrieval failed. Unable to get playing area corners",
        )
