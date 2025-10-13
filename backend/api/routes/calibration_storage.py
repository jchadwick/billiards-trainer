"""Calibration storage endpoints for persistent table calibration data.

Provides REST API endpoints for:
- Saving table calibration data to database
- Retrieving stored calibration data
- Managing default calibration settings
- Listing historical calibrations
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.calibration_db import (
    TableCalibration,
    delete_calibration,
    get_calibration_by_id,
    get_calibration_by_session_id,
    get_default_calibration,
    get_latest_calibration,
    list_calibrations,
    save_calibration,
    set_default_calibration,
    update_calibration,
)
from ..models.common import create_success_response
from ..models.responses import SuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calibration/storage", tags=["Calibration Storage"])


# =============================================================================
# Request/Response Models
# =============================================================================


class SaveCalibrationRequest(BaseModel):
    """Request model for saving calibration data."""

    session_id: str = Field(..., description="Calibration session identifier")
    calibration_type: str = Field(
        default="standard", description="Type of calibration (quick/standard/advanced)"
    )
    status: str = Field(default="applied", description="Calibration status")
    points: list[dict[str, Any]] = Field(..., description="Calibration points data")
    transformation_matrix: Optional[list[list[float]]] = Field(
        None, description="3x3 homography transformation matrix"
    )
    accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Accuracy score"
    )
    max_error: Optional[float] = Field(None, ge=0.0, description="Maximum error")
    mean_error: Optional[float] = Field(None, ge=0.0, description="Mean error")
    rms_error: Optional[float] = Field(None, ge=0.0, description="RMS error")
    is_default: bool = Field(default=False, description="Set as default calibration")
    notes: Optional[str] = Field(None, description="Additional notes")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class CalibrationDataResponse(BaseModel):
    """Response model for calibration data."""

    success: bool = Field(default=True, description="Operation success")
    id: int = Field(..., description="Database ID")
    session_id: str = Field(..., description="Session identifier")
    calibration_type: str = Field(..., description="Calibration type")
    status: str = Field(..., description="Calibration status")
    created_at: str = Field(..., description="Creation timestamp")
    applied_at: Optional[str] = Field(None, description="Application timestamp")
    points_captured: int = Field(..., description="Number of captured points")
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    transformation_matrix: Optional[list[list[float]]] = Field(
        None, description="Transformation matrix"
    )
    is_default: bool = Field(..., description="Is default calibration")


class CalibrationListResponse(BaseModel):
    """Response model for listing calibrations."""

    success: bool = Field(default=True, description="Operation success")
    calibrations: list[dict[str, Any]] = Field(..., description="List of calibrations")
    total: int = Field(..., description="Total number of calibrations")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/save", response_model=CalibrationDataResponse, status_code=201)
async def save_calibration_data(
    request: SaveCalibrationRequest,
    db: Session = Depends(get_db),
) -> CalibrationDataResponse:
    """Save table calibration data to database.

    Persists calibration session data including transformation matrices,
    calibration points, and quality metrics.

    Args:
        request: Calibration data to save
        db: Database session

    Returns:
        Saved calibration data with database ID

    Raises:
        HTTPException: If save operation fails

    Example:
        POST /api/v1/vision/calibration/storage/save
        {
            "session_id": "cal_20231013_120000_abc123",
            "calibration_type": "standard",
            "status": "applied",
            "points": [...],
            "transformation_matrix": [[...], [...], [...]],
            "accuracy": 0.95,
            "is_default": true
        }
    """
    try:
        # Check if calibration with this session_id already exists
        existing = get_calibration_by_session_id(db, request.session_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Calibration with session_id '{request.session_id}' already exists. Use update endpoint to modify.",
            )

        # Validate transformation matrix dimensions if provided
        if request.transformation_matrix:
            if len(request.transformation_matrix) != 3:
                raise HTTPException(
                    status_code=400,
                    detail="Transformation matrix must be 3x3",
                )
            for row in request.transformation_matrix:
                if len(row) != 3:
                    raise HTTPException(
                        status_code=400,
                        detail="Transformation matrix must be 3x3",
                    )

        # Validate points data
        if not request.points or len(request.points) < 4:
            raise HTTPException(
                status_code=400,
                detail="At least 4 calibration points required",
            )

        # Create calibration model
        now = datetime.now(timezone.utc)
        calibration = TableCalibration(
            session_id=request.session_id,
            calibration_type=request.calibration_type,
            status=request.status,
            created_at=now,
            applied_at=now if request.status == "applied" else None,
            points_required=len(request.points),
            points_captured=len(request.points),
            points=request.points,
            transformation_matrix=request.transformation_matrix,
            calibration_metadata=request.metadata or {},
            accuracy=request.accuracy,
            max_error=request.max_error,
            mean_error=request.mean_error,
            rms_error=request.rms_error,
            is_default=False,  # Will be set later if needed
            backup_created=False,
            created_by="api",
            applied_by="api" if request.status == "applied" else None,
            notes=request.notes,
        )

        # Save to database
        saved = save_calibration(db, calibration)

        # Set as default if requested
        if request.is_default and request.status == "applied":
            set_default_calibration(db, request.session_id)
            saved.is_default = True

        logger.info(f"Saved calibration: {request.session_id} (ID: {saved.id})")

        return CalibrationDataResponse(
            success=True,
            id=saved.id,
            session_id=saved.session_id,
            calibration_type=saved.calibration_type,
            status=saved.status,
            created_at=saved.created_at.isoformat(),
            applied_at=saved.applied_at.isoformat() if saved.applied_at else None,
            points_captured=saved.points_captured,
            accuracy=saved.accuracy,
            transformation_matrix=saved.transformation_matrix,
            is_default=saved.is_default,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save calibration data: {str(e)}",
        )


@router.get("/retrieve/{identifier}", response_model=dict)
async def retrieve_calibration_data(
    identifier: str,
    by_session_id: bool = Query(
        True, description="True to search by session_id, False to search by database ID"
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve table calibration data from database.

    Fetches stored calibration data by session ID or database ID.

    Args:
        identifier: Session ID or database ID (based on by_session_id param)
        by_session_id: Whether to search by session ID (True) or DB ID (False)
        db: Database session

    Returns:
        Complete calibration data

    Raises:
        HTTPException: If calibration not found

    Example:
        GET /api/v1/vision/calibration/storage/retrieve/cal_20231013_120000_abc123?by_session_id=true
    """
    try:
        # Fetch calibration
        if by_session_id:
            calibration = get_calibration_by_session_id(db, identifier)
        else:
            try:
                cal_id = int(identifier)
                calibration = get_calibration_by_id(db, cal_id)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid ID format for database ID search",
                )

        if not calibration:
            raise HTTPException(
                status_code=404,
                detail=f"Calibration not found: {identifier}",
            )

        logger.info(f"Retrieved calibration: {calibration.session_id}")

        # Return complete calibration data
        return {
            "success": True,
            "calibration": calibration.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve calibration data: {str(e)}",
        )


@router.get("/default", response_model=dict)
async def get_default_calibration_data(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get the default table calibration.

    Retrieves the currently active default calibration configuration.

    Args:
        db: Database session

    Returns:
        Default calibration data or null if not set

    Example:
        GET /api/v1/vision/calibration/storage/default
    """
    try:
        calibration = get_default_calibration(db)

        if not calibration:
            return {
                "success": True,
                "calibration": None,
                "message": "No default calibration set",
            }

        logger.info(f"Retrieved default calibration: {calibration.session_id}")

        return {
            "success": True,
            "calibration": calibration.to_dict(),
        }

    except Exception as e:
        logger.error(f"Failed to get default calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get default calibration: {str(e)}",
        )


@router.get("/latest", response_model=dict)
async def get_latest_calibration_data(
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get the most recently applied calibration.

    Retrieves the latest calibration that was successfully applied.

    Args:
        db: Database session

    Returns:
        Latest applied calibration data

    Example:
        GET /api/v1/vision/calibration/storage/latest
    """
    try:
        calibration = get_latest_calibration(db)

        if not calibration:
            return {
                "success": True,
                "calibration": None,
                "message": "No applied calibrations found",
            }

        logger.info(f"Retrieved latest calibration: {calibration.session_id}")

        return {
            "success": True,
            "calibration": calibration.to_dict(),
        }

    except Exception as e:
        logger.error(f"Failed to get latest calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest calibration: {str(e)}",
        )


@router.get("/list", response_model=CalibrationListResponse)
async def list_calibration_data(
    status: Optional[str] = Query(
        None,
        pattern="^(in_progress|ready_to_apply|applied|expired|cancelled)$",
        description="Filter by calibration status",
    ),
    limit: int = Query(
        50, ge=1, le=500, description="Maximum number of calibrations to return"
    ),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> CalibrationListResponse:
    """List stored calibrations with filtering and pagination.

    Retrieves a list of calibration records from the database with optional
    status filtering and pagination support.

    Args:
        status: Filter by calibration status (optional)
        limit: Maximum results to return (1-500)
        offset: Pagination offset
        db: Database session

    Returns:
        List of calibrations with pagination info

    Example:
        GET /api/v1/vision/calibration/storage/list?status=applied&limit=10&offset=0
    """
    try:
        # Get calibrations
        calibrations = list_calibrations(db, status=status, limit=limit, offset=offset)

        # Convert to dictionaries
        calibration_dicts = [cal.to_dict() for cal in calibrations]

        # Get total count
        from sqlalchemy import func

        query = db.query(func.count(TableCalibration.id))
        if status:
            query = query.filter(TableCalibration.status == status)
        total = query.scalar()

        logger.info(f"Listed {len(calibrations)} calibrations (total: {total})")

        return CalibrationListResponse(
            success=True,
            calibrations=calibration_dicts,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Failed to list calibrations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list calibrations: {str(e)}",
        )


@router.put("/update/{session_id}", response_model=CalibrationDataResponse)
async def update_calibration_data(
    session_id: str,
    update_data: dict[str, Any],
    db: Session = Depends(get_db),
) -> CalibrationDataResponse:
    """Update existing calibration data.

    Modifies an existing calibration record. Can update status, notes, metadata,
    or mark as default.

    Args:
        session_id: Session identifier
        update_data: Fields to update
        db: Database session

    Returns:
        Updated calibration data

    Raises:
        HTTPException: If calibration not found or update fails

    Example:
        PUT /api/v1/vision/calibration/storage/update/cal_20231013_120000_abc123
        {
            "status": "applied",
            "notes": "Updated calibration after camera adjustment"
        }
    """
    try:
        # Validate transformation matrix if being updated
        if "transformation_matrix" in update_data:
            matrix = update_data["transformation_matrix"]
            if matrix and (len(matrix) != 3 or any(len(row) != 3 for row in matrix)):
                raise HTTPException(
                    status_code=400,
                    detail="Transformation matrix must be 3x3",
                )

        # Update calibration
        updated = update_calibration(db, session_id, update_data)

        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Calibration not found: {session_id}",
            )

        logger.info(f"Updated calibration: {session_id}")

        return CalibrationDataResponse(
            success=True,
            id=updated.id,
            session_id=updated.session_id,
            calibration_type=updated.calibration_type,
            status=updated.status,
            created_at=updated.created_at.isoformat(),
            applied_at=updated.applied_at.isoformat() if updated.applied_at else None,
            points_captured=updated.points_captured,
            accuracy=updated.accuracy,
            transformation_matrix=updated.transformation_matrix,
            is_default=updated.is_default,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update calibration: {str(e)}",
        )


@router.post("/set-default/{session_id}", response_model=SuccessResponse)
async def set_default_calibration_endpoint(
    session_id: str,
    db: Session = Depends(get_db),
) -> SuccessResponse:
    """Set a calibration as the default.

    Marks the specified calibration as the default, clearing the default flag
    from all other calibrations.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If calibration not found or not in applied status

    Example:
        POST /api/v1/vision/calibration/storage/set-default/cal_20231013_120000_abc123
    """
    try:
        success = set_default_calibration(db, session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Calibration not found or not in 'applied' status: {session_id}",
            )

        logger.info(f"Set default calibration: {session_id}")

        return create_success_response(
            f"Calibration {session_id} set as default",
            {"session_id": session_id, "is_default": True},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set default calibration: {str(e)}",
        )


@router.delete("/delete/{session_id}", response_model=SuccessResponse)
async def delete_calibration_data(
    session_id: str,
    db: Session = Depends(get_db),
) -> SuccessResponse:
    """Delete a calibration from the database.

    Permanently removes a calibration record. Cannot be undone.

    Args:
        session_id: Session identifier
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If calibration not found

    Example:
        DELETE /api/v1/vision/calibration/storage/delete/cal_20231013_120000_abc123
    """
    try:
        success = delete_calibration(db, session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Calibration not found: {session_id}",
            )

        logger.info(f"Deleted calibration: {session_id}")

        return create_success_response(
            f"Calibration {session_id} deleted successfully",
            {"session_id": session_id, "deleted": True},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete calibration: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete calibration: {str(e)}",
        )


__all__ = ["router"]
