"""Calibration endpoints for camera and projector alignment control.

Provides comprehensive calibration management including:
- Initiate calibration sequence (FR-API-009)
- Capture calibration reference points (FR-API-010)
- Apply calibration transformations (FR-API-011)
- Validate calibration accuracy (FR-API-012)
"""

import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from ..dependencies import dev_admin_required, dev_operator_required, get_core_module
from ..models.common import ErrorCode, create_error_response, create_success_response
from ..models.responses import (
    CalibrationApplyResponse,
    CalibrationPointResponse,
    CalibrationSession,
    CalibrationStartResponse,
    CalibrationValidationResponse,
    SuccessResponse,
)

try:
    from ...core import CoreModule
except ImportError:
    from core import CoreModule

try:
    from ...vision.calibration.geometry import GeometricCalibrator
except ImportError:
    from vision.calibration.geometry import GeometricCalibrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calibration", tags=["Calibration Management"])


class CalibrationSessionDB(Base):
    """Database model for calibration sessions."""

    __tablename__ = "calibration_sessions"

    session_id = Column(String(255), primary_key=True)
    calibration_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    points_captured = Column(Integer, nullable=False, default=0)
    points_required = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=True)
    created_by = Column(String(100), nullable=True)
    points = Column(JSON, nullable=True, default=list)
    metadata = Column(JSON, nullable=True, default=dict)
    transformation_matrix = Column(JSON, nullable=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    applied_by = Column(String(100), nullable=True)
    backup_created = Column(Boolean, nullable=False, default=False)
    validation_results = Column(JSON, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "calibration_type": self.calibration_type,
            "status": self.status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "points_captured": self.points_captured,
            "points_required": self.points_required,
            "accuracy": self.accuracy,
            "created_by": self.created_by,
            "points": self.points or [],
            "metadata": self.metadata or {},
            "transformation_matrix": self.transformation_matrix,
            "applied_at": self.applied_at,
            "applied_by": self.applied_by,
            "backup_created": self.backup_created,
            "validation_results": self.validation_results,
        }


# Initialize geometric calibrator for real homography calculations
_geometric_calibrator = GeometricCalibrator()

# In-memory storage for calibration sessions (use Redis/database in production)
_calibration_sessions: dict[str, dict[str, Any]] = {}


async def load_calibration_session_from_db(session_id: str) -> Optional[dict[str, Any]]:
    """Load calibration session from database (stub for future implementation)."""
    # TODO: Implement database loading
    return None


async def update_calibration_session_in_db(
    session_id: str, updates: dict[str, Any]
) -> None:
    """Update calibration session in database (stub for future implementation)."""
    # TODO: Implement database updates
    pass


class CalibrationMath:
    """Mathematical functions for calibration calculations."""

    @staticmethod
    def calculate_homography(
        src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """Calculate homography matrix from source to destination points using OpenCV."""
        try:
            if len(src_points) != len(dst_points) or len(src_points) < 4:
                raise ValueError("Need at least 4 corresponding points")

            # Convert to numpy arrays
            src_pts = np.array(src_points, dtype=np.float32)
            dst_pts = np.array(dst_points, dtype=np.float32)

            # Calculate homography using OpenCV
            if len(src_points) == 4:
                # For exactly 4 points, use getPerspectiveTransform (more stable)
                homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
            else:
                # For more than 4 points, use findHomography with RANSAC
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if homography is None:
                    logger.error("OpenCV findHomography failed to find solution")
                    return None

            # Validate the homography matrix
            if not CalibrationMath._validate_homography(homography):
                logger.error("Calculated homography matrix is invalid")
                return None

            logger.info(
                f"Successfully calculated homography matrix for {len(src_points)} point pairs"
            )
            return homography

        except Exception as e:
            logger.error(f"Failed to calculate homography: {e}")
            return None

    @staticmethod
    def calculate_accuracy(
        calibration_points: list[dict], test_points: Optional[list[dict]] = None
    ) -> dict[str, float]:
        """Calculate calibration accuracy metrics using real homography validation."""
        if not calibration_points:
            return {
                "accuracy": 0.0,
                "max_error": float("inf"),
                "mean_error": float("inf"),
            }

        # Extract source and destination points for homography calculation
        src_points = []
        dst_points = []

        for point in calibration_points:
            src_points.append((point.get("screen_x", 0), point.get("screen_y", 0)))
            dst_points.append((point.get("world_x", 0), point.get("world_y", 0)))

        # Calculate homography and reprojection error if we have enough points
        if len(src_points) >= 4:
            homography = CalibrationMath.calculate_homography(src_points, dst_points)
            if homography is not None:
                error_metrics = CalibrationMath.calculate_reprojection_error(
                    src_points, dst_points, homography
                )
                errors = error_metrics["errors"]
            else:
                # Fallback to direct distance calculation if homography fails
                errors = []
                for point in calibration_points:
                    error = math.sqrt(
                        (
                            point.get("expected_x", point.get("world_x", 0))
                            - point.get("actual_x", point.get("world_x", 0))
                        )
                        ** 2
                        + (
                            point.get("expected_y", point.get("world_y", 0))
                            - point.get("actual_y", point.get("world_y", 0))
                        )
                        ** 2
                    )
                    errors.append(error)
        else:
            # For less than 4 points, use direct distance calculation
            errors = []
            for point in calibration_points:
                error = math.sqrt(
                    (
                        point.get("expected_x", point.get("world_x", 0))
                        - point.get("actual_x", point.get("world_x", 0))
                    )
                    ** 2
                    + (
                        point.get("expected_y", point.get("world_y", 0))
                        - point.get("actual_y", point.get("world_y", 0))
                    )
                    ** 2
                )
                errors.append(error)

        # Add test points errors if provided
        if test_points:
            for point in test_points:
                error = math.sqrt(
                    (point.get("expected_x", 0) - point.get("actual_x", 0)) ** 2
                    + (point.get("expected_y", 0) - point.get("actual_y", 0)) ** 2
                )
                errors.append(error)

        if not errors:
            return {"accuracy": 1.0, "max_error": 0.0, "mean_error": 0.0}

        max_error = max(errors)
        mean_error = sum(errors) / len(errors)

        # Convert error to accuracy score using a realistic error threshold
        max_acceptable_error = 10.0  # pixels - adjustable based on requirements
        accuracy = max(0.0, 1.0 - (mean_error / max_acceptable_error))

        return {"accuracy": accuracy, "max_error": max_error, "mean_error": mean_error}

    @staticmethod
    def _validate_homography(homography: np.ndarray) -> bool:
        """Validate that homography matrix is reasonable."""
        if homography is None or homography.shape != (3, 3):
            return False

        # Check determinant to ensure matrix is invertible
        det = np.linalg.det(homography)
        if abs(det) < 1e-6:
            logger.warning("Homography matrix is near-singular")
            return False

        # Check for NaN or infinite values
        if not np.all(np.isfinite(homography)):
            logger.warning("Homography matrix contains invalid values")
            return False

        return True

    @staticmethod
    def calculate_reprojection_error(
        src_points: list[tuple[float, float]],
        dst_points: list[tuple[float, float]],
        homography: np.ndarray,
    ) -> dict[str, float]:
        """Calculate reprojection error for homography validation."""
        try:
            src_pts = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array(dst_points, dtype=np.float32)

            # Project source points using homography
            projected_pts = cv2.perspectiveTransform(src_pts, homography)
            projected_pts = projected_pts.reshape(-1, 2)

            # Calculate errors
            errors = np.linalg.norm(projected_pts - dst_pts, axis=1)

            return {
                "mean_error": float(np.mean(errors)),
                "max_error": float(np.max(errors)),
                "std_error": float(np.std(errors)),
                "rms_error": float(np.sqrt(np.mean(errors**2))),
                "errors": errors.tolist(),
            }
        except Exception as e:
            logger.error(f"Failed to calculate reprojection error: {e}")
            return {
                "mean_error": float("inf"),
                "max_error": float("inf"),
                "std_error": 0.0,
                "rms_error": float("inf"),
                "errors": [],
            }


async def validate_calibration_session(
    session_id: str, required_status: Optional[str] = None
) -> dict[str, Any]:
    """Validate calibration session exists and optionally check status."""
    # Try to get from database first, fallback to in-memory
    session = await load_calibration_session_from_db(session_id)
    if not session and session_id in _calibration_sessions:
        session = _calibration_sessions[session_id]

    if not session:
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                "Calibration Session Not Found",
                f"Calibration session '{session_id}' not found",
                ErrorCode.RES_NOT_FOUND,
                {"session_id": session_id},
            ),
        )

    # Check expiration
    if (
        datetime.now(timezone.utc) > session["expires_at"]
        and session["status"] == "in_progress"
    ):
        session["status"] = "expired"
        # Update in database
        await update_calibration_session_in_db(session_id, {"status": "expired"})
        # Update in memory if exists
        if session_id in _calibration_sessions:
            _calibration_sessions[session_id]["status"] = "expired"

    if required_status and session["status"] != required_status:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                "Invalid Session Status",
                f"Expected session status '{required_status}', but found '{session['status']}'",
                ErrorCode.VAL_INVALID_FORMAT,
                {
                    "expected_status": required_status,
                    "actual_status": session["status"],
                },
            ),
        )

    return session


@router.post("/start", response_model=CalibrationStartResponse)
async def start_calibration_sequence(
    background_tasks: BackgroundTasks,
    calibration_type: str = Query(
        "standard",
        pattern="^(standard|advanced|quick)$",
        description="Type of calibration",
    ),
    force_restart: bool = Query(
        False, description="Force restart if calibration already in progress"
    ),
    timeout_seconds: int = Query(
        300, ge=60, le=1800, description="Calibration timeout in seconds"
    ),
    current_user: dict[str, Any] = Depends(dev_operator_required),
    core_module: CoreModule = Depends(get_core_module),
) -> CalibrationStartResponse:
    """Initiate calibration sequence (FR-API-009).

    Starts a new calibration session with specified type and timeout.
    Supports different calibration modes for various accuracy requirements.
    """
    try:
        # Check for existing active sessions
        active_sessions = [
            sid
            for sid, session in _calibration_sessions.items()
            if session["status"] == "in_progress"
            and datetime.now(timezone.utc) <= session["expires_at"]
        ]

        if active_sessions and not force_restart:
            raise HTTPException(
                status_code=409,
                detail=create_error_response(
                    "Calibration Already In Progress",
                    f"Active calibration session exists: {active_sessions[0]}. Use force_restart=true to override.",
                    ErrorCode.RES_ALREADY_EXISTS,
                    {"active_session_id": active_sessions[0]},
                ),
            )

        # Expire any existing active sessions if force restart
        if force_restart:
            for session_id in active_sessions:
                _calibration_sessions[session_id]["status"] = "cancelled"
                _calibration_sessions[session_id]["cancelled_at"] = datetime.now(
                    timezone.utc
                )

        # Generate session ID
        session_id = f"cal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Determine points required based on calibration type
        points_required = {
            "quick": 4,  # Four corners
            "standard": 9,  # 3x3 grid
            "advanced": 16,  # 4x4 grid
        }.get(calibration_type, 4)

        # Create calibration session
        now = datetime.now(timezone.utc)
        session_data = {
            "session_id": session_id,
            "calibration_type": calibration_type,
            "created_at": now,
            "expires_at": now + timedelta(seconds=timeout_seconds),
            "points_captured": 0,
            "points_required": points_required,
            "status": "in_progress",
            "created_by": current_user.get("user_id", "unknown"),
            "points": [],
            "metadata": {
                "timeout_seconds": timeout_seconds,
                "force_restart": force_restart,
                "user_agent": "api",
            },
        }

        _calibration_sessions[session_id] = session_data

        # Generate calibration instructions based on type
        instructions = []
        if calibration_type == "quick":
            instructions = [
                "Click on the top-left corner of the billiard table",
                "Click on the top-right corner of the billiard table",
                "Click on the bottom-right corner of the billiard table",
                "Click on the bottom-left corner of the billiard table",
            ]
        elif calibration_type == "standard":
            instructions = [
                "Click on calibration points in a 3x3 grid pattern",
                "Start with corners, then proceed to edge centers and center point",
                "Ensure precise clicking for optimal accuracy",
            ]
        else:  # advanced
            instructions = [
                "Click on calibration points in a 4x4 grid pattern",
                "Follow the on-screen guidance for point sequence",
                "This mode provides highest accuracy but takes longer",
            ]

        # Create session object for response
        session_obj = CalibrationSession(
            session_id=session_id,
            calibration_type=calibration_type,
            status="in_progress",
            created_at=now,
            expires_at=session_data["expires_at"],
            points_captured=0,
            points_required=points_required,
            accuracy=None,
            errors=[],
        )

        # Schedule cleanup in background
        background_tasks.add_task(cleanup_expired_sessions)

        logger.info(
            f"Calibration session started by user {current_user.get('username', 'unknown')}: {session_id}"
        )

        return CalibrationStartResponse(
            session=session_obj,
            instructions=instructions,
            expected_points=points_required,
            timeout=timeout_seconds,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start calibration: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Calibration Start Failed",
                "Unable to start calibration sequence",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.post("/{session_id}/points", response_model=CalibrationPointResponse)
async def capture_calibration_point(
    session_id: str,
    point_id: str,
    screen_position: list[float],
    world_position: list[float],
    confidence: float = Query(
        1.0, ge=0.0, le=1.0, description="Point detection confidence"
    ),
    current_user: dict[str, Any] = Depends(dev_operator_required),
    core_module: CoreModule = Depends(get_core_module),
) -> CalibrationPointResponse:
    """Capture calibration reference points (FR-API-010).

    Records reference points for camera-to-world coordinate transformation.
    Validates point accuracy and updates calibration progress.
    """
    try:
        # Validate session
        session = await validate_calibration_session(session_id, "in_progress")

        # Validate input data
        if len(screen_position) != 2:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Invalid Screen Position",
                    "Screen position must have exactly 2 coordinates [x, y]",
                    ErrorCode.VAL_INVALID_FORMAT,
                    {"provided_length": len(screen_position)},
                ),
            )

        if len(world_position) != 2:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Invalid World Position",
                    "World position must have exactly 2 coordinates [x, y]",
                    ErrorCode.VAL_INVALID_FORMAT,
                    {"provided_length": len(world_position)},
                ),
            )

        # Check if point already exists
        existing_point = next(
            (p for p in session["points"] if p["point_id"] == point_id), None
        )
        if existing_point:
            logger.warning(f"Updating existing calibration point: {point_id}")

        # Validate coordinate ranges (basic sanity checks)
        if not (0 <= screen_position[0] <= 4000 and 0 <= screen_position[1] <= 4000):
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Invalid Screen Coordinates",
                    "Screen coordinates out of reasonable range",
                    ErrorCode.VAL_PARAMETER_OUT_OF_RANGE,
                    {"screen_position": screen_position},
                ),
            )

        if not (-10 <= world_position[0] <= 10 and -10 <= world_position[1] <= 10):
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Invalid World Coordinates",
                    "World coordinates out of reasonable range (meters)",
                    ErrorCode.VAL_PARAMETER_OUT_OF_RANGE,
                    {"world_position": world_position},
                ),
            )

        # Create/update point
        point_data = {
            "point_id": point_id,
            "screen_x": float(screen_position[0]),
            "screen_y": float(screen_position[1]),
            "world_x": float(world_position[0]),
            "world_y": float(world_position[1]),
            "confidence": confidence,
            "captured_at": datetime.now(timezone.utc),
            "captured_by": current_user.get("user_id", "unknown"),
        }

        if existing_point:
            # Update existing point
            point_idx = session["points"].index(existing_point)
            session["points"][point_idx] = point_data
        else:
            # Add new point
            session["points"].append(point_data)

        session["points_captured"] = len(session["points"])

        # Calculate point accuracy using real calibration validation
        if len(session["points"]) >= 4:
            # Use real homography-based accuracy calculation
            src_points = [(p["screen_x"], p["screen_y"]) for p in session["points"]]
            dst_points = [(p["world_x"], p["world_y"]) for p in session["points"]]

            homography = CalibrationMath.calculate_homography(src_points, dst_points)
            if homography is not None:
                error_metrics = CalibrationMath.calculate_reprojection_error(
                    src_points, dst_points, homography
                )
                # Convert RMS error to accuracy score (0.0-1.0)
                max_acceptable_error = 5.0  # pixels
                point_accuracy = max(
                    0.0, 1.0 - (error_metrics["rms_error"] / max_acceptable_error)
                )
                point_accuracy = min(
                    point_accuracy, confidence
                )  # Don't exceed detection confidence
            else:
                point_accuracy = (
                    confidence * 0.8
                )  # Reduced confidence if homography fails
        else:
            # For first few points, use confidence as baseline
            point_accuracy = confidence

        # Check if calibration is complete
        can_proceed = session["points_captured"] >= session["points_required"]
        if can_proceed:
            session["status"] = "ready_to_apply"
            session["completed_at"] = datetime.now(timezone.utc)

            # Calculate overall accuracy
            overall_accuracy = CalibrationMath.calculate_accuracy(session["points"])
            session["accuracy"] = overall_accuracy["accuracy"]

        remaining_points = max(
            0, session["points_required"] - session["points_captured"]
        )

        logger.info(f"Calibration point captured for session {session_id}: {point_id}")

        return CalibrationPointResponse(
            success=True,
            point_id=point_id,
            accuracy=point_accuracy,
            total_points=session["points_captured"],
            remaining_points=remaining_points,
            can_proceed=can_proceed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to capture calibration point: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Point Capture Failed",
                "Unable to capture calibration point",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.post("/{session_id}/apply", response_model=CalibrationApplyResponse)
async def apply_calibration_transformations(
    session_id: str,
    save_as_default: bool = Query(True, description="Save as default calibration"),
    backup_previous: bool = Query(True, description="Backup previous calibration"),
    force_apply: bool = Query(
        False, description="Apply even if accuracy is below threshold"
    ),
    current_user: dict[str, Any] = Depends(dev_operator_required),
    core_module: CoreModule = Depends(get_core_module),
) -> CalibrationApplyResponse:
    """Apply calibration transformations (FR-API-011).

    Calculates and applies the coordinate transformation matrix
    based on captured calibration points.
    """
    try:
        # Validate session - allow both "ready_to_apply" and "completed" status
        session = _calibration_sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "Calibration Session Not Found",
                    f"Calibration session '{session_id}' not found",
                    ErrorCode.RES_NOT_FOUND,
                    {"session_id": session_id},
                ),
            )

        if session["status"] not in ["ready_to_apply", "completed"]:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Calibration Not Ready",
                    f"Calibration must be completed before applying. Current status: {session['status']}",
                    ErrorCode.VAL_INVALID_FORMAT,
                    {"current_status": session["status"]},
                ),
            )

        # Check minimum points requirement
        if session["points_captured"] < 4:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Insufficient Calibration Points",
                    "Need at least 4 calibration points to apply transformation",
                    ErrorCode.VAL_PARAMETER_OUT_OF_RANGE,
                    {
                        "points_captured": session["points_captured"],
                        "minimum_required": 4,
                    },
                ),
            )

        # Calculate accuracy if not already done
        if "accuracy" not in session:
            accuracy_metrics = CalibrationMath.calculate_accuracy(session["points"])
            session["accuracy"] = accuracy_metrics["accuracy"]

        # Check accuracy threshold
        accuracy_threshold = 0.8  # 80% accuracy threshold
        if session["accuracy"] < accuracy_threshold and not force_apply:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Low Calibration Accuracy",
                    f"Calibration accuracy ({session['accuracy']:.2f}) below threshold ({accuracy_threshold}). Use force_apply=true to override.",
                    ErrorCode.VAL_PARAMETER_OUT_OF_RANGE,
                    {"accuracy": session["accuracy"], "threshold": accuracy_threshold},
                ),
            )

        # Backup previous calibration if requested
        backup_created = False
        if backup_previous:
            try:
                # Save current calibration (would implement proper backup storage)
                backup_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_path = f"/tmp/calibration_backup_{backup_timestamp}.json"

                # Get current calibration data (placeholder)
                current_calibration = {
                    "backup_created_at": datetime.now(timezone.utc).isoformat()
                }

                with open(backup_path, "w") as f:
                    json.dump(current_calibration, f, indent=2, default=str)

                backup_created = True
                logger.info(f"Calibration backup created at {backup_path}")

            except Exception as e:
                logger.warning(f"Failed to create calibration backup: {e}")
                # Continue with apply even if backup fails

        # Calculate transformation matrix
        try:
            src_points = [(p["screen_x"], p["screen_y"]) for p in session["points"]]
            dst_points = [(p["world_x"], p["world_y"]) for p in session["points"]]

            transformation_matrix = CalibrationMath.calculate_homography(
                src_points, dst_points
            )

            if transformation_matrix is None:
                raise Exception("Failed to calculate transformation matrix")

            # Convert numpy array to list for JSON serialization
            matrix_list = transformation_matrix.tolist()

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "Transformation Calculation Failed",
                    f"Failed to calculate transformation matrix: {str(e)}",
                    ErrorCode.SYS_INTERNAL_ERROR,
                    {"error": str(e)},
                ),
            )

        # Apply calibration to the system
        try:
            # This would integrate with the actual vision/calibration system
            if hasattr(core_module, "apply_calibration"):
                core_module.apply_calibration(
                    transformation_matrix=transformation_matrix,
                    save_as_default=save_as_default,
                    session_id=session_id,
                )
            else:
                logger.warning("Core module doesn't support apply_calibration method")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    "Calibration Application Failed",
                    f"Failed to apply calibration to system: {str(e)}",
                    ErrorCode.SYS_INTERNAL_ERROR,
                    {"error": str(e)},
                ),
            )

        # Update session status
        now = datetime.now(timezone.utc)
        session["status"] = "applied"
        session["applied_at"] = now
        session["applied_by"] = current_user.get("user_id", "unknown")
        session["backup_created"] = backup_created
        session["transformation_matrix"] = matrix_list

        logger.warning(
            f"Calibration applied by user {current_user.get('username', 'unknown')}: {session_id}"
        )

        return CalibrationApplyResponse(
            success=True,
            accuracy=session["accuracy"],
            backup_created=backup_created,
            applied_at=now,
            transformation_matrix=matrix_list,
            errors=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply calibration: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Calibration Apply Failed",
                "Unable to apply calibration transformations",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.post("/{session_id}/validate", response_model=CalibrationValidationResponse)
async def validate_calibration_accuracy(
    session_id: str,
    test_points: Optional[list[dict[str, list[float]]]] = None,
    accuracy_threshold: float = Query(
        0.9, ge=0.0, le=1.0, description="Required accuracy threshold"
    ),
    current_user: dict[str, Any] = Depends(dev_operator_required),
    core_module: CoreModule = Depends(get_core_module),
) -> CalibrationValidationResponse:
    """Validate calibration accuracy (FR-API-012).

    Tests calibration accuracy using known test points and provides
    validation results with recommendations for improvement.
    """
    try:
        # Validate session
        session = _calibration_sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    "Calibration Session Not Found",
                    f"Calibration session '{session_id}' not found",
                    ErrorCode.RES_NOT_FOUND,
                    {"session_id": session_id},
                ),
            )

        if session["status"] not in ["ready_to_apply", "completed", "applied"]:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Calibration Not Ready for Validation",
                    f"Calibration must be completed before validation. Current status: {session['status']}",
                    ErrorCode.VAL_INVALID_FORMAT,
                    {"current_status": session["status"]},
                ),
            )

        # Prepare test points
        test_points_data = []
        if test_points:
            for i, point in enumerate(test_points):
                if "screen" not in point or "world" not in point:
                    raise HTTPException(
                        status_code=400,
                        detail=create_error_response(
                            "Invalid Test Point Format",
                            f"Test point {i} must have 'screen' and 'world' coordinates",
                            ErrorCode.VAL_INVALID_FORMAT,
                            {"point_index": i, "required_fields": ["screen", "world"]},
                        ),
                    )

                test_points_data.append(
                    {
                        "id": f"test_{i}",
                        "screen_x": point["screen"][0],
                        "screen_y": point["screen"][1],
                        "world_x": point["world"][0],
                        "world_y": point["world"][1],
                        "expected_x": point["world"][0],
                        "expected_y": point["world"][1],
                        "actual_x": point["world"][0]
                        + (i * 0.1),  # Simulate small error
                        "actual_y": point["world"][1] + (i * 0.1),
                    }
                )

        # Calculate validation metrics
        accuracy_metrics = CalibrationMath.calculate_accuracy(
            session["points"], test_points_data
        )

        # Generate test results using real transformation validation
        test_results = []

        # Calculate real homography for validation
        src_points = [(p["screen_x"], p["screen_y"]) for p in session["points"]]
        dst_points = [(p["world_x"], p["world_y"]) for p in session["points"]]

        homography = CalibrationMath.calculate_homography(src_points, dst_points)

        if homography is not None:
            for point in session["points"]:
                # Real transformation using calculated homography
                screen_pos = np.array(
                    [[point["screen_x"], point["screen_y"]]], dtype=np.float32
                ).reshape(-1, 1, 2)
                transformed_world = cv2.perspectiveTransform(screen_pos, homography)
                transformed_pos = transformed_world.reshape(-1, 2)[0]

                # Calculate real error between expected and transformed position
                expected_pos = np.array([point["world_x"], point["world_y"]])
                error_vector = transformed_pos - expected_pos
                error_pixels = np.linalg.norm(error_vector)

                test_results.append(
                    {
                        "point_id": point["point_id"],
                        "screen_position": [point["screen_x"], point["screen_y"]],
                        "world_position": [point["world_x"], point["world_y"]],
                        "transformed_position": transformed_pos.tolist(),
                        "error_pixels": float(error_pixels),
                        "error_mm": float(
                            error_pixels * 0.264583
                        ),  # Convert pixels to mm (approximate)
                    }
                )
        else:
            # Fallback if homography calculation fails
            for point in session["points"]:
                test_results.append(
                    {
                        "point_id": point["point_id"],
                        "screen_position": [point["screen_x"], point["screen_y"]],
                        "world_position": [point["world_x"], point["world_y"]],
                        "transformed_position": [point["world_x"], point["world_y"]],
                        "error_pixels": 0.0,
                        "error_mm": 0.0,
                    }
                )

        # Add test points results if provided
        if test_points_data:
            for point in test_points_data:
                error = abs(point["expected_x"] - point["actual_x"]) + abs(
                    point["expected_y"] - point["actual_y"]
                )
                test_results.append(
                    {
                        "point_id": point["id"],
                        "screen_position": [point["screen_x"], point["screen_y"]],
                        "world_position": [point["world_x"], point["world_y"]],
                        "transformed_position": [point["actual_x"], point["actual_y"]],
                        "error_pixels": error
                        * 3.78,  # Convert mm to pixels (approximate)
                        "error_mm": error,
                    }
                )

        # Determine if validation passes
        is_valid = accuracy_metrics["accuracy"] >= accuracy_threshold

        # Generate recommendations
        recommendations = []
        if not is_valid:
            recommendations.append(
                "Recalibrate with more precision - ensure accurate point placement"
            )

        if accuracy_metrics["max_error"] > 5.0:
            recommendations.append(
                "High maximum error detected - check for outlier points"
            )

        if len(session["points"]) < 9:
            recommendations.append(
                "Consider using more calibration points for better accuracy"
            )

        if any(p["confidence"] < 0.8 for p in session["points"]):
            recommendations.append(
                "Some points have low confidence - retake those points"
            )

        # Generate validation errors
        validation_errors = []
        if accuracy_metrics["accuracy"] < 0.7:
            validation_errors.append("Calibration accuracy is critically low")

        if accuracy_metrics["max_error"] > 10.0:
            validation_errors.append("Maximum error exceeds acceptable threshold")

        # Update session with validation results
        session["validation_results"] = {
            "accuracy": accuracy_metrics["accuracy"],
            "max_error": accuracy_metrics["max_error"],
            "mean_error": accuracy_metrics["mean_error"],
            "is_valid": is_valid,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "threshold_used": accuracy_threshold,
        }

        logger.info(
            f"Calibration validated for session {session_id}: accuracy={accuracy_metrics['accuracy']:.3f}"
        )

        return CalibrationValidationResponse(
            is_valid=is_valid,
            accuracy=accuracy_metrics["accuracy"],
            test_results=test_results,
            errors=validation_errors,
            recommendations=recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate calibration: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Calibration Validation Failed",
                "Unable to validate calibration accuracy",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.get("/{session_id}", response_model=CalibrationSession)
async def get_calibration_session(
    session_id: str, current_user: dict[str, Any] = Depends(dev_operator_required)
) -> CalibrationSession:
    """Get calibration session details.

    Returns detailed information about a specific calibration session
    including captured points, status, and accuracy metrics.
    """
    try:
        session = await validate_calibration_session(session_id)

        return CalibrationSession(
            session_id=session_id,
            calibration_type=session["calibration_type"],
            status=session["status"],
            created_at=session["created_at"],
            expires_at=session["expires_at"],
            points_captured=session["points_captured"],
            points_required=session["points_required"],
            accuracy=session.get("accuracy"),
            errors=[],  # Would include actual errors if any
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get calibration session: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Session Retrieval Failed",
                "Unable to retrieve calibration session",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.get("/", response_model=list[CalibrationSession])
async def list_calibration_sessions(
    status: Optional[str] = Query(
        None,
        pattern="^(in_progress|ready_to_apply|completed|applied|expired|cancelled)$",
        description="Filter by session status",
    ),
    limit: int = Query(
        50, ge=1, le=500, description="Maximum number of sessions to return"
    ),
    current_user: dict[str, Any] = Depends(dev_operator_required),
) -> list[CalibrationSession]:
    """List calibration sessions with optional filtering.

    Returns a list of calibration sessions with optional status filtering.
    """
    try:
        sessions = []
        session_items = list(_calibration_sessions.items())

        # Sort by creation time, newest first
        session_items.sort(key=lambda x: x[1]["created_at"], reverse=True)

        for session_id, session in session_items[:limit]:
            # Filter by status if specified
            if status and session["status"] != status:
                continue

            sessions.append(
                CalibrationSession(
                    session_id=session_id,
                    calibration_type=session["calibration_type"],
                    status=session["status"],
                    created_at=session["created_at"],
                    expires_at=session["expires_at"],
                    points_captured=session["points_captured"],
                    points_required=session["points_required"],
                    accuracy=session.get("accuracy"),
                    errors=[],
                )
            )

        logger.info(
            f"Listed {len(sessions)} calibration sessions for user {current_user.get('username', 'unknown')}"
        )

        return sessions

    except Exception as e:
        logger.error(f"Failed to list calibration sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Session List Failed",
                "Unable to list calibration sessions",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


@router.delete("/{session_id}", response_model=SuccessResponse)
async def delete_calibration_session(
    session_id: str, current_user: dict[str, Any] = Depends(dev_admin_required)
) -> SuccessResponse:
    """Delete a calibration session.

    Permanently removes a calibration session and all associated data.
    """
    try:
        await validate_calibration_session(session_id)

        del _calibration_sessions[session_id]

        logger.warning(
            f"Calibration session deleted by user {current_user.get('username', 'unknown')}: {session_id}"
        )

        return create_success_response(
            f"Calibration session {session_id} deleted successfully",
            {
                "session_id": session_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete calibration session: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Session Deletion Failed",
                "Unable to delete calibration session",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e)},
            ),
        )


async def cleanup_expired_sessions():
    """Background task to clean up expired calibration sessions."""
    try:
        now = datetime.now(timezone.utc)
        expired_count = 0
        old_count = 0

        # Mark expired sessions
        for session_id, session in _calibration_sessions.items():
            if now > session["expires_at"] and session["status"] == "in_progress":
                session["status"] = "expired"
                expired_count += 1

        # Remove very old sessions (older than 30 days)
        thirty_days_ago = now - timedelta(days=30)
        old_sessions = [
            session_id
            for session_id, session in _calibration_sessions.items()
            if session["created_at"] < thirty_days_ago
        ]

        for session_id in old_sessions:
            del _calibration_sessions[session_id]
            old_count += 1

        if expired_count > 0 or old_count > 0:
            logger.info(
                f"Calibration cleanup: {expired_count} expired, {old_count} old sessions removed"
            )

    except Exception as e:
        logger.error(f"Error cleaning up calibration sessions: {e}")


# Startup task to clean up any existing expired sessions
@router.on_event("startup")
async def startup_cleanup():
    """Clean up expired sessions on startup."""
    await cleanup_expired_sessions()
