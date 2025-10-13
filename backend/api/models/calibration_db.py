"""Database models for calibration data persistence.

Provides SQLAlchemy models for storing and retrieving table calibration data.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Session

from ..database import Base


class TableCalibration(Base):
    """Table calibration data model.

    Stores complete calibration sessions including transformation matrices,
    calibration points, and accuracy metrics.
    """

    __tablename__ = "table_calibrations"

    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Session identification
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    calibration_type = Column(
        String(50), nullable=False, default="standard"
    )  # quick, standard, advanced

    # Status and timestamps
    status = Column(
        String(50), nullable=False, default="in_progress", index=True
    )  # in_progress, ready_to_apply, applied, expired, cancelled
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    expires_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    applied_at = Column(DateTime, nullable=True)

    # Calibration requirements and progress
    points_required = Column(Integer, nullable=False, default=4)
    points_captured = Column(Integer, nullable=False, default=0)

    # Calibration data (stored as JSON)
    points = Column(JSON, nullable=False, default=list)  # List of calibration points
    transformation_matrix = Column(
        JSON, nullable=True
    )  # 3x3 homography matrix as nested list
    calibration_metadata = Column(
        JSON, nullable=True, default=dict
    )  # Additional metadata (renamed from metadata to avoid SQLAlchemy conflict)

    # Quality metrics
    accuracy = Column(Float, nullable=True)  # Overall accuracy score (0.0-1.0)
    max_error = Column(Float, nullable=True)  # Maximum reprojection error
    mean_error = Column(Float, nullable=True)  # Mean reprojection error
    rms_error = Column(Float, nullable=True)  # RMS reprojection error

    # Flags
    is_default = Column(
        Boolean, nullable=False, default=False, index=True
    )  # Is this the default calibration?
    backup_created = Column(Boolean, nullable=False, default=False)

    # User tracking
    created_by = Column(String(100), nullable=False, default="api")
    applied_by = Column(String(100), nullable=True)

    # Notes and description
    notes = Column(Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary.

        Returns:
            Dictionary representation of the calibration
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "calibration_type": self.calibration_type,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "points_required": self.points_required,
            "points_captured": self.points_captured,
            "points": self.points,
            "transformation_matrix": self.transformation_matrix,
            "metadata": self.calibration_metadata,
            "accuracy": self.accuracy,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "rms_error": self.rms_error,
            "is_default": self.is_default,
            "backup_created": self.backup_created,
            "created_by": self.created_by,
            "applied_by": self.applied_by,
            "notes": self.notes,
        }

    @classmethod
    def from_session_data(cls, session_data: dict[str, Any]) -> "TableCalibration":
        """Create model from session data dictionary.

        Args:
            session_data: Session data dictionary

        Returns:
            TableCalibration model instance
        """
        return cls(
            session_id=session_data.get("session_id"),
            calibration_type=session_data.get("calibration_type", "standard"),
            status=session_data.get("status", "in_progress"),
            created_at=session_data.get("created_at"),
            expires_at=session_data.get("expires_at"),
            completed_at=session_data.get("completed_at"),
            applied_at=session_data.get("applied_at"),
            points_required=session_data.get("points_required", 4),
            points_captured=session_data.get("points_captured", 0),
            points=session_data.get("points", []),
            transformation_matrix=session_data.get("transformation_matrix"),
            calibration_metadata=session_data.get("metadata", {}),
            accuracy=session_data.get("accuracy"),
            max_error=session_data.get("max_error"),
            mean_error=session_data.get("mean_error"),
            rms_error=session_data.get("rms_error"),
            is_default=session_data.get("is_default", False),
            backup_created=session_data.get("backup_created", False),
            created_by=session_data.get("created_by", "api"),
            applied_by=session_data.get("applied_by"),
            notes=session_data.get("notes"),
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"<TableCalibration(id={self.id}, session_id='{self.session_id}', status='{self.status}', accuracy={self.accuracy})>"


# Database utility functions


def save_calibration(db: Session, calibration: TableCalibration) -> TableCalibration:
    """Save calibration to database.

    Args:
        db: Database session
        calibration: Calibration model to save

    Returns:
        Saved calibration with ID populated
    """
    db.add(calibration)
    db.commit()
    db.refresh(calibration)
    return calibration


def get_calibration_by_session_id(
    db: Session, session_id: str
) -> Optional[TableCalibration]:
    """Get calibration by session ID.

    Args:
        db: Database session
        session_id: Session identifier

    Returns:
        TableCalibration or None if not found
    """
    return (
        db.query(TableCalibration)
        .filter(TableCalibration.session_id == session_id)
        .first()
    )


def get_calibration_by_id(
    db: Session, calibration_id: int
) -> Optional[TableCalibration]:
    """Get calibration by database ID.

    Args:
        db: Database session
        calibration_id: Database ID

    Returns:
        TableCalibration or None if not found
    """
    return (
        db.query(TableCalibration).filter(TableCalibration.id == calibration_id).first()
    )


def get_default_calibration(db: Session) -> Optional[TableCalibration]:
    """Get the default calibration.

    Args:
        db: Database session

    Returns:
        Default TableCalibration or None if not set
    """
    return (
        db.query(TableCalibration)
        .filter(TableCalibration.is_default is True)
        .filter(TableCalibration.status == "applied")
        .order_by(TableCalibration.applied_at.desc())
        .first()
    )


def get_latest_calibration(db: Session) -> Optional[TableCalibration]:
    """Get the most recently applied calibration.

    Args:
        db: Database session

    Returns:
        Latest applied TableCalibration or None
    """
    return (
        db.query(TableCalibration)
        .filter(TableCalibration.status == "applied")
        .order_by(TableCalibration.applied_at.desc())
        .first()
    )


def list_calibrations(
    db: Session,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[TableCalibration]:
    """List calibrations with optional filtering.

    Args:
        db: Database session
        status: Filter by status (optional)
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of TableCalibration models
    """
    query = db.query(TableCalibration)

    if status:
        query = query.filter(TableCalibration.status == status)

    return (
        query.order_by(TableCalibration.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )


def update_calibration(
    db: Session, session_id: str, update_data: dict[str, Any]
) -> Optional[TableCalibration]:
    """Update existing calibration.

    Args:
        db: Database session
        session_id: Session identifier
        update_data: Dictionary of fields to update

    Returns:
        Updated TableCalibration or None if not found
    """
    calibration = get_calibration_by_session_id(db, session_id)
    if not calibration:
        return None

    # Update fields
    for key, value in update_data.items():
        if hasattr(calibration, key):
            setattr(calibration, key, value)

    db.commit()
    db.refresh(calibration)
    return calibration


def delete_calibration(db: Session, session_id: str) -> bool:
    """Delete calibration by session ID.

    Args:
        db: Database session
        session_id: Session identifier

    Returns:
        True if deleted, False if not found
    """
    calibration = get_calibration_by_session_id(db, session_id)
    if not calibration:
        return False

    db.delete(calibration)
    db.commit()
    return True


def set_default_calibration(db: Session, session_id: str) -> bool:
    """Set a calibration as the default.

    Clears is_default flag from other calibrations and sets it on the specified one.

    Args:
        db: Database session
        session_id: Session identifier

    Returns:
        True if successful, False if not found
    """
    calibration = get_calibration_by_session_id(db, session_id)
    if not calibration or calibration.status != "applied":
        return False

    # Clear is_default from all other calibrations
    db.query(TableCalibration).filter(TableCalibration.is_default is True).update(
        {"is_default": False}
    )

    # Set this calibration as default
    calibration.is_default = True
    db.commit()

    return True


__all__ = [
    "TableCalibration",
    "save_calibration",
    "get_calibration_by_session_id",
    "get_calibration_by_id",
    "get_default_calibration",
    "get_latest_calibration",
    "list_calibrations",
    "update_calibration",
    "delete_calibration",
    "set_default_calibration",
]
