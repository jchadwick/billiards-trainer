"""Projector calibration module.

This module provides comprehensive calibration functionality for projector systems,
including keystone correction, geometric mapping, and calibration persistence.
"""

from .geometric import CalibrationTarget, GeometricCalibrator, TableDimensions
from .keystone import CornerPoints, KeystoneCalibrator, KeystoneParams
from .manager import (
    CalibrationManager,
    CalibrationMethod,
    CalibrationSettings,
    CalibrationState,
)
from .persistence import CalibrationPersistence, CalibrationProfile

__all__ = [
    # Keystone calibration
    "KeystoneCalibrator",
    "KeystoneParams",
    "CornerPoints",
    # Geometric calibration
    "GeometricCalibrator",
    "TableDimensions",
    "CalibrationTarget",
    # Persistence
    "CalibrationPersistence",
    "CalibrationProfile",
    # Main manager
    "CalibrationManager",
    "CalibrationState",
    "CalibrationMethod",
    "CalibrationSettings",
]
