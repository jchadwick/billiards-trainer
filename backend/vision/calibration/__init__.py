"""Vision Calibration System.

Comprehensive calibration system for billiards vision tracking including:
- Camera intrinsic and extrinsic calibration
- Color threshold calibration with lighting adaptation
- Geometric perspective correction and coordinate mapping
- System validation and testing
- Interactive calibration tools
- Import/export functionality

Main Classes:
    CalibrationManager: Central coordinator for all calibration operations
    CameraCalibrator: Camera intrinsic parameters and distortion correction
    ColorCalibrator: Color threshold detection and lighting adaptation
    GeometricCalibrator: Perspective correction and coordinate transformation
    CalibrationValidator: System validation and accuracy testing

Usage Example:
    from ...vision.calibration import CalibrationManager

    # Initialize calibration system
    manager = CalibrationManager()
    manager.start_new_session("table_setup_1")

    # Perform calibrations
    success, camera_params = manager.calibrate_camera(calibration_images)
    color_profile = manager.calibrate_colors(frame, ball_samples)
    geometry_cal = manager.calibrate_geometry(frame, table_corners)

    # Validate system
    report = manager.validate_system(test_data)

    # Check if ready for operation
    if manager.is_system_ready():
        # Use calibrated system
        world_point = manager.pixel_to_world(pixel_point)
        corrected_frame = manager.process_frame(frame)

Requirements Implemented:
    FR-VIS-039: Automatic camera calibration
    FR-VIS-040: Camera intrinsic parameter calculation
    FR-VIS-041: Camera-to-table transformation
    FR-VIS-042: Lens distortion compensation
    FR-VIS-043: Manual calibration adjustment
    FR-VIS-044: Auto-detect optimal color thresholds
    FR-VIS-045: Adapt to ambient lighting changes
    FR-VIS-046: Color picker interface
    FR-VIS-047: Save and load calibration profiles
"""

from .camera import CameraCalibrator, CameraParameters, TableTransform
from .color import ColorCalibrator, ColorProfile, ColorThresholds
from .geometry import (
    CoordinateMapping,
    GeometricCalibration,
    GeometricCalibrator,
    PerspectiveCorrection,
)

# Optional interactive GUI (may not be available in all environments)
try:
    from .interactive import InteractiveCalibrationGUI, run_calibration_gui

    _HAS_INTERACTIVE = True
except ImportError:
    _HAS_INTERACTIVE = False
    InteractiveCalibrationGUI = None
    run_calibration_gui = None

from .manager import CalibrationManager, CalibrationSession

# Test functions (optional)
try:
    from .test_calibration import run_calibration_tests
except ImportError:
    run_calibration_tests = None

from .validation import CalibrationReport, CalibrationValidator, ValidationResult

__all__ = [
    # Main manager class
    "CalibrationManager",
    "CalibrationSession",
    # Core calibration classes
    "CameraCalibrator",
    "ColorCalibrator",
    "GeometricCalibrator",
    "CalibrationValidator",
    # Data classes
    "CameraParameters",
    "TableTransform",
    "ColorProfile",
    "ColorThresholds",
    "GeometricCalibration",
    "PerspectiveCorrection",
    "CoordinateMapping",
    "ValidationResult",
    "CalibrationReport",
    # Interactive tools
    "InteractiveCalibrationGUI",
    "run_calibration_gui",
    # Testing
    "run_calibration_tests",
]

# Version info
__version__ = "1.0.0"
__author__ = "Billiards Trainer Development Team"


# Quick access functions for common operations
def quick_setup(base_dir: str = None) -> CalibrationManager:
    """Quick setup for calibration system.

    Args:
        base_dir: Base directory for calibration data

    Returns:
        Configured CalibrationManager instance
    """
    manager = CalibrationManager(base_dir)
    manager.start_new_session("quick_setup")
    return manager


def auto_calibrate_from_frame(
    frame, manager: CalibrationManager = None
) -> CalibrationManager:
    """Automatic calibration from single frame (for testing/demo).

    Args:
        frame: Input frame containing table
        manager: Optional existing manager instance

    Returns:
        CalibrationManager with basic calibration applied
    """
    if manager is None:
        manager = quick_setup()

    # Auto-calibrate colors
    manager.calibrate_colors(frame, profile_name="auto_calibrated")

    # Auto-detect and calibrate geometry
    manager.calibrate_geometry(frame)

    return manager
