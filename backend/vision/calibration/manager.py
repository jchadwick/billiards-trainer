"""Calibration system manager and integration."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .camera import CameraCalibrator, CameraParameters
from .color import ColorCalibrator, ColorProfile
from .geometry import GeometricCalibration, GeometricCalibrator
from .validation import CalibrationReport, CalibrationValidator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSession:
    """Complete calibration session data."""

    session_id: str
    created_date: str
    camera_calibration: Optional[CameraParameters]
    color_profile: Optional[ColorProfile]
    geometric_calibration: Optional[GeometricCalibration]
    validation_report: Optional[CalibrationReport]
    system_ready: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "session_id": self.session_id,
            "created_date": self.created_date,
            "camera_calibration": self.camera_calibration.to_dict()
            if self.camera_calibration
            else None,
            "color_profile": self.color_profile.to_dict()
            if self.color_profile
            else None,
            "geometric_calibration": self.geometric_calibration.to_dict()
            if self.geometric_calibration
            else None,
            "validation_report": self.validation_report.to_dict()
            if self.validation_report
            else None,
            "system_ready": self.system_ready,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_date=data["created_date"],
            camera_calibration=CameraParameters.from_dict(data["camera_calibration"])
            if data["camera_calibration"]
            else None,
            color_profile=ColorProfile.from_dict(data["color_profile"])
            if data["color_profile"]
            else None,
            geometric_calibration=GeometricCalibration.from_dict(
                data["geometric_calibration"]
            )
            if data["geometric_calibration"]
            else None,
            validation_report=CalibrationReport.from_dict(data["validation_report"])
            if data["validation_report"]
            else None,
            system_ready=data["system_ready"],
        )


class CalibrationManager:
    """Comprehensive calibration system manager.

    Coordinates all calibration components and provides unified interface for:
    - Camera intrinsic/extrinsic calibration
    - Color threshold calibration
    - Geometric transformation calibration
    - System validation and testing
    - Import/export functionality
    - Session management
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize calibration manager.

        Args:
            base_dir: Base directory for calibration data storage
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "calibration_data"
        self.base_dir.mkdir(exist_ok=True)

        # Initialize calibration components
        self.camera_calibrator = CameraCalibrator(str(self.base_dir / "camera"))
        self.color_calibrator = ColorCalibrator(str(self.base_dir / "color"))
        self.geometry_calibrator = GeometricCalibrator(str(self.base_dir / "geometry"))
        self.validator = CalibrationValidator(str(self.base_dir / "validation"))

        # Session management
        self.current_session: Optional[CalibrationSession] = None
        self.sessions_dir = self.base_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)

    def start_new_session(self, session_name: Optional[str] = None) -> str:
        """Start new calibration session.

        Args:
            session_name: Optional session name

        Returns:
            Session ID
        """
        import uuid

        session_id = str(uuid.uuid4())[:8]
        if session_name:
            session_id = f"{session_name}_{session_id}"

        self.current_session = CalibrationSession(
            session_id=session_id,
            created_date=datetime.now().isoformat(),
            camera_calibration=None,
            color_profile=None,
            geometric_calibration=None,
            validation_report=None,
            system_ready=False,
        )

        logger.info(f"Started new calibration session: {session_id}")
        return session_id

    def calibrate_camera(
        self, calibration_images: list[np.ndarray], save_session: bool = True
    ) -> tuple[bool, Optional[CameraParameters]]:
        """Perform camera calibration.

        Args:
            calibration_images: List of chessboard calibration images
            save_session: Whether to save session after calibration

        Returns:
            Tuple of (success, camera_parameters)
        """
        try:
            logger.info(
                f"Starting camera calibration with {len(calibration_images)} images"
            )

            success, camera_params = self.camera_calibrator.calibrate_intrinsics(
                calibration_images
            )

            if success and self.current_session:
                self.current_session.camera_calibration = camera_params
                if save_session:
                    self.save_current_session()

            return success, camera_params

        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            return False, None

    def calibrate_colors(
        self,
        frame: np.ndarray,
        ball_samples: Optional[dict[str, list[tuple[int, int, int, int]]]] = None,
        profile_name: str = "default",
        save_session: bool = True,
    ) -> Optional[ColorProfile]:
        """Perform color calibration.

        Args:
            frame: Frame for color calibration
            ball_samples: Optional ball color samples
            profile_name: Name for color profile
            save_session: Whether to save session after calibration

        Returns:
            Color profile or None if failed
        """
        try:
            logger.info("Starting color calibration")

            # Auto-calibrate table color
            table_thresholds = self.color_calibrator.auto_calibrate_table_color(frame)

            # Calibrate ball colors if samples provided
            ball_thresholds = self.color_calibrator.default_ball_colors.copy()
            if ball_samples:
                ball_thresholds.update(
                    self.color_calibrator.calibrate_ball_colors(frame, ball_samples)
                )

            # Create color profile
            ambient_light = self.color_calibrator._estimate_ambient_light(frame)
            color_profile = ColorProfile(
                name=profile_name,
                table_color=table_thresholds,
                ball_colors=ball_thresholds,
                lighting_condition="calibrated",
                creation_date=datetime.now().isoformat(),
                ambient_light_level=ambient_light,
            )

            self.color_calibrator.current_profile = color_profile

            if self.current_session:
                self.current_session.color_profile = color_profile
                if save_session:
                    self.save_current_session()

            logger.info("Color calibration completed successfully")
            return color_profile

        except Exception as e:
            logger.error(f"Color calibration failed: {e}")
            return None

    def calibrate_geometry(
        self,
        frame: np.ndarray,
        table_corners: Optional[list[tuple[float, float]]] = None,
        table_dimensions: Optional[tuple[float, float]] = None,
        save_session: bool = True,
    ) -> Optional[GeometricCalibration]:
        """Perform geometric calibration.

        Args:
            frame: Frame containing table
            table_corners: Optional manual table corners
            table_dimensions: Real world table dimensions
            save_session: Whether to save session after calibration

        Returns:
            Geometric calibration or None if failed
        """
        try:
            logger.info("Starting geometric calibration")

            # Apply camera distortion correction if available
            corrected_frame = frame
            if self.camera_calibrator.camera_params:
                corrected_frame = self.camera_calibrator.undistort_image(frame)

            # Perform geometric calibration
            calibration = self.geometry_calibrator.calibrate_table_geometry(
                corrected_frame, table_corners, table_dimensions
            )

            if self.current_session:
                self.current_session.geometric_calibration = calibration
                if save_session:
                    self.save_current_session()

            logger.info("Geometric calibration completed successfully")
            return calibration

        except Exception as e:
            logger.error(f"Geometric calibration failed: {e}")
            return None

    def validate_system(
        self, test_data: Optional[dict[str, Any]] = None
    ) -> Optional[CalibrationReport]:
        """Validate complete calibration system.

        Args:
            test_data: Optional test data for validation

        Returns:
            Validation report or None if failed
        """
        try:
            logger.info("Starting system validation")

            if test_data is None:
                test_data = {}

            report = self.validator.generate_comprehensive_report(
                self.camera_calibrator,
                self.color_calibrator,
                self.geometry_calibrator,
                test_data,
            )

            # Update system ready status
            system_ready = report.overall_score > 0.8  # 80% threshold

            if self.current_session:
                self.current_session.validation_report = report
                self.current_session.system_ready = system_ready
                self.save_current_session()

            logger.info(
                f"System validation completed. Overall score: {report.overall_score:.3f}"
            )
            return report

        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with all calibrations applied.

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        processed_frame = frame.copy()

        try:
            # Apply camera distortion correction
            if self.camera_calibrator.camera_params:
                processed_frame = self.camera_calibrator.undistort_image(
                    processed_frame
                )

            # Apply perspective correction
            if (
                self.geometry_calibrator.current_calibration
                and self.geometry_calibrator.current_calibration.perspective_correction
            ):
                processed_frame = self.geometry_calibrator.correct_keystone_distortion(
                    processed_frame,
                    self.geometry_calibrator.current_calibration.perspective_correction,
                )

        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")

        return processed_frame

    def pixel_to_world(
        self, pixel_point: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert pixel coordinates to world coordinates.

        Args:
            pixel_point: Point in pixel coordinates

        Returns:
            Point in world coordinates or None if not available
        """
        return self.geometry_calibrator.pixel_to_world_coordinates(pixel_point)

    def world_to_pixel(
        self, world_point: tuple[float, float]
    ) -> Optional[tuple[float, float]]:
        """Convert world coordinates to pixel coordinates.

        Args:
            world_point: Point in world coordinates

        Returns:
            Point in pixel coordinates or None if not available
        """
        return self.geometry_calibrator.world_to_pixel_coordinates(world_point)

    def is_system_ready(self) -> bool:
        """Check if calibration system is ready for operation.

        Returns:
            True if system is calibrated and validated
        """
        if self.current_session:
            return self.current_session.system_ready

        # Check individual components
        camera_ready = self.camera_calibrator.camera_params is not None
        color_ready = self.color_calibrator.current_profile is not None
        geometry_ready = self.geometry_calibrator.current_calibration is not None

        return camera_ready and color_ready and geometry_ready

    def get_calibration_status(self) -> dict[str, Any]:
        """Get current calibration status.

        Returns:
            Dictionary with calibration status information
        """
        status = {
            "session_active": self.current_session is not None,
            "camera_calibrated": self.camera_calibrator.camera_params is not None,
            "color_calibrated": self.color_calibrator.current_profile is not None,
            "geometry_calibrated": self.geometry_calibrator.current_calibration
            is not None,
            "system_ready": self.is_system_ready(),
        }

        if self.current_session:
            status["session_id"] = self.current_session.session_id
            status["session_date"] = self.current_session.created_date

        if self.camera_calibrator.camera_params:
            status[
                "camera_error"
            ] = self.camera_calibrator.camera_params.calibration_error

        if self.color_calibrator.current_profile:
            status["color_profile"] = self.color_calibrator.current_profile.name
            status[
                "lighting_level"
            ] = self.color_calibrator.current_profile.ambient_light_level

        if self.geometry_calibrator.current_calibration:
            status[
                "geometry_error"
            ] = self.geometry_calibrator.current_calibration.calibration_error

        return status

    def save_current_session(self) -> bool:
        """Save current calibration session.

        Returns:
            True if saved successfully
        """
        if not self.current_session:
            logger.warning("No active session to save")
            return False

        try:
            session_file = self.sessions_dir / f"{self.current_session.session_id}.json"
            with open(session_file, "w") as f:
                json.dump(self.current_session.to_dict(), f, indent=2)

            logger.info(f"Session saved: {session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self, session_id: str) -> bool:
        """Load calibration session.

        Args:
            session_id: Session ID to load

        Returns:
            True if loaded successfully
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                logger.error(f"Session file not found: {session_file}")
                return False

            with open(session_file) as f:
                data = json.load(f)

            session = CalibrationSession.from_dict(data)

            # Load individual calibrations
            if session.camera_calibration:
                self.camera_calibrator.camera_params = session.camera_calibration

            if session.color_profile:
                self.color_calibrator.current_profile = session.color_profile

            if session.geometric_calibration:
                self.geometry_calibrator.current_calibration = (
                    session.geometric_calibration
                )

            self.current_session = session
            logger.info(f"Session loaded: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def list_sessions(self) -> list[str]:
        """List available calibration sessions.

        Returns:
            List of session IDs
        """
        session_files = list(self.sessions_dir.glob("*.json"))
        return [f.stem for f in session_files]

    def export_calibration(self, export_path: str) -> bool:
        """Export complete calibration data.

        Args:
            export_path: Path to export file

        Returns:
            True if exported successfully
        """
        if not self.current_session:
            logger.error("No active session to export")
            return False

        try:
            export_data = {
                "export_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "session": self.current_session.to_dict(),
            }

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Calibration exported to: {export_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def import_calibration(self, import_path: str) -> bool:
        """Import calibration data.

        Args:
            import_path: Path to import file

        Returns:
            True if imported successfully
        """
        try:
            with open(import_path) as f:
                data = json.load(f)

            if "session" not in data:
                logger.error("Invalid import file format")
                return False

            session = CalibrationSession.from_dict(data["session"])

            # Load individual calibrations
            if session.camera_calibration:
                self.camera_calibrator.camera_params = session.camera_calibration

            if session.color_profile:
                self.color_calibrator.current_profile = session.color_profile

            if session.geometric_calibration:
                self.geometry_calibrator.current_calibration = (
                    session.geometric_calibration
                )

            # Update session ID for imported data
            import uuid

            session.session_id = f"imported_{str(uuid.uuid4())[:8]}"
            session.created_date = datetime.now().isoformat()

            self.current_session = session
            self.save_current_session()

            logger.info(f"Calibration imported from: {import_path}")
            return True

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False

    def reset_calibration(self, component: Optional[str] = None) -> bool:
        """Reset calibration data.

        Args:
            component: Specific component to reset ('camera', 'color', 'geometry') or None for all

        Returns:
            True if reset successfully
        """
        try:
            if component is None or component == "camera":
                self.camera_calibrator.camera_params = None
                if self.current_session:
                    self.current_session.camera_calibration = None

            if component is None or component == "color":
                self.color_calibrator.current_profile = None
                if self.current_session:
                    self.current_session.color_profile = None

            if component is None or component == "geometry":
                self.geometry_calibrator.current_calibration = None
                if self.current_session:
                    self.current_session.geometric_calibration = None

            if component is None and self.current_session:
                self.current_session.validation_report = None
                self.current_session.system_ready = False

            logger.info(f"Reset calibration: {component or 'all components'}")
            return True

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return False

    def adaptive_recalibration(
        self, frame: np.ndarray, lighting_threshold: float = 0.3
    ) -> bool:
        """Perform adaptive recalibration based on current conditions.

        Args:
            frame: Current frame for analysis
            lighting_threshold: Threshold for lighting change detection

        Returns:
            True if recalibration was performed
        """
        if not self.color_calibrator.current_profile:
            return False

        try:
            # Check lighting conditions
            current_light = self.color_calibrator._estimate_ambient_light(frame)
            reference_light = self.color_calibrator.current_profile.ambient_light_level

            light_ratio = abs(current_light - reference_light) / reference_light

            if light_ratio > lighting_threshold:
                logger.info(
                    f"Lighting change detected ({light_ratio:.2f}), adapting colors"
                )

                # Adapt color calibration
                adapted_profile = self.color_calibrator.adapt_to_lighting(
                    frame, self.color_calibrator.current_profile
                )

                self.color_calibrator.current_profile = adapted_profile

                if self.current_session:
                    self.current_session.color_profile = adapted_profile
                    self.save_current_session()

                return True

        except Exception as e:
            logger.warning(f"Adaptive recalibration failed: {e}")

        return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save session if active."""
        if self.current_session:
            self.save_current_session()
