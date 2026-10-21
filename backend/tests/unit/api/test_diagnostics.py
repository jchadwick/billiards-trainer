"""Tests for hardware-free diagnostic behavior."""

from pathlib import Path
from types import SimpleNamespace

from backend.api.routes import diagnostics
from backend.vision.calibration.camera import CameraCalibrator


def test_fisheye_diagnostic_reads_canonical_camera_parameters(monkeypatch):
    """Calibration diagnostics use camera_params without deprecated attributes."""
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: path.name == "camera_fisheye_default.yaml",
    )

    def initialize_calibrator(calibrator):
        calibrator.camera_params = None

    def load_calibration(calibrator, _path):
        calibrator.camera_params = SimpleNamespace(
            camera_matrix=object(), distortion_coefficients=object()
        )
        return True

    monkeypatch.setattr(CameraCalibrator, "__init__", initialize_calibrator)
    monkeypatch.setattr(CameraCalibrator, "load_calibration", load_calibration)

    result = diagnostics._run_fisheye_diagnostic_sync()

    assert result["score"] == 100.0
    assert result["details"]["has_camera_matrix"] is True
    assert result["details"]["has_distortion_coeffs"] is True
