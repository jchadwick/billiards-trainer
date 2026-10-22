"""Tests for canonical API route registration."""

from backend.tests.support.compat_api import create_compatibility_app

app = create_compatibility_app()


def test_unimplemented_module_lifecycle_routes_are_not_registered() -> None:
    """Removed fake module controls must not remain exposed as API routes."""
    registered_paths = {getattr(route, "path", "") for route in app.routes}

    assert not any(path == "/api/v1/modules" for path in registered_paths)
    assert not any(path.startswith("/api/v1/modules/") for path in registered_paths)


def test_camera_calibration_routes_expose_supported_chessboard_workflow_only() -> None:
    """Keep supported capture routes registered and remove dead camera actions."""
    registered_paths = set(app.openapi()["paths"])

    assert {
        "/api/v1/vision/calibration/camera/mode/start",
        "/api/v1/vision/calibration/camera/images/capture",
        "/api/v1/vision/calibration/camera/process",
        "/api/v1/vision/calibration/camera/mode/stop",
    } <= registered_paths
    assert "/api/v1/vision/calibration/camera/apply" not in registered_paths
    assert "/api/v1/vision/calibration/camera/auto-calibrate" not in registered_paths
