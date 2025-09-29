"""Integration tests for validation utilities with configuration system."""

from pathlib import Path

import pytest

from backend.config.models.schemas import (
    ApplicationConfig,
    CameraSettings,
    VisionConfig,
    create_development_config,
)
from backend.config.utils.differ import ConfigDiffer
from backend.config.validator.rules import ValidationRules
from backend.config.validator.types import TypeChecker


class TestValidationIntegration:
    """Test integration of validation utilities with configuration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ValidationRules()
        self.type_checker = TypeChecker()
        self.differ = ConfigDiffer()

    def test_camera_settings_validation(self):
        """Test validating camera settings with ValidationRules."""
        # Valid camera settings
        valid_camera = {
            "device_id": 0,
            "resolution": (1920, 1080),
            "fps": 30,
            "gain": 1.0,
            "brightness": 0.5,
        }

        # Test range validation
        assert self.validator.check_range(valid_camera["device_id"], 0, 10, "device_id")
        assert self.validator.check_range(valid_camera["fps"], 15, 120, "fps")
        assert self.validator.check_range(valid_camera["gain"], 0.0, 10.0, "gain")
        assert self.validator.check_range(
            valid_camera["brightness"], 0.0, 1.0, "brightness"
        )

        # Test type validation
        assert self.validator.check_type(valid_camera["device_id"], int, "device_id")
        assert self.validator.check_type(
            valid_camera["resolution"], tuple, "resolution"
        )
        assert self.validator.check_type(valid_camera["fps"], int, "fps")

        # Test invalid values
        assert not self.validator.check_range(-1, 0, 10, "device_id")  # Below min
        assert not self.validator.check_range(200, 15, 120, "fps")  # Above max

    def test_type_checker_with_pydantic_models(self):
        """Test TypeChecker with Pydantic configuration models."""
        # Test CameraSettings type checking
        camera_dict = {
            "device_id": 0,
            "resolution": (1920, 1080),
            "fps": 30,
            "gain": 1.0,
            "brightness": 0.5,
        }

        # Should pass type checking for dict to Pydantic model
        assert self.type_checker.check(camera_dict, CameraSettings)

        # Test invalid camera settings
        invalid_camera = {
            "device_id": "not_an_int",
            "resolution": "not_a_tuple",
            "fps": -10,
        }

        assert not self.type_checker.check(invalid_camera, CameraSettings)

    def test_config_diffing_with_real_configs(self):
        """Test configuration diffing with realistic configuration changes."""
        # Original configuration
        config_v1 = {
            "vision": {
                "camera": {"device_id": 0, "fps": 30, "resolution": (1920, 1080)},
                "detection": {"min_ball_radius": 10, "max_ball_radius": 40},
            },
            "core": {"physics": {"gravity": 9.81, "friction": 0.01}},
        }

        # Updated configuration
        config_v2 = {
            "vision": {
                "camera": {
                    "device_id": 1,  # Changed
                    "fps": 60,  # Changed
                    "resolution": (1920, 1080),  # Unchanged
                },
                "detection": {
                    "min_ball_radius": 15,  # Changed
                    "max_ball_radius": 40,  # Unchanged
                    "sensitivity": 0.8,  # Added
                },
            },
            "core": {
                "physics": {"gravity": 9.81, "friction": 0.02}  # Unchanged  # Changed
            },
        }

        diff_result = self.differ.diff(config_v1, config_v2)

        # Check that changes were detected
        assert diff_result["summary"].total_changes > 0
        assert diff_result["summary"].added_count >= 1  # sensitivity added
        assert (
            diff_result["summary"].modified_count >= 3
        )  # device_id, fps, friction, min_ball_radius

        # Check specific change paths
        change_paths = [change.path for change in diff_result["changes"]]
        assert "vision.camera.device_id" in change_paths
        assert "vision.camera.fps" in change_paths
        assert "vision.detection.sensitivity" in change_paths
        assert "core.physics.friction" in change_paths

    def test_comprehensive_config_validation(self):
        """Test comprehensive validation of a complete configuration."""
        # Create a development configuration
        dev_config = create_development_config()
        config_dict = dev_config.model_dump()

        # Test type checking against the full application schema
        assert self.type_checker.check(config_dict, ApplicationConfig)

        # Test specific subsection validations
        vision_config = config_dict.get("vision", {})
        assert self.type_checker.check(vision_config, VisionConfig)

        camera_config = vision_config.get("camera", {})
        assert self.type_checker.check(camera_config, CameraSettings)

    def test_validation_error_reporting(self):
        """Test comprehensive error reporting across validation utilities."""
        # Invalid configuration with multiple errors
        invalid_config = {
            "vision": {
                "camera": {
                    "device_id": -1,  # Out of range
                    "fps": "not_int",  # Wrong type
                    "gain": 15.0,  # Out of range
                },
                "detection": {"min_ball_radius": "not_numeric"},  # Wrong type
            }
        }

        # Collect validation errors
        self.validator.clear_validation_errors()

        # Range validations
        self.validator.check_range(
            invalid_config["vision"]["camera"]["device_id"], 0, 10, "device_id"
        )
        self.validator.check_range(
            invalid_config["vision"]["camera"]["gain"], 0.0, 10.0, "gain"
        )

        # Type validations
        self.validator.check_type(invalid_config["vision"]["camera"]["fps"], int, "fps")
        self.validator.check_type(
            invalid_config["vision"]["detection"]["min_ball_radius"],
            (int, float),
            "min_ball_radius",
        )

        errors = self.validator.get_validation_errors()
        assert len(errors) >= 4  # Should have collected multiple errors

        # Check that errors contain helpful information
        error_text = " ".join(errors)
        assert "device_id" in error_text
        assert "fps" in error_text
        assert "gain" in error_text
        assert "min_ball_radius" in error_text
