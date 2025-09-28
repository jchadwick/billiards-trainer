"""Comprehensive tests for calibration system."""

import tempfile
import unittest

import cv2
import numpy as np

from .camera import CameraCalibrator, CameraParameters, TableTransform
from .color import ColorCalibrator, ColorProfile, ColorThresholds
from .geometry import GeometricCalibrator
from .validation import CalibrationValidator


class TestCameraCalibration(unittest.TestCase):
    """Test camera calibration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.calibrator = CameraCalibrator(self.temp_dir)

    def create_synthetic_chessboard_images(
        self, num_images: int = 15
    ) -> list[np.ndarray]:
        """Create synthetic chessboard images for testing."""
        images = []
        pattern_size = self.calibrator.chessboard_size
        square_size = int(self.calibrator.square_size * 1000)  # Convert to mm

        for i in range(num_images):
            # Create a synthetic chessboard
            board_width = pattern_size[0] * square_size + 100
            board_height = pattern_size[1] * square_size + 100

            # Create chessboard pattern
            chessboard = np.zeros((board_height, board_width), dtype=np.uint8)

            for row in range(pattern_size[1] + 1):
                for col in range(pattern_size[0] + 1):
                    if (row + col) % 2 == 0:
                        y1 = row * square_size + 50
                        y2 = (row + 1) * square_size + 50
                        x1 = col * square_size + 50
                        x2 = (col + 1) * square_size + 50
                        chessboard[y1:y2, x1:x2] = 255

            # Add some perspective distortion
            h, w = chessboard.shape
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

            # Random perspective transformation
            offset = 20 * (i % 3)  # Vary perspective
            pts2 = np.float32(
                [[offset, offset], [w - offset, offset], [w, h], [0, h - offset]]
            )

            M = cv2.getPerspectiveTransform(pts1, pts2)
            transformed = cv2.warpPerspective(chessboard, M, (w, h))

            # Convert to color image
            color_image = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

            # Add some noise
            noise = np.random.normal(0, 10, color_image.shape).astype(np.uint8)
            noisy_image = cv2.add(color_image, noise)

            images.append(noisy_image)

        return images

    def test_camera_calibration_success(self):
        """Test successful camera calibration."""
        # Create synthetic images
        test_images = self.create_synthetic_chessboard_images(15)

        # Perform calibration
        success, camera_params = self.calibrator.calibrate_intrinsics(test_images)

        assert success, "Camera calibration should succeed with valid images"
        assert isinstance(camera_params, CameraParameters)
        assert camera_params.camera_matrix is not None
        assert camera_params.distortion_coefficients is not None
        assert (
            camera_params.calibration_error < 5.0
        ), "Calibration error should be reasonable"

    def test_camera_calibration_insufficient_images(self):
        """Test calibration failure with insufficient images."""
        # Create too few images
        test_images = self.create_synthetic_chessboard_images(5)

        success, camera_params = self.calibrator.calibrate_intrinsics(test_images)

        assert not success, "Calibration should fail with insufficient images"
        assert camera_params is None

    def test_table_transformation(self):
        """Test table coordinate transformation."""
        # Create a test image
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Define table corners (simulated)
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]

        # Calculate transformation
        transform = self.calibrator.calibrate_table_transform(test_image, table_corners)

        assert isinstance(transform, TableTransform)
        assert len(transform.table_corners_pixel) == 4
        assert transform.homography_matrix is not None

    def test_undistortion(self):
        """Test image undistortion."""
        # Create test camera parameters
        camera_matrix = np.array(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32
        )
        distortion = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        # Create test image
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        # Apply undistortion
        undistorted = self.calibrator.undistort_image(
            test_image, camera_matrix, distortion
        )

        assert undistorted.shape[:2] == test_image.shape[:2]
        assert isinstance(undistorted, np.ndarray)

    def test_coordinate_transformation(self):
        """Test pixel to world coordinate transformation."""
        # Set up table transformation
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        self.calibrator.calibrate_table_transform(test_image, table_corners)

        # Test transformation
        pixel_point = (400, 300)  # Center of table
        world_point = self.calibrator.pixel_to_world(pixel_point)

        assert world_point is not None
        assert len(world_point) == 2

        # Test reverse transformation
        back_to_pixel = self.calibrator.world_to_pixel(world_point)
        assert back_to_pixel is not None

        # Check round-trip accuracy
        error = np.linalg.norm(np.array(pixel_point) - np.array(back_to_pixel))
        assert error < 5.0, "Round-trip transformation error should be small"

    def test_calibration_persistence(self):
        """Test saving and loading calibration data."""
        # Create and save camera parameters
        camera_matrix = np.array(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32
        )
        distortion = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        from datetime import datetime

        camera_params = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            resolution=(800, 600),
            calibration_error=1.5,
            calibration_date=datetime.now().isoformat(),
        )

        self.calibrator.camera_params = camera_params
        self.calibrator._save_camera_params()

        # Create new calibrator and load
        new_calibrator = CameraCalibrator(self.temp_dir)
        success = new_calibrator.load_camera_params()

        assert success, "Should successfully load camera parameters"
        assert new_calibrator.camera_params is not None
        np.testing.assert_array_almost_equal(
            new_calibrator.camera_params.camera_matrix, camera_matrix
        )


class TestColorCalibration(unittest.TestCase):
    """Test color calibration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.calibrator = ColorCalibrator(self.temp_dir)

    def create_synthetic_table_image(self) -> np.ndarray:
        """Create synthetic table image for testing."""
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Create green table surface
        table_color = (60, 180, 100)  # Green in HSV
        image[:, :] = table_color

        # Add some balls
        ball_positions = [(200, 200), (400, 300), (600, 400)]
        ball_colors = [
            (0, 255, 255),
            (120, 255, 255),
            (30, 255, 255),
        ]  # White, blue, yellow

        for pos, color in zip(ball_positions, ball_colors):
            cv2.circle(image, pos, 20, color, -1)

        # Convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def test_auto_table_color_calibration(self):
        """Test automatic table color calibration."""
        test_image = self.create_synthetic_table_image()

        thresholds = self.calibrator.auto_calibrate_table_color(test_image)

        assert isinstance(thresholds, ColorThresholds)
        assert thresholds.hue_min >= 0
        assert thresholds.hue_max <= 179
        assert thresholds.saturation_min >= 0
        assert thresholds.saturation_max <= 255

    def test_ball_color_calibration(self):
        """Test ball color calibration."""
        test_image = self.create_synthetic_table_image()

        # Define sample regions for different ball types
        ball_samples = {
            "cue": [(180, 180, 40, 40)],  # White ball region
            "blue": [(380, 280, 40, 40)],  # Blue ball region
            "yellow": [(580, 380, 40, 40)],  # Yellow ball region
        }

        thresholds = self.calibrator.calibrate_ball_colors(test_image, ball_samples)

        assert "cue" in thresholds
        assert "blue" in thresholds
        assert "yellow" in thresholds

        for _ball_type, threshold in thresholds.items():
            assert isinstance(threshold, ColorThresholds)

    def test_color_threshold_mask_application(self):
        """Test color threshold mask application."""
        test_image = self.create_synthetic_table_image()
        hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

        # Create threshold for green table
        thresholds = ColorThresholds(45, 75, 100, 255, 50, 255)

        mask = thresholds.apply_mask(hsv_image)

        assert mask.shape == hsv_image.shape[:2]
        assert mask.dtype == np.uint8
        assert np.sum(mask > 0) > 1000, "Should detect significant green area"

    def test_lighting_adaptation(self):
        """Test lighting adaptation."""
        test_image = self.create_synthetic_table_image()

        # Create reference profile
        profile = self.calibrator.create_default_profile(test_image, "test_profile")

        # Simulate different lighting (brighten image)
        bright_image = cv2.convertScaleAbs(test_image, alpha=1.5, beta=20)

        # Adapt to new lighting
        adapted_profile = self.calibrator.adapt_to_lighting(bright_image, profile)

        assert isinstance(adapted_profile, ColorProfile)
        assert adapted_profile.name != profile.name
        assert adapted_profile.ambient_light_level != profile.ambient_light_level

    def test_profile_persistence(self):
        """Test color profile saving and loading."""
        test_image = self.create_synthetic_table_image()
        profile = self.calibrator.create_default_profile(test_image, "test_profile")

        # Save profile
        success = self.calibrator.save_profile(profile)
        assert success, "Should successfully save profile"

        # Load profile
        loaded_profile = self.calibrator.load_profile(
            f"{profile.name}_{profile.creation_date[:10]}.json"
        )
        assert loaded_profile is not None
        assert loaded_profile.name == profile.name

    def test_hue_wraparound_handling(self):
        """Test handling of hue wraparound (red colors)."""
        # Create thresholds that wrap around (red color)
        thresholds = ColorThresholds(170, 10, 100, 255, 100, 255)

        # Create test image with red color
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = (5, 200, 200)  # Red in HSV
        hsv_image = cv2.cvtColor(test_image, cv2.COLOR_HSV2BGR)
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

        mask = thresholds.apply_mask(hsv_image)

        # Should detect red pixels
        assert np.sum(mask > 0) > 5000, "Should detect red color with wraparound"


class TestGeometricCalibration(unittest.TestCase):
    """Test geometric calibration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.calibrator = GeometricCalibrator(self.temp_dir)

    def create_synthetic_table_image(self) -> np.ndarray:
        """Create synthetic table image with clear boundaries."""
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Create table with distinct edges
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]

        # Fill table area
        pts = np.array(table_corners, np.int32)
        cv2.fillPoly(image, [pts], (0, 150, 0))

        # Draw clear boundaries
        cv2.polylines(image, [pts], True, (255, 255, 255), 3)

        return image

    def test_table_corner_detection(self):
        """Test automatic table corner detection."""
        test_image = self.create_synthetic_table_image()

        corners = self.calibrator.detect_table_corners(test_image)

        assert len(corners) == 4, "Should detect 4 corners"
        for corner in corners:
            assert len(corner) == 2, "Each corner should have x, y coordinates"

    def test_manual_corner_specification(self):
        """Test manual corner specification."""
        test_image = self.create_synthetic_table_image()
        manual_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]

        corners = self.calibrator.detect_table_corners(test_image, manual_corners)

        assert corners == manual_corners, "Should use manual corners when provided"

    def test_perspective_transform_calculation(self):
        """Test perspective transformation calculation."""
        source_points = [(100, 100), (700, 100), (700, 500), (100, 500)]

        correction = self.calibrator.calculate_perspective_transform(source_points)

        assert correction.transform_matrix is not None
        assert correction.inverse_matrix is not None
        assert correction.transform_matrix.shape == (3, 3)
        assert correction.correction_quality > 0.0

    def test_keystone_correction(self):
        """Test keystone distortion correction."""
        test_image = self.create_synthetic_table_image()
        source_points = [(100, 100), (700, 100), (700, 500), (100, 500)]

        correction = self.calibrator.calculate_perspective_transform(source_points)
        corrected_image = self.calibrator.correct_keystone_distortion(
            test_image, correction
        )

        assert isinstance(corrected_image, np.ndarray)
        assert len(corrected_image.shape) == 3

    def test_coordinate_mapping_creation(self):
        """Test coordinate mapping creation."""
        pixel_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        world_dimensions = (2.54, 1.27)  # Standard table

        mapping = self.calibrator.create_coordinate_mapping(
            pixel_corners, world_dimensions
        )

        assert mapping is not None
        assert mapping.scale_factor > 0
        assert len(mapping.pixel_bounds) == 2
        assert len(mapping.world_bounds) == 2

    def test_coordinate_transformations(self):
        """Test pixel-world coordinate transformations."""
        # Setup calibration
        test_image = self.create_synthetic_table_image()
        corners = [(100, 100), (700, 100), (700, 500), (100, 500)]

        self.calibrator.calibrate_table_geometry(test_image, corners)

        # Test center point transformation
        center_pixel = (400, 300)
        world_point = self.calibrator.pixel_to_world_coordinates(center_pixel)

        assert world_point is not None
        assert len(world_point) == 2

        # Test reverse transformation
        back_to_pixel = self.calibrator.world_to_pixel_coordinates(world_point)
        assert back_to_pixel is not None

        # Check accuracy
        error = np.linalg.norm(np.array(center_pixel) - np.array(back_to_pixel))
        assert error < 10.0, "Round-trip error should be small"

    def test_geometric_validation(self):
        """Test geometric calibration validation."""
        # Setup calibration
        test_image = self.create_synthetic_table_image()
        corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        self.calibrator.calibrate_table_geometry(test_image, corners)

        # Create test points
        test_points = [(200, 200), (400, 300), (600, 400)]
        expected_world = [(-0.85, -0.42), (0.0, 0.0), (0.85, 0.42)]

        validation_result = self.calibrator.validate_geometry(
            test_points, expected_world
        )

        assert "mean_error" in validation_result
        assert "max_error" in validation_result
        assert validation_result["num_test_points"] > 0

    def test_calibration_persistence(self):
        """Test geometric calibration saving and loading."""
        test_image = self.create_synthetic_table_image()
        corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        self.calibrator.calibrate_table_geometry(test_image, corners)

        # Save calibration
        assert self.calibrator.export_calibration("test_geometry.json")

        # Load in new calibrator
        new_calibrator = GeometricCalibrator(self.temp_dir)
        success = new_calibrator.load_calibration()

        assert success, "Should successfully load calibration"
        assert new_calibrator.current_calibration is not None


class TestCalibrationValidation(unittest.TestCase):
    """Test calibration validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = CalibrationValidator(self.temp_dir)

    def test_validation_result_serialization(self):
        """Test validation result serialization."""
        import time

        from .validation import ValidationResult

        result = ValidationResult(
            test_name="test_validation",
            timestamp=time.time(),
            passed=True,
            accuracy_score=0.85,
            error_metrics={"test_error": 0.15},
            details={"test_detail": "success"},
        )

        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        # Test deserialization
        restored_result = ValidationResult.from_dict(result_dict)
        assert restored_result.test_name == result.test_name
        assert restored_result.passed == result.passed

    def test_camera_validation_no_params(self):
        """Test camera validation without parameters."""
        camera_calibrator = CameraCalibrator()
        test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]

        result = self.validator.validate_camera_calibration(
            camera_calibrator, test_images
        )

        assert not result.passed
        assert result.accuracy_score == 0.0

    def test_color_validation_no_profile(self):
        """Test color validation without profile."""
        color_calibrator = ColorCalibrator()
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ground_truth = {"table": [(100, 100, 200, 200)]}

        result = self.validator.validate_color_calibration(
            color_calibrator, test_frame, ground_truth
        )

        assert not result.passed
        assert result.accuracy_score == 0.0

    def test_geometry_validation_no_calibration(self):
        """Test geometry validation without calibration."""
        geometry_calibrator = GeometricCalibrator()
        test_points = [(100, 100), (200, 200)]
        expected_points = [(0.1, 0.1), (0.2, 0.2)]

        result = self.validator.validate_geometric_calibration(
            geometry_calibrator, test_points, expected_points
        )

        assert not result.passed
        assert result.accuracy_score == 0.0

    def test_comprehensive_report_generation(self):
        """Test comprehensive validation report generation."""
        from .validation import CalibrationReport

        # Create calibrators
        camera_calibrator = CameraCalibrator()
        color_calibrator = ColorCalibrator()
        geometry_calibrator = GeometricCalibrator()

        # Empty test data
        test_data = {}

        report = self.validator.generate_comprehensive_report(
            camera_calibrator, color_calibrator, geometry_calibrator, test_data
        )

        assert isinstance(report, CalibrationReport)
        assert report.session_id is not None
        assert report.test_date is not None
        assert isinstance(report.recommendations, list)

    def test_stability_test(self):
        """Test calibration stability testing."""
        # Create mock calibrators
        calibrators = {
            "camera": CameraCalibrator(),
            "color": ColorCalibrator(),
            "geometry": GeometricCalibrator(),
        }

        # Create test frames
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]

        # Run short stability test (1 second)
        results = self.validator.stability_test(
            calibrators, test_frames, duration_minutes=0.02
        )

        assert isinstance(results, dict)
        assert "test_duration" in results
        assert "total_frames_tested" in results


class TestCalibrationIntegration(unittest.TestCase):
    """Test integration between calibration components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_full_calibration_pipeline(self):
        """Test complete calibration pipeline."""
        # Initialize all calibrators
        camera_calibrator = CameraCalibrator(self.temp_dir)
        color_calibrator = ColorCalibrator(self.temp_dir)
        geometry_calibrator = GeometricCalibrator(self.temp_dir)

        # Create synthetic test data
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)

        # 1. Test camera calibration (simplified)
        camera_matrix = np.array(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32
        )
        distortion = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        from datetime import datetime

        camera_params = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            resolution=(800, 600),
            calibration_error=1.5,
            calibration_date=datetime.now().isoformat(),
        )
        camera_calibrator.camera_params = camera_params

        # 2. Test color calibration
        color_profile = color_calibrator.create_default_profile(
            test_image, "integration_test"
        )
        assert color_profile is not None

        # 3. Test geometry calibration
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        geometry_calibration = geometry_calibrator.calibrate_table_geometry(
            test_image, table_corners
        )
        assert geometry_calibration is not None

        # 4. Test coordinate transformation integration
        test_pixel = (400, 300)

        # Apply camera undistortion
        undistorted_image = camera_calibrator.undistort_image(test_image)
        assert undistorted_image.shape == test_image.shape

        # Convert pixel to world coordinates
        world_point = geometry_calibrator.pixel_to_world_coordinates(test_pixel)
        assert world_point is not None

        # Convert back to pixels
        back_to_pixel = geometry_calibrator.world_to_pixel_coordinates(world_point)
        assert back_to_pixel is not None

    def test_calibration_data_consistency(self):
        """Test consistency of calibration data across components."""
        camera_calibrator = CameraCalibrator(self.temp_dir)
        geometry_calibrator = GeometricCalibrator(self.temp_dir)

        # Setup camera calibration
        camera_matrix = np.array(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32
        )
        distortion = np.array(
            [0.0, 0.0, 0, 0, 0], dtype=np.float32
        )  # No distortion for simplicity

        from datetime import datetime

        camera_params = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            resolution=(800, 600),
            calibration_error=0.5,
            calibration_date=datetime.now().isoformat(),
        )
        camera_calibrator.camera_params = camera_params

        # Setup geometry calibration
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]
        geometry_calibrator.calibrate_table_geometry(test_image, table_corners)

        # Test that image dimensions are consistent
        camera_resolution = camera_calibrator.camera_params.resolution
        geometry_corners = geometry_calibrator.current_calibration.table_corners_pixel

        # Check corners are within image bounds
        for corner in geometry_corners:
            assert corner[0] >= 0
            assert corner[1] >= 0
            assert corner[0] < camera_resolution[0]
            assert corner[1] < camera_resolution[1]

    def test_error_propagation(self):
        """Test error propagation through calibration pipeline."""
        camera_calibrator = CameraCalibrator(self.temp_dir)
        geometry_calibrator = GeometricCalibrator(self.temp_dir)

        # Create camera calibration with known error
        camera_matrix = np.array(
            [[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32
        )
        distortion = np.array([0.1, -0.1, 0, 0, 0], dtype=np.float32)

        from datetime import datetime

        camera_params = CameraParameters(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            resolution=(800, 600),
            calibration_error=2.0,  # Higher error
            calibration_date=datetime.now().isoformat(),
        )
        camera_calibrator.camera_params = camera_params

        # Test coordinate transformation accuracy with camera error
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        table_corners = [(100, 100), (700, 100), (700, 500), (100, 500)]

        # Apply distortion correction before geometry calibration
        corrected_image = camera_calibrator.undistort_image(test_image)
        geometry_calibration = geometry_calibrator.calibrate_table_geometry(
            corrected_image, table_corners
        )

        # The geometry calibration error should account for camera calibration error
        assert geometry_calibration.calibration_error is not None
        assert geometry_calibration.calibration_error >= 0


def run_calibration_tests():
    """Run all calibration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCameraCalibration,
        TestColorCalibration,
        TestGeometricCalibration,
        TestCalibrationValidation,
        TestCalibrationIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_calibration_tests()
    sys.exit(0 if success else 1)
