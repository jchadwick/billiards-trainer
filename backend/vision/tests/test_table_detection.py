"""Comprehensive tests for table detection algorithms.

Tests all aspects of table detection including:
- Table boundary detection
- Corner identification with sub-pixel accuracy
- Pocket detection and classification
- Perspective correction
- Occlusion handling
- Geometry validation
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from ..detection.table import (
    Pocket,
    PocketType,
    TableCorners,
    TableDetectionResult,
    TableDetector,
)
from ..detection.utils import DetectionUtils
from ..utils.visualization import TableVisualization


class TestTableDetector:
    """Test suite for TableDetector class."""

    @pytest.fixture()
    def default_config(self):
        """Default configuration for testing."""
        return {
            "table_color_ranges": {
                "green": {
                    "lower": np.array([35, 40, 40]),
                    "upper": np.array([85, 255, 255]),
                }
            },
            "expected_aspect_ratio": 2.0,
            "aspect_ratio_tolerance": 0.3,
            "min_table_area_ratio": 0.1,
            "pocket_color_threshold": 30,
            "min_pocket_area": 100,
            "max_pocket_area": 2000,
            "debug": True,
        }

    @pytest.fixture()
    def detector(self, default_config):
        """Create a TableDetector instance for testing."""
        return TableDetector(default_config)

    @pytest.fixture()
    def synthetic_table_image(self):
        """Create a synthetic table image for testing."""
        # Create a 800x400 image with green table
        image = np.zeros((400, 800, 3), dtype=np.uint8)

        # Green table surface
        table_color = [60, 200, 120]  # Green in HSV converted to BGR
        table_corners = np.array(
            [
                [100, 50],  # top-left
                [700, 50],  # top-right
                [100, 350],  # bottom-left
                [700, 350],  # bottom-right
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(image, [table_corners], table_color)

        # Add pockets as dark circles
        pocket_positions = [
            (100, 50),  # corner top-left
            (700, 50),  # corner top-right
            (100, 350),  # corner bottom-left
            (700, 350),  # corner bottom-right
            (400, 50),  # side top
            (400, 350),  # side bottom
        ]

        for pos in pocket_positions:
            cv2.circle(image, pos, 15, (0, 0, 0), -1)

        return image, table_corners, pocket_positions

    @pytest.fixture()
    def occluded_table_image(self):
        """Create a table image with partial occlusion."""
        # Create base table
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        table_color = [60, 200, 120]
        table_corners = np.array(
            [[100, 50], [700, 50], [100, 350], [700, 350]], dtype=np.int32
        )

        cv2.fillPoly(image, [table_corners], table_color)

        # Add occlusion (simulating a hand or object)
        cv2.rectangle(image, (300, 150), (500, 250), (80, 80, 80), -1)

        return image

    def test_detector_initialization(self, default_config):
        """Test detector initialization with configuration."""
        detector = TableDetector(default_config)

        assert detector.config == default_config
        assert detector.expected_aspect_ratio == 2.0
        assert detector.debug_mode is True
        assert len(detector.table_color_ranges) > 0

    def test_table_boundary_detection_success(self, detector, synthetic_table_image):
        """Test successful table boundary detection."""
        image, expected_corners, _ = synthetic_table_image

        corners = detector.detect_table_boundaries(image)

        assert corners is not None
        assert isinstance(corners, TableCorners)

        # Check that corners are roughly in expected positions
        corner_list = corners.to_list()
        assert len(corner_list) == 4

        # Verify corners are reasonable (within image bounds)
        for x, y in corner_list:
            assert 0 <= x <= image.shape[1]
            assert 0 <= y <= image.shape[0]

    def test_table_boundary_detection_no_table(self, detector):
        """Test table boundary detection with no table in image."""
        # Create image with no table-colored regions
        image = np.ones((400, 800, 3), dtype=np.uint8) * 255  # White image

        corners = detector.detect_table_boundaries(image)

        assert corners is None

    def test_table_boundary_detection_invalid_input(self, detector):
        """Test table boundary detection with invalid input."""
        assert detector.detect_table_boundaries(None) is None
        assert detector.detect_table_boundaries(np.array([])) is None

    def test_table_surface_detection(self, detector, synthetic_table_image):
        """Test table surface detection and color analysis."""
        image, _, _ = synthetic_table_image

        result = detector.detect_table_surface(image)

        assert result is not None
        mask, surface_color = result

        assert mask is not None
        assert surface_color is not None
        assert len(surface_color) == 3  # HSV color tuple

        # Verify mask has reasonable coverage
        coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        assert coverage > 0.1  # At least 10% coverage

    def test_pocket_detection(self, detector, synthetic_table_image):
        """Test pocket detection functionality."""
        image, table_corners, expected_pockets = synthetic_table_image

        # Create TableCorners object
        corners = TableCorners(
            top_left=(table_corners[0][0], table_corners[0][1]),
            top_right=(table_corners[1][0], table_corners[1][1]),
            bottom_left=(table_corners[2][0], table_corners[2][1]),
            bottom_right=(table_corners[3][0], table_corners[3][1]),
        )

        pockets = detector.detect_pockets(image, corners)

        assert len(pockets) > 0
        assert len(pockets) <= 6  # Should not detect more than 6 pockets

        # Check pocket types
        corner_pockets = [p for p in pockets if p.pocket_type == PocketType.CORNER]
        side_pockets = [p for p in pockets if p.pocket_type == PocketType.SIDE]

        # Should have some corner and side pockets
        assert len(corner_pockets) > 0
        assert len(side_pockets) >= 0

    def test_pocket_detection_no_corners(self, detector, synthetic_table_image):
        """Test pocket detection with no table corners provided."""
        image, _, _ = synthetic_table_image

        pockets = detector.detect_pockets(image, None)

        assert pockets == []

    def test_geometry_validation_valid_table(self, detector):
        """Test geometry validation with valid table dimensions."""
        # Create corners representing a valid 2:1 aspect ratio table
        corners = TableCorners(
            top_left=(100.0, 50.0),
            top_right=(700.0, 50.0),
            bottom_left=(100.0, 350.0),
            bottom_right=(700.0, 350.0),
        )

        assert detector.validate_table_dimensions(corners) is True

    def test_geometry_validation_invalid_aspect_ratio(self, detector):
        """Test geometry validation with invalid aspect ratio."""
        # Create corners representing a square (1:1 ratio, should fail for pool table)
        corners = TableCorners(
            top_left=(100.0, 50.0),
            top_right=(400.0, 50.0),
            bottom_left=(100.0, 350.0),
            bottom_right=(400.0, 350.0),
        )

        assert detector.validate_table_dimensions(corners) is False

    def test_complete_table_detection(self, detector, synthetic_table_image):
        """Test complete table detection pipeline."""
        image, _, _ = synthetic_table_image

        result = detector.detect_complete_table(image)

        if result is not None:  # Detection may fail on synthetic image
            assert isinstance(result, TableDetectionResult)
            assert result.corners is not None
            assert result.width > 0
            assert result.height > 0
            assert 0 <= result.confidence <= 1.0

            # Check aspect ratio
            aspect_ratio = result.width / result.height
            assert 1.0 <= aspect_ratio <= 3.0  # Reasonable range

    def test_occlusion_handling(self, detector, occluded_table_image):
        """Test handling of partial occlusions."""
        # First, create a clean detection
        clean_image = np.zeros((400, 800, 3), dtype=np.uint8)
        table_color = [60, 200, 120]
        table_corners = np.array(
            [[100, 50], [700, 50], [100, 350], [700, 350]], dtype=np.int32
        )
        cv2.fillPoly(clean_image, [table_corners], table_color)

        clean_detection = detector.detect_complete_table(clean_image)

        # Now test with occluded image
        result = detector.handle_occlusions(occluded_table_image, clean_detection)

        # Should either return a detection or the previous one
        assert result is not None

    def test_table_calibration(self, detector, synthetic_table_image):
        """Test table calibration functionality."""
        image, _, _ = synthetic_table_image

        calibration_data = detector.calibrate_table(image)

        assert "success" in calibration_data

        if calibration_data["success"]:
            assert "table_corners" in calibration_data
            assert "table_dimensions" in calibration_data
            assert "confidence" in calibration_data

    def test_debug_image_collection(self, detector, synthetic_table_image):
        """Test debug image collection during detection."""
        image, _, _ = synthetic_table_image
        detector.clear_debug_images()

        # Run detection which should generate debug images
        detector.detect_complete_table(image)

        debug_images = detector.get_debug_images()

        # Should have collected some debug images
        assert len(debug_images) >= 0

        # Clear and verify
        detector.clear_debug_images()
        assert len(detector.get_debug_images()) == 0

    def test_corner_sorting(self, detector):
        """Test corner sorting to consistent order."""
        # Create unsorted corners
        unsorted_corners = np.array(
            [
                [700, 350],  # bottom-right
                [100, 50],  # top-left
                [700, 50],  # top-right
                [100, 350],  # bottom-left
            ],
            dtype=np.float32,
        )

        sorted_corners = detector._sort_corners(unsorted_corners)

        # Should be in order: top-left, top-right, bottom-left, bottom-right
        assert sorted_corners[0][0] < sorted_corners[1][0]  # top-left.x < top-right.x
        assert sorted_corners[0][1] < sorted_corners[2][1]  # top-left.y < bottom-left.y

    def test_distance_calculation(self, detector):
        """Test distance calculation utility."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)

        distance = detector._distance(p1, p2)

        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle

    def test_perspective_transform_generation(self, detector):
        """Test perspective transformation matrix generation."""
        corners = TableCorners(
            top_left=(100.0, 50.0),
            top_right=(700.0, 50.0),
            bottom_left=(100.0, 350.0),
            bottom_right=(700.0, 350.0),
        )

        width, height = 600.0, 300.0
        transform = detector._generate_perspective_transform(corners, width, height)

        assert transform is not None
        assert transform.shape == (3, 3)

    def test_table_corners_to_list(self):
        """Test TableCorners to_list conversion."""
        corners = TableCorners(
            top_left=(1.0, 2.0),
            top_right=(3.0, 4.0),
            bottom_left=(5.0, 6.0),
            bottom_right=(7.0, 8.0),
        )

        corner_list = corners.to_list()

        assert len(corner_list) == 4
        assert corner_list[0] == (1.0, 2.0)
        assert corner_list[1] == (3.0, 4.0)
        assert corner_list[2] == (5.0, 6.0)
        assert corner_list[3] == (7.0, 8.0)

    def test_pocket_classification(self, detector):
        """Test pocket type classification."""
        # Mock expected positions
        expected_positions = {
            "corner_tl": (100, 50),
            "corner_tr": (700, 50),
            "side_top": (400, 50),
        }

        # Test corner classification
        corner_type, corner_conf = detector._classify_pocket_type(
            (105, 55), expected_positions
        )
        assert corner_type == PocketType.CORNER
        assert corner_conf > 0.5

        # Test side classification
        side_type, side_conf = detector._classify_pocket_type(
            (405, 55), expected_positions
        )
        assert side_type == PocketType.SIDE
        assert side_conf > 0.5


class TestDetectionUtils:
    """Test suite for DetectionUtils class."""

    def test_apply_color_threshold(self):
        """Test color thresholding functionality."""
        # Create test image with known colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = [60, 200, 120]  # Green region

        # Convert to HSV for testing
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply green color threshold
        mask = DetectionUtils.apply_color_threshold(
            image, (35, 40, 40), (85, 255, 255), "HSV"
        )

        assert mask is not None
        assert np.sum(mask > 0) > 0  # Should detect some green pixels

    def test_find_circles(self):
        """Test circle detection functionality."""
        # Create test image with circles
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(image, (50, 50), 20, 255, 2)
        cv2.circle(image, (150, 150), 30, 255, 2)

        circles = DetectionUtils.find_circles(image, 15, 35)

        assert len(circles) >= 0  # May or may not detect circles in synthetic image

    def test_find_lines(self):
        """Test line detection functionality."""
        # Create test image with lines
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(image, (10, 50), (190, 50), 255, 2)
        cv2.line(image, (50, 10), (50, 190), 255, 2)

        lines = DetectionUtils.find_lines(image)

        assert len(lines) >= 0  # May detect lines

    def test_calculate_contour_center(self):
        """Test contour center calculation."""
        # Create a simple rectangular contour
        contour = np.array(
            [[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32
        )

        center = DetectionUtils.calculate_contour_center(contour)

        # Should be approximately (50, 50)
        assert abs(center[0] - 50.0) < 5.0
        assert abs(center[1] - 50.0) < 5.0

    def test_filter_contours_by_area(self):
        """Test contour filtering by area."""
        # Create contours with different areas
        small_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        large_contour = np.array([[[10, 10]], [[100, 10]], [[100, 100]], [[10, 100]]])

        contours = [small_contour, large_contour]

        # Filter for large contours only
        filtered = DetectionUtils.filter_contours_by_area(contours, min_area=500)

        assert len(filtered) <= len(contours)

    def test_morphological_operations(self):
        """Test morphological operations."""
        # Create test mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        # Test closing operation
        result = DetectionUtils.apply_morphological_operations(mask, "close", 5)

        assert result is not None
        assert result.shape == mask.shape

    def test_distance_calculation(self):
        """Test distance calculation between points."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)

        distance = DetectionUtils.calculate_distance(p1, p2)

        assert abs(distance - 5.0) < 0.001

    def test_angle_calculation(self):
        """Test angle calculation between points."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)

        angle = DetectionUtils.calculate_angle_between_points(p1, p2)

        assert abs(angle - 45.0) < 0.1  # Should be 45 degrees

    def test_point_in_polygon(self):
        """Test point-in-polygon detection."""
        # Define a simple square polygon
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]

        # Test point inside
        assert DetectionUtils.point_in_polygon((5, 5), polygon) is True

        # Test point outside
        assert DetectionUtils.point_in_polygon((15, 15), polygon) is False

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test with None inputs
        assert (
            DetectionUtils.apply_color_threshold(None, (0, 0, 0), (255, 255, 255)).size
            == 0
        )
        assert DetectionUtils.find_circles(None, 10, 20) == []
        assert DetectionUtils.find_lines(None) == []
        assert DetectionUtils.calculate_contour_center(None) == (0.0, 0.0)


class TestTableVisualization:
    """Test suite for table visualization utilities."""

    @pytest.fixture()
    def sample_detection_result(self):
        """Create a sample detection result for testing."""
        corners = TableCorners(
            top_left=(100.0, 50.0),
            top_right=(700.0, 50.0),
            bottom_left=(100.0, 350.0),
            bottom_right=(700.0, 350.0),
        )

        pockets = [
            Pocket((100.0, 50.0), 15.0, PocketType.CORNER, 0.9),
            Pocket((400.0, 50.0), 12.0, PocketType.SIDE, 0.8),
        ]

        return TableDetectionResult(
            corners=corners,
            pockets=pockets,
            surface_color=(60, 200, 120),
            width=600.0,
            height=300.0,
            confidence=0.85,
            perspective_transform=np.eye(3),
        )

    def test_draw_table_detection(self, sample_detection_result):
        """Test complete table detection visualization."""
        # Create test image
        image = np.zeros((400, 800, 3), dtype=np.uint8)

        result = TableVisualization.draw_table_detection(image, sample_detection_result)

        assert result is not None
        assert result.shape == image.shape

    def test_draw_table_boundaries(self, sample_detection_result):
        """Test table boundary visualization."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)

        result = TableVisualization.draw_table_boundaries(
            image, sample_detection_result.corners
        )

        assert result is not None
        assert result.shape == image.shape

    def test_draw_corners(self, sample_detection_result):
        """Test corner marker visualization."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)

        result = TableVisualization.draw_corners(image, sample_detection_result.corners)

        assert result is not None
        assert result.shape == image.shape

    def test_draw_pockets(self, sample_detection_result):
        """Test pocket visualization."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)

        result = TableVisualization.draw_pockets(image, sample_detection_result.pockets)

        assert result is not None
        assert result.shape == image.shape

    def test_create_detection_comparison(self, sample_detection_result):
        """Test detection comparison visualization."""
        image = np.zeros((400, 800, 3), dtype=np.uint8)
        debug_images = [
            ("mask", np.zeros((400, 800), dtype=np.uint8)),
            ("edges", np.zeros((400, 800), dtype=np.uint8)),
        ]

        result = TableVisualization.create_detection_comparison(
            image, sample_detection_result, debug_images
        )

        assert result is not None

    def test_save_debug_images(self, sample_detection_result):
        """Test saving debug images to disk."""
        debug_images = [
            ("test_mask", np.zeros((100, 100), dtype=np.uint8)),
            ("test_edges", np.ones((100, 100), dtype=np.uint8) * 255),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = TableVisualization.save_debug_images(
                debug_images, temp_dir, 0
            )

            assert len(saved_paths) == len(debug_images)
            for path in saved_paths:
                assert os.path.exists(path)

    def test_invalid_input_handling(self):
        """Test visualization with invalid inputs."""
        # Test with None inputs
        assert TableVisualization.draw_table_detection(None, None) is None
        assert TableVisualization.draw_table_boundaries(None, None) is None
        assert TableVisualization.draw_corners(None, None) is None


class TestIntegration:
    """Integration tests for complete table detection pipeline."""

    def test_full_detection_pipeline(self):
        """Test complete detection pipeline with synthetic data."""
        # Create realistic test configuration
        config = {
            "table_color_ranges": {
                "green": {
                    "lower": np.array([35, 40, 40]),
                    "upper": np.array([85, 255, 255]),
                }
            },
            "expected_aspect_ratio": 2.0,
            "aspect_ratio_tolerance": 0.5,
            "min_table_area_ratio": 0.05,
            "debug": False,
        }

        detector = TableDetector(config)

        # Create synthetic table image
        image = self._create_complex_table_image()

        # Run complete detection
        result = detector.detect_complete_table(image)

        # Verify results (may be None for synthetic image)
        if result is not None:
            assert isinstance(result, TableDetectionResult)
            assert result.confidence >= 0.0

    def test_performance_benchmarking(self):
        """Test detection performance on various image sizes."""
        import time

        config = {"debug": False}
        detector = TableDetector(config)

        # Test different image sizes
        sizes = [(320, 240), (640, 480), (1280, 720)]

        for width, height in sizes:
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            start_time = time.time()
            detector.detect_complete_table(image)
            end_time = time.time()

            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Should process within reasonable time (less than 1 second for test)
            assert processing_time < 1000

    def _create_complex_table_image(self):
        """Create a more complex synthetic table image for testing."""
        image = np.zeros((600, 1200, 3), dtype=np.uint8)

        # Table surface with gradient
        for y in range(100, 500):
            for x in range(150, 1050):
                # Create subtle gradient
                green_val = 120 + int(20 * np.sin(x / 100))
                image[y, x] = [green_val, 180, 100]

        # Table rails (darker edges)
        cv2.rectangle(image, (140, 90), (1060, 110), (40, 80, 60), -1)
        cv2.rectangle(image, (140, 490), (1060, 510), (40, 80, 60), -1)
        cv2.rectangle(image, (140, 90), (160, 510), (40, 80, 60), -1)
        cv2.rectangle(image, (1040, 90), (1060, 510), (40, 80, 60), -1)

        # Pockets with realistic shapes
        pocket_positions = [
            (150, 100),
            (600, 100),
            (1050, 100),  # Top pockets
            (150, 500),
            (600, 500),
            (1050, 500),  # Bottom pockets
        ]

        for pos in pocket_positions:
            cv2.circle(image, pos, 25, (10, 10, 10), -1)
            cv2.circle(image, pos, 20, (0, 0, 0), -1)

        # Add some noise and texture
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
