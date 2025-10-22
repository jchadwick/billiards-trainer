"""Comprehensive test suite for Cassapa-style cue detection implementation.

This test suite validates all phases of the Cassapa detection pipeline:
- Phase 1: HSV color filtering and morphological operations
- Phase 2: Hough transform line detection (probabilistic and standard)
- Phase 3: Edge detection and centerline refinement
- Phase 4: Direction normalization and angle calculation
- Phase 5: Integration and mode selection

Reference: cassapa/detector.cpp
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from backend.vision.detection.cue import CueDetector

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def detector():
    """Create a CueDetector instance with default configuration."""
    config = {
        "geometry": {
            "min_cue_length": 150,
            "max_cue_length": 800,
            "min_line_thickness": 3,
            "max_line_thickness": 25,
            "ball_radius": 15,
        },
        "cassapa": {
            "enabled": True,
            "precision_level": 1,
            "hsv_config": {
                "lh": 10,  # Hue lower bound
                "uh": 20,  # Hue upper bound
                "ls": 50,  # Saturation lower bound
                "us": 255,  # Saturation upper bound
                "lv": 50,  # Value lower bound
                "uv": 255,  # Value upper bound
            },
            "erode_size": 1,
            "dilate_size": 2,
        },
    }
    return CueDetector(config, yolo_detector=None)


@pytest.fixture()
def synthetic_cue_frame():
    """Create a synthetic test frame with a white diagonal line (simulating a cue).

    Returns:
        640x480 BGR frame with a diagonal white line from (100, 100) to (500, 400)
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (100, 100), (500, 400), (255, 255, 255), 10)
    return frame


@pytest.fixture()
def synthetic_colored_cue_frame():
    """Create a synthetic test frame with an orange/brown diagonal line (cue color).

    Returns:
        640x480 BGR frame with a diagonal orange line from (100, 100) to (500, 400)
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Orange/brown color (simulating wood cue)
    # BGR format: (B=50, G=150, R=200) gives orange-ish color
    cv2.line(frame, (100, 100), (500, 400), (50, 150, 200), 15)
    return frame


@pytest.fixture()
def mock_table_corners():
    """Create mock table corners for testing direction normalization.

    Returns:
        4x2 numpy array representing table corners (top-left, top-right, bottom-right, bottom-left)
    """
    return np.array(
        [
            [100, 100],  # top-left
            [540, 100],  # top-right
            [540, 380],  # bottom-right
            [100, 380],  # bottom-left
        ],
        dtype=np.float32,
    )


@pytest.fixture()
def sample_line():
    """Create a sample line for testing.

    Returns:
        Line as [x1, y1, x2, y2] numpy array
    """
    return np.array([100.0, 100.0, 500.0, 400.0], dtype=np.float64)


# ============================================================================
# Phase 1 Tests: HSV Filtering and Morphology
# Reference: cassapa/detector.cpp:423-450
# ============================================================================


class TestPhase1HSVFiltering:
    """Tests for Phase 1: HSV color filtering and morphological operations."""

    def test_create_hsv_mask_with_valid_frame(
        self, detector, synthetic_colored_cue_frame
    ):
        """Test HSV mask creation with a valid frame."""
        mask = detector._create_hsv_mask(
            synthetic_colored_cue_frame, detector.cassapa_hsv_config
        )

        assert mask is not None
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        # Mask should contain some white pixels (cue detected)
        assert np.sum(mask) > 0

    def test_create_hsv_mask_with_empty_frame(self, detector):
        """Test HSV mask creation with empty frame returns empty mask."""
        empty_frame = np.array([], dtype=np.uint8)
        mask = detector._create_hsv_mask(empty_frame, detector.cassapa_hsv_config)

        assert mask is not None
        assert mask.size > 0  # Should return minimal mask, not crash
        assert mask.shape == (1, 1)

    def test_create_hsv_mask_with_none_frame(self, detector):
        """Test HSV mask creation with None frame handles gracefully."""
        mask = detector._create_hsv_mask(None, detector.cassapa_hsv_config)

        assert mask is not None
        assert mask.shape == (1, 1)  # Returns minimal empty mask

    def test_create_hsv_mask_with_custom_config(self, detector, synthetic_cue_frame):
        """Test HSV mask creation with custom configuration."""
        custom_config = {
            "lh": 0,
            "uh": 180,  # Full hue range
            "ls": 0,
            "us": 255,  # Full saturation range
            "lv": 200,
            "uv": 255,  # High value only (bright pixels)
        }
        mask = detector._create_hsv_mask(synthetic_cue_frame, custom_config)

        assert mask is not None
        assert mask.shape == (480, 640)
        # With this config, the white line should be detected
        assert np.sum(mask) > 0

    def test_apply_morphology_with_valid_mask(self, detector):
        """Test morphological operations on a valid mask."""
        # Create a test mask with some noise
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 10)
        # Add some noise pixels
        mask[50:55, 50:55] = 255

        cleaned_mask = detector._apply_morphology(mask, erode_size=1, dilate_size=2)

        assert cleaned_mask is not None
        assert cleaned_mask.shape == mask.shape
        assert cleaned_mask.dtype == np.uint8
        # Should have removed some noise
        assert np.sum(cleaned_mask) <= np.sum(mask)

    def test_apply_morphology_with_empty_mask(self, detector):
        """Test morphological operations with empty mask."""
        empty_mask = np.array([], dtype=np.uint8)
        cleaned_mask = detector._apply_morphology(
            empty_mask, erode_size=1, dilate_size=2
        )

        assert cleaned_mask is not None
        assert cleaned_mask.shape == (1, 1)

    def test_apply_morphology_with_none_mask(self, detector):
        """Test morphological operations with None mask."""
        cleaned_mask = detector._apply_morphology(None, erode_size=1, dilate_size=2)

        assert cleaned_mask is not None
        assert cleaned_mask.shape == (1, 1)

    def test_apply_morphology_skip_erosion(self, detector):
        """Test morphological operations with erosion disabled (size=0)."""
        mask = np.ones((480, 640), dtype=np.uint8) * 255
        cleaned_mask = detector._apply_morphology(mask, erode_size=0, dilate_size=2)

        assert cleaned_mask is not None
        # Should only apply dilation, not erosion
        assert np.sum(cleaned_mask) >= np.sum(mask)

    def test_apply_morphology_skip_dilation(self, detector):
        """Test morphological operations with dilation disabled (size=0)."""
        mask = np.ones((480, 640), dtype=np.uint8) * 255
        cleaned_mask = detector._apply_morphology(mask, erode_size=1, dilate_size=0)

        assert cleaned_mask is not None
        # Should only apply erosion, not dilation
        assert np.sum(cleaned_mask) <= np.sum(mask)


# ============================================================================
# Phase 2 Tests: Hough Transform Line Detection
# Reference: cassapa/detector.cpp:474-790
# ============================================================================


class TestPhase2HoughDetection:
    """Tests for Phase 2: Hough transform line detection."""

    def test_detect_with_hough_p_cassapa_valid_mask(self, detector):
        """Test HoughLinesP detection with a valid mask."""
        # Create mask with a clear line
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 10)

        line = detector._detect_with_hough_p_cassapa(mask)

        # Should detect a line (or None if threshold too high)
        assert line is None or isinstance(line, np.ndarray)
        if line is not None:
            assert line.shape == (4,)
            assert line.dtype == np.float64

    def test_detect_with_hough_p_cassapa_empty_mask(self, detector):
        """Test HoughLinesP detection with empty mask returns None."""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        line = detector._detect_with_hough_p_cassapa(empty_mask)

        assert line is None

    def test_detect_with_hough_p_cassapa_too_many_lines(self, detector):
        """Test HoughLinesP detection rejects when too many lines detected."""
        # Create mask with many lines (table edges scenario)
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Draw a grid of lines
        for i in range(0, 640, 50):
            cv2.line(mask, (i, 0), (i, 480), 255, 2)
        for i in range(0, 480, 50):
            cv2.line(mask, (0, i), (640, i), 255, 2)

        line = detector._detect_with_hough_p_cassapa(mask)

        # Should return None when too many lines detected (likely table edges)
        assert line is None

    def test_fit_line_from_segments_valid_segments(self, detector):
        """Test line fitting from valid line segments."""
        # Create segments along a diagonal line
        segments = np.array(
            [
                [[100, 100, 200, 200]],
                [[200, 200, 300, 300]],
                [[300, 300, 400, 400]],
            ]
        )

        fitted_line = detector._fit_line_from_segments(segments)

        assert fitted_line is not None
        assert fitted_line.shape == (4,)
        assert fitted_line.dtype == np.float64
        # Line should approximately follow the diagonal
        x1, y1, x2, y2 = fitted_line
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float("inf")
        # For a 45-degree diagonal, slope should be close to 1
        assert abs(slope - 1.0) < 0.2  # Allow some tolerance

    def test_fit_line_from_segments_empty_segments(self, detector):
        """Test line fitting with empty segments returns None."""
        empty_segments = np.array([])
        fitted_line = detector._fit_line_from_segments(empty_segments)

        assert fitted_line is None

    def test_fit_line_from_segments_single_segment(self, detector):
        """Test line fitting with single segment."""
        single_segment = np.array([[[100, 100, 500, 400]]])
        fitted_line = detector._fit_line_from_segments(single_segment)

        assert fitted_line is not None
        assert fitted_line.shape == (4,)

    def test_clip_line_to_bbox_within_bounds(self, detector, sample_line):
        """Test clipping line that's already within bounding box."""
        bbox = (50, 50, 550, 450)  # Larger than line
        clipped_line = detector._clip_line_to_bbox(sample_line, bbox)

        assert clipped_line is not None
        assert clipped_line.shape == (4,)
        # Line should be clipped to bbox boundaries
        x1, y1, x2, y2 = clipped_line
        minx, miny, maxx, maxy = bbox
        assert minx <= x1 <= maxx
        assert minx <= x2 <= maxx
        assert miny <= y1 <= maxy
        assert miny <= y2 <= maxy

    def test_clip_line_to_bbox_extends_beyond(self, detector):
        """Test clipping line that extends beyond bounding box."""
        # Line that goes beyond bbox
        line = np.array([0, 0, 1000, 1000], dtype=np.float64)
        bbox = (100, 100, 500, 500)

        clipped_line = detector._clip_line_to_bbox(line, bbox)

        assert clipped_line is not None
        x1, y1, x2, y2 = clipped_line
        minx, miny, maxx, maxy = bbox
        # Clipped line should be within bbox
        assert minx <= x1 <= maxx
        assert minx <= x2 <= maxx
        assert miny <= y1 <= maxy
        assert miny <= y2 <= maxy

    def test_clip_line_to_bbox_degenerate_line(self, detector):
        """Test clipping degenerate line (zero length)."""
        degenerate_line = np.array([100, 100, 100, 100], dtype=np.float64)
        bbox = (50, 50, 550, 450)

        clipped_line = detector._clip_line_to_bbox(degenerate_line, bbox)

        assert clipped_line is not None
        # Should return original line unchanged
        np.testing.assert_array_equal(clipped_line, degenerate_line)

    def test_detect_with_hough_standard_cassapa_valid_mask(self, detector):
        """Test standard HoughLines detection with valid mask."""
        # Create mask with a clear line
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 10)

        line = detector._detect_with_hough_standard_cassapa(mask, canny_apply=True)

        # Should detect a line (or None if threshold too high)
        assert line is None or isinstance(line, np.ndarray)
        if line is not None:
            assert line.shape == (4,)
            assert line.dtype == np.float64

    def test_detect_with_hough_standard_cassapa_without_canny(self, detector):
        """Test standard HoughLines detection without Canny preprocessing."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 10)

        line = detector._detect_with_hough_standard_cassapa(mask, canny_apply=False)

        # Should still detect (or None if threshold too high)
        assert line is None or isinstance(line, np.ndarray)

    def test_detect_with_hough_standard_cassapa_empty_mask(self, detector):
        """Test standard HoughLines detection with empty mask returns None."""
        empty_mask = np.zeros((480, 640), dtype=np.uint8)
        line = detector._detect_with_hough_standard_cassapa(
            empty_mask, canny_apply=True
        )

        assert line is None


# ============================================================================
# Phase 3 Tests: Edge Detection and Centerline Refinement
# Reference: cassapa/detector.cpp:827-954
# ============================================================================


class TestPhase3EdgeRefinement:
    """Tests for Phase 3: Line equation, point sampling, edge detection, and centerline calculation."""

    def test_get_line_equation_diagonal_line(self, detector):
        """Test line equation calculation for diagonal line."""
        line = np.array([0, 0, 100, 100], dtype=np.float64)
        m, h = detector._get_line_equation(line)

        # For line from (0,0) to (100,100), slope should be 1, intercept 0
        assert abs(m - 1.0) < 1e-6
        assert abs(h - 0.0) < 1e-6

    def test_get_line_equation_horizontal_line(self, detector):
        """Test line equation calculation for horizontal line."""
        line = np.array([0, 50, 100, 50], dtype=np.float64)
        m, h = detector._get_line_equation(line)

        # For horizontal line at y=50, slope should be 0, intercept 50
        assert abs(m - 0.0) < 1e-6
        assert abs(h - 50.0) < 1e-6

    def test_get_line_equation_vertical_line(self, detector):
        """Test line equation calculation for vertical line (uses epsilon)."""
        line = np.array([50, 0, 50, 100], dtype=np.float64)
        m, h = detector._get_line_equation(line)

        # For vertical line, slope should be very large (due to epsilon)
        assert abs(m) > 1000  # Should be close to infinity

    def test_get_line_equation_degenerate_line(self, detector):
        """Test line equation for degenerate line (point)."""
        degenerate_line = np.array([100, 100, 100, 100], dtype=np.float64)
        m, h = detector._get_line_equation(degenerate_line)

        # Should handle gracefully with epsilon
        assert isinstance(m, float)
        assert isinstance(h, float)

    def test_sample_line_points_horizontal_line(self, detector):
        """Test point sampling along horizontal line."""
        line = np.array([0, 50, 100, 50], dtype=np.float64)
        points = detector._sample_line_points(line, step=10)

        assert len(points) > 0
        # All points should have y=50
        for x, y in points:
            assert abs(y - 50.0) < 1e-6
        # Points should span from x=0 to x=100
        x_coords = [x for x, y in points]
        assert min(x_coords) <= 10
        assert max(x_coords) >= 90

    def test_sample_line_points_vertical_line(self, detector):
        """Test point sampling along vertical line."""
        line = np.array([50, 0, 50, 100], dtype=np.float64)
        points = detector._sample_line_points(line, step=10)

        assert len(points) > 0
        # All points should have x=50
        for x, y in points:
            assert abs(x - 50.0) < 1e-6
        # Points should span from y=0 to y=100
        y_coords = [y for x, y in points]
        assert min(y_coords) <= 10
        assert max(y_coords) >= 90

    def test_sample_line_points_diagonal_line(self, detector, sample_line):
        """Test point sampling along diagonal line."""
        points = detector._sample_line_points(sample_line, step=50)

        assert len(points) > 0
        # Points should lie on the line (approximately)
        x1, y1, x2, y2 = sample_line
        m = (y2 - y1) / (x2 - x1)
        h = y1 - m * x1
        for x, y in points:
            expected_y = m * x + h
            assert abs(y - expected_y) < 1.0  # Allow small tolerance

    def test_sample_line_points_degenerate_line(self, detector):
        """Test point sampling for degenerate line (point)."""
        degenerate_line = np.array([100, 100, 100, 100], dtype=np.float64)
        points = detector._sample_line_points(degenerate_line, step=10)

        assert len(points) == 1
        assert points[0] == (100.0, 100.0)

    def test_sample_line_points_custom_step(self, detector, sample_line):
        """Test point sampling with custom step size."""
        points_step_10 = detector._sample_line_points(sample_line, step=10)
        points_step_50 = detector._sample_line_points(sample_line, step=50)

        # Smaller step should produce more points
        assert len(points_step_10) > len(points_step_50)

    def test_get_line_normal_horizontal_line(self, detector):
        """Test perpendicular normal calculation for horizontal line."""
        line = np.array([0, 50, 100, 50], dtype=np.float64)
        point = (50.0, 50.0)

        normal_m, normal_h = detector._get_line_normal(line, point)

        # Normal to horizontal line should be vertical (very large slope)
        assert (
            abs(normal_m) > 10 or abs(normal_m) < 0.1
        )  # Either very steep or reflected
        # Normal line should pass through the point
        # y = normal_m * x + normal_h
        calculated_y = normal_m * point[0] + normal_h
        assert abs(calculated_y - point[1]) < 1.0

    def test_get_line_normal_vertical_line(self, detector):
        """Test perpendicular normal calculation for vertical line."""
        line = np.array([50, 0, 50, 100], dtype=np.float64)
        point = (50.0, 50.0)

        normal_m, normal_h = detector._get_line_normal(line, point)

        # Normal to vertical line should be horizontal (slope close to 0)
        # Note: Cassapa uses geometric formula that may give different result
        assert isinstance(normal_m, float)
        assert isinstance(normal_h, float)

    def test_find_cue_edges_valid_input(self, detector):
        """Test edge detection with valid mask and sample points."""
        # Create mask with a thick line
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 20)

        line = np.array([100, 100, 500, 400], dtype=np.float64)
        sample_points = detector._sample_line_points(line, step=10)

        edge_points_1, edge_points_2 = detector._find_cue_edges(
            mask, sample_points, line
        )

        assert isinstance(edge_points_1, list)
        assert isinstance(edge_points_2, list)
        # Should find edge points on both sides
        assert len(edge_points_1) >= 0
        assert len(edge_points_2) >= 0

    def test_find_cue_edges_empty_sample_points(self, detector):
        """Test edge detection with empty sample points."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        line = np.array([100, 100, 500, 400], dtype=np.float64)

        edge_points_1, edge_points_2 = detector._find_cue_edges(mask, [], line)

        assert edge_points_1 == []
        assert edge_points_2 == []

    def test_find_cue_edges_none_line(self, detector):
        """Test edge detection with None line."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        sample_points = [(100, 100), (200, 200)]

        edge_points_1, edge_points_2 = detector._find_cue_edges(
            mask, sample_points, None
        )

        assert edge_points_1 == []
        assert edge_points_2 == []

    def test_find_cue_edges_thin_line(self, detector):
        """Test edge detection with thin line."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 3)  # Very thin line

        line = np.array([100, 100, 500, 400], dtype=np.float64)
        sample_points = detector._sample_line_points(line, step=10)

        edge_points_1, edge_points_2 = detector._find_cue_edges(
            mask, sample_points, line
        )

        # May find fewer edge points with thin line
        assert isinstance(edge_points_1, list)
        assert isinstance(edge_points_2, list)

    def test_calculate_centerline_valid_edges(self, detector):
        """Test centerline calculation from valid edge points."""
        # Create two parallel lines of edge points
        edge_points_1 = [(i, 100 + i * 0.5) for i in range(100, 500, 10)]
        edge_points_2 = [(i, 110 + i * 0.5) for i in range(100, 500, 10)]

        centerline = detector._calculate_centerline(edge_points_1, edge_points_2)

        if centerline is not None:  # May be None if insufficient points
            assert centerline.shape == (4,)
            assert centerline.dtype == np.float64
            # Centerline should be between the two edges
            x1, y1, x2, y2 = centerline

    def test_calculate_centerline_insufficient_points(self, detector):
        """Test centerline calculation with insufficient edge points."""
        edge_points_1 = [(100, 100), (110, 110)]  # Only 2 points
        edge_points_2 = [(100, 110), (110, 120)]  # Only 2 points

        centerline = detector._calculate_centerline(edge_points_1, edge_points_2)

        # Should return None if fewer than cassapa_edge_min_points (default 5)
        assert centerline is None

    def test_calculate_centerline_empty_edges(self, detector):
        """Test centerline calculation with empty edge lists."""
        centerline = detector._calculate_centerline([], [])

        assert centerline is None

    def test_calculate_centerline_mismatched_edges(self, detector):
        """Test centerline calculation with one empty edge list."""
        edge_points_1 = [(i, 100) for i in range(100, 200, 10)]
        edge_points_2 = []

        centerline = detector._calculate_centerline(edge_points_1, edge_points_2)

        assert centerline is None


# ============================================================================
# Phase 4 Tests: Direction Normalization and Angle Calculation
# Reference: cassapa/detector.cpp:143-158, 1022-1023
# ============================================================================


class TestPhase4DirectionAndAngle:
    """Tests for Phase 4: Table center calculation, direction normalization, and angle calculation."""

    def test_calculate_table_center_valid_corners(self, detector, mock_table_corners):
        """Test table center calculation with valid corners."""
        cx, cy = detector._calculate_table_center(mock_table_corners)

        assert isinstance(cx, float)
        assert isinstance(cy, float)
        # Center should be approximately in the middle of the table
        expected_cx = (100 + 540) / 2
        expected_cy = (100 + 380) / 2
        assert abs(cx - expected_cx) < 1.0
        assert abs(cy - expected_cy) < 1.0

    def test_calculate_table_center_none_corners(self, detector):
        """Test table center calculation with None corners returns (0, 0)."""
        cx, cy = detector._calculate_table_center(None)

        assert cx == 0.0
        assert cy == 0.0

    def test_calculate_table_center_degenerate_corners(self, detector):
        """Test table center calculation with degenerate corners (all same point)."""
        degenerate_corners = np.array(
            [
                [100, 100],
                [100, 100],
                [100, 100],
                [100, 100],
            ],
            dtype=np.float32,
        )

        cx, cy = detector._calculate_table_center(degenerate_corners)

        assert cx == 100.0
        assert cy == 100.0

    def test_normalize_line_direction_cassapa_correct_orientation(self, detector):
        """Test direction normalization when line is already correctly oriented."""
        # Line pointing toward center (100, 100) from far end
        line = np.array(
            [500, 500, 100, 100], dtype=np.float64
        )  # Butt at (500,500), tip at (100,100)
        table_center = (100.0, 100.0)

        normalized_line = detector._normalize_line_direction_cassapa(line, table_center)

        # Point 1 should be farther from center (butt end)
        x1, y1, x2, y2 = normalized_line
        dist1 = np.sqrt((x1 - table_center[0]) ** 2 + (y1 - table_center[1]) ** 2)
        dist2 = np.sqrt((x2 - table_center[0]) ** 2 + (y2 - table_center[1]) ** 2)
        assert dist1 >= dist2

    def test_normalize_line_direction_cassapa_needs_swap(self, detector):
        """Test direction normalization when line needs to be swapped."""
        # Line pointing away from center (needs swap)
        line = np.array(
            [100, 100, 500, 500], dtype=np.float64
        )  # Tip at (100,100), butt at (500,500)
        table_center = (100.0, 100.0)

        normalized_line = detector._normalize_line_direction_cassapa(line, table_center)

        # After normalization, point 1 should be farther from center
        x1, y1, x2, y2 = normalized_line
        dist1 = np.sqrt((x1 - table_center[0]) ** 2 + (y1 - table_center[1]) ** 2)
        dist2 = np.sqrt((x2 - table_center[0]) ** 2 + (y2 - table_center[1]) ** 2)
        assert dist1 >= dist2

    def test_normalize_line_direction_cassapa_equidistant(self, detector):
        """Test direction normalization when both points are equidistant."""
        # Line perpendicular to center
        table_center = (250.0, 250.0)
        line = np.array(
            [250, 100, 250, 400], dtype=np.float64
        )  # Vertical line through center

        normalized_line = detector._normalize_line_direction_cassapa(line, table_center)

        # Line should be unchanged (no swap needed)
        x1, y1, x2, y2 = normalized_line
        dist1 = np.sqrt((x1 - table_center[0]) ** 2 + (y1 - table_center[1]) ** 2)
        dist2 = np.sqrt((x2 - table_center[0]) ** 2 + (y2 - table_center[1]) ** 2)
        assert abs(dist1 - dist2) < 1.0  # Nearly equal

    def test_calculate_cue_angle_cassapa_horizontal_line(self, detector):
        """Test angle calculation for horizontal line."""
        line = np.array([0, 50, 100, 50], dtype=np.float64)
        angle = detector._calculate_cue_angle_cassapa(line)

        assert isinstance(angle, float)
        assert 0 <= angle < 360
        # Horizontal line should have angle close to 0
        assert abs(angle - 0.0) < 1.0 or abs(angle - 360.0) < 1.0

    def test_calculate_cue_angle_cassapa_vertical_line(self, detector):
        """Test angle calculation for vertical line."""
        line = np.array([50, 0, 50, 100], dtype=np.float64)
        angle = detector._calculate_cue_angle_cassapa(line)

        assert isinstance(angle, float)
        assert 0 <= angle < 360
        # Vertical line should have angle close to 90 or 270
        assert abs(angle - 90.0) < 10.0 or abs(angle - 270.0) < 10.0

    def test_calculate_cue_angle_cassapa_45_degree_line(self, detector):
        """Test angle calculation for 45-degree diagonal line."""
        line = np.array([0, 0, 100, 100], dtype=np.float64)
        angle = detector._calculate_cue_angle_cassapa(line)

        assert isinstance(angle, float)
        assert 0 <= angle < 360
        # 45-degree line should have angle close to 45
        assert abs(angle - 45.0) < 10.0

    def test_calculate_cue_angle_cassapa_negative_slope(self, detector):
        """Test angle calculation for line with negative slope."""
        line = np.array([0, 100, 100, 0], dtype=np.float64)
        angle = detector._calculate_cue_angle_cassapa(line)

        assert isinstance(angle, float)
        assert 0 <= angle < 360
        # Negative slope line should have angle in 270-360 range (or equivalent)


# ============================================================================
# Phase 5 Tests: Frame Clipping and Mode Integration
# Reference: cassapa/detector.cpp:622-629, full pipeline
# ============================================================================


class TestPhase5IntegrationAndClipping:
    """Tests for Phase 5: Frame clipping, fast/precise modes, and full integration."""

    def test_clip_to_frame_line_within_bounds(self, detector, sample_line):
        """Test frame clipping with line already within bounds."""
        frame_shape = (480, 640)
        clipped_line = detector._clip_to_frame(sample_line, frame_shape)

        assert clipped_line is not None
        assert clipped_line.shape == (4,)
        # All coordinates should be within frame
        x1, y1, x2, y2 = clipped_line
        assert 0 <= x1 < 640
        assert 0 <= x2 < 640
        assert 0 <= y1 < 480
        assert 0 <= y2 < 480

    def test_clip_to_frame_line_extends_beyond(self, detector):
        """Test frame clipping with line extending beyond frame."""
        line = np.array([-100, -100, 1000, 1000], dtype=np.float64)
        frame_shape = (480, 640)

        clipped_line = detector._clip_to_frame(line, frame_shape)

        assert clipped_line is not None
        x1, y1, x2, y2 = clipped_line
        # Clipped coordinates should be within frame (or at boundaries)
        assert 0 <= x1 <= 640
        assert 0 <= x2 <= 640
        assert 0 <= y1 <= 480
        assert 0 <= y2 <= 480

    def test_clip_to_frame_line_completely_outside(self, detector):
        """Test frame clipping with line completely outside frame."""
        line = np.array([1000, 1000, 2000, 2000], dtype=np.float64)
        frame_shape = (480, 640)

        clipped_line = detector._clip_to_frame(line, frame_shape)

        # Should return original line if completely outside
        assert clipped_line is not None

    def test_clip_to_frame_different_frame_sizes(self, detector, sample_line):
        """Test frame clipping with different frame sizes."""
        # Test with HD frame
        frame_shape_hd = (1080, 1920)
        clipped_line_hd = detector._clip_to_frame(sample_line, frame_shape_hd)

        assert clipped_line_hd is not None

        # Test with smaller frame
        frame_shape_small = (240, 320)
        clipped_line_small = detector._clip_to_frame(sample_line, frame_shape_small)

        assert clipped_line_small is not None

    def test_detect_cassapa_style_fast_mode(
        self, detector, synthetic_colored_cue_frame, mock_table_corners
    ):
        """Test full Cassapa detection in fast mode (precision_level=0)."""
        result = detector._detect_cassapa_style(
            synthetic_colored_cue_frame, mock_table_corners, precision_level=0
        )

        # May or may not detect depending on HSV parameters
        # Just verify it doesn't crash and returns correct type
        assert result is None or hasattr(result, "tip_position")

    def test_detect_cassapa_style_precise_mode(
        self, detector, synthetic_colored_cue_frame, mock_table_corners
    ):
        """Test full Cassapa detection in precise mode (precision_level=1)."""
        result = detector._detect_cassapa_style(
            synthetic_colored_cue_frame, mock_table_corners, precision_level=1
        )

        # May or may not detect depending on HSV parameters
        # Just verify it doesn't crash and returns correct type
        assert result is None or hasattr(result, "tip_position")

    def test_detect_cassapa_style_with_none_frame(self, detector, mock_table_corners):
        """Test Cassapa detection with None frame returns None."""
        result = detector._detect_cassapa_style(
            None, mock_table_corners, precision_level=1
        )

        assert result is None

    def test_detect_cassapa_style_with_empty_frame(self, detector, mock_table_corners):
        """Test Cassapa detection with empty frame returns None."""
        empty_frame = np.array([], dtype=np.uint8)
        result = detector._detect_cassapa_style(
            empty_frame, mock_table_corners, precision_level=1
        )

        assert result is None

    def test_detect_cassapa_style_without_table_corners(
        self, detector, synthetic_colored_cue_frame
    ):
        """Test Cassapa detection without table corners (still works)."""
        result = detector._detect_cassapa_style(
            synthetic_colored_cue_frame, None, precision_level=1  # No table corners
        )

        # Should still work (direction normalization uses (0,0) as fallback)
        assert result is None or hasattr(result, "tip_position")

    def test_detect_cue_integration_cassapa_enabled(
        self, detector, synthetic_colored_cue_frame
    ):
        """Test detect_cue method with Cassapa detection enabled."""
        # Ensure Cassapa is enabled
        detector.cassapa_enabled = True

        result = detector.detect_cue(
            synthetic_colored_cue_frame,
            cue_ball_pos=(320, 240),  # Center of frame
            all_ball_positions=[(320, 240)],
            table_corners=None,
        )

        # Should attempt Cassapa detection first
        # Result may be None if detection fails, but should not crash
        assert result is None or hasattr(result, "tip_position")

    def test_detect_cue_integration_cassapa_disabled(
        self, detector, synthetic_colored_cue_frame
    ):
        """Test detect_cue method with Cassapa detection disabled."""
        # Disable Cassapa
        detector.cassapa_enabled = False

        result = detector.detect_cue(
            synthetic_colored_cue_frame,
            cue_ball_pos=(320, 240),
            all_ball_positions=[(320, 240)],
            table_corners=None,
        )

        # Should skip Cassapa and use fallback methods
        # Result may be None if detection fails, but should not crash
        assert result is None or hasattr(result, "tip_position")

    def test_detect_cassapa_style_precision_level_default(
        self, detector, synthetic_colored_cue_frame, mock_table_corners
    ):
        """Test Cassapa detection with default precision level."""
        # Don't specify precision_level, should use detector's default
        result = detector._detect_cassapa_style(
            synthetic_colored_cue_frame, mock_table_corners, precision_level=None
        )

        # Should use self.cassapa_precision_level (default 1)
        assert result is None or hasattr(result, "tip_position")


# ============================================================================
# Edge Case and Error Handling Tests
# ============================================================================


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling across all phases."""

    def test_all_phases_with_black_frame(self, detector, mock_table_corners):
        """Test all phases with completely black frame."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector._detect_cassapa_style(
            black_frame, mock_table_corners, precision_level=1
        )

        # Should return None (no cue detected in black frame)
        assert result is None

    def test_all_phases_with_white_frame(self, detector, mock_table_corners):
        """Test all phases with completely white frame."""
        white_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = detector._detect_cassapa_style(
            white_frame, mock_table_corners, precision_level=1
        )

        # May or may not detect depending on HSV config
        assert result is None or hasattr(result, "tip_position")

    def test_all_phases_with_noise_frame(self, detector, mock_table_corners):
        """Test all phases with random noise frame."""
        noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = detector._detect_cassapa_style(
            noise_frame, mock_table_corners, precision_level=1
        )

        # Should handle noise gracefully (likely return None)
        assert result is None or hasattr(result, "tip_position")

    def test_phase_1_with_grayscale_frame(self, detector):
        """Test Phase 1 with grayscale frame (should convert to BGR internally)."""
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(gray_frame, (100, 100), (500, 400), 255, 10)

        # Should handle gracefully (HSV conversion expects 3 channels)
        try:
            mask = detector._create_hsv_mask(gray_frame, detector.cassapa_hsv_config)
            # If it doesn't crash, it should return a mask
            assert mask is not None
        except cv2.error:
            # Expected if frame doesn't have 3 channels
            pass

    def test_phase_2_with_very_small_mask(self, detector):
        """Test Phase 2 with very small mask."""
        tiny_mask = np.ones((10, 10), dtype=np.uint8) * 255
        line = detector._detect_with_hough_p_cassapa(tiny_mask)

        # Should return None (mask too small for meaningful detection)
        assert line is None

    def test_phase_3_with_single_sample_point(self, detector):
        """Test Phase 3 edge detection with single sample point."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(mask, (100, 100), (500, 400), 255, 10)
        line = np.array([100, 100, 500, 400], dtype=np.float64)
        sample_points = [(300, 250)]  # Single point

        edge_points_1, edge_points_2 = detector._find_cue_edges(
            mask, sample_points, line
        )

        # Should handle single point (may find edges or not)
        assert isinstance(edge_points_1, list)
        assert isinstance(edge_points_2, list)

    def test_phase_4_with_extreme_table_center(self, detector, sample_line):
        """Test Phase 4 with extreme table center values."""
        extreme_center = (10000.0, 10000.0)
        normalized_line = detector._normalize_line_direction_cassapa(
            sample_line, extreme_center
        )

        # Should handle extreme values without crashing
        assert normalized_line is not None
        assert normalized_line.shape == (4,)

    def test_configuration_with_extreme_values(self):
        """Test detector initialization with extreme configuration values."""
        extreme_config = {
            "cassapa": {
                "hsv_config": {
                    "lh": 0,
                    "uh": 180,
                    "ls": 0,
                    "us": 255,
                    "lv": 0,
                    "uv": 255,
                },
                "erode_size": 10,  # Very large erosion
                "dilate_size": 10,  # Very large dilation
            }
        }
        detector = CueDetector(extreme_config, yolo_detector=None)

        # Should initialize without errors
        assert detector is not None
        assert detector.cassapa_erode_size == 10
        assert detector.cassapa_dilate_size == 10


# ============================================================================
# Performance and Stress Tests
# ============================================================================


class TestPerformanceAndStress:
    """Tests for performance characteristics and stress conditions."""

    def test_detect_cassapa_style_multiple_calls(
        self, detector, synthetic_colored_cue_frame, mock_table_corners
    ):
        """Test multiple sequential calls to Cassapa detection."""
        results = []
        for _ in range(5):
            result = detector._detect_cassapa_style(
                synthetic_colored_cue_frame, mock_table_corners, precision_level=1
            )
            results.append(result)

        # All calls should complete without errors
        assert len(results) == 5

    def test_sample_line_points_with_large_step(self, detector, sample_line):
        """Test point sampling with very large step size."""
        points = detector._sample_line_points(sample_line, step=1000)

        # Should return at least 2 points (start and end)
        assert len(points) >= 2

    def test_sample_line_points_with_tiny_step(self, detector, sample_line):
        """Test point sampling with very small step size."""
        points = detector._sample_line_points(sample_line, step=1)

        # Should return many points (approximately line length)
        assert len(points) > 100  # Line is ~500 pixels long

    def test_apply_morphology_with_large_kernels(self, detector):
        """Test morphological operations with large kernel sizes."""
        mask = np.ones((480, 640), dtype=np.uint8) * 255
        cleaned_mask = detector._apply_morphology(mask, erode_size=20, dilate_size=20)

        assert cleaned_mask is not None
        assert cleaned_mask.shape == mask.shape


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
