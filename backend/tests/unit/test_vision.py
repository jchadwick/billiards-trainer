"""Unit tests for the vision module."""

import time

import cv2
import numpy as np
import pytest
from vision.calibration.camera import CameraCalibrator
from vision.calibration.color import ColorCalibrator
from vision.detection.balls import BallDetector
from vision.detection.cue import CueDetector
from vision.detection.table import TableDetector
from vision.models import CameraFrame
from vision.preprocessing import FramePreprocessor
from vision.tracking.kalman import KalmanFilter
from vision.tracking.tracker import BallTracker
from vision.utils.visualization import draw_detection_overlay


@pytest.mark.unit()
class TestCameraFrame:
    """Test the CameraFrame model."""

    def test_frame_creation(self):
        """Test creating a camera frame."""
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
        timestamp = time.time()

        frame = CameraFrame(
            frame=frame_data, timestamp=timestamp, frame_id=1, width=640, height=480
        )

        assert frame.width == 640
        assert frame.height == 480
        assert frame.frame_id == 1
        assert frame.timestamp == timestamp
        assert frame.frame.shape == (480, 640, 3)

    def test_frame_properties(self):
        """Test frame property calculations."""
        frame_data = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = CameraFrame(
            frame=frame_data, timestamp=time.time(), frame_id=1, width=640, height=480
        )

        assert frame.aspect_ratio == 640 / 480
        assert frame.center == (320, 240)
        assert frame.size == (640, 480)

    def test_frame_copy(self):
        """Test frame copying."""
        frame_data = np.ones((480, 640, 3), dtype=np.uint8) * 128

        frame = CameraFrame(
            frame=frame_data, timestamp=time.time(), frame_id=1, width=640, height=480
        )

        frame_copy = frame.copy()
        assert np.array_equal(frame.frame, frame_copy.frame)
        assert frame.timestamp == frame_copy.timestamp
        assert frame.frame_id == frame_copy.frame_id

        # Modify original - copy should be unchanged
        frame.frame[0, 0] = [255, 0, 0]
        assert not np.array_equal(frame.frame, frame_copy.frame)


@pytest.mark.unit()
class TestDetectionResult:
    """Test the DetectionResult model."""

    def test_detection_result_creation(self, mock_detection_result):
        """Test creating a detection result."""
        assert len(mock_detection_result.balls) == 3
        assert len(mock_detection_result.table_corners) == 4
        assert mock_detection_result.confidence == 0.95

    def test_get_ball_by_id(self, mock_detection_result):
        """Test getting ball by ID from detection result."""
        cue_ball = mock_detection_result.get_ball("cue")
        assert cue_ball is not None
        assert cue_ball.id == "cue"

        nonexistent = mock_detection_result.get_ball("99")
        assert nonexistent is None

    def test_filter_balls_by_color(self, mock_detection_result):
        """Test filtering balls by color."""
        yellow_balls = mock_detection_result.filter_balls_by_color("yellow")
        assert len(yellow_balls) == 1
        assert yellow_balls[0].id == "1"

    def test_balls_count(self, mock_detection_result):
        """Test counting detected balls."""
        assert mock_detection_result.ball_count == 3

    def test_detection_quality(self, mock_detection_result):
        """Test detection quality assessment."""
        quality = mock_detection_result.quality_score
        assert 0 <= quality <= 1


@pytest.mark.unit()
class TestBallDetector:
    """Test the ball detector."""

    def test_detector_creation(self):
        """Test creating ball detector."""
        detector = BallDetector()
        assert detector is not None

    @pytest.mark.opencv_available()
    def test_detect_balls_empty_frame(self):
        """Test ball detection on empty frame."""
        detector = BallDetector()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        balls = detector.detect(empty_frame)
        assert isinstance(balls, list)
        # Should not detect any balls in empty frame
        assert len(balls) == 0

    @pytest.mark.opencv_available()
    def test_detect_balls_with_circles(self):
        """Test ball detection with synthetic circles."""
        detector = BallDetector()

        # Create frame with white circles (balls)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Green background

        # Add white circles
        cv2.circle(frame, (320, 240), 20, (255, 255, 255), -1)
        cv2.circle(frame, (200, 150), 20, (255, 255, 0), -1)

        balls = detector.detect(frame)
        assert len(balls) >= 1  # Should detect at least one ball

    def test_ball_color_classification(self):
        """Test ball color classification."""
        detector = BallDetector()

        # Test color classification
        white_color = np.array([255, 255, 255])
        color_name = detector.classify_color(white_color)
        assert color_name in ["white", "cue"]

        yellow_color = np.array([255, 255, 0])
        color_name = detector.classify_color(yellow_color)
        assert color_name in ["yellow", "1"]

    def test_filter_by_size(self):
        """Test filtering detected circles by size."""
        detector = BallDetector()

        # Mock circles with different radii
        circles = np.array(
            [
                [100, 100, 10],  # Too small
                [200, 200, 25],  # Good size
                [300, 300, 50],  # Too large
            ]
        )

        filtered = detector.filter_by_size(circles, min_radius=20, max_radius=30)
        assert len(filtered) == 1
        assert filtered[0][2] == 25

    def test_remove_overlapping_detections(self):
        """Test removing overlapping ball detections."""
        detector = BallDetector()

        # Mock overlapping balls
        balls = [
            {"x": 100, "y": 100, "radius": 20, "confidence": 0.9},
            {"x": 105, "y": 105, "radius": 20, "confidence": 0.8},  # Overlapping
            {"x": 200, "y": 200, "radius": 20, "confidence": 0.95},
        ]

        filtered = detector.remove_overlaps(balls)
        assert (
            len(filtered) == 2
        )  # Should remove the overlapping one with lower confidence


@pytest.mark.unit()
class TestTableDetector:
    """Test the table detector."""

    def test_detector_creation(self):
        """Test creating table detector."""
        detector = TableDetector()
        assert detector is not None

    @pytest.mark.opencv_available()
    def test_detect_table_edges(self):
        """Test table edge detection."""
        detector = TableDetector()

        # Create frame with rectangular table
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:380, 50:590] = [34, 139, 34]  # Green rectangle

        corners = detector.detect_corners(frame)
        assert isinstance(corners, list)

        if corners:
            assert len(corners) == 4
            # Corners should be roughly at the rectangle boundaries

    def test_table_perspective_correction(self):
        """Test table perspective correction."""
        detector = TableDetector()

        # Mock corners of a perspective-distorted table
        corners = [(100, 120), (500, 100), (520, 360), (80, 380)]

        corrected_frame = detector.correct_perspective(
            np.zeros((480, 640, 3), dtype=np.uint8), corners, output_size=(800, 400)
        )

        assert corrected_frame.shape == (400, 800, 3)

    def test_validate_table_shape(self):
        """Test table shape validation."""
        detector = TableDetector()

        # Valid rectangular corners
        valid_corners = [(0, 0), (100, 0), (100, 50), (0, 50)]
        assert detector.validate_corners(valid_corners)

        # Invalid corners (not rectangular)
        invalid_corners = [(0, 0), (100, 0), (90, 60), (10, 50)]
        assert not detector.validate_corners(invalid_corners)

    @pytest.mark.opencv_available()
    def test_detect_table_color(self):
        """Test table felt color detection."""
        detector = TableDetector()

        # Create frame with green table
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Green felt

        dominant_color = detector.detect_felt_color(frame)
        assert isinstance(dominant_color, (list, tuple, np.ndarray))
        assert len(dominant_color) == 3


@pytest.mark.unit()
class TestCueDetector:
    """Test the cue detector."""

    def test_detector_creation(self):
        """Test creating cue detector."""
        detector = CueDetector()
        assert detector is not None

    @pytest.mark.opencv_available()
    def test_detect_cue_stick_empty_frame(self):
        """Test cue detection on empty frame."""
        detector = CueDetector()
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cue_data = detector.detect(empty_frame)
        # Should not detect cue in empty frame
        assert cue_data is None or cue_data["confidence"] < 0.5

    @pytest.mark.opencv_available()
    def test_detect_cue_stick_with_line(self):
        """Test cue detection with synthetic line."""
        detector = CueDetector()

        # Create frame with a line (representing cue)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (100, 240), (500, 240), (139, 69, 19), 5)  # Brown line

        cue_data = detector.detect(frame)
        if cue_data:
            assert "position" in cue_data
            assert "angle" in cue_data
            assert "confidence" in cue_data

    def test_line_angle_calculation(self):
        """Test cue stick angle calculation."""
        detector = CueDetector()

        # Horizontal line
        angle = detector.calculate_line_angle((0, 0), (100, 0))
        assert abs(angle - 0) < 1e-6

        # Vertical line
        angle = detector.calculate_line_angle((0, 0), (0, 100))
        assert abs(angle - 90) < 1e-6

    def test_filter_cue_lines(self):
        """Test filtering detected lines for cue stick."""
        detector = CueDetector()

        # Mock detected lines
        lines = [
            [(0, 0, 100, 0)],  # Horizontal line
            [(0, 0, 0, 100)],  # Vertical line
            [(0, 0, 50, 50)],  # Diagonal line
            [(10, 10, 15, 12)],  # Very short line
        ]

        filtered = detector.filter_lines(lines, min_length=50)
        assert len(filtered) <= len(lines)


@pytest.mark.unit()
class TestBallTracker:
    """Test the ball tracker."""

    def test_tracker_creation(self):
        """Test creating ball tracker."""
        tracker = BallTracker()
        assert tracker is not None

    def test_add_detection(self):
        """Test adding detection to tracker."""
        tracker = BallTracker()

        detection = {
            "id": "cue",
            "x": 100,
            "y": 200,
            "confidence": 0.9,
            "timestamp": time.time(),
        }

        tracker.add_detection(detection)
        assert len(tracker.tracks) == 1

    def test_update_existing_track(self):
        """Test updating existing track."""
        tracker = BallTracker()

        # Add initial detection
        detection1 = {
            "id": "cue",
            "x": 100,
            "y": 200,
            "confidence": 0.9,
            "timestamp": time.time(),
        }
        tracker.add_detection(detection1)

        # Update with new position
        detection2 = {
            "id": "cue",
            "x": 110,
            "y": 210,
            "confidence": 0.85,
            "timestamp": time.time() + 0.033,  # 30ms later
        }
        tracker.update(detection2)

        track = tracker.get_track("cue")
        assert track is not None
        assert track["x"] == 110
        assert track["y"] == 210

    def test_track_velocity_calculation(self):
        """Test velocity calculation for tracked balls."""
        tracker = BallTracker()

        timestamp1 = time.time()
        timestamp2 = timestamp1 + 0.1  # 100ms later

        # Add initial detection
        detection1 = {
            "id": "cue",
            "x": 100,
            "y": 200,
            "confidence": 0.9,
            "timestamp": timestamp1,
        }
        tracker.add_detection(detection1)

        # Update with new position
        detection2 = {
            "id": "cue",
            "x": 110,
            "y": 220,
            "confidence": 0.85,
            "timestamp": timestamp2,
        }
        tracker.update(detection2)

        track = tracker.get_track("cue")
        if "velocity" in track:
            vx, vy = track["velocity"]
            # Should have moved 10 pixels in x and 20 in y over 0.1 seconds
            assert abs(vx - 100) < 10  # 100 px/s
            assert abs(vy - 200) < 10  # 200 px/s

    def test_remove_stale_tracks(self):
        """Test removing stale tracks."""
        tracker = BallTracker(max_age=1.0)  # 1 second timeout

        old_time = time.time() - 2.0  # 2 seconds ago
        detection = {
            "id": "cue",
            "x": 100,
            "y": 200,
            "confidence": 0.9,
            "timestamp": old_time,
        }
        tracker.add_detection(detection)

        # Update tracker (should remove stale track)
        tracker.cleanup_stale_tracks()
        assert len(tracker.tracks) == 0


@pytest.mark.unit()
class TestKalmanFilter:
    """Test the Kalman filter."""

    def test_filter_creation(self):
        """Test creating Kalman filter."""
        kf = KalmanFilter()
        assert kf is not None

    def test_filter_prediction(self):
        """Test Kalman filter prediction."""
        kf = KalmanFilter()

        # Initialize with position and velocity
        kf.init_state([100, 200, 10, 5])  # x, y, vx, vy

        # Predict next state
        predicted = kf.predict()
        assert len(predicted) == 4

        # Position should have changed based on velocity
        assert predicted[0] != 100  # x changed
        assert predicted[1] != 200  # y changed

    def test_filter_update(self):
        """Test Kalman filter update with measurement."""
        kf = KalmanFilter()

        # Initialize filter
        kf.init_state([100, 200, 0, 0])

        # Update with measurement
        measurement = [105, 205]  # Slight change in position
        updated = kf.update(measurement)

        assert len(updated) == 4
        # Position should be influenced by measurement
        assert 100 < updated[0] < 110
        assert 200 < updated[1] < 210

    def test_filter_uncertainty(self):
        """Test Kalman filter uncertainty tracking."""
        kf = KalmanFilter()

        # Initialize with high uncertainty
        kf.init_state([100, 200, 0, 0])
        initial_uncertainty = kf.get_uncertainty()

        # Add measurements to reduce uncertainty
        for i in range(10):
            kf.predict()
            kf.update([100 + i, 200 + i])

        final_uncertainty = kf.get_uncertainty()
        # Uncertainty should have decreased
        assert final_uncertainty < initial_uncertainty


@pytest.mark.unit()
class TestCameraCalibrator:
    """Test the camera calibrator."""

    def test_calibrator_creation(self):
        """Test creating camera calibrator."""
        calibrator = CameraCalibrator()
        assert calibrator is not None

    def test_generate_chessboard_points(self):
        """Test generating chessboard calibration points."""
        calibrator = CameraCalibrator()

        object_points = calibrator.generate_chessboard_points(
            board_size=(9, 6), square_size=25.0
        )

        assert object_points.shape == (54, 3)  # 9*6 = 54 points
        assert object_points[0, 2] == 0  # Z coordinate should be 0

    @pytest.mark.opencv_available()
    def test_detect_chessboard_corners(self):
        """Test detecting chessboard corners."""
        calibrator = CameraCalibrator()

        # Create synthetic chessboard image
        frame = np.zeros((480, 640), dtype=np.uint8)

        # Simple alternating pattern
        for i in range(0, 480, 60):
            for j in range(0, 640, 80):
                if (i // 60 + j // 80) % 2 == 0:
                    frame[i : i + 60, j : j + 80] = 255

        found, corners = calibrator.detect_corners(frame, (7, 5))
        # May or may not find corners in synthetic image
        assert isinstance(found, bool)
        if found:
            assert corners.shape[1] == 2  # 2D corners

    def test_calibration_data_storage(self):
        """Test storing calibration data."""
        calibrator = CameraCalibrator()

        # Mock calibration data
        object_points = [np.random.rand(54, 3)]
        image_points = [np.random.rand(54, 1, 2)]

        calibrator.add_calibration_data(object_points[0], image_points[0])
        assert len(calibrator.object_points) == 1
        assert len(calibrator.image_points) == 1


@pytest.mark.unit()
class TestColorCalibrator:
    """Test the color calibrator."""

    def test_calibrator_creation(self):
        """Test creating color calibrator."""
        calibrator = ColorCalibrator()
        assert calibrator is not None

    def test_sample_color_region(self):
        """Test sampling color from image region."""
        calibrator = ColorCalibrator()

        # Create frame with known color
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:280, 300:380] = [255, 255, 0]  # Yellow region

        # Sample from yellow region
        sampled_color = calibrator.sample_region(frame, center=(340, 240), radius=30)

        # Should be close to yellow
        assert sampled_color[0] > 200  # High red
        assert sampled_color[1] > 200  # High green
        assert sampled_color[2] < 50  # Low blue

    def test_color_threshold_calculation(self):
        """Test calculating color thresholds."""
        calibrator = ColorCalibrator()

        # Add color samples
        samples = [
            [250, 250, 10],  # Yellow-ish
            [255, 255, 0],  # Pure yellow
            [240, 240, 20],  # Yellow-ish
        ]

        for sample in samples:
            calibrator.add_sample("yellow", sample)

        thresholds = calibrator.calculate_thresholds("yellow")
        assert "lower" in thresholds
        assert "upper" in thresholds
        assert len(thresholds["lower"]) == 3
        assert len(thresholds["upper"]) == 3

    def test_color_distance_calculation(self):
        """Test color distance calculation."""
        calibrator = ColorCalibrator()

        # Same color
        distance = calibrator.color_distance([255, 0, 0], [255, 0, 0])
        assert distance == 0

        # Different colors
        distance = calibrator.color_distance([255, 0, 0], [0, 255, 0])
        assert distance > 0


@pytest.mark.unit()
class TestFramePreprocessor:
    """Test the frame preprocessor."""

    def test_preprocessor_creation(self):
        """Test creating frame preprocessor."""
        preprocessor = FramePreprocessor()
        assert preprocessor is not None

    @pytest.mark.opencv_available()
    def test_noise_reduction(self):
        """Test noise reduction."""
        preprocessor = FramePreprocessor()

        # Create noisy frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        denoised = preprocessor.reduce_noise(frame)
        assert denoised.shape == frame.shape
        assert denoised.dtype == frame.dtype

    @pytest.mark.opencv_available()
    def test_contrast_enhancement(self):
        """Test contrast enhancement."""
        preprocessor = FramePreprocessor()

        # Create low contrast frame
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        enhanced = preprocessor.enhance_contrast(frame)
        assert enhanced.shape == frame.shape

    @pytest.mark.opencv_available()
    def test_color_space_conversion(self):
        """Test color space conversion."""
        preprocessor = FramePreprocessor()

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Convert to HSV
        hsv_frame = preprocessor.convert_color_space(frame, "HSV")
        assert hsv_frame.shape == frame.shape

        # Convert to grayscale
        gray_frame = preprocessor.convert_color_space(frame, "GRAY")
        assert gray_frame.shape == (480, 640)

    @pytest.mark.opencv_available()
    def test_lighting_correction(self):
        """Test lighting correction."""
        preprocessor = FramePreprocessor()

        # Create frame with uneven lighting
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            frame[i, :] = int(255 * i / 480)  # Gradient

        corrected = preprocessor.correct_lighting(frame)
        assert corrected.shape == frame.shape

    def test_region_of_interest(self):
        """Test region of interest extraction."""
        preprocessor = FramePreprocessor()

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Extract center region
        roi = preprocessor.extract_roi(frame, center=(320, 240), size=(200, 150))

        assert roi.shape == (150, 200, 3)


@pytest.mark.unit()
class TestVisualizationUtils:
    """Test visualization utilities."""

    @pytest.mark.opencv_available()
    def test_draw_detection_overlay(self, mock_detection_result):
        """Test drawing detection overlay."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        overlay_frame = draw_detection_overlay(frame, mock_detection_result)
        assert overlay_frame.shape == frame.shape

        # Frame should have been modified (circles drawn)
        assert not np.array_equal(frame, overlay_frame)

    @pytest.mark.opencv_available()
    def test_draw_ball_with_id(self):
        """Test drawing ball with ID label."""
        from vision.utils.visualization import draw_ball

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        ball_data = {"x": 320, "y": 240, "radius": 20, "id": "cue", "color": "white"}

        modified_frame = draw_ball(frame, ball_data)
        assert modified_frame.shape == frame.shape
        assert not np.array_equal(frame, modified_frame)

    @pytest.mark.opencv_available()
    def test_draw_table_outline(self):
        """Test drawing table outline."""
        from vision.utils.visualization import draw_table

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        corners = [(100, 100), (540, 100), (540, 380), (100, 380)]

        modified_frame = draw_table(frame, corners)
        assert modified_frame.shape == frame.shape
        assert not np.array_equal(frame, modified_frame)

    @pytest.mark.opencv_available()
    def test_draw_trajectory_prediction(self):
        """Test drawing trajectory prediction."""
        from vision.utils.visualization import draw_trajectory

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = [(100, 100), (150, 120), (200, 140), (250, 160)]

        modified_frame = draw_trajectory(frame, trajectory)
        assert modified_frame.shape == frame.shape
        assert not np.array_equal(frame, modified_frame)
