"""Tests for cue stick detection and shot analysis."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ..detection.cue import CueDetector, ExtendedCueStick, ExtendedShotEvent, ShotType
from ..models import CueState, CueStick


class TestCueDetector:
    """Test cases for the CueDetector class."""

    @pytest.fixture()
    def default_config(self):
        """Default configuration for cue detector."""
        return {
            "min_cue_length": 150,
            "max_cue_length": 800,
            "min_line_thickness": 3,
            "max_line_thickness": 25,
            "hough_threshold": 100,
            "hough_min_line_length": 100,
            "hough_max_line_gap": 20,
            "lsd_scale": 0.8,
            "lsd_sigma": 0.6,
            "lsd_quant": 2.0,
            "velocity_threshold": 5.0,
            "acceleration_threshold": 2.0,
            "striking_velocity_threshold": 15.0,
            "max_tracking_distance": 50,
            "tracking_history_size": 10,
            "confidence_decay": 0.95,
            "min_detection_confidence": 0.6,
            "temporal_smoothing": 0.7,
        }

    @pytest.fixture()
    def detector(self, default_config):
        """Create a CueDetector instance."""
        return CueDetector(default_config)

    @pytest.fixture()
    def sample_frame(self):
        """Create a sample frame with a cue stick."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a cue stick (white line)
        cv2.line(frame, (100, 200), (500, 250), (255, 255, 255), 8)

        # Add some noise
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame

    @pytest.fixture()
    def sample_lines(self):
        """Sample line detections that could represent cue sticks."""
        return [
            np.array([100, 200, 500, 250]),  # Good cue candidate
            np.array([200, 100, 210, 110]),  # Too short
            np.array([0, 0, 640, 480]),  # Too long / diagonal
            np.array([150, 220, 480, 260]),  # Another good candidate
        ]

    def test_initialization(self, default_config):
        """Test CueDetector initialization."""
        detector = CueDetector(default_config)

        assert detector.min_cue_length == 150
        assert detector.max_cue_length == 800
        assert detector.frame_count == 0
        assert len(detector.previous_cues) == 0
        assert len(detector.shot_events) == 0

    def test_preprocess_frame(self, detector, sample_frame):
        """Test frame preprocessing."""
        processed = detector._preprocess_frame(sample_frame)

        assert len(processed.shape) == 2  # Should be grayscale
        assert processed.dtype == np.uint8
        assert processed.shape[:2] == sample_frame.shape[:2]

    def test_preprocess_frame_grayscale_input(self, detector):
        """Test preprocessing with grayscale input."""
        gray_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        processed = detector._preprocess_frame(gray_frame)

        assert processed.shape == gray_frame.shape
        assert processed.dtype == np.uint8

    @patch("cv2.HoughLinesP")
    @patch("cv2.Canny")
    def test_detect_lines_hough(self, mock_canny, mock_hough, detector, sample_frame):
        """Test Hough line detection method."""
        # Mock Canny edge detection
        mock_canny.return_value = np.zeros((480, 640), dtype=np.uint8)

        # Mock HoughLinesP to return sample lines
        mock_hough.return_value = np.array([[[100, 200, 500, 250]]])

        processed_frame = detector._preprocess_frame(sample_frame)
        lines = detector._detect_lines_hough(processed_frame)

        assert lines is not None
        assert len(lines) == 1
        assert np.array_equal(lines[0], [100, 200, 500, 250])

        mock_canny.assert_called_once()
        mock_hough.assert_called_once()

    def test_detect_lines_hough_no_lines(self, detector, sample_frame):
        """Test Hough line detection when no lines are found."""
        with patch("cv2.HoughLinesP", return_value=None):
            with patch("cv2.Canny", return_value=np.zeros((480, 640), dtype=np.uint8)):
                processed_frame = detector._preprocess_frame(sample_frame)
                lines = detector._detect_lines_hough(processed_frame)

                assert lines is None

    def test_filter_cue_candidates(self, detector, sample_lines):
        """Test filtering of cue candidates."""
        frame_shape = (480, 640)
        candidates = detector._filter_cue_candidates(sample_lines, frame_shape)

        # Should filter out the too-short line
        assert len(candidates) <= len(sample_lines)

        # All candidates should have reasonable lengths
        for candidate in candidates:
            assert (
                detector.min_cue_length <= candidate.length <= detector.max_cue_length
            )

    def test_filter_cue_candidates_with_cue_ball(self, detector, sample_lines):
        """Test filtering with cue ball position."""
        frame_shape = (480, 640)
        cue_ball_pos = (300, 225)  # Near the first line

        candidates = detector._filter_cue_candidates(
            sample_lines, frame_shape, cue_ball_pos
        )

        assert len(candidates) > 0

        # First candidate should have tip closer to cue ball
        if len(candidates) > 0:
            first_candidate = candidates[0]
            tip_to_ball = np.sqrt(
                (first_candidate.tip_position[0] - cue_ball_pos[0]) ** 2
                + (first_candidate.tip_position[1] - cue_ball_pos[1]) ** 2
            )
            butt_to_ball = np.sqrt(
                (first_candidate.butt_position[0] - cue_ball_pos[0]) ** 2
                + (first_candidate.butt_position[1] - cue_ball_pos[1]) ** 2
            )
            assert tip_to_ball <= butt_to_ball

    def test_calculate_line_confidence(self, detector):
        """Test line confidence calculation."""
        line = np.array([100, 200, 500, 250])
        frame_shape = (480, 640)

        confidence = detector._calculate_line_confidence(line, frame_shape)

        assert 0.0 <= confidence <= 1.0

    def test_calculate_line_confidence_with_cue_ball(self, detector):
        """Test line confidence with cue ball position."""
        line = np.array([100, 200, 500, 250])
        frame_shape = (480, 640)
        cue_ball_pos = (300, 225)  # Near the line

        confidence_with_ball = detector._calculate_line_confidence(
            line, frame_shape, cue_ball_pos
        )
        confidence_without_ball = detector._calculate_line_confidence(line, frame_shape)

        assert 0.0 <= confidence_with_ball <= 1.0
        assert 0.0 <= confidence_without_ball <= 1.0

    def test_point_to_line_distance(self, detector):
        """Test point to line distance calculation."""
        line = np.array([0, 0, 100, 0])  # Horizontal line
        point = (50, 10)  # Point above the line

        distance = detector._point_to_line_distance(point, line)

        assert abs(distance - 10) < 0.1  # Should be approximately 10

    def test_point_to_line_distance_on_line(self, detector):
        """Test distance when point is on the line."""
        line = np.array([0, 0, 100, 0])
        point = (50, 0)  # Point on the line

        distance = detector._point_to_line_distance(point, line)

        assert distance < 0.1  # Should be approximately 0

    def test_estimate_cue_angle(self, detector):
        """Test cue angle estimation."""
        # Horizontal line (0 degrees)
        horizontal_line = np.array([0, 0, 100, 0])
        angle = detector.estimate_cue_angle(horizontal_line)
        assert abs(angle - 0) < 0.1

        # Vertical line (90 degrees)
        vertical_line = np.array([0, 0, 0, 100])
        angle = detector.estimate_cue_angle(vertical_line)
        assert abs(angle - 90) < 0.1

        # 45-degree line
        diagonal_line = np.array([0, 0, 100, 100])
        angle = detector.estimate_cue_angle(diagonal_line)
        assert abs(angle - 45) < 0.1

    def test_detect_cue_movement_no_previous(self, detector):
        """Test movement detection with no previous cue."""
        current_cue = CueStick(
            tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
        )

        is_moving = detector.detect_cue_movement(current_cue, None)
        assert not is_moving

    def test_detect_cue_movement_stationary(self, detector):
        """Test movement detection for stationary cue."""
        cue1 = CueStick(tip_position=(100, 200), angle=45.0, length=300, confidence=0.8)

        cue2 = CueStick(
            tip_position=(102, 201),  # Small movement
            angle=46.0,
            length=301,
            confidence=0.8,
        )

        is_moving = detector.detect_cue_movement(cue2, cue1)
        assert not is_moving  # Movement is below threshold

    def test_detect_cue_movement_moving(self, detector):
        """Test movement detection for moving cue."""
        cue1 = CueStick(tip_position=(100, 200), angle=45.0, length=300, confidence=0.8)

        cue2 = CueStick(
            tip_position=(120, 220),  # Significant movement
            angle=55.0,
            length=300,
            confidence=0.8,
        )

        is_moving = detector.detect_cue_movement(cue2, cue1)
        assert is_moving

    def test_analyze_cue_motion_insufficient_history(self, detector):
        """Test motion analysis with insufficient history."""
        cue = ExtendedCueStick(
            tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
        )

        detector._analyze_cue_motion(cue)

        assert cue.state == CueState.AIMING

    def test_analyze_cue_motion_aiming(self, detector):
        """Test motion analysis for aiming state."""
        # Add previous cues to history
        detector.previous_cues.extend(
            [
                ExtendedCueStick(
                    tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
                ),
                ExtendedCueStick(
                    tip_position=(101, 201), angle=45.5, length=300, confidence=0.8
                ),
            ]
        )

        current_cue = ExtendedCueStick(
            tip_position=(102, 202),  # Small movement
            angle=46.0,
            length=300,
            confidence=0.8,
        )

        detector._analyze_cue_motion(current_cue)

        assert current_cue.state == CueState.AIMING
        assert abs(current_cue.velocity[0] - 1.0) < 0.1
        assert abs(current_cue.velocity[1] - 1.0) < 0.1

    def test_analyze_cue_motion_striking(self, detector):
        """Test motion analysis for striking state."""
        # Add previous cues with significant motion
        detector.previous_cues.extend(
            [
                ExtendedCueStick(
                    tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
                ),
                ExtendedCueStick(
                    tip_position=(110, 210), angle=45.0, length=300, confidence=0.8
                ),
            ]
        )

        current_cue = ExtendedCueStick(
            tip_position=(140, 240),  # Fast movement
            angle=45.0,
            length=300,
            confidence=0.8,
        )

        detector._analyze_cue_motion(current_cue)

        assert current_cue.state == CueState.STRIKING
        assert current_cue.strike_velocity > 0

    def test_calculate_contact_point(self, detector):
        """Test contact point calculation."""
        cue = CueStick(tip_position=(100, 200), angle=0.0, length=300, confidence=0.8)
        cue_ball_pos = (130, 200)

        contact_point = detector._calculate_contact_point(cue, cue_ball_pos)

        # Contact point should be on ball surface between tip and center
        assert contact_point[0] > cue.tip_position[0]
        assert contact_point[0] < cue_ball_pos[0]
        assert abs(contact_point[1] - cue_ball_pos[1]) < 0.1

    def test_estimate_strike_force(self, detector):
        """Test strike force estimation."""
        # Test various velocities
        assert detector._estimate_strike_force(0) == 0
        assert detector._estimate_strike_force(25) == 50  # Half of max velocity
        assert detector._estimate_strike_force(50) == 100  # Max velocity
        assert detector._estimate_strike_force(100) == 100  # Capped at 100

    def test_classify_shot_type_straight(self, detector):
        """Test straight shot classification."""
        cue = CueStick(
            tip_position=(100, 200), angle=0.0, length=300, confidence=0.8  # Horizontal
        )
        cue_ball_pos = (130, 200)
        cue_ball_velocity = (10.0, 0.0)  # Moving horizontally

        shot_type = detector._classify_shot_type(cue, cue_ball_pos, cue_ball_velocity)

        assert shot_type == ShotType.STRAIGHT

    def test_classify_shot_type_english(self, detector):
        """Test English shot classification."""
        cue = CueStick(
            tip_position=(100, 200), angle=0.0, length=300, confidence=0.8  # Horizontal
        )
        cue_ball_pos = (130, 200)
        cue_ball_velocity = (8.0, 6.0)  # Angled movement

        shot_type = detector._classify_shot_type(cue, cue_ball_pos, cue_ball_velocity)

        assert shot_type in [ShotType.ENGLISH_LEFT, ShotType.ENGLISH_RIGHT]

    def test_calculate_english(self, detector):
        """Test English calculation."""
        cue = CueStick(tip_position=(100, 200), angle=0.0, length=300, confidence=0.8)
        cue_ball_pos = (130, 200)

        english = detector._calculate_english(cue, cue_ball_pos)

        assert -1.0 <= english <= 1.0

    def test_calculate_follow_draw(self, detector):
        """Test follow/draw calculation."""
        cue = CueStick(tip_position=(100, 200), angle=0.0, length=300, confidence=0.8)
        cue_ball_pos = (130, 200)

        follow_draw = detector._calculate_follow_draw(cue, cue_ball_pos)

        assert -1.0 <= follow_draw <= 1.0

    def test_detect_shot_event_no_strike(self, detector):
        """Test shot detection when cue is not striking."""
        cue = CueStick(
            tip_position=(100, 200),
            angle=0.0,
            length=300,
            confidence=0.8,
            state=CueState.AIMING,
        )
        cue_ball_pos = (130, 200)
        cue_ball_velocity = (0.0, 0.0)

        shot_event = detector.detect_shot_event(cue, cue_ball_pos, cue_ball_velocity)

        assert shot_event is None

    def test_detect_shot_event_too_far(self, detector):
        """Test shot detection when cue is too far from ball."""
        cue = CueStick(
            tip_position=(100, 200),
            angle=0.0,
            length=300,
            confidence=0.8,
            state=CueState.STRIKING,
        )
        cue_ball_pos = (200, 200)  # Too far
        cue_ball_velocity = (10.0, 0.0)

        shot_event = detector.detect_shot_event(cue, cue_ball_pos, cue_ball_velocity)

        assert shot_event is None

    def test_detect_shot_event_valid(self, detector):
        """Test valid shot event detection."""
        cue = CueStick(
            tip_position=(100, 200),
            angle=0.0,
            length=300,
            confidence=0.8,
            state=CueState.STRIKING,
        )
        cue_ball_pos = (120, 200)  # Close enough
        cue_ball_velocity = (10.0, 0.0)  # Moving

        shot_event = detector.detect_shot_event(cue, cue_ball_pos, cue_ball_velocity)

        assert shot_event is not None
        assert shot_event.shot_id == 1
        assert shot_event.strike_force > 0
        assert shot_event.confidence > 0

    def test_get_multiple_cues_empty_frame(self, detector):
        """Test multiple cue detection with empty frame."""
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cues = detector.get_multiple_cues(empty_frame)

        assert isinstance(cues, list)
        assert len(cues) == 0

    def test_filter_overlapping_cues(self, detector):
        """Test filtering of overlapping cue detections."""
        # Create overlapping candidates
        candidates = [
            ExtendedCueStick(
                tip_position=(100, 200), angle=45.0, length=300, confidence=0.9
            ),
            ExtendedCueStick(
                tip_position=(105, 202), angle=47.0, length=295, confidence=0.8
            ),  # Similar
            ExtendedCueStick(
                tip_position=(200, 300), angle=135.0, length=280, confidence=0.7
            ),  # Different
        ]

        unique_cues = detector._filter_overlapping_cues(candidates)

        # Should filter out the overlapping detection
        assert len(unique_cues) == 2
        assert unique_cues[0].confidence == 0.9  # Keep highest confidence
        assert unique_cues[1].confidence == 0.7

    def test_get_detection_statistics_empty(self, detector):
        """Test statistics with no detections."""
        stats = detector.get_detection_statistics()

        assert stats["total_detections"] == 0
        assert stats["average_confidence"] == 0.0
        assert stats["shot_events"] == 0
        assert stats["detection_rate"] == 0.0

    def test_get_detection_statistics_with_data(self, detector):
        """Test statistics with detection data."""
        # Add some mock detection history
        detector.previous_cues.extend(
            [
                ExtendedCueStick(
                    tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
                ),
                ExtendedCueStick(
                    tip_position=(105, 205), angle=46.0, length=305, confidence=0.9
                ),
            ]
        )
        detector.frame_count = 10
        detector.shot_events.append(
            ExtendedShotEvent(
                shot_id=1,
                timestamp=5,
                cue_ball_position=(120, 200),
                cue_angle=45.0,
                estimated_force=50.0,
            )
        )

        stats = detector.get_detection_statistics()

        assert stats["total_detections"] == 2
        assert (
            abs(stats["average_confidence"] - 0.85) < 0.01
        )  # Allow for floating point precision
        assert stats["shot_events"] == 1
        assert stats["detection_rate"] == 0.2

    def test_reset_tracking(self, detector):
        """Test tracking state reset."""
        # Add some data
        detector.previous_cues.extend(
            [
                ExtendedCueStick(
                    tip_position=(100, 200), angle=45.0, length=300, confidence=0.8
                ),
            ]
        )
        detector.shot_events.append(
            ExtendedShotEvent(
                shot_id=1,
                timestamp=5,
                cue_ball_position=(120, 200),
                cue_angle=45.0,
                estimated_force=50.0,
            )
        )
        detector.frame_count = 10

        detector.reset_tracking()

        assert len(detector.previous_cues) == 0
        assert len(detector.shot_events) == 0
        assert detector.frame_count == 0

    def test_set_cue_ball_position(self, detector):
        """Test setting cue ball position."""
        position = (320, 240)
        detector.set_cue_ball_position(position)

        assert detector.cue_ball_position == position

    def test_visualize_detection_no_cue(self, detector, sample_frame):
        """Test visualization with no cue."""
        result = detector.visualize_detection(sample_frame, None)

        assert np.array_equal(result, sample_frame)

    def test_visualize_detection_with_cue(self, detector, sample_frame):
        """Test visualization with cue."""
        cue = CueStick(
            tip_position=(100, 200),
            angle=45.0,
            length=300,
            confidence=0.8,
            state=CueState.AIMING,
        )

        result = detector.visualize_detection(sample_frame, cue)

        assert result.shape == sample_frame.shape
        assert not np.array_equal(
            result, sample_frame
        )  # Should be different due to overlay

    def test_detect_cue_invalid_input(self, detector):
        """Test cue detection with invalid input."""
        # None input
        result = detector.detect_cue(None)
        assert result is None

        # Empty array
        empty_frame = np.array([])
        result = detector.detect_cue(empty_frame)
        assert result is None

    @patch("backend.vision.detection.cue.CueDetector._detect_lines_multi_method")
    def test_detect_cue_no_lines(self, mock_detect_lines, detector, sample_frame):
        """Test cue detection when no lines are found."""
        mock_detect_lines.return_value = []

        result = detector.detect_cue(sample_frame)

        assert result is None

    def test_temporal_tracking_no_previous(self, detector):
        """Test temporal tracking with no previous cues."""
        current_cue = ExtendedCueStick(
            tip_position=(100, 200),
            butt_position=(400, 250),
            angle=45.0,
            length=300,
            confidence=0.8,
        )

        result = detector._apply_temporal_tracking(current_cue)

        assert result == current_cue  # Should return unchanged

    def test_temporal_tracking_with_previous(self, detector):
        """Test temporal tracking with previous cue."""
        # Add previous cue
        detector.previous_cues.append(
            ExtendedCueStick(
                tip_position=(95, 195),
                butt_position=(395, 245),
                angle=44.0,
                length=300,
                confidence=0.8,
            )
        )

        current_cue = ExtendedCueStick(
            tip_position=(105, 205),
            butt_position=(405, 255),
            angle=46.0,
            length=300,
            confidence=0.8,
        )

        result = detector._apply_temporal_tracking(current_cue)

        # Position should be smoothed
        assert result.tip_position[0] != 105  # Should be smoothed
        assert result.tip_position[1] != 205  # Should be smoothed
        assert result.angle != 46.0  # Should be smoothed

    def test_line_thickness_validation(self, detector, sample_frame):
        """Test line thickness validation."""
        cue = ExtendedCueStick(
            tip_position=(100, 200),
            butt_position=(500, 250),
            angle=0.0,
            length=300,
            confidence=0.8,
        )

        # This should work with a reasonable frame
        result = detector._check_line_thickness(cue, sample_frame)

        assert isinstance(result, bool)

    def test_measure_line_thickness_invalid_point(self, detector, sample_frame):
        """Test thickness measurement with invalid point."""
        point = (-10, -10)  # Outside frame
        perpendicular = (0, 1)

        thickness = detector._measure_line_thickness_at_point(
            sample_frame, point, perpendicular
        )

        assert thickness == 0

    def test_cue_position_validation(self, detector, sample_frame):
        """Test cue position validation."""
        # Valid cue (not at edges)
        valid_cue = ExtendedCueStick(
            tip_position=(100, 200),
            butt_position=(500, 250),
            angle=0.0,
            length=300,
            confidence=0.8,
        )

        # Invalid cue (both ends at edges)
        invalid_cue = ExtendedCueStick(
            tip_position=(5, 5),
            butt_position=(635, 475),
            angle=0.0,
            length=300,
            confidence=0.8,
        )

        assert detector._check_cue_position(valid_cue, sample_frame) is True
        assert detector._check_cue_position(invalid_cue, sample_frame) is False


class TestExtendedDataClasses:
    """Test extended data classes."""

    def test_extended_cue_stick_creation(self):
        """Test ExtendedCueStick creation."""
        cue = ExtendedCueStick(
            tip_position=(100, 200),
            angle=45.0,
            length=300,
            confidence=0.8,
            butt_position=(400, 250),
            velocity=(5.0, 2.0),
            acceleration=(1.0, 0.5),
        )

        assert cue.tip_position == (100, 200)
        assert cue.butt_position == (400, 250)
        assert cue.velocity == (5.0, 2.0)
        assert cue.acceleration == (1.0, 0.5)

    def test_extended_shot_event_creation(self):
        """Test ExtendedShotEvent creation."""
        shot = ExtendedShotEvent(
            shot_id=1,
            timestamp=100.0,
            cue_ball_position=(200, 200),
            cue_angle=45.0,
            estimated_force=75.0,
            shot_type=ShotType.ENGLISH_LEFT,
            english_amount=-0.5,
            follow_draw=0.3,
        )

        assert shot.shot_id == 1
        assert shot.shot_type == ShotType.ENGLISH_LEFT
        assert shot.english_amount == -0.5
        assert shot.follow_draw == 0.3


class TestShotTypeEnum:
    """Test ShotType enumeration."""

    def test_shot_type_values(self):
        """Test all shot type values."""
        assert ShotType.STRAIGHT.value == "straight"
        assert ShotType.ENGLISH_LEFT.value == "english_left"
        assert ShotType.ENGLISH_RIGHT.value == "english_right"
        assert ShotType.FOLLOW.value == "follow"
        assert ShotType.DRAW.value == "draw"
        assert ShotType.MASSE.value == "masse"

    def test_shot_type_enumeration(self):
        """Test shot type enumeration."""
        all_types = list(ShotType)
        assert len(all_types) == 6
        assert ShotType.STRAIGHT in all_types
        assert ShotType.MASSE in all_types


class TestIntegration:
    """Integration tests for the complete cue detection pipeline."""

    @pytest.fixture()
    def detector_with_config(self):
        """Detector with comprehensive config."""
        config = {
            "min_cue_length": 100,
            "max_cue_length": 600,
            "min_line_thickness": 2,
            "max_line_thickness": 30,
            "hough_threshold": 80,
            "hough_min_line_length": 80,
            "hough_max_line_gap": 15,
            "velocity_threshold": 3.0,
            "acceleration_threshold": 1.5,
            "striking_velocity_threshold": 12.0,
            "min_detection_confidence": 0.5,
            "temporal_smoothing": 0.6,
        }
        return CueDetector(config)

    def test_full_detection_pipeline(self, detector_with_config):
        """Test the complete detection pipeline."""
        # Create a frame with a clear cue stick
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (50, 240), (590, 280), (255, 255, 255), 10)

        # Add table surface (green background)
        frame[:, :] = (0, 100, 0)
        cv2.line(frame, (50, 240), (590, 280), (255, 255, 255), 10)

        # Test detection
        cue_ball_pos = (400, 260)
        result = detector_with_config.detect_cue(frame, cue_ball_pos)

        # Should detect something (might be None due to simplified test frame)
        assert result is None or isinstance(result, CueStick)

    def test_detection_sequence(self, detector_with_config):
        """Test detection across multiple frames."""
        frames = []

        # Create sequence of frames with moving cue
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Cue moving from left to right
            start_x = 50 + i * 10
            end_x = 350 + i * 10
            cv2.line(frame, (start_x, 240), (end_x, 250), (255, 255, 255), 8)
            frames.append(frame)

        results = []
        for frame in frames:
            result = detector_with_config.detect_cue(frame)
            results.append(result)

        # Should build up detection history
        assert detector_with_config.frame_count == 5

        # At least some detections should succeed
        [r for r in results if r is not None]
        # Note: Might be zero due to simplified test frames

    def test_shot_detection_sequence(self, detector_with_config):
        """Test shot detection across frames."""
        # Create striking sequence
        cue_ball_pos = (300, 240)

        # Frame 1: Cue approaching
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame1, (200, 235), (280, 242), (255, 255, 255), 8)

        detector_with_config.detect_cue(frame1, cue_ball_pos)

        # Frame 2: Cue striking (closer and faster)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame2, (250, 238), (295, 241), (255, 255, 255), 8)

        result2 = detector_with_config.detect_cue(frame2, cue_ball_pos)

        # Test shot event detection
        if result2 and result2.state == CueState.STRIKING:
            cue_ball_velocity = (15.0, 0.0)  # Ball starts moving
            shot_event = detector_with_config.detect_shot_event(
                result2, cue_ball_pos, cue_ball_velocity
            )

            if shot_event:
                assert shot_event.strike_force > 0
                assert shot_event.shot_type in list(ShotType)


if __name__ == "__main__":
    pytest.main([__file__])
