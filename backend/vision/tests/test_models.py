"""Tests for vision module data models.

Tests the comprehensive data structures for vision processing including:
- Ball detection and tracking models
- Cue stick detection models
- Table detection models
- Processing statistics
- Calibration data structures
"""

import time

import numpy as np
import pytest

from backend.vision.models import (
    Ball,
    BallType,
    CalibrationData,
    CameraCalibration,
    ColorCalibration,
    CueState,
    CueStick,
    DetectionResult,
    DetectionSession,
    FrameStatistics,
    GeometricCalibration,
    Pocket,
    PocketType,
    ShotEvent,
    Table,
)


class TestEnums:
    """Test enumeration types."""

    def test_ball_type_enum(self):
        """Test BallType enumeration."""
        assert BallType.CUE.value == "cue"
        assert BallType.SOLID.value == "solid"
        assert BallType.STRIPE.value == "stripe"
        assert BallType.EIGHT.value == "eight"
        assert BallType.UNKNOWN.value == "unknown"

    def test_pocket_type_enum(self):
        """Test PocketType enumeration."""
        assert PocketType.CORNER.value == "corner"
        assert PocketType.SIDE.value == "side"

    def test_cue_state_enum(self):
        """Test CueState enumeration."""
        assert CueState.HIDDEN.value == "hidden"
        assert CueState.AIMING.value == "aiming"
        assert CueState.STRIKING.value == "striking"
        assert CueState.RETRACTING.value == "retracting"


class TestBall:
    """Test Ball data model."""

    def test_ball_creation(self):
        """Test basic ball creation."""
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.CUE,
            confidence=0.95,
        )

        assert ball.position == (100.0, 200.0)
        assert ball.radius == 15.0
        assert ball.ball_type == BallType.CUE
        assert ball.confidence == 0.95
        assert ball.number is None
        assert ball.velocity == (0.0, 0.0)
        assert ball.acceleration == (0.0, 0.0)
        assert not ball.is_moving
        assert ball.track_id is None
        assert ball.age == 0
        assert ball.hit_count == 0

    def test_ball_with_all_fields(self):
        """Test ball creation with all fields."""
        timestamp = time.time()
        ball = Ball(
            position=(150.0, 250.0),
            radius=12.0,
            ball_type=BallType.SOLID,
            number=3,
            confidence=0.88,
            velocity=(5.0, -3.0),
            acceleration=(0.1, 0.2),
            is_moving=True,
            track_id=42,
            last_seen=timestamp,
            age=10,
            hit_count=2,
            color_hsv=(60, 255, 200),
            occlusion_state=0.1,
        )

        assert ball.number == 3
        assert ball.velocity == (5.0, -3.0)
        assert ball.acceleration == (0.1, 0.2)
        assert ball.is_moving
        assert ball.track_id == 42
        assert ball.age == 10
        assert ball.hit_count == 2
        assert ball.color_hsv == (60, 255, 200)
        assert ball.occlusion_state == 0.1

    def test_ball_update_history(self):
        """Test ball position history management."""
        ball = Ball(position=(100.0, 100.0), radius=15.0, ball_type=BallType.CUE)

        # Add some positions
        positions = [(100.0, 100.0), (110.0, 105.0), (120.0, 110.0)]
        for pos in positions:
            ball.position = pos
            ball.update_history()

        assert len(ball.position_history) == 3
        assert ball.position_history == positions

        # Test max history limit
        for i in range(15):  # Add more than max (10)
            ball.position = (130.0 + i, 115.0 + i)
            ball.update_history()

        assert len(ball.position_history) == 10  # Should be limited to 10
        assert (
            ball.position_history[0] != positions[0]
        )  # Old position should be removed


class TestPocket:
    """Test Pocket data model."""

    def test_pocket_creation(self):
        """Test pocket creation."""
        corners = [(50.0, 50.0), (70.0, 50.0), (70.0, 70.0), (50.0, 70.0)]
        pocket = Pocket(
            position=(60.0, 60.0),
            pocket_type=PocketType.CORNER,
            radius=25.0,
            corners=corners,
        )

        assert pocket.position == (60.0, 60.0)
        assert pocket.pocket_type == PocketType.CORNER
        assert pocket.radius == 25.0
        assert pocket.corners == corners


class TestCueStick:
    """Test CueStick data model."""

    def test_cue_stick_creation(self):
        """Test basic cue stick creation."""
        cue = CueStick(
            tip_position=(300.0, 200.0), angle=45.0, length=200.0, confidence=0.92
        )

        assert cue.tip_position == (300.0, 200.0)
        assert cue.angle == 45.0
        assert cue.length == 200.0
        assert cue.confidence == 0.92
        assert cue.state == CueState.HIDDEN
        assert not cue.is_aiming
        assert cue.tip_velocity == (0.0, 0.0)
        assert cue.angular_velocity == 0.0
        assert cue.width == 0.0
        assert cue.target_ball_id is None

    def test_cue_stick_with_tracking(self):
        """Test cue stick with motion tracking."""
        shaft_points = [(300.0, 200.0), (280.0, 195.0), (260.0, 190.0)]
        aiming_line = [(300.0, 200.0), (320.0, 205.0), (340.0, 210.0)]

        cue = CueStick(
            tip_position=(300.0, 200.0),
            angle=15.0,
            length=180.0,
            confidence=0.85,
            state=CueState.AIMING,
            is_aiming=True,
            tip_velocity=(2.0, 1.0),
            angular_velocity=0.5,
            shaft_points=shaft_points,
            width=8.0,
            target_ball_id=1,
            predicted_contact_point=(350.0, 220.0),
            aiming_line=aiming_line,
        )

        assert cue.state == CueState.AIMING
        assert cue.is_aiming
        assert cue.tip_velocity == (2.0, 1.0)
        assert cue.angular_velocity == 0.5
        assert cue.shaft_points == shaft_points
        assert cue.width == 8.0
        assert cue.target_ball_id == 1
        assert cue.predicted_contact_point == (350.0, 220.0)
        assert cue.aiming_line == aiming_line


class TestTable:
    """Test Table data model."""

    def test_table_creation(self):
        """Test basic table creation."""
        corners = [(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)]
        pockets = [
            Pocket((0.0, 0.0), PocketType.CORNER, 25.0, []),
            Pocket((400.0, 0.0), PocketType.SIDE, 20.0, []),
            Pocket((800.0, 0.0), PocketType.CORNER, 25.0, []),
            Pocket((800.0, 400.0), PocketType.CORNER, 25.0, []),
            Pocket((400.0, 400.0), PocketType.SIDE, 20.0, []),
            Pocket((0.0, 400.0), PocketType.CORNER, 25.0, []),
        ]

        table = Table(
            corners=corners,
            pockets=pockets,
            width=800.0,
            height=400.0,
            surface_color=(60, 180, 120),
        )

        assert table.corners == corners
        assert len(table.pockets) == 6
        assert table.width == 800.0
        assert table.height == 400.0
        assert table.surface_color == (60, 180, 120)

    def test_table_pocket_methods(self):
        """Test table pocket utility methods."""
        corner_pockets = [
            Pocket((0.0, 0.0), PocketType.CORNER, 25.0, []),
            Pocket((800.0, 400.0), PocketType.CORNER, 25.0, []),
        ]
        side_pockets = [
            Pocket((400.0, 0.0), PocketType.SIDE, 20.0, []),
            Pocket((400.0, 400.0), PocketType.SIDE, 20.0, []),
        ]

        table = Table(
            corners=[(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)],
            pockets=corner_pockets + side_pockets,
            width=800.0,
            height=400.0,
            surface_color=(60, 180, 120),
        )

        # Test get_pocket_by_type
        corner_result = table.get_pocket_by_type(PocketType.CORNER)
        side_result = table.get_pocket_by_type(PocketType.SIDE)

        assert len(corner_result) == 2
        assert len(side_result) == 2
        assert all(p.pocket_type == PocketType.CORNER for p in corner_result)
        assert all(p.pocket_type == PocketType.SIDE for p in side_result)

        # Test nearest_pocket
        nearest = table.nearest_pocket((100.0, 100.0))
        assert nearest is not None
        assert nearest.position == (0.0, 0.0)  # Should be closest to origin

        # Test with no pockets
        empty_table = Table(
            corners=[(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)],
            pockets=[],
            width=800.0,
            height=400.0,
            surface_color=(60, 180, 120),
        )
        assert empty_table.nearest_pocket((100.0, 100.0)) is None


class TestFrameStatistics:
    """Test FrameStatistics data model."""

    def test_frame_statistics_creation(self):
        """Test frame statistics creation."""
        timestamp = time.time()
        stats = FrameStatistics(
            frame_number=42,
            timestamp=timestamp,
            processing_time=25.5,
            capture_time=2.0,
            preprocessing_time=8.0,
            detection_time=12.0,
            tracking_time=2.5,
            postprocessing_time=1.0,
            balls_detected=5,
            balls_tracked=4,
            cue_detected=True,
            table_detected=True,
            detection_confidence=0.89,
            tracking_quality=0.92,
            frame_quality=0.85,
        )

        assert stats.frame_number == 42
        assert stats.timestamp == timestamp
        assert stats.processing_time == 25.5
        assert stats.capture_time == 2.0
        assert stats.balls_detected == 5
        assert stats.balls_tracked == 4
        assert stats.cue_detected
        assert stats.table_detected
        assert stats.detection_confidence == 0.89


class TestDetectionResult:
    """Test DetectionResult data model."""

    def test_detection_result_creation(self):
        """Test basic detection result creation."""
        timestamp = time.time()

        balls = [
            Ball((100.0, 100.0), 15.0, BallType.CUE, confidence=0.95),
            Ball((200.0, 150.0), 12.0, BallType.SOLID, number=3, confidence=0.88),
        ]

        cue = CueStick((300.0, 200.0), 45.0, 200.0, confidence=0.92)

        table = Table(
            corners=[(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)],
            pockets=[],
            width=800.0,
            height=400.0,
            surface_color=(60, 180, 120),
        )

        stats = FrameStatistics(
            frame_number=42, timestamp=timestamp, processing_time=25.5
        )

        result = DetectionResult(
            frame_number=42,
            timestamp=timestamp,
            balls=balls,
            cue=cue,
            table=table,
            statistics=stats,
        )

        assert result.frame_number == 42
        assert result.timestamp == timestamp
        assert len(result.balls) == 2
        assert result.cue is not None
        assert result.table is not None
        assert result.statistics == stats
        assert result.is_complete
        assert not result.has_errors
        assert result.game_state == "unknown"

    def test_detection_result_ball_methods(self):
        """Test detection result ball utility methods."""
        balls = [
            Ball((100.0, 100.0), 15.0, BallType.CUE, confidence=0.95),
            Ball(
                (200.0, 150.0),
                12.0,
                BallType.SOLID,
                number=3,
                confidence=0.88,
                is_moving=True,
            ),
            Ball((300.0, 200.0), 12.0, BallType.STRIPE, number=9, confidence=0.90),
            Ball((400.0, 250.0), 12.0, BallType.EIGHT, confidence=0.92, is_moving=True),
        ]

        stats = FrameStatistics(
            frame_number=1, timestamp=time.time(), processing_time=25.0
        )

        result = DetectionResult(
            frame_number=1,
            timestamp=time.time(),
            balls=balls,
            cue=None,
            table=None,
            statistics=stats,
        )

        # Test get_balls_by_type
        cue_balls = result.get_balls_by_type(BallType.CUE)
        solid_balls = result.get_balls_by_type(BallType.SOLID)
        stripe_balls = result.get_balls_by_type(BallType.STRIPE)

        assert len(cue_balls) == 1
        assert len(solid_balls) == 1
        assert len(stripe_balls) == 1

        # Test get_cue_ball
        cue_ball = result.get_cue_ball()
        assert cue_ball is not None
        assert cue_ball.ball_type == BallType.CUE

        # Test get_moving_balls
        moving_balls = result.get_moving_balls()
        assert len(moving_balls) == 2
        assert all(ball.is_moving for ball in moving_balls)

        # Test get_stationary_balls
        stationary_balls = result.get_stationary_balls()
        assert len(stationary_balls) == 2
        assert all(not ball.is_moving for ball in stationary_balls)


class TestCalibrationData:
    """Test calibration data models."""

    def test_camera_calibration(self):
        """Test camera calibration data."""
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        distortion_coeffs = np.array([0.1, -0.2, 0.01, 0.02, 0.0])

        calibration = CameraCalibration(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coeffs,
            resolution=(640, 480),
            reprojection_error=0.5,
        )

        assert np.array_equal(calibration.camera_matrix, camera_matrix)
        assert np.array_equal(calibration.distortion_coefficients, distortion_coeffs)
        assert calibration.resolution == (640, 480)
        assert calibration.reprojection_error == 0.5

    def test_color_calibration(self):
        """Test color calibration data."""
        table_range = ((40, 50, 50), (80, 255, 255))
        ball_ranges = {
            BallType.CUE: ((0, 0, 200), (180, 30, 255)),
            BallType.SOLID: ((100, 100, 100), (120, 255, 255)),
        }

        calibration = ColorCalibration(
            table_color_range=table_range,
            ball_color_ranges=ball_ranges,
            white_balance=(1.0, 1.1, 0.9),
            ambient_light_level=0.7,
        )

        assert calibration.table_color_range == table_range
        assert calibration.ball_color_ranges == ball_ranges
        assert calibration.white_balance == (1.0, 1.1, 0.9)
        assert calibration.ambient_light_level == 0.7

    def test_geometric_calibration(self):
        """Test geometric calibration data."""
        pixel_corners = [(0.0, 0.0), (800.0, 0.0), (800.0, 400.0), (0.0, 400.0)]
        world_corners = [(0.0, 0.0), (2.84, 0.0), (2.84, 1.42), (0.0, 1.42)]
        homography = np.eye(3)
        inverse_homography = np.eye(3)

        calibration = GeometricCalibration(
            table_corners_pixel=pixel_corners,
            table_corners_world=world_corners,
            homography_matrix=homography,
            inverse_homography=inverse_homography,
            table_dimensions_real=(2.84, 1.42),
            pixels_per_meter=282.0,
            calibration_quality=0.95,
        )

        assert calibration.table_corners_pixel == pixel_corners
        assert calibration.table_corners_world == world_corners
        assert np.array_equal(calibration.homography_matrix, homography)
        assert calibration.table_dimensions_real == (2.84, 1.42)
        assert calibration.pixels_per_meter == 282.0
        assert calibration.calibration_quality == 0.95

    def test_calibration_data_completeness(self):
        """Test calibration data completeness checking."""
        # Empty calibration
        empty_cal = CalibrationData()
        assert not empty_cal.is_complete()
        assert not empty_cal.is_valid()

        # Partial calibration
        camera_cal = CameraCalibration(
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
            resolution=(640, 480),
            reprojection_error=0.5,
        )
        partial_cal = CalibrationData(camera=camera_cal)
        assert not partial_cal.is_complete()
        assert not partial_cal.is_valid()

        # Complete calibration
        color_cal = ColorCalibration(
            table_color_range=((40, 50, 50), (80, 255, 255)),
            ball_color_ranges={},
            white_balance=(1.0, 1.0, 1.0),
            ambient_light_level=0.7,
        )
        geo_cal = GeometricCalibration(
            table_corners_pixel=[(0, 0), (800, 0), (800, 400), (0, 400)],
            table_corners_world=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
            homography_matrix=np.eye(3),
            inverse_homography=np.eye(3),
            table_dimensions_real=(2.84, 1.42),
            pixels_per_meter=282.0,
            calibration_quality=0.95,
        )

        complete_cal = CalibrationData(
            camera=camera_cal, colors=color_cal, geometry=geo_cal
        )
        assert complete_cal.is_complete()
        assert complete_cal.is_valid()


class TestSessionData:
    """Test session and event data models."""

    def test_detection_session(self):
        """Test detection session data."""
        config = {"fps": 30, "resolution": (640, 480)}

        session = DetectionSession(
            session_id="test_session_001",
            start_time=time.time(),
            total_frames=1000,
            total_shots=5,
            average_fps=29.8,
            average_processing_time=15.2,
            detection_accuracy=0.94,
            total_errors=3,
            error_rate=0.003,
            config_snapshot=config,
        )

        assert session.session_id == "test_session_001"
        assert session.total_frames == 1000
        assert session.total_shots == 5
        assert session.average_fps == 29.8
        assert session.detection_accuracy == 0.94
        assert session.config_snapshot == config

    def test_shot_event(self):
        """Test shot event data."""
        shot = ShotEvent(
            shot_id=1,
            timestamp=time.time(),
            cue_ball_position=(300.0, 200.0),
            target_ball_position=(400.0, 250.0),
            cue_angle=45.0,
            estimated_force=0.8,
            contact_point=(310.0, 205.0),
            balls_potted=[3, 7],
            final_positions={1: (100.0, 100.0), 3: (0.0, 0.0)},
            shot_quality=0.85,
        )

        assert shot.shot_id == 1
        assert shot.cue_ball_position == (300.0, 200.0)
        assert shot.target_ball_position == (400.0, 250.0)
        assert shot.cue_angle == 45.0
        assert shot.estimated_force == 0.8
        assert shot.balls_potted == [3, 7]
        assert shot.shot_quality == 0.85


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
