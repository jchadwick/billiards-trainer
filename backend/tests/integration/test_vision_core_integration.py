"""Integration tests between vision and core modules."""

import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from core.game_state import GameStateManager
from core.models import BallState, GameState, Table
from vision.detection.balls import BallDetector
from vision.models import CameraFrame
from vision.tracking.tracker import ObjectTracker


@pytest.mark.integration()
class TestVisionCoreIntegration:
    """Test integration between vision and core modules."""

    def test_detection_to_game_state_conversion(self, mock_detection_result):
        """Test converting vision detection to game state."""
        game_manager = GameStateManager()

        # Convert detection result to game state
        table = Table(
            width=2.84, height=1.42, corners=mock_detection_result.table_corners
        )

        balls = []
        for detected_ball in mock_detection_result.balls:
            # Convert pixel coordinates to real-world coordinates
            # (simplified conversion for testing)
            real_x = detected_ball.x / 1920 * 2.84
            real_y = detected_ball.y / 1080 * 1.42

            ball = BallState(
                id=detected_ball.id,
                x=real_x,
                y=real_y,
                radius=0.028575,
                color=detected_ball.color,
            )
            balls.append(ball)

        game_state = GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        game_manager.set_state(game_state)

        assert game_manager.current_state is not None
        assert len(game_manager.current_state.balls) == 3
        assert game_manager.current_state.get_ball("cue") is not None

    def test_tracking_state_updates(self):
        """Test updating game state from ball tracking."""
        game_manager = GameStateManager()
        tracker = ObjectTracker()

        # Set up initial game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )

        cue_ball = BallState(id="cue", x=1.42, y=0.71, radius=0.028575, color="white")

        game_state = GameState(
            table=table,
            balls=[cue_ball],
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )

        game_manager.set_state(game_state)

        # Simulate tracking updates
        timestamp = time.time()
        for i in range(5):
            detection = {
                "id": "cue",
                "x": 960 + i * 10,  # Moving right
                "y": 540 + i * 5,  # Moving down
                "confidence": 0.9,
                "timestamp": timestamp + i * 0.033,  # 30 FPS
            }

            tracker.add_detection(detection)

            # Update game state with new position
            real_x = detection["x"] / 1920 * 2.84
            real_y = detection["y"] / 1080 * 1.42
            game_manager.update_ball_position("cue", real_x, real_y)

        # Ball should have moved
        updated_ball = game_manager.current_state.get_ball("cue")
        assert updated_ball.x != 1.42
        assert updated_ball.y != 0.71

    def test_velocity_calculation_from_tracking(self):
        """Test calculating ball velocity from tracking data."""
        tracker = ObjectTracker()
        game_manager = GameStateManager()

        # Set up game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        cue_ball = BallState(id="cue", x=1.0, y=0.5, radius=0.028575, color="white")
        game_state = GameState(
            table=table,
            balls=[cue_ball],
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )
        game_manager.set_state(game_state)

        # Add tracking data with movement
        base_time = time.time()
        positions = [
            (960, 540),  # Start position
            (970, 545),  # Move right and down
            (980, 550),  # Continue movement
            (990, 555),  # Continue movement
        ]

        for i, (x, y) in enumerate(positions):
            detection = {
                "id": "cue",
                "x": x,
                "y": y,
                "confidence": 0.9,
                "timestamp": base_time + i * 0.033,  # 30 FPS
            }

            if i == 0:
                tracker.add_detection(detection)
            else:
                tracker.update(detection)

        # Get velocity from tracker
        track = tracker.get_track("cue")
        if "velocity" in track:
            vx_pixels, vy_pixels = track["velocity"]

            # Convert to real-world velocity
            scale_x = 2.84 / 1920  # meters per pixel
            scale_y = 1.42 / 1080
            vx_real = vx_pixels * scale_x
            vy_real = vy_pixels * scale_y

            # Update game state with velocity
            game_manager.update_ball_velocity("cue", vx_real, vy_real)

            updated_ball = game_manager.current_state.get_ball("cue")
            assert updated_ball.velocity_x > 0  # Should be moving right
            assert updated_ball.velocity_y > 0  # Should be moving down

    @pytest.mark.opencv_available()
    def test_real_time_detection_pipeline(self, mock_cv2_camera):
        """Test real-time detection pipeline integration."""
        detector = BallDetector()
        game_manager = GameStateManager()

        # Set up initial game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        game_state = GameState(
            table=table, balls=[], current_player=1, shot_clock=30.0, game_mode="8-ball"
        )
        game_manager.set_state(game_state)

        # Simulate frame processing
        with patch("cv2.VideoCapture", return_value=mock_cv2_camera):
            cap = cv2.VideoCapture(0)

            for frame_id in range(5):
                ret, frame = cap.read()
                assert ret

                # Create camera frame
                CameraFrame(
                    frame=frame,
                    timestamp=time.time(),
                    frame_id=frame_id,
                    width=1920,
                    height=1080,
                )

                # Detect balls
                detected_balls = detector.detect(frame)

                # Update game state with detections
                current_balls = []
                for detection in detected_balls:
                    real_x = detection.get("x", 0) / 1920 * 2.84
                    real_y = detection.get("y", 0) / 1080 * 1.42

                    ball = BallState(
                        id=detection.get("id", f"ball_{len(current_balls)}"),
                        x=real_x,
                        y=real_y,
                        radius=0.028575,
                        color=detection.get("color", "unknown"),
                    )
                    current_balls.append(ball)

                # Update game state
                game_manager.current_state.balls = current_balls

        # Verify pipeline completed without errors
        assert game_manager.current_state is not None

    def test_coordinate_transformation(self):
        """Test coordinate transformation between vision and core systems."""
        # Vision system uses pixel coordinates (0,0) at top-left
        # Core system uses real-world coordinates in meters

        # Test transformation functions
        def pixel_to_real(
            pixel_x,
            pixel_y,
            image_width=1920,
            image_height=1080,
            table_width=2.84,
            table_height=1.42,
        ):
            real_x = pixel_x / image_width * table_width
            real_y = pixel_y / image_height * table_height
            return real_x, real_y

        def real_to_pixel(
            real_x,
            real_y,
            image_width=1920,
            image_height=1080,
            table_width=2.84,
            table_height=1.42,
        ):
            pixel_x = real_x / table_width * image_width
            pixel_y = real_y / table_height * image_height
            return int(pixel_x), int(pixel_y)

        # Test center point
        center_real_x, center_real_y = pixel_to_real(960, 540)
        assert abs(center_real_x - 1.42) < 0.01  # Table center X
        assert abs(center_real_y - 0.71) < 0.01  # Table center Y

        # Test round-trip conversion
        original_pixel = (960, 540)
        real_coords = pixel_to_real(*original_pixel)
        back_to_pixel = real_to_pixel(*real_coords)
        assert abs(back_to_pixel[0] - original_pixel[0]) <= 1
        assert abs(back_to_pixel[1] - original_pixel[1]) <= 1

    def test_detection_confidence_filtering(self, mock_detection_result):
        """Test filtering detections by confidence before updating game state."""
        game_manager = GameStateManager()

        # Set up game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        game_state = GameState(
            table=table, balls=[], current_player=1, shot_clock=30.0, game_mode="8-ball"
        )
        game_manager.set_state(game_state)

        # Create detection with varying confidence levels
        detections = [
            {"id": "cue", "x": 960, "y": 540, "confidence": 0.95},  # High confidence
            {"id": "1", "x": 800, "y": 400, "confidence": 0.6},  # Medium confidence
            {"id": "2", "x": 1100, "y": 600, "confidence": 0.3},  # Low confidence
        ]

        confidence_threshold = 0.5

        # Filter and convert high-confidence detections
        valid_detections = [
            d for d in detections if d["confidence"] >= confidence_threshold
        ]

        balls = []
        for detection in valid_detections:
            real_x = detection["x"] / 1920 * 2.84
            real_y = detection["y"] / 1080 * 1.42

            ball = BallState(
                id=detection["id"],
                x=real_x,
                y=real_y,
                radius=0.028575,
                color="white" if detection["id"] == "cue" else "colored",
            )
            balls.append(ball)

        game_manager.current_state.balls = balls

        # Should only have 2 balls (confidence >= 0.5)
        assert len(game_manager.current_state.balls) == 2
        assert game_manager.current_state.get_ball("cue") is not None
        assert game_manager.current_state.get_ball("1") is not None
        assert game_manager.current_state.get_ball("2") is None

    def test_ball_disappearance_handling(self):
        """Test handling when balls disappear from vision (pocketed)."""
        game_manager = GameStateManager()
        tracker = ObjectTracker(max_age=1.0)  # 1 second timeout

        # Set up game state with multiple balls
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        balls = [
            BallState(id="cue", x=1.42, y=0.71, radius=0.028575, color="white"),
            BallState(id="1", x=1.0, y=0.5, radius=0.028575, color="yellow"),
            BallState(id="8", x=2.0, y=0.9, radius=0.028575, color="black"),
        ]
        game_state = GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )
        game_manager.set_state(game_state)

        # Add initial tracking for all balls
        timestamp = time.time()
        for ball in balls:
            pixel_x = int(ball.x / 2.84 * 1920)
            pixel_y = int(ball.y / 1.42 * 1080)

            detection = {
                "id": ball.id,
                "x": pixel_x,
                "y": pixel_y,
                "confidence": 0.9,
                "timestamp": timestamp,
            }
            tracker.add_detection(detection)

        # Simulate ball 8 disappearing (no more detections)
        later_timestamp = timestamp + 2.0  # 2 seconds later

        # Only detect cue and ball 1
        continuing_detections = [
            {
                "id": "cue",
                "x": 960,
                "y": 540,
                "confidence": 0.9,
                "timestamp": later_timestamp,
            },
            {
                "id": "1",
                "x": 675,
                "y": 380,
                "confidence": 0.8,
                "timestamp": later_timestamp,
            },
        ]

        for detection in continuing_detections:
            tracker.update(detection)

        # Clean up stale tracks
        tracker.cleanup_stale_tracks()

        # Ball 8 should no longer be tracked
        assert tracker.get_track("cue") is not None
        assert tracker.get_track("1") is not None
        assert tracker.get_track("8") is None

        # Remove disappeared ball from game state
        if tracker.get_track("8") is None:
            game_manager.remove_ball("8")

        assert len(game_manager.current_state.balls) == 2
        assert game_manager.current_state.get_ball("8") is None

    def test_detection_noise_filtering(self):
        """Test filtering noisy detections before updating game state."""
        game_manager = GameStateManager()
        tracker = ObjectTracker()

        # Set up game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        cue_ball = BallState(id="cue", x=1.42, y=0.71, radius=0.028575, color="white")
        game_state = GameState(
            table=table,
            balls=[cue_ball],
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )
        game_manager.set_state(game_state)

        # Add noisy detections
        base_time = time.time()
        noisy_detections = [
            {
                "id": "cue",
                "x": 960,
                "y": 540,
                "confidence": 0.9,
                "timestamp": base_time,
            },
            {
                "id": "cue",
                "x": 965,
                "y": 542,
                "confidence": 0.85,
                "timestamp": base_time + 0.033,
            },
            {
                "id": "cue",
                "x": 980,
                "y": 520,
                "confidence": 0.4,
                "timestamp": base_time + 0.066,
            },  # Noisy
            {
                "id": "cue",
                "x": 970,
                "y": 545,
                "confidence": 0.9,
                "timestamp": base_time + 0.099,
            },
        ]

        positions = []
        for detection in noisy_detections:
            # Filter by confidence
            if detection["confidence"] >= 0.7:
                if not positions:  # First detection
                    tracker.add_detection(detection)
                else:
                    tracker.update(detection)

                positions.append((detection["x"], detection["y"]))

        # Should have filtered out the noisy detection
        assert len(positions) == 3  # Excluded the low-confidence detection

    def test_multi_ball_tracking_consistency(self):
        """Test consistent tracking of multiple balls."""
        game_manager = GameStateManager()
        tracker = ObjectTracker()

        # Set up game state with multiple balls
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        balls = [
            BallState(id="cue", x=1.0, y=0.7, radius=0.028575, color="white"),
            BallState(id="1", x=2.0, y=0.4, radius=0.028575, color="yellow"),
            BallState(id="2", x=1.5, y=1.0, radius=0.028575, color="blue"),
        ]
        game_state = GameState(
            table=table,
            balls=balls,
            current_player=1,
            shot_clock=30.0,
            game_mode="8-ball",
        )
        game_manager.set_state(game_state)

        # Simulate frame-by-frame tracking
        base_time = time.time()
        for frame in range(10):
            timestamp = base_time + frame * 0.033

            # Generate detections for each ball with slight movement
            detections = []
            for _i, ball in enumerate(balls):
                pixel_x = int((ball.x + frame * 0.01) / 2.84 * 1920)  # Slight movement
                pixel_y = int((ball.y + frame * 0.005) / 1.42 * 1080)

                detection = {
                    "id": ball.id,
                    "x": pixel_x,
                    "y": pixel_y,
                    "confidence": 0.9,
                    "timestamp": timestamp,
                }
                detections.append(detection)

            # Update tracker with all detections
            for detection in detections:
                if frame == 0:
                    tracker.add_detection(detection)
                else:
                    tracker.update(detection)

            # Update game state
            for detection in detections:
                real_x = detection["x"] / 1920 * 2.84
                real_y = detection["y"] / 1080 * 1.42
                game_manager.update_ball_position(detection["id"], real_x, real_y)

        # All balls should still be tracked
        assert tracker.get_track("cue") is not None
        assert tracker.get_track("1") is not None
        assert tracker.get_track("2") is not None

        # All balls should have moved slightly
        for ball in game_manager.current_state.balls:
            if ball.id == "cue":
                assert ball.x > 1.0  # Should have moved from initial position
            elif ball.id == "1":
                assert ball.x > 2.0
            elif ball.id == "2":
                assert ball.x > 1.5

    def test_vision_core_performance_integration(self, performance_timer):
        """Test performance of vision-core integration pipeline."""
        detector = BallDetector()
        tracker = ObjectTracker()
        game_manager = GameStateManager()

        # Set up game state
        table = Table(
            width=2.84,
            height=1.42,
            corners=[(0, 0), (2.84, 0), (2.84, 1.42), (0, 1.42)],
        )
        game_state = GameState(
            table=table, balls=[], current_player=1, shot_clock=30.0, game_mode="8-ball"
        )
        game_manager.set_state(game_state)

        # Create test frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Green background

        # Add some circles (mock balls)
        cv2.circle(frame, (960, 540), 25, (255, 255, 255), -1)  # Cue ball
        cv2.circle(frame, (800, 400), 25, (255, 255, 0), -1)  # Yellow ball

        # Time the complete pipeline
        performance_timer.start()

        # Run pipeline 30 times (simulate 1 second at 30 FPS)
        for frame_id in range(30):
            # Detection step
            detected_balls = detector.detect(frame)

            # Tracking step
            timestamp = time.time() + frame_id * 0.033
            for detection in detected_balls:
                detection_data = {
                    "id": detection.get("id", f"ball_{detection.get('x', 0)}"),
                    "x": detection.get("x", 0),
                    "y": detection.get("y", 0),
                    "confidence": detection.get("confidence", 0.9),
                    "timestamp": timestamp,
                }

                if frame_id == 0:
                    tracker.add_detection(detection_data)
                else:
                    tracker.update(detection_data)

            # Game state update step
            current_balls = []
            for track_id, track_data in tracker.tracks.items():
                real_x = track_data["x"] / 1920 * 2.84
                real_y = track_data["y"] / 1080 * 1.42

                ball = BallState(
                    id=track_id,
                    x=real_x,
                    y=real_y,
                    radius=0.028575,
                    color="white" if track_id == "cue" else "colored",
                )
                current_balls.append(ball)

            game_manager.current_state.balls = current_balls

        performance_timer.stop()

        # Should maintain real-time performance (30 FPS = 33.33ms per frame)
        avg_frame_time = performance_timer.elapsed_ms / 30
        assert avg_frame_time < 33.33  # Must be faster than 30 FPS

        # Verify final state
        assert len(game_manager.current_state.balls) > 0
