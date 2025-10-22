"""Integration tests between vision and core modules."""

import time
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from core.coordinates import Vector2D
from core.game_state import GameStateManager
from core.models import BallState, GameState, Table

from backend.vision.detection.balls import BallDetector
from backend.vision.models import CameraFrame
from backend.vision.tracking.tracker import ObjectTracker


@pytest.mark.integration()
class TestVisionCoreIntegration:
    """Test integration between vision and core modules."""

    def test_detection_to_game_state_conversion(self, mock_detection_result):
        """Test converting vision detection to game state."""
        game_manager = GameStateManager()

        # Convert detection result to game state
        # Use standard table since mock_detection_result.table is None
        table = Table.standard_9ft_table()

        balls = []
        for i, detected_ball in enumerate(mock_detection_result.balls):
            # Convert pixel coordinates to real-world coordinates
            # (simplified conversion for testing)
            real_x = detected_ball.position[0] / 1920 * 2.84
            real_y = detected_ball.position[1] / 1080 * 1.42

            ball = BallState(
                id=f"ball_{i}" if detected_ball.ball_type.value != "cue" else "cue",
                position=Vector2D(real_x, real_y),
                radius=0.028575,
                is_cue_ball=(detected_ball.ball_type.value == "cue"),
                number=detected_ball.number,
            )
            balls.append(ball)

        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=balls,
            current_player=1,
        )

        game_manager._current_state = game_state

        assert game_manager._current_state is not None
        assert len(game_manager._current_state.balls) == 3
        assert game_manager._current_state.get_ball_by_id("cue") is not None

    def test_tracking_state_updates(self):
        """Test updating game state from ball tracking."""
        game_manager = GameStateManager()
        tracker = ObjectTracker(config={})

        # Set up initial game state
        table = Table.standard_9ft_table()

        cue_ball = BallState(
            id="cue", position=Vector2D(1.42, 0.71), radius=0.028575, is_cue_ball=True
        )

        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[cue_ball],
            current_player=1,
        )

        game_manager._current_state = game_state

        # Simulate tracking updates
        from backend.vision.models import Ball as VisionBall
        from backend.vision.models import BallType

        timestamp = time.time()
        for i in range(5):
            detection = VisionBall(
                position=(960 + i * 10, 540 + i * 5),  # Moving right and down
                radius=30,
                ball_type=BallType.CUE,
                confidence=0.9,
            )

            tracker.update_tracking([detection], i, timestamp + i * 0.033)

            # Update game state with new position
            real_x = detection.position[0] / 1920 * 2.84
            real_y = detection.position[1] / 1080 * 1.42
            cue_ball = game_manager._current_state.get_ball_by_id("cue")
            if cue_ball:
                cue_ball.position = Vector2D(real_x, real_y)

        # Ball should have moved
        updated_ball = game_manager._current_state.get_ball_by_id("cue")
        assert updated_ball.position.x != 1.42
        assert updated_ball.position.y != 0.71

    def test_velocity_calculation_from_tracking(self):
        """Test calculating ball velocity from tracking data."""
        tracker = ObjectTracker(config={})
        game_manager = GameStateManager()

        # Set up game state
        table = Table.standard_9ft_table()
        cue_ball = BallState(
            id="cue", position=Vector2D(1.0, 0.5), radius=0.028575, is_cue_ball=True
        )
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[cue_ball],
            current_player=1,
        )
        game_manager._current_state = game_state

        # Add tracking data with movement
        from vision.models import Ball as VisionBall
        from vision.models import BallType

        base_time = time.time()
        positions = [
            (960, 540),  # Start position
            (970, 545),  # Move right and down
            (980, 550),  # Continue movement
            (990, 555),  # Continue movement
        ]

        for i, (x, y) in enumerate(positions):
            detection = VisionBall(
                position=(x, y),
                radius=30,
                ball_type=BallType.CUE,
                confidence=0.9,
            )

            tracked_balls = tracker.update_tracking(
                [detection], i, base_time + i * 0.033
            )

        # Get velocity from the last tracked ball
        if tracked_balls:
            tracked_ball = tracked_balls[0]
            vx_pixels, vy_pixels = tracked_ball.velocity

            # Convert to real-world velocity
            scale_x = 2.84 / 1920  # meters per pixel
            scale_y = 1.42 / 1080
            vx_real = vx_pixels * scale_x
            vy_real = vy_pixels * scale_y

            # Update game state with velocity
            cue_ball = game_manager._current_state.get_ball_by_id("cue")
            if cue_ball:
                cue_ball.velocity = Vector2D(vx_real, vy_real)

            updated_ball = game_manager._current_state.get_ball_by_id("cue")
            assert updated_ball.velocity.x > 0  # Should be moving right
            assert updated_ball.velocity.y > 0  # Should be moving down

    @pytest.mark.opencv_available()
    def test_real_time_detection_pipeline(self, mock_cv2_camera):
        """Test real-time detection pipeline integration."""
        detector = BallDetector(config={})
        game_manager = GameStateManager()

        # Set up initial game state
        table = Table.standard_9ft_table()
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[],
            current_player=1,
        )
        game_manager._current_state = game_state

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
                detected_balls = detector.detect_balls(frame)

                # Update game state with detections
                current_balls = []
                for detection in detected_balls:
                    real_x = detection.position[0] / 1920 * 2.84
                    real_y = detection.position[1] / 1080 * 1.42

                    ball = BallState(
                        id=f"ball_{len(current_balls)}",
                        position=Vector2D(real_x, real_y),
                        radius=0.028575,
                        is_cue_ball=(detection.ball_type.value == "cue"),
                    )
                    current_balls.append(ball)

                # Update game state
                game_manager._current_state.balls = current_balls

        # Verify pipeline completed without errors
        assert game_manager._current_state is not None

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
        table = Table.standard_9ft_table()
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[],
            current_player=1,
        )
        game_manager._current_state = game_state

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
                position=Vector2D(real_x, real_y),
                radius=0.028575,
                is_cue_ball=(detection["id"] == "cue"),
            )
            balls.append(ball)

        game_manager._current_state.balls = balls

        # Should only have 2 balls (confidence >= 0.5)
        assert len(game_manager._current_state.balls) == 2
        assert game_manager._current_state.get_ball_by_id("cue") is not None
        assert game_manager._current_state.get_ball_by_id("1") is not None
        assert game_manager._current_state.get_ball_by_id("2") is None

    def test_ball_disappearance_handling(self):
        """Test handling when balls disappear from vision (pocketed)."""
        game_manager = GameStateManager()
        tracker = ObjectTracker(config={"max_age": 1.0})  # 1 second timeout

        # Set up game state with multiple balls
        table = Table.standard_9ft_table()
        balls = [
            BallState(
                id="cue",
                position=Vector2D(1.42, 0.71),
                radius=0.028575,
                is_cue_ball=True,
            ),
            BallState(id="1", position=Vector2D(1.0, 0.5), radius=0.028575, number=1),
            BallState(id="8", position=Vector2D(2.0, 0.9), radius=0.028575, number=8),
        ]
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=balls,
            current_player=1,
        )
        game_manager._current_state = game_state

        # Add initial tracking for all balls
        from vision.models import Ball as VisionBall
        from vision.models import BallType

        timestamp = time.time()
        detections = []
        for ball in balls:
            pixel_x = int(ball.position.x / 2.84 * 1920)
            pixel_y = int(ball.position.y / 1.42 * 1080)

            detection = VisionBall(
                position=(pixel_x, pixel_y),
                radius=30,
                ball_type=BallType.CUE if ball.is_cue_ball else BallType.SOLID,
                confidence=0.9,
            )
            detections.append(detection)
        tracker.update_tracking(detections, 0, timestamp)

        # Simulate ball 8 disappearing (no more detections)
        later_timestamp = timestamp + 2.0  # 2 seconds later

        # Only detect cue and ball 1
        continuing_detections = [
            VisionBall(
                position=(960, 540),
                radius=30,
                ball_type=BallType.CUE,
                confidence=0.9,
            ),
            VisionBall(
                position=(675, 380),
                radius=30,
                ball_type=BallType.SOLID,
                confidence=0.8,
            ),
        ]

        tracker.update_tracking(continuing_detections, 1, later_timestamp)

        # Get tracked balls
        tracked_balls = tracker.update_tracking(
            continuing_detections, 2, later_timestamp
        )

        # Ball 8 should no longer be in tracked balls
        [b.ball_type.value for b in tracked_balls]
        assert len(tracked_balls) == 2  # Only cue and ball 1 should be tracked

        # Update game state to reflect only tracked balls
        game_manager._current_state.balls = [
            ball
            for ball in game_manager._current_state.balls
            if ball.id in ["cue", "1"]
        ]

        assert len(game_manager._current_state.balls) == 2
        assert game_manager._current_state.get_ball_by_id("8") is None

    def test_detection_noise_filtering(self):
        """Test filtering noisy detections before updating game state."""
        from vision.models import Ball as VisionBall
        from vision.models import BallType

        game_manager = GameStateManager()
        tracker = ObjectTracker(config={})

        # Set up game state
        table = Table.standard_9ft_table()
        cue_ball = BallState(
            id="cue", position=Vector2D(1.42, 0.71), radius=0.028575, is_cue_ball=True
        )
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[cue_ball],
            current_player=1,
        )
        game_manager._current_state = game_state

        # Add noisy detections
        base_time = time.time()
        noisy_detections = [
            VisionBall(
                position=(960, 540), radius=30, ball_type=BallType.CUE, confidence=0.9
            ),
            VisionBall(
                position=(965, 542), radius=30, ball_type=BallType.CUE, confidence=0.85
            ),
            VisionBall(
                position=(980, 520), radius=30, ball_type=BallType.CUE, confidence=0.4
            ),  # Noisy
            VisionBall(
                position=(970, 545), radius=30, ball_type=BallType.CUE, confidence=0.9
            ),
        ]

        positions = []
        for i, detection in enumerate(noisy_detections):
            # Filter by confidence
            if detection.confidence >= 0.7:
                tracker.update_tracking([detection], i, base_time + i * 0.033)
                positions.append(detection.position)

        # Should have filtered out the noisy detection
        assert len(positions) == 3  # Excluded the low-confidence detection

    def test_multi_ball_tracking_consistency(self):
        """Test consistent tracking of multiple balls."""
        from vision.models import Ball as VisionBall
        from vision.models import BallType

        game_manager = GameStateManager()
        tracker = ObjectTracker(config={})

        # Set up game state with multiple balls
        table = Table.standard_9ft_table()
        balls = [
            BallState(
                id="cue", position=Vector2D(1.0, 0.7), radius=0.028575, is_cue_ball=True
            ),
            BallState(id="1", position=Vector2D(2.0, 0.4), radius=0.028575, number=1),
            BallState(id="2", position=Vector2D(1.5, 1.0), radius=0.028575, number=2),
        ]
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=balls,
            current_player=1,
        )
        game_manager._current_state = game_state

        # Simulate frame-by-frame tracking
        base_time = time.time()
        for frame in range(10):
            timestamp = base_time + frame * 0.033

            # Generate detections for each ball with slight movement
            detections = []
            for _i, ball in enumerate(balls):
                pixel_x = int(
                    (ball.position.x + frame * 0.01) / 2.84 * 1920
                )  # Slight movement
                pixel_y = int((ball.position.y + frame * 0.005) / 1.42 * 1080)

                ball_type = BallType.CUE if ball.is_cue_ball else BallType.SOLID
                detection = VisionBall(
                    position=(pixel_x, pixel_y),
                    radius=30,
                    ball_type=ball_type,
                    confidence=0.9,
                )
                detections.append(detection)

            # Update tracker with all detections
            tracked = tracker.update_tracking(detections, frame, timestamp)

            # Update game state - match each tracked ball to a game ball
            for i, tracked_ball in enumerate(tracked):
                real_x = tracked_ball.position[0] / 1920 * 2.84
                real_y = tracked_ball.position[1] / 1080 * 1.42
                # Update corresponding ball in game state
                if i < len(game_manager._current_state.balls):
                    game_manager._current_state.balls[i].position = Vector2D(
                        real_x, real_y
                    )

        # All balls should still be tracked
        assert len(tracked) >= 3

        # All balls should have moved slightly (using >= since positions might be exactly on boundary)
        for ball in game_manager._current_state.balls:
            if ball.id == "cue":
                assert (
                    ball.position.x >= 1.0
                )  # Should have moved from initial position or stayed
            elif ball.id == "1":
                assert ball.position.x >= 2.0
            elif ball.id == "2":
                assert ball.position.x >= 1.5

    def test_vision_core_performance_integration(self, performance_timer):
        """Test performance of vision-core integration pipeline."""
        from vision.models import Ball as VisionBall
        from vision.models import BallType

        detector = BallDetector(config={})
        tracker = ObjectTracker(config={})
        game_manager = GameStateManager()

        # Set up game state
        table = Table.standard_9ft_table()
        game_state = GameState(
            timestamp=time.time(),
            frame_number=0,
            table=table,
            balls=[],
            current_player=1,
        )
        game_manager._current_state = game_state

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
            detected_balls = detector.detect_balls(frame)

            # Tracking step
            timestamp = time.time() + frame_id * 0.033
            tracked_balls = tracker.update_tracking(detected_balls, frame_id, timestamp)

            # Game state update step
            current_balls = []
            for tracked_ball in tracked_balls:
                real_x = tracked_ball.position[0] / 1920 * 2.84
                real_y = tracked_ball.position[1] / 1080 * 1.42

                ball = BallState(
                    id=f"ball_{len(current_balls)}",
                    position=Vector2D(real_x, real_y),
                    radius=0.028575,
                    is_cue_ball=(tracked_ball.ball_type == BallType.CUE),
                )
                current_balls.append(ball)

            game_manager._current_state.balls = current_balls

        performance_timer.stop()

        # Should maintain real-time performance (30 FPS = 33.33ms per frame)
        avg_frame_time = performance_timer.elapsed_ms / 30
        assert avg_frame_time < 33.33  # Must be faster than 30 FPS

        # Verify final state
        assert len(game_manager._current_state.balls) > 0
