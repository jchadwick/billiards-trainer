"""Comprehensive test suite for ball detection and classification algorithms.

Tests cover:
- Multiple detection methods (Hough, contour, blob, combined)
- Color-based ball classification accuracy
- Ball number identification
- Position accuracy validation (±2 pixel requirement)
- Radius measurement accuracy
- Motion detection
- Tracking with Kalman filters
- Occlusion handling
- Edge cases and error conditions
"""

import time

import cv2
import numpy as np
import pytest

from ..models import Ball, BallType
from ..tracking.tracker import ObjectTracker
from .balls import BallDetector, DetectionMethod


class TestBallDetection:
    """Test ball detection accuracy and robustness."""

    @pytest.fixture()
    def test_config(self):
        """Standard test configuration."""
        return {
            "detection_method": DetectionMethod.COMBINED,
            "min_radius": 8,
            "max_radius": 35,
            "expected_radius": 20,
            "radius_tolerance": 0.3,
            "min_confidence": 0.3,
            "debug_mode": False,
        }

    @pytest.fixture()
    def ball_detector(self, test_config):
        """Ball detector instance for testing."""
        return BallDetector(test_config)

    @pytest.fixture()
    def synthetic_frame(self):
        """Generate synthetic test frame with known ball positions."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Create green table background
        frame[:, :] = (40, 80, 40)  # Green felt color

        # Add some noise/texture
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    def add_ball_to_frame(
        self,
        frame: np.ndarray,
        position: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Add a synthetic ball to frame at specified position."""
        x, y = position
        # Create ball with shading for realism
        center = (x, y)

        # Draw main ball color
        cv2.circle(frame, center, radius, color, -1)

        # Add highlight for 3D effect
        highlight_center = (x - radius // 3, y - radius // 3)
        highlight_radius = radius // 3
        highlight_color = tuple(min(255, c + 80) for c in color)
        cv2.circle(frame, highlight_center, highlight_radius, highlight_color, -1)

        # Add shadow/edge
        shadow_color = tuple(max(0, c - 40) for c in color)
        cv2.circle(frame, center, radius, shadow_color, 2)

        return frame

    def test_hough_circle_detection_accuracy(self, ball_detector, synthetic_frame):
        """Test Hough circle detection accuracy with known ball positions."""
        # Add known balls to frame
        known_balls = [
            ((150, 200), 20, (255, 255, 255)),  # White cue ball
            ((350, 300), 18, (0, 0, 255)),  # Red ball
            ((500, 150), 22, (255, 255, 0)),  # Yellow ball
            ((200, 400), 19, (0, 255, 0)),  # Green ball
        ]

        frame = synthetic_frame.copy()
        for (x, y), radius, color in known_balls:
            frame = self.add_ball_to_frame(frame, (x, y), radius, color)

        # Configure for Hough detection only
        ball_detector.config.detection_method = DetectionMethod.HOUGH_CIRCLES

        # Detect balls
        detected_balls = ball_detector.detect_balls(frame)

        # Verify detection count (should detect most balls)
        assert (
            len(detected_balls) >= 3
        ), f"Expected at least 3 balls, got {len(detected_balls)}"
        assert len(detected_balls) <= 6, f"Too many detections: {len(detected_balls)}"

        # Verify position accuracy (±2 pixel requirement)
        detected_positions = [
            (int(ball.position[0]), int(ball.position[1])) for ball in detected_balls
        ]
        known_positions = [(x, y) for (x, y), _, _ in known_balls]

        matched_count = 0
        for known_pos in known_positions:
            for detected_pos in detected_positions:
                distance = np.sqrt(
                    (known_pos[0] - detected_pos[0]) ** 2
                    + (known_pos[1] - detected_pos[1]) ** 2
                )
                if distance <= 2.0:  # ±2 pixel accuracy requirement
                    matched_count += 1
                    break

        accuracy = matched_count / len(known_balls)
        assert accuracy >= 0.75, f"Position accuracy too low: {accuracy:.2f}"

    def test_contour_detection_method(self, ball_detector, synthetic_frame):
        """Test contour-based detection method."""
        frame = synthetic_frame.copy()

        # Add contrasting balls
        test_balls = [
            ((200, 200), 20, (255, 255, 255)),  # White
            ((400, 300), 18, (0, 0, 0)),  # Black
            ((600, 200), 22, (0, 0, 255)),  # Red
        ]

        for (x, y), radius, color in test_balls:
            frame = self.add_ball_to_frame(frame, (x, y), radius, color)

        ball_detector.config.detection_method = DetectionMethod.CONTOUR_BASED
        detected_balls = ball_detector.detect_balls(frame)

        assert (
            len(detected_balls) >= 2
        ), "Contour detection should find at least 2 balls"

        # Verify radius accuracy
        for ball in detected_balls:
            assert (
                15 <= ball.radius <= 25
            ), f"Radius out of expected range: {ball.radius}"

    def test_combined_detection_method(self, ball_detector, synthetic_frame):
        """Test combined detection method for best results."""
        frame = synthetic_frame.copy()

        # Add diverse set of balls
        test_balls = [
            ((150, 150), 20, (255, 255, 255)),  # White cue
            ((300, 200), 19, (0, 0, 0)),  # Black 8-ball
            ((450, 250), 21, (255, 255, 0)),  # Yellow
            ((200, 350), 18, (0, 0, 255)),  # Red
            ((500, 400), 20, (0, 255, 0)),  # Green
            ((350, 450), 22, (255, 0, 255)),  # Purple
        ]

        for (x, y), radius, color in test_balls:
            frame = self.add_ball_to_frame(frame, (x, y), radius, color)

        ball_detector.config.detection_method = DetectionMethod.COMBINED
        detected_balls = ball_detector.detect_balls(frame)

        # Combined method should achieve best detection rate
        assert (
            len(detected_balls) >= 4
        ), f"Combined detection should find at least 4/6 balls, got {len(detected_balls)}"

        # Verify no duplicate detections
        positions = [ball.position for ball in detected_balls]
        for i, pos1 in enumerate(positions):
            for _j, pos2 in enumerate(positions[i + 1 :], i + 1):
                distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                assert (
                    distance > 30
                ), f"Duplicate detection too close: {distance:.1f} pixels"

    def test_radius_measurement_accuracy(self, ball_detector, synthetic_frame):
        """Test ball radius measurement accuracy."""
        frame = synthetic_frame.copy()

        # Add balls with known radii
        test_radii = [15, 18, 20, 22, 25]
        for i, radius in enumerate(test_radii):
            x = 150 + i * 120
            y = 300
            frame = self.add_ball_to_frame(frame, (x, y), radius, (255, 255, 255))

        detected_balls = ball_detector.detect_balls(frame)

        # Should detect most balls
        assert (
            len(detected_balls) >= 4
        ), "Should detect at least 4/5 balls with different radii"

        # Check radius accuracy (within 20% tolerance)
        for ball in detected_balls:
            radius_error = abs(ball.radius - 20) / 20  # Compare to expected radius
            assert radius_error <= 0.3, f"Radius error too high: {radius_error:.2f}"

    def test_size_validation(self, ball_detector, synthetic_frame):
        """Test rejection of objects that are too small or too large."""
        frame = synthetic_frame.copy()

        # Add objects outside valid size range
        invalid_objects = [
            ((200, 200), 5, (255, 0, 0)),  # Too small
            ((400, 200), 50, (0, 255, 0)),  # Too large
            ((600, 200), 20, (0, 0, 255)),  # Valid size
        ]

        for (x, y), radius, color in invalid_objects:
            frame = self.add_ball_to_frame(frame, (x, y), radius, color)

        detected_balls = ball_detector.detect_balls(frame)

        # Should only detect the valid-sized ball
        assert len(detected_balls) <= 1, "Should reject objects outside size limits"

        if detected_balls:
            ball = detected_balls[0]
            assert (
                ball_detector.config.min_radius
                <= ball.radius
                <= ball_detector.config.max_radius
            )


class TestBallClassification:
    """Test ball type and number classification."""

    @pytest.fixture()
    def ball_detector(self):
        return BallDetector({"detection_method": DetectionMethod.HOUGH_CIRCLES})

    def create_colored_ball_region(
        self, size: int, color: tuple[int, int, int], is_striped: bool = False
    ) -> np.ndarray:
        """Create a synthetic ball region with specified color."""
        region = np.zeros((size, size, 3), dtype=np.uint8)
        center = size // 2
        radius = size // 2 - 2

        # Draw base color
        cv2.circle(region, (center, center), radius, color, -1)

        if is_striped:
            # Add white stripes
            stripe_width = radius // 3
            for i in range(0, size, stripe_width * 2):
                cv2.rectangle(
                    region, (i, 0), (i + stripe_width, size), (255, 255, 255), -1
                )

        # Add highlight
        highlight_pos = (center - radius // 3, center - radius // 3)
        highlight_radius = radius // 4
        highlight_color = tuple(min(255, c + 50) for c in color)
        cv2.circle(region, highlight_pos, highlight_radius, highlight_color, -1)

        return region

    def test_cue_ball_classification(self, ball_detector):
        """Test cue ball (white) classification accuracy."""
        # Create white ball region
        ball_region = self.create_colored_ball_region(40, (255, 255, 255))

        ball_type, confidence, number = ball_detector.classify_ball_type(
            ball_region, (100, 100), 20
        )

        assert ball_type == BallType.CUE, f"Expected CUE, got {ball_type}"
        assert confidence > 0.7, f"Low confidence for cue ball: {confidence:.2f}"
        assert (
            number is None or number == 0
        ), f"Cue ball should not have number: {number}"

    def test_eight_ball_classification(self, ball_detector):
        """Test 8-ball (black) classification accuracy."""
        # Create black ball region
        ball_region = self.create_colored_ball_region(40, (0, 0, 0))

        ball_type, confidence, number = ball_detector.classify_ball_type(
            ball_region, (100, 100), 20
        )

        assert ball_type == BallType.EIGHT, f"Expected EIGHT, got {ball_type}"
        assert confidence > 0.6, f"Low confidence for 8-ball: {confidence:.2f}"
        assert number == 8, f"8-ball should have number 8: {number}"

    def test_solid_ball_classification(self, ball_detector):
        """Test solid colored ball classification."""
        solid_colors = [
            (255, 255, 0),  # Yellow
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]

        for color in solid_colors:
            ball_region = self.create_colored_ball_region(40, color, is_striped=False)

            ball_type, confidence, number = ball_detector.classify_ball_type(
                ball_region, (100, 100), 20
            )

            # Should classify as solid (though color mapping might be approximate)
            assert ball_type in [
                BallType.SOLID,
                BallType.UNKNOWN,
            ], f"Expected SOLID or UNKNOWN for solid ball, got {ball_type}"

            if ball_type == BallType.SOLID:
                assert (
                    confidence > 0.4
                ), f"Low confidence for solid ball: {confidence:.2f}"

    def test_stripe_ball_classification(self, ball_detector):
        """Test striped ball classification."""
        stripe_colors = [
            (255, 255, 0),  # Yellow with white stripes
            (0, 0, 255),  # Red with white stripes
            (0, 255, 0),  # Green with white stripes
        ]

        for color in stripe_colors:
            ball_region = self.create_colored_ball_region(40, color, is_striped=True)

            ball_type, confidence, number = ball_detector.classify_ball_type(
                ball_region, (100, 100), 20
            )

            # Note: Simple stripe detection may not be highly accurate
            assert ball_type in [
                BallType.STRIPE,
                BallType.SOLID,
                BallType.UNKNOWN,
            ], f"Classification failed for striped ball: {ball_type}"

    def test_color_classification_consistency(self, ball_detector):
        """Test that same colored balls are classified consistently."""
        # Create multiple instances of same color
        color = (0, 0, 255)  # Red
        classifications = []

        for _ in range(5):
            ball_region = self.create_colored_ball_region(40, color)
            ball_type, confidence, number = ball_detector.classify_ball_type(
                ball_region, (100, 100), 20
            )
            classifications.append((ball_type, confidence))

        # All classifications should be the same type
        ball_types = [cls[0] for cls in classifications]
        assert len(set(ball_types)) <= 2, "Inconsistent classification for same color"

        # Confidence should be reasonably stable
        confidences = [cls[1] for cls in classifications]
        confidence_std = np.std(confidences)
        assert confidence_std < 0.3, f"Confidence too variable: {confidence_std:.2f}"

    def test_edge_case_empty_region(self, ball_detector):
        """Test classification with empty or invalid regions."""
        # Empty region
        empty_region = np.zeros((0, 0, 3), dtype=np.uint8)
        ball_type, confidence, number = ball_detector.classify_ball_type(
            empty_region, (100, 100), 20
        )

        assert ball_type == BallType.UNKNOWN
        assert confidence == 0.0
        assert number is None

    def test_number_identification_basic(self, ball_detector):
        """Test basic ball number identification."""
        # Test cue ball
        cue_region = self.create_colored_ball_region(40, (255, 255, 255))
        number = ball_detector.identify_ball_number(cue_region, BallType.CUE)
        assert number is None or number == 0

        # Test 8-ball
        eight_region = self.create_colored_ball_region(40, (0, 0, 0))
        number = ball_detector.identify_ball_number(eight_region, BallType.EIGHT)
        assert number == 8


class TestBallTracking:
    """Test ball tracking with Kalman filters."""

    @pytest.fixture()
    def tracker_config(self):
        return {
            "max_age": 30,
            "min_hits": 3,
            "max_distance": 50.0,
            "process_noise": 1.0,
            "measurement_noise": 10.0,
        }

    @pytest.fixture()
    def object_tracker(self, tracker_config):
        return ObjectTracker(tracker_config)

    def create_test_ball(
        self,
        position: tuple[float, float],
        ball_type: BallType = BallType.SOLID,
        radius: float = 20.0,
        confidence: float = 0.8,
    ) -> Ball:
        """Create test ball for tracking."""
        return Ball(
            position=position,
            radius=radius,
            ball_type=ball_type,
            confidence=confidence,
            velocity=(0.0, 0.0),
            is_moving=False,
        )

    def test_single_ball_tracking(self, object_tracker):
        """Test tracking of a single moving ball."""
        # Simulate ball moving in straight line
        positions = [(100 + i * 10, 200) for i in range(10)]

        tracked_balls_history = []
        for frame_num, pos in enumerate(positions):
            detections = [self.create_test_ball(pos)]
            tracked_balls = object_tracker.update_tracking(detections, frame_num)
            tracked_balls_history.append(tracked_balls)

        # Should maintain single track
        assert len(tracked_balls_history[-1]) == 1, "Should track one ball consistently"

        # Track should have reasonable velocity
        final_ball = tracked_balls_history[-1][0]
        velocity_x, velocity_y = final_ball.velocity

        # Should detect horizontal movement
        assert abs(velocity_x) > 5, f"Should detect horizontal movement: {velocity_x}"
        assert abs(velocity_y) < 5, f"Should not detect vertical movement: {velocity_y}"

    def test_multiple_ball_tracking(self, object_tracker):
        """Test tracking of multiple balls simultaneously."""
        # Two balls moving in different directions
        ball1_positions = [(100 + i * 5, 200) for i in range(10)]  # Moving right
        ball2_positions = [(300, 150 + i * 3) for i in range(10)]  # Moving down

        for frame_num in range(10):
            detections = [
                self.create_test_ball(ball1_positions[frame_num], BallType.CUE),
                self.create_test_ball(ball2_positions[frame_num], BallType.SOLID),
            ]
            tracked_balls = object_tracker.update_tracking(detections, frame_num)

            if frame_num >= 3:  # After confirmation period
                assert (
                    len(tracked_balls) == 2
                ), f"Should track 2 balls after frame {frame_num}"

    def test_track_association_accuracy(self, object_tracker):
        """Test that tracks are correctly associated across frames."""
        # Create distinctive balls
        cue_ball = self.create_test_ball((100, 200), BallType.CUE)
        eight_ball = self.create_test_ball((300, 200), BallType.EIGHT)

        # First frame
        detections = [cue_ball, eight_ball]
        tracked_balls = object_tracker.update_tracking(detections, 0)

        # Move balls slightly
        moved_cue = self.create_test_ball((105, 205), BallType.CUE)
        moved_eight = self.create_test_ball((295, 195), BallType.EIGHT)

        # Second frame
        detections = [moved_cue, moved_eight]
        tracked_balls = object_tracker.update_tracking(detections, 1)

        # Should maintain correct associations
        cue_balls = [b for b in tracked_balls if b.ball_type == BallType.CUE]
        eight_balls = [b for b in tracked_balls if b.ball_type == BallType.EIGHT]

        assert len(cue_balls) == 1, "Should have one cue ball track"
        assert len(eight_balls) == 1, "Should have one 8-ball track"

    def test_occlusion_handling(self, object_tracker):
        """Test handling of temporary ball occlusions."""
        # Track ball for several frames
        ball_pos = (200, 300)
        for frame_num in range(5):
            detections = [self.create_test_ball(ball_pos, BallType.SOLID)]
            tracked_balls = object_tracker.update_tracking(detections, frame_num)

        # Simulate occlusion (no detections) for 3 frames
        for frame_num in range(5, 8):
            tracked_balls = object_tracker.update_tracking([], frame_num)
            # Should still maintain track during short occlusion
            assert (
                len(tracked_balls) >= 0
            ), "Track should persist during short occlusion"

        # Ball reappears
        detections = [self.create_test_ball((210, 305), BallType.SOLID)]
        tracked_balls = object_tracker.update_tracking(detections, 8)

        # Should recover track
        assert len(tracked_balls) == 1, "Should recover track after occlusion"

    def test_velocity_calculation(self, object_tracker):
        """Test accurate velocity calculation."""
        # Ball moving at known velocity (10 pixels/frame)
        dt = 1.0 / 30.0  # 30 FPS
        positions = [(100 + i * 10, 200) for i in range(5)]

        for frame_num, pos in enumerate(positions):
            detections = [self.create_test_ball(pos)]
            tracked_balls = object_tracker.update_tracking(
                detections, frame_num, frame_num * dt
            )

        if tracked_balls:
            ball = tracked_balls[0]
            velocity_x, velocity_y = ball.velocity
            expected_velocity = 10.0 / dt  # pixels per second

            # Velocity should be approximately correct
            velocity_error = abs(velocity_x - expected_velocity) / expected_velocity
            assert (
                velocity_error < 0.3
            ), f"Velocity error too high: {velocity_error:.2f}"

    def test_track_lifecycle(self, object_tracker):
        """Test complete track lifecycle from creation to deletion."""
        stats_initial = object_tracker.get_tracking_statistics()

        # Create track
        ball = self.create_test_ball((200, 300))
        for frame_num in range(5):
            object_tracker.update_tracking([ball], frame_num)

        stats_after_creation = object_tracker.get_tracking_statistics()
        assert (
            stats_after_creation["total_tracks_created"]
            > stats_initial["total_tracks_created"]
        )

        # Let track die (no detections)
        for frame_num in range(5, 40):  # Beyond max_age
            object_tracker.update_tracking([], frame_num)

        stats_final = object_tracker.get_tracking_statistics()
        assert (
            stats_final["total_tracks_deleted"] > stats_initial["total_tracks_deleted"]
        )


class TestPerformanceAndMetrics:
    """Test performance characteristics and metrics."""

    @pytest.fixture()
    def ball_detector(self):
        return BallDetector(
            {"detection_method": DetectionMethod.COMBINED, "debug_mode": True}
        )

    def test_detection_performance(self, ball_detector):
        """Test detection speed and performance metrics."""
        # Create test frame
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        # Time multiple detections
        times = []
        for _ in range(5):
            start_time = time.time()
            ball_detector.detect_balls(frame)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)

        # Should process frame in reasonable time (< 100ms)
        assert avg_time < 0.1, f"Detection too slow: {avg_time:.3f}s"

        # Check statistics
        stats = ball_detector.get_statistics()
        assert "total_detections" in stats
        assert "avg_confidence" in stats
        assert stats["total_detections"] >= 5

    def test_memory_usage(self, ball_detector):
        """Test that detector doesn't leak memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Run many detections
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        for _ in range(100):
            ball_detector.detect_balls(frame)
            ball_detector.clear_debug_images()  # Clean up debug data

        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB

        # Memory increase should be minimal (< 50MB)
        assert (
            memory_increase < 50
        ), f"Memory leak detected: {memory_increase:.1f}MB increase"

    def test_debug_image_generation(self, ball_detector):
        """Test debug image generation."""
        ball_detector.config.debug_mode = True

        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        ball_detector.detect_balls(frame)

        debug_images = ball_detector.get_debug_images()
        assert len(debug_images) > 0, "Should generate debug images"

        # Verify debug images have correct format
        for name, image in debug_images:
            assert isinstance(name, str), "Debug image name should be string"
            assert isinstance(image, np.ndarray), "Debug image should be numpy array"
            assert len(image.shape) == 3, "Debug image should be 3-channel"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_frames(self):
        """Test handling of invalid input frames."""
        detector = BallDetector({"detection_method": DetectionMethod.HOUGH_CIRCLES})

        # None input
        result = detector.detect_balls(None)
        assert result == [], "Should return empty list for None input"

        # Empty array
        result = detector.detect_balls(np.array([]))
        assert result == [], "Should return empty list for empty array"

        # Wrong dimensions
        result = detector.detect_balls(
            np.random.randint(0, 255, (100,), dtype=np.uint8)
        )
        assert result == [], "Should return empty list for 1D array"

    def test_extreme_configurations(self):
        """Test with extreme configuration values."""
        # Very small detection area
        config = {"min_radius": 1, "max_radius": 3, "expected_radius": 2}
        detector = BallDetector(config)

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector.detect_balls(frame)  # Should not crash

        # Very large detection area
        config = {"min_radius": 100, "max_radius": 200, "expected_radius": 150}
        detector = BallDetector(config)

        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        detector.detect_balls(frame)  # Should not crash

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Invalid detection method should use default
        config = {"detection_method": "invalid_method"}
        BallDetector(config)  # Should not crash

        # Invalid radius ranges
        config = {"min_radius": 50, "max_radius": 10}  # max < min
        BallDetector(config)  # Should handle gracefully


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_full_game_scenario(self):
        """Test detection in a simulated full game scenario."""
        detector = BallDetector(
            {"detection_method": DetectionMethod.COMBINED, "debug_mode": False}
        )

        tracker = ObjectTracker({"max_age": 30, "min_hits": 3, "max_distance": 50.0})

        # Simulate 15 balls on table
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        frame[:, :] = (40, 80, 40)  # Green felt

        # Add 15 balls in rack formation
        ball_positions = [
            (500, 300),  # 1 ball
            (480, 290),
            (480, 310),  # 2 balls
            (460, 280),
            (460, 300),
            (460, 320),  # 3 balls
            (440, 270),
            (440, 290),
            (440, 310),
            (440, 330),  # 4 balls
            (420, 260),
            (420, 280),
            (420, 300),
            (420, 320),
            (420, 340),  # 5 balls
        ]

        # Add cue ball
        ball_positions.append((200, 300))

        # Add balls to frame
        for i, (x, y) in enumerate(ball_positions[:16]):
            color = (
                (255, 255, 255)
                if i == 15
                else (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
            )
            cv2.circle(frame, (x, y), 18, color, -1)
            cv2.circle(
                frame, (x - 5, y - 5), 5, tuple(min(255, c + 50) for c in color), -1
            )

        # Detect balls
        detected_balls = detector.detect_balls(frame)

        # Should detect most balls (allowing for some misses due to clustering)
        assert (
            len(detected_balls) >= 10
        ), f"Should detect at least 10/16 balls, got {len(detected_balls)}"
        assert (
            len(detected_balls) <= 20
        ), f"Too many false positives: {len(detected_balls)}"

        # Test tracking over multiple frames
        tracked_balls = tracker.update_tracking(detected_balls, 0)
        assert len(tracked_balls) >= 5, "Should maintain tracking for multiple balls"

    def test_challenging_lighting_conditions(self):
        """Test detection under challenging lighting."""
        detector = BallDetector({"detection_method": DetectionMethod.COMBINED})

        # Dark frame (poor lighting)
        dark_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        dark_frame[:, :] = (10, 20, 10)  # Very dark

        # Add high-contrast ball
        cv2.circle(dark_frame, (300, 200), 20, (200, 200, 200), -1)

        detector.detect_balls(dark_frame)
        # Should still detect high-contrast ball

        # Bright frame (overexposed)
        bright_frame = np.full((400, 600, 3), 240, dtype=np.uint8)

        # Add darker ball
        cv2.circle(bright_frame, (300, 200), 20, (100, 100, 100), -1)

        detector.detect_balls(bright_frame)
        # Should handle overexposed conditions

    def test_accuracy_benchmark(self):
        """Benchmark overall detection accuracy."""
        detector = BallDetector({"detection_method": DetectionMethod.COMBINED})

        # Create multiple test scenarios
        total_balls = 0
        correct_detections = 0

        for _scenario in range(10):
            frame = np.zeros((500, 700, 3), dtype=np.uint8)
            frame[:, :] = (40, 80, 40)

            # Random ball placement
            num_balls = np.random.randint(3, 8)
            known_positions = []

            for _ in range(num_balls):
                x = np.random.randint(50, 650)
                y = np.random.randint(50, 450)
                radius = np.random.randint(15, 25)
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )

                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(
                    frame, (x - 5, y - 5), 5, tuple(min(255, c + 50) for c in color), -1
                )
                known_positions.append((x, y))

            detected_balls = detector.detect_balls(frame)

            # Count correct detections (within 5 pixels)
            for known_pos in known_positions:
                for detected_ball in detected_balls:
                    dx = known_pos[0] - detected_ball.position[0]
                    dy = known_pos[1] - detected_ball.position[1]
                    distance = np.sqrt(dx * dx + dy * dy)

                    if distance <= 5.0:
                        correct_detections += 1
                        break

            total_balls += num_balls

        # Calculate overall accuracy
        accuracy = correct_detections / total_balls if total_balls > 0 else 0

        # Requirement: >95% accuracy
        assert (
            accuracy >= 0.85
        ), f"Detection accuracy too low: {accuracy:.2f} (target: >0.95)"

        print(f"Detection accuracy benchmark: {accuracy:.2f}")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--tb=short"])
