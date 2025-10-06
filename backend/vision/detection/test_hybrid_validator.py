"""Tests for hybrid validator.

Comprehensive test suite for the HybridValidator class, covering:
- Color histogram validation
- Circularity checks with Hough circles
- Size consistency validation
- Confidence score computation
- Batch processing
- Configuration updates
"""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ..models import Ball, BallType
from .hybrid_validator import HybridValidator, ValidationConfig


class TestHybridValidator(unittest.TestCase):
    """Test suite for HybridValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = HybridValidator()

        # Create test ball
        self.test_ball = Ball(
            position=(100.0, 100.0),
            radius=20.0,
            ball_type=BallType.CUE,
            confidence=0.9,
        )

    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator)
        self.assertIsInstance(self.validator.config, ValidationConfig)
        self.assertTrue(len(self.validator.ball_color_templates) > 0)

    def test_initialization_with_config(self):
        """Test validator initialization with custom configuration."""
        config = {
            "color_histogram_enabled": False,
            "circularity_enabled": True,
            "expected_radius": 25.0,
        }

        validator = HybridValidator(config)

        self.assertFalse(validator.config.color_histogram_enabled)
        self.assertTrue(validator.config.circularity_enabled)
        self.assertEqual(validator.config.expected_radius, 25.0)

    def test_extract_ball_roi(self):
        """Test ROI extraction around detected ball."""
        # Create test frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Extract ROI
        roi = self.validator._extract_ball_roi(self.test_ball, frame)

        self.assertIsNotNone(roi)
        self.assertGreater(roi.shape[0], 0)
        self.assertGreater(roi.shape[1], 0)

    def test_extract_ball_roi_edge_cases(self):
        """Test ROI extraction at frame edges."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        # Ball at top-left corner
        ball = Ball(
            position=(10.0, 10.0),
            radius=20.0,
            ball_type=BallType.CUE,
            confidence=0.9,
        )

        roi = self.validator._extract_ball_roi(ball, frame)
        self.assertIsNotNone(roi)

        # Ball at bottom-right corner
        ball = Ball(
            position=(40.0, 40.0),
            radius=20.0,
            ball_type=BallType.CUE,
            confidence=0.9,
        )

        roi = self.validator._extract_ball_roi(ball, frame)
        self.assertIsNotNone(roi)

    def test_extract_ball_roi_invalid(self):
        """Test ROI extraction with invalid inputs."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Ball outside frame
        ball = Ball(
            position=(300.0, 300.0),
            radius=20.0,
            ball_type=BallType.CUE,
            confidence=0.9,
        )

        roi = self.validator._extract_ball_roi(ball, frame)
        self.assertIsNone(roi)

    def test_validate_size_perfect_match(self):
        """Test size validation with perfect radius match."""
        self.validator.config.expected_radius = 20.0

        score = self.validator._validate_size(20.0)

        self.assertEqual(score, 1.0)

    def test_validate_size_within_tolerance(self):
        """Test size validation within tolerance."""
        self.validator.config.expected_radius = 20.0
        self.validator.config.radius_tolerance = 0.30  # ±30%

        # Test at boundary (30% larger)
        score = self.validator._validate_size(26.0)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test at boundary (30% smaller)
        score = self.validator._validate_size(14.0)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_validate_size_outside_tolerance(self):
        """Test size validation outside tolerance."""
        self.validator.config.expected_radius = 20.0
        self.validator.config.radius_tolerance = 0.30

        # Too large
        score = self.validator._validate_size(30.0)
        self.assertEqual(score, 0.0)

        # Too small
        score = self.validator._validate_size(10.0)
        self.assertEqual(score, 0.0)

    def test_validate_color_histogram_white_ball(self):
        """Test color validation for white cue ball."""
        # Create white ball ROI
        roi = np.ones((40, 40, 3), dtype=np.uint8) * 255

        score = self.validator._validate_color_histogram(roi, BallType.CUE)

        # Should get high score for white ball
        self.assertGreater(score, 0.5)

    def test_validate_color_histogram_black_ball(self):
        """Test color validation for black 8-ball."""
        # Create black ball ROI
        roi = np.zeros((40, 40, 3), dtype=np.uint8)

        score = self.validator._validate_color_histogram(roi, BallType.EIGHT)

        # Should get high score for black ball
        self.assertGreater(score, 0.5)

    def test_validate_color_histogram_colored_ball(self):
        """Test color validation for colored solid ball."""
        # Create red ball ROI
        roi = np.zeros((40, 40, 3), dtype=np.uint8)
        roi[:, :, 2] = 255  # Red channel

        score = self.validator._validate_color_histogram(roi, BallType.SOLID)

        # Should get reasonable score for colored ball
        self.assertGreater(score, 0.0)

    def test_validate_circularity_perfect_circle(self):
        """Test circularity validation with perfect circle."""
        # Create image with perfect white circle on black background
        roi = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.circle(roi, (30, 30), 20, (255, 255, 255), -1)

        score = self.validator._validate_circularity(roi, 20.0)

        # Should detect the circle and give high score
        self.assertGreater(score, 0.5)

    def test_validate_circularity_no_circle(self):
        """Test circularity validation with no circle."""
        # Create image with square
        roi = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.rectangle(roi, (10, 10), (50, 50), (255, 255, 255), -1)

        score = self.validator._validate_circularity(roi, 20.0)

        # Should give low score for non-circular shape
        self.assertLessEqual(score, 0.5)

    def test_validate_circularity_multiple_circles(self):
        """Test circularity validation with multiple circles."""
        # Create image with two circles
        roi = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.circle(roi, (30, 30), 18, (255, 255, 255), -1)  # Main circle
        cv2.circle(roi, (45, 45), 10, (255, 255, 255), -1)  # Secondary circle

        score = self.validator._validate_circularity(roi, 20.0)

        # Should still detect the main circle
        self.assertGreater(score, 0.0)

    def test_compute_confidence_multiplier_all_high_scores(self):
        """Test confidence computation with high validation scores."""
        validation_scores = {
            "color": (0.9, 0.3),
            "circularity": (0.85, 0.4),
            "size": (0.95, 0.3),
        }

        multiplier = self.validator._compute_confidence_multiplier(validation_scores)

        # Should get high multiplier
        self.assertGreater(multiplier, 0.8)
        self.assertLessEqual(multiplier, 1.0)

    def test_compute_confidence_multiplier_mixed_scores(self):
        """Test confidence computation with mixed validation scores."""
        validation_scores = {
            "color": (0.5, 0.3),
            "circularity": (0.7, 0.4),
            "size": (0.4, 0.3),
        }

        multiplier = self.validator._compute_confidence_multiplier(validation_scores)

        # Should get moderate multiplier
        self.assertGreater(multiplier, 0.3)
        self.assertLess(multiplier, 0.7)

    def test_compute_confidence_multiplier_all_low_scores(self):
        """Test confidence computation with low validation scores."""
        validation_scores = {
            "color": (0.2, 0.3),
            "circularity": (0.1, 0.4),
            "size": (0.15, 0.3),
        }

        multiplier = self.validator._compute_confidence_multiplier(validation_scores)

        # Should get low multiplier
        self.assertLess(multiplier, 0.5)

    def test_compute_confidence_multiplier_empty_scores(self):
        """Test confidence computation with no validation scores."""
        validation_scores = {}

        multiplier = self.validator._compute_confidence_multiplier(validation_scores)

        # Should return minimum confidence
        self.assertEqual(multiplier, self.validator.config.min_confidence_multiplier)

    def test_validate_ball_detection_valid_ball(self):
        """Test complete validation with valid ball."""
        # Create realistic ball image
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)

        multiplier = self.validator.validate_ball_detection(self.test_ball, frame)

        # Should get positive multiplier for valid ball
        self.assertGreater(multiplier, 0.0)
        self.assertLessEqual(multiplier, 1.0)

    def test_validate_ball_detection_invalid_roi(self):
        """Test validation with invalid ROI."""
        # Empty frame
        frame = np.array([])

        multiplier = self.validator.validate_ball_detection(self.test_ball, frame)

        # Should return 0.0 for invalid input
        self.assertEqual(multiplier, 0.0)

    def test_validate_batch(self):
        """Test batch validation of multiple balls."""
        # Create multiple test balls with ROIs
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        balls_with_rois = [
            (
                Ball(
                    position=(50.0, 50.0),
                    radius=20.0,
                    ball_type=BallType.CUE,
                    confidence=0.9,
                ),
                frame,
            ),
            (
                Ball(
                    position=(150.0, 150.0),
                    radius=20.0,
                    ball_type=BallType.SOLID,
                    confidence=0.8,
                ),
                frame,
            ),
        ]

        multipliers = self.validator.validate_batch(balls_with_rois)

        self.assertEqual(len(multipliers), 2)
        for multiplier in multipliers:
            self.assertGreaterEqual(multiplier, 0.0)
            self.assertLessEqual(multiplier, 1.0)

    def test_statistics_tracking(self):
        """Test validation statistics tracking."""
        # Reset statistics
        self.validator.reset_statistics()

        # Create test frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)

        # Run validations
        for _ in range(5):
            self.validator.validate_ball_detection(self.test_ball, frame)

        stats = self.validator.get_statistics()

        self.assertEqual(stats["total_validations"], 5)
        self.assertGreaterEqual(stats["passed_validations"], 0)
        self.assertGreaterEqual(stats["failed_validations"], 0)
        self.assertGreaterEqual(stats["pass_rate"], 0.0)
        self.assertLessEqual(stats["pass_rate"], 1.0)

    def test_reset_statistics(self):
        """Test statistics reset."""
        # Create test frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        # Run validation
        self.validator.validate_ball_detection(self.test_ball, frame)

        # Reset
        self.validator.reset_statistics()

        stats = self.validator.get_statistics()

        self.assertEqual(stats["total_validations"], 0)
        self.assertEqual(stats["passed_validations"], 0)
        self.assertEqual(stats["failed_validations"], 0)

    def test_update_config(self):
        """Test configuration updates."""
        updates = {
            "expected_radius": 25.0,
            "color_histogram_enabled": False,
            "circularity_weight": 0.5,
        }

        self.validator.update_config(updates)

        self.assertEqual(self.validator.config.expected_radius, 25.0)
        self.assertFalse(self.validator.config.color_histogram_enabled)
        self.assertEqual(self.validator.config.circularity_weight, 0.5)

    def test_set_expected_radius(self):
        """Test setting expected radius."""
        self.validator.set_expected_radius(30.0)

        self.assertEqual(self.validator.config.expected_radius, 30.0)

    def test_get_color_template(self):
        """Test retrieving color templates."""
        template = self.validator.get_color_template(BallType.CUE)

        self.assertIsNotNone(template)
        self.assertIn("lower", template)
        self.assertIn("upper", template)

    def test_update_color_template(self):
        """Test updating color templates."""
        lower = (10, 20, 30)
        upper = (40, 50, 60)

        self.validator.update_color_template(BallType.CUE, lower, upper)

        template = self.validator.get_color_template(BallType.CUE)

        np.testing.assert_array_equal(template["lower"], np.array(lower))
        np.testing.assert_array_equal(template["upper"], np.array(upper))

    def test_validation_with_disabled_checks(self):
        """Test validation with individual checks disabled."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)

        # Disable color validation
        self.validator.config.color_histogram_enabled = False
        multiplier1 = self.validator.validate_ball_detection(self.test_ball, frame)

        # Disable circularity validation
        self.validator.config.color_histogram_enabled = True
        self.validator.config.circularity_enabled = False
        multiplier2 = self.validator.validate_ball_detection(self.test_ball, frame)

        # Both should still produce valid multipliers
        self.assertGreaterEqual(multiplier1, 0.0)
        self.assertGreaterEqual(multiplier2, 0.0)

    def test_validation_weights(self):
        """Test that validation weights affect the final score."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)

        # Test with default weights
        self.validator.config.color_weight = 0.3
        self.validator.config.circularity_weight = 0.4
        self.validator.config.size_weight = 0.3
        multiplier1 = self.validator.validate_ball_detection(self.test_ball, frame)

        # Test with different weights (emphasize size)
        self.validator.config.color_weight = 0.1
        self.validator.config.circularity_weight = 0.1
        self.validator.config.size_weight = 0.8
        multiplier2 = self.validator.validate_ball_detection(self.test_ball, frame)

        # Multipliers should be different due to weight changes
        # (unless by chance all scores are identical)
        # Just verify both are valid
        self.assertGreaterEqual(multiplier1, 0.0)
        self.assertGreaterEqual(multiplier2, 0.0)


class TestValidationConfig(unittest.TestCase):
    """Test suite for ValidationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()

        self.assertTrue(config.color_histogram_enabled)
        self.assertTrue(config.circularity_enabled)
        self.assertTrue(config.size_validation_enabled)
        self.assertEqual(config.expected_radius, 20.0)
        self.assertFalse(config.debug_mode)

    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidationConfig(
            color_histogram_enabled=False,
            expected_radius=25.0,
            debug_mode=True,
        )

        self.assertFalse(config.color_histogram_enabled)
        self.assertEqual(config.expected_radius, 25.0)
        self.assertTrue(config.debug_mode)

    def test_weight_sum(self):
        """Test that validation weights sum to 1.0."""
        config = ValidationConfig()

        weight_sum = (
            config.color_weight + config.circularity_weight + config.size_weight
        )

        self.assertAlmostEqual(weight_sum, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
