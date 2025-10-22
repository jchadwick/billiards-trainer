"""Test vision integration 4K coordinate conversion.

This test verifies that the vision integration system correctly converts
vision detections from camera pixels to 4K canonical format using the
new resolution-based scaling approach.
"""

import pytest
from config import config
from core.validation.physics import PhysicsValidator
from integration_service_conversion_helpers import StateConversionHelpers

from backend.vision.models import Ball, BallType, CueStick


class TestVisionIntegration4KConversion:
    """Test suite for vision integration 4K coordinate conversion."""

    @pytest.fixture()
    def converter(self):
        """Create a state conversion helper instance."""
        return StateConversionHelpers(
            config=config, physics_validator=PhysicsValidator()
        )

    def test_ball_conversion_from_1080p_to_4k(self, converter):
        """Test ball position conversion from 1080p camera to 4K canonical."""
        # Create a ball detection at 1080p resolution
        ball_1080p = Ball(
            position=(960.0, 540.0),  # Center of 1920×1080 frame
            radius=18.0,  # Typical ball radius in pixels at 1080p
            ball_type=BallType.CUE,
            confidence=0.9,
            velocity=(100.0, 50.0),  # Pixels per second
            source_resolution=(1920, 1080),  # Camera resolution
            coordinate_space="pixel",
        )

        # Convert to BallState (4K canonical)
        ball_state = converter.vision_ball_to_ball_state(ball_1080p, is_target=False)

        # Verify conversion
        assert ball_state is not None, "Ball conversion should succeed"

        # Expected scale: 3840/1920 = 2.0, 2160/1080 = 2.0
        # Position should be scaled: (960*2, 540*2) = (1920, 1080)
        assert abs(ball_state.position.x - 1920.0) < 1.0, "X position should be doubled"
        assert abs(ball_state.position.y - 1080.0) < 1.0, "Y position should be doubled"

        # Velocity should also be scaled
        assert abs(ball_state.velocity.x - 200.0) < 1.0, "X velocity should be doubled"
        assert abs(ball_state.velocity.y - 100.0) < 1.0, "Y velocity should be doubled"

        # Radius should be scaled
        assert abs(ball_state.radius - 36.0) < 1.0, "Radius should be doubled"

        # Metadata should be preserved
        assert ball_state.is_cue_ball is True, "Ball type should be preserved"
        assert ball_state.confidence == 0.9, "Confidence should be preserved"

    def test_ball_conversion_from_720p_to_4k(self, converter):
        """Test ball position conversion from 720p camera to 4K canonical."""
        # Create a ball detection at 720p resolution
        ball_720p = Ball(
            position=(640.0, 360.0),  # Center of 1280×720 frame
            radius=12.0,  # Typical ball radius in pixels at 720p
            ball_type=BallType.EIGHT,
            confidence=0.85,
            velocity=(75.0, 25.0),  # Pixels per second
            source_resolution=(1280, 720),  # Camera resolution
            coordinate_space="pixel",
        )

        # Convert to BallState (4K canonical)
        ball_state = converter.vision_ball_to_ball_state(ball_720p, is_target=False)

        # Verify conversion
        assert ball_state is not None, "Ball conversion should succeed"

        # Expected scale: 3840/1280 = 3.0, 2160/720 = 3.0
        # Position should be scaled: (640*3, 360*3) = (1920, 1080)
        assert abs(ball_state.position.x - 1920.0) < 1.0, "X position should be tripled"
        assert abs(ball_state.position.y - 1080.0) < 1.0, "Y position should be tripled"

        # Velocity should also be scaled
        assert abs(ball_state.velocity.x - 225.0) < 1.0, "X velocity should be tripled"
        assert abs(ball_state.velocity.y - 75.0) < 1.0, "Y velocity should be tripled"

        # Radius should be scaled
        assert abs(ball_state.radius - 36.0) < 1.0, "Radius should be tripled"

    def test_cue_conversion_from_1080p_to_4k(self, converter):
        """Test cue stick conversion from 1080p camera to 4K canonical."""
        # Create a cue detection at 1080p resolution
        cue_1080p = CueStick(
            tip_position=(800.0, 500.0),
            angle=45.0,
            length=400.0,  # Pixels
            confidence=0.8,
            source_resolution=(1920, 1080),
            coordinate_space="pixel",
        )

        # Convert to CueState (4K canonical)
        cue_state = converter.vision_cue_to_cue_state(cue_1080p)

        # Verify conversion
        assert cue_state is not None, "Cue conversion should succeed"

        # Expected scale: 3840/1920 = 2.0, 2160/1080 = 2.0
        # Tip position should be scaled: (800*2, 500*2) = (1600, 1000)
        assert (
            abs(cue_state.tip_position.x - 1600.0) < 1.0
        ), "X tip position should be doubled"
        assert (
            abs(cue_state.tip_position.y - 1000.0) < 1.0
        ), "Y tip position should be doubled"

        # Length should be scaled
        assert abs(cue_state.length - 800.0) < 1.0, "Length should be doubled"

        # Angle should be preserved (not scaled)
        assert cue_state.angle == 45.0, "Angle should not be scaled"

        # Metadata should be preserved
        assert cue_state.confidence == 0.8, "Confidence should be preserved"
        assert cue_state.is_visible is True, "Visibility should be set"

    def test_conversion_stats(self, converter):
        """Test that conversion stats reflect the new coordinate system."""
        stats = converter.get_conversion_stats()

        # Verify new coordinate system is reported
        assert stats["coordinate_system"] == "4K_canonical", "Should report 4K system"
        assert (
            stats["resolution_based_conversion"] is True
        ), "Should use resolution conversion"

        # Stats should include conversion counts
        assert "ball_conversions" in stats
        assert "cue_conversions" in stats
        assert "coordinate_conversions" in stats

    def test_ball_position_clamping_out_of_bounds(self, converter):
        """Test that out-of-bounds positions are clamped to 4K frame."""
        # Create a ball detection with out-of-bounds position
        ball_oob = Ball(
            position=(5000.0, 3000.0),  # Way outside 1920×1080 frame
            radius=18.0,
            ball_type=BallType.OTHER,
            confidence=0.5,
            source_resolution=(1920, 1080),
            coordinate_space="pixel",
        )

        # Convert to BallState (should clamp to 4K bounds)
        ball_state = converter.vision_ball_to_ball_state(ball_oob, is_target=False)

        # Verify clamping
        assert ball_state is not None, "Conversion should still succeed"
        assert 0 <= ball_state.position.x <= 3840, "X should be clamped to 4K width"
        assert 0 <= ball_state.position.y <= 2160, "Y should be clamped to 4K height"

    def test_velocity_clamping_too_fast(self, converter):
        """Test that unreasonably fast velocities are clamped."""
        # Create a ball with extremely high velocity
        ball_fast = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            confidence=0.9,
            velocity=(50000.0, 50000.0),  # Unreasonably fast
            source_resolution=(1920, 1080),
            coordinate_space="pixel",
        )

        # Convert to BallState (should clamp velocity)
        ball_state = converter.vision_ball_to_ball_state(ball_fast, is_target=False)

        # Verify velocity was clamped
        assert ball_state is not None, "Conversion should still succeed"
        velocity_mag = (ball_state.velocity.x**2 + ball_state.velocity.y**2) ** 0.5
        max_velocity_4k = 12600.0  # Max reasonable velocity in 4K pixels/sec
        assert velocity_mag <= max_velocity_4k, "Velocity should be clamped"
