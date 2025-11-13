"""Tests for the unified Ball class.

This test suite validates:
- Ball creation and validation
- Coordinate conversions (pixel to 4K)
- BallState conversion
- Edge cases and error handling
- Coordinate metadata to prevent double-scaling bugs
"""

import numpy as np
import pytest

from backend.common.models import Ball
from backend.core.coordinates import Vector2D
from backend.core.models import BallState
from backend.vision.models import BallType


class TestBallCreation:
    """Test Ball instantiation and validation."""

    def test_minimal_ball_creation(self):
        """Test creating a ball with minimal required fields."""
        ball = Ball(
            position=(100.0, 200.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        assert ball.position == (100.0, 200.0)
        assert ball.radius == 36.0
        assert ball.ball_type == BallType.CUE
        assert ball.number is None  # Cue ball has no number
        assert ball.confidence == 0.0  # Default
        assert ball.coordinate_space == "4k_canonical"  # Default

    def test_full_ball_creation(self):
        """Test creating a ball with all fields specified."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_1,
            number=1,
            confidence=0.95,
            velocity=(50.0, 25.0),
            acceleration=(1.0, 2.0),
            is_moving=True,
            track_id=42,
            last_seen=1234567890.0,
            age=10,
            hit_count=2,
            color_hsv=(30, 255, 255),
            occlusion_state=0.1,
            predicted_position=(1925.0, 1082.5),
            position_history=[(1900.0, 1070.0), (1910.0, 1075.0)],
            coordinate_space="4k_canonical",
            source_resolution=None,
        )
        assert ball.position == (1920.0, 1080.0)
        assert ball.number == 1
        assert ball.confidence == 0.95
        assert ball.velocity == (50.0, 25.0)
        assert ball.is_moving is True
        assert ball.track_id == 42
        assert ball.color_hsv == (30, 255, 255)
        assert len(ball.position_history) == 2

    def test_number_auto_population_from_ball_type(self):
        """Test that number is automatically populated from ball_type."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.SOLID_5,
        )
        assert ball.number == 5

        ball_eight = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.EIGHT,
        )
        assert ball_eight.number == 8

        ball_stripe = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.STRIPE_15,
        )
        assert ball_stripe.number == 15

    def test_cue_ball_has_no_number(self):
        """Test that cue ball number is None."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        assert ball.number is None

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            Ball(
                position=(100.0, 100.0),
                radius=36.0,
                ball_type=BallType.CUE,
                confidence=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValueError, match="Confidence must be in"):
            Ball(
                position=(100.0, 100.0),
                radius=36.0,
                ball_type=BallType.CUE,
                confidence=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_occlusion_state_raises_error(self):
        """Test that invalid occlusion state values raise ValueError."""
        with pytest.raises(ValueError, match="Occlusion state must be in"):
            Ball(
                position=(100.0, 100.0),
                radius=36.0,
                ball_type=BallType.CUE,
                occlusion_state=1.5,  # Invalid
            )

    def test_invalid_radius_raises_error(self):
        """Test that non-positive radius raises ValueError."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            Ball(
                position=(100.0, 100.0),
                radius=0.0,  # Invalid
                ball_type=BallType.CUE,
            )

        with pytest.raises(ValueError, match="Radius must be positive"):
            Ball(
                position=(100.0, 100.0),
                radius=-10.0,  # Invalid
                ball_type=BallType.CUE,
            )

    def test_invalid_coordinate_space_raises_error(self):
        """Test that invalid coordinate space raises ValueError."""
        with pytest.raises(ValueError, match="Coordinate space must be one of"):
            Ball(
                position=(100.0, 100.0),
                radius=36.0,
                ball_type=BallType.CUE,
                coordinate_space="invalid_space",
            )

    def test_pixel_space_requires_source_resolution(self):
        """Test that pixel coordinate space requires source_resolution."""
        with pytest.raises(
            ValueError,
            match="source_resolution is required when coordinate_space is 'pixel'",
        ):
            Ball(
                position=(100.0, 100.0),
                radius=36.0,
                ball_type=BallType.CUE,
                coordinate_space="pixel",
                source_resolution=None,  # Invalid
            )


class TestCoordinateConversion:
    """Test coordinate conversion methods."""

    def test_to_4k_canonical_from_4k_is_idempotent(self):
        """Test that converting from 4K to 4K returns identical ball."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_1,
            velocity=(100.0, 50.0),
            coordinate_space="4k_canonical",
        )
        ball_4k = ball.to_4k_canonical()

        assert ball_4k.position == (1920.0, 1080.0)
        assert ball_4k.radius == 36.0
        assert ball_4k.velocity == (100.0, 50.0)
        assert ball_4k.coordinate_space == "4k_canonical"
        assert ball_4k.source_resolution is None

    def test_to_4k_canonical_from_1080p(self):
        """Test converting from 1080p to 4K canonical coordinates."""
        # Ball at center of 1080p frame
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.SOLID_1,
            velocity=(10.0, 5.0),
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()

        # Should scale by 2x in both dimensions
        assert ball_4k.position == (1920.0, 1080.0)
        assert ball_4k.radius == 36.0
        assert ball_4k.velocity == (20.0, 10.0)
        assert ball_4k.coordinate_space == "4k_canonical"
        assert ball_4k.source_resolution is None

    def test_to_4k_canonical_from_720p(self):
        """Test converting from 720p to 4K canonical coordinates."""
        # Ball at center of 720p frame (1280x720)
        # Scale factors: 3840/1280 = 3.0, 2160/720 = 3.0
        ball = Ball(
            position=(640.0, 360.0),
            radius=12.0,
            ball_type=BallType.SOLID_2,
            velocity=(15.0, 10.0),
            coordinate_space="pixel",
            source_resolution=(1280, 720),
        )
        ball_4k = ball.to_4k_canonical()

        # Should scale by 3x in both dimensions
        assert ball_4k.position == (1920.0, 1080.0)
        assert ball_4k.radius == 36.0
        assert ball_4k.velocity == (45.0, 30.0)
        assert ball_4k.coordinate_space == "4k_canonical"

    def test_to_4k_canonical_scales_acceleration(self):
        """Test that acceleration is properly scaled during conversion."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            acceleration=(2.0, 1.0),
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()

        assert ball_4k.acceleration == (4.0, 2.0)

    def test_to_4k_canonical_scales_predicted_position(self):
        """Test that predicted_position is scaled during conversion."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            predicted_position=(965.0, 542.5),
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()

        assert ball_4k.predicted_position == (1930.0, 1085.0)

    def test_to_4k_canonical_scales_position_history(self):
        """Test that position_history is scaled during conversion."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            position_history=[(950.0, 535.0), (955.0, 537.5)],
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()

        assert ball_4k.position_history == [(1900.0, 1070.0), (1910.0, 1075.0)]

    def test_to_4k_canonical_preserves_non_spatial_fields(self):
        """Test that non-spatial fields are preserved during conversion."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.SOLID_3,
            number=3,
            confidence=0.9,
            is_moving=True,
            track_id=7,
            last_seen=1234567890.0,
            age=15,
            hit_count=3,
            color_hsv=(0, 255, 255),
            occlusion_state=0.2,
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()

        # Non-spatial fields should be preserved
        assert ball_4k.ball_type == BallType.SOLID_3
        assert ball_4k.number == 3
        assert ball_4k.confidence == 0.9
        assert ball_4k.is_moving is True
        assert ball_4k.track_id == 7
        assert ball_4k.last_seen == 1234567890.0
        assert ball_4k.age == 15
        assert ball_4k.hit_count == 3
        assert ball_4k.color_hsv == (0, 255, 255)
        assert ball_4k.occlusion_state == 0.2

    def test_to_4k_canonical_without_source_resolution_raises_error(self):
        """Test that conversion from pixel space without source_resolution fails."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        # Manually break the source_resolution to test error handling
        ball.source_resolution = None

        with pytest.raises(ValueError, match="source_resolution required"):
            ball.to_4k_canonical()

    def test_to_4k_canonical_from_world_raises_error(self):
        """Test that conversion from world coordinates raises error."""
        ball = Ball(
            position=(1.0, 2.0),
            radius=0.03,  # meters
            ball_type=BallType.CUE,
            coordinate_space="world",
        )

        with pytest.raises(ValueError, match="Cannot convert from coordinate_space"):
            ball.to_4k_canonical()


class TestWorldCoordinateConversion:
    """Test conversion to world coordinates using homography."""

    def test_to_world_coordinates_with_identity_homography(self):
        """Test world coordinate conversion with identity matrix."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )

        # Identity matrix should return same coordinates
        H = np.eye(3, dtype=np.float64)
        world_pos = ball.to_world_coordinates(H)

        assert world_pos is not None
        assert abs(world_pos[0] - 1920.0) < 1e-6
        assert abs(world_pos[1] - 1080.0) < 1e-6

    def test_to_world_coordinates_without_homography_returns_none(self):
        """Test that conversion without homography returns None."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )

        world_pos = ball.to_world_coordinates(None)
        assert world_pos is None

    def test_to_world_coordinates_from_world_space_returns_none(self):
        """Test that conversion from world space returns None."""
        ball = Ball(
            position=(1.0, 2.0),
            radius=0.03,
            ball_type=BallType.CUE,
            coordinate_space="world",
        )

        H = np.eye(3, dtype=np.float64)
        world_pos = ball.to_world_coordinates(H)
        assert world_pos is None


class TestBallStateConversion:
    """Test conversion to core BallState."""

    def test_to_ball_state_basic(self):
        """Test basic conversion to BallState."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_1,
            velocity=(100.0, 50.0),
            confidence=0.95,
        )
        ball_state = ball.to_ball_state()

        assert isinstance(ball_state, BallState)
        assert ball_state.id == "ball_1"
        assert ball_state.position.x == 1920.0
        assert ball_state.position.y == 1080.0
        assert ball_state.velocity.x == 100.0
        assert ball_state.velocity.y == 50.0
        assert ball_state.radius == 36.0
        assert ball_state.number == 1
        assert ball_state.confidence == 0.95
        assert ball_state.is_cue_ball is False

    def test_to_ball_state_cue_ball(self):
        """Test conversion of cue ball to BallState."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        ball_state = ball.to_ball_state()

        assert ball_state.id == "cue_ball"
        assert ball_state.is_cue_ball is True
        assert ball_state.number is None

    def test_to_ball_state_with_custom_id(self):
        """Test conversion with custom ball ID."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_5,
        )
        ball_state = ball.to_ball_state(ball_id="custom_ball_5")

        assert ball_state.id == "custom_ball_5"

    def test_to_ball_state_with_custom_mass(self):
        """Test conversion with custom mass."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_1,
        )
        ball_state = ball.to_ball_state(mass=0.2)

        assert ball_state.mass == 0.2

    def test_to_ball_state_with_spin(self):
        """Test conversion with custom spin."""
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        spin = Vector2D.from_4k(10.0, 5.0)
        ball_state = ball.to_ball_state(spin=spin)

        assert ball_state.spin.x == 10.0
        assert ball_state.spin.y == 5.0

    def test_to_ball_state_converts_from_pixel_to_4k(self):
        """Test that conversion automatically scales from pixel to 4K."""
        # Ball at 1080p resolution
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.SOLID_2,
            velocity=(50.0, 25.0),
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_state = ball.to_ball_state()

        # Should be scaled to 4K
        assert ball_state.position.x == 1920.0
        assert ball_state.position.y == 1080.0
        assert ball_state.velocity.x == 100.0
        assert ball_state.velocity.y == 50.0
        assert ball_state.radius == 36.0

    def test_to_ball_state_preserves_last_seen(self):
        """Test that last_seen is preserved as last_update."""
        timestamp = 1234567890.0
        ball = Ball(
            position=(1920.0, 1080.0),
            radius=36.0,
            ball_type=BallType.SOLID_1,
            last_seen=timestamp,
        )
        ball_state = ball.to_ball_state()

        assert ball_state.last_update == timestamp


class TestPositionHistory:
    """Test position history management."""

    def test_update_history_appends_position(self):
        """Test that update_history adds current position."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        assert len(ball.position_history) == 0

        ball.update_history()
        assert len(ball.position_history) == 1
        assert ball.position_history[0] == (100.0, 100.0)

    def test_update_history_respects_max_length(self):
        """Test that history is capped at max_history."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )

        # Add 15 positions with max_history=10
        for i in range(15):
            ball.position = (100.0 + i, 100.0 + i)
            ball.update_history(max_history=10)

        assert len(ball.position_history) == 10
        # Should have positions 5-14 (oldest 5 removed)
        assert ball.position_history[0] == (105.0, 105.0)
        assert ball.position_history[-1] == (114.0, 114.0)


class TestDisplayName:
    """Test display_name property."""

    def test_display_name_numbered_ball(self):
        """Test display name for numbered balls."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.SOLID_5,
        )
        assert ball.display_name == "5"

    def test_display_name_cue_ball(self):
        """Test display name for cue ball."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        assert ball.display_name == "Cue Ball"

    def test_display_name_eight_ball(self):
        """Test display name for eight ball."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.EIGHT,
        )
        assert ball.display_name == "8"

    def test_display_name_unknown_ball(self):
        """Test display name for unknown ball type."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.UNKNOWN,
        )
        assert ball.display_name == "Unknown"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_velocity(self):
        """Test ball with zero velocity."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
            velocity=(0.0, 0.0),
        )
        assert ball.velocity == (0.0, 0.0)

    def test_negative_velocity(self):
        """Test ball with negative velocity components."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
            velocity=(-50.0, -25.0),
        )
        assert ball.velocity == (-50.0, -25.0)

    def test_very_small_radius(self):
        """Test ball with very small radius."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=0.001,
            ball_type=BallType.CUE,
        )
        assert ball.radius == 0.001

    def test_very_large_position(self):
        """Test ball with very large position values."""
        ball = Ball(
            position=(10000.0, 10000.0),
            radius=36.0,
            ball_type=BallType.CUE,
        )
        assert ball.position == (10000.0, 10000.0)

    def test_empty_position_history(self):
        """Test ball with empty position history."""
        ball = Ball(
            position=(100.0, 100.0),
            radius=36.0,
            ball_type=BallType.CUE,
            position_history=[],
        )
        ball_4k = ball.to_4k_canonical()
        assert ball_4k.position_history == []

    def test_none_predicted_position(self):
        """Test ball with None predicted_position."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            predicted_position=None,
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )
        ball_4k = ball.to_4k_canonical()
        assert ball_4k.predicted_position is None


class TestCoordinateMetadataPreventDobleScaling:
    """Test that coordinate metadata prevents double-scaling bugs."""

    def test_double_conversion_detection(self):
        """Test that converting twice doesn't double-scale."""
        # Start with 1080p ball
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )

        # Convert to 4K once
        ball_4k = ball.to_4k_canonical()
        assert ball_4k.position == (1920.0, 1080.0)
        assert ball_4k.coordinate_space == "4k_canonical"

        # Convert to 4K again (should be no-op)
        ball_4k_again = ball_4k.to_4k_canonical()
        assert ball_4k_again.position == (1920.0, 1080.0)
        assert ball_4k_again.coordinate_space == "4k_canonical"

        # Positions should be identical (no double scaling)
        assert ball_4k_again.position == ball_4k.position

    def test_coordinate_space_tracking_through_conversion(self):
        """Test that coordinate_space is correctly updated through conversions."""
        ball = Ball(
            position=(960.0, 540.0),
            radius=18.0,
            ball_type=BallType.CUE,
            coordinate_space="pixel",
            source_resolution=(1920, 1080),
        )

        # Original is in pixel space
        assert ball.coordinate_space == "pixel"
        assert ball.source_resolution == (1920, 1080)

        # After conversion, should be in 4K canonical
        ball_4k = ball.to_4k_canonical()
        assert ball_4k.coordinate_space == "4k_canonical"
        assert ball_4k.source_resolution is None  # No longer needed

    def test_source_resolution_cleared_after_4k_conversion(self):
        """Test that source_resolution is cleared after converting to 4K."""
        ball = Ball(
            position=(640.0, 360.0),
            radius=12.0,
            ball_type=BallType.CUE,
            coordinate_space="pixel",
            source_resolution=(1280, 720),
        )

        ball_4k = ball.to_4k_canonical()

        # Source resolution should be cleared for 4K canonical
        assert ball_4k.source_resolution is None
        # This prevents accidental re-scaling
        assert ball_4k.coordinate_space == "4k_canonical"
