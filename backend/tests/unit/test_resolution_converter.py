"""Comprehensive unit tests for ResolutionConverter.

This test suite provides full coverage of the ResolutionConverter class,
including all scaling methods, round-trip accuracy, edge cases, and
convenience functions.

Test Coverage:
- calculate_scale_to_4k: Scale factor calculations from various resolutions to 4K
- calculate_scale_from_4k: Scale factor calculations from 4K to various resolutions
- scale_to_4k: Coordinate conversion from various resolutions to 4K
- scale_from_4k: Coordinate conversion from 4K to various resolutions
- scale_between_resolutions: Direct conversion between arbitrary resolutions
- scale_distance_to_4k: Distance scaling from various resolutions to 4K
- scale_distance_from_4k: Distance scaling from 4K to various resolutions
- is_4k_canonical: Resolution validation
- get_aspect_ratio: Aspect ratio calculations
- Convenience functions: to_4k and from_4k
- Round-trip accuracy: Verify no loss when converting back and forth
- Edge cases: Zero coordinates, negative coordinates, fractional values
"""

import math

import pytest

from backend.core.constants_4k import CANONICAL_RESOLUTION
from backend.core.resolution_converter import ResolutionConverter, from_4k, to_4k

# ============================================================================
# Test Data - Common Resolutions
# ============================================================================

# Standard video resolutions for testing
RESOLUTION_1080P = (1920, 1080)
RESOLUTION_720P = (1280, 720)
RESOLUTION_4K = (3840, 2160)
RESOLUTION_480P = (854, 480)
RESOLUTION_1440P = (2560, 1440)
RESOLUTION_8K = (7680, 4320)
RESOLUTION_CUSTOM = (1600, 900)


# ============================================================================
# Tests: calculate_scale_to_4k
# ============================================================================


class TestCalculateScaleTo4K:
    """Tests for calculate_scale_to_4k method."""

    def test_scale_from_1080p_to_4k(self):
        """Test scale calculation from 1080p to 4K."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_1080P)
        assert scale_x == 2.0
        assert scale_y == 2.0

    def test_scale_from_720p_to_4k(self):
        """Test scale calculation from 720p to 4K."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_720P)
        assert scale_x == 3.0
        assert scale_y == 3.0

    def test_scale_from_4k_to_4k(self):
        """Test scale calculation from 4K to 4K (identity)."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_4K)
        assert scale_x == 1.0
        assert scale_y == 1.0

    def test_scale_from_480p_to_4k(self):
        """Test scale calculation from 480p to 4K."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_480P)
        assert scale_x == pytest.approx(3840 / 854, rel=1e-6)
        assert scale_y == pytest.approx(2160 / 480, rel=1e-6)

    def test_scale_from_1440p_to_4k(self):
        """Test scale calculation from 1440p to 4K."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_1440P)
        assert scale_x == 1.5
        assert scale_y == 1.5

    def test_scale_from_custom_resolution_to_4k(self):
        """Test scale calculation from custom resolution to 4K."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_CUSTOM)
        assert scale_x == pytest.approx(3840 / 1600, rel=1e-6)
        assert scale_y == pytest.approx(2160 / 900, rel=1e-6)

    def test_scale_returns_tuple(self):
        """Test that scale calculation returns a tuple."""
        result = ResolutionConverter.calculate_scale_to_4k(RESOLUTION_1080P)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Tests: calculate_scale_from_4k
# ============================================================================


class TestCalculateScaleFrom4K:
    """Tests for calculate_scale_from_4k method."""

    def test_scale_from_4k_to_1080p(self):
        """Test scale calculation from 4K to 1080p."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(RESOLUTION_1080P)
        assert scale_x == 0.5
        assert scale_y == 0.5

    def test_scale_from_4k_to_720p(self):
        """Test scale calculation from 4K to 720p."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(RESOLUTION_720P)
        assert scale_x == pytest.approx(1280 / 3840, rel=1e-6)
        assert scale_y == pytest.approx(720 / 2160, rel=1e-6)

    def test_scale_from_4k_to_4k(self):
        """Test scale calculation from 4K to 4K (identity)."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(RESOLUTION_4K)
        assert scale_x == 1.0
        assert scale_y == 1.0

    def test_scale_from_4k_to_1440p(self):
        """Test scale calculation from 4K to 1440p."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(RESOLUTION_1440P)
        assert scale_x == pytest.approx(2560 / 3840, rel=1e-6)
        assert scale_y == pytest.approx(1440 / 2160, rel=1e-6)

    def test_scale_from_4k_to_custom_resolution(self):
        """Test scale calculation from 4K to custom resolution."""
        scale_x, scale_y = ResolutionConverter.calculate_scale_from_4k(
            RESOLUTION_CUSTOM
        )
        assert scale_x == pytest.approx(1600 / 3840, rel=1e-6)
        assert scale_y == pytest.approx(900 / 2160, rel=1e-6)

    def test_scale_inverse_relationship(self):
        """Test that to_4k and from_4k scales are inverses."""
        to_scale_x, to_scale_y = ResolutionConverter.calculate_scale_to_4k(
            RESOLUTION_1080P
        )
        from_scale_x, from_scale_y = ResolutionConverter.calculate_scale_from_4k(
            RESOLUTION_1080P
        )

        assert to_scale_x == pytest.approx(1.0 / from_scale_x, rel=1e-6)
        assert to_scale_y == pytest.approx(1.0 / from_scale_y, rel=1e-6)

    def test_scale_returns_tuple(self):
        """Test that scale calculation returns a tuple."""
        result = ResolutionConverter.calculate_scale_from_4k(RESOLUTION_1080P)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Tests: scale_to_4k
# ============================================================================


class TestScaleTo4K:
    """Tests for scale_to_4k method."""

    def test_scale_center_1080p_to_4k(self):
        """Test scaling center point from 1080p to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(960, 540, RESOLUTION_1080P)
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_scale_origin_1080p_to_4k(self):
        """Test scaling origin from 1080p to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(0, 0, RESOLUTION_1080P)
        assert x_4k == 0.0
        assert y_4k == 0.0

    def test_scale_max_1080p_to_4k(self):
        """Test scaling max coordinates from 1080p to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(1920, 1080, RESOLUTION_1080P)
        assert x_4k == 3840.0
        assert y_4k == 2160.0

    def test_scale_center_720p_to_4k(self):
        """Test scaling center point from 720p to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(640, 360, RESOLUTION_720P)
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_scale_arbitrary_point_to_4k(self):
        """Test scaling arbitrary point from 1080p to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(100, 200, RESOLUTION_1080P)
        assert x_4k == 200.0
        assert y_4k == 400.0

    def test_scale_fractional_coordinates_to_4k(self):
        """Test scaling fractional coordinates to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(100.5, 200.75, RESOLUTION_1080P)
        assert x_4k == pytest.approx(201.0, rel=1e-6)
        assert y_4k == pytest.approx(401.5, rel=1e-6)

    def test_scale_negative_coordinates_to_4k(self):
        """Test scaling negative coordinates to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(-100, -200, RESOLUTION_1080P)
        assert x_4k == -200.0
        assert y_4k == -400.0

    def test_scale_4k_to_4k_identity(self):
        """Test that scaling 4K to 4K is identity operation."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(1920, 1080, RESOLUTION_4K)
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_scale_returns_tuple(self):
        """Test that coordinate scaling returns a tuple."""
        result = ResolutionConverter.scale_to_4k(100, 200, RESOLUTION_1080P)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Tests: scale_from_4k
# ============================================================================


class TestScaleFrom4K:
    """Tests for scale_from_4k method."""

    def test_scale_center_4k_to_1080p(self):
        """Test scaling center point from 4K to 1080p."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, RESOLUTION_1080P)
        assert x == 960.0
        assert y == 540.0

    def test_scale_origin_4k_to_1080p(self):
        """Test scaling origin from 4K to 1080p."""
        x, y = ResolutionConverter.scale_from_4k(0, 0, RESOLUTION_1080P)
        assert x == 0.0
        assert y == 0.0

    def test_scale_max_4k_to_1080p(self):
        """Test scaling max coordinates from 4K to 1080p."""
        x, y = ResolutionConverter.scale_from_4k(3840, 2160, RESOLUTION_1080P)
        assert x == 1920.0
        assert y == 1080.0

    def test_scale_center_4k_to_720p(self):
        """Test scaling center point from 4K to 720p."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, RESOLUTION_720P)
        assert x == pytest.approx(640.0, rel=1e-6)
        assert y == pytest.approx(360.0, rel=1e-6)

    def test_scale_arbitrary_point_from_4k(self):
        """Test scaling arbitrary point from 4K to 1080p."""
        x, y = ResolutionConverter.scale_from_4k(200, 400, RESOLUTION_1080P)
        assert x == 100.0
        assert y == 200.0

    def test_scale_fractional_coordinates_from_4k(self):
        """Test scaling fractional coordinates from 4K."""
        x, y = ResolutionConverter.scale_from_4k(201.0, 401.5, RESOLUTION_1080P)
        assert x == pytest.approx(100.5, rel=1e-6)
        assert y == pytest.approx(200.75, rel=1e-6)

    def test_scale_negative_coordinates_from_4k(self):
        """Test scaling negative coordinates from 4K."""
        x, y = ResolutionConverter.scale_from_4k(-200, -400, RESOLUTION_1080P)
        assert x == -100.0
        assert y == -200.0

    def test_scale_4k_to_4k_identity(self):
        """Test that scaling 4K to 4K is identity operation."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, RESOLUTION_4K)
        assert x == 1920.0
        assert y == 1080.0

    def test_scale_returns_tuple(self):
        """Test that coordinate scaling returns a tuple."""
        result = ResolutionConverter.scale_from_4k(100, 200, RESOLUTION_1080P)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ============================================================================
# Tests: Round-trip Accuracy
# ============================================================================


class TestRoundTripAccuracy:
    """Tests for round-trip conversion accuracy."""

    def test_roundtrip_1080p_center(self):
        """Test round-trip accuracy for 1080p center point."""
        original_x, original_y = 960, 540
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_1080P
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_1080P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)

    def test_roundtrip_720p_arbitrary(self):
        """Test round-trip accuracy for 720p arbitrary point."""
        original_x, original_y = 123.456, 789.012
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_720P
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_720P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)

    def test_roundtrip_1440p_fractional(self):
        """Test round-trip accuracy for 1440p fractional coordinates."""
        original_x, original_y = 1234.5678, 987.6543
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_1440P
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_1440P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)

    def test_roundtrip_custom_resolution(self):
        """Test round-trip accuracy for custom resolution."""
        original_x, original_y = 555.5, 333.3
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_CUSTOM
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_CUSTOM)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)

    def test_roundtrip_zero_coordinates(self):
        """Test round-trip accuracy for zero coordinates."""
        original_x, original_y = 0, 0
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_1080P
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_1080P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)

    def test_roundtrip_negative_coordinates(self):
        """Test round-trip accuracy for negative coordinates."""
        original_x, original_y = -100.5, -200.75
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, RESOLUTION_1080P
        )
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_1080P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)


# ============================================================================
# Tests: scale_between_resolutions
# ============================================================================


class TestScaleBetweenResolutions:
    """Tests for scale_between_resolutions method."""

    def test_scale_720p_to_1080p(self):
        """Test scaling from 720p to 1080p."""
        x, y = ResolutionConverter.scale_between_resolutions(
            640, 360, RESOLUTION_720P, RESOLUTION_1080P
        )
        assert x == 960.0
        assert y == 540.0

    def test_scale_1080p_to_720p(self):
        """Test scaling from 1080p to 720p."""
        x, y = ResolutionConverter.scale_between_resolutions(
            960, 540, RESOLUTION_1080P, RESOLUTION_720P
        )
        assert x == pytest.approx(640.0, rel=1e-6)
        assert y == pytest.approx(360.0, rel=1e-6)

    def test_scale_480p_to_1440p(self):
        """Test scaling from 480p to 1440p."""
        x, y = ResolutionConverter.scale_between_resolutions(
            427, 240, RESOLUTION_480P, RESOLUTION_1440P
        )
        expected_x = 427 * (3840 / 854) * (2560 / 3840)
        expected_y = 240 * (2160 / 480) * (1440 / 2160)
        assert x == pytest.approx(expected_x, rel=1e-6)
        assert y == pytest.approx(expected_y, rel=1e-6)

    def test_scale_same_resolution(self):
        """Test scaling from resolution to itself (identity)."""
        x, y = ResolutionConverter.scale_between_resolutions(
            100, 200, RESOLUTION_1080P, RESOLUTION_1080P
        )
        assert x == pytest.approx(100.0, rel=1e-6)
        assert y == pytest.approx(200.0, rel=1e-6)

    def test_scale_custom_to_custom(self):
        """Test scaling between custom resolutions."""
        source = (1024, 768)
        target = (2048, 1536)
        x, y = ResolutionConverter.scale_between_resolutions(512, 384, source, target)
        assert x == pytest.approx(1024.0, rel=1e-6)
        assert y == pytest.approx(768.0, rel=1e-6)

    def test_scale_fractional_between_resolutions(self):
        """Test scaling fractional coordinates between resolutions."""
        x, y = ResolutionConverter.scale_between_resolutions(
            123.456, 234.567, RESOLUTION_720P, RESOLUTION_1080P
        )
        x_4k, y_4k = ResolutionConverter.scale_to_4k(123.456, 234.567, RESOLUTION_720P)
        expected_x, expected_y = ResolutionConverter.scale_from_4k(
            x_4k, y_4k, RESOLUTION_1080P
        )
        assert x == pytest.approx(expected_x, rel=1e-6)
        assert y == pytest.approx(expected_y, rel=1e-6)


# ============================================================================
# Tests: scale_distance_to_4k
# ============================================================================


class TestScaleDistanceTo4K:
    """Tests for scale_distance_to_4k method."""

    def test_scale_ball_radius_1080p_to_4k(self):
        """Test scaling ball radius from 1080p to 4K."""
        # Ball radius is 36 pixels in 4K, so 18 pixels in 1080p
        distance_4k = ResolutionConverter.scale_distance_to_4k(18, RESOLUTION_1080P)
        assert distance_4k == 36.0

    def test_scale_zero_distance_to_4k(self):
        """Test scaling zero distance to 4K."""
        distance_4k = ResolutionConverter.scale_distance_to_4k(0, RESOLUTION_1080P)
        assert distance_4k == 0.0

    def test_scale_fractional_distance_to_4k(self):
        """Test scaling fractional distance to 4K."""
        distance_4k = ResolutionConverter.scale_distance_to_4k(10.5, RESOLUTION_1080P)
        assert distance_4k == 21.0

    def test_scale_large_distance_to_4k(self):
        """Test scaling large distance from 720p to 4K."""
        distance_4k = ResolutionConverter.scale_distance_to_4k(100, RESOLUTION_720P)
        assert distance_4k == 300.0

    def test_scale_distance_uses_x_scale(self):
        """Test that distance scaling uses X-axis scale factor."""
        # Custom resolution with different X/Y scales
        custom_res = (1920, 1000)
        distance_4k = ResolutionConverter.scale_distance_to_4k(10, custom_res)
        scale_x, _ = ResolutionConverter.calculate_scale_to_4k(custom_res)
        assert distance_4k == pytest.approx(10 * scale_x, rel=1e-6)

    def test_scale_distance_4k_to_4k(self):
        """Test scaling distance from 4K to 4K (identity)."""
        distance_4k = ResolutionConverter.scale_distance_to_4k(50, RESOLUTION_4K)
        assert distance_4k == 50.0


# ============================================================================
# Tests: scale_distance_from_4k
# ============================================================================


class TestScaleDistanceFrom4K:
    """Tests for scale_distance_from_4k method."""

    def test_scale_ball_radius_4k_to_1080p(self):
        """Test scaling ball radius from 4K to 1080p."""
        # Ball radius is 36 pixels in 4K, so 18 pixels in 1080p
        distance = ResolutionConverter.scale_distance_from_4k(36, RESOLUTION_1080P)
        assert distance == 18.0

    def test_scale_zero_distance_from_4k(self):
        """Test scaling zero distance from 4K."""
        distance = ResolutionConverter.scale_distance_from_4k(0, RESOLUTION_1080P)
        assert distance == 0.0

    def test_scale_fractional_distance_from_4k(self):
        """Test scaling fractional distance from 4K."""
        distance = ResolutionConverter.scale_distance_from_4k(21.0, RESOLUTION_1080P)
        assert distance == 10.5

    def test_scale_large_distance_from_4k(self):
        """Test scaling large distance from 4K to 720p."""
        distance = ResolutionConverter.scale_distance_from_4k(300, RESOLUTION_720P)
        assert distance == pytest.approx(100.0, rel=1e-6)

    def test_scale_distance_uses_x_scale(self):
        """Test that distance scaling uses X-axis scale factor."""
        # Custom resolution with different X/Y scales
        custom_res = (1920, 1000)
        distance = ResolutionConverter.scale_distance_from_4k(20, custom_res)
        scale_x, _ = ResolutionConverter.calculate_scale_from_4k(custom_res)
        assert distance == pytest.approx(20 * scale_x, rel=1e-6)

    def test_scale_distance_4k_to_4k(self):
        """Test scaling distance from 4K to 4K (identity)."""
        distance = ResolutionConverter.scale_distance_from_4k(50, RESOLUTION_4K)
        assert distance == 50.0

    def test_distance_roundtrip(self):
        """Test round-trip accuracy for distance scaling."""
        original_distance = 25.5
        distance_4k = ResolutionConverter.scale_distance_to_4k(
            original_distance, RESOLUTION_1080P
        )
        distance = ResolutionConverter.scale_distance_from_4k(
            distance_4k, RESOLUTION_1080P
        )
        assert distance == pytest.approx(original_distance, rel=1e-6)


# ============================================================================
# Tests: is_4k_canonical
# ============================================================================


class TestIs4KCanonical:
    """Tests for is_4k_canonical method."""

    def test_4k_resolution_is_canonical(self):
        """Test that 4K resolution is recognized as canonical."""
        assert ResolutionConverter.is_4k_canonical(RESOLUTION_4K) is True

    def test_1080p_not_canonical(self):
        """Test that 1080p is not canonical."""
        assert ResolutionConverter.is_4k_canonical(RESOLUTION_1080P) is False

    def test_720p_not_canonical(self):
        """Test that 720p is not canonical."""
        assert ResolutionConverter.is_4k_canonical(RESOLUTION_720P) is False

    def test_8k_not_canonical(self):
        """Test that 8K is not canonical."""
        assert ResolutionConverter.is_4k_canonical(RESOLUTION_8K) is False

    def test_custom_resolution_not_canonical(self):
        """Test that custom resolution is not canonical."""
        assert ResolutionConverter.is_4k_canonical(RESOLUTION_CUSTOM) is False

    def test_canonical_resolution_constant(self):
        """Test that CANONICAL_RESOLUTION matches expected 4K."""
        assert CANONICAL_RESOLUTION == (3840, 2160)
        assert ResolutionConverter.is_4k_canonical(CANONICAL_RESOLUTION) is True


# ============================================================================
# Tests: get_aspect_ratio
# ============================================================================


class TestGetAspectRatio:
    """Tests for get_aspect_ratio method."""

    def test_aspect_ratio_1080p(self):
        """Test aspect ratio calculation for 1080p."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio(RESOLUTION_1080P)
        assert aspect_ratio == pytest.approx(16 / 9, rel=1e-6)

    def test_aspect_ratio_720p(self):
        """Test aspect ratio calculation for 720p."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio(RESOLUTION_720P)
        assert aspect_ratio == pytest.approx(16 / 9, rel=1e-6)

    def test_aspect_ratio_4k(self):
        """Test aspect ratio calculation for 4K."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio(RESOLUTION_4K)
        assert aspect_ratio == pytest.approx(16 / 9, rel=1e-6)

    def test_aspect_ratio_1440p(self):
        """Test aspect ratio calculation for 1440p."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio(RESOLUTION_1440P)
        assert aspect_ratio == pytest.approx(16 / 9, rel=1e-6)

    def test_aspect_ratio_square(self):
        """Test aspect ratio calculation for square resolution."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio((1000, 1000))
        assert aspect_ratio == 1.0

    def test_aspect_ratio_vertical(self):
        """Test aspect ratio calculation for vertical resolution."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio((1080, 1920))
        assert aspect_ratio == pytest.approx(9 / 16, rel=1e-6)

    def test_aspect_ratio_custom(self):
        """Test aspect ratio calculation for custom resolution."""
        aspect_ratio = ResolutionConverter.get_aspect_ratio((1600, 900))
        assert aspect_ratio == pytest.approx(16 / 9, rel=1e-6)


# ============================================================================
# Tests: Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions to_4k and from_4k."""

    def test_to_4k_convenience_function(self):
        """Test to_4k convenience function matches scale_to_4k."""
        x1, y1 = to_4k(100, 200, RESOLUTION_1080P)
        x2, y2 = ResolutionConverter.scale_to_4k(100, 200, RESOLUTION_1080P)
        assert x1 == x2
        assert y1 == y2

    def test_from_4k_convenience_function(self):
        """Test from_4k convenience function matches scale_from_4k."""
        x1, y1 = from_4k(200, 400, RESOLUTION_1080P)
        x2, y2 = ResolutionConverter.scale_from_4k(200, 400, RESOLUTION_1080P)
        assert x1 == x2
        assert y1 == y2

    def test_to_4k_with_fractional_coords(self):
        """Test to_4k with fractional coordinates."""
        x, y = to_4k(123.456, 234.567, RESOLUTION_720P)
        assert x == pytest.approx(123.456 * 3.0, rel=1e-6)
        assert y == pytest.approx(234.567 * 3.0, rel=1e-6)

    def test_from_4k_with_fractional_coords(self):
        """Test from_4k with fractional coordinates."""
        x, y = from_4k(1920.5, 1080.75, RESOLUTION_1080P)
        assert x == pytest.approx(960.25, rel=1e-6)
        assert y == pytest.approx(540.375, rel=1e-6)

    def test_convenience_roundtrip(self):
        """Test round-trip accuracy using convenience functions."""
        original_x, original_y = 555.5, 333.3
        x_4k, y_4k = to_4k(original_x, original_y, RESOLUTION_1080P)
        x, y = from_4k(x_4k, y_4k, RESOLUTION_1080P)
        assert x == pytest.approx(original_x, rel=1e-6)
        assert y == pytest.approx(original_y, rel=1e-6)


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_resolution(self):
        """Test scaling with very small resolution."""
        small_res = (100, 100)
        x_4k, y_4k = ResolutionConverter.scale_to_4k(50, 50, small_res)
        assert x_4k == pytest.approx(3840 / 2, rel=1e-6)
        assert y_4k == pytest.approx(2160 / 2, rel=1e-6)

    def test_very_large_resolution(self):
        """Test scaling with 8K resolution."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, RESOLUTION_8K)
        assert x == pytest.approx(3840.0, rel=1e-6)
        assert y == pytest.approx(2160.0, rel=1e-6)

    def test_non_standard_aspect_ratio(self):
        """Test scaling with non-standard aspect ratio."""
        non_standard = (1280, 1024)  # 5:4 aspect ratio
        x_4k, y_4k = ResolutionConverter.scale_to_4k(640, 512, non_standard)
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, non_standard)
        assert x == pytest.approx(640.0, rel=1e-6)
        assert y == pytest.approx(512.0, rel=1e-6)

    def test_maximum_4k_coordinates(self):
        """Test scaling with maximum 4K coordinates."""
        x, y = ResolutionConverter.scale_from_4k(3840, 2160, RESOLUTION_1080P)
        assert x == 1920.0
        assert y == 1080.0

    def test_precision_with_repeating_decimals(self):
        """Test precision with resolutions that create repeating decimals."""
        # 720p creates scale factor of 3.0, which should be exact
        x_4k, y_4k = ResolutionConverter.scale_to_4k(427, 240, RESOLUTION_720P)
        # Round-trip should maintain precision
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, RESOLUTION_720P)
        assert x == pytest.approx(427.0, rel=1e-6)
        assert y == pytest.approx(240.0, rel=1e-6)

    def test_negative_distance_scaling(self):
        """Test that negative distances scale correctly."""
        # Negative distances should scale the same as positive
        distance_4k = ResolutionConverter.scale_distance_to_4k(-10, RESOLUTION_1080P)
        assert distance_4k == -20.0


# ============================================================================
# Test Summary
# ============================================================================
# Total test methods: 70+ tests covering:
# - calculate_scale_to_4k: 7 tests
# - calculate_scale_from_4k: 7 tests
# - scale_to_4k: 9 tests
# - scale_from_4k: 9 tests
# - Round-trip accuracy: 6 tests
# - scale_between_resolutions: 6 tests
# - scale_distance_to_4k: 6 tests
# - scale_distance_from_4k: 7 tests
# - is_4k_canonical: 6 tests
# - get_aspect_ratio: 7 tests
# - Convenience functions: 5 tests
# - Edge cases: 6 tests
# Total: 71 comprehensive tests
