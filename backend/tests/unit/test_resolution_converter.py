"""Unit tests for ResolutionConverter.

Tests verify:
- Scale factor calculations
- Coordinate conversions to/from 4K
- Distance/radius scaling
- Round-trip conversion accuracy
"""

import pytest
from core.resolution_converter import ResolutionConverter, from_4k, to_4k


class TestScaleCalculations:
    """Test scale factor calculation methods."""

    def test_scale_to_4k_from_1080p(self):
        """Test scale from 1920×1080 to 4K."""
        scale = ResolutionConverter.calculate_scale_to_4k((1920, 1080))
        assert scale == (2.0, 2.0)

    def test_scale_to_4k_from_720p(self):
        """Test scale from 1280×720 to 4K."""
        scale = ResolutionConverter.calculate_scale_to_4k((1280, 720))
        assert scale == (3.0, 3.0)

    def test_scale_to_4k_from_4k(self):
        """Test scale from 4K to 4K (should be 1.0)."""
        scale = ResolutionConverter.calculate_scale_to_4k((3840, 2160))
        assert scale == (1.0, 1.0)

    def test_scale_from_4k_to_1080p(self):
        """Test scale from 4K to 1920×1080."""
        scale = ResolutionConverter.calculate_scale_from_4k((1920, 1080))
        assert scale == (0.5, 0.5)

    def test_scale_from_4k_to_720p(self):
        """Test scale from 4K to 1280×720."""
        scale = ResolutionConverter.calculate_scale_from_4k((1280, 720))
        assert abs(scale[0] - 0.33333333) < 0.0001
        assert abs(scale[1] - 0.33333333) < 0.0001

    def test_scale_from_4k_to_4k(self):
        """Test scale from 4K to 4K (should be 1.0)."""
        scale = ResolutionConverter.calculate_scale_from_4k((3840, 2160))
        assert scale == (1.0, 1.0)

    def test_scale_to_4k_from_higher_res(self):
        """Test scale from 8K to 4K."""
        scale = ResolutionConverter.calculate_scale_to_4k((7680, 4320))
        assert scale == (0.5, 0.5)


class TestCoordinateConversion:
    """Test coordinate conversion methods."""

    def test_scale_to_4k_from_1080p_center(self):
        """Test converting 1080p center to 4K center."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(960, 540, (1920, 1080))
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_scale_to_4k_from_720p_center(self):
        """Test converting 720p center to 4K center."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(640, 360, (1280, 720))
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_scale_from_4k_to_1080p_center(self):
        """Test converting 4K center to 1080p center."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, (1920, 1080))
        assert x == 960.0
        assert y == 540.0

    def test_scale_from_4k_to_720p_center(self):
        """Test converting 4K center to 720p center."""
        x, y = ResolutionConverter.scale_from_4k(1920, 1080, (1280, 720))
        assert abs(x - 640.0) < 0.0001
        assert abs(y - 360.0) < 0.0001

    def test_scale_to_4k_origin(self):
        """Test converting origin (0,0) to 4K."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(0, 0, (1920, 1080))
        assert x_4k == 0.0
        assert y_4k == 0.0

    def test_scale_from_4k_origin(self):
        """Test converting origin (0,0) from 4K."""
        x, y = ResolutionConverter.scale_from_4k(0, 0, (1920, 1080))
        assert x == 0.0
        assert y == 0.0


class TestBetweenResolutions:
    """Test conversion between arbitrary resolutions."""

    def test_scale_between_1080p_to_720p(self):
        """Test converting from 1080p to 720p."""
        x, y = ResolutionConverter.scale_between_resolutions(
            960, 540, (1920, 1080), (1280, 720)
        )
        assert abs(x - 640.0) < 0.0001
        assert abs(y - 360.0) < 0.0001

    def test_scale_between_720p_to_1080p(self):
        """Test converting from 720p to 1080p."""
        x, y = ResolutionConverter.scale_between_resolutions(
            640, 360, (1280, 720), (1920, 1080)
        )
        assert abs(x - 960.0) < 0.0001
        assert abs(y - 540.0) < 0.0001

    def test_scale_between_same_resolution(self):
        """Test converting between same resolution (should be identity)."""
        x, y = ResolutionConverter.scale_between_resolutions(
            640, 360, (1280, 720), (1280, 720)
        )
        assert x == 640.0
        assert y == 360.0


class TestDistanceScaling:
    """Test distance/radius scaling methods."""

    def test_scale_distance_to_4k_from_1080p(self):
        """Test scaling distance from 1080p to 4K."""
        # Ball radius: 18px in 1080p → 36px in 4K
        distance_4k = ResolutionConverter.scale_distance_to_4k(18, (1920, 1080))
        assert distance_4k == 36.0

    def test_scale_distance_from_4k_to_1080p(self):
        """Test scaling distance from 4K to 1080p."""
        # Ball radius: 36px in 4K → 18px in 1080p
        distance = ResolutionConverter.scale_distance_from_4k(36, (1920, 1080))
        assert distance == 18.0

    def test_scale_distance_to_4k_from_720p(self):
        """Test scaling distance from 720p to 4K."""
        # Ball radius: 12px in 720p → 36px in 4K
        distance_4k = ResolutionConverter.scale_distance_to_4k(12, (1280, 720))
        assert distance_4k == 36.0

    def test_scale_distance_from_4k_to_720p(self):
        """Test scaling distance from 4K to 720p."""
        # Ball radius: 36px in 4K → 12px in 720p
        distance = ResolutionConverter.scale_distance_from_4k(36, (1280, 720))
        assert abs(distance - 12.0) < 0.0001


class TestRoundTripConversion:
    """Test round-trip conversion accuracy."""

    def test_round_trip_1080p(self):
        """Test round-trip conversion 1080p → 4K → 1080p."""
        original_x, original_y = 960.5, 540.25

        # Convert to 4K
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, (1920, 1080)
        )

        # Convert back to 1080p
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, (1920, 1080))

        # Should match original within floating-point precision
        assert abs(x - original_x) < 1e-6
        assert abs(y - original_y) < 1e-6

    def test_round_trip_720p(self):
        """Test round-trip conversion 720p → 4K → 720p."""
        original_x, original_y = 640.125, 360.375

        # Convert to 4K
        x_4k, y_4k = ResolutionConverter.scale_to_4k(
            original_x, original_y, (1280, 720)
        )

        # Convert back to 720p
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, (1280, 720))

        # Should match original within floating-point precision
        assert abs(x - original_x) < 1e-6
        assert abs(y - original_y) < 1e-6

    def test_round_trip_arbitrary_resolution(self):
        """Test round-trip with arbitrary resolution."""
        original_x, original_y = 500.5, 300.25
        resolution = (1600, 900)

        # Convert to 4K
        x_4k, y_4k = ResolutionConverter.scale_to_4k(original_x, original_y, resolution)

        # Convert back
        x, y = ResolutionConverter.scale_from_4k(x_4k, y_4k, resolution)

        # Should match original
        assert abs(x - original_x) < 1e-6
        assert abs(y - original_y) < 1e-6


class TestHelperMethods:
    """Test helper/utility methods."""

    def test_is_4k_canonical_true(self):
        """Test is_4k_canonical returns True for 4K."""
        assert ResolutionConverter.is_4k_canonical((3840, 2160))

    def test_is_4k_canonical_false(self):
        """Test is_4k_canonical returns False for non-4K."""
        assert not ResolutionConverter.is_4k_canonical((1920, 1080))
        assert not ResolutionConverter.is_4k_canonical((1280, 720))
        assert not ResolutionConverter.is_4k_canonical((7680, 4320))

    def test_get_aspect_ratio_16_9(self):
        """Test get_aspect_ratio for 16:9 resolutions."""
        ratio_1080p = ResolutionConverter.get_aspect_ratio((1920, 1080))
        ratio_4k = ResolutionConverter.get_aspect_ratio((3840, 2160))

        expected_ratio = 16 / 9
        assert abs(ratio_1080p - expected_ratio) < 0.0001
        assert abs(ratio_4k - expected_ratio) < 0.0001

    def test_get_aspect_ratio_4_3(self):
        """Test get_aspect_ratio for 4:3 resolutions."""
        ratio = ResolutionConverter.get_aspect_ratio((1024, 768))
        expected_ratio = 4 / 3
        assert abs(ratio - expected_ratio) < 0.0001


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_to_4k_convenience_function(self):
        """Test to_4k convenience function."""
        x_4k, y_4k = to_4k(960, 540, (1920, 1080))
        assert x_4k == 1920.0
        assert y_4k == 1080.0

    def test_from_4k_convenience_function(self):
        """Test from_4k convenience function."""
        x, y = from_4k(1920, 1080, (1920, 1080))
        assert x == 960.0
        assert y == 540.0

    def test_convenience_functions_match_class_methods(self):
        """Test convenience functions produce same results as class methods."""
        # Test to_4k
        conv_result = to_4k(960, 540, (1920, 1080))
        class_result = ResolutionConverter.scale_to_4k(960, 540, (1920, 1080))
        assert conv_result == class_result

        # Test from_4k
        conv_result = from_4k(1920, 1080, (1920, 1080))
        class_result = ResolutionConverter.scale_from_4k(1920, 1080, (1920, 1080))
        assert conv_result == class_result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scale_zero_coordinates(self):
        """Test scaling zero coordinates."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(0, 0, (1920, 1080))
        assert x_4k == 0.0
        assert y_4k == 0.0

    def test_scale_max_coordinates(self):
        """Test scaling maximum coordinates."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(1920, 1080, (1920, 1080))
        assert x_4k == 3840.0
        assert y_4k == 2160.0

    def test_scale_fractional_coordinates(self):
        """Test scaling fractional pixel coordinates."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(960.5, 540.25, (1920, 1080))
        assert x_4k == 1921.0
        assert y_4k == 1080.5

    def test_scale_negative_coordinates(self):
        """Test scaling negative coordinates (should work mathematically)."""
        x_4k, y_4k = ResolutionConverter.scale_to_4k(-100, -50, (1920, 1080))
        assert x_4k == -200.0
        assert y_4k == -100.0


class TestPrecision:
    """Test numerical precision and accuracy."""

    def test_scale_precision_1080p(self):
        """Test scaling maintains precision for 1080p."""
        test_coords = [(100.123, 200.456), (1500.789, 900.321)]

        for x, y in test_coords:
            x_4k, y_4k = ResolutionConverter.scale_to_4k(x, y, (1920, 1080))
            x_back, y_back = ResolutionConverter.scale_from_4k(x_4k, y_4k, (1920, 1080))

            assert abs(x_back - x) < 1e-10
            assert abs(y_back - y) < 1e-10

    def test_scale_precision_720p(self):
        """Test scaling maintains precision for 720p."""
        test_coords = [(100.123, 200.456), (1000.789, 600.321)]

        for x, y in test_coords:
            x_4k, y_4k = ResolutionConverter.scale_to_4k(x, y, (1280, 720))
            x_back, y_back = ResolutionConverter.scale_from_4k(x_4k, y_4k, (1280, 720))

            assert abs(x_back - x) < 1e-10
            assert abs(y_back - y) < 1e-10
