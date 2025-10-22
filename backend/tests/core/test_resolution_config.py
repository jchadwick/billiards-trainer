"""Tests for resolution and coordinate space configuration utilities."""

import pytest

from backend.core.coordinates import Vector2D
from backend.core.resolution_config import (
    CoordinateSpace,
    ResolutionConfig,
    StandardResolution,
    TableSize,
    get_standard_resolution,
    get_table_dimensions,
    validate_point_in_space,
)


class TestStandardResolution:
    """Tests for StandardResolution enum."""

    def test_resolution_values(self):
        """Test that standard resolutions have correct values."""
        assert StandardResolution.VGA.value == (640, 480)
        assert StandardResolution.HD_720.value == (1280, 720)
        assert StandardResolution.HD_1080.value == (1920, 1080)
        assert StandardResolution.QHD.value == (2560, 1440)
        assert StandardResolution.UHD_4K.value == (3840, 2160)
        assert StandardResolution.UHD_8K.value == (7680, 4320)

    def test_resolution_properties(self):
        """Test resolution property accessors."""
        hd = StandardResolution.HD_1080
        assert hd.width == 1920
        assert hd.height == 1080
        assert hd.aspect_ratio == pytest.approx(16 / 9, rel=1e-3)
        assert hd.total_pixels == 1920 * 1080

    def test_resolution_string_representation(self):
        """Test string representation of resolutions."""
        hd = StandardResolution.HD_1080
        assert "HD_1080" in str(hd)
        assert "1920" in str(hd)
        assert "1080" in str(hd)


class TestTableSize:
    """Tests for TableSize enum."""

    def test_table_dimensions(self):
        """Test that table sizes have correct dimensions."""
        assert TableSize.NINE_FOOT.value == (2.54, 1.27)
        assert TableSize.EIGHT_FOOT.value == (2.44, 1.22)
        assert TableSize.SEVEN_FOOT.value == (2.13, 1.07)

    def test_table_properties(self):
        """Test table property accessors."""
        table = TableSize.NINE_FOOT
        assert table.width == 2.54
        assert table.height == 1.27
        assert table.aspect_ratio == pytest.approx(2.0, rel=0.1)
        assert table.area == pytest.approx(2.54 * 1.27, rel=1e-3)

    def test_table_width_feet(self):
        """Test conversion to feet for standard naming."""
        assert TableSize.NINE_FOOT.width_feet == 9
        assert TableSize.EIGHT_FOOT.width_feet == 8
        assert TableSize.SEVEN_FOOT.width_feet == 7
        assert TableSize.SIX_FOOT.width_feet == 6

    def test_table_string_representation(self):
        """Test string representation of table sizes."""
        table = TableSize.NINE_FOOT
        assert "NINE_FOOT" in str(table)
        assert "2.54" in str(table)
        assert "1.27" in str(table)


class TestCoordinateSpace:
    """Tests for CoordinateSpace class."""

    def test_initialization(self):
        """Test basic coordinate space initialization."""
        space = CoordinateSpace(width=1920, height=1080, unit="pixels")
        assert space.width == 1920
        assert space.height == 1080
        assert space.origin_x == 0.0
        assert space.origin_y == 0.0
        assert space.unit == "pixels"

    def test_validation(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="Width must be positive"):
            CoordinateSpace(width=0, height=100)

        with pytest.raises(ValueError, match="Height must be positive"):
            CoordinateSpace(width=100, height=-1)

    def test_bounds_properties(self):
        """Test bound calculation properties."""
        space = CoordinateSpace(width=100, height=50, origin_x=10, origin_y=5)
        assert space.min_x == 10
        assert space.max_x == 110
        assert space.min_y == 5
        assert space.max_y == 55

    def test_center_calculation(self):
        """Test center point calculation."""
        space = CoordinateSpace(width=100, height=50)
        center = space.center
        assert center.x == 50
        assert center.y == 25

        # Test with offset origin
        space2 = CoordinateSpace(width=100, height=50, origin_x=10, origin_y=20)
        center2 = space2.center
        assert center2.x == 60  # 10 + 100/2
        assert center2.y == 45  # 20 + 50/2

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        space = CoordinateSpace(width=1920, height=1080)
        assert space.aspect_ratio == pytest.approx(16 / 9, rel=1e-3)

    def test_contains_point(self):
        """Test point containment checking."""
        space = CoordinateSpace(width=100, height=50)

        # Points inside
        assert space.contains_point(50, 25)
        assert space.contains_point(0, 0)  # On boundary
        assert space.contains_point(100, 50)  # On boundary

        # Points outside
        assert not space.contains_point(-1, 25)
        assert not space.contains_point(101, 25)
        assert not space.contains_point(50, -1)
        assert not space.contains_point(50, 51)

    def test_contains_point_with_margin(self):
        """Test point containment with margin."""
        space = CoordinateSpace(width=100, height=50)

        # With margin, edges are excluded
        assert space.contains_point(10, 10, margin=5)
        assert not space.contains_point(2, 25, margin=5)  # Too close to left edge
        assert not space.contains_point(98, 25, margin=5)  # Too close to right edge

    def test_contains_vector(self):
        """Test Vector2D containment checking."""
        space = CoordinateSpace(width=100, height=50)

        assert space.contains_vector(Vector2D(50, 25))
        assert not space.contains_vector(Vector2D(-1, 25))
        assert not space.contains_vector(Vector2D(101, 25))

    def test_clamp_point(self):
        """Test point clamping to bounds."""
        space = CoordinateSpace(width=100, height=50)

        # Points inside remain unchanged
        assert space.clamp_point(50, 25) == (50, 25)

        # Points outside are clamped to edges
        assert space.clamp_point(-10, 25) == (0, 25)
        assert space.clamp_point(110, 25) == (100, 25)
        assert space.clamp_point(50, -10) == (50, 0)
        assert space.clamp_point(50, 60) == (50, 50)

    def test_clamp_point_with_margin(self):
        """Test point clamping with margin."""
        space = CoordinateSpace(width=100, height=50)

        # With margin, clamping respects the margin
        assert space.clamp_point(2, 25, margin=5) == (5, 25)
        assert space.clamp_point(98, 25, margin=5) == (95, 25)

    def test_clamp_vector(self):
        """Test Vector2D clamping."""
        space = CoordinateSpace(width=100, height=50)

        # Test clamping
        clamped = space.clamp_vector(Vector2D(-10, 60))
        assert clamped.x == 0
        assert clamped.y == 50

    def test_normalize_point(self):
        """Test point normalization to [0, 1] range."""
        space = CoordinateSpace(width=100, height=50)

        # Center point should normalize to (0.5, 0.5)
        norm_x, norm_y = space.normalize_point(50, 25)
        assert norm_x == pytest.approx(0.5)
        assert norm_y == pytest.approx(0.5)

        # Corners
        assert space.normalize_point(0, 0) == (0.0, 0.0)
        assert space.normalize_point(100, 50) == (1.0, 1.0)

    def test_denormalize_point(self):
        """Test conversion from normalized coordinates back to actual."""
        space = CoordinateSpace(width=100, height=50)

        # Test denormalization
        assert space.denormalize_point(0.5, 0.5) == (50.0, 25.0)
        assert space.denormalize_point(0.0, 0.0) == (0.0, 0.0)
        assert space.denormalize_point(1.0, 1.0) == (100.0, 50.0)

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is reversible."""
        space = CoordinateSpace(width=100, height=50)

        # Test several points
        test_points = [(25, 12.5), (0, 0), (100, 50), (75, 37.5)]

        for x, y in test_points:
            norm_x, norm_y = space.normalize_point(x, y)
            denorm_x, denorm_y = space.denormalize_point(norm_x, norm_y)
            assert denorm_x == pytest.approx(x)
            assert denorm_y == pytest.approx(y)


class TestResolutionConfig:
    """Tests for ResolutionConfig utility class."""

    def test_get_resolution_by_name(self):
        """Test getting resolution by name."""
        assert ResolutionConfig.get_resolution("HD_1080") == (1920, 1080)
        assert ResolutionConfig.get_resolution("UHD_4K") == (3840, 2160)
        assert ResolutionConfig.get_resolution("VGA") == (640, 480)

    def test_get_resolution_with_aliases(self):
        """Test resolution retrieval with common aliases."""
        # Test common aliases
        assert ResolutionConfig.get_resolution("1080") == (1920, 1080)
        assert ResolutionConfig.get_resolution("1080p") == (1920, 1080)
        assert ResolutionConfig.get_resolution("FHD") == (1920, 1080)
        assert ResolutionConfig.get_resolution("4K") == (3840, 2160)
        assert ResolutionConfig.get_resolution("2K") == (2560, 1440)

    def test_get_resolution_case_insensitive(self):
        """Test that resolution lookup is case-insensitive."""
        assert ResolutionConfig.get_resolution("hd_1080") == (1920, 1080)
        assert ResolutionConfig.get_resolution("HD_1080") == (1920, 1080)
        assert ResolutionConfig.get_resolution("Hd_1080") == (1920, 1080)

    def test_get_resolution_invalid(self):
        """Test that invalid resolution names return None."""
        assert ResolutionConfig.get_resolution("INVALID") is None
        assert ResolutionConfig.get_resolution("999p") is None

    def test_get_table_dimensions_by_name(self):
        """Test getting table dimensions by name."""
        assert ResolutionConfig.get_table_dimensions("NINE_FOOT") == (2.54, 1.27)
        assert ResolutionConfig.get_table_dimensions("EIGHT_FOOT") == (2.44, 1.22)

    def test_get_table_dimensions_with_aliases(self):
        """Test table dimension retrieval with aliases."""
        assert ResolutionConfig.get_table_dimensions("9") == (2.54, 1.27)
        assert ResolutionConfig.get_table_dimensions("9ft") == (2.54, 1.27)
        assert ResolutionConfig.get_table_dimensions("9-foot") == (2.54, 1.27)

    def test_get_table_dimensions_invalid(self):
        """Test that invalid table sizes return None."""
        assert ResolutionConfig.get_table_dimensions("INVALID") is None

    def test_create_pixel_space(self):
        """Test creation of pixel coordinate space."""
        space = ResolutionConfig.create_pixel_space((1920, 1080))

        assert space.width == 1920
        assert space.height == 1080
        assert space.origin_x == 0.0
        assert space.origin_y == 0.0
        assert space.unit == "pixels"

    def test_create_table_space_default(self):
        """Test creation of table coordinate space (default origin)."""
        space = ResolutionConfig.create_table_space((2.54, 1.27))

        assert space.width == 2.54
        assert space.height == 1.27
        assert space.origin_x == 0.0
        assert space.origin_y == 0.0
        assert space.unit == "meters"

    def test_create_table_space_centered(self):
        """Test creation of centered table coordinate space."""
        space = ResolutionConfig.create_table_space((2.54, 1.27), centered=True)

        assert space.width == 2.54
        assert space.height == 1.27
        assert space.origin_x == pytest.approx(-1.27)
        assert space.origin_y == pytest.approx(-0.635)
        assert space.unit == "meters"

        # Center should be at (0, 0)
        center = space.center
        assert center.x == pytest.approx(0.0, abs=1e-10)
        assert center.y == pytest.approx(0.0, abs=1e-10)

    def test_validate_coordinates_valid(self):
        """Test validation of valid coordinates."""
        space = CoordinateSpace(width=100, height=50)
        is_valid, error = ResolutionConfig.validate_coordinates(50, 25, space)

        assert is_valid is True
        assert error is None

    def test_validate_coordinates_invalid(self):
        """Test validation of invalid coordinates."""
        space = CoordinateSpace(width=100, height=50)

        # Test out of bounds
        is_valid, error = ResolutionConfig.validate_coordinates(-10, 25, space)
        assert is_valid is False
        assert error is not None
        assert "out of bounds" in error.lower()

    def test_validate_vector(self):
        """Test validation of Vector2D positions."""
        space = CoordinateSpace(width=100, height=50)

        is_valid, _ = ResolutionConfig.validate_vector(Vector2D(50, 25), space)
        assert is_valid is True

        is_valid, _ = ResolutionConfig.validate_vector(Vector2D(-10, 25), space)
        assert is_valid is False

    def test_scale_coordinates(self):
        """Test coordinate scaling between spaces."""
        # From 1920x1080 pixels to 100x50 units
        from_space = CoordinateSpace(width=1920, height=1080, unit="pixels")
        to_space = CoordinateSpace(width=100, height=50, unit="units")

        # Center point should map to center
        scaled_x, scaled_y = ResolutionConfig.scale_coordinates(
            960, 540, from_space, to_space
        )
        assert scaled_x == pytest.approx(50.0)
        assert scaled_y == pytest.approx(25.0)

        # Origin should map to origin
        scaled_x, scaled_y = ResolutionConfig.scale_coordinates(
            0, 0, from_space, to_space
        )
        assert scaled_x == pytest.approx(0.0)
        assert scaled_y == pytest.approx(0.0)

    def test_scale_vector(self):
        """Test Vector2D scaling between spaces."""
        from_space = CoordinateSpace(width=1920, height=1080, unit="pixels")
        to_space = CoordinateSpace(width=100, height=50, unit="units")

        # Test center point
        scaled = ResolutionConfig.scale_vector(Vector2D(960, 540), from_space, to_space)
        assert scaled.x == pytest.approx(50.0)
        assert scaled.y == pytest.approx(25.0)

    def test_scale_coordinates_with_offsets(self):
        """Test scaling between coordinate spaces with different origins."""
        # Centered table space to pixel space
        from_space = CoordinateSpace(
            width=2.54, height=1.27, origin_x=-1.27, origin_y=-0.635, unit="meters"
        )
        to_space = CoordinateSpace(width=1920, height=1080, unit="pixels")

        # Center (0, 0) in table space should map to center in pixel space
        scaled_x, scaled_y = ResolutionConfig.scale_coordinates(
            0, 0, from_space, to_space
        )
        assert scaled_x == pytest.approx(960.0)
        assert scaled_y == pytest.approx(540.0)


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_get_standard_resolution(self):
        """Test get_standard_resolution convenience function."""
        assert get_standard_resolution("1080") == (1920, 1080)
        assert get_standard_resolution("4K") == (3840, 2160)

    def test_get_table_dimensions(self):
        """Test get_table_dimensions convenience function."""
        assert get_table_dimensions("9") == (2.54, 1.27)
        assert get_table_dimensions("NINE_FOOT") == (2.54, 1.27)

    def test_validate_point_in_space(self):
        """Test validate_point_in_space convenience function."""
        assert validate_point_in_space(50, 25, 100, 50) is True
        assert validate_point_in_space(-1, 25, 100, 50) is False
        assert validate_point_in_space(50, 60, 100, 50) is False

        # With margin
        assert validate_point_in_space(2, 25, 100, 50, margin=5) is False
        assert validate_point_in_space(10, 25, 100, 50, margin=5) is True


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_pixel_to_table_coordinate_mapping(self):
        """Test realistic scenario of mapping pixel coords to table coords."""
        # Setup: HD camera viewing a 9-foot table
        pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
        table_space = ResolutionConfig.create_table_space((2.54, 1.27))

        # Ball detected at center of image should be at center of table
        table_pos = ResolutionConfig.scale_vector(
            Vector2D(960, 540), pixel_space, table_space
        )

        assert table_pos.x == pytest.approx(1.27, rel=1e-3)  # Half of 2.54
        assert table_pos.y == pytest.approx(0.635, rel=1e-3)  # Half of 1.27

    def test_coordinate_validation_workflow(self):
        """Test complete validation workflow."""
        # Create table space
        table_space = ResolutionConfig.create_table_space((2.54, 1.27))

        # Validate some ball positions
        ball_positions = [
            Vector2D(1.27, 0.635),  # Center - valid
            Vector2D(0.1, 0.1),  # Near corner - valid
            Vector2D(-0.5, 0.5),  # Outside table - invalid
            Vector2D(3.0, 1.0),  # Outside table - invalid
        ]

        valid_count = 0
        for pos in ball_positions:
            is_valid, _ = ResolutionConfig.validate_vector(pos, table_space)
            if is_valid:
                valid_count += 1

        assert valid_count == 2  # Only first two are valid

    def test_multi_resolution_support(self):
        """Test that the system handles multiple resolutions correctly."""
        resolutions = ["720", "1080", "4K"]
        spaces = []

        for res_name in resolutions:
            res = get_standard_resolution(res_name)
            assert res is not None
            space = ResolutionConfig.create_pixel_space(res)
            spaces.append(space)

        # Verify all spaces were created with correct dimensions
        assert spaces[0].width == 1280  # 720p
        assert spaces[1].width == 1920  # 1080p
        assert spaces[2].width == 3840  # 4K

    def test_table_size_variations(self):
        """Test that different table sizes are handled correctly."""
        table_names = ["6", "7", "8", "9"]
        table_spaces = []

        for name in table_names:
            dims = get_table_dimensions(name)
            assert dims is not None
            space = ResolutionConfig.create_table_space(dims)
            table_spaces.append(space)

        # Verify dimensions increase with table size
        for i in range(len(table_spaces) - 1):
            assert table_spaces[i].width < table_spaces[i + 1].width
