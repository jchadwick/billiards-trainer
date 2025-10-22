"""Unit tests for the CoordinateConverter utility.

These tests verify the correctness of coordinate transformations between different
coordinate spaces used in the billiards trainer system.

Test coverage includes:
- Basic conversions between all coordinate spaces
- Batch conversion performance
- Resolution scaling
- Perspective transforms
- Round-trip conversion validation
- Edge cases and error handling
"""

import math

import numpy as np
import pytest
from core.coordinate_converter import (
    CAMERA_NATIVE_RESOLUTION,
    CoordinateConverter,
    CoordinateSpace,
    PerspectiveTransform,
    Resolution,
)
from core.coordinates import Vector2D


class TestResolution:
    """Test the Resolution dataclass."""

    def test_resolution_creation(self):
        """Test creating a resolution."""
        res = Resolution(1920, 1080)
        assert res.width == 1920
        assert res.height == 1080

    def test_resolution_string(self):
        """Test string representation."""
        res = Resolution(1920, 1080)
        assert str(res) == "1920x1080"

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        res = Resolution(1920, 1080)
        assert abs(res.aspect_ratio - 16 / 9) < 1e-6

    def test_scale_to(self):
        """Test scale factor calculation."""
        source = Resolution(1920, 1080)
        target = Resolution(640, 480)
        scale_x, scale_y = source.scale_to(target)
        assert abs(scale_x - 640 / 1920) < 1e-6
        assert abs(scale_y - 480 / 1080) < 1e-6

    def test_is_valid(self):
        """Test resolution validation."""
        assert Resolution(1920, 1080).is_valid()
        assert not Resolution(0, 1080).is_valid()
        assert not Resolution(1920, 0).is_valid()
        assert not Resolution(-1, 1080).is_valid()

    def test_scale_to_zero_resolution(self):
        """Test that scaling from zero resolution raises error."""
        zero_res = Resolution(0, 0)
        target = Resolution(1920, 1080)
        with pytest.raises(ValueError):
            zero_res.scale_to(target)


class TestPerspectiveTransform:
    """Test the PerspectiveTransform class."""

    def test_identity_transform(self):
        """Test identity transformation."""
        matrix = np.eye(3, dtype=np.float64)
        transform = PerspectiveTransform(matrix=matrix)

        point = Vector2D(100, 200)
        transformed = transform.apply(point)

        assert abs(transformed.x - 100) < 1e-6
        assert abs(transformed.y - 200) < 1e-6

    def test_translation_transform(self):
        """Test simple translation."""
        # Translation by (50, 100)
        matrix = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]], dtype=np.float64)
        transform = PerspectiveTransform(matrix=matrix)

        point = Vector2D(100, 200)
        transformed = transform.apply(point)

        assert abs(transformed.x - 150) < 1e-6
        assert abs(transformed.y - 300) < 1e-6

    def test_scale_transform(self):
        """Test simple scaling."""
        # Scale by 2x
        matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        transform = PerspectiveTransform(matrix=matrix)

        point = Vector2D(100, 200)
        transformed = transform.apply(point)

        assert abs(transformed.x - 200) < 1e-6
        assert abs(transformed.y - 400) < 1e-6

    def test_inverse_transform(self):
        """Test inverse transformation."""
        # Translation by (50, 100)
        matrix = np.array([[1, 0, 50], [0, 1, 100], [0, 0, 1]], dtype=np.float64)
        transform = PerspectiveTransform(matrix=matrix)

        point = Vector2D(100, 200)
        transformed = transform.apply(point)
        back = transform.apply_inverse(transformed)

        assert abs(back.x - point.x) < 1e-6
        assert abs(back.y - point.y) < 1e-6

    def test_invalid_matrix_shape(self):
        """Test that invalid matrix shape raises error."""
        matrix = np.eye(2, dtype=np.float64)  # Wrong shape
        with pytest.raises(ValueError):
            PerspectiveTransform(matrix=matrix)

    def test_singular_matrix(self):
        """Test that singular matrix raises error."""
        # Singular matrix (determinant = 0)
        matrix = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(ValueError):
            PerspectiveTransform(matrix=matrix)


class TestCoordinateConverter:
    """Test the CoordinateConverter class."""

    @pytest.fixture()
    def converter(self):
        """Create a standard converter for testing."""
        return CoordinateConverter(
            table_width_meters=2.54,
            table_height_meters=1.27,
            pixels_per_meter=754.0,
            camera_resolution=Resolution(1920, 1080),
        )

    def test_converter_creation(self, converter):
        """Test creating a coordinate converter."""
        assert converter.table_width_meters == 2.54
        assert converter.table_height_meters == 1.27
        assert converter.pixels_per_meter == 754.0
        assert converter.camera_resolution == Resolution(1920, 1080)

    def test_invalid_table_dimensions(self):
        """Test that invalid table dimensions raise error."""
        with pytest.raises(ValueError):
            CoordinateConverter(table_width_meters=-1)

        with pytest.raises(ValueError):
            CoordinateConverter(table_height_meters=0)

    def test_invalid_pixels_per_meter(self):
        """Test that invalid pixels_per_meter raises error."""
        with pytest.raises(ValueError):
            CoordinateConverter(pixels_per_meter=-100)

        with pytest.raises(ValueError):
            CoordinateConverter(pixels_per_meter=0)

    def test_camera_pixels_to_world_meters(self, converter):
        """Test converting camera pixels to world meters."""
        # Center of 1920x1080 should be approximately center of table
        center_pixels = Vector2D(960, 540)
        world = converter.camera_pixels_to_world_meters(
            center_pixels, Resolution(1920, 1080)
        )

        # Should be around 1.27m x 0.72m (center of 2.54m x 1.27m table)
        assert abs(world.x - 960 / 754.0) < 0.01  # ~1.27m
        assert abs(world.y - 540 / 754.0) < 0.01  # ~0.72m

    def test_world_meters_to_camera_pixels(self, converter):
        """Test converting world meters to camera pixels."""
        # Table center in meters
        world = Vector2D(1.27, 0.635)
        pixels = converter.world_meters_to_camera_pixels(world, Resolution(1920, 1080))

        # Should be around center of camera view
        expected_x = 1.27 * 754.0  # ~958 pixels
        expected_y = 0.635 * 754.0  # ~479 pixels

        assert abs(pixels.x - expected_x) < 1.0
        assert abs(pixels.y - expected_y) < 1.0

    def test_camera_world_round_trip(self, converter):
        """Test round-trip conversion camera <-> world."""
        original = Vector2D(960, 540)
        resolution = Resolution(1920, 1080)

        # Convert to world and back
        world = converter.camera_pixels_to_world_meters(original, resolution)
        back = converter.world_meters_to_camera_pixels(world, resolution)

        assert abs(back.x - original.x) < 1e-6
        assert abs(back.y - original.y) < 1e-6

    def test_resolution_scaling(self, converter):
        """Test that different resolutions scale correctly."""
        # Same position at different resolutions should map to same world position
        high_res = Vector2D(1920, 1080)
        low_res = Vector2D(640, 360)

        world_high = converter.camera_pixels_to_world_meters(
            high_res, Resolution(1920, 1080)
        )
        world_low = converter.camera_pixels_to_world_meters(
            low_res, Resolution(640, 360)
        )

        # Both represent bottom-right corner, should be same in world coordinates
        assert abs(world_high.x - world_low.x) < 0.01
        assert abs(world_high.y - world_low.y) < 0.01

    def test_normalized_to_camera_pixels(self, converter):
        """Test normalized coordinate conversion."""
        # Center in normalized coordinates
        normalized = Vector2D(0.5, 0.5)
        pixels = converter.normalized_to_camera_pixels(
            normalized, Resolution(1920, 1080)
        )

        assert abs(pixels.x - 960) < 1e-6
        assert abs(pixels.y - 540) < 1e-6

    def test_camera_pixels_to_normalized(self, converter):
        """Test camera to normalized conversion."""
        pixels = Vector2D(960, 540)
        normalized = converter.camera_pixels_to_normalized(
            pixels, Resolution(1920, 1080)
        )

        assert abs(normalized.x - 0.5) < 1e-6
        assert abs(normalized.y - 0.5) < 1e-6

    def test_normalized_invalid_range(self, converter):
        """Test that out-of-range normalized coordinates raise error."""
        invalid = Vector2D(1.5, 0.5)  # x > 1
        with pytest.raises(ValueError):
            converter.normalized_to_camera_pixels(invalid, Resolution(1920, 1080))

        invalid = Vector2D(0.5, -0.1)  # y < 0
        with pytest.raises(ValueError):
            converter.normalized_to_camera_pixels(invalid, Resolution(1920, 1080))

    def test_table_pixels_to_world_meters_simple(self, converter):
        """Test simple table pixel conversion (no perspective)."""
        # Center of 640x360 table space
        table_pos = Vector2D(320, 180)
        world = converter.table_pixels_to_world_meters(table_pos, Resolution(640, 360))

        # Should be center of table
        assert abs(world.x - 1.27) < 0.01  # Half of 2.54m
        assert abs(world.y - 0.635) < 0.01  # Half of 1.27m

    def test_world_meters_to_table_pixels_simple(self, converter):
        """Test simple world to table conversion (no perspective)."""
        # Table center
        world = Vector2D(1.27, 0.635)
        table = converter.world_meters_to_table_pixels(world, Resolution(640, 360))

        # Should be center of table pixel space
        assert abs(table.x - 320) < 0.5
        assert abs(table.y - 180) < 0.5

    def test_generic_convert_same_space(self, converter):
        """Test generic convert with same source and target space."""
        original = Vector2D(100, 100)
        result = converter.convert(
            original,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.CAMERA_PIXELS,
            from_resolution=Resolution(1920, 1080),
            to_resolution=Resolution(1920, 1080),
        )

        assert result.x == original.x
        assert result.y == original.y

    def test_generic_convert_different_resolutions(self, converter):
        """Test generic convert with different resolutions in same space."""
        original = Vector2D(1920, 1080)
        result = converter.convert(
            original,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.CAMERA_PIXELS,
            from_resolution=Resolution(1920, 1080),
            to_resolution=Resolution(640, 360),
        )

        # Should scale down
        assert abs(result.x - 640) < 1.0
        assert abs(result.y - 360) < 1.0

    def test_generic_convert_camera_to_world(self, converter):
        """Test generic convert from camera to world."""
        camera_pos = Vector2D(960, 540)
        world = converter.convert(
            camera_pos,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.WORLD_METERS,
            from_resolution=Resolution(1920, 1080),
        )

        expected_x = 960 / 754.0
        expected_y = 540 / 754.0

        assert abs(world.x - expected_x) < 0.01
        assert abs(world.y - expected_y) < 0.01

    def test_generic_convert_world_to_normalized(self, converter):
        """Test generic convert from world to normalized."""
        world = Vector2D(1.27, 0.635)
        normalized = converter.convert(
            world,
            CoordinateSpace.WORLD_METERS,
            CoordinateSpace.TABLE_NORMALIZED,
        )

        # Should be center (0.5, 0.5)
        assert abs(normalized.x - 0.5) < 0.01
        assert abs(normalized.y - 0.5) < 0.01

    def test_batch_conversion_empty(self, converter):
        """Test batch conversion with empty list."""
        result = converter.camera_pixels_to_world_meters_batch(
            [], Resolution(1920, 1080)
        )
        assert result == []

    def test_batch_conversion_camera_to_world(self, converter):
        """Test batch conversion from camera to world."""
        positions = [
            Vector2D(0, 0),
            Vector2D(960, 540),
            Vector2D(1920, 1080),
        ]

        result = converter.camera_pixels_to_world_meters_batch(
            positions, Resolution(1920, 1080)
        )

        assert len(result) == 3

        # Check first position (origin)
        assert abs(result[0].x - 0) < 0.01
        assert abs(result[0].y - 0) < 0.01

        # Check middle position
        assert abs(result[1].x - 960 / 754.0) < 0.01
        assert abs(result[1].y - 540 / 754.0) < 0.01

        # Check last position
        assert abs(result[2].x - 1920 / 754.0) < 0.01
        assert abs(result[2].y - 1080 / 754.0) < 0.01

    def test_batch_conversion_world_to_camera(self, converter):
        """Test batch conversion from world to camera."""
        positions = [
            Vector2D(0, 0),
            Vector2D(1.27, 0.635),
            Vector2D(2.54, 1.27),
        ]

        result = converter.world_meters_to_camera_pixels_batch(
            positions, Resolution(1920, 1080)
        )

        assert len(result) == 3

        # Check first position
        assert abs(result[0].x - 0) < 1.0
        assert abs(result[0].y - 0) < 1.0

        # Check middle position
        assert abs(result[1].x - 1.27 * 754.0) < 1.0
        assert abs(result[1].y - 0.635 * 754.0) < 1.0

        # Check last position
        assert abs(result[2].x - 2.54 * 754.0) < 1.0
        assert abs(result[2].y - 1.27 * 754.0) < 1.0

    def test_batch_conversion_consistency(self, converter):
        """Test that batch conversion gives same results as individual."""
        positions = [
            Vector2D(100, 100),
            Vector2D(500, 500),
            Vector2D(1000, 1000),
        ]
        resolution = Resolution(1920, 1080)

        # Batch conversion
        batch_result = converter.camera_pixels_to_world_meters_batch(
            positions, resolution
        )

        # Individual conversions
        individual_result = [
            converter.camera_pixels_to_world_meters(pos, resolution)
            for pos in positions
        ]

        # Should be identical
        for batch, individual in zip(batch_result, individual_result):
            assert abs(batch.x - individual.x) < 1e-6
            assert abs(batch.y - individual.y) < 1e-6

    def test_validate_conversion_success(self, converter):
        """Test conversion validation succeeds for valid conversion."""
        original = Vector2D(960, 540)
        is_valid = converter.validate_conversion(
            original,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.WORLD_METERS,
            from_resolution=Resolution(1920, 1080),
        )

        assert is_valid

    def test_validate_conversion_with_tolerance(self, converter):
        """Test conversion validation with custom tolerance."""
        original = Vector2D(960, 540)

        # Should pass with large tolerance
        assert converter.validate_conversion(
            original,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.WORLD_METERS,
            tolerance=1.0,
            from_resolution=Resolution(1920, 1080),
        )

        # May fail with very small tolerance due to floating point
        # (but should still pass for this simple case)
        assert converter.validate_conversion(
            original,
            CoordinateSpace.CAMERA_PIXELS,
            CoordinateSpace.WORLD_METERS,
            tolerance=1e-10,
            from_resolution=Resolution(1920, 1080),
        )

    def test_get_calibration_info(self, converter):
        """Test getting calibration info."""
        info = converter.get_calibration_info()

        assert info["table_width_meters"] == 2.54
        assert info["table_height_meters"] == 1.27
        assert info["pixels_per_meter"] == 754.0
        assert info["camera_resolution"]["width"] == 1920
        assert info["camera_resolution"]["height"] == 1080
        assert info["has_perspective_transform"] is False

    def test_repr(self, converter):
        """Test string representation."""
        repr_str = repr(converter)
        assert "CoordinateConverter" in repr_str
        assert "2.54x1.27" in repr_str
        assert "754.0" in repr_str
        assert "1920x1080" in repr_str

    def test_unsupported_coordinate_space(self, converter):
        """Test that unsupported coordinate spaces raise errors."""
        # Create a fake coordinate space (if we add more in the future)
        # For now, just test the error path exists
        with pytest.raises(ValueError):
            converter._to_world_meters(
                Vector2D(100, 100), None, Resolution(1920, 1080), None  # Invalid space
            )


class TestPerspectiveCorrection:
    """Test perspective correction functionality."""

    @pytest.fixture()
    def converter_with_perspective(self):
        """Create converter with perspective transform."""
        # Simple translation transform for testing
        matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
        transform = PerspectiveTransform(matrix=matrix)

        return CoordinateConverter(
            table_width_meters=2.54,
            table_height_meters=1.27,
            pixels_per_meter=754.0,
            camera_resolution=Resolution(1920, 1080),
            perspective_transform=transform,
        )

    def test_perspective_applied_camera_to_world(self, converter_with_perspective):
        """Test that perspective transform is applied in camera to world conversion."""
        original = Vector2D(100, 100)
        world = converter_with_perspective.camera_pixels_to_world_meters(
            original, Resolution(1920, 1080)
        )

        # Should be different from non-perspective version due to transform
        # Expected: (100, 100) -> (110, 120) via transform -> meters
        expected_x = 110 / 754.0
        expected_y = 120 / 754.0

        assert abs(world.x - expected_x) < 0.01
        assert abs(world.y - expected_y) < 0.01

    def test_perspective_round_trip(self, converter_with_perspective):
        """Test round-trip with perspective correction."""
        original = Vector2D(500, 500)
        resolution = Resolution(1920, 1080)

        # Convert to world and back
        world = converter_with_perspective.camera_pixels_to_world_meters(
            original, resolution
        )
        back = converter_with_perspective.world_meters_to_camera_pixels(
            world, resolution
        )

        # Should round-trip correctly
        assert abs(back.x - original.x) < 1e-6
        assert abs(back.y - original.y) < 1e-6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture()
    def converter(self):
        """Create a standard converter for testing."""
        return CoordinateConverter()

    def test_origin_conversion(self, converter):
        """Test converting the origin."""
        origin = Vector2D(0, 0)
        world = converter.camera_pixels_to_world_meters(origin, Resolution(1920, 1080))

        assert abs(world.x - 0) < 1e-6
        assert abs(world.y - 0) < 1e-6

    def test_large_coordinates(self, converter):
        """Test converting very large coordinates."""
        large = Vector2D(1e6, 1e6)
        world = converter.camera_pixels_to_world_meters(large, Resolution(1920, 1080))

        # Should still work, just be a very large world coordinate
        assert world.x > 0
        assert world.y > 0
        assert math.isfinite(world.x)
        assert math.isfinite(world.y)

    def test_negative_coordinates(self, converter):
        """Test converting negative coordinates."""
        negative = Vector2D(-100, -100)
        world = converter.camera_pixels_to_world_meters(
            negative, Resolution(1920, 1080)
        )

        # Should work fine, just negative world coordinates
        assert world.x < 0
        assert world.y < 0
        assert math.isfinite(world.x)
        assert math.isfinite(world.y)

    def test_fractional_pixels(self, converter):
        """Test converting fractional pixel coordinates."""
        fractional = Vector2D(100.5, 200.7)
        world = converter.camera_pixels_to_world_meters(
            fractional, Resolution(1920, 1080)
        )

        # Should handle fractional values
        assert math.isfinite(world.x)
        assert math.isfinite(world.y)

    def test_very_small_resolution(self, converter):
        """Test with very small resolution."""
        tiny_res = Resolution(10, 10)
        pos = Vector2D(5, 5)

        world = converter.camera_pixels_to_world_meters(pos, tiny_res)
        assert math.isfinite(world.x)
        assert math.isfinite(world.y)

    def test_very_large_resolution(self, converter):
        """Test with very large resolution."""
        huge_res = Resolution(10000, 10000)
        pos = Vector2D(5000, 5000)

        world = converter.camera_pixels_to_world_meters(pos, huge_res)
        assert math.isfinite(world.x)
        assert math.isfinite(world.y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
