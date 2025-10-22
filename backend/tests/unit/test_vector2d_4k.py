"""Comprehensive tests for the new Vector2D with mandatory scale metadata.

Tests cover:
- Factory methods (from_4k, from_resolution)
- Conversions (to_4k_canonical, to_resolution)
- Scale preservation in operations
- Round-trip accuracy
- Geometric operations
- Serialization
"""

import math

import pytest
from core.coordinates import Vector2D


class TestVector2DFactory:
    """Test factory methods for creating Vector2D instances."""

    def test_from_4k_creates_canonical_scale(self):
        """Test that from_4k creates vectors with scale = (1.0, 1.0)."""
        v = Vector2D.from_4k(1920.0, 1080.0)
        assert v.x == 1920.0
        assert v.y == 1080.0
        assert v.scale == (1.0, 1.0)

    def test_from_resolution_1080p(self):
        """Test creating from 1080p resolution."""
        v = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        assert v.x == 960.0
        assert v.y == 540.0
        assert v.scale == (2.0, 2.0)

    def test_from_resolution_720p(self):
        """Test creating from 720p resolution."""
        v = Vector2D.from_resolution(640.0, 360.0, (1280, 720))
        assert v.x == 640.0
        assert v.y == 360.0
        assert v.scale == (3.0, 3.0)

    def test_from_resolution_4k(self):
        """Test creating from 4K resolution (should have scale 1.0)."""
        v = Vector2D.from_resolution(1920.0, 1080.0, (3840, 2160))
        assert v.x == 1920.0
        assert v.y == 1080.0
        assert v.scale == (1.0, 1.0)

    def test_zero_factory(self):
        """Test zero vector factory method."""
        v = Vector2D.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.scale == (1.0, 1.0)

    def test_zero_factory_with_scale(self):
        """Test zero vector with custom scale."""
        v = Vector2D.zero(scale=(2.0, 2.0))
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.scale == (2.0, 2.0)

    def test_unit_x(self):
        """Test unit_x factory method."""
        v = Vector2D.unit_x()
        assert v.x == 1.0
        assert v.y == 0.0
        assert v.scale == (1.0, 1.0)

    def test_unit_y(self):
        """Test unit_y factory method."""
        v = Vector2D.unit_y()
        assert v.x == 0.0
        assert v.y == 1.0
        assert v.scale == (1.0, 1.0)


class TestVector2DValidation:
    """Test validation of Vector2D instances."""

    def test_missing_scale_raises_error(self):
        """Test that creating Vector2D without scale raises error."""
        with pytest.raises(TypeError):
            Vector2D(100.0, 50.0)  # Missing scale parameter

    def test_negative_scale_raises_error(self):
        """Test that negative scale factors raise error."""
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            Vector2D(100.0, 50.0, scale=(-1.0, 1.0))

    def test_zero_scale_raises_error(self):
        """Test that zero scale factors raise error."""
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            Vector2D(100.0, 50.0, scale=(0.0, 1.0))

    def test_invalid_scale_type_raises_error(self):
        """Test that invalid scale type raises error."""
        with pytest.raises(ValueError):
            Vector2D(100.0, 50.0, scale=(1.0,))  # Only one value


class TestVector2DConversions:
    """Test coordinate conversions between resolutions."""

    def test_to_4k_canonical_from_1080p(self):
        """Test converting 1080p coordinates to 4K canonical."""
        v_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        v_4k = v_1080p.to_4k_canonical()

        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0
        assert v_4k.scale == (1.0, 1.0)

    def test_to_4k_canonical_from_720p(self):
        """Test converting 720p coordinates to 4K canonical."""
        v_720p = Vector2D.from_resolution(640.0, 360.0, (1280, 720))
        v_4k = v_720p.to_4k_canonical()

        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0
        assert v_4k.scale == (1.0, 1.0)

    def test_to_4k_canonical_already_4k(self):
        """Test converting 4K coordinates to 4K (should be identity)."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_4k_result = v_4k.to_4k_canonical()

        assert v_4k_result.x == 1920.0
        assert v_4k_result.y == 1080.0
        assert v_4k_result.scale == (1.0, 1.0)

    def test_to_resolution_4k_to_1080p(self):
        """Test converting 4K to 1080p."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_1080p = v_4k.to_resolution((1920, 1080))

        assert v_1080p.x == 960.0
        assert v_1080p.y == 540.0
        assert v_1080p.scale == (2.0, 2.0)

    def test_to_resolution_4k_to_720p(self):
        """Test converting 4K to 720p."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_720p = v_4k.to_resolution((1280, 720))

        assert v_720p.x == 640.0
        assert v_720p.y == 360.0
        assert v_720p.scale == (3.0, 3.0)

    def test_round_trip_1080p(self):
        """Test round-trip conversion 1080p -> 4K -> 1080p."""
        original = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        v_4k = original.to_4k_canonical()
        back_to_1080p = v_4k.to_resolution((1920, 1080))

        assert abs(back_to_1080p.x - original.x) < 1e-10
        assert abs(back_to_1080p.y - original.y) < 1e-10
        assert back_to_1080p.scale == original.scale

    def test_round_trip_720p(self):
        """Test round-trip conversion 720p -> 4K -> 720p."""
        original = Vector2D.from_resolution(640.0, 360.0, (1280, 720))
        v_4k = original.to_4k_canonical()
        back_to_720p = v_4k.to_resolution((1280, 720))

        assert abs(back_to_720p.x - original.x) < 1e-10
        assert abs(back_to_720p.y - original.y) < 1e-10
        assert back_to_720p.scale == original.scale

    def test_round_trip_with_fractional_coordinates(self):
        """Test round-trip with fractional coordinates."""
        original = Vector2D.from_resolution(123.456, 789.012, (1920, 1080))
        v_4k = original.to_4k_canonical()
        back = v_4k.to_resolution((1920, 1080))

        assert abs(back.x - original.x) < 1e-10
        assert abs(back.y - original.y) < 1e-10


class TestVector2DGeometricOperations:
    """Test geometric operations preserve scale correctly."""

    def test_magnitude(self):
        """Test magnitude calculation."""
        v = Vector2D.from_4k(3.0, 4.0)
        assert v.magnitude() == 5.0

    def test_magnitude_squared(self):
        """Test squared magnitude calculation."""
        v = Vector2D.from_4k(3.0, 4.0)
        assert v.magnitude_squared() == 25.0

    def test_normalize_preserves_scale(self):
        """Test that normalize preserves scale metadata."""
        v = Vector2D.from_resolution(300.0, 400.0, (1920, 1080))
        v_normalized = v.normalize()

        assert abs(v_normalized.magnitude() - 1.0) < 1e-10
        assert v_normalized.scale == v.scale

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero."""
        v = Vector2D.zero()
        v_normalized = v.normalize()

        assert v_normalized.x == 0.0
        assert v_normalized.y == 0.0

    def test_dot_product(self):
        """Test dot product calculation."""
        v1 = Vector2D.from_4k(1.0, 0.0)
        v2 = Vector2D.from_4k(0.0, 1.0)
        assert v1.dot(v2) == 0.0

        v3 = Vector2D.from_4k(2.0, 3.0)
        v4 = Vector2D.from_4k(4.0, 5.0)
        assert v3.dot(v4) == 23.0  # 2*4 + 3*5

    def test_cross_product(self):
        """Test 2D cross product (returns scalar)."""
        v1 = Vector2D.from_4k(1.0, 0.0)
        v2 = Vector2D.from_4k(0.0, 1.0)
        assert v1.cross(v2) == 1.0

    def test_distance_to(self):
        """Test distance calculation."""
        v1 = Vector2D.from_4k(0.0, 0.0)
        v2 = Vector2D.from_4k(3.0, 4.0)
        assert v1.distance_to(v2) == 5.0

    def test_angle_to(self):
        """Test angle calculation."""
        origin = Vector2D.from_4k(0.0, 0.0)
        right = Vector2D.from_4k(1.0, 0.0)
        up = Vector2D.from_4k(0.0, 1.0)

        assert abs(origin.angle_to(right)) < 1e-10
        assert abs(origin.angle_to(up) - math.pi / 2) < 1e-10

    def test_rotate_preserves_scale(self):
        """Test that rotation preserves scale metadata."""
        v = Vector2D.from_resolution(1.0, 0.0, (1920, 1080))
        v_rotated = v.rotate(math.pi / 2)

        assert abs(v_rotated.x) < 1e-10  # Should be ~0
        assert abs(v_rotated.y - 1.0) < 1e-10  # Should be ~1
        assert v_rotated.scale == v.scale

    def test_scale_by_preserves_scale_metadata(self):
        """Test that scale_by preserves scale metadata (not the values)."""
        v = Vector2D.from_resolution(10.0, 20.0, (1920, 1080))
        v_scaled = v.scale_by(2.0)

        assert v_scaled.x == 20.0
        assert v_scaled.y == 40.0
        assert v_scaled.scale == v.scale  # Scale metadata preserved


class TestVector2DOperators:
    """Test operator overloads work correctly with scale."""

    def test_addition_same_scale(self):
        """Test adding vectors with same scale."""
        v1 = Vector2D.from_4k(1.0, 2.0)
        v2 = Vector2D.from_4k(3.0, 4.0)
        v3 = v1 + v2

        assert v3.x == 4.0
        assert v3.y == 6.0
        assert v3.scale == (1.0, 1.0)

    def test_addition_different_scales(self):
        """Test adding vectors with different scales converts to 4K."""
        v1 = Vector2D.from_resolution(10.0, 20.0, (1920, 1080))  # scale = 2.0
        v2 = Vector2D.from_resolution(30.0, 40.0, (1280, 720))  # scale = 3.0

        # Should convert both to 4K, add, then return in v1's scale
        v3 = v1 + v2

        # v1 in 4K: (20, 40)
        # v2 in 4K: (90, 120)
        # Sum in 4K: (110, 160)
        # Back to v1's scale (2.0): (55, 80)
        assert v3.x == 55.0
        assert v3.y == 80.0
        assert v3.scale == v1.scale

    def test_subtraction_same_scale(self):
        """Test subtracting vectors with same scale."""
        v1 = Vector2D.from_4k(5.0, 7.0)
        v2 = Vector2D.from_4k(2.0, 3.0)
        v3 = v1 - v2

        assert v3.x == 3.0
        assert v3.y == 4.0
        assert v3.scale == (1.0, 1.0)

    def test_multiplication_by_scalar(self):
        """Test scalar multiplication."""
        v = Vector2D.from_4k(3.0, 4.0)
        v2 = v * 2.0

        assert v2.x == 6.0
        assert v2.y == 8.0
        assert v2.scale == v.scale

    def test_division_by_scalar(self):
        """Test scalar division."""
        v = Vector2D.from_4k(6.0, 8.0)
        v2 = v / 2.0

        assert v2.x == 3.0
        assert v2.y == 4.0
        assert v2.scale == v.scale

    def test_negation(self):
        """Test vector negation."""
        v = Vector2D.from_4k(3.0, 4.0)
        v_neg = -v

        assert v_neg.x == -3.0
        assert v_neg.y == -4.0
        assert v_neg.scale == v.scale

    def test_equality(self):
        """Test vector equality (based on x, y only)."""
        v1 = Vector2D.from_4k(1.0, 2.0)
        v2 = Vector2D.from_4k(1.0, 2.0)
        v3 = Vector2D.from_4k(1.0, 3.0)

        assert v1 == v2
        assert v1 != v3

    def test_equality_ignores_scale(self):
        """Test that equality ignores scale metadata."""
        v1 = Vector2D(1.0, 2.0, scale=(1.0, 1.0))
        v2 = Vector2D(1.0, 2.0, scale=(2.0, 2.0))

        assert v1 == v2  # Equal despite different scales


class TestVector2DSerialization:
    """Test serialization and deserialization."""

    def test_to_dict_includes_scale(self):
        """Test that to_dict includes scale metadata."""
        v = Vector2D.from_4k(100.0, 200.0)
        d = v.to_dict()

        assert d["x"] == 100.0
        assert d["y"] == 200.0
        assert d["scale"] == [1.0, 1.0]

    def test_to_dict_with_custom_scale(self):
        """Test to_dict with custom scale."""
        v = Vector2D.from_resolution(50.0, 100.0, (1920, 1080))
        d = v.to_dict()

        assert d["x"] == 50.0
        assert d["y"] == 100.0
        assert d["scale"] == [2.0, 2.0]

    def test_from_dict(self):
        """Test creating Vector2D from dictionary."""
        data = {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]}
        v = Vector2D.from_dict(data)

        assert v.x == 100.0
        assert v.y == 200.0
        assert v.scale == (1.0, 1.0)

    def test_from_dict_missing_scale_raises_error(self):
        """Test that from_dict without scale raises error."""
        data = {"x": 100.0, "y": 200.0}
        with pytest.raises(ValueError, match="Scale is required"):
            Vector2D.from_dict(data)

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = Vector2D.from_resolution(123.456, 789.012, (1920, 1080))
        data = original.to_dict()
        restored = Vector2D.from_dict(data)

        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.scale == original.scale


class TestVector2DEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_coordinates(self):
        """Test with very small coordinates."""
        v = Vector2D.from_4k(0.001, 0.002)
        v_4k = v.to_4k_canonical()

        assert v_4k.x == 0.001
        assert v_4k.y == 0.002

    def test_very_large_coordinates(self):
        """Test with very large coordinates."""
        v = Vector2D.from_4k(10000.0, 20000.0)
        v_4k = v.to_4k_canonical()

        assert v_4k.x == 10000.0
        assert v_4k.y == 20000.0

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        v = Vector2D.from_4k(-100.0, -200.0)
        assert v.x == -100.0
        assert v.y == -200.0

    def test_anisotropic_scaling(self):
        """Test with non-uniform (anisotropic) scaling."""
        # Create a resolution that would have different x/y scales
        v = Vector2D.from_resolution(100.0, 100.0, (1920, 1440))
        # Scale should be (2.0, 1.5) for 4K
        assert v.scale[0] == 2.0
        assert v.scale[1] == 1.5

        v_4k = v.to_4k_canonical()
        assert v_4k.x == 200.0  # 100 * 2.0
        assert v_4k.y == 150.0  # 100 * 1.5


class TestVector2DIntegration:
    """Integration tests combining multiple operations."""

    def test_complex_operation_chain(self):
        """Test a chain of operations preserves correctness."""
        # Start with 1080p
        v = Vector2D.from_resolution(100.0, 200.0, (1920, 1080))

        # Normalize
        v_norm = v.normalize()

        # Scale by 10
        v_scaled = v_norm.scale_by(10.0)

        # Rotate 90 degrees
        v_rotated = v_scaled.rotate(math.pi / 2)

        # Convert to 4K
        v_4k = v_rotated.to_4k_canonical()

        # All operations should preserve scale metadata until final conversion
        assert v_4k.scale == (1.0, 1.0)

    def test_vector_math_in_different_resolutions(self):
        """Test that vector math works correctly across resolutions."""
        # Create two vectors in different resolutions
        v1_1080p = Vector2D.from_resolution(100.0, 100.0, (1920, 1080))
        v2_720p = Vector2D.from_resolution(50.0, 50.0, (1280, 720))

        # Add them (should convert to 4K, add, return in v1's resolution)
        v_sum = v1_1080p + v2_720p

        # v1 in 4K: (200, 200)
        # v2 in 4K: (150, 150)
        # Sum in 4K: (350, 350)
        # Back to 1080p scale: (175, 175)
        assert v_sum.x == 175.0
        assert v_sum.y == 175.0
        assert v_sum.scale == v1_1080p.scale
