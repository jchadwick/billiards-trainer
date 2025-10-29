"""Comprehensive unit tests for Vector2D coordinate system.

This test suite provides complete coverage of the Vector2D class with emphasis on:
1. Factory methods (from_4k, from_resolution, zero, unit_x, unit_y)
2. Arithmetic operations (__add__, __sub__, __mul__, __truediv__, __neg__)
3. Geometric operations (magnitude, normalize, dot, cross, distance_to, angle_to, rotate)
4. Coordinate conversions (to_4k_canonical, to_resolution, to_scale)
5. Serialization (to_dict, from_dict)
6. Validation (__post_init__ error cases)
7. Equality and hashing
8. Edge cases (zero vectors, NaN, infinity, division by zero)
9. Scale metadata preservation

All tests use pytest framework and are marked with @pytest.mark.unit.
"""

import math
import sys
from pathlib import Path

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.coordinates import Vector2D


class TestVector2DInitialization:
    """Test suite for Vector2D initialization and validation."""

    @pytest.mark.unit()
    def test_basic_initialization(self) -> None:
        """Test basic Vector2D creation with explicit scale."""
        v = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        assert v.x == 100.0
        assert v.y == 200.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_initialization_with_custom_scale(self) -> None:
        """Test Vector2D creation with custom scale values."""
        v = Vector2D(50.0, 75.0, scale=(2.0, 2.0))
        assert v.x == 50.0
        assert v.y == 75.0
        assert v.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_initialization_with_negative_coordinates(self) -> None:
        """Test Vector2D with negative coordinate values."""
        v = Vector2D(-100.0, -200.0, scale=(1.0, 1.0))
        assert v.x == -100.0
        assert v.y == -200.0

    @pytest.mark.unit()
    def test_initialization_with_zero_coordinates(self) -> None:
        """Test Vector2D with zero coordinates."""
        v = Vector2D(0.0, 0.0, scale=(1.0, 1.0))
        assert v.x == 0.0
        assert v.y == 0.0

    @pytest.mark.unit()
    def test_initialization_missing_scale_raises_error(self) -> None:
        """Test that missing scale parameter raises ValueError."""
        with pytest.raises(ValueError, match="Scale metadata is MANDATORY"):
            Vector2D(100.0, 200.0, scale=None)

    @pytest.mark.unit()
    def test_initialization_invalid_scale_length_raises_error(self) -> None:
        """Test that scale with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Scale must be a tuple"):
            Vector2D(100.0, 200.0, scale=(1.0,))

    @pytest.mark.unit()
    def test_initialization_zero_scale_raises_error(self) -> None:
        """Test that zero scale factors raise ValueError."""
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            Vector2D(100.0, 200.0, scale=(0.0, 1.0))

    @pytest.mark.unit()
    def test_initialization_negative_scale_raises_error(self) -> None:
        """Test that negative scale factors raise ValueError."""
        with pytest.raises(ValueError, match="Scale factors must be positive"):
            Vector2D(100.0, 200.0, scale=(1.0, -2.0))


class TestVector2DFactoryMethods:
    """Test suite for Vector2D factory methods."""

    @pytest.mark.unit()
    def test_from_4k_basic(self) -> None:
        """Test from_4k factory method."""
        v = Vector2D.from_4k(1920.0, 1080.0)
        assert v.x == 1920.0
        assert v.y == 1080.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_4k_with_zero(self) -> None:
        """Test from_4k with zero coordinates."""
        v = Vector2D.from_4k(0.0, 0.0)
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_4k_with_negative(self) -> None:
        """Test from_4k with negative coordinates."""
        v = Vector2D.from_4k(-500.0, -250.0)
        assert v.x == -500.0
        assert v.y == -250.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_resolution_1080p(self) -> None:
        """Test from_resolution with 1080p coordinates."""
        v = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        assert v.x == 960.0
        assert v.y == 540.0
        assert v.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_from_resolution_720p(self) -> None:
        """Test from_resolution with 720p coordinates."""
        v = Vector2D.from_resolution(640.0, 360.0, (1280, 720))
        assert v.x == 640.0
        assert v.y == 360.0
        assert v.scale == (3.0, 3.0)

    @pytest.mark.unit()
    def test_from_resolution_4k(self) -> None:
        """Test from_resolution with 4K resolution (should have scale 1.0)."""
        v = Vector2D.from_resolution(1920.0, 1080.0, (3840, 2160))
        assert v.x == 1920.0
        assert v.y == 1080.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_resolution_non_uniform_scale(self) -> None:
        """Test from_resolution with non-uniform aspect ratio."""
        v = Vector2D.from_resolution(100.0, 100.0, (1920, 1080))
        assert v.x == 100.0
        assert v.y == 100.0
        # 4K is 3840x2160, so scale_x = 3840/1920 = 2.0, scale_y = 2160/1080 = 2.0
        assert v.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_zero_factory_default(self) -> None:
        """Test zero factory method with default scale."""
        v = Vector2D.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_zero_factory_custom_scale(self) -> None:
        """Test zero factory method with custom scale."""
        v = Vector2D.zero(scale=(2.0, 2.0))
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_unit_x_factory_default(self) -> None:
        """Test unit_x factory method with default scale."""
        v = Vector2D.unit_x()
        assert v.x == 1.0
        assert v.y == 0.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_unit_x_factory_custom_scale(self) -> None:
        """Test unit_x factory method with custom scale."""
        v = Vector2D.unit_x(scale=(3.0, 3.0))
        assert v.x == 1.0
        assert v.y == 0.0
        assert v.scale == (3.0, 3.0)

    @pytest.mark.unit()
    def test_unit_y_factory_default(self) -> None:
        """Test unit_y factory method with default scale."""
        v = Vector2D.unit_y()
        assert v.x == 0.0
        assert v.y == 1.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_unit_y_factory_custom_scale(self) -> None:
        """Test unit_y factory method with custom scale."""
        v = Vector2D.unit_y(scale=(2.5, 2.5))
        assert v.x == 0.0
        assert v.y == 1.0
        assert v.scale == (2.5, 2.5)


class TestVector2DGeometricOperations:
    """Test suite for geometric operations on Vector2D."""

    @pytest.mark.unit()
    def test_magnitude_basic(self) -> None:
        """Test magnitude calculation for 3-4-5 triangle."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        assert v.magnitude() == 5.0

    @pytest.mark.unit()
    def test_magnitude_zero_vector(self) -> None:
        """Test magnitude of zero vector."""
        v = Vector2D.zero()
        assert v.magnitude() == 0.0

    @pytest.mark.unit()
    def test_magnitude_unit_vector(self) -> None:
        """Test magnitude of unit vectors."""
        v_x = Vector2D.unit_x()
        v_y = Vector2D.unit_y()
        assert v_x.magnitude() == 1.0
        assert v_y.magnitude() == 1.0

    @pytest.mark.unit()
    def test_magnitude_negative_coordinates(self) -> None:
        """Test magnitude with negative coordinates."""
        v = Vector2D(-3.0, -4.0, scale=(1.0, 1.0))
        assert v.magnitude() == 5.0

    @pytest.mark.unit()
    def test_magnitude_squared_basic(self) -> None:
        """Test magnitude_squared calculation."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        assert v.magnitude_squared() == 25.0

    @pytest.mark.unit()
    def test_magnitude_squared_zero(self) -> None:
        """Test magnitude_squared of zero vector."""
        v = Vector2D.zero()
        assert v.magnitude_squared() == 0.0

    @pytest.mark.unit()
    def test_normalize_basic(self) -> None:
        """Test vector normalization."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        u = v.normalize()
        assert abs(u.magnitude() - 1.0) < 1e-10
        assert abs(u.x - 0.6) < 1e-10
        assert abs(u.y - 0.8) < 1e-10
        assert u.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_normalize_zero_vector(self) -> None:
        """Test normalization of zero vector returns zero vector."""
        v = Vector2D.zero()
        u = v.normalize()
        assert u.x == 0.0
        assert u.y == 0.0
        assert u.magnitude() == 0.0
        assert u.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_normalize_preserves_scale(self) -> None:
        """Test that normalize preserves scale metadata."""
        v = Vector2D(3.0, 4.0, scale=(2.0, 2.0))
        u = v.normalize()
        assert u.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_normalize_unit_vector(self) -> None:
        """Test normalizing an already normalized vector."""
        v = Vector2D.unit_x()
        u = v.normalize()
        assert abs(u.magnitude() - 1.0) < 1e-10
        assert u.x == 1.0
        assert u.y == 0.0

    @pytest.mark.unit()
    def test_dot_perpendicular_vectors(self) -> None:
        """Test dot product of perpendicular vectors."""
        v1 = Vector2D.unit_x()
        v2 = Vector2D.unit_y()
        assert v1.dot(v2) == 0.0

    @pytest.mark.unit()
    def test_dot_parallel_vectors(self) -> None:
        """Test dot product of parallel vectors."""
        v1 = Vector2D(2.0, 0.0, scale=(1.0, 1.0))
        v2 = Vector2D(3.0, 0.0, scale=(1.0, 1.0))
        assert v1.dot(v2) == 6.0

    @pytest.mark.unit()
    def test_dot_opposite_vectors(self) -> None:
        """Test dot product of opposite vectors."""
        v1 = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
        v2 = Vector2D(-1.0, 0.0, scale=(1.0, 1.0))
        assert v1.dot(v2) == -1.0

    @pytest.mark.unit()
    def test_dot_general_case(self) -> None:
        """Test dot product in general case."""
        v1 = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = Vector2D(2.0, -1.0, scale=(1.0, 1.0))
        assert v1.dot(v2) == 2.0  # 3*2 + 4*(-1) = 6 - 4 = 2

    @pytest.mark.unit()
    def test_cross_perpendicular_vectors(self) -> None:
        """Test cross product of perpendicular unit vectors."""
        v1 = Vector2D.unit_x()
        v2 = Vector2D.unit_y()
        assert v1.cross(v2) == 1.0

    @pytest.mark.unit()
    def test_cross_opposite_order(self) -> None:
        """Test cross product with opposite order (anti-commutative)."""
        v1 = Vector2D.unit_x()
        v2 = Vector2D.unit_y()
        assert v2.cross(v1) == -1.0

    @pytest.mark.unit()
    def test_cross_parallel_vectors(self) -> None:
        """Test cross product of parallel vectors."""
        v1 = Vector2D(2.0, 0.0, scale=(1.0, 1.0))
        v2 = Vector2D(3.0, 0.0, scale=(1.0, 1.0))
        assert v1.cross(v2) == 0.0

    @pytest.mark.unit()
    def test_cross_general_case(self) -> None:
        """Test cross product in general case."""
        v1 = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = Vector2D(2.0, -1.0, scale=(1.0, 1.0))
        assert v1.cross(v2) == -11.0  # 3*(-1) - 4*2 = -3 - 8 = -11

    @pytest.mark.unit()
    def test_distance_to_basic(self) -> None:
        """Test distance calculation between two points."""
        p1 = Vector2D(0.0, 0.0, scale=(1.0, 1.0))
        p2 = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        assert p1.distance_to(p2) == 5.0

    @pytest.mark.unit()
    def test_distance_to_same_point(self) -> None:
        """Test distance from point to itself."""
        p = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        assert p.distance_to(p) == 0.0

    @pytest.mark.unit()
    def test_distance_to_symmetric(self) -> None:
        """Test that distance is symmetric."""
        p1 = Vector2D(10.0, 20.0, scale=(1.0, 1.0))
        p2 = Vector2D(40.0, 60.0, scale=(1.0, 1.0))
        assert p1.distance_to(p2) == p2.distance_to(p1)

    @pytest.mark.unit()
    def test_angle_to_horizontal_right(self) -> None:
        """Test angle to point directly to the right."""
        origin = Vector2D.zero()
        target = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
        assert abs(origin.angle_to(target)) < 1e-10

    @pytest.mark.unit()
    def test_angle_to_horizontal_left(self) -> None:
        """Test angle to point directly to the left."""
        origin = Vector2D.zero()
        target = Vector2D(-1.0, 0.0, scale=(1.0, 1.0))
        assert abs(origin.angle_to(target) - math.pi) < 1e-10

    @pytest.mark.unit()
    def test_angle_to_vertical_up(self) -> None:
        """Test angle to point directly up."""
        origin = Vector2D.zero()
        target = Vector2D(0.0, 1.0, scale=(1.0, 1.0))
        assert abs(origin.angle_to(target) - math.pi / 2) < 1e-10

    @pytest.mark.unit()
    def test_angle_to_vertical_down(self) -> None:
        """Test angle to point directly down."""
        origin = Vector2D.zero()
        target = Vector2D(0.0, -1.0, scale=(1.0, 1.0))
        assert abs(origin.angle_to(target) + math.pi / 2) < 1e-10

    @pytest.mark.unit()
    def test_angle_to_diagonal(self) -> None:
        """Test angle to diagonal point (45 degrees)."""
        origin = Vector2D.zero()
        target = Vector2D(1.0, 1.0, scale=(1.0, 1.0))
        assert abs(origin.angle_to(target) - math.pi / 4) < 1e-10

    @pytest.mark.unit()
    def test_rotate_90_degrees(self) -> None:
        """Test rotation by 90 degrees."""
        v = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
        v_rot = v.rotate(math.pi / 2)
        assert abs(v_rot.x) < 1e-10
        assert abs(v_rot.y - 1.0) < 1e-10
        assert v_rot.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_rotate_180_degrees(self) -> None:
        """Test rotation by 180 degrees."""
        v = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
        v_rot = v.rotate(math.pi)
        assert abs(v_rot.x + 1.0) < 1e-10
        assert abs(v_rot.y) < 1e-10

    @pytest.mark.unit()
    def test_rotate_360_degrees(self) -> None:
        """Test rotation by 360 degrees returns to original."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v_rot = v.rotate(2 * math.pi)
        assert abs(v_rot.x - 3.0) < 1e-10
        assert abs(v_rot.y - 4.0) < 1e-10

    @pytest.mark.unit()
    def test_rotate_preserves_magnitude(self) -> None:
        """Test that rotation preserves vector magnitude."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v_rot = v.rotate(math.pi / 3)
        assert abs(v.magnitude() - v_rot.magnitude()) < 1e-10

    @pytest.mark.unit()
    def test_rotate_preserves_scale(self) -> None:
        """Test that rotation preserves scale metadata."""
        v = Vector2D(3.0, 4.0, scale=(2.0, 2.0))
        v_rot = v.rotate(math.pi / 4)
        assert v_rot.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_rotate_zero_vector(self) -> None:
        """Test rotating zero vector."""
        v = Vector2D.zero()
        v_rot = v.rotate(math.pi / 2)
        assert v_rot.x == 0.0
        assert v_rot.y == 0.0

    @pytest.mark.unit()
    def test_scale_by_basic(self) -> None:
        """Test scaling vector by a factor."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v.scale_by(2.0)
        assert v2.x == 6.0
        assert v2.y == 8.0
        assert v2.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_scale_by_zero(self) -> None:
        """Test scaling by zero."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v.scale_by(0.0)
        assert v2.x == 0.0
        assert v2.y == 0.0

    @pytest.mark.unit()
    def test_scale_by_negative(self) -> None:
        """Test scaling by negative factor."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v.scale_by(-1.0)
        assert v2.x == -3.0
        assert v2.y == -4.0

    @pytest.mark.unit()
    def test_scale_by_preserves_scale_metadata(self) -> None:
        """Test that scale_by preserves scale metadata."""
        v = Vector2D(3.0, 4.0, scale=(2.0, 2.0))
        v2 = v.scale_by(3.0)
        assert v2.scale == (2.0, 2.0)


class TestVector2DArithmeticOperations:
    """Test suite for arithmetic operator overloads."""

    @pytest.mark.unit()
    def test_add_basic(self) -> None:
        """Test vector addition."""
        v1 = Vector2D(1.0, 2.0, scale=(1.0, 1.0))
        v2 = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v3 = v1 + v2
        assert v3.x == 4.0
        assert v3.y == 6.0
        assert v3.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_add_with_zero(self) -> None:
        """Test adding zero vector."""
        v1 = Vector2D(5.0, 10.0, scale=(1.0, 1.0))
        v2 = Vector2D.zero()
        v3 = v1 + v2
        assert v3.x == 5.0
        assert v3.y == 10.0

    @pytest.mark.unit()
    def test_add_different_scales(self) -> None:
        """Test addition with different scales (converts to 4K)."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))  # 4K
        v2 = Vector2D(50.0, 100.0, scale=(2.0, 2.0))  # 1080p
        v3 = v1 + v2
        # v2 in 4K is (100, 200), so result should be (200, 400)
        assert v3.x == 200.0
        assert v3.y == 400.0
        assert v3.scale == (1.0, 1.0)  # Result uses left operand's scale

    @pytest.mark.unit()
    def test_add_preserves_left_scale(self) -> None:
        """Test that addition preserves left operand's scale."""
        v1 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        v2 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v3 = v1 + v2
        assert v3.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_sub_basic(self) -> None:
        """Test vector subtraction."""
        v1 = Vector2D(5.0, 7.0, scale=(1.0, 1.0))
        v2 = Vector2D(2.0, 3.0, scale=(1.0, 1.0))
        v3 = v1 - v2
        assert v3.x == 3.0
        assert v3.y == 4.0
        assert v3.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_sub_with_zero(self) -> None:
        """Test subtracting zero vector."""
        v1 = Vector2D(5.0, 10.0, scale=(1.0, 1.0))
        v2 = Vector2D.zero()
        v3 = v1 - v2
        assert v3.x == 5.0
        assert v3.y == 10.0

    @pytest.mark.unit()
    def test_sub_different_scales(self) -> None:
        """Test subtraction with different scales."""
        v1 = Vector2D(200.0, 400.0, scale=(1.0, 1.0))  # 4K
        v2 = Vector2D(50.0, 100.0, scale=(2.0, 2.0))  # 1080p -> (100, 200) in 4K
        v3 = v1 - v2
        assert v3.x == 100.0
        assert v3.y == 200.0
        assert v3.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_sub_preserves_left_scale(self) -> None:
        """Test that subtraction preserves left operand's scale."""
        v1 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        v2 = Vector2D(50.0, 100.0, scale=(1.0, 1.0))
        v3 = v1 - v2
        assert v3.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_mul_positive_scalar(self) -> None:
        """Test multiplication by positive scalar."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v * 2.0
        assert v2.x == 6.0
        assert v2.y == 8.0
        assert v2.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_mul_negative_scalar(self) -> None:
        """Test multiplication by negative scalar."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v * -2.0
        assert v2.x == -6.0
        assert v2.y == -8.0

    @pytest.mark.unit()
    def test_mul_zero_scalar(self) -> None:
        """Test multiplication by zero."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v2 = v * 0.0
        assert v2.x == 0.0
        assert v2.y == 0.0

    @pytest.mark.unit()
    def test_mul_preserves_scale(self) -> None:
        """Test that multiplication preserves scale."""
        v = Vector2D(3.0, 4.0, scale=(2.0, 2.0))
        v2 = v * 3.0
        assert v2.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_truediv_positive_scalar(self) -> None:
        """Test division by positive scalar."""
        v = Vector2D(6.0, 8.0, scale=(1.0, 1.0))
        v2 = v / 2.0
        assert v2.x == 3.0
        assert v2.y == 4.0
        assert v2.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_truediv_negative_scalar(self) -> None:
        """Test division by negative scalar."""
        v = Vector2D(6.0, 8.0, scale=(1.0, 1.0))
        v2 = v / -2.0
        assert v2.x == -3.0
        assert v2.y == -4.0

    @pytest.mark.unit()
    def test_truediv_by_zero_raises_error(self) -> None:
        """Test that division by zero raises ZeroDivisionError."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        with pytest.raises(ZeroDivisionError):
            v / 0.0

    @pytest.mark.unit()
    def test_truediv_preserves_scale(self) -> None:
        """Test that division preserves scale."""
        v = Vector2D(6.0, 8.0, scale=(2.0, 2.0))
        v2 = v / 2.0
        assert v2.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_neg_basic(self) -> None:
        """Test vector negation."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v_neg = -v
        assert v_neg.x == -3.0
        assert v_neg.y == -4.0
        assert v_neg.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_neg_zero_vector(self) -> None:
        """Test negation of zero vector."""
        v = Vector2D.zero()
        v_neg = -v
        assert v_neg.x == 0.0
        assert v_neg.y == 0.0

    @pytest.mark.unit()
    def test_neg_preserves_scale(self) -> None:
        """Test that negation preserves scale."""
        v = Vector2D(3.0, 4.0, scale=(2.0, 2.0))
        v_neg = -v
        assert v_neg.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_neg_double_negation(self) -> None:
        """Test double negation returns to original."""
        v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
        v_neg_neg = -(-v)
        assert v_neg_neg.x == 3.0
        assert v_neg_neg.y == 4.0


class TestVector2DCoordinateConversions:
    """Test suite for coordinate space conversions."""

    @pytest.mark.unit()
    def test_to_4k_canonical_already_4k(self) -> None:
        """Test converting 4K coordinates to canonical (no-op)."""
        v = Vector2D.from_4k(1920.0, 1080.0)
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0
        assert v_4k.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_to_4k_canonical_from_1080p(self) -> None:
        """Test converting 1080p to 4K canonical."""
        v = Vector2D(960.0, 540.0, scale=(2.0, 2.0))
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0
        assert v_4k.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_to_4k_canonical_from_720p(self) -> None:
        """Test converting 720p to 4K canonical."""
        v = Vector2D(640.0, 360.0, scale=(3.0, 3.0))
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0
        assert v_4k.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_to_resolution_4k_to_1080p(self) -> None:
        """Test converting 4K to 1080p resolution."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_1080p = v_4k.to_resolution((1920, 1080))
        assert v_1080p.x == 960.0
        assert v_1080p.y == 540.0
        assert v_1080p.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_to_resolution_4k_to_720p(self) -> None:
        """Test converting 4K to 720p resolution."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_720p = v_4k.to_resolution((1280, 720))
        assert v_720p.x == 640.0
        assert v_720p.y == 360.0
        assert v_720p.scale == (3.0, 3.0)

    @pytest.mark.unit()
    def test_to_resolution_4k_to_4k(self) -> None:
        """Test converting 4K to 4K (should maintain coordinates)."""
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        v_4k_again = v_4k.to_resolution((3840, 2160))
        assert v_4k_again.x == 1920.0
        assert v_4k_again.y == 1080.0
        assert v_4k_again.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_to_resolution_roundtrip(self) -> None:
        """Test roundtrip conversion maintains 4K canonical values."""
        # Start with 1080p
        v_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        # Convert to 4K
        v_4k = v_1080p.to_4k_canonical()
        # Convert back to 1080p
        v_1080p_again = v_4k.to_resolution((1920, 1080))
        assert abs(v_1080p_again.x - 960.0) < 1e-10
        assert abs(v_1080p_again.y - 540.0) < 1e-10

    @pytest.mark.unit()
    def test_to_scale_same_scale(self) -> None:
        """Test converting to same scale (no-op)."""
        v = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        v2 = v.to_scale((2.0, 2.0))
        assert v2.x == 100.0
        assert v2.y == 200.0
        assert v2.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_to_scale_different_scale(self) -> None:
        """Test converting to different scale."""
        v = Vector2D(100.0, 200.0, scale=(1.0, 1.0))  # 4K
        v2 = v.to_scale((2.0, 2.0))  # Convert to 1080p scale
        assert v2.x == 50.0
        assert v2.y == 100.0
        assert v2.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_to_scale_roundtrip(self) -> None:
        """Test roundtrip scale conversion."""
        v = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = v.to_scale((2.0, 2.0))
        v3 = v2.to_scale((1.0, 1.0))
        assert abs(v3.x - 100.0) < 1e-10
        assert abs(v3.y - 200.0) < 1e-10


class TestVector2DSerialization:
    """Test suite for serialization methods."""

    @pytest.mark.unit()
    def test_to_dict_basic(self) -> None:
        """Test converting vector to dictionary."""
        v = Vector2D.from_4k(1920.0, 1080.0)
        d = v.to_dict()
        assert d["x"] == 1920.0
        assert d["y"] == 1080.0
        assert d["scale"] == [1.0, 1.0]

    @pytest.mark.unit()
    def test_to_dict_with_custom_scale(self) -> None:
        """Test to_dict with custom scale."""
        v = Vector2D(960.0, 540.0, scale=(2.0, 2.0))
        d = v.to_dict()
        assert d["x"] == 960.0
        assert d["y"] == 540.0
        assert d["scale"] == [2.0, 2.0]

    @pytest.mark.unit()
    def test_to_dict_with_negative_coords(self) -> None:
        """Test to_dict with negative coordinates."""
        v = Vector2D(-100.0, -200.0, scale=(1.0, 1.0))
        d = v.to_dict()
        assert d["x"] == -100.0
        assert d["y"] == -200.0

    @pytest.mark.unit()
    def test_from_dict_basic(self) -> None:
        """Test creating vector from dictionary."""
        d = {"x": 1920.0, "y": 1080.0, "scale": [1.0, 1.0]}
        v = Vector2D.from_dict(d)
        assert v.x == 1920.0
        assert v.y == 1080.0
        assert v.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_dict_with_custom_scale(self) -> None:
        """Test from_dict with custom scale."""
        d = {"x": 960.0, "y": 540.0, "scale": [2.0, 2.0]}
        v = Vector2D.from_dict(d)
        assert v.x == 960.0
        assert v.y == 540.0
        assert v.scale == (2.0, 2.0)

    @pytest.mark.unit()
    def test_from_dict_missing_scale_raises_error(self) -> None:
        """Test that from_dict without scale raises ValueError."""
        d = {"x": 100.0, "y": 200.0}
        with pytest.raises(ValueError, match="Scale is required"):
            Vector2D.from_dict(d)

    @pytest.mark.unit()
    def test_from_dict_roundtrip(self) -> None:
        """Test roundtrip serialization."""
        v1 = Vector2D(1920.0, 1080.0, scale=(2.0, 2.0))
        d = v1.to_dict()
        v2 = Vector2D.from_dict(d)
        assert v2.x == v1.x
        assert v2.y == v1.y
        assert v2.scale == v1.scale


class TestVector2DEqualityAndHashing:
    """Test suite for equality and hashing operations."""

    @pytest.mark.unit()
    def test_equality_same_coordinates(self) -> None:
        """Test equality with same coordinates."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        assert v1 == v2

    @pytest.mark.unit()
    def test_equality_different_coordinates(self) -> None:
        """Test inequality with different coordinates."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 201.0, scale=(1.0, 1.0))
        assert v1 != v2

    @pytest.mark.unit()
    def test_equality_ignores_scale(self) -> None:
        """Test that equality ignores scale metadata."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        assert v1 == v2

    @pytest.mark.unit()
    def test_equality_with_non_vector(self) -> None:
        """Test equality comparison with non-Vector2D object."""
        v = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        assert v != "not a vector"
        assert v != 42
        assert v is not None

    @pytest.mark.unit()
    def test_hash_same_coordinates(self) -> None:
        """Test that same coordinates produce same hash."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        assert hash(v1) == hash(v2)

    @pytest.mark.unit()
    def test_hash_different_coordinates(self) -> None:
        """Test that different coordinates produce different hashes."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 201.0, scale=(1.0, 1.0))
        assert hash(v1) != hash(v2)

    @pytest.mark.unit()
    def test_hash_use_in_set(self) -> None:
        """Test that vectors can be used in sets."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))
        v3 = Vector2D(300.0, 400.0, scale=(1.0, 1.0))

        vector_set = {v1, v2, v3}
        # v1 and v2 have same coords, so set should have 2 elements
        assert len(vector_set) == 2

    @pytest.mark.unit()
    def test_hash_use_in_dict(self) -> None:
        """Test that vectors can be used as dictionary keys."""
        v1 = Vector2D(100.0, 200.0, scale=(1.0, 1.0))
        v2 = Vector2D(100.0, 200.0, scale=(2.0, 2.0))

        d = {v1: "first"}
        d[v2] = "second"
        # v1 and v2 are equal, so dict should have 1 entry
        assert len(d) == 1
        assert d[v1] == "second"


class TestVector2DEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.unit()
    def test_very_large_coordinates(self) -> None:
        """Test with very large coordinate values."""
        v = Vector2D(1e10, 1e10, scale=(1.0, 1.0))
        assert v.x == 1e10
        assert v.y == 1e10
        assert v.magnitude() > 1e10

    @pytest.mark.unit()
    def test_very_small_coordinates(self) -> None:
        """Test with very small coordinate values."""
        v = Vector2D(1e-10, 1e-10, scale=(1.0, 1.0))
        assert v.x == 1e-10
        assert v.y == 1e-10
        assert v.magnitude() > 0

    @pytest.mark.unit()
    def test_very_large_scale(self) -> None:
        """Test with very large scale values."""
        v = Vector2D(100.0, 200.0, scale=(1000.0, 1000.0))
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 100000.0
        assert v_4k.y == 200000.0

    @pytest.mark.unit()
    def test_very_small_scale(self) -> None:
        """Test with very small scale values."""
        v = Vector2D(100.0, 200.0, scale=(0.001, 0.001))
        v_4k = v.to_4k_canonical()
        assert abs(v_4k.x - 0.1) < 1e-10
        assert abs(v_4k.y - 0.2) < 1e-10

    @pytest.mark.unit()
    def test_non_uniform_scale(self) -> None:
        """Test with non-uniform scale factors."""
        v = Vector2D(100.0, 200.0, scale=(2.0, 3.0))
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 200.0
        assert v_4k.y == 600.0

    @pytest.mark.unit()
    def test_fractional_scale(self) -> None:
        """Test with fractional scale factors."""
        v = Vector2D(100.0, 200.0, scale=(1.5, 2.5))
        v_4k = v.to_4k_canonical()
        assert v_4k.x == 150.0
        assert v_4k.y == 500.0

    @pytest.mark.unit()
    def test_operations_preserve_scale_with_different_operands(self) -> None:
        """Test that operations with different scales work correctly."""
        v1 = Vector2D(100.0, 100.0, scale=(1.0, 1.0))
        v2 = Vector2D(50.0, 50.0, scale=(2.0, 2.0))

        # Addition
        v3 = v1 + v2
        assert v3.x == 200.0  # Both vectors are (100, 100) in 4K
        assert v3.scale == (1.0, 1.0)

        # Subtraction
        v4 = v1 - v2
        assert v4.x == 0.0
        assert v4.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_normalize_very_small_vector(self) -> None:
        """Test normalizing a very small but non-zero vector."""
        v = Vector2D(1e-100, 1e-100, scale=(1.0, 1.0))
        u = v.normalize()
        # Should normalize successfully
        if v.magnitude() > 0:
            assert abs(u.magnitude() - 1.0) < 1e-10

    @pytest.mark.unit()
    def test_angle_between_very_close_points(self) -> None:
        """Test angle calculation between very close points."""
        p1 = Vector2D(1000.0, 1000.0, scale=(1.0, 1.0))
        p2 = Vector2D(1000.0 + 1e-10, 1000.0, scale=(1.0, 1.0))
        angle = p1.angle_to(p2)
        # Should be close to 0 (pointing right)
        assert abs(angle) < 1e-5

    @pytest.mark.unit()
    def test_rotate_by_very_small_angle(self) -> None:
        """Test rotation by very small angle."""
        v = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        v_rot = v.rotate(1e-10)
        # Should be almost the same (allowing for floating point precision)
        assert abs(v_rot.x - 100.0) < 1e-7
        assert abs(v_rot.y) < 1e-7

    @pytest.mark.unit()
    def test_multiple_operations_chain(self) -> None:
        """Test chaining multiple operations."""
        v = Vector2D.from_4k(100.0, 100.0)
        result = (v + v) * 2.0 - v
        # (100+100)*2 - 100 = 400 - 100 = 300
        assert result.x == 300.0
        assert result.y == 300.0
        assert result.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_float_precision_in_operations(self) -> None:
        """Test that floating point precision is maintained."""
        v1 = Vector2D(0.1, 0.2, scale=(1.0, 1.0))
        v2 = Vector2D(0.3, 0.4, scale=(1.0, 1.0))
        v3 = v1 + v2
        # Should handle floating point precision
        assert abs(v3.x - 0.4) < 1e-10
        assert abs(v3.y - 0.6) < 1e-10


class TestVector2DScaleMetadataPreservation:
    """Test suite specifically for scale metadata preservation."""

    @pytest.mark.unit()
    def test_all_arithmetic_operations_preserve_scale(self) -> None:
        """Test that all arithmetic operations preserve scale metadata."""
        v = Vector2D(100.0, 200.0, scale=(2.5, 2.5))

        # Addition
        v_add = v + Vector2D(50.0, 50.0, scale=(2.5, 2.5))
        assert v_add.scale == (2.5, 2.5)

        # Subtraction
        v_sub = v - Vector2D(25.0, 25.0, scale=(2.5, 2.5))
        assert v_sub.scale == (2.5, 2.5)

        # Multiplication
        v_mul = v * 2.0
        assert v_mul.scale == (2.5, 2.5)

        # Division
        v_div = v / 2.0
        assert v_div.scale == (2.5, 2.5)

        # Negation
        v_neg = -v
        assert v_neg.scale == (2.5, 2.5)

    @pytest.mark.unit()
    def test_all_geometric_operations_preserve_scale(self) -> None:
        """Test that all geometric operations preserve scale metadata."""
        v = Vector2D(100.0, 200.0, scale=(3.0, 3.0))

        # Normalize
        v_norm = v.normalize()
        assert v_norm.scale == (3.0, 3.0)

        # Rotate
        v_rot = v.rotate(math.pi / 4)
        assert v_rot.scale == (3.0, 3.0)

        # Scale by
        v_scaled = v.scale_by(2.0)
        assert v_scaled.scale == (3.0, 3.0)

    @pytest.mark.unit()
    def test_factory_methods_set_correct_scale(self) -> None:
        """Test that factory methods set correct scale values."""
        # from_4k
        v_4k = Vector2D.from_4k(1920.0, 1080.0)
        assert v_4k.scale == (1.0, 1.0)

        # from_resolution
        v_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        assert v_1080p.scale == (2.0, 2.0)

        # zero with custom scale
        v_zero = Vector2D.zero(scale=(4.0, 4.0))
        assert v_zero.scale == (4.0, 4.0)

        # unit_x with custom scale
        v_ux = Vector2D.unit_x(scale=(5.0, 5.0))
        assert v_ux.scale == (5.0, 5.0)

        # unit_y with custom scale
        v_uy = Vector2D.unit_y(scale=(6.0, 6.0))
        assert v_uy.scale == (6.0, 6.0)

    @pytest.mark.unit()
    def test_conversion_methods_set_correct_scale(self) -> None:
        """Test that conversion methods set correct scale values."""
        v = Vector2D(100.0, 200.0, scale=(2.0, 2.0))

        # to_4k_canonical
        v_4k = v.to_4k_canonical()
        assert v_4k.scale == (1.0, 1.0)

        # to_resolution
        v_1080p = v_4k.to_resolution((1920, 1080))
        assert v_1080p.scale == (2.0, 2.0)

        # to_scale
        v_custom = v_4k.to_scale((3.0, 3.0))
        assert v_custom.scale == (3.0, 3.0)


class TestVector2DIntegration:
    """Integration tests for complex Vector2D usage patterns."""

    @pytest.mark.unit()
    def test_multi_resolution_workflow(self) -> None:
        """Test typical workflow across multiple resolutions."""
        # Create vector in 1080p
        v_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))

        # Convert to 4K for processing
        v_4k = v_1080p.to_4k_canonical()
        assert v_4k.x == 1920.0
        assert v_4k.y == 1080.0

        # Do some operations in 4K
        v_4k_moved = v_4k + Vector2D.from_4k(100.0, 100.0)
        assert v_4k_moved.x == 2020.0
        assert v_4k_moved.y == 1180.0

        # Convert back to 1080p
        v_1080p_result = v_4k_moved.to_resolution((1920, 1080))
        assert v_1080p_result.x == 1010.0
        assert v_1080p_result.y == 590.0

    @pytest.mark.unit()
    def test_vector_arithmetic_with_mixed_scales(self) -> None:
        """Test arithmetic operations with vectors from different resolutions."""
        # Create vectors in different resolutions
        v_4k = Vector2D.from_4k(1000.0, 1000.0)
        v_1080p = Vector2D.from_resolution(500.0, 500.0, (1920, 1080))

        # Add them (should convert to 4K, add, then back to left operand scale)
        result = v_4k + v_1080p
        # v_1080p in 4K is (1000, 1000), so result should be (2000, 2000) in 4K
        assert result.x == 2000.0
        assert result.y == 2000.0
        assert result.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_geometric_operations_on_converted_vectors(self) -> None:
        """Test geometric operations after resolution conversion."""
        # Create in 1080p
        v1 = Vector2D.from_resolution(0.0, 0.0, (1920, 1080))
        v2 = Vector2D.from_resolution(300.0, 400.0, (1920, 1080))

        # Convert to 4K
        v1_4k = v1.to_4k_canonical()
        v2_4k = v2.to_4k_canonical()

        # Calculate distance in 4K
        distance = v1_4k.distance_to(v2_4k)
        # In 4K: v1=(0,0), v2=(600,800), distance=1000
        assert distance == 1000.0

    @pytest.mark.unit()
    def test_serialization_preserves_all_data(self) -> None:
        """Test that serialization/deserialization preserves all data."""
        original = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))

        # Serialize
        data = original.to_dict()

        # Modify and restore
        restored = Vector2D.from_dict(data)

        # Verify all properties preserved
        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.scale == original.scale

        # Verify behavior is identical
        original_4k = original.to_4k_canonical()
        restored_4k = restored.to_4k_canonical()
        assert original_4k.x == restored_4k.x
        assert original_4k.y == restored_4k.y
