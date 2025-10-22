"""Unit tests for 4K coordinate system constants.

Tests verify:
- Constant values are correct
- Table dimensions maintain proper aspect ratio
- Bounds calculations are accurate
- Validation helpers work correctly
"""

import pytest
from core.constants_4k import (
    BALL_DIAMETER_4K,
    BALL_RADIUS_4K,
    CANONICAL_HEIGHT,
    CANONICAL_RESOLUTION,
    CANONICAL_WIDTH,
    CUSHION_WIDTH_4K,
    POCKET_POSITIONS_4K,
    POCKET_RADIUS_4K,
    TABLE_BOTTOM_4K,
    TABLE_CENTER_4K,
    TABLE_HEIGHT_4K,
    TABLE_LEFT_4K,
    TABLE_RIGHT_4K,
    TABLE_TOP_4K,
    TABLE_WIDTH_4K,
    get_table_bounds_4k,
    is_on_table,
    is_valid_4k_coordinate,
)


class TestCanonicalResolution:
    """Test canonical resolution constants."""

    def test_canonical_resolution_tuple(self):
        """Test canonical resolution is 4K UHD."""
        assert CANONICAL_RESOLUTION == (3840, 2160)

    def test_canonical_width(self):
        """Test canonical width is 3840."""
        assert CANONICAL_WIDTH == 3840

    def test_canonical_height(self):
        """Test canonical height is 2160."""
        assert CANONICAL_HEIGHT == 2160

    def test_canonical_aspect_ratio(self):
        """Test canonical resolution maintains 16:9 aspect ratio."""
        aspect_ratio = CANONICAL_WIDTH / CANONICAL_HEIGHT
        assert abs(aspect_ratio - 16 / 9) < 0.0001


class TestTableDimensions:
    """Test table dimension constants."""

    def test_table_width(self):
        """Test table width is 3200 pixels."""
        assert TABLE_WIDTH_4K == 3200

    def test_table_height(self):
        """Test table height is 1600 pixels."""
        assert TABLE_HEIGHT_4K == 1600

    def test_table_aspect_ratio(self):
        """Test table maintains 2:1 aspect ratio."""
        aspect_ratio = TABLE_WIDTH_4K / TABLE_HEIGHT_4K
        assert aspect_ratio == 2.0

    def test_table_center(self):
        """Test table is centered in 4K frame."""
        assert TABLE_CENTER_4K == (1920, 1080)
        assert TABLE_CENTER_4K[0] == CANONICAL_WIDTH // 2
        assert TABLE_CENTER_4K[1] == CANONICAL_HEIGHT // 2


class TestTableBounds:
    """Test table boundary constants."""

    def test_table_left_bound(self):
        """Test left table boundary."""
        expected_left = TABLE_CENTER_4K[0] - TABLE_WIDTH_4K // 2
        assert expected_left == TABLE_LEFT_4K
        assert TABLE_LEFT_4K == 320

    def test_table_top_bound(self):
        """Test top table boundary."""
        expected_top = TABLE_CENTER_4K[1] - TABLE_HEIGHT_4K // 2
        assert expected_top == TABLE_TOP_4K
        assert TABLE_TOP_4K == 280

    def test_table_right_bound(self):
        """Test right table boundary."""
        expected_right = TABLE_LEFT_4K + TABLE_WIDTH_4K
        assert expected_right == TABLE_RIGHT_4K
        assert TABLE_RIGHT_4K == 3520

    def test_table_bottom_bound(self):
        """Test bottom table boundary."""
        expected_bottom = TABLE_TOP_4K + TABLE_HEIGHT_4K
        assert expected_bottom == TABLE_BOTTOM_4K
        assert TABLE_BOTTOM_4K == 1880

    def test_table_fits_in_4k_frame(self):
        """Test table fits within 4K canonical frame."""
        assert 0 <= TABLE_LEFT_4K < CANONICAL_WIDTH
        assert 0 <= TABLE_TOP_4K < CANONICAL_HEIGHT
        assert 0 < TABLE_RIGHT_4K <= CANONICAL_WIDTH
        assert 0 < TABLE_BOTTOM_4K <= CANONICAL_HEIGHT

    def test_table_margins(self):
        """Test table has equal margins on left/right and top/bottom."""
        left_margin = TABLE_LEFT_4K
        right_margin = CANONICAL_WIDTH - TABLE_RIGHT_4K
        top_margin = TABLE_TOP_4K
        bottom_margin = CANONICAL_HEIGHT - TABLE_BOTTOM_4K

        assert left_margin == right_margin == 320
        assert top_margin == bottom_margin == 280


class TestBallDimensions:
    """Test ball dimension constants."""

    def test_ball_radius(self):
        """Test ball radius is 36 pixels."""
        assert BALL_RADIUS_4K == 36

    def test_ball_diameter(self):
        """Test ball diameter is 72 pixels."""
        assert BALL_DIAMETER_4K == 72

    def test_ball_radius_diameter_relationship(self):
        """Test ball radius is half of diameter."""
        assert BALL_RADIUS_4K * 2 == BALL_DIAMETER_4K


class TestPocketDimensions:
    """Test pocket dimension constants."""

    def test_pocket_radius(self):
        """Test pocket radius is 72 pixels."""
        assert POCKET_RADIUS_4K == 72

    def test_pocket_positions_count(self):
        """Test there are 6 pocket positions."""
        assert len(POCKET_POSITIONS_4K) == 6

    def test_pocket_positions_corners(self):
        """Test corner pocket positions are at table corners."""
        # Top-left corner
        assert POCKET_POSITIONS_4K[0] == (TABLE_LEFT_4K, TABLE_TOP_4K)
        # Top-right corner
        assert POCKET_POSITIONS_4K[2] == (TABLE_RIGHT_4K, TABLE_TOP_4K)
        # Bottom-left corner
        assert POCKET_POSITIONS_4K[3] == (TABLE_LEFT_4K, TABLE_BOTTOM_4K)
        # Bottom-right corner
        assert POCKET_POSITIONS_4K[5] == (TABLE_RIGHT_4K, TABLE_BOTTOM_4K)

    def test_pocket_positions_middle(self):
        """Test middle pocket positions are centered on table edges."""
        # Top-middle
        assert POCKET_POSITIONS_4K[1] == (TABLE_CENTER_4K[0], TABLE_TOP_4K)
        # Bottom-middle
        assert POCKET_POSITIONS_4K[4] == (TABLE_CENTER_4K[0], TABLE_BOTTOM_4K)


class TestCushionDimensions:
    """Test cushion dimension constants."""

    def test_cushion_width(self):
        """Test cushion width is 48 pixels."""
        assert CUSHION_WIDTH_4K == 48


class TestValidationHelpers:
    """Test coordinate validation helper functions."""

    def test_is_valid_4k_coordinate_true(self):
        """Test valid 4K coordinates."""
        # Center
        assert is_valid_4k_coordinate(1920, 1080)
        # Origin
        assert is_valid_4k_coordinate(0, 0)
        # Max bounds
        assert is_valid_4k_coordinate(3840, 2160)
        # Table center
        assert is_valid_4k_coordinate(*TABLE_CENTER_4K)

    def test_is_valid_4k_coordinate_false(self):
        """Test invalid 4K coordinates."""
        # Negative coordinates
        assert not is_valid_4k_coordinate(-1, 1080)
        assert not is_valid_4k_coordinate(1920, -1)
        # Out of bounds
        assert not is_valid_4k_coordinate(3841, 1080)
        assert not is_valid_4k_coordinate(1920, 2161)

    def test_is_on_table_true(self):
        """Test coordinates that are on the table."""
        # Table center
        assert is_on_table(*TABLE_CENTER_4K)
        # Table bounds
        assert is_on_table(TABLE_LEFT_4K, TABLE_TOP_4K)
        assert is_on_table(TABLE_RIGHT_4K, TABLE_BOTTOM_4K)
        # Inside table
        assert is_on_table(1920, 1080)

    def test_is_on_table_false(self):
        """Test coordinates that are not on the table."""
        # Outside table bounds
        assert not is_on_table(100, 100, include_cushions=False)
        assert not is_on_table(3700, 2000, include_cushions=False)

    def test_is_on_table_with_cushions(self):
        """Test is_on_table with cushion area included."""
        # Just outside playing surface but within cushions
        cushion_x = TABLE_LEFT_4K - CUSHION_WIDTH_4K // 2
        cushion_y = TABLE_TOP_4K - CUSHION_WIDTH_4K // 2
        assert is_on_table(cushion_x, cushion_y, include_cushions=True)
        assert not is_on_table(cushion_x, cushion_y, include_cushions=False)

    def test_get_table_bounds_without_cushions(self):
        """Test get_table_bounds_4k without cushions."""
        bounds = get_table_bounds_4k(include_cushions=False)
        assert bounds == (TABLE_LEFT_4K, TABLE_TOP_4K, TABLE_RIGHT_4K, TABLE_BOTTOM_4K)
        assert bounds == (320, 280, 3520, 1880)

    def test_get_table_bounds_with_cushions(self):
        """Test get_table_bounds_4k with cushions."""
        bounds = get_table_bounds_4k(include_cushions=True)
        expected = (
            TABLE_LEFT_4K - CUSHION_WIDTH_4K,
            TABLE_TOP_4K - CUSHION_WIDTH_4K,
            TABLE_RIGHT_4K + CUSHION_WIDTH_4K,
            TABLE_BOTTOM_4K + CUSHION_WIDTH_4K,
        )
        assert bounds == expected
        assert bounds == (272, 232, 3568, 1928)

    def test_get_table_bounds_structure(self):
        """Test get_table_bounds_4k returns (left, top, right, bottom)."""
        bounds = get_table_bounds_4k()
        assert len(bounds) == 4
        left, top, right, bottom = bounds
        assert left < right
        assert top < bottom


class TestConstantRelationships:
    """Test relationships between constants."""

    def test_ball_fits_on_table(self):
        """Test ball can fit on table (radius < table dimensions)."""
        assert BALL_RADIUS_4K < TABLE_WIDTH_4K
        assert BALL_RADIUS_4K < TABLE_HEIGHT_4K

    def test_pocket_larger_than_ball(self):
        """Test pocket is larger than ball (can accommodate ball)."""
        assert POCKET_RADIUS_4K > BALL_RADIUS_4K

    def test_table_centered_in_frame(self):
        """Test table is perfectly centered in 4K frame."""
        left_margin = TABLE_LEFT_4K
        right_margin = CANONICAL_WIDTH - TABLE_RIGHT_4K
        top_margin = TABLE_TOP_4K
        bottom_margin = CANONICAL_HEIGHT - TABLE_BOTTOM_4K

        assert left_margin == right_margin
        assert top_margin == bottom_margin
