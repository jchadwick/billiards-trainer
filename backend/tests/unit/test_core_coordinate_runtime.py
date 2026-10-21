"""Regression coverage for canonical core geometry and correction behavior."""

import pytest

from backend.core.coordinates import Vector2D
from backend.core.models import BallState, TableState
from backend.core.validation.correction import ErrorCorrector


def _table_with_boundary(corners: list[Vector2D]) -> TableState:
    """Build a calibrated table with canonical dimensions and six pockets."""
    return TableState(
        width=1200.0,
        height=1000.0,
        pocket_positions=[Vector2D.from_4k(index * 200.0, 0.0) for index in range(6)],
        playing_area_corners=corners,
    )


def test_ball_clamp_scales_source_position_and_uses_calibrated_polygon() -> None:
    table = _table_with_boundary(
        [
            Vector2D.from_4k(400.0, 100.0),
            Vector2D.from_4k(1000.0, 400.0),
            Vector2D.from_4k(800.0, 900.0),
            Vector2D.from_4k(200.0, 600.0),
        ]
    )
    corrector = ErrorCorrector(enable_logging=False)
    # This point is (40, 600) in canonical space, well left of both the
    # calibrated polygon and its axis-aligned table dimensions.
    source_position = Vector2D.from_resolution(20.0, 300.0, (1920, 1080))

    clamped = corrector._move_ball_to_table(source_position, 40.0, table)

    assert clamped.scale == (1.0, 1.0)
    assert clamped.x > 200.0
    assert table.is_point_on_table(clamped, ball_radius=40.0)
    assert not table.is_point_on_table(source_position, ball_radius=40.0)


def test_calibrated_polygon_margin_is_perpendicular_pixel_distance() -> None:
    table = _table_with_boundary(
        [
            Vector2D.from_4k(500.0, 100.0),
            Vector2D.from_4k(900.0, 500.0),
            Vector2D.from_4k(500.0, 900.0),
            Vector2D.from_4k(100.0, 500.0),
        ]
    )

    # At 40 pixels below the top vertex, the distance to each sloped side is
    # only about 28 pixels, so a 36-pixel ball cannot fit there.
    assert not table.is_point_in_playing_area(Vector2D.from_4k(500.0, 140.0), 36.0)
    assert table.is_point_in_playing_area(Vector2D.from_4k(500.0, 200.0), 36.0)


def test_coincident_balls_are_separated_in_canonical_coordinates() -> None:
    first = BallState(id="first", position=Vector2D.from_4k(600.0, 500.0))
    second = BallState(id="second", position=Vector2D.from_4k(600.0, 500.0))
    corrector = ErrorCorrector(enable_logging=False)

    corrector._separate_balls(first, second)

    assert first.position.scale == second.position.scale == (1.0, 1.0)
    assert first.distance_to(second) == pytest.approx(first.radius + second.radius)
