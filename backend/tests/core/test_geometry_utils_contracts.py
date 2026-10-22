"""Contracts for geometry helpers used by aiming and trajectory features."""

from __future__ import annotations

import math
from typing import Any

import pytest

import backend.core.utils.geometry as geometry_utils
from backend.core.coordinates import Vector2D
from backend.core.utils.geometry import (
    GeometryUtils,
    angle_between_points,
    distance,
    find_ball_cue_is_pointing_at,
    normalize_vector,
    point_in_polygon,
)


def _v(x: float, y: float, scale: tuple[float, float] = (1.0, 1.0)) -> Vector2D:
    return Vector2D(x, y, scale)


@pytest.mark.unit
def test_basic_coordinate_helpers_cover_direction_distance_and_zero_vector() -> None:
    assert angle_between_points(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0)
    assert distance(0.0, 0.0, 3.0, 4.0) == 5.0
    assert normalize_vector(3.0, 4.0) == pytest.approx((0.6, 0.8))
    assert normalize_vector(0.0, 0.0) == (0.0, 0.0)
    assert GeometryUtils.distance_between_points((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert GeometryUtils.distance_between_vectors(_v(0.0, 0.0), _v(3.0, 4.0)) == 5.0
    assert GeometryUtils.angle_between_vectors((1.0, 0.0), (0.0, 1.0)) == pytest.approx(
        math.pi / 2
    )
    assert GeometryUtils.angle_between_vectors((0.0, 0.0), (1.0, 1.0)) == 0.0
    assert GeometryUtils.angle_between_vectors(
        (1.0, 0.0), (-1.0, 0.0)
    ) == pytest.approx(math.pi)
    assert GeometryUtils.angle_from_points(_v(0.0, 0.0), _v(0.0, 1.0)) == pytest.approx(
        math.pi / 2
    )


@pytest.mark.unit
def test_segment_circle_intersections_cover_crossing_tangent_outside_and_point_segments() -> (
    None
):
    center = _v(0.0, 0.0)
    crossing = GeometryUtils.line_circle_intersection(
        _v(-2.0, 0.0, (2.0, 2.0)), _v(2.0, 0.0, (2.0, 2.0)), center, 1.0
    )
    assert [(point.x, point.y) for point in crossing] == pytest.approx(
        [(-1.0, 0.0), (1.0, 0.0)]
    )
    assert all(point.scale == (2.0, 2.0) for point in crossing)
    assert (
        GeometryUtils.line_circle_intersection(_v(-2.0, 2.0), _v(2.0, 2.0), center, 1.0)
        == []
    )
    tangent = GeometryUtils.line_circle_intersection(
        _v(-2.0, 1.0), _v(2.0, 1.0), center, 1.0
    )
    assert [(point.x, point.y) for point in tangent] == pytest.approx([(0.0, 1.0)])
    assert (
        GeometryUtils.line_circle_intersection(_v(2.0, 0.0), _v(3.0, 0.0), center, 1.0)
        == []
    )
    assert GeometryUtils.line_circle_intersection(
        _v(1.0, 0.0), _v(1.0, 0.0), center, 1.0
    ) == [_v(1.0, 0.0)]
    assert (
        GeometryUtils.line_circle_intersection(_v(0.0, 0.0), _v(0.0, 0.0), center, 1.0)
        == []
    )


@pytest.mark.unit
def test_point_to_segment_distance_clamps_projection_and_handles_degenerate_segment() -> (
    None
):
    start, end = _v(0.0, 0.0), _v(4.0, 0.0)
    assert GeometryUtils.point_line_distance(_v(2.0, 3.0), start, end) == 3.0
    assert GeometryUtils.point_line_distance(_v(-2.0, 0.0), start, end) == 2.0
    assert GeometryUtils.point_line_distance(_v(7.0, 0.0), start, end) == 3.0
    assert GeometryUtils.point_line_distance(_v(3.0, 4.0), start, start) == 5.0


@pytest.mark.unit
def test_vector_reflection_rotation_and_angle_helpers_preserve_geometry() -> None:
    reflected = GeometryUtils.reflect_vector(_v(2.0, -1.0), _v(0.0, 1.0))
    assert (reflected.x, reflected.y) == pytest.approx((2.0, 1.0))
    assert reflected.scale == (1.0, 1.0)
    rotated = GeometryUtils.rotate_point(
        _v(2.0, 0.0, (2.0, 2.0)), _v(1.0, 0.0), math.pi / 2
    )
    assert (rotated.x, rotated.y) == pytest.approx((1.0, 1.0))
    assert rotated.scale == (2.0, 2.0)
    assert GeometryUtils.normalize_angle(-math.pi / 2) == pytest.approx(3 * math.pi / 2)
    assert GeometryUtils.normalize_angle(2 * math.pi) == 0.0
    assert GeometryUtils.angle_difference(
        math.pi - 0.1, -math.pi + 0.1
    ) == pytest.approx(0.2)


@pytest.mark.unit
def test_polygon_checks_cover_inside_outside_boundaries_and_empty_shapes() -> None:
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert point_in_polygon(2.0, 2.0, square)
    assert not point_in_polygon(5.0, 2.0, square)
    assert not point_in_polygon(2.0, 2.0, [])
    vector_square = [_v(x, y) for x, y in square]
    assert GeometryUtils.is_point_in_polygon(_v(2.0, 2.0), vector_square)
    assert not GeometryUtils.is_point_in_polygon(_v(5.0, 2.0), vector_square)
    assert not GeometryUtils.is_point_in_polygon(_v(1.0, 1.0), [])


@pytest.mark.unit
def test_circle_circle_intersection_covers_disjoint_contained_identical_and_tangent_cases() -> (
    None
):
    origin = _v(0.0, 0.0)
    assert (
        GeometryUtils.circle_circle_intersection(origin, 1.0, _v(3.0, 0.0), 1.0) == []
    )
    assert (
        GeometryUtils.circle_circle_intersection(origin, 3.0, _v(0.0, 0.0), 1.0) == []
    )
    assert GeometryUtils.circle_circle_intersection(origin, 1.0, origin, 1.0) == []
    crossing = GeometryUtils.circle_circle_intersection(origin, 1.0, _v(1.0, 0.0), 1.0)
    assert len(crossing) == 2
    assert [(point.x, point.y) for point in crossing] == pytest.approx(
        [(0.5, -math.sqrt(0.75)), (0.5, math.sqrt(0.75))]
    )
    tangent = GeometryUtils.circle_circle_intersection(origin, 1.0, _v(2.0, 0.0), 1.0)
    assert len(tangent) == 1
    assert (tangent[0].x, tangent[0].y) == pytest.approx((1.0, 0.0))


@pytest.mark.unit
def test_triangle_bezier_and_path_smoothing_cover_geometry_and_short_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p1, p2, p3 = _v(0.0, 0.0), _v(4.0, 0.0), _v(0.0, 3.0)
    assert GeometryUtils.calculate_triangle_area(p1, p2, p3) == 6.0
    assert GeometryUtils.point_in_triangle(_v(1.0, 1.0), p1, p2, p3)
    assert not GeometryUtils.point_in_triangle(_v(4.0, 4.0), p1, p2, p3)
    assert not GeometryUtils.point_in_triangle(
        _v(0.0, 0.0), _v(0.0, 0.0), _v(0.0, 0.0), _v(0.0, 0.0)
    )
    lerped = GeometryUtils.lerp(p1, p2, 0.25)
    assert (lerped.x, lerped.y) == (1.0, 0.0)
    bezier = GeometryUtils.bezier_curve(
        _v(0.0, 0.0), _v(0.0, 1.0), _v(1.0, 1.0), _v(1.0, 0.0), 0.5
    )
    assert (bezier.x, bezier.y) == pytest.approx((0.5, 0.75))

    short_path = [p1, p2]
    short_result = GeometryUtils.smooth_path(short_path, smoothing_factor=0.5)
    assert short_result == short_path
    assert short_result is not short_path
    path = [_v(0.0, 0.0), _v(3.0, 6.0), _v(6.0, 0.0)]
    unsmoothed = GeometryUtils.smooth_path(path, smoothing_factor=0.0)
    assert [(point.x, point.y) for point in unsmoothed] == [
        (0.0, 0.0),
        (3.0, 6.0),
        (6.0, 0.0),
    ]
    assert GeometryUtils.smooth_path(path)[1].y == pytest.approx(5.6)

    class ConfigStub:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "core.utils.geometry.tolerance.triangle_point_test":
                return 0.0
            if key == "core.utils.geometry.smoothing.default_factor":
                return 1.0
            return default

    monkeypatch.setattr(geometry_utils, "_get_config", ConfigStub)
    assert not GeometryUtils.point_in_triangle(_v(1.0, 1.0), p1, p2, p3)
    assert GeometryUtils.smooth_path(path)[1].y == pytest.approx(2.0)


@pytest.mark.unit
def test_cue_targeting_supports_tuples_vectors_angles_and_skips_invalid_candidates() -> (
    None
):
    balls: list[tuple[float, float] | Vector2D] = [
        (-2.0, 0.0),
        (3.0, 10.0),
        _v(5.0, 1.0),
        (8.0, 0.0),
    ]
    assert (
        find_ball_cue_is_pointing_at(
            (0.0, 0.0), (4.0, 0.0), balls=balls, max_perpendicular_distance=2.0
        )
        == 2
    )
    assert (
        find_ball_cue_is_pointing_at(
            _v(0.0, 0.0), cue_angle=0.0, balls=balls, max_perpendicular_distance=0.5
        )
        == 3
    )
    assert find_ball_cue_is_pointing_at((0.0, 0.0), (0.0, 0.0), balls=balls) is None
    assert find_ball_cue_is_pointing_at((0.0, 0.0), balls=[]) is None
    with pytest.raises(ValueError, match="Either cue_direction or cue_angle"):
        find_ball_cue_is_pointing_at((0.0, 0.0), balls=[(1.0, 0.0)])
