"""Behavioral contracts for the shared billiards math helpers."""

from __future__ import annotations

import math
from typing import Any

import pytest

import backend.core.utils.math as math_utils
from backend.core.coordinates import Vector2D
from backend.core.utils.math import MathUtils


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "expected"),
    [(-2.0, 0.0, 1.0, 0.0), (0.4, 0.0, 1.0, 0.4), (3.0, 0.0, 1.0, 1.0)],
)
def test_module_clamp_and_interpolation_helpers(
    value: float, minimum: float, maximum: float, expected: float
) -> None:
    assert math_utils.clamp(value, minimum, maximum) == expected
    assert MathUtils.clamp(value, minimum, maximum) == expected
    assert math_utils.lerp(2.0, 6.0, 0.25) == 3.0
    assert MathUtils.lerp(2.0, 6.0, 0.25) == 3.0


@pytest.mark.unit
def test_angle_helpers_cover_wrapping_and_custom_ranges() -> None:
    assert math_utils.normalize_angle(-10.0) == 350.0
    assert math_utils.normalize_angle(360.0) == 0.0
    assert MathUtils.degrees_to_radians(180.0) == pytest.approx(math.pi)
    assert MathUtils.radians_to_degrees(math.pi) == pytest.approx(180.0)
    assert MathUtils.wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert MathUtils.wrap_angle(-1.0, 0.0, 2 * math.pi) == pytest.approx(
        2 * math.pi - 1.0
    )
    assert MathUtils.normalize_angle_0_2pi(-math.pi / 2) == pytest.approx(
        3 * math.pi / 2
    )
    assert MathUtils.angle_difference(math.pi - 0.1, -math.pi + 0.1) == pytest.approx(
        0.2
    )


@pytest.mark.unit
def test_math_config_defaults_are_read_for_tolerances_and_angle_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ConfigStub:
        def get(self, key: str, default: Any = None) -> Any:
            return {
                "core.utils.math.tolerance.default": 0.02,
                "core.utils.math.tolerance.zero_threshold": 0.03,
                "core.utils.math.angles.wrap_min_radians": -2.0,
                "core.utils.math.angles.wrap_max_radians": 2.0,
            }.get(key, default)

    monkeypatch.setattr(math_utils, "_get_config", ConfigStub)
    assert MathUtils.approximately_equal(1.0, 1.01)
    assert MathUtils.approximately_zero(0.02)
    assert MathUtils.wrap_angle(5.0) == pytest.approx(1.0)
    assert MathUtils.approximately_equal(1.0, 1.1, tolerance=0.2)
    assert MathUtils.approximately_zero(-0.1, tolerance=0.2)


@pytest.mark.unit
def test_scalar_math_and_polynomial_helpers_cover_boundary_cases() -> None:
    assert MathUtils.smoothstep(0.0, 1.0, -1.0) == 0.0
    assert MathUtils.smoothstep(0.0, 1.0, 0.5) == pytest.approx(0.5)
    assert MathUtils.smoothstep(0.0, 1.0, 2.0) == 1.0
    assert [MathUtils.sign(value) for value in (-2.0, 0.0, 2.0)] == [-1, 0, 1]
    assert MathUtils.calculate_polynomial([], 10.0) == 0.0
    assert MathUtils.calculate_polynomial([2.0, 3.0, 4.0], 2.0) == 18.0
    assert MathUtils.moving_average([1.0, 2.0, 3.0], 3) == [1.5, 2.0, 2.5]
    original = [1.0, 2.0]
    unchanged = MathUtils.moving_average(original, 0)
    assert unchanged == original
    assert unchanged is not original
    assert MathUtils.moving_average(original, 4) == original
    assert MathUtils.gaussian(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert MathUtils.sigmoid(0.0) == 0.5
    assert MathUtils.exponential_decay(10.0, 2.0, 0.5, 2.0) == pytest.approx(
        2.0 + 8.0 * math.exp(-1.0)
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("coefficients", "expected"),
    [
        ((0.0, 2.0, -4.0), [2.0]),
        ((0.0, 0.0, 3.0), []),
        ((1.0, 0.0, 1.0), []),
        ((1.0, -2.0, 1.0), [1.0]),
        ((1.0, -3.0, 2.0), [1.0, 2.0]),
    ],
)
def test_quadratic_solver_handles_linear_complex_repeated_and_two_roots(
    coefficients: tuple[float, float, float], expected: list[float]
) -> None:
    assert MathUtils.solve_quadratic(*coefficients) == pytest.approx(expected)


@pytest.mark.unit
def test_remapping_and_numerical_methods_cover_degenerate_and_convergent_paths() -> (
    None
):
    assert MathUtils.remap(5.0, 0.0, 10.0, 0.0, 100.0) == 50.0
    assert MathUtils.remap(5.0, 1.0, 1.0, 7.0, 9.0) == 7.0
    assert MathUtils.find_roots_bisection(
        lambda value: value * value - 2, 0, 2
    ) == pytest.approx(math.sqrt(2), abs=1e-6)
    assert MathUtils.find_roots_bisection(lambda value: value * value + 1, 0, 2) is None
    assert MathUtils.numerical_derivative(
        lambda value: value**2, 3.0, h=1e-4
    ) == pytest.approx(6.0, rel=1e-7)
    assert MathUtils.numerical_integration_trapezoidal(
        lambda value: value, 0.0, 1.0, n=100
    ) == pytest.approx(0.5)
    assert MathUtils.numerical_integration_trapezoidal(
        lambda value: value, 0.0, 1.0
    ) == pytest.approx(0.5)


@pytest.mark.unit
def test_vector_trajectory_and_collision_geometry_preserve_scale_metadata() -> None:
    position = Vector2D(2.0, 3.0, scale=(2.0, 3.0))
    velocity = Vector2D(4.0, 5.0, scale=(2.0, 3.0))
    point = MathUtils.calculate_trajectory_parabola(position, velocity, 10.0, 0.5)
    assert (point.x, point.y, point.scale) == (4.0, 4.25, (2.0, 3.0))

    right = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
    up = Vector2D(0.0, 1.0, scale=(1.0, 1.0))
    assert (
        MathUtils.rotate_vector_2d(right, math.pi / 2).x,
        MathUtils.rotate_vector_2d(right, math.pi / 2).y,
    ) == pytest.approx((0.0, 1.0))
    reflected = MathUtils.reflect_vector_2d(Vector2D(2.0, -1.0, (1.0, 1.0)), up)
    assert (reflected.x, reflected.y) == pytest.approx((2.0, 1.0))
    projected = MathUtils.project_vector_onto_line(
        Vector2D(3.0, 4.0, (1.0, 1.0)), right
    )
    assert (projected.x, projected.y) == pytest.approx((3.0, 0.0))
    perpendicular = MathUtils.perpendicular_vector_2d(Vector2D(3.0, 4.0, (2.0, 2.0)))
    assert (perpendicular.x, perpendicular.y, perpendicular.scale) == (
        -4.0,
        3.0,
        (2.0, 2.0),
    )
    assert MathUtils.calculate_impact_parameter(
        Vector2D(0.0, 2.0, (1.0, 1.0)), right, Vector2D.zero()
    ) == pytest.approx(2.0)
    assert MathUtils.is_point_in_circle(
        Vector2D(1.0, 0.0, (1.0, 1.0)), Vector2D.zero(), 1.0
    )
    assert not MathUtils.is_point_in_circle(
        Vector2D(2.0, 0.0, (1.0, 1.0)), Vector2D.zero(), 1.0
    )
    assert MathUtils.circle_line_intersection_times(
        Vector2D(-2.0, 0.0, (1.0, 1.0)), right, Vector2D.zero(), 1.0
    ) == pytest.approx([1.0, 3.0])
    assert (
        MathUtils.circle_line_intersection_times(
            Vector2D(-2.0, 2.0, (1.0, 1.0)), right, Vector2D.zero(), 1.0
        )
        == []
    )


@pytest.mark.unit
def test_force_energy_drag_and_rolling_helpers_cover_stops_and_motion() -> None:
    vector = Vector2D(3.0, 4.0, (1.0, 1.0))
    assert MathUtils.calculate_momentum(2.0, vector).x == 6.0
    assert MathUtils.calculate_kinetic_energy(2.0, vector) == 25.0
    assert MathUtils.calculate_centripetal_force(2.0, 3.0, 0.0) == 0.0
    assert MathUtils.calculate_centripetal_force(2.0, 3.0, 1.5) == 12.0
    assert MathUtils.calculate_angular_velocity(3.0, 0.0) == 0.0
    assert MathUtils.calculate_angular_velocity(6.0, 2.0) == 3.0
    assert MathUtils.calculate_friction_force(10.0, 0.2) == 2.0

    stopped = Vector2D.zero()
    assert MathUtils.apply_drag_force(stopped, 0.5, 1.0) is stopped
    dragged = MathUtils.apply_drag_force(Vector2D(2.0, 0.0, (1.0, 1.0)), 0.1, 0.1)
    assert (dragged.x, dragged.y) == pytest.approx((1.96, 0.0))
    reversed_velocity = MathUtils.apply_drag_force(
        Vector2D(1.0, 0.0, (1.0, 1.0)), 10.0, 1.0
    )
    assert reversed_velocity.magnitude() == 0.0

    no_rolling = MathUtils.calculate_rolling_resistance(
        stopped, mass=2.0, friction_coefficient=0.1, gravity=10.0
    )
    assert no_rolling.magnitude() == 0.0
    resistance = MathUtils.calculate_rolling_resistance(
        vector, mass=2.0, friction_coefficient=0.1, gravity=10.0
    )
    assert (resistance.x, resistance.y) == pytest.approx((-1.2, -1.6))


@pytest.mark.unit
def test_elastic_collision_conserves_momentum_and_obeys_restitution() -> None:
    # Equal masses exchange velocities in a perfectly elastic head-on collision.
    assert MathUtils.elastic_collision_1d(1.0, 1.0, 1.0, 0.0) == pytest.approx(
        (0.0, 1.0)
    )
    # With zero restitution, both bodies leave at their shared center-of-mass speed.
    assert MathUtils.elastic_collision_1d(
        1.0, 1.0, 1.0, 0.0, restitution=0.0
    ) == pytest.approx((0.5, 0.5))
    unequal_result = MathUtils.elastic_collision_1d(2.0, 1.0, 1.0, 0.0)
    assert unequal_result == pytest.approx((1 / 3, 4 / 3))
    assert 2.0 * unequal_result[0] + unequal_result[1] == pytest.approx(2.0)
    assert unequal_result[0] - unequal_result[1] == pytest.approx(-1.0)
    # A zero total mass cannot define a collision response, so preserve inputs.
    assert MathUtils.elastic_collision_1d(1.0, 2.0, -1.0, 4.0) == (2.0, 4.0)


@pytest.mark.unit
def test_weighted_average_handles_valid_and_undefined_inputs() -> None:
    assert MathUtils.weighted_average([10.0, 20.0], [1.0, 3.0]) == 17.5
    assert MathUtils.weighted_average([], []) == 0.0
    assert MathUtils.weighted_average([10.0], []) == 0.0
    assert MathUtils.weighted_average([10.0, 20.0], [1.0, -1.0]) == 0.0
