"""Behavioral contracts for game-state error correction."""

from __future__ import annotations

import math

import pytest

from backend.core.constants_4k import BALL_MASS_KG, BALL_RADIUS_4K
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, GameState, TableState
from backend.core.validation.correction import (
    CorrectionStrategy,
    CorrectionType,
    ErrorCorrector,
)


def make_table(*, corners: list[Vector2D] | None = None) -> TableState:
    return TableState(
        width=1000,
        height=600,
        pocket_positions=[
            Vector2D.from_4k(x, y)
            for x, y in (
                (0, 0),
                (500, 0),
                (1000, 0),
                (0, 600),
                (500, 600),
                (1000, 600),
            )
        ],
        playing_area_corners=corners,
    )


def make_ball(
    ball_id: str,
    x: float,
    y: float,
    *,
    velocity: Vector2D | None = None,
    radius: float = 20,
    pocketed: bool = False,
    cue: bool = False,
    confidence: float = 0.9,
) -> BallState:
    return BallState(
        id=ball_id,
        position=Vector2D.from_4k(x, y),
        velocity=velocity or Vector2D.zero(),
        radius=radius,
        is_pocketed=pocketed,
        is_cue_ball=cue,
        confidence=confidence,
    )


def make_state(*balls: BallState, table: TableState | None = None) -> GameState:
    return GameState(
        timestamp=10.0,
        frame_number=42,
        balls=list(balls),
        table=table or make_table(),
    )


def test_overlapping_ball_correction_separates_copy_and_records_statistics() -> None:
    original = [make_ball("a", 200, 200), make_ball("b", 230, 200)]
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_overlapping_balls(original)

    assert corrected is not original
    assert corrected[0] is not original[0]
    assert corrected[1] is not original[1]
    assert corrected[0].distance_to(corrected[1]) == pytest.approx(40)
    assert original[0].distance_to(original[1]) == pytest.approx(30)
    assert corrector.stats.total_corrections == 1
    assert corrector.stats.corrections_by_type == {CorrectionType.BALL_OVERLAP.value: 1}
    assert corrector.correction_history[0].ball_id == "a,b"


def test_overlap_skips_pocketed_balls_and_non_overlapping_pairs() -> None:
    balls = [
        make_ball("pocketed", 200, 200, pocketed=True),
        make_ball("nearby", 205, 200),
        make_ball("clear", 400, 200),
    ]
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_overlapping_balls(balls)

    assert [ball.position for ball in corrected] == [ball.position for ball in balls]
    assert corrector.stats.total_corrections == 0


def test_overlapping_coincident_centers_use_finite_deterministic_separation() -> None:
    balls = [make_ball("a", 200, 200), make_ball("b", 200, 200)]
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_overlapping_balls(balls)

    assert corrected[0].distance_to(corrected[1]) == pytest.approx(40)
    assert all(math.isfinite(ball.position.x) for ball in corrected)
    assert all(math.isfinite(ball.position.y) for ball in corrected)


def test_out_of_bounds_correction_clamps_ball_center_by_its_radius() -> None:
    ball = make_ball("edge", 5, 300, radius=20)
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_out_of_bounds_balls([ball], make_table())

    assert corrected[0].position.x == pytest.approx(20)
    assert make_table().is_point_on_table(corrected[0].position, corrected[0].radius)
    assert ball.position.x == 5
    assert (
        corrector.correction_history[0].correction_type is CorrectionType.OUT_OF_BOUNDS
    )


def test_out_of_bounds_correction_preserves_pocketed_ball_without_recording() -> None:
    pocketed = make_ball("pocketed", -50, -50, pocketed=True)
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_out_of_bounds_balls([pocketed], make_table())

    assert corrected[0].position == pocketed.position
    assert corrected[0] is not pocketed
    assert corrector.stats.total_corrections == 0


def test_invalid_velocity_is_scaled_to_limit_without_changing_input() -> None:
    ball = make_ball(
        "fast",
        200,
        200,
        velocity=Vector2D.from_4k(600, 800),
        confidence=0.7,
    )
    corrector = ErrorCorrector(
        max_velocity=100,
        enable_logging=False,
        oscillation_prevention=False,
    )

    corrected = corrector.correct_invalid_velocities([ball])

    assert corrected[0].velocity.magnitude() == pytest.approx(100)
    assert corrected[0].velocity.x == pytest.approx(60)
    assert corrected[0].velocity.y == pytest.approx(80)
    assert ball.velocity.magnitude() == pytest.approx(1000)
    assert corrector.correction_history[0].severity == "medium"


@pytest.mark.parametrize("invalid_component", [float("nan"), float("inf")])
def test_non_finite_velocity_is_zeroed(invalid_component: float) -> None:
    velocity = Vector2D.from_4k(1, 0)
    # Model construction correctly rejects invalid vectors. A malformed value
    # can still arrive through mutation or a legacy deserializer, which the
    # corrector's defensive path is intended to sanitize.
    ball = make_ball("bad", 200, 200)
    velocity.x = invalid_component
    ball.velocity = velocity
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_invalid_velocities([ball])

    assert corrected[0].velocity.magnitude() == 0
    assert "NaN/Inf" in corrector.correction_history[0].description


def test_velocity_at_or_below_limit_is_left_unchanged_without_a_record() -> None:
    balls = [
        make_ball("below", 200, 200, velocity=Vector2D.from_4k(30, 40)),
        make_ball("at", 300, 200, velocity=Vector2D.from_4k(60, 80)),
    ]
    corrector = ErrorCorrector(max_velocity=100, enable_logging=False)

    corrected = corrector.correct_invalid_velocities(balls)

    assert [ball.velocity.magnitude() for ball in corrected] == pytest.approx([50, 100])
    assert corrector.stats.total_corrections == 0


def test_oscillation_limit_blocks_fourth_correction_inside_window(monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now():
            class Current:
                @staticmethod
                def timestamp() -> float:
                    return 100.0

            return Current()

    monkeypatch.setattr("backend.core.validation.correction.datetime", FixedDateTime)
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=True)

    assert [corrector._should_apply_correction("ball") for _ in range(4)] == [
        True,
        True,
        True,
        False,
    ]


def test_oscillation_window_discards_old_attempts_before_allowing_new_correction(
    monkeypatch,
) -> None:
    now = [100.0]

    class FixedDateTime:
        @staticmethod
        def now():
            class Current:
                @staticmethod
                def timestamp() -> float:
                    return now[0]

            return Current()

    monkeypatch.setattr("backend.core.validation.correction.datetime", FixedDateTime)
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=True)
    assert [corrector._should_apply_correction("ball") for _ in range(3)] == [
        True,
        True,
        True,
    ]
    now[0] += corrector._oscillation_window + 0.01

    assert corrector._should_apply_correction("ball") is True
    assert corrector._recent_corrections["ball"] == [now[0]]


def test_correct_errors_copies_state_and_adds_a_correction_event() -> None:
    state = make_state(make_ball("a", 200, 200), make_ball("b", 230, 200))
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_errors(
        state, ["Balls a and b are overlapping", "unrecognized validator message"]
    )

    assert corrected is not state
    assert corrected.balls[0].distance_to(corrected.balls[1]) == pytest.approx(40)
    assert state.balls[0].distance_to(state.balls[1]) == pytest.approx(30)
    assert len(corrected.events) == 1
    assert corrected.events[0].event_type == "correction"
    assert (
        corrected.events[0].data["corrections"][0]["correction_type"] == "ball_overlap"
    )
    assert corrector.stats.total_corrections == 1


def test_correct_errors_clamps_the_ball_named_by_an_bounds_error() -> None:
    state = make_state(make_ball("edge", 5, 300, radius=20))
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_errors(state, ["Ball edge is outside table bounds"])

    assert corrected.balls[0].position.x == pytest.approx(20)
    assert state.balls[0].position.x == 5
    assert len(corrector.correction_history) == 1
    assert corrected.events[0].event_type == "correction"


def test_correct_errors_clamps_velocity_and_counts_one_applied_correction() -> None:
    state = make_state(make_ball("fast", 200, 200, velocity=Vector2D.from_4k(300, 400)))
    corrector = ErrorCorrector(
        max_velocity=100,
        enable_logging=False,
        oscillation_prevention=False,
    )

    corrected = corrector.correct_errors(
        state, ["Ball fast velocity 500 exceeds maximum 100 px/s"]
    )

    assert corrected.balls[0].velocity.magnitude() == pytest.approx(100)
    assert state.balls[0].velocity.magnitude() == pytest.approx(500)
    assert len(corrector.correction_history) == 1
    assert corrector.stats.total_corrections == 1


def test_correct_errors_logs_individual_failure_and_continues(
    monkeypatch, caplog
) -> None:
    state = make_state(make_ball("a", 200, 200))
    corrector = ErrorCorrector(enable_logging=True)

    def raise_for_one_error(_state: GameState, error: str) -> list:
        if error == "bad validator message":
            raise RuntimeError("failed correction attempt")
        return []

    monkeypatch.setattr(corrector, "_parse_and_correct_error", raise_for_one_error)

    corrected = corrector.correct_errors(
        state, ["bad validator message", "another validation error"]
    )

    assert corrected is not state
    assert corrected.events == []
    assert "failed to correct error 'bad validator message'" in caplog.text.lower()


def test_correct_errors_handles_duplicate_ids_and_extra_cue_designations() -> None:
    state = make_state(
        make_ball("same", 100, 100, cue=True),
        make_ball("same", 200, 100, cue=True),
    )
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrected = corrector.correct_errors(
        state,
        ["Duplicate ball IDs found", "Expected 1 cue ball, found 2"],
    )

    assert [ball.id for ball in corrected.balls] == ["same", "same_dup_1"]
    assert [ball.is_cue_ball for ball in corrected.balls] == [True, False]
    assert [record.correction_type for record in corrector.correction_history] == [
        CorrectionType.STATE_INCONSISTENCY,
        CorrectionType.STATE_INCONSISTENCY,
    ]
    assert len(corrected.events) == 1


def test_cue_ball_recovery_prefers_active_ball_named_cue() -> None:
    state = make_state(
        make_ball("ordinary", 100, 100),
        make_ball("pocketed-cue", 200, 100, pocketed=True),
        make_ball("Cue Ball", 300, 100),
    )
    corrector = ErrorCorrector(enable_logging=False)

    corrected = corrector.correct_errors(state, ["Expected 1 cue ball, found 0"])

    assert [ball.is_cue_ball for ball in corrected.balls] == [False, False, True]
    assert corrector.correction_history[0].ball_id == "Cue Ball"


def test_cue_ball_recovery_does_not_designate_arbitrary_ball() -> None:
    state = make_state(make_ball("one", 100, 100), make_ball("two", 200, 100))
    corrector = ErrorCorrector(enable_logging=False)

    corrected = corrector.correct_errors(state, ["Expected 1 cue ball, found 0"])

    assert not any(ball.is_cue_ball for ball in corrected.balls)
    assert corrected.events == []
    assert corrector.stats.total_corrections == 0


def test_unparseable_overlap_uses_general_overlap_fallback() -> None:
    state = make_state(make_ball("a", 200, 200), make_ball("b", 230, 200))
    corrector = ErrorCorrector(enable_logging=False)

    corrected = corrector.correct_errors(
        state,
        [
            "overlapping balls without parseable IDs",
            "Ball absent is outside table bounds",
            "Ball absent velocity exceeds maximum",
            "some unrelated validation failure",
        ],
    )

    assert corrected.balls[0].distance_to(corrected.balls[1]) == pytest.approx(40)
    assert len(corrected.events) == 1
    # The fallback records its pair correction in the shared corrector history.
    assert corrector.stats.total_corrections == 1


@pytest.mark.parametrize(
    ("polygon", "expected_error"),
    [
        ([(0, 0), (1, 1)], "at least three"),
        ([(0, 0), (10, 0), (20, 0)], "degenerate"),
        ([(0, 0), (10, 0), (10, 10), (10, 10), (0, 10)], "zero-length"),
    ],
)
def test_invalid_boundary_geometry_is_rejected(polygon, expected_error: str) -> None:
    with pytest.raises(ValueError, match=expected_error):
        ErrorCorrector._inset_convex_polygon(polygon, 1)


def test_polygon_inset_supports_both_windings_and_rejects_ball_too_large() -> None:
    clockwise = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    counterclockwise = list(reversed(clockwise))

    for polygon in (clockwise, counterclockwise):
        inset = ErrorCorrector._inset_convex_polygon(polygon, 10)
        assert ErrorCorrector._point_in_convex_polygon((10, 10), inset)
        assert not ErrorCorrector._point_in_convex_polygon((9, 50), inset)

    with pytest.raises(ValueError, match="too large"):
        ErrorCorrector._inset_convex_polygon(clockwise, 60)


def test_closest_point_projection_clamps_to_segment_endpoints() -> None:
    segment = ((0.0, 0.0), (10.0, 0.0))

    assert ErrorCorrector._closest_point_on_segment((4.0, 3.0), *segment) == (4, 0)
    assert ErrorCorrector._closest_point_on_segment((-4.0, 3.0), *segment) == (0, 0)
    assert ErrorCorrector._closest_point_on_segment((14.0, 3.0), *segment) == (10, 0)
    assert ErrorCorrector._closest_point_on_segment((4, 3), (2, 2), (2, 2)) == (2, 2)


def test_statistics_serialization_frequency_and_reset(monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now():
            class Current:
                @staticmethod
                def timestamp() -> float:
                    return 1000.0

            return Current()

    monkeypatch.setattr("backend.core.validation.correction.datetime", FixedDateTime)
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)
    state = make_state(make_ball("a", 200, 200), make_ball("b", 230, 200))
    corrector.correct_errors(state, ["Balls a and b are overlapping"])

    result = corrector.get_correction_statistics()

    assert result["stats"]["total_corrections"] == 1
    assert result["stats"]["average_confidence"] == pytest.approx(0.9)
    assert result["correction_frequency"] == {"per_minute": 1 / 60, "per_hour": 1}
    assert result["recent_corrections"][0]["correction_type"] == "ball_overlap"
    assert result["oscillation_status"]["active_windows"] == {}

    corrector.reset_statistics()
    assert corrector.get_correction_statistics()["total_history_records"] == 0
    assert corrector.stats.total_corrections == 0


def test_correction_record_serialization_keeps_public_scalar_fields() -> None:
    state = make_state(make_ball("a", 200, 200), make_ball("b", 230, 200))
    corrector = ErrorCorrector(enable_logging=False, oscillation_prevention=False)

    corrector.correct_errors(state, ["Balls a and b are overlapping"])
    record = corrector.correction_history[0].to_dict()

    assert record["correction_type"] == "ball_overlap"
    assert record["ball_id"] == "a,b"
    assert isinstance(record["original_value"], str)
    assert isinstance(record["corrected_value"], str)
    assert record["severity"] == "medium"


def test_set_strategy_updates_active_policy() -> None:
    corrector = ErrorCorrector(enable_logging=False)

    corrector.set_strategy(CorrectionStrategy.IMMEDIATE)

    assert corrector.strategy is CorrectionStrategy.IMMEDIATE
    assert corrector._should_apply_correction("ball") is True


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected"),
    [
        ("mass", -1.0, BALL_MASS_KG),
        ("mass", float("nan"), BALL_MASS_KG),
        ("mass", float("inf"), BALL_MASS_KG),
        ("radius", -1.0, BALL_RADIUS_4K),
        ("radius", float("nan"), BALL_RADIUS_4K),
        ("radius", float("inf"), BALL_RADIUS_4K),
        ("confidence", 1.5, 1.0),
        ("confidence", float("nan"), 0.0),
        ("confidence", float("inf"), 0.0),
    ],
)
def test_physics_violation_correction_repairs_mutated_invalid_ball_fields(
    field: str, invalid_value: float, expected: float
) -> None:
    ball = make_ball("mutated", 200, 200)
    setattr(ball, field, invalid_value)
    state = make_state(ball)
    corrector = ErrorCorrector(enable_logging=False)

    corrected = corrector.correct_physics_violations(state)

    assert getattr(corrected.balls[0], field) == pytest.approx(expected)
    original_value = getattr(state.balls[0], field)
    if math.isnan(invalid_value):
        assert math.isnan(original_value)
    else:
        assert original_value == invalid_value
    assert corrector.stats.total_corrections == 1
    assert (
        corrector.correction_history[0].correction_type
        is CorrectionType.PHYSICS_VIOLATION
    )


def test_invalid_velocity_correction_repairs_missing_velocity() -> None:
    ball = make_ball("missing-velocity", 200, 200)
    ball.velocity = None
    corrector = ErrorCorrector(enable_logging=False)

    corrected = corrector.correct_invalid_velocities([ball])

    assert corrected[0].velocity.magnitude() == 0
    assert ball.velocity is None
    assert corrector.stats.total_corrections == 1
    assert (
        corrector.correction_history[0].correction_type
        is CorrectionType.INVALID_VELOCITY
    )
