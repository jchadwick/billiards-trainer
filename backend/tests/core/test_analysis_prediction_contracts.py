"""Behavioral contracts for the deliberately secondary shot-analysis helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from backend.core.analysis.prediction import TrajectoryPredictor
from backend.core.analysis.shot import ShotAnalyzer
from backend.core.constants_4k import PIXELS_PER_METER_REFERENCE
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState, GameState, ShotType, TableState


def _ball(
    ball_id: str,
    x: float,
    y: float,
    *,
    cue: bool = False,
    pocketed: bool = False,
    radius: float = 10.0,
) -> BallState:
    return BallState(
        id=ball_id,
        position=Vector2D.from_4k(x, y),
        is_cue_ball=cue,
        is_pocketed=pocketed,
        radius=radius,
    )


def _state(*balls: BallState, is_break: bool = False) -> GameState:
    return GameState(
        timestamp=1.0,
        frame_number=1,
        balls=list(balls),
        table=TableState(
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
        ),
        is_break=is_break,
    )


def test_predictor_returns_empty_when_there_is_no_active_cue_ball() -> None:
    predictor = TrajectoryPredictor()
    state = _state(_ball("pocketed-cue", 100, 100, cue=True, pocketed=True))

    assert (
        predictor.predict_cue_shot(
            CueState(tip_position=Vector2D.from_4k(0, 0), angle=0), state
        )
        == {}
    )


@pytest.mark.parametrize(
    ("angle", "expected_x", "expected_y"),
    [
        (0, 2.0 * PIXELS_PER_METER_REFERENCE, 0.0),
        (90, 0.0, 2.0 * PIXELS_PER_METER_REFERENCE),
        (180, -2.0 * PIXELS_PER_METER_REFERENCE, 0.0),
        (-90, 0.0, -2.0 * PIXELS_PER_METER_REFERENCE),
    ],
)
def test_predictor_converts_cue_angle_to_canonical_velocity(
    angle: float, expected_x: float, expected_y: float
) -> None:
    velocity = TrajectoryPredictor()._calculate_shot_velocity(angle)

    assert velocity.x == pytest.approx(expected_x, abs=1e-9)
    assert velocity.y == pytest.approx(expected_y, abs=1e-9)
    assert math.isfinite(velocity.magnitude())


def test_predictor_uses_copied_active_balls_and_returns_only_trajectory_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cue_ball = _ball("cue", 100, 100, cue=True)
    target = _ball("target", 300, 100)
    pocketed = _ball("pocketed", 500, 100, pocketed=True)
    state = _state(cue_ball, target, pocketed)
    cue = CueState(tip_position=Vector2D.from_4k(10, 10), angle=0)
    point = Vector2D.from_4k(125, 100)
    received: dict[str, Any] = {}

    def predict(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            trajectories={
                "cue": SimpleNamespace(points=[SimpleNamespace(position=point)])
            }
        )

    predictor = TrajectoryPredictor()
    monkeypatch.setattr(
        predictor.trajectory_calculator, "predict_multiball_cue_shot", predict
    )

    result = predictor.predict_cue_shot(cue, state)

    assert result == {"cue": [point]}
    assert received["cue_state"] is cue
    assert received["ball_state"] is not cue_ball
    assert received["ball_state"].velocity.magnitude() > 0
    assert [ball.id for ball in received["other_balls"]] == ["target"]
    assert received["max_collision_depth"] == 2
    assert cue_ball.velocity.magnitude() == 0


def test_shot_analyzer_requires_an_active_cue_and_an_available_target() -> None:
    analyzer = ShotAnalyzer()

    with pytest.raises(ValueError, match="No cue ball"):
        analyzer.analyze_shot(_state(_ball("object", 300, 100)))

    with pytest.raises(ValueError, match="No target balls"):
        analyzer.analyze_shot(_state(_ball("cue", 100, 100, cue=True)))

    state = _state(_ball("cue", 100, 100, cue=True), _ball("object", 300, 100))
    with pytest.raises(ValueError, match="Target ball missing not found"):
        analyzer.analyze_shot(state, target_ball="missing")


def test_shot_analyzer_identifies_break_and_direct_shots() -> None:
    cue, target = _ball("cue", 100, 100, cue=True), _ball("target", 300, 100)
    analyzer = ShotAnalyzer()

    break_analysis = analyzer.analyze_shot(_state(cue, target, is_break=True))
    direct_analysis = analyzer.analyze_shot(_state(cue, target))

    assert break_analysis.shot_type is ShotType.BREAK
    assert break_analysis.target_ball_id == "target"
    assert break_analysis.is_legal
    assert break_analysis.illegal_reasons == []
    assert direct_analysis.shot_type is ShotType.DIRECT


def test_shot_analyzer_detects_an_active_intervening_ball_only() -> None:
    cue, target = _ball("cue", 100, 100, cue=True), _ball("target", 300, 100)
    analyzer = ShotAnalyzer()

    blocked = analyzer.analyze_shot(_state(cue, target, _ball("blocker", 200, 100)))
    clear = analyzer.analyze_shot(
        _state(cue, target, _ball("pocketed-blocker", 200, 100, pocketed=True))
    )

    assert blocked.shot_type is ShotType.COMBINATION
    assert clear.shot_type is ShotType.DIRECT
