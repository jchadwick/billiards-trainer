"""Regression tests for the public CoreModule prediction and assistance APIs."""

import asyncio

import pytest

from backend.core import CoreModule, CoreModuleConfig
from backend.core.analysis.assistance import AssistanceShotAnalysis
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, GameState, TableState


@pytest.fixture
def calibrated_game_state() -> GameState:
    """Build a deterministic table and shot in canonical pixel space."""
    pockets = [
        Vector2D.from_4k(x, y) for x in (0.0, 1600.0, 3200.0) for y in (0.0, 1600.0)
    ]
    table = TableState(
        width=3200.0,
        height=1600.0,
        pocket_positions=pockets,
        playing_area_corners=[
            Vector2D.from_4k(0.0, 0.0),
            Vector2D.from_4k(3200.0, 0.0),
            Vector2D.from_4k(3200.0, 1600.0),
            Vector2D.from_4k(0.0, 1600.0),
        ],
    )
    balls = [
        BallState(
            id="cue",
            position=Vector2D.from_4k(800.0, 800.0),
            is_cue_ball=True,
        ),
        BallState(
            id="object-1",
            position=Vector2D.from_4k(1600.0, 800.0),
            number=1,
        ),
    ]
    return GameState(timestamp=1.0, frame_number=1, balls=balls, table=table)


@pytest.mark.unit
def test_suggest_shots_returns_geometry_backed_assessments(
    calibrated_game_state: GameState,
) -> None:
    core = CoreModule(CoreModuleConfig(async_processing=False))
    core._current_state = calibrated_game_state

    suggestions = asyncio.run(core.suggest_shots(max_suggestions=2))

    assert suggestions
    assert all(isinstance(item, AssistanceShotAnalysis) for item in suggestions)
    assert all(0.0 <= item.difficulty <= 1.0 for item in suggestions)
    assert all(0.0 <= item.success_probability <= 1.0 for item in suggestions)
    assert all(item.recommended_aim_point.scale == (1.0, 1.0) for item in suggestions)


@pytest.mark.unit
def test_predict_outcomes_uses_copied_canonical_inputs(
    calibrated_game_state: GameState,
) -> None:
    core = CoreModule(CoreModuleConfig(async_processing=False))
    core._current_state = calibrated_game_state
    original_cue = calibrated_game_state.get_cue_ball()
    assert original_cue is not None

    outcomes = asyncio.run(
        core.predict_outcomes(Vector2D.from_4k(100.0, 0.0), num_predictions=1)
    )

    assert len(outcomes) == 1
    assert outcomes[0]["ball_id"] == original_cue.id
    assert outcomes[0]["final_position"]["scale"] == [1.0, 1.0]
    assert outcomes[0]["final_velocity"]["scale"] == [1.0, 1.0]
    assert original_cue.position == Vector2D.from_4k(800.0, 800.0)
    assert original_cue.velocity == Vector2D.zero()


@pytest.mark.unit
def test_predict_outcomes_with_zero_limit_has_no_prediction(
    calibrated_game_state: GameState,
) -> None:
    core = CoreModule(CoreModuleConfig(async_processing=False))
    core._current_state = calibrated_game_state

    outcomes = asyncio.run(
        core.predict_outcomes(Vector2D.from_4k(100.0, 0.0), num_predictions=0)
    )

    assert outcomes == []
