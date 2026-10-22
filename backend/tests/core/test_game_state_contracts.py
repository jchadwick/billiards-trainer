"""Lifecycle contracts for authoritative Core game state."""

from __future__ import annotations

from typing import Any

import pytest

from backend.core.coordinates import Vector2D
from backend.core.game_state import GameStateManager
from backend.core.models import BallState, GameType, TableState


def _table() -> TableState:
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D.from_4k(x, y)
            for x, y in (
                (320, 280),
                (1920, 280),
                (3520, 280),
                (320, 1880),
                (1920, 1880),
                (3520, 1880),
            )
        ],
    )


def _ball(x: float, *, pocketed: bool = False) -> BallState:
    return BallState.from_4k(
        "cue",
        x,
        1080,
        number=0,
        is_cue_ball=True,
        is_pocketed=pocketed,
    )


def _observation(*balls: BallState, table: TableState | None = None) -> dict[str, Any]:
    return {"balls": list(balls), "table": table or _table()}


@pytest.fixture
def manager(tmp_path: Any):
    instance = GameStateManager(table=_table(), persistence_path=tmp_path / "state")
    try:
        yield instance
    finally:
        instance._event_manager.stop_enhanced_processing()


def test_failed_observation_retains_authoritative_balls_until_observed_empty(
    manager: GameStateManager,
) -> None:
    initial = manager.update_state(_observation(_ball(1000)))
    retained = manager.update_state(
        {
            **_observation(),
            "observation_status": {"ball_detection_status": "failed"},
        }
    )

    assert len(retained.balls) == 1
    assert retained.balls[0].position == initial.balls[0].position
    assert retained.balls[0].last_update == initial.balls[0].last_update
    assert retained.balls[0] is not initial.balls[0]

    # A successful empty result is authoritative evidence that the ball left
    # view, while a later unchanged detection must not invent motion from the
    # failed frame in between.
    observed_empty = manager.update_state(
        {
            **_observation(),
            "observation_status": {"ball_detection_status": "observed"},
        }
    )
    assert observed_empty.balls == []

    reacquired = manager.update_state(_observation(_ball(1000)))
    assert len(reacquired.balls) == 1
    assert reacquired.balls[0].position == Vector2D.from_4k(1000, 1080)
    assert not any(event.event_type == "ball_motion" for event in reacquired.events)


def test_significant_changes_emit_events_but_small_jitter_does_not(
    manager: GameStateManager,
) -> None:
    published: list[dict[str, Any]] = []
    manager.subscribe_to_events(
        "state_updated", lambda _event_type, payload: published.append(payload)
    )

    first = manager.update_state(_observation(_ball(1000)))
    jitter = manager.update_state(_observation(_ball(1003)))
    moved = manager.update_state(_observation(_ball(1009)))
    pocketed = manager.update_state(_observation(_ball(1010, pocketed=True)))

    assert [item["frame_number"] for item in published] == [1, 3, 4]
    assert first.frame_number == 1
    assert jitter.frame_number == 2
    assert moved.events[0].event_type == "ball_motion"
    assert pocketed.balls[0].is_pocketed is True
    assert any(event.event_type == "ball_pocketed" for event in pocketed.events)
    assert published[-1]["balls_count"] == 1


def test_reset_clears_frame_history_and_motion_baseline(
    manager: GameStateManager,
) -> None:
    reset_events: list[dict[str, Any]] = []
    manager.subscribe_to_events(
        "game_reset", lambda _event_type, payload: reset_events.append(payload)
    )
    manager.update_state(_observation(_ball(1000)))
    manager.update_state(_observation(_ball(1010)))

    manager.reset_game(GameType.NINE_BALL)

    reset_state = manager.get_current_state()
    assert reset_state is not None
    assert reset_state.frame_number == 0
    assert reset_state.balls == []
    assert reset_state.game_type is GameType.NINE_BALL
    assert reset_state.current_player == 1
    assert reset_state.scores == {1: 0, 2: 0}
    assert manager.get_state_history() == []
    assert reset_events[0]["game_type"] == "9ball"

    first_after_reset = manager.update_state(_observation(_ball(1010)))
    assert first_after_reset.frame_number == 1
    assert not any(
        event.event_type == "ball_motion" for event in first_after_reset.events
    )
    assert manager.get_ball_by_id("cue") is first_after_reset.balls[0]
    assert manager.get_cue_ball() is first_after_reset.balls[0]


def test_reset_without_any_calibrated_table_fails_before_mutating_state(
    tmp_path: Any,
) -> None:
    instance = GameStateManager(persistence_path=tmp_path / "uncalibrated")
    try:
        existing = instance.update_state(_observation(_ball(1000)))
        with pytest.raises(ValueError, match="calibrated table"):
            instance.reset_game()

        assert instance.get_current_state() is existing
        assert instance.get_statistics()["current_frame"] == 1
        assert instance.get_state_history() == []
    finally:
        instance._event_manager.stop_enhanced_processing()


def test_save_and_reload_preserve_current_ball_baseline_and_bounded_history(
    tmp_path: Any,
) -> None:
    source = GameStateManager(
        table=_table(), max_history_frames=2, persistence_path=tmp_path / "source"
    )
    restored = GameStateManager(
        table=_table(), max_history_frames=2, persistence_path=tmp_path / "restored"
    )
    try:
        source.update_state(_observation(_ball(1000)))
        source.update_state(_observation(_ball(1010)))
        source.update_state(_observation(_ball(1020)))
        save_path = source.save_state(tmp_path / "snapshot.pkl")

        restored.load_state(save_path)
        assert restored.get_current_state() is not None
        assert restored.get_current_state().frame_number == 3
        assert [state.frame_number for state in restored.get_state_history()] == [1, 2]

        next_state = restored.update_state(_observation(_ball(1030)))
        assert next_state.frame_number == 4
        assert [state.frame_number for state in restored.get_state_history()] == [2, 3]
        assert [event.event_type for event in next_state.events] == ["ball_motion"]
        assert next_state.events[0].data["from_position"]["x"] == 1020
    finally:
        source._event_manager.stop_enhanced_processing()
        restored._event_manager.stop_enhanced_processing()
