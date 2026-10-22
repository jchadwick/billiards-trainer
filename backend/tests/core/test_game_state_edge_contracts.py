"""Edge contracts for Core's authoritative game-state lifecycle.

These tests exercise manager behavior at state transitions and failure
boundaries. Collaborators are kept local and deterministic so failures point
to the manager contract rather than event-worker timing or external files.
"""

from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any

import pytest

from backend.core.coordinates import Vector2D
from backend.core.game_state import GameStateManager, StateValidationError
from backend.core.models import BallState, GameType, TableState

if TYPE_CHECKING:
    from pathlib import Path


def _table(*, offset: float = 0.0) -> TableState:
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D.from_4k(x + offset, y + offset)
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


def _ball(
    ball_id: str,
    x: float,
    *,
    cue: bool = False,
    number: int | None = None,
    pocketed: bool = False,
) -> BallState:
    return BallState.from_4k(
        ball_id,
        x,
        1080,
        number=number,
        is_cue_ball=cue,
        is_pocketed=pocketed,
    )


def _observation(*balls: BallState, table: TableState | None = None) -> dict[str, Any]:
    return {"balls": list(balls), "table": table or _table()}


@pytest.fixture
def manager(tmp_path: Path):
    instance = GameStateManager(table=_table(), persistence_path=tmp_path / "state")
    try:
        yield instance
    finally:
        instance._event_manager.stop_enhanced_processing()


def test_validation_failure_does_not_replace_authoritative_state_or_history(
    manager: GameStateManager,
) -> None:
    first = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    manager.set_validation_config(enabled=True, auto_correct=False)

    with pytest.raises(StateValidationError, match="No active cue ball"):
        manager.update_state(_observation(_ball("object", 1500, number=3)))

    assert manager.get_current_state() is first
    assert manager.get_state_history() == []
    assert manager.get_statistics()["current_balls_count"] == 1


def test_small_ball_overlap_is_corrected_before_state_is_committed(
    manager: GameStateManager,
) -> None:
    state = manager.update_state(
        _observation(
            _ball("cue", 1500, cue=True, number=0),
            _ball("object", 1570, number=3),
        )
    )

    cue = state.get_ball_by_id("cue")
    object_ball = state.get_ball_by_id("object")
    assert cue is not None
    assert object_ball is not None
    assert (
        cue.position.distance_to(object_ball.position) > cue.radius + object_ball.radius
    )
    assert state.is_valid is True
    assert state.validation_errors == []


def test_history_retention_and_requested_window_are_bounded(
    tmp_path: Path,
) -> None:
    instance = GameStateManager(
        table=_table(), max_history_frames=2, persistence_path=tmp_path / "bounded"
    )
    try:
        states = [
            instance.update_state(_observation(_ball("cue", x, cue=True, number=0)))
            for x in (1000, 1010, 1020, 1030)
        ]

        assert [state.frame_number for state in instance.get_state_history()] == [2, 3]
        assert [state.frame_number for state in instance.get_state_history(1)] == [3]
        assert [state.frame_number for state in instance.get_state_history(0)] == [2, 3]
        assert states[-1].frame_number == 4
    finally:
        instance._event_manager.stop_enhanced_processing()


def test_getters_and_statistics_track_active_and_pocketed_balls(
    manager: GameStateManager,
) -> None:
    assert manager.get_ball_by_id("cue") is None
    assert manager.get_cue_ball() is None

    state = manager.update_state(
        _observation(
            _ball("cue", 1000, cue=True, number=0),
            _ball("eight", 1400, number=8, pocketed=True),
        )
    )

    assert manager.get_ball_by_id("cue") is state.balls[0]
    assert manager.get_ball_by_id("missing") is None
    assert manager.get_cue_ball() is state.balls[0]
    assert manager.get_statistics()["current_balls_count"] == 2
    assert manager.get_statistics()["active_balls_count"] == 1


def test_game_type_changes_are_carried_into_the_next_observation(
    manager: GameStateManager,
) -> None:
    manager.set_game_type(GameType.EIGHT_BALL)
    assert manager.get_current_state() is None

    current = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    manager.set_game_type(GameType.EIGHT_BALL)
    assert current.game_type is GameType.EIGHT_BALL
    next_state = manager.update_state(
        _observation(_ball("cue", 1002, cue=True, number=0))
    )

    assert next_state.game_type is GameType.EIGHT_BALL


def test_reset_can_select_an_explicit_calibrated_table_and_game_type(
    manager: GameStateManager,
) -> None:
    manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    chosen_table = _table(offset=12)
    manager.reset_game(GameType.NINE_BALL, table_config=chosen_table)

    state = manager.get_current_state()
    assert state is not None
    assert state.table is chosen_table
    assert state.game_type is GameType.NINE_BALL
    assert state.frame_number == 0
    assert state.is_break is True


def test_cue_angle_change_uses_shortest_path_across_zero_degrees(
    manager: GameStateManager,
) -> None:
    updates: list[dict[str, Any]] = []
    manager.subscribe_to_events(
        "state_updated", lambda _event_type, payload: updates.append(payload)
    )

    def cue_observation(angle: float) -> dict[str, Any]:
        return {
            "table": _table(),
            "balls": [],
            "cue": {
                "tip_position": {"x": 1000, "y": 1000, "scale": [1.0, 1.0]},
                "angle": angle,
            },
        }

    manager.update_state(cue_observation(359.0))
    manager.update_state(cue_observation(0.5))
    manager.update_state(cue_observation(2.5))

    # A 1.5 degree wraparound jitter is below the configured threshold; the
    # following 2 degree change reaches it and publishes a state update.
    assert [update["frame_number"] for update in updates] == [1, 3]


def test_state_event_callback_failure_does_not_abort_state_commit(
    manager: GameStateManager,
) -> None:
    def fail_callback(_event_type: str, _payload: dict[str, Any]) -> None:
        raise RuntimeError("subscriber failed")

    manager.subscribe_to_events("state_updated", fail_callback)
    state = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))

    assert manager.get_current_state() is state
    assert manager.get_statistics()["current_frame"] == 1
    assert manager._event_manager.stats["errors"] == 1


def test_unsubscribe_stops_future_state_notifications(
    manager: GameStateManager,
) -> None:
    notifications: list[int] = []
    subscription = manager.subscribe_to_events(
        "state_updated",
        lambda _event_type, payload: notifications.append(payload["frame_number"]),
    )

    manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    assert manager.unsubscribe(subscription) is True
    assert manager.unsubscribe(subscription) is False
    manager.update_state(_observation(_ball("cue", 1010, cue=True, number=0)))

    assert notifications == [1]


def test_force_validation_and_physics_validation_report_missing_state(
    manager: GameStateManager,
) -> None:
    valid, errors = manager.force_validation()
    physics = manager.validate_with_physics()

    assert valid is False
    assert errors == ["No current state"]
    assert physics.is_valid is False
    assert physics.errors == ["No current state to validate"]


def test_validation_statistics_expose_runtime_configuration(
    manager: GameStateManager,
) -> None:
    manager.set_validation_config(enabled=False, auto_correct=False)
    stats = manager.get_validation_statistics()

    assert stats["validation_enabled"] is False
    assert stats["auto_correct_enabled"] is False
    assert stats["validator_type"] == "StateValidator"
    assert stats["validation_config"]["auto_correction_enabled"] is False


def test_persistence_and_json_export_error_boundaries(
    manager: GameStateManager, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="State file not found"):
        manager.load_state(tmp_path / "missing.pkl")
    corrupt_path = tmp_path / "corrupt.pkl"
    corrupt_path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="Corrupted state file"):
        manager.load_state(corrupt_path)

    with pytest.raises(ValueError, match="No current state"):
        manager.export_state_json(tmp_path / "empty.json")

    state = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    export_path = tmp_path / "current.json"
    manager.export_state_json(export_path)
    exported = json.loads(export_path.read_text())
    assert exported["frame_number"] == state.frame_number
    assert exported["balls"][0]["id"] == "cue"


def test_corrupted_save_is_rejected_without_partially_replacing_live_state(
    manager: GameStateManager, tmp_path: Path
) -> None:
    current = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    original_history = list(manager._state_history)
    original_frame_counter = manager._frame_counter
    original_ball_positions = dict(manager._last_ball_positions)

    other_manager = GameStateManager(table=_table(), max_history_frames=32)
    replacement_state = other_manager.update_state(
        _observation(_ball("cue", 2000, cue=True, number=0))
    )
    malformed_data = {
        "current_state": replacement_state,
        # The remaining required fields intentionally omit state_history.
        "frame_counter": 99,
        "start_time": 1.0,
    }
    malformed_path = tmp_path / "missing-history.pkl"
    malformed_path.write_bytes(pickle.dumps(malformed_data))

    with pytest.raises(ValueError, match="Corrupted state file"):
        manager.load_state(malformed_path)

    assert manager.get_current_state() is current
    assert list(manager._state_history) == original_history
    assert manager._frame_counter == original_frame_counter
    assert manager._last_ball_positions == original_ball_positions


def test_loading_valid_empty_state_clears_cached_ball_positions(
    manager: GameStateManager, tmp_path: Path
) -> None:
    manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    empty_path = tmp_path / "empty-state.pkl"
    empty_path.write_bytes(
        pickle.dumps(
            {
                "current_state": None,
                "state_history": [],
                "frame_counter": 0,
                "start_time": 1.0,
            }
        )
    )

    manager.load_state(empty_path)

    assert manager.get_current_state() is None
    assert manager._last_ball_positions == {}


def test_missing_required_observation_data_leaves_current_state_unchanged(
    manager: GameStateManager,
) -> None:
    first = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))

    with pytest.raises(ValueError, match="requires a nested position"):
        manager.update_state(
            {
                "table": _table(),
                "balls": [{"id": "cue", "velocity": {"x": 0, "y": 0}}],
            }
        )

    assert manager.get_current_state() is first
    assert manager.get_state_history() == []


def test_validation_error_state_is_not_serialized_as_committed_history(
    manager: GameStateManager,
    tmp_path: Path,
) -> None:
    manager.set_validation_config(enabled=True, auto_correct=False)
    with pytest.raises(StateValidationError):
        manager.update_state(_observation(_ball("object", 1500, number=3)))

    save_path = manager.save_state(tmp_path / "rejected.pkl")
    with save_path.open("rb") as saved_file:
        saved = pickle.load(saved_file)

    assert saved["current_state"] is None
    assert saved["state_history"] == []


def test_validation_can_be_disabled_for_explicitly_unvalidated_ingress(
    manager: GameStateManager,
) -> None:
    manager.set_validation_config(enabled=False, auto_correct=False)
    state = manager.update_state(_observation(_ball("object", 1500, number=3)))

    assert state.balls[0].id == "object"
    assert state.is_valid is True
    assert manager.force_validation() == (False, ["No active cue ball found"])


def test_unavailable_detector_status_can_be_an_enum_like_value(
    manager: GameStateManager,
) -> None:
    class Status:
        value = " MODEL_UNAVAILABLE "

    first = manager.update_state(_observation(_ball("cue", 1000, cue=True, number=0)))
    retained = manager.update_state(
        {
            **_observation(),
            "observation_status": {"ball_detection_status": Status()},
        }
    )

    assert len(retained.balls) == 1
    assert retained.balls[0].position == first.balls[0].position
    assert retained.balls[0] is not first.balls[0]
