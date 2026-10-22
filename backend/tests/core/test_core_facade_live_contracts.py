"""Deterministic contracts for CoreModule's live-state facade.

These tests keep camera and network work out of the loop while exercising the
actual Core-to-GameStateManager handoff. They protect the current authoritative
ball set, the state published to consumers, and wrapper lifecycle behavior.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import backend.core as core_api
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, Collision, GameState, GameType, TableState
from backend.core.physics.engine import TrajectoryPoint
from backend.core.validation.physics import ValidationError as PhysicsValidationError
from backend.core.validation.physics import ValidationResult as PhysicsValidationResult


def _table() -> TableState:
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D.from_4k(x, y)
            for x, y in (
                (0, 0),
                (1600, 0),
                (3200, 0),
                (0, 1600),
                (1600, 1600),
                (3200, 1600),
            )
        ],
    )


def _ball(
    ball_id: str = "cue",
    x: float = 1200,
    y: float = 800,
    *,
    velocity: Vector2D | None = None,
    is_cue_ball: bool = True,
) -> BallState:
    return BallState.from_4k(
        ball_id,
        x,
        y,
        number=0 if is_cue_ball else 8,
        velocity=velocity or Vector2D.zero(),
        is_cue_ball=is_cue_ball,
    )


def _game_state(*balls: BallState) -> GameState:
    return GameState(
        timestamp=123.0,
        frame_number=7,
        balls=list(balls),
        table=_table(),
    )


@pytest.fixture
def core_module(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Use the production state manager with per-test persistence and cleanup."""
    original_manager = core_api.GameStateManager
    managers: list[Any] = []

    def create_manager(table: TableState | None = None) -> Any:
        manager = original_manager(
            table=table, persistence_path=tmp_path / f"state-{len(managers)}"
        )
        managers.append(manager)
        return manager

    monkeypatch.setattr(core_api, "GameStateManager", create_manager)
    module = core_api.CoreModule(
        core_api.CoreModuleConfig(
            async_processing=False,
            physics_validation_enabled=False,
            state_history_limit=2,
        )
    )
    try:
        yield module
    finally:
        module.event_manager.stop_enhanced_processing()
        for manager in managers:
            manager._event_manager.stop_enhanced_processing()


def _observation(balls: list[BallState], *, status: str = "observed") -> dict[str, Any]:
    return {
        "balls": balls,
        "table": _table(),
        "observation_status": {"ball_detection_status": status},
    }


@pytest.mark.unit
def test_core_update_preserves_authoritative_balls_until_observed_empty(
    core_module: Any,
) -> None:
    """Failed observations retain balls; a successful empty frame clears them."""
    publications: list[dict[str, Any]] = []
    core_module.subscribe_to_events(
        "state_updated", lambda _event_type, state: publications.append(state)
    )

    first = asyncio.run(core_module.update_state(_observation([_ball(x=900)])))
    failed = asyncio.run(core_module.update_state(_observation([], status="failed")))
    empty = asyncio.run(core_module.update_state(_observation([])))

    assert len(first.balls) == 1
    assert len(failed.balls) == 1
    assert failed.balls[0].position == first.balls[0].position
    assert failed.balls[0] is not first.balls[0]
    assert empty.balls == []
    assert core_module.get_current_state() is empty
    assert [item["frame_number"] for item in publications] == [1, 2, 3]
    assert publications[1]["balls"][0]["position"]["x"] == 900
    assert publications[2]["balls"] == []
    assert all("state" not in item for item in publications)
    assert [state.frame_number for state in core_module.get_state_history()] == [2, 3]
    metrics = core_module.get_performance_metrics()
    assert metrics.total_updates == 3
    assert metrics.errors_count == 0
    assert metrics.last_update_time >= 0


@pytest.mark.unit
def test_core_update_wraps_state_manager_failure_without_publishing_partial_state(
    core_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_observation: dict[str, Any]) -> GameState:
        raise ValueError("invalid source frame")

    monkeypatch.setattr(core_module.state_manager, "update_state", fail)

    with pytest.raises(core_api.CoreModuleError, match="invalid source frame"):
        asyncio.run(core_module.update_state(_observation([_ball()])))

    assert core_module.get_current_state() is None
    assert core_module.get_state_history() == []
    assert core_module.get_performance_metrics().errors_count == 1


@pytest.mark.unit
def test_async_state_wrapper_runs_manager_off_the_calling_thread(
    core_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller_thread = threading.current_thread().name
    observed_threads: list[str] = []
    expected = _game_state(_ball())

    class Manager:
        def update_state(self, _observation: dict[str, Any]) -> GameState:
            observed_threads.append(threading.current_thread().name)
            return expected

    monkeypatch.setattr(core_module, "state_manager", Manager())
    core_module.config.async_processing = True

    result = asyncio.run(core_module._async_update_state({"frame_number": 8}))

    assert result is expected
    assert len(observed_threads) == 1
    assert observed_threads[0] != caller_thread


@pytest.mark.unit
def test_reset_clears_core_history_caches_and_metrics_after_state_manager_reset(
    core_module: Any,
) -> None:
    table = _table()
    core_module.state_manager._default_table = table
    asyncio.run(core_module.update_state(_observation([_ball()])))
    core_module.trajectory_cache.set("trajectory", [Vector2D.from_4k(1, 2)])
    core_module.analysis_cache.set("analysis", object())
    core_module.collision_cache.set("collision", object())
    reset_events: list[dict[str, Any]] = []
    core_module.subscribe_to_events(
        "game_reset", lambda _event_type, event: reset_events.append(event)
    )

    asyncio.run(core_module.reset_game(GameType.NINE_BALL))

    reset_state = core_module.state_manager.get_current_state()
    assert core_module.get_current_state() is None
    assert core_module.get_state_history() == []
    assert reset_state is not None
    assert reset_state.frame_number == 0
    assert reset_state.game_type is GameType.NINE_BALL
    assert core_module.trajectory_cache.get_stats()["size"] == 0
    assert core_module.analysis_cache.get_stats()["size"] == 0
    assert core_module.collision_cache.get_stats()["size"] == 0
    assert core_module.get_performance_metrics().total_updates == 0
    assert reset_events == [{"game_type": GameType.NINE_BALL.value}]


@pytest.mark.unit
def test_basic_state_validation_reports_position_speed_and_overlap_issues(
    core_module: Any,
) -> None:
    core_module._current_state = _game_state(
        _ball(x=-5, velocity=Vector2D.from_4k(60, 0)),
        _ball("object", x=0, y=800, is_cue_ball=False),
    )

    result = asyncio.run(core_module.validate_state())

    assert result["valid"] is False
    assert "Ball cue has invalid position" in result["issues"]
    assert "Ball cue has unrealistic velocity" in result["issues"]
    assert any("are overlapping" in issue for issue in result["issues"])
    assert result["detailed_validation"] is False
    assert result["confidence"] == 0.5


@pytest.mark.unit
def test_physics_state_validation_separates_errors_and_warnings(
    core_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _game_state(_ball())
    core_module._current_state = state
    error = PhysicsValidationError("speed", "critical", "too fast", {})
    soft_issue = PhysicsValidationError("drift", "warning", "minor drift", {})
    validator_warning = PhysicsValidationError("table", "warning", "low confidence", {})
    core_module.physics_validator = SimpleNamespace(
        detailed_validation=True,
        conservation_checks=False,
        energy_tolerance=0.1,
        momentum_tolerance=0.2,
        velocity_tolerance=1.0,
        position_tolerance=2.0,
        max_velocity=100.0,
        max_acceleration=200.0,
        max_spin=50.0,
        max_force=10.0,
        validate_system_state=lambda balls, table: (
            assert_system_state_arguments(balls, table, state.balls, state.table)
            or PhysicsValidationResult(
                False, [error, soft_issue], [validator_warning], 0.25
            )
        ),
    )

    result = asyncio.run(core_module.validate_state())

    assert result == {
        "valid": False,
        "issues": ["speed: too fast"],
        "warnings": ["drift: minor drift", "table: low confidence"],
        "confidence": 0.25,
        "timestamp": state.timestamp,
        "detailed_validation": True,
    }


def assert_system_state_arguments(
    balls: list[BallState],
    table: TableState,
    expected_balls: list[BallState],
    expected_table: TableState,
) -> None:
    assert balls is expected_balls
    assert table is expected_table


@pytest.mark.unit
def test_validation_summary_reflects_enabled_and_disabled_physics_validator(
    core_module: Any,
) -> None:
    core_module.physics_validator = None
    disabled = core_module.get_validation_summary()
    assert disabled == {
        "validation_enabled": False,
        "detailed_validation": False,
        "conservation_checks": False,
        "tolerances": {},
        "limits": {},
    }

    core_module.physics_validator = SimpleNamespace(
        detailed_validation=True,
        conservation_checks=True,
        energy_tolerance=0.05,
        momentum_tolerance=0.01,
        velocity_tolerance=1.0,
        position_tolerance=2.0,
        max_velocity=100.0,
        max_acceleration=200.0,
        max_spin=50.0,
        max_force=10.0,
    )
    enabled = core_module.get_validation_summary()
    assert enabled["validation_enabled"] is True
    assert enabled["tolerances"] == {
        "energy": 0.05,
        "momentum": 0.01,
        "velocity": 1.0,
        "position": 2.0,
    }
    assert enabled["limits"] == {
        "max_velocity": 100.0,
        "max_acceleration": 200.0,
        "max_spin": 50.0,
        "max_force": 10.0,
    }


@pytest.mark.unit
def test_trajectory_wrapper_caches_result_and_publishes_complete_geometry(
    core_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _game_state(_ball())
    core_module._current_state = state
    computed: list[tuple[Any, ...]] = []
    points = [
        TrajectoryPoint(0.0, Vector2D.from_4k(1200, 800), Vector2D.zero()),
        TrajectoryPoint(0.1, Vector2D.from_4k(1210, 800), Vector2D.zero()),
    ]

    class Physics:
        def calculate_trajectory(self, *args: Any) -> list[TrajectoryPoint]:
            computed.append(args)
            return points

    monkeypatch.setattr(core_module, "physics_engine", Physics())
    events: list[dict[str, Any]] = []
    core_module.subscribe_to_events(
        "trajectory_calculated", lambda _event_type, data: events.append(data)
    )

    first = asyncio.run(
        core_module.calculate_trajectory("cue", Vector2D.from_4k(10, 0))
    )
    second = asyncio.run(
        core_module.calculate_trajectory("cue", Vector2D.from_4k(10, 0))
    )

    assert first == [point.position for point in points]
    assert second == first
    assert len(computed) == 1
    assert len(events) == 1
    trajectory = events[0]["trajectory"]
    assert trajectory["ball_id"] == "cue"
    assert len(trajectory["lines"]) == 1
    assert trajectory["lines"][0]["coordinate_unit"] == "canonical_pixels"
    assert trajectory["points"] == [point.to_dict() for point in first]
    assert trajectory["coordinate_space"] == "4k_canonical"


@pytest.mark.unit
def test_trajectory_and_validation_wrappers_reject_missing_or_disabled_inputs(
    core_module: Any,
) -> None:
    with pytest.raises(core_api.CoreModuleError, match="No current game state"):
        asyncio.run(core_module.calculate_trajectory("cue", Vector2D.from_4k(1, 0)))
    with pytest.raises(
        core_api.CoreModuleError, match="Physics validation is disabled"
    ):
        asyncio.run(core_module.validate_trajectory("cue", Vector2D.from_4k(1, 0)))
    with pytest.raises(
        core_api.CoreModuleError, match="Physics validation is disabled"
    ):
        asyncio.run(core_module.validate_conservation_laws([], []))

    core_module._current_state = _game_state(_ball())
    with pytest.raises(core_api.CoreModuleError, match="Ball missing not found"):
        asyncio.run(core_module.calculate_trajectory("missing", Vector2D.from_4k(1, 0)))


@pytest.mark.unit
def test_collision_and_conservation_wrappers_forward_authoritative_ball_states(
    core_module: Any,
) -> None:
    cue = _ball()
    object_ball = _ball("object", 1400, 800, is_cue_ball=False)
    core_module._current_state = _game_state(cue, object_ball)
    collision = Collision(
        time=0.2,
        position=Vector2D.from_4k(1300, 800),
        ball1_id="cue",
        ball2_id="object",
    )
    collision_result = PhysicsValidationResult(True, [], [], 1.0)
    conservation_result = PhysicsValidationResult(True, [], [], 0.9)
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class Validator:
        def validate_collision(self, *args: Any) -> PhysicsValidationResult:
            calls.append(("collision", args))
            return collision_result

        def validate_energy_conservation(self, *args: Any) -> PhysicsValidationResult:
            calls.append(("energy", args))
            return conservation_result

        def validate_momentum_conservation(self, *args: Any) -> PhysicsValidationResult:
            calls.append(("momentum", args))
            return conservation_result

    core_module.physics_validator = Validator()

    assert asyncio.run(core_module.validate_collision(collision, "cue", "object")) is (
        collision_result
    )
    results = asyncio.run(core_module.validate_conservation_laws([cue], [object_ball]))

    assert results == {"energy": conservation_result, "momentum": conservation_result}
    assert calls[0] == ("collision", (collision, cue, object_ball))
    assert calls[1] == ("energy", ([cue], [object_ball]))
    assert calls[2] == ("momentum", ([cue], [object_ball]))
