"""Public game-route contracts, exercised without a running camera or Core.

The game endpoints are intentionally unauthenticated on the local-control API.
These tests pin their state, pagination, reset, and export behavior at the HTTP
boundary while using small deterministic Core fakes for optional capabilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any, NoReturn, cast

import pytest
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_state, get_core_module
from backend.api.routes import game as game_route
from backend.core.coordinates import Vector2D
from backend.core.models import (
    BallState,
    CueState,
    GameEvent,
    GameState,
    GameType,
    TableState,
)
from backend.perception.rust_game_session import (
    RustGameSessionAuthorityRejected,
    RustGameSessionSnapshot,
)
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher


def _table(width: float = 3200, height: float = 1600) -> TableState:
    """Return stable calibrated geometry accepted by game-state converters."""
    return TableState(
        width=width,
        height=height,
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
        playing_area_corners=[
            Vector2D.from_4k(x, y)
            for x, y in ((320, 280), (3520, 280), (3520, 1880), (320, 1880))
        ],
        real_width_meters=2.54,
        real_height_meters=1.27,
    )


def _state(
    *, frame_number: int = 1, timestamp: float = 100.0, table: TableState | None = None
) -> GameState:
    return GameState(
        timestamp=timestamp,
        frame_number=frame_number,
        balls=[
            BallState(
                id="cue",
                position=Vector2D.from_4k(1000 + frame_number, 1000),
                velocity=Vector2D.from_4k(0, 0),
                is_cue_ball=True,
                confidence=0.97,
                last_update=timestamp,
            )
        ],
        table=table or _table(),
        cue=CueState(
            tip_position=Vector2D.from_4k(900, 1000),
            angle=0,
            is_visible=True,
            confidence=0.9,
            last_update=timestamp,
        ),
        game_type=GameType.EIGHT_BALL,
        is_valid=True,
        state_confidence=0.93,
        events=[
            GameEvent(
                timestamp=timestamp,
                event_type="shot",
                description="Cue ball struck",
                data={"ball_id": "cue"},
                frame_number=frame_number,
            )
        ],
    )


def _client(core: Any = None, app_state: Any = None) -> TestClient:
    """Build an isolated app with only the game router and no auth middleware."""
    app = FastAPI()
    app.include_router(game_route.router, prefix="/api/v1")
    app.dependency_overrides[get_core_module] = lambda: core
    app.dependency_overrides[get_app_state] = lambda: app_state or SimpleNamespace(
        core_module=core
    )
    return TestClient(app)


def test_state_is_public_and_uses_latest_core_state_without_authentication() -> None:
    state = _state(frame_number=38)
    core = SimpleNamespace(get_current_state=lambda: state)

    with _client(core) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 200
    assert response.json()["frame_number"] == 38
    assert response.json()["balls"][0]["id"] == "cue"
    assert response.json()["events"][0]["event_type"] == "shot"


@pytest.mark.parametrize(
    ("core", "detail"),
    [
        (None, "Core module not available"),
        (SimpleNamespace(get_current_state=lambda: None), "Game state is unavailable"),
    ],
)
def test_state_reports_unavailable_core_state_as_service_unavailable(
    core: Any, detail: str
) -> None:
    with _client(core) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 503
    assert detail in response.json()["detail"]


def test_state_falls_back_to_manager_and_can_omit_recent_events() -> None:
    state = _state(frame_number=5)
    core = SimpleNamespace(
        get_current_state=lambda: None,
        state_manager=SimpleNamespace(get_current_state=lambda: state),
    )

    with _client(core) as client:
        response = client.get("/api/v1/game/state?include_events=false")

    assert response.status_code == 200
    assert response.json()["frame_number"] == 5
    assert response.json()["events"] == []


def test_scene_snapshot_route_serves_localization_without_initialized_core() -> None:
    class Broadcaster:
        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            return None

    scene_contract = (
        Path(__file__).parents[4]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_contract.read_text(encoding="utf-8"))
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(publisher.publish(scene)) is True

    with _client(
        None,
        SimpleNamespace(core_module=None, scene_state_publisher=publisher),
    ) as client:
        response = client.get("/api/v1/game/state?include_events=false")

    assert response.status_code == 200
    body = response.json()
    assert body["perception_source"] == "rust_scene"
    assert body["core_metadata_status"] == "unavailable"
    assert body["balls"]
    assert body["events"] == []


def test_rust_session_metadata_is_exposed_without_replacing_scene_locations() -> None:
    class Broadcaster:
        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            return None

    class Authority:
        def snapshot(self) -> RustGameSessionSnapshot:
            return RustGameSessionSnapshot(
                schema_version=1,
                revision=12,
                game_type="9ball",
                active_player=2,
                scores=(4, 7),
                is_break=False,
            )

    scene_contract = (
        Path(__file__).parents[4]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(
        publisher.publish(json.loads(scene_contract.read_text(encoding="utf-8")))
    )
    expected_positions = [
        ball["position"] for ball in publisher.read_model.payload.balls
    ]

    with _client(
        None,
        SimpleNamespace(
            core_module=None,
            scene_state_publisher=publisher,
            game_session_authority=Authority(),
        ),
    ) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 200
    body = response.json()
    assert body["game_type"] == "9ball"
    assert body["session"] == {
        "schema_version": 1,
        "revision": 12,
        "game_type": "9ball",
        "active_player": 2,
        "scores": [4, 7],
        "is_break": False,
    }
    assert [ball["position"] for ball in body["balls"]] == expected_positions
    assert body["core_metadata_status"] == "unavailable"


def test_session_metadata_failure_does_not_hide_scene_localization() -> None:
    class Broadcaster:
        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            return None

    class Authority:
        def snapshot(self) -> None:
            raise ValueError("invalid session request bounds")

    scene_contract = (
        Path(__file__).parents[4]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(
        publisher.publish(json.loads(scene_contract.read_text(encoding="utf-8")))
    )

    with _client(
        None,
        SimpleNamespace(
            core_module=None,
            scene_state_publisher=publisher,
            game_session_authority=Authority(),
        ),
    ) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 200
    assert response.json()["balls"]
    assert response.json()["session"] is None
    assert response.json()["perception_source"] == "rust_scene"


def test_rust_session_reset_changes_only_session_and_returns_actual_scene() -> None:
    class Broadcaster:
        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            return None

    class Authority:
        def __init__(self) -> None:
            self.reset_call: tuple[int, str] | None = None

        def snapshot(self) -> RustGameSessionSnapshot:
            return RustGameSessionSnapshot(1, 20, "8ball", 2, (8, 9), False)

        def reset(
            self, *, expected_revision: int, game_type: str
        ) -> RustGameSessionSnapshot:
            self.reset_call = (expected_revision, game_type)
            updated_scene = json.loads(scene_contract.read_text(encoding="utf-8"))
            updated_scene["transport_sequence"] = 2
            updated_scene["snapshot_revision"] = 2
            updated_scene["source"]["frame_id"] = "frame-0002"
            updated_scene["balls"][0]["measurement"]["image_center_px"]["x"] += 50.0
            asyncio.run(publisher.publish(updated_scene))
            return RustGameSessionSnapshot(1, 21, game_type, 1, (0, 0), True)  # type: ignore[arg-type]

    scene_contract = (
        Path(__file__).parents[4]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(
        publisher.publish(json.loads(scene_contract.read_text(encoding="utf-8")))
    )
    expected_positions = [
        ball["position"] for ball in publisher.read_model.payload.balls
    ]
    authority = Authority()

    with _client(
        None,
        SimpleNamespace(
            core_module=None,
            scene_state_publisher=publisher,
            game_session_authority=authority,
        ),
    ) as client:
        response = client.post("/api/v1/game/reset?game_type=9ball")

    assert response.status_code == 200
    body = response.json()
    assert authority.reset_call == (20, "9ball")
    assert body["success"] is True
    assert body["backup_created"] is False
    assert body["new_state"]["perception_source"] == "rust_scene"
    assert body["new_state"]["session"]["revision"] == 21
    assert body["new_state"]["session"]["game_type"] == "9ball"
    assert body["new_state"]["frame_number"] == 2
    assert [
        ball["position"] for ball in body["new_state"]["balls"]
    ] != expected_positions


def test_rust_session_reset_rejects_table_mutation_flag() -> None:
    class Authority:
        def snapshot(self) -> None:
            raise AssertionError("reset must be rejected before authority access")

    with _client(
        None,
        SimpleNamespace(core_module=None, game_session_authority=Authority()),
    ) as client:
        response = client.post("/api/v1/game/reset?preserve_table=false")

    assert response.status_code == 400
    assert "does not alter table calibration" in response.json()["detail"]


def test_rust_session_reset_maps_stale_compare_and_swap_to_conflict() -> None:
    class Authority:
        def snapshot(self) -> RustGameSessionSnapshot:
            return RustGameSessionSnapshot(1, 3, "practice", 1, (0, 0), True)

        def reset(self, **_kwargs: Any) -> None:
            raise RustGameSessionAuthorityRejected(
                "session changed", code="stale_revision", current_revision=4
            )

    with _client(
        None,
        SimpleNamespace(
            core_module=None,
            scene_state_publisher=SimpleNamespace(read_model=object()),
            game_session_authority=Authority(),
        ),
    ) as client:
        response = client.post("/api/v1/game/reset")

    assert response.status_code == 409
    assert "changed during reset" in response.json()["detail"]


def test_rust_session_reset_fails_closed_when_optional_authority_is_missing() -> None:
    with _client(
        None,
        SimpleNamespace(
            core_module=None,
            scene_state_publisher=object(),
            game_session_authority=None,
        ),
    ) as client:
        response = client.post("/api/v1/game/reset")

    assert response.status_code == 503
    assert response.json()["detail"] == "Rust game-session authority is unavailable"


@pytest.mark.parametrize(
    ("status", "scheduler_enabled", "expected"),
    [
        (None, False, True),
        (None, True, False),
        ({"shot_detected": True}, False, False),
        ({"balls_in_motion": 1}, False, False),
        ({"game_state": "shooting"}, False, False),
        ({"phase": "many-balls-moving"}, False, False),
        ({"phase_aware": True}, False, False),
        ({"phase_aware": True}, True, False),
        (
            {
                "phase": "ready",
                "aim_calculation_eligible": False,
                "map_trusted": True,
            },
            True,
            False,
        ),
        ({"aim_calculation_eligible": True, "map_trusted": True}, True, True),
        ({"aim_calculation_eligible": True, "map_trusted": False}, True, False),
        ({"aim_calculation_eligible": True, "map_trusted": True}, False, True),
    ],
)
def test_trajectory_eligibility_respects_motion_and_map_trust(
    status: dict[str, Any] | None, scheduler_enabled: bool, expected: bool
) -> None:
    assert (
        game_route._live_trajectory_is_eligible(status, scheduler_enabled) is expected
    )


def test_route_conversion_helpers_preserve_shared_transport_contract() -> None:
    state = _state()

    ball_info = game_route.convert_ball_state_to_info(state.balls[0])
    cue_info = game_route.convert_cue_state_to_info(state.cue)  # type: ignore[arg-type]
    table_info = game_route.convert_table_state_to_info(state.table)
    response = game_route.convert_game_state_to_response(state)

    assert ball_info.id == "cue"
    assert ball_info.position.x == state.balls[0].position.x
    assert cue_info.tip_position.x == state.cue.tip_position.x  # type: ignore[union-attr]
    assert table_info.width == state.table.width
    assert response.frame_number == state.frame_number


def test_state_only_does_not_run_predictions_and_eligible_request_runs_once() -> None:
    state = _state()
    calls: list[GameState] = []
    core = SimpleNamespace(
        get_current_state=lambda: state,
        predict_trajectories=lambda value: calls.append(value),
    )
    app_state = SimpleNamespace(
        core_module=core,
        integration_service=SimpleNamespace(
            latest_observation_status={
                "aim_calculation_eligible": True,
                "map_trusted": True,
            }
        ),
        config_module=SimpleNamespace(get=lambda _path, default=False: default),
    )

    with _client(core, app_state) as client:
        plain = client.get("/api/v1/game/state")
        requested = client.get("/api/v1/game/state?include_trajectories=true")

    assert plain.status_code == requested.status_code == 200
    assert calls == [state]


def test_prediction_failures_do_not_break_current_state_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = _state()

    def fail_prediction(_state: GameState) -> None:
        raise RuntimeError("prediction worker unavailable")

    core = SimpleNamespace(
        get_current_state=lambda: state,
        predict_trajectories=fail_prediction,
    )
    app_state = SimpleNamespace(
        core_module=core,
        integration_service=SimpleNamespace(
            latest_observation_status={
                "aim_calculation_eligible": True,
                "map_trusted": True,
            }
        ),
    )

    with _client(core, app_state) as client:
        response = client.get("/api/v1/game/state?include_trajectories=true")

    assert response.status_code == 200
    assert response.json()["frame_number"] == state.frame_number
    assert "Failed to get trajectory predictions" in caplog.text


def test_state_conversion_failure_is_sanitized_as_internal_error() -> None:
    class BrokenState:
        timestamp = 1
        frame_number = 1
        balls: list[BallState] = []
        cue = None
        table = None
        game_type = "practice"
        is_valid = True
        state_confidence = 1.0
        events: list[GameEvent] = []

    core = SimpleNamespace(get_current_state=lambda: BrokenState())
    with _client(core) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve current game state"


@pytest.mark.asyncio
async def test_history_forwards_filters_and_returns_actual_page_range() -> None:
    first = _state(frame_number=7, timestamp=100.0)
    second = _state(frame_number=9, timestamp=104.0)
    received: dict[str, Any] = {}

    def get_history(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"states": [first, {"corrupt": True}, second], "total_count": 9}

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    response = await game_route.get_game_history(
        start_time=start,
        end_time=end,
        limit=2,
        offset=3,
        include_events=False,
        game_type="8ball",
        core_module=cast("Any", SimpleNamespace(get_game_history=get_history)),
    )

    assert received == {
        "start_time": start,
        "end_time": end,
        "limit": 2,
        "offset": 3,
        "game_type": "8ball",
    }
    assert [state.frame_number for state in response.states] == [7, 9]
    assert all(state.events == [] for state in response.states)
    assert response.total_count == 9
    assert response.has_more is True
    assert response.time_range == {
        "start": datetime.fromtimestamp(100.0),
        "end": datetime.fromtimestamp(104.0),
    }


@pytest.mark.asyncio
async def test_history_retains_events_when_the_caller_requests_them() -> None:
    state = _state(frame_number=8)
    response = await game_route.get_game_history(
        start_time=None,
        end_time=None,
        limit=10,
        offset=0,
        include_events=True,
        game_type=None,
        core_module=cast(
            "Any",
            SimpleNamespace(get_game_history=lambda **_kwargs: {"states": [state]}),
        ),
    )

    assert len(response.states) == 1
    assert [event.event_type for event in response.states[0].events] == ["shot"]


@pytest.mark.asyncio
async def test_history_rejects_reversed_window_and_handles_missing_or_failing_store() -> (
    None
):
    with pytest.raises(HTTPException) as error:
        await game_route.get_game_history(
            start_time=datetime(2025, 1, 2),
            end_time=datetime(2025, 1, 1),
            limit=10,
            offset=0,
            include_events=True,
            game_type=None,
            core_module=cast("Any", SimpleNamespace()),
        )
    assert error.value.status_code == 400

    missing = await game_route.get_game_history(
        start_time=None,
        end_time=None,
        limit=10,
        offset=0,
        include_events=True,
        game_type=None,
        core_module=cast("Any", SimpleNamespace()),
    )
    assert missing.states == []
    assert missing.total_count == 0
    assert missing.has_more is False

    def fail_history(**_kwargs: Any) -> None:
        raise RuntimeError("history unavailable")

    recovered = await game_route.get_game_history(
        start_time=None,
        end_time=None,
        limit=10,
        offset=0,
        include_events=True,
        game_type=None,
        core_module=cast("Any", SimpleNamespace(get_game_history=fail_history)),
    )
    assert recovered.states == []


@pytest.mark.asyncio
async def test_history_sanitizes_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        game_route,
        "convert_game_state_to_response",
        lambda _state: (_ for _ in ()).throw(RuntimeError("invalid stored state")),
    )

    with pytest.raises(HTTPException) as error:
        await game_route.get_game_history(
            start_time=None,
            end_time=None,
            limit=10,
            offset=0,
            include_events=True,
            game_type=None,
            core_module=cast(
                "Any",
                SimpleNamespace(
                    get_game_history=lambda **_kwargs: {"states": [_state()]}
                ),
            ),
        )

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to retrieve game history"


def test_history_query_validation_is_public_and_enforces_pagination_contract() -> None:
    with _client(SimpleNamespace(get_game_history=lambda **_kwargs: {})) as client:
        invalid_range = client.get("/api/v1/game/history?game_type=snooker")
        invalid_limit = client.get("/api/v1/game/history?limit=0")
        invalid_offset = client.get("/api/v1/game/history?offset=-1")

    assert invalid_range.status_code == 422
    assert invalid_limit.status_code == 422
    assert invalid_offset.status_code == 422


class _ResetCore:
    """Capture reset inputs while emulating Core's retained-state lifecycle."""

    def __init__(self, state: GameState | None, configured_table: TableState | None):
        self.current_state = state
        self.state_manager = SimpleNamespace(
            _default_table=configured_table,
            get_current_state=lambda: self.current_state,
            reset_game=self._manager_reset,
        )
        self.core_reset_types: list[GameType] = []
        self.manager_reset: tuple[GameType, TableState | None] | None = None

    def get_current_state(self) -> GameState | None:
        return self.current_state

    async def reset_game(self, game_type: GameType) -> None:
        self.core_reset_types.append(game_type)
        self.current_state = None

    def _manager_reset(
        self, game_type: GameType, table_config: TableState | None = None
    ) -> None:
        self.manager_reset = (game_type, table_config)
        self.current_state = _state(
            frame_number=0,
            timestamp=500.0,
            table=table_config,
        )
        self.current_state.game_type = game_type


@pytest.mark.asyncio
async def test_reset_preserves_live_table_and_initializes_requested_game_without_backup() -> (
    None
):
    retained_table = _table(width=3190)
    core = _ResetCore(_state(table=retained_table), configured_table=_table(width=2100))
    response = await game_route.reset_game_state(
        game_type="9ball",
        preserve_table=True,
        create_backup=False,
        core_module=core,  # type: ignore[arg-type]
    )

    assert response.success is True
    assert response.backup_created is False
    assert response.new_state.game_type == "9ball"
    assert response.new_state.frame_number == 0
    assert core.core_reset_types == [GameType.NINE_BALL]
    assert core.manager_reset == (GameType.NINE_BALL, retained_table)


@pytest.mark.asyncio
async def test_reset_without_live_state_uses_configured_table() -> None:
    configured_table = _table(width=2500)
    core = _ResetCore(None, configured_table=configured_table)
    response = await game_route.reset_game_state(
        game_type="practice",
        preserve_table=False,
        create_backup=False,
        core_module=core,  # type: ignore[arg-type]
    )

    assert response.new_state.table.width == configured_table.width
    assert core.manager_reset == (GameType.PRACTICE, configured_table)


@pytest.mark.asyncio
async def test_reset_backs_up_current_state_and_continues_when_backup_storage_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core = _ResetCore(_state(frame_number=15), configured_table=_table())
    real_open = open

    def write_backup_in_test_directory(path: str, mode: str) -> IO[Any]:
        return real_open(tmp_path / Path(path).name, mode)

    monkeypatch.setattr(
        game_route, "open", write_backup_in_test_directory, raising=False
    )
    response = await game_route.reset_game_state(
        game_type="straight",
        preserve_table=True,
        create_backup=True,
        core_module=core,  # type: ignore[arg-type]
    )

    backups = list(tmp_path.glob("game_state_backup_*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text())["frame_number"] == 15
    assert response.backup_created is True

    def fail_backup(_path: str, _mode: str) -> NoReturn:
        raise OSError("backup storage unavailable")

    monkeypatch.setattr(game_route, "open", fail_backup, raising=False)
    continued = await game_route.reset_game_state(
        game_type="practice",
        preserve_table=True,
        create_backup=True,
        core_module=_ResetCore(_state(), configured_table=_table()),  # type: ignore[arg-type]
    )
    assert continued.success is True
    assert continued.backup_created is False


@pytest.mark.asyncio
async def test_reset_requires_calibration_and_reports_core_failure() -> None:
    uncalibrated = _ResetCore(None, configured_table=None)
    with pytest.raises(HTTPException) as missing_table:
        await game_route.reset_game_state(
            game_type="practice",
            preserve_table=True,
            create_backup=False,
            core_module=uncalibrated,  # type: ignore[arg-type]
        )
    assert missing_table.value.status_code == 409

    class BrokenReset:
        state_manager = SimpleNamespace(_default_table=_table())

        def get_current_state(self) -> None:
            return None

        async def reset_game(self, _game_type: GameType) -> None:
            raise RuntimeError("reset failed")

    with pytest.raises(HTTPException) as failed_reset:
        await game_route.reset_game_state(
            game_type="practice",
            preserve_table=False,
            create_backup=False,
            core_module=BrokenReset(),  # type: ignore[arg-type]
        )
    assert failed_reset.value.status_code == 500
    assert failed_reset.value.detail == "Failed to reset game state"

    class ResetWithoutSnapshot(_ResetCore):
        def _manager_reset(
            self, game_type: GameType, table_config: TableState | None = None
        ) -> None:
            self.manager_reset = (game_type, table_config)
            self.current_state = None

    with pytest.raises(HTTPException) as missing_reset_state:
        await game_route.reset_game_state(
            game_type="practice",
            preserve_table=False,
            create_backup=False,
            core_module=ResetWithoutSnapshot(None, configured_table=_table()),  # type: ignore[arg-type]
        )
    assert missing_reset_state.value.status_code == 500

    with pytest.raises(HTTPException) as invalid_game_type:
        await game_route.reset_game_state(
            game_type="snooker",
            preserve_table=True,
            create_backup=False,
            core_module=cast("Any", SimpleNamespace()),
        )
    assert invalid_game_type.value.status_code == 400


def test_reset_route_validates_game_type_without_authentication() -> None:
    with _client(_ResetCore(_state(), _table())) as client:
        invalid = client.post("/api/v1/game/reset?game_type=snooker")
        valid = client.post("/api/v1/game/reset?game_type=practice&create_backup=false")

    assert invalid.status_code == 422
    assert valid.status_code == 200
    assert valid.json()["success"] is True


def test_export_json_contains_requested_data_and_verified_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(frame_number=17)
    history_calls: list[str | None] = []

    def get_history(*, session_id: str | None) -> dict[str, Any]:
        history_calls.append(session_id)
        return {"states": [state]}

    core = SimpleNamespace(
        get_current_state=lambda: state,
        get_game_history=get_history,
        get_game_events=lambda *, session_id: [
            {"event": "settled", "session": session_id}
        ],
        get_trajectory_history=lambda *, session_id: [
            {"ball_id": "cue", "session": session_id}
        ],
    )
    # FastAPI normally awaits the endpoint's 24-hour cleanup task before
    # TestClient returns. Capture that task so the test can inspect the
    # completed export and remove it deterministically in its own finally block.
    scheduled: list[tuple[Any, tuple[Any, ...]]] = []

    def capture_cleanup(
        _tasks: BackgroundTasks, function: Any, *args: Any, **_kwargs: Any
    ) -> None:
        scheduled.append((function, args))

    monkeypatch.setattr(BackgroundTasks, "add_task", capture_cleanup)
    with _client(core) as client:
        response = client.post(
            "/api/v1/game/export?session_id=session-42&format=json"
            "&include_raw_frames=true&include_processed_data=true"
            "&include_trajectories=true&include_events=true"
        )

    assert response.status_code == 200
    body = response.json()
    export_path = Path(body["file_path"])
    try:
        payload_bytes = export_path.read_bytes()
        payload = json.loads(payload_bytes)
        assert body["checksum"] == hashlib.sha256(payload_bytes).hexdigest()
        assert body["size"] == len(payload_bytes)
        assert payload["metadata"]["session_id"] == "session-42"
        assert payload["metadata"]["includes"] == {
            "raw_frames": True,
            "processed_data": True,
            "trajectories": True,
            "events": True,
        }
        assert payload["data"]["current_state"]["frame_number"] == 17
        assert payload["data"]["history"][0]["frame_number"] == 17
        assert payload["data"]["events"][0]["session"] == "session-42"
        assert payload["data"]["trajectories"][0]["ball_id"] == "cue"
        assert history_calls == ["session-42"]
        assert len(scheduled) == 1
        assert scheduled[0][0].__name__ == "cleanup_export_file"
    finally:
        export_path.unlink(missing_ok=True)


def test_export_zip_is_readable_and_optional_data_failures_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartialCore:
        def get_current_state(self) -> GameState:
            return _state(frame_number=6)

        def get_game_history(self, **_kwargs: Any) -> None:
            raise RuntimeError("history storage offline")

        def get_game_events(self, **_kwargs: Any) -> None:
            raise RuntimeError("event storage offline")

        def get_trajectory_history(self, **_kwargs: Any) -> None:
            raise RuntimeError("trajectory storage offline")

    monkeypatch.setattr(
        BackgroundTasks,
        "add_task",
        lambda *_args, **_kwargs: None,
    )
    with _client(PartialCore()) as client:
        response = client.post(
            "/api/v1/game/export?format=zip&include_raw_frames=true"
            "&include_processed_data=true&include_trajectories=true&include_events=true"
        )

    assert response.status_code == 200
    export_path = Path(response.json()["file_path"])
    try:
        with zipfile.ZipFile(export_path) as archive:
            names = set(archive.namelist())
            payload = json.loads(archive.read("session_data.json"))
            assert "frames/README.txt" in names
            assert payload["data"]["current_state"]["frame_number"] == 6
            assert "history" not in payload["data"]
            assert "events" not in payload["data"]
            assert "trajectories" not in payload["data"]
    finally:
        export_path.unlink(missing_ok=True)


def test_export_can_omit_optional_data_and_exports_empty_core_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BackgroundTasks, "add_task", lambda *_args, **_kwargs: None)
    with _client(SimpleNamespace()) as client:
        response = client.post(
            "/api/v1/game/export?format=zip&include_processed_data=false"
            "&include_trajectories=false&include_events=false"
        )

    assert response.status_code == 200
    export_path = Path(response.json()["file_path"])
    try:
        with zipfile.ZipFile(export_path) as archive:
            payload = json.loads(archive.read("session_data.json"))
        assert payload["data"] == {}
        assert payload["metadata"]["includes"] == {
            "raw_frames": False,
            "processed_data": False,
            "trajectories": False,
            "events": False,
        }
    finally:
        export_path.unlink(missing_ok=True)


def test_export_route_validates_options_and_download_reports_missing_file() -> None:
    with _client(SimpleNamespace()) as client:
        invalid_format = client.post("/api/v1/game/export?format=csv")
        invalid_compression = client.post("/api/v1/game/export?compression_level=10")
        missing_export = client.get("/api/v1/game/export/missing-id/download")

    assert invalid_format.status_code == 422
    assert invalid_compression.status_code == 422
    assert missing_export.status_code == 404


@pytest.mark.asyncio
async def test_export_file_failure_returns_sanitized_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_open(_path: str, _mode: str) -> NoReturn:
        raise OSError("export storage unavailable")

    monkeypatch.setattr(game_route, "open", fail_open, raising=False)
    with pytest.raises(HTTPException) as error:
        await game_route.export_session_data(
            background_tasks=BackgroundTasks(),
            session_id=None,
            include_raw_frames=False,
            include_processed_data=False,
            include_trajectories=False,
            include_events=False,
            format="json",
            compression_level=6,
            core_module=cast("Any", SimpleNamespace()),
        )

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to export session data"


@pytest.mark.parametrize(
    ("discovery", "suffix", "media_type"),
    [
        ("direct", "zip", "application/zip"),
        ("glob", "data", "application/json"),
    ],
)
def test_download_serves_matching_exports_and_legacy_glob_matches(
    tmp_path: Path, discovery: str, suffix: str, media_type: str
) -> None:
    export_id = f"game-route-{tmp_path.name}"
    filename = (
        f"session_export_{export_id}.{suffix}"
        if discovery == "direct"
        else f"legacy-{export_id}.{suffix}"
    )
    export_path = Path("/tmp") / filename
    export_path.write_bytes(b"saved session bytes")
    try:
        with _client(SimpleNamespace()) as client:
            response = client.get(f"/api/v1/game/export/{export_id}/download")

        assert response.status_code == 200
        assert response.headers["content-type"] == media_type
        assert response.content == b"saved session bytes"
        assert response.headers["content-disposition"].endswith(filename + '"')
    finally:
        export_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_download_storage_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPath:
        def __init__(self, _path: str) -> None:
            pass

        def exists(self) -> bool:
            raise OSError("export volume unavailable")

    monkeypatch.setattr(game_route, "Path", BrokenPath)
    with pytest.raises(HTTPException) as error:
        await game_route.download_session_export("unavailable")

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to download export file"


def test_statistics_defaults_and_forwards_window_to_core() -> None:
    received: dict[str, datetime] = {}

    def get_statistics(*, start_time: datetime, end_time: datetime) -> dict[str, Any]:
        received.update(start=start_time, end=end_time)
        return {"games": {"total_sessions": 4}, "custom": {"source": "core"}}

    with _client(SimpleNamespace(get_game_statistics=get_statistics)) as client:
        response = client.get("/api/v1/game/stats?time_range=7d")

    assert response.status_code == 200
    body = response.json()
    assert body["time_range"]["duration"] == "7d"
    assert body["games"]["total_sessions"] == 4
    assert body["custom"] == {"source": "core"}
    assert (received["end"] - received["start"]).total_seconds() == pytest.approx(
        7 * 24 * 60 * 60
    )


def test_statistics_survives_core_failure_and_rejects_unknown_window() -> None:
    def fail_statistics(**_kwargs: Any) -> None:
        raise RuntimeError("metrics backend offline")

    with _client(SimpleNamespace(get_game_statistics=fail_statistics)) as client:
        fallback = client.get("/api/v1/game/stats")
        invalid = client.get("/api/v1/game/stats?time_range=90d")

    assert fallback.status_code == 200
    assert fallback.json()["performance"]["average_fps"] == 0
    assert fallback.json()["games"]["total_sessions"] == 0
    assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_statistics_uses_defaults_without_optional_core_capability() -> None:
    response = await game_route.get_game_statistics(
        time_range="1h", core_module=cast("Any", SimpleNamespace())
    )

    assert response["time_range"]["duration"] == "1h"
    assert response["games"]["total_sessions"] == 0
    assert response["performance"]["successful_predictions"] == 0


@pytest.mark.asyncio
async def test_statistics_sanitizes_unexpected_range_from_internal_caller() -> None:
    with pytest.raises(HTTPException) as error:
        await game_route.get_game_statistics(
            time_range="90d", core_module=cast("Any", SimpleNamespace())
        )

    assert error.value.status_code == 500
    assert error.value.detail == "Unable to retrieve game statistics"
