"""Versioned golden contract for the retained game-state HTTP endpoint."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import get_app_state, get_core_module
from backend.api.routes.game import router as game_router
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState, GameState, GameType, TableState
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher

CONTRACT_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/contracts/v1/game_state_http.json"
)


def _contract_state() -> GameState:
    """Build the fixed state represented by the checked-in v1 fixture."""
    timestamp = 1_735_689_600.0
    balls = [
        BallState(
            id="cue",
            position=Vector2D.from_4k(1500, 1080),
            velocity=Vector2D.from_4k(0, 0),
            radius=36,
            is_cue_ball=True,
            confidence=0.99,
            last_update=timestamp,
        ),
        BallState(
            id="track-17",
            position=Vector2D.from_4k(2400, 1080),
            velocity=Vector2D.from_4k(120, 0),
            radius=36,
            number=8,
            confidence=0.87,
            is_observed=False,
            last_update=timestamp - 0.25,
        ),
    ]
    table = TableState(
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
        playing_area_corners=[
            Vector2D.from_4k(x, y)
            for x, y in ((320, 280), (3520, 280), (3520, 1880), (320, 1880))
        ],
        real_width_meters=2.54,
        real_height_meters=1.27,
    )
    cue = CueState(
        tip_position=Vector2D.from_4k(1300, 1080),
        angle=0,
        length=1.47,
        is_visible=True,
        confidence=0.95,
        last_update=timestamp,
    )
    return GameState(
        timestamp=timestamp,
        frame_number=42,
        balls=balls,
        table=table,
        cue=cue,
        game_type=GameType.EIGHT_BALL,
        is_valid=True,
        state_confidence=0.95,
    )


def test_game_state_http_response_matches_versioned_wire_fixture():
    state = _contract_state()
    core = SimpleNamespace(get_current_state=lambda: state)
    app = FastAPI()
    app.include_router(game_router, prefix="/api/v1")
    app.dependency_overrides[get_core_module] = lambda: core
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(core_module=core)

    with TestClient(app) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 200
    expected = json.loads(CONTRACT_FIXTURE.read_text(encoding="utf-8"))
    assert response.json() == expected


def test_python_only_http_path_still_requires_core_state():
    app = FastAPI()
    app.include_router(game_router, prefix="/api/v1")
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace()

    with TestClient(app) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 503


def test_scene_http_read_model_wins_over_core_ball_locations():
    """HTTP bootstrap exposes current scene locations without mutating Core."""

    class Broadcaster:
        async def broadcast_game_state(self, **kwargs):
            return None

    scene_path = (
        Path(__file__).parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["revisions"]["calibration"] = 12
    scene["balls"][0]["measurement"]["table_center_mm"] = {
        "x": 12.5,
        "y": -18.25,
    }
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(publisher.publish(scene))

    # This deliberately differs from the scene's (1882.4, 1024.7) location.
    core_state = _contract_state()
    core = SimpleNamespace(get_current_state=lambda: core_state)
    app = FastAPI()
    app.include_router(game_router, prefix="/api/v1")
    app.dependency_overrides[get_core_module] = lambda: core
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        core_module=core,
        scene_state_publisher=publisher,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/game/state?include_events=false")

    assert response.status_code == 200
    body = response.json()
    assert body["perception_source"] == "rust_scene"
    assert body["balls"][0]["position"] == {
        "x": 1882.4,
        "y": 1024.7,
        "scale": [1.0, 1.0],
    }
    assert body["balls"][0]["position"] != {
        "x": 1500.0,
        "y": 1080.0,
        "scale": [1.0, 1.0],
    }
    assert body["balls"][0]["table_position"] == {
        "x": 12.5,
        "y": -18.25,
        "unit": "millimeters",
        "origin": "playing_surface_center",
    }
    assert body["balls"][0]["velocity"] is None
    assert body["balls"][0]["is_cue_ball"] is None
    assert body["balls"][0]["is_pocketed"] is None
    assert body["balls"][0]["last_update"]
    assert body["balls"][0]["source_observation"] == {
        "engine_instance_id": "engine-rust-stage2-a",
        "session_id": "replay-session-01",
        "frame_id": "frame-0001",
        "timestamp": {
            "ticks": "1000000000",
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": "host-clock-a",
            "stage": "shared_memory_publication",
        },
    }
    assert body["observation_status"]["calibration_revision"] == 12
    assert body["frame_number"] == 1
    assert body["events"] == []
    assert body["table"] is None
    assert body["table_metadata_status"] == "unverified"
    assert body["scene_calibration_revision"] == 12
    assert body["table_calibration_revision"] is None
    assert body["core_metadata_status"] == "available"
    # The read path must not replace or update the retained Core state.
    assert core_state.balls[0].position.x == 1500


def test_scene_http_localization_does_not_require_core():
    """A usable scene remains available while optional Core is absent."""

    class Broadcaster:
        async def broadcast_game_state(self, **kwargs):
            return None

    scene_path = (
        Path(__file__).parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(publisher.publish(scene))

    app = FastAPI()
    app.include_router(game_router, prefix="/api/v1")
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        scene_state_publisher=publisher
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/game/state")

    assert response.status_code == 200
    body = response.json()
    assert body["perception_source"] == "rust_scene"
    assert len(body["balls"]) == 1
    assert body["table"] is None
    assert body["game_type"] is None
    assert body["core_metadata_status"] == "unavailable"
    assert body["core_metadata_timestamp"] is None
    assert body["table_metadata_status"] == "unavailable"


def test_scene_table_geometry_is_emitted_only_for_matching_revisions():
    """Unknown or changed calibration generations cannot make stale geometry look current."""

    class Broadcaster:
        async def broadcast_game_state(self, **kwargs):
            return None

    scene_path = (
        Path(__file__).parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["revisions"]["calibration"] = 12
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(publisher.publish(scene))
    read_model = publisher.read_model
    assert read_model is not None

    matching_core = _contract_state()
    matching_core.table.calibration_revision = 12
    from backend.api.models.converters import scene_state_to_response

    matched = scene_state_to_response(read_model, matching_core)
    assert matched.table_metadata_status == "matched"
    assert matched.table is not None
    assert matched.table_calibration_revision == matched.scene_calibration_revision

    mismatch_core = _contract_state()
    mismatch_core.table.calibration_revision = 13
    mismatched = scene_state_to_response(read_model, mismatch_core)
    assert mismatched.table_metadata_status == "mismatch"
    assert mismatched.table is None
    assert len(mismatched.balls) == 1


def test_scene_validity_requires_usable_positions_but_accepts_valid_empty_scene():
    """A failed geometry mapping is invalid; a successful empty detection is valid."""

    class Broadcaster:
        async def broadcast_game_state(self, **kwargs):
            return None

    scene_path = (
        Path(__file__).parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["balls"] = []
    publisher = SceneSnapshotStatePublisher(Broadcaster())
    assert asyncio.run(publisher.publish(scene))
    read_model = publisher.read_model
    assert read_model is not None
    from backend.api.models.converters import scene_state_to_response

    empty = scene_state_to_response(read_model, None)
    assert empty.balls == []
    assert empty.is_valid is True

    unusable = json.loads(scene_path.read_text(encoding="utf-8"))
    unusable["balls"][0]["status"] = "lost"
    unusable["balls"][0]["measurement"] = None
    unusable["balls"][0]["estimate"] = None
    assert asyncio.run(publisher.publish(unusable))
    read_model = publisher.read_model
    assert read_model is not None
    invalid = scene_state_to_response(read_model, None)
    assert invalid.balls == []
    assert invalid.observation_status["ball_positions_status"] == "unavailable"
    assert invalid.is_valid is False
