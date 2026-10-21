"""Versioned contract checks for anonymous Rust scene state on WebSocket."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.api.websocket.schemas import BallData, GameStateData
from backend.perception.scene_state_adapter import SceneSnapshotStateAdapter

FIXTURES = Path(__file__).parents[1] / "fixtures/contracts/v1"
SCENE_FIXTURE = (
    Path(__file__).parents[3]
    / "perception/contracts/v1/scene-rust-tracked-success.json"
)


def test_rust_scene_state_fixture_validates_and_preserves_source_evidence() -> None:
    """Scene positions remain usable without fabricated color or identity."""
    message = json.loads(
        (FIXTURES / "websocket_state_rust_scene.json").read_text(encoding="utf-8")
    )

    state = GameStateData.model_validate(message["data"])
    scene = json.loads(SCENE_FIXTURE.read_text(encoding="utf-8"))
    scene["balls"][0]["measurement"]["table_center_mm"] = {
        "x": 12.5,
        "y": -18.25,
    }
    payload = SceneSnapshotStateAdapter().adapt(scene)
    assert payload is not None
    producer_state = GameStateData.model_validate(
        {
            "balls": payload.balls,
            "cue": None,
            "table": payload.table,
            "game_type": payload.game_type,
            "ball_count": len(payload.balls),
            "frame_number": 1,
            "observation_status": payload.observation_status,
            "timestamp": message["data"]["timestamp"],
            "sequence": message["data"]["sequence"],
        }
    )

    assert state.ball_count == 1
    assert producer_state.balls == state.balls
    ball = state.balls[0]
    assert ball.id == "track-1"
    assert ball.position.x == 1882.4
    assert ball.position.y == 1024.7
    assert ball.color is None
    assert ball.number is None
    assert ball.velocity is None
    assert ball.identity_status == "unknown"
    assert ball.tracking_status == "observed"
    assert ball.position_kind == "measured"
    assert ball.table_position is not None
    assert ball.table_position.x == 12.5
    assert ball.table_position.y == -18.25
    assert ball.table_position.unit == "millimeters"
    assert ball.source_observation is not None
    assert ball.source_observation.session_id == "replay-session-01"
    assert ball.source_observation.frame_id == "frame-0001"
    assert ball.source_observation.timestamp.ticks == "1000000000"
    assert ball.source_observation.timestamp.stage == "shared_memory_publication"
    assert state.observation_status is not None
    assert state.observation_status["source_timestamp_kind"].endswith(
        "shared_memory_publication"
    )


def test_legacy_python_ball_without_color_or_with_numeric_id_remains_valid() -> None:
    """The additive Rust contract keeps legacy ball identity and payloads valid."""
    message = json.loads(
        (FIXTURES / "websocket_state.json").read_text(encoding="utf-8")
    )

    ball = BallData.model_validate(message["data"]["balls"][0])

    assert ball.id == 1
    assert ball.color is None
    assert ball.position.x == 1500.0
    assert ball.type == "cue"


def test_scene_source_clock_unit_is_validated() -> None:
    """Malformed source units cannot be mistaken for nanosecond freshness."""
    message = json.loads(
        (FIXTURES / "websocket_state_rust_scene.json").read_text(encoding="utf-8")
    )
    message["data"]["balls"][0]["source_observation"]["timestamp"][
        "unit"
    ] = "milliseconds"

    with pytest.raises(ValidationError, match="nanoseconds"):
        GameStateData.model_validate(message["data"])
