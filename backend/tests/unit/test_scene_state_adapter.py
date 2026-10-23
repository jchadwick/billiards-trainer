"""Compatibility and freshness tests for Rust scene publication."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from backend.api.websocket.broadcaster import MessageBroadcaster
from backend.core.constants_4k import BALL_RADIUS_4K
from backend.perception.scene_state_adapter import (
    SceneSnapshotStateAdapter,
    SceneSnapshotStatePublisher,
    SceneStateCompatibilityError,
)

SCENE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "perception/contracts/v1/scene-rust-tracked-success.json"
)
PUBLIC_STATE_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "perception/contracts/public-state/number-8.json"
)


def _scene() -> dict[str, Any]:
    value = json.loads(SCENE_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _native_delivery() -> dict[str, Any]:
    fixture = json.loads(PUBLIC_STATE_FIXTURE.read_text(encoding="utf-8"))
    scene = fixture["scene"]
    payload = fixture["expected"]
    payload["observation_status"].update(
        freshness_status="fresh", localization_age_ns="0"
    )
    return {
        "delivery_type": "billiards_public_state_delivery",
        "delivery_schema_version": 1,
        "source_snapshot": scene,
        "current_state": {
            "payload": payload,
            "epoch": 1,
            "updated_monotonic_ns": 1_000_000,
            "localization_received_monotonic_ns": 1_000_000,
            "localization_age_ns": 0,
            "ball_age_ns": {ball["id"]: 0 for ball in payload["balls"]},
            "freshness": "fresh",
        },
    }


def test_all_checked_in_public_state_fixtures_match_python_adapter() -> None:
    """Keep all checked-in cross-runtime examples tied to adapter behavior."""
    fixture_root = PUBLIC_STATE_FIXTURE.parent
    fixtures: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(fixture_root.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and {"scene", "table", "game_type", "expected"} <= (
            value.keys()
        ):
            fixtures.append((path, value))

    assert len(fixtures) == 35, "every versioned public-state fixture must be tested"
    adapter = SceneSnapshotStateAdapter()
    for path, fixture in fixtures:
        payload = adapter.adapt(
            fixture["scene"], table=fixture["table"], game_type=fixture["game_type"]
        )
        actual = asdict(payload) if payload is not None else None
        assert actual == fixture["expected"], path.name


def test_adapter_preserves_observation_identity_and_units() -> None:
    scene = _scene()
    scene["revisions"].update(calibration=12, calibration_state="valid")
    scene["balls"][0]["measurement"]["table_center_mm"] = {
        "x": 18.25,
        "y": -42.5,
    }

    payload = SceneSnapshotStateAdapter().adapt(
        scene,
        table={"calibrated": True},
        game_type="eight_ball",
    )

    assert payload is not None
    assert payload.game_type == "eight_ball"
    assert payload.table == {"calibrated": True}
    assert len(payload.balls) == 1
    ball = payload.balls[0]
    assert ball["id"] == "track-1"
    assert ball["position"] == {
        "x": 1882.4,
        "y": 1024.7,
        "scale": [1.0, 1.0],
    }
    assert ball["table_position"] == {
        "x": 18.25,
        "y": -42.5,
        "unit": "millimeters",
        "origin": "playing_surface_center",
    }
    assert ball["number"] is None
    assert ball["type"] == "unknown"
    assert ball["identity_status"] == "unknown"
    assert ball["identity_confidence"] is None
    assert ball["identity_evidence"] == []
    assert ball["is_observed"] is True
    assert ball["confidence_kind"] == "detector_observation"
    assert ball["source_observation"]["engine_instance_id"] == "engine-rust-stage2-a"
    assert ball["radius"] == BALL_RADIUS_4K
    assert ball["radius_source"] == "legacy_standard_pool_ball_assumption"
    assert payload.observation_status["source_timestamp"] == "1000000000"
    assert (
        payload.observation_status["source_timestamp_kind"]
        == "monotonic:host-clock-a:shared_memory_publication"
    )
    assert payload.observation_status["ball_observation_source_timestamp_unit"] == (
        "nanoseconds"
    )
    assert payload.observation_status["source_timestamp_is_exposure"] is False
    assert payload.observation_status["frame_publication_age_status"] == (
        "clock_domain_mismatch"
    )
    assert "frame_publication_age_seconds" not in payload.observation_status
    assert payload.observation_status["calibration_revision"] == 12
    assert payload.observation_status["observation_identity"] == {
        "engine_instance_id": "engine-rust-stage2-a",
        "session_id": "replay-session-01",
        "frame_id": "frame-0001",
        "snapshot_revision": 1,
    }


def test_adapter_reports_age_when_source_and_snapshot_share_monotonic_domain() -> None:
    scene = _scene()
    scene["published_at"].update(
        ticks="1500000000",
        clock="monotonic",
        domain_id="host-clock-a",
        stage="snapshot_publication",
    )

    payload = SceneSnapshotStateAdapter().adapt(scene)

    assert payload is not None
    assert payload.observation_status["frame_publication_age_seconds"] == 0.5
    assert payload.observation_status["frame_publication_age_status"] == "available"
    assert (
        payload.observation_status["frame_age_basis"]
        == "source_timestamp_to_snapshot_publication"
    )


def test_adapter_does_not_report_age_when_publication_precedes_source() -> None:
    scene = _scene()
    scene["published_at"].update(
        ticks="999999999",
        clock="monotonic",
        domain_id="host-clock-a",
        stage="snapshot_publication",
    )

    payload = SceneSnapshotStateAdapter().adapt(scene)

    assert payload is not None
    assert payload.observation_status["frame_publication_age_status"] == (
        "publication_precedes_source"
    )
    assert "frame_publication_age_seconds" not in payload.observation_status


def test_adapter_maps_source_resolution_to_canonical_pixels_once() -> None:
    scene = _scene()
    scene["source"].update(width_px=1920, height_px=1080)
    measurement = scene["balls"][0]["measurement"]
    measurement.update(image_width_px=1920, image_height_px=1080)
    measurement["image_center_px"] = {"x": 941.2, "y": 512.35}
    scene["cue"].update(image_width_px=1920, image_height_px=1080)

    payload = SceneSnapshotStateAdapter().adapt(scene)

    assert payload is not None
    assert payload.balls[0]["position"] == {
        "x": 1882.4,
        "y": 1024.7,
        "scale": [1.0, 1.0],
    }


def test_adapter_preserves_estimates_without_calling_them_observed() -> None:
    scene = _scene()
    ball = scene["balls"][0]
    ball["status"] = "occluded"
    ball["estimate"] = {
        "kind": "extrapolated",
        "as_of": copy.deepcopy(scene["published_at"]),
        "image_center_px": {"x": 1888.0, "y": 1022.0},
        "image_space": "raw",
        "image_width_px": 3840,
        "image_height_px": 2160,
        "table_center_mm": None,
        "table_position_uncertainty": {
            "status": "unavailable",
            "sigma_x_mm": None,
            "sigma_y_mm": None,
        },
    }

    payload = SceneSnapshotStateAdapter().adapt(scene)

    assert payload is not None
    ball_payload = payload.balls[0]
    assert ball_payload["position"] == {
        "x": 1888.0,
        "y": 1022.0,
        "scale": [1.0, 1.0],
    }
    assert ball_payload["position_kind"] == "extrapolated"
    assert ball_payload["is_observed"] is False
    assert ball_payload["confidence"] == 0.0
    assert ball_payload["confidence_kind"] == "unavailable"
    assert ball_payload["last_observation_confidence"] == 0.97
    assert payload.observation_status["ball_positions_status"] == "cached"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("update_kind", "annotation"),
        ("update_kind", "health"),
        ("outcome", "failed"),
        ("outcome", "degraded"),
        ("outcome", "skipped"),
    ],
)
def test_non_observation_or_non_success_does_not_publish_new_locations(
    field: str, value: str
) -> None:
    scene = _scene()
    scene[field] = value

    assert SceneSnapshotStateAdapter().adapt(scene) is None


def test_rectified_geometry_is_rejected_for_the_raw_preview_contract() -> None:
    scene = _scene()
    scene["balls"][0]["measurement"]["image_space"] = "rectified"

    with pytest.raises(SceneStateCompatibilityError, match="raw coordinates"):
        SceneSnapshotStateAdapter().adapt(scene)


def test_geometry_from_a_different_resolution_is_rejected() -> None:
    scene = _scene()
    scene["source"].update(width_px=1920, height_px=1080)

    with pytest.raises(SceneStateCompatibilityError, match="dimensions differ"):
        SceneSnapshotStateAdapter().adapt(scene)


def test_ball_without_any_image_position_is_omitted_and_reported() -> None:
    scene = _scene()
    ball = scene["balls"][0]
    ball["status"] = "lost"
    ball["measurement"] = None
    ball["estimate"] = None

    payload = SceneSnapshotStateAdapter().adapt(scene)

    assert payload is not None
    assert payload.balls == []
    assert payload.observation_status["omitted_track_ids_without_image_position"] == [
        "track-1"
    ]
    assert payload.observation_status["ball_positions_status"] == "unavailable"


def test_publisher_sends_only_successful_observations_without_core_round_trip() -> None:
    class Broadcaster:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            self.calls.append(
                {
                    "balls": balls,
                    "cue": cue,
                    "table": table,
                    "timestamp": timestamp,
                    "observation_status": observation_status,
                    "game_type": game_type,
                }
            )

    async def run() -> tuple[bool, bool, list[dict[str, Any]]]:
        broadcaster = Broadcaster()
        publisher = SceneSnapshotStatePublisher(broadcaster)
        rejected_scene = _scene()
        rejected_scene["outcome"] = "failed"
        published = await publisher.publish(_scene(), game_type="eight_ball")
        skipped = await publisher.publish(rejected_scene)
        return published, skipped, broadcaster.calls

    published, skipped, calls = asyncio.run(run())

    assert published is True
    assert skipped is False
    assert len(calls) == 1
    assert calls[0]["cue"] is None
    assert calls[0]["game_type"] == "eight_ball"
    assert calls[0]["balls"][0]["id"] == "track-1"
    assert calls[0]["observation_status"]["source_frame_id"] == "frame-0001"


def test_publisher_advances_rust_age_from_ingress_using_python_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broadcaster:
        def __init__(self) -> None:
            self.status: dict[str, Any] | None = None

        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            self.status = observation_status

    async def run() -> dict[str, Any]:
        broadcaster = Broadcaster()
        publisher = SceneSnapshotStatePublisher(broadcaster)
        scene = _scene()
        scene["published_at"].update(
            ticks="1500000000",
            clock="monotonic",
            domain_id="host-clock-a",
            stage="snapshot_publication",
        )
        monkeypatch.setattr(
            "backend.perception.scene_state_adapter.time.monotonic_ns",
            lambda: 750_000_000,
        )
        assert await publisher.publish(scene, received_monotonic_ns=500_000_000)
        assert broadcaster.status is not None
        return broadcaster.status

    status = asyncio.run(run())

    assert status["frame_publication_age_seconds"] == 0.75


def test_publisher_payload_passes_existing_broadcaster_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> dict[str, Any]:
        broadcaster = MessageBroadcaster()
        broadcast_stream = AsyncMock()
        monkeypatch.setattr(broadcaster, "_broadcast_to_stream", broadcast_stream)
        publisher = SceneSnapshotStatePublisher(broadcaster)

        assert await publisher.publish(_scene()) is True
        call = broadcast_stream.await_args
        assert call is not None
        return cast("dict[str, Any]", call.args[1])

    state_data = asyncio.run(run())

    assert state_data["balls"][0]["position"]["scale"] == [1.0, 1.0]
    assert "table_position" not in state_data["balls"][0]
    assert state_data["observation_status"]["source_frame_id"] == "frame-0001"


def test_native_public_state_uses_rust_payload_without_python_reinterpretation() -> (
    None
):
    class Broadcaster:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def broadcast_game_state(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    class RejectingAdapter:
        def adapt(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "native Rust state must not run through Python adaptation"
            )

    async def run() -> tuple[Any, Broadcaster, datetime]:
        broadcaster = Broadcaster()
        publisher = SceneSnapshotStatePublisher(
            broadcaster, adapter=cast("Any", RejectingAdapter())
        )
        delivery = _native_delivery()
        received_at = datetime(2026, 10, 22, tzinfo=timezone.utc)
        outcome = await publisher.publish_native_state(
            delivery,
            received_at=received_at,
            received_monotonic_ns=123_456,
        )
        assert outcome == "localization"
        return publisher.read_model, broadcaster, received_at

    read_model, broadcaster, received_at = asyncio.run(run())

    assert read_model is not None
    assert read_model.payload.balls[0]["number"] == 8
    assert read_model.payload.observation_status["freshness_status"] == "fresh"
    assert read_model.localization_received_at == received_at
    assert len(broadcaster.calls) == 1
    assert broadcaster.calls[0]["balls"][0]["id"] == "track-1"


def test_health_failure_publishes_cached_ball_as_stale_without_advancing_its_time() -> (
    None
):
    class Broadcaster:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            self.calls.append(
                {
                    "balls": balls,
                    "observation_status": observation_status,
                    "table": table,
                    "game_type": game_type,
                }
            )

    async def run() -> list[dict[str, Any]]:
        broadcaster = Broadcaster()
        publisher = SceneSnapshotStatePublisher(broadcaster)
        assert await publisher.publish(_scene(), game_type="eight_ball")
        health = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "perception/contracts/v1/scene-failure.json"
            ).read_text(encoding="utf-8")
        )
        assert isinstance(health, dict)
        health["engine_instance_id"] = "engine-rust-stage2-a"
        health["source"]["session_id"] = "replay-session-01"
        assert await publisher.publish_health(health)
        return broadcaster.calls

    states = asyncio.run(run())

    assert len(states) == 2
    stale_ball = states[1]["balls"][0]
    assert stale_ball["is_observed"] is False
    assert stale_ball["confidence"] == 0.0
    assert stale_ball["confidence_kind"] == "unavailable"
    assert stale_ball["last_observation_confidence"] == 0.97
    assert stale_ball["source_observation"]["timestamp"]["ticks"] == "1000000000"
    assert states[1]["observation_status"]["phase"] == "rust_scene_health"
    assert states[1]["observation_status"]["ball_positions_status"] == "cached"
    assert states[1]["observation_status"]["ball_detection_status"] == "failed"
    assert states[1]["observation_status"]["health_reason"] == "inference_timeout"


def test_read_model_is_a_defensive_copy_and_health_keeps_localization_receipt() -> None:
    class Broadcaster:
        async def broadcast_game_state(self, **kwargs: Any) -> None:
            return None

    async def run() -> tuple[Any, Any, Any]:
        publisher = SceneSnapshotStatePublisher(Broadcaster())
        ingress_receipt = datetime(2026, 10, 20, 14, 30, tzinfo=timezone.utc)
        assert await publisher.publish(_scene(), received_at=ingress_receipt)
        initial = publisher.read_model
        assert initial is not None
        assert initial.localization_received_at == ingress_receipt
        initial_ball_receipt = initial.ball_localization_received_at["track-1"]
        assert initial_ball_receipt == ingress_receipt
        initial.payload.balls[0]["position"]["x"] = -999
        assert publisher.read_model is not None
        assert publisher.read_model.payload.balls[0]["position"]["x"] == 1882.4

        newer_scene = _scene()
        newer_scene["source"]["frame_id"] = "frame-0002"
        newer_scene["source"]["timestamp"]["ticks"] = "2000000000"
        newer_scene["transport_sequence"] = 2
        newer_scene["published_at"]["ticks"] = "2000000001"
        newer_scene["balls"][0]["status"] = "occluded"
        newer_scene["balls"][0]["estimate"] = {
            "kind": "extrapolated",
            "as_of": copy.deepcopy(newer_scene["published_at"]),
            "image_center_px": {"x": 1888.0, "y": 1022.0},
            "image_space": "raw",
            "image_width_px": 3840,
            "image_height_px": 2160,
            "table_center_mm": None,
            "table_position_uncertainty": {
                "status": "unavailable",
                "sigma_x_mm": None,
                "sigma_y_mm": None,
            },
        }
        assert await publisher.publish(newer_scene)
        observed = publisher.read_model
        assert observed is not None
        assert observed.localization_received_at > initial.localization_received_at
        assert observed.ball_localization_received_at["track-1"] == initial_ball_receipt

        health = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "perception/contracts/v1/scene-failure.json"
            ).read_text(encoding="utf-8")
        )
        assert isinstance(health, dict)
        health["engine_instance_id"] = "engine-rust-stage2-a"
        health["source"]["session_id"] = "replay-session-01"
        assert await publisher.publish_health(health)
        stale = publisher.read_model
        assert stale is not None
        return observed, stale, publisher

    observed, stale, _publisher = asyncio.run(run())

    assert stale.updated_at >= observed.updated_at
    assert stale.localization_received_at == observed.localization_received_at
    assert (
        stale.ball_localization_received_at["track-1"]
        == observed.ball_localization_received_at["track-1"]
    )
    assert stale.payload.balls[0]["source_observation"]["timestamp"]["ticks"] == (
        "1000000000"
    )
    assert stale.payload.balls[0]["is_observed"] is False
    assert stale.payload.observation_status["phase"] == "rust_scene_health"


def test_health_failure_from_new_session_does_not_republish_old_session_positions() -> (
    None
):
    class Broadcaster:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def broadcast_game_state(
            self,
            balls: object,
            cue: dict[str, Any] | None = None,
            table: dict[str, Any] | None = None,
            timestamp: Any = None,
            observation_status: dict[str, Any] | None = None,
            game_type: str | None = None,
        ) -> None:
            self.calls.append(
                {"balls": balls, "observation_status": observation_status}
            )

    async def run() -> tuple[list[dict[str, Any]], Any]:
        broadcaster = Broadcaster()
        publisher = SceneSnapshotStatePublisher(broadcaster)
        assert await publisher.publish(_scene())
        health = json.loads(
            (
                Path(__file__).resolve().parents[3]
                / "perception/contracts/v1/scene-failure.json"
            ).read_text(encoding="utf-8")
        )
        assert isinstance(health, dict)
        health["engine_instance_id"] = "engine-restarted"
        health["source"]["session_id"] = "new-session"
        assert await publisher.publish_health(health)
        return broadcaster.calls, publisher.read_model

    states, read_model = asyncio.run(run())

    assert len(states) == 2
    assert states[1]["balls"] == []
    assert states[1]["observation_status"]["ball_positions_status"] == "unavailable"
    assert read_model is not None
    assert read_model.localization_received_at is None
    assert read_model.ball_localization_received_at == {}
