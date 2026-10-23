"""Replay recorded pixels through production processing and a real socket server.

The detector fixture is deliberately the only inference replacement. It makes
geometry repeatable without downloading a model or attaching a camera. The
separate model-assets test exercises installed inference assets explicitly.
"""

import asyncio
import copy
import json
import math
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import uvicorn
import websockets
from fastapi import FastAPI

from backend.api.dependencies import get_app_state, get_core_module
from backend.api.routes.game import router as game_router
from backend.api.websocket import broadcaster as broadcaster_module
from backend.api.websocket import endpoints
from backend.api.websocket import manager as manager_module
from backend.api.websocket.broadcaster import MessageBroadcaster
from backend.api.websocket.handler import WebSocketHandler
from backend.api.websocket.manager import StreamType, WebSocketManager
from backend.api.websocket.monitoring import ConnectionMonitor
from backend.api.websocket.rate_limit import ConnectionRateLimiter
from backend.common.models import Ball
from backend.config import config
from backend.core import CoreModule
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, GameState, GameType, TableState
from backend.integration_service import IntegrationService
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher
from backend.vision import VisionModule
from backend.vision.models import BallType

WEBSOCKET_STATE_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/contracts/v1/websocket_state.json"
)
WEBSOCKET_TRAJECTORY_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/contracts/v1/websocket_trajectory.json"
)
RUST_SCENE_FIXTURE = (
    Path(__file__).parents[3]
    / "perception/contracts/v1/scene-rust-tracked-success.json"
)
RUST_HEALTH_FAILURE_FIXTURE = (
    Path(__file__).parents[3] / "perception/contracts/v1/scene-failure.json"
)
CONTRACT_WALL_TIME = "2025-01-01T00:00:00+00:00"
CONTRACT_OBSERVATION_TIME = 1_735_689_600.0


def _generated_table_frame() -> np.ndarray:
    """Build a small deterministic table image without checked-in recordings."""
    height, width = 360, 640
    frame = np.full((height, width, 3), (35, 125, 55), dtype=np.uint8)
    frame[:18, :] = (35, 45, 55)
    frame[-18:, :] = (35, 45, 55)
    frame[:, :18] = (35, 45, 55)
    frame[:, -18:] = (35, 45, 55)

    # Add shaded, high-contrast disks so the opt-in detector smoke test exercises
    # image inference. This synthetic rendering is deliberately not accuracy data.
    y_grid, x_grid = np.ogrid[:height, :width]
    for center_x, center_y, color in (
        (210, 135, (245, 245, 245)),
        (390, 205, (30, 30, 220)),
        (465, 135, (25, 205, 235)),
    ):
        radius = 12
        disk = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2
        frame[disk] = color
    return frame


class FixtureDetector:
    """Deterministic inference boundary for a horizontal two-ball shot."""

    def detect_balls_with_classification(self, frame, **kwargs):
        height, width = frame.shape[:2]
        return [
            Ball(
                position=(1500.0, 1080.0),
                radius=36.0,
                ball_type=BallType.CUE,
                confidence=0.99,
                track_id=1,
                coordinate_space="4k_canonical",
                source_resolution=(width, height),
            ),
            Ball(
                position=(2400.0, 1080.0),
                radius=36.0,
                ball_type=BallType.SOLID_1,
                confidence=0.99,
                track_id=2,
                coordinate_space="4k_canonical",
                source_resolution=(width, height),
            ),
        ]

    def detect_cue(self, frame, **kwargs):
        height, width = frame.shape[:2]
        return SimpleNamespace(
            confidence=0.99,
            center=(1300 * width / 3840, height / 2),
            bbox=(
                1100 * width / 3840,
                height / 2 - 3,
                1400 * width / 3840,
                height / 2 + 3,
            ),
            angle=0.0,
        )


@dataclass
class ReplayMeasurements:
    frame_count: int
    elapsed_seconds: float
    mean_frame_to_delivery_ms: float
    max_frame_to_delivery_ms: float
    maximum_publication_age_seconds: float
    maximum_queue_depth: int
    dropped_updates: int
    correct_predictions: int


class ReadyServer(uvicorn.Server):
    """Expose startup as a synchronization event instead of polling a port."""

    def __init__(self, server_config):
        super().__init__(server_config)
        self.ready = asyncio.Event()

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        self.ready.set()


class SendGate:
    """Hold the server's send for one real client at the ASGI transport boundary."""

    def __init__(self, app):
        self.app = app
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, scope, receive, send):
        async def gated_send(message):
            if (
                scope["type"] == "websocket"
                and scope.get("query_string") == b"slow=1"
                and message["type"] == "websocket.send"
                and json.loads(message.get("text", "{}"))["type"] == "state"
            ):
                self.entered.set()
                await self.release.wait()
            await send(message)

        await self.app(scope, receive, gated_send)


def _http_metadata_state() -> GameState:
    """Provide deliberately stale Core geometry for scene/API integration."""
    timestamp = 1_735_689_600.0
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
    core_ball = BallState(
        id="core-stale-ball",
        position=Vector2D.from_4k(1500, 1080),
        velocity=Vector2D.from_4k(0, 0),
        radius=36,
        is_cue_ball=True,
        confidence=0.99,
        last_update=timestamp,
    )
    return GameState(
        timestamp=timestamp,
        frame_number=42,
        balls=[core_ball],
        table=table,
        cue=None,
        game_type=GameType.EIGHT_BALL,
        is_valid=True,
        state_confidence=0.95,
    )


@pytest.fixture
async def local_pipeline(monkeypatch, tmp_path):
    """Use production admission, endpoint, subscriptions, writer and broadcaster."""
    # Keep fixture geometry and all persistence independent of a developer's
    # table calibration and current working directory.
    original_get = config.get
    overrides = {
        "table.calibration_resolution_width": 3840,
        "table.calibration_resolution_height": 2160,
        "table.playing_area_corners": [
            {"x": 200, "y": 200},
            {"x": 3640, "y": 200},
            {"x": 3640, "y": 1960},
            {"x": 200, "y": 1960},
        ],
        "table.marker_dots": [],
        "vision.processing.phase_scheduler.enabled": False,
        "vision.performance.enable_profiling": False,
        "integration.max_collision_depth": 1,
    }
    monkeypatch.setattr(
        config,
        "get",
        lambda key, default=None: overrides.get(key, original_get(key, default)),
    )
    monkeypatch.chdir(tmp_path)
    manager = WebSocketManager()
    handler = WebSocketHandler(manager=manager)
    handler.replaceable_queue_capacity = 2
    limiter = ConnectionRateLimiter(
        config_get=lambda key, default=None: (
            "memory" if key == "api.rate_limiting.store" else default
        )
    )
    await limiter.initialize()
    monkeypatch.setattr(endpoints, "websocket_handler", handler)
    monkeypatch.setattr(endpoints, "websocket_manager", manager)
    monkeypatch.setattr(endpoints, "rate_limiter", limiter)
    monkeypatch.setattr(endpoints, "connection_monitor", ConnectionMonitor())
    monkeypatch.setattr(manager_module, "websocket_handler", handler)
    monkeypatch.setattr(broadcaster_module, "websocket_handler", handler)
    monkeypatch.setattr(broadcaster_module, "websocket_manager", manager)
    app = FastAPI()
    app.include_router(endpoints.websocket_router)
    app.include_router(game_router, prefix="/api/v1")
    core = SimpleNamespace(
        get_current_state=_http_metadata_state,
        update_state=Mock(),
        predict_trajectories=Mock(),
    )
    app_state = SimpleNamespace(
        core_module=core,
        scene_state_publisher=None,
        integration_service=None,
        config_module=None,
    )
    app.dependency_overrides[get_core_module] = lambda: core
    app.dependency_overrides[get_app_state] = lambda: app_state
    gate = SendGate(app)
    server = ReadyServer(uvicorn.Config(gate, log_level="error", lifespan="off"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(server.ready.wait(), timeout=5)
        yield SimpleNamespace(
            handler=handler,
            manager=manager,
            broadcaster=MessageBroadcaster(),
            url=f"ws://127.0.0.1:{port}/ws",
            gate=gate,
            app=app,
            app_state=app_state,
            core=core,
        )
    finally:
        gate.release.set()
        active_connections = list(handler.connections.values())
        await handler.shutdown()
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()
        await limiter.close()
        assert not handler.connections
        assert not manager.sessions
        assert not handler._disconnect_tasks
        assert all(
            connection.writer_task is None or connection.writer_task.done()
            for connection in active_connections
        )


async def receive_type(client, message_type):
    """Ignore unrelated control messages without losing transport ordering."""
    while True:
        message = json.loads(await asyncio.wait_for(client.recv(), timeout=5))
        if message["type"] == message_type:
            return message


@pytest.mark.integration
@pytest.mark.system
async def test_rust_scene_and_health_freshness_reach_real_websocket_clients(
    local_pipeline: SimpleNamespace,
) -> None:
    successful = json.loads(RUST_SCENE_FIXTURE.read_text(encoding="utf-8"))
    failure = json.loads(RUST_HEALTH_FAILURE_FIXTURE.read_text(encoding="utf-8"))
    failure["engine_instance_id"] = successful["engine_instance_id"]
    failure["source"]["session_id"] = successful["source"]["session_id"]
    publisher = SceneSnapshotStatePublisher(local_pipeline.broadcaster)
    local_pipeline.app_state.scene_state_publisher = publisher

    async with websockets.connect(local_pipeline.url) as client:
        await receive_type(client, "connection")
        await client.send(
            json.dumps({"type": "subscribe", "data": {"streams": ["state"]}})
        )
        await receive_type(client, "subscribed")

        assert await publisher.publish(successful)
        observed = (await receive_type(client, "state"))["data"]
        observed_ball = observed["balls"][0]
        observed_ball_timestamp = observed_ball["source_observation"]["timestamp"]
        assert observed["observation_status"]["phase"] == "rust_scene_observation"
        assert observed["observation_status"]["ball_positions_status"] == "observed"
        assert observed_ball["is_observed"] is True

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=local_pipeline.app),
            base_url="http://test",
        ) as http_client:
            http_observed_response = await http_client.get(
                "/api/v1/game/state?include_trajectories=true"
            )
        assert http_observed_response.status_code == 200
        http_observed = http_observed_response.json()
        assert http_observed["perception_source"] == "rust_scene"
        assert http_observed["balls"][0]["position"] == observed_ball["position"]
        assert (
            http_observed["balls"][0]["source_observation"]
            == observed_ball["source_observation"]
        )
        local_pipeline.core.update_state.assert_not_called()
        local_pipeline.core.predict_trajectories.assert_not_called()

        assert await publisher.publish_health(failure)
        stale = (await receive_type(client, "state"))["data"]
        stale_ball = stale["balls"][0]
        assert stale["observation_status"]["phase"] == "rust_scene_health"
        assert stale["observation_status"]["ball_positions_status"] == "cached"
        assert stale["observation_status"]["ball_detection_status"] == "failed"
        assert stale_ball["is_observed"] is False
        assert stale_ball["confidence"] == 0.0
        assert stale_ball["confidence_kind"] == "unavailable"
        assert stale_ball["last_observation_confidence"] == observed_ball["confidence"]
        assert stale_ball["source_observation"]["timestamp"] == observed_ball_timestamp

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=local_pipeline.app),
            base_url="http://test",
        ) as http_client:
            http_stale_response = await http_client.get("/api/v1/game/state")
        assert http_stale_response.status_code == 200
        http_stale = http_stale_response.json()
        assert http_stale["observation_status"]["phase"] == "rust_scene_health"
        assert http_stale["balls"][0]["is_observed"] is False
        assert (
            http_stale["balls"][0]["last_update"]
            == http_observed["balls"][0]["last_update"]
        )
        assert (
            http_stale["balls"][0]["source_observation"]
            == http_observed["balls"][0]["source_observation"]
        )
        local_pipeline.core.update_state.assert_not_called()
        local_pipeline.core.predict_trajectories.assert_not_called()


def assert_scaled_points(value):
    """Require explicit canonical scale on every serialized x/y point."""
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            assert value.get("scale") == [1.0, 1.0]
        for child in value.values():
            assert_scaled_points(child)
    elif isinstance(value, list):
        for child in value:
            assert_scaled_points(child)


def normalize_state_wire_timestamps(message):
    """Normalize only wall-clock values that vary in the recorded replay."""
    normalized = copy.deepcopy(message)
    normalized["timestamp"] = CONTRACT_WALL_TIME
    normalized["data"]["timestamp"] = CONTRACT_WALL_TIME
    for ball in normalized["data"]["balls"]:
        ball["timestamp"] = CONTRACT_OBSERVATION_TIME
    return normalized


def normalize_trajectory_wire_timestamps(message):
    """Normalize only trajectory wall-clock fields that vary per delivery."""
    normalized = copy.deepcopy(message)
    normalized["timestamp"] = CONTRACT_WALL_TIME
    normalized["data"]["timestamp"] = CONTRACT_WALL_TIME
    return normalized


@pytest.mark.integration
@pytest.mark.system
async def test_trajectory_broadcast_matches_versioned_real_socket_fixture(
    local_pipeline,
):
    """Pin trajectory serialization and delivery independently of physics."""
    async with websockets.connect(local_pipeline.url) as client:
        await receive_type(client, "connection")
        await client.send(
            json.dumps({"type": "subscribe", "data": {"streams": ["trajectory"]}})
        )
        await receive_type(client, "subscribed")
        await local_pipeline.broadcaster.broadcast_trajectory(
            lines=[
                {
                    "start": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
                    "end": {"x": 300.0, "y": 200.0, "scale": [1.0, 1.0]},
                    "type": "primary",
                },
                {
                    "start": {"x": 300.0, "y": 200.0, "scale": [1.0, 1.0]},
                    "end": {"x": 280.0, "y": 180.0, "scale": [1.0, 1.0]},
                    "type": "reflection",
                },
            ],
            collisions=[
                {
                    "type": "ball",
                    "ball_ids": [1, 2],
                    "position": {"x": 280.0, "y": 200.0, "scale": [1.0, 1.0]},
                }
            ],
            confidence=0.9,
            calculation_time_ms=2.5,
            prediction_status="live",
            spatial_valid=True,
            phase="motion",
            source_frame_number=42,
            source_timestamp=CONTRACT_OBSERVATION_TIME,
            source_timestamp_kind="replay_publication",
            map_revision="map-7",
            calibration_revision="calibration-3",
            prediction_settings={"max_bounces": 2},
        )
        message = await receive_type(client, "trajectory")

    expected = json.loads(WEBSOCKET_TRAJECTORY_FIXTURE.read_text(encoding="utf-8"))
    assert normalize_trajectory_wire_timestamps(message) == expected


@pytest.mark.integration
@pytest.mark.system
async def test_generated_frame_replay_to_real_websocket(
    local_pipeline, tmp_path, record_property
):
    frame = _generated_table_frame()
    vision = VisionModule(
        {
            "enable_ball_detection": False,
            "enable_tracking": False,
            "background_image_path": None,
            "phase_scheduler": {"enabled": False},
        }
    )
    vision.config.enable_ball_detection = True
    vision.detector = FixtureDetector()
    core = CoreModule()
    integration = IntegrationService(vision, core, local_pipeline.broadcaster)
    integration.running = True
    latencies = []
    ages = []
    correct = 0
    started = time.perf_counter()
    try:
        async with websockets.connect(local_pipeline.url) as client:
            welcome = await receive_type(client, "connection")
            client_id = welcome["data"]["client_id"]
            await local_pipeline.manager.subscribe_to_stream(
                client_id, StreamType.TRAJECTORY
            )
            await local_pipeline.manager.subscribe_to_stream(
                client_id, StreamType.STATE
            )
            for sequence in range(1, 7):
                published_at = time.time()
                frame_start = time.perf_counter()
                detection = await asyncio.to_thread(
                    vision._process_single_frame,
                    frame,
                    frame_number=sequence,
                    timestamp=published_at,
                    source_frame_number=sequence,
                    source_timestamp=published_at,
                    source_timestamp_kind="replay_publication",
                )
                assert detection is not None
                assert len(detection.balls) == 2
                assert detection.cue is not None
                converted = integration._convert_detection_to_core_format(detection)
                await local_pipeline.broadcaster.broadcast_game_state(
                    converted["balls"],
                    cue=converted.get("cue"),
                    table=converted.get("table"),
                )
                state_message = await receive_type(client, "state")
                state_data = state_message["data"]
                assert state_data["balls"] == converted["balls"]
                if sequence == 1:
                    expected_wire_message = json.loads(
                        WEBSOCKET_STATE_FIXTURE.read_text(encoding="utf-8")
                    )
                    assert normalize_state_wire_timestamps(state_message) == (
                        expected_wire_message
                    )
                assert_scaled_points(state_data["balls"])
                assert_scaled_points(state_data["cue"])
                assert_scaled_points(state_data["table"])
                assert all(
                    "scale" in corner for corner in state_data["table"]["corners"]
                )
                await integration._process_detection(detection)
                await integration._drain_prediction_task()
                message = await receive_type(client, "trajectory")
                data = message["data"]
                latencies.append((time.perf_counter() - frame_start) * 1000)
                ages.append(time.time() - data["source_timestamp"])
                assert data["source_frame_number"] == sequence
                assert data["source_timestamp_kind"] == "replay_publication"
                assert data["prediction_status"] == "live"
                assert len(data["lines"]) >= 2
                assert all(
                    math.isfinite(point[axis])
                    for line in data["lines"]
                    for point in (line["start"], line["end"])
                    for axis in ("x", "y")
                )
                assert_scaled_points(data["lines"])
                assert_scaled_points(data["collisions"])
                assert core.metrics.total_updates == sequence
                primary = [line for line in data["lines"] if line["type"] == "primary"]
                assert primary[0]["start"]["x"] == pytest.approx(1500.0, abs=0.01)
                assert primary[0]["start"]["y"] == pytest.approx(1080.0, abs=0.01)
                assert primary[0]["end"]["x"] > primary[0]["start"]["x"]
                assert primary[0]["end"]["y"] == pytest.approx(1080.0, abs=0.01)
                # Both sphere centers stay horizontal; contact is at one
                # diameter from the stationary object ball (72 canonical px).
                ball_collisions = [
                    c for c in data["collisions"] if c["type"] == "ball_ball"
                ]
                assert ball_collisions, data["collisions"]
                assert ball_collisions[0]["position"]["y"] == pytest.approx(
                    1080.0, abs=0.01
                )
                secondary = [
                    line for line in data["lines"] if line["type"] == "secondary"
                ]
                assert secondary, data["lines"]
                # Incoming-center collision coordinates remain one diameter
                # behind the observed target center. The outgoing branch must
                # start at the target center itself, without a collision jump.
                assert secondary[0]["start"]["x"] == pytest.approx(2400.0, abs=0.01)
                assert secondary[0]["start"]["y"] == pytest.approx(1080.0, abs=0.01)
                assert secondary[0]["end"]["x"] > secondary[0]["start"]["x"]
                correct += 1
            connection = local_pipeline.handler.connections[client_id]
            measurements = ReplayMeasurements(
                frame_count=len(latencies),
                elapsed_seconds=time.perf_counter() - started,
                mean_frame_to_delivery_ms=sum(latencies) / len(latencies),
                max_frame_to_delivery_ms=max(latencies),
                maximum_publication_age_seconds=max(ages),
                maximum_queue_depth=connection.max_outbound_queue_depth,
                dropped_updates=connection.dropped_updates,
                correct_predictions=correct,
            )
            report = json.dumps(asdict(measurements), indent=2)
            (tmp_path / "replay-measurements.json").write_text(report)
            record_property("replay_measurements", report)
            print(f"REPLAY_MEASUREMENTS={report}")
            assert correct == measurements.frame_count
            assert connection.max_outbound_queue_depth <= 2
    finally:
        await integration.stop()


@pytest.mark.integration
@pytest.mark.system
async def test_real_clients_disconnect_reconnect_and_slow_send_isolation(
    local_pipeline,
):
    async with websockets.connect(local_pipeline.url + "?slow=1") as slow:
        slow_id = (await receive_type(slow, "connection"))["data"]["client_id"]
        async with websockets.connect(local_pipeline.url) as healthy:
            healthy_id = (await receive_type(healthy, "connection"))["data"][
                "client_id"
            ]
            for client_id in (slow_id, healthy_id):
                await local_pipeline.manager.subscribe_to_stream(
                    client_id, StreamType.STATE
                )
            slow_connection = local_pipeline.handler.connections[slow_id]
            await local_pipeline.manager.broadcast_to_stream(StreamType.STATE, {"n": 0})
            await asyncio.wait_for(local_pipeline.gate.entered.wait(), timeout=5)
            assert (await receive_type(healthy, "state"))["data"]["n"] == 0
            for sequence in range(1, 12):
                await local_pipeline.manager.broadcast_to_stream(
                    StreamType.STATE, {"n": sequence}
                )
                assert (await receive_type(healthy, "state"))["data"]["n"] == sequence
            assert not local_pipeline.gate.release.is_set()
            assert len(slow_connection.outbound_messages) == 1
            assert slow_connection.dropped_updates == 10
            print(
                "SLOW_CLIENT_MEASUREMENTS="
                + json.dumps(
                    {
                        "healthy_updates_received": 12,
                        "blocked_client_queue_depth": len(
                            slow_connection.outbound_messages
                        ),
                        "blocked_client_dropped_updates": slow_connection.dropped_updates,
                        "queue_capacity": local_pipeline.handler.replaceable_queue_capacity,
                    }
                )
            )
            local_pipeline.gate.release.set()
            assert (await receive_type(slow, "state"))["data"]["n"] == 0
            assert (await receive_type(slow, "state"))["data"]["n"] == 11
            healthy_connection = local_pipeline.handler.connections[healthy_id]
        await asyncio.wait_for(healthy_connection.closed_event.wait(), timeout=5)
        assert healthy_id not in local_pipeline.manager.sessions
        async with websockets.connect(local_pipeline.url) as reconnected:
            new_id = (await receive_type(reconnected, "connection"))["data"][
                "client_id"
            ]
            assert new_id != healthy_id
            await local_pipeline.manager.subscribe_to_stream(new_id, StreamType.STATE)
            await local_pipeline.manager.broadcast_to_stream(
                StreamType.STATE, {"n": 12}
            )
            assert (await receive_type(reconnected, "state"))["data"]["n"] == 12
