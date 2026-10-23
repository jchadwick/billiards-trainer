"""Deterministic contracts for IntegrationService's live-state boundary.

These tests exercise the real integration service methods with small in-memory
Vision/Core doubles. They protect the properties that matter to a live view:
coordinates retain their source scale, concurrent callers cannot reorder Core
state, and an older observation must never replace a newer one.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.common.models import Ball
from backend.integration_service import IntegrationService
from backend.vision.models import BallType


def _service() -> IntegrationService:
    """Create a minimally initialized service without hardware or app startup."""
    service = IntegrationService.__new__(IntegrationService)
    service.vision = SimpleNamespace(stop_capture=Mock())
    service.core = SimpleNamespace(update_state=AsyncMock())
    service.running = False
    service.integration_task = None
    service._event_loop = None
    # Core invokes this callback from a synchronous update path, so production
    # intentionally protects this queue with threading.Lock rather than the
    # event-loop lock used around update_state.
    service._state_event_lock = threading.Lock()
    service._pending_state_event = None
    service._state_event_drain_scheduled = False
    service._state_event_drain_task = None
    service.coalesced_state_event_count = 0
    service._vision_instance_id = "vision-run-test"
    service._vision_process_id = 1234
    service._last_processed_observation_identity = None
    service._last_stale_observation_identity = None
    service.duplicate_observation_count = 0
    service.stale_observation_count = 0
    service.maximum_observation_age_seconds = 0.5
    service.latest_observation_status = None
    service._latest_observation_received_at = None
    service.error_count = 0
    service._core_update_provenance_lock = asyncio.Lock()
    service._state_event_observation_lock = threading.Lock()
    service._pending_state_event_observations = []
    service._check_trajectory_calculation = AsyncMock()
    service._prediction_task = None
    service._prediction_lifecycle_generation = 0
    service._pending_prediction = None
    return service


def _detection(
    *,
    frame_number: int,
    timestamp: float,
    source_process_id: int = 42,
    source_publication_monotonic_ns: int | None = None,
    frame_size: tuple[int, int] = (1280, 720),
    balls: list[Ball] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_number=frame_number,
        timestamp=timestamp,
        source_frame_number=frame_number,
        source_timestamp=timestamp,
        source_timestamp_kind="shared_memory_publication",
        source_process_id=source_process_id,
        source_publication_monotonic_ns=source_publication_monotonic_ns,
        frame_publication_age_seconds=None,
        frame_size=frame_size,
        balls=balls or [],
        cue=None,
        table=None,
    )


def _flattened_core_event(frame_number: int, timestamp: float) -> dict[str, object]:
    """Return the full Core state shape required to consume source provenance."""
    return {
        "balls": [],
        "table": {},
        "timestamp": timestamp,
        "frame_number": frame_number,
        "game_type": "eight_ball",
        "cue": None,
        "scores": [],
        "coordinate_metadata": {},
    }


def test_pixel_ball_conversion_uses_object_resolution_and_preserves_frame_size():
    service = _service()
    ball = Ball(
        position=(100.0, 80.0),
        radius=4.0,
        ball_type=BallType.SOLID_6,
        confidence=0.95,
        track_id=6,
        velocity=(2.0, 4.0),
        acceleration=(1.0, 2.0),
        last_seen=10.0,
        position_history=[(90.0, 70.0)],
        coordinate_space="pixel",
        source_resolution=(640, 480),
    )

    converted = service._convert_detection_to_core_format(
        _detection(frame_number=7, timestamp=10.0, balls=[ball])
    )

    assert converted["source_resolution"] == [1280, 720]
    serialized_ball = converted["balls"][0]
    assert serialized_ball["position"] == {
        "x": 600.0,
        "y": 360.0,
        "scale": [1.0, 1.0],
    }
    assert serialized_ball["velocity"]["x"] == 12.0
    assert serialized_ball["velocity"]["y"] == 18.0
    assert serialized_ball["radius"] == 24.0
    assert serialized_ball["source_to_canonical_scale"] == [6.0, 4.5]
    assert serialized_ball["position_history"][0]["x"] == 540.0
    assert serialized_ball["position_history"][0]["y"] == 315.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("older_frame", "older_timestamp"),
    [(8, 11.0), (10, 10.8)],
    ids=["lower-frame-number", "older-source-timestamp"],
)
async def test_older_source_observation_cannot_regress_latest_core_state(
    older_frame: int, older_timestamp: float
):
    service = _service()
    newest = _detection(frame_number=9, timestamp=10.9)
    delayed = _detection(frame_number=older_frame, timestamp=older_timestamp)

    assert await service._process_detection(newest) is True
    assert await service._process_detection(delayed) is False

    assert service.core.update_state.await_count == 1
    assert service.core.update_state.await_args.args[0]["frame_number"] == 9
    assert service.latest_observation_status["source_frame_number"] == 9
    assert service.stale_observation_count == 1


@pytest.mark.asyncio
async def test_older_same_host_publication_tick_cannot_regress_latest_core_state():
    service = _service()
    now_ns = time.monotonic_ns()
    newest = _detection(
        frame_number=9,
        timestamp=10.9,
        source_publication_monotonic_ns=now_ns - 100_000_000,
    )
    delayed = _detection(
        frame_number=10,
        timestamp=11.0,
        source_publication_monotonic_ns=now_ns - 110_000_000,
    )

    assert await service._process_detection(newest) is True
    assert await service._process_detection(delayed) is False

    assert service.core.update_state.await_count == 1
    assert service.core.update_state.await_args.args[0]["frame_number"] == 9
    assert service.latest_observation_status["source_frame_number"] == 9
    assert service.stale_observation_count == 1


@pytest.mark.asyncio
async def test_concurrent_core_updates_keep_observation_provenance_in_admission_order():
    service = _service()
    first_core_update_entered = asyncio.Event()
    release_first_core_update = asyncio.Event()
    core_frames: list[int] = []
    event_provenance: list[tuple[int, int | None]] = []

    async def update_state(state: dict[str, object]) -> None:
        frame_number = int(state["frame_number"])
        core_frames.append(frame_number)
        if frame_number == 1:
            first_core_update_entered.set()
            await release_first_core_update.wait()

        matched, status = service._take_state_event_observation(
            _flattened_core_event(frame_number, float(state["timestamp"]))
        )
        event_provenance.append(
            (frame_number, status.get("source_frame_number") if matched else None)
        )

    service.core.update_state = AsyncMock(side_effect=update_state)
    first = asyncio.create_task(
        service._process_detection(_detection(frame_number=1, timestamp=1.0))
    )
    await asyncio.wait_for(first_core_update_entered.wait(), timeout=1.0)
    second = asyncio.create_task(
        service._process_detection(_detection(frame_number=2, timestamp=2.0))
    )
    await asyncio.sleep(0)

    # The integration lock makes the second caller wait before mutating Core,
    # leaving the in-flight observation and its provenance paired correctly.
    assert core_frames == [1]
    release_first_core_update.set()
    assert await asyncio.gather(first, second) == [True, True]

    assert core_frames == [1, 2]
    assert event_provenance == [(1, 1), (2, 2)]
    assert service.latest_observation_status["source_frame_number"] == 2


@pytest.mark.asyncio
async def test_shutdown_cancels_integration_loop_and_isolates_capture_stop_failure():
    service = _service()
    service.running = True
    service.integration_task = asyncio.create_task(asyncio.Event().wait())
    service._event_loop = asyncio.get_running_loop()
    service._pending_state_event = (
        "state_updated",
        {"frame_number": 12},
        {"source_frame_number": 12},
    )
    service.vision.stop_capture.side_effect = RuntimeError("camera already detached")

    await service.stop()

    assert service.running is False
    assert service.integration_task.cancelled()
    assert service._event_loop is None
    assert service._pending_state_event is None
    service.vision.stop_capture.assert_called_once_with()
