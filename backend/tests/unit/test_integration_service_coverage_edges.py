"""Edge contracts for source freshness, Core correlation, and recovery."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.integration_service as integration_service_module
from backend.integration_service import IntegrationService


def _service() -> IntegrationService:
    """Build the live admission boundary without camera or Core startup."""
    service = IntegrationService.__new__(IntegrationService)
    service.vision = SimpleNamespace(stop_capture=Mock())
    service.core = SimpleNamespace(update_state=AsyncMock())
    service.running = False
    service._event_loop = None
    service._state_event_lock = threading.Lock()
    service._pending_state_event = None
    service._state_event_drain_scheduled = False
    service._state_event_drain_task = None
    service.coalesced_state_event_count = 0
    service._vision_instance_id = "vision-run-a"
    service._vision_process_id = 123
    service._last_processed_observation_identity = None
    service._last_stale_observation_identity = None
    service._latest_processed_source_watermark = None
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
    service._convert_detection_to_core_format = Mock(
        side_effect=lambda detection: {
            "frame_number": detection.frame_number,
            "timestamp": detection.timestamp,
            "balls": [],
        }
    )
    return service


def _detection(
    *,
    frame_number: int,
    timestamp: float,
    source_process_id: int = 42,
    source_frame_number: int | None = None,
    map_revision: int | None = None,
    calibration_revision: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_number=frame_number,
        timestamp=timestamp,
        source_frame_number=(
            frame_number if source_frame_number is None else source_frame_number
        ),
        source_timestamp=timestamp,
        source_timestamp_kind="shared_memory_publication",
        source_process_id=source_process_id,
        source_publication_monotonic_ns=None,
        frame_publication_age_seconds=None,
        map_revision=map_revision,
        calibration_revision=calibration_revision,
        balls=[],
        cue=None,
        table=None,
    )


def _core_state(frame_number: int = 1, timestamp: float = 1.0) -> dict:
    return {
        "balls": [],
        "table": {},
        "timestamp": timestamp,
        "frame_number": frame_number,
        "game_type": "eight_ball",
        "cue": None,
        "scores": {},
        "coordinate_metadata": {},
    }


@pytest.mark.asyncio
async def test_failed_core_update_does_not_advance_ordering_watermark() -> None:
    service = _service()
    service.core.update_state.side_effect = [
        None,
        RuntimeError("transient Core failure"),
        None,
    ]

    assert await service._process_detection(_detection(frame_number=10, timestamp=1.0))
    # Admission succeeds even though Core reports this update failure.
    assert await service._process_detection(_detection(frame_number=12, timestamp=1.2))

    # Frame 11 is newer than the last accepted frame (10) but older than the
    # failed attempt (12); the failed attempt must not make valid state stale.
    assert await service._process_detection(_detection(frame_number=11, timestamp=1.3))

    assert service.core.update_state.await_count == 3
    assert service.error_count == 1
    assert service.stale_observation_count == 0
    assert service.latest_observation_status["source_frame_number"] == 11
    assert service._latest_processed_source_watermark.source_frame_number == 11
    assert service._pending_state_event_observations == []


@pytest.mark.asyncio
async def test_restarted_vision_run_can_restart_frame_sequence_in_same_process() -> (
    None
):
    service = _service()

    assert await service._process_detection(_detection(frame_number=900, timestamp=5.0))
    service._vision_instance_id = "vision-run-after-restart"
    restarted = _detection(frame_number=1, timestamp=5.1)

    assert await service._process_detection(restarted)

    assert service.core.update_state.await_count == 2
    assert service.stale_observation_count == 0
    assert service.duplicate_observation_count == 0
    assert service.latest_observation_status["source_frame_number"] == 1
    assert service.latest_observation_status["vision_instance_id"] == (
        "vision-run-after-restart"
    )


@pytest.mark.asyncio
async def test_map_revision_change_does_not_reprocess_the_same_camera_observation() -> (
    None
):
    service = _service()
    detection = _detection(
        frame_number=27,
        timestamp=8.0,
        map_revision=3,
        calibration_revision="cal-5",
    )

    assert await service._process_detection(detection)
    revised_metadata = _detection(
        frame_number=27,
        timestamp=8.0,
        map_revision=4,
        calibration_revision="cal-6",
    )

    assert not await service._process_detection(revised_metadata)

    assert service.core.update_state.await_count == 1
    assert service.duplicate_observation_count == 1
    assert service.latest_observation_status["map_revision"] == 3
    assert service.latest_observation_status["calibration_revision"] == "cal-5"


def test_state_event_correlation_ignores_malformed_events_without_consuming_status() -> (
    None
):
    service = _service()
    token = object()
    original_status = {"source_frame_number": 44, "map_revision": 9}
    service._register_state_event_observation(token, original_status)

    assert service._take_state_event_observation({"frame_number": 44}) == (False, None)
    malformed_frame = _core_state(frame_number=44)
    malformed_frame["frame_number"] = True
    assert service._take_state_event_observation(malformed_frame) == (False, None)
    malformed_timestamp = _core_state(frame_number=44)
    malformed_timestamp["timestamp"] = float("nan")
    assert service._take_state_event_observation(malformed_timestamp) == (False, None)
    assert len(service._pending_state_event_observations) == 1

    matched, status = service._take_state_event_observation(
        _core_state(frame_number=44)
    )

    assert matched is True
    assert status == original_status
    assert status is not original_status
    status["map_revision"] = 100
    assert original_status["map_revision"] == 9
    assert service._pending_state_event_observations == []


@pytest.mark.parametrize("clock_value", [0, True, 1.25])
def test_invalid_monotonic_clock_reading_is_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, clock_value: object
) -> None:
    detection = _detection(frame_number=1, timestamp=1.0)
    detection.source_publication_monotonic_ns = 1
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(monotonic_ns=Mock(return_value=clock_value)),
    )

    age, error = IntegrationService._refresh_observation_publication_age(detection)

    assert age is None
    assert error == "monotonic_clock_unavailable"
    assert detection.frame_publication_age_seconds is None


@pytest.mark.asyncio
async def test_core_failure_discards_unconsumed_event_correlation_token() -> None:
    service = _service()
    service.core.update_state.side_effect = RuntimeError("Core unavailable")

    assert await service._process_detection(_detection(frame_number=5, timestamp=2.0))

    assert service.latest_observation_status is None
    assert service._latest_processed_source_watermark is None
    assert service._pending_state_event_observations == []
    assert service.error_count == 1
