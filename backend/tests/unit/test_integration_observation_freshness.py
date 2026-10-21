"""Regression coverage for unique source observations at the Core boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.integration_service as integration_service_module
from backend.integration_service import (
    BroadcastMetrics,
    CircuitBreaker,
    IntegrationService,
)


def _service() -> IntegrationService:
    service = IntegrationService.__new__(IntegrationService)
    service._vision_instance_id = "vision-run-a"
    service._vision_process_id = 100
    service._last_processed_observation_identity = None
    service.duplicate_observation_count = 0
    service.latest_observation_status = None
    service._latest_observation_received_at = None
    service.error_count = 0
    service._audit_recorder = object()
    service._audit_domains = {"state": True}
    service.core = SimpleNamespace(update_state=AsyncMock())
    service._convert_detection_to_core_format = Mock(
        return_value={"balls": [], "frame_number": 1}
    )
    service._check_trajectory_calculation = AsyncMock()
    service._record_audit_event = Mock()
    return service


def _detection(
    *,
    frame_number: int | None = 7,
    timestamp: float | None = 10.0,
    source_frame_number: int | None = 7,
    source_timestamp: float | None = 10.0,
    source_process_id: int | None = 42,
    publication_monotonic_ns: int | None = 9_000_000_000,
):
    return SimpleNamespace(
        frame_number=frame_number,
        timestamp=timestamp,
        balls=[],
        cue=None,
        source_frame_number=source_frame_number,
        source_timestamp=source_timestamp,
        source_timestamp_kind="shared_memory_publication",
        source_process_id=source_process_id,
        source_publication_monotonic_ns=publication_monotonic_ns,
    )


@pytest.mark.asyncio
async def test_disabled_audit_skips_profiler_timers_and_event_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._audit_recorder = None
    service._audit_domains = {"state": False}

    def fail_if_profiled() -> int:
        raise AssertionError("disabled state audit must not read the profiling clock")

    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(perf_counter_ns=fail_if_profiled, monotonic=lambda: 1.0),
    )

    assert await service._process_detection(_detection()) is True
    assert service.core.update_state.await_count == 1
    service._record_audit_event.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_cached_observation_does_not_refresh_or_advance_core(
    monkeypatch: pytest.MonkeyPatch,
):
    service = _service()
    monotonic = Mock(side_effect=[100.0, 110.0])
    real_time_module = integration_service_module.time
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic=monotonic,
            perf_counter_ns=real_time_module.perf_counter_ns,
        ),
    )
    detection = _detection()

    assert await service._process_detection(detection) is True
    first_received_at = service._latest_observation_received_at
    first_status = service.latest_observation_status
    assert first_received_at == 100.0
    assert service.core.update_state.await_count == 1
    assert service._check_trajectory_calculation.await_count == 1

    assert await service._process_detection(detection) is False

    assert service.core.update_state.await_count == 1
    assert service._check_trajectory_calculation.await_count == 1
    assert service._latest_observation_received_at == first_received_at
    assert service.latest_observation_status is first_status
    assert service.duplicate_observation_count == 1
    assert monotonic.call_count == 1
    duplicate_events = [
        call.args[1]
        for call in service._record_audit_event.call_args_list
        if call.args[1].get("event_type") == "integration_observation_duplicate"
    ]
    assert len(duplicate_events) == 1

    service.running = False
    service.frame_count = 1
    service.broadcast_metrics = BroadcastMetrics()
    service.circuit_breaker = CircuitBreaker()
    assert service.get_statistics()["duplicate_observations"] == 1


@pytest.mark.asyncio
async def test_integration_loop_counts_only_unique_admitted_observations():
    service = _service()
    service.running = True
    service.target_fps = 1_000
    service.log_interval_frames = 10
    service.frame_count = 0
    service.vision = SimpleNamespace()
    detection = _detection()
    poll_count = 0

    async def get_detection():
        nonlocal poll_count
        poll_count += 1
        if poll_count == 2:
            # End the loop after returning the repeated cached result.
            service.running = False
        return detection

    service._get_vision_detection = get_detection

    await service._integration_loop()

    assert poll_count == 2
    assert service.frame_count == 1
    assert service.duplicate_observation_count == 1
    assert service.core.update_state.await_count == 1


@pytest.mark.asyncio
async def test_core_failure_preserves_last_applied_freshness_and_skips_trajectory(
    monkeypatch: pytest.MonkeyPatch,
):
    service = _service()
    service.running = True
    service.target_fps = 1_000
    service.log_interval_frames = 10
    service.frame_count = 0
    service.vision = SimpleNamespace()
    service.core.update_state.side_effect = [
        None,
        RuntimeError("state update failed"),
    ]
    real_time_module = integration_service_module.time
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic=Mock(return_value=100.0),
            perf_counter_ns=real_time_module.perf_counter_ns,
        ),
    )
    detections = [
        _detection(),
        _detection(
            frame_number=8,
            timestamp=10.1,
            source_frame_number=8,
            source_timestamp=10.1,
            publication_monotonic_ns=9_100_000_000,
        ),
        _detection(
            frame_number=8,
            timestamp=10.1,
            source_frame_number=8,
            source_timestamp=10.1,
            publication_monotonic_ns=9_100_000_000,
        ),
    ]
    poll_count = 0

    async def get_detection() -> SimpleNamespace:
        nonlocal poll_count
        poll_count += 1
        if poll_count == len(detections):
            service.running = False
        return detections[poll_count - 1]

    service._get_vision_detection = get_detection

    await service._integration_loop()

    assert service.frame_count == 2  # Both unique source observations were tried.
    assert service.duplicate_observation_count == 1
    assert service.error_count == 1
    assert service.core.update_state.await_count == 2
    assert service._check_trajectory_calculation.await_count == 1
    assert service.latest_observation_status["source_frame_number"] == 7
    assert service._latest_observation_received_at == 100.0
    service.broadcast_metrics = BroadcastMetrics()
    service.circuit_breaker = CircuitBreaker()
    statistics = service.get_statistics()
    assert statistics["frames_processed"] == 2
    assert statistics["errors"] == 1
    processed_event = next(
        call.args[1]
        for call in service._record_audit_event.call_args_list
        if call.args[1].get("event_type") == "integration_observation_processed"
        and call.args[1].get("core_update_status") == "failed"
    )
    assert processed_event["core_update_status"] == "failed"
    assert processed_event["trajectory_admission_status"] == (
        "skipped_core_update_failed"
    )
    assert processed_event["integration_lock_wait_seconds"] >= 0


@pytest.mark.asyncio
async def test_new_frame_and_restarted_writer_with_reset_counter_are_accepted():
    service = _service()

    assert await service._process_detection(_detection()) is True
    assert (
        await service._process_detection(
            _detection(
                frame_number=8,
                timestamp=10.1,
                source_frame_number=8,
                source_timestamp=10.1,
                publication_monotonic_ns=9_100_000_000,
            )
        )
        is True
    )
    # The restarted producer resets its frame number, but its process identity
    # and publication provenance distinguish this source observation.
    assert (
        await service._process_detection(
            _detection(
                frame_number=1,
                timestamp=11.0,
                source_frame_number=1,
                source_timestamp=11.0,
                source_process_id=43,
                publication_monotonic_ns=10_000_000_000,
            )
        )
        is True
    )

    assert service.core.update_state.await_count == 3
    assert service._check_trajectory_calculation.await_count == 3
    assert service.duplicate_observation_count == 0


@pytest.mark.asyncio
async def test_legacy_timestamp_fallback_deduplicates_same_cached_result():
    service = _service()
    detection = _detection(
        frame_number=4,
        timestamp=12.5,
        source_frame_number=None,
        source_timestamp=None,
        source_process_id=None,
        publication_monotonic_ns=None,
    )

    assert await service._process_detection(detection) is True
    assert await service._process_detection(detection) is False
    assert service.core.update_state.await_count == 1
    assert service.duplicate_observation_count == 1


@pytest.mark.asyncio
async def test_result_without_any_usable_identity_is_processed_without_deduplication():
    service = _service()
    detection = _detection(
        frame_number=None,
        timestamp=None,
        source_frame_number=None,
        source_timestamp=None,
        source_process_id=None,
        publication_monotonic_ns=None,
    )

    assert await service._process_detection(detection) is True
    assert await service._process_detection(detection) is True
    assert service.core.update_state.await_count == 2
    assert service.duplicate_observation_count == 0


@pytest.mark.asyncio
async def test_coordinate_rejection_does_not_mark_observation_as_processed():
    service = _service()
    service._convert_detection_to_core_format = Mock(
        side_effect=[
            integration_service_module.CoordinateConversionError("invalid space"),
            {"balls": [], "frame_number": 7},
        ]
    )
    detection = _detection()

    assert await service._process_detection(detection) is False
    assert service._latest_observation_received_at is None
    assert service._last_processed_observation_identity is None

    assert await service._process_detection(detection) is True
    assert service.core.update_state.await_count == 1
    assert service.duplicate_observation_count == 0


@pytest.mark.asyncio
async def test_core_state_events_use_matching_observation_status_before_update_returns():
    service = _service()
    previous_status = {"source_frame_number": 3, "ball_detection_status": "observed"}
    service.latest_observation_status = previous_status
    service._latest_observation_received_at = 42.0
    service._event_loop = asyncio.get_running_loop()
    service._convert_detection_to_core_format = Mock(
        return_value={"balls": [], "frame_number": 7, "timestamp": 10.0}
    )
    delivered = asyncio.Event()
    published_statuses: list[dict | None] = []

    async def capture_state_event(
        event_type: str, event_data: dict, observation_status: dict | None
    ) -> None:
        published_statuses.append(observation_status)
        if len(published_statuses) == 2:
            delivered.set()

    service._on_state_updated_async = capture_state_event

    async def core_update_with_synchronous_events(state: dict) -> None:
        # Core's state event is synchronous from its event manager, even when
        # async_processing dispatches subscribers on a worker thread.
        await asyncio.to_thread(
            service._on_state_updated,
            "state_updated",
            {
                "balls": [],
                "table": {},
                "timestamp": 9.9,
                "frame_number": 99,
            },
        )
        await asyncio.to_thread(
            service._on_state_updated,
            "state_updated",
            {
                "balls": [],
                "table": {},
                "timestamp": state["timestamp"],
                "frame_number": state["frame_number"],
                "game_type": "practice",
                "cue": None,
                "scores": {},
                "coordinate_metadata": None,
            },
        )
        # An event may already have been emitted even when subsequent Core
        # work fails. It must carry request-local status, while the service's
        # freshness pointer still describes the last successful update.
        raise RuntimeError("Core failed after emitting its state event")

    service.core.update_state = core_update_with_synchronous_events
    detection = _detection()
    detection.ball_detection_status = "failed"

    assert await service._process_detection(detection) is True
    await asyncio.wait_for(delivered.wait(), timeout=1.0)

    assert len(published_statuses) == 2
    assert published_statuses[0] == previous_status
    assert published_statuses[1]["source_frame_number"] == 7
    assert published_statuses[1]["ball_detection_status"] == "failed"
    assert service.latest_observation_status is previous_status
    assert service._latest_observation_received_at == 42.0
    assert service._pending_state_event_observations == []
