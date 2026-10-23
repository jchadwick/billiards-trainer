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
    service._last_stale_observation_identity = None
    service.duplicate_observation_count = 0
    service.stale_observation_count = 0
    service.maximum_observation_age_seconds = 0.5
    service.latest_observation_status = None
    service._latest_observation_received_at = None
    service.error_count = 0
    service._state_event_lock = integration_service_module.threading.Lock()
    service._pending_state_event = None
    service._state_event_drain_scheduled = False
    service._state_event_drain_task = None
    service.coalesced_state_event_count = 0
    service.broadcast_metrics = BroadcastMetrics()
    service.circuit_breaker = CircuitBreaker()
    service.core = SimpleNamespace(update_state=AsyncMock())
    service._convert_detection_to_core_format = Mock(
        return_value={"balls": [], "frame_number": 1}
    )
    service._check_trajectory_calculation = AsyncMock()
    return service


def _detection(
    *,
    frame_number: int | None = 7,
    timestamp: float | None = 10.0,
    source_frame_number: int | None = 7,
    source_timestamp: float | None = 10.0,
    source_process_id: int | None = 42,
    publication_monotonic_ns: int | None = None,
    frame_publication_age_seconds: float | None = None,
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
        frame_publication_age_seconds=frame_publication_age_seconds,
    )


@pytest.mark.asyncio
async def test_stale_source_observation_is_rejected_before_core_update():
    service = _service()
    stale = _detection(frame_publication_age_seconds=0.5)

    assert await service._process_detection(stale) is False
    assert service.core.update_state.await_count == 0
    assert service.stale_observation_count == 1
    assert service._last_processed_observation_identity is None

    # The same stale source frame is not re-counted on every integration poll.
    assert await service._process_detection(stale) is False
    assert service.stale_observation_count == 1
    assert service.duplicate_observation_count == 1


@pytest.mark.asyncio
async def test_monotonic_publication_age_is_refreshed_at_core_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    detection = _detection(
        publication_monotonic_ns=1_000_000_000,
        frame_publication_age_seconds=0.1,
    )
    actual_time = integration_service_module.time
    monotonic_ns = Mock(side_effect=[1_100_000_000, 1_200_000_000])
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic_ns=monotonic_ns,
            monotonic=Mock(return_value=12.0),
            perf_counter_ns=actual_time.perf_counter_ns,
        ),
    )

    assert await service._process_detection(detection) is True

    assert monotonic_ns.call_count == 2
    assert detection.frame_publication_age_seconds == pytest.approx(0.2)
    core_state = service.core.update_state.await_args.args[0]
    assert core_state["observation_status"][
        "frame_publication_age_seconds"
    ] == pytest.approx(0.2)
    assert service.latest_observation_status[
        "frame_publication_age_seconds"
    ] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_observation_that_ages_out_while_waiting_for_core_lock_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    detection = _detection(publication_monotonic_ns=1_000_000_000)
    actual_time = integration_service_module.time
    monotonic_ns = Mock(side_effect=[1_000_000_000, 1_500_000_000])
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic_ns=monotonic_ns,
            monotonic=Mock(return_value=12.0),
            perf_counter_ns=actual_time.perf_counter_ns,
        ),
    )

    assert await service._process_detection(detection) is False

    assert monotonic_ns.call_count == 2
    assert service.core.update_state.await_count == 0
    assert service._last_processed_observation_identity is None
    assert service._last_stale_observation_identity is not None
    assert service.stale_observation_count == 1


@pytest.mark.asyncio
async def test_negative_monotonic_delta_rejects_stamped_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    detection = _detection(publication_monotonic_ns=2_000_000_000)
    actual_time = integration_service_module.time
    monotonic_ns = Mock(return_value=1_000_000_000)
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic_ns=monotonic_ns,
            monotonic=Mock(return_value=12.0),
            perf_counter_ns=actual_time.perf_counter_ns,
        ),
    )

    assert await service._process_detection(detection) is False

    assert monotonic_ns.call_count == 1
    assert detection.frame_publication_age_seconds is None
    assert service.core.update_state.await_count == 0


@pytest.mark.asyncio
async def test_unavailable_monotonic_clock_rejects_stamped_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    detection = _detection(
        publication_monotonic_ns=1_000_000_000,
        frame_publication_age_seconds=0.0,
    )
    actual_time = integration_service_module.time
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic_ns=Mock(side_effect=OSError("clock unavailable")),
            monotonic=Mock(return_value=12.0),
            perf_counter_ns=actual_time.perf_counter_ns,
        ),
    )

    assert await service._process_detection(detection) is False

    assert detection.frame_publication_age_seconds is None
    assert service.core.update_state.await_count == 0


@pytest.mark.asyncio
async def test_legacy_age_is_preserved_without_monotonic_publication_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    detection = _detection(
        publication_monotonic_ns=None,
        frame_publication_age_seconds=0.4,
    )
    actual_time = integration_service_module.time
    # There is deliberately no monotonic_ns attribute: legacy DetectionResult
    # values must continue to use the age calculated by the Vision producer.
    monkeypatch.setattr(
        integration_service_module,
        "time",
        SimpleNamespace(
            monotonic=Mock(return_value=12.0),
            perf_counter_ns=actual_time.perf_counter_ns,
        ),
    )

    assert await service._process_detection(detection) is True

    core_state = service.core.update_state.await_args.args[0]
    assert core_state["observation_status"][
        "frame_publication_age_seconds"
    ] == pytest.approx(0.4)


@pytest.mark.asyncio
@pytest.mark.asyncio
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
            publication_monotonic_ns=None,
        ),
        _detection(
            frame_number=8,
            timestamp=10.1,
            source_frame_number=8,
            source_timestamp=10.1,
            publication_monotonic_ns=None,
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


@pytest.mark.asyncio
@pytest.mark.asyncio
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
                publication_monotonic_ns=None,
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
                publication_monotonic_ns=None,
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
    published_states: list[dict] = []

    async def capture_state_event(
        event_type: str, event_data: dict, observation_status: dict | None
    ) -> None:
        published_statuses.append(observation_status)
        published_states.append(event_data)
        if len(published_statuses) == 1:
            delivered.set()

    service._on_state_updated_async = capture_state_event

    async def core_update_with_synchronous_events(state: dict) -> None:
        # Core's state event is synchronous from its event manager, even when
        # async_processing dispatches subscribers on a worker thread. Emit both
        # before yielding to the loop to verify that only the newest snapshot
        # consumes publication capacity.
        def emit_events() -> None:
            service._on_state_updated(
                "state_updated",
                {
                    "balls": [],
                    "table": {},
                    "timestamp": 9.9,
                    "frame_number": 99,
                },
            )
            service._on_state_updated(
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

        await asyncio.to_thread(emit_events)
        # An event may already have been emitted even when subsequent Core
        # work fails. It must carry request-local status, while the service's
        # freshness pointer still describes the last successful update.
        raise RuntimeError("Core failed after emitting its state event")

    service.core.update_state = core_update_with_synchronous_events
    detection = _detection()
    detection.ball_detection_status = "failed"

    assert await service._process_detection(detection) is True
    await asyncio.wait_for(delivered.wait(), timeout=1.0)

    assert len(published_statuses) == 1
    assert published_statuses[0]["source_frame_number"] == 7
    assert published_statuses[0]["ball_detection_status"] == "failed"
    assert published_states[0]["frame_number"] == 7
    assert service.coalesced_state_event_count == 1
    assert service.latest_observation_status is previous_status
    assert service._latest_observation_received_at == 42.0
    assert service._pending_state_event_observations == []


@pytest.mark.asyncio
async def test_state_event_flood_keeps_one_pending_latest_snapshot() -> None:
    service = _service()
    service._event_loop = asyncio.get_running_loop()
    first_publish_started = asyncio.Event()
    release_first_publish = asyncio.Event()
    newest_publish_finished = asyncio.Event()
    published_frame_numbers: list[int] = []

    async def blocked_state_event(
        _event_type: str, event_data: dict, _observation_status: dict | None
    ) -> None:
        frame_number = event_data["frame_number"]
        published_frame_numbers.append(frame_number)
        if frame_number == 1:
            first_publish_started.set()
            await release_first_publish.wait()
        if frame_number == 101:
            newest_publish_finished.set()

    service._on_state_updated_async = blocked_state_event

    service._on_state_updated("state_updated", {"frame_number": 1})
    await asyncio.wait_for(first_publish_started.wait(), timeout=1.0)
    drain_task = service._state_event_drain_task
    assert drain_task is not None
    assert not drain_task.done()

    for frame_number in range(2, 102):
        service._on_state_updated("state_updated", {"frame_number": frame_number})

    assert service._pending_state_event is not None
    assert service._pending_state_event[1]["frame_number"] == 101
    assert service._state_event_drain_task is drain_task
    assert not drain_task.done()
    assert service.coalesced_state_event_count == 99

    release_first_publish.set()
    await asyncio.wait_for(newest_publish_finished.wait(), timeout=1.0)
    await asyncio.wait_for(drain_task, timeout=1.0)

    assert published_frame_numbers == [1, 101]
    assert service._pending_state_event is None
    assert service._state_event_drain_scheduled is False


@pytest.mark.asyncio
async def test_failed_latest_state_publication_is_not_retried() -> None:
    service = _service()
    attempts = 0

    async def fail_publication(*_args, **_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise ConnectionError("subscriber is temporarily unavailable")

    service.broadcaster = SimpleNamespace(broadcast_game_state=fail_publication)
    service._broadcast_with_retry = AsyncMock(
        side_effect=AssertionError("ephemeral state must not use retry policy")
    )

    await service._on_state_updated_async(
        "state_updated",
        {"balls": [], "table": {}, "timestamp": 1.0, "cue": None},
        {"source_frame_number": 8},
    )

    assert attempts == 1
    assert service.broadcast_metrics.failed_broadcasts == 1
    assert service.broadcast_metrics.total_retries == 0
    service._broadcast_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_cancels_blocked_state_delivery_and_discards_pending_snapshot():
    service = _service()
    service._event_loop = asyncio.get_running_loop()
    publishing = asyncio.Event()

    async def blocked_state_event(*_args, **_kwargs) -> None:
        publishing.set()
        await asyncio.Event().wait()

    service._on_state_updated_async = blocked_state_event
    service._pending_state_event = (
        "state_updated",
        {"frame_number": 12},
        {"source_frame_number": 12},
    )
    service._state_event_drain_scheduled = True
    service._state_event_drain_task = asyncio.create_task(
        service._drain_state_event_updates()
    )
    await asyncio.wait_for(publishing.wait(), timeout=1.0)

    await service._cancel_state_event_drain()

    assert service._event_loop is None
    assert service._pending_state_event is None
    assert service._state_event_drain_task is None
    assert service._state_event_drain_scheduled is False
