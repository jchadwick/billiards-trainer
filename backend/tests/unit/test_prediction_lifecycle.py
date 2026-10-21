"""Focused deterministic coverage for the vision prediction lifecycle."""

import asyncio
import threading
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.api.routes.game import _live_trajectory_is_eligible
from backend.api.websocket.broadcaster import MessageBroadcaster
from backend.core.coordinates import Vector2D
from backend.core.physics.trajectory import TrajectoryQuality
from backend.integration_service import IntegrationService


def _service() -> IntegrationService:
    service = IntegrationService.__new__(IntegrationService)
    service.trajectory_frame_count = 0
    service.latest_observation_status = None
    service._last_vision_phase = "aiming"
    service._last_calibration_revision = "cal-1"
    service.last_live_prediction = None
    service.frozen_comparison_prediction = None
    service._broadcast_prediction_snapshot = AsyncMock()
    return service


def _detection(phase: str, calibration_revision: str = "cal-1"):
    return SimpleNamespace(
        phase=phase,
        map_revision=7,
        map_trusted=phase in {"aiming", "idle"},
        aim_calculation_eligible=phase == "aiming",
        component_observation_timestamps={"cue": 12.5, "balls": 12.4},
        component_observation_status={"cue": "fresh", "balls": "verified"},
        source_frame_number=90,
        source_timestamp=12.6,
        source_timestamp_kind="shared_memory_publication",
        frame_publication_age_seconds=0.2,
        ball_positions_status="observed",
        ball_detection_status="observed",
        ball_observation_source_timestamp=12.6,
        ball_observation_source_frame_number=90,
        ball_observation_source_timestamp_kind="shared_memory_publication",
        ball_observation_timestamp=12.5,
        maximum_ball_observation_gap_seconds=1.0,
        actual_path_history_max_points=120,
        calibration_revision=calibration_revision,
        moving_ball_count=3 if phase == "many_moving" else 0,
        map_observation_fresh=phase == "aiming",
    )


def _background_service(calculator):
    service = IntegrationService.__new__(IntegrationService)
    service.running = True
    service._prediction_lifecycle_generation = 1
    service._prediction_task = None
    service._pending_prediction = None
    service._trajectory_calculation_lock = threading.Lock()
    service._maximum_prediction_observation_age_seconds = 5.0
    service._latest_observation_received_at = time.monotonic()
    service.trajectory_calculator = SimpleNamespace(
        predict_multiball_cue_shot=calculator
    )
    service.trajectory_quality = TrajectoryQuality.LOW
    service.max_collision_depth = 2
    service.shot_speed_estimate = 2.0
    service.config = SimpleNamespace(get=lambda _key, default=None: default)
    service.latest_observation_status = {
        "phase_aware": True,
        "phase": "aiming",
        "aim_calculation_eligible": True,
        "map_trusted": True,
        "map_observation_fresh": True,
        "map_revision": 7,
        "calibration_revision": "cal-1",
        "frame_publication_age_seconds": 0.0,
    }
    service._emit_multiball_trajectory = AsyncMock()
    return service


def _prediction_request(service, frame_number: int):
    detection = SimpleNamespace(timestamp=12.0, frame_number=frame_number)
    state = SimpleNamespace(id=frame_number, position=[1.0, 2.0])
    return service._create_prediction_request(
        cue_state=SimpleNamespace(angle=20.0),
        ball_state=state,
        table_state=SimpleNamespace(width=100.0),
        other_balls=[SimpleNamespace(id="other", position=[3.0, 4.0])],
        detection=detection,
        status={
            **service.latest_observation_status,
            "source_frame_number": frame_number,
            "source_timestamp": 12.0,
            "source_timestamp_kind": "shared_memory_publication",
            "component_observation_timestamps": {"cue": 12.0},
            "component_observation_status": {"cue": "fresh"},
        },
    )


def test_vision_status_preserves_source_frame_publication_and_map_provenance():
    status = IntegrationService._observation_status(_detection("aiming"))

    assert status["phase"] == "aiming"
    assert status["map_revision"] == 7
    assert status["map_trusted"] is True
    assert status["source_frame_number"] == 90
    assert status["source_timestamp"] == 12.6
    assert status["source_timestamp_kind"] == "shared_memory_publication"
    assert status["frame_publication_age_seconds"] == 0.2
    assert status["component_observation_timestamps"]["cue"] == 12.5
    assert status["aim_calculation_eligible"] is True
    assert status["ball_positions_status"] == "observed"
    assert status["ball_observation_source_frame_number"] == 90
    assert status["maximum_ball_observation_gap_seconds"] == 1.0
    assert status["actual_path_history_max_points"] == 120


@pytest.mark.asyncio
async def test_motion_freezes_exact_last_live_prediction_once_through_recovery():
    service = _service()
    prediction = {
        "lines": [{"start": [1, 2], "end": [3, 4]}, {"start": [3, 4], "end": [5, 6]}],
        "collisions": [{"position": [3, 4], "time": 0.4}],
        "confidence": 0.87,
        "calculation_time_ms": 5.5,
        "source_frame_number": 82,
        "source_timestamp": 11.9,
        "source_timestamp_kind": "shared_memory_publication",
        "map_revision": 6,
        "calibration_revision": "cal-1",
        "prediction_settings": {"quality": "low", "depth": 2},
        "source_observations": {"cue": 11.8, "balls": 11.7},
    }
    service.last_live_prediction = deepcopy(prediction)

    await service._refresh_prediction_lifecycle(_detection("limited_motion"))

    expected = deepcopy(prediction)
    expected.update(
        {
            "prediction_status": "frozen_reference",
            "spatial_valid": True,
            "phase": "limited_motion",
        }
    )
    assert service.frozen_comparison_prediction == expected
    assert "is_reference" not in service.frozen_comparison_prediction
    service._broadcast_prediction_snapshot.assert_awaited_once_with(expected)

    # Later phases and the return to idle retain the same geometry/provenance.
    frozen_snapshot = deepcopy(service.frozen_comparison_prediction)
    await service._refresh_prediction_lifecycle(_detection("many_moving"))
    await service._refresh_prediction_lifecycle(_detection("recovering"))
    await service._refresh_prediction_lifecycle(_detection("idle"))
    assert service.frozen_comparison_prediction == frozen_snapshot
    assert service._broadcast_prediction_snapshot.await_count == 1


@pytest.mark.asyncio
async def test_calibration_change_marks_retained_prediction_spatially_invalid():
    service = _service()
    service.last_live_prediction = {
        "lines": [{"start": [0, 0], "end": [1, 1]}, {"start": [1, 1], "end": [2, 2]}],
        "collisions": [],
        "calibration_revision": "cal-1",
    }

    await service._refresh_prediction_lifecycle(
        _detection("recovering", calibration_revision="cal-2")
    )

    assert service.frozen_comparison_prediction["prediction_status"] == "invalidated"
    assert service.frozen_comparison_prediction["spatial_valid"] is False
    assert service.frozen_comparison_prediction["calibration_revision"] == "cal-1"
    assert service._broadcast_prediction_snapshot.await_count == 1


@pytest.mark.asyncio
async def test_next_valid_live_prediction_replaces_the_frozen_reference():
    service = _service()
    service.broadcaster = SimpleNamespace(broadcast_trajectory=object())
    service._broadcast_with_retry = AsyncMock(return_value=True)
    service.frozen_comparison_prediction = {"lines": [{"old": True}]}
    point_a = SimpleNamespace(position=Vector2D(1.0, 2.0, scale=(1.0, 1.0)))
    point_b = SimpleNamespace(position=Vector2D(3.0, 4.0, scale=(1.0, 1.0)))
    point_c = SimpleNamespace(position=Vector2D(5.0, 6.0, scale=(1.0, 1.0)))
    result = SimpleNamespace(
        trajectories={
            "cue": SimpleNamespace(points=[point_a, point_b, point_c], collisions=[])
        },
        primary_ball_id="cue",
        total_calculation_time=0.006,
    )

    await service._emit_multiball_trajectory(
        result,
        prediction_metadata={
            "source_frame_number": 91,
            "source_timestamp": 13.0,
            "map_revision": 8,
            "calibration_revision": "cal-1",
        },
    )

    assert service.frozen_comparison_prediction is None
    assert service.last_live_prediction["lines"] == [
        {
            "start": {"x": 1.0, "y": 2.0, "scale": [1.0, 1.0]},
            "end": {"x": 3.0, "y": 4.0, "scale": [1.0, 1.0]},
            "type": "primary",
            "ball_id": "cue",
        },
        {
            "start": {"x": 3.0, "y": 4.0, "scale": [1.0, 1.0]},
            "end": {"x": 5.0, "y": 6.0, "scale": [1.0, 1.0]},
            "type": "primary",
            "ball_id": "cue",
        },
    ]
    assert service.last_live_prediction["source_frame_number"] == 91
    assert "is_reference" not in service.last_live_prediction
    assert "is_reference" not in service._broadcast_with_retry.await_args.kwargs
    service._broadcast_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["limited_motion", "many_moving", "recovering"])
async def test_motion_and_recovery_do_not_run_automatic_aim_calculation(phase):
    service = _service()
    service._find_ball_cue_is_pointing_at = Mock(
        side_effect=AssertionError(
            "aim calculation continued during an ineligible phase"
        )
    )
    detection = _detection(phase)
    detection.cue = object()
    detection.balls = [object()]

    await service._check_trajectory_calculation(detection)

    service._find_ball_cue_is_pointing_at.assert_not_called()
    service._broadcast_prediction_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_trajectory_event_cannot_overwrite_during_motion():
    service = _service()
    service.config = SimpleNamespace(get=lambda *_args, **_kwargs: True)
    service.latest_observation_status = {
        "phase_aware": True,
        "phase": "many_moving",
        "aim_calculation_eligible": False,
        "map_trusted": False,
    }
    service.last_live_prediction = {"lines": [{"source": "prior"}]}
    service.broadcaster = SimpleNamespace(broadcast_trajectory=object())
    service._broadcast_with_retry = AsyncMock()

    await service._on_trajectory_calculated_async(
        "trajectory_calculated",
        {"trajectory": {"lines": [{"a": 1}, {"a": 2}], "collisions": []}},
    )

    service._broadcast_with_retry.assert_not_awaited()
    assert service.last_live_prediction == {"lines": [{"source": "prior"}]}


def test_requested_live_calculation_uses_the_same_phase_and_map_gate():
    assert not _live_trajectory_is_eligible(
        {
            "phase": "limited_motion",
            "aim_calculation_eligible": True,
            "map_trusted": True,
        },
        scheduler_enabled=True,
    )
    assert not _live_trajectory_is_eligible(
        {"phase": "aiming", "aim_calculation_eligible": True, "map_trusted": False},
        scheduler_enabled=True,
    )
    assert _live_trajectory_is_eligible(
        {"phase": "aiming", "aim_calculation_eligible": True, "map_trusted": True},
        scheduler_enabled=True,
    )
    assert not _live_trajectory_is_eligible(None, scheduler_enabled=True)
    assert _live_trajectory_is_eligible(None, scheduler_enabled=False)
    assert not _live_trajectory_is_eligible(
        {
            "phase_aware": False,
            "phase": "many_moving",
            "aim_calculation_eligible": None,
            "map_trusted": None,
        },
        scheduler_enabled=False,
    )
    assert not _live_trajectory_is_eligible(
        {"phase_aware": False, "game_state": "moving", "balls_in_motion": 1},
        scheduler_enabled=False,
    )


def test_prediction_request_owns_detached_physics_inputs():
    service = _background_service(Mock())
    cue_state = SimpleNamespace(angle=20.0)
    ball_state = SimpleNamespace(id=91, position=[1.0, 2.0])
    table_state = SimpleNamespace(width=100.0)
    other_balls = [SimpleNamespace(id="other", position=[3.0, 4.0])]
    request = service._create_prediction_request(
        cue_state=cue_state,
        ball_state=ball_state,
        table_state=table_state,
        other_balls=other_balls,
        detection=SimpleNamespace(timestamp=12.0, frame_number=91),
        status=service.latest_observation_status,
    )

    ball_state.position[0] = 99.0
    other_balls[0].position[0] = 99.0
    table_state.width = 99.0
    cue_state.angle = 99.0
    service._prediction_lifecycle_generation += 1

    # The objects passed to request construction are detached copies. Mutating
    # later live state cannot affect what the calculation thread will read.
    assert request.cue_state.angle == 20.0
    assert request.ball_state.position == [1.0, 2.0]
    assert request.other_balls[0].position == [3.0, 4.0]
    assert request.table_state.width == 100.0
    assert request.lifecycle_generation == 1


@pytest.mark.asyncio
async def test_prediction_worker_keeps_loop_responsive_and_coalesces_pending_frames():
    calculation_started = threading.Event()
    release_calculation = threading.Event()
    calculated_frames = []

    def calculate(*, ball_state, **_kwargs):
        calculated_frames.append(ball_state.id)
        if ball_state.id == 91:
            calculation_started.set()
            assert release_calculation.wait(timeout=2)
        return SimpleNamespace(trajectories={})

    service = _background_service(calculate)
    service._queue_prediction(_prediction_request(service, 91))
    assert await asyncio.to_thread(calculation_started.wait, 2)

    loop_tick = asyncio.Event()
    asyncio.get_running_loop().call_soon(loop_tick.set)
    await asyncio.wait_for(loop_tick.wait(), timeout=1)

    service._queue_prediction(_prediction_request(service, 92))
    service._queue_prediction(_prediction_request(service, 93))
    worker = service._prediction_task
    release_calculation.set()
    await worker

    assert calculated_frames == [91, 93]
    assert [
        call.kwargs["prediction_metadata"]["source_frame_number"]
        for call in service._emit_multiball_trajectory.await_args_list
    ] == [91, 93]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed_field", ["phase", "map_revision", "calibration_revision"]
)
async def test_prediction_worker_discards_results_when_lifecycle_context_changes(
    changed_field,
):
    calculation_started = threading.Event()
    release_calculation = threading.Event()

    def calculate(**_kwargs):
        calculation_started.set()
        assert release_calculation.wait(timeout=2)
        return SimpleNamespace(trajectories={})

    service = _background_service(calculate)
    service._queue_prediction(_prediction_request(service, 91))
    assert await asyncio.to_thread(calculation_started.wait, 2)

    replacements = {
        "phase": "limited_motion",
        "map_revision": 8,
        "calibration_revision": "cal-2",
    }
    service.latest_observation_status[changed_field] = replacements[changed_field]
    release_calculation.set()
    await service._prediction_task

    service._emit_multiball_trajectory.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_motion_status_discards_in_flight_prediction():
    calculation_started = threading.Event()
    release_calculation = threading.Event()

    def calculate(**_kwargs):
        calculation_started.set()
        assert release_calculation.wait(timeout=2)
        return SimpleNamespace(trajectories={})

    service = _background_service(calculate)
    service.latest_observation_status = {
        "phase_aware": False,
        "shot_detected": False,
        "balls_in_motion": 0,
        "game_state": "aiming",
    }
    service.frozen_comparison_prediction = {"lines": [{"source": "frozen"}]}
    frozen_snapshot = deepcopy(service.frozen_comparison_prediction)
    service._queue_prediction(_prediction_request(service, 91))
    assert await asyncio.to_thread(calculation_started.wait, 2)

    service.latest_observation_status.update(
        {"shot_detected": True, "balls_in_motion": 1, "game_state": "moving"}
    )
    release_calculation.set()
    await service._prediction_task

    service._emit_multiball_trajectory.assert_not_awaited()
    assert service.frozen_comparison_prediction == frozen_snapshot


@pytest.mark.asyncio
async def test_prediction_worker_discards_observations_past_configured_age():
    service = _background_service(lambda **_kwargs: SimpleNamespace(trajectories={}))
    request = replace(
        _prediction_request(service, 91),
        observation_age_seconds=1.0,
        maximum_observation_age_seconds=0.5,
    )
    service._queue_prediction(request)
    await service._prediction_task

    service._emit_multiball_trajectory.assert_not_awaited()


@pytest.mark.asyncio
async def test_prediction_worker_rechecks_latest_observation_age_before_publish():
    service = _background_service(lambda **_kwargs: SimpleNamespace(trajectories={}))
    service._maximum_prediction_observation_age_seconds = 0.5
    service._latest_observation_received_at = time.monotonic() - 1.0
    service._queue_prediction(_prediction_request(service, 91))
    await service._prediction_task

    service._emit_multiball_trajectory.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_waits_for_active_physics_thread_and_suppresses_its_result():
    calculation_started = threading.Event()
    release_calculation = threading.Event()
    calculated_frames = []

    def calculate(*, ball_state, **_kwargs):
        calculated_frames.append(ball_state.id)
        calculation_started.set()
        assert release_calculation.wait(timeout=2)
        return SimpleNamespace(trajectories={})

    service = _background_service(calculate)
    service.integration_task = None
    service.vision = SimpleNamespace(stop_capture=Mock())
    service._queue_prediction(_prediction_request(service, 91))
    assert await asyncio.to_thread(calculation_started.wait, 2)
    service._queue_prediction(_prediction_request(service, 92))

    stop_task = asyncio.create_task(service.stop())
    await asyncio.sleep(0)
    assert service.running is False
    assert not service._prediction_task.cancelled()
    assert not stop_task.done()

    release_calculation.set()
    await stop_task
    assert calculated_frames == [91]
    service._emit_multiball_trajectory.assert_not_awaited()


@pytest.mark.asyncio
async def test_state_broadcast_carries_observation_status():
    broadcaster = MessageBroadcaster.__new__(MessageBroadcaster)
    broadcaster.broadcast_stats = SimpleNamespace(validation_failures=0)
    broadcaster._last_game_state = None
    broadcaster._has_state_changed = Mock(return_value=False)
    broadcaster._get_next_sequence = Mock(return_value=3)
    broadcaster._broadcast_to_stream = AsyncMock()
    status = {
        "phase_aware": True,
        "phase": "aiming",
        "map_revision": 8,
        "map_trusted": True,
        "aim_calculation_eligible": True,
        "source_frame_number": 90,
        "source_timestamp": 12.6,
        "source_timestamp_kind": "shared_memory_publication",
    }

    await broadcaster.broadcast_game_state(
        balls=[{"position": {"x": 1.0, "y": 2.0, "scale": [1.0, 1.0]}}],
        observation_status=status,
    )

    payload = broadcaster._broadcast_to_stream.await_args.args[1]
    assert payload["observation_status"] == status
