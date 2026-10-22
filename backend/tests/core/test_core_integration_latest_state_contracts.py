"""Contracts for current ball-state handoff through Core integrations.

The tests keep state processing synchronous and local so they exercise the
authoritative Core boundary without camera, network, or projector hardware.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from backend.core.coordinates import Vector2D
from backend.core.events.manager import EventType
from backend.core.game_state import GameStateManager
from backend.core.integration import (
    CoreModuleIntegrator,
    ProjectorInterfaceImpl,
    VisionInterfaceImpl,
    _table_coordinate_context,
)
from backend.core.models import TableState


def _table(*, calibrated: bool = False) -> TableState:
    corners = [
        Vector2D.from_4k(x, y)
        for x, y in ((320, 280), (3520, 280), (3520, 1880), (320, 1880))
    ]
    return TableState(
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
        playing_area_corners=corners if calibrated else None,
        real_width_meters=2.0 if calibrated else None,
        real_height_meters=1.0 if calibrated else None,
    )


class _SynchronousEventManager:
    """Capture Core publications and synchronously route Vision observations."""

    def __init__(self) -> None:
        self.subscribers: dict[str, Any] = {}
        self.state_updates: list[dict[str, Any]] = []
        self.trajectories: list[dict[str, Any]] = []

    def subscribe_to_events(self, event_type: str, callback: Any) -> None:
        self.subscribers[event_type] = callback

    def receive_vision_data(self, detection_data: dict[str, Any]) -> None:
        callback = self.subscribers[EventType.VISION_DATA_RECEIVED.value]
        callback(
            EventType.VISION_DATA_RECEIVED.value,
            {"detection_data": detection_data},
        )

    def send_state_update_to_api(self, state: dict[str, Any]) -> None:
        self.state_updates.append(state)

    def provide_trajectory_to_projector(self, trajectory: dict[str, Any]) -> None:
        self.trajectories.append(trajectory)


def _ball_observation(x: float, *, status: str = "observed") -> dict[str, Any]:
    return {
        "timestamp": 1_800_000_000.0,
        "frame_number": int(x),
        "source_frame_id": f"camera:{int(x)}",
        "balls": [
            {
                "id": "cue",
                "position": {"x": x, "y": 1080.0, "scale": [1.0, 1.0]},
                "velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "scale": [1.0, 1.0],
                },
                "radius": 36.0,
                "radius_scale": [1.0, 1.0],
                "confidence": 0.97,
                "number": 0,
                "is_cue_ball": True,
            }
        ],
        "observation_status": {"ball_detection_status": status},
    }


def test_core_broadcast_tracks_latest_authoritative_ball_and_observed_empty(
    tmp_path: Any,
) -> None:
    """A failed detector frame retains state; observed empty clears it."""
    events = _SynchronousEventManager()
    manager = GameStateManager(table=_table(), persistence_path=tmp_path / "state")
    integrator = CoreModuleIntegrator(cast("Any", events), cast("Any", manager))
    vision = VisionInterfaceImpl(cast("Any", events))

    try:
        vision.receive_detection_data(_ball_observation(1400.0))
        first_state = manager.get_current_state()
        assert first_state is not None
        assert len(first_state.balls) == 1
        assert first_state.balls[0].position == Vector2D.from_4k(1400.0, 1080.0)
        assert events.state_updates[-1]["balls"][0]["position_unit"] == (
            "canonical_pixels"
        )

        vision.receive_detection_data(
            {
                **_ball_observation(0.0, status="failed"),
                "balls": [],
            }
        )
        retained_state = manager.get_current_state()
        assert retained_state is not None
        assert retained_state.frame_number == first_state.frame_number + 1
        assert len(retained_state.balls) == 1
        assert retained_state.balls[0].position == Vector2D.from_4k(1400.0, 1080.0)
        assert events.state_updates[-1]["frame_number"] == retained_state.frame_number
        assert events.state_updates[-1]["balls"][0]["position"]["x"] == 1400.0

        vision.receive_detection_data(
            {
                **_ball_observation(0.0),
                "balls": [],
            }
        )
        observed_empty_state = manager.get_current_state()
        assert observed_empty_state is not None
        assert observed_empty_state.frame_number == retained_state.frame_number + 1
        assert observed_empty_state.balls == []
        assert events.state_updates[-1]["balls"] == []
        assert integrator.integration_stats["messages_received"] == 3
    finally:
        manager._event_manager.stop_enhanced_processing()


def test_vision_adapter_lock_serializes_concurrent_state_handoffs() -> None:
    """An in-flight handoff holds the adapter lock until it completes."""
    first_entered_sink = threading.Event()
    second_entered_sink = threading.Event()
    release_sink = threading.Event()
    second_call_started = threading.Event()
    delivered: list[int] = []
    first_sink_release_results: list[bool] = []

    class BlockingSink:
        def receive_vision_data(self, data: dict[str, Any]) -> None:
            delivered.append(data["frame_number"])
            if data["frame_number"] == 1:
                first_entered_sink.set()
                first_sink_release_results.append(release_sink.wait(timeout=2.0))
            else:
                second_entered_sink.set()

    vision = VisionInterfaceImpl(cast("Any", BlockingSink()))
    first = threading.Thread(
        target=vision.receive_detection_data,
        args=(_ball_observation(1.0),),
        daemon=True,
    )

    def submit_second_observation() -> None:
        second_call_started.set()
        vision.receive_detection_data(_ball_observation(2.0))

    second = threading.Thread(target=submit_second_observation, daemon=True)
    first.start()
    assert first_entered_sink.wait(timeout=1.0)
    second.start()
    assert second_call_started.wait(timeout=1.0)
    try:
        assert not second_entered_sink.wait(timeout=0.05)
    finally:
        release_sink.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert first_sink_release_results == [True]
    assert delivered == [1, 2]


def test_calibrated_table_and_projector_conversion_preserve_table_location() -> None:
    """A canonical table center maps through meters to projector pixel center."""
    table = _table(calibrated=True)
    context = _table_coordinate_context(table)
    assert context is not None

    projector = ProjectorInterfaceImpl(cast("Any", object()))
    projector.calibration_status["is_calibrated"] = True
    projector.projection_settings["transformation_matrix"] = [
        [2.0 / 3840.0, 0.0, 0.0],
        [0.0, 1.0 / 2160.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    screen_point = projector._convert_to_screen_coordinates(
        [{"x": 1920.0, "y": 1080.0, "scale": [1.0, 1.0]}],
        coordinate_space="4k_canonical",
        coordinate_unit="canonical_pixels",
        table_geometry=context,
    )

    assert len(screen_point) == 1
    assert screen_point[0]["x"] == pytest.approx(960.0)
    assert screen_point[0]["y"] == pytest.approx(540.0)
    assert screen_point[0]["coordinate_space"] == "projector_pixels"
