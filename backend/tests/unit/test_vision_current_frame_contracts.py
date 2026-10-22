"""Current-frame intake contracts for live vision processing."""

from __future__ import annotations

import queue
from typing import Any

import numpy as np
import pytest

from backend.vision import VisionModule, VisionStatistics


class _RecordingAudit:
    def __init__(self) -> None:
        self.events: list[tuple[dict[str, Any], bool]] = []

    def record(self, event: dict[str, Any], *, force_keep: bool = False) -> None:
        self.events.append((event, force_keep))


def _bare_module() -> VisionModule:
    module = VisionModule.__new__(VisionModule)
    module.stats = VisionStatistics()
    module._frame_queue = queue.Queue(maxsize=1)
    module.audit_domains = {"state": False}
    module.audit_recorder = None
    return module


@pytest.mark.unit
def test_latest_frame_replacement_reports_each_evicted_source_identity() -> None:
    module = _bare_module()
    recorder = _RecordingAudit()
    module.audit_domains = {"state": True}
    module.audit_recorder = recorder
    module._frame_queue.put_nowait({"source_frame_number": 40, "source_process_id": 7})

    module._enqueue_frame({"source_frame_number": 41, "source_process_id": 7})
    module._enqueue_frame({"source_frame_number": 42, "source_process_id": 7})

    assert module._frame_queue.get_nowait()["source_frame_number"] == 42
    assert module.stats.frames_dropped == 2
    assert [event[0]["source_frame_number"] for event in recorder.events] == [40, 41]
    assert [
        event[0]["replacement_source_frame_number"] for event in recorder.events
    ] == [
        41,
        42,
    ]
    assert all(force_keep for _event, force_keep in recorder.events)


@pytest.mark.unit
def test_latest_frame_insert_retries_if_consumer_wins_full_queue_race() -> None:
    class ConsumerWonRaceQueue(queue.Queue[dict[str, Any]]):
        """Make the consumer take the old item after the producer sees Full."""

        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self.simulate_consumer_race = False

        def put_nowait(self, item: dict[str, Any]) -> None:
            if self.simulate_consumer_race:
                self.simulate_consumer_race = False
                # The first insertion observes a full queue, then a consumer
                # removes that item before the producer's eviction attempt.
                self.get_nowait()
                raise queue.Full
            super().put_nowait(item)

    module = _bare_module()
    frame_queue = ConsumerWonRaceQueue()
    frame_queue.put_nowait({"source_frame_number": 50})
    frame_queue.simulate_consumer_race = True
    module._frame_queue = frame_queue

    module._enqueue_frame({"source_frame_number": 51})

    assert frame_queue.get_nowait()["source_frame_number"] == 51
    assert module.stats.frames_dropped == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("incoming_frame", "incoming_timestamp", "expected_reason"),
    [
        (100, 21.0, "out_of_order_source_frame"),
        (101, 20.0, "non_monotonic_source_timestamp"),
    ],
)
def test_stale_source_observation_is_rejected_without_advancing_watermarks(
    incoming_frame: int,
    incoming_timestamp: float,
    expected_reason: str,
) -> None:
    module = VisionModule.__new__(VisionModule)
    module.stats = VisionStatistics()
    module._phase_last_source_frame_number = 100
    module._phase_last_source_process_id = 9
    module._phase_last_source_timestamp = 20.0
    recorder = _RecordingAudit()
    module.audit_domains = {"state": True}
    module.audit_recorder = recorder

    result = module._process_single_frame(
        np.zeros((2, 2, 3), dtype=np.uint8),
        frame_number=500,
        timestamp=30.0,
        source_frame_number=incoming_frame,
        source_timestamp=incoming_timestamp,
        source_process_id=9,
    )

    assert result is None
    assert module.stats.frames_dropped == 1
    assert module._phase_last_source_frame_number == 100
    assert module._phase_last_source_timestamp == 20.0
    assert len(recorder.events) == 1
    event, force_keep = recorder.events[0]
    assert event["event_type"] == "vision_source_frame_rejected"
    assert event["source_frame_number"] == incoming_frame
    assert event["reason"] == expected_reason
    assert force_keep is True
