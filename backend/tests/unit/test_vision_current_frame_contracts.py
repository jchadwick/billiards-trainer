"""Current-frame intake contracts for live vision processing."""

from __future__ import annotations

import queue
from typing import Any

import numpy as np
import pytest

from backend.vision import VisionModule, VisionStatistics


def _bare_module() -> VisionModule:
    module = VisionModule.__new__(VisionModule)
    module.stats = VisionStatistics()
    module._frame_queue = queue.Queue(maxsize=1)
    return module


@pytest.mark.unit
def test_latest_frame_replacement_keeps_current_frame_and_counts_drops() -> None:
    module = _bare_module()
    module._frame_queue.put_nowait({"source_frame_number": 40, "source_process_id": 7})

    module._enqueue_frame({"source_frame_number": 41, "source_process_id": 7})
    module._enqueue_frame({"source_frame_number": 42, "source_process_id": 7})

    assert module._frame_queue.get_nowait()["source_frame_number"] == 42
    assert module.stats.frames_dropped == 2


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
    ("incoming_frame", "incoming_timestamp"),
    [
        (100, 21.0),
        (101, 20.0),
    ],
)
def test_stale_source_observation_is_rejected_without_advancing_watermarks(
    incoming_frame: int,
    incoming_timestamp: float,
) -> None:
    module = VisionModule.__new__(VisionModule)
    module.stats = VisionStatistics()
    module._phase_last_source_frame_number = 100
    module._phase_last_source_process_id = 9
    module._phase_last_source_timestamp = 20.0
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
