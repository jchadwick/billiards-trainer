"""Checks that overload favors the newest captured frame."""

import queue
from types import SimpleNamespace

import pytest

from backend.vision import VisionModule, VisionStatistics


@pytest.mark.unit
def test_legacy_fifo_setting_cannot_disable_single_slot_latest_frame_queue() -> None:
    module = VisionModule.__new__(VisionModule)
    # A legacy false setting must not restore FIFO backlog processing.
    module.config = SimpleNamespace(latest_frame_queue_enabled=False)
    module._frame_queue = queue.Queue(maxsize=1)
    module.stats = VisionStatistics()
    module._frame_queue.put_nowait({"source_frame_number": 10})
    module._enqueue_frame({"source_frame_number": 11})

    module._enqueue_frame({"source_frame_number": 12})

    assert module._frame_queue.get_nowait() == {"source_frame_number": 12}
    assert module.stats.frames_dropped == 2
