"""Tests for software camera-read and capture-queue audit metadata."""

import queue

import numpy as np

from backend.vision.capture import CameraCapture, FrameInfo


def test_latest_frame_reports_intentionally_skipped_capture_frames() -> None:
    capture = CameraCapture.__new__(CameraCapture)
    capture._frame_queue = queue.Queue()
    for frame_number in (1, 2, 3):
        capture._frame_queue.put(
            (
                np.zeros((2, 2, 3), dtype=np.uint8),
                FrameInfo(frame_number, float(frame_number), (2, 2), 3),
            )
        )

    result = capture.get_latest_frame()

    assert result is not None
    _frame, metadata = result
    assert metadata.frame_number == 3
    assert metadata.capture_queue_frames_skipped == 2
    assert metadata.capture_queue_depth_after_drain == 0
    assert metadata.camera_read_seconds is None
