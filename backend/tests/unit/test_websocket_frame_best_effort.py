import sys
from unittest.mock import Mock

import numpy as np
import pytest

from backend.api.websocket.broadcaster import MessageBroadcaster


@pytest.mark.asyncio
async def test_disabled_websocket_video_records_cadence_without_encoding_or_retaining_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = Mock()
    cv2.imencode.side_effect = AssertionError("disabled video must not JPEG-encode")
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    broadcaster = MessageBroadcaster()
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    await broadcaster.broadcast_frame(frame, width=96, height=64)

    metrics = broadcaster.get_performance_metrics()["frame_metrics"]
    assert metrics["frames_received"] == 1
    assert metrics["frames_sent"] == 0
    assert metrics["bytes_sent"] == 0
    assert metrics["actual_fps"] == 0
    latest_frame = broadcaster.frame_buffer.get_latest_frame()
    assert latest_frame is not None
    assert "image" not in latest_frame
    cv2.imencode.assert_not_called()
