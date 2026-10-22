"""Coverage for sampled, opt-in viewer stream instrumentation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

import backend.api.routes.stream as stream_route

if TYPE_CHECKING:
    from starlette.requests import Request

    from backend.api.dependencies import ApplicationState


class _AuditRecorder:
    def __init__(self, selected: bool = True) -> None:
        self.selected = selected
        self.select_detail_sample = Mock(return_value=selected)
        self.record = Mock()


class _Config:
    def get(self, path: str, default: Any = None) -> Any:
        return {
            "video.shared_memory_name": "test-video",
            "video.shared_memory_attach_timeout_sec": 0.1,
            "performance_audit.detailed_event_sample_rate": 0.25,
            "api.stream": {
                "performance": {
                    "frame_log_interval": 30,
                    "sleep_interval_ms": 0,
                }
            },
        }.get(path, default)


class _Request:
    def __init__(self) -> None:
        self.disconnect_checks = 0

    async def is_disconnected(self) -> bool:
        self.disconnect_checks += 1
        return self.disconnect_checks > 1


class _FrameReader:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.detached = False
        self.read_count = 0

    def attach(self, *, timeout: float) -> None:
        assert timeout == 0.1

    def read_frame(self) -> tuple[np.ndarray | None, SimpleNamespace | None]:
        self.read_count += 1
        if self.read_count > 1:
            return None, None
        return self.frame, SimpleNamespace(frame_number=7, writer_pid=22)

    def detach(self) -> None:
        self.detached = True


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [True, False])
async def test_shared_memory_viewer_samples_before_encode_and_send_timing(
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
) -> None:
    recorder = _AuditRecorder(selected)
    vision_module = SimpleNamespace(
        audit_domains={"viewers": True},
        audit_recorder=recorder,
    )
    app_state = SimpleNamespace(
        vision_module=vision_module,
        config_module=_Config(),
    )
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    reader = _FrameReader(frame)
    monkeypatch.setattr(
        stream_route,
        "SharedMemoryFrameReader",
        lambda *, name: reader,
    )
    monkeypatch.setattr(
        stream_route.cv2,
        "imencode",
        Mock(return_value=(True, np.asarray([1, 2, 3], dtype=np.uint8))),
    )
    timer = Mock(side_effect=range(1, 20))
    monkeypatch.setattr(stream_route.time, "perf_counter_ns", timer)

    chunks = [
        chunk
        async for chunk in stream_route.generate_mjpeg_stream_from_shm(
            request=cast("Request", _Request()),
            app_state=cast("ApplicationState", app_state),
            quality=80,
            fps=30,
        )
    ]

    assert len(chunks) == 1
    assert reader.detached
    recorder.select_detail_sample.assert_called_once_with(
        "viewers",
        7,
        0.25,
        category="viewer_frame",
    )
    if selected:
        assert [
            call.args[0]["event_type"] for call in recorder.record.call_args_list
        ] == [
            "viewer_frame_encoded",
            "viewer_send_backpressure",
        ]
        assert all(call.kwargs["force_keep"] for call in recorder.record.call_args_list)
        assert timer.call_count == 6
    else:
        recorder.record.assert_not_called()
        # The source-frame identity is only available after the shared-memory
        # read, so those two audit clocks remain. Send duration is always timed
        # because the independent backpressure limit must work without auditing.
        assert timer.call_count == 4


def test_viewer_audit_domain_is_independently_opt_in() -> None:
    recorder = _AuditRecorder()
    vision_module = SimpleNamespace(
        audit_domains={"viewers": False},
        audit_recorder=recorder,
    )

    selected_recorder = stream_route._viewer_audit_recorder(vision_module)
    assert selected_recorder is None
    assert (
        stream_route._viewer_audit_detail_selected(selected_recorder, 7, 0.25) is False
    )
    recorder.select_detail_sample.assert_not_called()
