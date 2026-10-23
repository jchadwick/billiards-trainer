"""Lifecycle and frame-publication contracts for the standalone video process."""

import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import backend.video.process as video_process_module
from backend.video.ipc.shared_memory import FrameFormat
from backend.vision.capture import FrameInfo


class MappingConfig:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)


def make_process(**values: object) -> video_process_module.VideoProcess:
    return video_process_module.VideoProcess(MappingConfig(values))  # type: ignore[arg-type]


def frame_info(frame_number: int = 1) -> FrameInfo:
    return FrameInfo(
        frame_number=frame_number,
        timestamp=100.0 + frame_number,
        size=(2, 2),
        channels=3,
    )


def test_video_process_keeps_software_publication_clock_enabled() -> None:
    process = make_process()

    assert process.capture_publication_clock is True


def test_calibration_fallbacks_keep_capture_available(
    tmp_path: Path, monkeypatch
) -> None:
    process = make_process(
        **{
            "video.enable_distortion_correction": True,
            "video.calibration_file_path": str(tmp_path / "missing.yaml"),
        }
    )

    assert process._load_calibration() is True
    assert process._enable_distortion_correction is False

    process = make_process(**{"video.enable_distortion_correction": True})
    process._calibration_file_path = str(tmp_path / "present.yaml")
    Path(process._calibration_file_path).touch()

    class FailedCalibrator:
        camera_params = None

        def load_calibration(self, _path: str) -> bool:
            return False

    monkeypatch.setattr(video_process_module, "CameraCalibrator", FailedCalibrator)
    assert process._load_calibration() is True
    assert process._enable_distortion_correction is False
    assert process.calibrator is None


def test_calibration_load_and_map_creation_use_detected_frame_geometry(
    tmp_path: Path, monkeypatch
) -> None:
    calibration_path = tmp_path / "camera.yaml"
    calibration_path.touch()
    params = SimpleNamespace(
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        resolution=(640, 480),
        calibration_error=0.1,
    )

    class LoadedCalibrator:
        camera_params = params

        def load_calibration(self, path: str) -> bool:
            return path == str(calibration_path)

    process = make_process(**{"video.enable_distortion_correction": True})
    process._calibration_file_path = str(calibration_path)
    monkeypatch.setattr(video_process_module, "CameraCalibrator", LoadedCalibrator)
    assert process._load_calibration() is True
    assert process.calibrator is not None

    map1 = np.zeros((480, 640), dtype=np.float32)
    map2 = np.ones((480, 640), dtype=np.float32)
    monkeypatch.setattr(
        cv2,
        "getOptimalNewCameraMatrix",
        lambda _camera, _dist, size, _alpha, _new_size: (np.eye(3), (0, 0, *size)),
    )
    monkeypatch.setattr(
        cv2,
        "initUndistortRectifyMap",
        lambda *_args: (map1, map2),
    )

    assert process._initialize_undistortion_maps(640, 480) is True
    assert process._undistortion_maps is not None
    assert process._undistortion_maps[0] is map1
    assert process._undistortion_maps[1] is map2


def test_undistortion_uses_precomputed_maps_and_falls_back_to_original(
    monkeypatch,
) -> None:
    process = make_process()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    corrected = np.ones_like(frame)
    process._enable_distortion_correction = True
    process._undistortion_maps = (np.zeros((2, 2)), np.ones((2, 2)))
    monkeypatch.setattr(cv2, "remap", lambda *_args, **_kwargs: corrected)

    assert process._undistort_frame(frame) is corrected
    assert process._frames_undistorted == 1
    assert process._total_undistortion_time >= 0

    def fail_remap(*_args, **_kwargs):
        raise RuntimeError("map invalidated")

    monkeypatch.setattr(cv2, "remap", fail_remap)
    assert process._undistort_frame(frame) is frame


def test_camera_initialization_resolves_configured_video_source(monkeypatch) -> None:
    received: dict[str, object] = {}

    class Camera:
        def __init__(self, config: dict[str, object]) -> None:
            received.update(config)

        def start_capture(self) -> bool:
            return True

    monkeypatch.setattr(video_process_module, "CameraCapture", Camera)
    process = make_process(
        **{
            "vision.camera.device_id": 0,
            "vision.camera.video_file_path": "recording.mp4",
            "vision.camera.resolution": [1280, 720],
            "vision.camera.fps": 25,
        }
    )

    assert process._initialize_camera() is True
    assert received["device_id"] == "recording.mp4"
    assert received["resolution"] == [1280, 720]
    assert received["fps"] == 25


def test_camera_and_shared_memory_initialization_failures_are_reported(
    monkeypatch,
) -> None:
    class CameraThatFails:
        def __init__(self, _config: dict[str, object]) -> None:
            pass

        def start_capture(self) -> bool:
            return False

    monkeypatch.setattr(video_process_module, "CameraCapture", CameraThatFails)
    process = make_process()
    assert process._initialize_camera() is False

    monkeypatch.setattr(
        video_process_module,
        "SharedMemoryFrameWriter",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("allocation failed")),
    )
    assert process._initialize_ipc_writer(2, 2) is False


def test_ipc_writer_publishes_bgr_with_a_monotonic_publication_clock(
    monkeypatch,
) -> None:
    received: dict[str, object] = {}

    class Writer:
        total_size = 12

        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

        def initialize(self) -> None:
            received["initialized"] = True

    monkeypatch.setattr(video_process_module, "SharedMemoryFrameWriter", Writer)
    process = make_process(**{"video.shared_memory_name": "test_frames"})

    assert process._initialize_ipc_writer(2, 2) is True
    assert received["name"] == "test_frames"
    assert received["frame_format"] is FrameFormat.BGR24
    assert received["capture_publication_clock"] is True
    assert received["initialized"] is True


def test_main_loop_publishes_latest_camera_frame(monkeypatch) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    info = frame_info(7)
    process = make_process(**{"processing.main_loop_sleep": 0})

    class Camera:
        def __init__(self) -> None:
            self.reads = 0

        def get_latest_frame(self):
            self.reads += 1
            return (frame, info) if self.reads == 1 else None

        def get_health(self):
            return SimpleNamespace(frames_dropped=2)

    class Writer:
        frames: list[tuple[np.ndarray, int]] = []

        def write_frame(self, image: np.ndarray, frame_number: int) -> None:
            self.frames.append((image, frame_number))

    writer = Writer()
    writer.frames = []
    process.camera = Camera()  # type: ignore[assignment]
    process.ipc_writer = writer  # type: ignore[assignment]

    def stop_after_one_frame(_seconds: float) -> None:
        process.shutdown_event.set()

    monkeypatch.setattr(video_process_module.time, "sleep", stop_after_one_frame)
    monkeypatch.setattr(video_process_module.time, "time", lambda: 101.0)
    process._main_loop()

    assert writer.frames == [(frame, 7)]
    assert process._frames_captured == 1
    assert process._frames_written == 1


def test_statistics_and_cleanup_release_camera_and_writer(caplog) -> None:
    caplog.set_level(logging.INFO)
    process = make_process()
    events: list[str] = []

    class Camera:
        def get_health(self):
            return SimpleNamespace(fps=24.0, frames_dropped=3)

        def stop_capture(self) -> None:
            events.append("camera-stopped")
            raise RuntimeError("camera already closed")

    class Writer:
        def get_stats(self):
            return {"write_counter": 9}

        def cleanup(self) -> None:
            events.append("writer-cleaned")

    process.camera = Camera()  # type: ignore[assignment]
    process.ipc_writer = Writer()  # type: ignore[assignment]
    process._start_time = 10.0
    process._frames_written = 9
    process._log_statistics(13.0)
    assert "fps=3.0" in caplog.text
    assert "camera_dropped=3" in caplog.text

    process._cleanup()

    assert events == ["camera-stopped", "writer-cleaned"]
    assert process.camera is None
    assert process.ipc_writer is None


def test_start_initialization_failure_returns_error_and_always_cleans_up(
    monkeypatch,
) -> None:
    process = make_process()
    cleaned: list[bool] = []
    monkeypatch.setattr(video_process_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(process, "_load_calibration", lambda: False)
    monkeypatch.setattr(process, "_cleanup", lambda: cleaned.append(True))

    assert process.start() == 1
    assert cleaned == [True]


def test_start_runs_pipeline_with_first_frame_dimensions_then_cleans_up(
    monkeypatch,
) -> None:
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    process = make_process()
    calls: list[object] = []

    class Camera:
        def get_latest_frame(self):
            return frame, frame_info()

    process.camera = Camera()  # type: ignore[assignment]
    monkeypatch.setattr(video_process_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(process, "_load_calibration", lambda: True)
    monkeypatch.setattr(process, "_initialize_camera", lambda: True)
    monkeypatch.setattr(
        process,
        "_initialize_undistortion_maps",
        lambda width, height: calls.append((width, height)) or True,
    )
    monkeypatch.setattr(
        process,
        "_initialize_ipc_writer",
        lambda width, height: calls.append(("ipc", width, height)) or True,
    )
    monkeypatch.setattr(process, "_main_loop", lambda: calls.append("loop"))
    monkeypatch.setattr(process, "_cleanup", lambda: calls.append("cleanup"))

    assert process.start() == 0
    assert calls == [(4, 3), ("ipc", 4, 3), "loop", "cleanup"]
