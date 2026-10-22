"""Performance and accuracy contracts for the YOLO inference boundary.

All model behavior is represented by tiny deterministic fakes. These tests
exercise the production input, decoding, caching, and failure paths without
loading weights or requiring a platform-specific runtime.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from backend.vision.detection.yolo_detector import (
    BallDetectionStatus,
    DetectionFailureStage,
    ModelFormat,
    YOLODetector,
)


class TensorLike:
    """Minimal tensor surface used by Ultralytics' result objects."""

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, index: Any) -> TensorLike:
        return TensorLike(self.values[index])

    def cpu(self) -> TensorLike:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values


class RecordingModel:
    """Fake standard model that returns a fixed Ultralytics-shaped result."""

    def __init__(self, boxes: list[Any] | None = None) -> None:
        self.boxes = boxes or []
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.failure: Exception | None = None

    def __call__(self, frame: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((frame, kwargs))
        if self.failure is not None:
            raise self.failure
        return [SimpleNamespace(boxes=self.boxes)]


class ObservedLock:
    """Expose when a second caller is waiting on the detector cache lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self.acquire_count = 0
        self.second_acquire_attempted = threading.Event()

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        with self._count_lock:
            self.acquire_count += 1
            if self.acquire_count == 2:
                self.second_acquire_attempted.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> ObservedLock:
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


def _box(
    xyxy: tuple[float, float, float, float], class_id: int, confidence: float
) -> SimpleNamespace:
    return SimpleNamespace(
        xyxy=TensorLike([xyxy]),
        cls=TensorLike([class_id]),
        conf=TensorLike([confidence]),
    )


def _loaded_detector(model: RecordingModel) -> YOLODetector:
    detector = YOLODetector(device="cpu", confidence=0.2, min_ball_size=20)
    detector.model = model
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    return detector


@pytest.mark.unit
def test_current_frame_ball_and_cue_requests_share_one_concurrent_inference() -> None:
    """Concurrent consumers of one frame key must not launch duplicate work."""
    model_entered = threading.Event()
    release_model = threading.Event()

    class BlockingModel(RecordingModel):
        def __call__(self, frame: Any, **kwargs: Any) -> list[Any]:
            self.calls.append((frame, kwargs))
            model_entered.set()
            if not release_model.wait(timeout=3):
                raise TimeoutError("test did not release the fake model")
            return [SimpleNamespace(boxes=[])]

    model = BlockingModel()
    detector = _loaded_detector(model)
    cache_lock = ObservedLock()
    detector._cache_lock = cast("Any", cache_lock)
    frame = np.zeros((96, 160, 3), dtype=np.uint8)
    frame_key = (3, 42, "masked-full-frame")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(detector._run_inference, frame, frame_key=frame_key)
        assert model_entered.wait(timeout=2)
        second = executor.submit(detector._run_inference, frame, frame_key=frame_key)
        assert cache_lock.second_acquire_attempted.wait(timeout=2)
        try:
            assert len(model.calls) == 1
        finally:
            release_model.set()

        assert first.result(timeout=3) == []
        assert second.result(timeout=3) == []

    assert len(model.calls) == 1
    assert model.calls[0][0] is frame
    assert model.calls[0][1] == {
        "conf": 0.2,
        "iou": detector.nms_threshold,
        "verbose": False,
        "device": "cpu",
    }


@pytest.mark.unit
def test_multiclass_observation_preserves_ball_identity_and_rejects_non_ball_shapes() -> (
    None
):
    """Decoded ball IDs and table coordinates must survive ball filtering."""
    boxes = [
        _box((80.0, 100.0, 104.0, 124.0), 0, 0.98),
        _box((240.0, 160.0, 264.0, 184.0), 8, 0.94),
        _box((420.0, 260.0, 444.0, 284.0), 15, 0.91),
        _box((520.0, 100.0, 600.0, 120.0), 7, 0.96),  # long false positive
        _box((300.0, 400.0, 420.0, 404.0), 16, 0.99),  # cue stick
        _box((700.0, 500.0, 740.0, 540.0), 17, 0.99),  # table region
    ]
    model = RecordingModel(boxes)
    detector = _loaded_detector(model)
    detector.class_names = {
        0: "cue_ball",
        7: "ball_7",
        8: "ball_8",
        15: "ball_15",
        16: "cue_stick",
        17: "table",
    }
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    outcome = detector.detect_balls_outcome(frame, frame_key=(1, 101, "full"))

    assert outcome.status is BallDetectionStatus.DETECTIONS
    assert [detection.class_id for detection in outcome.detections] == [0, 8, 15]
    assert [detection.class_name for detection in outcome.detections] == [
        "cue_ball",
        "ball_8",
        "ball_15",
    ]
    assert [detection.center for detection in outcome.detections] == [
        (92.0, 112.0),
        (252.0, 172.0),
        (432.0, 272.0),
    ]
    assert [detection.confidence for detection in outcome.detections] == pytest.approx(
        [0.98, 0.94, 0.91]
    )
    assert model.calls[0][0] is frame


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_failure", "malformed_boxes", "expected_stage"),
    [
        (
            RuntimeError("private runtime detail"),
            False,
            DetectionFailureStage.INFERENCE,
        ),
        (None, True, DetectionFailureStage.DECODING),
    ],
)
def test_backend_failures_release_model_lock_without_counting_observations(
    model_failure: Exception | None,
    malformed_boxes: bool,
    expected_stage: DetectionFailureStage,
) -> None:
    model = RecordingModel(
        [SimpleNamespace(xyxy=object(), cls=TensorLike([0]), conf=TensorLike([0.9]))]
        if malformed_boxes
        else []
    )
    model.failure = model_failure
    detector = _loaded_detector(model)

    outcome = detector.detect_balls_outcome(np.zeros((48, 64, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is expected_stage
    assert detector.stats["total_inferences"] == 0
    assert detector._model_lock.acquire(blocking=False)
    detector._model_lock.release()


@pytest.mark.unit
@pytest.mark.parametrize(
    "outputs",
    [[], [np.zeros((1, 8, 6), dtype=np.float32)]],
    ids=["no-output-tensors", "unsupported-rank"],
)
def test_tpu_decoder_treats_missing_or_unsupported_output_as_empty_observation(
    outputs: list[np.ndarray[Any, Any]],
) -> None:
    detector = YOLODetector(device="cpu")

    detections = detector._parse_tpu_output(
        outputs, original_shape=(480, 640, 3), model_input_shape=(320, 320)
    )

    assert detections == []
