"""Tests that the detector applies and reports real inference placement."""

import platform
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backend.common.models import Ball
from backend.vision.detection.yolo_detector import (
    BallDetectionStatus,
    Detection,
    DetectionFailureStage,
    ModelFormat,
    YOLODetector,
)
from backend.vision.models import BallType


class RecordingModel:
    """Small Ultralytics stand-in that records placement without hardware."""

    def __init__(self) -> None:
        self.predictor = SimpleNamespace(device=None)
        self.calls: list[dict[str, Any]] = []
        self.result: list[Any] = []
        self.failure: Exception | None = None

    def __call__(self, frame: np.ndarray, **kwargs: Any) -> list[Any]:
        self.calls.append({"frame": frame, **kwargs})
        self.predictor.device = kwargs.get("device")
        if self.failure is not None:
            raise self.failure
        return self.result


class TensorLike:
    """Minimal tensor wrapper for exercising the Ultralytics result decoder."""

    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, index: Any) -> "TensorLike":
        return TensorLike(self.values[index])

    def cpu(self) -> "TensorLike":
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values


def _bare_detector(device: str) -> tuple[YOLODetector, RecordingModel]:
    detector = YOLODetector.__new__(YOLODetector)
    model = RecordingModel()
    detector.model_path = None
    detector.device = device
    detector.model = model
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    detector.class_names = {0: "ball", 1: "cue"}
    detector.confidence = 0.4
    detector.nms_threshold = 0.45
    detector.min_ball_size = 20
    detector._model_lock = threading.RLock()
    detector._cache_lock = threading.Lock()
    detector._detection_cache = {}
    detector._inference_model_generation = 0
    detector.stats = {
        "total_inferences": 0,
        "total_detections": 0,
        "avg_inference_time": 0.0,
        "fallback_mode": False,
        "using_tpu": False,
    }
    return detector, model


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
def test_pytorch_inference_uses_configured_device_in_both_paths(
    device: str,
) -> None:
    detector, model = _bare_detector(device)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    assert detector._run_standard_inference(frame) == []

    assert model.calls == [
        {
            "frame": frame,
            "conf": 0.4,
            "iou": 0.45,
            "verbose": False,
            "device": device,
        }
    ]
    info = detector.get_model_info()
    assert info["requested_device"] == device
    assert info["effective_device"] == device
    # The legacy field remains the configured request for API compatibility.
    assert info["device"] == device


@pytest.mark.unit
def test_effective_device_is_unavailable_until_runtime_reports_it() -> None:
    detector, _model = _bare_detector("mps")
    detector.model = None

    info = detector.get_model_info()

    assert info["requested_device"] == "mps"
    assert info["effective_device"] is None
    assert info["device"] == "mps"


@pytest.mark.unit
def test_auto_device_selects_native_coreml_for_mlpackage_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = YOLODetector.__new__(YOLODetector)
    detector.model_path = "models/billiards-yolov8n.mlpackage"
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    assert detector._auto_detect_device() == "coreml"


@pytest.mark.unit
def test_coreml_model_info_reports_backend_without_claiming_compute_unit() -> None:
    detector, _model = _bare_detector("coreml")
    detector.model_format = ModelFormat.COREML

    info = detector.get_model_info()

    assert info["effective_backend"] == "coreml"
    assert info["effective_device"] is None


@pytest.mark.unit
def test_coreml_loader_resolves_lazily_exposed_ultralytics_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, _model = _bare_detector("coreml")
    loaded_model = SimpleNamespace(names={0: "ball"})
    loaded_paths: list[str] = []
    lazy_lookups: list[str] = []

    def load_yolo(path: str) -> Any:
        loaded_paths.append(path)
        return loaded_model

    class LazyUltralyticsModule(ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name == "YOLO":
                lazy_lookups.append(name)
                return load_yolo
            raise AttributeError(name)

    ultralytics = LazyUltralyticsModule("ultralytics")
    assert "YOLO" not in ultralytics.__dict__
    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module",
        lambda _name: ultralytics,
    )

    detector._load_coreml_model("models/billiards-yolov8n.mlpackage")

    assert lazy_lookups == ["YOLO"]
    assert loaded_paths == ["models/billiards-yolov8n.mlpackage"]
    assert detector.model is loaded_model
    assert detector.model_loaded is True
    assert detector.class_names == {0: "ball"}
    assert detector.name_to_class == {"ball": 0}


@pytest.mark.unit
def test_coreml_loader_preserves_missing_runtime_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, _model = _bare_detector("coreml")

    def load_yolo(_path: str) -> Any:
        raise ImportError("No module named 'coremltools'")

    class LazyUltralyticsModule(ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name == "YOLO":
                return load_yolo
            raise AttributeError(name)

    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module",
        lambda _name: LazyUltralyticsModule("ultralytics"),
    )

    with pytest.raises(ImportError, match="coremltools") as error:
        detector._load_coreml_model("models/billiards-yolov8n.mlpackage")

    assert "ultralytics package not installed" not in str(error.value)


@pytest.mark.unit
def test_onnx_backend_does_not_receive_pytorch_device_argument() -> None:
    detector, model = _bare_detector("coreml")
    detector.model_format = ModelFormat.ONNX
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    assert detector._run_standard_inference(frame) == []

    assert model.calls == [{"frame": frame, "conf": 0.4, "iou": 0.45, "verbose": False}]


@pytest.mark.unit
def test_standard_inference_decodes_successful_ultralytics_result() -> None:
    detector, model = _bare_detector("cpu")
    detector.class_names = {2: "8-ball"}
    model.result = [
        SimpleNamespace(
            boxes=[
                SimpleNamespace(
                    xyxy=TensorLike([[12.0, 18.0, 32.0, 42.0]]),
                    cls=TensorLike([2.0]),
                    conf=TensorLike([0.93]),
                )
            ]
        )
    ]
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    detections = detector._run_standard_inference(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_id == 2
    assert detection.class_name == "8-ball"
    assert detection.confidence == pytest.approx(0.93)
    assert detection.bbox == pytest.approx((12.0, 18.0, 32.0, 42.0))
    assert detection.center == pytest.approx((22.0, 30.0))
    assert detection.width == pytest.approx(20.0)
    assert detection.height == pytest.approx(24.0)
    assert detector.stats["total_inferences"] == 1
    assert detector.stats["total_detections"] == 1


def _ultralytics_result(*, xyxy: Any = None, class_id: float = 0.0) -> list[Any]:
    """Build a small standard result without importing a model runtime."""
    if xyxy is None:
        xyxy = [[12.0, 18.0, 32.0, 42.0]]
    return [
        SimpleNamespace(
            boxes=[
                SimpleNamespace(
                    xyxy=TensorLike(xyxy),
                    cls=TensorLike([class_id]),
                    conf=TensorLike([0.93]),
                )
            ]
        )
    ]


@pytest.mark.unit
def test_ball_observation_outcome_distinguishes_successful_empty_from_unavailable() -> (
    None
):
    detector, model = _bare_detector("cpu")
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    empty = detector.detect_balls_outcome(frame)

    assert empty.status is BallDetectionStatus.NO_DETECTIONS
    assert empty.detections == ()
    assert model.calls

    detector.model_loaded = False
    model.calls.clear()

    unavailable = detector.detect_balls_outcome(frame)

    assert unavailable.status is BallDetectionStatus.MODEL_UNAVAILABLE
    assert unavailable.detections == ()
    assert not model.calls

    detector.model_loaded = True
    detector.model = None

    missing_backend = detector.detect_balls_outcome(frame)

    assert missing_backend.status is BallDetectionStatus.MODEL_UNAVAILABLE
    assert missing_backend.detections == ()


@pytest.mark.unit
def test_ball_observation_outcome_reports_decoded_detections() -> None:
    detector, model = _bare_detector("cpu")
    detector.class_names = {0: "8-ball", 1: "cue"}
    model.result = _ultralytics_result()

    outcome = detector.detect_balls_outcome(np.zeros((64, 96, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.DETECTIONS
    assert len(outcome.detections) == 1
    assert outcome.detections[0].class_id == 0


@pytest.mark.unit
def test_standard_backend_failure_is_not_reported_as_successful_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector, model = _bare_detector("cpu")
    model.failure = RuntimeError("private backend detail")

    outcome = detector.detect_balls_outcome(np.zeros((64, 96, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.INFERENCE
    assert outcome.detections == ()
    assert "private backend detail" not in caplog.text
    assert detector.detect_balls(np.zeros((64, 96, 3), dtype=np.uint8)) == []


@pytest.mark.unit
def test_standard_result_decoding_failure_has_safe_failure_stage() -> None:
    detector, model = _bare_detector("cpu")
    model.result = [
        SimpleNamespace(
            boxes=[
                SimpleNamespace(
                    xyxy=object(),
                    cls=TensorLike([0.0]),
                    conf=TensorLike([0.93]),
                )
            ]
        )
    ]

    outcome = detector.detect_balls_outcome(np.zeros((64, 96, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.DECODING


class FailingTPUInterpreter:
    """Minimal interpreter that fails during invocation without TPU hardware."""

    def set_tensor(self, _index: int, _input: np.ndarray) -> None:
        pass

    def invoke(self) -> None:
        raise RuntimeError("private TPU runtime detail")


@pytest.mark.unit
def test_tpu_backend_failure_is_not_reported_as_successful_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector, _model = _bare_detector("tpu")
    detector.model_format = ModelFormat.EDGETPU
    detector.tpu_interpreter = FailingTPUInterpreter()
    detector.tpu_input_details = [
        {
            "shape": np.asarray([1, 8, 8, 3]),
            "dtype": np.uint8,
            "index": 0,
        }
    ]
    detector.tpu_output_details = []

    outcome = detector.detect_balls_outcome(np.zeros((32, 48, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.INFERENCE
    assert outcome.detections == ()
    assert "private TPU runtime detail" not in caplog.text


@pytest.mark.unit
def test_list_apis_preserve_empty_fallback_and_classification_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, model = _bare_detector("cpu")
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    assert detector.detect_balls(frame) == []
    assert detector.detect_balls_with_classification(frame) == []

    model.result = _ultralytics_result()
    detections = detector.detect_balls(frame)
    assert len(detections) == 1
    classified_ball = Ball(
        position=(22.0, 30.0),
        radius=12.0,
        ball_type=BallType.EIGHT,
        confidence=0.93,
    )
    monkeypatch.setattr(
        detector,
        "_adapt_and_classify_balls",
        lambda *_args: [classified_ball],
    )

    assert detector.detect_balls_with_classification(frame) == [classified_ball]


@pytest.mark.unit
def test_classification_outcome_reports_successful_empty_and_classified_balls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector, model = _bare_detector("cpu")
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    empty = detector.detect_balls_with_classification_outcome(frame)

    assert empty.status is BallDetectionStatus.NO_DETECTIONS
    assert empty.detections == ()

    model.result = _ultralytics_result()
    monkeypatch.setattr(detector, "_adapt_and_classify_balls", lambda *_args: [])

    filtered_empty = detector.detect_balls_with_classification_outcome(frame)

    assert filtered_empty.status is BallDetectionStatus.NO_DETECTIONS
    assert filtered_empty.detections == ()

    classified_ball = Ball(
        position=(22.0, 30.0),
        radius=12.0,
        ball_type=BallType.EIGHT,
        confidence=0.93,
    )
    monkeypatch.setattr(
        detector,
        "_adapt_and_classify_balls",
        lambda *_args: [classified_ball],
    )

    classified = detector.detect_balls_with_classification_outcome(frame)

    assert classified.status is BallDetectionStatus.DETECTIONS
    assert classified.detections == (classified_ball,)
    assert isinstance(classified.detections[0], Ball)


@pytest.mark.unit
def test_classification_failure_is_stage_tagged_without_raw_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector, model = _bare_detector("cpu")
    model.result = _ultralytics_result()

    def fail_classification(*_args: Any) -> list[Ball]:
        raise RuntimeError("private classifier detail")

    monkeypatch.setattr(detector, "_adapt_and_classify_balls", fail_classification)

    outcome = detector.detect_balls_with_classification_outcome(
        np.zeros((64, 96, 3), dtype=np.uint8)
    )

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.CLASSIFICATION
    assert outcome.detections == ()
    assert "private classifier detail" not in caplog.text
    assert (
        detector.detect_balls_with_classification(np.zeros((64, 96, 3), dtype=np.uint8))
        == []
    )


@pytest.mark.unit
def test_classification_outcome_preserves_legacy_instance_and_subclass_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    detection = Detection(
        class_id=0,
        class_name="ball",
        confidence=0.93,
        bbox=(12.0, 18.0, 32.0, 42.0),
        center=(22.0, 30.0),
        width=20.0,
        height=24.0,
    )
    ball = Ball(
        position=(22.0, 30.0),
        radius=12.0,
        ball_type=BallType.EIGHT,
        confidence=0.93,
    )

    instance_detector, _model = _bare_detector("cpu")
    instance_detector.model_loaded = False
    monkeypatch.setattr(
        instance_detector,
        "detect_balls",
        lambda _frame, *, frame_key=None: [detection],
    )
    monkeypatch.setattr(
        instance_detector,
        "_adapt_and_classify_balls",
        lambda *_args: [ball],
    )

    instance_outcome = instance_detector.detect_balls_with_classification_outcome(
        frame, frame_key="instance-override"
    )

    assert instance_outcome.status is BallDetectionStatus.DETECTIONS
    assert instance_outcome.detections == (ball,)

    class LegacySubclass(YOLODetector):
        legacy_detections: list[Detection]
        observed_frame_key: Any

        def detect_balls(
            self, _frame: np.ndarray, *, frame_key: Any = None
        ) -> list[Detection]:
            self.observed_frame_key = frame_key
            return self.legacy_detections

    base_detector, _model = _bare_detector("cpu")
    subclass_detector = LegacySubclass.__new__(LegacySubclass)
    subclass_detector.__dict__.update(base_detector.__dict__)
    subclass_detector.model_loaded = False
    subclass_detector.legacy_detections = [detection]
    monkeypatch.setattr(
        subclass_detector,
        "_adapt_and_classify_balls",
        lambda *_args: [ball],
    )

    subclass_outcome = subclass_detector.detect_balls_with_classification_outcome(
        frame, frame_key="subclass-override"
    )

    assert subclass_outcome.status is BallDetectionStatus.DETECTIONS
    assert subclass_outcome.detections == (ball,)
    assert subclass_detector.observed_frame_key == "subclass-override"


@pytest.mark.unit
def test_classification_outcome_maps_legacy_override_exception_to_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    detector, _model = _bare_detector("cpu")
    detector.model_loaded = False

    def fail_override(_frame: np.ndarray, *, frame_key: Any = None) -> list[Detection]:
        raise RuntimeError("private override detail")

    monkeypatch.setattr(detector, "detect_balls", fail_override)

    outcome = detector.detect_balls_with_classification_outcome(
        np.zeros((64, 96, 3), dtype=np.uint8)
    )

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.UNKNOWN
    assert outcome.detections == ()
    assert "private override detail" not in caplog.text


@pytest.mark.unit
def test_standard_inference_duration_uses_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    detector, _model = _bare_detector("cpu")
    timestamps = iter([10.0, 10.001, 10.004, 10.006])
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(
        time,
        "time",
        lambda: pytest.fail("elapsed detector time must not use wall clock"),
    )

    assert detector._run_standard_inference(np.zeros((24, 32, 3), dtype=np.uint8)) == []

    # Six milliseconds total is fed into the existing 0.1 EMA coefficient.
    assert detector.stats["avg_inference_time"] == pytest.approx(0.6)


@pytest.mark.unit
@pytest.mark.parametrize("feature_axis_first", [False, True])
def test_edge_tpu_decoder_handles_both_prediction_axis_layouts(
    feature_axis_first: bool,
) -> None:
    detector, _model = _bare_detector("cpu")
    detector.class_names = {0: "ball", 1: "cue"}
    predictions = np.zeros((12, 6), dtype=np.float32)
    predictions[7] = [50.0, 25.0, 20.0, 10.0, 0.91, 0.09]
    output = (
        predictions.T[np.newaxis, :, :]
        if feature_axis_first
        else predictions[np.newaxis, :, :]
    )

    detections = detector._parse_tpu_output(
        [output], original_shape=(200, 400, 3), model_input_shape=(100, 200)
    )

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].class_name == "ball"
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bbox == pytest.approx((80.0, 40.0, 120.0, 60.0))
    assert detections[0].center == pytest.approx((100.0, 50.0))


@pytest.mark.unit
def test_tpu_nms_keeps_distinct_balls_at_large_image_coordinates() -> None:
    detector, _model = _bare_detector("cpu")
    detections = [
        Detection(
            class_id=0,
            class_name="ball",
            confidence=0.95,
            bbox=(1000.0, 500.0, 1020.0, 520.0),
            center=(1010.0, 510.0),
            width=20.0,
            height=20.0,
        ),
        Detection(
            class_id=0,
            class_name="ball",
            confidence=0.90,
            bbox=(1100.0, 500.0, 1120.0, 520.0),
            center=(1110.0, 510.0),
            width=20.0,
            height=20.0,
        ),
    ]

    retained = detector._apply_nms(detections, iou_threshold=0.45)

    assert retained == detections
