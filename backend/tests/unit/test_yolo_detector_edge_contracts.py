"""Deterministic edge-contract tests for optional YOLO runtimes.

The tests exercise failure handling and model lifecycle behavior with small fake
runtimes. They deliberately avoid loading model weights, accelerators, or
platform-specific inference libraries.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from backend.vision.detection.yolo_detector import (
    BallDetectionStatus,
    DetectionFailureStage,
    ModelFormat,
    ModelValidationError,
    YOLODetector,
    _DetectionBackendFailure,
)

if TYPE_CHECKING:
    from pathlib import Path


class CallableModel:
    """Minimal Ultralytics-like callable with an optional inference error."""

    names = {0: "ball"}

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, frame: Any, **kwargs: Any) -> list[Any]:
        self.calls.append((frame, kwargs))
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(boxes=[])]


@pytest.mark.unit
def test_onnx_validation_without_optional_onnx_package_is_bounded_and_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"\x08\x01" + b"x" * 1_100)
    # Setting sys.modules to None makes Python's normal import path report the
    # same ModuleNotFoundError that an installation without ONNX would report.
    monkeypatch.setitem(sys.modules, "onnx", None)

    metadata = YOLODetector.validate_onnx_model(str(model_path))

    assert metadata["valid"] is True
    assert metadata["file_size"] == 1_102
    assert "Full validation requires 'onnx' package" in metadata["warning"]

    tiny_path = tmp_path / "tiny.onnx"
    tiny_path.write_bytes(b"x" * 12)
    with pytest.raises(ModelValidationError, match="too small"):
        YOLODetector.validate_onnx_model(str(tiny_path))


@pytest.mark.unit
def test_onnx_validation_extracts_dynamic_shapes_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"model")

    def tensor_value(value: int) -> SimpleNamespace:
        return SimpleNamespace(dim_value=value)

    def tensor_info(values: list[int]) -> SimpleNamespace:
        dimensions = [tensor_value(value) for value in values]
        return SimpleNamespace(
            type=SimpleNamespace(
                tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
            )
        )

    model = SimpleNamespace(
        opset_import=[SimpleNamespace(version=17)],
        producer_name="test-exporter",
        producer_version="1.2",
        graph=SimpleNamespace(
            input=[SimpleNamespace(name="images", **vars(tensor_info([1, 3, 0, 640])))],
            output=[
                SimpleNamespace(name="predictions", **vars(tensor_info([1, 0, 84])))
            ],
        ),
    )
    fake_onnx = SimpleNamespace(
        load=lambda _path: model,
        checker=SimpleNamespace(
            check_model=lambda _model: None, ValidationError=ValueError
        ),
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    metadata = YOLODetector.validate_onnx_model(str(model_path))

    assert metadata["valid"] is True
    assert metadata["opset_version"] == 17
    assert metadata["producer"] == "test-exporter"
    assert metadata["producer_version"] == "1.2"
    assert metadata["input_name"] == "images"
    assert metadata["input_shape"] == (1, 3, -1, 640)
    assert metadata["output_name"] == "predictions"
    assert metadata["output_shape"] == (1, -1, 84)
    assert metadata["num_classes"] == 79


@pytest.mark.unit
def test_onnx_checker_rejection_is_reported_as_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"model")

    class CheckerValidationError(Exception):
        pass

    fake_onnx = SimpleNamespace(
        load=lambda _path: object(),
        checker=SimpleNamespace(
            check_model=lambda _model: (_ for _ in ()).throw(
                CheckerValidationError("invalid graph")
            ),
            ValidationError=CheckerValidationError,
        ),
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    with pytest.raises(
        ModelValidationError, match="ONNX validation failed: invalid graph"
    ):
        YOLODetector.validate_onnx_model(str(model_path))


@pytest.mark.unit
def test_coreml_provider_absence_falls_back_to_ultralytics_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"onnx artifact")
    model = CallableModel()
    runtime_calls: list[str] = []

    def import_runtime(name: str) -> Any:
        runtime_calls.append(name)
        if name == "ultralytics":
            return SimpleNamespace(YOLO=lambda _path: model)
        if name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: ["CPUExecutionProvider"]
            )
        pytest.fail(f"unexpected runtime import: {name}")

    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module", import_runtime
    )
    detector = YOLODetector(device="coreml")

    detector._load_model(str(model_path))

    assert runtime_calls == ["ultralytics", "onnxruntime"]
    assert detector.device == "cpu"
    assert detector.model is model
    assert detector.onnx_session is None
    assert detector.model_loaded is True


@pytest.mark.unit
def test_coreml_session_creation_failure_does_not_claim_model_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"onnx artifact")

    class BrokenSessionRuntime:
        GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=1)

        def SessionOptions(self) -> Any:
            return SimpleNamespace()

        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("provider initialization detail")

    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module",
        lambda name: (
            SimpleNamespace(YOLO=CallableModel)
            if name == "ultralytics"
            else BrokenSessionRuntime
        ),
    )
    detector = YOLODetector(device="coreml")

    with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
        detector._load_model(str(model_path))

    assert detector.model_loaded is False
    assert detector.onnx_session is None


@pytest.mark.unit
def test_standard_inference_failure_has_safe_status_and_releases_model_lock() -> None:
    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    detector.model = CallableModel(RuntimeError("private runtime path detail"))

    outcome = detector.detect_balls_outcome(np.zeros((16, 16, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.INFERENCE
    assert outcome.detections == ()
    assert detector._model_lock.acquire(blocking=False)
    detector._model_lock.release()


@pytest.mark.unit
def test_malformed_standard_result_is_classified_as_decode_failure() -> None:
    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    # The backend call succeeds, but the result's box tensor violates the
    # expected xyxy interface. This is a decode failure, not a model outage.
    detector.model = lambda *_args, **_kwargs: [
        SimpleNamespace(boxes=[SimpleNamespace(xyxy=None, cls=None, conf=None)])
    ]

    outcome = detector.detect_balls_outcome(np.zeros((16, 16, 3), dtype=np.uint8))

    assert outcome.status is BallDetectionStatus.FAILED
    assert outcome.failure_stage is DetectionFailureStage.DECODING
    assert detector._model_lock.acquire(blocking=False)
    detector._model_lock.release()


@pytest.mark.unit
def test_tpu_runtime_failure_is_stage_labeled_and_releases_lock() -> None:
    class BrokenInterpreter:
        def set_tensor(self, _index: int, _tensor: Any) -> None:
            raise RuntimeError("device failure details")

    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detector.model_format = ModelFormat.EDGETPU
    detector.tpu_interpreter = BrokenInterpreter()
    detector.tpu_input_details = [
        {"shape": np.array([1, 2, 2, 3]), "dtype": np.uint8, "index": 0}
    ]
    detector.tpu_output_details = [{"index": 0}]

    with pytest.raises(_DetectionBackendFailure) as error:
        detector._run_tpu_inference(np.zeros((2, 2, 3), dtype=np.uint8))

    assert error.value.stage is DetectionFailureStage.INFERENCE
    assert detector._model_lock.acquire(blocking=False)
    detector._model_lock.release()


@pytest.mark.unit
def test_tpu_decode_failure_is_reported_separately_from_runtime_failure() -> None:
    class InvalidOutputInterpreter:
        def set_tensor(self, _index: int, _tensor: Any) -> None:
            return None

        def invoke(self) -> None:
            return None

        def get_tensor(self, _index: int) -> np.ndarray[Any, Any]:
            raise ValueError("malformed output tensor")

    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detector.model_format = ModelFormat.EDGETPU
    detector.tpu_interpreter = InvalidOutputInterpreter()
    detector.tpu_input_details = [
        {"shape": np.array([1, 2, 2, 3]), "dtype": np.uint8, "index": 0}
    ]
    detector.tpu_output_details = [{"index": 0}]

    with pytest.raises(_DetectionBackendFailure) as error:
        detector._run_tpu_inference(np.zeros((2, 2, 3), dtype=np.uint8))

    assert error.value.stage is DetectionFailureStage.INFERENCE
    assert detector._model_lock.acquire(blocking=False)
    detector._model_lock.release()


@pytest.mark.unit
def test_successful_model_reload_invalidates_cached_frame_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "replacement.pt"
    model_path.write_bytes(b"x" * (100 * 1024))
    detector = YOLODetector(device="cpu")
    old_model = CallableModel()
    new_model = CallableModel()
    detector.model = old_model
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    detector.model_path = str(model_path)
    detector.stats["fallback_mode"] = True
    detector._detection_cache[(0, (12, "full"))] = []

    def load_replacement(path: str) -> None:
        assert path == str(model_path)
        detector.model = new_model
        detector.model_format = ModelFormat.PYTORCH
        detector.model_loaded = True

    monkeypatch.setattr(detector, "_load_model", load_replacement)

    assert detector.reload_model() is True
    assert detector.model is new_model
    assert detector.model_loaded is True
    assert detector._inference_model_generation == 1
    assert len(cast("Any", new_model).calls) == 1  # reload smoke-test inference
    assert detector.stats["fallback_mode"] is False


@pytest.mark.unit
def test_failed_model_reload_restores_usable_previous_model_and_reports_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path = tmp_path / "old.pt"
    old_path.write_bytes(b"x" * (100 * 1024))
    new_path = tmp_path / "new.pt"
    new_path.write_bytes(b"x" * (100 * 1024))
    detector = YOLODetector(device="cpu")
    old_model = CallableModel()
    detector.model = old_model
    detector.model_loaded = True
    detector.model_format = ModelFormat.PYTORCH
    detector.model_path = str(old_path)
    original_generation = detector._inference_model_generation

    def fail_loading(_path: str) -> None:
        detector.model_format = ModelFormat.ONNX
        raise RuntimeError("new backend cannot initialize")

    monkeypatch.setattr(detector, "_load_model", fail_loading)

    assert detector.reload_model(str(new_path)) is False
    assert detector.model is old_model
    assert detector.model_path == str(old_path)
    assert detector.model_format is ModelFormat.PYTORCH
    assert detector.model_loaded is True
    assert cast("Any", detector._inference_model_generation) == original_generation
    # The old model was actively called to verify restoration.
    assert len(cast("Any", old_model).calls) == 1
    assert detector.stats["fallback_mode"] is True
