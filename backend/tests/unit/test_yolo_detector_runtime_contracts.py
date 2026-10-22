"""Behavior tests for YOLO backend loading, TPU input, and detection contracts.

The tests use tiny fake runtimes and deterministic arrays so they exercise the
production adapter logic without loading model artifacts or requiring a GPU or
Coral device.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from backend.vision.detection.yolo_detector import (
    BallDetectionStatus,
    Detection,
    ModelFormat,
    ModelValidationError,
    TableElements,
    YOLODetector,
)

if TYPE_CHECKING:
    from pathlib import Path


class FakeTpuInterpreter:
    """Record the tensor contract used by a TPU runtime."""

    def __init__(self, input_dtype: Any = np.uint8) -> None:
        self.input_dtype = input_dtype
        self.input_tensor: np.ndarray[Any, Any] | None = None
        self.invoked = False

    def allocate_tensors(self) -> None:
        return None

    def get_input_details(self) -> list[dict[str, Any]]:
        return [
            {
                "shape": np.array([1, 2, 3, 3]),
                "dtype": self.input_dtype,
                "index": 7,
            }
        ]

    def get_output_details(self) -> list[dict[str, int]]:
        return [{"index": 9}]

    def set_tensor(self, index: int, tensor: np.ndarray[Any, Any]) -> None:
        assert index == 7
        self.input_tensor = tensor

    def invoke(self) -> None:
        self.invoked = True

    def get_tensor(self, index: int) -> np.ndarray[Any, Any]:
        assert index == 9
        # Use more prediction rows than features, as in the common [1, N, F]
        # export layout. The first row is a centered ball; remaining rows have
        # scores below the detector threshold.
        predictions = np.zeros((12, 6), dtype=np.float32)
        predictions[0] = [1.5, 1.0, 1.0, 1.0, 0.9, 0.1]
        predictions[1:, 4:] = 0.1
        return predictions[np.newaxis, :, :]


def _detection(
    class_id: int,
    name: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=confidence,
        bbox=bbox,
        center=((x1 + x2) / 2, (y1 + y2) / 2),
        width=x2 - x1,
        height=y2 - y1,
    )


@pytest.mark.unit
def test_model_loader_sets_format_and_model_class_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "balls.pt"
    model_path.write_bytes(b"test model artifact")
    loaded_model = SimpleNamespace(names={0: "cue", 1: "eight_ball"})
    loaded_paths: list[str] = []

    def load_model(path: str) -> Any:
        loaded_paths.append(path)
        return loaded_model

    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module",
        lambda name: (
            SimpleNamespace(YOLO=load_model)
            if name == "ultralytics"
            else pytest.fail(f"unexpected import: {name}")
        ),
    )
    detector = YOLODetector(device="cpu")

    detector._load_model(str(model_path))

    assert loaded_paths == [str(model_path)]
    assert detector.model is loaded_model
    assert detector.model_loaded is True
    assert detector.model_format is ModelFormat.PYTORCH
    assert detector.class_names == {0: "cue", 1: "eight_ball"}
    assert detector.name_to_class == {"cue": 0, "eight_ball": 1}


@pytest.mark.unit
def test_model_loader_rejects_missing_and_unsupported_artifacts(
    tmp_path: Path,
) -> None:
    detector = YOLODetector(device="cpu")

    with pytest.raises(FileNotFoundError, match="not found"):
        detector._load_model(str(tmp_path / "missing.pt"))

    unsupported = tmp_path / "balls.bin"
    unsupported.write_bytes(b"artifact")
    with pytest.raises(ValueError, match="Unsupported model format"):
        detector._load_model(str(unsupported))


@pytest.mark.unit
def test_constructor_fallback_policy_distinguishes_optional_and_required_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "missing.pt"

    optional = YOLODetector(model_path=str(model_path), device="cpu")
    assert optional.model_loaded is False
    assert optional.stats["fallback_mode"] is True


@pytest.mark.unit
def test_constructor_raises_when_model_loading_fails_without_fallback(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "missing.pt"
    with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
        YOLODetector(model_path=str(model_path), device="cpu", auto_fallback=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("device_path", "expected_device"),
    [
        ("usb", "usb"),
        ("pcie", "pci"),
        ("/dev/bus/usb/001/002", "/dev/bus/usb/001/002"),
        ("unknown", None),
        (None, None),
    ],
)
def test_tpu_loader_maps_explicit_device_or_uses_auto_detection(
    device_path: str | None,
    expected_device: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = FakeTpuInterpreter()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def make_interpreter(*args: Any, **kwargs: Any) -> FakeTpuInterpreter:
        calls.append((args, kwargs))
        return interpreter

    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.import_module",
        lambda _name: SimpleNamespace(make_interpreter=make_interpreter),
    )
    detector = YOLODetector(device="cpu")
    detector.tpu_device_path = device_path

    detector._load_tpu_model("balls.tflite")

    if expected_device is None:
        assert calls == [(("balls.tflite",), {})]
    else:
        assert calls == [(("balls.tflite",), {"device": expected_device})]
    assert detector.tpu_interpreter is interpreter
    assert detector.tpu_input_details is not None
    assert detector.tpu_input_details[0]["index"] == 7
    np.testing.assert_array_equal(
        detector.tpu_input_details[0]["shape"], np.array([1, 2, 3, 3])
    )
    assert detector.tpu_output_details == [{"index": 9}]
    assert detector.model_loaded is True


@pytest.mark.unit
@pytest.mark.parametrize("input_dtype", [np.uint8, np.float32])
def test_tpu_inference_converts_bgr_resizes_normalizes_and_decodes(
    input_dtype: Any,
) -> None:
    detector = YOLODetector(device="cpu")
    interpreter = FakeTpuInterpreter(input_dtype)
    detector.model_loaded = True
    detector.model_format = ModelFormat.EDGETPU
    detector.tpu_interpreter = interpreter
    detector.tpu_input_details = interpreter.get_input_details()
    detector.tpu_output_details = interpreter.get_output_details()
    detector.confidence = 0.5
    detector.nms_threshold = 0.45
    detector.class_names = {0: "ball", 1: "cue"}
    frame = np.empty((2, 3, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)  # BGR

    detections = detector._run_tpu_inference(frame)

    assert interpreter.invoked is True
    assert interpreter.input_tensor is not None
    assert interpreter.input_tensor.shape == (1, 2, 3, 3)
    expected_rgb = np.array([30, 20, 10], dtype=np.float32)
    expected = expected_rgb if input_dtype is np.uint8 else expected_rgb / 255.0
    np.testing.assert_allclose(interpreter.input_tensor[0, 0, 0], expected)
    assert len(detections) == 1
    assert detections[0].class_name == "ball"
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].bbox == pytest.approx((1.0, 0.5, 2.0, 1.5))
    assert detector.stats["total_inferences"] == 1
    assert detector.stats["total_detections"] == 1


@pytest.mark.unit
def test_ball_filter_excludes_cue_small_marker_and_non_round_boxes() -> None:
    detector = YOLODetector(device="cpu", min_ball_size=20)
    detections = [
        _detection(0, "ball", 0.9, (0.0, 0.0, 20.0, 20.0)),
        _detection(1, "cue", 0.9, (30.0, 0.0, 60.0, 5.0)),
        _detection(2, "ball_2", 0.9, (70.0, 0.0, 85.0, 15.0)),
        _detection(3, "ball_3", 0.9, (90.0, 0.0, 150.0, 10.0)),
        _detection(18, "pocket", 0.99, (160.0, 0.0, 200.0, 40.0)),
    ]

    assert detector._filter_ball_detections(detections) == [detections[0]]


@pytest.mark.unit
def test_ball_filter_accepts_named_generic_ball_class_from_two_class_model() -> None:
    detector = YOLODetector(device="cpu")
    detector.class_names = {0: "ball", 1: "cue"}
    generic_ball = _detection(19, "ball", 0.85, (10.0, 10.0, 32.0, 32.0))
    cue = _detection(1, "cue_stick", 0.95, (40.0, 0.0, 80.0, 4.0))

    assert detector._filter_ball_detections([generic_ball, cue]) == [generic_ball]


@pytest.mark.unit
def test_cue_selection_uses_highest_confidence_and_reuses_requested_frame_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detector.class_names = {0: "ball", 1: "cue"}
    lower = _detection(16, "cue_stick", 0.4, (0.0, 0.0, 50.0, 4.0))
    higher = _detection(1, "cue", 0.8, (0.0, 10.0, 60.0, 14.0))
    inference_calls: list[Any] = []

    def run_inference(_frame: Any, *, frame_key: Any = None) -> list[Detection]:
        inference_calls.append(frame_key)
        return [lower, higher]

    monkeypatch.setattr(
        detector,
        "_run_inference",
        run_inference,
    )
    monkeypatch.setattr(detector, "_estimate_cue_angle", lambda _frame, cue: 1.25)
    frame = np.zeros((24, 64, 3), dtype=np.uint8)

    cue = detector.detect_cue(frame, frame_key=(4, "full"))

    assert cue is higher
    assert cue.angle == pytest.approx(1.25)
    assert inference_calls == [(4, "full")]


@pytest.mark.unit
def test_table_detection_selects_best_surface_and_returns_all_pockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = YOLODetector(device="cpu")
    detector.model_loaded = True
    detections = [
        _detection(17, "table", 0.7, (0.0, 0.0, 400.0, 200.0)),
        _detection(17, "table", 0.95, (5.0, 8.0, 405.0, 208.0)),
        _detection(18, "pocket", 0.8, (0.0, 0.0, 12.0, 12.0)),
        _detection(18, "pocket", 0.75, (390.0, 190.0, 402.0, 202.0)),
        _detection(2, "ball_2", 0.9, (40.0, 40.0, 60.0, 60.0)),
    ]
    monkeypatch.setattr(detector, "_run_inference", lambda _frame: detections)

    elements = detector.detect_table_elements(np.zeros((220, 420, 3), dtype=np.uint8))

    assert isinstance(elements, TableElements)
    assert elements.table_bbox == detections[1].bbox
    assert elements.table_confidence == pytest.approx(0.95)
    assert elements.pockets == [detections[2], detections[3]]
    assert elements.rails == []


@pytest.mark.unit
def test_threshold_updates_are_validated_without_partial_update() -> None:
    detector = YOLODetector(device="cpu", confidence=0.2, nms_threshold=0.4)

    with pytest.raises(ValueError, match="Confidence must be between"):
        detector.update_thresholds(confidence=1.1, nms=0.8)
    assert detector.confidence == pytest.approx(0.2)
    assert detector.nms_threshold == pytest.approx(0.4)

    detector.update_thresholds(confidence=0.0, nms=1.0)
    assert detector.confidence == 0.0
    assert detector.nms_threshold == 1.0


@pytest.mark.unit
def test_unavailable_model_returns_empty_table_and_unavailable_ball_outcome() -> None:
    detector = YOLODetector(device="cpu")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    assert (
        detector.detect_balls_outcome(frame).status
        is BallDetectionStatus.MODEL_UNAVAILABLE
    )
    assert detector.detect_cue(frame) is None
    assert detector.detect_table_elements(frame) == TableElements()


@pytest.mark.unit
def test_model_file_validation_checks_extension_and_minimum_size(
    tmp_path: Path,
) -> None:
    detector = YOLODetector(device="cpu")
    unsupported = tmp_path / "model.tflite"
    unsupported.write_bytes(b"x" * 200_000)
    with pytest.raises(ModelValidationError, match="Invalid model format"):
        detector._validate_model_file(str(unsupported))

    small = tmp_path / "small.onnx"
    small.write_bytes(b"x" * 256)
    with pytest.raises(ModelValidationError, match="suspiciously small"):
        detector._validate_model_file(str(small))

    valid = tmp_path / "valid.engine"
    valid.write_bytes(b"x" * (100 * 1024))
    assert detector._validate_model_file(str(valid)) is True
