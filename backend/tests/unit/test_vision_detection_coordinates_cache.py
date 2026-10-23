"""Regression coverage for vision coordinate spaces and inference reuse."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from backend.common.models import Ball
from backend.config import config
from backend.vision import VisionModule, VisionStatistics
from backend.vision.detection.detector_adapter import canonical_point_to_source_pixels
from backend.vision.detection.yolo_detector import Detection, ModelFormat, YOLODetector
from backend.vision.models import BallType
from backend.vision.phase_scheduler import VisionPhaseScheduler


@pytest.mark.unit
def test_canonical_point_converts_back_to_source_image_pixels() -> None:
    assert canonical_point_to_source_pixels((1920.0, 1080.0), (1920, 1080)) == (
        960.0,
        540.0,
    )


@pytest.mark.unit
def test_invalid_region_filter_receives_camera_pixels_and_detector_shares_frame_key() -> (
    None
):
    source_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ball = Ball(
        position=(1920.0, 1080.0),
        radius=36.0,
        ball_type=BallType.UNKNOWN,
        confidence=0.9,
        coordinate_space="4k_canonical",
        source_resolution=(1920, 1080),
    )

    class FakeDetector:
        def __init__(self) -> None:
            self.ball_frame_key = None
            self.cue_frame_key = None
            self.classification_frame = None

        def detect_balls_with_classification(
            self, frame, *, frame_key=None, classification_frame=None
        ):
            self.ball_frame_key = frame_key
            self.classification_frame = classification_frame
            assert frame is not source_frame
            return [ball]

        def detect_cue(self, frame, *, frame_key=None):
            self.cue_frame_key = frame_key
            return None

    module = VisionModule.__new__(VisionModule)
    module.phase_scheduler = VisionPhaseScheduler({"enabled": False})
    module._frame_number = 0
    module.detector = FakeDetector()
    module.config = SimpleNamespace(
        enable_ball_detection=True,
        enable_tracking=False,
    )
    module.tracker = None
    module.stats = VisionStatistics()
    module._apply_all_masks = lambda frame: frame.copy()
    filtered_coordinates = []

    def record_filter_coordinates(x, y, width, height):
        filtered_coordinates.append((x, y, width, height))
        return False

    module._is_ball_in_invalid_region = record_filter_coordinates

    result = module._process_single_frame(
        source_frame,
        frame_number=17,
        source_frame_number=41,
        source_timestamp=123.5,
        source_timestamp_kind="publication",
    )

    assert result is not None
    assert filtered_coordinates == [(960.0, 540.0, 1920, 1080)]
    assert result.table is not None
    assert result.table.coordinate_space == "pixel"
    assert result.table.source_resolution == (1920, 1080)
    assert result.table.corners == [
        (0, 0),
        (1920, 0),
        (1920, 1080),
        (0, 1080),
    ]
    assert module.detector.classification_frame is source_frame
    assert module.detector.ball_frame_key == module.detector.cue_frame_key
    assert module.detector.ball_frame_key[1:5] == (
        17,
        41,
        123.5,
        "publication",
    )


def _bare_detector(detections: list[Detection]) -> YOLODetector:
    detector = YOLODetector.__new__(YOLODetector)
    detector.model_loaded = True
    detector.model_format = ModelFormat.ONNX
    detector._cache_lock = threading.Lock()
    detector._detection_cache = {}
    detector._run_standard_inference = Mock(return_value=detections)
    detector.class_names = {0: "ball", 16: "cue"}
    detector.min_ball_size = 4
    detector._estimate_cue_angle = Mock(return_value=0.0)
    detector._opencv_classifier = None
    return detector


@pytest.mark.unit
def test_ball_and_cue_methods_reuse_only_explicit_current_frame_view() -> None:
    ball = Detection(
        class_id=0,
        class_name="ball",
        confidence=0.9,
        bbox=(20.0, 20.0, 40.0, 40.0),
        center=(30.0, 30.0),
        width=20.0,
        height=20.0,
    )
    cue = Detection(
        class_id=16,
        class_name="cue",
        confidence=0.8,
        bbox=(50.0, 50.0, 90.0, 54.0),
        center=(70.0, 52.0),
        width=40.0,
        height=4.0,
    )
    detector = _bare_detector([ball, cue])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_key = ("source", 12, "masked-full-frame")

    assert detector.detect_balls(frame, frame_key=frame_key) == [ball]
    assert detector.detect_cue(frame, frame_key=frame_key) is cue
    assert detector._run_standard_inference.call_count == 1

    # A different view is a different inference key and replaces the sole entry.
    detector.detect_cue(frame, frame_key=("source", 12, "crop"))
    assert detector._run_standard_inference.call_count == 2
    assert len(detector._detection_cache) == 1

    # The legacy call shape remains supported and does not cache by buffer alone.
    detector._run_inference(frame)
    detector._run_inference(frame)
    assert detector._run_standard_inference.call_count == 4
    assert detector._detection_cache == {}


@pytest.mark.unit
@pytest.mark.parametrize("line_shape", [(1, 4), (1, 1, 4)])
def test_cue_angle_accepts_supported_opencv_hough_line_shapes(
    monkeypatch, line_shape
) -> None:
    detector = _bare_detector([])
    detector._estimate_cue_angle = YOLODetector._estimate_cue_angle.__get__(
        detector, YOLODetector
    )
    detection = Detection(
        class_id=16,
        class_name="cue",
        confidence=0.9,
        bbox=(0.0, 0.0, 100.0, 10.0),
        center=(50.0, 5.0),
        width=100.0,
        height=10.0,
    )
    line = np.array([10, 1, 50, 19], dtype=np.int32).reshape(line_shape)
    monkeypatch.setattr(
        "backend.vision.detection.yolo_detector.cv2.HoughLinesP",
        lambda *args, **kwargs: line,
    )

    angle = detector._estimate_cue_angle(
        np.zeros((100, 100, 3), dtype=np.uint8), detection
    )

    assert angle == pytest.approx(np.degrees(np.arctan2(18, 40)))


@pytest.mark.unit
def test_hybrid_classifier_crops_unmasked_frame_using_pixel_bbox(monkeypatch) -> None:
    class FakeClassifier:
        def __init__(self) -> None:
            self.region = None
            self.position = None
            self.radius = None

        def classify_ball_type(self, region, position, radius):
            self.region = region.copy()
            self.position = position
            self.radius = radius
            return BallType.CUE, 0.95, None

    detection = Detection(
        class_id=0,
        class_name="ball",
        confidence=0.9,
        bbox=(40.0, 40.0, 60.0, 60.0),
        center=(50.0, 50.0),
        width=20.0,
        height=20.0,
    )
    detector = _bare_detector([detection])
    detector.detect_balls = Mock(return_value=[detection])
    detector._opencv_classifier = FakeClassifier()
    monkeypatch.setattr(config, "get", lambda _path, default=None: default)

    inference_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    balls = detector.detect_balls_with_classification(
        inference_frame,
        frame_key=("source", 5, "masked-full-frame"),
        classification_frame=source_frame,
    )

    detector.detect_balls.assert_called_once_with(
        inference_frame, frame_key=("source", 5, "masked-full-frame")
    )
    assert len(balls) == 1
    assert detector._opencv_classifier.region.shape[:2] == (24, 24)
    assert np.all(detector._opencv_classifier.region == 200)
    assert detector._opencv_classifier.position == (50.0, 50.0)
    assert detector._opencv_classifier.radius == 10.0
    assert balls[0].ball_type is BallType.CUE
