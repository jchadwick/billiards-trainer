"""Deterministic contract tests for detector-to-Core ball localization.

The fake model matches Ultralytics' result shape but does not estimate visual
accuracy. These tests exercise the real decoder, adapter, IntegrationService
serializer, and Core state update without requiring a camera or model file.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from backend.core.coordinates import Vector2D
from backend.core.game_state import GameStateManager
from backend.core.models import TableState
from backend.integration_service import IntegrationService
from backend.integration_service_conversion_helpers import CoordinateConversionError
from backend.vision.detection.yolo_detector import (
    BallDetectionStatus,
    ModelFormat,
    YOLODetector,
)
from backend.vision.models import DetectionResult, FrameStatistics

if TYPE_CHECKING:
    from pathlib import Path

    from backend.common.models import Ball


class TensorLike:
    """Minimal tensor interface used by the actual Ultralytics decoder."""

    def __init__(self, values: Any) -> None:
        self._values = np.asarray(values)

    def __getitem__(self, index: Any) -> TensorLike:
        return TensorLike(self._values[index])

    def cpu(self) -> TensorLike:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self._values


class UltralyticsResultModel:
    """Recording stand-in returning boxes, classes, and confidences."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox
        self.predictor = SimpleNamespace(device=None)
        self.bbox = bbox
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    def __call__(self, frame: np.ndarray, **kwargs: Any) -> list[Any]:
        self.calls.append((frame, kwargs))
        self.predictor.device = kwargs.get("device")
        return [
            SimpleNamespace(
                boxes=[
                    SimpleNamespace(
                        xyxy=TensorLike([self.bbox]),
                        cls=TensorLike([0.0]),
                        conf=TensorLike([0.93]),
                    )
                ]
            )
        ]


@dataclass(frozen=True)
class ReplayCase:
    width: int
    height: int
    bbox: tuple[float, float, float, float]


@pytest.fixture(
    params=[
        ReplayCase(1920, 1080, (942.0, 522.0, 978.0, 558.0)),
        ReplayCase(3840, 2160, (1884.0, 1044.0, 1956.0, 1116.0)),
    ]
)
def replay_case(request: pytest.FixtureRequest) -> ReplayCase:
    return cast("ReplayCase", request.param)


def _detector_for(case: ReplayCase) -> tuple[YOLODetector, UltralyticsResultModel]:
    detector = YOLODetector.__new__(YOLODetector)
    model = UltralyticsResultModel(case.bbox)
    detector.model_path = None
    detector.device = "cpu"
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
    detector._opencv_classifier = None
    detector.stats = {
        "total_inferences": 0,
        "total_detections": 0,
        "avg_inference_time": 0.0,
        "fallback_mode": False,
        "using_tpu": False,
    }
    return detector, model


def _calibrated_table() -> TableState:
    return TableState(
        width=3200,
        height=1600,
        pocket_positions=[
            Vector2D.from_4k(x, y)
            for x, y in (
                (320, 280),
                (1920, 280),
                (3520, 280),
                (320, 1880),
                (1920, 1880),
                (3520, 1880),
            )
        ],
        playing_area_corners=[
            Vector2D.from_4k(x, y)
            for x, y in ((320, 280), (3520, 280), (3520, 1880), (320, 1880))
        ],
    )


def _vision_result(case: ReplayCase, balls: list[Ball]) -> DetectionResult:
    source_time = 1_800_000_000.25
    for ball in balls:
        ball.last_seen = source_time
    return DetectionResult(
        frame_number=11,
        timestamp=source_time,
        balls=balls,
        cue=None,
        table=None,
        statistics=FrameStatistics(
            frame_number=11, timestamp=source_time, processing_time=0.0
        ),
        frame_size=(case.width, case.height),
        source_frame_number=11,
        source_timestamp=source_time,
        source_timestamp_kind="shared_memory_publication",
        source_process_id=321,
        source_publication_monotonic_ns=987_654_321,
        ball_detection_status="observed",
        ball_positions_status="observed",
    )


def _integration_service() -> IntegrationService:
    """Construct only the stateless conversion surface needed in this test."""
    service = IntegrationService.__new__(IntegrationService)
    service._vision_instance_id = "detector-core-contract-run"
    service._vision_process_id = 654
    return service


@pytest.mark.integration
def test_real_detector_adapter_and_core_preserve_canonical_localization(
    replay_case: ReplayCase, tmp_path: Path
) -> None:
    frame = np.zeros((replay_case.height, replay_case.width, 3), dtype=np.uint8)
    detector, model = _detector_for(replay_case)

    outcome = detector.detect_balls_with_classification_outcome(
        frame, min_confidence=0.25, frame_key=("replay", replay_case.width)
    )

    assert outcome.status is BallDetectionStatus.DETECTIONS
    assert len(outcome.detections) == 1
    vision_ball = outcome.detections[0]
    assert vision_ball.position == pytest.approx((1920.0, 1080.0))
    assert vision_ball.radius == pytest.approx(36.0)
    assert vision_ball.confidence == pytest.approx(0.93)
    assert vision_ball.source_resolution == (replay_case.width, replay_case.height)
    assert vision_ball.ball_type.value == "unknown"
    assert vision_ball.number is None
    assert len(model.calls) == 1

    detection = _vision_result(replay_case, list(outcome.detections))
    integration = _integration_service()
    core_payload = integration._convert_detection_to_core_format(detection)
    assert core_payload["observation_status"]["source_frame_number"] == 11
    assert core_payload["observation_status"]["vision_instance_id"] == (
        "detector-core-contract-run"
    )
    assert core_payload["balls"][0]["position"] == {
        "x": 1920.0,
        "y": 1080.0,
        "scale": [1.0, 1.0],
    }
    assert core_payload["source_resolution"] == [
        replay_case.width,
        replay_case.height,
    ]

    state_manager = GameStateManager(
        table=_calibrated_table(),
        persistence_path=tmp_path / f"state-{replay_case.width}",
    )
    state = state_manager.update_state(core_payload)
    assert len(state.balls) == 1
    core_ball = state.balls[0]
    assert core_ball.position.x == pytest.approx(1920.0)
    assert core_ball.position.y == pytest.approx(1080.0)
    assert core_ball.radius == pytest.approx(36.0)
    assert core_ball.confidence == pytest.approx(0.93)
    assert core_ball.number is None
    assert core_ball.is_cue_ball is False
    assert state.coordinate_metadata is not None
    assert state.coordinate_metadata.camera_resolution == (
        replay_case.width,
        replay_case.height,
    )


@pytest.mark.integration
def test_pixel_ball_without_source_resolution_cannot_replace_valid_core_state(
    tmp_path: Path,
) -> None:
    case = ReplayCase(1920, 1080, (942.0, 522.0, 978.0, 558.0))
    frame = np.zeros((case.height, case.width, 3), dtype=np.uint8)
    detector, _model = _detector_for(case)
    outcome = detector.detect_balls_with_classification_outcome(frame)
    assert outcome.status is BallDetectionStatus.DETECTIONS

    service = _integration_service()
    state_manager = GameStateManager(
        table=_calibrated_table(), persistence_path=tmp_path / "state"
    )
    valid_state = state_manager.update_state(
        service._convert_detection_to_core_format(
            _vision_result(case, list(outcome.detections))
        )
    )
    retained_ball = valid_state.balls[0]

    invalid_ball = outcome.detections[0]
    invalid_ball.coordinate_space = "pixel"
    invalid_ball.source_resolution = None
    invalid_detection = _vision_result(ReplayCase(0, 0, case.bbox), [invalid_ball])
    with pytest.raises(CoordinateConversionError, match="no valid source resolution"):
        service._convert_detection_to_core_format(invalid_detection)

    assert state_manager.get_current_state() is valid_state
    state_after_failure = state_manager.get_current_state()
    assert state_after_failure is not None
    retained_after_failure = state_after_failure.balls[0]
    assert retained_after_failure.position == retained_ball.position
    assert retained_after_failure.number is None


@pytest.mark.integration
def test_unknown_frame_resolution_is_not_replaced_by_configured_resolution(
    tmp_path: Path,
) -> None:
    case = ReplayCase(1920, 1080, (942.0, 522.0, 978.0, 558.0))
    frame = np.zeros((case.height, case.width, 3), dtype=np.uint8)
    detector, _model = _detector_for(case)
    outcome = detector.detect_balls_with_classification_outcome(frame)
    assert outcome.status is BallDetectionStatus.DETECTIONS

    detection = _vision_result(case, list(outcome.detections))
    detection.frame_size = (0, 0)
    payload = _integration_service()._convert_detection_to_core_format(detection)
    assert payload["source_resolution"] is None

    manager = GameStateManager(
        table=_calibrated_table(), persistence_path=tmp_path / "unknown-resolution"
    )
    state = manager.update_state(payload)
    assert state.coordinate_metadata is not None
    assert state.coordinate_metadata.camera_resolution is None
