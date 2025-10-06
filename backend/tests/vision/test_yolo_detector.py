"""Comprehensive unit tests for YOLODetector.

Tests cover:
- Model loading (.pt and .onnx formats)
- Inference on sample images
- Class ID mapping correctness
- Confidence and NMS threshold filtering
- Error handling (missing model, invalid format)
- Model validation and hot-swapping
- Detection visualization
- Thread safety

Uses mocked YOLO models for fast, reproducible tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from vision.detection.yolo_detector import (
    BallClass,
    Detection,
    ModelFormat,
    ModelValidationError,
    TableElements,
    YOLODetector,
    ball_class_to_type,
    create_detector,
)


@pytest.fixture()
def sample_frame():
    """Create a sample BGR frame for testing."""
    # 640x640 frame with green table and white ball
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, :] = [34, 139, 34]  # Green background
    cv2.circle(frame, (320, 320), 20, (255, 255, 255), -1)  # White ball
    return frame


@pytest.fixture()
def mock_yolo_model():
    """Create a mock YOLO model that returns realistic detections."""

    class MockBox:
        """Mock YOLO box result."""

        def __init__(self, xyxy, cls, conf):
            self.xyxy = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(xyxy)))]
            self.cls = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(cls)))]
            self.conf = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(conf)))]

    class MockResult:
        """Mock YOLO result."""

        def __init__(self, boxes_data):
            self.boxes = boxes_data

    def create_mock_result(frame, conf=0.4, iou=0.45, verbose=False):
        """Create mock detection results."""
        # Simulate detections: cue ball, ball 1, ball 8, cue stick
        boxes = [
            MockBox([300, 300, 340, 340], 0, 0.95),  # Cue ball
            MockBox([100, 100, 130, 130], 1, 0.88),  # Ball 1
            MockBox([500, 500, 530, 530], 8, 0.92),  # Ball 8 (eight ball)
            MockBox([200, 300, 600, 320], 16, 0.75),  # Cue stick
        ]

        # Filter by confidence threshold
        filtered_boxes = [b for b in boxes if b.conf[0].cpu().numpy() >= conf]

        return [MockResult(filtered_boxes)]

    mock_model = MagicMock()
    mock_model.side_effect = create_mock_result
    return mock_model


@pytest.fixture()
def temp_model_file(tmp_path):
    """Create a temporary model file for testing."""
    model_file = tmp_path / "test_model.pt"
    # Create a non-empty file (mock model)
    model_file.write_bytes(b"mock_yolo_model_data" * 10000)  # ~200KB
    return str(model_file)


@pytest.fixture()
def temp_onnx_model(tmp_path):
    """Create a temporary ONNX model file for testing."""
    model_file = tmp_path / "test_model.onnx"
    # Create a non-empty file (mock ONNX model)
    model_file.write_bytes(b"mock_onnx_model_data" * 10000)  # ~200KB
    return str(model_file)


class TestYOLODetectorInitialization:
    """Test YOLODetector initialization and configuration."""

    def test_init_without_model(self):
        """Test initialization without model path (fallback mode)."""
        detector = YOLODetector(model_path=None)

        assert detector.model_path is None
        assert detector.model is None
        assert not detector.model_loaded
        assert detector.stats["fallback_mode"] is True
        assert detector.device == "cpu"
        assert detector.confidence == 0.4
        assert detector.nms_threshold == 0.45

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        detector = YOLODetector(
            model_path=None,
            device="cuda",
            confidence=0.6,
            nms_threshold=0.5,
            auto_fallback=False,
        )

        assert detector.device == "cuda"
        assert detector.confidence == 0.6
        assert detector.nms_threshold == 0.5
        assert detector.auto_fallback is False

    @patch("vision.detection.yolo_detector.YOLO")
    def test_init_with_valid_model_pt(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test initialization with valid .pt model."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        assert detector.model_path == temp_model_file
        assert detector.model is not None
        assert detector.model_loaded is True
        assert detector.model_format == ModelFormat.PYTORCH
        assert detector.stats["fallback_mode"] is False
        mock_yolo_class.assert_called_once()

    @patch("vision.detection.yolo_detector.YOLO")
    def test_init_with_valid_model_onnx(
        self, mock_yolo_class, temp_onnx_model, mock_yolo_model
    ):
        """Test initialization with valid .onnx model."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_onnx_model, auto_fallback=False)

        assert detector.model_path == temp_onnx_model
        assert detector.model is not None
        assert detector.model_loaded is True
        assert detector.model_format == ModelFormat.ONNX
        assert detector.stats["fallback_mode"] is False

    def test_init_with_missing_model_auto_fallback(self):
        """Test initialization with missing model and auto_fallback=True."""
        detector = YOLODetector(
            model_path="/nonexistent/model.pt",
            auto_fallback=True,
        )

        assert not detector.model_loaded
        assert detector.stats["fallback_mode"] is True

    def test_init_with_missing_model_no_fallback(self):
        """Test initialization with missing model and auto_fallback=False raises error."""
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            YOLODetector(
                model_path="/nonexistent/model.pt",
                auto_fallback=False,
            )

    def test_init_with_invalid_format(self, tmp_path):
        """Test initialization with invalid model format."""
        invalid_model = tmp_path / "model.txt"
        invalid_model.write_text("not a model")

        with pytest.raises(RuntimeError, match="Unsupported model format"):
            YOLODetector(model_path=str(invalid_model), auto_fallback=False)


class TestClassMapping:
    """Test class ID to name mapping."""

    def test_class_names_mapping(self):
        """Test class ID to name mapping is correct."""
        detector = YOLODetector(model_path=None)

        # Ball classes
        assert detector.class_names[0] == "cue_ball"
        assert detector.class_names[1] == "ball_1"
        assert detector.class_names[8] == "ball_8"
        assert detector.class_names[15] == "ball_15"

        # Other objects
        assert detector.class_names[16] == "cue_stick"
        assert detector.class_names[17] == "table"
        assert detector.class_names[18] == "pocket"

    def test_name_to_class_mapping(self):
        """Test reverse mapping (name to class ID)."""
        detector = YOLODetector(model_path=None)

        assert detector.name_to_class["cue_ball"] == 0
        assert detector.name_to_class["ball_8"] == 8
        assert detector.name_to_class["cue_stick"] == 16
        assert detector.name_to_class["table"] == 17

    def test_ball_class_enum(self):
        """Test BallClass enum values."""
        assert BallClass.CUE_BALL.value == 0
        assert BallClass.BALL_1.value == 1
        assert BallClass.BALL_8.value == 8
        assert BallClass.BALL_15.value == 15
        assert BallClass.CUE_STICK.value == 16

    def test_ball_class_to_type_function(self):
        """Test ball_class_to_type conversion function."""
        # Cue ball
        ball_type, number = ball_class_to_type(0)
        assert ball_type == "cue"
        assert number is None

        # Solid balls (1-7)
        ball_type, number = ball_class_to_type(1)
        assert ball_type == "solid"
        assert number == 1

        ball_type, number = ball_class_to_type(7)
        assert ball_type == "solid"
        assert number == 7

        # Eight ball
        ball_type, number = ball_class_to_type(8)
        assert ball_type == "eight"
        assert number == 8

        # Stripe balls (9-15)
        ball_type, number = ball_class_to_type(9)
        assert ball_type == "stripe"
        assert number == 9

        ball_type, number = ball_class_to_type(15)
        assert ball_type == "stripe"
        assert number == 15

        # Unknown
        ball_type, number = ball_class_to_type(99)
        assert ball_type == "unknown"
        assert number is None


class TestInference:
    """Test YOLO inference functionality."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_inference_basic(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test basic inference returns detections."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        detections = detector._run_inference(sample_frame)

        assert isinstance(detections, list)
        assert len(detections) > 0

        # Check detection structure
        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox) == 4
            assert len(det.center) == 2
            assert det.width > 0
            assert det.height > 0

    @patch("vision.detection.yolo_detector.YOLO")
    def test_detect_balls(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test ball detection filters correctly."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        ball_detections = detector.detect_balls(sample_frame)

        # Should only return ball classes (0-15)
        assert isinstance(ball_detections, list)
        for det in ball_detections:
            assert 0 <= det.class_id <= 15

    @patch("vision.detection.yolo_detector.YOLO")
    def test_detect_cue(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test cue stick detection."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        cue_detection = detector.detect_cue(sample_frame)

        assert cue_detection is not None
        assert cue_detection.class_id == 16
        assert cue_detection.class_name == "cue_stick"
        assert hasattr(cue_detection, "angle")

    @patch("vision.detection.yolo_detector.YOLO")
    def test_detect_table_elements(
        self, mock_yolo_class, temp_model_file, sample_frame
    ):
        """Test table element detection."""
        # Create mock model that returns table detections
        mock_model = MagicMock()

        class MockBox:
            def __init__(self, xyxy, cls, conf):
                self.xyxy = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(xyxy)))]
                self.cls = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(cls)))]
                self.conf = [Mock(cpu=lambda: Mock(numpy=lambda: np.array(conf)))]

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        boxes = [
            MockBox([0, 0, 640, 480], 17, 0.95),  # Table
            MockBox([10, 10, 30, 30], 18, 0.88),  # Pocket 1
            MockBox([610, 10, 630, 30], 18, 0.85),  # Pocket 2
        ]

        mock_model.return_value = [MockResult(boxes)]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        elements = detector.detect_table_elements(sample_frame)

        assert isinstance(elements, TableElements)
        assert elements.table_bbox is not None
        assert elements.table_confidence > 0
        assert len(elements.pockets) == 2

    def test_inference_without_model(self, sample_frame):
        """Test inference without loaded model returns empty list."""
        detector = YOLODetector(model_path=None)

        assert detector.detect_balls(sample_frame) == []
        assert detector.detect_cue(sample_frame) is None
        assert isinstance(detector.detect_table_elements(sample_frame), TableElements)


class TestConfidenceFiltering:
    """Test confidence threshold filtering."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_confidence_threshold_low(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test low confidence threshold includes more detections."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(
            model_path=temp_model_file,
            confidence=0.1,  # Very low threshold
            auto_fallback=False,
        )

        detections = detector._run_inference(sample_frame)
        low_conf_count = len(detections)

        # All mock detections should be included
        assert low_conf_count >= 4  # Mock returns 4 detections

    @patch("vision.detection.yolo_detector.YOLO")
    def test_confidence_threshold_high(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test high confidence threshold filters out low confidence detections."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(
            model_path=temp_model_file,
            confidence=0.9,  # Very high threshold
            auto_fallback=False,
        )

        detections = detector._run_inference(sample_frame)
        high_conf_count = len(detections)

        # Only high confidence detections should remain
        assert high_conf_count <= 2  # Only cue ball (0.95) and ball 8 (0.92)

        # All remaining detections should have confidence >= 0.9
        for det in detections:
            assert det.confidence >= 0.9

    @patch("vision.detection.yolo_detector.YOLO")
    def test_update_confidence_threshold(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test updating confidence threshold."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Update threshold
        detector.update_thresholds(confidence=0.7)
        assert detector.confidence == 0.7

        # Test invalid threshold
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            detector.update_thresholds(confidence=1.5)

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            detector.update_thresholds(confidence=-0.1)


class TestNMSThreshold:
    """Test Non-Maximum Suppression threshold."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_nms_threshold_setting(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test NMS threshold is set correctly."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(
            model_path=temp_model_file,
            nms_threshold=0.3,
            auto_fallback=False,
        )

        assert detector.nms_threshold == 0.3

    @patch("vision.detection.yolo_detector.YOLO")
    def test_update_nms_threshold(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test updating NMS threshold."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Update threshold
        detector.update_thresholds(nms=0.6)
        assert detector.nms_threshold == 0.6

        # Test invalid threshold
        with pytest.raises(
            ValueError, match="NMS threshold must be between 0.0 and 1.0"
        ):
            detector.update_thresholds(nms=1.5)

        with pytest.raises(
            ValueError, match="NMS threshold must be between 0.0 and 1.0"
        ):
            detector.update_thresholds(nms=-0.1)


class TestErrorHandling:
    """Test error handling for various failure modes."""

    def test_missing_model_error(self):
        """Test error when model file is missing."""
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            YOLODetector(
                model_path="/nonexistent/path/model.pt",
                auto_fallback=False,
            )

    def test_invalid_format_error(self, tmp_path):
        """Test error when model format is invalid."""
        invalid_file = tmp_path / "model.txt"
        invalid_file.write_text("not a model")

        with pytest.raises(RuntimeError):
            YOLODetector(model_path=str(invalid_file), auto_fallback=False)

    def test_empty_model_file_error(self, tmp_path):
        """Test error when model file is empty."""
        empty_file = tmp_path / "model.pt"
        empty_file.write_bytes(b"")

        with pytest.raises(RuntimeError):
            YOLODetector(model_path=str(empty_file), auto_fallback=False)

    @patch("vision.detection.yolo_detector.YOLO")
    def test_inference_error_handling(
        self, mock_yolo_class, temp_model_file, sample_frame
    ):
        """Test inference error handling."""
        # Mock model that raises exception
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("Inference failed")
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Should handle error gracefully and return empty list
        detections = detector._run_inference(sample_frame)
        assert detections == []

    @patch("vision.detection.yolo_detector.YOLO")
    def test_corrupted_results_handling(
        self, mock_yolo_class, temp_model_file, sample_frame
    ):
        """Test handling of corrupted/malformed results."""
        # Mock model that returns malformed results
        mock_model = MagicMock()
        mock_model.return_value = None  # Invalid result
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Should handle gracefully
        detections = detector._run_inference(sample_frame)
        assert detections == []


class TestModelValidation:
    """Test ONNX model validation functionality."""

    def test_validate_onnx_missing_file(self):
        """Test validation fails for missing file."""
        with pytest.raises(ModelValidationError, match="Model file not found"):
            YOLODetector.validate_onnx_model("/nonexistent/model.onnx")

    def test_validate_onnx_wrong_extension(self, tmp_path):
        """Test validation fails for wrong file extension."""
        wrong_file = tmp_path / "model.pt"
        wrong_file.write_bytes(b"data")

        with pytest.raises(ModelValidationError, match="not an ONNX model"):
            YOLODetector.validate_onnx_model(str(wrong_file))

    def test_validate_onnx_basic_without_onnx_package(self, temp_onnx_model):
        """Test basic validation works without onnx package."""
        # This will use the basic validation path
        with patch("vision.detection.yolo_detector.logger"):
            metadata = YOLODetector.validate_onnx_model(temp_onnx_model)

        assert metadata["valid"] is True
        assert metadata["file_size"] > 0
        assert "warning" in metadata or "error" not in metadata

    @patch("vision.detection.yolo_detector.onnx")
    def test_validate_onnx_with_package(self, mock_onnx, temp_onnx_model):
        """Test full validation with onnx package."""
        # Mock ONNX model
        mock_model = MagicMock()
        mock_model.opset_import = [MagicMock(version=13)]
        mock_model.producer_name = "pytorch"
        mock_model.producer_version = "1.13.0"

        # Mock input/output
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_input.type.tensor_type.shape.dim = [
            MagicMock(dim_value=1),
            MagicMock(dim_value=3),
            MagicMock(dim_value=640),
            MagicMock(dim_value=640),
        ]
        mock_model.graph.input = [mock_input]

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.type.tensor_type.shape.dim = [
            MagicMock(dim_value=1),
            MagicMock(dim_value=25200),
            MagicMock(dim_value=24),  # 19 classes + 5 (YOLO format)
        ]
        mock_model.graph.output = [mock_output]

        mock_onnx.load.return_value = mock_model
        mock_onnx.checker.check_model.return_value = None

        metadata = YOLODetector.validate_onnx_model(temp_onnx_model)

        assert metadata["valid"] is True
        assert metadata["opset_version"] == 13
        assert metadata["producer"] == "pytorch"
        assert metadata["input_shape"] == (1, 3, 640, 640)
        assert metadata["num_classes"] == 19  # 24 - 5


class TestModelReloading:
    """Test model hot-swapping and reloading."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_reload_model_same_path(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test reloading model from same path."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Reload same model
        success = detector.reload_model()
        assert success is True
        assert detector.model_loaded is True

    @patch("vision.detection.yolo_detector.YOLO")
    def test_reload_model_new_path(self, mock_yolo_class, tmp_path, mock_yolo_model):
        """Test reloading model from new path."""
        # Create two model files
        model1 = tmp_path / "model1.pt"
        model1.write_bytes(b"model1_data" * 10000)

        model2 = tmp_path / "model2.pt"
        model2.write_bytes(b"model2_data" * 10000)

        mock_yolo_class.return_value = mock_yolo_model

        # Start with model1
        detector = YOLODetector(model_path=str(model1), auto_fallback=False)
        assert detector.model_path == str(model1)

        # Reload with model2
        success = detector.reload_model(str(model2))
        assert success is True
        assert detector.model_path == str(model2)
        assert detector.model_loaded is True

    @patch("vision.detection.yolo_detector.YOLO")
    def test_reload_model_failure_with_fallback(self, mock_yolo_class, temp_model_file):
        """Test model reload failure with auto_fallback."""
        # First load succeeds
        mock_yolo_class.return_value = MagicMock()
        detector = YOLODetector(model_path=temp_model_file, auto_fallback=True)

        # Second load fails
        mock_yolo_class.side_effect = RuntimeError("Load failed")

        success = detector.reload_model("/invalid/path/model.pt")
        assert success is False
        assert detector.stats["fallback_mode"] is True

    def test_reload_without_path_error(self):
        """Test reload fails when no path is specified."""
        detector = YOLODetector(model_path=None)

        success = detector.reload_model()
        assert success is False


class TestStatistics:
    """Test detection statistics tracking."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_statistics_tracking(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test that statistics are tracked correctly."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        # Run inference multiple times
        detector._run_inference(sample_frame)
        detector._run_inference(sample_frame)

        stats = detector.get_statistics()

        assert stats["total_inferences"] == 2
        assert stats["total_detections"] > 0
        assert stats["avg_inference_time"] >= 0
        assert stats["fallback_mode"] is False

    def test_get_model_info(self):
        """Test getting model information."""
        detector = YOLODetector(
            model_path=None,
            device="cpu",
            confidence=0.5,
            nms_threshold=0.4,
        )

        info = detector.get_model_info()

        assert info["model_path"] is None
        assert info["model_format"] is None
        assert info["model_loaded"] is False
        assert info["device"] == "cpu"
        assert info["confidence_threshold"] == 0.5
        assert info["nms_threshold"] == 0.4
        assert info["fallback_mode"] is True


class TestVisualization:
    """Test detection visualization."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_visualize_detections(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test visualization of detections."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        detections = detector._run_inference(sample_frame)

        vis_frame = detector.visualize_detections(
            sample_frame,
            detections,
            show_labels=True,
            show_confidence=True,
        )

        assert vis_frame is not None
        assert vis_frame.shape == sample_frame.shape
        assert not np.array_equal(vis_frame, sample_frame)  # Should be modified

    @patch("vision.detection.yolo_detector.YOLO")
    def test_visualize_without_labels(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test visualization without labels."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        detections = detector._run_inference(sample_frame)

        vis_frame = detector.visualize_detections(
            sample_frame,
            detections,
            show_labels=False,
            show_confidence=False,
        )

        assert vis_frame is not None
        assert vis_frame.shape == sample_frame.shape

    def test_visualize_empty_detections(self, sample_frame):
        """Test visualization with no detections."""
        detector = YOLODetector(model_path=None)

        vis_frame = detector.visualize_detections(sample_frame, [])

        # Should return copy of original frame
        assert vis_frame is not None
        assert vis_frame.shape == sample_frame.shape


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_create_detector_with_config(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test create_detector convenience function."""
        mock_yolo_class.return_value = mock_yolo_model

        config = {
            "device": "cuda",
            "confidence": 0.6,
            "nms_threshold": 0.5,
            "auto_fallback": True,
        }

        detector = create_detector(temp_model_file, config)

        assert detector.device == "cuda"
        assert detector.confidence == 0.6
        assert detector.nms_threshold == 0.5
        assert detector.auto_fallback is True

    def test_create_detector_without_config(self):
        """Test create_detector with default config."""
        detector = create_detector(None, None)

        assert detector.device == "cpu"
        assert detector.confidence == 0.4
        assert detector.nms_threshold == 0.45
        assert detector.auto_fallback is True


class TestThreadSafety:
    """Test thread safety of model operations."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_concurrent_inference(
        self, mock_yolo_class, temp_model_file, mock_yolo_model, sample_frame
    ):
        """Test thread-safe concurrent inference."""
        import threading

        mock_yolo_class.return_value = mock_yolo_model
        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)

        results = []
        errors = []

        def run_inference():
            try:
                detections = detector._run_inference(sample_frame)
                results.append(len(detections))
            except Exception as e:
                errors.append(e)

        # Run multiple inferences concurrently
        threads = [threading.Thread(target=run_inference) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 5
        assert all(r > 0 for r in results)

    @patch("vision.detection.yolo_detector.YOLO")
    def test_concurrent_reload(self, mock_yolo_class, tmp_path, mock_yolo_model):
        """Test thread-safe model reloading."""
        import threading
        import time

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"model_data" * 10000)

        mock_yolo_class.return_value = mock_yolo_model
        detector = YOLODetector(model_path=str(model_file), auto_fallback=False)

        reload_results = []

        def reload_model():
            success = detector.reload_model()
            reload_results.append(success)

        # Attempt concurrent reloads
        threads = [threading.Thread(target=reload_model) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle gracefully (some may fail due to lock contention)
        assert len(reload_results) == 3


class TestModelInferenceTest:
    """Test model inference testing functionality."""

    @patch("vision.detection.yolo_detector.YOLO")
    def test_model_inference_test_success(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test successful model inference test."""
        mock_yolo_class.return_value = mock_yolo_model

        result = YOLODetector.test_model_inference(temp_model_file, device="cpu")

        assert result["success"] is True
        assert result["inference_time"] is not None
        assert result["inference_time"] > 0
        assert result["num_detections"] >= 0
        assert result["model_loaded"] is True
        assert result["error"] is None

    def test_model_inference_test_failure(self):
        """Test failed model inference test."""
        result = YOLODetector.test_model_inference(
            "/nonexistent/model.pt",
            device="cpu",
        )

        assert result["success"] is False
        assert result["error"] is not None
        assert "failed" in result["error"].lower()


class TestDetectionDataclass:
    """Test Detection dataclass."""

    def test_detection_creation(self):
        """Test creating Detection object."""
        det = Detection(
            class_id=0,
            class_name="cue_ball",
            confidence=0.95,
            bbox=(100, 100, 150, 150),
            center=(125, 125),
            width=50,
            height=50,
            angle=45.0,
        )

        assert det.class_id == 0
        assert det.class_name == "cue_ball"
        assert det.confidence == 0.95
        assert det.bbox == (100, 100, 150, 150)
        assert det.center == (125, 125)
        assert det.width == 50
        assert det.height == 50
        assert det.angle == 45.0
        assert det.mask is None


class TestTableElements:
    """Test TableElements dataclass."""

    def test_table_elements_creation(self):
        """Test creating TableElements object."""
        elements = TableElements(
            table_bbox=(0, 0, 640, 480),
            table_confidence=0.95,
            pockets=[Detection(18, "pocket", 0.9, (10, 10, 30, 30), (20, 20), 20, 20)],
        )

        assert elements.table_bbox == (0, 0, 640, 480)
        assert elements.table_confidence == 0.95
        assert len(elements.pockets) == 1
        assert len(elements.rails) == 0

    def test_table_elements_post_init(self):
        """Test TableElements post_init initialization."""
        elements = TableElements()

        assert elements.table_bbox is None
        assert elements.table_confidence == 0.0
        assert elements.pockets == []
        assert elements.rails == []


class TestIsAvailable:
    """Test model availability checking."""

    def test_is_available_without_model(self):
        """Test is_available returns False without model."""
        detector = YOLODetector(model_path=None)
        assert detector.is_available() is False

    @patch("vision.detection.yolo_detector.YOLO")
    def test_is_available_with_model(
        self, mock_yolo_class, temp_model_file, mock_yolo_model
    ):
        """Test is_available returns True with loaded model."""
        mock_yolo_class.return_value = mock_yolo_model

        detector = YOLODetector(model_path=temp_model_file, auto_fallback=False)
        assert detector.is_available() is True


# Coverage target: 85%+
# This test suite covers:
# - Initialization (with/without model, various configs)
# - Model loading (.pt, .onnx, invalid formats)
# - Class mapping and conversion
# - Inference (basic, balls, cue, table elements)
# - Confidence and NMS threshold filtering
# - Error handling (missing files, invalid formats, inference errors)
# - Model validation (ONNX)
# - Model reloading/hot-swapping
# - Statistics tracking
# - Visualization
# - Thread safety
# - Convenience functions
# - All dataclasses
